#!/usr/bin/env python3
"""Generate a label-blind ARC-AGI-2 pass@2 Kaggle submission.

This runner deliberately cannot score. It rejects test outputs, loads a frozen
base or PEFT LoRA strategy router, executes the selected deterministic ARC
strategy, and writes exactly two grids per test input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import numpy as np

from evaluate_student import parse_strategy, render_prompt
from pipeline import (
    SYSTEM_PROMPT,
    arc_types,
    canonical_sha256,
    file_sha256,
    load_teacher,
    compact_prompt_for,
    strategy_tool,
    valid_predictions,
    write_json,
)


PASS_K = 2
CANDIDATE_POLICIES = ("router-only", "router-then-teacher")


@dataclass(frozen=True)
class PredictionResult:
    guesses: list[np.ndarray]
    selected_strategy: str | None
    raw_output_sha256: str
    error: str | None = None
    candidate_sources: tuple[str, ...] = ()


def enable_mlx_compatible_lora_add(model: Any) -> int:
    """Match MLX's cast-before-add LoRA arithmetic on PEFT Linear layers."""
    from peft.tuners.lora.layer import Linear as PeftLoraLinear

    def forward(module: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("adapter_names") is not None:
            raise ValueError("mixed-adapter batches are outside the frozen ARC contract")
        if module.disable_adapters:
            if module.merged:
                module.unmerge()
            return module.base_layer(value, *args, **kwargs)
        if module.merged:
            return module.base_layer(value, *args, **kwargs)
        result = module.base_layer(value, *args, **kwargs)
        result_dtype = result.dtype
        for active_adapter in module.active_adapters:
            if active_adapter not in module.lora_A:
                continue
            lora_a = module.lora_A[active_adapter]
            lora_b = module.lora_B[active_adapter]
            dropped = module.lora_dropout[active_adapter](value.to(lora_a.weight.dtype))
            delta = lora_b(lora_a(dropped)) * module.scaling[active_adapter]
            result = result + delta.to(result_dtype)
        return result

    patched = 0
    for module in model.modules():
        if isinstance(module, PeftLoraLinear):
            module.forward = MethodType(forward, module)
            patched += 1
    return patched


def validate_grid(value: Any, location: str) -> None:
    if not isinstance(value, list) or not value or len(value) > 30:
        raise ValueError(f"{location} must have between 1 and 30 rows")
    width = len(value[0]) if isinstance(value[0], list) else 0
    if width < 1 or width > 30:
        raise ValueError(f"{location} must have between 1 and 30 columns")
    for row in value:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"{location} must be rectangular")
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell <= 9:
                raise ValueError(f"{location} cells must be integer ARC colors 0..9")


def validate_label_blind_challenges(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise ValueError("challenge file must be a non-empty task object")
    for task_id, task in value.items():
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("challenge task identifiers must be non-empty strings")
        if not isinstance(task, dict) or not isinstance(task.get("train"), list) or not task["train"]:
            raise ValueError(f"task {task_id} must contain non-empty training pairs")
        if not isinstance(task.get("test"), list) or not task["test"]:
            raise ValueError(f"task {task_id} must contain non-empty test inputs")
        for index, pair in enumerate(task["train"]):
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError(f"task {task_id} train[{index}] must contain only input and output")
            validate_grid(pair["input"], f"task {task_id} train[{index}].input")
            validate_grid(pair["output"], f"task {task_id} train[{index}].output")
        for index, pair in enumerate(task["test"]):
            if not isinstance(pair, dict) or set(pair) != {"input"}:
                raise ValueError(
                    f"task {task_id} test[{index}] must contain only input; test outputs are forbidden"
                )
            validate_grid(pair["input"], f"task {task_id} test[{index}].input")
    return value


def make_label_blind_problem(task_id: str, task: dict[str, Any], test_index: int) -> Any:
    ArcData, ArcProblem, ArcSet = arc_types()
    training = [
        ArcSet(ArcData(np.asarray(pair["input"], dtype=int)), ArcData(np.asarray(pair["output"], dtype=int)))
        for pair in task["train"]
    ]
    test_input = np.asarray(task["test"][test_index]["input"], dtype=int)
    return ArcProblem(task_id, training, ArcSet(ArcData(test_input)))


def execute_candidate_policy(
    executor: Any,
    problem: Any,
    selected_strategy: str | None,
    candidate_policy: str,
) -> tuple[list[np.ndarray], tuple[str, ...], str | None]:
    """Execute a router candidate and an optional provenance-bound teacher fallback."""
    if candidate_policy not in CANDIDATE_POLICIES:
        raise ValueError(f"unsupported candidate policy: {candidate_policy}")

    guesses: list[np.ndarray] = []
    sources: list[str] = []
    errors: list[str] = []

    def append_first_unique(values: Any, source: str) -> None:
        for candidate in valid_predictions(executor, values):
            if any(np.array_equal(candidate, prior) for prior in guesses):
                continue
            guesses.append(candidate)
            sources.append(source)
            return

    training = problem.training_set()
    test_input = problem.test_set().get_input_data().data()
    if selected_strategy is not None:
        try:
            append_first_unique(
                getattr(executor, selected_strategy)(training, test_input),
                "router_selected_strategy",
            )
        except Exception as exc:
            errors.append(f"router:{type(exc).__name__}: {exc}"[:300])

    if candidate_policy == "router-then-teacher":
        try:
            append_first_unique(
                executor.make_predictions(problem),
                "deterministic_teacher_fallback",
            )
        except Exception as exc:
            errors.append(f"teacher:{type(exc).__name__}: {exc}"[:300])

    return guesses[:PASS_K], tuple(sources[:PASS_K]), "; ".join(errors) or None


def normalize_attempts(guesses: list[np.ndarray], test_input: list[list[int]]) -> tuple[list[list[list[int]]], str | None]:
    unique: list[np.ndarray] = []
    for guess in guesses:
        candidate = np.asarray(guess)
        if candidate.ndim != 2 or candidate.size == 0 or candidate.shape[0] > 30 or candidate.shape[1] > 30:
            continue
        if not np.issubdtype(candidate.dtype, np.integer) or not np.all((0 <= candidate) & (candidate <= 9)):
            continue
        candidate = candidate.astype(int)
        if not any(np.array_equal(candidate, prior) for prior in unique):
            unique.append(candidate)
        if len(unique) == PASS_K:
            break

    fallback = None
    if not unique:
        unique.append(np.zeros_like(np.asarray(test_input, dtype=int)))
        fallback = "zero_grid_no_valid_prediction"
    if len(unique) == 1:
        unique.append(unique[0].copy())
        fallback = fallback or "duplicate_only_valid_prediction"
    return [value.tolist() for value in unique[:PASS_K]], fallback


def build_submission(
    challenges: dict[str, dict[str, Any]],
    predict: Callable[[str, dict[str, Any], int], PredictionResult],
    *,
    max_seconds: float = 0.0,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, list[dict[str, list[list[int]]]]], list[dict[str, Any]]]:
    if max_seconds < 0:
        raise ValueError("inference time bound must be nonnegative")
    submission: dict[str, list[dict[str, list[list[int]]]]] = {}
    audit: list[dict[str, Any]] = []
    total = sum(len(task["test"]) for task in challenges.values())
    started = time.monotonic()
    deadline = started + max_seconds if max_seconds else None
    for task_id, task in challenges.items():
        task_attempts = []
        for test_index, test_pair in enumerate(task["test"]):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"inference time bound exhausted after {len(audit)}/{total} test inputs"
                )
            result = predict(task_id, task, test_index)
            attempts, fallback = normalize_attempts(result.guesses, test_pair["input"])
            task_attempts.append({"attempt_1": attempts[0], "attempt_2": attempts[1]})
            audit.append(
                {
                    "task_ref_sha256": hashlib.sha256(task_id.encode("utf-8")).hexdigest(),
                    "test_index": test_index,
                    "selected_strategy": result.selected_strategy,
                    "raw_output_sha256": result.raw_output_sha256,
                    "attempt_1_sha256": canonical_sha256(attempts[0]),
                    "attempt_2_sha256": canonical_sha256(attempts[1]),
                    "fallback": fallback,
                    "error": result.error,
                    "candidate_sources": list(result.candidate_sources),
                }
            )
            if progress is not None:
                progress(
                    {
                        "completed": len(audit),
                        "total": total,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    }
                )
            if deadline is not None and time.monotonic() >= deadline and len(audit) < total:
                raise TimeoutError(
                    f"inference time bound exhausted after {len(audit)}/{total} test inputs"
                )
        submission[task_id] = task_attempts
    return submission, audit


class HfStrategyRouter:
    def __init__(
        self,
        *,
        arc_root: Path,
        model_path: Path,
        adapter_path: Path | None,
        device: str,
        dtype: str,
        peft_addition: str,
        max_new_tokens: int,
        max_generation_seconds: float,
        candidate_policy: str = "router-only",
    ) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.dtype_name = dtype
        self.max_new_tokens = max_new_tokens
        self.max_generation_seconds = max_generation_seconds
        if candidate_policy not in CANDIDATE_POLICIES:
            raise ValueError(f"unsupported candidate policy: {candidate_policy}")
        self.candidate_policy = candidate_policy
        self.executor, self.strategy_names, self.teacher = load_teacher(arc_root)
        self.allowed = set(self.strategy_names)
        self.tools = [strategy_tool(self.strategy_names, self.teacher["sha256"])]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        torch_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch_dtype)
        except TypeError:  # Transformers < 5 uses torch_dtype.
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch_dtype)
        if adapter_path is not None:
            from peft import (
                LoraConfig,
                get_peft_model,
                get_peft_model_state_dict,
                set_peft_model_state_dict,
            )
            from safetensors.torch import load_file

            adapter_config = LoraConfig.from_pretrained(adapter_path, local_files_only=True)
            model = get_peft_model(model, adapter_config)
            source_state = load_file(adapter_path / "adapter_model.safetensors", device="cpu")
            load_result = set_peft_model_state_dict(model, source_state, adapter_name="default")
            if load_result.unexpected_keys:
                raise ValueError(
                    f"frozen adapter has unexpected PEFT keys: {load_result.unexpected_keys}"
                )
            loaded_state = get_peft_model_state_dict(model, adapter_name="default")
            if set(loaded_state) != set(source_state) or any(
                not torch.equal(loaded_state[key].cpu(), source_state[key]) for key in source_state
            ):
                raise ValueError("frozen adapter state did not load with exact tensor parity")
            self.adapter_load_method = "topology_then_exact_state_dict"
            self.adapter_resident = True
        else:
            self.adapter_load_method = "none"
            self.adapter_resident = False
        self.patched_lora_layers = (
            enable_mlx_compatible_lora_add(model)
            if adapter_path is not None and peft_addition == "mlx-compatible"
            else 0
        )
        self.model = model.to(self.device).eval()
        self.adapter_enabled = self.adapter_resident

    def set_adapter_enabled(self, enabled: bool) -> None:
        """Select the controlled PEFT arm without reloading model weights."""
        if enabled and not self.adapter_resident:
            raise ValueError("cannot enable an adapter that is not resident")
        self.adapter_enabled = enabled

    @classmethod
    def from_preloaded(
        cls,
        *,
        arc_root: Path,
        model: Any,
        tokenizer: Any,
        torch: Any,
        device: str,
        dtype: str,
        max_new_tokens: int,
        max_generation_seconds: float,
        adapter_name: str,
        candidate_policy: str = "router-only",
    ) -> "HfStrategyRouter":
        """Bind a trained in-memory PEFT model without reloading its base weights."""
        instance = cls.__new__(cls)
        instance.torch = torch
        instance.device = torch.device(device)
        instance.dtype_name = dtype
        instance.max_new_tokens = max_new_tokens
        instance.max_generation_seconds = max_generation_seconds
        if candidate_policy not in CANDIDATE_POLICIES:
            raise ValueError(f"unsupported candidate policy: {candidate_policy}")
        instance.candidate_policy = candidate_policy
        instance.executor, instance.strategy_names, instance.teacher = load_teacher(arc_root)
        instance.allowed = set(instance.strategy_names)
        instance.tools = [strategy_tool(instance.strategy_names, instance.teacher["sha256"])]
        instance.tokenizer = tokenizer
        model.set_adapter(adapter_name)
        model.config.use_cache = True
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model, "disable_input_require_grads"):
            model.disable_input_require_grads()
        instance.model = model.to(instance.device).eval()
        instance.patched_lora_layers = 0
        instance.adapter_load_method = "preloaded_training_session"
        instance.adapter_resident = True
        instance.adapter_enabled = True
        return instance

    def __call__(self, task_id: str, task: dict[str, Any], test_index: int) -> PredictionResult:
        task_ref = hashlib.sha256(task_id.encode("utf-8")).hexdigest()

        def emit_stage(stage: str) -> None:
            print(
                json.dumps(
                    {
                        "prediction_stage": {
                            "stage": stage,
                            "task_ref_sha256": task_ref,
                            "test_index": test_index,
                        }
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        selected_task = {"train": task["train"], "test": [task["test"][test_index]]}
        row = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": compact_prompt_for(selected_task)},
            ],
            "tools": self.tools,
        }
        rendered = render_prompt(self.tokenizer, row)
        encoded = self.tokenizer(rendered, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        emit_stage("model_generation_started")
        adapter_context = (
            nullcontext()
            if self.adapter_enabled or not self.adapter_resident
            else self.model.disable_adapter()
        )
        with self.torch.inference_mode(), adapter_context:
            generation = {
                "do_sample": False,
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            if self.max_generation_seconds:
                generation["max_time"] = self.max_generation_seconds
            generated = self.model.generate(
                **encoded,
                **generation,
            )
        emit_stage("model_generation_completed")
        new_tokens = generated[0, encoded["input_ids"].shape[1] :]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        selected = parse_strategy(raw_output, self.allowed)
        problem = make_label_blind_problem(task_id, task, test_index)
        guesses, candidate_sources, error = execute_candidate_policy(
            self.executor,
            problem,
            selected,
            self.candidate_policy,
        )
        emit_stage("deterministic_execution_completed")
        return PredictionResult(
            guesses=guesses,
            selected_strategy=selected,
            raw_output_sha256=hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
            error=error,
            candidate_sources=candidate_sources,
        )


class MlxStrategyRouter:
    """Reference-only MLX router for local CPU parity and public evaluation."""

    def __init__(
        self,
        *,
        arc_root: Path,
        model_path: Path,
        adapter_path: Path | None,
        device: str,
        max_new_tokens: int,
        candidate_policy: str = "router-only",
    ) -> None:
        if device != "cpu":
            raise ValueError("the governed MLX reference path is CPU-only")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import mlx.core as mx
        from mlx_lm import load

        mx.set_default_device(mx.cpu)
        generate_module = importlib.import_module("mlx_lm.generate")
        generate_module.wired_limit = lambda *_args, **_kwargs: nullcontext()
        self.generate = generate_module.generate
        self.max_new_tokens = max_new_tokens
        if candidate_policy not in CANDIDATE_POLICIES:
            raise ValueError(f"unsupported candidate policy: {candidate_policy}")
        self.candidate_policy = candidate_policy
        self.executor, self.strategy_names, self.teacher = load_teacher(arc_root)
        self.allowed = set(self.strategy_names)
        self.tools = [strategy_tool(self.strategy_names, self.teacher["sha256"])]
        self.model, self.tokenizer = load(
            str(model_path),
            adapter_path=str(adapter_path) if adapter_path is not None else None,
        )
        self.patched_lora_layers = 0

    def __call__(self, task_id: str, task: dict[str, Any], test_index: int) -> PredictionResult:
        selected_task = {"train": task["train"], "test": [task["test"][test_index]]}
        row = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": compact_prompt_for(selected_task)},
            ],
            "tools": self.tools,
        }
        raw_output = self.generate(
            self.model,
            self.tokenizer,
            prompt=render_prompt(self.tokenizer, row),
            max_tokens=self.max_new_tokens,
            verbose=False,
        )
        selected = parse_strategy(raw_output, self.allowed)
        problem = make_label_blind_problem(task_id, task, test_index)
        guesses, candidate_sources, error = execute_candidate_policy(
            self.executor,
            problem,
            selected,
            self.candidate_policy,
        )
        return PredictionResult(
            guesses=guesses,
            selected_strategy=selected,
            raw_output_sha256=hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
            error=error,
            candidate_sources=candidate_sources,
        )


def write_governed_submission(
    args: argparse.Namespace,
    *,
    router: Any | None = None,
    progress_scope: str | None = None,
) -> int:
    if (args.arm == "lora") != (args.adapter is not None):
        raise ValueError("the lora arm requires --adapter and the base arm forbids it")
    if args.out.resolve() == args.receipt.resolve():
        raise ValueError("submission and receipt paths must be distinct")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    if args.max_generation_seconds <= 0 or args.max_inference_seconds <= 0:
        raise ValueError("generation and inference time bounds must be positive")
    if args.out.exists() or args.receipt.exists():
        raise ValueError("refusing to overwrite an existing submission or receipt")
    challenges = validate_label_blind_challenges(json.loads(args.challenges.read_text(encoding="utf-8")))
    preloaded_model = router is not None
    if router is None:
        router_type = HfStrategyRouter if args.backend == "hf" else MlxStrategyRouter
        router_kwargs = {
            "arc_root": args.arc_root.resolve(),
            "model_path": args.model.resolve(),
            "adapter_path": args.adapter.resolve() if args.adapter else None,
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
            "candidate_policy": args.candidate_policy,
        }
        if args.backend == "hf":
            router_kwargs.update(
                {
                    "dtype": args.dtype,
                    "peft_addition": args.peft_addition,
                    "max_generation_seconds": args.max_generation_seconds,
                }
            )
        router = router_type(**router_kwargs)
    if getattr(router, "candidate_policy", None) != args.candidate_policy:
        raise ValueError("preloaded router candidate policy does not match the submission contract")
    if hasattr(router, "adapter_enabled") and router.adapter_enabled != (args.adapter is not None):
        raise ValueError("preloaded router adapter state does not match the selected arm")
    scope = progress_scope or args.out.stem

    def emit_progress(value: dict[str, Any]) -> None:
        print(json.dumps({"inference": {"scope": scope, **value}}, sort_keys=True), flush=True)

    submission, audit = build_submission(
        challenges,
        router,
        max_seconds=args.max_inference_seconds,
        progress=emit_progress,
    )
    write_json(args.out, submission)
    args.out.chmod(0o600)
    receipt = {
        "format_version": "hfr.arcagi.kaggle_submission.v1",
        "benchmark": "ARC Prize 2026 ARC-AGI-2",
        "arm": args.arm,
        "label_blind": True,
        "pass_k": PASS_K,
        "network_disabled": True,
        "inference": {
            "backend": args.backend,
            "device": args.device,
            "dtype": args.dtype if args.backend == "hf" else "model-native-bfloat16",
            "max_new_tokens": args.max_new_tokens,
            "max_generation_seconds": args.max_generation_seconds,
            "max_inference_seconds": args.max_inference_seconds,
            "peft_addition": args.peft_addition if args.backend == "hf" else "not-applicable",
            "patched_lora_layers": router.patched_lora_layers,
            "preloaded_model": preloaded_model,
            "candidate_policy": args.candidate_policy,
            "adapter_load_method": router.adapter_load_method,
        },
        "model": {"id": args.model_id, "revision": args.model_revision},
        "adapter": {
            "enabled": args.adapter is not None,
            "sha256": file_sha256(
                args.adapter / ("adapter_model.safetensors" if args.backend == "hf" else "adapters.safetensors")
            )
            if args.adapter
            else None,
            "config_sha256": file_sha256(args.adapter / "adapter_config.json") if args.adapter else None,
        },
        "teacher": router.teacher,
        "challenge_sha256": file_sha256(args.challenges),
        "submission_sha256": file_sha256(args.out),
        "task_count": len(challenges),
        "test_input_count": len(audit),
        "invalid_strategy_count": sum(row["selected_strategy"] is None for row in audit),
        "fallback_count": sum(row["fallback"] is not None for row in audit),
        "error_count": sum(row["error"] is not None for row in audit),
        "selected_strategy_counts": dict(
            sorted(Counter(str(row["selected_strategy"]) for row in audit).items())
        ),
        "outcomes": audit,
        "claim_boundary": (
            "This receipt records a hybrid LoRA/base strategy router plus deterministic executor "
            f"under the {args.candidate_policy} candidate policy. "
            "Only an external Kaggle score can establish performance on hidden tasks."
        ),
    }
    write_json(args.receipt, receipt)
    args.receipt.chmod(0o600)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "task_count": receipt["task_count"],
                "test_input_count": receipt["test_input_count"],
                "invalid_strategy_count": receipt["invalid_strategy_count"],
                "fallback_count": receipt["fallback_count"],
                "error_count": receipt["error_count"],
                "submission_sha256": receipt["submission_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--challenges", type=Path, required=True)
    result.add_argument("--arc-root", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--adapter", type=Path)
    result.add_argument("--arm", choices=("base", "lora"), required=True)
    result.add_argument("--backend", choices=("hf", "mlx"), default="hf")
    result.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    result.add_argument("--model-revision", required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    result.add_argument(
        "--peft-addition",
        choices=("standard", "mlx-compatible"),
        default="mlx-compatible",
    )
    result.add_argument("--max-new-tokens", type=int, default=96)
    result.add_argument("--max-generation-seconds", type=float, default=120.0)
    result.add_argument("--max-inference-seconds", type=float, default=3600.0)
    result.add_argument("--candidate-policy", choices=CANDIDATE_POLICIES, default="router-only")
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--receipt", type=Path, required=True)
    return result


if __name__ == "__main__":
    raise SystemExit(write_governed_submission(parser().parse_args()))
