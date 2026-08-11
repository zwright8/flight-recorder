#!/usr/bin/env python3
"""Train or calibrate a Linux-native PEFT ARC strategy-router LoRA."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline import compact_prompt_from_native, file_sha256, require_fresh_output, secure_tree, write_json


TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TORCHAO_MINIMUM_PEFT_VERSION = "0.16.0"


@dataclass
class TrainingSession:
    """Successful in-memory training state for governed same-process inference."""

    model: Any
    tokenizer: Any
    torch: Any
    device: Any
    phase_one_receipt: dict[str, Any]
    continuation_receipt: dict[str, Any] | None


def disable_incompatible_torchao_dispatcher(
    peft_import_utils: Any,
    peft_torchao: Any,
    installed_version: str,
) -> bool:
    from packaging.version import Version

    if Version(installed_version) >= Version(TORCHAO_MINIMUM_PEFT_VERSION):
        return False
    cached_probe = peft_import_utils.is_torchao_available
    if hasattr(cached_probe, "cache_clear"):
        cached_probe.cache_clear()

    def unavailable() -> bool:
        return False

    # PEFT imports the optional TorchAO probe into its dispatcher module, so
    # patch both references.  This route uses ordinary fp16 Linear weights and
    # never invokes TorchAO quantization.
    peft_import_utils.is_torchao_available = unavailable
    peft_torchao.is_torchao_available = unavailable
    return True


def configure_optional_torchao_compatibility() -> dict[str, Any]:
    try:
        installed_version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return {"installed_version": None, "dispatcher_disabled": False}

    from peft import import_utils as peft_import_utils
    from peft.tuners.lora import torchao as peft_torchao

    disabled = disable_incompatible_torchao_dispatcher(
        peft_import_utils,
        peft_torchao,
        installed_version,
    )
    return {
        "installed_version": installed_version,
        "dispatcher_disabled": disabled,
        "reason": "unused_optional_package_below_peft_minimum" if disabled else None,
    }


def training_row_for_prompt_encoding(row: dict[str, Any], prompt_encoding: str) -> dict[str, Any]:
    if prompt_encoding == "native-json-v1":
        return row
    if prompt_encoding != "compact-grid-v1":
        raise ValueError(f"unsupported ARC prompt encoding: {prompt_encoding}")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3 or messages[-2].get("role") != "user":
        raise ValueError("compact ARC encoding requires a penultimate user message")
    encoded = dict(row)
    encoded["messages"] = [dict(message) for message in messages]
    encoded["messages"][-2]["content"] = compact_prompt_from_native(messages[-2].get("content"))
    return encoded


def render_training_row(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    prompt_encoding: str = "native-json-v1",
) -> tuple[list[int], list[int]]:
    row = training_row_for_prompt_encoding(row, prompt_encoding)
    messages = row.get("messages")
    tools = row.get("tools")
    if not isinstance(messages, list) or len(messages) < 2 or not isinstance(tools, list):
        raise ValueError("training rows require native messages and tools")
    if messages[-1].get("role") != "assistant" or not messages[-1].get("tool_calls"):
        raise ValueError("training rows must end in an assistant tool call")
    kwargs = {"tools": tools, "tokenize": False, "enable_thinking": False}
    try:
        prefix = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **kwargs)
        full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        prefix = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **kwargs)
        full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    input_ids = tokenizer.encode(full, add_special_tokens=False)
    if input_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("assistant completion is not a strict suffix of the rendered prompt")
    labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids) :]
    if not any(value != -100 for value in labels):
        raise ValueError("training row has no supervised assistant tokens")
    return input_ids, labels


def completion_loss_window(labels: list[int], ignore_index: int = -100) -> tuple[int, list[int]]:
    """Return the minimal causal-logit window for suffix-only supervision.

    The native ARC rows supervise only the final assistant tool call. Keeping
    the full prompt in the transformer context while projecting every prompt
    token through the 151k-token vocabulary wastes several gigabytes on long
    rows. The final prompt token predicts the first supervised token, so the
    required logit window is the supervised suffix plus one position.
    """
    try:
        first_supervised = next(index for index, value in enumerate(labels) if value != ignore_index)
    except StopIteration as exc:
        raise ValueError("training row has no supervised assistant tokens") from exc
    if any(value == ignore_index for value in labels[first_supervised:]):
        raise ValueError("training supervision must be a contiguous suffix")
    supervised_tokens = len(labels) - first_supervised
    logits_to_keep = supervised_tokens + 1
    shifted_labels = labels[1:] + [ignore_index]
    return logits_to_keep, shifted_labels[-logits_to_keep:]


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"training row {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError("training data is empty")
    return rows


def dtype_for(torch: Any, name: str) -> Any:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def training_result_status(completed_steps: int, target_steps: int) -> str:
    if completed_steps < 0 or target_steps < 1 or completed_steps > target_steps:
        raise ValueError("invalid training completion counts")
    return "succeeded" if completed_steps == target_steps else "time_bound_exhausted"


def training_phase_plan(
    base_plan: dict[str, Any],
    *,
    max_steps: int,
    max_training_seconds: float,
    initial_adapter_sha256: str | None,
    reuse_loaded_model: bool,
) -> dict[str, Any]:
    plan = copy.deepcopy(base_plan)
    plan["initial_adapter"] = {
        "enabled": initial_adapter_sha256 is not None,
        "weights_sha256": initial_adapter_sha256,
    }
    plan["configuration"].update(
        {
            "max_steps": max_steps,
            "max_training_seconds": max_training_seconds,
            "reuse_loaded_model": reuse_loaded_model,
            "optimizer_reset": True,
            "sampler_reset": True,
        }
    )
    return plan


def train_with_session(args: argparse.Namespace) -> tuple[int, TrainingSession | None]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torchao_compatibility = configure_optional_torchao_compatibility()
    continuation_out = args.continuation_out.resolve() if args.continuation_out else None

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if (
        args.max_steps < 1
        or args.gradient_accumulation_steps < 1
        or args.max_training_seconds < 0
        or args.continuation_steps < 0
        or args.continuation_training_seconds < 0
    ):
        raise ValueError(
            "training steps and gradient accumulation must be positive and the time bound nonnegative"
        )
    if (continuation_out is None) != (args.continuation_steps == 0):
        raise ValueError("continuation output and positive continuation steps must be provided together")
    if continuation_out is None and args.continuation_training_seconds != 0:
        raise ValueError("continuation time bound requires a continuation output")
    if continuation_out is not None and args.initial_adapter is not None:
        raise ValueError("single-process continuation cannot start from an external adapter")
    if continuation_out == args.out.resolve():
        raise ValueError("continuation output must differ from the phase-one output")
    if args.max_seq_length < 1 or not 0 < args.min_retained_fraction <= 1:
        raise ValueError("invalid sequence-length retention contract")
    model_path = args.model.resolve()
    if not model_path.is_dir():
        raise ValueError("offline base model directory does not exist")
    initial_adapter = args.initial_adapter.resolve() if args.initial_adapter else None
    if initial_adapter is not None and not initial_adapter.is_dir():
        raise ValueError("initial PEFT adapter directory does not exist")
    if initial_adapter is not None:
        initial_config = json.loads((initial_adapter / "adapter_config.json").read_text(encoding="utf-8"))
        expected_layers = list(range(args.first_layer, args.first_layer + args.num_layers))
        if (
            initial_config.get("r") != args.rank
            or initial_config.get("lora_alpha") != args.lora_alpha
            or float(initial_config.get("lora_dropout", -1)) != args.lora_dropout
            or initial_config.get("layers_to_transform") != expected_layers
        ):
            raise ValueError("initial adapter does not match the frozen LoRA training contract")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    source_rows = load_rows(args.data)
    prepared = []
    excluded = 0
    for row in source_rows:
        input_ids, labels = render_training_row(
            tokenizer,
            row,
            prompt_encoding=args.prompt_encoding,
        )
        if len(input_ids) > args.max_seq_length:
            excluded += 1
            continue
        prepared.append((input_ids, labels))
    retained_fraction = len(prepared) / len(source_rows)
    if retained_fraction < args.min_retained_fraction:
        raise ValueError(
            f"sequence-length gate retained {retained_fraction:.4f}, below {args.min_retained_fraction:.4f}"
        )
    if args.limit:
        prepared = prepared[: args.limit]
    if not prepared:
        raise ValueError("no training rows remain after filtering")

    base_plan = {
        "format_version": "hfr.arcagi.peft_strategy_training_plan.v1",
        "offline_enforced": True,
        "model": {"id": args.model_id, "revision": args.model_revision},
        "data": {
            "sha256": file_sha256(args.data),
            "source_rows": len(source_rows),
            "retained_rows": len(prepared),
            "sequence_length_excluded_rows": excluded,
            "retained_fraction": retained_fraction,
        },
        "configuration": {
            "device": args.device,
            "dtype": args.dtype,
            "trainable_parameter_dtype": "float32",
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_seq_length": args.max_seq_length,
            "prompt_encoding": args.prompt_encoding,
            "rank": args.rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_layers": list(range(args.first_layer, args.first_layer + args.num_layers)),
            "target_modules": list(TARGET_MODULES),
            "seed": args.seed,
        },
        "optional_package_compatibility": {"torchao": torchao_compatibility},
    }
    phase_one_initial_sha256 = (
        file_sha256(initial_adapter / "adapter_model.safetensors") if initial_adapter else None
    )
    phase_one_plan = training_phase_plan(
        base_plan,
        max_steps=args.max_steps,
        max_training_seconds=args.max_training_seconds,
        initial_adapter_sha256=phase_one_initial_sha256,
        reuse_loaded_model=False,
    )
    if args.dry_run:
        dry_run_plan = {"phase_one": phase_one_plan, "continuation": None}
        if continuation_out is not None:
            dry_run_plan["continuation"] = training_phase_plan(
                base_plan,
                max_steps=args.continuation_steps,
                max_training_seconds=args.continuation_training_seconds,
                initial_adapter_sha256="pending-phase-one-output",
                reuse_loaded_model=True,
            )
        print(json.dumps(dry_run_plan, sort_keys=True))
        return 0, None

    require_fresh_output(args.out)
    if continuation_out is not None:
        require_fresh_output(continuation_out)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype_for(torch, args.dtype),
        )
    except TypeError:  # Transformers < 5 uses torch_dtype.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype_for(torch, args.dtype),
        )
    model.config.use_cache = False
    if initial_adapter is not None:
        model = PeftModel.from_pretrained(model, initial_adapter, is_trainable=True, local_files_only=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=args.rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=list(TARGET_MODULES),
                layers_to_transform=list(range(args.first_layer, args.first_layer + args.num_layers)),
                layers_pattern="layers",
                bias="none",
            ),
        )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.to(device)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(device=device, dtype=torch.float32)
    model.train()
    scaler_enabled = args.device == "cuda" and args.dtype == "float16"
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    def run_phase(
        *,
        phase_plan: dict[str, Any],
        out: Path,
        max_steps: int,
        max_training_seconds: float,
    ) -> tuple[str, dict[str, Any]]:
        # Match the previous separate-process continuation contract without
        # reloading the model: reset all stochastic, optimizer, scaler, and
        # sampler state at each phase boundary.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except TypeError:  # PyTorch < 2.4 did not accept the device argument.
            scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        randomizer = random.Random(args.seed)
        order = list(range(len(prepared)))
        randomizer.shuffle(order)
        cursor = 0
        optimizer.zero_grad(set_to_none=True)
        losses = []
        started = time.monotonic()
        deadline = started + max_training_seconds if max_training_seconds else None
        completed_steps = 0
        while completed_steps < max_steps:
            if deadline is not None and time.monotonic() >= deadline:
                break
            accumulated_loss = 0.0
            for _ in range(args.gradient_accumulation_steps):
                if cursor == len(order):
                    randomizer.shuffle(order)
                    cursor = 0
                input_ids, labels = prepared[order[cursor]]
                cursor += 1
                batch_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
                logits_to_keep, shifted_labels = completion_loss_window(labels)
                batch_shifted_labels = torch.tensor(
                    [shifted_labels], dtype=torch.long, device=device
                )
                with torch.autocast(
                    device_type=args.device,
                    dtype=dtype_for(torch, args.dtype),
                    enabled=args.device == "cuda" and args.dtype != "float32",
                ):
                    logits = model(
                        input_ids=batch_ids,
                        attention_mask=torch.ones_like(batch_ids),
                        logits_to_keep=logits_to_keep,
                    ).logits
                    loss = torch.nn.functional.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]),
                        batch_shifted_labels.reshape(-1),
                        ignore_index=-100,
                    )
                    scaled_loss = loss / args.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                accumulated_loss += float(loss.detach().cpu())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            completed_steps += 1
            losses.append(accumulated_loss / args.gradient_accumulation_steps)
            if completed_steps == 1 or completed_steps % args.report_every == 0:
                print(
                    json.dumps(
                        {
                            "phase": out.name,
                            "step": completed_steps,
                            "loss": losses[-1],
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if deadline is not None and time.monotonic() >= deadline:
                break

        model.save_pretrained(out, safe_serialization=True)
        status = training_result_status(completed_steps, max_steps)
        loss_receipt = {
            "first": losses[0] if losses else None,
            "last": losses[-1] if losses else None,
            "minimum": min(losses) if losses else None,
        }
        receipt = {
            **phase_plan,
            "format_version": "hfr.arcagi.peft_strategy_training_result.v1",
            "status": status,
            "target_steps": max_steps,
            "completed_steps": completed_steps,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "loss": loss_receipt,
            "outputs": {
                "adapter_config_sha256": file_sha256(out / "adapter_config.json"),
                "adapter_model_sha256": file_sha256(out / "adapter_model.safetensors"),
            },
        }
        write_json(out / "training_receipt.json", receipt)
        secure_tree(out)
        print(
            json.dumps(
                {
                    "phase": out.name,
                    "status": status,
                    "target_steps": max_steps,
                    "completed_steps": completed_steps,
                    "outputs": receipt["outputs"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return status, receipt

    phase_one_status, phase_one_receipt = run_phase(
        phase_plan=phase_one_plan,
        out=args.out,
        max_steps=args.max_steps,
        max_training_seconds=args.max_training_seconds,
    )
    if phase_one_status != "succeeded":
        return 2, None
    if continuation_out is None:
        return 0, TrainingSession(
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            device=device,
            phase_one_receipt=phase_one_receipt,
            continuation_receipt=None,
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    continuation_plan = training_phase_plan(
        base_plan,
        max_steps=args.continuation_steps,
        max_training_seconds=args.continuation_training_seconds,
        initial_adapter_sha256=phase_one_receipt["outputs"]["adapter_model_sha256"],
        reuse_loaded_model=True,
    )
    continuation_status, continuation_receipt = run_phase(
        phase_plan=continuation_plan,
        out=continuation_out,
        max_steps=args.continuation_steps,
        max_training_seconds=args.continuation_training_seconds,
    )
    if continuation_status != "succeeded":
        return 2, None
    return 0, TrainingSession(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        device=device,
        phase_one_receipt=phase_one_receipt,
        continuation_receipt=continuation_receipt,
    )


def train(args: argparse.Namespace) -> int:
    status, _ = train_with_session(args)
    return status


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--initial-adapter", type=Path)
    result.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    result.add_argument("--model-revision", required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float16")
    result.add_argument("--max-steps", type=int, default=300)
    result.add_argument("--learning-rate", type=float, default=1e-5)
    result.add_argument("--gradient-accumulation-steps", type=int, default=1)
    result.add_argument("--max-seq-length", type=int, default=8192)
    result.add_argument("--min-retained-fraction", type=float, default=1.0)
    result.add_argument(
        "--prompt-encoding",
        choices=("native-json-v1", "compact-grid-v1"),
        default="native-json-v1",
    )
    result.add_argument("--rank", type=int, default=8)
    result.add_argument("--lora-alpha", type=int, default=160)
    result.add_argument("--lora-dropout", type=float, default=0.0)
    result.add_argument("--first-layer", type=int, default=20)
    result.add_argument("--num-layers", type=int, default=8)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--limit", type=int, default=0)
    result.add_argument("--report-every", type=int, default=10)
    result.add_argument("--max-training-seconds", type=float, default=0.0)
    result.add_argument("--continuation-out", type=Path)
    result.add_argument("--continuation-steps", type=int, default=0)
    result.add_argument("--continuation-training-seconds", type=float, default=0.0)
    result.add_argument("--dry-run", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(train(parser().parse_args()))
