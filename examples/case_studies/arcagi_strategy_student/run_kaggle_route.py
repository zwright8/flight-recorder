#!/usr/bin/env python3
"""Run the private, fail-closed ARC strategy-student Kaggle route."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MINIMUM_CUDA_CAPABILITY = (7, 0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_bundle(bundle_root: Path) -> dict[str, Any]:
    manifest_path = bundle_root / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != "hfr.arcagi.kaggle_private_bundle.v1":
        raise ValueError("unsupported or missing private-bundle manifest")
    if manifest.get("publication_allowed") is not False:
        raise ValueError("private bundle must explicitly forbid publication")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("private bundle manifest has no file inventory")
    for relative, expected in files.items():
        path = bundle_root / relative
        if not path.is_file() or file_sha256(path) != expected.get("sha256"):
            raise ValueError(f"private bundle integrity failure: {relative}")
    return manifest


def find_competition_challenges(input_root: Path, bundle_root: Path) -> Path:
    candidates = [
        path
        for path in input_root.rglob("arc-agi_test_challenges.json")
        if bundle_root not in path.parents
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one competition challenge file, found {len(candidates)}")
    return candidates[0]


def visible_gate(
    *,
    aggregate: dict[str, Any],
    router_receipts: list[dict[str, Any]],
    expected_total: int,
    minimum_passed: int,
) -> tuple[bool, list[str]]:
    failures = []
    if aggregate.get("total") != expected_total:
        failures.append("visible_total_mismatch")
    if aggregate.get("passed", -1) < minimum_passed:
        failures.append("visible_accuracy_below_threshold")
    for index, receipt in enumerate(router_receipts, start=1):
        if receipt.get("invalid_strategy_count") != 0:
            failures.append(f"router_{index}_invalid_strategy")
        if receipt.get("error_count") != 0:
            failures.append(f"router_{index}_execution_error")
    return not failures, failures


def checked(command: list[str], env: dict[str, str]) -> None:
    print(json.dumps({"command": command}, sort_keys=True), flush=True)
    completed = subprocess.run(command, check=False, env=env)
    print(
        json.dumps(
            {"command_result": {"returncode": completed.returncode}},
            sort_keys=True,
        ),
        flush=True,
    )
    completed.check_returncode()


def require_supported_cuda_capability(capability: tuple[int, int]) -> tuple[int, int]:
    if capability < MINIMUM_CUDA_CAPABILITY:
        raise ValueError(
            f"CUDA capability {capability[0]}.{capability[1]} is below the "
            f"required {MINIMUM_CUDA_CAPABILITY[0]}.{MINIMUM_CUDA_CAPABILITY[1]}"
        )
    return capability


def environment_receipt() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise ValueError("the Kaggle route requires CUDA and will not fall back to CPU")
    cuda_capability = require_supported_cuda_capability(torch.cuda.get_device_capability(0))
    packages = {}
    for name in ("accelerate", "numpy", "peft", "safetensors", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"required offline Kaggle package is missing: {name}") from exc
    return {
        "packages": packages,
        "python": platform.python_version(),
        "cuda_available": True,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_capability": list(cuda_capability),
    }


def run(args: argparse.Namespace) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    bundle_root = args.bundle_root.resolve()
    manifest = verify_bundle(bundle_root)
    runtime_environment = environment_receipt()
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise ValueError("refusing to overwrite non-empty Kaggle work directory")
    if args.submission_out.exists():
        raise ValueError("refusing to overwrite an existing Kaggle submission")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.chmod(0o700)

    case_dir = (
        args.runtime_case_dir.resolve()
        if args.runtime_case_dir is not None
        else bundle_root / "repo/examples/case_studies/arcagi_strategy_student"
    )
    required_runtime_sources = (
        "evaluate_student.py",
        "kaggle_submission.py",
        "merge_kaggle_submissions.py",
        "pipeline.py",
        "run_kaggle_route.py",
        "score_kaggle_submission.py",
        "train_peft_strategy.py",
    )
    missing_runtime_sources = [
        filename for filename in required_runtime_sources if not (case_dir / filename).is_file()
    ]
    if missing_runtime_sources:
        raise ValueError(f"runtime case directory is incomplete: {missing_runtime_sources}")
    sys.path.insert(0, str(bundle_root / "repo"))
    sys.path.insert(0, str(case_dir))
    import kaggle_submission
    import train_peft_strategy

    arc_root = bundle_root / "arc_runtime"
    model = bundle_root / "base_model"
    training_data = bundle_root / "private_data/train.jsonl"
    visible_challenges = bundle_root / "private_data/visible_challenges.json"
    visible_solutions = bundle_root / "private_data/visible_solutions.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(case_dir), str(bundle_root / "repo"), env.get("PYTHONPATH", "")]
    )
    python = sys.executable

    checkpoints = [args.work_dir / "checkpoint-0900", args.work_dir / "checkpoint-1200"]
    common_train = [
        python,
        str(case_dir / "train_peft_strategy.py"),
        "--data",
        str(training_data),
        "--model",
        str(model),
        "--model-id",
        MODEL_ID,
        "--model-revision",
        MODEL_REVISION,
        "--device",
        "cuda",
        "--dtype",
        "float16",
        "--learning-rate",
        str(args.learning_rate),
        "--gradient-accumulation-steps",
        "1",
        "--max-seq-length",
        "4608",
        "--min-retained-fraction",
        "1.0",
        "--prompt-encoding",
        "compact-grid-v1",
        "--rank",
        "8",
        "--lora-alpha",
        "160",
        "--lora-dropout",
        "0",
        "--first-layer",
        "20",
        "--num-layers",
        "8",
        "--seed",
        "17",
        "--report-every",
        "50",
    ]
    training_argv = common_train[2:] + [
        "--out",
        str(checkpoints[0]),
        "--max-steps",
        "900",
        "--max-training-seconds",
        "18000",
        "--continuation-out",
        str(checkpoints[1]),
        "--continuation-steps",
        "300",
        "--continuation-training-seconds",
        "7200",
    ]
    print(
        json.dumps(
            {
                "in_process_training": {
                    "argv": training_argv,
                    "reuse_loaded_model_for_inference": True,
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    training_status, training_session = train_peft_strategy.train_with_session(
        train_peft_strategy.parser().parse_args(training_argv)
    )
    if training_status != 0 or training_session is None:
        raise RuntimeError(
            f"same-process training failed before inference: status={training_status}"
        )
    for checkpoint, expected_steps in zip(checkpoints, (900, 300), strict=True):
        training_receipt = json.loads((checkpoint / "training_receipt.json").read_text(encoding="utf-8"))
        if (
            training_receipt.get("status") != "succeeded"
            or training_receipt.get("target_steps") != expected_steps
            or training_receipt.get("completed_steps") != expected_steps
        ):
            raise RuntimeError(
                f"training phase {checkpoint.name} failed its exact-step gate: "
                f"status={training_receipt.get('status')} "
                f"target={training_receipt.get('target_steps')} "
                f"completed={training_receipt.get('completed_steps')} "
                f"expected={expected_steps}"
            )
    gc.collect()
    training_session.torch.cuda.empty_cache()

    loaded_adapter_names = {"checkpoint-1200": "default"}

    def generate(scope: str, challenges: Path, adapter: Path) -> tuple[Path, Path]:
        submission = args.work_dir / f"{scope}-{adapter.name}.json"
        receipt = args.work_dir / f"{scope}-{adapter.name}-receipt.json"
        adapter_name = loaded_adapter_names.get(adapter.name)
        if adapter_name is None:
            adapter_name = adapter.name.replace("-", "_")
            training_session.model.load_adapter(
                str(adapter),
                adapter_name=adapter_name,
                is_trainable=False,
                local_files_only=True,
            )
            loaded_adapter_names[adapter.name] = adapter_name
        inference_seconds = (
            args.visible_inference_seconds if scope == "visible" else args.hidden_inference_seconds
        )
        inference_argv = [
            "--challenges",
            str(challenges),
            "--arc-root",
            str(arc_root),
            "--model",
            str(model),
            "--adapter",
            str(adapter),
            "--arm",
            "lora",
            "--backend",
            "hf",
            "--model-id",
            MODEL_ID,
            "--model-revision",
            MODEL_REVISION,
            "--device",
            "cuda",
            "--dtype",
            "float16",
            "--peft-addition",
            "standard",
            "--max-new-tokens",
            "96",
            "--max-generation-seconds",
            str(args.max_generation_seconds),
            "--max-inference-seconds",
            str(inference_seconds),
            "--out",
            str(submission),
            "--receipt",
            str(receipt),
        ]
        router = kaggle_submission.HfStrategyRouter.from_preloaded(
            arc_root=arc_root,
            model=training_session.model,
            tokenizer=training_session.tokenizer,
            torch=training_session.torch,
            device="cuda",
            dtype="float16",
            max_new_tokens=96,
            max_generation_seconds=args.max_generation_seconds,
            adapter_name=adapter_name,
        )
        print(
            json.dumps(
                {
                    "inference_router_ready": {
                        "scope": scope,
                        "adapter": adapter.name,
                        "adapter_model_sha256": file_sha256(
                            adapter / "adapter_model.safetensors"
                        ),
                        "base_model_reloaded": False,
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )
        result = kaggle_submission.write_governed_submission(
            kaggle_submission.parser().parse_args(inference_argv),
            router=router,
            progress_scope=f"{scope}-{adapter.name}",
        )
        if result != 0:
            raise RuntimeError(
                f"same-process inference failed for {scope}/{adapter.name}: status={result}"
            )
        return submission, receipt

    visible = [generate("visible", visible_challenges, checkpoint) for checkpoint in checkpoints]
    merged_visible = args.work_dir / "visible-pass2.json"
    merged_visible_receipt = args.work_dir / "visible-pass2-receipt.json"
    checked(
        [
            python,
            str(case_dir / "merge_kaggle_submissions.py"),
            "--primary",
            str(visible[0][0]),
            "--secondary",
            str(visible[1][0]),
            "--out",
            str(merged_visible),
            "--receipt",
            str(merged_visible_receipt),
        ],
        env,
    )
    visible_score = args.work_dir / "visible-pass2-score.json"
    checked(
        [
            python,
            str(case_dir / "score_kaggle_submission.py"),
            "--submission",
            str(merged_visible),
            "--solutions",
            str(visible_solutions),
            "--scope",
            "visible-family-linux-native-kaggle-gate",
            "--out",
            str(visible_score),
        ],
        env,
    )
    score_receipt = json.loads(visible_score.read_text(encoding="utf-8"))
    router_receipts = [json.loads(path.read_text(encoding="utf-8")) for _, path in visible]
    aggregate = score_receipt["aggregate"]
    passed, failures = visible_gate(
        aggregate=aggregate,
        router_receipts=router_receipts,
        expected_total=args.expected_visible_total,
        minimum_passed=args.minimum_visible_passed,
    )
    route_receipt = {
        "format_version": "hfr.arcagi.kaggle_route.v1",
        "bundle_manifest_sha256": file_sha256(bundle_root / "bundle_manifest.json"),
        "competition_challenge_sha256": None,
        "network_disabled": True,
        "runtime_environment": runtime_environment,
        "runtime_sources": {
            filename: file_sha256(case_dir / filename) for filename in required_runtime_sources
        },
        "label_blind_hidden_inference": passed,
        "visible_gate": {
            "passed": passed,
            "failures": failures,
            "aggregate": aggregate,
            "minimum_passed": args.minimum_visible_passed,
            "expected_total": args.expected_visible_total,
        },
        "submission_sha256": None,
        "claim_boundary": "Only Kaggle can score hidden ARC-AGI-2 tasks.",
        "source_bundle": manifest.get("dataset_id"),
    }
    if not passed:
        receipt_path = args.work_dir / "route_receipt.json"
        write_json(receipt_path, route_receipt)
        receipt_path.chmod(0o600)
        raise RuntimeError(f"visible parity gate failed closed: {failures}")

    competition_challenges = find_competition_challenges(
        args.competition_input_root.resolve(), bundle_root
    )
    route_receipt["competition_challenge_sha256"] = file_sha256(competition_challenges)
    hidden = [generate("hidden", competition_challenges, checkpoint) for checkpoint in checkpoints]
    hidden_merge_receipt = args.work_dir / "hidden-pass2-receipt.json"
    checked(
        [
            python,
            str(case_dir / "merge_kaggle_submissions.py"),
            "--primary",
            str(hidden[0][0]),
            "--secondary",
            str(hidden[1][0]),
            "--out",
            str(args.submission_out),
            "--receipt",
            str(hidden_merge_receipt),
        ],
        env,
    )
    route_receipt["submission_sha256"] = file_sha256(args.submission_out)
    receipt_path = args.work_dir / "route_receipt.json"
    write_json(receipt_path, route_receipt)
    receipt_path.chmod(0o600)
    print(json.dumps({"status": "submission_ready", "sha256": route_receipt["submission_sha256"]}))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--bundle-root", type=Path, required=True)
    result.add_argument("--runtime-case-dir", type=Path)
    result.add_argument("--competition-input-root", type=Path, default=Path("/kaggle/input"))
    result.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/hfr_arcagi_route"))
    result.add_argument("--submission-out", type=Path, default=Path("/kaggle/working/submission.json"))
    result.add_argument("--learning-rate", type=float, default=1e-5)
    result.add_argument("--expected-visible-total", type=int, default=52)
    result.add_argument("--minimum-visible-passed", type=int, default=52)
    result.add_argument("--max-generation-seconds", type=float, default=120.0)
    result.add_argument("--visible-inference-seconds", type=float, default=3600.0)
    result.add_argument("--hidden-inference-seconds", type=float, default=7200.0)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
