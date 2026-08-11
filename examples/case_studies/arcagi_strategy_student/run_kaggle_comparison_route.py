#!/usr/bin/env python3
"""Run a fail-closed, label-blind base-versus-LoRA Kaggle comparison."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from run_kaggle_route import (
    MODEL_ID,
    MODEL_REVISION,
    checked,
    environment_receipt,
    file_sha256,
    find_competition_challenges,
    verify_bundle,
    write_json,
)


CANDIDATE_POLICY = "router-only"


def comparison_visible_gate(
    *,
    aggregate: dict[str, Any],
    router_receipt: dict[str, Any],
    expected_total: int,
    minimum_passed: int,
) -> tuple[bool, list[str]]:
    """Gate a complete visible router arm while retaining invalidity as evidence."""
    failures = []
    if aggregate.get("total") != expected_total:
        failures.append("visible_total_mismatch")
    if aggregate.get("passed", -1) < minimum_passed:
        failures.append("visible_accuracy_below_threshold")
    if router_receipt.get("error_count") != 0:
        failures.append("router_execution_error")
    inference = router_receipt.get("inference", {})
    if inference.get("candidate_policy") != CANDIDATE_POLICY:
        failures.append("candidate_policy_mismatch")
    if router_receipt.get("task_count") != expected_total:
        failures.append("router_task_count_mismatch")
    invalid = router_receipt.get("invalid_strategy_count")
    if isinstance(invalid, bool) or not isinstance(invalid, int) or not 0 <= invalid <= expected_total:
        failures.append("invalid_strategy_count_malformed")
    return not failures, failures


def comparison_hidden_gate(
    *,
    router_receipt: dict[str, Any],
    expected_task_count: int,
    expected_test_input_count: int,
    expected_adapter_enabled: bool,
) -> tuple[bool, list[str]]:
    """Fail closed before promoting a label-blind hidden arm to submission.json."""
    failures = []
    if router_receipt.get("task_count") != expected_task_count:
        failures.append("hidden_task_count_mismatch")
    if router_receipt.get("test_input_count") != expected_test_input_count:
        failures.append("hidden_test_input_count_mismatch")
    if router_receipt.get("error_count") != 0:
        failures.append("hidden_router_execution_error")
    if router_receipt.get("inference", {}).get("candidate_policy") != CANDIDATE_POLICY:
        failures.append("hidden_candidate_policy_mismatch")
    if router_receipt.get("adapter", {}).get("enabled") is not expected_adapter_enabled:
        failures.append("hidden_adapter_state_mismatch")
    invalid = router_receipt.get("invalid_strategy_count")
    if (
        isinstance(invalid, bool)
        or not isinstance(invalid, int)
        or not 0 <= invalid <= expected_test_input_count
    ):
        failures.append("hidden_invalid_strategy_count_malformed")
    return not failures, failures


def verify_frozen_adapter(adapter_root: Path) -> dict[str, Any]:
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "training_receipt.json",
    }
    actual = {path.name for path in adapter_root.iterdir() if path.is_file()}
    if actual != required:
        raise ValueError(f"frozen adapter file set differs: expected={sorted(required)} actual={sorted(actual)}")
    receipt = json.loads((adapter_root / "training_receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "succeeded"
        or receipt.get("target_steps") != 900
        or receipt.get("completed_steps") != 900
    ):
        raise ValueError("frozen adapter did not pass the exact 900-step gate")
    outputs = receipt.get("outputs", {})
    expected = {
        "adapter_config_sha256": file_sha256(adapter_root / "adapter_config.json"),
        "adapter_model_sha256": file_sha256(adapter_root / "adapter_model.safetensors"),
    }
    if outputs != expected:
        raise ValueError("frozen adapter hashes do not match its training receipt")
    return receipt


def run(args: argparse.Namespace) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    bundle_root = args.bundle_root.resolve()
    manifest = verify_bundle(bundle_root)
    adapter_root = args.adapter_root.resolve()
    adapter_receipt = verify_frozen_adapter(adapter_root)
    runtime_environment = environment_receipt()
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise ValueError("refusing to overwrite non-empty Kaggle comparison work directory")
    for output in (args.base_submission_out, args.submission_out):
        if output.exists():
            raise ValueError(f"refusing to overwrite an existing Kaggle submission: {output}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.chmod(0o700)

    case_dir = args.runtime_case_dir.resolve()
    required_runtime_sources = (
        "evaluate_student.py",
        "kaggle_submission.py",
        "pipeline.py",
        "run_kaggle_comparison_route.py",
        "run_kaggle_route.py",
        "score_kaggle_submission.py",
    )
    missing = [name for name in required_runtime_sources if not (case_dir / name).is_file()]
    if missing:
        raise ValueError(f"comparison runtime case directory is incomplete: {missing}")
    sys.path.insert(0, str(bundle_root / "repo"))
    sys.path.insert(0, str(case_dir))
    import kaggle_submission
    import torch

    arc_root = bundle_root / "arc_runtime"
    model = bundle_root / "base_model"
    visible_challenges = bundle_root / "private_data/visible_challenges.json"
    visible_solutions = bundle_root / "private_data/visible_solutions.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(case_dir), str(bundle_root / "repo"), env.get("PYTHONPATH", "")]
    )

    print(
        json.dumps(
            {
                "comparison_model_load_started": {
                    "adapter_model_sha256": file_sha256(
                        adapter_root / "adapter_model.safetensors"
                    ),
                    "load_once_for_all_arms": True,
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    shared_router = kaggle_submission.HfStrategyRouter(
        arc_root=arc_root,
        model_path=model,
        adapter_path=adapter_root,
        device="cuda",
        dtype="float16",
        peft_addition="standard",
        max_new_tokens=96,
        max_generation_seconds=args.max_generation_seconds,
        candidate_policy=CANDIDATE_POLICY,
    )
    print(
        json.dumps(
            {
                "comparison_model_load_completed": {
                    "adapter_load_method": shared_router.adapter_load_method,
                    "load_once_for_all_arms": True,
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )

    def generate(scope: str, challenges: Path, arm: str, output: Path) -> Path:
        receipt = args.work_dir / f"{scope}-{arm}-receipt.json"
        adapter = adapter_root if arm == "lora" else None
        inference_seconds = (
            args.visible_inference_seconds if scope == "visible" else args.hidden_inference_seconds
        )
        argv = [
            "--challenges",
            str(challenges),
            "--arc-root",
            str(arc_root),
            "--model",
            str(model),
            "--arm",
            arm,
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
            "--candidate-policy",
            CANDIDATE_POLICY,
            "--out",
            str(output),
            "--receipt",
            str(receipt),
        ]
        if adapter is not None:
            argv.extend(["--adapter", str(adapter)])
        print(
            json.dumps(
                {
                    "comparison_arm_ready": {
                        "scope": scope,
                        "arm": arm,
                        "candidate_policy": CANDIDATE_POLICY,
                        "adapter_model_sha256": (
                            file_sha256(adapter / "adapter_model.safetensors") if adapter else None
                        ),
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )
        shared_router.set_adapter_enabled(arm == "lora")
        status = kaggle_submission.write_governed_submission(
            kaggle_submission.parser().parse_args(argv),
            router=shared_router,
            progress_scope=f"{scope}-{arm}",
        )
        if status != 0:
            raise RuntimeError(f"comparison inference failed for {scope}/{arm}: {status}")
        return receipt

    try:
        visible_receipts: dict[str, Path] = {}
        visible_scores: dict[str, dict[str, Any]] = {}
        for arm in ("base", "lora"):
            submission = args.work_dir / f"visible-{arm}.json"
            visible_receipts[arm] = generate("visible", visible_challenges, arm, submission)
            score_path = args.work_dir / f"visible-{arm}-score.json"
            checked(
                [
                    sys.executable,
                    str(case_dir / "score_kaggle_submission.py"),
                    "--submission",
                    str(submission),
                    "--solutions",
                    str(visible_solutions),
                    "--scope",
                    f"visible-family-{arm}-router-only-gate",
                    "--out",
                    str(score_path),
                ],
                env,
            )
            visible_scores[arm] = json.loads(score_path.read_text(encoding="utf-8"))

        visible_gates: dict[str, dict[str, Any]] = {}
        all_visible_passed = True
        for arm in ("base", "lora"):
            arm_receipt = json.loads(visible_receipts[arm].read_text(encoding="utf-8"))
            aggregate = visible_scores[arm]["aggregate"]
            passed, failures = comparison_visible_gate(
                aggregate=aggregate,
                router_receipt=arm_receipt,
                expected_total=args.expected_visible_total,
                minimum_passed=args.minimum_visible_passed,
            )
            all_visible_passed = all_visible_passed and passed
            visible_gates[arm] = {
                "passed": passed,
                "failures": failures,
                "aggregate": aggregate,
                "minimum_passed": args.minimum_visible_passed,
                "expected_total": args.expected_visible_total,
            }

        route_receipt = {
            "format_version": "hfr.arcagi.kaggle_controlled_comparison.v2",
            "bundle_manifest_sha256": file_sha256(bundle_root / "bundle_manifest.json"),
            "competition_challenge_sha256": None,
            "network_disabled": True,
            "runtime_environment": runtime_environment,
            "runtime_sources": {
                name: file_sha256(case_dir / name) for name in required_runtime_sources
            },
            "adapter": {
                "training_receipt_sha256": file_sha256(
                    adapter_root / "training_receipt.json"
                ),
                "adapter_config_sha256": adapter_receipt["outputs"][
                    "adapter_config_sha256"
                ],
                "adapter_model_sha256": adapter_receipt["outputs"][
                    "adapter_model_sha256"
                ],
                "completed_steps": adapter_receipt["completed_steps"],
            },
            "controls": {
                "candidate_policy": CANDIDATE_POLICY,
                "same_model_prompt_tools_context_and_decoding": True,
                "only_arm_difference": "frozen_peft_adapter_enabled",
                "deterministic_executor_identical": True,
                "teacher_used_only_for_training": True,
                "teacher_fallback_at_inference": False,
                "router_invalid_actions_retained_as_comparison_evidence": True,
            },
            "label_blind_hidden_inference": all_visible_passed,
            "visible_gates": visible_gates,
            "hidden_gates": None,
            "submission_sha256": {"base": None, "lora": None},
            "claim_boundary": (
                "Official score differences compare a frozen base strategy router with the "
                "HFR-trained LoRA router while holding the selected deterministic executor and "
                "all inference controls fixed. The executable teacher is not used as a sealed "
                "inference fallback."
            ),
            "source_bundle": manifest.get("dataset_id"),
        }
        receipt_path = args.work_dir / "comparison-route-receipt.json"
        if not all_visible_passed:
            write_json(receipt_path, route_receipt)
            receipt_path.chmod(0o600)
            raise RuntimeError("controlled comparison failed closed before hidden inference")

        competition_challenges = find_competition_challenges(
            args.competition_input_root.resolve(), bundle_root
        )
        route_receipt["competition_challenge_sha256"] = file_sha256(competition_challenges)
        hidden_challenges = kaggle_submission.validate_label_blind_challenges(
            json.loads(competition_challenges.read_text(encoding="utf-8"))
        )
        expected_hidden_tasks = len(hidden_challenges)
        expected_hidden_inputs = sum(
            len(task["test"]) for task in hidden_challenges.values()
        )
        hidden_outputs = {
            arm: args.work_dir / f"hidden-{arm}.json" for arm in ("base", "lora")
        }
        hidden_receipts = {
            arm: generate("hidden", competition_challenges, arm, hidden_outputs[arm])
            for arm in ("base", "lora")
        }
        hidden_gates = {}
        all_hidden_passed = True
        for arm in ("base", "lora"):
            passed, failures = comparison_hidden_gate(
                router_receipt=json.loads(
                    hidden_receipts[arm].read_text(encoding="utf-8")
                ),
                expected_task_count=expected_hidden_tasks,
                expected_test_input_count=expected_hidden_inputs,
                expected_adapter_enabled=arm == "lora",
            )
            all_hidden_passed = all_hidden_passed and passed
            hidden_gates[arm] = {"passed": passed, "failures": failures}
        route_receipt["hidden_gates"] = hidden_gates
        if not all_hidden_passed:
            write_json(receipt_path, route_receipt)
            receipt_path.chmod(0o600)
            raise RuntimeError("controlled comparison failed closed before submission promotion")

        for source, destination in (
            (hidden_outputs["base"], args.base_submission_out),
            (hidden_outputs["lora"], args.submission_out),
        ):
            shutil.copyfile(source, destination)
            destination.chmod(0o600)
        route_receipt["submission_sha256"] = {
            "base": file_sha256(args.base_submission_out),
            "lora": file_sha256(args.submission_out),
        }
        write_json(receipt_path, route_receipt)
        receipt_path.chmod(0o600)
        print(
            json.dumps(
                {"status": "controlled_submissions_ready", **route_receipt["submission_sha256"]},
                sort_keys=True,
            )
        )
        return 0
    finally:
        del shared_router
        gc.collect()
        torch.cuda.empty_cache()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--bundle-root", type=Path, required=True)
    result.add_argument("--runtime-case-dir", type=Path, required=True)
    result.add_argument("--adapter-root", type=Path, required=True)
    result.add_argument("--competition-input-root", type=Path, default=Path("/kaggle/input"))
    result.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/hfr_arcagi_comparison"))
    result.add_argument("--base-submission-out", type=Path, default=Path("/kaggle/working/submission-base.json"))
    result.add_argument("--submission-out", type=Path, default=Path("/kaggle/working/submission.json"))
    result.add_argument("--expected-visible-total", type=int, default=52)
    result.add_argument("--minimum-visible-passed", type=int, default=0)
    result.add_argument("--max-generation-seconds", type=float, default=120.0)
    result.add_argument("--visible-inference-seconds", type=float, default=3600.0)
    result.add_argument("--hidden-inference-seconds", type=float, default=7200.0)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
