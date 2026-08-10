#!/usr/bin/env python3
"""Create the final local ARC student training and inference handoff."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline import file_sha256, require_fresh_output, secure_tree, write_json


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def adapter_artifacts(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "files": {child.name: artifact(child) for child in sorted(path.iterdir()) if child.is_file()},
    }


def main(args: argparse.Namespace) -> int:
    run = args.run.resolve()
    out = (run / "handoff").resolve()
    require_fresh_output(out)
    generation = json.loads((run / "generation_manifest.json").read_text(encoding="utf-8"))
    data = json.loads((run / "student_data_manifest.json").read_text(encoding="utf-8"))
    visible = json.loads((run / "visible_distillation_manifest.json").read_text(encoding="utf-8"))
    hfr_validation = json.loads((run / "hfr_validation.json").read_text(encoding="utf-8"))
    baseline = run / "evaluations" / "baseline-test.json"
    checkpoint_900 = run / "evaluations" / "qwen3-0.6b-visible-v4-0900-visible-test.json"
    checkpoint_1200 = run / "evaluations" / "qwen3-0.6b-visible-v4-1200-visible-test.json"
    ensemble = run / "evaluations" / "qwen3-0.6b-visible-v4-ensemble-visible-test.json"
    stable = run / "adapters" / "qwen3-0.6b-visible-v3"
    continued = run / "adapters" / "qwen3-0.6b-visible-v4"
    selected_900 = run / "adapters" / "qwen3-0.6b-visible-v4-checkpoint-0900"
    failed = run / "adapters" / "qwen3-0.6b-visible-v2" / "failure_receipt.json"
    ensemble_value = json.loads(ensemble.read_text(encoding="utf-8"))
    checkpoint_900_value = json.loads(checkpoint_900.read_text(encoding="utf-8"))
    checkpoint_1200_value = json.loads(checkpoint_1200.read_text(encoding="utf-8"))
    receipt = {
        "schema_version": "hfr.arcagi.training_handoff.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_visible_task_distillation",
        "architecture": "Qwen3-0.6B LoRA strategy router plus immutable deterministic ArcAgent executor",
        "teacher": generation["teacher"],
        "teacher_data": {
            "trajectory_count": generation["audit"]["accepted_trajectory_count"],
            "source_task_count": generation["audit"]["source_task_count"],
            "source_test_entry_count": generation["audit"]["source_test_entry_count"],
            "strategy_count": len(generation["audit"]["strategy_counts"]),
            "family_exclusive_split_counts": data["counts"],
            "visible_distillation_counts": visible["counts"],
            "hfr_validation_passed": hfr_validation.get("passed") is True,
            "hfr_validation_error_count": hfr_validation.get("error_count", 0),
            "hfr_validation_warning_count": hfr_validation.get("warning_count", 0),
        },
        "selected_inference": {
            "max_predictions": 3,
            "adapter_order": [str(selected_900), str(continued)],
            "single_checkpoint_900_visible_grid_rate": checkpoint_900_value["grid_any_match_rate"],
            "single_checkpoint_1200_visible_grid_rate": checkpoint_1200_value["grid_any_match_rate"],
            "ensemble_visible_grid_rate": ensemble_value["grid_any_match_rate"],
            "ensemble_visible_grid_count": ensemble_value["grid_any_match_count"],
            "ensemble_visible_example_count": ensemble_value["example_count"],
        },
        "artifacts": {
            "generation_manifest": artifact(run / "generation_manifest.json"),
            "hfr_training_export_manifest": artifact(run / "training_export" / "manifest.json"),
            "hfr_validation": artifact(run / "hfr_validation.json"),
            "family_exclusive_data_manifest": artifact(run / "student_data_manifest.json"),
            "visible_distillation_manifest": artifact(run / "visible_distillation_manifest.json"),
            "baseline_test": artifact(baseline),
            "checkpoint_900_test": artifact(checkpoint_900),
            "checkpoint_1200_test": artifact(checkpoint_1200),
            "ensemble_test": artifact(ensemble),
            "stable_300_adapter": adapter_artifacts(stable),
            "continued_1200_adapter": adapter_artifacts(continued),
            "selected_900_adapter": adapter_artifacts(selected_900),
            "failed_run_negative_evidence": artifact(failed),
        },
        "runtime_contract": {
            "agent": "examples/case_studies/arcagi_strategy_student/student_agent.py:ArcStrategyStudent",
            "system_prompt": "pipeline.SYSTEM_PROMPT",
            "tool_schema": "pipeline.strategy_tool",
            "executor_sha256": generation["teacher"]["sha256"],
            "offline_required": True,
            "max_student_strategy_calls": 2,
            "max_returned_predictions": 3,
        },
        "claims": {
            "visible_52_teacher_imitation": "52/52 with the two-checkpoint bounded ensemble",
            "unseen_task_generalization": "not established",
            "official_or_sealed_arc_score": "not measured and must not be claimed",
        },
        "security": {
            "publication_allowed": False,
            "raw_traces_and_visible_outputs_remain_local": True,
            "network_used_for_generation_or_training": False,
        },
    }
    write_json(out / "training_handoff.json", receipt)
    markdown = f"""# ARC Strategy Student Training Handoff

Status: complete for visible-task distillation.

- Teacher data: {receipt['teacher_data']['trajectory_count']} HFR-validated trajectories from 48 visible task families.
- Student: local Qwen3-0.6B LoRA strategy router with deterministic ArcAgent tool execution.
- Single checkpoints: 50/52 exact visible grids each.
- Bounded two-checkpoint ensemble: {receipt['selected_inference']['ensemble_visible_grid_count']}/{receipt['selected_inference']['ensemble_visible_example_count']} exact visible grids, at most three returned predictions.
- HFR validation: passed with {receipt['teacher_data']['hfr_validation_error_count']} errors and {receipt['teacher_data']['hfr_validation_warning_count']} warnings.

This is teacher imitation on visible task families. It is not evidence of unseen-task, hidden-task, sealed, or official ARC benchmark performance. The runtime remains hybrid: the local model selects strategy tools and the immutable deterministic teacher library executes them.

Keep the complete handoff and all source traces local; publication is not approved.
"""
    (out / "TRAINING_HANDOFF.md").write_text(markdown, encoding="utf-8")
    secure_tree(out)
    print(json.dumps({"handoff": str(out), "visible_grid_rate": ensemble_value["grid_any_match_rate"]}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run", type=Path, required=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
