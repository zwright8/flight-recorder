"""CLI commands for the generation domain."""

from __future__ import annotations

from ..dataset_curation import DatasetCurationReceiptError, build_dataset_curation_receipt, write_dataset_curation_receipt
from ..model_grader import ModelGraderError, build_model_grader_disagreement_queue, build_model_grader_dry_run, build_model_grader_gate, build_model_grader_override_receipt, build_rubric_spec, write_model_grader_artifact
from pathlib import Path
from ..rejection_sampling import RejectionSamplingGateError, build_rejection_sampling_gate, write_rejection_sampling_gate
import argparse
from ..path_safety import assert_output_does_not_alias_sources, assert_output_outside_source_directories, locked_owned_output_directory, output_directory_lock_is_held, path_has_symlink_component, remove_directory_tree_if_identity
from ..rollout_generation import RolloutGenerationError, build_agentic_rollout_plan, build_agentic_rollout_receipt, write_agentic_rollout_plan, write_agentic_rollout_receipt
from ..heldout_manifest import HeldoutManifestError, build_heldout_manifest, write_heldout_manifest
import json
from ..atomic_json import AtomicJsonError, atomic_write_json_cas, json_file_sha256
from .shared import _non_negative_int_arg, _positive_int_arg, _rate_arg, _state_set_arg


def cmd_agentic_rollout_plan(args: argparse.Namespace) -> int:
    policies = dict(args.policy or [])
    plan = build_agentic_rollout_plan(
        out_path=args.out,
        iteration_id=args.iteration_id,
        scenario_paths=args.scenario,
        policies=policies,
        max_rollouts=args.max_rollouts,
        verifier_paths=args.verifier,
        environment_id=args.environment_id,
        preserve_paths=args.preserve_paths,
        created_at=args.created_at,
    )
    write_agentic_rollout_plan(args.out, plan)
    print(
        f"wrote {args.out} readiness={plan['readiness']} "
        f"planned_rollouts={plan['budget']['planned_rollouts']}/{plan['budget']['max_rollouts']}"
    )
    return 0 if plan["passed"] else 1


def cmd_agentic_rollout_receipt(args: argparse.Namespace) -> int:
    receipt = build_agentic_rollout_receipt(
        plan_path=args.plan,
        out_path=args.out,
        preserve_paths=args.preserve_paths,
        output_base_dir=Path(args.out).parent if args.out else None,
        created_at=args.created_at,
    )
    write_agentic_rollout_receipt(args.out, receipt)
    print(
        f"wrote {args.out} readiness={receipt['readiness']} "
        f"mock_rollouts={receipt['mock_rollout_count']}"
    )
    return 0 if receipt["passed"] else 1


def cmd_rejection_sampling_gate(args: argparse.Namespace) -> int:
    output_path = Path(args.out)
    source_paths = [
        Path(value)
        for values in (
            args.rollout_receipt,
            args.model_grader_gate,
            args.review_calibration,
            args.reviewed_gate,
        )
        for value in values
    ]
    try:
        assert_output_does_not_alias_sources(
            output_path,
            source_paths,
            label="rejection sampling gate",
        )
    except ValueError as exc:
        raise RejectionSamplingGateError(str(exc)) from exc
    expected_output_sha256 = json_file_sha256(output_path)
    gate = build_rejection_sampling_gate(
        rollout_receipt_paths=args.rollout_receipt,
        model_grader_gate_paths=args.model_grader_gate,
        review_calibration_paths=args.review_calibration,
        reviewed_gate_paths=args.reviewed_gate,
        out_path=args.out,
        preserve_paths=args.preserve_paths,
        created_at=args.created_at,
    )
    write_rejection_sampling_gate(
        output_path,
        gate,
        expected_sha256=expected_output_sha256,
    )
    print(
        f"wrote {args.out} readiness={gate['readiness']} "
        f"mock_rollouts={gate['rollout_summary']['mock_rollout_count']}"
    )
    return 0 if gate["passed"] else 1


def cmd_dataset_curation_receipt(args: argparse.Namespace) -> int:
    output_path = Path(args.out)
    training_exports = [Path(value) for value in args.training_export]
    try:
        assert_output_outside_source_directories(
            output_path,
            training_exports,
            label="dataset curation receipt",
        )
        assert_output_does_not_alias_sources(
            output_path,
            [Path(value) for value in args.rejection_sampling_gate],
            label="dataset curation receipt",
        )
    except ValueError as exc:
        raise DatasetCurationReceiptError(str(exc)) from exc
    expected_output_sha256 = json_file_sha256(output_path)
    receipt = build_dataset_curation_receipt(
        rejection_sampling_gate_paths=args.rejection_sampling_gate,
        training_export_paths=args.training_export,
        out_path=args.out,
        preserve_paths=args.preserve_paths,
        created_at=args.created_at,
    )
    write_dataset_curation_receipt(
        output_path,
        receipt,
        expected_sha256=expected_output_sha256,
    )
    print(
        f"wrote {args.out} readiness={receipt['readiness']} "
        f"training_exports={receipt['curation_summary']['training_export_count']}"
    )
    return 0 if receipt["passed"] else 1


def cmd_model_grader_rubric(args: argparse.Namespace) -> int:
    output_path, expected_output_sha256 = _model_grader_output_guard(
        args.out,
        source_directories=[args.review_export],
    )
    rubric = build_rubric_spec(
        review_export_dir=args.review_export,
        rubric_id=args.rubric_id,
        criteria=args.criterion,
        created_at=args.created_at,
        preserve_paths=args.preserve_paths,
        out_path=args.out,
    )
    write_model_grader_artifact(output_path, rubric, expected_sha256=expected_output_sha256)
    print(f"wrote {args.out} review_items={rubric['review_item_count']} criteria={rubric['criterion_count']}")
    return 0


def cmd_model_grader_dry_run(args: argparse.Namespace) -> int:
    output_path, expected_output_sha256 = _model_grader_output_guard(
        args.out,
        source_directories=[args.review_export],
        source_files=[args.rubric],
    )
    receipt = build_model_grader_dry_run(
        review_export_dir=args.review_export,
        rubric_path=args.rubric,
        grader_id=args.grader_id,
        provider=args.provider,
        created_at=args.created_at,
        preserve_paths=args.preserve_paths,
        out_path=args.out,
    )
    write_model_grader_artifact(output_path, receipt, expected_sha256=expected_output_sha256)
    print(
        f"wrote {args.out} readiness={receipt['readiness']} "
        f"graded_items={receipt['graded_item_count']}"
    )
    return 0 if receipt["passed"] else 1


def cmd_model_grader_disagreement_queue(args: argparse.Namespace) -> int:
    output_path, expected_output_sha256 = _model_grader_output_guard(
        args.out,
        source_files=[args.dry_run],
    )
    queue = build_model_grader_disagreement_queue(
        dry_run_path=args.dry_run,
        created_at=args.created_at,
        preserve_paths=args.preserve_paths,
        out_path=args.out,
    )
    write_model_grader_artifact(output_path, queue, expected_sha256=expected_output_sha256)
    print(
        f"wrote {args.out} readiness={queue['readiness']} "
        f"queue_items={queue['queue_count']}"
    )
    return 0 if queue["passed"] else 1


def cmd_model_grader_override_receipt(args: argparse.Namespace) -> int:
    output_path, expected_output_sha256 = _model_grader_output_guard(
        args.out,
        source_files=[args.dry_run, args.overrides],
    )
    receipt = build_model_grader_override_receipt(
        dry_run_path=args.dry_run,
        overrides_path=args.overrides,
        created_at=args.created_at,
        preserve_paths=args.preserve_paths,
        out_path=args.out,
    )
    write_model_grader_artifact(output_path, receipt, expected_sha256=expected_output_sha256)
    print(
        f"wrote {args.out} readiness={receipt['readiness']} "
        f"resolved={receipt['metrics']['resolved_queue_count']}/{receipt['queue']['dry_run_disagreement_queue_count']}"
    )
    return 0 if receipt["passed"] else 1


def cmd_model_grader_gate(args: argparse.Namespace) -> int:
    output_path, expected_output_sha256 = _model_grader_output_guard(
        args.out,
        source_files=[
            args.dry_run,
            args.rubric,
            args.review_calibration,
            args.override_receipt,
        ],
    )
    gate = build_model_grader_gate(
        dry_run_path=args.dry_run,
        rubric_path=args.rubric,
        calibration_path=args.review_calibration,
        override_receipt_path=args.override_receipt,
        min_calibration_agreement_rate=args.min_calibration_agreement_rate,
        max_disagreements=args.max_disagreements,
        created_at=args.created_at,
        preserve_paths=args.preserve_paths,
        out_path=args.out,
    )
    write_model_grader_artifact(output_path, gate, expected_sha256=expected_output_sha256)
    print(
        f"wrote {args.out} readiness={gate['readiness']} "
        f"checks={gate['check_count'] - gate['failed_check_count']}/{gate['check_count']}"
    )
    return 0 if gate["passed"] else 1


def _model_grader_output_guard(
    output: str | Path,
    *,
    source_directories: list[str | Path] | None = None,
    source_files: list[str | Path | None] | None = None,
) -> tuple[Path, str | None]:
    output_path = Path(output)
    try:
        assert_output_outside_source_directories(
            output_path,
            [Path(value) for value in source_directories or []],
            label="model-grader artifact",
        )
        assert_output_does_not_alias_sources(
            output_path,
            [Path(value) for value in source_files or [] if value is not None],
            label="model-grader artifact",
        )
    except ValueError as exc:
        raise ModelGraderError(str(exc)) from exc
    return output_path, json_file_sha256(output_path)


def cmd_heldout_manifest(args: argparse.Namespace) -> int:
    output_path = Path(args.out) if args.out else None
    expected_output_sha256 = (
        json_file_sha256(output_path) if output_path is not None else None
    )
    manifest = build_heldout_manifest(
        suite_summary_specs=args.suite_summary,
        preserve_paths=args.preserve_paths,
        out_path=output_path,
    )
    if output_path is not None:
        write_heldout_manifest(
            manifest,
            output_path,
            expected_sha256=expected_output_sha256,
        )
        print(f"wrote {args.out}")
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if manifest["ready"] else 1


def register_generation_1(subparsers: argparse._SubParsersAction) -> None:
    heldout_manifest = subparsers.add_parser("heldout-manifest", help="Build a held-out scenario manifest from suite summaries")
    heldout_manifest.add_argument(
        "--suite-summary",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="run-suite suite_summary.json to include; may be repeated",
    )
    heldout_manifest.add_argument("--out", help="Write held-out manifest JSON to this path")
    heldout_manifest.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in manifest output")
    heldout_manifest.set_defaults(func=cmd_heldout_manifest)



def register_generation_2(subparsers: argparse._SubParsersAction) -> None:
    agentic_rollout_plan = subparsers.add_parser("agentic-rollout-plan", help="Write a fail-closed rollout generation plan")
    agentic_rollout_plan.add_argument("--iteration-id", required=True, help="Stable iteration id")
    agentic_rollout_plan.add_argument("--scenario", action="append", required=True, help="Scenario JSON path; may be repeated")
    agentic_rollout_plan.add_argument(
        "--policy",
        action="append",
        required=True,
        type=_state_set_arg,
        metavar="ROLE=ID",
        help="Policy role mapping such as baseline=local/base; may be repeated",
    )
    agentic_rollout_plan.add_argument("--max-rollouts", type=_positive_int_arg, required=True, help="Maximum planned rollout count")
    agentic_rollout_plan.add_argument("--verifier", action="append", default=[], help="External-state verifier config path; may be repeated")
    agentic_rollout_plan.add_argument("--environment-id", default="offline_mock", help="Replayable environment descriptor id")
    agentic_rollout_plan.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    agentic_rollout_plan.add_argument("--out", required=True, help="Write hfr.agentic_rollout_plan.v1 JSON to this path")
    agentic_rollout_plan.add_argument("--preserve-paths", action="store_true", help="Preserve safe source path text in rollout plan; unsafe absolute or traversal refs remain redacted")
    agentic_rollout_plan.set_defaults(func=cmd_agentic_rollout_plan)

    agentic_rollout_receipt = subparsers.add_parser("agentic-rollout-receipt", help="Write a fail-closed mock rollout receipt")
    agentic_rollout_receipt.add_argument("--plan", required=True, help="hfr.agentic_rollout_plan.v1 JSON path")
    agentic_rollout_receipt.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    agentic_rollout_receipt.add_argument("--out", required=True, help="Write hfr.agentic_rollout_receipt.v1 JSON to this path")
    agentic_rollout_receipt.add_argument("--preserve-paths", action="store_true", help="Preserve safe source path text in rollout receipt; unsafe absolute or traversal refs remain redacted")
    agentic_rollout_receipt.set_defaults(func=cmd_agentic_rollout_receipt)

    rejection_sampling_gate = subparsers.add_parser("rejection-sampling-gate", help="Write a fail-closed rejection sampling admission gate")
    rejection_sampling_gate.add_argument("--rollout-receipt", action="append", required=True, help="agentic_rollout_receipt artifact; may be repeated")
    rejection_sampling_gate.add_argument("--model-grader-gate", action="append", required=True, help="model_grader_gate artifact; may be repeated")
    rejection_sampling_gate.add_argument("--review-calibration", action="append", required=True, help="review_calibration artifact; may be repeated")
    rejection_sampling_gate.add_argument("--reviewed-gate", action="append", required=True, help="reviewed_gate artifact; may be repeated")
    rejection_sampling_gate.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    rejection_sampling_gate.add_argument("--out", required=True, help="Write hfr.rejection_sampling_gate.v1 JSON to this path")
    rejection_sampling_gate.add_argument("--preserve-paths", action="store_true", help="Preserve safe source path text in gate refs; unsafe absolute or traversal refs remain redacted")
    rejection_sampling_gate.set_defaults(func=cmd_rejection_sampling_gate)

    dataset_curation_receipt = subparsers.add_parser("dataset-curation-receipt", help="Write a side-effect-free dataset curation receipt")
    dataset_curation_receipt.add_argument("--rejection-sampling-gate", action="append", required=True, help="rejection_sampling_gate artifact; may be repeated")
    dataset_curation_receipt.add_argument("--training-export", action="append", required=True, help="training export directory or manifest; may be repeated")
    dataset_curation_receipt.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    dataset_curation_receipt.add_argument("--out", required=True, help="Write hfr.dataset_curation_receipt.v1 JSON to this path")
    dataset_curation_receipt.add_argument("--preserve-paths", action="store_true", help="Preserve safe source path text in receipt refs; unsafe absolute or traversal refs remain redacted")
    dataset_curation_receipt.set_defaults(func=cmd_dataset_curation_receipt)

    model_grader = subparsers.add_parser("model-grader", help="Build fail-closed rubric/model-grader review contracts")
    model_grader_subparsers = model_grader.add_subparsers(dest="model_grader_command", required=True)
    model_grader_rubric = model_grader_subparsers.add_parser("rubric", help="Write a rubric spec bound to a review export")
    model_grader_rubric.add_argument("--review-export", required=True, help="export-review directory")
    model_grader_rubric.add_argument("--rubric-id", required=True, help="Stable rubric id")
    model_grader_rubric.add_argument("--criterion", action="append", default=[], help="Rubric criterion text; may be repeated")
    model_grader_rubric.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    model_grader_rubric.add_argument("--out", required=True, help="Write hfr.rubric_spec.v1 JSON to this path")
    model_grader_rubric.add_argument("--preserve-paths", action="store_true", help="Preserve safe source path text in rubric refs; unsafe absolute refs remain redacted")
    model_grader_rubric.set_defaults(func=cmd_model_grader_rubric)

    model_grader_dry_run = model_grader_subparsers.add_parser("dry-run", help="Write a mock model-grader dry-run receipt")
    model_grader_dry_run.add_argument("--review-export", required=True, help="export-review directory")
    model_grader_dry_run.add_argument("--rubric", required=True, help="rubric_spec artifact")
    model_grader_dry_run.add_argument("--grader-id", default="mock-grader", help="Stable grader id recorded in mock labels")
    model_grader_dry_run.add_argument("--provider", default="mock", help="Provider id recorded in the dry-run receipt")
    model_grader_dry_run.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    model_grader_dry_run.add_argument("--out", required=True, help="Write hfr.model_grader_dry_run.v1 JSON to this path")
    model_grader_dry_run.add_argument("--preserve-paths", action="store_true", help="Preserve safe source path text in dry-run refs; unsafe absolute refs remain redacted")
    model_grader_dry_run.set_defaults(func=cmd_model_grader_dry_run)

    model_grader_queue = model_grader_subparsers.add_parser(
        "disagreement-queue",
        help="Write a portable human-review queue from a model-grader dry-run receipt",
    )
    model_grader_queue.add_argument("--dry-run", required=True, help="model_grader_dry_run artifact")
    model_grader_queue.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    model_grader_queue.add_argument("--out", required=True, help="Write hfr.model_grader_disagreement_queue.v1 JSON to this path")
    model_grader_queue.add_argument("--preserve-paths", action="store_true", help="Preserve safe source path text in queue refs; unsafe absolute refs remain redacted")
    model_grader_queue.set_defaults(func=cmd_model_grader_disagreement_queue)

    model_grader_override = model_grader_subparsers.add_parser(
        "override-receipt",
        help="Resolve model-grader dry-run disagreement queue items with human overrides",
    )
    model_grader_override.add_argument("--dry-run", required=True, help="model_grader_dry_run artifact")
    model_grader_override.add_argument("--overrides", required=True, help="JSONL human override rows")
    model_grader_override.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    model_grader_override.add_argument("--out", required=True, help="Write hfr.model_grader_override_receipt.v1 JSON to this path")
    model_grader_override.add_argument("--preserve-paths", action="store_true", help="Preserve safe source path text in override refs; unsafe absolute refs remain redacted")
    model_grader_override.set_defaults(func=cmd_model_grader_override_receipt)

    model_grader_gate = model_grader_subparsers.add_parser("gate", help="Gate model-grader labels before training admission")
    model_grader_gate.add_argument("--dry-run", required=True, help="model_grader_dry_run artifact")
    model_grader_gate.add_argument("--rubric", required=True, help="rubric_spec artifact")
    model_grader_gate.add_argument("--review-calibration", help="review_calibration artifact required for a passing gate")
    model_grader_gate.add_argument("--override-receipt", help="model_grader_override_receipt required when dry-run labels need review")
    model_grader_gate.add_argument("--min-calibration-agreement-rate", type=_rate_arg, default=0.8)
    model_grader_gate.add_argument("--max-disagreements", type=_non_negative_int_arg)
    model_grader_gate.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    model_grader_gate.add_argument("--out", required=True, help="Write hfr.model_grader_gate.v1 JSON to this path")
    model_grader_gate.add_argument("--preserve-paths", action="store_true", help="Preserve safe source path text in gate refs; unsafe absolute refs remain redacted")
    model_grader_gate.set_defaults(func=cmd_model_grader_gate)

