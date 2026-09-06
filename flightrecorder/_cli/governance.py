"""CLI commands for the governance domain."""

from __future__ import annotations

from pathlib import Path
from ..calibration import ReviewCalibrationError, build_review_calibration, write_review_calibration
from ..review import REVIEW_LABELS, ReviewExportError, apply_review_labels, export_review_queue
import argparse
from ..path_safety import assert_output_does_not_alias_sources, assert_output_outside_source_directories, locked_owned_output_directory, output_directory_lock_is_held, path_has_symlink_component, remove_directory_tree_if_identity
from ..atomic_json import AtomicJsonError, atomic_write_json_cas, json_file_sha256
from ..agentic_training_flow import AgenticTrainingFlowError, build_agentic_training_flow, write_agentic_training_flow
from ..promotion_archive import PromotionArchiveError, build_promotion_archive
from ..governance import PromotionDecisionError, apply_promotion_aliases, build_promotion_cards, build_promotion_decision, build_promotion_release_record, build_promotion_rollback_receipt
from ..promotion_ledger import PromotionLedgerError, build_promotion_ledger
from ..reviewed_gate import REVIEWED_GATE_POLICY_SCHEMA_VERSION, ReviewedGateError, ReviewedGatePolicyError, build_reviewed_export_source_artifact, evaluate_reviewed_gate, load_reviewed_gate_policy, snapshot_reviewed_export
from ..trainer_archive import TrainerArchiveError, build_trainer_archive
from ..trainer_archive_check import TrainerArchiveCheckError, build_trainer_archive_check
from ..trainer_consumer_plan import TrainerConsumerPlanError, build_trainer_consumer_plan, reject_symlinked_archive_check_input
from ..scenario_draft import draft_scenario, safe_scenario_id, score_draft, title_from_id
import json
from ..adapters import AdapterError, normalize_trace
import os
from ..validation import EVAL_SUITE_MANIFEST_SCHEMA_VERSION, VALIDATION_SCHEMA_VERSION, validate_artifacts, validate_trainer_preflight
from .constants import TRACE_FORMAT_CHOICES
from .shared import _metadata_options, _non_negative_int_arg, _rate_arg, _read_json, _write_json


def cmd_promotion_release_record(args: argparse.Namespace) -> int:
    validation_summary = validate_artifacts(
        promotion_decision_paths=[args.promotion_decision],
        promotion_cards_paths=[args.promotion_cards],
        promotion_alias_apply_paths=[args.promotion_alias_apply],
        strict=True,
    )
    record = build_promotion_release_record(
        release_id=args.release_id,
        promotion_decision_path=args.promotion_decision,
        promotion_cards_path=args.promotion_cards,
        promotion_alias_apply_path=args.promotion_alias_apply,
        rollback_metadata_path=args.rollback_metadata,
        compare_gate_path=args.compare_gate,
        release_notes_path=args.release_notes,
        out_path=args.out,
        promotion_policy_path=args.promotion_policy,
        artifact_validation=validation_summary,
        preserve_paths=args.preserve_paths,
        metadata=_metadata_options(args.metadata),
    )
    print(
        f"{'READY' if record['passed'] else 'BLOCKED'} promotion-release-record "
        f"checks={record['check_count'] - record['failed_check_count']}/{record['check_count']} "
        f"out={args.out}"
    )
    return 0 if record["passed"] else 1


def cmd_promotion_decision(args: argparse.Namespace) -> int:
    output_path = Path(args.out) if args.out else None
    expected_output_sha256 = (
        json_file_sha256(output_path) if output_path is not None else None
    )
    decision = build_promotion_decision(
        candidate_id=args.candidate_id,
        champion_id=args.champion_id,
        rollback_id=args.rollback_id,
        candidate_class=args.candidate_class,
        champion_class=args.champion_class,
        out_path=args.out,
        evidence_bundle_path=args.evidence_bundle,
        eval_summary_path=args.eval_summary,
        external_eval_result_paths=args.external_eval_result,
        promotion_ledger_gate_path=args.promotion_ledger_gate,
        compare_gate_path=args.compare_gate,
        trainer_launch_check_path=args.trainer_launch_check,
        model_registry_entry_path=args.model_registry_entry,
        agentic_training_result_path=args.agentic_training_result,
        cloud_training_completion_receipt_path=args.cloud_training_completion_receipt,
        model_card_path=args.model_card,
        dataset_card_path=args.dataset_card,
        rollback_metadata_path=args.rollback_metadata,
        license_review_path=args.license_review,
        redaction_check_path=args.redaction_check,
        safety_gate_path=args.safety_gate,
        serving_profile_path=args.serving_profile,
        serving_report_path=args.serving_report,
        promotion_policy_path=args.promotion_policy,
        preserve_paths=args.preserve_paths,
        metadata=_metadata_options(args.metadata),
    )
    if output_path is not None:
        atomic_write_json_cas(
            output_path,
            decision,
            expected_sha256=expected_output_sha256,
        )
        print(f"wrote {args.out}")
    else:
        print(json.dumps(decision, indent=2, sort_keys=True, ensure_ascii=False) + "\n", end="")
    return 0 if decision["passed"] else 1


def cmd_promotion_ledger(args: argparse.Namespace) -> int:
    ledger = build_promotion_ledger(
        args.decision_gate,
        out_path=args.out,
        preserve_paths=args.preserve_paths,
    )
    rendered = json.dumps(ledger, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0


def cmd_promotion_archive(args: argparse.Namespace) -> int:
    archive = build_promotion_archive(
        out_dir=args.out,
        promotion_ledger_path=args.promotion_ledger,
        promotion_ledger_gate_path=args.promotion_ledger_gate,
        decision_gate_paths=args.decision_gate,
        promotion_release_record_paths=args.promotion_release_record,
        require_self_contained=args.require_self_contained,
        force=args.force,
        preserve_paths=args.preserve_paths,
    )
    print(
        f"{'READY' if archive['passed'] else 'INCOMPLETE'} promotion-archive "
        f"artifacts={archive['metrics']['artifact_count']} "
        f"missing={archive['metrics']['missing_count']} out={args.out}"
    )
    return 0 if archive["passed"] else 1


def cmd_trainer_archive(args: argparse.Namespace) -> int:
    archive = build_trainer_archive(
        out_dir=args.out,
        preflight_path=args.preflight,
        launch_check_path=args.launch_check,
        require_self_contained=args.require_self_contained,
        force=args.force,
        preserve_paths=args.preserve_paths,
    )
    print(
        f"{'READY' if archive['passed'] else 'BLOCKED'} trainer-archive "
        f"artifacts={archive['metrics']['artifact_count']} "
        f"missing={archive['metrics']['missing_count']} out={args.out}"
    )
    return 0 if archive["passed"] else 1


def cmd_trainer_archive_check(args: argparse.Namespace) -> int:
    validation_summary = validate_artifacts(trainer_archive_paths=[args.archive], strict=args.strict)
    check = build_trainer_archive_check(
        archive_path=args.archive,
        external_code_root=args.external_code_root,
        validation_summary=validation_summary,
        preserve_paths=args.preserve_paths,
    )
    _write_json(Path(args.out), check)
    print(
        f"{'READY' if check['passed'] else 'BLOCKED'} trainer-archive-check "
        f"external_paths={check['metrics']['external_command_path_count']} "
        f"missing_external={check['metrics']['missing_external_code_count']} out={args.out}"
    )
    return 0 if check["passed"] else 1


def cmd_trainer_consumer_plan(args: argparse.Namespace) -> int:
    archive_check_path = Path(args.archive_check)
    reject_symlinked_archive_check_input(archive_check_path)
    archive_check = _read_json(archive_check_path)
    validation_summary = validate_artifacts(trainer_archive_check_paths=[archive_check_path], strict=args.strict)
    plan = build_trainer_consumer_plan(
        out_path=args.out,
        archive_check_path=archive_check_path,
        archive_check=archive_check,
        validation_summary=validation_summary,
        preserve_paths=args.preserve_paths,
    )
    _write_json(Path(args.out), plan)
    print(
        f"{'READY' if plan['passed'] else 'BLOCKED'} trainer-consumer-plan "
        f"inputs={plan['metrics']['trainer_input_count']} "
        f"external_code={plan['metrics']['external_code_file_count']} out={args.out}"
    )
    return 0 if plan["passed"] else 1


def cmd_agentic_training_flow(args: argparse.Namespace) -> int:
    receipt = build_agentic_training_flow(
        plan_path=args.plan,
        runtime_preflight_path=args.runtime_preflight,
        trainer_consumer_plan_path=args.trainer_consumer_plan,
        out_path=args.out,
        flow_id=args.flow_id or "",
        preserve_paths=args.preserve_paths,
        created_at=args.created_at,
    )
    write_agentic_training_flow(args.out, receipt)
    print(
        f"{'READY' if receipt['passed'] else 'BLOCKED'} agentic-training-flow "
        f"mode={receipt['delegated_flow']['mode']} "
        f"stages={receipt['metrics']['stage_count']} out={args.out}"
    )
    return 0 if receipt["passed"] else 1


def cmd_export_review(args: argparse.Namespace) -> int:
    manifest = export_review_queue(
        args.runs,
        args.out,
        only_failed=args.only_failed,
        preserve_paths=args.preserve_paths,
    )
    print(f"wrote review queue {args.out} items={manifest['item_count']}")
    return 0


def cmd_apply_review(args: argparse.Namespace) -> int:
    manifest = apply_review_labels(
        args.review_export,
        args.out,
        labels_path=args.labels,
        max_pairs_per_family=args.max_pairs_per_family,
        preserve_paths=args.preserve_paths,
    )
    print(f"wrote reviewed export {args.out} labels={manifest['reviewed_label_count']}")
    return 0


def cmd_review_calibration(args: argparse.Namespace) -> int:
    reviewed_dir = Path(args.reviewed_export)
    output_path = Path(args.out)
    try:
        assert_output_outside_source_directories(
            output_path,
            [reviewed_dir],
            label="review calibration",
        )
        assert_output_does_not_alias_sources(
            output_path,
            [
                reviewed_dir / name
                for name in (
                    "manifest.json",
                    "dataset_registry.json",
                    "reviewed_labels.jsonl",
                    "reviewed_sft.jsonl",
                    "reviewed_reward_model.jsonl",
                    "reviewed_preferences.jsonl",
                    "reviewed_dpo.jsonl",
                    "provenance/review_items.jsonl",
                    "provenance/label_template.jsonl",
                    "provenance/review_manifest.json",
                    "provenance/completed_labels.jsonl",
                )
            ],
            label="review calibration",
        )
    except ValueError as exc:
        raise ReviewCalibrationError(str(exc)) from exc
    expected_output_sha256 = json_file_sha256(output_path)
    reviewed_export_display_path = os.path.relpath(
        reviewed_dir.resolve(),
        output_path.parent.resolve(),
    )
    reviewed_export_ref = Path(reviewed_export_display_path)
    if reviewed_export_display_path in {"", "."} or ".." in reviewed_export_ref.parts:
        raise ReviewCalibrationError(
            "review calibration output must be placed so --reviewed-export has a normalized, non-traversing relative path"
        )
    source_before = build_reviewed_export_source_artifact(
        reviewed_dir,
        display_path=reviewed_export_display_path,
    )
    validation_summary = (
        validate_artifacts(reviewed_export_dir=reviewed_dir, strict=args.strict_validation)
        if not args.skip_validation
        else None
    )
    calibration = build_review_calibration(
        reviewed_dir,
        min_agreement_rate=args.min_agreement_rate,
        max_disagreements=args.max_disagreements,
        max_false_positives=args.max_false_positives,
        max_false_negatives=args.max_false_negatives,
        min_comparable_labels=args.min_comparable_labels,
        validation_summary=validation_summary,
        require_valid_export=not args.skip_validation,
        strict_validation=args.strict_validation,
        preserve_paths=args.preserve_paths,
        out_path=args.out,
    )
    source_after = build_reviewed_export_source_artifact(
        reviewed_dir,
        display_path=reviewed_export_display_path,
    )
    if source_before != source_after:
        raise ReviewCalibrationError("reviewed export changed while calibration was being evaluated")
    write_review_calibration(
        args.out,
        calibration,
        expected_sha256=expected_output_sha256,
    )
    metrics = calibration["metrics"]
    print(
        "wrote review calibration "
        f"agreement_rate={metrics['agreement_rate']} "
        f"disagreements={metrics['disagreement_count']} out={args.out}"
    )
    return 0 if calibration["passed"] else 1


def cmd_draft_scenario(args: argparse.Namespace) -> int:
    if args.run:
        source_path = Path(args.run) / "normalized_trace.json"
        trace = _read_json(source_path)
        trace_format = "normalized_json"
        source_label = str(source_path)
    else:
        source_path = Path(args.trace)
        trace = normalize_trace(source_path, args.format)
        trace_format = args.format
        source_label = str(source_path)

    scenario_id = safe_scenario_id(args.id or Path(args.run or args.trace).stem)
    scenario = draft_scenario(
        trace,
        scenario_id=scenario_id,
        title=args.title or f"Draft: {title_from_id(scenario_id)}",
        prompt=args.prompt or "TODO: describe the intended task for this run.",
        trace_path=source_path,
        trace_format=trace_format,
        out_path=args.out,
        max_actions=args.max_actions,
        preserve_paths=args.preserve_paths,
    )
    _write_json(Path(args.out), scenario)
    source_score = score_draft(scenario, trace)
    print(
        f"wrote {args.out} from {source_label} "
        f"source_score={source_score['score']} source_passed={str(source_score['passed']).lower()}"
    )
    return 0


def register_governance_1(subparsers: argparse._SubParsersAction) -> None:
    agentic_training_flow = subparsers.add_parser("agentic-training-flow", help="Write a fail-closed delegated trainer-flow receipt")
    agentic_training_flow.add_argument("--plan", required=True, help="agentic_training_plan artifact")
    agentic_training_flow.add_argument("--runtime-preflight", required=True, help="agentic_training_runtime_preflight artifact")
    agentic_training_flow.add_argument("--trainer-consumer-plan", required=True, help="trainer_consumer_plan artifact")
    agentic_training_flow.add_argument("--flow-id", help="Optional stable flow id")
    agentic_training_flow.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    agentic_training_flow.add_argument("--out", required=True, help="Write hfr.agentic_training_flow.v1 JSON to this path")
    agentic_training_flow.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in flow refs")
    agentic_training_flow.set_defaults(func=cmd_agentic_training_flow)



def register_governance_2(subparsers: argparse._SubParsersAction) -> None:
    promotion_release_record = subparsers.add_parser(
        "promotion-release-record",
        help="Bind promotion decisions, cards, alias receipts, rollback, evals, and release notes",
    )
    promotion_release_record.add_argument("--release-id", required=True, help="Release identifier to bind to the governance artifacts")
    promotion_release_record.add_argument("--promotion-decision", required=True, help="Passing promotion_decision.json")
    promotion_release_record.add_argument("--promotion-cards", required=True, help="Promotion cards directory or promotion_cards.json")
    promotion_release_record.add_argument("--promotion-alias-apply", required=True, help="Passing promotion_alias_apply.json receipt")
    promotion_release_record.add_argument("--rollback-metadata", required=True, help="Rollback metadata JSON used by the promotion decision")
    promotion_release_record.add_argument("--compare-gate", required=True, help="compare_gate.json used as eval movement evidence")
    promotion_release_record.add_argument("--release-notes", required=True, help="Release notes markdown or text file")
    promotion_release_record.add_argument("--promotion-policy", help="Promotion policy JSON expected to match the decision policy fingerprint")
    promotion_release_record.add_argument("--out", required=True, help="Write promotion_release_record.json")
    promotion_release_record.add_argument(
        "--metadata",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Attach metadata to the release record; may be repeated",
    )
    promotion_release_record.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the release record")
    promotion_release_record.set_defaults(func=cmd_promotion_release_record)

    promotion_decision = subparsers.add_parser(
        "promotion-decision",
        help="Evaluate top-level governance evidence before registry alias movement",
    )
    promotion_decision.add_argument("--candidate-id", required=True, help="Model id proposed for promotion")
    promotion_decision.add_argument("--champion-id", required=True, help="Current champion model id")
    promotion_decision.add_argument("--rollback-id", help="Model id to assign to rollback if promotion passes")
    promotion_decision.add_argument(
        "--candidate-class",
        default="candidate",
        choices=["base", "trace-only", "frontier", "champion", "candidate"],
        help="Source class of the promoted candidate",
    )
    promotion_decision.add_argument(
        "--champion-class",
        default="champion",
        choices=["base", "trace-only", "frontier", "champion", "candidate"],
        help="Source class of the incumbent champion",
    )
    promotion_decision.add_argument("--evidence-bundle", help="evidence_bundle.json for the candidate run")
    promotion_decision.add_argument("--eval-summary", help="eval_summary.json for the candidate evaluation")
    promotion_decision.add_argument(
        "--external-eval-result",
        action="append",
        default=[],
        help="Execution-backed external_eval_result.json for the candidate; may be repeated",
    )
    promotion_decision.add_argument("--promotion-ledger-gate", help="promotion_ledger_gate.json proving clean promotion history")
    promotion_decision.add_argument("--compare-gate", help="compare_gate.json proving candidate/champion eval movement")
    promotion_decision.add_argument("--trainer-launch-check", help="trainer_launch_check.json proving trainer handoff readiness")
    promotion_decision.add_argument("--model-registry-entry", help="model_registry_entry.json for the promoted candidate")
    promotion_decision.add_argument("--agentic-training-result", help="agentic_training_result.json for the promoted candidate")
    promotion_decision.add_argument("--cloud-training-completion-receipt", help="Imported cloud_training_completion_receipt.json bound to the promoted candidate")
    promotion_decision.add_argument("--model-card", help="Candidate model card")
    promotion_decision.add_argument("--dataset-card", help="Dataset card for the training/eval data")
    promotion_decision.add_argument("--rollback-metadata", help="Rollback metadata JSON naming the rollback target")
    promotion_decision.add_argument("--license-review", help="License review JSON with known license status")
    promotion_decision.add_argument("--redaction-check", help="Redaction gate JSON")
    promotion_decision.add_argument("--safety-gate", help="Safety gate JSON")
    promotion_decision.add_argument("--serving-profile", help="serving_profile.json proving candidate endpoint readiness")
    promotion_decision.add_argument("--serving-report", help="Serving smoke or readiness JSON")
    promotion_decision.add_argument("--promotion-policy", help="Promotion policy JSON declaring required artifacts and zero-tolerance limits")
    promotion_decision.add_argument("--out", help="Write promotion decision JSON to this path")
    promotion_decision.add_argument(
        "--metadata",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Attach metadata to the promotion decision; may be repeated",
    )
    promotion_decision.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the decision output")
    promotion_decision.set_defaults(func=cmd_promotion_decision)

    promotion_ledger = subparsers.add_parser(
        "promotion-ledger",
        help="Summarize decision gates across promotion attempts",
    )
    promotion_ledger.add_argument(
        "--decision-gate",
        action="append",
        required=True,
        help="decision_gate.json in chronological order; may be repeated",
    )
    promotion_ledger.add_argument("--out", help="Write promotion ledger JSON to this path")
    promotion_ledger.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the ledger output")
    promotion_ledger.set_defaults(func=cmd_promotion_ledger)

    promotion_archive = subparsers.add_parser(
        "promotion-archive",
        help="Copy promotion-history evidence into a portable hash-checked archive",
    )
    promotion_archive.add_argument("--promotion-ledger", required=True, help="Path to promotion_ledger.json")
    promotion_archive.add_argument("--promotion-ledger-gate", help="Optional path to promotion_ledger_gate.json")
    promotion_archive.add_argument("--decision-gate", action="append", default=[], help="Decision gate JSON to include; may be repeated")
    promotion_archive.add_argument(
        "--promotion-release-record",
        action="append",
        default=[],
        help="Promotion release record JSON to include as final publication evidence; may be repeated",
    )
    promotion_archive.add_argument("--out", required=True, help="Output directory for the portable promotion archive")
    promotion_archive.add_argument(
        "--require-self-contained",
        action="store_true",
        help="Return nonzero unless all referenced decision gates and source artifacts were copied",
    )
    promotion_archive.add_argument("--force", action="store_true", help="Replace an existing non-empty archive directory")
    promotion_archive.add_argument(
        "--preserve-paths",
        action="store_true",
        help="Allow absolute original paths in the archive manifest; use only for private local debugging",
    )
    promotion_archive.set_defaults(func=cmd_promotion_archive)



def register_governance_3(subparsers: argparse._SubParsersAction) -> None:
    draft = subparsers.add_parser("draft-scenario", help="Draft a scenario JSON file from an existing run or trace")
    draft_source = draft.add_mutually_exclusive_group(required=True)
    draft_source.add_argument("--trace", help="Trace artifact to normalize and use as source evidence")
    draft_source.add_argument("--run", help="Run directory containing normalized_trace.json")
    draft.add_argument("--format", default="auto", choices=TRACE_FORMAT_CHOICES)
    draft.add_argument("--out", required=True, help="Scenario JSON output path")
    draft.add_argument("--id", help="Scenario id; defaults to the source stem")
    draft.add_argument("--title", help="Scenario title; defaults from --id")
    draft.add_argument("--prompt", help="Scenario prompt; defaults to a TODO placeholder")
    draft.add_argument("--max-actions", type=_non_negative_int_arg, default=8, help="Maximum tool-result actions to draft")
    draft.add_argument("--preserve-paths", action="store_true", help="Write absolute trace paths instead of paths relative to --out")
    draft.set_defaults(func=cmd_draft_scenario)



def register_governance_4(subparsers: argparse._SubParsersAction) -> None:
    trainer_archive = subparsers.add_parser(
        "trainer-archive",
        help="Copy trainer handoff evidence into a portable hash-checked archive",
    )
    trainer_archive.add_argument("--preflight", required=True, help="trainer_preflight.json to include")
    trainer_archive.add_argument("--launch-check", required=True, help="trainer_launch_check.json to include")
    trainer_archive.add_argument("--out", required=True, help="Output directory for the portable trainer archive")
    trainer_archive.add_argument(
        "--require-self-contained",
        action="store_true",
        help="Return nonzero unless all preflight-referenced gates, validations, exports, and schema contracts were copied",
    )
    trainer_archive.add_argument("--force", action="store_true", help="Replace an existing non-empty trainer archive directory")
    trainer_archive.add_argument(
        "--preserve-paths",
        action="store_true",
        help="Allow absolute original paths in the archive manifest; use only for private local debugging",
    )
    trainer_archive.set_defaults(func=cmd_trainer_archive)

    trainer_archive_check = subparsers.add_parser(
        "trainer-archive-check",
        help="Validate a trainer archive plus local external trainer code without executing it",
    )
    trainer_archive_check.add_argument("--archive", required=True, help="trainer archive directory or trainer_archive.json to verify")
    trainer_archive_check.add_argument("--external-code-root", default=".", help="Directory containing external trainer code paths such as train.py")
    trainer_archive_check.add_argument("--out", required=True, help="Write trainer archive check JSON to this path")
    trainer_archive_check.add_argument("--strict", action="store_true", help="Treat archive validation warnings as consumer-readiness blockers")
    trainer_archive_check.add_argument(
        "--preserve-paths",
        action="store_true",
        help="Allow absolute local paths in trainer archive check output; use only for private local debugging",
    )
    trainer_archive_check.set_defaults(func=cmd_trainer_archive_check)

    trainer_consumer_plan = subparsers.add_parser(
        "trainer-consumer-plan",
        help="Build a side-effect-free execution plan for an external trainer wrapper",
    )
    trainer_consumer_plan.add_argument("--archive-check", required=True, help="trainer_archive_check.json to convert into a consumer plan")
    trainer_consumer_plan.add_argument("--out", required=True, help="Write trainer consumer plan JSON to this path")
    trainer_consumer_plan.add_argument("--strict", action="store_true", help="Treat archive-check validation warnings as plan blockers")
    trainer_consumer_plan.add_argument(
        "--preserve-paths",
        action="store_true",
        help="Allow absolute local paths in trainer consumer plan output; use only for private local debugging",
    )
    trainer_consumer_plan.set_defaults(func=cmd_trainer_consumer_plan)



def register_governance_5(subparsers: argparse._SubParsersAction) -> None:
    export_review = subparsers.add_parser("export-review", help="Export completed runs as a human review queue")
    export_review.add_argument("--runs", required=True, help="Directory containing Flight Recorder run subdirectories")
    export_review.add_argument("--out", required=True, help="Output directory for review queue artifacts")
    export_review.add_argument("--only-failed", action="store_true", help="Include only failed runs in the review queue")
    export_review.add_argument("--preserve-paths", action="store_true", help="Preserve public-safe relative paths; redact absolute local paths")
    export_review.set_defaults(func=cmd_export_review)

    apply_review = subparsers.add_parser("apply-review", help="Apply completed human labels to a review queue")
    apply_review.add_argument("--review-export", required=True, help="Directory containing export-review artifacts")
    apply_review.add_argument("--out", required=True, help="Output directory for reviewed label/trainer artifacts")
    apply_review.add_argument("--labels", help="Completed labels JSONL; defaults to <review-export>/label_template.jsonl")
    apply_review.add_argument(
        "--max-pairs-per-family",
        type=_non_negative_int_arg,
        default=0,
        help="Maximum reviewed preference pairs per task family; 0 means unlimited",
    )
    apply_review.add_argument("--preserve-paths", action="store_true", help="Preserve public-safe relative paths; redact absolute local paths")
    apply_review.set_defaults(func=cmd_apply_review)

    review_calibration = subparsers.add_parser(
        "review-calibration",
        help="Compare deterministic scorecards with human-reviewed labels",
    )
    review_calibration.add_argument("--reviewed-export", required=True, help="Directory containing apply-review artifacts")
    review_calibration.add_argument("--out", required=True, help="Write calibration report JSON to this path")
    review_calibration.add_argument("--min-agreement-rate", type=_rate_arg, help="Minimum human/scorecard agreement rate")
    review_calibration.add_argument("--max-disagreements", type=_non_negative_int_arg, help="Maximum allowed comparable disagreements")
    review_calibration.add_argument("--max-false-positives", type=_non_negative_int_arg, help="Maximum scorecard-pass/human-negative rows")
    review_calibration.add_argument("--max-false-negatives", type=_non_negative_int_arg, help="Maximum scorecard-fail/human-accept rows")
    review_calibration.add_argument("--min-comparable-labels", type=_non_negative_int_arg, help="Minimum labels excluding needs_review")
    review_calibration.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail reviewed-export validation on warnings as well as errors",
    )
    review_calibration.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip reviewed export structure and artifact-fingerprint validation before calibration",
    )
    review_calibration.add_argument("--preserve-paths", action="store_true", help="Preserve safe source path text in calibration output; unsafe absolute or traversal refs remain redacted")
    review_calibration.set_defaults(func=cmd_review_calibration)

