"""CLI commands for the evidence domain."""

from __future__ import annotations

from pathlib import Path
from ..schema_registry import SchemaRegistryError, check_schema_file, check_schema_jsonl_file, list_schema_records, load_schema, write_schema_bundle
from ..governance import PromotionDecisionError, apply_promotion_aliases, build_promotion_cards, build_promotion_decision, build_promotion_release_record, build_promotion_rollback_receipt
import argparse
from ..action_ledger import ActionLedgerError, build_action_ledger
from ..bundle import HARNESS_RUN_MANIFEST_SCHEMA_VERSION, HARNESS_RUN_RESULT_SCHEMA_VERSION, EvidenceBundleError, build_evidence_bundle
from ..evidence import EvidenceCoverageError, build_evidence_coverage
from ..improvement_ledger import ImprovementLedgerError, build_improvement_ledger
from ..improvement_plan import ImprovementPlanError, build_improvement_plan
from ..repair import RepairQueueError, build_repair_queue
from ..trace_observability import TraceObservabilityError, build_trace_observability
import json
from .shared import _metadata_options, _non_negative_float_arg, _non_negative_int_arg, _rate_arg, _write_json


def cmd_schemas(args: argparse.Namespace) -> int:
    if args.check and args.check_jsonl:
        raise SchemaRegistryError("--check cannot be combined with --check-jsonl")
    if args.check:
        if args.write_dir:
            raise SchemaRegistryError("--check cannot be combined with --write-dir")
        if len(args.name) > 1:
            raise SchemaRegistryError("--check accepts at most one --name")
        result = check_schema_file(args.check, args.name[0] if args.name else None)
        rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(rendered, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(rendered, end="")
        return 0 if result["passed"] else 1
    if args.check_jsonl:
        if args.write_dir:
            raise SchemaRegistryError("--check-jsonl cannot be combined with --write-dir")
        if len(args.name) > 1:
            raise SchemaRegistryError("--check-jsonl accepts at most one --name")
        result = check_schema_jsonl_file(args.check_jsonl, args.name[0] if args.name else None)
        rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(rendered, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(rendered, end="")
        return 0 if result["passed"] else 1
    if args.write_dir:
        written = write_schema_bundle(args.write_dir, args.name or None, force=args.force)
        print(f"wrote {len(written)} schema file(s) to {args.write_dir}")
        return 0
    if args.out:
        if len(args.name) != 1:
            raise SchemaRegistryError("--out requires exactly one --name")
        _write_json(Path(args.out), load_schema(args.name[0]))
        print(f"wrote {args.out}")
        return 0
    if args.name:
        if len(args.name) != 1:
            raise SchemaRegistryError("printing to stdout requires exactly one --name")
        print(json.dumps(load_schema(args.name[0]), indent=2, sort_keys=True))
        return 0

    for record in list_schema_records():
        print(f"{record['name']}\t{record['artifact_schema_version']}\t{record['filename']}")
    return 0


def cmd_evidence_coverage(args: argparse.Namespace) -> int:
    coverage = build_evidence_coverage(
        args.runs,
        preserve_paths=args.preserve_paths,
        min_failed_rule_evidence_rate=args.min_failed_rule_evidence_rate,
        min_critical_failed_rule_evidence_rate=args.min_critical_failed_rule_evidence_rate,
        min_event_evidence_refs=args.min_event_evidence_refs,
        max_failed_rules_without_evidence=args.max_failed_rules_without_evidence,
        max_critical_failed_rules_without_evidence=args.max_critical_failed_rules_without_evidence,
        require_rule_evidence=args.require_rule_evidence,
    )
    rendered = json.dumps(coverage, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if coverage["passed"] else 1


def cmd_trace_observability(args: argparse.Namespace) -> int:
    observability = build_trace_observability(
        args.runs,
        preserve_paths=args.preserve_paths,
        min_average_events=args.min_average_events,
        min_event_type_count=args.min_event_type_count,
        min_tool_or_api_run_rate=args.min_tool_or_api_run_rate,
        max_empty_final_answers=args.max_empty_final_answers,
        require_event_types=args.require_event_type,
    )
    rendered = json.dumps(observability, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if observability["passed"] else 1


def cmd_repair_queue(args: argparse.Namespace) -> int:
    queue = build_repair_queue(
        args.runs,
        preserve_paths=args.preserve_paths,
        only_critical=args.only_critical,
        output_path=args.out,
    )
    rendered = json.dumps(queue, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0


def cmd_evidence_bundle(args: argparse.Namespace) -> int:
    bundle = build_evidence_bundle(
        out_path=args.out,
        runs_dir=args.runs,
        suite_summary_path=args.suite_summary,
        scenario_quality_path=args.scenario_quality,
        evidence_coverage_path=args.evidence_coverage,
        trace_observability_path=args.trace_observability,
        repair_queue_path=args.repair_queue,
        validation_path=args.validation,
        eval_summary_path=args.eval_summary,
        training_export_dir=args.training_export,
        compare_export_dir=args.compare_export,
        review_export_dir=args.review_export,
        reviewed_export_dir=args.reviewed_export,
        review_calibration_path=args.review_calibration,
        live_smoke_summary_path=args.live_smoke_summary,
        serving_lifecycle_path=args.serving_lifecycle,
        trainer_preflight_path=args.trainer_preflight,
        trainer_launch_check_path=args.trainer_launch_check,
        trainer_archive_path=args.trainer_archive,
        trainer_archive_check_path=args.trainer_archive_check,
        trainer_consumer_plan_path=args.trainer_consumer_plan,
        agentic_training_flow_path=args.agentic_training_flow,
        trainer_wrapper_dry_run_path=args.trainer_wrapper_dry_run,
        agentic_training_result_path=args.agentic_training_result,
        harness_manifest_paths=args.harness_manifest,
        harness_result_paths=args.harness_result,
        gate_paths=args.gate,
        require_harness=args.require_harness,
        require_gate=args.require_gate,
        preserve_paths=args.preserve_paths,
    )
    _write_json(Path(args.out), bundle)
    print(f"wrote {args.out}")
    return 0 if bundle["passed"] else 1


def cmd_improvement_plan(args: argparse.Namespace) -> int:
    plan = build_improvement_plan(
        out_path=args.out,
        evidence_bundle_path=args.evidence_bundle,
        repair_queue_path=args.repair_queue,
        training_export_dir=args.training_export,
        runs_dir=args.runs,
        eval_summary_path=args.eval_summary,
        preserve_paths=args.preserve_paths,
    )
    _write_json(Path(args.out), plan)
    print(f"wrote {args.out}")
    return 0


def cmd_improvement_ledger(args: argparse.Namespace) -> int:
    ledger = build_improvement_ledger(
        args.plan,
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


def cmd_action_ledger(args: argparse.Namespace) -> int:
    ledger = build_action_ledger(
        args.bundle,
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


def cmd_promotion_cards(args: argparse.Namespace) -> int:
    cards = build_promotion_cards(
        out_dir=args.out,
        candidate_id=args.candidate_id,
        dataset_id=args.dataset_id,
        model_source=args.model_source,
        license_status=args.license_status,
        evidence_bundle_path=args.evidence_bundle,
        training_export_path=args.training_export,
        compare_gate_path=args.compare_gate,
        redaction_check_path=args.redaction_check,
        safety_gate_path=args.safety_gate,
        preserve_paths=args.preserve_paths,
        metadata=_metadata_options(args.metadata),
    )
    print(
        f"{'READY' if cards['passed'] else 'BLOCKED'} promotion-cards "
        f"checks={cards['check_count'] - cards['failed_check_count']}/{cards['check_count']} "
        f"out={args.out}"
    )
    return 0 if cards["passed"] else 1


def cmd_promotion_alias_apply(args: argparse.Namespace) -> int:
    receipt = apply_promotion_aliases(
        registry_path=args.registry,
        promotion_decision_path=args.promotion_decision,
        out_path=args.out,
        preserve_paths=args.preserve_paths,
        metadata=_metadata_options(args.metadata),
    )
    print(
        f"{'APPLIED' if receipt['passed'] else 'BLOCKED'} promotion-alias-apply "
        f"checks={receipt['check_count'] - receipt['failed_check_count']}/{receipt['check_count']} "
        f"out={args.out}"
    )
    return 0 if receipt["passed"] else 1


def cmd_promotion_rollback_receipt(args: argparse.Namespace) -> int:
    receipt = build_promotion_rollback_receipt(
        registry_path=args.registry,
        rollback_id=args.rollback_id,
        champion_id=args.champion_id,
        out_path=args.out,
        preserve_paths=args.preserve_paths,
        metadata=_metadata_options(args.metadata),
    )
    print(
        f"{'READY' if receipt['passed'] else 'BLOCKED'} promotion-rollback-receipt "
        f"checks={receipt['check_count'] - receipt['failed_check_count']}/{receipt['check_count']} "
        f"out={args.out}"
    )
    return 0 if receipt["passed"] else 1


def register_evidence_1(subparsers: argparse._SubParsersAction) -> None:
    schemas = subparsers.add_parser("schemas", help="List or export bundled JSON Schema contracts")
    schemas.add_argument("--name", action="append", default=[], help="Schema name, filename, schema version, or $id; may be repeated with --write-dir")
    schemas.add_argument("--check", help="Check one JSON artifact against a bundled schema; infers by schema_version unless --name is supplied")
    schemas.add_argument("--check-jsonl", help="Check every non-empty JSONL row against a bundled schema; infers by schema_version unless --name is supplied")
    schemas.add_argument("--out", help="Write exactly one selected schema, or a --check result, to this JSON file")
    schemas.add_argument("--write-dir", help="Write the selected schemas, or all schemas, plus catalog manifest to this directory")
    schemas.add_argument("--force", action="store_true", help="Overwrite existing files when using --write-dir")
    schemas.set_defaults(func=cmd_schemas)

    evidence_coverage = subparsers.add_parser(
        "evidence-coverage",
        help="Summarize structured evidence-ref coverage across completed runs",
    )
    evidence_coverage.add_argument("--runs", required=True, help="Directory containing Flight Recorder run subdirectories")
    evidence_coverage.add_argument("--out", help="Write evidence coverage JSON to this path")
    evidence_coverage.add_argument(
        "--min-failed-rule-evidence-rate",
        type=_rate_arg,
        help="Minimum fraction of failed rules that must have structured evidence refs",
    )
    evidence_coverage.add_argument(
        "--min-critical-failed-rule-evidence-rate",
        type=_rate_arg,
        help="Minimum fraction of failed critical rules that must have structured evidence refs",
    )
    evidence_coverage.add_argument(
        "--min-event-evidence-refs",
        type=_non_negative_int_arg,
        help="Minimum evidence refs pointing at trace events",
    )
    evidence_coverage.add_argument(
        "--max-failed-rules-without-evidence",
        type=_non_negative_int_arg,
        help="Maximum failed rules allowed to have no structured evidence refs",
    )
    evidence_coverage.add_argument(
        "--max-critical-failed-rules-without-evidence",
        type=_non_negative_int_arg,
        help="Maximum failed critical rules allowed to have no structured evidence refs",
    )
    evidence_coverage.add_argument(
        "--require-rule-evidence",
        action="append",
        default=[],
        help="Fail unless this rule id has at least one structured evidence ref across the suite",
    )
    evidence_coverage.add_argument("--preserve-paths", action="store_true", help="Allow absolute run paths in coverage output")
    evidence_coverage.set_defaults(func=cmd_evidence_coverage)

    trace_observability = subparsers.add_parser(
        "trace-observability",
        help="Summarize whether completed runs contain enough trace signal",
    )
    trace_observability.add_argument("--runs", required=True, help="Directory containing Flight Recorder run subdirectories")
    trace_observability.add_argument("--out", help="Write trace observability JSON to this path")
    trace_observability.add_argument("--min-average-events", type=_non_negative_float_arg, help="Minimum average normalized trace events per run")
    trace_observability.add_argument("--min-event-type-count", type=_non_negative_int_arg, help="Minimum distinct event types across the suite")
    trace_observability.add_argument("--min-tool-or-api-run-rate", type=_rate_arg, help="Minimum fraction of runs with tool or API events")
    trace_observability.add_argument("--max-empty-final-answers", type=_non_negative_int_arg, help="Maximum runs allowed to have empty final answers")
    trace_observability.add_argument("--require-event-type", action="append", default=[], help="Fail unless this normalized event type appears")
    trace_observability.add_argument("--preserve-paths", action="store_true", help="Allow absolute run paths in observability output")
    trace_observability.set_defaults(func=cmd_trace_observability)

    repair_queue = subparsers.add_parser(
        "repair-queue",
        help="Export failed scorecard rules as deterministic repair tasks",
    )
    repair_queue.add_argument("--runs", required=True, help="Directory containing Flight Recorder run subdirectories")
    repair_queue.add_argument("--out", help="Write repair queue JSON to this path")
    repair_queue.add_argument("--only-critical", action="store_true", help="Include only failed rules marked critical")
    repair_queue.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in repair queue output")
    repair_queue.set_defaults(func=cmd_repair_queue)

    evidence_bundle = subparsers.add_parser(
        "evidence-bundle",
        help="Summarize a complete evidence handoff bundle and readiness checks",
    )
    evidence_bundle.add_argument("--out", required=True, help="Write evidence bundle summary JSON to this path")
    evidence_bundle.add_argument("--runs", help="Runs directory included in the handoff")
    evidence_bundle.add_argument("--suite-summary", help="run-suite suite_summary.json included in the handoff")
    evidence_bundle.add_argument("--scenario-quality", help="scenario_quality.json included in the handoff")
    evidence_bundle.add_argument("--evidence-coverage", help="evidence_coverage.json included in the handoff")
    evidence_bundle.add_argument("--trace-observability", help="trace_observability.json included in the handoff")
    evidence_bundle.add_argument("--repair-queue", help="repair_queue.json included in the handoff")
    evidence_bundle.add_argument("--validation", help="validation.json included in the handoff")
    evidence_bundle.add_argument("--eval-summary", help="eval_summary.json included in the handoff")
    evidence_bundle.add_argument("--training-export", help="export-rl directory included in the handoff")
    evidence_bundle.add_argument("--compare-export", help="export-compare-rl directory included in the handoff")
    evidence_bundle.add_argument("--review-export", help="export-review directory included in the handoff")
    evidence_bundle.add_argument("--reviewed-export", help="apply-review directory included in the handoff")
    evidence_bundle.add_argument("--review-calibration", help="review_calibration.json included in the handoff")
    evidence_bundle.add_argument("--live-smoke-summary", help="live_smoke_summary.json included in the handoff")
    evidence_bundle.add_argument("--serving-lifecycle", help="serving_lifecycle.json included in the handoff")
    evidence_bundle.add_argument("--trainer-preflight", help="trainer_preflight.json included in the handoff")
    evidence_bundle.add_argument("--trainer-launch-check", help="trainer_launch_check.json included in the handoff")
    evidence_bundle.add_argument("--trainer-archive", help="Trainer archive directory or trainer_archive.json included in the handoff")
    evidence_bundle.add_argument("--trainer-archive-check", help="trainer_archive_check.json included in the handoff")
    evidence_bundle.add_argument("--trainer-consumer-plan", help="trainer_consumer_plan.json included in the handoff")
    evidence_bundle.add_argument("--agentic-training-flow", help="agentic_training_flow.json included in the handoff")
    evidence_bundle.add_argument("--trainer-wrapper-dry-run", help="trainer_wrapper_dry_run.json included in the handoff")
    evidence_bundle.add_argument("--agentic-training-result", help="agentic_training_result.json included in the handoff")
    evidence_bundle.add_argument("--harness-manifest", action="append", default=[], help="harness_manifest.json included in the handoff; may be repeated")
    evidence_bundle.add_argument("--harness-result", action="append", default=[], help="harness_result.json included in the handoff; may be repeated")
    evidence_bundle.add_argument("--gate", action="append", default=[], help="Gate result JSON to require; may be repeated")
    evidence_bundle.add_argument("--require-harness", action="store_true", help="Block unless at least one matched harness manifest/result pair is included")
    evidence_bundle.add_argument("--require-gate", action="store_true", help="Block unless at least one gate summary is included")
    evidence_bundle.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the bundle summary")
    evidence_bundle.set_defaults(func=cmd_evidence_bundle)

    improvement_plan = subparsers.add_parser(
        "improvement-plan",
        help="Join bundle actions, repair items, curriculum priorities, and run digests into a next-iteration plan",
    )
    improvement_plan.add_argument("--evidence-bundle", required=True, help="evidence_bundle.json to summarize")
    improvement_plan.add_argument("--repair-queue", help="repair_queue.json with concrete failed-rule repair items")
    improvement_plan.add_argument("--training-export", help="export-rl directory containing curriculum.json")
    improvement_plan.add_argument("--runs", help="Runs directory containing per-run run_digest.json files")
    improvement_plan.add_argument("--eval-summary", help="eval_summary.json with eval repair/curriculum work items")
    improvement_plan.add_argument("--out", required=True, help="Write improvement plan JSON to this path")
    improvement_plan.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the plan output")
    improvement_plan.set_defaults(func=cmd_improvement_plan)

    improvement_ledger = subparsers.add_parser(
        "improvement-ledger",
        help="Summarize improvement-plan work items across improvement iterations",
    )
    improvement_ledger.add_argument("--plan", action="append", required=True, help="Improvement plan JSON in chronological order; may be repeated")
    improvement_ledger.add_argument("--out", help="Write improvement ledger JSON to this path")
    improvement_ledger.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the ledger output")
    improvement_ledger.set_defaults(func=cmd_improvement_ledger)



def register_evidence_2(subparsers: argparse._SubParsersAction) -> None:
    action_ledger = subparsers.add_parser(
        "action-ledger",
        help="Summarize evidence-bundle next actions across improvement iterations",
    )
    action_ledger.add_argument("--bundle", action="append", required=True, help="Evidence bundle JSON in chronological order; may be repeated")
    action_ledger.add_argument("--out", help="Write action ledger JSON to this path")
    action_ledger.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the ledger output")
    action_ledger.set_defaults(func=cmd_action_ledger)

    promotion_cards = subparsers.add_parser(
        "promotion-cards",
        help="Generate model and dataset cards for promotion governance",
    )
    promotion_cards.add_argument("--candidate-id", required=True, help="Model id proposed for promotion")
    promotion_cards.add_argument("--dataset-id", required=True, help="Dataset id or version represented by the dataset card")
    promotion_cards.add_argument("--model-source", required=True, help="Source model or training output summarized by the model card")
    promotion_cards.add_argument(
        "--license-status",
        default="unknown",
        help="Reviewed license status; unknown/unreviewed/missing blocks promotion-card readiness",
    )
    promotion_cards.add_argument("--evidence-bundle", help="evidence_bundle.json used as governance evidence")
    promotion_cards.add_argument("--training-export", help="Training export directory used to produce the dataset card")
    promotion_cards.add_argument("--compare-gate", help="compare_gate.json used as eval movement evidence")
    promotion_cards.add_argument("--redaction-check", help="Redaction gate JSON")
    promotion_cards.add_argument("--safety-gate", help="Safety gate JSON")
    promotion_cards.add_argument("--out", required=True, help="Output directory for MODEL_CARD.md, DATASET_CARD.md, and promotion_cards.json")
    promotion_cards.add_argument(
        "--metadata",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Attach metadata to the promotion cards manifest; may be repeated",
    )
    promotion_cards.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in card manifests")
    promotion_cards.set_defaults(func=cmd_promotion_cards)

    promotion_alias_apply = subparsers.add_parser(
        "promotion-alias-apply",
        help="Apply registry aliases from a validated passing promotion decision",
    )
    promotion_alias_apply.add_argument("--registry", required=True, help="Model registry JSON to mutate only if all checks pass")
    promotion_alias_apply.add_argument("--promotion-decision", required=True, help="Passing promotion_decision.json authorizing aliases")
    promotion_alias_apply.add_argument("--out", required=True, help="Write promotion alias application receipt JSON")
    promotion_alias_apply.add_argument(
        "--metadata",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Attach metadata to the alias-apply receipt; may be repeated",
    )
    promotion_alias_apply.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the receipt")
    promotion_alias_apply.set_defaults(func=cmd_promotion_alias_apply)

    promotion_rollback_receipt = subparsers.add_parser(
        "promotion-rollback-receipt",
        help="Prove the rollback target is registered before promotion",
    )
    promotion_rollback_receipt.add_argument("--registry", required=True, help="Model registry JSON used to verify the rollback target")
    promotion_rollback_receipt.add_argument("--rollback-id", required=True, help="Model id to use as rollback if the candidate is promoted")
    promotion_rollback_receipt.add_argument(
        "--champion-id",
        help="Expected current champion model id; defaults to the registry champion alias",
    )
    promotion_rollback_receipt.add_argument("--out", required=True, help="Write rollback receipt JSON")
    promotion_rollback_receipt.add_argument(
        "--metadata",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Attach metadata to the rollback receipt; may be repeated",
    )
    promotion_rollback_receipt.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in the receipt")
    promotion_rollback_receipt.set_defaults(func=cmd_promotion_rollback_receipt)

