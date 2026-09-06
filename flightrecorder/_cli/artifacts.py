"""CLI commands for the artifacts domain."""

from __future__ import annotations

from pathlib import Path
import argparse
from ..scenario_quality import build_scenario_quality
from ..artifacts import ArtifactError, build_suite_trend, compare_scorecards, compare_suites, write_compare_report, write_junit, write_markdown_summary, write_suite_compare_report, write_suite_trend_report
from ..scenario_check import check_scenarios, discover_scenarios
import json
from ..validation import EVAL_SUITE_MANIFEST_SCHEMA_VERSION, VALIDATION_SCHEMA_VERSION, validate_artifacts, validate_trainer_preflight
from ..report import write_index, write_report
from .replay_core import _audit_runs, _read_scorecard_ref
from .shared import _non_negative_int_arg, _rate_arg, _score_arg, _write_json


def cmd_index(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs)
    run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    write_index(run_dirs, args.out, artifacts_dir=runs_dir)
    print(f"wrote {args.out}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    summary = _audit_runs(Path(args.runs), args.forbid_text)
    rendered = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    if args.fail_on_leak and summary["leaks"]:
        return 1
    if args.fail_on_failed and summary["failed"] > 0:
        return 1
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    baseline, baseline_label = _read_scorecard_ref(Path(args.baseline))
    candidate, candidate_label = _read_scorecard_ref(Path(args.candidate))
    comparison = compare_scorecards(
        baseline,
        candidate,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
    )
    _write_json(Path(args.out), comparison)
    if args.html_out:
        write_compare_report(comparison, args.html_out)
    print(f"{'REGRESSION' if comparison['regressed'] else 'NO REGRESSION'} score_delta={comparison['score_delta']} wrote {args.out}")
    return 1 if args.fail_on_regression and comparison["regressed"] else 0


def cmd_compare_suite(args: argparse.Namespace) -> int:
    comparison = compare_suites(
        args.baseline,
        args.candidate,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        contract_scope=args.contract_scope,
    )
    _write_json(Path(args.out), comparison)
    if args.html_out:
        write_suite_compare_report(comparison, args.html_out)
    aggregate = comparison["aggregate"]
    print(
        f"{'REGRESSION' if comparison['regressed'] else 'NO REGRESSION'} "
        f"paired={aggregate['paired_count']} avg_score_delta={aggregate['avg_score_delta']} wrote {args.out}"
    )
    if args.fail_on_contract_drift and aggregate.get("contract_drift_count", 0) > 0:
        return 1
    if args.fail_on_unverified_contracts and aggregate.get("unverified_contract_count", 0) > 0:
        return 1
    return 1 if args.fail_on_regression and comparison["regressed"] else 0


def cmd_trend_suite(args: argparse.Namespace) -> int:
    trend = build_suite_trend(args.suite_summary)
    _write_json(Path(args.out), trend)
    if args.html_out:
        write_suite_trend_report(trend, args.html_out)
    print(f"TREND points={trend['point_count']} summary={trend['summary']} wrote {args.out}")
    return 0


def cmd_observer_template(args: argparse.Namespace) -> int:
    rendered = OBSERVER_TEMPLATE
    if args.out:
        path = Path(args.out)
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite existing file without --force: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(rendered, end="")
    return 0


def cmd_check_scenarios(args: argparse.Namespace) -> int:
    summary = check_scenarios(
        args.scenarios,
        pattern=args.pattern,
        recursive=args.recursive,
        require_traces=args.require_traces,
        strict=args.strict,
        preserve_paths=args.preserve_paths,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if summary["passed"] else 1


def cmd_scenario_quality(args: argparse.Namespace) -> int:
    summary = build_scenario_quality(
        args.scenarios,
        pattern=args.pattern,
        recursive=args.recursive,
        require_traces=args.require_traces,
        preserve_paths=args.preserve_paths,
        min_average_score=args.min_average_score,
        min_scenario_score=args.min_scenario_score,
        min_observable_rate=args.min_observable_rate,
        max_weak_scenarios=args.max_weak_scenarios,
        max_final_only_scenarios=args.max_final_only_scenarios,
        max_missing_traces=args.max_missing_traces,
        require_task_families=args.require_task_family,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if summary["passed"] else 1


VALIDATE_APPEND_OPTIONS = (
    ('--run', 'run', 'run_dirs', 'Validate one run directory; may be repeated'),
    ('--reviewed-gate', 'reviewed_gate', 'reviewed_gate_paths', 'Validate one reviewed_gate.json; may be repeated'),
    ('--evidence-coverage', 'evidence_coverage', 'evidence_coverage_paths', 'Validate one evidence_coverage.json; may be repeated'),
    ('--evidence-bundle', 'evidence_bundle', 'evidence_bundle_paths', 'Validate one evidence_bundle.json; may be repeated'),
    ('--improvement-plan', 'improvement_plan', 'improvement_plan_paths', 'Validate one improvement_plan.json; may be repeated'),
    ('--improvement-ledger', 'improvement_ledger', 'improvement_ledger_paths', 'Validate one improvement_ledger.json; may be repeated'),
    ('--improvement-ledger-gate', 'improvement_ledger_gate', 'improvement_ledger_gate_paths', 'Validate one improvement_ledger_gate.json; may be repeated'),
    ('--action-ledger', 'action_ledger', 'action_ledger_paths', 'Validate one action_ledger.json; may be repeated'),
    ('--action-ledger-gate', 'action_ledger_gate', 'action_ledger_gate_paths', 'Validate one action_ledger_gate.json; may be repeated'),
    ('--decision-gate', 'decision_gate', 'decision_gate_paths', 'Validate one decision_gate.json; may be repeated'),
    ('--promotion-cards', 'promotion_cards', 'promotion_cards_paths', 'Validate one promotion-cards directory or manifest; may be repeated'),
    ('--promotion-decision', 'promotion_decision', 'promotion_decision_paths', 'Validate one promotion_decision.json; may be repeated'),
    ('--promotion-alias-apply', 'promotion_alias_apply', 'promotion_alias_apply_paths', 'Validate one promotion_alias_apply.json receipt; may be repeated'),
    ('--promotion-rollback-receipt', 'promotion_rollback_receipt', 'promotion_rollback_receipt_paths', 'Validate one promotion rollback receipt; may be repeated'),
    ('--promotion-release-record', 'promotion_release_record', 'promotion_release_record_paths', 'Validate one promotion_release_record.json; may be repeated'),
    ('--promotion-policy', 'promotion_policy', 'promotion_policy_paths', 'Validate one promotion_policy.json; may be repeated'),
    ('--promotion-ledger', 'promotion_ledger', 'promotion_ledger_paths', 'Validate one promotion_ledger.json; may be repeated'),
    ('--promotion-ledger-gate', 'promotion_ledger_gate', 'promotion_ledger_gate_paths', 'Validate one promotion_ledger_gate.json; may be repeated'),
    ('--promotion-archive', 'promotion_archive', 'promotion_archive_paths', 'Validate one promotion archive directory or manifest; may be repeated'),
    ('--trainer-preflight', 'trainer_preflight', 'trainer_preflight_paths', 'Validate one trainer_preflight.json; may be repeated'),
    ('--trainer-launch-check', 'trainer_launch_check', 'trainer_launch_check_paths', 'Validate one trainer_launch_check.json; may be repeated'),
    ('--trainer-archive', 'trainer_archive', 'trainer_archive_paths', 'Validate one trainer archive directory or manifest; may be repeated'),
    ('--trainer-archive-check', 'trainer_archive_check', 'trainer_archive_check_paths', 'Validate one trainer_archive_check.json; may be repeated'),
    ('--trainer-consumer-plan', 'trainer_consumer_plan', 'trainer_consumer_plan_paths', 'Validate one trainer_consumer_plan.json; may be repeated'),
    ('--trainer-wrapper-dry-run', 'trainer_wrapper_dry_run', 'trainer_wrapper_dry_run_paths', 'Validate one trainer_wrapper_dry_run.json; may be repeated'),
    ('--model-scout-manifest', 'model_scout_manifest', 'model_scout_manifest_paths', 'Validate one model_scout_manifest.json; may be repeated'),
    ('--model-candidate', 'model_candidate', 'model_candidate_paths', 'Validate one model candidate JSON; may be repeated'),
    ('--model-compatibility-report', 'model_compatibility_report', 'model_compatibility_report_paths', 'Validate one model compatibility report JSON; may be repeated'),
    ('--model-serving-probe-receipt', 'model_serving_probe_receipt', 'model_serving_probe_receipt_paths', 'Validate one model serving-probe receipt JSON; may be repeated'),
    ('--model-adapter-manifest', 'model_adapter_manifest', 'model_adapter_manifest_paths', 'Validate one model adapter manifest JSON; may be repeated'),
    ('--model-registry-entry', 'model_registry_entry', 'model_registry_entry_paths', 'Validate one model registry entry JSON; may be repeated'),
    ('--model-registry', 'model_registry', 'model_registry_paths', 'Validate one model registry JSON; may be repeated'),
    ('--training-plan', 'training_plan', 'training_plan_paths', 'Validate one dry-run training plan JSON; may be repeated'),
    ('--agentic-training-plan', 'agentic_training_plan', 'agentic_training_plan_paths', 'Validate one agentic_training_plan.json dry-run trainer handoff contract; may be repeated'),
    ('--agentic-training-runtime-preflight', 'agentic_training_runtime_preflight', 'agentic_training_runtime_preflight_paths', 'Validate one agentic_training_runtime_preflight.json receipt; may be repeated'),
    ('--agentic-training-flow', 'agentic_training_flow', 'agentic_training_flow_paths', 'Validate one agentic_training_flow.json delegated trainer-flow receipt; may be repeated'),
    ('--agentic-training-result', 'agentic_training_result', 'agentic_training_result_paths', 'Validate one agentic_training_result.json receipt; may be repeated'),
    ('--agentic-loop-plan', 'agentic_loop_plan', 'agentic_training_loop_plan_paths', 'Validate one agentic_training_loop_plan.json contract; may be repeated'),
    ('--agentic-loop-ledger', 'agentic_loop_ledger', 'agentic_loop_ledger_paths', 'Validate one agentic_loop_ledger.json; may be repeated'),
    ('--agentic-loop-governance-receipt', 'agentic_loop_governance_receipt', 'agentic_loop_governance_receipt_paths', 'Validate one agentic_loop_governance_receipt.json; may be repeated'),
    ('--next-iteration-schedule', 'next_iteration_schedule', 'next_iteration_schedule_paths', 'Validate one next_iteration_schedule.json; may be repeated'),
    ('--cloud-training-provider-registry', 'cloud_training_provider_registry', 'cloud_training_provider_registry_paths', 'Validate one cloud training provider registry'),
    ('--cloud-training-preflight', 'cloud_training_preflight', 'cloud_training_preflight_paths', 'Validate one cloud training preflight'),
    ('--cloud-training-artifact-manifest', 'cloud_training_artifact_manifest', 'cloud_training_artifact_manifest_paths', 'Validate one cloud training artifact manifest'),
    ('--cloud-training-launch-plan', 'cloud_training_launch_plan', 'cloud_training_launch_plan_paths', 'Validate one cloud training launch plan'),
    ('--cloud-training-launch-receipt', 'cloud_training_launch_receipt', 'cloud_training_launch_receipt_paths', 'Validate one cloud training launch receipt'),
    ('--cloud-training-status-receipt', 'cloud_training_status_receipt', 'cloud_training_status_receipt_paths', 'Validate one cloud training status receipt'),
    ('--cloud-training-completion-receipt', 'cloud_training_completion_receipt', 'cloud_training_completion_receipt_paths', 'Validate one imported cloud training completion receipt'),
    ('--agentic-rollout-plan', 'agentic_rollout_plan', 'agentic_rollout_plan_paths', 'Validate one agentic rollout generation plan'),
    ('--agentic-rollout-receipt', 'agentic_rollout_receipt', 'agentic_rollout_receipt_paths', 'Validate one agentic mock rollout receipt'),
    ('--rejection-sampling-gate', 'rejection_sampling_gate', 'rejection_sampling_gate_paths', 'Validate one rejection sampling admission gate'),
    ('--dataset-curation-receipt', 'dataset_curation_receipt', 'dataset_curation_receipt_paths', 'Validate one dataset curation receipt'),
    ('--rubric-spec', 'rubric_spec', 'rubric_spec_paths', 'Validate one rubric_spec artifact'),
    ('--model-grader-dry-run', 'model_grader_dry_run', 'model_grader_dry_run_paths', 'Validate one model_grader_dry_run receipt'),
    ('--model-grader-disagreement-queue', 'model_grader_disagreement_queue', 'model_grader_disagreement_queue_paths', 'Validate one model_grader_disagreement_queue artifact'),
    ('--model-grader-override-receipt', 'model_grader_override_receipt', 'model_grader_override_receipt_paths', 'Validate one model_grader_override_receipt artifact'),
    ('--model-grader-gate', 'model_grader_gate', 'model_grader_gate_paths', 'Validate one model_grader_gate artifact'),
    ('--repair-queue', 'repair_queue', 'repair_queue_paths', 'Validate one repair_queue.json; may be repeated'),
    ('--replay-bundle', 'replay_bundle', 'replay_bundle_paths', 'Validate one replay-bundle directory or replay_bundle.json; may be repeated'),
    ('--trace-observability', 'trace_observability', 'trace_observability_paths', 'Validate one trace_observability.json; may be repeated'),
    ('--review-calibration', 'review_calibration', 'review_calibration_paths', 'Validate one review_calibration.json; may be repeated'),
    ('--scenario-check', 'scenario_check', 'scenario_check_paths', 'Validate one scenario_check.json; may be repeated'),
    ('--scenario-quality', 'scenario_quality', 'scenario_quality_paths', 'Validate one scenario_quality.json; may be repeated'),
    ('--suite-summary', 'suite_summary', 'suite_summary_paths', 'Validate one run-suite suite_summary.json; may be repeated'),
    ('--suite-trend', 'suite_trend', 'suite_trend_paths', 'Validate one trend-suite suite_trend.json; may be repeated'),
    ('--eval-suite-manifest', 'eval_suite_manifest', 'eval_suite_manifest_paths', 'Validate one hfr.eval_suite_manifest.v1 JSON file; may be repeated'),
    ('--state-snapshot', 'state_snapshot', 'state_snapshot_paths', 'Validate one hfr.state_snapshot.v1 JSON file; may be repeated'),
    ('--state-diff', 'state_diff', 'state_diff_paths', 'Validate one hfr.state_diff.v1 JSON file; may be repeated'),
    ('--run-digest', 'run_digest', 'run_digest_paths', 'Validate one hfr.run_digest.v1 JSON file; may be repeated'),
    ('--harness-manifest', 'harness_manifest', 'harness_manifest_paths', 'Validate one harness_manifest.json; may be repeated'),
    ('--harness-result', 'harness_result', 'harness_result_paths', 'Validate one harness_result.json; may be repeated'),
    ('--harness-replay-result', 'harness_replay_result', 'harness_replay_result_paths', 'Validate one harness_replay_result.json; may be repeated'),
    ('--harness-suite-result', 'harness_suite_result', 'harness_suite_result_paths', 'Validate one harness_suite_result.json; may be repeated'),
    ('--live-smoke-summary', 'live_smoke_summary', 'live_smoke_summary_paths', 'Validate one live_smoke_summary.json; may be repeated'),
    ('--eval-summary', 'eval_summary', 'eval_summary_paths', 'Validate one hfr.eval_summary.v1 JSON file; may be repeated'),
    ('--tool-capability-selection', 'tool_capability_selection', 'tool_capability_selection_paths', 'Validate one hfr.tool_capability_selection.v1 JSON file; may be repeated'),
    ('--adapter-route-decision', 'adapter_route_decision', 'adapter_route_decision_paths', 'Validate one hfr.adapter_route_decision.v1 JSON file; may be repeated'),
    ('--external-eval-plan', 'external_eval_plan', 'external_eval_plan_paths', 'Validate one hfr.external_eval_adapters.v1 JSON file; may be repeated'),
    ('--external-eval-receipt', 'external_eval_receipt', 'external_eval_receipt_paths', 'Validate one hfr.external_eval_receipt.v1 JSON file; may be repeated'),
    ('--external-eval-result', 'external_eval_result', 'external_eval_result_paths', 'Validate one hfr.external_eval_result.v1 JSON file; may be repeated'),
    ('--heldout-manifest', 'heldout_manifest', 'heldout_manifest_paths', 'Validate one hfr.heldout_scenario_manifest.v1 JSON file; may be repeated'),
    ('--serving-profile', 'serving_profile', 'serving_profile_paths', 'Validate one serving_profile.json; may be repeated'),
    ('--serving-compatibility-report', 'serving_compatibility_report', 'serving_compatibility_report_paths', 'Validate one compatibility_report.json; may be repeated'),
    ('--serving-endpoint-check', 'serving_endpoint_check', 'serving_endpoint_check_paths', 'Validate one serving_check.json; may be repeated'),
    ('--serving-lifecycle', 'serving_lifecycle', 'serving_lifecycle_paths', 'Validate one serving_lifecycle.json; may be repeated'),
    ('--serving-demo-run', 'serving_demo_run', 'serving_demo_run_paths', 'Validate one serving demo_run.json; may be repeated'),
)


def cmd_validate(args: argparse.Namespace) -> int:
    repeated_inputs = {keyword: getattr(args, dest) for _, dest, keyword, _ in VALIDATE_APPEND_OPTIONS}
    summary = validate_artifacts(
        runs_dir=args.runs,
        training_export_dir=args.training_export,
        compare_export_dir=args.compare_export,
        review_export_dir=args.review_export,
        reviewed_export_dir=args.reviewed_export,
        strict=args.strict,
        **repeated_inputs,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if summary["passed"] else 1


OBSERVER_TEMPLATE = '''"""Read-only Flight Recorder observer plugin.

Install `flight-recorder`, set FLIGHT_RECORDER_OUTPUT_DIR to a
restricted directory, then load this plugin through Hermes' plugin mechanism.
The collector records observer-hook JSONL only; it does not block or mutate
Hermes tools, prompts, memory, or model requests.
"""

from flightrecorder.hermes_plugin import register as register_flight_recorder


def register(ctx):
    return register_flight_recorder(ctx)
'''


def register_artifacts_1(subparsers: argparse._SubParsersAction) -> None:
    index = subparsers.add_parser("index", help="Build an index for generated run reports")
    index.add_argument("--runs", required=True)
    index.add_argument("--out", required=True)
    index.set_defaults(func=cmd_index)

    audit = subparsers.add_parser("audit", help="Summarize run outputs and scan generated artifacts")
    audit.add_argument("--runs", required=True)
    audit.add_argument("--out")
    audit.add_argument("--forbid-text", action="append", default=[], help="Literal text that must not appear in generated artifacts")
    audit.add_argument("--fail-on-leak", action="store_true", help="Exit nonzero if forbidden text is found")
    audit.add_argument("--fail-on-failed", action="store_true", help="Exit nonzero if any scorecard failed")
    audit.set_defaults(func=cmd_audit)

    compare = subparsers.add_parser("compare", help="Compare two scorecards or run directories")
    compare.add_argument("--baseline", required=True, help="Baseline scorecard.json or run directory")
    compare.add_argument("--candidate", required=True, help="Candidate scorecard.json or run directory")
    compare.add_argument("--out", required=True, help="Comparison JSON output path")
    compare.add_argument("--html-out", help="Optional static HTML comparison report")
    compare.add_argument("--fail-on-regression", action="store_true", help="Exit nonzero when the candidate regresses")
    compare.set_defaults(func=cmd_compare)

    compare_suite = subparsers.add_parser("compare-suite", help="Compare two directories of run scorecards")
    compare_suite.add_argument("--baseline", required=True, help="Baseline runs directory")
    compare_suite.add_argument("--candidate", required=True, help="Candidate runs directory")
    compare_suite.add_argument("--out", required=True, help="Suite comparison JSON output path")
    compare_suite.add_argument("--html-out", help="Optional static HTML suite comparison report")
    compare_suite.add_argument("--baseline-label", help="Human-readable baseline label")
    compare_suite.add_argument("--candidate-label", help="Human-readable candidate label")
    compare_suite.add_argument(
        "--contract-scope",
        default="scenario",
        choices=["scenario", "scenario-and-trace"],
        help="Fingerprint contract to compare: scenario for live improvement loops, scenario-and-trace for strict fixture replay",
    )
    compare_suite.add_argument("--fail-on-regression", action="store_true", help="Exit nonzero when the candidate suite regresses")
    compare_suite.add_argument("--fail-on-contract-drift", action="store_true", help="Exit nonzero when paired scenarios drift under --contract-scope")
    compare_suite.add_argument("--fail-on-unverified-contracts", action="store_true", help="Exit nonzero when paired scenarios are missing lineage fingerprints")
    compare_suite.set_defaults(func=cmd_compare_suite)

    trend_suite = subparsers.add_parser("trend-suite", help="Build a longitudinal trend over run-suite summaries")
    trend_suite.add_argument(
        "--suite-summary",
        action="append",
        required=True,
        help="Path to a suite_summary.json in chronological/order-of-comparison order; may be repeated",
    )
    trend_suite.add_argument("--out", required=True, help="Suite trend JSON output path")
    trend_suite.add_argument("--html-out", help="Optional static HTML suite trend report")
    trend_suite.set_defaults(func=cmd_trend_suite)

    check = subparsers.add_parser("check-scenarios", help="Validate scenario definitions before running them")
    check.add_argument("--scenarios", required=True, help="Directory containing scenario files")
    check.add_argument("--pattern", default="*.json", help="Scenario filename glob relative to --scenarios")
    check.add_argument("--recursive", action="store_true", help="Discover scenarios recursively with --pattern")
    check.add_argument("--out", help="Write scenario-check summary JSON to this path")
    check.add_argument("--require-traces", action="store_true", help="Fail when scenarios do not resolve to existing trace files")
    check.add_argument("--strict", action="store_true", help="Treat warnings as failure")
    check.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in generated check output")
    check.set_defaults(func=cmd_check_scenarios)

    quality = subparsers.add_parser("scenario-quality", help="Summarize and gate scenario contract quality")
    quality.add_argument("--scenarios", required=True, help="Directory containing scenario files")
    quality.add_argument("--pattern", default="*.json", help="Scenario filename glob relative to --scenarios")
    quality.add_argument("--recursive", action="store_true", help="Discover scenarios recursively with --pattern")
    quality.add_argument("--require-traces", action="store_true", help="Treat missing trace paths/files as scenario errors")
    quality.add_argument("--out", help="Write scenario-quality summary JSON to this path")
    quality.add_argument("--min-average-score", type=_score_arg, help="Minimum average scenario contract score")
    quality.add_argument("--min-scenario-score", type=_score_arg, help="Minimum allowed score for the weakest valid scenario")
    quality.add_argument("--min-observable-rate", type=_rate_arg, help="Minimum fraction of scenarios with observable assertions")
    quality.add_argument("--max-weak-scenarios", type=_non_negative_int_arg, help="Maximum allowed weak scenario contracts")
    quality.add_argument("--max-final-only-scenarios", type=_non_negative_int_arg, help="Maximum allowed final-answer-only contracts")
    quality.add_argument("--max-missing-traces", type=_non_negative_int_arg, help="Maximum valid scenarios with missing trace files")
    quality.add_argument("--require-task-family", action="append", default=[], help="Fail unless this derived task family is present")
    quality.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in generated quality output")
    quality.set_defaults(func=cmd_scenario_quality)

    validate = subparsers.add_parser("validate", help="Validate generated run and training artifacts")
    first_flag, first_dest, _, first_help = VALIDATE_APPEND_OPTIONS[0]
    validate.add_argument(first_flag, dest=first_dest, action="append", default=[], help=first_help)
    validate.add_argument("--runs", help="Validate every completed run directory inside this runs directory")
    validate.add_argument("--training-export", help="Validate an export-rl output directory")
    validate.add_argument("--compare-export", help="Validate an export-compare-rl output directory")
    validate.add_argument("--review-export", help="Validate an export-review output directory")
    validate.add_argument("--reviewed-export", help="Validate an apply-review output directory")
    for flag, dest, _, help_text in VALIDATE_APPEND_OPTIONS[1:]:
        validate.add_argument(flag, dest=dest, action="append", default=[], help=help_text)
    validate.add_argument("--out", help="Write validation summary JSON to this path")
    validate.add_argument("--strict", action="store_true", help="Treat warnings as validation failure")
    validate.set_defaults(func=cmd_validate)



def register_artifacts_2(subparsers: argparse._SubParsersAction) -> None:
    observer = subparsers.add_parser("observer-template", help="Print or write a read-only Hermes observer plugin template")
    observer.add_argument("--out", help="Write the template to this path instead of stdout")
    observer.add_argument("--force", action="store_true", help="Overwrite --out when it already exists")
    observer.set_defaults(func=cmd_observer_template)
