"""CLI commands for the gates domain."""

from __future__ import annotations

from ..decision_gate import DecisionGateError, evaluate_decision_gate, reject_symlinked_decision_artifact_input
from pathlib import Path
from ..review import REVIEW_LABELS, ReviewExportError, apply_review_labels, export_review_queue
from ..reviewed_gate import REVIEWED_GATE_POLICY_SCHEMA_VERSION, ReviewedGateError, ReviewedGatePolicyError, build_reviewed_export_source_artifact, evaluate_reviewed_gate, load_reviewed_gate_policy, snapshot_reviewed_export
from ..preflight import TrainerPreflightError, build_trainer_launch_check, build_trainer_preflight, snapshot_trainer_preflight
from ..validation import EVAL_SUITE_MANIFEST_SCHEMA_VERSION, VALIDATION_SCHEMA_VERSION, validate_artifacts, validate_trainer_preflight
import argparse
from ..path_safety import assert_output_does_not_alias_sources, assert_output_outside_source_directories, locked_owned_output_directory, output_directory_lock_is_held, path_has_symlink_component, remove_directory_tree_if_identity
from ..atomic_json import AtomicJsonError, atomic_write_json_cas, json_file_sha256
from ..action_gate import ACTION_LEDGER_GATE_POLICY_SCHEMA_VERSION, ActionLedgerGateError, ActionLedgerGatePolicyError, evaluate_action_ledger_gate, load_action_ledger_gate_policy
from ..compare_gate import COMPARE_GATE_POLICY_SCHEMA_VERSION, CompareGatePolicyError, evaluate_compare_gate, load_compare_gate_policy
from ..improvement_gate import IMPROVEMENT_LEDGER_GATE_POLICY_SCHEMA_VERSION, ImprovementLedgerGateError, ImprovementLedgerGatePolicyError, evaluate_improvement_ledger_gate, load_improvement_ledger_gate_policy
from ..promotion_gate import PROMOTION_LEDGER_GATE_POLICY_SCHEMA_VERSION, PromotionLedgerGateError, PromotionLedgerGatePolicyError, evaluate_promotion_ledger_gate, load_promotion_ledger_gate_policy
from ..suite_gate import SUITE_GATE_POLICY_SCHEMA_VERSION, SuiteGateError, SuiteGatePolicyError, evaluate_suite_gate, load_gate_policy
from ..training_gate import TRAINING_GATE_POLICY_SCHEMA_VERSION, TrainingGatePolicyError, evaluate_training_gate, load_training_gate_policy
import json
import sys
from .shared import _action_ledger_gate_options, _action_ledger_gate_policy_summary, _compare_gate_options, _compare_gate_policy_summary, _display_path, _display_path_for_output_source, _gate_policy_summary, _gate_suite_options, _improvement_ledger_gate_options, _improvement_ledger_gate_policy_summary, _metadata_arg, _metadata_options, _non_negative_float_arg, _non_negative_int_arg, _promotion_ledger_gate_options, _promotion_ledger_gate_policy_summary, _rate_arg, _read_json, _read_jsonl, _reviewed_gate_options, _reviewed_gate_policy_summary, _score_arg, _training_gate_options, _training_gate_policy_summary


def cmd_gate_suite(args: argparse.Namespace) -> int:
    suite_summary = _read_json(Path(args.suite_summary))
    options = _gate_suite_options(args)
    result = evaluate_suite_gate(
        suite_summary,
        suite_summary_path=_display_path(Path(args.suite_summary), args.preserve_paths),
        min_pass_rate=options["min_pass_rate"],
        min_average_score=options["min_average_score"],
        max_failed=options["max_failed"],
        max_errors=options["max_errors"],
        max_critical_failures=options["max_critical_failures"],
        forbid_failed_rules=options["forbid_failed_rules"],
        forbid_critical_rules=options["forbid_critical_rules"],
        task_family_gates=options["task_family_gates"],
    )
    if options["policy_path"]:
        result["policy"] = _gate_policy_summary(options)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


def cmd_gate_export(args: argparse.Namespace) -> int:
    export_dir = Path(args.training_export)
    metrics_path = export_dir / "dataset_metrics.json"
    dataset_metrics = _read_json(metrics_path)
    options = _training_gate_options(args)
    validation_summary = (
        validate_artifacts(training_export_dir=export_dir, strict=options["strict_validation"])
        if options["require_valid_export"]
        else None
    )
    result = evaluate_training_gate(
        dataset_metrics,
        training_export_path=_display_path(export_dir, args.preserve_paths),
        min_episodes=options["min_episodes"],
        min_pass_rate=options["min_pass_rate"],
        min_average_score=options["min_average_score"],
        min_preferences=options["min_preferences"],
        min_sft=options["min_sft"],
        min_dpo=options["min_dpo"],
        min_reward_model=options["min_reward_model"],
        min_step_rewards=options["min_step_rewards"],
        min_task_completion_configured=options["min_task_completion_configured"],
        min_task_completion_complete=options["min_task_completion_complete"],
        max_task_completion_incomplete=options["max_task_completion_incomplete"],
        min_task_completion_check_pass_rate=options["min_task_completion_check_pass_rate"],
        min_source_fingerprint_rate=options["min_source_fingerprint_rate"],
        max_unverified_source_fingerprints=options["max_unverified_source_fingerprints"],
        min_trainer_view_source_fingerprint_rate=options["min_trainer_view_source_fingerprint_rate"],
        max_unverified_trainer_view_source_fingerprints=options["max_unverified_trainer_view_source_fingerprints"],
        min_trace_average_events=options["min_trace_average_events"],
        min_trace_event_type_count=options["min_trace_event_type_count"],
        min_trace_final_answer_rate=options["min_trace_final_answer_rate"],
        min_trace_tool_or_api_rate=options["min_trace_tool_or_api_rate"],
        max_trace_empty_final_answers=options["max_trace_empty_final_answers"],
        max_trace_risk_count=options["max_trace_risk_count"],
        min_split_task_families=options["min_split_task_families"],
        min_train_episodes=options["min_train_episodes"],
        min_validation_episodes=options["min_validation_episodes"],
        min_test_episodes=options["min_test_episodes"],
        require_family_exclusive_splits=options["require_family_exclusive_splits"],
        max_quality_flags=options["max_quality_flags"],
        forbid_quality_flags=options["forbid_quality_flags"],
        forbid_quality_severities=options["forbid_quality_severities"],
        require_task_families=options["require_task_families"],
        require_trace_event_types=options["require_trace_event_types"],
        task_family_gates=options["task_family_gates"],
        validation_summary=validation_summary,
        require_valid_export=options["require_valid_export"],
    )
    if options["policy_path"]:
        result["policy"] = _training_gate_policy_summary(options)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


def cmd_gate_reviewed(args: argparse.Namespace) -> int:
    reviewed_dir = Path(args.reviewed_export)
    output_path = Path(args.out)
    try:
        assert_output_outside_source_directories(
            output_path,
            [reviewed_dir],
            label="reviewed gate",
        )
        source_files = [
            reviewed_dir / name
            for name in (
                "manifest.json",
                "dataset_registry.json",
                "reviewed_labels.jsonl",
                "reviewed_sft.jsonl",
                "reviewed_reward_model.jsonl",
                "reviewed_preferences.jsonl",
                "reviewed_dpo.jsonl",
            )
        ]
        if args.policy:
            source_files.append(Path(args.policy))
        assert_output_does_not_alias_sources(
            output_path,
            source_files,
            label="reviewed gate",
        )
    except ValueError as exc:
        raise ReviewedGateError(str(exc)) from exc
    expected_output_sha256 = json_file_sha256(output_path)
    reviewed_export_display_path = _display_path_for_output_source(
        reviewed_dir,
        output_path,
        False,
    )
    reviewed_export_ref = Path(reviewed_export_display_path)
    if (
        reviewed_export_ref.is_absolute()
        or reviewed_export_display_path.startswith("<redacted:")
        or reviewed_export_display_path in {"", "."}
        or ".." in reviewed_export_ref.parts
        or "\\" in reviewed_export_display_path
    ):
        raise ReviewedGateError(
            "reviewed gate output must be placed so --reviewed-export has a normalized, non-traversing relative path"
        )
    reviewed_snapshot = snapshot_reviewed_export(
        reviewed_dir,
        display_path=reviewed_export_display_path,
    )
    source_before = reviewed_snapshot.source_artifact
    manifest = reviewed_snapshot.manifest
    options = _reviewed_gate_options(args)
    if args.policy:
        options["policy_path"] = _display_path_for_output_source(
            Path(args.policy),
            output_path,
            False,
        )
    validation_summary = (
        validate_artifacts(reviewed_export_dir=reviewed_dir, strict=options["strict_validation"])
        if options["require_valid_export"]
        else None
    )
    result = evaluate_reviewed_gate(
        manifest,
        reviewed_export_path=reviewed_export_display_path,
        reviewed_export_source=source_before,
        min_reviewed_labels=options["min_reviewed_labels"],
        min_accepted=options["min_accepted"],
        min_rejected=options["min_rejected"],
        min_sft=options["min_sft"],
        min_reward_model=options["min_reward_model"],
        min_preferences=options["min_preferences"],
        min_dpo=options["min_dpo"],
        min_high_confidence_labels=options["min_high_confidence_labels"],
        min_medium_or_high_confidence_labels=options["min_medium_or_high_confidence_labels"],
        max_needs_review=options["max_needs_review"],
        max_low_confidence_labels=options["max_low_confidence_labels"],
        max_unknown_confidence_labels=options["max_unknown_confidence_labels"],
        forbid_labels=options["forbid_labels"],
        require_task_families=options["require_task_families"],
        validation_summary=validation_summary,
        require_valid_export=options["require_valid_export"],
        strict_validation=options["strict_validation"],
    )
    if options["policy_path"]:
        result["policy"] = _reviewed_gate_policy_summary(options)
    source_after = build_reviewed_export_source_artifact(
        reviewed_dir,
        display_path=reviewed_export_display_path,
    )
    if source_before != source_after:
        raise ReviewedGateError("reviewed export changed while its gate was being evaluated")
    atomic_write_json_cas(
        output_path,
        result,
        expected_sha256=expected_output_sha256,
        new_file_mode=0o666,
    )
    print(f"wrote {args.out}")
    return 0 if result["passed"] else 1


def cmd_gate_compare_export(args: argparse.Namespace) -> int:
    compare_dir = Path(args.compare_export)
    manifest = _read_json(compare_dir / "manifest.json")
    pairs = _read_jsonl(compare_dir / "improvement_pairs.jsonl")
    options = _compare_gate_options(args)
    validation_summary = (
        validate_artifacts(compare_export_dir=compare_dir, strict=options["strict_validation"])
        if options["require_valid_export"]
        else None
    )
    result = evaluate_compare_gate(
        manifest,
        pairs,
        compare_export_path=_display_path(compare_dir, args.preserve_paths),
        min_pairs=options["min_pairs"],
        min_dpo=options["min_dpo"],
        min_candidate_wins=options["min_candidate_wins"],
        min_task_completion_improvements=options["min_task_completion_improvements"],
        max_baseline_wins=options["max_baseline_wins"],
        max_task_completion_regressions=options["max_task_completion_regressions"],
        max_skipped_pairs=options["max_skipped_pairs"],
        max_contract_drifts=options["max_contract_drifts"],
        max_unverified_contracts=options["max_unverified_contracts"],
        require_scenarios=options["require_scenarios"],
        require_candidate_win_scenarios=options["require_candidate_win_scenarios"],
        require_task_completion_improvement_scenarios=options["require_task_completion_improvement_scenarios"],
        forbid_regression_scenarios=options["forbid_regression_scenarios"],
        forbid_task_completion_regression_scenarios=options["forbid_task_completion_regression_scenarios"],
        require_rule_fixes=options["require_rule_fixes"],
        forbid_rule_regressions=options["forbid_rule_regressions"],
        forbid_new_critical_failures=options["forbid_new_critical_failures"],
        task_family_gates=options["task_family_gates"],
        validation_summary=validation_summary,
        require_valid_export=options["require_valid_export"],
    )
    if options["policy_path"]:
        result["policy"] = _compare_gate_policy_summary(options)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


def cmd_gate_action_ledger(args: argparse.Namespace) -> int:
    ledger = _read_json(Path(args.action_ledger))
    options = _action_ledger_gate_options(args)
    output_path = Path(args.out) if args.out else None
    result = evaluate_action_ledger_gate(
        ledger,
        action_ledger_path=_display_path_for_output_source(
            Path(args.action_ledger),
            output_path,
            args.preserve_paths,
        ),
        min_bundles=options["min_bundles"],
        max_open_actions=options["max_open_actions"],
        max_new_actions=options["max_new_actions"],
        max_recurring_actions=options["max_recurring_actions"],
        min_resolved_actions=options["min_resolved_actions"],
        forbid_open_priorities=options["forbid_open_priorities"],
        forbid_open_actions=options["forbid_open_actions"],
        require_resolved_actions=options["require_resolved_actions"],
    )
    if options["policy_path"]:
        result["policy"] = _action_ledger_gate_policy_summary(options)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


def cmd_gate_improvement_ledger(args: argparse.Namespace) -> int:
    ledger = _read_json(Path(args.improvement_ledger))
    options = _improvement_ledger_gate_options(args)
    output_path = Path(args.out) if args.out else None
    result = evaluate_improvement_ledger_gate(
        ledger,
        improvement_ledger_path=_display_path_for_output_source(
            Path(args.improvement_ledger),
            output_path,
            args.preserve_paths,
        ),
        min_plans=options["min_plans"],
        max_open_work_items=options["max_open_work_items"],
        max_new_work_items=options["max_new_work_items"],
        max_recurring_work_items=options["max_recurring_work_items"],
        min_resolved_work_items=options["min_resolved_work_items"],
        max_critical_open_work_items=options["max_critical_open_work_items"],
        max_high_open_work_items=options["max_high_open_work_items"],
        forbid_open_priorities=options["forbid_open_priorities"],
        forbid_open_categories=options["forbid_open_categories"],
        forbid_open_work_keys=options["forbid_open_work_keys"],
        require_open_work_keys=options["require_open_work_keys"],
        require_resolved_work_keys=options["require_resolved_work_keys"],
    )
    if options["policy_path"]:
        result["policy"] = _improvement_ledger_gate_policy_summary(options)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


def cmd_gate_promotion_ledger(args: argparse.Namespace) -> int:
    promotion_ledger_path = Path(args.promotion_ledger)
    ledger = _read_json(promotion_ledger_path)
    options = _promotion_ledger_gate_options(args)
    output_path = Path(args.out) if args.out else None
    result = evaluate_promotion_ledger_gate(
        ledger,
        promotion_ledger_path=promotion_ledger_path,
        promotion_ledger_display_path=_display_path_for_output_source(
            promotion_ledger_path,
            output_path,
            args.preserve_paths,
        ),
        min_decisions=options["min_decisions"],
        min_allowed_count=options["min_allowed_count"],
        max_blocked_count=options["max_blocked_count"],
        max_blocked_rate=options["max_blocked_rate"],
        min_consecutive_allowed=options["min_consecutive_allowed"],
        max_consecutive_blocked=options["max_consecutive_blocked"],
        max_failed_decisions=options["max_failed_decisions"],
        require_latest_recommendation=options["require_latest_recommendation"],
        require_latest_passed=options["require_latest_passed"],
        require_source_recommendations=options["require_source_recommendations"],
        forbid_source_recommendations=options["forbid_source_recommendations"],
    )
    if options["policy_path"]:
        result["policy"] = _promotion_ledger_gate_policy_summary(options)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


def cmd_gate_decision(args: argparse.Namespace) -> int:
    artifact_path = Path(args.artifact)
    output_path = Path(args.out) if args.out else None
    if output_path is not None:
        try:
            assert_output_does_not_alias_sources(
                output_path,
                [artifact_path],
                label="decision gate",
            )
        except ValueError as exc:
            raise DecisionGateError(str(exc)) from exc
    expected_output_sha256 = (
        json_file_sha256(output_path) if output_path is not None else None
    )
    reject_symlinked_decision_artifact_input(artifact_path)
    artifact = _read_json(artifact_path)
    result = evaluate_decision_gate(
        artifact,
        artifact_path=artifact_path,
        artifact_display_path=_display_path_for_output_source(
            artifact_path,
            output_path,
            args.preserve_paths,
        ),
        expect_recommendation=args.expect_recommendation,
        expect_readiness=args.expect_readiness,
        require_passed=args.require_passed,
        preserve_paths=args.preserve_paths,
    )
    if output_path is not None:
        atomic_write_json_cas(
            output_path,
            result,
            expected_sha256=expected_output_sha256,
            new_file_mode=0o666,
        )
        print(f"wrote {args.out}")
    else:
        rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        print(rendered, end="")
    return 0 if result["passed"] else 1


def cmd_trainer_preflight(args: argparse.Namespace) -> int:
    output_path = Path(args.out)
    source_directories = [
        Path(value)
        for value in (
            args.training_export,
            args.compare_export,
            args.reviewed_export,
        )
        if value
    ]
    source_files = [Path(value) for value in args.gate]
    source_files.extend(
        Path(value)
        for value in (
            args.evidence_bundle,
            args.agentic_training_plan,
        )
        if value
    )
    source_files.extend(Path(value) for value in args.validation)
    try:
        assert_output_outside_source_directories(
            output_path,
            source_directories,
            label="trainer preflight",
        )
        assert_output_does_not_alias_sources(
            output_path,
            source_files,
            label="trainer preflight",
        )
    except ValueError as exc:
        raise TrainerPreflightError(str(exc)) from exc
    expected_output_sha256 = json_file_sha256(output_path)
    metadata = _metadata_options(args.metadata)
    preflight = build_trainer_preflight(
        out_path=args.out,
        gate_paths=args.gate,
        training_export_dir=args.training_export,
        compare_export_dir=args.compare_export,
        reviewed_export_dir=args.reviewed_export,
        evidence_bundle_path=args.evidence_bundle,
        agentic_training_plan_path=args.agentic_training_plan,
        validation_summary_paths=args.validation,
        require_gates=args.require_gate,
        required_dataset_versions=args.require_dataset_version,
        trainer_command=args.trainer_command,
        allow_unvalidated_gates=args.allow_unvalidated_gates,
        preserve_paths=args.preserve_paths,
        metadata=metadata,
    )
    atomic_write_json_cas(
        output_path,
        preflight,
        expected_sha256=expected_output_sha256,
        new_file_mode=0o666,
    )
    print(
        f"{'READY' if preflight['passed'] else 'BLOCKED'} trainer-preflight "
        f"gates={preflight['passed_gate_count']}/{preflight['gate_count']} out={args.out}"
    )
    return 0 if preflight["passed"] else 1


def cmd_trainer_launch_check(args: argparse.Namespace) -> int:
    if not args.out and not args.print_command:
        raise TrainerPreflightError(
            "trainer-launch-check requires --out unless --print-command is used"
        )
    preflight_path = Path(args.preflight)
    output_path = Path(args.out) if args.out else None
    if output_path is not None:
        try:
            assert_output_does_not_alias_sources(
                output_path,
                [preflight_path],
                label="trainer launch check",
            )
        except ValueError as exc:
            raise TrainerPreflightError(str(exc)) from exc
    expected_output_sha256 = (
        json_file_sha256(output_path) if output_path is not None else None
    )
    preflight_snapshot = snapshot_trainer_preflight(
        preflight_path,
        out_path=output_path,
        preserve_paths=args.preserve_paths,
    )
    preflight = preflight_snapshot.payload
    validation_target = validate_trainer_preflight(
        preflight_path,
        payload=preflight,
    )
    validation_error_count = len(validation_target.errors)
    validation_warning_count = len(validation_target.warnings)
    validation_summary = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "passed": validation_error_count == 0
        and (validation_warning_count == 0 or not args.strict),
        "strict": args.strict,
        "target_count": 1,
        "error_count": validation_error_count,
        "warning_count": validation_warning_count,
        "targets": [validation_target.as_dict()],
    }
    launch_check = build_trainer_launch_check(
        preflight_path=preflight_path,
        preflight=preflight,
        preflight_snapshot=preflight_snapshot,
        validation_summary=validation_summary,
        out_path=output_path,
        require_gates=args.require_gate,
        required_dataset_versions=args.require_dataset_version,
        require_metadata=_metadata_options(args.require_metadata),
        preserve_paths=args.preserve_paths,
    )
    if output_path is not None:
        atomic_write_json_cas(
            output_path,
            launch_check,
            expected_sha256=expected_output_sha256,
            new_file_mode=0o666,
        )
    if args.print_command:
        if launch_check["passed"]:
            print(launch_check["approved_command"]["shell"])
        else:
            print(
                "BLOCKED trainer-launch-check: the preflight did not authorize a command",
                file=sys.stderr,
            )
    else:
        print(
            f"{'READY' if launch_check['passed'] else 'BLOCKED'} trainer-launch-check "
            f"checks={launch_check['check_count'] - launch_check['failed_check_count']}/{launch_check['check_count']} "
            f"out={args.out}"
        )
    return 0 if launch_check["passed"] else 1


def register_gates_1(subparsers: argparse._SubParsersAction) -> None:
    gate_improvement_ledger = subparsers.add_parser(
        "gate-improvement-ledger",
        help="Evaluate concrete improvement-work thresholds against an improvement ledger",
    )
    gate_improvement_ledger.add_argument("--improvement-ledger", required=True, help="Path to improvement_ledger.json")
    gate_improvement_ledger.add_argument("--policy", help="Versioned improvement-ledger gate policy JSON file")
    gate_improvement_ledger.add_argument("--out", help="Write improvement-ledger gate JSON to this path")
    gate_improvement_ledger.add_argument("--min-plans", type=_non_negative_int_arg, help="Minimum improvement plans required")
    gate_improvement_ledger.add_argument(
        "--max-open-work-items",
        type=_non_negative_int_arg,
        help="Maximum concrete work items still open in the latest plan",
    )
    gate_improvement_ledger.add_argument(
        "--max-new-work-items",
        type=_non_negative_int_arg,
        help="Maximum work items first seen in the latest plan",
    )
    gate_improvement_ledger.add_argument(
        "--max-recurring-work-items",
        type=_non_negative_int_arg,
        help="Maximum work items recurring from earlier plans",
    )
    gate_improvement_ledger.add_argument(
        "--min-resolved-work-items",
        type=_non_negative_int_arg,
        help="Minimum work items resolved before the latest plan",
    )
    gate_improvement_ledger.add_argument(
        "--max-critical-open-work-items",
        type=_non_negative_int_arg,
        help="Maximum critical-priority work items still open",
    )
    gate_improvement_ledger.add_argument(
        "--max-high-open-work-items",
        type=_non_negative_int_arg,
        help="Maximum high-priority work items still open",
    )
    gate_improvement_ledger.add_argument(
        "--forbid-open-priority",
        action="append",
        default=[],
        choices=["critical", "high", "medium", "low"],
        help="Fail if any open work item has this priority; may be repeated",
    )
    gate_improvement_ledger.add_argument(
        "--forbid-open-category",
        action="append",
        default=[],
        choices=["bundle_action", "repair", "curriculum", "digest_action"],
        help="Fail if any open work item has this category; may be repeated",
    )
    gate_improvement_ledger.add_argument(
        "--forbid-open-work-key",
        action="append",
        default=[],
        help="Fail if this work key, routing key, item id, or fingerprint is open; may be repeated",
    )
    gate_improvement_ledger.add_argument(
        "--require-open-work-key",
        action="append",
        default=[],
        help="Fail unless this work key, routing key, item id, or fingerprint is open; may be repeated",
    )
    gate_improvement_ledger.add_argument(
        "--require-resolved-work-key",
        action="append",
        default=[],
        help="Fail unless this work key, routing key, item id, or fingerprint is resolved; may be repeated",
    )
    gate_improvement_ledger.add_argument("--preserve-paths", action="store_true", help="Allow absolute ledger paths in gate output")
    gate_improvement_ledger.set_defaults(func=cmd_gate_improvement_ledger)



def register_gates_2(subparsers: argparse._SubParsersAction) -> None:
    gate_promotion_ledger = subparsers.add_parser(
        "gate-promotion-ledger",
        help="Evaluate promotion-history thresholds against a promotion ledger",
    )
    gate_promotion_ledger.add_argument("--promotion-ledger", required=True, help="Path to promotion_ledger.json")
    gate_promotion_ledger.add_argument("--policy", help="Versioned promotion-ledger gate policy JSON file")
    gate_promotion_ledger.add_argument("--out", help="Write promotion-ledger gate JSON to this path")
    gate_promotion_ledger.add_argument("--min-decisions", type=_non_negative_int_arg, help="Minimum promotion decisions required")
    gate_promotion_ledger.add_argument("--min-allowed-count", type=_non_negative_int_arg, help="Minimum allowed promotion decisions required")
    gate_promotion_ledger.add_argument("--max-blocked-count", type=_non_negative_int_arg, help="Maximum blocked promotion decisions allowed")
    gate_promotion_ledger.add_argument("--max-blocked-rate", type=_rate_arg, help="Maximum blocked promotion decision rate, from 0.0 to 1.0")
    gate_promotion_ledger.add_argument("--min-consecutive-allowed", type=_non_negative_int_arg, help="Minimum consecutive allow decisions required at the ledger tail")
    gate_promotion_ledger.add_argument("--max-consecutive-blocked", type=_non_negative_int_arg, help="Maximum consecutive block decisions allowed at the ledger tail")
    gate_promotion_ledger.add_argument("--max-failed-decisions", type=_non_negative_int_arg, help="Maximum decision gates with failed checks")
    gate_promotion_ledger.add_argument(
        "--require-latest-recommendation",
        choices=["allow_promotion", "block_promotion"],
        help="Required latest promotion-ledger recommendation",
    )
    gate_promotion_ledger.add_argument("--require-latest-passed", action="store_true", help="Require the latest decision gate to have passed")
    gate_promotion_ledger.add_argument(
        "--require-source-recommendation",
        action="append",
        default=[],
        help="Fail unless this source recommendation appears in the ledger history; may be repeated",
    )
    gate_promotion_ledger.add_argument(
        "--forbid-source-recommendation",
        action="append",
        default=[],
        help="Fail if this source recommendation appears in the ledger history; may be repeated",
    )
    gate_promotion_ledger.add_argument("--preserve-paths", action="store_true", help="Allow absolute ledger paths in gate output")
    gate_promotion_ledger.set_defaults(func=cmd_gate_promotion_ledger)

    gate_action_ledger = subparsers.add_parser(
        "gate-action-ledger",
        help="Evaluate improvement-loop thresholds against an action ledger",
    )
    gate_action_ledger.add_argument("--action-ledger", required=True, help="Path to action_ledger.json")
    gate_action_ledger.add_argument("--policy", help="Versioned action-ledger gate policy JSON file")
    gate_action_ledger.add_argument("--out", help="Write action-ledger gate JSON to this path")
    gate_action_ledger.add_argument("--min-bundles", type=_non_negative_int_arg, help="Minimum evidence bundles required in the ledger")
    gate_action_ledger.add_argument("--max-open-actions", type=_non_negative_int_arg, help="Maximum actions still open in the latest bundle")
    gate_action_ledger.add_argument("--max-new-actions", type=_non_negative_int_arg, help="Maximum actions first seen in the latest bundle")
    gate_action_ledger.add_argument("--max-recurring-actions", type=_non_negative_int_arg, help="Maximum actions recurring from earlier bundles")
    gate_action_ledger.add_argument("--min-resolved-actions", type=_non_negative_int_arg, help="Minimum actions resolved before the latest bundle")
    gate_action_ledger.add_argument(
        "--forbid-open-priority",
        action="append",
        default=[],
        choices=["critical", "high", "medium", "low"],
        help="Fail if any open action has this priority; may be repeated",
    )
    gate_action_ledger.add_argument(
        "--forbid-open-action",
        action="append",
        default=[],
        help="Fail if this action id, routing key, or fingerprint is open; may be repeated",
    )
    gate_action_ledger.add_argument(
        "--require-resolved-action",
        action="append",
        default=[],
        help="Fail unless this action id, routing key, or fingerprint is resolved; may be repeated",
    )
    gate_action_ledger.add_argument("--preserve-paths", action="store_true", help="Allow absolute ledger paths in gate output")
    gate_action_ledger.set_defaults(func=cmd_gate_action_ledger)

    gate_decision = subparsers.add_parser(
        "gate-decision",
        help="Gate a Flight Recorder decision recommendation for CI promotion",
    )
    gate_decision.add_argument(
        "--artifact",
        required=True,
        help="Supported registered Flight Recorder decision artifact satisfying its bundled schema",
    )
    gate_decision.add_argument("--expect-recommendation", required=True, help="Required decision.recommendation value")
    gate_decision.add_argument("--expect-readiness", help="Optional required decision.readiness value")
    gate_decision.add_argument("--require-passed", action="store_true", help="Also require the source artifact root passed field to be true")
    gate_decision.add_argument("--out", help="Write decision gate JSON to this path")
    gate_decision.add_argument("--preserve-paths", action="store_true", help="Allow absolute artifact paths in gate output")
    gate_decision.set_defaults(func=cmd_gate_decision)



def register_gates_3(subparsers: argparse._SubParsersAction) -> None:
    gate_suite = subparsers.add_parser("gate-suite", help="Evaluate CI thresholds against a run-suite summary")
    gate_suite.add_argument("--suite-summary", required=True, help="Path to suite_summary.json")
    gate_suite.add_argument("--policy", help="Versioned suite gate policy JSON file with committed threshold defaults")
    gate_suite.add_argument("--out", help="Write gate result JSON to this path")
    gate_suite.add_argument("--min-pass-rate", type=_rate_arg, help="Minimum allowed suite pass rate, from 0.0 to 1.0")
    gate_suite.add_argument("--min-average-score", type=_score_arg, help="Minimum allowed average score, from 0 to 100")
    gate_suite.add_argument("--max-failed", type=_non_negative_int_arg, help="Maximum allowed failed scenarios")
    gate_suite.add_argument("--max-errors", type=_non_negative_int_arg, help="Maximum allowed suite execution errors; defaults to policy value or 0")
    gate_suite.add_argument("--max-critical-failures", type=_non_negative_int_arg, help="Maximum allowed total critical failures")
    gate_suite.add_argument("--forbid-failed-rule", action="append", default=[], help="Fail if this rule id appears in failed_rule_counts")
    gate_suite.add_argument("--forbid-critical-rule", action="append", default=[], help="Fail if this rule id appears in critical_failure_counts")
    gate_suite.add_argument("--preserve-paths", action="store_true", help="Allow absolute suite summary paths in gate output")
    gate_suite.set_defaults(func=cmd_gate_suite)

    gate_export = subparsers.add_parser("gate-export", help="Evaluate readiness thresholds against an export-rl dataset")
    gate_export.add_argument("--training-export", required=True, help="Directory containing export-rl artifacts")
    gate_export.add_argument("--policy", help="Versioned training gate policy JSON file with committed threshold defaults")
    gate_export.add_argument("--out", help="Write gate result JSON to this path")
    gate_export.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail export-integrity validation on warnings as well as errors",
    )
    gate_export.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip export structure and artifact-fingerprint validation before gating",
    )
    gate_export.add_argument("--min-episodes", type=_non_negative_int_arg, help="Minimum episode count")
    gate_export.add_argument("--min-pass-rate", type=_rate_arg, help="Minimum dataset pass rate, from 0.0 to 1.0")
    gate_export.add_argument("--min-average-score", type=_score_arg, help="Minimum average score, from 0 to 100")
    gate_export.add_argument("--min-preferences", type=_non_negative_int_arg, help="Minimum preference pair count")
    gate_export.add_argument("--min-sft", type=_non_negative_int_arg, help="Minimum SFT row count")
    gate_export.add_argument("--min-dpo", type=_non_negative_int_arg, help="Minimum DPO row count")
    gate_export.add_argument("--min-reward-model", type=_non_negative_int_arg, help="Minimum reward-model row count")
    gate_export.add_argument("--min-step-rewards", type=_non_negative_int_arg, help="Minimum step-reward row count")
    gate_export.add_argument(
        "--min-task-completion-configured",
        type=_non_negative_int_arg,
        help="Minimum episodes with task-completion evidence configured",
    )
    gate_export.add_argument(
        "--min-task-completion-complete",
        type=_non_negative_int_arg,
        help="Minimum episodes with complete task-completion evidence",
    )
    gate_export.add_argument(
        "--max-task-completion-incomplete",
        type=_non_negative_int_arg,
        help="Maximum episodes with incomplete task-completion evidence",
    )
    gate_export.add_argument(
        "--min-task-completion-check-pass-rate",
        type=_rate_arg,
        help="Minimum required task-evidence check pass rate, from 0.0 to 1.0",
    )
    gate_export.add_argument(
        "--min-source-fingerprint-rate",
        type=_rate_arg,
        help="Minimum fraction of episodes with scenario and source-trace SHA-256 fingerprints",
    )
    gate_export.add_argument(
        "--max-unverified-source-fingerprints",
        type=_non_negative_int_arg,
        help="Maximum episodes allowed to lack scenario or source-trace SHA-256 fingerprints",
    )
    gate_export.add_argument(
        "--min-trainer-view-source-fingerprint-rate",
        type=_rate_arg,
        help="Minimum fraction of SFT/DPO/reward-model rows with complete source fingerprints",
    )
    gate_export.add_argument(
        "--max-unverified-trainer-view-source-fingerprints",
        type=_non_negative_int_arg,
        help="Maximum SFT/DPO/reward-model rows allowed to lack complete source fingerprints",
    )
    gate_export.add_argument("--min-trace-average-events", type=_non_negative_float_arg, help="Minimum average normalized events per exported episode")
    gate_export.add_argument("--min-trace-event-type-count", type=_non_negative_int_arg, help="Minimum distinct normalized event types in the export")
    gate_export.add_argument("--min-trace-final-answer-rate", type=_rate_arg, help="Minimum fraction of episodes with final answers")
    gate_export.add_argument("--min-trace-tool-or-api-rate", type=_rate_arg, help="Minimum fraction of episodes with tool or API events")
    gate_export.add_argument("--max-trace-empty-final-answers", type=_non_negative_int_arg, help="Maximum exported episodes allowed to have empty final answers")
    gate_export.add_argument("--max-trace-risk-count", type=_non_negative_int_arg, help="Maximum total trace observability risks allowed")
    gate_export.add_argument("--require-trace-event-type", action="append", default=[], help="Fail unless this normalized event type appears in exported traces")
    gate_export.add_argument("--min-split-task-families", type=_non_negative_int_arg, help="Minimum task families represented in dataset split metadata")
    gate_export.add_argument("--min-train-episodes", type=_non_negative_int_arg, help="Minimum episodes assigned to the train split")
    gate_export.add_argument("--min-validation-episodes", type=_non_negative_int_arg, help="Minimum episodes assigned to the validation split")
    gate_export.add_argument("--min-test-episodes", type=_non_negative_int_arg, help="Minimum episodes assigned to the test split")
    gate_export.add_argument(
        "--require-family-exclusive-splits",
        action="store_true",
        help="Fail unless dataset split metadata reports task-family-exclusive train/validation/test splits",
    )
    gate_export.add_argument("--max-quality-flags", type=_non_negative_int_arg, help="Maximum allowed dataset quality flags")
    gate_export.add_argument("--forbid-quality-flag", action="append", default=[], help="Fail if this quality flag id appears")
    gate_export.add_argument(
        "--forbid-quality-severity",
        action="append",
        default=[],
        choices=["info", "warning", "error"],
        help="Fail if any quality flag has this severity",
    )
    gate_export.add_argument("--require-task-family", action="append", default=[], help="Fail unless this task family is present")
    gate_export.add_argument("--preserve-paths", action="store_true", help="Allow absolute export paths in gate output")
    gate_export.set_defaults(func=cmd_gate_export)

    gate_reviewed = subparsers.add_parser("gate-reviewed", help="Evaluate readiness thresholds against an apply-review export")
    gate_reviewed.add_argument("--reviewed-export", required=True, help="Directory containing apply-review artifacts")
    gate_reviewed.add_argument("--policy", help="Versioned reviewed gate policy JSON file with committed threshold defaults")
    gate_reviewed.add_argument("--out", required=True, help="Write the authoritative gate result JSON to this path")
    gate_reviewed.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail reviewed-export validation on warnings as well as errors",
    )
    gate_reviewed.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip reviewed export structure and artifact-fingerprint validation before gating",
    )
    gate_reviewed.add_argument("--min-reviewed-labels", type=_non_negative_int_arg, help="Minimum completed human label count")
    gate_reviewed.add_argument("--min-accepted", type=_non_negative_int_arg, help="Minimum accepted human label count")
    gate_reviewed.add_argument(
        "--min-rejected",
        type=_non_negative_int_arg,
        help="Minimum negative human label count across reject, unsafe, and incomplete labels",
    )
    gate_reviewed.add_argument("--min-sft", type=_non_negative_int_arg, help="Minimum reviewed SFT row count")
    gate_reviewed.add_argument("--min-reward-model", type=_non_negative_int_arg, help="Minimum reviewed reward-model row count")
    gate_reviewed.add_argument("--min-preferences", type=_non_negative_int_arg, help="Minimum reviewed preference pair count")
    gate_reviewed.add_argument("--min-dpo", type=_non_negative_int_arg, help="Minimum reviewed DPO row count")
    gate_reviewed.add_argument(
        "--min-high-confidence-labels",
        type=_non_negative_int_arg,
        help="Minimum reviewed labels marked reviewer_confidence='high'",
    )
    gate_reviewed.add_argument(
        "--min-medium-or-high-confidence-labels",
        type=_non_negative_int_arg,
        help="Minimum reviewed labels marked reviewer_confidence='medium' or 'high'",
    )
    gate_reviewed.add_argument("--max-needs-review", type=_non_negative_int_arg, help="Maximum allowed needs_review labels")
    gate_reviewed.add_argument(
        "--max-low-confidence-labels",
        type=_non_negative_int_arg,
        help="Maximum reviewed labels marked reviewer_confidence='low'",
    )
    gate_reviewed.add_argument(
        "--max-unknown-confidence-labels",
        type=_non_negative_int_arg,
        help="Maximum reviewed labels with missing or unknown reviewer_confidence",
    )
    gate_reviewed.add_argument(
        "--forbid-label",
        action="append",
        default=[],
        choices=list(REVIEW_LABELS),
        help="Fail if this human label appears in the reviewed export",
    )
    gate_reviewed.add_argument("--require-task-family", action="append", default=[], help="Fail unless this task family is present")
    gate_reviewed.add_argument(
        "--preserve-paths",
        action="store_true",
        help="Deprecated compatibility flag; authoritative gate references remain output-relative",
    )
    gate_reviewed.set_defaults(func=cmd_gate_reviewed)

    gate_compare = subparsers.add_parser("gate-compare-export", help="Evaluate readiness thresholds against an export-compare-rl dataset")
    gate_compare.add_argument("--compare-export", required=True, help="Directory containing export-compare-rl artifacts")
    gate_compare.add_argument("--policy", help="Versioned compare gate policy JSON file with committed threshold defaults")
    gate_compare.add_argument("--out", help="Write gate result JSON to this path")
    gate_compare.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail comparison-export validation on warnings as well as errors",
    )
    gate_compare.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip comparison export structure and artifact-fingerprint validation before gating",
    )
    gate_compare.add_argument("--min-pairs", type=_non_negative_int_arg, help="Minimum comparison pair count")
    gate_compare.add_argument("--min-dpo", type=_non_negative_int_arg, help="Minimum comparison DPO row count")
    gate_compare.add_argument("--min-candidate-wins", type=_non_negative_int_arg, help="Minimum candidate-win pair count")
    gate_compare.add_argument(
        "--min-task-completion-improvements",
        type=_non_negative_int_arg,
        help="Minimum pairs where the candidate completed the task and the baseline did not",
    )
    gate_compare.add_argument("--max-baseline-wins", type=_non_negative_int_arg, help="Maximum allowed baseline-win regression pairs")
    gate_compare.add_argument(
        "--max-task-completion-regressions",
        type=_non_negative_int_arg,
        help="Maximum pairs where the baseline completed the task and the candidate did not",
    )
    gate_compare.add_argument("--max-skipped-pairs", type=_non_negative_int_arg, help="Maximum allowed skipped paired scenarios")
    gate_compare.add_argument("--max-contract-drifts", type=_non_negative_int_arg, help="Maximum allowed drifted comparison contract fingerprints")
    gate_compare.add_argument(
        "--max-unverified-contracts",
        type=_non_negative_int_arg,
        help="Maximum allowed comparison pairs missing scenario or source-trace fingerprints",
    )
    gate_compare.add_argument("--require-scenario", action="append", default=[], help="Fail unless this scenario appears in the comparison pairs")
    gate_compare.add_argument(
        "--require-candidate-win-scenario",
        action="append",
        default=[],
        help="Fail unless this scenario is a candidate win",
    )
    gate_compare.add_argument(
        "--require-task-completion-improvement-scenario",
        action="append",
        default=[],
        help="Fail unless this scenario has candidate task completion and baseline non-completion",
    )
    gate_compare.add_argument(
        "--forbid-regression-scenario",
        action="append",
        default=[],
        help="Fail if this scenario is a baseline win",
    )
    gate_compare.add_argument(
        "--forbid-task-completion-regression-scenario",
        action="append",
        default=[],
        help="Fail if this scenario has baseline task completion and candidate non-completion",
    )
    gate_compare.add_argument("--require-rule-fix", action="append", default=[], help="Fail unless this rule id appears in rule_fixes")
    gate_compare.add_argument(
        "--forbid-rule-regression",
        action="append",
        default=[],
        help="Fail if this rule id appears in rule_regressions",
    )
    gate_compare.add_argument(
        "--forbid-new-critical-failure",
        action="append",
        default=[],
        help="Fail if this rule id appears in new_critical_failures",
    )
    gate_compare.add_argument("--preserve-paths", action="store_true", help="Allow absolute export paths in gate output")
    gate_compare.set_defaults(func=cmd_gate_compare_export)

    trainer_preflight = subparsers.add_parser(
        "trainer-preflight",
        help="Build a launch guard manifest over passed gates and trainer-facing artifacts",
    )
    trainer_preflight.add_argument("--gate", action="append", required=True, help="Gate JSON that must pass; may be repeated")
    trainer_preflight.add_argument("--out", required=True, help="Write trainer preflight JSON to this path")
    trainer_preflight.add_argument("--training-export", help="export-rl directory to fingerprint for the trainer handoff")
    trainer_preflight.add_argument("--compare-export", help="export-compare-rl directory to fingerprint for the trainer handoff")
    trainer_preflight.add_argument("--reviewed-export", help="apply-review directory to fingerprint for the trainer handoff")
    trainer_preflight.add_argument("--evidence-bundle", help="evidence_bundle.json that must pass before launch")
    trainer_preflight.add_argument("--agentic-training-plan", help="hfr.agentic_training_plan.v1 dry-run plan to fingerprint for the trainer handoff")
    trainer_preflight.add_argument(
        "--validation",
        action="append",
        default=[],
        help="flightrecorder validate --out summary that proves non-embedded gate artifacts; may be repeated",
    )
    trainer_preflight.add_argument("--require-gate", action="append", default=[], help="Require this gate id to be present")
    trainer_preflight.add_argument(
        "--require-dataset-version",
        action="append",
        default=[],
        help="Require an exact manifest dataset_version before trainer launch; may be repeated",
    )
    trainer_preflight.add_argument("--trainer-command", help="Trainer command to record but not execute")
    trainer_preflight.add_argument(
        "--allow-unvalidated-gates",
        action="store_true",
        help="Allow gates that skipped embedded or external validation proof",
    )
    trainer_preflight.add_argument("--preserve-paths", action="store_true", help="Allow absolute artifact paths in preflight output")
    trainer_preflight.add_argument(
        "--metadata",
        action="append",
        default=[],
        type=_metadata_arg,
        metavar="KEY=VALUE",
        help="Attach launch metadata to the preflight manifest; may be repeated",
    )
    trainer_preflight.set_defaults(func=cmd_trainer_preflight)

    trainer_launch_check = subparsers.add_parser(
        "trainer-launch-check",
        help="Validate a trainer preflight and emit the approved command without executing it",
    )
    trainer_launch_check.add_argument("--preflight", required=True, help="trainer_preflight.json to verify before trainer launch")
    trainer_launch_check.add_argument(
        "--out",
        help="Write trainer launch-check JSON to this path; required unless --print-command is used",
    )
    trainer_launch_check.add_argument("--require-gate", action="append", default=[], help="Require this preflight gate id to be present and passed")
    trainer_launch_check.add_argument(
        "--require-dataset-version",
        action="append",
        default=[],
        help="Require an exact dataset_version already selected by the trainer preflight; may be repeated",
    )
    trainer_launch_check.add_argument(
        "--require-metadata",
        action="append",
        default=[],
        type=_metadata_arg,
        metavar="KEY=VALUE",
        help="Require this metadata key/value from the preflight manifest; may be repeated",
    )
    trainer_launch_check.add_argument("--strict", action="store_true", help="Treat preflight validation warnings as launch blockers")
    trainer_launch_check.add_argument("--print-command", action="store_true", help="Print only the shell-escaped approved command when launch is allowed")
    trainer_launch_check.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in launch-check output")
    trainer_launch_check.set_defaults(func=cmd_trainer_launch_check)

