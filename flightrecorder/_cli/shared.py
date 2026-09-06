"""CLI commands for the shared domain."""

from __future__ import annotations

from ..action_gate import ACTION_LEDGER_GATE_POLICY_SCHEMA_VERSION, ActionLedgerGateError, ActionLedgerGatePolicyError, evaluate_action_ledger_gate, load_action_ledger_gate_policy
from typing import Any, Iterator
from ..artifacts import ArtifactError, build_suite_trend, compare_scorecards, compare_suites, write_compare_report, write_junit, write_markdown_summary, write_suite_compare_report, write_suite_trend_report
from ..atomic_json import AtomicJsonError, atomic_write_json_cas, json_file_sha256
from ..compare_gate import COMPARE_GATE_POLICY_SCHEMA_VERSION, CompareGatePolicyError, evaluate_compare_gate, load_compare_gate_policy
from ..improvement_gate import IMPROVEMENT_LEDGER_GATE_POLICY_SCHEMA_VERSION, ImprovementLedgerGateError, ImprovementLedgerGatePolicyError, evaluate_improvement_ledger_gate, load_improvement_ledger_gate_policy
from ..promotion_gate import PROMOTION_LEDGER_GATE_POLICY_SCHEMA_VERSION, PromotionLedgerGateError, PromotionLedgerGatePolicyError, evaluate_promotion_ledger_gate, load_promotion_ledger_gate_policy
from pathlib import Path
from ..reviewed_gate import REVIEWED_GATE_POLICY_SCHEMA_VERSION, ReviewedGateError, ReviewedGatePolicyError, build_reviewed_export_source_artifact, evaluate_reviewed_gate, load_reviewed_gate_policy, snapshot_reviewed_export
from ..runtime_adapter_router import RuntimeAdapterRouterError, build_adapter_route_decision, build_tool_capability_selection
from ..suite_gate import SUITE_GATE_POLICY_SCHEMA_VERSION, SuiteGateError, SuiteGatePolicyError, evaluate_suite_gate, load_gate_policy
from ..training_gate import TRAINING_GATE_POLICY_SCHEMA_VERSION, TrainingGatePolicyError, evaluate_training_gate, load_training_gate_policy
import argparse
import json
import os
from ..path_safety import assert_output_does_not_alias_sources, assert_output_outside_source_directories, locked_owned_output_directory, output_directory_lock_is_held, path_has_symlink_component, remove_directory_tree_if_identity


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ArtifactError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_new_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise AtomicJsonError(f"refusing to overwrite existing JSON target: {path}")
    atomic_write_json_cas(path, payload, expected_sha256=None)


def _artifact_ref_for_output_relative_file(source_path: Path, output_path: Path, *, label: str) -> dict[str, Any]:
    if path_has_symlink_component(source_path, include_leaf=True):
        raise RuntimeAdapterRouterError(f"{label} path must not contain symlink components")
    if not source_path.exists() or not source_path.is_file():
        raise RuntimeAdapterRouterError(f"{label} path must be an existing file")
    output_dir = output_path.parent
    relative = os.path.relpath(source_path.resolve(), output_dir.resolve())
    if not _is_safe_runtime_router_ref(relative):
        raise RuntimeAdapterRouterError(f"{label} path must be safely referenceable relative to route output")
    return {
        "path": relative,
        "sha256": json_file_sha256(source_path),
        "size_bytes": source_path.stat().st_size,
    }


def _is_safe_runtime_router_ref(value: str) -> bool:
    path = Path(value)
    return (
        bool(value)
        and not path.is_absolute()
        and not _is_windows_absolute(value)
        and "\\" not in value
        and ".." not in path.parts
        and all(not part.startswith("~") for part in path.parts)
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(rendered, encoding="utf-8")


def _rate_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number from 0.0 to 1.0") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be from 0.0 to 1.0")
    return parsed


def _score_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number from 0 to 100") from exc
    if not 0.0 <= parsed <= 100.0:
        raise argparse.ArgumentTypeError("must be from 0 to 100")
    return parsed


def _non_negative_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_int_arg(value: str) -> int:
    parsed = _non_negative_int_arg(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_float_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative number")
    return parsed


def _metadata_arg(value: str) -> tuple[str, str]:
    key, separator, raw_value = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("must be KEY=VALUE")
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("metadata key must be non-empty")
    if any(char.isspace() for char in key):
        raise argparse.ArgumentTypeError("metadata key must not contain whitespace")
    return key, raw_value


def _key_path_arg(value: str) -> tuple[str, str]:
    key, separator, raw_path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("must be KEY=PATH")
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("key must be non-empty")
    if any(char.isspace() for char in key):
        raise argparse.ArgumentTypeError("key must not contain whitespace")
    if not raw_path:
        raise argparse.ArgumentTypeError("path must be non-empty")
    return key, raw_path


def _state_set_arg(value: str) -> tuple[str, Any]:
    key, separator, raw_value = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("must be PATH=VALUE")
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("path must be non-empty")
    try:
        parsed: Any = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = raw_value
    return key, parsed


def _text_set_arg(value: str) -> tuple[str, str]:
    key, separator, raw_value = value.partition("=")
    if not separator or not key.strip() or not raw_value:
        raise argparse.ArgumentTypeError("must be KEY=VALUE")
    return key.strip(), raw_value


def _metadata_options(items: list[tuple[str, str]]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key, value in items:
        metadata[key] = value
    return metadata


def _write_score_outputs(scorecard: dict[str, Any], args: argparse.Namespace) -> None:
    if getattr(args, "junit_out", None):
        write_junit(scorecard, args.junit_out)
    if getattr(args, "markdown_out", None):
        write_markdown_summary(scorecard, args.markdown_out)


def _gate_suite_options(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_gate_policy(args.policy) if args.policy else {}
    return {
        "policy_path": _display_path(Path(args.policy), args.preserve_paths) if args.policy else None,
        "policy_description": policy.get("description"),
        "min_pass_rate": args.min_pass_rate if args.min_pass_rate is not None else policy.get("min_pass_rate"),
        "min_average_score": args.min_average_score if args.min_average_score is not None else policy.get("min_average_score"),
        "max_failed": args.max_failed if args.max_failed is not None else policy.get("max_failed"),
        "max_errors": args.max_errors if args.max_errors is not None else policy.get("max_errors", 0),
        "max_critical_failures": (
            args.max_critical_failures if args.max_critical_failures is not None else policy.get("max_critical_failures")
        ),
        "forbid_failed_rules": _merge_gate_rule_ids(policy.get("forbid_failed_rules", []), args.forbid_failed_rule),
        "forbid_critical_rules": _merge_gate_rule_ids(policy.get("forbid_critical_rules", []), args.forbid_critical_rule),
        "task_family_gates": policy.get("task_family_gates", []),
    }


def _gate_policy_summary(options: dict[str, Any]) -> dict[str, Any]:
    effective_fields = (
        "min_pass_rate",
        "min_average_score",
        "max_failed",
        "max_errors",
        "max_critical_failures",
        "forbid_failed_rules",
        "forbid_critical_rules",
        "task_family_gates",
    )
    effective = {
        field: options[field]
        for field in effective_fields
        if options.get(field) is not None and options.get(field) != []
    }
    summary: dict[str, Any] = {
        "schema_version": SUITE_GATE_POLICY_SCHEMA_VERSION,
        "path": options["policy_path"],
        "effective": effective,
    }
    if options.get("policy_description"):
        summary["description"] = options["policy_description"]
    return summary


def _training_gate_options(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_training_gate_policy(args.policy) if args.policy else {}
    return {
        "policy_path": _display_path(Path(args.policy), args.preserve_paths) if args.policy else None,
        "policy_description": policy.get("description"),
        "require_valid_export": False if args.skip_validation else policy.get("require_valid_export", True),
        "strict_validation": bool(args.strict_validation or policy.get("strict_validation", False)),
        "min_episodes": args.min_episodes if args.min_episodes is not None else policy.get("min_episodes"),
        "min_pass_rate": args.min_pass_rate if args.min_pass_rate is not None else policy.get("min_pass_rate"),
        "min_average_score": args.min_average_score if args.min_average_score is not None else policy.get("min_average_score"),
        "min_preferences": args.min_preferences if args.min_preferences is not None else policy.get("min_preferences"),
        "min_sft": args.min_sft if args.min_sft is not None else policy.get("min_sft"),
        "min_dpo": args.min_dpo if args.min_dpo is not None else policy.get("min_dpo"),
        "min_reward_model": args.min_reward_model if args.min_reward_model is not None else policy.get("min_reward_model"),
        "min_step_rewards": args.min_step_rewards if args.min_step_rewards is not None else policy.get("min_step_rewards"),
        "min_task_completion_configured": (
            args.min_task_completion_configured
            if args.min_task_completion_configured is not None
            else policy.get("min_task_completion_configured")
        ),
        "min_task_completion_complete": (
            args.min_task_completion_complete
            if args.min_task_completion_complete is not None
            else policy.get("min_task_completion_complete")
        ),
        "max_task_completion_incomplete": (
            args.max_task_completion_incomplete
            if args.max_task_completion_incomplete is not None
            else policy.get("max_task_completion_incomplete")
        ),
        "min_task_completion_check_pass_rate": (
            args.min_task_completion_check_pass_rate
            if args.min_task_completion_check_pass_rate is not None
            else policy.get("min_task_completion_check_pass_rate")
        ),
        "min_source_fingerprint_rate": (
            args.min_source_fingerprint_rate
            if args.min_source_fingerprint_rate is not None
            else policy.get("min_source_fingerprint_rate")
        ),
        "max_unverified_source_fingerprints": (
            args.max_unverified_source_fingerprints
            if args.max_unverified_source_fingerprints is not None
            else policy.get("max_unverified_source_fingerprints")
        ),
        "min_trainer_view_source_fingerprint_rate": (
            args.min_trainer_view_source_fingerprint_rate
            if args.min_trainer_view_source_fingerprint_rate is not None
            else policy.get("min_trainer_view_source_fingerprint_rate")
        ),
        "max_unverified_trainer_view_source_fingerprints": (
            args.max_unverified_trainer_view_source_fingerprints
            if args.max_unverified_trainer_view_source_fingerprints is not None
            else policy.get("max_unverified_trainer_view_source_fingerprints")
        ),
        "min_trace_average_events": (
            args.min_trace_average_events
            if args.min_trace_average_events is not None
            else policy.get("min_trace_average_events")
        ),
        "min_trace_event_type_count": (
            args.min_trace_event_type_count
            if args.min_trace_event_type_count is not None
            else policy.get("min_trace_event_type_count")
        ),
        "min_trace_final_answer_rate": (
            args.min_trace_final_answer_rate
            if args.min_trace_final_answer_rate is not None
            else policy.get("min_trace_final_answer_rate")
        ),
        "min_trace_tool_or_api_rate": (
            args.min_trace_tool_or_api_rate
            if args.min_trace_tool_or_api_rate is not None
            else policy.get("min_trace_tool_or_api_rate")
        ),
        "max_trace_empty_final_answers": (
            args.max_trace_empty_final_answers
            if args.max_trace_empty_final_answers is not None
            else policy.get("max_trace_empty_final_answers")
        ),
        "max_trace_risk_count": (
            args.max_trace_risk_count
            if args.max_trace_risk_count is not None
            else policy.get("max_trace_risk_count")
        ),
        "min_split_task_families": (
            args.min_split_task_families
            if args.min_split_task_families is not None
            else policy.get("min_split_task_families")
        ),
        "min_train_episodes": args.min_train_episodes if args.min_train_episodes is not None else policy.get("min_train_episodes"),
        "min_validation_episodes": (
            args.min_validation_episodes
            if args.min_validation_episodes is not None
            else policy.get("min_validation_episodes")
        ),
        "min_test_episodes": args.min_test_episodes if args.min_test_episodes is not None else policy.get("min_test_episodes"),
        "require_family_exclusive_splits": bool(
            args.require_family_exclusive_splits or policy.get("require_family_exclusive_splits", False)
        ),
        "max_quality_flags": args.max_quality_flags if args.max_quality_flags is not None else policy.get("max_quality_flags"),
        "forbid_quality_flags": _merge_unique_strings(policy.get("forbid_quality_flags", []), args.forbid_quality_flag),
        "forbid_quality_severities": _merge_unique_strings(
            policy.get("forbid_quality_severities", []),
            args.forbid_quality_severity,
        ),
        "require_task_families": _merge_unique_strings(policy.get("require_task_families", []), args.require_task_family),
        "require_trace_event_types": _merge_unique_strings(policy.get("require_trace_event_types", []), args.require_trace_event_type),
        "task_family_gates": policy.get("task_family_gates", []),
    }


def _goal3_training_gate_args(args: argparse.Namespace, training_export_dir: Path, gate_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        training_export=str(training_export_dir),
        policy=args.policy,
        out=str(gate_path),
        strict_validation=args.strict,
        skip_validation=False,
        min_episodes=None,
        min_pass_rate=None,
        min_average_score=None,
        min_preferences=None,
        min_sft=None,
        min_dpo=None,
        min_reward_model=None,
        min_step_rewards=None,
        min_task_completion_configured=None,
        min_task_completion_complete=None,
        max_task_completion_incomplete=None,
        min_task_completion_check_pass_rate=None,
        min_source_fingerprint_rate=None,
        max_unverified_source_fingerprints=None,
        min_trainer_view_source_fingerprint_rate=None,
        max_unverified_trainer_view_source_fingerprints=None,
        min_trace_average_events=None,
        min_trace_event_type_count=None,
        min_trace_final_answer_rate=None,
        min_trace_tool_or_api_rate=None,
        max_trace_empty_final_answers=None,
        max_trace_risk_count=None,
        require_trace_event_type=[],
        min_split_task_families=None,
        min_train_episodes=None,
        min_validation_episodes=None,
        min_test_episodes=None,
        require_family_exclusive_splits=False,
        max_quality_flags=None,
        forbid_quality_flag=[],
        forbid_quality_severity=[],
        require_task_family=[],
        preserve_paths=args.preserve_paths,
    )


def _training_gate_policy_summary(options: dict[str, Any]) -> dict[str, Any]:
    effective_fields = (
        "min_episodes",
        "min_pass_rate",
        "min_average_score",
        "min_preferences",
        "min_sft",
        "min_dpo",
        "min_reward_model",
        "min_step_rewards",
        "min_task_completion_configured",
        "min_task_completion_complete",
        "max_task_completion_incomplete",
        "min_task_completion_check_pass_rate",
        "min_source_fingerprint_rate",
        "max_unverified_source_fingerprints",
        "min_trainer_view_source_fingerprint_rate",
        "max_unverified_trainer_view_source_fingerprints",
        "min_trace_average_events",
        "min_trace_event_type_count",
        "min_trace_final_answer_rate",
        "min_trace_tool_or_api_rate",
        "max_trace_empty_final_answers",
        "max_trace_risk_count",
        "min_split_task_families",
        "min_train_episodes",
        "min_validation_episodes",
        "min_test_episodes",
        "require_family_exclusive_splits",
        "max_quality_flags",
        "forbid_quality_flags",
        "forbid_quality_severities",
        "require_task_families",
        "require_trace_event_types",
        "task_family_gates",
        "require_valid_export",
        "strict_validation",
    )
    effective = {
        field: options[field]
        for field in effective_fields
        if options.get(field) is not None and options.get(field) != []
    }
    summary: dict[str, Any] = {
        "schema_version": TRAINING_GATE_POLICY_SCHEMA_VERSION,
        "path": options["policy_path"],
        "effective": effective,
    }
    if options.get("policy_description"):
        summary["description"] = options["policy_description"]
    return summary


def _reviewed_gate_options(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_reviewed_gate_policy(args.policy) if args.policy else {}
    return {
        "policy_path": _display_path(Path(args.policy), args.preserve_paths) if args.policy else None,
        "policy_description": policy.get("description"),
        "min_reviewed_labels": (
            args.min_reviewed_labels if args.min_reviewed_labels is not None else policy.get("min_reviewed_labels")
        ),
        "min_accepted": args.min_accepted if args.min_accepted is not None else policy.get("min_accepted"),
        "min_rejected": args.min_rejected if args.min_rejected is not None else policy.get("min_rejected"),
        "min_sft": args.min_sft if args.min_sft is not None else policy.get("min_sft"),
        "min_reward_model": args.min_reward_model if args.min_reward_model is not None else policy.get("min_reward_model"),
        "min_preferences": args.min_preferences if args.min_preferences is not None else policy.get("min_preferences"),
        "min_dpo": args.min_dpo if args.min_dpo is not None else policy.get("min_dpo"),
        "min_high_confidence_labels": (
            args.min_high_confidence_labels
            if args.min_high_confidence_labels is not None
            else policy.get("min_high_confidence_labels")
        ),
        "min_medium_or_high_confidence_labels": (
            args.min_medium_or_high_confidence_labels
            if args.min_medium_or_high_confidence_labels is not None
            else policy.get("min_medium_or_high_confidence_labels")
        ),
        "max_needs_review": args.max_needs_review if args.max_needs_review is not None else policy.get("max_needs_review"),
        "max_low_confidence_labels": (
            args.max_low_confidence_labels
            if args.max_low_confidence_labels is not None
            else policy.get("max_low_confidence_labels")
        ),
        "max_unknown_confidence_labels": (
            args.max_unknown_confidence_labels
            if args.max_unknown_confidence_labels is not None
            else policy.get("max_unknown_confidence_labels")
        ),
        "forbid_labels": _merge_unique_strings(policy.get("forbid_labels", []), args.forbid_label),
        "require_task_families": _merge_unique_strings(policy.get("require_task_families", []), args.require_task_family),
        "require_valid_export": False if args.skip_validation else policy.get("require_valid_export", True),
        "strict_validation": bool(args.strict_validation or policy.get("strict_validation", False)),
    }


def _reviewed_gate_policy_summary(options: dict[str, Any]) -> dict[str, Any]:
    effective_fields = (
        "min_reviewed_labels",
        "min_accepted",
        "min_rejected",
        "min_sft",
        "min_reward_model",
        "min_preferences",
        "min_dpo",
        "min_high_confidence_labels",
        "min_medium_or_high_confidence_labels",
        "max_needs_review",
        "max_low_confidence_labels",
        "max_unknown_confidence_labels",
        "forbid_labels",
        "require_task_families",
        "require_valid_export",
        "strict_validation",
    )
    effective = {
        field: options[field]
        for field in effective_fields
        if options.get(field) is not None and options.get(field) != []
    }
    summary: dict[str, Any] = {
        "schema_version": REVIEWED_GATE_POLICY_SCHEMA_VERSION,
        "path": options["policy_path"],
        "effective": effective,
    }
    if options.get("policy_description"):
        summary["description"] = options["policy_description"]
    return summary


def _compare_gate_options(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_compare_gate_policy(args.policy) if args.policy else {}
    return {
        "policy_path": _display_path(Path(args.policy), args.preserve_paths) if args.policy else None,
        "policy_description": policy.get("description"),
        "require_valid_export": False if args.skip_validation else policy.get("require_valid_export", True),
        "strict_validation": bool(args.strict_validation or policy.get("strict_validation", False)),
        "min_pairs": args.min_pairs if args.min_pairs is not None else policy.get("min_pairs"),
        "min_dpo": args.min_dpo if args.min_dpo is not None else policy.get("min_dpo"),
        "min_candidate_wins": (
            args.min_candidate_wins if args.min_candidate_wins is not None else policy.get("min_candidate_wins")
        ),
        "min_task_completion_improvements": (
            args.min_task_completion_improvements
            if args.min_task_completion_improvements is not None
            else policy.get("min_task_completion_improvements")
        ),
        "max_baseline_wins": (
            args.max_baseline_wins if args.max_baseline_wins is not None else policy.get("max_baseline_wins")
        ),
        "max_task_completion_regressions": (
            args.max_task_completion_regressions
            if args.max_task_completion_regressions is not None
            else policy.get("max_task_completion_regressions")
        ),
        "max_skipped_pairs": args.max_skipped_pairs if args.max_skipped_pairs is not None else policy.get("max_skipped_pairs"),
        "max_contract_drifts": (
            args.max_contract_drifts if args.max_contract_drifts is not None else policy.get("max_contract_drifts")
        ),
        "max_unverified_contracts": (
            args.max_unverified_contracts
            if args.max_unverified_contracts is not None
            else policy.get("max_unverified_contracts")
        ),
        "require_scenarios": _merge_unique_strings(policy.get("require_scenarios", []), args.require_scenario),
        "require_candidate_win_scenarios": _merge_unique_strings(
            policy.get("require_candidate_win_scenarios", []),
            args.require_candidate_win_scenario,
        ),
        "require_task_completion_improvement_scenarios": _merge_unique_strings(
            policy.get("require_task_completion_improvement_scenarios", []),
            args.require_task_completion_improvement_scenario,
        ),
        "forbid_regression_scenarios": _merge_unique_strings(
            policy.get("forbid_regression_scenarios", []),
            args.forbid_regression_scenario,
        ),
        "forbid_task_completion_regression_scenarios": _merge_unique_strings(
            policy.get("forbid_task_completion_regression_scenarios", []),
            args.forbid_task_completion_regression_scenario,
        ),
        "require_rule_fixes": _merge_unique_strings(policy.get("require_rule_fixes", []), args.require_rule_fix),
        "forbid_rule_regressions": _merge_unique_strings(
            policy.get("forbid_rule_regressions", []),
            args.forbid_rule_regression,
        ),
        "forbid_new_critical_failures": _merge_unique_strings(
            policy.get("forbid_new_critical_failures", []),
            args.forbid_new_critical_failure,
        ),
        "task_family_gates": policy.get("task_family_gates", []),
    }


def _compare_gate_policy_summary(options: dict[str, Any]) -> dict[str, Any]:
    effective_fields = (
        "min_pairs",
        "min_dpo",
        "min_candidate_wins",
        "min_task_completion_improvements",
        "max_baseline_wins",
        "max_task_completion_regressions",
        "max_skipped_pairs",
        "max_contract_drifts",
        "max_unverified_contracts",
        "require_scenarios",
        "require_candidate_win_scenarios",
        "require_task_completion_improvement_scenarios",
        "forbid_regression_scenarios",
        "forbid_task_completion_regression_scenarios",
        "require_rule_fixes",
        "forbid_rule_regressions",
        "forbid_new_critical_failures",
        "task_family_gates",
        "require_valid_export",
        "strict_validation",
    )
    effective = {
        field: options[field]
        for field in effective_fields
        if options.get(field) is not None and options.get(field) != []
    }
    summary: dict[str, Any] = {
        "schema_version": COMPARE_GATE_POLICY_SCHEMA_VERSION,
        "path": options["policy_path"],
        "effective": effective,
    }
    if options.get("policy_description"):
        summary["description"] = options["policy_description"]
    return summary


def _action_ledger_gate_options(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_action_ledger_gate_policy(args.policy) if args.policy else {}
    return {
        "policy_path": _display_path(Path(args.policy), args.preserve_paths) if args.policy else None,
        "policy_description": policy.get("description"),
        "min_bundles": args.min_bundles if args.min_bundles is not None else policy.get("min_bundles"),
        "max_open_actions": args.max_open_actions if args.max_open_actions is not None else policy.get("max_open_actions"),
        "max_new_actions": args.max_new_actions if args.max_new_actions is not None else policy.get("max_new_actions"),
        "max_recurring_actions": (
            args.max_recurring_actions if args.max_recurring_actions is not None else policy.get("max_recurring_actions")
        ),
        "min_resolved_actions": (
            args.min_resolved_actions if args.min_resolved_actions is not None else policy.get("min_resolved_actions")
        ),
        "forbid_open_priorities": _merge_unique_strings(policy.get("forbid_open_priorities", []), args.forbid_open_priority),
        "forbid_open_actions": _merge_unique_strings(policy.get("forbid_open_actions", []), args.forbid_open_action),
        "require_resolved_actions": _merge_unique_strings(policy.get("require_resolved_actions", []), args.require_resolved_action),
    }


def _action_ledger_gate_policy_summary(options: dict[str, Any]) -> dict[str, Any]:
    effective_fields = (
        "min_bundles",
        "max_open_actions",
        "max_new_actions",
        "max_recurring_actions",
        "min_resolved_actions",
        "forbid_open_priorities",
        "forbid_open_actions",
        "require_resolved_actions",
    )
    effective = {
        field: options[field]
        for field in effective_fields
        if options.get(field) is not None and options.get(field) != []
    }
    summary: dict[str, Any] = {
        "schema_version": ACTION_LEDGER_GATE_POLICY_SCHEMA_VERSION,
        "path": options["policy_path"],
        "effective": effective,
    }
    if options.get("policy_description"):
        summary["description"] = options["policy_description"]
    return summary


def _improvement_ledger_gate_options(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_improvement_ledger_gate_policy(args.policy) if args.policy else {}
    return {
        "policy_path": _display_path(Path(args.policy), args.preserve_paths) if args.policy else None,
        "policy_description": policy.get("description"),
        "min_plans": args.min_plans if args.min_plans is not None else policy.get("min_plans"),
        "max_open_work_items": (
            args.max_open_work_items if args.max_open_work_items is not None else policy.get("max_open_work_items")
        ),
        "max_new_work_items": (
            args.max_new_work_items if args.max_new_work_items is not None else policy.get("max_new_work_items")
        ),
        "max_recurring_work_items": (
            args.max_recurring_work_items
            if args.max_recurring_work_items is not None
            else policy.get("max_recurring_work_items")
        ),
        "min_resolved_work_items": (
            args.min_resolved_work_items
            if args.min_resolved_work_items is not None
            else policy.get("min_resolved_work_items")
        ),
        "max_critical_open_work_items": (
            args.max_critical_open_work_items
            if args.max_critical_open_work_items is not None
            else policy.get("max_critical_open_work_items")
        ),
        "max_high_open_work_items": (
            args.max_high_open_work_items
            if args.max_high_open_work_items is not None
            else policy.get("max_high_open_work_items")
        ),
        "forbid_open_priorities": _merge_unique_strings(policy.get("forbid_open_priorities", []), args.forbid_open_priority),
        "forbid_open_categories": _merge_unique_strings(policy.get("forbid_open_categories", []), args.forbid_open_category),
        "forbid_open_work_keys": _merge_unique_strings(policy.get("forbid_open_work_keys", []), args.forbid_open_work_key),
        "require_open_work_keys": _merge_unique_strings(policy.get("require_open_work_keys", []), args.require_open_work_key),
        "require_resolved_work_keys": _merge_unique_strings(
            policy.get("require_resolved_work_keys", []),
            args.require_resolved_work_key,
        ),
    }


def _improvement_ledger_gate_policy_summary(options: dict[str, Any]) -> dict[str, Any]:
    effective_fields = (
        "min_plans",
        "max_open_work_items",
        "max_new_work_items",
        "max_recurring_work_items",
        "min_resolved_work_items",
        "max_critical_open_work_items",
        "max_high_open_work_items",
        "forbid_open_priorities",
        "forbid_open_categories",
        "forbid_open_work_keys",
        "require_open_work_keys",
        "require_resolved_work_keys",
    )
    effective = {
        field: options[field]
        for field in effective_fields
        if options.get(field) is not None and options.get(field) != []
    }
    summary: dict[str, Any] = {
        "schema_version": IMPROVEMENT_LEDGER_GATE_POLICY_SCHEMA_VERSION,
        "path": options["policy_path"],
        "effective": effective,
    }
    if options.get("policy_description"):
        summary["description"] = options["policy_description"]
    return summary


def _promotion_ledger_gate_options(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_promotion_ledger_gate_policy(args.policy) if args.policy else {}
    return {
        "policy_path": _display_path(Path(args.policy), args.preserve_paths) if args.policy else None,
        "policy_description": policy.get("description"),
        "min_decisions": args.min_decisions if args.min_decisions is not None else policy.get("min_decisions"),
        "min_allowed_count": args.min_allowed_count if args.min_allowed_count is not None else policy.get("min_allowed_count"),
        "max_blocked_count": args.max_blocked_count if args.max_blocked_count is not None else policy.get("max_blocked_count"),
        "max_blocked_rate": args.max_blocked_rate if args.max_blocked_rate is not None else policy.get("max_blocked_rate"),
        "min_consecutive_allowed": (
            args.min_consecutive_allowed
            if args.min_consecutive_allowed is not None
            else policy.get("min_consecutive_allowed")
        ),
        "max_consecutive_blocked": (
            args.max_consecutive_blocked
            if args.max_consecutive_blocked is not None
            else policy.get("max_consecutive_blocked")
        ),
        "max_failed_decisions": (
            args.max_failed_decisions if args.max_failed_decisions is not None else policy.get("max_failed_decisions")
        ),
        "require_latest_recommendation": (
            args.require_latest_recommendation
            if args.require_latest_recommendation is not None
            else policy.get("require_latest_recommendation")
        ),
        "require_latest_passed": args.require_latest_passed or bool(policy.get("require_latest_passed")),
        "require_source_recommendations": _merge_unique_strings(
            policy.get("require_source_recommendations", []),
            args.require_source_recommendation,
        ),
        "forbid_source_recommendations": _merge_unique_strings(
            policy.get("forbid_source_recommendations", []),
            args.forbid_source_recommendation,
        ),
    }


def _promotion_ledger_gate_policy_summary(options: dict[str, Any]) -> dict[str, Any]:
    effective_fields = (
        "min_decisions",
        "min_allowed_count",
        "max_blocked_count",
        "max_blocked_rate",
        "min_consecutive_allowed",
        "max_consecutive_blocked",
        "max_failed_decisions",
        "require_latest_recommendation",
        "require_latest_passed",
        "require_source_recommendations",
        "forbid_source_recommendations",
    )
    effective = {
        field: options[field]
        for field in effective_fields
        if options.get(field) is not None and options.get(field) != []
    }
    summary: dict[str, Any] = {
        "schema_version": PROMOTION_LEDGER_GATE_POLICY_SCHEMA_VERSION,
        "path": options["policy_path"],
        "effective": effective,
    }
    if options.get("policy_description"):
        summary["description"] = options["policy_description"]
    return summary


def _merge_gate_rule_ids(policy_values: Any, cli_values: list[str]) -> list[str]:
    return _merge_unique_strings(policy_values, cli_values)


def _merge_unique_strings(policy_values: Any, cli_values: list[str]) -> list[str]:
    merged: list[str] = []
    for value in [*(policy_values or []), *cli_values]:
        if value not in merged:
            merged.append(value)
    return merged


def _lineage_record_path(lineage: dict[str, Any], collection_name: str, record_name: str) -> str | None:
    records = lineage.get(collection_name)
    if not isinstance(records, list):
        return None
    for record in records:
        if isinstance(record, dict) and record.get("name") == record_name and isinstance(record.get("path"), str):
            return record["path"]
    return None


def _display_path(path: Path, preserve_paths: bool = False) -> str:
    raw = str(path)
    if preserve_paths:
        return raw
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return f"<redacted:{resolved.name}>"


def _display_path_for_output_source(path: Path, out_path: Path | None, preserve_paths: bool = False) -> str:
    if preserve_paths or out_path is None:
        return _display_path(path, preserve_paths)
    raw = str(path)
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    resolved = path.resolve()
    out_dir = out_path.parent.resolve()
    return os.path.relpath(resolved, out_dir)


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"
