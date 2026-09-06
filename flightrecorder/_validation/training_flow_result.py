"""Extracted validation implementation."""

from __future__ import annotations

import json
import shlex
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Any
from ..agentic_training_result import AGENTIC_TRAINING_RESULT_SCHEMA_VERSION, BLOCK_REGISTRATION_RECOMMENDATION, FAILURE_CLASSES, OUTPUT_ARTIFACT_ROLES, RECOVERABLE_FAILURE_CLASSES, REGISTER_FAILURE_RECOMMENDATION, REGISTER_RESULT_RECOMMENDATION, RESULT_STATUSES
from ..agentic_training_flow import AGENTIC_TRAINING_FLOW_SCHEMA_VERSION, BLOCKED_FLOW_MODE_CATEGORIES, BLOCKED_TRAINER_FLOW_MODES, BLOCKED_TRAINER_FLOW_STAGES, EXECUTABLE_FLOW_MODES, EXECUTABLE_STAGES, FLOW_BLOCK_RECOMMENDATION, FLOW_READY_RECOMMENDATION
from ..agentic_training_runtime import AGENTIC_TRAINING_RUNTIME_PREFLIGHT_SCHEMA_VERSION, PLAN_READY_RECOMMENDATION, RUNTIME_BLOCK_RECOMMENDATION, RUNTIME_READY_RECOMMENDATION
from ..source_contract import get_active_opaque_output_attestation, inspect_artifact_source
from ..agentic_training_plan import ADVANCED_REWARD_MODES as PLAN_ADVANCED_REWARD_MODES, AGENTIC_TRAINING_PLAN_SCHEMA_VERSION, DEFAULT_EXECUTABLE_MODES as PLAN_DEFAULT_EXECUTABLE_MODES, FUTURE_RL_MODES as PLAN_FUTURE_RL_MODES, MODE_STAGE_SEQUENCES as PLAN_MODE_STAGE_SEQUENCES, MODE_VIEW_REQUIREMENTS as PLAN_MODE_VIEW_REQUIREMENTS, SUPPORTED_MODES as PLAN_SUPPORTED_MODES
from ..path_safety import path_has_symlink_component as _path_has_symlink_component, resolve_artifact_reference_path
from ..trainer_consumer_plan import TRAINER_CONSUMER_PLAN_SCHEMA_VERSION
from ..hashing import sha256_file as _sha256
from .primitives import ValidationTarget, _is_non_negative_int, _is_sha256, _is_string_list, _looks_absolute, _read_object, _require_equal, _sha256, _validate_allowed_keys, _validate_gate_like_checks
from .training_plan_runtime import _AGENTIC_TRAINING_PLAN_READY_RECOMMENDATION

def validate_agentic_training_flow(path: str | Path) -> ValidationTarget:
    """Validate a delegated agentic trainer-flow receipt."""
    flow_path = Path(path)
    target = ValidationTarget("agentic_training_flow", str(flow_path))
    flow = _read_object(flow_path, target, "agentic_training_flow.json")
    if flow is not None:
        _validate_agentic_training_flow(flow, target, flow_path)
    return target

def validate_agentic_training_result(path: str | Path) -> ValidationTarget:
    """Validate a side-effect-free agentic training result receipt."""
    result_path = Path(path)
    target = ValidationTarget("agentic_training_result", str(result_path))
    result = _read_object(result_path, target, "agentic_training_result.json")
    if result is not None:
        _validate_agentic_training_result(result, target, result_path)
    return target

_AGENTIC_TRAINING_FLOW_KEYS = {
    "schema_version",
    "created_at",
    "flow_path",
    "flow_id",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "blocked_reasons",
    "mode_contract_check",
    "flow_mode_gate",
    "source_artifacts",
    "delegated_flow",
    "metrics",
    "execution_boundary",
    "handoff_contract",
    "notes",
}

_AGENTIC_TRAINING_FLOW_CHECK_KEYS = {"id", "passed", "actual", "expected", "summary"}

_AGENTIC_TRAINING_FLOW_SOURCE_ARTIFACT_KEYS = {
    "agentic_training_plan",
    "agentic_training_runtime_preflight",
    "trainer_consumer_plan",
}

_AGENTIC_TRAINING_FLOW_SOURCE_REF_KEYS = {
    "role",
    "path",
    "exists",
    "regular_file",
    "schema_version",
    "passed",
    "recommendation",
    "sha256",
    "size_bytes",
}

_AGENTIC_TRAINING_FLOW_DELEGATED_KEYS = {"mode", "backend", "stage_sequence", "stages", "command"}

_AGENTIC_TRAINING_FLOW_STAGE_KEYS = {
    "stage_index",
    "stage_id",
    "view_name",
    "view_path",
    "view_schema_version",
    "row_count",
    "view_ready",
}

_AGENTIC_TRAINING_FLOW_COMMAND_KEYS = {
    "execution_cwd",
    "archive_root",
    "external_code_root",
    "command_argv",
    "command_shell",
    "command_arg_count",
    "trainer_input_count",
    "external_code_file_count",
}

_AGENTIC_TRAINING_FLOW_MODE_CONTRACT_KEYS = {
    "mode",
    "category",
    "present",
    "mode_matches_plan",
    "planning_gate_open",
    "planning_required_flag",
    "data_requirement_count",
    "unsatisfied_data_requirement_ids",
    "reward_contract",
    "side_effect_boundary",
    "external_runner_contract",
    "passed",
    "error_count",
    "errors",
}

_AGENTIC_TRAINING_FLOW_REWARD_CONTRACT_KEYS = {
    "kind",
    "required",
    "external_runner_must_supply",
    "external_runner_must_validate",
    "flight_recorder_supplies_callable",
    "may_call_paid_services_by_default",
    "may_require_secrets_by_default",
    "must_not_use_unredacted_traces",
    "callable_signature",
}

_AGENTIC_TRAINING_FLOW_SIDE_EFFECT_KEYS = {
    "dry_run_only",
    "training_started",
    "cloud_jobs_started",
    "model_downloads_started",
    "paid_model_grader_calls_started",
    "weights_updated",
    "provider_credentials_required_by_flight_recorder",
}

_AGENTIC_TRAINING_FLOW_EXTERNAL_RUNNER_KEYS = {
    "runner_owns_execution",
    "runner_must_revalidate_inputs",
    "runner_must_require_recommendation",
    "runner_must_validate_reward_contract",
    "runner_must_block_unredacted_traces",
}

_AGENTIC_TRAINING_FLOW_MODE_GATE_KEYS = {
    "mode",
    "category",
    "executable_by_default",
    "blocked_by_default",
    "promotion_required",
    "promotion_status",
    "required_plan_opt_in_flag",
    "mode_contract_ready",
    "reward_contract_kind",
    "external_runner_must_supply_reward",
    "external_runner_must_validate_reward",
    "reason",
}

_AGENTIC_TRAINING_FLOW_METRIC_KEYS = {
    "check_count",
    "failed_check_count",
    "stage_count",
    "executable_stage_count",
    "selected_view_count",
    "command_arg_count",
    "trainer_input_count",
    "external_code_file_count",
}

_AGENTIC_TRAINING_FLOW_BOUNDARY_KEYS = {
    "flow_plan_only",
    "training_started",
    "trainer_command_executed",
    "subprocess_started",
    "cloud_jobs_started",
    "model_downloads_started",
    "weights_updated_by_flight_recorder",
    "trainer_modules_imported",
    "credential_values_recorded",
}

_AGENTIC_TRAINING_FLOW_HANDOFF_KEYS = {
    "runner_owns_execution",
    "runner_must_require_recommendation",
    "runner_must_emit_result_schema",
    "requires_runtime_preflight",
    "requires_trainer_consumer_plan",
    "requires_mode_contract",
    "requires_mode_contract_ready",
    "requires_registered_model_and_dataset",
    "requires_redacted_dataset",
    "executable_modes",
    "blocked_modes",
    "blocked_mode_categories",
    "blocked_mode_stages",
    "flight_recorder_executed_trainer",
}

def _validate_agentic_training_flow(flow: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(flow, _AGENTIC_TRAINING_FLOW_KEYS, target, "agentic_training_flow")
    _require_equal(flow, "schema_version", AGENTIC_TRAINING_FLOW_SCHEMA_VERSION, target, prefix="agentic_training_flow.")
    checks = flow.get("checks")
    if not isinstance(checks, list):
        target.errors.append("agentic_training_flow.checks must be a list.")
        checks = []
    for index, check in enumerate(checks):
        if isinstance(check, dict):
            _validate_allowed_keys(check, _AGENTIC_TRAINING_FLOW_CHECK_KEYS, target, f"agentic_training_flow.checks[{index}]")
    failed_checks = _validate_gate_like_checks(checks, target, "agentic_training_flow.checks")
    if flow.get("check_count") != len(checks):
        target.errors.append(f"agentic_training_flow.check_count expected {len(checks)}, got {flow.get('check_count')!r}.")
    if flow.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"agentic_training_flow.failed_check_count expected {failed_checks}, got {flow.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if not isinstance(flow.get("passed"), bool):
        target.errors.append("agentic_training_flow.passed must be a boolean.")
    elif flow.get("passed") != expected_passed:
        target.errors.append("agentic_training_flow.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = FLOW_READY_RECOMMENDATION if expected_passed else FLOW_BLOCK_RECOMMENDATION
    if flow.get("readiness") != expected_readiness:
        target.errors.append(f"agentic_training_flow.readiness expected {expected_readiness!r}, got {flow.get('readiness')!r}.")
    if flow.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"agentic_training_flow.recommendation expected {expected_recommendation!r}, got {flow.get('recommendation')!r}."
        )
    if not isinstance(flow.get("flow_id"), str) or not flow.get("flow_id"):
        target.errors.append("agentic_training_flow.flow_id must be a non-empty string.")
    flow_path = flow.get("flow_path")
    if not isinstance(flow_path, str):
        target.errors.append("agentic_training_flow.flow_path must be a string.")
    elif flow_path and not _is_safe_training_flow_metadata_path(flow_path):
        target.errors.append("agentic_training_flow.flow_path must be a safe relative path without traversal.")
    if not _is_string_list(flow.get("blocked_reasons")):
        target.errors.append("agentic_training_flow.blocked_reasons must be a list of strings.")
    elif expected_passed and flow.get("blocked_reasons"):
        target.errors.append("agentic_training_flow.blocked_reasons must be empty when passed.")
    elif not expected_passed and len(flow.get("blocked_reasons", [])) != failed_checks:
        target.errors.append("agentic_training_flow.blocked_reasons must match failed_check_count.")

    _validate_agentic_training_flow_sources(flow.get("source_artifacts"), target, source_path)
    flow_counts = _validate_agentic_training_flow_delegated_flow(flow.get("delegated_flow"), target, expected_passed)
    mode_contract_check = _validate_agentic_training_flow_mode_contract_check(
        flow.get("mode_contract_check"),
        target,
        flow_counts.get("mode", ""),
        expected_passed,
    )
    _validate_agentic_training_flow_mode_gate(
        flow.get("flow_mode_gate"),
        target,
        flow_counts.get("mode", ""),
        mode_contract_check,
        expected_passed,
    )
    _validate_agentic_training_flow_metrics(flow.get("metrics"), target, flow_counts, len(checks), failed_checks)
    _validate_agentic_training_flow_boundary(flow.get("execution_boundary"), target)
    _validate_agentic_training_flow_contract(flow.get("handoff_contract"), target)
    if not _is_string_list(flow.get("notes")):
        target.errors.append("agentic_training_flow.notes must be a list of strings.")
    target.details.update(
        {
            "passed": flow.get("passed"),
            "mode": flow_counts.get("mode"),
            "stage_count": flow_counts.get("stage_count"),
            "command_arg_count": flow_counts.get("command_arg_count"),
        }
    )

def _validate_agentic_training_flow_sources(value: Any, target: ValidationTarget, source_path: Path) -> None:
    if not isinstance(value, dict):
        target.errors.append("agentic_training_flow.source_artifacts must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_FLOW_SOURCE_ARTIFACT_KEYS, target, "agentic_training_flow.source_artifacts")
    expected = {
        "agentic_training_plan": AGENTIC_TRAINING_PLAN_SCHEMA_VERSION,
        "agentic_training_runtime_preflight": AGENTIC_TRAINING_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
        "trainer_consumer_plan": TRAINER_CONSUMER_PLAN_SCHEMA_VERSION,
    }
    for role, schema_version in expected.items():
        ref = value.get(role)
        label = f"agentic_training_flow.source_artifacts.{role}"
        if not isinstance(ref, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_agentic_training_flow_source_ref(ref, target, label, role, schema_version, source_path)

def _validate_agentic_training_flow_source_ref(
    ref: dict[str, Any],
    target: ValidationTarget,
    label: str,
    role: str,
    schema_version: str,
    source_path: Path,
) -> None:
    _validate_allowed_keys(ref, _AGENTIC_TRAINING_FLOW_SOURCE_REF_KEYS, target, label)
    if ref.get("role") != role:
        target.errors.append(f"{label}.role must be {role!r}.")
    path_value = ref.get("path")
    if not isinstance(path_value, str) or not path_value:
        target.errors.append(f"{label}.path must be a non-empty string.")
        path_value = ""
    elif not _is_safe_or_redacted_training_flow_path(path_value):
        target.errors.append(f"{label}.path must be a safe relative path without traversal.")
    if ref.get("schema_version") != schema_version:
        target.errors.append(f"{label}.schema_version must be {schema_version!r}.")
    if ref.get("exists") is not True:
        target.errors.append(f"{label}.exists must be true.")
    if ref.get("regular_file") is not True:
        target.errors.append(f"{label}.regular_file must be true.")
    if ref.get("passed") is not True:
        target.errors.append(f"{label}.passed must be true.")
    if not isinstance(ref.get("recommendation"), str) or not ref.get("recommendation"):
        target.errors.append(f"{label}.recommendation must be a non-empty string.")
    if not _is_sha256(ref.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string.")
    if not _is_non_negative_int(ref.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    if isinstance(path_value, str) and path_value and not path_value.startswith("<"):
        source_dir = source_path.resolve().parent
        candidate_path = source_dir / path_value
        current_path = candidate_path.resolve()
        allowed_roots = [source_dir]
        repo_root = _repo_root_for_artifact(source_path)
        if repo_root is not None:
            allowed_roots.append(repo_root)
        if not any(current_path.is_relative_to(root) for root in allowed_roots):
            target.errors.append(f"{label}.path must resolve under the flow artifact directory or repository root.")
            return
        if _path_has_symlink_component(candidate_path, include_leaf=True):
            target.errors.append(f"{label}.path must resolve to a regular non-symlink file.")
            return
        if not current_path.exists():
            target.errors.append(f"{label}.path must resolve to an existing file.")
            return
        if current_path.is_symlink() or not current_path.is_file():
            target.errors.append(f"{label}.path must resolve to a regular file.")
            return
        if _is_non_negative_int(ref.get("size_bytes")) and current_path.stat().st_size != ref.get("size_bytes"):
            target.errors.append(f"{label}.size_bytes does not match path.")
        if _is_sha256(ref.get("sha256")) and _sha256(current_path) != ref.get("sha256"):
            target.errors.append(f"{label}.sha256 does not match path.")

def _repo_root_for_artifact(source_path: Path) -> Path | None:
    for root in source_path.resolve().parents:
        if (root / ".git").exists() or (root / "pyproject.toml").exists():
            return root
    return None

def _validate_agentic_training_flow_delegated_flow(
    value: Any,
    target: ValidationTarget,
    expected_passed: bool,
) -> dict[str, Any]:
    counts: dict[str, Any] = {
        "mode": "",
        "stage_count": 0,
        "executable_stage_count": 0,
        "selected_view_count": 0,
        "command_arg_count": 0,
        "trainer_input_count": 0,
        "external_code_file_count": 0,
    }
    if not isinstance(value, dict):
        target.errors.append("agentic_training_flow.delegated_flow must be an object.")
        return counts
    _validate_allowed_keys(value, _AGENTIC_TRAINING_FLOW_DELEGATED_KEYS, target, "agentic_training_flow.delegated_flow")
    mode = value.get("mode")
    counts["mode"] = mode if isinstance(mode, str) else ""
    blocked_advanced_mode = mode in BLOCKED_TRAINER_FLOW_MODES and not expected_passed
    if expected_passed and mode not in EXECUTABLE_FLOW_MODES:
        target.errors.append(f"agentic_training_flow.delegated_flow.mode must be one of {list(EXECUTABLE_FLOW_MODES)!r}.")
    elif mode not in EXECUTABLE_FLOW_MODES and not blocked_advanced_mode:
        target.errors.append(
            f"agentic_training_flow.delegated_flow.mode must be executable or an explicitly blocked mode, got {mode!r}."
        )
    if expected_passed and mode in BLOCKED_TRAINER_FLOW_MODES:
        target.errors.append("agentic_training_flow.delegated_flow.mode must not be an advanced reward or RL mode.")
    if not isinstance(value.get("backend"), str) or not value.get("backend"):
        target.errors.append("agentic_training_flow.delegated_flow.backend must be a non-empty string.")
    stage_sequence = value.get("stage_sequence")
    if not _is_string_list(stage_sequence):
        target.errors.append("agentic_training_flow.delegated_flow.stage_sequence must be a list of strings.")
        stage_sequence = []
    stages = value.get("stages")
    if not isinstance(stages, list):
        target.errors.append("agentic_training_flow.delegated_flow.stages must be a list.")
        stages = []
    if len(stages) != len(stage_sequence):
        target.errors.append("agentic_training_flow.delegated_flow.stages must match stage_sequence length.")
    counts["stage_count"] = len([stage for stage in stages if isinstance(stage, dict)])
    seen_views: set[str] = set()
    for index, stage in enumerate(stages):
        label = f"agentic_training_flow.delegated_flow.stages[{index}]"
        if not isinstance(stage, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(stage, _AGENTIC_TRAINING_FLOW_STAGE_KEYS, target, label)
        stage_id = stage.get("stage_id")
        allowed_stages = (*EXECUTABLE_STAGES, *BLOCKED_TRAINER_FLOW_STAGES) if blocked_advanced_mode else EXECUTABLE_STAGES
        if stage_id not in allowed_stages:
            target.errors.append(f"{label}.stage_id must be one of {list(allowed_stages)!r}.")
        else:
            if stage_id in EXECUTABLE_STAGES:
                counts["executable_stage_count"] += 1
        if stage.get("stage_index") != index:
            target.errors.append(f"{label}.stage_index expected {index}, got {stage.get('stage_index')!r}.")
        if index < len(stage_sequence) and stage_id != stage_sequence[index]:
            target.errors.append(f"{label}.stage_id must match delegated_flow.stage_sequence[{index}].")
        if stage.get("view_ready") is not True:
            target.errors.append(f"{label}.view_ready must be true.")
        if not isinstance(stage.get("view_name"), str) or not stage.get("view_name"):
            target.errors.append(f"{label}.view_name must be a non-empty string.")
        else:
            seen_views.add(stage["view_name"])
        if not isinstance(stage.get("view_path"), str) or not stage.get("view_path"):
            target.errors.append(f"{label}.view_path must be a non-empty string.")
        if not isinstance(stage.get("view_schema_version"), str) or not stage.get("view_schema_version"):
            target.errors.append(f"{label}.view_schema_version must be a non-empty string.")
        if not _is_non_negative_int(stage.get("row_count")) or stage.get("row_count") <= 0:
            target.errors.append(f"{label}.row_count must be a positive integer.")
    counts["selected_view_count"] = len(seen_views)
    command_counts = _validate_agentic_training_flow_command(value.get("command"), target)
    counts.update(command_counts)
    return counts

def _validate_agentic_training_flow_mode_contract_check(
    value: Any,
    target: ValidationTarget,
    mode: str,
    expected_passed: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        target.errors.append("agentic_training_flow.mode_contract_check must be an object.")
        return {}
    _validate_allowed_keys(
        value,
        _AGENTIC_TRAINING_FLOW_MODE_CONTRACT_KEYS,
        target,
        "agentic_training_flow.mode_contract_check",
    )
    for field_name in ("mode", "category"):
        if not isinstance(value.get(field_name), str):
            target.errors.append(f"agentic_training_flow.mode_contract_check.{field_name} must be a string.")
    if mode and value.get("mode") != mode:
        target.errors.append("agentic_training_flow.mode_contract_check.mode must match delegated_flow.mode.")
    for field_name in ("present", "mode_matches_plan", "planning_gate_open", "passed"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"agentic_training_flow.mode_contract_check.{field_name} must be a boolean.")
    if not isinstance(value.get("planning_required_flag"), (str, type(None))):
        target.errors.append("agentic_training_flow.mode_contract_check.planning_required_flag must be a string or null.")
    if not _is_non_negative_int(value.get("data_requirement_count")):
        target.errors.append("agentic_training_flow.mode_contract_check.data_requirement_count must be a non-negative integer.")
    if not _is_string_list(value.get("unsatisfied_data_requirement_ids")):
        target.errors.append("agentic_training_flow.mode_contract_check.unsatisfied_data_requirement_ids must be a list of strings.")
    errors = value.get("errors")
    if not _is_string_list(errors):
        target.errors.append("agentic_training_flow.mode_contract_check.errors must be a list of strings.")
        errors = []
    if not _is_non_negative_int(value.get("error_count")):
        target.errors.append("agentic_training_flow.mode_contract_check.error_count must be a non-negative integer.")
    elif value.get("error_count") != len(errors):
        target.errors.append("agentic_training_flow.mode_contract_check.error_count must match errors length.")
    _validate_agentic_training_flow_reward_contract(value.get("reward_contract"), target)
    _validate_agentic_training_flow_side_effect_boundary(value.get("side_effect_boundary"), target)
    _validate_agentic_training_flow_external_runner_contract(value.get("external_runner_contract"), target)
    if expected_passed and value.get("passed") is not True:
        target.errors.append("agentic_training_flow.mode_contract_check.passed must be true for ready delegated flow receipts.")
    if mode in BLOCKED_TRAINER_FLOW_MODES:
        if value.get("present") is not True:
            target.errors.append("agentic_training_flow.mode_contract_check.present must be true for blocked advanced-mode receipts.")
        if value.get("mode_matches_plan") is not True:
            target.errors.append("agentic_training_flow.mode_contract_check.mode_matches_plan must be true for blocked advanced-mode receipts.")
    return value

def _validate_agentic_training_flow_reward_contract(value: Any, target: ValidationTarget) -> None:
    label = "agentic_training_flow.mode_contract_check.reward_contract"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_FLOW_REWARD_CONTRACT_KEYS, target, label)
    if not isinstance(value.get("kind"), str):
        target.errors.append(f"{label}.kind must be a string.")
    if not isinstance(value.get("callable_signature"), str):
        target.errors.append(f"{label}.callable_signature must be a string.")
    for field_name in (
        "required",
        "external_runner_must_supply",
        "external_runner_must_validate",
    ):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if value.get("flight_recorder_supplies_callable") is not False:
        target.errors.append(f"{label}.flight_recorder_supplies_callable must be false.")
    if value.get("may_call_paid_services_by_default") is not False:
        target.errors.append(f"{label}.may_call_paid_services_by_default must be false.")
    if value.get("may_require_secrets_by_default") is not False:
        target.errors.append(f"{label}.may_require_secrets_by_default must be false.")
    if value.get("must_not_use_unredacted_traces") is not True:
        target.errors.append(f"{label}.must_not_use_unredacted_traces must be true.")

def _validate_agentic_training_flow_side_effect_boundary(value: Any, target: ValidationTarget) -> None:
    label = "agentic_training_flow.mode_contract_check.side_effect_boundary"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_FLOW_SIDE_EFFECT_KEYS, target, label)
    if value.get("dry_run_only") is not True:
        target.errors.append(f"{label}.dry_run_only must be true.")
    for field_name in (
        "training_started",
        "cloud_jobs_started",
        "model_downloads_started",
        "paid_model_grader_calls_started",
        "weights_updated",
        "provider_credentials_required_by_flight_recorder",
    ):
        if value.get(field_name) is not False:
            target.errors.append(f"{label}.{field_name} must be false.")

def _validate_agentic_training_flow_external_runner_contract(value: Any, target: ValidationTarget) -> None:
    label = "agentic_training_flow.mode_contract_check.external_runner_contract"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_FLOW_EXTERNAL_RUNNER_KEYS, target, label)
    if value.get("runner_must_require_recommendation") != _AGENTIC_TRAINING_PLAN_READY_RECOMMENDATION:
        target.errors.append(f"{label}.runner_must_require_recommendation must be ready_for_external_trainer_plan.")
    for field_name in (
        "runner_owns_execution",
        "runner_must_revalidate_inputs",
        "runner_must_block_unredacted_traces",
    ):
        if value.get(field_name) is not True:
            target.errors.append(f"{label}.{field_name} must be true.")
    if not isinstance(value.get("runner_must_validate_reward_contract"), bool):
        target.errors.append(f"{label}.runner_must_validate_reward_contract must be a boolean.")

def _validate_agentic_training_flow_mode_gate(
    value: Any,
    target: ValidationTarget,
    mode: str,
    mode_contract_check: dict[str, Any],
    expected_passed: bool,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("agentic_training_flow.flow_mode_gate must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_FLOW_MODE_GATE_KEYS, target, "agentic_training_flow.flow_mode_gate")
    if value.get("mode") != mode:
        target.errors.append("agentic_training_flow.flow_mode_gate.mode must match delegated_flow.mode.")
    category = value.get("category")
    if not isinstance(category, str):
        target.errors.append("agentic_training_flow.flow_mode_gate.category must be a string.")
    elif mode_contract_check.get("category") and category != mode_contract_check.get("category"):
        target.errors.append("agentic_training_flow.flow_mode_gate.category must match mode_contract_check.category.")
    expected_executable = mode in EXECUTABLE_FLOW_MODES
    expected_blocked = mode in BLOCKED_TRAINER_FLOW_MODES
    expected_promotion_required = not expected_executable
    expected_status = "default_executable" if expected_executable else "blocked_until_flow_promotion"
    expected_values = {
        "executable_by_default": expected_executable,
        "blocked_by_default": expected_blocked,
        "promotion_required": expected_promotion_required,
        "mode_contract_ready": mode_contract_check.get("passed") is True,
        "external_runner_must_supply_reward": mode_contract_check.get("reward_contract", {}).get("external_runner_must_supply") is True,
        "external_runner_must_validate_reward": mode_contract_check.get("reward_contract", {}).get("external_runner_must_validate") is True,
    }
    for field_name, expected_value in expected_values.items():
        if value.get(field_name) is not expected_value:
            target.errors.append(f"agentic_training_flow.flow_mode_gate.{field_name} must be {expected_value!r}.")
    if value.get("promotion_status") != expected_status:
        target.errors.append(f"agentic_training_flow.flow_mode_gate.promotion_status must be {expected_status!r}.")
    if value.get("required_plan_opt_in_flag") != mode_contract_check.get("planning_required_flag"):
        target.errors.append("agentic_training_flow.flow_mode_gate.required_plan_opt_in_flag must match mode_contract_check.")
    if value.get("reward_contract_kind") != mode_contract_check.get("reward_contract", {}).get("kind", ""):
        target.errors.append("agentic_training_flow.flow_mode_gate.reward_contract_kind must match mode_contract_check.")
    if not isinstance(value.get("reason"), str):
        target.errors.append("agentic_training_flow.flow_mode_gate.reason must be a string.")
    elif expected_blocked and not value.get("reason"):
        target.errors.append("agentic_training_flow.flow_mode_gate.reason must explain blocked advanced-mode receipts.")
    if expected_passed and expected_blocked:
        target.errors.append("agentic_training_flow.flow_mode_gate must not allow a ready advanced-mode flow.")
    expected_category = BLOCKED_FLOW_MODE_CATEGORIES.get(mode)
    if expected_category and category != expected_category:
        target.errors.append(f"agentic_training_flow.flow_mode_gate.category must be {expected_category!r}.")

def _validate_agentic_training_flow_command(value: Any, target: ValidationTarget) -> dict[str, int]:
    counts = {"command_arg_count": 0, "trainer_input_count": 0, "external_code_file_count": 0}
    if not isinstance(value, dict):
        target.errors.append("agentic_training_flow.delegated_flow.command must be an object.")
        return counts
    _validate_allowed_keys(value, _AGENTIC_TRAINING_FLOW_COMMAND_KEYS, target, "agentic_training_flow.delegated_flow.command")
    label = "agentic_training_flow.delegated_flow.command"
    for field_name in ("execution_cwd", "archive_root", "external_code_root", "command_shell"):
        if not isinstance(value.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    for field_name in ("execution_cwd", "archive_root", "external_code_root"):
        _validate_agentic_training_flow_command_path(target, f"{label}.{field_name}", value.get(field_name))
    argv = value.get("command_argv")
    if not _is_string_list(argv):
        target.errors.append(f"{label}.command_argv must be a list of strings.")
        argv = []
    clean_argv = [item for item in argv if isinstance(item, str)]
    for index, item in enumerate(clean_argv):
        _validate_agentic_training_flow_command_token_public_path(target, f"{label}.command_argv[{index}]", item)
    counts["command_arg_count"] = len(clean_argv)
    if "command_arg_count" in value and value.get("command_arg_count") != len(clean_argv):
        target.errors.append(f"{label}.command_arg_count must match command_argv length.")
    expected_shell = shlex.join(clean_argv) if clean_argv else ""
    if value.get("command_shell") != expected_shell:
        target.errors.append(f"{label}.command_shell must match command_argv.")
    for field_name in ("trainer_input_count", "external_code_file_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
        else:
            counts[field_name] = int(value[field_name])
    if not clean_argv:
        target.errors.append(f"{label}.command_argv must not be empty.")
    return counts

def _validate_agentic_training_flow_command_path(target: ValidationTarget, label: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        return
    if _is_safe_training_flow_metadata_path(value):
        return
    target.errors.append(f"{label} must be a safe relative path or redacted placeholder.")

def _validate_agentic_training_flow_command_token_public_path(target: ValidationTarget, label: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        return
    if _looks_absolute(value):
        target.errors.append(f"{label} must use a relative command token or redacted placeholder.")
        return
    _, separator, token_value = value.partition("=")
    if separator and _looks_absolute(token_value):
        target.errors.append(f"{label} must use a relative command token or redacted placeholder.")

def _validate_agentic_training_flow_metrics(
    value: Any,
    target: ValidationTarget,
    counts: dict[str, Any],
    check_count: int,
    failed_check_count: int,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("agentic_training_flow.metrics must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_FLOW_METRIC_KEYS, target, "agentic_training_flow.metrics")
    expected = {
        "check_count": check_count,
        "failed_check_count": failed_check_count,
        "stage_count": counts["stage_count"],
        "executable_stage_count": counts["executable_stage_count"],
        "selected_view_count": counts["selected_view_count"],
        "command_arg_count": counts["command_arg_count"],
        "trainer_input_count": counts["trainer_input_count"],
        "external_code_file_count": counts["external_code_file_count"],
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"agentic_training_flow.metrics.{field_name} expected {expected_value}, got {value.get(field_name)!r}.")

def _validate_agentic_training_flow_boundary(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("agentic_training_flow.execution_boundary must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_FLOW_BOUNDARY_KEYS, target, "agentic_training_flow.execution_boundary")
    if value.get("flow_plan_only") is not True:
        target.errors.append("agentic_training_flow.execution_boundary.flow_plan_only must be true.")
    for field_name in (
        "training_started",
        "trainer_command_executed",
        "subprocess_started",
        "cloud_jobs_started",
        "model_downloads_started",
        "weights_updated_by_flight_recorder",
        "trainer_modules_imported",
        "credential_values_recorded",
    ):
        if value.get(field_name) is not False:
            target.errors.append(f"agentic_training_flow.execution_boundary.{field_name} must be false.")

def _validate_agentic_training_flow_contract(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("agentic_training_flow.handoff_contract must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_FLOW_HANDOFF_KEYS, target, "agentic_training_flow.handoff_contract")
    if value.get("runner_owns_execution") is not True:
        target.errors.append("agentic_training_flow.handoff_contract.runner_owns_execution must be true.")
    if value.get("runner_must_require_recommendation") != FLOW_READY_RECOMMENDATION:
        target.errors.append(
            "agentic_training_flow.handoff_contract.runner_must_require_recommendation must be ready_for_delegated_trainer_execution."
        )
    if value.get("runner_must_emit_result_schema") != AGENTIC_TRAINING_RESULT_SCHEMA_VERSION:
        target.errors.append("agentic_training_flow.handoff_contract.runner_must_emit_result_schema must be hfr.agentic_training_result.v1.")
    for field_name in (
        "requires_runtime_preflight",
        "requires_trainer_consumer_plan",
        "requires_mode_contract",
        "requires_mode_contract_ready",
        "requires_registered_model_and_dataset",
        "requires_redacted_dataset",
    ):
        if value.get(field_name) is not True:
            target.errors.append(f"agentic_training_flow.handoff_contract.{field_name} must be true.")
    if value.get("flight_recorder_executed_trainer") is not False:
        target.errors.append("agentic_training_flow.handoff_contract.flight_recorder_executed_trainer must be false.")
    if sorted(value.get("executable_modes", [])) != sorted(EXECUTABLE_FLOW_MODES):
        target.errors.append("agentic_training_flow.handoff_contract.executable_modes must match executable trainer flow modes.")
    if sorted(value.get("blocked_modes", [])) != sorted(BLOCKED_TRAINER_FLOW_MODES):
        target.errors.append("agentic_training_flow.handoff_contract.blocked_modes must match blocked trainer flow modes.")
    if sorted(value.get("blocked_mode_categories", [])) != sorted(set(BLOCKED_FLOW_MODE_CATEGORIES.values())):
        target.errors.append("agentic_training_flow.handoff_contract.blocked_mode_categories must match blocked trainer flow categories.")
    if sorted(value.get("blocked_mode_stages", [])) != sorted(BLOCKED_TRAINER_FLOW_STAGES):
        target.errors.append("agentic_training_flow.handoff_contract.blocked_mode_stages must match blocked trainer flow stages.")

def _is_safe_or_redacted_training_flow_path(value: str) -> bool:
    if value.startswith("<redacted:"):
        return True
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return (
        bool(value)
        and not path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and "\\" not in value
        and "~" not in path.parts
    )

def _is_safe_training_flow_metadata_path(value: str) -> bool:
    path = Path(value)
    return _is_safe_or_redacted_training_flow_path(value) and ".." not in path.parts

_AGENTIC_TRAINING_RESULT_KEYS = {
    "schema_version",
    "created_at",
    "artifact_path",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "blocked_reasons",
    "training_result",
    "lineage",
    "failure",
    "artifacts",
    "metrics",
    "registry_update",
    "execution_boundary",
    "handoff_contract",
    "notes",
}

_AGENTIC_TRAINING_RESULT_CHECK_KEYS = {"id", "passed", "actual", "expected", "summary"}

_AGENTIC_TRAINING_RESULT_TRAINING_KEYS = {
    "status",
    "runner_id",
    "run_id",
    "mode",
    "backend",
    "output_dir",
    "external_runner_reported_status",
    "flight_recorder_executed_training",
    "model_downloads_started_by_flight_recorder",
}

_AGENTIC_TRAINING_RESULT_LINEAGE_KEYS = {"plan", "runtime_preflight", "model", "dataset", "agentic_training_flow"}

_AGENTIC_TRAINING_RESULT_LINEAGE_REF_KEYS = {"path", "exists", "regular_file", "schema_name", "sha256", "size_bytes"}

_AGENTIC_TRAINING_RESULT_MANIFEST_REF_KEYS = {"id", "path", "sha256", "size_bytes", "license_allows_training"}

_AGENTIC_TRAINING_RESULT_FAILURE_KEYS = {"class", "message", "recoverable", "source"}

_AGENTIC_TRAINING_RESULT_ARTIFACT_KEYS = {"role", "path", "exists", "regular_file", "sha256", "size_bytes"}

_AGENTIC_TRAINING_RESULT_METRIC_KEYS = {
    "artifact_count",
    "regular_artifact_count",
    "output_artifact_count",
    "config_count",
    "metrics_file_count",
    "adapter_count",
    "checkpoint_count",
    "log_count",
    "failure_report_count",
}

_AGENTIC_TRAINING_RESULT_BOUNDARY_KEYS = {
    "archive_only",
    "runner_owns_execution",
    "flight_recorder_launched_training",
    "training_started_by_flight_recorder",
    "model_downloads_started_by_flight_recorder",
    "trainer_modules_imported_by_flight_recorder",
}

_AGENTIC_TRAINING_RESULT_HANDOFF_KEYS = {
    "runner_owns_execution",
    "requires_agentic_training_plan",
    "requires_runtime_preflight",
    "requires_agentic_training_flow_for_completed",
    "requires_runtime_ready_for_completed",
    "requires_flow_ready_for_completed",
    "requires_classified_failure_for_non_completed",
    "requires_output_artifact_for_completed",
    "requires_registered_model",
    "requires_registered_dataset",
    "requires_redacted_dataset",
    "flight_recorder_launched_training",
    "model_downloads_started_by_flight_recorder",
}

_AGENTIC_TRAINING_RESULT_REGISTRY_UPDATE_KEYS = {
    "applied",
    "side_effect_free",
    "ready_to_apply",
    "target_model_id",
    "target_dataset_id",
    "links",
    "notes",
}

_AGENTIC_TRAINING_RESULT_REGISTRY_LINK_KEYS = {
    "collection",
    "artifact_id",
    "kind",
    "status",
    "path",
    "sha256",
    "size_bytes",
}

def _validate_agentic_training_result(result: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(result, _AGENTIC_TRAINING_RESULT_KEYS, target, "agentic_training_result")
    _require_equal(result, "schema_version", AGENTIC_TRAINING_RESULT_SCHEMA_VERSION, target, prefix="agentic_training_result.")
    checks = result.get("checks")
    if not isinstance(checks, list):
        target.errors.append("agentic_training_result.checks must be a list.")
        checks = []
    for index, check in enumerate(checks):
        if isinstance(check, dict):
            _validate_allowed_keys(check, _AGENTIC_TRAINING_RESULT_CHECK_KEYS, target, f"agentic_training_result.checks[{index}]")
    failed_checks = _validate_gate_like_checks(checks, target, "agentic_training_result.checks")
    if result.get("check_count") != len(checks):
        target.errors.append(f"agentic_training_result.check_count expected {len(checks)}, got {result.get('check_count')!r}.")
    if result.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"agentic_training_result.failed_check_count expected {failed_checks}, got {result.get('failed_check_count')!r}."
        )

    expected_passed = failed_checks == 0
    if not isinstance(result.get("passed"), bool):
        target.errors.append("agentic_training_result.passed must be a boolean.")
    elif result["passed"] != expected_passed:
        target.errors.append("agentic_training_result.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    if result.get("readiness") != expected_readiness:
        target.errors.append(
            f"agentic_training_result.readiness expected {expected_readiness!r}, got {result.get('readiness')!r}."
        )
    artifact_path = result.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        target.errors.append("agentic_training_result.artifact_path must be a non-empty string.")
    elif not _is_safe_agentic_training_result_path(artifact_path):
        target.errors.append("agentic_training_result.artifact_path must be a safe relative path without traversal.")

    training_result = result.get("training_result")
    if not isinstance(training_result, dict):
        target.errors.append("agentic_training_result.training_result must be an object.")
        training_result = {}
    else:
        _validate_allowed_keys(
            training_result,
            _AGENTIC_TRAINING_RESULT_TRAINING_KEYS,
            target,
            "agentic_training_result.training_result",
        )
    status = training_result.get("status")
    if status not in RESULT_STATUSES:
        target.errors.append(f"agentic_training_result.training_result.status must be one of {list(RESULT_STATUSES)!r}.")
    if training_result.get("external_runner_reported_status") != status:
        target.errors.append("agentic_training_result.training_result.external_runner_reported_status must match status.")
    if training_result.get("flight_recorder_executed_training") is not False:
        target.errors.append("agentic_training_result.training_result.flight_recorder_executed_training must be false.")
    if training_result.get("model_downloads_started_by_flight_recorder") is not False:
        target.errors.append("agentic_training_result.training_result.model_downloads_started_by_flight_recorder must be false.")
    output_dir = training_result.get("output_dir")
    if not isinstance(output_dir, str):
        target.errors.append("agentic_training_result.training_result.output_dir must be a string.")
    elif output_dir and not _is_safe_agentic_training_result_path(output_dir):
        target.errors.append("agentic_training_result.training_result.output_dir must be a safe relative path without traversal.")

    failure = result.get("failure")
    if not isinstance(failure, dict):
        target.errors.append("agentic_training_result.failure must be an object.")
        failure = {}
    else:
        _validate_allowed_keys(failure, _AGENTIC_TRAINING_RESULT_FAILURE_KEYS, target, "agentic_training_result.failure")
    failure_class = failure.get("class")
    failure_message = failure.get("message")
    if failure_class not in FAILURE_CLASSES:
        target.errors.append(f"agentic_training_result.failure.class must be one of {list(FAILURE_CLASSES)!r}.")
    if not isinstance(failure_message, str):
        target.errors.append("agentic_training_result.failure.message must be a string.")
        failure_message = ""
    if isinstance(failure_class, str):
        expected_recoverable = failure_class in RECOVERABLE_FAILURE_CLASSES
        if failure.get("recoverable") != expected_recoverable:
            target.errors.append(
                f"agentic_training_result.failure.recoverable expected {expected_recoverable!r}, got {failure.get('recoverable')!r}."
            )
    if status == "completed" and failure_class != "none":
        target.errors.append("agentic_training_result.failure.class must be none for completed results.")
    if status in {"failed", "blocked", "aborted"} and (failure_class in {"none", "unknown"} or not failure_message.strip()):
        target.errors.append("agentic_training_result non-completed receipts require a classified failure and message.")

    artifact_counts = _validate_agentic_training_result_artifacts(
        result.get("artifacts"), target, source_path
    )
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("agentic_training_result.metrics must be an object.")
        metrics = {}
    else:
        _validate_allowed_keys(metrics, _AGENTIC_TRAINING_RESULT_METRIC_KEYS, target, "agentic_training_result.metrics")
    for field_name, expected in artifact_counts.items():
        if metrics.get(field_name) != expected:
            target.errors.append(f"agentic_training_result.metrics.{field_name} expected {expected}, got {metrics.get(field_name)!r}.")
    if expected_passed and artifact_counts["regular_artifact_count"] != artifact_counts["artifact_count"]:
        target.errors.append("agentic_training_result passed receipts require all artifact refs to be regular files.")
    if status == "completed" and artifact_counts["output_artifact_count"] == 0:
        target.errors.append("agentic_training_result completed receipts must include an adapter or checkpoint artifact.")

    if expected_passed and status == "completed":
        expected_recommendation = REGISTER_RESULT_RECOMMENDATION
    elif expected_passed:
        expected_recommendation = REGISTER_FAILURE_RECOMMENDATION
    else:
        expected_recommendation = BLOCK_REGISTRATION_RECOMMENDATION
    if result.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"agentic_training_result.recommendation expected {expected_recommendation!r}, got {result.get('recommendation')!r}."
        )
    blocked_reasons = result.get("blocked_reasons")
    if not _is_string_list(blocked_reasons):
        target.errors.append("agentic_training_result.blocked_reasons must be a list of strings.")
    elif expected_passed and blocked_reasons:
        target.errors.append("agentic_training_result.blocked_reasons must be empty when passed.")
    elif not expected_passed and len(blocked_reasons) != failed_checks:
        target.errors.append("agentic_training_result.blocked_reasons must match failed_check_count.")

    _validate_agentic_training_result_lineage(result.get("lineage"), target, source_path, status, expected_passed)
    _validate_agentic_training_result_boundary(result.get("execution_boundary"), target, "execution_boundary")
    _validate_agentic_training_result_contract(result.get("handoff_contract"), target)
    if "registry_update" in result:
        _validate_agentic_training_result_registry_update(
            result.get("registry_update"), expected_passed, target, result.get("artifacts"), status, artifact_path
        )
    elif status == "completed":
        target.errors.append("agentic_training_result.registry_update is required for completed receipts.")
    notes = result.get("notes")
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        target.errors.append("agentic_training_result.notes must be a list of strings.")

    target.details.update(
        {
            "status": status,
            "recommendation": result.get("recommendation"),
            "artifact_count": artifact_counts["artifact_count"],
            "output_artifact_count": artifact_counts["output_artifact_count"],
        }
    )

def _validate_agentic_training_result_artifacts(
    value: Any,
    target: ValidationTarget,
    source_path: Path,
) -> dict[str, int]:
    counts = {
        "artifact_count": 0,
        "regular_artifact_count": 0,
        "output_artifact_count": 0,
        "config_count": 0,
        "metrics_file_count": 0,
        "adapter_count": 0,
        "checkpoint_count": 0,
        "log_count": 0,
        "failure_report_count": 0,
    }
    if not isinstance(value, list):
        target.errors.append("agentic_training_result.artifacts must be a list.")
        return counts
    counts["artifact_count"] = len(value)
    role_metric_fields = {
        "config": "config_count",
        "metrics": "metrics_file_count",
        "adapter": "adapter_count",
        "checkpoint": "checkpoint_count",
        "log": "log_count",
        "failure_report": "failure_report_count",
    }
    for index, artifact in enumerate(value):
        label = f"agentic_training_result.artifacts[{index}]"
        if not isinstance(artifact, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(artifact, _AGENTIC_TRAINING_RESULT_ARTIFACT_KEYS, target, label)
        role = artifact.get("role")
        if not isinstance(role, str) or not role:
            target.errors.append(f"{label}.role must be a non-empty string.")
        elif role in role_metric_fields:
            counts[role_metric_fields[role]] += 1
        if role in OUTPUT_ARTIFACT_ROLES:
            counts["output_artifact_count"] += 1
        path_value = artifact.get("path")
        path_is_safe = isinstance(path_value, str) and _is_safe_agentic_training_result_path(path_value)
        current_path = _agentic_training_result_reference_path(path_value, source_path) if path_is_safe else None
        opaque_attestation = (
            get_active_opaque_output_attestation(current_path)
            if role in OUTPUT_ARTIFACT_ROLES and current_path is not None
            else None
        )
        if not isinstance(path_value, str) or not path_value:
            target.errors.append(f"{label}.path must be a non-empty string.")
        elif not path_is_safe:
            target.errors.append(f"{label}.path must be a safe relative path without traversal.")
        for field_name in ("exists", "regular_file"):
            if not isinstance(artifact.get(field_name), bool):
                target.errors.append(f"{label}.{field_name} must be a boolean.")
        current_exists = opaque_attestation is not None or (
            current_path is not None and current_path.exists()
        )
        current_regular = opaque_attestation is not None or (
            current_path is not None
            and current_path.is_file()
            and not current_path.is_symlink()
            and not _path_has_symlink_component(current_path, include_leaf=False)
        )
        if artifact.get("exists") is True and current_path is not None and not current_exists:
            target.errors.append(f"{label}.path does not resolve to the current file.")
        if (
            artifact.get("regular_file") is True
            and current_path is not None
            and opaque_attestation is None
            and current_path.is_symlink()
        ):
            target.errors.append(f"{label}.path must not be a symlink.")
        if artifact.get("regular_file") is True and current_path is not None and not current_regular:
            target.errors.append(f"{label}.path does not resolve to a regular file.")
        if artifact.get("regular_file") is True:
            counts["regular_artifact_count"] += 1
            if not _is_sha256(artifact.get("sha256")):
                target.errors.append(f"{label}.sha256 must be a SHA-256 hex string for regular files.")
        elif artifact.get("sha256") is not None and not _is_sha256(artifact.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string or null.")
        if not _is_non_negative_int(artifact.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
        elif artifact.get("regular_file") is True and current_path is not None and current_regular:
            current_size = (
                opaque_attestation.size_bytes
                if opaque_attestation is not None
                else current_path.stat().st_size
            )
            current_sha256 = (
                opaque_attestation.sha256
                if opaque_attestation is not None
                else _sha256(current_path)
            )
            if current_size != artifact.get("size_bytes"):
                target.errors.append(f"{label}.size_bytes does not match the current file.")
            if _is_sha256(artifact.get("sha256")) and current_sha256 != artifact.get("sha256"):
                target.errors.append(f"{label}.sha256 does not match the current file.")
    return counts

def _validate_agentic_training_result_lineage(
    value: Any,
    target: ValidationTarget,
    source_path: Path,
    status: Any,
    expected_passed: bool,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("agentic_training_result.lineage must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_RESULT_LINEAGE_KEYS, target, "agentic_training_result.lineage")
    plan_inputs = _agentic_training_result_plan_input_manifests(value, source_path)
    for field_name in ("plan", "runtime_preflight"):
        ref = value.get(field_name)
        label = f"agentic_training_result.lineage.{field_name}"
        if not isinstance(ref, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(ref, _AGENTIC_TRAINING_RESULT_LINEAGE_REF_KEYS, target, label)
        for string_field in ("path", "schema_name"):
            if not isinstance(ref.get(string_field), str) or not ref.get(string_field):
                target.errors.append(f"{label}.{string_field} must be a non-empty string.")
        if isinstance(ref.get("path"), str) and ref.get("path") and not _is_safe_agentic_training_result_path(ref["path"]):
            target.errors.append(f"{label}.path must be a safe relative path without traversal.")
        for bool_field in ("exists", "regular_file"):
            if not isinstance(ref.get(bool_field), bool):
                target.errors.append(f"{label}.{bool_field} must be a boolean.")
        if not _is_sha256(ref.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string.")
        if not _is_non_negative_int(ref.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
        _validate_agentic_training_result_lineage_file(ref, target, label, source_path)
    for field_name in ("model", "dataset"):
        ref = value.get(field_name)
        label = f"agentic_training_result.lineage.{field_name}"
        if not isinstance(ref, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(ref, _AGENTIC_TRAINING_RESULT_MANIFEST_REF_KEYS, target, label)
        for string_field in ("id", "path", "sha256"):
            if not isinstance(ref.get(string_field), str):
                target.errors.append(f"{label}.{string_field} must be a string.")
        if isinstance(ref.get("path"), str) and ref.get("path") and not _is_safe_agentic_training_result_path(ref["path"]):
            target.errors.append(f"{label}.path must be a safe relative path without traversal.")
        if ref.get("sha256") and not _is_sha256(ref.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when present.")
        if not _is_non_negative_int(ref.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
        if ref.get("license_allows_training") is not True:
            target.errors.append(f"{label}.license_allows_training must be true.")
        expected_ref = plan_inputs.get(field_name)
        if isinstance(expected_ref, dict):
            _validate_agentic_training_result_manifest_matches_plan(ref, expected_ref, target, label, field_name)
    flow_ref = value.get("agentic_training_flow")
    if status == "completed" and expected_passed and not isinstance(flow_ref, dict):
        target.errors.append("agentic_training_result.lineage.agentic_training_flow is required for completed receipts.")
    if isinstance(flow_ref, dict):
        label = "agentic_training_result.lineage.agentic_training_flow"
        _validate_allowed_keys(flow_ref, _AGENTIC_TRAINING_RESULT_LINEAGE_REF_KEYS, target, label)
        for string_field in ("path", "schema_name"):
            if not isinstance(flow_ref.get(string_field), str) or not flow_ref.get(string_field):
                target.errors.append(f"{label}.{string_field} must be a non-empty string.")
        if flow_ref.get("schema_name") != "agentic_training_flow":
            target.errors.append(f"{label}.schema_name must be 'agentic_training_flow'.")
        if isinstance(flow_ref.get("path"), str) and flow_ref.get("path") and not _is_safe_agentic_training_result_path(flow_ref["path"]):
            target.errors.append(f"{label}.path must be a safe relative path without traversal.")
        for bool_field in ("exists", "regular_file"):
            if not isinstance(flow_ref.get(bool_field), bool):
                target.errors.append(f"{label}.{bool_field} must be a boolean.")
        if not _is_sha256(flow_ref.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string.")
        if not _is_non_negative_int(flow_ref.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
        _validate_agentic_training_result_lineage_file(flow_ref, target, label, source_path)
        _validate_agentic_training_result_flow_lineage(flow_ref, value, target, source_path, status, expected_passed)

def _validate_agentic_training_result_flow_lineage(
    flow_ref: dict[str, Any],
    lineage: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
    status: Any,
    expected_passed: bool,
) -> None:
    path_value = flow_ref.get("path")
    if not isinstance(path_value, str) or not _is_safe_agentic_training_result_path(path_value):
        return
    flow_path = _agentic_training_result_reference_path(path_value, source_path)
    if not flow_path.is_file() or flow_path.is_symlink() or _path_has_symlink_component(flow_path, include_leaf=False):
        return
    try:
        flow = json.loads(flow_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        if status == "completed" and expected_passed:
            target.errors.append("agentic_training_result.lineage.agentic_training_flow.path must contain readable JSON.")
        return
    if not isinstance(flow, dict):
        if status == "completed" and expected_passed:
            target.errors.append("agentic_training_result.lineage.agentic_training_flow.path must contain a JSON object.")
        return
    if status != "completed" or not expected_passed:
        return
    flow_target = ValidationTarget("agentic_training_flow", str(flow_path))
    _validate_agentic_training_flow(flow, flow_target, flow_path)
    if flow_target.errors:
        for error in flow_target.errors:
            target.errors.append(f"agentic_training_result.lineage.agentic_training_flow invalid: {error}")
    if flow.get("schema_version") != AGENTIC_TRAINING_FLOW_SCHEMA_VERSION:
        target.errors.append(
            f"agentic_training_result.lineage.agentic_training_flow.schema_version must be {AGENTIC_TRAINING_FLOW_SCHEMA_VERSION!r}."
        )
    if flow.get("passed") is not True:
        target.errors.append("agentic_training_result.lineage.agentic_training_flow.passed must be true for completed receipts.")
    if flow.get("recommendation") != FLOW_READY_RECOMMENDATION:
        target.errors.append(
            "agentic_training_result.lineage.agentic_training_flow.recommendation must be ready_for_delegated_trainer_execution."
        )
    flow_mode_gate = flow.get("flow_mode_gate") if isinstance(flow.get("flow_mode_gate"), dict) else {}
    if flow_mode_gate.get("executable_by_default") is not True:
        target.errors.append("agentic_training_result.lineage.agentic_training_flow.flow_mode_gate.executable_by_default must be true.")
    if flow_mode_gate.get("blocked_by_default") is not False:
        target.errors.append("agentic_training_result.lineage.agentic_training_flow.flow_mode_gate.blocked_by_default must be false.")
    source_artifacts = flow.get("source_artifacts") if isinstance(flow.get("source_artifacts"), dict) else {}
    expected = {
        "agentic_training_plan": ("plan", "plan"),
        "agentic_training_runtime_preflight": ("runtime_preflight", "runtime_preflight"),
    }
    for flow_role, (lineage_role, label_role) in expected.items():
        flow_source = source_artifacts.get(flow_role) if isinstance(source_artifacts.get(flow_role), dict) else {}
        lineage_source = lineage.get(lineage_role) if isinstance(lineage.get(lineage_role), dict) else {}
        if flow_source.get("sha256") != lineage_source.get("sha256"):
            target.errors.append(
                "agentic_training_result.lineage.agentic_training_flow.source_artifacts."
                f"{flow_role}.sha256 must match lineage.{label_role}.sha256."
            )

def _agentic_training_result_plan_input_manifests(lineage: dict[str, Any], source_path: Path) -> dict[str, Any]:
    plan_ref = lineage.get("plan")
    if not isinstance(plan_ref, dict):
        return {}
    path_value = plan_ref.get("path")
    if not isinstance(path_value, str) or not path_value:
        return {}
    if not _is_safe_agentic_training_result_path(path_value):
        return {}
    plan_path = _agentic_training_result_reference_path(path_value, source_path)
    if not plan_path.is_file() or plan_path.is_symlink() or _path_has_symlink_component(plan_path, include_leaf=False):
        return {}
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    manifests = payload.get("input_manifests")
    return manifests if isinstance(manifests, dict) else {}

def _validate_agentic_training_result_manifest_matches_plan(
    ref: dict[str, Any],
    expected_ref: dict[str, Any],
    target: ValidationTarget,
    label: str,
    field_name: str,
) -> None:
    expected_label = f"agentic_training_plan.input_manifests.{field_name}"
    for string_field in ("id", "sha256"):
        if isinstance(expected_ref.get(string_field), str) and ref.get(string_field) != expected_ref.get(string_field):
            target.errors.append(f"{label}.{string_field} must match {expected_label}.{string_field}.")
    if isinstance(expected_ref.get("path"), str):
        expected_path = _agentic_training_result_redacted_path(expected_ref["path"])
        if ref.get("path") != expected_path:
            target.errors.append(f"{label}.path must match the safe {expected_label}.path.")
    if _is_non_negative_int(expected_ref.get("size_bytes")) and ref.get("size_bytes") != expected_ref.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes must match {expected_label}.size_bytes.")
    if isinstance(expected_ref.get("license_allows_training"), bool) and ref.get("license_allows_training") != expected_ref.get(
        "license_allows_training"
    ):
        target.errors.append(f"{label}.license_allows_training must match {expected_label}.license_allows_training.")

def _validate_agentic_training_result_lineage_file(
    ref: dict[str, Any], target: ValidationTarget, label: str, source_path: Path
) -> None:
    path_value = ref.get("path")
    if not isinstance(path_value, str) or not path_value:
        return
    if not _is_safe_agentic_training_result_path(path_value):
        return
    current_path = _agentic_training_result_reference_path(path_value, source_path)
    if ref.get("exists") is not True:
        target.errors.append(f"{label}.exists must be true.")
    if ref.get("regular_file") is not True:
        target.errors.append(f"{label}.regular_file must be true.")
    if ref.get("exists") is True and not current_path.exists():
        target.errors.append(f"{label}.path does not resolve to the current file.")
        return
    if ref.get("regular_file") is True and not current_path.is_file():
        target.errors.append(f"{label}.path does not resolve to a regular file.")
        return
    if not current_path.exists() or not current_path.is_file():
        return
    if current_path.is_symlink():
        target.errors.append(f"{label}.path must not be a symlink.")
        return
    if _path_has_symlink_component(current_path, include_leaf=False):
        target.errors.append(f"{label}.path must not traverse symlinked components.")
        return
    if _is_non_negative_int(ref.get("size_bytes")) and current_path.stat().st_size != ref.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if _is_sha256(ref.get("sha256")) and _sha256(current_path) != ref.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current file.")

def _agentic_training_result_reference_path(value: str, source_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return source_path.parent / path

def _validate_agentic_training_result_boundary(value: Any, target: ValidationTarget, field_name: str) -> None:
    label = f"agentic_training_result.{field_name}"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_RESULT_BOUNDARY_KEYS, target, label)
    expected = {
        "archive_only": True,
        "runner_owns_execution": True,
        "flight_recorder_launched_training": False,
        "training_started_by_flight_recorder": False,
        "model_downloads_started_by_flight_recorder": False,
        "trainer_modules_imported_by_flight_recorder": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            target.errors.append(f"{label}.{key} expected {expected_value!r}, got {value.get(key)!r}.")

def _validate_agentic_training_result_contract(value: Any, target: ValidationTarget) -> None:
    label = "agentic_training_result.handoff_contract"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_RESULT_HANDOFF_KEYS, target, label)
    expected = {
        "runner_owns_execution": True,
        "requires_agentic_training_plan": True,
        "requires_runtime_preflight": True,
        "requires_agentic_training_flow_for_completed": True,
        "requires_runtime_ready_for_completed": True,
        "requires_flow_ready_for_completed": True,
        "requires_classified_failure_for_non_completed": True,
        "requires_output_artifact_for_completed": True,
        "requires_registered_model": True,
        "requires_registered_dataset": True,
        "requires_redacted_dataset": True,
        "flight_recorder_launched_training": False,
        "model_downloads_started_by_flight_recorder": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            target.errors.append(f"{label}.{key} expected {expected_value!r}, got {value.get(key)!r}.")

def _validate_agentic_training_result_registry_update(
    value: Any, expected_ready: bool, target: ValidationTarget, artifacts: Any, status: Any, receipt_artifact_path: Any
) -> None:
    label = "agentic_training_result.registry_update"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object when present.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_RESULT_REGISTRY_UPDATE_KEYS, target, label)
    for key, expected_value in {"applied": False, "side_effect_free": True, "ready_to_apply": expected_ready}.items():
        if value.get(key) != expected_value:
            target.errors.append(f"{label}.{key} expected {expected_value!r}, got {value.get(key)!r}.")
    for field_name in ("target_model_id", "target_dataset_id"):
        if not isinstance(value.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    links = value.get("links")
    if not isinstance(links, list):
        target.errors.append(f"{label}.links must be a list.")
        return
    notes = value.get("notes")
    if notes is not None and (not isinstance(notes, list) or not all(isinstance(item, str) for item in notes)):
        target.errors.append(f"{label}.notes must be a list of strings when present.")
    if not links:
        target.errors.append(f"{label}.links must not be empty.")
    output_artifact_counter = _agentic_training_result_output_artifact_counter(artifacts)
    linked_output_counter: Counter[tuple[Any, Any, Any, Any]] = Counter()
    for index, link in enumerate(links):
        link_label = f"{label}.links[{index}]"
        if not isinstance(link, dict):
            target.errors.append(f"{link_label} must be an object.")
            continue
        _validate_allowed_keys(link, _AGENTIC_TRAINING_RESULT_REGISTRY_LINK_KEYS, target, link_label)
        for field_name in ("collection", "artifact_id", "kind", "status", "path"):
            if not isinstance(link.get(field_name), str) or not link.get(field_name):
                target.errors.append(f"{link_label}.{field_name} must be a non-empty string.")
        path_value = link.get("path")
        if isinstance(path_value, str) and path_value and not _is_safe_agentic_training_result_path(path_value):
            target.errors.append(f"{link_label}.path must be a safe relative path without traversal.")
        if link.get("collection") == "training_runs":
            if isinstance(receipt_artifact_path, str) and link.get("path") != receipt_artifact_path:
                target.errors.append(f"{link_label}.path must match agentic_training_result.artifact_path.")
            if "sha256" in link and link.get("sha256") is not None and not _is_sha256(link.get("sha256")):
                target.errors.append(f"{link_label}.sha256 must be a SHA-256 hex string or null.")
            if "size_bytes" in link and not _is_non_negative_int(link.get("size_bytes")):
                target.errors.append(f"{link_label}.size_bytes must be a non-negative integer.")
            continue
        if not _is_sha256(link.get("sha256")):
            target.errors.append(f"{link_label}.sha256 must be a SHA-256 hex string for artifact registry links.")
        if not _is_non_negative_int(link.get("size_bytes")):
            target.errors.append(f"{link_label}.size_bytes must be a non-negative integer for artifact registry links.")
        key = (link.get("path"), link.get("sha256"), link.get("size_bytes"), link.get("kind"))
        if output_artifact_counter.get(key, 0) == 0:
            target.errors.append(f"{link_label} must match a supplied output artifact ref by path, sha256, size_bytes, and kind.")
            matching_roles = sorted(
                str(candidate[3])
                for candidate in output_artifact_counter
                if candidate[:3] == key[:3] and isinstance(candidate[3], str)
            )
            if matching_roles and link.get("kind") not in matching_roles:
                target.errors.append(f"{link_label}.kind must match the supplied output artifact role.")
        else:
            if link.get("collection") != "adapters":
                target.errors.append(f"{link_label}.collection must be 'adapters' for output artifact registry links.")
            if link.get("collection") == "adapters":
                linked_output_counter[key] += 1
    if links and isinstance(links[0], dict) and links[0].get("collection") != "training_runs":
        target.errors.append(f"{label}.links[0].collection must be 'training_runs'.")
    if status == "completed":
        missing_links = sorted(
            key[0]
            for key, expected_count in output_artifact_counter.items()
            if linked_output_counter.get(key, 0) < expected_count
        )
        if missing_links:
            target.errors.append(
                f"{label}.links must include size-bound registry links for completed output artifacts: {missing_links!r}."
            )
        extra_links = sorted(
            key[0]
            for key, linked_count in linked_output_counter.items()
            if linked_count > output_artifact_counter.get(key, 0)
        )
        if extra_links:
            target.errors.append(f"{label}.links must not duplicate output artifact registry links: {extra_links!r}.")

def _agentic_training_result_output_artifact_counter(artifacts: Any) -> Counter[tuple[Any, Any, Any, Any]]:
    if not isinstance(artifacts, list):
        return Counter()
    counter: Counter[tuple[Any, Any, Any, Any]] = Counter()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        if artifact.get("role") not in OUTPUT_ARTIFACT_ROLES:
            continue
        counter[(artifact.get("path"), artifact.get("sha256"), artifact.get("size_bytes"), artifact.get("role"))] += 1
    return counter

def _agentic_training_result_redacted_path(value: str) -> str:
    if _is_safe_agentic_training_result_path(value):
        return value
    normalized = value.replace("\\", "/")
    basename = normalized.rstrip("/").rsplit("/", 1)[-1]
    return basename if _is_safe_agentic_training_result_path(basename) else "artifact"

def _is_safe_agentic_training_result_path(value: str) -> bool:
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return (
        bool(value)
        and not path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and "\\" not in value
        and "~" not in path.parts
        and ".." not in path.parts
    )

def _model_scout_reference_path(value: Any, source_path: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    local_path = source_path.parent / path
    return local_path if local_path.exists() else path
