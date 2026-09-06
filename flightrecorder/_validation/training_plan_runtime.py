"""Extracted validation implementation."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any
from ..agentic_training_runtime import AGENTIC_TRAINING_RUNTIME_PREFLIGHT_SCHEMA_VERSION, PLAN_READY_RECOMMENDATION, RUNTIME_BLOCK_RECOMMENDATION, RUNTIME_READY_RECOMMENDATION
from ..schema_registry import SchemaRegistryError, check_schema_contract, check_schema_file
from ..agentic_training_plan import ADVANCED_REWARD_MODES as PLAN_ADVANCED_REWARD_MODES, AGENTIC_TRAINING_PLAN_SCHEMA_VERSION, DEFAULT_EXECUTABLE_MODES as PLAN_DEFAULT_EXECUTABLE_MODES, FUTURE_RL_MODES as PLAN_FUTURE_RL_MODES, MODE_STAGE_SEQUENCES as PLAN_MODE_STAGE_SEQUENCES, MODE_VIEW_REQUIREMENTS as PLAN_MODE_VIEW_REQUIREMENTS, SUPPORTED_MODES as PLAN_SUPPORTED_MODES
from ..path_safety import path_has_symlink_component as _path_has_symlink_component, resolve_artifact_reference_path
from ..hashing import sha256_file as _sha256
from .primitives import ValidationTarget, _is_non_negative_int, _is_sha256, _is_string_list, _looks_absolute, _read_object, _require_equal, _sha256, _validate_allowed_keys, _validate_gate_like_checks

def validate_agentic_training_plan(path: str | Path) -> ValidationTarget:
    """Validate a dry-run agentic training-plan artifact."""
    plan_path = Path(path)
    target = ValidationTarget("agentic_training_plan", str(plan_path))
    plan = _read_object(plan_path, target, "agentic_training_plan.json")
    if plan is not None:
        _validate_agentic_training_plan(plan, target, plan_path)
    return target

def validate_agentic_training_runtime_preflight(path: str | Path) -> ValidationTarget:
    """Validate a side-effect-free agentic trainer runtime preflight."""
    preflight_path = Path(path)
    target = ValidationTarget("agentic_training_runtime_preflight", str(preflight_path))
    preflight = _read_object(preflight_path, target, "agentic_training_runtime_preflight.json")
    if preflight is None:
        return target
    schema_check = check_schema_contract(preflight, name_or_id="agentic_training_runtime_preflight")
    target.errors.extend(
        f"agentic_training_runtime_preflight schema: {error}"
        for error in schema_check.get("errors", [])
    )
    _validate_agentic_training_runtime_preflight(preflight, target, preflight_path)
    return target

_AGENTIC_TRAINING_PLAN_READY_RECOMMENDATION = "ready_for_external_trainer_plan"

_AGENTIC_TRAINING_PLAN_BLOCK_RECOMMENDATION = "block_external_training"

_AGENTIC_TRAINING_PLAN_DEFAULT_REQUIRED_FLAG = None

_AGENTIC_TRAINING_PLAN_ADVANCED_REQUIRED_FLAG = "--allow-advanced-training"

_AGENTIC_TRAINING_PLAN_FUTURE_RL_REQUIRED_FLAG = "--allow-future-rl"

_AGENTIC_TRAINING_PLAN_REWARD_VALIDATE_MODES = {
    "dpo",
    "sft_then_dpo",
    "reward_model",
    "process_rewards",
    "grpo",
    "rl",
}

_AGENTIC_TRAINING_PLAN_REWARD_KIND = {
    "sft": "not_applicable",
    "action_sft": "not_applicable",
    "dpo": "preference_pairs",
    "sft_then_dpo": "preference_pairs",
    "reward_model": "scalar_or_preference_rewards",
    "process_rewards": "step_rewards",
    "grpo": "trl_grpo_reward_function",
    "rl": "external_rl_reward_function",
}

_AGENTIC_TRAINING_PLAN_REWARD_SIGNATURE = {
    "sft": "",
    "action_sft": "",
    "dpo": "",
    "sft_then_dpo": "",
    "reward_model": "",
    "process_rewards": "",
    "grpo": "reward_fn(prompts, completions, **kwargs) -> list[float]",
    "rl": "reward_fn(episodes, actions, **kwargs) -> list[float]",
}

_AGENTIC_TRAINING_PLAN_KEYS = {
    "schema_version",
    "created_at",
    "plan_path",
    "mode",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "blocked_reasons",
    "input_manifests",
    "selected_views",
    "mode_contract",
    "trainer_plan",
    "execution",
    "handoff_contract",
    "notes",
}

_AGENTIC_TRAINING_PLAN_CHECK_KEYS = {"id", "passed", "actual", "expected", "summary"}

_AGENTIC_TRAINING_PLAN_INPUT_MANIFEST_KEYS = {"model", "dataset"}

_AGENTIC_TRAINING_PLAN_MODEL_MANIFEST_KEYS = {
    "path",
    "sha256",
    "size_bytes",
    "schema_version",
    "id",
    "candidate_id",
    "license_status",
    "license_allow_training",
    "license_allows_training",
    "compatibility_passed",
    "base_model",
}

_AGENTIC_TRAINING_PLAN_DATASET_MANIFEST_KEYS = {
    "path",
    "sha256",
    "size_bytes",
    "schema_version",
    "id",
    "version",
    "license_status",
    "license_allow_training",
    "license_allows_training",
    "redaction_status",
    "contains_unredacted_traces",
    "redaction_passed",
    "view_count",
    "views",
}

_AGENTIC_TRAINING_PLAN_VIEW_KEYS = {"name", "path", "row_count", "split", "schema_version"}

_AGENTIC_TRAINING_PLAN_MODE_CONTRACT_KEYS = {
    "mode",
    "category",
    "planning_gate",
    "stage_sequence",
    "required_view_groups",
    "selected_view_names",
    "data_requirements",
    "reward_contract",
    "side_effect_boundary",
    "external_runner_contract",
}

_AGENTIC_TRAINING_PLAN_PLANNING_GATE_KEYS = {"open", "required_flag", "reason"}

_AGENTIC_TRAINING_PLAN_DATA_REQUIREMENT_KEYS = {
    "id",
    "description",
    "view_groups",
    "required_schema_names",
    "minimum_rows_per_group",
    "satisfied",
    "evidence",
}

_AGENTIC_TRAINING_PLAN_DATA_REQUIREMENT_EVIDENCE_KEYS = {
    "candidate_views",
    "selected_view",
    "row_count",
    "schema_version",
}

_AGENTIC_TRAINING_PLAN_REWARD_CONTRACT_KEYS = {
    "required",
    "kind",
    "source_views",
    "external_runner_must_supply",
    "external_runner_must_validate",
    "flight_recorder_supplies_callable",
    "may_call_paid_services_by_default",
    "may_require_secrets_by_default",
    "must_not_use_unredacted_traces",
    "requires_calibration_or_human_review_gate",
    "callable_signature",
    "notes",
}

_AGENTIC_TRAINING_PLAN_SIDE_EFFECT_KEYS = {
    "dry_run_only",
    "training_started",
    "cloud_jobs_started",
    "model_downloads_started",
    "paid_model_grader_calls_started",
    "weights_updated",
    "provider_credentials_required_by_flight_recorder",
}

_AGENTIC_TRAINING_PLAN_EXTERNAL_RUNNER_KEYS = {
    "runner_owns_execution",
    "runner_must_revalidate_inputs",
    "runner_must_require_recommendation",
    "runner_must_validate_reward_contract",
    "runner_must_block_unredacted_traces",
}

_AGENTIC_TRAINING_PLAN_TRAINER_PLAN_KEYS = {
    "backend",
    "output_dir",
    "limit",
    "stage_sequence",
    "mode_required_view_groups",
    "extension_points",
}

_AGENTIC_TRAINING_PLAN_EXTENSION_POINT_KEYS = {"id", "status", "scope"}

_AGENTIC_TRAINING_PLAN_EXECUTION_KEYS = {
    "dry_run_only",
    "training_started",
    "model_downloads_started",
    "cloud_jobs_started",
    "paid_model_grader_calls_started",
    "weights_updated",
    "external_runner_command",
}

_AGENTIC_TRAINING_PLAN_HANDOFF_KEYS = {
    "flight_recorder_executed_training",
    "runner_owns_execution",
    "runner_must_require_recommendation",
    "requires_registered_model",
    "requires_registered_dataset",
    "requires_known_license_status",
    "requires_redacted_dataset",
    "disallow_unredacted_traces",
    "advanced_modes_require_explicit_opt_in",
    "future_rl_requires_explicit_opt_in",
    "flight_recorder_started_cloud_provider",
    "paid_model_grader_calls_started",
    "weights_updated_by_flight_recorder",
    "reward_function_owned_by_external_runner",
    "allowed_modes",
    "default_executable_modes",
    "advanced_reward_modes_blocked_by_default",
    "future_rl_modes_blocked_by_default",
}

def _validate_agentic_training_plan(plan: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(plan, _AGENTIC_TRAINING_PLAN_KEYS, target, "agentic_training_plan")
    _require_equal(plan, "schema_version", AGENTIC_TRAINING_PLAN_SCHEMA_VERSION, target, prefix="agentic_training_plan.")
    checks = plan.get("checks")
    if not isinstance(checks, list):
        target.errors.append("agentic_training_plan.checks must be a list.")
        checks = []
    for index, check in enumerate(checks):
        if isinstance(check, dict):
            _validate_allowed_keys(check, _AGENTIC_TRAINING_PLAN_CHECK_KEYS, target, f"agentic_training_plan.checks[{index}]")
    failed_checks = _validate_gate_like_checks(checks, target, "agentic_training_plan.checks")
    if plan.get("check_count") != len(checks):
        target.errors.append(f"agentic_training_plan.check_count expected {len(checks)}, got {plan.get('check_count')!r}.")
    if plan.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"agentic_training_plan.failed_check_count expected {failed_checks}, got {plan.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if not isinstance(plan.get("passed"), bool):
        target.errors.append("agentic_training_plan.passed must be a boolean.")
    elif plan.get("passed") != expected_passed:
        target.errors.append("agentic_training_plan.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = (
        _AGENTIC_TRAINING_PLAN_READY_RECOMMENDATION if expected_passed else _AGENTIC_TRAINING_PLAN_BLOCK_RECOMMENDATION
    )
    if plan.get("readiness") != expected_readiness:
        target.errors.append(f"agentic_training_plan.readiness expected {expected_readiness!r}, got {plan.get('readiness')!r}.")
    if plan.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"agentic_training_plan.recommendation expected {expected_recommendation!r}, got {plan.get('recommendation')!r}."
        )
    mode = plan.get("mode")
    if mode not in PLAN_SUPPORTED_MODES:
        target.errors.append(f"agentic_training_plan.mode must be one of {list(PLAN_SUPPORTED_MODES)!r}.")
        mode = ""
    if not isinstance(plan.get("created_at"), str) or not plan.get("created_at"):
        target.errors.append("agentic_training_plan.created_at must be a non-empty string.")
    if not isinstance(plan.get("plan_path"), str) or not plan.get("plan_path"):
        target.errors.append("agentic_training_plan.plan_path must be a non-empty string.")
    if not _is_string_list(plan.get("blocked_reasons")):
        target.errors.append("agentic_training_plan.blocked_reasons must be a list of strings.")
    elif expected_passed and plan.get("blocked_reasons"):
        target.errors.append("agentic_training_plan.blocked_reasons must be empty when passed.")
    elif not expected_passed and len(plan.get("blocked_reasons", [])) != failed_checks:
        target.errors.append("agentic_training_plan.blocked_reasons must match failed_check_count.")
    if not _is_string_list(plan.get("notes")):
        target.errors.append("agentic_training_plan.notes must be a list of strings.")

    _validate_agentic_training_plan_input_manifests(plan.get("input_manifests"), target)
    selected_view_names = _validate_agentic_training_plan_views(
        plan.get("selected_views"),
        target,
        "agentic_training_plan.selected_views",
    )
    _validate_agentic_training_plan_mode_contract(
        plan.get("mode_contract"),
        target,
        mode,
        selected_view_names,
        expected_passed,
    )
    _validate_agentic_training_plan_trainer_plan(plan.get("trainer_plan"), target, mode)
    _validate_agentic_training_plan_execution(plan.get("execution"), target)
    _validate_agentic_training_plan_handoff_contract(plan.get("handoff_contract"), target)
    target.details.update({"mode": mode, "passed": plan.get("passed"), "failed_check_count": failed_checks})

def _validate_agentic_training_plan_input_manifests(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("agentic_training_plan.input_manifests must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_PLAN_INPUT_MANIFEST_KEYS, target, "agentic_training_plan.input_manifests")
    _validate_agentic_training_plan_manifest_ref(
        value.get("model"),
        target,
        "agentic_training_plan.input_manifests.model",
        _AGENTIC_TRAINING_PLAN_MODEL_MANIFEST_KEYS,
        "model",
    )
    _validate_agentic_training_plan_manifest_ref(
        value.get("dataset"),
        target,
        "agentic_training_plan.input_manifests.dataset",
        _AGENTIC_TRAINING_PLAN_DATASET_MANIFEST_KEYS,
        "dataset",
    )

def _validate_agentic_training_plan_manifest_ref(
    value: Any,
    target: ValidationTarget,
    label: str,
    allowed_keys: set[str],
    kind: str,
) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, allowed_keys, target, label)
    if not isinstance(value.get("path"), str) or not value.get("path"):
        target.errors.append(f"{label}.path must be a non-empty string.")
    if not _is_sha256(value.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string.")
    if not _is_non_negative_int(value.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    for field_name in ("schema_version", "id", "license_status"):
        if not isinstance(value.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    for field_name in ("license_allow_training", "license_allows_training"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if kind == "model":
        for field_name in ("candidate_id", "base_model"):
            if not isinstance(value.get(field_name), str):
                target.errors.append(f"{label}.{field_name} must be a string.")
        if not isinstance(value.get("compatibility_passed"), bool):
            target.errors.append(f"{label}.compatibility_passed must be a boolean.")
        return
    if kind == "dataset":
        for field_name in ("version", "redaction_status"):
            if not isinstance(value.get(field_name), str):
                target.errors.append(f"{label}.{field_name} must be a string.")
        for field_name in ("contains_unredacted_traces", "redaction_passed"):
            if not isinstance(value.get(field_name), bool):
                target.errors.append(f"{label}.{field_name} must be a boolean.")
        views = value.get("views")
        if not isinstance(views, dict):
            target.errors.append(f"{label}.views must be an object.")
            view_count = 0
        else:
            view_count = len(views)
            for name, view in views.items():
                if not isinstance(name, str) or not name:
                    target.errors.append(f"{label}.views keys must be non-empty strings.")
                    continue
                _validate_agentic_training_plan_view(view, target, f"{label}.views.{name}")
        if value.get("view_count") != view_count:
            target.errors.append(f"{label}.view_count expected {view_count}, got {value.get('view_count')!r}.")

def _validate_agentic_training_plan_views(value: Any, target: ValidationTarget, label: str) -> list[str]:
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list.")
        return []
    names: list[str] = []
    for index, view in enumerate(value):
        view_label = f"{label}[{index}]"
        if _validate_agentic_training_plan_view(view, target, view_label):
            names.append(str(view["name"]))
    return names

def _validate_agentic_training_plan_view(value: Any, target: ValidationTarget, label: str) -> bool:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return False
    _validate_allowed_keys(value, _AGENTIC_TRAINING_PLAN_VIEW_KEYS, target, label)
    ok = True
    for field_name in ("name", "path", "split", "schema_version"):
        if not isinstance(value.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
            ok = False
    if not isinstance(value.get("name"), str) or not value.get("name"):
        target.errors.append(f"{label}.name must be a non-empty string.")
        ok = False
    if not _is_non_negative_int(value.get("row_count")):
        target.errors.append(f"{label}.row_count must be a non-negative integer.")
        ok = False
    return ok

def _validate_agentic_training_plan_mode_contract(
    value: Any,
    target: ValidationTarget,
    mode: str,
    selected_view_names: list[str],
    expected_passed: bool,
) -> None:
    label = "agentic_training_plan.mode_contract"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_PLAN_MODE_CONTRACT_KEYS, target, label)
    if mode and value.get("mode") != mode:
        target.errors.append(f"{label}.mode must match agentic_training_plan.mode.")
    expected_category = _agentic_training_plan_mode_category(mode)
    if expected_category and value.get("category") != expected_category:
        target.errors.append(f"{label}.category expected {expected_category!r}, got {value.get('category')!r}.")
    expected_stage_sequence = list(PLAN_MODE_STAGE_SEQUENCES.get(mode, []))
    if value.get("stage_sequence") != expected_stage_sequence:
        target.errors.append(f"{label}.stage_sequence expected {expected_stage_sequence!r}, got {value.get('stage_sequence')!r}.")
    expected_view_groups = [list(group) for group in PLAN_MODE_VIEW_REQUIREMENTS.get(mode, ())]
    if value.get("required_view_groups") != expected_view_groups:
        target.errors.append(f"{label}.required_view_groups expected {expected_view_groups!r}, got {value.get('required_view_groups')!r}.")
    if value.get("selected_view_names") != selected_view_names:
        target.errors.append(f"{label}.selected_view_names must match selected_views names.")
    _validate_agentic_training_plan_planning_gate(value.get("planning_gate"), target, mode, expected_passed)
    _validate_agentic_training_plan_data_requirements(value.get("data_requirements"), target)
    _validate_agentic_training_plan_reward_contract(value.get("reward_contract"), target, mode)
    _validate_agentic_training_plan_side_effect_boundary(value.get("side_effect_boundary"), target)
    _validate_agentic_training_plan_external_runner_contract(value.get("external_runner_contract"), target, mode)

def _validate_agentic_training_plan_planning_gate(
    value: Any,
    target: ValidationTarget,
    mode: str,
    expected_passed: bool,
) -> None:
    label = "agentic_training_plan.mode_contract.planning_gate"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_PLAN_PLANNING_GATE_KEYS, target, label)
    if "required_flag" not in value:
        target.errors.append(f"{label}.required_flag is missing.")
    expected_flag = _agentic_training_plan_required_flag(mode)
    if value.get("required_flag") != expected_flag:
        target.errors.append(f"{label}.required_flag expected {expected_flag!r}, got {value.get('required_flag')!r}.")
    if not isinstance(value.get("open"), bool):
        target.errors.append(f"{label}.open must be a boolean.")
    elif expected_passed and value.get("open") is not True:
        target.errors.append(f"{label}.open must be true when the plan passed.")
    if not isinstance(value.get("reason"), str) or not value.get("reason"):
        target.errors.append(f"{label}.reason must be a non-empty string.")
    if mode in PLAN_DEFAULT_EXECUTABLE_MODES and value.get("open") is not True:
        target.errors.append(f"{label}.open must be true for default executable modes.")

def _validate_agentic_training_plan_data_requirements(value: Any, target: ValidationTarget) -> None:
    label = "agentic_training_plan.mode_contract.data_requirements"
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list.")
        return
    if not value:
        target.errors.append(f"{label} must not be empty.")
    for index, requirement in enumerate(value):
        item_label = f"{label}[{index}]"
        if not isinstance(requirement, dict):
            target.errors.append(f"{item_label} must be an object.")
            continue
        _validate_allowed_keys(requirement, _AGENTIC_TRAINING_PLAN_DATA_REQUIREMENT_KEYS, target, item_label)
        for field_name in ("id", "description"):
            if not isinstance(requirement.get(field_name), str) or not requirement.get(field_name):
                target.errors.append(f"{item_label}.{field_name} must be a non-empty string.")
        if not _is_nested_string_list(requirement.get("view_groups")):
            target.errors.append(f"{item_label}.view_groups must be a list of string lists.")
        if not _is_string_list(requirement.get("required_schema_names")):
            target.errors.append(f"{item_label}.required_schema_names must be a list of strings.")
        if requirement.get("minimum_rows_per_group") != 1:
            target.errors.append(f"{item_label}.minimum_rows_per_group must be 1.")
        if not isinstance(requirement.get("satisfied"), bool):
            target.errors.append(f"{item_label}.satisfied must be a boolean.")
        evidence = requirement.get("evidence")
        if not isinstance(evidence, list):
            target.errors.append(f"{item_label}.evidence must be a list.")
            continue
        for evidence_index, row in enumerate(evidence):
            _validate_agentic_training_plan_data_requirement_evidence(row, target, f"{item_label}.evidence[{evidence_index}]")

def _validate_agentic_training_plan_data_requirement_evidence(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_PLAN_DATA_REQUIREMENT_EVIDENCE_KEYS, target, label)
    if not _is_string_list(value.get("candidate_views")):
        target.errors.append(f"{label}.candidate_views must be a list of strings.")
    if not isinstance(value.get("selected_view"), str):
        target.errors.append(f"{label}.selected_view must be a string.")
    if not _is_non_negative_int(value.get("row_count")):
        target.errors.append(f"{label}.row_count must be a non-negative integer.")
    if not isinstance(value.get("schema_version"), str):
        target.errors.append(f"{label}.schema_version must be a string.")

def _validate_agentic_training_plan_reward_contract(value: Any, target: ValidationTarget, mode: str) -> None:
    label = "agentic_training_plan.mode_contract.reward_contract"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_PLAN_REWARD_CONTRACT_KEYS, target, label)
    expected_values = {
        "required": mode not in {"sft", "action_sft"},
        "external_runner_must_supply": mode in PLAN_FUTURE_RL_MODES,
        "external_runner_must_validate": mode in _AGENTIC_TRAINING_PLAN_REWARD_VALIDATE_MODES,
        "flight_recorder_supplies_callable": False,
        "may_call_paid_services_by_default": False,
        "may_require_secrets_by_default": False,
        "must_not_use_unredacted_traces": True,
        "requires_calibration_or_human_review_gate": mode in {"reward_model", "process_rewards", "grpo", "rl"},
    }
    for field_name, expected_value in expected_values.items():
        if value.get(field_name) is not expected_value:
            target.errors.append(f"{label}.{field_name} must be {expected_value!r}.")
    expected_kind = _AGENTIC_TRAINING_PLAN_REWARD_KIND.get(mode)
    if expected_kind and value.get("kind") != expected_kind:
        target.errors.append(f"{label}.kind expected {expected_kind!r}, got {value.get('kind')!r}.")
    expected_signature = _AGENTIC_TRAINING_PLAN_REWARD_SIGNATURE.get(mode)
    if expected_signature is not None and value.get("callable_signature") != expected_signature:
        target.errors.append(f"{label}.callable_signature expected {expected_signature!r}, got {value.get('callable_signature')!r}.")
    if not _is_string_list(value.get("source_views")):
        target.errors.append(f"{label}.source_views must be a list of strings.")
    if not _is_string_list(value.get("notes")):
        target.errors.append(f"{label}.notes must be a list of strings.")

def _validate_agentic_training_plan_side_effect_boundary(value: Any, target: ValidationTarget) -> None:
    label = "agentic_training_plan.mode_contract.side_effect_boundary"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_PLAN_SIDE_EFFECT_KEYS, target, label)
    expected_values = {
        "dry_run_only": True,
        "training_started": False,
        "cloud_jobs_started": False,
        "model_downloads_started": False,
        "paid_model_grader_calls_started": False,
        "weights_updated": False,
        "provider_credentials_required_by_flight_recorder": False,
    }
    for field_name, expected_value in expected_values.items():
        if value.get(field_name) is not expected_value:
            target.errors.append(f"{label}.{field_name} must be {expected_value!r}.")

def _validate_agentic_training_plan_external_runner_contract(value: Any, target: ValidationTarget, mode: str) -> None:
    label = "agentic_training_plan.mode_contract.external_runner_contract"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_PLAN_EXTERNAL_RUNNER_KEYS, target, label)
    expected_values = {
        "runner_owns_execution": True,
        "runner_must_revalidate_inputs": True,
        "runner_must_require_recommendation": _AGENTIC_TRAINING_PLAN_READY_RECOMMENDATION,
        "runner_must_validate_reward_contract": mode in _AGENTIC_TRAINING_PLAN_REWARD_VALIDATE_MODES,
        "runner_must_block_unredacted_traces": True,
    }
    for field_name, expected_value in expected_values.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} must be {expected_value!r}.")

def _validate_agentic_training_plan_trainer_plan(value: Any, target: ValidationTarget, mode: str) -> None:
    label = "agentic_training_plan.trainer_plan"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_PLAN_TRAINER_PLAN_KEYS, target, label)
    if not isinstance(value.get("backend"), str) or not value.get("backend"):
        target.errors.append(f"{label}.backend must be a non-empty string.")
    if not isinstance(value.get("output_dir"), str):
        target.errors.append(f"{label}.output_dir must be a string.")
    if "limit" not in value:
        target.errors.append(f"{label}.limit is missing.")
    limit = value.get("limit")
    if "limit" in value and limit is not None and (not _is_non_negative_int(limit) or limit <= 0):
        target.errors.append(f"{label}.limit must be a positive integer or null.")
    expected_stage_sequence = list(PLAN_MODE_STAGE_SEQUENCES.get(mode, []))
    if value.get("stage_sequence") != expected_stage_sequence:
        target.errors.append(f"{label}.stage_sequence expected {expected_stage_sequence!r}, got {value.get('stage_sequence')!r}.")
    expected_view_groups = [list(group) for group in PLAN_MODE_VIEW_REQUIREMENTS.get(mode, ())]
    if value.get("mode_required_view_groups") != expected_view_groups:
        target.errors.append(
            f"{label}.mode_required_view_groups expected {expected_view_groups!r}, got {value.get('mode_required_view_groups')!r}."
        )
    extension_points = value.get("extension_points")
    if not isinstance(extension_points, list):
        target.errors.append(f"{label}.extension_points must be a list.")
        return
    for index, extension_point in enumerate(extension_points):
        point_label = f"{label}.extension_points[{index}]"
        if not isinstance(extension_point, dict):
            target.errors.append(f"{point_label} must be an object.")
            continue
        _validate_allowed_keys(extension_point, _AGENTIC_TRAINING_PLAN_EXTENSION_POINT_KEYS, target, point_label)
        for field_name in ("id", "status", "scope"):
            if not isinstance(extension_point.get(field_name), str) or not extension_point.get(field_name):
                target.errors.append(f"{point_label}.{field_name} must be a non-empty string.")

def _validate_agentic_training_plan_execution(value: Any, target: ValidationTarget) -> None:
    label = "agentic_training_plan.execution"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_PLAN_EXECUTION_KEYS, target, label)
    expected_values = {
        "dry_run_only": True,
        "training_started": False,
        "model_downloads_started": False,
        "cloud_jobs_started": False,
        "paid_model_grader_calls_started": False,
        "weights_updated": False,
    }
    for field_name, expected_value in expected_values.items():
        if value.get(field_name) is not expected_value:
            target.errors.append(f"{label}.{field_name} must be {expected_value!r}.")
    command = value.get("external_runner_command")
    if not _is_string_list(command):
        target.errors.append(f"{label}.external_runner_command must be a list of strings.")
    elif not command:
        target.errors.append(f"{label}.external_runner_command must not be empty.")
    else:
        for index, item in enumerate(command):
            _validate_agentic_training_plan_command_token_public_path(
                target,
                f"{label}.external_runner_command[{index}]",
                item,
            )

def _validate_agentic_training_plan_command_token_public_path(target: ValidationTarget, label: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        return
    if _looks_absolute(value):
        target.errors.append(f"{label} must use a relative command token or redacted placeholder.")
        return
    _, separator, token_value = value.partition("=")
    if separator and _looks_absolute(token_value):
        target.errors.append(f"{label} must use a relative command token or redacted placeholder.")

def _validate_agentic_training_plan_handoff_contract(value: Any, target: ValidationTarget) -> None:
    label = "agentic_training_plan.handoff_contract"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_PLAN_HANDOFF_KEYS, target, label)
    expected_values = {
        "flight_recorder_executed_training": False,
        "runner_owns_execution": True,
        "runner_must_require_recommendation": _AGENTIC_TRAINING_PLAN_READY_RECOMMENDATION,
        "requires_registered_model": True,
        "requires_registered_dataset": True,
        "requires_known_license_status": True,
        "requires_redacted_dataset": True,
        "disallow_unredacted_traces": True,
        "advanced_modes_require_explicit_opt_in": True,
        "future_rl_requires_explicit_opt_in": True,
        "flight_recorder_started_cloud_provider": False,
        "paid_model_grader_calls_started": False,
        "weights_updated_by_flight_recorder": False,
        "reward_function_owned_by_external_runner": True,
    }
    for field_name, expected_value in expected_values.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} must be {expected_value!r}.")
    expected_lists = {
        "allowed_modes": list(PLAN_SUPPORTED_MODES),
        "default_executable_modes": sorted(PLAN_DEFAULT_EXECUTABLE_MODES),
        "advanced_reward_modes_blocked_by_default": sorted(PLAN_ADVANCED_REWARD_MODES),
        "future_rl_modes_blocked_by_default": sorted(PLAN_FUTURE_RL_MODES),
    }
    for field_name, expected_value in expected_lists.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} expected {expected_value!r}, got {value.get(field_name)!r}.")

def _agentic_training_plan_mode_category(mode: str) -> str:
    if mode in PLAN_DEFAULT_EXECUTABLE_MODES:
        return "default_executable"
    if mode in PLAN_ADVANCED_REWARD_MODES:
        return "advanced_reward"
    if mode in PLAN_FUTURE_RL_MODES:
        return "future_rl"
    return ""

def _agentic_training_plan_required_flag(mode: str) -> str | None:
    if mode in PLAN_ADVANCED_REWARD_MODES:
        return _AGENTIC_TRAINING_PLAN_ADVANCED_REQUIRED_FLAG
    if mode in PLAN_FUTURE_RL_MODES:
        return _AGENTIC_TRAINING_PLAN_FUTURE_RL_REQUIRED_FLAG
    if mode in PLAN_DEFAULT_EXECUTABLE_MODES:
        return _AGENTIC_TRAINING_PLAN_DEFAULT_REQUIRED_FLAG
    return None

def _is_nested_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(_is_string_list(item) for item in value)

def _validate_agentic_training_runtime_preflight(
    preflight: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    from ..agentic_training_runtime import (
        _blocked_reasons,
        _dependency_checks,
        _dependency_policy,
        _json_schema_record,
        _mode_contract_check,
        _read_json_object,
        _view_checks,
    )

    _require_equal(
        preflight,
        "schema_version",
        AGENTIC_TRAINING_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
        target,
        prefix="agentic_training_runtime_preflight.",
    )
    plan_path = _resolve_runtime_preflight_plan_path(preflight.get("plan_path"), source_path)
    if plan_path is None:
        plan_payload: dict[str, Any] = {}
        plan_read_errors = ["plan_path is missing or invalid"]
        plan_schema = {"passed": False, "error_count": 1, "errors": plan_read_errors}
        expected_view_checks: list[dict[str, Any]] = []
    elif plan_path.is_symlink() or _path_has_symlink_component(plan_path, include_leaf=False):
        plan_payload = {}
        plan_read_errors = ["plan path traverses a symlink component"]
        plan_schema = {"passed": False, "error_count": 1, "errors": plan_read_errors}
        expected_view_checks = []
    else:
        plan_payload, plan_read_errors = _read_json_object(plan_path)
        plan_schema = _json_schema_record(plan_path, "agentic_training_plan")
        expected_view_checks = _view_checks(plan_payload, plan_path) if not plan_read_errors else []

    plan_semantic_errors: list[str] = []
    if plan_path is not None and plan_path.is_file() and not plan_read_errors:
        plan_semantic_errors = validate_agentic_training_plan(plan_path).errors
    trainer_plan = plan_payload.get("trainer_plan") if isinstance(plan_payload.get("trainer_plan"), dict) else {}
    expected_mode = str(plan_payload.get("mode") or "")
    expected_backend = str(trainer_plan.get("backend") or "external")
    expected_mode_contract = _mode_contract_check(plan_payload, expected_mode)
    plan_ready = (
        not plan_read_errors
        and plan_schema.get("passed") is True
        and not plan_semantic_errors
        and plan_payload.get("passed") is True
        and plan_payload.get("recommendation") == PLAN_READY_RECOMMENDATION
    )

    if preflight.get("plan_mode") != expected_mode:
        target.errors.append(
            f"agentic_training_runtime_preflight.plan_mode expected {expected_mode!r}, got {preflight.get('plan_mode')!r}."
        )
    if preflight.get("backend") != expected_backend:
        target.errors.append(
            f"agentic_training_runtime_preflight.backend expected {expected_backend!r}, got {preflight.get('backend')!r}."
        )
    if plan_path is not None and plan_path.is_file() and preflight.get("plan_sha256") != _sha256(plan_path):
        target.errors.append("agentic_training_runtime_preflight.plan_sha256 does not match the current plan file.")

    plan_check = preflight.get("plan_check") if isinstance(preflight.get("plan_check"), dict) else {}
    expected_plan_check = {
        "exists": bool(plan_path and plan_path.exists()),
        "regular_file": bool(plan_path and plan_path.is_file()),
        "schema_passed": plan_schema.get("passed") is True,
        "error_count": plan_schema.get("error_count"),
        "errors": plan_schema.get("errors"),
        "required_recommendation": PLAN_READY_RECOMMENDATION,
        "observed_recommendation": plan_payload.get("recommendation"),
        "observed_passed": plan_payload.get("passed"),
    }
    for field_name, expected_value in expected_plan_check.items():
        if plan_check.get(field_name) != expected_value:
            target.errors.append(
                "agentic_training_runtime_preflight.plan_check."
                f"{field_name} expected {expected_value!r}, got {plan_check.get(field_name)!r}."
            )

    view_checks = preflight.get("view_checks")
    if view_checks != expected_view_checks:
        target.errors.append(
            "agentic_training_runtime_preflight.view_checks do not match the current plan-selected trainer views."
        )
        view_checks = view_checks if isinstance(view_checks, list) else []
    mode_contract = preflight.get("mode_contract_check")
    if mode_contract != expected_mode_contract:
        target.errors.append(
            "agentic_training_runtime_preflight.mode_contract_check does not match the source plan mode contract."
        )

    dependency_policy = preflight.get("dependency_policy")
    if not isinstance(dependency_policy, dict):
        target.errors.append("agentic_training_runtime_preflight.dependency_policy must be an object.")
        dependency_policy = {}
    recorded_overrides = dependency_policy.get("override_modules")
    recorded_skip_defaults = dependency_policy.get("skip_default_modules")
    expected_dependency_policy = _dependency_policy(
        expected_backend,
        recorded_overrides if isinstance(recorded_overrides, list) else (),
        recorded_skip_defaults if isinstance(recorded_skip_defaults, bool) else False,
    )
    if dependency_policy != expected_dependency_policy:
        target.errors.append(
            "agentic_training_runtime_preflight.dependency_policy must match the backend and declared override policy."
        )

    dependency_checks = preflight.get("dependency_checks")
    if not isinstance(dependency_checks, list):
        target.errors.append("agentic_training_runtime_preflight.dependency_checks must be a list.")
        dependency_checks = []
    expected_dependency_checks = _dependency_checks(expected_dependency_policy)
    if dependency_checks != expected_dependency_checks:
        target.errors.append(
            "agentic_training_runtime_preflight.dependency_checks must match fresh dependency probes for the declared policy."
        )
    for index, check in enumerate(dependency_checks):
        if not isinstance(check, dict):
            target.errors.append(f"agentic_training_runtime_preflight.dependency_checks[{index}] must be an object.")
            continue
        expected_passed = check.get("available") is True if check.get("required") is True else True
        if check.get("passed") is not expected_passed:
            target.errors.append(
                f"agentic_training_runtime_preflight.dependency_checks[{index}].passed must match availability."
            )

    expected_check_states = {
        "plan_json_readable": not plan_read_errors,
        "plan_schema_passed": plan_schema.get("passed") is True,
        "plan_recommendation_ready": plan_ready,
        "selected_views_schema_passed": bool(expected_view_checks)
        and all(view.get("passed") is True for view in expected_view_checks),
        "mode_contract_ready": expected_mode_contract.get("passed") is True,
        "runtime_dependencies_available": all(
            isinstance(check, dict) and check.get("passed") is True for check in expected_dependency_checks
        )
        and bool(expected_dependency_checks),
        "flight_recorder_did_not_launch_training": True,
        "model_downloads_not_started": True,
    }
    checks = preflight.get("checks")
    if not isinstance(checks, list):
        target.errors.append("agentic_training_runtime_preflight.checks must be a list.")
        checks = []
    failed_check_count = _validate_gate_like_checks(
        checks,
        target,
        "agentic_training_runtime_preflight.checks",
    )
    checks_by_id = {
        check.get("id"): check
        for check in checks
        if isinstance(check, dict) and isinstance(check.get("id"), str)
    }
    if len(checks_by_id) != len(checks):
        target.errors.append("agentic_training_runtime_preflight.checks must have unique string ids.")
    if set(checks_by_id) != set(expected_check_states):
        target.errors.append("agentic_training_runtime_preflight.checks must contain the canonical runtime check ids.")
    for check_id, expected_passed in expected_check_states.items():
        check = checks_by_id.get(check_id)
        if isinstance(check, dict) and check.get("passed") is not expected_passed:
            target.errors.append(
                f"agentic_training_runtime_preflight.checks[{check_id!r}].passed expected {expected_passed!r}."
            )

    expected_passed = all(expected_check_states.values())
    if preflight.get("check_count") != len(checks):
        target.errors.append(
            f"agentic_training_runtime_preflight.check_count expected {len(checks)}, got {preflight.get('check_count')!r}."
        )
    if preflight.get("failed_check_count") != failed_check_count:
        target.errors.append(
            "agentic_training_runtime_preflight.failed_check_count must match failed checks."
        )
    if preflight.get("passed") is not expected_passed:
        target.errors.append("agentic_training_runtime_preflight.passed must match recomputed runtime readiness.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = RUNTIME_READY_RECOMMENDATION if expected_passed else RUNTIME_BLOCK_RECOMMENDATION
    if preflight.get("readiness") != expected_readiness:
        target.errors.append(
            f"agentic_training_runtime_preflight.readiness expected {expected_readiness!r}."
        )
    if preflight.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"agentic_training_runtime_preflight.recommendation expected {expected_recommendation!r}."
        )

    blocked_reasons = preflight.get("blocked_reasons")
    expected_blocked_reasons = _blocked_reasons(
        [check for check in checks if isinstance(check, dict) and check.get("passed") is False],
        expected_dependency_checks,
        [check for check in view_checks if isinstance(check, dict)],
        expected_mode_contract,
    )
    if blocked_reasons != expected_blocked_reasons:
        target.errors.append(
            "agentic_training_runtime_preflight.blocked_reasons must match failed checks and source readiness."
        )

    _validate_runtime_preflight_side_effect_boundary(preflight.get("execution_boundary"), target)
    _validate_runtime_preflight_handoff(preflight.get("handoff_contract"), target)
    if not _is_string_list(preflight.get("notes")):
        target.errors.append("agentic_training_runtime_preflight.notes must be a list of strings.")
    target.details.update(
        {
            "passed": preflight.get("passed"),
            "readiness": preflight.get("readiness"),
            "backend": preflight.get("backend"),
            "check_count": len(checks),
        }
    )

def _resolve_runtime_preflight_plan_path(value: Any, source_path: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return resolve_artifact_reference_path(value, source_path)

def _validate_runtime_preflight_side_effect_boundary(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("agentic_training_runtime_preflight.execution_boundary must be an object.")
        return
    expected = {
        "dry_run_preflight_only": True,
        "flight_recorder_launched_training": False,
        "training_started": False,
        "model_downloads_started": False,
        "cloud_jobs_started": False,
        "paid_model_grader_calls_started": False,
        "weights_updated": False,
        "trainer_modules_imported": False,
        "dependency_resolution_method": "importlib.util.find_spec",
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(
                "agentic_training_runtime_preflight.execution_boundary."
                f"{field_name} expected {expected_value!r}."
            )

def _validate_runtime_preflight_handoff(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("agentic_training_runtime_preflight.handoff_contract must be an object.")
        return
    expected = {
        "runner_owns_execution": True,
        "runner_must_require_recommendation": RUNTIME_READY_RECOMMENDATION,
        "requires_registered_inputs": True,
        "requires_registered_model": True,
        "requires_registered_dataset": True,
        "requires_mode_contract": True,
        "requires_mode_contract_ready": True,
        "requires_known_license_status": True,
        "requires_redacted_dataset": True,
        "disallow_unredacted_traces": True,
        "flight_recorder_launched_training": False,
        "model_downloads_started": False,
        "flight_recorder_started_cloud_provider": False,
        "paid_model_grader_calls_started": False,
        "weights_updated_by_flight_recorder": False,
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(
                f"agentic_training_runtime_preflight.handoff_contract.{field_name} expected {expected_value!r}."
            )
