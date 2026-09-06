"""Extracted validation implementation."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any
from ..agentic_training_loop_plan import AGENTIC_TRAINING_LOOP_PLAN_SCHEMA_VERSION, CLOUD_TRAINING_LINEAGE_ARTIFACT_ROLES, CLOUD_TRAINING_HANDOFF_LINEAGE_LINKS, PHASES as AGENTIC_TRAINING_LOOP_PHASES, PLAN_READINESS_CHECK_IDS, PLAN_REQUIRED_ARTIFACT_ROLES, build_agentic_training_loop_plan as _build_agentic_training_loop_plan, _cloud_training_launch_receipt_semantic_passed as _loop_cloud_training_launch_receipt_semantic_passed, _cloud_training_status_receipt_semantic_passed as _loop_cloud_training_status_receipt_semantic_passed, _cloud_training_completion_state as _loop_cloud_training_completion_state, _cloud_training_completion_status as _loop_cloud_training_completion_status, _agentic_training_result_plan_bound as _loop_agentic_training_result_plan_bound, _execution_completion as _loop_execution_completion, _execution_result_status as _loop_execution_result_status, _external_eval_receipt_semantic_passed as _loop_external_eval_receipt_semantic_passed, _phase_row as _build_agentic_training_loop_phase
from ..agentic_loop_ledger import AGENTIC_LOOP_LEDGER_SCHEMA_VERSION, _decision as _build_agentic_loop_ledger_decision, _iteration_record as _build_agentic_loop_ledger_iteration, _metrics as _build_agentic_loop_ledger_metrics, _readiness_digest as _build_agentic_loop_ledger_readiness_digest
from ..agentic_loop_governance import AGENTIC_LOOP_GOVERNANCE_RECEIPT_SCHEMA_VERSION, GOVERNANCE_ACTIONS, _execution_boundary as _build_agentic_loop_governance_boundary, _receipt_projection as _build_agentic_loop_governance_projection, _source_ledger_ref as _build_agentic_loop_governance_ledger_ref
from ..source_contract import get_active_opaque_output_attestation, inspect_artifact_source
from ..next_iteration_schedule import NEXT_ITERATION_SCHEDULE_SCHEMA_VERSION
from ..path_safety import path_has_symlink_component as _path_has_symlink_component, resolve_artifact_reference_path
from ..hashing import sha256_file as _sha256
from .cloud import _cloud_training_check_by_id
from .evaluation_serving import validate_eval_summary
from .governance import _resolve_promotion_decision_artifact_path, _validate_promotion_decision_artifact_hash, validate_promotion_decision, validate_promotion_ledger
from .primitives import ValidationTarget, _directory_contains_symlink, _directory_tree_fingerprint, _is_non_negative_int, _is_sha256, _is_string_list, _read_json_object_silent, _read_object, _require_equal, _sha256, _validate_allowed_keys, _validate_gate_like_checks
from .training_flow_result import _is_safe_agentic_training_result_path

def validate_agentic_training_loop_plan(path: str | Path) -> ValidationTarget:
    """Validate a closed-loop agentic training iteration plan."""
    plan_path = Path(path)
    target = ValidationTarget("agentic_training_loop_plan", str(plan_path))
    plan = _read_object(plan_path, target, "agentic_training_loop_plan.json")
    if plan is not None:
        _validate_agentic_training_loop_plan(plan, target, plan_path)
    return target

def validate_agentic_loop_ledger(path: str | Path) -> ValidationTarget:
    """Validate a longitudinal closed-loop iteration ledger."""
    ledger_path = Path(path)
    target = ValidationTarget("agentic_loop_ledger", str(ledger_path))
    ledger = _read_object(ledger_path, target, "agentic_loop_ledger.json")
    if ledger is not None:
        _validate_agentic_loop_ledger(ledger, target, ledger_path)
    return target

def validate_agentic_loop_governance_receipt(path: str | Path) -> ValidationTarget:
    """Validate a side-effect-free governance receipt over an agentic-loop ledger."""
    receipt_path = Path(path)
    target = ValidationTarget("agentic_loop_governance_receipt", str(receipt_path))
    receipt = _read_object(receipt_path, target, "agentic_loop_governance_receipt.json")
    if receipt is not None:
        _validate_agentic_loop_governance_receipt(receipt, target, receipt_path)
    return target

def validate_next_iteration_schedule(path: str | Path) -> ValidationTarget:
    """Validate one next-iteration schedule proposal."""
    schedule_path = Path(path)
    target = ValidationTarget("next_iteration_schedule", str(schedule_path))
    schedule = _read_object(schedule_path, target, "next_iteration_schedule.json")
    if schedule is not None:
        _validate_next_iteration_schedule(schedule, target, schedule_path)
    return target

AGENTIC_LOOP_CLOUD_TRAINING_HANDOFF_ROLES: tuple[str, ...] = (
    "cloud_training_provider_registry",
    "cloud_training_preflight",
    "cloud_training_artifact_manifest",
    "cloud_training_launch_plan",
    "cloud_training_launch_receipt",
    "cloud_training_status_receipt",
)

AGENTIC_LOOP_CLOUD_TRAINING_ROLES: tuple[str, ...] = (
    *AGENTIC_LOOP_CLOUD_TRAINING_HANDOFF_ROLES,
    "cloud_training_completion_receipt",
)

_AGENTIC_TRAINING_LOOP_PLAN_KEYS = {
    "schema_version",
    "created_at",
    "iteration_id",
    "plan_path",
    "objective",
    "participants",
    "passed",
    "plan_readiness",
    "execution_completion",
    "governance_readiness",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "blocked_reasons",
    "missing_phase_inputs",
    "artifact_count",
    "artifact_role_counts",
    "source_artifacts",
    "phases",
    "budget",
    "provider_constraints",
    "cloud_training",
    "cloud_training_receipt_state",
    "cloud_training_completion_state",
    "cloud_training_lineage",
    "external_eval_receipt_state",
    "execution_boundary",
    "handoff_contract",
    "next_iteration",
    "notes",
}

_AGENTIC_TRAINING_LOOP_CHECK_KEYS = {"id", "passed", "actual", "expected", "summary"}

_AGENTIC_TRAINING_LOOP_ARTIFACT_REF_KEYS = {
    "role",
    "path",
    "kind",
    "exists",
    "sha256",
    "size_bytes",
    "file_count",
    "contains_symlinks",
    "schema_version",
    "passed",
    "readiness",
}

_AGENTIC_TRAINING_LOOP_BUDGET_KEYS = {
    "max_rollouts",
    "max_training_examples",
    "max_cloud_cost_usd",
    "max_gpu_hours",
    "live_spend_allowed",
}

_AGENTIC_TRAINING_LOOP_PARTICIPANTS_KEYS = {"baseline_policy", "candidate_policy", "teacher_policy"}

_AGENTIC_TRAINING_LOOP_PROVIDER_CONSTRAINTS_KEYS = {
    "providers",
    "regions",
    "gpu_classes",
    "requires_cost_estimate",
    "requires_region_allowlist",
    "requires_secret_redaction",
}

_AGENTIC_TRAINING_LOOP_CLOUD_TRAINING_KEYS = {
    "required_artifacts",
    "present_artifacts",
    "missing_artifacts",
    "artifact_count",
    "provider_registry_present",
    "preflight_present",
    "artifact_manifest_present",
    "launch_plan_present",
    "launch_receipt_present",
    "status_receipt_present",
    "provider_api_calls_started",
    "cloud_jobs_started",
    "credential_values_recorded",
    "live_spend_allowed",
}

_AGENTIC_TRAINING_LOOP_CLOUD_RECEIPT_STATE_KEYS = {
    "launch_receipt_count",
    "status_receipt_count",
    "launch_receipt_passed",
    "status_receipt_passed",
    "receipts_passed",
    "launch_mode",
    "launch_readiness",
    "launch_recommendation",
    "live_launch_requested",
    "status_provider_status",
    "status_terminal",
    "status_not_started",
    "status_readiness",
    "status_recommendation",
    "provider_api_calls_started",
    "cloud_jobs_started",
    "provider_cancel_called",
    "credential_values_recorded",
    "cost_incurred_usd",
    "fail_closed",
}

_AGENTIC_TRAINING_LOOP_CLOUD_COMPLETION_STATE_KEYS = {
    "receipt_count",
    "integrity_passed",
    "execution_status",
    "execution_terminal",
    "successful",
    "governance_readiness",
    "cloud_training_completion_claims_allowed",
    "provider_id",
    "provider_job_id",
    "execution_id",
    "provider_identity_complete",
    "pipeline_provider_id",
    "provider_matches_pipeline",
    "candidate_model_id",
    "loop_candidate_model_id",
    "candidate_matches_loop",
    "training_result_candidate_model_id",
    "candidate_matches_training_result",
    "launch_plan_bound",
    "launch_receipt_bound",
    "status_receipt_bound",
    "source_bindings_complete",
    "output_artifact_manifest_sha256",
    "output_artifact_manifest_bound",
    "output_artifact_set_sha256",
    "output_artifact_set_bound",
    "artifact_count",
    "regular_artifact_count",
    "output_artifact_count",
}

_AGENTIC_TRAINING_LOOP_EXTERNAL_EVAL_RECEIPT_STATE_KEYS = {
    "receipt_count",
    "receipt_passed_count",
    "receipts_passed",
    "launch_mode",
    "readiness",
    "recommendation",
    "adapter_count",
    "ready_adapter_count",
    "dry_run_only",
    "live_benchmark_requested",
    "live_benchmarks_started",
    "provider_api_calls_started",
    "model_downloads_started",
    "credential_values_recorded",
    "weights_updated_by_flight_recorder",
    "cost_incurred_usd",
    "fail_closed",
}

_AGENTIC_TRAINING_LOOP_CLOUD_LINEAGE_KEYS = {
    "passed",
    "required_link_count",
    "matched_link_count",
    "missing_link_count",
    "mismatched_link_count",
    "ambiguous_link_count",
    "duplicate_role_count",
    "missing_links",
    "mismatched_links",
    "ambiguous_links",
    "duplicate_roles",
    "role_counts",
    "provider",
    "links",
}

_AGENTIC_TRAINING_LOOP_CLOUD_LINEAGE_LINK_KEYS = {
    "id",
    "source_role",
    "source_ref",
    "target_role",
    "source_artifact_count",
    "target_artifact_count",
    "source_schema_version",
    "target_schema_version",
    "source_ref_sha256",
    "target_sha256",
    "passed",
    "status",
}

_AGENTIC_TRAINING_LOOP_CLOUD_LINEAGE_PROVIDER_KEYS = {
    "registry_provider_ids",
    "pipeline_provider_ids",
    "pipeline_provider_id",
    "provider_by_role",
    "provider_consistent",
    "registry_contains_pipeline_provider",
}

_AGENTIC_TRAINING_LOOP_EXECUTION_BOUNDARY_KEYS = {
    "dry_run_plan_only",
    "cloud_jobs_started",
    "paid_model_grader_calls_started",
    "live_benchmarks_started",
    "model_downloads_started",
    "weights_updated_by_flight_recorder",
    "credential_values_recorded",
    "public_artifact_paths_redacted",
}

_AGENTIC_TRAINING_LOOP_HANDOFF_CONTRACT_KEYS = {
    "flight_recorder_controls_preflight_and_receipts",
    "external_trainers_own_weight_updates",
    "live_launch_requires_explicit_opt_in",
    "requires_environment_credentials_for_live",
    "requires_trainer_preflight",
    "requires_trainer_launch_check",
    "requires_calibrated_review_before_training_data",
    "requires_heldout_eval_before_promotion",
    "requires_governance_decision_before_alias_update",
    "default_live_execution_allowed",
}

_AGENTIC_TRAINING_LOOP_NEXT_ITERATION_KEYS = {"scheduled", "requires_governance_decision", "schedule", "recommendation"}

_AGENTIC_TRAINING_LOOP_PHASE_KEYS = {
    "id",
    "name",
    "status",
    "required_artifacts",
    "present_required_artifacts",
    "missing_required_artifacts",
    "non_completed_required_artifacts",
    "produces",
    "gate",
}

_AGENTIC_TRAINING_LOOP_ROLE_COUNT_KEYS = {"role", "count"}

def _validate_agentic_training_loop_object_keys(
    value: Any,
    allowed_keys: set[str],
    target: ValidationTarget,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return {}
    _validate_allowed_keys(value, allowed_keys, target, label)
    return value

def _validate_agentic_training_loop_role_count_rows(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, list):
        return
    for index, row in enumerate(value):
        if isinstance(row, dict):
            _validate_allowed_keys(row, _AGENTIC_TRAINING_LOOP_ROLE_COUNT_KEYS, target, f"{label}[{index}]")

def _validate_agentic_training_loop_plan(plan: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(plan, _AGENTIC_TRAINING_LOOP_PLAN_KEYS, target, "agentic_training_loop_plan")
    _require_equal(plan, "schema_version", AGENTIC_TRAINING_LOOP_PLAN_SCHEMA_VERSION, target, prefix="agentic_training_loop_plan.")
    checks = plan.get("checks")
    if not isinstance(checks, list):
        target.errors.append("agentic_training_loop_plan.checks must be a list.")
        checks = []
    else:
        for index, check in enumerate(checks):
            if isinstance(check, dict):
                _validate_allowed_keys(check, _AGENTIC_TRAINING_LOOP_CHECK_KEYS, target, f"agentic_training_loop_plan.checks[{index}]")
    failed_checks = _validate_gate_like_checks(checks, target, "agentic_training_loop_plan.checks")
    if plan.get("check_count") != len(checks):
        target.errors.append(f"agentic_training_loop_plan.check_count expected {len(checks)}, got {plan.get('check_count')!r}.")
    if plan.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"agentic_training_loop_plan.failed_check_count expected {failed_checks}, got {plan.get('failed_check_count')!r}."
        )
    expected_check_ids = PLAN_READINESS_CHECK_IDS | {
        "external_trainer_execution_completed",
        "external_cloud_training_completion_imported",
        "heldout_eval_is_fail_closed",
        "governance_required_for_promotion",
        "required_phase_inputs_present",
    }
    actual_check_ids = [str(check.get("id") or "") for check in checks if isinstance(check, dict)]
    check_contract_complete = len(actual_check_ids) == len(expected_check_ids) and set(actual_check_ids) == expected_check_ids
    if not check_contract_complete:
        target.errors.append(
            "agentic_training_loop_plan.checks must contain every canonical readiness, execution, and governance check exactly once."
        )
    if not isinstance(plan.get("iteration_id"), str) or not plan.get("iteration_id"):
        target.errors.append("agentic_training_loop_plan.iteration_id must be a non-empty string.")
    plan_path = plan.get("plan_path")
    if not isinstance(plan_path, str) or not plan_path:
        target.errors.append("agentic_training_loop_plan.plan_path must be a non-empty string.")
    elif not _is_safe_agentic_training_result_path(plan_path):
        target.errors.append("agentic_training_loop_plan.plan_path must be a safe relative path without traversal.")
    _validate_agentic_training_loop_object_keys(
        plan.get("budget"),
        _AGENTIC_TRAINING_LOOP_BUDGET_KEYS,
        target,
        "agentic_training_loop_plan.budget",
    )
    _validate_agentic_training_loop_object_keys(
        plan.get("participants"),
        _AGENTIC_TRAINING_LOOP_PARTICIPANTS_KEYS,
        target,
        "agentic_training_loop_plan.participants",
    )
    _validate_agentic_training_loop_object_keys(
        plan.get("provider_constraints"),
        _AGENTIC_TRAINING_LOOP_PROVIDER_CONSTRAINTS_KEYS,
        target,
        "agentic_training_loop_plan.provider_constraints",
    )
    _validate_agentic_training_loop_object_keys(
        plan.get("next_iteration"),
        _AGENTIC_TRAINING_LOOP_NEXT_ITERATION_KEYS,
        target,
        "agentic_training_loop_plan.next_iteration",
    )
    _validate_agentic_training_loop_role_count_rows(
        plan.get("artifact_role_counts"),
        target,
        "agentic_training_loop_plan.artifact_role_counts",
    )

    source_artifacts = plan.get("source_artifacts")
    artifact_count = 0
    if not isinstance(source_artifacts, dict):
        target.errors.append("agentic_training_loop_plan.source_artifacts must be an object.")
        source_artifacts = {}
    else:
        for role, refs in source_artifacts.items():
            if not isinstance(role, str) or not role:
                target.errors.append("agentic_training_loop_plan.source_artifacts keys must be non-empty strings.")
            if not isinstance(refs, list):
                target.errors.append(f"agentic_training_loop_plan.source_artifacts.{role} must be a list.")
                continue
            artifact_count += len(refs)
            for index, ref in enumerate(refs):
                _validate_agentic_training_loop_plan_ref(
                    ref,
                    target,
                    f"agentic_training_loop_plan.source_artifacts.{role}[{index}]",
                    source_path,
                )
    if plan.get("artifact_count") != artifact_count:
        target.errors.append(f"agentic_training_loop_plan.artifact_count expected {artifact_count}, got {plan.get('artifact_count')!r}.")
    expected_role_counts = [
        {"role": role, "count": len(refs)}
        for role, refs in sorted(source_artifacts.items())
        if isinstance(refs, list) and refs
    ]
    if plan.get("artifact_role_counts") != expected_role_counts:
        target.errors.append("agentic_training_loop_plan.artifact_role_counts must match source_artifacts exactly.")
    resolved_artifact_paths = _agentic_training_loop_resolved_artifact_paths(source_artifacts, source_path)
    cloud_training_receipt_state = _expected_agentic_training_loop_cloud_training_receipt_state(
        source_artifacts,
        source_path,
    )
    _validate_agentic_training_loop_cloud_training(
        plan.get("cloud_training"),
        source_artifacts,
        cloud_training_receipt_state,
        target,
        "agentic_training_loop_plan.cloud_training",
        source_path,
    )
    _validate_agentic_training_loop_cloud_training_receipt_state(
        plan.get("cloud_training_receipt_state"),
        cloud_training_receipt_state,
        target,
        "agentic_training_loop_plan.cloud_training_receipt_state",
    )
    cloud_training_completion_state = _loop_cloud_training_completion_state(
        resolved_artifact_paths,
        source_artifacts,
        loop_candidate_model_id=(
            plan.get("participants", {}).get("candidate_policy")
            if isinstance(plan.get("participants"), dict)
            else ""
        ),
    )
    _validate_agentic_training_loop_cloud_training_completion_state(
        plan.get("cloud_training_completion_state"),
        cloud_training_completion_state,
        target,
        "agentic_training_loop_plan.cloud_training_completion_state",
    )
    _validate_agentic_training_loop_cloud_training_lineage(
        plan.get("cloud_training_lineage"),
        source_artifacts,
        target,
        "agentic_training_loop_plan.cloud_training_lineage",
        source_path,
    )
    external_eval_receipt_state = _expected_agentic_training_loop_external_eval_receipt_state(
        source_artifacts,
        source_path,
    )
    _validate_agentic_training_loop_external_eval_receipt_state(
        plan.get("external_eval_receipt_state"),
        external_eval_receipt_state,
        target,
        "agentic_training_loop_plan.external_eval_receipt_state",
    )
    training_result_status = _loop_execution_result_status(resolved_artifact_paths, "agentic_training_result")
    cloud_training_completion_status = _loop_cloud_training_completion_status(
        cloud_training_completion_state
    )
    external_eval_result_status = _loop_execution_result_status(resolved_artifact_paths, "external_eval_result")
    execution_result_statuses = {
        "agentic_training_result": training_result_status,
        "cloud_training_completion_receipt": cloud_training_completion_status,
        "external_eval_result": external_eval_result_status,
    }
    expected_execution_completion = _loop_execution_completion(execution_result_statuses)
    eval_summary_state = _agentic_training_loop_source_validation_state(
        source_artifacts,
        "eval_summary",
        source_path,
        validate_eval_summary,
        require_passed=True,
    )
    promotion_decision_state = _agentic_training_loop_source_validation_state(
        source_artifacts,
        "promotion_decision",
        source_path,
        validate_promotion_decision,
        require_passed=True,
    )
    promotion_ledger_state = _agentic_training_loop_source_validation_state(
        source_artifacts,
        "promotion_ledger",
        source_path,
        validate_promotion_ledger,
    )
    check = _cloud_training_check_by_id(
        checks,
        "cloud_training_receipts_are_side_effect_free",
        target,
        "agentic_training_loop_plan.checks",
    )
    if check is not None and check.get("passed") != cloud_training_receipt_state["fail_closed"]:
        target.errors.append(
            "agentic_training_loop_plan.checks.cloud_training_receipts_are_side_effect_free.passed must match receipt state."
        )
    training_execution_check = _cloud_training_check_by_id(
        checks,
        "external_trainer_execution_completed",
        target,
        "agentic_training_loop_plan.checks",
    )
    training_result_plan_bound = _loop_agentic_training_result_plan_bound(
        resolved_artifact_paths,
        source_artifacts,
    )
    expected_training_execution_passed = (
        training_result_status == "completed"
        and training_result_plan_bound
        and cloud_training_completion_status == "completed"
    )
    if training_execution_check is not None and training_execution_check.get("passed") != expected_training_execution_passed:
        target.errors.append(
            "agentic_training_loop_plan.checks.external_trainer_execution_completed.passed must match the training-result and cloud-completion state."
        )
    completion_check = _cloud_training_check_by_id(
        checks,
        "external_cloud_training_completion_imported",
        target,
        "agentic_training_loop_plan.checks",
    )
    if (
        completion_check is not None
        and completion_check.get("passed")
        != cloud_training_completion_state["successful"]
    ):
        target.errors.append(
            "agentic_training_loop_plan.checks.external_cloud_training_completion_imported.passed must match imported completion state."
        )
    external_eval_handoff_check = _cloud_training_check_by_id(
        checks,
        "external_eval_handoff_is_preflighted",
        target,
        "agentic_training_loop_plan.checks",
    )
    expected_external_eval_handoff_passed = (
        _agentic_training_loop_role_source_ready(source_artifacts, "heldout_manifest", source_path)
        and _agentic_training_loop_role_source_ready(source_artifacts, "external_eval_plan", source_path)
        and _agentic_training_loop_role_source_ready(source_artifacts, "external_eval_receipt", source_path)
        and external_eval_receipt_state["receipts_passed"]
        and external_eval_receipt_state["fail_closed"]
    )
    if (
        external_eval_handoff_check is not None
        and external_eval_handoff_check.get("passed") != expected_external_eval_handoff_passed
    ):
        target.errors.append(
            "agentic_training_loop_plan.checks.external_eval_handoff_is_preflighted.passed must match external eval preflight evidence."
        )
    heldout_eval_check = _cloud_training_check_by_id(
        checks,
        "heldout_eval_is_fail_closed",
        target,
        "agentic_training_loop_plan.checks",
    )
    expected_heldout_eval_passed = (
        _agentic_training_loop_role_source_ready(source_artifacts, "heldout_manifest", source_path)
        and _agentic_training_loop_role_source_ready(source_artifacts, "external_eval_plan", source_path)
        and _agentic_training_loop_role_source_ready(source_artifacts, "external_eval_receipt", source_path)
        and external_eval_result_status == "completed"
        and _agentic_training_loop_role_source_ready(source_artifacts, "eval_summary", source_path)
        and eval_summary_state["valid"]
        and eval_summary_state["passed"]
        and external_eval_receipt_state["receipts_passed"]
        and external_eval_receipt_state["fail_closed"]
    )
    if heldout_eval_check is not None and heldout_eval_check.get("passed") != expected_heldout_eval_passed:
        target.errors.append(
            "agentic_training_loop_plan.checks.heldout_eval_is_fail_closed.passed must match external eval receipt and eval summary state."
        )
    governance_check = _cloud_training_check_by_id(
        checks,
        "governance_required_for_promotion",
        target,
        "agentic_training_loop_plan.checks",
    )
    expected_governance_passed = (
        "promotion_decision" in source_artifacts
        and promotion_decision_state["valid"]
        and promotion_decision_state["passed"]
        and "promotion_ledger" in source_artifacts
        and promotion_ledger_state["valid"]
    )
    if governance_check is not None and governance_check.get("passed") != expected_governance_passed:
        target.errors.append(
            "agentic_training_loop_plan.checks.governance_required_for_promotion.passed must match promotion decision and ledger validation state."
        )

    phases = plan.get("phases")
    if not isinstance(phases, list) or not phases:
        target.errors.append("agentic_training_loop_plan.phases must be a non-empty list.")
        phase_missing_inputs: list[str] = []
    else:
        for index, phase in enumerate(phases):
            _validate_agentic_training_loop_plan_phase(phase, target, f"agentic_training_loop_plan.phases[{index}]")
        phase_missing_inputs = sorted(
            {
                item
                for phase in phases
                if isinstance(phase, dict) and isinstance(phase.get("missing_required_artifacts"), list)
                for item in phase["missing_required_artifacts"]
                if isinstance(item, str)
            }
        )
        expected_phases = [
            _build_agentic_training_loop_phase(spec, source_artifacts, execution_result_statuses)
            for spec in AGENTIC_TRAINING_LOOP_PHASES
        ]
        if phases != expected_phases:
            target.errors.append(
                "agentic_training_loop_plan.phases must match canonical phase contracts and current source artifact completion."
            )
    if plan.get("missing_phase_inputs") != phase_missing_inputs:
        target.errors.append("agentic_training_loop_plan.missing_phase_inputs must match phase missing_required_artifacts.")
    failed_check_ids = {
        str(check.get("id") or "")
        for check in checks
        if isinstance(check, dict) and check.get("passed") is not True
    }
    missing_plan_inputs = sorted(
        role
        for role in PLAN_REQUIRED_ARTIFACT_ROLES
        if not _agentic_training_loop_role_source_ready(source_artifacts, role, source_path)
    )
    expected_plan_readiness = (
        "ready_to_execute"
        if check_contract_complete
        and not (failed_check_ids & PLAN_READINESS_CHECK_IDS)
        and not missing_plan_inputs
        else "blocked"
    )
    expected_governance_readiness = (
        "ready_for_review"
        if expected_plan_readiness == "ready_to_execute"
        and expected_execution_completion == "completed"
        and failed_checks == 0
        and not phase_missing_inputs
        else "blocked"
    )
    if plan.get("plan_readiness") != expected_plan_readiness:
        target.errors.append(
            f"agentic_training_loop_plan.plan_readiness expected {expected_plan_readiness!r}, got {plan.get('plan_readiness')!r}."
        )
    if plan.get("execution_completion") != expected_execution_completion:
        target.errors.append(
            "agentic_training_loop_plan.execution_completion must match semantic training and external eval result states."
        )
    if plan.get("governance_readiness") != expected_governance_readiness:
        target.errors.append(
            f"agentic_training_loop_plan.governance_readiness expected {expected_governance_readiness!r}, got {plan.get('governance_readiness')!r}."
        )
    expected_passed = expected_governance_readiness == "ready_for_review"
    if plan.get("passed") != expected_passed:
        target.errors.append("agentic_training_loop_plan.passed must match governance_readiness.")
    expected_readiness = "ready_for_governance_review" if expected_passed else "planned_fail_closed"
    if plan.get("readiness") != expected_readiness:
        target.errors.append(f"agentic_training_loop_plan.readiness expected {expected_readiness!r}, got {plan.get('readiness')!r}.")
    if expected_execution_completion == "failed":
        expected_recommendation = "investigate_failed_execution"
    elif expected_plan_readiness == "blocked":
        expected_recommendation = "collect_missing_plan_evidence"
    elif expected_execution_completion == "incomplete":
        expected_recommendation = "execute_ready_plan"
    elif expected_governance_readiness == "blocked":
        expected_recommendation = "resolve_governance_blockers"
    else:
        expected_recommendation = "submit_for_governance_review"
    if plan.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"agentic_training_loop_plan.recommendation expected {expected_recommendation!r}, got {plan.get('recommendation')!r}."
        )
    expected_blocked_reasons = [
        str(check.get("summary") or "")
        for check in checks
        if isinstance(check, dict) and check.get("passed") is not True
    ]
    if plan.get("blocked_reasons") != expected_blocked_reasons:
        target.errors.append("agentic_training_loop_plan.blocked_reasons must match failed check summaries.")

    boundary = plan.get("execution_boundary")
    if not isinstance(boundary, dict):
        target.errors.append("agentic_training_loop_plan.execution_boundary must be an object.")
    else:
        _validate_allowed_keys(boundary, _AGENTIC_TRAINING_LOOP_EXECUTION_BOUNDARY_KEYS, target, "agentic_training_loop_plan.execution_boundary")
        for field_name in ("dry_run_plan_only", "public_artifact_paths_redacted"):
            if boundary.get(field_name) is not True:
                target.errors.append(f"agentic_training_loop_plan.execution_boundary.{field_name} must be true.")
        for field_name in (
            "cloud_jobs_started",
            "paid_model_grader_calls_started",
            "live_benchmarks_started",
            "model_downloads_started",
            "weights_updated_by_flight_recorder",
            "credential_values_recorded",
        ):
            if boundary.get(field_name) is not False:
                target.errors.append(f"agentic_training_loop_plan.execution_boundary.{field_name} must be false.")

    contract = plan.get("handoff_contract")
    if not isinstance(contract, dict):
        target.errors.append("agentic_training_loop_plan.handoff_contract must be an object.")
    else:
        _validate_allowed_keys(contract, _AGENTIC_TRAINING_LOOP_HANDOFF_CONTRACT_KEYS, target, "agentic_training_loop_plan.handoff_contract")
        for field_name in (
            "flight_recorder_controls_preflight_and_receipts",
            "external_trainers_own_weight_updates",
            "live_launch_requires_explicit_opt_in",
            "requires_environment_credentials_for_live",
            "requires_trainer_preflight",
            "requires_trainer_launch_check",
            "requires_calibrated_review_before_training_data",
            "requires_heldout_eval_before_promotion",
            "requires_governance_decision_before_alias_update",
        ):
            if contract.get(field_name) is not True:
                target.errors.append(f"agentic_training_loop_plan.handoff_contract.{field_name} must be true.")
        if contract.get("default_live_execution_allowed") is not False:
            target.errors.append("agentic_training_loop_plan.handoff_contract.default_live_execution_allowed must be false.")

    participants = plan.get("participants") if isinstance(plan.get("participants"), dict) else {}
    budget = plan.get("budget") if isinstance(plan.get("budget"), dict) else {}
    provider_constraints = (
        plan.get("provider_constraints") if isinstance(plan.get("provider_constraints"), dict) else {}
    )
    next_iteration = plan.get("next_iteration") if isinstance(plan.get("next_iteration"), dict) else {}
    try:
        replayed_plan = _build_agentic_training_loop_plan(
            out_path=source_path,
            iteration_id=str(plan.get("iteration_id") or ""),
            artifact_paths=resolved_artifact_paths,
            objective=str(plan.get("objective") or ""),
            candidate=str(participants.get("candidate_policy") or ""),
            baseline=str(participants.get("baseline_policy") or ""),
            teacher=str(participants.get("teacher_policy") or ""),
            budget=budget,
            provider_constraints=provider_constraints,
            schedule=(next_iteration.get("schedule") if isinstance(next_iteration.get("schedule"), dict) else {}),
            preserve_paths=False,
            created_at=str(plan.get("created_at") or ""),
        )
    except (OSError, TypeError, ValueError) as exc:
        target.errors.append(f"agentic_training_loop_plan could not replay current source artifacts: {exc}")
    else:
        if plan != replayed_plan:
            target.errors.append(
                "agentic_training_loop_plan must match deterministic replay of its current source artifacts exactly."
            )

    target.details.update(
        {
            "iteration_id": plan.get("iteration_id"),
            "plan_readiness": plan.get("plan_readiness"),
            "execution_completion": plan.get("execution_completion"),
            "governance_readiness": plan.get("governance_readiness"),
            "readiness": plan.get("readiness"),
            "artifact_count": plan.get("artifact_count"),
            "phase_count": len(phases) if isinstance(phases, list) else 0,
        }
    )

def _validate_agentic_training_loop_plan_ref(ref: Any, target: ValidationTarget, label: str, source_path: Path) -> None:
    if not isinstance(ref, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(ref, _AGENTIC_TRAINING_LOOP_ARTIFACT_REF_KEYS, target, label)
    for field_name in ("role", "path", "kind"):
        if not isinstance(ref.get(field_name), str) or not ref.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    path_value = ref.get("path")
    if isinstance(path_value, str) and path_value and not _is_safe_agentic_training_result_path(path_value):
        target.errors.append(f"{label}.path must be a safe relative path without traversal.")
    if ref.get("kind") not in {"file", "directory"}:
        target.errors.append(f"{label}.kind must be 'file' or 'directory'.")
    if not isinstance(ref.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    if ref.get("exists") is False:
        for field_name in ("sha256", "size_bytes", "file_count", "contains_symlinks"):
            if ref.get(field_name) is not None:
                target.errors.append(f"{label}.{field_name} must be null when the source artifact is missing.")
        return
    if ref.get("kind") == "file" and ref.get("exists") is True:
        if not isinstance(ref.get("sha256"), str) or len(ref.get("sha256", "")) != 64:
            target.errors.append(f"{label}.sha256 must be a SHA-256 string for existing files.")
        if not isinstance(ref.get("size_bytes"), int) or ref.get("size_bytes") < 0:
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for existing files.")
        if ref.get("file_count") is not None:
            target.errors.append(f"{label}.file_count must be null for file artifacts.")
        if ref.get("contains_symlinks") is not None:
            target.errors.append(f"{label}.contains_symlinks must be null for file artifacts.")
        file_path = _resolve_agentic_training_loop_plan_ref_path(path_value, source_path)
        if file_path is not None and _path_has_symlink_component(file_path, include_leaf=True):
            target.errors.append(f"{label}.path must resolve to a regular non-symlink file.")
            return
        if file_path is None or not file_path.exists() or not file_path.is_file():
            target.errors.append(f"{label}.path does not resolve to an existing file.")
            return
        if isinstance(ref.get("size_bytes"), int) and ref.get("size_bytes") >= 0 and file_path.stat().st_size != ref.get("size_bytes"):
            target.errors.append(f"{label}.size_bytes does not match the current file.")
        if isinstance(ref.get("sha256"), str) and len(ref.get("sha256", "")) == 64 and _sha256(file_path) != ref.get("sha256"):
            target.errors.append(f"{label}.sha256 does not match the current file.")
    if ref.get("kind") == "directory" and ref.get("exists") is True:
        if not isinstance(ref.get("sha256"), str) or len(ref.get("sha256", "")) != 64:
            target.errors.append(f"{label}.sha256 must be a SHA-256 string for existing directories.")
        if not isinstance(ref.get("size_bytes"), int) or ref.get("size_bytes") < 0:
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for existing directories.")
        if not isinstance(ref.get("file_count"), int) or ref.get("file_count") < 0:
            target.errors.append(f"{label}.file_count must be a non-negative integer for existing directories.")
        if ref.get("contains_symlinks") is not False:
            target.errors.append(f"{label}.contains_symlinks must be false for existing directories.")
        directory_path = _resolve_agentic_training_loop_plan_ref_path(path_value, source_path)
        if directory_path is not None and _path_has_symlink_component(directory_path, include_leaf=True):
            target.errors.append(f"{label}.path must resolve to a regular non-symlink directory.")
            return
        if directory_path is None or not directory_path.exists() or not directory_path.is_dir():
            target.errors.append(f"{label}.path does not resolve to an existing directory.")
            return
        if _directory_contains_symlink(directory_path):
            target.errors.append(f"{label}.path must not contain symlink descendants.")
            return
        tree = _directory_tree_fingerprint(directory_path)
        if isinstance(ref.get("file_count"), int) and ref.get("file_count") >= 0 and tree["file_count"] != ref.get("file_count"):
            target.errors.append(f"{label}.file_count does not match the current directory.")
        if isinstance(ref.get("size_bytes"), int) and ref.get("size_bytes") >= 0 and tree["size_bytes"] != ref.get("size_bytes"):
            target.errors.append(f"{label}.size_bytes does not match the current directory.")
        if isinstance(ref.get("sha256"), str) and len(ref.get("sha256", "")) == 64 and tree["sha256"] != ref.get("sha256"):
            target.errors.append(f"{label}.sha256 does not match the current directory.")

    if ref.get("exists") is True:
        role = ref.get("role")
        artifact_path = _resolve_agentic_training_loop_plan_ref_path(path_value, source_path)
        if isinstance(role, str) and role and artifact_path is not None:
            inspection = inspect_artifact_source(artifact_path, role)
            if inspection.get("ready") is not True:
                target.errors.append(
                    f"{label}.exists cannot be true because the referenced {role} artifact is not semantically ready."
                )

def _agentic_training_loop_role_source_ready(
    source_artifacts: dict[str, Any],
    role: str,
    source_path: Path,
) -> bool:
    refs = source_artifacts.get(role)
    if not isinstance(refs, list) or not refs:
        return False
    for ref in refs:
        if not isinstance(ref, dict) or ref.get("exists") is not True:
            return False
        ref_path = _resolve_agentic_training_loop_plan_ref_path(ref.get("path"), source_path)
        if ref_path is None or inspect_artifact_source(ref_path, role).get("ready") is not True:
            return False
    return True

def _agentic_training_loop_source_validation_state(
    source_artifacts: dict[str, Any],
    role: str,
    source_path: Path,
    validator: Any,
    *,
    require_passed: bool = False,
) -> dict[str, bool]:
    refs = source_artifacts.get(role)
    if not isinstance(refs, list) or not refs:
        return {"present": False, "valid": False, "passed": False}
    valid = True
    passed = True
    for ref in refs:
        if not isinstance(ref, dict):
            valid = False
            passed = False
            continue
        ref_path = _resolve_agentic_training_loop_plan_ref_path(ref.get("path"), source_path)
        if (
            ref_path is None
            or not ref_path.exists()
            or _path_has_symlink_component(ref_path, include_leaf=True)
            or not ref_path.is_file()
        ):
            valid = False
            passed = False
            continue
        source_target = validator(ref_path)
        source_payload = _read_json_object_silent(ref_path)
        if source_target.errors or source_target.warnings or _agentic_training_loop_payload_has_public_unsafe_path(source_payload):
            valid = False
        if require_passed and source_target.details.get("passed") is not True:
            passed = False
    return {"present": True, "valid": valid, "passed": passed}

def _agentic_training_loop_payload_has_public_unsafe_path(value: Any, field_name: str = "") -> bool:
    if isinstance(value, str):
        if not _agentic_training_loop_public_path_field(field_name):
            return False
        raw_path = Path(value)
        windows_path = PureWindowsPath(value)
        return raw_path.is_absolute() or windows_path.is_absolute() or "~" in raw_path.parts or "~" in windows_path.parts
    if isinstance(value, dict):
        return any(
            _agentic_training_loop_payload_has_public_unsafe_path(item, str(key))
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_agentic_training_loop_payload_has_public_unsafe_path(item, field_name) for item in value)
    return False

def _agentic_training_loop_public_path_field(field_name: str) -> bool:
    normalized = field_name.lower()
    return normalized in {"path", "paths", "file", "files", "dir", "dirs"} or normalized.endswith(
        (
            "_path",
            "_paths",
            "_file",
            "_files",
            "_dir",
            "_dirs",
            "_url",
            "_urls",
        )
    )

def _resolve_agentic_training_loop_plan_ref_path(value: Any, source_path: Path) -> Path | None:
    if not isinstance(value, str) or not _is_safe_agentic_training_result_path(value):
        return None
    return source_path.parent / value

def _agentic_training_loop_resolved_artifact_paths(
    source_artifacts: dict[str, Any],
    source_path: Path,
) -> dict[str, list[Path]]:
    resolved: dict[str, list[Path]] = {}
    for role, refs in source_artifacts.items():
        if not isinstance(role, str) or not isinstance(refs, list):
            continue
        paths = [
            path
            for ref in refs
            if isinstance(ref, dict)
            for path in [_resolve_agentic_training_loop_plan_ref_path(ref.get("path"), source_path)]
            if path is not None
        ]
        if paths:
            resolved[role] = paths
    return resolved

def _validate_agentic_training_loop_plan_phase(phase: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(phase, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(phase, _AGENTIC_TRAINING_LOOP_PHASE_KEYS, target, label)
    for field_name in ("id", "name", "gate"):
        if not isinstance(phase.get(field_name), str) or not phase.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if phase.get("status") not in {"planned", "blocked", "ready"}:
        target.errors.append(f"{label}.status has an unsupported value.")
    required = phase.get("required_artifacts")
    present = phase.get("present_required_artifacts")
    missing = phase.get("missing_required_artifacts")
    non_completed = phase.get("non_completed_required_artifacts")
    produces = phase.get("produces")
    if not _is_string_list(required):
        target.errors.append(f"{label}.required_artifacts must be a list of strings.")
        required = []
    if not _is_string_list(present):
        target.errors.append(f"{label}.present_required_artifacts must be a list of strings.")
        present = []
    if not _is_string_list(missing):
        target.errors.append(f"{label}.missing_required_artifacts must be a list of strings.")
        missing = []
    if not _is_string_list(non_completed):
        target.errors.append(f"{label}.non_completed_required_artifacts must be a list of strings.")
        non_completed = []
    if not _is_string_list(produces):
        target.errors.append(f"{label}.produces must be a list of strings.")
    if sorted(set(present) | set(missing)) != sorted(set(required)):
        target.errors.append(f"{label}.present_required_artifacts plus missing_required_artifacts must match required_artifacts.")
    if not set(non_completed).issubset(set(present)):
        target.errors.append(f"{label}.non_completed_required_artifacts must be a subset of present_required_artifacts.")
    if phase.get("status") == "ready" and (missing or non_completed):
        target.errors.append(f"{label}.status cannot be ready while required artifacts are missing or incomplete.")
    if phase.get("status") == "blocked" and not missing and not non_completed:
        target.errors.append(f"{label}.status cannot be blocked without missing or incomplete required artifacts.")

def _validate_agentic_training_loop_cloud_training(
    value: Any,
    source_artifacts: dict[str, Any],
    receipt_state: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path,
) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_LOOP_CLOUD_TRAINING_KEYS, target, label)
    present = [
        role
        for role in AGENTIC_LOOP_CLOUD_TRAINING_HANDOFF_ROLES
        if _agentic_training_loop_role_source_ready(source_artifacts, role, source_path)
    ]
    missing = [role for role in AGENTIC_LOOP_CLOUD_TRAINING_HANDOFF_ROLES if role not in present]
    expected = {
        "required_artifacts": list(AGENTIC_LOOP_CLOUD_TRAINING_HANDOFF_ROLES),
        "present_artifacts": present,
        "missing_artifacts": missing,
        "artifact_count": sum(_agentic_loop_role_count(source_artifacts, role) for role in AGENTIC_LOOP_CLOUD_TRAINING_HANDOFF_ROLES),
        "provider_registry_present": "cloud_training_provider_registry" in present,
        "preflight_present": "cloud_training_preflight" in present,
        "artifact_manifest_present": "cloud_training_artifact_manifest" in present,
        "launch_plan_present": "cloud_training_launch_plan" in present,
        "launch_receipt_present": "cloud_training_launch_receipt" in present,
        "status_receipt_present": "cloud_training_status_receipt" in present,
        "provider_api_calls_started": receipt_state["provider_api_calls_started"],
        "cloud_jobs_started": receipt_state["cloud_jobs_started"],
        "credential_values_recorded": receipt_state["credential_values_recorded"],
        "live_spend_allowed": False,
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} must match cloud training source artifacts.")

def _validate_agentic_training_loop_cloud_training_receipt_state(
    value: Any,
    expected: dict[str, Any],
    target: ValidationTarget,
    label: str,
) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_LOOP_CLOUD_RECEIPT_STATE_KEYS, target, label)
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} must match cloud training receipt artifacts.")

def _validate_agentic_training_loop_cloud_training_completion_state(
    value: Any,
    expected: dict[str, Any],
    target: ValidationTarget,
    label: str,
) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(
        value,
        _AGENTIC_TRAINING_LOOP_CLOUD_COMPLETION_STATE_KEYS,
        target,
        label,
    )
    if value != expected:
        target.errors.append(
            f"{label} must match replayed cloud completion evidence exactly."
        )

def _validate_agentic_training_loop_external_eval_receipt_state(
    value: Any,
    expected: dict[str, Any],
    target: ValidationTarget,
    label: str,
) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_LOOP_EXTERNAL_EVAL_RECEIPT_STATE_KEYS, target, label)
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} must match external eval receipt artifacts.")

def _validate_agentic_training_loop_cloud_training_lineage(
    value: Any,
    source_artifacts: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path,
) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_LOOP_CLOUD_LINEAGE_KEYS, target, label)
    _validate_agentic_training_loop_object_keys(
        value.get("provider"),
        _AGENTIC_TRAINING_LOOP_CLOUD_LINEAGE_PROVIDER_KEYS,
        target,
        f"{label}.provider",
    )
    _validate_agentic_training_loop_role_count_rows(value.get("role_counts"), target, f"{label}.role_counts")
    expected = _expected_agentic_training_loop_cloud_training_lineage(source_artifacts, source_path)
    for field_name in (
        "passed",
        "required_link_count",
        "matched_link_count",
        "missing_link_count",
        "mismatched_link_count",
        "ambiguous_link_count",
        "duplicate_role_count",
        "missing_links",
        "mismatched_links",
        "ambiguous_links",
        "duplicate_roles",
        "role_counts",
    ):
        if value.get(field_name) != expected[field_name]:
            target.errors.append(f"{label}.{field_name} must match cloud training source lineage.")
    if value.get("provider") != expected["provider"]:
        target.errors.append(f"{label}.provider must match cloud training provider lineage.")
    links = value.get("links")
    if not isinstance(links, list):
        target.errors.append(f"{label}.links must be a list.")
        links = []
    if len(links) != len(expected["links"]):
        target.errors.append(f"{label}.links must include every required cloud training source link.")
    for index, expected_link in enumerate(expected["links"]):
        if index >= len(links) or not isinstance(links[index], dict):
            target.errors.append(f"{label}.links[{index}] must be an object.")
            continue
        _validate_allowed_keys(links[index], _AGENTIC_TRAINING_LOOP_CLOUD_LINEAGE_LINK_KEYS, target, f"{label}.links[{index}]")
        for field_name, expected_value in expected_link.items():
            if links[index].get(field_name) != expected_value:
                target.errors.append(f"{label}.links[{index}].{field_name} must match cloud training source lineage.")

def _expected_agentic_training_loop_cloud_training_lineage(
    source_artifacts: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    provider = _expected_agentic_training_loop_cloud_training_provider_lineage(source_artifacts, source_path)
    link_specs = CLOUD_TRAINING_HANDOFF_LINEAGE_LINKS
    links = [
        _expected_agentic_training_loop_cloud_training_link(
            source_artifacts,
            source_path,
            spec,
        )
        for spec in link_specs
    ]
    missing_links = [link["id"] for link in links if link["status"].startswith("missing_")]
    mismatched_links = [link["id"] for link in links if link["status"] == "mismatched_sha256"]
    ambiguous_links = [link["id"] for link in links if link["status"].startswith("ambiguous_")]
    role_counts = _expected_agentic_training_loop_cloud_training_role_counts(source_artifacts)
    active_roles = (
        {"cloud_training_provider_registry"}
        | {spec["source_role"] for spec in link_specs}
        | {spec["target_role"] for spec in link_specs}
    )
    duplicate_roles = [
        row["role"]
        for row in role_counts
        if row["role"] in active_roles and row["count"] > 1
    ]
    matched_link_count = sum(1 for link in links if link["passed"])
    passed = (
        provider["provider_consistent"]
        and provider["registry_contains_pipeline_provider"]
        and not missing_links
        and not mismatched_links
        and not ambiguous_links
        and not duplicate_roles
    )
    return {
        "passed": passed,
        "required_link_count": len(links),
        "matched_link_count": matched_link_count,
        "missing_link_count": len(missing_links),
        "mismatched_link_count": len(mismatched_links),
        "ambiguous_link_count": len(ambiguous_links),
        "duplicate_role_count": len(duplicate_roles),
        "missing_links": missing_links,
        "mismatched_links": mismatched_links,
        "ambiguous_links": ambiguous_links,
        "duplicate_roles": duplicate_roles,
        "role_counts": role_counts,
        "provider": provider,
        "links": links,
    }

def _expected_agentic_training_loop_cloud_training_receipt_state(
    source_artifacts: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    launch_records = _agentic_training_loop_payload_records(source_artifacts, "cloud_training_launch_receipt", source_path)
    status_records = _agentic_training_loop_payload_records(source_artifacts, "cloud_training_status_receipt", source_path)
    launch_payloads = [record["payload"] for record in launch_records]
    status_payloads = [record["payload"] for record in status_records]
    all_payloads = [*launch_payloads, *status_payloads]
    first_launch_payload = launch_payloads[0] if launch_payloads else {}
    first_status_payload = status_payloads[0] if status_payloads else {}
    first_launch = _agentic_training_loop_cloud_training_launch(first_launch_payload)
    first_status = _agentic_training_loop_cloud_training_status(first_status_payload)
    provider_api_calls_started = any(
        _agentic_training_loop_cloud_training_launch(payload).get("provider_api_called") is True for payload in launch_payloads
    ) or any(
        _agentic_training_loop_cloud_training_status(payload).get("provider_api_called") is True for payload in status_payloads
    ) or any(
        _agentic_training_loop_cloud_training_boundary(payload).get("provider_api_called") is True
        for payload in all_payloads
    )
    cloud_jobs_started = any(
        _agentic_training_loop_cloud_training_launch(payload).get("cloud_job_started") is True for payload in launch_payloads
    ) or any(
        _agentic_training_loop_cloud_training_boundary(payload).get("cloud_job_started") is True
        for payload in all_payloads
    )
    provider_cancel_called = any(
        _agentic_training_loop_cloud_training_status(payload).get("provider_cancel_called") is True for payload in status_payloads
    )
    credential_values_recorded = any(
        _agentic_training_loop_cloud_training_boundary(payload).get("credential_values_recorded") is True
        for payload in all_payloads
    )
    cost_incurred_usd = sum(
        _safe_non_negative_number(_agentic_training_loop_cloud_training_launch(payload).get("cost_incurred_usd"))
        for payload in launch_payloads
    ) + sum(
        _safe_non_negative_number(_agentic_training_loop_cloud_training_status(payload).get("cost_incurred_usd"))
        for payload in status_payloads
    ) + sum(
        _safe_non_negative_number(_agentic_training_loop_cloud_training_boundary(payload).get("cloud_cost_incurred_usd"))
        for payload in all_payloads
    )
    launch_mode = str(first_launch.get("mode") or "")
    status_provider_status = str(first_status.get("provider_status") or "")
    live_launch_requested = any(
        _agentic_training_loop_cloud_training_launch(payload).get("mode") == "live"
        or _agentic_training_loop_cloud_training_boundary(payload).get("live_requested") is True
        for payload in launch_payloads
    ) or any(
        _agentic_training_loop_cloud_training_boundary(payload).get("live_requested") is True for payload in status_payloads
    )
    fail_closed = (
        provider_api_calls_started is False
        and cloud_jobs_started is False
        and provider_cancel_called is False
        and credential_values_recorded is False
        and live_launch_requested is False
        and cost_incurred_usd == 0
    )
    launch_receipt_passed = bool(launch_records) and all(
        record["payload"].get("passed") is True
        and _loop_cloud_training_launch_receipt_semantic_passed(record["path"], record["payload"])
        for record in launch_records
    )
    status_receipt_passed = bool(status_records) and all(
        record["payload"].get("passed") is True
        and _loop_cloud_training_status_receipt_semantic_passed(record["path"], record["payload"])
        for record in status_records
    )
    return {
        "launch_receipt_count": len(launch_payloads),
        "status_receipt_count": len(status_payloads),
        "launch_receipt_passed": launch_receipt_passed,
        "status_receipt_passed": status_receipt_passed,
        "receipts_passed": launch_receipt_passed and status_receipt_passed,
        "launch_mode": launch_mode,
        "launch_readiness": str(first_launch_payload.get("readiness") or ""),
        "launch_recommendation": str(first_launch_payload.get("recommendation") or ""),
        "live_launch_requested": live_launch_requested,
        "status_provider_status": status_provider_status,
        "status_terminal": bool(status_payloads)
        and all(_agentic_training_loop_cloud_training_status(payload).get("terminal") is True for payload in status_payloads),
        "status_not_started": status_provider_status == "not_started",
        "status_readiness": str(first_status_payload.get("readiness") or ""),
        "status_recommendation": str(first_status_payload.get("recommendation") or ""),
        "provider_api_calls_started": provider_api_calls_started,
        "cloud_jobs_started": cloud_jobs_started,
        "provider_cancel_called": provider_cancel_called,
        "credential_values_recorded": credential_values_recorded,
        "cost_incurred_usd": cost_incurred_usd,
        "fail_closed": fail_closed,
    }

def _expected_agentic_training_loop_external_eval_receipt_state(
    source_artifacts: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    records = _agentic_training_loop_payload_records(source_artifacts, "external_eval_receipt", source_path)
    payloads = [record["payload"] for record in records]
    first_payload = payloads[0] if payloads else {}
    first_launch = _agentic_training_loop_external_eval_launch(first_payload)
    adapter_rows = [row for payload in payloads for row in _agentic_training_loop_external_eval_adapter_receipts(payload)]
    adapter_contracts = [_agentic_training_loop_external_eval_adapter_contract(row) for row in adapter_rows]
    live_benchmark_requested = any(
        _agentic_training_loop_external_eval_launch(payload).get("mode") == "live" for payload in payloads
    )
    live_benchmarks_started = any(
        _agentic_training_loop_external_eval_launch(payload).get("live_benchmarks_started") is True
        or _agentic_training_loop_external_eval_boundary(payload).get("live_benchmarks_started") is True
        for payload in payloads
    ) or any(row.get("live_benchmark_started") is True for row in adapter_rows)
    provider_api_calls_started = any(
        _agentic_training_loop_external_eval_launch(payload).get("provider_api_called") is True
        or _agentic_training_loop_external_eval_boundary(payload).get("provider_api_called") is True
        for payload in payloads
    ) or any(row.get("provider_api_called") is True for row in adapter_rows) or any(
        contract.get("provider_api_called_by_flight_recorder") is True for contract in adapter_contracts
    )
    model_downloads_started = any(
        _agentic_training_loop_external_eval_launch(payload).get("model_downloads_started") is True
        or _agentic_training_loop_external_eval_boundary(payload).get("model_downloads_started") is True
        for payload in payloads
    ) or any(row.get("model_downloads_started") is True for row in adapter_rows) or any(
        contract.get("model_downloads_started_by_flight_recorder") is True for contract in adapter_contracts
    )
    credential_values_recorded = any(
        _agentic_training_loop_external_eval_boundary(payload).get("credential_values_recorded") is True
        for payload in payloads
    ) or any(row.get("credential_values_recorded") is True for row in adapter_rows) or any(
        contract.get("credential_values_recorded") is True for contract in adapter_contracts
    )
    weights_updated_by_flight_recorder = any(
        _agentic_training_loop_external_eval_boundary(payload).get("weights_updated_by_flight_recorder") is True
        for payload in payloads
    )
    cost_incurred_usd = (
        sum(_safe_non_negative_number(_agentic_training_loop_external_eval_launch(payload).get("cost_incurred_usd")) for payload in payloads)
        + sum(
            _safe_non_negative_number(_agentic_training_loop_external_eval_boundary(payload).get("cloud_cost_incurred_usd"))
            for payload in payloads
        )
        + sum(_safe_non_negative_number(row.get("cost_incurred_usd")) for row in adapter_rows)
        + sum(_safe_non_negative_number(contract.get("cost_incurred_usd")) for contract in adapter_contracts)
    )
    dry_run_only = bool(payloads) and all(
        _agentic_training_loop_external_eval_boundary(payload).get("dry_run_only") is not False for payload in payloads
    )
    receipt_passed_count = sum(
        1
        for record in records
        if record["payload"].get("passed") is True
        and _loop_external_eval_receipt_semantic_passed(record["path"], record["payload"])
    )
    fail_closed = (
        live_benchmark_requested is False
        and live_benchmarks_started is False
        and provider_api_calls_started is False
        and model_downloads_started is False
        and credential_values_recorded is False
        and weights_updated_by_flight_recorder is False
        and cost_incurred_usd == 0
        and (not payloads or dry_run_only)
    )
    return {
        "receipt_count": len(payloads),
        "receipt_passed_count": receipt_passed_count,
        "receipts_passed": bool(payloads) and receipt_passed_count == len(payloads),
        "launch_mode": str(first_launch.get("mode") or ""),
        "readiness": str(first_payload.get("readiness") or ""),
        "recommendation": str(first_payload.get("recommendation") or ""),
        "adapter_count": sum(_safe_non_negative_int(payload.get("adapter_count")) for payload in payloads),
        "ready_adapter_count": sum(_safe_non_negative_int(payload.get("ready_adapter_count")) for payload in payloads),
        "dry_run_only": dry_run_only,
        "live_benchmark_requested": live_benchmark_requested,
        "live_benchmarks_started": live_benchmarks_started,
        "provider_api_calls_started": provider_api_calls_started,
        "model_downloads_started": model_downloads_started,
        "credential_values_recorded": credential_values_recorded,
        "weights_updated_by_flight_recorder": weights_updated_by_flight_recorder,
        "cost_incurred_usd": cost_incurred_usd,
        "fail_closed": fail_closed,
    }

def _agentic_training_loop_cloud_training_launch(payload: dict[str, Any]) -> dict[str, Any]:
    launch = payload.get("launch")
    return launch if isinstance(launch, dict) else {}

def _agentic_training_loop_cloud_training_status(payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get("status")
    return status if isinstance(status, dict) else {}

def _agentic_training_loop_cloud_training_boundary(payload: dict[str, Any]) -> dict[str, Any]:
    boundary = payload.get("execution_boundary")
    return boundary if isinstance(boundary, dict) else {}

def _agentic_training_loop_external_eval_launch(payload: dict[str, Any]) -> dict[str, Any]:
    launch = payload.get("launch")
    return launch if isinstance(launch, dict) else {}

def _agentic_training_loop_external_eval_boundary(payload: dict[str, Any]) -> dict[str, Any]:
    boundary = payload.get("execution_boundary")
    return boundary if isinstance(boundary, dict) else {}

def _agentic_training_loop_external_eval_adapter_receipts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("adapter_receipts")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

def _agentic_training_loop_external_eval_adapter_contract(row: dict[str, Any]) -> dict[str, Any]:
    contract = row.get("adapter_contract")
    return contract if isinstance(contract, dict) else {}

def _expected_agentic_training_loop_cloud_training_provider_lineage(
    source_artifacts: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    registry_provider_ids = sorted(
        {
            provider_id
            for payload in _agentic_training_loop_payloads(source_artifacts, "cloud_training_provider_registry", source_path)
            for provider_id in _agentic_training_loop_registry_provider_ids(payload)
        }
    )
    provider_by_role = {
        role: _agentic_training_loop_provider_id(_agentic_training_loop_first_payload(source_artifacts, role, source_path))
        for role in ("cloud_training_preflight", "cloud_training_artifact_manifest", "cloud_training_launch_plan")
    }
    pipeline_provider_ids = sorted({provider_id for provider_id in provider_by_role.values() if provider_id})
    pipeline_provider_id = pipeline_provider_ids[0] if len(pipeline_provider_ids) == 1 else ""
    return {
        "registry_provider_ids": registry_provider_ids,
        "pipeline_provider_ids": pipeline_provider_ids,
        "pipeline_provider_id": pipeline_provider_id,
        "provider_by_role": provider_by_role,
        "provider_consistent": len(pipeline_provider_ids) == 1,
        "registry_contains_pipeline_provider": bool(pipeline_provider_id) and pipeline_provider_id in registry_provider_ids,
    }

def _expected_agentic_training_loop_cloud_training_link(
    source_artifacts: dict[str, Any],
    source_path: Path,
    spec: dict[str, str],
) -> dict[str, Any]:
    source_role = spec["source_role"]
    target_role = spec["target_role"]
    source_ref_name = spec["source_ref"]
    source_count = _agentic_loop_role_count(source_artifacts, source_role)
    target_count = _agentic_loop_role_count(source_artifacts, target_role)
    source_ref = _agentic_training_loop_first_ref(source_artifacts, source_role)
    target_ref = _agentic_training_loop_first_ref(source_artifacts, target_role)
    source_payload = _agentic_training_loop_first_payload(source_artifacts, source_role, source_path)
    nested_field = (
        "sources"
        if source_role == "cloud_training_completion_receipt"
        else "source_artifacts"
    )
    nested_refs = (
        source_payload.get(nested_field)
        if isinstance(source_payload.get(nested_field), dict)
        else {}
    )
    nested_ref = nested_refs.get(source_ref_name) if isinstance(nested_refs, dict) else None
    nested_ref = nested_ref if isinstance(nested_ref, dict) else {}
    nested_sha = nested_ref.get("sha256") if isinstance(nested_ref.get("sha256"), str) else ""
    target_sha = target_ref.get("sha256") if isinstance(target_ref.get("sha256"), str) else ""
    status = "matched"
    if source_count > 1:
        status = "ambiguous_source_artifacts"
    elif target_count > 1:
        status = "ambiguous_target_artifacts"
    elif not source_ref:
        status = "missing_source_artifact"
    elif not target_ref:
        status = "missing_target_artifact"
    elif not nested_ref:
        status = "missing_source_link"
    elif not nested_sha:
        status = "missing_source_link_sha256"
    elif not target_sha:
        status = "missing_target_sha256"
    elif nested_sha != target_sha:
        status = "mismatched_sha256"
    return {
        "id": spec["id"],
        "source_role": source_role,
        "source_ref": source_ref_name,
        "target_role": target_role,
        "source_artifact_count": source_count,
        "target_artifact_count": target_count,
        "source_schema_version": source_ref.get("schema_version", "") if source_ref else "",
        "target_schema_version": target_ref.get("schema_version", "") if target_ref else "",
        "source_ref_sha256": nested_sha,
        "target_sha256": target_sha,
        "passed": status == "matched",
        "status": status,
    }

def _expected_agentic_training_loop_cloud_training_role_counts(source_artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": role, "count": _agentic_loop_role_count(source_artifacts, role)}
        for role in CLOUD_TRAINING_LINEAGE_ARTIFACT_ROLES
    ]

def _agentic_training_loop_first_ref(source_artifacts: dict[str, Any], role: str) -> dict[str, Any]:
    rows = source_artifacts.get(role)
    return rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else {}

def _agentic_training_loop_payload_records(source_artifacts: dict[str, Any], role: str, source_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rows = source_artifacts.get(role)
    if not isinstance(rows, list):
        return records
    for row in rows:
        if not isinstance(row, dict):
            continue
        path = _resolve_agentic_training_loop_plan_ref_path(row.get("path"), source_path)
        if (
            path is None
            or not path.exists()
            or _path_has_symlink_component(path, include_leaf=True)
            or not path.is_file()
        ):
            continue
        payload = _read_json_object_silent(path)
        if payload:
            records.append({"path": path, "payload": payload})
    return records

def _agentic_training_loop_payloads(source_artifacts: dict[str, Any], role: str, source_path: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for record in _agentic_training_loop_payload_records(source_artifacts, role, source_path):
        payload = record.get("payload")
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads

def _agentic_training_loop_first_payload(source_artifacts: dict[str, Any], role: str, source_path: Path) -> dict[str, Any]:
    payloads = _agentic_training_loop_payloads(source_artifacts, role, source_path)
    return payloads[0] if payloads else {}

def _agentic_training_loop_registry_provider_ids(payload: dict[str, Any]) -> list[str]:
    providers = payload.get("providers")
    if not isinstance(providers, list):
        return []
    return [str(provider.get("id")) for provider in providers if isinstance(provider, dict) and str(provider.get("id") or "")]

def _agentic_training_loop_provider_id(payload: dict[str, Any]) -> str:
    provider = payload.get("provider")
    if not isinstance(provider, dict):
        return ""
    return str(provider.get("id") or "")

_AGENTIC_LOOP_LEDGER_KEYS = {
    "schema_version",
    "ledger_path",
    "passed",
    "iteration_count",
    "iterations",
    "metrics",
    "decision",
    "readiness_digest",
    "execution_boundary",
    "notes",
}

_AGENTIC_LOOP_LEDGER_METRICS_KEYS = {
    "iteration_count",
    "ready_iteration_count",
    "blocked_iteration_count",
    "latest_iteration_id",
    "latest_plan_readiness",
    "latest_execution_completion",
    "latest_governance_readiness",
    "latest_readiness",
    "latest_recommendation",
    "latest_missing_phase_input_count",
    "scheduled_next_iteration_count",
    "promotion_ready_count",
    "rollback_receipt_count",
    "total_max_cloud_cost_usd",
    "total_max_gpu_hours",
    "readiness_counts",
    "recommendation_counts",
    "artifact_group_totals",
}

_AGENTIC_LOOP_LEDGER_DECISION_KEYS = {
    "readiness",
    "recommendation",
    "recommended_governance_action",
    "governance_action_count",
    "governance_actions",
    "summary",
    "latest_iteration_index",
    "latest_iteration_id",
    "blocked_iteration_count",
}

_AGENTIC_LOOP_LEDGER_GOVERNANCE_ACTION_KEYS = {
    "action",
    "available",
    "blocked_reason_count",
    "blocked_reasons",
    "summary",
}

_AGENTIC_LOOP_LEDGER_DIGEST_KEYS = {
    "latest_iteration_index",
    "latest_iteration_id",
    "plan_readiness",
    "execution_completion",
    "governance_readiness",
    "readiness",
    "recommendation",
    "decision_readiness",
    "decision_recommendation",
    "recommended_governance_action",
    "ready_for_governance_review",
    "missing_phase_input_count",
    "missing_phase_inputs",
    "missing_artifact_group_count",
    "missing_artifact_groups",
    "blocked_reason_count",
    "cloud_training_ambiguous_link_count",
    "cloud_training_duplicate_role_count",
    "cloud_training_lineage_bound",
    "cloud_training_receipts_fail_closed",
    "cloud_training_completion_successful",
    "cloud_training_completion_integrity_passed",
    "cloud_training_completion_execution_status",
    "cloud_training_completion_execution_terminal",
    "cloud_training_completion_governance_readiness",
    "cloud_training_completion_claims_allowed",
    "cloud_training_completion_provider_id",
    "cloud_training_completion_provider_job_id",
    "cloud_training_completion_execution_id",
    "cloud_training_completion_provider_matches_pipeline",
    "cloud_training_completion_candidate_model_id",
    "cloud_training_completion_loop_candidate_model_id",
    "cloud_training_completion_candidate_matches_loop",
    "cloud_training_completion_candidate_matches_training_result",
    "cloud_training_completion_source_bindings_complete",
    "cloud_training_completion_output_artifact_manifest_bound",
    "cloud_training_completion_output_artifact_set_bound",
    "cloud_training_completion_output_artifact_count",
    "cloud_training_live_launch_requested",
    "cloud_training_cost_incurred_usd",
    "cloud_training_launch_mode",
    "cloud_training_status_provider_status",
    "cloud_training_provider_id",
    "cloud_training_missing_link_count",
    "cloud_training_mismatched_link_count",
    "external_eval_receipt_count",
    "external_eval_adapter_count",
    "external_eval_ready_adapter_count",
    "external_eval_receipts_passed",
    "external_eval_receipts_fail_closed",
    "external_eval_live_benchmark_requested",
    "external_eval_live_benchmarks_started",
    "external_eval_provider_api_calls_started",
    "external_eval_model_downloads_started",
    "external_eval_credential_values_recorded",
    "external_eval_cost_incurred_usd",
    "external_eval_launch_mode",
    "next_action_scheduled",
    "next_action_recommendation",
    "requires_governance_decision",
    "promotion_decision_present",
    "promotion_ledger_present",
    "rollback_receipt_present",
    "live_spend_allowed",
    "side_effects_started",
    "summary",
}

_AGENTIC_LOOP_LEDGER_BOUNDARY_KEYS = {
    "ledger_only",
    "cloud_jobs_started",
    "paid_model_grader_calls_started",
    "live_benchmarks_started",
    "model_downloads_started",
    "weights_updated_by_flight_recorder",
    "credential_values_recorded",
}

_AGENTIC_LOOP_LEDGER_ITERATION_KEYS = {
    "index",
    "path",
    "exists",
    "schema_version",
    "iteration_id",
    "passed",
    "plan_readiness",
    "execution_completion",
    "governance_readiness",
    "readiness",
    "recommendation",
    "missing_phase_inputs",
    "blocked_reason_count",
    "artifact_count",
    "phase_status_counts",
    "artifact_group_counts",
    "artifact_role_counts",
    "cloud_training",
    "cloud_training_receipt_state",
    "cloud_training_completion_state",
    "cloud_training_lineage",
    "cost_estimate",
    "serving",
    "evals",
    "external_eval_receipt_state",
    "training_outputs",
    "governance",
    "next_actions",
    "size_bytes",
    "sha256",
}

_AGENTIC_LOOP_LEDGER_COST_KEYS = {"max_cloud_cost_usd", "max_gpu_hours", "live_spend_allowed"}

_AGENTIC_LOOP_LEDGER_BASIC_GROUP_KEYS = {"group", "artifact_count", "roles_present", "roles_missing"}

_AGENTIC_LOOP_LEDGER_CLOUD_TRAINING_KEYS = _AGENTIC_LOOP_LEDGER_BASIC_GROUP_KEYS | {
    "provider_registry_present",
    "preflight_present",
    "artifact_manifest_present",
    "launch_plan_present",
    "launch_receipt_present",
    "status_receipt_present",
    "completion_receipt_present",
    "provider_api_calls_started",
    "cloud_jobs_started",
    "credential_values_recorded",
    "live_spend_allowed",
}

_AGENTIC_LOOP_LEDGER_GOVERNANCE_KEYS = _AGENTIC_LOOP_LEDGER_BASIC_GROUP_KEYS | {
    "promotion_decision_present",
    "promotion_ledger_present",
    "rollback_receipt_present",
    "cloud_jobs_started",
    "paid_model_grader_calls_started",
    "weights_updated_by_flight_recorder",
}

_AGENTIC_LOOP_LEDGER_NEXT_ACTION_KEYS = {
    "scheduled",
    "requires_governance_decision",
    "recommendation",
    "schedule",
}

def _validate_agentic_loop_ledger(ledger: dict[str, Any], target: ValidationTarget, ledger_path: Path) -> None:
    _validate_allowed_keys(ledger, _AGENTIC_LOOP_LEDGER_KEYS, target, "agentic_loop_ledger")
    _require_equal(ledger, "schema_version", AGENTIC_LOOP_LEDGER_SCHEMA_VERSION, target, prefix="agentic_loop_ledger.")
    if ledger.get("passed") is not True:
        target.errors.append("agentic_loop_ledger.passed must be true.")
    iterations = ledger.get("iterations")
    if not isinstance(iterations, list) or not iterations:
        target.errors.append("agentic_loop_ledger.iterations must be a non-empty list.")
        iterations = []
    if ledger.get("iteration_count") != len(iterations):
        target.errors.append(f"agentic_loop_ledger.iteration_count expected {len(iterations)}, got {ledger.get('iteration_count')!r}.")
    ready_count = 0
    blocked_count = 0
    for index, row in enumerate(iterations):
        label = f"agentic_loop_ledger.iterations[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(row, _AGENTIC_LOOP_LEDGER_ITERATION_KEYS, target, label)
        if row.get("index") != index:
            target.errors.append(f"{label}.index expected {index}, got {row.get('index')!r}.")
        if row.get("schema_version") != AGENTIC_TRAINING_LOOP_PLAN_SCHEMA_VERSION:
            target.errors.append(f"{label}.schema_version must be {AGENTIC_TRAINING_LOOP_PLAN_SCHEMA_VERSION!r}.")
        if not isinstance(row.get("iteration_id"), str) or not row.get("iteration_id"):
            target.errors.append(f"{label}.iteration_id must be a non-empty string.")
        if row.get("governance_readiness") == "ready_for_review":
            ready_count += 1
        else:
            blocked_count += 1
        if row.get("plan_readiness") not in {"ready_to_execute", "blocked"}:
            target.errors.append(f"{label}.plan_readiness has an unsupported value.")
        if row.get("execution_completion") not in {"completed", "incomplete", "failed"}:
            target.errors.append(f"{label}.execution_completion has an unsupported value.")
        if row.get("governance_readiness") not in {"ready_for_review", "blocked"}:
            target.errors.append(f"{label}.governance_readiness has an unsupported value.")
        if not _is_string_list(row.get("missing_phase_inputs")):
            target.errors.append(f"{label}.missing_phase_inputs must be a list of strings.")
        if not _is_non_negative_int(row.get("blocked_reason_count")):
            target.errors.append(f"{label}.blocked_reason_count must be a non-negative integer.")
        if not _is_non_negative_int(row.get("artifact_count")):
            target.errors.append(f"{label}.artifact_count must be a non-negative integer.")
        _validate_agentic_loop_ledger_counts(row.get("phase_status_counts"), target, f"{label}.phase_status_counts", "id")
        _validate_agentic_loop_ledger_counts(row.get("artifact_group_counts"), target, f"{label}.artifact_group_counts", "group")
        _validate_agentic_loop_ledger_counts(row.get("artifact_role_counts"), target, f"{label}.artifact_role_counts", "role")
        _validate_agentic_loop_ledger_source(row, target, ledger_path, label)
        _validate_agentic_loop_ledger_cost_estimate(row.get("cost_estimate"), target, f"{label}.cost_estimate")
        _validate_agentic_loop_ledger_basic_group(row.get("serving"), target, f"{label}.serving")
        _validate_agentic_loop_ledger_basic_group(row.get("evals"), target, f"{label}.evals")
        _validate_agentic_loop_ledger_basic_group(row.get("training_outputs"), target, f"{label}.training_outputs")
        governance = row.get("governance") if isinstance(row.get("governance"), dict) else {}
        if isinstance(row.get("governance"), dict):
            _validate_allowed_keys(governance, _AGENTIC_LOOP_LEDGER_GOVERNANCE_KEYS, target, f"{label}.governance")
        else:
            target.errors.append(f"{label}.governance must be an object.")
        for field_name in ("cloud_jobs_started", "paid_model_grader_calls_started", "weights_updated_by_flight_recorder"):
            if governance.get(field_name) is not False:
                target.errors.append(f"{label}.governance.{field_name} must be false.")
        _validate_agentic_loop_ledger_next_actions(row.get("next_actions"), target, f"{label}.next_actions")
        _validate_agentic_loop_ledger_cloud_training(row.get("cloud_training"), row, target, f"{label}.cloud_training")
        _validate_agentic_loop_ledger_cloud_training_receipt_state(
            row.get("cloud_training_receipt_state"),
            row,
            target,
            ledger_path,
            f"{label}.cloud_training_receipt_state",
        )
        _validate_agentic_loop_ledger_cloud_training_completion_state(
            row.get("cloud_training_completion_state"),
            row,
            target,
            ledger_path,
            f"{label}.cloud_training_completion_state",
        )
        _validate_agentic_loop_ledger_external_eval_receipt_state(
            row.get("external_eval_receipt_state"),
            row,
            target,
            ledger_path,
            f"{label}.external_eval_receipt_state",
        )
        _validate_agentic_loop_ledger_cloud_training_lineage(row.get("cloud_training_lineage"), row, target, ledger_path, label)
    metrics = ledger.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("agentic_loop_ledger.metrics must be an object.")
        metrics = {}
    else:
        _validate_allowed_keys(metrics, _AGENTIC_LOOP_LEDGER_METRICS_KEYS, target, "agentic_loop_ledger.metrics")
    if metrics.get("iteration_count") != len(iterations):
        target.errors.append("agentic_loop_ledger.metrics.iteration_count must match iteration_count.")
    if metrics.get("ready_iteration_count") != ready_count:
        target.errors.append("agentic_loop_ledger.metrics.ready_iteration_count does not match iterations.")
    if metrics.get("blocked_iteration_count") != blocked_count:
        target.errors.append("agentic_loop_ledger.metrics.blocked_iteration_count does not match iterations.")
    _validate_agentic_loop_ledger_counts(metrics.get("readiness_counts"), target, "agentic_loop_ledger.metrics.readiness_counts", "id")
    _validate_agentic_loop_ledger_counts(metrics.get("recommendation_counts"), target, "agentic_loop_ledger.metrics.recommendation_counts", "id")
    _validate_agentic_loop_ledger_counts(metrics.get("artifact_group_totals"), target, "agentic_loop_ledger.metrics.artifact_group_totals", "group")
    expected_metrics = _build_agentic_loop_ledger_metrics(iterations)
    if metrics != expected_metrics:
        target.errors.append("agentic_loop_ledger.metrics must match the current iteration projections exactly.")
    expected_decision = _build_agentic_loop_ledger_decision(iterations, expected_metrics)
    expected_digest = _build_agentic_loop_ledger_readiness_digest(iterations, expected_decision)
    _validate_agentic_loop_ledger_decision(ledger.get("decision"), expected_decision, target)
    _validate_agentic_loop_ledger_digest(ledger.get("readiness_digest"), iterations, metrics, expected_decision, expected_digest, target)
    boundary = ledger.get("execution_boundary")
    if not isinstance(boundary, dict):
        target.errors.append("agentic_loop_ledger.execution_boundary must be an object.")
    else:
        _validate_allowed_keys(boundary, _AGENTIC_LOOP_LEDGER_BOUNDARY_KEYS, target, "agentic_loop_ledger.execution_boundary")
        if boundary.get("ledger_only") is not True:
            target.errors.append("agentic_loop_ledger.execution_boundary.ledger_only must be true.")
        for field_name in (
            "cloud_jobs_started",
            "paid_model_grader_calls_started",
            "live_benchmarks_started",
            "model_downloads_started",
            "weights_updated_by_flight_recorder",
            "credential_values_recorded",
        ):
            if boundary.get(field_name) is not False:
                target.errors.append(f"agentic_loop_ledger.execution_boundary.{field_name} must be false.")
    if not _is_string_list(ledger.get("notes")):
        target.errors.append("agentic_loop_ledger.notes must be a list of strings.")
    target.details.update(
        {
            "iteration_count": len(iterations),
            "latest_iteration_id": metrics.get("latest_iteration_id"),
            "latest_readiness": metrics.get("latest_readiness"),
        }
    )

def _validate_agentic_loop_ledger_digest(
    digest: Any,
    iterations: list[Any],
    metrics: dict[str, Any],
    decision: dict[str, Any],
    expected_digest: dict[str, Any],
    target: ValidationTarget,
) -> None:
    if not isinstance(digest, dict):
        target.errors.append("agentic_loop_ledger.readiness_digest must be an object.")
        return
    _validate_allowed_keys(digest, _AGENTIC_LOOP_LEDGER_DIGEST_KEYS, target, "agentic_loop_ledger.readiness_digest")
    latest = iterations[-1] if iterations and isinstance(iterations[-1], dict) else {}
    missing_phase_inputs = digest.get("missing_phase_inputs")
    if not _is_string_list(missing_phase_inputs):
        target.errors.append("agentic_loop_ledger.readiness_digest.missing_phase_inputs must be a list of strings.")
        missing_phase_inputs = []
    if digest.get("missing_phase_input_count") != len(missing_phase_inputs):
        target.errors.append("agentic_loop_ledger.readiness_digest.missing_phase_input_count must match missing_phase_inputs.")
    missing_groups = digest.get("missing_artifact_groups")
    if not _is_string_list(missing_groups):
        target.errors.append("agentic_loop_ledger.readiness_digest.missing_artifact_groups must be a list of strings.")
        missing_groups = []
    if digest.get("missing_artifact_group_count") != len(missing_groups):
        target.errors.append("agentic_loop_ledger.readiness_digest.missing_artifact_group_count must match missing_artifact_groups.")
    if digest.get("side_effects_started") is not False:
        target.errors.append("agentic_loop_ledger.readiness_digest.side_effects_started must be false.")
    if latest:
        latest_missing_phase_inputs = (
            [item for item in latest.get("missing_phase_inputs", []) if isinstance(item, str)]
            if isinstance(latest.get("missing_phase_inputs"), list)
            else []
        )
        latest_group_counts = {
            row.get("group"): _safe_non_negative_int(row.get("count"))
            for row in latest.get("artifact_group_counts", [])
            if isinstance(row, dict) and isinstance(row.get("group"), str)
        }
        latest_missing_groups = sorted(group for group, count in latest_group_counts.items() if count == 0)
        if digest.get("latest_iteration_id") != latest.get("iteration_id"):
            target.errors.append("agentic_loop_ledger.readiness_digest.latest_iteration_id must match the latest iteration.")
        if digest.get("latest_iteration_index") != latest.get("index"):
            target.errors.append("agentic_loop_ledger.readiness_digest.latest_iteration_index must match the latest iteration.")
        if digest.get("readiness") != latest.get("readiness"):
            target.errors.append("agentic_loop_ledger.readiness_digest.readiness must match the latest iteration.")
        if digest.get("recommendation") != latest.get("recommendation"):
            target.errors.append("agentic_loop_ledger.readiness_digest.recommendation must match the latest iteration.")
        if digest.get("recommended_governance_action") != decision.get("recommended_governance_action"):
            target.errors.append("agentic_loop_ledger.readiness_digest.recommended_governance_action must match decision.")
        latest_governance = latest.get("governance") if isinstance(latest.get("governance"), dict) else {}
        if digest.get("promotion_decision_present") != (latest_governance.get("promotion_decision_present") is True):
            target.errors.append("agentic_loop_ledger.readiness_digest.promotion_decision_present must match the latest iteration.")
        if digest.get("promotion_ledger_present") != (latest_governance.get("promotion_ledger_present") is True):
            target.errors.append("agentic_loop_ledger.readiness_digest.promotion_ledger_present must match the latest iteration.")
        if digest.get("rollback_receipt_present") != (latest_governance.get("rollback_receipt_present") is True):
            target.errors.append("agentic_loop_ledger.readiness_digest.rollback_receipt_present must match the latest iteration.")
        if missing_phase_inputs != latest_missing_phase_inputs:
            target.errors.append("agentic_loop_ledger.readiness_digest.missing_phase_inputs must match the latest iteration.")
        if missing_groups != latest_missing_groups:
            target.errors.append("agentic_loop_ledger.readiness_digest.missing_artifact_groups must match the latest iteration.")
        latest_lineage = latest.get("cloud_training_lineage") if isinstance(latest.get("cloud_training_lineage"), dict) else {}
        latest_lineage_provider = latest_lineage.get("provider") if isinstance(latest_lineage.get("provider"), dict) else {}
        latest_receipt_state = (
            latest.get("cloud_training_receipt_state") if isinstance(latest.get("cloud_training_receipt_state"), dict) else {}
        )
        latest_completion_state = (
            latest.get("cloud_training_completion_state")
            if isinstance(latest.get("cloud_training_completion_state"), dict)
            else {}
        )
        latest_external_eval_receipt_state = (
            latest.get("external_eval_receipt_state") if isinstance(latest.get("external_eval_receipt_state"), dict) else {}
        )
        expected_ready = (
            latest.get("plan_readiness") == "ready_to_execute"
            and latest.get("execution_completion") == "completed"
            and latest.get("governance_readiness") == "ready_for_review"
            and latest.get("readiness") == "ready_for_governance_review"
            and not latest_missing_phase_inputs
            and latest_lineage.get("passed") is True
            and latest_receipt_state.get("fail_closed") is True
            and latest_completion_state.get("successful") is True
            and latest_external_eval_receipt_state.get("fail_closed") is True
            and latest_external_eval_receipt_state.get("receipts_passed") is True
        )
        if digest.get("ready_for_governance_review") != expected_ready:
            target.errors.append("agentic_loop_ledger.readiness_digest.ready_for_governance_review must match latest iteration readiness.")
        if digest.get("cloud_training_lineage_bound") != (latest_lineage.get("passed") is True):
            target.errors.append("agentic_loop_ledger.readiness_digest.cloud_training_lineage_bound must match the latest iteration.")
        if digest.get("cloud_training_receipts_fail_closed") != (latest_receipt_state.get("fail_closed") is True):
            target.errors.append("agentic_loop_ledger.readiness_digest.cloud_training_receipts_fail_closed must match the latest iteration.")
        if digest.get("cloud_training_live_launch_requested") != (latest_receipt_state.get("live_launch_requested") is True):
            target.errors.append(
                "agentic_loop_ledger.readiness_digest.cloud_training_live_launch_requested must match the latest iteration."
            )
        if digest.get("cloud_training_cost_incurred_usd") != _safe_non_negative_number(latest_receipt_state.get("cost_incurred_usd")):
            target.errors.append("agentic_loop_ledger.readiness_digest.cloud_training_cost_incurred_usd must match the latest iteration.")
        if digest.get("cloud_training_launch_mode") != str(latest_receipt_state.get("launch_mode") or ""):
            target.errors.append("agentic_loop_ledger.readiness_digest.cloud_training_launch_mode must match the latest iteration.")
        if digest.get("cloud_training_status_provider_status") != str(latest_receipt_state.get("status_provider_status") or ""):
            target.errors.append(
                "agentic_loop_ledger.readiness_digest.cloud_training_status_provider_status must match the latest iteration."
            )
        if digest.get("cloud_training_provider_id") != str(latest_lineage_provider.get("pipeline_provider_id") or ""):
            target.errors.append("agentic_loop_ledger.readiness_digest.cloud_training_provider_id must match the latest iteration.")
        if digest.get("cloud_training_missing_link_count") != _safe_non_negative_int(latest_lineage.get("missing_link_count")):
            target.errors.append("agentic_loop_ledger.readiness_digest.cloud_training_missing_link_count must match the latest iteration.")
        if digest.get("cloud_training_mismatched_link_count") != _safe_non_negative_int(latest_lineage.get("mismatched_link_count")):
            target.errors.append("agentic_loop_ledger.readiness_digest.cloud_training_mismatched_link_count must match the latest iteration.")
        if digest.get("cloud_training_ambiguous_link_count") != _safe_non_negative_int(latest_lineage.get("ambiguous_link_count")):
            target.errors.append("agentic_loop_ledger.readiness_digest.cloud_training_ambiguous_link_count must match the latest iteration.")
        if digest.get("cloud_training_duplicate_role_count") != _safe_non_negative_int(latest_lineage.get("duplicate_role_count")):
            target.errors.append("agentic_loop_ledger.readiness_digest.cloud_training_duplicate_role_count must match the latest iteration.")
        if digest.get("external_eval_receipt_count") != _safe_non_negative_int(latest_external_eval_receipt_state.get("receipt_count")):
            target.errors.append("agentic_loop_ledger.readiness_digest.external_eval_receipt_count must match the latest iteration.")
        if digest.get("external_eval_adapter_count") != _safe_non_negative_int(latest_external_eval_receipt_state.get("adapter_count")):
            target.errors.append("agentic_loop_ledger.readiness_digest.external_eval_adapter_count must match the latest iteration.")
        if digest.get("external_eval_ready_adapter_count") != _safe_non_negative_int(
            latest_external_eval_receipt_state.get("ready_adapter_count")
        ):
            target.errors.append("agentic_loop_ledger.readiness_digest.external_eval_ready_adapter_count must match the latest iteration.")
        if digest.get("external_eval_receipts_passed") != (latest_external_eval_receipt_state.get("receipts_passed") is True):
            target.errors.append("agentic_loop_ledger.readiness_digest.external_eval_receipts_passed must match the latest iteration.")
        if digest.get("external_eval_receipts_fail_closed") != (latest_external_eval_receipt_state.get("fail_closed") is True):
            target.errors.append("agentic_loop_ledger.readiness_digest.external_eval_receipts_fail_closed must match the latest iteration.")
        if digest.get("external_eval_live_benchmark_requested") != (
            latest_external_eval_receipt_state.get("live_benchmark_requested") is True
        ):
            target.errors.append("agentic_loop_ledger.readiness_digest.external_eval_live_benchmark_requested must match the latest iteration.")
        if digest.get("external_eval_live_benchmarks_started") != (
            latest_external_eval_receipt_state.get("live_benchmarks_started") is True
        ):
            target.errors.append("agentic_loop_ledger.readiness_digest.external_eval_live_benchmarks_started must match the latest iteration.")
        if digest.get("external_eval_provider_api_calls_started") != (
            latest_external_eval_receipt_state.get("provider_api_calls_started") is True
        ):
            target.errors.append("agentic_loop_ledger.readiness_digest.external_eval_provider_api_calls_started must match the latest iteration.")
        if digest.get("external_eval_model_downloads_started") != (
            latest_external_eval_receipt_state.get("model_downloads_started") is True
        ):
            target.errors.append("agentic_loop_ledger.readiness_digest.external_eval_model_downloads_started must match the latest iteration.")
        if digest.get("external_eval_credential_values_recorded") != (
            latest_external_eval_receipt_state.get("credential_values_recorded") is True
        ):
            target.errors.append("agentic_loop_ledger.readiness_digest.external_eval_credential_values_recorded must match the latest iteration.")
        if digest.get("external_eval_cost_incurred_usd") != _safe_non_negative_number(
            latest_external_eval_receipt_state.get("cost_incurred_usd")
        ):
            target.errors.append("agentic_loop_ledger.readiness_digest.external_eval_cost_incurred_usd must match the latest iteration.")
        if digest.get("external_eval_launch_mode") != str(latest_external_eval_receipt_state.get("launch_mode") or ""):
            target.errors.append("agentic_loop_ledger.readiness_digest.external_eval_launch_mode must match the latest iteration.")
        expected_summary = str(expected_digest.get("summary") or "")
        if digest.get("summary") != expected_summary:
            target.errors.append("agentic_loop_ledger.readiness_digest.summary must match latest iteration readiness.")
    if digest.get("latest_iteration_id") != metrics.get("latest_iteration_id"):
        target.errors.append("agentic_loop_ledger.readiness_digest.latest_iteration_id must match metrics.latest_iteration_id.")
    if digest != expected_digest:
        target.errors.append("agentic_loop_ledger.readiness_digest must match the canonical latest-iteration projection exactly.")

def _validate_agentic_loop_ledger_decision(
    decision: Any,
    expected: dict[str, Any],
    target: ValidationTarget,
) -> None:
    if not isinstance(decision, dict):
        target.errors.append("agentic_loop_ledger.decision must be an object.")
        return
    _validate_allowed_keys(decision, _AGENTIC_LOOP_LEDGER_DECISION_KEYS, target, "agentic_loop_ledger.decision")
    actions = decision.get("governance_actions")
    if isinstance(actions, list):
        for index, action in enumerate(actions):
            if isinstance(action, dict):
                _validate_allowed_keys(
                    action,
                    _AGENTIC_LOOP_LEDGER_GOVERNANCE_ACTION_KEYS,
                    target,
                    f"agentic_loop_ledger.decision.governance_actions[{index}]",
                )
    for field_name in (
        "readiness",
        "recommendation",
        "recommended_governance_action",
        "governance_action_count",
        "latest_iteration_index",
        "latest_iteration_id",
        "blocked_iteration_count",
        "summary",
    ):
        if decision.get(field_name) != expected.get(field_name):
            target.errors.append(f"agentic_loop_ledger.decision.{field_name} must match latest iteration state.")
    if decision.get("governance_actions") != expected["governance_actions"]:
        target.errors.append("agentic_loop_ledger.decision.governance_actions must match latest iteration state.")

def _validate_agentic_loop_governance_receipt(receipt: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(
        receipt,
        "schema_version",
        AGENTIC_LOOP_GOVERNANCE_RECEIPT_SCHEMA_VERSION,
        target,
        prefix="agentic_loop_governance_receipt.",
    )
    _validate_allowed_keys(
        receipt,
        {
            "schema_version",
            "created_at",
            "receipt_path",
            "passed",
            "readiness",
            "recommendation",
            "check_count",
            "failed_check_count",
            "checks",
            "source_ledger",
            "requested_action",
            "decision",
            "execution_boundary",
            "notes",
        },
        target,
        "agentic_loop_governance_receipt",
    )
    receipt_path = receipt.get("receipt_path")
    if not isinstance(receipt_path, str):
        target.errors.append("agentic_loop_governance_receipt.receipt_path must be a string.")
    elif receipt_path and not _is_safe_agentic_loop_governance_path(receipt_path):
        target.errors.append("agentic_loop_governance_receipt.receipt_path must be a safe relative path or redacted placeholder.")
    checks = receipt.get("checks")
    if not isinstance(checks, list):
        target.errors.append("agentic_loop_governance_receipt.checks must be a list.")
        checks = []
    _validate_agentic_loop_governance_checks(checks, target, "agentic_loop_governance_receipt.checks")
    failed_checks = _validate_gate_like_checks(checks, target, "agentic_loop_governance_receipt.checks")
    if not _is_non_negative_int(receipt.get("check_count")):
        target.errors.append("agentic_loop_governance_receipt.check_count must be a non-negative integer.")
    elif receipt.get("check_count") != len(checks):
        target.errors.append(
            f"agentic_loop_governance_receipt.check_count expected {len(checks)}, got {receipt.get('check_count')!r}."
        )
    if not _is_non_negative_int(receipt.get("failed_check_count")):
        target.errors.append("agentic_loop_governance_receipt.failed_check_count must be a non-negative integer.")
    elif receipt.get("failed_check_count") != failed_checks:
        target.errors.append(
            "agentic_loop_governance_receipt.failed_check_count "
            f"expected {failed_checks}, got {receipt.get('failed_check_count')!r}."
        )
    if not isinstance(receipt.get("passed"), bool):
        target.errors.append("agentic_loop_governance_receipt.passed must be a boolean.")
    elif receipt.get("passed") != (failed_checks == 0):
        target.errors.append("agentic_loop_governance_receipt.passed must match failed_check_count.")
    expected_readiness_from_checks = "recorded" if failed_checks == 0 else "blocked"
    if receipt.get("readiness") != expected_readiness_from_checks:
        target.errors.append(
            "agentic_loop_governance_receipt.readiness must be recorded when checks pass and blocked otherwise."
        )

    source_ledger = receipt.get("source_ledger")
    ledger_ref = _validate_agentic_loop_governance_source_ledger(source_ledger, target, source_path)
    requested_action = receipt.get("requested_action")
    action = _validate_agentic_loop_governance_requested_action(requested_action, target)
    _validate_agentic_loop_governance_decision(receipt.get("decision"), target)
    if action in GOVERNANCE_ACTIONS and ledger_ref is not None:
        expected = _build_agentic_loop_governance_projection(ledger_ref, action)
        for field_name in ("passed", "readiness", "recommendation", "failed_check_count"):
            if receipt.get(field_name) != expected.get(field_name):
                target.errors.append(f"agentic_loop_governance_receipt.{field_name} must match current ledger action state.")
        if receipt.get("check_count") != len(expected["checks"]):
            target.errors.append("agentic_loop_governance_receipt.check_count must match replayed checks.")
        if checks != expected["checks"]:
            target.errors.append("agentic_loop_governance_receipt.checks must match current ledger action state.")
        _validate_agentic_loop_governance_requested_action_matches(
            requested_action,
            expected["requested_action"],
            target,
        )
        if receipt.get("decision") != expected["decision"]:
            target.errors.append("agentic_loop_governance_receipt.decision must match current ledger action state.")
    elif action in GOVERNANCE_ACTIONS and isinstance(source_ledger, dict):
        target.errors.append("agentic_loop_governance_receipt.source_ledger must resolve to a replayable source ledger.")
        expected = _build_agentic_loop_governance_projection(source_ledger, action)
        _validate_agentic_loop_governance_unreplayable_source_state(receipt, checks, requested_action, expected, target)
    else:
        if receipt.get("recommendation") != "fix_governance_inputs":
            target.errors.append("agentic_loop_governance_receipt.recommendation must be fix_governance_inputs when action or ledger is invalid.")

    boundary = receipt.get("execution_boundary")
    expected_boundary = _build_agentic_loop_governance_boundary()
    if isinstance(boundary, dict):
        _validate_allowed_keys(boundary, set(expected_boundary), target, "agentic_loop_governance_receipt.execution_boundary")
    if boundary != expected_boundary:
        target.errors.append("agentic_loop_governance_receipt.execution_boundary must remain receipt-only and fail-closed.")
    notes = receipt.get("notes")
    if not _is_string_list(notes):
        target.errors.append("agentic_loop_governance_receipt.notes must be a list of strings.")
    target.details.update(
        {
            "passed": receipt.get("passed"),
            "recommendation": receipt.get("recommendation"),
            "action": action,
            "latest_iteration_id": (
                ledger_ref.get("decision", {}).get("latest_iteration_id")
                if isinstance(ledger_ref, dict) and isinstance(ledger_ref.get("decision"), dict)
                else None
            ),
        }
    )

def _validate_agentic_loop_governance_source_ledger(
    value: Any,
    target: ValidationTarget,
    source_path: Path,
) -> dict[str, Any] | None:
    label = "agentic_loop_governance_receipt.source_ledger"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return None
    _validate_allowed_keys(
        value,
        {
            "role",
            "path",
            "kind",
            "exists",
            "sha256",
            "size_bytes",
            "schema_version",
            "passed",
            "decision",
            "readiness_digest",
            "execution_boundary",
        },
        target,
        label,
    )
    _validate_agentic_loop_governance_source_snapshots(value, target, label)
    if value.get("role") != "agentic_loop_ledger":
        target.errors.append(f"{label}.role must be 'agentic_loop_ledger'.")
    if value.get("kind") not in {"file", "missing"}:
        target.errors.append(f"{label}.kind must be file or missing.")
    if not isinstance(value.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    if not isinstance(value.get("path"), str) or not value.get("path"):
        target.errors.append(f"{label}.path must be a non-empty string.")
    elif not _is_safe_agentic_loop_governance_path(value["path"]):
        target.errors.append(f"{label}.path must be a safe relative path or redacted placeholder.")
    ledger_path = _resolve_promotion_decision_artifact_path(value.get("path"), source_path, value.get("kind"))
    if value.get("exists") is False and ledger_path is not None and ledger_path.exists():
        target.errors.append(f"{label}.exists must be true when path resolves to an existing file.")
    if value.get("exists") is True:
        if value.get("kind") != "file":
            target.errors.append(f"{label}.kind must be file when exists is true.")
        if not _is_sha256(value.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when exists is true.")
        if not _is_non_negative_int(value.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer when exists is true.")
    else:
        if value.get("kind") != "missing":
            target.errors.append(f"{label}.kind must be missing when exists is false.")
        if value.get("sha256") is not None:
            target.errors.append(f"{label}.sha256 must be null when exists is false.")
        if value.get("size_bytes") is not None:
            target.errors.append(f"{label}.size_bytes must be null when exists is false.")
    _validate_promotion_decision_artifact_hash(value, target, label, source_path)

    if ledger_path is None or value.get("exists") is not True:
        return None
    if _path_has_symlink_component(ledger_path, include_leaf=True) or not ledger_path.is_file():
        target.errors.append(f"{label}.path must resolve to a regular non-symlink source ledger.")
        return None
    ledger_payload = _read_json_object_silent(ledger_path)
    if not ledger_payload:
        target.errors.append(f"{label}.path must resolve to a JSON object.")
        return None
    ledger_errors_before = len(target.errors)
    _validate_agentic_loop_ledger(ledger_payload, target, ledger_path)
    ledger_replay_passed = len(target.errors) == ledger_errors_before
    expected = _build_agentic_loop_governance_ledger_ref(
        ledger_path,
        preserve_paths=False,
        source_ledger_replay_passed=ledger_replay_passed,
    )
    for field_name in ("schema_version", "passed", "decision", "readiness_digest", "execution_boundary"):
        if value.get(field_name) != expected.get(field_name):
            target.errors.append(f"{label}.{field_name} must match the current source ledger.")
    return expected

def _validate_agentic_loop_governance_source_snapshots(value: dict[str, Any], target: ValidationTarget, label: str) -> None:
    decision = value.get("decision")
    if isinstance(decision, dict):
        _validate_allowed_keys(
            decision,
            {
                "readiness",
                "recommendation",
                "recommended_governance_action",
                "latest_iteration_id",
                "latest_iteration_index",
                "summary",
                "governance_actions",
            },
            target,
            f"{label}.decision",
        )
        actions = decision.get("governance_actions")
        if isinstance(actions, list):
            for index, action in enumerate(actions):
                action_label = f"{label}.decision.governance_actions[{index}]"
                if isinstance(action, dict):
                    _validate_agentic_loop_governance_source_action(action, target, action_label)
                else:
                    target.errors.append(f"{action_label} must be an object.")
        elif actions is not None:
            target.errors.append(f"{label}.decision.governance_actions must be a list.")
    else:
        target.errors.append(f"{label}.decision must be an object.")
    digest = value.get("readiness_digest")
    if isinstance(digest, dict):
        _validate_allowed_keys(
            digest,
            {
                "latest_iteration_id",
                "latest_iteration_index",
                "plan_readiness",
                "execution_completion",
                "governance_readiness",
                "readiness",
                "recommendation",
                "ready_for_governance_review",
                "recommended_governance_action",
                "promotion_decision_present",
                "promotion_ledger_present",
                "rollback_receipt_present",
                "cloud_training_completion_successful",
                "cloud_training_completion_integrity_passed",
                "cloud_training_completion_execution_status",
                "cloud_training_completion_execution_terminal",
                "cloud_training_completion_governance_readiness",
                "cloud_training_completion_claims_allowed",
                "cloud_training_completion_provider_id",
                "cloud_training_completion_provider_job_id",
                "cloud_training_completion_execution_id",
                "cloud_training_completion_provider_matches_pipeline",
                "cloud_training_completion_candidate_model_id",
                "cloud_training_completion_loop_candidate_model_id",
                "cloud_training_completion_candidate_matches_loop",
                "cloud_training_completion_candidate_matches_training_result",
                "cloud_training_completion_source_bindings_complete",
                "cloud_training_completion_output_artifact_manifest_bound",
                "cloud_training_completion_output_artifact_set_bound",
                "cloud_training_completion_output_artifact_count",
                "side_effects_started",
                "summary",
            },
            target,
            f"{label}.readiness_digest",
        )
    else:
        target.errors.append(f"{label}.readiness_digest must be an object.")
    boundary = value.get("execution_boundary")
    if isinstance(boundary, dict):
        _validate_allowed_keys(
            boundary,
            {
                "ledger_only",
                "cloud_jobs_started",
                "paid_model_grader_calls_started",
                "live_benchmarks_started",
                "model_downloads_started",
                "weights_updated_by_flight_recorder",
                "credential_values_recorded",
            },
            target,
            f"{label}.execution_boundary",
        )
    else:
        target.errors.append(f"{label}.execution_boundary must be an object.")

def _validate_agentic_loop_governance_source_action(value: dict[str, Any], target: ValidationTarget, label: str) -> None:
    _validate_allowed_keys(
        value,
        {"action", "available", "blocked_reason_count", "blocked_reasons", "summary"},
        target,
        label,
    )
    if value.get("action") not in GOVERNANCE_ACTIONS:
        target.errors.append(f"{label}.action must be one of {sorted(GOVERNANCE_ACTIONS)!r}.")
    if not isinstance(value.get("available"), bool):
        target.errors.append(f"{label}.available must be a boolean.")
    blocked_reasons = value.get("blocked_reasons")
    if not _is_string_list(blocked_reasons):
        target.errors.append(f"{label}.blocked_reasons must be a list of strings.")
        blocked_reasons = []
    if not _is_non_negative_int(value.get("blocked_reason_count")):
        target.errors.append(f"{label}.blocked_reason_count must be a non-negative integer.")
    elif value.get("blocked_reason_count") != len(blocked_reasons):
        target.errors.append(f"{label}.blocked_reason_count must match blocked_reasons.")
    if not isinstance(value.get("summary"), str):
        target.errors.append(f"{label}.summary must be a string.")

def _is_safe_agentic_loop_governance_path(value: str) -> bool:
    if value.startswith("<redacted:") and value.endswith(">"):
        basename = value.removeprefix("<redacted:").removesuffix(">")
        return bool(basename) and "/" not in basename and "\\" not in basename and ".." not in basename and "~" not in basename
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

def _validate_agentic_loop_governance_requested_action(value: Any, target: ValidationTarget) -> str:
    label = "agentic_loop_governance_receipt.requested_action"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return ""
    _validate_allowed_keys(
        value,
        {
            "action",
            "available",
            "blocked_reason_count",
            "blocked_reasons",
            "summary",
            "requested_by",
            "reason",
        },
        target,
        label,
    )
    action = value.get("action")
    if action not in GOVERNANCE_ACTIONS:
        target.errors.append(f"{label}.action must be one of {sorted(GOVERNANCE_ACTIONS)!r}.")
        action = ""
    if not isinstance(value.get("available"), bool):
        target.errors.append(f"{label}.available must be a boolean.")
    reasons = value.get("blocked_reasons")
    if not _is_string_list(reasons):
        target.errors.append(f"{label}.blocked_reasons must be a list of strings.")
        reasons = []
    if not _is_non_negative_int(value.get("blocked_reason_count")):
        target.errors.append(f"{label}.blocked_reason_count must be a non-negative integer.")
    elif value.get("blocked_reason_count") != len(reasons):
        target.errors.append(f"{label}.blocked_reason_count must match blocked_reasons.")
    for field_name in ("summary", "requested_by", "reason"):
        if not isinstance(value.get(field_name), str) or not value.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    return str(action or "")

def _validate_agentic_loop_governance_requested_action_matches(
    actual: Any,
    expected: dict[str, Any],
    target: ValidationTarget,
) -> None:
    if not isinstance(actual, dict):
        return
    for field_name in ("action", "available", "blocked_reason_count", "blocked_reasons", "summary"):
        if actual.get(field_name) != expected.get(field_name):
            target.errors.append(f"agentic_loop_governance_receipt.requested_action.{field_name} must match current ledger action state.")

def _validate_agentic_loop_governance_decision(value: Any, target: ValidationTarget) -> None:
    label = "agentic_loop_governance_receipt.decision"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(
        value,
        {
            "readiness",
            "recommendation",
            "summary",
            "selected_action",
            "ledger_recommended_governance_action",
            "latest_iteration_id",
            "blocking_check_count",
            "blocking_checks",
        },
        target,
        label,
    )
    blocking_checks = value.get("blocking_checks")
    if not _is_non_negative_int(value.get("blocking_check_count")):
        target.errors.append(f"{label}.blocking_check_count must be a non-negative integer.")
    if isinstance(blocking_checks, list):
        if _is_non_negative_int(value.get("blocking_check_count")) and value.get("blocking_check_count") != len(blocking_checks):
            target.errors.append(f"{label}.blocking_check_count must match blocking_checks.")
        for index, check in enumerate(blocking_checks):
            check_label = f"{label}.blocking_checks[{index}]"
            if isinstance(check, dict):
                _validate_allowed_keys(check, {"id", "summary", "scope"}, target, check_label)
                scope = check.get("scope")
                if isinstance(scope, dict):
                    _validate_allowed_keys(scope, set(), target, f"{check_label}.scope")
                else:
                    target.errors.append(f"{check_label}.scope must be an object.")
            else:
                target.errors.append(f"{check_label} must be an object.")
    elif blocking_checks is not None:
        target.errors.append(f"{label}.blocking_checks must be a list.")

def _validate_agentic_loop_governance_checks(checks: list[Any], target: ValidationTarget, label: str) -> None:
    for index, check in enumerate(checks):
        check_label = f"{label}[{index}]"
        if isinstance(check, dict):
            _validate_allowed_keys(check, {"id", "passed", "actual", "expected", "summary"}, target, check_label)
        else:
            target.errors.append(f"{check_label} must be an object.")

def _validate_agentic_loop_governance_unreplayable_source_state(
    receipt: dict[str, Any],
    checks: list[Any],
    requested_action: Any,
    expected: dict[str, Any],
    target: ValidationTarget,
) -> None:
    if receipt.get("passed") is not False:
        target.errors.append("agentic_loop_governance_receipt.passed must be false when source ledger cannot be replayed.")
    if receipt.get("readiness") != "blocked":
        target.errors.append("agentic_loop_governance_receipt.readiness must be blocked when source ledger cannot be replayed.")
    if receipt.get("recommendation") != "fix_governance_inputs":
        target.errors.append(
            "agentic_loop_governance_receipt.recommendation must be fix_governance_inputs when source ledger cannot be replayed."
        )
    if not _is_non_negative_int(receipt.get("failed_check_count")) or receipt.get("failed_check_count") <= 0:
        target.errors.append("agentic_loop_governance_receipt.failed_check_count must be positive when source ledger cannot be replayed.")
    if receipt.get("check_count") != len(expected["checks"]):
        target.errors.append("agentic_loop_governance_receipt.check_count must match fail-closed source-ledger checks.")
    if checks != expected["checks"]:
        target.errors.append("agentic_loop_governance_receipt.checks must match fail-closed source-ledger checks.")
    _validate_agentic_loop_governance_requested_action_matches(
        requested_action,
        expected["requested_action"],
        target,
    )
    if receipt.get("decision") != expected["decision"]:
        target.errors.append("agentic_loop_governance_receipt.decision must match fail-closed source-ledger checks.")

def _validate_agentic_loop_ledger_cost_estimate(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_LOOP_LEDGER_COST_KEYS, target, label)
    for field_name in ("max_cloud_cost_usd", "max_gpu_hours"):
        metric = value.get(field_name)
        if metric is not None and (not isinstance(metric, (int, float)) or isinstance(metric, bool) or metric < 0):
            target.errors.append(f"{label}.{field_name} must be a non-negative number or null.")
    if not isinstance(value.get("live_spend_allowed"), bool):
        target.errors.append(f"{label}.live_spend_allowed must be a boolean.")

def _validate_agentic_loop_ledger_basic_group(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_LOOP_LEDGER_BASIC_GROUP_KEYS, target, label)
    if not isinstance(value.get("group"), str) or not value.get("group"):
        target.errors.append(f"{label}.group must be a non-empty string.")
    if not _is_non_negative_int(value.get("artifact_count")):
        target.errors.append(f"{label}.artifact_count must be a non-negative integer.")
    for field_name in ("roles_present", "roles_missing"):
        if not _is_string_list(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a list of strings.")

def _validate_agentic_loop_ledger_next_actions(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_LOOP_LEDGER_NEXT_ACTION_KEYS, target, label)
    for field_name in ("scheduled", "requires_governance_decision"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if not isinstance(value.get("recommendation"), str):
        target.errors.append(f"{label}.recommendation must be a string.")
    if not isinstance(value.get("schedule"), dict):
        target.errors.append(f"{label}.schedule must be an object.")

def _validate_agentic_loop_ledger_cloud_training(value: Any, row: dict[str, Any], target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_LOOP_LEDGER_CLOUD_TRAINING_KEYS, target, label)
    role_counts = _agentic_loop_count_map(row.get("artifact_role_counts"), "role")
    present = [role for role in AGENTIC_LOOP_CLOUD_TRAINING_ROLES if role_counts.get(role, 0) > 0]
    missing = [role for role in AGENTIC_LOOP_CLOUD_TRAINING_ROLES if role not in present]
    expected = {
        "group": "cloud_training",
        "artifact_count": sum(role_counts.get(role, 0) for role in AGENTIC_LOOP_CLOUD_TRAINING_ROLES),
        "roles_present": present,
        "roles_missing": missing,
        "provider_registry_present": "cloud_training_provider_registry" in present,
        "preflight_present": "cloud_training_preflight" in present,
        "artifact_manifest_present": "cloud_training_artifact_manifest" in present,
        "launch_plan_present": "cloud_training_launch_plan" in present,
        "launch_receipt_present": "cloud_training_launch_receipt" in present,
        "status_receipt_present": "cloud_training_status_receipt" in present,
        "completion_receipt_present": "cloud_training_completion_receipt" in present,
        "provider_api_calls_started": False,
        "cloud_jobs_started": False,
        "credential_values_recorded": False,
        "live_spend_allowed": False,
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} must match ledger cloud training role counts.")

def _validate_agentic_loop_ledger_cloud_training_receipt_state(
    value: Any,
    row: dict[str, Any],
    target: ValidationTarget,
    ledger_path: Path,
    label: str,
) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    source_path = _resolve_agentic_loop_ledger_source_path(row, ledger_path)
    if (
        source_path is None
        or not source_path.exists()
        or _path_has_symlink_component(source_path, include_leaf=True)
        or not source_path.is_file()
    ):
        return
    source_plan = _read_json_object_silent(source_path)
    source_artifacts = source_plan.get("source_artifacts") if isinstance(source_plan.get("source_artifacts"), dict) else {}
    expected = _expected_agentic_training_loop_cloud_training_receipt_state(source_artifacts, source_path)
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} must match source loop plan cloud training receipt artifacts.")

def _validate_agentic_loop_ledger_cloud_training_completion_state(
    value: Any,
    row: dict[str, Any],
    target: ValidationTarget,
    ledger_path: Path,
    label: str,
) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(
        value,
        _AGENTIC_TRAINING_LOOP_CLOUD_COMPLETION_STATE_KEYS,
        target,
        label,
    )
    source_path = _resolve_agentic_loop_ledger_source_path(row, ledger_path)
    if (
        source_path is None
        or not source_path.exists()
        or _path_has_symlink_component(source_path, include_leaf=True)
        or not source_path.is_file()
    ):
        return
    source_plan = _read_json_object_silent(source_path)
    expected = (
        source_plan.get("cloud_training_completion_state")
        if isinstance(source_plan.get("cloud_training_completion_state"), dict)
        else {}
    )
    if value != expected:
        target.errors.append(
            f"{label} must match source loop plan cloud completion state exactly."
        )

def _validate_agentic_loop_ledger_external_eval_receipt_state(
    value: Any,
    row: dict[str, Any],
    target: ValidationTarget,
    ledger_path: Path,
    label: str,
) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_LOOP_EXTERNAL_EVAL_RECEIPT_STATE_KEYS, target, label)
    source_path = _resolve_agentic_loop_ledger_source_path(row, ledger_path)
    if (
        source_path is None
        or not source_path.exists()
        or _path_has_symlink_component(source_path, include_leaf=True)
        or not source_path.is_file()
    ):
        return
    source_plan = _read_json_object_silent(source_path)
    source_artifacts = source_plan.get("source_artifacts") if isinstance(source_plan.get("source_artifacts"), dict) else {}
    expected = _expected_agentic_training_loop_external_eval_receipt_state(source_artifacts, source_path)
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} must match source loop plan external eval receipt artifacts.")

def _validate_agentic_loop_ledger_cloud_training_lineage(
    value: Any,
    row: dict[str, Any],
    target: ValidationTarget,
    ledger_path: Path,
    row_label: str,
) -> None:
    label = f"{row_label}.cloud_training_lineage"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_TRAINING_LOOP_CLOUD_LINEAGE_KEYS, target, label)
    source_path = _resolve_agentic_loop_ledger_source_path(row, ledger_path)
    if (
        source_path is None
        or not source_path.exists()
        or _path_has_symlink_component(source_path, include_leaf=True)
        or not source_path.is_file()
    ):
        return
    source_plan = _read_json_object_silent(source_path)
    expected = source_plan.get("cloud_training_lineage") if isinstance(source_plan.get("cloud_training_lineage"), dict) else {}
    if value != expected:
        target.errors.append(f"{label} must match the source loop plan cloud_training_lineage.")

def _agentic_loop_role_count(source_artifacts: dict[str, Any], role: str) -> int:
    rows = source_artifacts.get(role)
    return len(rows) if isinstance(rows, list) else 0

def _agentic_loop_role_ready(source_artifacts: dict[str, Any], role: str) -> bool:
    rows = source_artifacts.get(role)
    return bool(rows) and all(isinstance(row, dict) and row.get("exists") is True for row in rows)

def _agentic_loop_count_map(value: Any, key_name: str) -> dict[str, int]:
    if not isinstance(value, list):
        return {}
    counts: dict[str, int] = {}
    for row in value:
        if isinstance(row, dict) and isinstance(row.get(key_name), str):
            counts[row[key_name]] = _safe_non_negative_int(row.get("count"))
    return counts

def _validate_agentic_loop_ledger_counts(value: Any, target: ValidationTarget, label: str, key_name: str) -> None:
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list.")
        return
    seen: set[str] = set()
    for index, row in enumerate(value):
        row_label = f"{label}[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{row_label} must be an object.")
            continue
        _validate_allowed_keys(row, {key_name, "count"}, target, row_label)
        key = row.get(key_name)
        if not isinstance(key, str) or not key:
            target.errors.append(f"{row_label}.{key_name} must be a non-empty string.")
        elif key in seen:
            target.errors.append(f"{row_label}.{key_name} duplicates {key!r}.")
        seen.add(str(key))
        if not _is_non_negative_int(row.get("count")):
            target.errors.append(f"{row_label}.count must be a non-negative integer.")

def _validate_agentic_loop_ledger_source(row: dict[str, Any], target: ValidationTarget, ledger_path: Path, label: str) -> None:
    path_value = row.get("path")
    if not isinstance(path_value, str) or not path_value:
        target.errors.append(f"{label}.path must be a non-empty string.")
        return
    if not _is_safe_agentic_training_result_path(path_value):
        target.errors.append(f"{label}.path must be a safe relative path without traversal.")
        return
    source_path = _resolve_agentic_loop_ledger_source_path(row, ledger_path)
    if source_path is None:
        target.errors.append(f"{label}.path must be a safe relative path without traversal.")
        return
    if not source_path.exists():
        target.errors.append(f"{label}.path must resolve to an existing source loop plan.")
        return
    if _path_has_symlink_component(source_path, include_leaf=True) or not source_path.is_file():
        target.errors.append(f"{label}.path must resolve to a regular non-symlink source loop plan.")
        return
    if row.get("exists") is not True:
        target.errors.append(f"{label}.exists must be true.")
    size = row.get("size_bytes")
    if not _is_non_negative_int(size):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    elif source_path.stat().st_size != size:
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    sha = row.get("sha256")
    if not _is_sha256(sha):
        target.errors.append(f"{label}.sha256 must be a sha256 hex digest.")
    elif _sha256(source_path) != sha:
        target.errors.append(f"{label}.sha256 does not match the current file.")
    source_plan = _read_json_object_silent(source_path)
    if not source_plan:
        target.errors.append(f"{label}.path must resolve to a JSON object source loop plan.")
        return
    _validate_agentic_training_loop_plan(source_plan, target, source_path)
    index = row.get("index")
    if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
        expected = _build_agentic_loop_ledger_iteration(
            source_path,
            source_plan,
            index,
            ledger_path,
            False,
        )
        if row != expected:
            target.errors.append(f"{label} must match the current source loop plan projection exactly.")

def _resolve_agentic_loop_ledger_source_path(row: dict[str, Any], ledger_path: Path) -> Path | None:
    path_value = row.get("path")
    if not isinstance(path_value, str) or not path_value or not _is_safe_agentic_training_result_path(path_value):
        return None
    return ledger_path.parent / path_value

def _validate_next_iteration_schedule(schedule: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(schedule, "schema_version", NEXT_ITERATION_SCHEDULE_SCHEMA_VERSION, target, prefix="next_iteration_schedule.")
    _validate_allowed_keys(
        schedule,
        {
            "schema_version",
            "created_at",
            "schedule_path",
            "passed",
            "readiness",
            "recommendation",
            "check_count",
            "failed_check_count",
            "checks",
            "blocked_reasons",
            "source_ledgers",
            "pressure",
            "next_iteration",
            "execution_boundary",
            "notes",
        },
        target,
        "next_iteration_schedule",
    )
    schedule_path = schedule.get("schedule_path")
    if not isinstance(schedule_path, str):
        target.errors.append("next_iteration_schedule.schedule_path must be a string.")
    elif schedule_path and not _is_safe_next_iteration_schedule_path(schedule_path):
        target.errors.append("next_iteration_schedule.schedule_path must be a safe relative path or redacted placeholder.")
    checks = schedule.get("checks")
    if not isinstance(checks, list):
        target.errors.append("next_iteration_schedule.checks must be a list.")
        checks = []
    failed_checks = _validate_gate_like_checks(checks, target, "next_iteration_schedule.checks")
    for index, check in enumerate(checks):
        if isinstance(check, dict):
            _validate_allowed_keys(
                check,
                {"id", "passed", "actual", "expected", "summary"},
                target,
                f"next_iteration_schedule.checks[{index}]",
            )
    if schedule.get("check_count") != len(checks):
        target.errors.append(f"next_iteration_schedule.check_count expected {len(checks)}, got {schedule.get('check_count')!r}.")
    if schedule.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"next_iteration_schedule.failed_check_count expected {failed_checks}, got {schedule.get('failed_check_count')!r}."
        )
    if schedule.get("passed") != (failed_checks == 0):
        target.errors.append("next_iteration_schedule.passed must match failed_check_count.")
    expected_readiness = "ready_to_schedule" if failed_checks == 0 else "blocked"
    if schedule.get("readiness") != expected_readiness:
        target.errors.append(f"next_iteration_schedule.readiness expected {expected_readiness!r}, got {schedule.get('readiness')!r}.")
    expected_recommendations = {"create_next_loop_plan", "monitor_or_promote_without_new_iteration"} if failed_checks == 0 else {"fix_schedule_inputs"}
    if schedule.get("recommendation") not in expected_recommendations:
        target.errors.append("next_iteration_schedule.recommendation does not match readiness.")
    if not _is_string_list(schedule.get("blocked_reasons")):
        target.errors.append("next_iteration_schedule.blocked_reasons must be a list of strings.")
    else:
        expected_blocked_reasons = [check["summary"] for check in checks if isinstance(check, dict) and check.get("passed") is False]
        if schedule.get("blocked_reasons") != expected_blocked_reasons:
            target.errors.append("next_iteration_schedule.blocked_reasons must match failed check summaries.")

    ledgers = schedule.get("source_ledgers")
    if not isinstance(ledgers, dict):
        target.errors.append("next_iteration_schedule.source_ledgers must be an object.")
        ledgers = {}
    else:
        _validate_allowed_keys(
            ledgers,
            {"agentic_loop_ledger", "action_ledger", "improvement_ledger"},
            target,
            "next_iteration_schedule.source_ledgers",
        )
    expected_sources = {
        "agentic_loop_ledger": "hfr.agentic_loop_ledger.v1",
        "action_ledger": "hfr.action_ledger.v1",
        "improvement_ledger": "hfr.improvement_ledger.v1",
    }
    source_metrics: dict[str, dict[str, Any]] = {}
    for role, schema_version in expected_sources.items():
        rows = ledgers.get(role)
        if not isinstance(rows, list) or len(rows) != 1:
            target.errors.append(f"next_iteration_schedule.source_ledgers.{role} must contain exactly one ref.")
            continue
        source_metrics[role] = _validate_next_iteration_schedule_ref(
            rows[0],
            target,
            f"next_iteration_schedule.source_ledgers.{role}[0]",
            role,
            schema_version,
            source_path,
        )

    pressure = schedule.get("pressure")
    if not isinstance(pressure, dict):
        target.errors.append("next_iteration_schedule.pressure must be an object.")
        pressure = {}
    else:
        _validate_allowed_keys(
            pressure,
            {
                "latest_loop_readiness",
                "latest_missing_phase_input_count",
                "open_action_count",
                "recurring_action_count",
                "open_work_item_count",
                "critical_open_work_item_count",
                "high_open_work_item_count",
                "total_open_signal_count",
            },
            target,
            "next_iteration_schedule.pressure",
        )
        for field_name in (
            "latest_missing_phase_input_count",
            "open_action_count",
            "recurring_action_count",
            "open_work_item_count",
            "critical_open_work_item_count",
            "high_open_work_item_count",
            "total_open_signal_count",
        ):
            if not _is_non_negative_int(pressure.get(field_name)):
                target.errors.append(f"next_iteration_schedule.pressure.{field_name} must be a non-negative integer.")
        expected_total = (
            _safe_non_negative_int(pressure.get("latest_missing_phase_input_count"))
            + _safe_non_negative_int(pressure.get("open_action_count"))
            + _safe_non_negative_int(pressure.get("open_work_item_count"))
        )
        if _is_non_negative_int(pressure.get("total_open_signal_count")) and pressure.get("total_open_signal_count") != expected_total:
            target.errors.append("next_iteration_schedule.pressure.total_open_signal_count must equal missing phases + open actions + open work.")
        _validate_next_iteration_pressure_matches_sources(pressure, source_metrics, target)

    if failed_checks == 0:
        expected_recommendation = (
            "create_next_loop_plan"
            if _safe_non_negative_int(pressure.get("total_open_signal_count")) > 0
            else "monitor_or_promote_without_new_iteration"
        )
    else:
        expected_recommendation = "fix_schedule_inputs"
    if schedule.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"next_iteration_schedule.recommendation expected {expected_recommendation!r}, got {schedule.get('recommendation')!r}."
        )

    next_iteration = schedule.get("next_iteration")
    if not isinstance(next_iteration, dict):
        target.errors.append("next_iteration_schedule.next_iteration must be an object.")
    else:
        _validate_allowed_keys(
            next_iteration,
            {"iteration_id", "objective", "latest_iteration_id", "scheduled", "trigger", "schedule", "reason"},
            target,
            "next_iteration_schedule.next_iteration",
        )
        for field_name in ("iteration_id", "objective", "trigger", "reason"):
            if not isinstance(next_iteration.get(field_name), str) or not next_iteration.get(field_name):
                target.errors.append(f"next_iteration_schedule.next_iteration.{field_name} must be a non-empty string.")
        if not isinstance(next_iteration.get("latest_iteration_id"), str):
            target.errors.append("next_iteration_schedule.next_iteration.latest_iteration_id must be a string.")
        loop_metrics = source_metrics.get("agentic_loop_ledger", {})
        latest_iteration_id = loop_metrics.get("latest_iteration_id")
        if isinstance(latest_iteration_id, str) and next_iteration.get("latest_iteration_id") != latest_iteration_id:
            target.errors.append("next_iteration_schedule.next_iteration.latest_iteration_id must match source loop ledger metrics.")
        if next_iteration.get("scheduled") is not False:
            target.errors.append("next_iteration_schedule.next_iteration.scheduled must be false.")
        if next_iteration.get("trigger") != "manual_or_external_scheduler":
            target.errors.append("next_iteration_schedule.next_iteration.trigger must be manual_or_external_scheduler.")
        if not isinstance(next_iteration.get("schedule"), dict):
            target.errors.append("next_iteration_schedule.next_iteration.schedule must be an object.")

    boundary = schedule.get("execution_boundary")
    if not isinstance(boundary, dict):
        target.errors.append("next_iteration_schedule.execution_boundary must be an object.")
    else:
        _validate_allowed_keys(
            boundary,
            {
                "schedule_only",
                "automations_created",
                "codex_threads_created",
                "calendar_events_created",
                "cloud_jobs_started",
                "weights_updated_by_flight_recorder",
                "credential_values_recorded",
            },
            target,
            "next_iteration_schedule.execution_boundary",
        )
        if boundary.get("schedule_only") is not True:
            target.errors.append("next_iteration_schedule.execution_boundary.schedule_only must be true.")
        for field_name in (
            "automations_created",
            "codex_threads_created",
            "calendar_events_created",
            "cloud_jobs_started",
            "weights_updated_by_flight_recorder",
            "credential_values_recorded",
        ):
            if boundary.get(field_name) is not False:
                target.errors.append(f"next_iteration_schedule.execution_boundary.{field_name} must be false.")
    target.details.update(
        {
            "readiness": schedule.get("readiness"),
            "recommendation": schedule.get("recommendation"),
            "next_iteration_id": next_iteration.get("iteration_id") if isinstance(next_iteration, dict) else None,
        }
    )

def _validate_next_iteration_schedule_ref(
    row: Any,
    target: ValidationTarget,
    label: str,
    role: str,
    schema_version: str,
    source_path: Path,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        target.errors.append(f"{label} must be an object.")
        return {}
    _validate_allowed_keys(
        row,
        {
            "role",
            "path",
            "kind",
            "exists",
            "sha256",
            "size_bytes",
            "schema_version",
            "passed",
            "metrics",
            "decision",
        },
        target,
        label,
    )
    if row.get("role") != role:
        target.errors.append(f"{label}.role must be {role!r}.")
    if row.get("kind") != "file":
        target.errors.append(f"{label}.kind must be 'file'.")
    if row.get("exists") is not True:
        target.errors.append(f"{label}.exists must be true.")
    if row.get("schema_version") != schema_version:
        target.errors.append(f"{label}.schema_version must be {schema_version!r}.")
    if row.get("passed") is not True:
        target.errors.append(f"{label}.passed must be true.")
    path_is_safe = False
    if not isinstance(row.get("path"), str) or not row.get("path"):
        target.errors.append(f"{label}.path must be a non-empty string.")
    else:
        path_is_safe = _is_safe_next_iteration_schedule_path(row["path"])
        if not path_is_safe:
            target.errors.append(f"{label}.path must be a safe relative path or redacted placeholder.")
    if not _is_sha256(row.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string.")
    if not _is_non_negative_int(row.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    metrics = row.get("metrics")
    if metrics is not None:
        if not isinstance(metrics, dict):
            target.errors.append(f"{label}.metrics must be an object when present.")
            metrics = {}
        else:
            _validate_next_iteration_schedule_ref_metrics(metrics, target, f"{label}.metrics")
    decision = row.get("decision")
    if decision is not None:
        if not isinstance(decision, dict):
            target.errors.append(f"{label}.decision must be an object when present.")
        else:
            _validate_allowed_keys(decision, {"readiness", "recommendation", "summary"}, target, f"{label}.decision")
            for field_name in ("readiness", "recommendation", "summary"):
                if not isinstance(decision.get(field_name), str):
                    target.errors.append(f"{label}.decision.{field_name} must be a string.")

    path_value = row.get("path")
    if not isinstance(path_value, str) or not path_value:
        return metrics if isinstance(metrics, dict) else {}
    if not path_is_safe or path_value.startswith("<redacted:"):
        return metrics if isinstance(metrics, dict) else {}
    ledger_path = _next_iteration_schedule_ref_path(path_value, source_path)
    if not ledger_path.is_file():
        target.errors.append(f"{label}.path does not resolve to an existing ledger file.")
        return metrics if isinstance(metrics, dict) else {}
    if _path_has_symlink_component(ledger_path, include_leaf=True):
        target.errors.append(f"{label}.path must resolve to a regular non-symlink ledger file.")
        return metrics if isinstance(metrics, dict) else {}
    if _is_non_negative_int(row.get("size_bytes")) and ledger_path.stat().st_size != row.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if _is_sha256(row.get("sha256")) and _sha256(ledger_path) != row.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current file.")
    payload = _read_object(ledger_path, target, f"{label}.path")
    if payload is None:
        return metrics if isinstance(metrics, dict) else {}
    if payload.get("schema_version") != row.get("schema_version"):
        target.errors.append(f"{label}.schema_version must match the current file.")
    if payload.get("passed") != row.get("passed"):
        target.errors.append(f"{label}.passed must match the current file.")
    current_metrics = _next_iteration_compact_metrics(payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {})
    if isinstance(metrics, dict) and metrics != current_metrics:
        target.errors.append(f"{label}.metrics must match the current file.")
    current_decision = _next_iteration_compact_decision(payload.get("decision") if isinstance(payload.get("decision"), dict) else {})
    if isinstance(decision, dict) and decision != current_decision:
        target.errors.append(f"{label}.decision must match the current file.")
    return current_metrics

def _validate_next_iteration_schedule_ref_metrics(metrics: dict[str, Any], target: ValidationTarget, label: str) -> None:
    _validate_allowed_keys(
        metrics,
        {
            "latest_iteration_id",
            "latest_readiness",
            "latest_missing_phase_input_count",
            "open_action_count",
            "new_action_count",
            "recurring_action_count",
            "open_work_item_count",
            "critical_open_work_item_count",
            "high_open_work_item_count",
            "resolved_work_item_count",
        },
        target,
        label,
    )
    for field_name in ("latest_iteration_id", "latest_readiness"):
        if field_name in metrics and not isinstance(metrics.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string when present.")
    for field_name in (
        "latest_missing_phase_input_count",
        "open_action_count",
        "new_action_count",
        "recurring_action_count",
        "open_work_item_count",
        "critical_open_work_item_count",
        "high_open_work_item_count",
        "resolved_work_item_count",
    ):
        if field_name in metrics and not _is_non_negative_int(metrics.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer when present.")

def _validate_next_iteration_pressure_matches_sources(
    pressure: dict[str, Any],
    source_metrics: dict[str, dict[str, Any]],
    target: ValidationTarget,
) -> None:
    loop_metrics = source_metrics.get("agentic_loop_ledger", {})
    action_metrics = source_metrics.get("action_ledger", {})
    improvement_metrics = source_metrics.get("improvement_ledger", {})
    expected = {
        "latest_loop_readiness": str(loop_metrics.get("latest_readiness") or ""),
        "latest_missing_phase_input_count": _safe_non_negative_int(loop_metrics.get("latest_missing_phase_input_count")),
        "open_action_count": _safe_non_negative_int(action_metrics.get("open_action_count")),
        "recurring_action_count": _safe_non_negative_int(action_metrics.get("recurring_action_count")),
        "open_work_item_count": _safe_non_negative_int(improvement_metrics.get("open_work_item_count")),
        "critical_open_work_item_count": _safe_non_negative_int(improvement_metrics.get("critical_open_work_item_count")),
        "high_open_work_item_count": _safe_non_negative_int(improvement_metrics.get("high_open_work_item_count")),
    }
    expected["total_open_signal_count"] = (
        expected["latest_missing_phase_input_count"] + expected["open_action_count"] + expected["open_work_item_count"]
    )
    for field_name, expected_value in expected.items():
        if pressure.get(field_name) != expected_value:
            target.errors.append(f"next_iteration_schedule.pressure.{field_name} must match source ledgers.")

def _next_iteration_compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "latest_iteration_id",
        "latest_readiness",
        "latest_missing_phase_input_count",
        "open_action_count",
        "new_action_count",
        "recurring_action_count",
        "open_work_item_count",
        "critical_open_work_item_count",
        "high_open_work_item_count",
        "resolved_work_item_count",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}

def _next_iteration_compact_decision(decision: dict[str, Any]) -> dict[str, Any]:
    if not decision:
        return {}
    return {
        "readiness": str(decision.get("readiness") or ""),
        "recommendation": str(decision.get("recommendation") or ""),
        "summary": str(decision.get("summary") or ""),
    }

def _next_iteration_schedule_ref_path(value: str, source_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return source_path.parent / path

def _is_safe_next_iteration_schedule_path(value: str) -> bool:
    if value.startswith("<redacted:") and value.endswith(">"):
        basename = value.removeprefix("<redacted:").removesuffix(">")
        return bool(basename) and "/" not in basename and "\\" not in basename and ".." not in basename and "~" not in basename
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

def _safe_non_negative_int(value: Any) -> int:
    return value if _is_non_negative_int(value) else 0

def _safe_non_negative_number(value: Any) -> int | float:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else 0
