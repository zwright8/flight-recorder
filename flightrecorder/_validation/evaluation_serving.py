"""Extracted validation implementation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any
from ..artifacts import CONTRACT_SCOPES, SUITE_TREND_SCHEMA_VERSION
from ..schema_registry import SchemaRegistryError, check_schema_contract, check_schema_file
from ..source_contract import get_active_opaque_output_attestation, inspect_artifact_source
from ..dataset_curation import DATASET_CURATION_RECEIPT_SCHEMA_VERSION, training_export_lineage_status
from ..evidence import EVIDENCE_COVERAGE_SCHEMA_VERSION
from ..eval_summary import EVAL_SUMMARY_SCHEMA_VERSION, LabeledPath, _external_adapter_plan as _build_eval_summary_external_plan, _external_adapter_result as _build_eval_summary_external_result, _external_result_association_risks as _build_eval_summary_external_result_risks, _heldout_scenario_summary as _build_eval_summary_heldout, _suite_arm
from ..external_eval import ADAPTERS, EXTERNAL_EVAL_ADAPTER_CONTRACT_VERSION, EXTERNAL_EVAL_ADAPTER_RECEIPT_TYPES, EXTERNAL_EVAL_PLAN_SCHEMA_VERSION, EXTERNAL_EVAL_RECEIPT_SCHEMA_VERSION, ExternalEvalPlanError, build_external_eval_receipt, external_eval_plan_semantic_errors
from ..external_eval_result import EXTERNAL_EVAL_RESULT_SCHEMA_VERSION, ExternalEvalResultError, build_external_eval_result
from ..heldout_manifest import HELDOUT_MANIFEST_SCHEMA_VERSION, _manifest_status as _build_heldout_manifest_status, _source_fields_from_suite_summary as _build_heldout_source_fields
from ..model_grader import MODEL_GRADER_DISAGREEMENT_QUEUE_SCHEMA_VERSION, MODEL_GRADER_DRY_RUN_SCHEMA_VERSION, MODEL_GRADER_GATE_SCHEMA_VERSION, MODEL_GRADER_OVERRIDE_RECEIPT_SCHEMA_VERSION, RUBRIC_SPEC_SCHEMA_VERSION, ModelGraderError, _model_grader_review_lineage_status, build_model_grader_gate
from ..rejection_sampling import REJECTION_SAMPLING_GATE_SCHEMA_VERSION, _review_lineage_status
from ..review import REVIEW_CONFIDENCE_LEVELS, REVIEW_ITEM_SCHEMA_VERSION, REVIEW_LABEL_SCHEMA_VERSION, REVIEW_LABELS, REVIEW_MANIFEST_SCHEMA_VERSION, TRAINING_NEGATIVE_LABELS, review_item_sha256, _reviewed_dpo as _build_reviewed_dpo, _reviewed_labels as _build_reviewed_labels, _reviewed_preferences as _build_reviewed_preferences, _reviewed_reward_model as _build_reviewed_reward_model, _reviewed_action_sft as _build_reviewed_action_sft, _reviewed_sft as _build_reviewed_sft, REVIEWED_DPO_SCHEMA_VERSION, REVIEWED_LABEL_SCHEMA_VERSION, REVIEWED_MANIFEST_SCHEMA_VERSION, REVIEWED_PREFERENCE_SCHEMA_VERSION, REVIEWED_REWARD_MODEL_SCHEMA_VERSION, REVIEWED_SFT_SCHEMA_VERSION, _reviewed_dataset_version_id
from ..rollout_generation import AGENTIC_ROLLOUT_PLAN_SCHEMA_VERSION, AGENTIC_ROLLOUT_RECEIPT_SCHEMA_VERSION
from ..path_safety import path_has_symlink_component as _path_has_symlink_component, resolve_artifact_reference_path
from ..scenario_check import SCENARIO_CHECK_SCHEMA_VERSION
from ..scenario_quality import SCENARIO_QUALITY_SCHEMA_VERSION
from ..hashing import sha256_file as _sha256
from .constants import EVAL_SUITE_MANIFEST_SCHEMA_VERSION, RUN_SUITE_SCHEMA_VERSION, SERVING_CAPABILITY_STATUSES, SERVING_COMPATIBILITY_REPORT_SCHEMA_VERSION, SERVING_DEMO_RUN_SCHEMA_VERSION, SERVING_ENDPOINT_CHECK_SCHEMA_VERSION, SERVING_LIFECYCLE_PREFLIGHT_ARTIFACTS, SERVING_LIFECYCLE_SCHEMA_VERSION, SERVING_PROFILE_SCHEMA_VERSION
from .primitives import ValidationTarget, _average_number, _count_rows, _count_strings, _expected_suite_trend_delta, _expected_suite_trend_summary, _is_int_between, _is_lowercase_sha256, _is_non_negative_int, _is_number_between, _is_optional_non_negative_int, _is_optional_non_negative_number, _is_optional_number, _is_optional_rate, _is_redacted_placeholder, _is_sha256, _is_string_list, _is_windows_absolute, _looks_absolute, _merge_count_rows, _non_negative_int_value, _number_value, _rate_value, _read_json_object_silent, _read_jsonl_objects, _read_object, _reject_symlinked_validation_path, _require_equal, _score_value, _sha256, _validate_allowed_keys, _validate_count_map_object, _validate_count_rows, _validate_gate_like_checks, _validate_metadata, _validate_metric_count_fields, _validate_metric_source, _warn_absolute_public_path
from .review import validate_review_calibration
from .runs import _validate_harness_suite_result, validate_run_dir

def validate_suite_summary(path: str | Path) -> ValidationTarget:
    """Validate one run-suite summary artifact."""
    summary_path = Path(path)
    target = ValidationTarget("suite_summary", str(summary_path))
    summary = _read_object(summary_path, target, "suite_summary.json")
    if summary is None:
        return target
    _validate_suite_summary(summary, target, summary_path)
    return target

def validate_suite_summary_payload_consistency(summary: dict[str, Any]) -> ValidationTarget:
    """Validate suite record and aggregate consistency without resolving source files."""
    target = ValidationTarget("suite_summary", "<in-memory>")
    _validate_suite_summary(summary, target, validate_sources=False)
    return target

def validate_eval_suite_manifest(path: str | Path) -> ValidationTarget:
    """Validate one eval suite manifest artifact."""
    manifest_path = Path(path)
    target = ValidationTarget("eval_suite_manifest", str(manifest_path))
    manifest = _read_object(manifest_path, target, "eval_suite_manifest.json")
    if manifest is not None:
        _validate_eval_suite_manifest(manifest, target)
    return target

def validate_eval_summary(path: str | Path) -> ValidationTarget:
    """Validate one governance-ready eval summary artifact."""
    summary_path = Path(path)
    target = ValidationTarget("eval_summary", str(summary_path))
    summary = _read_object(summary_path, target, "eval_summary.json")
    if summary is not None:
        _validate_eval_summary(summary, target, source_path=summary_path)
    return target

def validate_external_eval_plan(path: str | Path) -> ValidationTarget:
    """Validate one external eval adapter readiness plan."""
    plan_path = Path(path)
    target = ValidationTarget("external_eval_plan", str(plan_path))
    plan = _read_object(plan_path, target, "external_eval_plan.json")
    if plan is not None:
        _validate_external_eval_plan(plan, target, plan_path)
    return target

def validate_external_eval_receipt(path: str | Path) -> ValidationTarget:
    """Validate one external eval dry-run or blocked live receipt."""
    receipt_path = Path(path)
    target = ValidationTarget("external_eval_receipt", str(receipt_path))
    receipt = _read_object(receipt_path, target, "external_eval_receipt.json")
    if receipt is not None:
        _validate_external_eval_receipt(receipt, target, receipt_path)
    return target

def validate_external_eval_result(path: str | Path) -> ValidationTarget:
    """Validate and replay one import-only external evaluation result."""
    result_path = Path(path)
    target = ValidationTarget("external_eval_result", str(result_path))
    result = _read_object(result_path, target, "external_eval_result.json")
    if result is not None:
        _validate_external_eval_result(result, target, result_path)
    return target

def validate_agentic_rollout_receipt(path: str | Path) -> ValidationTarget:
    """Validate one deterministic mock rollout receipt."""
    receipt_path = Path(path)
    target = ValidationTarget("agentic_rollout_receipt", str(receipt_path))
    receipt = _read_object(receipt_path, target, "agentic_rollout_receipt.json")
    if receipt is not None:
        _validate_agentic_rollout_receipt(receipt, target, receipt_path)
    return target

def validate_rejection_sampling_gate(path: str | Path) -> ValidationTarget:
    """Validate one rejection sampling admission gate."""
    gate_path = Path(path)
    target = ValidationTarget("rejection_sampling_gate", str(gate_path))
    gate = _read_object(gate_path, target, "rejection_sampling_gate.json")
    if gate is not None:
        _validate_rejection_sampling_gate(gate, target, gate_path)
    return target

def validate_dataset_curation_receipt(path: str | Path) -> ValidationTarget:
    """Validate one dataset curation readiness receipt."""
    receipt_path = Path(path)
    target = ValidationTarget("dataset_curation_receipt", str(receipt_path))
    receipt = _read_object(receipt_path, target, "dataset_curation_receipt.json")
    if receipt is not None:
        _validate_dataset_curation_receipt(receipt, target, receipt_path)
    return target

def validate_heldout_manifest(path: str | Path) -> ValidationTarget:
    """Validate one held-out scenario manifest."""
    manifest_path = Path(path)
    target = ValidationTarget("heldout_manifest", str(manifest_path))
    manifest = _read_object(manifest_path, target, "heldout_manifest.json")
    if manifest is not None:
        _validate_heldout_manifest(manifest, target, manifest_path)
    return target

def validate_serving_profile(path: str | Path) -> ValidationTarget:
    """Validate one serving_profile.json artifact."""
    profile_path = Path(path)
    target = ValidationTarget("serving_profile", str(profile_path))
    profile = _read_object(profile_path, target, "serving_profile.json")
    if profile is not None:
        _validate_serving_profile(profile, target)
    return target

def validate_serving_compatibility_report(path: str | Path) -> ValidationTarget:
    """Validate one compatibility_report.json artifact."""
    report_path = Path(path)
    target = ValidationTarget("serving_compatibility_report", str(report_path))
    report = _read_object(report_path, target, "compatibility_report.json")
    if report is not None:
        _validate_serving_compatibility_report(report, target)
    return target

def validate_serving_endpoint_check(path: str | Path) -> ValidationTarget:
    """Validate one serving_check.json artifact."""
    check_path = Path(path)
    target = ValidationTarget("serving_endpoint_check", str(check_path))
    check = _read_object(check_path, target, "serving_check.json")
    if check is not None:
        _validate_serving_endpoint_check(check, target)
    return target

def validate_serving_lifecycle(path: str | Path) -> ValidationTarget:
    """Validate one managed serving_lifecycle.json artifact."""
    lifecycle_path = Path(path)
    target = ValidationTarget("serving_lifecycle", str(lifecycle_path))
    lifecycle = _read_object(lifecycle_path, target, "serving_lifecycle.json")
    if lifecycle is not None:
        _validate_serving_lifecycle(lifecycle, target, lifecycle_path)
    return target

def validate_serving_demo_run(path: str | Path) -> ValidationTarget:
    """Validate one serving demo run artifact."""
    demo_path = Path(path)
    target = ValidationTarget("serving_demo_run", str(demo_path))
    demo = _read_object(demo_path, target, "serving_demo_run.json")
    if demo is not None:
        _validate_serving_demo_run(demo, target)
    return target

def validate_evidence_coverage(path: str | Path) -> ValidationTarget:
    """Validate an evidence-coverage artifact."""
    coverage_path = Path(path)
    target = ValidationTarget("evidence_coverage", str(coverage_path))
    coverage = _read_object(coverage_path, target, "evidence_coverage.json")
    if coverage is not None:
        _validate_evidence_coverage(coverage, target)
    return target

def validate_agentic_rollout_plan(path: str | Path) -> ValidationTarget:
    """Validate an agentic rollout generation plan."""
    plan_path = Path(path)
    target = ValidationTarget("agentic_rollout_plan", str(plan_path))
    plan = _read_object(plan_path, target, "agentic_rollout_plan.json")
    if plan is not None:
        _validate_agentic_rollout_plan(plan, target, plan_path)
    return target

def validate_rubric_spec(path: str | Path) -> ValidationTarget:
    """Validate a rubric spec artifact."""
    rubric_path = Path(path)
    target = ValidationTarget("rubric_spec", str(rubric_path))
    if _reject_symlinked_validation_path(rubric_path, target, "rubric_spec.path", "file"):
        return target
    rubric = _read_object(rubric_path, target, "rubric_spec.json")
    if rubric is not None:
        _validate_rubric_spec(rubric, target, rubric_path)
    return target

def validate_model_grader_dry_run(path: str | Path) -> ValidationTarget:
    """Validate a dry-run model-grader receipt."""
    receipt_path = Path(path)
    target = ValidationTarget("model_grader_dry_run", str(receipt_path))
    if _reject_symlinked_validation_path(receipt_path, target, "model_grader_dry_run.path", "file"):
        return target
    receipt = _read_object(receipt_path, target, "model_grader_dry_run.json")
    if receipt is not None:
        _validate_model_grader_dry_run(receipt, target, receipt_path)
    return target

def validate_model_grader_disagreement_queue(path: str | Path) -> ValidationTarget:
    """Validate a portable human-review queue derived from a model-grader dry run."""
    queue_path = Path(path)
    target = ValidationTarget("model_grader_disagreement_queue", str(queue_path))
    if _reject_symlinked_validation_path(queue_path, target, "model_grader_disagreement_queue.path", "file"):
        return target
    queue = _read_object(queue_path, target, "model_grader_disagreement_queue.json")
    if queue is not None:
        _validate_model_grader_disagreement_queue_artifact(queue, target, queue_path)
    return target

def validate_model_grader_override_receipt(path: str | Path) -> ValidationTarget:
    """Validate a human override receipt for model-grader dry-run queue items."""
    receipt_path = Path(path)
    target = ValidationTarget("model_grader_override_receipt", str(receipt_path))
    if _reject_symlinked_validation_path(receipt_path, target, "model_grader_override_receipt.path", "file"):
        return target
    receipt = _read_object(receipt_path, target, "model_grader_override_receipt.json")
    if receipt is not None:
        _validate_model_grader_override_receipt(receipt, target, receipt_path)
    return target

def validate_model_grader_gate(path: str | Path) -> ValidationTarget:
    """Validate a model-grader training-admission gate."""
    gate_path = Path(path)
    target = ValidationTarget("model_grader_gate", str(gate_path))
    if _reject_symlinked_validation_path(gate_path, target, "model_grader_gate.path", "file"):
        return target
    gate = _read_object(gate_path, target, "model_grader_gate.json")
    if gate is not None:
        _validate_model_grader_gate(gate, target, gate_path)
    return target

def validate_harness_suite_result(path: str | Path) -> ValidationTarget:
    """Validate a harness_suite_result.json artifact."""
    result_path = Path(path)
    target = ValidationTarget("harness_suite_result", str(result_path))
    result = _read_object(result_path, target, "harness_suite_result.json")
    if result is not None:
        _validate_harness_suite_result(result, target, source_dir=result_path.parent)
    return target

def validate_scenario_check(path: str | Path) -> ValidationTarget:
    """Validate a scenario-check artifact."""
    check_path = Path(path)
    target = ValidationTarget("scenario_check", str(check_path))
    check = _read_object(check_path, target, "scenario_check.json")
    if check is not None:
        _validate_scenario_check(check, target)
    return target

def validate_scenario_quality(path: str | Path) -> ValidationTarget:
    """Validate a scenario-quality artifact."""
    quality_path = Path(path)
    target = ValidationTarget("scenario_quality", str(quality_path))
    quality = _read_object(quality_path, target, "scenario_quality.json")
    if quality is not None:
        _validate_scenario_quality(quality, target)
    return target

_AGENTIC_ROLLOUT_CHECK_KEYS = {"id", "passed", "actual", "expected", "summary"}

_AGENTIC_ROLLOUT_PLAN_KEYS = {
    "schema_version",
    "created_at",
    "iteration_id",
    "plan_path",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "blocked_reasons",
    "environment",
    "budget",
    "policies",
    "scenarios",
    "harness_batches",
    "rejection_sampling",
    "lineage",
    "execution_boundary",
    "notes",
}

_AGENTIC_ROLLOUT_PLAN_BUDGET_KEYS = {"max_rollouts", "planned_rollouts", "live_provider_calls_allowed"}

_AGENTIC_ROLLOUT_POLICY_KEYS = {"role", "id", "live_calls_allowed"}

_AGENTIC_ROLLOUT_SCENARIO_KEYS = {"id", "path", "exists", "sha256", "schema_version"}

_AGENTIC_ROLLOUT_BATCH_KEYS = {
    "batch_id",
    "scenario_id",
    "scenario_sha256",
    "policy_role",
    "policy_id",
    "harness_mode",
    "status",
}

_AGENTIC_ROLLOUT_REJECTION_SAMPLING_KEYS = {
    "enabled",
    "requires_scorecard",
    "requires_task_completion",
    "requires_review_calibration_before_training",
    "accepted_dataset_roles",
}

_AGENTIC_ROLLOUT_PLAN_LINEAGE_KEYS = {
    "dataset_rows_created",
    "expected_trace_artifacts",
    "preserve_source_hashes",
}

_AGENTIC_ROLLOUT_PLAN_BOUNDARY_KEYS = {
    "plan_only",
    "rollouts_started",
    "model_provider_calls_started",
    "paid_model_grader_calls_started",
    "dataset_rows_written",
}

_AGENTIC_ROLLOUT_ENVIRONMENT_KEYS = {
    "id",
    "replayable",
    "network_default",
    "external_state_verifiers",
    "external_state_verifier_gate",
}

_AGENTIC_ROLLOUT_VERIFIER_REF_KEYS = {"role", "path", "exists", "sha256", "size_bytes"}

_AGENTIC_ROLLOUT_VERIFIER_GATE_KEYS = {
    "declared_count",
    "resolved_count",
    "all_declared_verifiers_resolved",
    "required_for_external_state_checks",
    "verification_side_effects_started",
    "credential_values_recorded",
}

_AGENTIC_ROLLOUT_RECEIPT_KEYS = {
    "schema_version",
    "created_at",
    "receipt_path",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "blocked_reasons",
    "source_plan",
    "iteration_id",
    "environment",
    "mock_rollout_count",
    "mock_rollouts",
    "lineage",
    "execution_boundary",
    "notes",
}

_AGENTIC_ROLLOUT_RECEIPT_SOURCE_PLAN_KEYS = {
    "path",
    "exists",
    "sha256",
    "size_bytes",
    "schema_version",
    "passed",
    "readiness",
}

_AGENTIC_ROLLOUT_MOCK_ROW_KEYS = {
    "rollout_id",
    "batch_id",
    "scenario_id",
    "scenario_sha256",
    "policy_role",
    "policy_id",
    "harness_mode",
    "status",
    "model_provider_called",
    "trace_written",
    "scorecard_written",
    "dataset_row_written",
}

_AGENTIC_ROLLOUT_RECEIPT_LINEAGE_KEYS = {
    "dataset_rows_created",
    "trace_files_written",
    "scorecards_written",
    "ready_for_rejection_sampling",
}

_AGENTIC_ROLLOUT_RECEIPT_BOUNDARY_KEYS = {
    "mock_receipt_only",
    "mock_rollouts_recorded",
    "live_rollouts_started",
    "model_provider_calls_started",
    "paid_model_grader_calls_started",
    "dataset_rows_written",
}

def _validate_agentic_rollout_plan(plan: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(plan, _AGENTIC_ROLLOUT_PLAN_KEYS, target, "agentic_rollout_plan")
    _require_equal(plan, "schema_version", AGENTIC_ROLLOUT_PLAN_SCHEMA_VERSION, target, prefix="agentic_rollout_plan.")
    checks = plan.get("checks")
    if not isinstance(checks, list):
        target.errors.append("agentic_rollout_plan.checks must be a list.")
        checks = []
    failed_checks = _validate_gate_like_checks(checks, target, "agentic_rollout_plan.checks")
    for index, check in enumerate(checks):
        if isinstance(check, dict):
            _validate_allowed_keys(check, _AGENTIC_ROLLOUT_CHECK_KEYS, target, f"agentic_rollout_plan.checks[{index}]")
    if plan.get("check_count") != len(checks):
        target.errors.append(f"agentic_rollout_plan.check_count expected {len(checks)}, got {plan.get('check_count')!r}.")
    if plan.get("failed_check_count") != failed_checks:
        target.errors.append(f"agentic_rollout_plan.failed_check_count expected {failed_checks}, got {plan.get('failed_check_count')!r}.")
    if plan.get("passed") != (failed_checks == 0):
        target.errors.append("agentic_rollout_plan.passed must match failed_check_count.")
    expected_readiness = "ready_for_harness_batch" if failed_checks == 0 else "blocked"
    if plan.get("readiness") != expected_readiness:
        target.errors.append(f"agentic_rollout_plan.readiness expected {expected_readiness!r}, got {plan.get('readiness')!r}.")
    expected_recommendation = "run_mock_or_opted_in_harness_batch" if failed_checks == 0 else "fix_rollout_plan_inputs"
    if plan.get("recommendation") != expected_recommendation:
        target.errors.append(f"agentic_rollout_plan.recommendation expected {expected_recommendation!r}, got {plan.get('recommendation')!r}.")
    if not _is_string_list(plan.get("blocked_reasons")):
        target.errors.append("agentic_rollout_plan.blocked_reasons must be a list of strings.")
    if not isinstance(plan.get("plan_path"), str) or not plan.get("plan_path"):
        target.errors.append("agentic_rollout_plan.plan_path must be a non-empty string.")
    elif not _is_safe_or_redacted_agentic_rollout_ref_path(plan.get("plan_path")):
        target.errors.append("agentic_rollout_plan.plan_path must be a safe relative path or redacted placeholder.")
    budget = plan.get("budget") if isinstance(plan.get("budget"), dict) else {}
    if isinstance(plan.get("budget"), dict):
        _validate_allowed_keys(budget, _AGENTIC_ROLLOUT_PLAN_BUDGET_KEYS, target, "agentic_rollout_plan.budget")
    else:
        target.errors.append("agentic_rollout_plan.budget must be an object.")
    batches = plan.get("harness_batches") if isinstance(plan.get("harness_batches"), list) else []
    if not isinstance(plan.get("harness_batches"), list):
        target.errors.append("agentic_rollout_plan.harness_batches must be a list.")
    if not _is_non_negative_int(budget.get("max_rollouts")) or budget.get("max_rollouts") <= 0:
        target.errors.append("agentic_rollout_plan.budget.max_rollouts must be positive.")
    if budget.get("planned_rollouts") != len(batches):
        target.errors.append(f"agentic_rollout_plan.budget.planned_rollouts expected {len(batches)}, got {budget.get('planned_rollouts')!r}.")
    if budget.get("live_provider_calls_allowed") is not False:
        target.errors.append("agentic_rollout_plan.budget.live_provider_calls_allowed must be false.")
    _validate_agentic_rollout_policies(plan.get("policies"), target)
    _validate_agentic_rollout_scenarios(plan.get("scenarios"), target, source_path)
    for index, batch in enumerate(batches):
        _validate_agentic_rollout_batch(batch, index, target)
    _validate_agentic_rollout_rejection_sampling(plan.get("rejection_sampling"), target)
    lineage = plan.get("lineage")
    if not isinstance(lineage, dict):
        target.errors.append("agentic_rollout_plan.lineage must be an object.")
    else:
        _validate_allowed_keys(lineage, _AGENTIC_ROLLOUT_PLAN_LINEAGE_KEYS, target, "agentic_rollout_plan.lineage")
        if lineage.get("dataset_rows_created") is not False:
            target.errors.append("agentic_rollout_plan.lineage.dataset_rows_created must be false.")
        if not _is_string_list(lineage.get("expected_trace_artifacts")):
            target.errors.append("agentic_rollout_plan.lineage.expected_trace_artifacts must be a list of strings.")
        if lineage.get("preserve_source_hashes") is not True:
            target.errors.append("agentic_rollout_plan.lineage.preserve_source_hashes must be true.")
    _validate_agentic_rollout_environment(plan.get("environment"), target, "agentic_rollout_plan.environment", source_path)
    boundary = plan.get("execution_boundary")
    if not isinstance(boundary, dict):
        target.errors.append("agentic_rollout_plan.execution_boundary must be an object.")
    else:
        _validate_allowed_keys(boundary, _AGENTIC_ROLLOUT_PLAN_BOUNDARY_KEYS, target, "agentic_rollout_plan.execution_boundary")
        for field_name in ("plan_only",):
            if boundary.get(field_name) is not True:
                target.errors.append(f"agentic_rollout_plan.execution_boundary.{field_name} must be true.")
        for field_name in ("rollouts_started", "model_provider_calls_started", "paid_model_grader_calls_started", "dataset_rows_written"):
            if boundary.get(field_name) is not False:
                target.errors.append(f"agentic_rollout_plan.execution_boundary.{field_name} must be false.")
    if not _is_string_list(plan.get("notes")):
        target.errors.append("agentic_rollout_plan.notes must be a list of strings.")
    target.details.update(
        {
            "iteration_id": plan.get("iteration_id"),
            "planned_rollouts": budget.get("planned_rollouts"),
            "readiness": plan.get("readiness"),
        }
    )

def _validate_agentic_rollout_receipt(receipt: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(receipt, _AGENTIC_ROLLOUT_RECEIPT_KEYS, target, "agentic_rollout_receipt")
    _require_equal(receipt, "schema_version", AGENTIC_ROLLOUT_RECEIPT_SCHEMA_VERSION, target, prefix="agentic_rollout_receipt.")
    checks = receipt.get("checks")
    if not isinstance(checks, list):
        target.errors.append("agentic_rollout_receipt.checks must be a list.")
        checks = []
    failed_checks = _validate_gate_like_checks(checks, target, "agentic_rollout_receipt.checks")
    for index, check in enumerate(checks):
        if isinstance(check, dict):
            _validate_allowed_keys(check, _AGENTIC_ROLLOUT_CHECK_KEYS, target, f"agentic_rollout_receipt.checks[{index}]")
    if receipt.get("check_count") != len(checks):
        target.errors.append(f"agentic_rollout_receipt.check_count expected {len(checks)}, got {receipt.get('check_count')!r}.")
    if receipt.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"agentic_rollout_receipt.failed_check_count expected {failed_checks}, got {receipt.get('failed_check_count')!r}."
        )
    if receipt.get("passed") != (failed_checks == 0):
        target.errors.append("agentic_rollout_receipt.passed must match failed_check_count.")
    expected_readiness = "mock_rollouts_recorded" if failed_checks == 0 else "blocked"
    if receipt.get("readiness") != expected_readiness:
        target.errors.append(f"agentic_rollout_receipt.readiness expected {expected_readiness!r}, got {receipt.get('readiness')!r}.")
    expected_recommendation = "score_and_review_mock_rollouts" if failed_checks == 0 else "fix_rollout_plan_before_mock_execution"
    if receipt.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"agentic_rollout_receipt.recommendation expected {expected_recommendation!r}, got {receipt.get('recommendation')!r}."
        )
    if not _is_string_list(receipt.get("blocked_reasons")):
        target.errors.append("agentic_rollout_receipt.blocked_reasons must be a list of strings.")

    source_plan = receipt.get("source_plan")
    if not isinstance(source_plan, dict):
        target.errors.append("agentic_rollout_receipt.source_plan must be an object.")
    else:
        _validate_agentic_rollout_receipt_source_plan(source_plan, target, source_path)

    _validate_agentic_rollout_environment(receipt.get("environment"), target, "agentic_rollout_receipt.environment", source_path)

    rollouts = receipt.get("mock_rollouts")
    if not isinstance(rollouts, list):
        target.errors.append("agentic_rollout_receipt.mock_rollouts must be a list.")
        rollouts = []
    if receipt.get("mock_rollout_count") != len(rollouts):
        target.errors.append(
            f"agentic_rollout_receipt.mock_rollout_count expected {len(rollouts)}, got {receipt.get('mock_rollout_count')!r}."
        )
    for index, row in enumerate(rollouts):
        _validate_agentic_mock_rollout(row, index, target)

    lineage = receipt.get("lineage")
    if not isinstance(lineage, dict):
        target.errors.append("agentic_rollout_receipt.lineage must be an object.")
    else:
        _validate_allowed_keys(lineage, _AGENTIC_ROLLOUT_RECEIPT_LINEAGE_KEYS, target, "agentic_rollout_receipt.lineage")
        for field_name in ("dataset_rows_created", "trace_files_written", "scorecards_written"):
            if lineage.get(field_name) is not False:
                target.errors.append(f"agentic_rollout_receipt.lineage.{field_name} must be false.")
        if lineage.get("ready_for_rejection_sampling") != receipt.get("passed"):
            target.errors.append("agentic_rollout_receipt.lineage.ready_for_rejection_sampling must match passed.")

    boundary = receipt.get("execution_boundary")
    if not isinstance(boundary, dict):
        target.errors.append("agentic_rollout_receipt.execution_boundary must be an object.")
    else:
        _validate_allowed_keys(boundary, _AGENTIC_ROLLOUT_RECEIPT_BOUNDARY_KEYS, target, "agentic_rollout_receipt.execution_boundary")
        if boundary.get("mock_receipt_only") is not True:
            target.errors.append("agentic_rollout_receipt.execution_boundary.mock_receipt_only must be true.")
        if boundary.get("mock_rollouts_recorded") != bool(rollouts):
            target.errors.append("agentic_rollout_receipt.execution_boundary.mock_rollouts_recorded must match mock_rollouts presence.")
        for field_name in (
            "live_rollouts_started",
            "model_provider_calls_started",
            "paid_model_grader_calls_started",
            "dataset_rows_written",
        ):
            if boundary.get(field_name) is not False:
                target.errors.append(f"agentic_rollout_receipt.execution_boundary.{field_name} must be false.")
    if not _is_string_list(receipt.get("notes")):
        target.errors.append("agentic_rollout_receipt.notes must be a list of strings.")

    target.details.update(
        {
            "passed": receipt.get("passed"),
            "readiness": receipt.get("readiness"),
            "mock_rollout_count": len(rollouts),
        }
    )

def _validate_agentic_rollout_environment(value: Any, target: ValidationTarget, label: str, source_path: Path) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_ROLLOUT_ENVIRONMENT_KEYS, target, label)
    if not isinstance(value.get("id"), str) or not value.get("id"):
        target.errors.append(f"{label}.id must be a non-empty string.")
    if value.get("replayable") is not True:
        target.errors.append(f"{label}.replayable must be true.")
    if value.get("network_default") != "disabled":
        target.errors.append(f"{label}.network_default must be disabled.")
    verifier_refs = value.get("external_state_verifiers")
    if not isinstance(verifier_refs, list):
        target.errors.append(f"{label}.external_state_verifiers must be a list.")
        verifier_refs = []
    resolved_count = 0
    for index, ref in enumerate(verifier_refs):
        ref_label = f"{label}.external_state_verifiers[{index}]"
        if not isinstance(ref, dict):
            target.errors.append(f"{ref_label} must be an object.")
            continue
        _validate_allowed_keys(ref, _AGENTIC_ROLLOUT_VERIFIER_REF_KEYS, target, ref_label)
        if ref.get("role") != "verifier_config":
            target.errors.append(f"{ref_label}.role must be verifier_config.")
        if not isinstance(ref.get("path"), str) or not ref.get("path"):
            target.errors.append(f"{ref_label}.path must be a non-empty string.")
        elif not _is_safe_or_redacted_agentic_rollout_ref_path(ref.get("path")):
            target.errors.append(f"{ref_label}.path must be relative to the rollout artifact.")
        if not isinstance(ref.get("exists"), bool):
            target.errors.append(f"{ref_label}.exists must be a boolean.")
        elif ref.get("exists") is True:
            resolved_count += 1
            path_value = ref.get("path")
            if not isinstance(path_value, str) or not _is_replayable_agentic_rollout_ref_path(path_value):
                target.errors.append(f"{ref_label}.path must be relative to the rollout artifact when exists is true.")
                continue
            verifier_path = _agentic_rollout_reference_path(path_value, source_path)
            if _path_has_symlink_component(verifier_path, include_leaf=True):
                target.errors.append(f"{ref_label}.path must resolve to a regular non-symlink verifier config file.")
                continue
            if not verifier_path.is_file():
                target.errors.append(f"{ref_label}.path does not resolve to a verifier config file.")
                continue
            if not _is_sha256(ref.get("sha256")):
                target.errors.append(f"{ref_label}.sha256 must be a SHA-256 hex string when exists is true.")
            elif _sha256(verifier_path) != ref.get("sha256"):
                target.errors.append(f"{ref_label}.sha256 does not match the current file.")
            if not _is_non_negative_int(ref.get("size_bytes")):
                target.errors.append(f"{ref_label}.size_bytes must be a non-negative integer when exists is true.")
            elif verifier_path.stat().st_size != ref.get("size_bytes"):
                target.errors.append(f"{ref_label}.size_bytes does not match the current file.")
        else:
            if ref.get("sha256") is not None:
                target.errors.append(f"{ref_label}.sha256 must be null when exists is false.")
            if ref.get("size_bytes") is not None:
                target.errors.append(f"{ref_label}.size_bytes must be null when exists is false.")
    gate = value.get("external_state_verifier_gate")
    if not isinstance(gate, dict):
        target.errors.append(f"{label}.external_state_verifier_gate must be an object.")
        return
    _validate_allowed_keys(gate, _AGENTIC_ROLLOUT_VERIFIER_GATE_KEYS, target, f"{label}.external_state_verifier_gate")
    if gate.get("declared_count") != len(verifier_refs):
        target.errors.append(f"{label}.external_state_verifier_gate.declared_count must match external_state_verifiers.")
    if gate.get("resolved_count") != resolved_count:
        target.errors.append(f"{label}.external_state_verifier_gate.resolved_count must match existing verifier refs.")
    if gate.get("all_declared_verifiers_resolved") != (resolved_count == len(verifier_refs)):
        target.errors.append(f"{label}.external_state_verifier_gate.all_declared_verifiers_resolved must match verifier refs.")
    if gate.get("required_for_external_state_checks") != bool(verifier_refs):
        target.errors.append(f"{label}.external_state_verifier_gate.required_for_external_state_checks must match verifier refs.")
    for field_name in ("verification_side_effects_started", "credential_values_recorded"):
        if gate.get(field_name) is not False:
            target.errors.append(f"{label}.external_state_verifier_gate.{field_name} must be false.")

def _validate_agentic_rollout_receipt_source_plan(source_plan: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    label = "agentic_rollout_receipt.source_plan"
    _validate_allowed_keys(source_plan, _AGENTIC_ROLLOUT_RECEIPT_SOURCE_PLAN_KEYS, target, label)
    if source_plan.get("schema_version") != AGENTIC_ROLLOUT_PLAN_SCHEMA_VERSION:
        target.errors.append(f"{label}.schema_version must be {AGENTIC_ROLLOUT_PLAN_SCHEMA_VERSION!r}.")
    if source_plan.get("exists") is not True:
        if source_plan.get("sha256") is not None:
            target.errors.append(f"{label}.sha256 must be null when exists is false.")
        if source_plan.get("size_bytes") is not None:
            target.errors.append(f"{label}.size_bytes must be null when exists is false.")
        if source_plan.get("passed") is not None:
            target.errors.append(f"{label}.passed must be null when exists is false.")
        if source_plan.get("readiness") not in ("", None):
            target.errors.append(f"{label}.readiness must be empty when exists is false.")
        return
    path_value = source_plan.get("path")
    if not isinstance(path_value, str) or not path_value:
        target.errors.append(f"{label}.path must be a non-empty string when exists is true.")
        return
    if not _is_replayable_agentic_rollout_ref_path(path_value):
        target.errors.append(f"{label}.path must be relative to the agentic rollout receipt.")
        return
    plan_path = _agentic_rollout_reference_path(path_value, source_path)
    if _path_has_symlink_component(plan_path, include_leaf=True):
        target.errors.append(f"{label}.path must resolve to a regular non-symlink agentic rollout plan file.")
        return
    if not plan_path.is_file():
        target.errors.append(f"{label}.path does not resolve to an agentic rollout plan file.")
        return
    if not _is_non_negative_int(source_plan.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer when exists is true.")
    elif plan_path.stat().st_size != source_plan.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if not _is_sha256(source_plan.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when exists is true.")
    elif _sha256(plan_path) != source_plan.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current file.")
    plan = _read_object(plan_path, target, f"{label}.path")
    if plan is not None:
        if plan.get("schema_version") != source_plan.get("schema_version"):
            target.errors.append(f"{label}.schema_version must match the current file.")
        if plan.get("passed") != source_plan.get("passed"):
            target.errors.append(f"{label}.passed must match the current file.")
        if plan.get("readiness") != source_plan.get("readiness"):
            target.errors.append(f"{label}.readiness must match the current file.")

def _validate_agentic_mock_rollout(row: Any, index: int, target: ValidationTarget) -> None:
    label = f"agentic_rollout_receipt.mock_rollouts[{index}]"
    if not isinstance(row, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(row, _AGENTIC_ROLLOUT_MOCK_ROW_KEYS, target, label)
    for field_name in ("rollout_id", "batch_id", "scenario_id", "policy_role", "policy_id"):
        if not isinstance(row.get(field_name), str) or not row.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if row.get("scenario_sha256") is not None and not _is_sha256(row.get("scenario_sha256")):
        target.errors.append(f"{label}.scenario_sha256 must be a SHA-256 hex string or null.")
    if row.get("harness_mode") != "offline_mock":
        target.errors.append(f"{label}.harness_mode must be 'offline_mock'.")
    if row.get("status") != "mock_recorded":
        target.errors.append(f"{label}.status must be 'mock_recorded'.")
    for field_name in ("model_provider_called", "trace_written", "scorecard_written", "dataset_row_written"):
        if row.get(field_name) is not False:
            target.errors.append(f"{label}.{field_name} must be false.")

def _validate_agentic_rollout_policies(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, list) or not value:
        target.errors.append("agentic_rollout_plan.policies must be a non-empty list.")
        return
    for index, row in enumerate(value):
        label = f"agentic_rollout_plan.policies[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(row, _AGENTIC_ROLLOUT_POLICY_KEYS, target, label)
        if row.get("role") not in {"baseline", "candidate", "teacher"}:
            target.errors.append(f"{label}.role must be baseline, candidate, or teacher.")
        if not isinstance(row.get("id"), str) or not row.get("id"):
            target.errors.append(f"{label}.id must be a non-empty string.")
        if row.get("live_calls_allowed") is not False:
            target.errors.append(f"{label}.live_calls_allowed must be false.")

def _validate_agentic_rollout_scenarios(value: Any, target: ValidationTarget, source_path: Path) -> None:
    if not isinstance(value, list) or not value:
        target.errors.append("agentic_rollout_plan.scenarios must be a non-empty list.")
        return
    for index, row in enumerate(value):
        label = f"agentic_rollout_plan.scenarios[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(row, _AGENTIC_ROLLOUT_SCENARIO_KEYS, target, label)
        for field_name in ("id", "path", "schema_version"):
            if not isinstance(row.get(field_name), str):
                target.errors.append(f"{label}.{field_name} must be a string.")
        path_value = row.get("path")
        if isinstance(path_value, str) and path_value and not _is_safe_or_redacted_agentic_rollout_ref_path(path_value):
            target.errors.append(f"{label}.path must be relative to the agentic rollout plan.")
        if not isinstance(row.get("exists"), bool):
            target.errors.append(f"{label}.exists must be a boolean.")
        elif row.get("exists") is True:
            if not isinstance(path_value, str) or not _is_replayable_agentic_rollout_ref_path(path_value):
                target.errors.append(f"{label}.path must be relative to the agentic rollout plan when exists is true.")
                continue
            scenario_path = _agentic_rollout_reference_path(path_value, source_path)
            if _path_has_symlink_component(scenario_path, include_leaf=True):
                target.errors.append(f"{label}.path must resolve to a regular non-symlink scenario file.")
                continue
            if not scenario_path.is_file():
                target.errors.append(f"{label}.path does not resolve to a scenario file.")
                continue
            if not _is_sha256(row.get("sha256")):
                target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when exists is true.")
            elif _sha256(scenario_path) != row.get("sha256"):
                target.errors.append(f"{label}.sha256 does not match the current file.")
        elif row.get("exists") is False and row.get("sha256") is not None:
            target.errors.append(f"{label}.sha256 must be null when exists is false.")

def _validate_agentic_rollout_batch(row: Any, index: int, target: ValidationTarget) -> None:
    label = f"agentic_rollout_plan.harness_batches[{index}]"
    if not isinstance(row, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(row, _AGENTIC_ROLLOUT_BATCH_KEYS, target, label)
    for field_name in ("batch_id", "scenario_id", "policy_role", "policy_id"):
        if not isinstance(row.get(field_name), str) or not row.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if row.get("scenario_sha256") is not None and not _is_sha256(row.get("scenario_sha256")):
        target.errors.append(f"{label}.scenario_sha256 must be a SHA-256 hex string or null.")
    if row.get("policy_role") not in {"baseline", "candidate", "teacher"}:
        target.errors.append(f"{label}.policy_role must be baseline, candidate, or teacher.")
    if row.get("harness_mode") != "offline_mock":
        target.errors.append(f"{label}.harness_mode must be offline_mock.")
    if row.get("status") != "planned":
        target.errors.append(f"{label}.status must be planned.")

def _is_safe_or_redacted_agentic_rollout_ref_path(value: str) -> bool:
    if value.startswith("<redacted:") and value.endswith(">"):
        basename = value.removeprefix("<redacted:").removesuffix(">")
        return bool(basename) and "/" not in basename and "\\" not in basename and ".." not in basename and "~" not in basename
    return _is_replayable_agentic_rollout_ref_path(value)

def _is_replayable_agentic_rollout_ref_path(value: str) -> bool:
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

def _agentic_rollout_reference_path(value: str, source_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return source_path.parent / path

def _validate_agentic_rollout_rejection_sampling(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("agentic_rollout_plan.rejection_sampling must be an object.")
        return
    _validate_allowed_keys(value, _AGENTIC_ROLLOUT_REJECTION_SAMPLING_KEYS, target, "agentic_rollout_plan.rejection_sampling")
    for field_name in ("enabled", "requires_scorecard", "requires_task_completion", "requires_review_calibration_before_training"):
        if value.get(field_name) is not True:
            target.errors.append(f"agentic_rollout_plan.rejection_sampling.{field_name} must be true.")
    if not _is_string_list(value.get("accepted_dataset_roles")):
        target.errors.append("agentic_rollout_plan.rejection_sampling.accepted_dataset_roles must be a list of strings.")

_REJECTION_DATASET_CHECK_KEYS = {"id", "passed", "actual", "expected", "summary"}

_REJECTION_SAMPLING_GATE_KEYS = {
    "schema_version",
    "created_at",
    "gate_path",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "blocked_reasons",
    "input_artifacts",
    "rollout_summary",
    "admission_policy",
    "execution_boundary",
    "notes",
}

_REJECTION_SAMPLING_GATE_INPUT_KEYS = {
    "agentic_rollout_receipt",
    "model_grader_gate",
    "review_calibration",
    "reviewed_gate",
}

_REJECTION_SAMPLING_GATE_REF_KEYS = {
    "role",
    "path",
    "kind",
    "exists",
    "sha256",
    "size_bytes",
    "schema_version",
    "passed",
    "readiness",
}

_REJECTION_SAMPLING_GATE_ROLLOUT_REF_KEYS = _REJECTION_SAMPLING_GATE_REF_KEYS | {
    "mock_rollout_count",
    "mock_receipt_only",
    "live_rollouts_started",
    "dataset_rows_written",
}

_REJECTION_SAMPLING_GATE_SUMMARY_KEYS = {
    "receipt_count",
    "mock_rollout_count",
    "live_rollouts_started",
    "dataset_rows_created",
}

_REJECTION_SAMPLING_GATE_POLICY_KEYS = {
    "requires_mock_rollout_receipt",
    "requires_calibrated_review",
    "requires_model_grader_gate",
    "requires_reviewed_gate",
    "accepts_uncalibrated_labels",
    "accepted_dataset_roles",
}

_REJECTION_SAMPLING_GATE_BOUNDARY_KEYS = {
    "gate_only",
    "dataset_rows_written",
    "model_provider_calls_started",
    "paid_model_grader_calls_started",
    "weights_updated_by_flight_recorder",
}

_DATASET_CURATION_RECEIPT_KEYS = {
    "schema_version",
    "created_at",
    "receipt_path",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "blocked_reasons",
    "input_artifacts",
    "curation_summary",
    "trainer_handoff",
    "execution_boundary",
    "notes",
}

_DATASET_CURATION_INPUT_KEYS = {"rejection_sampling_gate", "training_export"}

_DATASET_CURATION_REF_KEYS = {
    "role",
    "path",
    "kind",
    "exists",
    "sha256",
    "size_bytes",
    "schema_version",
    "passed",
    "readiness",
}

_DATASET_CURATION_DIRECTORY_REF_KEYS = _DATASET_CURATION_REF_KEYS | {
    "manifest_path",
    "manifest_exists",
    "manifest_sha256",
    "manifest_size_bytes",
}

_DATASET_CURATION_SUMMARY_KEYS = {
    "rejection_sampling_gate_count",
    "training_export_count",
    "curated_rows_written",
    "accepted_rows_written",
    "rejected_rows_written",
    "dataset_registry_updated",
}

_DATASET_CURATION_TRAINER_HANDOFF_KEYS = {
    "dataset_rows_source",
    "allowed_dataset_roles",
    "requires_rejection_sampling_gate",
    "requires_training_gate_before_live_training",
    "requires_trainer_preflight",
}

_DATASET_CURATION_BOUNDARY_KEYS = {
    "receipt_only",
    "dataset_rows_written",
    "dataset_registry_updated",
    "cloud_jobs_started",
    "weights_updated_by_flight_recorder",
}

def _validate_rejection_sampling_gate(gate: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(gate, _REJECTION_SAMPLING_GATE_KEYS, target, "rejection_sampling_gate")
    _require_equal(gate, "schema_version", REJECTION_SAMPLING_GATE_SCHEMA_VERSION, target, prefix="rejection_sampling_gate.")
    if not isinstance(gate.get("gate_path"), str):
        target.errors.append("rejection_sampling_gate.gate_path must be a string.")
    elif gate.get("gate_path") and not _is_public_rejection_sampling_ref_path(gate.get("gate_path")):
        target.errors.append("rejection_sampling_gate.gate_path must be a safe relative path or redacted placeholder.")
    checks = gate.get("checks")
    if not isinstance(checks, list):
        target.errors.append("rejection_sampling_gate.checks must be a list.")
        checks = []
    failed_checks = _validate_gate_like_checks(checks, target, "rejection_sampling_gate.checks")
    for index, check in enumerate(checks):
        if isinstance(check, dict):
            _validate_allowed_keys(check, _REJECTION_DATASET_CHECK_KEYS, target, f"rejection_sampling_gate.checks[{index}]")
    if gate.get("check_count") != len(checks):
        target.errors.append(f"rejection_sampling_gate.check_count expected {len(checks)}, got {gate.get('check_count')!r}.")
    if gate.get("failed_check_count") != failed_checks:
        target.errors.append(f"rejection_sampling_gate.failed_check_count expected {failed_checks}, got {gate.get('failed_check_count')!r}.")
    if gate.get("passed") != (failed_checks == 0):
        target.errors.append("rejection_sampling_gate.passed must match failed_check_count.")
    expected_readiness = "ready_for_dataset_curation" if failed_checks == 0 else "blocked"
    if gate.get("readiness") != expected_readiness:
        target.errors.append(f"rejection_sampling_gate.readiness expected {expected_readiness!r}, got {gate.get('readiness')!r}.")
    expected_recommendation = "curate_accepted_training_rows" if failed_checks == 0 else "collect_calibrated_reviews_before_sampling"
    if gate.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"rejection_sampling_gate.recommendation expected {expected_recommendation!r}, got {gate.get('recommendation')!r}."
        )
    if not _is_string_list(gate.get("blocked_reasons")):
        target.errors.append("rejection_sampling_gate.blocked_reasons must be a list of strings.")

    artifacts = gate.get("input_artifacts")
    if not isinstance(artifacts, dict):
        target.errors.append("rejection_sampling_gate.input_artifacts must be an object.")
        artifacts = {}
    else:
        _validate_allowed_keys(artifacts, _REJECTION_SAMPLING_GATE_INPUT_KEYS, target, "rejection_sampling_gate.input_artifacts")
    required_roles = {
        "agentic_rollout_receipt": "hfr.agentic_rollout_receipt.v1",
        "model_grader_gate": "hfr.model_grader_gate.v1",
        "review_calibration": "hfr.review_calibration.v1",
        "reviewed_gate": "hfr.reviewed_gate.v1",
    }
    for role, schema_version in required_roles.items():
        rows = artifacts.get(role)
        if not isinstance(rows, list) or not rows:
            target.errors.append(f"rejection_sampling_gate.input_artifacts.{role} must contain at least one artifact ref.")
            continue
        for index, row in enumerate(rows):
            _validate_rejection_sampling_gate_ref(
                row,
                target,
                f"rejection_sampling_gate.input_artifacts.{role}[{index}]",
                role,
                schema_version,
                source_path,
            )
    _validate_rejection_sampling_review_lineage(artifacts, checks, target, source_path)

    summary = gate.get("rollout_summary")
    if not isinstance(summary, dict):
        target.errors.append("rejection_sampling_gate.rollout_summary must be an object.")
    else:
        _validate_allowed_keys(summary, _REJECTION_SAMPLING_GATE_SUMMARY_KEYS, target, "rejection_sampling_gate.rollout_summary")
        if not _is_non_negative_int(summary.get("receipt_count")):
            target.errors.append("rejection_sampling_gate.rollout_summary.receipt_count must be a non-negative integer.")
        if not _is_non_negative_int(summary.get("mock_rollout_count")):
            target.errors.append("rejection_sampling_gate.rollout_summary.mock_rollout_count must be a non-negative integer.")
        if summary.get("live_rollouts_started") is not False:
            target.errors.append("rejection_sampling_gate.rollout_summary.live_rollouts_started must be false.")
        if summary.get("dataset_rows_created") is not False:
            target.errors.append("rejection_sampling_gate.rollout_summary.dataset_rows_created must be false.")

    policy = gate.get("admission_policy")
    if not isinstance(policy, dict):
        target.errors.append("rejection_sampling_gate.admission_policy must be an object.")
    else:
        _validate_allowed_keys(policy, _REJECTION_SAMPLING_GATE_POLICY_KEYS, target, "rejection_sampling_gate.admission_policy")
        for field_name in ("requires_mock_rollout_receipt", "requires_calibrated_review", "requires_model_grader_gate", "requires_reviewed_gate"):
            if policy.get(field_name) is not True:
                target.errors.append(f"rejection_sampling_gate.admission_policy.{field_name} must be true.")
        if policy.get("accepts_uncalibrated_labels") is not False:
            target.errors.append("rejection_sampling_gate.admission_policy.accepts_uncalibrated_labels must be false.")
        if not _is_string_list(policy.get("accepted_dataset_roles")):
            target.errors.append("rejection_sampling_gate.admission_policy.accepted_dataset_roles must be a list of strings.")

    boundary = gate.get("execution_boundary")
    if not isinstance(boundary, dict):
        target.errors.append("rejection_sampling_gate.execution_boundary must be an object.")
    else:
        _validate_allowed_keys(boundary, _REJECTION_SAMPLING_GATE_BOUNDARY_KEYS, target, "rejection_sampling_gate.execution_boundary")
        if boundary.get("gate_only") is not True:
            target.errors.append("rejection_sampling_gate.execution_boundary.gate_only must be true.")
        for field_name in (
            "dataset_rows_written",
            "model_provider_calls_started",
            "paid_model_grader_calls_started",
            "weights_updated_by_flight_recorder",
        ):
            if boundary.get(field_name) is not False:
                target.errors.append(f"rejection_sampling_gate.execution_boundary.{field_name} must be false.")
    if not _is_string_list(gate.get("notes")):
        target.errors.append("rejection_sampling_gate.notes must be a list of strings.")

    target.details.update(
        {
            "passed": gate.get("passed"),
            "readiness": gate.get("readiness"),
            "mock_rollout_count": summary.get("mock_rollout_count") if isinstance(summary, dict) else None,
        }
    )

def _validate_rejection_sampling_gate_ref(
    row: Any,
    target: ValidationTarget,
    label: str,
    role: str,
    schema_version: str,
    source_path: Path,
) -> None:
    if not isinstance(row, dict):
        target.errors.append(f"{label} must be an object.")
        return
    allowed = _REJECTION_SAMPLING_GATE_ROLLOUT_REF_KEYS if role == "agentic_rollout_receipt" else _REJECTION_SAMPLING_GATE_REF_KEYS
    _validate_allowed_keys(row, allowed, target, label)
    for field_name in ("role", "path", "kind", "schema_version", "readiness"):
        if not isinstance(row.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if isinstance(row.get("path"), str) and row.get("path"):
        if row["path"].startswith("<redacted:"):
            if row.get("exists") is True:
                target.errors.append(f"{label}.path cannot be redacted when exists is true.")
        elif not _is_public_rejection_sampling_ref_path(row["path"]):
            target.errors.append(f"{label}.path must be a safe relative path or redacted placeholder.")
    if row.get("role") != role:
        target.errors.append(f"{label}.role must be {role!r}.")
    if row.get("kind") not in {"file", "directory"}:
        target.errors.append(f"{label}.kind must be 'file' or 'directory'.")
    if row.get("exists") is not True:
        target.errors.append(f"{label}.exists must be true.")
    if row.get("schema_version") != schema_version:
        target.errors.append(f"{label}.schema_version must be {schema_version!r}.")
    if row.get("passed") is not True:
        target.errors.append(f"{label}.passed must be true.")
    if row.get("kind") == "file":
        if not _is_sha256(row.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string for file refs.")
        if not _is_non_negative_int(row.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for file refs.")
        _validate_rejection_sampling_gate_file_ref(row, target, label, source_path)
    if row.get("role") == "agentic_rollout_receipt":
        if row.get("readiness") != "mock_rollouts_recorded":
            target.errors.append(f"{label}.readiness must be 'mock_rollouts_recorded'.")
        if not _is_non_negative_int(row.get("mock_rollout_count")) or row.get("mock_rollout_count") <= 0:
            target.errors.append(f"{label}.mock_rollout_count must be positive.")
        if row.get("mock_receipt_only") is not True:
            target.errors.append(f"{label}.mock_receipt_only must be true.")
        if row.get("live_rollouts_started") is not False:
            target.errors.append(f"{label}.live_rollouts_started must be false.")
        if row.get("dataset_rows_written") is not False:
            target.errors.append(f"{label}.dataset_rows_written must be false.")

def _validate_rejection_sampling_gate_file_ref(
    row: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path,
) -> None:
    if row.get("exists") is not True:
        return
    file_path = _resolve_rejection_sampling_gate_ref_path(row.get("path"), source_path)
    if file_path is None:
        return
    if _path_has_symlink_component(file_path, include_leaf=True):
        target.errors.append(f"{label}.path must resolve to a regular non-symlink file.")
        return
    if not file_path.exists() or not file_path.is_file():
        target.errors.append(f"{label}.path must resolve to an existing file.")
        return
    if _is_non_negative_int(row.get("size_bytes")) and file_path.stat().st_size != row.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if _is_sha256(row.get("sha256")) and _sha256(file_path) != row.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current file.")
    role = row.get("role")
    if isinstance(role, str) and role:
        inspection = inspect_artifact_source(file_path, role)
        if inspection.get("ready") is not True:
            target.errors.append(f"{label}.path does not reference a semantically ready {role} artifact.")

def _validate_rejection_sampling_review_lineage(
    artifacts: dict[str, Any],
    checks: list[Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    resolved: dict[str, list[Path]] = {}
    for role in ("model_grader_gate", "review_calibration", "reviewed_gate"):
        paths: list[Path] = []
        rows = artifacts.get(role)
        for row in rows if isinstance(rows, list) else []:
            path = (
                _resolve_rejection_sampling_gate_ref_path(row.get("path"), source_path)
                if isinstance(row, dict)
                else None
            )
            if path is not None:
                paths.append(path)
        resolved[role] = paths
    passed, actual = _review_lineage_status(
        resolved["model_grader_gate"],
        resolved["review_calibration"],
        resolved["reviewed_gate"],
    )
    expected = {
        "one_reviewed_dataset_version": True,
        "model_grader_uses_supplied_calibration": True,
    }
    expected_check = {
        "id": "review_dataset_lineage_converges",
        "passed": passed,
        "actual": actual,
        "expected": expected,
        "summary": f"review_dataset_lineage_converges: passed={passed}",
    }
    matching_checks = [
        check
        for check in checks
        if isinstance(check, dict) and check.get("id") == "review_dataset_lineage_converges"
    ]
    if matching_checks != [expected_check]:
        target.errors.append(
            "rejection_sampling_gate review_dataset_lineage_converges check must match current review inputs exactly."
        )
    if not passed:
        target.errors.append(
            "rejection_sampling_gate review inputs must converge on one reviewed dataset and calibration artifact."
        )

def _resolve_rejection_sampling_gate_ref_path(value: Any, source_path: Path) -> Path | None:
    if not isinstance(value, str) or value.startswith("<redacted:") or not _is_public_rejection_sampling_ref_path(value):
        return None
    return source_path.parent / value

def _is_public_rejection_sampling_ref_path(value: str) -> bool:
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return (
        bool(value)
        and (value.startswith("<redacted:") or not path.is_absolute())
        and not windows_path.is_absolute()
        and not windows_path.drive
        and "\\" not in value
        and ".." not in path.parts
        and all(not part.startswith("~") for part in path.parts)
    )

def _validate_dataset_curation_receipt(receipt: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(receipt, _DATASET_CURATION_RECEIPT_KEYS, target, "dataset_curation_receipt")
    _require_equal(receipt, "schema_version", DATASET_CURATION_RECEIPT_SCHEMA_VERSION, target, prefix="dataset_curation_receipt.")
    checks = receipt.get("checks")
    if not isinstance(checks, list):
        target.errors.append("dataset_curation_receipt.checks must be a list.")
        checks = []
    failed_checks = _validate_gate_like_checks(checks, target, "dataset_curation_receipt.checks")
    for index, check in enumerate(checks):
        if isinstance(check, dict):
            _validate_allowed_keys(check, _REJECTION_DATASET_CHECK_KEYS, target, f"dataset_curation_receipt.checks[{index}]")
    if receipt.get("check_count") != len(checks):
        target.errors.append(f"dataset_curation_receipt.check_count expected {len(checks)}, got {receipt.get('check_count')!r}.")
    if receipt.get("failed_check_count") != failed_checks:
        target.errors.append(f"dataset_curation_receipt.failed_check_count expected {failed_checks}, got {receipt.get('failed_check_count')!r}.")
    if receipt.get("passed") != (failed_checks == 0):
        target.errors.append("dataset_curation_receipt.passed must match failed_check_count.")
    expected_readiness = "ready_for_external_trainer_handoff" if failed_checks == 0 else "blocked"
    if receipt.get("readiness") != expected_readiness:
        target.errors.append(f"dataset_curation_receipt.readiness expected {expected_readiness!r}, got {receipt.get('readiness')!r}.")
    expected_recommendation = "run_training_gate_and_trainer_preflight" if failed_checks == 0 else "fix_rejection_sampling_or_training_exports"
    if receipt.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"dataset_curation_receipt.recommendation expected {expected_recommendation!r}, got {receipt.get('recommendation')!r}."
        )
    if not _is_string_list(receipt.get("blocked_reasons")):
        target.errors.append("dataset_curation_receipt.blocked_reasons must be a list of strings.")
    if not isinstance(receipt.get("receipt_path"), str):
        target.errors.append("dataset_curation_receipt.receipt_path must be a string.")
    elif receipt.get("receipt_path") and not _is_public_dataset_curation_ref_path(receipt.get("receipt_path")):
        target.errors.append("dataset_curation_receipt.receipt_path must be a safe relative path or redacted placeholder.")

    artifacts = receipt.get("input_artifacts")
    if not isinstance(artifacts, dict):
        target.errors.append("dataset_curation_receipt.input_artifacts must be an object.")
        artifacts = {}
    else:
        _validate_allowed_keys(artifacts, _DATASET_CURATION_INPUT_KEYS, target, "dataset_curation_receipt.input_artifacts")
    gate_rows = artifacts.get("rejection_sampling_gate")
    if not isinstance(gate_rows, list) or not gate_rows:
        target.errors.append("dataset_curation_receipt.input_artifacts.rejection_sampling_gate must contain at least one ref.")
    else:
        for index, row in enumerate(gate_rows):
            _validate_dataset_curation_ref(
                row,
                target,
                f"dataset_curation_receipt.input_artifacts.rejection_sampling_gate[{index}]",
                "rejection_sampling_gate",
                "hfr.rejection_sampling_gate.v1",
                "ready_for_dataset_curation",
                source_path,
            )
    export_rows = artifacts.get("training_export")
    if not isinstance(export_rows, list) or not export_rows:
        target.errors.append("dataset_curation_receipt.input_artifacts.training_export must contain at least one ref.")
    else:
        for index, row in enumerate(export_rows):
            _validate_dataset_curation_ref(
                row,
                target,
                f"dataset_curation_receipt.input_artifacts.training_export[{index}]",
                "training_export",
                "",
                "",
                source_path,
            )
    _validate_dataset_curation_lineage(
        gate_rows,
        export_rows,
        checks,
        target,
        source_path,
    )

    summary = receipt.get("curation_summary")
    if not isinstance(summary, dict):
        target.errors.append("dataset_curation_receipt.curation_summary must be an object.")
    else:
        _validate_allowed_keys(summary, _DATASET_CURATION_SUMMARY_KEYS, target, "dataset_curation_receipt.curation_summary")
        for field_name in ("rejection_sampling_gate_count", "training_export_count"):
            if not _is_non_negative_int(summary.get(field_name)):
                target.errors.append(f"dataset_curation_receipt.curation_summary.{field_name} must be a non-negative integer.")
        for field_name in ("curated_rows_written", "accepted_rows_written", "rejected_rows_written"):
            if summary.get(field_name) != 0:
                target.errors.append(f"dataset_curation_receipt.curation_summary.{field_name} must be 0.")
        if summary.get("dataset_registry_updated") is not False:
            target.errors.append("dataset_curation_receipt.curation_summary.dataset_registry_updated must be false.")

    handoff = receipt.get("trainer_handoff")
    if not isinstance(handoff, dict):
        target.errors.append("dataset_curation_receipt.trainer_handoff must be an object.")
    else:
        _validate_allowed_keys(handoff, _DATASET_CURATION_TRAINER_HANDOFF_KEYS, target, "dataset_curation_receipt.trainer_handoff")
        if handoff.get("dataset_rows_source") != "existing_training_exports":
            target.errors.append("dataset_curation_receipt.trainer_handoff.dataset_rows_source must be existing_training_exports.")
        if not _is_string_list(handoff.get("allowed_dataset_roles")):
            target.errors.append("dataset_curation_receipt.trainer_handoff.allowed_dataset_roles must be a list of strings.")
        for field_name in ("requires_rejection_sampling_gate", "requires_training_gate_before_live_training", "requires_trainer_preflight"):
            if handoff.get(field_name) is not True:
                target.errors.append(f"dataset_curation_receipt.trainer_handoff.{field_name} must be true.")

    boundary = receipt.get("execution_boundary")
    if not isinstance(boundary, dict):
        target.errors.append("dataset_curation_receipt.execution_boundary must be an object.")
    else:
        _validate_allowed_keys(boundary, _DATASET_CURATION_BOUNDARY_KEYS, target, "dataset_curation_receipt.execution_boundary")
        if boundary.get("receipt_only") is not True:
            target.errors.append("dataset_curation_receipt.execution_boundary.receipt_only must be true.")
        for field_name in ("dataset_rows_written", "dataset_registry_updated", "cloud_jobs_started", "weights_updated_by_flight_recorder"):
            if boundary.get(field_name) is not False:
                target.errors.append(f"dataset_curation_receipt.execution_boundary.{field_name} must be false.")
    if not _is_string_list(receipt.get("notes")):
        target.errors.append("dataset_curation_receipt.notes must be a list of strings.")

    target.details.update(
        {
            "passed": receipt.get("passed"),
            "readiness": receipt.get("readiness"),
            "training_export_count": summary.get("training_export_count") if isinstance(summary, dict) else None,
        }
    )

def _validate_dataset_curation_lineage(
    gate_rows: Any,
    export_rows: Any,
    checks: list[Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    gate_paths: list[Path] = []
    for row in gate_rows if isinstance(gate_rows, list) else []:
        path = (
            _resolve_dataset_curation_ref_path(row.get("path"), source_path)
            if isinstance(row, dict)
            else None
        )
        if path is not None:
            gate_paths.append(path)
    export_paths: list[Path] = []
    for row in export_rows if isinstance(export_rows, list) else []:
        path = (
            _resolve_dataset_curation_ref_path(row.get("path"), source_path)
            if isinstance(row, dict)
            else None
        )
        if path is not None:
            export_paths.append(path)
    passed, actual = training_export_lineage_status(gate_paths, export_paths)
    expected = {
        "training_exports_all_complete": True,
        "reviewed_item_missing_count": 0,
        "reviewed_item_mismatch_count": 0,
        "reviewed_label_mismatch_count": 0,
        "rollout_scenario_missing_count": 0,
        "rollout_scenario_mismatch_count": 0,
    }
    expected_check = {
        "id": "training_exports_cover_admitted_lineage",
        "passed": passed,
        "actual": actual,
        "expected": expected,
        "summary": f"training_exports_cover_admitted_lineage: passed={passed}",
    }
    matching = [
        check
        for check in checks
        if isinstance(check, dict)
        and check.get("id") == "training_exports_cover_admitted_lineage"
    ]
    if matching != [expected_check]:
        target.errors.append(
            "dataset_curation_receipt training_exports_cover_admitted_lineage "
            "check must match current rejection and training-export inputs exactly."
        )
    if not passed:
        target.errors.append(
            "dataset_curation_receipt training exports must cover every reviewed "
            "item and rollout scenario admitted by rejection sampling."
        )

def _validate_dataset_curation_ref(
    row: Any,
    target: ValidationTarget,
    label: str,
    role: str,
    schema_version: str,
    readiness: str,
    source_path: Path,
) -> None:
    if not isinstance(row, dict):
        target.errors.append(f"{label} must be an object.")
        return
    allowed = _DATASET_CURATION_DIRECTORY_REF_KEYS if row.get("kind") == "directory" else _DATASET_CURATION_REF_KEYS
    _validate_allowed_keys(row, allowed, target, label)
    for field_name in ("role", "path", "kind"):
        if not isinstance(row.get(field_name), str) or not row.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if isinstance(row.get("path"), str) and row.get("path"):
        if row["path"].startswith("<redacted:"):
            if row.get("exists") is True:
                target.errors.append(f"{label}.path cannot be redacted when exists is true.")
        elif not _is_public_dataset_curation_ref_path(row["path"]):
            target.errors.append(f"{label}.path must be a safe relative path or redacted placeholder.")
    if row.get("role") != role:
        target.errors.append(f"{label}.role must be {role!r}.")
    if row.get("kind") not in {"file", "directory"}:
        target.errors.append(f"{label}.kind must be 'file' or 'directory'.")
    if row.get("exists") is not True:
        target.errors.append(f"{label}.exists must be true.")
    if schema_version:
        if row.get("schema_version") != schema_version:
            target.errors.append(f"{label}.schema_version must be {schema_version!r}.")
        if row.get("passed") is not True:
            target.errors.append(f"{label}.passed must be true.")
    if readiness and row.get("readiness") != readiness:
        target.errors.append(f"{label}.readiness must be {readiness!r}.")
    if row.get("kind") == "file":
        if not _is_sha256(row.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string for file refs.")
        if not _is_non_negative_int(row.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for file refs.")
        _validate_dataset_curation_file_ref(row, target, label, source_path, role)
    if row.get("kind") == "directory":
        if not isinstance(row.get("manifest_path"), str) or not row.get("manifest_path"):
            target.errors.append(f"{label}.manifest_path must be a non-empty string for directory refs.")
        elif isinstance(row.get("manifest_path"), str):
            if row["manifest_path"].startswith("<redacted:"):
                if row.get("manifest_exists") is True:
                    target.errors.append(f"{label}.manifest_path cannot be redacted when manifest_exists is true.")
            elif not _is_public_dataset_curation_ref_path(row["manifest_path"]):
                target.errors.append(f"{label}.manifest_path must be a safe relative path or redacted placeholder.")
        if row.get("manifest_exists") is not True:
            target.errors.append(f"{label}.manifest_exists must be true for directory refs.")
        if not _is_sha256(row.get("manifest_sha256")):
            target.errors.append(f"{label}.manifest_sha256 must be a SHA-256 hex string for directory refs.")
        if not _is_non_negative_int(row.get("manifest_size_bytes")):
            target.errors.append(f"{label}.manifest_size_bytes must be a non-negative integer for directory refs.")
        _validate_dataset_curation_directory_ref(row, target, label, source_path, role)

def _validate_dataset_curation_file_ref(
    row: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path,
    role: str,
) -> None:
    if row.get("exists") is not True:
        return
    file_path = _resolve_dataset_curation_ref_path(row.get("path"), source_path)
    if file_path is None:
        return
    if _path_has_symlink_component(file_path, include_leaf=True):
        target.errors.append(f"{label}.path must resolve to a regular non-symlink file.")
        return
    if not file_path.exists() or not file_path.is_file():
        target.errors.append(f"{label}.path must resolve to an existing file.")
        return
    if _is_non_negative_int(row.get("size_bytes")) and file_path.stat().st_size != row.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if _is_sha256(row.get("sha256")) and _sha256(file_path) != row.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current file.")
    source = inspect_artifact_source(file_path, role)
    if source.get("ready") is not True:
        target.errors.append(f"{label}.path must remain semantically ready for role {role!r}.")

def _validate_dataset_curation_directory_ref(
    row: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path,
    role: str,
) -> None:
    if row.get("exists") is not True:
        return
    directory_path = _resolve_dataset_curation_ref_path(row.get("path"), source_path)
    if directory_path is not None:
        if _path_has_symlink_component(directory_path, include_leaf=True):
            target.errors.append(f"{label}.path must resolve to a regular non-symlink directory.")
            return
        if not directory_path.exists() or not directory_path.is_dir():
            target.errors.append(f"{label}.path must resolve to an existing directory.")
            return
    if row.get("manifest_exists") is not True:
        return
    manifest_path = _resolve_dataset_curation_ref_path(row.get("manifest_path"), source_path)
    if manifest_path is None:
        return
    if _path_has_symlink_component(manifest_path, include_leaf=True):
        target.errors.append(f"{label}.manifest_path must resolve to a regular non-symlink file.")
        return
    if not manifest_path.exists() or not manifest_path.is_file():
        target.errors.append(f"{label}.manifest_path must resolve to an existing file.")
        return
    if _is_non_negative_int(row.get("manifest_size_bytes")) and manifest_path.stat().st_size != row.get("manifest_size_bytes"):
        target.errors.append(f"{label}.manifest_size_bytes does not match the current file.")
    if _is_sha256(row.get("manifest_sha256")) and _sha256(manifest_path) != row.get("manifest_sha256"):
        target.errors.append(f"{label}.manifest_sha256 does not match the current file.")
    if directory_path is not None:
        source = inspect_artifact_source(directory_path, role)
        if source.get("ready") is not True:
            target.errors.append(f"{label}.path must remain semantically ready for role {role!r}.")

def _resolve_dataset_curation_ref_path(value: Any, source_path: Path) -> Path | None:
    if not isinstance(value, str) or value.startswith("<redacted:") or not _is_public_dataset_curation_ref_path(value):
        return None
    return source_path.parent / value

def _is_public_dataset_curation_ref_path(value: str) -> bool:
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return (
        bool(value)
        and (value.startswith("<redacted:") or not path.is_absolute())
        and not windows_path.is_absolute()
        and not windows_path.drive
        and "\\" not in value
        and ".." not in path.parts
        and all(not part.startswith("~") for part in path.parts)
    )

def _validate_rubric_spec(rubric: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(rubric, _MODEL_GRADER_RUBRIC_SPEC_KEYS, target, "rubric_spec")
    _require_equal(rubric, "schema_version", RUBRIC_SPEC_SCHEMA_VERSION, target, prefix="rubric_spec.")
    review_export_paths = _validate_model_grader_review_export_ref(
        rubric.get("review_export"),
        target,
        "rubric_spec.review_export",
        source_path,
    )
    if not isinstance(rubric.get("rubric_id"), str) or not rubric.get("rubric_id"):
        target.errors.append("rubric_spec.rubric_id must be a non-empty string.")
    criteria = rubric.get("criteria")
    if not isinstance(criteria, list):
        target.errors.append("rubric_spec.criteria must be a list.")
        criteria = []
    for index, row in enumerate(criteria):
        label = f"rubric_spec.criteria[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(row, _MODEL_GRADER_RUBRIC_CRITERION_KEYS, target, label)
        if not isinstance(row.get("criterion_id"), str) or not row.get("criterion_id"):
            target.errors.append(f"{label}.criterion_id must be a non-empty string.")
        if not isinstance(row.get("description"), str) or not row.get("description"):
            target.errors.append(f"{label}.description must be a non-empty string.")
        if row.get("required") is not True:
            target.errors.append(f"{label}.required must be true.")
    if rubric.get("criterion_count") != len(criteria):
        target.errors.append(f"rubric_spec.criterion_count expected {len(criteria)}, got {rubric.get('criterion_count')!r}.")
    labels = rubric.get("label_options")
    if labels != list(REVIEW_LABELS):
        target.errors.append(f"rubric_spec.label_options must be {list(REVIEW_LABELS)!r}.")
    fingerprints = rubric.get("review_item_fingerprints")
    if not isinstance(fingerprints, list):
        target.errors.append("rubric_spec.review_item_fingerprints must be a list.")
        fingerprints = []
    if rubric.get("review_item_count") != len(fingerprints):
        target.errors.append(
            f"rubric_spec.review_item_count expected {len(fingerprints)}, got {rubric.get('review_item_count')!r}."
        )
    for index, row in enumerate(fingerprints):
        label = f"rubric_spec.review_item_fingerprints[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(row, _MODEL_GRADER_RUBRIC_FINGERPRINT_KEYS, target, label)
        for field_name in ("review_item_id", "episode_id", "scenario_id"):
            if not isinstance(row.get(field_name), str) or not row.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if not _is_sha256(row.get("review_item_sha256")):
            target.errors.append(f"{label}.review_item_sha256 must be a sha256 hex digest.")
    _validate_rubric_review_item_fingerprints(
        fingerprints,
        review_export_paths.get("review_items"),
        target,
    )
    requirements = rubric.get("calibration_requirements") if isinstance(rubric.get("calibration_requirements"), dict) else {}
    if isinstance(rubric.get("calibration_requirements"), dict):
        _validate_allowed_keys(
            requirements,
            _MODEL_GRADER_RUBRIC_CALIBRATION_REQUIREMENT_KEYS,
            target,
            "rubric_spec.calibration_requirements",
        )
    if requirements.get("required_before_training_admission") is not True:
        target.errors.append("rubric_spec.calibration_requirements.required_before_training_admission must be true.")
    if not isinstance(requirements.get("min_calibration_agreement_rate"), (int, float)) or not 0 <= requirements.get("min_calibration_agreement_rate") <= 1:
        target.errors.append("rubric_spec.calibration_requirements.min_calibration_agreement_rate must be a number from 0 to 1.")
    if requirements.get("max_uncalibrated_labels_admitted") != 0:
        target.errors.append("rubric_spec.calibration_requirements.max_uncalibrated_labels_admitted must be 0.")
    if requirements.get("human_override_queue_required") is not True:
        target.errors.append("rubric_spec.calibration_requirements.human_override_queue_required must be true.")
    _validate_model_grader_boundary(rubric.get("execution_boundary"), target, "rubric_spec")
    target.details.update(
        {
            "rubric_id": rubric.get("rubric_id"),
            "criterion_count": rubric.get("criterion_count"),
            "review_item_count": rubric.get("review_item_count"),
        }
    )

_MODEL_GRADER_GATE_KEYS = {
    "schema_version",
    "created_at",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "blocked_reasons",
    "source_artifacts",
    "admission",
    "metrics",
    "execution_boundary",
    "notes",
}

_MODEL_GRADER_DRY_RUN_KEYS = {
    "schema_version",
    "created_at",
    "grader",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "blocked_reasons",
    "source_artifacts",
    "graded_item_count",
    "label_counts",
    "grader_labels",
    "disagreement_queue",
    "human_review_overrides",
    "training_admission",
    "execution_boundary",
    "notes",
}

_MODEL_GRADER_DISAGREEMENT_QUEUE_KEYS = {
    "schema_version",
    "created_at",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "blocked_reasons",
    "source_artifacts",
    "queue_count",
    "required_review_item_ids",
    "queue",
    "override_requirements",
    "training_admission",
    "execution_boundary",
    "notes",
}

_MODEL_GRADER_RUBRIC_SPEC_KEYS = {
    "schema_version",
    "rubric_id",
    "created_at",
    "review_export",
    "criterion_count",
    "criteria",
    "label_options",
    "calibration_requirements",
    "review_item_count",
    "review_item_fingerprints",
    "execution_boundary",
    "notes",
}

_MODEL_GRADER_OVERRIDE_RECEIPT_KEYS = {
    "schema_version",
    "created_at",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "blocked_reasons",
    "source_artifacts",
    "queue",
    "overrides",
    "metrics",
    "training_admission",
    "execution_boundary",
    "notes",
}

_MODEL_GRADER_DRY_RUN_SOURCE_KEYS = {"review_export", "rubric_spec"}

_MODEL_GRADER_DISAGREEMENT_QUEUE_SOURCE_KEYS = {"dry_run_receipt"}

_MODEL_GRADER_OVERRIDE_SOURCE_KEYS = {"dry_run_receipt", "override_rows"}

_MODEL_GRADER_REVIEW_EXPORT_KEYS = {"path", "manifest", "review_items"}

_MODEL_GRADER_GRADER_KEYS = {
    "provider",
    "grader_id",
    "mode",
    "transport",
    "provider_api_called",
    "paid_model_grader_calls_started",
}

_MODEL_GRADER_GATE_SOURCE_KEYS = {
    "dry_run_receipt",
    "rubric_spec",
    "review_calibration",
    "model_grader_override_receipt",
}

_MODEL_GRADER_CHECK_KEYS = {"id", "passed", "actual", "expected", "summary"}

_MODEL_GRADER_RUBRIC_CRITERION_KEYS = {"criterion_id", "description", "required"}

_MODEL_GRADER_RUBRIC_FINGERPRINT_KEYS = {
    "review_item_id",
    "episode_id",
    "scenario_id",
    "review_item_sha256",
}

_MODEL_GRADER_RUBRIC_CALIBRATION_REQUIREMENT_KEYS = {
    "required_before_training_admission",
    "min_calibration_agreement_rate",
    "max_uncalibrated_labels_admitted",
    "human_override_queue_required",
}

_MODEL_GRADER_LABEL_KEYS = {
    "review_item_id",
    "episode_id",
    "scenario_id",
    "task_family",
    "review_item_sha256",
    "mock_model_label",
    "mock_confidence",
    "requires_human_review",
    "rationale",
    "grader_id",
    "provider",
    "label_sha256",
}

_MODEL_GRADER_LABEL_COUNT_KEYS = {"label", "count"}

_MODEL_GRADER_DISAGREEMENT_KEYS = {
    "review_item_id",
    "episode_id",
    "scenario_id",
    "task_family",
    "review_item_sha256",
    "mock_model_label",
    "reason",
    "source_report",
}

_MODEL_GRADER_OVERRIDE_QUEUE_KEYS = {
    "dry_run_disagreement_queue_count",
    "required_review_item_ids",
    "resolved_review_item_ids",
    "unresolved_review_item_ids",
}

_MODEL_GRADER_OVERRIDE_ROW_KEYS = {
    "index",
    "review_item_id",
    "review_item_sha256",
    "human_label",
    "reviewer_confidence",
    "reviewer",
    "reviewed_at",
    "notes",
    "resolves_queue_item",
    "accepted",
    "errors",
    "override_sha256",
}

_MODEL_GRADER_OVERRIDE_METRICS_KEYS = {
    "override_row_count",
    "resolved_queue_count",
    "unresolved_queue_count",
    "unmatched_override_count",
    "invalid_override_count",
}

_MODEL_GRADER_DISAGREEMENT_QUEUE_OVERRIDE_REQUIREMENT_KEYS = {
    "required_before_training_admission",
    "required_override_count",
    "final_label_options",
    "minimum_reviewer_confidence",
}

_MODEL_GRADER_DRY_RUN_HUMAN_REVIEW_OVERRIDES_KEYS = {
    "required_before_training_admission",
    "override_queue_path",
    "override_count",
    "accepted_override_count",
}

_MODEL_GRADER_DRY_RUN_TRAINING_ADMISSION_KEYS = {
    "labels_allowed_for_training",
    "labels_admitted_count",
    "requires_calibrated_gate",
}

_MODEL_GRADER_OVERRIDE_TRAINING_ADMISSION_KEYS = {
    "labels_allowed_for_training",
    "labels_admitted_count",
    "requires_model_grader_gate",
}

_MODEL_GRADER_FILE_REF_KEYS = {
    "role",
    "path",
    "exists",
    "sha256",
    "size_bytes",
    "schema_name",
    "schema_passed",
    "schema_error_count",
    "schema_errors",
    "source_passed",
    "source_recommendation",
}

_MODEL_GRADER_GATE_ADMISSION_KEYS = {
    "labels_allowed_for_training",
    "labels_admitted_count",
    "uncalibrated_labels_admitted",
    "human_override_required_for_disagreements",
}

_MODEL_GRADER_GATE_METRICS_KEYS = {
    "graded_item_count",
    "agreement_rate",
    "calibration_disagreement_count",
    "dry_run_disagreement_queue_count",
    "dry_run_labels_requiring_human_review_count",
    "human_override_receipt_present",
    "human_override_resolved_count",
    "human_override_unresolved_count",
}

_MODEL_GRADER_BOUNDARY_KEYS = {
    "dry_run_only",
    "provider_api_called",
    "paid_model_grader_calls_started",
    "cloud_cost_incurred_usd",
    "credential_values_recorded",
    "labels_admitted_to_training",
    "weights_updated_by_flight_recorder",
}

def _validate_model_grader_dry_run(receipt: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(receipt, _MODEL_GRADER_DRY_RUN_KEYS, target, "model_grader_dry_run")
    _require_equal(receipt, "schema_version", MODEL_GRADER_DRY_RUN_SCHEMA_VERSION, target, prefix="model_grader_dry_run.")
    _validate_model_grader_checked_artifact(receipt, target, "model_grader_dry_run")
    _validate_model_grader_check_keys(receipt.get("checks"), target, "model_grader_dry_run.checks")
    source_paths = _validate_model_grader_dry_run_sources(receipt.get("source_artifacts"), target, source_path)
    grader = receipt.get("grader") if isinstance(receipt.get("grader"), dict) else {}
    if isinstance(receipt.get("grader"), dict):
        _validate_allowed_keys(grader, _MODEL_GRADER_GRADER_KEYS, target, "model_grader_dry_run.grader")
    if grader.get("mode") != "dry_run":
        target.errors.append("model_grader_dry_run.grader.mode must be dry_run.")
    if grader.get("transport") != "mock":
        target.errors.append("model_grader_dry_run.grader.transport must be mock.")
    if grader.get("provider_api_called") is not False:
        target.errors.append("model_grader_dry_run.grader.provider_api_called must be false.")
    if grader.get("paid_model_grader_calls_started") is not False:
        target.errors.append("model_grader_dry_run.grader.paid_model_grader_calls_started must be false.")
    labels = receipt.get("grader_labels")
    if not isinstance(labels, list):
        target.errors.append("model_grader_dry_run.grader_labels must be a list.")
        labels = []
    if receipt.get("graded_item_count") != len(labels):
        target.errors.append(
            f"model_grader_dry_run.graded_item_count expected {len(labels)}, got {receipt.get('graded_item_count')!r}."
        )
    labels_requiring_review = 0
    expected_disagreement_queue: list[dict[str, str]] = []
    observed_label_source_keys: list[dict[str, Any]] = []
    label_counts = _model_grader_label_counts(receipt.get("label_counts"), target, "model_grader_dry_run.label_counts")
    expected_counts = {label: 0 for label in REVIEW_LABELS}
    for index, row in enumerate(labels):
        label = f"model_grader_dry_run.grader_labels[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(row, _MODEL_GRADER_LABEL_KEYS, target, label)
        for field_name in ("review_item_id", "episode_id", "scenario_id", "task_family", "review_item_sha256", "mock_model_label", "mock_confidence", "rationale", "grader_id", "provider", "label_sha256"):
            if not isinstance(row.get(field_name), str) or not row.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if row.get("mock_model_label") not in REVIEW_LABELS:
            target.errors.append(f"{label}.mock_model_label must be one of {list(REVIEW_LABELS)!r}.")
        elif row["mock_model_label"] in expected_counts:
            expected_counts[row["mock_model_label"]] += 1
        if row.get("mock_confidence") not in REVIEW_CONFIDENCE_LEVELS:
            target.errors.append(f"{label}.mock_confidence must be one of {list(REVIEW_CONFIDENCE_LEVELS)!r}.")
        if not isinstance(row.get("requires_human_review"), bool):
            target.errors.append(f"{label}.requires_human_review must be a boolean.")
        elif row.get("requires_human_review") is True:
            labels_requiring_review += 1
            expected_disagreement_queue.append(_model_grader_disagreement_queue_key(row))
        if not _is_sha256(row.get("review_item_sha256")):
            target.errors.append(f"{label}.review_item_sha256 must be a sha256 hex digest.")
        label_hash = row.get("label_sha256")
        if not _is_sha256(label_hash):
            target.errors.append(f"{label}.label_sha256 must be a sha256 hex digest.")
        elif label_hash != _model_grader_label_sha256(row):
            target.errors.append(f"{label}.label_sha256 does not match label contents.")
        observed_label_source_keys.append(_model_grader_label_source_key(row))
    _validate_model_grader_dry_run_labels_match_review_items(
        observed_label_source_keys,
        source_paths.get("review_items"),
        grader,
        target,
    )
    for label_name, expected in expected_counts.items():
        if label_counts.get(label_name) != expected:
            target.errors.append(
                f"model_grader_dry_run.label_counts[{label_name}] expected {expected}, got {label_counts.get(label_name)!r}."
            )
    disagreement_queue = receipt.get("disagreement_queue")
    if not isinstance(disagreement_queue, list):
        target.errors.append("model_grader_dry_run.disagreement_queue must be a list.")
        disagreement_queue = []
    observed_disagreement_queue = _validate_model_grader_disagreement_queue(disagreement_queue, target)
    if len(disagreement_queue) != labels_requiring_review:
        target.errors.append(
            "model_grader_dry_run.disagreement_queue must contain one item for each label requiring human review."
        )
    if observed_disagreement_queue != expected_disagreement_queue:
        target.errors.append("model_grader_dry_run.disagreement_queue must match labels requiring human review.")
    admission = receipt.get("training_admission") if isinstance(receipt.get("training_admission"), dict) else {}
    if isinstance(receipt.get("training_admission"), dict):
        _validate_allowed_keys(admission, _MODEL_GRADER_DRY_RUN_TRAINING_ADMISSION_KEYS, target, "model_grader_dry_run.training_admission")
    if admission.get("labels_allowed_for_training") is not False:
        target.errors.append("model_grader_dry_run.training_admission.labels_allowed_for_training must be false.")
    if admission.get("labels_admitted_count") != 0:
        target.errors.append("model_grader_dry_run.training_admission.labels_admitted_count must be 0.")
    if admission.get("requires_calibrated_gate") is not True:
        target.errors.append("model_grader_dry_run.training_admission.requires_calibrated_gate must be true.")
    overrides = receipt.get("human_review_overrides")
    if not isinstance(overrides, dict):
        target.errors.append("model_grader_dry_run.human_review_overrides must be an object.")
    else:
        _validate_allowed_keys(
            overrides,
            _MODEL_GRADER_DRY_RUN_HUMAN_REVIEW_OVERRIDES_KEYS,
            target,
            "model_grader_dry_run.human_review_overrides",
        )
        if overrides.get("required_before_training_admission") is not True:
            target.errors.append("model_grader_dry_run.human_review_overrides.required_before_training_admission must be true.")
        if overrides.get("override_queue_path") is not None:
            target.errors.append("model_grader_dry_run.human_review_overrides.override_queue_path must be null.")
        if overrides.get("override_count") != 0:
            target.errors.append("model_grader_dry_run.human_review_overrides.override_count must be 0.")
        if overrides.get("accepted_override_count") != 0:
            target.errors.append("model_grader_dry_run.human_review_overrides.accepted_override_count must be 0.")
    _validate_model_grader_boundary(receipt.get("execution_boundary"), target, "model_grader_dry_run")
    target.details.update(
        {
            "graded_item_count": receipt.get("graded_item_count"),
            "readiness": receipt.get("readiness"),
            "recommendation": receipt.get("recommendation"),
        }
    )

def _validate_model_grader_disagreement_queue_artifact(queue: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(queue, _MODEL_GRADER_DISAGREEMENT_QUEUE_KEYS, target, "model_grader_disagreement_queue")
    _require_equal(
        queue,
        "schema_version",
        MODEL_GRADER_DISAGREEMENT_QUEUE_SCHEMA_VERSION,
        target,
        prefix="model_grader_disagreement_queue.",
    )
    failed_checks = _validate_model_grader_checked_artifact(queue, target, "model_grader_disagreement_queue")
    _validate_model_grader_check_keys(queue.get("checks"), target, "model_grader_disagreement_queue.checks")
    dry_run_path = _validate_model_grader_disagreement_queue_sources(queue.get("source_artifacts"), target, source_path)
    rows = queue.get("queue")
    if not isinstance(rows, list):
        target.errors.append("model_grader_disagreement_queue.queue must be a list.")
        rows = []
    observed_queue = _validate_model_grader_disagreement_queue(rows, target, "model_grader_disagreement_queue.queue")
    if queue.get("queue_count") != len(rows):
        target.errors.append(f"model_grader_disagreement_queue.queue_count expected {len(rows)}, got {queue.get('queue_count')!r}.")
    required_ids = queue.get("required_review_item_ids")
    if not _is_string_list(required_ids):
        target.errors.append("model_grader_disagreement_queue.required_review_item_ids must be a list of strings.")
        required_ids = []
    expected_ids = sorted(row["review_item_id"] for row in observed_queue if row.get("review_item_id"))
    if required_ids != expected_ids:
        target.errors.append("model_grader_disagreement_queue.required_review_item_ids must match queue review_item_id values.")
    if dry_run_path is not None:
        dry_run = _read_object(dry_run_path, target, "model_grader_disagreement_queue.source_artifacts.dry_run_receipt") or {}
        expected_rows = dry_run.get("disagreement_queue") if isinstance(dry_run.get("disagreement_queue"), list) else []
        expected_queue = [_model_grader_disagreement_queue_key(row) for row in expected_rows if isinstance(row, dict)]
        if rows != expected_rows:
            target.errors.append("model_grader_disagreement_queue.queue must match source dry-run disagreement_queue.")
        if observed_queue != expected_queue:
            target.errors.append("model_grader_disagreement_queue.queue keys must match source dry-run disagreement_queue.")
    override_requirements = queue.get("override_requirements") if isinstance(queue.get("override_requirements"), dict) else {}
    if isinstance(queue.get("override_requirements"), dict):
        _validate_allowed_keys(
            override_requirements,
            _MODEL_GRADER_DISAGREEMENT_QUEUE_OVERRIDE_REQUIREMENT_KEYS,
            target,
            "model_grader_disagreement_queue.override_requirements",
        )
    else:
        target.errors.append("model_grader_disagreement_queue.override_requirements must be an object.")
    if override_requirements.get("required_before_training_admission") is not True:
        target.errors.append("model_grader_disagreement_queue.override_requirements.required_before_training_admission must be true.")
    if override_requirements.get("required_override_count") != len(rows):
        target.errors.append("model_grader_disagreement_queue.override_requirements.required_override_count must match queue_count.")
    final_options = override_requirements.get("final_label_options")
    if final_options != [label for label in REVIEW_LABELS if label != "needs_review"]:
        target.errors.append("model_grader_disagreement_queue.override_requirements.final_label_options must be finalized review labels.")
    if override_requirements.get("minimum_reviewer_confidence") != "medium":
        target.errors.append("model_grader_disagreement_queue.override_requirements.minimum_reviewer_confidence must be medium.")
    admission = queue.get("training_admission") if isinstance(queue.get("training_admission"), dict) else {}
    if isinstance(queue.get("training_admission"), dict):
        _validate_allowed_keys(
            admission,
            _MODEL_GRADER_OVERRIDE_TRAINING_ADMISSION_KEYS,
            target,
            "model_grader_disagreement_queue.training_admission",
        )
    else:
        target.errors.append("model_grader_disagreement_queue.training_admission must be an object.")
    if admission.get("labels_allowed_for_training") is not False:
        target.errors.append("model_grader_disagreement_queue.training_admission.labels_allowed_for_training must be false.")
    if admission.get("labels_admitted_count") != 0:
        target.errors.append("model_grader_disagreement_queue.training_admission.labels_admitted_count must be 0.")
    if admission.get("requires_model_grader_gate") is not True:
        target.errors.append("model_grader_disagreement_queue.training_admission.requires_model_grader_gate must be true.")
    expected_readiness = "blocked"
    expected_recommendation = "fix_dry_run_receipt"
    if failed_checks == 0 and rows:
        expected_readiness = "ready_for_human_review"
        expected_recommendation = "collect_human_overrides"
    elif failed_checks == 0:
        expected_readiness = "queue_empty"
        expected_recommendation = "no_human_override_required"
    if queue.get("readiness") != expected_readiness:
        target.errors.append(
            f"model_grader_disagreement_queue.readiness expected {expected_readiness!r}, got {queue.get('readiness')!r}."
        )
    if queue.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"model_grader_disagreement_queue.recommendation expected {expected_recommendation!r}, got {queue.get('recommendation')!r}."
        )
    _validate_model_grader_boundary(queue.get("execution_boundary"), target, "model_grader_disagreement_queue")
    target.details.update(
        {
            "readiness": queue.get("readiness"),
            "queue_count": queue.get("queue_count"),
        }
    )

def _validate_model_grader_override_receipt(receipt: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(receipt, _MODEL_GRADER_OVERRIDE_RECEIPT_KEYS, target, "model_grader_override_receipt")
    _require_equal(
        receipt,
        "schema_version",
        MODEL_GRADER_OVERRIDE_RECEIPT_SCHEMA_VERSION,
        target,
        prefix="model_grader_override_receipt.",
    )
    failed_checks = _validate_model_grader_checked_artifact(receipt, target, "model_grader_override_receipt")
    _validate_model_grader_check_keys(receipt.get("checks"), target, "model_grader_override_receipt.checks")
    _validate_model_grader_override_sources(receipt.get("source_artifacts"), target, source_path)
    expected_readiness = "ready_for_model_grader_gate" if failed_checks == 0 else "blocked"
    if receipt.get("readiness") != expected_readiness:
        target.errors.append(
            f"model_grader_override_receipt.readiness expected {expected_readiness!r}, got {receipt.get('readiness')!r}."
        )
    queue = receipt.get("queue") if isinstance(receipt.get("queue"), dict) else {}
    if isinstance(receipt.get("queue"), dict):
        _validate_allowed_keys(queue, _MODEL_GRADER_OVERRIDE_QUEUE_KEYS, target, "model_grader_override_receipt.queue")
    for field_name in ("required_review_item_ids", "resolved_review_item_ids", "unresolved_review_item_ids"):
        if not _is_string_list(queue.get(field_name)):
            target.errors.append(f"model_grader_override_receipt.queue.{field_name} must be a list of strings.")
    required_ids = set(queue.get("required_review_item_ids", [])) if isinstance(queue.get("required_review_item_ids"), list) else set()
    resolved_ids = set(queue.get("resolved_review_item_ids", [])) if isinstance(queue.get("resolved_review_item_ids"), list) else set()
    unresolved_ids = set(queue.get("unresolved_review_item_ids", [])) if isinstance(queue.get("unresolved_review_item_ids"), list) else set()
    if not _is_non_negative_int(queue.get("dry_run_disagreement_queue_count")):
        target.errors.append("model_grader_override_receipt.queue.dry_run_disagreement_queue_count must be a non-negative integer.")
    elif queue.get("dry_run_disagreement_queue_count") != len(required_ids):
        target.errors.append("model_grader_override_receipt.queue.dry_run_disagreement_queue_count must match required_review_item_ids.")
    if resolved_ids - required_ids:
        target.errors.append("model_grader_override_receipt.queue.resolved_review_item_ids must be a subset of required_review_item_ids.")
    if unresolved_ids != required_ids - resolved_ids:
        target.errors.append("model_grader_override_receipt.queue.unresolved_review_item_ids must be required minus resolved ids.")
    overrides = receipt.get("overrides")
    if not isinstance(overrides, list):
        target.errors.append("model_grader_override_receipt.overrides must be a list.")
        overrides = []
    accepted_count = 0
    for index, row in enumerate(overrides):
        label = f"model_grader_override_receipt.overrides[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(row, _MODEL_GRADER_OVERRIDE_ROW_KEYS, target, label)
        for field_name in ("review_item_id", "human_label", "reviewer_confidence", "reviewer", "reviewed_at", "notes"):
            if not isinstance(row.get(field_name), str):
                target.errors.append(f"{label}.{field_name} must be a string.")
        if not row.get("review_item_id"):
            target.errors.append(f"{label}.review_item_id must be non-empty.")
        if row.get("human_label") not in set(REVIEW_LABELS) - {"needs_review"}:
            target.errors.append(f"{label}.human_label must be a finalized review label.")
        if row.get("reviewer_confidence") not in {"high", "medium"}:
            target.errors.append(f"{label}.reviewer_confidence must be high or medium.")
        if not row.get("reviewer"):
            target.errors.append(f"{label}.reviewer must be non-empty.")
        if not row.get("reviewed_at"):
            target.errors.append(f"{label}.reviewed_at must be non-empty.")
        if not isinstance(row.get("resolves_queue_item"), bool):
            target.errors.append(f"{label}.resolves_queue_item must be a boolean.")
        if not isinstance(row.get("accepted"), bool):
            target.errors.append(f"{label}.accepted must be a boolean.")
        elif row.get("accepted") is True:
            accepted_count += 1
        if not isinstance(row.get("errors"), list) or not all(isinstance(item, str) for item in row.get("errors", [])):
            target.errors.append(f"{label}.errors must be a list of strings.")
        override_hash = row.get("override_sha256")
        if not _is_sha256(override_hash):
            target.errors.append(f"{label}.override_sha256 must be a sha256 hex digest.")
        elif override_hash != _model_grader_override_sha256(row):
            target.errors.append(f"{label}.override_sha256 does not match override contents.")
    metrics = receipt.get("metrics") if isinstance(receipt.get("metrics"), dict) else {}
    if isinstance(receipt.get("metrics"), dict):
        _validate_allowed_keys(metrics, _MODEL_GRADER_OVERRIDE_METRICS_KEYS, target, "model_grader_override_receipt.metrics")
    for field_name in (
        "override_row_count",
        "resolved_queue_count",
        "unresolved_queue_count",
        "unmatched_override_count",
        "invalid_override_count",
    ):
        if not _is_non_negative_int(metrics.get(field_name)):
            target.errors.append(f"model_grader_override_receipt.metrics.{field_name} must be a non-negative integer.")
    if metrics.get("override_row_count") != len(overrides):
        target.errors.append("model_grader_override_receipt.metrics.override_row_count must match overrides length.")
    if metrics.get("resolved_queue_count") != len(resolved_ids):
        target.errors.append("model_grader_override_receipt.metrics.resolved_queue_count must match resolved ids.")
    if metrics.get("unresolved_queue_count") != len(unresolved_ids):
        target.errors.append("model_grader_override_receipt.metrics.unresolved_queue_count must match unresolved ids.")
    if failed_checks == 0 and metrics.get("unresolved_queue_count") != 0:
        target.errors.append("model_grader_override_receipt.metrics.unresolved_queue_count must be 0 when passed.")
    if failed_checks == 0 and accepted_count != len(required_ids):
        target.errors.append("model_grader_override_receipt accepted overrides must cover all required queue ids when passed.")
    admission = receipt.get("training_admission") if isinstance(receipt.get("training_admission"), dict) else {}
    if isinstance(receipt.get("training_admission"), dict):
        _validate_allowed_keys(
            admission,
            _MODEL_GRADER_OVERRIDE_TRAINING_ADMISSION_KEYS,
            target,
            "model_grader_override_receipt.training_admission",
        )
    if admission.get("labels_allowed_for_training") is not False:
        target.errors.append("model_grader_override_receipt.training_admission.labels_allowed_for_training must be false.")
    if admission.get("labels_admitted_count") != 0:
        target.errors.append("model_grader_override_receipt.training_admission.labels_admitted_count must be 0.")
    if admission.get("requires_model_grader_gate") is not True:
        target.errors.append("model_grader_override_receipt.training_admission.requires_model_grader_gate must be true.")
    _validate_model_grader_boundary(receipt.get("execution_boundary"), target, "model_grader_override_receipt")
    target.details.update(
        {
            "readiness": receipt.get("readiness"),
            "resolved_queue_count": metrics.get("resolved_queue_count"),
            "unresolved_queue_count": metrics.get("unresolved_queue_count"),
        }
    )

def _validate_model_grader_gate(gate: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(gate, _MODEL_GRADER_GATE_KEYS, target, "model_grader_gate")
    _require_equal(gate, "schema_version", MODEL_GRADER_GATE_SCHEMA_VERSION, target, prefix="model_grader_gate.")
    failed_checks = _validate_model_grader_checked_artifact(gate, target, "model_grader_gate")
    _validate_model_grader_check_keys(gate.get("checks"), target, "model_grader_gate.checks")
    source_paths = _validate_model_grader_gate_sources(gate.get("source_artifacts"), target, source_path)
    expected_readiness = "labels_calibrated_for_curated_handoff" if failed_checks == 0 else "blocked"
    if gate.get("readiness") != expected_readiness:
        target.errors.append(f"model_grader_gate.readiness expected {expected_readiness!r}, got {gate.get('readiness')!r}.")
    admission = gate.get("admission") if isinstance(gate.get("admission"), dict) else {}
    if isinstance(gate.get("admission"), dict):
        _validate_allowed_keys(admission, _MODEL_GRADER_GATE_ADMISSION_KEYS, target, "model_grader_gate.admission")
    labels_allowed = admission.get("labels_allowed_for_training")
    if labels_allowed != (failed_checks == 0):
        target.errors.append("model_grader_gate.admission.labels_allowed_for_training must match gate pass/fail.")
    if admission.get("uncalibrated_labels_admitted") != 0:
        target.errors.append("model_grader_gate.admission.uncalibrated_labels_admitted must be 0.")
    if failed_checks and admission.get("labels_admitted_count") != 0:
        target.errors.append("model_grader_gate.admission.labels_admitted_count must be 0 when the gate is blocked.")
    metrics = gate.get("metrics") if isinstance(gate.get("metrics"), dict) else {}
    for field_name in (
        "graded_item_count",
        "calibration_disagreement_count",
        "dry_run_disagreement_queue_count",
        "dry_run_labels_requiring_human_review_count",
        "human_override_resolved_count",
        "human_override_unresolved_count",
    ):
        if not _is_non_negative_int(metrics.get(field_name)):
            target.errors.append(f"model_grader_gate.metrics.{field_name} must be a non-negative integer.")
    if not isinstance(metrics.get("agreement_rate"), (int, float)) or not 0 <= metrics.get("agreement_rate") <= 1:
        target.errors.append("model_grader_gate.metrics.agreement_rate must be a number from 0 to 1.")
    if not isinstance(metrics.get("human_override_receipt_present"), bool):
        target.errors.append("model_grader_gate.metrics.human_override_receipt_present must be a boolean.")
    if failed_checks == 0:
        admitted_count = admission.get("labels_admitted_count")
        if admitted_count != metrics.get("graded_item_count"):
            target.errors.append("model_grader_gate.admission.labels_admitted_count must equal graded_item_count when passed.")
        unresolved_source_count = max(
            metrics.get("dry_run_disagreement_queue_count", 0),
            metrics.get("dry_run_labels_requiring_human_review_count", 0),
        )
        if unresolved_source_count:
            if metrics.get("human_override_receipt_present") is not True:
                target.errors.append("model_grader_gate.metrics.human_override_receipt_present must be true when queued labels pass.")
            if metrics.get("human_override_unresolved_count") != 0:
                target.errors.append("model_grader_gate.metrics.human_override_unresolved_count must be 0 when passed.")
            if metrics.get("human_override_resolved_count", 0) < unresolved_source_count:
                target.errors.append("model_grader_gate.metrics.human_override_resolved_count must cover queued labels when passed.")
    if isinstance(gate.get("metrics"), dict):
        _validate_allowed_keys(metrics, _MODEL_GRADER_GATE_METRICS_KEYS, target, "model_grader_gate.metrics")
    _validate_model_grader_boundary(gate.get("execution_boundary"), target, "model_grader_gate")
    _replay_model_grader_gate(gate, target, source_path, source_paths)
    target.details.update(
        {
            "readiness": gate.get("readiness"),
            "recommendation": gate.get("recommendation"),
            "admitted_count": admission.get("labels_admitted_count"),
        }
    )

def _replay_model_grader_gate(
    gate: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
    source_paths: dict[str, Path | None],
) -> None:
    dry_run_path = source_paths.get("dry_run_receipt")
    rubric_path = source_paths.get("rubric_spec")
    if dry_run_path is None or rubric_path is None:
        return
    policy = _model_grader_gate_replay_policy(gate, target)
    if policy is None:
        return
    calibration_path = source_paths.get("review_calibration")
    if calibration_path is not None:
        lineage_matches, _lineage = _model_grader_review_lineage_status(
            dry_run_path,
            rubric_path,
            calibration_path,
        )
        if not lineage_matches and gate.get("passed") is True:
            target.errors.append(
                "model_grader_gate review calibration must derive from the same review items as its dry-run and rubric."
            )
    created_at = gate.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        target.errors.append("model_grader_gate.created_at must be a non-empty string for deterministic replay.")
        return

    source_before = _model_grader_gate_source_attestations(source_paths, target)
    replayed: dict[str, Any] | None = None
    try:
        replayed = build_model_grader_gate(
            dry_run_path=dry_run_path,
            rubric_path=rubric_path,
            calibration_path=source_paths.get("review_calibration"),
            override_receipt_path=source_paths.get("model_grader_override_receipt"),
            min_calibration_agreement_rate=policy[0],
            max_disagreements=policy[1],
            created_at=created_at,
            preserve_paths=False,
            out_path=source_path,
        )
    except (ModelGraderError, OSError, TypeError, ValueError) as exc:
        target.errors.append(f"model_grader_gate could not replay its recorded sources and policy: {exc}")
    source_after = _model_grader_gate_source_attestations(source_paths, target)
    if source_before is not None and source_after is not None and source_before != source_after:
        target.errors.append("model_grader_gate source artifacts changed while validation was running.")
    if replayed is not None and gate != replayed:
        target.errors.append(
            "model_grader_gate must match deterministic replay of its recorded sources, policy, and created_at."
        )

def _model_grader_gate_replay_policy(
    gate: dict[str, Any],
    target: ValidationTarget,
) -> tuple[float, int | None] | None:
    checks = gate.get("checks")
    if not isinstance(checks, list):
        return None
    minimum_checks = [
        check
        for check in checks
        if isinstance(check, dict) and check.get("id") == "min_calibration_agreement_rate"
    ]
    if len(minimum_checks) != 1:
        target.errors.append(
            "model_grader_gate.checks must contain exactly one min_calibration_agreement_rate policy check."
        )
        return None
    minimum_expected = minimum_checks[0].get("expected")
    minimum = minimum_expected.get("min") if isinstance(minimum_expected, dict) else None
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not 0 <= minimum <= 1:
        target.errors.append(
            "model_grader_gate min_calibration_agreement_rate expected.min must be a number from 0 to 1."
        )
        return None

    maximum_checks = [
        check
        for check in checks
        if isinstance(check, dict) and check.get("id") == "max_calibration_disagreements"
    ]
    if len(maximum_checks) > 1:
        target.errors.append(
            "model_grader_gate.checks may contain at most one max_calibration_disagreements policy check."
        )
        return None
    maximum: int | None = None
    if maximum_checks:
        maximum_expected = maximum_checks[0].get("expected")
        maximum = maximum_expected.get("max") if isinstance(maximum_expected, dict) else None
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
            target.errors.append(
                "model_grader_gate max_calibration_disagreements expected.max must be a non-negative integer."
            )
            return None
    return float(minimum), maximum

def _model_grader_gate_source_attestations(
    source_paths: dict[str, Path | None],
    target: ValidationTarget,
) -> dict[str, tuple[int | str, ...] | None] | None:
    attestations: dict[str, tuple[int | str, ...] | None] = {}
    for role, source_path in source_paths.items():
        if source_path is None:
            attestations[role] = None
            continue
        try:
            digest = hashlib.sha256()
            with source_path.open("rb") as handle:
                stat_before = os.fstat(handle.fileno())
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                stat_after = os.fstat(handle.fileno())
            pathname_after = source_path.stat(follow_symlinks=False)
        except OSError as exc:
            target.errors.append(f"model_grader_gate source artifact {role!r} could not be reattested: {exc}")
            return None
        signatures = {
            (
                item.st_mode,
                item.st_dev,
                item.st_ino,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )
            for item in (stat_before, stat_after, pathname_after)
        }
        if len(signatures) != 1 or not stat.S_ISREG(stat_before.st_mode):
            target.errors.append(f"model_grader_gate source artifact {role!r} changed while it was being reattested.")
            return None
        attestations[role] = (*next(iter(signatures)), digest.hexdigest())
    return attestations

def _validate_model_grader_checked_artifact(payload: dict[str, Any], target: ValidationTarget, label: str) -> int:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        target.errors.append(f"{label}.checks must be a list.")
        checks = []
    failed_checks = _validate_gate_like_checks(checks, target, f"{label}.checks")
    if payload.get("check_count") != len(checks):
        target.errors.append(f"{label}.check_count expected {len(checks)}, got {payload.get('check_count')!r}.")
    if payload.get("failed_check_count") != failed_checks:
        target.errors.append(f"{label}.failed_check_count expected {failed_checks}, got {payload.get('failed_check_count')!r}.")
    if isinstance(payload.get("passed"), bool) and payload["passed"] != (failed_checks == 0):
        target.errors.append(f"{label}.passed must match failed_check_count.")
    if not isinstance(payload.get("blocked_reasons"), list) or not all(isinstance(item, str) for item in payload.get("blocked_reasons", [])):
        target.errors.append(f"{label}.blocked_reasons must be a list of strings.")
    return failed_checks

def _validate_model_grader_check_keys(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, list):
        return
    for index, check in enumerate(value):
        if isinstance(check, dict):
            _validate_allowed_keys(check, _MODEL_GRADER_CHECK_KEYS, target, f"{label}[{index}]")

def _validate_model_grader_dry_run_sources(value: Any, target: ValidationTarget, source_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if not isinstance(value, dict):
        target.errors.append("model_grader_dry_run.source_artifacts must be an object.")
        return paths
    _validate_allowed_keys(value, _MODEL_GRADER_DRY_RUN_SOURCE_KEYS, target, "model_grader_dry_run.source_artifacts")
    review_export_paths = _validate_model_grader_review_export_ref(
        value.get("review_export"),
        target,
        "model_grader_dry_run.source_artifacts.review_export",
        source_path,
    )
    if "review_items" in review_export_paths:
        paths["review_items"] = review_export_paths["review_items"]
    rubric_path = _validate_model_grader_referenced_artifact(
        value.get("rubric_spec"),
        target,
        "model_grader_dry_run.source_artifacts.rubric_spec",
        source_path,
        validate_rubric_spec,
        allow_missing=False,
    )
    if rubric_path is not None:
        paths["rubric_spec"] = rubric_path
        _validate_model_grader_dry_run_review_export_matches_rubric(review_export_paths, rubric_path, target)
    return paths

def _validate_model_grader_disagreement_queue_sources(value: Any, target: ValidationTarget, source_path: Path) -> Path | None:
    if not isinstance(value, dict):
        target.errors.append("model_grader_disagreement_queue.source_artifacts must be an object.")
        return None
    _validate_allowed_keys(
        value,
        _MODEL_GRADER_DISAGREEMENT_QUEUE_SOURCE_KEYS,
        target,
        "model_grader_disagreement_queue.source_artifacts",
    )
    return _validate_model_grader_referenced_artifact(
        value.get("dry_run_receipt"),
        target,
        "model_grader_disagreement_queue.source_artifacts.dry_run_receipt",
        source_path,
        validate_model_grader_dry_run,
        allow_missing=False,
    )

def _validate_rubric_review_item_fingerprints(
    fingerprints: list[Any],
    review_items_path: Path | None,
    target: ValidationTarget,
) -> None:
    if review_items_path is None:
        return
    review_items = _read_jsonl_objects(
        review_items_path,
        target,
        "rubric_spec.review_export.review_items",
    )
    expected = [
        {
            "review_item_id": str(item.get("review_item_id") or ""),
            "episode_id": str(item.get("episode_id") or ""),
            "scenario_id": str(item.get("scenario_id") or ""),
            "review_item_sha256": review_item_sha256(item),
        }
        for item in review_items
    ]
    if fingerprints != expected:
        target.errors.append("rubric_spec.review_item_fingerprints must match review_export.review_items.")

def _validate_model_grader_dry_run_labels_match_review_items(
    label_source_keys: list[dict[str, Any]],
    review_items_path: Path | None,
    grader: dict[str, Any],
    target: ValidationTarget,
) -> None:
    if review_items_path is None:
        return
    review_items = _read_jsonl_objects(
        review_items_path,
        target,
        "model_grader_dry_run.source_artifacts.review_export.review_items",
    )
    expected = [_model_grader_expected_label_source_key(item, grader) for item in review_items]
    if label_source_keys != expected:
        target.errors.append("model_grader_dry_run.grader_labels must match source_artifacts.review_export.review_items.")

def _validate_model_grader_dry_run_review_export_matches_rubric(
    dry_run_paths: dict[str, Path],
    rubric_path: Path,
    target: ValidationTarget,
) -> None:
    if not dry_run_paths:
        return
    try:
        rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(rubric, dict):
        return
    rubric_paths = _validate_model_grader_review_export_ref(
        rubric.get("review_export"),
        target,
        "model_grader_dry_run.source_artifacts.rubric_spec.review_export",
        rubric_path,
    )
    for field_name in ("manifest", "review_items"):
        dry_path = dry_run_paths.get(field_name)
        rubric_review_path = rubric_paths.get(field_name)
        if dry_path is not None and rubric_review_path is not None and not _same_model_grader_source_file(
            dry_path,
            rubric_review_path,
        ):
            target.errors.append("model_grader_dry_run.source_artifacts.review_export must match rubric_spec.review_export.")
            return

def _validate_model_grader_disagreement_queue(
    queue: list[Any],
    target: ValidationTarget,
    label_prefix: str = "model_grader_dry_run.disagreement_queue",
) -> list[dict[str, str]]:
    observed: list[dict[str, str]] = []
    for index, row in enumerate(queue):
        label = f"{label_prefix}[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(row, _MODEL_GRADER_DISAGREEMENT_KEYS, target, label)
        for field_name in ("review_item_id", "episode_id", "scenario_id", "task_family", "review_item_sha256", "mock_model_label", "reason"):
            if not isinstance(row.get(field_name), str) or not row.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if not _is_sha256(row.get("review_item_sha256")):
            target.errors.append(f"{label}.review_item_sha256 must be a sha256 hex digest.")
        if row.get("mock_model_label") not in REVIEW_LABELS:
            target.errors.append(f"{label}.mock_model_label must be one of {list(REVIEW_LABELS)!r}.")
        if row.get("reason") != "human_review_required_by_rubric_or_low_confidence":
            target.errors.append(f"{label}.reason must be human_review_required_by_rubric_or_low_confidence.")
        if row.get("source_report") is not None and not isinstance(row.get("source_report"), str):
            target.errors.append(f"{label}.source_report must be a string or null.")
        observed.append(_model_grader_disagreement_queue_key(row))
    return observed

def _model_grader_disagreement_queue_key(row: dict[str, Any]) -> dict[str, str]:
    return {
        "review_item_id": str(row.get("review_item_id") or ""),
        "episode_id": str(row.get("episode_id") or ""),
        "scenario_id": str(row.get("scenario_id") or ""),
        "task_family": str(row.get("task_family") or ""),
        "review_item_sha256": str(row.get("review_item_sha256") or ""),
        "mock_model_label": str(row.get("mock_model_label") or ""),
    }

def _model_grader_label_source_key(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_item_id": str(row.get("review_item_id") or ""),
        "episode_id": str(row.get("episode_id") or ""),
        "scenario_id": str(row.get("scenario_id") or ""),
        "task_family": str(row.get("task_family") or ""),
        "review_item_sha256": str(row.get("review_item_sha256") or ""),
        "mock_model_label": str(row.get("mock_model_label") or ""),
        "mock_confidence": str(row.get("mock_confidence") or ""),
        "requires_human_review": row.get("requires_human_review") if isinstance(row.get("requires_human_review"), bool) else None,
        "grader_id": str(row.get("grader_id") or ""),
        "provider": str(row.get("provider") or ""),
    }

def _model_grader_expected_label_source_key(item: dict[str, Any], grader: dict[str, Any]) -> dict[str, Any]:
    scorecard = item.get("scorecard") if isinstance(item.get("scorecard"), dict) else {}
    suggested = str(item.get("suggested_human_label") or ("accept" if scorecard.get("passed") is True else "reject"))
    if suggested not in REVIEW_LABELS:
        suggested = "needs_review"
    confidence = "medium" if suggested in {"accept", "reject"} else "low"
    return {
        "review_item_id": str(item.get("review_item_id") or ""),
        "episode_id": str(item.get("episode_id") or ""),
        "scenario_id": str(item.get("scenario_id") or ""),
        "task_family": str(item.get("task_family") or "unknown"),
        "review_item_sha256": review_item_sha256(item),
        "mock_model_label": suggested,
        "mock_confidence": confidence,
        "requires_human_review": suggested in {"needs_review", "unsafe", "incomplete"} or confidence == "low",
        "grader_id": str(grader.get("grader_id") or ""),
        "provider": str(grader.get("provider") or ""),
    }

def _model_grader_label_sha256(row: dict[str, Any]) -> str:
    return _model_grader_row_sha256(row)

def _model_grader_override_sha256(row: dict[str, Any]) -> str:
    payload = dict(row)
    payload["override_sha256"] = ""
    return _model_grader_row_sha256(payload)

def _model_grader_row_sha256(row: dict[str, Any]) -> str:
    payload = {key: item for key, item in row.items() if key != "label_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _validate_model_grader_gate_sources(
    value: Any,
    target: ValidationTarget,
    source_path: Path,
) -> dict[str, Path | None]:
    paths: dict[str, Path | None] = {}
    if not isinstance(value, dict):
        target.errors.append("model_grader_gate.source_artifacts must be an object.")
        return paths
    _validate_allowed_keys(value, _MODEL_GRADER_GATE_SOURCE_KEYS, target, "model_grader_gate.source_artifacts")
    paths["dry_run_receipt"] = _validate_model_grader_referenced_artifact(
        value.get("dry_run_receipt"),
        target,
        "model_grader_gate.source_artifacts.dry_run_receipt",
        source_path,
        validate_model_grader_dry_run,
        allow_missing=False,
    )
    paths["rubric_spec"] = _validate_model_grader_referenced_artifact(
        value.get("rubric_spec"),
        target,
        "model_grader_gate.source_artifacts.rubric_spec",
        source_path,
        validate_rubric_spec,
        allow_missing=False,
    )
    paths["review_calibration"] = _validate_model_grader_referenced_artifact(
        value.get("review_calibration"),
        target,
        "model_grader_gate.source_artifacts.review_calibration",
        source_path,
        validate_review_calibration,
        allow_missing=True,
    )
    paths["model_grader_override_receipt"] = _validate_model_grader_referenced_artifact(
        value.get("model_grader_override_receipt"),
        target,
        "model_grader_gate.source_artifacts.model_grader_override_receipt",
        source_path,
        validate_model_grader_override_receipt,
        allow_missing=True,
    )
    return paths

def _validate_model_grader_override_sources(value: Any, target: ValidationTarget, source_path: Path) -> None:
    if not isinstance(value, dict):
        target.errors.append("model_grader_override_receipt.source_artifacts must be an object.")
        return
    _validate_allowed_keys(value, _MODEL_GRADER_OVERRIDE_SOURCE_KEYS, target, "model_grader_override_receipt.source_artifacts")
    _validate_model_grader_referenced_artifact(
        value.get("dry_run_receipt"),
        target,
        "model_grader_override_receipt.source_artifacts.dry_run_receipt",
        source_path,
        validate_model_grader_dry_run,
        allow_missing=False,
    )
    _validate_model_grader_source_file_ref(
        value.get("override_rows"),
        target,
        "model_grader_override_receipt.source_artifacts.override_rows",
        source_path,
        allow_missing=False,
    )

def _validate_model_grader_review_export_ref(
    value: Any,
    target: ValidationTarget,
    label: str,
    source_path: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return paths
    _validate_allowed_keys(value, _MODEL_GRADER_REVIEW_EXPORT_KEYS, target, label)
    if not isinstance(value.get("path"), str) or not value.get("path"):
        target.errors.append(f"{label}.path must be a non-empty string.")
    manifest_path = _validate_model_grader_source_file_ref(
        value.get("manifest"),
        target,
        f"{label}.manifest",
        source_path,
        allow_missing=False,
    )
    if manifest_path is not None:
        paths["manifest"] = manifest_path
    review_items_path = _validate_model_grader_source_file_ref(
        value.get("review_items"),
        target,
        f"{label}.review_items",
        source_path,
        allow_missing=False,
    )
    if review_items_path is not None:
        paths["review_items"] = review_items_path
    return paths

def _validate_model_grader_referenced_artifact(
    record: Any,
    target: ValidationTarget,
    label: str,
    source_path: Path,
    validator: Any,
    *,
    allow_missing: bool,
) -> Path | None:
    if allow_missing and record is None:
        return None
    artifact_path = _validate_model_grader_source_file_ref(record, target, label, source_path, allow_missing=allow_missing)
    if artifact_path is None:
        return None
    referenced = validator(artifact_path)
    target.warnings.extend(f"{label}: {warning}" for warning in referenced.warnings)
    target.errors.extend(f"{label}: {error}" for error in referenced.errors)
    return artifact_path

def _validate_model_grader_source_file_ref(
    record: Any,
    target: ValidationTarget,
    label: str,
    source_path: Path,
    *,
    allow_missing: bool,
) -> Path | None:
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return None
    _validate_allowed_keys(record, _MODEL_GRADER_FILE_REF_KEYS, target, label)
    path_value = record.get("path")
    artifact_path: Path | None = None
    if isinstance(path_value, str) and path_value:
        if path_value.startswith("<redacted:"):
            if record.get("exists") is True:
                target.errors.append(f"{label}.path cannot be redacted when exists is true.")
                return None
            if allow_missing and record.get("exists") is False:
                return None
        elif not _is_public_model_grader_ref_path(path_value):
            target.errors.append(f"{label}.path must be a relative path or redacted placeholder.")
            return None
        else:
            artifact_path = _resolve_model_grader_source_path(path_value, source_path)
            if artifact_path is None:
                target.errors.append(f"{label}.path must resolve from the model-grader artifact location.")
                return None
            if _path_has_symlink_component(artifact_path, include_leaf=True):
                target.errors.append(f"{label}.path must resolve to a regular non-symlink file.")
                return None
    exists = record.get("exists")
    if exists is not True:
        if allow_missing and exists is False:
            return None
        target.errors.append(f"{label}.exists must be true.")
        return None
    if artifact_path is None:
        target.errors.append(f"{label}.path must be a non-empty string when exists is true.")
        return None
    if not _is_non_negative_int(record.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer when exists is true.")
    if not _is_lowercase_sha256(record.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a lowercase SHA-256 hex string when exists is true.")
    if not artifact_path.exists() or not artifact_path.is_file():
        target.errors.append(f"{label}.path must resolve to an existing file when exists is true.")
        return None
    if _is_non_negative_int(record.get("size_bytes")) and artifact_path.stat().st_size != record.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if _is_lowercase_sha256(record.get("sha256")) and _sha256(artifact_path) != record.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current file.")
    return artifact_path

def _resolve_model_grader_source_path(value: Any, source_path: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("<redacted:") or not _is_public_model_grader_ref_path(value):
        return None
    path = Path(value)
    return source_path.parent / path

def _is_public_model_grader_ref_path(value: str) -> bool:
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

def _same_model_grader_source_file(left: Path, right: Path) -> bool:
    try:
        if left.resolve() == right.resolve():
            return True
        if left.stat().st_size != right.stat().st_size:
            return False
        return _sha256(left) == _sha256(right)
    except OSError:
        return False

def _validate_model_grader_boundary(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label}.execution_boundary must be an object.")
        return
    _validate_allowed_keys(value, _MODEL_GRADER_BOUNDARY_KEYS, target, f"{label}.execution_boundary")
    true_fields = ("dry_run_only",)
    false_fields = (
        "provider_api_called",
        "paid_model_grader_calls_started",
        "credential_values_recorded",
        "labels_admitted_to_training",
        "weights_updated_by_flight_recorder",
    )
    for field_name in true_fields:
        if value.get(field_name) is not True:
            target.errors.append(f"{label}.execution_boundary.{field_name} must be true.")
    for field_name in false_fields:
        if value.get(field_name) is not False:
            target.errors.append(f"{label}.execution_boundary.{field_name} must be false.")
    if value.get("cloud_cost_incurred_usd") != 0:
        target.errors.append(f"{label}.execution_boundary.cloud_cost_incurred_usd must be 0.")

def _model_grader_label_counts(value: Any, target: ValidationTarget, label: str) -> dict[str, int]:
    counts = {name: 0 for name in REVIEW_LABELS}
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list.")
        return counts
    seen: set[str] = set()
    for index, row in enumerate(value):
        row_label = f"{label}[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{row_label} must be an object.")
            continue
        _validate_allowed_keys(row, _MODEL_GRADER_LABEL_COUNT_KEYS, target, row_label)
        name = row.get("label")
        count = row.get("count")
        if name not in REVIEW_LABELS:
            target.errors.append(f"{row_label}.label must be one of {list(REVIEW_LABELS)!r}.")
            continue
        if name in seen:
            target.errors.append(f"{row_label}.label duplicates {name!r}.")
        seen.add(name)
        if not _is_non_negative_int(count):
            target.errors.append(f"{row_label}.count must be a non-negative integer.")
            continue
        counts[name] = int(count)
    missing = sorted(set(REVIEW_LABELS) - seen)
    if missing:
        target.errors.append(f"{label} missing label count row(s): {', '.join(missing)}.")
    return counts

def validate_suite_trend(path: str | Path) -> ValidationTarget:
    """Validate one trend-suite artifact."""
    trend_path = Path(path)
    target = ValidationTarget("suite_trend", str(trend_path))
    trend = _read_object(trend_path, target, "suite_trend.json")
    if trend is None:
        return target
    _validate_suite_trend(trend, target)
    return target

def _validate_suite_summary(
    summary: dict[str, Any],
    target: ValidationTarget,
    source_path: Path | None = None,
    *,
    validate_sources: bool = True,
) -> None:
    _require_equal(summary, "schema_version", RUN_SUITE_SCHEMA_VERSION, target)
    runs = summary.get("runs")
    if not isinstance(runs, list):
        target.errors.append("suite_summary.runs must be a list.")
        runs = []
    errors = summary.get("errors")
    if not isinstance(errors, list):
        target.errors.append("suite_summary.errors must be a list.")
        errors = []

    if summary.get("total") != len(runs):
        target.errors.append(f"suite_summary.total expected {len(runs)}, got {summary.get('total')!r}.")
    passed = sum(1 for run in runs if isinstance(run, dict) and run.get("passed") is True)
    failed = len(runs) - passed
    if summary.get("passed") != passed:
        target.errors.append(f"suite_summary.passed expected {passed}, got {summary.get('passed')!r}.")
    if summary.get("failed") != failed:
        target.errors.append(f"suite_summary.failed expected {failed}, got {summary.get('failed')!r}.")
    if summary.get("error_count") != len(errors):
        target.errors.append(f"suite_summary.error_count expected {len(errors)}, got {summary.get('error_count')!r}.")

    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            target.errors.append(f"suite_summary.runs[{index}] must be an object.")
            continue
        for field_name in (
            "scenario_id",
            "scenario_title",
            "task_family",
            "scenario_path",
            "trace_path",
            "run_dir",
            "report",
            "scorecard",
            "run_digest",
            "lineage",
        ):
            if not isinstance(run.get(field_name), str) or not run.get(field_name):
                target.errors.append(f"suite_summary.runs[{index}].{field_name} must be a non-empty string.")
        for field_name in ("scenario_path", "trace_path", "run_dir", "report", "scorecard", "run_digest", "lineage"):
            _warn_absolute_public_path(target, f"suite_summary.runs[{index}].{field_name}", run.get(field_name))
        if not isinstance(run.get("passed"), bool):
            target.errors.append(f"suite_summary.runs[{index}].passed must be a boolean.")
        if not _is_int_between(run.get("score"), 0, 100):
            target.errors.append(f"suite_summary.runs[{index}].score must be an integer from 0 to 100.")
        if not _is_string_list(run.get("failed_rules")):
            target.errors.append(f"suite_summary.runs[{index}].failed_rules must be a list of strings.")
        if not _is_string_list(run.get("critical_failures")):
            target.errors.append(f"suite_summary.runs[{index}].critical_failures must be a list of strings.")
        for field_name in ("scenario_sha256", "trace_sha256"):
            if field_name in run and run.get(field_name) is not None and not _is_sha256(run.get(field_name)):
                target.errors.append(f"suite_summary.runs[{index}].{field_name} must be a SHA-256 hex string or null.")
        if validate_sources:
            label = f"suite_summary.runs[{index}]"
            _validate_suite_run_input_ref(run, "scenario_path", "scenario_sha256", target, label, source_path)
            _validate_suite_run_input_ref(run, "trace_path", "trace_sha256", target, label, source_path)
            for field_name in ("report", "scorecard", "run_digest", "lineage"):
                _validate_suite_run_artifact_ref(run, field_name, target, label, source_path)
            _validate_suite_run_dir(run, target, label, source_path)

    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("suite_summary.metrics must be an object.")
    else:
        _validate_suite_metrics(metrics, target, [run for run in runs if isinstance(run, dict)])

    artifacts = summary.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, dict):
        target.errors.append("suite_summary.artifacts must be an object when present.")
    if "metadata" in summary:
        _validate_metadata(summary.get("metadata"), target, "suite_summary.metadata")

    target.details.update(
        {
            "total": len(runs),
            "passed": passed,
            "failed": failed,
            "error_count": len(errors),
        }
    )

def _validate_suite_run_input_ref(
    run: dict[str, Any],
    path_field: str,
    sha_field: str,
    target: ValidationTarget,
    label: str,
    source_path: Path | None,
) -> None:
    raw_path = run.get(path_field)
    if not isinstance(raw_path, str) or not raw_path:
        return
    if _is_redacted_placeholder(raw_path):
        target.warnings.append(f"{label}.{path_field} is redacted and could not be source-validated.")
        return
    expected_sha = run.get(sha_field)
    if not _is_sha256(expected_sha):
        if expected_sha is None:
            target.errors.append(f"{label}.{sha_field} must be a SHA-256 hex string for an unredacted input.")
        return
    path = _resolve_suite_summary_ref_path(raw_path, source_path)
    if path is None:
        return
    if _path_has_symlink_component(path, include_leaf=True):
        target.errors.append(f"{label}.{path_field} must resolve to a regular non-symlink file.")
        return
    if not path.is_file():
        target.errors.append(f"{label}.{path_field} must resolve to an existing file.")
        return
    if _sha256(path) != expected_sha:
        target.errors.append(f"{label}.{sha_field} does not match the current file.")

def _validate_suite_run_artifact_ref(
    run: dict[str, Any],
    field_name: str,
    target: ValidationTarget,
    label: str,
    source_path: Path | None,
) -> None:
    sha_field = f"{field_name}_sha256"
    size_field = f"{field_name}_size_bytes"
    raw_path = run.get(field_name)
    if isinstance(raw_path, str) and _is_redacted_placeholder(raw_path):
        target.warnings.append(f"{label}.{field_name} is redacted and could not be source-validated.")
    expected_sha = run.get(sha_field)
    expected_size = run.get(size_field)
    if not _is_sha256(expected_sha):
        target.errors.append(f"{label}.{sha_field} must be a SHA-256 hex string.")
    if not _is_non_negative_int(expected_size):
        target.errors.append(f"{label}.{size_field} must be a non-negative integer.")
    path = _resolve_suite_summary_ref_path(run.get(field_name), source_path)
    if path is None:
        return
    if _path_has_symlink_component(path, include_leaf=True):
        target.errors.append(f"{label}.{field_name} must resolve to a regular non-symlink file.")
        return
    if not path.is_file():
        target.errors.append(f"{label}.{field_name} must resolve to an existing file.")
        return
    if _is_non_negative_int(expected_size) and path.stat().st_size != expected_size:
        target.errors.append(f"{label}.{size_field} does not match the current file.")
    if _is_sha256(expected_sha) and _sha256(path) != expected_sha:
        target.errors.append(f"{label}.{sha_field} does not match the current file.")

def _validate_suite_run_dir(
    run: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path | None,
) -> None:
    raw_run_dir = run.get("run_dir")
    if isinstance(raw_run_dir, str) and _is_redacted_placeholder(raw_run_dir):
        target.warnings.append(f"{label}.run_dir is redacted and could not be source-validated.")
        return
    run_dir = _resolve_suite_summary_ref_path(run.get("run_dir"), source_path)
    if run_dir is None:
        return
    if _path_has_symlink_component(run_dir, include_leaf=True):
        target.errors.append(f"{label}.run_dir must resolve to a regular non-symlink directory.")
        return
    if not run_dir.is_dir():
        target.errors.append(f"{label}.run_dir must resolve to an existing directory.")
        return

    expected_artifacts = {
        "report": "report.html",
        "scorecard": "scorecard.json",
        "run_digest": "run_digest.json",
        "lineage": "artifact_lineage.json",
    }
    for field_name, basename in expected_artifacts.items():
        artifact_path = _resolve_suite_summary_ref_path(run.get(field_name), source_path)
        if artifact_path is None:
            continue
        expected_path = run_dir / basename
        try:
            matches = artifact_path.resolve(strict=False) == expected_path.resolve(strict=False)
        except OSError:
            matches = False
        if not matches:
            target.errors.append(f"{label}.{field_name} must resolve to {basename} inside run_dir.")

    nested = validate_run_dir(run_dir)
    target.errors.extend(f"{label}.run_dir: {error}" for error in nested.errors)
    target.warnings.extend(f"{label}.run_dir: {warning}" for warning in nested.warnings)
    _validate_suite_run_scorecard_consistency(run, run_dir, target, label)
    _validate_suite_run_input_lineage_consistency(run, run_dir, target, label)

def _validate_suite_run_scorecard_consistency(
    run: dict[str, Any],
    run_dir: Path,
    target: ValidationTarget,
    label: str,
) -> None:
    scorecard_path = run_dir / "scorecard.json"
    if _path_has_symlink_component(scorecard_path, include_leaf=True) or not scorecard_path.is_file():
        return
    try:
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return
    if not isinstance(scorecard, dict):
        return
    failed_rules = [
        str(rule.get("id"))
        for rule in scorecard.get("rules", [])
        if isinstance(rule, dict) and rule.get("id") and not rule.get("passed")
    ]
    expected_fields = {
        "scenario_id": scorecard.get("scenario_id"),
        "passed": scorecard.get("passed"),
        "score": scorecard.get("score"),
        "failed_rules": failed_rules,
        "critical_failures": scorecard.get("critical_failures"),
    }
    for field_name, expected in expected_fields.items():
        if run.get(field_name) != expected:
            target.errors.append(
                f"{label}.{field_name} must match run scorecard.{field_name}: "
                f"expected {expected!r}, got {run.get(field_name)!r}."
            )

def _validate_suite_run_input_lineage_consistency(
    run: dict[str, Any],
    run_dir: Path,
    target: ValidationTarget,
    label: str,
) -> None:
    lineage_path = run_dir / "artifact_lineage.json"
    if _path_has_symlink_component(lineage_path, include_leaf=True) or not lineage_path.is_file():
        return
    try:
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return
    if not isinstance(lineage, dict) or not isinstance(lineage.get("inputs"), list):
        return
    input_hashes = {
        record.get("name"): record.get("sha256")
        for record in lineage["inputs"]
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }
    for sha_field, lineage_name in (
        ("scenario_sha256", "scenario"),
        ("trace_sha256", "source_trace"),
    ):
        expected = input_hashes.get(lineage_name)
        if _is_sha256(run.get(sha_field)) and _is_sha256(expected) and run[sha_field] != expected:
            target.errors.append(
                f"{label}.{sha_field} must match run artifact_lineage.inputs.{lineage_name}.sha256: "
                f"expected {expected!r}, got {run.get(sha_field)!r}."
            )

def _resolve_suite_summary_ref_path(value: Any, source_path: Path | None) -> Path | None:
    if not isinstance(value, str) or not value or _is_redacted_placeholder(value):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if source_path is not None:
        return source_path.parent / path
    return path

_SERVING_PROFILE_KEYS = {
    "schema_version",
    "generated_at",
    "profile_id",
    "arm",
    "provider",
    "engine",
    "endpoint",
    "model_identity",
    "capabilities",
    "artifacts",
    "eval_preflight",
    "environment",
    "adapter_strategy",
}

_SERVING_PROFILE_ENDPOINT_KEYS = {"base_url", "models_url", "chat_completions_url"}

_SERVING_PROFILE_IDENTITY_KEYS = {
    "requested_model",
    "served_model_id",
    "observed_model_ids",
    "metadata_model",
    "chat_response_model",
    "adapter",
    "adapter_strategy",
}

_SERVING_PROFILE_CAPABILITY_KEYS = {
    "health",
    "models",
    "model_metadata",
    "chat_completions",
    "streaming",
    "tool_calls",
    "structured_outputs",
}

_SERVING_READINESS_KEYS = {"ready", "readiness", "failed_checks"}

_SERVING_COMPATIBILITY_REPORT_KEYS = {
    "schema_version",
    "generated_at",
    "profile_id",
    "model",
    "served_model_id",
    "engine",
    "checks",
}

_SERVING_COMPATIBILITY_CHECKS_KEYS = {"openai_core", "streaming", "tool_calls", "structured_outputs"}

_SERVING_CAPABILITY_CHECK_KEYS = {
    "status",
    "response_ok",
    "error",
    "event_count",
    "done_seen",
    "text",
    "tool_call_count",
    "tool_calls",
    "json_parse_passed",
    "parsed",
}

_SERVING_ENDPOINT_CHECK_KEYS = {
    "schema_version",
    "generated_at",
    "passed",
    "readiness",
    "profile_id",
    "arm",
    "model",
    "served_model_id",
    "base_url",
    "checks",
    "failed_checks",
    "artifacts",
    "adapter_strategy",
}

_SERVING_ENDPOINT_CHECK_ROW_KEYS = {"id", "passed", "details"}

_SERVING_LIFECYCLE_KEYS = {
    "schema_version",
    "generated_at",
    "finished_at",
    "duration_ms",
    "profile",
    "engine",
    "arm",
    "provider",
    "model",
    "served_model_name",
    "adapter",
    "adapter_strategy",
    "endpoint",
    "launch",
    "process",
    "environment",
    "artifacts_root",
    "readiness_probe",
    "smoke_check",
    "teardown",
    "ready",
    "readiness",
    "passed",
    "artifacts",
    "errors",
    "logs",
}

_SERVING_LIFECYCLE_ENDPOINT_KEYS = {"base_url", "host", "port"}

_SERVING_LIFECYCLE_ENVIRONMENT_KEYS = {"python_version", "platform"}

_SERVING_LIFECYCLE_LOG_KEYS = {"stdout_tail", "stderr_tail"}

_SERVING_LIFECYCLE_READINESS_PROBE_KEYS = {"ready", "summary", "attempts", "url", "exit_code", "timeout_s"}

_SERVING_LIFECYCLE_SMOKE_CHECK_KEYS = {"attempted", "passed", "summary", "readiness", "failed_checks", "artifacts"}

_SERVING_LIFECYCLE_TEARDOWN_KEYS = {
    "attempted",
    "started_at",
    "already_exited",
    "exit_code_before_teardown",
    "exit_code_after_teardown",
    "terminated",
    "killed",
    "clean",
    "running_after_teardown",
}

_SERVING_DEMO_RUN_KEYS = {
    "schema_version",
    "generated_at",
    "candidate_arm",
    "same_scenario_ids",
    "scenario_sets",
    "arms",
    "claims",
    "comparisons",
    "endpoint_suite",
    "scenarios",
}

def _validate_serving_profile(profile: dict[str, Any], target: ValidationTarget) -> None:
    _validate_allowed_keys(profile, _SERVING_PROFILE_KEYS, target, "serving_profile")
    _require_equal(profile, "schema_version", SERVING_PROFILE_SCHEMA_VERSION, target, prefix="serving_profile.")
    for field_name in ("generated_at", "profile_id", "arm"):
        if not isinstance(profile.get(field_name), str) or not profile.get(field_name):
            target.errors.append(f"serving_profile.{field_name} must be a non-empty string.")

    endpoint = profile.get("endpoint") if isinstance(profile.get("endpoint"), dict) else {}
    if not endpoint:
        target.errors.append("serving_profile.endpoint must be an object.")
    if endpoint:
        _validate_allowed_keys(endpoint, _SERVING_PROFILE_ENDPOINT_KEYS, target, "serving_profile.endpoint")
    for field_name in ("base_url", "models_url", "chat_completions_url"):
        if field_name in endpoint and (not isinstance(endpoint.get(field_name), str) or not endpoint.get(field_name)):
            target.errors.append(f"serving_profile.endpoint.{field_name} must be a non-empty string when present.")

    identity = profile.get("model_identity") if isinstance(profile.get("model_identity"), dict) else {}
    if not identity:
        target.errors.append("serving_profile.model_identity must be an object.")
    if identity:
        _validate_allowed_keys(identity, _SERVING_PROFILE_IDENTITY_KEYS, target, "serving_profile.model_identity")
    for field_name in ("requested_model", "served_model_id"):
        if field_name in identity and (not isinstance(identity.get(field_name), str) or not identity.get(field_name)):
            target.errors.append(f"serving_profile.model_identity.{field_name} must be a non-empty string when present.")
    observed = identity.get("observed_model_ids")
    if observed is not None and not _is_string_list(observed):
        target.errors.append("serving_profile.model_identity.observed_model_ids must be a list of strings when present.")
    adapter = identity.get("adapter")
    if adapter is not None and not isinstance(adapter, dict):
        target.errors.append("serving_profile.model_identity.adapter must be an object when present.")
    elif isinstance(adapter, dict) and adapter.get("present") is True:
        if not isinstance(adapter.get("local"), bool):
            target.errors.append("serving_profile observed adapter identity must declare whether it is local.")
        observation_source = adapter.get("observation_source")
        if observation_source not in {"endpoint_model_metadata", "local_artifact_sha256"}:
            target.errors.append(
                "serving_profile.model_identity.adapter.observation_source must identify endpoint model metadata or a local artifact SHA-256."
            )
        if adapter.get("local") is True and observation_source != "local_artifact_sha256":
            target.errors.append(
                "serving_profile local adapter identity must use local_artifact_sha256 observation."
            )
        if adapter.get("local") is False and observation_source != "endpoint_model_metadata":
            target.errors.append(
                "serving_profile remote adapter identity must use endpoint_model_metadata observation."
            )
        if adapter.get("immutable") is not True:
            target.errors.append("serving_profile observed adapter identity must be immutable.")

    capabilities = profile.get("capabilities") if isinstance(profile.get("capabilities"), dict) else {}
    if not capabilities:
        target.errors.append("serving_profile.capabilities must be an object.")
    if capabilities:
        _validate_allowed_keys(capabilities, _SERVING_PROFILE_CAPABILITY_KEYS, target, "serving_profile.capabilities")
    for field_name in ("health", "models", "model_metadata", "chat_completions"):
        if field_name in capabilities and not isinstance(capabilities.get(field_name), bool):
            target.errors.append(f"serving_profile.capabilities.{field_name} must be a boolean when present.")
    for field_name in ("streaming", "tool_calls", "structured_outputs"):
        if capabilities.get(field_name) not in SERVING_CAPABILITY_STATUSES:
            target.errors.append(f"serving_profile.capabilities.{field_name} must be supported or not_verified.")

    eval_preflight = profile.get("eval_preflight") if isinstance(profile.get("eval_preflight"), dict) else {}
    if not eval_preflight:
        target.errors.append("serving_profile.eval_preflight must be an object.")
    if eval_preflight:
        _validate_allowed_keys(eval_preflight, _SERVING_READINESS_KEYS, target, "serving_profile.eval_preflight")
    _validate_serving_readiness(
        eval_preflight,
        target,
        "serving_profile.eval_preflight",
        ready_field="ready",
    )
    artifacts = profile.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, dict):
        target.errors.append("serving_profile.artifacts must be an object when present.")

    target.details.update(
        {
            "profile_id": profile.get("profile_id"),
            "arm": profile.get("arm"),
            "readiness": eval_preflight.get("readiness"),
            "served_model_id": identity.get("served_model_id"),
        }
    )

def _validate_serving_compatibility_report(report: dict[str, Any], target: ValidationTarget) -> None:
    _validate_allowed_keys(
        report,
        _SERVING_COMPATIBILITY_REPORT_KEYS,
        target,
        "serving_compatibility_report",
    )
    _require_equal(
        report,
        "schema_version",
        SERVING_COMPATIBILITY_REPORT_SCHEMA_VERSION,
        target,
        prefix="serving_compatibility_report.",
    )
    for field_name in ("generated_at", "profile_id", "model", "served_model_id", "engine"):
        if not isinstance(report.get(field_name), str) or not report.get(field_name):
            target.errors.append(f"serving_compatibility_report.{field_name} must be a non-empty string.")

    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    if not checks:
        target.errors.append("serving_compatibility_report.checks must be an object.")
    if checks:
        _validate_allowed_keys(
            checks,
            _SERVING_COMPATIBILITY_CHECKS_KEYS,
            target,
            "serving_compatibility_report.checks",
        )
    openai_core = checks.get("openai_core") if isinstance(checks.get("openai_core"), dict) else {}
    if not openai_core:
        target.errors.append("serving_compatibility_report.checks.openai_core must be a non-empty object.")
    for check_id, passed in openai_core.items():
        if not isinstance(check_id, str) or not check_id:
            target.errors.append("serving_compatibility_report.checks.openai_core keys must be non-empty strings.")
        if not isinstance(passed, bool):
            target.errors.append(f"serving_compatibility_report.checks.openai_core.{check_id} must be a boolean.")

    for field_name in ("streaming", "tool_calls", "structured_outputs"):
        _validate_serving_capability_check(
            checks.get(field_name),
            target,
            f"serving_compatibility_report.checks.{field_name}",
            capability=field_name,
        )

    target.details.update(
        {
            "profile_id": report.get("profile_id"),
            "model": report.get("model"),
            "served_model_id": report.get("served_model_id"),
            "engine": report.get("engine"),
            "openai_core_passed": sum(1 for passed in openai_core.values() if passed is True),
        }
    )

def _validate_serving_endpoint_check(check: dict[str, Any], target: ValidationTarget) -> None:
    _validate_allowed_keys(check, _SERVING_ENDPOINT_CHECK_KEYS, target, "serving_endpoint_check")
    _require_equal(check, "schema_version", SERVING_ENDPOINT_CHECK_SCHEMA_VERSION, target, prefix="serving_endpoint_check.")
    for field_name in ("generated_at", "profile_id", "arm", "model", "served_model_id", "base_url"):
        if not isinstance(check.get(field_name), str) or not check.get(field_name):
            target.errors.append(f"serving_endpoint_check.{field_name} must be a non-empty string.")

    checks = check.get("checks")
    if not isinstance(checks, list) or not checks:
        target.errors.append("serving_endpoint_check.checks must be a non-empty list.")
        checks = []
    failed_from_rows: list[str] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(checks):
        label = f"serving_endpoint_check.checks[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(item, _SERVING_ENDPOINT_CHECK_ROW_KEYS, target, label)
        check_id = item.get("id")
        if not isinstance(check_id, str) or not check_id:
            target.errors.append(f"{label}.id must be a non-empty string.")
        elif check_id in seen_ids:
            target.errors.append(f"serving_endpoint_check.checks has duplicate id {check_id!r}.")
        else:
            seen_ids.add(check_id)
        if not isinstance(item.get("passed"), bool):
            target.errors.append(f"{label}.passed must be a boolean.")
        elif item.get("passed") is False and isinstance(check_id, str):
            failed_from_rows.append(check_id)
        if not isinstance(item.get("details"), dict):
            target.errors.append(f"{label}.details must be an object.")

    failed_checks = check.get("failed_checks")
    if not _is_string_list(failed_checks):
        target.errors.append("serving_endpoint_check.failed_checks must be a list of strings.")
        failed_checks = []
    if failed_checks != failed_from_rows:
        target.errors.append(f"serving_endpoint_check.failed_checks expected {failed_from_rows!r}, got {failed_checks!r}.")
    if not isinstance(check.get("artifacts"), dict):
        target.errors.append("serving_endpoint_check.artifacts must be an object.")
    _validate_serving_readiness(
        check,
        target,
        "serving_endpoint_check",
        ready_field="passed",
        failed_checks=failed_checks,
    )

    target.details.update(
        {
            "profile_id": check.get("profile_id"),
            "arm": check.get("arm"),
            "readiness": check.get("readiness"),
            "passed": check.get("passed"),
            "check_count": len(checks),
            "failed_check_count": len(failed_from_rows),
        }
    )

def _validate_serving_lifecycle(lifecycle: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(lifecycle, _SERVING_LIFECYCLE_KEYS, target, "serving_lifecycle")
    _require_equal(lifecycle, "schema_version", SERVING_LIFECYCLE_SCHEMA_VERSION, target, prefix="serving_lifecycle.")
    for field_name in ("generated_at", "finished_at", "profile", "engine", "model"):
        if not isinstance(lifecycle.get(field_name), str) or not lifecycle.get(field_name):
            target.errors.append(f"serving_lifecycle.{field_name} must be a non-empty string.")
    if not _is_non_negative_int(lifecycle.get("duration_ms")):
        target.errors.append("serving_lifecycle.duration_ms must be a non-negative integer.")

    for field_name in ("endpoint", "launch", "process", "adapter_strategy"):
        if not isinstance(lifecycle.get(field_name), dict):
            target.errors.append(f"serving_lifecycle.{field_name} must be an object.")
    endpoint = lifecycle.get("endpoint") if isinstance(lifecycle.get("endpoint"), dict) else {}
    if endpoint:
        _validate_allowed_keys(endpoint, _SERVING_LIFECYCLE_ENDPOINT_KEYS, target, "serving_lifecycle.endpoint")
    environment = lifecycle.get("environment") if isinstance(lifecycle.get("environment"), dict) else {}
    if environment:
        _validate_allowed_keys(
            environment,
            _SERVING_LIFECYCLE_ENVIRONMENT_KEYS,
            target,
            "serving_lifecycle.environment",
        )
    logs = lifecycle.get("logs") if isinstance(lifecycle.get("logs"), dict) else {}
    if logs:
        _validate_allowed_keys(logs, _SERVING_LIFECYCLE_LOG_KEYS, target, "serving_lifecycle.logs")
    if not isinstance(lifecycle.get("passed"), bool):
        target.errors.append("serving_lifecycle.passed must be a boolean.")
    if not isinstance(lifecycle.get("ready"), bool):
        target.errors.append("serving_lifecycle.ready must be a boolean.")
    if lifecycle.get("readiness") not in {"ready", "blocked"}:
        target.errors.append("serving_lifecycle.readiness must be ready or blocked.")
    if not _is_string_list(lifecycle.get("errors")):
        target.errors.append("serving_lifecycle.errors must be a list of strings.")

    readiness_probe = lifecycle.get("readiness_probe") if isinstance(lifecycle.get("readiness_probe"), dict) else {}
    smoke_check = lifecycle.get("smoke_check") if isinstance(lifecycle.get("smoke_check"), dict) else {}
    teardown = lifecycle.get("teardown") if isinstance(lifecycle.get("teardown"), dict) else {}
    if not readiness_probe:
        target.errors.append("serving_lifecycle.readiness_probe must be an object.")
    if not smoke_check:
        target.errors.append("serving_lifecycle.smoke_check must be an object.")
    if not teardown:
        target.errors.append("serving_lifecycle.teardown must be an object.")

    _validate_serving_lifecycle_readiness_probe(readiness_probe, target)
    _validate_serving_lifecycle_smoke_check(smoke_check, target)
    _validate_serving_lifecycle_teardown(teardown, target)

    errors = lifecycle.get("errors") if _is_string_list(lifecycle.get("errors")) else []
    expected_passed = (
        readiness_probe.get("ready") is True
        and smoke_check.get("passed") is True
        and teardown.get("clean") is True
        and not errors
    )
    if isinstance(lifecycle.get("passed"), bool) and lifecycle["passed"] != expected_passed:
        target.errors.append("serving_lifecycle.passed must match readiness_probe, smoke_check, teardown, and errors.")
    if isinstance(lifecycle.get("ready"), bool) and lifecycle["ready"] != expected_passed:
        target.errors.append("serving_lifecycle.ready must match computed pass state.")
    expected_readiness = "ready" if expected_passed else "blocked"
    if lifecycle.get("readiness") != expected_readiness:
        target.errors.append(f"serving_lifecycle.readiness expected {expected_readiness!r}, got {lifecycle.get('readiness')!r}.")
    if expected_passed:
        _validate_serving_lifecycle_preflight_artifacts(lifecycle, target, source_path)
    elif not errors and not smoke_check.get("failed_checks"):
        target.errors.append("serving_lifecycle.errors or smoke_check.failed_checks must explain blocked readiness.")

    target.details.update(
        {
            "profile": lifecycle.get("profile"),
            "engine": lifecycle.get("engine"),
            "model": lifecycle.get("model"),
            "readiness": lifecycle.get("readiness"),
            "passed": lifecycle.get("passed"),
        }
    )

def _validate_serving_lifecycle_readiness_probe(value: dict[str, Any], target: ValidationTarget) -> None:
    label = "serving_lifecycle.readiness_probe"
    if value:
        _validate_allowed_keys(value, _SERVING_LIFECYCLE_READINESS_PROBE_KEYS, target, label)
    if value and not isinstance(value.get("ready"), bool):
        target.errors.append(f"{label}.ready must be a boolean.")
    if value and (not isinstance(value.get("summary"), str) or not value.get("summary")):
        target.errors.append(f"{label}.summary must be a non-empty string.")
    attempts = value.get("attempts") if value else None
    if not isinstance(attempts, list):
        target.errors.append(f"{label}.attempts must be a list.")
        attempts = []
    if value.get("ready") is True and not attempts:
        target.errors.append(f"{label}.ready requires at least one readiness attempt.")

def _validate_serving_lifecycle_smoke_check(value: dict[str, Any], target: ValidationTarget) -> None:
    label = "serving_lifecycle.smoke_check"
    if value:
        _validate_allowed_keys(value, _SERVING_LIFECYCLE_SMOKE_CHECK_KEYS, target, label)
    if value and not isinstance(value.get("attempted"), bool):
        target.errors.append(f"{label}.attempted must be a boolean.")
    if value and not isinstance(value.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    if value.get("passed") is True and value.get("attempted") is not True:
        target.errors.append(f"{label}.passed requires attempted true.")
    if value.get("attempted") is True:
        _validate_serving_readiness(value, target, label, ready_field="passed")
        artifacts = value.get("artifacts")
        if not isinstance(artifacts, dict):
            target.errors.append(f"{label}.artifacts must be an object when attempted is true.")

def _validate_serving_lifecycle_teardown(value: dict[str, Any], target: ValidationTarget) -> None:
    label = "serving_lifecycle.teardown"
    if value:
        _validate_allowed_keys(value, _SERVING_LIFECYCLE_TEARDOWN_KEYS, target, label)
    for field_name in ("attempted", "clean", "running_after_teardown"):
        if value and not isinstance(value.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if value.get("clean") is True and value.get("running_after_teardown") is True:
        target.errors.append(f"{label}.clean cannot be true when running_after_teardown is true.")

def _validate_serving_lifecycle_preflight_artifacts(
    lifecycle: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    artifacts = lifecycle.get("artifacts") if isinstance(lifecycle.get("artifacts"), dict) else {}
    if not artifacts:
        target.errors.append("serving_lifecycle.artifacts must be an object.")
    smoke_check = lifecycle.get("smoke_check") if isinstance(lifecycle.get("smoke_check"), dict) else {}
    smoke_artifacts = smoke_check.get("artifacts") if isinstance(smoke_check.get("artifacts"), dict) else {}
    for role in SERVING_LIFECYCLE_PREFLIGHT_ARTIFACTS:
        value = artifacts.get(role)
        if not isinstance(value, str) or not value:
            target.errors.append(f"serving_lifecycle.artifacts.{role} must be a non-empty string when passed.")
            continue
        if smoke_artifacts and smoke_artifacts.get(role) != value:
            target.errors.append(f"serving_lifecycle.artifacts.{role} must match smoke_check.artifacts.{role}.")
        artifact_path = _resolve_serving_lifecycle_artifact_path(value, source_path)
        if artifact_path is None or not artifact_path.is_file():
            target.errors.append(f"serving_lifecycle.artifacts.{role} must point at an existing file when passed.")
        elif _path_has_symlink_component(artifact_path, include_leaf=True):
            target.errors.append(f"serving_lifecycle.artifacts.{role} must point at a regular non-symlink file when passed.")

def _resolve_serving_lifecycle_artifact_path(value: Any, source_path: Path) -> Path | None:
    if not isinstance(value, str) or not value or value.startswith("<redacted:"):
        return None
    path = Path(value)
    return path if path.is_absolute() else source_path.parent / path

def _validate_serving_demo_run(demo: dict[str, Any], target: ValidationTarget) -> None:
    _validate_allowed_keys(demo, _SERVING_DEMO_RUN_KEYS, target, "serving_demo_run")
    _require_equal(demo, "schema_version", SERVING_DEMO_RUN_SCHEMA_VERSION, target, prefix="serving_demo_run.")
    candidate_arm = demo.get("candidate_arm")
    if not isinstance(candidate_arm, str) or not candidate_arm:
        target.errors.append("serving_demo_run.candidate_arm must be a non-empty string.")
    if not isinstance(demo.get("same_scenario_ids"), bool):
        target.errors.append("serving_demo_run.same_scenario_ids must be a boolean.")

    arms = demo.get("arms")
    if not isinstance(arms, list):
        target.errors.append("serving_demo_run.arms must be a list.")
        arms = []
    if len(arms) < 2:
        target.errors.append("serving_demo_run.arms must include at least two arms.")
    arm_names: list[str] = []
    for index, arm in enumerate(arms):
        if not isinstance(arm, dict):
            target.errors.append(f"serving_demo_run.arms[{index}] must be an object.")
            continue
        name = arm.get("name")
        if not isinstance(name, str) or not name:
            target.errors.append(f"serving_demo_run.arms[{index}].name must be a non-empty string.")
        elif name in arm_names:
            target.errors.append(f"serving_demo_run.arms has duplicate name {name!r}.")
        else:
            arm_names.append(name)
        if not isinstance(arm.get("source"), str) or not arm.get("source"):
            target.errors.append(f"serving_demo_run.arms[{index}].source must be a non-empty string.")
        if arm.get("serving_profile") is not None and not isinstance(arm.get("serving_profile"), str):
            target.errors.append(f"serving_demo_run.arms[{index}].serving_profile must be a string or null.")
        metrics = arm.get("metrics")
        if not isinstance(metrics, dict):
            target.errors.append(f"serving_demo_run.arms[{index}].metrics must be an object.")
        else:
            _validate_serving_demo_metrics(metrics, index, target)
    if isinstance(candidate_arm, str) and candidate_arm and candidate_arm not in arm_names:
        target.errors.append("serving_demo_run.candidate_arm must match one arm name.")

    endpoint_suite = demo.get("endpoint_suite")
    if endpoint_suite is not None:
        _validate_serving_demo_endpoint_suite(endpoint_suite, set(arm_names), target)

    scenario_sets = demo.get("scenario_sets")
    if not isinstance(scenario_sets, dict):
        target.errors.append("serving_demo_run.scenario_sets must be an object.")
        scenario_sets = {}
    normalized_sets: dict[str, list[str]] = {}
    for arm_name in arm_names:
        values = scenario_sets.get(arm_name)
        if not _is_string_list(values):
            target.errors.append(f"serving_demo_run.scenario_sets.{arm_name} must be a list of strings.")
            normalized_sets[arm_name] = []
        else:
            normalized_sets[arm_name] = list(values)
    expected_same = bool(normalized_sets) and len({tuple(values) for values in normalized_sets.values()}) == 1
    if isinstance(demo.get("same_scenario_ids"), bool) and demo.get("same_scenario_ids") != expected_same:
        target.errors.append(f"serving_demo_run.same_scenario_ids expected {expected_same}, got {demo.get('same_scenario_ids')!r}.")

    claims = demo.get("claims")
    if not isinstance(claims, list):
        target.errors.append("serving_demo_run.claims must be a list.")
        claims = []
    for index, claim in enumerate(claims):
        _validate_serving_demo_claim(claim, index, target, set(arm_names))
    if expected_same is False and any(isinstance(claim, dict) and claim.get("id") != "scenario_sets_differ" for claim in claims):
        target.errors.append("serving_demo_run.claims must not include behavior claims when scenario sets differ.")

    comparisons = demo.get("comparisons", [])
    if not isinstance(comparisons, list):
        target.errors.append("serving_demo_run.comparisons must be a list when present.")
        comparisons = []
    for index, comparison in enumerate(comparisons):
        _validate_serving_demo_comparison(comparison, index, target, set(arm_names), candidate_arm)

    scenarios = demo.get("scenarios")
    if not isinstance(scenarios, list):
        target.errors.append("serving_demo_run.scenarios must be a list.")
        scenarios = []
    seen_scenarios: set[str] = set()
    for index, scenario in enumerate(scenarios):
        _validate_serving_demo_scenario(
            scenario,
            index,
            target,
            set(arm_names),
            require_all_arms=expected_same,
            seen_scenarios=seen_scenarios,
        )
    _validate_serving_demo_arm_metrics_against_scenarios(arms, scenarios, target)

    target.details.update(
        {
            "candidate_arm": candidate_arm,
            "arm_count": len(arms),
            "comparison_count": len(comparisons),
            "claim_count": len(claims),
            "scenario_count": len(scenarios),
            "same_scenario_ids": demo.get("same_scenario_ids"),
        }
    )

def _validate_serving_demo_endpoint_suite(value: Any, arm_names: set[str], target: ValidationTarget) -> None:
    label = "serving_demo_run.endpoint_suite"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object when present.")
        return
    if value.get("schema_version") != "hfr.serving_endpoint_suite.v1":
        target.errors.append(f"{label}.schema_version must be 'hfr.serving_endpoint_suite.v1'.")
    if not isinstance(value.get("path"), str) or not value.get("path"):
        target.errors.append(f"{label}.path must be a non-empty string.")
    if not isinstance(value.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    if not _is_string_list(value.get("failed_checks")):
        target.errors.append(f"{label}.failed_checks must be a list of strings.")
    if not isinstance(value.get("requirements"), dict):
        target.errors.append(f"{label}.requirements must be an object.")
    arms = value.get("arms")
    if not isinstance(arms, list):
        target.errors.append(f"{label}.arms must be a list.")
        arms = []
    suite_arm_names: set[str] = set()
    for index, arm in enumerate(arms):
        arm_label = f"{label}.arms[{index}]"
        if not isinstance(arm, dict):
            target.errors.append(f"{arm_label} must be an object.")
            continue
        name = arm.get("arm")
        if not isinstance(name, str) or not name:
            target.errors.append(f"{arm_label}.arm must be a non-empty string.")
        else:
            suite_arm_names.add(name)
        if not isinstance(arm.get("ready_for_eval"), bool):
            target.errors.append(f"{arm_label}.ready_for_eval must be a boolean.")
        if not _is_string_list(arm.get("failed_checks")):
            target.errors.append(f"{arm_label}.failed_checks must be a list of strings.")
        for field_name in ("profile_path", "served_model_id", "requested_model", "lifecycle_path"):
            if arm.get(field_name) is not None and not isinstance(arm.get(field_name), str):
                target.errors.append(f"{arm_label}.{field_name} must be a string or null.")
    missing_arms = sorted(arm_names - suite_arm_names)
    if missing_arms:
        target.errors.append(f"{label}.arms is missing demo arm(s): {missing_arms}.")
    alignment = value.get("demo_alignment")
    if not isinstance(alignment, dict):
        target.errors.append(f"{label}.demo_alignment must be an object.")
        return
    if not isinstance(alignment.get("passed"), bool):
        target.errors.append(f"{label}.demo_alignment.passed must be a boolean.")
    if not _is_string_list(alignment.get("failed_checks")):
        target.errors.append(f"{label}.demo_alignment.failed_checks must be a list of strings.")
    if not isinstance(alignment.get("checks"), list):
        target.errors.append(f"{label}.demo_alignment.checks must be a list.")

def _validate_serving_readiness(
    value: dict[str, Any],
    target: ValidationTarget,
    label: str,
    *,
    ready_field: str,
    failed_checks: list[str] | None = None,
) -> None:
    if not isinstance(value.get(ready_field), bool):
        target.errors.append(f"{label}.{ready_field} must be a boolean.")
    if value.get("readiness") not in {"ready", "blocked"}:
        target.errors.append(f"{label}.readiness must be ready or blocked.")
    raw_failed = value.get("failed_checks") if failed_checks is None else failed_checks
    if not _is_string_list(raw_failed):
        target.errors.append(f"{label}.failed_checks must be a list of strings.")
        raw_failed = []
    if value.get(ready_field) is True:
        if value.get("readiness") != "ready":
            target.errors.append(f"{label}.{ready_field} requires readiness ready.")
        if raw_failed:
            target.errors.append(f"{label}.{ready_field} requires an empty failed_checks list.")
    if value.get(ready_field) is False and value.get("readiness") == "ready":
        target.errors.append(f"{label}.readiness cannot be ready when {ready_field} is false.")
    if value.get("readiness") == "blocked" and not raw_failed:
        target.errors.append(f"{label}.failed_checks must explain blocked readiness.")

def _validate_serving_capability_check(
    value: Any,
    target: ValidationTarget,
    label: str,
    *,
    capability: str,
) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _SERVING_CAPABILITY_CHECK_KEYS, target, label)
    if value.get("status") not in SERVING_CAPABILITY_STATUSES:
        target.errors.append(f"{label}.status must be supported or not_verified.")
    if not isinstance(value.get("response_ok"), bool):
        target.errors.append(f"{label}.response_ok must be a boolean.")
    if value.get("status") == "supported" and value.get("response_ok") is not True:
        target.errors.append(f"{label}.response_ok must be true when status is supported.")
    if value.get("response_ok") is True and value.get("status") != "supported":
        target.errors.append(f"{label}.status must be supported when response_ok is true.")
    if "error" in value and value.get("error") is not None and not isinstance(value.get("error"), str):
        target.errors.append(f"{label}.error must be a string or null when present.")
    error_value = value.get("error")
    if value.get("status") == "supported" and error_value is not None and error_value != "":
        target.errors.append(f"{label}.error must be empty when status is supported.")

    if capability == "streaming":
        event_count = value.get("event_count")
        done_seen = value.get("done_seen")
        text = value.get("text")
        if not _is_non_negative_int(event_count):
            target.errors.append(f"{label}.event_count must be a non-negative integer.")
            event_count = 0
        if not isinstance(done_seen, bool):
            target.errors.append(f"{label}.done_seen must be a boolean.")
        if not isinstance(text, str):
            target.errors.append(f"{label}.text must be a string.")
            text = ""
        evidence_supported = (
            value.get("response_ok") is True
            and done_seen is True
            and event_count > 0
            and bool(text)
        )
        if value.get("status") == "supported" and done_seen is not True:
            target.errors.append(f"{label}.done_seen must be true when status is supported.")
    elif capability == "tool_calls":
        tool_call_count = value.get("tool_call_count")
        tool_calls = value.get("tool_calls")
        if not _is_non_negative_int(tool_call_count):
            target.errors.append(f"{label}.tool_call_count must be a non-negative integer.")
            tool_call_count = 0
        if not isinstance(tool_calls, list):
            target.errors.append(f"{label}.tool_calls must be a list.")
            tool_calls = []
        if _is_non_negative_int(tool_call_count) and tool_call_count != len(tool_calls):
            target.errors.append(f"{label}.tool_call_count must match tool_calls length.")
        evidence_supported = value.get("response_ok") is True and tool_call_count > 0
    elif capability == "structured_outputs":
        json_parse_passed = value.get("json_parse_passed")
        parsed = value.get("parsed")
        text = value.get("text")
        if not isinstance(json_parse_passed, bool):
            target.errors.append(f"{label}.json_parse_passed must be a boolean.")
        if parsed is not None and not isinstance(parsed, dict):
            target.errors.append(f"{label}.parsed must be an object or null.")
        if not isinstance(text, str):
            target.errors.append(f"{label}.text must be a string.")
        if isinstance(json_parse_passed, bool) and json_parse_passed != isinstance(parsed, dict):
            target.errors.append(f"{label}.json_parse_passed must match whether parsed is an object.")
        evidence_supported = (
            value.get("response_ok") is True
            and json_parse_passed is True
            and isinstance(parsed, dict)
        )
    else:  # pragma: no cover - internal callers use the three declared capabilities
        target.errors.append(f"{label} has unsupported capability type {capability!r}.")
        evidence_supported = False

    expected_status = "supported" if evidence_supported else "not_verified"
    if value.get("status") != expected_status:
        target.errors.append(
            f"{label}.status must replay to {expected_status!r} from its capability evidence."
        )

def _validate_serving_demo_metrics(metrics: dict[str, Any], index: int, target: ValidationTarget) -> None:
    label = f"serving_demo_run.arms[{index}].metrics"
    for field_name in ("total", "passed", "failed", "critical_failure_total"):
        if field_name in metrics and not _is_non_negative_int(metrics.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer when present.")
    if "pass_rate" in metrics and not _is_optional_rate(metrics.get("pass_rate")):
        target.errors.append(f"{label}.pass_rate must be null or a number from 0.0 to 1.0.")
    if "average_score" in metrics and not _is_optional_non_negative_number(metrics.get("average_score")):
        target.errors.append(f"{label}.average_score must be null or a non-negative number.")
    if _is_non_negative_int(metrics.get("passed")) and _is_non_negative_int(metrics.get("failed")):
        expected_total = metrics.get("passed") + metrics.get("failed")
        if metrics.get("total") is not None and metrics.get("total") != expected_total:
            target.errors.append(f"{label}.total expected {expected_total}, got {metrics.get('total')!r}.")

def _validate_serving_demo_arm_metrics_against_scenarios(
    arms: list[Any],
    scenarios: list[Any],
    target: ValidationTarget,
) -> None:
    expected_by_arm = _serving_demo_expected_arm_metrics(scenarios)
    for index, arm in enumerate(arms):
        if not isinstance(arm, dict):
            continue
        name = arm.get("name")
        metrics = arm.get("metrics")
        if not isinstance(name, str) or not name or not isinstance(metrics, dict):
            continue
        expected = expected_by_arm.get(
            name,
            {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "average_score": 0.0,
                "critical_failure_total": 0,
            },
        )
        label = f"serving_demo_run.arms[{index}].metrics"
        for field_name in ("total", "passed", "failed", "critical_failure_total"):
            if _is_non_negative_int(metrics.get(field_name)) and metrics.get(field_name) != expected[field_name]:
                target.errors.append(f"{label}.{field_name} expected {expected[field_name]!r}, got {metrics.get(field_name)!r}.")
        for field_name in ("pass_rate", "average_score"):
            if _is_optional_non_negative_number(metrics.get(field_name)) and metrics.get(field_name) is not None:
                if metrics.get(field_name) != expected[field_name]:
                    target.errors.append(f"{label}.{field_name} expected {expected[field_name]!r}, got {metrics.get(field_name)!r}.")

def _serving_demo_expected_arm_metrics(scenarios: list[Any]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        runs = scenario.get("arms")
        if not isinstance(runs, dict):
            continue
        for arm_name, run in runs.items():
            if not isinstance(arm_name, str) or not isinstance(run, dict):
                continue
            bucket = buckets.setdefault(
                arm_name,
                {"total": 0, "passed": 0, "failed": 0, "scores": [], "critical_failure_total": 0},
            )
            bucket["total"] += 1
            if run.get("passed") is True:
                bucket["passed"] += 1
            elif run.get("passed") is False:
                bucket["failed"] += 1
            if _is_number_between(run.get("score"), 0, float("inf")):
                bucket["scores"].append(_number_value(run.get("score")))
            if isinstance(run.get("critical_failures"), list):
                bucket["critical_failure_total"] += len(run["critical_failures"])
    expected: dict[str, dict[str, Any]] = {}
    for arm_name, bucket in buckets.items():
        total = int(bucket["total"])
        scores = bucket["scores"]
        expected[arm_name] = {
            "total": total,
            "passed": bucket["passed"],
            "failed": bucket["failed"],
            "pass_rate": round(bucket["passed"] / total, 4) if total else 0.0,
            "average_score": _average_number(scores),
            "critical_failure_total": bucket["critical_failure_total"],
        }
    return expected

def _validate_serving_demo_comparison(
    comparison: Any,
    index: int,
    target: ValidationTarget,
    arm_names: set[str],
    candidate_arm: Any,
) -> None:
    label = f"serving_demo_run.comparisons[{index}]"
    if not isinstance(comparison, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if comparison.get("candidate_arm") != candidate_arm:
        target.errors.append(f"{label}.candidate_arm must match serving_demo_run.candidate_arm.")
    reference_arm = comparison.get("reference_arm")
    if reference_arm not in arm_names:
        target.errors.append(f"{label}.reference_arm must match a demo arm.")
    if reference_arm == candidate_arm:
        target.errors.append(f"{label}.reference_arm must not equal candidate_arm.")
    if not isinstance(comparison.get("same_scenario_ids"), bool):
        target.errors.append(f"{label}.same_scenario_ids must be a boolean.")
    deltas = comparison.get("metric_deltas")
    if not isinstance(deltas, dict):
        target.errors.append(f"{label}.metric_deltas must be an object.")
        deltas = {}
    for field_name in ("pass_rate", "average_score", "passed", "failed", "critical_failure_total"):
        if field_name in deltas and not _is_optional_number(deltas.get(field_name)):
            target.errors.append(f"{label}.metric_deltas.{field_name} must be a number or null.")
    outcomes = comparison.get("scenario_outcomes")
    if not isinstance(outcomes, list):
        target.errors.append(f"{label}.scenario_outcomes must be a list.")
        return
    for outcome_index, outcome in enumerate(outcomes):
        _validate_serving_demo_comparison_outcome(outcome, outcome_index, target, label)

def _validate_serving_demo_comparison_outcome(outcome: Any, index: int, target: ValidationTarget, parent_label: str) -> None:
    label = f"{parent_label}.scenario_outcomes[{index}]"
    if not isinstance(outcome, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if not isinstance(outcome.get("scenario_id"), str) or not outcome.get("scenario_id"):
        target.errors.append(f"{label}.scenario_id must be a non-empty string.")
    for field_name in ("candidate_passed", "reference_passed"):
        if outcome.get(field_name) is not None and not isinstance(outcome.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean or null.")
    for field_name in ("candidate_score", "reference_score", "score_delta"):
        if not _is_optional_number(outcome.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a number or null.")
    if outcome.get("outcome") not in {"candidate_repaired", "candidate_regressed", "both_passed", "both_failed", "missing_candidate", "missing_reference"}:
        target.errors.append(f"{label}.outcome has an unsupported value.")

def _validate_serving_demo_claim(claim: Any, index: int, target: ValidationTarget, arm_names: set[str]) -> None:
    label = f"serving_demo_run.claims[{index}]"
    if not isinstance(claim, dict):
        target.errors.append(f"{label} must be an object.")
        return
    for field_name in ("id", "summary"):
        if not isinstance(claim.get(field_name), str) or not claim.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    evidence = claim.get("evidence")
    if not isinstance(evidence, list):
        target.errors.append(f"{label}.evidence must be a list.")
        return
    if claim.get("id") != "scenario_sets_differ" and not evidence:
        target.errors.append(f"{label}.evidence must not be empty for behavior claims.")
    for evidence_index, item in enumerate(evidence):
        evidence_label = f"{label}.evidence[{evidence_index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{evidence_label} must be an object.")
            continue
        if item.get("arm") not in arm_names:
            target.errors.append(f"{evidence_label}.arm must match a demo arm.")
        if not isinstance(item.get("scenario_id"), str) or not item.get("scenario_id"):
            target.errors.append(f"{evidence_label}.scenario_id must be a non-empty string.")
        for field_name in ("evaluation_summary", "suite_summary", "trace_path", "scorecard", "run_digest", "report"):
            if field_name in item and item.get(field_name) is not None and not isinstance(item.get(field_name), str):
                target.errors.append(f"{evidence_label}.{field_name} must be a string when present.")

def _validate_serving_demo_scenario(
    scenario: Any,
    index: int,
    target: ValidationTarget,
    arm_names: set[str],
    *,
    require_all_arms: bool,
    seen_scenarios: set[str],
) -> None:
    label = f"serving_demo_run.scenarios[{index}]"
    if not isinstance(scenario, dict):
        target.errors.append(f"{label} must be an object.")
        return
    scenario_id = scenario.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        target.errors.append(f"{label}.scenario_id must be a non-empty string.")
    elif scenario_id in seen_scenarios:
        target.errors.append(f"serving_demo_run.scenarios has duplicate scenario_id {scenario_id!r}.")
    else:
        seen_scenarios.add(scenario_id)
    runs = scenario.get("arms")
    if not isinstance(runs, dict):
        target.errors.append(f"{label}.arms must be an object.")
        return
    unknown_arms = sorted(set(runs) - arm_names)
    if unknown_arms:
        target.errors.append(f"{label}.arms contains unknown arm(s): {', '.join(unknown_arms)}.")
    if require_all_arms:
        missing_arms = sorted(arm_names - set(runs))
        if missing_arms:
            target.errors.append(f"{label}.arms missing arm(s): {', '.join(missing_arms)}.")
    for arm_name, run in runs.items():
        run_label = f"{label}.arms.{arm_name}"
        if not isinstance(run, dict):
            target.errors.append(f"{run_label} must be an object.")
            continue
        if run.get("scenario_id") != scenario_id:
            target.errors.append(f"{run_label}.scenario_id must match {scenario_id!r}.")
        if not isinstance(run.get("passed"), bool):
            target.errors.append(f"{run_label}.passed must be a boolean.")
        if "score" in run and run.get("score") is not None and not _is_optional_non_negative_number(run.get("score")):
            target.errors.append(f"{run_label}.score must be null or a non-negative number.")
        if not _is_string_list(run.get("critical_failures")):
            target.errors.append(f"{run_label}.critical_failures must be a list of strings.")
        for field_name in ("trace_path", "scorecard", "run_digest", "report"):
            if field_name in run and run.get(field_name) is not None and not isinstance(run.get(field_name), str):
                target.errors.append(f"{run_label}.{field_name} must be a string when present.")

_EVAL_SUMMARY_KEYS = {
    "schema_version",
    "generated_at",
    "passed",
    "governance_ready",
    "arm_count",
    "comparison_count",
    "gate_count",
    "external_adapter_plan_count",
    "external_adapter_result_count",
    "heldout_scenarios",
    "arms",
    "comparisons",
    "compare_gates",
    "external_adapter_plans",
    "external_adapter_results",
    "repair_curriculum",
    "serving_preflight",
    "risks",
    "conclusion",
}

_EVAL_SUMMARY_ARM_KEYS = {
    "label",
    "path",
    "sha256",
    "size_bytes",
    "schema_version",
    "scenario_count",
    "scenario_ids",
    "total",
    "passed",
    "failed",
    "error_count",
    "pass_rate",
    "average_score",
    "failed_rule_counts",
    "critical_failure_counts",
    "operational_metrics",
    "serving_preflight",
    "validation",
    "source_validation",
    "blocking_reasons",
}

_EVAL_SUMMARY_OPERATIONAL_KEYS = {"cost", "latency", "tokens", "task_completion"}

_EVAL_SUMMARY_METRIC_BASE_KEYS = {"source", "known_run_count", "missing_run_count"}

_EVAL_SUMMARY_COST_KEYS = _EVAL_SUMMARY_METRIC_BASE_KEYS | {"total_usd"}

_EVAL_SUMMARY_LATENCY_KEYS = _EVAL_SUMMARY_METRIC_BASE_KEYS | {"average_ms", "p50_ms", "p95_ms", "max_ms"}

_EVAL_SUMMARY_TOKENS_KEYS = _EVAL_SUMMARY_METRIC_BASE_KEYS | {"prompt_tokens", "completion_tokens", "total_tokens"}

_EVAL_SUMMARY_TASK_COMPLETION_KEYS = {
    "source",
    "configured_count",
    "complete_count",
    "incomplete_count",
    "not_applicable_count",
    "unknown_count",
    "passed_count",
    "failed_count",
    "pass_rate",
}

_EVAL_SUMMARY_HELDOUT_KEYS = {
    "status",
    "identical",
    "cross_arm_claims_allowed",
    "scenario_count",
    "scenario_ids",
    "arms",
    "mismatches",
    "blocking_reasons",
}

_EVAL_SUMMARY_HELDOUT_ARM_KEYS = {"label", "scenario_ids", "scenario_count"}

_EVAL_SUMMARY_HELDOUT_MISMATCH_KEYS = {"label", "missing_from_arm", "extra_in_arm"}

_EVAL_SUMMARY_COMPARISON_KEYS = {
    "label",
    "path",
    "manifest",
    "manifest_sha256",
    "manifest_size_bytes",
    "schema_version",
    "claims_allowed",
    "passed",
    "blocking_reasons",
    "raw_movement",
    "governance_claims",
}

_EVAL_SUMMARY_RAW_MOVEMENT_KEYS = {
    "pair_count",
    "candidate_win_count",
    "baseline_win_count",
    "candidate_win_scenarios",
    "baseline_win_scenarios",
    "task_completion_improvement_count",
    "task_completion_regression_count",
    "task_completion_improvement_scenarios",
    "task_completion_regression_scenarios",
    "fixed_rule_counts",
    "regressed_rule_counts",
    "new_critical_failure_counts",
    "contract_drift_count",
    "unverified_contract_count",
    "skipped_pair_count",
    "missing_in_candidate",
    "new_in_candidate",
}

_EVAL_SUMMARY_GOVERNANCE_CLAIMS_KEYS = {
    "candidate_win_count",
    "candidate_win_scenarios",
    "task_completion_improvement_count",
    "task_completion_improvement_scenarios",
    "suppressed_raw_claims",
    "suppression_reasons",
}

_EVAL_SUMMARY_GATE_KEYS = {
    "label",
    "path",
    "sha256",
    "size_bytes",
    "schema_version",
    "passed",
    "check_count",
    "failed_check_count",
    "failed_checks",
    "blocking_reasons",
}

_EVAL_SUMMARY_GATE_FAILED_CHECK_KEYS = {"id", "summary", "scope"}

_EVAL_SUMMARY_EXTERNAL_ADAPTER_KEYS = {
    "label",
    "path",
    "sha256",
    "size_bytes",
    "schema_version",
    "ready",
    "adapter_count",
    "ready_adapter_count",
    "selected_adapters",
    "blocking_reasons",
}

_EVAL_SUMMARY_EXTERNAL_RESULT_KEYS = {
    "label",
    "path",
    "sha256",
    "size_bytes",
    "schema_version",
    "adapter_id",
    "model_id",
    "source_plan_sha256",
    "heldout_manifest_sha256",
    "integrity_passed",
    "execution_status",
    "benchmark_status",
    "coverage_complete",
    "governance_readiness",
    "external_eval_claims_allowed",
    "blocking_reasons",
}

_EVAL_SUMMARY_REPAIR_KEYS = {
    "work_item_count",
    "critical_work_item_count",
    "priority_counts",
    "category_counts",
    "items",
    "notes",
}

_EVAL_SUMMARY_WORK_ITEM_KEYS = {
    "work_item_id",
    "category",
    "priority",
    "source",
    "label",
    "reason",
    "summary",
    "suggested_action",
    "scenario_id",
    "rule_id",
    "count",
}

_EVAL_SUMMARY_RISK_KEYS = {"source", "label", "reason"}

_EVAL_SUMMARY_CONCLUSION_KEYS = {"status", "recommendation", "risk_count"}

_EVAL_SUMMARY_SERVING_PREFLIGHT_KEYS = {
    "provided",
    "required",
    "path",
    "sha256",
    "size_bytes",
    "schema_version",
    "passed",
    "readiness",
    "profile_id",
    "model",
    "served_model_id",
    "base_url",
    "failed_checks",
    "artifacts",
    "blocking_reasons",
}

_EVAL_SUMMARY_SERVING_PREFLIGHT_SUMMARY_KEYS = {
    "required",
    "input_count",
    "attached_count",
    "unmatched_labels",
    "duplicate_labels",
    "blocking_reasons",
}

def _validate_eval_summary(summary: dict[str, Any], target: ValidationTarget, *, source_path: Path | None = None) -> None:
    _validate_allowed_keys(summary, _EVAL_SUMMARY_KEYS, target, "eval_summary")
    _require_equal(summary, "schema_version", EVAL_SUMMARY_SCHEMA_VERSION, target)
    source_dir = source_path.parent if source_path is not None else None
    for field_name in ("passed", "governance_ready"):
        if not isinstance(summary.get(field_name), bool):
            target.errors.append(f"eval_summary.{field_name} must be a boolean.")
    if isinstance(summary.get("passed"), bool) and summary.get("governance_ready") != summary.get("passed"):
        target.errors.append("eval_summary.governance_ready must match eval_summary.passed.")

    arms = summary.get("arms")
    if not isinstance(arms, list):
        target.errors.append("eval_summary.arms must be a list.")
        arms = []
    comparisons = summary.get("comparisons")
    if not isinstance(comparisons, list):
        target.errors.append("eval_summary.comparisons must be a list.")
        comparisons = []
    gates = summary.get("compare_gates")
    if not isinstance(gates, list):
        target.errors.append("eval_summary.compare_gates must be a list.")
        gates = []
    adapters = summary.get("external_adapter_plans")
    if not isinstance(adapters, list):
        target.errors.append("eval_summary.external_adapter_plans must be a list.")
        adapters = []
    external_results = summary.get("external_adapter_results")
    if not isinstance(external_results, list):
        target.errors.append("eval_summary.external_adapter_results must be a list.")
        external_results = []
    repair_curriculum = summary.get("repair_curriculum")
    if not isinstance(repair_curriculum, dict):
        target.errors.append("eval_summary.repair_curriculum must be an object.")
        repair_curriculum = {}
    risks = summary.get("risks")
    if not isinstance(risks, list):
        target.errors.append("eval_summary.risks must be a list.")
        risks = []
    serving_preflight = summary.get("serving_preflight") if isinstance(summary.get("serving_preflight"), dict) else None
    serving_required = False
    if "serving_preflight" in summary:
        if serving_preflight is None:
            target.errors.append("eval_summary.serving_preflight must be an object.")
        else:
            _validate_allowed_keys(
                serving_preflight,
                _EVAL_SUMMARY_SERVING_PREFLIGHT_SUMMARY_KEYS,
                target,
                "eval_summary.serving_preflight",
            )
            serving_required = serving_preflight.get("required") is True
            _validate_eval_summary_serving_preflight_summary(serving_preflight, target)

    expected_counts = {
        "arm_count": len(arms),
        "comparison_count": len(comparisons),
        "gate_count": len(gates),
        "external_adapter_plan_count": len(adapters),
        "external_adapter_result_count": len(external_results),
    }
    for field_name, expected in expected_counts.items():
        if summary.get(field_name) != expected:
            target.errors.append(f"eval_summary.{field_name} expected {expected}, got {summary.get(field_name)!r}.")

    heldout = summary.get("heldout_scenarios")
    if not isinstance(heldout, dict):
        target.errors.append("eval_summary.heldout_scenarios must be an object.")
        heldout = {}
    _validate_eval_summary_heldout(heldout, target, bool(comparisons))

    replayed_arms: list[dict[str, Any]] = []
    for index, arm in enumerate(arms):
        replayed_arm = _validate_eval_summary_arm(
            arm,
            index,
            target,
            serving_required=serving_required,
            source_dir=source_dir,
        )
        if replayed_arm is not None:
            replayed_arms.append(replayed_arm)
    if len(replayed_arms) == len(arms):
        _validate_eval_summary_heldout_replay(heldout, replayed_arms, target)
    if serving_preflight is not None:
        _validate_eval_summary_serving_preflight_consistency(serving_preflight, arms, target)
    for index, comparison in enumerate(comparisons):
        _validate_eval_summary_comparison(comparison, index, target, heldout, source_dir=source_dir)
    for index, gate in enumerate(gates):
        _validate_eval_summary_gate(gate, index, target, source_dir=source_dir)
    for index, adapter in enumerate(adapters):
        _validate_eval_summary_external_adapter(adapter, index, target, source_dir=source_dir)
    for index, result in enumerate(external_results):
        _validate_eval_summary_external_result(result, index, target, source_dir=source_dir)
    _validate_eval_summary_repair_curriculum(repair_curriculum, target)
    for index, risk in enumerate(risks):
        if not isinstance(risk, dict):
            target.errors.append(f"eval_summary.risks[{index}] must be an object.")
            continue
        _validate_allowed_keys(risk, _EVAL_SUMMARY_RISK_KEYS, target, f"eval_summary.risks[{index}]")
        for field_name in ("source", "reason"):
            if not isinstance(risk.get(field_name), str) or not risk.get(field_name):
                target.errors.append(f"eval_summary.risks[{index}].{field_name} must be a non-empty string.")

    expected_external_risks = _build_eval_summary_external_result_risks(adapters, external_results)
    expected_external_risks.extend(
        {
            "source": "external_eval_result",
            "label": str(result.get("label") or ""),
            "reason": reason,
        }
        for result in external_results
        if isinstance(result, dict)
        for reason in result.get("blocking_reasons", [])
        if isinstance(reason, str)
    )
    expected_external_risk_keys = {
        (str(risk.get("label") or ""), str(risk.get("reason") or ""))
        for risk in expected_external_risks
    }
    actual_external_risk_keys = {
        (str(risk.get("label") or ""), str(risk.get("reason") or ""))
        for risk in risks
        if isinstance(risk, dict) and risk.get("source") == "external_eval_result"
    }
    if actual_external_risk_keys != expected_external_risk_keys:
        target.errors.append(
            "eval_summary external_eval_result risks must match result readiness and plan/result associations."
        )

    has_failed_child = any(isinstance(item, dict) and item.get("passed") is False for item in [*comparisons, *gates])
    has_blocked_adapter = any(isinstance(item, dict) and item.get("ready") is False for item in adapters)
    has_blocked_external_result = any(
        isinstance(item, dict) and item.get("blocking_reasons") for item in external_results
    )
    has_arm_blockers = any(isinstance(item, dict) and item.get("blocking_reasons") for item in arms)
    has_blocking_status = bool(comparisons) and heldout.get("cross_arm_claims_allowed") is not True
    has_serving_blockers = bool(serving_preflight and serving_preflight.get("blocking_reasons"))
    expected_passed = (
        not risks
        and not has_failed_child
        and not has_blocked_adapter
        and not has_blocked_external_result
        and not has_arm_blockers
        and not has_blocking_status
        and not has_serving_blockers
    )
    if isinstance(summary.get("passed"), bool) and summary.get("passed") != expected_passed:
        target.errors.append(f"eval_summary.passed expected {expected_passed}, got {summary.get('passed')!r}.")

    conclusion = summary.get("conclusion")
    if not isinstance(conclusion, dict):
        target.errors.append("eval_summary.conclusion must be an object.")
    elif conclusion.get("status") not in {"ready", "blocked"}:
        target.errors.append("eval_summary.conclusion.status must be 'ready' or 'blocked'.")
    else:
        _validate_allowed_keys(conclusion, _EVAL_SUMMARY_CONCLUSION_KEYS, target, "eval_summary.conclusion")

    target.details.update(
        {
            "passed": summary.get("passed"),
            "arm_count": len(arms),
            "comparison_count": len(comparisons),
            "risk_count": len(risks),
            "heldout_status": heldout.get("status"),
            "repair_curriculum_work_item_count": repair_curriculum.get("work_item_count"),
        }
    )

def _validate_eval_summary_heldout(heldout: dict[str, Any], target: ValidationTarget, has_comparisons: bool) -> None:
    _validate_allowed_keys(heldout, _EVAL_SUMMARY_HELDOUT_KEYS, target, "eval_summary.heldout_scenarios")
    status = heldout.get("status")
    if not isinstance(status, str) or status not in {
        "missing_suite_summaries",
        "single_arm",
        "identical",
        "mismatched",
        "empty",
        "blocked",
    }:
        target.errors.append("eval_summary.heldout_scenarios.status has an unsupported value.")
    for field_name in ("identical", "cross_arm_claims_allowed"):
        if not isinstance(heldout.get(field_name), bool):
            target.errors.append(f"eval_summary.heldout_scenarios.{field_name} must be a boolean.")
    if not _is_non_negative_int(heldout.get("scenario_count")):
        target.errors.append("eval_summary.heldout_scenarios.scenario_count must be a non-negative integer.")
    if not _is_string_list(heldout.get("scenario_ids")):
        target.errors.append("eval_summary.heldout_scenarios.scenario_ids must be a list of strings.")
    if not isinstance(heldout.get("arms"), list):
        target.errors.append("eval_summary.heldout_scenarios.arms must be a list.")
    else:
        for index, arm in enumerate(heldout["arms"]):
            if isinstance(arm, dict):
                _validate_allowed_keys(
                    arm,
                    _EVAL_SUMMARY_HELDOUT_ARM_KEYS,
                    target,
                    f"eval_summary.heldout_scenarios.arms[{index}]",
                )
    if not isinstance(heldout.get("mismatches"), list):
        target.errors.append("eval_summary.heldout_scenarios.mismatches must be a list.")
    else:
        for index, mismatch in enumerate(heldout["mismatches"]):
            if isinstance(mismatch, dict):
                _validate_allowed_keys(
                    mismatch,
                    _EVAL_SUMMARY_HELDOUT_MISMATCH_KEYS,
                    target,
                    f"eval_summary.heldout_scenarios.mismatches[{index}]",
                )
    if not _is_string_list(heldout.get("blocking_reasons")):
        target.errors.append("eval_summary.heldout_scenarios.blocking_reasons must be a list of strings.")
    if heldout.get("cross_arm_claims_allowed") is True and heldout.get("status") != "identical":
        target.errors.append("eval_summary.heldout_scenarios.cross_arm_claims_allowed requires status 'identical'.")
    if has_comparisons and heldout.get("cross_arm_claims_allowed") is not True and not heldout.get("blocking_reasons"):
        target.errors.append("eval_summary.heldout_scenarios.blocking_reasons must explain disallowed comparisons.")

def _validate_eval_summary_heldout_replay(
    heldout: dict[str, Any],
    replayed_arms: list[dict[str, Any]],
    target: ValidationTarget,
) -> None:
    expected = _build_eval_summary_heldout(replayed_arms)
    fields = (
        "status",
        "identical",
        "cross_arm_claims_allowed",
        "scenario_count",
        "scenario_ids",
        "arms",
        "mismatches",
        "blocking_reasons",
    )
    mismatched_fields = [field_name for field_name in fields if heldout.get(field_name) != expected[field_name]]
    if mismatched_fields:
        target.errors.append(
            "eval_summary.heldout_scenarios does not match current arm scenario IDs and fingerprints for: "
            f"{', '.join(mismatched_fields)}."
        )

def _validate_eval_summary_arm(
    arm: Any,
    index: int,
    target: ValidationTarget,
    *,
    serving_required: bool = False,
    source_dir: Path | None = None,
) -> dict[str, Any] | None:
    if not isinstance(arm, dict):
        target.errors.append(f"eval_summary.arms[{index}] must be an object.")
        return None
    label = f"eval_summary.arms[{index}]"
    _validate_allowed_keys(arm, _EVAL_SUMMARY_ARM_KEYS, target, label)
    for field_name in ("label", "path"):
        if not isinstance(arm.get(field_name), str) or not arm.get(field_name):
            target.errors.append(f"eval_summary.arms[{index}].{field_name} must be a non-empty string.")
    _validate_eval_summary_source_file_ref(arm, "path", "sha256", "size_bytes", target, label, source_dir)
    replayed_arm = _validate_eval_summary_suite_source_schema(arm, target, label, source_dir)
    for field_name in ("scenario_count", "total", "passed", "failed", "error_count"):
        if not _is_non_negative_int(arm.get(field_name)):
            target.errors.append(f"eval_summary.arms[{index}].{field_name} must be a non-negative integer.")
    if not _is_string_list(arm.get("scenario_ids")):
        target.errors.append(f"eval_summary.arms[{index}].scenario_ids must be a list of strings.")
    if not _is_string_list(arm.get("blocking_reasons")):
        target.errors.append(f"eval_summary.arms[{index}].blocking_reasons must be a list of strings.")
    _validate_eval_summary_operational_metrics(arm.get("operational_metrics"), index, target)
    serving = arm.get("serving_preflight")
    if serving is None:
        if serving_required:
            target.errors.append(f"eval_summary.arms[{index}].serving_preflight is required.")
    else:
        _validate_eval_summary_arm_serving_preflight(
            serving,
            index,
            target,
            serving_required=serving_required,
            source_dir=source_dir,
        )
    return replayed_arm

def _validate_eval_summary_suite_source_schema(
    arm: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_dir: Path | None,
) -> dict[str, Any] | None:
    raw_path = arm.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    suite_path = _resolve_eval_summary_source_path(raw_path, source_dir)
    if suite_path is None:
        return None
    try:
        if (
            not suite_path.exists()
            or not suite_path.is_file()
            or _path_has_symlink_component(suite_path, include_leaf=True)
        ):
            return None
    except (OSError, ValueError, RuntimeError):
        target.errors.append(f"{label}.path could not be inspected as a run_suite source.")
        return None
    try:
        schema_check = check_schema_file(suite_path, "run_suite")
    except (OSError, UnicodeError, json.JSONDecodeError, SchemaRegistryError) as exc:
        target.errors.append(f"{label}.path could not be checked against the run_suite schema: {exc}")
        return None
    if schema_check.get("passed") is not True:
        target.errors.append(f"{label}.path must satisfy the run_suite schema.")
        return None
    serving_preflight = arm.get("serving_preflight")
    if not isinstance(serving_preflight, dict):
        serving_preflight = None
    try:
        expected = _suite_arm(
            LabeledPath(label=str(arm.get("label") or ""), path=suite_path),
            preserve_paths=True,
            display_base_dir=None,
            serving_preflight=serving_preflight,
            require_serving_preflight=bool(serving_preflight and serving_preflight.get("required") is True),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        target.errors.append(f"{label}.path could not be replayed as a run_suite source: {exc}")
        return None
    derived_fields = (
        "schema_version",
        "scenario_count",
        "scenario_ids",
        "total",
        "passed",
        "failed",
        "error_count",
        "pass_rate",
        "average_score",
        "failed_rule_counts",
        "critical_failure_counts",
        "operational_metrics",
        "validation",
        "source_validation",
        "blocking_reasons",
    )
    mismatched = [field for field in derived_fields if arm.get(field) != expected.get(field)]
    if mismatched:
        target.errors.append(
            f"{label} does not match the referenced run_suite source for: {', '.join(mismatched)}."
        )
    return expected

def _validate_eval_summary_source_file_ref(
    record: dict[str, Any],
    path_field: str,
    sha_field: str,
    size_field: str,
    target: ValidationTarget,
    label: str,
    source_dir: Path | None,
) -> None:
    raw_path = record.get(path_field)
    if not isinstance(raw_path, str) or not raw_path:
        target.errors.append(f"{label}.{path_field} must be a non-empty string.")
        return
    _warn_absolute_public_path(target, f"{label}.{path_field}", raw_path)
    expected_sha = record.get(sha_field)
    expected_size = record.get(size_field)
    if not _is_lowercase_sha256(expected_sha):
        target.errors.append(f"{label}.{sha_field} must be a lowercase SHA-256 hex string.")
    if not _is_non_negative_int(expected_size):
        target.errors.append(f"{label}.{size_field} must be a non-negative integer.")

    source_path = _resolve_eval_summary_source_path(raw_path, source_dir)
    if source_path is None:
        target.errors.append(f"{label}.{path_field} must resolve to an existing file.")
        return
    try:
        if not source_path.exists() or not source_path.is_file():
            target.errors.append(f"{label}.{path_field} must resolve to an existing file.")
            return
        if _path_has_symlink_component(source_path, include_leaf=True):
            target.errors.append(f"{label}.{path_field} must resolve to a regular non-symlink file.")
            return
        if _is_non_negative_int(expected_size) and source_path.stat().st_size != expected_size:
            target.errors.append(f"{label}.{size_field} does not match the current file.")
        if _is_lowercase_sha256(expected_sha) and _sha256(source_path) != expected_sha:
            target.errors.append(f"{label}.{sha_field} does not match the current file.")
    except (OSError, ValueError, RuntimeError):
        target.errors.append(f"{label}.{path_field} could not be inspected.")

def _resolve_eval_summary_source_path(value: str, source_dir: Path | None) -> Path | None:
    path = Path(value)
    if path.is_absolute():
        return path
    if source_dir is None:
        return None
    return source_dir / path

def _validate_eval_summary_serving_preflight_summary(summary: dict[str, Any], target: ValidationTarget) -> None:
    for field_name in ("required",):
        if not isinstance(summary.get(field_name), bool):
            target.errors.append(f"eval_summary.serving_preflight.{field_name} must be a boolean.")
    for field_name in ("input_count", "attached_count"):
        if not _is_non_negative_int(summary.get(field_name)):
            target.errors.append(f"eval_summary.serving_preflight.{field_name} must be a non-negative integer.")
    for field_name in ("unmatched_labels", "duplicate_labels", "blocking_reasons"):
        if not _is_string_list(summary.get(field_name)):
            target.errors.append(f"eval_summary.serving_preflight.{field_name} must be a list of strings.")
    if summary.get("unmatched_labels") and "serving_preflight_unmatched_arm" not in summary.get("blocking_reasons", []):
        target.errors.append("eval_summary.serving_preflight.blocking_reasons must include serving_preflight_unmatched_arm when labels are unmatched.")
    if summary.get("duplicate_labels") and "duplicate_serving_preflight_labels" not in summary.get("blocking_reasons", []):
        target.errors.append("eval_summary.serving_preflight.blocking_reasons must include duplicate_serving_preflight_labels when labels are duplicated.")

def _validate_eval_summary_serving_preflight_consistency(
    summary: dict[str, Any],
    arms: list[Any],
    target: ValidationTarget,
) -> None:
    attached_count = 0
    for index, arm in enumerate(arms):
        if not isinstance(arm, dict):
            continue
        serving = arm.get("serving_preflight")
        if not isinstance(serving, dict):
            continue
        if serving.get("provided") is True:
            attached_count += 1
        if (
            isinstance(summary.get("required"), bool)
            and isinstance(serving.get("required"), bool)
            and serving.get("required") != summary.get("required")
        ):
            target.errors.append(
                f"eval_summary.arms[{index}].serving_preflight.required must match eval_summary.serving_preflight.required."
            )

    if _is_non_negative_int(summary.get("attached_count")) and summary.get("attached_count") != attached_count:
        target.errors.append(
            f"eval_summary.serving_preflight.attached_count expected {attached_count}, got {summary.get('attached_count')!r}."
        )

    unmatched_labels = summary.get("unmatched_labels") if _is_string_list(summary.get("unmatched_labels")) else []
    duplicate_labels = summary.get("duplicate_labels") if _is_string_list(summary.get("duplicate_labels")) else []
    expected_min_input_count = attached_count + len(unmatched_labels) + len(duplicate_labels)
    if _is_non_negative_int(summary.get("input_count")) and summary.get("input_count") < expected_min_input_count:
        target.errors.append(
            "eval_summary.serving_preflight.input_count "
            f"expected at least {expected_min_input_count}, got {summary.get('input_count')!r}."
        )

    expected_blocking_reasons = []
    if unmatched_labels:
        expected_blocking_reasons.append("serving_preflight_unmatched_arm")
    if duplicate_labels:
        expected_blocking_reasons.append("duplicate_serving_preflight_labels")
    blocking_reasons = summary.get("blocking_reasons") if _is_string_list(summary.get("blocking_reasons")) else []
    if set(blocking_reasons) != set(expected_blocking_reasons):
        target.errors.append(
            "eval_summary.serving_preflight.blocking_reasons "
            f"expected {expected_blocking_reasons!r}, got {blocking_reasons!r}."
        )

def _validate_eval_summary_arm_serving_preflight(
    serving: Any,
    index: int,
    target: ValidationTarget,
    *,
    serving_required: bool,
    source_dir: Path | None = None,
) -> None:
    label = f"eval_summary.arms[{index}].serving_preflight"
    if not isinstance(serving, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(serving, _EVAL_SUMMARY_SERVING_PREFLIGHT_KEYS, target, label)
    for field_name in ("provided", "required", "passed"):
        if not isinstance(serving.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if serving_required and serving.get("required") is not True:
        target.errors.append(f"{label}.required must be true when eval_summary requires serving preflight.")
    if serving.get("readiness") not in {"ready", "blocked", "missing"}:
        target.errors.append(f"{label}.readiness has an unsupported value.")
    if serving.get("path") is not None and not isinstance(serving.get("path"), str):
        target.errors.append(f"{label}.path must be a string or null.")
    for field_name in ("failed_checks", "blocking_reasons"):
        if not _is_string_list(serving.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a list of strings.")
    if not isinstance(serving.get("artifacts"), dict):
        target.errors.append(f"{label}.artifacts must be an object.")
    if serving.get("provided") is True:
        if serving.get("schema_version") != "hfr.serving_endpoint_check.v1":
            target.errors.append(f"{label}.schema_version must be hfr.serving_endpoint_check.v1 when provided.")
        _validate_eval_summary_source_file_ref(serving, "path", "sha256", "size_bytes", target, label, source_dir)
        for field_name in ("profile_id", "model", "served_model_id", "base_url"):
            if not isinstance(serving.get(field_name), str):
                target.errors.append(f"{label}.{field_name} must be a string.")
        if serving.get("readiness") == "missing":
            target.errors.append(f"{label}.readiness cannot be missing when provided is true.")
        if serving.get("passed") is True and serving.get("readiness") != "ready":
            target.errors.append(f"{label}.passed requires readiness ready.")
        failed_checks = serving.get("failed_checks") if _is_string_list(serving.get("failed_checks")) else []
        blocking_reasons = serving.get("blocking_reasons") if _is_string_list(serving.get("blocking_reasons")) else []
        if (
            serving.get("passed") is not True or serving.get("readiness") != "ready" or failed_checks
        ) and "serving_preflight_blocked" not in blocking_reasons:
            target.errors.append(
                f"{label}.blocking_reasons must include serving_preflight_blocked when the preflight is blocked."
            )
        if serving.get("passed") is False and not serving.get("blocking_reasons"):
            target.errors.append(f"{label}.blocking_reasons must explain failed serving preflight.")
    if serving_required and serving.get("provided") is not True and "serving_preflight_missing" not in serving.get("blocking_reasons", []):
        target.errors.append(f"{label}.blocking_reasons must include serving_preflight_missing when required and missing.")

def _validate_eval_summary_operational_metrics(value: Any, index: int, target: ValidationTarget) -> None:
    label = f"eval_summary.arms[{index}].operational_metrics"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _EVAL_SUMMARY_OPERATIONAL_KEYS, target, label)
    for section_name in ("cost", "latency", "tokens", "task_completion"):
        if not isinstance(value.get(section_name), dict):
            target.errors.append(f"{label}.{section_name} must be an object.")

    cost = value.get("cost") if isinstance(value.get("cost"), dict) else {}
    if cost:
        _validate_allowed_keys(cost, _EVAL_SUMMARY_COST_KEYS, target, f"{label}.cost")
    _validate_metric_source(cost, f"{label}.cost", target)
    _validate_metric_count_fields(cost, f"{label}.cost", target)
    if not _is_optional_non_negative_number(cost.get("total_usd")):
        target.errors.append(f"{label}.cost.total_usd must be null or a non-negative number.")

    latency = value.get("latency") if isinstance(value.get("latency"), dict) else {}
    if latency:
        _validate_allowed_keys(latency, _EVAL_SUMMARY_LATENCY_KEYS, target, f"{label}.latency")
    _validate_metric_source(latency, f"{label}.latency", target)
    _validate_metric_count_fields(latency, f"{label}.latency", target)
    for field_name in ("average_ms", "p50_ms", "p95_ms", "max_ms"):
        if not _is_optional_non_negative_number(latency.get(field_name)):
            target.errors.append(f"{label}.latency.{field_name} must be null or a non-negative number.")

    tokens = value.get("tokens") if isinstance(value.get("tokens"), dict) else {}
    if tokens:
        _validate_allowed_keys(tokens, _EVAL_SUMMARY_TOKENS_KEYS, target, f"{label}.tokens")
    _validate_metric_source(tokens, f"{label}.tokens", target)
    _validate_metric_count_fields(tokens, f"{label}.tokens", target)
    for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if not _is_optional_non_negative_int(tokens.get(field_name)):
            target.errors.append(f"{label}.tokens.{field_name} must be null or a non-negative integer.")

    task = value.get("task_completion") if isinstance(value.get("task_completion"), dict) else {}
    if task:
        _validate_allowed_keys(task, _EVAL_SUMMARY_TASK_COMPLETION_KEYS, target, f"{label}.task_completion")
    _validate_metric_source(task, f"{label}.task_completion", target)
    for field_name in (
        "configured_count",
        "complete_count",
        "incomplete_count",
        "not_applicable_count",
        "unknown_count",
        "passed_count",
        "failed_count",
    ):
        if not _is_non_negative_int(task.get(field_name)):
            target.errors.append(f"{label}.task_completion.{field_name} must be a non-negative integer.")
    if not _is_optional_rate(task.get("pass_rate")):
        target.errors.append(f"{label}.task_completion.pass_rate must be null or a number between 0 and 1.")

def _validate_eval_summary_comparison(
    comparison: Any,
    index: int,
    target: ValidationTarget,
    heldout: dict[str, Any],
    *,
    source_dir: Path | None = None,
) -> None:
    if not isinstance(comparison, dict):
        target.errors.append(f"eval_summary.comparisons[{index}] must be an object.")
        return
    label = f"eval_summary.comparisons[{index}]"
    _validate_allowed_keys(comparison, _EVAL_SUMMARY_COMPARISON_KEYS, target, label)
    for field_name in ("label", "path", "manifest"):
        if not isinstance(comparison.get(field_name), str) or not comparison.get(field_name):
            target.errors.append(f"eval_summary.comparisons[{index}].{field_name} must be a non-empty string.")
    _validate_eval_summary_source_file_ref(
        comparison,
        "manifest",
        "manifest_sha256",
        "manifest_size_bytes",
        target,
        label,
        source_dir,
    )
    for field_name in ("claims_allowed", "passed"):
        if not isinstance(comparison.get(field_name), bool):
            target.errors.append(f"eval_summary.comparisons[{index}].{field_name} must be a boolean.")
    if not _is_string_list(comparison.get("blocking_reasons")):
        target.errors.append(f"eval_summary.comparisons[{index}].blocking_reasons must be a list of strings.")
    raw = comparison.get("raw_movement")
    if not isinstance(raw, dict):
        target.errors.append(f"eval_summary.comparisons[{index}].raw_movement must be an object.")
        raw = {}
    elif raw:
        _validate_allowed_keys(raw, _EVAL_SUMMARY_RAW_MOVEMENT_KEYS, target, f"{label}.raw_movement")
    for field_name in (
        "pair_count",
        "candidate_win_count",
        "baseline_win_count",
        "task_completion_improvement_count",
        "task_completion_regression_count",
        "contract_drift_count",
        "unverified_contract_count",
        "skipped_pair_count",
    ):
        if not _is_non_negative_int(raw.get(field_name)):
            target.errors.append(f"eval_summary.comparisons[{index}].raw_movement.{field_name} must be a non-negative integer.")
    claims = comparison.get("governance_claims")
    if not isinstance(claims, dict):
        target.errors.append(f"eval_summary.comparisons[{index}].governance_claims must be an object.")
        claims = {}
    elif claims:
        _validate_allowed_keys(claims, _EVAL_SUMMARY_GOVERNANCE_CLAIMS_KEYS, target, f"{label}.governance_claims")
    if comparison.get("claims_allowed") is True and heldout.get("cross_arm_claims_allowed") is not True:
        target.errors.append(f"eval_summary.comparisons[{index}].claims_allowed requires identical held-out scenarios.")
    if comparison.get("claims_allowed") is False:
        if not comparison.get("blocking_reasons"):
            target.errors.append(f"eval_summary.comparisons[{index}].blocking_reasons must explain disallowed claims.")
        if claims.get("candidate_win_count") != 0:
            target.errors.append(f"eval_summary.comparisons[{index}].governance_claims.candidate_win_count must be 0 when claims are disallowed.")
        if claims.get("task_completion_improvement_count") != 0:
            target.errors.append(
                f"eval_summary.comparisons[{index}].governance_claims.task_completion_improvement_count must be 0 when claims are disallowed."
            )
        if claims.get("candidate_win_scenarios") not in ([], None):
            target.errors.append(
                f"eval_summary.comparisons[{index}].governance_claims.candidate_win_scenarios must be empty when claims are disallowed."
            )
        if claims.get("suppressed_raw_claims") is not True:
            target.errors.append(f"eval_summary.comparisons[{index}].governance_claims.suppressed_raw_claims must be true when claims are disallowed.")
    if comparison.get("passed") is True and comparison.get("blocking_reasons"):
        target.errors.append(f"eval_summary.comparisons[{index}].passed cannot be true with blocking_reasons.")

def _validate_eval_summary_gate(gate: Any, index: int, target: ValidationTarget, *, source_dir: Path | None = None) -> None:
    if not isinstance(gate, dict):
        target.errors.append(f"eval_summary.compare_gates[{index}] must be an object.")
        return
    label = f"eval_summary.compare_gates[{index}]"
    _validate_allowed_keys(gate, _EVAL_SUMMARY_GATE_KEYS, target, label)
    _validate_eval_summary_source_file_ref(gate, "path", "sha256", "size_bytes", target, label, source_dir)
    if not isinstance(gate.get("passed"), bool):
        target.errors.append(f"eval_summary.compare_gates[{index}].passed must be a boolean.")
    if not _is_string_list(gate.get("blocking_reasons")):
        target.errors.append(f"eval_summary.compare_gates[{index}].blocking_reasons must be a list of strings.")
    if gate.get("passed") is False and "compare_gate_failed" not in gate.get("blocking_reasons", []):
        target.errors.append(f"eval_summary.compare_gates[{index}].blocking_reasons must include compare_gate_failed when failed.")
    failed_checks = gate.get("failed_checks") if isinstance(gate.get("failed_checks"), list) else []
    for failed_index, failed_check in enumerate(failed_checks):
        if isinstance(failed_check, dict):
            _validate_allowed_keys(
                failed_check,
                _EVAL_SUMMARY_GATE_FAILED_CHECK_KEYS,
                target,
                f"{label}.failed_checks[{failed_index}]",
            )

def _validate_eval_summary_external_adapter(
    adapter: Any,
    index: int,
    target: ValidationTarget,
    *,
    source_dir: Path | None = None,
) -> None:
    if not isinstance(adapter, dict):
        target.errors.append(f"eval_summary.external_adapter_plans[{index}] must be an object.")
        return
    label = f"eval_summary.external_adapter_plans[{index}]"
    _validate_allowed_keys(adapter, _EVAL_SUMMARY_EXTERNAL_ADAPTER_KEYS, target, label)
    _validate_eval_summary_source_file_ref(adapter, "path", "sha256", "size_bytes", target, label, source_dir)
    if not isinstance(adapter.get("ready"), bool):
        target.errors.append(f"eval_summary.external_adapter_plans[{index}].ready must be a boolean.")
    for field_name in ("adapter_count", "ready_adapter_count"):
        if not _is_non_negative_int(adapter.get(field_name)):
            target.errors.append(f"eval_summary.external_adapter_plans[{index}].{field_name} must be a non-negative integer.")
    if not _is_string_list(adapter.get("blocking_reasons")):
        target.errors.append(f"eval_summary.external_adapter_plans[{index}].blocking_reasons must be a list of strings.")
    if not _is_string_list(adapter.get("selected_adapters")):
        target.errors.append(f"eval_summary.external_adapter_plans[{index}].selected_adapters must be a list of strings.")
    if adapter.get("ready") is False and not adapter.get("blocking_reasons"):
        target.errors.append(f"eval_summary.external_adapter_plans[{index}].blocking_reasons must explain why the plan is not ready.")
    raw_path = adapter.get("path")
    plan_path = _resolve_eval_summary_source_path(raw_path, source_dir) if isinstance(raw_path, str) else None
    if plan_path is None or not plan_path.is_file() or _path_has_symlink_component(plan_path, include_leaf=True):
        return
    try:
        expected = _build_eval_summary_external_plan(
            LabeledPath(label=str(adapter.get("label") or ""), path=plan_path),
            preserve_paths=False,
            display_base_dir=source_dir,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        target.errors.append(f"{label}.path could not be replayed as an external eval plan: {exc}")
        return
    if {key: value for key, value in adapter.items() if key != "path"} != {
        key: value for key, value in expected.items() if key != "path"
    }:
        target.errors.append(f"{label} must match the current external eval plan source projection exactly.")

def _validate_eval_summary_external_result(
    result: Any,
    index: int,
    target: ValidationTarget,
    *,
    source_dir: Path | None = None,
) -> None:
    if not isinstance(result, dict):
        target.errors.append(f"eval_summary.external_adapter_results[{index}] must be an object.")
        return
    label = f"eval_summary.external_adapter_results[{index}]"
    _validate_allowed_keys(result, _EVAL_SUMMARY_EXTERNAL_RESULT_KEYS, target, label)
    _validate_eval_summary_source_file_ref(result, "path", "sha256", "size_bytes", target, label, source_dir)
    for field_name in ("integrity_passed", "coverage_complete", "external_eval_claims_allowed"):
        if not isinstance(result.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if not _is_string_list(result.get("blocking_reasons")):
        target.errors.append(f"{label}.blocking_reasons must be a list of strings.")
    raw_path = result.get("path")
    result_path = _resolve_eval_summary_source_path(raw_path, source_dir) if isinstance(raw_path, str) else None
    if (
        result_path is None
        or not result_path.is_file()
        or _path_has_symlink_component(result_path, include_leaf=True)
    ):
        return
    try:
        expected = _build_eval_summary_external_result(
            LabeledPath(label=str(result.get("label") or ""), path=result_path),
            preserve_paths=False,
            display_base_dir=source_dir,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        target.errors.append(f"{label}.path could not be replayed as an external eval result: {exc}")
        return
    if {key: value for key, value in result.items() if key != "path"} != {
        key: value for key, value in expected.items() if key != "path"
    }:
        target.errors.append(f"{label} must match the current external eval result source projection exactly.")

def _validate_eval_summary_repair_curriculum(value: dict[str, Any], target: ValidationTarget) -> None:
    _validate_allowed_keys(value, _EVAL_SUMMARY_REPAIR_KEYS, target, "eval_summary.repair_curriculum")
    items = value.get("items")
    if not isinstance(items, list):
        target.errors.append("eval_summary.repair_curriculum.items must be a list.")
        items = []
    if value.get("work_item_count") != len(items):
        target.errors.append(f"eval_summary.repair_curriculum.work_item_count expected {len(items)}, got {value.get('work_item_count')!r}.")
    critical_count = sum(1 for item in items if isinstance(item, dict) and item.get("priority") == "critical")
    if value.get("critical_work_item_count") != critical_count:
        target.errors.append(
            "eval_summary.repair_curriculum.critical_work_item_count "
            f"expected {critical_count}, got {value.get('critical_work_item_count')!r}."
        )
    priority_counts = _validate_count_rows(value.get("priority_counts"), target, "eval_summary.repair_curriculum.priority_counts")
    expected_priority_counts = _count_eval_summary_work_item_field(items, "priority")
    if priority_counts != expected_priority_counts:
        target.errors.append("eval_summary.repair_curriculum.priority_counts do not match work items.")
    category_counts = _validate_count_rows(value.get("category_counts"), target, "eval_summary.repair_curriculum.category_counts")
    expected_category_counts = _count_eval_summary_work_item_field(items, "category")
    if category_counts != expected_category_counts:
        target.errors.append("eval_summary.repair_curriculum.category_counts do not match work items.")
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        _validate_eval_summary_work_item(item, index, target, seen_ids)
    notes = value.get("notes")
    if notes is not None and not _is_string_list(notes):
        target.errors.append("eval_summary.repair_curriculum.notes must be a list of strings when present.")

def _validate_eval_summary_work_item(item: Any, index: int, target: ValidationTarget, seen_ids: set[str]) -> None:
    label = f"eval_summary.repair_curriculum.items[{index}]"
    if not isinstance(item, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(item, _EVAL_SUMMARY_WORK_ITEM_KEYS, target, label)
    for field_name in ("work_item_id", "category", "priority", "source", "label", "reason", "summary", "suggested_action"):
        if not isinstance(item.get(field_name), str) or not item.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    item_id = item.get("work_item_id")
    if isinstance(item_id, str):
        if item_id in seen_ids:
            target.errors.append(f"{label}.work_item_id duplicates {item_id!r}.")
        seen_ids.add(item_id)
    if item.get("priority") not in {"critical", "high", "medium", "low"}:
        target.errors.append(f"{label}.priority must be critical, high, medium, or low.")
    if item.get("category") not in {"repair", "curriculum", "eval_gate", "eval_harness"}:
        target.errors.append(f"{label}.category must be repair, curriculum, eval_gate, or eval_harness.")
    for field_name in ("scenario_id", "rule_id"):
        if field_name in item and (not isinstance(item.get(field_name), str) or not item.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-empty string when present.")
    if "count" in item and not _is_non_negative_int(item.get("count")):
        target.errors.append(f"{label}.count must be a non-negative integer when present.")

def _count_eval_summary_work_item_field(items: list[Any], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get(field_name)
        if not isinstance(value, str) or not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts

def _validate_external_eval_result(result: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    schema_check = check_schema_contract(result, name_or_id="external_eval_result", artifact_path=source_path)
    for error in schema_check.get("errors", []):
        target.errors.append(f"external_eval_result schema: {error}")
    if result.get("schema_version") != EXTERNAL_EVAL_RESULT_SCHEMA_VERSION:
        target.errors.append(
            f"external_eval_result.schema_version must be {EXTERNAL_EVAL_RESULT_SCHEMA_VERSION!r}."
        )

    sources = result.get("sources") if isinstance(result.get("sources"), dict) else {}
    resolved: dict[str, Path | None] = {}
    for name in ("plan", "heldout_manifest", "raw_result", "runner_metadata"):
        ref = sources.get(name) if isinstance(sources.get(name), dict) else None
        if ref is None:
            target.errors.append(f"external_eval_result.sources.{name} must be an object.")
            resolved[name] = None
            continue
        optional = name == "runner_metadata"
        resolved[name] = _validate_external_eval_result_source_ref(
            ref,
            target,
            f"external_eval_result.sources.{name}",
            source_path,
            optional=optional,
        )

    identity = result.get("identity") if isinstance(result.get("identity"), dict) else {}
    normalizer = result.get("normalizer") if isinstance(result.get("normalizer"), dict) else {}
    execution = result.get("execution") if isinstance(result.get("execution"), dict) else {}
    failure = execution.get("failure") if isinstance(execution.get("failure"), dict) else {}
    required_sources = (resolved.get("plan"), resolved.get("heldout_manifest"), resolved.get("raw_result"))
    if all(path is not None for path in required_sources):
        try:
            expected = build_external_eval_result(
                plan_path=resolved["plan"],
                heldout_manifest_path=resolved["heldout_manifest"],
                raw_result_path=resolved["raw_result"],
                runner_metadata_path=resolved.get("runner_metadata"),
                runner_observation=(
                    result.get("runner_observation")
                    if resolved.get("runner_metadata") is None
                    and isinstance(result.get("runner_observation"), dict)
                    else None
                ),
                adapter_id=str(identity.get("adapter_id") or ""),
                execution_id=str(identity.get("execution_id") or ""),
                model_id=str(identity.get("model_id") or ""),
                normalizer_id=str(normalizer.get("id") or ""),
                normalizer_version=str(normalizer.get("version") or ""),
                raw_format=str(normalizer.get("input_format") or ""),
                execution_status=str(execution.get("status") or ""),
                failure_class=str(failure.get("class") or "none"),
                failure_message=str(failure.get("message") or ""),
                out_path=source_path,
                created_at=str(result.get("created_at") or ""),
            )
        except (ExternalEvalResultError, OSError, TypeError, ValueError) as exc:
            target.errors.append(f"external_eval_result could not replay imported sources: {exc}")
        else:
            if result != expected:
                target.errors.append(
                    "external_eval_result must match deterministic normalization of its current source files."
                )

    integrity = result.get("integrity") if isinstance(result.get("integrity"), dict) else {}
    coverage = result.get("coverage") if isinstance(result.get("coverage"), dict) else {}
    outcome = result.get("benchmark_outcome") if isinstance(result.get("benchmark_outcome"), dict) else {}
    governance = result.get("governance") if isinstance(result.get("governance"), dict) else {}
    target.details.update(
        {
            "integrity_passed": integrity.get("passed") is True,
            "execution_status": execution.get("status"),
            "coverage_complete": coverage.get("complete") is True,
            "benchmark_outcome": outcome.get("status"),
            "governance_readiness": governance.get("readiness"),
        }
    )

def _validate_external_eval_result_source_ref(
    ref: dict[str, Any],
    target: ValidationTarget,
    label: str,
    result_path: Path,
    *,
    optional: bool,
) -> Path | None:
    path_value = ref.get("path")
    if path_value is None and optional:
        if any(
            ref.get(field) not in expected
            for field, expected in (
                ("exists", {False}),
                ("regular_file", {False}),
                ("replayable", {True}),
                ("sha256", {None}),
                ("size_bytes", {None}),
                ("schema_version", {None}),
            )
        ):
            target.errors.append(f"{label} must be an empty optional source reference when path is null.")
        return None
    if not isinstance(path_value, str) or not path_value:
        target.errors.append(f"{label}.path must be a non-empty relative path.")
        return None
    if not _is_replayable_external_eval_ref_path(path_value):
        target.errors.append(f"{label}.path must be a safe relative path without traversal.")
        return None
    path = result_path.parent / path_value
    if _path_has_symlink_component(path, include_leaf=True) or not path.is_file():
        target.errors.append(f"{label}.path must resolve to a regular non-symlink file.")
        return None
    if ref.get("exists") is not True or ref.get("regular_file") is not True or ref.get("replayable") is not True:
        target.errors.append(f"{label} must describe an existing replayable regular file.")
    if ref.get("size_bytes") != path.stat().st_size:
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if ref.get("sha256") != _sha256(path):
        target.errors.append(f"{label}.sha256 does not match the current file.")
    return path

def _validate_external_eval_plan(plan: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(plan, "schema_version", EXTERNAL_EVAL_PLAN_SCHEMA_VERSION, target)
    _validate_allowed_keys(
        plan,
        {
            "schema_version",
            "generated_at",
            "ready",
            "adapter_count",
            "ready_adapter_count",
            "selected_adapters",
            "allow_installed",
            "inputs",
            "adapters",
            "blocking_reasons",
            "governance_handoff",
        },
        target,
        "external_eval_plan",
    )
    if not isinstance(plan.get("ready"), bool):
        target.errors.append("external_eval_plan.ready must be a boolean.")
    if not isinstance(plan.get("allow_installed"), bool):
        target.errors.append("external_eval_plan.allow_installed must be a boolean.")

    adapters = plan.get("adapters")
    if not isinstance(adapters, list):
        target.errors.append("external_eval_plan.adapters must be a list.")
        adapters = []
    blocking_reasons = plan.get("blocking_reasons")
    if not _is_string_list(blocking_reasons):
        target.errors.append("external_eval_plan.blocking_reasons must be a list of strings.")
        blocking_reasons = []
    selected = plan.get("selected_adapters")
    if not _is_string_list(selected):
        target.errors.append("external_eval_plan.selected_adapters must be a list of strings.")
        selected = []

    if plan.get("adapter_count") != len(adapters):
        target.errors.append(f"external_eval_plan.adapter_count expected {len(adapters)}, got {plan.get('adapter_count')!r}.")
    ready_count = sum(1 for adapter in adapters if isinstance(adapter, dict) and adapter.get("ready") is True)
    if plan.get("ready_adapter_count") != ready_count:
        target.errors.append(
            f"external_eval_plan.ready_adapter_count expected {ready_count}, got {plan.get('ready_adapter_count')!r}."
        )
    if sorted(selected) != sorted(adapter.get("id") for adapter in adapters if isinstance(adapter, dict)):
        target.errors.append("external_eval_plan.selected_adapters must match adapter ids.")

    inputs = plan.get("inputs")
    if not isinstance(inputs, dict):
        target.errors.append("external_eval_plan.inputs must be an object.")
        inputs = {}
    _validate_external_eval_inputs(inputs, target, source_path)

    for index, adapter in enumerate(adapters):
        _validate_external_eval_adapter_plan(adapter, index, target, inputs)

    expected_ready = bool(adapters) and all(isinstance(adapter, dict) and adapter.get("ready") is True for adapter in adapters)
    if isinstance(plan.get("ready"), bool) and plan.get("ready") != expected_ready:
        target.errors.append(f"external_eval_plan.ready expected {expected_ready}, got {plan.get('ready')!r}.")
    if plan.get("ready") is True and blocking_reasons:
        target.errors.append("external_eval_plan.blocking_reasons must be empty when ready is true.")
    if plan.get("ready") is False and not blocking_reasons:
        target.errors.append("external_eval_plan.blocking_reasons must explain why ready is false.")

    handoff = plan.get("governance_handoff")
    if not isinstance(handoff, dict):
        target.errors.append("external_eval_plan.governance_handoff must be an object.")
    else:
        _validate_allowed_keys(
            handoff,
            {"external_eval_claims_allowed", "requires_identical_heldout_scenarios", "recommendation"},
            target,
            "external_eval_plan.governance_handoff",
        )
        if handoff.get("requires_identical_heldout_scenarios") is not True:
            target.errors.append("external_eval_plan.governance_handoff.requires_identical_heldout_scenarios must be true.")
        if handoff.get("external_eval_claims_allowed") is not False:
            target.errors.append(
                "external_eval_plan.governance_handoff.external_eval_claims_allowed must remain false until a passing external eval result exists."
            )
        if not isinstance(handoff.get("recommendation"), str) or not handoff.get("recommendation"):
            target.errors.append("external_eval_plan.governance_handoff.recommendation must be a non-empty string.")

    target.details.update(
        {
            "ready": plan.get("ready"),
            "adapter_count": len(adapters),
            "ready_adapter_count": ready_count,
            "selected_adapters": selected,
        }
    )
    for error in external_eval_plan_semantic_errors(plan):
        message = f"external_eval_plan semantic validation: {error}."
        if message not in target.errors:
            target.errors.append(message)

def _validate_external_eval_receipt(receipt: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(receipt, "schema_version", EXTERNAL_EVAL_RECEIPT_SCHEMA_VERSION, target, prefix="external_eval_receipt.")
    _validate_allowed_keys(
        receipt,
        {
            "schema_version",
            "created_at",
            "passed",
            "readiness",
            "recommendation",
            "check_count",
            "failed_check_count",
            "checks",
            "blocked_reasons",
            "source_plan",
            "adapter_count",
            "ready_adapter_count",
            "adapter_receipts",
            "launch",
            "execution_boundary",
            "notes",
        },
        target,
        "external_eval_receipt",
    )
    checks = receipt.get("checks")
    if not isinstance(checks, list):
        target.errors.append("external_eval_receipt.checks must be a list.")
        checks = []
    failed_checks = _validate_gate_like_checks(checks, target, "external_eval_receipt.checks")
    for index, check in enumerate(checks):
        if isinstance(check, dict):
            _validate_allowed_keys(
                check,
                {"id", "passed", "actual", "expected", "summary"},
                target,
                f"external_eval_receipt.checks[{index}]",
            )
    if receipt.get("check_count") != len(checks):
        target.errors.append(f"external_eval_receipt.check_count expected {len(checks)}, got {receipt.get('check_count')!r}.")
    if receipt.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"external_eval_receipt.failed_check_count expected {failed_checks}, got {receipt.get('failed_check_count')!r}."
        )
    if receipt.get("passed") != (failed_checks == 0):
        target.errors.append("external_eval_receipt.passed must match failed_check_count.")
    expected_readiness = "dry_run_recorded" if failed_checks == 0 else "blocked"
    if receipt.get("readiness") != expected_readiness:
        target.errors.append(f"external_eval_receipt.readiness expected {expected_readiness!r}, got {receipt.get('readiness')!r}.")
    expected_recommendation = "archive_external_eval_dry_run" if failed_checks == 0 else "keep_external_eval_claims_disabled"
    if receipt.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"external_eval_receipt.recommendation expected {expected_recommendation!r}, got {receipt.get('recommendation')!r}."
        )
    if not _is_string_list(receipt.get("blocked_reasons")):
        target.errors.append("external_eval_receipt.blocked_reasons must be a list of strings.")

    source_plan = receipt.get("source_plan")
    source_plan_path = None
    if not isinstance(source_plan, dict):
        target.errors.append("external_eval_receipt.source_plan must be an object.")
    else:
        source_plan_path = _validate_external_eval_receipt_source_plan(source_plan, target, source_path)

    adapter_receipts = receipt.get("adapter_receipts")
    if not isinstance(adapter_receipts, list):
        target.errors.append("external_eval_receipt.adapter_receipts must be a list.")
        adapter_receipts = []
    if receipt.get("adapter_count") != len(adapter_receipts):
        target.errors.append(
            f"external_eval_receipt.adapter_count expected {len(adapter_receipts)}, got {receipt.get('adapter_count')!r}."
        )
    ready_count = sum(1 for row in adapter_receipts if isinstance(row, dict) and row.get("ready") is True)
    if receipt.get("ready_adapter_count") != ready_count:
        target.errors.append(
            f"external_eval_receipt.ready_adapter_count expected {ready_count}, got {receipt.get('ready_adapter_count')!r}."
        )
    for index, row in enumerate(adapter_receipts):
        _validate_external_eval_adapter_receipt(row, index, target)

    _validate_external_eval_receipt_launch(receipt.get("launch"), target)
    _validate_external_eval_receipt_boundary(receipt.get("execution_boundary"), target)
    _validate_external_eval_receipt_replay(receipt, source_plan_path, adapter_receipts, target, source_path.parent)
    target.details.update(
        {
            "passed": receipt.get("passed"),
            "readiness": receipt.get("readiness"),
            "adapter_count": len(adapter_receipts),
            "ready_adapter_count": ready_count,
        }
    )

def _validate_external_eval_receipt_source_plan(source_plan: dict[str, Any], target: ValidationTarget, source_path: Path) -> Path | None:
    label = "external_eval_receipt.source_plan"
    _validate_allowed_keys(
        source_plan,
        {"path", "exists", "sha256", "size_bytes", "schema_version", "ready", "adapter_count"},
        target,
        label,
    )
    path_value = source_plan.get("path")
    if isinstance(path_value, str) and path_value and not _is_safe_or_redacted_external_eval_ref_path(path_value):
        target.errors.append(f"{label}.path must be relative to the external eval receipt.")
        return None
    if source_plan.get("exists") is not True:
        if source_plan.get("sha256") is not None:
            target.errors.append(f"{label}.sha256 must be null when exists is false.")
        if source_plan.get("size_bytes") is not None:
            target.errors.append(f"{label}.size_bytes must be null when exists is false.")
        if source_plan.get("schema_version") is not None:
            target.errors.append(f"{label}.schema_version must be null when exists is false.")
        if source_plan.get("ready") is not None:
            target.errors.append(f"{label}.ready must be null when exists is false.")
        if source_plan.get("adapter_count") is not None:
            target.errors.append(f"{label}.adapter_count must be null when exists is false.")
        return None
    if not isinstance(path_value, str) or not path_value:
        target.errors.append(f"{label}.path must be a non-empty string when exists is true.")
        return None
    if not _is_replayable_external_eval_ref_path(path_value):
        target.errors.append(f"{label}.path must be relative to the external eval receipt.")
        return None
    if source_plan.get("schema_version") != EXTERNAL_EVAL_PLAN_SCHEMA_VERSION:
        target.errors.append(f"{label}.schema_version must be {EXTERNAL_EVAL_PLAN_SCHEMA_VERSION!r}.")
    if not isinstance(source_plan.get("ready"), bool):
        target.errors.append(f"{label}.ready must be a boolean.")
    if not _is_non_negative_int(source_plan.get("adapter_count")):
        target.errors.append(f"{label}.adapter_count must be a non-negative integer.")
    plan_path = _external_eval_reference_path(path_value, source_path)
    if _path_has_symlink_component(plan_path, include_leaf=True):
        target.errors.append(f"{label}.path must resolve to a regular non-symlink external eval plan file.")
        return None
    if not plan_path.is_file():
        target.errors.append(f"{label}.path does not resolve to an external eval plan file.")
        return None
    if not _is_non_negative_int(source_plan.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer when exists is true.")
    elif plan_path.stat().st_size != source_plan.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if not _is_sha256(source_plan.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when exists is true.")
    elif _sha256(plan_path) != source_plan.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current file.")
    plan = _read_object(plan_path, target, f"{label}.path")
    if plan is not None:
        if plan.get("schema_version") != source_plan.get("schema_version"):
            target.errors.append(f"{label}.schema_version must match the current file.")
        if plan.get("ready") != source_plan.get("ready"):
            target.errors.append(f"{label}.ready must match the current file.")
        if plan.get("adapter_count") != source_plan.get("adapter_count"):
            target.errors.append(f"{label}.adapter_count must match the current file.")
    return plan_path

def _is_safe_or_redacted_external_eval_ref_path(value: str) -> bool:
    if value.startswith("<redacted:") and value.endswith(">"):
        basename = value.removeprefix("<redacted:").removesuffix(">")
        return bool(basename) and "/" not in basename and "\\" not in basename and ".." not in basename and "~" not in basename
    return _is_replayable_external_eval_ref_path(value)

def _is_replayable_external_eval_ref_path(value: str) -> bool:
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

def _validate_external_eval_receipt_replay(
    receipt: dict[str, Any],
    source_plan_path: Path | None,
    adapter_receipts: list[Any],
    target: ValidationTarget,
    output_base_dir: Path,
) -> None:
    if source_plan_path is None:
        return
    launch = receipt.get("launch") if isinstance(receipt.get("launch"), dict) else {}
    mode = launch.get("mode")
    if mode not in {"dry_run", "live"}:
        return
    adapter_ids = [row.get("id") for row in adapter_receipts if isinstance(row, dict) and row.get("id") in ADAPTERS]
    if len(adapter_ids) != len(adapter_receipts):
        return
    plan = _read_json_object_silent(source_plan_path)
    selected_adapters = plan.get("selected_adapters") if _is_string_list(plan.get("selected_adapters")) else []
    if sorted(adapter_ids) != sorted(selected_adapters):
        target.errors.append("external_eval_receipt.adapter_receipts must match current source plan selected_adapters.")
    if len(adapter_ids) != len(set(adapter_ids)):
        target.errors.append("external_eval_receipt.adapter_receipts must not contain duplicate adapter ids.")
        return
    try:
        expected = build_external_eval_receipt(
            plan_path=source_plan_path,
            adapters=selected_adapters,
            live=mode == "live",
            created_at=receipt.get("created_at") if isinstance(receipt.get("created_at"), str) else None,
            output_base_dir=output_base_dir,
        )
    except ExternalEvalPlanError as exc:
        target.errors.append(f"external_eval_receipt could not replay current source plan: {exc}")
        return
    for field_name in (
        "passed",
        "readiness",
        "recommendation",
        "check_count",
        "failed_check_count",
        "checks",
        "blocked_reasons",
        "adapter_count",
        "ready_adapter_count",
        "adapter_receipts",
        "launch",
        "execution_boundary",
    ):
        if receipt.get(field_name) != expected.get(field_name):
            target.errors.append(f"external_eval_receipt.{field_name} must match current source plan replay.")

def _validate_external_eval_adapter_receipt(row: Any, index: int, target: ValidationTarget) -> None:
    label = f"external_eval_receipt.adapter_receipts[{index}]"
    if not isinstance(row, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(
        row,
        {
            "id",
            "name",
            "domain",
            "ready",
            "planned_ready",
            "blocking_reasons",
            "dependency_status",
            "required_inputs",
            "provided_inputs",
            "adapter_contract",
            "live_benchmark_started",
            "provider_api_called",
            "model_downloads_started",
            "credential_values_recorded",
            "cost_incurred_usd",
        },
        target,
        label,
    )
    adapter_id = row.get("id")
    if adapter_id not in ADAPTERS:
        target.errors.append(f"{label}.id must be one of {sorted(ADAPTERS)!r}.")
        spec = None
    else:
        spec = ADAPTERS[adapter_id]
    if not isinstance(row.get("ready"), bool):
        target.errors.append(f"{label}.ready must be a boolean.")
    if not isinstance(row.get("planned_ready"), bool):
        target.errors.append(f"{label}.planned_ready must be a boolean.")
    if not _is_string_list(row.get("blocking_reasons")):
        target.errors.append(f"{label}.blocking_reasons must be a list of strings.")
    if row.get("live_benchmark_started") is not False:
        target.errors.append(f"{label}.live_benchmark_started must be false.")
    if row.get("provider_api_called") is not False:
        target.errors.append(f"{label}.provider_api_called must be false.")
    if row.get("model_downloads_started") is not False:
        target.errors.append(f"{label}.model_downloads_started must be false.")
    if row.get("credential_values_recorded") is not False:
        target.errors.append(f"{label}.credential_values_recorded must be false.")
    if row.get("cost_incurred_usd") != 0:
        target.errors.append(f"{label}.cost_incurred_usd must be 0.")
    _validate_external_eval_adapter_contract(row.get("adapter_contract"), target, f"{label}.adapter_contract", adapter_id)
    if spec is not None:
        if row.get("required_inputs") != spec["required_inputs"]:
            target.errors.append(f"{label}.required_inputs must match adapter contract.")
        if not isinstance(row.get("name"), str) or not row.get("name"):
            target.errors.append(f"{label}.name must be a non-empty string.")
        if row.get("domain") != spec["domain"]:
            target.errors.append(f"{label}.domain must match adapter contract.")
    if row.get("ready") is True and row.get("blocking_reasons"):
        target.errors.append(f"{label}.ready cannot be true while blockers remain.")

def _validate_external_eval_receipt_launch(launch: Any, target: ValidationTarget) -> None:
    label = "external_eval_receipt.launch"
    if not isinstance(launch, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(
        launch,
        {"mode", "live_benchmarks_started", "provider_api_called", "model_downloads_started", "cost_incurred_usd"},
        target,
        label,
    )
    if launch.get("mode") not in {"dry_run", "live"}:
        target.errors.append(f"{label}.mode must be dry_run or live.")
    for field_name in ("live_benchmarks_started", "provider_api_called", "model_downloads_started"):
        if launch.get(field_name) is not False:
            target.errors.append(f"{label}.{field_name} must be false.")
    if launch.get("cost_incurred_usd") != 0:
        target.errors.append(f"{label}.cost_incurred_usd must be 0.")

def _validate_external_eval_receipt_boundary(boundary: Any, target: ValidationTarget) -> None:
    label = "external_eval_receipt.execution_boundary"
    if not isinstance(boundary, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(
        boundary,
        {
            "dry_run_only",
            "live_benchmarks_started",
            "provider_api_called",
            "model_downloads_started",
            "cloud_cost_incurred_usd",
            "credential_values_recorded",
            "weights_updated_by_flight_recorder",
        },
        target,
        label,
    )
    if boundary.get("dry_run_only") is not True:
        target.errors.append(f"{label}.dry_run_only must be true.")
    for field_name in (
        "live_benchmarks_started",
        "provider_api_called",
        "model_downloads_started",
        "credential_values_recorded",
        "weights_updated_by_flight_recorder",
    ):
        if boundary.get(field_name) is not False:
            target.errors.append(f"{label}.{field_name} must be false.")
    if boundary.get("cloud_cost_incurred_usd") != 0:
        target.errors.append(f"{label}.cloud_cost_incurred_usd must be 0.")

def _validate_external_eval_inputs(inputs: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(
        inputs,
        {
            "scenario_manifest",
            "model_endpoint",
            "model",
            "tool_schema_set",
            "inspect_task_set",
            "lm_eval_task_list",
            "swe_bench_task_set",
            "sandbox_policy",
        },
        target,
        "external_eval_plan.inputs",
    )
    manifest = inputs.get("scenario_manifest")
    if not isinstance(manifest, dict):
        target.errors.append("external_eval_plan.inputs.scenario_manifest must be an object.")
    else:
        label = "external_eval_plan.inputs.scenario_manifest"
        _validate_allowed_keys(
            manifest,
            {"path", "exists", "sha256", "size_bytes", "schema_version", "ready", "scenario_count"},
            target,
            label,
        )
        if manifest.get("path") is not None and not isinstance(manifest.get("path"), str):
            target.errors.append(f"{label}.path must be a string or null.")
        if not isinstance(manifest.get("exists"), bool):
            target.errors.append(f"{label}.exists must be a boolean.")
        if manifest.get("sha256") is not None and not _is_sha256(manifest.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string or null.")
        if manifest.get("size_bytes") is not None and not _is_non_negative_int(manifest.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer or null.")
        if manifest.get("schema_version") is not None and not isinstance(manifest.get("schema_version"), str):
            target.errors.append(f"{label}.schema_version must be a string or null.")
        if manifest.get("ready") is not None and not isinstance(manifest.get("ready"), bool):
            target.errors.append(f"{label}.ready must be a boolean or null.")
        if manifest.get("scenario_count") is not None and not _is_non_negative_int(manifest.get("scenario_count")):
            target.errors.append(f"{label}.scenario_count must be a non-negative integer or null.")
        _validate_external_eval_scenario_manifest_file(manifest, target, label, source_path)
    for field_name in (
        "model_endpoint",
        "model",
        "tool_schema_set",
        "inspect_task_set",
        "swe_bench_task_set",
        "sandbox_policy",
    ):
        if inputs.get(field_name) is not None and not isinstance(inputs.get(field_name), str):
            target.errors.append(f"external_eval_plan.inputs.{field_name} must be a string or null.")
    if not _is_string_list(inputs.get("lm_eval_task_list")):
        target.errors.append("external_eval_plan.inputs.lm_eval_task_list must be a list of strings.")

def _validate_external_eval_adapter_plan(
    adapter: Any,
    index: int,
    target: ValidationTarget,
    inputs: dict[str, Any],
) -> None:
    if not isinstance(adapter, dict):
        target.errors.append(f"external_eval_plan.adapters[{index}] must be an object.")
        return
    _validate_allowed_keys(
        adapter,
        {
            "id",
            "name",
            "full_name",
            "domain",
            "suite_tags",
            "required_inputs",
            "provided_inputs",
            "dependency_status",
            "execution_contract",
            "adapter_contract",
            "ready",
            "blocking_reasons",
        },
        target,
        f"external_eval_plan.adapters[{index}]",
    )
    adapter_id = adapter.get("id")
    if adapter_id not in ADAPTERS:
        target.errors.append(f"external_eval_plan.adapters[{index}].id must be one of {sorted(ADAPTERS)!r}.")
        spec = None
    else:
        spec = ADAPTERS[adapter_id]
    for field_name in ("name", "full_name", "domain"):
        if not isinstance(adapter.get(field_name), str) or not adapter.get(field_name):
            target.errors.append(f"external_eval_plan.adapters[{index}].{field_name} must be a non-empty string.")
    for field_name in ("suite_tags", "required_inputs", "provided_inputs", "blocking_reasons"):
        if not _is_string_list(adapter.get(field_name)):
            target.errors.append(f"external_eval_plan.adapters[{index}].{field_name} must be a list of strings.")
    if not isinstance(adapter.get("ready"), bool):
        target.errors.append(f"external_eval_plan.adapters[{index}].ready must be a boolean.")
    dependency = adapter.get("dependency_status")
    available = False
    if not isinstance(dependency, dict):
        target.errors.append(f"external_eval_plan.adapters[{index}].dependency_status must be an object.")
    else:
        _validate_allowed_keys(
            dependency,
            {"available", "imports", "commands"},
            target,
            f"external_eval_plan.adapters[{index}].dependency_status",
        )
        if not isinstance(dependency.get("available"), bool):
            target.errors.append(f"external_eval_plan.adapters[{index}].dependency_status.available must be a boolean.")
        available = dependency.get("available") is True
        for field_name in ("imports", "commands"):
            values = dependency.get(field_name)
            if not isinstance(values, dict) or not all(isinstance(key, str) and isinstance(value, bool) for key, value in values.items()):
                target.errors.append(f"external_eval_plan.adapters[{index}].dependency_status.{field_name} must be an object of booleans.")

    contract = adapter.get("execution_contract")
    if not isinstance(contract, dict):
        target.errors.append(f"external_eval_plan.adapters[{index}].execution_contract must be an object.")
    else:
        _validate_allowed_keys(
            contract,
            {"requires_identical_heldout_scenarios", "scenario_manifest_sha256", "boundary"},
            target,
            f"external_eval_plan.adapters[{index}].execution_contract",
        )
        if contract.get("requires_identical_heldout_scenarios") is not True:
            target.errors.append(
                f"external_eval_plan.adapters[{index}].execution_contract.requires_identical_heldout_scenarios must be true."
            )
        manifest_sha = inputs.get("scenario_manifest", {}).get("sha256") if isinstance(inputs.get("scenario_manifest"), dict) else None
        if contract.get("scenario_manifest_sha256") != manifest_sha:
            target.errors.append(f"external_eval_plan.adapters[{index}].execution_contract.scenario_manifest_sha256 must match inputs.")
        if not isinstance(contract.get("boundary"), str) or not contract.get("boundary"):
            target.errors.append(f"external_eval_plan.adapters[{index}].execution_contract.boundary must be a non-empty string.")

    _validate_external_eval_adapter_contract(
        adapter.get("adapter_contract"),
        target,
        f"external_eval_plan.adapters[{index}].adapter_contract",
        adapter_id,
    )

    if spec is not None:
        if adapter.get("required_inputs") != spec["required_inputs"]:
            target.errors.append(f"external_eval_plan.adapters[{index}].required_inputs must match adapter contract.")
        expected_input_blockers = {
            reason
            for name in spec["required_inputs"]
            for reason in _external_eval_input_blockers(inputs, name)
        }
        adapter_blockers = set(adapter.get("blocking_reasons") if isinstance(adapter.get("blocking_reasons"), list) else [])
        if not available and "dependencies_missing" not in adapter_blockers:
            target.errors.append(f"external_eval_plan.adapters[{index}].blocking_reasons must include dependencies_missing.")
        if not expected_input_blockers.issubset(adapter_blockers):
            target.errors.append(f"external_eval_plan.adapters[{index}].blocking_reasons must include all missing required inputs.")
        expected_ready = available and not expected_input_blockers and not adapter_blockers
        if adapter.get("ready") is True and not expected_ready:
            target.errors.append(f"external_eval_plan.adapters[{index}].ready cannot be true while blockers remain.")
        if adapter.get("ready") is False and not adapter_blockers:
            target.errors.append(f"external_eval_plan.adapters[{index}].blocking_reasons must explain why ready is false.")

def _validate_external_eval_adapter_contract(contract: Any, target: ValidationTarget, label: str, adapter_id: Any) -> None:
    if not isinstance(contract, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(
        contract,
        {
            "schema_version",
            "adapter_id",
            "external_adapter_id",
            "receipt_types",
            "dry_run_transport",
            "live_benchmark_supported",
            "provider_api_called_by_flight_recorder",
            "model_downloads_started_by_flight_recorder",
            "credential_values_recorded",
            "cost_incurred_usd",
            "requires_identical_heldout_scenarios",
            "requires_external_runner_receipt_for_live",
            "requires_dependency_probe_before_live",
            "requires_explicit_live_opt_in",
        },
        target,
        label,
    )
    if contract.get("schema_version") != EXTERNAL_EVAL_ADAPTER_CONTRACT_VERSION:
        target.errors.append(f"{label}.schema_version must be {EXTERNAL_EVAL_ADAPTER_CONTRACT_VERSION!r}.")
    if contract.get("external_adapter_id") != adapter_id:
        target.errors.append(f"{label}.external_adapter_id must match adapter id.")
    if not isinstance(contract.get("adapter_id"), str) or not contract.get("adapter_id"):
        target.errors.append(f"{label}.adapter_id must be a non-empty string.")
    receipt_types = contract.get("receipt_types")
    if not _is_string_list(receipt_types):
        target.errors.append(f"{label}.receipt_types must be a list of strings.")
        receipt_types = []
    missing_receipts = sorted(set(EXTERNAL_EVAL_ADAPTER_RECEIPT_TYPES) - set(receipt_types))
    if missing_receipts:
        target.errors.append(f"{label}.receipt_types missing required receipt types: {', '.join(missing_receipts)}.")
    unsupported_receipts = sorted(set(receipt_types) - set(EXTERNAL_EVAL_ADAPTER_RECEIPT_TYPES))
    if unsupported_receipts:
        target.errors.append(f"{label}.receipt_types contains unsupported receipt types: {', '.join(unsupported_receipts)}.")
    if len(set(receipt_types)) != len(receipt_types):
        target.errors.append(f"{label}.receipt_types must not contain duplicates.")
    if contract.get("dry_run_transport") != "plan_and_receipt_only":
        target.errors.append(f"{label}.dry_run_transport must be plan_and_receipt_only.")
    for field_name in (
        "requires_identical_heldout_scenarios",
        "requires_external_runner_receipt_for_live",
        "requires_dependency_probe_before_live",
        "requires_explicit_live_opt_in",
    ):
        if contract.get(field_name) is not True:
            target.errors.append(f"{label}.{field_name} must be true.")
    for field_name in (
        "live_benchmark_supported",
        "provider_api_called_by_flight_recorder",
        "model_downloads_started_by_flight_recorder",
        "credential_values_recorded",
    ):
        if contract.get(field_name) is not False:
            target.errors.append(f"{label}.{field_name} must be false.")
    if contract.get("cost_incurred_usd") != 0:
        target.errors.append(f"{label}.cost_incurred_usd must be 0.")

def _external_eval_input_blockers(inputs: dict[str, Any], name: str) -> list[str]:
    if _external_eval_input_present(inputs, name):
        return []
    if name == "scenario_manifest":
        manifest = inputs.get("scenario_manifest")
        if isinstance(manifest, dict) and manifest.get("exists") is True and manifest.get("sha256") and manifest.get("ready") is False:
            return ["scenario_manifest_not_ready"]
    return [f"missing_{name}"]

def _external_eval_input_present(inputs: dict[str, Any], name: str) -> bool:
    if name == "scenario_manifest":
        manifest = inputs.get("scenario_manifest")
        return (
            isinstance(manifest, dict)
            and manifest.get("exists") is True
            and _is_sha256(manifest.get("sha256"))
            and _is_non_negative_int(manifest.get("size_bytes"))
            and manifest.get("ready") is not False
        )
    value = inputs.get(name)
    if isinstance(value, list):
        return bool(value)
    return isinstance(value, str) and bool(value)

def _validate_external_eval_scenario_manifest_file(
    manifest: dict[str, Any], target: ValidationTarget, label: str, source_path: Path
) -> None:
    path_value = manifest.get("path")
    if isinstance(path_value, str) and path_value and not _is_safe_or_redacted_external_eval_ref_path(path_value):
        target.errors.append(f"{label}.path must be relative to the external eval plan.")
        return
    if manifest.get("exists") is not True:
        if manifest.get("sha256") is not None:
            target.errors.append(f"{label}.sha256 must be null when exists is false.")
        if manifest.get("size_bytes") is not None:
            target.errors.append(f"{label}.size_bytes must be null when exists is false.")
        if manifest.get("schema_version") is not None:
            target.errors.append(f"{label}.schema_version must be null when exists is false.")
        if manifest.get("ready") is not None:
            target.errors.append(f"{label}.ready must be null when exists is false.")
        if manifest.get("scenario_count") is not None:
            target.errors.append(f"{label}.scenario_count must be null when exists is false.")
        return
    if not isinstance(path_value, str) or not path_value:
        target.errors.append(f"{label}.path must be a non-empty string when exists is true.")
        return
    if not _is_replayable_external_eval_ref_path(path_value):
        target.errors.append(f"{label}.path must be relative to the external eval plan.")
        return
    file_path = _external_eval_reference_path(path_value, source_path)
    if _path_has_symlink_component(file_path, include_leaf=True):
        target.errors.append(f"{label}.path must resolve to a regular non-symlink manifest file.")
        return
    if not file_path.is_file():
        target.errors.append(f"{label}.path does not resolve to a manifest file.")
        return
    current_manifest = _read_object(file_path, target, f"{label}.path")
    if current_manifest is not None:
        current_schema = current_manifest.get("schema_version")
        if current_schema != HELDOUT_MANIFEST_SCHEMA_VERSION:
            target.errors.append(f"{label}.path must reference a {HELDOUT_MANIFEST_SCHEMA_VERSION!r} manifest.")
        if manifest.get("schema_version") != current_schema:
            target.errors.append(f"{label}.schema_version must match the current file.")
        current_inspection = inspect_artifact_source(file_path, "heldout_manifest")
        current_ready = current_inspection.get("ready") is True
        if manifest.get("ready") != current_ready:
            target.errors.append(f"{label}.ready must match the current file.")
        if manifest.get("ready") is True and not current_ready:
            target.errors.append(f"{label}.path must pass held-out semantic validation when ready is true.")
        current_scenario_count = current_manifest.get("scenario_count")
        if _is_non_negative_int(current_scenario_count) and manifest.get("scenario_count") != current_scenario_count:
            target.errors.append(f"{label}.scenario_count must match the current file.")
    if not _is_non_negative_int(manifest.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer when exists is true.")
    elif file_path.stat().st_size != manifest.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if not _is_sha256(manifest.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when exists is true.")
    elif _sha256(file_path) != manifest.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current file.")

def _external_eval_reference_path(value: str, source_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return source_path.parent / path

_HELDOUT_MANIFEST_KEYS = {
    "schema_version",
    "generated_at",
    "ready",
    "status",
    "identical",
    "cross_arm_claims_allowed",
    "source_count",
    "scenario_count",
    "scenario_ids",
    "sources",
    "mismatches",
    "blocking_reasons",
    "governance_handoff",
}

_HELDOUT_MANIFEST_GOVERNANCE_KEYS = {
    "external_adapter_manifest_allowed",
    "cross_arm_claims_allowed",
    "recommendation",
}

_HELDOUT_MANIFEST_SOURCE_KEYS = {
    "label",
    "path",
    "schema_version",
    "scenario_count",
    "scenario_ids",
    "scenario_fingerprints",
    "duplicate_scenario_ids",
    "blocking_reasons",
}

_HELDOUT_MANIFEST_MISMATCH_KEYS = {
    "label",
    "missing_from_source",
    "extra_in_source",
    "fingerprint_mismatches",
}

_HELDOUT_MANIFEST_FINGERPRINT_MISMATCH_KEYS = {
    "scenario_id",
    "reference_sha256",
    "source_sha256",
}

def _validate_heldout_manifest(manifest: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(manifest, _HELDOUT_MANIFEST_KEYS, target, "heldout_manifest")
    _require_equal(manifest, "schema_version", HELDOUT_MANIFEST_SCHEMA_VERSION, target)
    status = manifest.get("status")
    if not isinstance(status, str) or status not in {
        "single_source",
        "identical",
        "mismatched",
        "empty",
        "blocked",
    }:
        target.errors.append("heldout_manifest.status has an unsupported value.")
    for field_name in ("ready", "identical", "cross_arm_claims_allowed"):
        if not isinstance(manifest.get(field_name), bool):
            target.errors.append(f"heldout_manifest.{field_name} must be a boolean.")
    if not _is_non_negative_int(manifest.get("source_count")):
        target.errors.append("heldout_manifest.source_count must be a non-negative integer.")
    if not _is_non_negative_int(manifest.get("scenario_count")):
        target.errors.append("heldout_manifest.scenario_count must be a non-negative integer.")
    manifest_scenario_ids = manifest.get("scenario_ids")
    if not _is_string_list(manifest_scenario_ids):
        target.errors.append("heldout_manifest.scenario_ids must be a list of strings.")
        manifest_scenario_ids = []
    if not _is_string_list(manifest.get("blocking_reasons")):
        target.errors.append("heldout_manifest.blocking_reasons must be a list of strings.")
    blocking_reasons = manifest.get("blocking_reasons") if isinstance(manifest.get("blocking_reasons"), list) else []

    sources = manifest.get("sources")
    if not isinstance(sources, list):
        target.errors.append("heldout_manifest.sources must be a list.")
        sources = []
    mismatches = manifest.get("mismatches")
    if not isinstance(mismatches, list):
        target.errors.append("heldout_manifest.mismatches must be a list.")
        mismatches = []

    if manifest.get("source_count") != len(sources):
        target.errors.append(f"heldout_manifest.source_count expected {len(sources)}, got {manifest.get('source_count')!r}.")
    if manifest.get("scenario_count") != len(manifest_scenario_ids):
        target.errors.append("heldout_manifest.scenario_count must match scenario_ids length.")
    if len(set(manifest_scenario_ids)) != len(manifest_scenario_ids):
        target.errors.append("heldout_manifest.scenario_ids must not contain duplicates.")

    for index, source in enumerate(sources):
        _validate_heldout_manifest_source(source, index, target, source_path)
    for index, mismatch in enumerate(mismatches):
        _validate_heldout_manifest_mismatch(mismatch, index, target)

    expected_status = manifest.get("status")
    expected_scenario_ids = manifest.get("scenario_ids") or []
    expected_mismatches = mismatches
    expected_blocking_reasons = blocking_reasons
    if _heldout_manifest_sources_replayable(sources):
        canonical_sources = _heldout_manifest_sources_with_identities(sources, source_path, target)
        (
            expected_status,
            expected_scenario_ids,
            expected_mismatches,
            expected_blocking_reasons,
        ) = _build_heldout_manifest_status(canonical_sources)
        derived_fields = {
            "status": expected_status,
            "scenario_ids": expected_scenario_ids,
            "mismatches": expected_mismatches,
            "blocking_reasons": expected_blocking_reasons,
        }
        mismatched_fields = [
            field_name
            for field_name, expected_value in derived_fields.items()
            if manifest.get(field_name) != expected_value
        ]
        if mismatched_fields:
            target.errors.append(
                "heldout_manifest does not match current source scenario IDs and fingerprints for: "
                f"{', '.join(mismatched_fields)}."
            )

    expected_ready = bool(expected_scenario_ids) and not expected_blocking_reasons
    expected_identical = expected_status == "identical"
    if isinstance(manifest.get("ready"), bool) and manifest.get("ready") != expected_ready:
        target.errors.append(f"heldout_manifest.ready expected {expected_ready}, got {manifest.get('ready')!r}.")
    if manifest.get("ready") is False and not blocking_reasons:
        target.errors.append("heldout_manifest.blocking_reasons must explain why ready is false.")
    if manifest.get("ready") is True and blocking_reasons:
        target.errors.append("heldout_manifest.blocking_reasons must be empty when ready is true.")
    if isinstance(manifest.get("identical"), bool) and manifest.get("identical") != expected_identical:
        target.errors.append(
            f"heldout_manifest.identical expected {expected_identical}, got {manifest.get('identical')!r}."
        )
    if (
        isinstance(manifest.get("cross_arm_claims_allowed"), bool)
        and manifest.get("cross_arm_claims_allowed") != expected_identical
    ):
        target.errors.append(
            "heldout_manifest.cross_arm_claims_allowed must match current source scenario fingerprint identity."
        )
    if (
        manifest.get("status") == "mismatched"
        and "heldout_scenario_set_mismatch" in expected_blocking_reasons
        and "heldout_scenario_set_mismatch" not in blocking_reasons
    ):
        target.errors.append(
            "heldout_manifest.blocking_reasons must include heldout_scenario_set_mismatch when mismatched."
        )
    if (
        manifest.get("status") == "mismatched"
        and "heldout_scenario_fingerprint_mismatch" in expected_blocking_reasons
        and "heldout_scenario_fingerprint_mismatch" not in blocking_reasons
    ):
        target.errors.append(
            "heldout_manifest.blocking_reasons must include heldout_scenario_fingerprint_mismatch when fingerprints differ."
        )

    handoff = manifest.get("governance_handoff")
    if not isinstance(handoff, dict):
        target.errors.append("heldout_manifest.governance_handoff must be an object.")
    else:
        _validate_allowed_keys(
            handoff,
            _HELDOUT_MANIFEST_GOVERNANCE_KEYS,
            target,
            "heldout_manifest.governance_handoff",
        )
        if handoff.get("external_adapter_manifest_allowed") != expected_ready:
            target.errors.append(
                "heldout_manifest.governance_handoff.external_adapter_manifest_allowed must match current source readiness."
            )
        if handoff.get("cross_arm_claims_allowed") != expected_identical:
            target.errors.append(
                "heldout_manifest.governance_handoff.cross_arm_claims_allowed must match current source scenario fingerprint identity."
            )
        if not isinstance(handoff.get("recommendation"), str) or not handoff.get("recommendation"):
            target.errors.append("heldout_manifest.governance_handoff.recommendation must be a non-empty string.")

    target.details.update(
        {
            "ready": manifest.get("ready"),
            "status": manifest.get("status"),
            "source_count": len(sources),
            "scenario_count": len(manifest_scenario_ids),
        }
    )

def _heldout_manifest_sources_replayable(sources: list[Any]) -> bool:
    return bool(sources) and all(
        isinstance(source, dict)
        and isinstance(source.get("label"), str)
        and bool(source.get("label"))
        and isinstance(source.get("path"), str)
        and bool(source.get("path"))
        and _is_string_list(source.get("scenario_ids"))
        and isinstance(source.get("scenario_fingerprints"), dict)
        and all(
            isinstance(scenario_id, str) and _is_lowercase_sha256(sha)
            for scenario_id, sha in source.get("scenario_fingerprints", {}).items()
        )
        and _is_string_list(source.get("blocking_reasons"))
        for source in sources
    )

def _heldout_manifest_sources_with_identities(
    sources: list[dict[str, Any]],
    manifest_path: Path,
    target: ValidationTarget,
) -> list[dict[str, Any]]:
    canonical_sources = []
    for index, source in enumerate(sources):
        source_file = _heldout_manifest_reference_path(source.get("path"), manifest_path)
        identity = str(source.get("path") or "")
        content_identity = None
        try:
            if source_file is not None:
                identity = str(source_file.resolve(strict=False))
            if (
                source_file is not None
                and source_file.is_file()
                and not _path_has_symlink_component(source_file, include_leaf=True)
            ):
                content_identity = _sha256(source_file)
        except (OSError, ValueError, RuntimeError):
            target.errors.append(
                f"heldout_manifest.sources[{index}].path could not be replayed for identity validation."
            )
        canonical_sources.append(
            {
                **source,
                "_source_identity": identity,
                "_source_content_identity": content_identity,
            }
        )
    return canonical_sources

def _validate_heldout_manifest_source(source: Any, index: int, target: ValidationTarget, source_path: Path) -> None:
    if not isinstance(source, dict):
        target.errors.append(f"heldout_manifest.sources[{index}] must be an object.")
        return
    label = f"heldout_manifest.sources[{index}]"
    _validate_allowed_keys(source, _HELDOUT_MANIFEST_SOURCE_KEYS, target, label)
    for field_name in ("label", "path"):
        if not isinstance(source.get(field_name), str) or not source.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if isinstance(source.get("path"), str) and source.get("path"):
        _warn_absolute_public_path(target, f"{label}.path", source.get("path"))
    if not _is_non_negative_int(source.get("scenario_count")):
        target.errors.append(f"{label}.scenario_count must be a non-negative integer.")
    if not _is_string_list(source.get("scenario_ids")):
        target.errors.append(f"{label}.scenario_ids must be a list of strings.")
    elif source.get("scenario_count") != len(source.get("scenario_ids")):
        target.errors.append(f"{label}.scenario_count must match scenario_ids length.")
    if not _is_string_list(source.get("duplicate_scenario_ids")):
        target.errors.append(f"{label}.duplicate_scenario_ids must be a list of strings.")
    if not _is_string_list(source.get("blocking_reasons")):
        target.errors.append(f"{label}.blocking_reasons must be a list of strings.")
    fingerprints = source.get("scenario_fingerprints")
    if not isinstance(fingerprints, dict):
        target.errors.append(f"{label}.scenario_fingerprints must be an object.")
    else:
        scenario_ids = source.get("scenario_ids") if _is_string_list(source.get("scenario_ids")) else []
        missing = sorted(set(scenario_ids) - set(fingerprints))
        extra = sorted(set(fingerprints) - set(scenario_ids))
        declared_blockers = (
            source.get("blocking_reasons") if _is_string_list(source.get("blocking_reasons")) else []
        )
        if extra or (missing and "missing_scenario_fingerprints" not in declared_blockers):
            target.errors.append(
                f"{label}.scenario_fingerprints keys must exactly match scenario_ids; "
                f"missing={missing!r}, extra={extra!r}."
            )
        for scenario_id, sha in fingerprints.items():
            if not isinstance(scenario_id, str) or not _is_lowercase_sha256(sha):
                target.errors.append(
                    f"{label}.scenario_fingerprints[{scenario_id!r}] must be a lowercase SHA-256 hex string."
                )
                break
    _validate_heldout_manifest_source_file(source, target, label, source_path)

def _validate_heldout_manifest_source_file(source: dict[str, Any], target: ValidationTarget, label: str, source_path: Path) -> None:
    file_path = _heldout_manifest_reference_path(source.get("path"), source_path)
    if file_path is None:
        target.errors.append(f"{label}.path must be replayable for held-out identity validation.")
        return
    try:
        if _path_has_symlink_component(file_path, include_leaf=True):
            target.errors.append(f"{label}.path must resolve to a regular non-symlink suite summary.")
            return
        if not file_path.exists() or not file_path.is_file():
            target.errors.append(f"{label}.path must resolve to an existing suite summary.")
            return
    except (OSError, ValueError, RuntimeError):
        target.errors.append(f"{label}.path could not be inspected as a suite summary.")
        return
    summary = _read_object(file_path, target, f"{label}.path")
    if summary is None:
        return
    current_source = _heldout_manifest_source_from_suite_summary(summary, file_path)
    for field_name in ("schema_version", "scenario_count", "scenario_ids", "scenario_fingerprints", "duplicate_scenario_ids", "blocking_reasons"):
        if source.get(field_name) != current_source[field_name]:
            target.errors.append(f"{label}.{field_name} must match the current suite summary.")

def _heldout_manifest_source_from_suite_summary(
    summary: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    return _build_heldout_source_fields(summary, source_path=source_path)

def _heldout_manifest_reference_path(value: Any, source_path: Path) -> Path | None:
    if not isinstance(value, str) or not value or value.startswith("<redacted:") or _is_windows_absolute(value):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return source_path.parent / path

def _validate_heldout_manifest_mismatch(mismatch: Any, index: int, target: ValidationTarget) -> None:
    if not isinstance(mismatch, dict):
        target.errors.append(f"heldout_manifest.mismatches[{index}] must be an object.")
        return
    _validate_allowed_keys(mismatch, _HELDOUT_MANIFEST_MISMATCH_KEYS, target, f"heldout_manifest.mismatches[{index}]")
    if not isinstance(mismatch.get("label"), str) or not mismatch.get("label"):
        target.errors.append(f"heldout_manifest.mismatches[{index}].label must be a non-empty string.")
    for field_name in ("missing_from_source", "extra_in_source"):
        if not _is_string_list(mismatch.get(field_name)):
            target.errors.append(f"heldout_manifest.mismatches[{index}].{field_name} must be a list of strings.")
    fingerprint_mismatches = mismatch.get("fingerprint_mismatches")
    if not isinstance(fingerprint_mismatches, list):
        target.errors.append(
            f"heldout_manifest.mismatches[{index}].fingerprint_mismatches must be a list."
        )
        return
    for fingerprint_index, fingerprint_mismatch in enumerate(fingerprint_mismatches):
        label = (
            f"heldout_manifest.mismatches[{index}]."
            f"fingerprint_mismatches[{fingerprint_index}]"
        )
        if not isinstance(fingerprint_mismatch, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(
            fingerprint_mismatch,
            _HELDOUT_MANIFEST_FINGERPRINT_MISMATCH_KEYS,
            target,
            label,
        )
        if not isinstance(fingerprint_mismatch.get("scenario_id"), str) or not fingerprint_mismatch.get(
            "scenario_id"
        ):
            target.errors.append(f"{label}.scenario_id must be a non-empty string.")
        for field_name in ("reference_sha256", "source_sha256"):
            value = fingerprint_mismatch.get(field_name)
            if value is not None and not _is_lowercase_sha256(value):
                target.errors.append(f"{label}.{field_name} must be a lowercase SHA-256 or null.")

def _validate_eval_suite_manifest(manifest: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(manifest, "schema_version", EVAL_SUITE_MANIFEST_SCHEMA_VERSION, target, prefix="eval_suite_manifest.")
    for field_name in ("suite_id", "description"):
        if not isinstance(manifest.get(field_name), str) or not manifest.get(field_name):
            target.errors.append(f"eval_suite_manifest.{field_name} must be a non-empty string.")

    tags = manifest.get("tags")
    if not _is_string_list(tags) or not tags or any(not tag for tag in tags or []):
        target.errors.append("eval_suite_manifest.tags must be a non-empty list of non-empty strings.")
        tags = []
    elif len(set(tags)) != len(tags):
        target.errors.append("eval_suite_manifest.tags must not contain duplicates.")

    scenario_ids = manifest.get("scenario_ids")
    if not _is_string_list(scenario_ids) or not scenario_ids or any(not scenario_id for scenario_id in scenario_ids or []):
        target.errors.append("eval_suite_manifest.scenario_ids must be a non-empty list of non-empty strings.")
        scenario_ids = []
    elif len(set(scenario_ids)) != len(scenario_ids):
        target.errors.append("eval_suite_manifest.scenario_ids must not contain duplicates.")

    notes = manifest.get("notes")
    if notes is not None and not _is_string_list(notes):
        target.errors.append("eval_suite_manifest.notes must be a list of strings when present.")

    target.details.update(
        {
            "suite_id": manifest.get("suite_id"),
            "scenario_count": len(scenario_ids),
            "tag_count": len(tags),
        }
    )

def _validate_suite_metrics(
    metrics: dict[str, Any],
    target: ValidationTarget,
    runs: list[dict[str, Any]],
) -> None:
    scores = [_score_value(run.get("score")) for run in runs]
    passed = sum(1 for run in runs if run.get("passed") is True)
    failed = len(runs) - passed
    expected_pass_rate = round(passed / len(runs), 4) if runs else 0.0
    expected_average = round(sum(scores) / len(scores), 2) if scores else 0.0
    expected_min = min(scores) if scores else None
    expected_max = max(scores) if scores else None
    expected_failed_rules = _count_strings(rule for run in runs for rule in run.get("failed_rules", []))
    expected_critical = _count_strings(rule for run in runs for rule in run.get("critical_failures", []))

    expected_values = {
        "pass_rate": expected_pass_rate,
        "average_score": expected_average,
        "min_score": expected_min,
        "max_score": expected_max,
        "passed": passed,
        "failed": failed,
    }
    for field_name, expected in expected_values.items():
        if metrics.get(field_name) != expected:
            target.errors.append(f"suite_summary.metrics.{field_name} expected {expected!r}, got {metrics.get(field_name)!r}.")

    if _count_rows(metrics.get("failed_rule_counts")) != expected_failed_rules:
        target.errors.append("suite_summary.metrics.failed_rule_counts does not match run failed_rules.")
    if _count_rows(metrics.get("critical_failure_counts")) != expected_critical:
        target.errors.append("suite_summary.metrics.critical_failure_counts does not match run critical_failures.")
    _validate_suite_family_metrics(metrics.get("task_families"), target, runs)

def _validate_suite_family_metrics(value: Any, target: ValidationTarget, runs: list[dict[str, Any]]) -> None:
    if not isinstance(value, list):
        target.errors.append("suite_summary.metrics.task_families must be a list.")
        return

    expected: dict[str, dict[str, Any]] = {}
    for run in runs:
        family = str(run.get("task_family") or "unknown")
        bucket = expected.setdefault(family, {"runs": []})
        bucket["runs"].append(run)
    expected_rows: dict[str, dict[str, Any]] = {}
    for family, bucket in expected.items():
        family_runs = bucket["runs"]
        scores = [_score_value(run.get("score")) for run in family_runs]
        passed = sum(1 for run in family_runs if run.get("passed") is True)
        expected_rows[family] = {
            "total": len(family_runs),
            "passed": passed,
            "failed": len(family_runs) - passed,
            "pass_rate": round(passed / len(family_runs), 4) if family_runs else 0.0,
            "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "failed_rule_counts": _count_strings(rule for run in family_runs for rule in run.get("failed_rules", [])),
            "critical_failure_counts": _count_strings(rule for run in family_runs for rule in run.get("critical_failures", [])),
        }

    actual_families: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            target.errors.append(f"suite_summary.metrics.task_families[{index}] must be an object.")
            continue
        family = row.get("task_family")
        if not isinstance(family, str) or not family:
            target.errors.append(f"suite_summary.metrics.task_families[{index}].task_family must be a non-empty string.")
            continue
        actual_families.add(family)
        expected_row = expected_rows.get(family)
        if expected_row is None:
            target.errors.append(f"suite_summary.metrics.task_families[{index}] has unknown task_family {family!r}.")
            continue
        for field_name in ("total", "passed", "failed", "pass_rate", "average_score"):
            if row.get(field_name) != expected_row[field_name]:
                target.errors.append(
                    f"suite_summary.metrics.task_families[{index}].{field_name} "
                    f"expected {expected_row[field_name]!r}, got {row.get(field_name)!r}."
                )
        if _count_rows(row.get("failed_rule_counts")) != expected_row["failed_rule_counts"]:
            target.errors.append(f"suite_summary.metrics.task_families[{index}].failed_rule_counts does not match runs.")
        if "critical_failure_counts" not in row:
            target.warnings.append(
                f"suite_summary.metrics.task_families[{index}].critical_failure_counts is missing; rerun run-suite to refresh family metrics."
            )
        elif _count_rows(row.get("critical_failure_counts")) != expected_row["critical_failure_counts"]:
            target.errors.append(f"suite_summary.metrics.task_families[{index}].critical_failure_counts does not match runs.")
    missing = sorted(set(expected_rows) - actual_families)
    if missing:
        target.errors.append(f"suite_summary.metrics.task_families missing families: {missing!r}.")

def _validate_evidence_coverage(coverage: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(coverage, "schema_version", EVIDENCE_COVERAGE_SCHEMA_VERSION, target)
    runs = coverage.get("runs")
    if not isinstance(runs, list):
        target.errors.append("evidence_coverage.runs must be a list.")
        runs = []
    metrics = coverage.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("evidence_coverage.metrics must be an object.")
        metrics = {}
    checks = coverage.get("checks")
    if not isinstance(checks, list):
        target.errors.append("evidence_coverage.checks must be a list.")
        checks = []
    if not isinstance(coverage.get("passed"), bool):
        target.errors.append("evidence_coverage.passed must be a boolean.")

    failed_checks = 0
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            target.errors.append(f"evidence_coverage.checks[{index}] must be an object.")
            continue
        if not isinstance(check.get("id"), str) or not check.get("id"):
            target.errors.append(f"evidence_coverage.checks[{index}].id must be a non-empty string.")
        if not isinstance(check.get("passed"), bool):
            target.errors.append(f"evidence_coverage.checks[{index}].passed must be a boolean.")
        elif not check["passed"]:
            failed_checks += 1

    if coverage.get("check_count") != len(checks):
        target.errors.append(f"evidence_coverage.check_count expected {len(checks)}, got {coverage.get('check_count')!r}.")
    if coverage.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"evidence_coverage.failed_check_count expected {failed_checks}, got {coverage.get('failed_check_count')!r}."
        )
    if isinstance(coverage.get("passed"), bool) and coverage["passed"] != (failed_checks == 0):
        target.errors.append("evidence_coverage.passed must match failed_check_count.")

    run_totals = _validate_evidence_coverage_runs(runs, target)
    _validate_evidence_coverage_metrics(metrics, run_totals, target)

    warnings = coverage.get("warnings")
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        target.errors.append("evidence_coverage.warnings must be a list of strings.")

    target.details.update(
        {
            "run_count": run_totals["run_count"],
            "failed_rule_count": run_totals["failed_rule_count"],
            "failed_rule_evidence_rate": metrics.get("failed_rule_evidence_rate"),
        }
    )

def _validate_evidence_coverage_runs(runs: list[Any], target: ValidationTarget) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "run_count": len(runs),
        "rule_count": 0,
        "failed_rule_count": 0,
        "critical_failed_rule_count": 0,
        "evidence_ref_count": 0,
        "failed_rule_evidence_ref_count": 0,
        "critical_failed_rule_evidence_ref_count": 0,
        "failed_rules_with_evidence": 0,
        "failed_rules_without_evidence": 0,
        "critical_failed_rules_with_evidence": 0,
        "critical_failed_rules_without_evidence": 0,
        "task_evidence_ref_count": 0,
        "evidence_target_counts": {},
        "failed_rule_evidence_target_counts": {},
    }
    for index, run in enumerate(runs):
        label = f"evidence_coverage.runs[{index}]"
        if not isinstance(run, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("scenario_id", "scenario_title", "run_dir"):
            if not isinstance(run.get(field_name), str) or not run.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        _warn_absolute_public_path(target, f"{label}.run_dir", run.get("run_dir"))
        if not isinstance(run.get("passed"), bool):
            target.errors.append(f"{label}.passed must be a boolean.")
        if not _is_int_between(run.get("score"), 0, 100):
            target.errors.append(f"{label}.score must be an integer from 0 to 100.")
        for field_name in (
            "rule_count",
            "failed_rule_count",
            "critical_failed_rule_count",
            "evidence_ref_count",
            "failed_rule_evidence_ref_count",
            "critical_failed_rule_evidence_ref_count",
            "failed_rules_with_evidence",
            "critical_failed_rules_with_evidence",
            "task_evidence_ref_count",
        ):
            if not _is_non_negative_int(run.get(field_name)):
                target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
                continue
            totals[field_name] += run[field_name]
        for field_name in ("failed_rules_without_evidence", "critical_failed_rules_without_evidence"):
            values = run.get(field_name)
            if not _is_string_list(values):
                target.errors.append(f"{label}.{field_name} must be a list of strings.")
                continue
            totals[field_name] += len(values)
        event_count = run.get("event_count")
        if event_count is not None and not _is_non_negative_int(event_count):
            target.errors.append(f"{label}.event_count must be a non-negative integer or null.")
        _merge_count_rows(totals["evidence_target_counts"], run.get("evidence_target_counts"), target, f"{label}.evidence_target_counts")
        _merge_count_rows(
            totals["failed_rule_evidence_target_counts"],
            run.get("failed_rule_evidence_target_counts"),
            target,
            f"{label}.failed_rule_evidence_target_counts",
        )
        rules = run.get("rules")
        if not isinstance(rules, list):
            target.errors.append(f"{label}.rules must be a list.")
        elif len(rules) != run.get("rule_count"):
            target.errors.append(f"{label}.rule_count must match rules length.")
    return totals

def _validate_evidence_coverage_metrics(metrics: dict[str, Any], totals: dict[str, Any], target: ValidationTarget) -> None:
    expected_int_fields = (
        "run_count",
        "rule_count",
        "failed_rule_count",
        "critical_failed_rule_count",
        "evidence_ref_count",
        "failed_rule_evidence_ref_count",
        "critical_failed_rule_evidence_ref_count",
        "failed_rules_with_evidence",
        "failed_rules_without_evidence",
        "critical_failed_rules_with_evidence",
        "critical_failed_rules_without_evidence",
        "task_evidence_ref_count",
    )
    for field_name in expected_int_fields:
        if metrics.get(field_name) != totals[field_name]:
            target.errors.append(f"evidence_coverage.metrics.{field_name} expected {totals[field_name]!r}, got {metrics.get(field_name)!r}.")

    expected_failed_rate = _rate_value(totals["failed_rules_with_evidence"], totals["failed_rule_count"])
    expected_critical_rate = _rate_value(totals["critical_failed_rules_with_evidence"], totals["critical_failed_rule_count"])
    if metrics.get("failed_rule_evidence_rate") != expected_failed_rate:
        target.errors.append(
            f"evidence_coverage.metrics.failed_rule_evidence_rate expected {expected_failed_rate!r}, "
            f"got {metrics.get('failed_rule_evidence_rate')!r}."
        )
    if metrics.get("critical_failed_rule_evidence_rate") != expected_critical_rate:
        target.errors.append(
            f"evidence_coverage.metrics.critical_failed_rule_evidence_rate expected {expected_critical_rate!r}, "
            f"got {metrics.get('critical_failed_rule_evidence_rate')!r}."
        )

    evidence_counts = _count_rows(metrics.get("evidence_target_counts"))
    failed_counts = _count_rows(metrics.get("failed_rule_evidence_target_counts"))
    if evidence_counts != totals["evidence_target_counts"]:
        target.errors.append("evidence_coverage.metrics.evidence_target_counts does not match runs.")
    if failed_counts != totals["failed_rule_evidence_target_counts"]:
        target.errors.append("evidence_coverage.metrics.failed_rule_evidence_target_counts does not match runs.")
    if metrics.get("event_evidence_ref_count") != totals["evidence_target_counts"].get("event", 0):
        target.errors.append("evidence_coverage.metrics.event_evidence_ref_count does not match evidence_target_counts.")
    if metrics.get("final_answer_evidence_ref_count") != totals["evidence_target_counts"].get("final_answer", 0):
        target.errors.append("evidence_coverage.metrics.final_answer_evidence_ref_count does not match evidence_target_counts.")
    if metrics.get("episode_evidence_ref_count") != totals["evidence_target_counts"].get("episode", 0):
        target.errors.append("evidence_coverage.metrics.episode_evidence_ref_count does not match evidence_target_counts.")

    rule_coverage = metrics.get("rule_coverage")
    if not isinstance(rule_coverage, list):
        target.errors.append("evidence_coverage.metrics.rule_coverage must be a list.")
        return
    for index, row in enumerate(rule_coverage):
        if not isinstance(row, dict):
            target.errors.append(f"evidence_coverage.metrics.rule_coverage[{index}] must be an object.")
            continue
        if "target_counts" in row:
            target.errors.append(f"evidence_coverage.metrics.rule_coverage[{index}].target_counts is internal and must not be present.")
        if not isinstance(row.get("rule_id"), str) or not row.get("rule_id"):
            target.errors.append(f"evidence_coverage.metrics.rule_coverage[{index}].rule_id must be a non-empty string.")
        for field_name in (
            "rule_count",
            "passed",
            "failed",
            "critical_failed",
            "evidence_ref_count",
            "negative_evidence_ref_count",
            "failed_with_evidence",
            "failed_without_evidence",
        ):
            if not _is_non_negative_int(row.get(field_name)):
                target.errors.append(f"evidence_coverage.metrics.rule_coverage[{index}].{field_name} must be a non-negative integer.")

def _validate_scenario_check(check: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(check, "schema_version", SCENARIO_CHECK_SCHEMA_VERSION, target, prefix="scenario_check.")
    if not isinstance(check.get("scenarios_dir"), str) or not check.get("scenarios_dir"):
        target.errors.append("scenario_check.scenarios_dir must be a non-empty string.")
    else:
        _warn_absolute_public_path(target, "scenario_check.scenarios_dir", check.get("scenarios_dir"))
    if not isinstance(check.get("pattern"), str) or not check.get("pattern"):
        target.errors.append("scenario_check.pattern must be a non-empty string.")
    for field_name in ("recursive", "strict", "require_traces", "passed"):
        if not isinstance(check.get(field_name), bool):
            target.errors.append(f"scenario_check.{field_name} must be a boolean.")

    scenarios = check.get("scenarios")
    if not isinstance(scenarios, list):
        target.errors.append("scenario_check.scenarios must be a list.")
        scenarios = []
    duplicates = check.get("duplicates")
    if not isinstance(duplicates, list):
        target.errors.append("scenario_check.duplicates must be a list.")
        duplicates = []

    totals = _validate_scenario_check_rows(scenarios, target)
    duplicate_count = _validate_scenario_check_duplicates(duplicates, target)
    expected_passed = totals["error_count"] == 0 and (totals["warning_count"] == 0 or check.get("strict") is not True)
    expected_counts = {
        "total": len(scenarios),
        "error_count": totals["error_count"],
        "warning_count": totals["warning_count"],
        "duplicate_id_count": duplicate_count,
    }
    for field_name, expected_value in expected_counts.items():
        if check.get(field_name) != expected_value:
            target.errors.append(f"scenario_check.{field_name} expected {expected_value!r}, got {check.get(field_name)!r}.")
    if isinstance(check.get("passed"), bool) and check["passed"] != expected_passed:
        target.errors.append("scenario_check.passed must match errors, warnings, and strict mode.")
    target.details.update(
        {
            "scenario_count": len(scenarios),
            "error_count": totals["error_count"],
            "warning_count": totals["warning_count"],
            "duplicate_id_count": duplicate_count,
        }
    )

def _validate_scenario_check_rows(scenarios: list[Any], target: ValidationTarget) -> dict[str, int]:
    totals = {"error_count": 0, "warning_count": 0}
    for index, row in enumerate(scenarios):
        label = f"scenario_check.scenarios[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        if not isinstance(row.get("path"), str) or not row.get("path"):
            target.errors.append(f"{label}.path must be a non-empty string.")
        _warn_absolute_public_path(target, f"{label}.path", row.get("path"))
        if not isinstance(row.get("passed"), bool):
            target.errors.append(f"{label}.passed must be a boolean.")
        for field_name in ("id", "title"):
            if field_name in row and (not isinstance(row.get(field_name), str) or not row.get(field_name)):
                target.errors.append(f"{label}.{field_name} must be a non-empty string when present.")
        for field_name in ("trace_exists", "before_state_exists", "state_exists", "trace_required"):
            if field_name in row and not isinstance(row.get(field_name), bool):
                target.errors.append(f"{label}.{field_name} must be a boolean when present.")
        for field_name in ("trace_path", "before_state_path", "state_path"):
            if field_name in row:
                if not isinstance(row.get(field_name), str) or not row.get(field_name):
                    target.errors.append(f"{label}.{field_name} must be a non-empty string when present.")
                _warn_absolute_public_path(target, f"{label}.{field_name}", row.get(field_name))

        errors = row.get("errors")
        if not _is_string_list(errors):
            target.errors.append(f"{label}.errors must be a list of strings.")
            errors = []
        warnings = row.get("warnings")
        if not _is_string_list(warnings):
            target.errors.append(f"{label}.warnings must be a list of strings.")
            warnings = []
        totals["error_count"] += len(errors)
        totals["warning_count"] += len(warnings)
        if isinstance(row.get("passed"), bool) and row["passed"] != (len(errors) == 0):
            target.errors.append(f"{label}.passed must match errors.")
    return totals

def _validate_scenario_check_duplicates(duplicates: list[Any], target: ValidationTarget) -> int:
    duplicate_count = 0
    for index, row in enumerate(duplicates):
        label = f"scenario_check.duplicates[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        duplicate_count += 1
        for field_name in ("id", "first_path", "duplicate_path"):
            if not isinstance(row.get(field_name), str) or not row.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        _warn_absolute_public_path(target, f"{label}.first_path", row.get("first_path"))
        _warn_absolute_public_path(target, f"{label}.duplicate_path", row.get("duplicate_path"))
    return duplicate_count

def _validate_scenario_quality(quality: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(quality, "schema_version", SCENARIO_QUALITY_SCHEMA_VERSION, target)
    if not isinstance(quality.get("scenarios_dir"), str) or not quality.get("scenarios_dir"):
        target.errors.append("scenario_quality.scenarios_dir must be a non-empty string.")
    else:
        _warn_absolute_public_path(target, "scenario_quality.scenarios_dir", quality.get("scenarios_dir"))
    scenarios = quality.get("scenarios")
    if not isinstance(scenarios, list):
        target.errors.append("scenario_quality.scenarios must be a list.")
        scenarios = []
    metrics = quality.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("scenario_quality.metrics must be an object.")
        metrics = {}
    checks = quality.get("checks")
    if not isinstance(checks, list):
        target.errors.append("scenario_quality.checks must be a list.")
        checks = []
    if not isinstance(quality.get("passed"), bool):
        target.errors.append("scenario_quality.passed must be a boolean.")

    failed_checks = 0
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            target.errors.append(f"scenario_quality.checks[{index}] must be an object.")
            continue
        if not isinstance(check.get("id"), str) or not check.get("id"):
            target.errors.append(f"scenario_quality.checks[{index}].id must be a non-empty string.")
        if not isinstance(check.get("passed"), bool):
            target.errors.append(f"scenario_quality.checks[{index}].passed must be a boolean.")
        elif not check["passed"]:
            failed_checks += 1
    if quality.get("check_count") != len(checks):
        target.errors.append(f"scenario_quality.check_count expected {len(checks)}, got {quality.get('check_count')!r}.")
    if quality.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"scenario_quality.failed_check_count expected {failed_checks}, got {quality.get('failed_check_count')!r}."
        )
    if isinstance(quality.get("passed"), bool) and quality["passed"] != (failed_checks == 0):
        target.errors.append("scenario_quality.passed must match failed_check_count.")

    totals = _validate_scenario_quality_rows(scenarios, target)
    _validate_scenario_quality_metrics(metrics, totals, target)
    target.details.update(
        {
            "scenario_count": totals["scenario_count"],
            "average_contract_score": metrics.get("average_contract_score"),
            "observable_scenario_rate": metrics.get("observable_scenario_rate"),
        }
    )

def _validate_scenario_quality_rows(scenarios: list[Any], target: ValidationTarget) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "scenario_count": len(scenarios),
        "valid_scenario_count": 0,
        "invalid_scenario_count": 0,
        "scores": [],
        "task_families": set(),
        "observable_scenario_count": 0,
        "weak_scenario_count": 0,
        "final_only_scenario_count": 0,
        "missing_trace_count": 0,
        "missing_state_count": 0,
        "risk_counts": {},
    }
    for index, row in enumerate(scenarios):
        label = f"scenario_quality.scenarios[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("path", "id", "title", "task_family", "quality"):
            if not isinstance(row.get(field_name), str) or not row.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        _warn_absolute_public_path(target, f"{label}.path", row.get("path"))
        if row.get("quality") not in {"strong", "moderate", "weak", "invalid"}:
            target.errors.append(f"{label}.quality must be strong, moderate, weak, or invalid.")
        if not _is_int_between(row.get("contract_score"), 0, 100):
            target.errors.append(f"{label}.contract_score must be an integer from 0 to 100.")
        errors = row.get("errors")
        if not isinstance(errors, list) or not all(isinstance(item, str) for item in errors):
            target.errors.append(f"{label}.errors must be a list of strings.")
            errors = []
        risks = row.get("risks")
        if not isinstance(risks, list) or not all(isinstance(item, str) for item in risks):
            target.errors.append(f"{label}.risks must be a list of strings.")
            risks = []
        if errors:
            totals["invalid_scenario_count"] += 1
            continue
        totals["valid_scenario_count"] += 1
        totals["scores"].append(row["contract_score"])
        totals["task_families"].add(str(row.get("task_family") or "unknown"))
        signals = row.get("signals")
        if not isinstance(signals, dict):
            target.errors.append(f"{label}.signals must be an object for valid scenarios.")
            signals = {}
        if _non_negative_int_value(signals.get("observable_assertion_count")) > 0:
            totals["observable_scenario_count"] += 1
        if row.get("quality") == "weak":
            totals["weak_scenario_count"] += 1
        if "final_only_contract" in risks:
            totals["final_only_scenario_count"] += 1
        trace = row.get("trace")
        if isinstance(trace, dict):
            _warn_absolute_public_path(target, f"{label}.trace.trace_path", trace.get("trace_path"))
        if isinstance(trace, dict) and trace.get("trace_exists") is not True:
            totals["missing_trace_count"] += 1
        state = row.get("state")
        if isinstance(state, dict):
            for field_name in ("before_state_path", "state_path"):
                _warn_absolute_public_path(target, f"{label}.state.{field_name}", state.get(field_name))
        if "missing_state_file" in risks or "required_state_without_snapshot_path" in risks:
            totals["missing_state_count"] += 1
        for risk in risks:
            totals["risk_counts"][risk] = totals["risk_counts"].get(risk, 0) + 1
    return totals

def _validate_scenario_quality_metrics(metrics: dict[str, Any], totals: dict[str, Any], target: ValidationTarget) -> None:
    scores = totals["scores"]
    expected = {
        "scenario_count": totals["scenario_count"],
        "valid_scenario_count": totals["valid_scenario_count"],
        "invalid_scenario_count": totals["invalid_scenario_count"],
        "task_family_count": len(totals["task_families"]),
        "average_contract_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "min_contract_score": min(scores) if scores else 0,
        "max_contract_score": max(scores) if scores else 0,
        "observable_scenario_count": totals["observable_scenario_count"],
        "observable_scenario_rate": _rate_zero(totals["observable_scenario_count"], totals["valid_scenario_count"]),
        "weak_scenario_count": totals["weak_scenario_count"],
        "final_only_scenario_count": totals["final_only_scenario_count"],
        "missing_trace_count": totals["missing_trace_count"],
        "missing_state_count": totals["missing_state_count"],
    }
    for field_name, expected_value in expected.items():
        if metrics.get(field_name) != expected_value:
            target.errors.append(f"scenario_quality.metrics.{field_name} expected {expected_value!r}, got {metrics.get(field_name)!r}.")
    task_families = metrics.get("task_families")
    if task_families != sorted(totals["task_families"]):
        target.errors.append("scenario_quality.metrics.task_families does not match scenarios.")
    if _count_rows(metrics.get("risk_counts")) != totals["risk_counts"]:
        target.errors.append("scenario_quality.metrics.risk_counts does not match scenarios.")

def _rate_zero(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)

def _validate_suite_trend(trend: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(trend, "schema_version", SUITE_TREND_SCHEMA_VERSION, target)
    points = trend.get("points")
    if not isinstance(points, list):
        target.errors.append("suite_trend.points must be a list.")
        points = []
    if trend.get("point_count") != len(points):
        target.errors.append(f"suite_trend.point_count expected {len(points)}, got {trend.get('point_count')!r}.")

    point_labels: list[str] = []
    failed_counts_by_point: list[dict[str, int]] = []
    critical_counts_by_point: list[dict[str, int]] = []
    previous_point: dict[str, Any] | None = None
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            target.errors.append(f"suite_trend.points[{index}] must be an object.")
            point_labels.append("")
            failed_counts_by_point.append({})
            critical_counts_by_point.append({})
            continue
        failed_counts, critical_counts = _validate_suite_trend_point(point, target, index, previous_point)
        point_labels.append(str(point.get("label") or ""))
        failed_counts_by_point.append(failed_counts)
        critical_counts_by_point.append(critical_counts)
        previous_point = point

    _validate_suite_trend_count_rows(
        trend.get("failed_rule_trends"),
        target,
        "suite_trend.failed_rule_trends",
        points,
        point_labels,
        failed_counts_by_point,
    )
    _validate_suite_trend_count_rows(
        trend.get("critical_failure_trends"),
        target,
        "suite_trend.critical_failure_trends",
        points,
        point_labels,
        critical_counts_by_point,
    )

    summary = trend.get("summary")
    if not isinstance(summary, str) or not summary:
        target.errors.append("suite_trend.summary must be a non-empty string.")
    else:
        expected_summary = _expected_suite_trend_summary([point for point in points if isinstance(point, dict)])
        if summary != expected_summary:
            target.errors.append(f"suite_trend.summary expected {expected_summary!r}, got {summary!r}.")

    target.details.update(
        {
            "point_count": len(points),
            "failed_rule_trend_count": len(trend.get("failed_rule_trends", []))
            if isinstance(trend.get("failed_rule_trends"), list)
            else None,
            "critical_failure_trend_count": len(trend.get("critical_failure_trends", []))
            if isinstance(trend.get("critical_failure_trends"), list)
            else None,
        }
    )

def _validate_suite_trend_point(
    point: dict[str, Any],
    target: ValidationTarget,
    index: int,
    previous_point: dict[str, Any] | None,
) -> tuple[dict[str, int], dict[str, int]]:
    label = f"suite_trend.points[{index}]"
    if point.get("index") != index:
        target.errors.append(f"{label}.index expected {index}, got {point.get('index')!r}.")
    for field_name in ("label", "path"):
        if not isinstance(point.get(field_name), str) or not point.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if isinstance(point.get("path"), str) and _looks_absolute(point["path"]):
        target.warnings.append(f"{label}.path is absolute; prefer redacted or relative trend artifacts for sharing.")
    if "metadata" in point:
        _validate_metadata(point.get("metadata"), target, f"{label}.metadata")

    for field_name in ("total", "passed", "failed", "error_count", "failed_rule_count", "critical_failure_count"):
        if not _is_non_negative_int(point.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    if _is_non_negative_int(point.get("total")) and _is_non_negative_int(point.get("passed")) and _is_non_negative_int(point.get("failed")):
        expected_total = point["passed"] + point["failed"]
        if point["total"] != expected_total:
            target.errors.append(f"{label}.total expected passed + failed ({expected_total}), got {point['total']!r}.")

    if not _is_number_between(point.get("pass_rate"), 0.0, 1.0):
        target.errors.append(f"{label}.pass_rate must be numeric from 0.0 to 1.0.")
    if not _is_number_between(point.get("average_score"), 0.0, 100.0):
        target.errors.append(f"{label}.average_score must be numeric from 0.0 to 100.0.")

    failed_counts = _validate_count_map_object(point.get("failed_rule_counts"), target, f"{label}.failed_rule_counts")
    critical_counts = _validate_count_map_object(point.get("critical_failure_counts"), target, f"{label}.critical_failure_counts")
    if _is_non_negative_int(point.get("failed_rule_count")) and point["failed_rule_count"] != sum(failed_counts.values()):
        target.errors.append(
            f"{label}.failed_rule_count expected sum of failed_rule_counts ({sum(failed_counts.values())}), got {point['failed_rule_count']!r}."
        )
    if _is_non_negative_int(point.get("critical_failure_count")) and point["critical_failure_count"] != sum(critical_counts.values()):
        target.errors.append(
            f"{label}.critical_failure_count expected sum of critical_failure_counts ({sum(critical_counts.values())}), "
            f"got {point['critical_failure_count']!r}."
        )

    delta = point.get("delta_from_previous")
    if index == 0:
        if delta is not None:
            target.errors.append(f"{label}.delta_from_previous must be null for the first point.")
    elif not isinstance(delta, dict):
        target.errors.append(f"{label}.delta_from_previous must be an object.")
    elif previous_point is not None:
        expected_delta = _expected_suite_trend_delta(previous_point, point)
        for field_name, expected in expected_delta.items():
            if delta.get(field_name) != expected:
                target.errors.append(f"{label}.delta_from_previous.{field_name} expected {expected!r}, got {delta.get(field_name)!r}.")
    return failed_counts, critical_counts

def _validate_suite_trend_count_rows(
    rows: Any,
    target: ValidationTarget,
    label: str,
    points: list[Any],
    point_labels: list[str],
    counts_by_point: list[dict[str, int]],
) -> None:
    if not isinstance(rows, list):
        target.errors.append(f"{label} must be a list.")
        return
    expected_ids = sorted({rule_id for counts in counts_by_point for rule_id in counts})
    actual_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        row_label = f"{label}[{row_index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{row_label} must be an object.")
            continue
        rule_id = row.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            target.errors.append(f"{row_label}.id must be a non-empty string.")
            continue
        actual_ids.add(rule_id)
        counts = row.get("counts")
        if not isinstance(counts, list):
            target.errors.append(f"{row_label}.counts must be a list.")
            counts = []
        if len(counts) != len(points):
            target.errors.append(f"{row_label}.counts expected {len(points)} rows, got {len(counts)}.")

        observed_counts: list[int] = []
        for point_index, count_row in enumerate(counts):
            count_label = f"{row_label}.counts[{point_index}]"
            if not isinstance(count_row, dict):
                target.errors.append(f"{count_label} must be an object.")
                observed_counts.append(0)
                continue
            expected_count = counts_by_point[point_index].get(rule_id, 0) if point_index < len(counts_by_point) else 0
            count = count_row.get("count")
            if count_row.get("index") != point_index:
                target.errors.append(f"{count_label}.index expected {point_index}, got {count_row.get('index')!r}.")
            expected_label = point_labels[point_index] if point_index < len(point_labels) else ""
            if count_row.get("label") != expected_label:
                target.errors.append(f"{count_label}.label expected {expected_label!r}, got {count_row.get('label')!r}.")
            if not _is_non_negative_int(count):
                target.errors.append(f"{count_label}.count must be a non-negative integer.")
                observed_counts.append(0)
            else:
                observed_counts.append(count)
                if count != expected_count:
                    target.errors.append(f"{count_label}.count expected {expected_count}, got {count!r}.")

        expected_first = observed_counts[0] if observed_counts else 0
        expected_last = observed_counts[-1] if observed_counts else 0
        expected_delta = expected_last - expected_first
        for field_name, expected in (
            ("first_count", expected_first),
            ("last_count", expected_last),
            ("delta", expected_delta),
        ):
            if row.get(field_name) != expected:
                target.errors.append(f"{row_label}.{field_name} expected {expected!r}, got {row.get(field_name)!r}.")

    missing = sorted(set(expected_ids) - actual_ids)
    unexpected = sorted(actual_ids - set(expected_ids))
    if missing:
        target.errors.append(f"{label} missing rule IDs: {missing!r}.")
    if unexpected:
        target.errors.append(f"{label} has unexpected rule IDs: {unexpected!r}.")
