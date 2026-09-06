"""Extracted validation implementation."""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Any
from ..cloud_training import CLOUD_TRAINING_ARTIFACT_MANIFEST_SCHEMA_VERSION, CLOUD_TRAINING_LAUNCH_PLAN_SCHEMA_VERSION, CLOUD_TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION, CLOUD_TRAINING_PREFLIGHT_SCHEMA_VERSION, CLOUD_TRAINING_PROVIDER_REGISTRY_SCHEMA_VERSION, CLOUD_TRAINING_STATUS_RECEIPT_SCHEMA_VERSION, PROVIDER_ADAPTER_RECEIPT_TYPES, PROVIDER_ADAPTER_RECEIPT_TYPES_BY_VERSION
from ..cloud_training_completion import CLOUD_TRAINING_COMPLETION_RECEIPT_SCHEMA_VERSION, build_cloud_training_completion_receipt
from ..schema_registry import SchemaRegistryError, check_schema_contract, check_schema_file
from ..source_contract import get_active_opaque_output_attestation, inspect_artifact_source
from ..path_safety import path_has_symlink_component as _path_has_symlink_component, resolve_artifact_reference_path
from ..hashing import sha256_file as _sha256
from .evaluation_serving import _is_replayable_external_eval_ref_path
from .primitives import ValidationTarget, _is_lowercase_sha256, _is_non_negative_int, _is_string_list, _looks_absolute, _read_json_object_silent, _read_object, _require_equal, _sha256, _validate_allowed_keys, _validate_gate_like_checks

def validate_cloud_training_provider_registry(path: str | Path) -> ValidationTarget:
    """Validate a cloud-training provider registry artifact."""
    registry_path = Path(path)
    target = ValidationTarget("cloud_training_provider_registry", str(registry_path))
    registry = _read_object(registry_path, target, "cloud_training_provider_registry.json")
    if registry is not None:
        _validate_cloud_training_contract(registry, target, CLOUD_TRAINING_PROVIDER_REGISTRY_SCHEMA_VERSION)
        providers = registry.get("providers")
        if not isinstance(providers, list):
            target.errors.append("cloud_training_provider_registry.providers must be a list.")
        elif registry.get("provider_count") != len(providers):
            target.errors.append(
                f"cloud_training_provider_registry.provider_count expected {len(providers)}, got {registry.get('provider_count')!r}."
            )
        for index, provider in enumerate(providers if isinstance(providers, list) else []):
            _validate_cloud_training_provider_record(provider, target, f"cloud_training_provider_registry.providers[{index}]")
    return target

def validate_cloud_training_preflight(path: str | Path) -> ValidationTarget:
    """Validate a cloud-training preflight artifact."""
    preflight_path = Path(path)
    target = ValidationTarget("cloud_training_preflight", str(preflight_path))
    preflight = _read_object(preflight_path, target, "cloud_training_preflight.json")
    if preflight is not None:
        _validate_cloud_training_contract(preflight, target, CLOUD_TRAINING_PREFLIGHT_SCHEMA_VERSION, source_path=preflight_path)
        _validate_cloud_training_credentials(preflight.get("credential_checks"), target)
        _validate_cloud_training_live_preflight(preflight.get("live_preflight"), target)
    return target

def validate_cloud_training_artifact_manifest(path: str | Path) -> ValidationTarget:
    """Validate a cloud-training upload/download artifact manifest."""
    manifest_path = Path(path)
    target = ValidationTarget("cloud_training_artifact_manifest", str(manifest_path))
    manifest = _read_object(manifest_path, target, "cloud_training_artifact_manifest.json")
    if manifest is not None:
        _validate_cloud_training_contract(manifest, target, CLOUD_TRAINING_ARTIFACT_MANIFEST_SCHEMA_VERSION, source_path=manifest_path)
    return target

def validate_cloud_training_launch_plan(path: str | Path) -> ValidationTarget:
    """Validate a cloud-training launch plan."""
    plan_path = Path(path)
    target = ValidationTarget("cloud_training_launch_plan", str(plan_path))
    plan = _read_object(plan_path, target, "cloud_training_launch_plan.json")
    if plan is not None:
        _validate_cloud_training_contract(plan, target, CLOUD_TRAINING_LAUNCH_PLAN_SCHEMA_VERSION, source_path=plan_path)
    return target

def validate_cloud_training_launch_receipt(path: str | Path) -> ValidationTarget:
    """Validate a cloud-training launch receipt."""
    receipt_path = Path(path)
    target = ValidationTarget("cloud_training_launch_receipt", str(receipt_path))
    receipt = _read_object(receipt_path, target, "cloud_training_launch_receipt.json")
    if receipt is not None:
        _validate_cloud_training_contract(receipt, target, CLOUD_TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION, source_path=receipt_path)
    return target

def validate_cloud_training_status_receipt(path: str | Path) -> ValidationTarget:
    """Validate a cloud-training status/cancel receipt."""
    receipt_path = Path(path)
    target = ValidationTarget("cloud_training_status_receipt", str(receipt_path))
    receipt = _read_object(receipt_path, target, "cloud_training_status_receipt.json")
    if receipt is not None:
        _validate_cloud_training_contract(receipt, target, CLOUD_TRAINING_STATUS_RECEIPT_SCHEMA_VERSION, source_path=receipt_path)
    return target

def validate_cloud_training_completion_receipt(path: str | Path) -> ValidationTarget:
    """Validate and deterministically replay one imported cloud completion receipt."""
    receipt_path = Path(path)
    target = ValidationTarget("cloud_training_completion_receipt", str(receipt_path))
    receipt = _read_object(
        receipt_path,
        target,
        "cloud_training_completion_receipt.json",
    )
    if receipt is None:
        return target
    try:
        schema_check = check_schema_contract(
            receipt,
            name_or_id="cloud_training_completion_receipt",
            artifact_path=receipt_path,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, SchemaRegistryError) as exc:
        target.errors.append(f"cloud_training_completion_receipt schema: {exc}")
    else:
        target.errors.extend(
            f"cloud_training_completion_receipt schema: {error}"
            for error in schema_check.get("errors", [])
        )
    if receipt.get("schema_version") != CLOUD_TRAINING_COMPLETION_RECEIPT_SCHEMA_VERSION:
        target.errors.append(
            "cloud_training_completion_receipt.schema_version must be "
            f"{CLOUD_TRAINING_COMPLETION_RECEIPT_SCHEMA_VERSION!r}."
        )

    sources = receipt.get("sources")
    source_rows = sources if isinstance(sources, dict) else {}
    if not isinstance(sources, dict):
        target.errors.append("cloud_training_completion_receipt.sources must be an object.")
    resolved = {
        name: _validate_cloud_training_completion_source_ref(
            source_rows.get(name),
            target,
            f"cloud_training_completion_receipt.sources.{name}",
            receipt_path,
        )
        for name in (
            "launch_plan",
            "launch_receipt",
            "status_receipt",
            "runner_metadata",
            "raw_provider_result",
            "output_artifact_manifest",
        )
    }
    if all(path is not None for path in resolved.values()):
        try:
            expected = build_cloud_training_completion_receipt(
                launch_plan_path=resolved["launch_plan"],
                launch_receipt_path=resolved["launch_receipt"],
                status_receipt_path=resolved["status_receipt"],
                runner_metadata_path=resolved["runner_metadata"],
                raw_provider_result_path=resolved["raw_provider_result"],
                output_artifact_manifest_path=resolved[
                    "output_artifact_manifest"
                ],
                out_path=receipt_path,
                created_at=str(receipt.get("created_at") or ""),
            )
        except (OSError, TypeError, ValueError) as exc:
            target.errors.append(
                "cloud_training_completion_receipt could not replay imported "
                f"sources: {exc}"
            )
        else:
            if receipt != expected:
                target.errors.append(
                    "cloud_training_completion_receipt must exactly match "
                    "deterministic replay of its current imported sources."
                )
    execution = receipt.get("execution")
    governance = receipt.get("governance")
    identity = receipt.get("identity")
    target.details.update(
        {
            "integrity_passed": receipt.get("passed") is True,
            "execution_status": execution.get("status")
            if isinstance(execution, dict)
            else None,
            "governance_readiness": governance.get("readiness")
            if isinstance(governance, dict)
            else None,
            "provider_id": identity.get("provider_id")
            if isinstance(identity, dict)
            else None,
            "candidate_model_id": identity.get("candidate_model_id")
            if isinstance(identity, dict)
            else None,
        }
    )
    return target

def _validate_cloud_training_completion_source_ref(
    value: Any,
    target: ValidationTarget,
    label: str,
    receipt_path: Path,
) -> Path | None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return None
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        target.errors.append(f"{label}.path must be a non-empty relative path.")
        return None
    if not _is_replayable_external_eval_ref_path(raw_path):
        target.errors.append(f"{label}.path must be a safe relative path without traversal.")
        return None
    source_path = receipt_path.parent / raw_path
    opaque_attestation = get_active_opaque_output_attestation(source_path)
    if opaque_attestation is None and (
        _path_has_symlink_component(source_path, include_leaf=True)
        or not source_path.is_file()
    ):
        target.errors.append(f"{label}.path must resolve to a regular non-symlink file.")
        return None
    if value.get("exists") is not True or value.get("regular_file") is not True:
        target.errors.append(f"{label} must describe an existing regular file.")
    if opaque_attestation is not None:
        source_size = opaque_attestation.size_bytes
        source_sha256 = opaque_attestation.sha256
    else:
        try:
            source_size = source_path.stat().st_size
            source_sha256 = _sha256(source_path)
        except OSError as exc:
            target.errors.append(f"{label}.path could not be fingerprinted: {exc}")
            return None
    if value.get("size_bytes") != source_size:
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if value.get("sha256") != source_sha256:
        target.errors.append(f"{label}.sha256 does not match the current file.")
    return source_path

_CLOUD_TRAINING_RECEIPT_CHECK_KEYS = {"id", "passed", "actual", "expected", "summary"}

_CLOUD_TRAINING_PROVIDER_REGISTRY_KEYS = {
    "schema_version",
    "created_at",
    "provider_count",
    "providers",
    "execution_boundary",
    "notes",
}

_CLOUD_TRAINING_PREFLIGHT_KEYS = {
    "schema_version",
    "created_at",
    "provider",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "blocked_reasons",
    "constraints",
    "credential_checks",
    "live_preflight",
    "source_artifacts",
    "execution_boundary",
    "handoff_contract",
    "notes",
}

_CLOUD_TRAINING_CONSTRAINT_KEYS = {
    "region",
    "gpu_class",
    "max_cost_usd",
    "requires_region_allowlist",
    "requires_gpu_class_allowlist",
    "requires_cost_estimate",
    "live_spend_allowed",
}

_CLOUD_TRAINING_CREDENTIAL_CHECK_KEYS = {"env_var", "present", "value_recorded"}

_CLOUD_TRAINING_LIVE_PREFLIGHT_KEYS = {
    "requested",
    "transport",
    "provider_api_called",
    "client_modules_imported",
    "credential_values_recorded",
    "credential_env_vars",
    "credential_present_count",
    "credential_required_count",
    "client_dependency_checks",
    "client_dependency_available_count",
    "client_dependency_required_count",
}

_CLOUD_TRAINING_LIVE_DEPENDENCY_CHECK_KEYS = {"module", "available", "module_imported"}

_CLOUD_TRAINING_ARTIFACT_MANIFEST_KEYS = {
    "schema_version",
    "created_at",
    "provider",
    "passed",
    "readiness",
    "check_count",
    "failed_check_count",
    "checks",
    "blocked_reasons",
    "upload_artifacts",
    "expected_download_artifacts",
    "artifact_protocols",
    "transfer_plan",
    "execution_boundary",
}

_CLOUD_TRAINING_TRANSFER_ARTIFACT_KEYS = {"role", "path", "exists", "sha256", "size_bytes"}

_CLOUD_TRAINING_TRANSFER_PLAN_KEYS = {
    "mode",
    "upload_count",
    "expected_download_count",
    "upload_size_bytes",
    "artifact_protocols",
    "requires_external_runner_upload",
    "requires_external_runner_download",
    "download_artifacts_expected_to_exist_before_launch",
    "flight_recorder_uploaded_artifacts",
    "flight_recorder_downloaded_artifacts",
    "provider_api_called",
    "credential_values_recorded",
}

_CLOUD_TRAINING_LAUNCH_PLAN_KEYS = {
    "schema_version",
    "created_at",
    "provider",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "blocked_reasons",
    "source_artifacts",
    "provider_chain",
    "launch",
    "execution_boundary",
    "handoff_contract",
}

_CLOUD_TRAINING_PROVIDER_KEYS = {
    "id",
    "display_name",
    "credential_env_vars",
    "regions",
    "gpu_classes",
    "job_modes",
    "artifact_protocols",
    "client_import_names",
    "live_status",
    "default_live_execution_allowed",
    "adapter_contract",
}

_CLOUD_TRAINING_ADAPTER_CONTRACT_KEYS = {
    "schema_version",
    "adapter_id",
    "provider_id",
    "receipt_types",
    "dry_run_transport",
    "live_preflight_transport",
    "live_preflight_supported",
    "live_launch_supported",
    "status_cancel_receipts_supported",
    "provider_api_called_by_flight_recorder",
    "client_modules_imported_by_flight_recorder",
    "credential_values_recorded",
    "cloud_cost_incurred_usd",
    "model_downloads_started",
    "weights_updated_by_flight_recorder",
    "requires_explicit_live_opt_in",
    "requires_environment_credentials_for_live",
    "requires_cost_limit",
    "requires_region_and_gpu_constraints",
}

_CLOUD_TRAINING_SOURCE_ARTIFACT_KEYS = {
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

_CLOUD_TRAINING_EXECUTION_BOUNDARY_KEYS = {
    "dry_run_only",
    "live_preflight_requested",
    "live_requested",
    "allow_live",
    "provider_api_called",
    "cloud_job_started",
    "cloud_cost_incurred_usd",
    "model_downloads_started",
    "weights_updated_by_flight_recorder",
    "credential_values_recorded",
}

_CLOUD_TRAINING_LAUNCH_KEYS = {
    "mode",
    "cloud_job_started",
    "provider_job_id",
    "provider_api_called",
    "cost_incurred_usd",
}

_CLOUD_TRAINING_LAUNCH_PLAN_LAUNCH_KEYS = {
    "mode",
    "live_launch_supported",
    "provider_api_call_planned",
    "command",
}

_CLOUD_TRAINING_LAUNCH_PLAN_PROVIDER_CHAIN_KEYS = {
    "preflight_provider_id",
    "artifact_manifest_provider_id",
    "artifact_manifest_required",
    "provider_consistent",
    "mismatched_provider_ids",
}

_CLOUD_TRAINING_HANDOFF_CONTRACT_KEYS = {
    "default_live_execution_allowed",
    "requires_explicit_live_opt_in",
    "requires_environment_credentials_for_live",
    "requires_cost_limit",
    "requires_region_and_gpu_constraints",
    "requires_artifact_upload_manifest",
    "requires_status_and_cancel_receipts",
    "flight_recorder_controls_preflight_only",
    "external_provider_owns_execution",
}

_CLOUD_TRAINING_STATUS_KEYS = {
    "provider_status",
    "terminal",
    "cancel_requested",
    "provider_cancel_called",
    "provider_api_called",
    "cost_incurred_usd",
}

def _validate_cloud_training_contract(
    payload: dict[str, Any],
    target: ValidationTarget,
    expected_schema_version: str,
    *,
    source_path: Path | None = None,
) -> None:
    _validate_cloud_training_receipt_allowed_keys(payload, target, expected_schema_version)
    _require_equal(payload, "schema_version", expected_schema_version, target, prefix=f"{target.target_type}.")
    checks = payload.get("checks")
    if checks is not None:
        if not isinstance(checks, list):
            target.errors.append(f"{target.target_type}.checks must be a list.")
            checks = []
        failed_checks = _validate_gate_like_checks(checks, target, f"{target.target_type}.checks")
        if expected_schema_version in {
            CLOUD_TRAINING_PREFLIGHT_SCHEMA_VERSION,
            CLOUD_TRAINING_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            CLOUD_TRAINING_LAUNCH_PLAN_SCHEMA_VERSION,
            CLOUD_TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION,
            CLOUD_TRAINING_STATUS_RECEIPT_SCHEMA_VERSION,
        }:
            _validate_cloud_training_receipt_checks(checks, target, f"{target.target_type}.checks")
        if payload.get("check_count") != len(checks):
            target.errors.append(f"{target.target_type}.check_count expected {len(checks)}, got {payload.get('check_count')!r}.")
        if payload.get("failed_check_count") != failed_checks:
            target.errors.append(
                f"{target.target_type}.failed_check_count expected {failed_checks}, got {payload.get('failed_check_count')!r}."
            )
        if isinstance(payload.get("passed"), bool) and payload.get("passed") != (failed_checks == 0):
            target.errors.append(f"{target.target_type}.passed must match failed_check_count.")
    boundary = payload.get("execution_boundary")
    if not isinstance(boundary, dict):
        target.errors.append(f"{target.target_type}.execution_boundary must be an object.")
    else:
        if expected_schema_version in {
            CLOUD_TRAINING_PROVIDER_REGISTRY_SCHEMA_VERSION,
            CLOUD_TRAINING_PREFLIGHT_SCHEMA_VERSION,
            CLOUD_TRAINING_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            CLOUD_TRAINING_LAUNCH_PLAN_SCHEMA_VERSION,
            CLOUD_TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION,
            CLOUD_TRAINING_STATUS_RECEIPT_SCHEMA_VERSION,
        }:
            _validate_allowed_keys(
                boundary,
                _CLOUD_TRAINING_EXECUTION_BOUNDARY_KEYS,
                target,
                f"{target.target_type}.execution_boundary",
            )
        if boundary.get("dry_run_only") is not True:
            target.errors.append(f"{target.target_type}.execution_boundary.dry_run_only must be true.")
        if boundary.get("provider_api_called") is not False:
            target.errors.append(f"{target.target_type}.execution_boundary.provider_api_called must be false.")
        if boundary.get("cloud_job_started") is not False:
            target.errors.append(f"{target.target_type}.execution_boundary.cloud_job_started must be false.")
        if boundary.get("model_downloads_started") is not False:
            target.errors.append(f"{target.target_type}.execution_boundary.model_downloads_started must be false.")
        if boundary.get("credential_values_recorded") is not False:
            target.errors.append(f"{target.target_type}.execution_boundary.credential_values_recorded must be false.")
        if boundary.get("weights_updated_by_flight_recorder") is not False:
            target.errors.append(f"{target.target_type}.execution_boundary.weights_updated_by_flight_recorder must be false.")
        if boundary.get("cloud_cost_incurred_usd") != 0:
            target.errors.append(f"{target.target_type}.execution_boundary.cloud_cost_incurred_usd must be 0.")
    provider = payload.get("provider")
    if isinstance(provider, dict):
        _validate_cloud_training_provider_record(provider, target, f"{target.target_type}.provider")
    if expected_schema_version in {
        CLOUD_TRAINING_PREFLIGHT_SCHEMA_VERSION,
        CLOUD_TRAINING_LAUNCH_PLAN_SCHEMA_VERSION,
        CLOUD_TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION,
        CLOUD_TRAINING_STATUS_RECEIPT_SCHEMA_VERSION,
    }:
        _validate_cloud_training_source_artifacts(
            payload.get("source_artifacts"),
            target,
            f"{target.target_type}.source_artifacts",
            source_path,
            _cloud_training_required_source_artifacts(expected_schema_version),
        )
    if expected_schema_version == CLOUD_TRAINING_LAUNCH_PLAN_SCHEMA_VERSION:
        expected_provider_chain = _expected_cloud_training_launch_plan_provider_chain(
            payload.get("source_artifacts"),
            source_path,
        )
        _validate_cloud_training_launch_plan_source_refs(
            payload,
            checks if isinstance(checks, list) else [],
            payload.get("source_artifacts"),
            target,
            "cloud_training_launch_plan.source_artifacts",
        )
        _validate_cloud_training_launch_plan_provider_chain(
            payload.get("provider_chain"),
            expected_provider_chain,
            target,
        )
        _validate_cloud_training_launch_plan_provider_matches_chain(payload.get("provider"), expected_provider_chain, target)
        _validate_cloud_training_launch_plan_launch(payload.get("launch"), target)
        _validate_cloud_training_handoff_contract(
            payload.get("handoff_contract"),
            target,
            "cloud_training_launch_plan.handoff_contract",
        )
        _validate_cloud_training_launch_plan_readiness(
            payload,
            checks if isinstance(checks, list) else [],
            expected_provider_chain,
            target,
            source_path,
        )
    if expected_schema_version == CLOUD_TRAINING_PREFLIGHT_SCHEMA_VERSION:
        _validate_cloud_training_constraints(payload.get("constraints"), target)
        _validate_cloud_training_handoff_contract(
            payload.get("handoff_contract"),
            target,
            "cloud_training_preflight.handoff_contract",
        )
        _validate_cloud_training_preflight_source_readiness(
            payload,
            checks if isinstance(checks, list) else [],
            target,
            source_path,
        )
    if expected_schema_version == CLOUD_TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION:
        _validate_cloud_training_launch_receipt_readiness(
            payload,
            checks if isinstance(checks, list) else [],
            target,
            source_path,
        )
    if expected_schema_version == CLOUD_TRAINING_STATUS_RECEIPT_SCHEMA_VERSION:
        _validate_cloud_training_status_receipt_readiness(
            payload,
            checks if isinstance(checks, list) else [],
            target,
            source_path,
        )
    if expected_schema_version == CLOUD_TRAINING_ARTIFACT_MANIFEST_SCHEMA_VERSION:
        upload_artifacts = payload.get("upload_artifacts")
        expected_download_artifacts = payload.get("expected_download_artifacts")
        _validate_cloud_training_artifact_refs(
            upload_artifacts,
            target,
            f"{target.target_type}.upload_artifacts",
            source_path,
            require_non_empty=True,
        )
        _validate_cloud_training_artifact_refs(
            expected_download_artifacts,
            target,
            f"{target.target_type}.expected_download_artifacts",
            source_path,
        )
        _validate_cloud_training_transfer_plan(
            payload.get("transfer_plan"),
            target,
            upload_artifacts,
            expected_download_artifacts,
            payload.get("artifact_protocols"),
        )
    target.details.update(
        {
            "schema_version": payload.get("schema_version"),
            "readiness": payload.get("readiness"),
            "recommendation": payload.get("recommendation"),
            "check_count": payload.get("check_count"),
        }
    )

def _validate_cloud_training_receipt_allowed_keys(
    payload: dict[str, Any],
    target: ValidationTarget,
    expected_schema_version: str,
) -> None:
    common = {
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
        "execution_boundary",
    }
    if expected_schema_version == CLOUD_TRAINING_PROVIDER_REGISTRY_SCHEMA_VERSION:
        _validate_allowed_keys(payload, _CLOUD_TRAINING_PROVIDER_REGISTRY_KEYS, target, "cloud_training_provider_registry")
    if expected_schema_version == CLOUD_TRAINING_PREFLIGHT_SCHEMA_VERSION:
        _validate_allowed_keys(payload, common | {"provider", "constraints", "credential_checks", "live_preflight", "handoff_contract", "notes"}, target, "cloud_training_preflight")
    if expected_schema_version == CLOUD_TRAINING_ARTIFACT_MANIFEST_SCHEMA_VERSION:
        _validate_allowed_keys(payload, _CLOUD_TRAINING_ARTIFACT_MANIFEST_KEYS, target, "cloud_training_artifact_manifest")
    if expected_schema_version == CLOUD_TRAINING_LAUNCH_PLAN_SCHEMA_VERSION:
        _validate_allowed_keys(payload, common | {"provider", "provider_chain", "launch", "handoff_contract"}, target, "cloud_training_launch_plan")
    if expected_schema_version == CLOUD_TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION:
        _validate_allowed_keys(payload, common | {"launch"}, target, "cloud_training_launch_receipt")
    if expected_schema_version == CLOUD_TRAINING_STATUS_RECEIPT_SCHEMA_VERSION:
        _validate_allowed_keys(payload, common | {"status"}, target, "cloud_training_status_receipt")

def _validate_cloud_training_receipt_checks(checks: list[Any], target: ValidationTarget, label: str) -> None:
    for index, check in enumerate(checks):
        if isinstance(check, dict):
            _validate_allowed_keys(check, _CLOUD_TRAINING_RECEIPT_CHECK_KEYS, target, f"{label}[{index}]")

def _validate_cloud_training_provider_record(record: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(record, _CLOUD_TRAINING_PROVIDER_KEYS, target, label)
    provider_id = record.get("id")
    if not isinstance(provider_id, str) or not provider_id:
        target.errors.append(f"{label}.id must be a non-empty string.")
        provider_id = ""
    if record.get("default_live_execution_allowed") is not False:
        target.errors.append(f"{label}.default_live_execution_allowed must be false.")
    if record.get("live_status") != "preflight_only":
        target.errors.append(f"{label}.live_status must be preflight_only.")
    contract = record.get("adapter_contract")
    _validate_cloud_training_adapter_contract(contract, target, f"{label}.adapter_contract", provider_id)

def _validate_cloud_training_adapter_contract(contract: Any, target: ValidationTarget, label: str, provider_id: str) -> None:
    if not isinstance(contract, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(contract, _CLOUD_TRAINING_ADAPTER_CONTRACT_KEYS, target, label)
    contract_version = contract.get("schema_version")
    expected_receipt_types = PROVIDER_ADAPTER_RECEIPT_TYPES_BY_VERSION.get(
        contract_version
    )
    if expected_receipt_types is None:
        target.errors.append(
            f"{label}.schema_version must be one of "
            f"{sorted(PROVIDER_ADAPTER_RECEIPT_TYPES_BY_VERSION)!r}."
        )
        expected_receipt_types = PROVIDER_ADAPTER_RECEIPT_TYPES
    if contract.get("provider_id") != provider_id:
        target.errors.append(f"{label}.provider_id must match provider id.")
    if not isinstance(contract.get("adapter_id"), str) or not contract.get("adapter_id"):
        target.errors.append(f"{label}.adapter_id must be a non-empty string.")
    receipt_types = contract.get("receipt_types")
    if not _is_string_list(receipt_types):
        target.errors.append(f"{label}.receipt_types must be a list of strings.")
        receipt_types = []
    missing_receipts = sorted(set(expected_receipt_types) - set(receipt_types))
    if missing_receipts:
        target.errors.append(f"{label}.receipt_types missing required receipt types: {', '.join(missing_receipts)}.")
    unsupported_receipts = sorted(set(receipt_types) - set(expected_receipt_types))
    if unsupported_receipts:
        target.errors.append(f"{label}.receipt_types contains unsupported receipt types: {', '.join(unsupported_receipts)}.")
    if len(set(receipt_types)) != len(receipt_types):
        target.errors.append(f"{label}.receipt_types must not contain duplicates.")
    if contract.get("dry_run_transport") != "mock_receipts":
        target.errors.append(f"{label}.dry_run_transport must be mock_receipts.")
    if contract.get("live_preflight_transport") != "metadata_only":
        target.errors.append(f"{label}.live_preflight_transport must be metadata_only.")
    for field_name in (
        "live_preflight_supported",
        "status_cancel_receipts_supported",
        "requires_explicit_live_opt_in",
        "requires_environment_credentials_for_live",
        "requires_cost_limit",
        "requires_region_and_gpu_constraints",
    ):
        if contract.get(field_name) is not True:
            target.errors.append(f"{label}.{field_name} must be true.")
    for field_name in (
        "live_launch_supported",
        "provider_api_called_by_flight_recorder",
        "client_modules_imported_by_flight_recorder",
        "credential_values_recorded",
        "model_downloads_started",
        "weights_updated_by_flight_recorder",
    ):
        if contract.get(field_name) is not False:
            target.errors.append(f"{label}.{field_name} must be false.")
    if contract.get("cloud_cost_incurred_usd") != 0:
        target.errors.append(f"{label}.cloud_cost_incurred_usd must be 0.")

def _validate_cloud_training_source_artifacts(
    value: Any,
    target: ValidationTarget,
    label: str,
    source_path: Path | None,
    required_names: tuple[str, ...],
) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, set(required_names), target, label)
    for name in required_names:
        if name not in value:
            target.errors.append(f"{label}.{name} is required.")
    for name, record in value.items():
        if not isinstance(name, str) or not name:
            target.errors.append(f"{label} keys must be non-empty strings.")
            continue
        _validate_cloud_training_artifact_ref(record, target, f"{label}.{name}", source_path)

def _validate_cloud_training_launch_plan_source_refs(
    payload: dict[str, Any],
    checks: list[Any],
    value: Any,
    target: ValidationTarget,
    label: str,
) -> None:
    if not isinstance(value, dict):
        return
    preflight_ref = value.get("preflight")
    if isinstance(preflight_ref, dict) and preflight_ref.get("exists") is not True:
        target.errors.append(f"{label}.preflight.exists must be true.")
    artifact_manifest_ref = value.get("artifact_manifest")
    if not isinstance(artifact_manifest_ref, dict) or artifact_manifest_ref.get("exists") is True:
        return
    if artifact_manifest_ref.get("path") != "":
        target.errors.append(
            "cloud_training_launch_plan.source_artifacts.artifact_manifest.path must be empty when artifact_manifest is missing."
        )
    failed_check_ids = {check.get("id") for check in checks if isinstance(check, dict) and check.get("passed") is False}
    for check_id in ("artifact_manifest_ready", "provider_chain_consistent"):
        if check_id not in failed_check_ids:
            target.errors.append(f"cloud_training_launch_plan.checks must include failed {check_id} when artifact_manifest is missing.")
    provider_chain = payload.get("provider_chain") if isinstance(payload.get("provider_chain"), dict) else {}
    if provider_chain.get("artifact_manifest_required") is not True:
        target.errors.append("cloud_training_launch_plan.provider_chain.artifact_manifest_required must be true.")
    if provider_chain.get("provider_consistent") is not False:
        target.errors.append(
            "cloud_training_launch_plan.provider_chain.provider_consistent must be false when artifact_manifest is missing."
        )
    if payload.get("readiness") != "blocked":
        target.errors.append("cloud_training_launch_plan.readiness must be blocked when artifact_manifest is missing.")
    if payload.get("recommendation") != "block_launch_receipt":
        target.errors.append("cloud_training_launch_plan.recommendation must be block_launch_receipt when artifact_manifest is missing.")

def _validate_cloud_training_artifact_refs(
    value: Any,
    target: ValidationTarget,
    label: str,
    source_path: Path | None,
    *,
    require_non_empty: bool = False,
) -> None:
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list.")
        return
    if require_non_empty and not value:
        target.errors.append(f"{label} must include at least one artifact.")
    for index, record in enumerate(value):
        _validate_cloud_training_artifact_ref(
            record,
            target,
            f"{label}[{index}]",
            source_path,
            allowed_keys=_CLOUD_TRAINING_TRANSFER_ARTIFACT_KEYS,
        )

def _validate_cloud_training_transfer_plan(
    value: Any,
    target: ValidationTarget,
    upload_artifacts: Any,
    expected_download_artifacts: Any,
    artifact_protocols: Any,
) -> None:
    label = "cloud_training_artifact_manifest.transfer_plan"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _CLOUD_TRAINING_TRANSFER_PLAN_KEYS, target, label)
    uploads = upload_artifacts if isinstance(upload_artifacts, list) else []
    downloads = expected_download_artifacts if isinstance(expected_download_artifacts, list) else []
    protocols = artifact_protocols if _is_string_list(artifact_protocols) else []
    upload_size_bytes = 0
    for artifact in uploads:
        if isinstance(artifact, dict) and artifact.get("exists") is True and _is_non_negative_int(artifact.get("size_bytes")):
            upload_size_bytes += artifact["size_bytes"]
    expected_values = {
        "mode": "dry_run_manifest_only",
        "upload_count": len(uploads),
        "expected_download_count": len(downloads),
        "upload_size_bytes": upload_size_bytes,
        "artifact_protocols": protocols,
        "requires_external_runner_upload": bool(uploads),
        "requires_external_runner_download": bool(downloads),
        "download_artifacts_expected_to_exist_before_launch": False,
    }
    for field_name, expected in expected_values.items():
        if value.get(field_name) != expected:
            target.errors.append(f"{label}.{field_name} must match cloud training artifact rows.")
    for field_name in (
        "flight_recorder_uploaded_artifacts",
        "flight_recorder_downloaded_artifacts",
        "provider_api_called",
        "credential_values_recorded",
    ):
        if value.get(field_name) is not False:
            target.errors.append(f"{label}.{field_name} must be false.")

def _validate_cloud_training_launch_plan_launch(value: Any, target: ValidationTarget) -> None:
    label = "cloud_training_launch_plan.launch"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _CLOUD_TRAINING_LAUNCH_PLAN_LAUNCH_KEYS, target, label)
    if value.get("mode") != "dry_run":
        target.errors.append(f"{label}.mode must be dry_run.")
    if value.get("live_launch_supported") is not False:
        target.errors.append(f"{label}.live_launch_supported must be false.")
    if value.get("provider_api_call_planned") is not False:
        target.errors.append(f"{label}.provider_api_call_planned must be false.")
    command = value.get("command")
    if not _is_string_list(command):
        target.errors.append(f"{label}.command must be a list of strings.")
    else:
        for index, item in enumerate(command):
            _validate_cloud_training_command_token_public_path(target, f"{label}.command[{index}]", item)

def _validate_cloud_training_command_token_public_path(target: ValidationTarget, label: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        return
    if _looks_absolute(value):
        target.errors.append(f"{label} must use a relative command token or redacted placeholder.")
        return
    _, separator, token_value = value.partition("=")
    if separator and _looks_absolute(token_value):
        target.errors.append(f"{label} must use a relative command token or redacted placeholder.")

def _validate_cloud_training_handoff_contract(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _CLOUD_TRAINING_HANDOFF_CONTRACT_KEYS, target, label)
    for field_name in (
        "requires_explicit_live_opt_in",
        "requires_environment_credentials_for_live",
        "requires_cost_limit",
        "requires_region_and_gpu_constraints",
        "requires_artifact_upload_manifest",
        "requires_status_and_cancel_receipts",
        "flight_recorder_controls_preflight_only",
        "external_provider_owns_execution",
    ):
        if value.get(field_name) is not True:
            target.errors.append(f"{label}.{field_name} must be true.")
    if value.get("default_live_execution_allowed") is not False:
        target.errors.append(f"{label}.default_live_execution_allowed must be false.")

def _validate_cloud_training_constraints(value: Any, target: ValidationTarget) -> None:
    label = "cloud_training_preflight.constraints"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _CLOUD_TRAINING_CONSTRAINT_KEYS, target, label)
    for field_name in ("region", "gpu_class"):
        if not isinstance(value.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    max_cost = value.get("max_cost_usd")
    if max_cost is not None and (
        not isinstance(max_cost, (int, float)) or isinstance(max_cost, bool) or max_cost < 0
    ):
        target.errors.append(f"{label}.max_cost_usd must be a non-negative number or null.")
    for field_name in ("requires_region_allowlist", "requires_gpu_class_allowlist", "requires_cost_estimate"):
        if value.get(field_name) is not True:
            target.errors.append(f"{label}.{field_name} must be true.")
    if value.get("live_spend_allowed") is not False:
        target.errors.append(f"{label}.live_spend_allowed must be false.")

def _validate_cloud_training_launch_plan_provider_chain(
    value: Any,
    expected: dict[str, Any],
    target: ValidationTarget,
) -> None:
    label = "cloud_training_launch_plan.provider_chain"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _CLOUD_TRAINING_LAUNCH_PLAN_PROVIDER_CHAIN_KEYS, target, label)
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} must match launch-plan source providers.")

def _validate_cloud_training_launch_plan_provider_matches_chain(
    value: Any,
    expected_chain: dict[str, Any],
    target: ValidationTarget,
) -> None:
    expected_provider_id = expected_chain.get("preflight_provider_id")
    if not expected_provider_id:
        return
    provider_id = value.get("id") if isinstance(value, dict) else None
    if provider_id != expected_provider_id:
        target.errors.append("cloud_training_launch_plan.provider.id must match provider_chain.preflight_provider_id.")

def _validate_cloud_training_preflight_source_readiness(
    payload: dict[str, Any],
    checks: list[Any],
    target: ValidationTarget,
    source_path: Path | None,
) -> None:
    sources = payload.get("source_artifacts") if isinstance(payload.get("source_artifacts"), dict) else {}
    role_checks = (
        ("agentic_training_plan", "agentic_training_plan_ready"),
        ("trainer_preflight", "trainer_preflight_ready"),
        ("trainer_launch_check", "trainer_launch_check_ready"),
    )
    for role, check_id in role_checks:
        state = _cloud_training_source_state(sources.get(role), source_path, role)
        _validate_cloud_training_source_state(
            f"cloud_training_preflight.source_artifacts.{role}",
            state,
            target,
        )
        check = _cloud_training_check_by_id(
            checks,
            check_id,
            target,
            "cloud_training_preflight.checks",
        )
        if check is not None and check.get("passed") != state["ready"]:
            target.errors.append(
                f"cloud_training_preflight.checks.{check_id}.passed must match semantic source readiness."
            )

def _validate_cloud_training_launch_plan_readiness(
    payload: dict[str, Any],
    checks: list[Any],
    expected_chain: dict[str, Any],
    target: ValidationTarget,
    source_path: Path | None,
) -> None:
    sources = payload.get("source_artifacts") if isinstance(payload.get("source_artifacts"), dict) else {}
    preflight_state = _cloud_training_source_state(
        sources.get("preflight"),
        source_path,
        "cloud_training_preflight",
    )
    artifact_manifest_state = _cloud_training_source_state(
        sources.get("artifact_manifest"),
        source_path,
        "cloud_training_artifact_manifest",
    )
    for role, state in (("preflight", preflight_state), ("artifact_manifest", artifact_manifest_state)):
        _validate_cloud_training_source_state(f"cloud_training_launch_plan.source_artifacts.{role}", state, target)
    expected_checks = {
        "preflight_ready": preflight_state["ready"],
        "artifact_manifest_ready": artifact_manifest_state["ready"],
        "provider_chain_consistent": expected_chain["provider_consistent"],
        "dry_run_launch_only": True,
    }
    for check_id, expected_passed in expected_checks.items():
        check = _cloud_training_check_by_id(checks, check_id, target, "cloud_training_launch_plan.checks")
        if check is None:
            continue
        if check.get("passed") != expected_passed:
            target.errors.append(f"cloud_training_launch_plan.checks.{check_id}.passed must match source readiness.")
    _validate_cloud_training_expected_state(
        "cloud_training_launch_plan",
        payload,
        expected_checks,
        ready_readiness="ready_for_dry_run_launch",
        ready_recommendation="emit_dry_run_launch_receipt",
        blocked_recommendation="block_launch_receipt",
        target=target,
    )

def _validate_cloud_training_launch_receipt_readiness(
    payload: dict[str, Any],
    checks: list[Any],
    target: ValidationTarget,
    source_path: Path | None,
) -> None:
    sources = payload.get("source_artifacts") if isinstance(payload.get("source_artifacts"), dict) else {}
    launch_plan_state = _cloud_training_source_state(
        sources.get("launch_plan"),
        source_path,
        "cloud_training_launch_plan",
    )
    _validate_cloud_training_source_state(
        "cloud_training_launch_receipt.source_artifacts.launch_plan",
        launch_plan_state,
        target,
    )
    launch = payload.get("launch") if isinstance(payload.get("launch"), dict) else {}
    if not isinstance(payload.get("launch"), dict):
        target.errors.append("cloud_training_launch_receipt.launch must be an object.")
    else:
        _validate_allowed_keys(launch, _CLOUD_TRAINING_LAUNCH_KEYS, target, "cloud_training_launch_receipt.launch")
    live_requested = launch.get("mode") == "live"
    if launch.get("mode") not in {"dry_run", "live"}:
        target.errors.append("cloud_training_launch_receipt.launch.mode must be dry_run or live.")
    if launch.get("cloud_job_started") is not False:
        target.errors.append("cloud_training_launch_receipt.launch.cloud_job_started must be false.")
    if launch.get("provider_api_called") is not False:
        target.errors.append("cloud_training_launch_receipt.launch.provider_api_called must be false.")
    if launch.get("provider_job_id") is not None:
        target.errors.append("cloud_training_launch_receipt.launch.provider_job_id must be null.")
    if launch.get("cost_incurred_usd") != 0:
        target.errors.append("cloud_training_launch_receipt.launch.cost_incurred_usd must be 0.")
    boundary = payload.get("execution_boundary") if isinstance(payload.get("execution_boundary"), dict) else {}
    if boundary.get("live_requested") != live_requested:
        target.errors.append("cloud_training_launch_receipt.execution_boundary.live_requested must match launch.mode.")
    if boundary.get("allow_live") is not False:
        target.errors.append("cloud_training_launch_receipt.execution_boundary.allow_live must be false.")
    cloud_job_not_started = (
        launch.get("cloud_job_started") is False
        and launch.get("provider_api_called") is False
        and launch.get("provider_job_id") is None
        and launch.get("cost_incurred_usd") == 0
    )
    expected_checks = {
        "launch_plan_ready": launch_plan_state["ready"],
        "live_launch_not_implemented": not live_requested,
        "cloud_job_not_started": cloud_job_not_started,
    }
    for check_id, expected_passed in expected_checks.items():
        check = _cloud_training_check_by_id(checks, check_id, target, "cloud_training_launch_receipt.checks")
        if check is not None and check.get("passed") != expected_passed:
            target.errors.append(f"cloud_training_launch_receipt.checks.{check_id}.passed must match source readiness.")
    _validate_cloud_training_expected_state(
        "cloud_training_launch_receipt",
        payload,
        expected_checks,
        ready_readiness="dry_run_recorded",
        ready_recommendation="safe_to_archive_dry_run_receipt",
        blocked_recommendation="block_live_cloud_launch",
        target=target,
    )

def _validate_cloud_training_status_receipt_readiness(
    payload: dict[str, Any],
    checks: list[Any],
    target: ValidationTarget,
    source_path: Path | None,
) -> None:
    sources = payload.get("source_artifacts") if isinstance(payload.get("source_artifacts"), dict) else {}
    launch_receipt_state = _cloud_training_source_state(
        sources.get("launch_receipt"),
        source_path,
        "cloud_training_launch_receipt",
    )
    _validate_cloud_training_source_state(
        "cloud_training_status_receipt.source_artifacts.launch_receipt",
        launch_receipt_state,
        target,
    )
    status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
    if not isinstance(payload.get("status"), dict):
        target.errors.append("cloud_training_status_receipt.status must be an object.")
    else:
        _validate_allowed_keys(status, _CLOUD_TRAINING_STATUS_KEYS, target, "cloud_training_status_receipt.status")
    if status.get("provider_status") != "not_started":
        target.errors.append("cloud_training_status_receipt.status.provider_status must be not_started.")
    if status.get("terminal") is not True:
        target.errors.append("cloud_training_status_receipt.status.terminal must be true.")
    if not isinstance(status.get("cancel_requested"), bool):
        target.errors.append("cloud_training_status_receipt.status.cancel_requested must be a boolean.")
    if status.get("provider_cancel_called") is not False:
        target.errors.append("cloud_training_status_receipt.status.provider_cancel_called must be false.")
    if status.get("provider_api_called") is not False:
        target.errors.append("cloud_training_status_receipt.status.provider_api_called must be false.")
    if status.get("cost_incurred_usd") != 0:
        target.errors.append("cloud_training_status_receipt.status.cost_incurred_usd must be 0.")
    expected_checks = {
        "launch_receipt_readable": launch_receipt_state["ready"],
        "status_check_did_not_call_provider": status.get("provider_api_called") is False,
        "cancel_is_dry_run": status.get("provider_cancel_called") is False,
    }
    for check_id, expected_passed in expected_checks.items():
        check = _cloud_training_check_by_id(checks, check_id, target, "cloud_training_status_receipt.checks")
        if check is not None and check.get("passed") != expected_passed:
            target.errors.append(f"cloud_training_status_receipt.checks.{check_id}.passed must match source readiness.")
    _validate_cloud_training_expected_state(
        "cloud_training_status_receipt",
        payload,
        expected_checks,
        ready_readiness="status_recorded",
        ready_recommendation="archive_status_receipt",
        blocked_recommendation="inspect_launch_receipt",
        target=target,
    )

def _validate_cloud_training_expected_state(
    label: str,
    payload: dict[str, Any],
    expected_checks: dict[str, bool],
    *,
    ready_readiness: str,
    ready_recommendation: str,
    blocked_recommendation: str,
    target: ValidationTarget,
) -> None:
    expected_failed_count = sum(1 for passed in expected_checks.values() if not passed)
    expected_passed = expected_failed_count == 0
    if payload.get("failed_check_count") != expected_failed_count:
        target.errors.append(f"{label}.failed_check_count expected {expected_failed_count}.")
    if payload.get("passed") != expected_passed:
        target.errors.append(f"{label}.passed must match source readiness.")
    expected_readiness = ready_readiness if expected_passed else "blocked"
    expected_recommendation = ready_recommendation if expected_passed else blocked_recommendation
    if payload.get("readiness") != expected_readiness:
        target.errors.append(f"{label}.readiness must be {expected_readiness}.")
    if payload.get("recommendation") != expected_recommendation:
        target.errors.append(f"{label}.recommendation must be {expected_recommendation}.")

def _validate_cloud_training_source_state(label: str, state: dict[str, Any], target: ValidationTarget) -> None:
    ref = state["ref"]
    if not isinstance(ref, dict):
        return
    for field_name in ("schema_name", "schema_passed", "source_passed", "source_recommendation"):
        expected_value = state[field_name]
        if ref.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} must match the referenced source artifact.")

def _cloud_training_source_state(record: Any, source_path: Path | None, schema_name: str) -> dict[str, Any]:
    state = {
        "ref": record,
        "ref_exists": False,
        "schema_name": schema_name,
        "schema_passed": False,
        "source_passed": None,
        "source_recommendation": "",
        "ready": False,
    }
    if not isinstance(record, dict) or record.get("exists") is not True:
        return state
    artifact_path = _resolve_regular_cloud_training_artifact_path(record.get("path"), source_path)
    if artifact_path is None:
        return state
    inspection = inspect_artifact_source(artifact_path, schema_name)
    payload = inspection.get("payload") if isinstance(inspection.get("payload"), dict) else {}
    schema_passed = inspection.get("schema_valid") is True
    source_passed = payload.get("passed") if isinstance(payload.get("passed"), bool) else None
    state.update(
        {
            "ref_exists": True,
            "schema_passed": schema_passed,
            "source_passed": source_passed,
            "source_recommendation": str(payload.get("recommendation") or ""),
            "ready": inspection.get("ready") is True,
        }
    )
    return state

def _cloud_training_schema_check_passed(path: Path | None, schema_name: str) -> bool:
    if path is None:
        return False
    try:
        return check_schema_file(path, schema_name).get("passed") is True
    except (OSError, json.JSONDecodeError, SchemaRegistryError):
        return False

def _cloud_training_check_by_id(
    checks: list[Any],
    check_id: str,
    target: ValidationTarget,
    label: str,
) -> dict[str, Any] | None:
    matches = [check for check in checks if isinstance(check, dict) and check.get("id") == check_id]
    if len(matches) != 1:
        target.errors.append(f"{label} must include exactly one {check_id} check.")
        return None
    return matches[0]

def _expected_cloud_training_launch_plan_provider_chain(source_artifacts: Any, source_path: Path | None) -> dict[str, Any]:
    sources = source_artifacts if isinstance(source_artifacts, dict) else {}
    preflight_provider_id = _cloud_training_provider_id_from_source_ref(sources.get("preflight"), source_path)
    artifact_manifest_ref = sources.get("artifact_manifest") if isinstance(sources.get("artifact_manifest"), dict) else {}
    artifact_manifest_required = True
    artifact_manifest_provider_id = _cloud_training_provider_id_from_source_ref(artifact_manifest_ref, source_path)
    provider_consistent = (
        bool(preflight_provider_id)
        and (
            not artifact_manifest_required
            or (bool(artifact_manifest_provider_id) and preflight_provider_id == artifact_manifest_provider_id)
        )
    )
    return {
        "preflight_provider_id": preflight_provider_id,
        "artifact_manifest_provider_id": artifact_manifest_provider_id,
        "artifact_manifest_required": artifact_manifest_required,
        "provider_consistent": provider_consistent,
        "mismatched_provider_ids": (
            []
            if provider_consistent
            else [provider_id for provider_id in (preflight_provider_id, artifact_manifest_provider_id) if provider_id]
        ),
    }

def _cloud_training_provider_id_from_source_ref(record: Any, source_path: Path | None) -> str:
    if not isinstance(record, dict) or record.get("exists") is not True:
        return ""
    artifact_path = _resolve_regular_cloud_training_artifact_path(record.get("path"), source_path)
    payload = _read_json_object_silent(artifact_path)
    provider = payload.get("provider") if isinstance(payload.get("provider"), dict) else {}
    return str(provider.get("id") or "") if isinstance(provider, dict) else ""

def _cloud_training_required_source_artifacts(schema_version: str) -> tuple[str, ...]:
    return {
        CLOUD_TRAINING_PREFLIGHT_SCHEMA_VERSION: ("agentic_training_plan", "trainer_preflight", "trainer_launch_check"),
        CLOUD_TRAINING_LAUNCH_PLAN_SCHEMA_VERSION: ("preflight", "artifact_manifest"),
        CLOUD_TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION: ("launch_plan",),
        CLOUD_TRAINING_STATUS_RECEIPT_SCHEMA_VERSION: ("launch_receipt",),
    }.get(schema_version, ())

def _validate_cloud_training_artifact_ref(
    record: Any,
    target: ValidationTarget,
    label: str,
    source_path: Path | None,
    *,
    allowed_keys: set[str] | None = None,
) -> None:
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(record, allowed_keys or _CLOUD_TRAINING_SOURCE_ARTIFACT_KEYS, target, label)
    if not isinstance(record.get("role"), str) or not record.get("role"):
        target.errors.append(f"{label}.role must be a non-empty string.")
    if not isinstance(record.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
        return
    path_value = record.get("path")
    if isinstance(path_value, str) and path_value and not _is_safe_or_redacted_cloud_training_ref_path(path_value):
        target.errors.append(f"{label}.path must be relative to the cloud-training artifact.")
        return
    if record.get("exists") is not True:
        if record.get("sha256") is not None:
            target.errors.append(f"{label}.sha256 must be null when exists is false.")
        if record.get("size_bytes") is not None:
            target.errors.append(f"{label}.size_bytes must be null when exists is false.")
        return
    if not isinstance(path_value, str) or not path_value:
        target.errors.append(f"{label}.path must be a non-empty string when exists is true.")
        return
    if not _is_replayable_cloud_training_ref_path(path_value):
        target.errors.append(f"{label}.path must be relative to the cloud-training artifact.")
        return
    artifact_path = _resolve_cloud_training_artifact_path(path_value, source_path)
    if artifact_path is not None and _path_has_symlink_component(artifact_path, include_leaf=True):
        target.errors.append(f"{label}.path must resolve to a regular non-symlink file when exists is true.")
        return
    if artifact_path is None or not artifact_path.exists() or not artifact_path.is_file():
        target.errors.append(f"{label}.path must resolve to an existing file when exists is true.")
        return
    if not _is_lowercase_sha256(record.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a lowercase SHA-256 hex string when exists is true.")
    if not _is_non_negative_int(record.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer when exists is true.")
    if _is_non_negative_int(record.get("size_bytes")) and artifact_path.stat().st_size != record.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if _is_lowercase_sha256(record.get("sha256")) and _sha256(artifact_path) != record.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current file.")

def _resolve_cloud_training_artifact_path(value: Any, source_path: Path | None) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if _looks_absolute(value):
        return None
    if source_path is None:
        return None
    return source_path.parent / path

def _resolve_regular_cloud_training_artifact_path(value: Any, source_path: Path | None) -> Path | None:
    artifact_path = _resolve_cloud_training_artifact_path(value, source_path)
    if (
        artifact_path is None
        or not artifact_path.exists()
        or _path_has_symlink_component(artifact_path, include_leaf=True)
        or not artifact_path.is_file()
    ):
        return None
    return artifact_path

def _is_safe_or_redacted_cloud_training_ref_path(value: str) -> bool:
    if value.startswith("<redacted:") and value.endswith(">"):
        basename = value.removeprefix("<redacted:").removesuffix(">")
        return bool(basename) and "/" not in basename and "\\" not in basename and ".." not in basename and "~" not in basename
    return _is_replayable_cloud_training_ref_path(value)

def _is_replayable_cloud_training_ref_path(value: str) -> bool:
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

def _validate_cloud_training_credentials(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, list):
        target.errors.append("cloud_training_preflight.credential_checks must be a list.")
        return
    for index, row in enumerate(value):
        label = f"cloud_training_preflight.credential_checks[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(row, _CLOUD_TRAINING_CREDENTIAL_CHECK_KEYS, target, label)
        if not isinstance(row.get("env_var"), str) or not row.get("env_var"):
            target.errors.append(f"{label}.env_var must be a non-empty string.")
        if not isinstance(row.get("present"), bool):
            target.errors.append(f"{label}.present must be a boolean.")
        if row.get("value_recorded") is not False:
            target.errors.append(f"{label}.value_recorded must be false.")

def _validate_cloud_training_live_preflight(value: Any, target: ValidationTarget) -> None:
    if value is None:
        return
    label = "cloud_training_preflight.live_preflight"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object when present.")
        return
    _validate_allowed_keys(value, _CLOUD_TRAINING_LIVE_PREFLIGHT_KEYS, target, label)
    if not isinstance(value.get("requested"), bool):
        target.errors.append(f"{label}.requested must be a boolean.")
    if value.get("transport") != "metadata_only":
        target.errors.append(f"{label}.transport must be metadata_only.")
    for field_name in ("provider_api_called", "client_modules_imported", "credential_values_recorded"):
        if value.get(field_name) is not False:
            target.errors.append(f"{label}.{field_name} must be false.")
    dependency_checks = value.get("client_dependency_checks")
    if not isinstance(dependency_checks, list):
        target.errors.append(f"{label}.client_dependency_checks must be a list.")
        dependency_checks = []
    credential_env_vars = value.get("credential_env_vars")
    if not _is_string_list(credential_env_vars):
        target.errors.append(f"{label}.credential_env_vars must be a list of strings.")
        credential_env_vars = []
    available_count = 0
    for index, row in enumerate(dependency_checks):
        row_label = f"{label}.client_dependency_checks[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{row_label} must be an object.")
            continue
        _validate_allowed_keys(row, _CLOUD_TRAINING_LIVE_DEPENDENCY_CHECK_KEYS, target, row_label)
        if not isinstance(row.get("module"), str) or not row.get("module"):
            target.errors.append(f"{row_label}.module must be a non-empty string.")
        if not isinstance(row.get("available"), bool):
            target.errors.append(f"{row_label}.available must be a boolean.")
        elif row["available"]:
            available_count += 1
        if row.get("module_imported") is not False:
            target.errors.append(f"{row_label}.module_imported must be false.")
    if value.get("client_dependency_required_count") != len(dependency_checks):
        target.errors.append(
            f"{label}.client_dependency_required_count expected {len(dependency_checks)}, got {value.get('client_dependency_required_count')!r}."
        )
    if value.get("client_dependency_available_count") != available_count:
        target.errors.append(
            f"{label}.client_dependency_available_count expected {available_count}, got {value.get('client_dependency_available_count')!r}."
        )
    if not _is_non_negative_int(value.get("credential_required_count")):
        target.errors.append(f"{label}.credential_required_count must be a non-negative integer.")
    if not _is_non_negative_int(value.get("credential_present_count")):
        target.errors.append(f"{label}.credential_present_count must be a non-negative integer.")
    if (
        _is_non_negative_int(value.get("credential_required_count"))
        and _is_non_negative_int(value.get("credential_present_count"))
        and value.get("credential_present_count") > value.get("credential_required_count")
    ):
        target.errors.append(f"{label}.credential_present_count must not exceed credential_required_count.")
