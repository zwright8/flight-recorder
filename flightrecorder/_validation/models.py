"""Extracted validation implementation."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any
from ..model_registry import MODEL_ADAPTER_MANIFEST_SCHEMA_VERSION, MODEL_CANDIDATE_SCHEMA_VERSION, MODEL_COMPATIBILITY_REPORT_SCHEMA_VERSION, MODEL_REGISTRY_ENTRY_SCHEMA_VERSION, MODEL_REGISTRY_SCHEMA_VERSION, MODEL_SCOUT_MANIFEST_SCHEMA_VERSION, MODEL_SERVING_PROBE_RECEIPT_SCHEMA_VERSION, TRAINING_PLAN_SCHEMA_VERSION, is_training_license_approved, model_adapter_manifest_errors, model_candidate_errors, model_compatibility_report_errors, model_registry_entry_errors, model_registry_errors, model_scout_manifest_errors, model_serving_probe_receipt_errors, training_plan_errors
from ..path_safety import path_has_symlink_component as _path_has_symlink_component, resolve_artifact_reference_path
from ..hashing import sha256_file as _sha256
from .primitives import ValidationTarget, _is_non_negative_int, _is_sha256, _read_object, _require_equal, _sha256
from .training_flow_result import _model_scout_reference_path

def validate_model_scout_manifest(path: str | Path) -> ValidationTarget:
    """Validate a model-scout manifest and any referenced candidate artifacts."""
    manifest_path = Path(path)
    target = ValidationTarget("model_scout_manifest", str(manifest_path))
    manifest = _read_object(manifest_path, target, "model_scout_manifest.json")
    if manifest is not None:
        _validate_model_scout_manifest(manifest, target, manifest_path)
    return target

def validate_model_candidate(path: str | Path) -> ValidationTarget:
    """Validate a model-candidate artifact."""
    candidate_path = Path(path)
    target = ValidationTarget("model_candidate", str(candidate_path))
    candidate = _read_object(candidate_path, target, "model_candidate.json")
    if candidate is not None:
        _validate_model_candidate(candidate, target)
    return target

def validate_model_compatibility_report(path: str | Path) -> ValidationTarget:
    """Validate a model-compatibility report artifact."""
    report_path = Path(path)
    target = ValidationTarget("model_compatibility_report", str(report_path))
    report = _read_object(report_path, target, "model_compatibility_report.json")
    if report is not None:
        _validate_model_compatibility_report(report, target)
    return target

def validate_model_serving_probe_receipt(path: str | Path) -> ValidationTarget:
    """Validate a no-download model serving-probe receipt artifact."""
    receipt_path = Path(path)
    target = ValidationTarget("model_serving_probe_receipt", str(receipt_path))
    receipt = _read_object(receipt_path, target, "model_serving_probe_receipt.json")
    if receipt is not None:
        _validate_model_serving_probe_receipt(receipt, target, receipt_path)
    return target

def validate_model_adapter_manifest(path: str | Path) -> ValidationTarget:
    """Validate a no-download planned adapter manifest artifact."""
    manifest_path = Path(path)
    target = ValidationTarget("model_adapter_manifest", str(manifest_path))
    manifest = _read_object(manifest_path, target, "model_adapter_manifest.json")
    if manifest is not None:
        _validate_model_adapter_manifest(manifest, target, manifest_path)
    return target

def validate_model_registry_entry(path: str | Path) -> ValidationTarget:
    """Validate a single model-registry entry artifact."""
    entry_path = Path(path)
    target = ValidationTarget("model_registry_entry", str(entry_path))
    entry = _read_object(entry_path, target, "model_registry_entry.json")
    if entry is not None:
        _validate_model_registry_entry(entry, target, entry_path)
    return target

def validate_model_registry(path: str | Path) -> ValidationTarget:
    """Validate a model registry artifact."""
    registry_path = Path(path)
    target = ValidationTarget("model_registry", str(registry_path))
    registry = _read_object(registry_path, target, "model_registry.json")
    if registry is not None:
        _validate_model_registry(registry, target, registry_path)
    return target

def validate_training_plan(path: str | Path) -> ValidationTarget:
    """Validate a dry-run training-plan artifact."""
    plan_path = Path(path)
    target = ValidationTarget("training_plan", str(plan_path))
    plan = _read_object(plan_path, target, "training_plan.json")
    if plan is not None:
        _validate_training_plan(plan, target, plan_path)
    return target

def _validate_model_scout_manifest(manifest: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(manifest, "schema_version", MODEL_SCOUT_MANIFEST_SCHEMA_VERSION, target, prefix="model_scout_manifest.")
    target.errors.extend(model_scout_manifest_errors(manifest))
    selection_policy = manifest.get("selection_policy") if isinstance(manifest.get("selection_policy"), dict) else {}
    require_compatibility_metadata = selection_policy.get("require_compatibility_metadata") is True
    candidates = manifest.get("candidates") if isinstance(manifest.get("candidates"), list) else []
    candidate_count = 0
    training_eligible_count = 0
    compatibility_report_count = 0
    blocked_training_selection_count = 0

    for index, ref in enumerate(candidates):
        if not isinstance(ref, dict):
            continue
        candidate_count += 1
        label = f"model_scout_manifest.candidates[{index}]"
        candidate_path = _model_scout_reference_path(ref.get("manifest_path"), source_path)
        candidate = _read_object(candidate_path, target, f"{label}.manifest_path") if candidate_path is not None else None
        if candidate is None:
            continue

        target.errors.extend(error.replace("model_candidate.", f"{label}.candidate.") for error in model_candidate_errors(candidate))
        if ref.get("candidate_id") != candidate.get("candidate_id"):
            target.errors.append(f"{label}.candidate_id must match referenced candidate.candidate_id.")
        license_review = candidate.get("license") if isinstance(candidate.get("license"), dict) else {}
        if isinstance(ref.get("license_status"), str) and ref.get("license_status") != license_review.get("status"):
            target.errors.append(f"{label}.license_status must match referenced candidate license.status.")
        if isinstance(ref.get("review_status"), str) and ref.get("review_status") != license_review.get("review_status"):
            target.errors.append(f"{label}.review_status must match referenced candidate license.review_status.")

        training_eligible = is_training_license_approved(candidate)
        if training_eligible:
            training_eligible_count += 1
        elif ref.get("training_selection_eligible") is True:
            blocked_training_selection_count += 1
            target.errors.append(
                f"{label}.training_selection_eligible cannot be true unless the referenced candidate license is approved for training."
            )
        if ref.get("training_selection_eligible") is False and training_eligible:
            target.warnings.append(
                f"{label}.training_selection_eligible is false although the referenced candidate is license-approved."
            )
        if require_compatibility_metadata and not isinstance(candidate.get("compatibility"), dict):
            target.errors.append(
                f"{label}.candidate.compatibility must be present when selection policy requires compatibility metadata."
            )

        report_path_value = ref.get("compatibility_report_path")
        if isinstance(report_path_value, str) and report_path_value:
            compatibility_report_count += 1
            report_path = _model_scout_reference_path(report_path_value, source_path)
            report = _read_object(report_path, target, f"{label}.compatibility_report_path") if report_path is not None else None
            if report is None:
                continue
            target.errors.extend(
                error.replace("model_compatibility_report.", f"{label}.compatibility_report.")
                for error in model_compatibility_report_errors(report)
            )
            if report.get("candidate_id") != candidate.get("candidate_id"):
                target.errors.append(f"{label}.compatibility_report.candidate_id must match referenced candidate.")
            if report.get("model_id") != candidate.get("model_id"):
                target.errors.append(f"{label}.compatibility_report.model_id must match referenced candidate.")
            if ref.get("training_selection_eligible") is True and report.get("passed") is not True:
                target.errors.append(
                    f"{label}.compatibility_report.passed must be true for training-selectable scout entries."
                )

    target.details.update(
        {
            "candidate_count": candidate_count,
            "compatibility_report_count": compatibility_report_count,
            "training_eligible_count": training_eligible_count,
            "blocked_training_selection_count": blocked_training_selection_count,
        }
    )

def _validate_existing_path_field(
    value: dict[str, Any],
    field_name: str,
    target: ValidationTarget,
    label: str,
    base_dir: Path | None = None,
) -> None:
    path_value = value.get(field_name)
    if not isinstance(path_value, str) or not path_value:
        target.errors.append(f"{label} must be a non-empty string.")
        return
    check_path = Path(path_value)
    if base_dir is not None and not check_path.is_absolute():
        check_path = base_dir / check_path
    if not check_path.exists():
        target.errors.append(f"{label} does not exist: {path_value}.")

def _validate_model_candidate(candidate: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(candidate, "schema_version", MODEL_CANDIDATE_SCHEMA_VERSION, target, prefix="model_candidate.")
    target.errors.extend(model_candidate_errors(candidate))
    license_review = candidate.get("license") if isinstance(candidate.get("license"), dict) else {}
    compatibility = candidate.get("compatibility") if isinstance(candidate.get("compatibility"), dict) else {}
    target.details.update(
        {
            "candidate_id": candidate.get("candidate_id"),
            "model_id": candidate.get("model_id"),
            "license_status": license_review.get("status"),
            "license_review_status": license_review.get("review_status"),
            "context_length": compatibility.get("context_length"),
        }
    )

def _validate_model_compatibility_report(report: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(
        report,
        "schema_version",
        MODEL_COMPATIBILITY_REPORT_SCHEMA_VERSION,
        target,
        prefix="model_compatibility_report.",
    )
    target.errors.extend(model_compatibility_report_errors(report))
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    target.details.update(
        {
            "candidate_id": report.get("candidate_id"),
            "model_id": report.get("model_id"),
            "readiness": report.get("readiness"),
            "probe_count": summary.get("probe_count"),
            "verified_count": summary.get("verified_count"),
            "metadata_only_count": summary.get("metadata_only_count"),
        }
    )

def _validate_model_serving_probe_receipt(
    receipt: dict[str, Any], target: ValidationTarget, source_path: Path | None = None
) -> None:
    _require_equal(
        receipt,
        "schema_version",
        MODEL_SERVING_PROBE_RECEIPT_SCHEMA_VERSION,
        target,
        prefix="model_serving_probe_receipt.",
    )
    target.errors.extend(model_serving_probe_receipt_errors(receipt))
    compatibility_report = receipt.get("compatibility_report")
    if source_path is not None and isinstance(compatibility_report, dict):
        _validate_model_layer_file_ref(
            compatibility_report,
            target,
            "model_serving_probe_receipt.compatibility_report",
            source_path,
        )
    summary = receipt.get("summary") if isinstance(receipt.get("summary"), dict) else {}
    profile = receipt.get("serving_profile") if isinstance(receipt.get("serving_profile"), dict) else {}
    target.details.update(
        {
            "entry_id": receipt.get("entry_id"),
            "candidate_id": receipt.get("candidate_id"),
            "model_id": receipt.get("model_id"),
            "probe_mode": receipt.get("probe_mode"),
            "readiness": receipt.get("readiness"),
            "profile_id": profile.get("profile_id"),
            "provider": profile.get("provider"),
            "serving_engine": profile.get("serving_engine"),
            "probe_count": summary.get("probe_count"),
            "verified_count": summary.get("verified_count"),
            "metadata_only_count": summary.get("metadata_only_count"),
            "not_run_count": summary.get("not_run_count"),
        }
    )

def _validate_model_adapter_manifest(
    manifest: dict[str, Any], target: ValidationTarget, source_path: Path | None = None
) -> None:
    _require_equal(
        manifest,
        "schema_version",
        MODEL_ADAPTER_MANIFEST_SCHEMA_VERSION,
        target,
        prefix="model_adapter_manifest.",
    )
    target.errors.extend(model_adapter_manifest_errors(manifest))
    base_model = manifest.get("base_model") if isinstance(manifest.get("base_model"), dict) else {}
    training_plan = manifest.get("training_plan") if isinstance(manifest.get("training_plan"), dict) else {}
    if source_path is not None:
        _validate_model_layer_file_ref(
            training_plan,
            target,
            "model_adapter_manifest.training_plan",
            source_path,
        )
    target.details.update(
        {
            "adapter_id": manifest.get("adapter_id"),
            "adapter_kind": manifest.get("adapter_kind"),
            "readiness": manifest.get("readiness"),
            "entry_id": base_model.get("entry_id"),
            "candidate_id": base_model.get("candidate_id"),
            "model_id": base_model.get("model_id"),
            "training_plan_path": training_plan.get("path"),
            "training_plan_sha256": training_plan.get("sha256"),
        }
    )

def _validate_model_registry_entry(entry: dict[str, Any], target: ValidationTarget, source_path: Path | None = None) -> None:
    _require_equal(entry, "schema_version", MODEL_REGISTRY_ENTRY_SCHEMA_VERSION, target, prefix="model_registry_entry.")
    target.errors.extend(model_registry_entry_errors(entry))
    links = entry.get("links") if isinstance(entry.get("links"), dict) else {}
    if source_path is not None:
        _validate_model_registry_links_files(links, target, "model_registry_entry.links", source_path)
    target.details.update(
        {
            "entry_id": entry.get("entry_id"),
            "candidate_id": entry.get("candidate_id"),
            "training_eligible": entry.get("training_eligible"),
            "license_status": entry.get("license_status"),
            "link_counts": {
                key: len(value)
                for key, value in links.items()
                if isinstance(key, str) and isinstance(value, list)
            },
        }
    )

def _validate_model_registry(registry: dict[str, Any], target: ValidationTarget, source_path: Path | None = None) -> None:
    _require_equal(registry, "schema_version", MODEL_REGISTRY_SCHEMA_VERSION, target, prefix="model_registry.")
    target.errors.extend(model_registry_errors(registry))
    entries = registry.get("entries") if isinstance(registry.get("entries"), dict) else {}
    aliases = registry.get("aliases") if isinstance(registry.get("aliases"), dict) else {}
    link_counts: dict[str, int] = {}
    for entry_id, entry in entries.items():
        links = entry.get("links") if isinstance(entry, dict) and isinstance(entry.get("links"), dict) else {}
        if source_path is not None:
            _validate_model_registry_links_files(links, target, f"model_registry.entries.{entry_id}.links", source_path)
        for collection, records in links.items():
            if isinstance(collection, str) and isinstance(records, list):
                link_counts[collection] = link_counts.get(collection, 0) + len(records)
    target.details.update(
        {
            "entry_count": len(entries),
            "training_eligible_count": sum(
                1 for entry in entries.values() if isinstance(entry, dict) and entry.get("training_eligible") is True
            ),
            "aliases": aliases,
            "link_counts": link_counts,
        }
    )

def _validate_model_registry_links_files(
    links: Any, target: ValidationTarget, label: str, source_path: Path
) -> None:
    if not isinstance(links, dict):
        return
    for collection, records in links.items():
        if not isinstance(collection, str) or not isinstance(records, list):
            continue
        for index, record in enumerate(records):
            record_label = f"{label}.{collection}[{index}]"
            if not isinstance(record, dict) or "path" not in record:
                continue
            path_value = record.get("path")
            if not isinstance(path_value, str) or not path_value:
                continue
            link_path = resolve_artifact_reference_path(path_value, source_path)
            if link_path.is_symlink():
                target.errors.append(f"{record_label}.path must not resolve to a symlink.")
                continue
            if _path_has_symlink_component(link_path, include_leaf=False):
                target.errors.append(f"{record_label}.path must not traverse symlinked components.")
                continue
            if not link_path.is_file():
                target.errors.append(f"{record_label}.path does not resolve to a linked artifact file.")
                continue
            if _is_non_negative_int(record.get("size_bytes")) and link_path.stat().st_size != record.get("size_bytes"):
                target.errors.append(f"{record_label}.size_bytes does not match the current file.")
            if _is_sha256(record.get("sha256")) and _sha256(link_path) != record.get("sha256"):
                target.errors.append(f"{record_label}.sha256 does not match the current file.")

def _validate_model_layer_file_ref(
    record: dict[str, Any], target: ValidationTarget, label: str, source_path: Path
) -> None:
    if "sha256" not in record and "size_bytes" not in record:
        return
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        return
    link_path = resolve_artifact_reference_path(path_value, source_path)
    if link_path.is_symlink():
        target.errors.append(f"{label}.path must not resolve to a symlink.")
        return
    if _path_has_symlink_component(link_path, include_leaf=False):
        target.errors.append(f"{label}.path must not traverse symlinked components.")
        return
    if not link_path.is_file():
        target.errors.append(f"{label}.path does not resolve to a linked artifact file.")
        return
    if not _is_non_negative_int(record.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer for path-backed refs.")
    elif link_path.stat().st_size != record.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if not _is_sha256(record.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a 64-character hex digest for path-backed refs.")
    elif _sha256(link_path) != record.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current file.")

def _validate_training_plan(plan: dict[str, Any], target: ValidationTarget, source_path: Path | None = None) -> None:
    _require_equal(plan, "schema_version", TRAINING_PLAN_SCHEMA_VERSION, target, prefix="training_plan.")
    target.errors.extend(training_plan_errors(plan))
    model = plan.get("model") if isinstance(plan.get("model"), dict) else {}
    dataset = plan.get("dataset") if isinstance(plan.get("dataset"), dict) else {}
    compatibility_report = plan.get("compatibility_report") if isinstance(plan.get("compatibility_report"), dict) else {}
    if source_path is not None:
        _validate_model_layer_file_ref(
            compatibility_report,
            target,
            "training_plan.compatibility_report",
            source_path,
        )
    target.details.update(
        {
            "model_ref": model.get("model_ref"),
            "candidate_id": model.get("candidate_id"),
            "dataset_id": dataset.get("dataset_id"),
            "compatibility_report_path": compatibility_report.get("path"),
            "compatibility_report_sha256": compatibility_report.get("sha256"),
            "dry_run": plan.get("dry_run"),
            "gpu_execution": plan.get("gpu_execution"),
            "no_weight_download": plan.get("no_weight_download"),
        }
    )
