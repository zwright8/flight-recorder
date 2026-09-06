"""Extracted validation implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any
from ..calibration import REVIEW_CALIBRATION_SCHEMA_VERSION, build_review_calibration
from ..schema_registry import SchemaRegistryError, check_schema_contract, check_schema_file
from ..data_governance import task_contract_fingerprint
from ..review import REVIEW_CONFIDENCE_LEVELS, REVIEW_ITEM_SCHEMA_VERSION, REVIEW_LABEL_SCHEMA_VERSION, REVIEW_LABELS, REVIEW_MANIFEST_SCHEMA_VERSION, TRAINING_NEGATIVE_LABELS, review_item_sha256, _reviewed_dpo as _build_reviewed_dpo, _reviewed_labels as _build_reviewed_labels, _reviewed_preferences as _build_reviewed_preferences, _reviewed_reward_model as _build_reviewed_reward_model, _reviewed_action_sft as _build_reviewed_action_sft, _reviewed_sft as _build_reviewed_sft, REVIEWED_DPO_SCHEMA_VERSION, REVIEWED_LABEL_SCHEMA_VERSION, REVIEWED_MANIFEST_SCHEMA_VERSION, REVIEWED_PREFERENCE_SCHEMA_VERSION, REVIEWED_REWARD_MODEL_SCHEMA_VERSION, REVIEWED_SFT_SCHEMA_VERSION, _reviewed_dataset_version_id
from ..reviewed_gate import REVIEWED_EXPORT_SOURCE_ARTIFACT_FIELDS, REVIEWED_GATE_SCHEMA_VERSION, ReviewedGateError, build_reviewed_export_source_artifact, evaluate_reviewed_gate
from ..review_semantics import REVIEWED_ACTION_SFT_SCHEMA_VERSION
from ..path_safety import path_has_symlink_component as _path_has_symlink_component, resolve_artifact_reference_path
from ..training import DATASET_SPLIT_ARTIFACTS, DATASET_SPLIT_NAMES, RL_CURRICULUM_SCHEMA_VERSION, RL_ACTION_SFT_SCHEMA_VERSION, RL_DATASET_REGISTRY_SCHEMA_VERSION, RL_DATASET_METRICS_SCHEMA_VERSION, RL_DATASET_SPLITS_SCHEMA_VERSION, RL_DPO_SCHEMA_VERSION, RL_EPISODE_SCHEMA_VERSION, RL_FAILURE_MODE_SCHEMA_VERSION, RL_LABEL_PROVENANCE_SCHEMA_VERSION, COMPARE_RL_DPO_SCHEMA_VERSION, COMPARE_RL_MANIFEST_SCHEMA_VERSION, COMPARE_RL_PAIR_SCHEMA_VERSION, RL_MANIFEST_SCHEMA_VERSION, RL_PREFERENCE_SCHEMA_VERSION, RL_REWARD_SCHEMA_VERSION, RL_REWARD_MODEL_SCHEMA_VERSION, RL_SFT_SCHEMA_VERSION, RL_STEP_REWARD_SCHEMA_VERSION, RL_TRAINER_VIEWS_CONTRACT_VERSION, REWARD_SCALES, build_label_provenance_summary, build_redaction_status, positive_label_eligible, redaction_scan_artifacts
from ..hashing import sha256_file as _sha256
from .exports import _review_export_artifact_paths, _reviewed_export_artifact_paths, _validate_manifest_artifact_fingerprints
from .primitives import ValidationTarget, _is_dataset_version, _is_int_between, _is_lowercase_sha256, _is_non_negative_int, _is_number_between, _is_sha256, _is_string_list, _rate_value, _read_jsonl_objects, _read_object, _read_object_optional, _reject_symlinked_validation_path, _require_equal, _sha256, _validate_allowed_keys, _validate_gate_like_checks, _validate_public_review_fingerprint_paths, _validate_public_review_ref_path

def validate_review_export(path: str | Path) -> ValidationTarget:
    """Validate a human-review export directory."""
    export_dir = Path(path)
    target = ValidationTarget("review_export", str(export_dir))
    if _reject_symlinked_validation_path(export_dir, target, "Review export path", "directory"):
        return target
    if not export_dir.exists():
        target.errors.append(f"Review export directory not found: {export_dir}")
        return target
    if not export_dir.is_dir():
        target.errors.append(f"Review export path is not a directory: {export_dir}")
        return target

    manifest_path = export_dir / "manifest.json"
    manifest = (
        None
        if _reject_symlinked_validation_path(manifest_path, target, "manifest.json", "file")
        else _read_object(manifest_path, target, "manifest.json")
    )
    items = _read_jsonl_objects(export_dir / "review_items.jsonl", target, "review_items.jsonl")
    labels = _read_jsonl_objects(export_dir / "label_template.jsonl", target, "label_template.jsonl")
    if not (export_dir / "REVIEW_INSTRUCTIONS.md").exists():
        target.warnings.append("REVIEW_INSTRUCTIONS.md is missing; review workflow is less self-documenting.")
    if manifest is not None:
        _validate_review_manifest(manifest, target, items, labels)
        _validate_manifest_artifact_fingerprints(
            manifest.get("artifact_fingerprints"),
            target,
            "manifest.artifact_fingerprints",
            _review_export_artifact_paths(export_dir),
        )
    _validate_review_items(items, target)
    _validate_review_labels(labels, target, items)
    target.details.update({"item_count": len(items), "label_count": len(labels)})
    return target

def validate_reviewed_export(path: str | Path) -> ValidationTarget:
    """Validate an apply-review output directory."""
    export_dir = Path(path)
    target = ValidationTarget("reviewed_export", str(export_dir))
    if _reject_symlinked_validation_path(export_dir, target, "Reviewed export path", "directory"):
        return target
    if not export_dir.exists():
        target.errors.append(f"Reviewed export directory not found: {export_dir}")
        return target
    if not export_dir.is_dir():
        target.errors.append(f"Reviewed export path is not a directory: {export_dir}")
        return target

    manifest_path = export_dir / "manifest.json"
    manifest = (
        None
        if _reject_symlinked_validation_path(manifest_path, target, "manifest.json", "file")
        else _read_object(manifest_path, target, "manifest.json")
    )
    labels = _read_jsonl_objects(export_dir / "reviewed_labels.jsonl", target, "reviewed_labels.jsonl")
    sft = _read_jsonl_objects(export_dir / "reviewed_sft.jsonl", target, "reviewed_sft.jsonl")
    action_sft = _read_jsonl_objects(
        export_dir / "reviewed_action_sft.jsonl",
        target,
        "reviewed_action_sft.jsonl",
    )
    reward_model = _read_jsonl_objects(export_dir / "reviewed_reward_model.jsonl", target, "reviewed_reward_model.jsonl")
    preferences = _read_jsonl_objects(export_dir / "reviewed_preferences.jsonl", target, "reviewed_preferences.jsonl")
    dpo = _read_jsonl_objects(export_dir / "reviewed_dpo.jsonl", target, "reviewed_dpo.jsonl")
    provenance_items = _read_jsonl_objects(
        export_dir / "provenance" / "review_items.jsonl",
        target,
        "provenance/review_items.jsonl",
    )
    provenance_labels = _read_jsonl_objects(
        export_dir / "provenance" / "completed_labels.jsonl",
        target,
        "provenance/completed_labels.jsonl",
    )
    provenance_label_template = _read_jsonl_objects(
        export_dir / "provenance" / "label_template.jsonl",
        target,
        "provenance/label_template.jsonl",
    )
    provenance_review_manifest = _read_object(
        export_dir / "provenance" / "review_manifest.json",
        target,
        "provenance/review_manifest.json",
    )
    dataset_registry_path = export_dir / "dataset_registry.json"
    dataset_registry = (
        None
        if _reject_symlinked_validation_path(dataset_registry_path, target, "dataset_registry.json", "file")
        else _read_object_optional(
            dataset_registry_path,
            target,
            "dataset_registry.json",
            "rerun apply-review to emit a selectable reviewed dataset registry",
        )
    )
    rows_by_artifact = {
        "reviewed_labels": labels,
        "reviewed_sft": sft,
        "reviewed_action_sft": action_sft,
        "reviewed_reward_model": reward_model,
        "reviewed_preferences": preferences,
        "reviewed_dpo": dpo,
        "provenance_review_items": provenance_items,
        "provenance_label_template": provenance_label_template,
        "provenance_completed_labels": provenance_labels,
    }
    expected_redaction_status = build_redaction_status(
        redaction_scan_artifacts(
            rows_by_artifact,
            extra_artifacts={
                "manifest": manifest or {},
                "dataset_registry": dataset_registry or {},
                "provenance_review_manifest": provenance_review_manifest or {},
            },
        )
    )
    expected_label_provenance = _expected_reviewed_label_provenance(
        labels,
        sft,
        action_sft,
        reward_model,
        preferences,
        dpo,
    )
    if expected_redaction_status.get("passed") is not True:
        target.errors.append("reviewed export contains unredacted secret-like values.")
    if manifest is not None:
        _validate_reviewed_manifest(
            manifest,
            target,
            labels,
            sft,
            action_sft,
            reward_model,
            preferences,
            dpo,
            dataset_registry,
            expected_redaction_status,
            expected_label_provenance,
            export_dir,
        )
        _validate_manifest_artifact_fingerprints(
            manifest.get("artifact_fingerprints"),
            target,
            "manifest.artifact_fingerprints",
            _reviewed_export_artifact_paths(export_dir),
        )
    if provenance_review_manifest is not None:
        _validate_review_manifest(
            provenance_review_manifest,
            target,
            provenance_items,
            provenance_label_template,
        )
        provenance_fingerprints = provenance_review_manifest.get(
            "artifact_fingerprints"
        )
        if isinstance(provenance_fingerprints, dict):
            replayable_fingerprints = {
                name: provenance_fingerprints.get(name)
                for name in ("review_items", "label_template")
            }
        else:
            replayable_fingerprints = provenance_fingerprints
        _validate_manifest_artifact_fingerprints(
            replayable_fingerprints,
            target,
            "provenance/review_manifest.json.artifact_fingerprints",
            {
                "review_items": export_dir / "provenance" / "review_items.jsonl",
                "label_template": export_dir
                / "provenance"
                / "label_template.jsonl",
            },
        )
    _validate_review_items(provenance_items, target)
    _validate_review_labels(
        provenance_label_template,
        target,
        provenance_items,
    )
    _validate_review_labels(
        provenance_labels,
        target,
        provenance_items,
    )
    _validate_reviewed_labels(labels, target)
    _validate_reviewed_sft(sft, target, labels)
    _validate_reviewed_action_sft(action_sft, target, labels)
    _validate_reviewed_reward_model(reward_model, target, labels)
    _validate_reviewed_preferences(preferences, target, labels)
    _validate_reviewed_dpo(dpo, target, preferences)
    item_ids = [item.get("review_item_id") for item in provenance_items if isinstance(item.get("review_item_id"), str)]
    if len(item_ids) != len(set(item_ids)):
        target.errors.append("provenance/review_items.jsonl must not contain duplicate review_item_id values.")
    expected_reviewed_labels: list[dict[str, Any]] | None = None
    try:
        expected_reviewed_labels = _build_reviewed_labels(
            {
                item["review_item_id"]: item
                for item in provenance_items
                if isinstance(item.get("review_item_id"), str) and item.get("review_item_id")
            },
            provenance_labels,
            export_dir / "provenance" / "completed_labels.jsonl",
            False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        target.errors.append(f"reviewed export provenance could not replay human labels: {exc}")
    else:
        for row in expected_reviewed_labels:
            row["source_label_file"] = "provenance/completed_labels.jsonl"
        if labels != expected_reviewed_labels:
            target.errors.append(
                "reviewed_labels.jsonl must match deterministic replay of provenance review_items and completed_labels."
            )
    _validate_reviewed_trainer_view_replay(
        manifest,
        expected_reviewed_labels,
        sft,
        action_sft,
        reward_model,
        preferences,
        dpo,
        target,
    )
    target.details.update(
        {
            "reviewed_label_count": len(labels),
            "sft_count": len(sft),
            "action_sft_count": len(action_sft),
            "reward_model_count": len(reward_model),
            "preference_count": len(preferences),
            "dpo_count": len(dpo),
        }
    )
    return target

def _validate_reviewed_trainer_view_replay(
    manifest: dict[str, Any] | None,
    reviewed_labels: list[dict[str, Any]] | None,
    sft: list[dict[str, Any]],
    action_sft: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    target: ValidationTarget,
) -> None:
    """Require every trainer view to be the canonical projection of human labels."""
    if manifest is None:
        target.errors.append(
            "reviewed trainer views could not be replayed because manifest.json is unavailable."
        )
        return
    max_pairs_per_family = manifest.get("max_pairs_per_family")
    if (
        not isinstance(max_pairs_per_family, int)
        or isinstance(max_pairs_per_family, bool)
        or max_pairs_per_family < 0
    ):
        target.errors.append(
            "manifest.max_pairs_per_family must be a non-negative integer for reviewed trainer-view replay."
        )
        return
    if reviewed_labels is None:
        target.errors.append(
            "reviewed trainer views could not be replayed because provenance labels did not produce a valid reviewed-label projection."
        )
        return

    try:
        expected_sft = _build_reviewed_sft(reviewed_labels)
        expected_action_sft = _build_reviewed_action_sft(reviewed_labels)
        expected_reward_model = _build_reviewed_reward_model(reviewed_labels)
        expected_preferences = _build_reviewed_preferences(
            reviewed_labels,
            max_pairs_per_family=max_pairs_per_family,
        )
        expected_dpo = _build_reviewed_dpo(expected_preferences)
    except (KeyError, TypeError, ValueError) as exc:
        target.errors.append(f"reviewed trainer views could not be replayed: {exc}")
        return

    for artifact_name, actual, expected in (
        ("reviewed_sft.jsonl", sft, expected_sft),
        ("reviewed_action_sft.jsonl", action_sft, expected_action_sft),
        ("reviewed_reward_model.jsonl", reward_model, expected_reward_model),
        ("reviewed_preferences.jsonl", preferences, expected_preferences),
        ("reviewed_dpo.jsonl", dpo, expected_dpo),
    ):
        if actual != expected:
            target.errors.append(
                f"{artifact_name} must match deterministic replay of reviewed_labels.jsonl."
            )

_REVIEWED_GATE_KEYS = {
    "schema_version",
    "reviewed_export",
    "source_artifacts",
    "effective_policy",
    "passed",
    "check_count",
    "failed_check_count",
    "checks",
    "metrics",
    "decision",
    "policy",
}

_REVIEWED_GATE_EFFECTIVE_POLICY_KEYS = {
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
}

def validate_reviewed_gate(path: str | Path) -> ValidationTarget:
    """Validate a reviewed gate against its exact current reviewed export."""
    from .dispatch import validate_artifacts
    source_path = Path(path)
    target = ValidationTarget("reviewed_gate", str(source_path))
    if _reject_symlinked_validation_path(source_path, target, "reviewed_gate", "file"):
        return target
    gate = _read_object(source_path, target, "reviewed_gate.json")
    if gate is None:
        return target

    try:
        schema = check_schema_contract(gate, name_or_id="reviewed_gate")
    except (SchemaRegistryError, TypeError, ValueError) as exc:
        target.errors.append(f"reviewed_gate schema contract could not be checked: {exc}")
    else:
        for error in schema.get("errors", []):
            target.errors.append(f"reviewed_gate schema: {error}")
    _validate_allowed_keys(gate, _REVIEWED_GATE_KEYS, target, "reviewed_gate")
    _require_equal(gate, "schema_version", REVIEWED_GATE_SCHEMA_VERSION, target, prefix="reviewed_gate.")

    source_artifacts = gate.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        target.errors.append("reviewed_gate.source_artifacts must be an object.")
        return target
    _validate_allowed_keys(source_artifacts, {"reviewed_export"}, target, "reviewed_gate.source_artifacts")
    source_record = source_artifacts.get("reviewed_export")
    if not isinstance(source_record, dict):
        target.errors.append("reviewed_gate.source_artifacts.reviewed_export must be an object.")
        return target
    _validate_allowed_keys(
        source_record,
        set(REVIEWED_EXPORT_SOURCE_ARTIFACT_FIELDS),
        target,
        "reviewed_gate.source_artifacts.reviewed_export",
    )
    path_value = source_record.get("path")
    if gate.get("reviewed_export") != path_value:
        target.errors.append("reviewed_gate.reviewed_export must match source_artifacts.reviewed_export.path.")
    export_path = _reviewed_gate_export_path(path_value, source_path, target)
    if export_path is None:
        return target

    try:
        source_before = build_reviewed_export_source_artifact(
            export_path,
            display_path=str(path_value),
        )
    except (OSError, ReviewedGateError, ValueError) as exc:
        target.errors.append(f"reviewed_gate reviewed export could not be fingerprinted: {exc}")
        return target

    current_validation = validate_artifacts(
        reviewed_export_dir=export_path,
        strict=bool(
            isinstance(gate.get("effective_policy"), dict)
            and gate["effective_policy"].get("strict_validation") is True
        ),
    )
    for validation_target in current_validation.get("targets", []):
        if not isinstance(validation_target, dict):
            continue
        for error in validation_target.get("errors", []):
            target.errors.append(f"reviewed_gate reviewed export: {error}")

    try:
        expected_source = build_reviewed_export_source_artifact(
            export_path,
            display_path=str(path_value),
        )
    except (OSError, ReviewedGateError, ValueError) as exc:
        target.errors.append(f"reviewed_gate reviewed export could not be fingerprinted: {exc}")
        return target
    if source_record != expected_source:
        target.errors.append(
            "reviewed_gate.source_artifacts.reviewed_export must match the current reviewed-export tree, manifest, and dataset version."
        )
    if source_before != expected_source:
        target.errors.append("reviewed_gate reviewed export changed while validation was running.")

    effective_policy = gate.get("effective_policy")
    if not isinstance(effective_policy, dict):
        target.errors.append("reviewed_gate.effective_policy must be an object.")
        return target
    _validate_allowed_keys(
        effective_policy,
        _REVIEWED_GATE_EFFECTIVE_POLICY_KEYS,
        target,
        "reviewed_gate.effective_policy",
    )
    missing_policy_fields = sorted(_REVIEWED_GATE_EFFECTIVE_POLICY_KEYS - set(effective_policy))
    if missing_policy_fields:
        target.errors.append(
            "reviewed_gate.effective_policy is missing field(s): " + ", ".join(missing_policy_fields) + "."
        )
        return target
    if effective_policy.get("require_valid_export") is not True:
        target.errors.append(
            "reviewed_gate.effective_policy.require_valid_export must be true before the gate can authorize downstream work."
        )

    manifest = _read_object(export_path / "manifest.json", target, "reviewed export manifest.json")
    if manifest is None:
        return target
    validation_summary = current_validation if effective_policy.get("require_valid_export") is True else None
    try:
        replayed = evaluate_reviewed_gate(
            manifest,
            reviewed_export_path=str(path_value),
            reviewed_export_source=expected_source,
            min_reviewed_labels=effective_policy.get("min_reviewed_labels"),
            min_accepted=effective_policy.get("min_accepted"),
            min_rejected=effective_policy.get("min_rejected"),
            min_sft=effective_policy.get("min_sft"),
            min_reward_model=effective_policy.get("min_reward_model"),
            min_preferences=effective_policy.get("min_preferences"),
            min_dpo=effective_policy.get("min_dpo"),
            min_high_confidence_labels=effective_policy.get("min_high_confidence_labels"),
            min_medium_or_high_confidence_labels=effective_policy.get("min_medium_or_high_confidence_labels"),
            max_needs_review=effective_policy.get("max_needs_review"),
            max_low_confidence_labels=effective_policy.get("max_low_confidence_labels"),
            max_unknown_confidence_labels=effective_policy.get("max_unknown_confidence_labels"),
            forbid_labels=effective_policy.get("forbid_labels"),
            require_task_families=effective_policy.get("require_task_families"),
            validation_summary=validation_summary,
            require_valid_export=effective_policy.get("require_valid_export") is True,
            strict_validation=effective_policy.get("strict_validation") is True,
        )
    except (TypeError, ValueError) as exc:
        target.errors.append(f"reviewed_gate could not replay its effective policy: {exc}")
        return target

    policy = gate.get("policy")
    if policy is not None:
        if not isinstance(policy, dict):
            target.errors.append("reviewed_gate.policy must be an object when present.")
        else:
            expected_effective = {
                field: effective_policy[field]
                for field in _REVIEWED_GATE_EFFECTIVE_POLICY_KEYS
                if effective_policy[field] is not None and effective_policy[field] != []
            }
            if policy.get("effective") != expected_effective:
                target.errors.append("reviewed_gate.policy.effective must match effective_policy.")
            replayed["policy"] = policy
    if gate != replayed:
        target.errors.append("reviewed_gate must match deterministic replay of its current source and effective policy exactly.")
    try:
        source_after = build_reviewed_export_source_artifact(
            export_path,
            display_path=str(path_value),
        )
    except (OSError, ReviewedGateError, ValueError) as exc:
        target.errors.append(f"reviewed_gate reviewed export could not be reattested: {exc}")
    else:
        if source_after != expected_source:
            target.errors.append("reviewed_gate reviewed export changed while replay was running.")

    target.details.update(
        {
            "passed": gate.get("passed"),
            "dataset_version": source_record.get("dataset_version"),
            "reviewed_export": str(path_value),
        }
    )
    return target

def _reviewed_gate_export_path(value: Any, source_path: Path, target: ValidationTarget) -> Path | None:
    label = "reviewed_gate.source_artifacts.reviewed_export.path"
    if not isinstance(value, str) or not value:
        target.errors.append(f"{label} must be a non-empty string.")
        return None
    windows_path = PureWindowsPath(value)
    if (
        Path(value).is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or "\\" in value
        or "\x00" in value
        or "://" in value
        or value.startswith("<redacted:")
        or value in {".", ".."}
        or ".." in Path(value).parts
        or "." in Path(value).parts
        or any(part.startswith("~") for part in Path(value).parts)
    ):
        target.errors.append(f"{label} must be a resolvable relative path.")
        return None
    export_path = source_path.parent / value
    if _reject_symlinked_validation_path(export_path, target, label, "directory"):
        return None
    if not export_path.exists() or not export_path.is_dir():
        target.errors.append(f"{label} does not resolve to an existing directory: {export_path}")
        return None
    return export_path

def validate_review_calibration(path: str | Path) -> ValidationTarget:
    """Validate a review-calibration report."""
    calibration_path = Path(path)
    target = ValidationTarget("review_calibration", str(calibration_path))
    if _reject_symlinked_validation_path(calibration_path, target, "review_calibration.path", "file"):
        return target
    calibration = _read_object(calibration_path, target, "review_calibration.json")
    if calibration is not None:
        _validate_review_calibration(calibration, target, calibration_path)
    return target

def _validate_review_manifest(
    manifest: dict[str, Any],
    target: ValidationTarget,
    items: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> None:
    _require_equal(manifest, "schema_version", REVIEW_MANIFEST_SCHEMA_VERSION, target)
    if manifest.get("item_count") != len(items):
        target.errors.append(f"manifest.item_count expected {len(items)}, got {manifest.get('item_count')!r}.")
    passed_count = sum(1 for item in items if _review_item_passed(item) is True)
    failed_count = sum(1 for item in items if _review_item_passed(item) is False)
    if manifest.get("passed_count") != passed_count:
        target.errors.append(f"manifest.passed_count expected {passed_count}, got {manifest.get('passed_count')!r}.")
    if manifest.get("failed_count") != failed_count:
        target.errors.append(f"manifest.failed_count expected {failed_count}, got {manifest.get('failed_count')!r}.")
    if len(labels) != len(items):
        target.errors.append(f"label_template row count expected {len(items)}, got {len(labels)}.")
    if manifest.get("label_options") != list(REVIEW_LABELS):
        target.errors.append(f"manifest.label_options must be {list(REVIEW_LABELS)!r}.")
    if manifest.get("confidence_options") != list(REVIEW_CONFIDENCE_LEVELS):
        target.errors.append(f"manifest.confidence_options must be {list(REVIEW_CONFIDENCE_LEVELS)!r}.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        target.errors.append("manifest.outputs must be an object.")
    else:
        for output_name in ("review_items", "label_template", "instructions", "manifest"):
            if output_name not in outputs:
                target.errors.append(f"manifest.outputs.{output_name} is missing.")
    _validate_public_review_ref_path(target, "manifest.source_runs_dir", manifest.get("source_runs_dir"))
    _validate_public_review_ref_path(target, "manifest.output_dir", manifest.get("output_dir"))
    if isinstance(outputs, dict):
        for output_name, output_path in outputs.items():
            _validate_public_review_ref_path(target, f"manifest.outputs.{output_name}", output_path)
    _validate_public_review_fingerprint_paths(
        manifest.get("artifact_fingerprints"),
        target,
        "manifest.artifact_fingerprints",
    )

def _validate_review_items(items: list[dict[str, Any]], target: ValidationTarget) -> None:
    seen: set[str] = set()
    for index, item in enumerate(items):
        _require_equal(item, "schema_version", REVIEW_ITEM_SCHEMA_VERSION, target, prefix=f"review_items[{index}].")
        item_id = item.get("review_item_id")
        if not isinstance(item_id, str) or not item_id:
            target.errors.append(f"review_items[{index}].review_item_id must be a non-empty string.")
            continue
        if item_id in seen:
            target.errors.append(f"review_items[{index}].review_item_id duplicates {item_id!r}.")
        seen.add(item_id)
        item_hash = item.get("review_item_sha256")
        if not _is_sha256(item_hash):
            target.errors.append(f"review_items[{index}].review_item_sha256 must be a SHA-256 hex string.")
        elif item_hash != review_item_sha256(item):
            target.errors.append(f"review_items[{index}].review_item_sha256 does not match review item contents.")
        for field_name in ("episode_id", "scenario_id", "scenario_title", "task_family", "prompt", "final_answer", "suggested_human_label"):
            if not isinstance(item.get(field_name), str):
                target.errors.append(f"review_items[{index}].{field_name} must be a string.")
        if item.get("suggested_human_label") not in REVIEW_LABELS:
            target.errors.append(f"review_items[{index}].suggested_human_label must be one of {list(REVIEW_LABELS)!r}.")
        if item.get("label_options") != list(REVIEW_LABELS):
            target.errors.append(f"review_items[{index}].label_options must be {list(REVIEW_LABELS)!r}.")
        if not isinstance(item.get("event_count"), int) or isinstance(item.get("event_count"), bool) or item.get("event_count") < 0:
            target.errors.append(f"review_items[{index}].event_count must be a non-negative integer.")
        if not _is_lowercase_sha256(item.get("episode_events_sha256")):
            target.errors.append(
                f"review_items[{index}].episode_events_sha256 must be a lowercase SHA-256 hex string."
            )
        source_artifacts = item.get("source_artifacts")
        if not isinstance(source_artifacts, dict):
            target.errors.append(f"review_items[{index}].source_artifacts must be an object.")
        else:
            for artifact_name in ("run_dir", "normalized_trace", "scorecard", "report"):
                if not isinstance(source_artifacts.get(artifact_name), str) or not source_artifacts.get(artifact_name):
                    target.errors.append(f"review_items[{index}].source_artifacts.{artifact_name} must be a non-empty string.")
                else:
                    _validate_public_review_ref_path(
                        target,
                        f"review_items[{index}].source_artifacts.{artifact_name}",
                        source_artifacts.get(artifact_name),
                    )
            for artifact_name in ("lineage", "regression_scenario"):
                if artifact_name in source_artifacts:
                    if not isinstance(source_artifacts.get(artifact_name), str) or not source_artifacts.get(artifact_name):
                        target.errors.append(f"review_items[{index}].source_artifacts.{artifact_name} must be a non-empty string when present.")
                    else:
                        _validate_public_review_ref_path(
                            target,
                            f"review_items[{index}].source_artifacts.{artifact_name}",
                            source_artifacts.get(artifact_name),
                        )
        source_fingerprints = item.get("source_artifact_fingerprints")
        if not isinstance(source_fingerprints, dict):
            target.errors.append(
                f"review_items[{index}].source_artifact_fingerprints must be an object."
            )
        else:
            for artifact_name in ("normalized_trace", "scorecard"):
                fingerprint = source_fingerprints.get(artifact_name)
                label = (
                    f"review_items[{index}].source_artifact_fingerprints."
                    f"{artifact_name}"
                )
                if not isinstance(fingerprint, dict):
                    target.errors.append(f"{label} must be an object.")
                    continue
                if fingerprint.get("algorithm") != "sha256-canonical-json-v1":
                    target.errors.append(
                        f"{label}.algorithm must be 'sha256-canonical-json-v1'."
                    )
                if not _is_lowercase_sha256(fingerprint.get("sha256")):
                    target.errors.append(
                        f"{label}.sha256 must be a lowercase SHA-256 hex string."
                    )
                size_bytes = fingerprint.get("size_bytes")
                if (
                    not isinstance(size_bytes, int)
                    or isinstance(size_bytes, bool)
                    or size_bytes < 0
                ):
                    target.errors.append(
                        f"{label}.size_bytes must be a non-negative integer."
                    )
        scorecard = item.get("scorecard")
        if not isinstance(scorecard, dict):
            target.errors.append(f"review_items[{index}].scorecard must be an object.")
        else:
            if not isinstance(scorecard.get("passed"), bool):
                target.errors.append(f"review_items[{index}].scorecard.passed must be a boolean.")
            if not _is_int_between(scorecard.get("score"), 0, 100):
                target.errors.append(f"review_items[{index}].scorecard.score must be an integer from 0 to 100.")
            if not _is_string_list(scorecard.get("failed_rules")):
                target.errors.append(f"review_items[{index}].scorecard.failed_rules must be a list of strings.")
            if not _is_string_list(scorecard.get("critical_failures")):
                target.errors.append(f"review_items[{index}].scorecard.critical_failures must be a list of strings.")
        if not isinstance(item.get("rule_summaries"), list):
            target.errors.append(f"review_items[{index}].rule_summaries must be a list.")
        if not isinstance(item.get("task_evidence"), list):
            target.errors.append(f"review_items[{index}].task_evidence must be a list.")
        if not isinstance(item.get("evidence_target_counts"), dict):
            target.errors.append(f"review_items[{index}].evidence_target_counts must be an object.")

def _validate_review_labels(labels: list[dict[str, Any]], target: ValidationTarget, items: list[dict[str, Any]]) -> None:
    item_by_id = {
        item.get("review_item_id"): item
        for item in items
        if isinstance(item.get("review_item_id"), str)
    }
    item_ids = set(item_by_id)
    seen: set[str] = set()
    for index, label in enumerate(labels):
        _require_equal(label, "schema_version", REVIEW_LABEL_SCHEMA_VERSION, target, prefix=f"label_template[{index}].")
        item_id = label.get("review_item_id")
        if not isinstance(item_id, str) or not item_id:
            target.errors.append(f"label_template[{index}].review_item_id must be a non-empty string.")
            continue
        if item_id in seen:
            target.errors.append(f"label_template[{index}].review_item_id duplicates {item_id!r}.")
        seen.add(item_id)
        if item_id not in item_ids:
            target.errors.append(f"label_template[{index}].review_item_id does not reference a review item.")
            item = None
        else:
            item = item_by_id[item_id]
        label_hash = label.get("review_item_sha256")
        if not _is_sha256(label_hash):
            target.errors.append(f"label_template[{index}].review_item_sha256 must be a SHA-256 hex string.")
        elif item is not None and label_hash != review_item_sha256(item):
            target.errors.append(f"label_template[{index}].review_item_sha256 does not match referenced review item.")
        if item is not None:
            for field_name in ("episode_id", "scenario_id", "suggested_human_label"):
                if label.get(field_name) != item.get(field_name):
                    target.errors.append(f"label_template[{index}].{field_name} does not match referenced review item.")
        suggested = label.get("suggested_human_label")
        if suggested not in REVIEW_LABELS:
            target.errors.append(f"label_template[{index}].suggested_human_label must be one of {list(REVIEW_LABELS)!r}.")
        human_label = label.get("human_label")
        if human_label is not None and human_label not in REVIEW_LABELS:
            target.errors.append(f"label_template[{index}].human_label must be null or one of {list(REVIEW_LABELS)!r}.")
        reviewer_confidence = label.get("reviewer_confidence")
        if human_label is not None and reviewer_confidence is None:
            target.errors.append(
                f"label_template[{index}].reviewer_confidence is required when human_label is set."
            )
        elif reviewer_confidence is not None and reviewer_confidence not in REVIEW_CONFIDENCE_LEVELS:
            target.errors.append(
                f"label_template[{index}].reviewer_confidence must be null or one of {list(REVIEW_CONFIDENCE_LEVELS)!r}."
            )
        if human_label is not None:
            if not isinstance(label.get("reviewer"), str) or not label.get("reviewer", "").strip():
                target.errors.append(f"label_template[{index}].reviewer must be a non-empty string when human_label is set.")
            if not isinstance(label.get("reviewed_at"), str) or not label.get("reviewed_at", "").strip():
                target.errors.append(f"label_template[{index}].reviewed_at must be a non-empty string when human_label is set.")
        corrected_score = label.get("corrected_score")
        if corrected_score is not None and not _is_int_between(corrected_score, 0, 100):
            target.errors.append(f"label_template[{index}].corrected_score must be null or an integer from 0 to 100.")
        for field_name in ("accepted_evidence_refs", "rejected_evidence_refs"):
            if not isinstance(label.get(field_name), list):
                target.errors.append(f"label_template[{index}].{field_name} must be a list.")
    missing = sorted(item_ids - seen)
    if missing:
        target.errors.append(f"label_template missing review_item_id values: {missing!r}.")

def _review_item_passed(item: dict[str, Any]) -> bool | None:
    scorecard = item.get("scorecard")
    if isinstance(scorecard, dict) and isinstance(scorecard.get("passed"), bool):
        return scorecard["passed"]
    return None

def _validate_reviewed_manifest(
    manifest: dict[str, Any],
    target: ValidationTarget,
    labels: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    action_sft: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    dataset_registry: dict[str, Any] | None,
    expected_redaction_status: dict[str, Any],
    expected_label_provenance: dict[str, Any],
    export_dir: Path,
) -> None:
    _require_equal(manifest, "schema_version", REVIEWED_MANIFEST_SCHEMA_VERSION, target)
    if not _is_dataset_version(manifest.get("dataset_version")):
        target.errors.append("manifest.dataset_version must be a non-empty hfrds-* dataset selection key.")
    artifact_fingerprints = manifest.get("artifact_fingerprints")
    source_review_artifacts = manifest.get("source_review_artifacts")
    labels_artifact = manifest.get("labels_artifact")
    if all(isinstance(value, dict) for value in (artifact_fingerprints, source_review_artifacts, labels_artifact)):
        expected_dataset_version = _reviewed_dataset_version_id(
            artifact_fingerprints,
            source_review_artifacts,
            labels_artifact,
        )
        if manifest.get("dataset_version") != expected_dataset_version:
            target.errors.append(
                f"manifest.dataset_version expected {expected_dataset_version!r} from reviewed artifact fingerprints, "
                f"got {manifest.get('dataset_version')!r}."
            )
    provenance_paths = {
        "review_items": export_dir / "provenance" / "review_items.jsonl",
        "label_template": export_dir / "provenance" / "label_template.jsonl",
        "review_manifest": export_dir / "provenance" / "review_manifest.json",
    }
    _validate_manifest_artifact_fingerprints(
        source_review_artifacts,
        target,
        "manifest.source_review_artifacts",
        provenance_paths,
    )
    _validate_manifest_artifact_fingerprints(
        {"labels_artifact": labels_artifact} if isinstance(labels_artifact, dict) else labels_artifact,
        target,
        "manifest.labels_artifact_binding",
        {"labels_artifact": export_dir / "provenance" / "completed_labels.jsonl"},
    )
    expected_provenance_paths = {
        "review_items": "provenance/review_items.jsonl",
        "label_template": "provenance/label_template.jsonl",
        "review_manifest": "provenance/review_manifest.json",
    }
    if isinstance(source_review_artifacts, dict):
        for name, expected_path in expected_provenance_paths.items():
            record = source_review_artifacts.get(name)
            if isinstance(record, dict) and record.get("path") != expected_path:
                target.errors.append(f"manifest.source_review_artifacts.{name}.path must be {expected_path!r}.")
    if isinstance(labels_artifact, dict) and labels_artifact.get("path") != "provenance/completed_labels.jsonl":
        target.errors.append("manifest.labels_artifact.path must be 'provenance/completed_labels.jsonl'.")
    expected_counts = {
        "reviewed_label_count": len(labels),
        "sft_count": len(sft),
        "action_sft_count": len(action_sft),
        "reward_model_count": len(reward_model),
        "preference_count": len(preferences),
        "dpo_count": len(dpo),
    }
    for field_name, expected in expected_counts.items():
        if manifest.get(field_name) != expected:
            target.errors.append(f"manifest.{field_name} expected {expected}, got {manifest.get(field_name)!r}.")
    expected_label_counts = _reviewed_label_counts(labels)
    if manifest.get("label_counts") != expected_label_counts:
        target.errors.append(f"manifest.label_counts expected {expected_label_counts!r}, got {manifest.get('label_counts')!r}.")
    expected_confidence_counts = _reviewed_confidence_counts(labels)
    manifest_confidence_counts = manifest.get("confidence_counts")
    if not isinstance(manifest_confidence_counts, dict):
        target.errors.append("manifest.confidence_counts must be an object.")
    elif manifest_confidence_counts != expected_confidence_counts:
        target.errors.append(
            f"manifest.confidence_counts expected {expected_confidence_counts!r}, got {manifest_confidence_counts!r}."
        )
    expected_confidence_fields = {
        "high_confidence_label_count": expected_confidence_counts["high"],
        "medium_or_high_confidence_label_count": expected_confidence_counts["high"] + expected_confidence_counts["medium"],
        "low_confidence_label_count": expected_confidence_counts["low"],
        "unknown_confidence_label_count": expected_confidence_counts["unknown"],
    }
    for field_name, expected in expected_confidence_fields.items():
        if field_name not in manifest:
            target.errors.append(f"manifest.{field_name} is missing.")
        elif manifest.get(field_name) != expected:
            target.errors.append(f"manifest.{field_name} expected {expected}, got {manifest.get(field_name)!r}.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        target.errors.append("manifest.outputs must be an object.")
    else:
        for output_name in (
            "reviewed_labels",
            "reviewed_sft",
            "reviewed_action_sft",
            "reviewed_reward_model",
            "reviewed_preferences",
            "reviewed_dpo",
            "dataset_registry",
            "manifest",
        ):
            if output_name not in outputs:
                target.errors.append(f"manifest.outputs.{output_name} is missing.")
    if manifest.get("redaction_status") != expected_redaction_status:
        target.errors.append("manifest.redaction_status must match recomputed reviewed redaction scan.")
    if manifest.get("label_provenance") != expected_label_provenance:
        target.errors.append("manifest.label_provenance must match recomputed reviewed label provenance.")
    reviewed_trainer_views = _validate_reviewed_trainer_views(
        manifest,
        target,
        sft,
        action_sft,
        reward_model,
        dpo,
    )
    registry = manifest.get("registry")
    if not isinstance(registry, dict):
        target.errors.append("manifest.registry must be an object.")
    else:
        _require_equal(registry, "schema_version", RL_DATASET_REGISTRY_SCHEMA_VERSION, target, prefix="manifest.registry.")
        if registry.get("selection_key") != manifest.get("dataset_version"):
            target.errors.append("manifest.registry.selection_key must match manifest.dataset_version.")
        if registry.get("redaction_passed") is not (expected_redaction_status.get("passed") is True):
            target.errors.append("manifest.registry.redaction_passed must match redaction_status.passed.")
        if registry.get("mode_to_view") != reviewed_trainer_views.get("mode_to_view"):
            target.errors.append("manifest.registry.mode_to_view must match manifest.trainer_views.mode_to_view.")
        if registry.get("root_views") != reviewed_trainer_views.get("root_views"):
            target.errors.append("manifest.registry.root_views must match manifest.trainer_views.root_views.")
    if dataset_registry is None:
        target.errors.append("manifest has no validated dataset_registry.json companion.")
    else:
        _validate_reviewed_dataset_registry(
            dataset_registry,
            target,
            manifest,
            export_dir,
            expected_redaction_status,
            expected_label_provenance,
            reviewed_trainer_views,
        )
    _validate_public_review_ref_path(target, "manifest.source_review_export", manifest.get("source_review_export"))
    _validate_public_review_ref_path(target, "manifest.labels_path", manifest.get("labels_path"))
    _validate_public_review_ref_path(target, "manifest.output_dir", manifest.get("output_dir"))
    if isinstance(outputs, dict):
        for output_name, output_path in outputs.items():
            _validate_public_review_ref_path(target, f"manifest.outputs.{output_name}", output_path)
    _validate_public_review_fingerprint_paths(
        manifest.get("source_review_artifacts"),
        target,
        "manifest.source_review_artifacts",
    )
    _validate_public_review_fingerprint_paths(
        {"labels_artifact": manifest.get("labels_artifact")},
        target,
        "manifest",
    )
    _validate_public_review_fingerprint_paths(
        manifest.get("artifact_fingerprints"),
        target,
        "manifest.artifact_fingerprints",
    )
    if isinstance(registry, dict):
        _validate_public_review_ref_path(target, "manifest.registry.path", registry.get("path"))
        _validate_public_review_ref_path(target, "manifest.registry.manifest_path", registry.get("manifest_path"))

def _expected_reviewed_label_provenance(
    labels: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    action_sft: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
) -> dict[str, Any]:
    label_counts = _reviewed_label_counts(labels)
    return {
        "schema_version": RL_LABEL_PROVENANCE_SCHEMA_VERSION,
        "policy": "Completed human labels bound to review_item_sha256 drive reviewed trainer views.",
        "reviewer_identity_assurance": "self_asserted",
        "reviewed_label_count": len(labels),
        "accepted_label_count": label_counts.get("accept", 0),
        "negative_label_count": sum(label_counts.get(label, 0) for label in sorted(TRAINING_NEGATIVE_LABELS)),
        "needs_review_excluded_count": label_counts.get("needs_review", 0),
        "trainer_view_counts": {
            "reviewed_sft": len(sft),
            "reviewed_action_sft": len(action_sft),
            "reviewed_reward_model": len(reward_model),
            "reviewed_preferences": len(preferences),
            "reviewed_dpo": len(dpo),
        },
        "notes": [
            "Reviewed SFT rows require human_label='accept'.",
            "Reviewed reward and preference rows use accept/reject/unsafe/incomplete labels only.",
            "Reviewer identifiers and timestamps are required but self-asserted; this receipt proves content integrity, not reviewer authentication.",
        ],
    }

def _validate_reviewed_trainer_views(
    manifest: dict[str, Any],
    target: ValidationTarget,
    sft: list[dict[str, Any]],
    action_sft: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
) -> dict[str, Any]:
    trainer_views = manifest.get("trainer_views")
    if not isinstance(trainer_views, dict):
        target.errors.append("manifest.trainer_views must be an object.")
        return {}

    expected_mode_to_view = {
        "sft": "reviewed_sft",
        "action_sft": "reviewed_action_sft",
        "dpo": "reviewed_dpo",
        "reward_model": "reviewed_reward_model",
    }
    expected_root_views = [
        "reviewed_sft.jsonl",
        "reviewed_action_sft.jsonl",
        "reviewed_dpo.jsonl",
        "reviewed_reward_model.jsonl",
    ]
    if trainer_views.get("contract_version") != RL_TRAINER_VIEWS_CONTRACT_VERSION:
        target.errors.append("manifest.trainer_views.contract_version must be hfr.rl.trainer_views.v1.")
    if trainer_views.get("mode_to_view") != expected_mode_to_view:
        target.errors.append("manifest.trainer_views.mode_to_view must map reviewed SFT, DPO, and reward-model modes.")
    if trainer_views.get("root_views") != expected_root_views:
        target.errors.append("manifest.trainer_views.root_views must list reviewed SFT, DPO, and reward-model artifacts.")

    views = trainer_views.get("views")
    if not isinstance(views, list):
        target.errors.append("manifest.trainer_views.views must be a list.")
        return trainer_views
    views_by_id = {view.get("view_id"): view for view in views if isinstance(view, dict)}
    expected_views = {
        "reviewed_sft": {
            "training_modes": ["sft"],
            "artifact_path": "reviewed_sft.jsonl",
            "schema_version": REVIEWED_SFT_SCHEMA_VERSION,
            "row_count": len(sft),
        },
        "reviewed_action_sft": {
            "training_modes": ["action_sft"],
            "artifact_path": "reviewed_action_sft.jsonl",
            "schema_version": REVIEWED_ACTION_SFT_SCHEMA_VERSION,
            "row_count": len(action_sft),
        },
        "reviewed_dpo": {
            "training_modes": ["dpo"],
            "artifact_path": "reviewed_dpo.jsonl",
            "schema_version": REVIEWED_DPO_SCHEMA_VERSION,
            "row_count": len(dpo),
        },
        "reviewed_reward_model": {
            "training_modes": ["reward_model"],
            "artifact_path": "reviewed_reward_model.jsonl",
            "schema_version": REVIEWED_REWARD_MODEL_SCHEMA_VERSION,
            "row_count": len(reward_model),
        },
    }
    for view_id, expected in expected_views.items():
        view = views_by_id.get(view_id)
        if not isinstance(view, dict):
            target.errors.append(f"manifest.trainer_views.views missing {view_id!r}.")
            continue
        for field_name, expected_value in expected.items():
            if view.get(field_name) != expected_value:
                target.errors.append(
                    f"manifest.trainer_views.views.{view_id}.{field_name} expected {expected_value!r}, got {view.get(field_name)!r}."
                )
        if view.get("available") is not True:
            target.errors.append(f"manifest.trainer_views.views.{view_id}.available must be true.")
    return trainer_views

def _validate_reviewed_dataset_registry(
    registry: dict[str, Any],
    target: ValidationTarget,
    manifest: dict[str, Any],
    export_dir: Path,
    expected_redaction_status: dict[str, Any],
    expected_label_provenance: dict[str, Any],
    reviewed_trainer_views: dict[str, Any],
) -> None:
    _require_equal(registry, "schema_version", RL_DATASET_REGISTRY_SCHEMA_VERSION, target, prefix="dataset_registry.")
    if registry.get("artifact_type") != "reviewed_export":
        target.errors.append("dataset_registry.artifact_type must be 'reviewed_export'.")
    if registry.get("dataset_version") != manifest.get("dataset_version"):
        target.errors.append("dataset_registry.dataset_version must match manifest.dataset_version.")
    if registry.get("manifest_sha256") != _sha256(export_dir / "manifest.json"):
        target.errors.append("dataset_registry.manifest_sha256 must match manifest.json contents.")
    selection = registry.get("selection")
    if not isinstance(selection, dict):
        target.errors.append("dataset_registry.selection must be an object.")
    elif selection.get("key") != manifest.get("dataset_version"):
        target.errors.append("dataset_registry.selection.key must match manifest.dataset_version.")
    else:
        if selection.get("mode_to_view") != reviewed_trainer_views.get("mode_to_view"):
            target.errors.append("dataset_registry.selection.mode_to_view must match manifest.trainer_views.mode_to_view.")
        if selection.get("root_views") != reviewed_trainer_views.get("root_views"):
            target.errors.append("dataset_registry.selection.root_views must match manifest.trainer_views.root_views.")
    if registry.get("trainer_views") != reviewed_trainer_views:
        target.errors.append("dataset_registry.trainer_views must match manifest.trainer_views.")
    if registry.get("redaction_status") != expected_redaction_status:
        target.errors.append("dataset_registry.redaction_status must match recomputed reviewed redaction scan.")
    if registry.get("label_provenance") != expected_label_provenance:
        target.errors.append("dataset_registry.label_provenance must match recomputed reviewed label provenance.")
    if registry.get("source_review_artifacts") != manifest.get("source_review_artifacts"):
        target.errors.append("dataset_registry.source_review_artifacts must match manifest.source_review_artifacts.")
    if registry.get("labels_artifact") != manifest.get("labels_artifact"):
        target.errors.append("dataset_registry.labels_artifact must match manifest.labels_artifact.")
    if registry.get("artifact_fingerprints") != manifest.get("artifact_fingerprints"):
        target.errors.append("dataset_registry.artifact_fingerprints must match manifest.artifact_fingerprints.")

def _validate_reviewed_labels(labels: list[dict[str, Any]], target: ValidationTarget) -> None:
    seen: set[str] = set()
    for index, row in enumerate(labels):
        _require_equal(row, "schema_version", REVIEWED_LABEL_SCHEMA_VERSION, target, prefix=f"reviewed_labels[{index}].")
        item_id = row.get("review_item_id")
        if not isinstance(item_id, str) or not item_id:
            target.errors.append(f"reviewed_labels[{index}].review_item_id must be a non-empty string.")
            continue
        if item_id in seen:
            target.errors.append(f"reviewed_labels[{index}].review_item_id duplicates {item_id!r}.")
        seen.add(item_id)
        for field_name in ("episode_id", "scenario_id", "task_family", "prompt", "response", "human_label", "source_label_file"):
            if not isinstance(row.get(field_name), str):
                target.errors.append(f"reviewed_labels[{index}].{field_name} must be a string.")
        _validate_public_review_ref_path(target, f"reviewed_labels[{index}].source_label_file", row.get("source_label_file"))
        if not _is_sha256(row.get("review_item_sha256")):
            target.errors.append(f"reviewed_labels[{index}].review_item_sha256 must be a SHA-256 hex string.")
        if not _is_sha256(row.get("source_label_sha256")):
            target.errors.append(f"reviewed_labels[{index}].source_label_sha256 must be a SHA-256 hex string.")
        if row.get("human_label") not in REVIEW_LABELS:
            target.errors.append(f"reviewed_labels[{index}].human_label must be one of {list(REVIEW_LABELS)!r}.")
        if row.get("suggested_human_label") is not None and row.get("suggested_human_label") not in REVIEW_LABELS:
            target.errors.append(f"reviewed_labels[{index}].suggested_human_label must be null or one of {list(REVIEW_LABELS)!r}.")
        if "reviewer_confidence" not in row:
            target.errors.append(f"reviewed_labels[{index}].reviewer_confidence is required.")
        elif row.get("reviewer_confidence") not in REVIEW_CONFIDENCE_LEVELS:
            target.errors.append(
                f"reviewed_labels[{index}].reviewer_confidence must be one of {list(REVIEW_CONFIDENCE_LEVELS)!r}."
            )
        if not isinstance(row.get("reviewer"), str) or not row.get("reviewer", "").strip():
            target.errors.append(f"reviewed_labels[{index}].reviewer must be a non-empty string.")
        if not isinstance(row.get("reviewed_at"), str) or not row.get("reviewed_at", "").strip():
            target.errors.append(f"reviewed_labels[{index}].reviewed_at must be a non-empty string.")
        if not _is_int_between(row.get("score"), 0, 100):
            target.errors.append(f"reviewed_labels[{index}].score must be an integer from 0 to 100.")
        if not isinstance(row.get("reward"), (int, float)):
            target.errors.append(f"reviewed_labels[{index}].reward must be numeric.")
        if not isinstance(row.get("accepted_evidence_refs"), list):
            target.errors.append(f"reviewed_labels[{index}].accepted_evidence_refs must be a list.")
        if not isinstance(row.get("rejected_evidence_refs"), list):
            target.errors.append(f"reviewed_labels[{index}].rejected_evidence_refs must be a list.")
        source_artifacts = row.get("source_artifacts")
        if not isinstance(source_artifacts, dict):
            target.errors.append(f"reviewed_labels[{index}].source_artifacts must be an object.")
        else:
            for artifact_name, artifact_path in source_artifacts.items():
                _validate_public_review_ref_path(
                    target,
                    f"reviewed_labels[{index}].source_artifacts.{artifact_name}",
                    artifact_path,
                )
        if not isinstance(row.get("scorecard"), dict):
            target.errors.append(f"reviewed_labels[{index}].scorecard must be an object.")

def _validate_reviewed_sft(sft: list[dict[str, Any]], target: ValidationTarget, labels: list[dict[str, Any]]) -> None:
    label_map = _reviewed_label_map(labels)
    for index, row in enumerate(sft):
        _require_equal(row, "schema_version", REVIEWED_SFT_SCHEMA_VERSION, target, prefix=f"reviewed_sft[{index}].")
        item = _reviewed_source_label(row, label_map, target, f"reviewed_sft[{index}]")
        if item is not None and item.get("human_label") != "accept":
            target.errors.append(f"reviewed_sft[{index}] must reference a reviewed label with human_label 'accept'.")
        if item is not None:
            _validate_review_item_hash_link(row, item, target, f"reviewed_sft[{index}]")
            _validate_review_confidence_link(row, item, target, f"reviewed_sft[{index}]")
        for field_name in ("prompt", "response", "source_artifact"):
            if not isinstance(row.get(field_name), str):
                target.errors.append(f"reviewed_sft[{index}].{field_name} must be a string.")
        if row.get("source_artifact") != "reviewed_labels.jsonl":
            target.errors.append(f"reviewed_sft[{index}].source_artifact must be 'reviewed_labels.jsonl'.")

def _validate_reviewed_action_sft(
    rows: list[dict[str, Any]],
    target: ValidationTarget,
    labels: list[dict[str, Any]],
) -> None:
    label_map = _reviewed_label_map(labels)
    for index, row in enumerate(rows):
        prefix = f"reviewed_action_sft[{index}]"
        _require_equal(
            row,
            "schema_version",
            REVIEWED_ACTION_SFT_SCHEMA_VERSION,
            target,
            prefix=f"{prefix}.",
        )
        item = _reviewed_source_label(row, label_map, target, prefix)
        if item is not None and item.get("human_label") != "accept":
            target.errors.append(f"{prefix} must reference a reviewed label with human_label 'accept'.")
        if item is not None:
            _validate_review_item_hash_link(row, item, target, prefix)
            _validate_review_confidence_link(row, item, target, prefix)
        messages = row.get("messages")
        tools = row.get("tools")
        if not isinstance(messages, list) or not messages:
            target.errors.append(f"{prefix}.messages must be a non-empty native message list.")
            continue
        if not isinstance(tools, list):
            target.errors.append(f"{prefix}.tools must be a list.")
        expected_contract = task_contract_fingerprint(row)
        if row.get("task_contract_fingerprint") != expected_contract:
            target.errors.append(f"{prefix}.task_contract_fingerprint must match the native task contract.")
        if row.get("quality_gate") != "human_reviewed_native_action_accept":
            target.errors.append(f"{prefix}.quality_gate must be 'human_reviewed_native_action_accept'.")
        if row.get("source_artifact") != "reviewed_labels.jsonl+action_sft.jsonl":
            target.errors.append(
                f"{prefix}.source_artifact must be 'reviewed_labels.jsonl+action_sft.jsonl'."
            )
        call_ids: set[str] = set()
        result_ids: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                target.errors.append(f"{prefix}.messages entries must be objects.")
                continue
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    call_id = call.get("id") if isinstance(call, dict) else None
                    if not isinstance(call_id, str) or not call_id:
                        target.errors.append(f"{prefix} contains a tool call without a non-empty id.")
                    elif call_id in call_ids:
                        target.errors.append(f"{prefix} duplicates tool call id {call_id!r}.")
                    else:
                        call_ids.add(call_id)
            if message.get("role") == "tool":
                result_id = message.get("tool_call_id")
                if not isinstance(result_id, str) or not result_id:
                    target.errors.append(f"{prefix} contains a tool result without tool_call_id.")
                else:
                    result_ids.append(result_id)
        for call_id in sorted(call_ids):
            if result_ids.count(call_id) != 1:
                target.errors.append(f"{prefix} tool call {call_id!r} must have exactly one result.")
        unmatched_results = sorted(set(result_ids) - call_ids)
        if unmatched_results:
            target.errors.append(f"{prefix} contains unmatched tool result ids: {unmatched_results!r}.")
        if call_ids and str(row.get("tool_schema_provenance") or "").startswith("inferred"):
            target.errors.append(f"{prefix} inferred tool schemas are not eligible for action training.")

def _validate_reviewed_reward_model(rows: list[dict[str, Any]], target: ValidationTarget, labels: list[dict[str, Any]]) -> None:
    label_map = _reviewed_label_map(labels)
    for index, row in enumerate(rows):
        _require_equal(row, "schema_version", REVIEWED_REWARD_MODEL_SCHEMA_VERSION, target, prefix=f"reviewed_reward_model[{index}].")
        item = _reviewed_source_label(row, label_map, target, f"reviewed_reward_model[{index}]")
        if item is not None and item.get("human_label") == "needs_review":
            target.errors.append(f"reviewed_reward_model[{index}] must not reference a needs_review label.")
        if item is not None:
            _validate_review_item_hash_link(row, item, target, f"reviewed_reward_model[{index}]")
            _validate_review_confidence_link(row, item, target, f"reviewed_reward_model[{index}]")
        for field_name in ("prompt", "response", "human_label", "source_artifact"):
            if not isinstance(row.get(field_name), str):
                target.errors.append(f"reviewed_reward_model[{index}].{field_name} must be a string.")
        if not _is_int_between(row.get("score"), 0, 100):
            target.errors.append(f"reviewed_reward_model[{index}].score must be an integer from 0 to 100.")
        if not isinstance(row.get("reward"), (int, float)):
            target.errors.append(f"reviewed_reward_model[{index}].reward must be numeric.")
        if row.get("source_artifact") != "reviewed_labels.jsonl":
            target.errors.append(f"reviewed_reward_model[{index}].source_artifact must be 'reviewed_labels.jsonl'.")

def _validate_reviewed_preferences(preferences: list[dict[str, Any]], target: ValidationTarget, labels: list[dict[str, Any]]) -> None:
    label_by_episode = _reviewed_label_by_episode(labels)
    seen: set[str] = set()
    for index, row in enumerate(preferences):
        _require_equal(row, "schema_version", REVIEWED_PREFERENCE_SCHEMA_VERSION, target, prefix=f"reviewed_preferences[{index}].")
        preference_id = row.get("preference_id")
        if not isinstance(preference_id, str) or not preference_id:
            target.errors.append(f"reviewed_preferences[{index}].preference_id must be a non-empty string.")
        elif preference_id in seen:
            target.errors.append(f"reviewed_preferences[{index}].preference_id duplicates {preference_id!r}.")
        else:
            seen.add(preference_id)
        chosen = label_by_episode.get(row.get("chosen_episode_id"))
        rejected = label_by_episode.get(row.get("rejected_episode_id"))
        if chosen is None:
            target.errors.append(f"reviewed_preferences[{index}].chosen_episode_id does not reference a reviewed label.")
        elif chosen.get("human_label") != "accept":
            target.errors.append(f"reviewed_preferences[{index}].chosen_episode_id must reference an accepted label.")
        else:
            _validate_pref_side_hash(row, chosen, "chosen", target, f"reviewed_preferences[{index}]")
            _validate_pref_side_confidence(row, chosen, "chosen", target, f"reviewed_preferences[{index}]")
        if rejected is None:
            target.errors.append(f"reviewed_preferences[{index}].rejected_episode_id does not reference a reviewed label.")
        elif rejected.get("human_label") not in {"reject", "unsafe", "incomplete"}:
            target.errors.append(f"reviewed_preferences[{index}].rejected_episode_id must reference a rejected/unsafe/incomplete label.")
        else:
            _validate_pref_side_hash(row, rejected, "rejected", target, f"reviewed_preferences[{index}]")
            _validate_pref_side_confidence(row, rejected, "rejected", target, f"reviewed_preferences[{index}]")
        if row.get("source_artifact") != "reviewed_labels.jsonl":
            target.errors.append(f"reviewed_preferences[{index}].source_artifact must be 'reviewed_labels.jsonl'.")

def _validate_reviewed_dpo(dpo: list[dict[str, Any]], target: ValidationTarget, preferences: list[dict[str, Any]]) -> None:
    preference_by_id = {
        row.get("preference_id"): row
        for row in preferences
        if isinstance(row.get("preference_id"), str)
    }
    for index, row in enumerate(dpo):
        _require_equal(row, "schema_version", REVIEWED_DPO_SCHEMA_VERSION, target, prefix=f"reviewed_dpo[{index}].")
        preference = preference_by_id.get(row.get("preference_id"))
        if preference is None:
            target.errors.append(f"reviewed_dpo[{index}].preference_id does not reference a reviewed preference.")
        for side in ("chosen", "rejected"):
            field_name = f"{side}_review_item_sha256"
            if not _is_sha256(row.get(field_name)):
                target.errors.append(f"reviewed_dpo[{index}].{field_name} must be a SHA-256 hex string.")
            elif preference is not None:
                if row.get(field_name) != preference.get(field_name):
                    target.errors.append(f"reviewed_dpo[{index}].{field_name} does not match reviewed preference.")
            confidence_field = f"{side}_reviewer_confidence"
            if confidence_field not in row:
                target.errors.append(f"reviewed_dpo[{index}].{confidence_field} is required.")
            else:
                _validate_confidence_value(row.get(confidence_field), target, f"reviewed_dpo[{index}].{confidence_field}")
                if preference is not None and row.get(confidence_field) != preference.get(confidence_field):
                    target.errors.append(f"reviewed_dpo[{index}].{confidence_field} does not match reviewed preference.")
        for field_name in ("prompt", "chosen", "rejected", "source_artifact"):
            if not isinstance(row.get(field_name), str):
                target.errors.append(f"reviewed_dpo[{index}].{field_name} must be a string.")
        if row.get("source_artifact") != "reviewed_preferences.jsonl":
            target.errors.append(f"reviewed_dpo[{index}].source_artifact must be 'reviewed_preferences.jsonl'.")

def _reviewed_source_label(
    row: dict[str, Any],
    label_map: dict[str, dict[str, Any]],
    target: ValidationTarget,
    label: str,
) -> dict[str, Any] | None:
    item_id = row.get("review_item_id")
    if not isinstance(item_id, str) or not item_id:
        target.errors.append(f"{label}.review_item_id must be a non-empty string.")
        return None
    item = label_map.get(item_id)
    if item is None:
        target.errors.append(f"{label}.review_item_id does not reference a reviewed label.")
    return item

def _validate_review_item_hash_link(
    row: dict[str, Any],
    item: dict[str, Any],
    target: ValidationTarget,
    label: str,
) -> None:
    if not _is_sha256(row.get("review_item_sha256")):
        target.errors.append(f"{label}.review_item_sha256 must be a SHA-256 hex string.")
    elif row.get("review_item_sha256") != item.get("review_item_sha256"):
        target.errors.append(f"{label}.review_item_sha256 does not match reviewed label.")

def _validate_review_confidence_link(
    row: dict[str, Any],
    item: dict[str, Any],
    target: ValidationTarget,
    label: str,
) -> None:
    if "reviewer_confidence" not in row:
        target.errors.append(f"{label}.reviewer_confidence is required.")
        return
    _validate_confidence_value(row.get("reviewer_confidence"), target, f"{label}.reviewer_confidence")
    expected = item.get("reviewer_confidence", "unknown")
    if row.get("reviewer_confidence") != expected:
        target.errors.append(f"{label}.reviewer_confidence does not match reviewed label.")

def _validate_pref_side_hash(
    row: dict[str, Any],
    item: dict[str, Any],
    side: str,
    target: ValidationTarget,
    label: str,
) -> None:
    field_name = f"{side}_review_item_sha256"
    if not _is_sha256(row.get(field_name)):
        target.errors.append(f"{label}.{field_name} must be a SHA-256 hex string.")
    elif row.get(field_name) != item.get("review_item_sha256"):
        target.errors.append(f"{label}.{field_name} does not match reviewed label.")
    nested = row.get(side)
    if isinstance(nested, dict):
        nested_hash = nested.get("review_item_sha256")
        if not _is_sha256(nested_hash):
            target.errors.append(f"{label}.{side}.review_item_sha256 must be a SHA-256 hex string.")
        elif nested_hash != item.get("review_item_sha256"):
            target.errors.append(f"{label}.{side}.review_item_sha256 does not match reviewed label.")

def _validate_pref_side_confidence(
    row: dict[str, Any],
    item: dict[str, Any],
    side: str,
    target: ValidationTarget,
    label: str,
) -> None:
    expected = item.get("reviewer_confidence", "unknown")
    field_name = f"{side}_reviewer_confidence"
    if field_name not in row:
        target.errors.append(f"{label}.{field_name} is required.")
    else:
        _validate_confidence_value(row.get(field_name), target, f"{label}.{field_name}")
        if row.get(field_name) != expected:
            target.errors.append(f"{label}.{field_name} does not match reviewed label.")
    nested = row.get(side)
    if isinstance(nested, dict):
        if "reviewer_confidence" not in nested:
            target.errors.append(f"{label}.{side}.reviewer_confidence is required.")
        else:
            _validate_confidence_value(nested.get("reviewer_confidence"), target, f"{label}.{side}.reviewer_confidence")
            if nested.get("reviewer_confidence") != expected:
                target.errors.append(f"{label}.{side}.reviewer_confidence does not match reviewed label.")

def _validate_confidence_value(value: Any, target: ValidationTarget, label: str) -> None:
    if value not in REVIEW_CONFIDENCE_LEVELS:
        target.errors.append(f"{label} must be one of {list(REVIEW_CONFIDENCE_LEVELS)!r}.")

def _reviewed_label_map(labels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["review_item_id"]): row
        for row in labels
        if isinstance(row.get("review_item_id"), str)
    }

def _reviewed_label_by_episode(labels: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    return {row.get("episode_id"): row for row in labels if row.get("episode_id") is not None}

def _reviewed_label_counts(labels: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in labels:
        label = row.get("human_label")
        if isinstance(label, str):
            counts[label] = counts.get(label, 0) + 1
    return counts

def _reviewed_confidence_counts(labels: list[dict[str, Any]]) -> dict[str, int]:
    counts = {level: 0 for level in REVIEW_CONFIDENCE_LEVELS}
    for row in labels:
        confidence = row.get("reviewer_confidence")
        if confidence not in REVIEW_CONFIDENCE_LEVELS:
            confidence = "unknown"
        counts[str(confidence)] += 1
    return counts

_REVIEW_CALIBRATION_KEYS = {
    "schema_version",
    "reviewed_export",
    "source",
    "source_artifacts",
    "effective_policy",
    "passed",
    "check_count",
    "failed_check_count",
    "checks",
    "metrics",
    "disagreements",
    "notes",
}

_REVIEW_CALIBRATION_EFFECTIVE_POLICY_KEYS = {
    "min_agreement_rate",
    "max_disagreements",
    "max_false_positives",
    "max_false_negatives",
    "min_comparable_labels",
    "require_valid_export",
    "strict_validation",
}

_REVIEW_CALIBRATION_SOURCE_KEYS = {"reviewed_labels"}

_REVIEW_CALIBRATION_CHECK_KEYS = {"id", "passed", "actual", "expected", "summary"}

_REVIEW_CALIBRATION_MIN_CHECK_IDS = {"min_comparable_labels", "min_agreement_rate"}

_REVIEW_CALIBRATION_MAX_CHECK_IDS = {"max_disagreements", "max_false_positives", "max_false_negatives"}

_REVIEW_CALIBRATION_EXPECTED_MIN_KEYS = {"min"}

_REVIEW_CALIBRATION_EXPECTED_MAX_KEYS = {"max"}

_REVIEW_CALIBRATION_EXPECTED_VALIDATION_KEYS = {"passed", "error_count"}

_REVIEW_CALIBRATION_SOURCE_PATH_ACTUAL_KEYS = {"reviewed_export", "reviewed_labels"}

_REVIEW_CALIBRATION_SOURCE_PATH_EXPECTED_KEYS = {"safe_relative_paths"}

_REVIEW_CALIBRATION_METRICS_KEYS = {
    "reviewed_label_count",
    "comparable_label_count",
    "needs_review_count",
    "agreement_count",
    "disagreement_count",
    "agreement_rate",
    "scorecard_positive_count",
    "scorecard_negative_count",
    "human_positive_count",
    "human_negative_count",
    "false_positive_count",
    "false_negative_count",
    "label_counts",
    "mean_score_by_human_label",
    "task_families",
    "validation",
}

_REVIEW_CALIBRATION_DISAGREEMENT_KEYS = {
    "review_item_id",
    "episode_id",
    "scenario_id",
    "task_family",
    "human_label",
    "scorecard_passed",
    "scorecard_score",
    "failed_rules",
    "critical_failures",
    "source_report",
    "source_lineage",
    "disagreement_type",
}

_REVIEW_CALIBRATION_LABEL_COUNT_KEYS = {"label", "count"}

_REVIEW_CALIBRATION_MEAN_SCORE_KEYS = {"label", "count", "average_score"}

_REVIEW_CALIBRATION_VALIDATION_KEYS = {"available", "passed", "strict", "target_count", "error_count", "warning_count"}

def _validate_review_calibration(calibration: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    from .dispatch import validate_artifacts
    _validate_allowed_keys(calibration, _REVIEW_CALIBRATION_KEYS, target, "review_calibration")
    _require_equal(calibration, "schema_version", REVIEW_CALIBRATION_SCHEMA_VERSION, target)
    export_path: Path | None = None
    if not isinstance(calibration.get("reviewed_export"), str) or not calibration.get("reviewed_export"):
        target.errors.append("review_calibration.reviewed_export must be a non-empty string.")
    elif not _is_public_review_calibration_ref_path(calibration.get("reviewed_export")):
        target.errors.append("review_calibration.reviewed_export must be a safe relative path.")
    else:
        export_path = _validate_review_calibration_export_ref(
            calibration.get("reviewed_export"),
            target,
            "review_calibration.reviewed_export",
            source_path,
        )
    source = calibration.get("source")
    if not isinstance(source, dict):
        target.errors.append("review_calibration.source must be an object.")
    else:
        _validate_allowed_keys(source, _REVIEW_CALIBRATION_SOURCE_KEYS, target, "review_calibration.source")
        if not isinstance(source.get("reviewed_labels"), str) or not source.get("reviewed_labels"):
            target.errors.append("review_calibration.source.reviewed_labels must be a non-empty string.")
        elif not _is_public_review_calibration_ref_path(source.get("reviewed_labels")):
            target.errors.append("review_calibration.source.reviewed_labels must be a safe relative path.")
        else:
            _validate_review_calibration_label_ref(
                source.get("reviewed_labels"),
                target,
                "review_calibration.source.reviewed_labels",
                source_path,
            )
    if not isinstance(calibration.get("passed"), bool):
        target.errors.append("review_calibration.passed must be a boolean.")

    checks = calibration.get("checks")
    if not isinstance(checks, list):
        target.errors.append("review_calibration.checks must be a list.")
        checks = []
    failed_checks = _validate_gate_like_checks(checks, target, "review_calibration.checks")
    for index, check in enumerate(checks):
        if isinstance(check, dict):
            check_label = f"review_calibration.checks[{index}]"
            _validate_allowed_keys(check, _REVIEW_CALIBRATION_CHECK_KEYS, target, check_label)
            _validate_review_calibration_check_payload(check, target, check_label)
    if calibration.get("check_count") != len(checks):
        target.errors.append(f"review_calibration.check_count expected {len(checks)}, got {calibration.get('check_count')!r}.")
    if calibration.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"review_calibration.failed_check_count expected {failed_checks}, got {calibration.get('failed_check_count')!r}."
        )
    if isinstance(calibration.get("passed"), bool) and calibration["passed"] != (failed_checks == 0):
        target.errors.append("review_calibration.passed must match failed_check_count.")

    metrics = calibration.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("review_calibration.metrics must be an object.")
        metrics = {}
    else:
        _validate_allowed_keys(metrics, _REVIEW_CALIBRATION_METRICS_KEYS, target, "review_calibration.metrics")
    disagreements = calibration.get("disagreements")
    if not isinstance(disagreements, list):
        target.errors.append("review_calibration.disagreements must be a list.")
        disagreements = []
    disagreement_counts = _validate_review_calibration_disagreements(disagreements, target)
    _validate_review_calibration_metrics(metrics, disagreement_counts, target)
    notes = calibration.get("notes")
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        target.errors.append("review_calibration.notes must be a list of strings.")

    source_artifacts = calibration.get("source_artifacts")
    source_record = source_artifacts.get("reviewed_export") if isinstance(source_artifacts, dict) else None
    if not isinstance(source_artifacts, dict) or set(source_artifacts) != {"reviewed_export"}:
        target.errors.append("review_calibration.source_artifacts must contain only reviewed_export.")
    if not isinstance(source_record, dict):
        target.errors.append("review_calibration.source_artifacts.reviewed_export must be an object.")

    effective_policy = calibration.get("effective_policy")
    if not isinstance(effective_policy, dict):
        target.errors.append("review_calibration.effective_policy must be an object.")
    else:
        _validate_allowed_keys(
            effective_policy,
            _REVIEW_CALIBRATION_EFFECTIVE_POLICY_KEYS,
            target,
            "review_calibration.effective_policy",
        )
        missing = sorted(_REVIEW_CALIBRATION_EFFECTIVE_POLICY_KEYS - set(effective_policy))
        if missing:
            target.errors.append(
                "review_calibration.effective_policy is missing field(s): " + ", ".join(missing) + "."
            )
        if effective_policy.get("require_valid_export") is not True:
            target.errors.append(
                "review_calibration.effective_policy.require_valid_export must be true before calibration can authorize downstream work."
            )

    if (
        isinstance(export_path, Path)
        and isinstance(source_record, dict)
        and isinstance(effective_policy, dict)
        and not (_REVIEW_CALIBRATION_EFFECTIVE_POLICY_KEYS - set(effective_policy))
    ):
        try:
            source_before_validation = build_reviewed_export_source_artifact(
                export_path,
                display_path=str(calibration.get("reviewed_export")),
            )
        except (OSError, ReviewedGateError, ValueError) as exc:
            target.errors.append(f"review_calibration reviewed export could not be fingerprinted: {exc}")
        else:
            if source_record != source_before_validation:
                target.errors.append(
                    "review_calibration.source_artifacts.reviewed_export must match the current reviewed-export tree, manifest, and dataset version."
                )
            current_validation = validate_artifacts(
                reviewed_export_dir=export_path,
                strict=effective_policy.get("strict_validation") is True,
            )
            try:
                source_after_validation = build_reviewed_export_source_artifact(
                    export_path,
                    display_path=str(calibration.get("reviewed_export")),
                )
            except (OSError, ReviewedGateError, ValueError) as exc:
                target.errors.append(
                    f"review_calibration reviewed export could not be reattested after validation: {exc}"
                )
            else:
                if source_after_validation != source_before_validation:
                    target.errors.append(
                        "review_calibration reviewed export changed while its current source was being validated."
                    )
                else:
                    try:
                        replayed = build_review_calibration(
                            export_path,
                            min_agreement_rate=effective_policy.get("min_agreement_rate"),
                            max_disagreements=effective_policy.get("max_disagreements"),
                            max_false_positives=effective_policy.get("max_false_positives"),
                            max_false_negatives=effective_policy.get("max_false_negatives"),
                            min_comparable_labels=effective_policy.get("min_comparable_labels"),
                            validation_summary=(
                                current_validation if effective_policy.get("require_valid_export") is True else None
                            ),
                            require_valid_export=effective_policy.get("require_valid_export") is True,
                            strict_validation=effective_policy.get("strict_validation") is True,
                            preserve_paths=False,
                            out_path=source_path,
                        )
                        source_after_replay = build_reviewed_export_source_artifact(
                            export_path,
                            display_path=str(calibration.get("reviewed_export")),
                        )
                    except (OSError, ReviewedGateError, TypeError, ValueError) as exc:
                        target.errors.append(f"review_calibration could not replay its current source and policy: {exc}")
                    else:
                        if source_after_replay != source_before_validation:
                            target.errors.append(
                                "review_calibration reviewed export changed while its policy was being replayed."
                            )
                        elif calibration != replayed:
                            target.errors.append(
                                "review_calibration must match deterministic replay of its current source and effective policy exactly."
                            )
    target.details.update(
        {
            "reviewed_label_count": metrics.get("reviewed_label_count"),
            "agreement_rate": metrics.get("agreement_rate"),
            "disagreement_count": metrics.get("disagreement_count"),
        }
    )

def _validate_review_calibration_export_ref(
    value: str,
    target: ValidationTarget,
    label: str,
    source_path: Path,
) -> Path | None:
    export_path = source_path.parent / value
    if _path_has_symlink_component(export_path, include_leaf=True):
        target.errors.append(f"{label} must resolve to a regular non-symlink directory.")
        return None
    if not export_path.exists() or not export_path.is_dir():
        target.errors.append(f"{label} must resolve to an existing directory.")
        return None
    return export_path

def _validate_review_calibration_label_ref(value: str, target: ValidationTarget, label: str, source_path: Path) -> None:
    label_path = source_path.parent / value
    if _path_has_symlink_component(label_path, include_leaf=True):
        target.errors.append(f"{label} must resolve to a regular non-symlink file.")
        return
    if not label_path.exists() or not label_path.is_file():
        target.errors.append(f"{label} must resolve to an existing file.")

def _is_public_review_calibration_ref_path(value: str) -> bool:
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return (
        bool(value)
        and not value.startswith("<redacted:")
        and not path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and "\\" not in value
        and ".." not in path.parts
        and all(not part.startswith("~") for part in path.parts)
    )

def _validate_review_calibration_check_payload(check: dict[str, Any], target: ValidationTarget, label: str) -> None:
    check_id = check.get("id")
    expected = check.get("expected")
    if check_id == "valid_reviewed_export":
        _validate_review_calibration_validation_metrics(check.get("actual"), target, f"{label}.actual")
        if not isinstance(expected, dict):
            return
        _validate_allowed_keys(expected, _REVIEW_CALIBRATION_EXPECTED_VALIDATION_KEYS, target, f"{label}.expected")
        if expected.get("passed") is not True:
            target.errors.append(f"{label}.expected.passed must be true.")
        if expected.get("error_count") != 0:
            target.errors.append(f"{label}.expected.error_count must be 0.")
        return
    if check_id == "source_paths_replayable":
        actual = check.get("actual")
        if not isinstance(actual, dict):
            target.errors.append(f"{label}.actual must be an object.")
        else:
            _validate_allowed_keys(actual, _REVIEW_CALIBRATION_SOURCE_PATH_ACTUAL_KEYS, target, f"{label}.actual")
            for field_name in ("reviewed_export", "reviewed_labels"):
                if not isinstance(actual.get(field_name), str) or not actual.get(field_name):
                    target.errors.append(f"{label}.actual.{field_name} must be a non-empty string.")
        if not isinstance(expected, dict):
            return
        _validate_allowed_keys(expected, _REVIEW_CALIBRATION_SOURCE_PATH_EXPECTED_KEYS, target, f"{label}.expected")
        if expected.get("safe_relative_paths") is not True:
            target.errors.append(f"{label}.expected.safe_relative_paths must be true.")
        return
    if check_id in _REVIEW_CALIBRATION_MIN_CHECK_IDS:
        if not _is_number_between(check.get("actual"), 0.0, float("inf")):
            target.errors.append(f"{label}.actual must be a non-negative number.")
        if not isinstance(expected, dict):
            return
        _validate_allowed_keys(expected, _REVIEW_CALIBRATION_EXPECTED_MIN_KEYS, target, f"{label}.expected")
        if not _is_number_between(expected.get("min"), 0.0, float("inf")):
            target.errors.append(f"{label}.expected.min must be a non-negative number.")
        return
    if check_id in _REVIEW_CALIBRATION_MAX_CHECK_IDS:
        if not _is_non_negative_int(check.get("actual")):
            target.errors.append(f"{label}.actual must be a non-negative integer.")
        if not isinstance(expected, dict):
            return
        _validate_allowed_keys(expected, _REVIEW_CALIBRATION_EXPECTED_MAX_KEYS, target, f"{label}.expected")
        if not _is_non_negative_int(expected.get("max")):
            target.errors.append(f"{label}.expected.max must be a non-negative integer.")
        return
    allowed_ids = sorted(_REVIEW_CALIBRATION_MIN_CHECK_IDS | _REVIEW_CALIBRATION_MAX_CHECK_IDS | {"source_paths_replayable", "valid_reviewed_export"})
    target.errors.append(f"{label}.id must be one of {allowed_ids!r}.")

def _validate_review_calibration_disagreements(disagreements: list[Any], target: ValidationTarget) -> dict[str, int]:
    counts = {"false_positive_count": 0, "false_negative_count": 0}
    for index, row in enumerate(disagreements):
        label = f"review_calibration.disagreements[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(row, _REVIEW_CALIBRATION_DISAGREEMENT_KEYS, target, label)
        for field_name in ("review_item_id", "episode_id", "scenario_id", "task_family", "human_label", "disagreement_type"):
            if not isinstance(row.get(field_name), str) or not row.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if not isinstance(row.get("scorecard_passed"), bool):
            target.errors.append(f"{label}.scorecard_passed must be a boolean.")
        if not _is_int_between(row.get("scorecard_score"), 0, 100):
            target.errors.append(f"{label}.scorecard_score must be an integer from 0 to 100.")
        if not _is_string_list(row.get("failed_rules")):
            target.errors.append(f"{label}.failed_rules must be a list of strings.")
        if not _is_string_list(row.get("critical_failures")):
            target.errors.append(f"{label}.critical_failures must be a list of strings.")
        if row.get("source_report") is not None and not isinstance(row.get("source_report"), str):
            target.errors.append(f"{label}.source_report must be a string or null.")
        if row.get("source_lineage") is not None and not isinstance(row.get("source_lineage"), str):
            target.errors.append(f"{label}.source_lineage must be a string or null.")
        if row.get("disagreement_type") == "scorecard_passed_human_rejected":
            counts["false_positive_count"] += 1
            if row.get("scorecard_passed") is not True:
                target.errors.append(f"{label}.scorecard_passed must be true for scorecard_passed_human_rejected.")
            if row.get("human_label") not in TRAINING_NEGATIVE_LABELS:
                target.errors.append(f"{label}.human_label must be a negative label for scorecard_passed_human_rejected.")
        elif row.get("disagreement_type") == "scorecard_failed_human_accepted":
            counts["false_negative_count"] += 1
            if row.get("scorecard_passed") is not False:
                target.errors.append(f"{label}.scorecard_passed must be false for scorecard_failed_human_accepted.")
            if row.get("human_label") != "accept":
                target.errors.append(f"{label}.human_label must be accept for scorecard_failed_human_accepted.")
        else:
            target.errors.append(
                f"{label}.disagreement_type must be scorecard_passed_human_rejected or scorecard_failed_human_accepted."
            )
    counts["disagreement_count"] = len(disagreements)
    return counts

def _validate_review_calibration_metrics(metrics: dict[str, Any], disagreement_counts: dict[str, int], target: ValidationTarget) -> None:
    label_counts = _label_count_rows(metrics.get("label_counts"), target)
    comparable_count = label_counts.get("accept", 0) + sum(label_counts.get(label, 0) for label in TRAINING_NEGATIVE_LABELS)
    expected = {
        "reviewed_label_count": sum(label_counts.values()),
        "comparable_label_count": comparable_count,
        "needs_review_count": label_counts.get("needs_review", 0),
        "human_positive_count": label_counts.get("accept", 0),
        "human_negative_count": sum(label_counts.get(label, 0) for label in TRAINING_NEGATIVE_LABELS),
        "disagreement_count": disagreement_counts["disagreement_count"],
        "false_positive_count": disagreement_counts["false_positive_count"],
        "false_negative_count": disagreement_counts["false_negative_count"],
    }
    expected["agreement_count"] = comparable_count - disagreement_counts["disagreement_count"]
    for field_name, expected_value in expected.items():
        if metrics.get(field_name) != expected_value:
            target.errors.append(f"review_calibration.metrics.{field_name} expected {expected_value!r}, got {metrics.get(field_name)!r}.")
    if metrics.get("agreement_rate") != _rate_value(expected["agreement_count"], comparable_count):
        target.errors.append("review_calibration.metrics.agreement_rate does not match agreement/comparable counts.")
    for field_name in ("scorecard_positive_count", "scorecard_negative_count"):
        if not _is_non_negative_int(metrics.get(field_name)):
            target.errors.append(f"review_calibration.metrics.{field_name} must be a non-negative integer.")
    if _is_non_negative_int(metrics.get("scorecard_positive_count")) and _is_non_negative_int(metrics.get("scorecard_negative_count")):
        actual_comparable = metrics["scorecard_positive_count"] + metrics["scorecard_negative_count"]
        if actual_comparable != comparable_count:
            target.errors.append("review_calibration.metrics scorecard positive/negative counts must sum to comparable_label_count.")
    if not _is_string_list(metrics.get("task_families")):
        target.errors.append("review_calibration.metrics.task_families must be a list of strings.")
    if "validation" in metrics:
        _validate_review_calibration_validation_metrics(metrics.get("validation"), target)
    _validate_mean_score_by_human_label(metrics.get("mean_score_by_human_label"), label_counts, target)

def _validate_review_calibration_validation_metrics(
    value: Any,
    target: ValidationTarget,
    label: str = "review_calibration.metrics.validation",
) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object when present.")
        return
    for field_name in ("available", "passed", "strict"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    _validate_allowed_keys(value, _REVIEW_CALIBRATION_VALIDATION_KEYS, target, label)
    for field_name in ("target_count", "error_count", "warning_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")

def _label_count_rows(value: Any, target: ValidationTarget) -> dict[str, int]:
    labels = set(REVIEW_LABELS)
    counts = {label: 0 for label in REVIEW_LABELS}
    if not isinstance(value, list):
        target.errors.append("review_calibration.metrics.label_counts must be a list.")
        return counts
    seen: set[str] = set()
    for index, row in enumerate(value):
        label = f"review_calibration.metrics.label_counts[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(row, _REVIEW_CALIBRATION_LABEL_COUNT_KEYS, target, label)
        row_label = row.get("label")
        count = row.get("count")
        if not isinstance(row_label, str) or row_label not in labels:
            target.errors.append(f"{label}.label must be one of {list(REVIEW_LABELS)!r}.")
            continue
        if row_label in seen:
            target.errors.append(f"{label}.label duplicates {row_label!r}.")
        seen.add(row_label)
        if not _is_non_negative_int(count):
            target.errors.append(f"{label}.count must be a non-negative integer.")
            continue
        counts[row_label] = count
    return counts

def _validate_mean_score_by_human_label(value: Any, label_counts: dict[str, int], target: ValidationTarget) -> None:
    if not isinstance(value, list):
        target.errors.append("review_calibration.metrics.mean_score_by_human_label must be a list.")
        return
    seen: set[str] = set()
    for index, row in enumerate(value):
        label = f"review_calibration.metrics.mean_score_by_human_label[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(row, _REVIEW_CALIBRATION_MEAN_SCORE_KEYS, target, label)
        row_label = row.get("label")
        if not isinstance(row_label, str) or not row_label:
            target.errors.append(f"{label}.label must be a non-empty string.")
            continue
        if row_label in seen:
            target.errors.append(f"{label}.label duplicates {row_label!r}.")
        seen.add(row_label)
        if row_label in label_counts and row.get("count") != label_counts[row_label]:
            target.errors.append(f"{label}.count must match label_counts for {row_label!r}.")
        elif not _is_non_negative_int(row.get("count")):
            target.errors.append(f"{label}.count must be a non-negative integer.")
        if not _is_number_between(row.get("average_score"), 0.0, 100.0):
            target.errors.append(f"{label}.average_score must be a number from 0 to 100.")
