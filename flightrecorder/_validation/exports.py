"""Extracted validation implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PureWindowsPath
from typing import Any
from ..artifacts import CONTRACT_SCOPES, SUITE_TREND_SCHEMA_VERSION
from ..compare_gate import compare_movement_summary
from ..trace_observability import TRACE_OBSERVABILITY_SCHEMA_VERSION, build_trace_signal
from ..training import DATASET_SPLIT_ARTIFACTS, DATASET_SPLIT_NAMES, RL_CURRICULUM_SCHEMA_VERSION, RL_ACTION_SFT_SCHEMA_VERSION, RL_DATASET_REGISTRY_SCHEMA_VERSION, RL_DATASET_METRICS_SCHEMA_VERSION, RL_DATASET_SPLITS_SCHEMA_VERSION, RL_DPO_SCHEMA_VERSION, RL_EPISODE_SCHEMA_VERSION, RL_FAILURE_MODE_SCHEMA_VERSION, RL_LABEL_PROVENANCE_SCHEMA_VERSION, COMPARE_RL_DPO_SCHEMA_VERSION, COMPARE_RL_MANIFEST_SCHEMA_VERSION, COMPARE_RL_PAIR_SCHEMA_VERSION, RL_MANIFEST_SCHEMA_VERSION, RL_PREFERENCE_SCHEMA_VERSION, RL_REWARD_SCHEMA_VERSION, RL_REWARD_MODEL_SCHEMA_VERSION, RL_SFT_SCHEMA_VERSION, RL_STEP_REWARD_SCHEMA_VERSION, RL_TRAINER_VIEWS_CONTRACT_VERSION, REWARD_SCALES, build_label_provenance_summary, build_redaction_status, positive_label_eligible, redaction_scan_artifacts
from ..hashing import sha256_file as _sha256
from .primitives import ValidationTarget, _average_number, _count_family, _count_rows, _count_strings, _is_dataset_version, _is_int_between, _is_non_negative_int, _is_plain_int, _is_sha256, _is_string_list, _looks_absolute, _non_negative_int_value, _outcome_strings, _path_resolves_inside, _read_jsonl_objects, _read_jsonl_objects_optional, _read_object, _read_object_optional, _reject_symlinked_validation_path, _require_equal, _score_value, _sha256, _validate_evidence_refs, _validate_metadata
from .runs import _validate_state_diff_summary, _validate_task_completion

def validate_training_export(path: str | Path) -> ValidationTarget:
    """Validate an RL/training export directory."""
    export_dir = Path(path)
    target = ValidationTarget("training_export", str(export_dir))
    if _reject_symlinked_validation_path(export_dir, target, "Training export path", "directory"):
        return target
    if not export_dir.exists():
        target.errors.append(f"Training export directory not found: {export_dir}")
        return target
    if not export_dir.is_dir():
        target.errors.append(f"Training export path is not a directory: {export_dir}")
        return target

    manifest_path = export_dir / "manifest.json"
    manifest = (
        None
        if _reject_symlinked_validation_path(manifest_path, target, "manifest.json", "file")
        else _read_object(manifest_path, target, "manifest.json")
    )
    episodes = _read_jsonl_objects(export_dir / "episodes.jsonl", target, "episodes.jsonl")
    rewards = _read_jsonl_objects(export_dir / "rewards.jsonl", target, "rewards.jsonl")
    step_rewards = _read_jsonl_objects_optional(export_dir / "step_rewards.jsonl", target, "step_rewards.jsonl")
    preferences = _read_jsonl_objects(export_dir / "preferences.jsonl", target, "preferences.jsonl")
    failure_modes = _read_jsonl_objects(export_dir / "failure_modes.jsonl", target, "failure_modes.jsonl")
    curriculum = _read_object(export_dir / "curriculum.json", target, "curriculum.json")
    sft_path = export_dir / "sft.jsonl"
    action_sft_path = export_dir / "action_sft.jsonl"
    dpo_path = export_dir / "dpo.jsonl"
    reward_model_path = export_dir / "reward_model.jsonl"
    dataset_metrics_path = export_dir / "dataset_metrics.json"
    dataset_splits_path = export_dir / "dataset_splits.json"
    dataset_registry_path = export_dir / "dataset_registry.json"
    dataset_card_path = export_dir / "DATASET_CARD.md"
    sft = _read_jsonl_objects_optional(sft_path, target, "sft.jsonl", "rerun export-rl to emit trainer-ready SFT rows")
    action_sft = (
        _read_jsonl_objects_optional(
            action_sft_path,
            target,
            "action_sft.jsonl",
            "rerun export-rl to emit native tool-trajectory SFT rows",
        )
        if action_sft_path.exists()
        else []
    )
    dpo = _read_jsonl_objects_optional(dpo_path, target, "dpo.jsonl", "rerun export-rl to emit trainer-ready DPO rows")
    reward_model = _read_jsonl_objects_optional(
        reward_model_path,
        target,
        "reward_model.jsonl",
        "rerun export-rl to emit trainer-ready reward-model rows",
    )
    dataset_metrics = _read_object_optional(
        dataset_metrics_path,
        target,
        "dataset_metrics.json",
        "rerun export-rl to emit dataset-level metrics",
    )
    dataset_splits = _read_object_optional(
        dataset_splits_path,
        target,
        "dataset_splits.json",
        "rerun export-rl to emit deterministic train/validation/test split metadata",
    )
    dataset_registry = _read_object_optional(
        dataset_registry_path,
        target,
        "dataset_registry.json",
        "rerun export-rl to emit a selectable dataset registry",
    )
    split_rows = _read_training_split_rows(export_dir, target)
    rows_by_artifact = {
        "episodes": episodes,
        "rewards": rewards,
        "step_rewards": step_rewards,
        "preferences": preferences,
        "failure_modes": failure_modes,
        "sft": sft,
        "action_sft": action_sft,
        "dpo": dpo,
        "reward_model": reward_model,
    }
    metadata = manifest.get("metadata") if isinstance(manifest, dict) and isinstance(manifest.get("metadata"), dict) else None
    expected_redaction_status = build_redaction_status(
        redaction_scan_artifacts(
            rows_by_artifact,
            curriculum,
            metadata=metadata,
            extra_artifacts={
                "manifest": manifest or {},
                "dataset_metrics": dataset_metrics or {},
                "dataset_registry": dataset_registry or {},
            },
        )
    )
    expected_label_provenance = build_label_provenance_summary(
        episodes,
        sft,
        dpo,
        reward_model,
        action_sft if action_sft_path.exists() else None,
    )
    if expected_redaction_status.get("passed") is not True:
        target.errors.append("training export contains unredacted secret-like values.")
    if manifest is not None:
        _validate_training_manifest(
            manifest,
            target,
            episodes,
            rewards,
            step_rewards,
            preferences,
            failure_modes,
            curriculum,
            sft,
            action_sft if action_sft_path.exists() else None,
            dpo,
            reward_model,
            dataset_metrics,
            dataset_splits,
            dataset_registry,
            expected_redaction_status,
            expected_label_provenance,
            dataset_card_path.exists(),
            export_dir,
        )
    _validate_episodes(episodes, target)
    _validate_rewards(rewards, target, episodes)
    _validate_step_rewards(step_rewards, target, episodes, rewards)
    _validate_preferences(preferences, target, episodes)
    _validate_failure_modes(failure_modes, target, episodes)
    if curriculum is not None:
        _validate_curriculum(curriculum, target, episodes, failure_modes)
    if sft_path.exists():
        _validate_sft_records(sft, target, episodes)
    if action_sft_path.exists():
        _validate_action_sft_records(action_sft, target, episodes)
    if dpo_path.exists():
        _validate_dpo_records(dpo, target, preferences, episodes)
    if reward_model_path.exists():
        _validate_reward_model_records(reward_model, target, episodes)
    if dataset_metrics is not None:
        _validate_dataset_metrics(
            dataset_metrics,
            target,
            episodes,
            rewards,
            step_rewards,
            preferences,
            failure_modes,
            sft,
            action_sft if action_sft_path.exists() else None,
            dpo,
            reward_model,
            dataset_splits,
            expected_redaction_status,
            expected_label_provenance,
        )
    if dataset_splits is not None:
        _validate_dataset_splits(dataset_splits, target, rows_by_artifact, split_rows)
    if dataset_registry is not None and manifest is not None:
        _validate_dataset_registry(
            dataset_registry,
            target,
            manifest,
            export_dir,
            expected_redaction_status,
            expected_label_provenance,
            dataset_splits,
            dataset_metrics,
            episodes,
        )
    if dataset_card_path.exists():
        _validate_dataset_card(dataset_card_path, target)
    else:
        target.warnings.append("DATASET_CARD.md is missing; rerun export-rl to emit the human-readable dataset card.")
    target.details.update(
        {
            "episode_count": len(episodes),
            "reward_count": len(rewards),
            "step_reward_count": len(step_rewards),
            "preference_count": len(preferences),
            "failure_mode_count": len(failure_modes),
            "sft_count": len(sft),
            "action_sft_count": len(action_sft),
            "dpo_count": len(dpo),
            "reward_model_count": len(reward_model),
            "quality_flag_count": len(dataset_metrics.get("quality_flags", [])) if isinstance(dataset_metrics, dict) else None,
            "split_episode_counts": dataset_splits.get("summary") if isinstance(dataset_splits, dict) else None,
        }
    )
    return target

def validate_compare_export(path: str | Path) -> ValidationTarget:
    """Validate an export-compare-rl output directory."""
    export_dir = Path(path)
    target = ValidationTarget("compare_export", str(export_dir))
    if _reject_symlinked_validation_path(export_dir, target, "Compare export path", "directory"):
        return target
    if not export_dir.exists():
        target.errors.append(f"Compare export directory not found: {export_dir}")
        return target
    if not export_dir.is_dir():
        target.errors.append(f"Compare export path is not a directory: {export_dir}")
        return target

    manifest_path = export_dir / "manifest.json"
    manifest = (
        None
        if _reject_symlinked_validation_path(manifest_path, target, "manifest.json", "file")
        else _read_object(manifest_path, target, "manifest.json")
    )
    pairs = _read_jsonl_objects(export_dir / "improvement_pairs.jsonl", target, "improvement_pairs.jsonl")
    dpo = _read_jsonl_objects(export_dir / "improvement_dpo.jsonl", target, "improvement_dpo.jsonl")
    card_path = export_dir / "IMPROVEMENT_CARD.md"
    if manifest is not None:
        _validate_compare_manifest(manifest, target, pairs, dpo, card_path.exists(), export_dir)
    _validate_compare_pairs(pairs, target)
    _validate_compare_dpo(dpo, target, pairs)
    if card_path.exists():
        _validate_improvement_card(card_path, target)
    else:
        target.warnings.append("IMPROVEMENT_CARD.md is missing; comparison export is less reviewable.")
    target.details.update(
        {
            "pair_count": len(pairs),
            "dpo_count": len(dpo),
            "candidate_win_count": sum(1 for pair in pairs if pair.get("chosen_side") == "candidate"),
            "baseline_win_count": sum(1 for pair in pairs if pair.get("chosen_side") == "baseline"),
        }
    )
    return target

def _validate_training_manifest(
    manifest: dict[str, Any],
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
    rewards: list[dict[str, Any]],
    step_rewards: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    failure_modes: list[dict[str, Any]],
    curriculum: dict[str, Any] | None,
    sft: list[dict[str, Any]],
    action_sft: list[dict[str, Any]] | None,
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
    dataset_metrics: dict[str, Any] | None,
    dataset_splits: dict[str, Any] | None,
    dataset_registry: dict[str, Any] | None,
    expected_redaction_status: dict[str, Any],
    expected_label_provenance: dict[str, Any],
    has_dataset_card: bool,
    export_dir: Path,
) -> None:
    _require_equal(manifest, "schema_version", RL_MANIFEST_SCHEMA_VERSION, target)
    dataset_version = manifest.get("dataset_version")
    versioned_manifest = _is_dataset_version(dataset_version)
    if not versioned_manifest:
        target.warnings.append("manifest.dataset_version is missing; rerun export-rl to emit a selectable dataset version.")
    expected_counts = {
        "episode_count": len(episodes),
        "reward_count": len(rewards),
        "preference_count": len(preferences),
        "failure_mode_count": len(failure_modes),
    }
    if "step_reward_count" in manifest:
        expected_counts["step_reward_count"] = len(step_rewards)
    else:
        target.warnings.append("manifest.step_reward_count is missing; rerun export-rl to refresh training artifacts.")
    trainer_view_counts = {
        "sft_count": len(sft),
        "dpo_count": len(dpo),
        "reward_model_count": len(reward_model),
    }
    if action_sft is not None:
        trainer_view_counts["action_sft_count"] = len(action_sft)
    for field_name, expected in trainer_view_counts.items():
        if field_name in manifest:
            expected_counts[field_name] = expected
        else:
            target.warnings.append(f"manifest.{field_name} is missing; rerun export-rl to refresh trainer-ready views.")
    if "quality_flag_count" in manifest and dataset_metrics is not None:
        expected_counts["quality_flag_count"] = len(dataset_metrics.get("quality_flags", [])) if isinstance(dataset_metrics.get("quality_flags"), list) else 0
    elif "quality_flag_count" not in manifest:
        target.warnings.append("manifest.quality_flag_count is missing; rerun export-rl to refresh dataset-level metrics.")
    for field_name, expected in expected_counts.items():
        if manifest.get(field_name) != expected:
            target.errors.append(f"manifest.{field_name} expected {expected}, got {manifest.get(field_name)!r}.")
    if not isinstance(manifest.get("outputs"), dict):
        target.errors.append("manifest.outputs must be an object.")
    else:
        for output_name in ("episodes", "rewards", "preferences", "failure_modes", "curriculum", "manifest"):
            if output_name not in manifest["outputs"]:
                target.errors.append(f"manifest.outputs.{output_name} is missing.")
        if "step_rewards" not in manifest["outputs"]:
            target.warnings.append("manifest.outputs.step_rewards is missing; rerun export-rl to refresh training artifacts.")
        trainer_output_names = ["sft", "dpo", "reward_model"]
        if action_sft is not None:
            trainer_output_names.insert(1, "action_sft")
        for output_name in trainer_output_names:
            if output_name not in manifest["outputs"]:
                target.warnings.append(f"manifest.outputs.{output_name} is missing; rerun export-rl to refresh trainer-ready views.")
        if "dataset_metrics" not in manifest["outputs"]:
            target.warnings.append("manifest.outputs.dataset_metrics is missing; rerun export-rl to refresh dataset-level metrics.")
        if "dataset_splits" not in manifest["outputs"]:
            target.warnings.append("manifest.outputs.dataset_splits is missing; rerun export-rl to refresh deterministic split artifacts.")
        if "dataset_card" not in manifest["outputs"]:
            target.warnings.append("manifest.outputs.dataset_card is missing; rerun export-rl to refresh the dataset card.")
        if "dataset_registry" not in manifest["outputs"]:
            if versioned_manifest:
                target.errors.append("manifest.outputs.dataset_registry is missing.")
            else:
                target.warnings.append("manifest.outputs.dataset_registry is missing; rerun export-rl to emit dataset lineage.")
        for split_name in DATASET_SPLIT_NAMES:
            for artifact_name in DATASET_SPLIT_ARTIFACTS:
                if artifact_name == "action_sft" and action_sft is None:
                    continue
                output_name = f"{split_name}_{artifact_name}"
                if output_name not in manifest["outputs"]:
                    target.warnings.append(f"manifest.outputs.{output_name} is missing; rerun export-rl to refresh split artifacts.")
    _validate_manifest_artifact_fingerprints(
        manifest.get("artifact_fingerprints"),
        target,
        "manifest.artifact_fingerprints",
        _training_export_artifact_paths(export_dir),
    )
    if dataset_metrics is None:
        target.warnings.append("manifest has no validated dataset_metrics.json companion.")
    if dataset_splits is None:
        target.warnings.append("manifest has no validated dataset_splits.json companion.")
    elif manifest.get("dataset_splits") != dataset_splits.get("summary"):
        target.errors.append("manifest.dataset_splits must match dataset_splits.summary.")
    if dataset_registry is None:
        if versioned_manifest:
            target.errors.append("manifest has no validated dataset_registry.json companion.")
        else:
            target.warnings.append("manifest has no dataset_registry.json companion; rerun export-rl to emit dataset lineage.")
    if "redaction_status" in manifest:
        if manifest.get("redaction_status") != expected_redaction_status:
            target.errors.append("manifest.redaction_status must match recomputed redaction scan.")
    else:
        target.warnings.append("manifest.redaction_status is missing; rerun export-rl to emit redaction proof.")
    if "label_provenance" in manifest:
        if manifest.get("label_provenance") != expected_label_provenance:
            target.errors.append("manifest.label_provenance must match recomputed label provenance.")
    else:
        target.warnings.append("manifest.label_provenance is missing; rerun export-rl to emit label provenance.")
    registry = manifest.get("registry")
    if not isinstance(registry, dict):
        if versioned_manifest:
            target.errors.append("manifest.registry must be an object.")
        else:
            target.warnings.append("manifest.registry is missing; rerun export-rl to emit dataset lineage.")
    else:
        _require_equal(registry, "schema_version", RL_DATASET_REGISTRY_SCHEMA_VERSION, target, prefix="manifest.registry.")
        if registry.get("selection_key") != dataset_version:
            target.errors.append("manifest.registry.selection_key must match manifest.dataset_version.")
        if registry.get("redaction_passed") is not (expected_redaction_status.get("passed") is True):
            target.errors.append("manifest.registry.redaction_passed must match redaction_status.passed.")
        leakage = dataset_splits.get("leakage_checks") if isinstance(dataset_splits, dict) else {}
        if registry.get("heldout_scenario_exclusive") is not (isinstance(leakage, dict) and leakage.get("heldout_scenario_exclusive") is True):
            target.errors.append("manifest.registry.heldout_scenario_exclusive must match dataset_splits.leakage_checks.")
    if not has_dataset_card:
        target.warnings.append("manifest has no DATASET_CARD.md companion.")
    if "metadata" in manifest:
        _validate_metadata(manifest.get("metadata"), target, "manifest.metadata")
    if dataset_metrics is not None and "metadata" in manifest and "metadata" in dataset_metrics and manifest.get("metadata") != dataset_metrics.get("metadata"):
        target.errors.append("manifest.metadata does not match dataset_metrics.metadata.")
    if curriculum is not None and curriculum.get("failure_mode_count") != len(failure_modes):
        target.errors.append(
            f"curriculum.failure_mode_count expected {len(failure_modes)}, got {curriculum.get('failure_mode_count')!r}."
        )
    if _looks_absolute(str(manifest.get("source_runs_dir", ""))):
        target.warnings.append("manifest.source_runs_dir is absolute; prefer redacted or relative exports for sharing.")
    if _looks_absolute(str(manifest.get("output_dir", ""))):
        target.warnings.append("manifest.output_dir is absolute; prefer redacted or relative exports for sharing.")

def _validate_compare_manifest(
    manifest: dict[str, Any],
    target: ValidationTarget,
    pairs: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    has_card: bool,
    export_dir: Path,
) -> None:
    _require_equal(manifest, "schema_version", COMPARE_RL_MANIFEST_SCHEMA_VERSION, target)
    expected_counts = {
        "pair_count": len(pairs),
        "dpo_count": len(dpo),
        "candidate_win_count": sum(1 for pair in pairs if pair.get("chosen_side") == "candidate"),
        "baseline_win_count": sum(1 for pair in pairs if pair.get("chosen_side") == "baseline"),
    }
    for field_name, expected in expected_counts.items():
        if manifest.get(field_name) != expected:
            target.errors.append(f"compare_manifest.{field_name} expected {expected}, got {manifest.get(field_name)!r}.")
    expected_movement = compare_movement_summary(pairs)
    for field_name, expected in expected_movement.items():
        if field_name not in manifest:
            target.errors.append(f"compare_manifest.{field_name} is missing.")
        elif manifest.get(field_name) != expected:
            target.errors.append(f"compare_manifest.{field_name} expected {expected!r}, got {manifest.get(field_name)!r}.")
    for field_name in ("baseline_run_count", "candidate_run_count", "paired_scenario_count", "skipped_pair_count", "min_score_gap"):
        if not _is_non_negative_int(manifest.get(field_name)):
            target.errors.append(f"compare_manifest.{field_name} must be a non-negative integer.")
    expected_contract_drift_count = sum(1 for pair in pairs if pair.get("contract_fingerprint_status") == "drifted")
    expected_unverified_contract_count = sum(1 for pair in pairs if pair.get("contract_fingerprint_status") == "unverified")
    if manifest.get("contract_drift_count") != expected_contract_drift_count:
        target.errors.append(
            f"compare_manifest.contract_drift_count expected {expected_contract_drift_count}, got {manifest.get('contract_drift_count')!r}."
        )
    if manifest.get("unverified_contract_count") != expected_unverified_contract_count:
        target.errors.append(
            "compare_manifest.unverified_contract_count "
            f"expected {expected_unverified_contract_count}, got {manifest.get('unverified_contract_count')!r}."
        )
    if not isinstance(manifest.get("outputs"), dict):
        target.errors.append("compare_manifest.outputs must be an object.")
    else:
        for output_name in ("improvement_pairs", "improvement_dpo", "manifest", "improvement_card"):
            if output_name not in manifest["outputs"]:
                target.errors.append(f"compare_manifest.outputs.{output_name} is missing.")
    _validate_manifest_artifact_fingerprints(
        manifest.get("artifact_fingerprints"),
        target,
        "compare_manifest.artifact_fingerprints",
        _compare_export_artifact_paths(export_dir),
    )
    for field_name in ("missing_in_candidate", "new_in_candidate"):
        if not _is_string_list(manifest.get(field_name)):
            target.errors.append(f"compare_manifest.{field_name} must be a list of strings.")
    skipped = manifest.get("skipped_pairs")
    if not isinstance(skipped, list):
        target.errors.append("compare_manifest.skipped_pairs must be a list.")
    if "metadata" in manifest:
        _validate_metadata(manifest.get("metadata"), target, "compare_manifest.metadata")
    if not has_card:
        target.warnings.append("compare_manifest has no IMPROVEMENT_CARD.md companion.")
    if "contract_scope" in manifest and manifest.get("contract_scope") not in CONTRACT_SCOPES:
        target.errors.append(f"compare_manifest.contract_scope must be one of {sorted(CONTRACT_SCOPES)!r}.")
    if manifest.get("contract_scope") in CONTRACT_SCOPES:
        for index, pair in enumerate(pairs):
            if isinstance(pair, dict) and pair.get("contract_fingerprint_scope") not in {None, manifest.get("contract_scope")}:
                target.errors.append(
                    f"improvement_pairs[{index}].contract_fingerprint_scope must match compare_manifest.contract_scope."
                )
    for field_name in ("baseline_runs_dir", "candidate_runs_dir", "output_dir"):
        if _looks_absolute(str(manifest.get(field_name, ""))):
            target.warnings.append(f"compare_manifest.{field_name} is absolute; prefer redacted or relative exports for sharing.")

def _validate_manifest_artifact_fingerprints(
    value: Any,
    target: ValidationTarget,
    label: str,
    expected_paths: dict[str, Path],
) -> None:
    if value is None:
        target.errors.append(f"{label} is missing; rerun the export to emit artifact integrity hashes.")
        return
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return

    expected_names = set(expected_paths)
    actual_names = {name for name in value if isinstance(name, str)}
    for name in sorted(actual_names - expected_names):
        target.errors.append(f"{label}.{name} is not a known export artifact.")
    for name in value:
        if not isinstance(name, str) or not name:
            target.errors.append(f"{label} keys must be non-empty strings.")

    for name, path in expected_paths.items():
        record = value.get(name)
        record_label = f"{label}.{name}"
        if not isinstance(record, dict):
            target.errors.append(f"{record_label} must be an object.")
            continue
        if not isinstance(record.get("path"), str) or not record.get("path"):
            target.errors.append(f"{record_label}.path must be a non-empty string.")
        elif _looks_absolute(record["path"]):
            target.warnings.append(f"{record_label}.path is absolute; prefer redacted or relative exports for sharing.")
        if record.get("exists") is not True:
            target.errors.append(f"{record_label}.exists must be true for generated export artifacts.")
        if path.is_symlink():
            target.errors.append(f"{record_label} file must not be a symlink: {path}")
            continue
        if not _path_resolves_inside(path, path.parent):
            target.errors.append(f"{record_label} file must resolve inside the export directory: {path}")
            continue
        if not path.exists() or not path.is_file():
            target.errors.append(f"{record_label} file is missing: {path}")
            continue
        size_bytes = record.get("size_bytes")
        if not _is_non_negative_int(size_bytes):
            target.errors.append(f"{record_label}.size_bytes must be a non-negative integer.")
        elif size_bytes != path.stat().st_size:
            target.errors.append(f"{record_label}.size_bytes does not match current file size.")
        expected_sha = record.get("sha256")
        if not _is_sha256(expected_sha):
            target.errors.append(f"{record_label}.sha256 must be a SHA-256 hex string.")
        elif _sha256(path) != expected_sha:
            target.errors.append(f"{record_label}.sha256 does not match current file contents.")

def _training_export_artifact_paths(export_dir: Path) -> dict[str, Path]:
    paths = {
        "curriculum": export_dir / "curriculum.json",
        "dataset_card": export_dir / "DATASET_CARD.md",
        "dataset_metrics": export_dir / "dataset_metrics.json",
        "dataset_splits": export_dir / "dataset_splits.json",
        "dpo": export_dir / "dpo.jsonl",
        "episodes": export_dir / "episodes.jsonl",
        "failure_modes": export_dir / "failure_modes.jsonl",
        "preferences": export_dir / "preferences.jsonl",
        "reward_model": export_dir / "reward_model.jsonl",
        "rewards": export_dir / "rewards.jsonl",
        "sft": export_dir / "sft.jsonl",
        "step_rewards": export_dir / "step_rewards.jsonl",
    }
    if (export_dir / "action_sft.jsonl").exists():
        paths["action_sft"] = export_dir / "action_sft.jsonl"
    for split_name in DATASET_SPLIT_NAMES:
        for artifact_name in DATASET_SPLIT_ARTIFACTS:
            if artifact_name == "action_sft" and not (export_dir / "action_sft.jsonl").exists():
                continue
            paths[f"{split_name}_{artifact_name}"] = export_dir / "splits" / split_name / f"{artifact_name}.jsonl"
    return paths

def _read_training_split_rows(export_dir: Path, target: ValidationTarget) -> dict[str, dict[str, list[dict[str, Any]]]]:
    split_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split_name in DATASET_SPLIT_NAMES:
        split_rows[split_name] = {}
        for artifact_name in DATASET_SPLIT_ARTIFACTS:
            if artifact_name == "action_sft" and not (export_dir / "action_sft.jsonl").exists():
                split_rows[split_name][artifact_name] = []
                continue
            label = f"splits/{split_name}/{artifact_name}.jsonl"
            split_rows[split_name][artifact_name] = _read_jsonl_objects_optional(
                export_dir / "splits" / split_name / f"{artifact_name}.jsonl",
                target,
                label,
                "rerun export-rl to emit deterministic train/validation/test split artifacts",
            )
    return split_rows

def _validate_dataset_splits(
    value: dict[str, Any],
    target: ValidationTarget,
    rows_by_artifact: dict[str, list[dict[str, Any]]],
    split_rows: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    _require_equal(value, "schema_version", RL_DATASET_SPLITS_SCHEMA_VERSION, target, prefix="dataset_splits.")
    if value.get("strategy") != "task_family_hash":
        target.errors.append("dataset_splits.strategy must be 'task_family_hash'.")
    if value.get("split_names") != list(DATASET_SPLIT_NAMES):
        target.errors.append(f"dataset_splits.split_names must be {list(DATASET_SPLIT_NAMES)!r}.")
    legacy_artifacts = [name for name in DATASET_SPLIT_ARTIFACTS if name != "action_sft"]
    if value.get("artifact_names") != list(DATASET_SPLIT_ARTIFACTS) and value.get("artifact_names") != legacy_artifacts:
        target.errors.append(
            f"dataset_splits.artifact_names must be {list(DATASET_SPLIT_ARTIFACTS)!r} "
            f"(or legacy {legacy_artifacts!r})."
        )

    episodes = rows_by_artifact.get("episodes", [])
    episode_by_id = {str(episode.get("episode_id")): episode for episode in episodes if isinstance(episode.get("episode_id"), str)}
    family_episode_ids: dict[str, list[str]] = {}
    family_scenario_ids: dict[str, set[str]] = {}
    for episode in episodes:
        family = str(episode.get("task_family") or "unknown")
        family_episode_ids.setdefault(family, []).append(str(episode.get("episode_id") or ""))
        family_scenario_ids.setdefault(family, set()).add(str(episode.get("scenario_id") or ""))

    assignments = value.get("assignments")
    if not isinstance(assignments, list):
        target.errors.append("dataset_splits.assignments must be a list.")
        assignments = []
    family_to_split: dict[str, str] = {}
    assigned_episode_ids: set[str] = set()
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            target.errors.append(f"dataset_splits.assignments[{index}] must be an object.")
            continue
        family = assignment.get("task_family")
        split_name = assignment.get("split")
        if not isinstance(family, str) or not family:
            target.errors.append(f"dataset_splits.assignments[{index}].task_family must be a non-empty string.")
            continue
        if family in family_to_split:
            target.errors.append(f"dataset_splits.assignments[{index}].task_family duplicates {family!r}.")
        if split_name not in DATASET_SPLIT_NAMES:
            target.errors.append(f"dataset_splits.assignments[{index}].split must be one of {list(DATASET_SPLIT_NAMES)!r}.")
            split_name = "train"
        family_to_split[family] = str(split_name)
        expected_episode_ids = sorted(family_episode_ids.get(family, []))
        expected_scenario_ids = sorted(family_scenario_ids.get(family, set()))
        if assignment.get("episode_count") != len(expected_episode_ids):
            target.errors.append(
                f"dataset_splits.assignments[{index}].episode_count expected {len(expected_episode_ids)}, got {assignment.get('episode_count')!r}."
            )
        if assignment.get("episode_ids") != expected_episode_ids:
            target.errors.append(f"dataset_splits.assignments[{index}].episode_ids must match exported episodes for family {family!r}.")
        if assignment.get("scenario_ids") != expected_scenario_ids:
            target.errors.append(f"dataset_splits.assignments[{index}].scenario_ids must match exported scenarios for family {family!r}.")
        assigned_episode_ids.update(item for item in expected_episode_ids if item)

    expected_families = set(family_episode_ids)
    missing_families = sorted(expected_families - set(family_to_split))
    unknown_families = sorted(set(family_to_split) - expected_families)
    if missing_families:
        target.errors.append(f"dataset_splits.assignments missing task families: {missing_families!r}.")
    if unknown_families:
        target.errors.append(f"dataset_splits.assignments contain unknown task families: {unknown_families!r}.")
    if assigned_episode_ids != set(episode_by_id):
        target.errors.append("dataset_splits.assignments episode_ids must cover exported episodes exactly.")

    placement_errors = _validate_split_row_placement(rows_by_artifact, split_rows, family_to_split, episode_by_id, target)
    cross_split_families = _cross_split_families(split_rows)
    family_exclusive = not cross_split_families and not placement_errors
    split_scenario_ids = _split_scenario_ids(split_rows)
    train_scenario_ids = split_scenario_ids["train"]
    heldout_scenario_ids = sorted(set(split_scenario_ids["validation"]) | set(split_scenario_ids["test"]))
    cross_split_scenario_ids = sorted(set(train_scenario_ids) & set(heldout_scenario_ids))
    heldout_scenario_exclusive = not cross_split_scenario_ids
    train_task_families = _split_task_families(split_rows)["train"]
    heldout_task_families = sorted(set(_split_task_families(split_rows)["validation"]) | set(_split_task_families(split_rows)["test"]))
    if value.get("split_scenario_ids") != split_scenario_ids:
        target.errors.append("dataset_splits.split_scenario_ids must match split episode scenario IDs.")
    active_artifacts = [
        name for name in DATASET_SPLIT_ARTIFACTS if name in (value.get("artifact_names") or [])
    ]
    _validate_dataset_split_counts(value.get("split_counts"), target, assignments, split_rows, active_artifacts)
    _validate_dataset_split_summary(
        value.get("summary"),
        target,
        family_to_split,
        episodes,
        split_rows,
        family_exclusive,
        train_scenario_ids,
        heldout_scenario_ids,
        heldout_scenario_exclusive,
    )
    _validate_dataset_split_leakage(
        value.get("leakage_checks"),
        target,
        family_exclusive,
        cross_split_families,
        train_scenario_ids,
        heldout_scenario_ids,
        cross_split_scenario_ids,
        heldout_scenario_exclusive,
        train_task_families,
        heldout_task_families,
    )

def _split_scenario_ids(split_rows: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, list[str]]:
    return {
        split_name: sorted(
            {
                str(episode.get("scenario_id") or "")
                for episode in split_rows[split_name]["episodes"]
                if str(episode.get("scenario_id") or "")
            }
        )
        for split_name in DATASET_SPLIT_NAMES
    }

def _split_task_families(split_rows: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, list[str]]:
    return {
        split_name: sorted({str(episode.get("task_family") or "unknown") for episode in split_rows[split_name]["episodes"]})
        for split_name in DATASET_SPLIT_NAMES
    }

def _validate_split_row_placement(
    rows_by_artifact: dict[str, list[dict[str, Any]]],
    split_rows: dict[str, dict[str, list[dict[str, Any]]]],
    family_to_split: dict[str, str],
    episode_by_id: dict[str, dict[str, Any]],
    target: ValidationTarget,
) -> int:
    errors = 0
    episode_family = {
        episode_id: str(episode.get("task_family") or "unknown")
        for episode_id, episode in episode_by_id.items()
    }
    for artifact_name in DATASET_SPLIT_ARTIFACTS:
        expected_total = len(rows_by_artifact.get(artifact_name, []))
        actual_total = sum(len(split_rows[split_name][artifact_name]) for split_name in DATASET_SPLIT_NAMES)
        if actual_total != expected_total:
            target.errors.append(
                f"dataset_splits split rows for {artifact_name} expected {expected_total}, got {actual_total}."
            )
            errors += 1
    for split_name in DATASET_SPLIT_NAMES:
        for artifact_name in DATASET_SPLIT_ARTIFACTS:
            for row_index, row in enumerate(split_rows[split_name][artifact_name]):
                expected_split = _expected_row_split(row, family_to_split, episode_family)
                if expected_split != split_name:
                    target.errors.append(
                        f"splits/{split_name}/{artifact_name}.jsonl row {row_index + 1} belongs in {expected_split!r}."
                    )
                    errors += 1
                if artifact_name in {"preferences", "dpo"}:
                    errors += _validate_pair_row_split_locality(row, artifact_name, row_index, episode_family, family_to_split, target)
    return errors

def _validate_pair_row_split_locality(
    row: dict[str, Any],
    artifact_name: str,
    row_index: int,
    episode_family: dict[str, str],
    family_to_split: dict[str, str],
    target: ValidationTarget,
) -> int:
    chosen_id = row.get("chosen_episode_id")
    rejected_id = row.get("rejected_episode_id")
    if not isinstance(chosen_id, str) or not isinstance(rejected_id, str):
        return 0
    chosen_split = family_to_split.get(episode_family.get(chosen_id, ""))
    rejected_split = family_to_split.get(episode_family.get(rejected_id, ""))
    if chosen_split and rejected_split and chosen_split != rejected_split:
        target.errors.append(
            f"{artifact_name}[{row_index}] crosses dataset splits: chosen={chosen_split!r}, rejected={rejected_split!r}."
        )
        return 1
    return 0

def _expected_row_split(row: dict[str, Any], family_to_split: dict[str, str], episode_family: dict[str, str]) -> str:
    family = row.get("task_family")
    if isinstance(family, str) and family in family_to_split:
        return family_to_split[family]
    episode_id = row.get("episode_id")
    if isinstance(episode_id, str):
        return family_to_split.get(episode_family.get(episode_id, ""), "train")
    for field_name in ("chosen_episode_id", "rejected_episode_id"):
        candidate = row.get(field_name)
        if isinstance(candidate, str):
            split_name = family_to_split.get(episode_family.get(candidate, ""))
            if split_name:
                return split_name
    return "train"

def _cross_split_families(split_rows: dict[str, dict[str, list[dict[str, Any]]]]) -> list[str]:
    family_splits: dict[str, set[str]] = {}
    for split_name in DATASET_SPLIT_NAMES:
        for episode in split_rows[split_name]["episodes"]:
            family_splits.setdefault(str(episode.get("task_family") or "unknown"), set()).add(split_name)
    return sorted(family for family, splits in family_splits.items() if len(splits) > 1)

def _validate_dataset_split_counts(
    value: Any,
    target: ValidationTarget,
    assignments: list[Any],
    split_rows: dict[str, dict[str, list[dict[str, Any]]]],
    active_artifacts: list[str],
) -> None:
    if not isinstance(value, dict):
        target.errors.append("dataset_splits.split_counts must be an object.")
        return
    for split_name in DATASET_SPLIT_NAMES:
        split_count = value.get(split_name)
        if not isinstance(split_count, dict):
            target.errors.append(f"dataset_splits.split_counts.{split_name} must be an object.")
            continue
        expected_family_count = sum(
            1 for assignment in assignments if isinstance(assignment, dict) and assignment.get("split") == split_name
        )
        expected_episode_count = len(split_rows[split_name]["episodes"])
        if split_count.get("task_family_count") != expected_family_count:
            target.errors.append(
                f"dataset_splits.split_counts.{split_name}.task_family_count expected {expected_family_count}, got {split_count.get('task_family_count')!r}."
            )
        if split_count.get("episode_count") != expected_episode_count:
            target.errors.append(
                f"dataset_splits.split_counts.{split_name}.episode_count expected {expected_episode_count}, got {split_count.get('episode_count')!r}."
            )
        artifacts = split_count.get("artifacts")
        if not isinstance(artifacts, dict):
            target.errors.append(f"dataset_splits.split_counts.{split_name}.artifacts must be an object.")
            continue
        for artifact_name in active_artifacts:
            expected_artifact_count = len(split_rows[split_name][artifact_name])
            if artifacts.get(artifact_name) != expected_artifact_count:
                target.errors.append(
                    f"dataset_splits.split_counts.{split_name}.artifacts.{artifact_name} expected {expected_artifact_count}, got {artifacts.get(artifact_name)!r}."
                )

def _validate_dataset_split_summary(
    value: Any,
    target: ValidationTarget,
    family_to_split: dict[str, str],
    episodes: list[dict[str, Any]],
    split_rows: dict[str, dict[str, list[dict[str, Any]]]],
    family_exclusive: bool,
    train_scenario_ids: list[str],
    heldout_scenario_ids: list[str],
    heldout_scenario_exclusive: bool,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("dataset_splits.summary must be an object.")
        return
    expected = {
        "task_family_count": len(family_to_split),
        "episode_count": len(episodes),
        "train_episode_count": len(split_rows["train"]["episodes"]),
        "validation_episode_count": len(split_rows["validation"]["episodes"]),
        "test_episode_count": len(split_rows["test"]["episodes"]),
        "family_exclusive": family_exclusive,
        "train_scenario_count": len(train_scenario_ids),
        "heldout_scenario_count": len(heldout_scenario_ids),
        "heldout_scenario_exclusive": heldout_scenario_exclusive,
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"dataset_splits.summary.{field_name} expected {expected_value!r}, got {value.get(field_name)!r}.")

def _validate_dataset_split_leakage(
    value: Any,
    target: ValidationTarget,
    family_exclusive: bool,
    cross_split_families: list[str],
    train_scenario_ids: list[str],
    heldout_scenario_ids: list[str],
    cross_split_scenario_ids: list[str],
    heldout_scenario_exclusive: bool,
    train_task_families: list[str],
    heldout_task_families: list[str],
) -> None:
    if not isinstance(value, dict):
        target.errors.append("dataset_splits.leakage_checks must be an object.")
        return
    if value.get("family_exclusive") != family_exclusive:
        target.errors.append(
            f"dataset_splits.leakage_checks.family_exclusive expected {family_exclusive!r}, got {value.get('family_exclusive')!r}."
        )
    if value.get("cross_split_task_families") != cross_split_families:
        target.errors.append(
            f"dataset_splits.leakage_checks.cross_split_task_families expected {cross_split_families!r}, got {value.get('cross_split_task_families')!r}."
        )
    expected = {
        "train_scenario_ids": train_scenario_ids,
        "heldout_scenario_ids": heldout_scenario_ids,
        "cross_split_scenario_ids": cross_split_scenario_ids,
        "heldout_scenario_exclusive": heldout_scenario_exclusive,
        "train_task_families": train_task_families,
        "heldout_task_families": heldout_task_families,
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(
                f"dataset_splits.leakage_checks.{field_name} expected {expected_value!r}, got {value.get(field_name)!r}."
            )

def _validate_dataset_registry(
    registry: dict[str, Any],
    target: ValidationTarget,
    manifest: dict[str, Any],
    export_dir: Path,
    expected_redaction_status: dict[str, Any],
    expected_label_provenance: dict[str, Any],
    dataset_splits: dict[str, Any] | None,
    dataset_metrics: dict[str, Any] | None,
    episodes: list[dict[str, Any]],
) -> None:
    _require_equal(registry, "schema_version", RL_DATASET_REGISTRY_SCHEMA_VERSION, target, prefix="dataset_registry.")
    if registry.get("artifact_type") != "training_export":
        target.errors.append("dataset_registry.artifact_type must be 'training_export'.")
    if registry.get("dataset_version") != manifest.get("dataset_version"):
        target.errors.append("dataset_registry.dataset_version must match manifest.dataset_version.")
    if registry.get("manifest_sha256") != _sha256(export_dir / "manifest.json"):
        target.errors.append("dataset_registry.manifest_sha256 must match manifest.json contents.")
    selection = registry.get("selection")
    if not isinstance(selection, dict):
        target.errors.append("dataset_registry.selection must be an object.")
    elif selection.get("key") != manifest.get("dataset_version"):
        target.errors.append("dataset_registry.selection.key must match manifest.dataset_version.")
    if registry.get("redaction_status") != expected_redaction_status:
        target.errors.append("dataset_registry.redaction_status must match recomputed redaction scan.")
    if registry.get("label_provenance") != expected_label_provenance:
        target.errors.append("dataset_registry.label_provenance must match recomputed label provenance.")
    if registry.get("dataset_splits") != (dataset_splits.get("summary") if isinstance(dataset_splits, dict) else None):
        target.errors.append("dataset_registry.dataset_splits must match dataset_splits.summary.")
    if registry.get("leakage_checks") != (dataset_splits.get("leakage_checks") if isinstance(dataset_splits, dict) else None):
        target.errors.append("dataset_registry.leakage_checks must match dataset_splits.leakage_checks.")
    if registry.get("source_fingerprint_coverage") != (
        dataset_metrics.get("source_fingerprint_coverage") if isinstance(dataset_metrics, dict) else None
    ):
        target.errors.append("dataset_registry.source_fingerprint_coverage must match dataset_metrics.source_fingerprint_coverage.")
    if registry.get("quality_flags") != (dataset_metrics.get("quality_flags") if isinstance(dataset_metrics, dict) else None):
        target.errors.append("dataset_registry.quality_flags must match dataset_metrics.quality_flags.")
    if registry.get("artifact_fingerprints") != manifest.get("artifact_fingerprints"):
        target.errors.append("dataset_registry.artifact_fingerprints must match manifest.artifact_fingerprints.")
    source_runs = registry.get("source_runs")
    if not isinstance(source_runs, list):
        target.errors.append("dataset_registry.source_runs must be a list.")
    elif len(source_runs) != len(episodes):
        target.errors.append(f"dataset_registry.source_runs expected {len(episodes)} rows, got {len(source_runs)}.")

def _compare_export_artifact_paths(export_dir: Path) -> dict[str, Path]:
    return {
        "improvement_card": export_dir / "IMPROVEMENT_CARD.md",
        "improvement_dpo": export_dir / "improvement_dpo.jsonl",
        "improvement_pairs": export_dir / "improvement_pairs.jsonl",
    }

def _review_export_artifact_paths(export_dir: Path) -> dict[str, Path]:
    return {
        "instructions": export_dir / "REVIEW_INSTRUCTIONS.md",
        "label_template": export_dir / "label_template.jsonl",
        "review_items": export_dir / "review_items.jsonl",
    }

def _reviewed_export_artifact_paths(export_dir: Path) -> dict[str, Path]:
    return {
        "reviewed_action_sft": export_dir / "reviewed_action_sft.jsonl",
        "reviewed_dpo": export_dir / "reviewed_dpo.jsonl",
        "reviewed_labels": export_dir / "reviewed_labels.jsonl",
        "reviewed_preferences": export_dir / "reviewed_preferences.jsonl",
        "reviewed_reward_model": export_dir / "reviewed_reward_model.jsonl",
        "reviewed_sft": export_dir / "reviewed_sft.jsonl",
    }

def _validate_compare_pairs(pairs: list[dict[str, Any]], target: ValidationTarget) -> None:
    seen: set[str] = set()
    for index, pair in enumerate(pairs):
        label = f"improvement_pairs[{index}]"
        _require_equal(pair, "schema_version", COMPARE_RL_PAIR_SCHEMA_VERSION, target, prefix=f"{label}.")
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            target.errors.append(f"{label}.pair_id must be a non-empty string.")
        elif pair_id in seen:
            target.errors.append(f"{label}.pair_id duplicates {pair_id!r}.")
        else:
            seen.add(pair_id)
        for field_name in ("scenario_id", "task_family", "chosen_episode_id", "rejected_episode_id", "baseline_episode_id", "candidate_episode_id", "reason"):
            if not isinstance(pair.get(field_name), str) or not pair.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if not isinstance(pair.get("prompt"), str):
            target.errors.append(f"{label}.prompt must be a string.")
        if pair.get("candidate_outcome") not in {"improved", "regressed"}:
            target.errors.append(f"{label}.candidate_outcome must be improved or regressed.")
        chosen_side = pair.get("chosen_side")
        rejected_side = pair.get("rejected_side")
        if chosen_side not in {"baseline", "candidate"}:
            target.errors.append(f"{label}.chosen_side must be baseline or candidate.")
        if rejected_side not in {"baseline", "candidate"}:
            target.errors.append(f"{label}.rejected_side must be baseline or candidate.")
        if chosen_side == rejected_side:
            target.errors.append(f"{label}.chosen_side and rejected_side must differ.")
        if not _is_plain_int(pair.get("candidate_score_delta")):
            target.errors.append(f"{label}.candidate_score_delta must be an integer.")
        for field_name in ("chosen_score", "rejected_score", "score_gap"):
            if not _is_int_between(pair.get(field_name), 0, 100):
                target.errors.append(f"{label}.{field_name} must be an integer from 0 to 100.")
        if _is_int_between(pair.get("chosen_score"), 0, 100) and _is_int_between(pair.get("rejected_score"), 0, 100):
            expected_gap = pair["chosen_score"] - pair["rejected_score"]
            if pair.get("score_gap") != expected_gap:
                target.errors.append(f"{label}.score_gap expected {expected_gap}, got {pair.get('score_gap')!r}.")
            if expected_gap <= 0:
                target.errors.append(f"{label}.chosen_score must be greater than rejected_score.")
        for field_name in ("rule_fixes", "rule_regressions", "new_critical_failures"):
            if not _is_string_list(pair.get(field_name)):
                target.errors.append(f"{label}.{field_name} must be a list of strings.")
        if "contract_fingerprint_status" in pair:
            _validate_contract_fingerprint_status(pair, target, label)
        baseline = _validate_compare_view(pair.get("baseline"), target, f"{label}.baseline")
        candidate = _validate_compare_view(pair.get("candidate"), target, f"{label}.candidate")
        chosen = _validate_compare_view(pair.get("chosen"), target, f"{label}.chosen")
        rejected = _validate_compare_view(pair.get("rejected"), target, f"{label}.rejected")
        _validate_compare_pair_links(pair, baseline, candidate, chosen, rejected, target, label)

def _validate_compare_pair_links(
    pair: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    target: ValidationTarget,
    label: str,
) -> None:
    if baseline and pair.get("baseline_episode_id") != baseline.get("episode_id"):
        target.errors.append(f"{label}.baseline_episode_id must match baseline.episode_id.")
    if candidate and pair.get("candidate_episode_id") != candidate.get("episode_id"):
        target.errors.append(f"{label}.candidate_episode_id must match candidate.episode_id.")
    chosen_side = pair.get("chosen_side")
    rejected_side = pair.get("rejected_side")
    expected_chosen = candidate if chosen_side == "candidate" else baseline if chosen_side == "baseline" else {}
    expected_rejected = candidate if rejected_side == "candidate" else baseline if rejected_side == "baseline" else {}
    if expected_chosen and chosen and chosen.get("episode_id") != expected_chosen.get("episode_id"):
        target.errors.append(f"{label}.chosen must match the chosen_side view.")
    if expected_rejected and rejected and rejected.get("episode_id") != expected_rejected.get("episode_id"):
        target.errors.append(f"{label}.rejected must match the rejected_side view.")
    if pair.get("candidate_outcome") == "improved" and not (_is_plain_int(pair.get("candidate_score_delta")) and pair["candidate_score_delta"] > 0):
        target.errors.append(f"{label}.candidate_score_delta must be positive when candidate_outcome is improved.")
    if pair.get("candidate_outcome") == "regressed" and not (_is_plain_int(pair.get("candidate_score_delta")) and pair["candidate_score_delta"] < 0):
        target.errors.append(f"{label}.candidate_score_delta must be negative when candidate_outcome is regressed.")

def _validate_compare_view(value: Any, target: ValidationTarget, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return {}
    for field_name in ("episode_id", "scenario_id"):
        if not isinstance(value.get(field_name), str) or not value.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if not isinstance(value.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    if not _is_int_between(value.get("score"), 0, 100):
        target.errors.append(f"{label}.score must be an integer from 0 to 100.")
    if not isinstance(value.get("reward"), (int, float)) or isinstance(value.get("reward"), bool):
        target.errors.append(f"{label}.reward must be numeric.")
    if not _is_string_list(value.get("failed_rules")):
        target.errors.append(f"{label}.failed_rules must be a list of strings.")
    if not isinstance(value.get("events"), list):
        target.errors.append(f"{label}.events must be a list.")
    if not isinstance(value.get("final_answer"), str):
        target.errors.append(f"{label}.final_answer must be a string.")
    if "task_completion" in value:
        _validate_task_completion(value.get("task_completion"), target, f"{label}.task_completion")
    _validate_source_fingerprint_fields(value, target, label, warn_if_missing=False)
    return value

def _validate_compare_dpo(dpo: list[dict[str, Any]], target: ValidationTarget, pairs: list[dict[str, Any]]) -> None:
    pair_by_id = {pair.get("pair_id"): pair for pair in pairs if isinstance(pair.get("pair_id"), str)}
    seen: set[str] = set()
    for index, row in enumerate(dpo):
        label = f"improvement_dpo[{index}]"
        _require_equal(row, "schema_version", COMPARE_RL_DPO_SCHEMA_VERSION, target, prefix=f"{label}.")
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            target.errors.append(f"{label}.pair_id must be a non-empty string.")
            pair = None
        elif pair_id in seen:
            target.errors.append(f"{label}.pair_id duplicates {pair_id!r}.")
            pair = pair_by_id.get(pair_id)
        else:
            seen.add(pair_id)
            pair = pair_by_id.get(pair_id)
        if row.get("preference_id") != pair_id:
            target.errors.append(f"{label}.preference_id must match pair_id.")
        if pair is None:
            target.errors.append(f"{label}.pair_id {pair_id!r} does not reference an improvement pair.")
            continue
        if row.get("source_artifact") != "improvement_pairs.jsonl":
            target.errors.append(f"{label}.source_artifact must be 'improvement_pairs.jsonl'.")
        for field_name in (
            "scenario_id",
            "task_family",
            "prompt",
            "chosen",
            "rejected",
            "chosen_side",
            "rejected_side",
            "candidate_outcome",
            "reason",
        ):
            if not isinstance(row.get(field_name), str):
                target.errors.append(f"{label}.{field_name} must be a string.")
        for field_name in ("chosen_score", "rejected_score", "score_gap"):
            if not _is_int_between(row.get(field_name), 0, 100):
                target.errors.append(f"{label}.{field_name} must be an integer from 0 to 100.")
        for field_name in ("chosen_task_completion_status", "rejected_task_completion_status"):
            if row.get(field_name) not in {"complete", "incomplete", "not_applicable"}:
                target.errors.append(f"{label}.{field_name} must be complete, incomplete, or not_applicable.")
        for field_name in ("chosen_task_completion_passed", "rejected_task_completion_passed"):
            if not isinstance(row.get(field_name), bool):
                target.errors.append(f"{label}.{field_name} must be a boolean.")
        if not _is_plain_int(row.get("candidate_score_delta")):
            target.errors.append(f"{label}.candidate_score_delta must be an integer.")
        if "contract_fingerprint_status" in row:
            _validate_contract_fingerprint_status(row, target, label)
        _validate_messages(row.get("chosen_messages"), target, f"{label}.chosen_messages")
        _validate_messages(row.get("rejected_messages"), target, f"{label}.rejected_messages")
        _compare_improvement_dpo_to_pair(row, pair, target, label)
    missing = sorted(set(pair_by_id) - seen)
    if missing:
        target.errors.append(f"improvement_dpo.jsonl missing improvement pairs: {missing!r}.")

def _compare_improvement_dpo_to_pair(row: dict[str, Any], pair: dict[str, Any], target: ValidationTarget, label: str) -> None:
    chosen = pair.get("chosen") if isinstance(pair.get("chosen"), dict) else {}
    rejected = pair.get("rejected") if isinstance(pair.get("rejected"), dict) else {}
    expected = {
        "scenario_id": pair.get("scenario_id"),
        "task_family": pair.get("task_family"),
        "prompt": pair.get("prompt"),
        "chosen": _compare_response_text(chosen),
        "rejected": _compare_response_text(rejected),
        "chosen_side": pair.get("chosen_side"),
        "rejected_side": pair.get("rejected_side"),
        "candidate_outcome": pair.get("candidate_outcome"),
        "candidate_score_delta": pair.get("candidate_score_delta"),
        "chosen_episode_id": pair.get("chosen_episode_id"),
        "rejected_episode_id": pair.get("rejected_episode_id"),
        "chosen_score": pair.get("chosen_score"),
        "rejected_score": pair.get("rejected_score"),
        "score_gap": pair.get("score_gap"),
        "chosen_task_completion_status": str((chosen.get("task_completion") or {}).get("status") or "not_applicable"),
        "rejected_task_completion_status": str((rejected.get("task_completion") or {}).get("status") or "not_applicable"),
        "chosen_task_completion_passed": bool((chosen.get("task_completion") or {}).get("passed", True)),
        "rejected_task_completion_passed": bool((rejected.get("task_completion") or {}).get("passed", True)),
        "contract_fingerprint_status": pair.get("contract_fingerprint_status"),
        "contract_fingerprint_scope": pair.get("contract_fingerprint_scope"),
        "contract_fingerprint_reasons": pair.get("contract_fingerprint_reasons"),
        "contract_fingerprints": pair.get("contract_fingerprints"),
        "reason": pair.get("reason"),
    }
    for field_name, expected_value in expected.items():
        if row.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} does not match improvement pair {pair.get('pair_id')!r}.")

def _compare_response_text(view: dict[str, Any]) -> str:
    lines = ["Observed behavior:"]
    task = view.get("task_completion") if isinstance(view.get("task_completion"), dict) else {}
    if task:
        lines.append(
            "- "
            + " ".join(
                [
                    "task_completion",
                    str(task.get("status") or "unknown"),
                    f"checks={task.get('passed_check_count', 0)}/{task.get('required_check_count', 0)}",
                ]
            )
        )
    events = view.get("events") if isinstance(view.get("events"), list) else []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type") or "event")
        if event_type not in {"tool_call", "tool_result", "assistant_message"}:
            continue
        parts = [event_type]
        tool_name = str(event.get("tool_name") or "").strip()
        status = str(event.get("status") or "").strip()
        if tool_name:
            parts.append(tool_name)
        if status:
            parts.append(status)
        detail = _compare_event_detail(event)
        if detail:
            parts.append(detail)
        lines.append("- " + " ".join(parts))
    final_answer = str(view.get("final_answer") or "")
    if final_answer:
        lines.append(f"Final answer: {final_answer}")
    return "\n".join(lines)

def _compare_event_detail(event: dict[str, Any]) -> str:
    for field_name in ("result", "args"):
        value = event.get(field_name)
        if isinstance(value, dict) and value:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
    text = str(event.get("text") or "").strip()
    return text[:500]

def _validate_improvement_card(path: Path, target: ValidationTarget) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        target.errors.append(f"IMPROVEMENT_CARD.md could not be read: {exc}")
        return
    required = [
        "# Flight Recorder Improvement Pair Card",
        "## Summary",
        "## Pairs",
        "## Boundaries",
    ]
    for marker in required:
        if marker not in text:
            target.errors.append(f"IMPROVEMENT_CARD.md missing section marker {marker!r}.")

def _validate_episodes(episodes: list[dict[str, Any]], target: ValidationTarget) -> None:
    seen: set[str] = set()
    for index, episode in enumerate(episodes):
        _require_equal(episode, "schema_version", RL_EPISODE_SCHEMA_VERSION, target, prefix=f"episodes[{index}].")
        episode_id = episode.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            target.errors.append(f"episodes[{index}].episode_id must be a non-empty string.")
            continue
        if episode_id in seen:
            target.errors.append(f"episodes[{index}].episode_id duplicates {episode_id!r}.")
        seen.add(episode_id)
        for field_name in ("scenario_id", "task_family", "final_answer"):
            if not isinstance(episode.get(field_name), str):
                target.errors.append(f"episodes[{index}].{field_name} must be a string.")
        if _looks_absolute(str(episode.get("source_run", ""))):
            target.warnings.append(f"episodes[{index}].source_run is absolute; prefer redacted or relative exports for sharing.")
        if "source_lineage" in episode and _looks_absolute(str(episode.get("source_lineage", ""))):
            target.warnings.append(f"episodes[{index}].source_lineage is absolute; prefer redacted or relative exports for sharing.")
        _validate_source_fingerprint_fields(episode, target, f"episodes[{index}]", warn_if_missing=True)
        if not isinstance(episode.get("events"), list):
            target.errors.append(f"episodes[{index}].events must be a list.")
        if "trace_signal" in episode:
            _validate_trace_signal(
                episode.get("trace_signal"),
                _expected_episode_trace_signal(episode),
                target,
                f"episodes[{index}].trace_signal",
            )
        else:
            target.warnings.append(f"episodes[{index}].trace_signal is missing; rerun export-rl to refresh trace-signal metrics.")
        if "task_completion" in episode:
            _validate_task_completion(episode.get("task_completion"), target, f"episodes[{index}].task_completion")
        else:
            target.warnings.append(f"episodes[{index}].task_completion is missing; rerun export-rl to refresh task evidence fields.")
        if "state_diff" in episode:
            _validate_state_diff_summary(episode.get("state_diff"), target, f"episodes[{index}].state_diff")
        outcome = episode.get("outcome")
        if not isinstance(outcome, dict):
            target.errors.append(f"episodes[{index}].outcome must be an object.")
            continue
        if not _is_int_between(outcome.get("score"), 0, 100):
            target.errors.append(f"episodes[{index}].outcome.score must be an integer from 0 to 100.")
        if not isinstance(outcome.get("passed"), bool):
            target.errors.append(f"episodes[{index}].outcome.passed must be a boolean.")
        if not isinstance(outcome.get("reward"), (int, float)):
            target.errors.append(f"episodes[{index}].outcome.reward must be numeric.")
        if isinstance(episode.get("task_completion"), dict):
            if outcome.get("task_completion_status") != episode["task_completion"].get("status"):
                target.errors.append(f"episodes[{index}].outcome.task_completion_status must match task_completion.status.")
            if outcome.get("task_completion_passed") != episode["task_completion"].get("passed"):
                target.errors.append(f"episodes[{index}].outcome.task_completion_passed must match task_completion.passed.")
        if isinstance(episode.get("state_diff"), dict):
            if outcome.get("state_changed") != episode["state_diff"].get("changed"):
                target.errors.append(f"episodes[{index}].outcome.state_changed must match state_diff.changed.")
            if outcome.get("state_change_count") != episode["state_diff"].get("change_count"):
                target.errors.append(f"episodes[{index}].outcome.state_change_count must match state_diff.change_count.")

def _validate_rewards(rewards: list[dict[str, Any]], target: ValidationTarget, episodes: list[dict[str, Any]]) -> None:
    episode_by_id = {episode.get("episode_id"): episode for episode in episodes if isinstance(episode.get("episode_id"), str)}
    seen: set[str] = set()
    for index, reward in enumerate(rewards):
        _require_equal(reward, "schema_version", RL_REWARD_SCHEMA_VERSION, target, prefix=f"rewards[{index}].")
        episode_id = reward.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            target.errors.append(f"rewards[{index}].episode_id must be a non-empty string.")
            continue
        if episode_id in seen:
            target.errors.append(f"rewards[{index}].episode_id duplicates {episode_id!r}.")
        seen.add(episode_id)
        episode = episode_by_id.get(episode_id)
        if episode is None:
            target.errors.append(f"rewards[{index}].episode_id {episode_id!r} does not reference an episode.")
        else:
            outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
            for field_name in ("score", "passed", "reward"):
                if reward.get(field_name) != outcome.get(field_name):
                    target.errors.append(
                        f"rewards[{index}].{field_name} does not match episode {episode_id!r} outcome."
                    )
            if "task_completion_status" in reward and reward.get("task_completion_status") != outcome.get("task_completion_status"):
                target.errors.append(f"rewards[{index}].task_completion_status does not match episode {episode_id!r} outcome.")
            if "task_completion_passed" in reward and reward.get("task_completion_passed") != outcome.get("task_completion_passed"):
                target.errors.append(f"rewards[{index}].task_completion_passed does not match episode {episode_id!r} outcome.")
            if "state_changed" in reward and reward.get("state_changed") != outcome.get("state_changed"):
                target.errors.append(f"rewards[{index}].state_changed does not match episode {episode_id!r} outcome.")
            if "state_change_count" in reward and reward.get("state_change_count") != outcome.get("state_change_count"):
                target.errors.append(f"rewards[{index}].state_change_count does not match episode {episode_id!r} outcome.")
            _validate_matching_source_fingerprints(reward, episode, target, f"rewards[{index}]")
        _validate_source_fingerprint_fields(reward, target, f"rewards[{index}]", warn_if_missing=False)
        if not isinstance(reward.get("rule_rewards"), list):
            target.errors.append(f"rewards[{index}].rule_rewards must be a list.")
        else:
            for rule_index, rule_reward in enumerate(reward["rule_rewards"]):
                if isinstance(rule_reward, dict) and "evidence_refs" in rule_reward:
                    _validate_evidence_refs(
                        rule_reward.get("evidence_refs"),
                        target,
                        f"rewards[{index}].rule_rewards[{rule_index}].evidence_refs",
                    )
        if not isinstance(reward.get("attribution"), list):
            target.errors.append(f"rewards[{index}].attribution must be a list.")

def _validate_step_rewards(
    step_rewards: list[dict[str, Any]],
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
    rewards: list[dict[str, Any]],
) -> None:
    episode_by_id = {episode.get("episode_id"): episode for episode in episodes if isinstance(episode.get("episode_id"), str)}
    rule_delta_by_key = _rule_reward_deltas_by_key(rewards)
    step_delta_by_key: dict[tuple[str, str], float] = {}
    seen: set[str] = set()
    for index, step_reward in enumerate(step_rewards):
        _require_equal(step_reward, "schema_version", RL_STEP_REWARD_SCHEMA_VERSION, target, prefix=f"step_rewards[{index}].")
        step_reward_id = step_reward.get("step_reward_id")
        if not isinstance(step_reward_id, str) or not step_reward_id:
            target.errors.append(f"step_rewards[{index}].step_reward_id must be a non-empty string.")
        elif step_reward_id in seen:
            target.errors.append(f"step_rewards[{index}].step_reward_id duplicates {step_reward_id!r}.")
        else:
            seen.add(step_reward_id)

        episode_id = step_reward.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            target.errors.append(f"step_rewards[{index}].episode_id must be a non-empty string.")
            episode = None
        else:
            episode = episode_by_id.get(episode_id)
            if episode is None:
                target.errors.append(f"step_rewards[{index}].episode_id {episode_id!r} does not reference an episode.")
            else:
                _validate_matching_source_fingerprints(step_reward, episode, target, f"step_rewards[{index}]")
        _validate_source_fingerprint_fields(step_reward, target, f"step_rewards[{index}]", warn_if_missing=False)

        rule_id = step_reward.get("rule_id")
        for field_name in ("scenario_id", "task_family", "rule_id", "rule_name", "evidence"):
            if not isinstance(step_reward.get(field_name), str):
                target.errors.append(f"step_rewards[{index}].{field_name} must be a string.")
        if step_reward.get("target") not in {"event", "final_answer", "episode", "state_snapshot"}:
            target.errors.append(f"step_rewards[{index}].target must be one of event, final_answer, episode, or state_snapshot.")
        if step_reward.get("reward_scale") not in REWARD_SCALES:
            target.errors.append(f"step_rewards[{index}].reward_scale must be one of {sorted(REWARD_SCALES)!r}.")
        for field_name in ("reward_delta", "rule_reward_delta", "allocation_weight", "episode_reward"):
            if not isinstance(step_reward.get(field_name), (int, float)) or isinstance(step_reward.get(field_name), bool):
                target.errors.append(f"step_rewards[{index}].{field_name} must be numeric.")
        if (
            not isinstance(step_reward.get("attribution_count"), int)
            or isinstance(step_reward.get("attribution_count"), bool)
            or step_reward.get("attribution_count") < 1
        ):
            target.errors.append(f"step_rewards[{index}].attribution_count must be a positive integer.")
        if (
            not isinstance(step_reward.get("allocation_index"), int)
            or isinstance(step_reward.get("allocation_index"), bool)
            or step_reward.get("allocation_index") < 0
        ):
            target.errors.append(f"step_rewards[{index}].allocation_index must be a non-negative integer.")
        if (
            isinstance(step_reward.get("allocation_index"), int)
            and not isinstance(step_reward.get("allocation_index"), bool)
            and isinstance(step_reward.get("attribution_count"), int)
            and not isinstance(step_reward.get("attribution_count"), bool)
            and step_reward["allocation_index"] >= step_reward["attribution_count"]
        ):
            target.errors.append(f"step_rewards[{index}].allocation_index must be less than attribution_count.")
        if not _is_int_between(step_reward.get("score"), 0, 100):
            target.errors.append(f"step_rewards[{index}].score must be an integer from 0 to 100.")
        if not isinstance(step_reward.get("passed"), bool):
            target.errors.append(f"step_rewards[{index}].passed must be a boolean.")
        if not isinstance(step_reward.get("critical"), bool):
            target.errors.append(f"step_rewards[{index}].critical must be a boolean.")
        if not _is_int_between(step_reward.get("penalty"), 0, 100):
            target.errors.append(f"step_rewards[{index}].penalty must be an integer from 0 to 100.")

        if step_reward.get("target") == "event":
            event_index = step_reward.get("event_index")
            if not isinstance(event_index, int) or isinstance(event_index, bool) or event_index < 0:
                target.errors.append(f"step_rewards[{index}].event_index must be a non-negative integer for event targets.")
            elif isinstance(episode, dict) and isinstance(episode.get("events"), list) and event_index >= len(episode["events"]):
                target.errors.append(
                    f"step_rewards[{index}].event_index {event_index} is outside episode {episode_id!r} events."
                )
        elif "event_index" in step_reward:
            target.errors.append(f"step_rewards[{index}].event_index is only valid when target is event.")

        if "evidence_ref" in step_reward:
            _validate_evidence_refs([step_reward.get("evidence_ref")], target, f"step_rewards[{index}].evidence_ref")

        reward_delta = step_reward.get("reward_delta")
        if (
            isinstance(episode_id, str)
            and isinstance(rule_id, str)
            and isinstance(reward_delta, (int, float))
            and not isinstance(reward_delta, bool)
        ):
            key = (episode_id, rule_id)
            step_delta_by_key[key] = round(step_delta_by_key.get(key, 0.0) + float(reward_delta), 6)
            if key not in rule_delta_by_key:
                target.errors.append(f"step_rewards[{index}] does not reference a terminal rule reward.")

    for key, actual in sorted(step_delta_by_key.items()):
        expected = rule_delta_by_key.get(key)
        if expected is not None and round(abs(actual - expected), 6) > 0.000001:
            episode_id, rule_id = key
            target.errors.append(
                f"step_rewards for episode {episode_id!r} rule {rule_id!r} sum to {actual}, expected {expected}."
            )

def _rule_reward_deltas_by_key(rewards: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    deltas: dict[tuple[str, str], float] = {}
    for reward in rewards:
        episode_id = reward.get("episode_id")
        if not isinstance(episode_id, str):
            continue
        rule_rewards = reward.get("rule_rewards")
        if not isinstance(rule_rewards, list):
            continue
        for rule_reward in rule_rewards:
            if not isinstance(rule_reward, dict) or rule_reward.get("passed") is True:
                continue
            rule_id = rule_reward.get("rule_id")
            reward_delta = rule_reward.get("reward_delta")
            if isinstance(rule_id, str) and isinstance(reward_delta, (int, float)) and not isinstance(reward_delta, bool):
                deltas[(episode_id, rule_id)] = round(float(reward_delta), 6)
    return deltas

def _validate_sft_records(
    sft: list[dict[str, Any]],
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
) -> None:
    episode_by_id = {episode.get("episode_id"): episode for episode in episodes if isinstance(episode.get("episode_id"), str)}
    expected_ids = {
        str(episode.get("episode_id"))
        for episode in episodes
        if isinstance(episode.get("episode_id"), str)
        and positive_label_eligible(episode)
    }
    seen: set[str] = set()
    for index, sample in enumerate(sft):
        _require_equal(sample, "schema_version", RL_SFT_SCHEMA_VERSION, target, prefix=f"sft[{index}].")
        episode_id = _validate_training_view_common(sample, target, f"sft[{index}]", episode_by_id)
        if episode_id:
            if episode_id in seen:
                target.errors.append(f"sft[{index}].episode_id duplicates {episode_id!r}.")
            seen.add(episode_id)
            episode = episode_by_id.get(episode_id)
            outcome = episode.get("outcome") if isinstance(episode, dict) and isinstance(episode.get("outcome"), dict) else {}
            if outcome.get("passed") is not True:
                target.errors.append(f"sft[{index}].episode_id {episode_id!r} does not reference a passing episode.")
            if episode is not None and not positive_label_eligible(episode):
                target.errors.append(f"sft[{index}].episode_id {episode_id!r} is not eligible for positive trainer labels.")
            if sample.get("quality_gate") != "passed_scorecard":
                target.errors.append(f"sft[{index}].quality_gate must be 'passed_scorecard'.")
            _validate_training_view_task_completion(sample, episode, target, f"sft[{index}]")
    missing = sorted(expected_ids - seen)
    if missing:
        target.errors.append(f"sft.jsonl missing passing episode samples: {missing!r}.")

def _validate_action_sft_records(
    action_sft: list[dict[str, Any]],
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
) -> None:
    episode_by_id = {episode.get("episode_id"): episode for episode in episodes if isinstance(episode.get("episode_id"), str)}
    expected_ids = {
        str(episode.get("episode_id"))
        for episode in episodes
        if isinstance(episode.get("episode_id"), str)
        and positive_label_eligible(episode)
        and isinstance(episode.get("trajectory_v2"), dict)
        and isinstance(episode["trajectory_v2"].get("action_training"), dict)
        and episode["trajectory_v2"]["action_training"].get("eligible") is True
    }
    seen: set[str] = set()
    for index, sample in enumerate(action_sft):
        label = f"action_sft[{index}]"
        _require_equal(sample, "schema_version", RL_ACTION_SFT_SCHEMA_VERSION, target, prefix=f"{label}.")
        sample_id = sample.get("sample_id")
        episode_id = sample.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            target.errors.append(f"{label}.episode_id must be a non-empty string.")
            continue
        if sample_id != episode_id:
            target.errors.append(f"{label}.sample_id must match episode_id.")
        if episode_id in seen:
            target.errors.append(f"{label}.episode_id duplicates {episode_id!r}.")
        seen.add(episode_id)
        episode = episode_by_id.get(episode_id)
        if episode is None:
            target.errors.append(f"{label}.episode_id {episode_id!r} does not reference an episode.")
            continue
        if not positive_label_eligible(episode):
            target.errors.append(f"{label}.episode_id {episode_id!r} is not eligible for positive trainer labels.")
        outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
        expected = {
            "scenario_id": episode.get("scenario_id"),
            "task_family": episode.get("task_family"),
            "prompt": episode.get("prompt"),
            "response": episode.get("final_answer"),
            "score": outcome.get("score"),
            "reward": outcome.get("reward"),
            "source_artifact": "episodes.jsonl",
            "quality_gate": "passed_scorecard_and_task_completion",
        }
        for field_name, expected_value in expected.items():
            if sample.get(field_name) != expected_value:
                target.errors.append(f"{label}.{field_name} does not match episode {episode_id!r}.")
        _validate_matching_source_fingerprints(sample, episode, target, label)
        _validate_training_view_task_completion(sample, episode, target, label)
        trajectory_v2 = episode.get("trajectory_v2") if isinstance(episode.get("trajectory_v2"), dict) else {}
        if sample.get("trajectory_v2") != trajectory_v2:
            target.errors.append(f"{label}.trajectory_v2 must match the source episode exactly.")
        expected_trajectory_sha = hashlib.sha256(
            json.dumps(trajectory_v2, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if sample.get("trajectory_v2_sha256") != expected_trajectory_sha:
            target.errors.append(f"{label}.trajectory_v2_sha256 does not match trajectory_v2.")
        if sample.get("governance") != trajectory_v2.get("governance"):
            target.errors.append(f"{label}.governance must match trajectory_v2 governance.")
        _validate_agentic_messages(sample, target, label)
    missing = sorted(expected_ids - seen)
    if missing:
        target.errors.append(f"action_sft.jsonl missing eligible episode samples: {missing!r}.")

def _validate_agentic_messages(sample: dict[str, Any], target: ValidationTarget, label: str) -> None:
    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        target.errors.append(f"{label}.messages must contain at least a user prompt and assistant action.")
        return
    first_user = next((index for index, message in enumerate(messages) if isinstance(message, dict) and message.get("role") == "user"), None)
    if first_user is None or any(
        isinstance(message, dict) and message.get("role") not in {"system", "developer"}
        for message in messages[:first_user]
    ):
        target.errors.append(f"{label}.messages must begin with optional system/developer context followed by a user message.")
    if not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
        target.errors.append(f"{label}.messages must end with an assistant message.")
    elif messages[-1].get("content") != sample.get("response"):
        target.errors.append(f"{label}.messages final assistant content must match response.")
    call_ids: set[str] = set()
    called_tools: set[str] = set()
    assistant_actions = 0
    tool_result_count = 0
    for message_index, message in enumerate(messages):
        message_label = f"{label}.messages[{message_index}]"
        if not isinstance(message, dict):
            target.errors.append(f"{message_label} must be an object.")
            continue
        role = message.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            target.errors.append(f"{message_label}.role is not supported.")
        if not isinstance(message.get("content"), str):
            target.errors.append(f"{message_label}.content must be a string.")
        if role == "assistant":
            assistant_actions += 1
            tool_calls = message.get("tool_calls", [])
            if not isinstance(tool_calls, list):
                target.errors.append(f"{message_label}.tool_calls must be a list when present.")
                continue
            for call_index, call in enumerate(tool_calls):
                call_label = f"{message_label}.tool_calls[{call_index}]"
                if not isinstance(call, dict):
                    target.errors.append(f"{call_label} must be an object.")
                    continue
                call_id = call.get("id")
                function = call.get("function")
                if not isinstance(call_id, str) or not call_id:
                    target.errors.append(f"{call_label}.id must be a non-empty string.")
                else:
                    call_ids.add(call_id)
                if call.get("type") != "function" or not isinstance(function, dict):
                    target.errors.append(f"{call_label} must describe a function call.")
                    continue
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    target.errors.append(f"{call_label}.function.name must be a non-empty string.")
                else:
                    called_tools.add(name)
                arguments = function.get("arguments")
                try:
                    parsed = json.loads(arguments) if isinstance(arguments, str) else None
                except json.JSONDecodeError:
                    parsed = None
                if not isinstance(parsed, dict):
                    target.errors.append(f"{call_label}.function.arguments must encode a JSON object.")
        elif role == "tool":
            tool_result_count += 1
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in call_ids:
                target.errors.append(f"{message_label}.tool_call_id must reference a preceding assistant tool call.")
    if sample.get("action_count") != assistant_actions:
        target.errors.append(f"{label}.action_count must match assistant messages.")
    if sample.get("tool_call_count") != len(call_ids):
        target.errors.append(f"{label}.tool_call_count must match unique assistant tool calls.")
    if sample.get("tool_result_count") != tool_result_count:
        target.errors.append(f"{label}.tool_result_count must match tool messages.")
    tools = sample.get("tools")
    if not isinstance(tools, list):
        target.errors.append(f"{label}.tools must be a list.")
        tools = []
    defined_tools = {
        str(tool.get("function", {}).get("name"))
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict) and tool.get("function", {}).get("name")
    }
    if defined_tools != called_tools:
        target.errors.append(f"{label}.tools must define exactly the tools called by messages.")
    expected_provenance = "recorded_exact"
    if sample.get("tool_schema_provenance") != expected_provenance:
        target.errors.append(f"{label}.tool_schema_provenance expected {expected_provenance!r}.")

def _validate_dpo_records(
    dpo: list[dict[str, Any]],
    target: ValidationTarget,
    preferences: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
) -> None:
    preference_by_id = {
        preference.get("preference_id"): preference
        for preference in preferences
        if isinstance(preference.get("preference_id"), str)
    }
    episode_by_id = {episode.get("episode_id"): episode for episode in episodes if isinstance(episode.get("episode_id"), str)}
    seen: set[str] = set()
    for index, pair in enumerate(dpo):
        _require_equal(pair, "schema_version", RL_DPO_SCHEMA_VERSION, target, prefix=f"dpo[{index}].")
        pair_id = pair.get("pair_id")
        preference_id = pair.get("preference_id")
        if not isinstance(pair_id, str) or not pair_id:
            target.errors.append(f"dpo[{index}].pair_id must be a non-empty string.")
        elif pair_id in seen:
            target.errors.append(f"dpo[{index}].pair_id duplicates {pair_id!r}.")
        else:
            seen.add(pair_id)
        if preference_id != pair_id:
            target.errors.append(f"dpo[{index}].preference_id must match pair_id.")
        preference = preference_by_id.get(preference_id)
        if preference is None:
            target.errors.append(f"dpo[{index}].preference_id {preference_id!r} does not reference a preference.")
            continue
        for field_name in ("task_family", "prompt", "chosen", "rejected", "reason", "source_artifact"):
            if not isinstance(pair.get(field_name), str):
                target.errors.append(f"dpo[{index}].{field_name} must be a string.")
        if pair.get("source_artifact") != "preferences.jsonl":
            target.errors.append(f"dpo[{index}].source_artifact must be 'preferences.jsonl'.")
        for field_name in ("chosen_score", "rejected_score", "score_gap"):
            if not _is_int_between(pair.get(field_name), 0, 100):
                target.errors.append(f"dpo[{index}].{field_name} must be an integer from 0 to 100.")
        native_trajectory = pair.get("trajectory_format") == "native_tool_messages" or "tools" in pair
        if native_trajectory:
            chosen_tools = _validate_dpo_trajectory_messages(
                pair.get("chosen_messages"), pair.get("chosen"), target, f"dpo[{index}].chosen_messages"
            )
            rejected_tools = _validate_dpo_trajectory_messages(
                pair.get("rejected_messages"), pair.get("rejected"), target, f"dpo[{index}].rejected_messages"
            )
            tools = pair.get("tools")
            if not isinstance(tools, list):
                target.errors.append(f"dpo[{index}].tools must be a list.")
                tools = []
            defined_tools = {
                str(tool.get("function", {}).get("name"))
                for tool in tools
                if isinstance(tool, dict) and isinstance(tool.get("function"), dict) and tool.get("function", {}).get("name")
            }
            if defined_tools != chosen_tools | rejected_tools:
                target.errors.append(f"dpo[{index}].tools must define exactly the tools used by both trajectories.")
            if pair.get("trajectory_format") != "native_tool_messages":
                target.errors.append(f"dpo[{index}].trajectory_format must be 'native_tool_messages'.")
        else:
            _validate_messages(pair.get("chosen_messages"), target, f"dpo[{index}].chosen_messages")
            _validate_messages(pair.get("rejected_messages"), target, f"dpo[{index}].rejected_messages")
        for field_name in ("chosen_episode_id", "rejected_episode_id"):
            episode_id = pair.get(field_name)
            if not isinstance(episode_id, str) or not episode_id:
                target.errors.append(f"dpo[{index}].{field_name} must be a non-empty string.")
            elif episode_id not in episode_by_id:
                target.errors.append(f"dpo[{index}].{field_name} {episode_id!r} does not reference an episode.")
        _compare_dpo_to_preference(pair, preference, target, index)
    missing = sorted(set(preference_by_id) - seen)
    if missing:
        target.errors.append(f"dpo.jsonl missing preference pairs: {missing!r}.")

def _validate_dpo_trajectory_messages(
    value: Any,
    expected_response: Any,
    target: ValidationTarget,
    label: str,
) -> set[str]:
    if not isinstance(value, list) or len(value) < 2:
        target.errors.append(f"{label} must contain a user prompt and assistant trajectory.")
        return set()
    if not isinstance(value[0], dict) or value[0].get("role") != "user":
        target.errors.append(f"{label} must begin with a user message.")
    if not isinstance(value[-1], dict) or value[-1].get("role") != "assistant":
        target.errors.append(f"{label} must end with an assistant message.")
    elif value[-1].get("content") != expected_response:
        target.errors.append(f"{label} final assistant content must match its response field.")
    call_ids: set[str] = set()
    called_tools: set[str] = set()
    for index, message in enumerate(value):
        message_label = f"{label}[{index}]"
        if not isinstance(message, dict):
            target.errors.append(f"{message_label} must be an object.")
            continue
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            target.errors.append(f"{message_label}.role is not supported.")
        if not isinstance(message.get("content"), str):
            target.errors.append(f"{message_label}.content must be a string.")
        if role == "assistant":
            for call in message.get("tool_calls", []) if isinstance(message.get("tool_calls", []), list) else []:
                if not isinstance(call, dict):
                    continue
                call_id = call.get("id")
                function = call.get("function")
                if isinstance(call_id, str) and call_id:
                    call_ids.add(call_id)
                if isinstance(function, dict) and isinstance(function.get("name"), str) and function.get("name"):
                    called_tools.add(function["name"])
                arguments = function.get("arguments") if isinstance(function, dict) else None
                try:
                    parsed = json.loads(arguments) if isinstance(arguments, str) else None
                except json.JSONDecodeError:
                    parsed = None
                if not isinstance(parsed, dict):
                    target.errors.append(f"{message_label}.tool_calls arguments must encode JSON objects.")
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in call_ids:
                target.errors.append(f"{message_label}.tool_call_id must reference a preceding assistant tool call.")
    return called_tools

def _validate_reward_model_records(
    reward_model: list[dict[str, Any]],
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
) -> None:
    episode_by_id = {episode.get("episode_id"): episode for episode in episodes if isinstance(episode.get("episode_id"), str)}
    expected_ids = {
        str(episode.get("episode_id"))
        for episode in episodes
        if isinstance(episode.get("episode_id"), str)
        and (
            not (isinstance(episode.get("outcome"), dict) and episode["outcome"].get("passed") is True)
            or positive_label_eligible(episode)
        )
    }
    seen: set[str] = set()
    for index, sample in enumerate(reward_model):
        _require_equal(sample, "schema_version", RL_REWARD_MODEL_SCHEMA_VERSION, target, prefix=f"reward_model[{index}].")
        episode_id = _validate_training_view_common(sample, target, f"reward_model[{index}]", episode_by_id)
        if episode_id:
            if episode_id in seen:
                target.errors.append(f"reward_model[{index}].episode_id duplicates {episode_id!r}.")
            seen.add(episode_id)
        if not isinstance(sample.get("passed"), bool):
            target.errors.append(f"reward_model[{index}].passed must be a boolean.")
        episode = episode_by_id.get(episode_id) if episode_id else None
        if isinstance(episode, dict):
            outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
            if outcome.get("passed") is True and not positive_label_eligible(episode):
                target.errors.append(f"reward_model[{index}].episode_id {episode_id!r} is not eligible as a positive reward label.")
            _validate_training_view_task_completion(sample, episode, target, f"reward_model[{index}]")
        for field_name in ("failed_rules", "critical_failures"):
            if not _is_string_list(sample.get(field_name)):
                target.errors.append(f"reward_model[{index}].{field_name} must be a list of strings.")
    missing = sorted(expected_ids - seen)
    if missing:
        target.errors.append(f"reward_model.jsonl missing eligible episode samples: {missing!r}.")

def _validate_training_view_common(
    sample: dict[str, Any],
    target: ValidationTarget,
    label: str,
    episode_by_id: dict[Any, dict[str, Any]],
) -> str | None:
    sample_id = sample.get("sample_id")
    episode_id = sample.get("episode_id")
    if not isinstance(sample_id, str) or not sample_id:
        target.errors.append(f"{label}.sample_id must be a non-empty string.")
    if not isinstance(episode_id, str) or not episode_id:
        target.errors.append(f"{label}.episode_id must be a non-empty string.")
        episode = None
    else:
        episode = episode_by_id.get(episode_id)
        if episode is None:
            target.errors.append(f"{label}.episode_id {episode_id!r} does not reference an episode.")
    if isinstance(sample_id, str) and isinstance(episode_id, str) and sample_id != episode_id:
        target.errors.append(f"{label}.sample_id must match episode_id.")
    for field_name in ("scenario_id", "task_family", "prompt", "response", "source_artifact"):
        if not isinstance(sample.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if sample.get("source_artifact") != "episodes.jsonl":
        target.errors.append(f"{label}.source_artifact must be 'episodes.jsonl'.")
    if not _is_int_between(sample.get("score"), 0, 100):
        target.errors.append(f"{label}.score must be an integer from 0 to 100.")
    if not isinstance(sample.get("reward"), (int, float)) or isinstance(sample.get("reward"), bool):
        target.errors.append(f"{label}.reward must be numeric.")
    _validate_messages(sample.get("messages"), target, f"{label}.messages")
    if isinstance(episode, dict):
        _validate_matching_source_fingerprints(sample, episode, target, label)
        outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
        expected = {
            "scenario_id": episode.get("scenario_id"),
            "task_family": episode.get("task_family"),
            "prompt": episode.get("prompt"),
            "response": episode.get("final_answer"),
            "score": outcome.get("score"),
            "reward": outcome.get("reward"),
        }
        for field_name, expected_value in expected.items():
            if sample.get(field_name) != expected_value:
                target.errors.append(f"{label}.{field_name} does not match episode {episode_id!r}.")
    return episode_id if isinstance(episode_id, str) and episode_id else None

def _validate_training_view_task_completion(
    sample: dict[str, Any],
    episode: dict[str, Any] | None,
    target: ValidationTarget,
    label: str,
) -> None:
    if not isinstance(episode, dict):
        return
    task = episode.get("task_completion") if isinstance(episode.get("task_completion"), dict) else {}
    if "task_completion_status" in sample and sample.get("task_completion_status") != task.get("status"):
        target.errors.append(f"{label}.task_completion_status does not match episode task_completion.status.")
    if "task_completion_passed" in sample and sample.get("task_completion_passed") != task.get("passed"):
        target.errors.append(f"{label}.task_completion_passed does not match episode task_completion.passed.")

def _compare_dpo_to_preference(
    pair: dict[str, Any],
    preference: dict[str, Any],
    target: ValidationTarget,
    index: int,
) -> None:
    expected = {
        "task_family": preference.get("task_family"),
        "prompt": preference.get("prompt"),
        "chosen_episode_id": preference.get("chosen_episode_id"),
        "rejected_episode_id": preference.get("rejected_episode_id"),
        "chosen_score": preference.get("chosen_score"),
        "rejected_score": preference.get("rejected_score"),
        "score_gap": preference.get("score_gap"),
        "reason": preference.get("reason"),
    }
    chosen = preference.get("chosen") if isinstance(preference.get("chosen"), dict) else {}
    rejected = preference.get("rejected") if isinstance(preference.get("rejected"), dict) else {}
    expected["chosen"] = chosen.get("final_answer")
    expected["rejected"] = rejected.get("final_answer")
    expected["chosen_source_fingerprint_status"] = chosen.get("source_fingerprint_status")
    expected["rejected_source_fingerprint_status"] = rejected.get("source_fingerprint_status")
    expected["chosen_source_fingerprints"] = chosen.get("source_fingerprints")
    expected["rejected_source_fingerprints"] = rejected.get("source_fingerprints")
    for field_name, expected_value in expected.items():
        if pair.get(field_name) != expected_value:
            target.errors.append(f"dpo[{index}].{field_name} does not match preference {preference.get('preference_id')!r}.")

def _validate_source_fingerprint_fields(
    row: dict[str, Any],
    target: ValidationTarget,
    label: str,
    *,
    warn_if_missing: bool,
) -> None:
    has_status = "source_fingerprint_status" in row
    has_fingerprints = "source_fingerprints" in row
    if not has_status and not has_fingerprints:
        if warn_if_missing:
            target.warnings.append(f"{label}.source_fingerprints is missing; rerun export-rl to refresh provenance fields.")
        return
    status = row.get("source_fingerprint_status")
    if status not in {"verified", "unverified"}:
        target.errors.append(f"{label}.source_fingerprint_status must be verified or unverified.")
    fingerprints = row.get("source_fingerprints")
    if not isinstance(fingerprints, dict):
        target.errors.append(f"{label}.source_fingerprints must be an object.")
        return
    scenario_verified = _validate_source_fingerprint_record(fingerprints.get("scenario"), target, f"{label}.source_fingerprints.scenario")
    trace_verified = _validate_source_fingerprint_record(
        fingerprints.get("source_trace"),
        target,
        f"{label}.source_fingerprints.source_trace",
    )
    if status == "verified" and not (scenario_verified and trace_verified):
        target.errors.append(f"{label}.source_fingerprint_status verified requires scenario and source_trace SHA-256 and size_bytes values.")
    if status == "unverified" and scenario_verified and trace_verified:
        target.errors.append(f"{label}.source_fingerprint_status should be verified when both source hashes and sizes are present.")

def _validate_source_fingerprint_record(value: Any, target: ValidationTarget, label: str) -> bool:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return False
    path = value.get("path")
    if path is not None and not isinstance(path, str):
        target.errors.append(f"{label}.path must be a string or null.")
    sha = value.get("sha256")
    if sha is not None and not _is_sha256(sha):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string or null.")
    size_bytes = value.get("size_bytes")
    if size_bytes is not None and not _is_non_negative_int(size_bytes):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer or null.")
    exists = value.get("exists")
    if exists is not None and not isinstance(exists, bool):
        target.errors.append(f"{label}.exists must be a boolean or null.")
    return _source_fingerprint_record_complete(value)

def _validate_matching_source_fingerprints(
    row: dict[str, Any],
    episode: dict[str, Any],
    target: ValidationTarget,
    label: str,
) -> None:
    row_has = "source_fingerprint_status" in row or "source_fingerprints" in row
    episode_has = "source_fingerprint_status" in episode or "source_fingerprints" in episode
    if episode_has and not row_has:
        target.errors.append(f"{label}.source_fingerprints missing while referenced episode has source fingerprints.")
        return
    if not row_has:
        return
    _validate_source_fingerprint_fields(row, target, label, warn_if_missing=False)
    if row.get("source_fingerprint_status") != episode.get("source_fingerprint_status"):
        target.errors.append(f"{label}.source_fingerprint_status does not match episode {episode.get('episode_id')!r}.")
    if row.get("source_fingerprints") != episode.get("source_fingerprints"):
        target.errors.append(f"{label}.source_fingerprints does not match episode {episode.get('episode_id')!r}.")

def _validate_contract_fingerprint_status(row: dict[str, Any], target: ValidationTarget, label: str) -> None:
    status = row.get("contract_fingerprint_status")
    if status not in {"matched", "drifted", "unverified"}:
        target.errors.append(f"{label}.contract_fingerprint_status must be matched, drifted, or unverified.")
    if "contract_fingerprint_scope" in row and row.get("contract_fingerprint_scope") not in CONTRACT_SCOPES:
        target.errors.append(f"{label}.contract_fingerprint_scope must be one of {sorted(CONTRACT_SCOPES)!r}.")
    reasons = row.get("contract_fingerprint_reasons")
    if not _is_string_list(reasons):
        target.errors.append(f"{label}.contract_fingerprint_reasons must be a list of strings.")
        reasons = []
    if status in {"drifted", "unverified"} and not reasons:
        target.errors.append(f"{label}.contract_fingerprint_reasons must explain non-matched contract status.")
    if status == "matched" and reasons:
        target.errors.append(f"{label}.contract_fingerprint_reasons must be empty when status is matched.")
    fingerprints = row.get("contract_fingerprints")
    if not isinstance(fingerprints, dict):
        target.errors.append(f"{label}.contract_fingerprints must be an object.")
        return
    for side in ("baseline", "candidate"):
        value = fingerprints.get(side)
        if not isinstance(value, dict):
            target.errors.append(f"{label}.contract_fingerprints.{side} must be an object.")
            continue
        _validate_contract_fingerprint_inputs(value, target, f"{label}.contract_fingerprints.{side}")

def _validate_contract_fingerprint_inputs(value: dict[str, Any], target: ValidationTarget, label: str) -> None:
    for name in ("scenario", "source_trace"):
        record = value.get(name)
        if not isinstance(record, dict):
            target.errors.append(f"{label}.{name} must be an object.")
            continue
        path = record.get("path")
        if path is not None and not isinstance(path, str):
            target.errors.append(f"{label}.{name}.path must be a string or null.")
        sha = record.get("sha256")
        if sha is not None and not _is_sha256(sha):
            target.errors.append(f"{label}.{name}.sha256 must be a SHA-256 hex string or null.")
        if "size_bytes" not in record:
            target.errors.append(f"{label}.{name}.size_bytes is required.")
        size_bytes = record.get("size_bytes")
        if size_bytes is not None and not _is_non_negative_int(size_bytes):
            target.errors.append(f"{label}.{name}.size_bytes must be a non-negative integer or null.")

def _validate_source_fingerprint_coverage(value: Any, target: ValidationTarget, episodes: list[dict[str, Any]]) -> None:
    if not isinstance(value, dict):
        target.errors.append("dataset_metrics.source_fingerprint_coverage must be an object.")
        return
    expected = _expected_source_fingerprint_coverage(episodes)
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(
                f"dataset_metrics.source_fingerprint_coverage.{field_name} expected {expected_value}, got {value.get(field_name)!r}."
            )

def _expected_source_fingerprint_coverage(episodes: list[dict[str, Any]]) -> dict[str, int]:
    with_scenario = 0
    with_trace = 0
    fully_verified = 0
    for episode in episodes:
        fingerprints = episode.get("source_fingerprints") if isinstance(episode.get("source_fingerprints"), dict) else {}
        scenario = fingerprints.get("scenario") if isinstance(fingerprints.get("scenario"), dict) else {}
        source_trace = fingerprints.get("source_trace") if isinstance(fingerprints.get("source_trace"), dict) else {}
        scenario_sha = scenario.get("sha256")
        trace_sha = source_trace.get("sha256")
        if _is_sha256(scenario_sha):
            with_scenario += 1
        if _is_sha256(trace_sha):
            with_trace += 1
        if _source_fingerprint_record_complete(scenario) and _source_fingerprint_record_complete(source_trace):
            fully_verified += 1
    return {
        "episodes": len(episodes),
        "with_scenario_sha256": with_scenario,
        "with_source_trace_sha256": with_trace,
        "fully_verified": fully_verified,
        "unverified": len(episodes) - fully_verified,
    }

def _validate_trainer_view_source_fingerprint_coverage(
    value: Any,
    target: ValidationTarget,
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
    action_sft: list[dict[str, Any]] | None,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("dataset_metrics.trainer_view_source_fingerprint_coverage must be an object.")
        return
    expected = _expected_trainer_view_source_fingerprint_coverage(sft, dpo, reward_model, action_sft)
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(
                "dataset_metrics.trainer_view_source_fingerprint_coverage."
                f"{field_name} expected {expected_value!r}, got {value.get(field_name)!r}."
            )

def _expected_trainer_view_source_fingerprint_coverage(
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
    action_sft: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    action_sft_rows = action_sft or []
    sft_verified = sum(1 for row in sft if _row_source_fingerprints_verified(row))
    action_sft_verified = sum(1 for row in action_sft_rows if _row_source_fingerprints_verified(row))
    dpo_verified = sum(1 for row in dpo if _dpo_source_fingerprints_verified(row))
    reward_model_verified = sum(1 for row in reward_model if _row_source_fingerprints_verified(row))
    row_count = len(sft) + len(action_sft_rows) + len(dpo) + len(reward_model)
    fully_verified = sft_verified + action_sft_verified + dpo_verified + reward_model_verified
    coverage = {
        "rows": row_count,
        "sft_rows": len(sft),
        "dpo_rows": len(dpo),
        "reward_model_rows": len(reward_model),
        "fully_verified": fully_verified,
        "unverified": row_count - fully_verified,
        "fully_verified_rate": round(fully_verified / row_count, 4) if row_count else 0.0,
    }
    if action_sft is not None:
        coverage["action_sft_rows"] = len(action_sft_rows)
    return coverage

def _row_source_fingerprints_verified(row: dict[str, Any]) -> bool:
    fingerprints = row.get("source_fingerprints") if isinstance(row.get("source_fingerprints"), dict) else {}
    return (
        row.get("source_fingerprint_status") == "verified"
        and _fingerprints_have_source_evidence(fingerprints)
    )

def _dpo_source_fingerprints_verified(row: dict[str, Any]) -> bool:
    return _paired_source_fingerprints_verified(row, "chosen") and _paired_source_fingerprints_verified(row, "rejected")

def _paired_source_fingerprints_verified(row: dict[str, Any], side: str) -> bool:
    fingerprints_key = f"{side}_source_fingerprints"
    status_key = f"{side}_source_fingerprint_status"
    fingerprints = row.get(fingerprints_key) if isinstance(row.get(fingerprints_key), dict) else {}
    return row.get(status_key) == "verified" and _fingerprints_have_source_evidence(fingerprints)

def _fingerprints_have_source_evidence(fingerprints: dict[str, Any]) -> bool:
    scenario = fingerprints.get("scenario") if isinstance(fingerprints.get("scenario"), dict) else {}
    source_trace = fingerprints.get("source_trace") if isinstance(fingerprints.get("source_trace"), dict) else {}
    return _source_fingerprint_record_complete(scenario) and _source_fingerprint_record_complete(source_trace)

def _source_fingerprint_record_complete(value: Any) -> bool:
    return isinstance(value, dict) and _is_sha256(value.get("sha256")) and _is_non_negative_int(value.get("size_bytes"))

def _validate_messages(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        target.errors.append(f"{label} must be a two-message user/assistant list.")
        return
    expected_roles = ["user", "assistant"]
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            target.errors.append(f"{label}[{index}] must be an object.")
            continue
        if message.get("role") != expected_roles[index]:
            target.errors.append(f"{label}[{index}].role must be {expected_roles[index]!r}.")
        if not isinstance(message.get("content"), str):
            target.errors.append(f"{label}[{index}].content must be a string.")

def _validate_dataset_metrics(
    metrics: dict[str, Any],
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
    rewards: list[dict[str, Any]],
    step_rewards: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    failure_modes: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    action_sft: list[dict[str, Any]] | None,
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
    dataset_splits: dict[str, Any] | None,
    expected_redaction_status: dict[str, Any],
    expected_label_provenance: dict[str, Any],
) -> None:
    _require_equal(metrics, "schema_version", RL_DATASET_METRICS_SCHEMA_VERSION, target, prefix="dataset_metrics.")
    artifact_counts = metrics.get("artifact_counts")
    if not isinstance(artifact_counts, dict):
        target.errors.append("dataset_metrics.artifact_counts must be an object.")
        artifact_counts = {}
    expected_counts = {
        "episodes": len(episodes),
        "rewards": len(rewards),
        "step_rewards": len(step_rewards),
        "preferences": len(preferences),
        "failure_modes": len(failure_modes),
        "sft": len(sft),
        "dpo": len(dpo),
        "reward_model": len(reward_model),
    }
    if action_sft is not None:
        expected_counts["action_sft"] = len(action_sft)
    for field_name, expected in expected_counts.items():
        if artifact_counts.get(field_name) != expected:
            target.errors.append(f"dataset_metrics.artifact_counts.{field_name} expected {expected}, got {artifact_counts.get(field_name)!r}.")

    scores = [_score_value(episode.get("outcome", {}).get("score")) for episode in episodes if isinstance(episode.get("outcome"), dict)]
    reward_values = [
        float(reward.get("reward"))
        for reward in rewards
        if isinstance(reward.get("reward"), (int, float)) and not isinstance(reward.get("reward"), bool)
    ]
    passed = sum(1 for episode in episodes if isinstance(episode.get("outcome"), dict) and episode["outcome"].get("passed") is True)
    failed = len(episodes) - passed
    expected_scalars = {
        "episode_count": len(episodes),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / len(episodes), 4) if episodes else 0.0,
        "average_score": _average_number(scores),
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "average_reward": _average_number(reward_values),
        "min_reward": min(reward_values) if reward_values else None,
        "max_reward": max(reward_values) if reward_values else None,
    }
    for field_name, expected in expected_scalars.items():
        if metrics.get(field_name) != expected:
            target.errors.append(f"dataset_metrics.{field_name} expected {expected!r}, got {metrics.get(field_name)!r}.")

    expected_failed = _count_strings(rule for episode in episodes for rule in _outcome_strings(episode, "failed_rules"))
    expected_critical = _count_strings(rule for episode in episodes for rule in _outcome_strings(episode, "critical_failures"))
    if _count_rows(metrics.get("failed_rule_counts")) != expected_failed:
        target.errors.append("dataset_metrics.failed_rule_counts does not match episode failed_rules.")
    if _count_rows(metrics.get("critical_failure_counts")) != expected_critical:
        target.errors.append("dataset_metrics.critical_failure_counts does not match episode critical_failures.")

    if "source_fingerprint_coverage" in metrics:
        _validate_source_fingerprint_coverage(metrics.get("source_fingerprint_coverage"), target, episodes)
    else:
        target.warnings.append("dataset_metrics.source_fingerprint_coverage is missing; rerun export-rl to refresh provenance metrics.")
    if "trainer_view_source_fingerprint_coverage" in metrics:
        _validate_trainer_view_source_fingerprint_coverage(
            metrics.get("trainer_view_source_fingerprint_coverage"),
            target,
            sft,
            dpo,
            reward_model,
            action_sft,
        )
    else:
        target.warnings.append(
            "dataset_metrics.trainer_view_source_fingerprint_coverage is missing; rerun export-rl to refresh trainer-view provenance metrics."
        )
    if "task_completion" in metrics:
        expected_task_completion = _expected_task_completion_metrics(episodes)
        actual_task_completion = metrics.get("task_completion")
        if not isinstance(actual_task_completion, dict):
            target.errors.append("dataset_metrics.task_completion must be an object.")
        else:
            for field_name, expected in expected_task_completion.items():
                if actual_task_completion.get(field_name) != expected:
                    target.errors.append(
                        f"dataset_metrics.task_completion.{field_name} expected {expected!r}, got {actual_task_completion.get(field_name)!r}."
                    )
    else:
        target.warnings.append("dataset_metrics.task_completion is missing; rerun export-rl to refresh task-completion metrics.")
    if "trace_signal" in metrics:
        _validate_trace_signal_metrics(
            metrics.get("trace_signal"),
            _expected_trace_signal_metrics(episodes),
            target,
            "dataset_metrics.trace_signal",
        )
    else:
        target.warnings.append("dataset_metrics.trace_signal is missing; rerun export-rl to refresh trace-signal metrics.")
    if "dataset_splits" in metrics:
        if not isinstance(metrics.get("dataset_splits"), dict):
            target.errors.append("dataset_metrics.dataset_splits must be an object.")
        elif dataset_splits is not None and metrics.get("dataset_splits") != dataset_splits.get("summary"):
            target.errors.append("dataset_metrics.dataset_splits must match dataset_splits.summary.")
    else:
        target.warnings.append("dataset_metrics.dataset_splits is missing; rerun export-rl to refresh split metrics.")
    if "redaction_status" in metrics:
        if metrics.get("redaction_status") != expected_redaction_status:
            target.errors.append("dataset_metrics.redaction_status must match recomputed redaction scan.")
    else:
        target.warnings.append("dataset_metrics.redaction_status is missing; rerun export-rl to emit redaction proof.")
    if "label_provenance" in metrics:
        if metrics.get("label_provenance") != expected_label_provenance:
            target.errors.append("dataset_metrics.label_provenance must match recomputed label provenance.")
    else:
        target.warnings.append("dataset_metrics.label_provenance is missing; rerun export-rl to emit label provenance.")
    _validate_dataset_family_metrics(metrics.get("task_families"), target, episodes, step_rewards, failure_modes, sft, dpo, reward_model)
    _validate_quality_flags(metrics.get("quality_flags"), target)
    if "metadata" in metrics:
        _validate_metadata(metrics.get("metadata"), target, "dataset_metrics.metadata")
    if not _is_string_list(metrics.get("recommended_checks")):
        target.errors.append("dataset_metrics.recommended_checks must be a list of strings.")

def _validate_dataset_family_metrics(
    value: Any,
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
    step_rewards: list[dict[str, Any]],
    failure_modes: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
) -> None:
    if not isinstance(value, list):
        target.errors.append("dataset_metrics.task_families must be a list.")
        return
    expected = _expected_dataset_family_metrics(episodes, step_rewards, failure_modes, sft, dpo, reward_model)
    actual_families: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            target.errors.append(f"dataset_metrics.task_families[{index}] must be an object.")
            continue
        family = row.get("task_family")
        if not isinstance(family, str) or not family:
            target.errors.append(f"dataset_metrics.task_families[{index}].task_family must be a non-empty string.")
            continue
        actual_families.add(family)
        expected_row = expected.get(family)
        if expected_row is None:
            target.errors.append(f"dataset_metrics.task_families[{index}] has unknown task_family {family!r}.")
            continue
        for field_name, expected_value in expected_row.items():
            if row.get(field_name) != expected_value:
                target.errors.append(
                    f"dataset_metrics.task_families[{index}].{field_name} expected {expected_value!r}, got {row.get(field_name)!r}."
                )
    missing = sorted(set(expected) - actual_families)
    if missing:
        target.errors.append(f"dataset_metrics.task_families missing families: {missing!r}.")

def _expected_dataset_family_metrics(
    episodes: list[dict[str, Any]],
    step_rewards: list[dict[str, Any]],
    failure_modes: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    reward_model: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    families = sorted(
        {
            str(item.get("task_family") or "unknown")
            for source in (episodes, step_rewards, failure_modes, sft, dpo, reward_model)
            for item in source
            if isinstance(item, dict)
        }
    )
    expected: dict[str, dict[str, Any]] = {}
    for family in families:
        family_episodes = [episode for episode in episodes if str(episode.get("task_family") or "unknown") == family]
        scores = [
            _score_value(episode.get("outcome", {}).get("score"))
            for episode in family_episodes
            if isinstance(episode.get("outcome"), dict)
        ]
        passed = sum(1 for episode in family_episodes if isinstance(episode.get("outcome"), dict) and episode["outcome"].get("passed") is True)
        task_metrics = _expected_task_completion_metrics(family_episodes)
        trace_metrics = _expected_trace_signal_metrics(family_episodes)
        expected[family] = {
            "task_family": family,
            "episode_count": len(family_episodes),
            "passed": passed,
            "failed": len(family_episodes) - passed,
            "pass_rate": round(passed / len(family_episodes), 4) if family_episodes else 0.0,
            "task_completion_configured": task_metrics["configured_count"],
            "task_completion_complete": task_metrics["complete_count"],
            "task_completion_incomplete": task_metrics["incomplete_count"],
            "trace_average_event_count": trace_metrics["average_event_count"],
            "trace_event_type_count": trace_metrics["event_type_count"],
            "trace_tool_or_api_episode_rate": trace_metrics["tool_or_api_episode_rate"],
            "trace_empty_final_answer_count": trace_metrics["empty_final_answer_count"],
            "trace_risk_count": trace_metrics["risk_count"],
            "average_score": _average_number(scores),
            "step_reward_count": _count_family(step_rewards, family),
            "failure_mode_count": _count_family(failure_modes, family),
            "sft_count": _count_family(sft, family),
            "dpo_count": _count_family(dpo, family),
            "reward_model_count": _count_family(reward_model, family),
        }
    return expected

def _expected_task_completion_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = {"complete": 0, "incomplete": 0, "not_applicable": 0, "unknown": 0}
    configured = 0
    required_checks = 0
    passed_checks = 0
    for episode in episodes:
        task = episode.get("task_completion") if isinstance(episode.get("task_completion"), dict) else {}
        status = str(task.get("status") or "unknown")
        if status not in statuses:
            status = "unknown"
        statuses[status] += 1
        if task.get("task_evidence_configured") is True:
            configured += 1
        if _is_non_negative_int(task.get("required_check_count")):
            required_checks += int(task["required_check_count"])
        if _is_non_negative_int(task.get("passed_check_count")):
            passed_checks += int(task["passed_check_count"])
    return {
        "episode_count": len(episodes),
        "configured_count": configured,
        "complete_count": statuses["complete"],
        "incomplete_count": statuses["incomplete"],
        "not_applicable_count": statuses["not_applicable"],
        "unknown_count": statuses["unknown"],
        "required_check_count": required_checks,
        "passed_check_count": passed_checks,
        "check_pass_rate": round(passed_checks / required_checks, 4) if required_checks else 0.0,
    }

def _expected_episode_trace_signal(episode: dict[str, Any]) -> dict[str, Any]:
    events = episode.get("events") if isinstance(episode.get("events"), list) else []
    final_answer = episode.get("final_answer") if isinstance(episode.get("final_answer"), str) else ""
    return build_trace_signal(events, final_answer)

def _validate_trace_signal(value: Any, expected: dict[str, Any], target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    for field_name in (
        "event_count",
        "event_type_count",
        "final_answer_chars",
        "tool_call_count",
        "tool_result_count",
        "api_call_count",
        "subagent_event_count",
        "approval_event_count",
    ):
        if value.get(field_name) != expected[field_name]:
            target.errors.append(f"{label}.{field_name} expected {expected[field_name]!r}, got {value.get(field_name)!r}.")
    for field_name in ("has_final_answer", "has_tool_or_api_events"):
        if value.get(field_name) != expected[field_name]:
            target.errors.append(f"{label}.{field_name} expected {expected[field_name]!r}, got {value.get(field_name)!r}.")
    if _count_rows(value.get("event_types")) != _count_rows(expected.get("event_types")):
        target.errors.append(f"{label}.event_types does not match episode events.")
    if value.get("risks") != expected.get("risks"):
        target.errors.append(f"{label}.risks expected {expected.get('risks')!r}, got {value.get('risks')!r}.")

def _expected_trace_signal_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    signals = [_expected_episode_trace_signal(episode) for episode in episodes]
    event_counts = [_non_negative_int_value(signal.get("event_count")) for signal in signals]
    event_type_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    for signal in signals:
        _merge_count_rows_quiet(event_type_counts, signal.get("event_types"))
        for risk in signal.get("risks", []):
            if isinstance(risk, str) and risk:
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
    episode_count = len(signals)
    with_final = sum(1 for signal in signals if signal.get("has_final_answer") is True)
    with_tool_or_api = sum(1 for signal in signals if signal.get("has_tool_or_api_events") is True)
    return {
        "episode_count": episode_count,
        "total_event_count": sum(event_counts),
        "average_event_count": round(sum(event_counts) / episode_count, 2) if episode_count else 0.0,
        "min_event_count": min(event_counts) if event_counts else 0,
        "max_event_count": max(event_counts) if event_counts else 0,
        "event_type_count": len(event_type_counts),
        "event_type_counts": event_type_counts,
        "episodes_with_final_answer": with_final,
        "empty_final_answer_count": episode_count - with_final,
        "final_answer_rate": round(with_final / episode_count, 4) if episode_count else 0.0,
        "episodes_with_tool_or_api_events": with_tool_or_api,
        "tool_or_api_episode_rate": round(with_tool_or_api / episode_count, 4) if episode_count else 0.0,
        "tool_call_count": sum(_non_negative_int_value(signal.get("tool_call_count")) for signal in signals),
        "tool_result_count": sum(_non_negative_int_value(signal.get("tool_result_count")) for signal in signals),
        "api_call_count": sum(_non_negative_int_value(signal.get("api_call_count")) for signal in signals),
        "subagent_event_count": sum(_non_negative_int_value(signal.get("subagent_event_count")) for signal in signals),
        "approval_event_count": sum(_non_negative_int_value(signal.get("approval_event_count")) for signal in signals),
        "risk_count": sum(risk_counts.values()),
        "risk_counts": risk_counts,
    }

def _validate_trace_signal_metrics(value: Any, expected: dict[str, Any], target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    for field_name in (
        "episode_count",
        "total_event_count",
        "average_event_count",
        "min_event_count",
        "max_event_count",
        "event_type_count",
        "episodes_with_final_answer",
        "empty_final_answer_count",
        "final_answer_rate",
        "episodes_with_tool_or_api_events",
        "tool_or_api_episode_rate",
        "tool_call_count",
        "tool_result_count",
        "api_call_count",
        "subagent_event_count",
        "approval_event_count",
        "risk_count",
    ):
        if value.get(field_name) != expected[field_name]:
            target.errors.append(f"{label}.{field_name} expected {expected[field_name]!r}, got {value.get(field_name)!r}.")
    if _count_rows(value.get("event_type_counts")) != expected["event_type_counts"]:
        target.errors.append(f"{label}.event_type_counts does not match episode trace_signal.")
    if _count_rows(value.get("risk_counts")) != expected["risk_counts"]:
        target.errors.append(f"{label}.risk_counts does not match episode trace_signal.")

def _merge_count_rows_quiet(counts: dict[str, int], rows: Any) -> None:
    if not isinstance(rows, list):
        return
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("id"), str) and isinstance(row.get("count"), int):
            counts[row["id"]] = counts.get(row["id"], 0) + row["count"]

def _validate_quality_flags(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, list):
        target.errors.append("dataset_metrics.quality_flags must be a list.")
        return
    seen: set[str] = set()
    for index, flag in enumerate(value):
        if not isinstance(flag, dict):
            target.errors.append(f"dataset_metrics.quality_flags[{index}] must be an object.")
            continue
        flag_id = flag.get("id")
        if not isinstance(flag_id, str) or not flag_id:
            target.errors.append(f"dataset_metrics.quality_flags[{index}].id must be a non-empty string.")
        elif flag_id in seen:
            target.errors.append(f"dataset_metrics.quality_flags[{index}].id duplicates {flag_id!r}.")
        else:
            seen.add(flag_id)
        if flag.get("severity") not in {"info", "warning", "error"}:
            target.errors.append(f"dataset_metrics.quality_flags[{index}].severity must be info, warning, or error.")
        if not isinstance(flag.get("message"), str) or not flag.get("message"):
            target.errors.append(f"dataset_metrics.quality_flags[{index}].message must be a non-empty string.")

def _validate_dataset_card(path: Path, target: ValidationTarget) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        target.errors.append(f"DATASET_CARD.md could not be read: {exc}")
        return
    required = [
        "# Flight Recorder Dataset Card",
        "## Summary",
        "## Source Fingerprints",
        "## Trace Signal",
        "## Dataset Splits",
        "## Artifact Counts",
        "## Task Families",
        "## Quality Flags",
        "## Boundaries",
    ]
    for marker in required:
        if marker not in text:
            target.errors.append(f"DATASET_CARD.md missing section marker {marker!r}.")

def _validate_preferences(preferences: list[dict[str, Any]], target: ValidationTarget, episodes: list[dict[str, Any]]) -> None:
    episode_by_id = {episode.get("episode_id"): episode for episode in episodes if isinstance(episode.get("episode_id"), str)}
    for index, preference in enumerate(preferences):
        _require_equal(preference, "schema_version", RL_PREFERENCE_SCHEMA_VERSION, target, prefix=f"preferences[{index}].")
        chosen_id = preference.get("chosen_episode_id")
        rejected_id = preference.get("rejected_episode_id")
        chosen = episode_by_id.get(chosen_id)
        rejected = episode_by_id.get(rejected_id)
        if chosen is None:
            target.errors.append(f"preferences[{index}].chosen_episode_id {chosen_id!r} does not reference an episode.")
        elif not positive_label_eligible(chosen):
            target.errors.append(f"preferences[{index}].chosen_episode_id {chosen_id!r} is not eligible for a positive preference label.")
        if rejected is None:
            target.errors.append(f"preferences[{index}].rejected_episode_id {rejected_id!r} does not reference an episode.")
        if chosen is not None and rejected is not None:
            chosen_score = chosen.get("outcome", {}).get("score") if isinstance(chosen.get("outcome"), dict) else None
            rejected_score = rejected.get("outcome", {}).get("score") if isinstance(rejected.get("outcome"), dict) else None
            if preference.get("chosen_score") != chosen_score:
                target.errors.append(f"preferences[{index}].chosen_score does not match chosen episode.")
            if preference.get("rejected_score") != rejected_score:
                target.errors.append(f"preferences[{index}].rejected_score does not match rejected episode.")
            if isinstance(chosen_score, int) and isinstance(rejected_score, int):
                expected_gap = chosen_score - rejected_score
                if preference.get("score_gap") != expected_gap:
                    target.errors.append(f"preferences[{index}].score_gap expected {expected_gap}, got {preference.get('score_gap')!r}.")
                if expected_gap <= 0:
                    target.errors.append(f"preferences[{index}] must prefer a strictly higher-scoring episode.")
            if chosen.get("task_family") != rejected.get("task_family"):
                target.errors.append(f"preferences[{index}] chosen/rejected task families differ.")
            if preference.get("task_family") != chosen.get("task_family"):
                target.errors.append(f"preferences[{index}].task_family does not match chosen episode.")

def _validate_failure_modes(
    failure_modes: list[dict[str, Any]],
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
) -> None:
    episode_ids = {episode.get("episode_id") for episode in episodes if isinstance(episode.get("episode_id"), str)}
    seen: set[str] = set()
    for index, failure in enumerate(failure_modes):
        _require_equal(failure, "schema_version", RL_FAILURE_MODE_SCHEMA_VERSION, target, prefix=f"failure_modes[{index}].")
        failure_id = failure.get("failure_id")
        if not isinstance(failure_id, str) or not failure_id:
            target.errors.append(f"failure_modes[{index}].failure_id must be a non-empty string.")
        elif failure_id in seen:
            target.errors.append(f"failure_modes[{index}].failure_id duplicates {failure_id!r}.")
        else:
            seen.add(failure_id)
        episode_id = failure.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            target.errors.append(f"failure_modes[{index}].episode_id must be a non-empty string.")
        elif episode_id not in episode_ids:
            target.errors.append(f"failure_modes[{index}].episode_id {episode_id!r} does not reference an episode.")
        for field_name in ("scenario_id", "task_family", "rule_id", "rule_name", "summary"):
            if not isinstance(failure.get(field_name), str):
                target.errors.append(f"failure_modes[{index}].{field_name} must be a string.")
        if not isinstance(failure.get("critical"), bool):
            target.errors.append(f"failure_modes[{index}].critical must be a boolean.")
        if not _is_int_between(failure.get("penalty"), 0, 100):
            target.errors.append(f"failure_modes[{index}].penalty must be an integer from 0 to 100.")
        if not _is_int_between(failure.get("score"), 0, 100):
            target.errors.append(f"failure_modes[{index}].score must be an integer from 0 to 100.")
        if not isinstance(failure.get("reward"), (int, float)):
            target.errors.append(f"failure_modes[{index}].reward must be numeric.")
        if not isinstance(failure.get("evidence"), list):
            target.errors.append(f"failure_modes[{index}].evidence must be a list.")
        if "evidence_refs" in failure:
            _validate_evidence_refs(failure.get("evidence_refs"), target, f"failure_modes[{index}].evidence_refs")
        if not isinstance(failure.get("attribution"), list):
            target.errors.append(f"failure_modes[{index}].attribution must be a list.")
        episode = next((episode for episode in episodes if episode.get("episode_id") == episode_id), None)
        if isinstance(episode, dict):
            _validate_matching_source_fingerprints(failure, episode, target, f"failure_modes[{index}]")
        _validate_source_fingerprint_fields(failure, target, f"failure_modes[{index}]", warn_if_missing=False)

def _validate_curriculum(
    curriculum: dict[str, Any],
    target: ValidationTarget,
    episodes: list[dict[str, Any]],
    failure_modes: list[dict[str, Any]],
) -> None:
    _require_equal(curriculum, "schema_version", RL_CURRICULUM_SCHEMA_VERSION, target, prefix="curriculum.")
    if curriculum.get("episode_count") != len(episodes):
        target.errors.append(f"curriculum.episode_count expected {len(episodes)}, got {curriculum.get('episode_count')!r}.")
    if curriculum.get("failure_mode_count") != len(failure_modes):
        target.errors.append(
            f"curriculum.failure_mode_count expected {len(failure_modes)}, got {curriculum.get('failure_mode_count')!r}."
        )
    families = curriculum.get("task_families")
    if not isinstance(families, list):
        target.errors.append("curriculum.task_families must be a list.")
        return
    for family_index, family in enumerate(families):
        if not isinstance(family, dict):
            target.errors.append(f"curriculum.task_families[{family_index}] must be an object.")
            continue
        if not isinstance(family.get("task_family"), str) or not family.get("task_family"):
            target.errors.append(f"curriculum.task_families[{family_index}].task_family must be a non-empty string.")
        for field_name in ("episode_count", "passed", "failed"):
            if not isinstance(family.get(field_name), int) or isinstance(family.get(field_name), bool) or family.get(field_name) < 0:
                target.errors.append(f"curriculum.task_families[{family_index}].{field_name} must be a non-negative integer.")
        if not isinstance(family.get("average_score"), (int, float)):
            target.errors.append(f"curriculum.task_families[{family_index}].average_score must be numeric.")
        modes = family.get("failure_modes")
        if not isinstance(modes, list):
            target.errors.append(f"curriculum.task_families[{family_index}].failure_modes must be a list.")
            continue
        for mode_index, mode in enumerate(modes):
            if not isinstance(mode, dict):
                target.errors.append(f"curriculum.task_families[{family_index}].failure_modes[{mode_index}] must be an object.")
                continue
            if not isinstance(mode.get("rule_id"), str) or not mode.get("rule_id"):
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].rule_id must be a non-empty string."
                )
            if not isinstance(mode.get("count"), int) or isinstance(mode.get("count"), bool) or mode.get("count") < 0:
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].count must be a non-negative integer."
                )
            if not isinstance(mode.get("critical_count"), int) or isinstance(mode.get("critical_count"), bool) or mode.get("critical_count") < 0:
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].critical_count must be a non-negative integer."
                )
            if not isinstance(mode.get("max_penalty"), int) or isinstance(mode.get("max_penalty"), bool) or mode.get("max_penalty") < 0:
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].max_penalty must be a non-negative integer."
                )
            if not isinstance(mode.get("average_penalty"), (int, float)) or isinstance(mode.get("average_penalty"), bool):
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].average_penalty must be numeric."
                )
            if not isinstance(mode.get("priority_score"), int) or isinstance(mode.get("priority_score"), bool) or mode.get("priority_score") < 0:
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].priority_score must be a non-negative integer."
                )
            elif _is_non_negative_int(mode.get("count")) and _is_non_negative_int(mode.get("critical_count")) and _is_non_negative_int(mode.get("max_penalty")):
                expected_priority = int(mode["count"]) * 10 + int(mode["critical_count"]) * 100 + int(mode["max_penalty"])
                if mode.get("priority_score") != expected_priority:
                    target.errors.append(
                        f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].priority_score expected "
                        f"{expected_priority}, got {mode.get('priority_score')!r}."
                    )
            if mode.get("priority_band") not in {"critical", "high", "medium", "low"}:
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].priority_band must be critical, high, medium, or low."
                )
            elif _is_non_negative_int(mode.get("priority_score")):
                expected_band = _expected_curriculum_priority_band(int(mode["priority_score"]))
                if mode.get("priority_band") != expected_band:
                    target.errors.append(
                        f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].priority_band expected "
                        f"{expected_band!r}, got {mode.get('priority_band')!r}."
                    )
            if not isinstance(mode.get("episode_ids"), list):
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].episode_ids must be a list."
                )
            if not isinstance(mode.get("scenario_ids"), list):
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].scenario_ids must be a list."
                )
            if not isinstance(mode.get("failure_ids"), list):
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].failure_ids must be a list."
                )
            if not isinstance(mode.get("example_evidence"), list):
                target.errors.append(
                    f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].example_evidence must be a list."
                )
            _validate_evidence_refs(
                mode.get("example_evidence_refs"),
                target,
                f"curriculum.task_families[{family_index}].failure_modes[{mode_index}].example_evidence_refs",
            )

def _expected_curriculum_priority_band(priority_score: int) -> str:
    if priority_score >= 150:
        return "critical"
    if priority_score >= 75:
        return "high"
    if priority_score >= 25:
        return "medium"
    return "low"
