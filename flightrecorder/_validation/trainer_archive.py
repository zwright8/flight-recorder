"""Extracted validation implementation."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path, PureWindowsPath
from typing import Any
from ..preflight import TRAINER_DIRECTORY_TREE_HASH_ALGORITHM, TRAINER_LAUNCH_CHECK_SCHEMA_VERSION, TRAINER_PREFLIGHT_SEMANTIC_GATE_ROLES, TRAINER_PREFLIGHT_SCHEMA_VERSION, TRAINER_PREFLIGHT_SOURCE_ARTIFACT_FIELDS, TrainerPreflightError, build_trainer_launch_check, build_trainer_preflight_source_artifact, trainer_preflight_gate_semantics_ready
from ..reviewed_gate import REVIEWED_EXPORT_SOURCE_ARTIFACT_FIELDS, REVIEWED_GATE_SCHEMA_VERSION, ReviewedGateError, build_reviewed_export_source_artifact, evaluate_reviewed_gate
from ..path_safety import path_has_symlink_component as _path_has_symlink_component, resolve_artifact_reference_path
from ..trainer_archive import TRAINER_ARCHIVE_SCHEMA_VERSION
from ..trainer_archive_check import TRAINER_ARCHIVE_CHECK_SCHEMA_VERSION
from ..trainer_consumer_plan import TRAINER_CONSUMER_PLAN_SCHEMA_VERSION
from ..hashing import sha256_file as _sha256
from .constants import TRAINER_WRAPPER_DRY_RUN_SCHEMA_VERSION
from .governance import _resolve_gate_source_path, _validate_action_ledger_count_rows
from .primitives import ValidationTarget, _archive_artifact_path, _archive_artifact_roles_by_name, _directory_tree_fingerprint, _is_dataset_version, _is_lowercase_sha256, _is_non_negative_int, _is_redacted_placeholder, _is_sha256, _is_string_list, _is_windows_absolute, _looks_absolute, _non_negative_int_value, _path_resolves_inside, _read_object, _reject_archive_artifact_symlink_path, _reject_symlinked_validation_path, _require_equal, _reviewed_export_source_record, _sha256, _validate_allowed_keys, _validate_archive_relationship, _validate_gate_like_checks, _validate_metadata, _warn_absolute_public_path
from .runs import _warn_command_token_public_path, _warn_shell_tokens_public_paths
from .training_flow_result import _repo_root_for_artifact

def validate_trainer_preflight(
    path: str | Path,
    *,
    payload: dict[str, Any] | None = None,
) -> ValidationTarget:
    """Validate a trainer-preflight launch guard artifact."""
    preflight_path = Path(path)
    target = ValidationTarget("trainer_preflight", str(preflight_path))
    preflight = payload if isinstance(payload, dict) else _read_object(preflight_path, target, "trainer_preflight.json")
    if preflight is not None:
        _validate_trainer_preflight(preflight, target, preflight_path)
    return target

def validate_trainer_launch_check(
    path: str | Path,
    *,
    payload: dict[str, Any] | None = None,
) -> ValidationTarget:
    """Validate a trainer launch-check consumer artifact."""
    launch_check_path = Path(path)
    target = ValidationTarget("trainer_launch_check", str(launch_check_path))
    if _reject_symlinked_validation_path(
        launch_check_path,
        target,
        "trainer_launch_check.path",
        "file",
    ):
        return target
    launch_check = payload if isinstance(payload, dict) else _read_object(launch_check_path, target, "trainer_launch_check.json")
    if launch_check is not None:
        _validate_trainer_launch_check(launch_check, target, launch_check_path)
    return target

def validate_trainer_archive(path: str | Path) -> ValidationTarget:
    """Validate a portable trainer handoff archive directory or manifest."""
    archive_path = Path(path)
    manifest_path = archive_path / "trainer_archive.json" if archive_path.is_dir() else archive_path
    archive_root = manifest_path.parent
    target = ValidationTarget("trainer_archive", str(archive_path))
    archive = _read_object(manifest_path, target, "trainer_archive.json")
    if archive is not None:
        _validate_trainer_archive(archive, target, archive_root)
    return target

def validate_trainer_archive_check(path: str | Path) -> ValidationTarget:
    """Validate a trainer archive consumer-readiness artifact."""
    check_path = Path(path)
    target = ValidationTarget("trainer_archive_check", str(check_path))
    check = _read_object(check_path, target, "trainer_archive_check.json")
    if check is not None:
        _validate_trainer_archive_check(check, target)
    return target

def validate_trainer_consumer_plan(path: str | Path) -> ValidationTarget:
    """Validate a trainer consumer plan artifact."""
    plan_path = Path(path)
    target = ValidationTarget("trainer_consumer_plan", str(plan_path))
    plan = _read_object(plan_path, target, "trainer_consumer_plan.json")
    if plan is not None:
        _validate_trainer_consumer_plan(plan, target, plan_path)
    return target

def validate_trainer_wrapper_dry_run(path: str | Path) -> ValidationTarget:
    """Validate a reference trainer-wrapper dry-run receipt."""
    receipt_path = Path(path)
    target = ValidationTarget("trainer_wrapper_dry_run", str(receipt_path))
    receipt = _read_object(receipt_path, target, "trainer_wrapper_dry_run.json")
    if receipt is not None:
        _validate_trainer_wrapper_dry_run(receipt, target)
    return target

_TRAINER_ARCHIVE_KEYS = {
    "schema_version",
    "archive_path",
    "manifest_path",
    "passed",
    "readiness",
    "recommendation",
    "self_contained",
    "require_self_contained",
    "ready_for_training",
    "launch_check_included",
    "approved_command",
    "trainer_inputs",
    "path_rewrites",
    "portable_command",
    "consumer_contract",
    "artifacts",
    "missing",
    "relationships",
    "metrics",
    "notes",
}

_TRAINER_ARCHIVE_APPROVED_COMMAND_KEYS = {"approved", "provided", "raw", "argv", "parseable", "shell"}

_TRAINER_ARCHIVE_ARTIFACT_KEYS = {
    "index",
    "name",
    "role",
    "kind",
    "path",
    "original_path",
    "exists",
    "size_bytes",
    "sha256",
    "schema_version",
    "source_passed",
    "tree_hash_algorithm",
    "file_count",
}

_TRAINER_ARCHIVE_MISSING_KEYS = {"role", "index", "name", "reason"}

_TRAINER_ARCHIVE_METRICS_KEYS = {
    "artifact_count",
    "file_artifact_count",
    "directory_artifact_count",
    "trainer_input_count",
    "path_rewrite_count",
    "external_command_path_count",
    "missing_count",
    "total_size_bytes",
    "role_counts",
    "missing_role_counts",
    "unique_sha256_count",
}

_TRAINER_ARCHIVE_TRAINER_INPUT_KEYS = {
    "artifact_index",
    "artifact_name",
    "kind",
    "original_path",
    "archive_path",
    "size_bytes",
    "sha256",
    "file_count",
    "tree_hash_algorithm",
}

_TRAINER_ARCHIVE_PATH_REWRITE_KEYS = {"artifact_name", "kind", "original_path", "archive_path"}

_TRAINER_ARCHIVE_PORTABLE_COMMAND_KEYS = {
    "approved",
    "available",
    "rewritten",
    "path_rewrite_count",
    "argv",
    "shell",
    "notes",
}

_TRAINER_ARCHIVE_CONSUMER_CONTRACT_KEYS = {
    "execution_cwd",
    "command_kind",
    "portable_command_available",
    "portable_command_rewritten",
    "trainer_input_count",
    "path_rewrite_count",
    "external_code_required",
    "external_command_path_count",
    "external_command_paths",
    "notes",
}

_TRAINER_ARCHIVE_EXTERNAL_COMMAND_PATH_KEYS = {"argv_index", "token", "path", "reason"}

def _validate_trainer_archive(archive: dict[str, Any], target: ValidationTarget, archive_root: Path) -> None:
    _validate_allowed_keys(archive, _TRAINER_ARCHIVE_KEYS, target, "trainer_archive")
    _require_equal(archive, "schema_version", TRAINER_ARCHIVE_SCHEMA_VERSION, target)
    for field_name in ("archive_path", "manifest_path", "readiness", "recommendation"):
        if not isinstance(archive.get(field_name), str) or not archive.get(field_name):
            target.errors.append(f"trainer_archive.{field_name} must be a non-empty string.")
    for field_name in ("archive_path", "manifest_path"):
        _warn_absolute_public_path(target, f"trainer_archive.{field_name}", archive.get(field_name))
    for field_name in ("passed", "self_contained", "require_self_contained", "ready_for_training", "launch_check_included"):
        if not isinstance(archive.get(field_name), bool):
            target.errors.append(f"trainer_archive.{field_name} must be a boolean.")
    artifacts = archive.get("artifacts")
    if not isinstance(artifacts, list):
        target.errors.append("trainer_archive.artifacts must be a list.")
        artifacts = []
    missing = archive.get("missing")
    if not isinstance(missing, list):
        target.errors.append("trainer_archive.missing must be a list.")
        missing = []
    relationships = archive.get("relationships")
    if not isinstance(relationships, list):
        target.errors.append("trainer_archive.relationships must be a list.")
        relationships = []
    metrics = archive.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("trainer_archive.metrics must be an object.")
        metrics = {}
    if not _is_string_list(archive.get("notes")):
        target.errors.append("trainer_archive.notes must be a list of strings.")

    for index, artifact in enumerate(artifacts):
        _validate_trainer_archive_artifact(artifact, target, f"trainer_archive.artifacts[{index}]", index, archive_root)
    for index, item in enumerate(missing):
        _validate_trainer_archive_missing(item, target, f"trainer_archive.missing[{index}]")
    artifact_roles_by_name = _archive_artifact_roles_by_name(artifacts, target, "trainer_archive", require_unique=False)
    allowed_relationship_edges = {"validates": {("trainer_launch_check", "trainer_preflight")}}
    for index, relationship in enumerate(relationships):
        _validate_archive_relationship(
            relationship,
            target,
            f"trainer_archive.relationships[{index}]",
            artifact_roles_by_name,
            allowed_relationship_edges,
        )

    valid_artifacts = [artifact for artifact in artifacts if isinstance(artifact, dict)]
    roles = {artifact.get("role") for artifact in valid_artifacts}
    for required_role in ("trainer_preflight", "trainer_launch_check"):
        if required_role not in roles:
            target.errors.append(f"trainer_archive.artifacts must include role {required_role}.")
    preflight = next((artifact for artifact in valid_artifacts if artifact.get("role") == "trainer_preflight"), {})
    launch_check = next((artifact for artifact in valid_artifacts if artifact.get("role") == "trainer_launch_check"), {})
    launch_included = bool(launch_check)
    ready_for_training = preflight.get("source_passed") is True and launch_check.get("source_passed") is True
    self_contained = len(missing) == 0
    expected_passed = ready_for_training and (self_contained or archive.get("require_self_contained") is not True)
    if isinstance(archive.get("launch_check_included"), bool) and archive["launch_check_included"] != launch_included:
        target.errors.append(f"trainer_archive.launch_check_included expected {launch_included}, got {archive.get('launch_check_included')!r}.")
    if isinstance(archive.get("ready_for_training"), bool) and archive["ready_for_training"] != ready_for_training:
        target.errors.append(f"trainer_archive.ready_for_training expected {ready_for_training}, got {archive.get('ready_for_training')!r}.")
    if isinstance(archive.get("self_contained"), bool) and archive["self_contained"] != self_contained:
        target.errors.append(f"trainer_archive.self_contained expected {self_contained}, got {archive.get('self_contained')!r}.")
    if isinstance(archive.get("passed"), bool) and archive["passed"] != expected_passed:
        target.errors.append(f"trainer_archive.passed expected {expected_passed}, got {archive.get('passed')!r}.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "handoff_ready" if expected_passed else "block_handoff"
    if archive.get("readiness") != expected_readiness:
        target.errors.append(f"trainer_archive.readiness expected {expected_readiness!r}, got {archive.get('readiness')!r}.")
    if archive.get("recommendation") != expected_recommendation:
        target.errors.append(f"trainer_archive.recommendation expected {expected_recommendation!r}, got {archive.get('recommendation')!r}.")
    _validate_trainer_archive_inputs(archive.get("trainer_inputs"), valid_artifacts, target)
    _validate_trainer_archive_rewrites(archive.get("path_rewrites"), archive.get("trainer_inputs"), target)
    _validate_trainer_archive_commands(
        archive.get("approved_command"),
        archive.get("portable_command"),
        archive.get("path_rewrites"),
        target,
    )
    _validate_trainer_archive_consumer_contract(
        archive.get("consumer_contract"),
        archive.get("portable_command"),
        archive.get("trainer_inputs"),
        archive.get("path_rewrites"),
        target,
    )
    _validate_trainer_archive_metrics(metrics, artifacts, missing, archive.get("consumer_contract"), target)
    target.details.update(
        {
            "artifact_count": len(artifacts),
            "missing_count": len(missing),
            "self_contained": archive.get("self_contained"),
            "ready_for_training": archive.get("ready_for_training"),
            "passed": archive.get("passed"),
        }
    )

def _validate_trainer_archive_artifact(
    artifact: Any,
    target: ValidationTarget,
    label: str,
    expected_index: int,
    archive_root: Path,
) -> None:
    if not isinstance(artifact, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(artifact, _TRAINER_ARCHIVE_ARTIFACT_KEYS, target, label)
    if artifact.get("index") != expected_index:
        target.errors.append(f"{label}.index expected {expected_index}, got {artifact.get('index')!r}.")
    for field_name in ("name", "role", "kind", "path", "original_path"):
        if not isinstance(artifact.get(field_name), str) or not artifact.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    _warn_absolute_public_path(target, f"{label}.original_path", artifact.get("original_path"))
    valid_roles = {"trainer_preflight", "trainer_launch_check", "gate", "validation_summary", "trainer_artifact", "schema_contract"}
    if artifact.get("role") not in valid_roles:
        target.errors.append(f"{label}.role is invalid.")
    if artifact.get("kind") not in {"file", "directory"}:
        target.errors.append(f"{label}.kind must be file or directory.")
    if artifact.get("exists") is not True:
        target.errors.append(f"{label}.exists must be true.")
    if not _is_non_negative_int(artifact.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    if not _is_sha256(artifact.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string.")
        return
    if artifact.get("source_passed") is not None and not isinstance(artifact.get("source_passed"), bool):
        target.errors.append(f"{label}.source_passed must be a boolean or null.")
    if artifact.get("role") == "trainer_preflight":
        if artifact.get("schema_version") != TRAINER_PREFLIGHT_SCHEMA_VERSION:
            target.errors.append(f"{label}.schema_version must be {TRAINER_PREFLIGHT_SCHEMA_VERSION}.")
    if artifact.get("role") == "trainer_launch_check":
        if artifact.get("schema_version") != TRAINER_LAUNCH_CHECK_SCHEMA_VERSION:
            target.errors.append(f"{label}.schema_version must be {TRAINER_LAUNCH_CHECK_SCHEMA_VERSION}.")

    artifact_path = _archive_artifact_path(artifact.get("path"), archive_root)
    if artifact_path is None:
        target.errors.append(f"{label}.path must be a relative archive path.")
        return
    if not _path_resolves_inside(artifact_path, archive_root):
        target.errors.append(f"{label}.path must resolve inside the archive.")
        return
    if _reject_archive_artifact_symlink_path(artifact_path, target, label):
        return
    if artifact.get("kind") == "file":
        if not artifact_path.exists() or not artifact_path.is_file():
            target.errors.append(f"{label}.path does not exist as a file inside the archive.")
            return
        if artifact_path.stat().st_size != artifact.get("size_bytes"):
            target.errors.append(f"{label}.size_bytes does not match the archived file.")
        if _sha256(artifact_path) != artifact.get("sha256"):
            target.errors.append(f"{label}.sha256 does not match the archived file.")
        return
    if not artifact_path.exists() or not artifact_path.is_dir():
        target.errors.append(f"{label}.path does not exist as a directory inside the archive.")
        return
    for child in artifact_path.rglob("*"):
        if child.is_symlink():
            target.errors.append(f"{label}.path contains symlink {child}.")
            return
    if artifact.get("tree_hash_algorithm") != "sha256(sorted-relative-path-size-file-sha256)":
        target.errors.append(f"{label}.tree_hash_algorithm is invalid.")
    if not _is_non_negative_int(artifact.get("file_count")):
        target.errors.append(f"{label}.file_count must be a non-negative integer for directories.")
        return
    tree = _trainer_archive_tree_fingerprint(artifact_path)
    if artifact.get("file_count") != tree["file_count"]:
        target.errors.append(f"{label}.file_count does not match the archived directory.")
    if artifact.get("size_bytes") != tree["size_bytes"]:
        target.errors.append(f"{label}.size_bytes does not match the archived directory.")
    if artifact.get("sha256") != tree["sha256"]:
        target.errors.append(f"{label}.sha256 does not match the archived directory tree.")

def _validate_trainer_archive_missing(item: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(item, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(item, _TRAINER_ARCHIVE_MISSING_KEYS, target, label)
    valid_roles = {"trainer_launch_check", "gate", "validation_summary", "trainer_artifact", "schema_contract"}
    if item.get("role") not in valid_roles:
        target.errors.append(f"{label}.role is invalid.")
    if not _is_non_negative_int(item.get("index")):
        target.errors.append(f"{label}.index must be a non-negative integer.")
    if "name" in item and not isinstance(item.get("name"), str):
        target.errors.append(f"{label}.name must be a string when present.")
    if not isinstance(item.get("reason"), str) or not item.get("reason"):
        target.errors.append(f"{label}.reason must be a non-empty string.")

def _validate_trainer_archive_metrics(
    metrics: dict[str, Any],
    artifacts: list[Any],
    missing: list[Any],
    consumer_contract: Any,
    target: ValidationTarget,
) -> None:
    _validate_allowed_keys(metrics, _TRAINER_ARCHIVE_METRICS_KEYS, target, "trainer_archive.metrics")
    valid_artifacts = [artifact for artifact in artifacts if isinstance(artifact, dict)]
    valid_missing = [item for item in missing if isinstance(item, dict)]
    expected = {
        "artifact_count": len(artifacts),
        "file_artifact_count": sum(1 for artifact in valid_artifacts if artifact.get("kind") == "file"),
        "directory_artifact_count": sum(1 for artifact in valid_artifacts if artifact.get("kind") == "directory"),
        "trainer_input_count": sum(1 for artifact in valid_artifacts if artifact.get("role") == "trainer_artifact"),
        "path_rewrite_count": consumer_contract.get("path_rewrite_count", 0) if isinstance(consumer_contract, dict) else 0,
        "external_command_path_count": consumer_contract.get("external_command_path_count", 0) if isinstance(consumer_contract, dict) else 0,
        "missing_count": len(missing),
        "total_size_bytes": sum(artifact.get("size_bytes", 0) for artifact in valid_artifacts if _is_non_negative_int(artifact.get("size_bytes"))),
        "unique_sha256_count": len({artifact.get("sha256") for artifact in valid_artifacts if isinstance(artifact.get("sha256"), str)}),
    }
    for field_name, expected_value in expected.items():
        if metrics.get(field_name) != expected_value:
            target.errors.append(f"trainer_archive.metrics.{field_name} expected {expected_value}, got {metrics.get(field_name)!r}.")
    role_counts = _trainer_archive_count_map(artifact.get("role") for artifact in valid_artifacts)
    missing_role_counts = _trainer_archive_count_map(item.get("role") for item in valid_missing)
    _validate_action_ledger_count_rows(metrics.get("role_counts"), role_counts, target, "trainer_archive.metrics.role_counts")
    _validate_action_ledger_count_rows(
        metrics.get("missing_role_counts"),
        missing_role_counts,
        target,
        "trainer_archive.metrics.missing_role_counts",
    )

def _validate_trainer_archive_inputs(value: Any, artifacts: list[dict[str, Any]], target: ValidationTarget) -> None:
    if not isinstance(value, list):
        target.errors.append("trainer_archive.trainer_inputs must be a list.")
        return
    expected = [_trainer_input_from_artifact(artifact) for artifact in artifacts if artifact.get("role") == "trainer_artifact"]
    if len(value) != len(expected):
        target.errors.append(f"trainer_archive.trainer_inputs expected {len(expected)} item(s), got {len(value)}.")
    for index, item in enumerate(value):
        label = f"trainer_archive.trainer_inputs[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(item, _TRAINER_ARCHIVE_TRAINER_INPUT_KEYS, target, label)
        for field_name in ("artifact_name", "kind", "original_path", "archive_path", "sha256"):
            if not isinstance(item.get(field_name), str) or not item.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        _warn_absolute_public_path(target, f"{label}.original_path", item.get("original_path"))
        for field_name in ("artifact_index", "size_bytes"):
            if not _is_non_negative_int(item.get(field_name)):
                target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
        if item.get("kind") == "directory":
            if not _is_non_negative_int(item.get("file_count")):
                target.errors.append(f"{label}.file_count must be a non-negative integer for directories.")
            if item.get("tree_hash_algorithm") != "sha256(sorted-relative-path-size-file-sha256)":
                target.errors.append(f"{label}.tree_hash_algorithm is invalid for directories.")
        if index < len(expected):
            for field_name, expected_value in expected[index].items():
                if item.get(field_name) != expected_value:
                    target.errors.append(f"{label}.{field_name} expected {expected_value!r}, got {item.get(field_name)!r}.")

def _validate_trainer_archive_rewrites(value: Any, inputs: Any, target: ValidationTarget) -> None:
    if not isinstance(value, list):
        target.errors.append("trainer_archive.path_rewrites must be a list.")
        return
    expected = _expected_trainer_archive_rewrites(inputs if isinstance(inputs, list) else [])
    expected_by_binding = {
        (item["artifact_name"], item["kind"], item["archive_path"]): item
        for item in expected
    }
    seen: set[tuple[str, str]] = set()
    canonical_rewrites: set[tuple[str, str, str, str]] = set()
    for index, item in enumerate(value):
        label = f"trainer_archive.path_rewrites[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(item, _TRAINER_ARCHIVE_PATH_REWRITE_KEYS, target, label)
        for field_name in ("artifact_name", "kind", "original_path", "archive_path"):
            if not isinstance(item.get(field_name), str) or not item.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        _warn_absolute_public_path(target, f"{label}.original_path", item.get("original_path"))
        binding = (item.get("artifact_name"), item.get("kind"), item.get("archive_path"))
        source = expected_by_binding.get(binding)
        if source is None:
            target.errors.append(f"{label} is not bound to a trainer input.")
            continue
        original = item.get("original_path")
        canonical = source["original_path"]
        if original != canonical and not _trainer_archive_alias_matches(original, canonical):
            target.errors.append(f"{label}.original_path is not a suffix-bound alias of {canonical!r}.")
        identity = (str(original), str(item.get("archive_path")))
        if identity in seen:
            target.errors.append(f"{label} duplicates an existing path rewrite.")
        seen.add(identity)
        canonical_rewrites.add(
            (str(item.get("artifact_name")), str(item.get("kind")), str(original), str(item.get("archive_path")))
        )
    for expected_item in expected:
        identity = (
            expected_item["artifact_name"],
            expected_item["kind"],
            expected_item["original_path"],
            expected_item["archive_path"],
        )
        if identity not in canonical_rewrites:
            target.errors.append(
                "trainer_archive.path_rewrites is missing the canonical rewrite for "
                f"{expected_item['artifact_name']!r}."
            )

def _trainer_archive_alias_matches(alias: Any, canonical: str) -> bool:
    if not isinstance(alias, str) or not alias:
        return False
    normalized_alias = alias.replace("\\", "/").rstrip("/")
    normalized_canonical = canonical.replace("\\", "/").strip("/")
    return bool(normalized_canonical) and normalized_alias.endswith("/" + normalized_canonical)

def _validate_trainer_archive_commands(
    approved_command: Any,
    portable_command: Any,
    path_rewrites: Any,
    target: ValidationTarget,
) -> None:
    if not isinstance(approved_command, dict):
        target.errors.append("trainer_archive.approved_command must be an object.")
        approved_command = {}
    else:
        _validate_allowed_keys(approved_command, _TRAINER_ARCHIVE_APPROVED_COMMAND_KEYS, target, "trainer_archive.approved_command")
    for field_name in ("approved", "provided", "parseable"):
        if not isinstance(approved_command.get(field_name), bool):
            target.errors.append(f"trainer_archive.approved_command.{field_name} must be a boolean.")
    if not isinstance(approved_command.get("raw"), str):
        target.errors.append("trainer_archive.approved_command.raw must be a string.")
    if not isinstance(approved_command.get("shell"), str):
        target.errors.append("trainer_archive.approved_command.shell must be a string.")
    argv = approved_command.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        target.errors.append("trainer_archive.approved_command.argv must be a list of strings.")
        argv = []
    for index, item in enumerate(item for item in argv if isinstance(item, str)):
        _warn_command_token_public_path(target, f"trainer_archive.approved_command.argv[{index}]", item)
    for field_name in ("raw", "shell"):
        value = approved_command.get(field_name)
        if isinstance(value, str) and value:
            _warn_shell_tokens_public_paths(value, target, f"trainer_archive.approved_command.{field_name}")

    if not isinstance(portable_command, dict):
        target.errors.append("trainer_archive.portable_command must be an object.")
        return
    _validate_allowed_keys(portable_command, _TRAINER_ARCHIVE_PORTABLE_COMMAND_KEYS, target, "trainer_archive.portable_command")
    for field_name in ("approved", "available", "rewritten"):
        if not isinstance(portable_command.get(field_name), bool):
            target.errors.append(f"trainer_archive.portable_command.{field_name} must be a boolean.")
    if not _is_non_negative_int(portable_command.get("path_rewrite_count")):
        target.errors.append("trainer_archive.portable_command.path_rewrite_count must be a non-negative integer.")
    if not isinstance(portable_command.get("shell"), str):
        target.errors.append("trainer_archive.portable_command.shell must be a string.")
    if not _is_string_list(portable_command.get("notes")):
        target.errors.append("trainer_archive.portable_command.notes must be a list of strings.")
    portable_argv = portable_command.get("argv")
    if not isinstance(portable_argv, list) or not all(isinstance(item, str) for item in portable_argv):
        target.errors.append("trainer_archive.portable_command.argv must be a list of strings.")
        portable_argv = []

    rewrites = path_rewrites if isinstance(path_rewrites, list) else []
    expected_argv, expected_rewrite_count = _rewrite_trainer_archive_command_argv([item for item in argv if isinstance(item, str)], rewrites)
    if portable_argv != expected_argv:
        target.errors.append("trainer_archive.portable_command.argv must match approved_command.argv rewritten through path_rewrites.")
    if portable_command.get("shell") != (shlex.join(expected_argv) if expected_argv else ""):
        target.errors.append("trainer_archive.portable_command.shell must match the rewritten argv.")
    if portable_command.get("path_rewrite_count") != expected_rewrite_count:
        target.errors.append(
            f"trainer_archive.portable_command.path_rewrite_count expected {expected_rewrite_count}, got {portable_command.get('path_rewrite_count')!r}."
        )
    if portable_command.get("rewritten") != (expected_rewrite_count > 0):
        target.errors.append("trainer_archive.portable_command.rewritten must match path_rewrite_count.")
    if portable_command.get("available") != bool(expected_argv):
        target.errors.append("trainer_archive.portable_command.available must match whether argv is present.")
    if portable_command.get("approved") != (approved_command.get("approved") is True):
        target.errors.append("trainer_archive.portable_command.approved must match approved_command.approved.")

def _validate_trainer_archive_consumer_contract(
    contract: Any,
    portable_command: Any,
    trainer_inputs: Any,
    path_rewrites: Any,
    target: ValidationTarget,
) -> None:
    if not isinstance(contract, dict):
        target.errors.append("trainer_archive.consumer_contract must be an object.")
        return
    _validate_allowed_keys(contract, _TRAINER_ARCHIVE_CONSUMER_CONTRACT_KEYS, target, "trainer_archive.consumer_contract")
    if contract.get("execution_cwd") != "archive_root":
        target.errors.append("trainer_archive.consumer_contract.execution_cwd must be archive_root.")
    if contract.get("command_kind") != "advisory_portable_command":
        target.errors.append("trainer_archive.consumer_contract.command_kind must be advisory_portable_command.")
    for field_name in ("portable_command_available", "portable_command_rewritten", "external_code_required"):
        if not isinstance(contract.get(field_name), bool):
            target.errors.append(f"trainer_archive.consumer_contract.{field_name} must be a boolean.")
    for field_name in ("trainer_input_count", "path_rewrite_count", "external_command_path_count"):
        if not _is_non_negative_int(contract.get(field_name)):
            target.errors.append(f"trainer_archive.consumer_contract.{field_name} must be a non-negative integer.")
    if not _is_string_list(contract.get("notes")):
        target.errors.append("trainer_archive.consumer_contract.notes must be a list of strings.")
    inputs = trainer_inputs if isinstance(trainer_inputs, list) else []
    rewrites = path_rewrites if isinstance(path_rewrites, list) else []
    portable = portable_command if isinstance(portable_command, dict) else {}
    portable_argv = portable.get("argv") if isinstance(portable.get("argv"), list) else []
    clean_portable_argv = [item for item in portable_argv if isinstance(item, str)]
    expected_external = _trainer_archive_external_command_paths(clean_portable_argv, inputs)
    expected = {
        "portable_command_available": portable.get("available") is True,
        "portable_command_rewritten": portable.get("rewritten") is True,
        "trainer_input_count": len(inputs),
        "path_rewrite_count": len(rewrites),
        "external_code_required": bool(expected_external),
        "external_command_path_count": len(expected_external),
    }
    for field_name, expected_value in expected.items():
        if contract.get(field_name) != expected_value:
            target.errors.append(f"trainer_archive.consumer_contract.{field_name} expected {expected_value!r}, got {contract.get(field_name)!r}.")

    external_paths = contract.get("external_command_paths")
    if not isinstance(external_paths, list):
        target.errors.append("trainer_archive.consumer_contract.external_command_paths must be a list.")
        return
    if len(external_paths) != len(expected_external):
        target.errors.append(
            f"trainer_archive.consumer_contract.external_command_paths expected {len(expected_external)} item(s), got {len(external_paths)}."
        )
    for index, item in enumerate(external_paths):
        label = f"trainer_archive.consumer_contract.external_command_paths[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(item, _TRAINER_ARCHIVE_EXTERNAL_COMMAND_PATH_KEYS, target, label)
        if not _is_non_negative_int(item.get("argv_index")):
            target.errors.append(f"{label}.argv_index must be a non-negative integer.")
        for field_name in ("token", "path", "reason"):
            if not isinstance(item.get(field_name), str) or not item.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if index < len(expected_external):
            for field_name, expected_value in expected_external[index].items():
                if item.get(field_name) != expected_value:
                    target.errors.append(f"{label}.{field_name} expected {expected_value!r}, got {item.get(field_name)!r}.")

_TRAINER_ARCHIVE_CHECK_KEYS = {
    "schema_version",
    "archive_path",
    "manifest_path",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "validation",
    "archive",
    "external_code_root",
    "portable_command",
    "consumer_contract",
    "external_code_checks",
    "trainer_input_checks",
    "metrics",
    "notes",
}

_TRAINER_ARCHIVE_CHECK_CHECK_KEYS = {"id", "passed", "actual", "expected", "scope", "summary"}

_TRAINER_ARCHIVE_CHECK_VALIDATION_KEYS = {
    "available",
    "passed",
    "strict",
    "target_count",
    "error_count",
    "warning_count",
    "errors",
    "warnings",
}

_TRAINER_ARCHIVE_CHECK_ARCHIVE_KEYS = {
    "path",
    "manifest_path",
    "schema_version",
    "passed",
    "self_contained",
    "ready_for_training",
    "trainer_input_count",
    "external_command_path_count",
}

_TRAINER_ARCHIVE_CHECK_EXTERNAL_ROOT_KEYS = {
    "path",
    "exists",
    "kind",
    "regular_directory",
    "symlink",
}

_TRAINER_ARCHIVE_CHECK_PORTABLE_COMMAND_KEYS = {
    "approved",
    "available",
    "rewritten",
    "path_rewrite_count",
    "argv",
    "shell",
}

_TRAINER_ARCHIVE_CHECK_CONSUMER_CONTRACT_KEYS = {
    "execution_cwd",
    "command_kind",
    "portable_command_available",
    "trainer_input_count",
    "path_rewrite_count",
    "external_code_required",
    "external_command_path_count",
    "external_command_paths",
}

_TRAINER_ARCHIVE_CHECK_EXTERNAL_PATH_KEYS = {"argv_index", "token", "path", "reason"}

_TRAINER_ARCHIVE_CHECK_EXTERNAL_CODE_KEYS = {
    "index",
    "argv_index",
    "token",
    "path",
    "resolved_path",
    "exists",
    "kind",
    "regular_file",
    "symlink",
    "passed",
    "reason",
    "size_bytes",
    "sha256",
}

_TRAINER_ARCHIVE_CHECK_TRAINER_INPUT_KEYS = {
    "index",
    "artifact_index",
    "artifact_name",
    "archive_path",
    "resolved_path",
    "kind",
    "exists",
    "regular_file",
    "regular_directory",
    "symlink",
    "expected_sha256",
    "expected_size_bytes",
    "expected_file_count",
    "file_count",
    "passed",
    "reason",
    "size_bytes",
    "sha256",
}

_TRAINER_ARCHIVE_CHECK_METRICS_KEYS = {
    "archive_validation_passed",
    "archive_validation_error_count",
    "archive_validation_warning_count",
    "external_command_path_count",
    "relative_external_command_path_count",
    "external_code_file_count",
    "missing_external_code_count",
    "trainer_input_count",
    "trainer_input_available_count",
    "missing_trainer_input_count",
    "check_count",
    "failed_check_count",
}

def _validate_trainer_archive_check(check: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(check, "schema_version", TRAINER_ARCHIVE_CHECK_SCHEMA_VERSION, target)
    _validate_allowed_keys(check, _TRAINER_ARCHIVE_CHECK_KEYS, target, "trainer_archive_check")
    for field_name in ("archive_path", "manifest_path", "readiness", "recommendation"):
        if not isinstance(check.get(field_name), str) or not check.get(field_name):
            target.errors.append(f"trainer_archive_check.{field_name} must be a non-empty string.")
    for field_name in ("archive_path", "manifest_path"):
        _warn_absolute_public_path(target, f"trainer_archive_check.{field_name}", check.get(field_name))
    if not isinstance(check.get("passed"), bool):
        target.errors.append("trainer_archive_check.passed must be a boolean.")
    checks = check.get("checks")
    if not isinstance(checks, list):
        target.errors.append("trainer_archive_check.checks must be a list.")
        checks = []
    validation = check.get("validation")
    if not isinstance(validation, dict):
        target.errors.append("trainer_archive_check.validation must be an object.")
        validation = {}
    metrics = check.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("trainer_archive_check.metrics must be an object.")
        metrics = {}
    if not _is_string_list(check.get("notes")):
        target.errors.append("trainer_archive_check.notes must be a list of strings.")

    for index, item in enumerate(checks):
        if isinstance(item, dict):
            _validate_allowed_keys(item, _TRAINER_ARCHIVE_CHECK_CHECK_KEYS, target, f"trainer_archive_check.checks[{index}]")
    failed_checks = _validate_gate_like_checks(checks, target, "trainer_archive_check.checks")
    if check.get("check_count") != len(checks):
        target.errors.append(f"trainer_archive_check.check_count expected {len(checks)}, got {check.get('check_count')!r}.")
    if check.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"trainer_archive_check.failed_check_count expected {failed_checks}, got {check.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(check.get("passed"), bool) and check.get("passed") != expected_passed:
        target.errors.append("trainer_archive_check.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "consumer_ready" if expected_passed else "block_consumer_launch"
    if check.get("readiness") != expected_readiness:
        target.errors.append(f"trainer_archive_check.readiness expected {expected_readiness!r}, got {check.get('readiness')!r}.")
    if check.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"trainer_archive_check.recommendation expected {expected_recommendation!r}, got {check.get('recommendation')!r}."
        )

    _validate_trainer_archive_check_validation(validation, target)
    _validate_trainer_archive_check_archive(check.get("archive"), target)
    _validate_trainer_archive_check_external_root(check.get("external_code_root"), target)
    _validate_trainer_archive_check_portable_command(check.get("portable_command"), target)
    _validate_trainer_archive_check_consumer_contract(check.get("consumer_contract"), target)
    external_code_checks = _validate_trainer_archive_check_external_code(check.get("external_code_checks"), target)
    trainer_input_checks = _validate_trainer_archive_check_inputs(check.get("trainer_input_checks"), target)
    _validate_trainer_archive_check_metrics(metrics, validation, external_code_checks, trainer_input_checks, len(checks), failed_checks, target)
    target.details.update(
        {
            "passed": check.get("passed"),
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "external_command_path_count": metrics.get("external_command_path_count"),
            "missing_external_code_count": metrics.get("missing_external_code_count"),
            "trainer_input_count": metrics.get("trainer_input_count"),
        }
    )

def _validate_trainer_archive_check_validation(value: dict[str, Any], target: ValidationTarget) -> None:
    _validate_allowed_keys(value, _TRAINER_ARCHIVE_CHECK_VALIDATION_KEYS, target, "trainer_archive_check.validation")
    _validate_trainer_compact_validation_record(
        value,
        target,
        "trainer_archive_check.validation",
        has_available=True,
        has_messages=True,
    )

def _validate_trainer_archive_check_archive(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("trainer_archive_check.archive must be an object.")
        return
    _validate_allowed_keys(value, _TRAINER_ARCHIVE_CHECK_ARCHIVE_KEYS, target, "trainer_archive_check.archive")
    for field_name in ("path", "manifest_path", "schema_version"):
        if not isinstance(value.get(field_name), str) or not value.get(field_name):
            target.errors.append(f"trainer_archive_check.archive.{field_name} must be a non-empty string.")
    for field_name in ("path", "manifest_path"):
        _warn_absolute_public_path(target, f"trainer_archive_check.archive.{field_name}", value.get(field_name))
    if value.get("schema_version") != TRAINER_ARCHIVE_SCHEMA_VERSION:
        target.errors.append(f"trainer_archive_check.archive.schema_version must be {TRAINER_ARCHIVE_SCHEMA_VERSION}.")
    for field_name in ("passed", "self_contained", "ready_for_training"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"trainer_archive_check.archive.{field_name} must be a boolean.")
    for field_name in ("trainer_input_count", "external_command_path_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"trainer_archive_check.archive.{field_name} must be a non-negative integer.")

def _validate_trainer_archive_check_external_root(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("trainer_archive_check.external_code_root must be an object.")
        return
    _validate_allowed_keys(value, _TRAINER_ARCHIVE_CHECK_EXTERNAL_ROOT_KEYS, target, "trainer_archive_check.external_code_root")
    for field_name in ("path", "kind"):
        if not isinstance(value.get(field_name), str) or not value.get(field_name):
            target.errors.append(f"trainer_archive_check.external_code_root.{field_name} must be a non-empty string.")
    _warn_absolute_public_path(target, "trainer_archive_check.external_code_root.path", value.get("path"))
    if value.get("kind") != "directory":
        target.errors.append("trainer_archive_check.external_code_root.kind must be directory.")
    for field_name in ("exists", "regular_directory", "symlink"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"trainer_archive_check.external_code_root.{field_name} must be a boolean.")

def _validate_trainer_archive_check_portable_command(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("trainer_archive_check.portable_command must be an object.")
        return
    _validate_allowed_keys(value, _TRAINER_ARCHIVE_CHECK_PORTABLE_COMMAND_KEYS, target, "trainer_archive_check.portable_command")
    for field_name in ("approved", "available", "rewritten"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"trainer_archive_check.portable_command.{field_name} must be a boolean.")
    if not _is_non_negative_int(value.get("path_rewrite_count")):
        target.errors.append("trainer_archive_check.portable_command.path_rewrite_count must be a non-negative integer.")
    if not isinstance(value.get("shell"), str):
        target.errors.append("trainer_archive_check.portable_command.shell must be a string.")
    argv = value.get("argv")
    if not _is_string_list(argv):
        target.errors.append("trainer_archive_check.portable_command.argv must be a list of strings.")
        argv = []
    for index, item in enumerate(item for item in argv if isinstance(item, str)):
        _validate_trainer_archive_check_command_token_public_path(
            target,
            f"trainer_archive_check.portable_command.argv[{index}]",
            item,
        )
    shell = value.get("shell")
    if isinstance(shell, str) and shell:
        _validate_trainer_archive_check_command_tokens_public_paths(shell, target, "trainer_archive_check.portable_command.shell")

def _validate_trainer_archive_check_command_tokens_public_paths(command: str, target: ValidationTarget, label: str) -> None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for index, token in enumerate(tokens):
        _validate_trainer_archive_check_command_token_public_path(target, f"{label}[{index}]", token)

def _validate_trainer_archive_check_command_token_public_path(target: ValidationTarget, label: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        return
    if _looks_absolute(value):
        target.errors.append(f"{label} must use a relative command token or redacted placeholder.")
        return
    _, separator, token_value = value.partition("=")
    if separator and _looks_absolute(token_value):
        target.errors.append(f"{label} must use a relative command token or redacted placeholder.")

def _validate_trainer_archive_check_consumer_contract(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("trainer_archive_check.consumer_contract must be an object.")
        return
    _validate_allowed_keys(value, _TRAINER_ARCHIVE_CHECK_CONSUMER_CONTRACT_KEYS, target, "trainer_archive_check.consumer_contract")
    for field_name in ("execution_cwd", "command_kind"):
        if not isinstance(value.get(field_name), str):
            target.errors.append(f"trainer_archive_check.consumer_contract.{field_name} must be a string.")
    for field_name in ("portable_command_available", "external_code_required"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"trainer_archive_check.consumer_contract.{field_name} must be a boolean.")
    for field_name in ("trainer_input_count", "path_rewrite_count", "external_command_path_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"trainer_archive_check.consumer_contract.{field_name} must be a non-negative integer.")
    external_paths = value.get("external_command_paths")
    if not isinstance(external_paths, list):
        target.errors.append("trainer_archive_check.consumer_contract.external_command_paths must be a list.")
        return
    for index, item in enumerate(external_paths):
        label = f"trainer_archive_check.consumer_contract.external_command_paths[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(item, _TRAINER_ARCHIVE_CHECK_EXTERNAL_PATH_KEYS, target, label)
        if not _is_non_negative_int(item.get("argv_index")):
            target.errors.append(f"{label}.argv_index must be a non-negative integer.")
        for field_name in ("token", "path", "reason"):
            if not isinstance(item.get(field_name), str):
                target.errors.append(f"{label}.{field_name} must be a string.")
        _warn_command_token_public_path(target, f"{label}.token", item.get("token"))
        _warn_absolute_public_path(target, f"{label}.path", item.get("path"))

def _validate_trainer_archive_check_external_code(value: Any, target: ValidationTarget) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        target.errors.append("trainer_archive_check.external_code_checks must be a list.")
        return []
    records = [item for item in value if isinstance(item, dict)]
    for index, item in enumerate(value):
        label = f"trainer_archive_check.external_code_checks[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(item, _TRAINER_ARCHIVE_CHECK_EXTERNAL_CODE_KEYS, target, label)
        if item.get("index") != index:
            target.errors.append(f"{label}.index expected {index}, got {item.get('index')!r}.")
        if not _is_non_negative_int(item.get("argv_index")):
            target.errors.append(f"{label}.argv_index must be a non-negative integer.")
        for field_name in ("token", "path", "resolved_path", "kind", "reason"):
            if not isinstance(item.get(field_name), str):
                target.errors.append(f"{label}.{field_name} must be a string.")
        _warn_command_token_public_path(target, f"{label}.token", item.get("token"))
        for field_name in ("path", "resolved_path"):
            _warn_absolute_public_path(target, f"{label}.{field_name}", item.get(field_name))
        if item.get("kind") != "file":
            target.errors.append(f"{label}.kind must be file.")
        for field_name in ("exists", "regular_file", "symlink", "passed"):
            if not isinstance(item.get(field_name), bool):
                target.errors.append(f"{label}.{field_name} must be a boolean.")
        expected_passed = item.get("exists") is True and item.get("regular_file") is True and item.get("symlink") is False
        if item.get("passed") is True and not expected_passed:
            target.errors.append(f"{label}.passed cannot be true unless the resolved path is a regular non-symlink file.")
        if item.get("passed") is True:
            if not _is_non_negative_int(item.get("size_bytes")):
                target.errors.append(f"{label}.size_bytes must be a non-negative integer when passed.")
            if not _is_sha256(item.get("sha256")):
                target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when passed.")
            _validate_visible_file_hash(item, target, label)
    return records

def _validate_trainer_archive_check_inputs(value: Any, target: ValidationTarget) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        target.errors.append("trainer_archive_check.trainer_input_checks must be a list.")
        return []
    records = [item for item in value if isinstance(item, dict)]
    for index, item in enumerate(value):
        label = f"trainer_archive_check.trainer_input_checks[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(item, _TRAINER_ARCHIVE_CHECK_TRAINER_INPUT_KEYS, target, label)
        if item.get("index") != index:
            target.errors.append(f"{label}.index expected {index}, got {item.get('index')!r}.")
        if not _is_non_negative_int(item.get("artifact_index")):
            target.errors.append(f"{label}.artifact_index must be a non-negative integer.")
        for field_name in ("artifact_name", "archive_path", "resolved_path", "kind", "expected_sha256", "reason"):
            if not isinstance(item.get(field_name), str):
                target.errors.append(f"{label}.{field_name} must be a string.")
        for field_name in ("archive_path", "resolved_path"):
            _warn_absolute_public_path(target, f"{label}.{field_name}", item.get(field_name))
        if item.get("kind") not in {"file", "directory"}:
            target.errors.append(f"{label}.kind must be file or directory.")
        for field_name in ("exists", "regular_file", "regular_directory", "symlink", "passed"):
            if not isinstance(item.get(field_name), bool):
                target.errors.append(f"{label}.{field_name} must be a boolean.")
        if item.get("expected_sha256") and not _is_sha256(item.get("expected_sha256")):
            target.errors.append(f"{label}.expected_sha256 must be a SHA-256 hex string when present.")
        if item.get("passed") is True:
            _validate_trainer_input_expected_payload(item, target, label)
            if item.get("kind") == "file":
                if item.get("regular_file") is not True or item.get("symlink") is True:
                    target.errors.append(f"{label}.passed file checks must be regular non-symlink files.")
                if not _is_non_negative_int(item.get("size_bytes")):
                    target.errors.append(f"{label}.size_bytes must be a non-negative integer when passed.")
                if not _is_sha256(item.get("sha256")):
                    target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when passed.")
                _validate_visible_file_hash(item, target, label)
            else:
                if item.get("regular_directory") is not True or item.get("symlink") is True:
                    target.errors.append(f"{label}.passed directory checks must be regular non-symlink directories.")
                if not _is_non_negative_int(item.get("size_bytes")):
                    target.errors.append(f"{label}.size_bytes must be a non-negative integer when passed.")
                if not _is_non_negative_int(item.get("file_count")):
                    target.errors.append(f"{label}.file_count must be a non-negative integer when passed.")
                if not _is_sha256(item.get("sha256")):
                    target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when passed.")
                _validate_visible_directory_hash(item, target, label)
    return records

def _validate_trainer_input_expected_payload(item: dict[str, Any], target: ValidationTarget, label: str) -> None:
    expected_sha = item.get("expected_sha256")
    if not _is_sha256(expected_sha):
        target.errors.append(f"{label}.expected_sha256 must be a SHA-256 hex string when passed.")
    elif _is_sha256(item.get("sha256")) and item.get("sha256") != expected_sha:
        target.errors.append(f"{label}.sha256 must match expected_sha256 when passed.")
    expected_size = item.get("expected_size_bytes")
    if not _is_non_negative_int(expected_size):
        target.errors.append(f"{label}.expected_size_bytes must be a non-negative integer when passed.")
    elif _is_non_negative_int(item.get("size_bytes")) and item.get("size_bytes") != expected_size:
        target.errors.append(f"{label}.size_bytes must match expected_size_bytes when passed.")
    if item.get("kind") == "directory":
        expected_file_count = item.get("expected_file_count")
        if not _is_non_negative_int(expected_file_count):
            target.errors.append(f"{label}.expected_file_count must be a non-negative integer for passed directories.")
        elif _is_non_negative_int(item.get("file_count")) and item.get("file_count") != expected_file_count:
            target.errors.append(f"{label}.file_count must match expected_file_count when passed.")

def _validate_trainer_archive_check_metrics(
    metrics: dict[str, Any],
    validation: dict[str, Any],
    external_code_checks: list[dict[str, Any]],
    trainer_input_checks: list[dict[str, Any]],
    check_count: int,
    failed_check_count: int,
    target: ValidationTarget,
) -> None:
    _validate_allowed_keys(metrics, _TRAINER_ARCHIVE_CHECK_METRICS_KEYS, target, "trainer_archive_check.metrics")
    external_count = len(external_code_checks)
    external_available = sum(1 for item in external_code_checks if item.get("passed") is True)
    input_count = len(trainer_input_checks)
    inputs_available = sum(1 for item in trainer_input_checks if item.get("passed") is True)
    expected = {
        "archive_validation_passed": validation.get("passed") is True,
        "archive_validation_error_count": _non_negative_int_value(validation.get("error_count")),
        "archive_validation_warning_count": _non_negative_int_value(validation.get("warning_count")),
        "external_command_path_count": external_count,
        "external_code_file_count": external_available,
        "missing_external_code_count": external_count - external_available,
        "trainer_input_count": input_count,
        "trainer_input_available_count": inputs_available,
        "missing_trainer_input_count": input_count - inputs_available,
        "check_count": check_count,
        "failed_check_count": failed_check_count,
    }
    for field_name, expected_value in expected.items():
        if metrics.get(field_name) != expected_value:
            target.errors.append(f"trainer_archive_check.metrics.{field_name} expected {expected_value}, got {metrics.get(field_name)!r}.")
    if not _is_non_negative_int(metrics.get("relative_external_command_path_count")):
        target.errors.append("trainer_archive_check.metrics.relative_external_command_path_count must be a non-negative integer.")

def _validate_visible_file_hash(item: dict[str, Any], target: ValidationTarget, label: str) -> None:
    path = _visible_local_path(item.get("resolved_path"))
    if path is None:
        return
    if not path.exists():
        target.errors.append(f"{label}.resolved_path must resolve to an existing file on disk.")
        return
    if path.is_symlink() or not path.is_file():
        target.errors.append(f"{label}.resolved_path is not a regular file on disk.")
        return
    if item.get("size_bytes") != path.stat().st_size:
        target.errors.append(f"{label}.size_bytes does not match resolved_path.")
    if item.get("sha256") != _sha256(path):
        target.errors.append(f"{label}.sha256 does not match resolved_path.")

def _validate_visible_directory_hash(item: dict[str, Any], target: ValidationTarget, label: str) -> None:
    path = _visible_local_path(item.get("resolved_path"))
    if path is None:
        return
    if not path.exists():
        target.errors.append(f"{label}.resolved_path must resolve to an existing directory on disk.")
        return
    if path.is_symlink() or not path.is_dir():
        target.errors.append(f"{label}.resolved_path is not a regular directory on disk.")
        return
    for child in path.rglob("*"):
        if child.is_symlink():
            target.errors.append(f"{label}.resolved_path contains symlink {child}.")
            return
    tree = _trainer_archive_tree_fingerprint(path)
    if item.get("file_count") != tree["file_count"]:
        target.errors.append(f"{label}.file_count does not match resolved_path.")
    if item.get("size_bytes") != tree["size_bytes"]:
        target.errors.append(f"{label}.size_bytes does not match resolved_path.")
    if item.get("sha256") != tree["sha256"]:
        target.errors.append(f"{label}.sha256 does not match resolved_path.")

def _visible_local_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value or value.startswith("<") or _is_windows_absolute(value):
        return None
    return Path(value)

_TRAINER_CONSUMER_PLAN_KEYS = {
    "schema_version",
    "plan_path",
    "archive_check_path",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "validation",
    "source_archive_check",
    "execution",
    "handoff_contract",
    "blocked_reasons",
    "metrics",
    "notes",
}

_TRAINER_CONSUMER_PLAN_CHECK_KEYS = {"id", "passed", "actual", "expected", "scope", "summary"}

_TRAINER_CONSUMER_PLAN_VALIDATION_KEYS = {
    "available",
    "passed",
    "strict",
    "target_count",
    "error_count",
    "warning_count",
    "errors",
    "warnings",
}

_TRAINER_CONSUMER_PLAN_SOURCE_KEYS = {
    "path",
    "schema_version",
    "passed",
    "readiness",
    "recommendation",
    "size_bytes",
    "sha256",
}

_TRAINER_CONSUMER_PLAN_EXECUTION_KEYS = {
    "execution_cwd",
    "archive_root",
    "external_code_root",
    "command_approved",
    "command_available",
    "command_argv",
    "command_shell",
    "external_code_files",
    "trainer_inputs",
}

_TRAINER_CONSUMER_PLAN_EXTERNAL_CODE_FILE_KEYS = {
    "index",
    "argv_index",
    "token",
    "path",
    "resolved_path",
    "exists",
    "regular_file",
    "symlink",
    "passed",
    "reason",
    "size_bytes",
    "sha256",
}

_TRAINER_CONSUMER_PLAN_TRAINER_INPUT_KEYS = {
    "index",
    "artifact_index",
    "artifact_name",
    "archive_path",
    "resolved_path",
    "kind",
    "exists",
    "regular_file",
    "regular_directory",
    "symlink",
    "expected_sha256",
    "expected_size_bytes",
    "expected_file_count",
    "file_count",
    "passed",
    "reason",
    "size_bytes",
    "sha256",
}

_TRAINER_CONSUMER_PLAN_HANDOFF_KEYS = {
    "flight_recorder_executed_command",
    "runner_owns_execution",
    "runner_must_run_from",
    "runner_must_require_recommendation",
    "trainer_input_count",
    "external_code_file_count",
    "allowed_input_sets",
    "notes",
}

_TRAINER_CONSUMER_PLAN_METRICS_KEYS = {
    "check_count",
    "failed_check_count",
    "command_arg_count",
    "trainer_input_count",
    "trainer_input_ready_count",
    "external_code_file_count",
    "external_code_ready_count",
    "archive_check_error_count",
    "archive_check_warning_count",
}

_TRAINER_WRAPPER_DRY_RUN_KEYS = {
    "schema_version",
    "wrapper",
    "plan_path",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "validation",
    "would_run",
    "inputs",
    "metrics",
    "notes",
}

_TRAINER_WRAPPER_DRY_RUN_CHECK_KEYS = {"id", "passed", "actual", "expected", "scope", "summary"}

_TRAINER_WRAPPER_DRY_RUN_CHECK_PAYLOAD_KEYS = {"passed"}

_TRAINER_WRAPPER_DRY_RUN_CHECK_SCOPE_KEYS = {
    "argc",
    "execution_cwd",
    "external_code_file_count",
    "mode",
    "plan",
    "recommendation",
    "runner_owns_execution",
    "schema_version",
    "trainer_input_count",
}

_TRAINER_WRAPPER_DRY_RUN_VALIDATION_KEYS = {
    "passed",
    "strict",
    "target_count",
    "error_count",
    "warning_count",
}

_TRAINER_WRAPPER_DRY_RUN_WOULD_RUN_KEYS = {
    "mode",
    "execution_cwd",
    "archive_root",
    "external_code_root",
    "argv",
    "shell",
}

_TRAINER_WRAPPER_DRY_RUN_INPUTS_KEYS = {"trainer_inputs", "external_code_files"}

_TRAINER_WRAPPER_DRY_RUN_EXTERNAL_CODE_FILE_KEYS = {
    "path",
    "resolved_path",
    "sha256",
    "passed",
    "size_bytes",
}

_TRAINER_WRAPPER_DRY_RUN_TRAINER_INPUT_KEYS = {
    "artifact_name",
    "archive_path",
    "resolved_path",
    "kind",
    "sha256",
    "expected_sha256",
    "passed",
    "expected_file_count",
    "expected_size_bytes",
    "file_count",
    "size_bytes",
}

_TRAINER_WRAPPER_DRY_RUN_METRICS_KEYS = {
    "command_arg_count",
    "trainer_input_count",
    "trainer_input_ready_count",
    "external_code_file_count",
    "external_code_ready_count",
}

def _validate_trainer_consumer_plan(plan: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(plan, "schema_version", TRAINER_CONSUMER_PLAN_SCHEMA_VERSION, target)
    _validate_allowed_keys(plan, _TRAINER_CONSUMER_PLAN_KEYS, target, "trainer_consumer_plan")
    for field_name in ("plan_path", "archive_check_path", "readiness", "recommendation"):
        if not isinstance(plan.get(field_name), str) or not plan.get(field_name):
            target.errors.append(f"trainer_consumer_plan.{field_name} must be a non-empty string.")
    if not isinstance(plan.get("passed"), bool):
        target.errors.append("trainer_consumer_plan.passed must be a boolean.")
    checks = plan.get("checks")
    if not isinstance(checks, list):
        target.errors.append("trainer_consumer_plan.checks must be a list.")
        checks = []
    validation = plan.get("validation")
    if not isinstance(validation, dict):
        target.errors.append("trainer_consumer_plan.validation must be an object.")
        validation = {}
    metrics = plan.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("trainer_consumer_plan.metrics must be an object.")
        metrics = {}
    if not _is_string_list(plan.get("notes")):
        target.errors.append("trainer_consumer_plan.notes must be a list of strings.")
    if not _is_string_list(plan.get("blocked_reasons")):
        target.errors.append("trainer_consumer_plan.blocked_reasons must be a list of strings.")

    for index, check in enumerate(checks):
        if isinstance(check, dict):
            _validate_allowed_keys(check, _TRAINER_CONSUMER_PLAN_CHECK_KEYS, target, f"trainer_consumer_plan.checks[{index}]")
    failed_checks = _validate_gate_like_checks(checks, target, "trainer_consumer_plan.checks")
    if plan.get("check_count") != len(checks):
        target.errors.append(f"trainer_consumer_plan.check_count expected {len(checks)}, got {plan.get('check_count')!r}.")
    if plan.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"trainer_consumer_plan.failed_check_count expected {failed_checks}, got {plan.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(plan.get("passed"), bool) and plan.get("passed") != expected_passed:
        target.errors.append("trainer_consumer_plan.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "ready_for_external_trainer" if expected_passed else "block_external_trainer"
    if plan.get("readiness") != expected_readiness:
        target.errors.append(f"trainer_consumer_plan.readiness expected {expected_readiness!r}, got {plan.get('readiness')!r}.")
    if plan.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"trainer_consumer_plan.recommendation expected {expected_recommendation!r}, got {plan.get('recommendation')!r}."
        )

    _validate_trainer_consumer_plan_validation(validation, target)
    _validate_trainer_consumer_plan_source(plan.get("source_archive_check"), target, source_path)
    execution_counts = _validate_trainer_consumer_plan_execution(plan.get("execution"), target)
    _validate_trainer_consumer_plan_handoff(plan.get("handoff_contract"), execution_counts, target)
    _validate_trainer_consumer_plan_metrics(metrics, validation, execution_counts, len(checks), failed_checks, target)
    target.details.update(
        {
            "passed": plan.get("passed"),
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "trainer_input_count": metrics.get("trainer_input_count"),
            "external_code_file_count": metrics.get("external_code_file_count"),
        }
    )

def _validate_trainer_consumer_plan_validation(value: dict[str, Any], target: ValidationTarget) -> None:
    _validate_allowed_keys(value, _TRAINER_CONSUMER_PLAN_VALIDATION_KEYS, target, "trainer_consumer_plan.validation")
    _validate_trainer_compact_validation_record(
        value,
        target,
        "trainer_consumer_plan.validation",
        has_available=True,
        has_messages=True,
    )

def _validate_trainer_consumer_plan_source(value: Any, target: ValidationTarget, source_path: Path) -> None:
    if not isinstance(value, dict):
        target.errors.append("trainer_consumer_plan.source_archive_check must be an object.")
        return
    _validate_allowed_keys(value, _TRAINER_CONSUMER_PLAN_SOURCE_KEYS, target, "trainer_consumer_plan.source_archive_check")
    for field_name in ("path", "schema_version", "readiness", "recommendation"):
        if not isinstance(value.get(field_name), str) or not value.get(field_name):
            target.errors.append(f"trainer_consumer_plan.source_archive_check.{field_name} must be a non-empty string.")
    if value.get("schema_version") != TRAINER_ARCHIVE_CHECK_SCHEMA_VERSION:
        target.errors.append(f"trainer_consumer_plan.source_archive_check.schema_version must be {TRAINER_ARCHIVE_CHECK_SCHEMA_VERSION}.")
    if not isinstance(value.get("passed"), bool):
        target.errors.append("trainer_consumer_plan.source_archive_check.passed must be a boolean.")
    if "size_bytes" in value and not _is_non_negative_int(value.get("size_bytes")):
        target.errors.append("trainer_consumer_plan.source_archive_check.size_bytes must be a non-negative integer when present.")
    if "sha256" in value and not _is_sha256(value.get("sha256")):
        target.errors.append("trainer_consumer_plan.source_archive_check.sha256 must be a SHA-256 hex string when present.")
    path = _trainer_consumer_plan_source_path(value.get("path"), source_path)
    if path is None:
        return
    if _path_has_symlink_component(path, include_leaf=True):
        target.errors.append("trainer_consumer_plan.source_archive_check.path must resolve to a regular non-symlink file.")
        return
    if not path.exists():
        target.errors.append("trainer_consumer_plan.source_archive_check.path must resolve to an existing file.")
        return
    if path.is_symlink() or not path.is_file():
        target.errors.append("trainer_consumer_plan.source_archive_check.path must resolve to a regular file.")
        return
    if not _is_non_negative_int(value.get("size_bytes")):
        target.errors.append("trainer_consumer_plan.source_archive_check.size_bytes must be present for visible path-backed refs.")
    elif value.get("size_bytes") != path.stat().st_size:
        target.errors.append("trainer_consumer_plan.source_archive_check.size_bytes does not match path.")
    if not _is_sha256(value.get("sha256")):
        target.errors.append("trainer_consumer_plan.source_archive_check.sha256 must be present for visible path-backed refs.")
    elif value.get("sha256") != _sha256(path):
        target.errors.append("trainer_consumer_plan.source_archive_check.sha256 does not match path.")

def _trainer_consumer_plan_source_path(value: Any, source_path: Path) -> Path | None:
    if not isinstance(value, str) or not value or value.startswith("<") or _is_windows_absolute(value):
        return None
    raw = Path(value)
    if raw.is_absolute():
        return raw
    source_dir = source_path.resolve().parent
    source_relative = source_dir / raw
    if source_relative.exists() or _path_has_symlink_component(source_relative, include_leaf=True):
        return source_relative
    repo_root = _repo_root_for_artifact(source_path)
    if repo_root is not None:
        return repo_root / raw
    return source_relative

def _validate_trainer_consumer_plan_execution(value: Any, target: ValidationTarget) -> dict[str, int]:
    counts = {
        "command_arg_count": 0,
        "trainer_input_count": 0,
        "trainer_input_ready_count": 0,
        "external_code_file_count": 0,
        "external_code_ready_count": 0,
    }
    if not isinstance(value, dict):
        target.errors.append("trainer_consumer_plan.execution must be an object.")
        return counts
    label = "trainer_consumer_plan.execution"
    _validate_allowed_keys(value, _TRAINER_CONSUMER_PLAN_EXECUTION_KEYS, target, label)
    for field_name in ("execution_cwd", "archive_root", "external_code_root", "command_shell"):
        if not isinstance(value.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    for field_name in ("archive_root", "external_code_root"):
        _validate_trainer_consumer_plan_command_path(target, f"{label}.{field_name}", value.get(field_name))
    if value.get("execution_cwd") != "archive_root":
        target.errors.append(f"{label}.execution_cwd must be archive_root.")
    for field_name in ("command_approved", "command_available"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    argv = value.get("command_argv")
    if not _is_string_list(argv):
        target.errors.append(f"{label}.command_argv must be a list of strings.")
        argv = []
    clean_argv = [item for item in argv if isinstance(item, str)]
    for index, item in enumerate(clean_argv):
        _validate_trainer_consumer_plan_command_token_public_path(target, f"{label}.command_argv[{index}]", item)
    counts["command_arg_count"] = len(clean_argv)
    expected_shell = shlex.join(clean_argv) if clean_argv else ""
    if value.get("command_shell") != expected_shell:
        target.errors.append(f"{label}.command_shell must match command_argv.")
    external_code_files = value.get("external_code_files")
    if not isinstance(external_code_files, list):
        target.errors.append(f"{label}.external_code_files must be a list.")
        external_code_files = []
    trainer_inputs = value.get("trainer_inputs")
    if not isinstance(trainer_inputs, list):
        target.errors.append(f"{label}.trainer_inputs must be a list.")
        trainer_inputs = []
    counts["external_code_file_count"] = len([item for item in external_code_files if isinstance(item, dict)])
    counts["trainer_input_count"] = len([item for item in trainer_inputs if isinstance(item, dict)])
    for index, item in enumerate(external_code_files):
        label = f"trainer_consumer_plan.execution.external_code_files[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_trainer_consumer_plan_external_code_file(item, target, label)
        if item.get("passed") is True:
            counts["external_code_ready_count"] += 1
    for index, item in enumerate(trainer_inputs):
        label = f"trainer_consumer_plan.execution.trainer_inputs[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_trainer_consumer_plan_trainer_input(item, target, label)
        if item.get("passed") is True:
            counts["trainer_input_ready_count"] += 1
    return counts

def _validate_trainer_consumer_plan_command_path(target: ValidationTarget, label: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        return
    if _is_safe_trainer_consumer_plan_command_path(value):
        return
    target.errors.append(f"{label} must be a safe relative path or redacted placeholder.")

def _validate_trainer_consumer_plan_command_token_public_path(target: ValidationTarget, label: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        return
    if _looks_absolute(value):
        target.errors.append(f"{label} must use a relative command token or redacted placeholder.")
        return
    _, separator, token_value = value.partition("=")
    if separator and _looks_absolute(token_value):
        target.errors.append(f"{label} must use a relative command token or redacted placeholder.")

def _is_safe_trainer_consumer_plan_command_path(value: str) -> bool:
    if _is_redacted_placeholder(value):
        return True
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return (
        bool(value)
        and not path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and "\\" not in value
        and ".." not in path.parts
        and all(not part.startswith("~") for part in path.parts)
    )

def _validate_trainer_consumer_plan_external_code_file(item: dict[str, Any], target: ValidationTarget, label: str) -> None:
    _validate_allowed_keys(item, _TRAINER_CONSUMER_PLAN_EXTERNAL_CODE_FILE_KEYS, target, label)
    for field_name in ("index", "argv_index"):
        if not _is_non_negative_int(item.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    for field_name in ("token", "path", "resolved_path", "reason"):
        if not isinstance(item.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    for field_name in ("exists", "regular_file", "symlink", "passed"):
        if not isinstance(item.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if item.get("passed") is True:
        if item.get("exists") is not True or item.get("regular_file") is not True or item.get("symlink") is True:
            target.errors.append(f"{label}.passed cannot be true unless the file is present, regular, and non-symlink.")
        if not _is_non_negative_int(item.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer when passed.")
        if not _is_sha256(item.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when passed.")
        _validate_visible_file_hash(item, target, label)

def _validate_trainer_consumer_plan_trainer_input(item: dict[str, Any], target: ValidationTarget, label: str) -> None:
    _validate_allowed_keys(item, _TRAINER_CONSUMER_PLAN_TRAINER_INPUT_KEYS, target, label)
    for field_name in ("index", "artifact_index"):
        if not _is_non_negative_int(item.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    for field_name in ("artifact_name", "archive_path", "resolved_path", "kind", "expected_sha256", "reason"):
        if not isinstance(item.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if item.get("kind") not in {"file", "directory"}:
        target.errors.append(f"{label}.kind must be file or directory.")
    for field_name in ("exists", "regular_file", "regular_directory", "symlink", "passed"):
        if not isinstance(item.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if item.get("expected_sha256") and not _is_sha256(item.get("expected_sha256")):
        target.errors.append(f"{label}.expected_sha256 must be a SHA-256 hex string when present.")
    if item.get("passed") is True:
        _validate_trainer_input_expected_payload(item, target, label)
        if not _is_non_negative_int(item.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer when passed.")
        if not _is_sha256(item.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when passed.")
        if item.get("kind") == "file":
            _validate_visible_file_hash(item, target, label)
        else:
            if not _is_non_negative_int(item.get("file_count")):
                target.errors.append(f"{label}.file_count must be a non-negative integer when passed.")
            _validate_visible_directory_hash(item, target, label)

def _validate_trainer_consumer_plan_handoff(value: Any, counts: dict[str, int], target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("trainer_consumer_plan.handoff_contract must be an object.")
        return
    _validate_allowed_keys(value, _TRAINER_CONSUMER_PLAN_HANDOFF_KEYS, target, "trainer_consumer_plan.handoff_contract")
    if value.get("flight_recorder_executed_command") is not False:
        target.errors.append("trainer_consumer_plan.handoff_contract.flight_recorder_executed_command must be false.")
    if value.get("runner_owns_execution") is not True:
        target.errors.append("trainer_consumer_plan.handoff_contract.runner_owns_execution must be true.")
    for field_name in ("runner_must_run_from", "runner_must_require_recommendation"):
        if not isinstance(value.get(field_name), str) or not value.get(field_name):
            target.errors.append(f"trainer_consumer_plan.handoff_contract.{field_name} must be a non-empty string.")
    if value.get("runner_must_run_from") != "archive_root":
        target.errors.append("trainer_consumer_plan.handoff_contract.runner_must_run_from must be archive_root.")
    if value.get("runner_must_require_recommendation") != "ready_for_external_trainer":
        target.errors.append(
            "trainer_consumer_plan.handoff_contract.runner_must_require_recommendation must be ready_for_external_trainer."
        )
    if value.get("trainer_input_count") != counts["trainer_input_count"]:
        target.errors.append(
            f"trainer_consumer_plan.handoff_contract.trainer_input_count expected {counts['trainer_input_count']}, "
            f"got {value.get('trainer_input_count')!r}."
        )
    if value.get("external_code_file_count") != counts["external_code_file_count"]:
        target.errors.append(
            f"trainer_consumer_plan.handoff_contract.external_code_file_count expected {counts['external_code_file_count']}, "
            f"got {value.get('external_code_file_count')!r}."
        )
    if not _is_string_list(value.get("allowed_input_sets")):
        target.errors.append("trainer_consumer_plan.handoff_contract.allowed_input_sets must be a list of strings.")
    if not _is_string_list(value.get("notes")):
        target.errors.append("trainer_consumer_plan.handoff_contract.notes must be a list of strings.")

def _validate_trainer_consumer_plan_metrics(
    metrics: dict[str, Any],
    validation: dict[str, Any],
    counts: dict[str, int],
    check_count: int,
    failed_check_count: int,
    target: ValidationTarget,
) -> None:
    _validate_allowed_keys(metrics, _TRAINER_CONSUMER_PLAN_METRICS_KEYS, target, "trainer_consumer_plan.metrics")
    expected = {
        "check_count": check_count,
        "failed_check_count": failed_check_count,
        "command_arg_count": counts["command_arg_count"],
        "trainer_input_count": counts["trainer_input_count"],
        "trainer_input_ready_count": counts["trainer_input_ready_count"],
        "external_code_file_count": counts["external_code_file_count"],
        "external_code_ready_count": counts["external_code_ready_count"],
        "archive_check_error_count": _non_negative_int_value(validation.get("error_count")),
        "archive_check_warning_count": _non_negative_int_value(validation.get("warning_count")),
    }
    for field_name, expected_value in expected.items():
        if metrics.get(field_name) != expected_value:
            target.errors.append(f"trainer_consumer_plan.metrics.{field_name} expected {expected_value}, got {metrics.get(field_name)!r}.")

def _validate_trainer_wrapper_dry_run(receipt: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(receipt, "schema_version", TRAINER_WRAPPER_DRY_RUN_SCHEMA_VERSION, target)
    _validate_allowed_keys(receipt, _TRAINER_WRAPPER_DRY_RUN_KEYS, target, "trainer_wrapper_dry_run")
    for field_name in ("wrapper", "plan_path", "readiness", "recommendation"):
        if not isinstance(receipt.get(field_name), str) or not receipt.get(field_name):
            target.errors.append(f"trainer_wrapper_dry_run.{field_name} must be a non-empty string.")
    if not isinstance(receipt.get("passed"), bool):
        target.errors.append("trainer_wrapper_dry_run.passed must be a boolean.")
    checks = receipt.get("checks")
    if not isinstance(checks, list):
        target.errors.append("trainer_wrapper_dry_run.checks must be a list.")
        checks = []
    validation = receipt.get("validation")
    if not isinstance(validation, dict):
        target.errors.append("trainer_wrapper_dry_run.validation must be an object.")
        validation = {}
    would_run = receipt.get("would_run")
    if not isinstance(would_run, dict):
        target.errors.append("trainer_wrapper_dry_run.would_run must be an object.")
        would_run = {}
    inputs = receipt.get("inputs")
    if not isinstance(inputs, dict):
        target.errors.append("trainer_wrapper_dry_run.inputs must be an object.")
        inputs = {}
    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("trainer_wrapper_dry_run.metrics must be an object.")
        metrics = {}
    if not _is_string_list(receipt.get("notes")):
        target.errors.append("trainer_wrapper_dry_run.notes must be a list of strings.")

    _validate_trainer_wrapper_checks(checks, target)
    failed_checks = _validate_gate_like_checks(checks, target, "trainer_wrapper_dry_run.checks")
    if receipt.get("check_count") != len(checks):
        target.errors.append(f"trainer_wrapper_dry_run.check_count expected {len(checks)}, got {receipt.get('check_count')!r}.")
    if receipt.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"trainer_wrapper_dry_run.failed_check_count expected {failed_checks}, got {receipt.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(receipt.get("passed"), bool) and receipt.get("passed") != expected_passed:
        target.errors.append("trainer_wrapper_dry_run.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "dry_run_ready" if expected_passed else "block_dry_run"
    if receipt.get("readiness") != expected_readiness:
        target.errors.append(f"trainer_wrapper_dry_run.readiness expected {expected_readiness!r}, got {receipt.get('readiness')!r}.")
    if receipt.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"trainer_wrapper_dry_run.recommendation expected {expected_recommendation!r}, got {receipt.get('recommendation')!r}."
        )

    _validate_trainer_wrapper_validation(validation, target)
    command_arg_count = _validate_trainer_wrapper_would_run(would_run, target)
    input_counts = _validate_trainer_wrapper_inputs(inputs, target)
    _validate_trainer_wrapper_metrics(metrics, command_arg_count, input_counts, target)
    target.details.update(
        {
            "passed": receipt.get("passed"),
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "trainer_input_count": metrics.get("trainer_input_count"),
            "external_code_file_count": metrics.get("external_code_file_count"),
        }
    )

def _validate_trainer_wrapper_checks(checks: list[Any], target: ValidationTarget) -> None:
    for index, check in enumerate(checks):
        label = f"trainer_wrapper_dry_run.checks[{index}]"
        if not isinstance(check, dict):
            continue
        _validate_allowed_keys(check, _TRAINER_WRAPPER_DRY_RUN_CHECK_KEYS, target, label)
        _validate_trainer_wrapper_check_payload(check.get("actual"), target, f"{label}.actual")
        _validate_trainer_wrapper_check_payload(check.get("expected"), target, f"{label}.expected")
        scope = check.get("scope")
        if not isinstance(scope, dict):
            target.errors.append(f"{label}.scope must be an object.")
            continue
        _validate_allowed_keys(scope, _TRAINER_WRAPPER_DRY_RUN_CHECK_SCOPE_KEYS, target, f"{label}.scope")
        for field_name, field_value in scope.items():
            if not isinstance(field_value, str):
                target.errors.append(f"{label}.scope.{field_name} must be a string.")

def _validate_trainer_wrapper_check_payload(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _TRAINER_WRAPPER_DRY_RUN_CHECK_PAYLOAD_KEYS, target, label)
    if not isinstance(value.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")

def _validate_trainer_wrapper_validation(value: dict[str, Any], target: ValidationTarget) -> None:
    _validate_allowed_keys(value, _TRAINER_WRAPPER_DRY_RUN_VALIDATION_KEYS, target, "trainer_wrapper_dry_run.validation")
    _validate_trainer_compact_validation_record(
        value,
        target,
        "trainer_wrapper_dry_run.validation",
        has_available=False,
        has_messages=False,
    )

def _validate_trainer_compact_validation_record(
    value: dict[str, Any],
    target: ValidationTarget,
    label: str,
    *,
    has_available: bool,
    has_messages: bool,
) -> None:
    bool_fields = ("available", "passed", "strict") if has_available else ("passed", "strict")
    for field_name in bool_fields:
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    for field_name in ("target_count", "error_count", "warning_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    if has_messages:
        if not _is_string_list(value.get("errors")):
            target.errors.append(f"{label}.errors must be a list of strings.")
        if not _is_string_list(value.get("warnings")):
            target.errors.append(f"{label}.warnings must be a list of strings.")

    if has_available and value.get("available") is False and value.get("passed") is True:
        target.errors.append(f"{label}.passed cannot be true when validation is unavailable.")
    if not all(_is_non_negative_int(value.get(field_name)) for field_name in ("target_count", "error_count", "warning_count")):
        return
    if not isinstance(value.get("passed"), bool) or not isinstance(value.get("strict"), bool):
        return
    expected_passed = (
        value["target_count"] > 0
        and value["error_count"] == 0
        and (value["warning_count"] == 0 or value.get("strict") is not True)
    )
    if value["passed"] != expected_passed:
        target.errors.append(f"{label}.passed expected {expected_passed}, got {value.get('passed')!r}.")
    if has_messages:
        if value["error_count"] > 0 and isinstance(value.get("errors"), list) and not value["errors"]:
            target.errors.append(f"{label}.errors must explain nonzero error_count.")
        if value["warning_count"] > 0 and isinstance(value.get("warnings"), list) and not value["warnings"]:
            target.errors.append(f"{label}.warnings must explain nonzero warning_count.")

def _validate_trainer_wrapper_would_run(value: dict[str, Any], target: ValidationTarget) -> int:
    label = "trainer_wrapper_dry_run.would_run"
    _validate_allowed_keys(value, _TRAINER_WRAPPER_DRY_RUN_WOULD_RUN_KEYS, target, label)
    for field_name in ("mode", "execution_cwd", "archive_root", "external_code_root", "shell"):
        if not isinstance(value.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    for field_name in ("archive_root", "external_code_root"):
        _validate_trainer_wrapper_would_run_path(target, f"{label}.{field_name}", value.get(field_name))
    if value.get("mode") != "dry_run":
        target.errors.append(f"{label}.mode must be dry_run.")
    if value.get("execution_cwd") not in {"", "archive_root"}:
        target.errors.append(f"{label}.execution_cwd must be archive_root for ready receipts.")
    argv = value.get("argv")
    if not _is_string_list(argv):
        target.errors.append(f"{label}.argv must be a list of strings.")
        argv = []
    clean_argv = [item for item in argv if isinstance(item, str)]
    for index, item in enumerate(clean_argv):
        _validate_trainer_wrapper_would_run_token_public_path(target, f"{label}.argv[{index}]", item)
    expected_shell = shlex.join(clean_argv) if clean_argv else ""
    if value.get("shell") != expected_shell:
        target.errors.append(f"{label}.shell must match argv.")
    return len(clean_argv)

def _validate_trainer_wrapper_would_run_path(target: ValidationTarget, label: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        return
    if _is_safe_trainer_wrapper_would_run_path(value):
        return
    target.errors.append(f"{label} must be a safe relative path or redacted placeholder.")

def _validate_trainer_wrapper_would_run_token_public_path(target: ValidationTarget, label: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        return
    if _looks_absolute(value):
        target.errors.append(f"{label} must use a relative command token or redacted placeholder.")
        return
    _, separator, token_value = value.partition("=")
    if separator and _looks_absolute(token_value):
        target.errors.append(f"{label} must use a relative command token or redacted placeholder.")

def _is_safe_trainer_wrapper_would_run_path(value: str) -> bool:
    if _is_redacted_placeholder(value):
        return True
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return (
        bool(value)
        and not path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and "\\" not in value
        and ".." not in path.parts
        and all(not part.startswith("~") for part in path.parts)
    )

def _validate_trainer_wrapper_inputs(value: dict[str, Any], target: ValidationTarget) -> dict[str, int]:
    _validate_allowed_keys(value, _TRAINER_WRAPPER_DRY_RUN_INPUTS_KEYS, target, "trainer_wrapper_dry_run.inputs")
    trainer_inputs = value.get("trainer_inputs")
    if not isinstance(trainer_inputs, list):
        target.errors.append("trainer_wrapper_dry_run.inputs.trainer_inputs must be a list.")
        trainer_inputs = []
    external_code_files = value.get("external_code_files")
    if not isinstance(external_code_files, list):
        target.errors.append("trainer_wrapper_dry_run.inputs.external_code_files must be a list.")
        external_code_files = []
    counts = {
        "trainer_input_count": 0,
        "trainer_input_ready_count": 0,
        "external_code_file_count": 0,
        "external_code_ready_count": 0,
    }
    for index, item in enumerate(trainer_inputs):
        label = f"trainer_wrapper_dry_run.inputs.trainer_inputs[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        counts["trainer_input_count"] += 1
        _validate_trainer_wrapper_input(item, target, label)
        if item.get("passed") is True:
            counts["trainer_input_ready_count"] += 1
    for index, item in enumerate(external_code_files):
        label = f"trainer_wrapper_dry_run.inputs.external_code_files[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        counts["external_code_file_count"] += 1
        _validate_trainer_wrapper_external_code(item, target, label)
        if item.get("passed") is True:
            counts["external_code_ready_count"] += 1
    return counts

def _validate_trainer_wrapper_input(item: dict[str, Any], target: ValidationTarget, label: str) -> None:
    _validate_allowed_keys(item, _TRAINER_WRAPPER_DRY_RUN_TRAINER_INPUT_KEYS, target, label)
    for field_name in ("artifact_name", "archive_path", "resolved_path", "kind", "sha256"):
        if not isinstance(item.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if item.get("kind") not in {"file", "directory"}:
        target.errors.append(f"{label}.kind must be file or directory.")
    if not isinstance(item.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    if item.get("sha256") and not _is_sha256(item.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when present.")
    if "size_bytes" in item and not _is_non_negative_int(item.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer when present.")
    if "file_count" in item and not _is_non_negative_int(item.get("file_count")):
        target.errors.append(f"{label}.file_count must be a non-negative integer when present.")
    if item.get("passed") is True:
        if not _is_non_negative_int(item.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer when passed.")
        if not _is_sha256(item.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when passed.")
        if item.get("kind") == "directory" and not _is_non_negative_int(item.get("file_count")):
            target.errors.append(f"{label}.file_count must be a non-negative integer when passed.")
        _validate_trainer_input_expected_payload(item, target, label)
        if item.get("kind") == "file":
            _validate_visible_file_hash(item, target, label)
        elif item.get("kind") == "directory":
            _validate_visible_directory_hash(item, target, label)

def _validate_trainer_wrapper_external_code(item: dict[str, Any], target: ValidationTarget, label: str) -> None:
    _validate_allowed_keys(item, _TRAINER_WRAPPER_DRY_RUN_EXTERNAL_CODE_FILE_KEYS, target, label)
    for field_name in ("path", "resolved_path", "sha256"):
        if not isinstance(item.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if not isinstance(item.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    if item.get("sha256") and not _is_sha256(item.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when present.")
    if "size_bytes" in item and not _is_non_negative_int(item.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer when present.")
    if item.get("passed") is True:
        if not _is_non_negative_int(item.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer when passed.")
        if not _is_sha256(item.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when passed.")
        _validate_visible_file_hash(item, target, label)

def _validate_trainer_wrapper_metrics(
    metrics: dict[str, Any],
    command_arg_count: int,
    input_counts: dict[str, int],
    target: ValidationTarget,
) -> None:
    _validate_allowed_keys(metrics, _TRAINER_WRAPPER_DRY_RUN_METRICS_KEYS, target, "trainer_wrapper_dry_run.metrics")
    expected = {
        "command_arg_count": command_arg_count,
        "trainer_input_count": input_counts["trainer_input_count"],
        "trainer_input_ready_count": input_counts["trainer_input_ready_count"],
        "external_code_file_count": input_counts["external_code_file_count"],
        "external_code_ready_count": input_counts["external_code_ready_count"],
    }
    for field_name, expected_value in expected.items():
        if metrics.get(field_name) != expected_value:
            target.errors.append(f"trainer_wrapper_dry_run.metrics.{field_name} expected {expected_value}, got {metrics.get(field_name)!r}.")

def _trainer_input_from_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {
        "artifact_index": artifact.get("index"),
        "artifact_name": artifact.get("name"),
        "kind": artifact.get("kind"),
        "original_path": artifact.get("original_path"),
        "archive_path": artifact.get("path"),
        "size_bytes": artifact.get("size_bytes"),
        "sha256": artifact.get("sha256"),
    }
    if artifact.get("kind") == "directory":
        item["file_count"] = artifact.get("file_count")
        item["tree_hash_algorithm"] = artifact.get("tree_hash_algorithm")
    return item

def _expected_trainer_archive_rewrites(inputs: list[Any]) -> list[dict[str, Any]]:
    rewrites: list[dict[str, Any]] = []
    for item in inputs:
        if not isinstance(item, dict):
            continue
        original = item.get("original_path")
        archive_path = item.get("archive_path")
        if not isinstance(original, str) or not original or original.startswith("<"):
            continue
        if not isinstance(archive_path, str) or not archive_path:
            continue
        rewrites.append(
            {
                "artifact_name": str(item.get("artifact_name") or ""),
                "kind": str(item.get("kind") or ""),
                "original_path": original,
                "archive_path": archive_path,
            }
        )
    return rewrites

def _rewrite_trainer_archive_command_argv(argv: list[str], rewrites: list[Any]) -> tuple[list[str], int]:
    valid_rewrites = [item for item in rewrites if isinstance(item, dict)]
    ordered = sorted(valid_rewrites, key=lambda item: len(str(item.get("original_path") or "")), reverse=True)
    rewritten: list[str] = []
    rewrite_count = 0
    for token in argv:
        new_token = _rewrite_trainer_archive_command_token(token, ordered)
        if new_token != token:
            rewrite_count += 1
        rewritten.append(new_token)
    return rewritten, rewrite_count

def _rewrite_trainer_archive_command_token(token: str, rewrites: list[dict[str, Any]]) -> str:
    for item in rewrites:
        original = item.get("original_path")
        archive_path = item.get("archive_path")
        if not isinstance(original, str) or not original:
            continue
        if not isinstance(archive_path, str) or not archive_path:
            continue
        replacement = _replace_trainer_archive_path_value(token, original, archive_path)
        if replacement != token:
            return replacement
        if "=" in token:
            key, value = token.split("=", 1)
            rewritten_value = _replace_trainer_archive_path_value(value, original, archive_path)
            if rewritten_value != value:
                return f"{key}={rewritten_value}"
    return token

def _replace_trainer_archive_path_value(value: str, original: str, archive_path: str) -> str:
    if value == original:
        return archive_path
    prefix = original.rstrip("/") + "/"
    if value.startswith(prefix):
        return archive_path.rstrip("/") + "/" + value[len(prefix) :]
    return value

def _trainer_archive_external_command_paths(argv: list[str], trainer_inputs: list[Any]) -> list[dict[str, Any]]:
    archive_paths = [
        str(item.get("archive_path") or "")
        for item in trainer_inputs
        if isinstance(item, dict) and isinstance(item.get("archive_path"), str)
    ]
    external: list[dict[str, Any]] = []
    for index, token in enumerate(argv):
        if not token or token.startswith("-"):
            if "=" in token:
                key, value = token.split("=", 1)
                if _trainer_archive_is_external_command_path(value, archive_paths):
                    external.append({"argv_index": index, "token": token, "path": value, "reason": f"{key} references a path outside archive inputs"})
            continue
        if _trainer_archive_is_external_command_path(token, archive_paths):
            external.append({"argv_index": index, "token": token, "path": token, "reason": "path-like token is not one of the copied trainer inputs"})
    return external

def _trainer_archive_is_external_command_path(value: str, archive_paths: list[str]) -> bool:
    if not _trainer_archive_looks_like_path(value):
        return False
    normalized = value.replace("\\", "/")
    for archive_path in archive_paths:
        clean = archive_path.replace("\\", "/").rstrip("/")
        if normalized == clean or normalized.startswith(clean + "/"):
            return False
    return True

def _trainer_archive_looks_like_path(value: str) -> bool:
    if not value or value.startswith("-"):
        return False
    normalized = value.replace("\\", "/")
    if normalized.startswith(("./", "../", "/", "~")) or "/" in normalized:
        return True
    return Path(normalized).suffix.lower() in {
        ".py",
        ".sh",
        ".bash",
        ".js",
        ".mjs",
        ".ts",
        ".ipynb",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".csv",
        ".txt",
    }

def _trainer_archive_count_map(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts

def _trainer_archive_tree_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_hash = _sha256(path)
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        size_bytes += size
    return {"sha256": digest.hexdigest(), "file_count": file_count, "size_bytes": size_bytes}

_TRAINER_PREFLIGHT_KEYS = {
    "schema_version",
    "preflight_path",
    "passed",
    "readiness",
    "recommendation",
    "gate_count",
    "passed_gate_count",
    "required_gates",
    "required_dataset_versions",
    "check_count",
    "failed_check_count",
    "checks",
    "gates",
    "validation_summaries",
    "schema_contracts",
    "artifacts",
    "dataset_selection",
    "trainer_command",
    "metadata",
    "notes",
}

_TRAINER_PREFLIGHT_CHECK_KEYS = {"id", "passed", "actual", "expected", "scope", "summary"}

_TRAINER_PREFLIGHT_GATE_KEYS = {
    "id",
    "path",
    "exists",
    "schema_version",
    "passed",
    "size_bytes",
    "sha256",
    "validation",
}

_TRAINER_PREFLIGHT_GATE_VALIDATION_KEYS = {
    "available",
    "passed",
    "strict",
    "error_count",
    "warning_count",
    "source",
    "target_type",
    "summary_passed",
}

_TRAINER_PREFLIGHT_ARTIFACT_KEYS = {
    "path",
    "exists",
    "kind",
    "regular_file",
    "regular_directory",
    "symlink",
    "entry_count",
    "file_count",
    "size_bytes",
    "sha256",
    "tree_hash_algorithm",
}

_TRAINER_PREFLIGHT_SCHEMA_CONTRACT_KEYS = {
    "path",
    "exists",
    "kind",
    "schema_name",
    "regular_file",
    "symlink",
    "passed",
    "error_count",
    "errors",
    "row_count",
    "row_schema_counts",
    "size_bytes",
    "sha256",
}

_TRAINER_PREFLIGHT_ROW_SCHEMA_COUNT_KEYS = {"name", "count"}

_TRAINER_PREFLIGHT_VALIDATION_SUMMARY_KEYS = {
    "path",
    "exists",
    "kind",
    "regular_file",
    "symlink",
    "schema_version",
    "passed",
    "strict",
    "target_count",
    "error_count",
    "warning_count",
    "targets",
    "size_bytes",
    "sha256",
}

_TRAINER_PREFLIGHT_VALIDATION_TARGET_KEYS = {
    "type",
    "path",
    "passed",
    "error_count",
    "warning_count",
}

_TRAINER_PREFLIGHT_DATASET_SELECTION_KEYS = {
    "artifact",
    "root",
    "manifest_path",
    "manifest_sha256",
    "registry_path",
    "registry_sha256",
    "dataset_version",
    "required_dataset_versions",
    "matches_required",
    "registry_dataset_version",
    "registry_selection_key",
    "registry_manifest_sha256",
    "redaction_passed",
    "trainer_views",
    "trainer_modes",
    "heldout_scenario_exclusive",
    "heldout_scenario_ids",
}

_TRAINER_PREFLIGHT_TRAINER_VIEWS_KEYS = {"contract_version", "mode_to_view", "root_views", "views", "notes"}

_TRAINER_PREFLIGHT_TRAINER_VIEW_KEYS = {
    "view_id",
    "training_modes",
    "artifact",
    "artifact_path",
    "artifact_format",
    "schema_version",
    "row_count",
    "source_artifacts",
    "label_policy",
    "available",
    "split_paths",
    "notes",
}

_TRAINER_PREFLIGHT_TRAINER_COMMAND_KEYS = {"provided", "raw", "argv", "parseable"}

def _validate_trainer_preflight(preflight: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(preflight, "schema_version", TRAINER_PREFLIGHT_SCHEMA_VERSION, target)
    _validate_allowed_keys(preflight, _TRAINER_PREFLIGHT_KEYS, target, "trainer_preflight")
    if not isinstance(preflight.get("preflight_path"), str) or not preflight.get("preflight_path"):
        target.errors.append("trainer_preflight.preflight_path must be a non-empty string.")
    if not isinstance(preflight.get("passed"), bool):
        target.errors.append("trainer_preflight.passed must be a boolean.")
    checks = preflight.get("checks")
    if not isinstance(checks, list):
        target.errors.append("trainer_preflight.checks must be a list.")
        checks = []
    gates = preflight.get("gates")
    if not isinstance(gates, list):
        target.errors.append("trainer_preflight.gates must be a list.")
        gates = []
    artifacts = preflight.get("artifacts")
    if not isinstance(artifacts, dict):
        target.errors.append("trainer_preflight.artifacts must be an object.")
        artifacts = {}
    schema_contracts = preflight.get("schema_contracts", {})
    if not isinstance(schema_contracts, dict):
        target.errors.append("trainer_preflight.schema_contracts must be an object when present.")
        schema_contracts = {}
    if not artifacts:
        target.errors.append("trainer_preflight.artifacts must not be empty.")
    if not _is_string_list(preflight.get("required_gates")):
        target.errors.append("trainer_preflight.required_gates must be a list of strings.")
    if not _is_string_list(preflight.get("required_dataset_versions")):
        target.errors.append("trainer_preflight.required_dataset_versions must be a list of strings.")
    if not _is_string_list(preflight.get("notes")):
        target.errors.append("trainer_preflight.notes must be a list of strings.")
    if "metadata" in preflight:
        _validate_metadata(preflight.get("metadata"), target, "trainer_preflight.metadata")
    validation_summaries = preflight.get("validation_summaries", [])
    _validate_trainer_preflight_validation_summaries(validation_summaries, target, source_path)

    for index, check in enumerate(checks):
        if isinstance(check, dict):
            _validate_allowed_keys(check, _TRAINER_PREFLIGHT_CHECK_KEYS, target, f"trainer_preflight.checks[{index}]")
    failed_checks = _validate_handoff_checks(checks, target, "trainer_preflight.checks")
    if preflight.get("check_count") != len(checks):
        target.errors.append(f"trainer_preflight.check_count expected {len(checks)}, got {preflight.get('check_count')!r}.")
    if preflight.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"trainer_preflight.failed_check_count expected {failed_checks}, got {preflight.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(preflight.get("passed"), bool) and preflight.get("passed") != expected_passed:
        target.errors.append("trainer_preflight.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "launch_allowed" if expected_passed else "block_launch"
    if preflight.get("readiness") != expected_readiness:
        target.errors.append(f"trainer_preflight.readiness expected {expected_readiness!r}, got {preflight.get('readiness')!r}.")
    if preflight.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"trainer_preflight.recommendation expected {expected_recommendation!r}, got {preflight.get('recommendation')!r}."
        )

    passed_gates = 0
    for index, gate in enumerate(gates):
        if _validate_trainer_preflight_gate(gate, target, f"trainer_preflight.gates[{index}]", source_path):
            passed_gates += 1
    if preflight.get("gate_count") != len(gates):
        target.errors.append(f"trainer_preflight.gate_count expected {len(gates)}, got {preflight.get('gate_count')!r}.")
    if preflight.get("passed_gate_count") != passed_gates:
        target.errors.append(
            f"trainer_preflight.passed_gate_count expected {passed_gates}, got {preflight.get('passed_gate_count')!r}."
        )
    for name, record in artifacts.items():
        _validate_trainer_preflight_artifact_record(name, record, target, source_path)
    for name, record in schema_contracts.items():
        _validate_trainer_preflight_schema_contract_record(name, record, target, source_path)
    _validate_trainer_preflight_control_artifact_declarations(
        checks,
        artifacts,
        schema_contracts,
        target,
    )
    dataset_selection = preflight.get("dataset_selection", [])
    _validate_trainer_preflight_dataset_selection(dataset_selection, target)
    _validate_trainer_preflight_reviewed_binding(gates, artifacts, target, source_path)
    _validate_trainer_command(preflight.get("trainer_command"), target)
    target.details.update(
        {
            "readiness": preflight.get("readiness"),
            "gate_count": len(gates),
            "failed_check_count": failed_checks,
            "artifact_count": len(artifacts),
            "schema_contract_count": len(schema_contracts),
            "dataset_selection_count": len(dataset_selection) if isinstance(dataset_selection, list) else 0,
            "validation_summary_count": len(validation_summaries) if isinstance(validation_summaries, list) else 0,
        }
    )

_TRAINER_LAUNCH_CHECK_KEYS = {
    "schema_version",
    "preflight_path",
    "source_artifacts",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "validation",
    "required_gates",
    "required_dataset_versions",
    "required_metadata",
    "gates",
    "gate_count",
    "passed_gate_count",
    "artifacts",
    "dataset_selection",
    "approved_command",
    "notes",
}

_TRAINER_LAUNCH_CHECK_CHECK_KEYS = {"id", "passed", "actual", "expected", "scope", "summary"}

_TRAINER_LAUNCH_CHECK_GATE_KEYS = {"id", "path", "schema_version", "passed"}

_TRAINER_LAUNCH_CHECK_VALIDATION_KEYS = {
    "passed",
    "strict",
    "target_count",
    "error_count",
    "warning_count",
    "errors",
    "warnings",
}

_TRAINER_LAUNCH_CHECK_ARTIFACTS_KEYS = {"count", "names"}

_TRAINER_LAUNCH_CHECK_APPROVED_COMMAND_KEYS = {
    "approved",
    "provided",
    "raw",
    "argv",
    "parseable",
    "shell",
}

def _validate_trainer_launch_check(
    launch_check: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    _require_equal(launch_check, "schema_version", TRAINER_LAUNCH_CHECK_SCHEMA_VERSION, target)
    _validate_allowed_keys(launch_check, _TRAINER_LAUNCH_CHECK_KEYS, target, "trainer_launch_check")
    if not isinstance(launch_check.get("preflight_path"), str) or not launch_check.get("preflight_path"):
        target.errors.append("trainer_launch_check.preflight_path must be a non-empty string.")
    if not isinstance(launch_check.get("passed"), bool):
        target.errors.append("trainer_launch_check.passed must be a boolean.")
    checks = launch_check.get("checks")
    if not isinstance(checks, list):
        target.errors.append("trainer_launch_check.checks must be a list.")
        checks = []
    for index, check in enumerate(checks):
        if isinstance(check, dict):
            _validate_allowed_keys(check, _TRAINER_LAUNCH_CHECK_CHECK_KEYS, target, f"trainer_launch_check.checks[{index}]")
    failed_checks = _validate_handoff_checks(checks, target, "trainer_launch_check.checks")
    if launch_check.get("check_count") != len(checks):
        target.errors.append(f"trainer_launch_check.check_count expected {len(checks)}, got {launch_check.get('check_count')!r}.")
    if launch_check.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"trainer_launch_check.failed_check_count expected {failed_checks}, got {launch_check.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(launch_check.get("passed"), bool) and launch_check.get("passed") != expected_passed:
        target.errors.append("trainer_launch_check.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "launch_allowed" if expected_passed else "block_launch"
    if launch_check.get("readiness") != expected_readiness:
        target.errors.append(
            f"trainer_launch_check.readiness expected {expected_readiness!r}, got {launch_check.get('readiness')!r}."
        )
    if launch_check.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"trainer_launch_check.recommendation expected {expected_recommendation!r}, got {launch_check.get('recommendation')!r}."
        )
    if not _is_string_list(launch_check.get("required_gates")):
        target.errors.append("trainer_launch_check.required_gates must be a list of strings.")
    if not _is_string_list(launch_check.get("required_dataset_versions")):
        target.errors.append("trainer_launch_check.required_dataset_versions must be a list of strings.")
    _validate_trainer_preflight_dataset_selection(
        launch_check.get("dataset_selection", []),
        target,
        label="trainer_launch_check.dataset_selection",
    )
    required_metadata = launch_check.get("required_metadata")
    if not isinstance(required_metadata, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in required_metadata.items()):
        target.errors.append("trainer_launch_check.required_metadata must be an object of string values.")
    gates = launch_check.get("gates")
    if not isinstance(gates, list):
        target.errors.append("trainer_launch_check.gates must be a list.")
        gates = []
    passed_gates = 0
    for index, gate in enumerate(gates):
        if _validate_launch_check_gate(gate, target, f"trainer_launch_check.gates[{index}]"):
            passed_gates += 1
    if launch_check.get("gate_count") != len(gates):
        target.errors.append(f"trainer_launch_check.gate_count expected {len(gates)}, got {launch_check.get('gate_count')!r}.")
    if launch_check.get("passed_gate_count") != passed_gates:
        target.errors.append(
            f"trainer_launch_check.passed_gate_count expected {passed_gates}, got {launch_check.get('passed_gate_count')!r}."
        )
    _validate_launch_validation_record(launch_check.get("validation"), target)
    _validate_launch_artifacts_summary(launch_check.get("artifacts"), target)
    _validate_approved_command(launch_check.get("approved_command"), target, expected_passed)
    if not _is_string_list(launch_check.get("notes")):
        target.errors.append("trainer_launch_check.notes must be a list of strings.")
    _validate_trainer_launch_source_and_replay(launch_check, target, source_path)
    target.details.update(
        {
            "readiness": launch_check.get("readiness"),
            "gate_count": len(gates),
            "failed_check_count": failed_checks,
        }
    )

def _validate_trainer_launch_source_and_replay(
    launch_check: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    from .dispatch import validate_artifacts
    source_artifacts = launch_check.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        target.errors.append("trainer_launch_check.source_artifacts must be an object.")
        return
    _validate_allowed_keys(
        source_artifacts,
        {"trainer_preflight"},
        target,
        "trainer_launch_check.source_artifacts",
    )
    source_record = source_artifacts.get("trainer_preflight")
    if not isinstance(source_record, dict):
        target.errors.append(
            "trainer_launch_check.source_artifacts.trainer_preflight must be an object."
        )
        return
    label = "trainer_launch_check.source_artifacts.trainer_preflight"
    _validate_allowed_keys(
        source_record,
        set(TRAINER_PREFLIGHT_SOURCE_ARTIFACT_FIELDS),
        target,
        label,
    )
    if launch_check.get("preflight_path") != source_record.get("path"):
        target.errors.append(
            "trainer_launch_check.preflight_path must match source_artifacts.trainer_preflight.path."
        )
    for field_name, expected in (
        ("kind", "file"),
        ("exists", True),
        ("regular_file", True),
        ("symlink", False),
        ("schema_version", TRAINER_PREFLIGHT_SCHEMA_VERSION),
    ):
        if source_record.get(field_name) != expected:
            target.errors.append(f"{label}.{field_name} must be {expected!r}.")
    if not _is_non_negative_int(source_record.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    if not _is_lowercase_sha256(source_record.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a lowercase SHA-256 hex string.")

    preflight_path = _resolve_trainer_launch_preflight_path(
        source_record.get("path"),
        source_path,
        target,
    )
    if preflight_path is None:
        return
    display_path = str(source_record.get("path"))
    try:
        source_before = build_trainer_preflight_source_artifact(
            preflight_path,
            display_path=display_path,
        )
    except TrainerPreflightError as exc:
        target.errors.append(f"trainer_launch_check trainer preflight could not be attested: {exc}")
        return
    if source_record != source_before:
        target.errors.append(
            "trainer_launch_check.source_artifacts.trainer_preflight must match the current trainer-preflight bytes."
        )

    current_preflight = _read_object(
        preflight_path,
        target,
        "trainer_launch_check trainer_preflight.json",
    )
    if current_preflight is None:
        return
    validation_record = launch_check.get("validation")
    strict = bool(
        isinstance(validation_record, dict)
        and validation_record.get("strict") is True
    )
    current_validation = validate_artifacts(
        trainer_preflight_paths=[preflight_path],
        strict=strict,
    )
    for validation_target in current_validation.get("targets", []):
        if not isinstance(validation_target, dict):
            continue
        for error in validation_target.get("errors", []):
            if isinstance(error, str):
                target.errors.append(f"trainer_launch_check trainer preflight: {error}")

    required_gates = launch_check.get("required_gates")
    required_dataset_versions = launch_check.get("required_dataset_versions")
    required_metadata = launch_check.get("required_metadata")
    try:
        replayed = build_trainer_launch_check(
            preflight_path=preflight_path,
            preflight=current_preflight,
            validation_summary=current_validation,
            out_path=source_path,
            require_gates=(
                [item for item in required_gates if isinstance(item, str)]
                if isinstance(required_gates, list)
                else []
            ),
            required_dataset_versions=(
                [item for item in required_dataset_versions if isinstance(item, str)]
                if isinstance(required_dataset_versions, list)
                else []
            ),
            require_metadata=(
                {
                    str(key): str(value)
                    for key, value in required_metadata.items()
                    if isinstance(key, str) and isinstance(value, str)
                }
                if isinstance(required_metadata, dict)
                else {}
            ),
            preserve_paths=False,
        )
    except TrainerPreflightError as exc:
        target.errors.append(
            f"trainer_launch_check could not replay its current trainer preflight: {exc}"
        )
    else:
        if launch_check != replayed:
            target.errors.append(
                "trainer_launch_check must match deterministic replay of its current trainer preflight, requirements, and strictness exactly."
            )

    try:
        source_after = build_trainer_preflight_source_artifact(
            preflight_path,
            display_path=display_path,
        )
    except TrainerPreflightError as exc:
        target.errors.append(f"trainer_launch_check trainer preflight could not be reattested: {exc}")
    else:
        if source_after != source_before:
            target.errors.append(
                "trainer_launch_check trainer preflight changed while validation was running."
            )
    target.details.update(
        {
            "preflight_sha256": source_record.get("sha256"),
            "preflight_schema_version": source_record.get("schema_version"),
        }
    )

def _resolve_trainer_launch_preflight_path(
    value: Any,
    source_path: Path,
    target: ValidationTarget,
) -> Path | None:
    label = "trainer_launch_check.source_artifacts.trainer_preflight.path"
    if not isinstance(value, str) or not value:
        target.errors.append(f"{label} must be a non-empty string.")
        return None
    windows_path = PureWindowsPath(value)
    parts = value.split("/")
    seen_named_part = False
    canonical = True
    for part in parts:
        if part in {"", "."} or part.startswith("~"):
            canonical = False
            break
        if part == "..":
            if seen_named_part:
                canonical = False
                break
        else:
            seen_named_part = True
    if (
        Path(value).is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or "\\" in value
        or "\x00" in value
        or "://" in value
        or value.startswith("<redacted:")
        or not canonical
        or not seen_named_part
    ):
        target.errors.append(f"{label} must be a canonical output-relative path.")
        return None
    preflight_path = source_path.parent.joinpath(*parts)
    if _reject_symlinked_validation_path(preflight_path, target, label, "file"):
        return None
    if not preflight_path.exists() or not preflight_path.is_file():
        target.errors.append(f"{label} does not resolve to an existing file.")
        return None
    return preflight_path

def _validate_launch_check_gate(gate: Any, target: ValidationTarget, label: str) -> bool:
    if not isinstance(gate, dict):
        target.errors.append(f"{label} must be an object.")
        return False
    _validate_allowed_keys(gate, _TRAINER_LAUNCH_CHECK_GATE_KEYS, target, label)
    for field_name in ("id", "path", "schema_version"):
        if not isinstance(gate.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if not isinstance(gate.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    return gate.get("passed") is True

def _validate_launch_validation_record(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("trainer_launch_check.validation must be an object.")
        return
    _validate_allowed_keys(value, _TRAINER_LAUNCH_CHECK_VALIDATION_KEYS, target, "trainer_launch_check.validation")
    for field_name in ("passed", "strict"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"trainer_launch_check.validation.{field_name} must be a boolean.")
    for field_name in ("target_count", "error_count", "warning_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"trainer_launch_check.validation.{field_name} must be a non-negative integer.")
    for field_name in ("errors", "warnings"):
        if not _is_string_list(value.get(field_name)):
            target.errors.append(f"trainer_launch_check.validation.{field_name} must be a list of strings.")

def _validate_launch_artifacts_summary(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("trainer_launch_check.artifacts must be an object.")
        return
    _validate_allowed_keys(value, _TRAINER_LAUNCH_CHECK_ARTIFACTS_KEYS, target, "trainer_launch_check.artifacts")
    if not _is_non_negative_int(value.get("count")):
        target.errors.append("trainer_launch_check.artifacts.count must be a non-negative integer.")
    names = value.get("names")
    if not _is_string_list(names):
        target.errors.append("trainer_launch_check.artifacts.names must be a list of strings.")
    elif _is_non_negative_int(value.get("count")) and value.get("count") != len(names):
        target.errors.append(f"trainer_launch_check.artifacts.count expected {len(names)}, got {value.get('count')!r}.")

def _validate_approved_command(command: Any, target: ValidationTarget, expected_approved: bool) -> None:
    if not isinstance(command, dict):
        target.errors.append("trainer_launch_check.approved_command must be an object.")
        return
    label = "trainer_launch_check.approved_command"
    _validate_allowed_keys(command, _TRAINER_LAUNCH_CHECK_APPROVED_COMMAND_KEYS, target, label)
    for field_name in ("approved", "provided", "parseable"):
        if not isinstance(command.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if isinstance(command.get("approved"), bool) and command.get("approved") != expected_approved:
        target.errors.append(f"{label}.approved must match launch check passed.")
    if not isinstance(command.get("raw"), str):
        target.errors.append(f"{label}.raw must be a string.")
    if not isinstance(command.get("shell"), str):
        target.errors.append(f"{label}.shell must be a string.")
    argv = command.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in command.get("argv", [])):
        target.errors.append(f"{label}.argv must be a list of strings.")
        argv = []
    for index, item in enumerate(item for item in argv if isinstance(item, str)):
        _validate_launch_check_command_token_public_path(target, f"{label}.argv[{index}]", item)
    for field_name in ("raw", "shell"):
        value = command.get(field_name)
        if isinstance(value, str) and value:
            _validate_launch_check_command_tokens_public_paths(value, target, f"{label}.{field_name}")
    if expected_approved and not command.get("shell"):
        target.errors.append(f"{label}.shell must be non-empty when approved.")

def _validate_launch_check_command_tokens_public_paths(command: str, target: ValidationTarget, label: str) -> None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for index, token in enumerate(tokens):
        _validate_launch_check_command_token_public_path(target, f"{label}[{index}]", token)

def _validate_launch_check_command_token_public_path(target: ValidationTarget, label: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        return
    if _looks_absolute(value):
        target.errors.append(f"{label} must use a relative command token or redacted placeholder.")
        return
    _, separator, token_value = value.partition("=")
    if separator and _looks_absolute(token_value):
        target.errors.append(f"{label} must use a relative command token or redacted placeholder.")

def _validate_handoff_checks(checks: list[Any], target: ValidationTarget, label: str) -> int:
    failed = 0
    for index, check in enumerate(checks):
        check_label = f"{label}[{index}]"
        if not isinstance(check, dict):
            target.errors.append(f"{check_label} must be an object.")
            failed += 1
            continue
        if not isinstance(check.get("id"), str) or not check.get("id"):
            target.errors.append(f"{check_label}.id must be a non-empty string.")
        if not isinstance(check.get("passed"), bool):
            target.errors.append(f"{check_label}.passed must be a boolean.")
        elif check.get("passed") is False:
            failed += 1
        for field_name in ("actual", "expected", "scope"):
            if not isinstance(check.get(field_name), dict):
                target.errors.append(f"{check_label}.{field_name} must be an object.")
        if not isinstance(check.get("summary"), str):
            target.errors.append(f"{check_label}.summary must be a string.")
    return failed

def _validate_trainer_preflight_gate(gate: Any, target: ValidationTarget, label: str, source_path: Path) -> bool:
    if not isinstance(gate, dict):
        target.errors.append(f"{label} must be an object.")
        return False
    _validate_allowed_keys(gate, _TRAINER_PREFLIGHT_GATE_KEYS, target, label)
    for field_name in ("id", "path", "schema_version"):
        if not isinstance(gate.get(field_name), str) or not gate.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if not isinstance(gate.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    if not isinstance(gate.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    _validate_preflight_file_hash(gate, target, label, source_path, require_kind=False)
    validation = gate.get("validation")
    if validation is not None:
        if not isinstance(validation, dict):
            target.errors.append(f"{label}.validation must be an object when present.")
        else:
            _validate_allowed_keys(validation, _TRAINER_PREFLIGHT_GATE_VALIDATION_KEYS, target, f"{label}.validation")
            for field_name in ("available", "passed", "strict"):
                if not isinstance(validation.get(field_name), bool):
                    target.errors.append(f"{label}.validation.{field_name} must be a boolean.")
            for field_name in ("error_count", "warning_count"):
                if not _is_non_negative_int(validation.get(field_name)):
                    target.errors.append(f"{label}.validation.{field_name} must be a non-negative integer.")
    role = TRAINER_PREFLIGHT_SEMANTIC_GATE_ROLES.get(gate.get("schema_version"))
    semantic_ready = role is not None or gate.get("passed") is not True
    gate_path = _resolve_gate_source_path(gate.get("path"), source_path)
    if role is not None:
        if gate.get("id") != role:
            target.errors.append(
                f"{label}.id must be {role!r} for schema_version {gate.get('schema_version')!r}."
            )
        semantic_ready = bool(
            gate_path is not None
            and trainer_preflight_gate_semantics_ready(
                gate_path,
                role,
                expected_sha256=gate.get("sha256"),
                expected_size_bytes=gate.get("size_bytes"),
            )
        )
        if not semantic_ready:
            target.errors.append(f"{label}.path must reference a semantically ready {role} artifact.")
    elif gate.get("passed") is True:
        target.errors.append(
            f"{label}.schema_version is not an allowed semantic trainer gate."
        )
    return gate.get("passed") is True and semantic_ready

def _validate_trainer_preflight_reviewed_binding(
    gates: list[Any],
    artifacts: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    review_roles = {
        "hfr.reviewed_gate.v1": "reviewed_gate",
        "hfr.review_calibration.v1": "review_calibration",
    }
    gate_records = [
        (index, gate, review_roles[gate["schema_version"]])
        for index, gate in enumerate(gates)
        if isinstance(gate, dict)
        and gate.get("schema_version") in review_roles
    ]
    export_record = artifacts.get("reviewed_export")
    if not gate_records or export_record is None:
        return
    export_path = _resolve_gate_source_path(
        export_record.get("path") if isinstance(export_record, dict) else None,
        source_path,
    )
    if export_path is None:
        target.errors.append(
            "trainer_preflight review gate and reviewed export paths must be resolvable."
        )
        return

    seen_gate_paths: dict[str, set[Path]] = {
        role: set() for role in review_roles.values()
    }
    for index, gate_record, role in gate_records:
        gate_path = _resolve_gate_source_path(gate_record.get("path"), source_path)
        if gate_path is None:
            target.errors.append(
                f"trainer_preflight.gates[{index}] {role} path must be resolvable."
            )
            continue
        canonical_gate_path = gate_path.resolve(strict=False)
        if canonical_gate_path in seen_gate_paths[role]:
            target.errors.append(
                f"trainer_preflight must not contain duplicate {role} paths."
            )
            continue
        seen_gate_paths[role].add(canonical_gate_path)
        try:
            gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
            gate_source = _reviewed_export_source_record(
                gate_payload if isinstance(gate_payload, dict) else {}
            )
            expected_source = build_reviewed_export_source_artifact(
                export_path,
                display_path=str(gate_source.get("path") or ""),
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ReviewedGateError,
            ValueError,
        ):
            target.errors.append(
                f"trainer_preflight.gates[{index}] could not verify {role} to reviewed export binding."
            )
            continue
        if gate_source != expected_source:
            target.errors.append(
                f"trainer_preflight {role} must bind the selected reviewed_export exactly."
            )

def _validate_trainer_preflight_validation_summaries(value: Any, target: ValidationTarget, source_path: Path) -> None:
    if not isinstance(value, list):
        target.errors.append("trainer_preflight.validation_summaries must be a list when present.")
        return
    for index, summary in enumerate(value):
        label = f"trainer_preflight.validation_summaries[{index}]"
        if not isinstance(summary, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(summary, _TRAINER_PREFLIGHT_VALIDATION_SUMMARY_KEYS, target, label)
        for field_name in ("path", "schema_version"):
            if not isinstance(summary.get(field_name), str) or not summary.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if summary.get("kind") != "file":
            target.errors.append(f"{label}.kind must be file.")
        for field_name in ("exists", "regular_file", "symlink", "passed", "strict"):
            if not isinstance(summary.get(field_name), bool):
                target.errors.append(f"{label}.{field_name} must be a boolean.")
        for field_name in ("target_count", "error_count", "warning_count"):
            if not _is_non_negative_int(summary.get(field_name)):
                target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
        _validate_preflight_file_hash(summary, target, label, source_path)
        targets = summary.get("targets")
        if not isinstance(targets, list):
            target.errors.append(f"{label}.targets must be a list.")
            targets = []
        if _is_non_negative_int(summary.get("target_count")) and summary.get("target_count") != len(targets):
            target.errors.append(f"{label}.target_count expected {len(targets)}, got {summary.get('target_count')!r}.")
        for target_index, summary_target in enumerate(targets):
            target_label = f"{label}.targets[{target_index}]"
            if not isinstance(summary_target, dict):
                target.errors.append(f"{target_label} must be an object.")
                continue
            _validate_allowed_keys(summary_target, _TRAINER_PREFLIGHT_VALIDATION_TARGET_KEYS, target, target_label)
            for field_name in ("type", "path"):
                if not isinstance(summary_target.get(field_name), str) or not summary_target.get(field_name):
                    target.errors.append(f"{target_label}.{field_name} must be a non-empty string.")
            if not isinstance(summary_target.get("passed"), bool):
                target.errors.append(f"{target_label}.passed must be a boolean.")
            for field_name in ("error_count", "warning_count"):
                if not _is_non_negative_int(summary_target.get(field_name)):
                    target.errors.append(f"{target_label}.{field_name} must be a non-negative integer.")

def _validate_trainer_preflight_schema_contract_record(name: Any, record: Any, target: ValidationTarget, source_path: Path) -> None:
    if not isinstance(name, str) or not name:
        target.errors.append("trainer_preflight.schema_contracts keys must be non-empty strings.")
    label = f"trainer_preflight.schema_contracts.{name}"
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(record, _TRAINER_PREFLIGHT_SCHEMA_CONTRACT_KEYS, target, label)
    for field_name in ("path", "schema_name"):
        if not isinstance(record.get(field_name), str) or not record.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if record.get("kind") not in {"json", "jsonl"}:
        target.errors.append(f"{label}.kind must be json or jsonl.")
    for field_name in ("exists", "regular_file", "symlink", "passed"):
        if not isinstance(record.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if not _is_non_negative_int(record.get("error_count")):
        target.errors.append(f"{label}.error_count must be a non-negative integer.")
    if not _is_string_list(record.get("errors")):
        target.errors.append(f"{label}.errors must be a list of strings.")
    elif _is_non_negative_int(record.get("error_count")) and record.get("error_count") < len(record.get("errors", [])):
        target.errors.append(f"{label}.error_count must be at least the number of retained errors.")
    if record.get("kind") == "jsonl":
        if not _is_non_negative_int(record.get("row_count")):
            target.errors.append(f"{label}.row_count must be a non-negative integer for JSONL contracts.")
        row_counts = record.get("row_schema_counts")
        if not isinstance(row_counts, list):
            target.errors.append(f"{label}.row_schema_counts must be a list for JSONL contracts.")
        else:
            for index, row_count in enumerate(row_counts):
                row_label = f"{label}.row_schema_counts[{index}]"
                if not isinstance(row_count, dict):
                    target.errors.append(f"{row_label} must be an object.")
                    continue
                _validate_allowed_keys(row_count, _TRAINER_PREFLIGHT_ROW_SCHEMA_COUNT_KEYS, target, row_label)
                if not isinstance(row_count.get("name"), str) or not row_count.get("name"):
                    target.errors.append(f"{row_label}.name must be a non-empty string.")
                if not _is_non_negative_int(row_count.get("count")):
                    target.errors.append(f"{row_label}.count must be a non-negative integer.")
    _validate_preflight_file_hash(record, target, label, source_path, require_kind=False)

def _validate_trainer_preflight_artifact_record(name: Any, record: Any, target: ValidationTarget, source_path: Path) -> None:
    if not isinstance(name, str) or not name:
        target.errors.append("trainer_preflight.artifacts keys must be non-empty strings.")
    label = f"trainer_preflight.artifacts.{name}"
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(record, _TRAINER_PREFLIGHT_ARTIFACT_KEYS, target, label)
    if not isinstance(record.get("path"), str) or not record.get("path"):
        target.errors.append(f"{label}.path must be a non-empty string.")
    if not isinstance(record.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    if record.get("kind") not in {"file", "directory"}:
        target.errors.append(f"{label}.kind must be file or directory.")
    semantic_role = {
        "evidence_bundle": "evidence_bundle",
        "agentic_training_plan": "agentic_training_plan",
    }.get(name)
    if record.get("kind") == "file":
        _validate_preflight_file_hash(record, target, label, source_path)
    if record.get("kind") == "directory":
        if "regular_directory" in record and not isinstance(record.get("regular_directory"), bool):
            target.errors.append(f"{label}.regular_directory must be a boolean when present.")
        if "symlink" in record and not isinstance(record.get("symlink"), bool):
            target.errors.append(f"{label}.symlink must be a boolean when present.")
        if record.get("regular_directory") is True and not _is_non_negative_int(record.get("entry_count")):
            target.errors.append(f"{label}.entry_count must be a non-negative integer for existing directories.")
        _validate_preflight_directory_hash(record, target, label, source_path)
    artifact_path = _resolve_gate_source_path(record.get("path"), source_path)
    if semantic_role is not None and not (
        record.get("kind") == "file"
        and record.get("exists") is True
        and record.get("regular_file") is True
        and artifact_path is not None
        and trainer_preflight_gate_semantics_ready(
            artifact_path,
            semantic_role,
            expected_sha256=record.get("sha256"),
            expected_size_bytes=record.get("size_bytes"),
        )
    ):
        target.errors.append(
            f"{label}.path must reference a semantically ready {semantic_role} artifact."
        )

def _validate_trainer_preflight_control_artifact_declarations(
    checks: list[Any],
    artifacts: dict[str, Any],
    schema_contracts: dict[str, Any],
    target: ValidationTarget,
) -> None:
    declarations = {
        "evidence_bundle": ("evidence_bundle_exists", "evidence_bundle_passed"),
        "agentic_training_plan": (
            "agentic_training_plan_exists",
            "agentic_training_plan_ready",
        ),
    }
    for role, required_check_ids in declarations.items():
        check_counts = {
            check_id: sum(
                1
                for check in checks
                if isinstance(check, dict) and check.get("id") == check_id
            )
            for check_id in required_check_ids
        }
        declared = bool(
            role in artifacts
            or role in schema_contracts
            or any(check_counts.values())
        )
        if not declared:
            continue
        if role not in artifacts:
            target.errors.append(
                f"trainer_preflight.artifacts must contain {role!r} when its control-artifact checks are present."
            )
        if role not in schema_contracts:
            target.errors.append(
                f"trainer_preflight.schema_contracts must contain {role!r} when its control artifact is present."
            )
        for check_id, count in check_counts.items():
            if count != 1:
                target.errors.append(
                    f"trainer_preflight.checks must contain exactly one {check_id!r} check when {role!r} is declared."
                )

def _validate_trainer_preflight_dataset_selection(
    value: Any,
    target: ValidationTarget,
    *,
    label: str = "trainer_preflight.dataset_selection",
) -> None:
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list.")
        return
    for index, record in enumerate(value):
        row_label = f"{label}[{index}]"
        if not isinstance(record, dict):
            target.errors.append(f"{row_label} must be an object.")
            continue
        _validate_allowed_keys(record, _TRAINER_PREFLIGHT_DATASET_SELECTION_KEYS, target, row_label)
        if record.get("artifact") not in {"training_export", "reviewed_export"}:
            target.errors.append(f"{row_label}.artifact must be training_export or reviewed_export.")
        if not _is_dataset_version(record.get("dataset_version")):
            target.errors.append(f"{row_label}.dataset_version must be an hfrds-* selection key.")
        for field_name in ("manifest_path", "registry_path"):
            if not isinstance(record.get(field_name), str) or not record.get(field_name):
                target.errors.append(f"{row_label}.{field_name} must be a non-empty string.")
        for field_name in ("manifest_sha256", "registry_sha256", "registry_manifest_sha256"):
            if record.get(field_name) and not _is_sha256(record.get(field_name)):
                target.errors.append(f"{row_label}.{field_name} must be a SHA-256 hex string when present.")
        if record.get("manifest_sha256") and record.get("registry_manifest_sha256") and record.get("manifest_sha256") != record.get("registry_manifest_sha256"):
            target.errors.append(f"{row_label}.registry_manifest_sha256 must match manifest_sha256.")
        for field_name in ("registry_dataset_version", "registry_selection_key"):
            if record.get(field_name) != record.get("dataset_version"):
                target.errors.append(f"{row_label}.{field_name} must match dataset_version.")
        if not isinstance(record.get("matches_required"), bool):
            target.errors.append(f"{row_label}.matches_required must be a boolean.")
        if record.get("redaction_passed") is not True:
            target.errors.append(f"{row_label}.redaction_passed must be true.")
        if record.get("artifact") == "training_export" and record.get("heldout_scenario_exclusive") is not True:
            target.errors.append(f"{row_label}.heldout_scenario_exclusive must be true for training exports.")
        if not _is_string_list(record.get("required_dataset_versions")):
            target.errors.append(f"{row_label}.required_dataset_versions must be a list of strings.")
        _validate_trainer_preflight_trainer_views(record.get("trainer_views"), target, f"{row_label}.trainer_views")
        if not _is_string_list(record.get("trainer_modes")):
            target.errors.append(f"{row_label}.trainer_modes must be a list of strings.")

def _validate_trainer_preflight_trainer_views(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _TRAINER_PREFLIGHT_TRAINER_VIEWS_KEYS, target, label)
    if "contract_version" in value and not isinstance(value.get("contract_version"), str):
        target.errors.append(f"{label}.contract_version must be a string when present.")
    if "notes" in value and not _is_string_list(value.get("notes")):
        target.errors.append(f"{label}.notes must be a list of strings when present.")
    mode_to_view = value.get("mode_to_view", {})
    if not isinstance(mode_to_view, dict) or not all(isinstance(key, str) and isinstance(item, str) for key, item in mode_to_view.items()):
        target.errors.append(f"{label}.mode_to_view must be an object of string values when present.")
    if "root_views" in value and not _is_string_list(value.get("root_views")):
        target.errors.append(f"{label}.root_views must be a list of strings when present.")
    views = value.get("views", [])
    if "views" in value and not isinstance(views, list):
        target.errors.append(f"{label}.views must be a list when present.")
        return
    for index, view in enumerate(views if isinstance(views, list) else []):
        view_label = f"{label}.views[{index}]"
        if not isinstance(view, dict):
            target.errors.append(f"{view_label} must be an object.")
            continue
        _validate_allowed_keys(view, _TRAINER_PREFLIGHT_TRAINER_VIEW_KEYS, target, view_label)
        for field_name in ("view_id", "artifact", "artifact_path", "artifact_format", "schema_version", "label_policy"):
            if not isinstance(view.get(field_name), str):
                target.errors.append(f"{view_label}.{field_name} must be a string.")
        if not _is_string_list(view.get("training_modes")):
            target.errors.append(f"{view_label}.training_modes must be a list of strings.")
        if not _is_non_negative_int(view.get("row_count")):
            target.errors.append(f"{view_label}.row_count must be a non-negative integer.")
        if not _is_string_list(view.get("source_artifacts")):
            target.errors.append(f"{view_label}.source_artifacts must be a list of strings.")
        if not isinstance(view.get("available"), bool):
            target.errors.append(f"{view_label}.available must be a boolean.")
        split_paths = view.get("split_paths")
        if not isinstance(split_paths, dict) or not all(isinstance(key, str) and isinstance(item, str) for key, item in split_paths.items()):
            target.errors.append(f"{view_label}.split_paths must be an object of string values.")
        if not _is_string_list(view.get("notes")):
            target.errors.append(f"{view_label}.notes must be a list of strings.")

def _validate_preflight_file_hash(
    record: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path,
    *,
    require_kind: bool = True,
) -> None:
    if require_kind and record.get("kind") != "file":
        return
    if record.get("exists") is not True:
        return
    if "regular_file" in record and not isinstance(record.get("regular_file"), bool):
        target.errors.append(f"{label}.regular_file must be a boolean when present.")
    if "symlink" in record and not isinstance(record.get("symlink"), bool):
        target.errors.append(f"{label}.symlink must be a boolean when present.")
    if record.get("regular_file") is False:
        return
    if not _is_non_negative_int(record.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer for existing files.")
    if not _is_sha256(record.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string for existing files.")
        return
    file_path = _resolve_preflight_record_path(record.get("path"), source_path)
    if file_path is None:
        return
    if file_path.is_symlink():
        target.errors.append(f"{label}.path must not resolve to a symlink.")
        return
    if _path_has_symlink_component(file_path, include_leaf=False):
        target.errors.append(f"{label}.path must not resolve through a symlink.")
        return
    if not file_path.exists() or not file_path.is_file():
        target.errors.append(f"{label}.path does not resolve to an existing file.")
        return
    if file_path.stat().st_size != record.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if _sha256(file_path) != record.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current file.")

def _validate_preflight_directory_hash(
    record: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path,
) -> None:
    if record.get("exists") is not True or record.get("regular_directory") is False:
        return
    has_tree_fields = any(field_name in record for field_name in ("file_count", "size_bytes", "sha256", "tree_hash_algorithm"))
    if not has_tree_fields:
        return
    if record.get("tree_hash_algorithm") != TRAINER_DIRECTORY_TREE_HASH_ALGORITHM:
        target.errors.append(f"{label}.tree_hash_algorithm is invalid.")
    if not _is_non_negative_int(record.get("file_count")):
        target.errors.append(f"{label}.file_count must be a non-negative integer for existing directories.")
    if not _is_non_negative_int(record.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer for existing directories.")
    if not _is_sha256(record.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string for existing directories.")
        return
    directory_path = _resolve_preflight_record_path(record.get("path"), source_path)
    if directory_path is None:
        return
    if directory_path.is_symlink():
        target.errors.append(f"{label}.path must not resolve to a symlink.")
        return
    if _path_has_symlink_component(directory_path, include_leaf=False):
        target.errors.append(f"{label}.path must not resolve through a symlink.")
        return
    if not directory_path.exists() or not directory_path.is_dir():
        target.errors.append(f"{label}.path does not resolve to an existing directory.")
        return
    tree = _directory_tree_fingerprint(directory_path)
    if _is_non_negative_int(record.get("file_count")) and tree["file_count"] != record.get("file_count"):
        target.errors.append(f"{label}.file_count does not match the current directory.")
    if _is_non_negative_int(record.get("size_bytes")) and tree["size_bytes"] != record.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current directory.")
    if tree["sha256"] != record.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current directory.")

def _resolve_preflight_record_path(value: Any, source_path: Path) -> Path | None:
    if not isinstance(value, str) or not value or value.startswith("<redacted:") or _is_windows_absolute(value):
        return None
    raw = Path(value)
    if raw.is_absolute():
        return raw
    return source_path.parent / raw

def _validate_trainer_command(command: Any, target: ValidationTarget) -> None:
    if not isinstance(command, dict):
        target.errors.append("trainer_preflight.trainer_command must be an object.")
        return
    _validate_allowed_keys(command, _TRAINER_PREFLIGHT_TRAINER_COMMAND_KEYS, target, "trainer_preflight.trainer_command")
    if not isinstance(command.get("provided"), bool):
        target.errors.append("trainer_preflight.trainer_command.provided must be a boolean.")
    if not isinstance(command.get("raw"), str):
        target.errors.append("trainer_preflight.trainer_command.raw must be a string.")
    else:
        _warn_shell_tokens_public_paths(command.get("raw"), target, "trainer_preflight.trainer_command.raw")
    if not isinstance(command.get("argv"), list) or not all(isinstance(item, str) for item in command.get("argv", [])):
        target.errors.append("trainer_preflight.trainer_command.argv must be a list of strings.")
    else:
        for index, item in enumerate(command.get("argv", [])):
            _warn_command_token_public_path(target, f"trainer_preflight.trainer_command.argv[{index}]", item)
    if "parseable" in command and not isinstance(command.get("parseable"), bool):
        target.errors.append("trainer_preflight.trainer_command.parseable must be a boolean when present.")
