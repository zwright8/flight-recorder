"""Extracted validation implementation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any
from ..path_safety import path_has_symlink_component as _path_has_symlink_component, resolve_artifact_reference_path
from ..hashing import sha256_file as _sha256

@dataclass
class ValidationTarget:
    target_type: str
    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.target_type,
            "path": self.path,
            "passed": not self.errors,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }

def _reject_symlinked_validation_path(path: Path, target: ValidationTarget, label: str, kind: str) -> bool:
    if _path_has_symlink_component(path, include_leaf=True):
        target.errors.append(f"{label} must resolve to a regular non-symlink {kind}.")
        return True
    return False

def _read_json_object_silent(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}

def _validate_allowed_keys(value: dict[str, Any], allowed: set[str], target: ValidationTarget, label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        target.errors.append(f"{label} contains unknown field(s): {unknown!r}.")

def _reviewed_export_source_record(payload: dict[str, Any]) -> dict[str, Any]:
    sources = payload.get("source_artifacts")
    record = sources.get("reviewed_export") if isinstance(sources, dict) else None
    return record if isinstance(record, dict) else {}

_ARCHIVE_RELATIONSHIP_KEYS = {"from", "to", "type"}

def _archive_artifact_roles_by_name(
    artifacts: list[Any],
    target: ValidationTarget,
    prefix: str,
    *,
    require_unique: bool,
) -> dict[str, set[str]]:
    roles_by_name: dict[str, set[str]] = {}
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            continue
        name = artifact.get("name")
        role = artifact.get("role")
        if not isinstance(name, str) or not name or not isinstance(role, str) or not role:
            continue
        if name in roles_by_name:
            if require_unique:
                target.errors.append(f"{prefix}.artifacts[{index}].name must be unique.")
            roles_by_name[name].add(role)
        else:
            roles_by_name[name] = {role}
    return roles_by_name

def _validate_archive_relationship(
    relationship: Any,
    target: ValidationTarget,
    label: str,
    artifact_roles_by_name: dict[str, set[str]],
    allowed_edges_by_type: dict[str, set[tuple[str, str]]],
) -> None:
    if not isinstance(relationship, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(relationship, _ARCHIVE_RELATIONSHIP_KEYS, target, label)
    for field_name in ("from", "to", "type"):
        if not isinstance(relationship.get(field_name), str) or not relationship.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    from_name = relationship.get("from")
    to_name = relationship.get("to")
    relationship_type = relationship.get("type")
    from_roles = artifact_roles_by_name.get(from_name) if isinstance(from_name, str) else None
    to_roles = artifact_roles_by_name.get(to_name) if isinstance(to_name, str) else None
    if isinstance(from_name, str) and from_name and from_roles is None:
        target.errors.append(f"{label}.from must reference an archived artifact name.")
    if isinstance(to_name, str) and to_name and to_roles is None:
        target.errors.append(f"{label}.to must reference an archived artifact name.")
    if not isinstance(relationship_type, str) or not relationship_type:
        return
    allowed_edges = allowed_edges_by_type.get(relationship_type)
    if allowed_edges is None:
        target.errors.append(f"{label}.type is invalid.")
        return
    if from_roles is not None and to_roles is not None:
        has_allowed_edge = any((from_role, to_role) in allowed_edges for from_role in from_roles for to_role in to_roles)
        if not has_allowed_edge:
            from_role_text = "|".join(sorted(from_roles))
            to_role_text = "|".join(sorted(to_roles))
            target.errors.append(f"{label}.type is invalid for {from_role_text} -> {to_role_text}.")

def _archive_artifact_path(value: Any, archive_root: Path) -> Path | None:
    if not isinstance(value, str) or not value or value.startswith("<"):
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    return archive_root / path

def _reject_archive_artifact_symlink_path(path: Path, target: ValidationTarget, label: str) -> bool:
    if path.is_symlink():
        target.errors.append(f"{label}.path must not be a symlink.")
        return True
    if _path_has_symlink_component(path, include_leaf=False):
        target.errors.append(f"{label}.path must not resolve through a symlink.")
        return True
    return False

def _path_resolves_inside(path: Path, root: Path) -> bool:
    try:
        root_resolved = root.resolve()
        path_resolved = path.resolve(strict=False)
    except OSError:
        return False
    return path_resolved == root_resolved or path_resolved.is_relative_to(root_resolved)

def _increment(counts: dict[str, int], value: Any) -> None:
    if isinstance(value, str) and value:
        counts[value] = counts.get(value, 0) + 1

def _validate_gate_like_checks(checks: list[Any], target: ValidationTarget, label: str) -> int:
    failed_checks = 0
    for index, check in enumerate(checks):
        item_label = f"{label}[{index}]"
        if not isinstance(check, dict):
            target.errors.append(f"{item_label} must be an object.")
            continue
        if not isinstance(check.get("id"), str) or not check.get("id"):
            target.errors.append(f"{item_label}.id must be a non-empty string.")
        if not isinstance(check.get("passed"), bool):
            target.errors.append(f"{item_label}.passed must be a boolean.")
        elif not check["passed"]:
            failed_checks += 1
        if "actual" not in check:
            target.errors.append(f"{item_label}.actual is missing.")
        if not isinstance(check.get("expected"), dict):
            target.errors.append(f"{item_label}.expected must be an object.")
        if not isinstance(check.get("summary"), str) or not check.get("summary"):
            target.errors.append(f"{item_label}.summary must be a non-empty string.")
    return failed_checks

def _merge_count_rows(target_counts: dict[str, int], value: Any, target: ValidationTarget, label: str) -> None:
    counts = _count_rows(value)
    if counts is None:
        target.errors.append(f"{label} must be count rows.")
        return
    for key, count in counts.items():
        target_counts[key] = target_counts.get(key, 0) + count

def _rate_value(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(numerator / denominator, 4)

def _validate_evidence_refs(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list when present.")
        return
    for index, ref in enumerate(value):
        if not isinstance(ref, dict):
            target.errors.append(f"{label}[{index}] must be an object.")
            continue
        ref_target = ref.get("target")
        if ref_target not in {"event", "final_answer", "episode", "state_snapshot"}:
            target.errors.append(f"{label}[{index}].target must be one of event, final_answer, episode, or state_snapshot.")
        if ref_target == "event":
            event_index = ref.get("event_index")
            if not isinstance(event_index, int) or isinstance(event_index, bool) or event_index < 0:
                target.errors.append(f"{label}[{index}].event_index must be a non-negative integer for event refs.")
        if "reason" in ref and not isinstance(ref.get("reason"), str):
            target.errors.append(f"{label}[{index}].reason must be a string when present.")
        if "passed" in ref and not isinstance(ref.get("passed"), bool):
            target.errors.append(f"{label}[{index}].passed must be a boolean when present.")

def _read_object(path: Path, target: ValidationTarget, label: str) -> dict[str, Any] | None:
    if _path_has_symlink_component(path, include_leaf=True):
        target.errors.append(f"{label} must resolve to a regular non-symlink file.")
        return None
    if not path.exists():
        target.errors.append(f"{label} is missing.")
        return None
    if not path.is_file():
        target.errors.append(f"{label} must be a file.")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        target.errors.append(f"{label} contains invalid JSON: {exc}")
        return None
    if not isinstance(value, dict):
        target.errors.append(f"{label} must contain a JSON object.")
        return None
    return value

def _read_object_optional(path: Path, target: ValidationTarget, label: str, refresh_message: str) -> dict[str, Any] | None:
    if _path_has_symlink_component(path, include_leaf=True):
        target.errors.append(f"{label} must resolve to a regular non-symlink file.")
        return None
    if not path.exists():
        target.warnings.append(f"{label} is missing; {refresh_message}.")
        return None
    return _read_object(path, target, label)

def _read_jsonl_objects(path: Path, target: ValidationTarget, label: str) -> list[dict[str, Any]]:
    if _path_has_symlink_component(path, include_leaf=True):
        target.errors.append(f"{label} must resolve to a regular non-symlink file.")
        return []
    if not path.exists():
        target.errors.append(f"{label} is missing.")
        return []
    if not path.is_file():
        target.errors.append(f"{label} must be a file.")
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            target.errors.append(f"{label}:{line_number} contains invalid JSON: {exc}")
            continue
        if not isinstance(value, dict):
            target.errors.append(f"{label}:{line_number} must contain a JSON object.")
            continue
        rows.append(value)
    return rows

def _read_jsonl_objects_optional(
    path: Path,
    target: ValidationTarget,
    label: str,
    refresh_message: str = "rerun export-rl to emit step-level reward attribution",
) -> list[dict[str, Any]]:
    if _path_has_symlink_component(path, include_leaf=True):
        target.errors.append(f"{label} must resolve to a regular non-symlink file.")
        return []
    if not path.exists():
        target.warnings.append(f"{label} is missing; {refresh_message}.")
        return []
    return _read_jsonl_objects(path, target, label)

def _require_equal(
    obj: dict[str, Any],
    field_name: str,
    expected: Any,
    target: ValidationTarget,
    *,
    prefix: str = "",
) -> None:
    if obj.get(field_name) != expected:
        target.errors.append(f"{prefix}{field_name} expected {expected!r}, got {obj.get(field_name)!r}.")

def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)

def _is_non_empty_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item for item in value) and bool(value)

def _validate_metadata(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object when present.")
        return
    for key, raw_value in value.items():
        if not isinstance(key, str) or not key:
            target.errors.append(f"{label} keys must be non-empty strings.")
        elif any(char.isspace() for char in key):
            target.errors.append(f"{label}.{key!r} key must not contain whitespace.")
        if not isinstance(raw_value, str):
            target.errors.append(f"{label}.{key} must be a string.")

def _is_int_between(value: Any, minimum: int, maximum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum

def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)

def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0

def _is_optional_non_negative_int(value: Any) -> bool:
    return value is None or _is_non_negative_int(value)

def _is_number_between(value: Any, minimum: float, maximum: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and minimum <= float(value) <= maximum

def _is_optional_non_negative_number(value: Any) -> bool:
    return value is None or _is_number_between(value, 0, float("inf"))

def _is_optional_number(value: Any) -> bool:
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))

def _is_optional_rate(value: Any) -> bool:
    return value is None or _is_number_between(value, 0, 1)

def _validate_metric_source(value: dict[str, Any], label: str, target: ValidationTarget) -> None:
    if value.get("source") not in {"run_rows", "suite_metrics", "missing"}:
        target.errors.append(f"{label}.source must be run_rows, suite_metrics, or missing.")

def _validate_metric_count_fields(value: dict[str, Any], label: str, target: ValidationTarget) -> None:
    for field_name in ("known_run_count", "missing_run_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")

def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())

def _is_lowercase_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and value == value.lower() and all(
        char in "0123456789abcdef" for char in value
    )

def _is_dataset_version(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("hfrds-")
        and len(value) > len("hfrds-")
        and all(char in "0123456789abcdef" for char in value.removeprefix("hfrds-").lower())
    )

def _looks_absolute(value: str) -> bool:
    return value.startswith("/") or _is_windows_absolute(value)

def _warn_absolute_public_path(target: ValidationTarget, label: str, value: Any) -> None:
    if isinstance(value, str) and value and _looks_absolute(value):
        target.warnings.append(f"{label} is absolute; use a relative path or redacted placeholder for public bundles.")

def _validate_public_review_ref_path(target: ValidationTarget, label: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        return
    if _is_public_review_ref_path(value):
        return
    target.errors.append(f"{label} must be a safe relative path or redacted placeholder.")

def _validate_public_review_fingerprint_paths(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        return
    for name, record in value.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            continue
        _validate_public_review_ref_path(target, f"{label}.{name}.path", record.get("path"))

def _is_public_review_ref_path(value: str) -> bool:
    if _is_redacted_placeholder(value):
        return True
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return (
        not path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and "\\" not in value
        and ".." not in path.parts
        and all(not part.startswith("~") for part in path.parts)
    )

def _is_redacted_placeholder(value: str) -> bool:
    if not (value.startswith("<redacted:") and value.endswith(">")):
        return False
    name = value.removeprefix("<redacted:").removesuffix(">")
    return bool(name) and "/" not in name and "\\" not in name and name not in {".", ".."}

def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file() and not candidate.is_symlink()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()

def _directory_tree_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file() and not candidate.is_symlink()):
        relative = item.relative_to(path)
        size = item.stat().st_size
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha256(item).encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        size_bytes += size
    return {"sha256": digest.hexdigest(), "file_count": file_count, "size_bytes": size_bytes}

def _directory_contains_symlink(path: Path) -> bool:
    return any(item.is_symlink() for item in path.rglob("*"))

def _score_value(value: Any) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0

def _number_value(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0

def _number_delta(before: Any, after: Any) -> float:
    return round(_number_value(after) - _number_value(before), 4)

def _average_number(values: list[int] | list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0

def _count_family(rows: list[dict[str, Any]], family: str) -> int:
    return sum(1 for row in rows if isinstance(row, dict) and str(row.get("task_family") or "unknown") == family)

def _outcome_strings(episode: dict[str, Any], field_name: str) -> list[str]:
    outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
    values = outcome.get(field_name)
    return values if _is_string_list(values) else []

def _count_strings(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts

def _validate_count_map_object(value: Any, target: ValidationTarget, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return {}
    counts: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key:
            target.errors.append(f"{label} keys must be non-empty strings.")
            continue
        if not _is_non_negative_int(count):
            target.errors.append(f"{label}.{key} must be a non-negative integer.")
            continue
        counts[key] = count
    return counts

def _validate_count_rows(value: Any, target: ValidationTarget, label: str) -> dict[str, int]:
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list.")
        return {}
    counts: dict[str, int] = {}
    previous_id = ""
    for index, row in enumerate(value):
        row_label = f"{label}[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{row_label} must be an object.")
            continue
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            target.errors.append(f"{row_label}.id must be a non-empty string.")
            continue
        if row_id in counts:
            target.errors.append(f"{row_label}.id duplicates {row_id!r}.")
        if previous_id and row_id < previous_id:
            target.errors.append(f"{label} must be sorted by id.")
        previous_id = row_id
        count = row.get("count")
        if not _is_non_negative_int(count):
            target.errors.append(f"{row_label}.count must be a non-negative integer.")
            continue
        counts[row_id] = count
    return counts

def _expected_suite_trend_delta(previous_point: dict[str, Any], point: dict[str, Any]) -> dict[str, Any]:
    return {
        "pass_rate_delta": _number_delta(previous_point.get("pass_rate"), point.get("pass_rate")),
        "average_score_delta": _number_delta(previous_point.get("average_score"), point.get("average_score")),
        "failed_rule_count_delta": _non_negative_int_value(point.get("failed_rule_count"))
        - _non_negative_int_value(previous_point.get("failed_rule_count")),
        "critical_failure_count_delta": _non_negative_int_value(point.get("critical_failure_count"))
        - _non_negative_int_value(previous_point.get("critical_failure_count")),
    }

def _expected_suite_trend_summary(points: list[dict[str, Any]]) -> str:
    if not points:
        return "TREND: no suite summaries."
    if len(points) == 1:
        point = points[0]
        return (
            f"TREND: one point; pass rate {point.get('pass_rate')}; "
            f"average score {point.get('average_score')}."
        )
    first = points[0]
    last = points[-1]
    pass_delta = _number_delta(first.get("pass_rate"), last.get("pass_rate"))
    score_delta = _number_delta(first.get("average_score"), last.get("average_score"))
    failed_delta = _non_negative_int_value(last.get("failed_rule_count")) - _non_negative_int_value(first.get("failed_rule_count"))
    critical_delta = _non_negative_int_value(last.get("critical_failure_count")) - _non_negative_int_value(
        first.get("critical_failure_count")
    )
    return (
        f"TREND: {len(points)} points; pass_rate_delta={pass_delta}; "
        f"average_score_delta={score_delta}; failed_rule_delta={failed_delta}; "
        f"critical_failure_delta={critical_delta}."
    )

def _non_negative_int_value(value: Any) -> int:
    return value if _is_non_negative_int(value) else 0

def _count_rows(value: Any) -> dict[str, int] | None:
    if not isinstance(value, list):
        return None
    counts: dict[str, int] = {}
    for row in value:
        if not isinstance(row, dict):
            return None
        row_id = row.get("id")
        count = row.get("count")
        if not isinstance(row_id, str) or not isinstance(count, int) or isinstance(count, bool):
            return None
        counts[row_id] = count
    return counts

def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")
