"""CLI commands for the run core domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator
from ..artifacts import ArtifactError, build_suite_trend, compare_scorecards, compare_suites, write_compare_report, write_junit, write_markdown_summary, write_suite_compare_report, write_suite_trend_report
from ..lineage import LINEAGE_SCHEMA_VERSION, REPLAY_BUNDLE_SCHEMA_VERSION, write_run_lineage
from pathlib import Path
from ..digest import RunDigestError, build_run_digest, render_run_digest_markdown
from ..schema_registry import SchemaRegistryError, check_schema_file, check_schema_jsonl_file, list_schema_records, load_schema, write_schema_bundle
import argparse
from ..state_diff import StateDiffError, build_state_diff
import hashlib
import json
from ..schema import ScenarioError, load_scenario, resolve_trace_path
from ..state import StateSnapshotError, load_state_snapshot, resolve_before_state_snapshot_path, resolve_state_snapshot_path, sanitize_state_snapshot
from ..path_safety import assert_output_does_not_alias_sources, assert_output_outside_source_directories, locked_owned_output_directory, output_directory_lock_is_held, path_has_symlink_component, remove_directory_tree_if_identity
from ..adapters import AdapterError, normalize_trace
import os
import re
from ..redaction import sanitize_trace
from ..scorers import score_trace
import shlex
import stat
import sys
import tempfile
from ..trajectory_v2 import trajectory_v2_context_from_path, trajectory_v2_from_trace, write_trajectory_v2
from ..report import write_index, write_report
from .constants import _OPTIONAL_RUN_ARTIFACTS, _OPTIONAL_RUN_ARTIFACT_RECORDS, _REQUIRED_RUN_ARTIFACTS, _RUN_LINEAGE_OUTPUT_NAMES
from .replay_core import _regression_scenario, _sha256_file
from .shared import _display_path, _lineage_record_path, _read_json, _write_json


@dataclass(frozen=True)
class _RunDirectoryAttestation:
    identity: tuple[int, int]
    tree_sha256: str
    file_count: int
    size_bytes: int


def _load_digest_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    run_dir = Path(args.run) if args.run else None
    trace_path = Path(args.trace) if args.trace else (run_dir / "normalized_trace.json" if run_dir is not None else None)
    score_path = Path(args.score) if args.score else (run_dir / "scorecard.json" if run_dir is not None else None)
    state_diff_path = Path(args.state_diff) if args.state_diff else None
    if state_diff_path is None and run_dir is not None and (run_dir / "state_diff.json").exists():
        state_diff_path = run_dir / "state_diff.json"
    if trace_path is None or score_path is None:
        raise RunDigestError("--trace and --score are required when --run is not supplied")
    trace = _read_json(trace_path)
    scorecard = _read_json(score_path)
    state_diff = _read_json(state_diff_path) if state_diff_path is not None else None
    if args.scenario:
        scenario = load_scenario(args.scenario)
    elif run_dir is not None:
        scenario = _scenario_stub_from_run(run_dir, scorecard)
    else:
        raise RunDigestError("--scenario is required when --run is not supplied")
    return scenario, trace, scorecard, state_diff


def _scenario_stub_from_run(run_dir: Path, scorecard: dict[str, Any]) -> dict[str, Any]:
    lineage_path = run_dir / "artifact_lineage.json"
    if lineage_path.exists():
        lineage = _read_json(lineage_path)
        scenario_path = _lineage_record_path(lineage, "inputs", "scenario")
        if scenario_path is not None and not _looks_redacted_path(scenario_path):
            candidate = Path(scenario_path)
            if not candidate.is_absolute():
                candidate = (lineage_path.parent / candidate).resolve()
            if candidate.exists():
                return load_scenario(candidate)
        scenario = lineage.get("scenario") if isinstance(lineage.get("scenario"), dict) else {}
        if scenario.get("id") or scenario.get("title"):
            return {
                "id": str(scenario.get("id") or scorecard.get("scenario_id") or "unknown"),
                "title": str(scenario.get("title") or scorecard.get("scenario_title") or scenario.get("id") or "unknown"),
                "policy": {"secret_patterns": []},
                "assertions": {},
            }
    return {
        "id": str(scorecard.get("scenario_id") or "unknown"),
        "title": str(scorecard.get("scenario_title") or scorecard.get("scenario_id") or "unknown"),
        "policy": {"secret_patterns": []},
        "assertions": {},
    }


def _looks_redacted_path(value: str) -> bool:
    return value.startswith("<redacted:") or value.startswith("<missing-")


def _default_digest_out(args: argparse.Namespace) -> Path | None:
    return Path(args.run) / "run_digest.json" if args.run else None


def _remove_stale_optional_run_artifacts(
    run_dir: Path,
    *,
    expected_identity: tuple[int, int],
    junit_out: str | Path | None,
    markdown_out: str | Path | None,
) -> None:
    if not output_directory_lock_is_held(run_dir):
        raise ArtifactError(
            f"refusing to remove optional artifacts without the run publication lock: {run_dir}"
        )
    _assert_run_directory_identity(run_dir, expected_identity)
    expected_fingerprints = _stale_lineage_optional_fingerprints(run_dir)
    expected_fingerprints.update(
        _stale_lineage_report_sidecars(
            run_dir,
            requested={"junit": junit_out, "markdown": markdown_out},
        )
    )
    stale_names = [*_OPTIONAL_RUN_ARTIFACTS, *expected_fingerprints]
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        run_fd = os.open(run_dir, flags)
    except OSError as exc:
        raise ArtifactError(f"could not open run output for stale-artifact removal: {exc}") from exc
    try:
        opened = os.fstat(run_fd)
        if (opened.st_dev, opened.st_ino) != expected_identity:
            raise ArtifactError(f"run output directory identity changed during publication: {run_dir}")
        for name in dict.fromkeys(stale_names):
            try:
                expected_file = _regular_file_identity_at(run_fd, name)
            except FileNotFoundError:
                continue
            expected_fingerprint = expected_fingerprints.get(name)
            if expected_fingerprint is None or (
                expected_file[2],
                expected_file[4],
            ) != expected_fingerprint:
                raise ArtifactError(
                    f"optional run artifact does not match prior lineage fingerprint: {run_dir / name}"
                )
            _quarantine_and_unlink_optional_run_artifact(
                run_fd,
                run_dir=run_dir,
                name=name,
                expected_file=expected_file,
                expected_directory=expected_identity,
            )
    finally:
        os.close(run_fd)


def _stale_lineage_optional_fingerprints(run_dir: Path) -> dict[str, tuple[int, str]]:
    lineage = _read_existing_run_lineage(run_dir)
    outputs = lineage.get("outputs") if isinstance(lineage, dict) else None
    if not isinstance(outputs, list):
        return {}
    records = {
        record.get("name"): record
        for record in outputs
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }
    fingerprints: dict[str, tuple[int, str]] = {}
    for name, filename in _OPTIONAL_RUN_ARTIFACT_RECORDS.items():
        record = records.get(name)
        if isinstance(record, dict) and _lineage_output_fingerprint_matches(
            run_dir,
            record,
            filename,
        ):
            fingerprints[filename] = (record["size_bytes"], record["sha256"])
    return fingerprints


def _stale_lineage_report_sidecars(
    run_dir: Path,
    *,
    requested: dict[str, str | Path | None],
) -> dict[str, tuple[int, str]]:
    lineage = _read_existing_run_lineage(run_dir)
    outputs = lineage.get("outputs") if isinstance(lineage, dict) else None
    if not isinstance(outputs, list):
        return {}
    records = {
        record.get("name"): record
        for record in outputs
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }
    stale: dict[str, tuple[int, str]] = {}
    for name, requested_path in requested.items():
        prior_path = _verified_lineage_sidecar_in_run(run_dir, records.get(name))
        if prior_path is None:
            continue
        if requested_path is not None and Path(requested_path).resolve(strict=False) == prior_path:
            continue
        record = records[name]
        stale[prior_path.name] = (record["size_bytes"], record["sha256"])
    return stale


def _read_existing_run_lineage(run_dir: Path) -> dict[str, Any] | None:
    lineage_path = run_dir / "artifact_lineage.json"
    if not lineage_path.is_file() or path_has_symlink_component(lineage_path, include_leaf=True):
        return None
    try:
        value = json.loads(lineage_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _verified_lineage_sidecar_in_run(
    run_dir: Path,
    record: Any,
) -> Path | None:
    if not isinstance(record, dict) or record.get("role") != "output" or record.get("exists") is not True:
        return None
    path_label = record.get("path")
    if not isinstance(path_label, str) or path_label.startswith("<redacted:"):
        return None
    candidate = Path(path_label)
    lexical = candidate if candidate.is_absolute() else Path.cwd() / candidate
    if path_has_symlink_component(lexical, include_leaf=True):
        return None
    resolved = lexical.resolve(strict=False)
    resolved_run = run_dir.resolve(strict=False)
    if resolved.parent != resolved_run or resolved.name in {"", ".", ".."}:
        return None
    if (
        not resolved.is_file()
        or not _is_non_negative_int(record.get("size_bytes"))
        or not _is_sha256(record.get("sha256"))
    ):
        return None
    try:
        if resolved.stat().st_size != record["size_bytes"] or _sha256_file(resolved) != record["sha256"]:
            return None
    except OSError:
        return None
    return resolved


def _regular_file_identity_at(
    directory_fd: int,
    relative_path: str,
) -> tuple[int, int, int, int, str]:
    before = os.stat(relative_path, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ArtifactError(f"optional run artifact is not a regular file: {relative_path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_fd = os.open(relative_path, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ArtifactError(f"could not safely open optional run artifact {relative_path}: {exc}") from exc
    try:
        opened = os.fstat(file_fd)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
            raise ArtifactError(f"optional run artifact changed while being opened: {relative_path}")
        digest = hashlib.sha256()
        while chunk := os.read(file_fd, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(file_fd)
    after = os.stat(relative_path, dir_fd=directory_fd, follow_symlinks=False)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ArtifactError(f"optional run artifact changed while being inspected: {relative_path}")
    return (*identity, digest.hexdigest())


def _quarantine_and_unlink_optional_run_artifact(
    directory_fd: int,
    *,
    run_dir: Path,
    name: str,
    expected_file: tuple[int, int, int, int, str],
    expected_directory: tuple[int, int],
) -> None:
    quarantine_name = f".hfr-stale-{os.getpid()}-{os.urandom(12).hex()}"
    try:
        os.mkdir(quarantine_name, mode=0o700, dir_fd=directory_fd)
    except OSError as exc:
        raise ArtifactError(f"could not create stale-artifact quarantine in {run_dir}: {exc}") from exc
    quarantined = f"{quarantine_name}/{name}"
    moved = False
    try:
        os.rename(name, quarantined, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        moved = True
        if _regular_file_identity_at(directory_fd, quarantined) != expected_file:
            raise ArtifactError(f"optional run artifact changed during quarantine: {run_dir / name}")
        _assert_run_directory_identity(run_dir, expected_directory)
        if _regular_file_identity_at(directory_fd, quarantined) != expected_file:
            raise ArtifactError(f"optional run artifact changed before removal: {run_dir / name}")
        os.unlink(quarantined, dir_fd=directory_fd)
        moved = False
    except (ArtifactError, OSError) as exc:
        if moved:
            _restore_quarantined_optional_run_artifact(
                directory_fd,
                quarantined=quarantined,
                original=name,
            )
        if isinstance(exc, ArtifactError):
            raise
        raise ArtifactError(f"could not safely remove optional run artifact {run_dir / name}: {exc}") from exc
    finally:
        try:
            os.rmdir(quarantine_name, dir_fd=directory_fd)
        except OSError:
            pass
    _assert_run_directory_identity(run_dir, expected_directory)


def _restore_quarantined_optional_run_artifact(
    directory_fd: int,
    *,
    quarantined: str,
    original: str,
) -> None:
    try:
        os.stat(original, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        try:
            os.rename(
                quarantined,
                original,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        except OSError:
            pass


def _is_owned_run_directory(run_dir: Path) -> bool:
    lineage_path = run_dir / "artifact_lineage.json"
    if not lineage_path.is_file() or path_has_symlink_component(lineage_path, include_leaf=True):
        return False
    try:
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not _lineage_has_durable_run_shape(lineage):
        return False
    output_records = lineage["outputs"]
    records_by_name = {
        str(record["name"]): record
        for record in output_records
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }
    if len(records_by_name) != len(output_records):
        return False
    for name, filename in {
        **_REQUIRED_RUN_ARTIFACTS,
        **{
            name: filename
            for name, filename in _OPTIONAL_RUN_ARTIFACT_RECORDS.items()
            if name in records_by_name
        },
    }.items():
        record = records_by_name.get(name)
        if not isinstance(record, dict) or not _lineage_output_fingerprint_matches(
            run_dir,
            record,
            filename,
        ):
            return False

    trace_path = run_dir / _REQUIRED_RUN_ARTIFACTS["normalized_trace"]
    scorecard_path = run_dir / _REQUIRED_RUN_ARTIFACTS["scorecard"]
    try:
        if check_schema_file(trace_path, "trace").get("passed") is not True:
            return False
        if check_schema_file(scorecard_path, "scorecard").get("passed") is not True:
            return False
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, SchemaRegistryError):
        return False
    events = trace.get("events") if isinstance(trace, dict) else None
    scenario = lineage["scenario"]
    lineage_trace = lineage["trace"]
    lineage_scorecard = lineage["scorecard"]
    return (
        isinstance(events, list)
        and scenario.get("id") == scorecard.get("scenario_id")
        and lineage_trace.get("schema_version") == trace.get("schema_version")
        and lineage_trace.get("event_count") == len(events)
        and lineage_scorecard.get("schema_version") == scorecard.get("schema_version")
        and lineage_scorecard.get("score") == scorecard.get("score")
        and lineage_scorecard.get("passed") == scorecard.get("passed")
        and lineage_scorecard.get("critical_failures") == scorecard.get("critical_failures")
    )


def _lineage_has_durable_run_shape(lineage: Any) -> bool:
    if not isinstance(lineage, dict) or lineage.get("schema_version") != LINEAGE_SCHEMA_VERSION:
        return False
    scenario = lineage.get("scenario")
    trace = lineage.get("trace")
    scorecard = lineage.get("scorecard")
    inputs = lineage.get("inputs")
    outputs = lineage.get("outputs")
    evidence_links = lineage.get("evidence_links")
    graph = lineage.get("graph")
    replay = lineage.get("replay")
    summary = lineage.get("summary")
    if not (
        isinstance(scenario, dict)
        and isinstance(scenario.get("id"), str)
        and bool(scenario["id"])
        and isinstance(scenario.get("title"), str)
        and bool(scenario["title"])
        and isinstance(trace, dict)
        and isinstance(scorecard, dict)
        and isinstance(inputs, list)
        and len(inputs) >= 2
        and isinstance(outputs, list)
        and isinstance(evidence_links, list)
        and isinstance(graph, list)
        and bool(graph)
        and isinstance(replay, dict)
        and replay.get("tool") == "flightrecorder"
        and isinstance(replay.get("argv"), list)
        and isinstance(replay.get("input_fingerprints"), dict)
        and isinstance(replay.get("self_contained"), bool)
        and isinstance(summary, dict)
        and summary.get("input_count") == len(inputs)
        and summary.get("output_count") == len(outputs)
        and summary.get("evidence_link_count") == len(evidence_links)
        and summary.get("self_contained_replay") == replay.get("self_contained")
    ):
        return False
    input_records = {
        record.get("name"): record
        for record in inputs
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }
    for name in ("scenario", "source_trace"):
        record = input_records.get(name)
        if not isinstance(record, dict) or record.get("role") != "input":
            return False
        if record.get("exists") is not True or not _is_sha256(record.get("sha256")):
            return False
        if not _is_non_negative_int(record.get("size_bytes")):
            return False
    return True


def _lineage_output_fingerprint_matches(
    run_dir: Path,
    record: dict[str, Any],
    filename: str,
) -> bool:
    path_label = record.get("path")
    path = run_dir / filename
    if (
        record.get("name") not in _RUN_LINEAGE_OUTPUT_NAMES
        or record.get("role") != "output"
        or record.get("exists") is not True
        or not isinstance(record.get("sensitive"), bool)
        or not isinstance(path_label, str)
        or _lineage_output_basename(path_label) != filename
        or not path.is_file()
        or path_has_symlink_component(path, include_leaf=True)
        or not _is_non_negative_int(record.get("size_bytes"))
        or not _is_sha256(record.get("sha256"))
    ):
        return False
    try:
        return path.stat().st_size == record["size_bytes"] and _sha256_file(path) == record["sha256"]
    except OSError:
        return False


def _lineage_output_basename(path_label: str) -> str:
    if path_label.startswith("<redacted:") and path_label.endswith(">"):
        return path_label[len("<redacted:") : -1]
    return path_label.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _run_publication_target_is_owned(run_dir: Path) -> bool:
    """Recognize only a complete run tree with no unclaimed local data."""
    if not _is_owned_run_directory(run_dir):
        return False
    lineage = _read_existing_run_lineage(run_dir)
    if lineage is None:
        return False
    outputs = lineage.get("outputs")
    if not isinstance(outputs, list):
        return False
    records = {
        record.get("name"): record
        for record in outputs
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }
    if len(records) != len(outputs) or not set(records).issubset(_RUN_LINEAGE_OUTPUT_NAMES):
        return False

    owned_names = {"artifact_lineage.json", *_REQUIRED_RUN_ARTIFACTS.values()}
    for name, filename in _OPTIONAL_RUN_ARTIFACT_RECORDS.items():
        if name in records:
            owned_names.add(filename)
    for name in ("junit", "markdown"):
        record = records.get(name)
        if not isinstance(record, dict):
            continue
        path_label = record.get("path")
        if not isinstance(path_label, str):
            return False
        filename = _lineage_output_basename(path_label)
        candidate = run_dir / filename
        if candidate.exists() or candidate.is_symlink():
            if not _lineage_output_fingerprint_matches(run_dir, record, filename):
                return False
            owned_names.add(filename)

    try:
        entries = list(run_dir.iterdir())
    except OSError:
        return False
    return all(
        entry.name in owned_names
        and entry.is_file()
        and not path_has_symlink_component(entry, include_leaf=True)
        for entry in entries
    ) and {entry.name for entry in entries} == owned_names


def _run_directory_identity(run_dir: Path) -> tuple[int, int]:
    try:
        observed = run_dir.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtifactError(f"could not inspect run output directory {run_dir}: {exc}") from exc
    if not stat.S_ISDIR(observed.st_mode):
        raise ArtifactError(f"run output is not a directory: {run_dir}")
    return observed.st_dev, observed.st_ino


def _assert_run_directory_identity(run_dir: Path, expected: tuple[int, int]) -> None:
    if _run_directory_identity(run_dir) != expected:
        raise ArtifactError(f"run output directory identity changed during publication: {run_dir}")


def _attest_run_directory(
    run_dir: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> _RunDirectoryAttestation:
    identity = _run_directory_identity(run_dir)
    if expected_identity is not None and identity != expected_identity:
        raise ArtifactError(f"run output directory identity changed during attestation: {run_dir}")
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    try:
        paths = sorted(run_dir.rglob("*"))
        for path in paths:
            relative = path.relative_to(run_dir).as_posix()
            before = path.stat(follow_symlinks=False)
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            if stat.S_ISDIR(before.st_mode):
                digest.update(b"directory\0")
                continue
            if not stat.S_ISREG(before.st_mode):
                raise ArtifactError(f"run output contains a non-regular path: {path}")
            file_hash = _sha256_file(path)
            after = path.stat(follow_symlinks=False)
            before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            if before_identity != after_identity:
                raise ArtifactError(f"run output file changed during attestation: {path}")
            digest.update(b"file\0")
            digest.update(str(after.st_size).encode("ascii"))
            digest.update(b"\0")
            digest.update(file_hash.encode("ascii"))
            digest.update(b"\0")
            file_count += 1
            size_bytes += after.st_size
    except OSError as exc:
        raise ArtifactError(f"could not attest run output {run_dir}: {exc}") from exc
    if _run_directory_identity(run_dir) != identity:
        raise ArtifactError(f"run output directory identity changed during attestation: {run_dir}")
    return _RunDirectoryAttestation(identity, digest.hexdigest(), file_count, size_bytes)


def _require_run_attestation(
    run_dir: Path,
    expected: _RunDirectoryAttestation,
    label: str,
) -> None:
    try:
        actual = _attest_run_directory(run_dir, expected_identity=expected.identity)
    except ArtifactError as exc:
        raise ArtifactError(f"{label} changed before publication: {exc}") from exc
    if actual != expected:
        raise ArtifactError(f"{label} contents changed before publication")


def _create_private_run_directory(target: Path) -> tuple[Path, tuple[int, int]]:
    try:
        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name or 'run'}.hfr-stage-", dir=target.parent)
        )
        staging.chmod(0o700)
    except OSError as exc:
        raise ArtifactError(f"could not create private run staging directory: {exc}") from exc
    return staging, _run_directory_identity(staging)


def _remove_owned_run_directory(run_dir: Path, identity: tuple[int, int]) -> bool:
    if _run_path_identity(run_dir) is None:
        return True
    return remove_directory_tree_if_identity(run_dir, identity)


def _remove_attested_run_directory(
    run_dir: Path,
    expected: _RunDirectoryAttestation,
) -> bool:
    try:
        _require_run_attestation(run_dir, expected, "displaced run output")
    except ArtifactError:
        return False
    return _remove_owned_run_directory(run_dir, expected.identity)


def _fsync_run_tree(run_dir: Path) -> None:
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        try:
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ArtifactError(f"could not sync staged run artifact {path}: {exc}") from exc
    for directory in sorted(
        [run_dir, *(item for item in run_dir.rglob("*") if item.is_dir())],
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_run_directory(directory)


def _fsync_run_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_rename_new_run_directory(source: Path, destination: Path) -> None:
    if sys.platform == "darwin":
        _darwin_atomic_run_rename(source, destination, 0x00000004)  # RENAME_EXCL
        return
    if sys.platform.startswith("linux"):
        _linux_atomic_run_rename(source, destination, 0x00000001)  # RENAME_NOREPLACE
        return
    raise ArtifactError("atomic exclusive run publication is unavailable on this platform")


def _atomic_exchange_run_directories(first: Path, second: Path) -> None:
    if sys.platform == "darwin":
        _darwin_atomic_run_rename(first, second, 0x00000002)  # RENAME_SWAP
        return
    if sys.platform.startswith("linux"):
        _linux_atomic_run_rename(first, second, 0x00000002)  # RENAME_EXCHANGE
        return
    raise ArtifactError("atomic run directory exchange is unavailable on this platform")


def _darwin_atomic_run_rename(source: Path, destination: Path, flags: int) -> None:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renamex_np = getattr(libc, "renamex_np", None)
    if renamex_np is None:
        raise ArtifactError("atomic run publication is unavailable: renamex_np is missing")
    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int
    if renamex_np(os.fsencode(source), os.fsencode(destination), flags) != 0:
        error_number = ctypes.get_errno()
        raise ArtifactError(f"atomic run publication failed: {os.strerror(error_number)}")


def _linux_atomic_run_rename(source: Path, destination: Path, flags: int) -> None:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ArtifactError("atomic run publication is unavailable: renameat2 is missing")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), flags) != 0:
        error_number = ctypes.get_errno()
        raise ArtifactError(f"atomic run publication failed: {os.strerror(error_number)}")


def _run_path_identity(path: Path) -> tuple[int, int] | None:
    try:
        observed = path.stat(follow_symlinks=False)
    except OSError:
        return None
    return observed.st_dev, observed.st_ino


def _publish_staged_run_directory(
    staging: Path,
    target: Path,
    *,
    staged_attestation: _RunDirectoryAttestation,
    target_attestation: _RunDirectoryAttestation | None,
) -> None:
    _require_run_attestation(staging, staged_attestation, "staged run output")
    if target_attestation is None:
        if _run_path_identity(target) is not None:
            raise ArtifactError("run output appeared after publication admission")
        _atomic_rename_new_run_directory(staging, target)
        _require_run_attestation(target, staged_attestation, "published run output")
        _fsync_run_directory(target.parent)
        return

    _require_run_attestation(target, target_attestation, "existing run output")
    try:
        _atomic_exchange_run_directories(staging, target)
    except BaseException as exc:
        if (
            _run_path_identity(target) == staged_attestation.identity
            and _run_path_identity(staging) is not None
        ):
            try:
                _atomic_exchange_run_directories(staging, target)
                _fsync_run_directory(target.parent)
            except BaseException as rollback_exc:
                raise ArtifactError(
                    f"run publication was interrupted after exchange and atomic rollback failed: {rollback_exc}"
                ) from exc
        raise

    try:
        _require_run_attestation(target, staged_attestation, "published run output")
        _require_run_attestation(staging, target_attestation, "displaced run output")
    except ArtifactError as exc:
        rollback_error: BaseException | None = None
        if (
            _run_path_identity(target) == staged_attestation.identity
            and _run_path_identity(staging) is not None
        ):
            try:
                _atomic_exchange_run_directories(staging, target)
                _fsync_run_directory(target.parent)
            except BaseException as rollback_exc:
                rollback_error = rollback_exc
        detail = f"run publication attestation failed: {exc}"
        if rollback_error is not None:
            detail += f"; atomic rollback failed: {rollback_error}"
        raise ArtifactError(detail) from exc

    _fsync_run_directory(target.parent)
    if not _remove_attested_run_directory(staging, target_attestation):
        raise ArtifactError(
            f"run published, but the displaced output changed and was retained at {staging}"
        )
    _fsync_run_directory(target.parent)


def _run_work_directory_is_admissible(run_dir: Path) -> bool:
    """Allow a new private stage or a compound output already holding the lock."""
    if _is_owned_run_directory(run_dir):
        return True
    marker_names = {
        *_REQUIRED_RUN_ARTIFACTS.values(),
        *_OPTIONAL_RUN_ARTIFACTS,
        "artifact_lineage.json",
    }
    return not any(
        (run_dir / name).exists() or (run_dir / name).is_symlink()
        for name in marker_names
    )


def _stage_run_sidecar_path(
    requested: str | Path | None,
    *,
    target: Path,
    staging: Path,
) -> str | Path | None:
    if requested is None:
        return None
    path = Path(requested)
    if path.resolve(strict=False).parent == target.resolve(strict=False):
        return staging / path.name
    return requested


def _is_compound_run_output(run_dir: Path, scenario_path: str | Path) -> bool:
    """Identify smoke/harness workdirs that deliberately keep inputs beside outputs."""
    scenario = Path(scenario_path)
    if path_has_symlink_component(scenario, include_leaf=True):
        return False
    return scenario.resolve(strict=False).parent == run_dir.resolve(strict=False)


def _run_scenario_artifacts(
    scenario_path: str | Path,
    out_dir: str | Path,
    *,
    trace_override: str | Path | None = None,
    state_override: str | Path | None = None,
    before_state_override: str | Path | None = None,
    trace_format: str = "auto",
    write_sensitive_trace: bool = False,
    preserve_paths: bool = False,
    junit_out: str | Path | None = None,
    markdown_out: str | Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(out_dir)
    if output_directory_lock_is_held(run_dir):
        return _run_scenario_artifacts_locked(
            scenario_path,
            out_dir,
            trace_override=trace_override,
            state_override=state_override,
            before_state_override=before_state_override,
            trace_format=trace_format,
            write_sensitive_trace=write_sensitive_trace,
            preserve_paths=preserve_paths,
            junit_out=junit_out,
            markdown_out=markdown_out,
            publication_target=run_dir,
            display_run_dir=run_dir,
            display_junit_out=junit_out,
            display_markdown_out=markdown_out,
        )
    compound_output = _is_compound_run_output(run_dir, scenario_path)
    try:
        with locked_owned_output_directory(
            run_dir,
            repo_root=Path(__file__).resolve().parents[2],
            force=False,
            label="run output",
            is_owned=lambda target: _run_publication_target_is_owned(target)
            or (compound_output and _run_work_directory_is_admissible(target)),
            keep_existing=True,
        ):
            if (
                compound_output
                and run_dir.exists()
                and not _run_publication_target_is_owned(run_dir)
            ):
                return _run_scenario_artifacts_locked(
                    scenario_path,
                    run_dir,
                    trace_override=trace_override,
                    state_override=state_override,
                    before_state_override=before_state_override,
                    trace_format=trace_format,
                    write_sensitive_trace=write_sensitive_trace,
                    preserve_paths=preserve_paths,
                    junit_out=junit_out,
                    markdown_out=markdown_out,
                    expected_run_dir_identity=_run_directory_identity(run_dir),
                    publication_target=run_dir,
                    display_run_dir=run_dir,
                    display_junit_out=junit_out,
                    display_markdown_out=markdown_out,
                )
            if run_dir.exists() and any(run_dir.iterdir()) and not _run_publication_target_is_owned(run_dir):
                raise ArtifactError(f"refusing to publish over unrecognized run output: {run_dir}")
            target_attestation = _attest_run_directory(run_dir) if run_dir.exists() else None
            staging, staging_identity = _create_private_run_directory(run_dir)
            staged_attestation: _RunDirectoryAttestation | None = None
            published = False
            try:
                staged_junit = _stage_run_sidecar_path(
                    junit_out,
                    target=run_dir,
                    staging=staging,
                )
                staged_markdown = _stage_run_sidecar_path(
                    markdown_out,
                    target=run_dir,
                    staging=staging,
                )
                result = _run_scenario_artifacts_locked(
                    scenario_path,
                    staging,
                    trace_override=trace_override,
                    state_override=state_override,
                    before_state_override=before_state_override,
                    trace_format=trace_format,
                    write_sensitive_trace=write_sensitive_trace,
                    preserve_paths=preserve_paths,
                    junit_out=staged_junit,
                    markdown_out=staged_markdown,
                    expected_run_dir_identity=staging_identity,
                    publication_target=run_dir,
                    display_run_dir=run_dir,
                    display_junit_out=junit_out,
                    display_markdown_out=markdown_out,
                )
                if not _run_publication_target_is_owned(staging):
                    raise ArtifactError(
                        f"generated run failed complete publication-tree verification: {staging}"
                    )
                _fsync_run_tree(staging)
                staged_attestation = _attest_run_directory(
                    staging,
                    expected_identity=staging_identity,
                )
                _publish_staged_run_directory(
                    staging,
                    run_dir,
                    staged_attestation=staged_attestation,
                    target_attestation=target_attestation,
                )
                published = True
                return result
            finally:
                if not published:
                    if staged_attestation is None:
                        _remove_owned_run_directory(staging, staging_identity)
                    else:
                        _remove_attested_run_directory(staging, staged_attestation)
    except ArtifactError:
        raise
    except ValueError as exc:
        raise ArtifactError(str(exc)) from exc


def _rewrite_staged_run_lineage(
    lineage: dict[str, Any],
    *,
    display_run_dir: Path,
    preserve_paths: bool,
    artifact_paths: dict[str, str | Path | None],
) -> dict[str, Any]:
    """Replace private staging labels without changing staged fingerprints."""
    outputs = lineage.get("outputs")
    if isinstance(outputs, list):
        for record in outputs:
            if not isinstance(record, dict):
                continue
            name = record.get("name")
            display_path = artifact_paths.get(name) if isinstance(name, str) else None
            if display_path is not None:
                record["path"] = _display_path(Path(display_path), preserve_paths)
    replay = lineage.get("replay")
    if isinstance(replay, dict) and isinstance(replay.get("argv"), list):
        argv = [str(value) for value in replay["argv"]]
        try:
            out_index = argv.index("--out") + 1
        except (ValueError, IndexError):
            pass
        else:
            argv[out_index] = _display_path(display_run_dir, preserve_paths)
            replay["argv"] = argv
            replay["command"] = " ".join(shlex.quote(value) for value in argv)
    return lineage


def _run_scenario_artifacts_locked(
    scenario_path: str | Path,
    out_dir: str | Path,
    *,
    trace_override: str | Path | None = None,
    state_override: str | Path | None = None,
    before_state_override: str | Path | None = None,
    trace_format: str = "auto",
    write_sensitive_trace: bool = False,
    preserve_paths: bool = False,
    junit_out: str | Path | None = None,
    markdown_out: str | Path | None = None,
    expected_run_dir_identity: tuple[int, int] | None = None,
    publication_target: Path | None = None,
    display_run_dir: Path | None = None,
    display_junit_out: str | Path | None = None,
    display_markdown_out: str | Path | None = None,
) -> dict[str, Any]:
    scenario = load_scenario(scenario_path)
    trace_path = resolve_trace_path(scenario, trace_override)
    before_state_path = resolve_before_state_snapshot_path(scenario, before_state_override)
    state_path = resolve_state_snapshot_path(scenario, state_override)
    run_dir = Path(out_dir)
    publication_target = publication_target or run_dir
    display_run_dir = display_run_dir or run_dir
    if path_has_symlink_component(run_dir, include_leaf=True):
        raise ArtifactError(f"run output must not contain symlink components: {run_dir}")
    if not output_directory_lock_is_held(publication_target):
        raise ArtifactError(f"run output publication lock is not held: {publication_target}")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir_identity = _run_directory_identity(run_dir)
    if expected_run_dir_identity is not None and run_dir_identity != expected_run_dir_identity:
        raise ArtifactError(f"run output directory identity changed before publication: {run_dir}")
    if not _run_work_directory_is_admissible(run_dir):
        raise ArtifactError(f"run output is not a recognized prior run or a new work directory: {run_dir}")

    raw_trace = normalize_trace(trace_path, trace_format)
    raw_before_state_snapshot = load_state_snapshot(before_state_path) if before_state_path is not None else None
    raw_state_snapshot = load_state_snapshot(state_path) if state_path is not None else None
    trace_label = _display_path(trace_path, preserve_paths)
    scenario.setdefault("trace", {})["path"] = trace_label
    if before_state_path is not None:
        scenario.setdefault("state", {})["before_path"] = _display_path(before_state_path, preserve_paths)
        scenario["state"]["format"] = "json"
    if state_path is not None:
        scenario.setdefault("state", {})["path"] = _display_path(state_path, preserve_paths)
        scenario["state"]["format"] = "json"
    scorecard = score_trace(scenario, raw_trace, raw_state_snapshot, raw_before_state_snapshot)
    secret_patterns = scenario.get("policy", {}).get("secret_patterns") or []
    trace = sanitize_trace(raw_trace, secret_patterns)
    raw_trajectory_context = trajectory_v2_context_from_path(trace_path)
    trajectory_v2 = None
    if raw_trajectory_context:
        trajectory_v2 = trajectory_v2_from_trace(
            trace,
            source_path=trace_path,
            source_format=str(trace.get("session", {}).get("source_format") or trace_format),
            context=_sanitize_trajectory_context(raw_trajectory_context, secret_patterns),
        )
    before_state_snapshot = (
        sanitize_state_snapshot(raw_before_state_snapshot, secret_patterns)
        if raw_before_state_snapshot is not None
        else None
    )
    state_snapshot = sanitize_state_snapshot(raw_state_snapshot, secret_patterns) if raw_state_snapshot is not None else None

    normalized_path = run_dir / "normalized_trace.json"
    trajectory_v2_path = run_dir / "trajectory_v2.json" if trajectory_v2 is not None else None
    score_path = run_dir / "scorecard.json"
    task_completion_path = run_dir / "task_completion.json"
    before_state_snapshot_path = run_dir / "before_state_snapshot.json" if before_state_snapshot is not None else None
    state_snapshot_path = run_dir / "state_snapshot.json" if state_snapshot is not None else None
    state_diff = (
        build_state_diff(before_state_snapshot, state_snapshot)
        if before_state_snapshot is not None and state_snapshot is not None
        else None
    )
    state_diff_path = run_dir / "state_diff.json" if state_diff is not None else None
    run_digest_path = run_dir / "run_digest.json"
    report_path = run_dir / "report.html"
    lineage_path = run_dir / "artifact_lineage.json"
    if run_dir.resolve(strict=False) == publication_target.resolve(strict=False):
        _remove_stale_optional_run_artifacts(
            run_dir,
            expected_identity=run_dir_identity,
            junit_out=junit_out,
            markdown_out=markdown_out,
        )
    _write_json(normalized_path, trace)
    if trajectory_v2_path is not None and trajectory_v2 is not None:
        write_trajectory_v2(trajectory_v2_path, trajectory_v2)
    if before_state_snapshot_path is not None:
        _write_json(before_state_snapshot_path, before_state_snapshot)
    if state_snapshot_path is not None:
        _write_json(state_snapshot_path, state_snapshot)
    if state_diff_path is not None and state_diff is not None:
        _write_json(state_diff_path, state_diff)
    _write_json(score_path, scorecard)
    _write_json(task_completion_path, scorecard["task_completion"])
    run_digest = build_run_digest(scenario, trace, scorecard, state_diff=state_diff)
    _write_json(run_digest_path, run_digest)
    raw_trace_path = None
    if write_sensitive_trace:
        raw_trace_path = run_dir / "raw_trace.sensitive.json"
        _write_json(raw_trace_path, raw_trace)

    regression_path = None
    regression_display = None
    if not scorecard["passed"]:
        regression_path = run_dir / "regression_scenario.json"
        display_regression_path = display_run_dir / "regression_scenario.json"
        regression_display = _display_path(display_regression_path, preserve_paths)
        regression = _regression_scenario(
            scenario,
            trace_path,
            display_regression_path,
            preserve_paths,
        )
        _write_json(regression_path, regression)

    if junit_out:
        write_junit(scorecard, junit_out)
    if markdown_out:
        write_markdown_summary(scorecard, markdown_out)
    write_report(scenario, trace, scorecard, report_path, regression_display, state_diff)
    lineage = write_run_lineage(
        scenario=scenario,
        trace=trace,
        scorecard=scorecard,
        source_trace_path=trace_path,
        source_before_state_snapshot_path=before_state_path,
        source_state_snapshot_path=state_path,
        artifacts={
            "normalized_trace": normalized_path,
            "trajectory_v2": trajectory_v2_path,
            "before_state_snapshot": before_state_snapshot_path,
            "state_snapshot": state_snapshot_path,
            "state_diff": state_diff_path,
            "scorecard": score_path,
            "task_completion": task_completion_path,
            "run_digest": run_digest_path,
            "report": report_path,
            "regression_scenario": regression_path,
            "junit": junit_out,
            "markdown": markdown_out,
            "raw_trace_sensitive": raw_trace_path,
        },
        out_path=lineage_path,
        preserve_paths=preserve_paths,
    )
    lineage = _rewrite_staged_run_lineage(
        lineage,
        display_run_dir=display_run_dir,
        preserve_paths=preserve_paths,
        artifact_paths={
            "normalized_trace": display_run_dir / "normalized_trace.json",
            "trajectory_v2": (
                display_run_dir / "trajectory_v2.json" if trajectory_v2_path is not None else None
            ),
            "before_state_snapshot": (
                display_run_dir / "before_state_snapshot.json"
                if before_state_snapshot_path is not None
                else None
            ),
            "state_snapshot": (
                display_run_dir / "state_snapshot.json"
                if state_snapshot_path is not None
                else None
            ),
            "state_diff": (
                display_run_dir / "state_diff.json" if state_diff_path is not None else None
            ),
            "scorecard": display_run_dir / "scorecard.json",
            "task_completion": display_run_dir / "task_completion.json",
            "run_digest": display_run_dir / "run_digest.json",
            "report": display_run_dir / "report.html",
            "regression_scenario": (
                display_run_dir / "regression_scenario.json" if regression_path is not None else None
            ),
            "junit": display_junit_out if display_junit_out is not None else junit_out,
            "markdown": display_markdown_out if display_markdown_out is not None else markdown_out,
            "raw_trace_sensitive": (
                display_run_dir / "raw_trace.sensitive.json" if raw_trace_path is not None else None
            ),
        },
    )
    _write_json(lineage_path, lineage)
    _assert_run_directory_identity(run_dir, run_dir_identity)
    if not _is_owned_run_directory(run_dir):
        raise ArtifactError(
            f"generated run failed durable shape and fingerprint verification: {run_dir}"
        )
    _assert_run_directory_identity(run_dir, run_dir_identity)

    return {
        "scenario": scenario,
        "trace_path": trace_path,
        "before_state_path": before_state_path,
        "state_path": state_path,
        "scorecard": scorecard,
        "paths": {
            "run_dir": display_run_dir,
            "normalized_trace": display_run_dir / "normalized_trace.json",
            "trajectory_v2": (
                display_run_dir / "trajectory_v2.json" if trajectory_v2_path is not None else None
            ),
            "before_state_snapshot": (
                display_run_dir / "before_state_snapshot.json"
                if before_state_snapshot_path is not None
                else None
            ),
            "state_snapshot": (
                display_run_dir / "state_snapshot.json" if state_snapshot_path is not None else None
            ),
            "state_diff": display_run_dir / "state_diff.json" if state_diff_path is not None else None,
            "scorecard": display_run_dir / "scorecard.json",
            "task_completion": display_run_dir / "task_completion.json",
            "run_digest": display_run_dir / "run_digest.json",
            "report": display_run_dir / "report.html",
            "lineage": display_run_dir / "artifact_lineage.json",
        },
        "lineage": lineage,
    }


def _sanitize_trajectory_context(
    context: dict[str, Any],
    secret_patterns: list[str],
) -> dict[str, Any]:
    """Redact user-bearing context without corrupting immutable identity names.

    Scenario regexes such as ``token`` are intentionally broad for message
    content, but applying them to a tokenizer ID would destroy the recorded
    identity. Generic secret-key/assignment redaction still applies everywhere.
    """

    sanitized = sanitize_trace(context, [])
    for field in ("messages", "approvals", "event_metadata", "metadata", "governance"):
        if field in context:
            sanitized[field] = sanitize_trace({field: context[field]}, secret_patterns)[field]
    return sanitized
