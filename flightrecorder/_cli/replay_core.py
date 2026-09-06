"""CLI commands for the replay core domain."""

from __future__ import annotations

from typing import Any, Iterator
from pathlib import Path
from ..lineage import LINEAGE_SCHEMA_VERSION, REPLAY_BUNDLE_SCHEMA_VERSION, write_run_lineage
import copy
import shlex
import shutil
from .constants import FAMILY_SUFFIX_RE
from .shared import _display_path, _read_json
from ..hashing import sha256_file as _sha256_file


class ReplayError(ValueError):
    """Raised when a lineage replay contract cannot be safely rerun."""


def _copy_replay_input(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": destination,
        "relative_path": f"inputs/{destination.name}",
        "source_path": source,
        "sha256": _sha256_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def _trace_bundle_name(trace_path: Path) -> str:
    suffix = "".join(trace_path.suffixes)
    return "source_trace" + (suffix if suffix else ".jsonl")


def _portable_replay_lineage(
    *,
    lineage: dict[str, Any],
    source_lineage_path: Path,
    copied_inputs: dict[str, dict[str, Any]],
    preserve_paths: bool,
) -> dict[str, Any]:
    bundle_lineage = copy.deepcopy(lineage)
    replay = bundle_lineage.get("replay")
    if not isinstance(replay, dict):
        replay = {}
        bundle_lineage["replay"] = replay
    argv = replay.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ReplayError("artifact_lineage.replay.argv must be a list of strings")
    argv = list(argv)
    _replace_replay_flag(argv, "--scenario", _bundle_relative_path(copied_inputs["scenario"]))
    _replace_replay_flag(argv, "--trace", _bundle_relative_path(copied_inputs["source_trace"]))
    _replace_replay_flag(argv, "--out", "replay")
    if "source_before_state_snapshot" in copied_inputs:
        _replace_replay_flag(
            argv,
            "--before-state",
            _bundle_relative_path(copied_inputs["source_before_state_snapshot"]),
            required=False,
        )
    if "source_state_snapshot" in copied_inputs:
        _replace_replay_flag(argv, "--state", _bundle_relative_path(copied_inputs["source_state_snapshot"]), required=False)
    replay["argv"] = argv
    replay["command"] = " ".join(shlex.quote(arg) for arg in argv)
    replay["self_contained"] = True
    input_fingerprints = replay.get("input_fingerprints")
    if not isinstance(input_fingerprints, dict):
        input_fingerprints = {}
        replay["input_fingerprints"] = input_fingerprints
    for name, copied in copied_inputs.items():
        record = input_fingerprints.get(name)
        if not isinstance(record, dict):
            record = {}
            input_fingerprints[name] = record
        record["path"] = _bundle_relative_path(copied)
        record["sha256"] = copied["sha256"]
        record["size_bytes"] = copied["size_bytes"]
        record["exists"] = True
    _rewrite_lineage_inputs_for_bundle(bundle_lineage, copied_inputs)
    summary = bundle_lineage.get("summary")
    if isinstance(summary, dict):
        summary["self_contained_replay"] = True
    bundle_lineage["portable_replay_bundle"] = {
        "schema_version": REPLAY_BUNDLE_SCHEMA_VERSION,
        "source_lineage": _display_path(source_lineage_path, preserve_paths),
        "input_count": len(copied_inputs),
        "notes": [
            "This lineage was rewritten for a portable replay bundle.",
            "Replay input paths are relative to the directory containing artifact_lineage.json.",
        ],
    }
    return bundle_lineage


def _rewrite_lineage_inputs_for_bundle(lineage: dict[str, Any], copied_inputs: dict[str, dict[str, Any]]) -> None:
    inputs = lineage.get("inputs")
    if not isinstance(inputs, list):
        return
    for record in inputs:
        if not isinstance(record, dict):
            continue
        name = record.get("name")
        copied = copied_inputs.get(name) if isinstance(name, str) else None
        if copied is None:
            continue
        record["path"] = _bundle_relative_path(copied)
        record["exists"] = True
        record["sha256"] = copied["sha256"]
        record["size_bytes"] = copied["size_bytes"]


def _replace_replay_flag(argv: list[str], flag: str, value: str, *, required: bool = True) -> None:
    if flag not in argv:
        if required:
            raise ReplayError(f"artifact_lineage.replay.argv missing {flag}")
        argv.extend([flag, value])
        return
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ReplayError(f"artifact_lineage.replay.argv missing value for {flag}")
    argv[index + 1] = value


def _replay_bundle_manifest(
    *,
    bundle_lineage: dict[str, Any],
    bundle_lineage_path: Path,
    source_lineage_path: Path,
    copied_inputs: dict[str, dict[str, Any]],
    preserve_paths: bool,
) -> dict[str, Any]:
    replay = bundle_lineage.get("replay") if isinstance(bundle_lineage.get("replay"), dict) else {}
    return {
        "schema_version": REPLAY_BUNDLE_SCHEMA_VERSION,
        "lineage": bundle_lineage_path.name,
        "source_lineage": _display_path(source_lineage_path, preserve_paths),
        "input_count": len(copied_inputs),
        "inputs": [
            {
                "name": name,
                "path": _bundle_relative_path(copied),
                "sha256": copied["sha256"],
                "size_bytes": copied["size_bytes"],
                "source_path": _display_path(copied["source_path"], preserve_paths),
            }
            for name, copied in sorted(copied_inputs.items())
        ],
        "replay": {
            "argv": replay.get("argv", []),
            "command": replay.get("command", ""),
            "self_contained": replay.get("self_contained") is True,
        },
        "notes": [
            "Move this directory as a unit, then run flightrecorder replay --lineage artifact_lineage.json --out <fresh-run> from anywhere.",
            "The replay command verifies copied input hashes before regenerating artifacts.",
        ],
    }


def _bundle_relative_path(copied: dict[str, Any]) -> str:
    return str(copied["relative_path"])


def _int_score(value: Any) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


def _task_family(scenario_id: str) -> str:
    family = FAMILY_SUFFIX_RE.sub("", scenario_id).strip("_-")
    return family or scenario_id or "unknown"


def _regression_scenario(scenario: dict[str, Any], trace_path: Path, regression_path: Path, preserve_paths: bool) -> dict[str, Any]:
    regression = {
        key: value
        for key, value in scenario.items()
        if not key.startswith("_")
    }
    trace_ref = _display_path(trace_path, preserve_paths)
    scenario_ref = _display_path(regression_path, preserve_paths)
    regression["id"] = f"{scenario['id']}_regression"
    regression["title"] = f"Regression: {scenario['title']}"
    regression["trace"] = {"format": "auto", "path": trace_ref}
    if trace_ref.startswith("<redacted:"):
        regression["trace"]["path_redacted"] = True
    regression["rerun_command"] = (
        f"python -m flightrecorder run --scenario {shlex.quote(scenario_ref)} "
        f"--trace {shlex.quote(trace_ref)} --out {shlex.quote('runs/' + scenario['id'] + '_replay')}"
    )
    return regression


def _read_scorecard_ref(path: Path) -> tuple[dict[str, Any], str]:
    score_path = path / "scorecard.json" if path.is_dir() else path
    return _read_json(score_path), _display_path(score_path)


def _audit_runs(runs_dir: Path, forbidden_text: list[str]) -> dict[str, Any]:
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    if not runs_dir.is_dir():
        raise NotADirectoryError(f"Runs path is not a directory: {runs_dir}")

    scorecards: list[dict[str, Any]] = []
    leaks: list[dict[str, str]] = []
    for score_path in sorted(runs_dir.glob("*/scorecard.json")):
        scorecard = _read_json(score_path)
        scorecards.append(
            {
                "run": score_path.parent.name,
                "scenario_id": scorecard.get("scenario_id"),
                "passed": bool(scorecard.get("passed")),
                "score": scorecard.get("score"),
                "critical_failures": scorecard.get("critical_failures", []),
            }
        )

    needles = [needle for needle in forbidden_text if needle]
    if needles and runs_dir.exists():
        for path in sorted(runs_dir.rglob("*")):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in needles:
                if needle in text:
                    leaks.append({"path": str(path), "text": needle})

    passed = sum(1 for item in scorecards if item["passed"])
    failed = len(scorecards) - passed
    return {
        "runs_dir": str(runs_dir),
        "total": len(scorecards),
        "passed": passed,
        "failed": failed,
        "leaks": leaks,
        "scorecards": scorecards,
    }
