"""CLI commands for the suite core domain."""

from __future__ import annotations

from typing import Any, Iterator
from ..bundle import HARNESS_RUN_MANIFEST_SCHEMA_VERSION, HARNESS_RUN_RESULT_SCHEMA_VERSION, EvidenceBundleError, build_evidence_bundle
from pathlib import Path
import hashlib
import os
from .constants import RUN_SUITE_SCHEMA_VERSION
from .replay_core import ReplayError, _int_score, _sha256_file, _task_family
from .shared import _display_path_for_output_source, _lineage_record_path, _read_json, _write_json


def _write_run_suite_harness_handoff(
    out_dir: Path,
    runs: list[dict[str, Any]],
    *,
    summary_path: Path,
) -> dict[str, Path] | None:
    selected = _select_run_suite_harness_run(out_dir, runs)
    if selected is None:
        return None

    run, run_dir = selected
    harness_dir = out_dir / "harness_handoff"
    manifest_path = harness_dir / "harness_manifest.json"
    result_path = harness_dir / "harness_result.json"
    trace_path = run_dir / "normalized_trace.json"
    scorecard_path = run_dir / "scorecard.json"
    run_digest_path = run_dir / "run_digest.json"
    report_path = run_dir / "report.html"
    lineage_path = run_dir / "artifact_lineage.json"

    trace = _read_json(trace_path)
    scorecard = _read_json(scorecard_path)
    lineage = _read_json(lineage_path)
    scenario_id = str(run["scenario_id"])
    model_id = _run_suite_harness_model_id(trace)
    provider = _run_suite_harness_provider(trace)
    suite = _run_suite_harness_suite_context(
        harness_dir=harness_dir,
        out_dir=out_dir,
        summary_path=summary_path,
        runs=runs,
        selected_scenario_id=scenario_id,
        selected_run_dir=run_dir,
    )
    sandbox = {
        "root": _harness_relative_path(harness_dir, out_dir),
        "home": _harness_relative_path(harness_dir, out_dir),
        "workspace": _harness_relative_path(harness_dir, run_dir),
        "events": _harness_relative_path(harness_dir, trace_path),
        "fake_secret_canaries": [
            {
                "name": "HFR_RUN_SUITE_FAKE_SECRET_CANARY",
                "sha256": hashlib.sha256(b"HFR_RUN_SUITE_FAKE_SECRET_CANARY").hexdigest(),
            }
        ],
    }
    tool_policy = {
        "source": "flightrecorder run-suite --evidence-handoff",
        "scenario_policy": {},
        "runtime_policy": {
            "mode": "local_run_artifacts",
            "allowed_tools": [],
            "denied_tools": [],
            "network": {"mode": "not_required", "allowed_hosts": []},
        },
        "blocked_action_canaries": [],
    }

    manifest = {
        "schema_version": HARNESS_RUN_MANIFEST_SCHEMA_VERSION,
        "runner": "flightrecorder_run_suite",
        "provider": provider,
        "model": {"id": model_id},
        "scenario": {
            "id": scenario_id,
            "path": _run_suite_harness_scenario_path(lineage, run),
        },
        "outputs": {
            "run_dir": _harness_relative_path(harness_dir, run_dir),
            "manifest": "harness_manifest.json",
            "result": "harness_result.json",
        },
        "sandbox": sandbox,
        "tool_policy": tool_policy,
        "suite": suite,
    }
    result = {
        "schema_version": HARNESS_RUN_RESULT_SCHEMA_VERSION,
        "runner": "flightrecorder_run_suite",
        "provider": provider,
        "model": {"id": model_id},
        "scenario_id": scenario_id,
        "sandbox": sandbox,
        "tool_policy": tool_policy,
        "trace": {
            "path": _harness_relative_path(harness_dir, trace_path),
            "sha256": _sha256_file(trace_path),
            "size_bytes": trace_path.stat().st_size,
            "format": "normalized_json",
            "source_format": _run_suite_harness_source_format(trace),
        },
        "scorecard": {
            "path": _harness_relative_path(harness_dir, scorecard_path),
            "sha256": _sha256_file(scorecard_path),
            "size_bytes": scorecard_path.stat().st_size,
            "passed": scorecard.get("passed") is True,
            "score": scorecard.get("score", 0),
        },
        "artifacts": _run_suite_harness_artifacts(
            harness_dir,
            {
                "normalized_trace": trace_path,
                "scorecard": scorecard_path,
                "run_digest": run_digest_path,
                "report": report_path,
                "lineage": lineage_path,
            },
        ),
        "replay": {
            "lineage": _harness_relative_path(harness_dir, lineage_path),
            "lineage_sha256": _sha256_file(lineage_path),
            "lineage_size_bytes": lineage_path.stat().st_size,
            "self_contained": _run_suite_harness_replay_self_contained(lineage),
        },
        "suite": suite,
    }
    _write_json(manifest_path, manifest)
    _write_json(result_path, result)
    return {"harness_manifest": manifest_path, "harness_result": result_path}


def _run_suite_harness_artifacts(harness_dir: Path, artifacts: dict[str, Path]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, path in artifacts.items():
        records[name] = _harness_relative_path(harness_dir, path)
        records[f"{name}_sha256"] = _sha256_file(path)
        records[f"{name}_size_bytes"] = path.stat().st_size
    return records


def _run_suite_harness_suite_context(
    *,
    harness_dir: Path,
    out_dir: Path,
    summary_path: Path,
    runs: list[dict[str, Any]],
    selected_scenario_id: str,
    selected_run_dir: Path,
) -> dict[str, Any]:
    passed = sum(1 for run in runs if run.get("passed") is True)
    total = len(runs)
    return {
        "source": "flightrecorder run-suite --evidence-handoff",
        "schema_version": RUN_SUITE_SCHEMA_VERSION,
        "summary": _harness_relative_path(harness_dir, summary_path),
        "runs_dir": _harness_relative_path(harness_dir, out_dir),
        "selected_scenario_id": selected_scenario_id,
        "selected_run_id": selected_run_dir.name,
        "selected_run_dir": _harness_relative_path(harness_dir, selected_run_dir),
        "total": total,
        "passed": passed,
        "failed": total - passed,
    }


def _select_run_suite_harness_run(out_dir: Path, runs: list[dict[str, Any]]) -> tuple[dict[str, Any], Path] | None:
    for run in sorted(runs, key=lambda item: str(item.get("scenario_id") or "")):
        scenario_id = run.get("scenario_id")
        if run.get("passed") is not True or not isinstance(scenario_id, str) or not scenario_id:
            continue
        run_dir = out_dir / _safe_run_id(scenario_id)
        required = (
            run_dir / "normalized_trace.json",
            run_dir / "scorecard.json",
            run_dir / "run_digest.json",
            run_dir / "report.html",
            run_dir / "artifact_lineage.json",
        )
        if all(path.exists() for path in required):
            return run, run_dir
    return None


def _run_suite_harness_model_id(trace: dict[str, Any]) -> str:
    session = trace.get("session") if isinstance(trace.get("session"), dict) else {}
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    for value in (session.get("model"), metadata.get("model"), metadata.get("model_id")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _run_suite_harness_provider(trace: dict[str, Any]) -> str:
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    for value in (metadata.get("provider"), metadata.get("source_provider")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    source_format = _run_suite_harness_source_format(trace)
    normalized_source_format = source_format.lower()
    if "fixture" in normalized_source_format or "observer" in normalized_source_format:
        return "fixture"
    return "flightrecorder"


def _run_suite_harness_source_format(trace: dict[str, Any]) -> str:
    session = trace.get("session") if isinstance(trace.get("session"), dict) else {}
    value = session.get("source_format")
    return value.strip() if isinstance(value, str) and value.strip() else "unknown"


def _run_suite_harness_scenario_path(lineage: dict[str, Any], run: dict[str, Any]) -> str:
    path = _lineage_record_path(lineage, "inputs", "scenario")
    if isinstance(path, str) and path:
        return path
    scenario_path = run.get("scenario_path")
    return scenario_path if isinstance(scenario_path, str) and scenario_path else str(run.get("scenario_id") or "unknown")


def _run_suite_harness_replay_self_contained(lineage: dict[str, Any]) -> bool:
    replay = lineage.get("replay") if isinstance(lineage.get("replay"), dict) else {}
    return replay.get("self_contained") is True


def _harness_relative_path(base_dir: Path, path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base_dir.resolve())).as_posix()


def _safe_run_id(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._-")
    return cleaned or "scenario"


def _run_suite_summary(
    *,
    scenarios_dir: Path,
    out_dir: Path,
    summary_path: Path,
    runs: list[dict[str, Any]],
    errors: list[dict[str, str]],
    artifacts: dict[str, str],
    preserve_paths: bool,
    training_manifest: dict[str, Any] | None,
    validation_summary: dict[str, Any] | None,
    metadata: dict[str, str] | None,
) -> dict[str, Any]:
    passed = sum(1 for run in runs if run["passed"])
    failed = len(runs) - passed
    summary: dict[str, Any] = {
        "schema_version": RUN_SUITE_SCHEMA_VERSION,
        "scenarios_dir": _display_path_for_output_source(scenarios_dir, summary_path, preserve_paths),
        "out_dir": _display_path_for_output_source(out_dir, summary_path, preserve_paths),
        "total": len(runs),
        "passed": passed,
        "failed": failed,
        "error_count": len(errors),
        "errors": errors,
        "metrics": _suite_metrics(runs),
        "runs": runs,
        "artifacts": artifacts,
    }
    if metadata:
        summary["metadata"] = dict(sorted(metadata.items()))
    if training_manifest is not None:
        summary["training_export"] = {
            "episode_count": training_manifest.get("episode_count"),
            "reward_count": training_manifest.get("reward_count"),
            "step_reward_count": training_manifest.get("step_reward_count"),
            "preference_count": training_manifest.get("preference_count"),
            "failure_mode_count": training_manifest.get("failure_mode_count"),
            "sft_count": training_manifest.get("sft_count"),
            "dpo_count": training_manifest.get("dpo_count"),
            "reward_model_count": training_manifest.get("reward_model_count"),
            "quality_flag_count": training_manifest.get("quality_flag_count"),
        }
    if validation_summary is not None:
        summary["validation"] = {
            "passed": validation_summary.get("passed"),
            "target_count": validation_summary.get("target_count"),
            "error_count": validation_summary.get("error_count"),
            "warning_count": validation_summary.get("warning_count"),
        }
    return summary


def _run_suite_artifact_fingerprint(field_name: str, path: Path) -> dict[str, Any]:
    return {
        f"{field_name}_sha256": _sha256_file(path),
        f"{field_name}_size_bytes": path.stat().st_size,
    }


def _suite_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [_int_score(run.get("score")) for run in runs]
    passed = sum(1 for run in runs if run.get("passed") is True)
    failed = len(runs) - passed
    return {
        "pass_rate": round(passed / len(runs), 4) if runs else 0.0,
        "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "failed_rule_counts": _count_values(
            str(rule_id)
            for run in runs
            for rule_id in run.get("failed_rules", [])
        ),
        "critical_failure_counts": _count_values(
            str(rule_id)
            for run in runs
            for rule_id in run.get("critical_failures", [])
        ),
        "task_families": _task_family_metrics(runs),
        "failed": failed,
        "passed": passed,
    }


def _task_family_metrics(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        buckets.setdefault(str(run.get("task_family") or _task_family(str(run.get("scenario_id") or ""))), []).append(run)

    metrics: list[dict[str, Any]] = []
    for family, family_runs in sorted(buckets.items()):
        scores = [_int_score(run.get("score")) for run in family_runs]
        passed = sum(1 for run in family_runs if run.get("passed") is True)
        metrics.append(
            {
                "task_family": family,
                "total": len(family_runs),
                "passed": passed,
                "failed": len(family_runs) - passed,
                "pass_rate": round(passed / len(family_runs), 4) if family_runs else 0.0,
                "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
                "failed_rule_counts": _count_values(
                    str(rule_id)
                    for run in family_runs
                    for rule_id in run.get("failed_rules", [])
                ),
                "critical_failure_counts": _count_values(
                    str(rule_id)
                    for run in family_runs
                    for rule_id in run.get("critical_failures", [])
                ),
            }
        )
    return metrics


def _count_values(values: Any) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return [
        {"id": key, "count": count}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _failed_rule_ids(scorecard: dict[str, Any]) -> list[str]:
    return [
        str(rule.get("id"))
        for rule in scorecard.get("rules", [])
        if isinstance(rule, dict) and rule.get("id") and not rule.get("passed")
    ]


def _lineage_input_hash(lineage: dict[str, Any], name: str) -> str | None:
    for record in lineage.get("inputs", []):
        if isinstance(record, dict) and record.get("name") == name and isinstance(record.get("sha256"), str):
            return record["sha256"]
    return None


def _default_replay_base_dir(lineage_path: Path, lineage: dict[str, Any]) -> Path:
    if isinstance(lineage.get("portable_replay_bundle"), dict):
        return lineage_path.parent
    return Path.cwd()


def _replay_flag_path(argv: list[str], flag: str, base_dir: Path, *, required: bool = True) -> Path | None:
    if flag not in argv:
        if required:
            raise ReplayError(f"artifact_lineage.replay.argv missing {flag}")
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv) or not argv[index + 1]:
        raise ReplayError(f"artifact_lineage.replay.argv missing value for {flag}")
    raw = argv[index + 1]
    if raw.startswith("<redacted:") or raw.startswith("<missing-"):
        raise ReplayError(f"artifact_lineage.replay.argv contains non-replayable path for {flag}: {raw}")
    path = Path(raw)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _verify_replay_input(name: str, path: Path, fingerprints: dict[str, Any]) -> None:
    record = fingerprints.get(name)
    if not isinstance(record, dict):
        raise ReplayError(f"artifact_lineage.replay.input_fingerprints missing {name}")
    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ReplayError(f"artifact_lineage.replay.input_fingerprints.{name}.sha256 is missing")
    if not path.exists() or not path.is_file():
        raise ReplayError(f"replay input {name} not found: {path}")
    expected_size = record.get("size_bytes")
    if expected_size is not None:
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
            raise ReplayError(f"artifact_lineage.replay.input_fingerprints.{name}.size_bytes is invalid")
        if path.stat().st_size != expected_size:
            raise ReplayError(f"replay input {name} size mismatch: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise ReplayError(f"replay input {name} sha256 mismatch: {path}")
