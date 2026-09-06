"""CLI commands for the run commands domain."""

from __future__ import annotations

from typing import Any, Iterator
from dataclasses import dataclass
from contextlib import contextmanager
from ..validation import EVAL_SUITE_MANIFEST_SCHEMA_VERSION, VALIDATION_SCHEMA_VERSION, validate_artifacts, validate_trainer_preflight
from pathlib import Path
from ..lineage import LINEAGE_SCHEMA_VERSION, REPLAY_BUNDLE_SCHEMA_VERSION, write_run_lineage
from ..schema import ScenarioError, load_scenario, resolve_trace_path
from ..schema_registry import SchemaRegistryError, check_schema_file, check_schema_jsonl_file, list_schema_records, load_schema, write_schema_bundle
import argparse
import json
from ..path_safety import assert_output_does_not_alias_sources, assert_output_outside_source_directories, locked_owned_output_directory, output_directory_lock_is_held, path_has_symlink_component, remove_directory_tree_if_identity
from .constants import TRACE_FORMAT_CHOICES
from .replay_core import ReplayError, _copy_replay_input, _portable_replay_lineage, _replay_bundle_manifest, _trace_bundle_name
from .run_core import _run_scenario_artifacts
from .shared import _read_json, _write_json
from .suite_core import _default_replay_base_dir, _replay_flag_path, _verify_replay_input


@dataclass(frozen=True, slots=True)
class _ReplayInputs:
    lineage_path: Path
    lineage: dict[str, Any]
    scenario_path: Path
    trace_path: Path
    state_path: Path | None
    before_state_path: Path | None


def _load_replay_inputs(args: argparse.Namespace, *, enforce_self_contained: bool) -> _ReplayInputs:
    lineage_path = Path(args.lineage)
    lineage = _read_json(lineage_path)
    replay = lineage.get("replay")
    if not isinstance(replay, dict):
        raise ReplayError("artifact_lineage.replay is missing; rerun the original run to emit replay metadata")
    if enforce_self_contained and replay.get("self_contained") is not True:
        raise ReplayError("replay contract is not self-contained; restore paths or pass --allow-non-self-contained")
    argv = replay.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ReplayError("artifact_lineage.replay.argv must be a list of strings")

    base_dir = Path(args.base_dir) if args.base_dir else _default_replay_base_dir(lineage_path, lineage)
    scenario_path = _replay_flag_path(argv, "--scenario", base_dir)
    trace_path = _replay_flag_path(argv, "--trace", base_dir)
    state_path = _replay_flag_path(argv, "--state", base_dir, required=False)
    before_state_path = _replay_flag_path(argv, "--before-state", base_dir, required=False)
    fingerprints = replay.get("input_fingerprints") if isinstance(replay.get("input_fingerprints"), dict) else {}
    _verify_replay_input("scenario", scenario_path, fingerprints)
    _verify_replay_input("source_trace", trace_path, fingerprints)
    if before_state_path is not None:
        _verify_replay_input("source_before_state_snapshot", before_state_path, fingerprints)
    if state_path is not None:
        _verify_replay_input("source_state_snapshot", state_path, fingerprints)
    return _ReplayInputs(
        lineage_path=lineage_path,
        lineage=lineage,
        scenario_path=scenario_path,
        trace_path=trace_path,
        state_path=state_path,
        before_state_path=before_state_path,
    )


def cmd_run(args: argparse.Namespace) -> int:
    result = _run_scenario_artifacts(
        args.scenario,
        args.out,
        trace_override=args.trace,
        state_override=args.state,
        before_state_override=args.before_state,
        trace_format=args.format,
        write_sensitive_trace=args.write_sensitive_trace,
        preserve_paths=args.preserve_paths,
        junit_out=args.junit_out,
        markdown_out=args.markdown_out,
    )
    scorecard = result["scorecard"]
    scenario = result["scenario"]
    report_path = result["paths"]["report"]
    print(f"{'PASS' if scorecard['passed'] else 'FAIL'} {scenario['id']} score={scorecard['score']} report={report_path}")
    return 1 if args.fail_on_score and not scorecard["passed"] else 0


def cmd_replay(args: argparse.Namespace) -> int:
    inputs = _load_replay_inputs(
        args,
        enforce_self_contained=not args.allow_non_self_contained,
    )

    result = _run_scenario_artifacts(
        inputs.scenario_path,
        args.out,
        trace_override=inputs.trace_path,
        state_override=inputs.state_path,
        before_state_override=inputs.before_state_path,
        trace_format=args.format,
        write_sensitive_trace=args.write_sensitive_trace,
        preserve_paths=args.preserve_paths,
    )
    scorecard = result["scorecard"]
    scenario = result["scenario"]
    print(f"{'PASS' if scorecard['passed'] else 'FAIL'} replay {scenario['id']} score={scorecard['score']} out={args.out}")
    return 1 if args.fail_on_score and not scorecard["passed"] else 0


def cmd_replay_bundle(args: argparse.Namespace) -> int:
    inputs = _load_replay_inputs(args, enforce_self_contained=False)

    out_dir = Path(args.out)
    with _owned_output_directory_lock(
        out_dir,
        force=bool(args.force),
        label="replay bundle",
        manifest_name="replay_bundle.json",
        schema_version=REPLAY_BUNDLE_SCHEMA_VERSION,
        error_type=ReplayError,
        replay_bundle=True,
    ):
        return _build_replay_bundle_locked(
            args,
            lineage=inputs.lineage,
            lineage_path=inputs.lineage_path,
            scenario_path=inputs.scenario_path,
            trace_path=inputs.trace_path,
            state_path=inputs.state_path,
            before_state_path=inputs.before_state_path,
            out_dir=out_dir,
        )


def _build_replay_bundle_locked(
    args: argparse.Namespace,
    *,
    lineage: dict[str, Any],
    lineage_path: Path,
    scenario_path: Path,
    trace_path: Path,
    state_path: Path | None,
    before_state_path: Path | None,
    out_dir: Path,
) -> int:
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    copied_inputs = {
        "scenario": _copy_replay_input(scenario_path, inputs_dir / "scenario.json"),
        "source_trace": _copy_replay_input(trace_path, inputs_dir / _trace_bundle_name(trace_path)),
    }
    if state_path is not None:
        copied_inputs["source_state_snapshot"] = _copy_replay_input(state_path, inputs_dir / "source_state_snapshot.json")
    if before_state_path is not None:
        copied_inputs["source_before_state_snapshot"] = _copy_replay_input(
            before_state_path,
            inputs_dir / "source_before_state_snapshot.json",
        )

    bundle_lineage = _portable_replay_lineage(
        lineage=lineage,
        source_lineage_path=lineage_path,
        copied_inputs=copied_inputs,
        preserve_paths=args.preserve_paths,
    )
    bundle_lineage_path = out_dir / "artifact_lineage.json"
    _write_json(bundle_lineage_path, bundle_lineage)

    manifest = _replay_bundle_manifest(
        bundle_lineage=bundle_lineage,
        bundle_lineage_path=bundle_lineage_path,
        source_lineage_path=lineage_path,
        copied_inputs=copied_inputs,
        preserve_paths=args.preserve_paths,
    )
    manifest_path = out_dir / "replay_bundle.json"
    _write_json(manifest_path, manifest)
    print(f"wrote replay bundle {out_dir}")
    return 0


@contextmanager
def _owned_output_directory_lock(
    target: Path,
    *,
    force: bool,
    label: str,
    manifest_name: str,
    schema_version: str,
    error_type: type[ValueError],
    schema_name: str | None = None,
    replay_bundle: bool = False,
) -> Iterator[None]:
    try:
        with locked_owned_output_directory(
            target,
            repo_root=Path(__file__).resolve().parents[2],
            force=force,
            label=f"{label} output",
            is_owned=lambda path: _owned_output_manifest_is_valid(
                path,
                manifest_name=manifest_name,
                schema_version=schema_version,
                schema_name=schema_name,
                replay_bundle=replay_bundle,
            ),
        ):
            yield
    except ValueError as exc:
        raise error_type(str(exc)) from exc


def _owned_output_manifest_is_valid(
    target: Path,
    *,
    manifest_name: str,
    schema_version: str,
    schema_name: str | None,
    replay_bundle: bool,
) -> bool:
    manifest_path = target / manifest_name
    if not manifest_path.is_file() or path_has_symlink_component(manifest_path, include_leaf=True):
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or payload.get("schema_version") != schema_version:
        return False
    if schema_name is not None:
        try:
            return check_schema_file(manifest_path, schema_name).get("passed") is True
        except (OSError, UnicodeError, json.JSONDecodeError, SchemaRegistryError):
            return False
    if replay_bundle:
        return validate_artifacts(replay_bundle_paths=[target], strict=False).get("passed") is True
    return True


def _scenario_paths_from_suite_manifest(scenario_paths: list[Path], manifest_path: Path) -> list[Path]:
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != EVAL_SUITE_MANIFEST_SCHEMA_VERSION:
        raise ScenarioError(
            f"suite manifest schema_version must be {EVAL_SUITE_MANIFEST_SCHEMA_VERSION!r}; got {manifest.get('schema_version')!r}"
        )
    scenario_ids = manifest.get("scenario_ids")
    if not isinstance(scenario_ids, list) or not scenario_ids or not all(isinstance(item, str) and item for item in scenario_ids):
        raise ScenarioError("suite manifest scenario_ids must be a non-empty list of strings")
    duplicates = sorted({scenario_id for scenario_id in scenario_ids if scenario_ids.count(scenario_id) > 1})
    if duplicates:
        raise ScenarioError(f"suite manifest has duplicate scenario_ids: {', '.join(duplicates)}")

    by_id: dict[str, Path] = {}
    for scenario_path in scenario_paths:
        scenario = load_scenario(scenario_path)
        scenario_id = str(scenario["id"])
        if scenario_id in by_id:
            raise ScenarioError(f"Duplicate discovered scenario id {scenario_id!r}: {scenario_path} conflicts with {by_id[scenario_id]}")
        by_id[scenario_id] = scenario_path
    missing = [scenario_id for scenario_id in scenario_ids if scenario_id not in by_id]
    if missing:
        raise ScenarioError(f"suite manifest references missing scenario_ids: {', '.join(missing)}")
    return [by_id[scenario_id] for scenario_id in scenario_ids]


def register_run_commands_1(subparsers: argparse._SubParsersAction) -> None:
    run = subparsers.add_parser("run", help="Normalize, score, and report in one command")
    run.add_argument("--scenario", required=True)
    run.add_argument("--trace")
    run.add_argument("--state", help="Optional JSON state snapshot for required_state assertions")
    run.add_argument("--before-state", help="Optional JSON pre-run state snapshot for required_state_transitions assertions")
    run.add_argument("--format", default="auto", choices=TRACE_FORMAT_CHOICES)
    run.add_argument("--out", required=True)
    run.add_argument("--write-sensitive-trace", action="store_true", help="Also write raw_trace.sensitive.json with unredacted evidence")
    run.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in generated reports and regression files")
    run.add_argument("--junit-out", help="Also write a JUnit XML score report")
    run.add_argument("--markdown-out", help="Also write a Markdown score summary")
    run.add_argument("--fail-on-score", action="store_true", help="Exit nonzero when the scenario score fails")
    run.set_defaults(func=cmd_run)

    replay = subparsers.add_parser("replay", help="Rerun a scenario from artifact lineage replay metadata")
    replay.add_argument("--lineage", required=True, help="Path to artifact_lineage.json with replay metadata")
    replay.add_argument("--out", required=True, help="Output directory for replayed run artifacts")
    replay.add_argument("--base-dir", help="Base directory for relative replay paths; defaults to current directory")
    replay.add_argument("--format", default="auto", choices=TRACE_FORMAT_CHOICES)
    replay.add_argument("--write-sensitive-trace", action="store_true", help="Also write raw_trace.sensitive.json with unredacted evidence")
    replay.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in generated replay artifacts")
    replay.add_argument("--allow-non-self-contained", action="store_true", help="Attempt replay even when replay.self_contained is false")
    replay.add_argument("--fail-on-score", action="store_true", help="Exit nonzero when the replayed score fails")
    replay.set_defaults(func=cmd_replay)

    replay_bundle = subparsers.add_parser("replay-bundle", help="Create a portable replay bundle from artifact lineage")
    replay_bundle.add_argument("--lineage", required=True, help="Path to source artifact_lineage.json")
    replay_bundle.add_argument("--out", required=True, help="Output directory for the portable replay bundle")
    replay_bundle.add_argument("--base-dir", help="Base directory for relative source replay paths; defaults to current directory")
    replay_bundle.add_argument("--force", action="store_true", help="Replace an existing non-empty bundle directory")
    replay_bundle.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in generated bundle metadata")
    replay_bundle.set_defaults(func=cmd_replay_bundle)
