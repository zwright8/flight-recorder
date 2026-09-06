"""CLI commands for the basic domain."""

from __future__ import annotations

from pathlib import Path
from ..digest import RunDigestError, build_run_digest, render_run_digest_markdown
from ..state_validators import StateValidatorError, build_monitor_catalog, build_state_validator_assertions, render_monitor_catalog_markdown
import argparse
from ..state_diff import StateDiffError, build_state_diff
from ..state_capture import StateCaptureError, capture_state_snapshot
from ..verifiers import VerifierError, capture_verified_state
import json
from ..schema import ScenarioError, load_scenario, resolve_trace_path
from ..state import StateSnapshotError, load_state_snapshot, resolve_before_state_snapshot_path, resolve_state_snapshot_path, sanitize_state_snapshot
from ..adapters import AdapterError, normalize_trace
from ..redaction import sanitize_trace
from ..scorers import score_trace
from ..report import write_index, write_report
from .constants import TRACE_FORMAT_CHOICES
from .run_core import _default_digest_out, _load_digest_inputs
from .shared import _key_path_arg, _non_negative_int_arg, _read_json, _state_set_arg, _write_json, _write_score_outputs


def cmd_normalize(args: argparse.Namespace) -> int:
    trace = normalize_trace(args.trace, args.format)
    if not args.no_redact:
        trace = sanitize_trace(trace, args.secret_pattern)
    _write_json(Path(args.out), trace)
    print(f"wrote {args.out}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    trace = _read_json(Path(args.trace))
    state_path = resolve_state_snapshot_path(scenario, args.state)
    before_state_path = resolve_before_state_snapshot_path(scenario, args.before_state)
    state_snapshot = load_state_snapshot(state_path) if state_path is not None else None
    before_state_snapshot = load_state_snapshot(before_state_path) if before_state_path is not None else None
    scorecard = score_trace(scenario, trace, state_snapshot, before_state_snapshot)
    _write_json(Path(args.out), scorecard)
    _write_score_outputs(scorecard, args)
    print(f"wrote {args.out}")
    return 0 if scorecard["passed"] else 1


def cmd_report(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    trace = _read_json(Path(args.trace))
    scorecard = _read_json(Path(args.score))
    state_diff = _read_json(Path(args.state_diff)) if args.state_diff else None
    write_report(scenario, trace, scorecard, args.out, state_diff=state_diff)
    print(f"wrote {args.out}")
    return 0


def cmd_digest(args: argparse.Namespace) -> int:
    scenario, trace, scorecard, state_diff = _load_digest_inputs(args)
    digest = build_run_digest(scenario, trace, scorecard, state_diff=state_diff)
    out_path = Path(args.out) if args.out else _default_digest_out(args)
    if out_path is None:
        raise RunDigestError("--out is required when --run is not supplied")
    _write_json(out_path, digest)
    if args.markdown_out:
        markdown_path = Path(args.markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_run_digest_markdown(digest), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


def cmd_capture_state(args: argparse.Namespace) -> int:
    snapshot = capture_state_snapshot(
        files=args.file,
        directories=args.directory,
        json_sources=args.json_source,
        observations=args.observation,
        include_file_text=args.include_file_text,
        max_text_chars=args.max_text_chars,
        max_dir_entries=args.max_dir_entries,
        preserve_paths=args.preserve_paths,
        secret_patterns=args.secret_pattern,
    )
    _write_json(Path(args.out), snapshot)
    print(f"wrote {args.out}")
    return 0


def cmd_verify_state(args: argparse.Namespace) -> int:
    snapshot = capture_verified_state(
        args.config,
        preserve_paths=args.preserve_paths,
        secret_patterns=args.secret_pattern,
    )
    _write_json(Path(args.out), snapshot)
    print(f"wrote {args.out}")
    return 0


def cmd_state_validators(args: argparse.Namespace) -> int:
    if args.list:
        catalog = build_monitor_catalog()
        if args.out:
            _write_json(Path(args.out), catalog)
            print(f"wrote {args.out}")
        else:
            print(json.dumps(catalog, indent=2, sort_keys=True))
        if args.markdown_out:
            markdown_path = Path(args.markdown_out)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(render_monitor_catalog_markdown(catalog), encoding="utf-8")
            print(f"wrote {args.markdown_out}")
        return 0

    if not args.config:
        raise StateValidatorError("state-validators requires --list or --config")
    compiled = build_state_validator_assertions(args.config)
    if args.out:
        _write_json(Path(args.out), compiled)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(compiled, indent=2, sort_keys=True))
    return 0


def cmd_diff_state(args: argparse.Namespace) -> int:
    before = sanitize_state_snapshot(load_state_snapshot(args.before), args.secret_pattern)
    after = sanitize_state_snapshot(load_state_snapshot(args.after), args.secret_pattern)
    diff = build_state_diff(before, after, max_changes=args.max_changes)
    _write_json(Path(args.out), diff)
    print(f"wrote {args.out}")
    return 0


def register_basic_1(subparsers: argparse._SubParsersAction) -> None:
    normalize = subparsers.add_parser("normalize", help="Normalize a Hermes trace artifact")
    normalize.add_argument("--trace", required=True)
    normalize.add_argument("--format", default="auto", choices=TRACE_FORMAT_CHOICES)
    normalize.add_argument("--out", required=True)
    normalize.add_argument("--secret-pattern", action="append", default=[], help="Additional regex to redact from normalized output")
    normalize.add_argument("--no-redact", action="store_true", help="Write raw normalized trace without redaction")
    normalize.set_defaults(func=cmd_normalize)

    score = subparsers.add_parser("score", help="Score a normalized trace against a scenario")
    score.add_argument("--scenario", required=True)
    score.add_argument("--trace", required=True)
    score.add_argument("--state", help="Optional JSON state snapshot for required_state assertions")
    score.add_argument("--before-state", help="Optional JSON pre-run state snapshot for required_state_transitions assertions")
    score.add_argument("--out", required=True)
    score.add_argument("--junit-out", help="Also write a JUnit XML score report")
    score.add_argument("--markdown-out", help="Also write a Markdown score summary")
    score.set_defaults(func=cmd_score)

    report = subparsers.add_parser("report", help="Render a static HTML report")
    report.add_argument("--scenario", required=True)
    report.add_argument("--trace", required=True)
    report.add_argument("--score", required=True)
    report.add_argument("--state-diff", help="Optional hfr.state_diff.v1 JSON to render as a state-change table")
    report.add_argument("--out", required=True)
    report.set_defaults(func=cmd_report)

    digest = subparsers.add_parser("digest", help="Write a compact per-run evidence digest")
    digest.add_argument("--run", help="Existing run directory containing normalized_trace.json and scorecard.json")
    digest.add_argument("--scenario", help="Scenario JSON; optional with --run when lineage/scorecard metadata is enough")
    digest.add_argument("--trace", help="Normalized trace JSON; defaults to <run>/normalized_trace.json with --run")
    digest.add_argument("--score", help="Scorecard JSON; defaults to <run>/scorecard.json with --run")
    digest.add_argument("--state-diff", help="Optional hfr.state_diff.v1 JSON; defaults to <run>/state_diff.json if present")
    digest.add_argument("--out", help="Digest JSON output path; defaults to <run>/run_digest.json with --run")
    digest.add_argument("--markdown-out", help="Optional Markdown digest output path")
    digest.set_defaults(func=cmd_digest)

    capture_state = subparsers.add_parser(
        "capture-state",
        help="Capture a JSON state snapshot from local evidence sources",
    )
    capture_state.add_argument("--out", required=True, help="State snapshot JSON output path")
    capture_state.add_argument(
        "--file",
        action="append",
        default=[],
        type=_key_path_arg,
        metavar="KEY=PATH",
        help="Capture file existence, size, and sha256; may be repeated",
    )
    capture_state.add_argument(
        "--dir",
        dest="directory",
        action="append",
        default=[],
        type=_key_path_arg,
        metavar="KEY=PATH",
        help="Capture a directory listing; may be repeated",
    )
    capture_state.add_argument(
        "--json",
        dest="json_source",
        action="append",
        default=[],
        type=_key_path_arg,
        metavar="KEY=PATH",
        help="Import a JSON file under json.KEY; may be repeated",
    )
    capture_state.add_argument(
        "--set",
        dest="observation",
        action="append",
        default=[],
        type=_state_set_arg,
        metavar="PATH=VALUE",
        help="Set an observed value under observations using a dot path; VALUE may be JSON",
    )
    capture_state.add_argument("--include-file-text", action="store_true", help="Include UTF-8 file text for --file sources")
    capture_state.add_argument(
        "--max-text-chars",
        type=_non_negative_int_arg,
        default=4096,
        help="Maximum text characters captured per --file when --include-file-text is set",
    )
    capture_state.add_argument(
        "--max-dir-entries",
        type=_non_negative_int_arg,
        default=200,
        help="Maximum direct entries captured per --dir",
    )
    capture_state.add_argument("--secret-pattern", action="append", default=[], help="Regex pattern to redact from captured state")
    capture_state.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in the snapshot")
    capture_state.set_defaults(func=cmd_capture_state)

    verify_state = subparsers.add_parser(
        "verify-state",
        help="Capture a JSON state snapshot from read-only external verifier adapters",
    )
    verify_state.add_argument("--config", required=True, help="Verifier config JSON path")
    verify_state.add_argument("--out", required=True, help="State snapshot JSON output path")
    verify_state.add_argument("--secret-pattern", action="append", default=[], help="Regex pattern to redact from captured state")
    verify_state.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in the snapshot")
    verify_state.set_defaults(func=cmd_verify_state)

    state_validators = subparsers.add_parser(
        "state-validators",
        help="List external monitor targets or compile state-validator configs into scenario assertions",
    )
    state_validators.add_argument("--list", action="store_true", help="List monitorable external tool/state areas")
    state_validators.add_argument("--config", help="State-validator config JSON path")
    state_validators.add_argument("--out", help="JSON output path; prints to stdout when omitted")
    state_validators.add_argument("--markdown-out", help="Optional Markdown monitor catalog output path with --list")
    state_validators.set_defaults(func=cmd_state_validators)

    diff_state = subparsers.add_parser(
        "diff-state",
        help="Write a deterministic hfr.state_diff.v1 artifact from before/after state snapshots",
    )
    diff_state.add_argument("--before", required=True, help="Pre-run state snapshot JSON")
    diff_state.add_argument("--after", required=True, help="Post-run state snapshot JSON")
    diff_state.add_argument("--out", required=True)
    diff_state.add_argument(
        "--max-changes",
        type=_non_negative_int_arg,
        default=200,
        help="Maximum number of changed paths to include while still counting all changes",
    )
    diff_state.add_argument("--secret-pattern", action="append", default=[], help="Additional regex to redact from the diff")
    diff_state.set_defaults(func=cmd_diff_state)

