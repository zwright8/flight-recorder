"""CLI commands for the suite domain."""

from __future__ import annotations

from ..adapters import AdapterError, normalize_trace
from typing import Any, Iterator
from ..artifacts import ArtifactError, build_suite_trend, compare_scorecards, compare_suites, write_compare_report, write_junit, write_markdown_summary, write_suite_compare_report, write_suite_trend_report
from pathlib import Path
from ..schema import ScenarioError, load_scenario, resolve_trace_path
from ..training import TrainingExportError, export_compare_rl_dataset, export_rl_dataset
import argparse
from ..bundle import HARNESS_RUN_MANIFEST_SCHEMA_VERSION, HARNESS_RUN_RESULT_SCHEMA_VERSION, EvidenceBundleError, build_evidence_bundle
from ..evidence import EvidenceCoverageError, build_evidence_coverage
from ..repair import RepairQueueError, build_repair_queue
from ..scenario_quality import build_scenario_quality
from ..trace_observability import TraceObservabilityError, build_trace_observability
from datetime import datetime, timezone
from ..scenario_check import check_scenarios, discover_scenarios
import json
from ..validation import EVAL_SUITE_MANIFEST_SCHEMA_VERSION, VALIDATION_SCHEMA_VERSION, validate_artifacts, validate_trainer_preflight
from ..report import write_index, write_report
from .constants import GOAL3_HANDOFF_SCHEMA_VERSION, TRACE_FORMAT_CHOICES
from .gates import cmd_gate_export, cmd_trainer_preflight
from .replay_core import _task_family
from .run_commands import _owned_output_directory_lock, _scenario_paths_from_suite_manifest
from .run_core import _run_scenario_artifacts
from .shared import _display_path, _display_path_for_output_source, _goal3_training_gate_args, _metadata_arg, _metadata_options, _read_json, _write_json
from .suite_core import _failed_rule_ids, _lineage_input_hash, _run_suite_artifact_fingerprint, _run_suite_summary, _safe_run_id, _write_run_suite_harness_handoff


def cmd_run_suite(args: argparse.Namespace) -> int:
    scenario_paths = discover_scenarios(Path(args.scenarios), args.pattern, args.recursive)
    if args.suite_manifest:
        scenario_paths = _scenario_paths_from_suite_manifest(scenario_paths, Path(args.suite_manifest))
    out_dir = Path(args.out)
    summary_path = Path(args.summary_out) if args.summary_out else out_dir / "suite_summary.json"

    def summary_ref(path: Path) -> str:
        return _display_path_for_output_source(path, summary_path, args.preserve_paths)

    metadata = _metadata_options(args.metadata)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_run_ids: dict[str, Path] = {}
    for scenario_path in scenario_paths:
        try:
            scenario = load_scenario(scenario_path)
            run_id = _safe_run_id(str(scenario["id"]))
            if run_id in seen_run_ids:
                raise ScenarioError(
                    f"Duplicate scenario id/run directory {scenario['id']!r}: "
                    f"{scenario_path} conflicts with {seen_run_ids[run_id]}"
                )
            seen_run_ids[run_id] = scenario_path
            run_dir = out_dir / run_id
            result = _run_scenario_artifacts(
                scenario_path,
                run_dir,
                trace_format=args.format,
                write_sensitive_trace=args.write_sensitive_trace,
                preserve_paths=args.preserve_paths,
                junit_out=run_dir / "scorecard.junit.xml" if args.junit else None,
                markdown_out=run_dir / "scorecard.md" if args.markdown else None,
            )
            scorecard = result["scorecard"]
            runs.append(
                {
                    "scenario_id": result["scenario"]["id"],
                    "scenario_title": result["scenario"].get("title", result["scenario"]["id"]),
                    "task_family": _task_family(str(result["scenario"]["id"])),
                    "scenario_path": summary_ref(scenario_path),
                    "scenario_sha256": _lineage_input_hash(result["lineage"], "scenario"),
                    "trace_path": summary_ref(result["trace_path"]),
                    "trace_sha256": _lineage_input_hash(result["lineage"], "source_trace"),
                    "before_state_path": (
                        summary_ref(result["before_state_path"])
                        if result.get("before_state_path")
                        else None
                    ),
                    "before_state_sha256": _lineage_input_hash(result["lineage"], "source_before_state_snapshot"),
                    "state_path": summary_ref(result["state_path"]) if result.get("state_path") else None,
                    "state_sha256": _lineage_input_hash(result["lineage"], "source_state_snapshot"),
                    "run_dir": summary_ref(run_dir),
                    "report": summary_ref(result["paths"]["report"]),
                    **_run_suite_artifact_fingerprint("report", result["paths"]["report"]),
                    "scorecard": summary_ref(result["paths"]["scorecard"]),
                    **_run_suite_artifact_fingerprint("scorecard", result["paths"]["scorecard"]),
                    "run_digest": summary_ref(result["paths"]["run_digest"]),
                    **_run_suite_artifact_fingerprint("run_digest", result["paths"]["run_digest"]),
                    "lineage": summary_ref(result["paths"]["lineage"]),
                    **_run_suite_artifact_fingerprint("lineage", result["paths"]["lineage"]),
                    "passed": bool(scorecard["passed"]),
                    "score": scorecard["score"],
                    "failed_rules": _failed_rule_ids(scorecard),
                    "critical_failures": scorecard.get("critical_failures", []),
                }
            )
            print(
                f"{'PASS' if scorecard['passed'] else 'FAIL'} "
                f"{result['scenario']['id']} score={scorecard['score']} report={result['paths']['report']}"
            )
        except (AdapterError, ScenarioError, TrainingExportError, OSError, json.JSONDecodeError) as exc:
            errors.append({"scenario_path": summary_ref(scenario_path), "error": str(exc)})
            print(f"ERROR {scenario_path}: {exc}")

    artifacts: dict[str, str] = {}
    index_path = Path(args.index_out) if args.index_out else out_dir / "index.html"
    if not args.no_index:
        completed_run_dirs = [out_dir / _safe_run_id(str(run["scenario_id"])) for run in runs]
        write_index(completed_run_dirs, index_path, artifacts_dir=out_dir)
        artifacts["index"] = summary_ref(index_path)

    training_manifest: dict[str, Any] | None = None
    training_out = Path(args.training_export_out) if args.training_export_out else out_dir / "training_export"
    if args.export_rl:
        if runs:
            training_manifest = export_rl_dataset(
                out_dir,
                training_out,
                reward_scale=args.reward_scale,
                min_score_gap=args.min_score_gap,
                max_pairs_per_family=args.max_pairs_per_family,
                preserve_paths=args.preserve_paths,
                metadata=metadata,
            )
            artifacts["training_export"] = summary_ref(training_out)
        else:
            errors.append(
                {
                    "scenario_path": summary_ref(Path(args.scenarios)),
                    "error": "Cannot export RL artifacts because no scenario runs completed.",
                }
            )

    validation_path = Path(args.validation_out) if args.validation_out else out_dir / "validation.json"
    handoff_paths: dict[str, Path] = {}
    handoff_bundle: dict[str, Any] | None = None
    if args.evidence_handoff:
        if runs:
            handoff_paths = {
                "scenario_quality": out_dir / "scenario_quality.json",
                "evidence_coverage": out_dir / "evidence_coverage.json",
                "trace_observability": out_dir / "trace_observability.json",
                "repair_queue": out_dir / "repair_queue.json",
                "evidence_bundle": out_dir / "evidence_bundle.json",
            }
            scenario_quality = build_scenario_quality(
                Path(args.scenarios),
                pattern=args.pattern,
                recursive=args.recursive,
                require_traces=True,
                preserve_paths=args.preserve_paths,
            )
            _write_json(handoff_paths["scenario_quality"], scenario_quality)
            artifacts["scenario_quality"] = summary_ref(handoff_paths["scenario_quality"])

            evidence_coverage = build_evidence_coverage(out_dir, preserve_paths=args.preserve_paths)
            _write_json(handoff_paths["evidence_coverage"], evidence_coverage)
            artifacts["evidence_coverage"] = summary_ref(handoff_paths["evidence_coverage"])

            trace_observability = build_trace_observability(out_dir, preserve_paths=args.preserve_paths)
            _write_json(handoff_paths["trace_observability"], trace_observability)
            artifacts["trace_observability"] = summary_ref(handoff_paths["trace_observability"])

            repair_queue = build_repair_queue(out_dir, preserve_paths=args.preserve_paths, output_path=handoff_paths["repair_queue"])
            _write_json(handoff_paths["repair_queue"], repair_queue)
            artifacts["repair_queue"] = summary_ref(handoff_paths["repair_queue"])

            harness_paths = _write_run_suite_harness_handoff(out_dir, runs, summary_path=summary_path)
            if harness_paths:
                handoff_paths.update(harness_paths)
                artifacts["harness_manifest"] = summary_ref(harness_paths["harness_manifest"])
                artifacts["harness_result"] = summary_ref(harness_paths["harness_result"])
            artifacts["evidence_bundle"] = summary_ref(handoff_paths["evidence_bundle"])
        else:
            errors.append(
                {
                    "scenario_path": summary_ref(Path(args.scenarios)),
                    "error": "Cannot build evidence handoff because no scenario runs completed.",
                }
            )

    validation_summary: dict[str, Any] | None = None
    if args.validate:
        artifacts["validation"] = summary_ref(validation_path)

    summary = _run_suite_summary(
        scenarios_dir=Path(args.scenarios),
        out_dir=out_dir,
        summary_path=summary_path,
        runs=runs,
        errors=errors,
        artifacts=artifacts,
        preserve_paths=args.preserve_paths,
        training_manifest=training_manifest,
        validation_summary=None,
        metadata=metadata,
    )
    _write_json(summary_path, summary)

    if args.validate:
        validation_summary = validate_artifacts(
            runs_dir=out_dir,
            training_export_dir=training_out if args.export_rl else None,
            evidence_coverage_paths=[handoff_paths["evidence_coverage"]] if args.evidence_handoff and handoff_paths else None,
            trace_observability_paths=[handoff_paths["trace_observability"]] if args.evidence_handoff and handoff_paths else None,
            scenario_quality_paths=[handoff_paths["scenario_quality"]] if args.evidence_handoff and handoff_paths else None,
            repair_queue_paths=[handoff_paths["repair_queue"]] if args.evidence_handoff and handoff_paths else None,
            harness_manifest_paths=(
                [handoff_paths["harness_manifest"]]
                if args.evidence_handoff and "harness_manifest" in handoff_paths
                else None
            ),
            harness_result_paths=(
                [handoff_paths["harness_result"]]
                if args.evidence_handoff and "harness_result" in handoff_paths
                else None
            ),
            suite_summary_paths=[summary_path],
            strict=args.strict,
        )
        _write_json(validation_path, validation_summary)

    summary = _run_suite_summary(
        scenarios_dir=Path(args.scenarios),
        out_dir=out_dir,
        summary_path=summary_path,
        runs=runs,
        errors=errors,
        artifacts=artifacts,
        preserve_paths=args.preserve_paths,
        training_manifest=training_manifest,
        validation_summary=validation_summary,
        metadata=metadata,
    )
    _write_json(summary_path, summary)

    if args.evidence_handoff and handoff_paths:
        handoff_bundle = build_evidence_bundle(
            out_path=handoff_paths["evidence_bundle"],
            runs_dir=out_dir,
            suite_summary_path=summary_path,
            scenario_quality_path=handoff_paths["scenario_quality"],
            evidence_coverage_path=handoff_paths["evidence_coverage"],
            trace_observability_path=handoff_paths["trace_observability"],
            repair_queue_path=handoff_paths["repair_queue"],
            validation_path=validation_path if args.validate else None,
            training_export_dir=training_out if args.export_rl else None,
            harness_manifest_paths=(
                [handoff_paths["harness_manifest"]] if "harness_manifest" in handoff_paths else None
            ),
            harness_result_paths=(
                [handoff_paths["harness_result"]] if "harness_result" in handoff_paths else None
            ),
            require_harness=True,
            preserve_paths=args.preserve_paths,
        )
        _write_json(handoff_paths["evidence_bundle"], handoff_bundle)

    if not args.no_index:
        completed_run_dirs = [out_dir / _safe_run_id(str(run["scenario_id"])) for run in runs]
        write_index(completed_run_dirs, index_path, artifacts_dir=out_dir)

    print(
        f"SUITE total={summary['total']} passed={summary['passed']} failed={summary['failed']} "
        f"errors={summary['error_count']} summary={summary_path}"
    )

    if errors:
        return 1
    if args.validate and validation_summary and not validation_summary["passed"]:
        return 1
    if args.evidence_handoff and handoff_bundle and not handoff_bundle["passed"]:
        return 1
    if args.fail_on_failed and summary["failed"] > 0:
        return 1
    return 0


def cmd_goal3_handoff(args: argparse.Namespace) -> int:
    target = Path(args.out)
    with _owned_output_directory_lock(
        target,
        force=bool(args.force),
        label="goal3 handoff",
        manifest_name="goal3_handoff.json",
        schema_version=GOAL3_HANDOFF_SCHEMA_VERSION,
        error_type=ArtifactError,
        schema_name="goal3_handoff",
    ):
        return _build_goal3_handoff_locked(args, target)


def _build_goal3_handoff_locked(args: argparse.Namespace, target: Path) -> int:
    target.mkdir(parents=True, exist_ok=True)

    metadata = _metadata_options(args.metadata)
    runs_dir = target / "runs"
    training_export_dir = target / "training_export"
    suite_summary_path = target / "suite_summary.json"
    validation_path = target / "validation.json"
    index_path = target / "index.html"
    gate_path = target / "training_gate.json"
    preflight_path = target / "trainer_preflight.json"
    handoff_path = target / "goal3_handoff.json"
    evidence_bundle_path = runs_dir / "evidence_bundle.json"

    suite_code = cmd_run_suite(
        argparse.Namespace(
            scenarios=args.scenarios,
            pattern=args.pattern,
            recursive=args.recursive,
            suite_manifest=args.suite_manifest,
            out=str(runs_dir),
            format=args.format,
            summary_out=str(suite_summary_path),
            index_out=str(index_path),
            no_index=False,
            junit=False,
            markdown=False,
            export_rl=True,
            training_export_out=str(training_export_dir),
            reward_scale=args.reward_scale,
            min_score_gap=args.min_score_gap,
            max_pairs_per_family=args.max_pairs_per_family,
            validate=True,
            validation_out=str(validation_path),
            strict=args.strict,
            evidence_handoff=True,
            write_sensitive_trace=False,
            preserve_paths=args.preserve_paths,
            metadata=args.metadata,
            fail_on_failed=False,
        )
    )

    training_manifest = _read_json(training_export_dir / "manifest.json")
    validation_summary = _read_json(validation_path)
    evidence_bundle = _read_json(evidence_bundle_path)
    dataset_version = str(training_manifest.get("dataset_version") or "")

    gate_code = cmd_gate_export(_goal3_training_gate_args(args, training_export_dir, gate_path))
    gate = _read_json(gate_path)

    preflight_code = cmd_trainer_preflight(
        argparse.Namespace(
            out=str(preflight_path),
            gate=[str(gate_path)],
            training_export=str(training_export_dir),
            compare_export=None,
            reviewed_export=None,
            evidence_bundle=str(evidence_bundle_path),
            agentic_training_plan=None,
            validation=[str(validation_path)],
            require_gate=["training_gate"],
            require_dataset_version=[dataset_version] if dataset_version else [],
            trainer_command=args.trainer_command,
            allow_unvalidated_gates=False,
            preserve_paths=args.preserve_paths,
            metadata=args.metadata,
        )
    )
    preflight = _read_json(preflight_path)

    artifacts = {
        "runs": _display_path(runs_dir, args.preserve_paths),
        "suite_summary": _display_path(suite_summary_path, args.preserve_paths),
        "training_export": _display_path(training_export_dir, args.preserve_paths),
        "validation": _display_path(validation_path, args.preserve_paths),
        "evidence_bundle": _display_path(evidence_bundle_path, args.preserve_paths),
        "training_gate": _display_path(gate_path, args.preserve_paths),
        "trainer_preflight": _display_path(preflight_path, args.preserve_paths),
    }
    if args.policy:
        artifacts["training_gate_policy"] = _display_path(Path(args.policy), args.preserve_paths)

    stages = [
        {
            "id": "run_suite",
            "passed": suite_code == 0,
            "artifact": artifacts["suite_summary"],
            "summary": "Scenario suite, optional RL export, validation, and evidence handoff generation.",
        },
        {
            "id": "training_export",
            "passed": bool(dataset_version),
            "artifact": artifacts["training_export"],
            "summary": "Trainer-ready RL dataset export with dataset_version selection key.",
        },
        {
            "id": "validation",
            "passed": validation_summary.get("passed") is True,
            "artifact": artifacts["validation"],
            "summary": "Structural validation over runs, training export, and evidence handoff artifacts.",
        },
        {
            "id": "evidence_bundle",
            "passed": evidence_bundle.get("passed") is True,
            "artifact": artifacts["evidence_bundle"],
            "summary": "Evidence handoff bundle over scenario quality, evidence coverage, observability, repair queue, and harness artifacts.",
        },
        {
            "id": "training_gate",
            "passed": gate.get("passed") is True and gate_code == 0,
            "artifact": artifacts["training_gate"],
            "summary": "Training dataset readiness gate.",
        },
        {
            "id": "trainer_preflight",
            "passed": preflight.get("passed") is True and preflight_code == 0,
            "artifact": artifacts["trainer_preflight"],
            "summary": "Trainer launch guard manifest that records but does not execute the trainer command.",
        },
    ]
    passed = all(stage["passed"] for stage in stages)
    handoff = {
        "schema_version": GOAL3_HANDOFF_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "recommendation": "handoff_ready" if passed else "fix_handoff",
        "output_dir": _display_path(target, args.preserve_paths),
        "dataset_version": dataset_version,
        "metadata": metadata,
        "artifacts": artifacts,
        "stages": stages,
        "notes": [
            "Goal 3 handoff builds export, validation, gate, evidence bundle, and trainer preflight artifacts in one reproducible sequence.",
            "The trainer command is recorded for downstream launch checks; it is not executed by this command.",
        ],
    }
    _write_json(handoff_path, handoff)
    print(
        f"{'READY' if passed else 'BLOCKED'} goal3-handoff "
        f"dataset_version={dataset_version or 'missing'} out={handoff_path}"
    )
    return 0 if passed else 1


def register_suite_1(subparsers: argparse._SubParsersAction) -> None:
    run_suite = subparsers.add_parser("run-suite", help="Run a directory of scenarios into a complete evidence bundle")
    run_suite.add_argument("--scenarios", required=True, help="Directory containing scenario JSON files")
    run_suite.add_argument("--out", required=True, help="Output directory for per-scenario run directories and suite artifacts")
    run_suite.add_argument("--pattern", default="*.json", help="Scenario filename glob relative to --scenarios")
    run_suite.add_argument("--recursive", action="store_true", help="Discover scenarios recursively with --pattern")
    run_suite.add_argument("--suite-manifest", help="Eval suite manifest with an explicit scenario_ids list to run")
    run_suite.add_argument("--format", default="auto", choices=TRACE_FORMAT_CHOICES)
    run_suite.add_argument("--summary-out", help="Suite summary JSON output path; defaults to <out>/suite_summary.json")
    run_suite.add_argument("--index-out", help="Report index output path; defaults to <out>/index.html")
    run_suite.add_argument("--no-index", action="store_true", help="Skip writing the report index")
    run_suite.add_argument("--junit", action="store_true", help="Write scorecard.junit.xml inside each run directory")
    run_suite.add_argument("--markdown", action="store_true", help="Write scorecard.md inside each run directory")
    run_suite.add_argument("--export-rl", action="store_true", help="Also export evidence and trainer-ready artifacts for the completed suite")
    run_suite.add_argument("--training-export-out", help="RL export directory; defaults to <out>/training_export")
    run_suite.add_argument("--reward-scale", default="score", choices=["score", "binary", "signed"])
    run_suite.add_argument("--min-score-gap", type=int, default=1)
    run_suite.add_argument("--max-pairs-per-family", type=int, default=0)
    run_suite.add_argument("--validate", action="store_true", help="Also validate generated run and optional training artifacts")
    run_suite.add_argument("--validation-out", help="Validation JSON output path; defaults to <out>/validation.json")
    run_suite.add_argument("--strict", action="store_true", help="Treat validation warnings as validation failure")
    run_suite.add_argument(
        "--evidence-handoff",
        action="store_true",
        help="Also write scenario quality, evidence coverage, trace observability, harness handoff, and evidence bundle artifacts",
    )
    run_suite.add_argument("--write-sensitive-trace", action="store_true", help="Also write raw_trace.sensitive.json for each scenario")
    run_suite.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in generated artifacts")
    run_suite.add_argument(
        "--metadata",
        action="append",
        default=[],
        type=_metadata_arg,
        metavar="KEY=VALUE",
        help="Attach experiment metadata to suite and optional training-export artifacts; may be repeated",
    )
    run_suite.add_argument("--fail-on-failed", action="store_true", help="Exit nonzero when any scenario score fails")
    run_suite.set_defaults(func=cmd_run_suite)

    goal3_handoff = subparsers.add_parser(
        "goal3-handoff",
        help="Build a Goal 3 training-data handoff with export, validation, gate, evidence bundle, and preflight artifacts",
    )
    goal3_handoff.add_argument("--scenarios", required=True, help="Directory containing scenario JSON files")
    goal3_handoff.add_argument("--out", required=True, help="Output directory for the reproducible Goal 3 handoff")
    goal3_handoff.add_argument("--pattern", default="*.json", help="Scenario filename glob relative to --scenarios")
    goal3_handoff.add_argument("--recursive", action="store_true", help="Discover scenarios recursively with --pattern")
    goal3_handoff.add_argument("--suite-manifest", help="Eval suite manifest with an explicit scenario_ids list to run")
    goal3_handoff.add_argument("--format", default="auto", choices=TRACE_FORMAT_CHOICES)
    goal3_handoff.add_argument("--policy", help="Versioned training gate policy JSON file")
    goal3_handoff.add_argument("--trainer-command", required=True, help="Trainer command to record in preflight; not executed")
    goal3_handoff.add_argument("--reward-scale", default="score", choices=["score", "binary", "signed"])
    goal3_handoff.add_argument("--min-score-gap", type=int, default=1)
    goal3_handoff.add_argument("--max-pairs-per-family", type=int, default=0)
    goal3_handoff.add_argument("--strict", action="store_true", help="Treat validation warnings as handoff blockers")
    goal3_handoff.add_argument("--force", action="store_true", help="Replace an existing non-empty handoff directory")
    goal3_handoff.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in generated handoff artifacts")
    goal3_handoff.add_argument(
        "--metadata",
        action="append",
        default=[],
        type=_metadata_arg,
        metavar="KEY=VALUE",
        help="Attach metadata to suite, training export, and trainer preflight artifacts; may be repeated",
    )
    goal3_handoff.set_defaults(func=cmd_goal3_handoff)

