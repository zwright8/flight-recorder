"""Extracted validation implementation."""

from __future__ import annotations

import json
import shlex
from pathlib import Path, PureWindowsPath
from typing import Any
from ..adapters import TRACE_SCHEMA_VERSION
from ..bundle import EVIDENCE_BUNDLE_NOTES, EVIDENCE_BUNDLE_SCHEMA_VERSION, HARNESS_RUN_MANIFEST_SCHEMA_VERSION, HARNESS_RUN_RESULT_SCHEMA_VERSION, _decision_key_metrics as _build_evidence_bundle_decision_key_metrics, _decision_text as _build_evidence_bundle_decision_text, _next_actions as _build_evidence_bundle_next_actions
from ..schema_registry import SchemaRegistryError, check_schema_contract, check_schema_file
from ..trajectory_v2 import check_trajectory_v2
from ..digest import RUN_DIGEST_SCHEMA_VERSION
from ..hermes_plugin import LIVE_SMOKE_SUMMARY_SCHEMA_VERSION
from ..lineage import LINEAGE_SCHEMA_VERSION, REPLAY_BUNDLE_SCHEMA_VERSION
from ..governance import PROMOTION_ALIAS_APPLY_SCHEMA_VERSION, PROMOTION_ALIAS_APPLY_CHECK_IDS, PROMOTION_CARDS_REQUIRED_INPUTS, PROMOTION_CARDS_SCHEMA_VERSION, PROMOTION_DECISION_REQUIRED_ARTIFACTS, PROMOTION_DECISION_REQUIRED_PASS_CHECK_IDS, PROMOTION_DECISION_SCHEMA_VERSION, PROMOTION_POLICY_DEFAULT_LIMITS, PROMOTION_POLICY_REQUIRED_FORBIDDEN_RULES, PROMOTION_POLICY_SCHEMA_VERSION, PROMOTION_ROLLBACK_RECEIPT_SCHEMA_VERSION, PROMOTION_RELEASE_RECORD_REQUIRED_ARTIFACTS, PROMOTION_RELEASE_RECORD_VALIDATED_ARTIFACTS, PROMOTION_RELEASE_RECORD_SCHEMA_VERSION, _JSON_ARTIFACT_ROLES as _PROMOTION_JSON_ARTIFACT_ROLES, build_promotion_decision as _build_promotion_decision, _decision_metrics as _build_promotion_decision_metrics, _promotion_external_eval_checks as _build_promotion_external_eval_checks, _promotion_external_eval_lineage as _build_promotion_external_eval_lineage, promotion_release_record_check_ids
from ..runtime_adapter_router import ADAPTER_ROUTE_DECISION_SCHEMA_VERSION, TOOL_CAPABILITY_SELECTION_SCHEMA_VERSION, canonical_sha256 as _router_canonical_sha256, validate_adapter_route_decision as _validate_adapter_route_decision_semantics, validate_tool_capability_selection as _validate_tool_capability_selection_semantics
from ..path_safety import path_has_symlink_component as _path_has_symlink_component, resolve_artifact_reference_path
from ..scorers import SCORE_SCHEMA_VERSION, TASK_COMPLETION_SCHEMA_VERSION
from ..state_capture import STATE_SNAPSHOT_SCHEMA_VERSION
from ..state_diff import STATE_DIFF_SCHEMA_VERSION, resolve_state_diff_semantics
from ..trace_observability import TRACE_OBSERVABILITY_SCHEMA_VERSION, build_trace_signal
from ..verifiers import VERIFIER_SOURCES_SCHEMA_VERSION
from ..hashing import sha256_file as _sha256
from .constants import HARNESS_REPLAY_RESULT_SCHEMA_VERSION, HARNESS_SUITE_RESULT_SCHEMA_VERSION, LEGACY_LIVE_SMOKE_SUMMARY_SCHEMA_VERSIONS, _COMPANION_NOT_PROVIDED
from .primitives import ValidationTarget, _count_rows, _is_int_between, _is_lowercase_sha256, _is_non_empty_string_list, _is_non_negative_int, _is_sha256, _is_string_list, _looks_absolute, _merge_count_rows, _non_negative_int_value, _rate_value, _read_object, _read_object_optional, _reject_symlinked_validation_path, _require_equal, _sha256, _validate_allowed_keys, _validate_evidence_refs, _warn_absolute_public_path

def validate_runs_dir(path: str | Path) -> list[ValidationTarget]:
    """Validate every completed run directory inside a runs directory."""
    root = Path(path)
    target = ValidationTarget("runs", str(root))
    if _reject_symlinked_validation_path(root, target, "Runs directory", "directory"):
        return [target]
    if not root.exists():
        return [ValidationTarget("runs", str(root), errors=[f"Runs directory not found: {root}"])]
    if not root.is_dir():
        return [ValidationTarget("runs", str(root), errors=[f"Runs path is not a directory: {root}"])]

    targets: list[ValidationTarget] = []
    for child in sorted(item for item in root.iterdir() if item.is_dir()):
        has_trace = (child / "normalized_trace.json").exists()
        has_scorecard = (child / "scorecard.json").exists()
        if has_trace or has_scorecard:
            targets.append(validate_run_dir(child))
    if not targets:
        targets.append(ValidationTarget("runs", str(root), errors=[f"No completed run directories found in {root}"]))
    return targets

def validate_run_dir(path: str | Path) -> ValidationTarget:
    """Validate one Flight Recorder run directory."""
    run_dir = Path(path)
    target = ValidationTarget("run", str(run_dir))
    if _reject_symlinked_validation_path(run_dir, target, "Run path", "directory"):
        return target
    if not run_dir.exists():
        target.errors.append(f"Run directory not found: {run_dir}")
        return target
    if not run_dir.is_dir():
        target.errors.append(f"Run path is not a directory: {run_dir}")
        return target

    trace_path = run_dir / "normalized_trace.json"
    trajectory_v2_path = run_dir / "trajectory_v2.json"
    score_path = run_dir / "scorecard.json"
    task_completion_path = run_dir / "task_completion.json"
    run_digest_path = run_dir / "run_digest.json"
    state_snapshot_path = run_dir / "state_snapshot.json"
    state_diff_path = run_dir / "state_diff.json"
    report_path = run_dir / "report.html"
    lineage_path = run_dir / "artifact_lineage.json"
    trace = _read_object(trace_path, target, "normalized_trace.json")
    trajectory_v2 = (
        _read_object(trajectory_v2_path, target, "trajectory_v2.json")
        if trajectory_v2_path.exists()
        else None
    )
    scorecard = _read_object(score_path, target, "scorecard.json")
    task_completion = _read_object_optional(
        task_completion_path,
        target,
        "task_completion.json",
        "rerun the run to emit the standalone task-completion verdict",
    )
    run_digest = _read_object_optional(
        run_digest_path,
        target,
        "run_digest.json",
        "rerun the run to emit the compact per-run evidence digest",
    )
    lineage = _read_object_optional(lineage_path, target, "artifact_lineage.json", "rerun the run to emit provenance metadata")
    state_snapshot = _read_object(state_snapshot_path, target, "state_snapshot.json") if state_snapshot_path.exists() else None
    state_diff = _read_object(state_diff_path, target, "state_diff.json") if state_diff_path.exists() else None
    if trace is not None:
        _validate_trace(trace, target)
    if trajectory_v2 is not None:
        trajectory_validation = check_trajectory_v2(trajectory_v2)
        target.errors.extend(
            f"trajectory_v2.json: {error}"
            for error in trajectory_validation.get("errors", [])
        )
        if trajectory_validation.get("quarantined") is True:
            target.warnings.append(
                "trajectory_v2.json is quarantined from action training: "
                + ", ".join(trajectory_validation.get("quarantine_reasons", []))
            )
    if scorecard is not None:
        _validate_scorecard(scorecard, target)
    if task_completion is not None:
        _validate_task_completion(task_completion, target, "task_completion")
        if isinstance(scorecard, dict) and scorecard.get("task_completion") != task_completion:
            target.errors.append("task_completion.json must match scorecard.task_completion.")
    if run_digest is not None:
        _validate_run_digest(
            run_digest,
            target,
            "run_digest",
            trace=trace,
            scorecard=scorecard,
            task_completion=task_completion,
            state_diff=state_diff,
        )
    if lineage is not None:
        _validate_lineage(lineage, target, run_dir, trace, scorecard)
    if isinstance(state_snapshot, dict) and state_snapshot.get("schema_version") == STATE_SNAPSHOT_SCHEMA_VERSION:
        _validate_state_snapshot(state_snapshot, target, "state_snapshot")
    if state_diff is not None:
        _validate_state_diff(state_diff, target, "state_diff")
    if trace is not None and scorecard is not None:
        target.details.update(
            {
                "scenario_id": scorecard.get("scenario_id"),
                "score": scorecard.get("score"),
                "passed": scorecard.get("passed"),
                "event_count": len(trace.get("events", [])) if isinstance(trace.get("events"), list) else None,
            }
        )
    if not report_path.exists():
        target.warnings.append("report.html is missing; run evidence is less reviewable.")
    if isinstance(scorecard, dict) and scorecard.get("passed") is False and not (run_dir / "regression_scenario.json").exists():
        target.warnings.append("failing run is missing regression_scenario.json.")
    return target

def validate_tool_capability_selection_artifact(path: str | Path) -> ValidationTarget:
    """Validate a governed tool-capability selection artifact."""
    artifact_path = Path(path)
    target = ValidationTarget("tool_capability_selection", str(artifact_path))
    if _reject_symlinked_validation_path(artifact_path, target, "tool_capability_selection", "file"):
        return target
    artifact = _read_object(artifact_path, target, "tool_capability_selection.json")
    if artifact is None:
        return target
    _validate_schema_contract_target(
        artifact,
        "tool_capability_selection",
        target,
        artifact_path,
    )
    if artifact.get("schema_version") != TOOL_CAPABILITY_SELECTION_SCHEMA_VERSION:
        target.errors.append(
            f"tool_capability_selection.schema_version must be {TOOL_CAPABILITY_SELECTION_SCHEMA_VERSION!r}."
        )
    for error in _validate_tool_capability_selection_semantics(artifact):
        target.errors.append(f"tool_capability_selection semantic validation: {error}.")
    target.details.update(
        {
            "passed": artifact.get("passed"),
            "decision_fingerprint": artifact.get("decision_fingerprint"),
            "selected_tool_count": len(artifact.get("selected_tools", [])) if isinstance(artifact.get("selected_tools"), list) else 0,
        }
    )
    return target

def validate_adapter_route_decision_artifact(path: str | Path) -> ValidationTarget:
    """Validate a governed exact-one-adapter route decision artifact."""
    route_path = Path(path)
    target = ValidationTarget("adapter_route_decision", str(route_path))
    if _reject_symlinked_validation_path(route_path, target, "adapter_route_decision", "file"):
        return target
    artifact = _read_object(route_path, target, "adapter_route_decision.json")
    if artifact is None:
        return target
    _validate_schema_contract_target(
        artifact,
        "adapter_route_decision",
        target,
        route_path,
    )
    if artifact.get("schema_version") != ADAPTER_ROUTE_DECISION_SCHEMA_VERSION:
        target.errors.append(
            f"adapter_route_decision.schema_version must be {ADAPTER_ROUTE_DECISION_SCHEMA_VERSION!r}."
        )
    capability_selection = _validated_router_ref_payload(
        artifact.get("capability_selection_ref"),
        target,
        route_path,
        "adapter_route_decision.capability_selection_ref",
    )
    if capability_selection is not None and isinstance(artifact.get("capability_selection_ref"), dict):
        nested = validate_tool_capability_selection_artifact(
            route_path.parent / str(artifact["capability_selection_ref"]["path"])
        )
        for error in nested.errors:
            target.errors.append(f"adapter_route_decision capability selection: {error}")
        for warning in nested.warnings:
            target.warnings.append(f"adapter_route_decision capability selection: {warning}")
    verified_capability_ref = (
        capability_selection is not None
        and isinstance(artifact.get("capability_selection_ref"), dict)
        and artifact["capability_selection_ref"].get("sha256") == artifact.get("capability_selection_sha256")
    )
    for error in _validate_adapter_route_decision_semantics(
        artifact,
        capability_selection=capability_selection,
    ):
        if error == "capability_selection_sha256_mismatch" and verified_capability_ref:
            continue
        target.errors.append(f"adapter_route_decision semantic validation: {error}.")
    _validate_adapter_route_promotion_refs(artifact, target, route_path)
    target.details.update(
        {
            "passed": artifact.get("passed"),
            "route_fingerprint": artifact.get("route_fingerprint"),
            "selected_candidate": (
                artifact.get("selected_candidate", {}).get("candidate_id")
                if isinstance(artifact.get("selected_candidate"), dict)
                else None
            ),
        }
    )
    return target

def _validate_schema_contract_target(
    artifact: dict[str, Any],
    schema_name: str,
    target: ValidationTarget,
    artifact_path: Path,
) -> None:
    try:
        schema_check = check_schema_contract(
            artifact,
            name_or_id=schema_name,
            artifact_path=artifact_path,
        )
    except (SchemaRegistryError, TypeError, ValueError) as exc:
        target.errors.append(f"{schema_name} schema contract could not be checked: {exc}")
        return
    for error in schema_check.get("errors", []):
        target.errors.append(f"{schema_name} schema: {error}")

def _validated_router_ref_payload(
    ref: Any,
    target: ValidationTarget,
    source_path: Path,
    label: str,
) -> dict[str, Any] | None:
    resolved = _resolve_router_artifact_ref(ref, target, source_path, label)
    if resolved is None:
        return None
    ref_path, ref_record = resolved
    if ref_path.stat().st_size != ref_record.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    current_sha256 = _sha256(ref_path)
    if current_sha256 != ref_record.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current file.")
    try:
        payload = json.loads(ref_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        target.errors.append(f"{label}.path contains invalid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        target.errors.append(f"{label}.path must contain a JSON object.")
        return None
    return payload

def _resolve_router_artifact_ref(
    ref: Any,
    target: ValidationTarget,
    source_path: Path,
    label: str,
) -> tuple[Path, dict[str, Any]] | None:
    if not isinstance(ref, dict):
        target.errors.append(f"{label} must be an object.")
        return None
    _validate_allowed_keys(ref, {"path", "sha256", "size_bytes"}, target, label)
    path_value = ref.get("path")
    if not isinstance(path_value, str) or not path_value:
        target.errors.append(f"{label}.path must be a non-empty string.")
        return None
    if not _is_safe_runtime_router_ref_path(path_value):
        target.errors.append(f"{label}.path must be a safe relative path without traversal, Windows prefixes, tilde, or backslashes.")
        return None
    if not _is_sha256(ref.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string.")
    if not _is_non_negative_int(ref.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    ref_path = source_path.parent / path_value
    if _path_has_symlink_component(ref_path, include_leaf=True):
        target.errors.append(f"{label}.path must not contain symlink components.")
        return None
    if not ref_path.exists():
        target.errors.append(f"{label}.path does not exist.")
        return None
    if not ref_path.is_file():
        target.errors.append(f"{label}.path must resolve to a file.")
        return None
    return ref_path, ref

def _is_safe_runtime_router_ref_path(value: str) -> bool:
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

def _validate_adapter_route_promotion_refs(
    artifact: dict[str, Any],
    target: ValidationTarget,
    route_path: Path,
) -> None:
    rows = artifact.get("evaluated_candidates")
    if not isinstance(rows, list):
        return
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        row_label = f"adapter_route_decision.evaluated_candidates[{index}]"
        evidence = row.get("promotion_evidence")
        if not isinstance(evidence, dict):
            target.errors.append(f"{row_label}.promotion_evidence must be an object.")
            continue
        promotion = _validated_router_ref_payload(
            evidence.get("artifact_ref"),
            target,
            route_path,
            f"{row_label}.promotion_evidence.artifact_ref",
        )
        if promotion is not None:
            _validate_schema_contract_target(
                promotion,
                "promotion_decision",
                target,
                route_path.parent / str(evidence["artifact_ref"]["path"]),
            )
            if promotion.get("schema_version") != PROMOTION_DECISION_SCHEMA_VERSION:
                target.errors.append(
                    f"{row_label}.promotion_evidence.artifact_ref must reference {PROMOTION_DECISION_SCHEMA_VERSION!r}."
                )
            if evidence.get("decision_sha256") != _router_canonical_sha256(promotion):
                target.errors.append(f"{row_label}.promotion_evidence.decision_sha256 does not match the referenced promotion decision.")
            expected_summary = _router_promotion_decision_summary(promotion)
            if evidence.get("decision_summary") != expected_summary:
                target.errors.append(f"{row_label}.promotion_evidence.decision_summary does not match the referenced promotion decision.")
        binding = row.get("promotion_binding")
        if not isinstance(binding, dict):
            target.errors.append(f"{row_label}.promotion_binding must be an object.")
            continue
        for field_name in ("training_result_ref", "evaluation_result_ref"):
            _validated_router_ref_payload(
                binding.get(field_name),
                target,
                route_path,
                f"{row_label}.promotion_binding.{field_name}",
            )

def _router_promotion_decision_summary(promotion: dict[str, Any]) -> dict[str, Any]:
    alias_update = promotion.get("alias_update") if isinstance(promotion.get("alias_update"), dict) else {}
    models = promotion.get("models") if isinstance(promotion.get("models"), dict) else {}
    candidate = models.get("candidate") if isinstance(models.get("candidate"), dict) else {}
    return {
        "schema_version": str(promotion.get("schema_version") or ""),
        "passed": promotion.get("passed") is True,
        "recommendation": str(promotion.get("recommendation") or ""),
        "alias_update_authorized": promotion.get("alias_update_authorized") is True or alias_update.get("authorized") is True,
        "models_candidate_id": str(candidate.get("id") or ""),
    }

def validate_live_smoke_summary(path: str | Path) -> ValidationTarget:
    """Validate a live Hermes observer-smoke summary artifact."""
    summary_path = Path(path)
    target = ValidationTarget("live_smoke_summary", str(summary_path))
    summary = _read_object(summary_path, target, "live_smoke_summary.json")
    if summary is not None:
        _validate_live_smoke_summary(summary, target)
    return target

def validate_trace_observability(path: str | Path) -> ValidationTarget:
    """Validate a trace-observability summary artifact."""
    observability_path = Path(path)
    target = ValidationTarget("trace_observability", str(observability_path))
    observability = _read_object(observability_path, target, "trace_observability.json")
    if observability is not None:
        _validate_trace_observability(observability, target)
    return target

def validate_state_snapshot(path: str | Path) -> ValidationTarget:
    """Validate a captured state snapshot artifact."""
    snapshot_path = Path(path)
    target = ValidationTarget("state_snapshot", str(snapshot_path))
    snapshot = _read_object(snapshot_path, target, "state_snapshot.json")
    if snapshot is not None:
        _validate_state_snapshot(snapshot, target, "state_snapshot", source_path=snapshot_path)
    return target

def validate_state_diff(path: str | Path) -> ValidationTarget:
    """Validate a before/after state diff artifact."""
    diff_path = Path(path)
    target = ValidationTarget("state_diff", str(diff_path))
    diff = _read_object(diff_path, target, "state_diff.json")
    if diff is not None:
        _validate_state_diff(diff, target, "state_diff")
    return target

def validate_run_digest(path: str | Path) -> ValidationTarget:
    """Validate a compact per-run evidence digest artifact."""
    digest_path = Path(path)
    target = ValidationTarget("run_digest", str(digest_path))
    digest = _read_object(digest_path, target, "run_digest.json")
    if digest is not None:
        _validate_run_digest(digest, target, "run_digest")
    return target

def validate_harness_run_manifest(path: str | Path) -> ValidationTarget:
    """Validate a harness run manifest handoff artifact."""
    manifest_path = Path(path)
    target = ValidationTarget("harness_run_manifest", str(manifest_path))
    manifest = _read_object(manifest_path, target, "harness_manifest.json")
    if manifest is not None:
        _validate_harness_run_manifest(manifest, target)
    return target

def validate_harness_run_result(path: str | Path) -> ValidationTarget:
    """Validate a harness run result handoff artifact."""
    result_path = Path(path)
    target = ValidationTarget("harness_run_result", str(result_path))
    result = _read_object(result_path, target, "harness_result.json")
    if result is not None:
        _validate_harness_run_result(result, target, source_dir=result_path.parent)
    return target

def validate_harness_replay_result(path: str | Path) -> ValidationTarget:
    """Validate a harness_replay_result.json artifact."""
    result_path = Path(path)
    target = ValidationTarget("harness_replay_result", str(result_path))
    result = _read_object(result_path, target, "harness_replay_result.json")
    if result is not None:
        _validate_harness_replay_result(result, target, source_dir=result_path.parent)
    return target

def _validate_harness_run_manifest(manifest: dict[str, Any], target: ValidationTarget) -> None:
    if manifest.get("schema_version") != HARNESS_RUN_MANIFEST_SCHEMA_VERSION:
        target.errors.append(
            f"harness_manifest.schema_version must be {HARNESS_RUN_MANIFEST_SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}."
        )
    _warn_absolute_public_path(target, "harness_manifest.manifest_path", manifest.get("manifest_path"))
    for field_name in ("runner", "provider"):
        if not isinstance(manifest.get(field_name), str) or not manifest.get(field_name):
            target.errors.append(f"harness_manifest.{field_name} must be a non-empty string.")
    for field_name in ("model", "scenario", "outputs", "sandbox", "tool_policy"):
        if not isinstance(manifest.get(field_name), dict):
            target.errors.append(f"harness_manifest.{field_name} must be an object.")
    model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    if not isinstance(model.get("id"), str) or not model.get("id"):
        target.errors.append("harness_manifest.model.id must be a non-empty string.")
    scenario = manifest.get("scenario") if isinstance(manifest.get("scenario"), dict) else {}
    for field_name in ("id", "path"):
        if not isinstance(scenario.get(field_name), str) or not scenario.get(field_name):
            target.errors.append(f"harness_manifest.scenario.{field_name} must be a non-empty string.")
    _warn_absolute_public_path(target, "harness_manifest.scenario.path", scenario.get("path"))
    source_trace = manifest.get("source_trace") if isinstance(manifest.get("source_trace"), dict) else {}
    _warn_absolute_public_path(target, "harness_manifest.source_trace.path", source_trace.get("path"))
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    for field_name in ("run_dir", "manifest", "result"):
        if not isinstance(outputs.get(field_name), str) or not outputs.get(field_name):
            target.errors.append(f"harness_manifest.outputs.{field_name} must be a non-empty string.")
        _warn_absolute_public_path(target, f"harness_manifest.outputs.{field_name}", outputs.get(field_name))
    sandbox = manifest.get("sandbox") if isinstance(manifest.get("sandbox"), dict) else {}
    for field_name in ("root", "home", "workspace", "events"):
        if not isinstance(sandbox.get(field_name), str) or not sandbox.get(field_name):
            target.errors.append(f"harness_manifest.sandbox.{field_name} must be a non-empty string.")
    _warn_harness_sandbox_public_paths(sandbox, target, "harness_manifest.sandbox")
    canaries = sandbox.get("fake_secret_canaries")
    if not isinstance(canaries, list) or not canaries:
        target.errors.append("harness_manifest.sandbox.fake_secret_canaries must be a non-empty list.")
    else:
        _validate_harness_fake_secret_canaries(canaries, target)
    _validate_harness_tool_policy(manifest.get("tool_policy"), target, "harness_manifest.tool_policy")

def _validate_harness_fake_secret_canaries(canaries: list[Any], target: ValidationTarget) -> None:
    for index, canary in enumerate(canaries):
        label = f"harness_manifest.sandbox.fake_secret_canaries[{index}]"
        if not isinstance(canary, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        if not isinstance(canary.get("name"), str) or not canary.get("name"):
            target.errors.append(f"{label}.name must be a non-empty string.")
        if not _is_lowercase_sha256(canary.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a lowercase SHA-256 hex string.")

def _validate_harness_run_result(result: dict[str, Any], target: ValidationTarget, source_dir: Path | None = None) -> None:
    if result.get("schema_version") != HARNESS_RUN_RESULT_SCHEMA_VERSION:
        target.errors.append(
            f"harness_result.schema_version must be {HARNESS_RUN_RESULT_SCHEMA_VERSION!r}, got {result.get('schema_version')!r}."
        )
    for field_name in ("runner", "provider", "scenario_id"):
        if not isinstance(result.get(field_name), str) or not result.get(field_name):
            target.errors.append(f"harness_result.{field_name} must be a non-empty string.")
    for field_name in ("model", "sandbox", "tool_policy", "trace", "scorecard", "artifacts", "replay"):
        if not isinstance(result.get(field_name), dict):
            target.errors.append(f"harness_result.{field_name} must be an object.")
    trace = result.get("trace") if isinstance(result.get("trace"), dict) else {}
    _validate_harness_named_file_ref(trace, "path", "sha256", "size_bytes", target, "harness_result.trace", source_dir)
    scorecard = result.get("scorecard") if isinstance(result.get("scorecard"), dict) else {}
    _validate_harness_named_file_ref(scorecard, "path", "sha256", "size_bytes", target, "harness_result.scorecard", source_dir)
    if not isinstance(scorecard.get("passed"), bool):
        target.errors.append("harness_result.scorecard.passed must be a boolean.")
    if not isinstance(scorecard.get("score"), (int, float)) or isinstance(scorecard.get("score"), bool):
        target.errors.append("harness_result.scorecard.score must be numeric.")
    if "fake_secret_canary_check" in result:
        _validate_harness_fake_secret_canary_check(
            result.get("fake_secret_canary_check"),
            target,
            "harness_result.fake_secret_canary_check",
            source_dir,
        )
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    for field_name in ("normalized_trace", "scorecard", "run_digest", "report", "lineage"):
        _validate_harness_run_artifact_ref(artifacts, field_name, target, "harness_result.artifacts", source_dir)
    _validate_harness_matching_ref(
        scorecard,
        "path",
        "sha256",
        "size_bytes",
        artifacts,
        "scorecard",
        target,
        "harness_result.scorecard",
    )
    replay = result.get("replay") if isinstance(result.get("replay"), dict) else {}
    _validate_harness_named_file_ref(
        replay,
        "lineage",
        "lineage_sha256",
        "lineage_size_bytes",
        target,
        "harness_result.replay",
        source_dir,
    )
    _validate_harness_matching_ref(
        replay,
        "lineage",
        "lineage_sha256",
        "lineage_size_bytes",
        artifacts,
        "lineage",
        target,
        "harness_result.replay",
    )
    _warn_replay_metadata_public_paths(replay, target, "harness_result.replay")
    if not isinstance(replay.get("self_contained"), bool):
        target.errors.append("harness_result.replay.self_contained must be a boolean.")
    manifest = result.get("manifest") if isinstance(result.get("manifest"), dict) else {}
    _warn_absolute_public_path(target, "harness_result.manifest.path", manifest.get("path"))
    sandbox = result.get("sandbox") if isinstance(result.get("sandbox"), dict) else {}
    _warn_harness_sandbox_public_paths(sandbox, target, "harness_result.sandbox")
    _validate_harness_tool_policy(result.get("tool_policy"), target, "harness_result.tool_policy")

def _warn_harness_sandbox_public_paths(sandbox: dict[str, Any], target: ValidationTarget, label: str) -> None:
    for field_name in ("root", "home", "workspace", "events"):
        _warn_absolute_public_path(target, f"{label}.{field_name}", sandbox.get(field_name))
    fake_secret_files = sandbox.get("fake_secret_files")
    if isinstance(fake_secret_files, list):
        for index, path_value in enumerate(fake_secret_files):
            _warn_absolute_public_path(target, f"{label}.fake_secret_files[{index}]", path_value)

def _validate_harness_named_file_ref(
    value: dict[str, Any],
    path_field: str,
    sha_field: str,
    size_field: str,
    target: ValidationTarget,
    label: str,
    source_dir: Path | None,
) -> Path | None:
    path_label = f"{label}.{path_field}"
    _warn_absolute_public_path(target, path_label, value.get(path_field))
    path = _resolve_harness_artifact_path(value.get(path_field), source_dir)
    if path is None:
        target.errors.append(f"{path_label} must be a non-empty path.")
        return None
    expected_sha = value.get(sha_field)
    expected_size = value.get(size_field)
    if not _is_lowercase_sha256(expected_sha):
        target.errors.append(f"{label}.{sha_field} must be a lowercase SHA-256 hex string.")
    if not _is_non_negative_int(expected_size):
        target.errors.append(f"{label}.{size_field} must be a non-negative integer.")
    if _harness_path_uses_symlink(path):
        target.errors.append(f"{path_label} must resolve to a regular non-symlink file.")
        return path
    if not path.is_file():
        target.errors.append(f"{path_label} must resolve to an existing file.")
        return path
    if _is_non_negative_int(expected_size) and path.stat().st_size != expected_size:
        target.errors.append(f"{label}.{size_field} does not match the current file.")
    if _is_lowercase_sha256(expected_sha) and _sha256(path) != expected_sha:
        target.errors.append(f"{label}.{sha_field} does not match the current file.")
    return path

def _validate_harness_matching_ref(
    value: dict[str, Any],
    path_field: str,
    sha_field: str,
    size_field: str,
    artifacts: dict[str, Any],
    artifact_field: str,
    target: ValidationTarget,
    label: str,
) -> None:
    artifact_label = f"harness_result.artifacts.{artifact_field}"
    expected = {
        path_field: artifacts.get(artifact_field),
        sha_field: artifacts.get(f"{artifact_field}_sha256"),
        size_field: artifacts.get(f"{artifact_field}_size_bytes"),
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} must match {artifact_label}.")

def _validate_harness_run_artifact_ref(
    artifacts: dict[str, Any],
    field_name: str,
    target: ValidationTarget,
    label: str,
    source_dir: Path | None,
) -> None:
    _validate_harness_named_file_ref(
        artifacts,
        field_name,
        f"{field_name}_sha256",
        f"{field_name}_size_bytes",
        target,
        label,
        source_dir,
    )

def _resolve_harness_artifact_path(value: Any, source_dir: Path | None) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or source_dir is None:
        return path
    return source_dir / path

def _validate_harness_tool_policy(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        return
    if not isinstance(value.get("source"), str) or not value.get("source"):
        target.errors.append(f"{label}.source must be a non-empty string.")
    if not isinstance(value.get("scenario_policy"), dict):
        target.errors.append(f"{label}.scenario_policy must be an object.")
    runtime_policy = value.get("runtime_policy")
    if not isinstance(runtime_policy, dict):
        target.errors.append(f"{label}.runtime_policy must be an object.")
    else:
        if not isinstance(runtime_policy.get("mode"), str) or not runtime_policy.get("mode"):
            target.errors.append(f"{label}.runtime_policy.mode must be a non-empty string.")
        for field_name in ("allowed_tools", "denied_tools"):
            if field_name in runtime_policy and not _is_string_list(runtime_policy.get(field_name)):
                target.errors.append(f"{label}.runtime_policy.{field_name} must be a list of strings.")
        network = runtime_policy.get("network")
        if network is not None:
            if not isinstance(network, dict):
                target.errors.append(f"{label}.runtime_policy.network must be an object.")
            else:
                if not isinstance(network.get("mode"), str) or not network.get("mode"):
                    target.errors.append(f"{label}.runtime_policy.network.mode must be a non-empty string.")
                if "allowed_hosts" in network and not _is_string_list(network.get("allowed_hosts")):
                    target.errors.append(f"{label}.runtime_policy.network.allowed_hosts must be a list of strings.")
    canaries = value.get("blocked_action_canaries")
    if not isinstance(canaries, list):
        target.errors.append(f"{label}.blocked_action_canaries must be a list.")
        return
    for index, canary in enumerate(canaries):
        if not isinstance(canary, dict):
            target.errors.append(f"{label}.blocked_action_canaries[{index}] must be an object.")
            continue
        for field_name in ("type", "pattern", "expected"):
            if not isinstance(canary.get(field_name), str) or not canary.get(field_name):
                target.errors.append(f"{label}.blocked_action_canaries[{index}].{field_name} must be a non-empty string.")

def _validate_harness_fake_secret_canary_check(value: Any, target: ValidationTarget, label: str, source_dir: Path | None) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if not isinstance(value.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    for field_name in ("canary_count", "checked_artifact_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    checked_artifacts = value.get("checked_artifacts")
    if not isinstance(checked_artifacts, list):
        target.errors.append(f"{label}.checked_artifacts must be a list.")
        checked_artifacts = []
    else:
        for index, artifact in enumerate(checked_artifacts):
            _validate_harness_canary_artifact_record(artifact, target, f"{label}.checked_artifacts[{index}]", source_dir)
    if _is_non_negative_int(value.get("checked_artifact_count")) and value.get("checked_artifact_count") != len(checked_artifacts):
        target.errors.append(f"{label}.checked_artifact_count expected {len(checked_artifacts)}, got {value.get('checked_artifact_count')!r}.")
    checked_artifact_keys = {_harness_canary_artifact_key(item) for item in checked_artifacts if isinstance(item, dict)}
    leaked_artifacts = value.get("leaked_artifacts")
    if not isinstance(leaked_artifacts, list):
        target.errors.append(f"{label}.leaked_artifacts must be a list.")
        leaked_artifacts = []
    else:
        for index, artifact in enumerate(leaked_artifacts):
            record_label = f"{label}.leaked_artifacts[{index}]"
            _validate_harness_canary_artifact_record(artifact, target, record_label, source_dir)
            if isinstance(artifact, dict):
                if not _is_non_empty_string_list(artifact.get("canary_names")):
                    target.errors.append(f"{record_label}.canary_names must be a non-empty list of strings.")
                if _harness_canary_artifact_key(artifact) not in checked_artifact_keys:
                    target.errors.append(f"{record_label} must also appear in checked_artifacts by artifact and path.")
    if isinstance(value.get("passed"), bool) and value.get("passed") != (not leaked_artifacts):
        target.errors.append(f"{label}.passed must be true only when leaked_artifacts is empty.")

def _validate_harness_canary_artifact_record(value: Any, target: ValidationTarget, label: str, source_dir: Path | None) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    for field_name in ("artifact", "path"):
        if not isinstance(value.get(field_name), str) or not value.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    _warn_absolute_public_path(target, f"{label}.path", value.get("path"))
    if not isinstance(value.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    if value.get("exists") is True:
        if not _is_non_negative_int(value.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for existing files.")
        if not _is_sha256(value.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string for existing files.")
    current_path = _resolve_harness_canary_artifact_path(value.get("path"), source_dir)
    if current_path is None:
        return
    if _harness_path_uses_symlink(current_path):
        target.errors.append(f"{label}.path must resolve to a regular non-symlink file.")
        return
    if value.get("exists") is True:
        if not current_path.exists():
            target.errors.append(f"{label}.path must resolve to an existing file when exists is true.")
            return
        if not current_path.is_file():
            target.errors.append(f"{label}.path must resolve to a file when exists is true.")
            return
    elif not current_path.is_file():
        return
    if value.get("exists") is not True:
        target.errors.append(f"{label}.exists must be true when path resolves to a file.")
    if _is_non_negative_int(value.get("size_bytes")) and current_path.stat().st_size != value.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if _is_sha256(value.get("sha256")) and _sha256(current_path) != value.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current file.")

def _resolve_harness_canary_artifact_path(value: Any, source_dir: Path | None) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or source_dir is None:
        return path
    return source_dir / path

def _harness_canary_artifact_key(value: dict[str, Any]) -> tuple[Any, Any]:
    return (value.get("artifact"), value.get("path"))

def _validate_harness_replay_result(
    result: dict[str, Any],
    target: ValidationTarget,
    source_dir: Path | None = None,
) -> None:
    if result.get("schema_version") != HARNESS_REPLAY_RESULT_SCHEMA_VERSION:
        target.errors.append(
            f"harness_replay_result.schema_version must be {HARNESS_REPLAY_RESULT_SCHEMA_VERSION!r}, got {result.get('schema_version')!r}."
        )
    _validate_harness_replay_artifact_ref(result, "lineage", target, "harness_replay_result", source_dir)
    out_dir = result.get("out_dir")
    if not isinstance(out_dir, str) or not out_dir:
        target.errors.append("harness_replay_result.out_dir must be a non-empty string.")
    else:
        _warn_absolute_public_path(target, "harness_replay_result.out_dir", out_dir)
        out_path = Path(out_dir)
        if not out_path.is_absolute() and source_dir is not None:
            out_path = source_dir / out_path
        if _path_has_symlink_component(out_path, include_leaf=True):
            target.errors.append("harness_replay_result.out_dir must resolve to a non-symlink directory.")
        elif not out_path.is_dir():
            target.errors.append(f"harness_replay_result.out_dir does not exist or is not a directory: {out_dir}.")
    if not _is_non_negative_int(result.get("exit_code")):
        target.errors.append("harness_replay_result.exit_code must be a non-negative integer.")
    scorecard_path: Path | None = None
    if result.get("scorecard") is not None:
        scorecard_path = _validate_harness_replay_artifact_ref(
            result,
            "scorecard",
            target,
            "harness_replay_result",
            source_dir,
        )
    elif result.get("passed") is True:
        target.errors.append("harness_replay_result.scorecard must be present when passed is true.")
    elif "scorecard_sha256" in result or "scorecard_size_bytes" in result:
        target.errors.append("harness_replay_result scorecard fingerprints must be omitted when scorecard is null.")
    if not isinstance(result.get("passed"), bool):
        target.errors.append("harness_replay_result.passed must be a boolean.")
    if scorecard_path is not None and scorecard_path.exists() and scorecard_path.is_file():
        scorecard = _read_object(scorecard_path, target, "harness_replay_result.scorecard")
        if isinstance(scorecard, dict):
            _validate_scorecard(scorecard, target)
            if isinstance(result.get("passed"), bool) and isinstance(scorecard.get("passed"), bool) and result["passed"] != scorecard["passed"]:
                target.errors.append("harness_replay_result.passed must match referenced scorecard.passed.")

def _validate_harness_replay_artifact_ref(
    result: dict[str, Any],
    field_name: str,
    target: ValidationTarget,
    label: str,
    source_dir: Path | None,
) -> Path | None:
    path_label = f"{label}.{field_name}"
    _warn_absolute_public_path(target, path_label, result.get(field_name))
    path = _resolve_harness_replay_artifact_path(result.get(field_name), source_dir)
    if path is None:
        target.errors.append(f"{path_label} must be a non-empty path.")
        return None
    sha_field = f"{field_name}_sha256"
    size_field = f"{field_name}_size_bytes"
    expected_sha = result.get(sha_field)
    expected_size = result.get(size_field)
    if not _is_lowercase_sha256(expected_sha):
        target.errors.append(f"{label}.{sha_field} must be a lowercase SHA-256 hex string.")
    if not _is_non_negative_int(expected_size):
        target.errors.append(f"{label}.{size_field} must be a non-negative integer.")
    if _harness_path_uses_symlink(path):
        target.errors.append(f"{path_label} must resolve to a regular non-symlink file.")
        return path
    if not path.is_file():
        target.errors.append(f"{path_label} must resolve to an existing file.")
        return path
    if _is_non_negative_int(expected_size) and path.stat().st_size != expected_size:
        target.errors.append(f"{label}.{size_field} does not match the current file.")
    if _is_lowercase_sha256(expected_sha) and _sha256(path) != expected_sha:
        target.errors.append(f"{label}.{sha_field} does not match the current file.")
    return path

def _resolve_harness_replay_artifact_path(value: Any, source_dir: Path | None) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or source_dir is None:
        return path
    return source_dir / path

def _validate_harness_suite_result(result: dict[str, Any], target: ValidationTarget, source_dir: Path | None) -> None:
    if result.get("schema_version") != HARNESS_SUITE_RESULT_SCHEMA_VERSION:
        target.errors.append(
            f"harness_suite_result.schema_version must be {HARNESS_SUITE_RESULT_SCHEMA_VERSION!r}, got {result.get('schema_version')!r}."
        )
    for field_name in ("runner_interface", "scenarios_dir", "out_dir", "pattern", "runner", "provider"):
        if not isinstance(result.get(field_name), str) or not result.get(field_name):
            target.errors.append(f"harness_suite_result.{field_name} must be a non-empty string.")
    for field_name in ("scenarios_dir", "out_dir"):
        _warn_absolute_public_path(target, f"harness_suite_result.{field_name}", result.get(field_name))
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    _warn_absolute_public_path(target, "harness_suite_result.artifacts.suite_result", artifacts.get("suite_result"))
    if not isinstance(result.get("recursive"), bool):
        target.errors.append("harness_suite_result.recursive must be a boolean.")
    runs = result.get("runs")
    if not isinstance(runs, list):
        target.errors.append("harness_suite_result.runs must be a list.")
        runs = []
    errors = result.get("errors")
    if not isinstance(errors, list):
        target.errors.append("harness_suite_result.errors must be a list.")
        errors = []
    for index, error in enumerate(errors):
        if isinstance(error, dict):
            _warn_absolute_public_path(target, f"harness_suite_result.errors[{index}].scenario_path", error.get("scenario_path"))
    passed_count = 0
    for index, run in enumerate(runs):
        label = f"harness_suite_result.runs[{index}]"
        if not isinstance(run, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        if not isinstance(run.get("scenario_id"), str) or not run.get("scenario_id"):
            target.errors.append(f"{label}.scenario_id must be a non-empty string.")
        if not isinstance(run.get("passed"), bool):
            target.errors.append(f"{label}.passed must be a boolean.")
        elif run["passed"]:
            passed_count += 1
        _validate_harness_suite_artifact_ref(run, "manifest", target, label, source_dir)
        _validate_harness_suite_artifact_ref(run, "result", target, label, source_dir)
        _warn_harness_suite_run_public_paths(run, target, label)
    _require_equal(result, "total", len(runs), target, prefix="harness_suite_result.")
    _require_equal(result, "passed", passed_count, target, prefix="harness_suite_result.")
    _require_equal(result, "failed", len(runs) - passed_count, target, prefix="harness_suite_result.")
    _require_equal(result, "error_count", len(errors), target, prefix="harness_suite_result.")

def _validate_harness_suite_artifact_ref(
    run: dict[str, Any],
    field_name: str,
    target: ValidationTarget,
    label: str,
    source_dir: Path | None,
) -> None:
    path_label = f"{label}.{field_name}"
    _warn_absolute_public_path(target, path_label, run.get(field_name))
    path = _resolve_harness_suite_artifact_path(run.get(field_name), source_dir)
    if path is None:
        target.errors.append(f"{path_label} must be a non-empty path.")
        return
    sha_field = f"{field_name}_sha256"
    size_field = f"{field_name}_size_bytes"
    expected_sha = run.get(sha_field)
    expected_size = run.get(size_field)
    if not _is_lowercase_sha256(expected_sha):
        target.errors.append(f"{label}.{sha_field} must be a lowercase SHA-256 hex string.")
    if not _is_non_negative_int(expected_size):
        target.errors.append(f"{label}.{size_field} must be a non-negative integer.")
    if _harness_path_uses_symlink(path):
        target.errors.append(f"{path_label} must resolve to a regular non-symlink file.")
        return
    if not path.is_file():
        target.errors.append(f"{path_label} must resolve to an existing file.")
        return
    if _is_non_negative_int(expected_size) and path.stat().st_size != expected_size:
        target.errors.append(f"{label}.{size_field} does not match the current file.")
    if _is_lowercase_sha256(expected_sha) and _sha256(path) != expected_sha:
        target.errors.append(f"{label}.{sha_field} does not match the current file.")

def _warn_harness_suite_run_public_paths(run: dict[str, Any], target: ValidationTarget, label: str) -> None:
    _warn_absolute_public_path(target, f"{label}.run_dir", run.get("run_dir"))
    trace = run.get("trace") if isinstance(run.get("trace"), dict) else {}
    _warn_absolute_public_path(target, f"{label}.trace.path", trace.get("path"))
    scorecard = run.get("scorecard") if isinstance(run.get("scorecard"), dict) else {}
    _warn_absolute_public_path(target, f"{label}.scorecard.path", scorecard.get("path"))
    replay = run.get("replay") if isinstance(run.get("replay"), dict) else {}
    _warn_absolute_public_path(target, f"{label}.replay.lineage", replay.get("lineage"))
    _warn_replay_metadata_public_paths(replay, target, f"{label}.replay")

def _warn_replay_metadata_public_paths(replay: dict[str, Any], target: ValidationTarget, label: str) -> None:
    argv = replay.get("argv")
    if isinstance(argv, list):
        for index, item in enumerate(argv):
            _warn_command_token_public_path(target, f"{label}.argv[{index}]", item)
    command = replay.get("command")
    if not isinstance(command, str) or not command:
        return
    _warn_shell_tokens_public_paths(command, target, f"{label}.command")

def _warn_shell_tokens_public_paths(command: str, target: ValidationTarget, label: str) -> None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for index, token in enumerate(tokens):
        _warn_command_token_public_path(target, f"{label}[{index}]", token)

def _warn_command_token_public_path(target: ValidationTarget, label: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        return
    if _looks_absolute(value):
        _warn_absolute_public_path(target, label, value)
        return
    _, separator, token_value = value.partition("=")
    if separator and _looks_absolute(token_value):
        target.warnings.append(f"{label} contains absolute path; use relative or redacted command tokens for public bundles.")

def _resolve_harness_suite_artifact_path(value: Any, source_dir: Path | None) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or source_dir is None:
        return path
    return source_dir / path

def _harness_path_uses_symlink(path: Path) -> bool:
    return _path_has_symlink_component(path, include_leaf=False) or path.is_symlink()

def _validate_live_smoke_summary(summary: dict[str, Any], target: ValidationTarget) -> None:
    schema_version = summary.get("schema_version")
    allowed_versions = {LIVE_SMOKE_SUMMARY_SCHEMA_VERSION, *LEGACY_LIVE_SMOKE_SUMMARY_SCHEMA_VERSIONS}
    if schema_version not in allowed_versions:
        target.errors.append(
            f"live_smoke_summary.schema_version expected one of {sorted(allowed_versions)!r}, got {schema_version!r}."
        )
    if schema_version in LEGACY_LIVE_SMOKE_SUMMARY_SCHEMA_VERSIONS:
        target.warnings.append(
            "live_smoke_summary is a legacy schema without required runtime provenance; regenerate it with the current live smoke script."
        )
    if not isinstance(summary.get("passed"), bool):
        target.errors.append("live_smoke_summary.passed must be a boolean.")
    score = summary.get("score")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
        target.errors.append("live_smoke_summary.score must be an integer from 0 to 100.")
        score = 0
    for field_name in ("hermes_exit_code", "mock_request_count", "chat_completion_request_count"):
        if not _is_non_negative_int(summary.get(field_name)):
            target.errors.append(f"live_smoke_summary.{field_name} must be a non-negative integer.")
    required_paths = ["observer_file", "report", "lineage", "task_completion", "summary"]
    if schema_version == LIVE_SMOKE_SUMMARY_SCHEMA_VERSION:
        required_paths.append("run_digest")
    for field_name in required_paths:
        if not isinstance(summary.get(field_name), str) or not summary.get(field_name):
            target.errors.append(f"live_smoke_summary.{field_name} must be a non-empty string.")
        else:
            _warn_absolute_public_path(target, f"live_smoke_summary.{field_name}", summary.get(field_name))
    hooks = summary.get("hooks")
    if not _is_string_list(hooks):
        target.errors.append("live_smoke_summary.hooks must be a list of strings.")
        hooks = []
    missing_hooks = summary.get("missing_hooks")
    if not _is_string_list(missing_hooks):
        target.errors.append("live_smoke_summary.missing_hooks must be a list of strings.")
        missing_hooks = []
    environment = summary.get("environment")
    environment_details: dict[str, Any] = {}
    if schema_version == LIVE_SMOKE_SUMMARY_SCHEMA_VERSION:
        environment_details = _validate_live_smoke_environment(environment, target)
    elif isinstance(environment, dict):
        environment_details = _validate_live_smoke_environment(environment, target)

    expected_passed = (
        summary.get("hermes_exit_code") == 0
        and isinstance(score, int)
        and score >= 90
        and not missing_hooks
        and _is_non_negative_int(summary.get("chat_completion_request_count"))
        and int(summary.get("chat_completion_request_count")) > 0
    )
    if isinstance(summary.get("passed"), bool) and summary.get("passed") != expected_passed:
        target.errors.append(
            "live_smoke_summary.passed must match exit code, score, missing hooks, and chat completion request count."
        )
    target.details.update(
        {
            "passed": summary.get("passed"),
            "score": summary.get("score"),
            "hook_count": len(hooks),
            "missing_hook_count": len(missing_hooks),
            "chat_completion_request_count": summary.get("chat_completion_request_count"),
            **environment_details,
        }
    )

def _validate_live_smoke_environment(environment: Any, target: ValidationTarget) -> dict[str, Any]:
    if not isinstance(environment, dict):
        target.errors.append("live_smoke_summary.environment must be an object.")
        return {}
    details: dict[str, Any] = {}
    required_strings = (
        "python_version",
        "python_implementation",
        "platform",
        "hermes_root",
        "hermes_git_commit",
        "flight_recorder_root",
        "flight_recorder_git_commit",
    )
    for field_name in required_strings:
        value = environment.get(field_name)
        if not isinstance(value, str) or not value:
            target.errors.append(f"live_smoke_summary.environment.{field_name} must be a non-empty string.")
    for field_name in ("hermes_root", "flight_recorder_root"):
        _warn_absolute_public_path(target, f"live_smoke_summary.environment.{field_name}", environment.get(field_name))
    for field_name in ("hermes_git_dirty", "flight_recorder_git_dirty"):
        value = environment.get(field_name)
        if value is not None and not isinstance(value, bool):
            target.errors.append(f"live_smoke_summary.environment.{field_name} must be a boolean or null.")
    for field_name in ("platform", "hermes_git_commit", "flight_recorder_git_commit"):
        value = environment.get(field_name)
        if isinstance(value, str) and value:
            details[field_name] = value
    return details

def _validate_state_snapshot(
    snapshot: dict[str, Any],
    target: ValidationTarget,
    label: str,
    *,
    source_path: Path | None = None,
) -> None:
    _require_equal(snapshot, "schema_version", STATE_SNAPSHOT_SCHEMA_VERSION, target, prefix=f"{label}.")
    filesystem = snapshot.get("filesystem")
    if not isinstance(filesystem, dict):
        target.errors.append(f"{label}.filesystem must be an object.")
        filesystem = {}
    files = filesystem.get("files")
    if not isinstance(files, dict):
        target.errors.append(f"{label}.filesystem.files must be an object.")
        files = {}
    directories = filesystem.get("directories")
    if not isinstance(directories, dict):
        target.errors.append(f"{label}.filesystem.directories must be an object.")
        directories = {}
    for key, record in files.items():
        if not isinstance(key, str) or not key:
            target.errors.append(f"{label}.filesystem.files contains an empty source key.")
            continue
        _validate_snapshot_file_record(record, target, f"{label}.filesystem.files.{key}", source_path)
    for key, record in directories.items():
        if not isinstance(key, str) or not key:
            target.errors.append(f"{label}.filesystem.directories contains an empty source key.")
            continue
        _validate_snapshot_directory_record(record, target, f"{label}.filesystem.directories.{key}")

    json_sources = snapshot.get("json_sources")
    if not isinstance(json_sources, dict):
        target.errors.append(f"{label}.json_sources must be an object.")
        json_sources = {}
    json_payloads = snapshot.get("json")
    if not isinstance(json_payloads, dict):
        target.errors.append(f"{label}.json must be an object.")
        json_payloads = {}
    for key, record in json_sources.items():
        if not isinstance(key, str) or not key:
            target.errors.append(f"{label}.json_sources contains an empty source key.")
            continue
        _validate_snapshot_file_record(record, target, f"{label}.json_sources.{key}", source_path)
        if isinstance(record, dict) and record.get("exists") is True and record.get("kind") == "file" and key not in json_payloads:
            target.errors.append(f"{label}.json.{key} must contain imported JSON for an existing JSON source.")
    observations = snapshot.get("observations")
    if not isinstance(observations, dict):
        target.errors.append(f"{label}.observations must be an object.")
    verifiers = snapshot.get("verifiers")
    if verifiers is not None:
        _validate_snapshot_verifiers(verifiers, target, f"{label}.verifiers")

def _validate_snapshot_verifiers(verifiers: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(verifiers, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _require_equal(verifiers, "schema_version", VERIFIER_SOURCES_SCHEMA_VERSION, target, prefix=f"{label}.")
    sources = verifiers.get("sources")
    if not isinstance(sources, dict):
        target.errors.append(f"{label}.sources must be an object.")
        sources = {}
    source_count = verifiers.get("source_count")
    if not isinstance(source_count, int):
        target.errors.append(f"{label}.source_count must be an integer.")
    elif source_count != len(sources):
        target.errors.append(f"{label}.source_count must equal the number of sources.")
    for source_id, source in sources.items():
        source_label = f"{label}.sources.{source_id}"
        if not isinstance(source_id, str) or not source_id:
            target.errors.append(f"{label}.sources contains an empty source id.")
            continue
        if not isinstance(source, dict):
            target.errors.append(f"{source_label} must be an object.")
            continue
        if not isinstance(source.get("type"), str) or not source.get("type"):
            target.errors.append(f"{source_label}.type must be a non-empty string.")
        if source.get("status") not in {"ok", "error"}:
            target.errors.append(f"{source_label}.status must be ok or error.")
        if source.get("readonly") is not True:
            target.errors.append(f"{source_label}.readonly must be true.")
        if "data" not in source:
            target.errors.append(f"{source_label}.data is required.")

def _validate_state_diff(diff: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(diff, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _require_equal(diff, "schema_version", STATE_DIFF_SCHEMA_VERSION, target, prefix=f"{label}.")
    changed = diff.get("changed")
    if not isinstance(changed, bool):
        target.errors.append(f"{label}.changed must be a boolean.")
        changed = False
    change_count = diff.get("change_count")
    if not _is_non_negative_int(change_count):
        target.errors.append(f"{label}.change_count must be a non-negative integer.")
        change_count = 0
    max_changes = diff.get("max_changes")
    if not _is_non_negative_int(max_changes):
        target.errors.append(f"{label}.max_changes must be a non-negative integer.")
        max_changes = 0
    truncated = diff.get("truncated")
    if not isinstance(truncated, bool):
        target.errors.append(f"{label}.truncated must be a boolean.")
        truncated = False
    comparison_truncated = diff.get("comparison_truncated", False)
    if not isinstance(comparison_truncated, bool):
        target.errors.append(f"{label}.comparison_truncated must be a boolean when present.")
        comparison_truncated = False
    comparison_complete = diff.get("comparison_complete")
    if not isinstance(comparison_complete, bool):
        target.errors.append(f"{label}.comparison_complete must be a boolean.")
        comparison_complete = False
    elif comparison_complete == comparison_truncated:
        target.errors.append(f"{label}.comparison_complete must be the inverse of comparison_truncated.")
    expected_change_status = "changed" if bool(change_count) else "unknown" if not comparison_complete else "unchanged"
    change_status = diff.get("change_status")
    if change_status != expected_change_status:
        target.errors.append(
            f"{label}.change_status expected {expected_change_status!r}, got {change_status!r}."
        )
    if comparison_truncated:
        if diff.get("change_count_exact") is not False:
            target.errors.append(f"{label}.change_count_exact must be false when comparison_truncated is true.")
        if diff.get("truncation_reason") not in {
            "incomplete_snapshot",
            "max_changes",
            "max_depth",
            "max_nodes",
        }:
            target.errors.append(f"{label}.truncation_reason must identify why comparison stopped.")
    elif "truncation_reason" in diff:
        target.errors.append(f"{label}.truncation_reason is only valid for an incomplete comparison.")
    changes = diff.get("changes")
    if not isinstance(changes, list):
        target.errors.append(f"{label}.changes must be a list.")
        changes = []
    if changed != bool(change_count):
        target.errors.append(f"{label}.changed must match whether change_count is greater than zero.")
    if isinstance(change_count, int) and isinstance(max_changes, int):
        if len(changes) > max_changes:
            target.errors.append(f"{label}.changes length must not exceed max_changes.")
        expected_truncated = change_count > len(changes) or comparison_truncated
        if truncated != expected_truncated:
            target.errors.append(f"{label}.truncated must match whether change_count exceeds emitted changes.")
        if not truncated and len(changes) != change_count:
            target.errors.append(f"{label}.changes length must equal change_count when not truncated.")
    for index, change in enumerate(changes):
        _validate_state_diff_change(change, target, f"{label}.changes[{index}]")
    if not isinstance(diff.get("summary"), str) or not diff.get("summary"):
        target.errors.append(f"{label}.summary must be a non-empty string.")

def _validate_state_diff_change(change: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(change, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if not isinstance(change.get("path"), str) or not change.get("path"):
        target.errors.append(f"{label}.path must be a non-empty string.")
    if change.get("kind") not in {"added", "removed", "changed"}:
        target.errors.append(f"{label}.kind must be added, removed, or changed.")
    if "before" not in change:
        target.errors.append(f"{label}.before is required.")
    if "after" not in change:
        target.errors.append(f"{label}.after is required.")

def _validate_state_diff_summary(summary: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(summary, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _require_equal(summary, "schema_version", "hfr.state_diff.summary.v1", target, prefix=f"{label}.")
    if not isinstance(summary.get("available"), bool):
        target.errors.append(f"{label}.available must be a boolean.")
    if not isinstance(summary.get("changed"), bool):
        target.errors.append(f"{label}.changed must be a boolean.")
    if not _is_non_negative_int(summary.get("change_count")):
        target.errors.append(f"{label}.change_count must be a non-negative integer.")
    if not isinstance(summary.get("truncated"), bool):
        target.errors.append(f"{label}.truncated must be a boolean.")
    comparison_complete = summary.get("comparison_complete")
    if comparison_complete is not True:
        target.errors.append(f"{label}.comparison_complete must be true for training artifacts.")
    expected_status = (
        "unavailable"
        if summary.get("available") is False
        else "changed"
        if summary.get("changed") is True
        else "unchanged"
    )
    if summary.get("change_status") != expected_status:
        target.errors.append(
            f"{label}.change_status expected {expected_status!r}, got {summary.get('change_status')!r}."
        )
    if not isinstance(summary.get("summary"), str):
        target.errors.append(f"{label}.summary must be a string.")
    changes = summary.get("changes")
    if not isinstance(changes, list):
        target.errors.append(f"{label}.changes must be a list.")
        return
    for index, change in enumerate(changes):
        if not isinstance(change, dict):
            target.errors.append(f"{label}.changes[{index}] must be an object.")
            continue
        if not isinstance(change.get("path"), str):
            target.errors.append(f"{label}.changes[{index}].path must be a string.")
        if change.get("kind") not in {"", "added", "removed", "changed"}:
            target.errors.append(f"{label}.changes[{index}].kind must be added, removed, changed, or empty.")

def _validate_run_digest(
    digest: Any,
    target: ValidationTarget,
    label: str,
    *,
    trace: dict[str, Any] | None = None,
    scorecard: dict[str, Any] | None = None,
    task_completion: dict[str, Any] | None = None,
    state_diff: Any = _COMPANION_NOT_PROVIDED,
) -> None:
    if not isinstance(digest, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _require_equal(digest, "schema_version", RUN_DIGEST_SCHEMA_VERSION, target, prefix=f"{label}.")

    scenario = digest.get("scenario")
    if not isinstance(scenario, dict):
        target.errors.append(f"{label}.scenario must be an object.")
        scenario = {}
    for field_name in ("id", "title", "task_family"):
        if not isinstance(scenario.get(field_name), str) or not scenario.get(field_name):
            target.errors.append(f"{label}.scenario.{field_name} must be a non-empty string.")
    if scorecard is not None:
        if scenario.get("id") != scorecard.get("scenario_id"):
            target.errors.append(f"{label}.scenario.id must match scorecard.scenario_id.")
        if scenario.get("title") != scorecard.get("scenario_title"):
            target.errors.append(f"{label}.scenario.title must match scorecard.scenario_title.")

    outcome = digest.get("outcome")
    if not isinstance(outcome, dict):
        target.errors.append(f"{label}.outcome must be an object.")
        outcome = {}
    _validate_run_digest_outcome(outcome, target, f"{label}.outcome", scorecard, task_completion)

    trace_signal = digest.get("trace_signal")
    if not isinstance(trace_signal, dict):
        target.errors.append(f"{label}.trace_signal must be an object.")
        trace_signal = {}
    _validate_run_digest_trace_signal(trace_signal, target, f"{label}.trace_signal", trace)

    state_changes = digest.get("state_changes")
    if not isinstance(state_changes, dict):
        target.errors.append(f"{label}.state_changes must be an object.")
        state_changes = {}
    _validate_run_digest_state_changes(state_changes, target, f"{label}.state_changes", state_diff)

    rules = digest.get("rules")
    if not isinstance(rules, dict):
        target.errors.append(f"{label}.rules must be an object.")
        rules = {}
    _validate_run_digest_rules(rules, target, f"{label}.rules", scorecard)

    evidence = digest.get("evidence")
    if not isinstance(evidence, dict):
        target.errors.append(f"{label}.evidence must be an object.")
        evidence = {}
    _validate_run_digest_evidence(evidence, target, f"{label}.evidence", scorecard, task_completion)

    training = digest.get("training_signals")
    if not isinstance(training, dict):
        target.errors.append(f"{label}.training_signals must be an object.")
        training = {}
    _validate_run_digest_training(training, target, f"{label}.training_signals", outcome, state_changes, rules)

    actions = digest.get("recommended_actions")
    if not isinstance(actions, list):
        target.errors.append(f"{label}.recommended_actions must be a list.")
    else:
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                target.errors.append(f"{label}.recommended_actions[{index}] must be an object.")
                continue
            for field_name in ("id", "priority", "reason"):
                if not isinstance(action.get(field_name), str) or not action.get(field_name):
                    target.errors.append(f"{label}.recommended_actions[{index}].{field_name} must be a non-empty string.")

def _validate_run_digest_outcome(
    outcome: dict[str, Any],
    target: ValidationTarget,
    label: str,
    scorecard: dict[str, Any] | None,
    task_completion: dict[str, Any] | None,
) -> None:
    if not isinstance(outcome.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    if not _is_int_between(outcome.get("score"), 0, 100):
        target.errors.append(f"{label}.score must be an integer from 0 to 100.")
    if not _is_int_between(outcome.get("pass_threshold"), 0, 100):
        target.errors.append(f"{label}.pass_threshold must be an integer from 0 to 100.")
    if not _is_string_list(outcome.get("critical_failures")):
        target.errors.append(f"{label}.critical_failures must be a list of strings.")
    if not isinstance(outcome.get("summary"), str):
        target.errors.append(f"{label}.summary must be a string.")
    if outcome.get("task_completion_status") not in {"complete", "incomplete", "not_applicable"}:
        target.errors.append(f"{label}.task_completion_status must be complete, incomplete, or not_applicable.")
    if not isinstance(outcome.get("task_completion_passed"), bool):
        target.errors.append(f"{label}.task_completion_passed must be a boolean.")
    if scorecard is not None:
        for field_name in ("passed", "score", "pass_threshold", "critical_failures", "summary"):
            if outcome.get(field_name) != scorecard.get(field_name):
                target.errors.append(f"{label}.{field_name} must match scorecard.{field_name}.")
    task = task_completion
    if task is None and isinstance(scorecard, dict) and isinstance(scorecard.get("task_completion"), dict):
        task = scorecard["task_completion"]
    if isinstance(task, dict):
        if outcome.get("task_completion_status") != task.get("status"):
            target.errors.append(f"{label}.task_completion_status must match task_completion.status.")
        if outcome.get("task_completion_passed") != task.get("passed"):
            target.errors.append(f"{label}.task_completion_passed must match task_completion.passed.")

def _validate_run_digest_trace_signal(
    signal: dict[str, Any],
    target: ValidationTarget,
    label: str,
    trace: dict[str, Any] | None,
) -> None:
    for field_name in (
        "event_count",
        "tool_call_count",
        "tool_result_count",
        "api_call_count",
        "subagent_start_count",
        "max_subagent_depth",
    ):
        if not _is_non_negative_int(signal.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    if not _is_string_list(signal.get("event_types")):
        target.errors.append(f"{label}.event_types must be a list of strings.")
    for field_name in ("has_final_answer", "has_tool_or_api_events"):
        if not isinstance(signal.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    for field_name in ("source_format", "model"):
        if not isinstance(signal.get(field_name), str) or not signal.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    if trace is None:
        return
    expected = _run_digest_trace_signal(trace)
    for field_name, expected_value in expected.items():
        if signal.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} expected {expected_value!r}, got {signal.get(field_name)!r}.")

def _validate_run_digest_state_changes(
    state_changes: dict[str, Any],
    target: ValidationTarget,
    label: str,
    state_diff: Any,
) -> None:
    for field_name in ("available", "changed", "truncated", "comparison_complete"):
        if not isinstance(state_changes.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if state_changes.get("change_status") not in {"changed", "unchanged", "unknown"}:
        target.errors.append(f"{label}.change_status must be changed, unchanged, or unknown.")
    if not _is_non_negative_int(state_changes.get("change_count")):
        target.errors.append(f"{label}.change_count must be a non-negative integer.")
    if not isinstance(state_changes.get("summary"), str):
        target.errors.append(f"{label}.summary must be a string.")
    top_changes = state_changes.get("top_changes")
    if not isinstance(top_changes, list):
        target.errors.append(f"{label}.top_changes must be a list.")
        top_changes = []
    for index, change in enumerate(top_changes):
        if not isinstance(change, dict):
            target.errors.append(f"{label}.top_changes[{index}] must be an object.")
            continue
        if not isinstance(change.get("path"), str):
            target.errors.append(f"{label}.top_changes[{index}].path must be a string.")
        if not isinstance(change.get("kind"), str):
            target.errors.append(f"{label}.top_changes[{index}].kind must be a string.")
    if state_diff is _COMPANION_NOT_PROVIDED:
        return
    if state_diff is None:
        if state_changes.get("available") is not False:
            target.errors.append(f"{label}.available must be false when state_diff.json is absent.")
        if state_changes.get("comparison_complete") is not False:
            target.errors.append(f"{label}.comparison_complete must be false when state_diff.json is absent.")
        if state_changes.get("change_status") != "unknown":
            target.errors.append(f"{label}.change_status must be unknown when state_diff.json is absent.")
        return
    if state_changes.get("available") is not True:
        target.errors.append(f"{label}.available must be true when state_diff.json is present.")
    comparison_complete, change_status = resolve_state_diff_semantics(state_diff)
    expected_fields = {
        "changed": state_diff.get("changed"),
        "change_count": state_diff.get("change_count"),
        "truncated": state_diff.get("truncated"),
        "comparison_complete": comparison_complete,
        "change_status": change_status,
        "summary": state_diff.get("summary"),
    }
    for field_name, expected_value in expected_fields.items():
        if state_changes.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} must match state_diff.{field_name}.")
    expected_top = [
        {"path": str(change.get("path") or ""), "kind": str(change.get("kind") or "")}
        for change in (state_diff.get("changes") if isinstance(state_diff.get("changes"), list) else [])[:10]
        if isinstance(change, dict)
    ]
    if top_changes != expected_top:
        target.errors.append(f"{label}.top_changes must match the first state_diff changes by path/kind.")

def _validate_run_digest_rules(
    rules: dict[str, Any],
    target: ValidationTarget,
    label: str,
    scorecard: dict[str, Any] | None,
) -> None:
    for field_name in ("total_count", "failed_count", "critical_failed_count"):
        if not _is_non_negative_int(rules.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    failed = rules.get("failed")
    if not isinstance(failed, list):
        target.errors.append(f"{label}.failed must be a list.")
        failed = []
    for index, rule in enumerate(failed):
        if not isinstance(rule, dict):
            target.errors.append(f"{label}.failed[{index}] must be an object.")
            continue
        for field_name in ("id", "name"):
            if not isinstance(rule.get(field_name), str) or not rule.get(field_name):
                target.errors.append(f"{label}.failed[{index}].{field_name} must be a non-empty string.")
        if not isinstance(rule.get("critical"), bool):
            target.errors.append(f"{label}.failed[{index}].critical must be a boolean.")
        if not _is_int_between(rule.get("penalty"), 0, 100):
            target.errors.append(f"{label}.failed[{index}].penalty must be an integer from 0 to 100.")
        if not _is_non_negative_int(rule.get("evidence_ref_count")):
            target.errors.append(f"{label}.failed[{index}].evidence_ref_count must be a non-negative integer.")
        if not isinstance(rule.get("evidence"), list):
            target.errors.append(f"{label}.failed[{index}].evidence must be a list.")
        if not isinstance(rule.get("evidence_refs"), list):
            target.errors.append(f"{label}.failed[{index}].evidence_refs must be a list.")
    if scorecard is None:
        return
    score_rules = [rule for rule in scorecard.get("rules", []) if isinstance(rule, dict)]
    failed_score_rules = [rule for rule in score_rules if rule.get("passed") is False]
    if rules.get("total_count") != len(score_rules):
        target.errors.append(f"{label}.total_count must match scorecard.rules length.")
    if rules.get("failed_count") != len(failed_score_rules):
        target.errors.append(f"{label}.failed_count must match failed scorecard rules.")
    critical_failed = sum(1 for rule in failed_score_rules if rule.get("critical") is True)
    if rules.get("critical_failed_count") != critical_failed:
        target.errors.append(f"{label}.critical_failed_count must match failed critical scorecard rules.")
    expected_failed_ids = [str(rule.get("id") or "unknown") for rule in failed_score_rules]
    actual_failed_ids = [str(rule.get("id") or "unknown") for rule in failed if isinstance(rule, dict)]
    if actual_failed_ids != expected_failed_ids:
        target.errors.append(f"{label}.failed ids must match scorecard failed rule order.")

def _validate_run_digest_evidence(
    evidence: dict[str, Any],
    target: ValidationTarget,
    label: str,
    scorecard: dict[str, Any] | None,
    task_completion: dict[str, Any] | None,
) -> None:
    for field_name in (
        "rule_evidence_ref_count",
        "failed_rule_evidence_ref_count",
        "critical_failed_rule_evidence_ref_count",
        "task_completion_evidence_ref_count",
        "missing_evidence_ref_count",
        "total_evidence_ref_count",
    ):
        if not _is_non_negative_int(evidence.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    if scorecard is None:
        return
    expected = _run_digest_evidence_counts(scorecard, task_completion)
    for field_name, expected_value in expected.items():
        if evidence.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} expected {expected_value}, got {evidence.get(field_name)!r}.")

def _validate_run_digest_training(
    training: dict[str, Any],
    target: ValidationTarget,
    label: str,
    outcome: dict[str, Any],
    state_changes: dict[str, Any],
    rules: dict[str, Any],
) -> None:
    reward = training.get("score_reward")
    if not isinstance(reward, (int, float)) or isinstance(reward, bool) or not 0 <= float(reward) <= 1:
        target.errors.append(f"{label}.score_reward must be numeric from 0 to 1.")
    if training.get("binary_reward") not in {0, 1}:
        target.errors.append(f"{label}.binary_reward must be 0 or 1.")
    if training.get("task_completion_reward") not in {0, 1}:
        target.errors.append(f"{label}.task_completion_reward must be 0 or 1.")
    if training.get("task_completion_status") != outcome.get("task_completion_status"):
        target.errors.append(f"{label}.task_completion_status must match outcome.task_completion_status.")
    if training.get("task_completion_passed") != outcome.get("task_completion_passed"):
        target.errors.append(f"{label}.task_completion_passed must match outcome.task_completion_passed.")
    if training.get("state_changed") != state_changes.get("changed"):
        target.errors.append(f"{label}.state_changed must match state_changes.changed.")
    if training.get("state_change_count") != state_changes.get("change_count"):
        target.errors.append(f"{label}.state_change_count must match state_changes.change_count.")
    expected_binary = 1 if outcome.get("passed") is True else 0
    if training.get("binary_reward") != expected_binary:
        target.errors.append(f"{label}.binary_reward must match outcome.passed.")
    expected_task_reward = 1 if outcome.get("task_completion_passed") is True else 0
    if training.get("task_completion_reward") != expected_task_reward:
        target.errors.append(f"{label}.task_completion_reward must match outcome.task_completion_passed.")
    failure_modes = training.get("failure_modes")
    if not isinstance(failure_modes, list):
        target.errors.append(f"{label}.failure_modes must be a list.")
    elif _is_non_negative_int(rules.get("failed_count")) and len(failure_modes) != rules.get("failed_count"):
        target.errors.append(f"{label}.failure_modes length must match rules.failed_count.")

def _validate_snapshot_file_record(
    record: Any,
    target: ValidationTarget,
    label: str,
    source_path: Path | None,
) -> None:
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    path_label = record.get("path")
    if not isinstance(path_label, str) or not path_label:
        target.errors.append(f"{label}.path must be a non-empty string.")
    exists = record.get("exists")
    if not isinstance(exists, bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    kind = record.get("kind")
    if kind not in {"missing", "file", "directory", "other"}:
        target.errors.append(f"{label}.kind must be missing, file, directory, or other.")
    if exists is False and kind != "missing":
        target.errors.append(f"{label}.kind must be missing when exists is false.")
    if exists is True and kind == "missing":
        target.errors.append(f"{label}.kind must not be missing when exists is true.")
    if kind == "file":
        if not _is_non_negative_int(record.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for files.")
        if not _is_sha256(record.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a 64-character hex digest for files.")
        if "text" in record and not isinstance(record.get("text"), str):
            target.errors.append(f"{label}.text must be a string when present.")
        if "text_truncated" in record and not isinstance(record.get("text_truncated"), bool):
            target.errors.append(f"{label}.text_truncated must be a boolean when present.")
        _validate_snapshot_record_fingerprint(record, target, label, source_path)

def _validate_snapshot_directory_record(record: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    path_label = record.get("path")
    if not isinstance(path_label, str) or not path_label:
        target.errors.append(f"{label}.path must be a non-empty string.")
    exists = record.get("exists")
    if not isinstance(exists, bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    kind = record.get("kind")
    if kind not in {"missing", "directory", "not_directory"}:
        target.errors.append(f"{label}.kind must be missing, directory, or not_directory.")
    if kind == "directory":
        entry_count = record.get("entry_count")
        if not _is_non_negative_int(entry_count):
            target.errors.append(f"{label}.entry_count must be a non-negative integer.")
            entry_count = None
        if not isinstance(record.get("entries_truncated"), bool):
            target.errors.append(f"{label}.entries_truncated must be a boolean.")
        entries = record.get("entries")
        if not isinstance(entries, list):
            target.errors.append(f"{label}.entries must be a list.")
            entries = []
        if isinstance(entry_count, int) and len(entries) > entry_count:
            target.errors.append(f"{label}.entries length must not exceed entry_count.")
        scan_fields = {
            "entry_count_is_lower_bound",
            "scanned_entry_count",
            "scan_limit",
            "scan_incomplete",
            "entry_selection",
        }
        if any(field_name in record for field_name in scan_fields):
            missing_scan_fields = sorted(field_name for field_name in scan_fields if field_name not in record)
            if missing_scan_fields:
                target.errors.append(
                    f"{label} bounded-scan metadata is incomplete; missing {missing_scan_fields!r}."
                )
            scan_incomplete = record.get("scan_incomplete")
            lower_bound = record.get("entry_count_is_lower_bound")
            scanned_entry_count = record.get("scanned_entry_count")
            scan_limit = record.get("scan_limit")
            if not isinstance(scan_incomplete, bool):
                target.errors.append(f"{label}.scan_incomplete must be a boolean.")
            if not isinstance(lower_bound, bool):
                target.errors.append(f"{label}.entry_count_is_lower_bound must be a boolean.")
            if not _is_non_negative_int(scanned_entry_count):
                target.errors.append(f"{label}.scanned_entry_count must be a non-negative integer.")
            if not _is_non_negative_int(scan_limit) or scan_limit < 1:
                target.errors.append(f"{label}.scan_limit must be a positive integer.")
            if record.get("entry_selection") != "lexicographic_within_scanned_prefix":
                target.errors.append(
                    f"{label}.entry_selection must be 'lexicographic_within_scanned_prefix'."
                )
            if isinstance(scan_incomplete, bool):
                if lower_bound is not scan_incomplete:
                    target.errors.append(
                        f"{label}.entry_count_is_lower_bound must match scan_incomplete."
                    )
                if record.get("entries_truncated") is not scan_incomplete:
                    target.errors.append(f"{label}.entries_truncated must match scan_incomplete.")
            if _is_non_negative_int(scanned_entry_count) and isinstance(entry_count, int):
                if entry_count != scanned_entry_count:
                    target.errors.append(f"{label}.entry_count must match scanned_entry_count.")
            if _is_non_negative_int(scanned_entry_count) and _is_non_negative_int(scan_limit) and scan_limit >= 1:
                if scanned_entry_count > scan_limit:
                    target.errors.append(f"{label}.scanned_entry_count must not exceed scan_limit.")
                expected_incomplete = scanned_entry_count == scan_limit
                if isinstance(scan_incomplete, bool) and scan_incomplete is not expected_incomplete:
                    target.errors.append(
                        f"{label}.scan_incomplete must match whether scanned_entry_count reached scan_limit."
                    )
                expected_entries = scanned_entry_count - 1 if expected_incomplete else scanned_entry_count
                if len(entries) != expected_entries:
                    target.errors.append(
                        f"{label}.entries length must match the bounded scan selection."
                    )
        for index, entry in enumerate(entries):
            _validate_snapshot_directory_entry(entry, target, f"{label}.entries[{index}]")

def _validate_snapshot_directory_entry(entry: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(entry, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if not isinstance(entry.get("name"), str) or not entry.get("name"):
        target.errors.append(f"{label}.name must be a non-empty string.")
    kind = entry.get("kind")
    if kind not in {"file", "directory", "other", "missing"}:
        target.errors.append(f"{label}.kind must be file, directory, other, or missing.")
    if kind == "file":
        if not _is_non_negative_int(entry.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for files.")
        if not _is_sha256(entry.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a 64-character hex digest for files.")

def _validate_snapshot_record_fingerprint(
    record: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path | None,
) -> None:
    path_label = record.get("path")
    if not isinstance(path_label, str) or path_label.startswith("<redacted:"):
        return
    path = Path(path_label)
    if not path.is_absolute():
        candidates = [path.resolve()]
        if source_path is not None:
            candidates.append((source_path.parent / path).resolve())
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not path.exists() or not path.is_file():
        return
    expected_size = record.get("size_bytes")
    if _is_non_negative_int(expected_size) and path.stat().st_size != expected_size:
        target.errors.append(f"{label}.size_bytes does not match current file size.")
    expected = record.get("sha256")
    if _is_sha256(expected):
        actual = _sha256(path)
        if actual != expected:
            target.errors.append(f"{label}.sha256 does not match current file contents.")

def _validate_trace(trace: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(trace, "schema_version", TRACE_SCHEMA_VERSION, target)
    session = trace.get("session")
    if not isinstance(session, dict):
        target.errors.append("trace.session must be an object.")
    else:
        for field_name in ("id", "source_format", "model"):
            if not isinstance(session.get(field_name), str) or not session.get(field_name):
                target.errors.append(f"trace.session.{field_name} must be a non-empty string.")
    events = trace.get("events")
    if not isinstance(events, list):
        target.errors.append("trace.events must be a list.")
        events = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            target.errors.append(f"trace.events[{index}] must be an object.")
            continue
        if not isinstance(event.get("type"), str) or not event.get("type"):
            target.errors.append(f"trace.events[{index}].type must be a non-empty string.")
        if "session_id" in event and event.get("session_id") is not None and not isinstance(event.get("session_id"), str):
            target.errors.append(f"trace.events[{index}].session_id must be a string or null.")
        if "args" in event and not isinstance(event.get("args"), dict):
            target.errors.append(f"trace.events[{index}].args must be an object when present.")
    if not isinstance(trace.get("final_answer", ""), str):
        target.errors.append("trace.final_answer must be a string.")

def _validate_scorecard(scorecard: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(scorecard, "schema_version", SCORE_SCHEMA_VERSION, target)
    for field_name in ("scenario_id", "scenario_title", "summary"):
        if not isinstance(scorecard.get(field_name), str) or not scorecard.get(field_name):
            target.errors.append(f"scorecard.{field_name} must be a non-empty string.")
    score = scorecard.get("score")
    threshold = scorecard.get("pass_threshold")
    if not _is_int_between(score, 0, 100):
        target.errors.append("scorecard.score must be an integer from 0 to 100.")
    if not _is_int_between(threshold, 0, 100):
        target.errors.append("scorecard.pass_threshold must be an integer from 0 to 100.")
    if not isinstance(scorecard.get("passed"), bool):
        target.errors.append("scorecard.passed must be a boolean.")
    critical_failures = scorecard.get("critical_failures")
    if not _is_string_list(critical_failures):
        target.errors.append("scorecard.critical_failures must be a list of strings.")
        critical_failures = []
    rules = scorecard.get("rules")
    if not isinstance(rules, list):
        target.errors.append("scorecard.rules must be a list.")
        rules = []

    failed_critical: list[str] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            target.errors.append(f"scorecard.rules[{index}] must be an object.")
            continue
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            target.errors.append(f"scorecard.rules[{index}].id must be a non-empty string.")
            rule_id = f"rule[{index}]"
        if not isinstance(rule.get("name"), str) or not rule.get("name"):
            target.errors.append(f"scorecard.rules[{index}].name must be a non-empty string.")
        if not isinstance(rule.get("passed"), bool):
            target.errors.append(f"scorecard.rules[{index}].passed must be a boolean.")
        if not isinstance(rule.get("critical"), bool):
            target.errors.append(f"scorecard.rules[{index}].critical must be a boolean.")
        if not _is_int_between(rule.get("penalty"), 0, 100):
            target.errors.append(f"scorecard.rules[{index}].penalty must be an integer from 0 to 100.")
        if not isinstance(rule.get("evidence"), list):
            target.errors.append(f"scorecard.rules[{index}].evidence must be a list.")
        if "evidence_refs" in rule:
            _validate_evidence_refs(rule.get("evidence_refs"), target, f"scorecard.rules[{index}].evidence_refs")
        if rule.get("critical") is True and rule.get("passed") is False:
            failed_critical.append(str(rule_id))

    if isinstance(critical_failures, list) and sorted(critical_failures) != sorted(failed_critical):
        target.errors.append(
            "scorecard.critical_failures must match failed critical rules: "
            f"expected {sorted(failed_critical)!r}, got {sorted(critical_failures)!r}."
        )
    if _is_int_between(score, 0, 100) and _is_int_between(threshold, 0, 100) and isinstance(scorecard.get("passed"), bool):
        expected_passed = int(score) >= int(threshold) and not failed_critical
        if scorecard["passed"] != expected_passed:
            target.errors.append(
                f"scorecard.passed inconsistent with score/threshold/critical failures: expected {expected_passed!r}."
            )
    if "task_completion" in scorecard:
        _validate_task_completion(scorecard.get("task_completion"), target, "scorecard.task_completion")
    else:
        target.warnings.append("scorecard.task_completion is missing; rerun scoring to emit task-completion evidence.")

def _validate_task_completion(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _require_equal(value, "schema_version", TASK_COMPLETION_SCHEMA_VERSION, target, prefix=f"{label}.")
    status = value.get("status")
    if status not in {"complete", "incomplete", "not_applicable"}:
        target.errors.append(f"{label}.status must be complete, incomplete, or not_applicable.")
    passed = value.get("passed")
    if not isinstance(passed, bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    elif status == "complete" and passed is not True:
        target.errors.append(f"{label}.passed must be true when status is complete.")
    elif status == "incomplete" and passed is not False:
        target.errors.append(f"{label}.passed must be false when status is incomplete.")
    elif status == "not_applicable" and passed is not True:
        target.errors.append(f"{label}.passed must be true when status is not_applicable.")
    configured = value.get("task_evidence_configured")
    if not isinstance(configured, bool):
        target.errors.append(f"{label}.task_evidence_configured must be a boolean.")
    elif configured is False and status != "not_applicable":
        target.errors.append(f"{label}.status must be not_applicable when no task evidence is configured.")
    elif configured is True and status == "not_applicable":
        target.errors.append(f"{label}.status must not be not_applicable when task evidence is configured.")
    counts: dict[str, int] = {}
    for field_name in ("required_check_count", "passed_check_count", "failed_check_count"):
        raw_count = value.get(field_name)
        if not _is_non_negative_int(raw_count):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
            continue
        counts[field_name] = int(raw_count)
    if len(counts) == 3:
        if counts["passed_check_count"] + counts["failed_check_count"] != counts["required_check_count"]:
            target.errors.append(f"{label}.passed_check_count + failed_check_count must equal required_check_count.")
        if counts["failed_check_count"] == 0 and status == "incomplete":
            target.errors.append(f"{label}.status must not be incomplete when failed_check_count is 0.")
        if counts["failed_check_count"] > 0 and status == "complete":
            target.errors.append(f"{label}.status must not be complete when failed_check_count is greater than 0.")
    if not _is_string_list(value.get("blocking_rule_ids")):
        target.errors.append(f"{label}.blocking_rule_ids must be a list of strings.")
    if not isinstance(value.get("summary"), str) or not value.get("summary"):
        target.errors.append(f"{label}.summary must be a non-empty string.")
    checks = value.get("checks")
    if not isinstance(checks, list):
        target.errors.append(f"{label}.checks must be a list.")
        checks = []
    if "required_check_count" in counts and len(checks) != counts["required_check_count"]:
        target.errors.append(f"{label}.checks length must match required_check_count.")
    observed_failed = 0
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            target.errors.append(f"{label}.checks[{index}] must be an object.")
            continue
        for field_name in ("id", "rule_id", "description", "evidence"):
            if not isinstance(check.get(field_name), str) or not check.get(field_name):
                target.errors.append(f"{label}.checks[{index}].{field_name} must be a non-empty string.")
        if not isinstance(check.get("passed"), bool):
            target.errors.append(f"{label}.checks[{index}].passed must be a boolean.")
        elif check.get("passed") is False:
            observed_failed += 1
        if "event_indices" in check and not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in check.get("event_indices", [])):
            target.errors.append(f"{label}.checks[{index}].event_indices must be a list of non-negative integers.")
        _validate_evidence_refs(check.get("evidence_refs"), target, f"{label}.checks[{index}].evidence_refs")
    if "failed_check_count" in counts and observed_failed != counts["failed_check_count"]:
        target.errors.append(f"{label}.failed_check_count must match failed checks.")
    _validate_evidence_refs(value.get("evidence_refs"), target, f"{label}.evidence_refs")
    _validate_evidence_refs(value.get("missing_evidence_refs"), target, f"{label}.missing_evidence_refs")

def _validate_lineage(
    lineage: dict[str, Any],
    target: ValidationTarget,
    run_dir: Path,
    trace: dict[str, Any] | None,
    scorecard: dict[str, Any] | None,
) -> None:
    _require_equal(lineage, "schema_version", LINEAGE_SCHEMA_VERSION, target, prefix="artifact_lineage.")
    scenario = lineage.get("scenario")
    if not isinstance(scenario, dict):
        target.errors.append("artifact_lineage.scenario must be an object.")
        scenario = {}
    if scorecard is not None and scenario.get("id") != scorecard.get("scenario_id"):
        target.errors.append("artifact_lineage.scenario.id must match scorecard.scenario_id.")
    lineage_trace = lineage.get("trace")
    if not isinstance(lineage_trace, dict):
        target.errors.append("artifact_lineage.trace must be an object.")
        lineage_trace = {}
    events = trace.get("events", []) if isinstance(trace, dict) and isinstance(trace.get("events"), list) else []
    if trace is not None and lineage_trace.get("event_count") != len(events):
        target.errors.append("artifact_lineage.trace.event_count must match normalized_trace.events.")
    lineage_scorecard = lineage.get("scorecard")
    if not isinstance(lineage_scorecard, dict):
        target.errors.append("artifact_lineage.scorecard must be an object.")
        lineage_scorecard = {}
    if scorecard is not None:
        for field_name in ("score", "passed", "critical_failures"):
            if lineage_scorecard.get(field_name) != scorecard.get(field_name):
                target.errors.append(f"artifact_lineage.scorecard.{field_name} must match scorecard.{field_name}.")
    inputs = _lineage_records(lineage.get("inputs"), target, "artifact_lineage.inputs")
    for name, record in inputs.items():
        _validate_lineage_input_record(name, record, target, f"artifact_lineage.inputs.{name}")
    outputs = _lineage_records(lineage.get("outputs"), target, "artifact_lineage.outputs")
    for output_name in ("normalized_trace", "scorecard", "task_completion", "report"):
        if output_name not in outputs:
            target.errors.append(f"artifact_lineage.outputs missing {output_name!r}.")
            continue
        _validate_lineage_file_record(outputs[output_name], run_dir, target, f"artifact_lineage.outputs.{output_name}")
    trajectory_v2_path = run_dir / "trajectory_v2.json"
    trajectory_v2_record = outputs.get("trajectory_v2")
    if trajectory_v2_path.exists() and trajectory_v2_record is None:
        target.errors.append("artifact_lineage.outputs missing 'trajectory_v2'.")
    if trajectory_v2_record is not None:
        path_label = trajectory_v2_record.get("path")
        if not isinstance(path_label, str) or _lineage_basename(path_label) != "trajectory_v2.json":
            target.errors.append(
                "artifact_lineage.outputs.trajectory_v2.path must identify trajectory_v2.json."
            )
        _validate_lineage_file_record(
            trajectory_v2_record,
            run_dir,
            target,
            "artifact_lineage.outputs.trajectory_v2",
        )
    if "run_digest" in outputs:
        _validate_lineage_file_record(outputs["run_digest"], run_dir, target, "artifact_lineage.outputs.run_digest")
    else:
        target.warnings.append("artifact_lineage.outputs missing 'run_digest'; rerun the run to fingerprint the evidence digest.")
    if "before_state_snapshot" in outputs:
        _validate_lineage_file_record(outputs["before_state_snapshot"], run_dir, target, "artifact_lineage.outputs.before_state_snapshot")
    if "state_snapshot" in outputs:
        _validate_lineage_file_record(outputs["state_snapshot"], run_dir, target, "artifact_lineage.outputs.state_snapshot")
    if "state_diff" in outputs:
        _validate_lineage_file_record(outputs["state_diff"], run_dir, target, "artifact_lineage.outputs.state_diff")
    evidence_links = lineage.get("evidence_links")
    if not isinstance(evidence_links, list):
        target.errors.append("artifact_lineage.evidence_links must be a list.")
        evidence_links = []
    expected_ref_count = _scorecard_evidence_ref_count(scorecard)
    if scorecard is not None and len(evidence_links) != expected_ref_count:
        target.errors.append(
            f"artifact_lineage.evidence_links expected {expected_ref_count}, got {len(evidence_links)}."
        )
    for index, link in enumerate(evidence_links):
        _validate_lineage_evidence_link(link, index, len(events), target)
    graph = lineage.get("graph")
    if not isinstance(graph, list):
        target.errors.append("artifact_lineage.graph must be a list.")
    replay = lineage.get("replay")
    if isinstance(replay, dict):
        _validate_lineage_replay(replay, inputs, target)
    else:
        target.warnings.append("artifact_lineage.replay is missing; rerun the run to emit replay instructions.")
    summary = lineage.get("summary")
    if not isinstance(summary, dict):
        target.errors.append("artifact_lineage.summary must be an object.")
    else:
        inputs_raw = lineage.get("inputs")
        expected_input_count = len(inputs_raw) if isinstance(inputs_raw, list) else None
        if summary.get("input_count") != expected_input_count:
            target.errors.append("artifact_lineage.summary.input_count must match inputs length.")
        outputs_raw = lineage.get("outputs")
        expected_output_count = len(outputs_raw) if isinstance(outputs_raw, list) else None
        if summary.get("output_count") != expected_output_count:
            target.errors.append("artifact_lineage.summary.output_count must match outputs length.")
        if summary.get("evidence_link_count") != len(evidence_links):
            target.errors.append("artifact_lineage.summary.evidence_link_count must match evidence_links length.")
        if isinstance(replay, dict) and summary.get("self_contained_replay") != replay.get("self_contained"):
            target.errors.append("artifact_lineage.summary.self_contained_replay must match replay.self_contained.")

def _validate_trace_observability(observability: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(observability, "schema_version", TRACE_OBSERVABILITY_SCHEMA_VERSION, target)
    runs = observability.get("runs")
    if not isinstance(runs, list):
        target.errors.append("trace_observability.runs must be a list.")
        runs = []
    metrics = observability.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("trace_observability.metrics must be an object.")
        metrics = {}
    checks = observability.get("checks")
    if not isinstance(checks, list):
        target.errors.append("trace_observability.checks must be a list.")
        checks = []
    if not isinstance(observability.get("passed"), bool):
        target.errors.append("trace_observability.passed must be a boolean.")

    failed_checks = 0
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            target.errors.append(f"trace_observability.checks[{index}] must be an object.")
            continue
        if not isinstance(check.get("id"), str) or not check.get("id"):
            target.errors.append(f"trace_observability.checks[{index}].id must be a non-empty string.")
        if not isinstance(check.get("passed"), bool):
            target.errors.append(f"trace_observability.checks[{index}].passed must be a boolean.")
        elif not check["passed"]:
            failed_checks += 1
    if observability.get("check_count") != len(checks):
        target.errors.append(f"trace_observability.check_count expected {len(checks)}, got {observability.get('check_count')!r}.")
    if observability.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"trace_observability.failed_check_count expected {failed_checks}, got {observability.get('failed_check_count')!r}."
        )
    if isinstance(observability.get("passed"), bool) and observability["passed"] != (failed_checks == 0):
        target.errors.append("trace_observability.passed must match failed_check_count.")

    run_totals = _validate_trace_observability_runs(runs, target)
    _validate_trace_observability_metrics(metrics, run_totals, target)
    warnings = observability.get("warnings")
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        target.errors.append("trace_observability.warnings must be a list of strings.")
    target.details.update(
        {
            "run_count": run_totals["run_count"],
            "average_event_count": metrics.get("average_event_count"),
            "event_type_count": metrics.get("event_type_count"),
            "tool_or_api_run_rate": metrics.get("tool_or_api_run_rate"),
        }
    )

def _validate_trace_observability_runs(runs: list[Any], target: ValidationTarget) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "run_count": len(runs),
        "total_event_count": 0,
        "runs_with_final_answer": 0,
        "empty_final_answer_count": 0,
        "runs_with_tool_or_api_events": 0,
        "tool_call_count": 0,
        "tool_result_count": 0,
        "api_call_count": 0,
        "subagent_event_count": 0,
        "approval_event_count": 0,
        "event_type_counts": {},
        "source_format_counts": {},
        "model_counts": {},
        "risk_counts": {},
        "event_counts": [],
    }
    for index, run in enumerate(runs):
        label = f"trace_observability.runs[{index}]"
        if not isinstance(run, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("run_dir", "scenario_id", "source_format", "model"):
            if not isinstance(run.get(field_name), str) or not run.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        _warn_absolute_public_path(target, f"{label}.run_dir", run.get("run_dir"))
        if run.get("passed") is not None and not isinstance(run.get("passed"), bool):
            target.errors.append(f"{label}.passed must be a boolean or null.")
        if run.get("score") is not None and not _is_int_between(run.get("score"), 0, 100):
            target.errors.append(f"{label}.score must be an integer from 0 to 100 or null.")
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
            if not _is_non_negative_int(run.get(field_name)):
                target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
        event_count = _non_negative_int_value(run.get("event_count"))
        totals["event_counts"].append(event_count)
        totals["total_event_count"] += event_count
        for field_name in ("tool_call_count", "tool_result_count", "api_call_count", "subagent_event_count", "approval_event_count"):
            totals[field_name] += _non_negative_int_value(run.get(field_name))
        if not isinstance(run.get("has_final_answer"), bool):
            target.errors.append(f"{label}.has_final_answer must be a boolean.")
        elif run["has_final_answer"]:
            totals["runs_with_final_answer"] += 1
        else:
            totals["empty_final_answer_count"] += 1
        if not isinstance(run.get("has_tool_or_api_events"), bool):
            target.errors.append(f"{label}.has_tool_or_api_events must be a boolean.")
        elif run["has_tool_or_api_events"]:
            totals["runs_with_tool_or_api_events"] += 1
        _merge_count_rows(totals["event_type_counts"], run.get("event_types"), target, f"{label}.event_types")
        event_types = _count_rows(run.get("event_types"))
        if event_types is not None and run.get("event_type_count") != len(event_types):
            target.errors.append(f"{label}.event_type_count must match event_types length.")
        source_format = str(run.get("source_format") or "unknown")
        model = str(run.get("model") or "unknown")
        totals["source_format_counts"][source_format] = totals["source_format_counts"].get(source_format, 0) + 1
        totals["model_counts"][model] = totals["model_counts"].get(model, 0) + 1
        risks = run.get("risks")
        if not _is_string_list(risks):
            target.errors.append(f"{label}.risks must be a list of strings.")
        else:
            for risk in risks:
                totals["risk_counts"][risk] = totals["risk_counts"].get(risk, 0) + 1
    return totals

def _validate_trace_observability_metrics(metrics: dict[str, Any], totals: dict[str, Any], target: ValidationTarget) -> None:
    event_counts = totals["event_counts"]
    expected = {
        "run_count": totals["run_count"],
        "total_event_count": totals["total_event_count"],
        "min_event_count": min(event_counts) if event_counts else 0,
        "max_event_count": max(event_counts) if event_counts else 0,
        "event_type_count": len(totals["event_type_counts"]),
        "runs_with_final_answer": totals["runs_with_final_answer"],
        "empty_final_answer_count": totals["empty_final_answer_count"],
        "runs_with_tool_or_api_events": totals["runs_with_tool_or_api_events"],
        "tool_call_count": totals["tool_call_count"],
        "tool_result_count": totals["tool_result_count"],
        "api_call_count": totals["api_call_count"],
        "subagent_event_count": totals["subagent_event_count"],
        "approval_event_count": totals["approval_event_count"],
    }
    for field_name, expected_value in expected.items():
        if metrics.get(field_name) != expected_value:
            target.errors.append(f"trace_observability.metrics.{field_name} expected {expected_value!r}, got {metrics.get(field_name)!r}.")
    average = round(totals["total_event_count"] / totals["run_count"], 2) if totals["run_count"] else 0.0
    if metrics.get("average_event_count") != average:
        target.errors.append(f"trace_observability.metrics.average_event_count expected {average!r}, got {metrics.get('average_event_count')!r}.")
    final_answer_rate = _rate_value(totals["runs_with_final_answer"], totals["run_count"])
    if metrics.get("final_answer_rate") != final_answer_rate:
        target.errors.append(f"trace_observability.metrics.final_answer_rate expected {final_answer_rate!r}, got {metrics.get('final_answer_rate')!r}.")
    tool_or_api_rate = _rate_value(totals["runs_with_tool_or_api_events"], totals["run_count"])
    if metrics.get("tool_or_api_run_rate") != tool_or_api_rate:
        target.errors.append(f"trace_observability.metrics.tool_or_api_run_rate expected {tool_or_api_rate!r}, got {metrics.get('tool_or_api_run_rate')!r}.")
    for field_name, expected_counts in (
        ("event_type_counts", totals["event_type_counts"]),
        ("source_format_counts", totals["source_format_counts"]),
        ("model_counts", totals["model_counts"]),
        ("risk_counts", totals["risk_counts"]),
    ):
        counts = _count_rows(metrics.get(field_name))
        if counts != expected_counts:
            target.errors.append(f"trace_observability.metrics.{field_name} does not match runs.")

def _lineage_records(value: Any, target: ValidationTarget, label: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list.")
        return records
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            target.errors.append(f"{label}[{index}] must be an object.")
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name:
            target.errors.append(f"{label}[{index}].name must be a non-empty string.")
            continue
        records[name] = record
    return records

def _validate_lineage_file_record(record: dict[str, Any], run_dir: Path, target: ValidationTarget, label: str) -> None:
    path_label = record.get("path")
    if not isinstance(path_label, str) or not path_label:
        target.errors.append(f"{label}.path must be a non-empty string.")
        return
    _warn_absolute_public_path(target, f"{label}.path", path_label)
    if record.get("exists") is not True:
        target.errors.append(f"{label}.exists must be true for required run outputs.")
        return
    basename = _lineage_basename(path_label)
    file_path = run_dir / basename
    if not file_path.exists():
        target.errors.append(f"{label}.path does not resolve inside the run directory.")
        return
    expected_size = record.get("size_bytes")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
        target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    elif file_path.stat().st_size != expected_size:
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    expected_hash = record.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string.")
    elif _sha256(file_path) != expected_hash:
        target.errors.append(f"{label}.sha256 does not match the current file.")

def _validate_lineage_input_record(name: str, record: dict[str, Any], target: ValidationTarget, label: str) -> None:
    if record.get("role") != "input":
        target.errors.append(f"{label}.role must be input.")
    path_label = record.get("path")
    if path_label is not None and not isinstance(path_label, str):
        target.errors.append(f"{label}.path must be a string or null.")
    else:
        _warn_absolute_public_path(target, f"{label}.path", path_label)
    if not isinstance(record.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    if record.get("exists") is True:
        if not _is_non_negative_int(record.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for existing inputs.")
        if not _is_sha256(record.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string for existing inputs.")
    if "sensitive" in record and not isinstance(record.get("sensitive"), bool):
        target.errors.append(f"{label}.sensitive must be a boolean when present.")
    if name in {"scenario", "source_trace", "source_before_state_snapshot", "source_state_snapshot"} and record.get("exists") is not True:
        target.warnings.append(f"{label}.exists is not true; replay may require restoring this input.")

def _validate_lineage_replay(replay: dict[str, Any], inputs: dict[str, dict[str, Any]], target: ValidationTarget) -> None:
    if replay.get("tool") != "flightrecorder":
        target.errors.append("artifact_lineage.replay.tool must be flightrecorder.")
    argv = replay.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) and item for item in argv):
        target.errors.append("artifact_lineage.replay.argv must be a list of non-empty strings.")
        argv = []
    else:
        _warn_replay_metadata_public_paths(replay, target, "artifact_lineage.replay")
        expected_prefix = ["python", "-m", "flightrecorder", "run"]
        if argv[:4] != expected_prefix:
            target.errors.append("artifact_lineage.replay.argv must start with python -m flightrecorder run.")
        for required_flag in ("--scenario", "--trace", "--out"):
            if required_flag not in argv:
                target.errors.append(f"artifact_lineage.replay.argv missing {required_flag}.")
    if not isinstance(replay.get("command"), str) or not replay.get("command"):
        target.errors.append("artifact_lineage.replay.command must be a non-empty string.")
    if not isinstance(replay.get("self_contained"), bool):
        target.errors.append("artifact_lineage.replay.self_contained must be a boolean.")
    notes = replay.get("notes")
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        target.errors.append("artifact_lineage.replay.notes must be a list of strings.")
    fingerprints = replay.get("input_fingerprints")
    if not isinstance(fingerprints, dict):
        target.errors.append("artifact_lineage.replay.input_fingerprints must be an object.")
        return
    for required_name in ("scenario", "source_trace"):
        if required_name not in fingerprints:
            target.errors.append(f"artifact_lineage.replay.input_fingerprints missing {required_name}.")
    for name, fingerprint in fingerprints.items():
        label = f"artifact_lineage.replay.input_fingerprints.{name}"
        if not isinstance(name, str) or not name:
            target.errors.append("artifact_lineage.replay.input_fingerprints keys must be non-empty strings.")
            continue
        if not isinstance(fingerprint, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        if "path" not in fingerprint or "sha256" not in fingerprint or "exists" not in fingerprint or "size_bytes" not in fingerprint:
            target.errors.append(f"{label} must contain path, sha256, size_bytes, and exists.")
        if fingerprint.get("path") is not None and not isinstance(fingerprint.get("path"), str):
            target.errors.append(f"{label}.path must be a string or null.")
        else:
            _warn_absolute_public_path(target, f"{label}.path", fingerprint.get("path"))
        if fingerprint.get("sha256") is not None and not _is_sha256(fingerprint.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string or null.")
        if fingerprint.get("size_bytes") is not None and not _is_non_negative_int(fingerprint.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer or null.")
        if fingerprint.get("exists") is not None and not isinstance(fingerprint.get("exists"), bool):
            target.errors.append(f"{label}.exists must be a boolean or null.")
        input_record = inputs.get(name)
        if input_record is None:
            target.errors.append(f"{label} does not match an artifact_lineage input record.")
            continue
        for field_name in ("path", "sha256", "size_bytes", "exists"):
            if fingerprint.get(field_name) != input_record.get(field_name):
                target.errors.append(f"{label}.{field_name} must match artifact_lineage.inputs.{name}.{field_name}.")

def _validate_lineage_evidence_link(link: Any, index: int, event_count: int, target: ValidationTarget) -> None:
    label = f"artifact_lineage.evidence_links[{index}]"
    if not isinstance(link, dict):
        target.errors.append(f"{label} must be an object.")
        return
    for field_name in ("rule_id", "rule_name", "scorecard_pointer", "target"):
        if not isinstance(link.get(field_name), str) or not link.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    ref_target = link.get("target")
    if ref_target not in {"event", "final_answer", "episode", "state_snapshot"}:
        target.errors.append(f"{label}.target must be one of event, final_answer, episode, or state_snapshot.")
    if ref_target == "event":
        event_index = link.get("event_index")
        if not isinstance(event_index, int) or isinstance(event_index, bool) or event_index < 0:
            target.errors.append(f"{label}.event_index must be a non-negative integer.")
        elif event_index >= event_count:
            target.errors.append(f"{label}.event_index must refer to an existing trace event.")
        if link.get("trace_pointer") != f"/events/{event_index}":
            target.errors.append(f"{label}.trace_pointer must point at the referenced trace event.")
    elif ref_target == "final_answer" and link.get("trace_pointer") != "/final_answer":
        target.errors.append(f"{label}.trace_pointer must point at /final_answer.")
    elif ref_target == "episode" and link.get("trace_pointer") != "/":
        target.errors.append(f"{label}.trace_pointer must point at the trace root.")
    elif ref_target == "state_snapshot" and link.get("state_pointer") != "/":
        target.errors.append(f"{label}.state_pointer must point at the state snapshot root.")
    if "rule_passed" in link and not isinstance(link.get("rule_passed"), bool):
        target.errors.append(f"{label}.rule_passed must be a boolean when present.")
    if "ref_passed" in link and not isinstance(link.get("ref_passed"), bool):
        target.errors.append(f"{label}.ref_passed must be a boolean when present.")

def _scorecard_evidence_ref_count(scorecard: dict[str, Any] | None) -> int:
    if not isinstance(scorecard, dict):
        return 0
    total = 0
    for rule in scorecard.get("rules", []):
        if isinstance(rule, dict) and isinstance(rule.get("evidence_refs"), list):
            total += len(rule["evidence_refs"])
    return total

def _lineage_basename(path_label: str) -> str:
    if path_label.startswith("<redacted:") and path_label.endswith(">"):
        return path_label[len("<redacted:") : -1]
    return path_label.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]

def _run_digest_trace_signal(trace: dict[str, Any]) -> dict[str, Any]:
    events = trace.get("events") if isinstance(trace.get("events"), list) else []
    typed_events = [event for event in events if isinstance(event, dict)]
    event_types = sorted({str(event.get("type")) for event in typed_events if event.get("type")})
    tool_call_count = sum(1 for event in typed_events if event.get("type") == "tool_call")
    api_call_count = _run_digest_api_call_count(trace, typed_events)
    final_answer = trace.get("final_answer")
    session = trace.get("session") if isinstance(trace.get("session"), dict) else {}
    return {
        "event_count": len(typed_events),
        "event_types": event_types,
        "tool_call_count": tool_call_count,
        "tool_result_count": sum(1 for event in typed_events if event.get("type") == "tool_result"),
        "api_call_count": api_call_count,
        "subagent_start_count": sum(1 for event in typed_events if event.get("type") == "subagent_start"),
        "max_subagent_depth": _run_digest_max_subagent_depth(typed_events),
        "has_final_answer": isinstance(final_answer, str) and bool(final_answer.strip()),
        "has_tool_or_api_events": tool_call_count > 0 or api_call_count > 0,
        "source_format": str(session.get("source_format") or "unknown"),
        "model": str(session.get("model") or "unknown"),
    }

def _run_digest_api_call_count(trace: dict[str, Any], events: list[dict[str, Any]]) -> int:
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    api_calls = metadata.get("api_calls")
    if isinstance(api_calls, int) and not isinstance(api_calls, bool) and api_calls >= 0:
        return api_calls
    return sum(1 for event in events if event.get("type") == "api_call")

def _run_digest_max_subagent_depth(events: list[dict[str, Any]]) -> int:
    parent_by_session: dict[str, str | None] = {}
    for event in events:
        if event.get("type") != "subagent_start":
            continue
        session_id = event.get("session_id")
        if isinstance(session_id, str) and session_id:
            parent = event.get("parent_session_id")
            parent_by_session[session_id] = parent if isinstance(parent, str) and parent else None
    max_depth = 0
    for session_id in parent_by_session:
        seen: set[str] = set()
        depth = 1
        parent = parent_by_session.get(session_id)
        while parent and parent not in seen:
            seen.add(parent)
            if parent in parent_by_session:
                depth += 1
                parent = parent_by_session[parent]
            else:
                break
        max_depth = max(max_depth, depth)
    return max_depth

def _run_digest_evidence_counts(
    scorecard: dict[str, Any],
    task_completion: dict[str, Any] | None,
) -> dict[str, int]:
    rules = [rule for rule in scorecard.get("rules", []) if isinstance(rule, dict)]
    failed_rules = [rule for rule in rules if rule.get("passed") is False]
    critical_failed_rules = [rule for rule in failed_rules if rule.get("critical") is True]
    task = task_completion
    if task is None and isinstance(scorecard.get("task_completion"), dict):
        task = scorecard["task_completion"]
    task_refs = task.get("evidence_refs") if isinstance(task, dict) and isinstance(task.get("evidence_refs"), list) else []
    missing_refs = (
        task.get("missing_evidence_refs")
        if isinstance(task, dict) and isinstance(task.get("missing_evidence_refs"), list)
        else []
    )
    rule_ref_count = _run_digest_rule_ref_count(rules)
    return {
        "rule_evidence_ref_count": rule_ref_count,
        "failed_rule_evidence_ref_count": _run_digest_rule_ref_count(failed_rules),
        "critical_failed_rule_evidence_ref_count": _run_digest_rule_ref_count(critical_failed_rules),
        "task_completion_evidence_ref_count": len(task_refs),
        "missing_evidence_ref_count": len(missing_refs),
        "total_evidence_ref_count": rule_ref_count + len(task_refs) + len(missing_refs),
    }

def _run_digest_rule_ref_count(rules: list[dict[str, Any]]) -> int:
    count = 0
    for rule in rules:
        refs = rule.get("evidence_refs")
        if isinstance(refs, list):
            count += len(refs)
    return count
