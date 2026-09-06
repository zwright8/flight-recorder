"""Extracted validation implementation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Any
from ..action_gate import ACTION_LEDGER_GATE_POLICY_SCHEMA_VERSION, ACTION_LEDGER_GATE_SCHEMA_VERSION, evaluate_action_ledger_gate
from ..action_ledger import ACTION_LEDGER_SCHEMA_VERSION
from ..bundle import EVIDENCE_BUNDLE_NOTES, EVIDENCE_BUNDLE_SCHEMA_VERSION, HARNESS_RUN_MANIFEST_SCHEMA_VERSION, HARNESS_RUN_RESULT_SCHEMA_VERSION, _decision_key_metrics as _build_evidence_bundle_decision_key_metrics, _decision_text as _build_evidence_bundle_decision_text, _next_actions as _build_evidence_bundle_next_actions
from ..schema_registry import SchemaRegistryError, check_schema_contract, check_schema_file
from ..decision_gate import DECISION_GATE_SCHEMA_VERSION, decision_gate_source_contract_errors
from ..eval_summary import EVAL_SUMMARY_SCHEMA_VERSION, LabeledPath, _external_adapter_plan as _build_eval_summary_external_plan, _external_adapter_result as _build_eval_summary_external_result, _external_result_association_risks as _build_eval_summary_external_result_risks, _heldout_scenario_summary as _build_eval_summary_heldout, _suite_arm
from ..external_eval_result import EXTERNAL_EVAL_RESULT_SCHEMA_VERSION, ExternalEvalResultError, build_external_eval_result
from ..improvement_gate import IMPROVEMENT_LEDGER_GATE_POLICY_SCHEMA_VERSION, IMPROVEMENT_LEDGER_GATE_SCHEMA_VERSION, evaluate_improvement_ledger_gate
from ..improvement_ledger import IMPROVEMENT_LEDGER_SCHEMA_VERSION, stable_work_key
from ..improvement_plan import IMPROVEMENT_PLAN_SCHEMA_VERSION, PRIORITIES, work_item_fingerprint
from ..lineage import LINEAGE_SCHEMA_VERSION, REPLAY_BUNDLE_SCHEMA_VERSION
from ..model_registry import MODEL_ADAPTER_MANIFEST_SCHEMA_VERSION, MODEL_CANDIDATE_SCHEMA_VERSION, MODEL_COMPATIBILITY_REPORT_SCHEMA_VERSION, MODEL_REGISTRY_ENTRY_SCHEMA_VERSION, MODEL_REGISTRY_SCHEMA_VERSION, MODEL_SCOUT_MANIFEST_SCHEMA_VERSION, MODEL_SERVING_PROBE_RECEIPT_SCHEMA_VERSION, TRAINING_PLAN_SCHEMA_VERSION, is_training_license_approved, model_adapter_manifest_errors, model_candidate_errors, model_compatibility_report_errors, model_registry_entry_errors, model_registry_errors, model_scout_manifest_errors, model_serving_probe_receipt_errors, training_plan_errors
from ..governance import PROMOTION_ALIAS_APPLY_SCHEMA_VERSION, PROMOTION_ALIAS_APPLY_CHECK_IDS, PROMOTION_CARDS_REQUIRED_INPUTS, PROMOTION_CARDS_SCHEMA_VERSION, PROMOTION_DECISION_REQUIRED_ARTIFACTS, PROMOTION_DECISION_REQUIRED_PASS_CHECK_IDS, PROMOTION_DECISION_SCHEMA_VERSION, PROMOTION_POLICY_DEFAULT_LIMITS, PROMOTION_POLICY_REQUIRED_FORBIDDEN_RULES, PROMOTION_POLICY_SCHEMA_VERSION, PROMOTION_ROLLBACK_RECEIPT_SCHEMA_VERSION, PROMOTION_RELEASE_RECORD_REQUIRED_ARTIFACTS, PROMOTION_RELEASE_RECORD_VALIDATED_ARTIFACTS, PROMOTION_RELEASE_RECORD_SCHEMA_VERSION, _JSON_ARTIFACT_ROLES as _PROMOTION_JSON_ARTIFACT_ROLES, build_promotion_decision as _build_promotion_decision, _decision_metrics as _build_promotion_decision_metrics, _promotion_external_eval_checks as _build_promotion_external_eval_checks, _promotion_external_eval_lineage as _build_promotion_external_eval_lineage, promotion_release_record_check_ids
from ..promotion_archive import PROMOTION_ARCHIVE_SCHEMA_VERSION
from ..promotion_gate import PROMOTION_LEDGER_GATE_POLICY_SCHEMA_VERSION, PROMOTION_LEDGER_GATE_SCHEMA_VERSION, evaluate_promotion_ledger_gate
from ..promotion_ledger import PROMOTION_LEDGER_SCHEMA_VERSION
from ..repair import REPAIR_ITEM_SCHEMA_VERSION, REPAIR_QUEUE_SCHEMA_VERSION
from ..path_safety import path_has_symlink_component as _path_has_symlink_component, resolve_artifact_reference_path
from ..hashing import sha256_file as _sha256
from .constants import RUN_SUITE_HARNESS_SOURCE, SERVING_LIFECYCLE_PREFLIGHT_ARTIFACTS
from .evaluation_serving import _validate_eval_summary, _validate_serving_lifecycle, validate_eval_summary, validate_external_eval_result
from .models import _validate_model_registry
from .primitives import ValidationTarget, _archive_artifact_path, _archive_artifact_roles_by_name, _count_rows, _directory_sha256, _increment, _is_int_between, _is_lowercase_sha256, _is_non_negative_int, _is_number_between, _is_sha256, _is_string_list, _is_windows_absolute, _looks_absolute, _path_resolves_inside, _read_object, _reject_archive_artifact_symlink_path, _require_equal, _sha256, _validate_allowed_keys, _validate_archive_relationship, _validate_count_rows, _validate_evidence_refs, _validate_gate_like_checks, _warn_absolute_public_path
from .runs import _lineage_records, _warn_replay_metadata_public_paths

def validate_evidence_bundle(path: str | Path) -> ValidationTarget:
    """Validate an evidence-bundle handoff summary artifact."""
    bundle_path = Path(path)
    target = ValidationTarget("evidence_bundle", str(bundle_path))
    bundle = _read_object(bundle_path, target, "evidence_bundle.json")
    if bundle is not None:
        _validate_evidence_bundle(bundle, target, bundle_path)
    return target

def validate_improvement_plan(path: str | Path) -> ValidationTarget:
    """Validate an improvement-plan handoff artifact."""
    plan_path = Path(path)
    target = ValidationTarget("improvement_plan", str(plan_path))
    plan = _read_object(plan_path, target, "improvement_plan.json")
    if plan is not None:
        _validate_improvement_plan(plan, target, plan_path)
    return target

def validate_improvement_ledger(path: str | Path) -> ValidationTarget:
    """Validate a longitudinal improvement-ledger artifact."""
    ledger_path = Path(path)
    target = ValidationTarget("improvement_ledger", str(ledger_path))
    ledger = _read_object(ledger_path, target, "improvement_ledger.json")
    if ledger is not None:
        _validate_improvement_ledger(ledger, target, ledger_path)
    return target

def validate_improvement_ledger_payload_consistency(ledger: dict[str, Any]) -> ValidationTarget:
    """Validate improvement-ledger semantics without resolving recorded source files."""
    target = ValidationTarget("improvement_ledger", "<in-memory>")
    _validate_improvement_ledger(ledger, target, Path("."), validate_sources=False)
    return target

def validate_improvement_ledger_gate(path: str | Path) -> ValidationTarget:
    """Validate an improvement-ledger gate artifact."""
    gate_path = Path(path)
    target = ValidationTarget("improvement_ledger_gate", str(gate_path))
    gate = _read_object(gate_path, target, "improvement_ledger_gate.json")
    if gate is not None:
        _validate_improvement_ledger_gate(gate, target, gate_path)
    return target

def validate_action_ledger(path: str | Path) -> ValidationTarget:
    """Validate a longitudinal action-ledger artifact."""
    ledger_path = Path(path)
    target = ValidationTarget("action_ledger", str(ledger_path))
    ledger = _read_object(ledger_path, target, "action_ledger.json")
    if ledger is not None:
        _validate_action_ledger(ledger, target, ledger_path)
    return target

def validate_action_ledger_payload_consistency(ledger: dict[str, Any]) -> ValidationTarget:
    """Validate action-ledger semantics without resolving recorded source files."""
    target = ValidationTarget("action_ledger", "<in-memory>")
    _validate_action_ledger(ledger, target, Path("."), validate_sources=False)
    return target

def validate_action_ledger_gate(path: str | Path) -> ValidationTarget:
    """Validate an action-ledger gate artifact."""
    gate_path = Path(path)
    target = ValidationTarget("action_ledger_gate", str(gate_path))
    gate = _read_object(gate_path, target, "action_ledger_gate.json")
    if gate is not None:
        _validate_action_ledger_gate(gate, target, gate_path)
    return target

def validate_decision_gate(path: str | Path) -> ValidationTarget:
    """Validate a decision-gate artifact."""
    gate_path = Path(path)
    target = ValidationTarget("decision_gate", str(gate_path))
    gate = _read_object(gate_path, target, "decision_gate.json")
    if gate is not None:
        _validate_decision_gate(gate, target, gate_path)
    return target

def validate_promotion_ledger(path: str | Path) -> ValidationTarget:
    """Validate a longitudinal promotion-ledger artifact."""
    ledger_path = Path(path)
    target = ValidationTarget("promotion_ledger", str(ledger_path))
    ledger = _read_object(ledger_path, target, "promotion_ledger.json")
    if ledger is not None:
        _validate_promotion_ledger(ledger, target, ledger_path)
    return target

def validate_promotion_ledger_payload_consistency(ledger: dict[str, Any]) -> ValidationTarget:
    """Validate promotion-ledger semantics without resolving recorded source files."""
    target = ValidationTarget("promotion_ledger", "<in-memory>")
    _validate_promotion_ledger(ledger, target, Path("."), validate_sources=False)
    return target

def validate_promotion_cards(path: str | Path) -> ValidationTarget:
    """Validate a promotion-cards directory or manifest."""
    cards_path = Path(path)
    manifest_path = cards_path / "promotion_cards.json" if cards_path.is_dir() else cards_path
    target = ValidationTarget("promotion_cards", str(path))
    cards = _read_object(manifest_path, target, "promotion_cards.json")
    if cards is not None:
        _validate_promotion_cards(cards, target, manifest_path)
    return target

def validate_promotion_decision(path: str | Path) -> ValidationTarget:
    """Validate a top-level governance promotion decision artifact."""
    decision_path = Path(path)
    target = ValidationTarget("promotion_decision", str(decision_path))
    decision = _read_object(decision_path, target, "promotion_decision.json")
    if decision is not None:
        _validate_promotion_decision(decision, target, decision_path)
    return target

def validate_promotion_alias_apply(path: str | Path) -> ValidationTarget:
    """Validate a guarded promotion alias application receipt."""
    receipt_path = Path(path)
    target = ValidationTarget("promotion_alias_apply", str(receipt_path))
    receipt = _read_object(receipt_path, target, "promotion_alias_apply.json")
    if receipt is not None:
        _validate_promotion_alias_apply(receipt, target, receipt_path)
    return target

def validate_promotion_rollback_receipt(path: str | Path) -> ValidationTarget:
    """Validate a promotion rollback target receipt."""
    receipt_path = Path(path)
    target = ValidationTarget("promotion_rollback_receipt", str(receipt_path))
    receipt = _read_object(receipt_path, target, "promotion_rollback_receipt.json")
    if receipt is not None:
        _validate_promotion_rollback_receipt(receipt, target, receipt_path)
    return target

def validate_promotion_release_record(path: str | Path) -> ValidationTarget:
    """Validate a promotion release record."""
    record_path = Path(path)
    target = ValidationTarget("promotion_release_record", str(record_path))
    record = _read_object(record_path, target, "promotion_release_record.json")
    if record is not None:
        _validate_promotion_release_record(record, target, record_path)
    return target

def validate_promotion_policy(path: str | Path) -> ValidationTarget:
    """Validate a raw promotion policy artifact."""
    policy_path = Path(path)
    target = ValidationTarget("promotion_policy", str(policy_path))
    policy = _read_object(policy_path, target, "promotion_policy.json")
    if policy is not None:
        _validate_raw_promotion_policy(policy, target)
    return target

def validate_promotion_ledger_gate(path: str | Path) -> ValidationTarget:
    """Validate a promotion-ledger gate artifact."""
    gate_path = Path(path)
    target = ValidationTarget("promotion_ledger_gate", str(gate_path))
    gate = _read_object(gate_path, target, "promotion_ledger_gate.json")
    if gate is not None:
        _validate_promotion_ledger_gate(gate, target, gate_path)
    return target

def validate_promotion_archive(path: str | Path) -> ValidationTarget:
    """Validate a portable promotion archive directory or manifest."""
    archive_path = Path(path)
    manifest_path = archive_path / "promotion_archive.json" if archive_path.is_dir() else archive_path
    archive_root = manifest_path.parent
    target = ValidationTarget("promotion_archive", str(archive_path))
    archive = _read_object(manifest_path, target, "promotion_archive.json")
    if archive is not None:
        _validate_promotion_archive(archive, target, archive_root)
    return target

def validate_repair_queue(path: str | Path) -> ValidationTarget:
    """Validate a repair-queue artifact."""
    queue_path = Path(path)
    target = ValidationTarget("repair_queue", str(queue_path))
    queue = _read_object(queue_path, target, "repair_queue.json")
    if queue is not None:
        _validate_repair_queue(queue, target, queue_path)
    return target

def validate_replay_bundle(path: str | Path) -> ValidationTarget:
    """Validate a portable replay-bundle directory or replay_bundle.json artifact."""
    raw_path = Path(path)
    bundle_dir = raw_path if raw_path.is_dir() else raw_path.parent
    manifest_path = raw_path / "replay_bundle.json" if raw_path.is_dir() else raw_path
    target = ValidationTarget("replay_bundle", str(raw_path))
    manifest = _read_object(manifest_path, target, "replay_bundle.json")
    if manifest is not None:
        _validate_replay_bundle(manifest, bundle_dir, target)
    return target

def _validate_evidence_bundle(bundle: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(bundle, "schema_version", EVIDENCE_BUNDLE_SCHEMA_VERSION, target)
    if not isinstance(bundle.get("bundle_path"), str) or not bundle.get("bundle_path"):
        target.errors.append("evidence_bundle.bundle_path must be a non-empty string.")
    else:
        _warn_absolute_public_path(target, "evidence_bundle.bundle_path", bundle.get("bundle_path"))
    if not isinstance(bundle.get("passed"), bool):
        target.errors.append("evidence_bundle.passed must be a boolean.")

    checks = bundle.get("checks")
    if not isinstance(checks, list):
        target.errors.append("evidence_bundle.checks must be a list.")
        checks = []
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, dict):
        target.errors.append("evidence_bundle.artifacts must be an object.")
        artifacts = {}
    metrics = bundle.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("evidence_bundle.metrics must be an object.")
        metrics = {}
    notes = bundle.get("notes")
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        target.errors.append("evidence_bundle.notes must be a list of strings.")
    elif notes != list(EVIDENCE_BUNDLE_NOTES):
        target.errors.append("evidence_bundle.notes must match the producer notes.")

    failed_checks = _validate_evidence_bundle_checks(checks, target)
    blocking_check_rows = _evidence_bundle_blocking_check_rows(checks)
    blocking_gate_rows = _evidence_bundle_blocking_gate_rows(metrics)
    if bundle.get("check_count") != len(checks):
        target.errors.append(f"evidence_bundle.check_count expected {len(checks)}, got {bundle.get('check_count')!r}.")
    if bundle.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"evidence_bundle.failed_check_count expected {failed_checks}, got {bundle.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(bundle.get("passed"), bool) and bundle["passed"] != expected_passed:
        target.errors.append("evidence_bundle.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    if bundle.get("readiness") != expected_readiness:
        target.errors.append(f"evidence_bundle.readiness expected {expected_readiness!r}, got {bundle.get('readiness')!r}.")
    if "decision" in bundle:
        _validate_evidence_bundle_decision(
            bundle.get("decision"),
            expected_readiness,
            failed_checks,
            blocking_check_rows,
            blocking_gate_rows,
            artifacts,
            metrics,
            target,
        )
    if not artifacts:
        target.errors.append("evidence_bundle.artifacts must not be empty.")
    for name, record in artifacts.items():
        _validate_evidence_bundle_artifact_record(name, record, target, source_path)
    _validate_evidence_bundle_metrics(metrics, target)
    _validate_evidence_bundle_harness_checks(metrics, checks, target)
    _validate_evidence_bundle_validation_checks(metrics, checks, target)
    _validate_evidence_bundle_eval_summary(metrics, artifacts, checks, target, source_path)
    _validate_evidence_bundle_serving_lifecycle(metrics, artifacts, checks, target, source_path)
    target.details.update(
        {
            "readiness": bundle.get("readiness"),
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "artifact_count": len(artifacts),
        }
    )

def _validate_improvement_plan(plan: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(plan, "schema_version", IMPROVEMENT_PLAN_SCHEMA_VERSION, target)
    if not isinstance(plan.get("plan_path"), str) or not plan.get("plan_path"):
        target.errors.append("improvement_plan.plan_path must be a non-empty string.")
    if plan.get("passed") is not True:
        target.errors.append("improvement_plan.passed must be true.")
    if plan.get("readiness") not in {"ready", "blocked"}:
        target.errors.append("improvement_plan.readiness must be ready or blocked.")

    source_artifacts = plan.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        target.errors.append("improvement_plan.source_artifacts must be an object.")
        source_artifacts = {}
    if "evidence_bundle" not in source_artifacts:
        target.errors.append("improvement_plan.source_artifacts.evidence_bundle is required.")
    for name, record in source_artifacts.items():
        _validate_improvement_source_artifact(name, record, target, source_path)

    work_items = plan.get("work_items")
    if not isinstance(work_items, list):
        target.errors.append("improvement_plan.work_items must be a list.")
        work_items = []
    if plan.get("work_item_count") != len(work_items):
        target.errors.append(f"improvement_plan.work_item_count expected {len(work_items)}, got {plan.get('work_item_count')!r}.")

    totals: dict[str, Any] = {
        "priority_counts": {},
        "category_counts": {},
        "task_family_counts": {},
        "rule_counts": {},
        "scenarios": set(),
        "task_families": set(),
        "rules": set(),
        "repair_backed_count": 0,
        "curriculum_backed_count": 0,
        "digest_backed_count": 0,
        "bundle_action_count": 0,
        "evidence_ref_count": 0,
    }
    seen_item_ids: set[str] = set()
    seen_routing_keys: set[str] = set()
    seen_fingerprints: set[str] = set()
    previous_sort_key: tuple[int, int, str, str, str, str] | None = None
    for index, item in enumerate(work_items):
        sort_key = _validate_improvement_work_item(
            item,
            target,
            f"improvement_plan.work_items[{index}]",
            seen_item_ids,
            seen_routing_keys,
            seen_fingerprints,
            totals,
        )
        if sort_key is not None:
            if previous_sort_key is not None and sort_key < previous_sort_key:
                target.errors.append("improvement_plan.work_items must be sorted by priority, category, task family, scenario, rule, and summary.")
            previous_sort_key = sort_key
    _validate_improvement_plan_eval_summary_linkage(source_artifacts, work_items, target, source_path)
    _validate_improvement_metrics(plan.get("metrics"), target, totals, len(work_items))
    _validate_improvement_decision(plan.get("decision"), target, plan.get("readiness"), len(work_items), totals)
    if "notes" in plan and not _is_string_list(plan.get("notes")):
        target.errors.append("improvement_plan.notes must be a list of strings when present.")
    target.details.update(
        {
            "readiness": plan.get("readiness"),
            "work_item_count": len(work_items),
            "critical_or_high_count": _count_value(totals["priority_counts"], "critical") + _count_value(totals["priority_counts"], "high"),
        }
    )

def _validate_improvement_source_artifact(name: Any, record: Any, target: ValidationTarget, source_path: Path) -> None:
    if not isinstance(name, str) or not name:
        target.errors.append("improvement_plan.source_artifacts keys must be non-empty strings.")
    label = f"improvement_plan.source_artifacts.{name}"
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    for field_name in ("kind", "path", "exists"):
        if field_name not in record:
            target.errors.append(f"{label}.{field_name} is required.")
    if record.get("kind") not in {"file", "directory"}:
        target.errors.append(f"{label}.kind must be file or directory.")
    if not isinstance(record.get("path"), str) or not record.get("path"):
        target.errors.append(f"{label}.path must be a non-empty string.")
    if not isinstance(record.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    if record.get("kind") == "file" and record.get("exists") is True:
        sha = record.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or sha != sha.lower() or any(char not in "0123456789abcdef" for char in sha):
            target.errors.append(f"{label}.sha256 must be a lowercase 64-character hex digest for existing files.")
        if not _is_non_negative_int(record.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for existing files.")
    if record.get("kind") == "file":
        _validate_improvement_source_artifact_file_fingerprint(record, target, label, source_path)
    if record.get("kind") == "directory" and record.get("exists") is True and not _is_non_negative_int(record.get("entry_count")):
        target.errors.append(f"{label}.entry_count must be a non-negative integer for existing directories.")
    if "schema_version" in record and record.get("schema_version") is not None and not isinstance(record.get("schema_version"), str):
        target.errors.append(f"{label}.schema_version must be a string or null.")
    if "passed" in record and record.get("passed") is not None and not isinstance(record.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean or null.")

def _validate_improvement_source_artifact_file_fingerprint(
    record: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path,
) -> None:
    current_path = _resolve_evidence_bundle_artifact_path(record.get("path"), source_path)
    if current_path is None:
        return
    if record.get("exists") is True:
        if not current_path.exists():
            target.errors.append(f"{label}.path must resolve to an existing file when exists is true.")
            return
        if _path_has_symlink_component(current_path, include_leaf=True) or not current_path.is_file():
            target.errors.append(f"{label}.path must resolve to a regular file when exists is true.")
            return
    elif not current_path.is_file():
        return
    if _is_non_negative_int(record.get("size_bytes")) and current_path.stat().st_size != record.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    sha = record.get("sha256")
    if isinstance(sha, str) and len(sha) == 64 and _sha256(current_path) != sha:
        target.errors.append(f"{label}.sha256 does not match the current file.")

def _validate_improvement_work_item(
    item: Any,
    target: ValidationTarget,
    label: str,
    seen_item_ids: set[str],
    seen_routing_keys: set[str],
    seen_fingerprints: set[str],
    totals: dict[str, Any],
) -> tuple[int, int, str, str, str, str] | None:
    if not isinstance(item, dict):
        target.errors.append(f"{label} must be an object.")
        return None
    for field_name in ("item_id", "category", "priority", "summary", "suggested_action", "fingerprint", "routing_key"):
        if not isinstance(item.get(field_name), str) or not item.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    item_id = item.get("item_id")
    if isinstance(item_id, str) and item_id:
        if item_id in seen_item_ids:
            target.errors.append(f"{label}.item_id duplicates {item_id!r}.")
        seen_item_ids.add(item_id)
    routing_key = item.get("routing_key")
    if isinstance(routing_key, str) and routing_key:
        if routing_key in seen_routing_keys:
            target.errors.append(f"{label}.routing_key duplicates {routing_key!r}.")
        seen_routing_keys.add(routing_key)
    category = item.get("category")
    if category not in {"bundle_action", "repair", "curriculum", "digest_action"}:
        target.errors.append(f"{label}.category must be bundle_action, repair, curriculum, or digest_action.")
        category = "digest_action"
    priority = item.get("priority")
    if priority not in set(PRIORITIES):
        target.errors.append(f"{label}.priority must be critical, high, medium, or low.")
        priority = "low"
    priority_rank = item.get("priority_rank")
    expected_rank = list(PRIORITIES).index(priority)
    if priority_rank != expected_rank:
        target.errors.append(f"{label}.priority_rank expected {expected_rank}, got {priority_rank!r}.")
    for field_name in ("scenario_id", "task_family", "rule_id", "rule_name", "task_completion_status"):
        if item.get(field_name) is not None and not isinstance(item.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string or null.")
    if item.get("score") is not None and not _is_int_between(item.get("score"), 0, 100):
        target.errors.append(f"{label}.score must be null or an integer from 0 to 100.")
    if not isinstance(item.get("sources"), dict):
        target.errors.append(f"{label}.sources must be an object.")
    else:
        _validate_improvement_item_sources(item["sources"], target, f"{label}.sources")
    _validate_evidence_refs(item.get("evidence_refs"), target, f"{label}.evidence_refs")
    if not isinstance(item.get("evidence_snippets"), list):
        target.errors.append(f"{label}.evidence_snippets must be a list.")
    if not isinstance(item.get("source_artifacts"), dict):
        target.errors.append(f"{label}.source_artifacts must be an object.")
    if not isinstance(item.get("replay"), dict):
        target.errors.append(f"{label}.replay must be an object.")

    fingerprint = item.get("fingerprint")
    expected_fingerprint = work_item_fingerprint(item)
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or fingerprint != fingerprint.lower() or any(
        char not in "0123456789abcdef" for char in fingerprint
    ):
        target.errors.append(f"{label}.fingerprint must be a lowercase 64-character hex digest.")
    elif fingerprint != expected_fingerprint:
        target.errors.append(f"{label}.fingerprint does not match item content.")
    elif fingerprint in seen_fingerprints:
        target.errors.append(f"{label}.fingerprint duplicates {fingerprint!r}.")
    elif isinstance(fingerprint, str):
        seen_fingerprints.add(fingerprint)
    if isinstance(fingerprint, str) and isinstance(item_id, str) and item_id != f"{category}:{fingerprint[:16]}":
        target.errors.append(f"{label}.item_id must match category and fingerprint.")
    if isinstance(fingerprint, str) and isinstance(routing_key, str) and routing_key != f"{category}:{priority}:{fingerprint[:12]}":
        target.errors.append(f"{label}.routing_key must match category, priority, and fingerprint.")

    _increment_count(totals["priority_counts"], priority)
    _increment_count(totals["category_counts"], category)
    _add_total(totals["scenarios"], item.get("scenario_id"))
    _add_total(totals["task_families"], item.get("task_family"))
    _add_total(totals["rules"], item.get("rule_id"))
    _increment_count(totals["task_family_counts"], item.get("task_family"))
    _increment_count(totals["rule_counts"], item.get("rule_id"))
    sources = item.get("sources") if isinstance(item.get("sources"), dict) else {}
    if _list_present(sources.get("repair_item_ids")):
        totals["repair_backed_count"] += 1
    if _list_present(sources.get("curriculum_priorities")):
        totals["curriculum_backed_count"] += 1
    if isinstance(sources.get("run_digest"), dict):
        totals["digest_backed_count"] += 1
    if category == "bundle_action":
        totals["bundle_action_count"] += 1
    evidence_refs = item.get("evidence_refs") if isinstance(item.get("evidence_refs"), list) else []
    totals["evidence_ref_count"] += len(evidence_refs)
    return (
        expected_rank,
        {"bundle_action": 0, "repair": 1, "curriculum": 2, "digest_action": 3}.get(str(category), 99),
        str(item.get("task_family") or ""),
        str(item.get("scenario_id") or ""),
        str(item.get("rule_id") or ""),
        str(item.get("summary") or ""),
    )

def _validate_improvement_item_sources(value: dict[str, Any], target: ValidationTarget, label: str) -> None:
    for field_name in ("bundle_action_ids", "repair_item_ids", "curriculum_priorities"):
        if not isinstance(value.get(field_name), list):
            target.errors.append(f"{label}.{field_name} must be a list.")
    if value.get("run_digest") is not None and not isinstance(value.get("run_digest"), dict):
        target.errors.append(f"{label}.run_digest must be an object or null.")
    for field_name in ("bundle_action_ids", "repair_item_ids"):
        if isinstance(value.get(field_name), list) and not all(isinstance(item, str) and item for item in value[field_name]):
            target.errors.append(f"{label}.{field_name} must contain non-empty strings.")
    if isinstance(value.get("curriculum_priorities"), list):
        for index, priority in enumerate(value["curriculum_priorities"]):
            priority_label = f"{label}.curriculum_priorities[{index}]"
            if not isinstance(priority, dict):
                target.errors.append(f"{priority_label} must be an object.")
                continue
            for field_name in ("task_family", "rule_id", "rule_name", "priority_band"):
                if not isinstance(priority.get(field_name), str) or not priority.get(field_name):
                    target.errors.append(f"{priority_label}.{field_name} must be a non-empty string.")
            for field_name in ("priority_score", "count", "critical_count", "max_penalty"):
                if not _is_non_negative_int(priority.get(field_name)):
                    target.errors.append(f"{priority_label}.{field_name} must be a non-negative integer.")
    if "eval_summary_items" in value:
        if not isinstance(value.get("eval_summary_items"), list):
            target.errors.append(f"{label}.eval_summary_items must be a list when present.")
        else:
            for index, source_item in enumerate(value["eval_summary_items"]):
                item_label = f"{label}.eval_summary_items[{index}]"
                if not isinstance(source_item, dict):
                    target.errors.append(f"{item_label} must be an object.")
                    continue
                for field_name in ("work_item_id", "category", "source", "label", "reason"):
                    if not isinstance(source_item.get(field_name), str) or not source_item.get(field_name):
                        target.errors.append(f"{item_label}.{field_name} must be a non-empty string.")
                if "count" in source_item and not _is_non_negative_int(source_item.get("count")):
                    target.errors.append(f"{item_label}.count must be a non-negative integer when present.")

def _validate_improvement_plan_eval_summary_linkage(
    source_artifacts: dict[str, Any],
    work_items: list[Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    artifact = source_artifacts.get("eval_summary")
    if artifact is None:
        return
    if not isinstance(artifact, dict):
        return
    eval_summary_path = _resolve_evidence_bundle_artifact_path(artifact.get("path"), source_path)
    if eval_summary_path is None or not eval_summary_path.exists() or not eval_summary_path.is_file():
        target.errors.append("improvement_plan.source_artifacts.eval_summary.path must resolve to an existing eval_summary file.")
        return
    try:
        summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        target.errors.append(f"improvement_plan.source_artifacts.eval_summary contains invalid JSON: {exc}")
        return
    if not isinstance(summary, dict):
        target.errors.append("improvement_plan.source_artifacts.eval_summary must contain a JSON object.")
        return
    _validate_eval_summary(summary, target, source_path=eval_summary_path)
    expected = _expected_eval_summary_item_records(summary)
    actual = _actual_improvement_eval_summary_item_records(work_items)
    missing = expected - actual
    extra = actual - expected
    if missing:
        target.errors.append(f"improvement_plan.work_items missing {len(missing)} eval_summary item(s) from source artifact.")
    if extra:
        target.errors.append(f"improvement_plan.work_items include {len(extra)} eval_summary item(s) not present in source artifact.")

def _expected_eval_summary_item_records(summary: dict[str, Any]) -> set[tuple[Any, ...]]:
    repair_curriculum = summary.get("repair_curriculum") if isinstance(summary.get("repair_curriculum"), dict) else {}
    items = repair_curriculum.get("items") if isinstance(repair_curriculum.get("items"), list) else []
    return {
        _eval_summary_item_record(item)
        for item in items
        if isinstance(item, dict)
    }

def _actual_improvement_eval_summary_item_records(work_items: list[Any]) -> set[tuple[Any, ...]]:
    records: set[tuple[Any, ...]] = set()
    for item in work_items:
        if not isinstance(item, dict):
            continue
        sources = item.get("sources") if isinstance(item.get("sources"), dict) else {}
        source_items = sources.get("eval_summary_items") if isinstance(sources.get("eval_summary_items"), list) else []
        for source_item in source_items:
            if isinstance(source_item, dict):
                records.add(_eval_summary_item_record(source_item))
    return records

def _eval_summary_item_record(item: dict[str, Any]) -> tuple[Any, ...]:
    raw_category = str(item.get("category") or "eval_harness")
    record = (
        str(item.get("work_item_id") or item.get("reason") or "eval_summary_item"),
        raw_category,
        str(item.get("source") or "eval_summary"),
        str(item.get("label") or "eval_summary"),
        str(item.get("reason") or "eval_summary_item"),
    )
    if "count" not in item:
        return (*record, False, None)
    count = item.get("count")
    return (*record, True, count if _is_non_negative_int(count) else 0)

def _validate_improvement_metrics(metrics: Any, target: ValidationTarget, totals: dict[str, Any], work_item_count: int) -> None:
    if not isinstance(metrics, dict):
        target.errors.append("improvement_plan.metrics must be an object.")
        return
    expected_scalars = {
        "work_item_count": work_item_count,
        "scenario_count": len(totals["scenarios"]),
        "task_family_count": len(totals["task_families"]),
        "rule_count": len(totals["rules"]),
        "repair_backed_count": totals["repair_backed_count"],
        "curriculum_backed_count": totals["curriculum_backed_count"],
        "digest_backed_count": totals["digest_backed_count"],
        "bundle_action_count": totals["bundle_action_count"],
        "evidence_ref_count": totals["evidence_ref_count"],
    }
    for field_name, expected in expected_scalars.items():
        if metrics.get(field_name) != expected:
            target.errors.append(f"improvement_plan.metrics.{field_name} expected {expected!r}, got {metrics.get(field_name)!r}.")
    expected_lists = {
        "scenarios": sorted(totals["scenarios"]),
        "task_families": sorted(totals["task_families"]),
        "rules": sorted(totals["rules"]),
    }
    for field_name, expected in expected_lists.items():
        if metrics.get(field_name) != expected:
            target.errors.append(f"improvement_plan.metrics.{field_name} expected {expected!r}, got {metrics.get(field_name)!r}.")
    for field_name in ("priority_counts", "category_counts", "task_family_counts", "rule_counts"):
        actual = _count_rows(metrics.get(field_name))
        if actual != totals[field_name]:
            target.errors.append(f"improvement_plan.metrics.{field_name} does not match work items.")

def _validate_improvement_decision(
    decision: Any,
    target: ValidationTarget,
    readiness: Any,
    work_item_count: int,
    totals: dict[str, Any],
) -> None:
    if not isinstance(decision, dict):
        target.errors.append("improvement_plan.decision must be an object.")
        return
    if decision.get("readiness") != readiness:
        target.errors.append("improvement_plan.decision.readiness must match improvement_plan.readiness.")
    if decision.get("recommendation") not in {"fix_handoff", "run_improvement_iteration", "review_improvement_opportunities", "promote_or_monitor"}:
        target.errors.append("improvement_plan.decision.recommendation has an unknown value.")
    if not isinstance(decision.get("summary"), str) or not decision.get("summary"):
        target.errors.append("improvement_plan.decision.summary must be a non-empty string.")
    if decision.get("work_item_count") != work_item_count:
        target.errors.append(f"improvement_plan.decision.work_item_count expected {work_item_count}, got {decision.get('work_item_count')!r}.")
    critical_or_high = _count_value(totals["priority_counts"], "critical") + _count_value(totals["priority_counts"], "high")
    if decision.get("critical_or_high_count") != critical_or_high:
        target.errors.append(
            f"improvement_plan.decision.critical_or_high_count expected {critical_or_high}, got {decision.get('critical_or_high_count')!r}."
        )
    if not _is_non_negative_int(decision.get("source_bundle_next_action_count")):
        target.errors.append("improvement_plan.decision.source_bundle_next_action_count must be a non-negative integer.")
    top = decision.get("top_work_items")
    if not isinstance(top, list):
        target.errors.append("improvement_plan.decision.top_work_items must be a list.")
    elif len(top) > 5:
        target.errors.append("improvement_plan.decision.top_work_items must contain at most five items.")

def _validate_improvement_ledger(
    ledger: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
    *,
    validate_sources: bool = True,
) -> None:
    _require_equal(ledger, "schema_version", IMPROVEMENT_LEDGER_SCHEMA_VERSION, target)
    if not isinstance(ledger.get("ledger_path"), str):
        target.errors.append("improvement_ledger.ledger_path must be a string.")
    else:
        _warn_absolute_public_path(target, "improvement_ledger.ledger_path", ledger.get("ledger_path"))
    if ledger.get("passed") is not True:
        target.errors.append("improvement_ledger.passed must be true.")

    plans = ledger.get("plans")
    if not isinstance(plans, list):
        target.errors.append("improvement_ledger.plans must be a list.")
        plans = []
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        target.errors.append("improvement_ledger.entries must be a list.")
        entries = []
    metrics = ledger.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("improvement_ledger.metrics must be an object.")
        metrics = {}
    if "notes" in ledger and not _is_string_list(ledger.get("notes")):
        target.errors.append("improvement_ledger.notes must be a list of strings when present.")

    for index, plan in enumerate(plans):
        _validate_improvement_ledger_plan(
            plan,
            target,
            f"improvement_ledger.plans[{index}]",
            index,
            source_path,
            validate_sources=validate_sources,
        )
    latest_index = len(plans) - 1
    totals: dict[str, Any] = {
        "work_item_count": 0,
        "open_work_item_count": 0,
        "new_work_item_count": 0,
        "recurring_work_item_count": 0,
        "resolved_work_item_count": 0,
        "critical_open_work_item_count": 0,
        "high_open_work_item_count": 0,
        "status_counts": {},
        "priority_counts": {},
        "open_priority_counts": {},
        "category_counts": {},
        "open_category_counts": {},
        "task_family_counts": {},
        "rule_counts": {},
    }
    seen_keys: set[str] = set()
    previous_sort_key: tuple[int, int, str, str, str, str] | None = None
    for index, entry in enumerate(entries):
        sort_key = _validate_improvement_ledger_entry(
            entry,
            target,
            f"improvement_ledger.entries[{index}]",
            latest_index,
            seen_keys,
            totals,
        )
        if sort_key is not None:
            if previous_sort_key is not None and sort_key < previous_sort_key:
                target.errors.append("improvement_ledger.entries must be sorted by status, priority, category, task family, rule, and key.")
            previous_sort_key = sort_key

    expected_plan_work_items = sum(
        plan.get("work_item_count") for plan in plans if isinstance(plan, dict) and _is_non_negative_int(plan.get("work_item_count"))
    )
    if ledger.get("plan_count") != len(plans):
        target.errors.append(f"improvement_ledger.plan_count expected {len(plans)}, got {ledger.get('plan_count')!r}.")
    if ledger.get("work_item_count") != expected_plan_work_items:
        target.errors.append(
            f"improvement_ledger.work_item_count expected {expected_plan_work_items}, got {ledger.get('work_item_count')!r}."
        )
    if totals["work_item_count"] != expected_plan_work_items:
        target.errors.append(
            f"improvement_ledger.entries occurrence total expected {expected_plan_work_items}, got {totals['work_item_count']}."
        )
    if ledger.get("unique_work_item_count") != len(entries):
        target.errors.append(f"improvement_ledger.unique_work_item_count expected {len(entries)}, got {ledger.get('unique_work_item_count')!r}.")
    _validate_improvement_ledger_metrics(metrics, target, totals, plans, len(entries))
    _validate_improvement_ledger_decision(ledger.get("decision"), target, plans, totals)
    target.details.update(
        {
            "plan_count": len(plans),
            "unique_work_item_count": len(entries),
            "open_work_item_count": totals["open_work_item_count"],
            "resolved_work_item_count": totals["resolved_work_item_count"],
        }
    )

def _validate_improvement_ledger_plan(
    value: Any,
    target: ValidationTarget,
    label: str,
    expected_index: int,
    source_path: Path,
    *,
    validate_sources: bool = True,
) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if value.get("index") != expected_index:
        target.errors.append(f"{label}.index expected {expected_index}, got {value.get('index')!r}.")
    for field_name in ("path", "schema_version", "readiness", "recommendation"):
        if not isinstance(value.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if isinstance(value.get("path"), str):
        _warn_absolute_public_path(target, f"{label}.path", value.get("path"))
    if value.get("schema_version") != IMPROVEMENT_PLAN_SCHEMA_VERSION:
        target.errors.append(f"{label}.schema_version must be {IMPROVEMENT_PLAN_SCHEMA_VERSION}.")
    if value.get("readiness") not in {"ready", "blocked"}:
        target.errors.append(f"{label}.readiness must be ready or blocked.")
    if value.get("exists") is not True:
        target.errors.append(f"{label}.exists must be true for source improvement plans.")
    if not isinstance(value.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean.")
    for field_name in ("work_item_count", "critical_or_high_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    if value.get("exists") is True:
        sha = value.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or sha != sha.lower() or any(char not in "0123456789abcdef" for char in sha):
            target.errors.append(f"{label}.sha256 must be a lowercase 64-character hex digest for existing files.")
        size_bytes = value.get("size_bytes")
        if not _is_non_negative_int(size_bytes):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for existing files.")
        if not validate_sources:
            return
        plan_path = _resolve_gate_source_path(value.get("path"), source_path)
        if plan_path is None or not plan_path.exists():
            target.errors.append(f"{label}.path must resolve to an existing source improvement plan.")
            return
        if plan_path.is_symlink() or _path_has_symlink_component(plan_path, include_leaf=False):
            target.errors.append(f"{label}.path must resolve to a regular non-symlink source improvement plan.")
            return
        if not plan_path.is_file():
            target.errors.append(f"{label}.path must resolve to a file.")
            return
        if _is_non_negative_int(size_bytes) and plan_path.stat().st_size != size_bytes:
            target.errors.append(f"{label}.size_bytes does not match the current file.")
        if _is_sha256(sha) and _sha256(plan_path) != sha:
            target.errors.append(f"{label}.sha256 does not match the current file.")

def _validate_improvement_ledger_entry(
    entry: Any,
    target: ValidationTarget,
    label: str,
    latest_index: int,
    seen_keys: set[str],
    totals: dict[str, Any],
) -> tuple[int, int, str, str, str, str] | None:
    if not isinstance(entry, dict):
        target.errors.append(f"{label} must be an object.")
        return None
    work_key = entry.get("work_key")
    if not isinstance(work_key, str) or not work_key:
        target.errors.append(f"{label}.work_key must be a non-empty string.")
        work_key = ""
    elif work_key in seen_keys:
        target.errors.append(f"{label}.work_key duplicates {work_key!r}.")
    seen_keys.add(work_key)
    category = entry.get("category")
    if category not in {"bundle_action", "repair", "curriculum", "digest_action"}:
        target.errors.append(f"{label}.category must be bundle_action, repair, curriculum, or digest_action.")
        category = "digest_action"
    priority = entry.get("priority")
    if priority not in set(PRIORITIES):
        target.errors.append(f"{label}.priority must be critical, high, medium, or low.")
        priority = "low"
    if entry.get("status") not in {"new", "recurring", "open", "resolved"}:
        target.errors.append(f"{label}.status must be new, recurring, open, or resolved.")
    if not isinstance(entry.get("open"), bool):
        target.errors.append(f"{label}.open must be a boolean.")
    for field_name in ("summary", "suggested_action", "first_seen_path", "last_seen_path", "latest_item_id", "latest_routing_key", "latest_fingerprint"):
        if not isinstance(entry.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    for field_name in ("first_seen_path", "last_seen_path"):
        if isinstance(entry.get(field_name), str):
            _warn_absolute_public_path(target, f"{label}.{field_name}", entry.get(field_name))
    for field_name in ("scenario_id", "task_family", "rule_id", "rule_name", "task_completion_status"):
        if entry.get(field_name) is not None and not isinstance(entry.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string or null.")
    if entry.get("score") is not None and not _is_int_between(entry.get("score"), 0, 100):
        target.errors.append(f"{label}.score must be null or an integer from 0 to 100.")
    if not _is_non_negative_int(entry.get("evidence_ref_count")):
        target.errors.append(f"{label}.evidence_ref_count must be a non-negative integer.")
    occurrences = entry.get("occurrences")
    if not isinstance(occurrences, list) or not occurrences:
        target.errors.append(f"{label}.occurrences must be a non-empty list.")
        occurrences = []
    indexes: list[int] = []
    for index, occurrence in enumerate(occurrences):
        occurrence_index = _validate_improvement_ledger_occurrence(occurrence, target, f"{label}.occurrences[{index}]", category, priority)
        if occurrence_index is not None:
            indexes.append(occurrence_index)
    plan_indexes = sorted(set(indexes))
    if entry.get("occurrence_count") != len(occurrences):
        target.errors.append(f"{label}.occurrence_count expected {len(occurrences)}, got {entry.get('occurrence_count')!r}.")
    if entry.get("plan_indexes") != plan_indexes:
        target.errors.append(f"{label}.plan_indexes expected {plan_indexes!r}, got {entry.get('plan_indexes')!r}.")
    if plan_indexes:
        first_seen = plan_indexes[0]
        last_seen = plan_indexes[-1]
        expected_open = latest_index in plan_indexes
        if entry.get("first_seen_index") != first_seen:
            target.errors.append(f"{label}.first_seen_index expected {first_seen}, got {entry.get('first_seen_index')!r}.")
        if entry.get("last_seen_index") != last_seen:
            target.errors.append(f"{label}.last_seen_index expected {last_seen}, got {entry.get('last_seen_index')!r}.")
        if entry.get("open") != expected_open:
            target.errors.append(f"{label}.open expected {expected_open}, got {entry.get('open')!r}.")
        expected_status = _expected_improvement_ledger_status(plan_indexes, latest_index)
        if entry.get("status") != expected_status:
            target.errors.append(f"{label}.status expected {expected_status!r}, got {entry.get('status')!r}.")
    latest_fingerprint = entry.get("latest_fingerprint")
    if not isinstance(latest_fingerprint, str) or len(latest_fingerprint) != 64 or latest_fingerprint != latest_fingerprint.lower() or any(
        char not in "0123456789abcdef" for char in latest_fingerprint
    ):
        target.errors.append(f"{label}.latest_fingerprint must be a lowercase 64-character hex digest.")

    stable_probe = {
        "category": category,
        "scenario_id": entry.get("scenario_id"),
        "task_family": entry.get("task_family"),
        "rule_id": entry.get("rule_id"),
        "summary": entry.get("summary"),
        "sources": {},
    }
    if work_key and category in {"repair", "curriculum"} and stable_work_key(stable_probe) != work_key:
        target.errors.append(f"{label}.work_key does not match entry content.")

    totals["work_item_count"] += len(occurrences)
    _increment_count(totals["status_counts"], entry.get("status"))
    _increment_count(totals["priority_counts"], priority)
    _increment_count(totals["category_counts"], category)
    if entry.get("open") is True:
        totals["open_work_item_count"] += 1
        if entry.get("status") == "new":
            totals["new_work_item_count"] += 1
        if entry.get("status") == "recurring":
            totals["recurring_work_item_count"] += 1
        if priority == "critical":
            totals["critical_open_work_item_count"] += 1
        if priority == "high":
            totals["high_open_work_item_count"] += 1
        _increment_count(totals["open_priority_counts"], priority)
        _increment_count(totals["open_category_counts"], category)
        _increment_count(totals["task_family_counts"], entry.get("task_family"))
        _increment_count(totals["rule_counts"], entry.get("rule_id"))
    if entry.get("status") == "resolved":
        totals["resolved_work_item_count"] += 1
    return (
        {"recurring": 0, "new": 1, "open": 2, "resolved": 3}.get(str(entry.get("status")), 99),
        list(PRIORITIES).index(priority),
        str(category or ""),
        str(entry.get("task_family") or ""),
        str(entry.get("rule_id") or ""),
        str(work_key or ""),
    )

def _validate_improvement_ledger_occurrence(
    occurrence: Any,
    target: ValidationTarget,
    label: str,
    expected_category: str,
    expected_priority: str,
) -> int | None:
    if not isinstance(occurrence, dict):
        target.errors.append(f"{label} must be an object.")
        return None
    for field_name in ("plan_path", "item_id", "routing_key", "fingerprint", "priority", "category", "summary"):
        if not isinstance(occurrence.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if isinstance(occurrence.get("plan_path"), str):
        _warn_absolute_public_path(target, f"{label}.plan_path", occurrence.get("plan_path"))
    if occurrence.get("category") != expected_category:
        target.errors.append(f"{label}.category must match entry category.")
    if occurrence.get("priority") != expected_priority:
        target.errors.append(f"{label}.priority must match entry priority.")
    if not _is_non_negative_int(occurrence.get("plan_index")):
        target.errors.append(f"{label}.plan_index must be a non-negative integer.")
        return None
    fingerprint = occurrence.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or fingerprint != fingerprint.lower() or any(
        char not in "0123456789abcdef" for char in fingerprint
    ):
        target.errors.append(f"{label}.fingerprint must be a lowercase 64-character hex digest.")
    return occurrence.get("plan_index")

def _validate_improvement_ledger_metrics(
    metrics: dict[str, Any],
    target: ValidationTarget,
    totals: dict[str, Any],
    plans: list[Any],
    unique_count: int,
) -> None:
    expected_scalars = {
        "plan_count": len(plans),
        "work_item_count": totals["work_item_count"],
        "unique_work_item_count": unique_count,
        "open_work_item_count": totals["open_work_item_count"],
        "new_work_item_count": totals["new_work_item_count"],
        "recurring_work_item_count": totals["recurring_work_item_count"],
        "resolved_work_item_count": totals["resolved_work_item_count"],
        "critical_open_work_item_count": totals["critical_open_work_item_count"],
        "high_open_work_item_count": totals["high_open_work_item_count"],
    }
    for field_name, expected in expected_scalars.items():
        if metrics.get(field_name) != expected:
            target.errors.append(f"improvement_ledger.metrics.{field_name} expected {expected!r}, got {metrics.get(field_name)!r}.")
    for field_name in (
        "status_counts",
        "priority_counts",
        "open_priority_counts",
        "category_counts",
        "open_category_counts",
        "task_family_counts",
        "rule_counts",
    ):
        actual = _count_rows(metrics.get(field_name))
        if actual != totals[field_name]:
            target.errors.append(f"improvement_ledger.metrics.{field_name} does not match ledger entries.")
    plan_counts = metrics.get("plan_work_item_counts")
    expected_counts = [
        {"index": plan.get("index"), "path": plan.get("path"), "work_item_count": plan.get("work_item_count")}
        for plan in plans
        if isinstance(plan, dict)
    ]
    if isinstance(plan_counts, list):
        for index, row in enumerate(plan_counts):
            if isinstance(row, dict) and isinstance(row.get("path"), str):
                _warn_absolute_public_path(target, f"improvement_ledger.metrics.plan_work_item_counts[{index}].path", row.get("path"))
    if plan_counts != expected_counts:
        target.errors.append("improvement_ledger.metrics.plan_work_item_counts does not match plans.")

def _validate_improvement_ledger_decision(decision: Any, target: ValidationTarget, plans: list[Any], totals: dict[str, Any]) -> None:
    if not isinstance(decision, dict):
        target.errors.append("improvement_ledger.decision must be an object.")
        return
    if decision.get("readiness") not in {"ready", "blocked"}:
        target.errors.append("improvement_ledger.decision.readiness must be ready or blocked.")
    if decision.get("recommendation") not in {"fix_handoff", "continue_improvement", "review_remaining_work", "promote_or_monitor"}:
        target.errors.append("improvement_ledger.decision.recommendation has an unknown value.")
    if not isinstance(decision.get("summary"), str) or not decision.get("summary"):
        target.errors.append("improvement_ledger.decision.summary must be a non-empty string.")
    latest_index = len(plans) - 1
    if decision.get("latest_plan_index") != latest_index:
        target.errors.append(f"improvement_ledger.decision.latest_plan_index expected {latest_index}, got {decision.get('latest_plan_index')!r}.")
    if decision.get("open_work_item_count") != totals["open_work_item_count"]:
        target.errors.append("improvement_ledger.decision.open_work_item_count must match metrics.")
    critical_or_high = totals["critical_open_work_item_count"] + totals["high_open_work_item_count"]
    if decision.get("critical_or_high_open_count") != critical_or_high:
        target.errors.append("improvement_ledger.decision.critical_or_high_open_count must match open priority counts.")
    if decision.get("resolved_work_item_count") != totals["resolved_work_item_count"]:
        target.errors.append("improvement_ledger.decision.resolved_work_item_count must match resolved count.")
    top = decision.get("top_open_work_items")
    if not isinstance(top, list):
        target.errors.append("improvement_ledger.decision.top_open_work_items must be a list.")
    elif len(top) > 5:
        target.errors.append("improvement_ledger.decision.top_open_work_items must contain at most five items.")

def _expected_improvement_ledger_status(plan_indexes: list[int], latest_index: int) -> str:
    if latest_index in plan_indexes and plan_indexes[0] == latest_index:
        return "new"
    if latest_index in plan_indexes and len(plan_indexes) > 1:
        return "recurring"
    if latest_index in plan_indexes:
        return "open"
    return "resolved"

def _validate_action_ledger(
    ledger: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
    *,
    validate_sources: bool = True,
) -> None:
    _require_equal(ledger, "schema_version", ACTION_LEDGER_SCHEMA_VERSION, target)
    if not isinstance(ledger.get("ledger_path"), str):
        target.errors.append("action_ledger.ledger_path must be a string.")
    else:
        _warn_absolute_public_path(target, "action_ledger.ledger_path", ledger.get("ledger_path"))
    if ledger.get("passed") is not True:
        target.errors.append("action_ledger.passed must be true.")

    bundles = ledger.get("bundles")
    if not isinstance(bundles, list):
        target.errors.append("action_ledger.bundles must be a list.")
        bundles = []
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        target.errors.append("action_ledger.entries must be a list.")
        entries = []
    metrics = ledger.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("action_ledger.metrics must be an object.")
        metrics = {}
    notes = ledger.get("notes")
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        target.errors.append("action_ledger.notes must be a list of strings.")

    for index, bundle in enumerate(bundles):
        _validate_action_ledger_bundle(bundle, target, f"action_ledger.bundles[{index}]", index)
    latest_index = len(bundles) - 1
    action_count = 0
    open_count = 0
    new_count = 0
    recurring_count = 0
    resolved_count = 0
    status_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    artifact_counts: dict[str, int] = {}
    routing_keys: set[str] = set()
    for index, entry in enumerate(entries):
        counts = _validate_action_ledger_entry(entry, target, f"action_ledger.entries[{index}]", latest_index)
        action_count += counts["occurrence_count"]
        open_count += counts["open"]
        new_count += counts["new"]
        recurring_count += counts["recurring"]
        resolved_count += counts["resolved"]
        if isinstance(entry, dict):
            routing_key = entry.get("routing_key")
            if isinstance(routing_key, str):
                if routing_key in routing_keys:
                    target.errors.append(f"action_ledger.entries[{index}].routing_key must be unique.")
                routing_keys.add(routing_key)
            _increment(status_counts, entry.get("status"))
            _increment(priority_counts, entry.get("priority"))
            _increment(artifact_counts, entry.get("artifact"))

    expected_bundle_action_count = sum(
        bundle.get("action_count") for bundle in bundles if isinstance(bundle, dict) and _is_non_negative_int(bundle.get("action_count"))
    )
    if ledger.get("bundle_count") != len(bundles):
        target.errors.append(f"action_ledger.bundle_count expected {len(bundles)}, got {ledger.get('bundle_count')!r}.")
    if ledger.get("action_count") != expected_bundle_action_count:
        target.errors.append(
            f"action_ledger.action_count expected {expected_bundle_action_count}, got {ledger.get('action_count')!r}."
        )
    if action_count != expected_bundle_action_count:
        target.errors.append(
            f"action_ledger.entries occurrence total expected {expected_bundle_action_count}, got {action_count}."
        )
    if ledger.get("unique_action_count") != len(entries):
        target.errors.append(f"action_ledger.unique_action_count expected {len(entries)}, got {ledger.get('unique_action_count')!r}.")

    expected_metrics = {
        "bundle_count": len(bundles),
        "action_count": expected_bundle_action_count,
        "unique_action_count": len(entries),
        "open_action_count": open_count,
        "new_action_count": new_count,
        "recurring_action_count": recurring_count,
        "resolved_action_count": resolved_count,
    }
    for field_name, expected in expected_metrics.items():
        if metrics.get(field_name) != expected:
            target.errors.append(f"action_ledger.metrics.{field_name} expected {expected}, got {metrics.get(field_name)!r}.")
    _validate_action_ledger_count_rows(metrics.get("status_counts"), status_counts, target, "action_ledger.metrics.status_counts")
    _validate_action_ledger_count_rows(metrics.get("priority_counts"), priority_counts, target, "action_ledger.metrics.priority_counts")
    _validate_action_ledger_count_rows(metrics.get("artifact_counts"), artifact_counts, target, "action_ledger.metrics.artifact_counts")
    _validate_action_ledger_bundle_action_counts(metrics.get("bundle_action_counts"), bundles, target)
    if validate_sources:
        _validate_action_ledger_bundle_linkage(bundles, entries, target, source_path)
    target.details.update(
        {
            "bundle_count": len(bundles),
            "unique_action_count": len(entries),
            "open_action_count": open_count,
            "resolved_action_count": resolved_count,
        }
    )

def _validate_action_ledger_gate(gate: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(gate, "schema_version", ACTION_LEDGER_GATE_SCHEMA_VERSION, target)
    if not isinstance(gate.get("action_ledger"), str) or not gate.get("action_ledger"):
        target.errors.append("action_ledger_gate.action_ledger must be a non-empty string.")
    else:
        _warn_absolute_public_path(target, "action_ledger_gate.action_ledger", gate.get("action_ledger"))
    if not isinstance(gate.get("passed"), bool):
        target.errors.append("action_ledger_gate.passed must be a boolean.")
    checks = gate.get("checks")
    if not isinstance(checks, list):
        target.errors.append("action_ledger_gate.checks must be a list.")
        checks = []
    metrics = gate.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("action_ledger_gate.metrics must be an object.")
        metrics = {}
    if "policy" in gate:
        _validate_action_ledger_gate_policy_summary(gate.get("policy"), target)
        _validate_action_ledger_gate_policy_check_coverage(gate.get("policy"), checks, target)

    failed_checks = _validate_gate_like_checks(checks, target, "action_ledger_gate.checks")
    if gate.get("check_count") != len(checks):
        target.errors.append(f"action_ledger_gate.check_count expected {len(checks)}, got {gate.get('check_count')!r}.")
    if gate.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"action_ledger_gate.failed_check_count expected {failed_checks}, got {gate.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(gate.get("passed"), bool) and gate.get("passed") != expected_passed:
        target.errors.append("action_ledger_gate.passed must match failed_check_count.")
    _validate_action_ledger_gate_metrics(metrics, target)
    _validate_action_ledger_gate_source_linkage(gate, checks, metrics, target, source_path)
    _validate_action_ledger_gate_decision(gate.get("decision"), expected_passed, failed_checks, metrics, target)
    target.details.update(
        {
            "passed": gate.get("passed"),
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "open_action_count": metrics.get("open_action_count"),
            "recurring_action_count": metrics.get("recurring_action_count"),
        }
    )

def _validate_action_ledger_gate_decision(
    value: Any,
    expected_passed: bool,
    failed_checks: int,
    metrics: dict[str, Any],
    target: ValidationTarget,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("action_ledger_gate.decision must be an object.")
        return
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "promote_iteration" if expected_passed else "block_iteration"
    if value.get("readiness") != expected_readiness:
        target.errors.append(f"action_ledger_gate.decision.readiness expected {expected_readiness!r}, got {value.get('readiness')!r}.")
    if value.get("recommendation") != expected_recommendation:
        target.errors.append(
            "action_ledger_gate.decision.recommendation expected "
            f"{expected_recommendation!r}, got {value.get('recommendation')!r}."
        )
    if not isinstance(value.get("summary"), str) or not value.get("summary"):
        target.errors.append("action_ledger_gate.decision.summary must be a non-empty string.")
    blocking_checks = value.get("blocking_checks")
    if not isinstance(blocking_checks, list):
        target.errors.append("action_ledger_gate.decision.blocking_checks must be a list.")
        blocking_checks = []
    if value.get("blocking_check_count") != failed_checks:
        target.errors.append(
            f"action_ledger_gate.decision.blocking_check_count expected {failed_checks}, got {value.get('blocking_check_count')!r}."
        )
    if len(blocking_checks) != failed_checks:
        target.errors.append(f"action_ledger_gate.decision.blocking_checks expected {failed_checks} entries, got {len(blocking_checks)}.")
    for index, check in enumerate(blocking_checks):
        label = f"action_ledger_gate.decision.blocking_checks[{index}]"
        if not isinstance(check, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("id", "summary"):
            if not isinstance(check.get(field_name), str) or not check.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if not isinstance(check.get("scope"), dict):
            target.errors.append(f"{label}.scope must be an object.")
    key_metrics = value.get("key_metrics")
    if not isinstance(key_metrics, dict):
        target.errors.append("action_ledger_gate.decision.key_metrics must be an object.")
        return
    for field_name in (
        "bundle_count",
        "unique_action_count",
        "open_action_count",
        "new_action_count",
        "recurring_action_count",
        "resolved_action_count",
        "open_priority_counts",
    ):
        if key_metrics.get(field_name) != metrics.get(field_name):
            target.errors.append(
                f"action_ledger_gate.decision.key_metrics.{field_name} must match action_ledger_gate.metrics.{field_name}."
            )

def _validate_action_ledger_gate_metrics(metrics: dict[str, Any], target: ValidationTarget) -> None:
    count_fields = (
        "bundle_count",
        "unique_action_count",
        "open_action_count",
        "new_action_count",
        "recurring_action_count",
        "resolved_action_count",
    )
    for field_name in count_fields:
        if not _is_non_negative_int(metrics.get(field_name)):
            target.errors.append(f"action_ledger_gate.metrics.{field_name} must be a non-negative integer.")
    if all(_is_non_negative_int(metrics.get(field_name)) for field_name in count_fields):
        if metrics["unique_action_count"] != metrics["open_action_count"] + metrics["resolved_action_count"]:
            target.errors.append("action_ledger_gate.metrics.unique_action_count must equal open_action_count + resolved_action_count.")
        if metrics["open_action_count"] < metrics["new_action_count"] + metrics["recurring_action_count"]:
            target.errors.append("action_ledger_gate.metrics.open_action_count must be at least new_action_count + recurring_action_count.")
        if metrics["bundle_count"] == 0 and metrics["unique_action_count"] > 0:
            target.errors.append("action_ledger_gate.metrics.bundle_count must be positive when actions are present.")

    priority_counts = _count_rows(metrics.get("open_priority_counts"))
    if priority_counts is None:
        target.errors.append("action_ledger_gate.metrics.open_priority_counts must be a list of {id, count} objects.")
    else:
        unknown = sorted(set(priority_counts) - {"critical", "high", "medium", "low"})
        if unknown:
            target.errors.append(f"action_ledger_gate.metrics.open_priority_counts has invalid priority value(s): {', '.join(unknown)}.")
        if _is_non_negative_int(metrics.get("open_action_count")) and sum(priority_counts.values()) > metrics["open_action_count"]:
            target.errors.append("action_ledger_gate.metrics.open_priority_counts total must not exceed open_action_count.")

def _validate_action_ledger_gate_source_linkage(
    gate: dict[str, Any],
    checks: list[Any],
    metrics: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    ledger_path = _resolve_gate_source_path(gate.get("action_ledger"), source_path)
    if ledger_path is None or not ledger_path.exists():
        target.errors.append("action_ledger_gate.action_ledger must resolve to an existing action ledger.")
        return
    if _path_has_symlink_component(ledger_path, include_leaf=True) or not ledger_path.is_file():
        target.errors.append("action_ledger_gate.action_ledger must resolve to a regular non-symlink action ledger.")
        return
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        target.errors.append(f"action_ledger_gate.action_ledger is not valid UTF-8: {exc}")
        return
    except json.JSONDecodeError as exc:
        target.errors.append(f"action_ledger_gate.action_ledger contains invalid JSON: {exc}")
        return
    if not isinstance(ledger, dict):
        target.errors.append("action_ledger_gate.action_ledger must contain a JSON object.")
        return
    _validate_action_ledger(ledger, target, ledger_path)
    try:
        expected = evaluate_action_ledger_gate(
            ledger,
            action_ledger_path=gate.get("action_ledger"),
            **_action_ledger_gate_replay_options(gate, checks),
        )
    except ValueError as exc:
        target.errors.append(f"action_ledger_gate.action_ledger could not be replayed: {exc}")
        return
    if metrics != expected.get("metrics"):
        target.errors.append("action_ledger_gate.metrics must match replayed source ledger metrics.")
    if checks != expected.get("checks"):
        target.errors.append("action_ledger_gate.checks must match replayed source ledger checks.")
    if gate.get("decision") != expected.get("decision"):
        target.errors.append("action_ledger_gate.decision must match replayed source ledger decision.")

def _validate_action_ledger_gate_policy_check_coverage(value: Any, checks: list[Any], target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        return
    effective = value.get("effective")
    if not isinstance(effective, dict):
        return
    expected: Counter[tuple[Any, ...]] = Counter()
    for field_name in (
        "min_bundles",
        "max_open_actions",
        "max_new_actions",
        "max_recurring_actions",
        "min_resolved_actions",
    ):
        if field_name in effective:
            expected[(field_name, None)] += 1
    priorities = effective.get("forbid_open_priorities")
    if isinstance(priorities, list):
        for priority in priorities:
            if isinstance(priority, str):
                expected[("forbid_open_priority", priority)] += 1
    forbidden_actions = effective.get("forbid_open_actions")
    if isinstance(forbidden_actions, list):
        for action in forbidden_actions:
            if isinstance(action, str):
                expected[("forbid_open_action", action)] += 1
    required_actions = effective.get("require_resolved_actions")
    if isinstance(required_actions, list):
        for action in required_actions:
            if isinstance(action, str):
                expected[("require_resolved_action", action)] += 1
    actual = Counter(_action_ledger_gate_check_policy_key(check) for check in checks if isinstance(check, dict))
    actual.pop(None, None)
    if expected != actual:
        target.errors.append("action_ledger_gate.checks must cover action_ledger_gate.policy.effective requirements.")

def _action_ledger_gate_check_policy_key(check: dict[str, Any]) -> tuple[Any, ...] | None:
    check_id = check.get("id")
    if check_id in {
        "min_bundles",
        "max_open_actions",
        "max_new_actions",
        "max_recurring_actions",
        "min_resolved_actions",
    }:
        return (check_id, None)
    scope = check.get("scope") if isinstance(check.get("scope"), dict) else {}
    if check_id == "forbid_open_priority":
        return (check_id, scope.get("priority"))
    if check_id in {"forbid_open_action", "require_resolved_action"}:
        return (check_id, scope.get("action"))
    return None

def _action_ledger_gate_replay_options(gate: dict[str, Any], checks: list[Any]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "min_bundles": None,
        "max_open_actions": None,
        "max_new_actions": None,
        "max_recurring_actions": None,
        "min_resolved_actions": None,
        "forbid_open_priorities": [],
        "forbid_open_actions": [],
        "require_resolved_actions": [],
    }
    policy = gate.get("policy") if isinstance(gate.get("policy"), dict) else {}
    effective = policy.get("effective") if isinstance(policy.get("effective"), dict) else None
    if effective is not None:
        for field_name in (
            "min_bundles",
            "max_open_actions",
            "max_new_actions",
            "max_recurring_actions",
            "min_resolved_actions",
        ):
            if _is_non_negative_int(effective.get(field_name)):
                options[field_name] = effective[field_name]
        if _is_string_list(effective.get("forbid_open_priorities")):
            options["forbid_open_priorities"] = effective["forbid_open_priorities"]
        if _is_string_list(effective.get("forbid_open_actions")):
            options["forbid_open_actions"] = effective["forbid_open_actions"]
        if _is_string_list(effective.get("require_resolved_actions")):
            options["require_resolved_actions"] = effective["require_resolved_actions"]
        return options

    for check in checks:
        if isinstance(check, dict):
            _merge_action_ledger_gate_check_option(options, check)
    return options

def _merge_action_ledger_gate_check_option(options: dict[str, Any], check: dict[str, Any]) -> None:
    expected = check.get("expected") if isinstance(check.get("expected"), dict) else {}
    check_id = check.get("id")
    if check_id in {"min_bundles", "min_resolved_actions"} and _is_non_negative_int(expected.get("min")):
        options[check_id] = expected["min"]
    elif check_id in {"max_open_actions", "max_new_actions", "max_recurring_actions"} and _is_non_negative_int(expected.get("max")):
        options[check_id] = expected["max"]
    elif check_id == "forbid_open_priority":
        scope = check.get("scope") if isinstance(check.get("scope"), dict) else {}
        priority = scope.get("priority")
        if isinstance(priority, str) and priority:
            options["forbid_open_priorities"].append(priority)
    elif check_id == "forbid_open_action":
        scope = check.get("scope") if isinstance(check.get("scope"), dict) else {}
        action = scope.get("action")
        if isinstance(action, str) and action:
            options["forbid_open_actions"].append(action)
    elif check_id == "require_resolved_action":
        scope = check.get("scope") if isinstance(check.get("scope"), dict) else {}
        action = scope.get("action")
        if isinstance(action, str) and action:
            options["require_resolved_actions"].append(action)

def _validate_action_ledger_gate_policy_summary(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("action_ledger_gate.policy must be an object when present.")
        return
    _require_equal(value, "schema_version", ACTION_LEDGER_GATE_POLICY_SCHEMA_VERSION, target, prefix="action_ledger_gate.policy.")
    if not isinstance(value.get("path"), str) or not value.get("path"):
        target.errors.append("action_ledger_gate.policy.path must be a non-empty string.")
    else:
        _warn_absolute_public_path(target, "action_ledger_gate.policy.path", value.get("path"))
    if "description" in value and not isinstance(value.get("description"), str):
        target.errors.append("action_ledger_gate.policy.description must be a string when present.")
    effective = value.get("effective")
    if not isinstance(effective, dict):
        target.errors.append("action_ledger_gate.policy.effective must be an object.")
        return
    allowed_fields = {
        "min_bundles",
        "max_open_actions",
        "max_new_actions",
        "max_recurring_actions",
        "min_resolved_actions",
        "forbid_open_priorities",
        "forbid_open_actions",
        "require_resolved_actions",
    }
    unknown = sorted(set(effective) - allowed_fields)
    if unknown:
        target.errors.append(f"action_ledger_gate.policy.effective has unknown field(s): {', '.join(unknown)}.")
    for field_name in (
        "min_bundles",
        "max_open_actions",
        "max_new_actions",
        "max_recurring_actions",
        "min_resolved_actions",
    ):
        if field_name in effective and not _is_non_negative_int(effective.get(field_name)):
            target.errors.append(f"action_ledger_gate.policy.effective.{field_name} must be a non-negative integer.")
    for field_name in ("forbid_open_priorities", "forbid_open_actions", "require_resolved_actions"):
        if field_name in effective and not _is_string_list(effective.get(field_name)):
            target.errors.append(f"action_ledger_gate.policy.effective.{field_name} must be a list of strings.")
    priorities = effective.get("forbid_open_priorities")
    if isinstance(priorities, list):
        unknown_priorities = sorted({item for item in priorities if isinstance(item, str)} - {"critical", "high", "medium", "low"})
        if unknown_priorities:
            target.errors.append(
                "action_ledger_gate.policy.effective.forbid_open_priorities has invalid priority value(s): "
                f"{', '.join(unknown_priorities)}."
            )

def _validate_improvement_ledger_gate(gate: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(gate, "schema_version", IMPROVEMENT_LEDGER_GATE_SCHEMA_VERSION, target)
    if not isinstance(gate.get("improvement_ledger"), str) or not gate.get("improvement_ledger"):
        target.errors.append("improvement_ledger_gate.improvement_ledger must be a non-empty string.")
    else:
        _warn_absolute_public_path(target, "improvement_ledger_gate.improvement_ledger", gate.get("improvement_ledger"))
    if not isinstance(gate.get("passed"), bool):
        target.errors.append("improvement_ledger_gate.passed must be a boolean.")
    checks = gate.get("checks")
    if not isinstance(checks, list):
        target.errors.append("improvement_ledger_gate.checks must be a list.")
        checks = []
    metrics = gate.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("improvement_ledger_gate.metrics must be an object.")
        metrics = {}
    if "policy" in gate:
        _validate_improvement_ledger_gate_policy_summary(gate.get("policy"), target)

    failed_checks = _validate_gate_like_checks(checks, target, "improvement_ledger_gate.checks")
    if gate.get("check_count") != len(checks):
        target.errors.append(f"improvement_ledger_gate.check_count expected {len(checks)}, got {gate.get('check_count')!r}.")
    if gate.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"improvement_ledger_gate.failed_check_count expected {failed_checks}, got {gate.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(gate.get("passed"), bool) and gate.get("passed") != expected_passed:
        target.errors.append("improvement_ledger_gate.passed must match failed_check_count.")
    _validate_improvement_ledger_gate_metrics(metrics, target)
    _validate_improvement_ledger_gate_source_linkage(gate, checks, metrics, target, source_path)
    _validate_improvement_ledger_gate_decision(gate.get("decision"), expected_passed, failed_checks, metrics, target)
    target.details.update(
        {
            "passed": gate.get("passed"),
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "open_work_item_count": metrics.get("open_work_item_count"),
            "recurring_work_item_count": metrics.get("recurring_work_item_count"),
        }
    )

def _validate_improvement_ledger_gate_source_linkage(
    gate: dict[str, Any],
    checks: list[Any],
    metrics: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    ledger_path = _resolve_gate_source_path(gate.get("improvement_ledger"), source_path)
    if ledger_path is None or not ledger_path.exists():
        target.errors.append("improvement_ledger_gate.improvement_ledger must resolve to an existing improvement ledger.")
        return
    if _path_has_symlink_component(ledger_path, include_leaf=True) or not ledger_path.is_file():
        target.errors.append("improvement_ledger_gate.improvement_ledger must resolve to a regular non-symlink improvement ledger.")
        return
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        target.errors.append(f"improvement_ledger_gate.improvement_ledger is not valid UTF-8: {exc}")
        return
    except json.JSONDecodeError as exc:
        target.errors.append(f"improvement_ledger_gate.improvement_ledger contains invalid JSON: {exc}")
        return
    if not isinstance(ledger, dict):
        target.errors.append("improvement_ledger_gate.improvement_ledger must contain a JSON object.")
        return
    _validate_improvement_ledger(ledger, target, ledger_path)
    try:
        expected = evaluate_improvement_ledger_gate(
            ledger,
            improvement_ledger_path=gate.get("improvement_ledger"),
            **_improvement_ledger_gate_replay_options(gate, checks),
        )
    except ValueError as exc:
        target.errors.append(f"improvement_ledger_gate.improvement_ledger could not be replayed: {exc}")
        return
    if metrics != expected.get("metrics"):
        target.errors.append("improvement_ledger_gate.metrics must match replayed source ledger metrics.")
    if checks != expected.get("checks"):
        target.errors.append("improvement_ledger_gate.checks must match replayed source ledger checks.")
    if gate.get("decision") != expected.get("decision"):
        target.errors.append("improvement_ledger_gate.decision must match replayed source ledger decision.")

def _improvement_ledger_gate_replay_options(gate: dict[str, Any], checks: list[Any]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "min_plans": None,
        "max_open_work_items": None,
        "max_new_work_items": None,
        "max_recurring_work_items": None,
        "min_resolved_work_items": None,
        "max_critical_open_work_items": None,
        "max_high_open_work_items": None,
        "forbid_open_priorities": [],
        "forbid_open_categories": [],
        "forbid_open_work_keys": [],
        "require_open_work_keys": [],
        "require_resolved_work_keys": [],
    }
    policy = gate.get("policy") if isinstance(gate.get("policy"), dict) else {}
    effective = policy.get("effective") if isinstance(policy.get("effective"), dict) else None
    if effective is not None:
        for field_name in (
            "min_plans",
            "max_open_work_items",
            "max_new_work_items",
            "max_recurring_work_items",
            "min_resolved_work_items",
            "max_critical_open_work_items",
            "max_high_open_work_items",
        ):
            if _is_non_negative_int(effective.get(field_name)):
                options[field_name] = effective[field_name]
        for field_name in (
            "forbid_open_priorities",
            "forbid_open_categories",
            "forbid_open_work_keys",
            "require_open_work_keys",
            "require_resolved_work_keys",
        ):
            if _is_string_list(effective.get(field_name)):
                options[field_name] = effective[field_name]
        return options

    for check in checks:
        if isinstance(check, dict):
            _merge_improvement_ledger_gate_check_option(options, check)
    return options

def _merge_improvement_ledger_gate_check_option(options: dict[str, Any], check: dict[str, Any]) -> None:
    expected = check.get("expected") if isinstance(check.get("expected"), dict) else {}
    check_id = check.get("id")
    if check_id in {"min_plans", "min_resolved_work_items"} and _is_non_negative_int(expected.get("min")):
        options[check_id] = expected["min"]
    elif (
        check_id
        in {
            "max_open_work_items",
            "max_new_work_items",
            "max_recurring_work_items",
            "max_critical_open_work_items",
            "max_high_open_work_items",
        }
        and _is_non_negative_int(expected.get("max"))
    ):
        options[check_id] = expected["max"]
    elif check_id == "forbid_open_priority":
        _append_scope_value(options, "forbid_open_priorities", check, "priority")
    elif check_id == "forbid_open_category":
        _append_scope_value(options, "forbid_open_categories", check, "category")
    elif check_id == "forbid_open_work_key":
        _append_scope_value(options, "forbid_open_work_keys", check, "work_key")
    elif check_id == "require_open_work_key":
        _append_scope_value(options, "require_open_work_keys", check, "work_key")
    elif check_id == "require_resolved_work_key":
        _append_scope_value(options, "require_resolved_work_keys", check, "work_key")

def _append_scope_value(options: dict[str, Any], option_name: str, check: dict[str, Any], scope_name: str) -> None:
    scope = check.get("scope") if isinstance(check.get("scope"), dict) else {}
    value = scope.get(scope_name)
    if isinstance(value, str) and value:
        options[option_name].append(value)

def _validate_improvement_ledger_gate_decision(
    value: Any,
    expected_passed: bool,
    failed_checks: int,
    metrics: dict[str, Any],
    target: ValidationTarget,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("improvement_ledger_gate.decision must be an object.")
        return
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "promote_iteration" if expected_passed else "block_iteration"
    if value.get("readiness") != expected_readiness:
        target.errors.append(
            f"improvement_ledger_gate.decision.readiness expected {expected_readiness!r}, got {value.get('readiness')!r}."
        )
    if value.get("recommendation") != expected_recommendation:
        target.errors.append(
            "improvement_ledger_gate.decision.recommendation expected "
            f"{expected_recommendation!r}, got {value.get('recommendation')!r}."
        )
    if not isinstance(value.get("summary"), str) or not value.get("summary"):
        target.errors.append("improvement_ledger_gate.decision.summary must be a non-empty string.")
    blocking_checks = value.get("blocking_checks")
    if not isinstance(blocking_checks, list):
        target.errors.append("improvement_ledger_gate.decision.blocking_checks must be a list.")
        blocking_checks = []
    if value.get("blocking_check_count") != failed_checks:
        target.errors.append(
            "improvement_ledger_gate.decision.blocking_check_count expected "
            f"{failed_checks}, got {value.get('blocking_check_count')!r}."
        )
    if len(blocking_checks) != failed_checks:
        target.errors.append(
            f"improvement_ledger_gate.decision.blocking_checks expected {failed_checks} entries, got {len(blocking_checks)}."
        )
    for index, check in enumerate(blocking_checks):
        label = f"improvement_ledger_gate.decision.blocking_checks[{index}]"
        if not isinstance(check, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("id", "summary"):
            if not isinstance(check.get(field_name), str) or not check.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if not isinstance(check.get("scope"), dict):
            target.errors.append(f"{label}.scope must be an object.")
    key_metrics = value.get("key_metrics")
    if not isinstance(key_metrics, dict):
        target.errors.append("improvement_ledger_gate.decision.key_metrics must be an object.")
        return
    for field_name in (
        "plan_count",
        "unique_work_item_count",
        "open_work_item_count",
        "new_work_item_count",
        "recurring_work_item_count",
        "resolved_work_item_count",
        "critical_open_work_item_count",
        "high_open_work_item_count",
        "open_priority_counts",
        "open_category_counts",
    ):
        if key_metrics.get(field_name) != metrics.get(field_name):
            target.errors.append(
                "improvement_ledger_gate.decision.key_metrics."
                f"{field_name} must match improvement_ledger_gate.metrics.{field_name}."
            )

def _validate_improvement_ledger_gate_metrics(metrics: dict[str, Any], target: ValidationTarget) -> None:
    count_fields = (
        "plan_count",
        "unique_work_item_count",
        "open_work_item_count",
        "new_work_item_count",
        "recurring_work_item_count",
        "resolved_work_item_count",
        "critical_open_work_item_count",
        "high_open_work_item_count",
    )
    for field_name in count_fields:
        if not _is_non_negative_int(metrics.get(field_name)):
            target.errors.append(f"improvement_ledger_gate.metrics.{field_name} must be a non-negative integer.")
    if all(_is_non_negative_int(metrics.get(field_name)) for field_name in count_fields):
        if metrics["unique_work_item_count"] != metrics["open_work_item_count"] + metrics["resolved_work_item_count"]:
            target.errors.append(
                "improvement_ledger_gate.metrics.unique_work_item_count must equal open_work_item_count + resolved_work_item_count."
            )
        if metrics["open_work_item_count"] < metrics["new_work_item_count"] + metrics["recurring_work_item_count"]:
            target.errors.append(
                "improvement_ledger_gate.metrics.open_work_item_count must be at least new_work_item_count + recurring_work_item_count."
            )
        critical_high = metrics["critical_open_work_item_count"] + metrics["high_open_work_item_count"]
        if critical_high > metrics["open_work_item_count"]:
            target.errors.append(
                "improvement_ledger_gate.metrics critical/high open counts must not exceed open_work_item_count."
            )
        if metrics["plan_count"] == 0 and metrics["unique_work_item_count"] > 0:
            target.errors.append("improvement_ledger_gate.metrics.plan_count must be positive when work items are present.")

    priority_counts = _count_rows(metrics.get("open_priority_counts"))
    if priority_counts is None:
        target.errors.append("improvement_ledger_gate.metrics.open_priority_counts must be a list of {id, count} objects.")
    else:
        unknown = sorted(set(priority_counts) - set(PRIORITIES))
        if unknown:
            target.errors.append(
                f"improvement_ledger_gate.metrics.open_priority_counts has invalid priority value(s): {', '.join(unknown)}."
            )
        if _is_non_negative_int(metrics.get("open_work_item_count")) and sum(priority_counts.values()) > metrics["open_work_item_count"]:
            target.errors.append("improvement_ledger_gate.metrics.open_priority_counts total must not exceed open_work_item_count.")

    category_counts = _count_rows(metrics.get("open_category_counts"))
    if category_counts is None:
        target.errors.append("improvement_ledger_gate.metrics.open_category_counts must be a list of {id, count} objects.")
    else:
        unknown = sorted(set(category_counts) - {"bundle_action", "repair", "curriculum", "digest_action"})
        if unknown:
            target.errors.append(
                f"improvement_ledger_gate.metrics.open_category_counts has invalid category value(s): {', '.join(unknown)}."
            )
        if _is_non_negative_int(metrics.get("open_work_item_count")) and sum(category_counts.values()) > metrics["open_work_item_count"]:
            target.errors.append("improvement_ledger_gate.metrics.open_category_counts total must not exceed open_work_item_count.")

def _validate_improvement_ledger_gate_policy_summary(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("improvement_ledger_gate.policy must be an object when present.")
        return
    _require_equal(
        value,
        "schema_version",
        IMPROVEMENT_LEDGER_GATE_POLICY_SCHEMA_VERSION,
        target,
        prefix="improvement_ledger_gate.policy.",
    )
    if not isinstance(value.get("path"), str) or not value.get("path"):
        target.errors.append("improvement_ledger_gate.policy.path must be a non-empty string.")
    else:
        _warn_absolute_public_path(target, "improvement_ledger_gate.policy.path", value.get("path"))
    if "description" in value and not isinstance(value.get("description"), str):
        target.errors.append("improvement_ledger_gate.policy.description must be a string when present.")
    effective = value.get("effective")
    if not isinstance(effective, dict):
        target.errors.append("improvement_ledger_gate.policy.effective must be an object.")
        return
    allowed_fields = {
        "min_plans",
        "max_open_work_items",
        "max_new_work_items",
        "max_recurring_work_items",
        "min_resolved_work_items",
        "max_critical_open_work_items",
        "max_high_open_work_items",
        "forbid_open_priorities",
        "forbid_open_categories",
        "forbid_open_work_keys",
        "require_open_work_keys",
        "require_resolved_work_keys",
    }
    unknown = sorted(set(effective) - allowed_fields)
    if unknown:
        target.errors.append(f"improvement_ledger_gate.policy.effective has unknown field(s): {', '.join(unknown)}.")
    for field_name in (
        "min_plans",
        "max_open_work_items",
        "max_new_work_items",
        "max_recurring_work_items",
        "min_resolved_work_items",
        "max_critical_open_work_items",
        "max_high_open_work_items",
    ):
        if field_name in effective and not _is_non_negative_int(effective.get(field_name)):
            target.errors.append(f"improvement_ledger_gate.policy.effective.{field_name} must be a non-negative integer.")
    for field_name in (
        "forbid_open_priorities",
        "forbid_open_categories",
        "forbid_open_work_keys",
        "require_open_work_keys",
        "require_resolved_work_keys",
    ):
        if field_name in effective and not _is_string_list(effective.get(field_name)):
            target.errors.append(f"improvement_ledger_gate.policy.effective.{field_name} must be a list of strings.")
    priorities = effective.get("forbid_open_priorities")
    if isinstance(priorities, list):
        unknown_priorities = sorted({item for item in priorities if isinstance(item, str)} - set(PRIORITIES))
        if unknown_priorities:
            target.errors.append(
                "improvement_ledger_gate.policy.effective.forbid_open_priorities has invalid priority value(s): "
                f"{', '.join(unknown_priorities)}."
            )
    categories = effective.get("forbid_open_categories")
    if isinstance(categories, list):
        unknown_categories = sorted(
            {item for item in categories if isinstance(item, str)} - {"bundle_action", "repair", "curriculum", "digest_action"}
        )
        if unknown_categories:
            target.errors.append(
                "improvement_ledger_gate.policy.effective.forbid_open_categories has invalid category value(s): "
                f"{', '.join(unknown_categories)}."
            )

def _validate_decision_gate(gate: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(gate, "schema_version", DECISION_GATE_SCHEMA_VERSION, target)
    if not isinstance(gate.get("artifact"), str) or not gate.get("artifact"):
        target.errors.append("decision_gate.artifact must be a non-empty string.")
    else:
        _warn_absolute_public_path(target, "decision_gate.artifact", gate.get("artifact"))
    source_artifact = gate.get("source_artifact")
    if not isinstance(source_artifact, dict):
        target.errors.append("decision_gate.source_artifact must be an object.")
        source_artifact = {}
    _validate_decision_gate_source_artifact(source_artifact, target, source_path)
    if isinstance(gate.get("artifact"), str) and isinstance(source_artifact.get("path"), str) and gate.get("artifact") != source_artifact.get("path"):
        target.errors.append("decision_gate.artifact must match decision_gate.source_artifact.path.")
    if not isinstance(gate.get("passed"), bool):
        target.errors.append("decision_gate.passed must be a boolean.")
    if not isinstance(gate.get("expected_recommendation"), str) or not gate.get("expected_recommendation"):
        target.errors.append("decision_gate.expected_recommendation must be a non-empty string.")
    if gate.get("expected_readiness") is not None and not isinstance(gate.get("expected_readiness"), str):
        target.errors.append("decision_gate.expected_readiness must be a string or null.")
    if not isinstance(gate.get("require_passed"), bool):
        target.errors.append("decision_gate.require_passed must be a boolean.")
    if not _is_string_list(gate.get("notes")):
        target.errors.append("decision_gate.notes must be a list of strings.")

    checks = gate.get("checks")
    if not isinstance(checks, list):
        target.errors.append("decision_gate.checks must be a list.")
        checks = []
    failed_checks = _validate_gate_like_checks(checks, target, "decision_gate.checks")
    if gate.get("check_count") != len(checks):
        target.errors.append(f"decision_gate.check_count expected {len(checks)}, got {gate.get('check_count')!r}.")
    if gate.get("failed_check_count") != failed_checks:
        target.errors.append(f"decision_gate.failed_check_count expected {failed_checks}, got {gate.get('failed_check_count')!r}.")
    expected_passed = failed_checks == 0
    if isinstance(gate.get("passed"), bool) and gate.get("passed") != expected_passed:
        target.errors.append("decision_gate.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "allow_promotion" if expected_passed else "block_promotion"
    if gate.get("readiness") != expected_readiness:
        target.errors.append(f"decision_gate.readiness expected {expected_readiness!r}, got {gate.get('readiness')!r}.")
    if gate.get("recommendation") != expected_recommendation:
        target.errors.append(f"decision_gate.recommendation expected {expected_recommendation!r}, got {gate.get('recommendation')!r}.")

    source = gate.get("source_decision")
    if not isinstance(source, dict):
        target.errors.append("decision_gate.source_decision must be an object.")
        source = {}
    for field_name in ("schema_version", "recommendation", "readiness", "summary"):
        if not isinstance(source.get(field_name), str):
            target.errors.append(f"decision_gate.source_decision.{field_name} must be a string.")
    if source.get("passed") is not None and not isinstance(source.get("passed"), bool):
        target.errors.append("decision_gate.source_decision.passed must be a boolean or null.")
    if source.get("blocking_check_count") is not None and not _is_non_negative_int(source.get("blocking_check_count")):
        target.errors.append("decision_gate.source_decision.blocking_check_count must be a non-negative integer or null.")
    if not isinstance(source.get("key_metrics"), dict):
        target.errors.append("decision_gate.source_decision.key_metrics must be an object.")
    _validate_decision_gate_source_decision_matches_artifact(source, source_artifact, target, source_path)
    target.details.update(
        {
            "passed": gate.get("passed"),
            "recommendation": gate.get("recommendation"),
            "source_recommendation": source.get("recommendation"),
            "source_sha256": source_artifact.get("sha256"),
            "failed_check_count": failed_checks,
        }
    )

def _validate_decision_gate_source_artifact(record: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    if not isinstance(record.get("path"), str) or not record.get("path"):
        target.errors.append("decision_gate.source_artifact.path must be a non-empty string.")
    else:
        _warn_absolute_public_path(target, "decision_gate.source_artifact.path", record.get("path"))
    if record.get("kind") != "file":
        target.errors.append("decision_gate.source_artifact.kind must be file.")
    if not isinstance(record.get("exists"), bool):
        target.errors.append("decision_gate.source_artifact.exists must be a boolean.")
    elif record.get("exists") is not True:
        target.errors.append("decision_gate.source_artifact.exists must be true.")
    _validate_decision_gate_source_artifact_hash(record, target, source_path)

def _validate_decision_gate_source_artifact_hash(record: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    if record.get("kind") != "file" or record.get("exists") is not True:
        return
    if "regular_file" in record and not isinstance(record.get("regular_file"), bool):
        target.errors.append("decision_gate.source_artifact.regular_file must be a boolean when present.")
    if "symlink" in record and not isinstance(record.get("symlink"), bool):
        target.errors.append("decision_gate.source_artifact.symlink must be a boolean when present.")
    if record.get("regular_file") is False:
        target.errors.append("decision_gate.source_artifact.regular_file must be true when present.")
    if not _is_non_negative_int(record.get("size_bytes")):
        target.errors.append("decision_gate.source_artifact.size_bytes must be a non-negative integer for existing files.")
    if not _is_sha256(record.get("sha256")):
        target.errors.append("decision_gate.source_artifact.sha256 must be a SHA-256 hex string for existing files.")
        return
    file_path = _resolve_gate_source_path(record.get("path"), source_path)
    if file_path is None:
        target.errors.append("decision_gate.source_artifact.path must resolve to a local file.")
        return
    if file_path.is_symlink():
        target.errors.append("decision_gate.source_artifact.path must not resolve to a symlink.")
        return
    if _path_has_symlink_component(file_path, include_leaf=False):
        target.errors.append("decision_gate.source_artifact.path must not traverse symlinked components.")
        return
    if not file_path.exists() or not file_path.is_file():
        target.errors.append("decision_gate.source_artifact.path does not resolve to an existing file.")
        return
    if file_path.stat().st_size != record.get("size_bytes"):
        target.errors.append("decision_gate.source_artifact.size_bytes does not match the current file.")
    if _sha256(file_path) != record.get("sha256"):
        target.errors.append("decision_gate.source_artifact.sha256 does not match the current file.")

def _validate_decision_gate_source_decision_matches_artifact(
    source_decision: dict[str, Any],
    source_artifact: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    file_path = _resolve_gate_source_path(source_artifact.get("path"), source_path)
    if file_path is None or not file_path.exists() or not file_path.is_file():
        return
    if file_path.is_symlink() or _path_has_symlink_component(file_path, include_leaf=False):
        return
    try:
        artifact = json.loads(file_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        target.errors.append(f"decision_gate.source_artifact is not valid UTF-8: {exc}")
        return
    except json.JSONDecodeError as exc:
        target.errors.append(f"decision_gate.source_artifact contains invalid JSON: {exc}")
        return
    if not isinstance(artifact, dict):
        target.errors.append("decision_gate.source_artifact must contain a JSON object.")
        return
    for error in decision_gate_source_contract_errors(artifact):
        target.errors.append(f"decision_gate.source_artifact contract error: {error}.")
    actual_decision = artifact.get("decision") if isinstance(artifact.get("decision"), dict) else {}
    expected = {
        "schema_version": str(artifact.get("schema_version") or ""),
        "passed": artifact.get("passed") if isinstance(artifact.get("passed"), bool) else None,
        "recommendation": str(actual_decision.get("recommendation") or ""),
        "readiness": str(actual_decision.get("readiness") or ""),
        "summary": str(actual_decision.get("summary") or ""),
        "blocking_check_count": actual_decision.get("blocking_check_count")
        if _is_non_negative_int(actual_decision.get("blocking_check_count"))
        else None,
        "key_metrics": actual_decision.get("key_metrics") if isinstance(actual_decision.get("key_metrics"), dict) else {},
    }
    for field_name, expected_value in expected.items():
        if source_decision.get(field_name) != expected_value:
            target.errors.append(f"decision_gate.source_decision.{field_name} must match current source artifact.")

def _validate_promotion_cards(cards: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(cards, "schema_version", PROMOTION_CARDS_SCHEMA_VERSION, target)
    for field_name in ("manifest_path", "cards_dir"):
        if not isinstance(cards.get(field_name), str) or not cards.get(field_name):
            target.errors.append(f"promotion_cards.{field_name} must be a non-empty string.")
    if not isinstance(cards.get("passed"), bool):
        target.errors.append("promotion_cards.passed must be a boolean.")
    checks = cards.get("checks")
    if not isinstance(checks, list):
        target.errors.append("promotion_cards.checks must be a list.")
        checks = []
    failed_checks = _validate_gate_like_checks(checks, target, "promotion_cards.checks")
    if cards.get("check_count") != len(checks):
        target.errors.append(f"promotion_cards.check_count expected {len(checks)}, got {cards.get('check_count')!r}.")
    if cards.get("failed_check_count") != failed_checks:
        target.errors.append(f"promotion_cards.failed_check_count expected {failed_checks}, got {cards.get('failed_check_count')!r}.")
    expected_passed = failed_checks == 0
    if isinstance(cards.get("passed"), bool) and cards["passed"] != expected_passed:
        target.errors.append("promotion_cards.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "use_cards_for_promotion_decision" if expected_passed else "regenerate_or_block_promotion"
    if cards.get("readiness") != expected_readiness:
        target.errors.append(f"promotion_cards.readiness expected {expected_readiness!r}, got {cards.get('readiness')!r}.")
    if cards.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"promotion_cards.recommendation expected {expected_recommendation!r}, got {cards.get('recommendation')!r}."
        )

    candidate = cards.get("candidate")
    if not isinstance(candidate, dict):
        target.errors.append("promotion_cards.candidate must be an object.")
        candidate = {}
    for field_name in ("id", "model_source", "license_status"):
        if not isinstance(candidate.get(field_name), str):
            target.errors.append(f"promotion_cards.candidate.{field_name} must be a string.")
    dataset = cards.get("dataset")
    if not isinstance(dataset, dict):
        target.errors.append("promotion_cards.dataset must be an object.")
        dataset = {}
    if not isinstance(dataset.get("id"), str):
        target.errors.append("promotion_cards.dataset.id must be a string.")

    artifacts = cards.get("artifacts")
    if not isinstance(artifacts, dict):
        target.errors.append("promotion_cards.artifacts must be an object.")
        artifacts = {}
    for role in ("model_card", "dataset_card", *PROMOTION_CARDS_REQUIRED_INPUTS):
        _validate_fingerprinted_artifact(artifacts.get(role), role, target, source_path, "promotion_cards.artifacts")
    _validate_promotion_card_text(artifacts.get("model_card"), target, source_path, "promotion_cards.artifacts.model_card")
    _validate_promotion_card_text(artifacts.get("dataset_card"), target, source_path, "promotion_cards.artifacts.dataset_card")
    _validate_promotion_cards_metrics(cards.get("metrics"), checks, target)
    if not _is_string_list(cards.get("notes")):
        target.errors.append("promotion_cards.notes must be a list of strings.")
    target.details.update(
        {
            "passed": cards.get("passed"),
            "recommendation": cards.get("recommendation"),
            "failed_check_count": failed_checks,
            "candidate_id": candidate.get("id"),
            "dataset_id": dataset.get("id"),
        }
    )

def _validate_promotion_card_text(value: Any, target: ValidationTarget, source_path: Path, label: str) -> None:
    if not isinstance(value, dict) or value.get("exists") is not True:
        return
    card_path = _resolve_promotion_decision_artifact_path(value.get("path"), source_path, "file")
    if card_path is None or not card_path.exists() or not card_path.is_file():
        return
    try:
        text = card_path.read_text(encoding="utf-8").lower()
    except OSError:
        return
    markers = [marker for marker in ("unsupported claim", "todo", "tbd") if marker in text]
    if markers:
        target.errors.append(f"{label} contains unsupported claim marker(s): {', '.join(markers)}.")

def _validate_promotion_cards_metrics(value: Any, checks: list[Any], target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("promotion_cards.metrics must be an object.")
        return
    expected_failed = sum(1 for check in checks if isinstance(check, dict) and check.get("passed") is False)
    expected = {
        "check_count": len(checks),
        "failed_check_count": expected_failed,
        "required_input_count": len(PROMOTION_CARDS_REQUIRED_INPUTS),
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"promotion_cards.metrics.{field_name} expected {expected_value!r}, got {value.get(field_name)!r}.")
    for field_name in ("task_completion_regression_count", "new_critical_failure_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"promotion_cards.metrics.{field_name} must be a non-negative integer.")

_PROMOTION_FINGERPRINTED_ARTIFACT_KEYS = {
    "role",
    "path",
    "exists",
    "kind",
    "sha256",
    "size_bytes",
    "file_count",
    "schema_version",
}

_PROMOTION_GATE_CHECK_KEYS = {"id", "passed", "actual", "expected", "summary", "scope"}

_PROMOTION_POLICY_KEYS = {
    "schema_version",
    "id",
    "description",
    "source",
    "required_artifacts",
    "release_required_artifacts",
    "allowed_candidate_classes",
    "allowed_champion_classes",
    "limits",
    "forbid_new_critical_rules",
    "forbid_regressed_rules",
    "requirements",
    "artifact",
}

_PROMOTION_POLICY_REQUIREMENT_KEYS = {
    "require_known_license",
    "require_accepted_terms",
    "require_rollback_metadata",
    "require_supported_cards",
    "require_artifact_validation",
}

_PROMOTION_DECISION_KEYS = {
    "schema_version",
    "decision_path",
    "passed",
    "readiness",
    "recommendation",
    "models",
    "decision",
    "external_eval_lineage",
    "check_count",
    "failed_check_count",
    "checks",
    "artifacts",
    "policy",
    "metrics",
    "alias_update",
    "notes",
    "metadata",
}

_PROMOTION_DECISION_MODELS_KEYS = {"candidate", "champion", "rollback"}

_PROMOTION_DECISION_MODEL_KEYS = {"id", "class"}

_PROMOTION_DECISION_BLOCK_KEYS = {
    "readiness",
    "recommendation",
    "summary",
    "blocking_check_count",
    "blocking_checks",
    "key_metrics",
}

_PROMOTION_DECISION_BLOCKING_CHECK_KEYS = {"id", "summary", "scope"}

_PROMOTION_DECISION_METRICS_KEYS = {
    "check_count",
    "failed_check_count",
    "external_eval_result_count",
    "required_artifact_count",
    "policy_required_artifact_count",
    "policy_release_required_artifact_count",
    "task_completion_regression_count",
    "baseline_win_count",
    "contract_drift_count",
    "unverified_contract_count",
    "new_critical_failure_count",
    "rule_regression_count",
}

_PROMOTION_DECISION_ALIAS_UPDATE_KEYS = {"authorized", "recommendation", "aliases"}

_PROMOTION_DECISION_ALIAS_ROW_KEYS = {"alias", "previous_target", "target"}

_PROMOTION_EXTERNAL_EVAL_LINEAGE_KEYS = {
    "result_count",
    "summary_result_count",
    "exact_result_set",
    "candidate_model_bound",
    "evidence_bundle_summary_bound",
    "evidence_bundle_semantically_valid",
    "eval_summary_semantically_valid",
    "external_results_semantically_valid",
    "semantic_validation_passed",
    "governance_ready",
    "passed",
    "results",
}

_PROMOTION_EXTERNAL_EVAL_RESULT_KEYS = {
    "artifact",
    "adapter_id",
    "model_id",
    "plan_sha256",
    "heldout_manifest_sha256",
    "integrity_passed",
    "execution_status",
    "benchmark_status",
    "coverage_complete",
    "governance_readiness",
    "external_eval_claims_allowed",
}

_PROMOTION_ALIAS_APPLY_KEYS = {
    "schema_version",
    "receipt_path",
    "passed",
    "readiness",
    "recommendation",
    "check_count",
    "failed_check_count",
    "checks",
    "promotion_decision",
    "promotion_decision_validation",
    "registry_before",
    "registry_after",
    "alias_history_entry",
    "artifacts",
    "metrics",
    "notes",
    "metadata",
}

_PROMOTION_ALIAS_APPLY_DECISION_REF_KEYS = {
    "path",
    "sha256",
    "size_bytes",
    "candidate_id",
    "champion_previous_target",
    "rollback_id",
}

_PROMOTION_ALIAS_APPLY_ARTIFACT_KEYS = {"registry", "promotion_decision"}

_PROMOTION_ALIAS_APPLY_METRICS_KEYS = {
    "check_count",
    "failed_check_count",
    "registered_model_count",
    "alias_count_before",
    "alias_count_after",
    "alias_history_count_before",
    "alias_history_count_after",
}

_PROMOTION_ALIAS_HISTORY_ENTRY_KEYS = {"promotion_decision_sha256", "previous_aliases", "updated_aliases"}

_PROMOTION_ROLLBACK_RECEIPT_KEYS = {
    "schema_version",
    "receipt_path",
    "passed",
    "available",
    "readiness",
    "recommendation",
    "rollback_id",
    "target_model_id",
    "champion_id",
    "check_count",
    "failed_check_count",
    "checks",
    "rollback",
    "registry",
    "artifacts",
    "metrics",
    "notes",
    "metadata",
}

_PROMOTION_ROLLBACK_BLOCK_KEYS = {"id", "target_model_id", "champion_id", "available"}

_PROMOTION_ROLLBACK_ARTIFACT_KEYS = {"registry"}

_PROMOTION_ROLLBACK_METRICS_KEYS = {"check_count", "failed_check_count", "registered_model_count", "alias_count"}

_PROMOTION_RELEASE_RECORD_KEYS = {
    "schema_version",
    "release_record_path",
    "passed",
    "readiness",
    "recommendation",
    "release",
    "check_count",
    "failed_check_count",
    "checks",
    "artifacts",
    "policy",
    "artifact_validation",
    "bindings",
    "metrics",
    "notes",
    "metadata",
}

_PROMOTION_RELEASE_RECORD_RELEASE_KEYS = {"id", "candidate_id", "champion_previous_target", "rollback_id", "dataset_id"}

_PROMOTION_RELEASE_RECORD_BINDINGS_KEYS = {
    "promotion_decision_sha256",
    "promotion_cards_sha256",
    "promotion_alias_apply_sha256",
    "rollback_metadata_sha256",
    "compare_gate_sha256",
    "release_notes_sha256",
    "model_card_sha256",
    "dataset_card_sha256",
}

_PROMOTION_RELEASE_RECORD_METRICS_KEYS = {
    "check_count",
    "failed_check_count",
    "required_artifact_count",
    "release_notes_size_bytes",
}

_PROMOTION_RELEASE_RECORD_POLICY_KEYS = {"promotion_decision_policy", "release_policy"}

_COMPACT_VALIDATION_SUMMARY_KEYS = {"passed", "target_count", "error_count", "warning_count", "targets"}

_COMPACT_VALIDATION_TARGET_KEYS = {"type", "passed", "error_count", "warning_count"}

_PROMOTION_ARCHIVE_KEYS = {
    "schema_version",
    "archive_path",
    "manifest_path",
    "passed",
    "self_contained",
    "require_self_contained",
    "artifacts",
    "missing",
    "relationships",
    "metrics",
    "notes",
}

_PROMOTION_ARCHIVE_ARTIFACT_KEYS = {
    "index",
    "name",
    "role",
    "path",
    "original_path",
    "exists",
    "schema_version",
    "size_bytes",
    "sha256",
}

_PROMOTION_ARCHIVE_MISSING_KEYS = {"role", "index", "reason"}

_PROMOTION_ARCHIVE_METRICS_KEYS = {
    "artifact_count",
    "decision_gate_count",
    "promotion_release_record_count",
    "source_artifact_count",
    "missing_count",
    "role_counts",
    "missing_role_counts",
    "unique_sha256_count",
}

def _validate_promotion_gate_check_keys(checks: list[Any], target: ValidationTarget, label: str) -> None:
    for index, check in enumerate(checks):
        if isinstance(check, dict):
            _validate_allowed_keys(check, _PROMOTION_GATE_CHECK_KEYS, target, f"{label}[{index}]")

def _validate_promotion_fingerprinted_artifact(
    value: Any,
    role: str,
    target: ValidationTarget,
    source_path: Path,
    prefix: str,
) -> None:
    if isinstance(value, dict):
        _validate_allowed_keys(value, _PROMOTION_FINGERPRINTED_ARTIFACT_KEYS, target, f"{prefix}.{role}")
    _validate_fingerprinted_artifact(value, role, target, source_path, prefix)

def _validate_promotion_alias_apply(receipt: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(receipt, _PROMOTION_ALIAS_APPLY_KEYS, target, "promotion_alias_apply")
    _require_equal(receipt, "schema_version", PROMOTION_ALIAS_APPLY_SCHEMA_VERSION, target)
    if not isinstance(receipt.get("receipt_path"), str):
        target.errors.append("promotion_alias_apply.receipt_path must be a string.")
    if not isinstance(receipt.get("passed"), bool):
        target.errors.append("promotion_alias_apply.passed must be a boolean.")
    checks = receipt.get("checks")
    if not isinstance(checks, list):
        target.errors.append("promotion_alias_apply.checks must be a list.")
        checks = []
    failed_checks = _validate_gate_like_checks(checks, target, "promotion_alias_apply.checks")
    _validate_promotion_gate_check_keys(checks, target, "promotion_alias_apply.checks")
    actual_check_ids = [
        check.get("id") if isinstance(check, dict) else None for check in checks
    ]
    if actual_check_ids != list(PROMOTION_ALIAS_APPLY_CHECK_IDS):
        target.errors.append(
            "promotion_alias_apply.checks must exactly match the required ordered alias-application check contract."
        )
    if receipt.get("check_count") != len(checks):
        target.errors.append(f"promotion_alias_apply.check_count expected {len(checks)}, got {receipt.get('check_count')!r}.")
    if receipt.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"promotion_alias_apply.failed_check_count expected {failed_checks}, got {receipt.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(receipt.get("passed"), bool) and receipt["passed"] != expected_passed:
        target.errors.append("promotion_alias_apply.passed must match failed_check_count.")
    expected_readiness = "applied" if expected_passed else "blocked"
    expected_recommendation = "alias_update_applied" if expected_passed else "hold_aliases"
    if receipt.get("readiness") != expected_readiness:
        target.errors.append(f"promotion_alias_apply.readiness expected {expected_readiness!r}, got {receipt.get('readiness')!r}.")
    if receipt.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"promotion_alias_apply.recommendation expected {expected_recommendation!r}, got {receipt.get('recommendation')!r}."
        )

    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        target.errors.append("promotion_alias_apply.artifacts must be an object.")
        artifacts = {}
    else:
        _validate_allowed_keys(artifacts, _PROMOTION_ALIAS_APPLY_ARTIFACT_KEYS, target, "promotion_alias_apply.artifacts")
    for role in ("registry", "promotion_decision"):
        artifact = artifacts.get(role)
        _validate_promotion_fingerprinted_artifact(
            artifact,
            role,
            target,
            source_path,
            "promotion_alias_apply.artifacts",
        )
        if expected_passed:
            record = artifact if isinstance(artifact, dict) else {}
            if record.get("exists") is not True or record.get("kind") != "file":
                target.errors.append(
                    f"promotion_alias_apply.artifacts.{role} must be an existing file for applied receipts."
                )
            if not _is_sha256(record.get("sha256")):
                target.errors.append(
                    f"promotion_alias_apply.artifacts.{role}.sha256 must be a SHA-256 hex string for applied receipts."
                )
            if not _is_non_negative_int(record.get("size_bytes")):
                target.errors.append(
                    f"promotion_alias_apply.artifacts.{role}.size_bytes must be a non-negative integer for applied receipts."
                )

    decision_ref = receipt.get("promotion_decision")
    if not isinstance(decision_ref, dict):
        target.errors.append("promotion_alias_apply.promotion_decision must be an object.")
        decision_ref = {}
    else:
        _validate_allowed_keys(
            decision_ref,
            _PROMOTION_ALIAS_APPLY_DECISION_REF_KEYS,
            target,
            "promotion_alias_apply.promotion_decision",
        )
    for field_name in ("path", "candidate_id", "champion_previous_target", "rollback_id"):
        if not isinstance(decision_ref.get(field_name), str):
            target.errors.append(f"promotion_alias_apply.promotion_decision.{field_name} must be a string.")
    if decision_ref.get("sha256") is not None and not _is_sha256(decision_ref.get("sha256")):
        target.errors.append("promotion_alias_apply.promotion_decision.sha256 must be a SHA-256 hex string or null.")
    if not _is_non_negative_int(decision_ref.get("size_bytes")):
        target.errors.append("promotion_alias_apply.promotion_decision.size_bytes must be a non-negative integer.")
    decision_artifact = artifacts.get("promotion_decision") if isinstance(artifacts.get("promotion_decision"), dict) else {}
    if decision_ref.get("sha256") != decision_artifact.get("sha256"):
        target.errors.append("promotion_alias_apply.promotion_decision.sha256 must match artifacts.promotion_decision.sha256.")
    _validate_ref_matches_fingerprinted_artifact(
        decision_ref,
        decision_artifact,
        target,
        "promotion_alias_apply.promotion_decision",
        "artifacts.promotion_decision",
        fields=("path", "sha256", "size_bytes"),
    )
    _validate_promotion_alias_apply_decision_validation(
        receipt.get("promotion_decision_validation"),
        expected_passed,
        target,
    )

    before = _validate_alias_snapshot(
        receipt.get("registry_before"), target, "promotion_alias_apply.registry_before", allow_fingerprint=False
    )
    after = _validate_alias_snapshot(
        receipt.get("registry_after"),
        target,
        "promotion_alias_apply.registry_after",
        require_fingerprint=expected_passed,
    )
    registry_artifact = artifacts.get("registry") if isinstance(artifacts.get("registry"), dict) else {}
    if after.get("sha256") != registry_artifact.get("sha256"):
        target.errors.append("promotion_alias_apply.registry_after.sha256 must match artifacts.registry.sha256.")
    _validate_ref_matches_fingerprinted_artifact(
        receipt.get("registry_after") if isinstance(receipt.get("registry_after"), dict) else {},
        registry_artifact,
        target,
        "promotion_alias_apply.registry_after",
        "artifacts.registry",
        fields=("path", "sha256", "size_bytes"),
    )
    if expected_passed:
        expected_aliases = {
            "candidate": str(decision_ref.get("candidate_id") or ""),
            "champion": str(decision_ref.get("candidate_id") or ""),
            "rollback": str(decision_ref.get("rollback_id") or ""),
        }
        for alias, expected_target in expected_aliases.items():
            if after["aliases"].get(alias) != expected_target:
                target.errors.append(
                    f"promotion_alias_apply.registry_after.aliases.{alias} expected {expected_target!r}, got {after['aliases'].get(alias)!r}."
                )
        previous_target = decision_ref.get("champion_previous_target")
        if before["aliases"].get("champion") != previous_target:
            target.errors.append("promotion_alias_apply.registry_before.aliases.champion must match promotion_decision champion_previous_target.")
    elif before["aliases"] != after["aliases"]:
        target.errors.append("promotion_alias_apply blocked receipts must not change registry aliases.")
    history_entry = _validate_alias_history_entry(
        receipt.get("alias_history_entry"),
        expected_passed,
        decision_ref.get("sha256"),
        before["aliases"],
        after["aliases"],
        target,
    )
    _validate_current_registry_matches_alias_receipt(
        registry_artifact,
        decision_artifact,
        decision_ref,
        after["aliases"],
        history_entry,
        expected_passed,
        target,
        source_path,
    )
    _validate_promotion_alias_apply_metrics(receipt.get("metrics"), checks, before["aliases"], after["aliases"], expected_passed, target)
    if not _is_string_list(receipt.get("notes")):
        target.errors.append("promotion_alias_apply.notes must be a list of strings.")
    target.details.update(
        {
            "passed": receipt.get("passed"),
            "recommendation": receipt.get("recommendation"),
            "failed_check_count": failed_checks,
            "candidate_id": decision_ref.get("candidate_id"),
            "rollback_id": decision_ref.get("rollback_id"),
        }
    )

def _validate_promotion_rollback_receipt(receipt: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(receipt, _PROMOTION_ROLLBACK_RECEIPT_KEYS, target, "promotion_rollback_receipt")
    _require_equal(receipt, "schema_version", PROMOTION_ROLLBACK_RECEIPT_SCHEMA_VERSION, target)
    if not isinstance(receipt.get("receipt_path"), str):
        target.errors.append("promotion_rollback_receipt.receipt_path must be a string.")
    if not isinstance(receipt.get("passed"), bool):
        target.errors.append("promotion_rollback_receipt.passed must be a boolean.")
    if not isinstance(receipt.get("available"), bool):
        target.errors.append("promotion_rollback_receipt.available must be a boolean.")
    checks = receipt.get("checks")
    if not isinstance(checks, list):
        target.errors.append("promotion_rollback_receipt.checks must be a list.")
        checks = []
    failed_checks = _validate_gate_like_checks(checks, target, "promotion_rollback_receipt.checks")
    _validate_promotion_gate_check_keys(checks, target, "promotion_rollback_receipt.checks")
    if receipt.get("check_count") != len(checks):
        target.errors.append(f"promotion_rollback_receipt.check_count expected {len(checks)}, got {receipt.get('check_count')!r}.")
    if receipt.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"promotion_rollback_receipt.failed_check_count expected {failed_checks}, got {receipt.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(receipt.get("passed"), bool) and receipt["passed"] != expected_passed:
        target.errors.append("promotion_rollback_receipt.passed must match failed_check_count.")
    if isinstance(receipt.get("available"), bool) and receipt["available"] != expected_passed:
        target.errors.append("promotion_rollback_receipt.available must match pass state.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "use_rollback_target" if expected_passed else "block_promotion"
    if receipt.get("readiness") != expected_readiness:
        target.errors.append(
            f"promotion_rollback_receipt.readiness expected {expected_readiness!r}, got {receipt.get('readiness')!r}."
        )
    if receipt.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"promotion_rollback_receipt.recommendation expected {expected_recommendation!r}, got {receipt.get('recommendation')!r}."
        )
    for field_name in ("rollback_id", "target_model_id", "champion_id"):
        if not isinstance(receipt.get(field_name), str):
            target.errors.append(f"promotion_rollback_receipt.{field_name} must be a string.")

    rollback = receipt.get("rollback")
    if not isinstance(rollback, dict):
        target.errors.append("promotion_rollback_receipt.rollback must be an object.")
        rollback = {}
    else:
        _validate_allowed_keys(rollback, _PROMOTION_ROLLBACK_BLOCK_KEYS, target, "promotion_rollback_receipt.rollback")
    for field_name in ("id", "target_model_id", "champion_id"):
        if not isinstance(rollback.get(field_name), str):
            target.errors.append(f"promotion_rollback_receipt.rollback.{field_name} must be a string.")
    if rollback.get("available") != expected_passed:
        target.errors.append("promotion_rollback_receipt.rollback.available must match pass state.")
    if rollback.get("id") != receipt.get("rollback_id"):
        target.errors.append("promotion_rollback_receipt.rollback.id must match rollback_id.")
    if receipt.get("target_model_id") != receipt.get("rollback_id") or rollback.get("target_model_id") != receipt.get("rollback_id"):
        target.errors.append("promotion_rollback_receipt target_model_id fields must match rollback_id.")
    if rollback.get("champion_id") != receipt.get("champion_id"):
        target.errors.append("promotion_rollback_receipt.rollback.champion_id must match champion_id.")

    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        target.errors.append("promotion_rollback_receipt.artifacts must be an object.")
        artifacts = {}
    else:
        _validate_allowed_keys(artifacts, _PROMOTION_ROLLBACK_ARTIFACT_KEYS, target, "promotion_rollback_receipt.artifacts")
    _validate_promotion_fingerprinted_artifact(
        artifacts.get("registry"),
        "registry",
        target,
        source_path,
        "promotion_rollback_receipt.artifacts",
    )
    registry_snapshot = _validate_alias_snapshot(
        receipt.get("registry"),
        target,
        "promotion_rollback_receipt.registry",
        require_fingerprint=expected_passed,
    )
    registry_artifact = artifacts.get("registry") if isinstance(artifacts.get("registry"), dict) else {}
    if registry_snapshot.get("sha256") != registry_artifact.get("sha256"):
        target.errors.append("promotion_rollback_receipt.registry.sha256 must match artifacts.registry.sha256.")
    _validate_ref_matches_fingerprinted_artifact(
        receipt.get("registry") if isinstance(receipt.get("registry"), dict) else {},
        registry_artifact,
        target,
        "promotion_rollback_receipt.registry",
        "artifacts.registry",
        fields=("path", "sha256", "size_bytes"),
    )
    _validate_registry_artifact_matches_alias_snapshot(
        registry_artifact,
        registry_snapshot["aliases"],
        target,
        source_path,
        "promotion_rollback_receipt",
    )
    champion_id = receipt.get("champion_id")
    rollback_id = receipt.get("rollback_id")
    if expected_passed and registry_snapshot["aliases"].get("champion") != champion_id:
        target.errors.append("promotion_rollback_receipt.registry.aliases.champion must match champion_id when passed.")
    if expected_passed and rollback_id != champion_id:
        target.errors.append("promotion_rollback_receipt.rollback_id must match champion_id when passed.")

    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("promotion_rollback_receipt.metrics must be an object.")
    else:
        _validate_allowed_keys(metrics, _PROMOTION_ROLLBACK_METRICS_KEYS, target, "promotion_rollback_receipt.metrics")
        expected_metrics = {
            "check_count": len(checks),
            "failed_check_count": failed_checks,
        }
        for field_name, expected_value in expected_metrics.items():
            if metrics.get(field_name) != expected_value:
                target.errors.append(f"promotion_rollback_receipt.metrics.{field_name} expected {expected_value!r}, got {metrics.get(field_name)!r}.")
        for field_name in ("registered_model_count", "alias_count"):
            if not _is_non_negative_int(metrics.get(field_name)):
                target.errors.append(f"promotion_rollback_receipt.metrics.{field_name} must be a non-negative integer.")
    if not _is_string_list(receipt.get("notes")):
        target.errors.append("promotion_rollback_receipt.notes must be a list of strings.")
    target.details.update(
        {
            "passed": receipt.get("passed"),
            "recommendation": receipt.get("recommendation"),
            "failed_check_count": failed_checks,
            "rollback_id": receipt.get("rollback_id"),
            "champion_id": receipt.get("champion_id"),
        }
    )

def _validate_promotion_release_record(record: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(record, _PROMOTION_RELEASE_RECORD_KEYS, target, "promotion_release_record")
    _require_equal(record, "schema_version", PROMOTION_RELEASE_RECORD_SCHEMA_VERSION, target)
    if not isinstance(record.get("release_record_path"), str):
        target.errors.append("promotion_release_record.release_record_path must be a string.")
    if not isinstance(record.get("passed"), bool):
        target.errors.append("promotion_release_record.passed must be a boolean.")
    checks = record.get("checks")
    if not isinstance(checks, list):
        target.errors.append("promotion_release_record.checks must be a list.")
        checks = []
    failed_checks = _validate_gate_like_checks(checks, target, "promotion_release_record.checks")
    _validate_promotion_gate_check_keys(checks, target, "promotion_release_record.checks")
    policy_value = record.get("policy")
    release_policy = (
        policy_value.get("release_policy")
        if isinstance(policy_value, dict)
        else None
    )
    record_artifacts = record.get("artifacts")
    rollback_record = (
        record_artifacts.get("rollback_metadata")
        if isinstance(record_artifacts, dict)
        else None
    )
    rollback_path = _promotion_existing_artifact_path(
        rollback_record,
        source_path,
    )
    rollback_payload = _promotion_read_json_payload(rollback_path)
    expected_check_ids = promotion_release_record_check_ids(
        release_policy_bound=isinstance(release_policy, dict)
        and bool(release_policy),
        rollback_receipt_bound=isinstance(rollback_payload, dict)
        and rollback_payload.get("schema_version")
        == PROMOTION_ROLLBACK_RECEIPT_SCHEMA_VERSION,
    )
    actual_check_ids = tuple(
        check.get("id") if isinstance(check, dict) else None for check in checks
    )
    if actual_check_ids != expected_check_ids:
        target.errors.append(
            "promotion_release_record.checks must exactly match the canonical ordered release check contract."
        )
    if record.get("check_count") != len(checks):
        target.errors.append(f"promotion_release_record.check_count expected {len(checks)}, got {record.get('check_count')!r}.")
    if record.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"promotion_release_record.failed_check_count expected {failed_checks}, got {record.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(record.get("passed"), bool) and record["passed"] != expected_passed:
        target.errors.append("promotion_release_record.passed must match failed_check_count.")
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "publish_release" if expected_passed else "hold_release"
    if record.get("readiness") != expected_readiness:
        target.errors.append(f"promotion_release_record.readiness expected {expected_readiness!r}, got {record.get('readiness')!r}.")
    if record.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"promotion_release_record.recommendation expected {expected_recommendation!r}, got {record.get('recommendation')!r}."
        )

    release = record.get("release")
    if not isinstance(release, dict):
        target.errors.append("promotion_release_record.release must be an object.")
        release = {}
    else:
        _validate_allowed_keys(release, _PROMOTION_RELEASE_RECORD_RELEASE_KEYS, target, "promotion_release_record.release")
    for field_name in ("id", "candidate_id", "champion_previous_target", "rollback_id", "dataset_id"):
        if not isinstance(release.get(field_name), str) or not release.get(field_name):
            target.errors.append(
                f"promotion_release_record.release.{field_name} must be a non-empty string."
            )

    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        target.errors.append("promotion_release_record.artifacts must be an object.")
        artifacts = {}
    else:
        _validate_allowed_keys(artifacts, set(PROMOTION_RELEASE_RECORD_REQUIRED_ARTIFACTS), target, "promotion_release_record.artifacts")
    for role in PROMOTION_RELEASE_RECORD_REQUIRED_ARTIFACTS:
        _validate_promotion_fingerprinted_artifact(artifacts.get(role), role, target, source_path, "promotion_release_record.artifacts")

    _validate_promotion_release_record_validation(record.get("artifact_validation"), expected_passed, target)
    _validate_promotion_release_record_bindings(record.get("bindings"), artifacts, target, source_path)
    _validate_promotion_release_record_metrics(record.get("metrics"), checks, target)
    _validate_promotion_release_record_policy(
        record.get("policy"),
        artifacts,
        target,
        source_path,
    )
    _validate_current_promotion_release_artifacts(
        artifacts,
        release,
        record.get("policy"),
        expected_passed,
        target,
        source_path,
    )
    if not _is_string_list(record.get("notes")):
        target.errors.append("promotion_release_record.notes must be a list of strings.")
    target.details.update(
        {
            "passed": record.get("passed"),
            "recommendation": record.get("recommendation"),
            "failed_check_count": failed_checks,
            "release_id": release.get("id"),
            "candidate_id": release.get("candidate_id"),
            "rollback_id": release.get("rollback_id"),
        }
    )

def _validate_promotion_release_record_validation(value: Any, expected_passed: bool, target: ValidationTarget) -> None:
    _validate_strict_compact_validation_summary(
        value,
        target,
        "promotion_release_record.artifact_validation",
        expected_parent_passed=expected_passed,
        required_target_types=set(PROMOTION_RELEASE_RECORD_VALIDATED_ARTIFACTS) if expected_passed else set(),
    )

def _validate_current_promotion_release_artifacts(
    artifacts: dict[str, Any],
    release: dict[str, Any],
    policy: Any,
    expected_passed: bool,
    target: ValidationTarget,
    source_path: Path,
) -> None:
    if not expected_passed:
        return

    artifact_paths: dict[str, Path] = {}
    for role in PROMOTION_RELEASE_RECORD_REQUIRED_ARTIFACTS:
        record = artifacts.get(role)
        artifact = record if isinstance(record, dict) else {}
        allowed_kinds = (
            {"directory", "file"} if role == "promotion_cards" else {"file"}
        )
        artifact_kind = artifact.get("kind")
        resolved = _resolve_promotion_decision_artifact_path(
            artifact.get("path"),
            source_path,
            artifact_kind,
        )
        kind_matches = (
            resolved is not None
            and resolved.exists()
            and (
                resolved.is_dir()
                if artifact_kind == "directory"
                else resolved.is_file()
            )
        )
        fingerprint_matches_kind = (
            _is_non_negative_int(artifact.get("file_count"))
            if artifact_kind == "directory"
            else _is_non_negative_int(artifact.get("size_bytes"))
        )
        if (
            artifact.get("exists") is not True
            or artifact_kind not in allowed_kinds
            or not kind_matches
            or not _is_sha256(artifact.get("sha256"))
            or not fingerprint_matches_kind
        ):
            expected_kind = (
                "file or directory" if role == "promotion_cards" else "file"
            )
            target.errors.append(
                f"promotion_release_record.artifacts.{role} must resolve to a fingerprinted existing {expected_kind} for ready releases."
            )
            continue
        artifact_paths[role] = resolved

    validators = {
        "promotion_decision": validate_promotion_decision,
        "promotion_cards": validate_promotion_cards,
        "promotion_alias_apply": validate_promotion_alias_apply,
    }
    for role, validator in validators.items():
        artifact_path = artifact_paths.get(role)
        if artifact_path is None:
            continue
        validation = validator(artifact_path)
        for error in validation.errors:
            target.errors.append(
                f"promotion_release_record current {role} is invalid: {error}"
            )
        for warning in validation.warnings:
            target.errors.append(
                f"promotion_release_record current {role} warning blocks strict replay: {warning}"
            )

    decision_path = artifact_paths.get("promotion_decision")
    decision = _promotion_read_json_payload(decision_path)
    cards_path = artifact_paths.get("promotion_cards")
    cards_manifest_path = (
        cards_path / "promotion_cards.json"
        if cards_path is not None and cards_path.is_dir()
        else cards_path
    )
    cards = _promotion_read_json_payload(cards_manifest_path)
    alias_receipt = _promotion_read_json_payload(
        artifact_paths.get("promotion_alias_apply")
    )
    rollback = _promotion_read_json_payload(
        artifact_paths.get("rollback_metadata")
    )
    compare_gate = _promotion_read_json_payload(
        artifact_paths.get("compare_gate")
    )

    decision_models = (
        decision.get("models") if isinstance(decision, dict) else None
    )
    candidate_id = _promotion_model_id(
        decision_models.get("candidate")
        if isinstance(decision_models, dict)
        else None
    )
    champion_id = _promotion_model_id(
        decision_models.get("champion")
        if isinstance(decision_models, dict)
        else None
    )
    rollback_id = _promotion_model_id(
        decision_models.get("rollback")
        if isinstance(decision_models, dict)
        else None
    )
    decision_alias_update = (
        decision.get("alias_update") if isinstance(decision, dict) else None
    )
    if (
        not isinstance(decision, dict)
        or decision.get("passed") is not True
        or decision.get("recommendation") != "apply_alias_update"
        or not isinstance(decision_alias_update, dict)
        or decision_alias_update.get("authorized") is not True
    ):
        target.errors.append(
            "promotion_release_record current promotion decision must pass and authorize alias application."
        )
    if not isinstance(cards, dict) or cards.get("passed") is not True:
        target.errors.append(
            "promotion_release_record current promotion cards must pass."
        )
    if not isinstance(alias_receipt, dict) or alias_receipt.get("passed") is not True:
        target.errors.append(
            "promotion_release_record current promotion alias receipt must pass."
        )
    if not isinstance(compare_gate, dict) or compare_gate.get("passed") is not True:
        target.errors.append(
            "promotion_release_record current compare gate must pass."
        )
    if (
        not isinstance(rollback, dict)
        or rollback.get("available") is not True
        or rollback.get("rollback_id") != rollback_id
    ):
        target.errors.append(
            "promotion_release_record current rollback metadata must make the decision rollback target available."
        )
    elif rollback.get("schema_version") == PROMOTION_ROLLBACK_RECEIPT_SCHEMA_VERSION:
        rollback_path = artifact_paths.get("rollback_metadata")
        rollback_validation = (
            validate_promotion_rollback_receipt(rollback_path)
            if rollback_path is not None
            else None
        )
        if rollback_validation is None or rollback_validation.errors or rollback_validation.warnings:
            target.errors.append(
                "promotion_release_record current rollback receipt failed strict replay."
            )

    if release.get("candidate_id") != candidate_id:
        target.errors.append(
            "promotion_release_record.release.candidate_id must match the current promotion decision."
        )
    if release.get("champion_previous_target") != champion_id:
        target.errors.append(
            "promotion_release_record.release.champion_previous_target must match the current promotion decision."
        )
    if release.get("rollback_id") != rollback_id:
        target.errors.append(
            "promotion_release_record.release.rollback_id must match the current promotion decision."
        )
    cards_dataset = cards.get("dataset") if isinstance(cards, dict) else None
    cards_candidate = cards.get("candidate") if isinstance(cards, dict) else None
    cards_candidate_id = (
        cards_candidate.get("id") if isinstance(cards_candidate, dict) else None
    )
    if cards_candidate_id != candidate_id:
        target.errors.append(
            "promotion_release_record current promotion cards candidate must match the current promotion decision."
        )
    dataset_id = (
        cards_dataset.get("id") if isinstance(cards_dataset, dict) else None
    )
    if release.get("dataset_id") != dataset_id:
        target.errors.append(
            "promotion_release_record.release.dataset_id must match the current promotion cards."
        )

    decision_artifacts = (
        decision.get("artifacts") if isinstance(decision, dict) else None
    )
    cards_artifacts = cards.get("artifacts") if isinstance(cards, dict) else None
    alias_decision = (
        alias_receipt.get("promotion_decision")
        if isinstance(alias_receipt, dict)
        else None
    )
    if (
        not isinstance(alias_decision, dict)
        or alias_decision.get("sha256")
        != artifacts.get("promotion_decision", {}).get("sha256")
        or alias_decision.get("candidate_id") != candidate_id
        or alias_decision.get("champion_previous_target") != champion_id
        or alias_decision.get("rollback_id") != rollback_id
    ):
        target.errors.append(
            "promotion_release_record current alias receipt must bind the current decision and model targets."
        )
    if (
        not isinstance(decision_artifacts, dict)
        or not isinstance(cards_artifacts, dict)
        or _promotion_artifact_sha(decision_artifacts.get("model_card"))
        != _promotion_artifact_sha(cards_artifacts.get("model_card"))
        or _promotion_artifact_sha(decision_artifacts.get("dataset_card"))
        != _promotion_artifact_sha(cards_artifacts.get("dataset_card"))
    ):
        target.errors.append(
            "promotion_release_record current cards must match the cards bound by the decision."
        )
    if (
        not isinstance(decision_artifacts, dict)
        or _promotion_artifact_sha(decision_artifacts.get("compare_gate"))
        != artifacts.get("compare_gate", {}).get("sha256")
    ):
        target.errors.append(
            "promotion_release_record current compare gate must match the decision compare gate."
        )

    record_policy = policy if isinstance(policy, dict) else {}
    decision_policy = decision.get("policy") if isinstance(decision, dict) else None
    if record_policy.get("promotion_decision_policy") != decision_policy:
        target.errors.append(
            "promotion_release_record policy must exactly match the current promotion decision policy."
        )
    release_policy = record_policy.get("release_policy")
    if (
        release_policy not in (None, {})
        and _promotion_policy_without_artifact_path(release_policy)
        != _promotion_policy_without_artifact_path(decision_policy)
    ):
        target.errors.append(
            "promotion_release_record release policy must exactly match the current promotion decision policy."
        )

    release_notes_path = artifact_paths.get("release_notes")
    try:
        release_notes = (
            release_notes_path.read_text(encoding="utf-8")
            if release_notes_path is not None
            else ""
        )
    except (OSError, UnicodeDecodeError):
        release_notes = ""
    unsupported_markers = [
        marker
        for marker in ("unsupported claim", "todo", "tbd")
        if marker in release_notes.lower()
    ]
    if not release_notes.strip() or unsupported_markers:
        target.errors.append(
            "promotion_release_record current release notes must be readable, non-empty, and contain no unsupported markers."
        )

def _promotion_artifact_sha(value: Any) -> Any:
    return value.get("sha256") if isinstance(value, dict) else None

def _promotion_policy_without_artifact_path(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    artifact = value.get("artifact")
    if isinstance(artifact, dict):
        normalized["artifact"] = {
            key: item for key, item in artifact.items() if key != "path"
        }
    return normalized

def _validate_strict_compact_validation_summary(
    value: Any,
    target: ValidationTarget,
    label: str,
    *,
    expected_parent_passed: bool,
    required_target_types: set[str],
) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _COMPACT_VALIDATION_SUMMARY_KEYS, target, label)
    passed = value.get("passed")
    if passed is not None and not isinstance(passed, bool):
        target.errors.append(f"{label}.passed must be a boolean or null.")
    counts_valid = True
    for field_name in ("target_count", "error_count", "warning_count"):
        if not _is_non_negative_int(value.get(field_name)):
            counts_valid = False
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")

    targets = value.get("targets")
    if not isinstance(targets, list):
        target.errors.append(f"{label}.targets must be a list.")
        targets = []
    if _is_non_negative_int(value.get("target_count")) and value["target_count"] != len(targets):
        target.errors.append(f"{label}.target_count expected {len(targets)}, got {value.get('target_count')!r}.")
    if required_target_types and _is_non_negative_int(value.get("target_count")) and value["target_count"] < len(required_target_types):
        target.errors.append(f"{label}.target_count must be at least {len(required_target_types)} for validated artifacts.")

    target_types: set[str] = set()
    target_counts_valid = True
    aggregate_error_count = 0
    aggregate_warning_count = 0
    for index, item in enumerate(targets):
        item_label = f"{label}.targets[{index}]"
        if not isinstance(item, dict):
            target_counts_valid = False
            target.errors.append(f"{item_label} must be an object.")
            continue
        _validate_allowed_keys(item, _COMPACT_VALIDATION_TARGET_KEYS, target, item_label)
        item_type = item.get("type")
        if not isinstance(item_type, str) or not item_type:
            target.errors.append(f"{item_label}.type must be a non-empty string.")
        else:
            target_types.add(item_type)
        item_passed = item.get("passed")
        if not isinstance(item_passed, bool):
            target_counts_valid = False
            target.errors.append(f"{item_label}.passed must be a boolean.")
        item_counts_valid = True
        for field_name in ("error_count", "warning_count"):
            if not _is_non_negative_int(item.get(field_name)):
                item_counts_valid = False
                target_counts_valid = False
                target.errors.append(f"{item_label}.{field_name} must be a non-negative integer.")
        if item_counts_valid:
            aggregate_error_count += item["error_count"]
            aggregate_warning_count += item["warning_count"]
            expected_item_passed = item["error_count"] == 0 and item["warning_count"] == 0
            if isinstance(item_passed, bool) and item_passed != expected_item_passed:
                target.errors.append(f"{item_label}.passed expected {expected_item_passed}, got {item_passed!r}.")

    if counts_valid and target_counts_valid:
        if value["error_count"] != aggregate_error_count:
            target.errors.append(f"{label}.error_count expected {aggregate_error_count}, got {value.get('error_count')!r}.")
        if value["warning_count"] != aggregate_warning_count:
            target.errors.append(f"{label}.warning_count expected {aggregate_warning_count}, got {value.get('warning_count')!r}.")
        expected_summary_passed = value["target_count"] > 0 and value["error_count"] == 0 and value["warning_count"] == 0
        if isinstance(passed, bool) and passed != expected_summary_passed:
            target.errors.append(f"{label}.passed expected {expected_summary_passed}, got {passed!r}.")
    if expected_parent_passed and passed is not True:
        target.errors.append(f"{label}.passed must be true when the parent artifact passed.")
    if required_target_types:
        missing_types = sorted(required_target_types - target_types)
        if missing_types:
            target.errors.append(f"{label}.targets missing required type(s): {', '.join(missing_types)}.")

def _validate_promotion_release_record_bindings(
    value: Any,
    artifacts: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("promotion_release_record.bindings must be an object.")
        return
    _validate_allowed_keys(value, _PROMOTION_RELEASE_RECORD_BINDINGS_KEYS, target, "promotion_release_record.bindings")
    expected_artifact_fields = {
        "promotion_decision_sha256": "promotion_decision",
        "promotion_cards_sha256": "promotion_cards",
        "promotion_alias_apply_sha256": "promotion_alias_apply",
        "rollback_metadata_sha256": "rollback_metadata",
        "compare_gate_sha256": "compare_gate",
        "release_notes_sha256": "release_notes",
    }
    for field_name, artifact_role in expected_artifact_fields.items():
        actual = value.get(field_name)
        if not _is_sha256(actual):
            target.errors.append(f"promotion_release_record.bindings.{field_name} must be a SHA-256 hex string.")
            continue
        artifact = artifacts.get(artifact_role) if isinstance(artifacts.get(artifact_role), dict) else {}
        if actual != artifact.get("sha256"):
            target.errors.append(f"promotion_release_record.bindings.{field_name} must match artifacts.{artifact_role}.sha256.")
    for field_name in ("model_card_sha256", "dataset_card_sha256"):
        if value.get(field_name) and not _is_sha256(value.get(field_name)):
            target.errors.append(f"promotion_release_record.bindings.{field_name} must be a SHA-256 hex string when present.")
    _validate_promotion_release_record_card_bindings(value, artifacts, target, source_path)

def _validate_promotion_release_record_card_bindings(
    bindings: dict[str, Any],
    artifacts: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    promotion_cards = artifacts.get("promotion_cards") if isinstance(artifacts.get("promotion_cards"), dict) else {}
    kind = promotion_cards.get("kind")
    cards_path = _resolve_promotion_decision_artifact_path(promotion_cards.get("path"), source_path, kind)
    if cards_path is None or not cards_path.exists():
        return
    manifest_path = cards_path / "promotion_cards.json" if cards_path.is_dir() else cards_path
    if not manifest_path.is_file():
        return
    try:
        cards = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        target.errors.append(f"promotion_release_record.artifacts.promotion_cards contains invalid JSON: {exc}")
        return
    if not isinstance(cards, dict):
        target.errors.append("promotion_release_record.artifacts.promotion_cards must contain a JSON object.")
        return
    cards_artifacts = cards.get("artifacts") if isinstance(cards.get("artifacts"), dict) else {}
    expected_card_fields = {
        "model_card_sha256": "model_card",
        "dataset_card_sha256": "dataset_card",
    }
    for field_name, card_role in expected_card_fields.items():
        actual = bindings.get(field_name)
        card_artifact = cards_artifacts.get(card_role) if isinstance(cards_artifacts.get(card_role), dict) else {}
        expected = card_artifact.get("sha256")
        if expected is None and not actual:
            continue
        if not _is_sha256(expected):
            target.errors.append(f"promotion_cards.artifacts.{card_role}.sha256 must be a SHA-256 hex string.")
            continue
        if actual != expected:
            target.errors.append(
                f"promotion_release_record.bindings.{field_name} must match promotion_cards.artifacts.{card_role}.sha256."
            )

def _validate_promotion_release_record_metrics(value: Any, checks: list[Any], target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("promotion_release_record.metrics must be an object.")
        return
    _validate_allowed_keys(value, _PROMOTION_RELEASE_RECORD_METRICS_KEYS, target, "promotion_release_record.metrics")
    expected_failed = sum(1 for check in checks if isinstance(check, dict) and check.get("passed") is False)
    expected = {
        "check_count": len(checks),
        "failed_check_count": expected_failed,
        "required_artifact_count": len(PROMOTION_RELEASE_RECORD_REQUIRED_ARTIFACTS),
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"promotion_release_record.metrics.{field_name} expected {expected_value!r}, got {value.get(field_name)!r}.")
    if not _is_non_negative_int(value.get("release_notes_size_bytes")):
        target.errors.append("promotion_release_record.metrics.release_notes_size_bytes must be a non-negative integer.")

def _validate_promotion_release_record_policy(
    value: Any,
    artifacts: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        target.errors.append("promotion_release_record.policy must be an object.")
        return
    _validate_allowed_keys(value, _PROMOTION_RELEASE_RECORD_POLICY_KEYS, target, "promotion_release_record.policy")
    decision_policy = value.get("promotion_decision_policy")
    if isinstance(decision_policy, dict) and decision_policy:
        decision_artifact = artifacts.get("promotion_decision")
        decision_path = _promotion_existing_artifact_path(
            decision_artifact,
            source_path,
        )
        _validate_promotion_policy_section(
            decision_policy,
            target,
            decision_path or source_path,
            "promotion_release_record.policy.promotion_decision_policy",
        )
    elif decision_policy not in ({}, None):
        target.errors.append("promotion_release_record.policy.promotion_decision_policy must be an object.")
    release_policy = value.get("release_policy")
    if release_policy is not None:
        _validate_promotion_policy_section(release_policy, target, source_path, "promotion_release_record.policy.release_policy")

def _validate_alias_snapshot(
    value: Any,
    target: ValidationTarget,
    label: str,
    *,
    require_fingerprint: bool = False,
    allow_fingerprint: bool = True,
) -> dict[str, Any]:
    snapshot = {"aliases": {}, "sha256": None, "size_bytes": None}
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return snapshot
    allowed_keys = {"aliases", "path", "sha256", "size_bytes"} if allow_fingerprint else {"aliases"}
    _validate_allowed_keys(value, allowed_keys, target, label)
    if not allow_fingerprint:
        for field_name in ("path", "sha256", "size_bytes"):
            if field_name in value:
                target.errors.append(f"{label}.{field_name} must be absent for snapshot-only refs.")
    else:
        if not isinstance(value.get("path"), str):
            target.errors.append(f"{label}.path must be a string.")
        if require_fingerprint and not _is_sha256(value.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string.")
        elif value.get("sha256") is not None and not _is_sha256(value.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string or null.")
        if "size_bytes" in value and not _is_non_negative_int(value.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
        elif "size_bytes" not in value and require_fingerprint:
            target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    aliases = value.get("aliases")
    if not isinstance(aliases, dict) or not all(isinstance(alias, str) and isinstance(target_id, str) for alias, target_id in aliases.items()):
        target.errors.append(f"{label}.aliases must be an object of string values.")
        aliases = {}
    snapshot["aliases"] = dict(aliases)
    snapshot["sha256"] = value.get("sha256")
    snapshot["size_bytes"] = value.get("size_bytes")
    return snapshot

def _validate_ref_matches_fingerprinted_artifact(
    ref: dict[str, Any],
    artifact: dict[str, Any],
    target: ValidationTarget,
    label: str,
    artifact_label: str,
    *,
    fields: tuple[str, ...],
) -> None:
    for field_name in fields:
        if ref.get(field_name) != artifact.get(field_name):
            target.errors.append(f"{label}.{field_name} must match {artifact_label}.{field_name}.")

def _validate_alias_history_entry(
    value: Any,
    expected_passed: bool,
    expected_decision_sha: Any,
    aliases_before: dict[str, str],
    aliases_after: dict[str, str],
    target: ValidationTarget,
) -> dict[str, Any] | None:
    if not expected_passed:
        if value is not None:
            target.errors.append("promotion_alias_apply.alias_history_entry must be null for blocked receipts.")
        return None
    if not isinstance(value, dict):
        target.errors.append("promotion_alias_apply.alias_history_entry must be an object for applied receipts.")
        return None
    _validate_allowed_keys(value, _PROMOTION_ALIAS_HISTORY_ENTRY_KEYS, target, "promotion_alias_apply.alias_history_entry")
    if value.get("promotion_decision_sha256") != expected_decision_sha:
        target.errors.append("promotion_alias_apply.alias_history_entry.promotion_decision_sha256 must match promotion_decision.sha256.")
    previous_aliases = value.get("previous_aliases")
    if previous_aliases != aliases_before:
        target.errors.append("promotion_alias_apply.alias_history_entry.previous_aliases must match registry_before.aliases.")
    updated_aliases = value.get("updated_aliases")
    if not isinstance(updated_aliases, dict) or not all(
        isinstance(alias, str) and isinstance(target_id, str) for alias, target_id in updated_aliases.items()
    ):
        target.errors.append("promotion_alias_apply.alias_history_entry.updated_aliases must be an object of string values.")
    else:
        for alias in ("candidate", "champion", "rollback"):
            if updated_aliases.get(alias) != aliases_after.get(alias):
                target.errors.append(
                    f"promotion_alias_apply.alias_history_entry.updated_aliases.{alias} expected {aliases_after.get(alias)!r}, got {updated_aliases.get(alias)!r}."
                )
    return value

def _validate_promotion_alias_apply_decision_validation(value: Any, expected_passed: bool, target: ValidationTarget) -> None:
    _validate_strict_compact_validation_summary(
        value,
        target,
        "promotion_alias_apply.promotion_decision_validation",
        expected_parent_passed=expected_passed,
        required_target_types={"promotion_decision"} if expected_passed else set(),
    )

def _validate_current_registry_matches_alias_receipt(
    registry_artifact: dict[str, Any],
    decision_artifact: dict[str, Any],
    decision_ref: dict[str, Any],
    expected_aliases: dict[str, str],
    expected_history_entry: dict[str, Any] | None,
    expected_passed: bool,
    target: ValidationTarget,
    source_path: Path,
) -> None:
    registry_path = _resolve_promotion_decision_artifact_path(registry_artifact.get("path"), source_path, "file")
    if registry_path is None or not registry_path.exists() or not registry_path.is_file():
        if expected_passed:
            target.errors.append(
                "promotion_alias_apply current registry must resolve to an existing file for applied receipts."
            )
        return
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        target.errors.append(f"promotion_alias_apply.artifacts.registry contains invalid JSON: {exc}")
        return
    if not isinstance(registry, dict):
        target.errors.append("promotion_alias_apply.artifacts.registry must contain a JSON object.")
        return
    if registry.get("schema_version") != MODEL_REGISTRY_SCHEMA_VERSION:
        target.errors.append(
            f"promotion_alias_apply.artifacts.registry.schema_version expected {MODEL_REGISTRY_SCHEMA_VERSION!r}, got {registry.get('schema_version')!r}."
        )
    if expected_passed:
        registry_validation = ValidationTarget("model_registry", str(registry_path))
        _validate_model_registry(registry, registry_validation, registry_path)
        for error in registry_validation.errors:
            target.errors.append(
                f"promotion_alias_apply current registry is invalid: {error}"
            )
        for warning in registry_validation.warnings:
            target.errors.append(
                f"promotion_alias_apply current registry warning blocks strict replay: {warning}"
            )
        _validate_alias_receipt_candidate_entry_binding(
            registry,
            decision_artifact,
            decision_ref,
            target,
            source_path,
        )
    aliases = registry.get("aliases")
    if not isinstance(aliases, dict):
        target.errors.append("promotion_alias_apply.artifacts.registry.aliases must be an object.")
        return
    current_aliases = {str(alias): str(target_id) for alias, target_id in aliases.items() if isinstance(alias, str) and isinstance(target_id, str)}
    for alias, expected_target in expected_aliases.items():
        if current_aliases.get(alias) != expected_target:
            target.errors.append(
                f"promotion_alias_apply current registry alias {alias!r} expected {expected_target!r}, got {current_aliases.get(alias)!r}."
            )
    if expected_history_entry is not None:
        history = registry.get("alias_history")
        if not isinstance(history, list):
            target.errors.append("promotion_alias_apply.artifacts.registry.alias_history must be a list for applied receipts.")
        elif not any(isinstance(entry, dict) and entry == expected_history_entry for entry in history):
            target.errors.append("promotion_alias_apply.artifacts.registry.alias_history must contain alias_history_entry.")

def _validate_alias_receipt_candidate_entry_binding(
    registry: dict[str, Any],
    decision_artifact: dict[str, Any],
    decision_ref: dict[str, Any],
    target: ValidationTarget,
    receipt_path: Path,
) -> None:
    decision_path = _promotion_existing_artifact_path(
        decision_artifact,
        receipt_path,
    )
    decision = _promotion_read_json_payload(decision_path)
    if decision_path is None or not isinstance(decision, dict):
        target.errors.append(
            "promotion_alias_apply current promotion decision could not be replayed."
        )
        return
    decision_validation = ValidationTarget("promotion_decision", str(decision_path))
    _validate_promotion_decision(decision, decision_validation, decision_path)
    for error in decision_validation.errors:
        target.errors.append(
            f"promotion_alias_apply current promotion decision is invalid: {error}"
        )
    for warning in decision_validation.warnings:
        target.errors.append(
            "promotion_alias_apply current promotion decision warning blocks "
            f"strict replay: {warning}"
        )
    decision_block = decision.get("decision")
    alias_update = decision.get("alias_update")
    if (
        decision.get("passed") is not True
        or decision.get("readiness") != "ready"
        or decision.get("recommendation") != "apply_alias_update"
        or not isinstance(decision_block, dict)
        or decision_block.get("recommendation") != "apply_alias_update"
        or not isinstance(alias_update, dict)
        or alias_update.get("authorized") is not True
        or alias_update.get("recommendation") != "apply_alias_update"
    ):
        target.errors.append(
            "promotion_alias_apply current promotion decision must be passing, ready, and authorize alias application."
        )
    models = decision.get("models")
    candidate_id = decision_ref.get("candidate_id")
    decision_candidate_id = _promotion_model_id(
        models.get("candidate") if isinstance(models, dict) else None
    )
    decision_champion_id = _promotion_model_id(
        models.get("champion") if isinstance(models, dict) else None
    )
    decision_rollback_id = _promotion_model_id(
        models.get("rollback") if isinstance(models, dict) else None
    )
    expected_ref_fields = {
        "candidate_id": decision_candidate_id,
        "champion_previous_target": decision_champion_id,
        "rollback_id": decision_rollback_id,
    }
    for field_name, expected_value in expected_ref_fields.items():
        if decision_ref.get(field_name) != expected_value:
            target.errors.append(
                "promotion_alias_apply promotion_decision."
                f"{field_name} must match the current decision model."
            )

    aliases = (
        alias_update.get("aliases")
        if isinstance(alias_update, dict)
        else None
    )
    expected_aliases = [
        {"alias": "candidate", "target": decision_candidate_id},
        {
            "alias": "champion",
            "previous_target": decision_champion_id,
            "target": decision_candidate_id,
        },
        {"alias": "rollback", "target": decision_rollback_id},
    ]
    if aliases != expected_aliases:
        target.errors.append(
            "promotion_alias_apply current promotion decision alias_update must "
            "match its canonical model targets."
        )
    if not isinstance(candidate_id, str) or decision_candidate_id != candidate_id:
        return
    artifacts = decision.get("artifacts")
    entry_record = (
        artifacts.get("model_registry_entry")
        if isinstance(artifacts, dict)
        else None
    )
    entry_path = _promotion_existing_artifact_path(entry_record, decision_path)
    entry = _promotion_read_json_payload(entry_path)
    entries = registry.get("entries")
    registered_entry = (
        entries.get(candidate_id) if isinstance(entries, dict) else None
    )
    if not isinstance(entry_record, dict) or entry_path is None or not isinstance(entry, dict):
        target.errors.append(
            "promotion_alias_apply decision model_registry_entry could not be replayed."
        )
        return
    if (
        entry_record.get("sha256") != _sha256(entry_path)
        or entry_record.get("size_bytes") != entry_path.stat().st_size
    ):
        target.errors.append(
            "promotion_alias_apply decision model_registry_entry fingerprint is stale."
        )
    if registered_entry != entry:
        target.errors.append(
            "promotion_alias_apply current registry candidate entry must exactly match the decision artifact."
        )

def _validate_registry_artifact_matches_alias_snapshot(
    registry_artifact: dict[str, Any],
    expected_aliases: dict[str, str],
    target: ValidationTarget,
    source_path: Path,
    label: str,
) -> None:
    registry_path = _resolve_promotion_decision_artifact_path(registry_artifact.get("path"), source_path, "file")
    if registry_path is None or not registry_path.exists() or not registry_path.is_file():
        return
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        target.errors.append(f"{label}.artifacts.registry contains invalid JSON: {exc}")
        return
    if not isinstance(registry, dict):
        target.errors.append(f"{label}.artifacts.registry must contain a JSON object.")
        return
    aliases = registry.get("aliases")
    if not isinstance(aliases, dict):
        target.errors.append(f"{label}.artifacts.registry.aliases must be an object.")
        return
    current_aliases = {alias: target_id for alias, target_id in aliases.items() if isinstance(alias, str) and isinstance(target_id, str)}
    for alias, expected_target in expected_aliases.items():
        if current_aliases.get(alias) != expected_target:
            target.errors.append(
                f"{label}.registry.aliases.{alias} must match artifacts.registry.aliases.{alias}: "
                f"expected {current_aliases.get(alias)!r}, got {expected_target!r}."
            )

def _validate_promotion_alias_apply_metrics(
    value: Any,
    checks: list[Any],
    aliases_before: dict[str, str],
    aliases_after: dict[str, str],
    expected_passed: bool,
    target: ValidationTarget,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("promotion_alias_apply.metrics must be an object.")
        return
    _validate_allowed_keys(value, _PROMOTION_ALIAS_APPLY_METRICS_KEYS, target, "promotion_alias_apply.metrics")
    expected_failed = sum(1 for check in checks if isinstance(check, dict) and check.get("passed") is False)
    expected = {
        "check_count": len(checks),
        "failed_check_count": expected_failed,
        "alias_count_before": len(aliases_before),
        "alias_count_after": len(aliases_after),
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(f"promotion_alias_apply.metrics.{field_name} expected {expected_value!r}, got {value.get(field_name)!r}.")
    if not _is_non_negative_int(value.get("registered_model_count")):
        target.errors.append("promotion_alias_apply.metrics.registered_model_count must be a non-negative integer.")
    for field_name in ("alias_history_count_before", "alias_history_count_after"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"promotion_alias_apply.metrics.{field_name} must be a non-negative integer.")
    if _is_non_negative_int(value.get("alias_history_count_before")) and _is_non_negative_int(value.get("alias_history_count_after")):
        expected_after = value["alias_history_count_before"] + (1 if expected_passed else 0)
        if value["alias_history_count_after"] != expected_after:
            target.errors.append(
                f"promotion_alias_apply.metrics.alias_history_count_after expected {expected_after!r}, got {value['alias_history_count_after']!r}."
            )

def _validate_promotion_decision(decision: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _validate_allowed_keys(decision, _PROMOTION_DECISION_KEYS, target, "promotion_decision")
    _require_equal(decision, "schema_version", PROMOTION_DECISION_SCHEMA_VERSION, target)
    if not isinstance(decision.get("decision_path"), str):
        target.errors.append("promotion_decision.decision_path must be a string.")
    if not isinstance(decision.get("passed"), bool):
        target.errors.append("promotion_decision.passed must be a boolean.")
    checks = decision.get("checks")
    if not isinstance(checks, list):
        target.errors.append("promotion_decision.checks must be a list.")
        checks = []
    failed_checks = _validate_gate_like_checks(checks, target, "promotion_decision.checks")
    _validate_promotion_gate_check_keys(checks, target, "promotion_decision.checks")
    if decision.get("check_count") != len(checks):
        target.errors.append(f"promotion_decision.check_count expected {len(checks)}, got {decision.get('check_count')!r}.")
    if decision.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"promotion_decision.failed_check_count expected {failed_checks}, got {decision.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(decision.get("passed"), bool) and decision["passed"] != expected_passed:
        target.errors.append("promotion_decision.passed must match failed_check_count.")
    if expected_passed:
        _validate_promotion_decision_required_pass_checks(checks, target)
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "apply_alias_update" if expected_passed else "block_promotion"
    if decision.get("readiness") != expected_readiness:
        target.errors.append(f"promotion_decision.readiness expected {expected_readiness!r}, got {decision.get('readiness')!r}.")
    if decision.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"promotion_decision.recommendation expected {expected_recommendation!r}, got {decision.get('recommendation')!r}."
        )

    models = decision.get("models")
    if not isinstance(models, dict):
        target.errors.append("promotion_decision.models must be an object.")
        models = {}
    else:
        _validate_allowed_keys(models, _PROMOTION_DECISION_MODELS_KEYS, target, "promotion_decision.models")
    for role in ("candidate", "champion", "rollback"):
        _validate_promotion_decision_model(models.get(role), target, f"promotion_decision.models.{role}")
    candidate_id = _promotion_model_id(models.get("candidate"))
    champion_id = _promotion_model_id(models.get("champion"))
    rollback_id = _promotion_model_id(models.get("rollback"))
    if candidate_id and champion_id and candidate_id == champion_id:
        target.errors.append("promotion_decision.models.candidate.id must differ from models.champion.id.")
    if expected_passed and not rollback_id:
        target.errors.append("promotion_decision.models.rollback.id must be present when promotion passed.")

    decision_block = decision.get("decision")
    _validate_promotion_decision_block(decision_block, expected_readiness, expected_recommendation, failed_checks, target)
    artifacts = decision.get("artifacts")
    if not isinstance(artifacts, dict):
        target.errors.append("promotion_decision.artifacts must be an object.")
        artifacts = {}
    else:
        _validate_allowed_keys(artifacts, set(PROMOTION_DECISION_REQUIRED_ARTIFACTS), target, "promotion_decision.artifacts")
    for role in PROMOTION_DECISION_REQUIRED_ARTIFACTS:
        _validate_promotion_decision_artifact(artifacts.get(role), role, target, source_path)
    _validate_promotion_external_eval_lineage(
        decision.get("external_eval_lineage"),
        artifacts,
        checks,
        candidate_id,
        target,
        source_path,
    )
    policy = decision.get("policy")
    _validate_promotion_policy_section(policy, target, source_path, "promotion_decision.policy")
    _validate_promotion_decision_check_replay(
        decision,
        artifacts,
        target,
        source_path,
    )
    metrics = decision.get("metrics")
    _validate_promotion_decision_metrics(
        metrics,
        checks,
        target,
        policy,
        decision.get("external_eval_lineage"),
        artifacts,
        source_path,
    )
    if (
        isinstance(decision_block, dict)
        and isinstance(decision_block.get("key_metrics"), dict)
        and decision_block.get("key_metrics") != metrics
    ):
        target.errors.append(
            "promotion_decision.decision.key_metrics must exactly match promotion_decision.metrics."
        )
    _validate_promotion_decision_alias_update(
        decision.get("alias_update"),
        expected_passed,
        candidate_id,
        champion_id,
        rollback_id,
        target,
    )
    notes = decision.get("notes")
    if not _is_string_list(notes):
        target.errors.append("promotion_decision.notes must be a list of strings.")
    target.details.update(
        {
            "passed": decision.get("passed"),
            "recommendation": decision.get("recommendation"),
            "failed_check_count": failed_checks,
            "candidate_id": candidate_id,
            "champion_id": champion_id,
            "rollback_id": rollback_id,
        }
    )

def _validate_promotion_decision_model(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _PROMOTION_DECISION_MODEL_KEYS, target, label)
    if not isinstance(value.get("id"), str):
        target.errors.append(f"{label}.id must be a string.")
    if not isinstance(value.get("class"), str) or not value.get("class"):
        target.errors.append(f"{label}.class must be a non-empty string.")

def _validate_promotion_decision_required_pass_checks(checks: list[Any], target: ValidationTarget) -> None:
    by_id = {check.get("id"): check for check in checks if isinstance(check, dict) and isinstance(check.get("id"), str)}
    missing = [check_id for check_id in PROMOTION_DECISION_REQUIRED_PASS_CHECK_IDS if check_id not in by_id]
    if missing:
        target.errors.append(f"promotion_decision.checks missing required passing check(s): {missing!r}.")
    failed_required = [
        check_id
        for check_id in PROMOTION_DECISION_REQUIRED_PASS_CHECK_IDS
        if isinstance(by_id.get(check_id), dict) and by_id[check_id].get("passed") is not True
    ]
    if failed_required:
        target.errors.append(f"promotion_decision.checks required pass check(s) are not passed: {failed_required!r}.")

def _promotion_model_id(value: Any) -> str:
    if isinstance(value, dict) and isinstance(value.get("id"), str):
        return value["id"]
    return ""

def _validate_promotion_decision_block(
    value: Any,
    expected_readiness: str,
    expected_recommendation: str,
    failed_checks: int,
    target: ValidationTarget,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("promotion_decision.decision must be an object.")
        return
    _validate_allowed_keys(value, _PROMOTION_DECISION_BLOCK_KEYS, target, "promotion_decision.decision")
    if value.get("readiness") != expected_readiness:
        target.errors.append(f"promotion_decision.decision.readiness expected {expected_readiness!r}, got {value.get('readiness')!r}.")
    if value.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"promotion_decision.decision.recommendation expected {expected_recommendation!r}, got {value.get('recommendation')!r}."
        )
    if not isinstance(value.get("summary"), str) or not value.get("summary"):
        target.errors.append("promotion_decision.decision.summary must be a non-empty string.")
    if value.get("blocking_check_count") != failed_checks:
        target.errors.append(
            f"promotion_decision.decision.blocking_check_count expected {failed_checks}, got {value.get('blocking_check_count')!r}."
        )
    blocking_checks = value.get("blocking_checks")
    if not isinstance(blocking_checks, list):
        target.errors.append("promotion_decision.decision.blocking_checks must be a list.")
        blocking_checks = []
    if len(blocking_checks) != failed_checks:
        target.errors.append(
            f"promotion_decision.decision.blocking_checks expected {failed_checks} entries, got {len(blocking_checks)}."
        )
    for index, check in enumerate(blocking_checks):
        label = f"promotion_decision.decision.blocking_checks[{index}]"
        if not isinstance(check, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        _validate_allowed_keys(check, _PROMOTION_DECISION_BLOCKING_CHECK_KEYS, target, label)
        for field_name in ("id", "summary"):
            if not isinstance(check.get(field_name), str) or not check.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if not isinstance(check.get("scope"), dict):
            target.errors.append(f"{label}.scope must be an object.")
    key_metrics = value.get("key_metrics")
    if not isinstance(key_metrics, dict):
        target.errors.append("promotion_decision.decision.key_metrics must be an object.")
    else:
        _validate_allowed_keys(key_metrics, _PROMOTION_DECISION_METRICS_KEYS, target, "promotion_decision.decision.key_metrics")

def _validate_promotion_decision_artifact(value: Any, role: str, target: ValidationTarget, source_path: Path) -> None:
    _validate_promotion_fingerprinted_artifact(value, role, target, source_path, "promotion_decision.artifacts")
    if (
        not isinstance(value, dict)
        or value.get("exists") is not True
        or value.get("kind") != "file"
    ):
        return
    expected_schema = _PROMOTION_JSON_ARTIFACT_ROLES.get(role)
    if expected_schema is None:
        return
    if value.get("schema_version") != expected_schema:
        target.errors.append(
            f"promotion_decision.artifacts.{role}.schema_version must be {expected_schema}."
        )
    artifact_path = _promotion_existing_artifact_path(value, source_path)
    payload = _promotion_read_json_payload(artifact_path)
    if not isinstance(payload, dict) or payload.get("schema_version") != expected_schema:
        target.errors.append(
            f"promotion_decision.artifacts.{role} source must use schema {expected_schema}."
        )

def _validate_promotion_external_eval_lineage(
    value: Any,
    artifacts: dict[str, Any],
    checks: list[Any],
    candidate_id: str,
    target: ValidationTarget,
    source_path: Path,
) -> None:
    label = "promotion_decision.external_eval_lineage"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        lineage: dict[str, Any] = {}
    else:
        lineage = value
        _validate_allowed_keys(lineage, _PROMOTION_EXTERNAL_EVAL_LINEAGE_KEYS, target, label)
    for field_name in ("result_count", "summary_result_count"):
        if not _is_non_negative_int(lineage.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    for field_name in (
        "exact_result_set",
        "candidate_model_bound",
        "evidence_bundle_summary_bound",
        "evidence_bundle_semantically_valid",
        "eval_summary_semantically_valid",
        "external_results_semantically_valid",
        "semantic_validation_passed",
        "governance_ready",
        "passed",
    ):
        if not isinstance(lineage.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")

    raw_results = lineage.get("results")
    if not isinstance(raw_results, list):
        target.errors.append(f"{label}.results must be a list.")
        raw_results = []
    result_artifacts: list[dict[str, Any]] = []
    result_paths: list[Path | None] = []
    for index, row in enumerate(raw_results):
        row_label = f"{label}.results[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{row_label} must be an object.")
            continue
        _validate_allowed_keys(row, _PROMOTION_EXTERNAL_EVAL_RESULT_KEYS, target, row_label)
        artifact = row.get("artifact")
        if not isinstance(artifact, dict):
            target.errors.append(f"{row_label}.artifact must be an object.")
            continue
        _validate_promotion_external_eval_result_artifact(
            artifact,
            target,
            source_path,
            f"{row_label}.artifact",
        )
        _validate_promotion_external_eval_result_projection(row, target, row_label)
        result_artifacts.append(artifact)
        result_paths.append(_promotion_existing_artifact_path(artifact, source_path))

    evidence_record = artifacts.get("evidence_bundle")
    summary_record = artifacts.get("eval_summary")
    evidence_path = _promotion_existing_artifact_path(evidence_record, source_path)
    summary_path = _promotion_existing_artifact_path(summary_record, source_path)
    evidence_payload = _promotion_read_json_payload(evidence_path)
    summary_payload = _promotion_read_json_payload(summary_path)
    result_payloads = [_promotion_read_json_payload(path) for path in result_paths]

    evidence_validation = validate_evidence_bundle(evidence_path) if evidence_path is not None else None
    summary_validation = validate_eval_summary(summary_path) if summary_path is not None else None
    result_validations = [
        validate_external_eval_result(path) if path is not None else None for path in result_paths
    ]
    evidence_semantically_valid = (
        evidence_validation is not None
        and not evidence_validation.errors
        and not evidence_validation.warnings
    )
    summary_semantically_valid = (
        summary_validation is not None
        and not summary_validation.errors
        and not summary_validation.warnings
    )
    results_semantically_valid = (
        bool(result_paths)
        and len(result_paths) == len(raw_results)
        and all(
            validation is not None
            and not validation.errors
            and not validation.warnings
            for validation in result_validations
        )
    )
    if evidence_path is not None and not evidence_semantically_valid:
        target.errors.append(
            "promotion_decision.artifacts.evidence_bundle failed semantic validation at decision replay."
        )
    if summary_path is not None and not summary_semantically_valid:
        target.errors.append(
            "promotion_decision.artifacts.eval_summary failed semantic validation at decision replay."
        )
    for index, validation in enumerate(result_validations):
        if validation is not None and (validation.errors or validation.warnings):
            target.errors.append(
                f"{label}.results[{index}].artifact failed semantic validation at decision replay."
            )

    expected = _build_promotion_external_eval_lineage(
        candidate_id=candidate_id,
        evidence_bundle=evidence_payload,
        eval_summary=summary_payload,
        eval_summary_artifact=summary_record if isinstance(summary_record, dict) else {},
        result_payloads=result_payloads,
        result_artifacts=result_artifacts,
        evidence_bundle_semantically_valid=evidence_semantically_valid,
        eval_summary_semantically_valid=summary_semantically_valid,
        external_results_semantically_valid=results_semantically_valid,
    )
    if lineage != expected:
        target.errors.append(
            "promotion_decision.external_eval_lineage must exactly match replayed eval-summary and external-result evidence."
        )
    _validate_promotion_external_eval_check_replay(
        checks,
        expected,
        target,
        evidence_bundle_issue_count=(
            len(evidence_validation.errors) + len(evidence_validation.warnings)
            if evidence_validation is not None
            else 1
        ),
        eval_summary_issue_count=(
            len(summary_validation.errors) + len(summary_validation.warnings)
            if summary_validation is not None
            else 1
        ),
        external_result_issue_count=sum(
            len(validation.errors) + len(validation.warnings)
            if validation is not None
            else 1
            for validation in result_validations
        ),
    )

def _validate_promotion_external_eval_result_artifact(
    artifact: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
    label: str,
) -> None:
    if artifact.get("role") != "external_eval_result":
        target.errors.append(
            f"{label}.role expected 'external_eval_result', got {artifact.get('role')!r}."
        )
    if not isinstance(artifact.get("path"), str):
        target.errors.append(f"{label}.path must be a string.")
    if not isinstance(artifact.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    kind = artifact.get("kind")
    if kind not in {"file", "directory", "missing", "other"}:
        target.errors.append(
            f"{label}.kind must be file, directory, missing, or other."
        )
    if artifact.get("exists") is True:
        if kind not in {"file", "directory"}:
            target.errors.append(
                f"{label}.kind must be file or directory when exists is true."
            )
        if not _is_sha256(artifact.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string.")
    if kind == "file":
        if artifact.get("schema_version") != EXTERNAL_EVAL_RESULT_SCHEMA_VERSION:
            target.errors.append(
                f"{label}.schema_version must be {EXTERNAL_EVAL_RESULT_SCHEMA_VERSION}."
            )
        if not _is_non_negative_int(artifact.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    if kind == "directory" and not _is_non_negative_int(artifact.get("file_count")):
        target.errors.append(f"{label}.file_count must be a non-negative integer.")
    _validate_promotion_decision_artifact_hash(artifact, target, label, source_path)

def _validate_promotion_external_eval_result_projection(
    row: dict[str, Any], target: ValidationTarget, label: str
) -> None:
    for field_name in ("adapter_id", "model_id"):
        if row.get(field_name) is not None and not isinstance(row.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string or null.")
    for field_name in ("plan_sha256", "heldout_manifest_sha256"):
        if row.get(field_name) is not None and not _is_sha256(row.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a SHA-256 value or null.")
    for field_name in (
        "integrity_passed",
        "coverage_complete",
        "external_eval_claims_allowed",
    ):
        if not isinstance(row.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    allowed_values = {
        "execution_status": {"completed", "incomplete", "failed", None},
        "benchmark_status": {"passed", "failed", "inconclusive", "not_available", None},
        "governance_readiness": {"ready_for_review", "blocked", None},
    }
    for field_name, allowed in allowed_values.items():
        if row.get(field_name) not in allowed:
            target.errors.append(f"{label}.{field_name} has an unsupported value.")

def _promotion_existing_artifact_path(value: Any, source_path: Path) -> Path | None:
    if not isinstance(value, dict) or value.get("exists") is not True or value.get("kind") != "file":
        return None
    return _resolve_promotion_decision_artifact_path(
        value.get("path"), source_path, value.get("kind")
    )

def _promotion_read_json_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None

def _validate_promotion_decision_check_replay(
    decision: dict[str, Any],
    artifacts: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    models = decision.get("models")
    model_rows = models if isinstance(models, dict) else {}
    candidate = model_rows.get("candidate")
    champion = model_rows.get("champion")
    rollback = model_rows.get("rollback")
    artifact_paths = {
        role: _promotion_replay_artifact_path(artifacts.get(role), source_path)
        for role in PROMOTION_DECISION_REQUIRED_ARTIFACTS
    }
    lineage = decision.get("external_eval_lineage")
    raw_results = lineage.get("results") if isinstance(lineage, dict) else None
    result_records = [
        row.get("artifact")
        for row in raw_results
        if isinstance(row, dict) and isinstance(row.get("artifact"), dict)
    ] if isinstance(raw_results, list) else []
    result_paths = [
        path
        for record in result_records
        if (path := _promotion_replay_artifact_path(record, source_path)) is not None
    ]
    policy = decision.get("policy")
    policy_artifact = policy.get("artifact") if isinstance(policy, dict) else None
    policy_path = _promotion_replay_artifact_path(policy_artifact, source_path)
    preserve_paths = _promotion_replay_uses_preserved_paths(
        [*artifacts.values(), *result_records, policy_artifact]
    )
    try:
        replay = _build_promotion_decision(
            candidate_id=_promotion_model_id(candidate),
            champion_id=_promotion_model_id(champion),
            rollback_id=_promotion_model_id(rollback) or None,
            candidate_class=(
                candidate.get("class") if isinstance(candidate, dict) else ""
            ),
            champion_class=(
                champion.get("class") if isinstance(champion, dict) else ""
            ),
            out_path=source_path,
            external_eval_result_paths=result_paths,
            promotion_policy_path=policy_path,
            preserve_paths=preserve_paths,
            **{
                f"{role}_path": path
                for role, path in artifact_paths.items()
            },
        )
    except (OSError, ValueError) as exc:
        target.errors.append(
            f"promotion_decision.checks could not be replayed from current source artifacts: {exc}"
        )
        return
    if decision.get("checks") != replay["checks"]:
        target.errors.append(
            "promotion_decision.checks must exactly match canonical replay from current source artifacts."
        )
    if decision.get("policy") != replay["policy"]:
        target.errors.append(
            "promotion_decision.policy must exactly match canonical replay from its current policy source."
        )

def _promotion_replay_artifact_path(value: Any, source_path: Path) -> Path | None:
    if not isinstance(value, dict):
        return None
    return _resolve_promotion_decision_artifact_path(
        value.get("path"), source_path, value.get("kind")
    )

def _promotion_replay_uses_preserved_paths(values: list[Any]) -> bool:
    return any(
        isinstance(value, dict)
        and isinstance(value.get("path"), str)
        and bool(value["path"])
        and Path(value["path"]).is_absolute()
        for value in values
    )

def _validate_promotion_external_eval_check_replay(
    checks: list[Any],
    expected: dict[str, Any],
    target: ValidationTarget,
    *,
    evidence_bundle_issue_count: int,
    eval_summary_issue_count: int,
    external_result_issue_count: int,
) -> None:
    expected_rows = _build_promotion_external_eval_checks(
        expected,
        evidence_bundle_semantically_valid=expected[
            "evidence_bundle_semantically_valid"
        ],
        eval_summary_semantically_valid=expected["eval_summary_semantically_valid"],
        evidence_bundle_error_count=evidence_bundle_issue_count,
        eval_summary_error_count=eval_summary_issue_count,
        external_result_error_count=external_result_issue_count,
    )
    for expected_row in expected_rows:
        check_id = expected_row["id"]
        matching = [
            check
            for check in checks
            if isinstance(check, dict) and check.get("id") == check_id
        ]
        if len(matching) != 1:
            target.errors.append(
                f"promotion_decision.checks must contain exactly one {check_id!r} replay check."
            )
            continue
        if matching[0] != expected_row:
            target.errors.append(
                f"promotion_decision.checks[{check_id!r}] must exactly match replayed external-eval lineage."
            )

def _validate_raw_promotion_policy(policy: dict[str, Any], target: ValidationTarget) -> None:
    _require_equal(policy, "schema_version", PROMOTION_POLICY_SCHEMA_VERSION, target, prefix="promotion_policy.")
    for field_name in ("id", "description"):
        if not isinstance(policy.get(field_name), str) or not policy.get(field_name):
            target.errors.append(f"promotion_policy.{field_name} must be a non-empty string.")
    required_artifacts = _validate_policy_role_list(policy.get("required_artifacts"), target, "promotion_policy.required_artifacts")
    release_required_artifacts = _validate_policy_role_list(
        policy.get("release_required_artifacts"),
        target,
        "promotion_policy.release_required_artifacts",
    )
    _validate_policy_exact_roles(
        required_artifacts,
        PROMOTION_DECISION_REQUIRED_ARTIFACTS,
        target,
        "promotion_policy.required_artifacts",
    )
    _validate_policy_exact_roles(
        release_required_artifacts,
        PROMOTION_RELEASE_RECORD_REQUIRED_ARTIFACTS,
        target,
        "promotion_policy.release_required_artifacts",
    )
    _validate_policy_class_list(policy.get("allowed_candidate_classes"), target, "promotion_policy.allowed_candidate_classes")
    _validate_policy_class_list(policy.get("allowed_champion_classes"), target, "promotion_policy.allowed_champion_classes")
    _validate_policy_limits(policy.get("limits"), target, "promotion_policy.limits")
    _validate_policy_conservative_limits(policy.get("limits"), target, "promotion_policy.limits")
    new_critical_rules = _validate_policy_forbidden_rules(
        policy.get("forbid_new_critical_rules"),
        target,
        "promotion_policy.forbid_new_critical_rules",
    )
    regressed_rules = _validate_policy_forbidden_rules(
        policy.get("forbid_regressed_rules"),
        target,
        "promotion_policy.forbid_regressed_rules",
    )
    _validate_policy_required_forbidden_rules(new_critical_rules, target, "promotion_policy.forbid_new_critical_rules")
    _validate_policy_required_forbidden_rules(regressed_rules, target, "promotion_policy.forbid_regressed_rules")
    for field_name in (
        "require_known_license",
        "require_accepted_terms",
        "require_rollback_metadata",
        "require_supported_cards",
        "require_artifact_validation",
    ):
        if not isinstance(policy.get(field_name), bool):
            target.errors.append(f"promotion_policy.{field_name} must be a boolean.")
        elif policy.get(field_name) is not True:
            target.errors.append(f"promotion_policy.{field_name} must be true; promotion policy cannot relax default safety requirements.")
    target.details.update({"policy_id": policy.get("id"), "schema_version": policy.get("schema_version")})

def _validate_promotion_policy_section(value: Any, target: ValidationTarget, source_path: Path, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _PROMOTION_POLICY_KEYS, target, label)
    _require_equal(value, "schema_version", PROMOTION_POLICY_SCHEMA_VERSION, target, prefix=f"{label}.")
    for field_name in ("id", "description", "source"):
        if not isinstance(value.get(field_name), str) or not value.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    source = value.get("source")
    if source not in {"default", "file"}:
        target.errors.append(f"{label}.source must be default or file.")
    required_artifacts = _validate_policy_role_list(value.get("required_artifacts"), target, f"{label}.required_artifacts")
    release_required_artifacts = _validate_policy_role_list(
        value.get("release_required_artifacts"),
        target,
        f"{label}.release_required_artifacts",
    )
    _validate_policy_class_list(value.get("allowed_candidate_classes"), target, f"{label}.allowed_candidate_classes")
    _validate_policy_class_list(value.get("allowed_champion_classes"), target, f"{label}.allowed_champion_classes")
    _validate_policy_limits(value.get("limits"), target, f"{label}.limits")
    _validate_policy_forbidden_rules(value.get("forbid_new_critical_rules"), target, f"{label}.forbid_new_critical_rules")
    _validate_policy_forbidden_rules(value.get("forbid_regressed_rules"), target, f"{label}.forbid_regressed_rules")
    _validate_policy_requirements(value.get("requirements"), target, f"{label}.requirements")
    artifact = value.get("artifact")
    if source == "file":
        _validate_promotion_fingerprinted_artifact(artifact, "promotion_policy", target, source_path, label)
    elif artifact is not None:
        target.errors.append(f"{label}.artifact must be absent for default policy sources.")
    if not required_artifacts or not release_required_artifacts:
        target.errors.append(f"{label} must declare non-empty decision and release artifact contracts.")

def _validate_policy_role_list(value: Any, target: ValidationTarget, label: str) -> list[str]:
    if not _is_string_list(value):
        target.errors.append(f"{label} must be a list of strings.")
        return []
    return list(value)

def _validate_policy_class_list(value: Any, target: ValidationTarget, label: str) -> None:
    allowed = {"base", "trace-only", "frontier", "champion", "candidate"}
    if not _is_string_list(value):
        target.errors.append(f"{label} must be a list of strings.")
        return
    unknown = sorted(set(value) - allowed)
    if unknown:
        target.errors.append(f"{label} contains unknown model classes: {unknown!r}.")

def _validate_policy_limits(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, set(PROMOTION_POLICY_DEFAULT_LIMITS), target, label)
    expected_fields = (
        "max_task_completion_regressions",
        "max_baseline_wins",
        "max_contract_drifts",
        "max_unverified_contracts",
        "max_new_critical_failures",
        "max_rule_regressions",
    )
    for field_name in expected_fields:
        field_value = value.get(field_name)
        if not _is_non_negative_int(field_value):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")

def _validate_policy_conservative_limits(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        return
    for field_name, maximum in PROMOTION_POLICY_DEFAULT_LIMITS.items():
        field_value = value.get(field_name)
        if _is_non_negative_int(field_value) and field_value > maximum:
            target.errors.append(f"{label}.{field_name} cannot exceed default maximum {maximum}.")

def _validate_policy_forbidden_rules(value: Any, target: ValidationTarget, label: str) -> list[str]:
    if not _is_string_list(value):
        target.errors.append(f"{label} must be a list of strings.")
        return []
    return list(value)

def _validate_policy_required_forbidden_rules(value: list[str], target: ValidationTarget, label: str) -> None:
    if not value:
        return
    missing = sorted(set(PROMOTION_POLICY_REQUIRED_FORBIDDEN_RULES) - set(value))
    if missing:
        target.errors.append(f"{label} must include required zero-tolerance rules: {missing!r}.")

def _validate_policy_exact_roles(value: list[str], required: tuple[str, ...], target: ValidationTarget, label: str) -> None:
    if not value:
        return
    actual = set(value)
    expected = set(required)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        target.errors.append(f"{label} is missing required role(s): {missing!r}.")
    if unknown:
        target.errors.append(f"{label} contains unknown role(s): {unknown!r}.")

def _validate_policy_requirements(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(value, _PROMOTION_POLICY_REQUIREMENT_KEYS, target, label)
    for field_name in (
        "require_known_license",
        "require_accepted_terms",
        "require_rollback_metadata",
        "require_supported_cards",
        "require_artifact_validation",
    ):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")

def _validate_fingerprinted_artifact(
    value: Any,
    role: str,
    target: ValidationTarget,
    source_path: Path,
    prefix: str,
) -> None:
    label = f"{prefix}.{role}"
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if value.get("role") != role:
        target.errors.append(f"{label}.role expected {role!r}, got {value.get('role')!r}.")
    if not isinstance(value.get("path"), str):
        target.errors.append(f"{label}.path must be a string.")
    if not isinstance(value.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    kind = value.get("kind")
    if kind not in {"file", "directory", "missing", "other"}:
        target.errors.append(f"{label}.kind must be file, directory, missing, or other.")
    if value.get("exists") is True:
        if kind not in {"file", "directory"}:
            target.errors.append(f"{label}.kind must be file or directory when exists is true.")
        if not _is_sha256(value.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string when exists is true.")
    if kind == "file" and not _is_non_negative_int(value.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer for files.")
    if kind == "directory" and not _is_non_negative_int(value.get("file_count")):
        target.errors.append(f"{label}.file_count must be a non-negative integer for directories.")
    _validate_promotion_decision_artifact_hash(value, target, label, source_path)

def _validate_promotion_decision_artifact_hash(
    record: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path,
) -> None:
    if record.get("exists") is not True:
        return
    kind = record.get("kind")
    current_path = _resolve_promotion_decision_artifact_path(record.get("path"), source_path, kind)
    if current_path is None:
        return
    if not current_path.exists():
        target.errors.append(f"{label}.path does not exist at validation time.")
        return
    if _promotion_artifact_path_uses_symlink_ancestor(record.get("path"), source_path):
        target.errors.append(f"{label}.path must not resolve through a symlink.")
        return
    if kind == "file":
        if current_path.is_symlink():
            target.errors.append(f"{label}.path must not resolve to a symlink.")
            return
        if not current_path.is_file():
            target.errors.append(f"{label}.path is not a file at validation time.")
            return
        if _is_non_negative_int(record.get("size_bytes")) and current_path.stat().st_size != record.get("size_bytes"):
            target.errors.append(f"{label}.size_bytes does not match the current file.")
        if _is_sha256(record.get("sha256")) and _sha256(current_path) != record.get("sha256"):
            target.errors.append(f"{label}.sha256 does not match the current file.")
    if kind == "directory":
        if current_path.is_symlink():
            target.errors.append(f"{label}.path must not resolve to a symlink.")
            return
        if not current_path.is_dir():
            target.errors.append(f"{label}.path is not a directory at validation time.")
            return
        file_count = sum(1 for item in current_path.rglob("*") if item.is_file())
        if _is_non_negative_int(record.get("file_count")) and file_count != record.get("file_count"):
            target.errors.append(f"{label}.file_count does not match the current directory.")
        if _is_sha256(record.get("sha256")) and _directory_sha256(current_path) != record.get("sha256"):
            target.errors.append(f"{label}.sha256 does not match the current directory.")

def _resolve_promotion_decision_artifact_path(value: Any, source_path: Path, kind: Any) -> Path | None:
    if not isinstance(value, str) or not value or value.startswith("<redacted:"):
        return None
    raw = Path(value)
    return raw if raw.is_absolute() else source_path.parent / raw

def _promotion_artifact_path_uses_symlink_ancestor(value: Any, source_path: Path) -> bool:
    if _path_has_symlink_component(source_path, include_leaf=True):
        return True
    if not isinstance(value, str) or not value or value.startswith("<redacted:"):
        return False
    current_path = _resolve_promotion_decision_artifact_path(value, source_path, None)
    return current_path is not None and _path_has_symlink_component(current_path, include_leaf=False)

def _validate_promotion_decision_metrics(
    value: Any,
    checks: list[Any],
    target: ValidationTarget,
    policy: Any,
    external_eval_lineage: Any,
    artifacts: dict[str, Any],
    source_path: Path,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("promotion_decision.metrics must be an object.")
        return
    _validate_allowed_keys(value, _PROMOTION_DECISION_METRICS_KEYS, target, "promotion_decision.metrics")
    policy_obj = policy if isinstance(policy, dict) else {}
    lineage_obj = external_eval_lineage if isinstance(external_eval_lineage, dict) else {}
    compare_path = _promotion_existing_artifact_path(artifacts.get("compare_gate"), source_path)
    compare_gate = _promotion_read_json_payload(compare_path)
    compare_metrics = (
        compare_gate.get("metrics")
        if isinstance(compare_gate, dict) and isinstance(compare_gate.get("metrics"), dict)
        else {}
    )
    expected_metrics = _build_promotion_decision_metrics(
        [check for check in checks if isinstance(check, dict)],
        compare_metrics,
        policy_obj,
        lineage_obj,
    )
    if value != expected_metrics:
        target.errors.append(
            "promotion_decision.metrics must exactly match replayed checks, policy, compare gate, and external-eval lineage."
        )
    for field_name in (
        "task_completion_regression_count",
        "baseline_win_count",
        "contract_drift_count",
        "unverified_contract_count",
        "new_critical_failure_count",
        "rule_regression_count",
    ):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"promotion_decision.metrics.{field_name} must be a non-negative integer.")

def _validate_promotion_decision_alias_update(
    value: Any,
    expected_passed: bool,
    candidate_id: str,
    champion_id: str,
    rollback_id: str,
    target: ValidationTarget,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("promotion_decision.alias_update must be an object.")
        return
    _validate_allowed_keys(value, _PROMOTION_DECISION_ALIAS_UPDATE_KEYS, target, "promotion_decision.alias_update")
    if value.get("authorized") != expected_passed:
        target.errors.append("promotion_decision.alias_update.authorized must match promotion pass state.")
    expected_recommendation = "apply_alias_update" if expected_passed else "hold_aliases"
    if value.get("recommendation") != expected_recommendation:
        target.errors.append(
            f"promotion_decision.alias_update.recommendation expected {expected_recommendation!r}, got {value.get('recommendation')!r}."
        )
    aliases = value.get("aliases")
    if not isinstance(aliases, list):
        target.errors.append("promotion_decision.alias_update.aliases must be a list.")
        return
    alias_map = {item.get("alias"): item for item in aliases if isinstance(item, dict) and isinstance(item.get("alias"), str)}
    for index, item in enumerate(aliases):
        if isinstance(item, dict):
            _validate_allowed_keys(item, _PROMOTION_DECISION_ALIAS_ROW_KEYS, target, f"promotion_decision.alias_update.aliases[{index}]")
    expected_targets = {"candidate": candidate_id, "champion": candidate_id, "rollback": rollback_id}
    for alias, expected_target in expected_targets.items():
        item = alias_map.get(alias)
        if not isinstance(item, dict):
            target.errors.append(f"promotion_decision.alias_update.aliases must include {alias!r}.")
            continue
        if item.get("target") != expected_target:
            target.errors.append(
                f"promotion_decision.alias_update.aliases[{alias}].target expected {expected_target!r}, got {item.get('target')!r}."
            )
    champion_alias = alias_map.get("champion")
    if isinstance(champion_alias, dict) and champion_alias.get("previous_target") != champion_id:
        target.errors.append(
            "promotion_decision.alias_update.aliases[champion].previous_target must match models.champion.id."
        )

def _validate_promotion_ledger(
    ledger: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
    *,
    validate_sources: bool = True,
) -> None:
    _require_equal(ledger, "schema_version", PROMOTION_LEDGER_SCHEMA_VERSION, target)
    if not isinstance(ledger.get("ledger_path"), str):
        target.errors.append("promotion_ledger.ledger_path must be a string.")
    if ledger.get("passed") is not True:
        target.errors.append("promotion_ledger.passed must be true.")

    records = ledger.get("records")
    if not isinstance(records, list):
        target.errors.append("promotion_ledger.records must be a list.")
        records = []
    metrics = ledger.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("promotion_ledger.metrics must be an object.")
        metrics = {}
    notes = ledger.get("notes")
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        target.errors.append("promotion_ledger.notes must be a list of strings.")

    for index, record in enumerate(records):
        _validate_promotion_ledger_record(
            record,
            target,
            f"promotion_ledger.records[{index}]",
            index,
            source_path,
            validate_sources=validate_sources,
        )

    if ledger.get("decision_count") != len(records):
        target.errors.append(f"promotion_ledger.decision_count expected {len(records)}, got {ledger.get('decision_count')!r}.")

    expected_metrics = _promotion_ledger_expected_metrics(records)
    for field_name in (
        "decision_count",
        "allowed_count",
        "blocked_count",
        "latest_recommendation",
        "latest_readiness",
        "latest_passed",
        "consecutive_allowed_count",
        "consecutive_blocked_count",
        "unique_source_artifact_count",
    ):
        if metrics.get(field_name) != expected_metrics[field_name]:
            target.errors.append(f"promotion_ledger.metrics.{field_name} expected {expected_metrics[field_name]!r}, got {metrics.get(field_name)!r}.")
    _validate_action_ledger_count_rows(
        metrics.get("recommendation_counts"),
        expected_metrics["recommendation_counts"],
        target,
        "promotion_ledger.metrics.recommendation_counts",
    )
    _validate_action_ledger_count_rows(
        metrics.get("source_recommendation_counts"),
        expected_metrics["source_recommendation_counts"],
        target,
        "promotion_ledger.metrics.source_recommendation_counts",
    )
    if metrics.get("decision_gate_results") != expected_metrics["decision_gate_results"]:
        target.errors.append("promotion_ledger.metrics.decision_gate_results must match promotion_ledger.records.")

    target.details.update(
        {
            "decision_count": len(records),
            "allowed_count": expected_metrics["allowed_count"],
            "blocked_count": expected_metrics["blocked_count"],
            "latest_recommendation": expected_metrics["latest_recommendation"],
        }
    )

def _validate_promotion_ledger_record(
    record: Any,
    target: ValidationTarget,
    label: str,
    expected_index: int,
    source_path: Path,
    *,
    validate_sources: bool = True,
) -> None:
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if record.get("index") != expected_index:
        target.errors.append(f"{label}.index expected {expected_index}, got {record.get('index')!r}.")
    for field_name in ("path", "schema_version", "readiness", "recommendation", "expected_recommendation"):
        if not isinstance(record.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if record.get("schema_version") != DECISION_GATE_SCHEMA_VERSION:
        target.errors.append(f"{label}.schema_version must be {DECISION_GATE_SCHEMA_VERSION}.")
    if record.get("expected_readiness") is not None and not isinstance(record.get("expected_readiness"), str):
        target.errors.append(f"{label}.expected_readiness must be a string or null.")
    for field_name in ("exists", "passed", "require_passed"):
        if not isinstance(record.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if record.get("exists") is False:
        target.errors.append(f"{label}.exists must be true.")
    for field_name in ("check_count", "failed_check_count"):
        if not _is_non_negative_int(record.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    if _is_non_negative_int(record.get("check_count")) and _is_non_negative_int(record.get("failed_check_count")):
        if record["failed_check_count"] > record["check_count"]:
            target.errors.append(f"{label}.failed_check_count must be less than or equal to check_count.")
    if record.get("exists") is True and validate_sources:
        _validate_promotion_ledger_record_file_hash(record, target, label, source_path)
    if record.get("recommendation") not in {"allow_promotion", "block_promotion"}:
        target.errors.append(f"{label}.recommendation must be allow_promotion or block_promotion.")
    expected_allowed = record.get("passed") is True and record.get("recommendation") == "allow_promotion"
    expected_blocked = record.get("passed") is not True or record.get("recommendation") == "block_promotion"
    if expected_allowed and expected_blocked:
        target.errors.append(f"{label} cannot be both allowed and blocked.")

    source = record.get("source")
    if not isinstance(source, dict):
        target.errors.append(f"{label}.source must be an object.")
        source = {}
    _validate_promotion_ledger_source(source, target, f"{label}.source")
    if validate_sources:
        _validate_promotion_ledger_record_matches_gate(record, source, target, label, source_path)

def _validate_promotion_ledger_source(source: dict[str, Any], target: ValidationTarget, label: str) -> None:
    for field_name in ("schema_version", "recommendation", "readiness", "artifact_path"):
        if not isinstance(source.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if source.get("passed") is not None and not isinstance(source.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean or null.")
    if source.get("blocking_check_count") is not None and not _is_non_negative_int(source.get("blocking_check_count")):
        target.errors.append(f"{label}.blocking_check_count must be a non-negative integer or null.")
    if not isinstance(source.get("artifact_exists"), bool):
        target.errors.append(f"{label}.artifact_exists must be a boolean.")
    if source.get("artifact_sha256") is not None and not _is_sha256(source.get("artifact_sha256")):
        target.errors.append(f"{label}.artifact_sha256 must be a SHA-256 hex string or null.")
    if source.get("artifact_exists") is True and not _is_sha256(source.get("artifact_sha256")):
        target.errors.append(f"{label}.artifact_sha256 must be present for existing source artifacts.")

def _validate_promotion_ledger_record_matches_gate(
    record: dict[str, Any],
    source: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path,
) -> None:
    file_path = _resolve_gate_source_path(record.get("path"), source_path)
    if file_path is None or not file_path.exists() or not file_path.is_file():
        return
    if file_path.is_symlink() or _path_has_symlink_component(file_path, include_leaf=False):
        return
    try:
        gate = json.loads(file_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        target.errors.append(f"{label}.path is not valid UTF-8: {exc}")
        return
    except json.JSONDecodeError as exc:
        target.errors.append(f"{label}.path contains invalid JSON: {exc}")
        return
    if not isinstance(gate, dict):
        target.errors.append(f"{label}.path must contain a JSON object.")
        return

    gate_schema_check = check_schema_contract(gate, name_or_id="decision_gate", artifact_path=file_path)
    nested_gate_target = ValidationTarget("decision_gate", str(file_path))
    _validate_decision_gate(gate, nested_gate_target, file_path)
    if not gate_schema_check.get("passed") or nested_gate_target.errors:
        schema_errors = gate_schema_check.get("errors") if isinstance(gate_schema_check.get("errors"), list) else []
        detail = str(schema_errors[0]) if schema_errors else nested_gate_target.errors[0]
        target.errors.append(f"{label}.path must reference a valid decision gate: {detail}")

    gate_source = gate.get("source_decision") if isinstance(gate.get("source_decision"), dict) else {}
    gate_artifact = gate.get("source_artifact") if isinstance(gate.get("source_artifact"), dict) else {}
    expected = {
        "schema_version": str(gate.get("schema_version") or ""),
        "passed": gate.get("passed") is True,
        "readiness": str(gate.get("readiness") or ""),
        "recommendation": str(gate.get("recommendation") or ""),
        "expected_recommendation": str(gate.get("expected_recommendation") or ""),
        "expected_readiness": gate.get("expected_readiness") if isinstance(gate.get("expected_readiness"), str) else None,
        "require_passed": gate.get("require_passed") is True,
        "check_count": gate.get("check_count") if _is_non_negative_int(gate.get("check_count")) else 0,
        "failed_check_count": gate.get("failed_check_count") if _is_non_negative_int(gate.get("failed_check_count")) else 0,
    }
    for field_name, expected_value in expected.items():
        if record.get(field_name) != expected_value:
            target.errors.append(f"{label}.{field_name} must match current decision gate.")

    expected_source = {
        "schema_version": str(gate_source.get("schema_version") or ""),
        "passed": gate_source.get("passed") if isinstance(gate_source.get("passed"), bool) else None,
        "recommendation": str(gate_source.get("recommendation") or ""),
        "readiness": str(gate_source.get("readiness") or ""),
        "blocking_check_count": gate_source.get("blocking_check_count") if _is_non_negative_int(gate_source.get("blocking_check_count")) else None,
        "artifact_path": str(gate_artifact.get("path") or ""),
        "artifact_exists": gate_artifact.get("exists") is True,
        "artifact_sha256": gate_artifact.get("sha256") if _is_sha256(gate_artifact.get("sha256")) else None,
    }
    for field_name, expected_value in expected_source.items():
        if source.get(field_name) != expected_value:
            target.errors.append(f"{label}.source.{field_name} must match current decision gate.")

def _validate_promotion_ledger_record_file_hash(
    record: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path,
) -> None:
    if "regular_file" in record and not isinstance(record.get("regular_file"), bool):
        target.errors.append(f"{label}.regular_file must be a boolean when present.")
    if "symlink" in record and not isinstance(record.get("symlink"), bool):
        target.errors.append(f"{label}.symlink must be a boolean when present.")
    if record.get("regular_file") is False:
        target.errors.append(f"{label}.regular_file must be true when present.")
    if not _is_non_negative_int(record.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer for existing files.")
    if not _is_sha256(record.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string for existing files.")
        return
    file_path = _resolve_gate_source_path(record.get("path"), source_path)
    if file_path is None:
        target.errors.append(f"{label}.path must resolve to a local file.")
        return
    if file_path.is_symlink():
        target.errors.append(f"{label}.path must not resolve to a symlink.")
        return
    if _path_has_symlink_component(file_path, include_leaf=False):
        target.errors.append(f"{label}.path must not traverse symlinked components.")
        return
    if not file_path.exists() or not file_path.is_file():
        target.errors.append(f"{label}.path does not resolve to an existing file.")
        return
    if file_path.stat().st_size != record.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if _sha256(file_path) != record.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current file.")

def _promotion_ledger_expected_metrics(records: list[Any]) -> dict[str, Any]:
    valid_records = [record for record in records if isinstance(record, dict)]
    latest = valid_records[-1] if valid_records else {}
    source_keys = {
        _promotion_source_artifact_key(record)
        for record in valid_records
        if _promotion_source_artifact_key(record)
    }
    return {
        "decision_count": len(records),
        "allowed_count": sum(1 for record in valid_records if _promotion_record_allowed(record)),
        "blocked_count": sum(1 for record in valid_records if _promotion_record_blocked(record)),
        "latest_recommendation": latest.get("recommendation") if valid_records else "",
        "latest_readiness": latest.get("readiness") if valid_records else "",
        "latest_passed": latest.get("passed") if valid_records else None,
        "consecutive_allowed_count": _promotion_consecutive(valid_records, _promotion_record_allowed),
        "consecutive_blocked_count": _promotion_consecutive(valid_records, _promotion_record_blocked),
        "unique_source_artifact_count": len(source_keys),
        "recommendation_counts": _promotion_count_map(record.get("recommendation") for record in valid_records),
        "source_recommendation_counts": _promotion_count_map(
            record.get("source", {}).get("recommendation") if isinstance(record.get("source"), dict) else ""
            for record in valid_records
        ),
        "decision_gate_results": [
            {
                "index": record.get("index"),
                "path": record.get("path"),
                "passed": record.get("passed"),
                "recommendation": record.get("recommendation"),
                "source_recommendation": record.get("source", {}).get("recommendation") if isinstance(record.get("source"), dict) else "",
                "failed_check_count": record.get("failed_check_count"),
            }
            for record in valid_records
        ],
    }

def _promotion_record_allowed(record: dict[str, Any]) -> bool:
    return record.get("passed") is True and record.get("recommendation") == "allow_promotion"

def _promotion_record_blocked(record: dict[str, Any]) -> bool:
    return record.get("passed") is not True or record.get("recommendation") == "block_promotion"

def _promotion_consecutive(records: list[dict[str, Any]], predicate: Any) -> int:
    count = 0
    for record in reversed(records):
        if not predicate(record):
            break
        count += 1
    return count

def _promotion_source_artifact_key(record: dict[str, Any]) -> str:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    sha256 = source.get("artifact_sha256")
    if isinstance(sha256, str) and sha256:
        return sha256
    path = source.get("artifact_path")
    return path if isinstance(path, str) and path else ""

def _promotion_count_map(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts

def _validate_promotion_ledger_gate(gate: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(gate, "schema_version", PROMOTION_LEDGER_GATE_SCHEMA_VERSION, target)
    if not isinstance(gate.get("promotion_ledger"), str) or not gate.get("promotion_ledger"):
        target.errors.append("promotion_ledger_gate.promotion_ledger must be a non-empty string.")
    else:
        _warn_absolute_public_path(target, "promotion_ledger_gate.promotion_ledger", gate.get("promotion_ledger"))
    if not isinstance(gate.get("passed"), bool):
        target.errors.append("promotion_ledger_gate.passed must be a boolean.")
    checks = gate.get("checks")
    if not isinstance(checks, list):
        target.errors.append("promotion_ledger_gate.checks must be a list.")
        checks = []
    metrics = gate.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("promotion_ledger_gate.metrics must be an object.")
        metrics = {}
    if "policy" in gate:
        _validate_promotion_ledger_gate_policy_summary(gate.get("policy"), target)
        _validate_promotion_ledger_gate_policy_check_coverage(gate.get("policy"), checks, target)

    failed_checks = _validate_gate_like_checks(checks, target, "promotion_ledger_gate.checks")
    if gate.get("check_count") != len(checks):
        target.errors.append(f"promotion_ledger_gate.check_count expected {len(checks)}, got {gate.get('check_count')!r}.")
    if gate.get("failed_check_count") != failed_checks:
        target.errors.append(
            f"promotion_ledger_gate.failed_check_count expected {failed_checks}, got {gate.get('failed_check_count')!r}."
        )
    expected_passed = failed_checks == 0
    if isinstance(gate.get("passed"), bool) and gate.get("passed") != expected_passed:
        target.errors.append("promotion_ledger_gate.passed must match failed_check_count.")
    _validate_promotion_ledger_gate_metrics(metrics, target)
    _validate_promotion_ledger_gate_source_linkage(gate, checks, metrics, target, source_path)
    _validate_promotion_ledger_gate_decision(gate.get("decision"), expected_passed, failed_checks, metrics, target)
    target.details.update(
        {
            "passed": gate.get("passed"),
            "check_count": len(checks),
            "failed_check_count": failed_checks,
            "decision_count": metrics.get("decision_count"),
            "latest_recommendation": metrics.get("latest_recommendation"),
        }
    )

def _validate_promotion_ledger_gate_metrics(metrics: dict[str, Any], target: ValidationTarget) -> None:
    count_fields = (
        "decision_count",
        "allowed_count",
        "blocked_count",
        "consecutive_allowed_count",
        "consecutive_blocked_count",
        "failed_decision_count",
        "unique_source_artifact_count",
    )
    for field_name in count_fields:
        if not _is_non_negative_int(metrics.get(field_name)):
            target.errors.append(f"promotion_ledger_gate.metrics.{field_name} must be a non-negative integer.")
    if all(_is_non_negative_int(metrics.get(field_name)) for field_name in ("decision_count", "allowed_count", "blocked_count")):
        if metrics["allowed_count"] + metrics["blocked_count"] != metrics["decision_count"]:
            target.errors.append("promotion_ledger_gate.metrics.allowed_count + blocked_count must equal decision_count.")
    if not _is_number_between(metrics.get("blocked_rate"), 0.0, 1.0):
        target.errors.append("promotion_ledger_gate.metrics.blocked_rate must be a number from 0.0 to 1.0.")
    elif _is_non_negative_int(metrics.get("decision_count")) and _is_non_negative_int(metrics.get("blocked_count")):
        expected_rate = round(metrics["blocked_count"] / metrics["decision_count"], 4) if metrics["decision_count"] else 0.0
        if metrics.get("blocked_rate") != expected_rate:
            target.errors.append(f"promotion_ledger_gate.metrics.blocked_rate expected {expected_rate}, got {metrics.get('blocked_rate')!r}.")
    for field_name in ("latest_recommendation", "latest_readiness"):
        if not isinstance(metrics.get(field_name), str):
            target.errors.append(f"promotion_ledger_gate.metrics.{field_name} must be a string.")
    if metrics.get("latest_passed") is not None and not isinstance(metrics.get("latest_passed"), bool):
        target.errors.append("promotion_ledger_gate.metrics.latest_passed must be a boolean or null.")
    if _count_rows(metrics.get("source_recommendation_counts")) is None:
        target.errors.append("promotion_ledger_gate.metrics.source_recommendation_counts must be a list of {id, count} objects.")

def _validate_promotion_ledger_gate_decision(
    value: Any,
    expected_passed: bool,
    failed_checks: int,
    metrics: dict[str, Any],
    target: ValidationTarget,
) -> None:
    if not isinstance(value, dict):
        target.errors.append("promotion_ledger_gate.decision must be an object.")
        return
    expected_readiness = "ready" if expected_passed else "blocked"
    expected_recommendation = "promote_iteration" if expected_passed else "block_iteration"
    if value.get("readiness") != expected_readiness:
        target.errors.append(f"promotion_ledger_gate.decision.readiness expected {expected_readiness!r}, got {value.get('readiness')!r}.")
    if value.get("recommendation") != expected_recommendation:
        target.errors.append(
            "promotion_ledger_gate.decision.recommendation expected "
            f"{expected_recommendation!r}, got {value.get('recommendation')!r}."
        )
    if not isinstance(value.get("summary"), str) or not value.get("summary"):
        target.errors.append("promotion_ledger_gate.decision.summary must be a non-empty string.")
    blocking_checks = value.get("blocking_checks")
    if not isinstance(blocking_checks, list):
        target.errors.append("promotion_ledger_gate.decision.blocking_checks must be a list.")
        blocking_checks = []
    if value.get("blocking_check_count") != failed_checks:
        target.errors.append(
            f"promotion_ledger_gate.decision.blocking_check_count expected {failed_checks}, got {value.get('blocking_check_count')!r}."
        )
    if len(blocking_checks) != failed_checks:
        target.errors.append(
            f"promotion_ledger_gate.decision.blocking_checks expected {failed_checks} entries, got {len(blocking_checks)}."
        )
    for index, check in enumerate(blocking_checks):
        label = f"promotion_ledger_gate.decision.blocking_checks[{index}]"
        if not isinstance(check, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("id", "summary"):
            if not isinstance(check.get(field_name), str) or not check.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if not isinstance(check.get("scope"), dict):
            target.errors.append(f"{label}.scope must be an object.")
    key_metrics = value.get("key_metrics")
    if not isinstance(key_metrics, dict):
        target.errors.append("promotion_ledger_gate.decision.key_metrics must be an object.")
        return
    for field_name in (
        "decision_count",
        "allowed_count",
        "blocked_count",
        "blocked_rate",
        "latest_recommendation",
        "latest_passed",
        "consecutive_allowed_count",
        "consecutive_blocked_count",
        "failed_decision_count",
        "source_recommendation_counts",
    ):
        if key_metrics.get(field_name) != metrics.get(field_name):
            target.errors.append(
                f"promotion_ledger_gate.decision.key_metrics.{field_name} must match promotion_ledger_gate.metrics.{field_name}."
            )

def _validate_promotion_ledger_gate_source_linkage(
    gate: dict[str, Any],
    checks: list[Any],
    metrics: dict[str, Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    ledger_path = _resolve_gate_source_path(gate.get("promotion_ledger"), source_path)
    if ledger_path is None or not ledger_path.exists():
        target.errors.append("promotion_ledger_gate.promotion_ledger must resolve to an existing promotion ledger.")
        return
    if _path_has_symlink_component(ledger_path, include_leaf=True) or not ledger_path.is_file():
        target.errors.append("promotion_ledger_gate.promotion_ledger must resolve to a regular non-symlink promotion ledger.")
        return
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        target.errors.append(f"promotion_ledger_gate.promotion_ledger is not valid UTF-8: {exc}")
        return
    except json.JSONDecodeError as exc:
        target.errors.append(f"promotion_ledger_gate.promotion_ledger contains invalid JSON: {exc}")
        return
    if not isinstance(ledger, dict):
        target.errors.append("promotion_ledger_gate.promotion_ledger must contain a JSON object.")
        return
    _validate_promotion_ledger(ledger, target, ledger_path)
    try:
        expected = evaluate_promotion_ledger_gate(
            ledger,
            promotion_ledger_path=ledger_path,
            promotion_ledger_display_path=gate.get("promotion_ledger"),
            **_promotion_ledger_gate_replay_options(gate, checks),
        )
    except ValueError as exc:
        target.errors.append(f"promotion_ledger_gate.promotion_ledger could not be replayed: {exc}")
        return
    if metrics != expected.get("metrics"):
        target.errors.append("promotion_ledger_gate.metrics must match replayed source ledger metrics.")
    if checks != expected.get("checks"):
        target.errors.append("promotion_ledger_gate.checks must match replayed source ledger checks.")
    if gate.get("decision") != expected.get("decision"):
        target.errors.append("promotion_ledger_gate.decision must match replayed source ledger decision.")

def _validate_promotion_ledger_gate_policy_check_coverage(value: Any, checks: list[Any], target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        return
    effective = value.get("effective")
    if not isinstance(effective, dict):
        return
    expected: Counter[tuple[Any, ...]] = Counter()
    for field_name in (
        "min_decisions",
        "min_allowed_count",
        "max_blocked_count",
        "max_blocked_rate",
        "min_consecutive_allowed",
        "max_consecutive_blocked",
        "max_failed_decisions",
        "require_latest_recommendation",
    ):
        if field_name in effective:
            expected[(field_name, None)] += 1
    if effective.get("require_latest_passed") is True:
        expected[("require_latest_passed", None)] += 1
    required_recommendations = effective.get("require_source_recommendations")
    if isinstance(required_recommendations, list):
        for recommendation in required_recommendations:
            if isinstance(recommendation, str):
                expected[("require_source_recommendation", recommendation)] += 1
    forbidden_recommendations = effective.get("forbid_source_recommendations")
    if isinstance(forbidden_recommendations, list):
        for recommendation in forbidden_recommendations:
            if isinstance(recommendation, str):
                expected[("forbid_source_recommendation", recommendation)] += 1
    actual = Counter(_promotion_ledger_gate_check_policy_key(check) for check in checks if isinstance(check, dict))
    actual.pop(None, None)
    if expected != actual:
        target.errors.append("promotion_ledger_gate.checks must cover promotion_ledger_gate.policy.effective requirements.")

def _promotion_ledger_gate_check_policy_key(check: dict[str, Any]) -> tuple[Any, ...] | None:
    check_id = check.get("id")
    if check_id in {
        "min_decisions",
        "min_allowed_count",
        "max_blocked_count",
        "max_blocked_rate",
        "min_consecutive_allowed",
        "max_consecutive_blocked",
        "max_failed_decisions",
        "require_latest_recommendation",
        "require_latest_passed",
    }:
        return (check_id, None)
    scope = check.get("scope") if isinstance(check.get("scope"), dict) else {}
    if check_id in {"require_source_recommendation", "forbid_source_recommendation"}:
        return (check_id, scope.get("recommendation"))
    return None

def _promotion_ledger_gate_replay_options(gate: dict[str, Any], checks: list[Any]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "min_decisions": None,
        "min_allowed_count": None,
        "max_blocked_count": None,
        "max_blocked_rate": None,
        "min_consecutive_allowed": None,
        "max_consecutive_blocked": None,
        "max_failed_decisions": None,
        "require_latest_recommendation": None,
        "require_latest_passed": False,
        "require_source_recommendations": [],
        "forbid_source_recommendations": [],
    }
    policy = gate.get("policy") if isinstance(gate.get("policy"), dict) else {}
    effective = policy.get("effective") if isinstance(policy.get("effective"), dict) else None
    if effective is not None:
        for field_name in (
            "min_decisions",
            "min_allowed_count",
            "max_blocked_count",
            "min_consecutive_allowed",
            "max_consecutive_blocked",
            "max_failed_decisions",
        ):
            if _is_non_negative_int(effective.get(field_name)):
                options[field_name] = effective[field_name]
        if _is_number_between(effective.get("max_blocked_rate"), 0.0, 1.0):
            options["max_blocked_rate"] = effective["max_blocked_rate"]
        if isinstance(effective.get("require_latest_recommendation"), str):
            options["require_latest_recommendation"] = effective["require_latest_recommendation"]
        if effective.get("require_latest_passed") is True:
            options["require_latest_passed"] = True
        if _is_string_list(effective.get("require_source_recommendations")):
            options["require_source_recommendations"] = effective["require_source_recommendations"]
        if _is_string_list(effective.get("forbid_source_recommendations")):
            options["forbid_source_recommendations"] = effective["forbid_source_recommendations"]
        return options

    for check in checks:
        if isinstance(check, dict):
            _merge_promotion_ledger_gate_check_option(options, check)
    return options

def _merge_promotion_ledger_gate_check_option(options: dict[str, Any], check: dict[str, Any]) -> None:
    expected = check.get("expected") if isinstance(check.get("expected"), dict) else {}
    check_id = check.get("id")
    if check_id in {"min_decisions", "min_allowed_count", "min_consecutive_allowed"} and _is_non_negative_int(expected.get("min")):
        options[check_id] = expected["min"]
    elif check_id in {"max_blocked_count", "max_consecutive_blocked", "max_failed_decisions"} and _is_non_negative_int(expected.get("max")):
        options[check_id] = expected["max"]
    elif check_id == "max_blocked_rate" and _is_number_between(expected.get("max"), 0.0, 1.0):
        options[check_id] = expected["max"]
    elif check_id == "require_latest_recommendation":
        value = expected.get("value")
        if isinstance(value, str) and value:
            options["require_latest_recommendation"] = value
    elif check_id == "require_latest_passed":
        if expected.get("value") is True:
            options["require_latest_passed"] = True
    elif check_id == "require_source_recommendation":
        scope = check.get("scope") if isinstance(check.get("scope"), dict) else {}
        recommendation = scope.get("recommendation")
        if isinstance(recommendation, str) and recommendation:
            options["require_source_recommendations"].append(recommendation)
    elif check_id == "forbid_source_recommendation":
        scope = check.get("scope") if isinstance(check.get("scope"), dict) else {}
        recommendation = scope.get("recommendation")
        if isinstance(recommendation, str) and recommendation:
            options["forbid_source_recommendations"].append(recommendation)

def _validate_promotion_ledger_gate_policy_summary(value: Any, target: ValidationTarget) -> None:
    if not isinstance(value, dict):
        target.errors.append("promotion_ledger_gate.policy must be an object when present.")
        return
    _require_equal(value, "schema_version", PROMOTION_LEDGER_GATE_POLICY_SCHEMA_VERSION, target, prefix="promotion_ledger_gate.policy.")
    if not isinstance(value.get("path"), str) or not value.get("path"):
        target.errors.append("promotion_ledger_gate.policy.path must be a non-empty string.")
    else:
        _warn_absolute_public_path(target, "promotion_ledger_gate.policy.path", value.get("path"))
    if "description" in value and not isinstance(value.get("description"), str):
        target.errors.append("promotion_ledger_gate.policy.description must be a string when present.")
    effective = value.get("effective")
    if not isinstance(effective, dict):
        target.errors.append("promotion_ledger_gate.policy.effective must be an object.")
        return
    allowed_fields = {
        "min_decisions",
        "min_allowed_count",
        "max_blocked_count",
        "max_blocked_rate",
        "min_consecutive_allowed",
        "max_consecutive_blocked",
        "max_failed_decisions",
        "require_latest_recommendation",
        "require_latest_passed",
        "require_source_recommendations",
        "forbid_source_recommendations",
    }
    unknown = sorted(set(effective) - allowed_fields)
    if unknown:
        target.errors.append(f"promotion_ledger_gate.policy.effective has unknown field(s): {', '.join(unknown)}.")
    for field_name in (
        "min_decisions",
        "min_allowed_count",
        "max_blocked_count",
        "min_consecutive_allowed",
        "max_consecutive_blocked",
        "max_failed_decisions",
    ):
        if field_name in effective and not _is_non_negative_int(effective.get(field_name)):
            target.errors.append(f"promotion_ledger_gate.policy.effective.{field_name} must be a non-negative integer.")
    if "max_blocked_rate" in effective and not _is_number_between(effective.get("max_blocked_rate"), 0.0, 1.0):
        target.errors.append("promotion_ledger_gate.policy.effective.max_blocked_rate must be a number from 0.0 to 1.0.")
    if "require_latest_recommendation" in effective and effective.get("require_latest_recommendation") not in {
        "allow_promotion",
        "block_promotion",
    }:
        target.errors.append("promotion_ledger_gate.policy.effective.require_latest_recommendation is invalid.")
    if "require_latest_passed" in effective and not isinstance(effective.get("require_latest_passed"), bool):
        target.errors.append("promotion_ledger_gate.policy.effective.require_latest_passed must be a boolean.")
    for field_name in ("require_source_recommendations", "forbid_source_recommendations"):
        if field_name in effective and not _is_string_list(effective.get(field_name)):
            target.errors.append(f"promotion_ledger_gate.policy.effective.{field_name} must be a list of strings.")

def _validate_promotion_archive(archive: dict[str, Any], target: ValidationTarget, archive_root: Path) -> None:
    _validate_allowed_keys(archive, _PROMOTION_ARCHIVE_KEYS, target, "promotion_archive")
    _require_equal(archive, "schema_version", PROMOTION_ARCHIVE_SCHEMA_VERSION, target)
    for field_name in ("archive_path", "manifest_path"):
        if not isinstance(archive.get(field_name), str) or not archive.get(field_name):
            target.errors.append(f"promotion_archive.{field_name} must be a non-empty string.")
        _warn_absolute_public_path(target, f"promotion_archive.{field_name}", archive.get(field_name))
    for field_name in ("passed", "self_contained", "require_self_contained"):
        if not isinstance(archive.get(field_name), bool):
            target.errors.append(f"promotion_archive.{field_name} must be a boolean.")
    artifacts = archive.get("artifacts")
    if not isinstance(artifacts, list):
        target.errors.append("promotion_archive.artifacts must be a list.")
        artifacts = []
    missing = archive.get("missing")
    if not isinstance(missing, list):
        target.errors.append("promotion_archive.missing must be a list.")
        missing = []
    relationships = archive.get("relationships")
    if not isinstance(relationships, list):
        target.errors.append("promotion_archive.relationships must be a list.")
        relationships = []
    metrics = archive.get("metrics")
    if not isinstance(metrics, dict):
        target.errors.append("promotion_archive.metrics must be an object.")
        metrics = {}
    if not _is_string_list(archive.get("notes")):
        target.errors.append("promotion_archive.notes must be a list of strings.")

    for index, artifact in enumerate(artifacts):
        _validate_promotion_archive_artifact(artifact, target, f"promotion_archive.artifacts[{index}]", index, archive_root)
    for index, item in enumerate(missing):
        _validate_promotion_archive_missing(item, target, f"promotion_archive.missing[{index}]")
    artifact_roles_by_name = _archive_artifact_roles_by_name(artifacts, target, "promotion_archive", require_unique=True)
    allowed_relationship_edges = {
        "gates": {("promotion_ledger_gate", "promotion_ledger")},
        "summarizes": {("promotion_ledger", "decision_gate")},
        "source_artifact": {("decision_gate", "source_artifact")},
        "release_record": {("promotion_release_record", "promotion_ledger")},
    }
    for index, relationship in enumerate(relationships):
        _validate_archive_relationship(
            relationship,
            target,
            f"promotion_archive.relationships[{index}]",
            artifact_roles_by_name,
            allowed_relationship_edges,
        )

    self_contained = len(missing) == 0
    if isinstance(archive.get("self_contained"), bool) and archive["self_contained"] != self_contained:
        target.errors.append(f"promotion_archive.self_contained expected {self_contained}, got {archive.get('self_contained')!r}.")
    expected_passed = self_contained or archive.get("require_self_contained") is not True
    if isinstance(archive.get("passed"), bool) and archive["passed"] != expected_passed:
        target.errors.append(f"promotion_archive.passed expected {expected_passed}, got {archive.get('passed')!r}.")
    _validate_promotion_archive_metrics(metrics, artifacts, missing, target)
    roles = {artifact.get("role") for artifact in artifacts if isinstance(artifact, dict)}
    for required_role in ("promotion_ledger",):
        if required_role not in roles:
            target.errors.append(f"promotion_archive.artifacts must include role {required_role}.")
    target.details.update(
        {
            "artifact_count": len(artifacts),
            "missing_count": len(missing),
            "self_contained": archive.get("self_contained"),
            "passed": archive.get("passed"),
        }
    )

def _validate_promotion_archive_artifact(
    artifact: Any,
    target: ValidationTarget,
    label: str,
    expected_index: int,
    archive_root: Path,
) -> None:
    if not isinstance(artifact, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(artifact, _PROMOTION_ARCHIVE_ARTIFACT_KEYS, target, label)
    if artifact.get("index") != expected_index:
        target.errors.append(f"{label}.index expected {expected_index}, got {artifact.get('index')!r}.")
    for field_name in ("name", "role", "path", "original_path", "schema_version"):
        if not isinstance(artifact.get(field_name), str) or not artifact.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    _warn_absolute_public_path(target, f"{label}.original_path", artifact.get("original_path"))
    if artifact.get("role") not in {"promotion_ledger", "promotion_ledger_gate", "decision_gate", "source_artifact", "promotion_release_record"}:
        target.errors.append(f"{label}.role is invalid.")
    if artifact.get("exists") is not True:
        target.errors.append(f"{label}.exists must be true.")
    if not _is_non_negative_int(artifact.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    if not _is_sha256(artifact.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string.")
        return
    artifact_path = _archive_artifact_path(artifact.get("path"), archive_root)
    if artifact_path is None:
        target.errors.append(f"{label}.path must be a relative archive path.")
        return
    if not _path_resolves_inside(artifact_path, archive_root):
        target.errors.append(f"{label}.path must resolve inside the archive.")
        return
    if _reject_archive_artifact_symlink_path(artifact_path, target, label):
        return
    if not artifact_path.exists() or not artifact_path.is_file():
        target.errors.append(f"{label}.path does not exist inside the archive.")
        return
    if artifact_path.stat().st_size != artifact.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the archived file.")
    if _sha256(artifact_path) != artifact.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the archived file.")

def _validate_promotion_archive_missing(item: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(item, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _validate_allowed_keys(item, _PROMOTION_ARCHIVE_MISSING_KEYS, target, label)
    if item.get("role") not in {"decision_gate", "source_artifact"}:
        target.errors.append(f"{label}.role must be decision_gate or source_artifact.")
    if not _is_non_negative_int(item.get("index")):
        target.errors.append(f"{label}.index must be a non-negative integer.")
    if not isinstance(item.get("reason"), str) or not item.get("reason"):
        target.errors.append(f"{label}.reason must be a non-empty string.")

def _validate_promotion_archive_metrics(
    metrics: dict[str, Any],
    artifacts: list[Any],
    missing: list[Any],
    target: ValidationTarget,
) -> None:
    _validate_allowed_keys(metrics, _PROMOTION_ARCHIVE_METRICS_KEYS, target, "promotion_archive.metrics")
    valid_artifacts = [artifact for artifact in artifacts if isinstance(artifact, dict)]
    valid_missing = [item for item in missing if isinstance(item, dict)]
    expected = {
        "artifact_count": len(artifacts),
        "decision_gate_count": sum(1 for artifact in valid_artifacts if artifact.get("role") == "decision_gate"),
        "source_artifact_count": sum(1 for artifact in valid_artifacts if artifact.get("role") == "source_artifact"),
        "missing_count": len(missing),
        "unique_sha256_count": len({artifact.get("sha256") for artifact in valid_artifacts if isinstance(artifact.get("sha256"), str)}),
    }
    if "promotion_release_record_count" in metrics:
        expected["promotion_release_record_count"] = sum(
            1 for artifact in valid_artifacts if artifact.get("role") == "promotion_release_record"
        )
    for field_name, expected_value in expected.items():
        if metrics.get(field_name) != expected_value:
            target.errors.append(f"promotion_archive.metrics.{field_name} expected {expected_value}, got {metrics.get(field_name)!r}.")
    role_counts = _promotion_archive_count_map(artifact.get("role") for artifact in valid_artifacts)
    missing_role_counts = _promotion_archive_count_map(item.get("role") for item in valid_missing)
    _validate_action_ledger_count_rows(metrics.get("role_counts"), role_counts, target, "promotion_archive.metrics.role_counts")
    _validate_action_ledger_count_rows(
        metrics.get("missing_role_counts"),
        missing_role_counts,
        target,
        "promotion_archive.metrics.missing_role_counts",
    )

def _promotion_archive_count_map(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts

def _validate_action_ledger_bundle(bundle: Any, target: ValidationTarget, label: str, expected_index: int) -> None:
    if not isinstance(bundle, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if bundle.get("index") != expected_index:
        target.errors.append(f"{label}.index expected {expected_index}, got {bundle.get('index')!r}.")
    for field_name in ("path", "schema_version", "readiness", "recommendation"):
        if not isinstance(bundle.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if isinstance(bundle.get("path"), str):
        _warn_absolute_public_path(target, f"{label}.path", bundle.get("path"))
    if bundle.get("schema_version") != EVIDENCE_BUNDLE_SCHEMA_VERSION:
        target.errors.append(f"{label}.schema_version must be {EVIDENCE_BUNDLE_SCHEMA_VERSION}.")
    for field_name in ("exists", "passed"):
        if not isinstance(bundle.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if not _is_non_negative_int(bundle.get("action_count")):
        target.errors.append(f"{label}.action_count must be a non-negative integer.")
    if bundle.get("exists") is True:
        if not _is_non_negative_int(bundle.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for existing files.")
        if not _is_sha256(bundle.get("sha256")):
            target.errors.append(f"{label}.sha256 must be a SHA-256 hex string for existing files.")

def _validate_action_ledger_entry(entry: Any, target: ValidationTarget, label: str, latest_index: int) -> dict[str, int]:
    counts = {"occurrence_count": 0, "open": 0, "new": 0, "recurring": 0, "resolved": 0}
    if not isinstance(entry, dict):
        target.errors.append(f"{label} must be an object.")
        return counts
    for field_name in ("routing_key", "action_fingerprint", "id", "priority", "artifact", "summary", "status", "first_seen_path", "last_seen_path"):
        if not isinstance(entry.get(field_name), str) or not entry.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    for field_name in ("first_seen_path", "last_seen_path"):
        if isinstance(entry.get(field_name), str):
            _warn_absolute_public_path(target, f"{label}.{field_name}", entry.get(field_name))
    if not _is_sha256(entry.get("action_fingerprint")):
        target.errors.append(f"{label}.action_fingerprint must be a SHA-256 hex string.")
    expected_routing_key = f"{entry.get('artifact')}:{entry.get('id')}:{str(entry.get('action_fingerprint') or '')[:12]}"
    if isinstance(entry.get("routing_key"), str) and entry.get("routing_key") != expected_routing_key:
        target.errors.append(f"{label}.routing_key expected {expected_routing_key!r}, got {entry.get('routing_key')!r}.")
    if entry.get("priority") not in {"critical", "high", "medium", "low"}:
        target.errors.append(f"{label}.priority must be critical, high, medium, or low.")
    if entry.get("status") not in {"new", "recurring", "open", "resolved"}:
        target.errors.append(f"{label}.status must be new, recurring, open, or resolved.")
    if not isinstance(entry.get("open"), bool):
        target.errors.append(f"{label}.open must be a boolean.")
    if not isinstance(entry.get("evidence"), dict):
        target.errors.append(f"{label}.evidence must be an object.")
    occurrences = entry.get("occurrences")
    if not isinstance(occurrences, list):
        target.errors.append(f"{label}.occurrences must be a list.")
        occurrences = []
    for index, occurrence in enumerate(occurrences):
        _validate_action_ledger_occurrence(occurrence, target, f"{label}.occurrences[{index}]")
    bundle_indexes = entry.get("bundle_indexes")
    if not isinstance(bundle_indexes, list) or not all(_is_non_negative_int(item) for item in bundle_indexes):
        target.errors.append(f"{label}.bundle_indexes must be a list of non-negative integers.")
        bundle_indexes = []
    else:
        sorted_indexes = sorted(set(bundle_indexes))
        if bundle_indexes != sorted_indexes:
            target.errors.append(f"{label}.bundle_indexes must be sorted and unique.")
    occurrence_indexes = sorted(
        {occurrence.get("bundle_index") for occurrence in occurrences if isinstance(occurrence, dict) and _is_non_negative_int(occurrence.get("bundle_index"))}
    )
    if bundle_indexes and occurrence_indexes and bundle_indexes != occurrence_indexes:
        target.errors.append(f"{label}.bundle_indexes must match occurrence bundle indexes.")
    occurrence_count = len(occurrences)
    counts["occurrence_count"] = occurrence_count
    if entry.get("occurrence_count") != occurrence_count:
        target.errors.append(f"{label}.occurrence_count expected {occurrence_count}, got {entry.get('occurrence_count')!r}.")
    if bundle_indexes:
        first_seen = bundle_indexes[0]
        last_seen = bundle_indexes[-1]
        if entry.get("first_seen_index") != first_seen:
            target.errors.append(f"{label}.first_seen_index expected {first_seen}, got {entry.get('first_seen_index')!r}.")
        if entry.get("last_seen_index") != last_seen:
            target.errors.append(f"{label}.last_seen_index expected {last_seen}, got {entry.get('last_seen_index')!r}.")
        open_in_latest = latest_index in bundle_indexes
        expected_status = "new" if open_in_latest and first_seen == latest_index else "recurring" if open_in_latest and len(bundle_indexes) > 1 else "open" if open_in_latest else "resolved"
        if entry.get("open") != open_in_latest:
            target.errors.append(f"{label}.open expected {open_in_latest}, got {entry.get('open')!r}.")
        if entry.get("status") != expected_status:
            target.errors.append(f"{label}.status expected {expected_status!r}, got {entry.get('status')!r}.")
        counts["open"] = 1 if open_in_latest else 0
        counts["new"] = 1 if expected_status == "new" else 0
        counts["recurring"] = 1 if expected_status == "recurring" else 0
        counts["resolved"] = 1 if expected_status == "resolved" else 0
    return counts

def _validate_action_ledger_occurrence(occurrence: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(occurrence, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if not _is_non_negative_int(occurrence.get("bundle_index")):
        target.errors.append(f"{label}.bundle_index must be a non-negative integer.")
    for field_name in ("bundle_path", "summary", "priority", "artifact"):
        if not isinstance(occurrence.get(field_name), str):
            target.errors.append(f"{label}.{field_name} must be a string.")
    if isinstance(occurrence.get("bundle_path"), str):
        _warn_absolute_public_path(target, f"{label}.bundle_path", occurrence.get("bundle_path"))
    if occurrence.get("priority") not in {"critical", "high", "medium", "low"}:
        target.errors.append(f"{label}.priority must be critical, high, medium, or low.")

def _validate_action_ledger_count_rows(value: Any, expected: dict[str, int], target: ValidationTarget, label: str) -> None:
    counts = _count_rows(value)
    if counts is None:
        target.errors.append(f"{label} must be a list of {{id, count}} objects.")
    elif counts != expected:
        target.errors.append(f"{label} expected {expected!r}, got {counts!r}.")

def _validate_action_ledger_bundle_action_counts(value: Any, bundles: list[Any], target: ValidationTarget) -> None:
    label = "action_ledger.metrics.bundle_action_counts"
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list.")
        return
    expected = [
        {"index": bundle.get("index"), "path": bundle.get("path"), "action_count": bundle.get("action_count")}
        for bundle in bundles
        if isinstance(bundle, dict)
    ]
    for index, row in enumerate(value):
        if isinstance(row, dict) and isinstance(row.get("path"), str):
            _warn_absolute_public_path(target, f"{label}[{index}].path", row.get("path"))
    if value != expected:
        target.errors.append(f"{label} must match action_ledger.bundles action counts.")

def _validate_action_ledger_bundle_linkage(
    bundles: list[Any],
    entries: list[Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    expected_records: Counter[tuple[Any, ...]] = Counter()
    for index, bundle_record in enumerate(bundles):
        if not isinstance(bundle_record, dict):
            continue
        bundle_path = _resolve_action_ledger_bundle_path(bundle_record.get("path"), source_path)
        if bundle_path is None or not bundle_path.exists():
            target.errors.append(f"action_ledger.bundles[{index}].path must resolve to an existing evidence bundle.")
            continue
        if _path_has_symlink_component(bundle_path, include_leaf=True) or not bundle_path.is_file():
            target.errors.append(f"action_ledger.bundles[{index}].path must resolve to a regular non-symlink evidence bundle.")
            continue
        expected_size = bundle_record.get("size_bytes")
        if _is_non_negative_int(expected_size) and bundle_path.stat().st_size != expected_size:
            target.errors.append(f"action_ledger.bundles[{index}].size_bytes does not match current file size.")
        expected_sha = bundle_record.get("sha256")
        if _is_sha256(expected_sha) and _sha256(bundle_path) != expected_sha:
            target.errors.append(f"action_ledger.bundles[{index}].sha256 does not match current file contents.")
        try:
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            target.errors.append(f"action_ledger.bundles[{index}].path is not valid UTF-8: {exc}")
            continue
        except json.JSONDecodeError as exc:
            target.errors.append(f"action_ledger.bundles[{index}].path contains invalid JSON: {exc}")
            continue
        if not isinstance(bundle, dict):
            target.errors.append(f"action_ledger.bundles[{index}].path must contain a JSON object.")
            continue
        _validate_evidence_bundle(bundle, target, bundle_path)
        decision = bundle.get("decision") if isinstance(bundle.get("decision"), dict) else {}
        actions = decision.get("next_actions") if isinstance(decision.get("next_actions"), list) else []
        expected_action_count = len([action for action in actions if isinstance(action, dict)])
        if bundle_record.get("action_count") != expected_action_count:
            target.errors.append(
                f"action_ledger.bundles[{index}].action_count expected {expected_action_count} from source bundle, "
                f"got {bundle_record.get('action_count')!r}."
            )
        bundle_index = bundle_record.get("index")
        if not _is_non_negative_int(bundle_index):
            continue
        ledger_bundle_path = bundle_record.get("path") if isinstance(bundle_record.get("path"), str) else ""
        for action in actions:
            if isinstance(action, dict):
                expected_records[_action_ledger_expected_action_record(action, bundle_index, ledger_bundle_path)] += 1

    actual_records = _action_ledger_actual_action_records(entries)
    missing = expected_records - actual_records
    extra = actual_records - expected_records
    if missing:
        target.errors.append(f"action_ledger.entries missing {sum(missing.values())} next_action occurrence(s) from source bundles.")
    if extra:
        target.errors.append(f"action_ledger.entries include {sum(extra.values())} next_action occurrence(s) not present in source bundles.")

def _action_ledger_expected_action_record(action: dict[str, Any], bundle_index: int, bundle_path: str) -> tuple[Any, ...]:
    evidence = action.get("evidence") if isinstance(action.get("evidence"), dict) else {}
    action_id = str(action.get("id") or "unknown_action")
    priority = str(action.get("priority") or "medium")
    artifact = str(action.get("artifact") or "unknown_artifact")
    summary = str(action.get("summary") or "")
    fingerprint = action.get("action_fingerprint")
    if not _is_sha256(fingerprint):
        fingerprint = _evidence_bundle_action_fingerprint(
            {"id": action_id, "priority": priority, "artifact": artifact, "evidence": evidence}
        )
    routing_key = action.get("routing_key")
    if not isinstance(routing_key, str) or not routing_key:
        routing_key = f"{artifact}:{action_id}:{fingerprint[:12]}"
    return (
        bundle_index,
        bundle_path,
        routing_key,
        fingerprint,
        action_id,
        priority,
        artifact,
        summary,
        _action_ledger_evidence_record(evidence),
        priority,
        artifact,
        summary,
    )

def _action_ledger_actual_action_records(entries: list[Any]) -> Counter[tuple[Any, ...]]:
    records: Counter[tuple[Any, ...]] = Counter()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        occurrences = entry.get("occurrences") if isinstance(entry.get("occurrences"), list) else []
        for occurrence in occurrences:
            if not isinstance(occurrence, dict) or not _is_non_negative_int(occurrence.get("bundle_index")):
                continue
            records[
                (
                    occurrence.get("bundle_index"),
                    occurrence.get("bundle_path"),
                    entry.get("routing_key"),
                    entry.get("action_fingerprint"),
                    entry.get("id"),
                    entry.get("priority"),
                    entry.get("artifact"),
                    entry.get("summary"),
                    _action_ledger_evidence_record(entry.get("evidence")),
                    occurrence.get("priority"),
                    occurrence.get("artifact"),
                    occurrence.get("summary"),
                )
            ] += 1
    return records

def _action_ledger_evidence_record(value: Any) -> str:
    evidence = value if isinstance(value, dict) else {}
    return json.dumps(evidence, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)

def _resolve_action_ledger_bundle_path(value: Any, source_path: Path) -> Path | None:
    return _resolve_gate_source_path(value, source_path)

def _resolve_gate_source_path(value: Any, source_path: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    if "://" in value:
        return None
    if value.startswith("<redacted:") and value.endswith(">"):
        basename = value.removeprefix("<redacted:").removesuffix(">")
        if basename in {"", ".", ".."} or Path(basename).name != basename:
            return None
        return source_path.parent / basename
    path = Path(value)
    if path.is_absolute():
        return path
    return source_path.parent / path

def _validate_repair_queue(queue: dict[str, Any], target: ValidationTarget, source_path: Path) -> None:
    _require_equal(queue, "schema_version", REPAIR_QUEUE_SCHEMA_VERSION, target)
    if not isinstance(queue.get("runs_dir"), str) or not queue.get("runs_dir"):
        target.errors.append("repair_queue.runs_dir must be a non-empty string.")
    _warn_absolute_public_path(target, "repair_queue.runs_dir", queue.get("runs_dir"))
    if not isinstance(queue.get("passed"), bool):
        target.errors.append("repair_queue.passed must be a boolean.")
    if not isinstance(queue.get("only_critical"), bool):
        target.errors.append("repair_queue.only_critical must be a boolean.")
    items = queue.get("items")
    if not isinstance(items, list):
        target.errors.append("repair_queue.items must be a list.")
        items = []
    if queue.get("item_count") != len(items):
        target.errors.append(f"repair_queue.item_count expected {len(items)}, got {queue.get('item_count')!r}.")

    seen_ids: set[str] = set()
    totals: dict[str, Any] = {
        "critical_item_count": 0,
        "scenario_ids": set(),
        "task_families": set(),
        "priority_counts": {},
        "rule_counts": {},
        "critical_rule_counts": {},
        "task_completion_status_counts": {},
    }
    for index, item in enumerate(items):
        _validate_repair_item(item, target, f"repair_queue.items[{index}]", seen_ids, totals, source_path)
    _validate_repair_queue_metrics(queue.get("metrics"), target, totals, len(items))
    if "notes" in queue and not _is_string_list(queue.get("notes")):
        target.errors.append("repair_queue.notes must be a list of strings when present.")
    target.details.update(
        {
            "item_count": len(items),
            "critical_item_count": totals["critical_item_count"],
            "scenario_count": len(totals["scenario_ids"]),
        }
    )

def _validate_repair_item(
    item: Any,
    target: ValidationTarget,
    label: str,
    seen_ids: set[str],
    totals: dict[str, Any],
    source_path: Path,
) -> None:
    if not isinstance(item, dict):
        target.errors.append(f"{label} must be an object.")
        return
    _require_equal(item, "schema_version", REPAIR_ITEM_SCHEMA_VERSION, target, prefix=f"{label}.")
    for field_name in (
        "repair_item_id",
        "run_id",
        "scenario_id",
        "scenario_title",
        "task_family",
        "priority",
        "rule_id",
        "rule_name",
        "summary",
        "suggested_action",
    ):
        if not isinstance(item.get(field_name), str) or not item.get(field_name):
            target.errors.append(f"{label}.{field_name} must be a non-empty string.")
    item_id = item.get("repair_item_id")
    if isinstance(item_id, str) and item_id:
        if item_id in seen_ids:
            target.errors.append(f"{label}.repair_item_id duplicates {item_id!r}.")
        seen_ids.add(item_id)
    if item.get("priority") not in {"critical", "high", "medium", "low"}:
        target.errors.append(f"{label}.priority must be critical, high, medium, or low.")
    if not isinstance(item.get("critical"), bool):
        target.errors.append(f"{label}.critical must be a boolean.")
    if not _is_non_negative_int(item.get("penalty")):
        target.errors.append(f"{label}.penalty must be a non-negative integer.")
    if not _is_int_between(item.get("score"), 0, 100):
        target.errors.append(f"{label}.score must be an integer from 0 to 100.")
    if "pass_threshold" in item and item.get("pass_threshold") is not None and not _is_int_between(item.get("pass_threshold"), 0, 100):
        target.errors.append(f"{label}.pass_threshold must be null or an integer from 0 to 100.")
    if not isinstance(item.get("task_completion_passed"), bool):
        target.errors.append(f"{label}.task_completion_passed must be a boolean.")
    if not _is_string_list(item.get("evidence")):
        target.errors.append(f"{label}.evidence must be a list of strings.")
    _validate_evidence_refs(item.get("evidence_refs"), target, f"{label}.evidence_refs")
    _validate_repair_evidence_snippets(item.get("evidence_snippets"), target, f"{label}.evidence_snippets")
    _validate_repair_source_artifacts(item.get("source_artifacts"), target, f"{label}.source_artifacts")
    _validate_repair_source_artifact_fingerprints(
        item.get("source_artifact_fingerprints"),
        target,
        f"{label}.source_artifact_fingerprints",
        source_path,
    )
    _validate_repair_replay(item.get("replay"), target, f"{label}.replay")

    if item.get("critical") is True:
        totals["critical_item_count"] += 1
    _add_total(totals["scenario_ids"], item.get("scenario_id"))
    _add_total(totals["task_families"], item.get("task_family"))
    _increment_count(totals["priority_counts"], item.get("priority"))
    _increment_count(totals["rule_counts"], item.get("rule_id"))
    if item.get("critical") is True:
        _increment_count(totals["critical_rule_counts"], item.get("rule_id"))
    _increment_count(totals["task_completion_status_counts"], item.get("task_completion_status"))

def _validate_repair_evidence_snippets(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, list):
        target.errors.append(f"{label} must be a list.")
        return
    for index, snippet in enumerate(value):
        snippet_label = f"{label}[{index}]"
        if not isinstance(snippet, dict):
            target.errors.append(f"{snippet_label} must be an object.")
            continue
        if snippet.get("target") not in {"event", "final_answer", "episode", "state_snapshot"}:
            target.errors.append(f"{snippet_label}.target must be event, final_answer, episode, or state_snapshot.")
        if not isinstance(snippet.get("reason"), str):
            target.errors.append(f"{snippet_label}.reason must be a string.")
        if not isinstance(snippet.get("text"), str):
            target.errors.append(f"{snippet_label}.text must be a string.")
        elif len(snippet["text"]) > 600:
            target.errors.append(f"{snippet_label}.text must be at most 600 characters.")
        if snippet.get("target") == "event":
            if not _is_non_negative_int(snippet.get("event_index")):
                target.errors.append(f"{snippet_label}.event_index must be a non-negative integer.")
            for field_name in ("event_type", "tool_name", "status"):
                if not isinstance(snippet.get(field_name), str):
                    target.errors.append(f"{snippet_label}.{field_name} must be a string.")

def _validate_repair_source_artifacts(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    for artifact_name in ("run_dir", "normalized_trace", "scorecard", "report"):
        if not isinstance(value.get(artifact_name), str) or not value.get(artifact_name):
            target.errors.append(f"{label}.{artifact_name} must be a non-empty string.")
    for artifact_name, artifact_path in value.items():
        if not isinstance(artifact_name, str) or not artifact_name:
            target.errors.append(f"{label} keys must be non-empty strings.")
        if not isinstance(artifact_path, str) or not artifact_path:
            target.errors.append(f"{label}.{artifact_name} must be a non-empty string.")
        else:
            _warn_absolute_public_path(target, f"{label}.{artifact_name}", artifact_path)

def _validate_repair_source_artifact_fingerprints(value: Any, target: ValidationTarget, label: str, source_path: Path) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    for artifact_name in ("normalized_trace", "scorecard", "report"):
        if artifact_name not in value:
            target.errors.append(f"{label}.{artifact_name} is required.")
    for artifact_name, record in value.items():
        if not isinstance(artifact_name, str) or not artifact_name:
            target.errors.append(f"{label} keys must be non-empty strings.")
            continue
        _validate_repair_source_file_ref(record, target, f"{label}.{artifact_name}", source_path)

def _validate_repair_source_file_ref(record: Any, target: ValidationTarget, label: str, source_path: Path) -> None:
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if record.get("kind") != "file":
        target.errors.append(f"{label}.kind must be file.")
    if record.get("exists") is not True:
        target.errors.append(f"{label}.exists must be true.")
    if not isinstance(record.get("path"), str) or not record.get("path"):
        target.errors.append(f"{label}.path must be a non-empty string.")
        return
    if _looks_absolute(record["path"]):
        target.errors.append(f"{label}.path must be relative to the repair queue artifact.")
        return
    if not _is_non_negative_int(record.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    if not _is_lowercase_sha256(record.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a lowercase SHA-256 hex string.")
    file_path = source_path.parent / record["path"]
    if file_path.is_symlink():
        target.errors.append(f"{label}.path must not resolve to a symlink.")
        return
    if not file_path.exists() or not file_path.is_file():
        target.errors.append(f"{label}.path does not resolve to an existing file.")
        return
    if _is_non_negative_int(record.get("size_bytes")) and file_path.stat().st_size != record.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    if _is_lowercase_sha256(record.get("sha256")) and _sha256(file_path) != record.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the current file.")

def _validate_repair_replay(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object.")
        return
    if not isinstance(value.get("available"), bool):
        target.errors.append(f"{label}.available must be a boolean.")
    if value.get("self_contained") is not None and not isinstance(value.get("self_contained"), bool):
        target.errors.append(f"{label}.self_contained must be a boolean or null.")
    if not isinstance(value.get("command"), str):
        target.errors.append(f"{label}.command must be a string.")
    if not isinstance(value.get("argv"), list) or not all(isinstance(part, str) for part in value.get("argv", [])):
        target.errors.append(f"{label}.argv must be a list of strings.")
    _warn_replay_metadata_public_paths(value, target, label)

def _validate_repair_queue_metrics(metrics: Any, target: ValidationTarget, totals: dict[str, Any], item_count: int) -> None:
    if not isinstance(metrics, dict):
        target.errors.append("repair_queue.metrics must be an object.")
        return
    expected = {
        "item_count": item_count,
        "critical_item_count": totals["critical_item_count"],
        "scenario_count": len(totals["scenario_ids"]),
        "task_family_count": len(totals["task_families"]),
        "scenarios": sorted(totals["scenario_ids"]),
        "task_families": sorted(totals["task_families"]),
        "priority_counts": totals["priority_counts"],
        "rule_counts": totals["rule_counts"],
        "critical_rule_counts": totals["critical_rule_counts"],
        "task_completion_status_counts": totals["task_completion_status_counts"],
    }
    for field_name in ("item_count", "critical_item_count", "scenario_count", "task_family_count"):
        if metrics.get(field_name) != expected[field_name]:
            target.errors.append(f"repair_queue.metrics.{field_name} expected {expected[field_name]!r}, got {metrics.get(field_name)!r}.")
    for field_name in ("scenarios", "task_families"):
        if metrics.get(field_name) != expected[field_name]:
            target.errors.append(f"repair_queue.metrics.{field_name} expected {expected[field_name]!r}, got {metrics.get(field_name)!r}.")
    for field_name in ("priority_counts", "rule_counts", "critical_rule_counts", "task_completion_status_counts"):
        actual_counts = _count_rows(metrics.get(field_name))
        if actual_counts != expected[field_name]:
            target.errors.append(f"repair_queue.metrics.{field_name} does not match repair items.")

def _add_total(target_set: set[str], value: Any) -> None:
    if isinstance(value, str) and value:
        target_set.add(value)

def _increment_count(target_counts: dict[str, int], value: Any) -> None:
    if isinstance(value, str) and value:
        target_counts[value] = target_counts.get(value, 0) + 1

def _count_value(target_counts: dict[str, int], key: str) -> int:
    value = target_counts.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

def _list_present(value: Any) -> bool:
    return isinstance(value, list) and bool(value)

def _validate_replay_bundle(manifest: dict[str, Any], bundle_dir: Path, target: ValidationTarget) -> None:
    _require_equal(manifest, "schema_version", REPLAY_BUNDLE_SCHEMA_VERSION, target, prefix="replay_bundle.")
    _warn_absolute_public_path(target, "replay_bundle.source_lineage", manifest.get("source_lineage"))
    lineage_name = manifest.get("lineage")
    if not isinstance(lineage_name, str) or not lineage_name:
        target.errors.append("replay_bundle.lineage must be a non-empty string.")
        lineage_name = "artifact_lineage.json"
    lineage_path = bundle_dir / lineage_name
    lineage = _read_object(lineage_path, target, "artifact_lineage.json")
    if lineage is not None:
        _validate_replay_bundle_lineage(lineage, manifest, bundle_dir, target)

    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        target.errors.append("replay_bundle.inputs must be a list.")
        inputs = []
    if manifest.get("input_count") != len(inputs):
        target.errors.append(f"replay_bundle.input_count expected {len(inputs)}, got {manifest.get('input_count')!r}.")
    input_records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(inputs):
        label = f"replay_bundle.inputs[{index}]"
        if not isinstance(record, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name:
            target.errors.append(f"{label}.name must be a non-empty string.")
            continue
        input_records[name] = record
        _validate_replay_bundle_input_record(record, bundle_dir, target, label)
    for required_name in ("scenario", "source_trace"):
        if required_name not in input_records:
            target.errors.append(f"replay_bundle.inputs missing {required_name}.")

    replay = manifest.get("replay")
    if not isinstance(replay, dict):
        target.errors.append("replay_bundle.replay must be an object.")
        replay = {}
    if replay.get("self_contained") is not True:
        target.errors.append("replay_bundle.replay.self_contained must be true.")
    argv = replay.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) and item for item in argv):
        target.errors.append("replay_bundle.replay.argv must be a list of non-empty strings.")
        argv = []
    else:
        if argv[:4] != ["python", "-m", "flightrecorder", "run"]:
            target.errors.append("replay_bundle.replay.argv must start with python -m flightrecorder run.")
        _warn_replay_metadata_public_paths(replay, target, "replay_bundle.replay")
        _validate_replay_bundle_argv_inputs(argv, input_records, target)
    if not isinstance(replay.get("command"), str) or not replay.get("command"):
        target.errors.append("replay_bundle.replay.command must be a non-empty string.")
    notes = manifest.get("notes")
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        target.errors.append("replay_bundle.notes must be a list of strings.")
    target.details.update(
        {
            "input_count": len(input_records),
            "lineage": lineage_name,
            "self_contained": replay.get("self_contained") is True,
        }
    )

def _validate_replay_bundle_lineage(
    lineage: dict[str, Any],
    manifest: dict[str, Any],
    bundle_dir: Path,
    target: ValidationTarget,
) -> None:
    _require_equal(lineage, "schema_version", LINEAGE_SCHEMA_VERSION, target, prefix="artifact_lineage.")
    portable = lineage.get("portable_replay_bundle")
    if not isinstance(portable, dict):
        target.errors.append("artifact_lineage.portable_replay_bundle must be an object.")
        portable = {}
    _require_equal(portable, "schema_version", REPLAY_BUNDLE_SCHEMA_VERSION, target, prefix="artifact_lineage.portable_replay_bundle.")
    _warn_absolute_public_path(
        target,
        "artifact_lineage.portable_replay_bundle.source_lineage",
        portable.get("source_lineage"),
    )
    if portable.get("input_count") != manifest.get("input_count"):
        target.errors.append("artifact_lineage.portable_replay_bundle.input_count must match replay_bundle.input_count.")
    replay = lineage.get("replay")
    manifest_replay = manifest.get("replay") if isinstance(manifest.get("replay"), dict) else {}
    if not isinstance(replay, dict):
        target.errors.append("artifact_lineage.replay must be an object.")
        return
    if replay.get("self_contained") is not True:
        target.errors.append("artifact_lineage.replay.self_contained must be true for replay bundles.")
    if replay.get("argv") != manifest_replay.get("argv"):
        target.errors.append("artifact_lineage.replay.argv must match replay_bundle.replay.argv.")
    if replay.get("command") != manifest_replay.get("command"):
        target.errors.append("artifact_lineage.replay.command must match replay_bundle.replay.command.")
    _warn_replay_metadata_public_paths(replay, target, "artifact_lineage.replay")

    inputs = _lineage_records(lineage.get("inputs"), target, "artifact_lineage.inputs")
    fingerprints = replay.get("input_fingerprints") if isinstance(replay.get("input_fingerprints"), dict) else {}
    if not isinstance(fingerprints, dict):
        target.errors.append("artifact_lineage.replay.input_fingerprints must be an object.")
        fingerprints = {}
    for input_record in manifest.get("inputs", []) if isinstance(manifest.get("inputs"), list) else []:
        if not isinstance(input_record, dict):
            continue
        name = input_record.get("name")
        if not isinstance(name, str) or not name:
            continue
        fingerprint = fingerprints.get(name)
        if not isinstance(fingerprint, dict):
            target.errors.append(f"artifact_lineage.replay.input_fingerprints missing {name}.")
            continue
        for field_name in ("path", "sha256", "size_bytes"):
            if fingerprint.get(field_name) != input_record.get(field_name):
                target.errors.append(f"artifact_lineage.replay.input_fingerprints.{name}.{field_name} must match replay_bundle input.")
        if fingerprint.get("exists") is not True:
            target.errors.append(f"artifact_lineage.replay.input_fingerprints.{name}.exists must be true.")
        lineage_input = inputs.get(name)
        if lineage_input is not None:
            for field_name in ("path", "sha256", "size_bytes"):
                if lineage_input.get(field_name) != input_record.get(field_name):
                    target.errors.append(f"artifact_lineage.inputs.{name}.{field_name} must match replay_bundle input.")
    _validate_replay_bundle_copied_lineage_inputs(inputs, bundle_dir, target)

def _validate_replay_bundle_input_record(record: dict[str, Any], bundle_dir: Path, target: ValidationTarget, label: str) -> None:
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        target.errors.append(f"{label}.path must be a non-empty string.")
        return
    if not _is_safe_relative_path(path_value):
        target.errors.append(f"{label}.path must be relative to the replay bundle directory.")
        return
    file_path = bundle_dir / path_value
    if _path_has_symlink_component(file_path, include_leaf=True):
        target.errors.append(f"{label}.path must resolve to a regular non-symlink file.")
        return
    if not file_path.exists() or not file_path.is_file():
        target.errors.append(f"{label}.path does not resolve to a bundled file.")
        return
    if not _is_non_negative_int(record.get("size_bytes")):
        target.errors.append(f"{label}.size_bytes must be a non-negative integer.")
    elif file_path.stat().st_size != record.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the bundled file.")
    if not _is_sha256(record.get("sha256")):
        target.errors.append(f"{label}.sha256 must be a SHA-256 hex string.")
    elif _sha256(file_path) != record.get("sha256"):
        target.errors.append(f"{label}.sha256 does not match the bundled file.")
    if "source_path" in record:
        if not isinstance(record.get("source_path"), str):
            target.errors.append(f"{label}.source_path must be a string when present.")
        else:
            _warn_absolute_public_path(target, f"{label}.source_path", record.get("source_path"))

def _validate_replay_bundle_argv_inputs(argv: list[str], inputs: dict[str, dict[str, Any]], target: ValidationTarget) -> None:
    flag_to_input = {"--scenario": "scenario", "--trace": "source_trace", "--state": "source_state_snapshot"}
    for flag, input_name in flag_to_input.items():
        if flag not in argv:
            if flag in {"--scenario", "--trace"}:
                target.errors.append(f"replay_bundle.replay.argv missing {flag}.")
            continue
        index = argv.index(flag)
        if index + 1 >= len(argv):
            target.errors.append(f"replay_bundle.replay.argv missing value for {flag}.")
            continue
        expected = inputs.get(input_name, {}).get("path")
        if expected is not None and argv[index + 1] != expected:
            target.errors.append(f"replay_bundle.replay.argv value for {flag} must match replay_bundle.inputs.{input_name}.path.")

def _validate_replay_bundle_copied_lineage_inputs(inputs: dict[str, dict[str, Any]], bundle_dir: Path, target: ValidationTarget) -> None:
    for name in ("scenario", "source_trace", "source_state_snapshot"):
        record = inputs.get(name)
        if record is None:
            if name != "source_state_snapshot":
                target.errors.append(f"artifact_lineage.inputs missing {name}.")
            continue
        path_value = record.get("path")
        if not isinstance(path_value, str) or not path_value:
            target.errors.append(f"artifact_lineage.inputs.{name}.path must be a non-empty string.")
            continue
        if not _is_safe_relative_path(path_value):
            target.errors.append(f"artifact_lineage.inputs.{name}.path must be relative to the replay bundle directory.")
            continue
        file_path = bundle_dir / path_value
        if _path_has_symlink_component(file_path, include_leaf=True):
            target.errors.append(f"artifact_lineage.inputs.{name}.path must resolve to a regular non-symlink file.")
            continue
        if not file_path.exists() or not file_path.is_file():
            target.errors.append(f"artifact_lineage.inputs.{name}.path does not resolve to a bundled file.")
            continue
        if record.get("exists") is not True:
            target.errors.append(f"artifact_lineage.inputs.{name}.exists must be true.")
        if not _is_non_negative_int(record.get("size_bytes")):
            target.errors.append(f"artifact_lineage.inputs.{name}.size_bytes must be a non-negative integer.")
        elif file_path.stat().st_size != record.get("size_bytes"):
            target.errors.append(f"artifact_lineage.inputs.{name}.size_bytes does not match the bundled file.")
        if not _is_sha256(record.get("sha256")):
            target.errors.append(f"artifact_lineage.inputs.{name}.sha256 must be a SHA-256 hex string.")
        elif _sha256(file_path) != record.get("sha256"):
            target.errors.append(f"artifact_lineage.inputs.{name}.sha256 does not match the bundled file.")

def _is_safe_relative_path(value: str) -> bool:
    path = Path(value)
    return not path.is_absolute() and not _is_windows_absolute(value) and ".." not in path.parts

def _validate_evidence_bundle_checks(checks: list[Any], target: ValidationTarget) -> int:
    failed_checks = 0
    for index, check in enumerate(checks):
        label = f"evidence_bundle.checks[{index}]"
        if not isinstance(check, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        if not isinstance(check.get("id"), str) or not check.get("id"):
            target.errors.append(f"{label}.id must be a non-empty string.")
        if not isinstance(check.get("passed"), bool):
            target.errors.append(f"{label}.passed must be a boolean.")
        elif not check["passed"]:
            failed_checks += 1
        for field_name in ("actual", "expected", "scope"):
            if not isinstance(check.get(field_name), dict):
                target.errors.append(f"{label}.{field_name} must be an object.")
        if not isinstance(check.get("summary"), str) or not check.get("summary"):
            target.errors.append(f"{label}.summary must be a non-empty string.")
    return failed_checks

def _evidence_bundle_blocking_check_rows(checks: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(check.get("id") or "unknown"),
            "summary": str(check.get("summary") or ""),
            "scope": check.get("scope") if isinstance(check.get("scope"), dict) else {},
        }
        for check in checks
        if isinstance(check, dict) and check.get("passed") is False
    ]

def _evidence_bundle_blocking_gate_rows(metrics: dict[str, Any]) -> list[dict[str, str]]:
    gates = metrics.get("gates") if isinstance(metrics.get("gates"), list) else []
    return [
        {"id": str(gate.get("id") or "unknown"), "path": str(gate.get("path") or "")}
        for gate in gates
        if isinstance(gate, dict) and gate.get("passed") is False
    ]

def _validate_evidence_bundle_decision(
    decision: Any,
    expected_readiness: str,
    failed_checks: int,
    blocking_check_rows: list[dict[str, Any]],
    blocking_gate_rows: list[dict[str, str]],
    artifacts: dict[str, Any],
    metrics: dict[str, Any],
    target: ValidationTarget,
) -> None:
    if not isinstance(decision, dict):
        target.errors.append("evidence_bundle.decision must be an object when present.")
        return
    if decision.get("readiness") != expected_readiness:
        target.errors.append(
            f"evidence_bundle.decision.readiness expected {expected_readiness!r}, got {decision.get('readiness')!r}."
        )
    expected_recommendation = "promote_handoff" if expected_readiness == "ready" else "block_handoff"
    if decision.get("recommendation") != expected_recommendation:
        target.errors.append(
            "evidence_bundle.decision.recommendation expected "
            f"{expected_recommendation!r}, got {decision.get('recommendation')!r}."
        )
    summary = decision.get("summary")
    if not isinstance(summary, str) or not summary:
        target.errors.append("evidence_bundle.decision.summary must be a non-empty string.")
    elif summary != _build_evidence_bundle_decision_text(expected_readiness, blocking_check_rows):
        target.errors.append("evidence_bundle.decision.summary must match bundle readiness and failed checks.")
    if decision.get("blocking_check_count") != failed_checks:
        target.errors.append(
            f"evidence_bundle.decision.blocking_check_count expected {failed_checks}, got {decision.get('blocking_check_count')!r}."
        )
    blocking_checks = decision.get("blocking_checks")
    if not isinstance(blocking_checks, list):
        target.errors.append("evidence_bundle.decision.blocking_checks must be a list.")
    else:
        if len(blocking_checks) != failed_checks:
            target.errors.append(
                f"evidence_bundle.decision.blocking_checks expected {failed_checks} entries, got {len(blocking_checks)}."
            )
        for index, check in enumerate(blocking_checks):
            label = f"evidence_bundle.decision.blocking_checks[{index}]"
            if not isinstance(check, dict):
                target.errors.append(f"{label} must be an object.")
                continue
            if not isinstance(check.get("id"), str) or not check.get("id"):
                target.errors.append(f"{label}.id must be a non-empty string.")
            if not isinstance(check.get("summary"), str):
                target.errors.append(f"{label}.summary must be a string.")
            if not isinstance(check.get("scope"), dict):
                target.errors.append(f"{label}.scope must be an object.")
        if blocking_checks != blocking_check_rows:
            target.errors.append("evidence_bundle.decision.blocking_checks must match failed evidence_bundle.checks.")
    blocking_gates = decision.get("blocking_gates")
    if not isinstance(blocking_gates, list):
        target.errors.append("evidence_bundle.decision.blocking_gates must be a list.")
    else:
        for index, gate in enumerate(blocking_gates):
            label = f"evidence_bundle.decision.blocking_gates[{index}]"
            if not isinstance(gate, dict):
                target.errors.append(f"{label} must be an object.")
                continue
            for field_name in ("id", "path"):
                if not isinstance(gate.get(field_name), str) or not gate.get(field_name):
                    target.errors.append(f"{label}.{field_name} must be a non-empty string.")
            _warn_absolute_public_path(target, f"{label}.path", gate.get("path"))
        if blocking_gates != blocking_gate_rows:
            target.errors.append("evidence_bundle.decision.blocking_gates must match failed evidence_bundle.metrics.gates.")
    expected_next_actions = _build_evidence_bundle_next_actions(blocking_check_rows, blocking_gate_rows, metrics)
    next_actions = decision.get("next_actions")
    if not isinstance(next_actions, list):
        target.errors.append("evidence_bundle.decision.next_actions must be a list.")
        next_actions = []
    else:
        valid_action_artifacts = set(artifacts) | set(metrics) | {"evidence_bundle"}
        for index, action in enumerate(next_actions):
            label = f"evidence_bundle.decision.next_actions[{index}]"
            if not isinstance(action, dict):
                target.errors.append(f"{label} must be an object.")
                continue
            for field_name in ("id", "priority", "artifact", "summary"):
                if not isinstance(action.get(field_name), str) or not action.get(field_name):
                    target.errors.append(f"{label}.{field_name} must be a non-empty string.")
            artifact_name = action.get("artifact")
            if isinstance(artifact_name, str) and artifact_name and artifact_name not in valid_action_artifacts:
                target.errors.append(
                    f"{label}.artifact must reference an evidence_bundle.artifacts key, metrics key, or evidence_bundle."
                )
            if action.get("priority") not in {"critical", "high", "medium", "low"}:
                target.errors.append(f"{label}.priority must be critical, high, medium, or low.")
            if not isinstance(action.get("evidence"), dict):
                target.errors.append(f"{label}.evidence must be an object.")
            expected_fingerprint = _evidence_bundle_action_fingerprint(action)
            if not _is_sha256(action.get("action_fingerprint")):
                target.errors.append(f"{label}.action_fingerprint must be a SHA-256 hex string.")
            elif action.get("action_fingerprint") != expected_fingerprint:
                target.errors.append(f"{label}.action_fingerprint does not match the action payload.")
            expected_routing_key = f"{action.get('artifact')}:{action.get('id')}:{expected_fingerprint[:12]}"
            if not isinstance(action.get("routing_key"), str) or not action.get("routing_key"):
                target.errors.append(f"{label}.routing_key must be a non-empty string.")
            elif action.get("routing_key") != expected_routing_key:
                target.errors.append(f"{label}.routing_key expected {expected_routing_key!r}, got {action.get('routing_key')!r}.")
        if next_actions != expected_next_actions:
            target.errors.append("evidence_bundle.decision.next_actions must match bundle blockers and metrics.")
    if decision.get("next_action_count") != len(expected_next_actions):
        target.errors.append(
            f"evidence_bundle.decision.next_action_count expected {len(expected_next_actions)}, got {decision.get('next_action_count')!r}."
        )
    evidence_artifacts = decision.get("evidence_artifacts")
    if not isinstance(evidence_artifacts, list) or not all(isinstance(item, str) and item for item in evidence_artifacts):
        target.errors.append("evidence_bundle.decision.evidence_artifacts must be a list of non-empty strings.")
    elif sorted(evidence_artifacts) != sorted(artifacts):
        target.errors.append("evidence_bundle.decision.evidence_artifacts must match evidence_bundle.artifacts keys.")
    gates = metrics.get("gates") if isinstance(metrics.get("gates"), list) else []
    if decision.get("gate_count") != len(gates):
        target.errors.append(f"evidence_bundle.decision.gate_count expected {len(gates)}, got {decision.get('gate_count')!r}.")
    expected_passed_gates = sum(1 for gate in gates if isinstance(gate, dict) and gate.get("passed") is True)
    if decision.get("passed_gate_count") != expected_passed_gates:
        target.errors.append(
            f"evidence_bundle.decision.passed_gate_count expected {expected_passed_gates}, got {decision.get('passed_gate_count')!r}."
        )
    key_metrics = decision.get("key_metrics")
    if not isinstance(key_metrics, dict):
        target.errors.append("evidence_bundle.decision.key_metrics must be an object.")
    elif key_metrics != _build_evidence_bundle_decision_key_metrics(metrics):
        target.errors.append("evidence_bundle.decision.key_metrics must match evidence_bundle.metrics.")

def _validate_evidence_bundle_artifact_record(name: Any, record: Any, target: ValidationTarget, source_path: Path) -> None:
    if not isinstance(name, str) or not name:
        target.errors.append("evidence_bundle.artifacts keys must be non-empty strings.")
    label = f"evidence_bundle.artifacts.{name}"
    if not isinstance(record, dict):
        target.errors.append(f"{label} must be an object.")
        return
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        target.errors.append(f"{label}.path must be a non-empty string.")
    else:
        _warn_absolute_public_path(target, f"{label}.path", path_value)
    if not isinstance(record.get("exists"), bool):
        target.errors.append(f"{label}.exists must be a boolean.")
    if record.get("kind") not in {"file", "directory"}:
        target.errors.append(f"{label}.kind must be file or directory.")
    if record.get("kind") == "file" and record.get("exists") is True:
        if not _is_non_negative_int(record.get("size_bytes")):
            target.errors.append(f"{label}.size_bytes must be a non-negative integer for existing files.")
        sha = record.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or sha != sha.lower() or any(char not in "0123456789abcdef" for char in sha):
            target.errors.append(f"{label}.sha256 must be a lowercase 64-character hex digest for existing files.")
    if record.get("kind") == "directory" and record.get("exists") is True and not _is_non_negative_int(record.get("entry_count")):
        target.errors.append(f"{label}.entry_count must be a non-negative integer for existing directories.")
    if "schema_version" in record and record.get("schema_version") is not None and not isinstance(record.get("schema_version"), str):
        target.errors.append(f"{label}.schema_version must be a string or null.")
    if "passed" in record and record.get("passed") is not None and not isinstance(record.get("passed"), bool):
        target.errors.append(f"{label}.passed must be a boolean or null.")
    _validate_evidence_bundle_artifact_file_fingerprint(record, target, label, source_path)
    _validate_evidence_bundle_artifact_directory_fingerprint(record, target, label, source_path)

def _validate_evidence_bundle_artifact_file_fingerprint(
    record: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path,
) -> None:
    if record.get("kind") != "file":
        return
    current_path = _resolve_evidence_bundle_artifact_path(record.get("path"), source_path)
    if current_path is None:
        return
    if record.get("exists") is True:
        if not current_path.exists():
            target.errors.append(f"{label}.path must resolve to an existing file when exists is true.")
            return
        if _path_has_symlink_component(current_path, include_leaf=True) or not current_path.is_file():
            target.errors.append(f"{label}.path must resolve to a regular file when exists is true.")
            return
    elif not current_path.is_file():
        return
    if _is_non_negative_int(record.get("size_bytes")) and current_path.stat().st_size != record.get("size_bytes"):
        target.errors.append(f"{label}.size_bytes does not match the current file.")
    sha = record.get("sha256")
    if isinstance(sha, str) and len(sha) == 64 and _sha256(current_path) != sha:
        target.errors.append(f"{label}.sha256 does not match the current file.")

def _validate_evidence_bundle_artifact_directory_fingerprint(
    record: dict[str, Any],
    target: ValidationTarget,
    label: str,
    source_path: Path,
) -> None:
    if record.get("kind") != "directory":
        return
    path_value = record.get("path")
    if isinstance(path_value, str) and _looks_absolute(path_value):
        return
    current_path = _resolve_evidence_bundle_artifact_path(record.get("path"), source_path)
    if current_path is None:
        return
    if record.get("exists") is True:
        if not current_path.exists():
            target.errors.append(f"{label}.path must resolve to an existing directory when exists is true.")
            return
        if _path_has_symlink_component(current_path, include_leaf=True) or not current_path.is_dir():
            target.errors.append(f"{label}.path must resolve to a regular directory when exists is true.")
            return

def _evidence_bundle_action_fingerprint(action: dict[str, Any]) -> str:
    evidence = action.get("evidence") if isinstance(action.get("evidence"), dict) else {}
    payload = {
        "id": action.get("id"),
        "priority": action.get("priority"),
        "artifact": action.get("artifact"),
        "evidence": evidence,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _validate_evidence_bundle_harness_checks(metrics: dict[str, Any], checks: list[Any], target: ValidationTarget) -> None:
    harness = metrics.get("harness_handoff")
    if not isinstance(harness, dict):
        return
    runs = harness.get("runs")
    if not isinstance(runs, list):
        return
    invalid_run_suite_rows = [
        row
        for row in runs
        if isinstance(row, dict)
        and row.get("runner") == "flightrecorder_run_suite"
        and row.get("run_suite_lineage_valid") is False
    ]
    invalid_artifact_rows = [
        row
        for row in runs
        if isinstance(row, dict) and row.get("artifact_refs_valid") is False
    ]
    if not invalid_run_suite_rows and not invalid_artifact_rows:
        return
    failed_check_ids = {
        check.get("id")
        for check in checks
        if isinstance(check, dict) and check.get("passed") is False and isinstance(check.get("id"), str)
    }
    if invalid_run_suite_rows and "run_suite_harness_lineage_valid" not in failed_check_ids:
        target.errors.append(
            "evidence_bundle.checks must include a failed run_suite_harness_lineage_valid check when run-suite lineage is invalid."
        )
    if invalid_artifact_rows and "harness_pair_artifacts_valid" not in failed_check_ids:
        target.errors.append(
            "evidence_bundle.checks must include a failed harness_pair_artifacts_valid check when harness artifact references are invalid."
        )

def _validate_evidence_bundle_validation_checks(metrics: dict[str, Any], checks: list[Any], target: ValidationTarget) -> None:
    failed_check_ids = {
        check.get("id")
        for check in checks
        if isinstance(check, dict) and check.get("passed") is False and isinstance(check.get("id"), str)
    }
    validation = metrics.get("validation")
    if isinstance(validation, dict):
        if not _bundle_validation_has_targets(validation) and "validation_has_targets" not in failed_check_ids:
            target.errors.append(
                "evidence_bundle.checks must include a failed validation_has_targets check when validation metrics have no targets."
            )
        if not _bundle_validation_counts_consistent(validation) and "validation_counts_consistent" not in failed_check_ids:
            target.errors.append(
                "evidence_bundle.checks must include a failed validation_counts_consistent check when validation metrics are inconsistent."
            )
    gates = metrics.get("gates")
    if not isinstance(gates, list):
        return
    invalid_gate_targets = []
    invalid_gate_counts = []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        validation_metrics = gate.get("validation")
        if not isinstance(validation_metrics, dict) or validation_metrics.get("available") is not True:
            continue
        if not _bundle_validation_has_targets(validation_metrics):
            invalid_gate_targets.append(gate)
        if not _bundle_validation_counts_consistent(validation_metrics):
            invalid_gate_counts.append(gate)
    if invalid_gate_targets and "gate_validation_has_targets" not in failed_check_ids:
        target.errors.append(
            "evidence_bundle.checks must include a failed gate_validation_has_targets check when gate validation metrics have no targets."
        )
    if invalid_gate_counts and "gate_validation_counts_consistent" not in failed_check_ids:
        target.errors.append(
            "evidence_bundle.checks must include a failed gate_validation_counts_consistent check when gate validation metrics are inconsistent."
        )

def _validate_evidence_bundle_serving_lifecycle(
    metrics: dict[str, Any],
    artifacts: dict[str, Any],
    checks: list[Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    value = metrics.get("serving_lifecycle")
    if not isinstance(value, dict):
        return
    failed_check_ids = {
        check.get("id")
        for check in checks
        if isinstance(check, dict) and check.get("passed") is False and isinstance(check.get("id"), str)
    }
    if value.get("passed") is not True and "serving_lifecycle_passed" not in failed_check_ids:
        target.errors.append("evidence_bundle.checks must include a failed serving_lifecycle_passed check when lifecycle metrics are not passed.")
    if (
        value.get("passed") is True
        and value.get("preflight_artifact_count") != len(SERVING_LIFECYCLE_PREFLIGHT_ARTIFACTS)
        and "serving_lifecycle_preflight_artifacts_present" not in failed_check_ids
    ):
        target.errors.append(
            "evidence_bundle.checks must include a failed serving_lifecycle_preflight_artifacts_present check when passed lifecycle preflight artifacts are incomplete."
        )

    artifact = artifacts.get("serving_lifecycle") if isinstance(artifacts.get("serving_lifecycle"), dict) else {}
    lifecycle_path = _resolve_evidence_bundle_artifact_path(artifact.get("path"), source_path)
    if lifecycle_path is None or not lifecycle_path.exists() or not lifecycle_path.is_file():
        target.errors.append("evidence_bundle.artifacts.serving_lifecycle.path must resolve to an existing file when serving_lifecycle metrics are present.")
        return
    if _path_has_symlink_component(lifecycle_path, include_leaf=True):
        target.errors.append("evidence_bundle.artifacts.serving_lifecycle.path must resolve to a regular non-symlink file when serving_lifecycle metrics are present.")
        return
    try:
        lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        target.errors.append(f"evidence_bundle.artifacts.serving_lifecycle is not valid UTF-8: {exc}")
        return
    except json.JSONDecodeError as exc:
        target.errors.append(f"evidence_bundle.artifacts.serving_lifecycle contains invalid JSON: {exc}")
        return
    if not isinstance(lifecycle, dict):
        target.errors.append("evidence_bundle.artifacts.serving_lifecycle must contain a JSON object.")
        return
    _validate_serving_lifecycle(lifecycle, target, lifecycle_path)
    expected = _serving_lifecycle_bundle_metrics(lifecycle)
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(
                f"evidence_bundle.metrics.serving_lifecycle.{field_name} expected {expected_value!r}, got {value.get(field_name)!r}."
            )

def _validate_evidence_bundle_eval_summary(
    metrics: dict[str, Any],
    artifacts: dict[str, Any],
    checks: list[Any],
    target: ValidationTarget,
    source_path: Path,
) -> None:
    value = metrics.get("eval_summary")
    if not isinstance(value, dict):
        return
    failed_check_ids = {
        check.get("id")
        for check in checks
        if isinstance(check, dict) and check.get("passed") is False and isinstance(check.get("id"), str)
    }
    if value.get("schema_version") != EVAL_SUMMARY_SCHEMA_VERSION and "eval_summary_schema_supported" not in failed_check_ids:
        target.errors.append("evidence_bundle.checks must include a failed eval_summary_schema_supported check when eval-summary schema is unsupported.")
    if value.get("passed") is not True and "eval_summary_passed" not in failed_check_ids:
        target.errors.append("evidence_bundle.checks must include a failed eval_summary_passed check when eval-summary metrics are not passed.")
    if value.get("governance_ready") is not True and "eval_summary_governance_ready" not in failed_check_ids:
        target.errors.append(
            "evidence_bundle.checks must include a failed eval_summary_governance_ready check when eval-summary is not governance-ready."
        )

    artifact = artifacts.get("eval_summary") if isinstance(artifacts.get("eval_summary"), dict) else None
    if artifact is None:
        target.errors.append("evidence_bundle.artifacts.eval_summary is required when eval_summary metrics are present.")
        return
    eval_summary_path = _resolve_evidence_bundle_artifact_path(artifact.get("path"), source_path)
    if eval_summary_path is None or not eval_summary_path.exists() or not eval_summary_path.is_file():
        target.errors.append("evidence_bundle.artifacts.eval_summary.path must resolve to an existing file when eval_summary metrics are present.")
        return
    if _path_has_symlink_component(eval_summary_path, include_leaf=True):
        target.errors.append("evidence_bundle.artifacts.eval_summary.path must resolve to a regular non-symlink file when eval_summary metrics are present.")
        return
    try:
        summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        target.errors.append(f"evidence_bundle.artifacts.eval_summary is not valid UTF-8: {exc}")
        return
    except json.JSONDecodeError as exc:
        target.errors.append(f"evidence_bundle.artifacts.eval_summary contains invalid JSON: {exc}")
        return
    if not isinstance(summary, dict):
        target.errors.append("evidence_bundle.artifacts.eval_summary must contain a JSON object.")
        return
    _validate_eval_summary(summary, target, source_path=eval_summary_path)
    expected = _eval_summary_bundle_metrics(summary)
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            target.errors.append(
                f"evidence_bundle.metrics.eval_summary.{field_name} expected {expected_value!r}, got {value.get(field_name)!r}."
            )

def _validate_evidence_bundle_metrics(metrics: dict[str, Any], target: ValidationTarget) -> None:
    expected_sections = (
        "suite_summary",
        "scenario_quality",
        "evidence_coverage",
        "trace_observability",
        "run_digest_coverage",
        "repair_queue",
        "validation",
        "eval_summary",
        "training_export",
        "compare_export",
        "review_export",
        "reviewed_export",
        "review_calibration",
        "live_smoke_summary",
        "serving_lifecycle",
        "trainer_handoff",
        "harness_handoff",
    )
    for section in expected_sections:
        if section in metrics and not isinstance(metrics[section], dict):
            target.errors.append(f"evidence_bundle.metrics.{section} must be an object when present.")
    training = metrics.get("training_export")
    if isinstance(training, dict):
        _validate_bundle_top_curriculum_priorities(training.get("top_curriculum_priorities"), target)
    run_digest_coverage = metrics.get("run_digest_coverage")
    if isinstance(run_digest_coverage, dict):
        _validate_bundle_run_digest_coverage(run_digest_coverage, target)
    validation = metrics.get("validation")
    if isinstance(validation, dict):
        _validate_bundle_validation_metrics(validation, target, "evidence_bundle.metrics.validation")
    eval_summary = metrics.get("eval_summary")
    if isinstance(eval_summary, dict):
        _validate_bundle_eval_summary_metrics(eval_summary, target)
    trainer_handoff = metrics.get("trainer_handoff")
    if isinstance(trainer_handoff, dict):
        _validate_bundle_trainer_handoff(trainer_handoff, target)
    harness_handoff = metrics.get("harness_handoff")
    if isinstance(harness_handoff, dict):
        _validate_bundle_harness_handoff(harness_handoff, target)
    live_smoke_summary = metrics.get("live_smoke_summary")
    if isinstance(live_smoke_summary, dict):
        _validate_bundle_live_smoke_summary_paths(live_smoke_summary, target)
    serving_lifecycle = metrics.get("serving_lifecycle")
    if isinstance(serving_lifecycle, dict):
        _validate_bundle_serving_lifecycle_metrics(serving_lifecycle, target)
    gates = metrics.get("gates")
    if gates is not None:
        if not isinstance(gates, list):
            target.errors.append("evidence_bundle.metrics.gates must be a list when present.")
            return
        for index, gate in enumerate(gates):
            if not isinstance(gate, dict):
                target.errors.append(f"evidence_bundle.metrics.gates[{index}] must be an object.")
                continue
            for field_name in ("id", "path"):
                if not isinstance(gate.get(field_name), str) or not gate.get(field_name):
                    target.errors.append(f"evidence_bundle.metrics.gates[{index}].{field_name} must be a non-empty string.")
            _warn_absolute_public_path(target, f"evidence_bundle.metrics.gates[{index}].path", gate.get("path"))
            if "schema_version" in gate and not isinstance(gate.get("schema_version"), str):
                target.errors.append(f"evidence_bundle.metrics.gates[{index}].schema_version must be a string when present.")
            if not isinstance(gate.get("passed"), bool):
                target.errors.append(f"evidence_bundle.metrics.gates[{index}].passed must be a boolean.")
            if "validation" in gate:
                _validate_bundle_gate_validation(gate.get("validation"), target, f"evidence_bundle.metrics.gates[{index}].validation")

def _validate_bundle_eval_summary_metrics(value: dict[str, Any], target: ValidationTarget) -> None:
    if value.get("schema_version") != EVAL_SUMMARY_SCHEMA_VERSION:
        target.errors.append(f"evidence_bundle.metrics.eval_summary.schema_version must be {EVAL_SUMMARY_SCHEMA_VERSION}.")
    for field_name in ("passed", "governance_ready", "cross_arm_claims_allowed", "serving_preflight_required"):
        if field_name in value and value.get(field_name) is not None and not isinstance(value.get(field_name), bool):
            target.errors.append(f"evidence_bundle.metrics.eval_summary.{field_name} must be a boolean or null.")
    for field_name in (
        "arm_count",
        "comparison_count",
        "gate_count",
        "external_adapter_plan_count",
        "external_adapter_result_count",
        "risk_count",
        "repair_work_item_count",
        "repair_critical_work_item_count",
        "heldout_scenario_count",
        "serving_preflight_input_count",
        "serving_preflight_attached_count",
        "serving_preflight_blocking_reason_count",
    ):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"evidence_bundle.metrics.eval_summary.{field_name} must be a non-negative integer.")
    if value.get("heldout_status") is not None and value.get("heldout_status") not in {
        "missing_suite_summaries",
        "single_arm",
        "identical",
        "mismatched",
        "empty",
    }:
        target.errors.append("evidence_bundle.metrics.eval_summary.heldout_status has an unsupported value.")
    risk_source_counts = value.get("risk_source_counts")
    if not isinstance(risk_source_counts, list):
        target.errors.append("evidence_bundle.metrics.eval_summary.risk_source_counts must be a list.")
    else:
        _validate_count_rows(risk_source_counts, target, "evidence_bundle.metrics.eval_summary.risk_source_counts")

def _validate_bundle_serving_lifecycle_metrics(value: dict[str, Any], target: ValidationTarget) -> None:
    for field_name in ("passed", "ready", "readiness_probe_ready", "smoke_attempted", "smoke_passed", "teardown_clean"):
        if field_name in value and value.get(field_name) is not None and not isinstance(value.get(field_name), bool):
            target.errors.append(f"evidence_bundle.metrics.serving_lifecycle.{field_name} must be a boolean or null.")
    if value.get("readiness") is not None and value.get("readiness") not in {"ready", "blocked"}:
        target.errors.append("evidence_bundle.metrics.serving_lifecycle.readiness must be ready, blocked, or null.")
    for field_name in ("profile", "engine", "model"):
        if field_name in value and value.get(field_name) is not None and not isinstance(value.get(field_name), str):
            target.errors.append(f"evidence_bundle.metrics.serving_lifecycle.{field_name} must be a string or null.")
    for field_name in ("duration_ms", "error_count", "preflight_artifact_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"evidence_bundle.metrics.serving_lifecycle.{field_name} must be a non-negative integer.")

def _serving_lifecycle_bundle_metrics(lifecycle: dict[str, Any]) -> dict[str, Any]:
    readiness_probe = lifecycle.get("readiness_probe") if isinstance(lifecycle.get("readiness_probe"), dict) else {}
    smoke_check = lifecycle.get("smoke_check") if isinstance(lifecycle.get("smoke_check"), dict) else {}
    teardown = lifecycle.get("teardown") if isinstance(lifecycle.get("teardown"), dict) else {}
    errors = lifecycle.get("errors") if isinstance(lifecycle.get("errors"), list) else []
    artifacts = lifecycle.get("artifacts") if isinstance(lifecycle.get("artifacts"), dict) else {}
    return {
        "passed": lifecycle.get("passed") if isinstance(lifecycle.get("passed"), bool) else None,
        "ready": lifecycle.get("ready") if isinstance(lifecycle.get("ready"), bool) else None,
        "readiness": lifecycle.get("readiness"),
        "profile": lifecycle.get("profile"),
        "engine": lifecycle.get("engine"),
        "model": lifecycle.get("model"),
        "duration_ms": lifecycle.get("duration_ms"),
        "readiness_probe_ready": readiness_probe.get("ready") if isinstance(readiness_probe.get("ready"), bool) else None,
        "smoke_attempted": smoke_check.get("attempted") if isinstance(smoke_check.get("attempted"), bool) else None,
        "smoke_passed": smoke_check.get("passed") if isinstance(smoke_check.get("passed"), bool) else None,
        "teardown_clean": teardown.get("clean") if isinstance(teardown.get("clean"), bool) else None,
        "error_count": len(errors),
        "preflight_artifact_count": sum(
            1 for role in SERVING_LIFECYCLE_PREFLIGHT_ARTIFACTS if isinstance(artifacts.get(role), str) and artifacts.get(role)
        ),
    }

def _eval_summary_bundle_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    heldout = summary.get("heldout_scenarios") if isinstance(summary.get("heldout_scenarios"), dict) else {}
    repair = summary.get("repair_curriculum") if isinstance(summary.get("repair_curriculum"), dict) else {}
    serving = summary.get("serving_preflight") if isinstance(summary.get("serving_preflight"), dict) else {}
    risks = summary.get("risks") if isinstance(summary.get("risks"), list) else []
    serving_blockers = serving.get("blocking_reasons") if isinstance(serving.get("blocking_reasons"), list) else []
    return {
        "schema_version": summary.get("schema_version"),
        "passed": summary.get("passed") if isinstance(summary.get("passed"), bool) else None,
        "governance_ready": summary.get("governance_ready") if isinstance(summary.get("governance_ready"), bool) else None,
        "arm_count": summary.get("arm_count"),
        "comparison_count": summary.get("comparison_count"),
        "gate_count": summary.get("gate_count"),
        "external_adapter_plan_count": summary.get("external_adapter_plan_count"),
        "external_adapter_result_count": summary.get("external_adapter_result_count"),
        "risk_count": len(risks),
        "risk_source_counts": _eval_summary_risk_source_counts(risks),
        "repair_work_item_count": repair.get("work_item_count"),
        "repair_critical_work_item_count": repair.get("critical_work_item_count"),
        "heldout_status": heldout.get("status"),
        "heldout_scenario_count": heldout.get("scenario_count"),
        "cross_arm_claims_allowed": heldout.get("cross_arm_claims_allowed"),
        "serving_preflight_required": serving.get("required") if isinstance(serving.get("required"), bool) else None,
        "serving_preflight_input_count": serving.get("input_count"),
        "serving_preflight_attached_count": serving.get("attached_count"),
        "serving_preflight_blocking_reason_count": len(serving_blockers),
    }

def _eval_summary_risk_source_counts(risks: list[Any]) -> list[dict[str, int | str]]:
    counts: dict[str, int] = {}
    for risk in risks:
        if not isinstance(risk, dict):
            continue
        source = risk.get("source")
        if isinstance(source, str) and source:
            counts[source] = counts.get(source, 0) + 1
    return [{"id": key, "count": counts[key]} for key in sorted(counts)]

def _resolve_evidence_bundle_artifact_path(value: Any, source_path: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("<redacted:") and value.endswith(">"):
        basename = value.removeprefix("<redacted:").removesuffix(">")
        return source_path.parent / basename
    path = Path(value)
    if path.is_absolute():
        return path
    return source_path.parent / path

def _validate_bundle_gate_validation(value: Any, target: ValidationTarget, label: str) -> None:
    if not isinstance(value, dict):
        target.errors.append(f"{label} must be an object when present.")
        return
    _validate_bundle_validation_metrics(value, target, label, require_available=True)

def _validate_bundle_validation_metrics(
    value: dict[str, Any],
    target: ValidationTarget,
    label: str,
    *,
    require_available: bool = False,
) -> None:
    if require_available and not isinstance(value.get("available"), bool):
        target.errors.append(f"{label}.available must be a boolean.")
    for field_name in ("available", "passed", "strict"):
        if field_name == "available" and not require_available and field_name not in value:
            continue
        if field_name in value and not isinstance(value.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    for field_name in ("target_count", "error_count", "warning_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")

def _bundle_validation_has_targets(value: dict[str, Any]) -> bool:
    target_count = value.get("target_count")
    return _is_non_negative_int(target_count) and int(target_count) > 0

def _bundle_validation_counts_consistent(value: dict[str, Any]) -> bool:
    passed = value.get("passed")
    error_count = value.get("error_count")
    warning_count = value.get("warning_count")
    if not isinstance(passed, bool):
        return True
    if not _is_non_negative_int(error_count) or not _is_non_negative_int(warning_count):
        return True
    strict = value.get("strict") is True
    expected_passed = int(error_count) == 0 and (int(warning_count) == 0 or not strict)
    return passed == expected_passed

def _validate_bundle_run_digest_coverage(value: dict[str, Any], target: ValidationTarget) -> None:
    label = "evidence_bundle.metrics.run_digest_coverage"
    if not isinstance(value.get("runs_dir"), str) or not value.get("runs_dir"):
        target.errors.append(f"{label}.runs_dir must be a non-empty string.")
    else:
        _warn_absolute_public_path(target, f"{label}.runs_dir", value.get("runs_dir"))
    for field_name in (
        "run_count",
        "digest_count",
        "missing_digest_count",
        "invalid_digest_count",
        "passed_digest_count",
        "failed_digest_count",
    ):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    if not _is_number_between(value.get("digest_coverage_rate"), 0.0, 1.0):
        target.errors.append(f"{label}.digest_coverage_rate must be numeric from 0.0 to 1.0.")

    run_count = value.get("run_count")
    digest_count = value.get("digest_count")
    missing_count = value.get("missing_digest_count")
    invalid_count = value.get("invalid_digest_count")
    passed_count = value.get("passed_digest_count")
    failed_count = value.get("failed_digest_count")
    if (
        _is_non_negative_int(run_count)
        and _is_non_negative_int(digest_count)
        and _is_non_negative_int(missing_count)
        and _is_non_negative_int(invalid_count)
    ):
        if int(digest_count) + int(missing_count) + int(invalid_count) != int(run_count):
            target.errors.append(f"{label}.digest_count + missing_digest_count + invalid_digest_count must equal run_count.")
        expected_rate = 1.0 if int(run_count) == 0 else round(int(digest_count) / int(run_count), 4)
        if value.get("digest_coverage_rate") != expected_rate:
            target.errors.append(f"{label}.digest_coverage_rate expected {expected_rate!r}, got {value.get('digest_coverage_rate')!r}.")
    if _is_non_negative_int(digest_count) and _is_non_negative_int(passed_count) and _is_non_negative_int(failed_count):
        if int(passed_count) + int(failed_count) != int(digest_count):
            target.errors.append(f"{label}.passed_digest_count + failed_digest_count must equal digest_count.")
    task_status_counts = _validate_count_rows(value.get("task_completion_status_counts"), target, f"{label}.task_completion_status_counts")
    if _is_non_negative_int(digest_count) and sum(task_status_counts.values()) != int(digest_count):
        target.errors.append(f"{label}.task_completion_status_counts total must equal digest_count.")
    _validate_count_rows(value.get("recommended_action_counts"), target, f"{label}.recommended_action_counts")
    for field_name in ("missing_digest_scenarios", "invalid_digest_scenarios"):
        if not _is_string_list(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a list of strings.")

def _validate_bundle_live_smoke_summary_paths(value: dict[str, Any], target: ValidationTarget) -> None:
    label = "evidence_bundle.metrics.live_smoke_summary"
    for field_name in ("hermes_root", "flight_recorder_root"):
        _warn_absolute_public_path(target, f"{label}.{field_name}", value.get(field_name))

def _validate_bundle_harness_handoff(value: dict[str, Any], target: ValidationTarget) -> None:
    label = "evidence_bundle.metrics.harness_handoff"
    for field_name in (
        "manifest_count",
        "result_count",
        "pair_count",
        "passed_pair_count",
        "failed_pair_count",
        "schema_valid_pair_count",
        "artifact_valid_pair_count",
        "consistent_pair_count",
        "missing_pair_count",
        "run_suite_pair_count",
        "run_suite_lineage_valid_pair_count",
    ):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    for field_name in ("runners", "providers", "models", "trace_formats"):
        _validate_count_rows(value.get(field_name), target, f"{label}.{field_name}")
    runs = value.get("runs")
    if not isinstance(runs, list):
        target.errors.append(f"{label}.runs must be a list.")
        runs = []
    if _is_non_negative_int(value.get("pair_count")) and int(value["pair_count"]) != len(runs):
        target.errors.append(f"{label}.pair_count expected {len(runs)}, got {value.get('pair_count')!r}.")

    passed_count = 0
    failed_count = 0
    schema_valid_count = 0
    artifact_valid_count = 0
    consistent_count = 0
    run_suite_count = 0
    run_suite_lineage_valid_count = 0
    for index, row in enumerate(runs):
        row_label = f"{label}.runs[{index}]"
        if not isinstance(row, dict):
            target.errors.append(f"{row_label} must be an object.")
            continue
        for field_name in (
            "id",
            "scenario_id",
            "runner",
            "provider",
            "model",
            "manifest_path",
            "result_path",
            "handoff_source",
            "trace_format",
        ):
            if not isinstance(row.get(field_name), str) or not row.get(field_name):
                target.errors.append(f"{row_label}.{field_name} must be a non-empty string.")
        for field_name in ("passed", "schema_valid", "artifact_refs_valid", "consistent", "run_suite_lineage_valid"):
            if not isinstance(row.get(field_name), bool):
                target.errors.append(f"{row_label}.{field_name} must be a boolean.")
        for field_name in ("suite_summary_path", "suite_selected_scenario_id"):
            if row.get(field_name) is not None and not isinstance(row.get(field_name), str):
                target.errors.append(f"{row_label}.{field_name} must be a string when present.")
        for field_name in ("manifest_path", "result_path", "trace_path", "suite_summary_path", "replay_lineage"):
            _warn_absolute_public_path(target, f"{row_label}.{field_name}", row.get(field_name))
        for field_name in ("suite_total", "suite_passed", "suite_failed"):
            if row.get(field_name) is not None and not _is_non_negative_int(row.get(field_name)):
                target.errors.append(f"{row_label}.{field_name} must be a non-negative integer or null.")
        if row.get("score") is not None and (not isinstance(row.get("score"), (int, float)) or isinstance(row.get("score"), bool)):
            target.errors.append(f"{row_label}.score must be numeric when present.")
        if not _is_string_list(row.get("schema_errors")):
            target.errors.append(f"{row_label}.schema_errors must be a list of strings.")
        if not _is_string_list(row.get("artifact_ref_errors")):
            target.errors.append(f"{row_label}.artifact_ref_errors must be a list of strings.")
        if not _is_string_list(row.get("consistency_errors")):
            target.errors.append(f"{row_label}.consistency_errors must be a list of strings.")
        if row.get("passed") is True:
            passed_count += 1
        elif row.get("passed") is False:
            failed_count += 1
        if row.get("schema_valid") is True:
            schema_valid_count += 1
        if row.get("artifact_refs_valid") is True:
            artifact_valid_count += 1
        if row.get("consistent") is True:
            consistent_count += 1
        if row.get("runner") == "flightrecorder_run_suite":
            run_suite_count += 1
            if row.get("run_suite_lineage_valid") is True:
                run_suite_lineage_valid_count += 1
                if row.get("handoff_source") != RUN_SUITE_HARNESS_SOURCE:
                    target.errors.append(f"{row_label}.handoff_source must be {RUN_SUITE_HARNESS_SOURCE!r}.")
                if not isinstance(row.get("suite_summary_path"), str) or not row.get("suite_summary_path"):
                    target.errors.append(f"{row_label}.suite_summary_path must be a non-empty string for run-suite handoffs.")
                if row.get("suite_selected_scenario_id") != row.get("scenario_id"):
                    target.errors.append(f"{row_label}.suite_selected_scenario_id must match scenario_id.")
                suite_total = row.get("suite_total")
                suite_passed = row.get("suite_passed")
                suite_failed = row.get("suite_failed")
                if not (
                    _is_non_negative_int(suite_total)
                    and _is_non_negative_int(suite_passed)
                    and _is_non_negative_int(suite_failed)
                ):
                    target.errors.append(f"{row_label}.suite_total, suite_passed, and suite_failed are required for run-suite handoffs.")
                elif int(suite_passed) + int(suite_failed) != int(suite_total):
                    target.errors.append(f"{row_label}.suite_passed + suite_failed must equal suite_total.")

    expected_counts = {
        "passed_pair_count": passed_count,
        "failed_pair_count": failed_count,
        "schema_valid_pair_count": schema_valid_count,
        "artifact_valid_pair_count": artifact_valid_count,
        "consistent_pair_count": consistent_count,
        "run_suite_pair_count": run_suite_count,
        "run_suite_lineage_valid_pair_count": run_suite_lineage_valid_count,
    }
    for field_name, expected in expected_counts.items():
        if _is_non_negative_int(value.get(field_name)) and int(value[field_name]) != expected:
            target.errors.append(f"{label}.{field_name} expected {expected}, got {value.get(field_name)!r}.")

def _validate_bundle_trainer_handoff(value: dict[str, Any], target: ValidationTarget) -> None:
    label = "evidence_bundle.metrics.trainer_handoff"
    expected_stage_ids = (
        "trainer_preflight",
        "trainer_launch_check",
        "trainer_archive",
        "trainer_archive_check",
        "trainer_consumer_plan",
        "agentic_training_flow",
        "trainer_wrapper_dry_run",
        "agentic_training_result",
    )
    for field_name in ("stage_count", "handoff_ready_count", "blocked_stage_count", "schema_supported_count"):
        if not _is_non_negative_int(value.get(field_name)):
            target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
    for field_name in ("complete_chain", "all_included_ready"):
        if not isinstance(value.get(field_name), bool):
            target.errors.append(f"{label}.{field_name} must be a boolean.")
    if not _is_string_list(value.get("missing_stage_ids")):
        target.errors.append(f"{label}.missing_stage_ids must be a list of strings.")
    stages = value.get("stages")
    if not isinstance(stages, list):
        target.errors.append(f"{label}.stages must be a list.")
        stages = []
    if _is_non_negative_int(value.get("stage_count")) and int(value["stage_count"]) != len(stages):
        target.errors.append(f"{label}.stage_count expected {len(stages)}, got {value.get('stage_count')!r}.")

    ids: list[str] = []
    handoff_ready_count = 0
    blocked_count = 0
    schema_supported_count = 0
    for index, stage in enumerate(stages):
        stage_label = f"{label}.stages[{index}]"
        if not isinstance(stage, dict):
            target.errors.append(f"{stage_label} must be an object.")
            continue
        stage_id = stage.get("id")
        if not isinstance(stage_id, str) or not stage_id:
            target.errors.append(f"{stage_label}.id must be a non-empty string.")
        else:
            ids.append(stage_id)
            if stage_id not in expected_stage_ids:
                target.errors.append(f"{stage_label}.id has unknown trainer handoff stage {stage_id!r}.")
        for field_name in ("path", "schema_version", "expected_schema_version", "readiness", "recommendation", "expected_recommendation"):
            if not isinstance(stage.get(field_name), str) or not stage.get(field_name):
                target.errors.append(f"{stage_label}.{field_name} must be a non-empty string.")
        _warn_absolute_public_path(target, f"{stage_label}.path", stage.get("path"))
        for field_name in ("schema_supported", "passed", "handoff_ready"):
            if not isinstance(stage.get(field_name), bool):
                target.errors.append(f"{stage_label}.{field_name} must be a boolean.")
        for field_name in ("check_count", "failed_check_count"):
            if not _is_non_negative_int(stage.get(field_name)):
                target.errors.append(f"{stage_label}.{field_name} must be a non-negative integer.")
        if (
            stage.get("handoff_ready") is True
            and _is_non_negative_int(stage.get("failed_check_count"))
            and int(stage["failed_check_count"]) > 0
        ):
            target.errors.append(f"{stage_label}.handoff_ready cannot be true when failed_check_count is greater than 0.")
        for field_name in (
            "gate_count",
            "passed_gate_count",
            "trainer_input_count",
            "trainer_input_ready_count",
            "trainer_input_available_count",
            "external_code_file_count",
            "external_code_ready_count",
            "missing_external_code_count",
            "missing_trainer_input_count",
            "command_arg_count",
            "artifact_count",
            "regular_artifact_count",
            "output_artifact_count",
            "config_count",
            "metrics_file_count",
            "adapter_count",
            "checkpoint_count",
            "log_count",
            "failure_report_count",
            "missing_count",
            "path_rewrite_count",
        ):
            if field_name in stage and not _is_non_negative_int(stage.get(field_name)):
                target.errors.append(f"{stage_label}.{field_name} must be a non-negative integer when present.")
        for field_name in ("status", "runner_id", "run_id", "failure_class"):
            if field_name in stage and (not isinstance(stage.get(field_name), str) or not stage.get(field_name)):
                target.errors.append(f"{stage_label}.{field_name} must be a non-empty string when present.")
        if "expected_recommendations" in stage and not _is_string_list(stage.get("expected_recommendations")):
            target.errors.append(f"{stage_label}.expected_recommendations must be a list of strings when present.")
        if "registry_update_ready" in stage and not isinstance(stage.get("registry_update_ready"), bool):
            target.errors.append(f"{stage_label}.registry_update_ready must be a boolean when present.")
        if stage.get("handoff_ready") is True:
            handoff_ready_count += 1
        elif stage.get("handoff_ready") is False:
            blocked_count += 1
        if stage.get("schema_supported") is True:
            schema_supported_count += 1

    expected_missing = [stage_id for stage_id in expected_stage_ids if stage_id not in ids]
    if value.get("missing_stage_ids") != expected_missing:
        target.errors.append(f"{label}.missing_stage_ids expected {expected_missing!r}, got {value.get('missing_stage_ids')!r}.")
    if isinstance(value.get("complete_chain"), bool) and value["complete_chain"] != (not expected_missing):
        target.errors.append(f"{label}.complete_chain must match missing_stage_ids.")
    if isinstance(value.get("all_included_ready"), bool) and value["all_included_ready"] != all(
        isinstance(stage, dict) and stage.get("handoff_ready") is True for stage in stages
    ):
        target.errors.append(f"{label}.all_included_ready must match stages[].handoff_ready.")
    expected_counts = {
        "handoff_ready_count": handoff_ready_count,
        "blocked_stage_count": blocked_count,
        "schema_supported_count": schema_supported_count,
    }
    for field_name, expected in expected_counts.items():
        if _is_non_negative_int(value.get(field_name)) and int(value[field_name]) != expected:
            target.errors.append(f"{label}.{field_name} expected {expected}, got {value.get(field_name)!r}.")

def _validate_bundle_top_curriculum_priorities(value: Any, target: ValidationTarget) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        target.errors.append("evidence_bundle.metrics.training_export.top_curriculum_priorities must be a list when present.")
        return
    previous_score: int | None = None
    for index, item in enumerate(value):
        label = f"evidence_bundle.metrics.training_export.top_curriculum_priorities[{index}]"
        if not isinstance(item, dict):
            target.errors.append(f"{label} must be an object.")
            continue
        for field_name in ("task_family", "rule_id", "rule_name", "priority_band"):
            if not isinstance(item.get(field_name), str) or not item.get(field_name):
                target.errors.append(f"{label}.{field_name} must be a non-empty string.")
        if item.get("priority_band") not in {"critical", "high", "medium", "low"}:
            target.errors.append(f"{label}.priority_band must be critical, high, medium, or low.")
        for field_name in ("priority_score", "count", "critical_count", "max_penalty"):
            if not _is_non_negative_int(item.get(field_name)):
                target.errors.append(f"{label}.{field_name} must be a non-negative integer.")
        score = item.get("priority_score")
        if _is_non_negative_int(score):
            if previous_score is not None and int(score) > previous_score:
                target.errors.append(f"{label}.priority_score must be sorted descending.")
            previous_score = int(score)
        for field_name in ("scenario_ids", "failure_ids"):
            if not _is_string_list(item.get(field_name)):
                target.errors.append(f"{label}.{field_name} must be a list of strings.")
        _validate_evidence_refs(item.get("example_evidence_refs"), target, f"{label}.example_evidence_refs")
