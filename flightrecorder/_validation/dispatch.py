"""Extracted validation implementation."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any
from .cloud import validate_cloud_training_artifact_manifest, validate_cloud_training_completion_receipt, validate_cloud_training_launch_plan, validate_cloud_training_launch_receipt, validate_cloud_training_preflight, validate_cloud_training_provider_registry, validate_cloud_training_status_receipt
from .constants import VALIDATION_SCHEMA_VERSION
from .evaluation_serving import validate_agentic_rollout_plan, validate_agentic_rollout_receipt, validate_dataset_curation_receipt, validate_eval_suite_manifest, validate_eval_summary, validate_evidence_coverage, validate_external_eval_plan, validate_external_eval_receipt, validate_external_eval_result, validate_harness_suite_result, validate_heldout_manifest, validate_model_grader_disagreement_queue, validate_model_grader_dry_run, validate_model_grader_gate, validate_model_grader_override_receipt, validate_rejection_sampling_gate, validate_rubric_spec, validate_scenario_check, validate_scenario_quality, validate_serving_compatibility_report, validate_serving_demo_run, validate_serving_endpoint_check, validate_serving_lifecycle, validate_serving_profile, validate_suite_summary, validate_suite_trend
from .exports import validate_compare_export, validate_training_export
from .governance import validate_action_ledger, validate_action_ledger_gate, validate_decision_gate, validate_evidence_bundle, validate_improvement_ledger, validate_improvement_ledger_gate, validate_improvement_plan, validate_promotion_alias_apply, validate_promotion_archive, validate_promotion_cards, validate_promotion_decision, validate_promotion_ledger, validate_promotion_ledger_gate, validate_promotion_policy, validate_promotion_release_record, validate_promotion_rollback_receipt, validate_repair_queue, validate_replay_bundle
from .models import validate_model_adapter_manifest, validate_model_candidate, validate_model_compatibility_report, validate_model_registry, validate_model_registry_entry, validate_model_scout_manifest, validate_model_serving_probe_receipt, validate_training_plan
from .primitives import ValidationTarget
from .review import validate_review_calibration, validate_review_export, validate_reviewed_export, validate_reviewed_gate
from .runs import validate_adapter_route_decision_artifact, validate_harness_replay_result, validate_harness_run_manifest, validate_harness_run_result, validate_live_smoke_summary, validate_run_digest, validate_run_dir, validate_runs_dir, validate_state_diff, validate_state_snapshot, validate_tool_capability_selection_artifact, validate_trace_observability
from .trainer_archive import validate_trainer_archive, validate_trainer_archive_check, validate_trainer_consumer_plan, validate_trainer_launch_check, validate_trainer_preflight, validate_trainer_wrapper_dry_run
from .training_flow_result import validate_agentic_training_flow, validate_agentic_training_result
from .training_loop import validate_agentic_loop_governance_receipt, validate_agentic_loop_ledger, validate_agentic_training_loop_plan, validate_next_iteration_schedule
from .training_plan_runtime import validate_agentic_training_plan, validate_agentic_training_runtime_preflight

def validate_artifacts(
    *,
    runs_dir: str | Path | None = None,
    run_dirs: list[str | Path] | None = None,
    training_export_dir: str | Path | None = None,
    compare_export_dir: str | Path | None = None,
    review_export_dir: str | Path | None = None,
    reviewed_export_dir: str | Path | None = None,
    reviewed_gate_paths: list[str | Path] | None = None,
    evidence_coverage_paths: list[str | Path] | None = None,
    evidence_bundle_paths: list[str | Path] | None = None,
    improvement_plan_paths: list[str | Path] | None = None,
    improvement_ledger_paths: list[str | Path] | None = None,
    improvement_ledger_gate_paths: list[str | Path] | None = None,
    action_ledger_paths: list[str | Path] | None = None,
    action_ledger_gate_paths: list[str | Path] | None = None,
    decision_gate_paths: list[str | Path] | None = None,
    promotion_cards_paths: list[str | Path] | None = None,
    promotion_decision_paths: list[str | Path] | None = None,
    promotion_alias_apply_paths: list[str | Path] | None = None,
    promotion_rollback_receipt_paths: list[str | Path] | None = None,
    promotion_release_record_paths: list[str | Path] | None = None,
    promotion_policy_paths: list[str | Path] | None = None,
    promotion_ledger_paths: list[str | Path] | None = None,
    promotion_ledger_gate_paths: list[str | Path] | None = None,
    promotion_archive_paths: list[str | Path] | None = None,
    trainer_preflight_paths: list[str | Path] | None = None,
    trainer_launch_check_paths: list[str | Path] | None = None,
    trainer_archive_paths: list[str | Path] | None = None,
    trainer_archive_check_paths: list[str | Path] | None = None,
    trainer_consumer_plan_paths: list[str | Path] | None = None,
    trainer_wrapper_dry_run_paths: list[str | Path] | None = None,
    model_scout_manifest_paths: list[str | Path] | None = None,
    model_candidate_paths: list[str | Path] | None = None,
    model_compatibility_report_paths: list[str | Path] | None = None,
    model_serving_probe_receipt_paths: list[str | Path] | None = None,
    model_adapter_manifest_paths: list[str | Path] | None = None,
    model_registry_entry_paths: list[str | Path] | None = None,
    model_registry_paths: list[str | Path] | None = None,
    training_plan_paths: list[str | Path] | None = None,
    agentic_training_plan_paths: list[str | Path] | None = None,
    agentic_training_runtime_preflight_paths: list[str | Path] | None = None,
    agentic_training_flow_paths: list[str | Path] | None = None,
    agentic_training_result_paths: list[str | Path] | None = None,
    agentic_training_loop_plan_paths: list[str | Path] | None = None,
    agentic_loop_ledger_paths: list[str | Path] | None = None,
    agentic_loop_governance_receipt_paths: list[str | Path] | None = None,
    next_iteration_schedule_paths: list[str | Path] | None = None,
    cloud_training_provider_registry_paths: list[str | Path] | None = None,
    cloud_training_preflight_paths: list[str | Path] | None = None,
    cloud_training_artifact_manifest_paths: list[str | Path] | None = None,
    cloud_training_launch_plan_paths: list[str | Path] | None = None,
    cloud_training_launch_receipt_paths: list[str | Path] | None = None,
    cloud_training_status_receipt_paths: list[str | Path] | None = None,
    cloud_training_completion_receipt_paths: list[str | Path] | None = None,
    agentic_rollout_plan_paths: list[str | Path] | None = None,
    agentic_rollout_receipt_paths: list[str | Path] | None = None,
    rejection_sampling_gate_paths: list[str | Path] | None = None,
    dataset_curation_receipt_paths: list[str | Path] | None = None,
    rubric_spec_paths: list[str | Path] | None = None,
    model_grader_dry_run_paths: list[str | Path] | None = None,
    model_grader_disagreement_queue_paths: list[str | Path] | None = None,
    model_grader_override_receipt_paths: list[str | Path] | None = None,
    model_grader_gate_paths: list[str | Path] | None = None,
    repair_queue_paths: list[str | Path] | None = None,
    replay_bundle_paths: list[str | Path] | None = None,
    trace_observability_paths: list[str | Path] | None = None,
    review_calibration_paths: list[str | Path] | None = None,
    scenario_check_paths: list[str | Path] | None = None,
    scenario_quality_paths: list[str | Path] | None = None,
    suite_summary_paths: list[str | Path] | None = None,
    suite_trend_paths: list[str | Path] | None = None,
    eval_suite_manifest_paths: list[str | Path] | None = None,
    state_snapshot_paths: list[str | Path] | None = None,
    state_diff_paths: list[str | Path] | None = None,
    run_digest_paths: list[str | Path] | None = None,
    harness_manifest_paths: list[str | Path] | None = None,
    harness_result_paths: list[str | Path] | None = None,
    harness_replay_result_paths: list[str | Path] | None = None,
    harness_suite_result_paths: list[str | Path] | None = None,
    live_smoke_summary_paths: list[str | Path] | None = None,
    eval_summary_paths: list[str | Path] | None = None,
    tool_capability_selection_paths: list[str | Path] | None = None,
    adapter_route_decision_paths: list[str | Path] | None = None,
    external_eval_plan_paths: list[str | Path] | None = None,
    external_eval_receipt_paths: list[str | Path] | None = None,
    external_eval_result_paths: list[str | Path] | None = None,
    heldout_manifest_paths: list[str | Path] | None = None,
    serving_profile_paths: list[str | Path] | None = None,
    serving_compatibility_report_paths: list[str | Path] | None = None,
    serving_endpoint_check_paths: list[str | Path] | None = None,
    serving_lifecycle_paths: list[str | Path] | None = None,
    serving_demo_run_paths: list[str | Path] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate generated Flight Recorder run and training artifacts."""
    targets: list[ValidationTarget] = []
    for run_dir in run_dirs or []:
        targets.append(validate_run_dir(run_dir))
    if runs_dir is not None:
        targets.extend(validate_runs_dir(runs_dir))
    if training_export_dir is not None:
        targets.append(validate_training_export(training_export_dir))
    if compare_export_dir is not None:
        targets.append(validate_compare_export(compare_export_dir))
    if review_export_dir is not None:
        targets.append(validate_review_export(review_export_dir))
    if reviewed_export_dir is not None:
        targets.append(validate_reviewed_export(reviewed_export_dir))
    for reviewed_gate_path in reviewed_gate_paths or []:
        targets.append(validate_reviewed_gate(reviewed_gate_path))
    for evidence_coverage_path in evidence_coverage_paths or []:
        targets.append(validate_evidence_coverage(evidence_coverage_path))
    for evidence_bundle_path in evidence_bundle_paths or []:
        targets.append(validate_evidence_bundle(evidence_bundle_path))
    for improvement_plan_path in improvement_plan_paths or []:
        targets.append(validate_improvement_plan(improvement_plan_path))
    for improvement_ledger_path in improvement_ledger_paths or []:
        targets.append(validate_improvement_ledger(improvement_ledger_path))
    for improvement_ledger_gate_path in improvement_ledger_gate_paths or []:
        targets.append(validate_improvement_ledger_gate(improvement_ledger_gate_path))
    for action_ledger_path in action_ledger_paths or []:
        targets.append(validate_action_ledger(action_ledger_path))
    for action_ledger_gate_path in action_ledger_gate_paths or []:
        targets.append(validate_action_ledger_gate(action_ledger_gate_path))
    for decision_gate_path in decision_gate_paths or []:
        targets.append(validate_decision_gate(decision_gate_path))
    for promotion_cards_path in promotion_cards_paths or []:
        targets.append(validate_promotion_cards(promotion_cards_path))
    for promotion_decision_path in promotion_decision_paths or []:
        targets.append(validate_promotion_decision(promotion_decision_path))
    for promotion_alias_apply_path in promotion_alias_apply_paths or []:
        targets.append(validate_promotion_alias_apply(promotion_alias_apply_path))
    for promotion_rollback_receipt_path in promotion_rollback_receipt_paths or []:
        targets.append(validate_promotion_rollback_receipt(promotion_rollback_receipt_path))
    for promotion_release_record_path in promotion_release_record_paths or []:
        targets.append(validate_promotion_release_record(promotion_release_record_path))
    for promotion_policy_path in promotion_policy_paths or []:
        targets.append(validate_promotion_policy(promotion_policy_path))
    for promotion_ledger_path in promotion_ledger_paths or []:
        targets.append(validate_promotion_ledger(promotion_ledger_path))
    for promotion_ledger_gate_path in promotion_ledger_gate_paths or []:
        targets.append(validate_promotion_ledger_gate(promotion_ledger_gate_path))
    for promotion_archive_path in promotion_archive_paths or []:
        targets.append(validate_promotion_archive(promotion_archive_path))
    for trainer_preflight_path in trainer_preflight_paths or []:
        targets.append(validate_trainer_preflight(trainer_preflight_path))
    for trainer_launch_check_path in trainer_launch_check_paths or []:
        targets.append(validate_trainer_launch_check(trainer_launch_check_path))
    for trainer_archive_path in trainer_archive_paths or []:
        targets.append(validate_trainer_archive(trainer_archive_path))
    for trainer_archive_check_path in trainer_archive_check_paths or []:
        targets.append(validate_trainer_archive_check(trainer_archive_check_path))
    for trainer_consumer_plan_path in trainer_consumer_plan_paths or []:
        targets.append(validate_trainer_consumer_plan(trainer_consumer_plan_path))
    for trainer_wrapper_dry_run_path in trainer_wrapper_dry_run_paths or []:
        targets.append(validate_trainer_wrapper_dry_run(trainer_wrapper_dry_run_path))
    for model_scout_manifest_path in model_scout_manifest_paths or []:
        targets.append(validate_model_scout_manifest(model_scout_manifest_path))
    for model_candidate_path in model_candidate_paths or []:
        targets.append(validate_model_candidate(model_candidate_path))
    for model_compatibility_report_path in model_compatibility_report_paths or []:
        targets.append(validate_model_compatibility_report(model_compatibility_report_path))
    for model_serving_probe_receipt_path in model_serving_probe_receipt_paths or []:
        targets.append(validate_model_serving_probe_receipt(model_serving_probe_receipt_path))
    for model_adapter_manifest_path in model_adapter_manifest_paths or []:
        targets.append(validate_model_adapter_manifest(model_adapter_manifest_path))
    for model_registry_entry_path in model_registry_entry_paths or []:
        targets.append(validate_model_registry_entry(model_registry_entry_path))
    for model_registry_path in model_registry_paths or []:
        targets.append(validate_model_registry(model_registry_path))
    for training_plan_path in training_plan_paths or []:
        targets.append(validate_training_plan(training_plan_path))
    for agentic_training_plan_path in agentic_training_plan_paths or []:
        targets.append(validate_agentic_training_plan(agentic_training_plan_path))
    for agentic_training_runtime_preflight_path in agentic_training_runtime_preflight_paths or []:
        targets.append(validate_agentic_training_runtime_preflight(agentic_training_runtime_preflight_path))
    for agentic_training_flow_path in agentic_training_flow_paths or []:
        targets.append(validate_agentic_training_flow(agentic_training_flow_path))
    for agentic_training_result_path in agentic_training_result_paths or []:
        targets.append(validate_agentic_training_result(agentic_training_result_path))
    for agentic_training_loop_plan_path in agentic_training_loop_plan_paths or []:
        targets.append(validate_agentic_training_loop_plan(agentic_training_loop_plan_path))
    for agentic_loop_ledger_path in agentic_loop_ledger_paths or []:
        targets.append(validate_agentic_loop_ledger(agentic_loop_ledger_path))
    for agentic_loop_governance_receipt_path in agentic_loop_governance_receipt_paths or []:
        targets.append(validate_agentic_loop_governance_receipt(agentic_loop_governance_receipt_path))
    for next_iteration_schedule_path in next_iteration_schedule_paths or []:
        targets.append(validate_next_iteration_schedule(next_iteration_schedule_path))
    for cloud_training_provider_registry_path in cloud_training_provider_registry_paths or []:
        targets.append(validate_cloud_training_provider_registry(cloud_training_provider_registry_path))
    for cloud_training_preflight_path in cloud_training_preflight_paths or []:
        targets.append(validate_cloud_training_preflight(cloud_training_preflight_path))
    for cloud_training_artifact_manifest_path in cloud_training_artifact_manifest_paths or []:
        targets.append(validate_cloud_training_artifact_manifest(cloud_training_artifact_manifest_path))
    for cloud_training_launch_plan_path in cloud_training_launch_plan_paths or []:
        targets.append(validate_cloud_training_launch_plan(cloud_training_launch_plan_path))
    for cloud_training_launch_receipt_path in cloud_training_launch_receipt_paths or []:
        targets.append(validate_cloud_training_launch_receipt(cloud_training_launch_receipt_path))
    for cloud_training_status_receipt_path in cloud_training_status_receipt_paths or []:
        targets.append(validate_cloud_training_status_receipt(cloud_training_status_receipt_path))
    for cloud_training_completion_receipt_path in cloud_training_completion_receipt_paths or []:
        targets.append(validate_cloud_training_completion_receipt(cloud_training_completion_receipt_path))
    for agentic_rollout_plan_path in agentic_rollout_plan_paths or []:
        targets.append(validate_agentic_rollout_plan(agentic_rollout_plan_path))
    for agentic_rollout_receipt_path in agentic_rollout_receipt_paths or []:
        targets.append(validate_agentic_rollout_receipt(agentic_rollout_receipt_path))
    for rejection_sampling_gate_path in rejection_sampling_gate_paths or []:
        targets.append(validate_rejection_sampling_gate(rejection_sampling_gate_path))
    for dataset_curation_receipt_path in dataset_curation_receipt_paths or []:
        targets.append(validate_dataset_curation_receipt(dataset_curation_receipt_path))
    for rubric_spec_path in rubric_spec_paths or []:
        targets.append(validate_rubric_spec(rubric_spec_path))
    for model_grader_dry_run_path in model_grader_dry_run_paths or []:
        targets.append(validate_model_grader_dry_run(model_grader_dry_run_path))
    for model_grader_disagreement_queue_path in model_grader_disagreement_queue_paths or []:
        targets.append(validate_model_grader_disagreement_queue(model_grader_disagreement_queue_path))
    for model_grader_override_receipt_path in model_grader_override_receipt_paths or []:
        targets.append(validate_model_grader_override_receipt(model_grader_override_receipt_path))
    for model_grader_gate_path in model_grader_gate_paths or []:
        targets.append(validate_model_grader_gate(model_grader_gate_path))
    for repair_queue_path in repair_queue_paths or []:
        targets.append(validate_repair_queue(repair_queue_path))
    for replay_bundle_path in replay_bundle_paths or []:
        targets.append(validate_replay_bundle(replay_bundle_path))
    for trace_observability_path in trace_observability_paths or []:
        targets.append(validate_trace_observability(trace_observability_path))
    for review_calibration_path in review_calibration_paths or []:
        targets.append(validate_review_calibration(review_calibration_path))
    for scenario_check_path in scenario_check_paths or []:
        targets.append(validate_scenario_check(scenario_check_path))
    for scenario_quality_path in scenario_quality_paths or []:
        targets.append(validate_scenario_quality(scenario_quality_path))
    for suite_summary_path in suite_summary_paths or []:
        targets.append(validate_suite_summary(suite_summary_path))
    for suite_trend_path in suite_trend_paths or []:
        targets.append(validate_suite_trend(suite_trend_path))
    for eval_suite_manifest_path in eval_suite_manifest_paths or []:
        targets.append(validate_eval_suite_manifest(eval_suite_manifest_path))
    for state_snapshot_path in state_snapshot_paths or []:
        targets.append(validate_state_snapshot(state_snapshot_path))
    for state_diff_path in state_diff_paths or []:
        targets.append(validate_state_diff(state_diff_path))
    for run_digest_path in run_digest_paths or []:
        targets.append(validate_run_digest(run_digest_path))
    for harness_manifest_path in harness_manifest_paths or []:
        targets.append(validate_harness_run_manifest(harness_manifest_path))
    for harness_result_path in harness_result_paths or []:
        targets.append(validate_harness_run_result(harness_result_path))
    for harness_replay_result_path in harness_replay_result_paths or []:
        targets.append(validate_harness_replay_result(harness_replay_result_path))
    for harness_suite_result_path in harness_suite_result_paths or []:
        targets.append(validate_harness_suite_result(harness_suite_result_path))
    for live_smoke_summary_path in live_smoke_summary_paths or []:
        targets.append(validate_live_smoke_summary(live_smoke_summary_path))
    for eval_summary_path in eval_summary_paths or []:
        targets.append(validate_eval_summary(eval_summary_path))
    for tool_capability_selection_path in tool_capability_selection_paths or []:
        targets.append(validate_tool_capability_selection_artifact(tool_capability_selection_path))
    for adapter_route_decision_path in adapter_route_decision_paths or []:
        targets.append(validate_adapter_route_decision_artifact(adapter_route_decision_path))
    for external_eval_plan_path in external_eval_plan_paths or []:
        targets.append(validate_external_eval_plan(external_eval_plan_path))
    for external_eval_receipt_path in external_eval_receipt_paths or []:
        targets.append(validate_external_eval_receipt(external_eval_receipt_path))
    for external_eval_result_path in external_eval_result_paths or []:
        targets.append(validate_external_eval_result(external_eval_result_path))
    for heldout_manifest_path in heldout_manifest_paths or []:
        targets.append(validate_heldout_manifest(heldout_manifest_path))
    for serving_profile_path in serving_profile_paths or []:
        targets.append(validate_serving_profile(serving_profile_path))
    for serving_compatibility_report_path in serving_compatibility_report_paths or []:
        targets.append(validate_serving_compatibility_report(serving_compatibility_report_path))
    for serving_endpoint_check_path in serving_endpoint_check_paths or []:
        targets.append(validate_serving_endpoint_check(serving_endpoint_check_path))
    for serving_lifecycle_path in serving_lifecycle_paths or []:
        targets.append(validate_serving_lifecycle(serving_lifecycle_path))
    for serving_demo_run_path in serving_demo_run_paths or []:
        targets.append(validate_serving_demo_run(serving_demo_run_path))
    if not targets:
        target = ValidationTarget("configuration", ".", errors=["No validation targets configured."])
        targets.append(target)

    error_count = sum(len(target.errors) for target in targets)
    warning_count = sum(len(target.warnings) for target in targets)
    passed = error_count == 0 and (warning_count == 0 or not strict)
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "passed": passed,
        "strict": strict,
        "target_count": len(targets),
        "error_count": error_count,
        "warning_count": warning_count,
        "targets": [target.as_dict() for target in targets],
    }
