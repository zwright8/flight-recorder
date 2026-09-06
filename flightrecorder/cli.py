"""Command line interface for Flight Recorder."""

from __future__ import annotations

import argparse
from .action_ledger import ActionLedgerError, build_action_ledger
from .action_gate import ACTION_LEDGER_GATE_POLICY_SCHEMA_VERSION, ActionLedgerGateError, ActionLedgerGatePolicyError, evaluate_action_ledger_gate, load_action_ledger_gate_policy
from .adapters import AdapterError, normalize_trace
from .agentic_loop_governance import GOVERNANCE_ACTIONS, AgenticLoopGovernanceReceiptError, build_agentic_loop_governance_receipt, write_agentic_loop_governance_receipt
from .agentic_loop_ledger import AgenticLoopLedgerError, build_agentic_loop_ledger, write_agentic_loop_ledger
from .agentic_training_flow import AgenticTrainingFlowError, build_agentic_training_flow, write_agentic_training_flow
from .agentic_training_loop_plan import AgenticTrainingLoopPlanError, build_agentic_training_loop_plan, write_agentic_training_loop_plan
from .artifacts import ArtifactError, build_suite_trend, compare_scorecards, compare_suites, write_compare_report, write_junit, write_markdown_summary, write_suite_compare_report, write_suite_trend_report
from .atomic_json import AtomicJsonError, atomic_write_json_cas, json_file_sha256
from .cloud_training_completion import CloudTrainingCompletionError, build_cloud_training_completion_receipt, write_cloud_training_completion_receipt
from .cloud_training import CloudTrainingError, build_cloud_training_artifact_manifest, build_cloud_training_launch_plan, build_cloud_training_launch_receipt, build_cloud_training_preflight, build_cloud_training_provider_registry, build_cloud_training_status_receipt, provider_choices as cloud_training_provider_choices
from .compare_gate import COMPARE_GATE_POLICY_SCHEMA_VERSION, CompareGatePolicyError, evaluate_compare_gate, load_compare_gate_policy
from .loop_controller import CommandControllerAdapter, ControllerError, InMemoryControllerAdapter, build_controller_plan, run_controller
from .data_governance import DataGovernanceError, apply_deletion_request, build_contamination_report, build_governance_receipt
from .dataset_curation import DatasetCurationReceiptError, build_dataset_curation_receipt, write_dataset_curation_receipt
from .decision_gate import DecisionGateError, evaluate_decision_gate, reject_symlinked_decision_artifact_input
from .eval_summary import EvalSummaryError, build_eval_summary, render_eval_summary_markdown
from .bundle import HARNESS_RUN_MANIFEST_SCHEMA_VERSION, HARNESS_RUN_RESULT_SCHEMA_VERSION, EvidenceBundleError, build_evidence_bundle
from .evidence import EvidenceCoverageError, build_evidence_coverage
from .external_eval import ExternalEvalPlanError, adapter_choices, build_external_eval_plan, build_external_eval_receipt, write_external_eval_plan, write_external_eval_receipt
from .external_eval_result import EXECUTION_STATUSES as EXTERNAL_EVAL_EXECUTION_STATUSES, FAILURE_CLASSES as EXTERNAL_EVAL_FAILURE_CLASSES, SUPPORTED_RAW_FORMATS as EXTERNAL_EVAL_RAW_FORMATS, ExternalEvalResultError, build_external_eval_result, write_external_eval_result
from .heldout_manifest import HeldoutManifestError, build_heldout_manifest, write_heldout_manifest
from .improvement_ledger import ImprovementLedgerError, build_improvement_ledger
from .improvement_gate import IMPROVEMENT_LEDGER_GATE_POLICY_SCHEMA_VERSION, ImprovementLedgerGateError, ImprovementLedgerGatePolicyError, evaluate_improvement_ledger_gate, load_improvement_ledger_gate_policy
from .improvement_plan import ImprovementPlanError, build_improvement_plan
from .intervention_router import InterventionRouterError, route_failure_cluster
from .model_grader import ModelGraderError, build_model_grader_disagreement_queue, build_model_grader_dry_run, build_model_grader_gate, build_model_grader_override_receipt, build_rubric_spec, write_model_grader_artifact
from .model_registry import ALIAS_NAMES, MODEL_ADAPTER_MANIFEST_STATUSES, MODEL_REGISTRY_LINK_COLLECTIONS, ModelRegistryError, build_model_adapter_manifest, build_dry_run_training_plan, build_model_compatibility_report, build_model_serving_probe_receipt, link_model_registry_artifact, list_model_registry_entries, load_model_registry, model_candidate_errors, move_model_alias, register_model_candidate
from .next_iteration_schedule import NextIterationScheduleError, build_next_iteration_schedule, write_next_iteration_schedule
from .promotion_archive import PromotionArchiveError, build_promotion_archive
from .governance import PromotionDecisionError, apply_promotion_aliases, build_promotion_cards, build_promotion_decision, build_promotion_release_record, build_promotion_rollback_receipt
from .promotion_ledger import PromotionLedgerError, build_promotion_ledger
from .promotion_gate import PROMOTION_LEDGER_GATE_POLICY_SCHEMA_VERSION, PromotionLedgerGateError, PromotionLedgerGatePolicyError, evaluate_promotion_ledger_gate, load_promotion_ledger_gate_policy
from .rejection_sampling import RejectionSamplingGateError, build_rejection_sampling_gate, write_rejection_sampling_gate
from .repair import RepairQueueError, build_repair_queue
from .calibration import ReviewCalibrationError, build_review_calibration, write_review_calibration
from .review import REVIEW_LABELS, ReviewExportError, apply_review_labels, export_review_queue
from .review_semantics import ReviewSemanticsError, build_action_credit, build_branch_replay_dataset, build_contract_preferences, curate_training_rows
from .reviewed_gate import REVIEWED_GATE_POLICY_SCHEMA_VERSION, ReviewedGateError, ReviewedGatePolicyError, build_reviewed_export_source_artifact, evaluate_reviewed_gate, load_reviewed_gate_policy, snapshot_reviewed_export
from .rollout_generation import RolloutGenerationError, build_agentic_rollout_plan, build_agentic_rollout_receipt, write_agentic_rollout_plan, write_agentic_rollout_receipt
from .digest import RunDigestError, build_run_digest, render_run_digest_markdown
from .runtime_adapter_router import RuntimeAdapterRouterError, build_adapter_route_decision, build_tool_capability_selection
from .schema import ScenarioError, load_scenario, resolve_trace_path
from .schema_registry import SchemaRegistryError, check_schema_file, check_schema_jsonl_file, list_schema_records, load_schema, write_schema_bundle
from .state_capture import StateCaptureError, capture_state_snapshot
from .state_diff import StateDiffError, build_state_diff
from .state import StateSnapshotError, load_state_snapshot, resolve_before_state_snapshot_path, resolve_state_snapshot_path, sanitize_state_snapshot
from .state_validators import StateValidatorError, build_monitor_catalog, build_state_validator_assertions, render_monitor_catalog_markdown
from .suite_gate import SUITE_GATE_POLICY_SCHEMA_VERSION, SuiteGateError, SuiteGatePolicyError, evaluate_suite_gate, load_gate_policy
from .trace_observability import TraceObservabilityError, build_trace_observability
from .trainer_archive_check import TrainerArchiveCheckError, build_trainer_archive_check
from .trainer_archive import TrainerArchiveError, build_trainer_archive
from .trainer_consumer_plan import TrainerConsumerPlanError, build_trainer_consumer_plan, reject_symlinked_archive_check_input
from .preflight import TrainerPreflightError, build_trainer_launch_check, build_trainer_preflight, snapshot_trainer_preflight
from .training import TrainingExportError, export_compare_rl_dataset, export_rl_dataset
from .training_gate import TRAINING_GATE_POLICY_SCHEMA_VERSION, TrainingGatePolicyError, evaluate_training_gate, load_training_gate_policy
from .verifiers import VerifierError, capture_verified_state
import json
from ._cli.parser import build_parser
from ._cli.artifacts import cmd_audit, cmd_check_scenarios, cmd_compare, cmd_compare_suite, cmd_index, cmd_observer_template, cmd_scenario_quality, cmd_trend_suite, cmd_validate
from ._cli.basic import cmd_capture_state, cmd_diff_state, cmd_digest, cmd_normalize, cmd_report, cmd_score, cmd_state_validators, cmd_verify_state
from ._cli.evals import cmd_agentic_loop_governance, cmd_agentic_loop_ledger, cmd_agentic_loop_plan, cmd_cloud_training_artifacts, cmd_cloud_training_plan, cmd_cloud_training_preflight, cmd_cloud_training_providers, cmd_data_governance_check, cmd_data_governance_contamination, cmd_data_governance_delete, cmd_eval_summary, cmd_external_eval_plan, cmd_external_eval_receipt, cmd_external_eval_result, cmd_next_iteration_schedule
from ._cli.evidence import cmd_action_ledger, cmd_evidence_bundle, cmd_evidence_coverage, cmd_improvement_ledger, cmd_improvement_plan, cmd_promotion_alias_apply, cmd_promotion_cards, cmd_promotion_rollback_receipt, cmd_repair_queue, cmd_schemas, cmd_trace_observability
from ._cli.exports import cmd_export_compare_rl, cmd_export_rl
from ._cli.gates import cmd_gate_action_ledger, cmd_gate_compare_export, cmd_gate_decision, cmd_gate_export, cmd_gate_improvement_ledger, cmd_gate_promotion_ledger, cmd_gate_reviewed, cmd_gate_suite, cmd_trainer_launch_check, cmd_trainer_preflight
from ._cli.generation import cmd_agentic_rollout_plan, cmd_agentic_rollout_receipt, cmd_dataset_curation_receipt, cmd_heldout_manifest, cmd_model_grader_disagreement_queue, cmd_model_grader_dry_run, cmd_model_grader_gate, cmd_model_grader_override_receipt, cmd_model_grader_rubric, cmd_rejection_sampling_gate
from ._cli.governance import cmd_agentic_training_flow, cmd_apply_review, cmd_draft_scenario, cmd_export_review, cmd_promotion_archive, cmd_promotion_decision, cmd_promotion_ledger, cmd_promotion_release_record, cmd_review_calibration, cmd_trainer_archive, cmd_trainer_archive_check, cmd_trainer_consumer_plan
from ._cli.loop import cmd_agentic_loop_controller_plan, cmd_agentic_loop_execute, cmd_cloud_training_import_completion, cmd_cloud_training_launch, cmd_cloud_training_status, cmd_intervention_route, cmd_review_semantics_action_credit, cmd_review_semantics_branch_replay, cmd_review_semantics_curate, cmd_review_semantics_preferences, cmd_runtime_router_adapter, cmd_runtime_router_tool_capabilities
from ._cli.models import cmd_model_candidate_compatibility_report, cmd_model_candidate_validate, cmd_model_registry_adapter_manifest, cmd_model_registry_alias, cmd_model_registry_link, cmd_model_registry_list, cmd_model_registry_register, cmd_model_registry_serving_probe_receipt, cmd_model_registry_validate, cmd_model_scout_validate, cmd_training_plan_dry_run
from ._cli.replay_core import ReplayError
from ._cli.run_commands import cmd_replay, cmd_replay_bundle, cmd_run
from ._cli.run_core import _is_owned_run_directory, _run_scenario_artifacts
from ._cli.shared import _write_json
from ._cli.suite import cmd_goal3_handoff, cmd_run_suite
from ._cli.suite_core import _failed_rule_ids, _lineage_input_hash, _run_suite_summary, _safe_run_id
from .hashing import sha256_file as _sha256_file


def _parser() -> argparse.ArgumentParser:
    return build_parser()


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        AdapterError,
        ArtifactError,
        ScenarioError,
        StateCaptureError,
        StateValidatorError,
        VerifierError,
        AtomicJsonError,
        StateDiffError,
        StateSnapshotError,
        SuiteGateError,
        SuiteGatePolicyError,
        ReviewExportError,
        ReviewedGateError,
        ReviewedGatePolicyError,
        RolloutGenerationError,
        RepairQueueError,
        TrainerPreflightError,
        TrainerArchiveError,
        TrainerArchiveCheckError,
        TrainerConsumerPlanError,
        TrainingExportError,
        TrainingGatePolicyError,
        CompareGatePolicyError,
        DecisionGateError,
        RunDigestError,
        EvidenceCoverageError,
        EvidenceBundleError,

        PromotionDecisionError,
        EvalSummaryError,
        ExternalEvalPlanError,
        ExternalEvalResultError,
        HeldoutManifestError,

        ReviewCalibrationError,
        CloudTrainingError,
        CloudTrainingCompletionError,
        TraceObservabilityError,
        ActionLedgerError,
        ActionLedgerGateError,
        ActionLedgerGatePolicyError,
        ImprovementLedgerGateError,
        ImprovementLedgerGatePolicyError,
        ImprovementLedgerError,
        ImprovementPlanError,
        PromotionLedgerGateError,
        PromotionLedgerGatePolicyError,
        PromotionLedgerError,
        PromotionArchiveError,
        AgenticTrainingLoopPlanError,
        AgenticLoopLedgerError,
        AgenticLoopGovernanceReceiptError,
        ModelGraderError,
        RejectionSamplingGateError,
        DatasetCurationReceiptError,
        DataGovernanceError,
        InterventionRouterError,
        ReviewSemanticsError,
        ControllerError,
        AgenticTrainingFlowError,
        NextIterationScheduleError,
        ReplayError,
        SchemaRegistryError,
        ModelRegistryError,
        RuntimeAdapterRouterError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        parser.exit(2, f"flightrecorder: error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "flightrecorder: interrupted\n")
    return 0
