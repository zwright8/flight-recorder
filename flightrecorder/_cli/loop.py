"""CLI commands for the loop domain."""

from __future__ import annotations

from ..loop_controller import CommandControllerAdapter, ControllerError, InMemoryControllerAdapter, build_controller_plan, run_controller
from ..agentic_loop_governance import GOVERNANCE_ACTIONS, AgenticLoopGovernanceReceiptError, build_agentic_loop_governance_receipt, write_agentic_loop_governance_receipt
from pathlib import Path
from ..runtime_adapter_router import RuntimeAdapterRouterError, build_adapter_route_decision, build_tool_capability_selection
import argparse
from ..review_semantics import ReviewSemanticsError, build_action_credit, build_branch_replay_dataset, build_contract_preferences, curate_training_rows
from ..cloud_training_completion import CloudTrainingCompletionError, build_cloud_training_completion_receipt, write_cloud_training_completion_receipt
from ..cloud_training import CloudTrainingError, build_cloud_training_artifact_manifest, build_cloud_training_launch_plan, build_cloud_training_launch_receipt, build_cloud_training_preflight, build_cloud_training_provider_registry, build_cloud_training_status_receipt, provider_choices as cloud_training_provider_choices
import json
from ..atomic_json import AtomicJsonError, atomic_write_json_cas, json_file_sha256
from ..intervention_router import InterventionRouterError, route_failure_cluster
from .evals import cmd_agentic_loop_governance, cmd_agentic_loop_ledger, cmd_agentic_loop_plan, cmd_cloud_training_artifacts, cmd_cloud_training_plan, cmd_cloud_training_preflight, cmd_cloud_training_providers
from .models import _emit_json_payload
from .shared import _artifact_ref_for_output_relative_file, _rate_arg, _read_json, _read_jsonl, _state_set_arg, _text_set_arg, _write_json, _write_jsonl, _write_new_json_atomically


def cmd_intervention_route(args: argparse.Namespace) -> int:
    route = route_failure_cluster(_read_json(Path(args.cluster)))
    _write_json(Path(args.out), route)
    print(f"wrote {args.out}")
    return 0


def cmd_runtime_router_tool_capabilities(args: argparse.Namespace) -> int:
    task_contract = _read_json(Path(args.task_contract))
    catalog = _read_json(Path(args.tool_catalog))
    policy = _read_json(Path(args.policy))
    environment = _read_json(Path(args.environment)) if args.environment else None
    tools = catalog.get("tools")
    if not isinstance(tools, list):
        raise RuntimeAdapterRouterError("tool catalog must be a JSON object with a tools array")
    artifact = build_tool_capability_selection(
        task_contract,
        tools,
        policy,
        environment=environment,
    )
    _write_new_json_atomically(Path(args.out), artifact)
    print(f"wrote {args.out}")
    return 0


def cmd_runtime_router_adapter(args: argparse.Namespace) -> int:
    task_contract = _read_json(Path(args.task_contract))
    capability_selection_path = Path(args.capability_selection)
    capability_selection = _read_json(capability_selection_path)
    catalog = _read_json(Path(args.candidate_catalog))
    routing_policy = _read_json(Path(args.routing_policy))
    runtime_environment = _read_json(Path(args.runtime_environment))
    candidates = catalog.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeAdapterRouterError("candidate catalog must be a JSON object with a candidates array")
    output_path = Path(args.out)
    capability_selection_ref = _artifact_ref_for_output_relative_file(
        capability_selection_path,
        output_path,
        label="capability selection",
    )
    artifact = build_adapter_route_decision(
        task_contract,
        capability_selection,
        candidates,
        routing_policy,
        runtime_environment=runtime_environment,
        capability_selection_sha256=capability_selection_ref["sha256"],
        capability_selection_ref=capability_selection_ref,
    )
    _write_new_json_atomically(output_path, artifact)
    print(f"wrote {args.out}")
    return 0


def cmd_review_semantics_action_credit(args: argparse.Namespace) -> int:
    credits = build_action_credit(_read_json(Path(args.trajectory)))
    _write_jsonl(Path(args.out), credits)
    print(f"wrote {args.out} rows={len(credits)}")
    return 0


def cmd_review_semantics_preferences(args: argparse.Namespace) -> int:
    preferences = build_contract_preferences(_read_jsonl(Path(args.input)))
    _write_jsonl(Path(args.out), preferences)
    print(f"wrote {args.out} rows={len(preferences)}")
    return 0


def cmd_review_semantics_branch_replay(args: argparse.Namespace) -> int:
    request = _read_json(Path(args.request))
    result = build_branch_replay_dataset(
        source_trajectory=request.get("source_trajectory", {}),
        replay_point=request.get("replay_point", {}),
        candidates=request.get("candidates", []),
        verifier_results=request.get("verifier_results", []),
        high_impact=request.get("high_impact") is True,
        novel_behavior=request.get("novel_behavior") is True,
        grader_disagreement=request.get("grader_disagreement") is True,
        review_confidence_threshold=float(request.get("review_confidence_threshold", 0.8)),
    )
    _write_json(Path(args.out), result)
    print(f"wrote {args.out}")
    return 0


def cmd_review_semantics_curate(args: argparse.Namespace) -> int:
    result = curate_training_rows(
        _read_jsonl(Path(args.input)),
        recipe=_read_json(Path(args.recipe)),
    )
    _write_json(Path(args.out), result)
    print(f"wrote {args.out} selected={result['selected_count']} excluded={result['excluded_count']}")
    return 0


def cmd_agentic_loop_controller_plan(args: argparse.Namespace) -> int:
    phase_overrides = {key: value for key, value in args.phase_config}
    if not all(isinstance(value, dict) for value in phase_overrides.values()):
        raise ControllerError("every --phase-config value must be a JSON object")
    plan = build_controller_plan(
        controller_id=args.controller_id,
        artifact_dir=args.artifact_dir,
        candidate_model=args.candidate_model,
        champion_model=args.champion_model,
        canary_percentages=args.canary_percentage,
        budget={
            "max_cost_usd": args.max_cost_usd,
            "max_duration_seconds": args.max_duration_seconds,
            "max_attempts": args.max_attempts,
        },
        deadline_at=args.deadline_at,
        canary_guardrails={
            "min_task_success_rate": args.min_task_success_rate,
            "max_critical_failures": args.max_critical_failures,
            "max_cost_delta": args.max_cost_delta,
            "max_latency_delta": args.max_latency_delta,
        },
        max_retries=args.max_retries,
        phase_overrides=phase_overrides,
    )
    _write_json(Path(args.out), plan)
    print(f"wrote {args.out}")
    return 0


def cmd_agentic_loop_execute(args: argparse.Namespace) -> int:
    plan = _read_json(Path(args.plan))
    approvals = dict(args.approval)
    if args.approve_all:
        approvals.update(
            {
                str(phase["id"]): str(plan.get("plan_fingerprint") or "")
                for phase in plan.get("phases", [])
                if isinstance(phase, dict) and phase.get("requires_approval") is True
            }
        )
        approvals["rollback"] = str(plan.get("plan_fingerprint") or "")
        approvals["post_rollback_smoke"] = str(plan.get("plan_fingerprint") or "")
    if args.adapter == "command":
        adapter = CommandControllerAdapter(allow_external=args.allow_external)
    else:
        outcomes = _read_json(Path(args.outcomes)) if args.outcomes else {}
        adapter = InMemoryControllerAdapter(outcomes=outcomes)
    state = run_controller(
        plan,
        state_path=args.state,
        adapter=adapter,
        approvals=approvals,
        owner_id=args.owner_id,
        max_steps=args.max_steps,
    )
    print(json.dumps({"status": state["status"], "state": args.state}, sort_keys=True))
    return 0 if state["status"] in {"complete", "rolled_back"} else 1


def cmd_cloud_training_launch(args: argparse.Namespace) -> int:
    receipt = build_cloud_training_launch_receipt(
        launch_plan_path=args.launch_plan,
        live=args.live,
        preserve_paths=args.preserve_paths,
        output_base_dir=Path(args.out).parent if args.out else None,
        created_at=args.created_at,
    )
    _emit_json_payload(receipt, args.out)
    return 0 if receipt["passed"] else 1


def cmd_cloud_training_status(args: argparse.Namespace) -> int:
    receipt = build_cloud_training_status_receipt(
        launch_receipt_path=args.launch_receipt,
        cancel_requested=args.cancel,
        preserve_paths=args.preserve_paths,
        output_base_dir=Path(args.out).parent if args.out else None,
        created_at=args.created_at,
    )
    _emit_json_payload(receipt, args.out)
    return 0 if receipt["passed"] else 1


def cmd_cloud_training_import_completion(args: argparse.Namespace) -> int:
    output_path = Path(args.out)
    expected_output_sha256 = json_file_sha256(output_path)
    receipt = build_cloud_training_completion_receipt(
        launch_plan_path=args.launch_plan,
        launch_receipt_path=args.launch_receipt,
        status_receipt_path=args.status_receipt,
        runner_metadata_path=args.runner_metadata,
        raw_provider_result_path=args.raw_provider_result,
        output_artifact_manifest_path=args.output_artifact_manifest,
        out_path=output_path,
        created_at=args.created_at,
    )
    write_cloud_training_completion_receipt(
        receipt,
        output_path,
        expected_sha256=expected_output_sha256,
    )
    print(f"wrote {args.out}")
    return 0 if receipt.get("passed") is True else 1


def register_loop_1(subparsers: argparse._SubParsersAction) -> None:
    intervention_route = subparsers.add_parser(
        "intervention-route",
        help="Route a failure cluster to the least-cost adequate agent intervention",
    )
    intervention_route.add_argument("--cluster", required=True, help="Failure cluster JSON")
    intervention_route.add_argument("--out", required=True, help="Write hfr.intervention_route.v1 JSON")
    intervention_route.set_defaults(func=cmd_intervention_route)

    runtime_router = subparsers.add_parser(
        "runtime-router",
        help="Build governed runtime tool-capability and adapter-route artifacts",
    )
    runtime_router_subparsers = runtime_router.add_subparsers(dest="runtime_router_command", required=True)
    runtime_router_tools = runtime_router_subparsers.add_parser(
        "tool-capabilities",
        help="Select task-visible tools from a governed tool catalog",
    )
    runtime_router_tools.add_argument("--task-contract", required=True, help="Trusted task contract JSON object")
    runtime_router_tools.add_argument("--tool-catalog", required=True, help="Tool catalog JSON object with a tools array")
    runtime_router_tools.add_argument("--policy", required=True, help="Tool routing policy JSON")
    runtime_router_tools.add_argument("--environment", help="Optional runtime environment JSON")
    runtime_router_tools.add_argument("--out", required=True, help="Write hfr.tool_capability_selection.v1 JSON")
    runtime_router_tools.set_defaults(func=cmd_runtime_router_tool_capabilities)

    runtime_router_adapter = runtime_router_subparsers.add_parser(
        "adapter",
        help="Route exactly one promoted adapter from a governed candidate catalog",
    )
    runtime_router_adapter.add_argument("--task-contract", required=True, help="Trusted task contract JSON object")
    runtime_router_adapter.add_argument("--capability-selection", required=True, help="Existing tool-capability selection JSON")
    runtime_router_adapter.add_argument("--candidate-catalog", required=True, help="Candidate catalog JSON object with a candidates array")
    runtime_router_adapter.add_argument("--routing-policy", required=True, help="Adapter routing policy JSON")
    runtime_router_adapter.add_argument("--runtime-environment", required=True, help="Runtime environment JSON")
    runtime_router_adapter.add_argument("--out", required=True, help="Write hfr.adapter_route_decision.v1 JSON")
    runtime_router_adapter.set_defaults(func=cmd_runtime_router_adapter)

    review_semantics = subparsers.add_parser(
        "review-semantics",
        help="Build native review, action-credit, branch-replay, and curation artifacts",
    )
    review_semantics_subparsers = review_semantics.add_subparsers(
        dest="review_semantics_command",
        required=True,
    )
    action_credit = review_semantics_subparsers.add_parser(
        "action-credit",
        help="Label each observed tool action from its matched result",
    )
    action_credit.add_argument("--trajectory", required=True, help="Native trajectory JSON")
    action_credit.add_argument("--out", required=True, help="Write hfr.action_credit.v1 JSONL")
    action_credit.set_defaults(func=cmd_review_semantics_action_credit)

    contract_preferences = review_semantics_subparsers.add_parser(
        "preferences",
        help="Pair reviewed completions only under an identical task contract",
    )
    contract_preferences.add_argument("--input", required=True, help="Reviewed trajectory JSONL")
    contract_preferences.add_argument("--out", required=True, help="Write reviewed contract preference JSONL")
    contract_preferences.set_defaults(func=cmd_review_semantics_preferences)

    branch_replay = review_semantics_subparsers.add_parser(
        "branch-replay",
        help="Bind state-matched replay candidates to verifier evidence and review routing",
    )
    branch_replay.add_argument("--request", required=True, help="Branch replay request JSON")
    branch_replay.add_argument("--out", required=True, help="Write hfr.branch_replay_dataset.v1 JSON")
    branch_replay.set_defaults(func=cmd_review_semantics_branch_replay)

    curate = review_semantics_subparsers.add_parser(
        "curate",
        help="Select a deterministic, capped, weighted training mixture",
    )
    curate.add_argument("--input", required=True, help="Candidate training rows JSONL")
    curate.add_argument("--recipe", required=True, help="Deterministic curation recipe JSON")
    curate.add_argument("--out", required=True, help="Write hfr.curated_dataset.v1 JSON")
    curate.set_defaults(func=cmd_review_semantics_curate)

    agentic_loop = subparsers.add_parser("agentic-loop", help="Plan closed-loop agentic training iterations")
    agentic_loop_subparsers = agentic_loop.add_subparsers(dest="agentic_loop_command", required=True)
    agentic_loop_plan = agentic_loop_subparsers.add_parser("plan", help="Write a fail-closed agentic training loop plan")
    agentic_loop_plan.add_argument("--iteration-id", required=True, help="Stable iteration id for this loop contract")
    agentic_loop_plan.add_argument("--out", required=True, help="Write hfr.agentic_training_loop_plan.v1 JSON to this path")
    agentic_loop_plan.add_argument("--objective", help="Human-readable iteration objective")
    agentic_loop_plan.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    agentic_loop_plan.add_argument("--baseline", help="Baseline policy/model id")
    agentic_loop_plan.add_argument("--candidate", help="Candidate policy/model id")
    agentic_loop_plan.add_argument("--teacher", help="Teacher policy/model id")
    agentic_loop_plan.add_argument("--provider", action="append", default=[], help="Allowed external trainer/provider id; may be repeated")
    agentic_loop_plan.add_argument("--region", action="append", default=[], help="Allowed cloud region; may be repeated")
    agentic_loop_plan.add_argument("--gpu-class", action="append", default=[], help="Allowed GPU class; may be repeated")
    agentic_loop_plan.add_argument("--budget", action="append", default=[], type=_state_set_arg, help="Attach budget KEY=JSON_VALUE; may be repeated")
    agentic_loop_plan.add_argument("--schedule", action="append", default=[], type=_state_set_arg, help="Attach next-iteration schedule KEY=JSON_VALUE; may be repeated")
    agentic_loop_plan.add_argument("--agentic-rollout-plan", action="append", default=[], help="agentic_rollout_plan artifact; may be repeated")
    agentic_loop_plan.add_argument("--agentic-rollout-receipt", action="append", default=[], help="agentic_rollout_receipt artifact; may be repeated")
    agentic_loop_plan.add_argument("--harness-manifest", action="append", default=[], help="harness_run_manifest artifact; may be repeated")
    agentic_loop_plan.add_argument("--harness-result", action="append", default=[], help="harness_run_result artifact; may be repeated")
    agentic_loop_plan.add_argument("--evidence-bundle", action="append", default=[], help="evidence_bundle artifact; may be repeated")
    agentic_loop_plan.add_argument("--review-calibration", action="append", default=[], help="review_calibration artifact; may be repeated")
    agentic_loop_plan.add_argument("--reviewed-gate", action="append", default=[], help="reviewed_gate artifact; may be repeated")
    agentic_loop_plan.add_argument("--rejection-sampling-gate", action="append", default=[], help="rejection_sampling_gate artifact; may be repeated")
    agentic_loop_plan.add_argument("--dataset-curation-receipt", action="append", default=[], help="dataset_curation_receipt artifact; may be repeated")
    agentic_loop_plan.add_argument("--training-export", action="append", default=[], help="export-rl directory; may be repeated")
    agentic_loop_plan.add_argument("--agentic-training-plan", action="append", default=[], help="agentic_training_plan artifact; may be repeated")
    agentic_loop_plan.add_argument(
        "--agentic-training-runtime-preflight",
        action="append",
        default=[],
        help="agentic_training_runtime_preflight artifact; may be repeated",
    )
    agentic_loop_plan.add_argument("--agentic-training-flow", action="append", default=[], help="agentic_training_flow artifact; may be repeated")
    agentic_loop_plan.add_argument("--agentic-training-result", action="append", default=[], help="agentic_training_result artifact; may be repeated")
    agentic_loop_plan.add_argument("--cloud-training-provider-registry", action="append", default=[], help="cloud_training_provider_registry artifact; may be repeated")
    agentic_loop_plan.add_argument("--cloud-training-preflight", action="append", default=[], help="cloud_training_preflight artifact; may be repeated")
    agentic_loop_plan.add_argument(
        "--cloud-training-artifact-manifest",
        action="append",
        default=[],
        help="cloud_training_artifact_manifest artifact; may be repeated",
    )
    agentic_loop_plan.add_argument("--cloud-training-launch-plan", action="append", default=[], help="cloud_training_launch_plan artifact; may be repeated")
    agentic_loop_plan.add_argument(
        "--cloud-training-launch-receipt",
        action="append",
        default=[],
        help="cloud_training_launch_receipt artifact; may be repeated",
    )
    agentic_loop_plan.add_argument("--cloud-training-status-receipt", action="append", default=[], help="cloud_training_status_receipt artifact; may be repeated")
    agentic_loop_plan.add_argument("--cloud-training-completion-receipt", action="append", default=[], help="cloud_training_completion_receipt artifact; may be repeated")
    agentic_loop_plan.add_argument("--trainer-preflight", action="append", default=[], help="trainer_preflight artifact; may be repeated")
    agentic_loop_plan.add_argument("--trainer-launch-check", action="append", default=[], help="trainer_launch_check artifact; may be repeated")
    agentic_loop_plan.add_argument("--serving-lifecycle", action="append", default=[], help="serving_lifecycle artifact; may be repeated")
    agentic_loop_plan.add_argument("--heldout-manifest", action="append", default=[], help="heldout manifest artifact; may be repeated")
    agentic_loop_plan.add_argument("--external-eval-plan", action="append", default=[], help="external_eval_plan artifact; may be repeated")
    agentic_loop_plan.add_argument("--external-eval-receipt", action="append", default=[], help="external_eval_receipt artifact; may be repeated")
    agentic_loop_plan.add_argument("--external-eval-result", action="append", default=[], help="external_eval_result artifact; may be repeated")
    agentic_loop_plan.add_argument("--eval-summary", action="append", default=[], help="eval_summary artifact; may be repeated")
    agentic_loop_plan.add_argument("--improvement-plan", action="append", default=[], help="improvement_plan artifact; may be repeated")
    agentic_loop_plan.add_argument("--improvement-ledger", action="append", default=[], help="improvement_ledger artifact; may be repeated")
    agentic_loop_plan.add_argument("--action-ledger", action="append", default=[], help="action_ledger artifact; may be repeated")
    agentic_loop_plan.add_argument(
        "--agentic-loop-governance-receipt",
        action="append",
        default=[],
        help="agentic_loop_governance_receipt artifact; may be repeated",
    )
    agentic_loop_plan.add_argument("--promotion-decision", action="append", default=[], help="promotion_decision artifact; may be repeated")
    agentic_loop_plan.add_argument("--promotion-ledger", action="append", default=[], help="promotion_ledger artifact; may be repeated")
    agentic_loop_plan.add_argument("--promotion-cards", action="append", default=[], help="promotion_cards artifact; may be repeated")
    agentic_loop_plan.add_argument("--promotion-alias-apply", action="append", default=[], help="promotion_alias_apply artifact; may be repeated")
    agentic_loop_plan.add_argument("--promotion-rollback-receipt", action="append", default=[], help="promotion_rollback_receipt artifact; may be repeated")
    agentic_loop_plan.add_argument("--promotion-release-record", action="append", default=[], help="promotion_release_record artifact; may be repeated")
    agentic_loop_plan.add_argument("--promotion-archive", action="append", default=[], help="promotion_archive artifact; may be repeated")
    agentic_loop_plan.add_argument("--next-iteration-schedule", action="append", default=[], help="next_iteration_schedule artifact; may be repeated")
    agentic_loop_plan.add_argument("--rubric-spec", action="append", default=[], help="rubric_spec artifact; may be repeated")
    agentic_loop_plan.add_argument("--model-grader-dry-run", action="append", default=[], help="model_grader_dry_run artifact; may be repeated")
    agentic_loop_plan.add_argument(
        "--model-grader-disagreement-queue",
        action="append",
        default=[],
        help="model_grader_disagreement_queue artifact; may be repeated",
    )
    agentic_loop_plan.add_argument(
        "--model-grader-override-receipt",
        action="append",
        default=[],
        help="model_grader_override_receipt artifact; may be repeated",
    )
    agentic_loop_plan.add_argument("--model-grader-gate", action="append", default=[], help="model_grader_gate artifact; may be repeated")
    agentic_loop_plan.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in plan output")
    agentic_loop_plan.set_defaults(func=cmd_agentic_loop_plan)

    agentic_loop_controller_plan = agentic_loop_subparsers.add_parser(
        "controller-plan",
        help="Write an opt-in resumable collect-to-canary controller plan",
    )
    agentic_loop_controller_plan.add_argument("--controller-id", required=True)
    agentic_loop_controller_plan.add_argument("--artifact-dir", required=True)
    agentic_loop_controller_plan.add_argument("--candidate-model", required=True)
    agentic_loop_controller_plan.add_argument("--champion-model", required=True)
    agentic_loop_controller_plan.add_argument(
        "--canary-percentage",
        type=int,
        action="append",
        default=[],
        help="Canary traffic percentage; repeat in increasing order",
    )
    agentic_loop_controller_plan.add_argument("--max-cost-usd", type=float, required=True)
    agentic_loop_controller_plan.add_argument("--max-duration-seconds", type=float, required=True)
    agentic_loop_controller_plan.add_argument("--max-attempts", type=int, required=True)
    agentic_loop_controller_plan.add_argument(
        "--deadline-at",
        required=True,
        help="Immutable absolute UTC deadline for all controller side effects",
    )
    agentic_loop_controller_plan.add_argument("--max-retries", type=int, default=2)
    agentic_loop_controller_plan.add_argument("--min-task-success-rate", type=_rate_arg, default=0.0)
    agentic_loop_controller_plan.add_argument("--max-critical-failures", type=int, default=0)
    agentic_loop_controller_plan.add_argument("--max-cost-delta", type=float, default=1.0)
    agentic_loop_controller_plan.add_argument("--max-latency-delta", type=float, default=1.0)
    agentic_loop_controller_plan.add_argument(
        "--phase-config",
        action="append",
        type=_state_set_arg,
        default=[],
        help="Override PHASE=JSON_OBJECT, including command argv/result path",
    )
    agentic_loop_controller_plan.add_argument("--out", required=True)
    agentic_loop_controller_plan.set_defaults(func=cmd_agentic_loop_controller_plan)

    agentic_loop_execute = agentic_loop_subparsers.add_parser(
        "execute",
        help="Run or resume an immutable controller plan",
    )
    agentic_loop_execute.add_argument("--plan", required=True)
    agentic_loop_execute.add_argument("--state", required=True)
    agentic_loop_execute.add_argument("--owner-id", required=True)
    agentic_loop_execute.add_argument("--adapter", choices=("in-memory", "command"), default="in-memory")
    agentic_loop_execute.add_argument("--outcomes", help="In-memory phase outcomes JSON")
    agentic_loop_execute.add_argument("--allow-external", action="store_true")
    agentic_loop_execute.add_argument(
        "--approval",
        action="append",
        type=_text_set_arg,
        default=[],
        help="Plan-bound PHASE=PLAN_FINGERPRINT approval; may be repeated",
    )
    agentic_loop_execute.add_argument(
        "--approve-all",
        action="store_true",
        help="Explicitly approve every side-effect phase in this exact immutable plan",
    )
    agentic_loop_execute.add_argument("--max-steps", type=int)
    agentic_loop_execute.set_defaults(func=cmd_agentic_loop_execute)

    agentic_loop_ledger = agentic_loop_subparsers.add_parser("ledger", help="Write a longitudinal ledger over loop plans")
    agentic_loop_ledger.add_argument("--plan", action="append", required=True, help="agentic_training_loop_plan JSON in chronological order; may be repeated")
    agentic_loop_ledger.add_argument("--out", help="Write hfr.agentic_loop_ledger.v1 JSON to this path")
    agentic_loop_ledger.add_argument(
        "--preserve-paths",
        action="store_true",
        help="Preserve source paths only when no ledger output file is written; ledger files require replayable relative plan paths",
    )
    agentic_loop_ledger.set_defaults(func=cmd_agentic_loop_ledger)

    agentic_loop_governance = agentic_loop_subparsers.add_parser(
        "governance",
        help="Write a side-effect-free governance receipt over the latest loop-ledger action",
    )
    agentic_loop_governance.add_argument("--ledger", required=True, help="agentic_loop_ledger JSON to govern")
    agentic_loop_governance.add_argument(
        "--action",
        required=True,
        choices=GOVERNANCE_ACTIONS,
        help="Governance action to record from the latest ledger action set",
    )
    agentic_loop_governance.add_argument("--reason", help="Human-readable reason for the governance action")
    agentic_loop_governance.add_argument("--requested-by", help="Reviewer, system, or process id requesting the action")
    agentic_loop_governance.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    agentic_loop_governance.add_argument("--out", help="Write hfr.agentic_loop_governance_receipt.v1 JSON to this path")
    agentic_loop_governance.add_argument(
        "--preserve-paths",
        action="store_true",
        help="Preserve source paths only when already public-safe relative; absolute source paths are redacted and fail closed",
    )
    agentic_loop_governance.set_defaults(func=cmd_agentic_loop_governance)



def register_loop_2(subparsers: argparse._SubParsersAction) -> None:
    cloud_training = subparsers.add_parser("cloud-training", help="Build fail-closed cloud training provider contracts")
    cloud_training_subparsers = cloud_training.add_subparsers(dest="cloud_training_command", required=True)
    cloud_training_providers = cloud_training_subparsers.add_parser("providers", help="Write cloud training provider registry")
    cloud_training_providers.add_argument("--provider", action="append", default=[], choices=cloud_training_provider_choices(), help="Provider id; may be repeated")
    cloud_training_providers.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    cloud_training_providers.add_argument("--out", help="Write provider registry JSON to this path")
    cloud_training_providers.set_defaults(func=cmd_cloud_training_providers)

    cloud_training_artifacts = cloud_training_subparsers.add_parser("artifacts", help="Write upload/download artifact manifest")
    cloud_training_artifacts.add_argument("--provider", required=True, choices=cloud_training_provider_choices(), help="Provider id")
    cloud_training_artifacts.add_argument("--upload", action="append", default=[], help="Upload artifact file path; may be repeated")
    cloud_training_artifacts.add_argument("--download", action="append", default=[], help="Expected provider output path; may be repeated")
    cloud_training_artifacts.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    cloud_training_artifacts.add_argument("--out", help="Write artifact manifest JSON to this path")
    cloud_training_artifacts.add_argument("--preserve-paths", action="store_true", help="Preserve safe source path text in artifact refs; unsafe absolute or traversal refs remain redacted")
    cloud_training_artifacts.set_defaults(func=cmd_cloud_training_artifacts)

    cloud_training_preflight = cloud_training_subparsers.add_parser("preflight", help="Write fail-closed provider preflight")
    cloud_training_preflight.add_argument("--provider", required=True, choices=cloud_training_provider_choices(), help="Provider id")
    cloud_training_preflight.add_argument("--agentic-training-plan", required=True, help="agentic_training_plan artifact")
    cloud_training_preflight.add_argument("--trainer-preflight", help="trainer_preflight artifact")
    cloud_training_preflight.add_argument("--trainer-launch-check", help="trainer_launch_check artifact")
    cloud_training_preflight.add_argument("--region", help="Requested provider region")
    cloud_training_preflight.add_argument("--gpu-class", help="Requested GPU class")
    cloud_training_preflight.add_argument("--max-cost-usd", type=float, help="Maximum allowed cloud cost for this launch")
    cloud_training_preflight.add_argument("--live-preflight", action="store_true", help="Probe credential/dependency readiness without calling provider APIs")
    cloud_training_preflight.add_argument("--live-requested", action="store_true", help="Record that a live launch was requested")
    cloud_training_preflight.add_argument("--allow-live", action="store_true", help="Explicit live opt-in for preflight only; no launch is performed")
    cloud_training_preflight.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    cloud_training_preflight.add_argument("--out", help="Write cloud training preflight JSON to this path")
    cloud_training_preflight.add_argument("--preserve-paths", action="store_true", help="Preserve safe source path text in artifact refs; unsafe absolute or traversal refs remain redacted")
    cloud_training_preflight.set_defaults(func=cmd_cloud_training_preflight)

    cloud_training_plan = cloud_training_subparsers.add_parser("plan", help="Write dry-run cloud launch plan")
    cloud_training_plan.add_argument("--preflight", required=True, help="cloud_training_preflight artifact")
    cloud_training_plan.add_argument("--artifact-manifest", help="cloud_training_artifact_manifest artifact")
    cloud_training_plan.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    cloud_training_plan.add_argument("--out", help="Write cloud training launch plan JSON to this path")
    cloud_training_plan.add_argument("--preserve-paths", action="store_true", help="Preserve safe source path text in artifact refs; unsafe absolute or traversal refs remain redacted")
    cloud_training_plan.set_defaults(func=cmd_cloud_training_plan)

    cloud_training_launch = cloud_training_subparsers.add_parser("launch", help="Write dry-run launch receipt or blocked live receipt")
    cloud_training_launch.add_argument("--launch-plan", required=True, help="cloud_training_launch_plan artifact")
    cloud_training_launch.add_argument("--live", action="store_true", help="Request a live launch receipt; remains blocked in this implementation")
    cloud_training_launch.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    cloud_training_launch.add_argument("--out", help="Write cloud training launch receipt JSON to this path")
    cloud_training_launch.add_argument("--preserve-paths", action="store_true", help="Preserve safe source path text in artifact refs; unsafe absolute or traversal refs remain redacted")
    cloud_training_launch.set_defaults(func=cmd_cloud_training_launch)

    cloud_training_status = cloud_training_subparsers.add_parser("status", help="Write dry-run status/cancel receipt")
    cloud_training_status.add_argument("--launch-receipt", required=True, help="cloud_training_launch_receipt artifact")
    cloud_training_status.add_argument("--cancel", action="store_true", help="Record a dry-run cancellation request")
    cloud_training_status.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    cloud_training_status.add_argument("--out", help="Write cloud training status receipt JSON to this path")
    cloud_training_status.add_argument("--preserve-paths", action="store_true", help="Preserve safe source path text in artifact refs; unsafe absolute or traversal refs remain redacted")
    cloud_training_status.set_defaults(func=cmd_cloud_training_status)

    cloud_training_completion = cloud_training_subparsers.add_parser(
        "import-completion",
        help="Import externally produced cloud training completion evidence without provider calls",
    )
    cloud_training_completion.add_argument("--launch-plan", required=True, help="cloud_training_launch_plan artifact")
    cloud_training_completion.add_argument("--launch-receipt", required=True, help="cloud_training_launch_receipt artifact")
    cloud_training_completion.add_argument("--status-receipt", required=True, help="cloud_training_status_receipt artifact")
    cloud_training_completion.add_argument("--runner-metadata", required=True, help="Public-safe external runner metadata JSON")
    cloud_training_completion.add_argument("--raw-provider-result", required=True, help="Opaque externally produced provider result file")
    cloud_training_completion.add_argument("--output-artifact-manifest", required=True, help="Direct agentic_training_result artifact describing exact outputs")
    cloud_training_completion.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    cloud_training_completion.add_argument("--out", required=True, help="Write cloud_training_completion_receipt JSON")
    cloud_training_completion.set_defaults(func=cmd_cloud_training_import_completion)

