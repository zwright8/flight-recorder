"""CLI commands for the evals domain."""

from __future__ import annotations

from ..external_eval_result import EXECUTION_STATUSES as EXTERNAL_EVAL_EXECUTION_STATUSES, FAILURE_CLASSES as EXTERNAL_EVAL_FAILURE_CLASSES, SUPPORTED_RAW_FORMATS as EXTERNAL_EVAL_RAW_FORMATS, ExternalEvalResultError, build_external_eval_result, write_external_eval_result
from pathlib import Path
from ..external_eval import ExternalEvalPlanError, adapter_choices, build_external_eval_plan, build_external_eval_receipt, write_external_eval_plan, write_external_eval_receipt
from ..data_governance import DataGovernanceError, apply_deletion_request, build_contamination_report, build_governance_receipt
import argparse
from ..agentic_loop_governance import GOVERNANCE_ACTIONS, AgenticLoopGovernanceReceiptError, build_agentic_loop_governance_receipt, write_agentic_loop_governance_receipt
from ..agentic_loop_ledger import AgenticLoopLedgerError, build_agentic_loop_ledger, write_agentic_loop_ledger
from ..agentic_training_loop_plan import AgenticTrainingLoopPlanError, build_agentic_training_loop_plan, write_agentic_training_loop_plan
from ..cloud_training import CloudTrainingError, build_cloud_training_artifact_manifest, build_cloud_training_launch_plan, build_cloud_training_launch_receipt, build_cloud_training_preflight, build_cloud_training_provider_registry, build_cloud_training_status_receipt, provider_choices as cloud_training_provider_choices
from ..eval_summary import EvalSummaryError, build_eval_summary, render_eval_summary_markdown
from ..next_iteration_schedule import NextIterationScheduleError, build_next_iteration_schedule, write_next_iteration_schedule
import json
from ..validation import EVAL_SUITE_MANIFEST_SCHEMA_VERSION, VALIDATION_SCHEMA_VERSION, validate_artifacts, validate_trainer_preflight
from .models import _emit_json_payload
from .shared import _rate_arg, _read_json, _read_jsonl, _state_set_arg, _write_json


def cmd_eval_summary(args: argparse.Namespace) -> int:
    summary = build_eval_summary(
        suite_summary_specs=args.suite_summary,
        compare_export_specs=args.compare_export,
        compare_gate_specs=args.compare_gate,
        external_adapter_plan_specs=args.external_adapter_plan,
        external_adapter_result_specs=args.external_adapter_result,
        serving_check_specs=args.serving_check,
        require_serving_preflight=args.require_serving_preflight,
        preserve_paths=args.preserve_paths,
        output_base_dir=Path(args.out).parent if args.out else None,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered, end="")
    if args.markdown_out:
        Path(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_out).write_text(render_eval_summary_markdown(summary), encoding="utf-8")
        print(f"wrote {args.markdown_out}")
    return 0 if summary["passed"] else 1


def cmd_external_eval_plan(args: argparse.Namespace) -> int:
    plan = build_external_eval_plan(
        adapters=args.adapter,
        scenario_manifest=args.scenario_manifest,
        model_endpoint=args.model_endpoint,
        model=args.model,
        tool_schema_set=args.tool_schema_set,
        inspect_task_set=args.inspect_task_set,
        lm_eval_task_list=args.lm_eval_task,
        swe_bench_task_set=args.swe_bench_task_set,
        sandbox_policy=args.sandbox_policy,
        allow_installed=args.allow_installed,
        preserve_paths=args.preserve_paths,
        output_base_dir=Path(args.out).parent if args.out else Path.cwd(),
    )
    if args.out:
        write_external_eval_plan(plan, args.out, preserve_paths=args.preserve_paths)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if plan["ready"] else 1


def cmd_external_eval_receipt(args: argparse.Namespace) -> int:
    receipt = build_external_eval_receipt(
        plan_path=args.plan,
        adapters=args.adapter,
        live=args.live,
        created_at=args.created_at,
        preserve_paths=args.preserve_paths,
        output_base_dir=Path(args.out).parent if args.out else None,
    )
    if args.out:
        write_external_eval_receipt(receipt, args.out, preserve_paths=args.preserve_paths)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if receipt["passed"] else 1


def cmd_external_eval_result(args: argparse.Namespace) -> int:
    result = build_external_eval_result(
        plan_path=args.plan,
        heldout_manifest_path=args.heldout_manifest,
        raw_result_path=args.raw_result,
        runner_metadata_path=args.runner_metadata,
        adapter_id=args.adapter,
        execution_id=args.execution_id,
        model_id=args.model_id,
        normalizer_id=args.normalizer_id,
        normalizer_version=args.normalizer_version,
        raw_format=args.raw_format,
        execution_status=args.status,
        failure_class=args.failure_class,
        failure_message=args.failure_message,
        out_path=args.out,
        created_at=args.created_at,
    )
    write_external_eval_result(result, args.out)
    print(f"wrote {args.out}")
    integrity = result.get("integrity") if isinstance(result.get("integrity"), dict) else {}
    return 0 if integrity.get("passed") is True else 1


def cmd_agentic_loop_plan(args: argparse.Namespace) -> int:
    artifact_paths = {
        "action_ledger": args.action_ledger,
        "agentic_rollout_plan": args.agentic_rollout_plan,
        "agentic_rollout_receipt": args.agentic_rollout_receipt,
        "agentic_training_plan": args.agentic_training_plan,
        "agentic_training_flow": args.agentic_training_flow,
        "agentic_training_result": args.agentic_training_result,
        "agentic_training_runtime_preflight": args.agentic_training_runtime_preflight,
        "cloud_training_artifact_manifest": args.cloud_training_artifact_manifest,
        "cloud_training_launch_plan": args.cloud_training_launch_plan,
        "cloud_training_launch_receipt": args.cloud_training_launch_receipt,
        "cloud_training_preflight": args.cloud_training_preflight,
        "cloud_training_provider_registry": args.cloud_training_provider_registry,
        "cloud_training_status_receipt": args.cloud_training_status_receipt,
        "cloud_training_completion_receipt": args.cloud_training_completion_receipt,
        "dataset_curation_receipt": args.dataset_curation_receipt,
        "evidence_bundle": args.evidence_bundle,
        "eval_summary": args.eval_summary,
        "external_eval_plan": args.external_eval_plan,
        "external_eval_receipt": args.external_eval_receipt,
        "external_eval_result": args.external_eval_result,
        "harness_manifest": args.harness_manifest,
        "harness_result": args.harness_result,
        "heldout_manifest": args.heldout_manifest,
        "improvement_ledger": args.improvement_ledger,
        "improvement_plan": args.improvement_plan,
        "agentic_loop_governance_receipt": args.agentic_loop_governance_receipt,
        "promotion_alias_apply": args.promotion_alias_apply,
        "promotion_archive": args.promotion_archive,
        "promotion_cards": args.promotion_cards,
        "promotion_decision": args.promotion_decision,
        "promotion_ledger": args.promotion_ledger,
        "promotion_release_record": args.promotion_release_record,
        "promotion_rollback_receipt": args.promotion_rollback_receipt,
        "rubric_spec": args.rubric_spec,
        "model_grader_dry_run": args.model_grader_dry_run,
        "model_grader_disagreement_queue": args.model_grader_disagreement_queue,
        "model_grader_override_receipt": args.model_grader_override_receipt,
        "model_grader_gate": args.model_grader_gate,
        "next_iteration_schedule": args.next_iteration_schedule,
        "review_calibration": args.review_calibration,
        "reviewed_gate": args.reviewed_gate,
        "rejection_sampling_gate": args.rejection_sampling_gate,
        "serving_lifecycle": args.serving_lifecycle,
        "trainer_launch_check": args.trainer_launch_check,
        "trainer_preflight": args.trainer_preflight,
        "training_export": args.training_export,
    }
    provider_constraints = {
        "providers": args.provider,
        "regions": args.region,
        "gpu_classes": args.gpu_class,
    }
    plan = build_agentic_training_loop_plan(
        out_path=args.out,
        iteration_id=args.iteration_id,
        objective=args.objective,
        candidate=args.candidate,
        baseline=args.baseline,
        teacher=args.teacher,
        artifact_paths=artifact_paths,
        budget=dict(args.budget or []),
        provider_constraints=provider_constraints,
        schedule=dict(args.schedule or []),
        preserve_paths=args.preserve_paths,
        created_at=args.created_at,
    )
    write_agentic_training_loop_plan(args.out, plan)
    print(
        f"wrote {args.out} readiness={plan['readiness']} "
        f"checks={plan['check_count'] - plan['failed_check_count']}/{plan['check_count']}"
    )
    return 0


def cmd_agentic_loop_ledger(args: argparse.Namespace) -> int:
    ledger = build_agentic_loop_ledger(
        args.plan,
        out_path=args.out,
        preserve_paths=args.preserve_paths,
    )
    if args.out:
        write_agentic_loop_ledger(args.out, ledger)
        print(
            f"wrote {args.out} iterations={ledger['iteration_count']} "
            f"latest={ledger['metrics']['latest_iteration_id']}"
        )
    else:
        print(json.dumps(ledger, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def cmd_agentic_loop_governance(args: argparse.Namespace) -> int:
    ledger_validation = validate_artifacts(agentic_loop_ledger_paths=[args.ledger], strict=True)
    receipt = build_agentic_loop_governance_receipt(
        ledger_path=args.ledger,
        action=args.action,
        reason=args.reason,
        requested_by=args.requested_by,
        out_path=args.out,
        preserve_paths=args.preserve_paths,
        created_at=args.created_at,
        source_ledger_replay_passed=ledger_validation.get("passed") is True,
    )
    if args.out:
        write_agentic_loop_governance_receipt(args.out, receipt)
        print(
            f"wrote {args.out} action={receipt['requested_action']['action']} "
            f"readiness={receipt['readiness']} recommendation={receipt['recommendation']}"
        )
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if receipt["passed"] else 1


def cmd_next_iteration_schedule(args: argparse.Namespace) -> int:
    schedule = build_next_iteration_schedule(
        loop_ledger_path=args.loop_ledger,
        action_ledger_path=args.action_ledger,
        improvement_ledger_path=args.improvement_ledger,
        next_iteration_id=args.next_iteration_id,
        objective=args.objective,
        schedule=dict(args.schedule or []),
        out_path=args.out,
        preserve_paths=args.preserve_paths,
        created_at=args.created_at,
    )
    write_next_iteration_schedule(args.out, schedule)
    print(
        f"wrote {args.out} readiness={schedule['readiness']} "
        f"recommendation={schedule['recommendation']}"
    )
    return 0 if schedule["passed"] else 1


def cmd_cloud_training_providers(args: argparse.Namespace) -> int:
    registry = build_cloud_training_provider_registry(provider_ids=args.provider, created_at=args.created_at)
    _emit_json_payload(registry, args.out)
    return 0


def cmd_cloud_training_artifacts(args: argparse.Namespace) -> int:
    manifest = build_cloud_training_artifact_manifest(
        provider_id=args.provider,
        upload_paths=args.upload,
        expected_downloads=args.download,
        preserve_paths=args.preserve_paths,
        output_base_dir=Path(args.out).parent if args.out else None,
        created_at=args.created_at,
    )
    _emit_json_payload(manifest, args.out)
    return 0 if manifest["passed"] else 1


def cmd_cloud_training_preflight(args: argparse.Namespace) -> int:
    preflight = build_cloud_training_preflight(
        provider_id=args.provider,
        agentic_training_plan_path=args.agentic_training_plan,
        trainer_preflight_path=args.trainer_preflight,
        trainer_launch_check_path=args.trainer_launch_check,
        region=args.region,
        gpu_class=args.gpu_class,
        max_cost_usd=args.max_cost_usd,
        live_preflight=args.live_preflight,
        live_requested=args.live_requested,
        allow_live=args.allow_live,
        preserve_paths=args.preserve_paths,
        output_base_dir=Path(args.out).parent if args.out else None,
        created_at=args.created_at,
    )
    _emit_json_payload(preflight, args.out)
    return 0 if preflight["passed"] else 1


def cmd_cloud_training_plan(args: argparse.Namespace) -> int:
    plan = build_cloud_training_launch_plan(
        preflight_path=args.preflight,
        artifact_manifest_path=args.artifact_manifest,
        preserve_paths=args.preserve_paths,
        output_base_dir=Path(args.out).parent if args.out else None,
        created_at=args.created_at,
    )
    _emit_json_payload(plan, args.out)
    return 0 if plan["passed"] else 1


def cmd_data_governance_check(args: argparse.Namespace) -> int:
    records = _read_jsonl(Path(args.input))
    receipt = build_governance_receipt(
        records,
        purpose=args.purpose,
        now=args.now,
        organization_entities=args.organization_entity,
        policy=_read_json(Path(args.policy)) if args.policy else None,
    )
    _write_json(Path(args.out), receipt)
    print(f"wrote {args.out}")
    return 0 if receipt["passed"] else 1


def cmd_data_governance_contamination(args: argparse.Namespace) -> int:
    rows = _read_jsonl(Path(args.input))
    protected = _read_jsonl(Path(args.protected)) if args.protected else []
    report = build_contamination_report(
        rows,
        protected_rows=protected,
        similarity_threshold=args.similarity_threshold,
    )
    _write_json(Path(args.out), report)
    print(f"wrote {args.out}")
    return 0 if report["passed"] else 1


def cmd_data_governance_delete(args: argparse.Namespace) -> int:
    request = _read_json(Path(args.request))
    receipt = apply_deletion_request(
        deletion_subject_ids=request.get("deletion_subject_ids", []),
        dataset_entries=request.get("dataset_entries", []),
        model_entries=request.get("model_entries", []),
        output_dir=args.out_dir,
        request_id=str(request.get("request_id") or Path(args.request).stem),
        erase_sources=request.get("erase_sources") is True,
        created_at=request.get("created_at"),
    )
    print(f"wrote {Path(args.out_dir) / 'deletion_receipt.json'}")
    return 0 if receipt["passed"] else 1


def register_evals_1(subparsers: argparse._SubParsersAction) -> None:
    external_eval_plan = subparsers.add_parser(
        "external-eval-plan",
        help="Plan fail-closed external BFCL/Inspect/lm-eval/SWE-bench adapter readiness",
    )
    external_eval_plan.add_argument(
        "--adapter",
        action="append",
        default=[],
        choices=adapter_choices(),
        help="External eval adapter to include; defaults to all supported adapters",
    )
    external_eval_plan.add_argument("--scenario-manifest", help="Held-out scenario manifest file shared by all external adapters")
    external_eval_plan.add_argument("--model-endpoint", help="Model endpoint or serving target used by external adapters")
    external_eval_plan.add_argument("--model", help="Model identifier included in adapter metadata")
    external_eval_plan.add_argument("--tool-schema-set", help="BFCL tool/function schema set identifier or file")
    external_eval_plan.add_argument("--inspect-task-set", help="Inspect AI task set identifier or file")
    external_eval_plan.add_argument("--lm-eval-task", action="append", default=[], help="lm-evaluation-harness task name; may be repeated")
    external_eval_plan.add_argument("--swe-bench-task-set", help="SWE-bench held-out task set identifier or file")
    external_eval_plan.add_argument("--sandbox-policy", help="Sandbox policy identifier or file for stateful external tasks")
    external_eval_plan.add_argument(
        "--allow-installed",
        action="store_true",
        help="Allow installed optional adapter dependencies to become ready when required inputs are present",
    )
    external_eval_plan.add_argument("--out", help="Write external eval adapter plan JSON to this path")
    external_eval_plan.add_argument(
        "--preserve-paths",
        action="store_true",
        help="Preserve safe source path text in plan output; unsafe absolute or traversal refs remain redacted",
    )
    external_eval_plan.set_defaults(func=cmd_external_eval_plan)

    external_eval_receipt = subparsers.add_parser(
        "external-eval-receipt",
        help="Archive a fail-closed dry-run receipt for an external eval adapter plan",
    )
    external_eval_receipt.add_argument("--plan", required=True, help="External eval plan JSON to bind")
    external_eval_receipt.add_argument(
        "--adapter",
        action="append",
        default=[],
        choices=adapter_choices(),
        help="External eval adapter to receipt; defaults to adapters selected by the plan",
    )
    external_eval_receipt.add_argument("--live", action="store_true", help="Record that live external eval was requested and blocked")
    external_eval_receipt.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    external_eval_receipt.add_argument("--out", help="Write external eval receipt JSON to this path")
    external_eval_receipt.add_argument("--preserve-paths", action="store_true", help="Preserve safe source path text in receipt output; unsafe absolute or traversal refs remain redacted")
    external_eval_receipt.set_defaults(func=cmd_external_eval_receipt)

    external_eval_result = subparsers.add_parser(
        "external-eval-result",
        help="Import and normalize externally executed benchmark evidence without running benchmark code",
    )
    external_eval_result.add_argument("--plan", required=True, help="Ready external eval plan JSON")
    external_eval_result.add_argument("--heldout-manifest", required=True, help="Held-out scenario manifest bound by the plan")
    external_eval_result.add_argument("--raw-result", required=True, help="Bounded JSON or JSONL output produced by the external runner")
    external_eval_result.add_argument("--runner-metadata", required=True, help="Public-safe runner metadata JSON")
    external_eval_result.add_argument("--adapter", required=True, choices=adapter_choices(), help="Adapter that produced the result")
    external_eval_result.add_argument("--execution-id", required=True, help="Public external execution identifier")
    external_eval_result.add_argument("--model-id", required=True, help="Model identity matching the external eval plan")
    external_eval_result.add_argument("--normalizer-id", required=True, help="Allowlisted result normalizer identifier")
    external_eval_result.add_argument("--normalizer-version", default="1", help="Allowlisted normalizer version")
    external_eval_result.add_argument("--raw-format", required=True, choices=EXTERNAL_EVAL_RAW_FORMATS, help="Raw result serialization contract")
    external_eval_result.add_argument("--status", required=True, choices=EXTERNAL_EVAL_EXECUTION_STATUSES, help="Externally reported execution status")
    external_eval_result.add_argument("--failure-class", default="none", choices=EXTERNAL_EVAL_FAILURE_CLASSES, help="Classified execution failure, if any")
    external_eval_result.add_argument("--failure-message", default="", help="Public-safe failure summary for incomplete or failed execution")
    external_eval_result.add_argument("--created-at", help="Override generated timestamp for deterministic receipts")
    external_eval_result.add_argument("--out", required=True, help="Write hfr.external_eval_result.v1 JSON")
    external_eval_result.set_defaults(func=cmd_external_eval_result)

    data_governance = subparsers.add_parser(
        "data-governance",
        help="Check training authorization, contamination, and deletion lineage",
    )
    data_governance_subparsers = data_governance.add_subparsers(
        dest="data_governance_command",
        required=True,
    )
    data_governance_check = data_governance_subparsers.add_parser(
        "check",
        help="Fail closed on missing governance metadata or unredacted personal data",
    )
    data_governance_check.add_argument("--input", required=True, help="Governed episode/training JSONL")
    data_governance_check.add_argument("--purpose", required=True, help="Requested allowed use, such as agent_training")
    data_governance_check.add_argument("--policy", help="Optional governance policy JSON")
    data_governance_check.add_argument(
        "--organization-entity",
        action="append",
        default=[],
        help="Organization-specific personal-data entity; may be repeated",
    )
    data_governance_check.add_argument("--now", help="Override evaluation time for deterministic tests")
    data_governance_check.add_argument("--out", required=True, help="Write hfr.data_governance_receipt.v1 JSON")
    data_governance_check.set_defaults(func=cmd_data_governance_check)

    data_governance_contamination = data_governance_subparsers.add_parser(
        "contamination",
        help="Cluster exact/near duplicates and detect protected-corpus leakage",
    )
    data_governance_contamination.add_argument("--input", required=True, help="Split-assigned training JSONL")
    data_governance_contamination.add_argument("--protected", help="Protected benchmark/canary JSONL")
    data_governance_contamination.add_argument(
        "--similarity-threshold",
        type=_rate_arg,
        default=0.9,
        help="Blocking similarity threshold from 0 to 1",
    )
    data_governance_contamination.add_argument("--out", required=True, help="Write contamination report JSON")
    data_governance_contamination.set_defaults(func=cmd_data_governance_contamination)

    data_governance_delete = data_governance_subparsers.add_parser(
        "delete",
        help="Erase authorized sources, rebuild descendants, and quarantine affected models",
    )
    data_governance_delete.add_argument("--request", required=True, help="Deletion request JSON with subjects/datasets/models")
    data_governance_delete.add_argument("--out-dir", required=True, help="Write rebuilt descendants and receipts here")
    data_governance_delete.set_defaults(func=cmd_data_governance_delete)



def register_evals_2(subparsers: argparse._SubParsersAction) -> None:
    next_iteration_schedule = subparsers.add_parser("next-iteration-schedule", help="Write a side-effect-free next-iteration schedule proposal")
    next_iteration_schedule.add_argument("--loop-ledger", required=True, help="agentic_loop_ledger artifact")
    next_iteration_schedule.add_argument("--action-ledger", required=True, help="action_ledger artifact")
    next_iteration_schedule.add_argument("--improvement-ledger", required=True, help="improvement_ledger artifact")
    next_iteration_schedule.add_argument("--next-iteration-id", help="Proposed next iteration id; defaults from latest loop id")
    next_iteration_schedule.add_argument("--objective", help="Human-readable next iteration objective")
    next_iteration_schedule.add_argument("--schedule", action="append", default=[], type=_state_set_arg, help="Attach schedule KEY=JSON_VALUE; may be repeated")
    next_iteration_schedule.add_argument("--created-at", help="Override generated timestamp for deterministic examples")
    next_iteration_schedule.add_argument("--out", required=True, help="Write hfr.next_iteration_schedule.v1 JSON to this path")
    next_iteration_schedule.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in schedule refs")
    next_iteration_schedule.set_defaults(func=cmd_next_iteration_schedule)



def register_evals_3(subparsers: argparse._SubParsersAction) -> None:
    eval_summary = subparsers.add_parser("eval-summary", help="Build a governance-ready held-out eval summary")
    eval_summary.add_argument(
        "--suite-summary",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="run-suite suite_summary.json to include; may be repeated",
    )
    eval_summary.add_argument(
        "--compare-export",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="export-compare-rl directory or manifest.json to summarize; may be repeated",
    )
    eval_summary.add_argument(
        "--compare-gate",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="gate-compare-export JSON output to include; may be repeated",
    )
    eval_summary.add_argument(
        "--external-adapter-plan",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="External eval adapter readiness plan JSON to include; may be repeated",
    )
    eval_summary.add_argument(
        "--external-adapter-result",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Imported hfr.external_eval_result.v1 evidence to include; may be repeated",
    )
    eval_summary.add_argument(
        "--serving-check",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="serving_check.json endpoint preflight to attach to a matching suite-summary label; may be repeated",
    )
    eval_summary.add_argument(
        "--require-serving-preflight",
        action="store_true",
        help="Block suite arms that do not have a ready serving_check.json preflight attached",
    )
    eval_summary.add_argument("--out", help="Write eval summary JSON to this path")
    eval_summary.add_argument("--markdown-out", help="Write a compact Markdown eval handoff report")
    eval_summary.add_argument("--preserve-paths", action="store_true", help="Allow absolute source paths in summary output")
    eval_summary.set_defaults(func=cmd_eval_summary)

