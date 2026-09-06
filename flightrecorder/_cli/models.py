"""CLI commands for the models domain."""

from __future__ import annotations

from ..model_registry import ALIAS_NAMES, MODEL_ADAPTER_MANIFEST_STATUSES, MODEL_REGISTRY_LINK_COLLECTIONS, ModelRegistryError, build_model_adapter_manifest, build_dry_run_training_plan, build_model_compatibility_report, build_model_serving_probe_receipt, link_model_registry_artifact, list_model_registry_entries, load_model_registry, model_candidate_errors, move_model_alias, register_model_candidate
from typing import Any, Iterator
from pathlib import Path
import argparse
from ..atomic_json import AtomicJsonError, atomic_write_json_cas, json_file_sha256
import json
from ..validation import EVAL_SUITE_MANIFEST_SCHEMA_VERSION, VALIDATION_SCHEMA_VERSION, validate_artifacts, validate_trainer_preflight
from .shared import _metadata_arg, _metadata_options, _read_json, _state_set_arg, _write_json


def cmd_model_scout_validate(args: argparse.Namespace) -> int:
    summary = validate_artifacts(model_scout_manifest_paths=[args.manifest], strict=args.strict)
    _emit_json_payload(summary, args.out)
    return 0 if summary["passed"] else 1


def cmd_model_candidate_validate(args: argparse.Namespace) -> int:
    candidate_path = Path(args.candidate)
    candidate = _read_json(candidate_path)
    errors = model_candidate_errors(candidate, require_training_eligible=args.require_training_eligible)
    target = {
        "type": "model_candidate",
        "path": str(candidate_path),
        "passed": not errors,
        "errors": errors,
        "warnings": [],
        "details": {
            "candidate_id": candidate.get("candidate_id") if isinstance(candidate, dict) else None,
            "model_id": candidate.get("model_id") if isinstance(candidate, dict) else None,
            "require_training_eligible": args.require_training_eligible,
        },
    }
    summary = {
        "schema_version": "hfr.validation.v1",
        "passed": not errors,
        "strict": False,
        "target_count": 1,
        "error_count": len(errors),
        "warning_count": 0,
        "targets": [target],
    }
    _emit_json_payload(summary, args.out)
    return 0 if summary["passed"] else 1


def cmd_model_candidate_compatibility_report(args: argparse.Namespace) -> int:
    candidate = _read_json(Path(args.candidate))
    report = build_model_compatibility_report(candidate, out_path=args.out, preserve_paths=args.preserve_paths)
    _write_json(Path(args.out), report)
    print(f"wrote {args.out}")
    return 0 if report["passed"] else 1


def cmd_model_registry_validate(args: argparse.Namespace) -> int:
    summary = validate_artifacts(model_registry_paths=[args.registry], strict=args.strict)
    _emit_json_payload(summary, args.out)
    return 0 if summary["passed"] else 1


def cmd_model_registry_register(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry)
    registry_sha256 = json_file_sha256(registry_path)
    registry = load_model_registry(registry_path)
    candidate = _read_json(Path(args.candidate))
    registry = register_model_candidate(registry, candidate, status=args.status)
    atomic_write_json_cas(registry_path, registry, expected_sha256=registry_sha256)
    entry = registry["entries"][candidate["candidate_id"]]
    if args.entry_out:
        _write_json(Path(args.entry_out), entry)
    print(f"registered {candidate['candidate_id']} in {registry_path}")
    return 0


def cmd_model_registry_list(args: argparse.Namespace) -> int:
    registry = load_model_registry(args.registry)
    rows = list_model_registry_entries(registry)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False))
    elif rows:
        print("entry_id\tmodel_id\tstatus\ttraining_eligible\tlicense_status\taliases")
        for row in rows:
            print(
                "\t".join(
                    [
                        str(row["entry_id"]),
                        str(row["model_id"]),
                        str(row["status"]),
                        str(row["training_eligible"]).lower(),
                        str(row["license_status"]),
                        ",".join(row["aliases"]),
                    ]
                )
            )
    else:
        print("no model registry entries")
    return 0


def cmd_model_registry_alias(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry)
    registry_sha256 = json_file_sha256(registry_path)
    registry = load_model_registry(registry_path)
    registry = move_model_alias(
        registry,
        alias=args.alias,
        target=args.target,
        rollback_target=args.rollback_target,
        reason=args.reason or "",
    )
    atomic_write_json_cas(registry_path, registry, expected_sha256=registry_sha256)
    print(f"moved {args.alias} -> {args.target} in {registry_path}")
    return 0


def cmd_model_registry_link(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry)
    registry_sha256 = json_file_sha256(registry_path)
    registry = load_model_registry(registry_path)
    registry = link_model_registry_artifact(
        registry,
        entry_id=args.entry,
        collection=args.collection,
        artifact_id=args.artifact_id,
        kind=args.kind,
        status=args.status,
        path=args.path,
        sha256=args.sha256,
        metadata=_metadata_options(args.metadata),
        preserve_paths=args.preserve_paths,
    )
    atomic_write_json_cas(registry_path, registry, expected_sha256=registry_sha256)
    if args.entry_out:
        _write_json(Path(args.entry_out), registry["entries"][args.entry])
    print(f"linked {args.collection}:{args.artifact_id} to {args.entry} in {registry_path}")
    return 0


def cmd_model_registry_serving_probe_receipt(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry)
    registry_sha256 = json_file_sha256(registry_path)
    registry = load_model_registry(registry_path)
    compatibility_report = _read_json(Path(args.compatibility_report)) if args.compatibility_report else None
    receipt = build_model_serving_probe_receipt(
        registry,
        model_ref=args.model_ref,
        out_path=args.out,
        profile_id=args.profile_id,
        provider=args.provider,
        serving_engine=args.serving_engine,
        base_url=args.base_url,
        probe_mode=args.probe_mode,
        compatibility_report=compatibility_report,
        compatibility_report_path=args.compatibility_report,
        preserve_paths=args.preserve_paths,
    )
    receipt_path = Path(args.out)
    _write_json(receipt_path, receipt)
    if args.link:
        artifact_id = args.artifact_id or receipt_path.stem
        registry = link_model_registry_artifact(
            registry,
            entry_id=receipt["entry_id"],
            collection="serving_probes",
            artifact_id=artifact_id,
            kind="model_serving_probe_receipt",
            status=args.link_status,
            path=receipt_path,
            metadata={
                "probe_mode": receipt["probe_mode"],
                "readiness": receipt["readiness"],
                "provider": receipt["serving_profile"]["provider"],
                "serving_engine": receipt["serving_profile"]["serving_engine"],
            },
            preserve_paths=args.preserve_paths,
        )
        atomic_write_json_cas(registry_path, registry, expected_sha256=registry_sha256)
        if args.entry_out:
            _write_json(Path(args.entry_out), registry["entries"][receipt["entry_id"]])
    print(f"wrote {args.out}")
    return 0 if receipt["passed"] else 1


def cmd_model_registry_adapter_manifest(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry)
    registry_sha256 = json_file_sha256(registry_path)
    registry = load_model_registry(registry_path)
    training_plan = _read_json(Path(args.training_plan))
    manifest = build_model_adapter_manifest(
        registry,
        model_ref=args.model_ref,
        adapter_id=args.adapter_id,
        training_plan=training_plan,
        training_plan_path=args.training_plan,
        out_path=args.out,
        adapter_kind=args.kind,
        status=args.status,
        output_dir=args.output_dir,
        preserve_paths=args.preserve_paths,
    )
    manifest_path = Path(args.out)
    _write_json(manifest_path, manifest)
    if args.link:
        registry = link_model_registry_artifact(
            registry,
            entry_id=manifest["base_model"]["entry_id"],
            collection="adapters",
            artifact_id=manifest["registry_link"]["artifact_id"],
            kind=manifest["registry_link"]["kind"],
            status=args.link_status,
            path=manifest_path,
            metadata={
                "adapter_kind": manifest["adapter_kind"],
                "readiness": manifest["readiness"],
                "training_plan": manifest["training_plan"]["path"],
                "training_plan_sha256": manifest["training_plan"]["sha256"],
            },
            preserve_paths=args.preserve_paths,
        )
        atomic_write_json_cas(registry_path, registry, expected_sha256=registry_sha256)
        if args.entry_out:
            _write_json(Path(args.entry_out), registry["entries"][manifest["base_model"]["entry_id"]])
    print(f"wrote {args.out}")
    return 0 if manifest["passed"] else 1


def cmd_training_plan_dry_run(args: argparse.Namespace) -> int:
    registry = load_model_registry(args.registry)
    compatibility_report = _read_json(Path(args.compatibility_report)) if args.compatibility_report else None
    plan = build_dry_run_training_plan(
        registry,
        model_ref=args.model_ref,
        dataset_id=args.dataset_id,
        dataset_manifest=args.dataset_manifest,
        trainer=args.trainer,
        mode=args.mode,
        output_dir=args.output_dir,
        out_path=args.out,
        hyperparameters=dict(args.hyperparameter),
        compute=dict(args.compute),
        compatibility_report=compatibility_report,
        compatibility_report_path=args.compatibility_report,
        preserve_paths=args.preserve_paths,
    )
    _write_json(Path(args.out), plan)
    print(f"wrote {args.out}")
    return 0


def _emit_json_payload(payload: dict[str, Any], out: str | None) -> None:
    if out:
        _write_json(Path(out), payload)
        print(f"wrote {out}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def register_models_1(subparsers: argparse._SubParsersAction) -> None:
    model_scout = subparsers.add_parser("model-scout", help="Validate model-scout manifests")
    model_scout_subparsers = model_scout.add_subparsers(dest="model_scout_command", required=True)
    model_scout_validate = model_scout_subparsers.add_parser("validate", help="Validate a model-scout manifest")
    model_scout_validate.add_argument("manifest", help="Path to model_scout_manifest.json")
    model_scout_validate.add_argument("--out", help="Write validation summary JSON to this path")
    model_scout_validate.add_argument("--strict", action="store_true", help="Treat warnings as validation failure")
    model_scout_validate.set_defaults(func=cmd_model_scout_validate)

    model_candidate = subparsers.add_parser("model-candidate", help="Validate and report model-candidate artifacts")
    model_candidate_subparsers = model_candidate.add_subparsers(dest="model_candidate_command", required=True)
    model_candidate_validate = model_candidate_subparsers.add_parser("validate", help="Validate one model candidate")
    model_candidate_validate.add_argument("candidate", help="Path to model candidate JSON")
    model_candidate_validate.add_argument(
        "--require-training-eligible",
        action="store_true",
        help="Require license and terms posture that allows training selection",
    )
    model_candidate_validate.add_argument("--out", help="Write validation summary JSON to this path")
    model_candidate_validate.set_defaults(func=cmd_model_candidate_validate)
    model_candidate_report = model_candidate_subparsers.add_parser(
        "compatibility-report",
        help="Write metadata-only compatibility report for a model candidate",
    )
    model_candidate_report.add_argument("--candidate", required=True, help="Path to model candidate JSON")
    model_candidate_report.add_argument("--out", required=True, help="Write model compatibility report JSON to this path")
    model_candidate_report.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in generated report")
    model_candidate_report.set_defaults(func=cmd_model_candidate_compatibility_report)

    model_registry = subparsers.add_parser("model-registry", help="Manage the local model registry")
    model_registry_subparsers = model_registry.add_subparsers(dest="model_registry_command", required=True)
    model_registry_validate = model_registry_subparsers.add_parser("validate", help="Validate a model registry")
    model_registry_validate.add_argument("--registry", default="experiments/registry/model_registry.json", help="Path to model_registry.json")
    model_registry_validate.add_argument("--out", help="Write validation summary JSON to this path")
    model_registry_validate.add_argument("--strict", action="store_true", help="Treat warnings as validation failure")
    model_registry_validate.set_defaults(func=cmd_model_registry_validate)
    model_registry_register = model_registry_subparsers.add_parser("register", help="Register or update a model candidate")
    model_registry_register.add_argument("--registry", default="experiments/registry/model_registry.json", help="Path to model_registry.json")
    model_registry_register.add_argument("--candidate", required=True, help="Path to model candidate JSON")
    model_registry_register.add_argument("--status", default="registered", help="Registry entry status")
    model_registry_register.add_argument("--entry-out", help="Optionally write the resulting registry entry JSON")
    model_registry_register.set_defaults(func=cmd_model_registry_register)
    model_registry_list = model_registry_subparsers.add_parser("list", help="List registered model candidates")
    model_registry_list.add_argument("--registry", default="experiments/registry/model_registry.json", help="Path to model_registry.json")
    model_registry_list.add_argument("--json", action="store_true", help="Print machine-readable JSON rows")
    model_registry_list.set_defaults(func=cmd_model_registry_list)
    model_registry_alias = model_registry_subparsers.add_parser("alias", help="Move candidate, champion, or rollback aliases")
    model_registry_alias.add_argument("--registry", default="experiments/registry/model_registry.json", help="Path to model_registry.json")
    model_registry_alias.add_argument("--alias", choices=list(ALIAS_NAMES), required=True, help="Alias to move")
    model_registry_alias.add_argument("--target", required=True, help="Registry entry id to target")
    model_registry_alias.add_argument("--rollback-target", help="Required when moving champion")
    model_registry_alias.add_argument("--reason", default="", help="Reason recorded in alias history")
    model_registry_alias.set_defaults(func=cmd_model_registry_alias)
    model_registry_link = model_registry_subparsers.add_parser("link", help="Link dataset, training, adapter, eval, serving, or promotion artifacts")
    model_registry_link.add_argument("--registry", default="experiments/registry/model_registry.json", help="Path to model_registry.json")
    model_registry_link.add_argument("--entry", required=True, help="Model registry entry id to update")
    model_registry_link.add_argument("--collection", choices=list(MODEL_REGISTRY_LINK_COLLECTIONS), required=True, help="Link collection to update")
    model_registry_link.add_argument("--artifact-id", required=True, help="Stable linked artifact id")
    model_registry_link.add_argument("--kind", required=True, help="Linked artifact kind")
    model_registry_link.add_argument("--status", default="recorded", help="Linked artifact lifecycle status")
    model_registry_link.add_argument("--path", help="Optional local artifact path to hash and record")
    model_registry_link.add_argument("--sha256", help="Optional SHA-256 digest for pathless artifact refs or path verification")
    model_registry_link.add_argument("--entry-out", help="Optionally write the updated registry entry JSON")
    model_registry_link.add_argument("--metadata", action="append", default=[], type=_metadata_arg, help="Attach link metadata KEY=VALUE; may be repeated")
    model_registry_link.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in link records")
    model_registry_link.set_defaults(func=cmd_model_registry_link)
    model_registry_serving_probe = model_registry_subparsers.add_parser(
        "serving-probe-receipt",
        help="Write a no-download model serving-probe receipt and optionally link it to the registry",
    )
    model_registry_serving_probe.add_argument("--registry", default="experiments/registry/model_registry.json", help="Path to model_registry.json")
    model_registry_serving_probe.add_argument("--model-ref", required=True, help="Registry entry id or alias to resolve")
    model_registry_serving_probe.add_argument("--out", required=True, help="Write model serving-probe receipt JSON to this path")
    model_registry_serving_probe.add_argument("--profile-id", required=True, help="Stable serving profile id")
    model_registry_serving_probe.add_argument("--provider", required=True, help="Serving provider label, such as metadata_only or local")
    model_registry_serving_probe.add_argument("--serving-engine", required=True, help="Serving engine label, such as vllm-compatible")
    model_registry_serving_probe.add_argument("--base-url", required=True, help="Endpoint URL or metadata-only placeholder")
    model_registry_serving_probe.add_argument(
        "--probe-mode",
        choices=["metadata_only", "external_receipt"],
        default="metadata_only",
        help="Receipt mode; metadata_only records no endpoint execution",
    )
    model_registry_serving_probe.add_argument("--compatibility-report", help="Optional model compatibility report JSON to bind by hash")
    model_registry_serving_probe.add_argument("--link", action="store_true", help="Link the written receipt under registry links.serving_probes")
    model_registry_serving_probe.add_argument("--artifact-id", help="Stable linked artifact id; defaults to output filename stem")
    model_registry_serving_probe.add_argument("--link-status", default="metadata_receipt", help="Registry link lifecycle status")
    model_registry_serving_probe.add_argument("--entry-out", help="Optionally write the updated registry entry JSON when --link is used")
    model_registry_serving_probe.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in generated receipt and link records")
    model_registry_serving_probe.set_defaults(func=cmd_model_registry_serving_probe_receipt)
    model_registry_adapter = model_registry_subparsers.add_parser(
        "adapter-manifest",
        help="Write a no-download planned-adapter manifest and optionally link it to the registry",
    )
    model_registry_adapter.add_argument("--registry", default="experiments/registry/model_registry.json", help="Path to model_registry.json")
    model_registry_adapter.add_argument("--model-ref", required=True, help="Registry entry id or alias to resolve")
    model_registry_adapter.add_argument("--adapter-id", required=True, help="Stable planned adapter id")
    model_registry_adapter.add_argument("--kind", default="lora", help="Adapter kind, such as lora or qlora")
    model_registry_adapter.add_argument("--status", choices=sorted(MODEL_ADAPTER_MANIFEST_STATUSES), default="planned", help="Adapter manifest lifecycle status")
    model_registry_adapter.add_argument("--training-plan", required=True, help="Dry-run training plan JSON to fingerprint")
    model_registry_adapter.add_argument("--output-dir", help="Planned adapter output directory; defaults to training_plan.output.output_dir")
    model_registry_adapter.add_argument("--out", required=True, help="Write model adapter manifest JSON to this path")
    model_registry_adapter.add_argument("--link", action="store_true", help="Link the written manifest under registry links.adapters")
    model_registry_adapter.add_argument("--link-status", default="planned_adapter", help="Registry link lifecycle status")
    model_registry_adapter.add_argument("--entry-out", help="Optionally write the updated registry entry JSON when --link is used")
    model_registry_adapter.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in generated manifest and link records")
    model_registry_adapter.set_defaults(func=cmd_model_registry_adapter_manifest)

    training_plan = subparsers.add_parser("training-plan", help="Generate dry-run training plans")
    training_plan_subparsers = training_plan.add_subparsers(dest="training_plan_command", required=True)
    training_plan_dry_run = training_plan_subparsers.add_parser(
        "dry-run",
        help="Write a registry-backed dry-run training plan without downloads or GPU work",
    )
    training_plan_dry_run.add_argument("--registry", default="experiments/registry/model_registry.json", help="Path to model_registry.json")
    training_plan_dry_run.add_argument("--model-ref", required=True, help="Registry entry id or alias, such as candidate")
    training_plan_dry_run.add_argument("--dataset-id", required=True, help="Dataset version id to record in the plan")
    training_plan_dry_run.add_argument("--dataset-manifest", required=True, help="Dataset manifest file to fingerprint")
    training_plan_dry_run.add_argument("--trainer", required=True, help="Trainer or recipe name")
    training_plan_dry_run.add_argument("--mode", required=True, help="Training mode, such as sft, action_sft, dpo, or sft_then_dpo")
    training_plan_dry_run.add_argument("--output-dir", required=True, help="Planned trainer output directory")
    training_plan_dry_run.add_argument("--out", required=True, help="Write dry-run training plan JSON to this path")
    training_plan_dry_run.add_argument("--compatibility-report", help="Optional model compatibility report JSON to bind by hash")
    training_plan_dry_run.add_argument(
        "--hyperparameter",
        action="append",
        default=[],
        type=_state_set_arg,
        help="Attach trainer hyperparameter KEY=JSON_VALUE; may be repeated",
    )
    training_plan_dry_run.add_argument(
        "--compute",
        action="append",
        default=[],
        type=_state_set_arg,
        help="Attach compute assumption KEY=JSON_VALUE; may be repeated",
    )
    training_plan_dry_run.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in generated plan")
    training_plan_dry_run.set_defaults(func=cmd_training_plan_dry_run)

