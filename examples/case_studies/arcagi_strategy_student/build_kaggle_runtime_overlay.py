#!/usr/bin/env python3
"""Build a small private Kaggle dataset containing reviewed ARC runtime code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_kaggle_bundle import PEFT_IMPORT_UTILS_SHA256, copy_file, validate_kaggle_id
from pipeline import file_sha256, require_fresh_output, secure_tree, write_json


RUNTIME_FILES = (
    "evaluate_student.py",
    "kaggle_submission.py",
    "merge_kaggle_submissions.py",
    "pipeline.py",
    "run_kaggle_comparison_route.py",
    "run_kaggle_route.py",
    "score_kaggle_submission.py",
    "train_peft_strategy.py",
)


def comparison_notebook_payload(runtime_manifest_sha256: str) -> dict[str, Any]:
    source = [
        "from pathlib import Path\n",
        "import hashlib\n",
        "import json\n",
        "import os\n",
        "import subprocess\n",
        "import sys\n",
        "\n",
        "bundle_manifests = list(Path('/kaggle/input').rglob('bundle_manifest.json'))\n",
        "if len(bundle_manifests) != 1:\n",
        "    raise RuntimeError(f'expected one private HFR bundle, found {len(bundle_manifests)}')\n",
        "bundle = bundle_manifests[0].parent\n",
        "runtime_manifests = list(Path('/kaggle/input').rglob('runtime_patch_manifest.json'))\n",
        "if len(runtime_manifests) != 1:\n",
        "    raise RuntimeError(f'expected one private comparison overlay, found {len(runtime_manifests)}')\n",
        "runtime_manifest_path = runtime_manifests[0]\n",
        "runtime_root = runtime_manifest_path.parent\n",
        "runtime_manifest_sha = hashlib.sha256(runtime_manifest_path.read_bytes()).hexdigest()\n",
        f"if runtime_manifest_sha != '{runtime_manifest_sha256}':\n",
        "    raise RuntimeError('private comparison overlay manifest hash mismatch')\n",
        "runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding='utf-8'))\n",
        "if runtime_manifest.get('format_version') != 'hfr.arcagi.kaggle_runtime_overlay.v1':\n",
        "    raise RuntimeError('unsupported comparison overlay format')\n",
        "if runtime_manifest.get('publication_allowed') is not False or runtime_manifest.get('contains_model_artifacts') is not True:\n",
        "    raise RuntimeError('comparison overlay privacy or adapter contract is invalid')\n",
        "if runtime_manifest.get('contains_raw_traces') is not False or runtime_manifest.get('contains_arc_tasks') is not False:\n",
        "    raise RuntimeError('comparison overlay contains forbidden sensitive inputs')\n",
        "runtime_files = runtime_manifest.get('files', {})\n",
        "for relative, expected in runtime_files.items():\n",
        "    path = runtime_root / relative\n",
        "    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected.get('sha256'):\n",
        "        raise RuntimeError(f'private comparison overlay integrity failure: {relative}')\n",
        "actual_runtime_files = {path.relative_to(runtime_root).as_posix() for path in runtime_root.rglob('*') if path.is_file() and path.name != 'runtime_patch_manifest.json'}\n",
        "if actual_runtime_files != set(runtime_files):\n",
        "    raise RuntimeError('private comparison overlay file set mismatch')\n",
        "runtime_case = runtime_root / 'case'\n",
        "adapter_root = runtime_root / 'adapter/checkpoint-0900'\n",
        "vendor = Path('/kaggle/working/hfr_arcagi_packages')\n",
        "wheels = sorted(str(path) for path in (bundle / 'wheels').glob('*.whl'))\n",
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-input', '--no-index', '--no-deps', '--target', str(vendor), *wheels], check=True)\n",
        "peft_probe = vendor / 'peft/import_utils.py'\n",
        "peft_source = peft_probe.read_text(encoding='utf-8')\n",
        "peft_before_sha = hashlib.sha256(peft_source.encode('utf-8')).hexdigest()\n",
        f"if peft_before_sha != '{PEFT_IMPORT_UTILS_SHA256}':\n",
        "    raise RuntimeError('pinned PEFT import-utils source hash mismatch')\n",
        "probe_start = '@lru_cache\\ndef is_torchao_available():'\n",
        "probe_end = '\\n\\n@lru_cache\\ndef is_xpu_available'\n",
        "start = peft_source.index(probe_start)\n",
        "end = peft_source.index(probe_end, start)\n",
        "disabled_probe = '@lru_cache\\ndef is_torchao_available():\\n    return False\\n'\n",
        "patched_peft = peft_source[:start] + disabled_probe + peft_source[end:]\n",
        "peft_probe.write_text(patched_peft, encoding='utf-8')\n",
        "print(json.dumps({'compatibility_patch': 'disable_unused_torchao_dispatcher', 'source_sha256': peft_before_sha, 'patched_sha256': hashlib.sha256(patched_peft.encode('utf-8')).hexdigest()}, sort_keys=True), flush=True)\n",
        "runner_relative = 'case/run_kaggle_comparison_route.py'\n",
        "runner_source_path = runtime_root / runner_relative\n",
        "runner_source = runner_source_path.read_text(encoding='utf-8')\n",
        "runner_before_sha = hashlib.sha256(runner_source.encode('utf-8')).hexdigest()\n",
        "if runner_before_sha != runtime_files[runner_relative]['sha256']:\n",
        "    raise RuntimeError('sealed comparison runner source hash mismatch')\n",
        "runner = Path('/kaggle/working/hfr_run_kaggle_controlled_comparison.py')\n",
        "runner.write_text(runner_source, encoding='utf-8')\n",
        "runner.chmod(0o700)\n",
        "runner_sha = hashlib.sha256(runner.read_bytes()).hexdigest()\n",
        "if runner_sha != runner_before_sha:\n",
        "    raise RuntimeError('controlled comparison runner copy hash mismatch')\n",
        "print(json.dumps({'runtime_contract': 'base_vs_hfr_lora_router_only_identical_executor', 'candidate_policy': 'router-only', 'runtime_overlay_manifest_sha256': runtime_manifest_sha, 'source_sha256': runner_before_sha, 'runner_sha256': runner_sha}, sort_keys=True), flush=True)\n",
        "env = dict(os.environ)\n",
        "env['PYTHONPATH'] = os.pathsep.join([str(runtime_case), str(vendor), env.get('PYTHONPATH', '')])\n",
        "subprocess.run([sys.executable, str(runner), '--bundle-root', str(bundle), '--runtime-case-dir', str(runtime_case), '--adapter-root', str(adapter_root)], check=True, env=env)\n",
    ]
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def comparison_kernel_metadata_payload(
    *, bundle_dataset_id: str, overlay_dataset_id: str, kernel_id: str, competition: str
) -> dict[str, Any]:
    return {
        "id": validate_kaggle_id(kernel_id),
        "title": "HFR ARCAGI controlled base LoRA comparison v1",
        "code_file": "arcagi_kaggle_comparison.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            validate_kaggle_id(bundle_dataset_id),
            validate_kaggle_id(overlay_dataset_id),
        ],
        "competition_sources": [competition],
        "kernel_sources": [],
        "model_sources": [],
    }


def build(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    case_dir = repo_root / "examples/case_studies/arcagi_strategy_student"
    dataset_id = validate_kaggle_id(args.dataset_id)
    out = args.out.resolve()
    require_fresh_output(out)

    runtime_dir = out / "case"
    for filename in RUNTIME_FILES:
        copy_file(case_dir / filename, runtime_dir / filename)
    files = {
        f"case/{filename}": {
            "sha256": file_sha256(runtime_dir / filename),
            "size_bytes": (runtime_dir / filename).stat().st_size,
        }
        for filename in RUNTIME_FILES
    }
    adapter_manifest = None
    adapter_root = getattr(args, "adapter_root", None)
    if adapter_root is not None:
        adapter_root = adapter_root.resolve()
        required_adapter_files = (
            "adapter_config.json",
            "adapter_model.safetensors",
            "training_receipt.json",
        )
        actual_adapter_files = {
            path.name for path in adapter_root.iterdir() if path.is_file()
        }
        if actual_adapter_files != set(required_adapter_files):
            raise ValueError("frozen adapter must contain exactly the reviewed three-file set")
        training_receipt = json.loads(
            (adapter_root / "training_receipt.json").read_text(encoding="utf-8")
        )
        if (
            training_receipt.get("status") != "succeeded"
            or training_receipt.get("target_steps") != 900
            or training_receipt.get("completed_steps") != 900
        ):
            raise ValueError("frozen adapter did not pass the exact 900-step gate")
        adapter_out = out / "adapter/checkpoint-0900"
        for filename in required_adapter_files:
            copy_file(adapter_root / filename, adapter_out / filename)
            relative = f"adapter/checkpoint-0900/{filename}"
            files[relative] = {
                "sha256": file_sha256(adapter_out / filename),
                "size_bytes": (adapter_out / filename).stat().st_size,
            }
        outputs = training_receipt.get("outputs", {})
        if outputs != {
            "adapter_config_sha256": files[
                "adapter/checkpoint-0900/adapter_config.json"
            ]["sha256"],
            "adapter_model_sha256": files[
                "adapter/checkpoint-0900/adapter_model.safetensors"
            ]["sha256"],
        }:
            raise ValueError("frozen adapter payload does not match its training receipt")
        adapter_manifest = {
            "checkpoint": "checkpoint-0900",
            "source_kernel": getattr(args, "adapter_source_kernel", None),
            "source_version": getattr(args, "adapter_source_version", None),
            "training_receipt_sha256": files[
                "adapter/checkpoint-0900/training_receipt.json"
            ]["sha256"],
        }
    manifest = {
        "format_version": "hfr.arcagi.kaggle_runtime_overlay.v1",
        "publication_allowed": False,
        "contains_raw_traces": False,
        "contains_arc_tasks": False,
        "contains_model_artifacts": adapter_manifest is not None,
        "adapter": adapter_manifest,
        "files": files,
    }
    write_json(out / "runtime_patch_manifest.json", manifest)
    write_json(
        out / "dataset-metadata.json",
        {
            "id": dataset_id,
            "title": "HFR ARC strategy student private runtime overlay",
            "licenses": [{"name": "other"}],
        },
    )
    kernel_out = getattr(args, "kernel_out", None)
    if kernel_out is not None:
        if adapter_manifest is None:
            raise ValueError("a controlled comparison kernel requires a frozen adapter")
        kernel_out = kernel_out.resolve()
        require_fresh_output(kernel_out)
        notebook_name = "arcagi_kaggle_comparison.ipynb"
        write_json(
            kernel_out / notebook_name,
            comparison_notebook_payload(file_sha256(out / "runtime_patch_manifest.json")),
        )
        write_json(
            kernel_out / "kernel-metadata.json",
            comparison_kernel_metadata_payload(
                bundle_dataset_id=args.bundle_dataset_id,
                overlay_dataset_id=dataset_id,
                kernel_id=args.kernel_id,
                competition=args.competition,
            ),
        )
        secure_tree(kernel_out)
    secure_tree(out)
    print(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "file_count": len(files),
                "manifest_sha256": file_sha256(out / "runtime_patch_manifest.json"),
                "out": str(out),
            },
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, required=True)
    result.add_argument("--dataset-id", required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--adapter-root", type=Path)
    result.add_argument("--adapter-source-kernel")
    result.add_argument("--adapter-source-version", type=int)
    result.add_argument("--kernel-out", type=Path)
    result.add_argument("--kernel-id")
    result.add_argument("--bundle-dataset-id")
    result.add_argument("--competition", default="arc-prize-2026-arc-agi-2")
    return result


if __name__ == "__main__":
    raise SystemExit(build(parser().parse_args()))
