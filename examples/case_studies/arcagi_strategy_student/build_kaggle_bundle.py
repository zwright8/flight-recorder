#!/usr/bin/env python3
"""Build private Kaggle dataset and kernel directories for the ARC route."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PEFT_IMPORT_UTILS_SHA256 = "c91d1933c155ad5a8e732d21feb4d35978c5c81ccfc173df83f01463accbcb3f"
TRAINING_SHA256 = "5e46e23ca114cc1e830e6a415f6277d44228a2feabf6655141232aa2ead5b58d"
TEACHER_SHA256 = "4e3396985cc35477c360c41b8d6a8068aa5d2c455d12b89f515e5418785815f1"
WHEEL_SHA256 = {
    "accelerate-1.14.0-py3-none-any.whl": "e94390c2863b873be18f623f9df48a0d8fe5eff13ea7f1a00092b0a7904888c6",
    "peft-0.19.1-py3-none-any.whl": "2113f72a81621b5913ef28f9022204c742df111890c5f49d812716a4a301e356",
}
MODEL_FILE_SHA256 = {
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
RUNTIME_FILES = (
    "pipeline.py",
    "evaluate_student.py",
    "kaggle_submission.py",
    "merge_kaggle_submissions.py",
    "score_kaggle_submission.py",
    "train_peft_strategy.py",
    "run_kaggle_route.py",
)
ARC_RUNTIME_FILES = (
    "ArcData.py",
    "ArcProblem.py",
    "ArcSet.py",
)
KAGGLE_MUTABLE_CONTROL_PLANE_FILES = frozenset({"dataset-metadata.json"})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_kaggle_id(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_-]*", value):
        raise ValueError("Kaggle identifiers must use owner/slug with lowercase letters, digits, _ or -")
    return value


def copy_file(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"required bundle source is not a file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The private upload tree must be independent of reviewed source artifacts.
    # A hard link would make chmod or any later upload-tree mutation affect the
    # original model/data inode as well.
    cloned = False
    if sys.platform == "darwin":
        clone = subprocess.run(
            ["/bin/cp", "-c", str(source), str(destination)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cloned = clone.returncode == 0
        if cloned:
            shutil.copystat(source, destination)
    if not cloned:
        shutil.copy2(source, destination)
    destination.chmod(0o600)


def copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        copy_file(path, destination / path.relative_to(source))


def inventory(root: Path, *, exclude: set[str] | None = None) -> dict[str, dict[str, Any]]:
    excluded = exclude or set()
    values = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        values[relative] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    return values


def sealed_payload_inventory(root: Path) -> dict[str, dict[str, Any]]:
    return inventory(
        root,
        exclude={"bundle_manifest.json", *KAGGLE_MUTABLE_CONTROL_PLANE_FILES},
    )


def verify_exact_file_set(root: Path, expected: dict[str, str], label: str) -> None:
    actual = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if actual != expected:
        raise ValueError(f"{label} files do not match the reviewed pinned set")


def notebook_payload() -> dict[str, Any]:
    source = [
        "from pathlib import Path\n",
        "import hashlib\n",
        "import json\n",
        "import os\n",
        "import subprocess\n",
        "import sys\n",
        "\n",
        "manifests = list(Path('/kaggle/input').rglob('bundle_manifest.json'))\n",
        "if len(manifests) != 1:\n",
        "    raise RuntimeError(f'expected one private HFR bundle, found {len(manifests)}')\n",
        "bundle = manifests[0].parent\n",
        "vendor = Path('/kaggle/working/hfr_arcagi_packages')\n",
        "wheels = sorted(str(path) for path in (bundle / 'wheels').glob('*.whl'))\n",
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', "
        "'--no-input', '--no-index', '--no-deps', '--target', str(vendor), *wheels], check=True)\n",
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
        "print(json.dumps({'compatibility_patch': 'disable_unused_torchao_dispatcher', "
        "'source_sha256': peft_before_sha, 'patched_sha256': "
        "hashlib.sha256(patched_peft.encode('utf-8')).hexdigest()}, sort_keys=True), flush=True)\n",
        "runner_relative = 'repo/examples/case_studies/arcagi_strategy_student/run_kaggle_route.py'\n",
        "runner_source_path = bundle / runner_relative\n",
        "runner_source = runner_source_path.read_text(encoding='utf-8')\n",
        "runner_before_sha = hashlib.sha256(runner_source.encode('utf-8')).hexdigest()\n",
        "manifest = json.loads((bundle / 'bundle_manifest.json').read_text(encoding='utf-8'))\n",
        "if runner_before_sha != manifest['files'][runner_relative]['sha256']:\n",
        "    raise RuntimeError('sealed Kaggle route source hash mismatch')\n",
        "compact_context_training = '        \"--max-seq-length\",\\n        \"4608\",\\n        \"--min-retained-fraction\",\\n        \"1.0\",\\n        \"--prompt-encoding\",\\n        \"compact-grid-v1\",\\n'\n",
        "if runner_source.count(compact_context_training) != 1:\n",
        "    raise RuntimeError('expected exactly one sealed compact-context training stanza')\n",
        "runner = Path('/kaggle/working/hfr_run_kaggle_route_gpu_compact_4608.py')\n",
        "runner.write_text(runner_source, encoding='utf-8')\n",
        "runner.chmod(0o700)\n",
        "runner_sha = hashlib.sha256(runner.read_bytes()).hexdigest()\n",
        "if runner_sha != runner_before_sha:\n",
        "    raise RuntimeError('compact-context runner copy hash mismatch')\n",
        "print(json.dumps({'runtime_contract': 'deterministic_t4_compact_context_suffix_logits', "
        "'max_seq_length': 4608, 'minimum_retained_fraction': 1.0, "
        "'prompt_encoding': 'compact-grid-v1', "
        "'source_sha256': runner_before_sha, 'runner_sha256': runner_sha}, sort_keys=True), flush=True)\n",
        "env = dict(os.environ)\n",
        "env['PYTHONPATH'] = os.pathsep.join([str(vendor), env.get('PYTHONPATH', '')])\n",
        "subprocess.run([sys.executable, str(runner), '--bundle-root', str(bundle)], check=True, env=env)\n",
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


def kernel_metadata_payload(
    *, dataset_id: str, kernel_id: str, competition: str, notebook_name: str
) -> dict[str, Any]:
    return {
        "id": kernel_id,
        # Keep the normalized title slug identical to kernel_id so Kaggle does
        # not silently create a differently named notebook resource.
        "title": "HFR ARCAGI strategy sealed v1",
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        # Kaggle's documented metadata contract represents these flags as
        # lowercase strings rather than JSON booleans.
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "dataset_sources": [dataset_id],
        "competition_sources": [competition],
        "kernel_sources": [],
        "model_sources": [],
    }


def build(args: argparse.Namespace) -> int:
    dataset_id = validate_kaggle_id(args.dataset_id)
    kernel_id = validate_kaggle_id(args.kernel_id)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("refusing to overwrite a non-empty Kaggle bundle directory")
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = args.out / "dataset"
    kernel = args.out / "kernel"
    dataset.mkdir()
    kernel.mkdir()

    if file_sha256(args.training_data) != TRAINING_SHA256:
        raise ValueError("training data does not match the reviewed governed ARC export")
    teacher_source = args.arc_root / "final_submission/ArcAgent.py"
    if file_sha256(teacher_source) != TEACHER_SHA256:
        raise ValueError("deterministic teacher does not match the reviewed ARC runtime")
    model_config = json.loads((args.base_model / "config.json").read_text(encoding="utf-8"))
    if model_config.get("model_type") != "qwen3":
        raise ValueError("base model is not the frozen Qwen3 family")
    verify_exact_file_set(args.base_model, MODEL_FILE_SHA256, "base model snapshot")

    case_source = args.repo_root / "examples/case_studies/arcagi_strategy_student"
    case_destination = dataset / "repo/examples/case_studies/arcagi_strategy_student"
    for name in RUNTIME_FILES:
        copy_file(case_source / name, case_destination / name)
    copy_tree(args.repo_root / "flightrecorder", dataset / "repo/flightrecorder")
    copy_file(teacher_source, dataset / "arc_runtime/final_submission/ArcAgent.py")
    for name in ARC_RUNTIME_FILES:
        copy_file(args.arc_root / name, dataset / "arc_runtime" / name)
    actual_wheels = {path.name: file_sha256(path) for path in args.wheel_dir.glob("*.whl")}
    if actual_wheels != WHEEL_SHA256:
        raise ValueError("offline Kaggle wheels do not match the reviewed dependency set")
    for name in sorted(WHEEL_SHA256):
        copy_file(args.wheel_dir / name, dataset / "wheels" / name)
    copy_file(args.training_data, dataset / "private_data/train.jsonl")
    copy_file(args.visible_challenges, dataset / "private_data/visible_challenges.json")
    copy_file(args.visible_solutions, dataset / "private_data/visible_solutions.json")
    copy_tree(args.base_model, dataset / "base_model")

    dataset_metadata = {
        "title": "HFR ARC strategy student private Kaggle runtime",
        "id": dataset_id,
        "licenses": [{"name": "other"}],
    }
    write_json(dataset / "dataset-metadata.json", dataset_metadata)
    # Kaggle rewrites dataset-metadata.json while ingesting a dataset.  Seal the
    # runtime payload, not provider-owned control-plane metadata whose bytes are
    # intentionally unstable after upload.
    files = sealed_payload_inventory(dataset)
    manifest = {
        "format_version": "hfr.arcagi.kaggle_private_bundle.v1",
        "dataset_id": dataset_id,
        "publication_allowed": False,
        "artifact_class": "sensitive-private-kaggle-input",
        "contains_visible_arc_outputs": True,
        "contains_credentials": False,
        "network_required_at_runtime": False,
        "offline_wheels": WHEEL_SHA256,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "files": MODEL_FILE_SHA256},
        "training_data_sha256": TRAINING_SHA256,
        "teacher_sha256": TEACHER_SHA256,
        "integrity_boundary": {
            "sealed_payload": "files",
            "excluded_control_plane_files": sorted(KAGGLE_MUTABLE_CONTROL_PLANE_FILES),
        },
        "files": files,
        "claim_boundary": "This private bundle contains visible-family imitation data; it is not a hidden ARC score.",
    }
    write_json(dataset / "bundle_manifest.json", manifest)

    notebook_name = "arcagi_kaggle_route.ipynb"
    write_json(kernel / notebook_name, notebook_payload())
    kernel_metadata = kernel_metadata_payload(
        dataset_id=dataset_id,
        kernel_id=kernel_id,
        competition=args.competition,
        notebook_name=notebook_name,
    )
    write_json(kernel / "kernel-metadata.json", kernel_metadata)
    for tree in (dataset, kernel):
        for path in tree.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        tree.chmod(0o700)
    print(
        json.dumps(
            {
                "dataset": str(dataset.resolve()),
                "kernel": str(kernel.resolve()),
                "file_count": len(files),
                "size_bytes": sum(row["size_bytes"] for row in files.values()),
                "private": True,
            },
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, required=True)
    result.add_argument("--arc-root", type=Path, required=True)
    result.add_argument("--training-data", type=Path, required=True)
    result.add_argument("--visible-challenges", type=Path, required=True)
    result.add_argument("--visible-solutions", type=Path, required=True)
    result.add_argument("--base-model", type=Path, required=True)
    result.add_argument("--wheel-dir", type=Path, required=True)
    result.add_argument("--dataset-id", required=True)
    result.add_argument("--kernel-id", required=True)
    result.add_argument("--competition", default="arc-prize-2026-arc-agi-2")
    result.add_argument("--out", type=Path, required=True)
    return result


if __name__ == "__main__":
    raise SystemExit(build(parser().parse_args()))
