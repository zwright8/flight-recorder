#!/usr/bin/env python3
"""Run an offline MLX LoRA job and write a hash-bound local receipt."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline import canonical_sha256, file_sha256, require_fresh_output, secure_tree, write_json


def main(args: argparse.Namespace) -> int:
    model = args.model.resolve()
    data = args.data.resolve()
    adapter = args.adapter.resolve()
    if not model.is_dir():
        raise ValueError(f"offline model path is not a directory: {model}")
    resume_adapter = args.resume_adapter_file.resolve() if args.resume_adapter_file else None
    if resume_adapter is not None and not resume_adapter.is_file():
        raise ValueError(f"resume adapter file not found: {resume_adapter}")
    for split in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        if not (data / split).is_file():
            raise ValueError(f"missing MLX dataset split: {data / split}")
    require_fresh_output(adapter)
    executable = Path(args.mlx_python).resolve()
    if not executable.is_file():
        raise ValueError(f"MLX Python executable not found: {executable}")
    single_process_entrypoint = Path(__file__).with_name("mlx_lora_single.py").resolve()

    command = [
        str(executable),
        str(single_process_entrypoint),
        "--model",
        str(model),
        "--train",
        "--data",
        str(data),
        "--fine-tune-type",
        "lora",
        "--mask-prompt",
        "--num-layers",
        str(args.num_layers),
        "--batch-size",
        str(args.batch_size),
        "--grad-accumulation-steps",
        str(args.grad_accumulation_steps),
        "--iters",
        str(args.iters),
        "--learning-rate",
        str(args.learning_rate),
        "--steps-per-report",
        str(args.steps_per_report),
        "--steps-per-eval",
        str(args.steps_per_eval),
        "--val-batches",
        "-1",
        "--max-seq-length",
        str(args.max_seq_length),
        "--save-every",
        str(args.save_every),
        "--seed",
        str(args.seed),
        "--adapter-path",
        str(adapter),
    ]
    if args.grad_checkpoint:
        command.append("--grad-checkpoint")
    if resume_adapter is not None:
        command.extend(["--resume-adapter-file", str(resume_adapter)])
    started = datetime.now(timezone.utc).isoformat()
    log_path = adapter / "training.log"
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["MLX_RANK"] = "0"
    env["MLX_SIZE"] = "1"
    interrupted = False
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            return_code = process.wait()
            log.write("\ntraining wrapper interrupted; child process terminated with SIGINT\n")
            log.flush()
            return_code = 130
    receipt = {
        "schema_version": "hfr.arcagi.local_training_receipt.v1",
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "return_code": return_code,
        "interrupted": interrupted,
        "offline_enforced": True,
        "training_backend": "mlx_lm.lora with explicit single-process ring backend",
        "model_path": str(model),
        "resume_adapter": {
            "path": str(resume_adapter),
            "sha256": file_sha256(resume_adapter),
            "size_bytes": resume_adapter.stat().st_size,
        }
        if resume_adapter is not None
        else None,
        "data_path": str(data),
        "data_files": {
            split: {"sha256": file_sha256(data / split), "size_bytes": (data / split).stat().st_size}
            for split in ("train.jsonl", "valid.jsonl", "test.jsonl")
        },
        "configuration": {
            "fine_tune_type": "lora",
            "mask_prompt": True,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "grad_accumulation_steps": args.grad_accumulation_steps,
            "iters": args.iters,
            "learning_rate": args.learning_rate,
            "max_seq_length": args.max_seq_length,
            "seed": args.seed,
            "grad_checkpoint": args.grad_checkpoint,
        },
        "command_sha256": canonical_sha256(command),
        "log_sha256": file_sha256(log_path),
        "adapter_files": {},
    }
    for path in sorted(adapter.iterdir()):
        if path.is_file() and path.name != "training_receipt.json":
            receipt["adapter_files"][path.name] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    write_json(adapter / "training_receipt.json", receipt)
    secure_tree(adapter)
    if return_code != 0:
        raise RuntimeError(f"MLX LoRA training failed with exit code {return_code}; see {log_path}")
    print(json.dumps({"adapter": str(adapter), "receipt": str(adapter / "training_receipt.json")}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--adapter", type=Path, required=True)
    result.add_argument("--mlx-python", default=sys.executable)
    result.add_argument("--num-layers", type=int, default=16)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--grad-accumulation-steps", type=int, default=4)
    result.add_argument("--iters", type=int, default=600)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--steps-per-report", type=int, default=20)
    result.add_argument("--steps-per-eval", type=int, default=100)
    result.add_argument("--max-seq-length", type=int, default=8192)
    result.add_argument("--save-every", type=int, default=200)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--grad-checkpoint", action="store_true")
    result.add_argument("--resume-adapter-file", type=Path)
    return result


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
