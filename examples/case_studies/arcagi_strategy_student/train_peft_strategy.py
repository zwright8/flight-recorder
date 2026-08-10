#!/usr/bin/env python3
"""Train or calibrate a Linux-native PEFT ARC strategy-router LoRA."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from pipeline import file_sha256, require_fresh_output, secure_tree, write_json


TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def render_training_row(tokenizer: Any, row: dict[str, Any]) -> tuple[list[int], list[int]]:
    messages = row.get("messages")
    tools = row.get("tools")
    if not isinstance(messages, list) or len(messages) < 2 or not isinstance(tools, list):
        raise ValueError("training rows require native messages and tools")
    if messages[-1].get("role") != "assistant" or not messages[-1].get("tool_calls"):
        raise ValueError("training rows must end in an assistant tool call")
    kwargs = {"tools": tools, "tokenize": False, "enable_thinking": False}
    try:
        prefix = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **kwargs)
        full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        prefix = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **kwargs)
        full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    input_ids = tokenizer.encode(full, add_special_tokens=False)
    if input_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("assistant completion is not a strict suffix of the rendered prompt")
    labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids) :]
    if not any(value != -100 for value in labels):
        raise ValueError("training row has no supervised assistant tokens")
    return input_ids, labels


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"training row {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError("training data is empty")
    return rows


def dtype_for(torch: Any, name: str) -> Any:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def train(args: argparse.Namespace) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if args.max_steps < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("training steps and gradient accumulation must be positive")
    if args.max_seq_length < 1 or not 0 < args.min_retained_fraction <= 1:
        raise ValueError("invalid sequence-length retention contract")
    model_path = args.model.resolve()
    if not model_path.is_dir():
        raise ValueError("offline base model directory does not exist")
    initial_adapter = args.initial_adapter.resolve() if args.initial_adapter else None
    if initial_adapter is not None and not initial_adapter.is_dir():
        raise ValueError("initial PEFT adapter directory does not exist")
    if initial_adapter is not None:
        initial_config = json.loads((initial_adapter / "adapter_config.json").read_text(encoding="utf-8"))
        expected_layers = list(range(args.first_layer, args.first_layer + args.num_layers))
        if (
            initial_config.get("r") != args.rank
            or initial_config.get("lora_alpha") != args.lora_alpha
            or float(initial_config.get("lora_dropout", -1)) != args.lora_dropout
            or initial_config.get("layers_to_transform") != expected_layers
        ):
            raise ValueError("initial adapter does not match the frozen LoRA training contract")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    source_rows = load_rows(args.data)
    prepared = []
    excluded = 0
    for row in source_rows:
        input_ids, labels = render_training_row(tokenizer, row)
        if len(input_ids) > args.max_seq_length:
            excluded += 1
            continue
        prepared.append((input_ids, labels))
    retained_fraction = len(prepared) / len(source_rows)
    if retained_fraction < args.min_retained_fraction:
        raise ValueError(
            f"sequence-length gate retained {retained_fraction:.4f}, below {args.min_retained_fraction:.4f}"
        )
    if args.limit:
        prepared = prepared[: args.limit]
    if not prepared:
        raise ValueError("no training rows remain after filtering")

    plan = {
        "format_version": "hfr.arcagi.peft_strategy_training_plan.v1",
        "offline_enforced": True,
        "model": {"id": args.model_id, "revision": args.model_revision},
        "data": {
            "sha256": file_sha256(args.data),
            "source_rows": len(source_rows),
            "retained_rows": len(prepared),
            "sequence_length_excluded_rows": excluded,
            "retained_fraction": retained_fraction,
        },
        "initial_adapter": {
            "enabled": initial_adapter is not None,
            "weights_sha256": file_sha256(initial_adapter / "adapter_model.safetensors")
            if initial_adapter
            else None,
        },
        "configuration": {
            "device": args.device,
            "dtype": args.dtype,
            "trainable_parameter_dtype": "float32",
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_seq_length": args.max_seq_length,
            "rank": args.rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_layers": list(range(args.first_layer, args.first_layer + args.num_layers)),
            "target_modules": list(TARGET_MODULES),
            "seed": args.seed,
        },
    }
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return 0

    require_fresh_output(args.out)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    randomizer = random.Random(args.seed)
    device = torch.device(args.device)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype_for(torch, args.dtype),
        )
    except TypeError:  # Transformers < 5 uses torch_dtype.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype_for(torch, args.dtype),
        )
    model.config.use_cache = False
    if initial_adapter is not None:
        model = PeftModel.from_pretrained(model, initial_adapter, is_trainable=True, local_files_only=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=args.rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=list(TARGET_MODULES),
                layers_to_transform=list(range(args.first_layer, args.first_layer + args.num_layers)),
                layers_pattern="layers",
                bias="none",
            ),
        )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.to(device)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(device=device, dtype=torch.float32)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    scaler_enabled = args.device == "cuda" and args.dtype == "float16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except TypeError:  # PyTorch < 2.4 did not accept the device argument.
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    order = list(range(len(prepared)))
    randomizer.shuffle(order)
    cursor = 0
    optimizer.zero_grad(set_to_none=True)
    losses = []
    started = time.monotonic()
    completed_steps = 0
    while completed_steps < args.max_steps:
        accumulated_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            if cursor == len(order):
                randomizer.shuffle(order)
                cursor = 0
            input_ids, labels = prepared[order[cursor]]
            cursor += 1
            batch_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            batch_labels = torch.tensor([labels], dtype=torch.long, device=device)
            with torch.autocast(
                device_type=args.device,
                dtype=dtype_for(torch, args.dtype),
                enabled=args.device == "cuda" and args.dtype != "float32",
            ):
                loss = model(input_ids=batch_ids, attention_mask=torch.ones_like(batch_ids), labels=batch_labels).loss
                scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach().cpu())
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        completed_steps += 1
        losses.append(accumulated_loss / args.gradient_accumulation_steps)
        if completed_steps == 1 or completed_steps % args.report_every == 0:
            print(
                json.dumps(
                    {
                        "step": completed_steps,
                        "loss": losses[-1],
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.max_training_seconds and time.monotonic() - started >= args.max_training_seconds:
            break

    model.save_pretrained(args.out, safe_serialization=True)
    receipt = {
        **plan,
        "format_version": "hfr.arcagi.peft_strategy_training_result.v1",
        "status": "succeeded",
        "completed_steps": completed_steps,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "loss": {"first": losses[0], "last": losses[-1], "minimum": min(losses)},
        "outputs": {
            "adapter_config_sha256": file_sha256(args.out / "adapter_config.json"),
            "adapter_model_sha256": file_sha256(args.out / "adapter_model.safetensors"),
        },
    }
    write_json(args.out / "training_receipt.json", receipt)
    secure_tree(args.out)
    print(json.dumps({"completed_steps": completed_steps, "outputs": receipt["outputs"]}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--initial-adapter", type=Path)
    result.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    result.add_argument("--model-revision", required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float16")
    result.add_argument("--max-steps", type=int, default=300)
    result.add_argument("--learning-rate", type=float, default=1e-5)
    result.add_argument("--gradient-accumulation-steps", type=int, default=1)
    result.add_argument("--max-seq-length", type=int, default=8192)
    result.add_argument("--min-retained-fraction", type=float, default=1.0)
    result.add_argument("--rank", type=int, default=8)
    result.add_argument("--lora-alpha", type=int, default=160)
    result.add_argument("--lora-dropout", type=float, default=0.0)
    result.add_argument("--first-layer", type=int, default=20)
    result.add_argument("--num-layers", type=int, default=8)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--limit", type=int, default=0)
    result.add_argument("--report-every", type=int, default=10)
    result.add_argument("--max-training-seconds", type=float, default=0.0)
    result.add_argument("--dry-run", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(train(parser().parse_args()))
