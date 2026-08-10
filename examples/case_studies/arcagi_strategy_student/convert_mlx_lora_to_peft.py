#!/usr/bin/env python3
"""Convert a frozen MLX LoRA checkpoint into a Linux-loadable PEFT adapter."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pipeline import canonical_sha256, file_sha256, require_fresh_output, secure_tree, write_json


MLX_LORA_KEY = re.compile(
    r"^(model\.layers\.(?P<layer>\d+)\.(?:self_attn|mlp)\."
    r"(?P<module>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))\."
    r"lora_(?P<side>a|b)$"
)


def peft_key(mlx_key: str) -> tuple[str, int, str, str]:
    match = MLX_LORA_KEY.fullmatch(mlx_key)
    if match is None:
        raise ValueError(f"unsupported MLX LoRA tensor: {mlx_key}")
    side = "A" if match.group("side") == "a" else "B"
    return (
        f"base_model.model.{match.group(1)}.lora_{side}.weight",
        int(match.group("layer")),
        match.group("module"),
        side,
    )


def conversion_contract(source_config: dict[str, Any], source_keys: list[str]) -> dict[str, Any]:
    lora = source_config.get("lora_parameters", {})
    rank = lora.get("rank")
    scale = lora.get("scale")
    dropout = lora.get("dropout")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("source adapter config has no valid LoRA rank")
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError("source adapter config has no valid LoRA scale")
    if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError("source adapter config has no valid LoRA dropout")

    parsed = [peft_key(key) for key in source_keys]
    layers = sorted({item[1] for item in parsed})
    modules = sorted({item[2] for item in parsed})
    pairs = {(item[1], item[2], item[3]) for item in parsed}
    expected = {(layer, module, side) for layer in layers for module in modules for side in ("A", "B")}
    if pairs != expected:
        raise ValueError("source checkpoint does not contain a complete A/B tensor matrix")
    if len(layers) != source_config.get("num_layers"):
        raise ValueError("source tensor layers do not match adapter_config num_layers")
    return {
        "rank": rank,
        "scale": float(scale),
        "lora_alpha": float(scale) * rank,
        "dropout": float(dropout),
        "layers": layers,
        "target_modules": modules,
        "tensor_count": len(parsed),
    }


def convert(args: argparse.Namespace) -> int:
    from peft import LoraConfig
    from safetensors import safe_open
    from safetensors.torch import save_file

    source_weights = args.mlx_adapter.resolve()
    source_config_path = args.mlx_config.resolve()
    if not source_weights.is_file() or not source_config_path.is_file():
        raise ValueError("MLX adapter weights and configuration must both exist")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    with safe_open(source_weights, framework="pt", device="cpu") as handle:
        source_keys = list(handle.keys())
        contract = conversion_contract(source_config, source_keys)
        converted = {}
        key_mapping = {}
        for source_key in source_keys:
            target_key, _layer, _module, _side = peft_key(source_key)
            converted[target_key] = handle.get_tensor(source_key).T.contiguous()
            key_mapping[source_key] = target_key

    require_fresh_output(args.out)
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=True,
        r=contract["rank"],
        lora_alpha=contract["lora_alpha"],
        lora_dropout=contract["dropout"],
        target_modules=contract["target_modules"],
        layers_to_transform=contract["layers"],
        layers_pattern="layers",
        bias="none",
    )
    peft_config.base_model_name_or_path = args.base_model_id
    peft_config.revision = args.base_model_revision
    peft_config.save_pretrained(args.out)
    output_weights = args.out / "adapter_model.safetensors"
    save_file(converted, output_weights, metadata={"format": "pt"})
    receipt = {
        "format_version": "hfr.arcagi.mlx_to_peft_conversion.v1",
        "source": {
            "adapter_sha256": file_sha256(source_weights),
            "config_sha256": file_sha256(source_config_path),
        },
        "base_model": {"id": args.base_model_id, "revision": args.base_model_revision},
        "contract": contract,
        "conversion": {
            "operation": "transpose each MLX A/B tensor into PEFT Linear weight orientation",
            "key_mapping_sha256": canonical_sha256(key_mapping),
        },
        "outputs": {
            "adapter_config_sha256": file_sha256(args.out / "adapter_config.json"),
            "adapter_model_sha256": file_sha256(output_weights),
        },
        "parity_status": "pending",
    }
    write_json(args.out / "conversion_receipt.json", receipt)
    secure_tree(args.out)
    print(
        json.dumps(
            {
                "adapter_model_sha256": receipt["outputs"]["adapter_model_sha256"],
                "layers": contract["layers"],
                "target_modules": contract["target_modules"],
                "tensor_count": contract["tensor_count"],
            },
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mlx-adapter", type=Path, required=True)
    result.add_argument("--mlx-config", type=Path, required=True)
    result.add_argument("--base-model-id", default="Qwen/Qwen3-0.6B")
    result.add_argument("--base-model-revision", required=True)
    result.add_argument("--out", type=Path, required=True)
    return result


if __name__ == "__main__":
    raise SystemExit(convert(parser().parse_args()))
