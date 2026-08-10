#!/usr/bin/env python3
"""Evaluate a local MLX ARC strategy router by executing its selected tool."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from pipeline import canonical_sha256, file_sha256, load_teacher, make_problem, secure_tree, valid_predictions, write_json


STRATEGY_RE = re.compile(r"(?:arc_)?(try_[a-z0-9_]+)")


def parse_strategy(text: str, allowed: set[str]) -> str | None:
    for match in STRATEGY_RE.finditer(text):
        if match.group(1) in allowed:
            return match.group(1)
    return None


def render_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    kwargs = {
        "tools": row["tools"],
        "add_generation_prompt": True,
        "tokenize": False,
    }
    try:
        return tokenizer.apply_chat_template(row["messages"], enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(row["messages"], **kwargs)


def evaluate(args: argparse.Namespace) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_path = args.model.resolve()
    if not model_path.is_dir():
        raise ValueError(f"offline model path is not a directory: {model_path}")
    adapter_path = args.adapter.resolve() if args.adapter else None
    if adapter_path is not None and not adapter_path.is_dir():
        raise ValueError(f"adapter path is not a directory: {adapter_path}")

    from mlx_lm import generate, load

    model, tokenizer = load(str(model_path), adapter_path=str(adapter_path) if adapter_path else None)
    agent, strategy_names, teacher = load_teacher(args.arc_root.resolve())
    allowed = set(strategy_names)
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    outcomes: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        rendered = render_prompt(tokenizer, row)
        output = generate(model, tokenizer, prompt=rendered, max_tokens=args.max_tokens, verbose=False)
        selected = parse_strategy(output, allowed)
        expected = np.asarray(row["task"]["test"][0]["output"], dtype=int)
        grid_match = False
        prediction_count = 0
        error = None
        if selected is not None:
            try:
                problem = make_problem(row["task_family"], row["task"], 0)
                candidates = valid_predictions(
                    agent,
                    getattr(agent, selected)(problem.training_set(), problem.test_set().get_input_data().data()),
                )
                prediction_count = len(candidates)
                grid_match = any(np.array_equal(candidate, expected) for candidate in candidates)
            except Exception as exc:  # The result is evidence; preserve a bounded error string.
                error = f"{type(exc).__name__}: {exc}"[:300]
        outcomes.append(
            {
                "episode_id": row["episode_id"],
                "task_family": row["task_family"],
                "teacher_strategy": row["strategy"],
                "selected_strategy": selected,
                "action_exact": selected == row["strategy"],
                "grid_any_match": grid_match,
                "prediction_count": prediction_count,
                "raw_output": output,
                "error": error,
            }
        )
        if index % 25 == 0:
            print(f"evaluated {index}/{len(rows)}", flush=True)

    action_exact = sum(row["action_exact"] for row in outcomes)
    grid_match = sum(row["grid_any_match"] for row in outcomes)
    invalid = sum(row["selected_strategy"] is None for row in outcomes)
    family_rows: list[dict[str, Any]] = []
    for family in sorted({row["task_family"] for row in outcomes}):
        selected_rows = [row for row in outcomes if row["task_family"] == family]
        family_rows.append(
            {
                "task_family": family,
                "count": len(selected_rows),
                "action_exact_rate": round(sum(row["action_exact"] for row in selected_rows) / len(selected_rows), 4),
                "grid_any_match_rate": round(sum(row["grid_any_match"] for row in selected_rows) / len(selected_rows), 4),
            }
        )
    receipt = {
        "schema_version": "hfr.arcagi.student_evaluation.v1",
        "model": {"path": str(model_path), "sha256": canonical_sha256(sorted(path.name for path in model_path.iterdir()))},
        "adapter": {"path": str(adapter_path), "config_sha256": file_sha256(adapter_path / "adapter_config.json")}
        if adapter_path
        else None,
        "teacher": teacher,
        "source_data": {"path": str(args.data), "sha256": file_sha256(args.data)},
        "example_count": len(outcomes),
        "action_exact_count": action_exact,
        "action_exact_rate": round(action_exact / len(outcomes), 4) if outcomes else 0.0,
        "grid_any_match_count": grid_match,
        "grid_any_match_rate": round(grid_match / len(outcomes), 4) if outcomes else 0.0,
        "invalid_action_count": invalid,
        "selected_strategy_counts": dict(sorted(Counter(str(row["selected_strategy"]) for row in outcomes).items())),
        "task_families": family_rows,
        "claims": [
            "Grid success means one of up to three outputs from the selected deterministic strategy matched exactly.",
            "All examples are visible-task derivatives; this is not a sealed ARC benchmark result.",
        ],
        "outcomes": outcomes,
    }
    if args.out.exists():
        raise ValueError(f"refusing to overwrite evaluation receipt: {args.out}")
    write_json(args.out, receipt)
    secure_tree(args.out.parent)
    print(json.dumps({key: receipt[key] for key in ("example_count", "action_exact_rate", "grid_any_match_rate", "invalid_action_count")}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--arc-root", type=Path, required=True)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--adapter", type=Path)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--limit", type=int, default=0)
    result.add_argument("--max-tokens", type=int, default=96)
    return result


if __name__ == "__main__":
    raise SystemExit(evaluate(parser().parse_args()))
