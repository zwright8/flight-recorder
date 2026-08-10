#!/usr/bin/env python3
"""Replay recorded student selections as a bounded ARC prediction ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline import file_sha256, load_teacher, make_problem, secure_tree, valid_predictions, write_json


def append_unique(values: list[np.ndarray], candidate: np.ndarray, limit: int) -> None:
    if len(values) < limit and not any(np.array_equal(candidate, prior) for prior in values):
        values.append(candidate)


def main(args: argparse.Namespace) -> int:
    if len(args.receipt) < 2:
        raise ValueError("at least two evaluation receipts are required")
    data_rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in args.receipt]
    expected_data_sha = file_sha256(args.data)
    for path, receipt in zip(args.receipt, receipts):
        if receipt.get("source_data", {}).get("sha256") != expected_data_sha:
            raise ValueError(f"evaluation receipt is bound to different source data: {path}")
    selections = [
        {row["episode_id"]: row for row in receipt["outcomes"]}
        for receipt in receipts
    ]
    agent, strategy_names, teacher = load_teacher(args.arc_root.resolve())
    allowed = set(strategy_names)
    outcomes: list[dict[str, Any]] = []
    for row in data_rows:
        problem = make_problem(row["task_family"], row["task"], 0)
        expected = problem.test_set().get_output_data().data()
        candidates: list[np.ndarray] = []
        selected: list[str] = []
        for source in selections:
            strategy = source[row["episode_id"]].get("selected_strategy")
            if strategy not in allowed or strategy in selected:
                continue
            selected.append(strategy)
            guesses = valid_predictions(
                agent,
                getattr(agent, strategy)(problem.training_set(), problem.test_set().get_input_data().data()),
            )
            for guess in guesses:
                append_unique(candidates, guess, args.max_predictions)
        outcomes.append(
            {
                "episode_id": row["episode_id"],
                "task_family": row["task_family"],
                "teacher_strategy": row["strategy"],
                "selected_strategies": selected,
                "prediction_count": len(candidates),
                "strategy_recalled": row["strategy"] in selected,
                "grid_any_match": any(np.array_equal(candidate, expected) for candidate in candidates),
            }
        )
    matched = sum(row["grid_any_match"] for row in outcomes)
    recalled = sum(row["strategy_recalled"] for row in outcomes)
    receipt = {
        "schema_version": "hfr.arcagi.student_ensemble_evaluation.v1",
        "teacher": teacher,
        "source_data": {"path": str(args.data), "sha256": expected_data_sha},
        "source_evaluations": [
            {"path": str(path), "sha256": file_sha256(path)} for path in args.receipt
        ],
        "max_predictions": args.max_predictions,
        "example_count": len(outcomes),
        "strategy_recall_count": recalled,
        "strategy_recall_rate": round(recalled / len(outcomes), 4) if outcomes else 0.0,
        "grid_any_match_count": matched,
        "grid_any_match_rate": round(matched / len(outcomes), 4) if outcomes else 0.0,
        "outcomes": outcomes,
        "warning": (
            "This ensemble is evaluated on untouched visible prompts whose task families occur in distillation "
            "training. It is not evidence of unseen-task or sealed ARC generalization."
        ),
    }
    if args.out.exists():
        raise ValueError(f"refusing to overwrite ensemble receipt: {args.out}")
    write_json(args.out, receipt)
    secure_tree(args.out.parent)
    print(
        json.dumps(
            {
                "example_count": receipt["example_count"],
                "strategy_recall_rate": receipt["strategy_recall_rate"],
                "grid_any_match_rate": receipt["grid_any_match_rate"],
            },
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--arc-root", type=Path, required=True)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--receipt", type=Path, action="append", required=True)
    result.add_argument("--max-predictions", type=int, default=3)
    result.add_argument("--out", type=Path, required=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
