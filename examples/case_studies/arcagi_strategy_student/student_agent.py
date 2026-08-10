#!/usr/bin/env python3
"""Local ARC agent: two small strategy routers plus the deterministic executor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_student import parse_strategy, render_prompt
from pipeline import SYSTEM_PROMPT, load_teacher, prompt_for, strategy_tool, valid_predictions


class ArcStrategyStudent:
    """Select up to two strategy tools and return at most three unique grids."""

    def __init__(self, arc_root: str | Path, model_path: str | Path, adapter_paths: list[str | Path]):
        if len(adapter_paths) != 2:
            raise ValueError("the validated ensemble contract requires exactly two adapter paths")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from mlx_lm import load

        self.arc_root = Path(arc_root).resolve()
        self.model_path = Path(model_path).resolve()
        self.executor, self.strategy_names, teacher = load_teacher(self.arc_root)
        self.allowed = set(self.strategy_names)
        self.tools = [strategy_tool(self.strategy_names, teacher["sha256"])]
        self.models: list[tuple[Any, Any]] = []
        for adapter_path in adapter_paths:
            resolved = Path(adapter_path).resolve()
            if not resolved.is_dir():
                raise ValueError(f"adapter path is not a directory: {resolved}")
            self.models.append(load(str(self.model_path), adapter_path=str(resolved)))

    @staticmethod
    def problem_payload(arc_problem: Any) -> dict[str, Any]:
        training = [
            {
                "input": pair.get_input_data().data().astype(int).tolist(),
                "output": pair.get_output_data().data().astype(int).tolist(),
            }
            for pair in arc_problem.training_set()
        ]
        return {
            "train": training,
            "test": [{"input": arc_problem.test_set().get_input_data().data().astype(int).tolist()}],
        }

    def selected_strategies(self, arc_problem: Any, *, max_tokens: int = 96) -> list[str]:
        from mlx_lm import generate

        task = self.problem_payload(arc_problem)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_for(task)},
        ]
        row = {"messages": messages, "tools": self.tools}
        selected: list[str] = []
        for model, tokenizer in self.models:
            output = generate(
                model,
                tokenizer,
                prompt=render_prompt(tokenizer, row),
                max_tokens=max_tokens,
                verbose=False,
            )
            strategy = parse_strategy(output, self.allowed)
            if strategy is not None and strategy not in selected:
                selected.append(strategy)
        return selected

    def make_predictions(self, arc_problem: Any) -> list[np.ndarray]:
        training = arc_problem.training_set()
        test_input = arc_problem.test_set().get_input_data().data()
        predictions: list[np.ndarray] = []
        for strategy in self.selected_strategies(arc_problem):
            guesses = valid_predictions(self.executor, getattr(self.executor, strategy)(training, test_input))
            for guess in guesses:
                if len(predictions) >= 3:
                    break
                if not any(np.array_equal(guess, prior) for prior in predictions):
                    predictions.append(guess)
        if not predictions:
            predictions.append(np.zeros_like(test_input))
        return predictions[:3]
