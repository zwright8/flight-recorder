#!/usr/bin/env python3
"""Score an ARC pass@2 submission against an explicitly provided solution set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from kaggle_submission import PASS_K, validate_grid
from pipeline import file_sha256, write_json


def score_submission(
    submission: dict[str, Any], solutions: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(submission) != set(solutions):
        raise ValueError("submission and solution task identifiers differ")
    outcomes = []
    for task_id, expected_outputs in solutions.items():
        attempts = submission[task_id]
        if not isinstance(expected_outputs, list) or len(attempts) != len(expected_outputs):
            raise ValueError(f"task {task_id} test-output counts differ")
        for test_index, (entry, expected) in enumerate(zip(attempts, expected_outputs)):
            if not isinstance(entry, dict) or set(entry) != {"attempt_1", "attempt_2"}:
                raise ValueError(f"task {task_id} test[{test_index}] must contain exactly two attempts")
            validate_grid(expected, f"task {task_id} solution[{test_index}]")
            candidates = [entry[f"attempt_{index}"] for index in range(1, PASS_K + 1)]
            for index, candidate in enumerate(candidates, start=1):
                validate_grid(candidate, f"task {task_id} test[{test_index}].attempt_{index}")
            matched_attempt = next(
                (index for index, candidate in enumerate(candidates, start=1) if candidate == expected),
                None,
            )
            outcomes.append(
                {
                    "task_ref_sha256": hashlib.sha256(task_id.encode("utf-8")).hexdigest(),
                    "test_index": test_index,
                    "passed": matched_attempt is not None,
                    "matched_attempt": matched_attempt,
                }
            )
    passed = sum(row["passed"] for row in outcomes)
    return {
        "passed": passed,
        "total": len(outcomes),
        "exact_rate": passed / len(outcomes) if outcomes else 0.0,
    }, outcomes


def main(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise ValueError(f"refusing to overwrite score receipt: {args.out}")
    submission = json.loads(args.submission.read_text(encoding="utf-8"))
    solutions = json.loads(args.solutions.read_text(encoding="utf-8"))
    aggregate, outcomes = score_submission(submission, solutions)
    receipt = {
        "format_version": "hfr.arcagi.pass2_score.v1",
        "scope": args.scope,
        "submission_sha256": file_sha256(args.submission),
        "solutions_sha256": file_sha256(args.solutions),
        "aggregate": aggregate,
        "outcomes": outcomes,
    }
    write_json(args.out, receipt)
    args.out.chmod(0o600)
    print(json.dumps(aggregate, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--submission", type=Path, required=True)
    result.add_argument("--solutions", type=Path, required=True)
    result.add_argument("--scope", required=True)
    result.add_argument("--out", type=Path, required=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
