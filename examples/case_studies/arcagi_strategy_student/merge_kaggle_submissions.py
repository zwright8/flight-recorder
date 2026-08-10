#!/usr/bin/env python3
"""Merge two label-blind ARC submissions into a deterministic pass@2 file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pipeline import canonical_sha256, file_sha256, write_json


def valid_grid(value: Any) -> bool:
    if not isinstance(value, list) or not value or len(value) > 30:
        return False
    if not isinstance(value[0], list) or not 1 <= len(value[0]) <= 30:
        return False
    width = len(value[0])
    return all(
        isinstance(row, list)
        and len(row) == width
        and all(isinstance(cell, int) and not isinstance(cell, bool) and 0 <= cell <= 9 for cell in row)
        for row in value
    )


def merge_submissions(
    primary: dict[str, Any], secondary: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(primary) != set(secondary) or not primary:
        raise ValueError("submission task identifiers must be non-empty and identical")
    merged: dict[str, Any] = {}
    audit: list[dict[str, Any]] = []
    for task_id in primary:
        first_rows = primary[task_id]
        second_rows = secondary[task_id]
        if not isinstance(first_rows, list) or len(first_rows) != len(second_rows) or not first_rows:
            raise ValueError(f"submission row counts differ for task {task_id}")
        merged_rows = []
        for test_index, (first, second) in enumerate(zip(first_rows, second_rows, strict=True)):
            candidates = []
            for row in (first, second):
                if not isinstance(row, dict) or set(row) != {"attempt_1", "attempt_2"}:
                    raise ValueError("each submission row must contain exactly attempt_1 and attempt_2")
                for name in ("attempt_1", "attempt_2"):
                    grid = row[name]
                    if not valid_grid(grid):
                        raise ValueError(f"invalid grid at {task_id}[{test_index}].{name}")
                    if not any(grid == prior for prior in candidates):
                        candidates.append(grid)
            if not candidates:
                raise ValueError("validated submission unexpectedly had no candidates")
            duplicated = len(candidates) == 1
            if duplicated:
                candidates.append(candidates[0])
            merged_rows.append({"attempt_1": candidates[0], "attempt_2": candidates[1]})
            audit.append(
                {
                    "task_ref_sha256": hashlib.sha256(task_id.encode("utf-8")).hexdigest(),
                    "test_index": test_index,
                    "attempt_1_sha256": canonical_sha256(candidates[0]),
                    "attempt_2_sha256": canonical_sha256(candidates[1]),
                    "duplicated_only_candidate": duplicated,
                }
            )
        merged[task_id] = merged_rows
    return merged, audit


def run(args: argparse.Namespace) -> int:
    if args.out.exists() or args.receipt.exists():
        raise ValueError("refusing to overwrite an existing merged submission or receipt")
    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    secondary = json.loads(args.secondary.read_text(encoding="utf-8"))
    merged, audit = merge_submissions(primary, secondary)
    write_json(args.out, merged)
    receipt = {
        "format_version": "hfr.arcagi.kaggle_pass2_merge.v1",
        "label_blind": True,
        "source_submission_sha256": [file_sha256(args.primary), file_sha256(args.secondary)],
        "submission_sha256": file_sha256(args.out),
        "task_count": len(merged),
        "test_input_count": len(audit),
        "duplicated_only_candidate_count": sum(row["duplicated_only_candidate"] for row in audit),
        "outcomes": audit,
    }
    write_json(args.receipt, receipt)
    args.out.chmod(0o600)
    args.receipt.chmod(0o600)
    print(
        json.dumps(
            {
                "submission_sha256": receipt["submission_sha256"],
                "task_count": receipt["task_count"],
                "test_input_count": receipt["test_input_count"],
            },
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--primary", type=Path, required=True)
    result.add_argument("--secondary", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--receipt", type=Path, required=True)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
