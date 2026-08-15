#!/usr/bin/env python3
"""Build governed ARC strategy-selection trajectories from an executable teacher.

The visible ARC test outputs are used only to verify the deterministic teacher
and its transformed variants.  They are never included in the student prompt.
Every derivative of one source task shares a ``task_family`` so Hermes Flight
Recorder keeps it in one dataset split.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib
import io
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np


HFR_ROOT = Path(__file__).resolve().parents[3]
if str(HFR_ROOT) not in sys.path:
    sys.path.insert(0, str(HFR_ROOT))

from flightrecorder.cli import main as flightrecorder_main  # noqa: E402


SCHEMA_VERSION = "hfr.arcagi.teacher_dataset.v1"
SYSTEM_PROMPT = (
    "You are an ARC strategy router. Inspect every visible training input/output pair, "
    "then call exactly one supplied read-only strategy tool for the test input. "
    "Do not invent a strategy and do not emit a grid directly."
)
NATIVE_ARC_PROMPT_PREFIX = "Choose one strategy tool for this ARC problem.\nARC_PROBLEM="
COMPACT_ARC_PROMPT_PREFIX = "Choose one strategy tool for this ARC problem.\nARC_COMPACT_V1 "
D4_NAMES = (
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "flip_left_right",
    "flip_up_down",
    "transpose",
    "anti_transpose",
)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_bytes(row))


def secure_tree(path: Path) -> None:
    for child in path.rglob("*"):
        child.chmod(0o700 if child.is_dir() else 0o600)
    path.chmod(0o700)


def require_fresh_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def load_teacher(arc_root: Path) -> tuple[Any, list[str], dict[str, Any]]:
    teacher_path = arc_root / "final_submission" / "ArcAgent.py"
    required = [
        teacher_path,
        arc_root / "ArcData.py",
        arc_root / "ArcProblem.py",
        arc_root / "ArcSet.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"missing ARC teacher inputs: {missing}")

    for name in (str(arc_root), str(teacher_path.parent)):
        if name not in sys.path:
            sys.path.insert(0, name)
    importlib.invalidate_caches()
    arc_agent_module = importlib.import_module("ArcAgent")
    source = teacher_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    strategy_names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "make_predictions":
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "strategy_order" for target in child.targets):
                continue
            if isinstance(child.value, ast.List):
                strategy_names = [
                    element.attr
                    for element in child.value.elts
                    if isinstance(element, ast.Attribute) and element.attr.startswith("try_")
                ]
    if not strategy_names:
        raise ValueError("could not extract strategy_order from final_submission/ArcAgent.py")
    return arc_agent_module.ArcAgent(), strategy_names, {
        "path": "final_submission/ArcAgent.py",
        "sha256": file_sha256(teacher_path),
        "strategy_count": len(strategy_names),
    }


def arc_types() -> tuple[Any, Any, Any]:
    return (
        importlib.import_module("ArcData").ArcData,
        importlib.import_module("ArcProblem").ArcProblem,
        importlib.import_module("ArcSet").ArcSet,
    )


def make_problem(task_id: str, task: dict[str, Any], test_index: int) -> Any:
    ArcData, ArcProblem, ArcSet = arc_types()
    training = [
        ArcSet(ArcData(np.asarray(pair["input"], dtype=int)), ArcData(np.asarray(pair["output"], dtype=int)))
        for pair in task["train"]
    ]
    test = task["test"][test_index]
    return ArcProblem(
        task_id,
        training,
        ArcSet(ArcData(np.asarray(test["input"], dtype=int)), ArcData(np.asarray(test["output"], dtype=int))),
    )


def d4(grid: list[list[int]], name: str) -> list[list[int]]:
    value = np.asarray(grid, dtype=int)
    transforms: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "identity": lambda item: item,
        "rotate_90": lambda item: np.rot90(item, 1),
        "rotate_180": lambda item: np.rot90(item, 2),
        "rotate_270": lambda item: np.rot90(item, 3),
        "flip_left_right": np.fliplr,
        "flip_up_down": np.flipud,
        "transpose": lambda item: item.T,
        "anti_transpose": lambda item: np.rot90(item.T, 2),
    }
    if name not in transforms:
        raise ValueError(f"unknown D4 transform: {name}")
    return transforms[name](value).astype(int).tolist()


def recolor(grid: list[list[int]], color_map: list[int]) -> list[list[int]]:
    value = np.asarray(grid, dtype=int)
    if len(color_map) != 10 or sorted(color_map) != list(range(10)):
        raise ValueError("color_map must be a permutation of ARC colors 0..9")
    return np.asarray(color_map, dtype=int)[value].tolist()


def transform_task(
    task: dict[str, Any],
    test_index: int,
    spatial: str,
    color_map: list[int],
    pair_order: list[int],
) -> dict[str, Any]:
    def grid(value: list[list[int]]) -> list[list[int]]:
        return recolor(d4(value, spatial), color_map)

    train = [task["train"][index] for index in pair_order]
    return {
        "train": [{"input": grid(pair["input"]), "output": grid(pair["output"])} for pair in train],
        "test": [
            {
                "input": grid(task["test"][test_index]["input"]),
                "output": grid(task["test"][test_index]["output"]),
            }
        ],
    }


def variant_contract(task: dict[str, Any], task_id: str, test_index: int, attempt: int) -> dict[str, Any]:
    if attempt == 0:
        spatial = "identity"
        color_map = list(range(10))
        pair_order = list(range(len(task["train"])))
    else:
        spatial = D4_NAMES[attempt % len(D4_NAMES)]
        seed = int(hashlib.sha256(f"arcagi-v1:{task_id}:{test_index}:{attempt}".encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        color_map = list(range(10))
        if attempt >= len(D4_NAMES):
            if attempt % 3 == 0:
                tail = color_map[1:]
                rng.shuffle(tail)
                color_map = [0, *tail]
            else:
                rng.shuffle(color_map)
        pair_order = list(range(len(task["train"])))
        if attempt % 4 == 1:
            pair_order.reverse()
        elif attempt % 4 in {2, 3}:
            rng.shuffle(pair_order)
    return {"spatial": spatial, "color_map": color_map, "pair_order": pair_order, "attempt": attempt}


def valid_predictions(agent: Any, guesses: Any) -> list[np.ndarray]:
    results: list[np.ndarray] = []
    for guess in agent.as_guess_list(guesses):
        if not agent.is_valid_grid(guess):
            continue
        candidate = np.asarray(guess, dtype=int)
        if not any(np.array_equal(candidate, prior) for prior in results):
            results.append(candidate)
    return results[:3]


def teacher_label(agent: Any, strategy_names: list[str], problem: Any) -> dict[str, Any] | None:
    expected = problem.test_set().get_output_data().data()
    predictions = valid_predictions(agent, agent.make_predictions(problem))
    teacher_slots = [index for index, value in enumerate(predictions) if np.array_equal(value, expected)]
    if not teacher_slots:
        return None
    training = problem.training_set()
    test_input = problem.test_set().get_input_data().data()
    for strategy_name in strategy_names:
        try:
            candidates = valid_predictions(agent, getattr(agent, strategy_name)(training, test_input))
        except Exception:
            continue
        winning_slots = [index for index, value in enumerate(candidates) if np.array_equal(value, expected)]
        if winning_slots:
            return {
                "strategy": strategy_name,
                "strategy_prediction_slot": winning_slots[0],
                "teacher_prediction_slot": teacher_slots[0],
                "predictions": [value.tolist() for value in candidates],
            }
    return None


def strategy_tool(strategy_names: list[str], teacher_revision: str) -> dict[str, Any]:
    return {
        "type": "function",
        "version": f"arcagent-{teacher_revision[:12]}",
        "capability": "arc_strategy",
        "risk": "read",
        "write_capable": False,
        "function": {
            "name": "arc_apply_strategy",
            "description": (
                "Apply one named deterministic ARC strategy after it validates every training pair; "
                "return up to three candidate grids. Strategy names describe the transformation they attempt."
            ),
            "parameters": {
                "type": "object",
                "properties": {"strategy": {"type": "string", "enum": strategy_names}},
                "required": ["strategy"],
                "additionalProperties": False,
            },
        },
    }


def prompt_for(task: dict[str, Any]) -> str:
    payload = {"training_pairs": task["train"], "test_input": task["test"][0]["input"]}
    return NATIVE_ARC_PROMPT_PREFIX + json.dumps(
        payload, separators=(",", ":"), ensure_ascii=False
    )


def compact_grid(grid_value: list[list[int]]) -> str:
    """Encode a validated ARC grid losslessly with row delimiters."""
    if not isinstance(grid_value, list) or not grid_value:
        raise ValueError("compact ARC grids must be non-empty lists")
    width = len(grid_value[0]) if isinstance(grid_value[0], list) else 0
    if width < 1:
        raise ValueError("compact ARC grids must contain non-empty rows")
    rows = []
    for row in grid_value:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("compact ARC grids must be rectangular")
        if any(isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell <= 9 for cell in row):
            raise ValueError("compact ARC grid cells must be integer colors 0..9")
        rows.append("".join(str(cell) for cell in row))
    return "/".join(rows)


def compact_prompt_for(task: dict[str, Any]) -> str:
    """Render the same ARC state as ``prompt_for`` with fewer tokens."""
    training_pairs = ";".join(
        f"{compact_grid(pair['input'])}>{compact_grid(pair['output'])}"
        for pair in task["train"]
    )
    return (
        f"{COMPACT_ARC_PROMPT_PREFIX}training_pairs={training_pairs}\n"
        f"test_input={compact_grid(task['test'][0]['input'])}"
    )


def compact_prompt_from_native(prompt: str) -> str:
    """Losslessly convert a recorded native JSON ARC prompt to compact v1."""
    if not isinstance(prompt, str) or not prompt.startswith(NATIVE_ARC_PROMPT_PREFIX):
        raise ValueError("training row does not contain a native ARC JSON prompt")
    payload = json.loads(prompt[len(NATIVE_ARC_PROMPT_PREFIX) :])
    if not isinstance(payload, dict) or set(payload) != {"training_pairs", "test_input"}:
        raise ValueError("native ARC prompt payload has unexpected fields")
    return compact_prompt_for(
        {
            "train": payload["training_pairs"],
            "test": [{"input": payload["test_input"]}],
        }
    )


def hfr_context(
    episode_id: str,
    tools: list[dict[str, Any]],
    teacher: dict[str, Any],
    dataset_revision: str,
) -> dict[str, Any]:
    policy = {"id": "arcagi-strategy-router", "version": "1", "teacher_sha256": teacher["sha256"]}
    environment = {"id": "arcagi-visible-bcd-local", "version": dataset_revision}
    chat_template = {"name": "hfr-native-tool-json", "revision": "1", "system_prompt": SYSTEM_PROMPT}
    return {
        "root_session_id": episode_id,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT, "session_id": episode_id}],
        "tools": tools,
        "model": {"provider": "local-deterministic", "name": "final_submission.ArcAgent", "revision": teacher["sha256"]},
        "tokenizer": {"name": "not-applicable-deterministic", "revision": teacher["sha256"]},
        "chat_template": {**chat_template, "sha256": canonical_sha256(chat_template)},
        "policy": {**policy, "sha256": canonical_sha256(policy)},
        "environment": {**environment, "sha256": canonical_sha256(environment)},
        "governance": {
            "owner": "local-project-owner",
            "tenant": "local-arcagi",
            "legal_basis": "owner-authorized-research",
            "allowed_purposes": ["agent_training", "internal_evaluation"],
            "sensitivity": "sensitive-local-training-data",
            "jurisdiction": "US",
            "retention_expires_at": "2036-01-01T00:00:00+00:00",
            "license": "private-coursework-not-for-publication",
            "provenance": {
                "source": "ARC visible Milestones B/C/D plus executable final_submission.ArcAgent",
                "teacher_sha256": teacher["sha256"],
            },
            "deletion_subject_ids": [episode_id],
        },
    }


def scenario_and_trace(
    row: dict[str, Any],
    tools: list[dict[str, Any]],
    teacher: dict[str, Any],
    dataset_revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    strategy = row["strategy"]
    tool_name = "arc_apply_strategy"
    call_id = f"{row['episode_id']}-call-001"
    predictions = row["predictions"]
    prediction_sha = canonical_sha256(predictions)
    result = {
        "status": "ok",
        "selected_strategy": strategy,
        "predictions": predictions,
        "prediction_sha256": prediction_sha,
    }
    trace = {
        "session_id": row["episode_id"],
        "conversations": [
            {"from": "human", "value": row["prompt"]},
            {
                "from": "gpt",
                "value": "<tool_call>\n"
                + json.dumps(
                    {"id": call_id, "name": tool_name, "arguments": {"strategy": strategy}},
                    separators=(",", ":"),
                )
                + "\n</tool_call>",
            },
            {
                "from": "tool",
                "value": "<tool_response>\n"
                + json.dumps({"tool_call_id": call_id, "name": tool_name, "content": result}, separators=(",", ":"))
                + "\n</tool_response>",
            },
            {
                "from": "gpt",
                "value": f"Executed {strategy} through {tool_name}; recorded prediction set {prediction_sha}.",
            },
        ],
        "timestamp": "2026-08-10T00:00:00+00:00",
        "model": "final_submission.ArcAgent",
        "completed": True,
        "api_calls": 0,
        "trajectory_context": hfr_context(row["episode_id"], tools, teacher, dataset_revision),
    }
    scenario = {
        "id": row["episode_id"],
        "title": f"ARC teacher strategy {row['task_family']} variant {row['variant_index']}",
        "task_family": row["task_family"],
        "prompt": row["prompt"],
        "policy": {
            "forbidden_tool_names": [],
            "forbidden_command_patterns": [],
            "forbidden_url_patterns": [],
            "secret_patterns": ["(?i)(api[_-]?key|secret|password)\\s*[:=]"],
            "max_tool_calls": 1,
            "max_api_calls": 0,
            "max_subagents": 0,
            "max_subagent_depth": 0,
        },
        "assertions": {
            "final_contains": [tool_name, strategy, prediction_sha],
            "required_actions": [
                {
                    "id": "strategy_called",
                    "description": "The attributed deterministic strategy was called.",
                    "event_type": "tool_call",
                    "tool_name": tool_name,
                    "field_equals": {"args.strategy": strategy},
                }
            ],
            "required_action_sequences": [
                {
                    "id": "strategy_call_then_result",
                    "description": "The exact strategy call precedes its result.",
                    "steps": [
                        {"event_type": "tool_call", "tool_name": tool_name},
                        {"event_type": "tool_result", "tool_name": tool_name},
                    ],
                }
            ],
            "required_event_counts": [
                {"id": "one_strategy_call", "event_type": "tool_call", "exact_count": 1},
                {"id": "one_strategy_result", "event_type": "tool_result", "exact_count": 1},
            ],
            "required_evidence": [
                {
                    "id": "prediction_hash_recorded",
                    "type": "event_matches",
                    "event_type": "tool_result",
                    "tool_name": tool_name,
                    "contains": prediction_sha,
                }
            ],
        },
        "scoring": {"pass_threshold": 100},
    }
    return scenario, trace


def run_hfr(args: list[str]) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return flightrecorder_main(args)


def source_tasks(arc_root: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    rows: list[tuple[str, Path, dict[str, Any]]] = []
    for milestone in ("B", "C", "D"):
        for path in sorted((arc_root / "Milestones" / milestone).glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value.get("train"), list) or not isinstance(value.get("test"), list):
                raise ValueError(f"invalid ARC task: {path}")
            rows.append((path.stem, path, value))
    if not rows:
        raise ValueError("no Milestones B/C/D tasks found")
    return rows


def build_examples(
    arc_root: Path,
    agent: Any,
    strategy_names: list[str],
    variants_per_example: int,
    max_attempt_multiplier: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    rejected = Counter()
    for task_id, source_path, task in source_tasks(arc_root):
        source_manifest.append(
            {
                "task_family": task_id,
                "source_path": str(source_path.relative_to(arc_root)),
                "source_sha256": file_sha256(source_path),
                "test_entry_count": len(task["test"]),
            }
        )
        for test_index in range(len(task["test"])):
            accepted = 0
            seen_prompts: set[str] = set()
            attempts = max(variants_per_example * max_attempt_multiplier, variants_per_example)
            for attempt in range(attempts):
                contract = variant_contract(task, task_id, test_index, attempt)
                transformed = transform_task(task, test_index, **{key: contract[key] for key in ("spatial", "color_map", "pair_order")})
                prompt = prompt_for(transformed)
                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                if prompt_hash in seen_prompts:
                    rejected["duplicate_prompt"] += 1
                    continue
                seen_prompts.add(prompt_hash)
                label = teacher_label(agent, strategy_names, make_problem(task_id, transformed, 0))
                if label is None:
                    rejected["teacher_not_exact"] += 1
                    continue
                episode_id = f"arc-{task_id}-t{test_index:02d}-v{accepted:03d}-{prompt_hash[:10]}"
                examples.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "episode_id": episode_id,
                        "task_family": task_id,
                        "source_test_index": test_index,
                        "variant_index": accepted,
                        "augmentation": contract,
                        "prompt": prompt,
                        "prompt_sha256": prompt_hash,
                        "strategy": label["strategy"],
                        "strategy_prediction_slot": label["strategy_prediction_slot"],
                        "teacher_prediction_slot": label["teacher_prediction_slot"],
                        "predictions": label["predictions"],
                        "task": transformed,
                        "expected_output_sha256": canonical_sha256(transformed["test"][0]["output"]),
                    }
                )
                accepted += 1
                if accepted >= variants_per_example:
                    break
            if accepted == 0:
                raise ValueError(f"teacher failed even the base task for {task_id} test entry {test_index}")
    audit = {
        "source_tasks": source_manifest,
        "source_task_count": len(source_manifest),
        "source_test_entry_count": sum(item["test_entry_count"] for item in source_manifest),
        "accepted_trajectory_count": len(examples),
        "rejected_candidates": dict(sorted(rejected.items())),
        "strategy_counts": dict(sorted(Counter(row["strategy"] for row in examples).items())),
        "family_counts": dict(sorted(Counter(row["task_family"] for row in examples).items())),
    }
    return examples, audit


def derive_student_data(out: Path, examples: list[dict[str, Any]]) -> dict[str, Any]:
    export = out / "training_export"
    split_contract = json.loads((export / "dataset_splits.json").read_text(encoding="utf-8"))
    family_split = {row["task_family"]: row["split"] for row in split_contract["assignments"]}
    by_episode = {row["episode_id"]: row for row in examples}
    student_root = out / "student_data"
    evaluation_root = out / "evaluation"
    counts: dict[str, int] = {}
    families: dict[str, list[str]] = {}
    for split, hfr_split in (("train", "train"), ("valid", "validation"), ("test", "test")):
        action_path = export / "splits" / hfr_split / "action_sft.jsonl"
        action_rows = [json.loads(line) for line in action_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        student_rows: list[dict[str, Any]] = []
        evaluation_rows: list[dict[str, Any]] = []
        for action in action_rows:
            episode_id = action["episode_id"]
            example = by_episode[episode_id]
            messages: list[dict[str, Any]] = []
            for message in action["messages"]:
                messages.append(message)
                if message.get("role") == "assistant" and message.get("tool_calls"):
                    break
            if not messages or not messages[-1].get("tool_calls"):
                raise ValueError(f"action-SFT row lacks a teacher tool call: {episode_id}")
            full_tools = [tool["definition"] for tool in action["trajectory_v2"]["tools"]]
            student_rows.append(
                {
                    "messages": messages,
                    "tools": full_tools,
                    "sample_id": episode_id,
                    "task_family": action["task_family"],
                    "source_action_sft_sha256": canonical_sha256(action),
                }
            )
            evaluation_rows.append(
                {
                    "episode_id": episode_id,
                    "task_family": example["task_family"],
                    "strategy": example["strategy"],
                    "task": example["task"],
                    "messages": messages[:-1],
                    "tools": full_tools,
                }
            )
        write_jsonl(student_root / f"{split}.jsonl", student_rows)
        write_jsonl(evaluation_root / f"{split}.jsonl", evaluation_rows)
        counts[split] = len(student_rows)
        families[split] = sorted({row["task_family"] for row in student_rows})

    overlap = {
        "train_valid": sorted(set(families["train"]) & set(families["valid"])),
        "train_test": sorted(set(families["train"]) & set(families["test"])),
        "valid_test": sorted(set(families["valid"]) & set(families["test"])),
    }
    manifest = {
        "schema_version": "hfr.arcagi.student_data.v1",
        "source_hfr_manifest_sha256": file_sha256(export / "manifest.json"),
        "split_strategy": split_contract["strategy"],
        "counts": counts,
        "task_families": families,
        "cross_split_task_families": overlap,
        "family_exclusive": not any(overlap.values()),
        "student_view": (
            "Each row is projected from a validated HFR action-SFT trajectory and ends at the teacher tool call. "
            "Tool results and visible test outputs are excluded from the student input."
        ),
        "files": {},
    }
    for root_name, root in (("student", student_root), ("evaluation", evaluation_root)):
        for path in sorted(root.glob("*.jsonl")):
            manifest["files"][f"{root_name}/{path.name}"] = {
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
    if not manifest["family_exclusive"]:
        raise ValueError(f"cross-split ARC task-family leakage detected: {overlap}")
    write_json(out / "student_data_manifest.json", manifest)
    return manifest


def _project_action_row(action: dict[str, Any], example: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in action["messages"]:
        messages.append(message)
        if message.get("role") == "assistant" and message.get("tool_calls"):
            break
    if not messages or not messages[-1].get("tool_calls"):
        raise ValueError(f"action-SFT row lacks a teacher tool call: {action['episode_id']}")
    full_tools = [tool["definition"] for tool in action["trajectory_v2"]["tools"]]
    student = {
        "messages": messages,
        "tools": full_tools,
        "sample_id": action["episode_id"],
        "task_family": action["task_family"],
        "source_action_sft_sha256": canonical_sha256(action),
    }
    evaluation = {
        "episode_id": action["episode_id"],
        "task_family": example["task_family"],
        "strategy": example["strategy"],
        "task": example["task"],
        "messages": messages[:-1],
        "tools": full_tools,
    }
    return student, evaluation


def derive_visible_distillation(out: Path, examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a non-generalization view that holds out the untouched visible tasks."""

    data_root = out / "visible_distillation_data"
    evaluation_root = out / "visible_distillation_evaluation"
    manifest_path = out / "visible_distillation_manifest.json"
    for path in (data_root, evaluation_root, manifest_path):
        if path.exists():
            raise ValueError(f"refusing to overwrite existing visible-distillation artifact: {path}")
    action_rows = [
        json.loads(line)
        for line in (out / "training_export" / "action_sft.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actions = {row["episode_id"]: row for row in action_rows}
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for example in examples:
        groups.setdefault((example["task_family"], int(example["source_test_index"])), []).append(example)

    split_examples: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for group_id, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: (int(row["variant_index"]), row["episode_id"]))
        if len(ordered) < 3:
            raise ValueError(f"visible-distillation group needs at least three variants: {group_id}")
        if ordered[0]["augmentation"]["attempt"] != 0:
            raise ValueError(f"visible-distillation test row is not the untouched task: {group_id}")
        split_examples["test"].append(ordered[0])
        split_examples["valid"].append(ordered[1])
        split_examples["train"].extend(ordered[2:])

    counts: dict[str, int] = {}
    prompt_sets: dict[str, set[str]] = {}
    strategy_sets: dict[str, set[str]] = {}
    for split in ("train", "valid", "test"):
        student_rows: list[dict[str, Any]] = []
        evaluation_rows: list[dict[str, Any]] = []
        for example in split_examples[split]:
            student, evaluation = _project_action_row(actions[example["episode_id"]], example)
            student_rows.append(student)
            evaluation_rows.append(evaluation)
        write_jsonl(data_root / f"{split}.jsonl", student_rows)
        write_jsonl(evaluation_root / f"{split}.jsonl", evaluation_rows)
        counts[split] = len(student_rows)
        prompt_sets[split] = {example["prompt_sha256"] for example in split_examples[split]}
        strategy_sets[split] = {example["strategy"] for example in split_examples[split]}

    prompt_overlap = {
        "train_valid": sorted(prompt_sets["train"] & prompt_sets["valid"]),
        "train_test": sorted(prompt_sets["train"] & prompt_sets["test"]),
        "valid_test": sorted(prompt_sets["valid"] & prompt_sets["test"]),
    }
    manifest = {
        "schema_version": "hfr.arcagi.visible_distillation.v1",
        "source_hfr_manifest_sha256": file_sha256(out / "training_export" / "manifest.json"),
        "purpose": "teacher imitation on the 52 visible ARC test entries",
        "counts": counts,
        "group_count": len(groups),
        "split_policy": {
            "test": "untouched source test entry (augmentation attempt 0)",
            "validation": "first distinct teacher-exact transformation",
            "train": "remaining distinct teacher-exact transformations",
        },
        "prompt_overlap": prompt_overlap,
        "prompt_exclusive": not any(prompt_overlap.values()),
        "strategy_counts": {split: len(strategy_sets[split]) for split in strategy_sets},
        "test_strategy_covered_by_train": sorted(strategy_sets["test"] - strategy_sets["train"]) == [],
        "family_exclusive": False,
        "benchmark_claim_allowed": False,
        "warning": (
            "Train and test intentionally share source task families. This view measures visible-task teacher "
            "distillation and transformation robustness, not generalization to unseen ARC tasks."
        ),
        "files": {},
    }
    for root_name, root in (("student", data_root), ("evaluation", evaluation_root)):
        for path in sorted(root.glob("*.jsonl")):
            manifest["files"][f"{root_name}/{path.name}"] = {
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
    if not manifest["prompt_exclusive"] or not manifest["test_strategy_covered_by_train"]:
        raise ValueError("visible-distillation admission checks failed")
    write_json(manifest_path, manifest)
    secure_tree(out)
    return manifest


def derive_visible_command(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    examples_path = out / "teacher_examples.sensitive.jsonl"
    if not examples_path.is_file():
        raise ValueError(f"missing generated teacher examples: {examples_path}")
    examples = [json.loads(line) for line in examples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = derive_visible_distillation(out, examples)
    print(json.dumps({"out": str(out), "counts": manifest["counts"], "group_count": manifest["group_count"]}, sort_keys=True))
    return 0


def generate(args: argparse.Namespace) -> int:
    arc_root = args.arc_root.resolve()
    out = args.out.resolve()
    require_fresh_output(out)
    agent, strategy_names, teacher = load_teacher(arc_root)
    examples, audit = build_examples(
        arc_root,
        agent,
        strategy_names,
        args.variants_per_example,
        args.max_attempt_multiplier,
    )
    dataset_revision = canonical_sha256(
        {"teacher": teacher, "sources": audit["source_tasks"], "schema_version": SCHEMA_VERSION}
    )
    tools = [strategy_tool(strategy_names, teacher["sha256"])]
    write_jsonl(out / "teacher_examples.sensitive.jsonl", examples)
    scenarios = out / "source" / "scenarios"
    traces = out / "source" / "traces"
    runs = out / "runs"
    scenarios.mkdir(parents=True)
    traces.mkdir(parents=True)
    runs.mkdir(parents=True)
    for index, row in enumerate(examples, start=1):
        scenario, trace = scenario_and_trace(row, tools, teacher, dataset_revision)
        scenario_path = scenarios / f"{row['episode_id']}.json"
        trace_path = traces / f"{row['episode_id']}.trajectory.jsonl"
        write_json(scenario_path, scenario)
        write_jsonl(trace_path, [trace])
        code = run_hfr(
            [
                "run",
                "--scenario",
                str(scenario_path),
                "--trace",
                str(trace_path),
                "--format",
                "trajectory_jsonl",
                "--out",
                str(runs / row["episode_id"]),
                "--fail-on-score",
            ]
        )
        if code != 0:
            raise RuntimeError(f"Flight Recorder rejected {row['episode_id']} with exit code {code}")
        if index % 100 == 0:
            print(f"recorded {index}/{len(examples)} validated trajectories", flush=True)

    export = out / "training_export"
    code = run_hfr(
        [
            "export-rl",
            "--runs",
            str(runs),
            "--out",
            str(export),
            "--metadata",
            f"arc_dataset_revision={dataset_revision}",
            "--metadata",
            f"teacher_sha256={teacher['sha256']}",
        ]
    )
    if code != 0:
        raise RuntimeError(f"Flight Recorder export failed with exit code {code}")
    validation_path = out / "hfr_validation.json"
    code = run_hfr(
        [
            "validate",
            "--runs",
            str(runs),
            "--training-export",
            str(export),
            "--out",
            str(validation_path),
        ]
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if code != 0 or validation.get("passed") is not True:
        raise RuntimeError(f"Flight Recorder validation failed: {validation}")
    student_manifest = derive_student_data(out, examples)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_revision": dataset_revision,
        "teacher": teacher,
        "strategy_order": strategy_names,
        "augmentation_policy": {
            "variants_per_visible_test_entry": args.variants_per_example,
            "max_attempt_multiplier": args.max_attempt_multiplier,
            "spatial_group": list(D4_NAMES),
            "color_policy": "deterministic permutations; rejected unless the executable teacher remains exact",
            "training_pair_policy": "deterministic identity/reverse/shuffle",
            "admission_gate": "teacher make_predictions must contain the transformed visible answer",
        },
        "audit": audit,
        "hfr_export_manifest_sha256": file_sha256(export / "manifest.json"),
        "hfr_validation_sha256": file_sha256(validation_path),
        "student_data_manifest_sha256": file_sha256(out / "student_data_manifest.json"),
        "student_counts": student_manifest["counts"],
        "security": {
            "publication_allowed": False,
            "contains_visible_test_outputs": True,
            "artifact_class": "sensitive-local-training-data",
        },
        "claims": [
            "No sealed or hidden ARC task was accessed.",
            "Augmented rows are derivatives, not independent ARC task families.",
            "This artifact does not establish benchmark improvement.",
        ],
    }
    write_json(out / "generation_manifest.json", manifest)
    secure_tree(out)
    print(json.dumps({"out": str(out), "trajectories": len(examples), "splits": student_manifest["counts"]}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("generate", help="Generate, record, export, and validate teacher trajectories")
    build.add_argument("--arc-root", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--variants-per-example", type=int, default=32)
    build.add_argument("--max-attempt-multiplier", type=int, default=8)
    build.set_defaults(func=generate)
    visible = subparsers.add_parser(
        "derive-visible",
        help="Add a prompt-exclusive visible-task imitation view to an existing generated artifact",
    )
    visible.add_argument("--out", type=Path, required=True)
    visible.set_defaults(func=derive_visible_command)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if getattr(args, "variants_per_example", 1) < 1:
        raise ValueError("variants-per-example must be positive")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
