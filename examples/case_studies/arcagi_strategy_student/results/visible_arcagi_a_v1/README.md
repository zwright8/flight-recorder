# Visible ARC-AGI-A Strategy Distillation Evidence

This package publishes the reviewed, privacy-safe evidence for a local
Qwen3-0.6B strategy-router experiment. Hermes Flight Recorder recorded and
validated the teacher trajectories used for action SFT. The frozen base and
LoRA arms were then evaluated with the same prompts, model revision, decoding,
tool catalog, deterministic strategy executor, prediction cap, and exact-grid
scorer.

## Paired result

| Configuration | Exact strategy | Exact grid | Invalid actions |
| --- | ---: | ---: | ---: |
| Frozen Qwen3-0.6B | 2/52 (3.85%) | 2/52 (3.85%) | 2 |
| HFR LoRA, step 900 | 50/52 (96.15%) | 50/52 (96.15%) | 0 |
| HFR LoRA, step 1200 | 50/52 (96.15%) | 50/52 (96.15%) | 0 |
| Two-checkpoint LoRA ensemble | 52/52 strategy recall | 52/52 (100%) | 0 |

For one adapter, training raised exact-grid success by 48 queries, or 92.31
percentage points. That is 25 times the base success rate and a 96% reduction
in errors. The complementary two-checkpoint ensemble recovered the remaining
two queries, subject to the same maximum of three candidate grids.

`outcomes.jsonl` contains one anonymous row per query. Summing each boolean
column reproduces every numerator in `metrics.json`; the repository regression
test performs that replay. Query identifiers, grids, prompts, strategy names,
raw generations, local paths, traces, and model weights are intentionally not
published. The cryptographic hashes in `metrics.json` bind this sanitized
surface to the retained local receipts and source data.

## What the model does

This is a hybrid agent, not direct neural grid generation. The LLM emits one
strategy-tool selection. The deterministic ARC teacher executes that selected
strategy and returns up to three candidate grids. Exact-grid success requires
one candidate to exactly match the visible expected output.

## Claim boundary

This package is **not an official or sealed ARC benchmark result**. The 52
evaluated prompts were untouched, but augmented training trajectories derived
from the same 48 source task families. The result therefore demonstrates
visible-task teacher imitation and transformation robustness; it does not
establish generalization to novel ARC task families.

An earlier base-only evaluation on a different family-exclusive split scored
0/160. It is not used in the improvement calculation because a comparable
LoRA arm was not evaluated on that exact split. No official or sealed task was
accessed, and no adapter or dataset is published by this evidence package.

## Reproduction surface

- `metrics.json` records the aggregate metrics, run controls, and hashes.
- `outcomes.jsonl` independently replays the aggregate numerators without
  exposing task content.
- [`../../README.md`](../../README.md) documents the local generation,
  training, evaluation, and bounded-ensemble commands.
- [`../../evaluate_student.py`](../../evaluate_student.py) implements the
  common base/adapter evaluator.
- [`../../evaluate_ensemble.py`](../../evaluate_ensemble.py) implements the
  capped ensemble replay.

The raw HFR trajectories, ARC task contents, evaluation receipts, adapters,
and training logs remain local sensitive artifacts under the ignored `local/`
tree.
