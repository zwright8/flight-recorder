# ARC-AGI-2 Kaggle Router-Only Evidence

This privacy-safe package records the controlled Kaggle-native comparison of
the frozen Qwen3-0.6B strategy router and its step-900 HFR LoRA adapter. Both
arms use the same base weights, compact prompt, tool catalog, deterministic
decoding, selected strategy executor, challenge file, and output normalizer.
The only arm difference is whether the frozen LoRA adapter is enabled.

The executable ARC teacher generated and validated the training trajectories,
but it is not used as a fallback during evaluation. That boundary matters:
an earlier hybrid attempt made both visible arms score 52/52 and therefore
masked the adapter's effect. It also stalled while searching the full teacher
strategy list on a sealed input. The router-only repair measures the student
and its selected deterministic tool directly.

## Controlled visible result

| Arm | Exact grid | Invalid actions | Execution errors |
| --- | ---: | ---: | ---: |
| Frozen base router | 0/52 (0%) | 1 | 0 |
| HFR LoRA, step 900 | 17/52 (32.69%) | 0 | 0 |

The LoRA adds 17 exact answers and 32.69 percentage points under the identical
router-only contract. This replay uses the same visible source families that
contributed transformed training trajectories, so it measures teacher
imitation rather than novel-family benchmark generalization.

## Sealed route status

Kaggle notebook version 7 completed label-blind inference over 240 sealed tasks
and 259 test inputs per arm. Both hidden artifact gates passed with zero
execution errors. Base produced 14 invalid strategy actions; LoRA produced
zero. The promoted files are independently hash-bound in `metrics.json`.

Kaggle submission 55427691 references the LoRA `submission.json` from the
completed private notebook. Its official score is pending. Kaggle permits one
submission per day for this competition; the base artifact is retained but
cannot receive an official score until the daily quota resets. No official
sealed improvement claim is made before both scores exist.

## Publication boundary

This package contains only aggregate counts, hashes, model provenance, and
submission status. It excludes ARC grids, prompts, task identifiers, raw model
outputs, selected strategy names, trajectories, adapters, private paths, and
Kaggle output files. Private receipts remain local sensitive evidence.

- [`metrics.json`](metrics.json) is the canonical sanitized result.
- [`metrics_summary.csv`](metrics_summary.csv) is the compact arm/status table.
- [`SHA256SUMS`](SHA256SUMS) fingerprints those machine-readable artifacts.
- [`../../README.md`](../../README.md) documents the governed route and its
  negative evidence.
