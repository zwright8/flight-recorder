# ARC-AGI Strategy Student

This case study distills an executable ARC solver into a small local tool-using
student. The student selects one deterministic strategy tool; the original
`ArcAgent` executes that strategy and returns up to three grids.

The data builder applies deterministic color, D4 spatial, and training-pair
order transformations. A transformed example is admitted only when the
unmodified executable teacher still returns the transformed visible answer.
All derivatives of one source ARC task use that task ID as their Hermes Flight
Recorder `task_family`, so train, validation, and test remain family-exclusive.

Generated traces and grids are sensitive local artifacts. Write them below the
repository's ignored `local/` directory and do not publish them.

## Result

The reviewed run generated 1,561 teacher-exact Flight Recorder trajectories
from 48 source task families. On the same 52 untouched visible queries, the
frozen Qwen3-0.6B base solved 2/52 (3.85%) and either individual HFR-trained
LoRA checkpoint solved 50/52 (96.15%). That controlled single-adapter change is
+48 exact queries, +92.31 percentage points, 25 times the base success rate,
and a 96% reduction in errors. A bounded ensemble of the complementary step-900
and step-1200 checkpoints solved 52/52.

These numbers measure visible-family teacher imitation, not unseen-task or
official ARC benchmark generalization. The prompts are untouched, but
transformations from the same source task families occur in training. See the
[sanitized evidence package](results/visible_arcagi_a_v1/README.md) for
anonymous per-query outcomes, aggregate replay, provenance hashes, and the
complete claim boundary.

```bash
.venv/bin/python examples/case_studies/arcagi_strategy_student/pipeline.py generate \
  --arc-root /absolute/path/to/ArcAgi_StarterCode_v1.3.0 \
  --out local/arcagi_strategy_student/run-001 \
  --variants-per-example 32

.venv/bin/python examples/case_studies/arcagi_strategy_student/pipeline.py derive-visible \
  --out local/arcagi_strategy_student/run-001

python examples/case_studies/arcagi_strategy_student/train_local.py \
  --model /absolute/path/to/a/cached/Qwen3-0.6B/snapshot \
  --data local/arcagi_strategy_student/run-001/student_data \
  --adapter local/arcagi_strategy_student/run-001/adapters/qwen3-0.6b-lora

python examples/case_studies/arcagi_strategy_student/evaluate_student.py \
  --arc-root /absolute/path/to/ArcAgi_StarterCode_v1.3.0 \
  --data local/arcagi_strategy_student/run-001/evaluation/test.jsonl \
  --model /absolute/path/to/a/cached/Qwen3-0.6B/snapshot \
  --adapter local/arcagi_strategy_student/run-001/adapters/qwen3-0.6b-lora \
  --out local/arcagi_strategy_student/run-001/evaluations/student-test.json
```

The validated deployment agent is `student_agent.ArcStrategyStudent`. It loads
two checkpoints of the same 0.6B base model, asks each for one strategy, and
returns no more than three deduplicated deterministic predictions. The
two-checkpoint choice and ordering are bound in the generated handoff receipt.

The receipt's `grid_any_match_rate` is the important hybrid-agent metric. It
counts a task only when one of the selected deterministic strategy's first
three predictions exactly matches the visible transformed answer. Its claim
scope depends on the selected data view: `student_data/` is family-exclusive,
whereas `visible_distillation_data/` intentionally shares task families across
splits. Neither is an official or sealed ARC benchmark score.

`student_data/` is the scientific task-family-exclusive view. The optional
`visible_distillation_data/` view instead trains on transformations from every
visible task and tests on the untouched visible prompts. It is useful for
building the requested local imitation model, but its manifest deliberately
marks benchmark claims as disallowed because source families occur in train.

The trainer wrapper forces MLX's single-process ring backend. This avoids an
abort in Conda environments that expose MPICH, which MLX cannot use as its MPI
backend, and does not alter the installed MLX package.
