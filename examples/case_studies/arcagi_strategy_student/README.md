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

A second [Kaggle-native evidence package](results/kaggle_arcagi2_router_only_v1/README.md)
records the controlled Linux PEFT replay. Under an identical router-only
contract, its frozen base scored 0/52 and its step-900 HFR LoRA scored 17/52
(32.69%), with zero execution errors. LoRA submission 55427691 completed, but
Kaggle returned no public or private score; no sealed improvement claim is made
without scored results for both arms.

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

## Private Kaggle sealed route

The published MLX checkpoint is not promoted directly to Kaggle. Although its
weights can be converted exactly into PEFT tensor layout, the converted
checkpoint did not preserve the visible-query behavior on Linux CPU. A
100-step Linux calibration and a clean 100-step Linux adapter also failed the
controlled 12-query promotion probe at 4/12. These are negative portability
results, not hidden benchmark results.

The governed Kaggle route retrains PEFT checkpoints natively in Kaggle's Linux
environment. Its preserved negative evidence includes a full-context CPU pilot
that stopped at 263/900 after five hours, a T4 adapter reload stall after a
successful 900-step run, a repaired 4,096-token run that excluded 210/1,457
rows and scored only 19/52 at pass@2, and two nonreproducible 8,192-token T4
attempts. None accessed hidden tasks or produced a competition submission.

The successful training route losslessly converts recorded native JSON grids
to versioned `compact-grid-v1` row strings. All 1,457 rows fit under its
4,608-token gate (maximum 4,518), and the original HFR rows remain unchanged.
It completed the exact 900-step checkpoint and a bounded 300-step continuation.
The two checkpoints scored 17/52 and 15/52 individually on the visible replay;
their pass@2 union scored 20/52, so that training route correctly failed closed
before hidden inference.

For the scientific base-versus-LoRA comparison, a private runtime overlay
freezes the step-900 adapter and loads it once with exact tensor parity. One
resident T4 model then toggles the adapter off and on, holding prompt, tools,
context, decoding, executor, normalization, and challenge bytes fixed. The
executable teacher remains the source of HFR training labels but is not an
inference fallback: an earlier fallback pilot made both visible arms 52/52,
masking the adapter effect, and later stalled inside the full teacher search on
a sealed input. That failed pilot is retained as negative evidence.

The repaired router-only route reproduced a controlled visible result of 0/52
for base and 17/52 for step-900 LoRA, then ran label-blind inference over 240
sealed tasks and 259 inputs per arm. Both hidden artifact gates passed with
zero execution errors before `submission.json` files were promoted. The hidden
challenge is never sent to the visible scorer, no hidden solution is available
to the notebook, and no Apple GPU is used.

Build the private dataset and private GPU-kernel directories locally:

```bash
.venv/bin/python examples/case_studies/arcagi_strategy_student/build_kaggle_bundle.py \
  --repo-root "$PWD" \
  --arc-root /absolute/path/to/ArcAgi_StarterCode_v1.3.0 \
  --training-data local/arcagi_strategy_student/run-v1/visible_distillation_data/train.jsonl \
  --visible-challenges local/arcagi_strategy_student/kaggle-v1/visible-label-blind-challenges.json \
  --visible-solutions local/arcagi_strategy_student/kaggle-v1/visible-solutions.json \
  --base-model /absolute/path/to/Qwen3-0.6B/pinned-snapshot \
  --wheel-dir local/arcagi_strategy_student/kaggle-v1/vendor-wheels \
  --dataset-id kaggle_owner/hfr-arcagi-strategy-private-v1 \
  --kernel-id kaggle_owner/hfr-arcagi-strategy-sealed-v1 \
  --out local/arcagi_strategy_student/kaggle-v1/upload-bundle
```

The builder refuses a nonempty destination, pins the reviewed teacher and
training export hashes, dereferences the offline model snapshot, inventories
every runtime payload file (excluding Kaggle-rewritten control-plane metadata),
pins offline PEFT/Accelerate wheels, and marks the bundle as private sensitive
input. It
includes only the minimum student rows needed for retraining; raw Flight
Recorder traces are excluded.

After reviewing the manifest, create the dataset with Kaggle's private flag and
push the private, GPU-enabled, internet-disabled kernel:

```bash
kaggle datasets create \
  -p local/arcagi_strategy_student/kaggle-v1/upload-bundle/dataset \
  --dir-mode zip

kaggle kernels push \
  --accelerator NvidiaTeslaT4 \
  -p local/arcagi_strategy_student/kaggle-v1/upload-bundle/kernel
```

Kaggle datasets are private by default. Do not add the CLI's `--public` flag;
`--dir-mode zip` is required so the nested offline runtime and model files are
included instead of skipped. The explicit T4 selector avoids Kaggle's default
P100, whose compute capability is unsupported by the current PyTorch image;
the runtime also rejects any CUDA device below compute capability 7.0.

The private export retains all 1,457 governed rows. The T4 training contract
losslessly maps each recorded native JSON prompt to `compact-grid-v1`, requires
every complete rendered prompt to fit within 4,608 tokens, and fails closed if
any row would be excluded. It does not truncate a grid, prompt, or assistant
label. The encoding, retained-row count, and fraction are recorded in each
training receipt. To bound memory, the trainer projects vocabulary
logits only for the contiguous supervised assistant suffix (plus its causal
predecessor), instead of materializing unused logits for masked prompt tokens.
This changes neither the retained input context nor the supervised-token loss.

Do not publish the private dataset, notebook outputs, adapter checkpoints,
visible solutions, or deterministic coursework executor. Only a sanitized
Kaggle score receipt may be considered for the public evidence package after
the external score is independently verified. This hybrid system uses a LoRA
router plus client-side deterministic strategy tools, so it is a Kaggle system
entry rather than a standalone ARC Verified-model submission.
