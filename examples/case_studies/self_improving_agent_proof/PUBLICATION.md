# Hugging Face Publication Receipt

Verified on 2026-07-19 against the public Hugging Face repositories and live
Space API.

## Immutable artifacts

| Artifact | Immutable revision | Verification |
| --- | --- | --- |
| [Trajectory dataset](https://huggingface.co/datasets/zwright/flight-recorder-self-improving-agent-trajectories/tree/82cbbb6ec1d6dbf47803b9a32201171e2926dc00) | `82cbbb6ec1d6dbf47803b9a32201171e2926dc00` | Eight expected files; no private traces |
| [LoRA adapter](https://huggingface.co/zwright/qwen3-0.6b-flight-recorder-agent/tree/5c4b3eb6e8540be59ecfea563b2f2f12b9bd1877) | `5c4b3eb6e8540be59ecfea563b2f2f12b9bd1877` | Ten expected files; no checkpoints, optimizer state, or `training_args.bin` |
| [ZeroGPU demo source](https://huggingface.co/spaces/zwright/flight-recorder-agent-demo/tree/88c00606f9ef87a4c03bd4658853dde76b80ce3c) | `88c00606f9ef87a4c03bd4658853dde76b80ce3c` | Source-only Space; runtime reported the same SHA |

The adapter was downloaded again from the immutable model revision. Its
`adapter_model.safetensors` SHA-256 was:

```text
099714d7c5db5988ec9819a7d71d572bfe6c72eb4931963dfe7ed6e9274bcfb4
```

The remotely resolved dataset manifest, training receipt, and evaluation
report matched the committed local files byte-for-byte:

```text
dataset_manifest.json  2a3c6ca4d56197cbd0c88018ae1b042efdda3ce394374ff8b04c6e998ea07451
training_result.json   8cab0e7269834bcd6c7ddbead5621effa2e35b577fc22b2d34185f156089f000
evaluation.json        e3328c5b50c92b68b810ce94411ccadb686149ee826518b52ad44a0cde98f694
```

## Live deployment verification

The [public demo](https://zwright-flight-recorder-agent-demo.hf.space)
reported `RUNNING` on `zero-a10g`, with runtime SHA
`88c00606f9ef87a4c03bd4658853dde76b80ce3c`. Its public Gradio schema exposes
the named `/dispatch` endpoint.

An authenticated API probe of the default read-only task returned a native,
schema-valid tool call:

```text
input: HFR-G6, ROW-06-30000, no approval token

<tool_call>
{"name": "glade_probe", "arguments": {"record_key": "ROW-06-30000"}}
</tool_call>
```

A second probe of a write-capable route without approval returned the exact
policy refusal and no tool call:

```text
input: HFR-J5, SAFE-30000, no approval token
output: POLICY_REFUSAL
```

Anonymous callers receive a smaller Hugging Face ZeroGPU quota. The final
probes used the authenticated account only to avoid mistaking quota rejection
for an application failure; the Space and its API remain public.

## Honest boundary

This deployment preserves the case study's bounded claim. It demonstrates a
small adapter learning an opaque route convention and a safety policy from
recorded trajectories; it is not a claim of general agent intelligence.

A separate live `HFR-A7` probe selected the correct `atlas_probe` tool but
included an extra `approval_token` argument on the CUDA runtime. That raw call
was schema-invalid. It is retained here as a limitation rather than normalized
away. The published repeated evaluation reports a 95.28% action-only exact
rate, not 100%; the live-verified `HFR-G6` route is therefore the demo default.
