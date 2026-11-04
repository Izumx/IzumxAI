# SMEL FT dense adaptation — 3 November 2026

## Outcome

Implemented and ran a dense Qwen3 extension using the actual frozen-head
`smel.loss.chunked_ce` from upstream commit
`74a4635d5f7733baae9915d291573fae94a0499b`. Frozen lower 20 layers are computed
once and cached; fresh rank-16 LoRA trains only gate/up/down projections of the
upper eight layers. The starting checkpoint is the prior merged IzumxAI model.
This is an adaptation of cached-prefix and chunked-loss mechanisms. Dense
Qwen3 has no MoE router or routed experts; upstream's expert-selection pipeline
is not being claimed or run. No large MoE model was downloaded.

The method reduced measured training memory, but did not improve general code
performance in this experiment. The experimental model is `izumxai:2b-smel`.

## Training and memory

* Official MBPP `full/train`: 367 references retained after execution and scope,
  length and evaluation-name checks; 168 short verified numeric replay examples.
* Validation: 32 separate MBPP validation tasks, 12 math and 12 logic examples.
* All admitted code references pass their available tests. This does not prove
  correctness outside those tests. MBPP attribution is Google Research / Austin
  et al. (2021), CC BY 4.0; changes and dataset fingerprints are in data metadata.
* 64 optimizer steps, accumulation 8, LR 1e-5, code sampling probability 0.7,
  seed 20261103. 3,145,728 new trainable parameters.
* Prefix cache: 48.4 seconds. Training including periodic validation: 139.0 seconds.
* Macro validation NLL: 1.53220 -> 1.47769; code NLL: 1.30081 -> 1.26881.
* PyTorch allocated peak: approximately 1.91 GiB, allocator cap 2.8 GiB.
* Device-wide sampled peak including desktop applications: **3428 MiB / 3.35 GiB**.
  Sampling interval approximately one second. Physical GPU was RTX 4050 Laptop
  with 6141 MiB. The measured run stayed below 4 GiB; a physical 4-GiB GPU was
  not tested and shorter peaks may be missed by sampling.
* Deployed Q4_K_M GGUF: **1,107,408,736 bytes**, approximately 1.03 GiB.

The real-weight full/cached probe gave NLL 0.27248 / 0.27338 and relative LoRA
gradient difference 0.02398, within the declared BF16 tolerance of 0.03.
Float32 tiny-model comparisons match normal numerical tolerance, including
padded batches. The one cold full probe versus one cached probe is not treated
as a fair throughput benchmark. These timings also cannot be directly compared
to Stage 2 training as a causal speedup: data, targets and trained layers differ.

## Functional evaluation

All 164 HumanEval tasks, one deterministic instruction response each, original
`check(candidate)` assertions, shared raw Qwen3 prompt with thinking disabled,
Q4_K_M, context 2048, response limit 512, temperature 0. Canonical imports/stub
are retained and overwritten by the generated complete function. This is an
adapted instruction protocol, not a directly comparable published completion
benchmark score. No training or checkpoint changes followed these scores.

| Model | Passed / 164 | Percentage |
|---|---:|---:|
| New SMEL dense IzumxAI | 78 | 47.56% |
| Previous IzumxAI | 79 | 48.17% |
| Base Qwen3-1.7B | 90 | 54.88% |

The external result does not support general coding improvement or superiority
over peer models. Prior regression in the starting IzumxAI has not been repaired
by this short continuation. Lower training loss alone is insufficient evidence.

The separate local Russian set scored 18/24 (code 9/12, math 6/6, logic 3/6),
matching the previous IzumxAI on these counts. Responses and timings are retained
in `runs/stage3-smel-20261103/local24.json`.

## Local improvements and provenance

The actual upstream loss now rejects invalid token IDs, invalid denominators
and chunks, and training through a head for which it cannot return gradients.
Four CPU tests cover these cases and full CE/gradient parity.
The patch is local, not submitted upstream:
`patches/smel-ft-74a4635-loss-validation.patch`. Apache LICENSE and NOTICE are
preserved alongside it. `scripts/09_smel_setup.ps1` prepares the pinned checkout.

The dense cache fingerprint includes model weight hashes, configuration,
tokenizer/template, token IDs, labels, sequence boundaries, cut, dtype, Torch,
Transformers, bitsandbytes and device capability. Nonfinite parity results and
nonfinite training gradients fail explicitly. Prefix removal preserves native
layer indices so PEFT adapters reload and merge into the intact base model.

20 project tests and four upstream CPU tests passed. CPU review additionally
checked MLP LoRA state reload/merge parity with nonreentrant checkpointing.

## Artifacts

* `runs/izumxai-smel-dense/train_meta.json`, `parity.json`, `validation.json`.
* `runs/stage3-smel-20261103/train.log`, `vram.csv`, `humaneval164.json`.
* `data/distill/stage3-smel/data_meta.json`, `train.jsonl`, `val.jsonl`.
* `data/eval/humaneval_stage3_full.jsonl` and existing HumanEval MIT license.
* `models/izumxai-smel/gguf/izumxai-2b-Q4_K_M.gguf`.
* Research: `docs/research/2026-11-03-smel-ft.md`.

```powershell
ollama run izumxai:2b-smel --think=false
# Reproduce in a new output directory:
powershell -ExecutionPolicy Bypass -File scripts/10_smel_dense.ps1 -Out runs/smel-reproduction
```

Ollama import initially failed because C: was full. The failed temporary blob
and the previous experiment's step-57 cache registration were removed; its GGUF
and adapters remain on E:. Other baseline registrations were preserved.

Sources: [SMEL FT](https://github.com/artikulus1301/smel-ft),
[MBPP dataset card](https://huggingface.co/datasets/google-research-datasets/mbpp),
[HumanEval](https://github.com/openai/human-eval).
