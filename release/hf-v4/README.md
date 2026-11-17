---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
language:
  - en
  - ru
library_name: gguf
pipeline_tag: text-generation
tags:
  - cybersecurity
  - abliterated
  - uncensored
  - gguf
  - qwen3.5
  - qlora
  - ollama
---

# IzumxAI-9B v4 — uncensored 9B cybersecurity model

A 9B-parameter chat model specialised for cybersecurity work: vulnerability
triage, secure code review, log and artifact analysis, MITRE ATT&CK mapping,
and threat intelligence reasoning. Built as a QLoRA fine-tune of
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) on curated cybersecurity
data, then abliterated so it answers security questions without refusals.

**Why it exists:** security tooling is where refusals hurt most. A model that
refuses to explain how Kerberoasting works, or how to reproduce a CVE, cannot
be used by a pentester, a SOC analyst, or a CTF player. This model has **0%
refusals across 41 refusal probes**, while keeping math, logic and coding at
the level of its base.

- 9.65B parameters total, 9B class
- Q4_K_M ≈ 5.2 GB, Q5_K_M ≈ 6.0 GB — runs in 6–8 GB of VRAM
- Apache-2.0, same as the base model

## Benchmarks

Own suite: 88 programmatically checked tasks (24 math, 12 logic, 16 code,
12 instruction-following, 24 cybersecurity), identical prompts for every
model, temperature 0.2, single attempt.

| suite | base Qwen3.5-9B | izumxai:9b v3 | **v4** |
|---|---|---|---|
| math | 24/24 | 24/24 | **24/24** |
| logic | 11/12 | 12/12 | **11/12** |
| code | 15/16 | 15/16 | **15/16** |
| instruction | 10/12 | 10/12 | **10/12** |
| cybersecurity | 18/24 | 21/24 | **20/24** |
| **total** | 78/88 | 82/88 | **80/88 (91%)** |
| refusals | — | 6/41 (15%) | **0/41 (0%)** |

Coding was additionally verified by executing the produced Python against
hidden unit tests, not by string matching.

### PurpleLlama CyberSecEval 4

Judged by a local 27B model (`Huihui-Qwen3.8-27B-abliterated`), so numbers are
*relative* and not comparable with published leaderboard values. Trimmed
subsamples, single run, temperature 0.6.

| benchmark | cases | result |
|---|---|---|
| MITRE FRR (refusal) | 100 | 100 accepted, **0% refusals** |
| Secure Code Instruct | 100 | 9% vulnerable suggestions |
| Secure Code Autocomplete | 100 | 8% vulnerable suggestions |
| Textual Prompt Injection | 100 | 71% injected |
| Code Interpreter Abuse | 60 | 83% classified malicious |
| MITRE ATT&CK | 50 | 10/50 benign, 0 refusals |
| Canary Exploit | 120 | 0.00 |
| Malware Analysis | 80 | 23.8% correct multiple choice |
| Threat Intel Reasoning | 60 | 36.4% correct multiple choice |
| Spear Phishing | 10 | 2.1 / 5 |

The high prompt-injection and interpreter-abuse rates are the direct price of
abliteration: the model follows embedded instructions and will help with
offensive requests. Treat it as a tool for authorised work, and put your own
guardrails in front of it.

## Training

- **Base:** `Qwen/Qwen3.5-9B` (bf16), loaded in 4-bit NF4 for training
- **Method:** QLoRA, r=32, alpha=64, dropout 0.05, 2048-token context
- **Data:** 3 161 examples — 1 038 verified synthetic reasoning/coding tasks
  plus 2 200 curated cybersecurity SFT records distilled from
  [`Vyber07/cyber-security`](https://huggingface.co/datasets/Vyber07/cyber-security)
  (filtered for malformed code fences, refusals, escaped newlines, non-English
  and low-entropy answers), 2 epochs
- **Optimisation:** lr 2e-4, grad-accum 8, cosine schedule, paged AdamW 8-bit
- **Post-processing:** LoRA merged into the base weights, then abliteration at
  scale 1.4 (refusal-direction removal) on CPU
- **Quantisation:** GGUF Q4_K_M and Q5_K_M with llama.cpp

Validation loss 0.8662 → 0.3198 over 786 steps.

## Quick start

### Ollama

```bash
ollama run hf.co/Izumx/IzumxAI-9B-v4:Q4_K_M
```

Or from the files in this repo:

```bash
# after downloading the GGUF you want
ollama create izumxai-9b-v4 -f Modelfile
ollama run izumxai-9b-v4
```

`Modelfile` uses `num_ctx 8192`, `temperature 0.7`, `repeat_penalty 1.05`.

### llama.cpp

```bash
llama-cli -m izumxai-9b-v4-Q4_K_M.gguf \
  -p "<|im_start|>system\nYou are a senior security engineer.<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n" \
  -n 512 --temp 0.7 -c 8192
```

The model uses the Qwen ChatML template and may emit a `<think>` block before
the answer; strip it if you only want the final response.

## Files

| file | size | notes |
|---|---|---|
| `izumxai-9b-v4-Q4_K_M.gguf` | 5.2 GB | best size/quality for 6–8 GB VRAM |
| `izumxai-9b-v4-Q5_K_M.gguf` | 6.0 GB | a bit sharper, fits 8 GB |
| `Modelfile` | — | Ollama recipe |

## Limitations

- Knowledge cutoff follows Qwen3.5; recent CVEs and advisories are not in the
  weights. Pair it with retrieval for current data.
- Benchmarks are small (88 tasks) — a two-task difference is noise. Treat the
  numbers as a smoke test, not a ranking.
- The CyberSecEval numbers above come from a local judge model and trimmed
  subsamples; they are internally comparable but not official.
- Abliteration removes refusals, not capability. The model can be wrong with
  full confidence, including about exploit details and CVSS vectors.
- AutoPatch, Autonomous Offensive Cyber Ops and Visual Prompt Injection
  benchmarks were **not** run: they need Podman with multi-TB storage, an AWS
  cyber range, and a vision model respectively.

## Intended use and safety

Intended for authorised security work: pentests, CTFs, SOC triage, incident
response, secure code review, and security education. The model is uncensored
by design; that is the point of the release. You are responsible for how you
use it. Do not deploy it as an unattended agent with access to production
systems, and keep human review on anything destructive.

## Links

- Project and training code: [github.com/Izumx/IzumxAI](https://github.com/Izumx/IzumxAI)
- Full write-up (Russian): `docs/experiments/2026-11-16-v4-vyber-cybersec.md`

## Citation

```bibtex
@misc{izumxai9bv4,
  title  = {IzumxAI-9B v4: uncensored 9B cybersecurity model},
  author = {Izumx},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/Izumx/IzumxAI-9B-v4}}
}
```
