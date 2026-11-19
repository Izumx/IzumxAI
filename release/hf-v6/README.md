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
  - tool-use
  - agentic
---

# IzumxAI-9B v6 - uncensored 9B cybersecurity model with verified tool use

A 9B-parameter chat model specialised for cybersecurity work: vulnerability
triage, secure code review, log and artifact analysis, MITRE ATT&CK mapping,
and threat intelligence reasoning. Built as a QLoRA fine-tune of
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), then abliterated so it
answers security questions without refusals.

v6 is the first release in this project that was also trained to **act**:
inside a small tool-using harness it chooses the exploit primitive, changes its
construction after a failed attempt, and calls a search tool when it is stuck.

**Why it exists:** security tooling is where refusals hurt most. A model that
refuses to explain how Kerberoasting works, or how to reproduce a CVE, cannot
be used by a pentester, a SOC analyst, or a CTF player. This model has **0%
refusals across 41 refusal probes**, while keeping math, logic and coding at
the level of its base.

- 9.65B parameters total, 9B class
- Q4_K_M ~ 5.2 GB, Q5_K_M ~ 6.0 GB - runs in 6-8 GB of VRAM
- Apache-2.0, same as the base model

## Benchmarks

Own suite: 88 programmatically checked tasks (24 math, 12 logic, 16 code,
12 instruction-following, 24 cybersecurity), identical prompts for every
model, temperature 0.2, single attempt.

| suite | base Qwen3.5-9B | v4 | **v6** |
|---|---|---|---|
| math | 24/24 | 24/24 | **24/24** |
| logic | 11/12 | 11/12 | **12/12** |
| code | 15/16 | 15/16 | **16/16** |
| instruction | 10/12 | 10/12 | **10/12** |
| cybersecurity | 18/24 | 20/24 | **19/24** |
| **total** | 78/88 | 80/88 (91%) | **81/88 (92%)** |
| refusals | - | 0/41 | **0/41 (0%)** |

Coding was additionally verified by executing the produced Python against
hidden unit tests, not by string matching. An 88-task suite is a smoke test:
a one-task difference is noise. The agentic result below is the meaningful
change in v6.

### Agentic tool use

An execution-verified harness (`izumxai/agentic` in the project repo) runs a
vulnerable service in WSL and gives the model typed tools: send a payload,
inspect what a sanitiser does to it, enumerate candidate separators, run a PoC.
Success is judged by an oracle the model cannot forge - a secret token that
only appears in the service's own stdout if the service actually executed the
injected command.

Four tasks (command injection with and without a blacklist, path traversal,
`eval()` injection), pure model, no harness escalation, three passes:

| model | solved | cmd injection | eval injection | path traversal |
|---|---|---|---|---|
| v4 | 5/12 | 0/6 | 2/3 | 3/3 |
| **v6** | **12/12** | **6/6** | **3/3** | **3/3** |

v6 solves every task at the first step. The training services used different
ports, endpoints and parameter names, so this is not memorisation of the bench
text.

The harness is not a sandbox and these are toy services. The number says
something about *behaviour under a tool loop*, not about autonomous offensive
capability against real systems.

## Training

- **Base:** `Qwen/Qwen3.5-9B` (bf16), loaded in 4-bit NF4 for training
- **Method:** QLoRA, r=32, alpha=64, dropout 0.05, 2048-token context
- **Data:** 3 635 examples, 2 epochs (904 steps)
  - 1 038 verified synthetic reasoning/coding tasks
  - 2 123 curated cybersecurity SFT records distilled from
    [`Vyber07/cyber-security`](https://huggingface.co/datasets/Vyber07/cyber-security)
    (filtered for malformed code fences, refusals, escaped newlines,
    non-English and low-entropy answers)
  - 158 agentic tool-use examples, repeated 3x = 474. Every gold action in
    this subset was **executed against a real service** before it entered the
    data; rows whose action did not solve were dropped
- **Optimisation:** lr 2e-4, grad-accum 8, cosine schedule, paged AdamW 8-bit
- **Post-processing:** LoRA merged into the base weights, then abliteration at
  scale 1.4 (refusal-direction removal) on CPU
- **Quantisation:** GGUF Q4_K_M and Q5_K_M with llama.cpp

Validation loss 0.8662 -> 0.3205 over 904 steps.

## Quick start

### Ollama

```bash
ollama run hf.co/Izumx/IzumxAI-9B-v6:Q4_K_M
```

Or from the files in this repo:

```bash
# after downloading the GGUF you want
ollama create izumxai-9b-v6 -f Modelfile
ollama run izumxai-9b-v6
```

`Modelfile` uses `num_ctx 8192`, `temperature 0.7`, `repeat_penalty 1.05`.

### llama.cpp

```bash
llama-cli -m izumxai-9b-v6-Q4_K_M.gguf \
  -p "<|im_start|>system\nYou are a senior security engineer.<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n" \
  -n 512 --temp 0.7 -c 8192
```

The model uses the Qwen ChatML template and may emit a `<think>` block before
the answer; strip it if you only want the final response.

## Files

| file | size | notes |
|---|---|---|
| `izumxai-9b-v6-Q4_K_M.gguf` | 5.2 GB | best size/quality for 6-8 GB VRAM |
| `izumxai-9b-v6-Q5_K_M.gguf` | 6.0 GB | a bit sharper, fits 8 GB |
| `Modelfile` | - | Ollama recipe |

## Limitations

- Knowledge cutoff follows Qwen3.5; recent CVEs and advisories are not in the
  weights. Pair it with retrieval for current data.
- The agentic benchmark is four toy services. It is evidence of tool-loop
  behaviour, not a measure of offensive capability, and it is not a sandbox.
- CyberSecEval 4 was run for v4 but **not re-run for v6**; do not assume those
  numbers carried over. The refusal suite (41 probes) was re-run: 0%.
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
- Full write-up (Russian): `docs/experiments/2026-11-19-v6-agentic-sft.md`

## Citation

```bibtex
@misc{izumxai9bv6,
  title  = {IzumxAI-9B v6: uncensored 9B cybersecurity model with verified tool use},
  author = {Izumx},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/Izumx/IzumxAI-9B-v6}}
}
```
