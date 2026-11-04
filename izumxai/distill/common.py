"""Общие утилиты distill-пайплайна."""

from __future__ import annotations

import torch

DEFAULT_SYSTEM = (
    "Ты — IzumxAI, точный и краткий ассистент. Отвечай по-русски, "
    "код — в блоках ```python, в конце решения строка «Ответ: <значение>»."
)


def response_loss(model, ids, labels, attention_mask):
    """Project only positions with supervised next tokens (Qwen3 logits_to_keep).

    Prompt positions still participate in attention and backpropagation, but their
    unused vocabulary logits are never materialized. Supports padded batches.
    """
    positions = (labels[:, 1:] != -100).any(dim=0).nonzero(as_tuple=True)[0]
    if not positions.numel():
        raise ValueError("response has no supervised next-token targets")
    logits = model(input_ids=ids, attention_mask=attention_mask, use_cache=False,
                   logits_to_keep=positions).logits
    targets = labels[:, positions + 1]
    return torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)


def cast_frozen_embeddings(model, dtype):
    """Undo k-bit preparation's FP32 upcast for frozen vocabulary matrices only."""
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module is not None and all(not p.requires_grad for p in module.parameters()):
            module.to(dtype=dtype)


def pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def print_vram(tag: str = "") -> None:
    if not torch.cuda.is_available():
        print(f"[vram] {tag}: cpu")
        return
    used = torch.cuda.memory_allocated() / 2**30
    peak = torch.cuda.max_memory_allocated() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"[vram] {tag}: занято {used:.2f} GiB, пик {peak:.2f} GiB из {total:.1f} GiB")


def configure_vram_budget(gib: float) -> None:
    """Cap the PyTorch allocator. Driver/other apps/bnb allocations are separate."""
    if not torch.cuda.is_available():
        raise ValueError("QLoRA training requires CUDA")
    total = torch.cuda.get_device_properties(0).total_memory
    if not 0 < gib * 2**30 <= total:
        raise ValueError(f"VRAM budget must be between 0 and {total / 2**30:.2f} GiB")
    torch.cuda.set_per_process_memory_fraction(gib * 2**30 / total, 0)
    torch.cuda.reset_peak_memory_stats()
    print(f"PyTorch allocator budget: {gib:.2f} GiB (other GPU allocations are separate)")


def build_labeled_example(tokenizer, messages: list[dict], max_len: int):
    """Собирает ids + labels, маскируя всё до ответа ассистента."""
    msgs = [m for m in messages if m.get("role") in {"system", "user", "assistant"}]
    if not msgs or msgs[-1]["role"] != "assistant":
        return None

    prompt_text = tokenizer.apply_chat_template(
        msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full_text = tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if not full_ids or len(full_ids) > max_len:
        # A verified answer must remain complete and attached to its prompt.
        return None
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template: prompt is not a prefix of the completed conversation")
    cut = len(prompt_ids)
    labels = [-100] * cut + full_ids[cut:]
    if all(l == -100 for l in labels[1:]):
        return None

    return {"input_ids": full_ids, "labels": labels}


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok
