"""Общие утилиты distill-пайплайна."""

from __future__ import annotations

import torch

DEFAULT_SYSTEM = (
    "Ты — IzumxAI, точный и краткий ассистент. Отвечай по-русски, "
    "код — в блоках ```python, в конце решения строка «Ответ: <значение>»."
)


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


def build_labeled_example(tokenizer, messages: list[dict], max_len: int):
    """Собирает ids + labels, маскируя всё до ответа ассистента."""
    msgs = [m for m in messages if m.get("role") in {"system", "user", "assistant"}]
    if not msgs or msgs[-1]["role"] != "assistant":
        return None

    prompt_text = tokenizer.apply_chat_template(
        msgs[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(msgs, tokenize=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if not full_ids:
        return None

    # обрезаем длинные примеры, но не отрезаем начало ответа
    if len(full_ids) > max_len:
        keep = max_len
        start = max(0, len(full_ids) - keep)
        if start > len(prompt_ids):
            full_ids = full_ids[start:]
            labels = full_ids[:]
        else:
            full_ids = full_ids[:keep]
            cut = min(len(prompt_ids), keep)
            labels = [-100] * cut + full_ids[cut:]
    else:
        cut = min(len(prompt_ids), len(full_ids))
        labels = [-100] * cut + full_ids[cut:]

    if all(l == -100 for l in labels):
        return None

    return {"input_ids": full_ids, "labels": labels}


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok
