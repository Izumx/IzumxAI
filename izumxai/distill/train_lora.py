"""Дистилляция IzumxAI:2B — QLoRA на Qwen3-1.7B по данным от учителя.

Учитель (qwen3:8b) даёт ответы, проверенные программно; ученик 1.7B
учится их воспроизводить. Влезает в 6 ГБ VRAM: 4-bit база + LoRA,
gradient checkpointing, batch 1 и накопление градиента.

python -m izumxai.distill.train_lora --data data/distill/train.jsonl --out runs/izumxai-2b-lora
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .common import (
    build_labeled_example, load_tokenizer, pick_dtype, print_vram,
)


class ChatSFTDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, max_len: int):
        self.items = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages") or []
            if len(msgs) < 2:
                continue
            ex = build_labeled_example(tokenizer, msgs, max_len)
            if ex is not None:
                self.items.append(ex)
        if not self.items:
            raise SystemExit(f"нет валидных примеров в {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data", default="data/distill/train.jsonl")
    ap.add_argument("--val", default="")
    ap.add_argument("--out", default="runs/izumxai-2b-lora")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = pick_dtype()
    tok = load_tokenizer(args.model)

    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    print(f"загружаю {args.model} ({'4-bit' if quant else 'bf16'})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        torch_dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if quant is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    print_vram("после загрузки")

    ds = ChatSFTDataset(args.data, tok, args.max_len)
    print(f"примеров: {len(ds)}")
    val = ChatSFTDataset(args.val, tok, args.max_len) if args.val else None

    steps_per_epoch = max(1, len(ds) // (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_ratio))

    try:
        from bitsandbytes.optim import PagedAdamW8bit
        opt = PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        print("оптимизатор: PagedAdamW8bit")
    except Exception:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        print("оптимизатор: AdamW (fp32)")

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    model.train()
    t0 = time.time()
    step, micro, seen, loss_acc = 0, 0, 0, 0.0
    order = list(range(len(ds)))
    random.shuffle(order)
    ptr = 0
    while step < total_steps:
        batch = []
        for _ in range(args.batch_size):
            if ptr >= len(order):
                random.shuffle(order)
                ptr = 0
            batch.append(ds[order[ptr]])
            ptr += 1
        ids, labels, attn = collate(batch, tok.pad_token_id)
        ids, labels, attn = ids.cuda(), labels.cuda(), attn.cuda()

        with torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
            loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
        (loss / args.grad_accum).backward()
        loss_acc += loss.item()
        seen += int(attn.sum())
        micro += 1

        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                dt = time.time() - t0
                avg = loss_acc / (args.log_every * args.grad_accum)
                print(f"step {step}/{total_steps} | loss {avg:.4f} | lr {lr:.2e} | "
                      f"{seen / dt:,.0f} tok/s | {dt / 60:.1f} мин", flush=True)
                loss_acc = 0.0
            if step % args.save_every == 0:
                model.save_pretrained(out / f"step_{step}")
                print(f"  адаптер сохранён: {out / f'step_{step}'}")

    model.save_pretrained(out / "final")
    tok.save_pretrained(out / "final")
    meta = {
        "base_model": args.model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "steps": step,
        "train_examples": len(ds),
        "max_len": args.max_len,
        "tokens_seen": seen,
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"готово за {(time.time() - t0) / 60:.1f} мин -> {out / 'final'}")
    print_vram("после обучения")


if __name__ == "__main__":
    main()
