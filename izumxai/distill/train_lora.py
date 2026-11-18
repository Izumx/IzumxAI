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
    auto_model_class, build_labeled_example, configure_vram_budget, load_tokenizer, pick_dtype,
    print_vram, response_loss, cast_frozen_embeddings,
)


class ChatSFTDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, max_len: int):
        self.items = []
        self.skipped = 0
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
            else:
                self.skipped += 1
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


@torch.no_grad()
def evaluate(model, dataset, pad_id: int, dtype) -> float:
    """Token-weighted response NLL, without retaining graphs or changing train mode."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    loss_sum, tokens = 0.0, 0
    try:
        for example in dataset:
            ids, labels, attn = (x.to(device) for x in collate([example], pad_id))
            count = int((labels[:, 1:] != -100).sum())
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                loss = response_loss(model, ids, labels, attn)
            loss_sum += loss.item() * count
            tokens += count
    finally:
        model.train(was_training)
    if not tokens:
        raise ValueError("validation has no supervised tokens")
    return loss_sum / tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data", default="data/distill/train.jsonl")
    ap.add_argument("--val", default="")
    ap.add_argument("--out", default="runs/izumxai-2b-lora")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--vram-budget-gb", type=float, default=4.0,
                    help="PyTorch allocator budget in GiB, excludes driver/other apps")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--arch", default="auto", choices=["auto", "text", "multimodal"],
                    help="text = AutoModelForCausalLM, multimodal = ImageTextToText (qwen3_5)")
    ap.add_argument("--vision-skip", default="visual,vision_tower,multi_modal_projector",
                    help="части, которые не квантуются в 4-bit (vision-башня)")
    ap.add_argument("--lora-targets", default="",
                    help="список суффиксов модулей через запятую (по умолчанию авто)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    if min(args.batch_size, args.grad_accum, args.max_len, args.eval_every,
           args.log_every, args.save_every, args.lora_r) < 1 or args.epochs <= 0:
        ap.error("batch, accumulation, lengths, intervals, rank and epochs must be positive")
    if pick_dtype() != torch.bfloat16:
        ap.error("this QLoRA trainer requires a CUDA GPU with BF16 support")
    try:
        configure_vram_budget(args.vram_budget_gb)
    except ValueError as exc:
        ap.error(str(exc))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, BitsAndBytesConfig

    dtype = pick_dtype()
    tok = load_tokenizer(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    archs = getattr(cfg, "architectures", None) or []
    multimodal = args.arch == "multimodal" or (
        args.arch == "auto" and any("ConditionalGeneration" in a for a in archs))
    skip = [s.strip() for s in args.vision_skip.split(",") if s.strip()]

    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=skip if multimodal else None,
        )
    loader = auto_model_class(args.model)
    print(f"загружаю {args.model} ({'4-bit' if quant else 'bf16'}, {loader.__name__})...")
    model = loader.from_pretrained(
        args.model,
        quantization_config=quant,
        torch_dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        attn_implementation="sdpa",
    )
    frozen = 0
    for name, p in model.named_parameters():
        if any(s in name for s in skip):
            p.requires_grad_(False)
            frozen += p.numel()
    if frozen:
        print(f"заморожено vision-параметров: {frozen / 1e6:.1f}M")
    model.config.use_cache = False
    if quant is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        cast_frozen_embeddings(model, dtype)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    if args.lora_targets:
        targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    elif multimodal:
        # Гибридное внимание: q/k/v/o есть только в full-attention слоях,
        # в linear-attention — in_proj_*/out_proj. Берём все линейные слои текста.
        targets = sorted({name.rsplit(".", 1)[-1] for name, m in model.named_modules()
                          if isinstance(m, torch.nn.Linear)
                          and not any(s in name for s in skip)
                          and "lm_head" not in name})
    else:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
    print(f"LoRA-цели ({len(targets)}): {', '.join(targets)}")

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    print_vram("после загрузки")

    ds = ChatSFTDataset(args.data, tok, args.max_len)
    print(f"примеров: {len(ds)}, пропущено (пустые/длинные): {ds.skipped}")
    val = ChatSFTDataset(args.val, tok, args.max_len) if args.val else None

    if val:
        # Tokenized prompts include the system message and the assistant prefix.
        def prompts(dataset):
            return {tuple(ex["input_ids"][:next(i for i, label in enumerate(ex["labels"])
                                                if label != -100)]) for ex in dataset}
        if prompts(ds) & prompts(val):
            raise SystemExit("train/val overlap: use separate prompts for validation")

    steps_per_epoch = max(1, math.ceil(len(ds) / (args.batch_size * args.grad_accum)))
    total_steps = args.max_steps if args.max_steps > 0 else max(1, math.ceil(steps_per_epoch * args.epochs))
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

    history = []
    best_val = float("inf")
    def validate(step):
        nonlocal best_val
        value = evaluate(model, val, tok.pad_token_id, dtype)
        history.append({"step": step, "val_loss": value})
        print(f"validation step {step}: {value:.4f}", flush=True)
        if value < best_val:
            best_val = value
            model.save_pretrained(out / "best")
            tok.save_pretrained(out / "best")
        (out / "validation.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        # Validation uses different sequence sizes; release its idle allocations
        # before the next training phase under a strict allocator budget.
        torch.cuda.empty_cache()

    if val:
        validate(0)
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

        try:
            with torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                loss = response_loss(model, ids, labels, attn)
            (loss / args.grad_accum).backward()
        except torch.OutOfMemoryError:
            # A single unusually long sample can still exceed a fragmented
            # allocator; drop it instead of losing the whole run.
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"  OOM на шаге {step} (len {ids.shape[1]}) — пример пропущен", flush=True)
            continue
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
            if val and (step % args.eval_every == 0 or step == total_steps):
                validate(step)

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
        "skipped_examples": ds.skipped,
        "vram_budget_gib": args.vram_budget_gb,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "best_val_loss": best_val if val else None,
        "validation": history,
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"готово за {(time.time() - t0) / 60:.1f} мин -> {out / 'final'}")
    print_vram("после обучения")


if __name__ == "__main__":
    main()
