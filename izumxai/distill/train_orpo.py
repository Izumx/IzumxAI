"""ORPO: дообучение на парах «хороший/плохой ответ» без модели-референса.

ORPO (Hong et al., 2024) объединяет обычную кросс-энтропию на хорошем ответе
и штраф за отношение правдоподобий пары:

    L = -log p(chosen|x) - λ * log σ( log_odds(chosen) - log_odds(rejected) )

Модель-референс не нужна, поэтому на 6 ГБ VRAM всё влезает: 4-bit база +
LoRA, два прохода на пару. Стартуем от адаптера после SFT, чтобы не терять
уже выученное.

    python -m izumxai.distill.train_orpo --pairs data/distill/prefs.jsonl \
        --init runs/izumxai-2b-lora/final --out runs/izumxai-2b-orpo
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from .common import (build_labeled_example, configure_vram_budget, load_tokenizer,
                     pick_dtype, print_vram, response_loss, cast_frozen_embeddings)


class PairDataset:
    def __init__(self, path: str | Path, tokenizer, max_len: int):
        self.items = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            base = obj["messages"]
            chosen = base + [{"role": "assistant", "content": obj["chosen"]}]
            rejected = base + [{"role": "assistant", "content": obj["rejected"]}]
            ex_c = build_labeled_example(tokenizer, chosen, max_len)
            ex_r = build_labeled_example(tokenizer, rejected, max_len)
            if ex_c is None or ex_r is None:
                continue
            if len(ex_c["input_ids"]) != len(ex_c["labels"]) or \
                    all(l == -100 for l in ex_c["labels"]):
                continue
            self.items.append({
                "chosen": ex_c, "rejected": ex_r,
                "kind": obj.get("task_type", "?"),
            })
        if not self.items:
            raise SystemExit(f"нет валидных пар в {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def response_logp(model, ex) -> torch.Tensor:
    """Средний лог-вероятность ответа (по токенам ответа, не по промпту)."""
    device = next(model.parameters()).device
    ids = torch.tensor([ex["input_ids"]], device=device)
    labels = torch.tensor([ex["labels"]], device=device)
    if not (labels[:, 1:] != -100).any():
        raise ValueError("response has no supervised next-token targets")
    return -response_loss(model, ids, labels, torch.ones_like(ids)).float()


def orpo_loss(lp_c, lp_r, lam):
    """Response-only SFT + odds-ratio preference loss, both branches differentiable."""
    pc = lp_c.float().clamp(max=-1e-7)
    pr = lp_r.float().clamp(max=-1e-7)
    # expm1 avoids cancellation for very likely responses.
    margin = (pc - torch.log(-torch.expm1(pc))) - (pr - torch.log(-torch.expm1(pr)))
    penalty = -torch.nn.functional.logsigmoid(margin)
    return -lp_c + lam * penalty, penalty, margin


def backward_pair(model, pair, lam: float, scale: float = 1.0):
    """Exact chain rule with one response graph in memory at a time.

    Dropout must be disabled so scalar scoring and recomputation agree.
    No optimizer step may occur between the two backward passes.
    """
    with torch.no_grad():
        lp_c = response_logp(model, pair["chosen"])
        lp_r = response_logp(model, pair["rejected"])
    lp_c = lp_c.detach().requires_grad_()
    lp_r = lp_r.detach().requires_grad_()
    loss, penalty, margin = orpo_loss(lp_c, lp_r, lam)
    dc, dr = torch.autograd.grad(loss, (lp_c, lp_r))
    for branch, coefficient in (("chosen", dc), ("rejected", dr)):
        (response_logp(model, pair[branch]) * (coefficient / scale)).backward()
    return {"sft": -lp_c.item(), "or": penalty.item(), "margin": margin.item()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--pairs", default="data/distill/prefs.jsonl")
    ap.add_argument("--init", default="runs/izumxai-2b-lora/final",
                    help="адаптер после SFT (пусто — обучать с нуля)")
    ap.add_argument("--out", default="runs/izumxai-2b-orpo")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--vram-budget-gb", type=float, default=4.0,
                    help="PyTorch allocator budget in GiB, excludes driver/other apps")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lam", type=float, default=0.1, help="вес OR-штрафа")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if min(args.batch_size, args.grad_accum, args.max_len, args.log_every, args.save_every) < 1:
        ap.error("batch-size, grad-accum, max-len, log-every and save-every must be positive")
    if args.epochs <= 0 or args.lam < 0:
        ap.error("epochs must be positive and lam must be nonnegative")
    if args.init and not Path(args.init).is_dir():
        ap.error(f"initial adapter does not exist: {args.init}; pass --init '' to start fresh")
    if not torch.cuda.is_available() or pick_dtype() != torch.bfloat16:
        ap.error("ORPO QLoRA requires a CUDA GPU with BF16 support")
    try:
        configure_vram_budget(args.vram_budget_gb)
    except ValueError as exc:
        ap.error(str(exc))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = pick_dtype()
    tok = load_tokenizer(args.model)

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    print(f"загружаю {args.model} (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, torch_dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cast_frozen_embeddings(model, dtype)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    if args.init and Path(args.init).exists():
        print(f"продолжаю с адаптера {args.init}")
        model = PeftModel.from_pretrained(model, args.init, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ))
    model.print_trainable_parameters()
    # Required by sequential recomputation, including loaded LoRA adapters.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    print_vram("после загрузки")

    ds = PairDataset(args.pairs, tok, args.max_len)
    print(f"пар: {len(ds)}")

    steps_per_epoch = max(1, math.ceil(len(ds) / (args.batch_size * args.grad_accum)))
    total_steps = args.max_steps if args.max_steps > 0 else max(1, int(steps_per_epoch * args.epochs))
    warmup = max(1, int(total_steps * args.warmup_ratio))

    try:
        from bitsandbytes.optim import PagedAdamW8bit
        opt = PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    except Exception:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    model.train()
    t0 = time.time()
    step, micro = 0, 0
    acc = {"sft": 0.0, "or": 0.0, "margin": 0.0}
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

        metrics = []
        for ex in batch:
            with torch.autocast("cuda", dtype=dtype):
                metrics.append(backward_pair(model, ex, args.lam,
                                             scale=args.grad_accum * args.batch_size))
        for key in acc:
            acc[key] += sum(m[key] for m in metrics) / len(metrics)
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
                k = args.log_every * args.grad_accum
                print(f"step {step}/{total_steps} | sft {acc['sft'] / k:.4f} | "
                      f"or {acc['or'] / k:.4f} | margin {acc['margin'] / k:.2f} | "
                      f"lr {lr:.2e} | {(time.time() - t0) / 60:.1f} мин", flush=True)
                acc = {"sft": 0.0, "or": 0.0, "margin": 0.0}
            if step % args.save_every == 0:
                model.save_pretrained(out / f"step_{step}")
                print(f"  адаптер сохранён: {out / f'step_{step}'}")

    model.save_pretrained(out / "final")
    tok.save_pretrained(out / "final")
    meta = {
        "base_model": args.model, "init": args.init, "pairs": len(ds),
        "steps": step, "lr": args.lr, "lam": args.lam, "max_len": args.max_len,
        "vram_budget_gib": args.vram_budget_gb,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "preference_backward": "sequential_exact",
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"готово за {(time.time() - t0) / 60:.1f} мин -> {out / 'final'}")
    print_vram("после обучения")


if __name__ == "__main__":
    main()
