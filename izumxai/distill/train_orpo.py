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

from .common import build_labeled_example, load_tokenizer, pick_dtype, print_vram


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
    ids = torch.tensor([ex["input_ids"]], device="cuda")
    labels = torch.tensor([ex["labels"]], device="cuda")
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    logits = out.logits[:, :-1, :]
    targets = ids[:, 1:]
    mask = (labels[:, 1:] != -100).float()
    tok = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) \
        - logits.logsumexp(-1)
    return (tok * mask).sum() / mask.sum().clamp(min=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--pairs", default="data/distill/prefs.jsonl")
    ap.add_argument("--init", default="runs/izumxai-2b-lora/final",
                    help="адаптер после SFT (пусто — обучать с нуля)")
    ap.add_argument("--out", default="runs/izumxai-2b-orpo")
    ap.add_argument("--max-len", type=int, default=1024)
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
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    if args.init and Path(args.init).exists():
        print(f"продолжаю с адаптера {args.init}")
        model = PeftModel.from_pretrained(model, args.init, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ))
    model.print_trainable_parameters()
    print_vram("после загрузки")

    ds = PairDataset(args.pairs, tok, args.max_len)
    print(f"пар: {len(ds)}")

    steps_per_epoch = max(1, len(ds) // (args.batch_size * args.grad_accum))
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

        losses = []
        for ex in batch:
            lp_c = response_logp(model, ex["chosen"])
            lp_r = response_logp(model, ex["rejected"]).detach()

            sft = -lp_c
            # odds(p) = log(p / (1 - p)); p — средняя вероятность ответа
            pc = lp_c.clamp(max=-1e-4)
            pr = lp_r.clamp(max=-1e-4)
            log_odds = (pc - torch.log1p(-torch.exp(pc))) - \
                       (pr - torch.log1p(-torch.exp(pr)))
            or_loss = -torch.nn.functional.logsigmoid(log_odds)
            loss = sft + args.lam * or_loss
            losses.append((loss, sft, or_loss, log_odds))

        n_micro = len(losses)
        for loss, sft, or_loss, margin in losses:
            (loss / args.grad_accum).backward()
        acc["sft"] += sum(float(x[1]) for x in losses) / n_micro
        acc["or"] += sum(float(x[2]) for x in losses) / n_micro
        acc["margin"] += sum(float(x[3]) for x in losses) / n_micro
        micro += n_micro

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
                k = args.log_every
                print(f"step {step}/{total_steps} | sft {-acc['sft'] / k:.4f} | "
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
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"готово за {(time.time() - t0) / 60:.1f} мин -> {out / 'final'}")
    print_vram("после обучения")


if __name__ == "__main__":
    main()
