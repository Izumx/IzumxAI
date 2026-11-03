"""Pretrain IzumxAI-nano с нуля.

python -m izumxai.train_pretrain --data data/bin/corpus.bin --tokenizer data/tokenizer/tokenizer.json \
       --preset nano-30m --steps 5000 --batch-size 16 --lr 3e-4 --out runs/nano30m
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import TokenDataset
from .model import Config, IzumxNano, memory_report
from .tokenizer import BPETokenizer


def cosine_lr(step: int, total: int, base_lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="*.bin из izumxai.build_bin")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--preset", default="nano-30m")
    ap.add_argument("--d-model", type=int, default=0)
    ap.add_argument("--n-layers", type=int, default=0)
    ap.add_argument("--n-heads", type=int, default=0)
    ap.add_argument("--n-kv-heads", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--out", default="runs/nano")
    ap.add_argument("--resume", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-bf16", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tok = BPETokenizer.load(args.tokenizer)

    overrides = {
        k: v for k, v in {
            "d_model": args.d_model, "n_layers": args.n_layers,
            "n_heads": args.n_heads, "n_kv_heads": args.n_kv_heads,
            "max_seq": args.max_seq,
        }.items() if v
    }
    cfg = Config.preset(args.preset, vocab_size=tok.vocab_size,
                        pad_id=tok.pad_id, **overrides)
    print(memory_report(cfg))

    ds = TokenDataset(args.data, cfg.max_seq)
    split = max(1, int(len(ds) * 0.99))
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [split, len(ds) - split], generator=torch.Generator().manual_seed(0)
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    model = IzumxNano(cfg).to(device)
    print(f"модель: {sum(p.numel() for p in model.parameters())/1e6:.2f}M параметров")
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"продолжаем с шага {start_step}")

    use_bf16 = (not args.no_bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    def save(step: int, tag: str = "") -> None:
        path = out / f"ckpt_{step:06d}{tag}.pt"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "step": step, "cfg": cfg.to_dict(), "preset": args.preset}, path)
        torch.save({"model": {k: v.half().cpu() for k, v in model.state_dict().items()},
                    "cfg": cfg.to_dict(), "step": step}, out / "latest.pt")

    model.train()
    step = start_step
    t0 = time.time()
    tokens_seen = 0
    it = iter(loader)
    while step < args.steps:
        lr = cosine_lr(step, args.steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            loss.backward()
            loss_acc += loss.item()
            tokens_seen += x.numel()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        step += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:6d} | loss {loss_acc:.4f} | ppl {math.exp(min(20, loss_acc)):8.2f} "
                  f"| lr {lr:.2e} | {tokens_seen / dt:,.0f} tok/s", flush=True)

        if step % args.eval_every == 0 and len(val_ds):
            model.eval()
            with torch.no_grad():
                vx, vy = val_ds[0]
                vx, vy = vx[None].to(device), vy[None].to(device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
                    _, vloss = model(vx, vy)
            print(f"  val loss {vloss.item():.4f} | ppl {math.exp(min(20, vloss.item())):.2f}")
            model.eval()

        if step % args.save_every == 0:
            save(step)

    save(step, tag="_final")
    (out / "config.json").write_text(json.dumps({"cfg": cfg.to_dict(), "preset": args.preset,
                                                 "tokenizer": str(args.tokenizer)}, indent=2),
                                     encoding="utf-8")

    model.eval()
    prompt = "IzumxAI — это"
    ids = torch.tensor([tok.encode(prompt, add_bos=True)], device=device)
    gen = model.generate(ids, max_new_tokens=80, temperature=0.8, top_k=40, eos_id=tok.eos_id)
    print("пример генерации:", tok.decode(gen[0].tolist()))
    print(f"обучение завершено за {(time.time() - t0) / 60:.1f} мин")


if __name__ == "__main__":
    main()
