"""Быстрый self-test без обучения: токенизатор, датасет, модель, генерация.

python tests/smoke.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from izumxai.dataset import TokenDataset, build_bin
from izumxai.model import Config, IzumxNano, memory_report
from izumxai.tokenizer import BPETokenizer


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {name} {extra}")
    if not cond:
        raise SystemExit(1)


def main() -> None:
    docs = ["Привет, это IzumxAI. " * 30, "def add(a, b):\n    return a + b\n" * 30]
    tok = BPETokenizer.train(docs, vocab_size=512)
    s = "IzumxAI: Привет, мир! 123"
    check("tokenizer roundtrip", tok.decode(tok.encode(s, add_bos=True)) == s)
    check("tokenizer vocab", 256 < tok.vocab_size <= 512, f"({tok.vocab_size})")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        bin_path = Path(td) / "c.bin"
        build_bin(docs, tok, bin_path)
        ds = TokenDataset(bin_path, block_size=32)
        x, y = ds[0]
        check("dataset shapes", x.shape == (32,) and y.shape == (32,))
        check("shifted targets", torch.equal(x[1:], y[:-1]))

        cfg = Config.preset("nano-30m", vocab_size=tok.vocab_size, max_seq=64,
                            d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128)
        print(memory_report(cfg))
        model = IzumxNano(cfg)
        batch = torch.randint(0, tok.vocab_size, (2, 32))
        logits, loss = model(batch, batch)
        check("forward logits", tuple(logits.shape) == (2, 32, tok.vocab_size))
        loss.backward()
        grads = sum(1 for p in model.parameters() if p.grad is not None)
        check("backward grads", grads > 0, f"({grads} тензоров)")

        out = model.generate(batch[:, :8], max_new_tokens=6, temperature=0.9, eos_id=tok.eos_id)
        check("generate", out.shape[1] >= 8)

        # KV-cache vs полный проход: логиты на последних шагах должны совпасть
        model.eval()
        seq = torch.randint(0, tok.vocab_size, (2, 17))
        with torch.no_grad():
            full, _ = model(seq)
            x0 = model.embed(seq[:, :16])
            caches = [None] * len(model.blocks)
            for i, blk in enumerate(model.blocks):
                x0, caches[i] = blk(x0, caches[i], 0)
            prefill_logits = model.lm_head(model.norm(x0))[:, -1]
            x1 = model.embed(seq[:, 16:17])
            for i, blk in enumerate(model.blocks):
                x1, caches[i] = blk(x1, caches[i], 16)
            step_logits = model.lm_head(model.norm(x1))[:, -1]
        diff = (full[:, 16] - step_logits).abs().max().item()
        pre_diff = (full[:, 15] - prefill_logits).abs().max().item()
        check("kv-cache = full forward", diff < 1e-4 and pre_diff < 1e-4,
              f"(diff={diff:.2e}, prefill={pre_diff:.2e})")

    print("\nвсе проверки пройдены")


if __name__ == "__main__":
    main()
