"""Обучение своего BPE-токенизатора IzumxAI.

python -m izumxai.train_tokenizer --raw data/raw --out data/tokenizer --vocab 8192
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .dataset import read_corpus
from .tokenizer import BPETokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="папка с .txt/.md/.jsonl")
    ap.add_argument("--out", default="data/tokenizer")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--limit-chars", type=int, default=0, help="0 = без ограничения")
    args = ap.parse_args()

    t0 = time.time()
    docs = read_corpus(args.raw)
    if not docs:
        raise SystemExit(f"в {args.raw} нет текстов (.txt/.md/.jsonl)")
    if args.limit_chars:
        docs = [d[: args.limit_chars] for d in docs]

    total_chars = sum(len(d) for d in docs)
    print(f"документов: {len(docs)}, символов: {total_chars / 1e6:.2f}M")

    tok = BPETokenizer.train(docs, vocab_size=args.vocab)
    out = Path(args.out)
    tok.save(out / "tokenizer.json")

    sample = docs[0][:400]
    ids = tok.encode(sample, add_bos=True)
    print(f"vocab_size={tok.vocab_size} (запрошено {args.vocab})")
    print(f"сжатие: {len(sample)} символов -> {len(ids)} токенов "
          f"({len(sample) / max(1, len(ids)):.2f} симв/токен)")
    print("пример декодирования:", tok.decode(ids)[:120].replace("\n", " "))
    print(f"готово за {time.time() - t0:.1f} c -> {out / 'tokenizer.json'}")


if __name__ == "__main__":
    main()
