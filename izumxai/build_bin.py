"""CLI: сырой текст -> бинарный поток токенов.

python -m izumxai.build_bin --raw data/raw --tokenizer data/tokenizer/tokenizer.json --out data/bin/corpus.bin
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .dataset import build_bin, read_corpus
from .tokenizer import BPETokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default="data/bin/corpus.bin")
    args = ap.parse_args()

    t0 = time.time()
    docs = read_corpus(args.raw)
    if not docs:
        raise SystemExit(f"в {args.raw} нет текстов")
    tok = BPETokenizer.load(args.tokenizer)
    print(f"документов: {len(docs)}")
    total = build_bin(docs, tok, args.out)
    print(f"токенов: {total:,} -> {Path(args.out)} за {time.time() - t0:.1f} c")


if __name__ == "__main__":
    main()
