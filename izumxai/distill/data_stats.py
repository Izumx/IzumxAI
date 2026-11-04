"""Статистика датасета: состав, длины, доля рассуждений.

Нужна, чтобы видеть, что реально попало в обучение, до запуска GPU.

Запуск:
    python -m izumxai.distill.data_stats data/distill/stage4/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--tokenizer", default="")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        from .common import load_tokenizer
        tok = load_tokenizer(args.tokenizer)

    for path in args.paths:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()]
        kinds = Counter(r.get("task_type") for r in rows)
        sources = Counter(r.get("source") for r in rows)
        think = [r for r in rows if "<think>" in json.dumps(r, ensure_ascii=False)]
        lengths = [len(m["content"]) for r in rows for m in r["messages"]
                   if m.get("role") == "assistant"]
        print(f"== {path}: строк {len(rows)}")
        print("   типы:", dict(kinds))
        print("   источники:", dict(sources))
        print(f"   с рассуждениями: {len(think)} ({len(think) / max(1, len(rows)):.0%})")
        if lengths:
            print(f"   длина ответа, символов: медиана {statistics.median(lengths):.0f}, "
                  f"максимум {max(lengths)}")
        if tok:
            tok_lens = []
            for r in rows:
                text = tok.apply_chat_template(r["messages"], tokenize=False,
                                               enable_thinking="<think>" in json.dumps(r))
                tok_lens.append(len(tok(text, add_special_tokens=False)["input_ids"]))
            tok_lens.sort()
            p95 = tok_lens[int(len(tok_lens) * 0.95) - 1]
            print(f"   токенов: медиана {tok_lens[len(tok_lens) // 2]}, p95 {p95}, "
                  f"максимум {tok_lens[-1]}")


if __name__ == "__main__":
    main()
