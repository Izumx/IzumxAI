"""Скачивание небольшого корпуса для pretrain nano-модели.

Что берём:
  * tiny-shakespeare (~1 MB, английский, быстрый тест пайплайна);
  * русская википедия (небольшая выборка, через HF datasets), если доступна.

python -m izumxai.fetch_corpus --out data/raw --ru-docs 2000
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

TINY = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def get_tiny_shakespeare(out: Path) -> None:
    path = out / "tiny_shakespeare.txt"
    if path.exists() and path.stat().st_size > 0:
        print(f"уже есть: {path}")
        return
    print("скачиваю tiny-shakespeare...")
    urllib.request.urlretrieve(TINY, path)
    print(f"  {path.stat().st_size / 1e6:.2f} MB")


def get_ru_wiki(out: Path, docs: int) -> None:
    path = out / "ru_wiki.jsonl"
    if path.exists() and path.stat().st_size > 0:
        print(f"уже есть: {path}")
        return
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets не установлен — пропускаю русскую википедию")
        return
    print(f"качаю русскую википедию ({docs} статей)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train", streaming=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in ds:
            text = (row.get("text") or "").strip()
            if len(text) < 400:
                continue
            f.write(json.dumps({"text": text[:8000]}, ensure_ascii=False) + "\n")
            n += 1
            if n >= docs:
                break
            if n % 200 == 0:
                print(f"  {n}/{docs}", flush=True)
    print(f"  {n} статей -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--ru-docs", type=int, default=2000)
    ap.add_argument("--no-ru", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    get_tiny_shakespeare(out)
    if not args.no_ru:
        get_ru_wiki(out, args.ru_docs)


if __name__ == "__main__":
    main()
