"""Чистка SFT-данных: убираем LaTeX и дубли условий.

Учитель иногда вставляет `$$ ... $$` и `\\times` вопреки системному промпту.
Ученик копирует стиль, поэтому данные приводим к одному виду: числа и знаки
обычным текстом, без формул. Заодно удаляем повторы условий.

    python -m izumxai.distill.clean_data --in data/distill/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MATH_SYMBOLS = {
    r"\times": "×", r"\cdot": "·", r"\div": "÷", r"\implies": "⇒",
    r"\Rightarrow": "⇒", r"\approx": "≈", r"\le": "≤", r"\ge": "≥",
    r"\ne": "≠", r"\neq": "≠", r"\%": "%",
}

DOLLAR_LINE = re.compile(r"^\s*\${1,2}\s*$")


def clean_text(text: str) -> str:
    out = []
    for line in text.splitlines():
        if DOLLAR_LINE.match(line):
            continue
        line = line.replace("$$", "").replace("$", "")
        for k, v in MATH_SYMBOLS.items():
            line = line.replace(k, v)
        out.append(line.rstrip())
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="data/distill/train.jsonl")
    ap.add_argument("--out", default="")
    ap.add_argument("--backup", action="store_true", default=True)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.out) if args.out else src
    if args.backup and dst == src:
        bak = src.with_suffix(".bak.jsonl")
        bak.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"резервная копия: {bak}")

    seen: set[str] = set()
    items, dropped = [], 0
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        msgs = obj.get("messages") or []
        if len(msgs) < 3:
            dropped += 1
            continue
        for m in msgs:
            if m["role"] == "assistant":
                m["content"] = clean_text(m["content"])
        key = msgs[1]["content"].strip().lower()
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        items.append(obj)

    dst.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in items),
                   encoding="utf-8")
    clean = sum(1 for o in items if "$" not in o["messages"][-1]["content"])
    print(f"примеров: {len(items)} (было {len(items) + dropped}, отброшено {dropped}), "
          f"без LaTeX: {clean} -> {dst}")


if __name__ == "__main__":
    main()
