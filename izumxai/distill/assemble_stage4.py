"""Собирает итоговый датасет stage4: рассуждения + проверенные обычные ответы.

Примеры с цепочкой рассуждений повторяются несколько раз: их меньше, но именно
они учат модель думать и проверять себя. Обычные примеры сохраняют навык
короткого ответа и не дают забыть форматы. Валидация берётся из отдельного
файла stage3 и из отложенной части рассуждений, чтобы не было пересечения
промптов между train и val.

Запуск:
    python -m izumxai.distill.assemble_stage4 \
        --think data/distill/stage4-plan/train.jsonl \
        --plain data/distill/stage3-smel/train.jsonl \
        --plain-val data/distill/stage3-smel/val.jsonl \
        --out data/distill/stage4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def key_of(row: dict) -> str:
    for m in row.get("messages", []):
        if m.get("role") == "user":
            return hashlib.sha1(" ".join(m["content"].split()).casefold()
                                .encode("utf-8")).hexdigest()
    return ""


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("verified") and row.get("messages"):
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--think", default="data/distill/stage4-plan/train.jsonl")
    ap.add_argument("--plain", default="data/distill/stage3-smel/train.jsonl")
    ap.add_argument("--plain-val", default="data/distill/stage3-smel/val.jsonl")
    ap.add_argument("--out", default="data/distill/stage4")
    ap.add_argument("--repeat-think", type=int, default=2)
    ap.add_argument("--val-think", type=int, default=32)
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    think = load(Path(args.think))
    plain = load(Path(args.plain))
    plain_val = load(Path(args.plain_val))

    # Откладываем часть рассуждений в валидацию и убираем их из обучения.
    rng.shuffle(think)
    held = think[:args.val_think]
    think = think[args.val_think:]

    val_keys = {key_of(r) for r in plain_val} | {key_of(r) for r in held}
    train = [r for r in plain if key_of(r) not in val_keys]
    train += [r for r in think if key_of(r) not in val_keys] * max(1, args.repeat_think)
    rng.shuffle(train)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train), encoding="utf-8")
    (out / "val.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in plain_val + held),
        encoding="utf-8")

    think_train = sum(1 for r in train if "<think>" in json.dumps(r, ensure_ascii=False))
    meta = {
        "think_source": args.think,
        "plain_source": args.plain,
        "repeat_think": args.repeat_think,
        "train": len(train),
        "val": len(plain_val) + len(held),
        "think_rows_in_train": think_train,
        "held_think_for_val": len(held),
    }
    (out / "data_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
