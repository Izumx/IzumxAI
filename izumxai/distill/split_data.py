"""Стратифицированное деление датасета на train/val по типу задачи.

Нужно, когда отдельная генерация отложенной выборки занимает больше времени,
чем обучение: берём 6% каждого типа, остальное — в обучение.

    python -m izumxai.distill.split_data --data data/distill/stage5_train.jsonl \
        --train data/distill/stage5_train_split.jsonl --val data/distill/stage5_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def split(rows: list[dict], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_kind[r.get("task_type", "?")].append(r)
    rng = random.Random(seed)
    train, val = [], []
    for items in by_kind.values():
        rng.shuffle(items)
        n_val = max(1, round(len(items) * frac))
        val += items[:n_val]
        train += items[n_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    train, val = split(rows, args.frac, args.seed)
    for path, items in ((args.train, train), (args.val, val)):
        Path(path).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8")
    kinds = sorted({r.get("task_type", "?") for r in rows})
    print(f"train {len(train)}, val {len(val)} | "
          f"val по типам: { {k: sum(1 for r in val if r.get('task_type') == k) for k in kinds} }")


if __name__ == "__main__":
    main()
