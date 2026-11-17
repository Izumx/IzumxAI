"""Смесь stage5 (свои проверенные задачи) + полный внешний cyber-SFT (v5).

Отличие от `mix_v4.py`: внешний источник — `vyber_v5.jsonl` (отбор из полного
датасета), квоты задаются флагами, а доля собственных примеров сохраняется.

    python -m izumxai.distill.mix_v5 --vyber-cap 12000
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def user_key(row: dict) -> str:
    text = "\n".join(
        m["content"] for m in row.get("messages", []) if m.get("role") == "user"
    )
    return hashlib.sha1(text.strip().lower().encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage5-train", default="data/distill/stage5_train_split.jsonl")
    ap.add_argument("--stage5-val", default="data/distill/stage5_val.jsonl")
    ap.add_argument("--vyber", default="data/distill/vyber_v5.jsonl")
    ap.add_argument("--vyber-cap", type=int, default=12000)
    ap.add_argument("--vyber-val", type=int, default=120)
    ap.add_argument("--out-train", default="data/distill/stage5_v5_train.jsonl")
    ap.add_argument("--out-val", default="data/distill/stage5_v5_val.jsonl")
    ap.add_argument("--stage5-repeat", type=int, default=1,
                    help="повторить проверенную часть (инструкции/математика/логика) N раз")
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stage5_train = load(Path(args.stage5_train))
    stage5_val = load(Path(args.stage5_val))
    vyber = load(Path(args.vyber))
    print(f"stage5 train: {len(stage5_train)}, val: {len(stage5_val)}, vyber: {len(vyber)}")
    if args.stage5_repeat > 1:
        # Внешние данные «размывают» дисциплину формата: повторяем проверенную часть.
        stage5_train = stage5_train * args.stage5_repeat
        print(f"stage5 train x{args.stage5_repeat} -> {len(stage5_train)}")

    known = {user_key(row) for row in stage5_train + stage5_val}
    before = len(vyber)
    vyber = [row for row in vyber if user_key(row) not in known]
    print(f"vyber: убрано пересечений со stage5: {before - len(vyber)}")

    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for row in vyber:
        buckets[row.get("task_type", "vyber5:unknown")].append(row)
    for rows in buckets.values():
        rng.shuffle(rows)

    total = sum(len(rows) for rows in buckets.values())
    if args.vyber_cap < total:
        factor = args.vyber_cap / total
        sampled: list[dict] = []
        for rows in buckets.values():
            take = max(1, int(len(rows) * factor))
            sampled.extend(rows[:take])
        rng.shuffle(sampled)
        sampled = sampled[: args.vyber_cap]
    else:
        sampled = [row for rows in buckets.values() for row in rows]
        rng.shuffle(sampled)
    print(f"взято из vyber: {len(sampled)} (доменов {len(buckets)})")

    vyber_val = sampled[: args.vyber_val]
    vyber_train = sampled[args.vyber_val :]
    train = stage5_train + vyber_train
    val = stage5_val + vyber_val
    rng.shuffle(train)

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((args.out_train, train), (args.out_val, val)):
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"train: {len(train)} -> {args.out_train}")
    print(f"val:   {len(val)} -> {args.out_val}")
    kinds = collections.Counter(row.get("task_type", "?").split(":")[0] for row in train)
    print("состав train:", dict(kinds))
    top = collections.Counter(
        row.get("task_type", "?").split(":", 1)[-1] for row in train
    ).most_common(12)
    print("топ доменов:", ", ".join(f"{k}={v}" for k, v in top))


if __name__ == "__main__":
    main()
