"""Смешивание проверенного stage5-датасета с внешним cyber-SFT (Vyber07).

Stage5 (1105 проверенных примеров: cyber/math/logic/code/instruction) идёт
полностью, внешние данные — с ограничением и стратификацией по домену,
чтобы не размыть математику/код и не перекормить модель одним разделом.

    python -m izumxai.distill.mix_v4 --vyber-cap 2200
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path


def load(path: Path) -> list[dict]:
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
    ap.add_argument("--vyber", default="data/distill/vyber_sft.jsonl")
    ap.add_argument("--vyber-cap", type=int, default=2200)
    ap.add_argument("--vyber-val", type=int, default=60)
    ap.add_argument("--out-train", default="data/distill/stage5_v4_train.jsonl")
    ap.add_argument("--out-val", default="data/distill/stage5_v4_val.jsonl")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stage5_train = load(Path(args.stage5_train))
    stage5_val = load(Path(args.stage5_val))
    vyber = load(Path(args.vyber)) if Path(args.vyber).exists() else []
    print(f"stage5 train: {len(stage5_train)}, val: {len(stage5_val)}, vyber: {len(vyber)}")

    known = {user_key(row) for row in stage5_train + stage5_val}
    before = len(vyber)
    vyber = [row for row in vyber if user_key(row) not in known]
    print(f"vyber: отброшено пересечений со stage5: {before - len(vyber)}")

    # стратифицированная выборка из внешних данных
    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for row in vyber:
        buckets[row.get("task_type", "vyber:unknown")].append(row)
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
    print(f"взято из vyber: {len(sampled)}")

    # val: сначала внешние (их не видел stage5-val), потом stage5-val
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


if __name__ == "__main__":
    main()
