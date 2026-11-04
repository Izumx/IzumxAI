"""Freeze verified SFT data and split unique user prompts before training."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def split_records(records, val_fraction: float = 0.1, seed: int = 42):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    unique = {}
    for record in records:
        messages = record.get("messages", [])
        if not record.get("verified") or not messages or messages[-1].get("role") != "assistant":
            continue
        prompt = tuple(" ".join(m.get("content", "").split()).casefold()
                       for m in messages if m.get("role") == "user")
        if not prompt or not all(prompt) or not messages[-1].get("content", "").strip():
            continue
        unique.setdefault(prompt, record)
    if len(unique) < 2:
        raise ValueError("at least two unique verified prompts are required")
    keys = sorted(unique)
    random.Random(seed).shuffle(keys)
    n_val = min(len(keys) - 1, max(1, round(len(keys) * val_fraction)))
    return ([unique[k] for k in keys[n_val:]], [unique[k] for k in keys[:n_val]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/distill/train.jsonl")
    ap.add_argument("--out", default="data/distill/4gb")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    source = Path(args.data)
    # An active generator can be appending; consume only newline-terminated rows.
    snapshot = source.read_bytes()
    lines = snapshot.splitlines(keepends=True)
    records = [json.loads(line) for line in lines if line.endswith(b"\n") and line.strip()]
    train, val = split_records(records, args.val_fraction, args.seed)
    out = Path(args.out)
    if source.resolve() in {(out / "train.jsonl").resolve(), (out / "val.jsonl").resolve()}:
        ap.error("output must be separate from the source dataset")
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        (out / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    meta = {"source": str(source.resolve()), "source_sha256": hashlib.sha256(snapshot).hexdigest(),
            "source_complete_rows": len(records), "train": len(train), "val": len(val),
            "seed": args.seed, "val_fraction": args.val_fraction}
    (out / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
