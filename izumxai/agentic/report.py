#!/usr/bin/env python3
"""Aggregate harness runs into the two numbers that matter.

    solved              - the oracle confirmed the exploit
    solved w/o escalation - the model did it before the harness policy stepped in

The second one is the honest agent metric: with ``--escalate-after N`` the
harness can solve a task for the model, so a raw solved count hides whether the
model chose anything useful.

    python -m izumxai.agentic.report
    python -m izumxai.agentic.report runs/agentic/20261119-120000
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent.parent / "runs" / "agentic"


def latest(root: Path) -> Path:
    dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    if not dirs:
        raise SystemExit(f"no runs under {root}")
    return dirs[-1]


def load(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"skip {path}: {exc}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None,
                    help="run directory; defaults to the latest")
    args = ap.parse_args()

    root = Path(args.run) if args.run else latest(RUNS)
    rows = load(root)
    if not rows:
        raise SystemExit(f"no records in {root}")

    per: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        per[(r["model"], r["task"])].append(r)

    models = sorted({r["model"] for r in rows})
    tasks = sorted({r["task"] for r in rows})

    def cell(recs: list[dict]) -> str:
        if not recs:
            return "-"
        ok = sum(r["solved"] for r in recs)
        pure = sum(r["solved"] and not r["escalated"] for r in recs)
        return f"{ok}/{len(recs)} ({pure})"

    print(f"run: {root}")
    width = max(20, *(len(m) for m in models))
    header = "model".ljust(width) + "".join(t[:16].rjust(18) for t in tasks)
    print(header)
    print("-" * len(header))
    for m in models:
        line = m.ljust(width)
        for t in tasks:
            line += cell(per[(m, t)]).rjust(18)
        print(line)

    print("\ntotal solved (solved without escalation)")
    for m in models:
        recs = [r for r in rows if r["model"] == m]
        ok = sum(r["solved"] for r in recs)
        pure = sum(r["solved"] and not r["escalated"] for r in recs)
        print(f"  {m:42} {ok}/{len(recs)} ({pure}/{len(recs)})")

    # Repeated passes over the same model/task: min..max shows stability.
    print("\nspread over passes")
    for (m, t), recs in sorted(per.items()):
        if len(recs) < 2:
            continue
        ok = sum(r["solved"] for r in recs)
        if len({r["solved"] for r in recs}) > 1:
            print(f"  {m:42} {t:20} unstable {ok}/{len(recs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
