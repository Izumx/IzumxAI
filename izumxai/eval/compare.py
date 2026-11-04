"""Сравнивает два отчёта решателя HumanEval по задачам.

Показывает, сколько задач исправлено и сломано относительно базового отчёта —
так видно, что даёт новый датасет, а не только итоговый счёт.

Запуск:
    python -m izumxai.eval.compare runs/solver/humaneval-think-r3.json \
        runs/solver/humaneval-stage4-r3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict[str, bool]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["id"]: bool(row["ok"]) for row in report["rows"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("candidate")
    args = ap.parse_args()

    base, cand = load(args.base), load(args.candidate)
    common = sorted(set(base) & set(cand))
    fixed = [i for i in common if cand[i] and not base[i]]
    broken = [i for i in common if base[i] and not cand[i]]
    passed = sum(cand[i] for i in common)
    print(f"задач: {len(common)} | было {sum(base[i] for i in common)} | стало {passed}")
    print(f"исправлено ({len(fixed)}): {', '.join(fixed) or '—'}")
    print(f"сломано ({len(broken)}): {', '.join(broken) or '—'}")


if __name__ == "__main__":
    main()
