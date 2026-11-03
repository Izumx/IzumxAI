"""Бенчмарк: IzumxAI:2B против qwen3:8b на одинаковых задачах.

Задачи берутся из общего банка (`izumxai.distill.tasks`), ответы проверяются
программно: математика/логика — точное число, код — реальный запуск тестов,
инструкции — строгий формат. Поэтому счёт не зависит от «впечатления».

    python -m izumxai.eval.bench --models izumxai:2b,qwen3:8b --level hard
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import requests

from ..distill.tasks import gen_task

OLLAMA = "http://localhost:11434"


def chat(model: str, system: str, user: str, temperature: float = 0.2,
         num_ctx: int = 4096, num_predict: int = 800) -> tuple[str, float, int]:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    t0 = time.time()
    r = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=900)
    r.raise_for_status()
    data = r.json()
    dt = time.time() - t0
    return data["message"]["content"].strip(), dt, data.get("eval_count", 0)


def build_tasks(args, rng: random.Random):
    """Наборы задач с уникальными условиями (без повторов между типами)."""
    plan = [
        ("math", args.n_math),
        ("logic", args.n_logic),
        ("code", args.n_code),
        ("instruction", args.n_instr),
    ]
    levels = ["easy", "hard"] if args.level == "mix" else [args.level]
    seen: set[str] = set()
    out = {}
    for kind, n in plan:
        items = []
        guard = 0
        while len(items) < n and guard < n * 40:
            guard += 1
            level = levels[len(items) % len(levels)]
            t = gen_task(rng, kind, level)
            if t.problem in seen:
                continue
            seen.add(t.problem)
            items.append((level, t))
        out[kind] = items
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="izumxai:2b,qwen3:8b")
    ap.add_argument("--n-math", type=int, default=24)
    ap.add_argument("--n-logic", type=int, default=12)
    ap.add_argument("--n-code", type=int, default=16)
    ap.add_argument("--n-instr", type=int, default=12)
    ap.add_argument("--level", default="hard", choices=["easy", "hard", "mix"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/bench.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bank = build_tasks(args, rng)
    total = sum(len(v) for v in bank.values())
    print(f"задач: {total} (уровень: {args.level})")

    results: dict[str, dict] = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {model} ===")
        res: dict = {"by_kind": {}, "rows": [], "errors": 0}
        for kind, items in bank.items():
            ok_n = 0
            for i, (level, t) in enumerate(items, 1):
                try:
                    ans, dt, ntok = chat(model, t.system, t.problem)
                except Exception as e:
                    res["errors"] += 1
                    print(f"  {kind} {i}: ошибка {e}")
                    continue
                ok = bool(t.check(ans))
                ok_n += int(ok)
                res["rows"].append({
                    "kind": kind, "level": level, "problem": t.problem,
                    "expected": t.expected, "answer": ans, "ok": ok,
                    "sec": round(dt, 2), "tokens": ntok,
                })
                print(f"  {kind} {i}/{len(items)} [{level}]: {'OK' if ok else 'fail'}"
                      f" (ждём {t.expected})", flush=True)
            acc = ok_n / max(1, len(items))
            res["by_kind"][kind] = {"acc": acc, "n": len(items), "ok": ok_n}
            print(f"  -> {kind}: {acc:.0%}")

        good = [r for r in res["rows"] if r["tokens"] and r["sec"]]
        res["acc"] = sum(r["ok"] for r in good) / max(1, len(good))
        res["tok_per_s"] = (sum(r["tokens"] for r in good)
                            / max(1e-9, sum(r["sec"] for r in good)))
        results[model] = res
        print(f"  итог: {res['acc']:.0%}, {res['tok_per_s']:.1f} tok/s")

    print("\n{:<18}".format("тип") + "".join(f"{m[:16]:>18}" for m in results))
    for kind in bank:
        line = "{:<18}".format(kind)
        for r in results.values():
            k = r["by_kind"].get(kind, {"ok": 0, "n": 1})
            line += f"{k['ok']}/{k['n']:<16}".rjust(18)
        print(line)
    line = "{:<18}".format("итого")
    for r in results.values():
        line += f"{r['acc']:.0%}".rjust(18)
    print(line)
    line = "{:<18}".format("tok/s")
    for r in results.values():
        line += f"{r['tok_per_s']:.1f}".rjust(18)
    print(line)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"результаты: {args.out}")


if __name__ == "__main__":
    main()
