"""Бенчмарк: IzumxAI:2B против qwen3:8b на одинаковых задачах.

Проверяет то, что заявлено: математика, код (реальный запуск тестов),
русские инструкции + скорость генерации.

python -m izumxai.eval.bench --models izumxai:2b,qwen3:8b --n-math 40 --n-code 20
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

from ..distill.gen_data import (
    CODE_TASKS, INSTRUCTION_TASKS, SEED_TEXTS, SYSTEM_RU, check_math, extract_code,
    gen_arithmetic, run_tests,
)

OLLAMA = "http://localhost:11434"


def chat(model: str, system: str, user: str, temperature: float = 0.2,
         num_ctx: int = 4096) -> tuple[str, float, int]:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": 800},
    }
    t0 = time.time()
    r = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=900)
    r.raise_for_status()
    data = r.json()
    dt = time.time() - t0
    out_tokens = data.get("eval_count", 0)
    return data["message"]["content"].strip(), dt, out_tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="izumxai:2b,qwen3:8b")
    ap.add_argument("--n-math", type=int, default=30)
    ap.add_argument("--n-code", type=int, default=12)
    ap.add_argument("--n-instr", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/bench.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    math_tasks = [gen_arithmetic(rng) for _ in range(args.n_math)]
    code_tasks = []
    for i in range(args.n_code):
        func, ref, desc, cases_fn = CODE_TASKS[i % len(CODE_TASKS)]
        code_tasks.append((func, desc, cases_fn(rng)))
    instr_tasks = []
    for i in range(args.n_instr):
        tpl, validator = INSTRUCTION_TASKS[i % len(INSTRUCTION_TASKS)]
        instr_tasks.append((tpl.format(text=SEED_TEXTS[i % len(SEED_TEXTS)]), validator))

    results: dict[str, dict] = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {model} ===")
        res = {"math": 0, "code": 0, "instr": 0, "speed": [], "errors": 0}

        for i, (problem, expected) in enumerate(math_tasks, 1):
            try:
                ans, dt, ntok = chat(model, SYSTEM_RU, problem)
            except Exception as e:
                res["errors"] += 1
                print(f"  math {i}: ошибка {e}")
                continue
            ok = check_math(ans, expected)
            res["math"] += int(ok)
            res["speed"].append(ntok / dt if dt else 0)
            print(f"  math {i}/{len(math_tasks)}: {'OK' if ok else 'fail'} "
                  f"(ждём {expected})", flush=True)

        for i, (func, desc, cases) in enumerate(code_tasks, 1):
            try:
                ans, dt, ntok = chat(model,
                                     "Ты пишешь только рабочий код Python в блоке ```python.",
                                     desc)
            except Exception as e:
                res["errors"] += 1
                print(f"  code {i}: ошибка {e}")
                continue
            src = extract_code(ans)
            ok = bool(src) and run_tests(src, func, cases)
            res["code"] += int(ok)
            res["speed"].append(ntok / dt if dt else 0)
            print(f"  code {i}/{len(code_tasks)}: {'OK' if ok else 'fail'}", flush=True)

        for i, (problem, validator) in enumerate(instr_tasks, 1):
            try:
                ans, dt, ntok = chat(model, SYSTEM_RU, problem)
            except Exception as e:
                res["errors"] += 1
                print(f"  instr {i}: ошибка {e}")
                continue
            ok = validator(ans)
            res["instr"] += int(ok)
            res["speed"].append(ntok / dt if dt else 0)
            print(f"  instr {i}/{len(instr_tasks)}: {'OK' if ok else 'fail'}", flush=True)

        res["math_acc"] = res["math"] / max(1, len(math_tasks))
        res["code_acc"] = res["code"] / max(1, len(code_tasks))
        res["instr_acc"] = res["instr"] / max(1, len(instr_tasks))
        res["tok_per_s"] = sum(res["speed"]) / max(1, len(res["speed"]))
        results[model] = res
        print(f"  итог: math {res['math_acc']:.0%}, code {res['code_acc']:.0%}, "
              f"instr {res['instr_acc']:.0%}, {res['tok_per_s']:.1f} tok/s")

    print("\n{:<20} {:>8} {:>8} {:>8} {:>10}".format("модель", "math", "code", "instr", "tok/s"))
    for model, r in results.items():
        print("{:<20} {:>7.0%} {:>8.0%} {:>8.0%} {:>10.1f}".format(
            model, r["math_acc"], r["code_acc"], r["instr_acc"], r["tok_per_s"]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"результаты: {args.out}")


if __name__ == "__main__":
    main()
