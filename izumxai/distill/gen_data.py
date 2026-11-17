"""Генерация данных дистилляции от учителя (qwen3:8b через Ollama).

Идея: учитель решает задачи, мы проверяем ответ ПРОГРАММНО (математика —
точное сравнение чисел, код — реальный запуск тестов, инструкции — строгая
проверка формата) и оставляем только верифицированные пары. Так 2B-ученик
получает сигнал качества 8B, но без галлюцинаций учителя.

Запуск:
    python -m izumxai.distill.gen_data --model qwen3:8b --n 1000 --level hard
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import requests

from .tasks import (
    CODE_TASKS, SYSTEM_RU, check_math, extract_code, gen_arithmetic, gen_logic,
    gen_task, last_number, run_tests,
)

__all__ = [
    "CODE_TASKS", "SYSTEM_RU", "check_math", "extract_code", "gen_arithmetic",
    "gen_logic", "gen_task", "last_number", "run_tests", "main",
]

OLLAMA = "http://localhost:11434"


def make_item(kind: str, problem: str, answer: str, level: str,
              expected: str | None = None, system: str = SYSTEM_RU) -> dict:
    item = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": answer},
        ],
        "task_type": kind,
        "level": level,
        "verified": True,
    }
    if expected is not None:
        item["expected"] = expected
    return item


def _key(problem: str) -> str:
    return hashlib.sha1(problem.strip().lower().encode("utf-8")).hexdigest()[:16]


def ask(session: requests.Session, model: str, messages: list[dict],
        temperature: float = 0.6, num_ctx: int = 4096, think: bool = False,
        num_predict: int = 1024, timeout: int = 600) -> str:
    payload = {
        "model": model, "messages": messages, "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    r = session.post(f"{OLLAMA}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/distill/train.jsonl")
    ap.add_argument("--mix", default="math:30,logic:20,code:30,instruction:20",
                    help="доли задач в процентах")
    ap.add_argument("--level", default="easy", choices=["easy", "hard", "mix"],
                    help="сложность задач")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--think", action="store_true", help="включить reasoning учителя")
    ap.add_argument("--num-ctx", type=int, default=2048)
    ap.add_argument("--chain-bias", type=float, default=None,
                    help="доля длинных цепочек среди cyber-задач (по умолчанию 0.25/0.45)")
    ap.add_argument("--num-predict", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="остановиться после N попыток (0 — без ограничения)")
    ap.add_argument("--code-only", action="store_true",
                    help="генерировать только «напиши функцию» (банк secure-code)")
    ap.add_argument("--fallback-reference", action="store_true",
                    help="если учитель не прошёл тесты, взять проверенное эталонное "
                         "решение из банка (для secure-code задач)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    kept = 0
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = obj.get("messages") or []
            if len(msgs) >= 2:
                seen.add(_key(msgs[1]["content"]))
            kept += 1
        if kept:
            print(f"найдено {kept} готовых примеров, продолжаем (повторы пропускаем)")

    mix = {}
    for part in args.mix.split(","):
        k, v = part.split(":")
        mix[k.strip()] = int(v)
    kinds = [k for k, v in mix.items() for _ in range(v)]

    levels = ["easy", "hard"] if args.level == "mix" else [args.level]
    rng = random.Random(args.seed)
    session = requests.Session()
    t0 = time.time()
    start_kept = kept
    attempts = 0
    fail = {"dup": 0, "parse": 0, "wrong": 0, "http": 0, "ref": 0}

    with out.open("a", encoding="utf-8") as f:
        while kept < args.n:
            if args.max_attempts and attempts >= args.max_attempts:
                print("достигнут лимит попыток")
                break
            kind = rng.choice(kinds)
            level = rng.choice(levels)
            attempts += 1
            if args.code_only:
                from .cyber_tasks import gen_cyber_code
                task = gen_cyber_code(rng, level, notes_rng=rng)
                kind = "cyber"
            else:
                task = gen_task(rng, kind, level, chain_bias=args.chain_bias)

            key = _key(task.problem)
            if key in seen:
                fail["dup"] += 1
                continue

            try:
                answer = ask(session, args.model,
                             [{"role": "system", "content": task.system},
                              {"role": "user", "content": task.problem}],
                             think=args.think, num_ctx=args.num_ctx,
                             num_predict=args.num_predict, timeout=args.timeout)
            except requests.RequestException as e:
                fail["http"] += 1
                print(f"HTTP ошибка: {e}", flush=True)
                time.sleep(2)
                continue

            if not task.check(answer):
                if args.fallback_reference and task.reference and task.check(task.reference):
                    answer = task.reference
                    fail["ref"] += 1
                else:
                    fail["wrong"] += 1
                    continue

            seen.add(key)
            f.write(json.dumps(
                make_item(kind, task.problem, answer, level, task.expected, task.system),
                ensure_ascii=False) + "\n")
            f.flush()
            kept += 1
            rate = (kept - start_kept) / max(1e-9, time.time() - t0) * 60
            print(f"[{kept}/{args.n}] {kind}/{level} ok | {rate:.1f} примеров/мин | "
                  f"брак: {fail}", flush=True)

    print(f"готово: {kept} примеров, попыток {attempts}, "
          f"время {(time.time() - t0) / 60:.1f} мин -> {out}")


if __name__ == "__main__":
    main()
