"""Сбор пар «хороший/плохой ответ» с ученика для ORPO.

Берём задачи из общего банка, просим текущую модель решить каждую K раз
при высокой температуре и проверяем ответы программно. Если среди вариантов
есть и верные, и неверные — получаем готовую пару предпочтений:
    chosen   — ответ, который прошёл проверку;
    rejected — правдоподобный, но неверный ответ той же модели.
Верные ответы, для которых пары нет, дополнительно копим для дообучения (RFT).

    python -m izumxai.distill.gen_prefs --model izumxai:2b --n 200 --k 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import requests

from .tasks import gen_task

OLLAMA = "http://localhost:11434"


def sample(session: requests.Session, model: str, system: str, prompt: str,
           temperature: float, num_ctx: int, num_predict: int, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "stream": False, "think": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx,
                    "num_predict": num_predict, "top_p": 0.95},
    }
    r = session.post(f"{OLLAMA}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _key(problem: str) -> str:
    return hashlib.sha1(problem.strip().lower().encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="izumxai:2b")
    ap.add_argument("--n", type=int, default=200, help="сколько задач обработать")
    ap.add_argument("--k", type=int, default=4, help="сколько ответов на задачу")
    ap.add_argument("--level", default="hard", choices=["easy", "hard", "mix"])
    ap.add_argument("--mix", default="math:30,logic:20,code:30,instruction:20")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--out", default="data/distill/prefs.jsonl")
    ap.add_argument("--rft", default="data/distill/rft.jsonl",
                    help="верные ответы без пары — для обычного дообучения")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-ctx", type=int, default=2048)
    ap.add_argument("--num-predict", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    mix = {}
    for part in args.mix.split(","):
        k, v = part.split(":")
        mix[k.strip()] = int(v)
    kinds = [k for k, v in mix.items() for _ in range(v)]
    levels = ["easy", "hard"] if args.level == "mix" else [args.level]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rft = Path(args.rft)
    rft.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    for path in (out, rft):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    msgs = json.loads(line)["messages"]
                except (json.JSONDecodeError, KeyError):
                    continue
                user = next((m["content"] for m in msgs if m["role"] == "user"), None)
                if user:
                    seen.add(_key(user))
    if seen:
        print(f"уже обработано задач: {len(seen)}")

    rng = random.Random(args.seed)
    session = requests.Session()
    t0 = time.time()
    done = 0
    stats = {"pairs": 0, "rft": 0, "all_wrong": 0, "all_right": 0, "dup": 0}

    with out.open("a", encoding="utf-8") as f_pair, rft.open("a", encoding="utf-8") as f_rft:
        while done < args.n:
            kind = rng.choice(kinds)
            level = rng.choice(levels)
            task = gen_task(rng, kind, level)
            key = _key(task.problem)
            if key in seen:
                stats["dup"] += 1
                continue
            seen.add(key)

            good, bad = [], []
            for _ in range(args.k):
                try:
                    ans = sample(session, args.model, task.system, task.problem,
                                 args.temperature, args.num_ctx, args.num_predict, args.timeout)
                except requests.RequestException as e:
                    print(f"HTTP ошибка: {e}", flush=True)
                    time.sleep(2)
                    break
                (good if task.check(ans) else bad).append(ans)

            if good and bad:
                item = {
                    "messages": [
                        {"role": "system", "content": task.system},
                        {"role": "user", "content": task.problem},
                    ],
                    "chosen": good[0],
                    "rejected": bad[0],
                    "task_type": kind,
                    "level": level,
                }
                f_pair.write(json.dumps(item, ensure_ascii=False) + "\n")
                f_pair.flush()
                stats["pairs"] += 1
            elif good:
                item = {
                    "messages": [
                        {"role": "system", "content": task.system},
                        {"role": "user", "content": task.problem},
                        {"role": "assistant", "content": good[0]},
                    ],
                    "task_type": kind,
                    "level": level,
                    "verified": True,
                }
                f_rft.write(json.dumps(item, ensure_ascii=False) + "\n")
                f_rft.flush()
                stats["rft"] += 1
                stats["all_right"] += 1
            else:
                stats["all_wrong"] += 1

            done += 1
            rate = done / max(1e-9, time.time() - t0) * 60
            print(f"[{done}/{args.n}] {kind}/{level} | {rate:.1f} задач/мин | {stats}", flush=True)

    print(f"готово за {(time.time() - t0) / 60:.1f} мин: пар {stats['pairs']}, "
          f"верных для RFT {stats['rft']}, полностью провалено {stats['all_wrong']}")
    print(f"пары: {out}\nRFT:  {rft}")


if __name__ == "__main__":
    main()
