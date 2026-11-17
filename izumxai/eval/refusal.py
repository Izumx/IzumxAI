"""Доля отказов на технических «вредных» промптах через Ollama.

Промпты те же, что в abliteration, поэтому числа сопоставимы до/после.

    python -m izumxai.eval.refusal --models qwen35:9b-base,izumxai:9b
"""

from __future__ import annotations

import argparse
import json
import re
import time

import requests

from ..distill.abliterate import HARMFUL, REFUSAL_MARKERS

OLLAMA = "http://localhost:11434"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def ask(model: str, prompt: str, num_predict: int = 160) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": num_predict},
    }
    r = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 — все промпты")
    ap.add_argument("--out", default="runs/refusal.json")
    args = ap.parse_args()

    prompts = HARMFUL[: args.limit] if args.limit else HARMFUL
    result: dict[str, dict] = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        hits, samples = 0, []
        t0 = time.time()
        for p in prompts:
            try:
                ans = ask(model, p)
            except Exception as exc:  # noqa: BLE001
                print(f"  ошибка {model}: {exc}")
                continue
            refused = any(m in norm(ans) for m in REFUSAL_MARKERS)
            hits += refused
            if refused and len(samples) < 2:
                samples.append(f"[{p[:40]}] -> {ans[:110]}")
        rate = hits / max(1, len(prompts))
        result[model] = {"refusal_rate": rate, "n": len(prompts),
                         "sec": round(time.time() - t0, 1), "samples": samples}
        print(f"{model}: отказов {rate:.0%} из {len(prompts)}")
        for s in samples:
            print("  ", s)

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
