"""Frozen local comparison on prompts excluded from the recorded SFT data.

This is a small project benchmark, not a public leaderboard benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

from ..distill.tasks import SYSTEM_CODE, SYSTEM_RU, check_math, extract_code, run_tests
from ..distill.common import load_tokenizer


def normalize(text):
    return " ".join(text.split()).casefold()


def load_tasks(path, training_paths):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts = [normalize(row["prompt"]) for row in rows]
    if not rows or len(prompts) != len(set(prompts)):
        raise ValueError("empty or duplicate holdout prompts")
    training = set()
    for source in training_paths:
        for line in Path(source).read_text(encoding="utf-8").splitlines():
            if line.strip():
                training.update(normalize(m["content"]) for m in json.loads(line).get("messages", [])
                                if m.get("role") == "user")
    if training.intersection(prompts):
        raise ValueError("training/holdout prompt overlap")
    return rows


def score(task, answer):
    if task["kind"] == "code":
        code = extract_code(answer)
        if code and "test_code" in task:
            # HumanEval's prompt supplies imports and a stub. The returned full
            # function replaces that stub, then the original check() is called.
            script = (task["source_prompt"] if "source_prompt" in task else task["prompt"]) + "\n\n" + code
            script += "\n\n" + task["test_code"] + f"\ncheck({task['function']})\n"
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "check.py"
                path.write_text(script, encoding="utf-8")
                try:
                    proc = subprocess.run([sys.executable, "-I", str(path)], cwd=td,
                                          capture_output=True, timeout=10)
                    return proc.returncode == 0
                except subprocess.TimeoutExpired:
                    return False
        return bool(code) and run_tests(code, task["function"], task["cases"])
    return check_math(answer, str(task["expected"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="data/eval/stage2_holdout.jsonl")
    ap.add_argument("--models", required=True)
    ap.add_argument("--exclude-data", action="append", default=[])
    ap.add_argument("--out", default="runs/stage2-20261103/holdout.json")
    args = ap.parse_args()
    tasks = load_tasks(args.tasks, args.exclude_data)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    options = {"temperature": 0, "seed": 20261103, "num_ctx": 2048, "num_predict": 512,
               "stop": ["<|im_end|>", "<|endoftext|>"]}
    tokenizer = load_tokenizer("models/izumxai-2b")
    report = {"tasks_sha256": hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest(),
              "tasks": len(tasks), "excluded_sources": args.exclude_data,
              "options": options, "prompt_mode": "raw Qwen3 chat template, enable_thinking=False",
              "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
              "models": {}}
    session = requests.Session()
    def save():
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in [s.strip() for s in args.models.split(",") if s.strip()]:
        result = {"rows": [], "by_kind": {}, "started_unix": time.time()}
        report["models"][name] = result
        print(f"=== {name} ===", flush=True)
        for task in tasks:
            row = {"id": task["id"], "kind": task["kind"], "ok": False}
            start = time.monotonic()
            try:
                prompt = tokenizer.apply_chat_template([
                    {"role": "system", "content": SYSTEM_CODE if task["kind"] == "code" else SYSTEM_RU},
                    {"role": "user", "content": task["prompt"]}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                response = session.post("http://localhost:11434/api/generate", json={
                    "model": name, "stream": False, "raw": True, "keep_alive": "2m",
                    "prompt": prompt, "options": options,
                }, timeout=300)
                response.raise_for_status()
                data = response.json()
                answer = data["response"].strip()
                row.update(answer=answer, ok=bool(score(task, answer)),
                           tokens=data.get("eval_count", 0),
                           generation_seconds=data.get("eval_duration", 0) / 1e9,
                           done_reason=data.get("done_reason"))
            except Exception as exc:
                row["error"] = str(exc)
            row["wall_seconds"] = time.monotonic() - start
            result["rows"].append(row)
            print(f"{task['id']}: {'OK' if row['ok'] else 'FAIL'} ({row['wall_seconds']:.1f}s)", flush=True)
            save()
        for kind in sorted({t["kind"] for t in tasks}):
            rows = [r for r in result["rows"] if r["kind"] == kind]
            result["by_kind"][kind] = {"ok": sum(r["ok"] for r in rows), "n": len(rows)}
        result["accuracy"] = sum(r["ok"] for r in result["rows"]) / len(tasks)
        tokens = sum(r.get("tokens", 0) for r in result["rows"])
        seconds = sum(r.get("generation_seconds", 0) for r in result["rows"])
        result["generation_tokens_per_second"] = tokens / max(seconds, 1e-9)
        result["finished_unix"] = time.time()
        save()
        print(json.dumps({"model": name, **{k:v for k,v in result.items() if k != 'rows'}}, ensure_ascii=False), flush=True)
        unload = session.post("http://localhost:11434/api/generate", json={"model": name, "keep_alive": 0}, timeout=60)
        unload.raise_for_status()


if __name__ == "__main__":
    main()
