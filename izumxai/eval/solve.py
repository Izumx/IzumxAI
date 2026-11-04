"""Code solver with reasoning and self-checking on the task's own examples.

Weights are unchanged; the gain comes from inference:
1. Qwen3 thinking mode (reasoning before code) instead of a terse one-shot answer.
2. Examples are parsed only from the task statement (doctests and lines like
   `f(x) == y`). Hidden tests are never seen by the solver.
3. If an example fails, the error goes back to the model and the code is repaired.

Evaluation on HumanEval uses the original check() only after the final answer.
"""
from __future__ import annotations

import argparse
import ast
import doctest
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

from ..distill.common import load_tokenizer
from ..distill.tasks import extract_code
from .holdout import score

SYSTEM = (
    "You are IzumxAI, an expert Python programmer. Think carefully about the "
    "specification, edge cases and every example before writing code. Answer with "
    "the complete function (and any imports) in one ```python block."
)
OLLAMA = "http://localhost:11434/api/generate"
ARROW_RE = re.compile(r"^\s*(?:>>>\s*)?([A-Za-z_]\w*\(.*\))\s*(?:==>|=>|==|➞|->|→|should return|returns?)\s*(.+?)\s*$")


def public_examples(prompt: str, function: str) -> list[tuple[str, str]]:
    """(call, expected) pairs visible in the statement."""
    try:
        node = next(n for n in ast.walk(ast.parse(prompt))
                    if isinstance(n, ast.FunctionDef) and n.name == function)
        prompt = ast.get_docstring(node) or ""
    except (SyntaxError, StopIteration):
        pass
    found = []
    try:
        doctests = doctest.DocTestParser().get_examples(prompt)
    except ValueError:
        doctests = []
    for ex in doctests:
        # A doctest's output runs until a blank line, so trailing prose may stick to it.
        want = (ex.want.strip().splitlines() or [""])[0].strip()
        try:
            ast.parse(want, mode="eval")
        except SyntaxError:
            continue
        if function in ex.source and len(ex.source.strip().splitlines()) == 1:
            found.append((ex.source.strip(), want))
    calls = {c for c, _ in found}
    for line in prompt.splitlines():
        line = line.strip().strip('"').strip("'")
        m = ARROW_RE.match(line)
        if not m or not m.group(1).startswith(function + "("):
            continue
        call, want = m.group(1), m.group(2).rstrip(".,;")
        try:
            ast.parse(call, mode="eval")
            ast.parse(want, mode="eval")
        except SyntaxError:
            continue
        if call not in calls:
            calls.add(call)
            found.append((call, want))
    return found


def check_examples(prefix: str, code: str, examples) -> tuple[int, str]:
    """Runs examples in a subprocess; returns (failures, first failure message)."""
    lines = [prefix, code, "import math", "_fails = []"]
    for call, want in examples:
        lines.append(
            "try:\n"
            f"    _got = {call}\n"
            f"    _exp = {want}\n"
            "    _ok = _got == _exp or (isinstance(_got, float) and isinstance(_exp, (int, float)) and math.isclose(_got, _exp, rel_tol=1e-6, abs_tol=1e-6))\n"
            f"    if not _ok: _fails.append({call!r} + ' returned ' + repr(_got) + ', expected ' + {want!r})\n"
            "except Exception as _e:\n"
            f"    _fails.append({call!r} + ' raised ' + type(_e).__name__ + ': ' + str(_e)[:200])")
    lines.append("print('__FAILS__', len(_fails)); print(_fails[0] if _fails else '')")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "examples.py"
        path.write_text("\n".join(lines), encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", str(path)], cwd=td, capture_output=True,
                                  timeout=10, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return len(examples) or 1, "timed out after 10 seconds (infinite loop or too slow)"
    out = proc.stdout.split("__FAILS__")
    if proc.returncode != 0 or len(out) < 2:
        return len(examples) or 1, (proc.stderr.strip().splitlines() or ["crashed"])[-1][:300]
    count, _, message = out[1].strip().partition("\n")
    return int(count), message.strip()


def generate(session, model, prompt, options):
    response = session.post(OLLAMA, json={"model": model, "stream": False, "raw": True,
                                          "keep_alive": "5m", "prompt": prompt, "options": options},
                            timeout=900)
    response.raise_for_status()
    return response.json()


def solve(session, tokenizer, model, task, think, rounds, options):
    function, prefix = task["function"], task.get("source_prompt", "")
    examples = public_examples(task.get("source_prompt", task["prompt"]), function)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}]
    best, trace, tokens = None, [], 0
    for attempt in range(rounds):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=think)
        opts = dict(options, seed=options["seed"] + attempt)
        data = generate(session, model, prompt, opts)
        tokens += data.get("eval_count", 0)
        text = data["response"]
        if think and "</think>" not in text:
            # Thinking budget exhausted: close the reasoning and ask for the code now.
            text = text.split("<think>")[-1].strip()
            forced = (prompt + "<think>\n" + text +
                      "\n\nI have thought enough, time to write the final solution.\n</think>\n\n")
            data = generate(session, model, forced, dict(opts, num_predict=1024))
            tokens += data.get("eval_count", 0)
            text = "</think>" + data["response"]
        answer = text.split("</think>")[-1].strip()
        code = extract_code(answer)
        if code is None:
            fails, message = len(examples) or 1, "no complete Python function in a ```python block"
        elif examples:
            fails, message = check_examples(prefix, code, examples)
        else:
            fails, message = 0, ""
        trace.append({"fails": fails, "done_reason": data.get("done_reason"), "message": message})
        if code is not None and (best is None or fails < best[0]):
            best = (fails, answer)
        if fails == 0:
            break
        messages += [{"role": "assistant", "content": answer},
                     {"role": "user", "content": f"Your code is wrong: {message}. "
                      "Find the bug, re-read the specification and return the corrected complete function."}]
    return (best[1] if best else answer), trace, tokens, len(examples)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="data/eval/humaneval_stage3_full.jsonl")
    ap.add_argument("--model", default="qwen3-local:1.7b")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-predict", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.tasks).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[:args.limit] if args.limit else rows
    # Qwen3 recommended sampling: thinking 0.6/0.95/20, non-thinking 0.7/0.8/20.
    options = {"temperature": 0.6 if args.think else 0.7, "top_p": 0.95 if args.think else 0.8,
               "top_k": 20, "seed": 20261103, "num_ctx": 8192, "num_predict": args.num_predict,
               "stop": ["<|im_end|>", "<|endoftext|>"]}
    tokenizer = load_tokenizer("models/izumxai-2b")
    report = {"tasks_sha256": hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest(),
              "model": args.model, "think": args.think, "rounds": args.rounds, "options": options,
              "rows": []}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    start = time.monotonic()
    for task in rows:
        t0 = time.monotonic()
        try:
            answer, trace, tokens, n_ex = solve(session, tokenizer, args.model, task, args.think,
                                                args.rounds, options)
            ok = bool(score(task, answer))
            row = {"id": task["id"], "ok": ok, "examples": n_ex, "trace": trace, "tokens": tokens,
                   "answer": answer}
        except Exception as exc:
            row = {"id": task["id"], "ok": False, "error": str(exc)}
        row["seconds"] = time.monotonic() - t0
        report["rows"].append(row)
        passed = sum(r["ok"] for r in report["rows"])
        print(f"{task['id']}: {'OK' if row['ok'] else 'FAIL'} rounds={len(row.get('trace', []))} "
              f"({row['seconds']:.0f}s) total {passed}/{len(report['rows'])}", flush=True)
        report["passed"], report["n"] = passed, len(report["rows"])
        report["wall_seconds"] = time.monotonic() - start
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
