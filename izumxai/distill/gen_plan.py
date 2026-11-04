"""Датасет решений с планом проверки: учитель решает, затем проверяет крайние случаи.

Зачем два шага. Учитель qwen3:8b в родном режиме рассуждений пишет 1500–2500
токенов на простую задачу — это часы генерации, а студенту 1.7B такое «многословие»
скорее вредит. Поэтому:

1. Учитель пишет только рабочее решение (быстро, temperature 0.2).
2. Отдельным запросом проверяет его на крайних случаях: пустой ввод, один
   элемент, повторы, границы. Это и есть та привычка, которой не хватало модели:
   27 из 53 провалов HumanEval проходили видимый пример и падали на скрытых.
3. Текст проверки кладётся в <think>, финальный код — после него. В обучение
   попадает только то, что реально проходит официальные тесты.

Итоговый ответ ассистента:

    <think>
    ...разбор крайних случаев и проверка решения...
    </think>

    ```python
    ...
    ```

Запуск:
    python -m izumxai.distill.gen_plan --n 300 --out data/distill/stage4-plan/train.jsonl
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

from .common import build_labeled_example, load_tokenizer
from .tasks import Task, extract_code, gen_task

OLLAMA = "http://localhost:11434"

SOLVE_SYSTEM = (
    "Ты — IzumxAI, сильный программист. Выдай только рабочий код Python в одном "
    "блоке ```python: нужные импорты, вспомогательные функции, точка входа. "
    "Без пояснений до и после кода."
)

REVIEW_SYSTEM = (
    "Ты — IzumxAI, сильный программист. Тебе дают задачу и готовое решение. "
    "Проверь его на крайних случаях: пустой ввод, один элемент, повторы, "
    "отрицательные числа, границы диапазонов, тип возврата. Сначала коротко "
    "перечисли случаи и вывод по каждому (не более 8 строк), затем выдай "
    "итоговый рабочий код в одном блоке ```python."
)

FALLBACK_NOTES = "Проверяю крайние случаи и пример из условия, затем пишу код."


def prompt_key(problem: str) -> str:
    return hashlib.sha1(problem.strip().lower().encode("utf-8")).hexdigest()[:16]


def ask(session: requests.Session, model: str, system: str, user: str,
        temperature: float, num_ctx: int, num_predict: int, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx,
                    "num_predict": num_predict},
    }
    r = session.post(f"{OLLAMA}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return (r.json()["message"].get("content") or "").strip()


def run_script(script: str, timeout: int = 15) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        path = f.name
    try:
        proc = subprocess.run([sys.executable, "-I", path], capture_output=True,
                              timeout=timeout, encoding="utf-8", errors="replace")
        return proc.returncode == 0 and "OK" in proc.stdout
    except Exception:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def mbpp_task(row: dict) -> tuple[str, callable]:
    """Промпт MBPP и проверка решения официальными тестами."""
    code = row["code"]
    setup = row.get("test_setup_code", "")
    tests = row["test_list"] + row.get("challenge_test_list", [])
    functions = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)]
    calls = {n.func.id for n in ast.walk(ast.parse("\n".join(row["test_list"])))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    entry = next((n for n in functions if n.name in calls), None)
    if entry is None:
        raise ValueError("нет точки входа")
    problem = (row["text"] + "\n\nImplement " + entry.name + "("
               + ast.unparse(entry.args) + "). Include required helpers and imports.\n"
               "Example:\n" + row["test_list"][0])

    def check(answer: str) -> bool:
        src = extract_code(answer)
        if not src:
            return False
        try:
            defined = {n.name for n in ast.walk(ast.parse(src))
                       if isinstance(n, ast.FunctionDef)}
        except SyntaxError:
            return False
        if entry.name not in defined:
            return False
        return run_script(setup + "\n" + src + "\n" + "\n".join(tests) + "\nprint('OK')\n")

    return problem, check


def split_review(text: str) -> tuple[str, str]:
    """Делит ответ проверки на заметки до первого блока кода и сам блок."""
    idx = text.find("```")
    if idx < 0:
        return text.strip(), ""
    return text[:idx].strip(), text[idx:].strip()


def trim_notes(notes: str) -> str:
    """Убирает хвостовые заголовки вида «Итоговый код:» и разделители."""
    lines = [line.rstrip() for line in notes.strip().splitlines()]
    while lines:
        last = lines[-1].strip()
        if (not last or last.startswith("#") or last.startswith("---")
                or last.startswith("**") and last.endswith(":**")
                or last.lower().endswith(("код:", "код.", "решение:"))):
            lines.pop()
        else:
            break
    return "\n".join(lines).strip()


def compose(notes: str, code: str) -> str:
    notes = notes.strip() or FALLBACK_NOTES
    # Маркеры списка и заголовки внутри <think> выглядят естественно,
    # но лишние ``` в заметках сломали бы разбор ответа.
    notes = notes.replace("```", "")
    return f"<think>\n{notes}\n</think>\n\n{code.strip()}"


def make_item(problem: str, answer: str, task_type: str, level: str,
              source: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SOLVE_SYSTEM},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": answer},
        ],
        "task_type": task_type,
        "level": level,
        "verified": True,
        "source": source,
        "think": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/distill/stage4-plan/train.jsonl")
    ap.add_argument("--mbpp", default="runs/mbpp-train-source.jsonl")
    ap.add_argument("--mix", default="mbpp:80,alg:20", help="веса источников кода")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--num-ctx", type=int, default=3072)
    ap.add_argument("--solve-cap", type=int, default=900)
    ap.add_argument("--review-cap", type=int, default=420)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-len", type=int, default=1024,
                    help="максимум токенов полной беседы; длинное отбрасывается")
    ap.add_argument("--tokenizer", default="models/izumxai-2b")
    ap.add_argument("--max-attempts", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(args.tokenizer)

    rows = [json.loads(line) for line in
            Path(args.mbpp).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("test_list")]
    random.Random(args.seed).shuffle(rows)
    mbpp_pool = iter(rows)

    seen: set[str] = set()
    kept = 0
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(obj.get("messages", [])) > 1:
                seen.add(prompt_key(obj["messages"][1]["content"]))
            kept += 1
        if kept:
            print(f"найдено {kept} готовых примеров, продолжаем (повторы пропускаем)")

    mix = {}
    for part in args.mix.split(","):
        name, weight = part.split(":")
        mix[name.strip()] = int(weight)
    kinds = [name for name, weight in mix.items() for _ in range(weight)]

    rng = random.Random(args.seed + 1)
    session = requests.Session()
    t0 = time.time()
    start = kept
    attempts = 0
    fail = {"dup": 0, "wrong": 0, "http": 0, "long": 0, "empty": 0, "bad_ref": 0}

    with out.open("a", encoding="utf-8") as f:
        while kept < args.n:
            if args.max_attempts and attempts >= args.max_attempts:
                print("достигнут лимит попыток")
                break
            attempts += 1
            kind = rng.choice(kinds)
            level = "hard"

            if kind == "mbpp":
                try:
                    row = next(mbpp_pool)
                except StopIteration:
                    print("пул MBPP исчерпан")
                    break
                try:
                    problem, check = mbpp_task(row)
                except (ValueError, SyntaxError):
                    fail["bad_ref"] += 1
                    continue
                ref = "```python\n" + row["test_setup_code"] + "\n" + row["code"] + "\n```"
                if not check(ref):
                    fail["bad_ref"] += 1
                    continue
                task_type, source = "code", "mbpp"
            else:
                level = rng.choice(["easy", "hard"])
                task: Task = gen_task(rng, "code", level)
                problem, check = task.problem, task.check
                task_type, source = "code", "algorithmic"

            key = prompt_key(problem)
            if key in seen:
                fail["dup"] += 1
                continue

            def solve(temperature: float) -> str:
                return ask(session, args.model, SOLVE_SYSTEM, problem, temperature,
                           args.num_ctx, args.solve_cap, args.timeout)

            try:
                code = solve(0.2)
                if not check(code):
                    code = solve(0.7)  # одна повторная попытка другими сэмплами
                if not check(code):
                    fail["wrong"] += 1
                    continue

                review = ask(session, args.model, REVIEW_SYSTEM,
                             f"Задача:\n{problem}\n\nТекущее решение:\n{code}",
                             0.3, args.num_ctx, args.review_cap, args.timeout)
            except requests.RequestException as e:
                fail["http"] += 1
                print(f"HTTP ошибка: {e}", flush=True)
                time.sleep(3)
                continue

            notes, reviewed_code = split_review(review)
            notes = trim_notes(notes)
            final = reviewed_code if reviewed_code and check(reviewed_code) else code
            if not notes:
                fail["empty"] += 1
                continue

            answer = compose(notes, final)
            item = make_item(problem, answer, task_type, level, source)
            if build_labeled_example(tok, item["messages"], args.max_len) is None:
                fail["long"] += 1
                continue

            seen.add(key)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            kept += 1
            rate = (kept - start) / max(1e-9, time.time() - t0) * 60
            print(f"[{kept}/{args.n}] {kind} ok | {rate:.1f} примеров/мин | брак: {fail}",
                  flush=True)

    print(f"итого: {kept} примеров, попыток {attempts}, "
          f"время {(time.time() - t0) / 60:.1f} мин -> {out}")


if __name__ == "__main__":
    main()
