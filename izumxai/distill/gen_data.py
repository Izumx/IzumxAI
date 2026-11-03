"""Генерация данных дистилляции от учителя (qwen3:8b через Ollama).

Идея: учитель решает задачи, мы проверяем ответ ПРОГРАММНО (математика —
точное сравнение, код — реальный запуск тестов) и оставляем только
верифицированные пары. Так 2B-ученик получает сигнал качества 8B,
но без галлюцинаций учителя.

Запуск:
    python -m izumxai.distill.gen_data --model qwen3:8b --n 500 --out data/distill/train.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

OLLAMA = "http://localhost:11434"

SYSTEM_RU = (
    "Ты — IzumxAI, точный и краткий ассистент. Решай задачу по шагам, "
    "без воды и без повторения условия. В конце обязательно строка «Ответ: <значение>»."
)

# --------------------------------------------------------------- математика
def gen_arithmetic(rng: random.Random) -> tuple[str, str]:
    a, b, c = rng.randint(11, 99), rng.randint(11, 99), rng.randint(2, 9)
    kind = rng.choice(["paren", "discount", "unit", "leftover", "speed"])
    if kind == "paren":
        text = f"Вычисли: {a} * {c} - {b} + {c} * {rng.randint(2, 9)}"
        d = rng.randint(2, 9)
        text = f"Вычисли: ({a} + {b}) * {c} - {d}"
        return text, str((a + b) * c - d)
    if kind == "discount":
        price = a * 100
        pct = rng.choice([5, 10, 15, 20, 25, 30, 40, 50])
        return (f"Товар стоит {price} рублей. Скидка {pct}%. "
                f"Сколько рублей составляет скидка?"), str(price * pct // 100)
    if kind == "unit":
        qty = rng.randint(2, 9)
        price_total = qty * b * 10
        want = rng.randint(2, 9)
        return (f"{qty} одинаковых ручек стоят {price_total} рублей. "
                f"Сколько рублей стоят {want} таких ручек?"), str(price_total // qty * want)
    if kind == "leftover":
        x = rng.randint(50, 200)
        add = rng.randint(10, 80)
        sold = rng.randint(5, x + add - 1)
        return (f"На складе было {x} коробок. Привезли ещё {add}, затем продали {sold}. "
                f"Сколько коробок осталось?"), str(x + add - sold)
    per_hour = rng.randint(3, 12)
    n, h = rng.randint(4, 10), rng.randint(2, 6)
    m, k = rng.randint(3, 9), rng.randint(2, 5)
    return (f"Бригада из {n} человек делает {per_hour * n * h} деталей за {h} часов. "
            f"Сколько деталей сделают {m} человек за {k} часов при той же "
            f"производительности труда?"), str(per_hour * m * k)


NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def last_number(text: str) -> str | None:
    m = NUM_RE.findall(text.replace(" ", ""))
    return m[-1].replace(",", ".") if m else None


def check_math(answer: str, expected: str) -> bool:
    got = last_number(answer.split("Ответ:")[-1] if "Ответ:" in answer else answer)
    if got is None:
        return False
    try:
        return abs(float(got) - float(expected)) < 1e-6
    except ValueError:
        return False


# --------------------------------------------------------------------- код
CODE_TASKS = [
    ("is_palindrome", "def is_palindrome(s):\n    return s == s[::-1]",
     "Напиши функцию is_palindrome(s), которая возвращает True, если строка читается одинаково в обе стороны (без учёта регистра и пробелов), иначе False.",
     lambda r: [("А роза упала на лапу Азора", True), ("привет", False), ("", True)]),
    ("count_vowels", "def count_vowels(s):\n    return sum(ch in 'аеёиоуыэюяaeiouy' for ch in s.lower())",
     "Напиши функцию count_vowels(s), которая считает количество гласных (русских и латинских) в строке.",
     lambda r: [("привет мир", 3), ("", 0), ("Python", 1)]),
    ("factorial", "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r",
     "Напиши функцию factorial(n), возвращающую факториал неотрицательного целого n.",
     lambda r: [(0, 1), (5, 120), (7, 5040)]),
    ("fib", "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
     "Напиши функцию fib(n), возвращающую n-е число Фибоначчи (fib(0)=0, fib(1)=1).",
     lambda r: [(0, 0), (10, 55), (15, 610)]),
    ("second_largest", "def second_largest(a):\n    u = sorted(set(a))\n    return u[-2] if len(u) > 1 else None",
     "Напиши функцию second_largest(a), возвращающую второй по величине уникальный элемент списка или None, если его нет.",
     lambda r: [([3, 1, 3, 2], 2), ([5], None), ([9, 8, 7], 8)]),
    ("is_prime", "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True",
     "Напиши функцию is_prime(n), проверяющую, является ли число простым.",
     lambda r: [(2, True), (9, False), (97, True), (1, False)]),
    ("gcd", "import math\ndef gcd(a, b):\n    return math.gcd(a, b)",
     "Напиши функцию gcd(a, b) для наибольшего общего делителя двух целых чисел.",
     lambda r: [(12, 18, 6), (7, 13, 1), (0, 5, 5)]),
    ("flatten", "def flatten(a):\n    out = []\n    for x in a:\n        out.extend(flatten(x) if isinstance(x, list) else [x])\n    return out",
     "Напиши функцию flatten(a), разворачивающую произвольно вложенные списки в плоский список.",
     lambda r: [([1, [2, [3, 4]], 5], [1, 2, 3, 4, 5]), ([], []), ([[1], [2]], [1, 2])]),
    ("binary_search", "def binary_search(a, t):\n    lo, hi = 0, len(a) - 1\n    while lo <= hi:\n        m = (lo + hi) // 2\n        if a[m] == t:\n            return m\n        if a[m] < t:\n            lo = m + 1\n        else:\n            hi = m - 1\n    return -1",
     "Напиши функцию binary_search(a, t), возвращающую индекс t в отсортированном списке a или -1.",
     lambda r: [([1, 3, 5, 7], 5, 2), ([1, 3, 5, 7], 4, -1)]),
    ("sum_even", "def sum_even(a):\n    return sum(x for x in a if x % 2 == 0)",
     "Напиши функцию sum_even(a), возвращающую сумму чётных чисел списка.",
     lambda r: [([1, 2, 3, 4], 6), ([], 0)]),
    ("word_freq", "def word_freq(s):\n    d = {}\n    for w in s.lower().split():\n        d[w] = d.get(w, 0) + 1\n    return d",
     "Напиши функцию word_freq(s), возвращающую словарь частот слов в строке (без учёта регистра).",
     lambda r: [("a b a", {"a": 2, "b": 1}), ("", {})]),
    ("reverse_words", "def reverse_words(s):\n    return ' '.join(s.split()[::-1])",
     "Напиши функцию reverse_words(s), меняющую порядок слов в строке на обратный.",
     lambda r: [("один два три", "три два один"), ("a", "a")]),
    ("clamp", "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
     "Напиши функцию clamp(x, lo, hi), ограничивающую значение x диапазоном [lo, hi].",
     lambda r: [(5, 1, 3, 3), (-2, 0, 10, 0), (2, 1, 3, 2)]),
]

CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def extract_code(answer: str) -> str | None:
    m = CODE_RE.search(answer)
    src = m.group(1) if m else answer
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    if not any(isinstance(n, ast.FunctionDef) for n in tree.body):
        return None
    return src


def run_tests(src: str, func: str, cases: list[tuple]) -> bool:
    lines = []
    for i, c in enumerate(cases):
        args = c[:-1]
        lines.append(f"assert {func}(*{list(args)!r}) == {c[-1]!r}, 'case {i}'")
    script = src + "\n" + "\n".join(lines) + "\nprint('OK')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        path = f.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True, timeout=10,
                              encoding="utf-8", errors="replace")
        return proc.returncode == 0 and "OK" in proc.stdout
    except Exception:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


# ------------------------------------------------------------------ логика
def gen_logic(rng: random.Random) -> tuple[str, str]:
    """Логические задачи с программно проверяемым целым ответом."""
    kind = rng.choice(["work", "age", "digits"])

    if kind == "work":
        while True:
            a, b = rng.randint(2, 9), rng.randint(2, 9)
            ta, tb = a * 12, b * 12
            if a != b and (ta * tb) % (ta + tb) == 0:
                break
        return (f"Первый насос наполняет резервуар за {ta} минут, второй — за {tb} минут. "
                f"За сколько минут они наполнят его вместе? Ответ — целое число минут."), \
               str(ta * tb // (ta + tb))

    if kind == "age":
        k = rng.randint(2, 4)
        son = rng.randint(6, 16)
        t = rng.randint(2, 12)
        father = k * (son + t) - t
        return (f"Сейчас отцу {father} лет, а сыну {son}. "
                f"Через сколько лет отец будет ровно в {k} раза старше сына?"), str(t)

    d = rng.randint(3, 9)
    s = rng.randint(3, 15)
    for n in range(10, 100):
        if n % d == 0 and sum(int(c) for c in str(n)) == s:
            return (f"Сумма цифр двузначного числа равна {s}, и оно делится на {d}. "
                    f"Назови наименьшее такое число."), str(n)
    # если комбинация невозможна — отдаём задачу только на сумму цифр
    lowest = max(10, s - 9) * 10 + (s - max(10, s - 9))
    return (f"Сумма цифр двузначного числа равна {s}. Назови наименьшее такое число."), str(lowest)


# -------------------------------------------------------------- инструкции
INSTRUCTION_TASKS = [
    ("Перепиши текст в 3 пунктах списка, каждый пункт не длиннее 12 слов:\n\n{text}",
     lambda a: a.count("\n") >= 2 and all(len(l.split()) <= 14 for l in a.splitlines() if l.strip()[:2] in {"1.", "2.", "3.", "- "})),
    ("Сократи текст до одного предложения, сохранив главный смысл:\n\n{text}",
     lambda a: len(a.split()) <= 40),
    ("Выдели из текста все даты и числа, выведи их через запятую:\n\n{text}",
     lambda a: len(a) < 300),
    ("Ответь на вопрос по тексту одним словом или короткой фразой:\n\n{text}\n\nВопрос: о чём этот текст?",
     lambda a: 0 < len(a.split()) <= 15),
]

SEED_TEXTS = [
    "Компания «Север» открыла второй завод в Казани в 2019 году. Инвестиции составили 1,2 млрд рублей, "
    "на предприятии работают 340 человек. К 2024 году объём выпуска вырос на 45%.",
    "Учёные выяснили, что сон менее шести часов повышает риск ошибок при вождении на 30%. "
    "В исследовании участвовали 2 400 водителей из восьми регионов.",
    "Курс рубля укрепился на 2,3% за неделю. Эксперты связывают это с ростом цен на нефть до 78 долларов за баррель.",
]


# ------------------------------------------------------------------ учитель
def ask(session: requests.Session, model: str, messages: list[dict],
        temperature: float = 0.6, num_ctx: int = 4096, think: bool = False) -> str:
    payload = {
        "model": model, "messages": messages, "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": 1024},
    }
    r = session.post(f"{OLLAMA}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def make_item(kind: str, problem: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_RU},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": answer},
        ],
        "task_type": kind,
        "verified": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/distill/train.jsonl")
    ap.add_argument("--mix", default="math:35,logic:15,code:30,instruction:20",
                    help="доли задач в процентах")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--think", action="store_true", help="включить reasoning учителя")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = sum(1 for _ in out.open(encoding="utf-8")) if out.exists() else 0
    if done:
        print(f"найдено {done} готовых примеров, продолжаем")

    mix = {}
    for part in args.mix.split(","):
        k, v = part.split(":")
        mix[k.strip()] = int(v)
    kinds = [k for k, v in mix.items() for _ in range(v)]

    rng = random.Random(args.seed)
    session = requests.Session()
    t0 = time.time()
    kept = done
    attempts = 0
    fail = {"parse": 0, "wrong": 0, "http": 0}

    with out.open("a", encoding="utf-8") as f:
        while kept < args.n:
            kind = rng.choice(kinds)
            attempts += 1
            try:
                if kind == "math":
                    problem, expected = gen_arithmetic(rng)
                    answer = ask(session, args.model, [
                        {"role": "system", "content": SYSTEM_RU},
                        {"role": "user", "content": problem}], think=args.think)
                    ok = check_math(answer, expected)
                    if not ok:
                        fail["wrong"] += 1
                        continue
                elif kind == "logic":
                    problem, expected = gen_logic(rng)
                    answer = ask(session, args.model, [
                        {"role": "system", "content": SYSTEM_RU},
                        {"role": "user", "content": problem}], think=args.think)
                    if not check_math(answer, expected):
                        fail["wrong"] += 1
                        continue
                elif kind == "code":
                    func, ref, desc, cases_fn = rng.choice(CODE_TASKS)
                    cases = cases_fn(rng)
                    problem = desc
                    answer = ask(session, args.model, [
                        {"role": "system", "content": "Ты пишешь только рабочий код Python в одном блоке ```python. Без лишних объяснений."},
                        {"role": "user", "content": desc}])
                    src = extract_code(answer)
                    if not src:
                        fail["parse"] += 1
                        continue
                    if not run_tests(src, func, cases):
                        fail["wrong"] += 1
                        continue
                elif kind == "instruction":
                    tpl, validator = rng.choice(INSTRUCTION_TASKS)
                    text = rng.choice(SEED_TEXTS)
                    problem = tpl.format(text=text)
                    answer = ask(session, args.model, [
                        {"role": "system", "content": SYSTEM_RU},
                        {"role": "user", "content": problem}])
                    if not validator(answer):
                        fail["wrong"] += 1
                        continue
                else:
                    continue
            except requests.RequestException as e:
                fail["http"] += 1
                print(f"HTTP ошибка: {e}", flush=True)
                time.sleep(2)
                continue

            f.write(json.dumps(make_item(kind, problem, answer), ensure_ascii=False) + "\n")
            f.flush()
            kept += 1
            rate = kept / max(1e-9, time.time() - t0) * 60
            print(f"[{kept}/{args.n}] {kind} ok | {rate:.1f} примеров/мин | "
                  f"брак: {fail}", flush=True)

    print(f"готово: {kept} примеров, попыток {attempts}, "
          f"время {(time.time() - t0) / 60:.1f} мин -> {out}")


if __name__ == "__main__":
    main()
