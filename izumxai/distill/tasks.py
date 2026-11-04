"""Банк задач с программной проверкой ответа.

Единый интерфейс `Task`: текст задачи, системный промпт и функция проверки.
Один и тот же банк используется генерацией данных дистилляции, бенчмарком
и генерацией предпочтений (ORPO), поэтому задачи в обучении и в замере
всегда из одного распределения.

Уровни сложности:
    easy — то, что уже умеет текущая модель (одно-двухшаговые задачи);
    hard — многошаговые задачи, большие числа, крайние случаи в коде,
           строгий формат ответа в инструкциях.

Запуск самопроверки банка:
    python -m izumxai.distill.tasks
"""

from __future__ import annotations

import ast
import json
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .code_tasks import ALGORITHM_TASKS

SYSTEM_RU = (
    "Ты — IzumxAI, точный и краткий ассистент. Решай задачу в 2–3 коротких шага, "
    "без LaTeX, без $$, без повторения условия. Последняя строка обязательно: «Ответ: <значение>»."
)

# Instruction tasks set their own output format; a trailing «Ответ:» line would break it.
SYSTEM_INSTR = (
    "Ты — IzumxAI, точный ассистент. Строго соблюдай формат ответа, который просит "
    "пользователь, и не добавляй ничего сверх него."
)

SYSTEM_CODE = (
    "Ты пишешь только рабочий код Python в одном блоке ```python. "
    "Без лишних объяснений."
)


@dataclass
class Task:
    kind: str
    problem: str
    check: Callable[[str], bool]
    system: str = SYSTEM_RU
    expected: str | None = None  # эталон (для проверки банка и для ORPO)


# --------------------------------------------------------------- числа/текст
NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def last_number(text: str) -> str | None:
    m = NUM_RE.findall(text.replace(" ", ""))
    return m[-1].replace(",", ".") if m else None


def check_math(answer: str, expected: str) -> bool:
    """Сравнивает последнее число в ответе (после «Ответ:») с эталоном."""
    tail = answer.split("Ответ:")[-1] if "Ответ:" in answer else answer
    got = last_number(tail)
    if got is None:
        return False
    try:
        return abs(float(got) - float(expected)) < 1e-6
    except ValueError:
        return False


def _num_check(expected: str) -> Callable[[str], bool]:
    return lambda answer: check_math(answer, expected)


def all_numbers(text: str) -> list[float]:
    return [float(x.replace(",", ".")) for x in NUM_RE.findall(text)]


# =========================================================== МАТЕМАТИКА easy
def gen_arithmetic(rng: random.Random) -> tuple[str, str]:
    """Лёгкая арифметика: возвращает (условие, эталонный ответ)."""
    a, b, c = rng.randint(11, 99), rng.randint(11, 99), rng.randint(2, 9)
    kind = rng.choice(["paren", "discount", "unit", "leftover", "speed"])
    if kind == "paren":
        d = rng.randint(2, 9)
        return f"Вычисли: ({a} + {b}) * {c} - {d}", str((a + b) * c - d)
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


# =========================================================== МАТЕМАТИКА hard
def _int_or_none(x) -> int | None:
    return int(x) if isinstance(x, int) or float(x).is_integer() else None


def _math_chain(rng: random.Random) -> tuple[str, str]:
    """Две последовательные скидки: цена после них."""
    price = rng.randint(5, 25) * 400
    d1, d2 = rng.choice([(10, 20), (20, 25), (25, 20), (50, 10), (10, 50)])
    after = price * (100 - d1) // 100 * (100 - d2) // 100
    return (f"Товар стоит {price} рублей. Сначала цену снизили на {d1}%, "
            f"затем ещё на {d2}% от новой цены. Сколько рублей он стоит теперь?"), str(after)


def _math_percent_chain(rng: random.Random) -> tuple[str, str]:
    """Вклад: процент за год начисляется дважды."""
    base = rng.randint(20, 90) * 1000
    p = rng.choice([5, 10, 20])
    first = base + base * p // 100
    second = first + first * p // 100
    return (f"Вкладчик положил {base} рублей под {p}% годовых. "
            f"Проценты начисляются раз в год и остаются на счёте. "
            f"Сколько рублей будет на счёте через два года?"), str(second)


def _math_speed2(rng: random.Random) -> tuple[str, str]:
    """Средняя скорость на двух участках пути (целое число км/ч)."""
    for _ in range(80):
        v1, v2 = rng.randint(4, 15) * 10, rng.randint(4, 15) * 10
        h1, h2 = rng.randint(2, 5), rng.randint(2, 5)
        dist = v1 * h1 + v2 * h2
        if dist % (h1 + h2) == 0:
            return (f"Автомобиль {h1} часа ехал со скоростью {v1} км/ч, затем {h2} часа — "
                    f"со скоростью {v2} км/ч. Какова его средняя скорость на всём пути?"), \
                   str(dist // (h1 + h2))
    return (f"Автомобиль 2 часа ехал со скоростью 60 км/ч. Какова его средняя скорость?"), "60"


def _math_negative(rng: random.Random) -> tuple[str, str]:
    a, b, c = rng.randint(12, 60), rng.randint(3, 12), rng.randint(2, 30)
    return f"Вычисли: {c} - {a} * {b} + {a}", str(c - a * b + a)


def _math_bigmul(rng: random.Random) -> tuple[str, str]:
    a = rng.randint(120, 980)
    b = rng.randint(12, 99)
    c = rng.randint(100, 9000)
    return f"Вычисли: {a} * {b} - {c}", str(a * b - c)


def _math_fraction(rng: random.Random) -> tuple[str, str]:
    den = rng.choice([4, 5, 6, 8, 10, 12])
    num = rng.randint(1, den - 1)
    total = den * rng.randint(20, 120)
    return (f"Найди {num}/{den} от числа {total}."), str(total * num // den)


def _math_change(rng: random.Random) -> tuple[str, str]:
    price = rng.randint(120, 900)
    qty = rng.randint(2, 6)
    paid = ((price * qty) // 1000 + 1) * 1000
    return (f"Покупатель купил {qty} товара по {price} рублей и отдал {paid} рублей. "
            f"Сколько рублей сдачи он получил?"), str(paid - price * qty)


def _math_mixture(rng: random.Random) -> tuple[str, str]:
    """Смесь двух растворов: итоговая концентрация в процентах (целая)."""
    for _ in range(60):
        a, b = rng.choice([10, 20, 30, 40]), rng.choice([50, 60, 70, 80])
        m, n = rng.randint(1, 5), rng.randint(1, 5)
        total = m + n
        mass = a * m + b * n
        if mass % total == 0 and a != b:
            return (f"Смешали {m} кг раствора с концентрацией {a}% и {n} кг раствора "
                    f"с концентрацией {b}%. Какова концентрация смеси в процентах?"), \
                   str(mass // total)
    return _math_chain(rng)


def _math_avg(rng: random.Random) -> tuple[str, str]:
    n = rng.randint(4, 7)
    avg = rng.randint(10, 90)
    nums = [avg + rng.randint(-8, 8) for _ in range(n - 1)]
    last = avg * n - sum(nums)
    nums.append(last)
    return (f"Найди среднее арифметическое чисел: {', '.join(map(str, nums))}. "
            f"Ответ — целое число."), str(avg)


MATH_HARD = [
    _math_chain, _math_percent_chain, _math_speed2, _math_negative,
    _math_bigmul, _math_fraction, _math_change, _math_mixture, _math_avg,
]


# ================================================================ ЛОГИКА easy
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
    lowest = max(10, s - 9) * 10 + (s - max(10, s - 9))
    return (f"Сумма цифр двузначного числа равна {s}. Назови наименьшее такое число."), str(lowest)


# ================================================================ ЛОГИКА hard
def _logic_work3(rng: random.Random) -> tuple[str, str]:
    """Три насоса: время совместной работы (целое число минут)."""
    for _ in range(80):
        ts = [rng.randint(2, 9) * 20 for _ in range(3)]
        # 1/t = 1/a + 1/b + 1/c
        from fractions import Fraction
        inv = sum(Fraction(1, t) for t in ts)
        if inv and (1 / inv).denominator == 1:
            return (f"Три насоса наполняют резервуар за {ts[0]}, {ts[1]} и {ts[2]} минут "
                    f"соответственно. За сколько минут они наполнят резервуар вместе? "
                    f"Ответ — целое число минут."), str(int(1 / inv))
    return _math_chain(rng)


def _logic_sequence(rng: random.Random) -> tuple[str, str]:
    a = rng.randint(2, 20)
    d = rng.randint(2, 9)
    n = rng.randint(6, 15)
    head = ", ".join(str(a + i * d) for i in range(4))
    return (f"Дана последовательность: {head}, ... . "
            f"Какое число стоит на {n}-м месте?"), str(a + (n - 1) * d)


def _logic_handshake(rng: random.Random) -> tuple[str, str]:
    n = rng.randint(4, 12)
    return (f"В комнате {n} человек, и каждый пожал руку каждому ровно один раз. "
            f"Сколько всего было рукопожатий?"), str(n * (n - 1) // 2)


def _logic_clock(rng: random.Random) -> tuple[str, str]:
    h, m = rng.randint(1, 12), rng.choice([0, 10, 15, 20, 30, 40, 45, 50])
    angle = abs(30 * h + 0.5 * m - 6 * m)
    angle = min(angle, 360 - angle)
    minute_word = f"{m:02d}"
    return (f"Какой наименьший угол в градусах между часовой и минутной стрелками "
            f"в {h}:{minute_word}? Ответ — целое число градусов."), str(int(angle))


def _logic_people(rng: random.Random) -> tuple[str, str]:
    """Задача на пересечение множеств (два круга, целые числа)."""
    both = rng.randint(3, 15)
    only_a = rng.randint(5, 30)
    only_b = rng.randint(5, 30)
    return (f"В классе {only_a + both} учеников любят математику, "
            f"{only_b + both} — физику. Оба предмета любят {both} учеников. "
            f"Сколько учеников любят хотя бы один из этих предметов?"), str(only_a + both + only_b)


def _logic_train(rng: random.Random) -> tuple[str, str]:
    """Догоняющая задача с целым временем."""
    for _ in range(80):
        v1 = rng.randint(2, 8) * 10
        dv = rng.randint(1, 5) * 10
        v2 = v1 + dv
        ahead = rng.randint(2, 6) * (dv // 10)
        t = v1 * ahead / dv
        if float(t).is_integer():
            return (f"Из пункта А выехал автомобиль со скоростью {v1} км/ч. Через {ahead} часа "
                    f"вслед за ним выехал второй со скоростью {v2} км/ч. Через сколько часов "
                    f"после своего выезда второй догонит первый?"), str(int(t))
    return ("Первый едет 40 км/ч, второй стартовал через 2 часа со скоростью 60 км/ч. "
            "Через сколько часов после выезда он догонит первого?"), "4"


LOGIC_HARD = [_logic_work3, _logic_sequence, _logic_handshake, _logic_clock,
              _logic_people, _logic_train]


# ======================================================================= КОД
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
    """Выполняет код в отдельном процессе на реальных тестах."""
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


# (имя, эталон, описание, генератор тестов, уровень)
CODE_TASKS: list[tuple[str, str, str, Callable, str]] = [
    # ---- easy
    ("is_palindrome",
     "def is_palindrome(s):\n    t = ''.join(ch.lower() for ch in s if ch.isalnum())\n    return t == t[::-1]",
     "Напиши функцию is_palindrome(s), которая возвращает True, если строка читается одинаково в обе стороны (без учёта регистра и пробелов), иначе False.",
     lambda r: [("А роза упала на лапу Азора", True), ("привет", False), ("", True)], "easy"),
    ("count_vowels",
     "def count_vowels(s):\n    return sum(ch in 'аеёиоуыэюяaeiouy' for ch in s.lower())",
     "Напиши функцию count_vowels(s), которая считает количество гласных (русских и латинских) в строке.",
     lambda r: [("привет мир", 3), ("", 0), ("Python", 2)], "easy"),
    ("factorial",
     "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r",
     "Напиши функцию factorial(n), возвращающую факториал неотрицательного целого n.",
     lambda r: [(0, 1), (5, 120), (7, 5040)], "easy"),
    ("fib",
     "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
     "Напиши функцию fib(n), возвращающую n-е число Фибоначчи (fib(0)=0, fib(1)=1).",
     lambda r: [(0, 0), (10, 55), (15, 610)], "easy"),
    ("second_largest",
     "def second_largest(a):\n    u = sorted(set(a))\n    return u[-2] if len(u) > 1 else None",
     "Напиши функцию second_largest(a), возвращающую второй по величине уникальный элемент списка или None, если его нет.",
     lambda r: [([3, 1, 3, 2], 2), ([5], None), ([9, 8, 7], 8)], "easy"),
    ("is_prime",
     "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True",
     "Напиши функцию is_prime(n), проверяющую, является ли число простым.",
     lambda r: [(2, True), (9, False), (97, True), (1, False)], "easy"),
    ("gcd",
     "import math\ndef gcd(a, b):\n    return math.gcd(a, b)",
     "Напиши функцию gcd(a, b) для наибольшего общего делителя двух целых чисел.",
     lambda r: [(12, 18, 6), (7, 13, 1), (0, 5, 5)], "easy"),
    ("flatten",
     "def flatten(a):\n    out = []\n    for x in a:\n        out.extend(flatten(x) if isinstance(x, list) else [x])\n    return out",
     "Напиши функцию flatten(a), разворачивающую произвольно вложенные списки в плоский список.",
     lambda r: [([1, [2, [3, 4]], 5], [1, 2, 3, 4, 5]), ([], []), ([[1], [2]], [1, 2])], "easy"),
    ("binary_search",
     "def binary_search(a, t):\n    lo, hi = 0, len(a) - 1\n    while lo <= hi:\n        m = (lo + hi) // 2\n        if a[m] == t:\n            return m\n        if a[m] < t:\n            lo = m + 1\n        else:\n            hi = m - 1\n    return -1",
     "Напиши функцию binary_search(a, t), возвращающую индекс t в отсортированном списке a или -1.",
     lambda r: [([1, 3, 5, 7], 5, 2), ([1, 3, 5, 7], 4, -1)], "easy"),
    ("sum_even",
     "def sum_even(a):\n    return sum(x for x in a if x % 2 == 0)",
     "Напиши функцию sum_even(a), возвращающую сумму чётных чисел списка.",
     lambda r: [([1, 2, 3, 4], 6), ([], 0), ([-2, -3], -2)], "easy"),
    ("word_freq",
     "def word_freq(s):\n    d = {}\n    for w in s.lower().split():\n        d[w] = d.get(w, 0) + 1\n    return d",
     "Напиши функцию word_freq(s), возвращающую словарь частот слов в строке (без учёта регистра).",
     lambda r: [("a b a", {"a": 2, "b": 1}), ("", {})], "easy"),
    ("reverse_words",
     "def reverse_words(s):\n    return ' '.join(s.split()[::-1])",
     "Напиши функцию reverse_words(s), меняющую порядок слов в строке на обратный.",
     lambda r: [("один два три", "три два один"), ("a", "a"), ("  a  b  ", "b a")], "easy"),
    ("clamp",
     "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
     "Напиши функцию clamp(x, lo, hi), ограничивающую значение x диапазоном [lo, hi].",
     lambda r: [(5, 1, 3, 3), (-2, 0, 10, 0), (2, 1, 3, 2)], "easy"),

    # ---- hard
    ("rle",
     "def rle(s):\n    out, i = [], 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]:\n            j += 1\n        out.append(s[i] + str(j - i))\n        i = j\n    return ''.join(out)",
     "Напиши функцию rle(s), которая кодирует строку длинами серий: 'aaabbc' -> 'a3b2c1'. Пустая строка -> пустая строка.",
     lambda r: [("aaabbc", "a3b2c1"), ("", ""), ("xyz", "x1y1z1")], "hard"),
    ("roman",
     "def roman(n):\n    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n    out = ''\n    for v, s in vals:\n        while n >= v:\n            out += s\n            n -= v\n    return out",
     "Напиши функцию roman(n), переводящую целое число от 1 до 3999 в римскую запись.",
     lambda r: [(1994, "MCMXCIV"), (4, "IV"), (3999, "MMMCMXCIX"), (1, "I")], "hard"),
    ("is_balanced",
     "def is_balanced(s):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    st = []\n    for ch in s:\n        if ch in '([{':\n            st.append(ch)\n        elif ch in pairs:\n            if not st or st.pop() != pairs[ch]:\n                return False\n    return not st",
     "Напиши функцию is_balanced(s), проверяющую правильность расстановки скобок трёх видов. Пустая строка считается правильной.",
     lambda r: [("([{}])", True), ("([)]", False), ("", True), ("((", False)], "hard"),
    ("merge_intervals",
     "def merge_intervals(a):\n    if not a:\n        return []\n    a = sorted(a)\n    out = [list(a[0])]\n    for s, e in a[1:]:\n        if s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out",
     "Напиши функцию merge_intervals(a), объединяющую пересекающиеся интервалы. Вход — список пар [начало, конец], выход — список списков.",
     lambda r: [([[1, 3], [2, 6], [8, 10]], [[1, 6], [8, 10]]), ([], []), ([[5, 7]], [[5, 7]])], "hard"),
    ("transpose",
     "def transpose(m):\n    return [list(row) for row in zip(*m)] if m else []",
     "Напиши функцию transpose(m), транспонирующую матрицу (список списков). Пустая матрица -> пустой список.",
     lambda r: [([[1, 2, 3], [4, 5, 6]], [[1, 4], [2, 5], [3, 6]]), ([], []), ([[1]], [[1]])], "hard"),
    ("dedupe",
     "def dedupe(a):\n    seen, out = set(), []\n    for x in a:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out",
     "Напиши функцию dedupe(a), удаляющую дубликаты с сохранением порядка первого появления.",
     lambda r: [([3, 1, 3, 2, 1], [3, 1, 2]), ([], []), (["b", "a", "b"], ["b", "a"])], "hard"),
    ("anagrams",
     "def anagrams(words):\n    d = {}\n    for w in words:\n        d.setdefault(''.join(sorted(w)), []).append(w)\n    return sorted(sorted(v) for v in d.values())",
     "Напиши функцию anagrams(words), группирующую анаграммы. Внутри группы слова по алфавиту, группы отсортированы.",
     lambda r: [(["eat", "tea", "tan", "ate", "nat", "bat"],
                 [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]), ([], [])], "hard"),
    ("caesar",
     "def caesar(s, k):\n    out = []\n    for ch in s:\n        if 'a' <= ch <= 'z':\n            out.append(chr((ord(ch) - 97 + k) % 26 + 97))\n        else:\n            out.append(ch)\n    return ''.join(out)",
     "Напиши функцию caesar(s, k), сдвигающую строчные латинские буквы на k позиций по кругу; остальные символы не меняются.",
     lambda r: [("abc", 1, "bcd"), ("xyz", 2, "zab"), ("a b", 1, "b c")], "hard"),
    ("is_anagram",
     "def is_anagram(a, b):\n    return sorted(a.lower()) == sorted(b.lower())",
     "Напиши функцию is_anagram(a, b), проверяющую, являются ли две строки анаграммами (без учёта регистра).",
     lambda r: [("listen", "silent", True), ("a", "ab", False), ("", "", True)], "hard"),
    ("chunk",
     "def chunk(a, n):\n    return [a[i:i + n] for i in range(0, len(a), n)]",
     "Напиши функцию chunk(a, n), разбивающую список на части длины n; последняя часть может быть короче.",
     lambda r: [([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]), ([], 3, []), ([1, 2], 5, [[1, 2]])], "hard"),
    ("max_subarray",
     "def max_subarray(a):\n    best = cur = a[0]\n    for x in a[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best",
     "Напиши функцию max_subarray(a), возвращающую максимальную сумму непрерывного непустого подотрезка.",
     lambda r: [([-2, 1, -3, 4, -1, 2, 1, -5, 4], 6), ([-5], -5), ([1, 2, 3], 6)], "hard"),
    ("two_sum",
     "def two_sum(a, t):\n    seen = {}\n    for i, x in enumerate(a):\n        if t - x in seen:\n            return [seen[t - x], i]\n        seen[x] = i\n    return []",
     "Напиши функцию two_sum(a, t), возвращающую список из двух индексов элементов, дающих в сумме t (первый найденный вариант), или пустой список.",
     lambda r: [([2, 7, 11, 15], 9, [0, 1]), ([1, 2], 5, []), ([3, 3], 6, [0, 1])], "hard"),
    ("count_long_words",
     "def count_long_words(s, n):\n    return sum(1 for w in s.split() if len(w) > n)",
     "Напиши функцию count_long_words(s, n), считающую слова длиннее n символов.",
     lambda r: [("a bb ccc dddd", 2, 2), ("", 0, 0), ("abcd", 3, 1)], "hard"),
    ("sum_digits",
     "def sum_digits(n):\n    n = abs(n)\n    return sum(int(c) for c in str(n))",
     "Напиши функцию sum_digits(n), считающую сумму цифр целого числа (модуль числа). sum_digits(0) == 0.",
     lambda r: [(123, 6), (-45, 9), (0, 0)], "hard"),
    ("rotate",
     "def rotate(a, k):\n    if not a:\n        return []\n    k %= len(a)\n    return a[-k:] + a[:-k] if k else list(a)",
     "Напиши функцию rotate(a, k), сдвигающую список вправо на k позиций (k может быть больше длины).",
     lambda r: [([1, 2, 3, 4], 1, [4, 1, 2, 3]), ([1, 2], 3, [2, 1]), ([], 5, [])], "hard"),
    ("compress_spaces",
     "def compress_spaces(s):\n    return ' '.join(s.split())",
     "Напиши функцию compress_spaces(s), заменяющую любые последовательности пробелов одним пробелом и убирающую пробелы по краям.",
     lambda r: [("  a   b ", "a b"), ("", ""), ("a", "a")], "hard"),
]


CODE_TASKS.extend(ALGORITHM_TASKS)

# ================================================================ ИНСТРУКЦИИ
_CITIES = ["Казань", "Тюмень", "Омск", "Сочи", "Пермь", "Уфа", "Самара"]
_COMPANIES = ["Север", "Восток", "Гранит", "Ротор", "Лада"]
_TOPICS = [
    "Компания «{company}» открыла завод в {city} в {year} году. Инвестиции составили "
    "{invest} млрд рублей, на предприятии работают {people} человек. К {year2} году "
    "объём выпуска вырос на {growth}%.",
    "Учёные выяснили, что сон менее {hours} часов повышает риск ошибок при вождении "
    "на {risk}%. В исследовании участвовали {people} водителей из {regions} регионов. "
    "Отчёт опубликован в {year} году.",
    "Курс рубля укрепился на {pct}% за неделю. Эксперты связывают это с ростом цен на "
    "нефть до {oil} долларов за баррель. Объём торгов вырос до {people} млн долларов в {year} году.",
    "Фестиваль в {city} собрал {people} гостей за {days} дня. В программе было "
    "{events} концертов, а бюджет составил {invest} млн рублей в {year} году.",
]


def make_text(rng: random.Random) -> str:
    tpl = rng.choice(_TOPICS)
    year = rng.randint(2015, 2024)
    return tpl.format(
        company=rng.choice(_COMPANIES), city=rng.choice(_CITIES), year=year,
        year2=year + rng.randint(3, 6), invest=rng.choice(["0,8", "1,2", "2,5", "3,4"]),
        people=rng.randint(120, 900), growth=rng.randint(5, 60),
        hours=rng.randint(3, 6), risk=rng.randint(10, 60), regions=rng.randint(2, 12),
        pct=rng.choice(["1,2", "2,3", "0,5", "3,1"]), oil=rng.randint(60, 95),
        days=rng.randint(2, 4), events=rng.randint(5, 40),
    )


def _instr_numbers(text: str) -> Callable[[str], bool]:
    expected = all_numbers(text)

    def check(answer: str) -> bool:
        got = all_numbers(answer)
        return got == expected

    return check


def _instr_json(text: str) -> Callable[[str], bool]:
    def check(answer: str) -> bool:
        m = re.search(r"\{.*\}", answer, re.S)
        if not m:
            return False
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return False
        return (isinstance(obj, dict) and len(obj) >= 2
                and all(isinstance(k, str) and k for k in obj))

    return check


def _instr_bullets(answer: str) -> bool:
    lines = [l.strip() for l in answer.splitlines() if l.strip()]
    bullets = [l for l in lines if re.match(r"^(\d[\).]|[-•*])\s+", l)]
    return len(bullets) == 3 and len(lines) == 3 and all(len(l.split()) <= 14 for l in bullets)


def _instr_short(answer: str) -> bool:
    return 0 < len(answer.split()) <= 25 and "\n" not in answer.strip()


def _instr_int(expected: int) -> Callable[[str], bool]:
    def check(answer: str) -> bool:
        nums = all_numbers(answer)
        return bool(nums) and abs(nums[-1] - expected) < 1e-6

    return check


def gen_instruction(rng: random.Random, level: str = "easy") -> Task:
    task = _gen_instruction(rng, level)
    task.system = SYSTEM_INSTR
    return task


def _gen_instruction(rng: random.Random, level: str = "easy") -> Task:
    """Задача на следование инструкции по заданному тексту."""
    text = make_text(rng)
    n_words = len(text.split())

    if level == "hard":
        kind = rng.choice(["numbers", "json", "bullets", "count_words"])
        if kind == "numbers":
            return Task("instruction", f"Выпиши все числа из текста в порядке появления, "
                                       f"через запятую, без слов:\n\n{text}", _instr_numbers(text))
        if kind == "json":
            return Task("instruction", f"Верни только JSON-объект с двумя ключами — "
                                       f"«суть» (до 12 слов) и «числа» (список чисел из текста):\n\n{text}",
                        _instr_json(text))
        if kind == "bullets":
            return Task("instruction",
                        f"Перепиши текст ровно тремя пунктами списка (1., 2., 3.), "
                        f"каждый не длиннее 14 слов, без вступления и вывода:\n\n{text}",
                        _instr_bullets)
        return Task("instruction", f"Сколько слов в тексте? Ответ — одно целое число:\n\n{text}",
                    _instr_int(n_words))

    kind = rng.choice(["bullets", "short", "numbers"])
    if kind == "bullets":
        return Task("instruction",
                    f"Перепиши текст в 3 пунктах списка, каждый пункт не длиннее 12 слов:\n\n{text}",
                    lambda a: _instr_bullets(a) or (a.count("\n") >= 2))
    if kind == "short":
        return Task("instruction", f"Сократи текст до одного предложения, сохранив главный смысл:\n\n{text}",
                    _instr_short)
    return Task("instruction", f"Выдели из текста все числа, выведи их через запятую:\n\n{text}",
                _instr_numbers(text))


# ================================================================== фабрика
def gen_math(rng: random.Random, level: str = "easy") -> Task:
    if level == "hard":
        text, expected = rng.choice(MATH_HARD)(rng)
    else:
        text, expected = gen_arithmetic(rng)
    return Task("math", text, _num_check(expected), expected=expected)


def gen_logic_task(rng: random.Random, level: str = "easy") -> Task:
    if level == "hard":
        text, expected = rng.choice(LOGIC_HARD)(rng)
    else:
        text, expected = gen_logic(rng)
    return Task("logic", text, _num_check(expected), expected=expected)


def gen_code(rng: random.Random, level: str = "easy") -> Task:
    pool = [t for t in CODE_TASKS if t[4] == level] or CODE_TASKS
    func, ref, desc, cases_fn, _ = rng.choice(pool)
    cases = cases_fn(rng)

    def check(answer: str) -> bool:
        src = extract_code(answer)
        return bool(src) and run_tests(src, func, cases)

    return Task("code", desc, check, SYSTEM_CODE)


def gen_task(rng: random.Random, kind: str, level: str = "easy") -> Task:
    if kind == "math":
        return gen_math(rng, level)
    if kind == "logic":
        return gen_logic_task(rng, level)
    if kind == "code":
        return gen_code(rng, level)
    if kind == "instruction":
        return gen_instruction(rng, level)
    raise ValueError(f"неизвестный тип задачи: {kind}")


# ============================================================== самопроверка
def _selftest() -> None:
    rng = random.Random(1234)
    fails = 0
    counts: dict[str, int] = {}

    # эталонные реализации кода должны проходить свои же тесты
    for func, ref, desc, cases_fn, level in CODE_TASKS:
        cases = cases_fn(rng)
        if not run_tests(ref, func, cases):
            print(f"ПРОВАЛ эталона: {func}")
            fails += 1

    # математика/логика: эталон проходит проверку, соседнее число — нет
    for level in ("easy", "hard"):
        for kind in ("math", "logic"):
            for _ in range(300):
                t = gen_task(rng, kind, level)
                counts[f"{kind}/{level}"] = counts.get(f"{kind}/{level}", 0) + 1
                good = f"Ответ: {t.expected}"
                if not t.check(good):
                    print(f"эталон не проходит: {kind}/{level} -> {t.expected} | {t.problem[:70]}")
                    fails += 1
                    continue
                wrong = f"Ответ: {float(t.expected) + 1:g}"
                if t.check(wrong):
                    print(f"проверка слишком мягкая: {kind}/{level} {t.expected} vs {wrong}")
                    fails += 1

    # код и инструкции: генераторы не падают, задачи непустые
    for level in ("easy", "hard"):
        for kind in ("code", "instruction"):
            for _ in range(30):
                t = gen_task(rng, kind, level)
                counts[f"{kind}/{level}"] = counts.get(f"{kind}/{level}", 0) + 1
                if not t.problem.strip():
                    print(f"пустая задача: {kind}/{level}")
                    fails += 1

    print("задач по типам:", json.dumps(counts, ensure_ascii=False))
    print(f"эталонов кода: {len(CODE_TASKS)} | провалов: {fails}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    _selftest()
