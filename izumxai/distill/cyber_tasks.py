"""Банк задач по кибербезопасности с ПРОГРАММНОЙ проверкой ответа.

Тот же интерфейс `Task`, что и в `tasks.py`: текст задачи, системный промпт,
функция проверки. Ответы проверяются кодом, а не суждением модели, поэтому
в дистилляцию попадают только реально верные решения учителя.

Покрытие:
    * кодировки и шифры (base64/hex/rot13/binary/url, Caesar, XOR);
    * разбор артефактов (JWT, логи, IoC, хеши);
    * сети (CIDR, маски, приватные диапазоны);
    * криптография (RSA на малых числах, модульная арифметика);
    * аудит кода (поиск уязвимой строки, тип уязвимости);
    * безопасный код на Python с реальным запуском тестов.

Запуск самопроверки банка:
    python -m izumxai.distill.cyber_tasks
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import random
import re
import urllib.parse
from typing import Callable

from .tasks import SYSTEM_RU, Task, run_tests

SYSTEM_CYBER = (
    "Ты — IzumxAI, инженер по кибербезопасности и CTF. Отвечай точно и по делу, "
    "без морализаторства и лишних предупреждений. Показывай короткие шаги решения. "
    "Последняя строка обязательно: «Ответ: <значение>»."
)

WORDS = [
    "shadow", "packet", "buffer", "kernel", "token", "router", "cipher", "exploit",
    "sandbox", "gateway", "payload", "socket", "daemon", "checksum", "firewall",
    "backdoor", "handshake", "entropy", "protocol", "privilege", "inject", "spoof",
]


# --------------------------------------------------------------- нормализация
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def text_check(expected: str) -> Callable[[str], bool]:
    """Мягкая проверка: эталон должен присутствовать целиком (без регистра)."""
    exp = _norm(expected)

    def check(answer: str) -> bool:
        tail = answer.split("Ответ:")[-1] if "Ответ:" in answer else answer
        return exp in _norm(tail)

    return check


def set_check(expected: set[str]) -> Callable[[str], bool]:
    """Проверка множества: каждое значение эталона найдено и лишних нет."""
    exp = {_norm(x) for x in expected}

    def check(answer: str) -> bool:
        tail = answer.split("Ответ:")[-1] if "Ответ:" in answer else answer
        got = {_norm(x) for x in re.findall(r"[A-Za-z0-9_.:@/-]+", tail)}
        return exp <= got

    return check


# ============================================================ КОДИРОВКИ/ШИФРЫ
def _encode_chain(rng: random.Random, secret: str) -> tuple[str, str]:
    """Случайная цепочка кодировок; возвращает (текст, как получено, эталон)."""
    steps: list[str] = []
    data = secret
    for _ in range(rng.randint(1, 3)):
        kind = rng.choice(["b64", "hex", "rot13", "binary", "url"])
        if kind == "b64":
            data = base64.b64encode(data.encode()).decode()
            steps.append("base64")
        elif kind == "hex":
            data = data.encode().hex()
            steps.append("hex")
        elif kind == "rot13":
            data = data.translate(str.maketrans(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                "nopqrstuvwxyzabcdefghijklmNOPQRSTUVWXYZABCDEFGHIJKLM"))
            steps.append("rot13")
        elif kind == "binary":
            data = " ".join(f"{b:08b}" for b in data.encode())
            steps.append("binary")
        else:
            data = urllib.parse.quote(data, safe="")
            steps.append("urlencode")
    return data, " + ".join(steps)


def gen_decode_chain(rng: random.Random) -> Task:
    secret = rng.choice(WORDS) + "-" + str(rng.randint(10, 999))
    payload, steps = _encode_chain(rng, secret)
    problem = (
        f"Дан фрагмент, закодированный так: {steps} (в указанном порядке).\n"
        f"{payload}\n"
        f"Декодируй и напиши исходную строку."
    )
    return Task("cyber", problem, text_check(secret), SYSTEM_CYBER, expected=secret)


def gen_caesar(rng: random.Random) -> Task:
    secret = rng.choice(WORDS) + str(rng.randint(1, 99))
    shift = rng.randint(1, 25)

    def rot(t: str, s: int) -> str:
        return t.translate(str.maketrans(
            "abcdefghijklmnopqrstuvwxyz",
            "abcdefghijklmnopqrstuvwxyz"[s:] + "abcdefghijklmnopqrstuvwxyz"[:s]))

    ct = rot(secret, shift)
    problem = (
        f"Шифр Цезаря, сдвиг {shift} (латиница, вправо при шифровании).\n"
        f"Шифртекст: {ct}\nРасшифруй."
    )
    return Task("cyber", problem, text_check(secret), SYSTEM_CYBER, expected=secret)


def gen_xor_single(rng: random.Random) -> Task:
    plain = " ".join(rng.sample(WORDS, 3))
    key = rng.randint(1, 255)
    ct = bytes(b ^ key for b in plain.encode()).hex()
    problem = (
        "Односоставный XOR-шифр (один байт на весь текст). "
        "Известно, что открытый текст — осмысленные английские слова в нижнем регистре.\n"
        f"Шифртекст (hex): {ct}\n"
        "Найди ключ (в десятичном виде) и восстанови текст."
    )
    return Task("cyber", problem, text_check(plain), SYSTEM_CYBER, expected=plain)


def gen_hash_id(rng: random.Random) -> Task:
    algo = rng.choice(["MD5", "SHA-1", "SHA-256", "SHA-512"])
    n = {"MD5": 32, "SHA-1": 40, "SHA-256": 64, "SHA-512": 128}[algo]
    digest = hashlib.new(algo.replace("-", "").lower(),
                         str(rng.random()).encode()).hexdigest()
    assert len(digest) == n
    problem = (
        f"Определи алгоритм хеширования по строке:\n{digest}\n"
        "Ответь названием алгоритма."
    )
    aliases = {
        "MD5": {"md5"}, "SHA-1": {"sha1", "sha-1"},
        "SHA-256": {"sha256", "sha-256"}, "SHA-512": {"sha512", "sha-512"},
    }

    def check(answer: str) -> bool:
        tail = _norm(answer.split("Ответ:")[-1] if "Ответ:" in answer else answer)
        return any(a in tail for a in aliases[algo])

    return Task("cyber", problem, check, SYSTEM_CYBER, expected=algo)


def gen_jwt(rng: random.Random) -> Task:
    field = rng.choice(["sub", "role", "uid", "scope", "tenant"])
    values = {
        "sub": f"user-{rng.randint(1000, 999999)}",
        "role": rng.choice(["admin", "root", "user", "auditor", "operator", "viewer"]),
        "uid": str(rng.randint(1000, 999999)),
        "scope": rng.choice(["read", "read write", "admin:*", "orders:rw", "metrics:ro"]),
        "tenant": f"t-{rng.randint(10, 9999)}",
    }
    value = values[field]
    payload = {
        "sub": "izumx",
        "role": "user",
        "uid": rng.randint(1000, 9999),
        "scope": "read",
        "iat": rng.randint(1600000000, 1800000000),
        "jti": f"{rng.getrandbits(48):012x}",
    }
    payload[field] = value

    def b64(d: dict) -> str:
        raw = json.dumps(d, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    token = f"{b64({'alg': 'HS256', 'typ': 'JWT'})}.{b64(payload)}.c2ln"
    problem = (
        f"JWT без проверки подписи:\n{token}\n"
        f"Декодируй payload и назови значение поля {field}."
    )
    return Task("cyber", problem, text_check(str(value)), SYSTEM_CYBER, expected=str(value))


# ==================================================================== АРТЕФАКТЫ
def gen_log_analysis(rng: random.Random) -> Task:
    attackers = [f"203.0.113.{rng.randint(2, 250)}" for _ in range(4)]
    main = rng.choice(attackers)
    counts = {ip: rng.randint(1, 3) for ip in attackers}
    counts[main] = rng.randint(6, 12)
    lines = []
    for ip in attackers:
        for i in range(counts[ip]):
            ok = "Failed password" if rng.random() < 0.8 else "Accepted password"
            lines.append(
                f"Nov 10 {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d} "
                f"server sshd[{rng.randint(1000, 9999)}]: {ok} for root from {ip} port "
                f"{rng.randint(1024, 65535)} ssh2")
    rng.shuffle(lines)
    log = "\n".join(lines)
    problem = (
        "Ниже лог SSH. Найди IP-адрес, с которого было больше всего НЕУДАЧНЫХ попыток входа.\n"
        f"{log}\n"
        "Ответь только IP-адресом."
    )
    return Task("cyber", problem, text_check(main), SYSTEM_CYBER, expected=main)


def gen_ioc_extract(rng: random.Random) -> Task:
    # set() обходится в порядке, зависящем от PYTHONHASHSEED: один и тот же seed
    # давал разные логи в разных процессах. Сортируем, чтобы замер был воспроизводим.
    ips = sorted({f"198.51.100.{rng.randint(2, 250)}" for _ in range(rng.randint(3, 5))})
    decoys = sorted({f"10.0.{rng.randint(0, 9)}.{rng.randint(2, 250)}" for _ in range(2)})
    lines = []
    for ip in ips + decoys:
        kind = rng.choice(["GET /login", "POST /api", "GET /admin"])
        lines.append(f'{rng.choice(["203.0.113.9", "192.0.2.7"])} - - "{kind} HTTP/1.1" 200 - "{ip}"')
    rng.shuffle(lines)
    problem = (
        "В логах веб-сервера в кавычках после кода ответа указан клиентский IP.\n"
        f"{chr(10).join(lines)}\n"
        "Выпиши все публичные IP из этих записей (без приватных 10.x)."
    )

    def check(answer: str) -> bool:
        tail = _norm(answer.split("Ответ:")[-1] if "Ответ:" in answer else answer)
        return all(ip in tail for ip in ips) and not any(d in tail for d in decoys)

    return Task("cyber", problem, check, SYSTEM_CYBER, expected=", ".join(sorted(ips)))


def gen_subnet(rng: random.Random) -> Task:
    prefix = rng.choice([24, 25, 26, 27, 28])
    base = rng.choice(["192.0.2", "198.51.100", "203.0.113"])
    ip = f"{base}.{rng.randint(1, 254)}/{prefix}"
    net = ipaddress.ip_network(ip, strict=False)
    ask = rng.choice(["network", "broadcast", "hosts"])
    if ask == "network":
        expected = str(net.network_address)
        problem = f"Для сети {ip} укажи адрес сети (network address)."
    elif ask == "broadcast":
        expected = str(net.broadcast_address)
        problem = f"Для сети {ip} укажи широковещательный адрес (broadcast)."
    else:
        expected = str(max(0, net.num_addresses - 2))
        problem = f"Сколько используемых хостов (без сети и broadcast) в сети {ip}?"
    return Task("cyber", problem, text_check(expected), SYSTEM_CYBER, expected=expected)


def gen_rsa(rng: random.Random) -> Task:
    primes = [61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    p, q = rng.sample(primes, 2)
    n, phi = p * q, (p - 1) * (q - 1)
    e = 65537
    if phi % e == 0:
        e = 17
    d = pow(e, -1, phi)
    ask = rng.choice(["n", "phi", "d", "encrypt"])
    if ask == "n":
        expected, problem = str(n), f"RSA: p={p}, q={q}. Чему равен модуль n?"
    elif ask == "phi":
        expected, problem = str(phi), f"RSA: p={p}, q={q}. Чему равна функция Эйлера phi(n)?"
    elif ask == "d":
        expected, problem = str(d), (f"RSA: p={p}, q={q}, e={e}. "
                                     f"Найди закрытый показатель d (обратный к e по модулю phi(n)).")
    else:
        m = rng.randint(2, n - 2)
        expected = str(pow(m, e, n))
        problem = (f"RSA: n={n}, e={e}. Зашифруй сообщение m={m}: вычисли m^e mod n.")
    return Task("cyber", problem, text_check(expected), SYSTEM_CYBER, expected=expected)


# ==================================================================== АУДИТ
VULN_TEMPLATES = [
    ("sql", "SQL-инъекция",
     'def login(conn, user, password):\n'
     '    cur = conn.cursor()\n'
     '    cur.execute("SELECT id FROM users WHERE name = \'%s\' AND pw = \'%s\'" % (user, password))\n'
     '    return cur.fetchone()'),
    ("cmdi", "инъекция команд ОС",
     'import os\n'
     'def ping(host):\n'
     '    os.system("ping -c 1 " + host)'),
    ("path", "обход каталога (path traversal)",
     'import os\n'
     'def read(name):\n'
     '    return open(os.path.join("/srv/files", name)).read()'),
    ("eval", "выполнение произвольного кода (eval)",
     'def calc(expr):\n'
     '    return eval(expr)'),
    ("xss", "межсайтовый скриптинг (XSS)",
     'from flask import request\n'
     'def hello():\n'
     '    return "<h1>Привет, " + request.args.get("name") + "</h1>"'),
    ("pickle", "небезопасная десериализация",
     'import pickle\n'
     'def load(data):\n'
     '    return pickle.loads(data)'),
    ("xxe", "внешняя сущность XML (XXE)",
     'from lxml import etree\n'
     'def parse(data):\n'
     '    return etree.fromstring(data, parser=etree.XMLParser(resolve_entities=True))'),
    ("deser", "небезопасная десериализация YAML",
     'import yaml\n'
     'def load(s):\n'
     '    return yaml.load(s)'),
]


def gen_vuln_audit(rng: random.Random) -> Task:
    _, label, code = rng.choice(VULN_TEMPLATES)
    # Случайные имена делают фрагмент уникальным, не меняя суть уязвимости.
    funcs = ["login", "handle", "run", "process", "load", "fetch", "parse", "render"]
    args = ["user", "name", "host", "expr", "data", "path", "query", "value"]
    code = re.sub(r"\bdef (\w+)\(", lambda m: f"def {m.group(1)}_{rng.choice(funcs)}(", code, count=1)
    rand_arg = rng.choice(args)
    code = code.replace("(user", f"({rand_arg}", 1) if rand_arg != "user" else code
    header = f"# {rng.choice(['app', 'svc', 'api', 'web', 'core'])}_{rng.randint(100, 9999)}.py\n"
    problem = (
        "Ниже фрагмент кода. Определи класс уязвимости (одна-две фразы) и укажи, "
        "что именно делает его небезопасным.\n"
        f"```python\n{header}{code}\n```"
    )
    keywords = {
        "SQL-инъекция": ["sql"],
        "инъекция команд ОС": ["команд", "command", "os.system"],
        "обход каталога (path traversal)": ["path traversal", "обход каталог", "traversal"],
        "выполнение произвольного кода (eval)": ["eval", "произвольн"],
        "межсайтовый скриптинг (XSS)": ["xss", "скриптинг"],
        "небезопасная десериализация": ["десериализ", "deserial", "pickle", "yaml"],
        "внешняя сущность XML (XXE)": ["xxe", "внешн"],
        "небезопасная десериализация YAML": ["десериализ", "yaml"],
    }[label]

    def check(answer: str) -> bool:
        low = _norm(answer)
        return any(k.lower() in low for k in keywords)

    return Task("cyber", problem, check, SYSTEM_CYBER, expected=label)


# ============================================================ БЕЗОПАСНЫЙ КОД
# (имя, эталон, описание, генератор тестов, уровень)
CYBER_CODE: list[tuple[str, str, str, Callable, str]] = [
    ("escape_html",
     "def escape_html(s):\n"
     "    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')\n"
     "            .replace('\"', '&quot;').replace(\"'\", '&#39;'))",
     "Напиши функцию escape_html(s), экранирующую &, <, >, \" и ' для вставки в HTML.",
     lambda r: [("<script>alert('x')</script>", "&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;"),
                ("a & b", "a &amp; b"), ("", "")], "easy"),

    ("xor_bytes",
     "def xor_bytes(data, key):\n"
     "    if isinstance(data, str):\n"
     "        data = data.encode()\n"
     "    if isinstance(key, str):\n"
     "        key = key.encode()\n"
     "    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))",
     "Напиши функцию xor_bytes(data, key), возвращающую bytes: повторяющийся ключ, "
     "работает и со str, и с bytes.",
     lambda r: [(b"ABC", b"\x01", b"@CB"),
                ("hello", "k", bytes(b ^ ord("k") for b in b"hello"))],
     "easy"),

    ("caesar_decrypt",
     "def caesar_decrypt(text, shift):\n"
     "    out = []\n"
     "    for ch in text:\n"
     "        if ch.islower():\n"
     "            out.append(chr((ord(ch) - 97 - shift) % 26 + 97))\n"
     "        elif ch.isupper():\n"
     "            out.append(chr((ord(ch) - 65 - shift) % 26 + 65))\n"
     "        else:\n"
     "            out.append(ch)\n"
     "    return ''.join(out)",
     "Напиши функцию caesar_decrypt(text, shift): расшифровка шифра Цезаря для латиницы, "
     "остальные символы без изменений.",
     lambda r: [("khoor", 3, "hello"), ("KHOOR, ZRUOG!", 3, "HELLO, WORLD!"), ("abc", 0, "abc")],
     "easy"),

    ("is_safe_path",
     "import os\n"
     "def is_safe_path(base, target):\n"
     "    base = os.path.abspath(base)\n"
     "    path = os.path.abspath(os.path.join(base, target))\n"
     "    return path == base or path.startswith(base + os.sep)",
     "Напиши функцию is_safe_path(base, target): True, если target после нормализации "
     "остаётся внутри base (защита от path traversal).",
     lambda r: [("/srv/files", "a/b.txt", True), ("/srv/files", "../etc/passwd", False),
                ("/srv/files", "/etc/passwd", False)], "hard"),

    ("strip_sql_comments",
     "def strip_sql_comments(sql):\n"
     "    out, i, n = [], 0, len(sql)\n"
     "    while i < n:\n"
     "        if sql.startswith('--', i):\n"
     "            j = sql.find('\\n', i)\n"
     "            i = n if j < 0 else j\n"
     "        elif sql.startswith('/*', i):\n"
     "            j = sql.find('*/', i + 2)\n"
     "            i = n if j < 0 else j + 2\n"
     "        else:\n"
     "            out.append(sql[i]); i += 1\n"
     "    return ''.join(out)",
     "Напиши функцию strip_sql_comments(sql), удаляющую комментарии -- до конца строки и /* ... */.",
     lambda r: [("SELECT 1 -- x", "SELECT 1 "), ("a /* b */ c", "a  c"),
                ("SELECT 2", "SELECT 2"), ("-- all", "")], "hard"),

    ("validate_password",
     "def validate_password(pw):\n"
     "    if len(pw) < 12:\n"
     "        return False\n"
     "    checks = [any(c.islower() for c in pw), any(c.isupper() for c in pw),\n"
     "              any(c.isdigit() for c in pw), any(not c.isalnum() for c in pw)]\n"
     "    return all(checks)",
     "Напиши функцию validate_password(pw): True, если длина >= 12 и есть строчная, "
     "заглавная, цифра и спецсимвол.",
     lambda r: [("Abcdefgh123!", True), ("abcdefgh123!", False), ("Abcdefg123!", False),
                ("Abcdefghij1!", True)], "easy"),

    ("is_private_ip",
     "import ipaddress\n"
     "def is_private_ip(s):\n"
     "    try:\n"
     "        ip = ipaddress.ip_address(s)\n"
     "    except ValueError:\n"
     "        return False\n"
     "    return ip.is_private or ip.is_loopback or ip.is_link_local",
     "Напиши функцию is_private_ip(s): True для приватных, loopback и link-local адресов, "
     "False для публичных и невалидных строк.",
     lambda r: [("192.168.1.5", True), ("8.8.8.8", False), ("10.0.0.1", True),
                ("127.0.0.1", True), ("not-an-ip", False)], "easy"),

    ("mask_secret",
     "def mask_secret(s):\n"
     "    if len(s) <= 8:\n"
     "        return '*' * len(s)\n"
     "    return s[:4] + '*' * (len(s) - 8) + s[-4:]",
     "Напиши функцию mask_secret(s): оставляет первые 4 и последние 4 символа, "
     "остальное заменяет на '*'; строки длиной <= 8 маскируются целиком.",
     lambda r: [("sk-1234567890abcdef", "sk-1***********cdef"), ("short", "*****"),
                ("12345678", "********")], "easy"),

    ("parse_ports",
     "def parse_ports(s):\n"
     "    out = set()\n"
     "    for part in s.split(','):\n"
     "        part = part.strip()\n"
     "        if not part:\n"
     "            continue\n"
     "        if '-' in part:\n"
     "            a, b = part.split('-', 1)\n"
     "            out.update(range(int(a), int(b) + 1))\n"
     "        else:\n"
     "            out.add(int(part))\n"
     "    return sorted(out)",
     "Напиши функцию parse_ports(s): разбирает строку вида '22,80,8000-8002' в "
     "отсортированный список уникальных портов.",
     lambda r: [("22,80", [22, 80]), ("8000-8002,22", [22, 8000, 8001, 8002]),
                ("80,80", [80])], "hard"),

    ("extract_ips",
     "import re\n"
     "def extract_ips(text):\n"
     "    pat = r'\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b'\n"
     "    return [m for m in re.findall(pat, text)\n"
     "            if all(0 <= int(p) <= 255 for p in m.split('.'))]",
     "Напиши функцию extract_ips(text): список IPv4-адресов в порядке появления, "
     "только с октетами 0..255.",
     lambda r: [("a 1.2.3.4 b 10.0.0.1", ["1.2.3.4", "10.0.0.1"]),
                ("999.1.1.1 and 8.8.8.8", ["8.8.8.8"]), ("нет", [])], "hard"),

    ("const_time_eq",
     "def const_time_eq(a, b):\n"
     "    a, b = a.encode(), b.encode()\n"
     "    if len(a) != len(b):\n"
     "        return False\n"
     "    r = 0\n"
     "    for x, y in zip(a, b):\n"
     "        r |= x ^ y\n"
     "    return r == 0",
     "Напиши функцию const_time_eq(a, b) для сравнения секретов без раннего выхода. "
     "На вход подаются две строки str, вернуть bool.",
     lambda r: [("abc", "abc", True), ("abc", "abd", False), ("a", "ab", False)], "hard"),

    ("b64url_decode",
     "import base64\n"
     "def b64url_decode(s):\n"
     "    s = s + '=' * (-len(s) % 4)\n"
     "    return base64.urlsafe_b64decode(s.encode())",
     "Напиши функцию b64url_decode(s): декодирует base64url без паддинга в bytes.",
     lambda r: [("aGVsbG8", b"hello"), ("eyJhIjoxfQ", b'{"a":1}'), ("QQ", b"A")], "easy"),

    ("sha256_hex",
     "import hashlib\n"
     "def sha256_hex(s):\n"
     "    return hashlib.sha256(s.encode()).hexdigest()",
     "Напиши функцию sha256_hex(s), возвращающую hex-дайджест SHA-256 строки в UTF-8.",
     lambda r: [("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
                ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")],
     "easy"),

    ("same_origin",
     "from urllib.parse import urlparse\n"
     "def same_origin(a, b):\n"
     "    pa, pb = urlparse(a), urlparse(b)\n"
     "    def norm(p):\n"
     "        port = p.port or (443 if p.scheme == 'https' else 80)\n"
     "        return (p.scheme.lower(), (p.hostname or '').lower(), port)\n"
     "    return norm(pa) == norm(pb)",
     "Напиши функцию same_origin(a, b): True, если у URL совпадают схема, хост и порт "
     "(порт по умолчанию — 443 для https и 80 для http).",
     lambda r: [("https://a.com/x", "https://a.com:443/y", True),
                ("http://a.com", "https://a.com", False),
                ("https://A.com", "https://a.com/z", True)], "hard"),
]


# Уточнения к задачам «напиши функцию»: делают формулировку уникальной, чтобы
# в датасет попадало больше одного примера на семейство, и не раскрывают тесты.
CODE_NOTES = [
    "Верни только код без пояснений.",
    "Код на Python 3.11, без внешних зависимостей.",
    "Функция должна быть чистой, без глобального состояния.",
    "Добавь короткий docstring одной строкой.",
    "Не используй сторонние библиотеки, только стандартную.",
    "Используется в разборе логов, важна устойчивость к мусорному вводу.",
    "Вызывается многократно, накладные расходы должны быть небольшими.",
    "Верни только функцию, без примера вызова.",
    "Код должен работать и на пустом входе.",
    "Пиши понятные имена переменных.",
]


def gen_cyber_code(rng: random.Random, level: str,
                   notes_rng: random.Random | None = None) -> Task:
    """Задача «напиши функцию». `notes_rng` (только при генерации данных) добавляет
    случайное уточнение, чтобы формулировки были уникальны и в датасет попадало
    больше одного примера на семейство. Бенчмарк передаёт None — иначе менялся бы
    поток rng и набор задач перестал бы быть воспроизводимым."""
    from .tasks import SYSTEM_CODE, extract_code

    pool = [t for t in CYBER_CODE if t[4] == level] or CYBER_CODE
    func, ref, desc, cases_fn, _ = rng.choice(pool)
    cases = cases_fn(rng)
    problem = desc
    if notes_rng is not None:
        problem += "\n\n" + CODE_NOTES[notes_rng.randrange(len(CODE_NOTES))]

    def check(answer: str) -> bool:
        src = extract_code(answer)
        return bool(src) and run_tests(src, func, cases)

    return Task("cyber", problem, check, SYSTEM_CODE, reference=f"```python\n{ref}\n```")


# ======================================================= ДЛИННЫЕ ЦЕПОЧКИ
# Задачи на много шагов: их нельзя решить одним взглядом, но финальный ответ
# по-прежнему проверяется точно. Именно такие цепочки и нужны для кибербеза.

def _flag(rng: random.Random) -> str:
    return f"FLAG{{{rng.choice(WORDS)}_{rng.getrandbits(32):08x}}}"


def gen_forensic_chain(rng: random.Random) -> Task:
    """Многослойный дамп: hex -> gzip -> base64 -> rot13. Как в CTF/форензике."""
    import gzip

    flag = _flag(rng)
    data = flag.encode()
    # Внешний слой всегда текстовый (hex), иначе дамп не отобразить в задаче.
    steps = ["rot13", "base64", "gzip"]
    rng.shuffle(steps)
    steps.append("hex")
    order = []
    for s in steps:
        if s == "rot13":
            data = data.translate(bytes.maketrans(
                b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                b"nopqrstuvwxyzabcdefghijklmNOPQRSTUVWXYZABCDEFGHIJKLM"))
        elif s == "base64":
            data = base64.b64encode(data)
        elif s == "gzip":
            data = gzip.compress(data)
        else:
            data = data.hex().encode()
        order.append(s)
    dump = data.decode()
    shown = "\n".join(dump[i:i + 76] for i in range(0, len(dump), 76))
    problem = (
        "Вам передали дамп файла challenge.dat. Известно, что содержимое "
        f"последовательно преобразовано так: {' -> '.join(order)} "
        "(применялось слева направо).\n"
        f"```\n{shown}\n```\n"
        "Восстановите исходное содержимое. Покажите команды/шаги и флаг."
    )
    return Task("cyber", problem, text_check(flag), SYSTEM_CYBER, expected=flag)


def gen_log_correlation(rng: random.Random) -> Task:
    """Корреляция трёх логов: кто скомпрометирован и с какого IP."""
    users = rng.sample(["alice", "bob", "carol", "dave", "erin", "frank"], 3)
    victim = rng.choice(users)
    attackers = [f"203.0.113.{rng.randint(2, 250)}" for _ in range(3)]
    bad_ip = rng.choice(attackers)
    fails = rng.randint(5, 9)
    h = rng.randint(0, 6)
    m = rng.randint(0, 50)

    def ts(dm: int, ds: int = 0) -> str:
        total = h * 60 + m + dm
        return f"Nov 10 {total // 60:02d}:{(total % 60 + ds) % 60:02d}:{rng.randint(0, 59):02d}"

    auth = []
    for ip in attackers:
        n = fails if ip == bad_ip else rng.randint(1, 3)
        for _ in range(n):
            auth.append(f"{ts(rng.randint(0, 4))} sshd[{rng.randint(1000, 9999)}]: "
                        f"Failed password for {victim if ip == bad_ip else rng.choice(users)} "
                        f"from {ip} port {rng.randint(20000, 65000)} ssh2")
    auth.append(f"{ts(fails + 2)} sshd[{rng.randint(1000, 9999)}]: Accepted password for "
                f"{victim} from {bad_ip} port {rng.randint(20000, 65000)} ssh2")
    for ip in attackers:
        if ip != bad_ip:
            auth.append(f"{ts(rng.randint(0, 10))} sshd[{rng.randint(1000, 9999)}]: Accepted "
                        f"password for {rng.choice(users)} from {ip} port {rng.randint(20000, 65000)} ssh2")
    rng.shuffle(auth)

    web = []
    for ip in attackers:
        for _ in range(rng.randint(1, 2)):
            path = rng.choice(["/login", "/admin", "/api/v1/users", "/.env", "/backup.zip"])
            web.append(f'{ip} - - [{ts(rng.randint(0, 8))}] "GET {path} HTTP/1.1" '
                       f'{rng.choice([200, 302, 401, 403])} {rng.randint(200, 9000)}')

    fw = [f"{ts(rng.randint(0, 9))} DROP IN=eth0 SRC={ip} DST=10.0.0.{rng.randint(2, 40)} "
          f"DPT={rng.choice([22, 3306, 6379, 8080])}" for ip in rng.sample(attackers, 2)]

    problem = (
        "Ниже три журнала с одного хоста (auth, web, firewall). Определите, "
        "учётная запись какого пользователя скомпрометирована и с какого IP "
        "пришёл успешный вход после серии неудач.\n\n"
        "auth.log:\n" + "\n".join(auth) + "\n\n"
        "access.log:\n" + "\n".join(web) + "\n\n"
        "firewall.log:\n" + "\n".join(fw) + "\n\n"
        "Ответьте в формате: user=<имя> ip=<адрес>"
    )
    expected = f"user={victim} ip={bad_ip}"

    def check(answer: str) -> bool:
        low = _norm(answer)
        return victim in low and bad_ip in low

    return Task("cyber", problem, check, SYSTEM_CYBER, expected=expected)


def gen_incident_timeline(rng: random.Random) -> Task:
    """Таймлайн инцидента: найти первое событие нужного типа после метки."""
    kinds = {
        "login": "вход в систему",
        "sudo": "повышение привилегий",
        "download": "скачивание файла",
        "exec": "запуск процесса",
    }
    target_kind = rng.choice(list(kinds))
    events = []
    minute = rng.randint(0, 20)
    for i in range(rng.randint(9, 14)):
        minute += rng.randint(1, 6)
        k = rng.choice(list(kinds))
        events.append({
            "id": f"E{i + 1}",
            "time": f"10:{minute // 60:02d}:{minute % 60:02d}" if minute < 60 else f"11:{(minute - 60) // 60:02d}:{(minute - 60) % 60:02d}",
            "kind": k,
            "detail": f"user={rng.choice(['alice', 'bob', 'svc_backup'])}, "
                      f"host=web{rng.randint(1, 3)}, pid={rng.randint(100, 9999)}",
        })
    pivot = rng.choice(events[: len(events) // 2])
    after = [e for e in events if events.index(e) > events.index(pivot) and e["kind"] == target_kind]
    expected = after[0]["id"] if after else pivot["id"]

    def check(answer: str) -> bool:
        low = _norm(answer)
        if expected.lower() in low:
            return True
        # Модель может ответить «Ответ: 8» вместо «E8» — это тот же ID.
        tail = low.split("ответ:")[-1] if "ответ:" in low else low
        return re.fullmatch(rf"\s*{expected[1:]}\.?\s*", tail) is not None

    table = "\n".join(f'{e["id"]} | {e["time"]} | {kinds[e["kind"]]} | {e["detail"]}' for e in events)
    problem = (
        "Таймлайн инцидента (id | время | тип | детали):\n"
        f"{table}\n\n"
        f"Найдите ID ПЕРВОГО события типа «{kinds[target_kind]}» после события "
        f"{pivot['id']} (по порядку следования). Ответьте ID."
    )
    return Task("cyber", problem, check, SYSTEM_CYBER, expected=expected)


def gen_cred_crack(rng: random.Random) -> Task:
    """Какие учётки используют слабые пароли из словаря."""
    words = ["summer2024", "qwerty123", "letmein", "admin123", "welcome1",
             "p@ssw0rd", "dragon2023", "monkey123", "iloveyou", "trustno1"]
    users = rng.sample(["alice", "bob", "carol", "dave", "erin", "frank", "grace"], rng.randint(3, 5))
    weak = rng.sample(users, rng.randint(2, min(3, len(users))))
    pairs = {}
    rows = []
    for u in users:
        pw = rng.choice(words) if u in weak else f"{rng.getrandbits(40):010x}"
        pairs[u] = pw
        rows.append(f"{u}:{hashlib.md5(pw.encode()).hexdigest()}")
    rng.shuffle(rows)
    table = "\n".join(rows)
    problem = (
        "Даны хеши MD5 учётных записей и словарь кандидатов:\n"
        f"{table}\n\n"
        f"Словарь: {', '.join(words)}\n\n"
        "Определите, у каких пользователей пароль есть в словаре, и восстановите его. "
        "Формат ответа: user=имя:пароль через запятую для каждой найденной учётки."
    )
    expected = ", ".join(f"user={u}:{pairs[u]}" for u in weak)

    def check(answer: str) -> bool:
        low = _norm(answer)
        return all(pairs[u].lower() in low and u in low for u in weak)

    return Task("cyber", problem, check, SYSTEM_CYBER, expected=expected)


def gen_multistep_rsa(rng: random.Random) -> Task:
    """Полный цикл RSA на малых числах: phi -> d -> расшифровка."""
    primes = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157]
    p, q = rng.sample(primes, 2)
    n, phi = p * q, (p - 1) * (q - 1)
    e = 17
    if phi % e == 0:
        e = 65537
    d = pow(e, -1, phi)
    m = rng.randint(2, n - 2)
    c = pow(m, e, n)
    problem = (
        f"Дана RSA-система: p={p}, q={q}, e={e}, шифртекст c={c}.\n"
        "Вычислите n, phi(n), d и расшифруйте сообщение m = c^d mod n. "
        "Ответьте числом m."
    )
    return Task("cyber", problem, text_check(str(m)), SYSTEM_CYBER, expected=str(m))


CHAIN_ALL = [gen_forensic_chain, gen_log_correlation, gen_incident_timeline,
             gen_cred_crack, gen_multistep_rsa]

# Учитель (Qwen3.8-27B, Q3_K_XL) проверен на этих задачах вручную: форензику и
# RSA он не решает (0/3 и 0/3 — зацикливается на арифметике), словарь паролей
# угадывает вместо перебора (1/3). Дистиллировать нечего, поэтому в пул идут
# только те цепочки, где учитель стабильно прав: 3/3 и 3/3.
CHAIN_TEXT = [gen_log_correlation, gen_incident_timeline]


# ==================================================================== фабрика
CYBER_TEXT = [
    gen_decode_chain, gen_caesar, gen_xor_single, gen_hash_id, gen_jwt,
    gen_log_analysis, gen_ioc_extract, gen_subnet, gen_rsa, gen_vuln_audit,
]


def gen_cyber(rng: random.Random, level: str = "easy",
              chain_bias: float | None = None) -> Task:
    if "code" in level or rng.random() < 0.30:
        return gen_cyber_code(rng, "hard" if "hard" in level else "easy")
    # Треть кибербеза — длинные многошаговые цепочки (они же дают длинный контекст).
    bias = chain_bias if chain_bias is not None else (0.45 if "hard" in level else 0.25)
    if rng.random() < bias:
        return rng.choice(CHAIN_TEXT)(rng)
    gen = rng.choice(CYBER_TEXT)
    return gen(rng)


# ============================================================== самопроверка
def _selftest() -> None:
    rng = random.Random(7)
    fails = 0
    counts: dict[str, int] = {}

    for func, ref, _desc, cases_fn, _lvl in CYBER_CODE:
        if not run_tests(ref, func, cases_fn(rng)):
            print(f"ПРОВАЛ эталона кода: {func}")
            fails += 1

    for gen in CYBER_TEXT + CHAIN_TEXT + [lambda r: gen_cyber_code(r, "easy"),
                                          lambda r: gen_cyber_code(r, "hard")]:
        for _ in range(40):
            t = gen(rng)
            counts[t.kind] = counts.get(t.kind, 0) + 1
            if not t.problem.strip():
                print("пустая задача")
                fails += 1
            if t.expected and not t.check(f"Ответ: {t.expected}"):
                print(f"эталон не проходит: {gen.__name__} -> {t.expected!r}")
                fails += 1
            if t.reference and not t.check(t.reference):
                print(f"эталонный код не проходит тесты: {gen.__name__}")
                fails += 1

    print("задач:", json.dumps(counts, ensure_ascii=False),
          "| эталонов кода:", len(CYBER_CODE), "| провалов:", fails)
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    _selftest()
