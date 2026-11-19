#!/usr/bin/env python3
"""Fixed calculator: whitelist the expression, then evaluate AST-free arithmetic."""
import ast
import operator
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT = 8815
OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
       ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("unsupported expression")


def calc(expr: str) -> str:
    return str(_eval(ast.parse(expr, mode="eval")))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        qs = parse_qs(urlparse(self.path).query)
        expr = qs.get("expr", ["0"])[0]
        try:
            body = calc(expr)
            code = 200
        except Exception as exc:  # noqa: BLE001
            body = f"{type(exc).__name__}: {exc}"
            code = 400
        print(f"RESP {body!r}", flush=True)
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
