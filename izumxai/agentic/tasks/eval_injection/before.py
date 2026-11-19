#!/usr/bin/env python3
"""Vulnerable calculator.  GET /calc?expr=<expr>  ->  str(eval(expr))"""
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT = 8815


def calc(expr: str) -> str:
    return str(eval(expr))  # noqa: S307 - deliberately vulnerable


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        qs = parse_qs(urlparse(self.path).query)
        expr = qs.get("expr", ["0"])[0]
        try:
            body = calc(expr)
            code = 200
        except Exception as exc:  # noqa: BLE001
            body = f"{type(exc).__name__}: {exc}"
            code = 500
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
