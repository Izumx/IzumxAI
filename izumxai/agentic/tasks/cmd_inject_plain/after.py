#!/usr/bin/env python3
"""Fixed ping wrapper: argv list, no shell.  GET /ping?host=<host>"""
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT = 8813


def run_ping(host: str) -> str:
    proc = subprocess.run(
        ["echo", "pinging", host],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return proc.stdout + proc.stderr


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        qs = parse_qs(urlparse(self.path).query)
        host = qs.get("host", [""])[0]
        try:
            body = run_ping(host)
            code = 200
        except Exception as exc:  # noqa: BLE001
            body = str(exc)
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
