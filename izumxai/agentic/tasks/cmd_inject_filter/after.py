#!/usr/bin/env python3
"""Fixed job runner: argv list, no shell.  GET /job?name=<name>"""
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT = 8811


def run_job(name: str) -> str:
    proc = subprocess.run(
        ["echo", "hello to", name],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return proc.stdout + proc.stderr


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        qs = parse_qs(urlparse(self.path).query)
        name = qs.get("name", [""])[0]
        try:
            body = run_job(name)
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
