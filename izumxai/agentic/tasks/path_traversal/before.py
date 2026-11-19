#!/usr/bin/env python3
"""Vulnerable static file server.  GET /file?name=<name>

IZUMX_ROOT points at a directory that contains public/ (served) and secret.txt.
"""
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT = 8814
ROOT = os.environ.get("IZUMX_ROOT", ".")
PUBLIC = os.path.join(ROOT, "public")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        qs = parse_qs(urlparse(self.path).query)
        name = qs.get("name", [""])[0]
        path = os.path.join(PUBLIC, name)
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
            code = 200
        except Exception as exc:  # noqa: BLE001
            raw = str(exc).encode()
            code = 404
        print(f"RESP {raw.decode('utf-8', 'replace')!r}", flush=True)
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
