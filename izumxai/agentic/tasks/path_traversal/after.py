#!/usr/bin/env python3
"""Fixed static file server: resolve and confine to PUBLIC.  GET /file?name=<name>"""
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
        path = os.path.realpath(os.path.join(PUBLIC, name))
        if not path.startswith(os.path.realpath(PUBLIC) + os.sep):
            raw = b"forbidden"
            code = 403
        else:
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
