#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


WATCH_DIRS = ("pages", "assets", ".")
WATCH_EXTS = {".html", ".css", ".js", ".png", ".svg", ".jpg", ".jpeg", ".webp"}


def compute_stamp(root: Path) -> str:
    parts: list[str] = []
    seen: set[Path] = set()

    for rel_dir in WATCH_DIRS:
        base = root / rel_dir
        if not base.exists() or not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            if path.name.startswith("."):
                continue
            if path.suffix.lower() not in WATCH_EXTS:
                continue
            rel = path.relative_to(root).as_posix()
            stat = path.stat()
            parts.append(f"{rel}:{stat.st_mtime_ns}:{stat.st_size}")

    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()
    return digest


class HotReloadHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/__hot":
            stamp = compute_stamp(self.directory_path)
            body = stamp.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    @property
    def directory_path(self) -> Path:
        return Path(self.directory or os.getcwd()).resolve()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    root = Path(__file__).resolve().parent.parent
    handler = partial(HotReloadHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"[hot] serving {root}")
    print(f"[hot] open http://127.0.0.1:{port}/presentation.html")
    print("[hot] watching pages/, assets/, and project html files")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[hot] stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
