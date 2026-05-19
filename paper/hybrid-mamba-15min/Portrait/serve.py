#!/usr/bin/env python3
"""Serve the ICLR poster locally and open it in the default browser."""

import http.server
import socketserver
import webbrowser
import threading
import os

PORT = 8080
POSTER = "academic_poster.html"
DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(DIR)

handler = http.server.SimpleHTTPRequestHandler
handler.log_message = lambda *a: None  # suppress access log


def _open():
    webbrowser.open(f"http://localhost:{PORT}/{POSTER}")


threading.Timer(0.6, _open).start()

print(f"  Poster → http://localhost:{PORT}/{POSTER}")
print("  Ctrl-C to stop.")

with socketserver.TCPServer(("", PORT), handler) as httpd:
    httpd.serve_forever()
