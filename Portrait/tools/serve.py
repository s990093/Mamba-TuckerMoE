#!/usr/bin/env python3
"""Serve the ICLR poster locally with HOT RELOAD and open it in the default browser."""

import http.server
import socketserver
import webbrowser
import threading
import os
import json

PORT = 8080
POSTER = "academic_poster.html"
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.chdir(DIR)

def get_max_mtime():
    max_mtime = 0
    for root, _, files in os.walk(DIR):
        for f in files:
            if f.endswith(('.html', '.css', '.md', '.png', '.jpg', '.js')):
                try:
                    mtime = os.path.getmtime(os.path.join(root, f))
                    if mtime > max_mtime:
                        max_mtime = mtime
                except OSError:
                    pass
    return max_mtime

class HotReloadHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass  # suppress access log

    def do_GET(self):
        if self.path == '/hot_reload_check':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            response = json.dumps({'max_mtime': get_max_mtime()})
            self.wfile.write(response.encode('utf-8'))
            return
            
        # Intercept main HTML file to inject the reload script
        if self.path == '/' or self.path.startswith('/' + POSTER):
            path = self.translate_path(self.path)
            if os.path.isfile(path):
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                inject_script = """
                <script>
                    (function() {
                        let lastMtime = null;
                        setInterval(() => {
                            fetch('/hot_reload_check')
                                .then(res => res.json())
                                .then(data => {
                                    if (lastMtime === null) {
                                        lastMtime = data.max_mtime;
                                    } else if (data.max_mtime > lastMtime) {
                                        console.log("Change detected, reloading...");
                                        location.reload();
                                    }
                                }).catch(err => {});
                        }, 800);
                    })();
                </script>
                """
                if '</body>' in content:
                    content = content.replace('</body>', inject_script + '</body>')
                else:
                    content += inject_script
                
                encoded = content.encode('utf-8')
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.end_headers()
                self.wfile.write(encoded)
                return

        # Avoid caching for sub-html files so reload fetches new ones
        if self.path.endswith('.html'):
            super().do_GET()
            # Note: SimpleHTTPRequestHandler doesn't easily let us add headers *after* do_GET
            # but it's ok, the browser usually refetches on full page reload anyway.
            return

        super().do_GET()

    def end_headers(self):
        # Force no caching globally to ensure hot updates show immediately
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def _open():
    webbrowser.open(f"http://localhost:{PORT}/{POSTER}")


threading.Timer(0.6, _open).start()

print(f"  Poster → http://localhost:{PORT}/{POSTER}")
print("  ⚡ Hot Reload is ACTIVE. Editing files will refresh the browser automatically.")
print("  Ctrl-C to stop.")

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), HotReloadHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()
