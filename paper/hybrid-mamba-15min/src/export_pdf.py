#!/usr/bin/env python3
"""Self-contained PDF exporter for the presentation slide deck.

Starts a temporary static HTTP server from the project root (so that
relative paths like ../assets/... in presentation.html resolve correctly),
uses Playwright to load the page with proper waits for KaTeX/fonts/images,
forces a print-friendly layout, then emits A4 landscape PDF.
"""

from __future__ import annotations

import http.server
import socketserver
import threading
import time
from functools import partial
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PDF = PROJECT_DIR / "presentation" / "presentation.pdf"


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def wait_for_render_ready(page) -> None:
    """Wait for external resources (fonts, KaTeX from CDN), images, and
    force all slides into a flow layout so @media print + page-break-after
    produces one page per slide.
    """
    # Let initial network activity settle a little
    page.wait_for_timeout(200)

    # 1. Web fonts (Google fonts + KaTeX)
    try:
        page.evaluate("() => document.fonts.ready")
    except Exception:
        pass

    # 2. Kick KaTeX rendering (scripts are deferred; initial DOMContentLoaded call may have raced)
    page.wait_for_timeout(250)
    page.evaluate("""
        () => {
            const stage = document.getElementById('stage') || document.body;
            if (typeof renderMathInElement === 'function') {
                try {
                    renderMathInElement(stage, {
                        delimiters: [
                            { left: '\\\\[', right: '\\\\]', display: true },
                            { left: '\\\\(', right: '\\\\)', display: false }
                        ],
                        throwOnError: false
                    });
                } catch (e) {}
            }
            return document.querySelectorAll('.katex').length;
        }
    """)

    # 3. Force print layout EARLY. This is critical:
    #    - JS keeps only .active slide visible with absolute positioning.
    #    - lazy iframes + images in later slides will only load reliably once visible in flow.
    page.evaluate("""
        () => {
            const root = document.documentElement;
            const body = document.body;
            const shell = document.querySelector('.shell');
            const stage = document.querySelector('.stage');
            const topbar = document.querySelector('.topbar');
            const progress = document.querySelector('.top-progress');
            const toolbar = document.querySelector('.toolbar');

            if (root) { root.style.height = 'auto'; root.style.overflow = 'visible'; }
            if (body) { body.style.height = 'auto'; body.style.overflow = 'visible'; }
            if (shell) {
                shell.style.height = 'auto';
                shell.style.minHeight = '0';
                shell.style.display = 'block';
                shell.style.overflow = 'visible';
            }
            if (stage) {
                stage.style.position = 'static';
                stage.style.height = 'auto';
                stage.style.overflow = 'visible';
            }

            // Hide chrome
            [topbar, progress, toolbar].forEach(el => { if (el) el.style.display = 'none'; });

            // Reveal + paginate all slides
            document.querySelectorAll('.slide').forEach((slide) => {
                slide.style.position = 'relative';
                slide.style.opacity = '1';
                slide.style.transform = 'none';
                slide.style.visibility = 'visible';
                slide.style.display = 'block';
                slide.style.pageBreakAfter = 'always';
                slide.style.breakAfter = 'page';
                slide.style.height = 'auto';
                slide.style.minHeight = '88vh';
                slide.style.overflow = 'visible';
            });
        }
    """)

    # 4. Now that everything is in flow, wait for images.
    #    Ignore the #fs-img placeholder (src="") and any images without a real src.
    try:
        page.wait_for_function(
            """() => {
                const allImgs = Array.from(document.querySelectorAll('img'));
                const realImgs = allImgs.filter(img => {
                    const s = (img.getAttribute('src') || '').trim();
                    if (!s || s === '') return false;
                    if (img.id === 'fs-img') return false;
                    return true;
                });
                if (realImgs.length === 0) return true;
                return realImgs.every(img => img.complete && img.naturalWidth > 0);
            }""",
            timeout=18000,
        )
    except Exception as e:
        # Don't fail the whole export for a slow or placeholder image.
        print(f"[warn] image readiness wait timed out or errored: {e}. Proceeding anyway...")

    # Give lazy-loaded iframes (simulator + middleware diagram) a chance to start
    page.wait_for_timeout(600)

    # Final settle for layout, font rasterization, image decode, and iframe paint
    page.wait_for_timeout(1600)


def export_pdf() -> None:
    print("Starting PDF export...")

    # Lazy import so we can give a nice error if playwright is missing
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: playwright. Install with:\n"
            "  pip install playwright\n"
            "  python -m playwright install chromium"
        ) from exc

    # Serve static files from the project root (presentation/ + assets/ + prototypes/ ...)
    # Quiet handler so request logs don't pollute export output.
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: N802
            pass

    handler = partial(QuietHandler, directory=str(PROJECT_DIR))

    # Ephemeral port avoids conflicts with dev servers
    server = ReusableTCPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[server] serving from {PROJECT_DIR} at http://127.0.0.1:{port}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            # Good viewport + scale for high-quality rasterized images inside slides
            context = browser.new_context(
                viewport={"width": 1580, "height": 1020},
                device_scale_factor=1.35,
            )
            page = context.new_page()

            url = f"http://127.0.0.1:{port}/presentation/presentation.html"
            print(f"Navigating to {url}")
            page.goto(url, wait_until="networkidle", timeout=90000)
            page.wait_for_load_state("load")

            wait_for_render_ready(page)

            OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving to {OUTPUT_PDF}")

            page.pdf(
                path=str(OUTPUT_PDF),
                landscape=True,
                format="A4",
                print_background=True,
                margin={"top": "0.35in", "right": "0.35in", "bottom": "0.35in", "left": "0.35in"},
            )

            browser.close()
            print(f"PDF exported successfully to {OUTPUT_PDF}")
    finally:
        server.shutdown()
        server.server_close()
        print("[server] stopped")


if __name__ == "__main__":
    export_pdf()
