#!/usr/bin/env python3
"""
Export academic poster HTML to high-DPI PDF
Requires: playwright (pip install playwright && playwright install chromium)
"""

import asyncio
import argparse
import http.server
import socket
import socketserver
import threading
from pathlib import Path
from playwright.async_api import async_playwright


class _PosterHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serve poster assets from the HTML file's directory."""

    def log_message(self, format, *args):
        pass


def _start_http_server(directory: Path, port: int = 0) -> tuple[socketserver.TCPServer, int]:
    """Start a background HTTP server; returns (server, bound_port)."""
    handler = lambda *args, **kwargs: _PosterHTTPRequestHandler(  # noqa: E731
        *args, directory=str(directory), **kwargs
    )
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    bound_port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, bound_port


def _pick_free_port(preferred: int) -> int:
    """Use preferred port if free, otherwise let the OS assign one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


async def export_poster_to_pdf(
    html_file: str,
    output_pdf: str,
    dpi: int = 300,
    width_mm: float = 841,
    height_mm: float = 1189,
    port: int = 8765,
):
    """
    Export HTML poster to PDF with high DPI
    
    Args:
        html_file: Path to HTML file
        output_pdf: Output PDF path
        dpi: DPI resolution (default 300)
        width_mm: Poster width in mm (A0 portrait = 841mm)
        height_mm: Poster height in mm (A0 portrait = 1189mm)
        port: HTTP server port (default 8765)
    """
    # Convert mm to inches (1 inch = 25.4mm)
    width_inch = width_mm / 25.4
    height_inch = height_mm / 25.4
    
    # Convert to pixels at target DPI
    width_px = int(width_inch * dpi)
    height_px = int(height_inch * dpi)
    
    print(f"📄 Exporting poster to PDF...")
    print(f"   Input: {html_file}")
    print(f"   Output: {output_pdf}")
    print(f"   Size: {width_mm}×{height_mm}mm (A0 portrait)")
    print(f"   DPI: {dpi}")
    print(f"   Resolution: {width_px}×{height_px}px")
    html_path = Path(html_file).resolve()
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_file}")

    serve_port = _pick_free_port(port)
    httpd, serve_port = _start_http_server(html_path.parent, serve_port)
    url = f"http://127.0.0.1:{serve_port}/{html_path.name}"
    print(f"   HTTP server: {url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": width_px, "height": height_px},
            device_scale_factor=1,
        )
        
        # Load HTML via HTTP
        await page.goto(url)
        
        # Wait for all content to load
        await page.wait_for_load_state("networkidle")
        
        # Wait for JavaScript to load all section content
        # Check that all slots have been populated (no more "Loading…" text)
        await page.wait_for_function(
            """() => {
                const slots = ['slot-phase1', 'slot-phase2', 'slot-phase3', 'slot-cot', 'slot-results'];
                return slots.every(id => {
                    const el = document.getElementById(id);
                    return el && !el.textContent.includes('Loading');
                });
            }""",
            timeout=30000
        )
        
        await asyncio.sleep(3)  # Extra wait for fonts/images to render
        
        # Export to PDF
        await page.pdf(
            path=output_pdf,
            width=f"{width_mm}mm",
            height=f"{height_mm}mm",
            print_background=True,
            prefer_css_page_size=False,
            scale=1.0,
        )
        
        await browser.close()

    httpd.shutdown()
    httpd.server_close()

    output_size = Path(output_pdf).stat().st_size / (1024 * 1024)
    print(f"✅ PDF exported successfully!")
    print(f"   File size: {output_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Export academic poster HTML to high-DPI PDF"
    )
    parser.add_argument(
        "html_file",
        help="Path to HTML poster file (e.g., academic_poster.html)",
    )
    parser.add_argument(
        "-o", "--output",
        default="poster.pdf",
        help="Output PDF filename (default: poster.pdf)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution (default: 300)",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=841,
        help="Poster width in mm (default: 841 for A0)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=1189,
        help="Poster height in mm (default: 1189 for A0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Preferred HTTP port (default: 8765; auto-picks free port if busy)",
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(export_poster_to_pdf(
            html_file=args.html_file,
            output_pdf=args.output,
            dpi=args.dpi,
            width_mm=args.width,
            height_mm=args.height,
            port=args.port,
        ))
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
