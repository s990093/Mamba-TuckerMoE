import time
from playwright.sync_api import sync_playwright
import os

def export_pdf():
    print("Starting PDF export...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # Navigate to the presentation URL
        url = 'http://127.0.0.1:8000/presentation.html'
        print(f"Navigating to {url}")
        page.goto(url, wait_until='networkidle')
        
        # Give it a bit more time to ensure MathJax and fonts are completely rendered
        time.sleep(3)
        
        output_file = 'presentation.pdf'
        print(f"Saving to {output_file}")
        
        # The CSS @media print is configured for A4 landscape with 0.5in margins
        # However, we should explicitly specify it here to be safe
        page.pdf(
            path=output_file,
            landscape=True,
            format='A4',
            print_background=True,
            margin={'top': '0', 'right': '0', 'bottom': '0', 'left': '0'} # The CSS already adds padding
        )
        
        browser.close()
        print(f"PDF exported successfully to {os.path.abspath(output_file)}")

if __name__ == '__main__':
    export_pdf()
