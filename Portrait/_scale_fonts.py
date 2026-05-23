"""Scale CSS font-size pt values across the whole poster by one uniform FACTOR.
Covers academic_poster.html (main shell) + all s_*.html slots.
Skips SVG font-size attributes (they live in fixed viewBoxes and would overflow).
"""
import re
from pathlib import Path

FACTOR = 1.3
HERE = Path(__file__).parent
FILES = [HERE / "academic_poster.html"] + sorted(HERE.glob("s_*.html"))

css_pt = re.compile(r"(font-size\s*:\s*)(\d+(?:\.\d+)?)(pt)")

def scale_pt(m):
    new_size = round(float(m.group(2)) * FACTOR, 1)
    return f"{m.group(1)}{new_size}{m.group(3)}"

for f in FILES:
    if not f.exists():
        continue
    txt = f.read_text(encoding="utf-8")
    new = css_pt.sub(scale_pt, txt)
    if new != txt:
        f.write_text(new, encoding="utf-8")
        print(f"updated {f.name}")
    else:
        print(f"no change {f.name}")
