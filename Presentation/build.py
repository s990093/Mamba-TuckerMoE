#!/usr/bin/env python3
"""
Build script: extracts SVGs from Portrait/ files and injects them into
Presentation/index_skeleton.html → Presentation/index.html
"""
import re, pathlib

ROOT = pathlib.Path(__file__).parent.parent
PORTRAIT = ROOT / "Portrait"
PRES = ROOT / "Presentation"

# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def read(name):
    return (PORTRAIT / name).read_text(encoding="utf-8")

def extract_svgs(html):
    """Return all top-level <svg ...>...</svg> elements as a list of strings."""
    chunks, i = [], 0
    while i < len(html):
        start = html.find("<svg", i)
        if start == -1:
            break
        depth, pos = 0, start
        while pos < len(html):
            open_  = html.find("<svg", pos + 1)
            close  = html.find("</svg>", pos)
            if close == -1:
                break
            if open_ != -1 and open_ < close:
                depth += 1
                pos = open_
            else:
                depth -= 1
                pos = close + 6
                if depth < 0:
                    chunks.append(html[start:pos])
                    break
        i = pos
    return chunks


def set_viewbox(svg_str, vb):
    return re.sub(r'viewBox="[^"]*"', f'viewBox="{vb}"', svg_str, count=1)


def set_style(svg_str, style):
    """Replace or inject style attribute on the root <svg> element."""
    # replace existing style="..." in the opening <svg ...> tag
    header_end = svg_str.index(">")
    header = svg_str[:header_end]
    if 'style="' in header:
        header = re.sub(r'style="[^"]*"', f'style="{style}"', header)
    else:
        header = header + f' style="{style}"'
    return header + svg_str[header_end:]


# ──────────────────────────────────────────────
# Extract SVGs
# ──────────────────────────────────────────────

phase1 = read("s_phase1.html")
phase2 = read("s_phase2.html")
phase3 = read("s_phase3.html")

svgs1 = extract_svgs(phase1)   # [0]=macro arch, [1]=Tucker full
svgs2 = extract_svgs(phase2)   # [0]=CoT timeline, [1]=expert balance, [2]=convergence (hidden)
svgs3 = extract_svgs(phase3)   # [0]=Jacobi

macro_arch_svg = svgs1[0]
tucker_full    = svgs1[1]
cot_timeline   = svgs2[0]
expert_balance = svgs2[1]
jacobi_svg     = svgs3[0]

# Slide 4: macro arch – full width, auto height
macro_arch_svg = set_style(macro_arch_svg,
    "font-family:'Times New Roman',Times,serif;width:100%;display:block;")

# Slide 6: Tucker panel i (x=2..378, y=8..512 → viewBox "2 8 376 504")
tucker_p1 = set_viewbox(tucker_full, "2 8 376 504")
tucker_p1 = set_style(tucker_p1,
    "font-family:'Times New Roman',Times,serif;display:block;width:360px;height:481px;flex-shrink:0;")

# Slide 7: Tucker panel ii (x=386..1128, y=8..512 → viewBox "386 8 742 504")
tucker_p2 = set_viewbox(tucker_full, "386 8 742 504")
tucker_p2 = set_style(tucker_p2,
    "font-family:'Times New Roman',Times,serif;display:block;width:100%;max-height:555px;")

# Slide 8: CoT timeline – full width, cap height
cot_timeline = set_style(cot_timeline,
    "font-family:'Times New Roman',Times,serif;width:100%;display:block;max-height:252px;background:#fff;")

# Slide 8: Expert balance – 490px wide
expert_balance = set_style(expert_balance,
    "font-family:Helvetica,Arial,sans-serif;width:490px;display:block;flex-shrink:0;background:#fff;")

# Slide 9: Jacobi – full width, cap height
jacobi_svg = set_style(jacobi_svg,
    "width:100%;display:block;max-height:470px;font-family:'Inter',Helvetica,Arial,sans-serif;")

# ──────────────────────────────────────────────
# Inject into skeleton
# ──────────────────────────────────────────────

skeleton = (PRES / "index_skeleton.html").read_text(encoding="utf-8")

replacements = {
    "<!-- MACRO_ARCH_SVG -->":    macro_arch_svg,
    "<!-- TUCKER_P1_SVG -->":     tucker_p1,
    "<!-- TUCKER_P2_SVG -->":     tucker_p2,
    "<!-- COT_TIMELINE_SVG -->":  cot_timeline,
    "<!-- EXPERT_BALANCE_SVG -->": expert_balance,
    "<!-- JACOBI_SVG -->":        jacobi_svg,
}

result = skeleton
for placeholder, svg in replacements.items():
    if placeholder in result:
        result = result.replace(placeholder, svg, 1)
    else:
        print(f"WARNING: placeholder not found: {placeholder}")

out = PRES / "index.html"
out.write_text(result, encoding="utf-8")
print(f"Written {out}  ({len(result):,} bytes)")
