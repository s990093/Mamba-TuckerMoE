#!/usr/bin/env python3
"""Build a single-author version of the report PDF (Hung-Wei Lai only).

Replaces the frontmatter in memory without touching any source file,
then builds output/report_solo.pdf using the same pipeline as
generate_report_pdf.py.

Usage:
    python3 src/build_solo_pdf.py
    python3 src/build_solo_pdf.py -f   # force rebuild
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Solo frontmatter (in-memory replacement, does NOT modify 00_frontmatter.md) ──
SOLO_FRONTMATTER = r"""---
date: "2026年5月"
---

```{=latex}
\thispagestyle{empty}
\begin{center}

\vspace*{0.3cm}

{\fontsize{18}{22}\selectfont\bfseries
Breaking the Memory Wall:\\[0.2em]
Compute-Bound TuckerMoE for Hybrid State Space Models\par}

\vspace{0.4em}

{\large Mamba-3 $+$ Sparse GQA Transformer $+$ Cross-Expert TuckerMoE\par}
\vspace{0.1em}
{\normalsize Multi-Strategy Speculative Jacobi Decoding $\cdot$ On-Device MLX Inference\par}

\vspace{1.1em}

{\large Hung-Wei Lai\par}

\vspace{0.4em}

{\normalsize
Department of Computer Science and Information Engineering,\\
National Kaohsiung University of Science and Technology, Kaohsiung, Taiwan\par}

\vspace{0.25em}
{\small lai09150915@gmail.com\par}
\vspace{0.15em}
{\small github.com/s990093/Mamba-TuckerMoE\quad$\cdot$\quad s990093.github.io/Mamba-TuckerMoE\par}
\vspace{0.25em}
{\normalsize 2026\,年\,5\,月\par}

\vspace{0.7em}
\noindent\rule{0.72\linewidth}{0.5pt}\par
\vspace{0.4em}

{\small\textbf{Keywords:}\quad
State Space Models, Mixture-of-Experts, Tucker Decomposition,
Speculative Jacobi Decoding, On-Device Inference, Apple MLX, Metal Kernels\par}

\end{center}
\vspace{0.6em}
```
"""

# ── Reuse generate_report_pdf machinery ──────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import generate_report_pdf as grp   # noqa: E402  (import after sys.path fix)


_original_read = grp.read_report_source   # save BEFORE patching to avoid recursion


def read_report_source_solo() -> str:
    """Same as grp.read_report_source() but injects the solo frontmatter."""
    text = _original_read()

    import re
    # Match YAML front-matter + {=latex} block (the whole 00_frontmatter.md content)
    pattern = re.compile(
        r'^---\n.*?---\n```\{=latex\}.*?```\n',
        re.DOTALL,
    )
    replaced = pattern.sub(lambda _: SOLO_FRONTMATTER, text, count=1)
    if replaced == text:
        replaced = SOLO_FRONTMATTER + "\n" + text
    return replaced


def main() -> None:
    parser = argparse.ArgumentParser(description="Build single-author (Hung-Wei Lai) PDF.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force rebuild even if output is up to date.")
    args = parser.parse_args()

    output_solo = PROJECT_DIR / "output" / "report_solo.pdf"
    output_solo.parent.mkdir(parents=True, exist_ok=True)

    from rich.console import Console
    console = Console()
    console.rule("[bold cyan]Solo Author PDF Builder[/] — Hung-Wei Lai only", style="cyan")

    # Monkey-patch read_report_source to use solo version
    grp.read_report_source = read_report_source_solo

    grp.build_pdf(
        report_md=PROJECT_DIR / "report" / "report.md",
        output_pdf=output_solo,
        cjk_font="Songti SC",
        diagram_format="png",
        main_font="STIX Two Text",
        math_font="STIX Two Math",
        force=args.force,
        pdf_engine=None,   # auto-detect tectonic/xelatex
    )


if __name__ == "__main__":
    main()
