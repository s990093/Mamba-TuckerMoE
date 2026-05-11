from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
ASSET_DIR = PROJECT_DIR / "assets" / "algorithms"
SOURCE_DIR = PROJECT_DIR / "archive" / "latex-sources"
BUILD_DIR = PROJECT_DIR / "archive" / "latex-build" / "algorithms"
LOG_DIR = PROJECT_DIR / "archive" / "latex-logs" / "algorithms"

XELATEX_CMD = shutil.which("xelatex") or "/usr/local/texlive/2025/bin/universal-darwin/xelatex"
PDFTOPPM_CMD = shutil.which("pdftoppm")
SIPS_CMD = shutil.which("sips")

DOCUMENT_TEMPLATE = r"""
\documentclass[varwidth=190mm,border=8pt]{standalone}
\usepackage{amsmath,amssymb,mathtools,bm}
\usepackage[T1]{fontenc}
\usepackage[noend]{algpseudocode}
\algrenewcommand\algorithmicrequire{\textbf{Input:}}
\algrenewcommand\algorithmicensure{\textbf{Output:}}
\algrenewcommand\algorithmicrequire{\textbf{Require:}}
\algrenewcommand\algorithmicensure{\textbf{Ensure:}}
\algrenewcommand\algorithmicreturn{\textbf{return}}
\begin{document}
\begin{minipage}{180mm}
\small
%%BODY%%
\end{minipage}
\end{document}
""".strip()

ALGORITHMS = [
    {
        "stem": "appendix_a1_tuckermoe_forward",
        "body": r"""
\noindent\textbf{Procedure 1} TuckerMoE Forward Pass\par
\vspace{2pt}\hrule\vspace{4pt}
\begin{algorithmic}[1]
\Require $x \in \mathbb{R}^{B\times L\times d_{\mathrm{in}}}$, router $W_r \in \mathbb{R}^{d_{\mathrm{in}}\times E}$
\Statex \hspace{\algorithmicindent} $U_{\mathrm{in}} \in \mathbb{R}^{d_{\mathrm{in}}\times r_3}$, $U_{\mathrm{exp}} \in \mathbb{R}^{E\times r_1}$
\Statex \hspace{\algorithmicindent} $\mathcal{G} \in \mathbb{R}^{r_1\times r_3\times r_2}$, $U_{\mathrm{out}} \in \mathbb{R}^{r_2\times d_{\mathrm{out}}}$, $b \in \mathbb{R}^{d_{\mathrm{out}}}$
\Statex \hspace{\algorithmicindent} temperature schedule $T(\mathrm{step})$
\Ensure $y \in \mathbb{R}^{B\times L\times d_{\mathrm{out}}}$, $\mathcal{L}_{\mathrm{LB}}$, $\mathcal{L}_{\mathrm{Z}}$
\State $R \gets x W_r$
\State $\widehat{R} \gets \operatorname{fast\_scaled\_tanh}(R, 10.0)$
\State $\mathcal{L}_{\mathrm{Z}} \gets \operatorname{mean}(\operatorname{logsumexp}(\widehat{R}, -1)^2)$
\State $\widetilde{R} \gets \widehat{R} / T(\mathrm{step})$
\State $I \gets \operatorname{TopKIndices}(\widetilde{R}, k)$
\State $P \gets \operatorname{Softmax}(\operatorname{Gather}(\widetilde{R}, I), -1)$
\State $\mathcal{L}_{\mathrm{LB}} \gets E \sum_{e=1}^{E} \bar{m}_e \bar{p}_e$
\State $G_{\mathrm{exp}} \gets \operatorname{einsum}(\texttt{'er,rst->est'}, U_{\mathrm{exp}}, \mathcal{G})$
\State $X_{\mathrm{shared}} \gets \operatorname{RMSNorm}(x U_{\mathrm{in}})$
\State $X_{\mathrm{core}} \gets \operatorname{FusedLatentMoE}(X_{\mathrm{shared}}, G_{\mathrm{exp}}, I, P)$
\State $y \gets X_{\mathrm{core}} U_{\mathrm{out}} + b$
\State \Return $y, \mathcal{L}_{\mathrm{LB}}, \mathcal{L}_{\mathrm{Z}}$
\end{algorithmic}
\vspace{4pt}\hrule
""".strip(),
    },
    {
        "stem": "appendix_a2_chunk_parallel_scan",
        "body": r"""
\noindent\textbf{Procedure 2} Chunk-Parallel Scan (SSD)\par
\vspace{2pt}\hrule\vspace{4pt}
\begin{algorithmic}[1]
\Require $x \in \mathbb{R}^{B\times L\times H\times P}$, $\Delta \in \mathbb{R}^{B\times L\times H}$, $A \in \mathbb{R}^{H\times N}$
\Ensure $y \in \mathbb{R}^{B\times L\times H\times P}$
\Statex \textbf{Parameters:} chunk size $C = 64$
\State $M \gets \lceil L / C \rceil$; partition $(x, \Delta)$ into chunks $\{\mathcal{C}_m\}_{m=1}^{M}$
\ForAll{$m \in \{1, \dots, M\}$ in parallel}
  \State $\bar{A}^{(m)}_t \gets \exp(\Delta_t A)$ for all $t \in \mathcal{C}_m$
  \State $S^{(m)} \gets \operatorname{AssociativeScan}(\bar{A}^{(m)}, x^{(m)})$
  \State $(\alpha^{(m)}, \beta^{(m)}) \gets \operatorname{ChunkSummary}(S^{(m)})$
\EndFor
\State $s^{(1)}_{\mathrm{in}} \gets 0$
\For{$m \gets 1$ to $M$}
  \If{$m > 1$}
    \State $s^{(m)}_{\mathrm{in}} \gets \alpha^{(m-1)} s^{(m-1)}_{\mathrm{in}} + \beta^{(m-1)}$
  \EndIf
  \State $y^{(m)} \gets \operatorname{ApplyBoundaryState}(S^{(m)}, s^{(m)}_{\mathrm{in}})$
\EndFor
\State \Return $\operatorname{Concat}(y^{(1)}, \dots, y^{(M)})$
\end{algorithmic}
\vspace{4pt}\hrule
""".strip(),
    },
    {
        "stem": "appendix_a3_router_temperature_annealing",
        "body": r"""
\noindent\textbf{Procedure 3} Router Temperature Annealing\par
\vspace{2pt}\hrule\vspace{4pt}
\begin{algorithmic}[1]
\Require current step $s$, total steps $S_{\max}$, warmup steps $W$, $T_{\mathrm{start}}$, $T_{\mathrm{end}}$
\Ensure $T(s)$
\If{$s < W$}
  \State \Return $T_{\mathrm{start}}$
\Else
  \State $p \gets \dfrac{s - W}{S_{\max} - W}$
  \State $T(s) \gets T_{\mathrm{end}} + \dfrac{1}{2}(T_{\mathrm{start}} - T_{\mathrm{end}})\bigl(1 + \cos(\pi p)\bigr)$
  \State \Return $T(s)$
\EndIf
\end{algorithmic}
\vspace{4pt}\hrule
""".strip(),
    },
    {
        "stem": "appendix_a4_tuckermoe_backward",
        "body": r"""
\noindent\textbf{Procedure 4} TuckerMoE Backward Pass\par
\vspace{2pt}\hrule\vspace{4pt}
\begin{algorithmic}[1]
\Require upstream gradient $\delta = \partial \mathcal{L}/\partial y$, $X_{\mathrm{shared}}$, $G_{\mathrm{exp}}$, selected indices $I$, routing weights $P$
\Statex \hspace{\algorithmicindent} shared factors $U_{\mathrm{in}}, U_{\mathrm{exp}}, \mathcal{G}, U_{\mathrm{out}}$
\Ensure $\partial \mathcal{L}/\partial x$, $\partial \mathcal{L}/\partial U_{\mathrm{in}}$, $\partial \mathcal{L}/\partial U_{\mathrm{exp}}$, $\partial \mathcal{L}/\partial \mathcal{G}$, $\partial \mathcal{L}/\partial U_{\mathrm{out}}$, $\partial \mathcal{L}/\partial P$
\State $\partial \mathcal{L}/\partial X_{\mathrm{shared}} \gets 0$, $\partial \mathcal{L}/\partial G_{\mathrm{exp}} \gets 0$, $\partial \mathcal{L}/\partial U_{\mathrm{out}} \gets 0$
\For{$k \gets 1$ to top-$k$}
  \State $e \gets I[:, k]$
  \State $\Delta_k \gets P[:, k] \odot \delta$
  \State $\partial \mathcal{L}/\partial X_{\mathrm{shared}} \gets \partial \mathcal{L}/\partial X_{\mathrm{shared}} + \Delta_k G_{\mathrm{exp}}[e]^\top$
  \State $\partial \mathcal{L}/\partial P[:, k] \gets \langle \delta, X_{\mathrm{shared}} G_{\mathrm{exp}}[e] U_{\mathrm{out}} \rangle$
  \State $\partial \mathcal{L}/\partial G_{\mathrm{exp}}[e] \gets \partial \mathcal{L}/\partial G_{\mathrm{exp}}[e] + X_{\mathrm{shared}}^\top(\Delta_k U_{\mathrm{out}}^\top)$
  \State $\partial \mathcal{L}/\partial U_{\mathrm{out}} \gets \partial \mathcal{L}/\partial U_{\mathrm{out}} + (X_{\mathrm{shared}} G_{\mathrm{exp}}[e])^\top \Delta_k$
\EndFor
\State $(\partial \mathcal{L}/\partial U_{\mathrm{exp}},\, \partial \mathcal{L}/\partial \mathcal{G}) \gets \operatorname{Mode1Backward}(\partial \mathcal{L}/\partial G_{\mathrm{exp}}, U_{\mathrm{exp}}, \mathcal{G})$
\State $(\partial \mathcal{L}/\partial x,\, \partial \mathcal{L}/\partial U_{\mathrm{in}}) \gets \operatorname{BackpropSharedProjection}(X_{\mathrm{shared}}, \partial \mathcal{L}/\partial X_{\mathrm{shared}})$
\State \Return all gradients
\end{algorithmic}
\vspace{4pt}\hrule
""".strip(),
    },
]


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def render_png(pdf_path: Path, png_path: Path) -> None:
    if PDFTOPPM_CMD:
        run([PDFTOPPM_CMD, "-png", "-singlefile", str(pdf_path), str(png_path.with_suffix(""))], PROJECT_DIR)
        return
    if SIPS_CMD:
        run([SIPS_CMD, "-s", "format", "png", str(pdf_path), "--out", str(png_path)], PROJECT_DIR)
        return
    raise RuntimeError("Neither pdftoppm nor sips is available for PDF-to-PNG conversion.")


def main() -> None:
    for path in (ASSET_DIR, SOURCE_DIR, BUILD_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)

    for spec in ALGORITHMS:
        stem = spec["stem"]
        tex_path = SOURCE_DIR / f"{stem}.tex"
        build_pdf_path = BUILD_DIR / f"{stem}.pdf"
        asset_pdf_path = ASSET_DIR / f"{stem}.pdf"
        asset_png_path = ASSET_DIR / f"{stem}.png"
        log_path = LOG_DIR / f"{stem}.log"
        aux_path = LOG_DIR / f"{stem}.aux"

        tex_path.write_text(DOCUMENT_TEMPLATE.replace("%%BODY%%", spec["body"]) + "\n", encoding="utf-8")

        run(
            [
                XELATEX_CMD,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(BUILD_DIR),
                str(tex_path),
            ],
            PROJECT_DIR,
        )

        if not build_pdf_path.exists():
            raise FileNotFoundError(f"Missing compiled PDF: {build_pdf_path}")

        shutil.copy2(build_pdf_path, asset_pdf_path)
        render_png(asset_pdf_path, asset_png_path)

        compiled_log_path = BUILD_DIR / f"{stem}.log"
        compiled_aux_path = BUILD_DIR / f"{stem}.aux"
        if compiled_log_path.exists():
            shutil.copy2(compiled_log_path, log_path)
        if compiled_aux_path.exists():
            shutil.copy2(compiled_aux_path, aux_path)

        print(f"Rendered {asset_pdf_path.relative_to(PROJECT_DIR)} and {asset_png_path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
