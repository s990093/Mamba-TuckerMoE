#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import os
import shutil
import socketserver
import subprocess
import threading
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TaskProgressColumn,
    )
    from rich.status import Status
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallbacks if rich not installed
    class Console:
        def print(self, *args, **kwargs): print(*args)
        def status(self, msg): 
            print(msg)
            class _Dummy:
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return _Dummy()
    # Minimal progress stub
    class Progress:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def add_task(self, *a, **k): return 0
        def update(self, *a, **k): pass
    Status = Console.status
    Table = None
    Panel = None

console = Console() if RICH_AVAILABLE else Console()


PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORT_MD = PROJECT_DIR / "report/report.md"   # legacy fallback only
SECTIONS_DIR = PROJECT_DIR / "report/sections"  # canonical source (split per subsection)
OUTPUT_PDF = PROJECT_DIR / "output/report.pdf"


def _get_section_sources() -> list[Path]:
    if SECTIONS_DIR.is_dir():
        return sorted(SECTIONS_DIR.glob("*.md"))
    return [REPORT_MD]


def _is_up_to_date(output: Path, sources: Iterable[Path], extras: Iterable[Path] = ()) -> bool:
    """Return True if output exists and is newer than all given source files."""
    if not output.exists():
        return False
    try:
        out_m = output.stat().st_mtime
        for p in list(sources) + list(extras):
            if p.exists() and p.stat().st_mtime > out_m:
                return False
        return True
    except OSError:
        return False



def read_report_source() -> str:
    """Assemble the report markdown from fine-grained section files.

    Preferred layout (for context-friendly editing):
        report/sections/
          00_frontmatter.md
          02_摘要.md
          04_01.md
          05_2.md
          08_5_01.md   # chapter 5, subsection 5.1 (## + ### 5.1)
          08_5_02.md   # 5.2
          08_5_03.md   # 5.3
          08_5_04.md   # 5.4 (or further split 08_5_04_01.md etc.)
          ...
    Files are concatenated by lexical sort of their filenames. This guarantees
    deterministic order and makes the full assembled text identical to the old
    monolithic report.md (required for the verbatim MD→LaTeX block replaces below).

    You can split as finely as you want ("一截一截") as long as sort order is correct.
    Falls back to the legacy single report/report.md only if sections/ has no .md files.

    File reads are done in parallel for a small speedup on many/small files.
    """
    if SECTIONS_DIR.is_dir():
        files = sorted(SECTIONS_DIR.glob("*.md"))
        if files:
            if RICH_AVAILABLE:
                console.print(f"[bold cyan]Assembling report from {len(files)} section files...[/]")

                # Show file list as a nice Rich Table (as requested)
                table = Table(title="Sections", show_header=True, header_style="bold magenta", box=None, padding=(0,1))
                table.add_column("#", style="dim", width=3)
                table.add_column("Filename", style="cyan")
                # Show in a compact way (3 columns would require Columns, use simple rows or limit display)
                for i, f in enumerate(files, 1):
                    table.add_row(str(i), f.name)
                console.print(table)
            else:
                console.print(f"[generate_report_pdf] Assembling from {len(files)} section files:")
                for f in files:
                    console.print(f"  - {f.name}")

            def _read(p: Path) -> str:
                return p.read_text(encoding="utf-8")

            # Parallel I/O (progress is handled at overall build level)
            contents = [None] * len(files)

            def _read_with_idx(idx: int, p: Path) -> str:
                return p.read_text(encoding="utf-8")

            with ThreadPoolExecutor(max_workers=min(8, len(files))) as ex:
                futures = {ex.submit(_read_with_idx, i, f): i for i, f in enumerate(files)}
                for fut in as_completed(futures):
                    i = futures[fut]
                    contents[i] = fut.result()

            return "".join(contents)
    console.print("[generate_report_pdf] Falling back to monolithic report/report.md")
    return REPORT_MD.read_text(encoding="utf-8")

DIAGRAM_SPECS = (
    ("prototypes/method_flowchart.html", "assets/method_flowchart.svg"),
    ("prototypes/architecture.html", "assets/images/architecture.svg"),
    ("prototypes/causal_mask.html", "assets/plots/causal_mask.svg"),
    ("prototypes/causal_mask_visualization_1.html", "assets/plots/causal_mask_visualization_1.svg"),
    ("prototypes/sft_loss_mask.html", "assets/plots/sft_loss_mask.svg"),
    ("prototypes/sft_mask_verification.html", "assets/plots/sft_mask_verification.svg"),
    ("prototypes/chatml_template.html", "assets/plots/chatml_template.svg"),
)

METHOD_PIPELINE_MD = "![Method Pipeline](../assets/method_flowchart.svg)"
METHOD_PIPELINE_LATEX = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{../assets/method_flowchart.png}
\caption{Method Pipeline}
\end{figure}
""".strip()

MODEL_ARCH_MD = "![Model Architecture](../assets/images/architecture.svg)"
MODEL_ARCH_CAPTION_MD = (
    "圖 2：Hybrid Mamba-TuckerMoE 詳細模型架構。每個 Macro Block 包含 4 個 "
    "Mamba3Block 與 1 個 TransformerBlock，整體堆疊 $N_\\text{macro}=6$ 次。"
)
MODEL_ARCH_LATEX_TEMPLATE = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{../assets/images/architecture.png}
\caption{Hybrid Mamba-TuckerMoE 詳細模型架構。每個 Macro Block 包含 4 個 Mamba3Block 與 1 個 TransformerBlock，整體堆疊 $N_\text{macro}=6$ 次。}
\end{figure}
""".strip()

FIGURE_A_MD = "![圖 A：TuckerMoE 前向傳播與反向傳播梯度流完整示意](../assets/plots/tuckermoe_forward_backward.png)"
FIGURE_A_LATEX = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{../assets/plots/tuckermoe_forward_backward.png}
\caption{圖 A：TuckerMoE 前向傳播與反向傳播梯度流完整示意}
\end{figure}
""".strip()

FIGURE_B_MD = "![圖 B：Token × Expert 稀疏門控矩陣（九宮格視角）](../assets/plots/tucker_sparse_matrix.png)"
FIGURE_B_LATEX = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{../assets/plots/tucker_sparse_matrix.png}
\caption{圖 B：Token $\times$ Expert 稀疏門控矩陣（九宮格視角）}
\end{figure}
""".strip()

# 必須與 sections/ 組裝後的「圖 1」區塊（在 frontmatter 或對應檔）逐字一致，
# 否則 replace 不生效 → pandoc 浮體跑版（僅剩圖說、圖不見）。
PARETO_FIGURE_BLOCK_MD = """![Pareto Frontier: 推論成本 vs. 有效模型容量](../assets/plots/pareto_frontier.png)

_圖 1：本文研究貢獻的視覺化總覽。推論成本（每 token active 參數量，M）vs. 模型有效容量（dense-equivalent 參數量，M）的 Pareto 前沿圖（雙對數座標）。灰色虛線為 Dense 對角線，對角線以上代表「以更低推論成本獲得更大容量」的 Pareto 優勢區。本模型（紫星）以 230M active 參數達到 2.4B dense-equivalent 容量，推論成本相較同容量 dense 基準降低約 **90%**。基準來源：Mamba (Gu & Dao, 2023)、Mamba-2 (Dao & Gu, 2024)、Pythia (Biderman et al., 2023)、Mixtral (Jiang et al., 2024)、Switch (Fedus et al., 2022)。_"""
PARETO_FIGURE_LATEX = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth,height=0.46\textheight,keepaspectratio]{../assets/plots/pareto_frontier.png}
\caption{圖 1：本文研究貢獻的視覺化總覽。推論成本（每 token active 參數量，M）vs.~模型有效容量（dense-equivalent 參數量，M）的 Pareto 前沿圖（雙對數座標）。灰色虛線為 Dense 對角線，對角線以上代表「以更低推論成本獲得更大容量」的 Pareto 優勢區。本模型（紫星）以 230M active 參數達到 2.4B dense-equivalent 容量，推論成本相較同容量 dense 基準降低約 \textbf{90\%}。基準來源：Mamba (Gu \& Dao, 2023)、Mamba-2 (Dao \& Gu, 2024)、Pythia (Biderman et al., 2023)、Mixtral (Jiang et al., 2024)、Switch (Fedus et al., 2022)。}
\end{figure}
""".strip()

FIGURE_8A_BLOCK_MD = """![Mamba AI Assistant — App 介面預覽](../assets/uiux/iphone_preview.png)

_圖 8a：Mamba AI 助手 iOS 原型介面，呈現待機、處理中、回應三種操作狀態。_"""
FIGURE_8A_LATEX = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{../assets/uiux/iphone_preview.png}
\caption{圖 8a：Mamba AI 助手 iOS 原型介面，呈現待機、處理中、回應三種操作狀態。}
\end{figure}
""".strip()

FIGURE_8B_BLOCK_MD = """![端到端系統資料流圖](../assets/uiux/audio_flow.png)

_圖 8b：端到端系統資料流。使用者語音輸入經 Apple Neural Engine 前處理後，由 MLX 後端驅動的 Hybrid Mamba-TuckerMoE 模型進行推論，輸出分兩路：語音合成（TTS）與 UI 狀態更新。_"""
FIGURE_8B_LATEX = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{../assets/uiux/audio_flow.png}
\caption{圖 8b：端到端系統資料流。使用者語音輸入經 Apple Neural Engine 前處理後，由 MLX 後端驅動的 Hybrid Mamba-TuckerMoE 模型進行推論，輸出分兩路：語音合成（TTS）與 UI 狀態更新。}
\end{figure}
""".strip()

FIGURE_8C_BLOCK_MD = """![實體情境模擬圖（一）](../assets/uiux/user_story1.png)

_圖 8c：使用情境模擬（一）。iPhone 16 Pro 搭配 MagSafe 旋轉支架，轉化為桌面免持 AI 助手，呈現日常辦公場景下的即時語音問答互動。_"""
FIGURE_8C_LATEX = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{../assets/uiux/user_story1.png}
\caption{圖 8c：使用情境模擬（一）。iPhone 16 Pro 搭配 MagSafe 旋轉支架，轉化為桌面免持 AI 助手，呈現日常辦公場景下的即時語音問答互動。}
\end{figure}
""".strip()

FIGURE_8D_BLOCK_MD = """![實體情境模擬圖（二）](../assets/uiux/user_story2.png)

_圖 8d：使用情境模擬（二）。展示不同場景下的裝置擺放與互動方式，說明本應用在有限硬體資源下支援全天候語音查詢的可行性。_"""
FIGURE_8D_LATEX = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{../assets/uiux/user_story2.png}
\caption{圖 8d：使用情境模擬（二）。展示不同場景下的裝置擺放與互動方式，說明本應用在有限硬體資源下支援全天候語音查詢的可行性。}
\end{figure}
""".strip()

CAUSAL_MASK_MD = "![圖 C1：Causal Attention Mask 矩陣（8-token 示例，下三角結構）](../assets/plots/causal_mask.png)"
CAUSAL_MASK_LATEX = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{../assets/plots/causal_mask.png}
\caption{圖 C1：Causal Attention Mask 矩陣（8-token 示例，下三角結構）}
\end{figure}
""".strip()

# 與 sections/ 組裝後的 §7.2.5 Markdown 區塊逐字一致；否則 replace 失效致浮體跑版。
# 因為圖片區塊現在可能散落在 10_7_02.md 等細檔內，組裝後字串仍需完全吻合。
CAUSAL_MASK_VISUAL_MD = """![圖 SFT-CM：Causal attention mask 與 SFT loss mask 對照（多輪 ChatML 截取示例）](../assets/plots/causal_mask_visualization_1.png)

_圖 SFT-CM：教學用意之**多輪 ChatML token 截取**（上為 position／token／U–A block 對齊）；**左**為對 queries-keys 之下三角 **Causal attention mask**（白／灰：可 attend／遮蔽）；**右**為 **SFT loss mask** 之區段級約束（綠區對 CE 有效、藍區不計 loss 仍可作前文條件）；與 §7.2.3、§7.2.6 之角色語法與終止符約定同構對讀。_
"""

# 勿同時設定 width + height（keepaspectratio 會取縮放的 min，橫長圖常被「高度上限」限死而寬度遠小於 \linewidth）。
# 此圖以橫向資訊為主：強制鋪滿版心寬度，視覺上占滿欄。
CAUSAL_MASK_VISUAL_LATEX = r"""
\begin{figure}[H]
\centering
\resizebox{\linewidth}{!}{\includegraphics{../assets/plots/causal_mask_visualization_1.png}}
\caption{圖 SFT-CM：教學用意之多輪 ChatML token 截取（上為 position/token 與 U--A block 對齊）；左為 queries--keys 之下三角 \textbf{Causal attention mask}（白/灰：可 attend/遮蔽）；右為 \textbf{SFT loss mask} 之區段約束（綠區對 CE、藍區不計仍可作前文條件）；與 \S\,7.2.3、\S\,7.2.6 之角色語法與終止符約定同構對讀。}
\end{figure}
""".strip()

SFT_LOSS_MASK_MD = "![圖 C2：SFT Token 級別 Loss Mask 逐 token 可視化](../assets/plots/sft_loss_mask.png)"
SFT_LOSS_MASK_LATEX = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{../assets/plots/sft_loss_mask.png}
\caption{圖 C2：SFT Token 級別 Loss Mask 逐 token 可視化}
\end{figure}
""".strip()

SFT_VERIFICATION_MD = "![圖 SFT-V 邊界細節：多輪對話邊界驗證與 SFT 標籤對齊](../assets/plots/sft_mask_verification.png)"
SFT_VERIFICATION_LATEX = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{../assets/plots/sft_mask_verification.png}
\caption{圖 SFT-V 邊界細節：多輪對話邊界驗證與 SFT 標籤對齊}
\end{figure}
""".strip()

CHATML_TEMPLATE_MD = "![圖 C3：ChatML 對話模板與推論前綴可視化](../assets/plots/chatml_template.png)"
CHATML_TEMPLATE_LATEX = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{../assets/plots/chatml_template.png}
\caption{圖 C3：ChatML 對話模板與推論前綴可視化}
\end{figure}
""".strip()

# §9.8：四宮格曲線圖尺寸大；pandoc 預設 figure 無 [H] 且中文圖題在環境外，易浮走或擠版。
SFT_TRAIN_VAL_MD = """![SFT training and validation logs](../assets/plots/sft_train_val_plots.png)

_圖 8：目前 SFT 訓練與驗證曲線。左上為訓練 loss 與 CE loss 的 raw / MA-20 平滑曲線；右上為 validation CE loss 與 validation mean loss；左下為 learning rate 與 gradient norm；右下為 step time 與 router temperature。_"""
SFT_TRAIN_VAL_LATEX = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth,height=0.72\textheight,keepaspectratio]{../assets/plots/sft_train_val_plots.png}
\caption{圖 8：目前 SFT 訓練與驗證曲線。左上為訓練 loss 與 CE loss 的 raw / MA-20 平滑曲線；右上為 validation CE loss 與 validation mean loss；左下為 learning rate 與 gradient norm；右下為 step time 與 router temperature。}
\end{figure}
""".strip()

APPENDIX_ALGO_LATEX = {
    "../assets/algorithms/appendix_a1_tuckermoe_forward.png": r"""
\begin{algorithm}[H]
\caption{TuckerMoE Forward Pass}
\begin{algorithmic}[1]
\Procedure{TuckerMoEForward}{$x, W_r, U_{\mathrm{in}}, U_{\mathrm{exp}}, \mathcal{G}, U_{\mathrm{out}}, b, T(\mathrm{step})$}
\Require $x \in \mathbb{R}^{B\times L\times d_{\mathrm{in}}}$, $W_r \in \mathbb{R}^{d_{\mathrm{in}}\times E}$
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
\EndProcedure
\end{algorithmic}
\end{algorithm}
""".strip(),
    "../assets/algorithms/appendix_a2_chunk_parallel_scan.png": r"""
\begin{algorithm}[H]
\caption{Chunk-Parallel Scan (SSD)}
\begin{algorithmic}[1]
\Procedure{ChunkParallelScan}{$x, \Delta, A$}
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
\EndProcedure
\end{algorithmic}
\end{algorithm}
""".strip(),
    "../assets/algorithms/appendix_a3_router_temperature_annealing.png": r"""
\begin{algorithm}[H]
\caption{Router Temperature Annealing}
\begin{algorithmic}[1]
\Procedure{RouterTemperatureAnnealing}{$s, S_{\max}, W, T_{\mathrm{start}}, T_{\mathrm{end}}$}
\Require current step $s$, total steps $S_{\max}$, warmup steps $W$, $T_{\mathrm{start}}$, $T_{\mathrm{end}}$
\Ensure $T(s)$
\If{$s < W$}
  \State \Return $T_{\mathrm{start}}$
\Else
  \State $p \gets \dfrac{s - W}{S_{\max} - W}$
  \State $T(s) \gets T_{\mathrm{end}} + \dfrac{1}{2}(T_{\mathrm{start}} - T_{\mathrm{end}})\bigl(1 + \cos(\pi p)\bigr)$
  \State \Return $T(s)$
\EndIf
\EndProcedure
\end{algorithmic}
\end{algorithm}
""".strip(),
    "../assets/algorithms/appendix_a4_tuckermoe_backward.png": r"""
\begin{algorithm}[H]
\caption{TuckerMoE Backward Pass}
\begin{algorithmic}[1]
\Procedure{TuckerMoEBackward}{$\delta, X_{\mathrm{shared}}, G_{\mathrm{exp}}, I, P, U_{\mathrm{in}}, U_{\mathrm{exp}}, \mathcal{G}, U_{\mathrm{out}}$}
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
\EndProcedure
\end{algorithmic}
\end{algorithm}
""".strip(),
}

# STIX Two Text 缺少数 Unicode 字形（實測 ≈、↓）時 xelatex 會警告；改以行內數學輸出。
# 注意：勿對「×」等做全域替換，否則「3×3」會變成「3$\times$3」，在鄰近 $...$ 時會破壞數學邊界。
# ★ (U+2605) 與 † (U+2020)：xelatex 若找不到字形會暫停等待輸入 → 直接導致 90s 超時，必須替換。
UNICODE_TEXT_MATH_REPLACEMENTS = {
    "→": r"$\to$",
    "←": r"$\leftarrow$",
    "↔": r"$\leftrightarrow$",
    "≈": r"$\approx$",
    "↓": r"$\downarrow$",
    "★": r"$\bigstar$",          # U+2605 BLACK STAR — 在 SJD 加速比表格中標示外推值
    "†": r"$\dagger$",           # U+2020 DAGGER — 在 SJD 表格中標示直接量測值
    "⁻": r"${}^{-}$",           # U+207B SUPERSCRIPT MINUS — lualatex/Songti SC 缺字
    "⁺": r"${}^{+}$",           # U+207A SUPERSCRIPT PLUS (預防性)
    # STIX Two Text lacks the circled digits used in the dLLM section tables.
    "①": "(1)",
    "②": "(2)",
    "③": "(3)",
    "④": "(4)",
    "⑤": "(5)",
}


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def sanitize_svg_markup(svg_markup: str) -> str:
    # XML parser used in pandoc/LaTeX SVG path does not accept HTML named entities.
    # Replace common HTML-only entities with XML-safe forms.
    return svg_markup.replace("&nbsp;", "&#160;")


def embed_runtime_svg_styles(svg_markup: str, runtime_css: str) -> str:
    if not runtime_css.strip():
        return svg_markup
    style_block = f"<style>{runtime_css}</style>"
    insert_at = svg_markup.find(">")
    if insert_at == -1:
        return svg_markup
    return svg_markup[: insert_at + 1] + style_block + svg_markup[insert_at + 1 :]


def render_svg_from_html(
    host: str = "127.0.0.1",
    port: int = 8765,
    timeout_ms: int = 45000,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: playwright. Install with `python3 -m pip install playwright` "
            "then run `python3 -m playwright install chromium`."
        ) from exc

    handler = http.server.SimpleHTTPRequestHandler
    old_cwd = os.getcwd()
    os.chdir(PROJECT_DIR)
    server = ReusableTCPServer((host, port), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # IMPORTANT: Each diagram is rendered in its own thread with its OWN browser.
        # Sharing a single browser/context across threads causes greenlet "cannot switch thread"
        # errors because Playwright sync API is not thread-safe for page/context objects.
        def _render_one(spec):
            html_rel_path, svg_rel_path = spec
            html_path = PROJECT_DIR / html_rel_path
            svg_out = PROJECT_DIR / svg_rel_path
            png_out = svg_out.with_suffix(".png")
            pdf_out = svg_out.with_suffix(".pdf")

            targets = [svg_out, png_out, pdf_out]
            if all(t.exists() and t.stat().st_mtime >= html_path.stat().st_mtime for t in targets if t.exists()):
                if RICH_AVAILABLE:
                    console.print(f"[dim]✓ diagram up-to-date: {svg_rel_path}[/dim]")
                else:
                    print(f"[render] up-to-date: {svg_rel_path}")
                return

            # Fully isolated browser per thread → safe with ThreadPoolExecutor
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={"width": 2400, "height": 1600}, device_scale_factor=2
                )
                page = context.new_page()
                try:
                    url = f"http://{host}:{port}/{html_rel_path}"
                    page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                    page.wait_for_timeout(1500)
                    page.evaluate(
                        """
                        () => {
                          if (window.MathJax && window.MathJax.typesetPromise) {
                            return window.MathJax.typesetPromise();
                          }
                        }
                        """
                    )

                    has_svg = page.evaluate("() => document.querySelector('svg') !== null && !document.querySelector('.page')")
                    output_path = svg_out
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    if has_svg:
                        svg_markup = page.eval_on_selector("svg", "el => el.outerHTML")
                        svg_markup = sanitize_svg_markup(svg_markup)
                        mjx_css = page.evaluate(
                            """
                            () => {
                              const el = document.querySelector('style#MJX-SVG-styles');
                              return el ? el.textContent : '';
                            }
                            """
                        )
                        shared_css = """
                        .math-box{
                          display:flex;
                          justify-content:center;
                          align-items:center;
                          width:100%;
                          height:100%;
                          white-space:nowrap;
                        }
                        """
                        svg_markup = embed_runtime_svg_styles(svg_markup, f"{mjx_css}\n{shared_css}")
                        output_path.write_text(svg_markup + "\n", encoding="utf-8")
                        if RICH_AVAILABLE:
                            console.print(f"[green]Rendered SVG[/]: {output_path.relative_to(PROJECT_DIR)}")
                        else:
                            print(f"Rendered SVG: {output_path.relative_to(PROJECT_DIR)}")

                    # High-res PNG
                    png_path = output_path.with_suffix(".png")
                    if has_svg:
                        target_el = page.locator("svg").first
                    elif page.locator(".page").count() > 0:
                        target_el = page.locator(".page").first
                    else:
                        target_el = page.locator("body").first
                    target_el.screenshot(path=str(png_path), scale="device")
                    if RICH_AVAILABLE:
                        console.print(f"[green]Rendered PNG[/]: {png_path.relative_to(PROJECT_DIR)}")
                    else:
                        print(f"Rendered PNG: {png_path.relative_to(PROJECT_DIR)}")

                    # Vector PDF
                    size = page.evaluate(
                        """
                        () => {
                          const isMainSvg = document.querySelector("svg") !== null && !document.querySelector(".page");
                          if (isMainSvg) {
                            const svg = document.querySelector("svg");
                            const vb = svg.viewBox && svg.viewBox.baseVal;
                            if (vb && vb.width > 0 && vb.height > 0) {
                              return { w: vb.width, h: vb.height };
                            }
                          }
                          const el = document.querySelector(".page") || document.body;
                          const r = el.getBoundingClientRect();
                          return { w: Math.max(1, r.width), h: Math.max(1, r.height) };
                        }
                        """
                    )
                    w = float(size["w"])
                    h = float(size["h"])
                    if has_svg:
                        page.add_style_tag(
                            content=f"""
                            @page {{ margin: 0; }}
                            html, body {{
                                margin: 0 !important;
                                padding: 0 !important;
                                background: #fff !important;
                            }}
                            body {{
                                display: block !important;
                            }}
                            .diagram-wrap {{
                                margin: 0 !important;
                                padding: 0 !important;
                                max-width: none !important;
                            }}
                            svg {{
                                width: {w}px !important;
                                min-width: 0 !important;
                                height: {h}px !important;
                                display: block !important;
                            }}
                            """
                        )
                    else:
                        page.add_style_tag(
                            content=f"""
                            @page {{ margin: 0; size: {w}px {h}px; }}
                            html, body {{
                                margin: 0 !important;
                                padding: 0 !important;
                                background: #f9f8f5 !important;
                                width: {w}px !important;
                                height: {h}px !important;
                                overflow: hidden !important;
                            }}
                            .page {{
                                margin: 0 auto !important;
                                padding: 32px !important;
                            }}
                            """
                        )
                    page.wait_for_timeout(1500)
                    pdf_path = output_path.with_suffix(".pdf")
                    page.pdf(
                        path=str(pdf_path),
                        width=f"{w}px",
                        height=f"{h}px",
                        print_background=True,
                        scale=1.0,
                        page_ranges="1",
                    )
                    if RICH_AVAILABLE:
                        console.print(f"[green]Rendered vector PDF[/]: {pdf_path.relative_to(PROJECT_DIR)}")
                    else:
                        print(f"Rendered vector PDF: {pdf_path.relative_to(PROJECT_DIR)}")
                finally:
                    page.close()
                    context.close()
                    browser.close()

        # Parallel rendering — each thread has its own full browser (thread-safe)
        max_workers = min(4, len(DIAGRAM_SPECS))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_render_one, spec): spec for spec in DIAGRAM_SPECS}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    if RICH_AVAILABLE:
                        console.print(f"[red]Error rendering {spec}: {exc}[/red]")
                    else:
                        print(f"[render] Error on {spec}: {exc}")
    finally:
        server.shutdown()
        server.server_close()
        os.chdir(old_cwd)


_QUICK_CSS = """
/* ── Quick-preview CSS ── */
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&family=Noto+Sans+SC&display=swap');
body {
  font-family: "Noto Serif SC", "Songti SC", "SimSun", serif;
  font-size: 11pt;
  line-height: 1.7;
  max-width: 820px;
  margin: 36px auto;
  padding: 0 24px;
  color: #111;
}
h1,h2,h3,h4 { font-weight: bold; margin-top: 1.4em; }
h1 { font-size: 1.8em; border-bottom: 2px solid #333; padding-bottom: .3em; }
h2 { font-size: 1.35em; border-bottom: 1px solid #ccc; }
h3 { font-size: 1.1em; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: .93em; }
th, td { border: 1px solid #bbb; padding: 5px 9px; }
th { background: #f0f0f0; }
code { font-family: "Courier New", monospace; font-size: .88em;
       background: #f5f5f5; padding: 1px 4px; border-radius: 3px; }
pre { background: #f5f5f5; padding: 12px; overflow-x: auto; font-size: .85em; }
img { max-width: 100%; }
blockquote { border-left: 3px solid #aaa; margin-left: 0; padding-left: 1em; color: #555; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
@media print {
  body { max-width: none; margin: 0; padding: 14mm 18mm; }
  h1,h2,h3 { page-break-after: avoid; }
  table,pre { page-break-inside: avoid; }
}
"""

_MATHJAX_SNIPPET = """
<script>
window.MathJax = {
  tex: { inlineMath: [['$','$']], displayMath: [['$$','$$']] },
  svg: { fontCache: 'global' }
};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
"""


def build_quick_pdf(
    report_md: Path,
    output_pdf: Path,
    force: bool = False,
) -> None:
    """Fast ≤10s preview PDF: pandoc → HTML (MathJax) → Playwright → PDF.

    Skips xelatex entirely — no CJK font loading, no LaTeX package pass.
    Raw {=latex} blocks are stripped (figures/algorithms won't appear);
    all text, math ($...$) and tables render correctly.
    Requires: pandoc (always needed) + playwright (pip install playwright
    && playwright install chromium).
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "Quick mode needs playwright: pip install playwright && playwright install chromium"
        )

    pandoc_cmd = shutil.which("pandoc")
    if not pandoc_cmd:
        raise RuntimeError("pandoc not found.")

    sources = _get_section_sources()
    if not force and _is_up_to_date(output_pdf, sources):
        if RICH_AVAILABLE:
            console.print("[bold green]✓ Quick PDF up to date[/] (--force to rebuild)")
        else:
            print("Quick PDF up to date.")
        return

    t_start = time.perf_counter()

    if RICH_AVAILABLE:
        console.rule("[bold cyan]Quick Preview Builder[/]", style="cyan")

    # ── 1. Assemble markdown ──────────────────────────────────────────────
    t0 = time.perf_counter()
    report_text = read_report_source()

    # Remove {=latex} raw blocks (pandoc HTML output ignores them, but we
    # strip them explicitly so the surrounding ``` fences don't show as code).
    import re
    report_text = re.sub(
        r"```\{=latex\}.*?```",
        "",
        report_text,
        flags=re.DOTALL,
    )
    # Keep unicode as-is for HTML (browsers handle ★ † → ≈ ↓ natively)
    # Only strip the safeguard replacement for ---
    _lines = report_text.split("\n")
    _dc = 0
    for _i, _ln in enumerate(_lines):
        if _ln.strip() == "---":
            _dc += 1
            if _dc > 2:
                _lines[_i] = ""          # remove extra HR in HTML (cosmetic)
    report_text = "\n".join(_lines)

    t1 = time.perf_counter()
    if RICH_AVAILABLE:
        console.print(f"[dim]Assemble + strip: {t1-t0:.2f}s[/]")

    # ── 2. pandoc → standalone HTML ───────────────────────────────────────
    report_dir = report_md.parent
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", encoding="utf-8",
        delete=False, dir=str(report_dir),
    ) as f:
        f.write(report_text)
        tmp_md = Path(f.name)

    tmp_html = tmp_md.with_suffix(".html")

    # Inject our CSS + MathJax into a temp header file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", encoding="utf-8", delete=False,
    ) as hf:
        hf.write(f"<style>{_QUICK_CSS}</style>\n{_MATHJAX_SNIPPET}\n")
        tmp_header = Path(hf.name)

    pandoc_html_cmd = [
        pandoc_cmd,
        str(tmp_md),
        "--from", "markdown+tex_math_dollars",
        "--to", "html5",
        "--standalone",
        "--include-in-header", str(tmp_header),
        "--output", str(tmp_html),
    ]

    t2 = time.perf_counter()
    subprocess.run(pandoc_html_cmd, cwd=str(report_dir), check=True,
                   capture_output=True)
    t3 = time.perf_counter()
    if RICH_AVAILABLE:
        console.print(f"[dim]pandoc → HTML: {t3-t2:.2f}s[/]")

    # ── 3. Playwright: HTML → PDF ─────────────────────────────────────────
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(tmp_html.as_uri(), wait_until="domcontentloaded")
        # Wait for MathJax to finish rendering
        try:
            page.wait_for_function(
                "() => window.MathJax && window.MathJax.startup && "
                "window.MathJax.startup.promise",
                timeout=6000,
            )
            page.evaluate("() => window.MathJax.startup.promise")
        except Exception:
            page.wait_for_timeout(2500)
        page.pdf(
            path=str(output_pdf),
            format="A4",
            print_background=True,
            margin={"top": "18mm", "bottom": "18mm",
                    "left": "18mm", "right": "18mm"},
        )
        browser.close()

    t4 = time.perf_counter()
    if RICH_AVAILABLE:
        console.print(f"[dim]Playwright → PDF: {t4-t3:.2f}s[/]")

    # Cleanup temps
    tmp_md.unlink(missing_ok=True)
    tmp_html.unlink(missing_ok=True)
    tmp_header.unlink(missing_ok=True)

    total = t4 - t_start
    if RICH_AVAILABLE:
        from rich.table import Table as RTable
        from rich.panel import Panel as RPanel
        t = RTable(show_header=False, box=None)
        t.add_row("Total", f"[bold green]{total:.1f}s[/]")
        t.add_row("PDF", str(output_pdf))
        t.add_row("Mode", "[yellow]quick/HTML[/] (no figures)")
        console.print(RPanel(t, border_style="cyan",
                             title="Quick Preview Built"))
    else:
        print(f"Quick PDF built in {total:.1f}s → {output_pdf}")


def build_pdf(
    report_md: Path,
    output_pdf: Path,
    cjk_font: str,
    diagram_format: str,
    main_font: str,
    math_font: str,
    force: bool = False,
    pdf_engine: str | None = None,
) -> None:
    t_start = time.perf_counter()

    pandoc_cmd = shutil.which("pandoc")
    if not pandoc_cmd:
        raise RuntimeError("pandoc not found. Install pandoc to build PDF.")

    # Choose engine (tectonic is significantly faster for most documents)
    if pdf_engine is None:
        pdf_engine = "tectonic" if shutil.which("tectonic") else "xelatex"
    if pdf_engine == "tectonic" and RICH_AVAILABLE:
        console.print("[bold blue]Using tectonic[/] (faster engine + better caching)")
    elif pdf_engine == "tectonic":
        print("[generate_report_pdf] Using tectonic engine (faster + better caching)")

    sources = _get_section_sources()

    # Also watch committed diagram assets (if plots change we want to rebuild)
    asset_extras = list((PROJECT_DIR / "assets").rglob("*.png")) + \
                   list((PROJECT_DIR / "assets").rglob("*.pdf"))

    # Fast path: skip full rebuild if nothing changed (biggest speed win during editing)
    if not force and _is_up_to_date(output_pdf, sources, asset_extras):
        if RICH_AVAILABLE:
            console.print("[bold green]✓ Report is up to date[/] (use --force to rebuild)")
            console.print(f"[dim]{output_pdf}[/dim]")
        else:
            print(f"[generate_report_pdf] {output_pdf} is up to date (use --force to rebuild).")
        return

    # Overall build progress bar (multi-stage) - start early
    overall_progress = None
    assemble_task = replace_task = compile_task = None
    if RICH_AVAILABLE:
        overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        overall_progress.start()
        assemble_task = overall_progress.add_task("[cyan]Stage 1/3: Assemble Markdown sections", total=1)
        replace_task = overall_progress.add_task("[yellow]Stage 2/3: Process figures & LaTeX", total=1)
        compile_task = overall_progress.add_task("[magenta]Stage 3/3: Compile PDF with pandoc", total=1)

    t0 = time.perf_counter()
    report_text = read_report_source()
    t1 = time.perf_counter()

    if RICH_AVAILABLE:
        console.print(f"[dim]Assemble sections: {t1-t0:.2f}s[/]")
        if overall_progress:
            overall_progress.update(assemble_task, completed=1)
    else:
        console.print(f"[generate_report_pdf]   assemble sections: {t1 - t0:.2f}s")

    # Build replacement list (for nice progress + safeguard)
    replacements = [
        (METHOD_PIPELINE_MD, f"```{{=latex}}\n{METHOD_PIPELINE_LATEX}\n```"),
        (f"{MODEL_ARCH_MD}\n{MODEL_ARCH_CAPTION_MD}", f"```{{=latex}}\n{MODEL_ARCH_LATEX_TEMPLATE}\n```"),
    ]
    for image_path, latex_snippet in APPENDIX_ALGO_LATEX.items():
        replacements.append((f"![]({image_path})", f"```{{=latex}}\n{latex_snippet}\n```"))
    replacements += [
        (FIGURE_A_MD, f"```{{=latex}}\n{FIGURE_A_LATEX}\n```"),
        (FIGURE_B_MD, f"```{{=latex}}\n{FIGURE_B_LATEX}\n```"),
        (PARETO_FIGURE_BLOCK_MD, f"```{{=latex}}\n{PARETO_FIGURE_LATEX}\n```"),
        (FIGURE_8A_BLOCK_MD, f"```{{=latex}}\n{FIGURE_8A_LATEX}\n```"),
        (FIGURE_8B_BLOCK_MD, f"```{{=latex}}\n{FIGURE_8B_LATEX}\n```"),
        (FIGURE_8C_BLOCK_MD, f"```{{=latex}}\n{FIGURE_8C_LATEX}\n```"),
        (FIGURE_8D_BLOCK_MD, f"```{{=latex}}\n{FIGURE_8D_LATEX}\n```"),
        (CAUSAL_MASK_MD, f"```{{=latex}}\n{CAUSAL_MASK_LATEX}\n```"),
        (CAUSAL_MASK_VISUAL_MD, f"```{{=latex}}\n{CAUSAL_MASK_VISUAL_LATEX}\n```"),
        (SFT_LOSS_MASK_MD, f"```{{=latex}}\n{SFT_LOSS_MASK_LATEX}\n```"),
        (SFT_VERIFICATION_MD, f"```{{=latex}}\n{SFT_VERIFICATION_LATEX}\n```"),
        (CHATML_TEMPLATE_MD, f"```{{=latex}}\n{CHATML_TEMPLATE_LATEX}\n```"),
        (SFT_TRAIN_VAL_MD, f"```{{=latex}}\n{SFT_TRAIN_VAL_LATEX}\n```"),
        # §9.9 商業競爭雷達圖 — 直接內嵌 LaTeX figure（圖說已在 {=latex} 塊內）
        # 這個 replace 只在有人把 radar chart 寫成 bare markdown image 時才會觸發；
        # 目前 9.9 節的雷達圖已用 {=latex} 直接寫入，故此條目僅作為保險備案。
    ]

    # Use the overall replace_task for progress (no nested Live)
    total_rep_steps = len(replacements) + 4
    if RICH_AVAILABLE and overall_progress:
        overall_progress.update(replace_task, total=total_rep_steps, description="[yellow]Stage 2/3: Process figures & LaTeX")

    for old, new in replacements:
        report_text = report_text.replace(old, new)
        if RICH_AVAILABLE and overall_progress:
            overall_progress.advance(replace_task)

    # Unicode replacements
    for old, new in UNICODE_TEXT_MATH_REPLACEMENTS.items():
        report_text = report_text.replace(old, new)
        if RICH_AVAILABLE and overall_progress:
            overall_progress.advance(replace_task)

    # Safeguard for YAML --- 
    _lines = report_text.split("\n")
    _dash_count = 0
    for _i, _ln in enumerate(_lines):
        if _ln.strip() == "---":
            _dash_count += 1
            if _dash_count > 2:
                _lines[_i] = _ln.replace("---", "***", 1)
    report_text = "\n".join(_lines)
    if RICH_AVAILABLE and overall_progress:
        overall_progress.advance(replace_task)

    # Format substitution
    report_text = report_text.replace(
        "../assets/method_flowchart.svg", f"../assets/method_flowchart.{diagram_format}"
    )
    if RICH_AVAILABLE and overall_progress:
        overall_progress.advance(replace_task)
    report_text = report_text.replace(
        "../assets/images/architecture.svg", f"../assets/images/architecture.{diagram_format}"
    )
    if RICH_AVAILABLE and overall_progress:
        overall_progress.advance(replace_task)

    t2 = time.perf_counter()
    if RICH_AVAILABLE:
        console.print(f"[dim]Replaces + safeguard: {t2-t1:.2f}s[/]")
        if overall_progress:
            overall_progress.update(replace_task, completed=total_rep_steps)
    else:
        console.print(f"[generate_report_pdf]   replaces: {t2 - t1:.2f}s")

    # Build from the report/ directory: section files' image paths are written
    # as ../assets/... (relative to report/), and raw-LaTeX \includegraphics
    # is resolved by xelatex relative to the build cwd — so both the temp md
    # and the pandoc cwd must live in report/ for ../assets to resolve.
    # The sections/ files are concatenated in read_report_source(); the resulting
    # string has the same content as the old monolithic report.md.
    report_dir = REPORT_MD.parent
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        encoding="utf-8",
        delete=False,
        dir=str(report_dir),
    ) as tmp_md:
        tmp_md.write(report_text)
        tmp_md_path = Path(tmp_md.name)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tex",
        encoding="utf-8",
        delete=False,
        dir=str(PROJECT_DIR),
    ) as header_file:
        header_file.write(
            "\\usepackage{fontspec}\n"
            "\\usepackage{unicode-math}\n"
            f"\\setmainfont{{{main_font}}}\n"
            f"\\setmathfont{{{math_font}}}\n"
            "\\usepackage{amsmath,amssymb,mathtools,bm}\n"
            "\\usepackage{array}\n"
            "\\usepackage{tabularx}\n"
            "\\usepackage{needspace}\n"
            "\\usepackage{float}\n"
            "\\usepackage{algorithm}\n"
            "\\usepackage[noend]{algpseudocode}\n"
            "\\algrenewcommand\\algorithmicrequire{\\textbf{Require:}}\n"
            "\\algrenewcommand\\algorithmicensure{\\textbf{Ensure:}}\n"
            "\\algrenewcommand\\algorithmicreturn{\\textbf{return}}\n"
            # hyperref: needed for \url{} in titlepage; hidelinks keeps text clean
            "\\usepackage[hidelinks,bookmarks=true,unicode=true]{hyperref}\n"
            # url package for line-break-safe \texttt URLs (fallback)
            "\\usepackage{url}\n"
            # Suppress pandoc's auto-\maketitle (we provide our own titlepage block).
            "\\AtBeginDocument{\\let\\maketitle\\relax}\n"
            # TOC: include subsubsection (###) and paragraph (####) levels.
            # Default tocdepth=3 misses #### entries (7.2.7, 8.4.x, 8.5.x etc.).
            "\\setcounter{tocdepth}{4}\n"
        )
        header_path = Path(header_file.name)

    cmd = [
        pandoc_cmd,
        str(tmp_md_path),
        "--from",
        "markdown+tex_math_dollars",
        f"--pdf-engine={pdf_engine}",
        "--include-in-header",
        str(header_path),
        "-V",
        f"CJKmainfont={cjk_font}",
        "-V",
        "CJKoptions=AutoFakeBold,AutoFakeSlant",
        "-V",
        "geometry:margin=1in",
        "--resource-path=.",
        "--output",
        str(output_pdf),
    ]
    # xelatex only: -interaction=nonstopmode prevents engine from pausing on
    # missing glyphs or minor errors — biggest single cause of 90s hangs.
    # tectonic handles this differently and does not accept this flag.
    if pdf_engine == "xelatex":
        cmd += ["--pdf-engine-opt=-interaction=nonstopmode",
                "--pdf-engine-opt=-halt-on-error"]

    t4 = time.perf_counter()
    if RICH_AVAILABLE and overall_progress:
        overall_progress.update(compile_task, description=f"[magenta]Stage 3/3: Compile PDF with {pdf_engine} (running...)")

    try:
        # Avoid nested Live: don't use console.status when overall Progress is active
        if RICH_AVAILABLE and overall_progress:
            subprocess.run(cmd, cwd=str(report_dir), check=True)
        elif RICH_AVAILABLE:
            with console.status(f"[bold green]Running pandoc + {pdf_engine} (this may take a while)...", spinner="dots"):
                subprocess.run(cmd, cwd=str(report_dir), check=True)
        else:
            subprocess.run(cmd, cwd=str(report_dir), check=True)
    finally:
        tmp_md_path.unlink(missing_ok=True)
        header_path.unlink(missing_ok=True)

    t5 = time.perf_counter()
    if RICH_AVAILABLE:
        console.print(f"[dim]pandoc + {pdf_engine}: {t5 - t4:.1f}s[/]")
        if overall_progress:
            overall_progress.update(compile_task, completed=1)
            overall_progress.stop()
    else:
        console.print(f"[generate_report_pdf]   pandoc + {pdf_engine}: {t5 - t4:.1f}s")

    try:
        shown = output_pdf.relative_to(PROJECT_DIR)
    except ValueError:
        shown = output_pdf

    total_time = t5 - t_start
    if RICH_AVAILABLE:
        # Beautiful summary
        table = Table(title="Build Summary", show_header=False, box=None)
        table.add_row("Total time", f"[bold]{total_time:.1f}s[/]")
        table.add_row("PDF", f"[link=file://{output_pdf}]{output_pdf}[/link]")
        table.add_row("Engine", pdf_engine)
        console.print(Panel(table, border_style="green"))
        console.print(f"[bold green]✓ Built PDF:[/] {shown}")
    else:
        console.print(f"[generate_report_pdf] Total: {total_time:.1f}s")
        console.print(f"Built PDF: {shown}")

    # Note on multi-threading (kept for developers)
    # - Section file reads and diagram rendering (when --render-diagrams) use ThreadPoolExecutor.
    # - Core PDF generation remains a single high-quality LaTeX process.


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build report PDF from report/sections/*.md.\n"
        "\n"
        "DEFAULT (quality):  pandoc + xelatex/tectonic — full CJK, math, figures (~60-90s)\n"
        "QUICK  (-q/--quick): pandoc → HTML + Playwright → PDF — text+math only, no figures (~4-8s)\n"
        "\n"
        "Use --quick during editing; use default for final PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="Fast preview mode: HTML + Playwright PDF in ~4-8s. "
             "Requires playwright (pip install playwright && playwright install chromium). "
             "Raw LaTeX blocks (figures, algorithms) are omitted; text and math render correctly.",
    )
    parser.add_argument(
        "--render-diagrams",
        action="store_true",
        help="Force regeneration of diagrams from prototypes/*.html (very slow; uses Playwright + Chromium). "
        "By default only stale diagrams are updated when this flag is given.",
    )
    parser.add_argument(
        "--skip-svg",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PDF,
        help="Output PDF path (default: output/report.pdf).",
    )
    parser.add_argument(
        "--cjk-font",
        default="Songti SC",
        help="CJK font family used by pandoc/xelatex (default: Songti SC). Quality mode only.",
    )
    parser.add_argument(
        "--diagram-format",
        choices=("pdf", "png"),
        default="png",
        help="Extension for method_flowchart.svg / architecture.svg (default: png). Quality mode only.",
    )
    parser.add_argument(
        "--main-font",
        default="STIX Two Text",
        help="Main Latin font (default: STIX Two Text). Quality mode only.",
    )
    parser.add_argument(
        "--math-font",
        default="STIX Two Math",
        help="Math font (default: STIX Two Math). Quality mode only.",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force full rebuild even if output PDF is newer than sources.",
    )
    parser.add_argument(
        "--pdf-engine",
        choices=("xelatex", "tectonic", "lualatex"),
        default=None,
        help="Override PDF engine (default: tectonic if installed, else xelatex). Quality mode only.",
    )
    args = parser.parse_args()

    if RICH_AVAILABLE:
        mode_tag = "[yellow]QUICK[/]" if args.quick else "[blue]QUALITY[/]"
        console.rule(f"[bold cyan]Hybrid Mamba Report Builder[/] — {mode_tag}", style="cyan")
    else:
        mode = "QUICK (~4-8s)" if args.quick else "QUALITY (xelatex)"
        print(f"[generate_report_pdf] Mode: {mode}")

    # ── QUICK path ────────────────────────────────────────────────────────
    if args.quick:
        output = args.output.resolve()
        # Default to a separate quick output so we don't overwrite the quality PDF
        if output == OUTPUT_PDF.resolve():
            output = output.with_stem(output.stem + "_quick")
        build_quick_pdf(REPORT_MD, output, force=args.force)
        return

    # ── QUALITY path ──────────────────────────────────────────────────────
    if not RICH_AVAILABLE:
        print("[yellow]Tip: pip install rich for progress bars and output[/]")

    if args.render_diagrams:
        t_r0 = time.perf_counter()
        render_svg_from_html()
        if RICH_AVAILABLE:
            console.print(f"[dim]Diagrams (parallel): {time.perf_counter() - t_r0:.1f}s[/]")
        else:
            print(f"[generate_report_pdf] diagrams (Playwright): {time.perf_counter() - t_r0:.1f}s")

    build_pdf(
        REPORT_MD,
        args.output.resolve(),
        args.cjk_font,
        args.diagram_format,
        args.main_font,
        args.math_font,
        force=args.force,
        pdf_engine=args.pdf_engine,
    )


if __name__ == "__main__":
    main()
