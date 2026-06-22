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
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORT_MD = PROJECT_DIR / "report/report.md"
OUTPUT_PDF = PROJECT_DIR / "output/report.pdf"

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

# 必須與 report.md 摘要內「圖 1」區塊逐字一致，否則 replace 不生效 → pandoc 浮體跑版（僅剩圖說、圖不見）。
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

# 與 report.md §7.2.5 Markdown 區塊逐字一致；否則 replace 失效致浮體跑版。
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
UNICODE_TEXT_MATH_REPLACEMENTS = {
    "→": r"$\to$",
    "←": r"$\leftarrow$",
    "↔": r"$\leftrightarrow$",
    "≈": r"$\approx$",
    "↓": r"$\downarrow$",
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
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 2400, "height": 1600}, device_scale_factor=2
            )
            page = context.new_page()

            for html_rel_path, svg_rel_path in DIAGRAM_SPECS:
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
                output_path = PROJECT_DIR / svg_rel_path
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
                    print(f"Rendered SVG: {output_path.relative_to(PROJECT_DIR)}")

                # Also export high-res PNG fallback.
                png_path = output_path.with_suffix(".png")
                # 勿使用 ".page, body".first：逗號選群在 DOM 中會先命中 body（祖先在子節點之前），
                # 導致整個 viewport（例 4800×）截圖、圖側大量留白。.page 存在時務必對其取景。
                if has_svg:
                    target_el = page.locator("svg").first
                elif page.locator(".page").count() > 0:
                    target_el = page.locator(".page").first
                else:
                    target_el = page.locator("body").first
                target_el.screenshot(path=str(png_path), scale="device")
                print(f"Rendered PNG: {png_path.relative_to(PROJECT_DIR)}")

                # Export vector PDF directly from source HTML page after MathJax/CSS
                # are fully applied, to avoid style loss from stripped SVG documents.
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
                    # For normal HTML, we want to hide everything outside .page and force its height to be 100vh
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
                print(f"Rendered vector PDF: {pdf_path.relative_to(PROJECT_DIR)}")

            context.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        os.chdir(old_cwd)


def build_pdf(
    report_md: Path,
    output_pdf: Path,
    cjk_font: str,
    diagram_format: str,
    main_font: str,
    math_font: str,
) -> None:
    pandoc_cmd = shutil.which("pandoc")
    if not pandoc_cmd:
        raise RuntimeError("pandoc not found. Install pandoc to build PDF.")

    report_text = report_md.read_text(encoding="utf-8")
    for old, new in UNICODE_TEXT_MATH_REPLACEMENTS.items():
        report_text = report_text.replace(old, new)
    report_text = report_text.replace(
        METHOD_PIPELINE_MD, f"```{{=latex}}\n{METHOD_PIPELINE_LATEX}\n```"
    )
    report_text = report_text.replace(
        f"{MODEL_ARCH_MD}\n{MODEL_ARCH_CAPTION_MD}",
        f"```{{=latex}}\n{MODEL_ARCH_LATEX_TEMPLATE}\n```",
    )
    for image_path, latex_snippet in APPENDIX_ALGO_LATEX.items():
        report_text = report_text.replace(
            f"![]({image_path})", f"```{{=latex}}\n{latex_snippet}\n```"
        )
    report_text = report_text.replace(FIGURE_A_MD, f"```{{=latex}}\n{FIGURE_A_LATEX}\n```")
    report_text = report_text.replace(FIGURE_B_MD, f"```{{=latex}}\n{FIGURE_B_LATEX}\n```")
    report_text = report_text.replace(PARETO_FIGURE_BLOCK_MD, f"```{{=latex}}\n{PARETO_FIGURE_LATEX}\n```")
    report_text = report_text.replace(FIGURE_8A_BLOCK_MD, f"```{{=latex}}\n{FIGURE_8A_LATEX}\n```")
    report_text = report_text.replace(FIGURE_8B_BLOCK_MD, f"```{{=latex}}\n{FIGURE_8B_LATEX}\n```")
    report_text = report_text.replace(FIGURE_8C_BLOCK_MD, f"```{{=latex}}\n{FIGURE_8C_LATEX}\n```")
    report_text = report_text.replace(FIGURE_8D_BLOCK_MD, f"```{{=latex}}\n{FIGURE_8D_LATEX}\n```")
    report_text = report_text.replace(CAUSAL_MASK_MD, f"```{{=latex}}\n{CAUSAL_MASK_LATEX}\n```")
    report_text = report_text.replace(
        CAUSAL_MASK_VISUAL_MD, f"```{{=latex}}\n{CAUSAL_MASK_VISUAL_LATEX}\n```"
    )
    report_text = report_text.replace(SFT_LOSS_MASK_MD, f"```{{=latex}}\n{SFT_LOSS_MASK_LATEX}\n```")
    report_text = report_text.replace(SFT_VERIFICATION_MD, f"```{{=latex}}\n{SFT_VERIFICATION_LATEX}\n```")
    report_text = report_text.replace(CHATML_TEMPLATE_MD, f"```{{=latex}}\n{CHATML_TEMPLATE_LATEX}\n```")
    report_text = report_text.replace(SFT_TRAIN_VAL_MD, f"```{{=latex}}\n{SFT_TRAIN_VAL_LATEX}\n```")
    report_text = report_text.replace(
        "../assets/method_flowchart.svg", f"../assets/method_flowchart.{diagram_format}"
    )
    report_text = report_text.replace(
        "../assets/images/architecture.svg", f"../assets/images/architecture.{diagram_format}"
    )

    # Build from the report/ directory: report.md's image paths are written
    # as ../assets/... (relative to report/), and raw-LaTeX \includegraphics
    # is resolved by xelatex relative to the build cwd — so both the temp md
    # and the pandoc cwd must live in report/ for ../assets to resolve.
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
        )
        header_path = Path(header_file.name)

    cmd = [
        pandoc_cmd,
        str(tmp_md_path),
        "--from",
        "markdown+tex_math_dollars",
        "--pdf-engine=xelatex",
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
    try:
        subprocess.run(cmd, cwd=str(report_dir), check=True)
    finally:
        tmp_md_path.unlink(missing_ok=True)
        header_path.unlink(missing_ok=True)
    try:
        shown = output_pdf.relative_to(PROJECT_DIR)
    except ValueError:
        shown = output_pdf
    print(f"Built PDF: {shown}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build report PDF from report.md using committed PNG assets (fast). "
        "Optional: --render-diagrams to regenerate assets from prototypes/*.html via Playwright."
    )
    parser.add_argument(
        "--render-diagrams",
        action="store_true",
        help="Regenerate SVG/PNG/PDF from prototypes/*.html (slow; requires playwright). "
        "Default: off — use existing files under assets/.",
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
        help="Output PDF path (default: report.pdf).",
    )
    parser.add_argument(
        "--cjk-font",
        default="Songti SC",
        help="CJK font family used by pandoc/xelatex (default: Songti SC).",
    )
    parser.add_argument(
        "--diagram-format",
        choices=("pdf", "png"),
        default="png",
        help="Extension substituted for method_flowchart.svg / architecture.svg in body (default: png).",
    )
    parser.add_argument(
        "--main-font",
        default="STIX Two Text",
        help="Main Latin font for PDF text rendering (default: STIX Two Text).",
    )
    parser.add_argument(
        "--math-font",
        default="STIX Two Math",
        help="Math font for Unicode/math symbols (default: STIX Two Math).",
    )
    args = parser.parse_args()

    if args.render_diagrams:
        render_svg_from_html()

    build_pdf(
        REPORT_MD,
        args.output.resolve(),
        args.cjk_font,
        args.diagram_format,
        args.main_font,
        args.math_font,
    )


if __name__ == "__main__":
    main()
