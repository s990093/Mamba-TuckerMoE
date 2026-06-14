#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自動化繪製 SFT-CoT 訓練曲線：讀取 train_sft_cot.py 產出的 CSV，
生成 6 面板圖表（Train Loss / Val Loss / PPL / LR+Grad / Tok/s+StepTime / Router Temp）。

CSV 欄位（train）：step, loss, ce_loss, lr, grad_norm, router_temp, tokens_seen, step_time_s
CSV 欄位（val）  ：step, val_ce_loss, val_loss_mean, val_batches

使用範例：
  # 自動偵測（從 sft_cot_bundle 根目錄）
  python3 scripts/plot_sft_train_val.py

  # 手動指定
  python3 scripts/plot_sft_train_val.py \\
      --train output/train_sft_cot_log.csv \\
      --val   output/val_sft_cot_log.csv \\
      -o output/sft_cot_curves.png -w 30

  # 終端機直接看 ASCII 圖表
  python3 scripts/plot_sft_train_val.py -t

  # 互動視窗
  python3 scripts/plot_sft_train_val.py -s

  # Import as API
  from plot_sft_train_val import plot_sft_train_val, _print_terminal_stats
  plot_sft_train_val("train_log.csv", val_csv="val_log.csv", terminal=True)
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def _parse_early_flags() -> tuple[bool, bool]:
    argv = set(sys.argv[1:])
    show = bool(argv & {"--show", "-s"})
    term = bool(argv & {"--term", "-t", "--terminal"})
    return show, term


_show_flag, _term_flag = _parse_early_flags()


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float | None:
    v = (row.get(key) or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _series(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for row in rows:
        s = _f(row, "step")
        y = _f(row, key)
        if s is not None and y is not None:
            xs.append(s)
            ys.append(y)
    return xs, ys


def _moving_average(data: list[float], window: int) -> list[float]:
    if window <= 1 or not data:
        return list(data)
    ma = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        ma.append(sum(data[start : i + 1]) / (i - start + 1))
    return ma


def _ppl_from_ce(ce_vals: list[float]) -> list[float]:
    out = []
    for v in ce_vals:
        try:
            out.append(math.exp(v) if v < 20.0 else float("inf"))
        except OverflowError:
            out.append(float("inf"))
    return out


def _find_default(name: str) -> Path | None:
    candidates = [
        Path(f"output/{name}"),
        Path(f"../output/{name}"),
    ]
    for c in candidates:
        if c.is_file() and c.stat().st_size > 0:
            return c.resolve()
    return None


# ---------------------------------------------------------------------------
# Terminal helpers (ASCII charts & summary)
# ---------------------------------------------------------------------------

def _print_terminal_stats(
    tr: list[dict],
    va: list[dict],
    window: int,
) -> None:
    """Print a compact numeric summary of training/validation stats."""
    xs_loss, ys_loss = _series(tr, "loss")
    xs_ce, ys_ce = _series(tr, "ce_loss")
    _, lr = _series(tr, "lr")
    _, gn = _series(tr, "grad_norm")
    _, rt = _series(tr, "router_temp")
    _, st = _series(tr, "step_time_s")
    _, tok = _series(tr, "tokens_seen")
    vx1, vce = _series(va, "val_ce_loss")
    _, vlm = _series(va, "val_loss_mean")

    print(f"\n{'=' * 60}")
    print(f"  Training Summary  (MA window = {window})")
    print(f"{'=' * 60}")

    last_step = _f(tr[-1], "step") if tr else 0
    print(f"  Total steps     : {int(last_step or 0)}")

    if ys_loss:
        ma_loss = _moving_average(ys_loss, window)
        print(f"  {'L_total':14s}: {ma_loss[0]:.4f} -> {ma_loss[-1]:.4f}  "
              f"(range {min(ys_loss):.4f}-{max(ys_loss):.4f})")
    if ys_ce:
        ma_ce = _moving_average(ys_ce, window)
        print(f"  {'CE loss':14s}: {ma_ce[0]:.4f} -> {ma_ce[-1]:.4f}  "
              f"(range {min(ys_ce):.4f}-{max(ys_ce):.4f})")
    if lr:
        print(f"  {'LR':14s}: {lr[0]:.2e} -> {lr[-1]:.2e}")
    if gn:
        print(f"  {'Grad norm':14s}: avg {sum(gn)/len(gn):.3f}  max {max(gn):.3f}")
    if rt:
        print(f"  {'Router temp':14s}: {rt[0]:.3f} -> {rt[-1]:.3f}")
    if st:
        print(f"  {'Step time':14s}: avg {sum(st)/len(st):.2f}s  max {max(st):.2f}s")

    has_val = bool(vce or vlm)
    if has_val:
        print(f"\n  Validation:")
        if vce:
            print(f"  {'val CE':14s}: {min(vce):.4f} (min)  last {vce[-1]:.4f}")
        if vlm:
            print(f"  {'val loss':14s}: {min(vlm):.4f} (min)  last {vlm[-1]:.4f}")

    print(f"{'=' * 60}\n")


def _plot_terminal(
    tr: list[dict],
    va: list[dict],
    window: int,
) -> None:
    """Render ASCII charts in terminal using plotext."""
    try:
        import plotext as plt_t
    except ImportError:
        print("\n[!] Install plotext for terminal charts:  pip install plotext\n")
        _print_terminal_stats(tr, va, window)
        return

    xs_loss, ys_loss = _series(tr, "loss")
    xs_ce, ys_ce = _series(tr, "ce_loss")
    xs_lr, lr = _series(tr, "lr")
    xs_gn, gn = _series(tr, "grad_norm")
    xs_rt, rt = _series(tr, "router_temp")
    xs_st, step_t = _series(tr, "step_time_s")
    _, tok_seen = _series(tr, "tokens_seen")
    vx1, vce = _series(va, "val_ce_loss")
    _, vlm = _series(va, "val_loss_mean")

    panels: list[tuple[str, callable]] = []

    # --- Training Loss ---
    def _tloss_panel():
        plt_t.clear_figure()
        has = False
        if xs_loss:
            plt_t.plot(xs_loss, _moving_average(ys_loss, window),
                       label=f"loss MA", color="blue")
            has = True
        if xs_ce:
            plt_t.plot(xs_ce, _moving_average(ys_ce, window),
                       label=f"ce_loss MA", color="orange")
            has = True
        if has:
            plt_t.title("Training Loss (MA)")
            plt_t.xlabel("Step"); plt_t.ylabel("Loss")
            plt_t.show()
    panels.append(("Training Loss", _tloss_panel))

    # --- Validation Loss ---
    if vce or vlm:
        def _vloss_panel():
            plt_t.clear_figure()
            has = False
            if vce:
                plt_t.scatter(vx1, vce, label="val CE", color="green")
                has = True
            if vlm:
                plt_t.scatter([], [])
                for i, step in enumerate(_series(va, "val_loss_mean")[0]):
                    pass
                if vlm:
                    vx2 = list(range(len(vlm)))
                    plt_t.plot(vx2, vlm, label="val loss mean", color="red")
                    has = True
            # simplified: just plot val_ce_loss and val_loss_mean
            if vce:
                plt_t.plot(vx1, vce, label="val CE", color="green")
            if vlm:
                vx2, vy2 = _series(va, "val_loss_mean")
                if vx2:
                    plt_t.plot(vx2, vy2, label="val loss", color="red")
            if has:
                plt_t.title("Validation Loss")
                plt_t.xlabel("Step"); plt_t.ylabel("Loss")
                plt_t.show()
        panels.append(("Validation Loss", _vloss_panel))

    # --- LR ---
    if lr:
        def _lr_panel():
            plt_t.clear_figure()
            plt_t.plot(xs_lr, lr, label="lr", color="purple")
            plt_t.title("Learning Rate")
            plt_t.xlabel("Step"); plt_t.ylabel("LR")
            plt_t.show()
        panels.append(("Learning Rate", _lr_panel))

    # --- Grad Norm ---
    if gn:
        def _gn_panel():
            plt_t.clear_figure()
            plt_t.plot(xs_gn, _moving_average(gn, window),
                       label="|grad| MA", color="brown")
            plt_t.title("Gradient Norm")
            plt_t.xlabel("Step"); plt_t.ylabel("|grad|")
            plt_t.show()
        panels.append(("Gradient Norm", _gn_panel))

    # --- Throughput / Step Time ---
    if xs_st and step_t and len(tok_seen) >= 2:
        dt = tok_seen[1] - tok_seen[0]
        if dt > 0:
            tps = [dt / max(t, 1e-6) for t in step_t]
            def _tps_panel():
                plt_t.clear_figure()
                plt_t.plot(xs_st, _moving_average(tps, window),
                           label="tok/s MA", color="cyan")
                plt_t.title("Throughput (tok/s)")
                plt_t.xlabel("Step"); plt_t.ylabel("tok/s")
                plt_t.show()
            panels.append(("Throughput", _tps_panel))

    # --- Router Temp ---
    if rt:
        def _rt_panel():
            plt_t.clear_figure()
            plt_t.plot(xs_rt, rt, label="router temp", color="gray")
            plt_t.title("Router Temperature")
            plt_t.xlabel("Step"); plt_t.ylabel("Temp")
            plt_t.show()
        panels.append(("Router Temp", _rt_panel))

    # Show numeric summary first
    _print_terminal_stats(tr, va, window)

    for name, fn in panels:
        fn()
        try:
            input(f"\n[{name}] Press Enter for next panel (q to quit)... ")
        except EOFError:
            break


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_sft_train_val(
    train_csv: Path | None = None,
    val_csv: Path | None = None,
    output: Path | None = None,
    window: int = 20,
    dpi: int = 200,
    show: bool = False,
    terminal: bool = False,
) -> Path:
    """Plot SFT-CoT training + validation curves (6-panel dashboard).

    Parameters
    ----------
    train_csv : Path, optional
        Training CSV. Auto-detected from output/train_sft_cot_log.csv if None.
    val_csv : Path, optional
        Validation CSV. Auto-detected from output/val_sft_cot_log.csv if None.
    output : Path, optional
        Output image path. Default output/sft_cot_curves.png.
    window : int
        MA smoothing window (default 20).
    dpi : int
        Output DPI (default 200).
    show : bool
        Open interactive matplotlib window after saving.
    terminal : bool
        Display ASCII charts and summary in terminal using plotext.

    Returns
    -------
    Path
        The saved image file path.
    """
    import matplotlib
    if show:
        for _backend in ("TkAgg", "Qt5Agg", "QtAgg"):
            try:
                matplotlib.use(_backend)
                break
            except Exception:
                continue
        else:
            matplotlib.use("Agg")
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Resolve paths
    _train_csv = train_csv or _find_default("train_sft_cot_log.csv")
    _val_csv = val_csv or _find_default("val_sft_cot_log.csv")
    out_path = output or Path("output/sft_cot_curves.png")

    if not _train_csv or not _train_csv.is_file():
        raise FileNotFoundError(
            f"找不到訓練 CSV: {_train_csv}。請用 train_csv= 指定，或確認 output/train_sft_cot_log.csv 已產生。"
        )

    tr = _read_rows(_train_csv)
    va = _read_rows(_val_csv) if _val_csv else []

    if not tr:
        raise ValueError("訓練 CSV 無資料列，請等訓練開始寫入後再執行。")

    print(f"Train CSV : {_train_csv}  ({len(tr)} rows)")
    if va:
        print(f"Val CSV   : {_val_csv}  ({len(va)} rows)")
    else:
        print("Val CSV   : (尚無資料或未找到)")
    print(f"Output    : {out_path}")

    plt.rcParams.update({
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#E0E0E0",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
    })

    C_LOSS = "#1f77b4"
    C_CE = "#ff7f0e"
    C_VCE = "#2ca02c"
    C_VLM = "#d62728"
    C_LR = "#9467bd"
    C_GRAD = "#8c564b"
    C_PPL = "#17becf"
    C_TIME = "#e377c2"
    C_TPS = "#bcbd22"
    C_TEMP = "#7f7f7f"

    W = window
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)

    last_step = _f(tr[-1], "step") if tr else 0
    fig.suptitle(
        f"SFT-CoT Training Dashboard  ·  {len(tr)} steps (last={int(last_step or 0)})"
        f"  |  MA window={W}",
        fontsize=15, fontweight="bold",
    )

    def _grid(ax):
        ax.grid(True, linestyle="--", alpha=0.35, color="#A0A0A0")

    # (0,0) Train Loss / CE Loss
    ax = axes[0, 0]
    sx, sy = _series(tr, "loss")
    if sx:
        ax.plot(sx, sy, color=C_LOSS, alpha=0.15, linewidth=1)
        ax.plot(sx, _moving_average(sy, W), color=C_LOSS, linewidth=1.8, label=f"loss (MA-{W})")
    sx2, sy2 = _series(tr, "ce_loss")
    if sx2:
        ax.plot(sx2, sy2, color=C_CE, alpha=0.15, linewidth=1)
        ax.plot(sx2, _moving_average(sy2, W), color=C_CE, linewidth=1.8, label=f"ce_loss (MA-{W})")
    if sx:
        ax.annotate(f"{_moving_average(sy, W)[-1]:.3f}", xy=(sx[-1], _moving_average(sy, W)[-1]),
                    fontsize=9, color=C_LOSS, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")
    if sx2:
        ax.annotate(f"{_moving_average(sy2, W)[-1]:.3f}", xy=(sx2[-1], _moving_average(sy2, W)[-1]),
                    fontsize=9, color=C_CE, fontweight="bold",
                    xytext=(5, -12), textcoords="offset points")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(loc="best")
    _grid(ax)

    # (0,1) Validation Loss
    ax = axes[0, 1]
    vx1, vce2 = _series(va, "val_ce_loss")
    vx2, vlm = _series(va, "val_loss_mean")
    if not vx1 and not vx2:
        ax.text(0.5, 0.5, "Validation data not yet available",
                ha="center", va="center", transform=ax.transAxes,
                color="gray", fontsize=12, fontstyle="italic")
    else:
        if vce2:
            ax.plot(vx1, vce2, color=C_VCE, linewidth=2, label="val_ce_loss")
            ax.annotate(f"{vce2[-1]:.3f}", xy=(vx1[-1], vce2[-1]),
                        fontsize=9, color=C_VCE, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")
        if vlm:
            ax.plot(vx2, vlm, color=C_VLM, linewidth=2, label="val_loss_mean")
            ax.annotate(f"{vlm[-1]:.3f}", xy=(vx2[-1], vlm[-1]),
                        fontsize=9, color=C_VLM, fontweight="bold",
                        xytext=(5, -12), textcoords="offset points")
        ax.legend(loc="best")
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss")
    _grid(ax)

    # (1,0) Perplexity
    ax = axes[1, 0]
    sx_ce, sy_ce = _series(tr, "ce_loss")
    if sx_ce:
        ppl_raw = _ppl_from_ce(sy_ce)
        ppl_ma = _moving_average(ppl_raw, W)
        ax.plot(sx_ce, ppl_raw, color=C_PPL, alpha=0.15, linewidth=1)
        ax.plot(sx_ce, ppl_ma, color=C_PPL, linewidth=1.8, label=f"train PPL (MA-{W})")
    if vce2:
        vppl = _ppl_from_ce(vce2)
        ax.plot(vx1, vppl, color=C_VCE, linewidth=2, marker="", label="val PPL")
    if sx_ce:
        last_ppl = ppl_ma[-1]
        ax.annotate(f"{last_ppl:.1f}", xy=(sx_ce[-1], last_ppl),
                    fontsize=9, color=C_PPL, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")
    if vce2 and vppl:
        ax.annotate(f"{vppl[-1]:.1f}", xy=(vx1[-1], vppl[-1]),
                    fontsize=9, color=C_VCE, fontweight="bold",
                    xytext=(5, -12), textcoords="offset points")
    if sx_ce or vce2:
        ax.legend(loc="best")
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity")
    _grid(ax)

    # (1,1) LR + Grad Norm
    ax = axes[1, 1]
    st_lr, lr2 = _series(tr, "lr")
    st_gn, gn2 = _series(tr, "grad_norm")
    ax2 = None
    if st_lr:
        ax.plot(st_lr, lr2, color=C_LR, linewidth=2.5, label="Learning Rate")
    if st_gn:
        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        gn_ma = _moving_average(gn2, W)
        ax2.plot(st_gn, gn2, color=C_GRAD, alpha=0.15, linewidth=1)
        ax2.plot(st_gn, gn_ma, color=C_GRAD, alpha=0.85, linewidth=1.5, label=f"|grad| (MA-{W})")
        ax2.set_ylabel("Gradient Norm", color=C_GRAD)
        ax2.tick_params(axis="y", labelcolor=C_GRAD)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate", color=C_LR)
    ax.tick_params(axis="y", labelcolor=C_LR)
    ax.set_title("Learning Rate & Gradient Norm")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = (ax2.get_legend_handles_labels() if ax2 else ([], []))
    if lines1 or lines2:
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    _grid(ax)

    # (2,0) Throughput + Step Time
    ax = axes[2, 0]
    st_t, step_t = _series(tr, "step_time_s")
    _, tok_seen = _series(tr, "tokens_seen")
    ax2b = None
    if st_t and step_t:
        batch_tokens = None
        if len(tok_seen) >= 2:
            dt_tok = tok_seen[1] - tok_seen[0]
            if dt_tok > 0:
                batch_tokens = dt_tok
        if batch_tokens:
            tps = [batch_tokens / max(t, 1e-6) for t in step_t]
            tps_ma = _moving_average(tps, W)
            ax.plot(st_t, tps, color=C_TPS, alpha=0.15, linewidth=1)
            ax.plot(st_t, tps_ma, color=C_TPS, linewidth=1.8, label=f"tok/s (MA-{W})")
            ax.set_ylabel("Tokens / sec", color=C_TPS)
            ax.tick_params(axis="y", labelcolor=C_TPS)
        else:
            ax.plot(st_t, step_t, color=C_TIME, alpha=0.5, linewidth=1.5, label="step_time (s)")
            ax.set_ylabel("Seconds", color=C_TIME)

        ax2b = ax.twinx()
        ax2b.spines["top"].set_visible(False)
        ax2b.spines["left"].set_visible(False)
        st_ma = _moving_average(step_t, W)
        ax2b.plot(st_t, step_t, color=C_TIME, alpha=0.15, linewidth=1)
        ax2b.plot(st_t, st_ma, color=C_TIME, linewidth=1.5, label=f"step_time (MA-{W})")
        ax2b.set_ylabel("Step Time (s)", color=C_TIME)
        ax2b.tick_params(axis="y", labelcolor=C_TIME)

    ax.set_xlabel("Step")
    ax.set_title("Throughput & Step Time")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = (ax2b.get_legend_handles_labels() if ax2b else ([], []))
    if lines1 or lines2:
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    _grid(ax)

    # (2,1) Router Temperature + Cumulative Tokens
    ax = axes[2, 1]
    st_rt, rt2 = _series(tr, "router_temp")
    ax2c = None
    if st_rt:
        ax.plot(st_rt, rt2, color=C_TEMP, drawstyle="steps-post", linewidth=2.5, label="Router Temp")
    ax.set_ylabel("Router Temperature", color=C_TEMP)
    ax.tick_params(axis="y", labelcolor=C_TEMP)

    if tok_seen:
        if len(tok_seen) >= 1:
            tok_m = [t / 1e6 for t in tok_seen]
            # use the step column for x-axis
            sx_t, _ = _series(tr, "step")
            ax2c = ax.twinx()
            ax2c.spines["top"].set_visible(False)
            ax2c.spines["left"].set_visible(False)
            ax2c.plot(sx_t[:len(tok_m)], tok_m, color="#2196F3", linewidth=1.8, alpha=0.7,
                       label="Cumulative Tokens (M)")
            ax2c.set_ylabel("Tokens (M)", color="#2196F3")
            ax2c.tick_params(axis="y", labelcolor="#2196F3")

    ax.set_xlabel("Step")
    ax.set_title("Router Temperature & Cumulative Tokens")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = (ax2c.get_legend_handles_labels() if ax2c else ([], []))
    if lines1 or lines2:
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    _grid(ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"圖表已儲存：{out_path.resolve()}")

    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"[!] Cannot show interactive plot: {e}")

    if terminal:
        _plot_terminal(tr, va, window)

    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="自動化繪製 SFT-CoT 訓練 & 驗證曲線（6 面板）。"
    )
    ap.add_argument("--train", type=Path, default=None, help="訓練 CSV（預設自動偵測 output/train_sft_cot_log.csv）")
    ap.add_argument("--val", type=Path, default=None, help="驗證 CSV（預設自動偵測 output/val_sft_cot_log.csv）")
    ap.add_argument("-o", "--output", type=Path, default=None, help="輸出圖檔（預設 output/sft_cot_curves.png）")
    ap.add_argument("-w", "--window", type=int, default=20, help="MA 平滑窗口（預設 20）")
    ap.add_argument("--dpi", type=int, default=200, help="圖檔 DPI（預設 200）")
    ap.add_argument("--show", "-s", action="store_true", default=False,
                    help="Open interactive plot window (requires GUI backend)")
    ap.add_argument("--term", "-t", action="store_true", default=False, dest="term",
                    help="Display ASCII charts in terminal (requires plotext)")
    ap.add_argument("--terminal", action="store_true", default=False, dest="term",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    plot_sft_train_val(
        train_csv=args.train,
        val_csv=args.val,
        output=args.output,
        window=args.window,
        dpi=args.dpi,
        show=args.show,
        terminal=args.term,
    )


if __name__ == "__main__":
    main()
