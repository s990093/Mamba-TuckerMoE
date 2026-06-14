#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced plot for train_sft + val logs, supporting FCP/SFT-GO/SCALe metrics.

Dependencies: pandas, matplotlib, plotext (optional, for --term)

Usage:
  # Save PNG only (default)
  python3 plot_sft_train_val_enhanced.py output/train_sft_log.csv output/val_sft_log.csv

  # Save PNG + open interactive window
  python3 plot_sft_train_val_enhanced.py output/train_sft_log.csv -s

  # Save PNG + show ASCII charts in terminal
  python3 plot_sft_train_val_enhanced.py output/train_sft_log.csv -t

  # Custom MA window and output path
  python3 plot_sft_train_val_enhanced.py output/train_sft_cot_log.csv output/val_sft_cot_log.csv \\
      -o output/sft_cot_plots.png --ma 200

  # Import as API
  from plot_sft_train_val_enhanced import plot_sft_train_val
  plot_sft_train_val("train_log.csv", val_csv="val_log.csv", show=True)

Features:
  - Moving average (--ma, default 300) with raw signal shown faintly
  - Perplexity (PPL = exp(CE)) on a secondary y-axis for loss panels
  - Auto-detects FCP / SFT-GO / SCALe / FinalEnhance / StructAcc / AuxLoss columns
  - Publication-quality style: Times-like fonts, clean grid, proper figure sizing
  - --show / -s : open interactive matplotlib window
  - --term / -t : display ASCII charts + stats in terminal (requires plotext)
  - Callable Python API: plot_sft_train_val(...)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional


def _parse_early_flags() -> tuple[bool, bool]:
    """Check sys.argv for --show / --term before argparse (needed for backend)."""
    argv = set(sys.argv[1:])
    show = bool(argv & {"--show", "-s"})
    term = bool(argv & {"--term", "-t", "--terminal"})
    return show, term


_show_flag, _term_flag = _parse_early_flags()

import matplotlib
if _show_flag:
    for _backend in ("TkAgg", "Qt5Agg", "QtAgg"):
        try:
            matplotlib.use(_backend)
            break
        except Exception:
            continue
    else:
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")  # headless default; --term also uses Agg (PNG save only)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Paper-quality rcParams
# ---------------------------------------------------------------------------
_PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman", "serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.75",
    "lines.linewidth": 1.4,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}

# Colour palette (colorblind-friendly, ordered)
_C = [
    "#1f77b4",  # blue     — raw train loss / loss
    "#d62728",  # red      — ce_loss
    "#2ca02c",  # green    — val_ce_loss
    "#ff7f0e",  # orange   — val_loss_mean
    "#9467bd",  # purple   — lr
    "#8c564b",  # brown    — grad_norm
    "#e377c2",  # pink     — fcp_penalty
    "#7f7f7f",  # grey     — eos_prob
    "#17becf",  # teal     — sftgo_loss
    "#bcbd22",  # yellow   — scale_w
    "#c44e52",  # red-pink — eos_prob_max
    "#2e8b57",  # sea-green — pdl_weight_mean
    "#ff1493",  # deep-pink — ie_weight_mean
    "#1f77b4",  # [13] blue     — think_end_acc (same as loss, distinct dash)
    "#d62728",  # [14] red      — final_end_acc
    "#2ca02c",  # [15] green    — im_end_acc
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "step" in df.columns:
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
    for col in df.columns:
        if col != "step":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["step"])
    df["step"] = df["step"].astype(np.int64)
    return df


def ma(series: pd.Series, window: int) -> pd.Series:
    """Centred moving average; falls back to trailing when < window points."""
    if window <= 1 or len(series) < 2:
        return series
    return series.rolling(window=min(window, len(series)), min_periods=1, center=True).mean()


def ppl(loss: pd.Series) -> pd.Series:
    """Perplexity = exp(CE); clamp to avoid overflow."""
    clipped = loss.clip(upper=20.0)
    return np.exp(clipped.values)


def detect_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
) -> dict[str, bool]:
    cols = set(train_df.columns)
    vcols = set(val_df.columns) if val_df is not None and len(val_df) > 0 else set()
    _acc_cols = {"think_end_acc", "final_end_acc", "im_end_acc"}
    # aux_loss: compute MoE auxiliary loss = loss - ce_loss when both exist
    _has_aux = "loss" in cols and "ce_loss" in cols
    return {
        "fcp": "fcp_penalty" in cols or "eos_prob" in cols or "eos_prob_max" in cols,
        "fcp_max": "eos_prob_max" in cols,
        "val_fcp": bool(
            vcols
            & {"val_fcp_penalty", "val_eos_prob", "val_eos_prob_max"}
        ),
        "sftgo": "sftgo_loss" in cols,
        "scale": "scale_w" in cols,
        "final_enhance": ("pdl_weight_mean" in cols) or ("ie_weight_mean" in cols),
        "struct_acc": bool(cols & _acc_cols),
        "aux_loss": _has_aux,
    }


# ---------------------------------------------------------------------------
# Per-panel plot helpers
# ---------------------------------------------------------------------------

def _raw_and_ma(
    ax,
    x: pd.Series,
    y: pd.Series,
    window: int,
    color: str,
    label: str,
    *,
    linestyle: str = "-",
    zorder: int = 2,
) -> None:
    """Plot raw signal (faint) and moving average (bold)."""
    valid = y.notna()
    xv, yv = x[valid], y[valid]
    if len(xv) == 0:
        return
    # Raw (translucent)
    ax.plot(xv, yv, color=color, linewidth=0.5, alpha=0.25, linestyle=linestyle, zorder=zorder)
    # MA (solid)
    yma = ma(yv.reset_index(drop=True), window)
    ax.plot(xv, yma.values, color=color, linewidth=1.6, label=f"{label} (MA{window})",
            linestyle=linestyle, zorder=zorder + 1)


def _add_ppl_axis(ax_primary, x: pd.Series, y_loss: pd.Series, window: int, color: str) -> None:
    """Add a right-hand PPL axis aligned to the CE loss on the left axis."""
    valid = y_loss.notna()
    xv, yv = x[valid], y_loss[valid]
    if len(xv) == 0:
        return
    ax2 = ax_primary.twinx()
    yma = ma(yv.reset_index(drop=True), window)
    ppl_vals = np.exp(np.clip(yma.values, None, 20.0))
    ax2.plot(xv, ppl_vals, color=color, linewidth=1.0, linestyle=":", alpha=0.7, label="PPL (MA)")
    ax2.set_ylabel("Perplexity", fontsize=8, color=color)
    ax2.tick_params(axis="y", labelsize=7, colors=color)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(color)
    ax2.spines["right"].set_linewidth(0.6)
    # Sync y-range so PPL ticks do not crowd
    ax2.set_ylim(bottom=max(1.0, ppl_vals.min() * 0.9))
    return ax2


# ---------------------------------------------------------------------------
# Panel functions
# ---------------------------------------------------------------------------

def panel_train_loss(ax, train_df: pd.DataFrame, window: int, title: str = "Training Loss") -> None:
    if "step" not in train_df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    x = train_df["step"]

    if "loss" in train_df.columns:
        _raw_and_ma(ax, x, train_df["loss"], window, _C[0], "L_total")

    if "ce_loss" in train_df.columns:
        _raw_and_ma(ax, x, train_df["ce_loss"], window, _C[1], "CE")
        # PPL on right axis (from CE)
        _add_ppl_axis(ax, x, train_df["ce_loss"], window, _C[1])

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontweight="bold")
    # Tidy legend — exclude the PPL twinx handle from left-axis legend
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(h, l, loc="upper right")


def panel_validation(ax, val_df: pd.DataFrame, window: int, title: str = "Validation Loss") -> None:
    if "step" not in val_df.columns or len(val_df) == 0:
        ax.text(0.5, 0.5, "No validation data", ha="center", va="center", transform=ax.transAxes)
        return
    x = val_df["step"]
    has_data = False

    if "val_ce_loss" in val_df.columns:
        y = val_df["val_ce_loss"]
        valid = y.notna()
        if valid.any():
            ax.plot(x[valid], y[valid], "o", color=_C[2], markersize=3.5,
                    alpha=0.6, zorder=3)
            yma = ma(y[valid].reset_index(drop=True), window)
            ax.plot(x[valid], yma.values, color=_C[2], linewidth=1.6, label="val CE (MA)")
            _add_ppl_axis(ax, x, y, window, _C[2])
            has_data = True

    if "val_loss_mean" in val_df.columns:
        y = val_df["val_loss_mean"]
        valid = y.notna()
        if valid.any():
            ax.plot(x[valid], y[valid], "s", color=_C[3], markersize=3, alpha=0.5, zorder=3)
            yma = ma(y[valid].reset_index(drop=True), window)
            ax.plot(x[valid], yma.values, color=_C[3], linewidth=1.2,
                    linestyle="--", label="val loss (MA)")
            has_data = True

    if not has_data:
        ax.text(0.5, 0.5, "No validation data", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontweight="bold")
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(h, l, loc="upper right")


def panel_lr(ax, train_df: pd.DataFrame, window: int, title: str = "Learning Rate") -> None:
    if "step" not in train_df.columns or "lr" not in train_df.columns:
        ax.text(0.5, 0.5, "No LR data", ha="center", va="center", transform=ax.transAxes)
        return
    x = train_df["step"]
    y = train_df["lr"]
    valid = y.notna() & (y > 0)
    if valid.any():
        ax.plot(x[valid], y[valid], color=_C[4], linewidth=1.4, label="lr")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.set_xlabel("Step")
    ax.set_ylabel("LR")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best")


def panel_grad(ax, train_df: pd.DataFrame, window: int, title: str = "Gradient Norm") -> None:
    if "step" not in train_df.columns or "grad_norm" not in train_df.columns:
        ax.text(0.5, 0.5, "No grad data", ha="center", va="center", transform=ax.transAxes)
        return
    x = train_df["step"]
    y = train_df["grad_norm"]
    valid = y.notna() & (y >= 0)
    if valid.any():
        _raw_and_ma(ax, x, y, window, _C[5], "|∇|")
    ax.set_xlabel("Step")
    ax.set_ylabel("|grad|")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best")


def _plot_p_eos_series(
    ax2,
    x: pd.Series,
    y: pd.Series,
    window: int,
    color: str,
    label: str,
    *,
    linestyle: str = "--",
) -> bool:
    """P(EOS) 類序列畫在右軸（raw 淡線 + MA）。"""
    valid = y.notna()
    xv, yv = x[valid], y[valid]
    if len(xv) == 0:
        return False
    ax2.plot(xv, yv, color=color, linewidth=0.5, alpha=0.3)
    yma = ma(yv.reset_index(drop=True), window)
    ax2.plot(
        xv,
        yma.values,
        color=color,
        linewidth=1.4,
        label=f"{label} (MA{window})",
        linestyle=linestyle,
    )
    return True


def _fcp_delta_refline(ax2, fcp_delta: float) -> None:
    if fcp_delta > 0:
        ax2.axhline(
            fcp_delta,
            color="0.45",
            linewidth=0.9,
            linestyle=":",
            label=f"δ={fcp_delta:g}",
            zorder=1,
        )


def panel_fcp(
    ax,
    train_df: pd.DataFrame,
    window: int,
    *,
    fcp_delta: float = 0.01,
    title: str = "Train FCP — EOS Penalty",
) -> None:
    if "step" not in train_df.columns:
        ax.text(0.5, 0.5, "No FCP data", ha="center", va="center", transform=ax.transAxes)
        return
    x = train_df["step"]
    has = False
    ax2 = None

    if "fcp_penalty" in train_df.columns:
        y = train_df["fcp_penalty"]
        if y.notna().any():
            _raw_and_ma(ax, x, y, window, _C[6], "fcp_penalty")
            has = True

    need_ax2 = (
        ("eos_prob" in train_df.columns and train_df["eos_prob"].notna().any())
        or ("eos_prob_max" in train_df.columns and train_df["eos_prob_max"].notna().any())
    )
    if need_ax2:
        ax2 = ax.twinx()
        ax2.set_ylabel("P(EOS) in think", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.spines["right"].set_visible(True)
        _fcp_delta_refline(ax2, fcp_delta)
        if "eos_prob" in train_df.columns:
            if _plot_p_eos_series(ax2, x, train_df["eos_prob"], window, _C[7], "P(EOS)", linestyle="--"):
                has = True
        if "eos_prob_max" in train_df.columns:
            if _plot_p_eos_series(
                ax2, x, train_df["eos_prob_max"], window, _C[10], "P(EOS)_max", linestyle="-."
            ):
                has = True

    if not has:
        ax.text(0.5, 0.5, "No FCP data", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Step")
    ax.set_ylabel("Penalty")
    ax.set_title(title, fontweight="bold")
    if ax2 is not None:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7)
    elif has:
        ax.legend(loc="best")


def panel_val_fcp(
    ax,
    val_df: pd.DataFrame,
    window: int,
    *,
    fcp_delta: float = 0.01,
    title: str = "Val FCP / P(EOS)",
) -> None:
    if "step" not in val_df.columns or len(val_df) == 0:
        ax.text(0.5, 0.5, "No val FCP data", ha="center", va="center", transform=ax.transAxes)
        return
    x = val_df["step"]
    has = False
    ax2 = None

    if "val_fcp_penalty" in val_df.columns:
        y = val_df["val_fcp_penalty"]
        if y.notna().any():
            valid = y.notna()
            ax.plot(x[valid], y[valid], "o", color=_C[6], markersize=3.5, alpha=0.55, zorder=3)
            yma = ma(y[valid].reset_index(drop=True), window)
            ax.plot(
                x[valid],
                yma.values,
                color=_C[6],
                linewidth=1.6,
                label=f"val_fcp (MA{window})",
            )
            has = True

    need_ax2 = False
    for col in ("val_eos_prob", "val_eos_prob_max"):
        if col in val_df.columns and val_df[col].notna().any():
            need_ax2 = True
            break
    if need_ax2:
        ax2 = ax.twinx()
        ax2.set_ylabel("P(EOS) in think", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.spines["right"].set_visible(True)
        _fcp_delta_refline(ax2, fcp_delta)
        if "val_eos_prob" in val_df.columns:
            valid = val_df["val_eos_prob"].notna()
            if valid.any():
                ax2.plot(
                    x[valid],
                    val_df.loc[valid, "val_eos_prob"],
                    "s",
                    color=_C[7],
                    markersize=3,
                    alpha=0.5,
                    zorder=3,
                )
                yma = ma(val_df.loc[valid, "val_eos_prob"].reset_index(drop=True), window)
                ax2.plot(
                    x[valid],
                    yma.values,
                    color=_C[7],
                    linewidth=1.4,
                    linestyle="--",
                    label=f"val P(EOS) (MA{window})",
                )
                has = True
        if "val_eos_prob_max" in val_df.columns:
            valid = val_df["val_eos_prob_max"].notna()
            if valid.any():
                ax2.plot(
                    x[valid],
                    val_df.loc[valid, "val_eos_prob_max"],
                    "^",
                    color=_C[10],
                    markersize=3,
                    alpha=0.55,
                    zorder=3,
                )
                yma = ma(val_df.loc[valid, "val_eos_prob_max"].reset_index(drop=True), window)
                ax2.plot(
                    x[valid],
                    yma.values,
                    color=_C[10],
                    linewidth=1.4,
                    linestyle="-.",
                    label=f"val P(EOS)_max (MA{window})",
                )
                has = True

    if not has:
        ax.text(0.5, 0.5, "No val FCP data", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Step")
    ax.set_ylabel("Val penalty")
    ax.set_title(title, fontweight="bold")
    if ax2 is not None:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7)
    elif has:
        ax.legend(loc="best")


def panel_sftgo(ax, train_df: pd.DataFrame, window: int, title: str = "SFT-GO: Weighted vs CE") -> None:
    if "step" not in train_df.columns or "sftgo_loss" not in train_df.columns:
        ax.text(0.5, 0.5, "No SFT-GO data", ha="center", va="center", transform=ax.transAxes)
        return
    x = train_df["step"]
    y = train_df["sftgo_loss"]
    if y.notna().any():
        _raw_and_ma(ax, x, y, window, _C[8], "sftgo_loss")
    if "ce_loss" in train_df.columns:
        y2 = train_df["ce_loss"]
        if y2.notna().any():
            yma = ma(y2[y2.notna()].reset_index(drop=True), window)
            ax.plot(x[y2.notna()], yma.values, color=_C[1], linewidth=1.0,
                    linestyle="--", alpha=0.7, label=f"CE baseline (MA{window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best")


def panel_scale(ax, train_df: pd.DataFrame, window: int, title: str = "SCALe — Think Weight") -> None:
    if "step" not in train_df.columns or "scale_w" not in train_df.columns:
        ax.text(0.5, 0.5, "No SCALe data", ha="center", va="center", transform=ax.transAxes)
        return
    x = train_df["step"]
    y = train_df["scale_w"]
    if y.notna().any():
        xv, yv = x[y.notna()], y[y.notna()]
        ax.fill_between(xv, 1.0, yv, where=(yv < 1.0),
                        color=_C[9], alpha=0.15, label="below neutral")
        ax.plot(xv, yv, color=_C[9], linewidth=1.5, label="η_think")
        ax.axhline(1.0, color="0.5", linewidth=0.8, linestyle="--", label="neutral")
        ax.set_ylim(0.0, max(1.1, float(yv.max()) * 1.05))
    ax.set_xlabel("Step")
    ax.set_ylabel("η (think weight)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best")


def panel_final_enhance(
    ax,
    train_df: pd.DataFrame,
    window: int,
    val_df: pd.DataFrame | None = None,
    title: str = "Final Enhance — PDL / InfoEntropy",
) -> None:
    """PDL weight mean (left axis) + InfoEntropy weight mean (right axis).

    Both are per-step batch means over the final region (1.0 = component off /
    region empty). Lets you watch how the dynamic IE weight and the static-but-
    batch-dependent PDL weight evolve over training.
    """
    has_pdl = "pdl_weight_mean" in train_df.columns
    has_ie = "ie_weight_mean" in train_df.columns
    if "step" not in train_df.columns or not (has_pdl or has_ie):
        ax.text(0.5, 0.5, "No Final-Enhance data", ha="center", va="center",
                transform=ax.transAxes)
        return
    x = train_df["step"]
    ax.axhline(1.0, color="0.5", linewidth=0.8, linestyle="--", label="neutral (1.0)")

    if has_pdl:
        y = train_df["pdl_weight_mean"]
        _raw_and_ma(ax, x, y, window, _C[11], "pdl_weight_mean (train)")
        if val_df is not None and "val_pdl_weight_mean" in val_df.columns:
            vv = val_df["val_pdl_weight_mean"]
            if vv.notna().any():
                ax.plot(val_df["step"][vv.notna()], vv[vv.notna()], color=_C[11],
                        linestyle=":", marker="o", markersize=3, linewidth=1.2,
                        label="pdl (val)")
    ax.set_xlabel("Step")
    ax.set_ylabel("PDL weight mean", color=_C[11])
    ax.tick_params(axis="y", labelcolor=_C[11])
    ax.set_title(title, fontweight="bold")

    if has_ie:
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        y2 = train_df["ie_weight_mean"]
        _raw_and_ma(ax2, x, y2, window, _C[12], "ie_weight_mean (train)")
        if val_df is not None and "val_ie_weight_mean" in val_df.columns:
            vv = val_df["val_ie_weight_mean"]
            if vv.notna().any():
                ax2.plot(val_df["step"][vv.notna()], vv[vv.notna()], color=_C[12],
                         linestyle=":", marker="s", markersize=3, linewidth=1.2,
                         label="ie (val)")
        ax2.set_ylabel("InfoEntropy weight mean", color=_C[12])
        ax2.tick_params(axis="y", labelcolor=_C[12])
        # merge legends from both axes
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    else:
        ax.legend(loc="best", fontsize=8)


def panel_struct_acc(
    ax,
    train_df: pd.DataFrame,
    window: int,
    title: str = "Structural Token Top-1 Accuracy",
) -> None:
    """Plot per-step top-1 accuracy for </think>, </final>, <|im_end|>.

    Each series is NaN when the token did not appear in the batch; values range
    [0, 1].  A horizontal dashed line at y=1.0 marks perfect accuracy.
    The panel is shown only when at least one accuracy column exists.
    """
    _spec = [
        ("think_end_acc",  _C[13], "-",  "</think> acc"),
        ("final_end_acc",  _C[14], "--", "</final> acc"),
        ("im_end_acc",     _C[15], "-.", "<|im_end|> acc"),
    ]
    if "step" not in train_df.columns:
        ax.text(0.5, 0.5, "No struct-acc data", ha="center", va="center",
                transform=ax.transAxes)
        return
    x = train_df["step"]
    has = False
    for col, color, ls, label in _spec:
        if col not in train_df.columns:
            continue
        y = train_df[col]
        valid = y.notna()
        if not valid.any():
            continue
        xv, yv = x[valid], y[valid]
        # raw (faint)
        ax.plot(xv, yv, color=color, linewidth=0.4, alpha=0.2, linestyle=ls)
        # MA
        yma = ma(yv.reset_index(drop=True), window)
        ax.plot(xv, yma.values, color=color, linewidth=1.6,
                linestyle=ls, label=f"{label} (MA{window})")
        has = True

    if not has:
        ax.text(0.5, 0.5, "No struct-acc data", ha="center", va="center",
                transform=ax.transAxes)
        return

    ax.axhline(1.0, color="0.5", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_ylim(-0.05, 1.08)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Step")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)


def panel_aux_loss(
    ax,
    train_df: pd.DataFrame,
    window: int,
    title: str = "MoE Auxiliary Loss  (L_total − CE)",
) -> None:
    """Plot MoE auxiliary loss = L_total - CE_loss (lb_loss + z_loss contribution).

    Left axis: aux_loss (MA smoothed).
    Right axis: L_total and CE_loss for context.
    Helps diagnose router instability — aux should shrink as routing stabilises.
    """
    if "loss" not in train_df.columns or "ce_loss" not in train_df.columns:
        ax.text(0.5, 0.5, "No aux-loss data", ha="center", va="center",
                transform=ax.transAxes)
        return
    x = train_df["step"]
    aux = (train_df["loss"] - train_df["ce_loss"]).clip(lower=0)
    valid = aux.notna() & train_df["loss"].notna() & train_df["ce_loss"].notna()
    if not valid.any():
        ax.text(0.5, 0.5, "No aux-loss data", ha="center", va="center",
                transform=ax.transAxes)
        return

    # Left axis — aux loss
    _raw_and_ma(ax, x, aux, window, "#e67e22", "aux (MoE lb+z)")   # orange
    ax.set_ylabel("Aux Loss (MoE)", color="#e67e22")
    ax.tick_params(axis="y", labelcolor="#e67e22")
    ax.axhline(0, color="0.7", linewidth=0.7, linestyle=":")

    # Right axis — L_total and CE for reference
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    _raw_and_ma(ax2, x, train_df["loss"],    window, _C[0], "L_total",  linestyle="--")
    _raw_and_ma(ax2, x, train_df["ce_loss"], window, _C[1], "CE",       linestyle=":")
    ax2.set_ylabel("Loss", fontsize=8)
    ax2.tick_params(labelsize=7)

    ax.set_xlabel("Step")
    ax.set_title(title, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7)


def panel_router(ax, train_df: pd.DataFrame, title: str = "MoE Router Temp") -> None:
    if "step" not in train_df.columns or "router_temp" not in train_df.columns:
        ax.text(0.5, 0.5, "No router data", ha="center", va="center", transform=ax.transAxes)
        return
    x = train_df["step"]
    y = train_df["router_temp"]
    valid = y.notna()
    if valid.any():
        ax.plot(x[valid], y[valid], color="saddlebrown", linewidth=1.2,
                drawstyle="steps-post", label="T_router")
    ax.set_xlabel("Step")
    ax.set_ylabel("Temperature")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best")


# ---------------------------------------------------------------------------
# Terminal helpers (ASCII charts & summary)
# ---------------------------------------------------------------------------

# ANSI codes — gracefully ignored if terminal doesn't support them
_A = {
    "rst":  "\033[0m",
    "bold": "\033[1m",
    "dim":  "\033[2m",
    "G":    "\033[32m",   # green   — improvement
    "R":    "\033[31m",   # red     — regression
    "C":    "\033[36m",   # cyan    — highlight
    "Y":    "\033[33m",   # yellow  — warning / neutral
    "M":    "\033[35m",   # magenta
    "B":    "\033[34m",   # blue
    "W":    "\033[37m",   # white
}


def _ansi(key: str, text: str) -> str:
    return f"{_A[key]}{text}{_A['rst']}"


def _bar(value: float, lo: float, hi: float, width: int = 20, char: str = "█") -> str:
    """ASCII progress bar scaled between lo and hi."""
    if hi <= lo:
        frac = 1.0
    else:
        frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = round(frac * width)
    return char * filled + "░" * (width - filled)


def _sparkline(series: "pd.Series | np.ndarray", width: int = 32) -> str:
    """Single-line block-character sparkline sampled to `width` chars."""
    _BLOCKS = " ▁▂▃▄▅▆▇█"
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return "─" * width
    if len(arr) > width:
        idx = np.linspace(0, len(arr) - 1, width).astype(int)
        arr = arr[idx]
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        return _BLOCKS[4] * len(arr)
    norm = (arr - lo) / (hi - lo)
    return "".join(_BLOCKS[int(round(v * (len(_BLOCKS) - 1)))] for v in norm)


def _fmt_duration(seconds: float) -> str:
    """Format seconds as Xh Xm Xs."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _infer_total_steps(train_df: pd.DataFrame) -> int | None:
    """Infer total training steps by inverting the cosine LR schedule from the CSV.

    Solves:  lr = 0.5 * lr_max * (1 + cos(π * progress))
             where progress = (cur_step - warmup) / (total - warmup)
    Falls back to dataset .bin file size if LR column is missing.
    """
    if "lr" not in train_df.columns or "step" not in train_df.columns:
        return None

    lr_s  = train_df["lr"].dropna()
    step_s = train_df.loc[lr_s.index, "step"]

    if len(lr_s) < 10:
        return None

    lr_max = float(lr_s.max())
    if lr_max <= 0:
        return None

    # End of warmup = first row where LR ≥ 99 % of peak
    warmup_mask = lr_s >= lr_max * 0.99
    if not warmup_mask.any():
        return None
    warmup_step = int(step_s[warmup_mask.idxmax()])

    # Use the last few rows (averaged) for a stable estimate of current LR
    tail = lr_s.iloc[-min(20, len(lr_s)):]
    lr_cur = float(tail.mean())
    cur_step = int(step_s.iloc[-1])

    # Guard: still in warmup or at peak
    if cur_step <= warmup_step or lr_cur >= lr_max * 0.995:
        return None

    # Cosine inversion (lr_min ≈ 0)
    cos_arg = max(-1.0, min(1.0, 2.0 * lr_cur / lr_max - 1.0))
    progress = math.acos(cos_arg) / math.pi  # fraction of post-warmup completed

    if progress <= 0.0 or progress >= 1.0:
        return None

    total = int(warmup_step + (cur_step - warmup_step) / progress)

    # Sanity: must exceed current step by at least 5 %
    if total <= int(cur_step * 1.05):
        return None

    return total


def _print_terminal_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    window: int,
    features: dict[str, bool],
    *,
    total_steps: int | None = None,
) -> None:
    """Print a rich, coloured numeric summary with sparklines and PPL."""
    import os
    try:
        W = min(os.get_terminal_size().columns, 88)
    except OSError:
        W = 76

    SP = 28  # sparkline width

    def _smooth(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        if len(valid) < 2:
            return valid
        return valid.rolling(min(window, len(valid)), min_periods=1, center=True).mean()

    def _ppl(loss_val: float) -> str:
        return f"{math.exp(min(loss_val, 20.0)):.2f}"

    def _delta_str(first: float, last: float, w: int = 9) -> str:
        pct = (last - first) / (abs(first) + 1e-9) * 100
        arrow = "↓" if pct < -0.01 else ("↑" if pct > 0.01 else "→")
        color = "G" if pct < -0.01 else ("R" if pct > 0.01 else "W")
        return _ansi(color, f"{arrow}{abs(pct):.1f}%".rjust(w))

    def _box_line(text: str) -> str:
        inner = W - 6
        return f"║  {text:<{inner}}  ║"

    def _metric_row(
        label: str,
        series: pd.Series,
        *,
        ppl_col: bool = False,
        val: bool = False,
        is_pct: bool = False,
    ) -> None:
        """Print one metric row: label  start→last  Δ  sparkline  [PPL]."""
        y_raw = series.dropna()
        if len(y_raw) < 1:
            return
        ys = _smooth(y_raw)
        first_v = float(ys.iloc[0])
        last_v  = float(ys.iloc[-1])
        min_v   = float(y_raw.min())

        color = "G" if last_v < first_v else "R"
        last_fmt = _ansi(color, f"{last_v:.4f}")
        delta_fmt = _delta_str(first_v, last_v)

        spark = _sparkline(_smooth(y_raw), width=SP)
        spark_col = "G" if last_v < first_v else "R"
        spark_fmt = _ansi(spark_col, spark)

        if ppl_col:
            ppl_fmt = "  " + _ansi("C", f"PPL {_ppl(last_v)}")
        else:
            ppl_fmt = ""

        fmt = "%" if is_pct else ""
        print(
            f"  {label:<13} {first_v:.4f}{fmt}→{last_fmt}{fmt}"
            f"  {delta_fmt}  {spark_fmt}{ppl_fmt}"
        )

    border_top = "╔" + "═" * (W - 2) + "╗"
    border_bot = "╚" + "═" * (W - 2) + "╝"

    # Auto-detect total steps from LR cosine schedule in the CSV
    if total_steps is None:
        total_steps = _infer_total_steps(train_df)

    cur_step = int(train_df["step"].max())

    print(f"\n{_ansi('bold', border_top)}")
    title = f"SFT-CoT  ·  MA={window}  ·  step {cur_step:,}"
    if total_steps:
        pct = cur_step / total_steps * 100
        title += f" / {total_steps:,}  ({pct:.1f}%)"
    print(_ansi("bold", _box_line(title)))
    print(_ansi("bold", border_bot))

    # ---- Progress block --------------------------------------------------------
    has_tokens   = "tokens_seen" in train_df.columns
    has_steptime = "step_time_s" in train_df.columns

    if has_tokens or has_steptime:
        print(f"\n  {_ansi('bold', 'Progress')}")
        print("  " + "─" * (W - 4))

        if has_tokens:
            tok_total = float(train_df["tokens_seen"].iloc[-1])
            tok_b = tok_total / 1e9
            tok_spark = _sparkline(train_df["tokens_seen"], width=SP)
            print(f"  {'Tokens seen':<13} {_ansi('C', f'{tok_b:.4f} B')}  {_ansi('dim', tok_spark)}")

        if has_steptime:
            st = train_df["step_time_s"].dropna()
            elapsed = float(st.sum())
            avg_s   = float(st.mean())
            # throughput = tok_per_step / avg_step_time
            tok_per_step = float(train_df["tokens_seen"].iloc[-1]) / cur_step if has_tokens else None
            tps_str = ""
            if tok_per_step:
                tps = tok_per_step / avg_s
                tps_str = f"  {_ansi('Y', f'{tps:,.0f} tok/s')}"
            print(f"  {'Elapsed':<13} {_ansi('C', _fmt_duration(elapsed))}  avg {avg_s:.1f}s/step{tps_str}")

            if total_steps and cur_step > 0:
                remaining = total_steps - cur_step
                eta_s = remaining * avg_s
                # progress bar width = W - 4 - 2
                bar_w = min(W - 20, 40)
                pct = cur_step / total_steps
                filled = round(pct * bar_w)
                bar = _ansi("G", "█" * filled) + _ansi("dim", "░" * (bar_w - filled))
                print(f"  {'ETA':<13} {_ansi('Y', _fmt_duration(eta_s))}  ({pct*100:.1f}%)")
                print(f"  {bar}")

        if has_tokens and has_steptime and total_steps:
            # Epoch info
            # rough steps_per_epoch from total
            tok_total_ds = float(train_df["tokens_seen"].iloc[-1])
            tok_per_step_v = tok_total_ds / cur_step
            # try to derive steps_per_epoch assuming EPOCHS=16
            steps_per_epoch = total_steps // 16
            if steps_per_epoch > 0:
                cur_epoch = (cur_step - 1) // steps_per_epoch + 1
                step_in_epoch = (cur_step - 1) % steps_per_epoch + 1
                print(f"  {'Epoch':<13} {_ansi('C', f'{cur_epoch}/16')}  step {step_in_epoch:,}/{steps_per_epoch:,} in epoch")

    # ---- Training losses -------------------------------------------------------
    print(f"\n  {_ansi('bold', 'Training')}")
    print("  " + "─" * (W - 4))
    for col, label, is_ppl in [("loss", "L_total", False), ("ce_loss", "CE loss", True)]:
        if col in train_df.columns:
            _metric_row(label, train_df[col], ppl_col=is_ppl)

    # ---- LR -------------------------------------------------------------------
    if "lr" in train_df.columns:
        lr = train_df["lr"].dropna()
        if len(lr) > 1:
            spark = _sparkline(lr, width=SP)
            print(
                f"  {'LR':<13} {lr.iloc[0]:.2e}→{_ansi('Y', f'{lr.iloc[-1]:.2e}')}"
                f"           {_ansi('M', spark)}"
            )

    # ---- Grad norm ------------------------------------------------------------
    if "grad_norm" in train_df.columns:
        gn = train_df["grad_norm"].dropna()
        if len(gn) > 1:
            spark = _sparkline(gn, width=SP)
            print(
                f"  {'Grad norm':<13} avg {gn.mean():.3f}  max {_ansi('Y', f'{gn.max():.3f}')}"
                f"      {_ansi('Y', spark)}"
            )

    # ---- Optional features ---------------------------------------------------
    feat_rows: list[tuple[str, str, pd.Series]] = []
    if features.get("fcp") and "fcp_penalty" in train_df.columns:
        feat_rows.append(("FCP penalty", "fcp_penalty", train_df["fcp_penalty"]))
    if features.get("sftgo") and "sftgo_loss" in train_df.columns:
        feat_rows.append(("SFT-GO loss", "sftgo_loss", train_df["sftgo_loss"]))
    if features.get("scale") and "scale_w" in train_df.columns:
        feat_rows.append(("SCALe η", "scale_w", train_df["scale_w"]))
    if features.get("struct_acc"):
        for col, lbl in [("think_end_acc", "</think> acc"), ("final_end_acc", "</final> acc"), ("im_end_acc", "<|im_end|> acc")]:
            if col in train_df.columns:
                feat_rows.append((lbl, col, train_df[col]))

    if feat_rows:
        print(f"\n  {_ansi('bold', 'Features')}")
        print("  " + "─" * (W - 4))
        for label, _, series in feat_rows:
            _metric_row(label, series)

    # ---- Validation ----------------------------------------------------------
    if len(val_df) > 1 and "step" in val_df.columns:
        print(f"\n  {_ansi('bold', 'Validation')}")
        print("  " + "─" * (W - 4))
        for col, label, is_ppl in [("val_ce_loss", "val CE", True), ("val_loss_mean", "val loss", False)]:
            if col not in val_df.columns:
                continue
            y = val_df[col].dropna()
            if len(y) < 1:
                continue
            min_v, last_v = float(y.min()), float(y.iloc[-1])
            is_best = abs(last_v - min_v) < 1e-6
            star = _ansi("Y", " ★") if is_best else ""
            spark = _sparkline(y, width=SP)
            spark_col = "G" if last_v <= min_v else "W"
            ppl_s = ("  " + _ansi("C", f"PPL {_ppl(last_v)}")) if is_ppl else ""
            print(
                f"  {label:<13} min={min_v:.4f}  last={_ansi('G', f'{last_v:.4f}')}{star}"
                f"  {_ansi(spark_col, spark)}{ppl_s}"
            )

    print(f"\n  {_ansi('dim', '─' * (W - 4))}\n")


def _plot_terminal(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    window: int,
    features: dict[str, bool],
    *,
    total_steps: int | None = None,
) -> None:
    """Print stats + sparklines, then PPL-only ASCII charts via plotext."""
    # ---- Text stats + sparklines (always) ------------------------------------
    _print_terminal_stats(train_df, val_df, window, features, total_steps=total_steps)

    # ---- PPL ASCII charts via plotext ----------------------------------------
    try:
        import plotext as plt_t
    except ImportError:
        print(_ansi("Y", "  [!] pip install plotext  to see PPL charts\n"))
        return

    x = train_df["step"].values
    x_val = val_df["step"].values if len(val_df) > 0 else np.array([])

    def _smooth_np(arr: np.ndarray, w: int) -> np.ndarray:
        if len(arr) < 2:
            return arr
        return pd.Series(arr).rolling(min(w, len(arr)), min_periods=1, center=True).mean().values

    def _to_ppl(arr: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(arr, None, 20.0))

    def _setup(title: str) -> None:
        plt_t.clear_figure()
        plt_t.theme("dark")
        plt_t.title(title)
        plt_t.xlabel("Step")
        plt_t.ylabel("PPL")
        plt_t.canvas_color("black")
        plt_t.axes_color("black")
        plt_t.ticks_color("white")

    panels: list[tuple[str, "callable[[], None]"]] = []

    # Train PPL
    if "ce_loss" in train_df.columns and train_df["ce_loss"].notna().any():
        def _train_ppl() -> None:
            y = train_df["ce_loss"].values
            valid = ~np.isnan(y)
            if valid.sum() < 2:
                return
            _setup(f"Train PPL = exp(CE)  ·  MA{window}")
            ppl_ma = _to_ppl(_smooth_np(y[valid], window))
            plt_t.plot(x[valid], ppl_ma, label=f"PPL MA{window}", color="red+")
            plt_t.show()
        panels.append(("Train PPL", _train_ppl))

    # Val PPL
    if len(x_val) > 0 and "val_ce_loss" in val_df.columns and val_df["val_ce_loss"].notna().any():
        def _val_ppl() -> None:
            y = val_df["val_ce_loss"].values
            valid = ~np.isnan(y)
            if valid.sum() < 1:
                return
            _setup(f"Val PPL = exp(val CE)  ·  MA{window}")
            ppl_pts = _to_ppl(y[valid])
            plt_t.scatter(x_val[valid], ppl_pts, label="val PPL", color="green+")
            if valid.sum() > 1:
                ppl_ma = _to_ppl(_smooth_np(y[valid], window))
                plt_t.plot(x_val[valid], ppl_ma, label=f"val PPL MA{window}", color="green+")
            plt_t.show()
        panels.append(("Val PPL", _val_ppl))

    if not panels:
        return

    total = len(panels)
    for idx, (name, fn) in enumerate(panels, start=1):
        print(_ansi("bold", f"\n  ── {name} ({'chart ' + str(idx) + '/' + str(total)}) {'─' * 32}"))
        fn()
        if idx < total:
            try:
                ans = input(_ansi("dim", "\n  [Enter] next  ·  [q] quit  "))
                if ans.strip().lower() == "q":
                    break
            except EOFError:
                break
    print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_sft_train_val(
    train_csv: Path,
    val_csv: Optional[Path] = None,
    output: Optional[Path] = None,
    window: int = 300,
    width: float = 12.0,
    dpi: int = 200,
    no_ppl: bool = False,
    fcp_delta: float = 0.01,
    show: bool = False,
    terminal: bool = False,
    total_steps: Optional[int] = None,
) -> Path:
    """Plot SFT training + validation curves from CSV logs.

    Parameters
    ----------
    train_csv : Path
        Training log CSV.
    val_csv : Path, optional
        Validation log CSV.
    output : Path, optional
        Output image path. Auto-generated from train_csv stem if None.
    window : int
        Moving-average window (default 300).
    width : float
        Figure width in inches (default 12.0).
    dpi : int
        Output DPI (default 200).
    no_ppl : bool
        Reserved (not currently implemented).
    fcp_delta : float
        FCP delta reference line for P(EOS) axis (default 0.01).
    show : bool
        Open interactive matplotlib window after saving.
    terminal : bool
        Display ASCII charts and summary in terminal using plotext.

    Returns
    -------
    Path
        The saved image file path.
    """
    # Apply paper style
    plt.rcParams.update(_PAPER_RC)

    # Load data
    print(f"Loading {train_csv}...")
    train_df = load_csv(train_csv)
    print(f"  {len(train_df)} rows  |  columns: {list(train_df.columns)}")

    val_df: pd.DataFrame
    if val_csv is not None and val_csv.is_file():
        print(f"Loading {val_csv}...")
        val_df = load_csv(val_csv)
        print(f"  {len(val_df)} rows")
        val_name = val_csv.name
    else:
        if val_csv is not None:
            print(f"Warning: val CSV not found at {val_csv}, skipping.")
        val_df = pd.DataFrame(columns=["step"])
        val_name = "—"

    features = detect_features(train_df, val_df)
    print(
        f"Detected: FCP={features['fcp']}  eos_prob_max={features['fcp_max']}  "
        f"val_FCP={features['val_fcp']}  SFT-GO={features['sftgo']}  SCALe={features['scale']}  "
        f"FinalEnhance={features['final_enhance']}"
    )
    print(f"MA window: {window}")

    # ---- Build panel list ------------------------------------------------
    has_router = "router_temp" in train_df.columns and train_df["router_temp"].notna().any()
    n_optional = sum([
        features["fcp"], features["sftgo"], features["scale"],
        features["final_enhance"], features["struct_acc"], features["aux_loss"],
    ])

    ncols = 2
    nrows = 2 + n_optional + (1 if has_router else 0)

    fig_h = nrows * 2.6 + 0.6

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width, fig_h),
        constrained_layout=True,
    )
    if nrows == 1:
        axes = axes.reshape(1, ncols)

    fig.suptitle(
        f"{train_csv.stem}  |  val: {val_name}  |  MA={window}",
        fontsize=10,
        fontweight="bold",
        y=1.005,
    )

    row = 0
    panel_train_loss(axes[row, 0], train_df, window, "Training Loss")
    panel_validation(axes[row, 1], val_df, window, "Validation Loss")
    row += 1

    panel_lr(axes[row, 0], train_df, window, "Learning Rate")
    panel_grad(axes[row, 1], train_df, window, "Gradient Norm")
    row += 1

    if features["fcp"]:
        panel_fcp(axes[row, 0], train_df, window, fcp_delta=fcp_delta,
                  title="Train FCP — EOS Penalty")
        if features["val_fcp"]:
            panel_val_fcp(axes[row, 1], val_df, window, fcp_delta=fcp_delta,
                          title="Val FCP / P(EOS)")
        else:
            axes[row, 1].axis("off")
        row += 1

    if features["sftgo"]:
        panel_sftgo(axes[row, 0], train_df, window, "SFT-GO: Weighted vs CE")
        axes[row, 1].axis("off")
        row += 1

    if features["scale"]:
        panel_scale(axes[row, 0], train_df, window, "SCALe — Think Weight (η)")
        axes[row, 1].axis("off")
        row += 1

    if features["final_enhance"]:
        panel_final_enhance(axes[row, 0], train_df, window, val_df,
                            "Final Enhance — PDL / InfoEntropy")
        axes[row, 1].axis("off")
        row += 1

    if features["struct_acc"]:
        panel_struct_acc(axes[row, 0], train_df, window,
                         "Structural Token Accuracy  (</think> / </final> / <|im_end|>)")
        axes[row, 1].axis("off")
        row += 1

    if features["aux_loss"]:
        panel_aux_loss(axes[row, 0], train_df, window,
                       "MoE Auxiliary Loss  (L_total − CE)")
        axes[row, 1].axis("off")
        row += 1

    if has_router:
        panel_router(axes[row, 0], train_df, "MoE Router Temperature")
        axes[row, 1].axis("off")
        row += 1

    # Output path
    if output is None:
        stem = train_csv.stem.replace("_log", "").replace("train_", "")
        output = train_csv.parent / f"{stem}_plots.png"

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    print(f"\nSaved -> {output.resolve()}")

    # Interactive window
    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"[!] Cannot show interactive plot: {e}")

    # Terminal charts
    if terminal:
        _plot_terminal(train_df, val_df, window, features, total_steps=total_steps)

    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Publication-quality plot for train + val CSV with MA smoothing and PPL."
    )
    ap.add_argument("train_csv", type=Path, help="Training log CSV")
    ap.add_argument("val_csv", type=Path, nargs="?", default=None,
                    help="Validation log CSV (optional)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output image path (default: auto-named alongside train_csv)")
    ap.add_argument("--ma", dest="window", type=int, default=300,
                    help="Moving-average window (default: 200)")
    ap.add_argument("-w", "--width", type=float, default=12.0,
                    help="Figure width in inches (default: 12)")
    ap.add_argument("-dpi", "--dpi", type=int, default=200,
                    help="Output DPI (default: 200)")
    ap.add_argument("--no-ppl", action="store_true",
                    help="Disable the PPL secondary axis")
    ap.add_argument(
        "--fcp-delta",
        type=float,
        default=0.01,
        help="FCP delta reference line (default 0.01)",
    )
    ap.add_argument("--show", "-s", action="store_true", default=False,
                    help="Open interactive plot window (requires GUI backend)")
    ap.add_argument("--term", "-t", action="store_true", default=False, dest="term",
                    help="Display ASCII charts in terminal (requires plotext)")
    ap.add_argument("--terminal", action="store_true", default=False, dest="term",
                    help=argparse.SUPPRESS)  # hidden alias for --term
    ap.add_argument("--total-steps", type=int, default=None, dest="total_steps",
                    help="Target total training steps for ETA/progress display (auto-detected if omitted)")
    args = ap.parse_args()

    plot_sft_train_val(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output=args.output,
        window=args.window,
        width=args.width,
        dpi=args.dpi,
        no_ppl=args.no_ppl,
        fcp_delta=args.fcp_delta,
        show=args.show,
        terminal=args.term,
        total_steps=args.total_steps,
    )


if __name__ == "__main__":
    main()
