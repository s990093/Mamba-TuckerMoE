#!/usr/bin/env python3
"""Plot speedup chart from `metal/results/moe_unified_benchmark.csv`."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="metal/results/moe_unified_benchmark.csv",
        help="Input benchmark CSV",
    )
    ap.add_argument(
        "--out",
        default="metal/results/moe_unified_speedup.png",
        help="Output plot path",
    )
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(
            "matplotlib is required. Install with: .venv/bin/pip install matplotlib"
        ) from e

    rows: list[dict[str, str]] = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"No rows found in {args.csv}")

    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in rows:
        suite = r["suite"]
        method = r["method"]
        speedup = float(r["speedup_vs_baseline"])
        grouped[suite].append((method, speedup))

    suites = sorted(grouped.keys())
    fig, axes = plt.subplots(
        nrows=len(suites),
        ncols=1,
        figsize=(12, max(4, 2.6 * len(suites))),
        constrained_layout=True,
    )
    if len(suites) == 1:
        axes = [axes]

    for ax, suite in zip(axes, suites):
        methods = [m for m, _ in grouped[suite]]
        vals = [v for _, v in grouped[suite]]
        colors = ["#4f8ef7" if v >= 1.0 else "#e85c5c" for v in vals]
        bars = ax.bar(methods, vals, color=colors)
        ax.axhline(1.0, color="#888", linestyle="--", linewidth=1.0)
        ax.set_title(f"{suite} speedup vs baseline")
        ax.set_ylabel("x")
        ax.tick_params(axis="x", rotation=18)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + max(0.01, 0.02 * max(vals)),
                f"{v:.2f}x",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ymax = max(max(vals) * 1.25, 1.2)
        ax.set_ylim(0, ymax)

    fig.suptitle("MoE Unified Benchmark Speedup", fontsize=14, fontweight="bold")
    fig.savefig(args.out, dpi=180)
    print(f"Wrote plot: {args.out}")


if __name__ == "__main__":
    main()

