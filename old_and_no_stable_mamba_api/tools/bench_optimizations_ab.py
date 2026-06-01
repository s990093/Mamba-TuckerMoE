#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B benchmark for system-level inference optimizations:
  1. Tucker Einsum Fuse  — eliminates gather/scatter by fusing expert selection into a single einsum
  2. Lookahead Router    — prefetches expert weights by routing from the previous layer's hidden state

Runs each configuration multiple times and reports a comparison table + optional bar chart.

Usage (from repo root):
  python inference/tools/bench_optimizations_ab.py
  python inference/tools/bench_optimizations_ab.py --decode-tokens 256 --trials 5 --dtype bf16
  python inference/tools/bench_optimizations_ab.py --chart   # save PNG bar chart
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import numpy as np

# ── repo path setup ──────────────────────────────────────────────────────────
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_INF_ROOT = os.path.dirname(_TOOLS_DIR)
_LIB_DIR = os.path.join(_INF_ROOT, "lib")
_REPO_ROOT = os.path.abspath(os.path.join(_INF_ROOT, ".."))
for _p in (_LIB_DIR, _INF_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_RESULTS_DIR = os.path.join(_INF_ROOT, "results")

from mlx_hybrid_infer import (
    Mamba3Config,
    Mamba3LanguageModel,
    attach_decode_compilation,
    resolve_mlx_checkpoint,
    strict_load_and_convert,
)
from benchmark_mlx import (
    _build_prompt_ids,
    _init_token_counts,
    _invalidate_tucker_caches,
    _materialize_cache_tree,
    _pad_transformer_caches,
)

# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    name: str
    lookahead_router: bool
    tucker_einsum_fuse: bool
    prefill_tps: list[float] = field(default_factory=list)
    decode_tps: list[float] = field(default_factory=list)
    prefill_ms: list[float] = field(default_factory=list)
    decode_ms: list[float] = field(default_factory=list)

    @property
    def avg_prefill_tps(self) -> float:
        return float(np.mean(self.prefill_tps)) if self.prefill_tps else 0.0

    @property
    def avg_decode_tps(self) -> float:
        return float(np.mean(self.decode_tps)) if self.decode_tps else 0.0

    @property
    def std_decode_tps(self) -> float:
        return float(np.std(self.decode_tps)) if len(self.decode_tps) > 1 else 0.0

    @property
    def avg_prefill_ms(self) -> float:
        return float(np.mean(self.prefill_ms)) if self.prefill_ms else 0.0

    @property
    def avg_decode_ms(self) -> float:
        return float(np.mean(self.decode_ms)) if self.decode_ms else 0.0


# ── Core benchmark logic ────────────────────────────────────────────────────

def _run_single_config(
    *,
    config: Mamba3Config,
    model: Mamba3LanguageModel,
    tokenizer: Any,
    prompt_ids: list[int],
    decode_tokens: int,
    warmup: int,
    target_dtype: mx.Dtype,
    kv_dtype: mx.Dtype,
    router_temp_val: float,
) -> tuple[float, float, float, float]:
    """Run prefill + decode once, return (prefill_tps, decode_tps, prefill_ms, decode_ms)."""
    router_temp = mx.array(router_temp_val, dtype=target_dtype)
    prefill_tokens = len(prompt_ids)
    max_cache_len = prefill_tokens + decode_tokens + 8

    # Attach per-layer compilation (lookahead requires per-layer, not full)
    attach_decode_compilation(
        model,
        max_cache_len=max_cache_len,
        kv_dtype=kv_dtype,
        compile_decode=True,
    )

    x_prefill = mx.array([prompt_ids], dtype=mx.int32)

    def prefill_forward(x: mx.array, rt: mx.array):
        return model(x, caches=None, seq_pos=None, router_temp=rt)

    run_prefill = mx.compile(prefill_forward)

    # Warmup
    for _ in range(warmup):
        logits, caches = run_prefill(x_prefill, router_temp)
        mx.eval(logits, caches)

    # Timed prefill
    t0 = time.perf_counter()
    logits, caches = run_prefill(x_prefill, router_temp)
    mx.eval(logits, caches)
    prefill_s = time.perf_counter() - t0
    prefill_tps = prefill_tokens / max(prefill_s, 1e-9)

    # Materialize caches
    caches = _materialize_cache_tree(caches)
    mx.eval(caches)
    caches = _pad_transformer_caches(caches, max_cache_len)
    mx.eval(caches)

    # Decode loop
    pos = prefill_tokens
    t1 = time.perf_counter()
    if decode_tokens > 0:
        row = logits[0, -1, :]
        last = mx.argmax(row, axis=-1)  # greedy for consistent benchmarking
        mx.eval(last)
        x_one = last.reshape(1, 1)
        for _ in range(decode_tokens - 1):
            pos_arr = mx.array(pos, dtype=mx.int32)
            logits_d, caches = model(
                x_one,
                caches=caches,
                seq_pos=pos_arr,
                router_temp=router_temp,
            )
            last = mx.argmax(logits_d[0, -1, :], axis=-1)
            mx.eval(last, caches)
            x_one = last.reshape(1, 1)
            pos += 1
    decode_s = time.perf_counter() - t1
    decode_tps = decode_tokens / max(decode_s, 1e-9)

    return prefill_tps, decode_tps, prefill_s * 1000, decode_s * 1000


def _build_model(
    config: Mamba3Config,
    vocab_size: int,
    target_dtype: mx.Dtype,
    checkpoint: str,
    npz_cache: str,
    force_pt: bool,
) -> Mamba3LanguageModel:
    """Build and load model weights."""
    model = Mamba3LanguageModel(config, vocab_size)
    resolved, kind = resolve_mlx_checkpoint(
        checkpoint,
        repo_root=_REPO_ROOT,
        npz_cache=npz_cache,
        force_pt=force_pt,
    )
    if resolved is None or kind == "none":
        print("  ⚠️  No checkpoint found — random weights (TPS smoke test only).")
    else:
        strict_load_and_convert(model, resolved)
    model.apply(lambda x: x.astype(target_dtype))
    mx.eval(model.parameters())
    _invalidate_tucker_caches(model)
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="A/B benchmark: Tucker einsum fuse (gather elim) vs Lookahead Router vs baseline"
    )
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--npz-cache", type=str, default="")
    p.add_argument("--force-pt", action="store_true")
    p.add_argument("--tokenizer", type=str, default=os.path.join(_INF_ROOT, "tokenizer"))
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--decode-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3, help="Warmup iterations per config")
    p.add_argument("--trials", type=int, default=3, help="Timed trials per config")
    p.add_argument("--vocab-size", type=int, default=32007)
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--kv-dtype", type=str, default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--router-temp", type=float, default=0.5)
    p.add_argument("--prompt", type=str, default="Hello! Write one short sentence about MLX on Apple Silicon.")
    p.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Do not wrap --prompt in ChatML (legacy literal tokenization).",
    )
    p.add_argument(
        "--chart",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH",
        help="Save bar chart PNG (default: inference/results/bench_optimizations_ab.png)",
    )
    p.add_argument(
        "--json",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH",
        help="Save JSON results (default: inference/results/bench_optimizations_ab.json)",
    )
    args = p.parse_args()

    compute_dtype_map = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
    target_dtype = compute_dtype_map[args.dtype]
    kv_map = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}
    kv_dtype = target_dtype if args.kv_dtype == "auto" else kv_map[args.kv_dtype]

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit("Please `pip install transformers` to load the tokenizer.") from e

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vocab_size = len(tokenizer) if args.vocab_size <= 0 else args.vocab_size
    prompt_ids = _build_prompt_ids(
        tokenizer, args.prompt, args.seq_len, chatml_user=not args.raw_prompt
    )

    # ── Define test configurations ───────────────────────────────────────
    configs = [
        {
            "name": "① Baseline",
            "lookahead_router": False,
            "tucker_einsum_fuse": False,
        },
        {
            "name": "② Einsum Fuse (Gather消除)",
            "lookahead_router": False,
            "tucker_einsum_fuse": True,
        },
        {
            "name": "③ Lookahead Router (路由預取)",
            "lookahead_router": True,
            "tucker_einsum_fuse": False,
        },
        {
            "name": "④ Both Combined (雙優化)",
            "lookahead_router": True,
            "tucker_einsum_fuse": True,
        },
    ]

    print("=" * 72)
    print("  A/B Benchmark: System-Level Inference Optimizations")
    print("=" * 72)
    print(f"  Prompt tokens : {len(prompt_ids)}")
    print(f"  Decode tokens : {args.decode_tokens}")
    print(f"  Trials/config : {args.trials}")
    print(f"  Warmup        : {args.warmup}")
    print(f"  Weight dtype  : {args.dtype}")
    print(f"  KV dtype      : {args.kv_dtype}")
    print(f"  Router temp   : {args.router_temp}")
    print("=" * 72)
    print()

    results: list[BenchResult] = []

    for ci, cfg in enumerate(configs):
        name = cfg["name"]
        la = cfg["lookahead_router"]
        ef = cfg["tucker_einsum_fuse"]

        print(f"── [{ci+1}/{len(configs)}] {name} ──")
        print(f"   lookahead_router={la}  tucker_einsum_fuse={ef}")

        # Build config
        mamba_config = Mamba3Config(
            d_model=768,
            d_state=64,
            d_head=64,
            expand=2,
            num_layers=6,
            mimo_rank=4,
            num_kv_heads=4,
            use_parallel_scan=True,
            chunk_size=64,
            use_kmoe=True,
            kmoe_num_experts=8,
            kmoe_top_k=2,
            kmoe_r1=32,
            kmoe_r2=512,
            kmoe_r3=256,
            ffn_expand=6,
        )
        mamba_config.lookahead_router = la
        mamba_config.tucker_einsum_fuse = ef

        model = _build_model(
            mamba_config,
            vocab_size,
            target_dtype,
            args.checkpoint,
            args.npz_cache,
            args.force_pt,
        )

        result = BenchResult(
            name=name,
            lookahead_router=la,
            tucker_einsum_fuse=ef,
        )

        for trial in range(args.trials):
            _invalidate_tucker_caches(model)
            pfill_tps, dec_tps, pfill_ms, dec_ms = _run_single_config(
                config=mamba_config,
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                decode_tokens=args.decode_tokens,
                warmup=args.warmup,
                target_dtype=target_dtype,
                kv_dtype=kv_dtype,
                router_temp_val=args.router_temp,
            )
            result.prefill_tps.append(pfill_tps)
            result.decode_tps.append(dec_tps)
            result.prefill_ms.append(pfill_ms)
            result.decode_ms.append(dec_ms)
            print(
                f"   Trial {trial+1}/{args.trials}: "
                f"prefill={pfill_tps:.1f} tok/s ({pfill_ms:.1f}ms)  "
                f"decode={dec_tps:.1f} tok/s ({dec_ms:.1f}ms)"
            )

        results.append(result)

        # Free model to release memory between configs
        del model
        gc.collect()
        print()

    # ── Summary table ────────────────────────────────────────────────────
    baseline_dec = results[0].avg_decode_tps if results else 1.0
    baseline_pre = results[0].avg_prefill_tps if results else 1.0

    print("=" * 90)
    print("  RESULTS SUMMARY")
    print("=" * 90)
    header = (
        f"{'Configuration':<32} │ {'Prefill tok/s':>13} │ {'Decode tok/s':>12} │ "
        f"{'Decode ±σ':>9} │ {'vs Baseline':>11}"
    )
    print(header)
    print("─" * 90)

    for r in results:
        speedup = r.avg_decode_tps / baseline_dec if baseline_dec > 0 else 0
        arrow = "▲" if speedup > 1.005 else ("▼" if speedup < 0.995 else "━")
        color_pct = f"{(speedup - 1) * 100:+.1f}%"
        print(
            f"  {r.name:<30} │ {r.avg_prefill_tps:>11.1f}   │ {r.avg_decode_tps:>10.1f}   │ "
            f"{r.std_decode_tps:>7.1f}   │ {arrow} {color_pct:>8}"
        )

    print("─" * 90)
    print()

    # Verdict
    best_decode = max(results, key=lambda r: r.avg_decode_tps)
    worst_decode = min(results, key=lambda r: r.avg_decode_tps)
    if best_decode.avg_decode_tps > baseline_dec * 1.02:
        print(f"  🏆 最快解碼配置: {best_decode.name}")
        print(f"     Decode TPS: {best_decode.avg_decode_tps:.1f} (比 baseline 快 {((best_decode.avg_decode_tps / baseline_dec) - 1) * 100:.1f}%)")
    else:
        print("  ⚖️  所有配置的解碼速度差異不大 (<2%)。")
        print("     這可能表示模型規模太小，優化的增益被其他固定開銷掩蓋。")

    # Check if any config is slower
    for r in results:
        if r is not results[0] and r.avg_decode_tps < baseline_dec * 0.97:
            print(f"  ⚠️  {r.name} 比 baseline 慢 {((1 - r.avg_decode_tps / baseline_dec)) * 100:.1f}%")
            print(f"     可能原因: einsum/lookahead 在小模型上的 JIT 編譯 overhead 超過實際增益")

    print()

    # ── JSON output ──────────────────────────────────────────────────────
    json_path = None
    if args.json is not None:
        json_path = (
            os.path.join(_RESULTS_DIR, "bench_optimizations_ab.json")
            if args.json == "__default__"
            else args.json
        )
        json_data = {
            "meta": {
                "prompt_tokens": len(prompt_ids),
                "decode_tokens": args.decode_tokens,
                "trials": args.trials,
                "warmup": args.warmup,
                "dtype": args.dtype,
                "kv_dtype": args.kv_dtype,
                "router_temp": args.router_temp,
            },
            "results": [
                {
                    "name": r.name,
                    "lookahead_router": r.lookahead_router,
                    "tucker_einsum_fuse": r.tucker_einsum_fuse,
                    "avg_prefill_tps": round(r.avg_prefill_tps, 2),
                    "avg_decode_tps": round(r.avg_decode_tps, 2),
                    "std_decode_tps": round(r.std_decode_tps, 2),
                    "prefill_tps_trials": [round(v, 2) for v in r.prefill_tps],
                    "decode_tps_trials": [round(v, 2) for v in r.decode_tps],
                    "speedup_vs_baseline": round(r.avg_decode_tps / baseline_dec, 4) if baseline_dec > 0 else None,
                }
                for r in results
            ],
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"  📊 JSON results → {json_path}")

    # ── Chart output ─────────────────────────────────────────────────────
    if args.chart is not None:
        chart_path = (
            os.path.join(_RESULTS_DIR, "bench_optimizations_ab.png")
            if args.chart == "__default__"
            else args.chart
        )
        _draw_chart(results, baseline_dec, chart_path, args)
        print(f"  📈 Chart saved → {chart_path}")


def _draw_chart(
    results: list[BenchResult],
    baseline_dec: float,
    out_path: str,
    args: Any,
) -> None:
    """Generate a publication-ready bar chart comparing configs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("  ⚠️  matplotlib not installed — skipping chart.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.patch.set_facecolor("#1a1a2e")

    names = [r.name for r in results]
    decode_means = [r.avg_decode_tps for r in results]
    decode_stds = [r.std_decode_tps for r in results]
    prefill_means = [r.avg_prefill_tps for r in results]

    colors = ["#4a6fa5", "#e07a5f", "#81b29a", "#f2cc8f"]
    edge_colors = ["#2d4a75", "#b5533f", "#5a8a6f", "#c4a05f"]

    # ── Decode TPS bar chart ─────────────────────────────────────────────
    ax1.set_facecolor("#16213e")
    bars1 = ax1.bar(
        range(len(names)),
        decode_means,
        yerr=decode_stds,
        color=colors[:len(names)],
        edgecolor=edge_colors[:len(names)],
        linewidth=1.5,
        capsize=5,
        error_kw={"elinewidth": 1.5, "capthick": 1.5, "color": "#ffffff"},
        zorder=3,
    )
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n.split("(")[0].strip() for n in names], fontsize=9, color="#e0e0e0", rotation=15, ha="right")
    ax1.set_ylabel("Decode (tok/s)", fontsize=11, color="#e0e0e0", fontweight="bold")
    ax1.set_title("Decode Throughput Comparison", fontsize=13, color="#ffffff", fontweight="bold", pad=12)
    ax1.tick_params(colors="#999999")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#555555")
    ax1.spines["bottom"].set_color("#555555")
    ax1.grid(axis="y", alpha=0.2, color="#888888", linestyle="--")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=8))

    # Add value labels on bars
    for bar, val in zip(bars1, decode_means):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(decode_stds) * 0.3 + 0.5,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=9, color="#ffffff", fontweight="bold",
        )

    # Add speedup annotation
    for i, r in enumerate(results):
        if i == 0:
            continue
        speedup = r.avg_decode_tps / baseline_dec if baseline_dec > 0 else 0
        pct = (speedup - 1) * 100
        color = "#81b29a" if pct >= 0 else "#e07a5f"
        ax1.text(
            i,
            max(decode_means) * 0.05,
            f"{pct:+.1f}%",
            ha="center", va="bottom", fontsize=10, color=color, fontweight="bold",
        )

    # ── Prefill TPS bar chart ────────────────────────────────────────────
    ax2.set_facecolor("#16213e")
    bars2 = ax2.bar(
        range(len(names)),
        prefill_means,
        color=colors[:len(names)],
        edgecolor=edge_colors[:len(names)],
        linewidth=1.5,
        alpha=0.85,
        zorder=3,
    )
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.split("(")[0].strip() for n in names], fontsize=9, color="#e0e0e0", rotation=15, ha="right")
    ax2.set_ylabel("Prefill (tok/s)", fontsize=11, color="#e0e0e0", fontweight="bold")
    ax2.set_title("Prefill Throughput Comparison", fontsize=13, color="#ffffff", fontweight="bold", pad=12)
    ax2.tick_params(colors="#999999")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#555555")
    ax2.spines["bottom"].set_color("#555555")
    ax2.grid(axis="y", alpha=0.2, color="#888888", linestyle="--")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=8))

    for bar, val in zip(bars2, prefill_means):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(prefill_means) * 0.015,
            f"{val:.0f}",
            ha="center", va="bottom", fontsize=9, color="#ffffff", fontweight="bold",
        )

    fig.suptitle(
        f"Mamba3-XR Inference Optimization A/B Test\n"
        f"dtype={args.dtype}  kv={args.kv_dtype}  "
        f"prompt={len(results[0].prefill_tps) if results else 0}×{args.decode_tokens} decode tok  "
        f"{args.trials} trials",
        fontsize=11,
        color="#cccccc",
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
