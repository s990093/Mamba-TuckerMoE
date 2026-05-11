#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_ultimate_kernels.py — 完整實驗矩陣 Benchmark

實驗矩陣（必須記錄）：
  Exp 1: Bank Conflict 消除效果（Naive vs Padding vs 離線轉置 vs XOR Swizzle）
  Exp 2: AMX vs SIMD FMA
  Exp 3: 融合層數效果（逐步加入各 kernel）
  Exp 4: BF16 vs FP32 精度對比
  Exp 5: 整體 tok/s 端對端目標（>100 tok/s）

所有結果自動記錄到 results/ultimate_experiment_log.md
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import numpy as np

from ultimate_kernel_lib import UltimateMambaKernels, verify_correctness, transpose_g_experts

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# 計時工具
# ══════════════════════════════════════════════════════════════════

def bench_fn(fn, warmup=20, trials=100):
    """精確計時：預熱 + 多次試驗取平均"""
    for _ in range(warmup):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(trials):
        mx.eval(fn())
    return (time.perf_counter() - t0) * 1000.0 / trials  # ms


@dataclass
class BenchResult:
    exp_id: str
    name: str
    config: Dict
    ms: float
    speedup: float = 1.0
    max_abs_err: float = 0.0
    notes: str = ""


# ══════════════════════════════════════════════════════════════════
# Exp 1: Bank Conflict 消除效果
# ══════════════════════════════════════════════════════════════════

def exp1_bank_conflict(args, kernels: UltimateMambaKernels) -> List[BenchResult]:
    """
    比較不同 Bank Conflict 消除策略：
    1. Naive: G_experts[E, R3, R2]，無優化（模擬 32-way conflict）
    2. Padding ROW_STRIDE=40: threadgroup memory padding
    3. 離線轉置 [E, R2, R3]: 完全連續存取（零衝突）
    """
    print("\n" + "="*60)
    print("📊 Exp 1: Bank Conflict 消除效果")
    print("="*60)

    r3, r2, e, k = args.r3, args.r2, args.e, args.k
    dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float32

    mx.random.seed(42)
    x   = mx.random.normal((r3,)).astype(dtype)
    g   = mx.random.normal((e, r3, r2)).astype(dtype)
    g_T = transpose_g_experts(g)
    ids = mx.array(list(range(k)), dtype=mx.uint32)

    results = []

    # Baseline: MLX matmul（純 Python for loop）
    def baseline_mlx():
        out = mx.zeros((r2,), dtype=dtype)
        for i in range(k):
            eid = int(ids[i].item())
            out = out + mx.matmul(x.reshape(1, -1), g[eid]).reshape(-1)
        return out
    y_ref = baseline_mlx()
    mx.eval(y_ref)
    base_ms = bench_fn(baseline_mlx, args.warmup, args.trials)
    results.append(BenchResult("exp1", "MLX baseline (Python loop)", {"r3":r3,"r2":r2,"e":e,"k":k}, base_ms))
    print(f"  MLX baseline:       {base_ms:.4f} ms")

    # Strategy 1: Scalar Metal kernel（模擬 Naive，無 Bank Conflict 優化）
    scalar_fn = kernels.tucker.build_scalar(r3, r2, e, k)

    def run_scalar():
        return kernels.tucker.run_scalar(scalar_fn, x, g, ids, r2, dtype)
    y_scalar = run_scalar(); mx.eval(y_scalar)
    err_scalar = float(mx.max(mx.abs(y_ref.astype(mx.float32) - y_scalar.astype(mx.float32))).item())
    scalar_ms = bench_fn(run_scalar, args.warmup, args.trials)
    sp1 = base_ms / scalar_ms
    results.append(BenchResult("exp1", "Metal scalar (naive)", {"stride": r2}, scalar_ms, sp1, err_scalar))
    print(f"  Metal scalar:       {scalar_ms:.4f} ms  speedup={sp1:.2f}x  err={err_scalar:.6f}")

    # Strategy 2: AMX + 離線轉置（零 Bank Conflict）
    if r3 % 32 == 0 and r2 % 32 == 0:
        amx_fn = kernels.tucker.build_amx(r3, r2, e, k)

        def run_amx():
            return kernels.tucker.run_amx(amx_fn, x, g_T, ids, r2, k, dtype)
        y_amx = run_amx(); mx.eval(y_amx)
        err_amx = float(mx.max(mx.abs(y_ref.astype(mx.float32) - y_amx.astype(mx.float32))).item())
        amx_ms = bench_fn(run_amx, args.warmup, args.trials)
        sp2 = base_ms / amx_ms
        results.append(BenchResult("exp1", "Metal AMX + Transpose (zero conflict)", {"row_stride":40}, amx_ms, sp2, err_amx))
        print(f"  Metal AMX+Transpose:{amx_ms:.4f} ms  speedup={sp2:.2f}x  err={err_amx:.6f}")
    else:
        print(f"  ⚠️  AMX kernel requires r3,r2 ∈ multiples of 32 (got r3={r3}, r2={r2}), skipped")

    return results


# ══════════════════════════════════════════════════════════════════
# Exp 2: AMX vs SIMD FMA vs FP32
# ══════════════════════════════════════════════════════════════════

def exp2_amx_vs_simd(args, kernels: UltimateMambaKernels) -> List[BenchResult]:
    print("\n" + "="*60)
    print("📊 Exp 2: AMX vs SIMD FMA vs 精度對比")
    print("="*60)

    r3, r2, e, k = args.r3, args.r2, args.e, args.k
    results = []

    for dtype_name, dtype in [("fp32", mx.float32), ("bf16", mx.bfloat16)]:
        mx.random.seed(42)
        x   = mx.random.normal((r3,)).astype(dtype)
        g   = mx.random.normal((e, r3, r2)).astype(dtype)
        g_T = transpose_g_experts(g)
        ids = mx.array(list(range(k)), dtype=mx.uint32)

        y_ref = mx.sum(mx.stack([
            mx.matmul(x.reshape(1,-1), g[int(ids[i].item())]).reshape(-1)
            for i in range(k)
        ]), axis=0)
        mx.eval(y_ref)

        scalar_fn = kernels.tucker.build_scalar(r3, r2, e, k)
        scalar_ms = bench_fn(
            lambda: kernels.tucker.run_scalar(scalar_fn, x, g, ids, r2, dtype),
            args.warmup, args.trials)

        label = f"Metal scalar [{dtype_name}]"
        results.append(BenchResult("exp2", label, {"dtype": dtype_name}, scalar_ms))
        print(f"  {label}: {scalar_ms:.4f} ms")

        if r3 % 32 == 0 and r2 % 32 == 0:
            amx_fn = kernels.tucker.build_amx(r3, r2, e, k)
            amx_ms = bench_fn(
                lambda: kernels.tucker.run_amx(amx_fn, x, g_T, ids, r2, k, dtype),
                args.warmup, args.trials)
            label2 = f"Metal AMX    [{dtype_name}]"
            results.append(BenchResult("exp2", label2, {"dtype": dtype_name}, amx_ms,
                                       scalar_ms / amx_ms))
            print(f"  {label2}: {amx_ms:.4f} ms  speedup={scalar_ms/amx_ms:.2f}x vs scalar")

    return results


# ══════════════════════════════════════════════════════════════════
# Exp 3: SSM Scan 對比
# ══════════════════════════════════════════════════════════════════

def exp3_ssm_scan(args, kernels: UltimateMambaKernels) -> List[BenchResult]:
    print("\n" + "="*60)
    print("📊 Exp 3: SSM Scan — Metal vs MLX")
    print("="*60)

    H, D = 64, 64
    B_nc, Lc = 1, args.seq_len
    results = []

    mx.random.seed(42)
    la_f32 = mx.random.normal((B_nc, Lc, H))
    u_f32  = mx.random.normal((B_nc, Lc, H, D))
    # 用 BF16 版作為 reference 和 metal 的共同輸入（公平比較）
    la_seq = la_f32.astype(mx.bfloat16)
    u_seq  = u_f32.astype(mx.bfloat16)

    # MLX reference（使用 BF16 輸入，FP32 累加，與 Metal 完全一致）
    def mlx_scan():
        h = mx.zeros((B_nc, H, D), dtype=mx.float32)
        outs = []
        for t in range(Lc):
            la = la_seq[:, t, :].astype(mx.float32)
            u  = u_seq[:, t, :, :].astype(mx.float32)
            alpha = mx.exp(mx.clip(la, -40, 40))[:, :, None]
            h = alpha * h + u
            outs.append(h[:, None, :, :])
        return mx.concatenate(outs, axis=1)

    ref = mlx_scan(); mx.eval(ref)
    base_ms = bench_fn(mlx_scan, 5, 20)
    results.append(BenchResult("exp3", "MLX O(N) scan (BF16→F32)", {"Lc": Lc}, base_ms))
    print(f"  MLX eager scan:    {base_ms:.4f} ms  (L={Lc})")

    ssm_fn = kernels.ssm.build_prefill_scan(H, D, B_nc, Lc)

    def metal_scan():
        return ssm_fn(
            inputs=[la_seq, u_seq],
            template=[("T", mx.bfloat16)],
            grid=(D, H, B_nc),
            threadgroup=(1, 1, 1),
            output_shapes=[(B_nc, Lc, H, D)],
            output_dtypes=[mx.float32],  # kernel writes float directly
        )[0]

    metal_out = metal_scan(); mx.eval(metal_out)
    err = float(mx.mean(mx.abs(ref - metal_out)).item())
    err_max = float(mx.max(mx.abs(ref - metal_out)).item())
    metal_ms = bench_fn(metal_scan, args.warmup, args.trials)
    sp = base_ms / metal_ms
    results.append(BenchResult("exp3", "Metal SSM scan (BF16→F32)", {"Lc": Lc}, metal_ms, sp, err_max))
    print(f"  Metal SSM scan:    {metal_ms:.4f} ms  speedup={sp:.2f}x  mean_err={err:.4f}  max_err={err_max:.4f}")


    return results


# ══════════════════════════════════════════════════════════════════
# Exp 4: 融合組合效果
# ══════════════════════════════════════════════════════════════════

def exp4_fusion_ablation(args, kernels: UltimateMambaKernels) -> List[BenchResult]:
    print("\n" + "="*60)
    print("📊 Exp 4: Kernel 融合組合效果")
    print("="*60)

    r3, r2, e, k = args.r3, args.r2, args.e, args.k
    dtype = mx.bfloat16
    results = []

    mx.random.seed(42)
    x   = mx.random.normal((r3,)).astype(dtype)
    g   = mx.random.normal((e, r3, r2)).astype(dtype)
    g_T = transpose_g_experts(g)
    ids = mx.array(list(range(k)), dtype=mx.uint32)

    # 基準：MLX for loop
    def mlx_baseline():
        out = mx.zeros((r2,), dtype=dtype)
        for i in range(k):
            eid = int(ids[i].item())
            out = out + mx.matmul(x.reshape(1,-1), g[eid]).reshape(-1)
        return out

    base_ms = bench_fn(mlx_baseline, args.warmup, args.trials)
    results.append(BenchResult("exp4", "No fusion (MLX baseline)", {}, base_ms, 1.0))
    print(f"  No fusion (MLX):   {base_ms:.4f} ms  [1.00x]")

    # Metal scalar（fusion level 1）
    scalar_fn = kernels.tucker.build_scalar(r3, r2, e, k)
    scalar_ms = bench_fn(
        lambda: kernels.tucker.run_scalar(scalar_fn, x, g, ids, r2, dtype),
        args.warmup, args.trials)
    sp = base_ms / scalar_ms
    results.append(BenchResult("exp4", "Metal scalar fusion", {}, scalar_ms, sp))
    print(f"  Metal scalar:      {scalar_ms:.4f} ms  [{sp:.2f}x]")

    # Metal AMX + transpose（fusion level 2）
    if r3 % 32 == 0 and r2 % 32 == 0:
        amx_fn = kernels.tucker.build_amx(r3, r2, e, k)
        amx_ms = bench_fn(
            lambda: kernels.tucker.run_amx(amx_fn, x, g_T, ids, r2, k, dtype),
            args.warmup, args.trials)
        sp2 = base_ms / amx_ms
        results.append(BenchResult("exp4", "Metal AMX+Transpose+Pipeline", {}, amx_ms, sp2))
        print(f"  Metal AMX+Pipe:    {amx_ms:.4f} ms  [{sp2:.2f}x]")

    return results


# ══════════════════════════════════════════════════════════════════
# 報告輸出
# ══════════════════════════════════════════════════════════════════

def write_report(all_results: List[BenchResult], args):
    """寫出完整實驗記錄 Markdown"""
    log_path = RESULTS_DIR / "ultimate_experiment_log.md"

    lines = [
        "# Ultimate Metal Kernel 實驗記錄",
        "",
        f"**實驗時間**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**配置**: r3={args.r3}, r2={args.r2}, e={args.e}, k={args.k}, dtype={args.dtype}",
        f"**序列長度**: L={args.seq_len}",
        "",
        "---",
        "",
    ]

    exp_groups: Dict[str, List[BenchResult]] = {}
    for r in all_results:
        exp_groups.setdefault(r.exp_id, []).append(r)

    exp_titles = {
        "exp1": "Exp 1: Bank Conflict 消除效果",
        "exp2": "Exp 2: AMX vs SIMD FMA & 精度對比",
        "exp3": "Exp 3: SSM Scan Metal vs MLX",
        "exp4": "Exp 4: Kernel 融合組合消融實驗",
    }

    for eid, res_list in exp_groups.items():
        lines.append(f"## {exp_titles.get(eid, eid)}")
        lines.append("")
        lines.append("| 策略 | 時間 (ms) | 加速比 | 最大誤差 | 備註 |")
        lines.append("|------|-----------|--------|---------|------|")
        for r in res_list:
            lines.append(
                f"| {r.name} | {r.ms:.4f} | {r.speedup:.2f}x "
                f"| {r.max_abs_err:.6f} | {r.notes} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## 關鍵結論",
        "",
        "| 優化項目 | 預期收益（來自 PDF）| 實測收益 |",
        "|---------|---------------------|---------|",
        "| 消除 32-way Bank Conflict | 最高 32× | 見 Exp 1 |",
        "| Tucker 降維後路由 | Permute 流量降低 93.75% | 見 Exp 4 |",
        "| AMX simdgroup_matrix | 消除 Threadgroup 往返 | 見 Exp 2 |",
        "| SSM O(N) scan | 2.59× (來自 README) | 見 Exp 3 |",
        "",
    ]

    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄 實驗報告已寫出：{log_path}")

    # 同時輸出 JSON
    json_path = RESULTS_DIR / "ultimate_experiment_results.json"
    json_path.write_text(
        json.dumps([asdict(r) for r in all_results], indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"📊 JSON 數據：{json_path}")


# ══════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Ultimate Metal Kernel Benchmark")
    ap.add_argument("--r3",       type=int, default=256,   help="Tucker r3 (latent input dim)")
    ap.add_argument("--r2",       type=int, default=1024,  help="Tucker r2 (latent output dim)")
    ap.add_argument("--e",        type=int, default=8,     help="Total num experts")
    ap.add_argument("--k",        type=int, default=2,     help="Active top-k experts")
    ap.add_argument("--dtype",    choices=["bf16","fp32"], default="bf16")
    ap.add_argument("--seq-len",  type=int, default=64,    help="Sequence length for SSM exp")
    ap.add_argument("--warmup",   type=int, default=20)
    ap.add_argument("--trials",   type=int, default=100)
    ap.add_argument("--exp",      type=str, default="all",
                    help="Which experiments: all, exp1, exp2, exp3, exp4 (comma-separated)")
    args = ap.parse_args()

    print("🚀 Ultimate Metal Kernel Benchmark")
    print(f"   Config: r3={args.r3}, r2={args.r2}, E={args.e}, K={args.k}, dtype={args.dtype}")

    kernels = UltimateMambaKernels()
    kernels.warmup(args.r3, args.r2, args.e, args.k,
                   mx.bfloat16 if args.dtype == "bf16" else mx.float32)

    run = set(args.exp.split(",")) if args.exp != "all" else {"exp1","exp2","exp3","exp4"}
    all_results: List[BenchResult] = []

    if "exp1" in run:
        all_results.extend(exp1_bank_conflict(args, kernels))
    if "exp2" in run:
        all_results.extend(exp2_amx_vs_simd(args, kernels))
    if "exp3" in run:
        all_results.extend(exp3_ssm_scan(args, kernels))
    if "exp4" in run:
        all_results.extend(exp4_fusion_ablation(args, kernels))

    write_report(all_results, args)

    # 最終摘要
    print("\n" + "="*60)
    print("🏆 最終摘要（各實驗最佳結果）")
    print("="*60)
    best_by_exp: Dict[str, BenchResult] = {}
    for r in all_results:
        if r.exp_id not in best_by_exp or r.speedup > best_by_exp[r.exp_id].speedup:
            best_by_exp[r.exp_id] = r
    for eid, r in sorted(best_by_exp.items()):
        print(f"  {eid}: {r.name}")
        print(f"         {r.ms:.4f} ms  speedup={r.speedup:.2f}x  err={r.max_abs_err:.6f}")


if __name__ == "__main__":
    main()
