#!/usr/bin/env python3
"""
Unified MoE benchmark report with CSV export.

Includes:
- Latent MoE benchmark from `benchmark_fused_latent_moe_kernel.py`
- Tucker MoE benchmark from `benchmark_tucker_moe_fused.py`
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from pathlib import Path

import mlx.core as mx

from benchmark_fused_latent_moe_kernel import (
    _build_fused_latent_kernel,
    baseline_mlx as latent_baseline_mlx,
    _build_fused_latent_amx_partial_kernel,
    fused_metal as latent_fused_metal,
    fused_metal_amx_reduce as latent_fused_amx_reduce,
)
from benchmark_tucker_moe_fused import (
    fused_tucker_moe_e2e_metal,
    reference_mlx_production_router_tucker,
    reference_mlx_z_dot_g_router_tucker,
)


def _measure_ms(fn, trials: int) -> float:
    t0 = time.perf_counter()
    for _ in range(trials):
        mx.eval(fn())
    return (time.perf_counter() - t0) * 1000.0 / max(trials, 1)


def _run_pure_metal_benchmark(
    *,
    r3: int,
    r2: int,
    e: int,
    k: int,
    warmup: int,
    trials: int,
) -> tuple[float, float] | None:
    """
    Run Swift pure-Metal benchmark and parse:
      - sync_partial+reduce ms
      - async_partial+reduce ms
    """
    cmd = [
        "swift",
        "metal/benchmark_fused_latent_moe_async.swift",
        "--r3",
        str(r3),
        "--r2",
        str(r2),
        "--e",
        str(e),
        "--k",
        str(k),
        "--warmup",
        str(warmup),
        "--trials",
        str(trials),
    ]
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except Exception:
        return None
    txt = proc.stdout
    m_sync = re.search(r"sync_partial\+reduce\s*:\s*([0-9.]+)\s*ms", txt)
    m_async = re.search(r"async_partial\+reduce:\s*([0-9.]+)\s*ms", txt)
    if not m_sync or not m_async:
        return None
    return float(m_sync.group(1)), float(m_async.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", type=str, default="metal/results/moe_unified_benchmark.csv")
    ap.add_argument(
        "--include-pure-metal",
        action="store_true",
        default=True,
        help="Also run Swift pure-Metal benchmark and append rows when available.",
    )
    ap.add_argument(
        "--no-include-pure-metal",
        dest="include_pure_metal",
        action="store_false",
        help="Skip Swift pure-Metal benchmark rows.",
    )

    # latent MoE shape
    ap.add_argument("--latent-r3", type=int, default=256)
    ap.add_argument("--latent-r2", type=int, default=1024)
    ap.add_argument("--latent-e", type=int, default=8)
    ap.add_argument("--latent-k", type=int, default=2)

    # tucker MoE shape
    ap.add_argument("--tucker-d", type=int, default=768)
    ap.add_argument("--tucker-r3", type=int, default=256)
    ap.add_argument("--tucker-r2", type=int, default=512)
    ap.add_argument("--tucker-d-out", type=int, default=4608)
    ap.add_argument("--tucker-e", type=int, default=8)
    ap.add_argument("--tucker-k", type=int, default=2)
    args = ap.parse_args()

    dt = mx.bfloat16 if args.dtype == "bf16" else mx.float32
    mx.random.seed(args.seed)
    rows: list[dict[str, str | float | int]] = []

    # ---- Latent MoE section ----
    lr3, lr2, le, lk = args.latent_r3, args.latent_r2, args.latent_e, args.latent_k
    x_shared = mx.random.normal((lr3,)).astype(dt)
    g_experts = mx.random.normal((le, lr3, lr2)).astype(dt)
    expert_indices = mx.random.permutation(mx.arange(le))[:lk].astype(mx.uint32)
    latent_kernel = _build_fused_latent_kernel(lr3, lr2, le, lk)
    latent_amx_kernel = _build_fused_latent_amx_partial_kernel(lr3, lr2, le, lk)

    for _ in range(args.warmup):
        mx.eval(
            latent_baseline_mlx(x_shared, g_experts, expert_indices),
            latent_fused_metal(latent_kernel, x_shared, g_experts, expert_indices, dtype=dt),
            latent_fused_amx_reduce(
                latent_amx_kernel, x_shared, g_experts, expert_indices, dtype=dt
            ),
        )

    ms_latent_base = _measure_ms(
        lambda: latent_baseline_mlx(x_shared, g_experts, expert_indices), args.trials
    )
    ms_latent_fused = _measure_ms(
        lambda: latent_fused_metal(latent_kernel, x_shared, g_experts, expert_indices, dtype=dt),
        args.trials,
    )
    ms_latent_fused_amx = _measure_ms(
        lambda: latent_fused_amx_reduce(
            latent_amx_kernel, x_shared, g_experts, expert_indices, dtype=dt
        ),
        args.trials,
    )

    rows.append(
        {
            "suite": "latent_moe",
            "method": "mlx_baseline",
            "dtype": args.dtype,
            "shape": f"R3={lr3},R2={lr2},E={le},K={lk}",
            "ms": ms_latent_base,
            "speedup_vs_baseline": 1.0,
        }
    )
    rows.append(
        {
            "suite": "latent_moe",
            "method": "metal_fused",
            "dtype": args.dtype,
            "shape": f"R3={lr3},R2={lr2},E={le},K={lk}",
            "ms": ms_latent_fused,
            "speedup_vs_baseline": ms_latent_base / max(ms_latent_fused, 1e-12),
        }
    )
    rows.append(
        {
            "suite": "latent_moe",
            "method": "metal_fused_amx_partial_reduce",
            "dtype": args.dtype,
            "shape": f"R3={lr3},R2={lr2},E={le},K={lk}",
            "ms": ms_latent_fused_amx,
            "speedup_vs_baseline": ms_latent_base / max(ms_latent_fused_amx, 1e-12),
        }
    )
    if args.include_pure_metal:
        pure = _run_pure_metal_benchmark(
            r3=lr3,
            r2=lr2,
            e=le,
            k=lk,
            warmup=args.warmup,
            trials=args.trials,
        )
        if pure is not None:
            ms_sync, ms_async = pure
            rows.append(
                {
                    "suite": "latent_moe_pure_metal",
                    "method": "sync_partial_reduce",
                    "dtype": "fp32",
                    "shape": f"R3={lr3},R2={lr2},E={le},K={lk}",
                    "ms": ms_sync,
                    "speedup_vs_baseline": 1.0,
                }
            )
            rows.append(
                {
                    "suite": "latent_moe_pure_metal",
                    "method": "async_partial_reduce",
                    "dtype": "fp32",
                    "shape": f"R3={lr3},R2={lr2},E={le},K={lk}",
                    "ms": ms_async,
                    "speedup_vs_baseline": ms_sync / max(ms_async, 1e-12),
                }
            )

    # ---- Tucker MoE section ----
    td, tr3, tr2 = args.tucker_d, args.tucker_r3, args.tucker_r2
    td_out, te, tk = args.tucker_d_out, args.tucker_e, args.tucker_k
    x = mx.random.normal((td,)).astype(dt)
    u_in_w = mx.random.normal((tr3, td)).astype(dt)
    core = mx.random.normal((te, tr3, tr2)).astype(dt)
    u_out_w = mx.random.normal((td_out, tr2)).astype(dt)
    u_out_t = mx.transpose(u_out_w)
    router_w = mx.random.normal((te, td)).astype(dt)
    g_r3_e = mx.random.normal((tr3, te)).astype(dt)

    y_prod, eid_prod, pr_prod = reference_mlx_production_router_tucker(
        x, router_w, u_in_w, core, u_out_w, tk
    )
    mx.eval(y_prod, eid_prod, pr_prod)

    for _ in range(args.warmup):
        mx.eval(
            reference_mlx_production_router_tucker(x, router_w, u_in_w, core, u_out_w, tk)[0],
            reference_mlx_z_dot_g_router_tucker(x, u_in_w, g_r3_e, core, u_out_w, tk)[0],
            fused_tucker_moe_e2e_metal(x, u_in_w, core, u_out_t, eid_prod, pr_prod),
        )

    ms_tucker_prod = _measure_ms(
        lambda: reference_mlx_production_router_tucker(x, router_w, u_in_w, core, u_out_w, tk)[0],
        args.trials,
    )
    ms_tucker_zg = _measure_ms(
        lambda: reference_mlx_z_dot_g_router_tucker(x, u_in_w, g_r3_e, core, u_out_w, tk)[0],
        args.trials,
    )
    ms_tucker_fused = _measure_ms(
        lambda: fused_tucker_moe_e2e_metal(x, u_in_w, core, u_out_t, eid_prod, pr_prod),
        args.trials,
    )

    tucker_shape = f"d={td},R3={tr3},R2={tr2},d_out={td_out},E={te},K={tk}"
    rows.extend(
        [
            {
                "suite": "tucker_moe",
                "method": "mlx_production_router",
                "dtype": args.dtype,
                "shape": tucker_shape,
                "ms": ms_tucker_prod,
                "speedup_vs_baseline": 1.0,
            },
            {
                "suite": "tucker_moe",
                "method": "mlx_z_dot_g_router",
                "dtype": args.dtype,
                "shape": tucker_shape,
                "ms": ms_tucker_zg,
                "speedup_vs_baseline": ms_tucker_prod / max(ms_tucker_zg, 1e-12),
            },
            {
                "suite": "tucker_moe",
                "method": "metal_scalar_fused",
                "dtype": args.dtype,
                "shape": tucker_shape,
                "ms": ms_tucker_fused,
                "speedup_vs_baseline": ms_tucker_prod / max(ms_tucker_fused, 1e-12),
            },
        ]
    )

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["suite", "method", "dtype", "shape", "ms", "speedup_vs_baseline"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote CSV: {csv_path}")
    print()
    print("| Suite | Method | ms | Speedup vs baseline |")
    print("|---|---|---:|---:|")
    for r in rows:
        print(
            f"| {r['suite']} | {r['method']} | {float(r['ms']):.4f} | {float(r['speedup_vs_baseline']):.2f}x |"
        )


if __name__ == "__main__":
    main()

