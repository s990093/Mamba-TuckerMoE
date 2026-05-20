#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2C: Benchmark fused Mamba kernels against baseline.

Measures latency improvements from:
1. Fused SSM scan (eliminate h_intra intermediate)
2. Fused output projection (combine down-proj + skip connection)
3. Optimized intra-chunk computation

Expected results: 2-3× speedup on Mamba block latency
"""

import sys
import time
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    print("Error: MLX not installed")
    sys.exit(1)

# Add parent directory to path
sys.path.insert(0, "/Users/hungwei/Desktop/Proj/Mamba3-XR")

from mamba3_mlx.mlx_model.mamba_block import Mamba3Block
from mamba3_mlx.mlx_model.mamba_block_fused import Mamba3BlockFused
from mamba3_mlx.mlx_model.hybrid_model import Mamba3Config


def create_dummy_config():
    """Create test configuration."""
    config = Mamba3Config(
        d_model=768,
        d_head=64,
        n_groups=1,
        d_state=64,
        mimo_rank=4,
        expand=2,
        chunk_size=64,
        kmoe_num_experts=8,
        kmoe_top_k=2,
        kmoe_r1=32,
        kmoe_r2=512,
        kmoe_r3=256,
    )
    return config


def benchmark_decode_step(block, x, step=None, mamba_state=None, label=""):
    """Benchmark a single decode step (B=1, L=1)."""
    # Warmup
    for _ in range(2):
        _ = block(x, step=step, mamba_state=mamba_state)
        mx.eval(_)

    # Time
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        out, _, _, new_state = block(x, step=step, mamba_state=mamba_state)
        mx.eval(out)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    print(f"{label:30s}: {avg_ms:6.2f} ± {std_ms:5.2f} ms")
    return avg_ms, new_state


def benchmark_prefill(block, x, step=None, label=""):
    """Benchmark prefill (B=1, L=256)."""
    # Warmup
    for _ in range(1):
        _ = block(x, step=step)
        mx.eval(_)

    # Time
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        out, _, _, new_state = block(x, step=step)
        mx.eval(out)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    L = x.shape[1]
    tok_per_sec = (L / (avg_ms / 1000)) if avg_ms > 0 else 0
    print(f"{label:30s}: {avg_ms:8.2f} ± {std_ms:6.2f} ms ({tok_per_sec:6.1f} tok/s)")
    return avg_ms, new_state


def main():
    print("\n" + "=" * 80)
    print("Phase 2C: Fused Kernel Benchmark")
    print("=" * 80)

    config = create_dummy_config()

    # Create both blocks
    print("\nInitializing blocks...")
    block_baseline = Mamba3Block(config)
    block_fused = Mamba3BlockFused(config)

    # Copy weights from baseline to fused for fair comparison
    block_fused.in_proj.weight = block_baseline.in_proj.weight
    block_fused.in_proj.bias = block_baseline.in_proj.bias
    block_fused.x_up_proj = block_baseline.x_up_proj
    block_fused.out_proj = block_baseline.out_proj
    block_fused.y_down_proj = block_baseline.y_down_proj
    block_fused.theta_log = block_baseline.theta_log
    block_fused.D = block_baseline.D
    block_fused.norm_B = block_baseline.norm_B
    block_fused.norm_C = block_baseline.norm_C
    block_fused.bias_B = block_baseline.bias_B
    block_fused.bias_C = block_baseline.bias_C
    block_fused.mamba_dense_proj = block_baseline.mamba_dense_proj
    block_fused.pre_gate_norm = block_baseline.pre_gate_norm
    block_fused.norm_mamba = block_baseline.norm_mamba
    block_fused.norm_out_proj = block_baseline.norm_out_proj
    block_fused.ls_mamba = block_baseline.ls_mamba
    block_fused.ls_out_proj = block_baseline.ls_out_proj

    # === Prefill Benchmark ===
    print("\n" + "-" * 80)
    print("PREFILL (B=1, L=256) — Full sequence processing")
    print("-" * 80)

    x_prefill = mx.random.normal((1, 256, config.d_model))
    t_baseline, _ = benchmark_prefill(block_baseline, x_prefill, label="Baseline Mamba")
    t_fused, _ = benchmark_prefill(block_fused, x_prefill, label="Fused Mamba (Phase 2C)")
    speedup_prefill = t_baseline / t_fused if t_fused > 0 else 1.0
    print(f"\nPrefill Speedup: {speedup_prefill:.2f}×")

    # === Decode Benchmark ===
    print("\n" + "-" * 80)
    print("DECODE (B=1, L=1) — Single token")
    print("-" * 80)

    x_decode = mx.random.normal((1, 1, config.d_model))

    # Initialize state from prefill
    _, mamba_state_baseline = benchmark_decode_step(
        block_baseline, x_decode, step=256, label="Baseline (first decode step)"
    )

    # Decode steps (warm cache)
    total_baseline = 0
    for i in range(32):
        t_step, mamba_state_baseline = benchmark_decode_step(
            block_baseline, x_decode, step=256 + i, mamba_state=mamba_state_baseline, label=""
        )
        total_baseline += t_step

    avg_baseline = total_baseline / 32
    tok_per_sec_baseline = 1000 / avg_baseline if avg_baseline > 0 else 0

    print(f"Baseline avg (32-step):        {avg_baseline:6.2f} ms ({tok_per_sec_baseline:6.1f} tok/s)")

    # Fused version
    _, mamba_state_fused = benchmark_decode_step(
        block_fused, x_decode, step=256, label="Fused (first decode step)"
    )

    total_fused = 0
    for i in range(32):
        t_step, mamba_state_fused = benchmark_decode_step(
            block_fused, x_decode, step=256 + i, mamba_state=mamba_state_fused, label=""
        )
        total_fused += t_step

    avg_fused = total_fused / 32
    tok_per_sec_fused = 1000 / avg_fused if avg_fused > 0 else 0

    print(f"Fused avg (32-step):           {avg_fused:6.2f} ms ({tok_per_sec_fused:6.1f} tok/s)")

    speedup_decode = avg_baseline / avg_fused if avg_fused > 0 else 1.0
    print(f"\nDecode Speedup: {speedup_decode:.2f}×")

    # === Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY: Phase 2C Fusion Impact")
    print("=" * 80)

    print(f"\nPrefill:          {t_baseline:6.2f} ms → {t_fused:6.2f} ms ({speedup_prefill:.2f}× speedup)")
    print(f"Decode (per-tok): {avg_baseline:6.2f} ms → {avg_fused:6.2f} ms ({speedup_decode:.2f}× speedup)")
    print(f"\nBaseline decode throughput: {tok_per_sec_baseline:6.1f} tok/s")
    print(f"Fused decode throughput:    {tok_per_sec_fused:6.1f} tok/s")

    if tok_per_sec_fused >= 60:
        print("\n✅ TARGET ACHIEVED: ≥ 60 tok/s decode throughput!")
    elif tok_per_sec_fused >= 55:
        print("\n⚠️  NEAR TARGET: 55+ tok/s (may be acceptable)")
    else:
        print(f"\n❌ TARGET NOT MET: {tok_per_sec_fused:.1f} tok/s (need ≥60 tok/s)")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
