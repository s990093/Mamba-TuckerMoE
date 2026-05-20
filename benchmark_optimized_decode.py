#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的 decode 基准 - 支持可选的融合优化。

演示：
- 基准路径（标准 MLX）
- 启用单个优化
- 启用所有优化
- 比较性能

所有优化都是完全解耦合的，不影响原始推理管道。
"""

import sys
import time

try:
    import mlx.core as mx
except ImportError:
    print("MLX not installed")
    sys.exit(1)

sys.path.insert(0, "/Users/hungwei/Desktop/Proj/Mamba3-XR")

from mamba3_mlx.mlx_model.hybrid_model import build_model
from mamba3_mlx.mlx_model.quantized_model import (
    QuantizationConfig,
    apply_quantization_to_model,
)
from mamba3_mlx.mlx_model.fusion_optimizer import get_fusion_optimizer


def benchmark_decode(
    model,
    num_tokens: int = 128,
    label: str = "",
) -> float:
    """基准 decode 性能。

    Returns:
        tok_s: Tokens per second
    """
    print(f"\n{label}")
    print(f"{'─'*70}")

    mamba_states, kv_caches = model.init_empty_states(batch_size=1)

    # Prefill
    t0 = time.perf_counter()
    logits, _, mamba_states, kv_caches = model(
        mx.array([[2000]]), mamba_states=mamba_states, kv_caches=kv_caches
    )
    mx.eval(logits)
    t1 = time.perf_counter()
    prefill_ms = (t1 - t0) * 1000

    # Decode
    t0 = time.perf_counter()
    for step in range(num_tokens):
        logits, _, mamba_states, kv_caches = model(
            mx.array([[2000]]),
            step=step,
            mamba_states=mamba_states,
            kv_caches=kv_caches,
        )
        mx.eval(logits)
    t1 = time.perf_counter()

    total_time = t1 - t0
    tok_s = num_tokens / total_time

    print(f"Prefill:       {prefill_ms:.0f}ms")
    print(f"Decode:        {total_time:.2f}s ({num_tokens} tokens)")
    print(f"Throughput:    {tok_s:.1f} tok/s")
    print(f"Per-token:     {(total_time/num_tokens)*1000:.2f}ms")

    return tok_s


def main():
    print("=" * 70)
    print("Optimized Decode Benchmark - Fused Kernels")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = build_model("checkpoints/latest_sft_cot_model.npz", dtype=mx.bfloat16)

    optimizer = get_fusion_optimizer()

    # Test 1: Baseline (no optimizations)
    print("\n" + "=" * 70)
    print("TEST 1: Baseline (Standard MLX - No Fusions)")
    print("=" * 70)
    optimizer.disable_all()
    optimizer.print_status()

    tok_s_baseline = benchmark_decode(model, num_tokens=128, label="Baseline (bf16, eager)")

    # Test 2: With Quantization
    print("\n" + "=" * 70)
    print("TEST 2: Baseline + 8-Bit Quantization")
    print("=" * 70)
    print("Applying quantization...")
    quant_config = QuantizationConfig(
        quantize_moe=True, quantize_attention=True, quantize_ssm=False
    )
    apply_quantization_to_model(model, quant_config)
    print("✓ Quantization applied")

    tok_s_quant = benchmark_decode(model, num_tokens=128, label="Baseline + 8-bit Quantization")

    # Test 3: With SSM+RoPE Fusion
    print("\n" + "=" * 70)
    print("TEST 3: Baseline + Quantization + SSM+RoPE Fusion")
    print("=" * 70)
    optimizer.enable("ssm_rope_fusion")
    optimizer.print_status()

    tok_s_ssm = benchmark_decode(model, num_tokens=128, label="+ SSM+RoPE Fusion (fused kernel)")

    # Test 4: With All Optimizations
    print("\n" + "=" * 70)
    print("TEST 4: Full Optimization Stack")
    print("=" * 70)
    optimizer.enable_all()
    optimizer.print_status()

    tok_s_all = benchmark_decode(model, num_tokens=128, label="All optimizations enabled")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY & PERFORMANCE ANALYSIS")
    print("=" * 70)

    results = [
        ("Baseline (bf16)", tok_s_baseline, 0),
        ("+ Quantization", tok_s_quant, (tok_s_quant - tok_s_baseline) / tok_s_baseline * 100),
        ("+ SSM+RoPE Fusion", tok_s_ssm, (tok_s_ssm - tok_s_baseline) / tok_s_baseline * 100),
        ("Full Stack", tok_s_all, (tok_s_all - tok_s_baseline) / tok_s_baseline * 100),
    ]

    print(f"\n{'Configuration':<25} {'Throughput':<15} {'Improvement':<15} {'Status'}")
    print("─" * 70)

    for config, tok_s, improvement in results:
        if improvement > 0:
            improvement_str = f"+{improvement:.1f}%"
        elif improvement < 0:
            improvement_str = f"{improvement:.1f}%"
        else:
            improvement_str = "-"

        status = "✓" if tok_s >= tok_s_baseline else "⚠️"
        print(f"{config:<25} {tok_s:>7.1f} tok/s      {improvement_str:>10}      {status}")

    # Target analysis
    print("\n" + "=" * 70)
    print("TARGET ANALYSIS (Scheme B: 40-50 tok/s)")
    print("=" * 70)

    target_min = 40
    target_max = 50

    if tok_s_baseline >= target_min:
        print(f"✅ Baseline alone meets minimum: {tok_s_baseline:.1f} tok/s")
    else:
        gap = target_min - tok_s_baseline
        print(f"⚠️  Baseline below minimum: {tok_s_baseline:.1f} tok/s (gap: {gap:.1f})")

    if tok_s_all >= target_max:
        print(f"✅ Full stack achieves maximum: {tok_s_all:.1f} tok/s")
    elif tok_s_all >= target_min:
        print(f"✅ Full stack achieves target range: {tok_s_all:.1f} tok/s")
    else:
        gap = target_min - tok_s_all
        print(f"⚠️  Full stack below minimum: {tok_s_all:.1f} tok/s (gap: {gap:.1f})")

    # Efficiency
    total_improvement = (tok_s_all - tok_s_baseline) / tok_s_baseline * 100
    print(f"\nTotal improvement (baseline → full stack): {total_improvement:+.1f}%")
    print(f"Per-token reduction: {((1/tok_s_baseline) - (1/tok_s_all))*1000:.2f}ms")

    print("=" * 70 + "\n")

    return {
        "baseline": tok_s_baseline,
        "quantized": tok_s_quant,
        "ssm_fused": tok_s_ssm,
        "all_optimized": tok_s_all,
    }


if __name__ == "__main__":
    try:
        results = main()
        print(f"✓ Benchmark complete")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
