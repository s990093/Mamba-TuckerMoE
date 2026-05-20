#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark quantization impact on decode performance.

Compare:
1. Baseline (fp32/bf16): 39.5 tok/s (current)
2. Quantized MoE+Attention: Expected 40-50 tok/s
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


def benchmark_decode(model, num_tokens=128, label=""):
    """Benchmark decode performance."""
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

    print(f"Prefill:     {prefill_ms:.0f}ms")
    print(f"Decode:      {total_time:.2f}s for {num_tokens} tokens")
    print(f"Throughput:  {tok_s:.1f} tok/s")
    print(f"Per-token:   {(total_time/num_tokens)*1000:.2f}ms")

    return tok_s


def main():
    print("=" * 70)
    print("Quantization Impact Benchmark (Scheme B)")
    print("=" * 70)

    print("\nLoading baseline model...")
    model_baseline = build_model("checkpoints/latest_sft_cot_model.npz", dtype=mx.bfloat16)

    # Test 1: Baseline
    tok_s_baseline = benchmark_decode(model_baseline, num_tokens=128, label="TEST 1: Baseline (bf16)")

    # Test 2: Quantized
    print("\nApplying 8-bit quantization...")
    quant_config = QuantizationConfig(
        quantize_moe=True, quantize_attention=True, quantize_ssm=False
    )
    apply_quantization_to_model(model_baseline, quant_config)
    print("✓ Quantization applied")

    tok_s_quant = benchmark_decode(model_baseline, num_tokens=128, label="TEST 2: Quantized (8-bit MoE+Attn)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline (bf16):        {tok_s_baseline:.1f} tok/s")
    print(f"Quantized (8-bit):      {tok_s_quant:.1f} tok/s")

    improvement = (tok_s_quant - tok_s_baseline) / tok_s_baseline * 100
    if tok_s_quant > tok_s_baseline:
        print(f"Improvement:            +{improvement:.1f}%")
        print(f"✓ Quantization helps")
    else:
        print(f"Change:                 {improvement:.1f}%")
        if tok_s_quant < tok_s_baseline * 0.95:
            print(f"✗ Quantization hurts performance")
        else:
            print(f"≈ Quantization neutral")

    target_toks = 40
    if tok_s_quant >= target_toks:
        print(f"\n✅ Target achieved: {tok_s_quant:.1f} tok/s (≥ {target_toks} tok/s)")
    else:
        gap = target_toks - tok_s_quant
        need_multiplier = target_toks / tok_s_quant
        print(f"\n⚠️  Need {need_multiplier:.2f}× improvement to reach {target_toks} tok/s")
        print(f"    Currently: {tok_s_quant:.1f} tok/s")
        print(f"    Gap: {gap:.1f} tok/s")

    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
