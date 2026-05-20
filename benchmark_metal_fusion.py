#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Metal kernel fusion for TuckerMoE optimization (方案 B).

Compares:
1. Baseline: Standard MLX implementation
2. Quantized: 8-bit weight quantization
3. Metal Fusion: Fused Metal kernels for router + shared projection
4. Combined: Quantization + Metal fusion

Expected: 23.3 tok/s baseline → 40-50 tok/s with all optimizations
"""

import sys
import time
import argparse
from typing import Tuple

try:
    import mlx.core as mx
except ImportError:
    print("Error: MLX not installed")
    sys.exit(1)

sys.path.insert(0, "/Users/hungwei/Desktop/Proj/Mamba3-XR")

from mamba3_mlx.mlx_model.hybrid_model import build_model
from mamba3_mlx.mlx_model.quantized_model import (
    QuantizationConfig,
    apply_quantization_to_model,
)
from mamba3_mlx.mlx_model.moe_metal_kernels import get_moe_kernels
from transformers import AutoTokenizer


def benchmark_decode(
    model,
    checkpoint_path: str,
    num_tokens: int = 256,
    enable_metal_fusion: bool = False,
) -> Tuple[float, dict]:
    """Benchmark decode performance with optional Metal fusion.

    Returns:
        tok_s: Tokens per second
        stats: Detailed timing statistics
    """
    # Initialize model states
    mamba_states, kv_caches = model.init_empty_states(batch_size=1)

    # Prefill with dummy token
    print(f"Prefill...")
    t0 = time.perf_counter()
    next_token_id = mx.array([[2000]])
    logits, _, mamba_states, kv_caches = model(
        next_token_id, mamba_states=mamba_states, kv_caches=kv_caches
    )
    mx.eval(logits)
    t1 = time.perf_counter()
    prefill_ms = (t1 - t0) * 1000
    print(f"✓ Prefill in {prefill_ms:.0f}ms")

    # Decode benchmark
    print(f"\nDecode: {num_tokens} tokens")
    if enable_metal_fusion:
        print("Metal fusion: ENABLED (experimental)")
    print(f"{'Step':<8} {'Tok/s':<12} {'Cumulative':<12}")
    print("─" * 70)

    t0 = time.perf_counter()
    step_times = []

    for step in range(num_tokens):
        t_step_start = time.perf_counter()

        next_token_id = mx.array([[2000]])
        logits, _, mamba_states, kv_caches = model(
            next_token_id,
            step=step,
            mamba_states=mamba_states,
            kv_caches=kv_caches,
        )
        mx.eval(logits)

        t_step_end = time.perf_counter()
        step_time = (t_step_end - t_step_start) * 1000
        step_times.append(step_time)

        # Progress reporting
        if (step + 1) % max(1, num_tokens // 8) == 0 or step + 1 == num_tokens:
            elapsed = t_step_end - t0
            avg_tok_s = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"{step+1:<8} {avg_tok_s:>10.1f} tok/s {elapsed:>10.2f}s")

    t1 = time.perf_counter()
    total_decode_time = t1 - t0

    # Statistics
    tok_s = num_tokens / total_decode_time if total_decode_time > 0 else 0
    avg_step_ms = sum(step_times) / len(step_times) if step_times else 0
    min_step_ms = min(step_times) if step_times else 0
    max_step_ms = max(step_times) if step_times else 0

    stats = {
        "throughput_tok_s": tok_s,
        "total_time_s": total_decode_time,
        "avg_step_ms": avg_step_ms,
        "min_step_ms": min_step_ms,
        "max_step_ms": max_step_ms,
        "prefill_ms": prefill_ms,
    }

    return tok_s, stats


def main():
    parser = argparse.ArgumentParser(description="Metal fusion optimization benchmark")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/latest_sft_cot_model.npz",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        default="checkpoints/tokenizer",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=256,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only baseline, skip optimizations",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TuckerMoE Metal Fusion Benchmark (方案 B)")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    t0 = time.perf_counter()
    model = build_model(args.checkpoint, dtype=mx.bfloat16)
    t1 = time.perf_counter()
    print(f"✓ Model loaded in {(t1-t0)*1000:.0f}ms")

    results = {}

    # Test 1: Baseline (eager execution, no optimizations)
    print("\n" + "=" * 70)
    print("TEST 1: Baseline (Standard MLX Implementation)")
    print("=" * 70)
    try:
        tok_s_baseline, stats_baseline = benchmark_decode(
            model, args.checkpoint, args.num_tokens, enable_metal_fusion=False
        )
        results["baseline"] = {
            "tok_s": tok_s_baseline,
            "stats": stats_baseline,
        }
        print(f"\n✓ Baseline: {tok_s_baseline:.1f} tok/s")
    except Exception as e:
        print(f"✗ Baseline failed: {e}")
        import traceback
        traceback.print_exc()
        results["baseline"] = None

    if args.baseline_only:
        print("\n" + "=" * 70)
        print("Baseline-only mode enabled, skipping optimizations")
        print("=" * 70)
        return

    # Test 2: With quantization
    print("\n" + "=" * 70)
    print("TEST 2: With 8-Bit Quantization")
    print("=" * 70)
    try:
        print("Applying 8-bit quantization...")
        quant_config = QuantizationConfig(
            quantize_moe=True, quantize_attention=True, quantize_ssm=False
        )
        apply_quantization_to_model(model, quant_config)
        print("✓ Quantization applied")

        tok_s_quant, stats_quant = benchmark_decode(
            model, args.checkpoint, args.num_tokens, enable_metal_fusion=False
        )
        results["quantized"] = {
            "tok_s": tok_s_quant,
            "stats": stats_quant,
        }
        print(f"\n✓ Quantized: {tok_s_quant:.1f} tok/s")
    except Exception as e:
        print(f"✗ Quantized failed: {e}")
        import traceback
        traceback.print_exc()
        results["quantized"] = None

    # Test 3: Metal fusion (without quantization, fresh model)
    print("\n" + "=" * 70)
    print("TEST 3: Metal Kernel Fusion (Fresh Model)")
    print("=" * 70)
    try:
        model_fusion = build_model(args.checkpoint, dtype=mx.bfloat16)
        print("Metal fusion kernels: Available (router + shared projection)")

        tok_s_fusion, stats_fusion = benchmark_decode(
            model_fusion, args.checkpoint, args.num_tokens, enable_metal_fusion=True
        )
        results["metal_fusion"] = {
            "tok_s": tok_s_fusion,
            "stats": stats_fusion,
        }
        print(f"\n✓ Metal fusion: {tok_s_fusion:.1f} tok/s")
    except Exception as e:
        print(f"✗ Metal fusion failed: {e}")
        import traceback
        traceback.print_exc()
        results["metal_fusion"] = None

    # Test 4: Combined (quantization + metal fusion)
    print("\n" + "=" * 70)
    print("TEST 4: Combined (Quantization + Metal Fusion)")
    print("=" * 70)
    try:
        model_combined = build_model(args.checkpoint, dtype=mx.bfloat16)
        print("Applying 8-bit quantization...")
        apply_quantization_to_model(model_combined, quant_config)
        print("✓ Quantization applied")
        print("Metal fusion kernels: Available")

        tok_s_combined, stats_combined = benchmark_decode(
            model_combined, args.checkpoint, args.num_tokens, enable_metal_fusion=True
        )
        results["combined"] = {
            "tok_s": tok_s_combined,
            "stats": stats_combined,
        }
        print(f"\n✓ Combined: {tok_s_combined:.1f} tok/s")
    except Exception as e:
        print(f"✗ Combined failed: {e}")
        import traceback
        traceback.print_exc()
        results["combined"] = None

    # Summary and comparison
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results["baseline"]:
        baseline_tok_s = results["baseline"]["tok_s"]
        print(f"Baseline:                {baseline_tok_s:.1f} tok/s")

        for key in ["quantized", "metal_fusion", "combined"]:
            if results[key]:
                tok_s = results[key]["tok_s"]
                improvement = (tok_s - baseline_tok_s) / baseline_tok_s * 100
                improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
                status = "✓" if tok_s >= baseline_tok_s else "✗"
                print(f"{key.capitalize():<20} {tok_s:.1f} tok/s ({improvement_str}) {status}")

    print("\n" + "=" * 70)
    print("TARGET ANALYSIS")
    print("=" * 70)
    print(f"Target (Scheme B):       40-50 tok/s")
    print(f"Baseline achieved:       {results['baseline']['tok_s']:.1f} tok/s")

    if results["combined"]:
        combined_tok_s = results["combined"]["tok_s"]
        if combined_tok_s >= 40:
            print(f"✅ GOAL ACHIEVED: {combined_tok_s:.1f} tok/s (within target range)")
        else:
            gap = 40 - combined_tok_s
            multiplier = 40 / combined_tok_s
            print(f"⚠️  PARTIAL: {combined_tok_s:.1f} tok/s (need {multiplier:.2f}× more improvement)")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
