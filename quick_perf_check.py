#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick performance sanity check - verify baseline still works."""

import sys
import time

try:
    import mlx.core as mx
except ImportError:
    print("MLX not installed")
    sys.exit(1)

sys.path.insert(0, "/Users/hungwei/Desktop/Proj/Mamba3-XR")

from mamba3_mlx.mlx_model.hybrid_model import build_model

def quick_benchmark(checkpoint_path, num_tokens=64):
    """Quick 64-token decode benchmark."""
    print(f"\n{'='*70}")
    print(f"Quick Performance Check (64 tokens)")
    print(f"{'='*70}")

    # Load model
    print(f"Loading model...")
    t0 = time.perf_counter()
    model = build_model(checkpoint_path, dtype=mx.bfloat16)
    t1 = time.perf_counter()
    print(f"✓ Model loaded in {(t1-t0)*1000:.0f}ms")

    # Initialize
    mamba_states, kv_caches = model.init_empty_states(batch_size=1)

    # Single prefill
    print(f"Prefill...")
    t0 = time.perf_counter()
    logits, _, mamba_states, kv_caches = model(
        mx.array([[2000]]), mamba_states=mamba_states, kv_caches=kv_caches
    )
    mx.eval(logits)
    t1 = time.perf_counter()
    print(f"✓ Prefill in {(t1-t0)*1000:.0f}ms")

    # Decode benchmark
    print(f"\nDecode benchmark...")
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

    print(f"{'─'*70}")
    print(f"Decode time: {total_time:.2f}s for {num_tokens} tokens")
    print(f"Throughput:  {tok_s:.1f} tok/s")
    print(f"Per-token:   {(total_time/num_tokens)*1000:.2f}ms")
    print(f"{'─'*70}")

    # Expected range
    print(f"\nPerformance assessment:")
    if tok_s >= 23:
        print(f"✓ Baseline OK ({tok_s:.1f} tok/s, target ~23 tok/s)")
    elif tok_s >= 20:
        print(f"⚠️  Slightly below baseline ({tok_s:.1f} tok/s, expected ~23 tok/s)")
    else:
        print(f"✗ Performance degradation ({tok_s:.1f} tok/s, expected ~23 tok/s)")

    print(f"{'='*70}\n")
    return tok_s

if __name__ == "__main__":
    try:
        tok_s = quick_benchmark("checkpoints/latest_sft_cot_model.npz", num_tokens=64)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
