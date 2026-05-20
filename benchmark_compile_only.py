#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compilation-only decode optimization (simplified).

Focus: Full-graph decode compilation to reduce Python dispatch overhead.
"""

import sys
import time
import argparse

try:
    import mlx.core as mx
except ImportError:
    print("Error: MLX not installed")
    sys.exit(1)

sys.path.insert(0, "/Users/hungwei/Desktop/Proj/Mamba3-XR")

from mamba3_mlx.mlx_model.hybrid_model import build_model
from transformers import AutoTokenizer


def benchmark_with_compilation(
    checkpoint_path,
    tokenizer_path,
    num_tokens=256,
    use_compile=True,
):
    """Benchmark decode with optional compilation."""

    print(f"\n{'='*70}")
    print(f"Decode Benchmark")
    print(f"Compilation: {'ON' if use_compile else 'OFF'}")
    print(f"{'='*70}")

    # Load model
    print(f"\nLoading model...")
    t0 = time.perf_counter()
    model = build_model(checkpoint_path, dtype=mx.bfloat16)
    t1 = time.perf_counter()
    print(f"✓ Model loaded in {(t1-t0)*1000:.0f}ms")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Prefill
    print(f"Prefill: 10 tokens")
    prompt = "Once upon a time, there was a beautiful"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if len(input_ids.shape) > 1:
        input_ids = input_ids[0]
    input_ids = mx.array([input_ids.tolist()])

    t0 = time.perf_counter()
    mamba_states, kv_caches = model.init_empty_states(batch_size=1)
    logits, _, mamba_states, kv_caches = model(
        input_ids, mamba_states=mamba_states, kv_caches=kv_caches
    )
    mx.eval(logits)
    t1 = time.perf_counter()
    prefill_ms = (t1 - t0) * 1000
    print(f"✓ Prefill in {prefill_ms:.0f}ms")

    # Decode
    print(f"\nDecode: {num_tokens} tokens")
    print(f"{'Step':<8} {'Tok/s':<12} {'Time':<12}")
    print("─" * 70)

    t0 = time.perf_counter()

    # Prepare model for compile if enabled
    if use_compile:
        print("(Compiling graph...)")
        # Compile the forward function
        model_fn = mx.compile(model.__call__)
    else:
        model_fn = model.__call__

    for step in range(num_tokens):
        t_step_start = time.perf_counter()

        # Single token decode
        next_token_id = mx.array([[2000]])
        logits, _, mamba_states, kv_caches = model_fn(
            next_token_id,
            step=input_ids.shape[1] + step,
            mamba_states=mamba_states,
            kv_caches=kv_caches,
        )
        mx.eval(logits)

        t_step_end = time.perf_counter()

        # Progress
        if (step + 1) % max(1, num_tokens // 8) == 0 or step + 1 == num_tokens:
            elapsed = t_step_end - t0
            tok_s = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"{step+1:<8} {tok_s:>10.1f} tok/s {elapsed:>10.2f}s")

    t1 = time.perf_counter()
    total_decode_time = t1 - t0
    avg_tok_s = num_tokens / total_decode_time if total_decode_time > 0 else 0

    print("\n" + "─" * 70)
    print(f"Decode Performance:")
    print(f"  Throughput: {avg_tok_s:.1f} tok/s")
    print(f"  Per-token: {(total_decode_time/num_tokens)*1000:.2f}ms")
    print(f"  Total time: {total_decode_time:.2f}s")
    print("─" * 70)

    return avg_tok_s


def main():
    parser = argparse.ArgumentParser(description="Decode optimization benchmark")
    parser.add_argument("--checkpoint", default="checkpoints/latest_sft_cot_model.npz")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer")
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    try:
        # Test with compilation
        print("\n" + "="*70)
        print("TEST 1: With Compilation")
        print("="*70)
        tok_s_compiled = benchmark_with_compilation(
            args.checkpoint, args.tokenizer, args.num_tokens, use_compile=True
        )

        # Test without compilation
        print("\n" + "="*70)
        print("TEST 2: Without Compilation")
        print("="*70)
        tok_s_baseline = benchmark_with_compilation(
            args.checkpoint, args.tokenizer, args.num_tokens, use_compile=False
        )

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Baseline (no compile):    {tok_s_baseline:.1f} tok/s")
        print(f"With compilation:         {tok_s_compiled:.1f} tok/s")

        if tok_s_compiled > tok_s_baseline:
            improvement = (tok_s_compiled - tok_s_baseline) / tok_s_baseline * 100
            print(f"Improvement:              +{improvement:.1f}%")
        else:
            print("No improvement (compilation overhead)")

        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
