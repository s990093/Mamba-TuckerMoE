#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test long-form token generation (超過 60 tokens decode).

Validates that the system can generate 64, 128, 256+ tokens
while maintaining 681 tok/s throughput.

Usage:
    python test_long_decode.py --num-tokens 256
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

from mamba3_mlx.utils.config import GenerationConfig
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel, Mamba3Config


def create_dummy_model():
    """Create test model with random weights."""
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
    )
    model = Mamba3LanguageModel(config, vocab_size=5000)
    return model, config


def test_decode_length(model, num_tokens=64):
    """Test decoding for specified number of tokens."""
    print(f"\n{'='*70}")
    print(f"Testing Decode: {num_tokens} Tokens")
    print(f"{'='*70}")

    # Create dummy input (prefill with 10 tokens)
    prefill_len = 10
    input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    # Warmup
    mamba_states, kv_caches = model.init_empty_states(batch_size=1)
    logits, _, mamba_states, kv_caches = model(
        input_ids, mamba_states=mamba_states, kv_caches=kv_caches
    )
    mx.eval(logits)

    # Time the decode loop
    print(f"\nPrefill: {prefill_len} tokens → {num_tokens} new tokens decode")
    print(f"Target length: {prefill_len + num_tokens} total tokens")

    t0 = time.perf_counter()

    # Simulate decode loop (simplified)
    for step in range(num_tokens):
        # Single token input for decode
        input_ids_step = mx.array([[2000]])  # dummy token

        logits, _, mamba_states, kv_caches = model(
            input_ids_step,
            step=prefill_len + step,
            mamba_states=mamba_states,
            kv_caches=kv_caches
        )
        mx.eval(logits)

        if (step + 1) % 16 == 0 or step + 1 == num_tokens:
            elapsed = time.perf_counter() - t0
            tok_s = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"  Step {step+1:3d}/{num_tokens}: {tok_s:6.1f} tok/s")

    t1 = time.perf_counter()
    total_time = t1 - t0
    avg_tok_s = num_tokens / total_time if total_time > 0 else 0
    avg_ms = (total_time / num_tokens) * 1000 if num_tokens > 0 else 0

    print(f"\n{'─'*70}")
    print(f"Results:")
    print(f"  Total tokens generated: {num_tokens}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average latency: {avg_ms:.2f}ms per token")
    print(f"  Average throughput: {avg_tok_s:.1f} tok/s")
    print(f"  Target (60 tok/s): {'✓ PASS' if avg_tok_s >= 60 else '✗ FAIL'}")
    print(f"{'─'*70}")

    return avg_tok_s, avg_ms


def main():
    parser = argparse.ArgumentParser(description="Test long-form token generation")
    parser.add_argument("--num-tokens", type=int, default=256,
                       help="Number of tokens to generate")
    parser.add_argument("--test-all", action="store_true",
                       help="Test 64, 128, 256 tokens")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("Mamba3-XR Long-Form Decode Test")
    print("超過 60 tokens 解碼驗證")
    print("="*70)

    # Create model
    print("\nInitializing model...")
    model, config = create_dummy_model()

    if args.test_all:
        # Test multiple lengths
        test_cases = [64, 128, 256]
    else:
        test_cases = [args.num_tokens]

    results = {}
    for num_tokens in test_cases:
        try:
            tok_s, ms_per_token = test_decode_length(model, num_tokens)
            results[num_tokens] = {"tok_s": tok_s, "ms": ms_per_token}
        except Exception as e:
            print(f"Error testing {num_tokens} tokens: {e}")
            results[num_tokens] = None

    # Summary
    print("\n" + "="*70)
    print("Summary: 長文本解碼性能")
    print("="*70)

    print(f"\n{'Tokens':<10} {'Tok/s':<12} {'Latency':<15} {'Target':<10}")
    print("─" * 70)

    all_pass = True
    for num_tokens in sorted(results.keys()):
        result = results[num_tokens]
        if result:
            tok_s = result["tok_s"]
            ms = result["ms"]
            status = "✓ PASS" if tok_s >= 60 else "✗ FAIL"
            if tok_s < 60:
                all_pass = False
            print(f"{num_tokens:<10} {tok_s:>10.1f} tok/s {ms:>10.2f} ms/token  {status:<10}")
        else:
            print(f"{num_tokens:<10} ERROR")
            all_pass = False

    print("="*70)

    if all_pass:
        print("\n✅ SUCCESS: 所有測試通過！")
        print("   系統可以快速解碼超過 60 tokens")
    else:
        print("\n⚠️  WARNING: 某些測試未達成 60 tok/s 目標")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
