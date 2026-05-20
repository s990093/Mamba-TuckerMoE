#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized decode benchmark with compilation and quantization.

方案 B: 量化 + 編譯優化達 40-50 tok/s
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
from mamba3_mlx.mlx_model.quantized_model import (
    CompiledDecodeConfig,
    QuantizationConfig,
    apply_quantization_to_model,
)
from transformers import AutoTokenizer


def benchmark_optimized_decode(
    checkpoint_path,
    tokenizer_path,
    num_tokens=256,
    enable_quantize=True,
    enable_compile=True,
):
    """Benchmark optimized decode with quantization and compilation."""

    print(f"\n{'='*70}")
    print(f"Optimized Decode Benchmark (方案 B)")
    print(f"{'='*70}")
    print(f"Quantization: {'ON' if enable_quantize else 'OFF'}")
    print(f"Compilation: {'ON' if enable_compile else 'OFF'}")

    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    t0 = time.perf_counter()
    model = build_model(checkpoint_path, dtype=mx.bfloat16)
    t1 = time.perf_counter()
    model_load_ms = (t1 - t0) * 1000
    print(f"✓ Model loaded in {model_load_ms:.0f}ms")

    # Apply optimizations
    if enable_quantize:
        print("\nApplying 8-bit quantization to MoE layers...")
        quant_config = QuantizationConfig(
            quantize_moe=True, quantize_attention=True, quantize_ssm=False
        )
        apply_quantization_to_model(model, quant_config)

    compile_config = None
    if enable_compile:
        print("Enabling full-graph compilation for decode...")
        compile_config = CompiledDecodeConfig(compile_full_decode=True)

    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"✓ Tokenizer loaded (vocab_size={len(tokenizer)})")

    # Prefill
    print(f"\nPrefill: 10 tokens")
    prompt = "Once upon a time, there was a beautiful"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if len(input_ids.shape) > 1:
        input_ids = input_ids[0]
    input_ids = mx.array([input_ids.tolist()])

    t0 = time.perf_counter()
    mamba_states, kv_caches = model.init_empty_states(batch_size=1)

    # Optionally compile prefill
    if enable_compile:
        forward_fn = mx.compile(model.__call__)
    else:
        forward_fn = model.__call__

    logits, _, mamba_states, kv_caches = forward_fn(
        input_ids, mamba_states=mamba_states, kv_caches=kv_caches
    )
    mx.eval(logits)
    t1 = time.perf_counter()
    prefill_ms = (t1 - t0) * 1000
    print(f"✓ Prefill completed in {prefill_ms:.0f}ms")

    # Decode benchmark
    print(f"\nDecode: {num_tokens} tokens (optimized)")
    print(f"{'Step':<8} {'Tok/s':<12} {'Cumulative':<12}")
    print("─" * 70)

    t0 = time.perf_counter()
    decode_times = []

    for step in range(num_tokens):
        t_step_start = time.perf_counter()

        # Get compiled function if available
        if compile_config and step < 256:
            forward_fn = compile_config.get_compiled_forward(model, step)
        else:
            forward_fn = model.__call__

        # Single token decode
        next_token_id = mx.array([[2000]])  # dummy token
        logits, _, mamba_states, kv_caches = forward_fn(
            next_token_id,
            step=input_ids.shape[1] + step,
            mamba_states=mamba_states,
            kv_caches=kv_caches,
        )
        mx.eval(logits)

        t_step_end = time.perf_counter()
        step_time = (t_step_end - t_step_start) * 1000
        decode_times.append(step_time)

        # Progress reporting
        if (step + 1) % max(1, num_tokens // 8) == 0 or step + 1 == num_tokens:
            elapsed = t_step_end - t0
            avg_tok_s = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"{step+1:<8} {avg_tok_s:>10.1f} tok/s {elapsed:>10.2f}s")

    t1 = time.perf_counter()
    total_decode_time = t1 - t0

    # Statistics
    print("\n" + "─" * 70)
    print(f"Results:")
    print(f"  Total decode time: {total_decode_time:.2f}s")
    print(f"  Average per-token: {(total_decode_time/num_tokens)*1000:.2f}ms")
    print(f"  Average throughput: {num_tokens/total_decode_time:.1f} tok/s")
    print(f"  Target (60 tok/s): {'✓ PASS' if (num_tokens/total_decode_time) >= 60 else '⚠️  PARTIAL'}")
    print("─" * 70)

    # Overall metrics
    print(f"\nPerformance Metrics:")
    print(f"  Prefill:        {prefill_ms:.0f}ms")
    print(f"  Decode:         {total_decode_time*1000:.0f}ms ({num_tokens} tokens)")
    print(f"  Throughput:     {num_tokens/total_decode_time:.1f} tok/s")

    return num_tokens / total_decode_time


def main():
    parser = argparse.ArgumentParser(
        description="Optimized decode benchmark with quantization and compilation"
    )
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
        "--no-quantize",
        action="store_true",
        help="Disable quantization",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable compilation",
    )
    args = parser.parse_args()

    try:
        tok_s = benchmark_optimized_decode(
            args.checkpoint,
            args.tokenizer,
            args.num_tokens,
            enable_quantize=not args.no_quantize,
            enable_compile=not args.no_compile,
        )

        print("\n" + "=" * 70)
        print(f"Performance: {tok_s:.1f} tok/s")

        # Compare to baseline
        baseline = 23.3
        improvement = (tok_s - baseline) / baseline * 100
        print(f"Baseline:    {baseline:.1f} tok/s")
        print(f"Improvement: +{improvement:.1f}%")

        if tok_s >= 40:
            print(f"\n✅ SUCCESS: Achieved {tok_s:.1f} tok/s (target: 40-50 tok/s)")
        elif tok_s >= 30:
            print(f"\n✓ GOOD: Achieved {tok_s:.1f} tok/s (方案 A range)")
        else:
            print(f"\n⚠️  Current: {tok_s:.1f} tok/s (keep optimizing)")

        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
