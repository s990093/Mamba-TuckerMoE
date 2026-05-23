#!/usr/bin/env python3
"""
Benchmark TuckerMoE: full_fuse (single Metal dispatch) vs scalar_fuse (multi-dispatch).
Tests with both dense (bf16) and quantized weights to measure dispatch savings.
"""
from __future__ import annotations
import time, sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "inference"))

import mlx.core as mx
import mlx.nn as nn

from inference.lib.mlx_hybrid_infer import (
    Mamba3Config, TuckerMoE, rms_norm_fast, fast_scaled_tanh, _topk_indices,
)
from metal.ultimate_kernel_lib import UltimateMambaKernels

WARMUP = 15
TRIALS = 60


def _bench(fn, warmup=WARMUP, trials=TRIALS):
    for _ in range(warmup):
        r = fn()
        if isinstance(r, mx.array):
            mx.eval(r)
        elif isinstance(r, (tuple, list)):
            for x in r:
                if isinstance(x, mx.array):
                    mx.eval(x)
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        r = fn()
        if isinstance(r, mx.array):
            mx.eval(r)
        elif isinstance(r, (tuple, list)):
            for x in r:
                if isinstance(x, mx.array):
                    mx.eval(x)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def _is_dense_linear(mod):
    return isinstance(mod, nn.Linear) and not isinstance(mod, nn.QuantizedLinear)


def create_test_moe(dim_in, dim_out, quantize_bits=0):
    """Create a TuckerMoE with random weights, optionally quantized."""
    moe = TuckerMoE(dim_in, dim_out, num_experts=8, top_k=2, r1=32, r2=512, r3=256)
    moe.apply(lambda x: x.astype(mx.bfloat16))
    if quantize_bits > 0:
        nn.quantize(moe, group_size=64, bits=quantize_bits)
    mx.eval(moe.parameters())
    return moe


def dequantize_tucker_moe(moe: TuckerMoE) -> TuckerMoE:
    """Replace QuantizedLinear back to dense nn.Linear (dequantize in-place)."""
    for attr_name in ("U_in", "U_out", "router"):
        layer = getattr(moe, attr_name)
        if isinstance(layer, nn.QuantizedLinear):
            w = layer.weight
            scales = layer.scales
            biases = layer.biases
            bits = layer.bits
            group_size = layer.group_size
            has_bias = hasattr(layer, "bias") and layer.bias is not None

            # Dequantize weight
            w_dense = mx.dequantize(w, scales, biases, group_size, bits)
            out_dim, in_dim = w_dense.shape

            new_linear = nn.Linear(in_dim, out_dim, bias=has_bias)
            new_linear.weight = w_dense.astype(mx.bfloat16)
            if has_bias:
                new_linear.bias = layer.bias.astype(mx.bfloat16)
            setattr(moe, attr_name, new_linear)
    mx.eval(moe.parameters())
    return moe


def benchmark_moe(moe, x_in, label):
    """Benchmark all fusion modes for a TuckerMoE."""
    rt = mx.array(0.5, dtype=x_in.dtype)
    mx.eval(x_in)

    dense_u_in = _is_dense_linear(moe.U_in)
    dense_u_out = _is_dense_linear(moe.U_out)

    # Ensure G cache is built
    moe._get_G()

    # 1. scalar_fuse
    def fn_scalar():
        return moe(x_in, rt, einsum_fuse=True, scalar_fuse=True)
    t_scalar = _bench(fn_scalar)

    # 2. einsum_fuse only
    def fn_einsum():
        return moe(x_in, rt, einsum_fuse=True, scalar_fuse=False)
    t_einsum = _bench(fn_einsum)

    # 3. full_fuse (only works with dense weights)
    t_full = None
    if dense_u_in and dense_u_out:
        def fn_full():
            return moe(x_in, rt, full_fuse=True, einsum_fuse=True, scalar_fuse=True)
        t_full = _bench(fn_full)

    # 4. no fuse (baseline)
    def fn_nofuse():
        return moe(x_in, rt, einsum_fuse=False, scalar_fuse=False)
    t_nofuse = _bench(fn_nofuse)

    print(f"\n  {label}")
    print(f"    U_in dense: {dense_u_in}  U_out dense: {dense_u_out}")
    print(f"    {'Mode':<25} {'Time (ms)':>10} {'Speedup':>8}")
    print(f"    {'─'*45}")
    print(f"    {'no fuse (baseline)':<25} {t_nofuse:>9.3f}  {1.0:>7.2f}×")
    print(f"    {'einsum_fuse':<25} {t_einsum:>9.3f}  {t_nofuse/t_einsum:>7.2f}×")
    print(f"    {'scalar_fuse':<25} {t_scalar:>9.3f}  {t_nofuse/t_scalar:>7.2f}×")
    if t_full is not None:
        print(f"    {'full_fuse (1 dispatch!)':<25} {t_full:>9.3f}  {t_nofuse/t_full:>7.2f}×")
    else:
        print(f"    {'full_fuse':<25} {'N/A (quantized)':>10}")

    return {
        "scalar": t_scalar, "einsum": t_einsum,
        "full": t_full, "nofuse": t_nofuse,
    }


def main():
    print("═" * 65)
    print("TuckerMoE Fusion Benchmark (batch=1, bfloat16)")
    print("═" * 65)

    # Test configs matching the actual model
    configs = [
        ("x_up_proj", 1536, 6144),
        ("out_proj", 768, 768),
        ("gate_proj (Xfmr FFN)", 768, 4608),
    ]

    for name, dim_in, dim_out in configs:
        x_in = mx.ones((1, dim_in), dtype=mx.bfloat16)
        mx.eval(x_in)

        print(f"\n{'─'*65}")
        print(f"  {name}: {dim_in} → {dim_out}")
        print(f"{'─'*65}")

        # A) Dense (no quantization) — full_fuse available
        moe_dense = create_test_moe(dim_in, dim_out, quantize_bits=0)
        r_dense = benchmark_moe(moe_dense, x_in, f"{name} [DENSE bf16]")

        # B) 4-bit quantized — full_fuse disabled
        moe_q4 = create_test_moe(dim_in, dim_out, quantize_bits=4)
        r_q4 = benchmark_moe(moe_q4, x_in, f"{name} [4-BIT QUANT]")

        # C) 4-bit quantized then dequantized — full_fuse re-enabled
        moe_deq = create_test_moe(dim_in, dim_out, quantize_bits=4)
        dequantize_tucker_moe(moe_deq)
        r_deq = benchmark_moe(moe_deq, x_in, f"{name} [DEQUANT → dense]")

        # Summary
        if r_dense["full"] and r_deq["full"]:
            print(f"\n    >>> full_fuse vs scalar_fuse(quant):")
            print(f"        scalar(q4): {r_q4['scalar']:.3f} ms")
            print(f"        full(dense): {r_dense['full']:.3f} ms")
            print(f"        full(deq):   {r_deq['full']:.3f} ms")
            speedup = r_q4['scalar'] / r_deq['full']
            print(f"        Potential speedup: {speedup:.2f}×")

    # Memory estimate
    print(f"\n{'═'*65}")
    print("MEMORY IMPACT OF DEQUANTIZING TuckerMoE")
    print(f"{'═'*65}")
    # Per MoE: router + U_in + U_out
    for name, dim_in, dim_out in configs:
        r3, r2 = 256, 512
        params = dim_in * 8 + dim_in * r3 + r2 * dim_out  # router + U_in + U_out
        mb_bf16 = params * 2 / 1024 / 1024
        mb_q4 = params * 0.5 / 1024 / 1024
        print(f"  {name}: {params/1e6:.1f}M params")
        print(f"    bf16: {mb_bf16:.1f} MB   q4: {mb_q4:.1f} MB   overhead: +{mb_bf16 - mb_q4:.1f} MB")

    # Total across model
    total_overhead = 0
    # 24 Mamba blocks: x_up(1536→6144) + out(768→768)
    for _ in range(24):
        total_overhead += (1536*8 + 1536*256 + 512*6144) * 1.5  # overhead = bf16 - q4
        total_overhead += (768*8 + 768*256 + 512*768) * 1.5
    # 6 Xfmr blocks: 3 × (gate/up: 768→4608, down: 4608→768)
    for _ in range(6):
        total_overhead += 2 * (768*8 + 768*256 + 512*4608) * 1.5
        total_overhead += (4608*8 + 4608*256 + 512*768) * 1.5
    print(f"\n  Total model overhead: +{total_overhead / 1024 / 1024:.1f} MB")
    print("═" * 65)


if __name__ == "__main__":
    main()
