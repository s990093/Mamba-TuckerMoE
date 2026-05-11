#!/usr/bin/env python3
"""
Profile TuckerMoE sub-components for batch=1 decode.
Identifies exactly which steps are slow within each MoE call.
"""
from __future__ import annotations
import time, sys, math
from pathlib import Path

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "inference"))

import mlx.core as mx
import mlx.nn as nn

from inference.lib.mlx_hybrid_infer import (
    Mamba3Config, Mamba3LanguageModel, rms_norm_fast, fast_scaled_tanh,
    resolve_mlx_checkpoint, strict_load_and_convert, _topk_indices,
)
from metal.ultimate_kernel_lib import UltimateMambaKernels

WARMUP = 10
TRIALS = 50


def _bench(fn, warmup=WARMUP, trials=TRIALS):
    for _ in range(warmup):
        r = fn()
        if isinstance(r, mx.array):
            mx.eval(r)
        elif isinstance(r, (tuple, list)):
            arrs = [x for x in r if isinstance(x, mx.array)]
            if arrs:
                mx.eval(*arrs)
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        r = fn()
        if isinstance(r, mx.array):
            mx.eval(r)
        elif isinstance(r, (tuple, list)):
            arrs = [x for x in r if isinstance(x, mx.array)]
            if arrs:
                mx.eval(*arrs)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def profile_tucker_moe(moe: nn.Module, x_in: mx.array, label: str):
    """Profile a single TuckerMoE(x_in, router_temp) call."""
    b, dim_in = x_in.shape[0], x_in.shape[-1]
    x_flat = x_in.reshape(-1, dim_in)
    rt = mx.array(0.5, dtype=x_in.dtype)
    mx.eval(x_flat)

    _uk = UltimateMambaKernels()

    # Check if quantized
    is_quant_u_in = not hasattr(moe.U_in, "weight") or moe.U_in.weight.dtype not in (mx.float32, mx.float16, mx.bfloat16)
    is_quant_u_out = not hasattr(moe.U_out, "weight") or moe.U_out.weight.dtype not in (mx.float32, mx.float16, mx.bfloat16)

    # 1. Router
    def fn_router():
        raw = moe.router(x_flat)
        capped = fast_scaled_tanh(raw, 10.0)
        t_val = mx.maximum(rt, mx.array(1e-4, dtype=rt.dtype))
        logits = capped / t_val
        probs = mx.softmax(logits, axis=-1)
        indices = _topk_indices(logits, moe.top_k)
        top_k_raw = mx.take_along_axis(probs, indices, axis=-1)
        eps = mx.array(1e-6, dtype=top_k_raw.dtype)
        return top_k_raw / (mx.sum(top_k_raw, axis=-1, keepdims=True) + eps), indices
    t_router = _bench(fn_router)

    probs, indices = fn_router()
    mx.eval(probs, indices)

    # 2. U_in
    def fn_u_in():
        return moe.U_in(x_flat)
    t_u_in = _bench(fn_u_in)
    u_in_out = fn_u_in()
    mx.eval(u_in_out)

    # 3. inner RMSNorm
    def fn_norm():
        return rms_norm_fast(u_in_out, moe.inner_norm)
    t_norm = _bench(fn_norm)
    x_shared = fn_norm()
    mx.eval(x_shared)

    # 4. Core einsum (with scalar fuse for batch=1)
    g_all = moe._get_G()
    num_experts, r3, r2 = g_all.shape

    # 4a. scalar fuse
    scalar_fn = _uk.tucker.build_scalar(r3, r2, num_experts, moe.top_k)
    idx_u32 = indices[0].astype(mx.uint32)
    mx.eval(idx_u32)

    def fn_scalar():
        return _uk.tucker.run_scalar(
            scalar_fn, x_shared[0], g_all, idx_u32, r2, dtype=x_flat.dtype
        )
    t_scalar = _bench(fn_scalar)

    # 4b. einsum fuse
    g_sel = g_all[indices]
    mx.eval(g_sel)
    def fn_einsum():
        return mx.einsum("br,bkrs->bks", x_shared, g_sel)
    t_einsum_plain = _bench(fn_einsum)

    # 4c. weighted sum (for non-scalar paths)
    partial = fn_scalar()
    mx.eval(partial)
    partial_k = partial[None, :]  # (1, r2) for scalar path
    def fn_weighted_sum():
        expert_outs = mx.einsum("br,bkrs->bks", x_shared, g_sel)
        return mx.sum(expert_outs * mx.expand_dims(probs, axis=-1), axis=1)
    t_weighted = _bench(fn_weighted_sum)

    # 5. U_out
    def fn_u_out():
        return moe.U_out(partial_k)
    t_u_out = _bench(fn_u_out)

    u_out_result = fn_u_out()
    mx.eval(u_out_result)

    # 6. Add bias
    def fn_bias():
        return u_out_result + moe.bias
    t_bias = _bench(fn_bias)

    # Full call (scalar_fuse path)
    def fn_full_scalar():
        return moe(x_in, rt, einsum_fuse=True, scalar_fuse=True)
    t_full = _bench(fn_full_scalar)

    # Full call (einsum_fuse path)
    def fn_full_einsum():
        return moe(x_in, rt, einsum_fuse=True, scalar_fuse=False)
    t_full_einsum = _bench(fn_full_einsum)

    total_parts = t_router + t_u_in + t_norm + t_scalar + t_u_out + t_bias

    dim_out = moe.U_out.weight.shape[0] if hasattr(moe.U_out, "weight") else "?"

    print(f"\n{'='*65}")
    print(f"{label}")
    print(f"  dim_in={dim_in}  dim_out={dim_out}  r3={r3}  r2={r2}  E={num_experts}  k={moe.top_k}")
    print(f"  U_in quantized: {is_quant_u_in}  U_out quantized: {is_quant_u_out}")
    print(f"{'='*65}")
    print(f"  {'Step':<30} {'Time (ms)':>10} {'%':>8}")
    print(f"  {'─'*50}")
    steps = [
        ("1. Router+softmax+topk", t_router),
        ("2. U_in (QuantizedLinear)", t_u_in),
        ("3. RMSNorm", t_norm),
        ("4. Core (scalar fuse)", t_scalar),
        ("5. U_out (QuantizedLinear)", t_u_out),
        ("6. + bias", t_bias),
    ]
    for name, t in steps:
        pct = t / total_parts * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:<30} {t:>9.3f}  {pct:>5.1f}% {bar}")
    print(f"  {'─'*50}")
    print(f"  {'Sum of parts':<30} {total_parts:>9.3f}")
    print(f"  {'Full (scalar_fuse)':<30} {t_full:>9.3f}")
    print(f"  {'Full (einsum_fuse only)':<30} {t_full_einsum:>9.3f}")
    print()
    return {
        "router": t_router, "u_in": t_u_in, "norm": t_norm,
        "core_scalar": t_scalar, "u_out": t_u_out, "bias": t_bias,
        "full_scalar": t_full, "full_einsum": t_full_einsum,
        "sum_parts": total_parts,
    }


def main():
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint", default="checkpoints/checkpoint_sft_s27510_model_only.pt")
    pa.add_argument("--quantize", type=int, default=4)
    args = pa.parse_args()

    config = Mamba3Config(
        d_model=768, d_state=64, d_head=64, expand=2,
        num_layers=6, mimo_rank=4, num_kv_heads=4,
        use_parallel_scan=True, chunk_size=64, use_kmoe=True,
        kmoe_num_experts=8, kmoe_top_k=2,
        kmoe_r1=32, kmoe_r2=512, kmoe_r3=256, ffn_expand=6,
    )
    config.tucker_einsum_fuse = True
    config.tucker_scalar_fuse = True

    vocab_size = 32007
    model = Mamba3LanguageModel(config, vocab_size)
    resolved, kind = resolve_mlx_checkpoint(args.checkpoint, repo_root=str(_ROOT))
    if resolved:
        strict_load_and_convert(model, resolved)
    model.apply(lambda x: x.astype(mx.bfloat16))
    if args.quantize > 0:
        nn.quantize(model, group_size=64, bits=args.quantize)
    mx.eval(model.parameters())

    layers = model.backbone.layers
    blk0 = layers[0]  # First Mamba block

    # Create batch=1, seq=1 input
    x_in = mx.ones((1, config.d_model), dtype=mx.bfloat16)
    mx.eval(x_in)

    # x_up_proj: h*p=1536 → h*p*r=6144
    h, p, r_dim = config.n_heads, config.d_head, config.mimo_rank
    x_up_in = mx.ones((1, h * p), dtype=mx.bfloat16)
    mx.eval(x_up_in)

    r1 = profile_tucker_moe(blk0.x_up_proj, x_up_in, "x_up_proj (Mamba Block 0)")
    r2 = profile_tucker_moe(blk0.out_proj, x_in, "out_proj (Mamba Block 0)")

    # Transformer block TuckerMoE (FFN)
    tblk = layers[4]  # First Transformer block
    d_ff = int(math.ceil(config.ffn_expand * config.d_model / 256) * 256)
    r3 = profile_tucker_moe(tblk.ffn.gate_proj, x_in, "gate_proj (Transformer Block)")

    print("═" * 65)
    print("SUMMARY: Optimization Opportunities")
    print("═" * 65)

    total_moe_time = (r1["full_scalar"] + r2["full_scalar"]) * 24 + r3["full_scalar"] * 3 * 6
    print(f"\n  Total MoE time per decode step:")
    print(f"    24× (x_up + out_proj) = {(r1['full_scalar'] + r2['full_scalar']) * 24:.2f} ms")
    print(f"    6× (3× FFN MoE)      = {r3['full_scalar'] * 3 * 6:.2f} ms")
    print(f"    Total MoE:            = {total_moe_time:.2f} ms")
    print()

    # Key: what % is U_in + U_out?
    u_in_out_per_call = r1["u_in"] + r1["u_out"]
    print(f"  U_in + U_out per x_up_proj call: {u_in_out_per_call:.3f} ms")
    print(f"  Router per call: {r1['router']:.3f} ms")
    print(f"  Core (scalar) per call: {r1['core_scalar']:.3f} ms")
    print()
    print(f"  Key insight: If U_in+U_out dominate, fusing won't help much —")
    print(f"  the bottleneck is quantized matmul dispatch overhead.")
    print(f"  Consider: skip quantizing TuckerMoE's small linear layers,")
    print(f"  or write a fused kernel with built-in 4-bit dequant.")
    print("═" * 65)


if __name__ == "__main__":
    main()
