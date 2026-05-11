#!/usr/bin/env python3
"""
Profile every component of a single decode step (batch=1, seq=1).
Forces mx.eval() between measurements for accurate per-component timing.
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
    Mamba3Config, Mamba3LanguageModel, rms_norm_fast, silu, apply_rope,
    resolve_mlx_checkpoint, strict_load_and_convert,
    fast_scaled_tanh,
)
from metal.ultimate_kernel_lib import UltimateMambaKernels

WARMUP = 8
TRIALS = 40


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


def main():
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint", default="checkpoints/checkpoint_sft_s27510_model_only.pt")
    pa.add_argument("--quantize", type=int, default=4)
    pa.add_argument("--fused", action="store_true", default=True)
    pa.add_argument("--no-fused", dest="fused", action="store_false")
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
    config.lookahead_router = False
    config.fused_mamba_mixer = args.fused

    vocab_size = 32007
    model = Mamba3LanguageModel(config, vocab_size)

    resolved, kind = resolve_mlx_checkpoint(args.checkpoint, repo_root=str(_ROOT))
    if resolved:
        strict_load_and_convert(model, resolved)

    model.apply(lambda x: x.astype(mx.bfloat16))
    if args.quantize > 0:
        nn.quantize(model, group_size=64, bits=args.quantize)
    mx.eval(model.parameters())

    _uk = UltimateMambaKernels()

    # Prefill + warm cache
    prefill_ids = mx.array([[1, 15043, 29892, 306, 626]])
    mx.eval(prefill_ids)
    logits, caches = model(prefill_ids)
    mx.eval(logits)

    next_id = mx.array([[29915]])
    seq_pos = mx.array([5])
    logits2, caches2 = model(next_id, caches=caches, seq_pos=seq_pos)
    mx.eval(logits2)

    # ── Full decode timing (no compile) ──
    def fn_full():
        return model(next_id, caches=caches2, seq_pos=mx.array([6]))
    t_full = _bench(fn_full)

    print(f"\n{'='*70}")
    print(f"Full decode step (no compile): {t_full:.3f} ms = {1000/t_full:.1f} tok/s")
    print(f"  quantize={args.quantize}-bit  fused={'ON' if args.fused else 'OFF'}")
    print(f"{'='*70}\n")

    layers = model.backbone.layers
    n_mamba = sum(1 for ly in layers if getattr(ly, "l_type", "mamba") == "mamba")
    n_xfmr = len(layers) - n_mamba

    # ── Per-layer timing with forced eval ──
    print("─" * 70)
    print(f"Per-Layer Timing (median of {TRIALS} trials)")
    print("─" * 70)

    layer_times_list = {i: [] for i in range(len(layers))}
    t_embed_list = []
    t_head_list = []

    for trial in range(WARMUP + TRIALS):
        emb = model.embed(next_id)
        mx.eval(emb)
        current_x = emb
        sp = mx.array([6])
        rt = mx.array(0.5, dtype=emb.dtype)

        for li, (layer, cache) in enumerate(zip(layers, caches2)):
            lt = getattr(layer, "l_type", "mamba")
            mx.eval(current_x)
            t0 = time.perf_counter()
            if lt == "transformer":
                out, nc = layer(current_x, cache=cache, seq_pos=sp, router_temp=rt)
            else:
                out, nc = layer(current_x, cache=cache, router_temp=rt)
            mx.eval(out)
            dt = (time.perf_counter() - t0) * 1000
            current_x = out
            if trial >= WARMUP:
                layer_times_list[li].append(dt)

        mx.eval(current_x)
        t0h = time.perf_counter()
        h = rms_norm_fast(current_x, model.norm)
        lg = fast_scaled_tanh(model.head(h / math.sqrt(config.d_model)), 30.0)
        mx.eval(lg)
        dth = (time.perf_counter() - t0h) * 1000
        if trial >= WARMUP:
            t_head_list.append(dth)

    total_mamba_ms = 0
    total_xfmr_ms = 0
    for i in range(len(layers)):
        lt = getattr(layers[i], "l_type", "mamba")
        times = sorted(layer_times_list[i])
        med = times[len(times)//2]
        tag = "M" if lt == "mamba" else "T"
        print(f"  Layer {i:2d} [{tag}] {med:>8.3f} ms")
        if lt == "mamba":
            total_mamba_ms += med
        else:
            total_xfmr_ms += med

    t_head_list.sort()
    t_head_med = t_head_list[len(t_head_list)//2]

    print(f"\n  {n_mamba}× Mamba total:        {total_mamba_ms:>8.2f} ms")
    print(f"  {n_xfmr}× Transformer total:   {total_xfmr_ms:>8.2f} ms")
    print(f"  LM Head:               {t_head_med:>8.2f} ms")
    sum_layers = total_mamba_ms + total_xfmr_ms + t_head_med
    print(f"  Sum of layers:         {sum_layers:>8.2f} ms = {1000/sum_layers:.1f} tok/s")
    print(f"  Full decode measured:  {t_full:>8.2f} ms = {1000/t_full:.1f} tok/s")
    print()

    # ── Mamba sub-component breakdown ──
    print("═" * 70)
    print("Mamba Block Sub-Component Profiling (Layer 0, median of 40)")
    print("═" * 70)

    blk = layers[0]
    cache0 = caches2[0]
    x_in = model.embed(next_id)
    mx.eval(x_in)
    b_sz, l, _ = x_in.shape
    h, g, p, n, r_dim = config.n_heads, config.n_groups, config.d_head, config.d_state, config.mimo_rank
    ratio = h // g
    rt_val = mx.array(0.5, dtype=x_in.dtype)
    _ef = config.tucker_einsum_fuse
    _sf = getattr(config, "tucker_scalar_fuse", False)

    # 1. Norm + in_proj
    def fn_inproj():
        u = rms_norm_fast(x_in, blk.norm_mamba)
        return blk.in_proj(u)
    t_inproj = _bench(fn_inproj)

    proj = fn_inproj()
    mx.eval(proj)
    z, x_prime_raw, b_param, c_param, dt, a_param, lam_param = mx.split(proj, blk._split_indices, axis=-1)
    mx.eval(z, x_prime_raw, b_param, c_param, dt, a_param, lam_param)

    # 2. x_up_proj (TuckerMoE)
    x_prime = x_prime_raw.reshape(b_sz, l, h, p)
    mx.eval(x_prime)
    def fn_xup():
        return blk.x_up_proj(x_prime.reshape(b_sz, l, -1), rt_val, einsum_fuse=_ef, scalar_fuse=_sf)
    t_xup = _bench(fn_xup)

    x_up = fn_xup()
    mx.eval(x_up)
    x_ssm = x_up.reshape(b_sz, l, h, p, r_dim)
    mx.eval(x_ssm)

    # Prepare SSM inputs
    dt_val = mx.logaddexp(mx.array(0.0, dt.dtype), dt)
    a = -mx.exp(a_param)
    dt_b = mx.repeat(mx.expand_dims(dt_val, -1), ratio, axis=2).squeeze(-1)
    a_b = mx.repeat(mx.expand_dims(a, -1), ratio, axis=2).squeeze(-1)
    if blk._theta_rep_cache is None:
        blk._theta_rep_cache = mx.repeat(mx.exp(blk.theta_log), ratio, axis=0)
        mx.eval(blk._theta_rep_cache)
    mx.eval(dt_b, a_b)
    prev_h, prev_input, prev_angle_sum = cache0

    # 3a. SSM core UNFUSED (for comparison)
    def fn_ssm_unfused():
        theta_rep = blk._theta_rep_cache
        ang_step = mx.einsum("blh, hn -> blhn", dt_b, theta_rep)
        angles = prev_angle_sum + ang_step
        b_re = rms_norm_fast(b_param.reshape(b_sz, l, g, n*r_dim), blk.norm_B).reshape(b_sz, l, g, n, r_dim)
        c_re = rms_norm_fast(c_param.reshape(b_sz, l, g, n*r_dim), blk.norm_C).reshape(b_sz, l, g, n, r_dim)
        b_rot = apply_rope(mx.repeat(b_re, ratio, axis=2) + blk.bias_B, angles)
        c_rot = apply_rope(mx.repeat(c_re, ratio, axis=2) + blk.bias_C, angles)
        inp_sig = mx.einsum("blhnr, blhpr -> blhnp", b_rot, x_ssm)
        lv = mx.sigmoid(mx.repeat(mx.expand_dims(lam_param, -1), ratio, axis=2).squeeze(-1)).reshape(b_sz, l, h, 1, 1)
        dv = dt_b.reshape(b_sz, l, h, 1, 1)
        av = mx.exp(dt_b * a_b).reshape(b_sz, l, h, 1, 1)
        u_ssm = lv * dv * inp_sig + (1.0 - lv) * dv * av * prev_input
        h_final = prev_h * av[:, 0] + u_ssm[:, 0]
        y_s = mx.einsum("bhnp, bhnr -> bhpr", h_final, c_rot[:, 0])[:, None, ...]
        return y_s
    t_ssm_unfused = _bench(fn_ssm_unfused)

    # 3b. SSM core FUSED
    t_ssm_fused = None
    if args.fused:
        kernel = _uk.mamba_mixer.build(h, n, p, r_dim)
        lv_flat = mx.sigmoid(mx.repeat(mx.expand_dims(lam_param, -1), ratio, axis=2).squeeze(-1)).reshape(-1)
        mx.eval(lv_flat)
        def fn_ssm_fused():
            return _uk.mamba_mixer.run(
                kernel,
                b_raw=b_param.reshape(-1), c_raw=c_param.reshape(-1),
                norm_b_w=blk.norm_B.weight, norm_c_w=blk.norm_C.weight,
                bias_b=mx.flatten(blk.bias_B), bias_c=mx.flatten(blk.bias_C),
                theta_rep=blk._theta_rep_cache, prev_angle=prev_angle_sum.reshape(h, n//2),
                x_ssm=x_ssm.reshape(h, p, r_dim),
                prev_h=prev_h.reshape(h, n, p), prev_input=prev_input.reshape(h, n, p),
                dt_b=dt_b.reshape(-1), a_b=a_b.reshape(-1), lv=lv_flat,
                h=h, n=n, p=p, r=r_dim,
            )
        t_ssm_fused = _bench(fn_ssm_fused)
        y_stack = fn_ssm_fused()[2].reshape(b_sz, l, h, p, r_dim)
    else:
        y_stack = fn_ssm_unfused()

    mx.eval(y_stack)

    # 4. y_down_proj + D skip
    if blk._D_rep_cache is None:
        blk._D_rep_cache = mx.repeat(blk.D, p, axis=0)
        mx.eval(blk._D_rep_cache)
    def fn_ydown():
        y = blk.y_down_proj(y_stack.reshape(b_sz, l, h, p*r_dim)).reshape(b_sz, l, h*p)
        return y + x_prime.reshape(b_sz, l, h*p) * blk._D_rep_cache
    t_ydown = _bench(fn_ydown)

    y = fn_ydown()
    mx.eval(y)

    # 5. gate_silu + mamba_dense_proj
    def fn_gate():
        return blk.mamba_dense_proj(rms_norm_fast(y, blk.pre_gate_norm) * silu(z))
    t_gate = _bench(fn_gate)

    mamba_out = fn_gate()
    mx.eval(mamba_out)
    mid_x = x_in + blk.ls_mamba(mamba_out)
    mx.eval(mid_x)

    # 6. out_proj (TuckerMoE)
    normed_mid = rms_norm_fast(mid_x, blk.norm_out_proj)
    mx.eval(normed_mid)
    def fn_outproj():
        return blk.out_proj(normed_mid, rt_val, einsum_fuse=_ef, scalar_fuse=_sf)
    t_outproj = _bench(fn_outproj)

    # ── Report ──
    t_ssm_active = t_ssm_fused if t_ssm_fused is not None else t_ssm_unfused
    mamba_layer_total = t_inproj + t_xup + t_ssm_active + t_ydown + t_gate + t_outproj

    parts = [
        ("norm + in_proj (Linear)", t_inproj),
        ("x_up_proj (TuckerMoE)", t_xup),
        ("SSM core (fused)" if args.fused else "SSM core (unfused)", t_ssm_active),
        ("y_down_proj + D skip", t_ydown),
        ("gate·SiLU + dense_proj", t_gate),
        ("out_proj (TuckerMoE)", t_outproj),
    ]

    estimated_total = mamba_layer_total * n_mamba + total_xfmr_ms + t_head_med

    print(f"\n{'Component':<40} {'1 layer':>8} {'×' + str(n_mamba):>4} {'layers':>7} {'% of total':>10}")
    print("─" * 75)
    for label, t in parts:
        t_all = t * n_mamba
        pct = t_all / estimated_total * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<38} {t:>7.3f}  {t_all:>9.2f} ms  {pct:>5.1f}% {bar}")

    print(f"\n  {'Mamba block (sum)':.<38} {mamba_layer_total:>7.3f}  {mamba_layer_total*n_mamba:>9.2f} ms")
    print(f"  {str(n_xfmr) + '× Transformer':.<38} {'':>7}  {total_xfmr_ms:>9.2f} ms  {total_xfmr_ms/estimated_total*100:>5.1f}%")
    print(f"  {'LM Head':.<38} {t_head_med:>7.3f}  {t_head_med:>9.2f} ms  {t_head_med/estimated_total*100:>5.1f}%")
    print("─" * 75)
    print(f"  {'Estimated total':.<38} {'':>7}  {estimated_total:>9.2f} ms = {1000/estimated_total:.1f} tok/s")
    print(f"  {'Measured full decode':.<38} {'':>7}  {t_full:>9.2f} ms = {1000/t_full:.1f} tok/s")
    overhead_pct = (estimated_total - t_full) / t_full * 100
    print(f"  Overhead from eval barriers:  {estimated_total - t_full:.2f} ms ({overhead_pct:.0f}% inflation)")
    print()

    if t_ssm_fused is not None:
        print(f"  SSM unfused: {t_ssm_unfused:.3f} ms   fused: {t_ssm_fused:.3f} ms   speedup: {t_ssm_unfused/t_ssm_fused:.2f}×")
        print()

    # Normalize to estimate real ms proportions
    scale = t_full / estimated_total
    print("═" * 70)
    print(f"BOTTLENECK RANKING (scaled to {t_full:.1f} ms actual decode)")
    print("═" * 70)
    all_parts = parts + [
        (f"{n_xfmr}× Transformer", total_xfmr_ms / n_mamba),
        ("LM Head", t_head_med / n_mamba),
    ]
    ranked = sorted(all_parts, key=lambda x: x[1], reverse=True)
    for i, (label, t_per_layer) in enumerate(ranked, 1):
        t_real = t_per_layer * n_mamba * scale
        pct = t_real / t_full * 100
        bar = "█" * int(pct / 2)
        print(f"  #{i}  {label:<38} ~{t_real:>6.2f} ms ({pct:>5.1f}%) {bar}")

    print()
    print(f"  Current: {t_full:.2f} ms/tok = {1000/t_full:.1f} tok/s")
    print(f"  Target:  10.0 ms/tok = 100 tok/s")
    print(f"  Gap:     {t_full - 10.0:.2f} ms to shave off")
    print("═" * 70)


if __name__ == "__main__":
    main()
