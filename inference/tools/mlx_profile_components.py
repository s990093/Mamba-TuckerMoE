# -*- coding: utf-8 -*-
"""
Mamba / Transformer sub-component timers for profile_mlx_infer.py only.
Mirrors mlx_hybrid_infer.Mamba3Block / TransformerBlock for L==1 decode — keep in sync on changes.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx

from mlx_hybrid_infer import (
    Mamba3Block,
    TransformerBlock,
    apply_rope,
    chunk_parallel_scan_mlx,
    fast_silu_gating,
    rms_norm_fast,
    silu,
)


def _sync() -> None:
    try:
        mx.synchronize()
    except Exception:
        pass


def _eval_any(out: Any) -> None:
    flat: list[mx.array] = []

    def walk(x: Any) -> None:
        if isinstance(x, mx.array):
            flat.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                walk(y)

    walk(out)
    if flat:
        mx.eval(*flat)


def _timed_run(fn):
    """Run fn(), eval outputs, sync; return (ms, result)."""
    _sync()
    t0 = time.perf_counter()
    result = fn()
    _eval_any(result)
    _sync()
    return (time.perf_counter() - t0) * 1000.0, result


def profile_mamba_decode_step(
    blk: Mamba3Block,
    x: mx.array,
    cache: Optional[Tuple[mx.array, mx.array, mx.array]],
    router_temp: mx.array,
) -> Tuple[mx.array, Tuple[mx.array, mx.array, mx.array], Dict[str, float]]:
    times: Dict[str, float] = {}
    b_sz, l, _ = x.shape
    assert l == 1, "profile_mamba_decode_step expects seq_len 1"
    h = blk.config.n_heads
    g = blk.config.n_groups
    p = blk.config.d_head
    n = blk.config.d_state
    r = blk.config.mimo_rank
    ratio = blk.ratio

    if router_temp is None:
        router_temp = mx.array(0.5, dtype=x.dtype)

    residual_mamba = x

    def seg01():
        u = rms_norm_fast(x, blk.norm_mamba)
        proj_out = blk.in_proj(u)
        z, x_prime, b_param, c_param, dt, a_param, lambda_param = mx.split(proj_out, blk._split_indices, axis=-1)
        x_prime = x_prime.reshape(b_sz, l, h, p)
        dt = mx.logaddexp(mx.array(0.0, dt.dtype), dt)
        a = -mx.exp(a_param)
        dt_b = blk._bg(mx.expand_dims(dt, -1)).squeeze(-1)
        a_b = blk._bg(mx.expand_dims(a, -1)).squeeze(-1)
        if blk._theta_rep_cache is None:
            theta = mx.exp(blk.theta_log)
            blk._theta_rep_cache = mx.repeat(theta, ratio, axis=0)
            mx.eval(blk._theta_rep_cache)
        theta_rep = blk._theta_rep_cache
        current_angle_step = mx.einsum("blh, hn -> blhn", dt_b, theta_rep)
        if cache is not None:
            _prev_h, _prev_in, prev_angle_sum = cache
            angles = prev_angle_sum + mx.cumsum(current_angle_step, axis=1)
        else:
            angles = mx.cumsum(current_angle_step, axis=1)
        new_angle_sum = angles[:, -1:]
        return z, x_prime, b_param, c_param, lambda_param, dt_b, a_b, angles, new_angle_sum

    ms, p01 = _timed_run(seg01)
    times["m01_in_norm_proj_angles"] = ms
    z, x_prime, b_param, c_param, lambda_param, dt_b, a_b, angles, new_angle_sum = p01

    def seg02():
        b_reshaped = rms_norm_fast(
            b_param.reshape(b_sz, l, g, n * r), blk.norm_B
        ).reshape(b_sz, l, g, n, r)
        c_reshaped = rms_norm_fast(
            c_param.reshape(b_sz, l, g, n * r), blk.norm_C
        ).reshape(b_sz, l, g, n, r)
        b_rotated = apply_rope(blk._bg(b_reshaped) + blk.bias_B, angles)
        c_rotated = apply_rope(blk._bg(c_reshaped) + blk.bias_C, angles)
        return b_rotated, c_rotated

    ms, p02 = _timed_run(seg02)
    times["m02_norm_bc_rope"] = ms
    b_rotated, c_rotated = p02

    def seg03():
        if blk.config.use_kmoe:
            x_up = blk.x_up_proj(x_prime.reshape(b_sz, l, -1), router_temp)
            x_ssm = x_up.reshape(b_sz, l, h, p, r)
        else:
            x_ssm = blk.x_up_proj(x_prime).reshape(b_sz, l, h, p, r)
        return (x_ssm,)

    ms, p03 = _timed_run(seg03)
    times["m03_x_up_proj"] = ms
    x_ssm = p03[0]

    def seg04():
        input_signal = mx.einsum("blhnr, blhpr -> blhnp", b_rotated, x_ssm)
        lv = mx.sigmoid(blk._bg(mx.expand_dims(lambda_param, -1)).squeeze(-1)).reshape(b_sz, l, h, 1, 1)
        dv = dt_b.reshape(b_sz, l, h, 1, 1)
        av = mx.exp(dt_b * a_b).reshape(b_sz, l, h, 1, 1)
        if cache is not None:
            prev_h, prev_input, _ = cache
            ip = prev_input
        else:
            ip = mx.concatenate([mx.zeros_like(input_signal[:, :1]), input_signal[:, :-1]], axis=1)
        u_ssm = lv * dv * input_signal + (1.0 - lv) * dv * av * ip
        if cache is not None:
            prev_h, _, _ = cache
            h_final = prev_h * av[:, 0] + u_ssm[:, 0]
            y_stack = mx.einsum("bhnp, bhnr -> bhpr", h_final, c_rotated[:, 0])[:, None, ...]
        elif blk.config.use_parallel_scan and l > 1:
            y_stack, h_final = chunk_parallel_scan_mlx(
                u_ssm, dt_b, a_b, c_rotated, blk.config.chunk_size
            )
        else:
            h_final = mx.zeros((b_sz, h, n, p), dtype=x.dtype)
            ys = []
            for t in range(l):
                h_final = h_final * av[:, t] + u_ssm[:, t]
                ys.append(mx.einsum("bhnp,bhnr->bhpr", h_final, c_rotated[:, t])[:, None, ...])
            y_stack = mx.concatenate(ys, axis=1)
        new_cache = (h_final, input_signal[:, -1:], new_angle_sum)
        return y_stack, new_cache

    ms, p04 = _timed_run(seg04)
    times["m04_ssm_core"] = ms
    y_stack, new_cache = p04

    def seg05():
        y = blk.y_down_proj(y_stack.reshape(b_sz, l, h, p * r)).reshape(b_sz, l, h * p)
        if blk._D_rep_cache is None:
            blk._D_rep_cache = mx.repeat(blk.D, p, axis=0)
            mx.eval(blk._D_rep_cache)
        y = y + x_prime.reshape(b_sz, l, h * p) * blk._D_rep_cache
        mamba_out = blk.mamba_dense_proj(rms_norm_fast(y, blk.pre_gate_norm) * silu(z))
        mid_x = residual_mamba + blk.ls_mamba(mamba_out)
        normed_mid = rms_norm_fast(mid_x, blk.norm_out_proj)
        return normed_mid, mid_x

    ms, p05 = _timed_run(seg05)
    times["m05_dense_branch"] = ms
    normed_mid, mid_x = p05

    def seg06():
        if blk.config.use_kmoe:
            proj_out = blk.out_proj(normed_mid, router_temp)
        else:
            proj_out = blk.out_proj(normed_mid)
        out = mid_x + blk.ls_out_proj(proj_out)
        return (out,)

    ms, p06 = _timed_run(seg06)
    times["m06_out_proj"] = ms
    out = p06[0]

    return out, new_cache, times


def profile_transformer_decode_step(
    blk: TransformerBlock,
    x: mx.array,
    cache: Optional[Tuple[mx.array, mx.array]],
    seq_pos: mx.array,
    router_temp: mx.array,
) -> Tuple[mx.array, Tuple[mx.array, mx.array], Dict[str, float]]:
    times: Dict[str, float] = {}
    b, l, d = x.shape
    assert l == 1
    if router_temp is None:
        router_temp = mx.array(0.5, dtype=x.dtype)

    residual = x

    def seg_qkv():
        nx = rms_norm_fast(x, blk.norm_attn)
        q = blk.q_proj(nx).reshape(b, l, blk.num_heads, 64).transpose(0, 2, 1, 3)
        k = blk.k_proj(nx).reshape(b, l, blk.num_kv_heads, 64).transpose(0, 2, 1, 3)
        v = blk.v_proj(nx).reshape(b, l, blk.num_kv_heads, 64).transpose(0, 2, 1, 3)
        k = mx.repeat(k, blk.kv_groups, axis=1)
        v = mx.repeat(v, blk.kv_groups, axis=1)
        return nx, q, k, v

    ms, pack = _timed_run(seg_qkv)
    times["xf_norm_qkv"] = ms
    nx, q, k, v = pack

    def seg_attn():
        k_cache, v_cache = cache
        k_cache = mx.slice_update(k_cache, k, start_indices=seq_pos, axes=(2,))
        v_cache = mx.slice_update(v_cache, v, start_indices=seq_pos, axes=(2,))
        new_cache = (k_cache, v_cache)
        q_attn = q
        if q_attn.dtype != k_cache.dtype:
            q_attn = q_attn.astype(k_cache.dtype)
        max_l = k_cache.shape[2]
        mask = mx.arange(max_l).reshape(1, 1, 1, max_l) > (seq_pos + l - 1)
        attn = mx.fast.scaled_dot_product_attention(
            q_attn, k_cache, v_cache, scale=1.0 / math.sqrt(64.0), mask=mask
        )
        attn_out = attn.transpose(0, 2, 1, 3).reshape(b, l, d)
        if attn_out.dtype != residual.dtype:
            attn_out = attn_out.astype(residual.dtype)
        x1 = residual + blk.ls_attn(blk.o_proj(attn_out))
        return x1, new_cache

    ms, pack2 = _timed_run(seg_attn)
    times["xf_attn_kv_sdpa_o"] = ms
    x_mid, new_cache = pack2

    def seg_ffn_norm():
        return (rms_norm_fast(x_mid, blk.norm_ffn),)

    ms, pack3 = _timed_run(seg_ffn_norm)
    times["xf_ffn_norm"] = ms
    h = pack3[0]

    if blk.use_kmoe:
        ffn = blk.ffn

        def g1():
            return (ffn.gate_proj(h, router_temp),)

        def g2():
            return (ffn.up_proj(h, router_temp),)

        ms, pk = _timed_run(g1)
        times["ffn_gate_proj"] = ms
        gate = pk[0]
        ms, pk = _timed_run(g2)
        times["ffn_up_proj"] = ms
        feat = pk[0]

        def g3():
            return (ffn.down_proj(fast_silu_gating(gate, feat), router_temp),)

        ms, pk = _timed_run(g3)
        times["ffn_down_proj"] = ms
        ffn_out = pk[0]
    else:

        def g_all():
            return (blk.ffn_down(fast_silu_gating(blk.ffn_gate(h), blk.ffn_up(h))),)

        times["ffn_gate_proj"] = 0.0
        times["ffn_up_proj"] = 0.0
        ms, pk = _timed_run(g_all)
        times["ffn_down_proj"] = ms
        ffn_out = pk[0]

    def seg_post():
        return (x_mid + blk.ls_ffn(ffn_out),)

    ms, pk = _timed_run(seg_post)
    times["xf_post_ffn_ls"] = ms
    out = pk[0]

    return out, new_cache, times


def aggregate_decode_component_profile(
    backbone: Any,
    h: mx.array,
    caches: list,
    seq_pos: mx.array,
    router_temp: mx.array,
) -> Tuple[mx.array, list, Dict[str, float], Dict[str, float]]:
    mamba_acc: Dict[str, float] = {}
    xf_acc: Dict[str, float] = {}
    x = h
    new_caches: list = []
    for layer, cache in zip(backbone.layers, caches):
        lt = getattr(layer, "l_type", None)
        if lt == "mamba":
            x, nc, td = profile_mamba_decode_step(layer, x, cache, router_temp)
            for k, v in td.items():
                mamba_acc[k] = mamba_acc.get(k, 0.0) + v
            new_caches.append(nc)
        elif lt == "transformer":
            x, nc, td = profile_transformer_decode_step(layer, x, cache, seq_pos, router_temp)
            for k, v in td.items():
                xf_acc[k] = xf_acc.get(k, 0.0) + v
            new_caches.append(nc)
        else:
            raise RuntimeError(f"unknown layer type {lt}")
    return x, new_caches, mamba_acc, xf_acc
