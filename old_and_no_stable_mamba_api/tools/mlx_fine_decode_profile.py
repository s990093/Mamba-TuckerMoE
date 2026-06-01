# -*- coding: utf-8 -*-
"""
Fine-grained decode op table for profile_mlx_infer.py (one token, L==1).

Each timed region is one mx.eval boundary → closer to "operator" view than segment rollups,
at the cost of extra synchronizations (times are not directly comparable to fused full-layer).

Aggregates sum(ms) and call counts across layers; optional shape hints use tensor shapes after op.
"""
from __future__ import annotations

import ast
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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


def _shape_str(a: mx.array) -> str:
    try:
        return str(tuple(int(x) for x in a.shape))
    except Exception:
        return "?"


def _timed(fn: Callable[[], Any]) -> Tuple[float, Any]:
    _sync()
    t0 = time.perf_counter()
    r = fn()
    _eval_any(r)
    _sync()
    return (time.perf_counter() - t0) * 1000.0, r


# Rough mapping for optimization heuristics (not literal kernel names).
KERNEL_HINT: Dict[str, str] = {
    "rms_norm_mamba_in": "fast_rms_norm / reduction-heavy",
    "linear_in_proj": "GEMM",
    "dt_a_lambda_elem": "elemwise+unary",
    "einsum_angle_step": "batched contraction / GEMM-like",
    "cumsum_angles": "scan / prefix-sum",
    "rms_norm_B": "fast_rms_norm / reduction-heavy",
    "rms_norm_C": "fast_rms_norm / reduction-heavy",
    "rope_B": "trig+elemwise",
    "rope_C": "trig+elemwise",
    "x_up_proj": "GEMM-heavy (Linear or Tucker MoE)",
    "einsum_b_x_input": "contraction",
    "ssm_gate_lv": "elemwise",
    "ssm_mix_u": "elemwise",
    "ssm_decode_recurrence_h": "elemwise (decode state)",
    "ssm_chunk_scan": "chunk-parallel scan (prefill-style)",
    "ssm_seq_scan": "sequential scan loop",
    "einsum_y_out": "contraction",
    "linear_y_down": "GEMM",
    "d_skip_mul": "elemwise",
    "rms_pre_gate": "fast_rms_norm / reduction-heavy",
    "linear_mamba_dense": "GEMM",
    "ls_residual_mid": "elemwise",
    "rms_norm_out_proj_in": "fast_rms_norm / reduction-heavy",
    "out_proj": "GEMM-heavy (Linear or Tucker MoE)",
    "ls_residual_out": "elemwise",
    "xf_rms_attn": "fast_rms_norm / reduction-heavy",
    "xf_linear_q": "GEMM",
    "xf_linear_k": "GEMM",
    "xf_linear_v": "GEMM",
    "xf_repeat_kv": "broadcast repeat",
    "xf_kv_write": "scatter/slice_update",
    "xf_arange_mask": "elemwise+compare",
    "xf_sdpa": "Metal SDPA (multi-kernel)",
    "xf_linear_o": "GEMM",
    "xf_ls_residual_attn": "elemwise",
    "xf_rms_ffn": "fast_rms_norm / reduction-heavy",
    "xf_ffn_gate": "GEMM-heavy (MoE)",
    "xf_ffn_up": "GEMM-heavy (MoE)",
    "xf_ffn_down": "GEMM-heavy (MoE)",
    "xf_ls_residual_ffn": "elemwise",
}


@dataclass
class FineOpStat:
    total_ms: float = 0.0
    calls: int = 0
    shape_examples: List[str] = field(default_factory=list)

    def add(self, ms: float, shape_hint: str = "") -> None:
        self.total_ms += ms
        self.calls += 1
        if shape_hint and len(self.shape_examples) < 3 and shape_hint not in self.shape_examples:
            self.shape_examples.append(shape_hint)


@dataclass
class FineDecodeTable:
    ops: Dict[str, FineOpStat] = field(default_factory=lambda: defaultdict(FineOpStat))

    def add(self, name: str, ms: float, arr: Optional[mx.array] = None) -> None:
        sh = _shape_str(arr) if arr is not None else ""
        self.ops[name].add(ms, sh)


def merge_fine_tables(dst: FineDecodeTable, src: FineDecodeTable) -> None:
    for k, st in src.ops.items():
        d = dst.ops[k]
        d.total_ms += st.total_ms
        d.calls += st.calls
        for ex in st.shape_examples:
            if len(d.shape_examples) < 3 and ex and ex not in d.shape_examples:
                d.shape_examples.append(ex)


def profile_mamba_decode_fine(
    blk: Mamba3Block,
    x: mx.array,
    cache: Optional[Tuple[Any, Any, Any]],
    router_temp: mx.array,
) -> Tuple[FineDecodeTable, mx.array, Tuple[Any, Any, Any]]:
    """Mirror decode (L==1) math from ``mlx_profile_components.profile_mamba_decode_step``; return activations + cache."""
    t = FineDecodeTable()
    b_sz, l, _ = x.shape
    assert l == 1
    h = blk.config.n_heads
    g = blk.config.n_groups
    p = blk.config.d_head
    n = blk.config.d_state
    r = blk.config.mimo_rank
    ratio = blk.ratio
    if router_temp is None:
        router_temp = mx.array(0.5, dtype=x.dtype)
    residual_mamba = x

    ms, u = _timed(lambda: rms_norm_fast(x, blk.norm_mamba))
    t.add("rms_norm_mamba_in", ms, u)

    ms, proj_out = _timed(lambda: blk.in_proj(u))
    t.add("linear_in_proj", ms, proj_out)

    z, x_prime, b_param, c_param, dt, a_param, lambda_param = mx.split(proj_out, blk._split_indices, axis=-1)
    x_prime = x_prime.reshape(b_sz, l, h, p)
    mx.eval(z, x_prime, b_param, c_param, dt, a_param, lambda_param)

    ms, pack_dt = _timed(
        lambda: (
            mx.logaddexp(mx.array(0.0, dt.dtype), dt),
            -mx.exp(a_param),
        )
    )
    dt, a = pack_dt
    t.add("dt_a_lambda_elem", ms, dt)

    dt_b = blk._bg(mx.expand_dims(dt, -1)).squeeze(-1)
    a_b = blk._bg(mx.expand_dims(a, -1)).squeeze(-1)
    mx.eval(dt_b, a_b)

    if blk._theta_rep_cache is None:
        theta = mx.exp(blk.theta_log)
        blk._theta_rep_cache = mx.repeat(theta, ratio, axis=0)
        mx.eval(blk._theta_rep_cache)
    theta_rep = blk._theta_rep_cache

    ms, current_angle_step = _timed(lambda: mx.einsum("blh, hn -> blhn", dt_b, theta_rep))
    t.add("einsum_angle_step", ms, current_angle_step)

    if cache is not None:
        _ph, _pi, prev_angle_sum = cache
        ms, angles = _timed(lambda: prev_angle_sum + mx.cumsum(current_angle_step, axis=1))
        t.add("cumsum_angles", ms, angles)
    else:
        ms, angles = _timed(lambda: mx.cumsum(current_angle_step, axis=1))
        t.add("cumsum_angles", ms, angles)
    new_angle_sum = angles[:, -1:]
    mx.eval(new_angle_sum)

    ms, b_reshaped = _timed(
        lambda: rms_norm_fast(b_param.reshape(b_sz, l, g, n * r), blk.norm_B).reshape(b_sz, l, g, n, r)
    )
    t.add("rms_norm_B", ms, b_reshaped)

    ms, c_reshaped = _timed(
        lambda: rms_norm_fast(c_param.reshape(b_sz, l, g, n * r), blk.norm_C).reshape(b_sz, l, g, n, r)
    )
    t.add("rms_norm_C", ms, c_reshaped)

    ms, b_rotated = _timed(lambda: apply_rope(blk._bg(b_reshaped) + blk.bias_B, angles))
    t.add("rope_B", ms, b_rotated)

    ms, c_rotated = _timed(lambda: apply_rope(blk._bg(c_reshaped) + blk.bias_C, angles))
    t.add("rope_C", ms, c_rotated)

    if blk.config.use_kmoe:
        ms, x_up = _timed(lambda: blk.x_up_proj(x_prime.reshape(b_sz, l, -1), router_temp))
        t.add("x_up_proj", ms, x_up)
        x_ssm = x_up.reshape(b_sz, l, h, p, r)
    else:
        ms, x_ssm = _timed(lambda: blk.x_up_proj(x_prime).reshape(b_sz, l, h, p, r))
        t.add("x_up_proj", ms, x_ssm)
    mx.eval(x_ssm)

    ms, input_signal = _timed(lambda: mx.einsum("blhnr, blhpr -> blhnp", b_rotated, x_ssm))
    t.add("einsum_b_x_input", ms, input_signal)

    ms, pack_g = _timed(
        lambda: (
            mx.sigmoid(blk._bg(mx.expand_dims(lambda_param, -1)).squeeze(-1)).reshape(b_sz, l, h, 1, 1),
            dt_b.reshape(b_sz, l, h, 1, 1),
            mx.exp(dt_b * a_b).reshape(b_sz, l, h, 1, 1),
        )
    )
    lv, dv, av = pack_g
    t.add("ssm_gate_lv", ms, lv)

    if cache is not None:
        _prev_h, prev_input, _ = cache
        ip = prev_input
    else:
        ip = mx.concatenate([mx.zeros_like(input_signal[:, :1]), input_signal[:, :-1]], axis=1)
        mx.eval(ip)

    ms, u_ssm = _timed(lambda: lv * dv * input_signal + (1.0 - lv) * dv * av * ip)
    t.add("ssm_mix_u", ms, u_ssm)

    if cache is not None:
        prev_h, _, _ = cache
        ms, h_final = _timed(lambda: prev_h * av[:, 0] + u_ssm[:, 0])
        t.add("ssm_decode_recurrence_h", ms, h_final)
        ms, y_stack = _timed(
            lambda: mx.einsum("bhnp, bhnr -> bhpr", h_final, c_rotated[:, 0])[:, None, ...]
        )
        t.add("einsum_y_out", ms, y_stack)
    elif blk.config.use_parallel_scan and l > 1:

        def _scan():
            return chunk_parallel_scan_mlx(u_ssm, dt_b, a_b, c_rotated, blk.config.chunk_size)

        ms, (y_stack, h_final) = _timed(_scan)
        t.add("ssm_chunk_scan", ms, h_final)
    else:

        def _seq():
            hf = mx.zeros((b_sz, h, n, p), dtype=x.dtype)
            ys = []
            for tt in range(l):
                hf = hf * av[:, tt] + u_ssm[:, tt]
                ys.append(mx.einsum("bhnp,bhnr->bhpr", hf, c_rotated[:, tt])[:, None, ...])
            return mx.concatenate(ys, axis=1), hf

        ms, (y_stack, h_final) = _timed(_seq)
        t.add("ssm_seq_scan", ms, h_final)

    new_cache = (h_final, input_signal[:, -1:], new_angle_sum)
    mx.eval(y_stack, new_cache)

    ms, y = _timed(lambda: blk.y_down_proj(y_stack.reshape(b_sz, l, h, p * r)).reshape(b_sz, l, h * p))
    t.add("linear_y_down", ms, y)

    if blk._D_rep_cache is None:
        blk._D_rep_cache = mx.repeat(blk.D, p, axis=0)
        mx.eval(blk._D_rep_cache)

    ms, y2 = _timed(lambda: y + x_prime.reshape(b_sz, l, h * p) * blk._D_rep_cache)
    t.add("d_skip_mul", ms, y2)

    ms, mamba_out = _timed(lambda: blk.mamba_dense_proj(rms_norm_fast(y2, blk.pre_gate_norm) * silu(z)))
    t.add("linear_mamba_dense", ms, mamba_out)

    ms, mid_x = _timed(lambda: residual_mamba + blk.ls_mamba(mamba_out))
    t.add("ls_residual_mid", ms, mid_x)

    ms, normed_mid = _timed(lambda: rms_norm_fast(mid_x, blk.norm_out_proj))
    t.add("rms_norm_out_proj_in", ms, normed_mid)

    if blk.config.use_kmoe:
        ms, proj_out = _timed(lambda: blk.out_proj(normed_mid, router_temp))
    else:
        ms, proj_out = _timed(lambda: blk.out_proj(normed_mid))
    t.add("out_proj", ms, proj_out)

    ms, out = _timed(lambda: mid_x + blk.ls_out_proj(proj_out))
    t.add("ls_residual_out", ms, out)

    return t, out, new_cache


def profile_transformer_decode_fine(
    blk: TransformerBlock,
    x: mx.array,
    cache: Optional[Tuple[mx.array, mx.array]],
    seq_pos: mx.array,
    router_temp: mx.array,
) -> Tuple[FineDecodeTable, mx.array, Tuple[mx.array, mx.array]]:
    t = FineDecodeTable()
    b, l, d = x.shape
    assert l == 1
    if router_temp is None:
        router_temp = mx.array(0.5, dtype=x.dtype)
    residual = x

    ms, nx = _timed(lambda: rms_norm_fast(x, blk.norm_attn))
    t.add("xf_rms_attn", ms, nx)

    ms, q = _timed(lambda: blk.q_proj(nx).reshape(b, l, blk.num_heads, 64).transpose(0, 2, 1, 3))
    t.add("xf_linear_q", ms, q)

    ms, k = _timed(lambda: blk.k_proj(nx).reshape(b, l, blk.num_kv_heads, 64).transpose(0, 2, 1, 3))
    t.add("xf_linear_k", ms, k)

    ms, v = _timed(lambda: blk.v_proj(nx).reshape(b, l, blk.num_kv_heads, 64).transpose(0, 2, 1, 3))
    t.add("xf_linear_v", ms, v)

    ms, pack_r = _timed(lambda: (mx.repeat(k, blk.kv_groups, axis=1), mx.repeat(v, blk.kv_groups, axis=1)))
    k2, v2 = pack_r
    t.add("xf_repeat_kv", ms, k2)

    k_cache, v_cache = cache
    ms, pack_kv = _timed(
        lambda: (
            mx.slice_update(k_cache, k2, start_indices=seq_pos, axes=(2,)),
            mx.slice_update(v_cache, v2, start_indices=seq_pos, axes=(2,)),
        )
    )
    k_cache, v_cache = pack_kv
    t.add("xf_kv_write", ms, k_cache)

    max_l = k_cache.shape[2]

    ms, mask = _timed(lambda: mx.arange(max_l).reshape(1, 1, 1, max_l) > (seq_pos + l - 1))
    t.add("xf_arange_mask", ms, mask)

    q_attn = q
    if q_attn.dtype != k_cache.dtype:
        q_attn = q_attn.astype(k_cache.dtype)

    ms, attn = _timed(
        lambda: mx.fast.scaled_dot_product_attention(
            q_attn, k_cache, v_cache, scale=1.0 / math.sqrt(64.0), mask=mask
        )
    )
    t.add("xf_sdpa", ms, attn)

    attn_out = attn.transpose(0, 2, 1, 3).reshape(b, l, d)
    if attn_out.dtype != residual.dtype:
        attn_out = attn_out.astype(residual.dtype)

    ms, oproj = _timed(lambda: blk.o_proj(attn_out))
    t.add("xf_linear_o", ms, oproj)
    ms, x1 = _timed(lambda: residual + blk.ls_attn(oproj))
    t.add("xf_ls_residual_attn", ms, x1)

    ms, h = _timed(lambda: rms_norm_fast(x1, blk.norm_ffn))
    t.add("xf_rms_ffn", ms, h)

    if blk.use_kmoe:
        ffn = blk.ffn
        ms, gate = _timed(lambda: ffn.gate_proj(h, router_temp))
        t.add("xf_ffn_gate", ms, gate)
        ms, feat = _timed(lambda: ffn.up_proj(h, router_temp))
        t.add("xf_ffn_up", ms, feat)
        ms, ffn_out = _timed(lambda: ffn.down_proj(fast_silu_gating(gate, feat), router_temp))
        t.add("xf_ffn_down", ms, ffn_out)
    else:
        ms, ffn_out = _timed(lambda: blk.ffn_down(fast_silu_gating(blk.ffn_gate(h), blk.ffn_up(h))))
        t.add("xf_ffn_down", ms, ffn_out)

    ms, out = _timed(lambda: x1 + blk.ls_ffn(ffn_out))
    t.add("xf_ls_residual_ffn", ms, out)

    return t, out, (k_cache, v_cache)


def aggregate_fine_decode_profile(
    backbone: Any,
    h: mx.array,
    caches: list,
    seq_pos: mx.array,
    router_temp: mx.array,
) -> Tuple[FineDecodeTable, mx.array, list]:
    """Single fused pass with per-op timing boundaries; returns merged table, final hidden, updated cache list."""
    total = FineDecodeTable()
    x = h
    new_caches: list = []
    for layer, cache in zip(backbone.layers, caches):
        lt = getattr(layer, "l_type", None)
        if lt == "mamba":
            tab, x, new_c = profile_mamba_decode_fine(layer, x, cache, router_temp)
            merge_fine_tables(total, tab)
            new_caches.append(new_c)
        elif lt == "transformer":
            tab, x, new_c = profile_transformer_decode_fine(layer, x, cache, seq_pos, router_temp)
            merge_fine_tables(total, tab)
            new_caches.append(new_c)
        else:
            raise RuntimeError(f"unknown layer type {lt}")
        nc = new_caches[-1]
        if isinstance(nc, tuple):
            mx.eval(x, *nc)
        else:
            mx.eval(x, nc)
    return total, x, new_caches


def print_fine_table(title: str, table: FineDecodeTable) -> None:
    rows: List[Tuple[str, float, int, float, str, str]] = []
    for name, st in sorted(table.ops.items(), key=lambda kv: -kv[1].total_ms):
        avg = st.total_ms / max(1, st.calls)
        hint = KERNEL_HINT.get(name, "(no hint)")
        ex = "; ".join(st.shape_examples) if st.shape_examples else "—"
        rows.append((name, st.total_ms, st.calls, avg, ex, hint))
    print(title)
    print("-" * 120)
    print(f"{'op_key':<26} {'Σ ms':<12} {'calls':<8} {'avg ms':<12} {'example shapes':<28} {'kernel hint (heuristic)':<40}")
    print("-" * 120)
    for name, sm, c, av, ex, hi in rows:
        print(f"{name:<26} {sm:<12.3f} {c:<8} {av:<12.4f} {ex:<28} {hi:<40}")
    print("-" * 120)
    print(f"{'Σ op times (double-count warning)':<26} {sum(r[1] for r in rows):<12.3f}  (sum of isolated evals ≫ single fused step)")
    print()


# --- mlx-profiler compatible JSON (Trace.save / Trace.load schema) -----------------

_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "uint8": 1, "int4": 0.5, "unknown": 4}

# Mirrors mlx_profiler.profiler._OP_CATEGORIES for common proxy names.
_MLX_OP_CATEGORIES: Dict[str, str] = {
    "matmul": "compute",
    "linear": "compute",
    "scaled_dot_product_attention": "compute",
    "rms_norm": "compute",
    "transpose": "memory",
    "silu": "activation",
    "sigmoid": "activation",
    "elementwise": "elementwise",
    "reduction": "reduction",
    "other": "other",
}


def profiler_dtype_string(cli_dtype: str) -> str:
    return {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}.get(cli_dtype, "float32")


def _fine_key_to_proxy_op_name(op_key: str) -> str:
    k = op_key.lower()
    if "sdpa" in k:
        return "scaled_dot_product_attention"
    if k.startswith("einsum") or "einsum" in k:
        return "matmul"
    if (
        k.startswith("linear_")
        or k.endswith("_proj")
        or "proj" in k
        or "mamba_dense" in k
        or k.startswith("xf_linear")
    ):
        return "linear"
    if "rms" in k or k.endswith("_norm") or "norm_" in k:
        return "rms_norm"
    if k.startswith("rope"):
        return "transpose"
    if k.startswith("ssm") or k.startswith("ls_") or k == "d_skip_mul" or "elem" in k:
        return "elementwise"
    if k.startswith("cumsum"):
        return "reduction"
    if k.startswith("xf_arange") or k.startswith("xf_repeat"):
        return "memory"
    return "other"


def _mlx_category_for_proxy(proxy: str) -> str:
    return _MLX_OP_CATEGORIES.get(proxy, "other")


def _parse_shape_tuple(s: str) -> List[int]:
    s = (s or "").strip()
    if not s or s == "—":
        return []
    try:
        t = ast.literal_eval(s)
        if isinstance(t, tuple):
            return [int(x) for x in t]
        if isinstance(t, list):
            return [int(x) for x in t]
    except Exception:
        return []
    return []


def _memory_bytes_like_mlx_profiler(
    input_shapes: List[List[int]],
    output_shapes: List[List[int]],
    dtype: str,
) -> int:
    b = _DTYPE_BYTES.get(dtype, 4)
    tot = 0
    for shape in input_shapes + output_shapes:
        e = 1
        for d in shape:
            e *= int(d)
        tot += int(e * b)
    return tot


def _flops_like_mlx_profiler(proxy_name: str, input_shapes: List[List[int]]) -> Optional[int]:
    if proxy_name not in ("matmul", "linear"):
        return None
    if len(input_shapes) < 2:
        return None
    a, b = input_shapes[0], input_shapes[1]
    if len(a) < 2 or len(b) < 2:
        return None
    m, k = a[-2], a[-1]
    n = b[-1]
    batch = 1
    for d in a[:-2]:
        batch *= int(d)
    return 2 * batch * int(m) * int(k) * int(n)


def _op_record_to_dict(
    *,
    name: str,
    category: str,
    start_ns: int,
    end_ns: int,
    input_shapes: List[List[int]],
    output_shapes: List[List[int]],
    dtype: str,
    device: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Same keys as mlx_profiler.trace.OpRecord.to_dict() for Trace.save compatibility."""
    duration_us = (end_ns - start_ns) / 1000.0
    flops = _flops_like_mlx_profiler(name, input_shapes)
    mem = _memory_bytes_like_mlx_profiler(input_shapes, output_shapes, dtype)
    intensity = (flops / mem) if flops is not None and mem > 0 else None
    d: Dict[str, Any] = {
        "name": name,
        "category": category,
        "start_ns": int(start_ns),
        "end_ns": int(end_ns),
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "dtype": dtype,
        "device": device,
        "metadata": metadata,
        "duration_us": duration_us,
        "flops": flops,
        "memory_bytes": mem,
        "arithmetic_intensity": intensity,
    }
    return d


def _segment_key_to_proxy(seg_key: str) -> str:
    sk = seg_key.lower()
    if "sdpa" in sk:
        return "scaled_dot_product_attention"
    if "norm" in sk and ("qkv" in sk or "ffn" in sk or sk.startswith("m0")):
        return "rms_norm"
    if "proj" in sk or sk.startswith("ffn"):
        return "linear"
    if "ssm" in sk or sk.startswith("m04") or "einsum" in sk:
        return "matmul"
    return "elementwise"


def fine_decode_table_to_mlx_profiler_ops(
    table: FineDecodeTable,
    *,
    dtype: str,
    start_cur_ns: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Build ``ops`` list compatible with ``mlx_profiler.Trace.save`` / ``Trace.load``.

    One synthetic OpRecord per aggregated fine key; ``start_ns``/``end_ns`` are sequential
    (not true overlap timeline). ``metadata`` holds ``op_key``, ``calls``, ``kernel_hint``.
    Returns ``(ops, next_cursor_ns)`` for chaining after component rollups.
    """
    cur = int(start_cur_ns if start_cur_ns is not None else time.perf_counter_ns())
    ops: List[Dict[str, Any]] = []
    for op_key, st in sorted(table.ops.items(), key=lambda kv: -kv[1].total_ms):
        dur_ns = max(1, int(st.total_ms * 1e6))
        s_ns = cur
        e_ns = cur + dur_ns
        cur = e_ns + 1000
        ex0 = st.shape_examples[0] if st.shape_examples else ""
        sh = _parse_shape_tuple(ex0)
        out_shapes: List[List[int]] = [sh] if sh else []
        proxy = _fine_key_to_proxy_op_name(op_key)
        cat = _mlx_category_for_proxy(proxy)
        meta = {
            "op_key": op_key,
            "profiler": "profile_mlx_infer.fine_decode",
            "calls": st.calls,
            "total_ms": st.total_ms,
            "avg_ms_per_call": st.total_ms / max(1, st.calls),
            "kernel_hint": KERNEL_HINT.get(op_key, ""),
            "mlx_proxy_name": proxy,
        }
        ops.append(
            _op_record_to_dict(
                name=proxy,
                category=cat,
                start_ns=s_ns,
                end_ns=e_ns,
                input_shapes=[],
                output_shapes=out_shapes,
                dtype=dtype,
                device="gpu",
                metadata=meta,
            )
        )
    return ops, cur


def component_rollups_to_mlx_profiler_ops(
    mamba_acc: Dict[str, float],
    xf_acc: Dict[str, float],
    *,
    dtype: str,
    start_cur_ns: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Decode component segment totals (``mlx_profile_components`` keys) as synthetic ops."""
    cur = int(start_cur_ns if start_cur_ns is not None else time.perf_counter_ns())
    ops: List[Dict[str, Any]] = []
    for label, acc in (("mamba", mamba_acc), ("transformer", xf_acc)):
        for seg_key, ms in sorted(acc.items(), key=lambda kv: -kv[1]):
            dur_ns = max(1, int(ms * 1e6))
            s_ns = cur
            e_ns = cur + dur_ns
            cur = e_ns + 1000
            proxy = _segment_key_to_proxy(seg_key)
            cat = _mlx_category_for_proxy(proxy)
            ops.append(
                _op_record_to_dict(
                    name=proxy,
                    category=cat,
                    start_ns=s_ns,
                    end_ns=e_ns,
                    input_shapes=[],
                    output_shapes=[],
                    dtype=dtype,
                    device="gpu",
                    metadata={
                        "tier": "decode_component_rollup",
                        "group": label,
                        "segment_key": seg_key,
                        "total_ms": ms,
                        "profiler": "profile_mlx_infer",
                    },
                )
            )
    return ops, cur


def build_mlx_profiler_trace_dict(
    *,
    name: str,
    dtype: str,
    fine_table: Optional[FineDecodeTable] = None,
    mamba_acc: Optional[Dict[str, float]] = None,
    xf_acc: Optional[Dict[str, float]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    start_time: Optional[float] = None,
    include_component_ops: bool = True,
    include_fine_ops: bool = True,
) -> Dict[str, Any]:
    """
    Top-level object written by ``mlx_profiler.Trace.save`` — loadable with ``Trace.load``
    or ``mlx-profiler view``.
    """
    md: Dict[str, Any] = {
        "source": "profile_mlx_infer",
        "mlx_profiler_schema": "mlx_profiler.trace.Trace v0.2.0 compatible",
    }
    if extra_metadata:
        md.update(extra_metadata)
    ops: List[Dict[str, Any]] = []
    cur: Optional[int] = None
    if include_component_ops and mamba_acc is not None and xf_acc is not None:
        comp_ops, cur = component_rollups_to_mlx_profiler_ops(mamba_acc, xf_acc, dtype=dtype, start_cur_ns=cur)
        ops.extend(comp_ops)
    if include_fine_ops and fine_table is not None:
        fine_ops, _ = fine_decode_table_to_mlx_profiler_ops(
            fine_table, dtype=dtype, start_cur_ns=cur if cur is not None else None
        )
        ops.extend(fine_ops)
    return {
        "name": name,
        "start_time": start_time if start_time is not None else time.time(),
        "metadata": md,
        "ops": ops,
    }
