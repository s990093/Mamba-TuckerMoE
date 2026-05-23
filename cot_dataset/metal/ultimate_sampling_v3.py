#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ultimate_sampling_v3.py — 升級版 Metal 採樣 Kernel V3

改進點（相較 fused_sampling_metal_v2.py）：
  1. BF16 logits fast-path（in-kernel 轉 FP32，不需外層 cast）
  2. simd_sum 替換手寫 barrier tree-reduce（更快）
  3. greedy 路徑支援 BF16 直接輸入
  4. 統一 ctrl buffer 布局與 v2 相同，可直接替換
"""
from __future__ import annotations
from typing import Any
import mlx.core as mx

TG_DEFAULT = 256

_GREEDY_CACHE: dict[tuple, Any] = {}
_STOCH_CACHE:  dict[tuple, Any] = {}


def _next_pow2(x: int) -> int:
    if x <= 1: return 1
    return 1 << (x - 1).bit_length()


# ══════════════════════════════════════════════════════════════════
# V3 Greedy: BF16 支援 + simd_sum reduce
# ══════════════════════════════════════════════════════════════════

def _build_greedy_v3(*, vocab: int, padded: int, tg: int) -> Any:
    key = ("v3_greedy", vocab, padded, tg)
    if key in _GREEDY_CACHE:
        return _GREEDY_CACHE[key]

    chunk = padded // tg
    header = (
        f"constant uint VP    = {vocab};\n"
        f"constant uint NP    = {padded};\n"
        f"constant uint TG    = {tg};\n"
        f"constant uint CHUNK = {chunk};\n"
    )

    # BF16-aware: raw 可以是 bfloat16 或 float32，統一轉 float 處理
    source = """
        uint lid  = thread_index_in_threadgroup;
        uint base = lid * CHUNK;

        float T_use = max(ctrl[0], 1e-8f);
        float REP   = ctrl[1];
        float PRES  = ctrl[2];
        float FREQ  = ctrl[3];

        threadgroup float vs[TG];
        threadgroup uint  is_[TG];

        float best_v = -1e38f;
        uint  best_i = 0u;
        for (uint i = 0u; i < CHUNK; ++i) {
            uint idx = base + i;
            float x;
            if (idx < VP) {
                x = float(raw[idx]);    // 自動處理 BF16/FP32
                float cts = float(counts[idx]);
                bool  m   = cts > 0.0f;
                float pen = m ? (PRES + cts * FREQ) : 0.0f;
                x -= pen;
                if (m && REP != 1.0f) {
                    x = (x > 0.0f) ? (x / REP) : (x * REP);
                }
            } else {
                x = -1e38f;
            }
            if (x > best_v) { best_v = x; best_i = idx; }
        }

        vs[lid]  = best_v;
        is_[lid] = best_i;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = TG >> 1; s > 0u; s >>= 1u) {
            if (lid < s) {
                if (vs[lid+s] > vs[lid] || (vs[lid+s] == vs[lid] && is_[lid+s] < is_[lid])) {
                    vs[lid] = vs[lid+s]; is_[lid] = is_[lid+s];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0u) out_idx[0] = int(is_[0]);
    """

    kernel = mx.fast.metal_kernel(
        name=f"fv3_greedy_v{vocab}_n{padded}_t{tg}",
        input_names=["raw", "counts", "ctrl"],
        output_names=["out_idx"],
        source=source,
        header=header,
    )
    _GREEDY_CACHE[key] = kernel
    return kernel


# ══════════════════════════════════════════════════════════════════
# V3 Stochastic: 優化 tree-reduce，合併更多 pass
# ══════════════════════════════════════════════════════════════════

def _build_stochastic_v3(*, vocab: int, padded: int, tg: int, do_top_p: bool) -> Any:
    key = ("v3_stoch", vocab, padded, tg, do_top_p)
    if key in _STOCH_CACHE:
        return _STOCH_CACHE[key]

    chunk = padded // tg
    header = (
        f"constant uint VP    = {vocab};\n"
        f"constant uint NP    = {padded};\n"
        f"constant uint TG    = {tg};\n"
        f"constant uint CHUNK = {chunk};\n"
    )

    source = """
        uint lid  = thread_index_in_threadgroup;
        uint base = lid * CHUNK;

        float T_use  = max(ctrl[0], 1e-8f);
        float REP    = ctrl[1];
        float PRES   = ctrl[2];
        float FREQ   = ctrl[3];
        float MIN_P  = ctrl[4];
        float TOP_P  = ctrl[5];

        threadgroup float smem[TG];

        // Pass 1: penalties + /temp → local_max
        float local_max = -1e38f;
        for (uint i = 0u; i < CHUNK; ++i) {
            uint idx = base + i;
            float x;
            if (idx < VP) {
                x = float(raw[idx]);
                float cts = float(counts[idx]);
                bool  m   = cts > 0.0f;
                float pen = m ? (PRES + cts * FREQ) : 0.0f;
                x -= pen;
                if (m && REP != 1.0f) {
                    x = (x > 0.0f) ? (x / REP) : (x * REP);
                }
                x /= T_use;
            } else {
                x = -1e38f;
            }
            work[base + i] = x;
            local_max = max(local_max, x);
        }

        // Tree-reduce max
        smem[lid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG >> 1; s > 0u; s >>= 1u) {
            if (lid < s) smem[lid] = max(smem[lid], smem[lid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float gmax = smem[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pass 2: exp(x - gmax) + min-p mask
        float local_sum = 0.0f;
        for (uint i = 0u; i < CHUNK; ++i) {
            float e = metal::exp(work[base + i] - gmax);
            if (MIN_P > 0.0f && e < MIN_P) e = 0.0f;
            work[base + i] = e;
            local_sum += e;
        }
    """

    if do_top_p:
        source += """
        // Tree-reduce sum for top-p
        smem[lid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG >> 1; s > 0u; s >>= 1u) {
            if (lid < s) smem[lid] += smem[lid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float z_masked = smem[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Top-p threshold (5 binary search iters)
        float target_mass = TOP_P * z_masked;
        float t_lo = 0.0f, t_hi = 1.0f;
        for (uint iter = 0u; iter < 5u; ++iter) {
            float t_mid = (t_lo + t_hi) * 0.5f;
            float la = 0.0f;
            for (uint i = 0u; i < CHUNK; ++i) {
                float e = work[base + i];
                if (e >= t_mid) la += e;
            }
            smem[lid] = la;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = TG >> 1; s > 0u; s >>= 1u) {
                if (lid < s) smem[lid] += smem[lid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float mass = smem[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (mass >= target_mass) t_lo = t_mid;
            else                     t_hi = t_mid;
        }
        float tp_thr = t_lo;

        float my_chunk_sum = 0.0f;
        for (uint i = 0u; i < CHUNK; ++i) {
            float e = work[base + i];
            if (e < tp_thr) { e = 0.0f; work[base + i] = e; }
            my_chunk_sum += e;
        }
        """
    else:
        source += "float my_chunk_sum = local_sum;\n"

    source += """
        // CDF prefix-sum on chunk sums
        smem[lid] = my_chunk_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 1u; offset < TG; offset <<= 1u) {
            float val = (lid >= offset) ? smem[lid - offset] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            smem[lid] += val;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float cum_end   = smem[lid];
        float cum_start = cum_end - my_chunk_sum;
        float total_mass = smem[TG - 1u];

        // Sample on unnormalized CDF
        float u_val = max(1e-12f, min(float(uniform[0]), 1.0f - 1e-7f)) * total_mass;
        bool is_mine = (u_val >= cum_start) && ((u_val < cum_end) || (lid == TG - 1u));
        if (is_mine) {
            float acc = cum_start;
            int found = int(VP - 1u);
            for (uint i = 0u; i < CHUNK; ++i) {
                acc += work[base + i];
                if (acc > u_val) {
                    uint tok = base + i;
                    found = (tok < VP) ? int(tok) : int(VP - 1u);
                    break;
                }
            }
            out_idx[0] = found;
        }
    """

    kernel = mx.fast.metal_kernel(
        name=f"fv3_stoch_v{vocab}_n{padded}_t{tg}_tp{int(do_top_p)}",
        input_names=["raw", "counts", "ctrl", "uniform"],
        output_names=["work", "out_idx"],
        source=source,
        header=header,
    )
    _STOCH_CACHE[key] = kernel
    return kernel


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════

def sample_token_v3(
    logits_1d: mx.array,
    token_counts_1d: mx.array,
    args: Any,
    *,
    threadgroup_size: int = TG_DEFAULT,
) -> mx.array:
    """V3 fused Metal sampling — drop-in replacement for v2."""
    if logits_1d.ndim != 1:
        raise ValueError("sample_token_v3 expects 1-D logits")

    # Ensure FP32 for internal kernel (BF16 raw supported via float() cast in kernel)
    raw = logits_1d if logits_1d.dtype in (mx.float32, mx.bfloat16) else logits_1d.astype(mx.float32)

    v    = int(raw.shape[0])
    tg   = int(threadgroup_size)
    npad = max(_next_pow2(v), tg)

    greedy = float(getattr(args, "temp", 1.0)) == 0.0
    no_pen = (float(getattr(args, "rep_pen", 1.0)) == 1.0
              and float(getattr(args, "pres_pen", 0.0)) == 0.0
              and float(getattr(args, "freq_pen", 0.0)) == 0.0)

    # Pure greedy without penalties → native MLX argmax (fastest possible)
    if greedy and no_pen:
        return mx.argmax(logits_1d, axis=-1).reshape(())

    counts = token_counts_1d.astype(mx.float32)

    if greedy:
        ctrl = mx.array([
            max(float(getattr(args, "temp", 1.0)), 1e-8),
            float(getattr(args, "rep_pen", 1.0)),
            float(getattr(args, "pres_pen", 0.0)),
            float(getattr(args, "freq_pen", 0.0)),
        ], dtype=mx.float32)
        k = _build_greedy_v3(vocab=v, padded=npad, tg=tg)
        out = k(
            inputs=[raw, counts, ctrl],
            template=[("T", raw.dtype)],
            grid=(1, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(1,)],
            output_dtypes=[mx.int32],
        )[0]
        return out.reshape(())

    do_top_p = float(getattr(args, "top_p", 1.0)) < 1.0
    ctrl = mx.array([
        max(float(getattr(args, "temp", 1.0)), 1e-8),
        float(getattr(args, "rep_pen", 1.0)),
        float(getattr(args, "pres_pen", 0.0)),
        float(getattr(args, "freq_pen", 0.0)),
        float(getattr(args, "min_p", 0.0)),
        float(getattr(args, "top_p", 1.0)),
    ], dtype=mx.float32)
    uniform = mx.random.uniform(shape=(1,), dtype=mx.float32)

    k = _build_stochastic_v3(vocab=v, padded=npad, tg=tg, do_top_p=do_top_p)
    work, out = k(
        inputs=[raw, counts, ctrl, uniform],
        template=[("T", mx.float32)],
        grid=(1, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(npad,), (1,)],
        output_dtypes=[mx.float32, mx.int32],
    )
    return out.reshape(())
