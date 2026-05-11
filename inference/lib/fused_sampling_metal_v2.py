#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fully-fused single-dispatch Metal sampling for MLX decode (v2).

Three kernel paths, each minimizing dispatch count:

  1) **Greedy** — 1 dispatch: penalties → argmax
  2) **Stochastic (no top-p)** — 1 dispatch: penalties+temp → exp → min-p mask → CDF → sample
  3) **Stochastic (with top-p)** — 1 dispatch: penalties+temp → exp → min-p mask → top-p sweep → CDF → sample

This implementation uses the "Unnormalized Probability Shortcut":
Since `pmax = 1/Z`, the condition `p_i >= min_p * pmax` simplifies to `exp(x_i - gmax) >= min_p`.
This allows us to skip all floating point division for renormalization! We sample directly
on the unnormalized cumulative distribution function (CDF).

Enable with ``--fused-sample-metal-v2`` in ``inference/benchmark_mlx.py``.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

TG_DEFAULT = 256

_GREEDY_CACHE: dict[tuple, Any] = {}
_STOCH_CACHE: dict[tuple, Any] = {}


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


# ═══════════════════════════════════════════════════════════════════════════
# Greedy kernel: penalties + argmax  (1 dispatch)
# ═══════════════════════════════════════════════════════════════════════════

def _build_greedy_kernel(*, vocab: int, padded: int, tg: int) -> Any:
    key = (vocab, padded, tg)
    if key in _GREEDY_CACHE:
        return _GREEDY_CACHE[key]

    chunk = padded // tg
    header = (
        f"constant uint VP  = {vocab};\n"
        f"constant uint NP  = {padded};\n"
        f"constant uint TG  = {tg};\n"
        f"constant uint CHUNK = {chunk};\n"
    )

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
                x = float(raw[idx]);
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
                float va = vs[lid];  uint ia = is_[lid];
                float vb = vs[lid+s]; uint ib = is_[lid+s];
                if (vb > va || (vb == va && ib < ia)) {
                    vs[lid] = vb; is_[lid] = ib;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0u) out_idx[0] = int(is_[0]);
    """

    kernel = mx.fast.metal_kernel(
        name=f"fv2_greedy_v{vocab}_n{padded}_t{tg}",
        input_names=["raw", "counts", "ctrl"],
        output_names=["out_idx"],
        source=source,
        header=header,
    )
    _GREEDY_CACHE[key] = kernel
    return kernel


# ═══════════════════════════════════════════════════════════════════════════
# Stochastic kernel: Unnormalized Mathematical Shortcut
# ═══════════════════════════════════════════════════════════════════════════

_TREE_REDUCE_SUM = """
        smem[lid] = _LOCAL_VAR_;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG >> 1; s > 0u; s >>= 1u) {
            if (lid < s) smem[lid] += smem[lid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
"""

_TREE_REDUCE_MAX = """
        smem[lid] = _LOCAL_VAR_;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG >> 1; s > 0u; s >>= 1u) {
            if (lid < s) smem[lid] = max(smem[lid], smem[lid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
"""

def _build_stochastic_kernel(
    *,
    vocab: int,
    padded: int,
    tg: int,
    do_top_p: bool,
) -> Any:
    key = (vocab, padded, tg, do_top_p)
    if key in _STOCH_CACHE:
        return _STOCH_CACHE[key]

    chunk = padded // tg
    header = (
        f"constant uint VP    = {vocab};\n"
        f"constant uint NP    = {padded};\n"
        f"constant uint TG    = {tg};\n"
        f"constant uint CHUNK = {chunk};\n"
    )

    # ctrl layout: [temp, rep, pres, freq, min_p, top_p]
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

        // ================================================================
        // Pass 1: penalties + /temp → local_max
        // ================================================================
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
    """
    source += _TREE_REDUCE_MAX.replace("_LOCAL_VAR_", "local_max")
    source += """
        float gmax = smem[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // Pass 2: exp(x - gmax) + min-p mask + local_sum
        // Shortcut: e_max is 1.0, so min_p threshold is exactly MIN_P!
        // ================================================================
        float local_sum = 0.0f;
        for (uint i = 0u; i < CHUNK; ++i) {
            float e = metal::exp(work[base + i] - gmax);
            if (MIN_P > 0.0f && e < MIN_P) {
                e = 0.0f;
            }
            work[base + i] = e;
            local_sum += e;
        }
    """
    
    if do_top_p:
        source += _TREE_REDUCE_SUM.replace("_LOCAL_VAR_", "local_sum")
        source += """
        float z_masked = smem[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // Pass 3 (optional): top-p binary search sweep
        // ================================================================
        float tp_thr = 0.0f;
        float target_mass = TOP_P * z_masked;
        float t_lo = 0.0f;
        float t_hi = 1.0f;
        
        // 5 binary-search iterations give 1/32 ≈ 0.03 precision on threshold
        for (uint iter = 0u; iter < 5u; ++iter) {
            float t_mid = (t_lo + t_hi) * 0.5f;
            float la = 0.0f;
            for (uint i = 0u; i < CHUNK; ++i) {
                float e = work[base + i];
                if (e >= t_mid) la += e;
            }
        """
        source += _TREE_REDUCE_SUM.replace("_LOCAL_VAR_", "la")
        source += """
            float mass = smem[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (mass >= target_mass) {
                t_lo = t_mid; // Target met, can try higher threshold
            } else {
                t_hi = t_mid; // Target missed, need lower threshold
            }
        }
        tp_thr = t_lo;

        // ================================================================
        // Pass 4 (optional): top-p mask & chunk sum
        // ================================================================
        float my_chunk_sum = 0.0f;
        for (uint i = 0u; i < CHUNK; ++i) {
            float e = work[base + i];
            if (e < tp_thr) {
                e = 0.0f;
                work[base + i] = e;
            }
            my_chunk_sum += e;
        }
        """
    else:
        source += """
        // No top-p sweep needed, my_chunk_sum is just the sum after min_p
        float my_chunk_sum = local_sum;
        """

    source += """
        // ================================================================
        // Pass 5: CDF prefix-sum on chunk sums
        // ================================================================
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

        // ================================================================
        // Pass 6: Sample directly on unnormalized CDF
        // ================================================================
        // uniform[0] is in [0, 1)
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


# ═══════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════

def sample_token_fused_v2(
    logits_1d: mx.array,
    token_counts_1d: mx.array,
    args: Any,
    *,
    threadgroup_size: int = TG_DEFAULT,
) -> mx.array:
    if logits_1d.ndim != 1:
        raise ValueError("sample_token_fused_v2 expects 1-D logits")

    v = int(logits_1d.shape[0])
    tg = int(threadgroup_size)
    n_pad = _next_pow2(v)
    if n_pad < tg:
        n_pad = tg

    greedy = bool(getattr(args, "fast_sample", False)) or float(args.temp) == 0.0
    no_pen = float(args.rep_pen) == 1.0 and float(args.pres_pen) == 0.0 and float(args.freq_pen) == 0.0

    if greedy and no_pen:
        # Fall back to highly-optimized native MLX grid-reduction for pure greedy
        return mx.argmax(logits_1d, axis=-1).reshape(())

    counts = token_counts_1d.astype(mx.float32)

    if greedy:
        ctrl = mx.array(
            [max(float(args.temp), 1e-8), float(args.rep_pen),
             float(args.pres_pen), float(args.freq_pen)],
            dtype=mx.float32,
        )
        k = _build_greedy_kernel(vocab=v, padded=n_pad, tg=tg)
        out = k(
            inputs=[logits_1d, counts, ctrl],
            template=[("T", logits_1d.dtype)],
            grid=(1, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(1,)],
            output_dtypes=[mx.int32],
        )[0]
        return out.reshape(())

    do_top_p = float(args.top_p) < 1.0

    ctrl = mx.array(
        [
            max(float(args.temp), 1e-8),
            float(args.rep_pen),
            float(args.pres_pen),
            float(args.freq_pen),
            float(args.min_p),
            float(args.top_p),
        ],
        dtype=mx.float32,
    )
    uniform = mx.random.uniform(shape=(1,), dtype=mx.float32)

    k = _build_stochastic_kernel(
        vocab=v, padded=n_pad, tg=tg,
        do_top_p=do_top_p,
    )
    work, out = k(
        inputs=[logits_1d, counts, ctrl, uniform],
        template=[("T", logits_1d.dtype)],
        grid=(1, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(n_pad,), (1,)],
        output_dtypes=[mx.float32, mx.int32],
    )
    return out.reshape(())
