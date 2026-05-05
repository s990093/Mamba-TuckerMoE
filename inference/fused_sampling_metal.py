#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Metal-side decode sampling (Apple MLX): penalties, temperature, min-p, top-k, top-p, stable softmax, and
inverse-CDF token draw — aligned with ``_apply_penalties_fast`` + ``_prepare_logits_for_categorical`` +
``mx.random.categorical`` semantics.

Graph flow (stochastic, no greedy):
    1) Penalties + optional ``/= temp`` into padded float32 workspace (tail padded with -inf for power-of-two sort).
    2) Stable softmax → probs (Metal).
    3) min-p: global p_max then mask logits (Metal).
    4) top-k: k sequential global argmax rounds on a scratch copy, then mask (Metal).  ``top_k`` capped at 256.
    5) top-p: MLX-only nucleus mask on the first ``vocab`` logits (argsort + cumsum); padded tail stays ``-inf``.
    6) Final softmax + ``cumsum`` + uniform inverse-CDF (MLX; scan not fused to keep code maintainable).

Greedy / ``fast_sample`` / ``temp==0``: penalties only (Metal) + global argmax (Metal).

Enable with ``args.fused_sample_metal = True`` (see ``--fused-sample-metal`` in ``stream_mlx.py`` / ``benchmark_mlx.py``).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

INF_NEG = -1e9
TOPK_CAP = 256
TG_DEFAULT = 256

_SOFTMAX_CACHE: dict[tuple[int, int], Any] = {}
_PEN_CACHE: dict[tuple[int, int, str], Any] = {}
_ARGMAX_CACHE: dict[tuple[int, int], Any] = {}
_MINP_CACHE: dict[tuple[int, int], Any] = {}
_TOPK_PICK_CACHE: dict[tuple[int, int, int], Any] = {}
_TOPK_MASK_CACHE: dict[tuple[int, int, int], Any] = {}
def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _mlx_top_p_mask_logits_1d(logits_v: mx.array, top_p: float) -> mx.array:
    """Mirror ``benchmark_mlx._prepare_logits_for_categorical`` top-p branch (logits shaped ``(V,)``)."""
    probs = mx.softmax(logits_v, axis=-1)
    sorted_indices = mx.argsort(-probs, axis=-1)
    sorted_probs = probs[sorted_indices]
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    mask = cumulative_probs > top_p
    shifted_mask = mx.concatenate([mx.array([False]), mask[:-1]])
    sorted_logits = logits_v[sorted_indices]
    neg = mx.array(-1e9, dtype=logits_v.dtype)
    sorted_logits = mx.where(shifted_mask, neg, sorted_logits)
    inverse_indices = mx.argsort(sorted_indices)
    return sorted_logits[inverse_indices]


def _build_penalties_kernel(*, vocab: int, padded: int, mode: str, tg: int) -> Any:
    key = (vocab, tg, mode)
    if key in _PEN_CACHE:
        return _PEN_CACHE[key]

    divide_temp = mode == "stochastic"

    header = (
        f"constant uint VP = {int(vocab)};\n"
        f"constant uint NP = {int(padded)};\n"
        f"constant uint TG = {int(tg)};\n"
    )

    body = """uint lid = thread_index_in_threadgroup;
        float T_use = ctrl[0];
        T_use = T_use > 1e-8f ? T_use : 1e-8f;
        float REP = ctrl[1];
        float PRES = ctrl[2];
        float FREQ = ctrl[3];

        for (uint i = lid; i < NP; i += TG) {
            if (i >= VP) {
                work[i] = -1e38f;
                continue;
            }
            float x = float(raw[i]);
            float cts = float(counts[i]);
            bool m = cts > 0.0f;
            float pen = m ? (PRES + cts * FREQ) : 0.0f;
            x = x - pen;
            if (m && REP != 1.0f) {
                x = (x > 0.0f) ? (x / REP) : (x * REP);
            }
"""

    if divide_temp:
        body += """            work[i] = x / T_use;\n"""
    else:
        body += """            work[i] = x;\n"""

    body += """        }\n"""

    kernel = mx.fast.metal_kernel(
        name=f"sample_pen_vp{vocab}_np{padded}_{mode}_{tg}",
        input_names=["raw", "counts", "ctrl"],
        output_names=["work"],
        source=body,
        header=header,
    )
    _PEN_CACHE[key] = kernel
    return kernel


def _build_softmax_kernel(*, padded: int, tg: int) -> Any:
    key = (padded, tg)
    if key in _SOFTMAX_CACHE:
        return _SOFTMAX_CACHE[key]

    header = (
        f"constant uint N = {int(padded)};\n"
        f"constant uint TG = {int(tg)};\n"
    )

    source = """
        uint lid = thread_index_in_threadgroup;
        threadgroup float reduce_smem[TG];

        float row_max = -1e38f;
        for (uint i = lid; i < N; i += TG) {
            row_max = max(row_max, float(logits[i]));
        }
        reduce_smem[lid] = row_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG >> 1; stride > 0u; stride >>= 1u) {
            if (lid < stride) {
                reduce_smem[lid] = max(reduce_smem[lid], reduce_smem[lid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float gmax = reduce_smem[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float partial = 0.0f;
        for (uint j = lid; j < N; j += TG) {
            partial += metal::exp(float(logits[j]) - gmax);
        }
        reduce_smem[lid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG >> 1; stride > 0u; stride >>= 1u) {
            if (lid < stride) {
                reduce_smem[lid] = reduce_smem[lid] + reduce_smem[lid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float Z = reduce_smem[0];
        float invZ = 1.0f / max(Z, 1e-20f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = lid; k < N; k += TG) {
            probs[k] = metal::exp(float(logits[k]) - gmax) * invZ;
        }
    """

    kernel = mx.fast.metal_kernel(
        name=f"sample_softmax_n{padded}_tg{tg}",
        input_names=["logits"],
        output_names=["probs"],
        source=source,
        header=header,
    )
    _SOFTMAX_CACHE[key] = kernel
    return kernel


def _build_argmax_kernel(*, padded: int, tg: int) -> Any:
    key = (padded, tg)
    if key in _ARGMAX_CACHE:
        return _ARGMAX_CACHE[key]

    header = (
        f"constant uint N = {int(padded)};\n"
        f"constant uint TG = {int(tg)};\n"
    )

    source = """
        uint lid = thread_index_in_threadgroup;
        threadgroup float vs[TG];
        threadgroup uint is_[TG];

        float best_v = -1e38f;
        uint best_i = 0u;
        for (uint idx = lid; idx < N; idx += TG) {
            float v = float(logits[idx]);
            if (v > best_v) { best_v = v; best_i = idx; }
        }
        vs[lid] = best_v;
        is_[lid] = best_i;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG >> 1; stride > 0u; stride >>= 1u) {
            if (lid < stride) {
                float va = vs[lid];
                uint ia = is_[lid];
                float vb = vs[lid + stride];
                uint ib = is_[lid + stride];
                if (vb > va || (vb == va && ib < ia)) {
                    vs[lid] = vb;
                    is_[lid] = ib;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0u) {
            out_idx[0] = int(is_[0]);
        }
    """

    kernel = mx.fast.metal_kernel(
        name=f"sample_argmax_n{padded}_tg{tg}",
        input_names=["logits"],
        output_names=["out_idx"],
        source=source,
        header=header,
    )
    _ARGMAX_CACHE[key] = kernel
    return kernel


def _build_min_p_kernel(*, padded: int, tg: int) -> Any:
    key = (padded, tg)
    if key in _MINP_CACHE:
        return _MINP_CACHE[key]

    header = (
        f"constant uint N = {int(padded)};\n"
        f"constant uint TG = {int(tg)};\n"
    )

    source = """
        uint lid = thread_index_in_threadgroup;
        threadgroup float reduce_smem[TG];

        float local_m = 0.0f;
        for (uint i = lid; i < N; i += TG) {
            local_m = max(local_m, float(probs[i]));
        }
        reduce_smem[lid] = local_m;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG >> 1; stride > 0u; stride >>= 1u) {
            if (lid < stride) {
                reduce_smem[lid] = max(reduce_smem[lid], reduce_smem[lid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float pmax = reduce_smem[0];
        float thr = float(ctrl[0]) * pmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = lid; j < N; j += TG) {
            float x = float(work[j]);
            float p = float(probs[j]);
            work_out[j] = (p < thr) ? -1e9f : x;
        }
    """

    kernel = mx.fast.metal_kernel(
        name=f"sample_minp_n{padded}_tg{tg}",
        input_names=["work", "probs", "ctrl"],
        output_names=["work_out"],
        source=source,
        header=header,
    )
    _MINP_CACHE[key] = kernel
    return kernel


def _build_topk_pick_kernel(*, padded: int, tg: int, k: int) -> Any:
    key = (padded, tg, k)
    if key in _TOPK_PICK_CACHE:
        return _TOPK_PICK_CACHE[key]

    header = (
        f"constant uint N = {int(padded)};\n"
        f"constant uint TG = {int(tg)};\n"
        f"constant uint KTOP = {int(k)};\n"
    )

    source = """
        uint lid = thread_index_in_threadgroup;
        threadgroup float vs[TG];
        threadgroup uint is_[TG];

        for (uint i = lid; i < N; i += TG) {
            scratch[i] = float(work[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint round = 0u; round < KTOP; ++round) {
            float best_v = -1e38f;
            uint best_i = 0u;
            for (uint idx_ = lid; idx_ < N; idx_ += TG) {
                float v = scratch[idx_];
                if (v > best_v) { best_v = v; best_i = idx_; }
            }
            vs[lid] = best_v;
            is_[lid] = best_i;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = TG >> 1; stride > 0u; stride >>= 1u) {
                if (lid < stride) {
                    float va = vs[lid];
                    uint ia = is_[lid];
                    float vb = vs[lid + stride];
                    uint ib = is_[lid + stride];
                    if (vb > va || (vb == va && ib < ia)) {
                        vs[lid] = vb;
                        is_[lid] = ib;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (lid == 0u) {
                picked[round] = int(is_[0]);
                scratch[uint(is_[0])] = -1e38f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """

    kernel = mx.fast.metal_kernel(
        name=f"sample_topk_pick_n{padded}_tg{tg}_k{k}",
        input_names=["work"],
        output_names=["scratch", "picked"],
        source=source,
        header=header,
    )
    _TOPK_PICK_CACHE[key] = kernel
    return kernel


def _build_topk_mask_kernel(*, vocab: int, padded: int, tg: int, k: int) -> Any:
    key = (vocab, padded, k)
    if key in _TOPK_MASK_CACHE:
        return _TOPK_MASK_CACHE[key]

    header = (
        f"constant uint VP = {int(vocab)};\n"
        f"constant uint N = {int(padded)};\n"
        f"constant uint TG = {int(tg)};\n"
        f"constant uint KTOP = {int(k)};\n"
    )

    source = """
        uint lid = thread_index_in_threadgroup;
        for (uint i = lid; i < VP; i += TG) {
            float x = float(work[i]);
            bool keep = false;
            for (uint j = 0u; j < KTOP; ++j) {
                if (int(i) == picked[j]) { keep = true; break; }
            }
            if (!keep) {
                x = -1e9f;
            }
            work_out[i] = x;
        }
        for (uint i = lid + VP; i < N; i += TG) {
            work_out[i] = float(work[i]);
        }
    """

    kernel = mx.fast.metal_kernel(
        name=f"sample_topk_mask_vp{vocab}_n{padded}_k{k}",
        input_names=["work", "picked"],
        output_names=["work_out"],
        source=source,
        header=header,
    )
    _TOPK_MASK_CACHE[key] = kernel
    return kernel


def _dispatch_softmax(logits: mx.array, *, padded: int, tg: int) -> mx.array:
    k = _build_softmax_kernel(padded=padded, tg=tg)
    return k(
        inputs=[logits],
        template=[("T", logits.dtype)],
        grid=(1, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(padded,)],
        output_dtypes=[mx.float32],
    )[0]


def _dispatch_argmax(logits: mx.array, *, padded: int, tg: int) -> mx.array:
    k = _build_argmax_kernel(padded=padded, tg=tg)
    return k(
        inputs=[logits],
        template=[("T", logits.dtype)],
        grid=(1, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.int32],
    )[0]


def inverted_cdf_token_from_probs(probs_1d: mx.array, u: mx.array | None = None) -> mx.array:
    if u is None:
        u = mx.random.uniform(shape=(1,), dtype=mx.float32)
    else:
        u = u.astype(mx.float32)
    u = mx.clip(u, 1e-12, 1.0 - 1e-12)
    cdf = mx.cumsum(probs_1d.astype(mx.float32), axis=-1)
    idx = mx.sum(cdf < u[0], axis=-1).astype(mx.int32)
    last = mx.array(probs_1d.shape[0] - 1, dtype=mx.int32)
    return mx.minimum(idx, last)


def sample_token_metal_full(
    logits_1d: mx.array,
    token_counts_1d: mx.array,
    args: Any,
    *,
    threadgroup_size: int = TG_DEFAULT,
) -> mx.array:
    """
    One decode step: raw model logits row + per-token counts → next token (greedy or sampled).
    Expects ``logits_1d`` shape ``(V,)``, ``token_counts_1d`` ``(V,)`` float32-compatible.
    """
    if logits_1d.ndim != 1:
        raise ValueError("sample_token_metal_full expects 1-D logits")
    if token_counts_1d.shape != logits_1d.shape:
        raise ValueError("token_counts must match logits shape")

    v = int(logits_1d.shape[0])
    tg = int(threadgroup_size)
    n_pad = _next_pow2(v)

    raw = logits_1d
    counts = token_counts_1d.astype(mx.float32)
    ctrl = mx.array(
        [max(float(args.temp), 1e-8), float(args.rep_pen), float(args.pres_pen), float(args.freq_pen)],
        dtype=mx.float32,
    )

    greedy = bool(getattr(args, "fast_sample", False)) or float(args.temp) == 0.0
    mode = "greedy" if greedy else "stochastic"
    k_pen = _build_penalties_kernel(vocab=v, padded=n_pad, mode=mode, tg=tg)
    work = k_pen(
        inputs=[raw, counts, ctrl],
        template=[("T", raw.dtype)],
        grid=(1, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(n_pad,)],
        output_dtypes=[mx.float32],
    )[0]

    if greedy:
        tok = _dispatch_argmax(work, padded=n_pad, tg=tg)
        return tok.reshape(())

    if int(args.top_k) > TOPK_CAP:
        raise ValueError(f"Metal top-k path supports top_k <= {TOPK_CAP} (got {args.top_k})")

    w = work
    if float(args.min_p) > 0.0:
        probs_m = _dispatch_softmax(w, padded=n_pad, tg=tg)
        km = _build_min_p_kernel(padded=n_pad, tg=tg)
        mpctrl = mx.array([float(args.min_p)], dtype=mx.float32)
        w = km(
            inputs=[w, probs_m, mpctrl],
            template=[("T", w.dtype)],
            grid=(1, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(n_pad,)],
            output_dtypes=[mx.float32],
        )[0]

    k_top = int(args.top_k)
    if k_top > 0:
        k_pick = _build_topk_pick_kernel(padded=n_pad, tg=tg, k=k_top)
        _sc, picked = k_pick(
            inputs=[w],
            template=[("T", w.dtype)],
            grid=(1, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(n_pad,), (k_top,)],
            output_dtypes=[mx.float32, mx.int32],
        )
        kmask = _build_topk_mask_kernel(vocab=v, padded=n_pad, tg=tg, k=k_top)
        w = kmask(
            inputs=[w, picked],
            template=[("T", w.dtype)],
            grid=(1, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(n_pad,)],
            output_dtypes=[mx.float32],
        )[0]

    if float(args.top_p) < 1.0:
        w_vp = _mlx_top_p_mask_logits_1d(w[:v], float(args.top_p))
        tail = mx.full((n_pad - v,), -1e38, dtype=mx.float32)
        w = mx.concatenate([w_vp.astype(mx.float32), tail], axis=-1)

    probs_f = _dispatch_softmax(w, padded=n_pad, tg=tg)
    probs_v = probs_f[:v]
    return inverted_cdf_token_from_probs(probs_v)
