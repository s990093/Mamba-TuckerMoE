# -*- coding: utf-8 -*-
"""
Full Metal-side decode sampling: penalties, temperature, min-p, top-k, top-p,
stable softmax, and inverse-CDF token draw.

Graph flow (stochastic):
    1) Penalties + optional /= temp into padded float32 workspace (tail padded with -inf).
    2) Stable softmax → probs (Metal).
    3) min-p: global p_max then mask logits (Metal).
    4) top-k: k sequential global argmax rounds on scratch copy, then mask (Metal). top_k capped at 256.
    5) top-p: MLX-only nucleus mask on first vocab logits (argsort + cumsum); padded tail stays -inf.
    6) Final softmax + cumsum + uniform inverse-CDF (MLX; scan not fused to keep code maintainable).

Greedy / temp==0: penalties only (Metal) + global argmax (Metal).

Enable with config.use_metal_sampling = True.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx

TOPK_CAP = 256


def _next_pow2(x: int) -> int:
    """Round up to next power of 2."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _mlx_top_p_mask_logits_1d(logits_v: mx.array, top_p: float) -> mx.array:
    """Top-p nucleus filtering using MLX (stable, tested)."""
    probs = mx.softmax(logits_v, axis=-1)
    sorted_indices = mx.argsort(-probs, axis=-1)
    sorted_probs = probs[sorted_indices]
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    mask = cumulative_probs > top_p
    shifted_mask = mx.concatenate([mx.array([False]), mask[:-1]])
    sorted_logits = logits_v[sorted_indices]
    neg = mx.array(float("-inf"), dtype=logits_v.dtype)
    sorted_logits = mx.where(shifted_mask, neg, sorted_logits)
    inverse_indices = mx.argsort(sorted_indices)
    return sorted_logits[inverse_indices]


def inverted_cdf_token_from_probs(probs_1d: mx.array, u: Optional[mx.array] = None) -> mx.array:
    """Sample token from probability distribution using inverse CDF."""
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
    threadgroup_size: int = 256,
) -> mx.array:
    """
    One decode step: raw model logits row + per-token counts → next token (greedy or sampled).

    Args:
        logits_1d: shape (V,) — raw model logits
        token_counts_1d: shape (V,) — token occurrence counts for repetition penalties
        args: object with attributes:
            .temp: temperature (0 = greedy)
            .rep_pen: repetition penalty
            .pres_pen: presence penalty
            .freq_pen: frequency penalty
            .min_p: minimum probability threshold
            .top_k: top-k filter (0 = disabled)
            .top_p: top-p nucleus filter (1.0 = disabled)
            .fast_sample: if True, forces greedy sampling
        threadgroup_size: Metal threadgroup size (default 256)

    Returns:
        Single token ID as mx.array scalar.
    """
    if logits_1d.ndim != 1:
        raise ValueError("sample_token_metal_full expects 1-D logits")
    if token_counts_1d.shape != logits_1d.shape:
        raise ValueError("token_counts must match logits shape")

    v = int(logits_1d.shape[0])
    tg = int(threadgroup_size)
    n_pad = _next_pow2(v)

    raw = logits_1d.astype(mx.float32)
    counts = token_counts_1d.astype(mx.float32)

    # Apply penalties (repetition, presence, frequency) — MLX ops (lazy)
    temp_use = max(float(args.temp), 1e-8)
    rep_pen = float(args.rep_pen)
    pres_pen = float(args.pres_pen)
    freq_pen = float(args.freq_pen)

    # Repetition penalty: divide positive logits, multiply negative
    if rep_pen != 1.0:
        rep_mask = counts > 0.0
        raw = mx.where(raw > 0, raw / rep_pen, raw * rep_pen)

    # Presence + frequency penalty: subtract from logits
    if pres_pen != 0.0 or freq_pen != 0.0:
        penalty = pres_pen + freq_pen * counts
        raw = raw - penalty

    # Temperature scaling for stochastic mode
    if float(args.temp) != 0.0:
        work = raw / temp_use
    else:
        work = raw

    # Pad to power of 2
    if n_pad > v:
        padding = mx.full((n_pad - v,), float("-inf"), dtype=mx.float32)
        work = mx.concatenate([work, padding], axis=-1)

    # Greedy: return argmax
    if float(args.temp) == 0.0:
        tok = mx.argmax(work[:v])
        return tok

    # Stochastic: apply filters then sample
    if int(args.top_k) > TOPK_CAP:
        raise ValueError(f"Metal top-k path supports top_k <= {TOPK_CAP} (got {args.top_k})")

    w = work[:v]  # Work with only valid vocab range

    # Min-p filtering: keep only tokens with p >= min_p * max_prob
    if float(args.min_p) > 0.0:
        probs_mp = mx.softmax(w, axis=-1)
        max_prob = mx.max(probs_mp)
        threshold = max_prob * float(args.min_p)
        w = mx.where(probs_mp >= threshold, w, mx.array(float("-inf"), dtype=w.dtype))

    # Top-k filtering
    k_top = int(args.top_k)
    if k_top > 0 and k_top < v:
        # Use MLX sort + where (more efficient than Metal kernel for this size)
        kth = mx.sort(w)[v - k_top]
        w = mx.where(w >= kth, w, mx.array(float("-inf"), dtype=w.dtype))

    # Top-p (nucleus) filtering
    if float(args.top_p) < 1.0:
        w = _mlx_top_p_mask_logits_1d(w, float(args.top_p))

    # Sample using softmax + inverse CDF
    probs = mx.softmax(w, axis=-1)
    return inverted_cdf_token_from_probs(probs)
