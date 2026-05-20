# -*- coding: utf-8 -*-
"""
Metal-optimized SSM scan (chunk-wise parallel prefix).

Replaces O(Lc²) matrix matmul with O(Lc) sequential scan + O(nc) inter-chunk carry.
All chunks' intra-scans run in parallel on Metal; inter-chunk scan is lightweight.

Exports:
  - chunk_parallel_scan_mlx()       : Full scan (prefill)
  - chunk_parallel_scan_with_init() : Scan with initial state (speculative decode)
  - LayerScale                      : Layer scaling module (fixed)

Compatible with mamba_block.py's shape expectations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# Phase 2B: Metal kernel optimization
try:
    from .ultimate_kernel_lib import ssm_sequential_scan_metal
    _METAL_KERNEL_AVAILABLE = True
except ImportError:
    _METAL_KERNEL_AVAILABLE = False


class LayerScale(nn.Module):
    """Layer scaling: y = x * gamma. Fixed typo from original."""
    def __init__(self, dim: int, init_value: float = 1e-2):
        super().__init__()
        self.gamma = mx.full((dim,), init_value)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


def _intra_chunk_scan_baseline(la_c: mx.array, u_c: mx.array, lc: int, b: int, h: int, n: int, p: int) -> mx.array:
    """
    Fallback intra-chunk scan using O(Lc²) matrix method.

    Used when Metal kernel is unavailable or fails.
    Computes h[t] = sum_{k<=t} exp(cumsum[t]-cumsum[k]) * u[k] per chunk.

    Args:
        la_c: (B, nc, Lc, H) — log-alpha
        u_c: (B, nc, Lc, H, N, P) — input signal
        lc: Chunk size
        b, h, n, p: Dimensions

    Returns:
        h_intra: (B, nc, Lc, H, N, P) — hidden state at each step
    """
    nc = la_c.shape[1]
    h_intra_list = []

    for c in range(nc):
        # Extract chunk c
        la_chunk_c = la_c[:, c:c+1, :, :]  # (B, 1, Lc, H)
        u_chunk_c = u_c[:, c:c+1, :, :, :, :]  # (B, 1, Lc, H, N, P)

        # Apply baseline matrix method for this chunk
        log_cum = mx.cumsum(la_chunk_c, axis=2)  # (B, 1, Lc, H)

        # Lower-triangular transition matrix
        log_cum_t = log_cum[:, :, :, None, :]  # (B, 1, Lc, 1, H)
        log_cum_k = log_cum[:, :, None, :, :]  # (B, 1, 1, Lc, H)
        log_M = log_cum_t - log_cum_k  # (B, 1, Lc, Lc, H)

        # Apply lower-triangular mask
        tril = mx.tril(mx.ones((lc, lc), dtype=mx.bool_))
        neg_inf = mx.full((1,), -1e30, dtype=log_M.dtype)
        log_M = mx.where(tril[None, None, :, :, None], log_M, neg_inf)
        M = mx.exp(log_M)

        # Matmul: (B, 1, H, Lc, Lc) @ (B, 1, H, Lc, NP)
        M_transposed = M.transpose(0, 1, 4, 2, 3)  # (B, 1, H, Lc, Lc)
        u_flat_c = u_chunk_c.transpose(0, 1, 3, 2, 4, 5).reshape(b, 1, h, lc, n * p)
        h_intra_flat_c = M_transposed @ u_flat_c
        h_intra_c = h_intra_flat_c.reshape(b, 1, h, lc, n, p).transpose(0, 1, 3, 2, 4, 5)

        h_intra_list.append(h_intra_c[:, 0])  # Remove singleton chunk dim

    return mx.stack(h_intra_list, axis=1)  # (B, nc, Lc, H, N, P)


def chunk_parallel_scan_mlx(
    u: mx.array,
    dt_b: mx.array,
    a_b: mx.array,
    c_rotated: mx.array,
    chunk_size: int,
    return_h_full_zero: bool = False,
) -> Tuple[mx.array, mx.array] | Tuple[mx.array, mx.array, mx.array]:
    """
    Chunk-wise parallel SSM scan (Metal optimized).

    Decomposes full scan into:
      1. Intra-chunk: O(Lc) per chunk (sequential scan on Metal)
      2. Inter-chunk: O(nc) lightweight carry propagation

    Args:
        u: (B, L, H, N, P) — SSM input signal
        dt_b: (B, L, H) — discretization step (log-space)
        a_b: (B, L, H) — state matrix (log-space, negative)
        c_rotated: (B, L, H, N, R) — output projection (RoPE-rotated)
        chunk_size: int — chunk length (e.g., 64)
        return_h_full_zero: bool — if True, return h at each position (zero-init)

    Returns:
        y: (B, L, H, P, R) — output
        h_prev: (B, H, N, P) — final hidden state
        [h_full_zero: (B, L, H, N, P) if return_h_full_zero]

    Note:
        Matches chunk_parallel_scan() from mamba_block.py in math & shapes.
    """
    b, l, h, n, p = u.shape
    r = c_rotated.shape[-1]
    l_orig = l

    # Pad to multiple of chunk_size
    pad = 0
    if l % chunk_size != 0:
        pad = chunk_size - (l % chunk_size)
        u = mx.pad(u, [(0, 0), (0, pad), (0, 0), (0, 0), (0, 0)])
        dt_b = mx.pad(dt_b, [(0, 0), (0, pad), (0, 0)])
        a_b = mx.pad(a_b, [(0, 0), (0, pad), (0, 0)])
        c_rotated = mx.pad(c_rotated, [(0, 0), (0, pad), (0, 0), (0, 0), (0, 0)])
        l = u.shape[1]

    nc = l // chunk_size
    log_alpha = dt_b * a_b  # (B, L, H)
    u_c = u.reshape(b, nc, chunk_size, h, n, p)
    la_c = log_alpha.reshape(b, nc, chunk_size, h)
    c_c = c_rotated.reshape(b, nc, chunk_size, c_rotated.shape[2], n, r)

    lc = la_c.shape[2]
    d_inner = n * p

    # ── Intra-chunk scan (O(Lc) Metal kernel or fallback) ──────────────────────────────────
    # Phase 2B: Optimize from O(Lc²) matmul to O(Lc) sequential scan via Metal kernel
    # For each chunk independently, compute h[t] = exp(la[t]) * h[t-1] + u[t]

    if _METAL_KERNEL_AVAILABLE:
        try:
            # Try Metal kernel first (O(Lc) per chunk, ~3-5× speedup)
            h_intra = ssm_sequential_scan_metal(la_c, u_c, dtype="bf16")
        except Exception as e:
            print(f"Warning: Metal kernel failed ({e}), falling back to baseline matrix method...")
            h_intra = _intra_chunk_scan_baseline(la_c, u_c, lc, b, h, n, p)
    else:
        # Fallback: baseline matrix method (O(Lc²))
        h_intra = _intra_chunk_scan_baseline(la_c, u_c, lc, b, h, n, p)

    # ── Diagonal contribution: y[c,t,h,p,r] = sum_n h[c,t,h,n,p] * C[c,t,h,n,r] ─
    y_diag = mx.einsum("bclhnp,bclhnr->bclhpr", h_intra, c_c)

    # ── Inter-chunk scan (lightweight carry) ────────────────────────────────────
    # Compute exclusive prefix: h_inter[c] = h_final[c-1] * exp(sum(la[c-1]))
    decay = mx.exp(mx.clip(la_c.sum(axis=2), -88.0, 88.0))  # (B, nc, H)

    h_inter_list = []
    h_prev = mx.zeros((b, h, n, p), dtype=u.dtype)

    for c in range(nc):
        # Store exclusive prefix for this chunk
        h_inter_list.append(h_prev)

        # Update h_prev for next chunk: h_prev = h_prev * decay + h_intra[c, -1]
        h_final_c = h_intra[:, c, -1]  # (B, H, N, P)
        h_prev = h_prev * decay[:, c, :, None, None] + h_final_c

    h_inter = mx.stack(h_inter_list, axis=1)  # (B, nc, H, N, P)

    # ── Off-diagonal contribution from h_inter ────────────────────────────────
    # c_dec[c,t,h,n,r] = C[c,t,h,n,r] * exp(cumsum_intra[c,t,h])
    cum_la_intra = mx.cumsum(la_c, axis=2)  # (B, nc, Lc, H) — intra-chunk cumsum
    cum_la_intra = mx.clip(cum_la_intra, -88.0, 88.0)
    c_dec = c_c * mx.exp(cum_la_intra)[:, :, :, :, None, None]

    y_off = mx.einsum("bchnp,bclhnr->bclhpr", h_inter, c_dec)

    y = (y_diag + y_off).reshape(b, l, h, p, r)

    # Trim padding
    if l_orig < l:
        y = y[:, :l_orig]

    if return_h_full_zero:
        # Full zero-init hidden state at each position:
        # h_full[c,t] = h_intra[c,t] + h_inter[c] * exp(cumsum_intra[c,t])
        h_inter_contrib = (
            h_inter[:, :, None, :, :, :].astype(mx.float32)
            * mx.exp(cum_la_intra[:, :, :, :, None, None])
        )
        h_full_zero = h_intra.astype(mx.float32) + h_inter_contrib
        h_full_zero_flat = h_full_zero.reshape(b, l, h, n, p)
        if l_orig < l:
            h_full_zero_flat = h_full_zero_flat[:, :l_orig]
        return y.astype(u.dtype), h_prev.astype(u.dtype), h_full_zero_flat.astype(u.dtype)

    return y.astype(u.dtype), h_prev.astype(u.dtype)


def chunk_parallel_scan_with_init(
    u: mx.array,
    dt_b: mx.array,
    a_b: mx.array,
    c_rotated: mx.array,
    chunk_size: int,
    h_init: mx.array,
    return_h_all: bool = False,
) -> Tuple[mx.array, mx.array] | Tuple[mx.array, mx.array, mx.array]:
    """
    Incremental parallel scan starting from non-zero initial SSM state.

    Used for Jacobi K-token verify step when existing cache state must be
    continued across K new draft tokens. Avoids K separate MLX dispatches.

    Math:
        h_t = h_{t-1} * alpha_t + u_t,   h_{-1} = h_init

    Decompose:
        h_t = h_t^{(zero)}  +  h_init * alpha_cum_t

    Both terms computed in O(1) parallel depth.

    Args:
        u, dt_b, a_b, c_rotated: same as chunk_parallel_scan_mlx
        h_init: (B, H, N, P) — cached SSM state from previous position
        return_h_all: bool — if True, return h at each position

    Returns:
        y: (B, L, H, P, R) — output
        h_final: (B, H, N, P) — final SSM state after all L tokens
        [h_all: (B, L, H, N, P) if return_h_all]
    """
    # Zero-init parallel scan
    if return_h_all:
        y_zero, h_final_zero, h_full_zero = chunk_parallel_scan_mlx(
            u, dt_b, a_b, c_rotated, chunk_size, return_h_full_zero=True
        )
    else:
        y_zero, h_final_zero = chunk_parallel_scan_mlx(
            u, dt_b, a_b, c_rotated, chunk_size, return_h_full_zero=False
        )

    # ── Initial-state correction (all in float32 to avoid bfloat16 cumsum drift) ──
    log_alpha_all = mx.clip(
        dt_b.astype(mx.float32) * a_b.astype(mx.float32), -88.0, 88.0
    )
    log_alpha_cum = mx.cumsum(log_alpha_all, axis=1)
    log_alpha_cum = mx.clip(log_alpha_cum, -88.0, 88.0)
    alpha_cum = mx.exp(log_alpha_cum)

    # h_t^{(init)} = h_init * alpha_cum_t
    h_init_t = h_init[:, None].astype(mx.float32) * alpha_cum[:, :, :, None, None]

    # y_t^{(init)} = einsum(h_init_t, c_rotated_t)
    y_init = mx.einsum("blhnp,blhnr->blhpr", h_init_t, c_rotated.astype(mx.float32))

    y = y_zero.astype(mx.float32) + y_init

    # Final state correction
    log_alpha_total = log_alpha_cum[:, -1]
    alpha_total = mx.exp(mx.clip(log_alpha_total, -88.0, 88.0))
    h_final = h_final_zero.astype(mx.float32) + h_init.astype(mx.float32) * alpha_total[:, :, None, None]

    if return_h_all:
        h_all_corrected = h_full_zero.astype(mx.float32) + h_init_t
        return y.astype(u.dtype), h_final.astype(u.dtype), h_all_corrected.astype(u.dtype)

    return y.astype(u.dtype), h_final.astype(u.dtype)
