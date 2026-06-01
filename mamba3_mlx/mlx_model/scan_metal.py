"""Stable pure-MLX SSM chunk scan for Mamba3Block.

This module exposes a single :func:`chunk_scan` public API that mirrors the
training-time ``chunk_parallel_scan`` in
``pre-train/sft_cot_bundle/scripts/model.py`` step-for-step:

    la_cum  = cumsum(la_c, axis=2)                     # float32
    log_M   = la_cum[i] - la_cum[j]                    # float32
    M       = exp(log_M).astype(u.dtype)               # bf16
    h_intra = einsum("bcijh,bcjhnp->bcihnp", M, u_c)   # bf16
    y_diag  = einsum("bclhnp,bclhnr->bclhpr", h_intra, C_c)
    inter-chunk carry: python for-loop over nc
    y_off   = einsum("bchnp,bclhnr->bclhpr", h_inter, c_dec)
    y       = (y_diag + y_off).reshape(B, L, H, P, R)

This is the formulation the model was trained with (modulo the training
Triton kernel that fuses the recurrence into one pass — the math is
identical). It supports bf16 throughout and is what `run.py` expects.

A previous version of this file shipped a custom Metal kernel (intra-chunk
sequential + inter-chunk single-dispatch) for speed. That path produced
subtly different outputs vs. the training reference and was the source of
the decode instability the user is fixing here. The kernel sources are
removed; only the reference math remains.

h_init support (multi-turn / non-zero initial state) is folded directly
into the python inter-chunk loop, which makes the math obviously identical
to training.

Usage:
    from .scan_metal import chunk_scan
    y, h_final = chunk_scan(u_ssm, la_full, C_rot, chunk_size, h_init=h_init)
"""

from __future__ import annotations

import mlx.core as mx


# ── Pure MLX reference path ───────────────────────────────────────────────────

def _chunk_scan_mlx(u, la, C_rot, tri_mask, chunk_size, h_init=None):
    """Reference MLX implementation — O(Lc²) intra-chunk, Python loop cross-chunk.

    u:        (B, L, H, N, P)
    la:       (B, L, H)  float32 — dt_b * A_b
    C_rot:    (B, L, H, N, R)
    tri_mask: (Lc, Lc) bool — cached lower-triangular mask
    h_init:   (B, H, N, P) or None
    Returns:  (y: (B, L, H, P, R), h_final: (B, H, N, P))
    """
    B, L, H, N, P = u.shape
    R = C_rot.shape[-1]
    Lc = chunk_size
    L_orig = L
    pad = (Lc - L % Lc) % Lc
    if pad:
        u = mx.pad(u, [(0, 0), (0, pad), (0, 0), (0, 0), (0, 0)])
        la = mx.pad(la, [(0, 0), (0, pad), (0, 0)])
        C_rot = mx.pad(C_rot, [(0, 0), (0, pad), (0, 0), (0, 0), (0, 0)])
        L = L + pad
    nc = L // Lc

    u_c = u.reshape(B, nc, Lc, H, N, P)
    la_c = la.reshape(B, nc, Lc, H)
    C_c = C_rot.reshape(B, nc, Lc, H, N, R)

    la_cum = mx.cumsum(la_c, axis=2)
    log_M = la_cum[:, :, :, None, :] - la_cum[:, :, None, :, :]
    log_M = mx.where(tri_mask[None, None, :, :, None], log_M,
                     mx.array(-1e9, dtype=log_M.dtype))
    M = mx.exp(log_M).astype(u.dtype)
    h_intra = mx.einsum("bcijh,bcjhnp->bcihnp", M, u_c)

    y_diag = mx.einsum("bclhnp,bclhnr->bclhpr", h_intra, C_c)

    decay = mx.exp(mx.sum(la_c, axis=2))
    h_prev = mx.zeros((B, H, N, P), dtype=u.dtype) if h_init is None else h_init.astype(u.dtype)
    h_inter_list = []
    for c in range(nc):
        h_inter_list.append(h_prev)
        h_prev = h_prev * decay[:, c].reshape(B, H, 1, 1).astype(u.dtype) + h_intra[:, c, -1]
    h_inter = mx.stack(h_inter_list, axis=1)

    cdec_scale = mx.exp(la_cum).astype(u.dtype)
    c_dec = C_c * cdec_scale[..., None, None]
    y_off = mx.einsum("bchnp,bclhnr->bclhpr", h_inter, c_dec)

    y = (y_diag + y_off).reshape(B, L, H, P, R)
    if L_orig != L:
        y = y[:, :L_orig]
    return y, h_prev


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_scan(u, la, C_rot, chunk_size, h_init=None, _tri_mask=None):
    """Chunk-parallel SSM scan — pure-MLX, training-equivalent.

    Args:
        u:          (B, L, H, N, P) — blended input signal
        la:         (B, L, H) float32 — precomputed dt_b * A_b
        C_rot:      (B, L, H, N, R) — RoPE-rotated C projection
        chunk_size: intra-chunk length Lc (must evenly divide L after padding)
        h_init:     (B, H, N, P) or None — initial SSM state
        _tri_mask:  (Lc, Lc) bool — cached triangular mask; recomputed if None

    Returns:
        y:       (B, L, H, P, R)
        h_final: (B, H, N, P)
    """
    Lc = chunk_size
    if _tri_mask is None:
        _tri_mask = mx.tri(Lc, dtype=mx.bool_)
    return _chunk_scan_mlx(u, la, C_rot, _tri_mask, chunk_size, h_init=h_init)


# ── Correctness test (bf16 path vs float32 ground truth) ──────────────────────

def _run_correctness_test(B=1, L=128, H=4, N=8, P=4, R=4, chunk_size=32,
                           dtype=mx.bfloat16, seed=42, use_h_init=True):
    """Sanity-check the bf16 chunk_scan against a float32 ground-truth pass.

    Run with: python -c "from mamba3_mlx.mlx_model.scan_metal import _run_correctness_test; _run_correctness_test()"
    """
    key = mx.random.key(seed)

    def _randn(*shape):
        nonlocal key
        key, sub = mx.random.split(key)
        return mx.random.normal(shape=shape, key=sub).astype(dtype)

    u = _randn(B, L, H, N, P)
    C_rot = _randn(B, L, H, N, R)
    h_init = _randn(B, H, N, P) if use_h_init else None
    la = (mx.random.normal(shape=(B, L, H), key=key) * 0.1 - 0.5).astype(mx.float32)
    mx.eval(u, C_rot, la)
    if h_init is not None:
        mx.eval(h_init)

    tri_mask = mx.tri(chunk_size, dtype=mx.bool_)

    # bf16 path (the production scan)
    y_bf, h_bf = chunk_scan(u, la, C_rot, chunk_size, h_init=h_init, _tri_mask=tri_mask)
    mx.eval(y_bf, h_bf)

    # float32 ground truth
    tri_mask_f32 = mx.tri(chunk_size, dtype=mx.bool_)
    y_gt, h_gt = _chunk_scan_mlx(
        u.astype(mx.float32), la, C_rot.astype(mx.float32), tri_mask_f32,
        chunk_size, h_init=(h_init.astype(mx.float32) if h_init is not None else None))
    mx.eval(y_gt, h_gt)

    err_y = float(mx.max(mx.abs(y_bf.astype(mx.float32) - y_gt.astype(mx.float32))).item())
    err_h = float(mx.max(mx.abs(h_bf.astype(mx.float32) - h_gt.astype(mx.float32))).item())

    print(f"[scan_metal correctness] B={B} L={L} H={H} N={N} P={P} R={R} Lc={chunk_size}  h_init={use_h_init}")
    print(f"  max |y_bf - y_f32| = {err_y:.2e}")
    print(f"  max |h_bf - h_f32| = {err_h:.2e}")
    ok = err_y < 1.0  # bf16 carries ~3 decimal digits; this is a smoke check
    print(f"  {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok


if __name__ == "__main__":
    _run_correctness_test(use_h_init=True)
    _run_correctness_test(use_h_init=False)
