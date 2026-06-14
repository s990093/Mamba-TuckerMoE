"""Stable pure-MLX SSM chunk scan for Mamba3Block.

This module exposes a single :func:`chunk_scan` public API that mirrors the
training-time ``chunk_parallel_scan`` in
``pre-train/sft_cot_bundle/scripts/model.py`` step-for-step:

    la_cum  = cumsum(la_c, axis=2)                     # float32
    log_M   = la_cum[i] - la_cum[j]                    # float32
    M       = exp(log_M).astype(u.dtype)               # bf16
    h_intra = einsum("bcijh,bcjhnp->bcihnp", M, u_c)   # bf16
    y_diag  = einsum("bclhnp,bclhnr->bclhpr", h_intra, C_c)
    inter-chunk carry: python for-loop over nc (Python overhead is negligible)
    y_off   = einsum("bchnp,bclhnr->bclhpr", h_inter, c_dec)
    y       = (y_diag + y_off).reshape(B, L, H, P, R)

Note on chunk_scan einsums
--------------------------
The heavy einsums (h_intra, y_diag, y_off) use MLX's strided batched GEMM
internally and must NOT be replaced with explicit transpose+matmul sequences:
non-contiguous transpose operations on 200 MB tensors cost more than they save
and make chunk_scan ~1.7× slower on M2 Pro.  The einsums remain.

Inter-chunk carry dtype: h_prev is kept in u.dtype (bf16) to match the
training-time numerical path. This preserves the stochastic identity
distribution that the model learned at train time.

Usage:
    from .scan_metal import chunk_scan
    y, h_final = chunk_scan(u_ssm, la_full, C_rot, chunk_size, h_init=h_init)
"""

from __future__ import annotations

import mlx.core as mx


def _chunk_scan_mlx(u, la, C_rot, tri_mask, chunk_size, h_init=None):
    """Chunk-parallel SSM scan — pure-MLX, training-equivalent.

    u:        (B, L, H, N, P)
    la:       (B, L, H)  float32
    C_rot:    (B, L, H, N, R)
    tri_mask: (Lc, Lc) bool
    h_init:   (B, H, N, P) or None
    Returns:  (y: (B, L, H, P, R), h_final: (B, H, N, P))
    """
    B, L, H, N, P = u.shape
    R = C_rot.shape[-1]
    Lc = chunk_size
    L_orig = L
    pad = (Lc - L % Lc) % Lc
    if pad:
        u     = mx.pad(u,     [(0,0),(0,pad),(0,0),(0,0),(0,0)])
        la    = mx.pad(la,    [(0,0),(0,pad),(0,0)])
        C_rot = mx.pad(C_rot, [(0,0),(0,pad),(0,0),(0,0),(0,0)])
        L = L + pad
    nc = L // Lc

    u_c  = u.reshape(B, nc, Lc, H, N, P)
    la_c = la.reshape(B, nc, Lc, H)
    C_c  = C_rot.reshape(B, nc, Lc, H, N, R)

    # ── Intra-chunk causal scan ───────────────────────────────────────────────
    la_cum = mx.cumsum(la_c, axis=2)
    log_M  = la_cum[:, :, :, None, :] - la_cum[:, :, None, :, :]
    log_M  = mx.where(tri_mask[None, None, :, :, None], log_M,
                      mx.array(-1e9, dtype=log_M.dtype))
    M = mx.exp(log_M).astype(u.dtype)
    h_intra = mx.einsum("bcijh,bcjhnp->bcihnp", M, u_c)         # (B, nc, Lc, H, N, P)
    y_diag  = mx.einsum("bclhnp,bclhnr->bclhpr", h_intra, C_c)  # (B, nc, Lc, H, P, R)

    # ── Inter-chunk carry ─────────────────────────────────────────────────────
    # Simple linear recurrence: h[c+1] = decay[c]*h[c] + delta[c].
    # Python loop is cheap (GPU work per iter is <0.01 ms even at nc=512).
    # h_prev kept in u.dtype (bf16) to match training-time numerical path exactly.
    decay = mx.exp(mx.sum(la_c, axis=2))                         # (B, nc, H)
    h_prev = (mx.zeros((B, H, N, P), dtype=u.dtype)
              if h_init is None else h_init.astype(u.dtype))
    h_inter_list = []
    for c in range(nc):
        h_inter_list.append(h_prev)
        h_prev = (h_prev * decay[:, c].reshape(B, H, 1, 1).astype(u.dtype)
                  + h_intra[:, c, -1])
    h_inter = mx.stack(h_inter_list, axis=1)                     # (B, nc, H, N, P)

    # ── Off-diagonal (h_inter contribution) ──────────────────────────────────
    cdec_scale = mx.exp(la_cum).astype(u.dtype)
    c_dec = C_c * cdec_scale[..., None, None]
    y_off = mx.einsum("bchnp,bclhnr->bclhpr", h_inter, c_dec)   # (B, nc, Lc, H, P, R)

    y = (y_diag + y_off).reshape(B, L, H, P, R)
    if L_orig != L:
        y = y[:, :L_orig]
    return y, h_prev


def chunk_scan(u, la, C_rot, chunk_size, h_init=None, _tri_mask=None):
    """Chunk-parallel SSM scan — pure-MLX, training-equivalent.

    Args:
        u:          (B, L, H, N, P)
        la:         (B, L, H) float32
        C_rot:      (B, L, H, N, R)
        chunk_size: intra-chunk length Lc
        h_init:     (B, H, N, P) or None
        _tri_mask:  (Lc, Lc) bool — recomputed if None
    Returns:
        y:       (B, L, H, P, R)
        h_final: (B, H, N, P)
    """
    if _tri_mask is None:
        _tri_mask = mx.tri(chunk_size, dtype=mx.bool_)
    return _chunk_scan_mlx(u, la, C_rot, _tri_mask, chunk_size, h_init=h_init)


# ── Metal-kernel-accelerated chunk_scan ──────────────────────────────────────
#
# Strategy: replace the two heavy einsums (h_intra, y_diag) with
#   Metal transpose → contiguous layout → standard batched matmul.
#
# The H dimension sits at the *last* position in both M (B,nc,Lc,Lc,H) and u_c
# (B,nc,Lc,H,N,P), so MLX's einsum dispatches a *strided* batched GEMM over
# nc×B batches with H-interleaved strides.  Moving H to the batch axis first
# (the "head-first" layout, bch = b*nc*H + c*H + h) produces fully contiguous
# (BncH, Lc, Lc) and (BncH, Lc, P*N) tensors on which matmul is ~2× faster.
# The Metal kernels perform this in-place transpose on the GPU (one pass over
# the data) so no Python-side non-contiguous copies are created.

_metal_cache: dict = {}


def _get_kernel(key: str, input_names, output_names, src: str):
    if key not in _metal_cache:
        _metal_cache[key] = mx.fast.metal_kernel(
            name=key,
            input_names=input_names,
            output_names=output_names,
            source=src,
            ensure_row_contiguous=True,
        )
    return _metal_cache[key]


def _transpose_M_hf(M, B: int, nc: int, Lc: int, H: int):
    """M (B,nc,Lc,Lc,H) float32 → M_hf (B*nc*H, Lc, Lc) float32."""
    total = B * nc * Lc * Lc * H
    key = f"scan_M_{B}_{nc}_{Lc}_{H}"
    src = f"""
        uint idx = thread_position_in_grid.x;
        if (idx >= {total}u) return;
        uint h = idx % {H}u;
        uint j = (idx / {H}u) % {Lc}u;
        uint i = (idx / {H * Lc}u) % {Lc}u;
        uint c = (idx / {H * Lc * Lc}u) % {nc}u;
        uint b = idx / {H * Lc * Lc * nc}u;
        uint bch = b * {nc * H}u + c * {H}u + h;
        M_hf[bch * {Lc * Lc}u + i * {Lc}u + j] = M[idx];
    """
    kern = _get_kernel(key, ["M"], ["M_hf"], src)
    return kern(
        inputs=[M],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B * nc * H, Lc, Lc)],
        output_dtypes=[mx.float32],
    )[0]


def _transpose_u_hf(u_c, B: int, nc: int, Lc: int, H: int, N: int, P: int):
    """u_c (B,nc,Lc,H,N,P) → u_hf (B*nc*H, Lc, P*N) [p-major inner index]."""
    total = B * nc * Lc * H * N * P
    key = f"scan_u_{B}_{nc}_{Lc}_{H}_{N}_{P}"
    src = f"""
        uint idx = thread_position_in_grid.x;
        if (idx >= {total}u) return;
        uint p = idx % {P}u;
        uint n = (idx / {P}u) % {N}u;
        uint h = (idx / {P * N}u) % {H}u;
        uint l = (idx / {P * N * H}u) % {Lc}u;
        uint c = (idx / {P * N * H * Lc}u) % {nc}u;
        uint b = idx / {P * N * H * Lc * nc}u;
        uint bch = b * {nc * H}u + c * {H}u + h;
        u_hf[bch * {Lc * P * N}u + l * {P * N}u + p * {N}u + n] = u_c[idx];
    """
    kern = _get_kernel(key, ["u_c"], ["u_hf"], src)
    return kern(
        inputs=[u_c],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B * nc * H, Lc, P * N)],
        output_dtypes=[u_c.dtype],
    )[0]


def _transpose_C_hf(C_c, B: int, nc: int, Lc: int, H: int, N: int, R: int):
    """C_c (B,nc,Lc,H,N,R) → C_hf (B*nc*H, Lc, N*R)."""
    total = B * nc * Lc * H * N * R
    key = f"scan_C_{B}_{nc}_{Lc}_{H}_{N}_{R}"
    src = f"""
        uint idx = thread_position_in_grid.x;
        if (idx >= {total}u) return;
        uint r = idx % {R}u;
        uint n = (idx / {R}u) % {N}u;
        uint h = (idx / {R * N}u) % {H}u;
        uint l = (idx / {R * N * H}u) % {Lc}u;
        uint c = (idx / {R * N * H * Lc}u) % {nc}u;
        uint b = idx / {R * N * H * Lc * nc}u;
        uint bch = b * {nc * H}u + c * {H}u + h;
        C_hf[bch * {Lc * N * R}u + l * {N * R}u + n * {R}u + r] = C_c[idx];
    """
    kern = _get_kernel(key, ["C_c"], ["C_hf"], src)
    return kern(
        inputs=[C_c],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B * nc * H, Lc, N * R)],
        output_dtypes=[C_c.dtype],
    )[0]


def _chunk_scan_metal_impl(u, la, C_rot, tri_mask, chunk_size, h_init=None):
    B, L, H, N, P = u.shape
    R = C_rot.shape[-1]
    Lc = chunk_size
    L_orig = L
    pad = (Lc - L % Lc) % Lc
    if pad:
        u     = mx.pad(u,     [(0,0),(0,pad),(0,0),(0,0),(0,0)])
        la    = mx.pad(la,    [(0,0),(0,pad),(0,0)])
        C_rot = mx.pad(C_rot, [(0,0),(0,pad),(0,0),(0,0),(0,0)])
        L = L + pad
    nc = L // Lc
    BncH = B * nc * H

    u_c  = u.reshape(B, nc, Lc, H, N, P)
    la_c = la.reshape(B, nc, Lc, H)
    C_c  = C_rot.reshape(B, nc, Lc, H, N, R)

    # ── Intra-chunk: Metal transpose → batched matmul ─────────────────────────
    la_cum = mx.cumsum(la_c, axis=2)
    log_M  = la_cum[:, :, :, None, :] - la_cum[:, :, None, :, :]
    log_M  = mx.where(tri_mask[None, None, :, :, None], log_M,
                      mx.array(-1e9, dtype=log_M.dtype))
    M = mx.exp(log_M)                                           # (B,nc,Lc,Lc,H) float32

    M_hf = _transpose_M_hf(M, B, nc, Lc, H)                   # (BncH, Lc, Lc) float32
    u_hf = _transpose_u_hf(u_c, B, nc, Lc, H, N, P)           # (BncH, Lc, P*N) dtype
    C_hf = _transpose_C_hf(C_c, B, nc, Lc, H, N, R)           # (BncH, Lc, N*R) dtype

    # h_intra_hf[bch, i, p*N+n] = sum_j M[b,c,i,j,h] * u[b,c,j,h,n,p]
    h_intra_hf = mx.matmul(M_hf.astype(u.dtype), u_hf)        # (BncH, Lc, P*N)

    # y_diag: per (b,c,l,h) do (P,N) @ (N,R) → (P,R)
    h_r    = h_intra_hf.reshape(BncH * Lc, P, N)
    C_r    = C_hf.reshape(BncH * Lc, N, R)
    y_diag = (mx.matmul(h_r, C_r)                             # (BncH*Lc, P, R)
              .reshape(B, nc, H, Lc, P, R)
              .transpose(0, 1, 3, 2, 4, 5))                    # (B, nc, Lc, H, P, R)

    # ── Inter-chunk carry ─────────────────────────────────────────────────────
    # Recover h_intra last token (B, nc, H, N, P) from h_intra_hf.
    # h_intra_hf.reshape(B,nc,H,Lc,P*N)[:,  :, :, -1, :] → (B,nc,H,P*N)
    # → reshape (B,nc,H,P,N) → transpose (B,nc,H,N,P)
    h_last = (h_intra_hf
              .reshape(B, nc, H, Lc, P * N)[:, :, :, -1, :]
              .reshape(B, nc, H, P, N)
              .transpose(0, 1, 2, 4, 3))                        # (B, nc, H, N, P)

    decay  = mx.exp(mx.sum(la_c, axis=2)).astype(u.dtype)       # (B, nc, H)
    h_prev = (mx.zeros((B, H, N, P), dtype=u.dtype)
              if h_init is None else h_init.astype(u.dtype))
    h_inter_list = []
    for c in range(nc):
        h_inter_list.append(h_prev)
        h_prev = (h_prev * decay[:, c].reshape(B, H, 1, 1)
                  + h_last[:, c])                               # h_last already u.dtype
    h_inter = mx.stack(h_inter_list, axis=1)                   # already u.dtype

    # ── Off-diagonal: einsum fine here (h_inter ≪ M in size) ─────────────────
    cdec_scale = mx.exp(la_cum).astype(u.dtype)
    c_dec  = C_c * cdec_scale[..., None, None]
    y_off  = mx.einsum("bchnp,bclhnr->bclhpr", h_inter, c_dec)

    y = (y_diag + y_off).reshape(B, L, H, P, R)
    if L_orig != L:
        y = y[:, :L_orig]
    return y, h_prev


def chunk_scan_metal(u, la, C_rot, chunk_size, h_init=None, _tri_mask=None):
    """Metal-kernel-accelerated chunk_scan — same API as chunk_scan.

    Uses GPU-parallel transpose kernels to convert M, u, C from the
    (…, H)-last layout (which forces strided GEMM) to a head-first
    contiguous layout (BncH, …) that maps to efficient batched matmul.

    Falls back to the pure-MLX path if mx.fast.metal_kernel is unavailable.
    """
    if _tri_mask is None:
        _tri_mask = mx.tri(chunk_size, dtype=mx.bool_)
    if not hasattr(mx.fast, "metal_kernel"):
        return _chunk_scan_mlx(u, la, C_rot, _tri_mask, chunk_size, h_init=h_init)
    return _chunk_scan_metal_impl(u, la, C_rot, _tri_mask, chunk_size, h_init=h_init)


# ── REJECTED: Flash-style fused chunk scan ────────────────────────────────────
#
# Attempt: fuse h_intra (M @ u) + C contraction into ONE Metal kernel, keeping
# h_intra in thread registers to avoid the ~200 MB GMEM roundtrip that the
# 2-einsum path pays at L=512.  Grid (B*nc*H*Lc, P), threadgroup (Lc,) sharing
# la_cum; each thread accumulates h_intra[i,:,p] in registers (N-tiled to 8),
# then contracts with C immediately.
#
# RESULT: 7-8× SLOWER across all L, and NaN at L>=1024.  Rejected.
#
# Root cause — the premise was wrong.  h_intra = M @ u is a batched GEMM, and
# MLX's einsum already dispatches it to Apple's simdgroup_matrix GEMM running
# near peak FLOPs.  chunk_scan is *compute-bound on that GEMM*, not bandwidth-
# bound, so the 200 MB "saving" buys nothing.  Meanwhile the hand-written kernel
# has terrible occupancy (only Lc=64 threads/TG), heavy warp divergence (causal
# j-loop: thread i runs i iterations), and uncoalesced strided u loads.
#
# Conclusion: the chunk_scan einsums (h_intra, y_diag, y_off) are already
# optimal on M2 Pro.  No custom Metal kernel beats MLX batched GEMM for these
# shapes.  Likewise chunk_scan_metal (transpose→contiguous GEMM) is ~1.7×
# slower (see note at top of file).  Prefill scan is solved; do not retry.


# ── Correctness test ──────────────────────────────────────────────────────────

def _run_correctness_test(B=1, L=128, H=4, N=8, P=4, R=4, chunk_size=32,
                           dtype=mx.bfloat16, seed=42, use_h_init=True):
    key = mx.random.key(seed)
    def _randn(*shape):
        nonlocal key
        key, sub = mx.random.split(key)
        return mx.random.normal(shape=shape, key=sub).astype(dtype)

    u = _randn(B, L, H, N, P)
    C_rot = _randn(B, L, H, N, R)
    h_init_arr = _randn(B, H, N, P) if use_h_init else None
    la = (mx.random.normal(shape=(B, L, H), key=key) * 0.1 - 0.5).astype(mx.float32)
    mx.eval(u, C_rot, la)
    if h_init_arr is not None:
        mx.eval(h_init_arr)

    tri = mx.tri(chunk_size, dtype=mx.bool_)
    y_bf, h_bf = chunk_scan(u, la, C_rot, chunk_size, h_init=h_init_arr, _tri_mask=tri)
    mx.eval(y_bf, h_bf)

    y_gt, h_gt = _chunk_scan_mlx(
        u.astype(mx.float32), la, C_rot.astype(mx.float32),
        mx.tri(chunk_size, dtype=mx.bool_), chunk_size,
        h_init=(h_init_arr.astype(mx.float32) if h_init_arr is not None else None))
    mx.eval(y_gt, h_gt)

    err_y = float(mx.max(mx.abs(y_bf.astype(mx.float32) - y_gt.astype(mx.float32))).item())
    err_h = float(mx.max(mx.abs(h_bf.astype(mx.float32) - h_gt.astype(mx.float32))).item())
    print(f"[scan] B={B} L={L} H={H} N={N} P={P} R={R} Lc={chunk_size} h_init={use_h_init}")
    print(f"  |y| = {err_y:.2e}   |h| = {err_h:.2e}   {'PASS ✓' if err_y < 1.0 else 'FAIL ✗'}")
    return err_y < 1.0


def _run_metal_correctness_test(B=1, L=128, H=4, N=8, P=4, R=4, chunk_size=32,
                                 dtype=mx.bfloat16, seed=42, use_h_init=True):
    """Compare chunk_scan_metal output against the float32 reference."""
    if not hasattr(mx.fast, "metal_kernel"):
        print("  [metal] mx.fast.metal_kernel not available — skip")
        return True

    key = mx.random.key(seed)
    def _randn(*shape):
        nonlocal key
        key, sub = mx.random.split(key)
        return mx.random.normal(shape=shape, key=sub).astype(dtype)

    u = _randn(B, L, H, N, P)
    C_rot = _randn(B, L, H, N, R)
    h_init_arr = _randn(B, H, N, P) if use_h_init else None
    la = (mx.random.normal(shape=(B, L, H), key=key) * 0.1 - 0.5).astype(mx.float32)
    mx.eval(u, C_rot, la)
    if h_init_arr is not None:
        mx.eval(h_init_arr)

    tri = mx.tri(chunk_size, dtype=mx.bool_)
    y_m, h_m = chunk_scan_metal(u, la, C_rot, chunk_size, h_init=h_init_arr, _tri_mask=tri)
    mx.eval(y_m, h_m)

    y_gt, h_gt = _chunk_scan_mlx(
        u.astype(mx.float32), la, C_rot.astype(mx.float32),
        tri, chunk_size,
        h_init=(h_init_arr.astype(mx.float32) if h_init_arr is not None else None))
    mx.eval(y_gt, h_gt)

    err_y = float(mx.max(mx.abs(y_m.astype(mx.float32) - y_gt.astype(mx.float32))).item())
    err_h = float(mx.max(mx.abs(h_m.astype(mx.float32) - h_gt.astype(mx.float32))).item())
    ok = err_y < 1.0
    print(f"[metal] B={B} L={L} H={H} N={N} P={P} R={R} Lc={chunk_size} h_init={use_h_init}")
    print(f"  |y| = {err_y:.2e}   |h| = {err_h:.2e}   {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok


if __name__ == "__main__":
    print("=== pure-MLX scan ===")
    for kw in [
        {},
        {"use_h_init": False},
        {"L": 256, "H": 4, "N": 8, "P": 4, "R": 4, "chunk_size": 32, "use_h_init": True},
        {"B": 2, "L": 512, "H": 8, "N": 16, "P": 8, "R": 4, "chunk_size": 64},
    ]:
        _run_correctness_test(**kw)

    print("\n=== Metal-kernel scan ===")
    for kw in [
        {},
        {"use_h_init": False},
        {"L": 256, "H": 4, "N": 8, "P": 4, "R": 4, "chunk_size": 32, "use_h_init": True},
        {"B": 2, "L": 512, "H": 8, "N": 16, "P": 8, "R": 4, "chunk_size": 64},
    ]:
        _run_metal_correctness_test(**kw)
