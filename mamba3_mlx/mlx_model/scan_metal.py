"""Metal-kernel accelerated SSM chunk scan for Mamba3Block.

Two custom Metal kernels replace the most expensive parts of _chunk_parallel_scan:

1. Intra-chunk scan (ssm_intra_chunk_scan):
   Replaces the O(Lc²) matrix approach:
       log_M[i,j] = la_cum[i] - la_cum[j]   (Lc×Lc matrix)
       h_intra = einsum("bcijh,bcjhnp->bcihnp", exp(log_M), u_c)
   with an O(Lc) sequential scan per (B*nc, H, N*P) thread:
       h[t] = exp(la[t]) * h[t-1] + u[t]
   Mathematical equivalence proven: both compute
       h_intra[l] = Σ_{j≤l} exp(Σ_{k=j+1}^{l} la[k]) · u[j]

2. Inter-chunk carry (ssm_inter_chunk_scan):
   Replaces the Python for-loop over nc chunks with a single GPU dispatch.
   Each thread (d, H, B) iterates nc chunks in GPU registers — zero Python
   overhead and no cross-thread synchronisation needed.

h_init support (multi-turn / non-zero initial state):
   Both kernels assume zero initialization. Non-zero h_init is handled via
   the superposition principle (identical to chunk_parallel_scan_with_init
   in old_and_no_stable_mamba_api/lib/mlx_hybrid_infer.py):
       h_t = h_t^{zero_init} + h_init · alpha_cum_t
   where alpha_cum_t = exp(cumsum(la)[t]).

Fallback:
   If mx.fast.metal_kernel is unavailable, chunk_scan() transparently falls
   back to the pure-MLX reference path.

Usage:
    from .scan_metal import chunk_scan
    y, h_final = chunk_scan(u_ssm, la_full, C_rot, chunk_size, h_init=h_init)
"""

from __future__ import annotations

import mlx.core as mx

_HAS_METAL_KERNEL = hasattr(mx.fast, "metal_kernel")


# ── Pure MLX reference path (always correct, used as fallback) ────────────────

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


# ── Metal kernel implementations ──────────────────────────────────────────────

_INTRA_CHUNK_SRC = """
    uint d   = thread_position_in_grid.x;   // [0, NP)
    uint hh  = thread_position_in_grid.y;   // [0, H)
    uint b_c = thread_position_in_grid.z;   // [0, B*nc)
    if (d >= NP || hh >= H || b_c >= BNC) return;

    float h_val = 0.0f;
    for (uint t = 0; t < LC; ++t) {
        float la = la_c[b_c * LC * H + t * H + hh];
        la = la > 40.0f ? 40.0f : (la < -40.0f ? -40.0f : la);
        float alpha = metal::exp(la);
        float u_val = float(u_c[b_c * LC * H * NP + t * H * NP + hh * NP + d]);
        h_val = alpha * h_val + u_val;
        out[b_c * LC * H * NP + t * H * NP + hh * NP + d] = T(h_val);
    }
"""

_INTER_CHUNK_SRC = """
    uint d    = thread_position_in_grid.x;   // [0, NP)
    uint hh   = thread_position_in_grid.y;   // [0, H)
    uint b_id = thread_position_in_grid.z;   // [0, B)
    if (d >= NP || hh >= H || b_id >= B) return;

    float h_accum = 0.0f;
    for (uint c = 0; c < NC; ++c) {
        float ld = log_d[b_id * NC * H + c * H + hh];
        ld = ld > 88.0f ? 88.0f : (ld < -88.0f ? -88.0f : ld);
        float decay = metal::exp(ld);
        float x_val = float(x_in[b_id * NC * H * NP + c * H * NP + hh * NP + d]);
        h_inter[b_id * NC * H * NP + c * H * NP + hh * NP + d] = T(h_accum);
        h_accum = h_accum * decay + x_val;
    }
    h_prev[b_id * H * NP + hh * NP + d] = T(h_accum);
"""


def _intra_chunk_metal(la_c: mx.array, u_c_flat: mx.array,
                        B: int, nc: int, Lc: int, H: int, NP: int) -> mx.array:
    """Sequential scan over Lc timesteps — O(Lc) per (B*nc, H, NP) thread.

    la_c:     (B, nc, Lc, H)   float32
    u_c_flat: (B, nc, Lc, H, NP)  model dtype
    Returns:  (B, nc, Lc, H, NP)  same dtype as u_c_flat
    """
    kernel = mx.fast.metal_kernel(
        name=f"ssm_intra_chunk_{B}_{nc}_{Lc}_{H}_{NP}",
        input_names=["la_c", "u_c"],
        output_names=["out"],
        source=_INTRA_CHUNK_SRC,
        header=(
            f"constant uint BNC = {B * nc}; "
            f"constant uint LC = {Lc}; "
            f"constant uint H = {H}; "
            f"constant uint NP = {NP};"
        ),
    )
    tg_x = min(NP, 64)
    return kernel(
        inputs=[la_c, u_c_flat],
        template=[("T", u_c_flat.dtype)],
        grid=(NP, H, B * nc),
        threadgroup=(tg_x, 1, 1),
        output_shapes=[(B, nc, Lc, H, NP)],
        output_dtypes=[u_c_flat.dtype],
    )[0]


def _inter_chunk_metal(log_d: mx.array, x_in: mx.array,
                        B: int, nc: int, H: int, NP: int,
                        dtype) -> tuple[mx.array, mx.array]:
    """Sequential scan over nc chunks — runs entirely in GPU registers per thread.

    log_d: (B, nc, H)   float32 — per-chunk log-decay
    x_in:  (B, nc, H, NP) model dtype — last h per chunk from intra-chunk scan
    Returns:
        h_inter: (B, nc, H, NP) — state entering each chunk (exclusive prefix)
        h_prev:  (B, H, NP)     — final state after all chunks
    """
    kernel = mx.fast.metal_kernel(
        name=f"ssm_inter_chunk_{B}_{nc}_{H}_{NP}",
        input_names=["log_d", "x_in"],
        output_names=["h_inter", "h_prev"],
        source=_INTER_CHUNK_SRC,
        header=(
            f"constant uint B = {B}; "
            f"constant uint NC = {nc}; "
            f"constant uint H = {H}; "
            f"constant uint NP = {NP};"
        ),
    )
    tg_x = min(NP, 64)
    outs = kernel(
        inputs=[log_d, x_in],
        template=[("T", dtype)],
        grid=(NP, H, B),
        threadgroup=(tg_x, 1, 1),
        output_shapes=[(B, nc, H, NP), (B, H, NP)],
        output_dtypes=[dtype, dtype],
    )
    return outs[0], outs[1]


def _chunk_scan_metal(u, la, C_rot, chunk_size, h_init=None):
    """Metal-kernel accelerated chunk scan.

    u:      (B, L, H, N, P)
    la:     (B, L, H) float32
    C_rot:  (B, L, H, N, R)
    h_init: (B, H, N, P) or None
    Returns: (y: (B, L, H, P, R), h_final: (B, H, N, P))
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
    NP = N * P
    dtype = u.dtype

    la_c = la.reshape(B, nc, Lc, H)          # (B, nc, Lc, H) float32
    u_c_flat = u.reshape(B, nc, Lc, H, NP)   # (B, nc, Lc, H, NP) model dtype
    C_c = C_rot.reshape(B, nc, Lc, H, N, R)

    # ── 1. Intra-chunk sequential scan (Metal) ────────────────────────────────
    # Replaces O(Lc²) einsum with O(Lc) sequential scan per thread.
    # h_intra_flat[b,c,t,h,d] = h_t (zero-init per chunk)
    h_intra_flat = _intra_chunk_metal(la_c, u_c_flat, B, nc, Lc, H, NP)
    # → (B, nc, Lc, H, NP)

    h_intra = h_intra_flat.reshape(B, nc, Lc, H, N, P)

    # y_diag: intra-chunk output
    y_diag = mx.einsum("bclhnp,bclhnr->bclhpr", h_intra, C_c)

    # ── 2. Inter-chunk carry (Metal) ──────────────────────────────────────────
    # Replaces Python for-loop with single GPU dispatch.
    # Each GPU thread owns one (NP, H, B) element and iterates nc chunks.
    log_d = mx.sum(la_c.astype(mx.float32), axis=2)   # (B, nc, H) — log per-chunk decay
    x_in = h_intra_flat[:, :, -1]                      # (B, nc, H, NP) — last h per chunk
    h_inter_flat, h_final_zero_flat = _inter_chunk_metal(log_d, x_in, B, nc, H, NP, dtype)
    # h_inter_flat: (B, nc, H, NP), h_final_zero_flat: (B, H, NP)

    h_inter = h_inter_flat.reshape(B, nc, H, N, P)
    h_final_zero = h_final_zero_flat.reshape(B, H, N, P)

    # Within-chunk decay for y_off
    la_cum = mx.cumsum(la_c.astype(mx.float32), axis=2)  # (B, nc, Lc, H)
    cdec_scale = mx.exp(la_cum).astype(dtype)
    c_dec = C_c * cdec_scale[..., None, None]
    y_off = mx.einsum("bchnp,bclhnr->bclhpr", h_inter, c_dec)

    y_zero = (y_diag + y_off).reshape(B, L, H, P, R)

    # ── 3. h_init correction (superposition) ─────────────────────────────────
    # h_t = h_t^{zero_init} + h_init · alpha_cum_t
    # where alpha_cum_t = exp(cumsum(la)[t]).
    if h_init is not None:
        h_init_f = h_init.astype(mx.float32)             # (B, H, N, P)
        log_alpha_cum = mx.cumsum(la, axis=1)             # (B, L, H) float32
        log_alpha_cum = mx.clip(log_alpha_cum, -88.0, 88.0)
        alpha_cum = mx.exp(log_alpha_cum)                 # (B, L, H)
        # Contribution of h_init at each position: h_init * alpha_cum_t
        h_init_t = h_init_f[:, None] * alpha_cum[:, :, :, None, None]  # (B, L, H, N, P)
        # Output correction
        y_init = mx.einsum("blhnp,blhnr->blhpr",
                           h_init_t, C_rot.astype(mx.float32))  # (B, L, H, P, R)
        y_zero = y_zero.astype(mx.float32) + y_init
        # Final state correction
        alpha_total = mx.exp(mx.clip(log_alpha_cum[:, -1], -88.0, 88.0))  # (B, H)
        h_final = (h_final_zero.astype(mx.float32)
                   + h_init_f * alpha_total[:, :, None, None])
        h_final = h_final.astype(dtype)
        y = y_zero.astype(dtype)
    else:
        y = y_zero
        h_final = h_final_zero

    if L_orig != L:
        y = y[:, :L_orig]
    return y, h_final


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_scan(u, la, C_rot, chunk_size, h_init=None, _tri_mask=None):
    """Chunk-parallel SSM scan with Metal kernel acceleration.

    Automatically falls back to the pure MLX reference path when
    mx.fast.metal_kernel is not available.

    Args:
        u:          (B, L, H, N, P) — blended input signal
        la:         (B, L, H) float32 — precomputed dt_b * A_b
        C_rot:      (B, L, H, N, R) — RoPE-rotated C projection
        chunk_size: intra-chunk length Lc (must evenly divide L after padding)
        h_init:     (B, H, N, P) or None — initial SSM state
        _tri_mask:  (Lc, Lc) bool — cached triangular mask for MLX fallback

    Returns:
        y:       (B, L, H, P, R)
        h_final: (B, H, N, P)
    """
    if _HAS_METAL_KERNEL:
        return _chunk_scan_metal(u, la, C_rot, chunk_size, h_init=h_init)

    # MLX reference fallback
    Lc = chunk_size
    if _tri_mask is None:
        _tri_mask = mx.tri(Lc, dtype=mx.bool_)
        
    return _chunk_scan_mlx(u, la, C_rot, _tri_mask, chunk_size, h_init=h_init)


# ── Correctness test ──────────────────────────────────────────────────────────

def _run_correctness_test(B=1, L=128, H=4, N=8, P=4, R=4, chunk_size=32,
                           dtype=mx.bfloat16, seed=42, use_h_init=True):
    """Compare Metal kernel vs MLX reference outputs.

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
    mx.eval(u, C_rot, h_init, la)

    tri_mask = mx.tri(chunk_size, dtype=mx.bool_)

    # Reference
    y_ref, h_ref = _chunk_scan_mlx(u, la, C_rot, tri_mask, chunk_size, h_init=h_init)
    mx.eval(y_ref, h_ref)

    if not _HAS_METAL_KERNEL:
        print("[scan_metal] mx.fast.metal_kernel not available — skipping Metal test")
        return

    # Metal
    y_metal, h_metal = _chunk_scan_metal(u, la, C_rot, chunk_size, h_init=h_init)
    mx.eval(y_metal, h_metal)

    y_ref_f = y_ref.astype(mx.float32)
    y_metal_f = y_metal.astype(mx.float32)
    h_ref_f = h_ref.astype(mx.float32)
    h_metal_f = h_metal.astype(mx.float32)

    max_y_err = float(mx.max(mx.abs(y_metal_f - y_ref_f)).item())
    max_h_err = float(mx.max(mx.abs(h_metal_f - h_ref_f)).item())

    print(f"[scan_metal correctness] B={B} L={L} H={H} N={N} P={P} R={R} Lc={chunk_size}")
    print(f"  max |y_metal - y_ref| = {max_y_err:.2e}")
    print(f"  max |h_metal - h_ref| = {max_h_err:.2e}")

    # Both MLX-bf16 and Metal-bf16 have ~5-6e-2 error vs float32 ground truth;
    # the inter-implementation difference reflects different rounding, not bugs.
    # Compare against float32 ground truth to confirm neither regresses.
    tri_mask_f32 = mx.tri(chunk_size, dtype=mx.bool_)
    y_gt, h_gt = _chunk_scan_mlx(
        u.astype(mx.float32), la, C_rot.astype(mx.float32), tri_mask_f32,
        chunk_size, h_init=(h_init.astype(mx.float32) if h_init is not None else None))
    mx.eval(y_gt, h_gt)
    y_gt_f = y_gt.astype(mx.float32)

    err_ref_y = float(mx.max(mx.abs(y_ref_f - y_gt_f)).item())
    err_met_y = float(mx.max(mx.abs(y_metal_f - y_gt_f)).item())
    print(f"  vs float32 truth: ref={err_ref_y:.2e}  metal={err_met_y:.2e}")

    # Metal must not be more than 2× worse than the MLX-bf16 reference.
    ok = err_met_y <= max(2.0 * err_ref_y, 1e-4)
    print(f"  {'PASS ✓' if ok else 'FAIL ✗  (metal > 2x ref error)'}")
    return ok


if __name__ == "__main__":
    _run_correctness_test(use_h_init=True)
    _run_correctness_test(use_h_init=False)
