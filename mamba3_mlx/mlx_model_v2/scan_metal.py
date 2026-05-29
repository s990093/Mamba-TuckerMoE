"""Metal-kernel accelerated SSM chunk scan for Mamba3Block (v2).

The v2 copy of ``scan_metal.py`` adds one new public API on top of v1:

* :func:`chunk_scan_per_pos` — returns ``(y, h_per_pos)`` instead of just
  ``(y, h_final)``.  This lets the speculative-decoder verify path get rid
  of its pure-MLX ``_scan_per_pos`` (which builds the dense O(Lc²) M matrix
  and einsums against ``u``) and replace it with a Metal kernel that runs
  the recurrence directly and reads off every intermediate state.

The base ``chunk_scan`` here is intentionally identical to v1 — the
SIMD-shuffle / Kogge-Stone parallel-prefix attempt is documented in
``LOOKAHEAD_COT_RESULTS.md`` §10.2 as a dead end (the bf16 reassociation
drift hurts cot-cache hit rate end-to-end despite kernel-level wins on the
intra-chunk micro-benchmark).

Two custom Metal kernels make up the chunk-parallel scan core:

1. Intra-chunk serial scan (``ssm_intra_chunk_scan``):
   One thread per ``(d, h, b_c)``; sequential O(Lc) loop inside that thread
   accumulates ``h[t] = exp(la[t]) · h[t-1] + u[t]`` with fp32
   accumulation and bf16 write-back.  Same kernel as v1.

2. Inter-chunk carry (``ssm_inter_chunk_scan``):
   One thread per ``(d, H, B)``; sequential nc-step propagation entirely in
   GPU registers — zero Python overhead, no cross-thread synchronisation.
   Kept serial since ``nc`` is typically 1–8 in our regime and the
   single-dispatch design already beats the prior Kogge-Stone MLX pass
   (see project memory ``scan_optimizations``).

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


# Experimental SIMD-shuffle Kogge-Stone kernel — kept for reference but NOT
# wired into ``chunk_scan`` because the bf16 reassociation drifts the model
# output enough to drop cot-cache hit rate by ~15% end-to-end (see results doc).
_INTRA_CHUNK_PP_SRC = """
    // Hardware-aware hierarchical parallel-prefix scan on Apple Silicon.
    //
    // Grid    : (LC, NP, H*B*nc)
    // TG      : (LC, 1, 1)        — each threadgroup owns one (d, hh, b_c)
    //                                element and runs LC cooperating threads.
    //
    // Three phases:
    //   1. Intra-SIMD Kogge-Stone via ``simd_shuffle_up`` (registers only,
    //      no threadgroup memory, no barriers).  For Apple's 32-lane SIMD
    //      groups this is 5 stages of register-to-register shuffles.
    //   2. Cross-SIMD aggregate exchange via threadgroup memory (1 barrier
    //      per group merge).  For LC=64 (NSG=2) this is a single offset
    //      propagation; for LC=128 (NSG=4) a tiny serial scan suffices.
    //   3. Downsweep: SIMD groups beyond the first combine the accumulated
    //      preceding-group aggregate with their local intra-SIMD prefix.
    //
    // The associative operator on pairs (alpha, h) is
    //     (a_prev, h_prev) ∘ (a_cur, h_cur) = (a_prev · a_cur,
    //                                          a_cur · h_prev + h_cur)
    // — same as the prior Hillis-Steele variant but without the per-stage
    // double-barrier overhead.  All accumulation stays in fp32; bf16 cast
    // only at the final write.
    //
    // NSG = ceil(LC / 32) is injected as a compile-time constant.

    uint t   = thread_position_in_grid.x;          // [0, LC)
    uint d   = thread_position_in_grid.y;          // [0, NP)
    uint bh  = thread_position_in_grid.z;          // packed (b_c, hh)
    if (t >= LC || d >= NP || bh >= BNC * H) return;
    uint hh  = bh % H;
    uint b_c = bh / H;

    uint sid = t & 31u;            // simd_lane_id (== t % 32, threadgroup is x-only)
    uint sgi = t >> 5;             // simd_group index within threadgroup (t / 32)

    // ── Leaf load: each thread holds its own (alpha_t, u_t) pair ──────────
    float la_t = la_c[b_c * LC * H + t * H + hh];
    la_t = la_t > 40.0f ? 40.0f : (la_t < -40.0f ? -40.0f : la_t);
    float a_t = metal::exp(la_t);
    float h_t = float(u_c[b_c * LC * H * NP + t * H * NP + hh * NP + d]);

    // ── Phase 1: Intra-SIMD Kogge-Stone via simd_shuffle_up ──────────────
    // No barriers — register-to-register shuffles inside the 32-lane group.
    // Stages: 1, 2, 4, 8, 16  (5 ops, fully unrolled by the compiler).
    for (uint off = 1u; off < 32u && off < LC; off <<= 1) {
        float a_prev = simd_shuffle_up(a_t, off);
        float h_prev = simd_shuffle_up(h_t, off);
        if (sid >= off) {
            float new_a = a_prev * a_t;
            float new_h = a_t * h_prev + h_t;
            a_t = new_a;
            h_t = new_h;
        }
    }

    // ── Phase 2 + 3: cross-SIMD aggregate + downsweep ──────────────────────
    // Only needed when LC spans more than one SIMD group (NSG > 1).
    if (NSG > 1) {
        threadgroup float sh_a[NSG];
        threadgroup float sh_h[NSG];

        // Each SIMD group's last lane (sid == 31) holds the local group
        // aggregate after Phase 1.  Publish to shared memory.
        if (sid == 31u) {
            sh_a[sgi] = a_t;
            sh_h[sgi] = h_t;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        // Phase 3 — every SIMD group except #0 combines the aggregate of all
        // strictly-preceding SIMD groups into its local prefix.  NSG is
        // small (≤4 for LC up to 128) so a tiny serial accumulator beats
        // any tree dance here.
        if (sgi > 0u) {
            float a_off = sh_a[0];
            float h_off = sh_h[0];
            for (uint k = 1u; k < sgi; ++k) {
                float a_g = sh_a[k];
                float h_g = sh_h[k];
                float new_a = a_off * a_g;
                float new_h = a_g * h_off + h_g;
                a_off = new_a;
                h_off = new_h;
            }
            // Apply the preceding-group aggregate to this lane's prefix.
            float new_a = a_off * a_t;
            float new_h = a_t * h_off + h_t;
            a_t = new_a;
            h_t = new_h;
        }
    }

    // ── Write back ────────────────────────────────────────────────────────
    out[b_c * LC * H * NP + t * H * NP + hh * NP + d] = T(h_t);
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

    Identical kernel as v1.  Produces per-position SSM state with fp32
    internal accumulation and bf16 write-back at every position.

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


def _intra_chunk_metal_pp(la_c: mx.array, u_c_flat: mx.array,
                           B: int, nc: int, Lc: int, H: int, NP: int) -> mx.array:
    """Experimental SIMD-shuffle Kogge-Stone variant.

    NOT used by the production code paths — the bf16 reassociation drift
    breaks end-to-end SJD's cot cache hit rate.  Kept here for kernel-level
    micro-benchmarking and future numerical-stability work (Kahan compensated
    summation in the merge step, etc.).
    """
    nsg = (Lc + 31) // 32
    dtype_tag = str(u_c_flat.dtype).replace(".", "_").replace(":", "_")
    kernel = mx.fast.metal_kernel(
        name=f"ssm_intra_chunk_pp_{B}_{nc}_{Lc}_{H}_{NP}_{dtype_tag}",
        input_names=["la_c", "u_c"],
        output_names=["out"],
        source=_INTRA_CHUNK_PP_SRC,
        header=(
            f"constant uint BNC = {B * nc}; "
            f"constant uint LC = {Lc}; "
            f"constant uint H = {H}; "
            f"constant uint NP = {NP}; "
            f"constant uint NSG = {nsg};"
        ),
    )
    return kernel(
        inputs=[la_c, u_c_flat],
        template=[("T", u_c_flat.dtype)],
        grid=(Lc, NP, B * nc * H),
        threadgroup=(Lc, 1, 1),
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


# ── Per-position scan (new v2 contribution) ──────────────────────────────────
#
# The speculative-decoder verify path (``speculative/forward.py::_scan_per_pos``)
# needs the SSM state at every position, not just the final state, so it can
# pluck out the post-accept state without a replay forward.  v1's pure-MLX
# implementation materialises a dense O(Lc²) ``M`` matrix and einsums it
# against ``u`` — fine for short Lc but wasteful FLOPs that the existing
# Metal serial kernel already computes for free as a by-product (the
# intra-chunk kernel writes ``h_t`` to every position).
#
# ``chunk_scan_per_pos`` reuses the same intra/inter-chunk Metal kernels as
# ``chunk_scan`` and additionally:
#   1. expands the intra-chunk output into the full per-position state via
#      ``h_inter * exp(la_cum) + h_intra`` (a pointwise broadcast — cheap);
#   2. adds the h_init superposition contribution
#      ``h_init * exp(cumsum(la)_full)`` at every position;
#   3. computes ``y_per_pos = h_per_pos @ C_rot`` once instead of v1's
#      separate ``y_diag + y_off`` decomposition.

def _chunk_scan_per_pos_metal(u, la, C_rot, chunk_size, h_init=None):
    """Per-position Metal scan.  Returns ``(y, h_per_pos)``.

    y:         (B, L, H, P, R)
    h_per_pos: (B, L, H, N, P)
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

    la_c = la.reshape(B, nc, Lc, H)
    u_c_flat = u.reshape(B, nc, Lc, H, NP)
    C_c = C_rot.reshape(B, nc, Lc, H, N, R)

    # 1. Intra-chunk per-position state (zero h_init within each chunk).
    h_intra_flat = _intra_chunk_metal(la_c, u_c_flat, B, nc, Lc, H, NP)
    h_intra = h_intra_flat.reshape(B, nc, Lc, H, N, P)

    # 2. Inter-chunk carry — state entering each chunk (zero-init globally).
    log_d = mx.sum(la_c.astype(mx.float32), axis=2)
    x_in = h_intra_flat[:, :, -1]
    h_inter_flat, _h_final = _inter_chunk_metal(log_d, x_in, B, nc, H, NP, dtype)
    h_inter = h_inter_flat.reshape(B, nc, H, N, P)  # (B, nc, H, N, P)

    # 3. Within-chunk decay scale.
    la_cum = mx.cumsum(la_c.astype(mx.float32), axis=2)   # (B, nc, Lc, H)
    cdec_scale = mx.exp(la_cum).astype(dtype)              # (B, nc, Lc, H)

    # 4. Per-position state assuming h_init=0 globally:
    #        h[c,l] = h_inter[c] · exp(la_cum[c,l]) + h_intra[c,l]
    h_per_pos_zero = (
        h_inter[:, :, None]                                # (B, nc, 1, H, N, P)
        * cdec_scale[..., None, None]                      # (B, nc, Lc, H, 1, 1)
        + h_intra                                           # (B, nc, Lc, H, N, P)
    )

    # 5. y_per_pos = h_per_pos @ C_rot (fused with C_c's RoPE rotation).
    y_zero = mx.einsum("bclhnp,bclhnr->bclhpr", h_per_pos_zero, C_c)
    y_zero = y_zero.reshape(B, L, H, P, R)
    h_per_pos = h_per_pos_zero.reshape(B, L, H, N, P)

    # 6. h_init superposition correction (per-position).
    if h_init is not None:
        h_init_f = h_init.astype(mx.float32)                                # (B, H, N, P)
        log_full = mx.clip(mx.cumsum(la, axis=1), -88.0, 88.0)              # (B, L, H)
        alpha_full = mx.exp(log_full)                                       # (B, L, H)
        h_init_per_pos = (
            h_init_f[:, None] * alpha_full[:, :, :, None, None]             # (B, L, H, N, P)
        )
        h_per_pos = (h_per_pos.astype(mx.float32) + h_init_per_pos).astype(dtype)
        y_init = mx.einsum(
            "blhnp,blhnr->blhpr",
            h_init_per_pos, C_rot.astype(mx.float32),
        )
        y = (y_zero.astype(mx.float32) + y_init).astype(dtype)
    else:
        y = y_zero

    if L_orig != L:
        y = y[:, :L_orig]
        h_per_pos = h_per_pos[:, :L_orig]
    return y, h_per_pos


def _chunk_scan_per_pos_mlx(u, la, C_rot, chunk_size, h_init=None):
    """Pure-MLX reference for ``chunk_scan_per_pos`` (Metal-fallback path).

    Mirrors the structure of v1's ``_chunk_scan_mlx`` but materialises every
    intermediate state — same math as the in-tree ``_scan_per_pos`` used by
    the speculative module today.
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
    la_c = la.reshape(B, nc, Lc, H).astype(mx.float32)
    C_c = C_rot.reshape(B, nc, Lc, H, N, R)

    la_cum = mx.cumsum(la_c, axis=2)
    log_M = la_cum[:, :, :, None, :] - la_cum[:, :, None, :, :]
    tri = mx.tri(Lc, dtype=mx.bool_)
    log_M = mx.where(tri[None, None, :, :, None], log_M,
                     mx.array(-1e9, dtype=log_M.dtype))
    M = mx.exp(log_M).astype(u.dtype)
    h_intra = mx.einsum("bcijh,bcjhnp->bcihnp", M, u_c)

    decay = mx.exp(mx.sum(la_c, axis=2))
    h_prev = (mx.zeros((B, H, N, P), dtype=u.dtype)
              if h_init is None else h_init.astype(u.dtype))
    h_inter_list = []
    for c in range(nc):
        h_inter_list.append(h_prev)
        h_prev = h_prev * decay[:, c].reshape(B, H, 1, 1).astype(u.dtype) + h_intra[:, c, -1]
    h_inter = mx.stack(h_inter_list, axis=1)        # (B, nc, H, N, P)

    cdec_scale = mx.exp(la_cum).astype(u.dtype)     # (B, nc, Lc, H)
    h_per_pos = (
        h_inter[:, :, None] * cdec_scale[..., None, None]
        + h_intra
    )                                                # (B, nc, Lc, H, N, P)
    y = mx.einsum("bclhnp,bclhnr->bclhpr", h_per_pos, C_c)
    y = y.reshape(B, L, H, P, R)
    h_per_pos = h_per_pos.reshape(B, L, H, N, P)
    if L_orig != L:
        y = y[:, :L_orig]
        h_per_pos = h_per_pos[:, :L_orig]
    return y, h_per_pos


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_scan_per_pos(u, la, C_rot, chunk_size, h_init=None):
    """Chunk-parallel SSM scan that returns the per-position state.

    Drop-in replacement for ``speculative.forward._scan_per_pos`` that goes
    through the Metal intra/inter chunk kernels (when available).  Numerics
    match the pure-MLX reference within bf16 quantisation noise.

    Args:
        u:          (B, L, H, N, P)
        la:         (B, L, H) float32 — precomputed dt_b * A_b
        C_rot:      (B, L, H, N, R) — RoPE-rotated C projection
        chunk_size: intra-chunk length Lc (must evenly divide L after pad)
        h_init:     (B, H, N, P) or None

    Returns:
        y:         (B, L, H, P, R)
        h_per_pos: (B, L, H, N, P)
    """
    if _HAS_METAL_KERNEL:
        return _chunk_scan_per_pos_metal(u, la, C_rot, chunk_size, h_init=h_init)
    return _chunk_scan_per_pos_mlx(u, la, C_rot, chunk_size, h_init=h_init)


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
