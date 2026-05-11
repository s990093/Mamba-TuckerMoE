#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark: Fused Mamba Mixer Kernel for decode (batch=1, seq_len=1).

Fuses the four most memory-intensive ops inside Mamba3Block.__call__:
  1. Einsum1: input_signal = einsum("hnr, hpr -> hnp", b_rotated, x_ssm)
  2. Lambda mixing: u = dv * (lv * inp + (1 - lv) * alpha * prev_input)
  3. SSM state update: h_new = prev_h * alpha + u
  4. Einsum2: y = einsum("hnp, hnr -> hpr", h_new, c_rotated)

Single Metal dispatch eliminates 3× (H×N×P) intermediate round-trips through
global memory (~600 KB saved per token for default config) and removes 6-9
kernel launch overheads.

Model dimensions (Mamba3Config defaults):
  H=24 heads, N=64 d_state, P=64 d_head, R=4 mimo_rank, G=1 groups

Usage:
  python metal/benchmark_fused_mamba_mixer.py
  python metal/benchmark_fused_mamba_mixer.py --trials 200 --warmup 20
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx

# ─── Model Dimensions ────────────────────────────────────────────────────────
H = 24   # n_heads  (d_inner / d_head = 1536 / 64)
N = 64   # d_state
P = 64   # d_head
R = 4    # mimo_rank


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Reference MLX Implementation (matches Mamba3Block decode L=1 exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def reference_mlx(
    b_rot: mx.array,      # (H, N, R)
    c_rot: mx.array,      # (H, N, R)
    x_ssm: mx.array,      # (H, P, R)
    prev_h: mx.array,     # (H, N, P)
    prev_input: mx.array, # (H, N, P)
    dt_b: mx.array,       # (H,)
    a_b: mx.array,        # (H,)
    lv: mx.array,         # (H,)
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Pure MLX reference matching lines 730-743 of mlx_hybrid_infer.py.

    Returns (h_new, new_input, y) where:
      h_new:     (H, N, P) updated SSM state
      new_input: (H, N, P) input_signal (stored as prev_input for next step)
      y:         (H, P, R) output of einsum2
    """
    alpha = mx.exp(dt_b * a_b)                       # (H,)
    dv = dt_b                                         # (H,)
    alpha_3 = alpha.reshape(H, 1, 1)
    dv_3 = dv.reshape(H, 1, 1)
    lv_3 = lv.reshape(H, 1, 1)

    # Einsum1: b_rot[h,n,r] × x_ssm[h,p,r] → input_signal[h,n,p]
    input_signal = mx.einsum("hnr, hpr -> hnp", b_rot, x_ssm)

    # Lambda mixing  (ip = prev_input for L=1 decode)
    u_ssm = lv_3 * dv_3 * input_signal + (1.0 - lv_3) * dv_3 * alpha_3 * prev_input

    # SSM state update
    h_new = prev_h * alpha_3 + u_ssm

    # Einsum2: h_new[h,n,p] × c_rot[h,n,r] → y[h,p,r]
    y = mx.einsum("hnp, hnr -> hpr", h_new, c_rot)

    return h_new, input_signal, y


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Fused Metal Kernel
# ═══════════════════════════════════════════════════════════════════════════════

_FUSED_MAMBA_MIXER_V1_SRC = r"""
    uint p = thread_position_in_grid.x;
    uint h = threadgroup_position_in_grid.y;
    uint lid = thread_index_in_threadgroup;

    if (p >= P_VAL || h >= H_VAL) return;

    // ── Cooperative load: b_rot[h,:,:] and c_rot[h,:,:] into SRAM ──
    threadgroup float b_sram[N_R];
    threadgroup float c_sram[N_R];

    uint bc_base = h * N_R;
    for (uint i = lid; i < N_R; i += P_VAL) {
        b_sram[i] = float(b_rot[bc_base + i]);
        c_sram[i] = float(c_rot[bc_base + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Per-thread: load x_ssm[h, p, 0..R-1] ──
    uint x_base = h * P_R + p * R_VAL;
    float xr0 = float(x_ssm[x_base + 0]);
    float xr1 = float(x_ssm[x_base + 1]);
    float xr2 = float(x_ssm[x_base + 2]);
    float xr3 = float(x_ssm[x_base + 3]);

    // ── Per-head scalars (precompute constants outside N loop) ──
    float dt  = float(dt_arr[h]);
    float a   = float(a_arr[h]);
    float lam = float(lv_arr[h]);

    float alpha = metal::exp(dt * a);
    alpha = alpha > 1e6f ? 1e6f : (alpha < -1e6f ? -1e6f : alpha);

    float coeff_inp  = dt * lam;
    float coeff_prev = dt * (1.0f - lam) * alpha;

    // ── Fused N-loop: einsum1 + lambda + ssm + einsum2 ──
    float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
    uint state_base = h * N_P + p;

    for (uint n = 0; n < N_VAL; ++n) {
        uint bn = n * R_VAL;

        // Einsum1: inp = Σ_r b[n,r] * x[r]
        float inp = b_sram[bn]     * xr0
                  + b_sram[bn + 1] * xr1
                  + b_sram[bn + 2] * xr2
                  + b_sram[bn + 3] * xr3;

        // Lambda mixing + SSM update (fused with precomputed coefficients)
        uint si = state_base + n * P_VAL;
        float h_val = float(prev_h[si]) * alpha + coeff_inp * inp + coeff_prev * float(prev_in[si]);

        new_h[si]       = T(h_val);
        new_prev_in[si] = T(inp);

        // Einsum2 accumulate: y[r] += h_val * c[n,r]
        y0 += h_val * c_sram[bn];
        y1 += h_val * c_sram[bn + 1];
        y2 += h_val * c_sram[bn + 2];
        y3 += h_val * c_sram[bn + 3];
    }

    // ── Write y output[h, p, :] ──
    uint yb = h * P_R + p * R_VAL;
    y_out[yb + 0] = T(y0);
    y_out[yb + 1] = T(y1);
    y_out[yb + 2] = T(y2);
    y_out[yb + 3] = T(y3);
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 2b. V2: Full Fused Kernel (Norm + RoPE + Einsum1 + Lambda + SSM + Einsum2)
# ═══════════════════════════════════════════════════════════════════════════════

_FUSED_MAMBA_MIXER_V2_SRC = r"""
    uint p = thread_position_in_grid.x;
    uint h = threadgroup_position_in_grid.y;
    uint lid = thread_index_in_threadgroup;

    if (p >= P_VAL || h >= H_VAL) return;

    // ── Shared memory ──
    threadgroup float b_sram[N_R];
    threadgroup float c_sram[N_R];

    // ═══ Phase 1: RMSNorm + Bias + RoPE for B and C ═══
    // b_param and c_param are the SAME for all heads (G=1, bg repeats)
    // but angles differ per head, so each threadgroup applies its own RoPE.

    // Step 1a: Compute sum-of-squares for RMSNorm (shared across heads since G=1)
    float b_ss = 0.0f, c_ss = 0.0f;

    for (uint i = lid; i < N_R; i += P_VAL) {
        float bv = float(b_raw[i]);
        float cv = float(c_raw[i]);
        b_ss += bv * bv;
        c_ss += cv * cv;
    }

    // Threadgroup reduce for RMS (sum of squares across all N*R values)
    threadgroup float rms_buf[64];
    rms_buf[lid] = b_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 32; s > 0; s >>= 1) {
        if (lid < s) rms_buf[lid] += rms_buf[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float b_inv_rms = metal::rsqrt(rms_buf[0] / float(N_R) + 1e-5f);

    // Barrier: all threads must read rms_buf[0] before any thread overwrites it
    threadgroup_barrier(mem_flags::mem_threadgroup);

    rms_buf[lid] = c_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 32; s > 0; s >>= 1) {
        if (lid < s) rms_buf[lid] += rms_buf[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float c_inv_rms = metal::rsqrt(rms_buf[0] / float(N_R) + 1e-5f);

    // Step 1b: Apply norm * weight + bias, then store in SRAM
    for (uint i = lid; i < N_R; i += P_VAL) {
        float bv = float(b_raw[i]) * b_inv_rms * float(norm_b_w[i]) + float(bias_b[i]);
        float cv = float(c_raw[i]) * c_inv_rms * float(norm_c_w[i]) + float(bias_c[i]);
        b_sram[i] = bv;
        c_sram[i] = cv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1c: RoPE rotation in-place on SRAM
    // angles[h, n_half] for this head
    // RoPE pairs: (n_even, n_odd) rows with sin/cos rotation over R
    uint n_half = N_VAL / 2;
    for (uint nh = lid; nh < n_half; nh += P_VAL) {
        float angle = float(prev_angle[h * n_half + nh])
                    + float(dt_arr[h]) * float(theta[h * n_half + nh]);

        float sin_a = metal::sin(angle);
        float cos_a = metal::cos(angle);

        uint even_base = (nh * 2) * R_VAL;
        uint odd_base  = (nh * 2 + 1) * R_VAL;
        for (uint r = 0; r < R_VAL; ++r) {
            float b1 = b_sram[even_base + r];
            float b2 = b_sram[odd_base + r];
            b_sram[even_base + r] = b1 * cos_a - b2 * sin_a;
            b_sram[odd_base + r]  = b2 * cos_a + b1 * sin_a;

            float c1 = c_sram[even_base + r];
            float c2 = c_sram[odd_base + r];
            c_sram[even_base + r] = c1 * cos_a - c2 * sin_a;
            c_sram[odd_base + r]  = c2 * cos_a + c1 * sin_a;
        }

        // Write updated angle
        new_angle[h * n_half + nh] = T(angle);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══ Phase 2: Fused Einsum1 + Lambda + SSM + Einsum2 (same as V1) ═══

    uint x_base = h * P_R + p * R_VAL;
    float xr0 = float(x_ssm[x_base + 0]);
    float xr1 = float(x_ssm[x_base + 1]);
    float xr2 = float(x_ssm[x_base + 2]);
    float xr3 = float(x_ssm[x_base + 3]);

    float dt  = float(dt_arr[h]);
    float a   = float(a_arr[h]);
    float lam = float(lv_arr[h]);

    float alpha = metal::exp(dt * a);
    alpha = alpha > 1e6f ? 1e6f : (alpha < -1e6f ? -1e6f : alpha);

    float coeff_inp  = dt * lam;
    float coeff_prev = dt * (1.0f - lam) * alpha;

    float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
    uint state_base = h * N_P + p;

    for (uint n = 0; n < N_VAL; ++n) {
        uint bn = n * R_VAL;

        float inp = b_sram[bn]     * xr0
                  + b_sram[bn + 1] * xr1
                  + b_sram[bn + 2] * xr2
                  + b_sram[bn + 3] * xr3;

        uint si = state_base + n * P_VAL;
        float h_val = float(prev_h[si]) * alpha + coeff_inp * inp + coeff_prev * float(prev_in[si]);

        new_h[si]       = T(h_val);
        new_prev_in[si] = T(inp);

        y0 += h_val * c_sram[bn];
        y1 += h_val * c_sram[bn + 1];
        y2 += h_val * c_sram[bn + 2];
        y3 += h_val * c_sram[bn + 3];
    }

    uint yb = h * P_R + p * R_VAL;
    y_out[yb + 0] = T(y0);
    y_out[yb + 1] = T(y1);
    y_out[yb + 2] = T(y2);
    y_out[yb + 3] = T(y3);
"""


def _make_header(h: int, n: int, p: int, r: int) -> str:
    return (
        f"constant uint H_VAL = {h};\n"
        f"constant uint N_VAL = {n};\n"
        f"constant uint P_VAL = {p};\n"
        f"constant uint R_VAL = {r};\n"
        f"constant uint N_R   = {n * r};\n"
        f"constant uint N_P   = {n * p};\n"
        f"constant uint P_R   = {p * r};\n"
    )


def build_fused_kernel(h: int, n: int, p: int, r: int):
    """Build V1: fused Einsum1 + Lambda + SSM + Einsum2."""
    return mx.fast.metal_kernel(
        name=f"fused_mamba_mixer_v1_h{h}_n{n}_p{p}_r{r}",
        input_names=[
            "b_rot", "c_rot", "x_ssm",
            "prev_h", "prev_in",
            "dt_arr", "a_arr", "lv_arr",
        ],
        output_names=["new_h", "new_prev_in", "y_out"],
        source=_FUSED_MAMBA_MIXER_V1_SRC,
        header=_make_header(h, n, p, r),
    )


def build_fused_kernel_v2(h: int, n: int, p: int, r: int):
    """Build V2: fused Norm + RoPE + Einsum1 + Lambda + SSM + Einsum2."""
    return mx.fast.metal_kernel(
        name=f"fused_mamba_mixer_v2_h{h}_n{n}_p{p}_r{r}",
        input_names=[
            "b_raw", "c_raw",
            "norm_b_w", "norm_c_w",
            "bias_b", "bias_c",
            "theta", "prev_angle",
            "x_ssm",
            "prev_h", "prev_in",
            "dt_arr", "a_arr", "lv_arr",
        ],
        output_names=["new_h", "new_prev_in", "y_out", "new_angle"],
        source=_FUSED_MAMBA_MIXER_V2_SRC,
        header=_make_header(h, n, p, r),
    )


def run_fused_kernel(
    kernel,
    b_rot: mx.array,
    c_rot: mx.array,
    x_ssm: mx.array,
    prev_h: mx.array,
    prev_input: mx.array,
    dt_b: mx.array,
    a_b: mx.array,
    lv: mx.array,
    *,
    h: int, n: int, p: int, r: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run V1 fused kernel. Returns (new_h, new_input, y)."""
    dtype = b_rot.dtype
    new_h, new_inp, y = kernel(
        inputs=[
            mx.flatten(b_rot),
            mx.flatten(c_rot),
            mx.flatten(x_ssm),
            mx.flatten(prev_h),
            mx.flatten(prev_input),
            dt_b, a_b, lv,
        ],
        template=[("T", dtype)],
        grid=(p, h, 1),
        threadgroup=(p, 1, 1),
        output_shapes=[(h, n, p), (h, n, p), (h, p, r)],
        output_dtypes=[dtype, dtype, dtype],
    )
    return new_h, new_inp, y


# ═══════════════════════════════════════════════════════════════════════════════
# V2 Reference + Runner (Full Norm + RoPE + Einsum + SSM)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_rope_ref(x: mx.array, angles: mx.array) -> mx.array:
    """Reference RoPE matching mlx_hybrid_infer.apply_rope. x: (H,N,R), angles: (H,N//2)."""
    n_half = angles.shape[-1]
    x_reshaped = x.reshape(H, n_half, 2, R)
    x1 = x_reshaped[:, :, 0, :]
    x2 = x_reshaped[:, :, 1, :]
    sin_a = mx.expand_dims(mx.sin(angles), -1)
    cos_a = mx.expand_dims(mx.cos(angles), -1)
    out = mx.stack([x1 * cos_a - x2 * sin_a, x2 * cos_a + x1 * sin_a], axis=2)
    return out.reshape(H, N, R)


def reference_mlx_v2(
    b_raw: mx.array,       # (N*R,) raw from in_proj
    c_raw: mx.array,       # (N*R,)
    norm_b_w: mx.array,    # (N*R,) RMSNorm weight
    norm_c_w: mx.array,    # (N*R,)
    bias_b: mx.array,      # (N*R,) flattened bias
    bias_c: mx.array,      # (N*R,)
    theta_rep: mx.array,   # (H, N//2) per-head theta
    prev_angle: mx.array,  # (H, N//2)
    x_ssm: mx.array,       # (H, P, R)
    prev_h: mx.array,      # (H, N, P)
    prev_input: mx.array,  # (H, N, P)
    dt_b: mx.array,        # (H,)
    a_b: mx.array,         # (H,)
    lv: mx.array,          # (H,)
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Full reference: Norm + RoPE + Einsum1 + Lambda + SSM + Einsum2."""
    n_half = N // 2
    eps = 1e-5

    # RMSNorm B
    b_sq = mx.sum(b_raw * b_raw)
    b_inv_rms = mx.rsqrt(b_sq / (N * R) + eps)
    b_normed = b_raw * b_inv_rms * norm_b_w

    # RMSNorm C
    c_sq = mx.sum(c_raw * c_raw)
    c_inv_rms = mx.rsqrt(c_sq / (N * R) + eps)
    c_normed = c_raw * c_inv_rms * norm_c_w

    # Add bias and expand groups (G=1 → H=24 via broadcast in einsum)
    b_biased = (b_normed + bias_b).reshape(N, R)
    c_biased = (c_normed + bias_c).reshape(N, R)

    # Expand to all heads (same data, just broadcast) → (H, N, R)
    b_expanded = mx.broadcast_to(b_biased[None, :, :], (H, N, R))
    c_expanded = mx.broadcast_to(c_biased[None, :, :], (H, N, R))

    # Compute new angles: prev + dt * theta
    new_angles = prev_angle + dt_b[:, None] * theta_rep

    # Apply RoPE
    b_rot = apply_rope_ref(b_expanded, new_angles)
    c_rot = apply_rope_ref(c_expanded, new_angles)

    # Einsum1 + Lambda + SSM + Einsum2
    h_new, inp_sig, y = reference_mlx(b_rot, c_rot, x_ssm, prev_h, prev_input, dt_b, a_b, lv)

    return h_new, inp_sig, y, new_angles


def run_fused_kernel_v2(
    kernel,
    b_raw, c_raw, norm_b_w, norm_c_w, bias_b, bias_c,
    theta_rep, prev_angle, x_ssm, prev_h, prev_input,
    dt_b, a_b, lv,
    *, h: int, n: int, p: int, r: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Run V2 fused kernel. Returns (new_h, new_input, y, new_angle)."""
    dtype = x_ssm.dtype
    n_half = n // 2
    new_h, new_inp, y, new_ang = kernel(
        inputs=[
            b_raw, c_raw,
            norm_b_w, norm_c_w,
            bias_b, bias_c,
            mx.flatten(theta_rep), mx.flatten(prev_angle),
            mx.flatten(x_ssm),
            mx.flatten(prev_h), mx.flatten(prev_input),
            dt_b, a_b, lv,
        ],
        template=[("T", dtype)],
        grid=(p, h, 1),
        threadgroup=(p, 1, 1),
        output_shapes=[(h, n, p), (h, n, p), (h, p, r), (h, n_half)],
        output_dtypes=[dtype, dtype, dtype, dtype],
    )
    return new_h, new_inp, y, new_ang


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def generate_test_data_v1(dtype=mx.bfloat16):
    """Generate random tensors for V1 (post-RoPE inputs)."""
    mx.random.seed(42)
    b_rot      = (mx.random.normal((H, N, R)) * 0.5).astype(dtype)
    c_rot      = (mx.random.normal((H, N, R)) * 0.5).astype(dtype)
    x_ssm      = (mx.random.normal((H, P, R)) * 0.5).astype(dtype)
    prev_h     = (mx.random.normal((H, N, P)) * 0.1).astype(dtype)
    prev_input = (mx.random.normal((H, N, P)) * 0.1).astype(dtype)
    dt_b       = (mx.abs(mx.random.normal((H,))) * 0.1 + 0.01).astype(dtype)
    a_b        = (-mx.abs(mx.random.normal((H,))) * 0.5).astype(dtype)
    lv         = mx.sigmoid(mx.random.normal((H,))).astype(dtype)
    mx.eval(b_rot, c_rot, x_ssm, prev_h, prev_input, dt_b, a_b, lv)
    return b_rot, c_rot, x_ssm, prev_h, prev_input, dt_b, a_b, lv


def generate_test_data_v2(dtype=mx.bfloat16):
    """Generate random tensors for V2 (pre-Norm inputs)."""
    mx.random.seed(42)
    n_half = N // 2
    b_raw      = (mx.random.normal((N * R,)) * 0.5).astype(dtype)
    c_raw      = (mx.random.normal((N * R,)) * 0.5).astype(dtype)
    norm_b_w   = mx.ones((N * R,), dtype=dtype)
    norm_c_w   = mx.ones((N * R,), dtype=dtype)
    bias_b     = (mx.random.normal((N * R,)) * 0.01).astype(dtype)
    bias_c     = (mx.random.normal((N * R,)) * 0.01).astype(dtype)
    theta_rep  = (mx.random.normal((H, n_half)) * 0.1).astype(dtype)
    prev_angle = (mx.random.normal((H, n_half)) * 0.5).astype(dtype)
    x_ssm      = (mx.random.normal((H, P, R)) * 0.5).astype(dtype)
    prev_h     = (mx.random.normal((H, N, P)) * 0.1).astype(dtype)
    prev_input = (mx.random.normal((H, N, P)) * 0.1).astype(dtype)
    dt_b       = (mx.abs(mx.random.normal((H,))) * 0.1 + 0.01).astype(dtype)
    a_b        = (-mx.abs(mx.random.normal((H,))) * 0.5).astype(dtype)
    lv         = mx.sigmoid(mx.random.normal((H,))).astype(dtype)
    all_tensors = (b_raw, c_raw, norm_b_w, norm_c_w, bias_b, bias_c,
                   theta_rep, prev_angle, x_ssm, prev_h, prev_input, dt_b, a_b, lv)
    mx.eval(*all_tensors)
    return all_tensors


def accuracy_check(
    ref_h: mx.array, ref_inp: mx.array, ref_y: mx.array,
    fused_h: mx.array, fused_inp: mx.array, fused_y: mx.array,
) -> dict[str, float]:
    """Compare reference vs fused outputs."""
    def _err(a, b, label):
        diff = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
        mx.eval(diff)
        mean_e = float(mx.mean(diff).item())
        max_e = float(mx.max(diff).item())
        return {f"{label}_mean": mean_e, f"{label}_max": max_e}

    results = {}
    results.update(_err(ref_h, fused_h, "h_state"))
    results.update(_err(ref_inp, fused_inp, "input_signal"))
    results.update(_err(ref_y, fused_y, "y_output"))
    return results


def time_fn(fn, warmup: int, trials: int) -> float:
    """Time a function; returns median ms per call."""
    for _ in range(warmup):
        outs = fn()
        if isinstance(outs, tuple):
            mx.eval(*outs)
        else:
            mx.eval(outs)

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        outs = fn()
        if isinstance(outs, tuple):
            mx.eval(*outs)
        else:
            mx.eval(outs)
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    n = len(times)
    return times[n // 2]


def _print_accuracy(errs: dict[str, float]) -> bool:
    print("─" * 50)
    print("Accuracy Report")
    print("─" * 50)
    all_pass = True
    for k, v in sorted(errs.items()):
        threshold = 0.05
        status = "PASS" if v < threshold else "FAIL"
        if v >= threshold:
            all_pass = False
        print(f"  {k:25s}  {v:.6f}  [{status}]")
    print("─" * 50)
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print()
    return all_pass


def _print_perf(name: str, ref_ms: float, fused_ms: float, current_ms: float = 17.0):
    speedup = ref_ms / max(fused_ms, 1e-9)
    saved_per_layer = ref_ms - fused_ms
    total_saved = saved_per_layer * 24
    projected_ms = current_ms - total_saved
    projected_tps = 1000.0 / max(projected_ms, 0.1)

    print(f"  [{name}]")
    print(f"  Reference MLX (median):  {ref_ms:.3f} ms")
    print(f"  Fused Metal  (median):   {fused_ms:.3f} ms")
    print(f"  Speedup:                 {speedup:.2f}×")
    print(f"  Savings per layer:       {saved_per_layer:.3f} ms")
    print(f"  Total savings (×24):     {total_saved:.2f} ms")
    print(f"  Projected:               ~{projected_ms:.1f} ms/tok (~{projected_tps:.0f} tok/s)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Fused Mamba Mixer kernel benchmark")
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--v2", action="store_true", help="Also benchmark V2 (Norm+RoPE fused)")
    args = parser.parse_args()

    dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float32
    print("=" * 70)
    print("Fused Mamba Mixer Kernel Benchmark")
    print(f"  H={H}  N={N}  P={P}  R={R}  dtype={args.dtype}")
    print(f"  warmup={args.warmup}  trials={args.trials}")
    print(f"  State per head: N×P = {N*P} = {N*P*2//1024:.1f} KB (bf16)")
    print(f"  Total state (24 heads): {H*N*P*2//1024:.0f} KB")
    print(f"  Intermediate eliminated: ~{3*H*N*P*2//1024:.0f} KB per token")
    print("=" * 70)

    # ═══════════════════════ V1 Benchmark ═══════════════════════
    print("\n>>> V1: Fused Einsum1 + Lambda + SSM + Einsum2 <<<\n")
    data_v1 = generate_test_data_v1(dtype)

    print("[V1 1/3] Reference MLX...")
    ref_h, ref_inp, ref_y = reference_mlx(*data_v1)
    mx.eval(ref_h, ref_inp, ref_y)

    print("[V1 2/3] Fused Metal kernel...")
    kernel_v1 = build_fused_kernel(H, N, P, R)
    fused_h, fused_inp, fused_y = run_fused_kernel(kernel_v1, *data_v1, h=H, n=N, p=P, r=R)
    mx.eval(fused_h, fused_inp, fused_y)

    print("[V1 3/3] Accuracy check...")
    errs_v1 = accuracy_check(ref_h, ref_inp, ref_y, fused_h, fused_inp, fused_y)
    v1_pass = _print_accuracy(errs_v1)

    if v1_pass:
        ref_fn_v1 = lambda: reference_mlx(*data_v1)
        fused_fn_v1 = lambda: run_fused_kernel(kernel_v1, *data_v1, h=H, n=N, p=P, r=R)
        ref_ms_v1 = time_fn(ref_fn_v1, args.warmup, args.trials)
        fused_ms_v1 = time_fn(fused_fn_v1, args.warmup, args.trials)
        _print_perf("V1: Einsum+Lambda+SSM", ref_ms_v1, fused_ms_v1)

    # ═══════════════════════ V2 Benchmark ═══════════════════════
    if args.v2:
        print("\n>>> V2: Fused Norm + RoPE + Einsum1 + Lambda + SSM + Einsum2 <<<\n")
        data_v2 = generate_test_data_v2(dtype)

        print("[V2 1/3] Reference MLX (Norm + RoPE + Einsum + SSM)...")
        ref_h2, ref_inp2, ref_y2, ref_ang2 = reference_mlx_v2(*data_v2)
        mx.eval(ref_h2, ref_inp2, ref_y2, ref_ang2)

        print("[V2 2/3] Fused Metal V2 kernel...")
        kernel_v2 = build_fused_kernel_v2(H, N, P, R)
        fused_h2, fused_inp2, fused_y2, fused_ang2 = run_fused_kernel_v2(
            kernel_v2, *data_v2, h=H, n=N, p=P, r=R
        )
        mx.eval(fused_h2, fused_inp2, fused_y2, fused_ang2)

        print("[V2 3/3] Accuracy check...")
        errs_v2 = accuracy_check(ref_h2, ref_inp2, ref_y2, fused_h2, fused_inp2, fused_y2)
        ang_diff = mx.abs(ref_ang2.astype(mx.float32) - fused_ang2.astype(mx.float32))
        mx.eval(ang_diff)
        errs_v2["angle_mean"] = float(mx.mean(ang_diff).item())
        errs_v2["angle_max"] = float(mx.max(ang_diff).item())
        v2_pass = _print_accuracy(errs_v2)

        if v2_pass:
            ref_fn_v2 = lambda: reference_mlx_v2(*data_v2)
            fused_fn_v2 = lambda: run_fused_kernel_v2(kernel_v2, *data_v2, h=H, n=N, p=P, r=R)
            ref_ms_v2 = time_fn(ref_fn_v2, args.warmup, args.trials)
            fused_ms_v2 = time_fn(fused_fn_v2, args.warmup, args.trials)
            _print_perf("V2: Norm+RoPE+Einsum+SSM", ref_ms_v2, fused_ms_v2)

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
