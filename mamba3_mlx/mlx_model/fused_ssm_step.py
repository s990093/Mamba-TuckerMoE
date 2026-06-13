"""Fused Metal kernel for the Mamba L=1 decode inner chain (G == 1).

Replaces ~20 small MLX kernels per Mamba block per token:

    softplus/exp/sigmoid scalars → RoPE angles → sin/cos → rotate B/C
    → input_signal einsum → lv/dv/av blend → h recurrence → y einsum

with ONE custom kernel launch.  The surrounding matmuls (in_proj, TuckerMoE,
y_down_proj, dense_proj) and norms stay on MLX's tuned kernels.

Numerics: every binary op rounds to bfloat16 at exactly the boundaries the
unfused MLX op sequence does (intermediate products/sums are cast to bf16
before the next op); reductions accumulate in float32 and round once, like
mx.einsum.  Transcendentals use metal::precise.  Verified token-exact against
the reference decode path (see bench_static.py --verify with --metal-fuse).

Layout assumptions (asserted in the wrapper):
    B == 1, L == 1, G == 1  →  theta/angles identical across heads, so the
    rotated B/C are shared by all heads and computed per (n, r) only.

Grid: (N, P, H) threads, threadgroup (N, 1, 1) — one threadgroup per (p, h),
the n-dimension lives inside the group so y[h, p, :] can be reduced over n
in threadgroup memory (serial, ascending n — deterministic).
"""

from __future__ import annotations

import mlx.core as mx

_KERNEL_CACHE: dict[tuple, object] = {}


def _build_source(H: int, N: int, P: int, R: int,
                  norm_eps: float | None = None) -> str:
    NR = N * R
    PR = P * R
    K2 = N // 2

    if norm_eps is None:
        norm_prologue = ""
        read = {"b1": f"Bp[b * {NR} + (2u * k)      * R + r]",
                "b2": f"Bp[b * {NR} + (2u * k + 1u) * R + r]",
                "c1": f"Cp[b * {NR} + (2u * k)      * R + r]",
                "c2": f"Cp[b * {NR} + (2u * k + 1u) * R + r]"}
    else:
        # norm-fold: replicate MLX rms_single_row (N_READS=4, 64-thread tg,
        # simd_sum tree — verified 4000/4000 bitwise vs mx.fast.rms_norm)
        # for B_raw and C_raw, then norm+bias inline at each read:
        #   bf16( f32(w[i] * bf16(f32(x[i]) * inv)) + f32(bias[i]) )
        assert NR == 4 * N, "norm-fold assumes N_READS=4 with one tg of N threads"
        norm_prologue = f"""
    const uint lid_  = thread_position_in_threadgroup.x;     // == n
    const uint slane = thread_index_in_simdgroup;
    const uint sgid  = simdgroup_index_in_threadgroup;
    threadgroup float tg_red[32];
    threadgroup float tg_inv[2];

    float accr = 0.0f;
    for (uint i = 0; i < 4; ++i) {{
        float xi = static_cast<float>(B_raw[b * {NR} + lid_ * 4u + i]);
        accr += xi * xi;
    }}
    accr = metal::simd_sum(accr);
    if (sgid == 0u) {{ tg_red[slane] = 0.0f; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slane == 0u) {{ tg_red[sgid] = accr; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {{
        float a2 = metal::simd_sum(tg_red[slane]);
        if (slane == 0u) {{
            tg_inv[0] = metal::precise::rsqrt(a2 / {float(NR)}f + {norm_eps:.9e}f);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    accr = 0.0f;
    for (uint i = 0; i < 4; ++i) {{
        float xi = static_cast<float>(C_raw[b * {NR} + lid_ * 4u + i]);
        accr += xi * xi;
    }}
    accr = metal::simd_sum(accr);
    if (sgid == 0u) {{ tg_red[slane] = 0.0f; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slane == 0u) {{ tg_red[sgid] = accr; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {{
        float a2 = metal::simd_sum(tg_red[slane]);
        if (slane == 0u) {{
            tg_inv[1] = metal::precise::rsqrt(a2 / {float(NR)}f + {norm_eps:.9e}f);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invB = tg_inv[0];
    const float invC = tg_inv[1];
"""

        def _nrm(raw, w, b, idx):
            return (f"static_cast<bfloat16_t>(static_cast<float>({w}[{idx}] * "
                    f"static_cast<bfloat16_t>(static_cast<float>({raw}[b * "
                    f"{NR} + {idx}]) * inv{raw[0]})) "
                    f"+ static_cast<float>({b}[{idx}]))")

        read = {"b1": _nrm("B_raw", "wB", "bB", "(2u * k)      * R + r"),
                "b2": _nrm("B_raw", "wB", "bB", "(2u * k + 1u) * R + r"),
                "c1": _nrm("C_raw", "wC", "bC", "(2u * k)      * R + r"),
                "c2": _nrm("C_raw", "wC", "bC", "(2u * k + 1u) * R + r")}

    return f"""
    constexpr uint H = {H};
    constexpr uint N = {N};
    constexpr uint P = {P};
    constexpr uint R = {R};
    constexpr uint K2 = {K2};

    const uint n = thread_position_in_grid.x;     // 0..N-1 (whole threadgroup)
    const uint p = thread_position_in_grid.y;     // 0..P-1
    const uint zz = thread_position_in_grid.z;    // 0..B*H-1
    const uint h = zz % H;
    const uint b = zz / H;
{norm_prologue}
    // ── scalar chain — formulas copied from MLX's Metal kernels so every
    //    bf16 rounding matches bit-for-bit (verified over the full bf16
    //    domain: softplus/sigmoid/exp all 0 mismatches excl. NaN payloads) ──
    const bfloat16_t zero_b = static_cast<bfloat16_t>(0.0f);
    const bfloat16_t one_b  = static_cast<bfloat16_t>(1.0f);
    // dt = softplus(dt_p) = LogAddExp(dt_p, 0): MLX binary_ops.h LogAddExp
    bfloat16_t xb   = dt_p[b];
    bfloat16_t maxv = metal::max(xb, zero_b);
    bfloat16_t minv = metal::min(xb, zero_b);
    bfloat16_t dt_b = maxv + log1p(metal::exp(minv - maxv));
    // A = -exp(A_p): MLX unary Exp (precise) on bfloat, then exact negate
    bfloat16_t eA  = metal::precise::exp(A_p[b]);
    bfloat16_t A_b = static_cast<bfloat16_t>(-static_cast<float>(eA));
    // la = float(bf16(dt * A)); av = bf16(exp(la))  [f32 exp — AR cast order]
    bfloat16_t dtA = static_cast<bfloat16_t>(static_cast<float>(dt_b) * static_cast<float>(A_b));
    bfloat16_t av  = static_cast<bfloat16_t>(metal::precise::exp(static_cast<float>(dtA)));
    // lv = sigmoid(lam): MLX unary_ops.h Sigmoid (stable form)
    bfloat16_t lb = lam[b];
    bfloat16_t y0 = one_b / (one_b + metal::exp(metal::abs(lb)));
    bfloat16_t lv = (lb < zero_b) ? y0 : one_b - y0;
    bfloat16_t dv = dt_b;
    bfloat16_t one_minus_lv = static_cast<bfloat16_t>(1.0f - static_cast<float>(lv));

    // ── RoPE angle for pair k = n/2 (f32 accumulate, bf16 for sin/cos) ──────
    const uint k = n >> 1;
    float delta = static_cast<float>(dt_b) * theta[k];          // dt.astype(f32) * theta_h
    float ac    = delta + ac_prev[(b * H + h) * K2 + k];                  // f32 accumulate
    bfloat16_t ang = static_cast<bfloat16_t>(ac);               // angles.astype(bf16)
    bfloat16_t sin_a = static_cast<bfloat16_t>(metal::precise::sin(static_cast<float>(ang)));
    bfloat16_t cos_a = static_cast<bfloat16_t>(metal::precise::cos(static_cast<float>(ang)));

    // ── rotate B and C rows (shared across heads when G == 1) ───────────────
    // x_r[k, j, r] = orig[2k + j, r];  j = n & 1
    //   j==0: x1*cos - x2*sin      j==1: x2*cos + x1*sin
    const uint j = n & 1u;
    bfloat16_t Brot[R];
    bfloat16_t Crot[R];
    for (uint r = 0; r < R; ++r) {{
        bfloat16_t b1 = {read["b1"]};
        bfloat16_t b2 = {read["b2"]};
        bfloat16_t c1 = {read["c1"]};
        bfloat16_t c2 = {read["c2"]};
        bfloat16_t bm1 = static_cast<bfloat16_t>(static_cast<float>(b1) * static_cast<float>(cos_a));
        bfloat16_t bm2 = static_cast<bfloat16_t>(static_cast<float>(b2) * static_cast<float>(sin_a));
        bfloat16_t bm3 = static_cast<bfloat16_t>(static_cast<float>(b2) * static_cast<float>(cos_a));
        bfloat16_t bm4 = static_cast<bfloat16_t>(static_cast<float>(b1) * static_cast<float>(sin_a));
        bfloat16_t cm1 = static_cast<bfloat16_t>(static_cast<float>(c1) * static_cast<float>(cos_a));
        bfloat16_t cm2 = static_cast<bfloat16_t>(static_cast<float>(c2) * static_cast<float>(sin_a));
        bfloat16_t cm3 = static_cast<bfloat16_t>(static_cast<float>(c2) * static_cast<float>(cos_a));
        bfloat16_t cm4 = static_cast<bfloat16_t>(static_cast<float>(c1) * static_cast<float>(sin_a));
        Brot[r] = (j == 0u)
            ? static_cast<bfloat16_t>(static_cast<float>(bm1) - static_cast<float>(bm2))
            : static_cast<bfloat16_t>(static_cast<float>(bm3) + static_cast<float>(bm4));
        Crot[r] = (j == 0u)
            ? static_cast<bfloat16_t>(static_cast<float>(cm1) - static_cast<float>(cm2))
            : static_cast<bfloat16_t>(static_cast<float>(cm3) + static_cast<float>(cm4));
    }}

    // ── input_signal[h,n,p] = sum_r Brot[n,r] * x_ssm[h,p,r]  (f32 acc) ─────
    float acc = 0.0f;
    for (uint r = 0; r < R; ++r) {{
        acc += static_cast<float>(Brot[r])
             * static_cast<float>(x_ssm[(b * H + h) * {PR} + p * R + r]);
    }}
    bfloat16_t is_ = static_cast<bfloat16_t>(acc);

    // ── u_ssm = lv*dv*is + (1-lv)*dv*av*ip  (bf16 at every binary op) ───────
    const uint idx = ((b * H + h) * N + n) * P + p;
    bfloat16_t t1 = static_cast<bfloat16_t>(static_cast<float>(lv) * static_cast<float>(dv));
    t1 = static_cast<bfloat16_t>(static_cast<float>(t1) * static_cast<float>(is_));
    bfloat16_t t2 = static_cast<bfloat16_t>(static_cast<float>(one_minus_lv) * static_cast<float>(dv));
    t2 = static_cast<bfloat16_t>(static_cast<float>(t2) * static_cast<float>(av));
    t2 = static_cast<bfloat16_t>(static_cast<float>(t2) * static_cast<float>(ip[idx]));
    bfloat16_t u = static_cast<bfloat16_t>(static_cast<float>(t1) + static_cast<float>(t2));

    // ── h recurrence: h_new = bf16(bf16(av*h_prev) + u) ─────────────────────
    bfloat16_t ah = static_cast<bfloat16_t>(static_cast<float>(av) * static_cast<float>(h_prev[idx]));
    bfloat16_t hn = static_cast<bfloat16_t>(static_cast<float>(ah) + static_cast<float>(u));

    h_new[idx]  = hn;
    new_ip[idx] = is_;
    if (p == 0u && j == 0u) {{
        new_ac[(b * H + h) * K2 + k] = ac;
    }}

    // ── y[h,p,r] = sum_n h_new[h,n,p] * Crot[n,r]  (reduce over the group) ──
    threadgroup float sh[N * R];
    for (uint r = 0; r < R; ++r) {{
        sh[n * R + r] = static_cast<float>(hn) * static_cast<float>(Crot[r]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (n == 0u) {{
        for (uint r = 0; r < R; ++r) {{
            float s = 0.0f;
            for (uint nn = 0; nn < N; ++nn) {{
                s += sh[nn * R + r];
            }}
            y[(b * H + h) * {PR} + p * R + r] = static_cast<bfloat16_t>(s);
        }}
    }}
"""


def get_fused_ssm_kernel(H: int, N: int, P: int, R: int):
    key = (H, N, P, R)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    kernel = mx.fast.metal_kernel(
        name=f"mamba_ssm_decode_h{H}n{N}p{P}r{R}",
        input_names=["dt_p", "A_p", "lam", "theta", "ac_prev",
                     "Bp", "Cp", "x_ssm", "h_prev", "ip"],
        output_names=["h_new", "new_ip", "y", "new_ac"],
        source=_build_source(H, N, P, R),
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def get_fused_ssm_norm_kernel(H: int, N: int, P: int, R: int, eps: float):
    """SSM kernel variant with norm_B/norm_C (+bias) folded in: takes RAW
    B/C in_proj slices and computes the RMS norms in-kernel (replicating
    MLX's rms_single_row reduction tree bit-for-bit), removing 2 rms_norm
    + 2 bias-add launches per Mamba block.  Requires R == 4 (N_READS)."""
    key = ("ssm_nf", H, N, P, R, eps)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    kernel = mx.fast.metal_kernel(
        name=f"mamba_ssm_decode_nf_h{H}n{N}p{P}r{R}",
        input_names=["dt_p", "A_p", "lam", "theta", "ac_prev",
                     "B_raw", "C_raw", "wB", "wC", "bB", "bC",
                     "x_ssm", "h_prev", "ip"],
        output_names=["h_new", "new_ip", "y", "new_ac"],
        source=_build_source(H, N, P, R, norm_eps=eps),
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def _build_tucker_source(E: int, TOPK: int, R3: int, R2: int,
                         router_scale: float, temperature: float) -> str:
    assert TOPK == 2, "kernel implements top-2 selection"
    return f"""
    constexpr uint E  = {E};
    constexpr uint R3 = {R3};
    constexpr uint R2 = {R2};

    const uint s = thread_position_in_grid.x;          // 0..R2-1

    // ── router chain in f32, op-for-op the mx sequence ──────────────────────
    // capped = scaled_tanh(logits.astype(f32), scale); rl = capped / temp
    float rl[E];
    for (uint e = 0; e < E; ++e) {{
        float t  = static_cast<float>(logits_in[e]) * {1.0 / router_scale}f;
        float u2 = 2.0f * t;
        float y0 = 1.0f / (1.0f + metal::exp(metal::abs(u2)));   // MLX Sigmoid f32
        float sg = (u2 < 0.0f) ? y0 : 1.0f - y0;
        float th = 2.0f * sg - 1.0f;                              // tanh_approx
        rl[e] = (th * {router_scale}f) / {temperature}f;
    }}

    // top-2 selection on rl (ties → lower index, matching argpartition output)
    uint e0 = 0;
    for (uint e = 1; e < E; ++e) {{
        if (rl[e] > rl[e0]) {{ e0 = e; }}
    }}
    uint e1 = (e0 == 0) ? 1 : 0;
    for (uint e = 0; e < E; ++e) {{
        if (e != e0 && rl[e] > rl[e1]) {{ e1 = e; }}
    }}

    // softmax over rl (f32) → top-2 raw probs → normalized → bf16 weights
    float m = rl[0];
    for (uint e = 1; e < E; ++e) {{
        m = metal::max(m, rl[e]);
    }}
    float den = 0.0f;
    for (uint e = 0; e < E; ++e) {{
        den += metal::exp(rl[e] - m);
    }}
    float p0r = metal::exp(rl[e0] - m) / den;
    float p1r = metal::exp(rl[e1] - m) / den;
    float psum = p0r + p1r + 1e-6f;
    bfloat16_t p0 = static_cast<bfloat16_t>(p0r / psum);
    bfloat16_t p1 = static_cast<bfloat16_t>(p1r / psum);

    // ── x_core[s] = sum_r x_shared[r] * bf16(bf16(G[e0,r,s]*p0) + bf16(G[e1,r,s]*p1)) ──
    const ulong base0 = (ulong)e0 * R3 * R2;
    const ulong base1 = (ulong)e1 * R3 * R2;
    float acc = 0.0f;
    for (uint r = 0; r < R3; ++r) {{
        bfloat16_t g0 = static_cast<bfloat16_t>(
            static_cast<float>(G[base0 + (ulong)r * R2 + s]) * static_cast<float>(p0));
        bfloat16_t g1 = static_cast<bfloat16_t>(
            static_cast<float>(G[base1 + (ulong)r * R2 + s]) * static_cast<float>(p1));
        bfloat16_t gw = static_cast<bfloat16_t>(
            static_cast<float>(g0) + static_cast<float>(g1));
        acc += static_cast<float>(x_shared[r]) * static_cast<float>(gw);
    }}
    x_core[s] = static_cast<bfloat16_t>(acc);
"""


def _build_tucker_gw_source(E: int, R3: int, R2: int,
                            router_scale: float, temperature: float) -> str:
    """Router chain + top-2 weighted expert blend → G_w, one launch.

    Thread (s, r) computes the 8-wide routing redundantly (trivial compute)
    and writes G_w[r, s] = bf16(G[e0,r,s]*p0) + bf16(G[e1,r,s]*p1).  The
    softmax replicates MLX softmax.h exactly: fast::exp, serial 4-chunk
    partial sums (N_READS=4) combined pairwise; index order is irrelevant
    because the rounded products are combined with an exact commutative add.
    """
    assert E == 8, "softmax chunk replication assumes axis_size == 8"
    return f"""
    constexpr uint E  = {E};
    constexpr uint R3 = {R3};
    constexpr uint R2 = {R2};

    const uint s = thread_position_in_grid.x;          // 0..R2-1
    const uint r = thread_position_in_grid.y;          // 0..R3-1
    const uint b = thread_position_in_grid.z;          // 0..B-1

    // capped = scaled_tanh(logits.astype(f32), scale); rl = capped / temp
    float rl[E];
    for (uint e = 0; e < E; ++e) {{
        float t  = static_cast<float>(logits_in[b * E + e]) * {1.0 / router_scale}f;
        float u2 = 2.0f * t;
        float y0 = 1.0f / (1.0f + metal::exp(metal::abs(u2)));   // MLX Sigmoid f32
        float sg = (u2 < 0.0f) ? y0 : 1.0f - y0;
        float th = 2.0f * sg - 1.0f;
        rl[e] = (th * {router_scale}f) / {temperature}f;
    }}

    // top-2 (ties → lower index; order does not affect the blended output)
    uint e0 = 0;
    for (uint e = 1; e < E; ++e) {{
        if (rl[e] > rl[e0]) {{ e0 = e; }}
    }}
    uint e1 = (e0 == 0) ? 1 : 0;
    for (uint e = 0; e < E; ++e) {{
        if (e != e0 && rl[e] > rl[e1]) {{ e1 = e; }}
    }}

    // softmax over 8 — MLX softmax.h order: fast::exp, two serial 4-chunks
    float m = rl[0];
    for (uint e = 1; e < E; ++e) {{
        m = metal::max(m, rl[e]);
    }}
    float c0 = 0.0f;
    float c1 = 0.0f;
    for (uint e = 0; e < 4; ++e) {{
        c0 += metal::fast::exp(rl[e] - m);
    }}
    for (uint e = 4; e < 8; ++e) {{
        c1 += metal::fast::exp(rl[e] - m);
    }}
    float den = c0 + c1;
    float p0r = metal::fast::exp(rl[e0] - m) / den;
    float p1r = metal::fast::exp(rl[e1] - m) / den;
    float psum = p0r + p1r + 1e-6f;
    bfloat16_t p0 = static_cast<bfloat16_t>(p0r / psum);
    bfloat16_t p1 = static_cast<bfloat16_t>(p1r / psum);

    // G_w[b, r, s] = bf16(G[e0,r,s] * p0) + bf16(G[e1,r,s] * p1)
    const ulong o = (ulong)r * R2 + s;
    bfloat16_t g0 = static_cast<bfloat16_t>(
        static_cast<float>(G[(ulong)e0 * R3 * R2 + o]) * static_cast<float>(p0));
    bfloat16_t g1 = static_cast<bfloat16_t>(
        static_cast<float>(G[(ulong)e1 * R3 * R2 + o]) * static_cast<float>(p1));
    G_w[(ulong)b * R3 * R2 + o] =
        static_cast<bfloat16_t>(static_cast<float>(g0) + static_cast<float>(g1));
"""


def get_fused_tucker_gw_kernel(E: int = 8, R3: int = 256, R2: int = 512,
                               router_scale: float = 10.0, temperature: float = 0.5):
    key = ("tucker_gw", E, R3, R2, router_scale, temperature)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    kernel = mx.fast.metal_kernel(
        name=f"tucker_gw_e{E}r{R3}x{R2}",
        input_names=["logits_in", "G"],
        output_names=["G_w"],
        source=_build_tucker_gw_source(E, R3, R2, router_scale, temperature),
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def fused_tucker_gw(kernel, R3, R2, logits, G_bf16):
    """logits (B, E) bf16 + G (E, R3, R2) bf16 → blended G_w (B, R3, R2)."""
    B = logits.shape[0]
    return kernel(
        inputs=[logits, G_bf16],
        grid=(R2, R3, B),
        threadgroup=(64, 4, 1),
        output_shapes=[(B, R3, R2)],
        output_dtypes=[G_bf16.dtype],
    )[0]


def get_fused_tucker_kernel(E: int = 8, TOPK: int = 2, R3: int = 256, R2: int = 512,
                            router_scale: float = 10.0, temperature: float = 0.5):
    key = ("tucker", E, TOPK, R3, R2, router_scale, temperature)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    kernel = mx.fast.metal_kernel(
        name=f"tucker_route_core_e{E}k{TOPK}r{R3}x{R2}",
        input_names=["logits_in", "x_shared", "G"],
        output_names=["x_core"],
        source=_build_tucker_source(E, TOPK, R3, R2, router_scale, temperature),
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def fused_tucker_route_core(kernel, R2, logits, x_shared, G_bf16):
    """router-logits → capped/softmax/top-2/weights → weighted Tucker core
    contraction, in one launch.  logits (1, E) bf16; x_shared (1, R3) bf16;
    G_bf16 (E, R3, R2) bf16.  Returns x_core (1, R2) bf16."""
    return kernel(
        inputs=[logits, x_shared, G_bf16],
        grid=(R2, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(1, R2)],
        output_dtypes=[x_shared.dtype],
    )[0]


def _build_tucker_core_v2_source(E: int, R3: int, R2: int, router_scale: float,
                                 temperature: float, eps: float) -> str:
    """inner_norm(rms) + routing chain + expert blend + core contraction in
    ONE launch — the 5th contraction attempt, fixing why the previous one
    (fused_tucker_route_core, 512 serial-loop threads) lost to MLX:

      * grid (R3, R2): one 256-thread threadgroup PER OUTPUT s → 131K threads,
        same occupancy as the gw kernel, no serial r-loop;
      * G is pre-TRANSPOSED to (E, R2, R3) so lanes r read consecutive
        addresses (coalesced) instead of R2-strided columns;
      * the G_w (R3, R2) intermediate write+readback disappears entirely.

    QUANT-PATH ONLY: f32 reduction order is a serial r-loop, not MLX's gemv
    order, so this is NOT bit-exact vs the unfused ops — the bf16 bit-exact
    path keeps using tucker_gw + mx.einsum.  Per-element rounding still
    mirrors the unfused path (G·p products and their sum round to bf16).

    v2b shape: ONE thread per output s, ZERO barriers.  The transposed G
    makes each thread's r-loop stream CONTIGUOUS memory (vec4 loads) — the
    original fused_tucker_route_core lost because its column reads jumped
    R2·2 bytes every element; the first v2 lost because 512 threadgroups
    ran two barrier reductions each for one multiply of work per thread.

    Output is PADDED to R2+64: x_core[R2] = 1.0, rest 0 — the U_out qmv then
    consumes a weight matrix whose row R2 is the bias (bias-add launch gone;
    the bias row is the max of its own quant group, so it quantizes to
    ≤1/254 relative error)."""
    assert E == 8 and R3 % 4 == 0
    return f"""
    constexpr uint E  = {E};
    constexpr uint R3 = {R3};
    constexpr uint R2 = {R2};
    using bf4 = vec<bfloat16_t, 4>;

    const uint s = thread_position_in_grid.x;               // 0..R2+63
    if (s >= R2) {{
        x_core[s] = (s == R2) ? static_cast<bfloat16_t>(1.0f)
                              : static_cast<bfloat16_t>(0.0f);
        return;
    }}

    // ── inner_norm rms scalar (redundant per thread, xs_raw is L1-hot) ─────
    float a = 0.0f;
    const device bf4* xr4 = reinterpret_cast<const device bf4*>(xs_raw);
    for (uint i = 0; i < R3 / 4; ++i) {{
        bf4 v = xr4[i];
        for (uint j = 0; j < 4; ++j) {{
            float f = static_cast<float>(v[j]);
            a += f * f;
        }}
    }}
    const float inv = metal::precise::rsqrt(a / {float(R3)}f + {eps:.9e}f);

    // ── routing chain — same formulas as the tucker_gw kernel ──────────────
    float rl[E];
    for (uint e = 0; e < E; ++e) {{
        float t  = static_cast<float>(logits_in[e]) * {1.0 / router_scale}f;
        float u2 = 2.0f * t;
        float y0 = 1.0f / (1.0f + metal::exp(metal::abs(u2)));
        float sg = (u2 < 0.0f) ? y0 : 1.0f - y0;
        rl[e] = ((2.0f * sg - 1.0f) * {router_scale}f) / {temperature}f;
    }}
    uint e0 = 0;
    for (uint e = 1; e < E; ++e) {{
        if (rl[e] > rl[e0]) {{ e0 = e; }}
    }}
    uint e1 = (e0 == 0) ? 1 : 0;
    for (uint e = 0; e < E; ++e) {{
        if (e != e0 && rl[e] > rl[e1]) {{ e1 = e; }}
    }}
    float m = rl[0];
    for (uint e = 1; e < E; ++e) {{
        m = metal::max(m, rl[e]);
    }}
    float c0 = 0.0f;
    float c1 = 0.0f;
    for (uint e = 0; e < 4; ++e) {{
        c0 += metal::fast::exp(rl[e] - m);
    }}
    for (uint e = 4; e < 8; ++e) {{
        c1 += metal::fast::exp(rl[e] - m);
    }}
    float den = c0 + c1;
    float p0r = metal::fast::exp(rl[e0] - m) / den;
    float p1r = metal::fast::exp(rl[e1] - m) / den;
    float psum = p0r + p1r + 1e-6f;
    float p0 = static_cast<float>(static_cast<bfloat16_t>(p0r / psum));
    float p1 = static_cast<float>(static_cast<bfloat16_t>(p1r / psum));

    // ── blend + contraction: x_core[s] = Σ_r xs[r]·G_w(r,s) ────────────────
    // per-thread CONTIGUOUS streams over both expert rows (vec4 loads)
    const device bf4* g0r = reinterpret_cast<const device bf4*>(
        G_T + ((ulong)e0 * R2 + s) * R3);
    const device bf4* g1r = reinterpret_cast<const device bf4*>(
        G_T + ((ulong)e1 * R2 + s) * R3);
    const device bf4* nw4 = reinterpret_cast<const device bf4*>(norm_w);
    float acc = 0.0f;
    for (uint i = 0; i < R3 / 4; ++i) {{
        bf4 xv = xr4[i];
        bf4 g0v = g0r[i];
        bf4 g1v = g1r[i];
        bf4 nv = nw4[i];
        for (uint j = 0; j < 4; ++j) {{
            bfloat16_t xs = nv[j] * static_cast<bfloat16_t>(
                static_cast<float>(xv[j]) * inv);
            bfloat16_t g0 = static_cast<bfloat16_t>(
                static_cast<float>(g0v[j]) * p0);
            bfloat16_t g1 = static_cast<bfloat16_t>(
                static_cast<float>(g1v[j]) * p1);
            bfloat16_t gw = static_cast<bfloat16_t>(
                static_cast<float>(g0) + static_cast<float>(g1));
            acc += static_cast<float>(xs) * static_cast<float>(gw);
        }}
    }}
    x_core[s] = static_cast<bfloat16_t>(acc);
"""


def _build_pregate_source(D: int, eps: float) -> str:
    """gated = rms_norm(y, w, eps) * silu(z) in ONE launch (replaces the
    rmsbfloat16 kernel + the fused sigmoid·mul elementwise kernel).

    BIT-EXACT: the rms reduction replicates MLX rms_single_row (N_READS=4,
    D/4-thread tg, simd_sum tree — same recipe as the SSM norm-fold, verified
    4000/4000); sigmoid uses MLX's stable form; every binary op rounds to
    bf16 at the unfused op boundaries."""
    assert D % 128 == 0
    NT = D // 4
    return f"""
    constexpr uint D = {D};

    const uint lid   = thread_position_in_threadgroup.x;     // 0..{NT - 1}
    const uint bb_   = thread_position_in_grid.y;             // batch row
    const uint slane = thread_index_in_simdgroup;
    const uint sgid  = simdgroup_index_in_threadgroup;

    threadgroup float local_sums[32];
    threadgroup float tg_inv[1];

    float acc = 0.0f;
    for (uint i = 0; i < 4; ++i) {{
        float xi = static_cast<float>(y[bb_ * D + lid * 4u + i]);
        acc += xi * xi;
    }}
    acc = metal::simd_sum(acc);
    if (sgid == 0u) {{ local_sums[slane] = 0.0f; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slane == 0u) {{ local_sums[sgid] = acc; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {{
        acc = metal::simd_sum(local_sums[slane]);
        if (slane == 0u) {{
            tg_inv[0] = metal::precise::rsqrt(acc / {float(D)}f + {eps:.9e}f);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = tg_inv[0];

    const bfloat16_t zero_b = static_cast<bfloat16_t>(0.0f);
    const bfloat16_t one_b  = static_cast<bfloat16_t>(1.0f);
    for (uint i = 0; i < 4; ++i) {{
        const uint ix = lid * 4u + i;
        bfloat16_t nrm = w[ix] * static_cast<bfloat16_t>(
            static_cast<float>(y[bb_ * D + ix]) * inv);
        bfloat16_t zb = z[bb_ * D + ix];
        bfloat16_t y0 = one_b / (one_b + metal::exp(metal::abs(zb)));
        bfloat16_t sg = (zb < zero_b) ? y0 : one_b - y0;
        bfloat16_t sz = static_cast<bfloat16_t>(
            static_cast<float>(zb) * static_cast<float>(sg));
        gated[bb_ * D + ix] = static_cast<bfloat16_t>(
            static_cast<float>(nrm) * static_cast<float>(sz));
    }}
"""


def get_fused_pregate_kernel(D: int, eps: float):
    key = ("pregate", D, eps)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    kernel = mx.fast.metal_kernel(
        name=f"pregate_silu_d{D}",
        input_names=["y", "z", "w"],
        output_names=["gated"],
        source=_build_pregate_source(D, eps),
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def fused_pregate(kernel, D, y, z, w):
    """rms(y, w) * silu(z) — y/z (B, 1, D) bf16, w (D,) bf16 → (B, 1, D)."""
    B = y.shape[0]
    return kernel(
        inputs=[y, z, w],
        grid=(D // 4, B, 1),
        threadgroup=(D // 4, 1, 1),
        output_shapes=[(B, 1, D)],
        output_dtypes=[y.dtype],
    )[0]


def get_fused_tucker_core_v2_kernel(E: int = 8, R3: int = 256, R2: int = 512,
                                    router_scale: float = 10.0,
                                    temperature: float = 0.5,
                                    eps: float = 1e-5):
    key = ("tucker_core_v2", E, R3, R2, router_scale, temperature, eps)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    kernel = mx.fast.metal_kernel(
        name=f"tucker_core_v2_e{E}r{R3}x{R2}",
        input_names=["logits_in", "xs_raw", "G_T", "norm_w"],
        output_names=["x_core"],
        source=_build_tucker_core_v2_source(E, R3, R2, router_scale,
                                            temperature, eps),
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def fused_tucker_core_v2(kernel, R3, R2, logits, xs_raw, G_T, norm_w):
    """inner_norm + routing + blend + contraction, one launch (quant path).

    logits (1, E) bf16; xs_raw (1, R3) bf16 PRE-norm U_in output;
    G_T (E, R2, R3) bf16 pre-transposed core; norm_w (R3,) bf16.
    Returns x_core (1, R2) bf16."""
    return kernel(
        inputs=[logits, xs_raw, G_T, norm_w],
        grid=(R2 + 64, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, R2 + 64)],
        output_dtypes=[xs_raw.dtype],
    )[0]


def fused_ssm_decode_step(kernel, H, N, P, R,
                          dt_p, A_p, lam, theta_f32, ac_prev,
                          B_p, C_p, x_ssm, h_prev, ip):
    """Single fused launch for the SSM inner chain.

    dt_p/A_p/lam : (1,) bf16 slices of the in_proj output
    theta_f32    : (N//2,) float32 — exp(theta_log)[0] (G == 1)
    ac_prev      : (1, H, N//2) float32
    B_p / C_p    : (N*R,) bf16 — post norm_B/norm_C + bias, single group
    x_ssm        : (1, 1, H*P*R) bf16 — x_up_proj output
    h_prev, ip   : (1, H, N, P) bf16

    Returns (y (1,1,H,P*R) bf16, h_new (1,H,N,P) bf16,
             new_ip (1,H,N,P) bf16, new_ac (1,H,N//2) f32).
    """
    B = h_prev.shape[0]
    outs = kernel(
        inputs=[dt_p, A_p, lam, theta_f32, ac_prev, B_p, C_p, x_ssm, h_prev, ip],
        grid=(N, P, H * B),
        threadgroup=(N, 1, 1),
        output_shapes=[(B, H, N, P), (B, H, N, P), (B, 1, H, P * R), (B, H, N // 2)],
        output_dtypes=[h_prev.dtype, h_prev.dtype, h_prev.dtype, mx.float32],
    )
    h_new, new_ip, y, new_ac = outs
    return y, h_new, new_ip, new_ac


def fused_ssm_norm_decode_step(kernel, H, N, P, R,
                               dt_p, A_p, lam, theta_f32, ac_prev,
                               B_raw, C_raw, wB, wC, bB, bC,
                               x_ssm, h_prev, ip):
    """fused_ssm_decode_step with in-kernel norm_B/norm_C + bias.

    B_raw / C_raw : (N*R,) bf16 — RAW in_proj slices (pre-norm)
    wB / wC       : (N*R,) bf16 — norm_B / norm_C weights
    bB / bC       : (N*R,) bf16 — bias_B / bias_C flattened (G == 1)
    Remaining arguments and returns: see fused_ssm_decode_step.
    """
    B = h_prev.shape[0]
    outs = kernel(
        inputs=[dt_p, A_p, lam, theta_f32, ac_prev,
                B_raw, C_raw, wB, wC, bB, bC, x_ssm, h_prev, ip],
        grid=(N, P, H * B),
        threadgroup=(N, 1, 1),
        output_shapes=[(B, H, N, P), (B, H, N, P), (B, 1, H, P * R), (B, H, N // 2)],
        output_dtypes=[h_prev.dtype, h_prev.dtype, h_prev.dtype, mx.float32],
    )
    h_new, new_ip, y, new_ac = outs
    return y, h_new, new_ip, new_ac
