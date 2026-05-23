#!/usr/bin/env python3
"""
Benchmark a fused "Tucker Front" kernel: Router + U_in + RMSNorm + Core
in a single Metal dispatch for batch=1 TuckerMoE decode.

Compared to the current multi-dispatch path:
  1. router Linear (dispatch)
  2. softmax+topk (dispatch)
  3. U_in Linear (dispatch)
  4. RMSNorm (dispatch)
  5. scalar core kernel (dispatch)
  6. U_out Linear (dispatch)
  7. + bias (dispatch)

Tucker Front fuses steps 1-5 into ONE dispatch, leaving only U_out + bias.
"""
from __future__ import annotations
import time, sys, math
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "inference"))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

WARMUP = 15
TRIALS = 80


def _bench(fn, warmup=WARMUP, trials=TRIALS):
    for _ in range(warmup):
        r = fn()
        if isinstance(r, mx.array):
            mx.eval(r)
        elif isinstance(r, (tuple, list)):
            for x in r:
                if isinstance(x, mx.array):
                    mx.eval(x)
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        r = fn()
        if isinstance(r, mx.array):
            mx.eval(r)
        elif isinstance(r, (tuple, list)):
            for x in r:
                if isinstance(x, mx.array):
                    mx.eval(x)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# ═══════════════════════════════════════════════════════════════
# Reference: MLX path (Router + U_in + Norm + Core scalar)
# ═══════════════════════════════════════════════════════════════

def reference_tucker_front(
    x: mx.array,          # (dim_in,) bf16
    router_w: mx.array,   # (E, dim_in) bf16
    router_b: mx.array,   # (E,) bf16 or None
    u_in_w: mx.array,     # (r3, dim_in) bf16
    norm_w: mx.array,     # (r3,) bf16
    g_all: mx.array,      # (E, r3, r2) bf16
    top_k: int = 2,
    norm_eps: float = 1e-5,
) -> mx.array:
    """Reference: steps 1-5 in Python/MLX."""
    from inference.lib.mlx_hybrid_infer import fast_scaled_tanh
    # 1. Router
    logits = x @ router_w.T
    if router_b is not None:
        logits = logits + router_b
    capped = fast_scaled_tanh(logits, 10.0)
    t = mx.array(0.5, dtype=x.dtype)
    router_logits = capped / mx.maximum(t, mx.array(1e-4, dtype=t.dtype))
    probs = mx.softmax(router_logits, axis=-1)

    # top-k
    indices = mx.argpartition(-probs, kth=top_k - 1, axis=-1)[:top_k]
    top_probs = probs[indices]
    top_probs = top_probs / (mx.sum(top_probs) + 1e-6)

    # 2. U_in
    h = x @ u_in_w.T  # (r3,)

    # 3. RMSNorm
    ss = mx.sum(h * h) / h.shape[0]
    inv_rms = mx.rsqrt(ss + norm_eps)
    h = h * inv_rms * norm_w

    # 4. Core einsum
    out = mx.zeros((g_all.shape[2],), dtype=mx.float32)
    for ki in range(top_k):
        eid = int(indices[ki].item())
        p = float(top_probs[ki].item())
        out = out + p * (h.astype(mx.float32) @ g_all[eid].astype(mx.float32))
    return out.astype(x.dtype)


# ═══════════════════════════════════════════════════════════════
# Fused Tucker Front Kernel (Metal)
# ═══════════════════════════════════════════════════════════════

def build_tucker_front(dim_in, r3, r2, E, K, norm_eps=1e-5):
    """
    Fused kernel: Router + U_in + Norm + Core in ONE dispatch.

    Design for batch=1:
      Grid:  (r2, 1, 1)
      TG:    (TG_SIZE, 1, 1) where TG_SIZE = min(256, r2)

    Each threadgroup:
      Phase 1: Cooperatively compute router logits (E outputs) → softmax → top-K
      Phase 2: Cooperatively compute U_in matmul (r3 outputs)
      Phase 3: Cooperatively compute RMSNorm
      Phase 4: Each thread computes its assigned core output elements
    """
    TG_SIZE = min(256, r2)
    BANK_SHIFT = 5

    r3_pad = r3 + ((r3 + ((1 << BANK_SHIFT) - 1)) >> BANK_SHIFT)

    source = f"""
        uint gid = thread_position_in_grid.x;
        uint lid = thread_index_in_threadgroup;

        // ── Phase 1: Router (all threads cooperate) ──
        threadgroup float router_buf[{E}];
        threadgroup uint top_idx[{K}];
        threadgroup float top_prob[{K}];

        // Compute router logits
        for (uint e = lid; e < {E}; e += {TG_SIZE}) {{
            float acc = 0.0f;
            for (uint i = 0; i < {dim_in}; ++i) {{
                acc += float(x[i]) * float(router_w[e * {dim_in} + i]);
            }}
            // Scaled tanh capping
            float capped = 10.0f * metal::precise::tanh(acc / 10.0f);
            router_buf[e] = capped / 0.5f;  // router_temp = 0.5
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Softmax (single thread for small E)
        if (lid == 0) {{
            float max_val = router_buf[0];
            for (uint e = 1; e < {E}; ++e) max_val = metal::max(max_val, router_buf[e]);
            float sum_exp = 0.0f;
            for (uint e = 0; e < {E}; ++e) {{
                router_buf[e] = metal::exp(router_buf[e] - max_val);
                sum_exp += router_buf[e];
            }}
            float inv_sum = 1.0f / sum_exp;
            for (uint e = 0; e < {E}; ++e) router_buf[e] *= inv_sum;

            // Top-K selection (K={K}, simple for K<=4)
            for (uint k = 0; k < {K}; ++k) {{
                float best = -1e9f;
                uint best_idx = 0;
                for (uint e = 0; e < {E}; ++e) {{
                    if (router_buf[e] > best) {{
                        best = router_buf[e];
                        best_idx = e;
                    }}
                }}
                top_idx[k] = best_idx;
                top_prob[k] = best;
                router_buf[best_idx] = -1e9f;
            }}
            // Normalize top-K probs
            float prob_sum = 0.0f;
            for (uint k = 0; k < {K}; ++k) prob_sum += top_prob[k];
            float inv_ps = 1.0f / (prob_sum + 1e-6f);
            for (uint k = 0; k < {K}; ++k) top_prob[k] *= inv_ps;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 2: U_in matmul → threadgroup SRAM ──
        threadgroup float x_sram[{r3_pad}];
        for (uint r = lid; r < {r3}; r += {TG_SIZE}) {{
            float acc = 0.0f;
            uint w_base = r * {dim_in};
            for (uint i = 0; i < {dim_in}; i += 4) {{
                acc += float(x[i])   * float(u_in_w[w_base + i]);
                acc += float(x[i+1]) * float(u_in_w[w_base + i+1]);
                acc += float(x[i+2]) * float(u_in_w[w_base + i+2]);
                acc += float(x[i+3]) * float(u_in_w[w_base + i+3]);
            }}
            x_sram[r + (r >> {BANK_SHIFT})] = acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 3: RMSNorm ──
        // Each thread computes partial sum for RMS
        threadgroup float rms_buf[{TG_SIZE}];
        float local_ss = 0.0f;
        for (uint r = lid; r < {r3}; r += {TG_SIZE}) {{
            float v = x_sram[r + (r >> {BANK_SHIFT})];
            local_ss += v * v;
        }}
        rms_buf[lid] = local_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree reduction
        for (uint s = {TG_SIZE} >> 1; s > 0; s >>= 1) {{
            if (lid < s) rms_buf[lid] += rms_buf[lid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        float inv_rms = metal::rsqrt(rms_buf[0] / float({r3}) + {norm_eps}f);
        for (uint r = lid; r < {r3}; r += {TG_SIZE}) {{
            uint rp = r + (r >> {BANK_SHIFT});
            x_sram[rp] = x_sram[rp] * inv_rms * float(norm_w[r]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 4: Core einsum for this thread's output elements ──
        if (gid >= {r2}) return;

        float sum = 0.0f;
        for (uint k = 0; k < {K}; ++k) {{
            uint eid = top_idx[k];
            float prob = top_prob[k];
            uint g_base = eid * {r3} * {r2} + gid;
            float expert_sum = 0.0f;
            for (uint r = 0; r < {r3}; r += 4) {{
                float x0 = x_sram[r + (r >> {BANK_SHIFT})];
                float x1 = x_sram[(r+1) + ((r+1) >> {BANK_SHIFT})];
                float x2 = x_sram[(r+2) + ((r+2) >> {BANK_SHIFT})];
                float x3 = x_sram[(r+3) + ((r+3) >> {BANK_SHIFT})];
                expert_sum += x0 * float(g_all[g_base + r * {r2}]);
                expert_sum += x1 * float(g_all[g_base + (r+1) * {r2}]);
                expert_sum += x2 * float(g_all[g_base + (r+2) * {r2}]);
                expert_sum += x3 * float(g_all[g_base + (r+3) * {r2}]);
            }}
            sum += prob * expert_sum;
        }}
        out[gid] = T(sum);
    """

    kernel = mx.fast.metal_kernel(
        name=f"tucker_front_d{dim_in}_r3{r3}_r2{r2}_e{E}_k{K}",
        input_names=["x", "router_w", "u_in_w", "norm_w", "g_all"],
        output_names=["out"],
        source=source,
    )
    return kernel, TG_SIZE


def run_tucker_front(kernel, tg_size, x, router_w, u_in_w, norm_w, g_all, r2):
    return kernel(
        inputs=[x, router_w, u_in_w, norm_w, mx.flatten(g_all)],
        template=[("T", x.dtype)],
        grid=(r2, 1, 1),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[(r2,)],
        output_dtypes=[x.dtype],
    )[0]


def main():
    print("═" * 65)
    print("Tucker Front Kernel Benchmark (batch=1)")
    print("═" * 65)

    configs = [
        ("x_up_proj", 1536, 256, 512, 6144),
        ("out_proj", 768, 256, 512, 768),
        ("gate_proj (FFN)", 768, 256, 512, 4608),
    ]

    for name, dim_in, r3, r2, dim_out in configs:
        print(f"\n{'─'*65}")
        print(f"  {name}: dim_in={dim_in} r3={r3} r2={r2} dim_out={dim_out}")
        print(f"{'─'*65}")

        E, K = 8, 2

        # Create test data (bf16)
        x = mx.random.normal((dim_in,)).astype(mx.bfloat16)
        router_w = mx.random.normal((E, dim_in)).astype(mx.bfloat16) * 0.01
        u_in_w = mx.random.normal((r3, dim_in)).astype(mx.bfloat16) * 0.01
        norm_w = mx.ones((r3,), dtype=mx.bfloat16)
        g_all = mx.random.normal((E, r3, r2)).astype(mx.bfloat16) * 0.01
        u_out_w = mx.random.normal((dim_out, r2)).astype(mx.bfloat16) * 0.01
        bias = mx.zeros((dim_out,), dtype=mx.bfloat16)
        mx.eval(x, router_w, u_in_w, norm_w, g_all, u_out_w, bias)

        # Accuracy check
        ref = reference_tucker_front(x, router_w, None, u_in_w, norm_w, g_all, K)
        mx.eval(ref)

        kernel, tg_size = build_tucker_front(dim_in, r3, r2, E, K)
        fused = run_tucker_front(kernel, tg_size, x, router_w, u_in_w, norm_w, g_all, r2)
        mx.eval(fused)

        err = float(mx.max(mx.abs(ref.astype(mx.float32) - fused.astype(mx.float32))).item())
        status = "PASS" if err < 0.1 else "FAIL"
        print(f"  Accuracy: {status} (max_err={err:.6f})")

        # Benchmark: Fused Tucker Front only
        def fn_fused_front():
            return run_tucker_front(kernel, tg_size, x, router_w, u_in_w, norm_w, g_all, r2)
        t_fused = _bench(fn_fused_front)

        # Benchmark: Fused Front + U_out matmul (full MoE minus bias)
        def fn_fused_full():
            core = run_tucker_front(kernel, tg_size, x, router_w, u_in_w, norm_w, g_all, r2)
            return core[None, :] @ u_out_w.T  # (1, dim_out)
        t_fused_full = _bench(fn_fused_full)

        # Benchmark: Current scalar_fuse path (reference MLX multi-dispatch)
        from inference.lib.mlx_hybrid_infer import TuckerMoE, rms_norm_fast, _topk_indices, fast_scaled_tanh
        from metal.ultimate_kernel_lib import UltimateMambaKernels
        _uk = UltimateMambaKernels()

        moe_dense = TuckerMoE(dim_in, dim_out, num_experts=E, top_k=K, r1=32, r2=r2, r3=r3)
        moe_dense.apply(lambda v: v.astype(mx.bfloat16))
        mx.eval(moe_dense.parameters())
        moe_dense._get_G()

        rt = mx.array(0.5, dtype=mx.bfloat16)
        x_2d = x[None, :]

        def fn_scalar():
            return moe_dense(x_2d, rt, einsum_fuse=True, scalar_fuse=True)
        t_scalar_dense = _bench(fn_scalar)

        # Quantized path
        moe_q4 = TuckerMoE(dim_in, dim_out, num_experts=E, top_k=K, r1=32, r2=r2, r3=r3)
        moe_q4.apply(lambda v: v.astype(mx.bfloat16))
        nn.quantize(moe_q4, group_size=64, bits=4)
        mx.eval(moe_q4.parameters())
        moe_q4._get_G()

        def fn_scalar_q4():
            return moe_q4(x_2d, rt, einsum_fuse=True, scalar_fuse=True)
        t_scalar_q4 = _bench(fn_scalar_q4)

        # Results
        print(f"\n  {'Mode':<35} {'Time (ms)':>10} {'vs scalar(q4)':>14}")
        print(f"  {'─'*60}")
        print(f"  {'scalar_fuse (q4, current)':<35} {t_scalar_q4:>9.3f}  {'baseline':>13}")
        print(f"  {'scalar_fuse (dense)':<35} {t_scalar_dense:>9.3f}  {t_scalar_q4/t_scalar_dense:>12.2f}×")
        print(f"  {'FUSED Front only':<35} {t_fused:>9.3f}  {t_scalar_q4/t_fused:>12.2f}×")
        print(f"  {'FUSED Front + U_out (dense)':<35} {t_fused_full:>9.3f}  {t_scalar_q4/t_fused_full:>12.2f}×")

        # Savings estimate
        savings_per_call = t_scalar_q4 - t_fused
        n_calls = 48 if "up" in name or "out" in name else 18
        print(f"\n  Savings per call: {savings_per_call:.3f} ms")
        print(f"  Projected decode savings ({n_calls} calls): {savings_per_call * n_calls:.2f} ms")

    print(f"\n{'═'*65}")
    print("SUMMARY")
    print(f"{'═'*65}")
    print("  If Tucker Front achieves 2-3× over scalar_fuse(q4),")
    print("  total decode savings = 66 calls × ~0.15ms = ~10ms (raw)")
    print("  With compile: ~2-3ms savings → 88.2 → 100+ tok/s")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()
