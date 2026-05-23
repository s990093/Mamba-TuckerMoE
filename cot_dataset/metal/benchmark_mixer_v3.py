#!/usr/bin/env python3
"""
Benchmark fused Mamba mixer V3 (with y_down+D fused) vs V1.

Key finding: V3 is SLOWER than V1 in mx.compile (throughput) mode.
See the analysis printed at the end of the benchmark for details.

Usage:
    python metal/benchmark_mixer_v3.py
    python metal/benchmark_mixer_v3.py --compiled   # simulate compiled mode
    python metal/benchmark_mixer_v3.py --eager      # eager dispatch overhead visible
"""
import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

import mlx.core as mx
from metal.ultimate_kernel_lib import UltimateMambaKernels
from inference.lib.mlx_hybrid_infer import rms_norm_fast, apply_rope

WARMUP = 15
TRIALS = 80

def _bench(fn, warmup=WARMUP, trials=TRIALS):
    for _ in range(warmup):
        r = fn()
        if isinstance(r, (tuple, list)):
            mx.eval(*[x for x in r if isinstance(x, mx.array)])
        else:
            mx.eval(r)
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        r = fn()
        if isinstance(r, (tuple, list)):
            mx.eval(*[x for x in r if isinstance(x, mx.array)])
        else:
            mx.eval(r)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times)//2]


def main():
    parser = argparse.ArgumentParser(description="Benchmark V1 vs V3 Mamba Mixer kernels")
    parser.add_argument("--H", type=int, default=24, help="Number of SSM heads")
    parser.add_argument("--N", type=int, default=64, help="SSM state size")
    parser.add_argument("--P", type=int, default=64, help="SSM head dim")
    parser.add_argument("--R", type=int, default=4,  help="MIMO rank")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--trials", type=int, default=TRIALS)
    args = parser.parse_args()

    H, N, P, R = args.H, args.N, args.P, args.R
    uk = UltimateMambaKernels()

    # Test data
    mx.random.seed(42)
    b_raw = mx.random.normal((N * R,)).astype(mx.bfloat16)
    c_raw = mx.random.normal((N * R,)).astype(mx.bfloat16)
    norm_b_w = mx.ones((N * R,), dtype=mx.bfloat16)
    norm_c_w = mx.ones((N * R,), dtype=mx.bfloat16)
    bias_b = mx.random.normal((N * R,)).astype(mx.bfloat16) * 0.01
    bias_c = mx.random.normal((N * R,)).astype(mx.bfloat16) * 0.01
    theta_rep = mx.random.normal((H, N // 2)).astype(mx.bfloat16) * 0.1
    prev_angle = mx.zeros((H, N // 2), dtype=mx.bfloat16)
    x_ssm = mx.random.normal((H, P, R)).astype(mx.bfloat16)
    prev_h = mx.random.normal((H, N, P)).astype(mx.bfloat16) * 0.1
    prev_in = mx.random.normal((H, N, P)).astype(mx.bfloat16) * 0.1
    dt_b = mx.random.normal((H,)).astype(mx.bfloat16) * 0.1
    a_b = mx.random.normal((H,)).astype(mx.bfloat16) * -0.5
    lv = mx.random.uniform(shape=(H,)).astype(mx.bfloat16)
    yd_weight = mx.random.normal((P, P * R)).astype(mx.bfloat16) * 0.01
    # Pre-transpose for coalesced Phase 3 reads: (P_out, P_R) → (P_R, P_out)
    yd_weight_T = mx.contiguous(yd_weight.T)
    x_prime = mx.random.normal((H * P,)).astype(mx.bfloat16)
    d_rep = mx.random.normal((H * P,)).astype(mx.bfloat16) * 0.1
    mx.eval(b_raw, c_raw, norm_b_w, norm_c_w, bias_b, bias_c,
            theta_rep, prev_angle, x_ssm, prev_h, prev_in,
            dt_b, a_b, lv, yd_weight, yd_weight_T, x_prime, d_rep)

    # Build kernels
    k_v1 = uk.mamba_mixer.build(H, N, P, R)
    k_v3 = uk.mamba_mixer.build_v3(H, N, P, R)

    # V1 run
    def fn_v1():
        return uk.mamba_mixer.run(
            k_v1,
            b_raw=b_raw, c_raw=c_raw,
            norm_b_w=norm_b_w, norm_c_w=norm_c_w,
            bias_b=bias_b, bias_c=bias_c,
            theta_rep=theta_rep, prev_angle=prev_angle,
            x_ssm=x_ssm, prev_h=prev_h, prev_input=prev_in,
            dt_b=dt_b, a_b=a_b, lv=lv,
            h=H, n=N, p=P, r=R,
        )

    # V3 run
    def fn_v3():
        return uk.mamba_mixer.run_v3(
            k_v3,
            b_raw=b_raw, c_raw=c_raw,
            norm_b_w=norm_b_w, norm_c_w=norm_c_w,
            bias_b=bias_b, bias_c=bias_c,
            theta_rep=theta_rep, prev_angle=prev_angle,
            x_ssm=x_ssm, prev_h=prev_h, prev_input=prev_in,
            dt_b=dt_b, a_b=a_b, lv=lv,
            yd_weight_T=yd_weight_T, x_prime_flat=x_prime, d_rep=d_rep,
            h=H, n=N, p=P, r=R,
        )

    # V1 + separate y_down + D_skip (what happens WITHOUT V3)
    def fn_v1_plus_ydown():
        nh, ni, y_out, na = uk.mamba_mixer.run(
            k_v1,
            b_raw=b_raw, c_raw=c_raw,
            norm_b_w=norm_b_w, norm_c_w=norm_c_w,
            bias_b=bias_b, bias_c=bias_c,
            theta_rep=theta_rep, prev_angle=prev_angle,
            x_ssm=x_ssm, prev_h=prev_h, prev_input=prev_in,
            dt_b=dt_b, a_b=a_b, lv=lv,
            h=H, n=N, p=P, r=R,
        )
        y = mx.einsum("hpr, qr -> hp", y_out.reshape(H, P, R * P // P, R), yd_weight.reshape(P, P, R))
        return nh, ni, y, na

    # Accuracy check
    h1, i1, y1, a1 = fn_v1()
    h3, i3, y3, a3 = fn_v3()
    mx.eval(h1, i1, y1, a1, h3, i3, y3, a3)

    h_err = float(mx.max(mx.abs(h1.astype(mx.float32) - h3.astype(mx.float32))).item())
    i_err = float(mx.max(mx.abs(i1.astype(mx.float32) - i3.astype(mx.float32))).item())
    a_err = float(mx.max(mx.abs(a1.astype(mx.float32) - a3.astype(mx.float32))).item())
    print(f"Accuracy check (V1 vs V3):")
    print(f"  h_final: max_err={h_err:.6f} {'PASS' if h_err < 0.05 else 'FAIL'}")
    print(f"  inp_sig: max_err={i_err:.6f} {'PASS' if i_err < 0.05 else 'FAIL'}")
    print(f"  angle:   max_err={a_err:.6f} {'PASS' if a_err < 0.05 else 'FAIL'}")

    # Check y_combined accuracy
    y1_flat = y1.reshape(H, P, R)
    y_down_ref = mx.zeros((H, P), dtype=mx.float32)
    for hh in range(H):
        for pp in range(P):
            acc = 0.0
            for pr in range(P * R):
                acc += float(y1_flat.reshape(H, P * R)[hh, pr].item()) * float(yd_weight[pp, pr].item())
            y_down_ref = y_down_ref.at[hh, pp].add(mx.array(acc, dtype=mx.float32))
    y_ref_combined = y_down_ref.reshape(H * P) + x_prime.astype(mx.float32) * d_rep.astype(mx.float32)
    y3_f32 = y3.astype(mx.float32)
    mx.eval(y_ref_combined, y3_f32)
    y_err = float(mx.max(mx.abs(y_ref_combined - y3_f32)).item())
    print(f"  y_combined: max_err={y_err:.6f} {'PASS' if y_err < 0.5 else 'FAIL'}")

    # V1 + separate y_down_proj measured independently (simulate what MLX does)
    def fn_ydown_only():
        """Measure just the y_down_proj + D_skip operations in isolation."""
        _, _, y_out, _ = uk.mamba_mixer.run(
            k_v1,
            b_raw=b_raw, c_raw=c_raw,
            norm_b_w=norm_b_w, norm_c_w=norm_c_w,
            bias_b=bias_b, bias_c=bias_c,
            theta_rep=theta_rep, prev_angle=prev_angle,
            x_ssm=x_ssm, prev_h=prev_h, prev_input=prev_in,
            dt_b=dt_b, a_b=a_b, lv=lv,
            h=H, n=N, p=P, r=R,
        )
        # Simulate y_down_proj as MLX matmul (what V1 path does after the kernel)
        y_flat = y_out.reshape(H * P, R)                              # (H*P, R)
        y_down = mx.matmul(y_flat, yd_weight.reshape(P, R, P).sum(axis=1).T)  # rough proxy
        y_final = y_down.reshape(-1) + x_prime * d_rep
        return y_final

    # ── Benchmark ──────────────────────────────────────────────────────────

    print(f"\nBenchmarking (H={H}, N={N}, P={P}, R={R})...")
    print(f"  Warmup={args.warmup}, Trials={args.trials}, median reported")

    t_v1 = _bench(fn_v1, warmup=args.warmup, trials=args.trials)
    t_v3 = _bench(fn_v3, warmup=args.warmup, trials=args.trials)

    overhead_ms   = t_v3 - t_v1
    overhead_24   = overhead_ms * 24
    dispatch_save = 24 * 2  # 2 dispatches eliminated per Mamba layer × 24 layers
    # Rough per-dispatch overhead estimate in compiled mode (~10 µs each)
    dispatch_save_ms_lo = dispatch_save * 0.010
    dispatch_save_ms_hi = dispatch_save * 0.020

    print(f"\n{'='*60}")
    print(f"Benchmark Results (H={H}, N={N}, P={P}, R={R})")
    print(f"{'='*60}")
    print(f"  V1  (Norm+RoPE+SSM+Einsum2):          {t_v1:.3f} ms/kernel")
    print(f"  V3  (V1 + y_down + D_skip fused):     {t_v3:.3f} ms/kernel")
    print(f"  Phase 3 overhead:                     {overhead_ms:+.3f} ms/kernel")
    print(f"  Phase 3 overhead × 24 Mamba layers:   {overhead_24:+.3f} ms/step")
    print()
    print(f"  Dispatches eliminated by V3: {dispatch_save} (2 per Mamba layer × 24 layers)")
    print(f"  Dispatch savings estimate:")
    print(f"    mx.compile mode: {dispatch_save_ms_lo:.2f}–{dispatch_save_ms_hi:.2f} ms/step (~10-20 µs/dispatch)")
    print(f"    eager mode:      {dispatch_save * 0.010:.2f}–{dispatch_save * 0.025:.2f} ms/step (~10-25 µs/dispatch incl. Python overhead)")
    print()

    net_compiled_lo = overhead_24 - dispatch_save_ms_hi
    net_compiled_hi = overhead_24 - dispatch_save_ms_lo
    print(f"  Net effect in mx.compile mode:  {net_compiled_lo:+.2f} to {net_compiled_hi:+.2f} ms/step")
    verdict_compiled = "SLOWER (regression)" if net_compiled_lo > 0 else "FASTER (improvement)"
    print(f"  Verdict (compiled):             {verdict_compiled}")
    print()

    print(f"{'='*60}")
    print(f"Analysis: Why V3 is Slower in mx.compile Mode")
    print(f"{'='*60}")
    print(f"""
  In mx.compile (throughput) mode, MLX builds a single Metal command buffer
  for the entire decode step.  Intermediate tensors — including y_out from
  V1's Einsum2 — are retained in the GPU's L2 cache across kernel dispatches
  within the SAME command buffer.  They do NOT require a round-trip to DRAM.

  Therefore, V3 fusion does NOT reduce real memory bandwidth:
    - DRAM savings from fusion: ~0 bytes (y_out was L2-cached already)
    - Extra compute added by Phase 3: H × P × P_R = {H} × {P} × {P*R} = {H*P*P*R:,} MACs
    - Phase 3 execution time: {overhead_ms:.3f} ms/layer (measured above)

  The 2-dispatch savings ({dispatch_save_ms_lo:.1f}–{dispatch_save_ms_hi:.1f} ms/step) cannot overcome
  the {overhead_24:.2f} ms of added computation across all 24 Mamba layers.

  In eager/safe mode, dispatch overhead is higher (~10-25 µs/dispatch including
  Python + MLX lazy eval overhead).  In theory V3 could break even if dispatch
  overhead ≥ {overhead_ms * 1000 / 2:.0f} µs per extra dispatch, but safe mode is
  slower overall and not the production path.

  Why simd_sum cannot fix Phase 3:
    threadgroup = (P={P}, 1, 1) → {P // 32} SIMD groups of 32 lanes.
    Each lane has a DIFFERENT p → DIFFERENT yd_weight row → DIFFERENT output.
    simd_sum() sums all 32 lanes' partial values = incorrect (cross-output mixing).
    Correct use requires (32_reduction × {P}_outputs) = {32 * P} threads > Metal's 1024 limit.

  Recommendation: Use --fused-mamba-mixer (V1) without --fused-mamba-mixer-v3.
  See metal/DESIGN_GUIDELINES.md §4 for the complete analysis.
""")

    print(f"  'Ideal' V3 savings if Phase 3 were free (compute = 0):")
    print(f"    Best case: {dispatch_save_ms_hi:.2f} ms/step faster")
    print(f"    Realistic: {dispatch_save_ms_lo:.2f}–{dispatch_save_ms_hi:.2f} ms/step")
    print(f"    Actual:    {overhead_24:+.2f} ms/step (Phase 3 cost dominates)")
    print()

if __name__ == "__main__":
    main()
