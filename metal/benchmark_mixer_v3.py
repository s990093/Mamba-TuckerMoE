#!/usr/bin/env python3
"""Benchmark fused Mamba mixer V3 (with y_down+D fused) vs V1."""
import time, sys
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
    H, N, P, R = 24, 64, 64, 4
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
    x_prime = mx.random.normal((H * P,)).astype(mx.bfloat16)
    d_rep = mx.random.normal((H * P,)).astype(mx.bfloat16) * 0.1
    mx.eval(b_raw, c_raw, norm_b_w, norm_c_w, bias_b, bias_c,
            theta_rep, prev_angle, x_ssm, prev_h, prev_in,
            dt_b, a_b, lv, yd_weight, x_prime, d_rep)

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
            yd_weight=yd_weight, x_prime_flat=x_prime, d_rep=d_rep,
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

    # Benchmark
    t_v1 = _bench(fn_v1)
    t_v3 = _bench(fn_v3)

    print(f"\nBenchmark (H={H}, N={N}, P={P}, R={R}):")
    print(f"  V1 (Norm+RoPE+SSM+Ein2):         {t_v1:.3f} ms")
    print(f"  V3 (V1 + y_down + D_skip fused):  {t_v3:.3f} ms")
    print(f"  Overhead of y_down+D fusion:      {t_v3 - t_v1:.3f} ms")
    print(f"\n  V3 eliminates y_down_proj + D_skip dispatches →")
    print(f"  Estimated savings per decode: 24 layers × ~0.015-0.030 ms/dispatch")
    print(f"  = {24 * 0.020:.2f} ms saved from dispatch elimination")

if __name__ == "__main__":
    main()
