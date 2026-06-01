"""
benchmark_tree_scan.py — Speed + accuracy comparison for MLX scan paths.

Tests:
  A) inter-chunk scan: serial O(nc) vs Kogge-Stone O(log nc) vs Metal kernel (single dispatch)
  B) chunk_parallel_scan_with_init: old (bf16 cumsum) vs new (f32 cumsum)
  C) single-step SSM update: bf16 vs f32 intermediate
  D) end-to-end chunk_parallel_scan_mlx: determinism + throughput across sequence lengths

Usage:
    python inference/benchmark_tree_scan.py              # nc=1..64, standard
    python inference/benchmark_tree_scan.py --full       # nc=1..512, long-sequence stress
    python inference/benchmark_tree_scan.py --nc 64      # specific nc only
"""
import argparse
import sys
import time
from typing import List

sys.path.insert(0, "inference")
import mlx.core as mx

from lib.mlx_hybrid_infer import chunk_parallel_scan_mlx, chunk_parallel_scan_with_init


# ═══════════════════════════════════════════════════════════════════════════════
# A: inter-chunk scan — three implementations
# ═══════════════════════════════════════════════════════════════════════════════

def _serial_inter_chunk(la_c, h_intra, b, nc, h, n, p, out_dtype):
    """Original O(nc) Python loop — baseline."""
    _CLIP = 40.0
    decay = mx.exp(mx.clip(mx.sum(la_c, axis=2), -_CLIP, _CLIP))
    h_prev = mx.zeros((b, h, n, p), dtype=out_dtype)
    chunks: List[mx.array] = []
    for c in range(nc):
        chunks.append(h_prev)
        dec = mx.expand_dims(mx.expand_dims(decay[:, c], -1), -1)
        h_prev = h_prev * dec + h_intra[:, c, -1]
    return mx.stack(chunks, axis=1), h_prev


def _ks_inter_chunk(la_c, h_intra, b, nc, h, n, p, out_dtype):
    """Kogge-Stone O(log nc) parallel prefix scan."""
    log_d_f32 = mx.clip(mx.sum(la_c.astype(mx.float32), axis=2), -88.0, 88.0)
    d_prefix = mx.exp(log_d_f32)
    h_prefix = h_intra[:, :, -1].astype(mx.float32)
    stride = 1
    while stride < nc:
        mask = mx.arange(nc) >= stride
        d_left = mx.concatenate(
            [mx.ones((b, stride, h), dtype=mx.float32), d_prefix[:, :nc - stride]], axis=1
        )
        h_left = mx.concatenate(
            [mx.zeros((b, stride, h, n, p), dtype=mx.float32), h_prefix[:, :nc - stride]], axis=1
        )
        d_new = d_prefix * d_left
        h_new = d_prefix[:, :, :, None, None] * h_left + h_prefix
        d_prefix = mx.where(mask[None, :, None], d_new, d_prefix)
        h_prefix = mx.where(mask[None, :, None, None, None], h_new, h_prefix)
        stride <<= 1
    h_inter = mx.concatenate(
        [mx.zeros((b, 1, h, n, p), dtype=mx.float32), h_prefix[:, :-1]], axis=1
    ).astype(out_dtype)
    return h_inter, h_prefix[:, -1].astype(out_dtype)


_INTER_METAL_SRC = """
    uint d    = thread_position_in_grid.x;
    uint hh   = thread_position_in_grid.y;
    uint b_id = thread_position_in_grid.z;
    if (d >= NP || hh >= H || b_id >= B) return;
    float h_accum = 0.0f;
    for (uint c = 0; c < NC; ++c) {
        float log_decay = log_d[b_id * NC * H + c * H + hh];
        float decay     = metal::exp(metal::clamp(log_decay, -88.0f, 88.0f));
        float x_val     = float(x_in[b_id * NC * H * NP + c * H * NP + hh * NP + d]);
        h_inter[b_id * NC * H * NP + c * H * NP + hh * NP + d] = h_accum;
        h_accum = h_accum * decay + x_val;
    }
    h_prev[b_id * H * NP + hh * NP + d] = h_accum;
"""


def _metal_inter_chunk(la_c, h_intra_flat, b, nc, h, n, p, d_inner, out_dtype):
    """Metal kernel: single dispatch — nc loop in GPU registers, zero Python overhead."""
    log_d_f32 = mx.clip(mx.sum(la_c.astype(mx.float32), axis=2), -88.0, 88.0)
    x_in_flat = h_intra_flat[:, :, -1]
    kernel = mx.fast.metal_kernel(
        name=f"inter_chunk_scan_{b}_{nc}_{h}_{d_inner}",
        input_names=["log_d", "x_in"],
        output_names=["h_inter", "h_prev"],
        source=_INTER_METAL_SRC,
        header=f"constant uint B={b}; constant uint NC={nc}; constant uint H={h}; constant uint NP={d_inner};",
    )
    out = kernel(
        inputs=[log_d_f32, x_in_flat],
        template=[("T", x_in_flat.dtype)],
        grid=(d_inner, h, b),
        threadgroup=(min(d_inner, 32), 1, 1),
        output_shapes=[(b, nc, h, d_inner), (b, h, d_inner)],
        output_dtypes=[mx.float32, mx.float32],
    )
    h_inter = out[0].reshape(b, nc, h, n, p).astype(out_dtype)
    h_prev  = out[1].reshape(b, h, n, p).astype(out_dtype)
    return h_inter, h_prev


# ═══════════════════════════════════════════════════════════════════════════════
# B: chunk_parallel_scan_with_init — old bf16 vs new f32 cumsum
# ═══════════════════════════════════════════════════════════════════════════════

def _with_init_old(u, dt_b, a_b, c_rotated, chunk_size, h_init):
    """Original: bf16 cumsum, dt_b*a_b computed twice."""
    _CLIP = mx.array(40.0, dtype=dt_b.dtype)
    y_zero, h_final_zero = chunk_parallel_scan_mlx(u, dt_b, a_b, c_rotated, chunk_size)
    log_alpha_cum = mx.clip(mx.cumsum(dt_b * a_b, axis=1), -_CLIP, _CLIP)
    alpha_cum = mx.exp(log_alpha_cum)
    h_init_t = h_init[:, None] * alpha_cum[:, :, :, None, None]
    y_init = mx.einsum("blhnp,blgnr->blhpr", h_init_t, c_rotated)
    y = y_zero + y_init
    log_alpha_total = mx.clip(mx.sum(dt_b * a_b, axis=1), -_CLIP, _CLIP)
    alpha_total = mx.exp(log_alpha_total)
    h_final = h_final_zero + h_init * alpha_total[:, :, None, None]
    return y.astype(u.dtype), h_final.astype(u.dtype)


def _with_init_new(u, dt_b, a_b, c_rotated, chunk_size, h_init):
    """New: f32 cumsum throughout, dt_b*a_b computed once."""
    return chunk_parallel_scan_with_init(u, dt_b, a_b, c_rotated, chunk_size, h_init)


# ═══════════════════════════════════════════════════════════════════════════════
# C: single-step SSM update — bf16 vs f32 intermediate
# ═══════════════════════════════════════════════════════════════════════════════

def _step_old(prev_h, av0, u0, out_dtype):
    return (prev_h * av0 + u0).astype(out_dtype)


def _step_new(prev_h, av0, u0, out_dtype):
    return (
        prev_h.astype(mx.float32) * av0.astype(mx.float32)
        + u0.astype(mx.float32)
    ).astype(out_dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_scan_inputs(nc, chunk_size=16, b=1, h=4, n=8, p=8, seed=42):
    mx.random.seed(seed)
    lc = chunk_size
    d_inner = n * p
    la_c = -mx.abs(mx.random.normal((b, nc, lc, h))).astype(mx.bfloat16) * 0.15
    h_intra = mx.random.normal((b, nc, lc, h, n, p)).astype(mx.bfloat16) * 0.3
    h_intra_flat = h_intra.reshape(b, nc, lc, h, d_inner)
    return la_c, h_intra, h_intra_flat, d_inner


def _make_full_scan_inputs(L, chunk_size=16, b=1, h=4, n=8, p=8, g=1, r=1, seed=7):
    mx.random.seed(seed)
    u = mx.random.normal((b, L, h, n, p)).astype(mx.bfloat16) * 0.1
    dt_b = mx.random.uniform(0.01, 0.5, (b, L, h)).astype(mx.bfloat16)
    a_b = -mx.random.uniform(0.1, 2.0, (b, L, h)).astype(mx.bfloat16)
    c_rotated = mx.random.normal((b, L, g, n, r)).astype(mx.bfloat16)
    h_init = mx.random.normal((b, h, n, p)).astype(mx.bfloat16) * 0.2
    return u, dt_b, a_b, c_rotated, h_init


def _max_err(a, b_arr):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b_arr.astype(mx.float32))).item())


def _time_ms(fn, *args, repeats=20, warmup=3):
    times = []
    for i in range(warmup + repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        dt = (time.perf_counter() - t0) * 1e3
        if i >= warmup:
            times.append(dt)
    times.sort()
    return times[len(times) // 2]


# ═══════════════════════════════════════════════════════════════════════════════
# Section A: inter-chunk scan — serial vs KS vs Metal
# ═══════════════════════════════════════════════════════════════════════════════

def bench_inter_chunk(nc_list, chunk_size, repeats, b=1, h=4, n=8, p=8):
    print("\n" + "═" * 90)
    print("A) inter-chunk scan: serial O(nc) vs Kogge-Stone O(log nc) vs Metal kernel")
    print("═" * 90)
    hdr = f"  {'nc':>5}  {'err_KS':>8}  {'err_Metal':>10}  {'serial_ms':>10}  {'KS_ms':>8}  {'Metal_ms':>9}  {'KS_spd':>8}  {'Metal_spd':>10}"
    print(hdr)
    print("  " + "─" * 78)
    all_ok = True
    for nc in nc_list:
        la_c, h_intra, h_intra_flat, d_inner = _make_scan_inputs(nc, chunk_size, b, h, n, p)

        # Reference: serial (ground truth for this comparison)
        hi_s, hp_s = _serial_inter_chunk(la_c, h_intra, b, nc, h, n, p, mx.bfloat16)
        hi_ks, hp_ks = _ks_inter_chunk(la_c, h_intra, b, nc, h, n, p, mx.bfloat16)
        hi_m, hp_m = _metal_inter_chunk(la_c, h_intra_flat, b, nc, h, n, p, d_inner, mx.bfloat16)
        mx.eval(hi_s, hp_s, hi_ks, hp_ks, hi_m, hp_m)

        e_ks   = max(_max_err(hi_s, hi_ks), _max_err(hp_s, hp_ks))
        e_metal = max(_max_err(hi_s, hi_m), _max_err(hp_s, hp_m))
        ok = e_ks < 1e-2 and e_metal < 1e-2

        t_s  = _time_ms(lambda la, hi: _serial_inter_chunk(la, hi, b, nc, h, n, p, mx.bfloat16), la_c, h_intra, repeats=repeats)
        t_ks = _time_ms(lambda la, hi: _ks_inter_chunk(la, hi, b, nc, h, n, p, mx.bfloat16), la_c, h_intra, repeats=repeats)
        t_m  = _time_ms(lambda la, hf: _metal_inter_chunk(la, hf, b, nc, h, n, p, d_inner, mx.bfloat16), la_c, h_intra_flat, repeats=repeats)

        tag = "✓" if ok else "FAIL"
        print(f"  {nc:>5}  {e_ks:>8.4f}  {e_metal:>10.4f}  {t_s:>10.3f}  {t_ks:>8.3f}  {t_m:>9.3f}  {t_s/t_ks:>7.2f}×  {t_s/t_m:>9.2f}×  {tag}")
        if not ok:
            all_ok = False
    print(f"\n  Accuracy threshold 1e-2 (≤1 bf16 ULP).  All: {'PASSED ✓' if all_ok else 'FAILED ✗'}")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: chunk_parallel_scan_with_init
# ═══════════════════════════════════════════════════════════════════════════════

def bench_with_init(L_list, chunk_size, repeats, b=1, h=4, n=8, p=8):
    print("\n" + "═" * 72)
    print("B) chunk_parallel_scan_with_init — old bf16 vs new f32 cumsum")
    print("═" * 72)
    print(f"  {'L':>6}  {'err_y':>10}  {'err_h':>10}  {'old_ms':>10}  {'new_ms':>8}  {'speedup':>8}")
    print("  " + "─" * 60)
    for L in L_list:
        u, dt_b, a_b, c_rotated, h_init = _make_full_scan_inputs(L, chunk_size, b, h, n, p)
        y_old, hf_old = _with_init_old(u, dt_b, a_b, c_rotated, chunk_size, h_init)
        y_new, hf_new = _with_init_new(u, dt_b, a_b, c_rotated, chunk_size, h_init)
        mx.eval(y_old, hf_old, y_new, hf_new)
        e_y = _max_err(y_old, y_new)
        e_h = _max_err(hf_old, hf_new)
        t_old = _time_ms(_with_init_old, u, dt_b, a_b, c_rotated, chunk_size, h_init, repeats=repeats)
        t_new = _time_ms(_with_init_new, u, dt_b, a_b, c_rotated, chunk_size, h_init, repeats=repeats)
        print(f"  {L:>6}  {e_y:>10.6f}  {e_h:>10.6f}  {t_old:>10.3f}  {t_new:>8.3f}  {t_old/t_new:>7.2f}×")
    print(f"\n  err_y/err_h = divergence introduced by bf16 (old) vs f32 (new) cumsum.")


# ═══════════════════════════════════════════════════════════════════════════════
# Section C: single-step SSM update
# ═══════════════════════════════════════════════════════════════════════════════

def bench_step(h_list, n=8, p=8, repeats=50, b=1):
    print("\n" + "═" * 72)
    print("C) Single-step SSM update — bf16 vs f32 intermediate")
    print("═" * 72)
    print(f"  {'H':>4}  {'e_old':>12}  {'e_new':>12}  {'old_ms':>8}  {'new_ms':>8}  note")
    print("  " + "─" * 64)
    for h in h_list:
        mx.random.seed(99)
        prev_h = mx.random.normal((b, h, n, p)).astype(mx.bfloat16) * 0.5
        av0 = mx.ones((b, h, 1, 1), dtype=mx.bfloat16) * 0.999
        u0 = mx.random.normal((b, h, n, p)).astype(mx.bfloat16) * 0.01
        ref = (prev_h.astype(mx.float32) * av0.astype(mx.float32)
               + u0.astype(mx.float32))
        h_old = _step_old(prev_h, av0, u0, mx.bfloat16).astype(mx.float32)
        h_new = _step_new(prev_h, av0, u0, mx.bfloat16).astype(mx.float32)
        mx.eval(ref, h_old, h_new)
        e_old = _max_err(ref, h_old)
        e_new = _max_err(ref, h_new)
        t_old = _time_ms(_step_old, prev_h, av0, u0, mx.bfloat16, repeats=repeats)
        t_new = _time_ms(_step_new, prev_h, av0, u0, mx.bfloat16, repeats=repeats)
        gain = "same" if abs(e_old - e_new) < 1e-8 else f"{e_old/max(e_new,1e-12):.2f}× better"
        print(f"  {h:>4}  {e_old:>12.4e}  {e_new:>12.4e}  {t_old:>8.3f}  {t_new:>8.3f}  {gain}")
    print(f"\n  f32 intermediate prevents cumulative mantissa erosion over 512+ decode steps.")


# ═══════════════════════════════════════════════════════════════════════════════
# Section D: end-to-end chunk_parallel_scan_mlx — determinism + throughput
# ═══════════════════════════════════════════════════════════════════════════════

def bench_e2e(nc_list, chunk_size, repeats, b=1, h=4, n=8, p=8):
    print("\n" + "═" * 72)
    print("D) End-to-end chunk_parallel_scan_mlx (Metal inter-chunk kernel, full fn)")
    print("═" * 72)
    print(f"  {'nc':>5}  {'L':>6}  {'y_err':>8}  {'h_err':>8}  {'time_ms':>10}  note")
    print("  " + "─" * 60)
    all_ok = True
    for nc in nc_list:
        L = nc * chunk_size
        u, dt_b, a_b, c_rotated, h_init = _make_full_scan_inputs(L, chunk_size, b, h, n, p)
        y1, hf1 = chunk_parallel_scan_mlx(u, dt_b, a_b, c_rotated, chunk_size)
        y2, hf2 = chunk_parallel_scan_mlx(u, dt_b, a_b, c_rotated, chunk_size)
        mx.eval(y1, hf1, y2, hf2)
        e_y = _max_err(y1, y2)
        e_h = _max_err(hf1, hf2)
        ok = e_y == 0.0 and e_h == 0.0
        t = _time_ms(chunk_parallel_scan_mlx, u, dt_b, a_b, c_rotated, chunk_size, repeats=repeats)
        note = "deterministic ✓" if ok else "NON-DETERMINISTIC ✗"
        print(f"  {nc:>5}  {L:>6}  {e_y:>8.2e}  {e_h:>8.2e}  {t:>10.3f}  {note}")
        if not ok:
            all_ok = False
    print(f"\n  {'All runs deterministic ✓' if all_ok else 'DETERMINISM FAILURE'}")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="Extended nc list including long sequences")
    ap.add_argument("--nc", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    cs = args.chunk_size

    if args.nc is not None:
        nc_list = [args.nc]
        L_list  = [args.nc * cs]
    elif args.full:
        nc_list = [1, 4, 8, 16, 32, 64, 128, 256, 512]
        L_list  = [nc * cs for nc in nc_list]
    else:
        nc_list = [1, 4, 8, 16, 32, 64]
        L_list  = [nc * cs for nc in nc_list]

    print(f"\nMamba3-XR MLX scan benchmark  chunk_size={cs}  repeats={args.repeats}")

    bench_inter_chunk(nc_list, cs, args.repeats)
    bench_with_init(L_list, cs, args.repeats)
    bench_step([4, 8, 16], repeats=args.repeats)
    bench_e2e(nc_list, cs, args.repeats)

    print("\n" + "═" * 72)
    print("Summary")
    print("═" * 72)
    print("  inter-chunk scan (Section A):")
    print("    Serial  → Kogge-Stone : 1–2×  (Python overhead dominates at small nc)")
    print("    Serial  → Metal kernel: 3–8×  (single dispatch, nc loop in GPU registers)")
    print("  chunk_parallel_scan_with_init (Section B):")
    print("    f32 cumsum removes bf16 drift — same speed, more accurate for long seqs")
    print("  single-step SSM (Section C):")
    print("    f32 intermediate: no per-step gain, but prevents 512-step mantissa erosion")
    print("  end-to-end (Section D):")
    print("    Metal inter-chunk kernel is now the live path in chunk_parallel_scan_mlx")


if __name__ == "__main__":
    main()
