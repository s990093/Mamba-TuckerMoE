#!/usr/bin/env python3
"""
Microbenchmark: end-to-end Tucker MoE slice (B=1) — fused Metal vs split MLX.

User ``moe_fused_all_in_one``-style shaders vs this repo:

  - ``thread float z[256]`` + ``threadgroup_barrier``: ``thread`` memory is **private** per
    lane; sharing ``z[r]`` across lanes needs **threadgroup** (or device) scratch, not a
    barrier between unrelated private arrays.
  - Top-K with ``for (e = tid; e < E; e += 256)`` + local insertion: each lane only sees a
    **subset** of experts unless you add a **reduction** phase; global Top-K matches
    ``scores = z @ G`` with ``G`` shaped ``(r3, E)``, then ``topk`` on the ``E`` scores.
  - ``G[i * E + e]`` is **[r3][E]** layout (``G_T``); production routing is ``Linear(d→E)``
    on **x** in ``mlx_hybrid_infer.TuckerMoE``, not ``z`` dot a fixed ``G``.
  - Stage 3 sums experts **without** softmax weights; production uses ``top_k_probs``.

Benchmarks: production-style MLX router path, z·G-router MLX path (correct global top-k),
and the scalar fused Metal kernel from earlier.
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx


def fused_tucker_moe_e2e_metal(
    x: mx.array,
    u_in_w: mx.array,
    core_bf16: mx.array,
    u_out_rowmajor_T: mx.array,
    expert_ids: mx.array,
    probs: mx.array,
) -> mx.array:
    """
    Single-kernel fused path for B=1, K experts.

    x: (d,) bf16/fp32 — single token flattened
    u_in_w: (r3, d) as MLX Linear weight (bf16/fp32)
    core_bf16: (E, r3, r2) bf16
    u_out_rowmajor_T: (r2, d_out) bf16/fp32 — transpose of MLX U_out.weight so U[j,o] = W[o,j]
    expert_ids: (K,) int32
    probs: (K,) fp32/bf16
    returns: (d_out,)
    """
    d = int(x.shape[0])
    r3, d_in = u_in_w.shape
    assert d_in == d, (d_in, d)
    E, cr3, r2 = core_bf16.shape
    assert cr3 == r3, (cr3, r3)
    r2t, d_out = u_out_rowmajor_T.shape
    assert r2t == r2, (r2t, r2)
    K = int(expert_ids.shape[0])

    source = r"""
        // MLX treats grid.xyz as ranges for thread_position_in_grid (see MLX exp example).
        // grid=(D_OUT,1,1): one logical thread per output element.
        uint o = thread_position_in_grid.x;
        if (o >= D_OUT) {
            return;
        }

        float z_local[R3];
        for (uint r = 0; r < R3; ++r) {
            float acc_z = 0.0f;
            uint base = r * D;
            for (uint i = 0; i < D; ++i) {
                acc_z += float(x[i]) * float(u_in[base + i]);
            }
            z_local[r] = acc_z;
        }

        float y = 0.0f;
        for (uint k = 0; k < K; ++k) {
            int e_rd = expert_ids[k];
            if (e_rd < 0 || uint(e_rd) >= E) {
                continue;
            }
            uint e = uint(e_rd);
            float pk = float(probs[k]);
            float acc_k = 0.0f;
            for (uint s = 0; s < R2; ++s) {
                float zdot = 0.0f;
                uint r = 0;
                for (; r + 3 < R3; r += 4) {
                    uint i0 = e * R3 * R2 + r * R2 + s;
                    uint i1 = e * R3 * R2 + (r + 1) * R2 + s;
                    uint i2 = e * R3 * R2 + (r + 2) * R2 + s;
                    uint i3 = e * R3 * R2 + (r + 3) * R2 + s;
                    zdot += z_local[r] * float(core[i0]);
                    zdot += z_local[r + 1] * float(core[i1]);
                    zdot += z_local[r + 2] * float(core[i2]);
                    zdot += z_local[r + 3] * float(core[i3]);
                }
                for (; r < R3; ++r) {
                    uint ix = e * R3 * R2 + r * R2 + s;
                    zdot += z_local[r] * float(core[ix]);
                }
                acc_k += zdot * float(u_out[s * D_OUT + o]);
            }
            y += pk * acc_k;
        }
        out[o] = T(y);
    """

    header = (
        f"constant uint D={d}; constant uint R3={r3}; constant uint R2={r2}; "
        f"constant uint D_OUT={d_out}; constant uint E={E}; constant uint K={K};"
    )

    kernel = mx.fast.metal_kernel(
        name=f"tucker_moe_e2e_fused_d{d}_r3{r3}_r2{r2}_o{d_out}_k{K}",
        input_names=["x", "u_in", "core", "u_out", "expert_ids", "probs"],
        output_names=["out"],
        source=source,
        header=header,
    )

    out = kernel(
        inputs=[
            x.reshape(1, -1)[0],
            u_in_w,
            mx.flatten(core_bf16),
            u_out_rowmajor_T,
            expert_ids.astype(mx.int32),
            probs,
        ],
        template=[("T", x.dtype)],
        grid=(max(d_out, 1), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(d_out,)],
        output_dtypes=[x.dtype],
    )[0]
    return out


def reference_mlx_tucker_from_z(
    z: mx.array,
    core: mx.array,
    u_out_w: mx.array,
    expert_ids: mx.array,
    probs: mx.array,
) -> mx.array:
    """Given latent z (r3,), apply weighted expert cores and U_out (MLX layout)."""
    _, r3, r2 = core.shape
    K = int(expert_ids.shape[0])
    accum = mx.zeros((r3, r2), dtype=z.dtype)
    for k in range(K):
        e = int(expert_ids[k].item())
        accum = accum + probs[k] * core[e]
    h = mx.matmul(z.reshape(1, -1), accum).reshape(-1)
    return mx.matmul(h.reshape(1, -1), u_out_w.T).reshape(-1)


def reference_mlx_b1(
    x: mx.array,
    u_in_w: mx.array,
    core: mx.array,
    u_out_w: mx.array,
    expert_ids: mx.array,
    probs: mx.array,
) -> mx.array:
    """B=1: z = x @ U_in.T, then Tucker with given experts and probs."""
    z = mx.matmul(x.reshape(1, -1), u_in_w.T).reshape(-1)
    return reference_mlx_tucker_from_z(z, core, u_out_w, expert_ids, probs)


def reference_mlx_production_router_tucker(
    x: mx.array,
    router_w: mx.array,
    u_in_w: mx.array,
    core: mx.array,
    u_out_w: mx.array,
    K: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Same structure as ``mlx_hybrid_infer.TuckerMoE`` for B=1 (minus tanh/temperature):
    router logits from x, softmax on Top-K slice, weighted Tucker.
    ``router_w`` shape (E, d) like ``nn.Linear(d, E).weight``.
    Returns (y, expert_ids, probs).
    """
    logits = mx.matmul(x.reshape(1, -1), router_w.T).reshape(-1)
    expert_ids = mx.argsort(-logits)[:K].astype(mx.int32)
    sel_logits = logits[expert_ids]
    probs = mx.softmax(sel_logits.astype(mx.float32))
    z = mx.matmul(x.reshape(1, -1), u_in_w.T).reshape(-1)
    y = reference_mlx_tucker_from_z(z, core, u_out_w, expert_ids, probs)
    return y, expert_ids, probs


def reference_mlx_z_dot_g_router_tucker(
    x: mx.array,
    u_in_w: mx.array,
    G_r3_E: mx.array,
    core: mx.array,
    u_out_w: mx.array,
    K: int,
) -> tuple[mx.array, mx.array]:
    """
    User-kernel-style routing: scores_e = z · G[:, e] with G shape (r3, E), row-major
    ``G[i*E+e]``. Top-K experts, **equal weights** (no softmax) like the user's Stage 3.
    Returns (y, expert_ids).
    """
    z = mx.matmul(x.reshape(1, -1), u_in_w.T).reshape(-1)
    scores = mx.matmul(z.reshape(1, -1), G_r3_E).reshape(-1)
    expert_ids = mx.argsort(-scores)[:K].astype(mx.int32)
    probs = mx.ones((K,), dtype=mx.float32)
    y = reference_mlx_tucker_from_z(z, core, u_out_w, expert_ids, probs)
    return y, expert_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--r3", type=int, default=256)
    ap.add_argument("--r2", type=int, default=512)
    ap.add_argument("--d-out", type=int, default=4608)
    ap.add_argument("--e", type=int, default=8)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()

    dt = mx.bfloat16 if args.dtype == "bf16" else mx.float32
    d, r3, r2, d_out, E, K = args.d, args.r3, args.r2, args.d_out, args.e, args.k

    mx.random.seed(0)
    x = mx.random.normal((d,)).astype(dt)
    u_in_w = mx.random.normal((r3, d)).astype(dt)
    core = mx.random.normal((E, r3, r2)).astype(dt)
    u_out_w = mx.random.normal((d_out, r2)).astype(dt)
    u_out_T = mx.transpose(u_out_w)  # (r2, d_out) row-major matches kernel
    router_w = mx.random.normal((E, d)).astype(dt)
    G_r3_E = mx.random.normal((r3, E)).astype(dt)

    y_prod, eid_prod, pr_prod = reference_mlx_production_router_tucker(
        x, router_w, u_in_w, core, u_out_w, K
    )
    y_zg, _ = reference_mlx_z_dot_g_router_tucker(
        x, u_in_w, G_r3_E, core, u_out_w, K
    )
    y_m = fused_tucker_moe_e2e_metal(
        x, u_in_w, core, u_out_T, eid_prod, pr_prod
    )
    mx.eval(y_prod, y_zg, y_m)
    y_rf = reference_mlx_tucker_from_z(
        mx.matmul(x.reshape(1, -1), u_in_w.T).reshape(-1),
        core,
        u_out_w,
        eid_prod,
        pr_prod,
    ).astype(mx.float32)
    y_mf = y_m.astype(mx.float32)
    err = float(mx.abs(y_rf - y_mf).max().item())
    rel = float((mx.abs(y_rf - y_mf) / (mx.abs(y_rf) + 1e-6)).max().item())
    print(f"Metal fused vs MLX Tucker (same prod router ids): max_abs_err={err:.4f} max_rel_err={rel:.4f}")
    if dt == mx.bfloat16 and rel > 0.35:
        print("warning: bf16 fused vs MLX Tucker diverged (expected some gap)")
    elif dt == mx.float32 and err > 0.05:
        print("warning: fp32 sanity check failed")

    for _ in range(10):
        mx.eval(
            reference_mlx_production_router_tucker(
                x, router_w, u_in_w, core, u_out_w, K
            )[0],
            reference_mlx_z_dot_g_router_tucker(
                x, u_in_w, G_r3_E, core, u_out_w, K
            )[0],
            fused_tucker_moe_e2e_metal(
                x, u_in_w, core, u_out_T, eid_prod, pr_prod
            ),
        )

    def t_production():
        return reference_mlx_production_router_tucker(
            x, router_w, u_in_w, core, u_out_w, K
        )[0]

    def t_z_dot_g():
        return reference_mlx_z_dot_g_router_tucker(
            x, u_in_w, G_r3_E, core, u_out_w, K
        )[0]

    def t_fused_only():
        return fused_tucker_moe_e2e_metal(
            x, u_in_w, core, u_out_T, eid_prod, pr_prod
        )

    t0 = time.perf_counter()
    for _ in range(args.trials):
        mx.eval(t_production())
    ms_prod = (time.perf_counter() - t0) * 1000 / args.trials

    t1 = time.perf_counter()
    for _ in range(args.trials):
        mx.eval(t_z_dot_g())
    ms_zg = (time.perf_counter() - t1) * 1000 / args.trials

    t2 = time.perf_counter()
    for _ in range(args.trials):
        mx.eval(t_fused_only())
    ms_fused = (time.perf_counter() - t2) * 1000 / args.trials

    print(
        f"shapes B=1 d={d} r3={r3} r2={r2} d_out={d_out} E={E} K={K} dtype={args.dtype}"
    )
    print(f"MLX production (Linear router on x + softmax Top-K + weighted Tucker): {ms_prod:.4f} ms")
    print(
        f"MLX z·G router (scores z@G, [r3][E] layout, Top-K, unweighted Tucker): {ms_zg:.4f} ms"
    )
    print(
        f"Metal scalar fused (Tucker only; same ids/probs as frozen prod step):     {ms_fused:.4f} ms"
    )
    print(
        f"  → fused vs production Tucker+router: {ms_prod/ms_fused:.2f}x (often <<1; GEMMs win)"
    )
    print()
    print(
        "``moe_fused_all_in_one`` must use threadgroup z, global top-k on z@G (or a reduction), "
        "and match whether you want production Linear(d→E) or latent z·G routing."
    )


if __name__ == "__main__":
    main()
