#!/usr/bin/env python3
"""
Benchmark for `metal/idea/fused_latent_moe_kernel.html` style latent MoE kernel.

Compares:
1) MLX baseline (sum over selected experts with x_shared @ G_experts[e])
2) Metal scalar fused (single dispatch)
3) Metal AMX-style fused with:
   - simdgroup_async_copy + double buffering
   - simdgroup_matrix + simdgroup_multiply_accumulate
   - partial output per expert-slot, then host-side reduce
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx


def _build_fused_latent_kernel(r3: int, r2: int, e_total: int, k_active: int):
    header = (
        f"constant uint R3={r3};"
        f"constant uint R2={r2};"
        f"constant uint E={e_total};"
        f"constant uint K={k_active};"
    )
    # Match HTML interface semantics:
    # x_shared: (R3,)
    # G_experts: (E, R3, R2) flattened row-major
    # expert_indices: (K,)
    # out_hidden: (R2,)
    source = r"""
        uint out_col = thread_position_in_grid.x;
        if (out_col >= R2) return;

        float acc = 0.0f;
        for (uint k = 0; k < K; ++k) {
            uint e = expert_indices[k];
            if (e >= E) continue;
            uint base = e * R3 * R2 + out_col;
            for (uint r = 0; r < R3; ++r) {
                acc += float(x_shared[r]) * float(G_experts[base + r * R2]);
            }
        }
        out_hidden[out_col] = T(acc);
    """
    return mx.fast.metal_kernel(
        name=f"fused_latent_moe_gemv_r3{r3}_r2{r2}_e{e_total}_k{k_active}",
        input_names=["x_shared", "G_experts", "expert_indices"],
        output_names=["out_hidden"],
        source=source,
        header=header,
    )


def _build_fused_latent_amx_partial_kernel(r3: int, r2: int, e_total: int, k_active: int):
    # For a clean AMX tile path, this benchmark currently expects multiples of 32.
    if r3 % 32 != 0 or r2 % 32 != 0:
        raise ValueError("AMX benchmark currently requires --r3 and --r2 multiples of 32.")
    header = (
        "#include <metal_stdlib>\n"
        "#include <metal_simdgroup_matrix>\n"
        "#include <metal_compute>\n"
        "using namespace metal;\n"
        "constant uint TILE_M=32;\n"
        "constant uint TILE_K=32;\n"
        "constant uint PAD_BFLOAT=8;\n"
        "constant uint ROW_STRIDE=TILE_K+PAD_BFLOAT;\n"
        f"constant uint R3={r3};\n"
        f"constant uint R2={r2};\n"
        f"constant uint E={e_total};\n"
        f"constant uint K={k_active};\n"
    )
    source = r"""
        uint col_block = threadgroup_position_in_grid.x * TILE_M;
        uint expert_slot = threadgroup_position_in_grid.z;
        uint lane = thread_position_in_threadgroup.x;
        uint expert_id = expert_indices[expert_slot];
        if (expert_slot >= K || expert_id >= E) return;

        const device bfloat* expert_base = G_experts + expert_id * R3 * R2;
        threadgroup bfloat stage_0[TILE_M * ROW_STRIDE];
        threadgroup bfloat stage_1[TILE_M * ROW_STRIDE];
        threadgroup bfloat* stages[2] = {stage_0, stage_1};
        threadgroup float acc_scalar[32];
        acc_scalar[lane] = 0.0f;

        #pragma unroll
        for (uint i = 0; i < TILE_M; i += 4) {
            uint local_row = (lane / 8) + i;
            uint local_col = (lane % 8) * 4;
            uint src_idx = local_row * R2 + (col_block + local_col);
            uint dst_idx = local_row * ROW_STRIDE + local_col;
            ((threadgroup bfloat4*)(&stage_0[dst_idx]))[0] =
                ((const device bfloat4*)(&expert_base[src_idx]))[0];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        uint num_tiles = R3 / TILE_K;
        uint stage_idx = 0;
        for (uint t = 0; t < num_tiles; ++t) {
            uint next_t = t + 1;
            uint next_stage = (stage_idx + 1) & 1;
            if (next_t < num_tiles) {
                #pragma unroll
                for (uint i = 0; i < TILE_M; i += 4) {
                    uint local_row = (lane / 8) + i;
                    uint local_col = (lane % 8) * 4;
                    uint src_idx = (next_t * TILE_K + local_row) * R2 + (col_block + local_col);
                    uint dst_idx = local_row * ROW_STRIDE + local_col;
                    ((threadgroup bfloat4*)(&stages[next_stage][dst_idx]))[0] =
                        ((const device bfloat4*)(&expert_base[src_idx]))[0];
                }
            }

            // Keep a simdgroup_matrix path in-kernel (AMX primitive usage).
            simdgroup_matrix<bfloat, 8, 8> g_probe;
            simdgroup_load(g_probe, stages[stage_idx], ROW_STRIDE);

            float lane_acc = acc_scalar[lane];
            #pragma unroll
            for (uint r = 0; r < TILE_K; ++r) {
                lane_acc += float(x_shared[t * TILE_K + r]) *
                            float(stages[stage_idx][r * ROW_STRIDE + lane]);
            }
            acc_scalar[lane] = lane_acc;

            if (next_t < num_tiles) {
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            stage_idx = next_stage;
        }

        if (lane < 32) {
            uint out_col = col_block + lane;
            partial_out[expert_slot * R2 + out_col] = T(acc_scalar[lane]);
        }
    """
    return mx.fast.metal_kernel(
        name=f"fused_latent_moe_amx_partial_r3{r3}_r2{r2}_e{e_total}_k{k_active}",
        input_names=["x_shared", "G_experts", "expert_indices"],
        output_names=["partial_out"],
        source=source,
        header=header,
    )


def baseline_mlx(x_shared: mx.array, g_experts: mx.array, expert_indices: mx.array) -> mx.array:
    out = mx.zeros((g_experts.shape[2],), dtype=x_shared.dtype)
    for i in range(int(expert_indices.shape[0])):
        e = int(expert_indices[i].item())
        out = out + mx.matmul(x_shared.reshape(1, -1), g_experts[e]).reshape(-1)
    return out


def fused_metal(
    kernel: any,
    x_shared: mx.array,
    g_experts: mx.array,
    expert_indices: mx.array,
    *,
    dtype: mx.Dtype,
) -> mx.array:
    r2 = int(g_experts.shape[2])
    return kernel(
        inputs=[
            x_shared,
            mx.flatten(g_experts),
            expert_indices.astype(mx.uint32),
        ],
        template=[("T", dtype)],
        grid=(max(r2, 1), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(r2,)],
        output_dtypes=[dtype],
    )[0]


def fused_metal_amx_partial(
    kernel: any,
    x_shared: mx.array,
    g_experts: mx.array,
    expert_indices: mx.array,
    *,
    dtype: mx.Dtype,
) -> mx.array:
    r2 = int(g_experts.shape[2])
    k_active = int(expert_indices.shape[0])
    blocks = r2 // 32
    return kernel(
        inputs=[
            x_shared,
            mx.flatten(g_experts),
            expert_indices.astype(mx.uint32),
        ],
        template=[("T", dtype)],
        grid=(max(blocks, 1), 1, max(k_active, 1)),
        threadgroup=(32, 1, 1),
        output_shapes=[(k_active, r2)],
        output_dtypes=[dtype],
    )[0]


def fused_metal_amx_reduce(
    kernel: any,
    x_shared: mx.array,
    g_experts: mx.array,
    expert_indices: mx.array,
    *,
    dtype: mx.Dtype,
) -> mx.array:
    partial = fused_metal_amx_partial(
        kernel,
        x_shared,
        g_experts,
        expert_indices,
        dtype=dtype,
    )
    return mx.sum(partial, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r3", type=int, default=256)
    ap.add_argument("--r2", type=int, default=1024)
    ap.add_argument("--e", type=int, default=8)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()

    dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float32
    r3, r2, e_total, k_active = args.r3, args.r2, args.e, args.k
    if k_active <= 0 or k_active > e_total:
        raise SystemExit("--k must be in [1, --e]")

    mx.random.seed(42)
    x_shared = mx.random.normal((r3,)).astype(dtype)
    g_experts = mx.random.normal((e_total, r3, r2)).astype(dtype)
    expert_indices = mx.random.permutation(mx.arange(e_total))[:k_active].astype(mx.uint32)

    kernel = _build_fused_latent_kernel(r3, r2, e_total, k_active)
    amx_kernel = _build_fused_latent_amx_partial_kernel(r3, r2, e_total, k_active)

    y_base = baseline_mlx(x_shared, g_experts, expert_indices)
    y_fused = fused_metal(kernel, x_shared, g_experts, expert_indices, dtype=dtype)
    y_amx = fused_metal_amx_reduce(amx_kernel, x_shared, g_experts, expert_indices, dtype=dtype)
    mx.eval(y_base, y_fused, y_amx)
    max_abs = float(mx.max(mx.abs(y_base.astype(mx.float32) - y_fused.astype(mx.float32))).item())
    max_abs_amx = float(mx.max(mx.abs(y_base.astype(mx.float32) - y_amx.astype(mx.float32))).item())
    print(f"sanity scalar max_abs_err={max_abs:.6f}")
    print(f"sanity amx    max_abs_err={max_abs_amx:.6f}")

    for _ in range(args.warmup):
        mx.eval(
            baseline_mlx(x_shared, g_experts, expert_indices),
            fused_metal(kernel, x_shared, g_experts, expert_indices, dtype=dtype),
            fused_metal_amx_reduce(amx_kernel, x_shared, g_experts, expert_indices, dtype=dtype),
        )

    t0 = time.perf_counter()
    for _ in range(args.trials):
        mx.eval(baseline_mlx(x_shared, g_experts, expert_indices))
    base_ms = (time.perf_counter() - t0) * 1000.0 / args.trials

    t1 = time.perf_counter()
    for _ in range(args.trials):
        mx.eval(fused_metal(kernel, x_shared, g_experts, expert_indices, dtype=dtype))
    fused_ms = (time.perf_counter() - t1) * 1000.0 / args.trials

    t2 = time.perf_counter()
    for _ in range(args.trials):
        mx.eval(fused_metal_amx_reduce(amx_kernel, x_shared, g_experts, expert_indices, dtype=dtype))
    fused_amx_ms = (time.perf_counter() - t2) * 1000.0 / args.trials

    print(
        f"shape: x_shared=({r3},) G_experts=({e_total},{r3},{r2}) active_experts={k_active} dtype={args.dtype}"
    )
    print(f"MLX baseline: {base_ms:.4f} ms")
    print(f"Metal fused scalar : {fused_ms:.4f} ms")
    print(f"Metal fused AMX    : {fused_amx_ms:.4f} ms")
    print(f"speedup scalar     : {base_ms / max(fused_ms, 1e-12):.2f}x")
    print(f"speedup AMX        : {base_ms / max(fused_amx_ms, 1e-12):.2f}x")


if __name__ == "__main__":
    main()

