#include <metal_stdlib>
#include <metal_compute>
using namespace metal;

constant uint TILE_M = 32;
constant uint PAD_FLOAT = 8;
constant uint ROW_STRIDE = TILE_M + PAD_FLOAT; // 40

constant uint R3 [[function_constant(0)]];
constant uint R2 [[function_constant(1)]];
constant uint E [[function_constant(2)]];
constant uint K [[function_constant(3)]];

inline void copy_expert_tile_sync(
    threadgroup float* dst,
    const device float* src,
    uint row_offset,
    uint col_offset,
    uint lane
) {
    #pragma unroll
    for (uint i = 0; i < TILE_M; i += 4) {
        uint local_row = (lane / 8) + i;
        uint local_col = (lane % 8) * 4;
        uint src_idx = (row_offset + local_row) * R2 + (col_offset + local_col);
        uint dst_idx = local_row * ROW_STRIDE + local_col;
        ((threadgroup float4*)(&dst[dst_idx]))[0] =
            ((const device float4*)(&src[src_idx]))[0];
    }
}

inline void copy_expert_tile_async(
    threadgroup float* dst,
    const device float* src,
    uint row_offset,
    uint col_offset,
    uint lane
) {
    // Fallback path: this toolchain doesn't expose simdgroup_async_copy.
    // Keep the same API so host benchmarking remains stable.
    copy_expert_tile_sync(dst, src, row_offset, col_offset, lane);
}

kernel void fused_latent_moe_sync_partial(
    const device float* x_shared [[buffer(0)]],
    const device float* G_experts [[buffer(1)]],
    const device uint* expert_indices [[buffer(2)]],
    device float* partial_out [[buffer(3)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    uint expert_slot = tg_pos.z;
    uint col_block = tg_pos.x * TILE_M;
    if (expert_slot >= K) return;
    uint expert_id = expert_indices[expert_slot];
    if (expert_id >= E) return;

    const device float* expert_base = G_experts + expert_id * R3 * R2;
    threadgroup float stage0[TILE_M * ROW_STRIDE];
    threadgroup float stage1[TILE_M * ROW_STRIDE];
    threadgroup float* stages[2] = {stage0, stage1};

    float acc = 0.0f;
    uint num_tiles = R3 / TILE_M;
    uint stage_idx = 0;

    copy_expert_tile_sync(stages[0], expert_base, 0, col_block, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < num_tiles; ++t) {
        uint next_t = t + 1;
        uint next_stage = (stage_idx + 1) & 1;
        if (next_t < num_tiles) {
            copy_expert_tile_sync(stages[next_stage], expert_base, next_t * TILE_M, col_block, lane);
        }

        #pragma unroll
        for (uint r = 0; r < TILE_M; ++r) {
            acc += x_shared[t * TILE_M + r] * stages[stage_idx][r * ROW_STRIDE + lane];
        }

        if (next_t < num_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        stage_idx = next_stage;
    }

    partial_out[expert_slot * R2 + col_block + lane] = acc;
}

kernel void fused_latent_moe_async_partial(
    const device float* x_shared [[buffer(0)]],
    const device float* G_experts [[buffer(1)]],
    const device uint* expert_indices [[buffer(2)]],
    device float* partial_out [[buffer(3)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    uint expert_slot = tg_pos.z;
    uint col_block = tg_pos.x * TILE_M;
    if (expert_slot >= K) return;
    uint expert_id = expert_indices[expert_slot];
    if (expert_id >= E) return;

    const device float* expert_base = G_experts + expert_id * R3 * R2;
    threadgroup float stage0[TILE_M * ROW_STRIDE];
    threadgroup float stage1[TILE_M * ROW_STRIDE];
    threadgroup float* stages[2] = {stage0, stage1};

    float acc = 0.0f;
    uint num_tiles = R3 / TILE_M;
    uint stage_idx = 0;

    copy_expert_tile_async(stages[0], expert_base, 0, col_block, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < num_tiles; ++t) {
        uint next_t = t + 1;
        uint next_stage = (stage_idx + 1) & 1;
        if (next_t < num_tiles) {
            copy_expert_tile_async(stages[next_stage], expert_base, next_t * TILE_M, col_block, lane);
        }

        #pragma unroll
        for (uint r = 0; r < TILE_M; ++r) {
            acc += x_shared[t * TILE_M + r] * stages[stage_idx][r * ROW_STRIDE + lane];
        }

        if (next_t < num_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        stage_idx = next_stage;
    }

    partial_out[expert_slot * R2 + col_block + lane] = acc;
}

