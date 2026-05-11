// ultimate_gate_silu_bf16.metal
// =====================================================================
// 融合 GateSiLU + LayerScale Elementwise Kernel（BF16）
//
// 融合點：
//   1. GateSiLU：out = norm_y * sigmoid(z) * z
//   2. LayerScale residual：residual += gamma * out
//
// 優化策略：
//   1. 向量化 bfloat4 讀寫（每次 8 bytes）
//   2. 融合兩個 elementwise 為單一 pass（省一次 HBM 往返）
//   3. SIMD-group 合併存取，無 Bank Conflict
// =====================================================================

#include <metal_stdlib>
using namespace metal;

// ───────────────── Kernel 1：純 GateSiLU ────────────────────────
// out = norm_y * silu(z) = norm_y * z * sigmoid(z)

kernel void gate_silu_bf16(
    const device bfloat*  norm_y    [[buffer(0)]],  // [N,] — pre_gate_norm 輸出
    const device bfloat*  z         [[buffer(1)]],  // [N,] — gate
    device       bfloat*  out       [[buffer(2)]],  // [N,]
    constant     uint&    N         [[buffer(3)]],
    uint lane [[thread_position_in_grid]]
) {
    if (lane >= N) return;
    float ny = float(norm_y[lane]);
    float zv = float(z[lane]);
    float silu_z = zv * (1.0f / (1.0f + metal::exp(-zv)));
    out[lane] = bfloat(ny * silu_z);
}


// ───────────────── Kernel 2：向量化 GateSiLU（bfloat4）──────────
// 每個 lane 處理 4 個元素，吞吐量 4x

kernel void gate_silu_vec4_bf16(
    const device bfloat*  norm_y    [[buffer(0)]],  // [N,]
    const device bfloat*  z         [[buffer(1)]],  // [N,]
    device       bfloat*  out       [[buffer(2)]],  // [N,]
    constant     uint&    N_div4    [[buffer(3)]],  // N/4
    uint lane [[thread_position_in_grid]]
) {
    if (lane >= N_div4) return;

    bfloat4 ny_bf = ((const device bfloat4*)(norm_y))[lane];
    bfloat4 z_bf  = ((const device bfloat4*)(z))[lane];

    float4 ny = float4(ny_bf);
    float4 zv = float4(z_bf);

    // SiLU(z) = z * sigmoid(z)
    float4 sig_z = 1.0f / (1.0f + metal::exp(-zv));
    float4 result = ny * zv * sig_z;

    ((device bfloat4*)(out))[lane] = bfloat4(result);
}


// ───────────────── Kernel 3：融合 GateSiLU + LayerScale Residual ─
// 完整融合版本：一次讀寫完成 GateSiLU 和殘差加法
// residual += gamma * (norm_y * silu(z))
// 節省 2 次 HBM 往返（省去寫 gate_out 和讀 gate_out）

kernel void gate_silu_residual_bf16(
    const device bfloat*  norm_y    [[buffer(0)]],  // [N,]
    const device bfloat*  z         [[buffer(1)]],  // [N,]
    const device bfloat*  gamma     [[buffer(2)]],  // [D,] — LayerScale gamma
    device       bfloat*  residual  [[buffer(3)]],  // [N,] — in-place 更新
    constant     uint&    N         [[buffer(4)]],
    constant     uint&    D         [[buffer(5)]],  // gamma 的維度
    uint lane [[thread_position_in_grid]]
) {
    if (lane >= N) return;

    float ny = float(norm_y[lane]);
    float zv = float(z[lane]);
    float gm = float(gamma[lane % D]);

    float silu_z = zv * (1.0f / (1.0f + metal::exp(-zv)));
    float gate_out = ny * silu_z;

    float res = float(residual[lane]);
    residual[lane] = bfloat(res + gm * gate_out);
}


// ───────────────── Kernel 4：向量化融合版（bfloat4）─────────────
// 最高吞吐量版本

kernel void gate_silu_residual_vec4_bf16(
    const device bfloat*  norm_y    [[buffer(0)]],  // [N,]
    const device bfloat*  z         [[buffer(1)]],  // [N,]
    const device bfloat*  gamma     [[buffer(2)]],  // [D,]
    device       bfloat*  residual  [[buffer(3)]],  // [N,] — in-place
    constant     uint&    N_div4    [[buffer(4)]],  // N/4
    constant     uint&    D         [[buffer(5)]],
    uint lane [[thread_position_in_grid]]
) {
    if (lane >= N_div4) return;

    bfloat4 ny_bf  = ((const device bfloat4*)(norm_y))[lane];
    bfloat4 z_bf   = ((const device bfloat4*)(z))[lane];
    bfloat4 res_bf = ((const device bfloat4*)(residual))[lane];

    // gamma 讀取（循環模 D）
    uint base_d = (lane * 4) % D;
    float4 gm = float4(
        float(gamma[(base_d + 0) % D]),
        float(gamma[(base_d + 1) % D]),
        float(gamma[(base_d + 2) % D]),
        float(gamma[(base_d + 3) % D])
    );

    float4 ny  = float4(ny_bf);
    float4 zv  = float4(z_bf);
    float4 res = float4(res_bf);

    float4 sig_z = 1.0f / (1.0f + metal::exp(-zv));
    float4 gate_out = ny * zv * sig_z;
    float4 result   = res + gm * gate_out;

    ((device bfloat4*)(residual))[lane] = bfloat4(result);
}


// ───────────────── Kernel 5：融合 GateSiLU + LayerScale（輸出版）─
// 不做 in-place，而是寫入新輸出張量（用於需要保留舊殘差的情況）

kernel void gate_silu_layer_scale_bf16(
    const device bfloat*  norm_y    [[buffer(0)]],  // [N,]
    const device bfloat*  z         [[buffer(1)]],  // [N,]
    const device bfloat*  gamma     [[buffer(2)]],  // [D,]
    device       bfloat*  out       [[buffer(3)]],  // [N,]
    constant     uint&    N         [[buffer(4)]],
    constant     uint&    D         [[buffer(5)]],
    uint lane [[thread_position_in_grid]]
) {
    if (lane >= N) return;

    float ny = float(norm_y[lane]);
    float zv = float(z[lane]);
    float gm = float(gamma[lane % D]);

    float silu_z = zv / (1.0f + metal::exp(-zv));
    out[lane] = bfloat(gm * ny * silu_z);
}


// ───────────────── Kernel 6：融合 Mamba Gate（特殊版）───────────
// 對應 Mamba3Block 的特定 gate 計算：
// out = pre_gate_norm(y) * silu(z_from_in_proj)
// 支援 y 和 z 有不同的形狀前處理

kernel void mamba_gate_silu_bf16(
    const device bfloat*  y          [[buffer(0)]],  // [B, L, H*P] — SSM 輸出
    const device bfloat*  z          [[buffer(1)]],  // [B, L, H*P] — gate
    const device bfloat*  gamma_norm [[buffer(2)]],  // [H*P,] — pre_gate_norm weight
    device       bfloat*  out        [[buffer(3)]],  // [B, L, H*P]
    constant     uint&    N          [[buffer(4)]],  // B*L*H*P
    constant     uint&    HP         [[buffer(5)]],  // H*P（norm 維度）
    uint lane [[thread_position_in_grid]]
) {
    if (lane >= N) return;

    float yv = float(y[lane]);
    float zv = float(z[lane]);
    float gm = float(gamma_norm[lane % HP]);

    // RMSNorm 近似（這裡每個元素獨立，完整 RMSNorm 需要 simd_sum）
    // 注意：這是簡化版，完整版請使用 rms_norm_linear_bf16.metal
    float ny = yv * gm;  // 假設 norm 已在外層做，這裡只做 scale

    float silu_z = zv / (1.0f + metal::exp(-zv));
    out[lane] = bfloat(ny * silu_z);
}
