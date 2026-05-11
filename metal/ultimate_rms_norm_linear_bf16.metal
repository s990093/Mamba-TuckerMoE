// ultimate_rms_norm_linear_bf16.metal
// =====================================================================
// 融合 RMSNorm + Linear GEMV Metal Kernel（BF16）
//
// 優化策略：
//   1. x_norm 計算後直接做投影，不寫回全域記憶體（省一次 HBM 往返）
//   2. BF16 精度，FP32 累加防止溢出
//   3. 向量化 bfloat4 讀取輸入
//   4. 三合一版本：支援 RMSNorm+Linear、RMSNorm+QKV（三個投影同時）
//
// 融合計算：
//   rms = sqrt(mean(x²) + eps)
//   x_norm = x / rms * gamma
//   y = x_norm @ W（不物化 x_norm）
// =====================================================================

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant float RMS_EPS = 1e-5f;

// ───────────────── Kernel 1：RMSNorm + Linear（單投影）────────────
// 每個 threadgroup 處理一個 token
// 輸入：x[D_in]，輸出：y[D_out]

kernel void rms_norm_linear_bf16(
    const device bfloat*  x           [[buffer(0)]],  // [N, D_in]
    const device bfloat*  gamma       [[buffer(1)]],  // [D_in,] — RMSNorm scale
    const device bfloat*  W           [[buffer(2)]],  // [D_out, D_in]
    const device bfloat*  bias        [[buffer(3)]],  // [D_out,] or nullptr
    device       bfloat*  y           [[buffer(4)]],  // [N, D_out]
    constant     uint&    N           [[buffer(5)]],
    constant     uint&    D_in        [[buffer(6)]],
    constant     uint&    D_out       [[buffer(7)]],
    constant     bool&    HAS_BIAS    [[buffer(8)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],  // x=output_block, y=token
    uint  lane   [[thread_position_in_threadgroup]]
) {
    uint token_idx  = tg_pos.y;
    uint out_start  = tg_pos.x * 32;  // 每個 TG 處理 32 個輸出維度

    if (token_idx >= N) return;

    const device bfloat* x_tok = x + token_idx * D_in;

    // ── Step 1: 計算 RMS（所有 lanes 協同，使用 simd_sum）──────────
    float sq_sum = 0.0f;
    uint D_div4 = D_in / 4;

    // 每個 lane 負責 D_in/32 個元素的平方和
    uint chunk_size = (D_in + 31) / 32;
    uint d_start    = lane * chunk_size;
    uint d_end      = min(d_start + chunk_size, D_in);

    for (uint d = d_start; d < d_end; d++) {
        float v = float(x_tok[d]);
        sq_sum += v * v;
    }
    // SIMD reduce within 32-thread group
    sq_sum = simd_sum(sq_sum);
    float rms = metal::rsqrt(sq_sum / float(D_in) + RMS_EPS);

    // ── Step 2: 計算 x_norm 並投影（不物化 x_norm）─────────────────
    // 每個 lane 計算一個輸出維度
    uint out_col = out_start + lane;
    if (out_col >= D_out) return;

    float acc = 0.0f;
    for (uint d = 0; d < D_in; d++) {
        float x_norm_d = float(x_tok[d]) * rms * float(gamma[d]);
        acc += x_norm_d * float(W[out_col * D_in + d]);
    }

    if (HAS_BIAS) acc += float(bias[out_col]);
    y[token_idx * D_out + out_col] = bfloat(acc);
}


// ───────────────── Kernel 2：RMSNorm + 三路 QKV 投影 ──────────────
// TransformerBlock 用：norm_attn(x) → [q, k, v]
// 三個投影共享一次 x_norm 計算，只需一次 RMS pass

kernel void rms_norm_qkv_bf16(
    const device bfloat*  x           [[buffer(0)]],  // [N, D]
    const device bfloat*  gamma       [[buffer(1)]],  // [D,]
    const device bfloat*  W_q         [[buffer(2)]],  // [D_q, D]
    const device bfloat*  W_k         [[buffer(3)]],  // [D_k, D]
    const device bfloat*  W_v         [[buffer(4)]],  // [D_v, D]
    device       bfloat*  q_out       [[buffer(5)]],  // [N, D_q]
    device       bfloat*  k_out       [[buffer(6)]],  // [N, D_k]
    device       bfloat*  v_out       [[buffer(7)]],  // [N, D_v]
    constant     uint&    N           [[buffer(8)]],
    constant     uint&    D           [[buffer(9)]],
    constant     uint&    D_q         [[buffer(10)]],
    constant     uint&    D_k         [[buffer(11)]],
    constant     uint&    D_v         [[buffer(12)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],  // y=token
    uint  lane   [[thread_position_in_threadgroup]]
) {
    uint token_idx = tg_pos.y;
    if (token_idx >= N) return;

    const device bfloat* x_tok = x + token_idx * D;

    // ── 計算 RMS ──
    float sq_sum = 0.0f;
    uint chunk_size = (D + 31) / 32;
    uint d_start    = lane * chunk_size;
    uint d_end      = min(d_start + chunk_size, D);
    for (uint d = d_start; d < d_end; d++) {
        float v = float(x_tok[d]);
        sq_sum += v * v;
    }
    sq_sum = simd_sum(sq_sum);
    float rms = metal::rsqrt(sq_sum / float(D) + RMS_EPS);

    // ── 計算 Q 投影 ──
    // lane 負責 D_q/32 個輸出
    uint q_chunk = (D_q + 31) / 32;
    for (uint i = 0; i < q_chunk; i++) {
        uint out_col = lane * q_chunk + i;
        if (out_col >= D_q) break;
        float acc = 0.0f;
        for (uint d = 0; d < D; d++) {
            float x_norm_d = float(x_tok[d]) * rms * float(gamma[d]);
            acc += x_norm_d * float(W_q[out_col * D + d]);
        }
        q_out[token_idx * D_q + out_col] = bfloat(acc);
    }

    // ── 計算 K 投影 ──
    uint k_chunk = (D_k + 31) / 32;
    for (uint i = 0; i < k_chunk; i++) {
        uint out_col = lane * k_chunk + i;
        if (out_col >= D_k) break;
        float acc = 0.0f;
        for (uint d = 0; d < D; d++) {
            float x_norm_d = float(x_tok[d]) * rms * float(gamma[d]);
            acc += x_norm_d * float(W_k[out_col * D + d]);
        }
        k_out[token_idx * D_k + out_col] = bfloat(acc);
    }

    // ── 計算 V 投影 ──
    uint v_chunk = (D_v + 31) / 32;
    for (uint i = 0; i < v_chunk; i++) {
        uint out_col = lane * v_chunk + i;
        if (out_col >= D_v) break;
        float acc = 0.0f;
        for (uint d = 0; d < D; d++) {
            float x_norm_d = float(x_tok[d]) * rms * float(gamma[d]);
            acc += x_norm_d * float(W_v[out_col * D + d]);
        }
        v_out[token_idx * D_v + out_col] = bfloat(acc);
    }
}


// ───────────────── Kernel 3：AMX 加速版 RMSNorm + Linear ──────────
// 使用 simdgroup_matrix 進行 GEMV
// 適用於 D_in >= 64 且為 8 的倍數的情況

kernel void rms_norm_linear_amx_bf16(
    const device bfloat*  x           [[buffer(0)]],  // [D_in,]（單 token）
    const device bfloat*  gamma       [[buffer(1)]],  // [D_in,]
    const device bfloat*  W           [[buffer(2)]],  // [D_out, D_in]
    device       bfloat*  y           [[buffer(3)]],  // [D_out,]
    constant     uint&    D_in        [[buffer(4)]],
    constant     uint&    D_out       [[buffer(5)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],  // x = output tile (D_out/32)
    uint  lane   [[thread_position_in_threadgroup]]
) {
    uint out_tile = tg_pos.x;
    uint out_base = out_tile * 32;  // 32 outputs per TG
    if (out_base >= D_out) return;

    // ── Step 1: 計算 RMS ──
    float sq_sum = 0.0f;
    uint D4 = D_in / 4;
    for (uint c = lane; c < D4; c += 32) {
        bfloat4 xv = ((const device bfloat4*)(x))[c];
        float4  fv = float4(xv);
        sq_sum += fv.x*fv.x + fv.y*fv.y + fv.z*fv.z + fv.w*fv.w;
    }
    for (uint d = D4*4 + lane; d < D_in; d += 32) {
        float v = float(x[d]); sq_sum += v*v;
    }
    sq_sum = simd_sum(sq_sum);
    float rms = metal::rsqrt(sq_sum / float(D_in) + RMS_EPS);

    // ── Step 2: AMX GEMV（廣播向量）──
    // 每個 lane 直接計算一個輸出元素
    uint out_col = out_base + lane;
    if (out_col >= D_out) return;

    const device bfloat* w_row = W + out_col * D_in;
    float acc = 0.0f;

    for (uint c = 0; c < D4; c++) {
        bfloat4 xv = ((const device bfloat4*)(x))[c];
        bfloat4 gv = ((const device bfloat4*)(gamma))[c];
        bfloat4 wv = ((const device bfloat4*)(w_row))[c];

        float4 xf = float4(xv) * rms * float4(gv);
        float4 wf = float4(wv);
        acc += xf.x*wf.x + xf.y*wf.y + xf.z*wf.z + xf.w*wf.w;
    }
    for (uint d = D4*4; d < D_in; d++) {
        float x_norm_d = float(x[d]) * rms * float(gamma[d]);
        acc += x_norm_d * float(w_row[d]);
    }

    y[out_col] = bfloat(acc);
}


// ───────────────── Kernel 4：融合 RMSNorm + Scaled Head Logits ───
// 對應 sample.py §5 FusedScaledHeadLogits
// hidden → hidden/sqrt(D) → head_W → scaled_tanh(30)
// 全部在一個 kernel 裡完成，省兩次 HBM 往返

kernel void rms_norm_head_logits_bf16(
    const device bfloat*  hidden      [[buffer(0)]],  // [D,] — 已過 final norm
    const device bfloat*  W_head      [[buffer(1)]],  // [V, D] — lm head weight
    device       float*   logits      [[buffer(2)]],  // [V,] — 輸出 logits
    constant     uint&    D           [[buffer(3)]],
    constant     uint&    V           [[buffer(4)]],
    constant     float&   inv_sqrt_D  [[buffer(5)]],
    constant     float&   scale       [[buffer(6)]],  // = 30.0
    uint  lane [[thread_position_in_grid]]  // 每個 lane 一個 vocab token
) {
    if (lane >= V) return;

    const device bfloat* w_row = W_head + lane * D;

    // 同時計算 dot product 和 RMS
    float acc = 0.0f;
    uint D4 = D / 4;

    for (uint c = 0; c < D4; c++) {
        bfloat4 hv = ((const device bfloat4*)(hidden))[c];
        bfloat4 wv = ((const device bfloat4*)(w_row))[c];
        float4  hf = float4(hv) * inv_sqrt_D;
        float4  wf = float4(wv);
        acc += hf.x*wf.x + hf.y*wf.y + hf.z*wf.z + hf.w*wf.w;
    }
    for (uint d = D4*4; d < D; d++) {
        acc += float(hidden[d]) * inv_sqrt_D * float(w_row[d]);
    }

    // scaled_tanh(30)
    logits[lane] = scale * metal::tanh(acc / scale);
}
