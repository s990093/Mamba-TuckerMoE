// ultimate_tucker_moe_v2_split.metal
// =====================================================================
// TuckerMoE 拆解版 — 解決 Register Spilling 問題（備案）
//
// 當 ultimate_tucker_moe_bf16.metal 的完整融合 Kernel 導致 Register Spilling
// 時，改用此「兩個協作 Kernel」方案：
//
// Kernel A: tucker_dense_u_in + router + scatter
//   x[DIM_IN] → x_shared[R3] + expert_id
//
// Kernel B: tucker_grouped_g + u_out
//   x_shared[R3] + G_expert[R2,R3] → y_shared[R2] → y[DIM_OUT]
//
// 優點：每個 kernel 暫存器需求大幅降低，保持高佔用率（>9 TG/core）
// 缺點：需要一次額外的全域記憶體寫入（x_shared 中間結果）
//
// 來源：summary.md §14.1 Register Spilling 分析
// =====================================================================

#include <metal_stdlib>
using namespace metal;

constant float LA_CLAMP = 40.0f;

// ───────────────── Kernel A：U_in 投影 + Router ─────────────────
// 輸入：x[B, DIM_IN]
// 輸出：x_shared[B, R3]，expert_ids[B, TOP_K]（argmax 路由）

kernel void tucker_split_a_project_route(
    const device bfloat*  x              [[buffer(0)]],  // [B, DIM_IN]
    const device bfloat*  U_in           [[buffer(1)]],  // [DIM_IN, R3]
    const device bfloat*  W_router       [[buffer(2)]],  // [DIM_IN, NUM_EXPERTS] 或 [R3, NUM_EXPERTS]
    device       bfloat*  x_shared       [[buffer(3)]],  // [B, R3]（輸出）
    device       uint*    expert_ids     [[buffer(4)]],  // [B, TOP_K]（輸出）
    device       float*   router_logits  [[buffer(5)]],  // [B, NUM_EXPERTS]（可選debug用）
    constant     uint&    B              [[buffer(6)]],
    constant     uint&    DIM_IN         [[buffer(7)]],
    constant     uint&    R3             [[buffer(8)]],
    constant     uint&    NUM_EXPERTS    [[buffer(9)]],
    constant     uint&    TOP_K          [[buffer(10)]],
    uint2 pos [[thread_position_in_grid]]  // x=r3_or_expert, y=batch
) {
    uint b = pos.y;
    if (b >= B) return;

    const device bfloat* x_b = x + b * DIM_IN;

    // ── Step 1: x @ U_in → x_shared[b, R3] ──
    uint r3_out = pos.x;
    if (r3_out < R3) {
        float acc = 0.0f;
        for (uint d = 0; d < DIM_IN; d++) {
            acc += float(x_b[d]) * float(U_in[d * R3 + r3_out]);
        }
        x_shared[b * R3 + r3_out] = bfloat(acc);
    }

    // ── Step 2: Router（只讓 thread 0 做，避免 race）──
    // 這裡使用 x_shared（即 x @ U_in 的結果）作為 router 輸入
    // 注意：需要先讓所有 r3 執行緒完成 Step 1（在 host 端用兩個 dispatch）
}


// ───────────────── Kernel A2：Router（分開的 dispatch）─────────
// 在 Kernel A1 完成後執行，讀取 x_shared 做路由決策

kernel void tucker_split_a2_router(
    const device bfloat*  x_shared       [[buffer(0)]],  // [B, R3]
    const device bfloat*  W_router       [[buffer(1)]],  // [R3, NUM_EXPERTS]
    device       uint*    expert_ids     [[buffer(2)]],  // [B, TOP_K]（輸出）
    device       float*   router_logits  [[buffer(3)]],  // [B, NUM_EXPERTS]（debug）
    constant     uint&    B              [[buffer(4)]],
    constant     uint&    R3             [[buffer(5)]],
    constant     uint&    NUM_EXPERTS    [[buffer(6)]],
    constant     uint&    TOP_K          [[buffer(7)]],
    uint pos [[thread_position_in_grid]]  // b = batch index
) {
    uint b = pos;
    if (b >= B) return;

    const device bfloat* xs = x_shared + b * R3;

    // 計算每個 expert 的 logit
    float logits[64];  // NUM_EXPERTS <= 64
    for (uint e = 0; e < NUM_EXPERTS; e++) {
        float acc = 0.0f;
        for (uint r = 0; r < R3; r++) {
            acc += float(xs[r]) * float(W_router[r * NUM_EXPERTS + e]);
        }
        logits[e] = acc;
        router_logits[b * NUM_EXPERTS + e] = acc;
    }

    // Top-K argmax（簡單選取，不需要 softmax）
    for (uint k = 0; k < TOP_K; k++) {
        float best_val = -1e38f;
        uint  best_idx = 0;
        for (uint e = 0; e < NUM_EXPERTS; e++) {
            if (logits[e] > best_val) {
                best_val = logits[e];
                best_idx = e;
            }
        }
        expert_ids[b * TOP_K + k] = best_idx;
        logits[best_idx] = -1e38f;  // 抹掉已選的
    }
}


// ───────────────── Kernel B：Grouped Tucker GEMV + U_out ────────
// 輸入：x_shared[B, R3]，G_experts_T[E, R2, R3]，expert_ids[B, TOP_K]
// 輸出：y_shared[B, R2]（累加 TOP_K 個專家）→ y[B, DIM_OUT]
//
// 此 Kernel 不做 U_out 投影（留給 Kernel C 做），降低暫存器壓力

kernel void tucker_split_b_grouped_core(
    const device bfloat*  x_shared       [[buffer(0)]],  // [B, R3]
    const device bfloat*  G_experts_T    [[buffer(1)]],  // [E, R2, R3]（轉置版）
    const device uint*    expert_ids     [[buffer(2)]],  // [B, TOP_K]
    device       float*   y_shared       [[buffer(3)]],  // [B, R2]（輸出，float 累加）
    constant     uint&    B              [[buffer(4)]],
    constant     uint&    R3             [[buffer(5)]],
    constant     uint&    R2             [[buffer(6)]],
    constant     uint&    E_total        [[buffer(7)]],
    constant     uint&    TOP_K          [[buffer(8)]],
    uint2 pos [[thread_position_in_grid]]  // x=r2, y=batch
) {
    uint r2_out = pos.x;
    uint b      = pos.y;
    if (r2_out >= R2 || b >= B) return;

    const device bfloat* xs = x_shared + b * R3;

    float acc = 0.0f;
    for (uint k = 0; k < TOP_K; k++) {
        uint eid = expert_ids[b * TOP_K + k];
        if (eid >= E_total) continue;

        // G_T[eid, r2_out, r3=0..R3-1] — 連續的 R3 個元素
        const device bfloat* g_row = G_experts_T + eid * R2 * R3 + r2_out * R3;

        for (uint r3 = 0; r3 < R3; r3++) {
            acc += float(xs[r3]) * float(g_row[r3]);
        }
    }

    y_shared[b * R2 + r2_out] = acc;
}


// ───────────────── Kernel C：U_out 升維投影 ─────────────────────
// 輸入：y_shared[B, R2]
// 輸出：y[B, DIM_OUT]（最終輸出）

kernel void tucker_split_c_project_out(
    const device float*   y_shared       [[buffer(0)]],  // [B, R2]
    const device bfloat*  U_out          [[buffer(1)]],  // [R2, DIM_OUT]
    device       bfloat*  y_out          [[buffer(2)]],  // [B, DIM_OUT]
    constant     uint&    B              [[buffer(3)]],
    constant     uint&    R2             [[buffer(4)]],
    constant     uint&    DIM_OUT        [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]  // x=dim_out, y=batch
) {
    uint d_out = pos.x;
    uint b     = pos.y;
    if (d_out >= DIM_OUT || b >= B) return;

    const device float*   ys    = y_shared + b * R2;
    const device bfloat*  u_col = U_out + d_out;  // U_out[r2, d_out]，stride=DIM_OUT

    float acc = 0.0f;
    for (uint r2 = 0; r2 < R2; r2++) {
        acc += ys[r2] * float(u_col[r2 * DIM_OUT]);
    }

    y_out[b * DIM_OUT + d_out] = bfloat(acc);
}


// ───────────────── Kernel D：BatchNorm-Fused RMSNorm（殘差版）──
// 在拆解版中，殘差加法單獨處理，以保持各 Kernel 暫存器在 limit 內

kernel void tucker_split_d_residual_add(
    const device bfloat*  residual_in    [[buffer(0)]],  // [B, DIM_OUT]
    const device bfloat*  y_out          [[buffer(1)]],  // [B, DIM_OUT]
    const device bfloat*  layer_scale    [[buffer(2)]],  // [DIM_OUT,]
    device       bfloat*  residual_out   [[buffer(3)]],  // [B, DIM_OUT]（in-place 可）
    constant     uint&    N              [[buffer(4)]],  // B * DIM_OUT
    uint lane [[thread_position_in_grid]]
) {
    if (lane >= N) return;
    float scale = float(layer_scale[lane % (N / 1)]);  // 簡化：假設 B=1
    residual_out[lane] = bfloat(float(residual_in[lane]) + scale * float(y_out[lane]));
}
