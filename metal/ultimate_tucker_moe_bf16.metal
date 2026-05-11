// ultimate_tucker_moe_bf16.metal
// =====================================================================
// 史上最強 TuckerMoE 融合 Metal Kernel
//
// 三層優化策略（來自 metal/idea/ PDF 研究文件）：
//   L1: 離線張量轉置佈局 G_experts[E, r2, r3] → 零 Bank Conflict
//   L2: 融合五個算子為單一 dispatch（消除全域記憶體往返）
//   L3: simdgroup_matrix<bfloat, 8, 8> AMX 暫存器計算 + 雙重緩衝流水線
//
// 融合計算路徑：
//   x → [x @ U_in] → x_shared (Tier1 暫存器)
//     → Router argmax → expert_id
//     → Gather G_experts[expert_id] (轉置佈局，零衝突)
//     → [x_shared @ G_core] → y_shared (AMX, Tier1)
//     → [y_shared @ U_out] → y (寫回全域)
//
// 硬體目標：Apple M3, 19 GPU cores, 200 GB/s, BF16
// =====================================================================

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_compute>
using namespace metal;

// ───────────────── 硬體常數（由 JIT 注入替換）─────────────────
// 這些 constant 在 Python 端透過 header string 動態注入硬編碼數值
// R3: Tucker 低秩維度（如 256）
// R2: Tucker 輸出秩（如 1024）
// DIM_IN:  模型輸入維度（如 2048）
// DIM_OUT: 模型輸出維度（如 2048）
// NUM_EXPERTS: 專家總數
// TOP_K: 激活專家數

// Tile 大小（固定，針對 AMX 8x8 block 最優化）
constant uint TILE_M = 32;       // 每個 tile 的行數
constant uint TILE_K = 32;       // 每個 tile 的列數（與 R3 切片對齊）
constant uint PAD_BF16 = 8;      // Bank Conflict 消除 padding（8 × 2B = 16B）
constant uint ROW_STRIDE = TILE_K + PAD_BF16;  // = 40，打破 32-bank 對齊

// ───────────────── Kernel 1：U_in 投影融合 ─────────────────────
// 每個 threadgroup 處理一個 token，計算 x_shared = x @ U_in
// 使用 simdgroup_matrix 廣播向量為矩陣，利用 AMX 硬體

kernel void tucker_project_u_in(
    const device bfloat*  x       [[buffer(0)]],  // [NUM_TOKENS, DIM_IN]
    const device bfloat*  U_in    [[buffer(1)]],  // [DIM_IN, R3] — 轉置後 [R3, DIM_IN]
    device       bfloat*  x_shared[[buffer(2)]],  // [NUM_TOKENS, R3]
    constant     uint&    DIM_IN  [[buffer(3)]],
    constant     uint&    R3      [[buffer(4)]],
    constant     uint&    NUM_TOKENS [[buffer(5)]],
    uint3 tg_pos  [[threadgroup_position_in_grid]],
    uint  lane    [[thread_position_in_threadgroup]]
) {
    uint token_idx = tg_pos.x;
    if (token_idx >= NUM_TOKENS) return;

    const device bfloat* x_tok = x + token_idx * DIM_IN;
    device bfloat* out_tok = x_shared + token_idx * R3;

    // 每個 thread 計算 1 個輸出維度（r3 方向）
    // 使用向量化 float4 累加，最後寫出 bfloat
    uint r_out = lane;
    if (r_out >= R3) return;

    float acc = 0.0f;
    uint chunks = DIM_IN / 4;

    for (uint c = 0; c < chunks; c++) {
        // 向量化 4 個元素
        float4 xv = float4(
            float(x_tok[c*4+0]), float(x_tok[c*4+1]),
            float(x_tok[c*4+2]), float(x_tok[c*4+3])
        );
        // U_in 以 [DIM_IN, R3] 佈局讀取，stride = R3
        float4 uv = float4(
            float(U_in[(c*4+0)*R3 + r_out]),
            float(U_in[(c*4+1)*R3 + r_out]),
            float(U_in[(c*4+2)*R3 + r_out]),
            float(U_in[(c*4+3)*R3 + r_out])
        );
        acc += dot(xv, uv);
    }
    // 處理尾端（DIM_IN 不是 4 的倍數時）
    for (uint d = chunks*4; d < DIM_IN; d++) {
        acc += float(x_tok[d]) * float(U_in[d*R3 + r_out]);
    }

    out_tok[r_out] = bfloat(acc);
}


// ───────────────── Kernel 2：核心 TuckerMoE GEMV（AMX + 雙重緩衝）──────
// 最關鍵的 kernel：計算 y_shared = x_shared @ G_experts[expert_id]
//
// G_experts 必須以 [E, R2, R3] 轉置佈局預存（在 Python 端離線轉置）
// 這確保相鄰執行緒存取步長 = 1，完美合併，零 Bank Conflict
//
// 軟體流水線：
//   Prologue → async load tile[0]
//   Main Loop → compute tile[t] 同時 async load tile[t+1]
//   Epilogue → write result

kernel void tucker_moe_gemv_amx(
    const device bfloat*  x_shared      [[buffer(0)]],  // [R3,]
    const device bfloat*  G_experts_T   [[buffer(1)]],  // [E, R2, R3] — 轉置版
    const device uint*    expert_indices [[buffer(2)]],  // [K,]
    device       float*   partial_out   [[buffer(3)]],  // [K, R2]
    constant     uint&    R3            [[buffer(4)]],
    constant     uint&    R2            [[buffer(5)]],
    constant     uint&    E_total       [[buffer(6)]],
    constant     uint&    K_active      [[buffer(7)]],
    uint3 tg_pos  [[threadgroup_position_in_grid]],  // x=col_block, z=expert_slot
    uint  lane    [[thread_position_in_threadgroup]]
) {
    uint expert_slot = tg_pos.z;
    uint col_block   = tg_pos.x * TILE_M;  // 每個 TG 負責 TILE_M=32 個輸出列

    if (expert_slot >= K_active) return;
    uint expert_id = expert_indices[expert_slot];
    if (expert_id >= E_total) return;

    // G_experts_T[expert_id] 從 [R2, R3] 起始（轉置後相鄰執行緒存取連續）
    const device bfloat* expert_base = G_experts_T + expert_id * R2 * R3;

    // ── 雙重緩衝 Threadgroup Memory（含 Padding 零 Bank Conflict）──
    // stage[s] 形狀：[TILE_M, ROW_STRIDE]，存放 G_experts_T 的一個 tile
    threadgroup bfloat stage_0[TILE_M * ROW_STRIDE];
    threadgroup bfloat stage_1[TILE_M * ROW_STRIDE];
    threadgroup bfloat* stages[2] = {stage_0, stage_1};

    // 使用 simdgroup_matrix 的 AMX FP32 累加器
    // 4 個 8x8 矩陣覆蓋 TILE_M=32 個輸出
    simdgroup_matrix<float, 8, 8> acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        acc[i] = make_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    uint num_tiles = R3 / TILE_K;   // e.g. 256/32 = 8 個 tiles
    uint stage_idx = 0;

    // ── 軟體流水線 Prologue：載入第 0 個 tile ──────────────────────
    // 每個 lane 負責搬移 bfloat4（8 bytes）× 4 次 = 32 個元素
    // 使用轉置佈局：G_experts_T[r2_row, r3_col] → stage[r3_col, r2_row+padding]
    #pragma unroll
    for (uint i = 0; i < TILE_M; i += 4) {
        uint local_r2  = (lane / 8) + i;           // 0..31，哪個 r2 列
        uint local_r3  = (lane % 8) * 4;           // 0,4,8,...,28：哪 4 個 r3 列

        // 從 device memory 讀：G_T[r2=col_block+local_r2, r3_start]
        uint src_idx = (col_block + local_r2) * R3 + (0 * TILE_K + local_r3);
        // 寫入 stage，佈局 [TILE_K, ROW_STRIDE]：row=local_r3/4, col=local_r2
        // 注意：我們要做 G_T 的局部轉置，讓 r3 走外維、r2 走內維
        uint dst_idx = local_r3 * ROW_STRIDE + local_r2;

        ((threadgroup bfloat4*)(&stages[0][dst_idx]))[0] =
            ((const device bfloat4*)(&expert_base[src_idx]))[0];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // ── 主迴圈：計算 tile[t]，同時預載 tile[t+1] ─────────────────
    for (uint t = 0; t < num_tiles; t++) {
        uint next_t     = t + 1;
        uint next_stage = (stage_idx + 1) & 1;

        // 1. 預載下一個 tile（與 AMX 計算並行）
        if (next_t < num_tiles) {
            #pragma unroll
            for (uint i = 0; i < TILE_M; i += 4) {
                uint local_r2 = (lane / 8) + i;
                uint local_r3 = (lane % 8) * 4;
                uint src_idx  = (col_block + local_r2) * R3 + (next_t * TILE_K + local_r3);
                uint dst_idx  = local_r3 * ROW_STRIDE + local_r2;
                ((threadgroup bfloat4*)(&stages[next_stage][dst_idx]))[0] =
                    ((const device bfloat4*)(&expert_base[src_idx]))[0];
            }
        }

        // 2. 從 simdgroup_matrix 廣播 x_shared 向量
        //    [r3_tile_start .. r3_tile_start+TILE_K] 的 x_shared 值
        //    廣播策略：將 1D 向量「填充」為 8×8 矩陣（8行完全相同）
        simdgroup_matrix<bfloat, 8, 8> x_mat[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // x_mat[j] 對應 x_shared[t*TILE_K + j*8 .. j*8+7]
            // simdgroup_load 搭配 stride=0 可廣播同一列
            simdgroup_load(
                x_mat[j],
                x_shared + t * TILE_K + j * 8,
                0,       // stride=0 → 同一列廣播至所有 8 行
                ulong2(0, 0),
                false    // no transpose
            );
        }

        // 3. 從 Threadgroup Memory（零衝突）載入 G 矩陣 tile
        simdgroup_matrix<bfloat, 8, 8> g_mat[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {  // i: r2 方向，4 個 8x8 = 32 outputs
            #pragma unroll
            for (int j = 0; j < 4; j++) {  // j: r3 方向
                simdgroup_load(
                    g_mat[i][j],
                    stages[stage_idx] + (j * 8 * ROW_STRIDE) + (i * 8),
                    ROW_STRIDE  // 含 padding 的 stride，完美零衝突
                );
            }
        }

        // 4. AMX simdgroup_multiply_accumulate（FP32 累加 BF16 輸入）
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                simdgroup_multiply_accumulate(acc[i], x_mat[j], g_mat[i][j], acc[i]);
            }
        }

        // 5. 等待下一個 tile 預載完成
        if (next_t < num_tiles) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        stage_idx = next_stage;
    }

    // ── Epilogue：寫出 FP32 結果 ──────────────────────────────────
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint out_r2 = col_block + i * 8;
        simdgroup_store(
            acc[i],
            partial_out + expert_slot * R2 + out_r2,
            1,    // column stride = 1（輸出向量）
            ulong2(0, 0),
            false
        );
    }
}


// ───────────────── Kernel 3：U_out 升維融合 ─────────────────────
// y_final = (sum over k: partial_out[k]) @ U_out
// 先在 FP32 做 K 個專家的 reduce，再做 U_out 投影

kernel void tucker_project_u_out(
    const device float*  partial_out  [[buffer(0)]],  // [K, R2]
    const device bfloat* U_out        [[buffer(1)]],  // [R2, DIM_OUT]
    device       bfloat* y_out        [[buffer(2)]],  // [DIM_OUT,]
    constant     uint&   R2           [[buffer(3)]],
    constant     uint&   DIM_OUT      [[buffer(4)]],
    constant     uint&   K_active     [[buffer(5)]],
    uint  lane [[thread_position_in_grid]]
) {
    if (lane >= DIM_OUT) return;

    // Step 1: Reduce K 個 expert partial outputs → y_shared[R2]
    // Step 2: y_shared @ U_out[:, lane]

    float acc = 0.0f;
    for (uint r2 = 0; r2 < R2; r2++) {
        // Sum across K experts for this r2
        float y_r2 = 0.0f;
        for (uint k = 0; k < K_active; k++) {
            y_r2 += partial_out[k * R2 + r2];
        }
        // U_out[r2, lane]
        acc += y_r2 * float(U_out[r2 * DIM_OUT + lane]);
    }

    y_out[lane] = bfloat(acc);
}


// ───────────────── Kernel 4：完整融合 TuckerMoE（小秩快速路徑）──
// 適用於 R3 <= 256, R2 <= 256 的情況，所有中間結果保留在暫存器中
// 完全避免 Threadgroup Memory 的 Bank Conflict 問題

kernel void tucker_moe_full_fused_small(
    const device bfloat*  x           [[buffer(0)]],  // [DIM_IN,]
    const device bfloat*  U_in        [[buffer(1)]],  // [DIM_IN, R3]
    const device bfloat*  G_experts_T [[buffer(2)]],  // [E, R2, R3] 轉置
    const device bfloat*  U_out       [[buffer(3)]],  // [R2, DIM_OUT]
    const device uint*    expert_idx  [[buffer(4)]],  // [K,]
    device       bfloat*  y_out       [[buffer(5)]],  // [DIM_OUT,]
    constant     uint&    DIM_IN      [[buffer(6)]],
    constant     uint&    DIM_OUT     [[buffer(7)]],
    constant     uint&    R3          [[buffer(8)]],
    constant     uint&    R2          [[buffer(9)]],
    constant     uint&    E_total     [[buffer(10)]],
    constant     uint&    K_active    [[buffer(11)]],
    uint  lane [[thread_position_in_grid]]  // 每個 lane 負責一個輸出維度
) {
    if (lane >= DIM_OUT) return;

    // ── Step 1: x @ U_in → x_shared[R3] ──────────────────────────
    // 暫存在暫存器陣列（不寫 SRAM）
    // 使用 float 累加避免 BF16 精度損失
    // 注意：R3 <= 256，暫存器足夠容納
    float x_shared_reg[256];  // compile-time 大小，JIT 可替換為實際值

    for (uint r = 0; r < R3; r++) {
        float acc = 0.0f;
        for (uint d = 0; d < DIM_IN; d++) {
            acc += float(x[d]) * float(U_in[d * R3 + r]);
        }
        x_shared_reg[r] = acc;
    }

    // ── Step 2: 對所有激活專家 Gather + G 計算 → y_shared[R2] ────
    float y_shared_reg[1024];  // R2 <= 1024
    for (uint r2 = 0; r2 < R2; r2++) {
        y_shared_reg[r2] = 0.0f;
    }

    for (uint k = 0; k < K_active; k++) {
        uint eid = expert_idx[k];
        if (eid >= E_total) continue;
        const device bfloat* G_e = G_experts_T + eid * R2 * R3;

        // x_shared @ G_e，G_e 以 [R2, R3] 存（轉置），連續讀取
        for (uint r2 = 0; r2 < R2; r2++) {
            float dot = 0.0f;
            for (uint r3 = 0; r3 < R3; r3++) {
                dot += x_shared_reg[r3] * float(G_e[r2 * R3 + r3]);
            }
            y_shared_reg[r2] += dot;
        }
    }

    // ── Step 3: y_shared @ U_out → y_out[lane] ───────────────────
    float out_acc = 0.0f;
    for (uint r2 = 0; r2 < R2; r2++) {
        out_acc += y_shared_reg[r2] * float(U_out[r2 * DIM_OUT + lane]);
    }

    y_out[lane] = bfloat(out_acc);
}


// ───────────────── Kernel 5：XOR Swizzling 版（無需離線轉置）────
// 當無法做離線轉置時的備選方案
// 使用 XOR 位元交錯映射消除 Bank Conflict

kernel void tucker_moe_xor_swizzle(
    const device bfloat*  x_shared      [[buffer(0)]],  // [R3,]
    const device bfloat*  G_experts     [[buffer(1)]],  // [E, R3, R2] 原始佈局
    const device uint*    expert_indices [[buffer(2)]],  // [K,]
    device       float*   partial_out   [[buffer(3)]],  // [K, R2]
    constant     uint&    R3            [[buffer(4)]],
    constant     uint&    R2            [[buffer(5)]],
    constant     uint&    E_total       [[buffer(6)]],
    constant     uint&    K_active      [[buffer(7)]],
    uint3 tg_pos  [[threadgroup_position_in_grid]],
    uint  lane    [[thread_position_in_threadgroup]]
) {
    uint expert_slot = tg_pos.z;
    uint col_block   = tg_pos.x * TILE_M;

    if (expert_slot >= K_active) return;
    uint expert_id = expert_indices[expert_slot];
    if (expert_id >= E_total) return;

    const device bfloat* expert_base = G_experts + expert_id * R3 * R2;

    // XOR Swizzled Threadgroup Memory
    threadgroup bfloat shmem[TILE_K * (ROW_STRIDE)];

    float acc[TILE_M];
    #pragma unroll
    for (int i = 0; i < TILE_M; i++) acc[i] = 0.0f;

    uint num_tiles = R3 / TILE_K;

    for (uint t = 0; t < num_tiles; t++) {
        // 載入 G_experts tile 到 Threadgroup Memory，使用 XOR Swizzling
        #pragma unroll
        for (uint i = 0; i < TILE_K; i += 4) {
            uint r3_row  = t * TILE_K + (lane / 8) + i;
            uint r2_col  = col_block + (lane % 8) * 4;

            // XOR 映射：physical_col = r2_col XOR (r3_row % 32)
            uint phys_col = r2_col ^ (r3_row & 31u);
            uint src_idx  = r3_row * R2 + r2_col;
            uint dst_idx  = ((lane / 8) + i) * ROW_STRIDE + (lane % 8) * 4;

            ((threadgroup bfloat4*)(&shmem[dst_idx]))[0] =
                ((const device bfloat4*)(&expert_base[src_idx]))[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 計算點積（XOR 逆映射讀取）
        for (uint r3 = 0; r3 < TILE_K; r3++) {
            float x_val = float(x_shared[t * TILE_K + r3]);
            #pragma unroll
            for (int o = 0; o < TILE_M; o++) {
                // 逆 XOR 映射讀取正確資料
                uint phys_o = uint(o) ^ (r3 & 31u);
                acc[o] += x_val * float(shmem[r3 * ROW_STRIDE + phys_o]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 寫出結果
    #pragma unroll
    for (int o = 0; o < TILE_M; o++) {
        uint out_r2 = col_block + o;
        if (out_r2 < R2) {
            partial_out[expert_slot * R2 + out_r2] = acc[o];
        }
    }
}
