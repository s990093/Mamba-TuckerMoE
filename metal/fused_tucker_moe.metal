#include <metal_stdlib>
using namespace metal;

// 假設常量，實際由 MLX dynamic compilation 帶入
// constant uint DIM_IN = ...;
// constant uint R3 = ...;
// constant uint R2 = ...;
// constant uint DIM_OUT = ...;
// constant uint NUM_TOKENS = ...;
// constant uint THREADS_PER_GROUP = ...;

kernel void fused_tucker_moe_kernel(
    device const float* x [[buffer(0)]],
    device const int* router_indices [[buffer(1)]],
    device const float* U_in_global [[buffer(2)]],
    device const float* G_cores [[buffer(3)]],
    device const float* U_out_global [[buffer(4)]],
    device float* out [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint token_idx [[thread_position_in_grid]])
{
    // ==========================================
    // Step 1: 載入共用字典 (SRAM)
    // 將共享的 U_in 和 U_out 載入到 GPU 內最快的快取 Threadgroup Memory
    // 因為這兩個矩陣很小，大家可以一起看。
    // ==========================================
    threadgroup float U_in_sram[DIM_IN * R3];
    threadgroup float U_out_sram[R2 * DIM_OUT];
    
    // 協同合作搬移資料
    for (uint i = tid; i < DIM_IN * R3; i += THREADS_PER_GROUP) {
        U_in_sram[i] = U_in_global[i];
    }
    for (uint i = tid; i < R2 * DIM_OUT; i += THREADS_PER_GROUP) {
        U_out_sram[i] = U_out_global[i];
    }
    // 等待所有 Thread 都搬完
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ==========================================
    // Step 2: 各自領取任務
    // 每個 Thread 負責一個 Token
    // ==========================================
    if (token_idx >= NUM_TOKENS) return;
    int expert_id = router_indices[token_idx];

    // ==========================================
    // Step 3: 晶片內連乘 (核心魔法)
    // 這裡絕對不能把它 return 或寫回主記憶體
    // ==========================================
    
    // 3a. 讀取自己的 Token，乘上 SRAM 裡的 U_in，得到降維後的 x_s
    float x_s[R3] = {0};
    for (uint i = 0; i < DIM_IN; ++i) {
        float token_val = x[token_idx * DIM_IN + i];
        for (uint r = 0; r < R3; ++r) {
            x_s[r] += token_val * U_in_sram[i * R3 + r];
        }
    }

    // 3b. 去主記憶體把這個 Token 專屬的 G_e 拉進來 (唯一一次去主記憶體拿動態資料)
    // 立刻把 x_s * G_e 算完
    float y_s[R2] = {0};
    uint expert_offset = expert_id * R3 * R2;
    for (uint r = 0; r < R3; ++r) {
        for (uint s = 0; s < R2; ++s) {
            float g_val = G_cores[expert_offset + r * R2 + s];
            y_s[s] += x_s[r] * g_val;
        }
    }

    // 3c. 算完的結果立刻再乘上 SRAM 裡的 U_out，得到最終的 y
    float y[DIM_OUT] = {0};
    for (uint s = 0; s < R2; ++s) {
        for (uint j = 0; j < DIM_OUT; ++j) {
            y[j] += y_s[s] * U_out_sram[s * DIM_OUT + j];
        }
    }

    // ==========================================
    // Step 4: 功德圓滿才放行
    // 把最終的 y 寫回主記憶體 (Unified Memory)
    // ==========================================
    for (uint j = 0; j < DIM_OUT; ++j) {
        out[token_idx * DIM_OUT + j] = y[j];
    }
}
