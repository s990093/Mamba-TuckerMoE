#include <metal_stdlib>
using namespace metal;

// 假設常量，實際由 MLX dynamic compilation 帶入
// constant uint B = ...;
// constant uint nc = ...;
// constant uint Lc = ...;
// constant uint H = ...;
// constant uint D = ...;

kernel void ssm_chunk_scan(
    device const float* la_c [[buffer(0)]],
    device const float* u_c [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint d [[thread_position_in_grid]],  // d = thread_position_in_grid.x
    uint h [[thread_position_in_grid_y]],// h = thread_position_in_grid.y (Note: MLX sets grid(D, H, B*nc) -> x,y,z)
    uint b_c [[thread_position_in_grid_z]])
{
    // 在 MLX fast kernel 中，如果是用源碼字串，它會自動產生 signature。
    // 但如果寫成獨立檔案，我們用標準 Metal 寫法。
    // 注意：為了與 MLX mx.fast.metal_kernel 相容，實際讀取進 Python 時，
    // 我們可能只取函式主體，或者整份編譯。
    
    // 如果這是一份獨立的 Metal kernel，主體邏輯如下：
    if (d >= D || h >= H || b_c >= B * nc) return;
    
    float h_val = 0.0;
    for (uint t = 0; t < Lc; ++t) {
        float la = la_c[b_c * Lc * H + t * H + h];
        float u  = u_c[b_c * Lc * H * D + t * H * D + h * D + d];
        
        la = la > float(40.0) ? float(40.0) : la;
        la = la < float(-40.0) ? float(-40.0) : la;
        
        h_val = metal::exp(la) * h_val + u;
        out[b_c * Lc * H * D + t * H * D + h * D + d] = h_val;
    }
}
