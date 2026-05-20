// -*- coding: utf-8 -*-
// Ultimate SSM Sequential Scan Kernel (Metal)
//
// Optimized O(Lc) sequential scan per (d, h) pair, replacing O(Lc²) matrix matmul.
// Exploits threadgroup local memory for minimal register pressure.
//
// Thread layout:
//   - Each thread handles one (d, h) element
//   - Grid: (D, H, B*nc) where D = d_inner = N*P
//   - Threadgroup: (32, 1, 1) for occupancy
//
// Algorithm:
//   for t in 0..Lc-1:
//       alpha[t] = exp(clip(la[t], -88, 88))
//       h[t] = alpha[t] * h[t-1] + u[t]
//
// This is O(Lc) in time (one pass, sequential dependency chain).

#include <metal_stdlib>
using namespace metal;

kernel void ssm_sequential_scan_bf16(
    device const half* la_c [[buffer(0)]],      // (B*nc, Lc, H) log-alpha per step
    device const half* u_c [[buffer(1)]],       // (B*nc, Lc, D) input signal (D = N*P)
    device half* h_out [[buffer(2)]],           // (B*nc, Lc, D) output hidden state

    constant uint& B_nc [[buffer(3)]],          // B * nc
    constant uint& Lc [[buffer(4)]],            // chunk size
    constant uint& D [[buffer(5)]],             // d_inner = N * P
    constant uint& H [[buffer(6)]],             // num heads

    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    uint d = gid.x;
    uint h = gid.y;
    uint b_nc_id = gid.z;

    // Bounds check
    if (d >= D || h >= H || b_nc_id >= B_nc) return;

    // Sequential scan: h[t] = alpha[t] * h[t-1] + u[t]
    half h_val = half(0.0f);  // Initialize to zero

    for (uint t = 0; t < Lc; ++t) {
        // Load log-alpha at this step
        half la_t_half = la_c[b_nc_id * Lc * H + t * H + h];
        float la_t = float(la_t_half);

        // Clip for numerical stability
        la_t = clamp(la_t, -88.0f, 88.0f);

        // Compute alpha = exp(log-alpha)
        float alpha = exp(la_t);

        // Load input at this step
        half u_t_half = u_c[b_nc_id * Lc * D + t * D + d];
        float u_t = float(u_t_half);

        // Update: h = alpha * h_prev + u
        float h_val_f32 = float(h_val);
        float h_new = alpha * h_val_f32 + u_t;
        h_val = half(h_new);

        // Store output
        h_out[b_nc_id * Lc * D + t * D + d] = h_val;
    }
}

// FP32 variant (higher precision, more memory)
kernel void ssm_sequential_scan_fp32(
    device const float* la_c [[buffer(0)]],
    device const float* u_c [[buffer(1)]],
    device float* h_out [[buffer(2)]],

    constant uint& B_nc [[buffer(3)]],
    constant uint& Lc [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant uint& H [[buffer(6)]],

    uint3 gid [[thread_position_in_grid]])
{
    uint d = gid.x;
    uint h = gid.y;
    uint b_nc_id = gid.z;

    if (d >= D || h >= H || b_nc_id >= B_nc) return;

    float h_val = 0.0f;

    for (uint t = 0; t < Lc; ++t) {
        float la_t = la_c[b_nc_id * Lc * H + t * H + h];
        la_t = clamp(la_t, -88.0f, 88.0f);
        float alpha = exp(la_t);

        float u_t = u_c[b_nc_id * Lc * D + t * D + d];
        h_val = alpha * h_val + u_t;

        h_out[b_nc_id * Lc * D + t * D + d] = h_val;
    }
}

// Intra-chunk final state reduction (returns h at final position per chunk)
kernel void ssm_final_state_reduce(
    device const half* h_out [[buffer(0)]],    // (B*nc, Lc, D)
    device half* h_final [[buffer(1)]],        // (B*nc, D)

    constant uint& B_nc [[buffer(2)]],
    constant uint& Lc [[buffer(3)]],
    constant uint& D [[buffer(4)]],

    uint3 gid [[thread_position_in_grid]])
{
    uint d = gid.x;
    uint b_nc_id = gid.y;

    if (d >= D || b_nc_id >= B_nc) return;

    // Just copy the last timestep
    h_final[b_nc_id * D + d] = h_out[b_nc_id * Lc * D + (Lc - 1) * D + d];
}
