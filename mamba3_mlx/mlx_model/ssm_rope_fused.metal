// Metal Fused SSM + RoPE Kernel
// Fuses: RoPE application → Input signal computation → SSM state update
// Target: Reduce sequential operations to single Metal dispatch
// Expected: 2-4ms savings per token (10-15% improvement)

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Kernel 1: Fused RoPE + Input Signal Computation
// ============================================================================
kernel void rope_input_signal_fused(
    device const half* B_param [[buffer(0)]],           // (B, L, G, N*R) or (B, G, N*R) for decode
    device const half* x_ssm [[buffer(1)]],             // (B, L, H, P, R) or (B, H, P, R)
    device const half* angles [[buffer(2)]],            // (B, L, H, N//2) or (B, H, N//2)
    device const half* bias_B [[buffer(3)]],            // (G, N, R)

    device float* B_rotated [[buffer(4)]],              // Output: (B, L, H, N, R)
    device float* input_signal [[buffer(5)]],           // Output: (B, L, H, N, P)

    constant uint& B [[buffer(6)]],
    constant uint& L [[buffer(7)]],
    constant uint& H [[buffer(8)]],
    constant uint& G [[buffer(9)]],
    constant uint& N [[buffer(10)]],
    constant uint& P [[buffer(11)]],
    constant uint& R [[buffer(12)]],
    constant uint& ratio [[buffer(13)]],  // H / G

    uint3 gid [[thread_position_in_grid]])
{
    // gid.x = batch index (b)
    // gid.y = sequence index (l)
    // gid.z = head index (h)

    uint b = gid.x;
    uint l = gid.y;
    uint h = gid.z;

    if (b >= B || l >= L || h >= H) return;

    uint g = h / ratio;  // Group index
    uint N_half = N / 2;

    // Step 1: Load B parameter, apply bias, apply RoPE
    // B_param shape: (B, L, G, N*R)
    // angles shape: (B, L, H, N//2)

    for (uint n = 0; n < N; n++) {
        for (uint r = 0; r < R; r++) {
            uint param_idx = b * (L * G * N * R) + l * (G * N * R) + g * (N * R) + n * R + r;
            float b_val = float(B_param[param_idx]) + float(bias_B[g * (N * R) + n * R + r]);

            // Apply RoPE if n < N_half (has rotation angle)
            if (n < N_half) {
                float angle = float(angles[b * (L * H * N_half) + l * (H * N_half) + h * N_half + n]);

                // RoPE rotation: [x, y] → [x*cos(θ) - y*sin(θ), x*sin(θ) + y*cos(θ)]
                // For even/odd pair: (n, n+N_half)
                float cos_a = cos(angle);
                float sin_a = sin(angle);

                float b_pair = 0.0f;
                if (n + N_half < N) {
                    // Get paired element
                    uint pair_idx = b * (L * G * N * R) + l * (G * N * R) + g * (N * R) + (n + N_half) * R + r;
                    b_pair = float(B_param[pair_idx]) + float(bias_B[g * (N * R) + (n + N_half) * R + r]);
                }

                // Apply 2D rotation
                float b_rot = b_val * cos_a - b_pair * sin_a;
                B_rotated[b * (L * H * N * R) + l * (H * N * R) + h * (N * R) + n * R + r] = b_rot;

                if (n + N_half < N) {
                    float b_pair_rot = b_val * sin_a + b_pair * cos_a;
                    B_rotated[b * (L * H * N * R) + l * (H * N * R) + h * (N * R) + (n + N_half) * R + r] = b_pair_rot;
                }
            } else {
                // Second half, already rotated above
                if (n < N_half) continue;  // Skip, already handled
                B_rotated[b * (L * H * N * R) + l * (H * N * R) + h * (N * R) + n * R + r] = b_val;
            }
        }
    }

    // Step 2: Compute input_signal = einsum("blhnr, blhpr -> blhnp", B_rotated, x_ssm)
    for (uint n = 0; n < N; n++) {
        for (uint p = 0; p < P; p++) {
            float sig = 0.0f;
            for (uint r = 0; r < R; r++) {
                uint b_idx = b * (L * H * N * R) + l * (H * N * R) + h * (N * R) + n * R + r;
                uint x_idx = b * (L * H * P * R) + l * (H * P * R) + h * (P * R) + p * R + r;

                float b_rot = B_rotated[b_idx];
                float x = float(x_ssm[x_idx]);
                sig += b_rot * x;
            }
            uint out_idx = b * (L * H * N * P) + l * (H * N * P) + h * (N * P) + n * P + p;
            input_signal[out_idx] = sig;
        }
    }
}


// ============================================================================
// Kernel 2: SSM State Update (Trapezoidal Discretization)
// ============================================================================
kernel void ssm_discretization_fused(
    device const float* input_signal [[buffer(0)]],     // (B, L, H, N, P)
    device const half* dt_b [[buffer(1)]],              // (B, L, H)
    device const half* A_b [[buffer(2)]],               // (H, N, N)
    device const half* lam_b [[buffer(3)]],             // (B, L, H)
    device const float* last_ip [[buffer(4)]],          // (B, H, N, P) for decode state

    device float* u_ssm [[buffer(5)]],                  // Output: (B, L, H, N, P)
    device float* h_new [[buffer(6)]],                  // Output: (B, L, H, N, P) or (B, H, N, P)

    constant uint& B [[buffer(7)]],
    constant uint& L [[buffer(8)]],
    constant uint& H [[buffer(9)]],
    constant uint& N [[buffer(10)]],
    constant uint& P [[buffer(11)]],
    constant bool& is_decode [[buffer(12)]],

    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x;
    uint l = gid.y;
    uint h = gid.z;

    if (b >= B || l >= L || h >= H) return;

    // Load scalar values for this position
    float dt = float(dt_b[b * (L * H) + l * H + h]);
    float lam = float(lam_b[b * (L * H) + l * H + h]);

    // lv = sigmoid(lam), dv = dt, av = exp(dt * A)
    float lv = 1.0f / (1.0f + exp(-lam));  // sigmoid
    float dv = dt;
    float av = exp(dt * float(A_b[h * (N * N) + 0]));  // Simplified: use diagonal

    // Compute u_ssm using trapezoidal discretization
    // u_ssm[t] = lv*dv*is[t] + (1-lv)*dv*av*is[t-1]
    for (uint n = 0; n < N; n++) {
        for (uint p = 0; p < P; p++) {
            float is_t = 0.0f;
            float is_prev = 0.0f;

            if (is_decode && l == 0) {
                // First token in decode: use previous state from mamba_state
                is_t = input_signal[b * (L * H * N * P) + l * (H * N * P) + h * (N * P) + n * P + p];
                is_prev = last_ip[b * (H * N * P) + h * (N * P) + n * P + p];
            } else if (l == 0) {
                // First timestep in prefill: is_prev = 0
                is_t = input_signal[b * (L * H * N * P) + l * (H * N * P) + h * (N * P) + n * P + p];
                is_prev = 0.0f;
            } else {
                // Normal case: use previous timestep
                is_t = input_signal[b * (L * H * N * P) + l * (H * N * P) + h * (N * P) + n * P + p];
                is_prev = input_signal[b * (L * H * N * P) + (l - 1) * (H * N * P) + h * (N * P) + n * P + p];
            }

            float u = lv * dv * is_t + (1.0f - lv) * dv * av * is_prev;
            uint u_idx = b * (L * H * N * P) + l * (H * N * P) + h * (N * P) + n * P + p;
            u_ssm[u_idx] = u;
        }
    }
}


// ============================================================================
// Kernel 3: SSM Scan + Output Projection (Fused)
// ============================================================================
kernel void ssm_scan_output_fused(
    device const float* u_ssm [[buffer(0)]],            // (B, L, H, N, P)
    device const half* la_b [[buffer(1)]],              // (B, L, H) = dt_b * A_b
    device const half* C_param [[buffer(2)]],           // (B, L, G, N*R)
    device const half* angles [[buffer(3)]],            // (B, L, H, N//2) for RoPE
    device const float* h_prev [[buffer(4)]],           // (B, H, N, P) - from mamba_state

    device float* y_output [[buffer(5)]],               // Output: (B, L, H, P, R)
    device float* h_final [[buffer(6)]],                // Output: (B, H, N, P) - new state

    constant uint& B [[buffer(7)]],
    constant uint& L [[buffer(8)]],
    constant uint& H [[buffer(9)]],
    constant uint& N [[buffer(10)]],
    constant uint& P [[buffer(11)]],
    constant uint& R [[buffer(12)]],
    constant uint& G [[buffer(13)]],
    constant bool& is_decode [[buffer(14)]],

    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x;
    uint l = gid.y;
    uint h = gid.z;

    if (b >= B || l >= L || h >= H) return;

    uint g = h / (H / G);

    // Step 1: SSM recurrence: h_new = A*h_prev + B*u_ssm
    // Simplified: single-step scan (loop over n,p)
    float h_new[256];  // Max N*P = 256

    for (uint n = 0; n < N; n++) {
        for (uint p = 0; p < P; p++) {
            float h_val = 0.0f;

            if (h_prev != nullptr && l == 0) {
                // First decode step: use saved state
                h_val = float(h_prev[b * (H * N * P) + h * (N * P) + n * P + p]);
            }

            // Add u_ssm contribution
            float la = float(la_b[b * (L * H) + l * H + h]);
            float u_contrib = exp(la) * float(u_ssm[b * (L * H * N * P) + l * (H * N * P) + h * (N * P) + n * P + p]);
            h_val = h_val * exp(la) + u_contrib;

            h_new[n * P + p] = h_val;
        }
    }

    // Step 2: Apply RoPE to C and compute output
    // y[p,r] = sum_n h_new[n,p] * C_rotated[n,r]
    uint N_half = N / 2;

    for (uint p = 0; p < P; p++) {
        for (uint r = 0; r < R; r++) {
            float y = 0.0f;

            for (uint n = 0; n < N; n++) {
                // RoPE applied to C
                float c_val = float(C_param[b * (L * G * N * R) + l * (G * N * R) + g * (N * R) + n * R + r]);

                if (n < N_half) {
                    float angle = float(angles[b * (L * H * N_half) + l * (H * N_half) + h * N_half + n]);
                    float cos_a = cos(angle);
                    float sin_a = sin(angle);

                    // Paired rotation
                    float c_pair = 0.0f;
                    if (n + N_half < N) {
                        c_pair = float(C_param[b * (L * G * N * R) + l * (G * N * R) + g * (N * R) + (n + N_half) * R + r]);
                    }
                    c_val = c_val * cos_a - c_pair * sin_a;
                }

                y += h_new[n * P + p] * c_val;
            }

            uint y_idx = b * (L * H * P * R) + l * (H * P * R) + h * (P * R) + p * R + r;
            y_output[y_idx] = y;
        }
    }

    // Step 3: Save final state for next decode step
    if (l == L - 1) {  // Last timestep
        for (uint n = 0; n < N; n++) {
            for (uint p = 0; p < P; p++) {
                uint h_idx = b * (H * N * P) + h * (N * P) + n * P + p;
                h_final[h_idx] = h_new[n * P + p];
            }
        }
    }
}
