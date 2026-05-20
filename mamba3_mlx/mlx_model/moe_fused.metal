// Metal Fused TuckerMoE Kernel
// Fuses: Router projection → Softmax → Top-K selection → Expert dispatch → Weighted output
// Target: Reduce 5-7 kernel dispatches to 1
// Expected: 10-15% speedup on MoE layer (3-5ms reduction per token)

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Kernel 1: Router Projection + Softmax + Top-K Selection (fused)
// ============================================================================
kernel void router_topk_fused(
    device const half* input [[buffer(0)]],              // (B*L, dim_in)
    device const half* router_weight [[buffer(1)]],      // (num_experts, dim_in)

    device int* selected_experts [[buffer(2)]],          // (B*L, top_k)
    device float* selected_logits [[buffer(3)]],         // (B*L, top_k)
    device float* top_k_normalized [[buffer(4)]],        // (B*L, top_k) - normalized probs

    constant uint& B_L [[buffer(5)]],
    constant uint& dim_in [[buffer(6)]],
    constant uint& num_experts [[buffer(7)]],
    constant uint& top_k [[buffer(8)]],
    constant float& temperature [[buffer(9)]],

    uint gid [[thread_position_in_grid]])
{
    uint token_id = gid;
    if (token_id >= B_L) return;

    // Step 1: Router projection - compute logits for all experts
    // Uses shared memory to avoid redundant computation
    threadgroup float logits[256];  // Max 256 experts

    // Parallel reduction across threads in threadgroup
    for (uint e = 0; e < num_experts; e++) {
        float logit = 0.0f;
        for (uint i = 0; i < dim_in; i++) {
            float inp = float(input[token_id * dim_in + i]);
            float w = float(router_weight[e * dim_in + i]);
            logit += inp * w;
        }
        logits[e] = logit;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Apply capping (tanh approximation) and temperature scaling
    for (uint e = 0; e < num_experts; e++) {
        float logit = logits[e];
        // Fast tanh approximation: 2 * sigmoid(2*x) - 1
        // But we use: min(max(x, -10), 10) for capping
        logit = min(max(logit * 10.0f, -10.0f), 10.0f);
        logit = logit / temperature;
        logits[e] = logit;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Softmax - find max for numerical stability
    float max_logit = logits[0];
    for (uint e = 1; e < num_experts; e++) {
        max_logit = max(max_logit, logits[e]);
    }

    // Compute exp(logit - max) and sum
    float sum_exp = 0.0f;
    threadgroup float exp_logits[256];
    for (uint e = 0; e < num_experts; e++) {
        float exp_val = exp(logits[e] - max_logit);
        exp_logits[e] = exp_val;
        sum_exp += exp_val;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize probabilities
    threadgroup float probs[256];
    sum_exp = max(sum_exp, 1e-8f);
    for (uint e = 0; e < num_experts; e++) {
        probs[e] = exp_logits[e] / sum_exp;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Top-K selection - find top k experts
    // Simple O(n*k) selection for small num_experts
    threadgroup int top_indices[256];
    threadgroup float top_values[256];

    for (uint k = 0; k < min(top_k, num_experts); k++) {
        float max_prob = -1e8f;
        int max_idx = -1;

        // Find max among non-selected
        for (uint e = 0; e < num_experts; e++) {
            bool already_selected = false;
            for (uint j = 0; j < k; j++) {
                if (top_indices[j] == e) {
                    already_selected = true;
                    break;
                }
            }

            if (!already_selected && probs[e] > max_prob) {
                max_prob = probs[e];
                max_idx = e;
            }
        }

        top_indices[k] = max_idx;
        top_values[k] = max_prob;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5: Normalize top-k probabilities
    float top_k_sum = 0.0f;
    for (uint k = 0; k < min(top_k, num_experts); k++) {
        top_k_sum += top_values[k];
    }
    top_k_sum = max(top_k_sum, 1e-8f);

    // Write outputs
    for (uint k = 0; k < min(top_k, num_experts); k++) {
        selected_experts[token_id * top_k + k] = top_indices[k];
        selected_logits[token_id * top_k + k] = logits[top_indices[k]];
        top_k_normalized[token_id * top_k + k] = top_values[k] / top_k_sum;
    }
}


// ============================================================================
// Kernel 2: Shared Projection + Normalization
// ============================================================================
kernel void shared_projection_norm(
    device const half* input [[buffer(0)]],              // (B*L, dim_in)
    device const half* U_in [[buffer(1)]],               // (dim_in, r3)
    device const half* norm_scale [[buffer(2)]],         // (r3,) - RMSNorm scale

    device float* x_shared [[buffer(3)]],                // (B*L, r3) - output

    constant uint& B_L [[buffer(4)]],
    constant uint& dim_in [[buffer(5)]],
    constant uint& r3 [[buffer(6)]],

    uint2 gid [[thread_position_in_grid]])
{
    uint token_id = gid.x;
    uint dim_idx = gid.y;

    if (token_id >= B_L || dim_idx >= r3) return;

    // Matrix multiplication: input[token] @ U_in[:, dim_idx]
    float result = 0.0f;
    for (uint i = 0; i < dim_in; i++) {
        float inp = float(input[token_id * dim_in + i]);
        float w = float(U_in[i * r3 + dim_idx]);
        result += inp * w;
    }

    x_shared[token_id * r3 + dim_idx] = result;
}


kernel void rms_normalization(
    device float* x_shared [[buffer(0)]],                // (B*L, r3) - in-place
    device const half* norm_weight [[buffer(1)]],        // (r3,)

    constant uint& B_L [[buffer(2)]],
    constant uint& r3 [[buffer(3)]],
    constant float& eps [[buffer(4)]],

    uint2 gid [[thread_position_in_grid]])
{
    uint token_id = gid.x;
    uint dim_idx = gid.y;

    if (token_id >= B_L || dim_idx >= r3) return;

    // Compute RMS for this token (parallel reduction would be better)
    float sum_sq = 0.0f;
    for (uint i = 0; i < r3; i++) {
        float val = x_shared[token_id * r3 + i];
        sum_sq += val * val;
    }

    float rms = sqrt(sum_sq / r3 + eps);
    float normalized = x_shared[token_id * r3 + dim_idx] / rms;
    float scaled = normalized * float(norm_weight[dim_idx]);

    x_shared[token_id * r3 + dim_idx] = scaled;
}


// ============================================================================
// Kernel 3: Expert Selection + Weighted Output Computation
// ============================================================================
kernel void expert_weighted_forward(
    device const float* x_shared [[buffer(0)]],          // (B*L, r3)
    device const int* selected_experts [[buffer(1)]],    // (B*L, top_k)
    device const float* top_k_probs [[buffer(2)]],       // (B*L, top_k)

    device const half* U_expert [[buffer(3)]],           // (num_experts, r1)
    device const half* core [[buffer(4)]],               // (r1, r3, r2)
    device const half* U_out [[buffer(5)]],              // (r2, dim_out)
    device const half* bias [[buffer(6)]],               // (dim_out,)

    device float* output [[buffer(7)]],                  // (B*L, dim_out)

    constant uint& B_L [[buffer(8)]],
    constant uint& r1 [[buffer(9)]],
    constant uint& r2 [[buffer(10)]],
    constant uint& r3 [[buffer(11)]],
    constant uint& dim_out [[buffer(12)]],
    constant uint& top_k [[buffer(13)]],
    constant uint& num_experts [[buffer(14)]],

    uint2 gid [[thread_position_in_grid]])
{
    uint token_id = gid.x;
    uint out_dim = gid.y;

    if (token_id >= B_L || out_dim >= dim_out) return;

    // Compute weighted expert output
    float x_core_val = 0.0f;

    // For each selected expert, compute contribution
    for (uint k = 0; k < min(top_k, num_experts); k++) {
        int expert_idx = selected_experts[token_id * top_k + k];
        float prob = top_k_probs[token_id * top_k + k];

        if (expert_idx < 0 || expert_idx >= num_experts) continue;

        // Compute: x_shared @ G_expert[expert_idx] @ U_out[out_dim]
        // Where G_expert[e] = U_expert[e] @ core.reshape(r1, r3*r2)

        // Step 1: x_shared @ core for this expert
        // core[r1, r3, r2] is indexed as core[r1][r3][r2]
        float expert_contribution = 0.0f;

        for (uint r2_idx = 0; r2_idx < r2; r2_idx++) {
            // G_expert[e, r3, r2] = sum_r1 U_expert[e, r1] * core[r1, r3, r2]
            // Then x_shared @ G_expert gives sum over r3

            float expert_out = 0.0f;
            for (uint r3_idx = 0; r3_idx < r3; r3_idx++) {
                float x_val = x_shared[token_id * r3 + r3_idx];

                // G[r3, r2] = sum_r1 U_expert[r1] * core[r1, r3, r2]
                float g_val = 0.0f;
                for (uint r1_idx = 0; r1_idx < r1; r1_idx++) {
                    float u_exp = float(U_expert[expert_idx * r1 + r1_idx]);
                    // Core is stored as core[r1 * r3 * r2 + r3 * r2 + r2_idx]
                    int core_idx = r1_idx * (r3 * r2) + r3_idx * r2 + r2_idx;
                    float c = float(core[core_idx]);
                    g_val += u_exp * c;
                }

                expert_out += x_val * g_val;
            }

            // expert_out is now sum over r3 of x_shared[r3] * G[r3, r2]
            // Apply U_out projection
            float u_out_val = float(U_out[r2_idx * dim_out + out_dim]);
            expert_contribution += expert_out * u_out_val;
        }

        x_core_val += prob * expert_contribution;
    }

    // Add bias
    float final_out = x_core_val + float(bias[out_dim]);
    output[token_id * dim_out + out_dim] = final_out;
}


// ============================================================================
// Simple Kernel: Fully Fused Router + Softmax + Top-K (without expert compute)
// ============================================================================
kernel void router_topk_simple(
    device const half* input [[buffer(0)]],              // (B*L, dim_in)
    device const half* router_weight [[buffer(1)]],      // (num_experts, dim_in)
    device const half* tanh_coeff [[buffer(2)]],         // scalar 10.0

    device int* selected_experts [[buffer(3)]],          // (B*L, top_k)
    device float* selected_probs [[buffer(4)]],          // (B*L, top_k)

    constant uint& B_L [[buffer(5)]],
    constant uint& dim_in [[buffer(6)]],
    constant uint& num_experts [[buffer(7)]],
    constant uint& top_k [[buffer(8)]],
    constant float& temperature [[buffer(9)]],

    uint gid [[thread_position_in_grid]])
{
    uint token_id = gid;
    if (token_id >= B_L) return;

    // Compute router logits for all experts
    float logits[256];  // Stack allocation, max 256 experts
    float max_logit = -1e8f;

    for (uint e = 0; e < num_experts; e++) {
        float logit = 0.0f;
        for (uint i = 0; i < dim_in; i++) {
            float inp = float(input[token_id * dim_in + i]);
            float w = float(router_weight[e * dim_in + i]);
            logit += inp * w;
        }

        // Cap and scale
        logit = min(max(logit * 10.0f, -10.0f), 10.0f) / temperature;
        logits[e] = logit;
        max_logit = max(max_logit, logit);
    }

    // Softmax with numerical stability
    float sum_exp = 0.0f;
    float exp_logits[256];
    for (uint e = 0; e < num_experts; e++) {
        float exp_val = exp(logits[e] - max_logit);
        exp_logits[e] = exp_val;
        sum_exp += exp_val;
    }
    sum_exp = max(sum_exp, 1e-8f);

    // Top-K selection
    for (uint k = 0; k < min(top_k, num_experts); k++) {
        float max_prob = -1.0f;
        int max_idx = -1;

        for (uint e = 0; e < num_experts; e++) {
            bool selected = false;
            for (uint j = 0; j < k; j++) {
                if (selected_experts[token_id * top_k + j] == e) {
                    selected = true;
                    break;
                }
            }

            float prob = exp_logits[e] / sum_exp;
            if (!selected && prob > max_prob) {
                max_prob = prob;
                max_idx = e;
            }
        }

        selected_experts[token_id * top_k + k] = max_idx;
        selected_probs[token_id * top_k + k] = max_prob;
    }
}
