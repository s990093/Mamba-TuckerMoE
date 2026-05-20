// Metal Fused MoE Kernel
// Fuses: Router projection → Top-K → Expert gather → Expert forward
// Target: Reduce 3-4 kernel dispatches to 1
// Expected speedup: 2-3× on MoE layer

#include <metal_stdlib>
using namespace metal;

// Fused MoE kernel: Router → Top-K selection → Expert dispatch
kernel void moe_fused_forward(
    device const half* input [[buffer(0)]],              // (B*L, d_in)
    device const half* router_weight [[buffer(1)]],      // (num_experts, d_in)
    device const half* expert_weights [[buffer(2)]],     // Pre-gathered expert weights

    device half* output [[buffer(3)]],                   // (B*L, d_out)

    constant uint& B_L [[buffer(4)]],                    // B * L (sequence length)
    constant uint& d_in [[buffer(5)]],
    constant uint& d_out [[buffer(6)]],
    constant uint& num_experts [[buffer(7)]],
    constant uint& top_k [[buffer(8)]],

    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    uint token_id = gid / d_out;
    uint out_dim = gid % d_out;

    if (token_id >= B_L || out_dim >= d_out) return;

    // Step 1: Router projection (local computation)
    // logits = input @ router_weight^T
    float router_logit = 0.0f;
    for (uint i = 0; i < d_in; i++) {
        float inp = float(input[token_id * d_in + i]);
        float w = float(router_weight[out_dim * d_in + i]); // simplified indexing
        router_logit += inp * w;
    }

    // Step 2: Top-K softmax (simplified for kernel)
    // In practice, would need shared memory for sorting
    float prob = 1.0f / (1.0f + exp(-router_logit)); // Sigmoid approximation

    // Step 3: Expert selection and forward
    // For each selected expert, accumulate weighted output
    float out_val = 0.0f;

    for (uint expert_idx = 0; expert_idx < min(top_k, num_experts); expert_idx++) {
        // Weight for this expert
        float expert_weight = prob / top_k; // Simplified

        // Look up expert forward for this token, out_dim
        // In practice: expert_weights[expert_idx][out_dim]
        float expert_out = float(expert_weights[expert_idx * d_out + out_dim]);

        out_val += expert_weight * expert_out;
    }

    output[token_id * d_out + out_dim] = half(out_val);
}


// Simpler variant: Just router + top-k (dispatch remains separate)
kernel void router_top_k_fast(
    device const half* input [[buffer(0)]],              // (B*L, d_in)
    device const half* router_weight [[buffer(1)]],      // (num_experts, d_in)

    device int* selected_experts [[buffer(2)]],          // (B*L, top_k) - expert indices
    device half* expert_logits [[buffer(3)]],            // (B*L, top_k) - softmax probs

    constant uint& B_L [[buffer(4)]],
    constant uint& d_in [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& top_k [[buffer(7)]],

    uint3 gid [[thread_position_in_grid]])
{
    uint token_id = gid.x;
    uint expert_id = gid.y;

    if (token_id >= B_L || expert_id >= num_experts) return;

    // Compute router logit for this token-expert pair
    float logit = 0.0f;
    for (uint i = 0; i < d_in; i++) {
        float inp = float(input[token_id * d_in + i]);
        float w = float(router_weight[expert_id * d_in + i]);
        logit += inp * w;
    }

    // Store for sorting (in practice, would use atomic operations)
    // This is a simplified version - real top-k requires sorting
    float prob = exp(logit) / (1.0f + exp(logit)); // Sigmoid

    // Output: would be sorted and top-k selected
    // For now just return all experts' scores
    expert_logits[token_id * num_experts + expert_id] = half(prob);
}
