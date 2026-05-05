// Reference Metal for fused stable softmax over 1-D logits (vocab vector).
// The active path is compiled from Python via ``mx.fast.metal_kernel`` in
// ``inference/fused_sampling_metal.py`` with embedded ``constant uint V / TG``.
//
// Stages (single threadgroup, ``TG`` threads, each striding over ``V``):
//   1) Parallel max of logits (promoted to float).
//   2) Parallel sum of exp(logit - max).
//   3) Write probs[i] = exp(logit[i] - max) / Z.
//
// Follow-up sampling (uniform + cumsum) stays in MLX for simplicity and to reuse its scan.

#include <metal_stdlib>
using namespace metal;

// Example signature when wiring manually (constants must match build):
// kernel void fused_stable_softmax_probs(
//     device const float* logits [[buffer(0)]],
//     device float* probs [[buffer(1)]],
//     uint lid [[thread_index_in_threadgroup]]);
