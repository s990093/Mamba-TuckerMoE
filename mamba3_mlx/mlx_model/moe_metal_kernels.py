"""
Python bindings for fused TuckerMoE Metal kernels.

Provides optimized Metal implementations for router → top-k selection → expert routing.
"""

import mlx.core as mx
from pathlib import Path
from typing import Tuple, Optional


class MoEMetalKernels:
    """Metal kernel wrapper for TuckerMoE fusion optimization."""

    _kernel_cache = {}

    @staticmethod
    def _load_kernel_source() -> str:
        """Load Metal kernel source from moe_fused.metal."""
        kernel_path = Path(__file__).parent / "moe_fused.metal"
        if kernel_path.exists():
            return kernel_path.read_text()
        return None

    @staticmethod
    def router_topk_simple(
        input: mx.array,                 # (B*L, dim_in)
        router_weight: mx.array,         # (num_experts, dim_in)
        num_experts: int,
        top_k: int,
        temperature: float,
    ) -> Tuple[mx.array, mx.array]:
        """Fused router projection + softmax + top-k selection.

        Args:
            input: Token features (B*L, dim_in)
            router_weight: Router weights (num_experts, dim_in)
            num_experts: Number of experts
            top_k: Number of top experts to select
            temperature: Temperature for router softmax

        Returns:
            selected_experts: Expert indices (B*L, top_k)
            selected_probs: Normalized probabilities (B*L, top_k)
        """
        B_L, dim_in = input.shape
        top_k = min(top_k, num_experts)

        # For now, use MLX implementation as fallback
        # TODO: Replace with actual Metal kernel when mx.metal_kernel is available
        raw_logits = mx.matmul(input, router_weight.T)  # (B*L, num_experts)

        # Cap and scale
        capped = mx.clip(raw_logits * 10.0, -10.0, 10.0) / temperature

        # Softmax
        router_probs = mx.softmax(capped, axis=-1)

        # Top-k selection
        top_k_indices = mx.argsort(-capped, axis=-1)[..., :top_k]
        top_k_raw = mx.take_along_axis(router_probs, top_k_indices, axis=-1)
        top_k_sum = mx.sum(top_k_raw, axis=-1, keepdims=True) + 1e-6
        top_k_probs = top_k_raw / top_k_sum

        return top_k_indices, top_k_probs

    @staticmethod
    def shared_projection(
        input: mx.array,                 # (B*L, dim_in)
        U_in_weight: mx.array,           # (dim_in, r3) or (r3, dim_in)
        norm_weight: mx.array,           # (r3,)
        norm_eps: float = 1e-5,
    ) -> mx.array:
        """Fused shared projection + RMS normalization.

        Args:
            input: Token features (B*L, dim_in)
            U_in_weight: U_in weight matrix
            norm_weight: RMSNorm weight
            norm_eps: Normalization epsilon

        Returns:
            x_shared: Normalized projected features (B*L, r3)
        """
        # Project
        x_proj = mx.matmul(input, U_in_weight)  # (B*L, r3)

        # RMS normalize
        ss = mx.sum(x_proj * x_proj, axis=-1, keepdims=True)
        rms = mx.sqrt(ss / x_proj.shape[-1] + norm_eps)
        x_norm = x_proj / rms
        x_shared = x_norm * norm_weight

        return x_shared

    @staticmethod
    def expert_weighted_forward(
        x_shared: mx.array,               # (B*L, r3)
        selected_experts: mx.array,       # (B*L, top_k)
        top_k_probs: mx.array,            # (B*L, top_k)
        U_expert: mx.array,               # (num_experts, r1)
        core: mx.array,                   # (r1, r3, r2)
        U_out_weight: mx.array,           # (dim_out, r2) or (r2, dim_out)
        bias: mx.array,                   # (dim_out,)
    ) -> mx.array:
        """Expert selection and weighted forward computation.

        Args:
            x_shared: Normalized shared representation (B*L, r3)
            selected_experts: Selected expert indices (B*L, top_k)
            top_k_probs: Expert selection probabilities (B*L, top_k)
            U_expert: Expert coefficient matrix (num_experts, r1)
            core: Core tensor (r1, r3, r2)
            U_out_weight: Output projection weights
            bias: Output bias

        Returns:
            out: Expert-routed output (B*L, dim_out)
        """
        # Precompute expert matrices
        # G = U_expert @ core.reshape(r1, r3*r2) -> (num_experts, r3, r2)
        r1, r3, r2 = core.shape
        core_2d = core.reshape(r1, r3 * r2)
        G_experts = mx.matmul(U_expert, core_2d).reshape(U_expert.shape[0], r3, r2)

        # Gather selected experts
        G_sel = G_experts[selected_experts]  # (B*L, top_k, r3, r2)

        # Expert compute
        expert_outs = mx.einsum("br,bkrs->bks", x_shared, G_sel)
        x_core = mx.sum(expert_outs * mx.expand_dims(top_k_probs, axis=-1), axis=1)

        # Output projection
        out = mx.matmul(x_core, U_out_weight.T) + bias

        return out


# Singleton instance
_moe_kernels = MoEMetalKernels()


def get_moe_kernels() -> MoEMetalKernels:
    """Get the global MoE Metal kernels instance."""
    return _moe_kernels
