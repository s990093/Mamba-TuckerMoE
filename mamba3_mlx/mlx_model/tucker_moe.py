# -*- coding: utf-8 -*-
from typing import Optional
import mlx.core as mx
import mlx.nn as nn

from .ops import RMSNorm, fast_scaled_tanh, get_router_temperature


def _is_dense_linear(layer: nn.Module) -> bool:
    """Check if a layer is a standard (non-quantized) Linear layer."""
    return isinstance(layer, nn.Linear)


def _topk_indices(x: mx.array, k: int) -> mx.array:
    """Get top-k indices along the last axis."""
    return mx.argsort(-x, axis=-1)[..., :k]


def rms_norm_fast(x: mx.array, norm_layer: RMSNorm) -> mx.array:
    """RMS normalization with weight scaling."""
    x_float = x.astype(mx.float32)
    ss = mx.sum(x_float * x_float, axis=-1, keepdims=True)
    rms = mx.sqrt(ss / x.shape[-1] + 1e-5)
    normalized = x_float / rms
    return (normalized * norm_layer.weight).astype(x.dtype)


class TuckerMoE(nn.Module):
    """
    Tucker-decomposed Mixture-of-Experts with multiple optimization paths.

    Supports:
    - full_fuse: Complete kernel fusion (dense-only, Metal)
    - amx_partial_fuse: AMX simdgroup matrix operations (single-token bf16)
    - scalar_fuse: Scalar expert path (fastest for b=1)
    - einsum_fuse: Fused einsum with Metal kernel
    - Default: Standard MLX operations

    G_e = einsum('er, rst -> est', U_expert[e], core)  →  shape (num_experts, r3, r2).
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_experts: int = 8,
        top_k: int = 2,
        r1: int = 4,
        r2: int = 1024,
        r3: int = 256,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.r1, self.r2, self.r3 = r1, r2, r3

        self.router = nn.Linear(dim_in, num_experts, bias=False)
        self.U_expert = mx.zeros((num_experts, r1))
        self.U_in = nn.Linear(dim_in, r3, bias=False)
        self.U_out = nn.Linear(r2, dim_out, bias=False)
        self.core = mx.zeros((r1, r3, r2))
        self.bias = mx.zeros((dim_out,))
        self.inner_norm = nn.RMSNorm(r3)

        self._G_cache: Optional[mx.array] = None
        self._amx_kernel_cache: dict = {}

    def invalidate_g_cache(self) -> None:
        """Invalidate cached G matrices when weights change."""
        self._G_cache = None

    def _get_G(self) -> mx.array:
        """Get or compute precomputed expert matrices G = U_expert @ core.reshape()."""
        if self._G_cache is None:
            r1, r3, r2 = self.core.shape
            core_2d = self.core.reshape(r1, r3 * r2)
            self._G_cache = mx.matmul(self.U_expert, core_2d).reshape(self.num_experts, r3, r2)
            mx.eval(self._G_cache)
        return self._G_cache

    def __call__(
        self,
        x: mx.array,
        router_temp: mx.array,
        *,
        router_x: Optional[mx.array] = None,
        einsum_fuse: Optional[bool] = None,
        full_fuse: Optional[bool] = None,
        amx_partial_fuse: Optional[bool] = None,
        scalar_fuse: Optional[bool] = None,
    ) -> mx.array:
        """Forward pass with optional fusion optimization paths.

        Args:
            x: Input tensor (B*L, dim_in)
            router_temp: Router temperature for scaling logits
            router_x: Optional separate routing input
            einsum_fuse: Enable fused einsum Metal kernel
            full_fuse: Enable complete kernel fusion
            amx_partial_fuse: Enable AMX simdgroup operations
            scalar_fuse: Enable scalar expert path
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        rx = x_flat if router_x is None else router_x.reshape(-1, router_x.shape[-1])

        # Router: compute expert selection
        raw_logits = self.router(rx)
        rt_dt = router_temp.dtype
        t = mx.maximum(router_temp, mx.array(1e-4, dtype=rt_dt))
        capped = fast_scaled_tanh(raw_logits, 10.0)
        router_logits = capped / t
        router_probs = mx.softmax(router_logits, axis=-1)
        top_k_indices = _topk_indices(router_logits, self.top_k)
        top_k_raw = mx.take_along_axis(router_probs, top_k_indices, axis=-1)
        eps = mx.array(1e-6, dtype=top_k_raw.dtype)
        top_k_probs = top_k_raw / (mx.sum(top_k_raw, axis=-1, keepdims=True) + eps)

        g_all = self._get_G()

        # Shared projection + normalization
        x_shared = rms_norm_fast(self.U_in(x_flat), self.inner_norm)

        # Select top-k expert matrices
        g_sel = g_all[top_k_indices]

        # Compute expert contributions
        expert_outs = mx.einsum("br,bkrs->bks", x_shared, g_sel)
        out_core = mx.sum(expert_outs * mx.expand_dims(top_k_probs, axis=-1), axis=1)

        # Output projection
        out = self.U_out(out_core).reshape(*orig_shape[:-1], -1)
        return out + self.bias
