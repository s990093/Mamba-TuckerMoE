# -*- coding: utf-8 -*-
import mlx.core as mx
import mlx.nn as nn

from .ops import RMSNorm, fast_scaled_tanh, get_router_temperature


class TuckerMoE(nn.Module):
    """
    Tucker-decomposed Mixture-of-Experts (loop-free, vectorised).

    G_e = einsum('r, rst -> st', U_expert[e], core)  →  shape (r3, r2).
    Forward:
      x_shared = x @ U_in  →  inner_norm                (B, r3)
      Gather top-k expert matrices via fancy indexing    (B, top_k, r3, r2)
      Weighted sum of expert outputs                     (B, r2)
      out = x_core @ U_out + bias                        (B, dim_out)
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_experts: int = 8,
        top_k: int = 2,
        r1: int = 32,
        r2: int = 512,
        r3: int = 256,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.r1, self.r2, self.r3 = r1, r2, r3

        self.router = nn.Linear(dim_in, num_experts, bias=False)

        # Tucker decomposition parameters (initialised as zeros; loaded from checkpoint)
        self.U_expert = mx.zeros((num_experts, r1))
        self.U_in     = mx.zeros((dim_in, r3))
        self.U_out    = mx.zeros((r2, dim_out))
        self.core     = mx.zeros((r1, r3, r2))
        self.bias     = mx.zeros((dim_out,))

        self.inner_norm = RMSNorm(r3)

    def __call__(self, x, step=None):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])   # (B_flat, dim_in)

        temperature = get_router_temperature(step)
        raw_logits  = self.router(x_flat)                       # (B_flat, E)
        capped      = fast_scaled_tanh(raw_logits, 10.0)
        router_logits = capped / temperature
        router_probs  = mx.softmax(router_logits, axis=-1)      # (B_flat, E)

        # Top-k selection (descending)
        top_k_indices = mx.argsort(-router_logits, axis=-1)[..., :self.top_k]  # (B_flat, top_k)

        # Gather and normalise top-k probabilities
        top_k_raw  = mx.take_along_axis(router_probs, top_k_indices, axis=-1)
        top_k_sum  = top_k_raw.sum(axis=-1, keepdims=True) + 1e-6
        top_k_probs = top_k_raw / top_k_sum                     # (B_flat, top_k)

        # Shared projection
        x_shared = x_flat @ self.U_in                           # (B_flat, r3)
        x_shared = self.inner_norm(x_shared)

        # Precompute all expert matrices: G_e = U_expert[e] @ core.reshape(r1, r3*r2)
        G_experts = mx.einsum("er, rst -> est", self.U_expert, self.core)  # (E, r3, r2)

        # Gather selected expert matrices: (B_flat, top_k, r3, r2)
        selected_G = G_experts[top_k_indices]

        # Vectorised expert compute: (B_flat, r3) × (B_flat, top_k, r3, r2) → (B_flat, top_k, r2)
        contributions = mx.einsum("br, bkrs -> bks", x_shared, selected_G)

        # Weighted sum over top-k
        x_core = (contributions * top_k_probs[:, :, None]).sum(axis=1)  # (B_flat, r2)

        out = (x_core @ self.U_out) + self.bias                 # (B_flat, dim_out)
        return out.reshape(*orig_shape[:-1], -1), 0.0, 0.0
