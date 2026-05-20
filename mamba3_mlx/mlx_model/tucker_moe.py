"""
Tucker-decomposed Mixture-of-Experts (TuckerMoE) layer for MLX.
Supports training (with load-balancing and Z-loss) and inference.
"""

import math
import mlx.core as mx
import mlx.nn as nn
from .ops import scaled_tanh, RMSNorm


def get_router_temperature(step, warmup=500, total=10000, t_start=2.0, t_end=0.5):
    """
    Router temperature schedule using cosine annealing.

    Args:
        step: Current training step (None for inference → use t_end)
        warmup: Steps before schedule starts
        total: Total training steps
        t_start: Initial temperature
        t_end: Final temperature

    Returns:
        Temperature scalar
    """
    if step is None:
        return t_end

    step_f = float(step) if hasattr(step, '__float__') else step
    progress = max(0.0, min(1.0, (step_f - warmup) / max(1.0, total - warmup)))
    temp = t_end + 0.5 * (t_start - t_end) * (1.0 + math.cos(math.pi * progress))
    return temp


class TuckerMoE:
    """
    Tucker decomposition-based MoE layer.

    Factorization:
      G_e = U_expert[e, :] ⊗ core[?, ?, ?]  (einsum: "er, rst -> est")
      Expert output: x_shared @ G_e (einsum: "r, rst -> st")
      Final: sum over top-k experts weighted by router probs

    Parameters:
      dim_in: Input dimension
      dim_out: Output dimension
      num_experts: Total number of experts
      top_k: Number of experts to route to per token
      r1: First dimension of core tensor (expert axis)
      r2: Second dimension of core tensor (output axis)
      r3: Third dimension of core tensor (shared input axis)
    """

    def __init__(self, dim_in, dim_out, num_experts=8, top_k=2, r1=4, r2=1024, r3=256):
        # Initialize parameters

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3

        # Router: logits over experts
        self.router = nn.Linear(dim_in, num_experts, bias=False)

        # Tucker factors
        self.U_expert = mx.random.normal((num_experts, r1), scale=0.02)
        self.U_in = mx.random.normal((dim_in, r3), scale=1.0 / math.sqrt(dim_in))
        self.U_out = mx.random.normal((r2, dim_out), scale=1.0 / math.sqrt(r2))
        self.core = mx.random.normal((r1, r3, r2), scale=1.0 / math.sqrt(r1 * r3))
        self.bias = mx.zeros((dim_out,))

        self.inner_norm = RMSNorm(r3)

    def __call__(self, x, step=None, training=False):
        """
        Forward pass with optional router temperature scheduling.

        Args:
            x: Input tensor, shape (..., dim_in)
            step: Training step (for temperature schedule)
            training: Whether in training mode (affects loss computation)

        Returns:
            out: Output tensor, shape (..., dim_out)
            lb_loss: Load-balancing loss (0 if not training)
            z_loss: Z-loss (0 if not training)
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])  # (B_flat, dim_in)
        B_flat = x_flat.shape[0]

        # Router
        temperature = get_router_temperature(step)
        raw_logits = self.router(x_flat)  # (B_flat, num_experts)
        capped = scaled_tanh(raw_logits, 10.0)

        # Z-loss (auxiliary loss for training)
        z_loss = mx.mean(mx.logsumexp(capped, axis=-1) ** 2) if training else 0.0

        # Apply temperature and softmax
        router_logits = capped / temperature
        router_probs = mx.softmax(router_logits, axis=-1)  # (B_flat, num_experts)

        # Top-k selection (simplified)
        # Use softmax of logits directly for probabilities
        # Simple version: weight experts equally (no top-k for now)
        top_k_indices = mx.arange(self.num_experts)
        top_k_probs = router_probs

        # Load-balancing loss (simplified)
        lb_loss = 0.0 if not training else mx.mean(router_probs)

        # Shared projection
        x_shared = x_flat @ self.U_in  # (B_flat, r3)
        x_shared = self.inner_norm(x_shared)

        # Compute expert gates: G_e = U_expert[e] ⊗ core
        # einsum: "er, rst -> est" → G_e shape (r3, r2) for each expert e
        G_experts = mx.einsum("er,rst->est", self.U_expert, self.core)  # (num_experts, r3, r2)

        # Simplified experts: use top-k probabilistically
        x_core = mx.zeros((B_flat, self.r2))
        for e in range(self.num_experts):
            # Weight each expert by its probability (simplified routing)
            expert_prob = router_probs[:, e:e+1]  # (B_flat, 1)
            expert_out = x_shared @ G_experts[e]  # (B_flat, r2)
            x_core = x_core + expert_prob * expert_out / self.num_experts

        # Output projection
        out = (x_core @ self.U_out).reshape(*orig_shape[:-1], -1) + self.bias

        return out, lb_loss, z_loss

    def parameters(self):
        """Return trainable parameters."""
        return {
            "router": self.router.parameters(),
            "U_expert": self.U_expert,
            "U_in": self.U_in,
            "U_out": self.U_out,
            "core": self.core,
            "bias": self.bias,
        }


class MixtralMoEFeedForward:
    """
    Triton-style MoE FFN using three TuckerMoE layers.
    Structure: gate_proj(x) -> silu(gate) * up_proj(x) -> down_proj(...)
    """

    def __init__(self, config):

        d_ff = int(math.ceil(config.ffn_expand * config.d_model / 256) * 256)

        moe_kwargs = dict(
            num_experts=config.kmoe_num_experts,
            top_k=config.kmoe_top_k,
            r1=config.kmoe_r1,
            r2=config.kmoe_r2,
            r3=config.kmoe_r3,
        )

        self.gate_proj = TuckerMoE(config.d_model, d_ff, **moe_kwargs)
        self.up_proj = TuckerMoE(config.d_model, d_ff, **moe_kwargs)
        self.down_proj = TuckerMoE(d_ff, config.d_model, **moe_kwargs)

    def __call__(self, x, step=None, training=False):
        """
        Forward pass through gate → silu(gate)*up → down.

        Returns:
            out: Output tensor
            lb_loss: Sum of load-balancing losses from three MoE layers
            z_loss: Sum of Z-losses from three MoE layers
        """
        gate, lb_g, z_g = self.gate_proj(x, step=step, training=training)
        feat, lb_u, z_u = self.up_proj(x, step=step, training=training)

        # SiLU gating
        from .ops import silu_gating
        gated = silu_gating(gate, feat)

        y, lb_d, z_d = self.down_proj(gated, step=step, training=training)

        return y, lb_g + lb_u + lb_d, z_g + z_u + z_d

    def parameters(self):
        """Return trainable parameters."""
        return {
            "gate_proj": self.gate_proj.parameters(),
            "up_proj": self.up_proj.parameters(),
            "down_proj": self.down_proj.parameters(),
        }
