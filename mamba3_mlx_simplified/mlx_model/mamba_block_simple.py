"""
Simplified Mamba3 block for rapid prototyping.
Full SSM implementation can be added later.
"""

import mlx.core as mx
import mlx.nn as nn
from .ops import RMSNorm, LayerScale, silu


class SimpleMambaBlock(nn.Module):
    """Simplified Mamba block (skips full SSM for now)."""

    def __init__(self, d_model, d_inner=None, expand=4):
        super().__init__()
        if d_inner is None:
            d_inner = expand * d_model

        self.d_model = d_model
        self.d_inner = d_inner

        # Simple linear projections
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # Normalization
        self.norm = RMSNorm(d_model)
        self.layer_scale = LayerScale(d_model)

    def __call__(self, x, state=None, training=False):
        """Simplified forward pass."""
        residual = x
        x_norm = self.norm(x)

        # Project
        x_proj = self.in_proj(x_norm)
        gate, feat = x_proj[..., :self.d_inner], x_proj[..., self.d_inner:]

        # Gate and output
        y = silu(gate) * feat
        out = self.out_proj(y)

        # Residual
        out = residual + self.layer_scale(out)

        # Return output and dummy state
        new_state = {"h_prev": None, "dt_cumsum_prev": None}
        return out, new_state
