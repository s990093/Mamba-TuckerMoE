"""
Transformer block with Grouped-Query Attention (GQA) for efficient inference.
"""

import math
import mlx.core as mx
import mlx.nn as nn
from .ops import RMSNorm, LayerScale, silu_gating
from .tucker_moe import TuckerMoE


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA): efficient variant of multi-head attention.

    Reduces KV cache size by grouping query heads to share KV heads.
    """

    def __init__(self, d_model, n_heads, num_kv_heads):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_groups = n_heads // num_kv_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % num_kv_heads == 0, "n_heads must be divisible by num_kv_heads"

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, (d_model // self.kv_groups), bias=False)
        self.v_proj = nn.Linear(d_model, (d_model // self.kv_groups), bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x, kv_cache=None):
        """
        GQA forward pass.

        Args:
            x: Input tensor (B, L, d_model)
            kv_cache: Optional KV cache for decoding

        Returns:
            out: Output tensor (B, L, d_model)
            new_kv_cache: Updated KV cache
        """
        B, L, _ = x.shape

        # Project Q, K, V
        Q = self.q_proj(x)  # (B, L, d_model)
        K = self.k_proj(x)  # (B, L, d_model // kv_groups)
        V = self.v_proj(x)  # (B, L, d_model // kv_groups)

        # Reshape for multi-head
        Q = Q.reshape(B, L, self.n_heads, self.head_dim)
        K = K.reshape(B, L, self.num_kv_heads, self.head_dim)
        V = V.reshape(B, L, self.num_kv_heads, self.head_dim)

        # Use MLX's efficient attention
        # Expand KV to match Q heads via repetition
        K_expanded = mx.repeat(K, self.kv_groups, axis=2)  # (B, L, n_heads, head_dim)
        V_expanded = mx.repeat(V, self.kv_groups, axis=2)  # (B, L, n_heads, head_dim)

        # Apply attention
        out = mx.fast.scaled_dot_product_attention(
            Q, K_expanded, V_expanded,
            scale=1.0 / math.sqrt(self.head_dim)
        )  # (B, L, n_heads, head_dim)

        # Reshape back
        out = out.reshape(B, L, self.d_model)

        # Output projection
        out = self.out_proj(out)

        # KV cache management (optional, for future use)
        new_kv_cache = None

        return out, new_kv_cache

    def parameters(self):
        """Return trainable parameters."""
        return {
            "q_proj": self.q_proj.parameters(),
            "k_proj": self.k_proj.parameters(),
            "v_proj": self.v_proj.parameters(),
            "out_proj": self.out_proj.parameters(),
        }


class TransformerBlock(nn.Module):
    """
    Transformer block with GQA attention and optional MoE FFN.

    Args:
        d_model: Model dimension
        n_heads: Number of query heads
        num_kv_heads: Number of KV heads (for GQA)
        ffn_config: FFN config (use_moe, moe_config, or standard MLX FFN)
        layer_scale_init: Initial value for LayerScale
    """

    def __init__(
        self,
        d_model,
        n_heads=8,
        num_kv_heads=4,
        use_moe=False,
        moe_config=None,
        ffn_hidden_dim=None,
        layer_scale_init=1e-2,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.num_kv_heads = num_kv_heads
        self.use_moe = use_moe

        if ffn_hidden_dim is None:
            ffn_hidden_dim = 4 * d_model

        # Attention
        self.attn = GroupedQueryAttention(d_model, n_heads, num_kv_heads)
        self.attn_norm = RMSNorm(d_model)

        # FFN
        if use_moe and moe_config:
            self.ffn = TuckerMoE(
                d_model, d_model,
                num_experts=moe_config.get("num_experts", 8),
                top_k=moe_config.get("top_k", 2),
                r1=moe_config.get("r1", 4),
                r2=moe_config.get("r2", 512),
                r3=moe_config.get("r3", 256),
            )
        else:
            # Standard FFN with SiLU gating
            self.ffn = StandardFFN(d_model, ffn_hidden_dim)

        self.ffn_norm = RMSNorm(d_model)

        # Layer scaling
        self.layer_scale = LayerScale(d_model, init_value=layer_scale_init)

    def __call__(self, x, kv_cache=None, training=False):
        """
        Forward pass.

        Args:
            x: Input (B, L, d_model)
            kv_cache: Optional KV cache for decoding
            training: Training mode flag

        Returns:
            out: Output (B, L, d_model)
            new_kv_cache: Updated KV cache
            aux_loss: MoE auxiliary loss (0 if no MoE)
        """
        residual = x

        # Attention block
        x_norm = self.attn_norm(x)
        attn_out, new_kv_cache = self.attn(x_norm, kv_cache)
        x = residual + self.layer_scale(attn_out)

        # FFN block
        residual = x
        x_norm = self.ffn_norm(x)

        if self.use_moe:
            ffn_out, lb_loss, z_loss = self.ffn(x_norm, training=training)
            aux_loss = lb_loss + z_loss
        else:
            ffn_out = self.ffn(x_norm)
            aux_loss = 0.0

        x = residual + self.layer_scale(ffn_out)

        return x, new_kv_cache, aux_loss

    def parameters(self):
        """Return trainable parameters."""
        params = {
            "attn": self.attn.parameters(),
            "attn_norm": self.attn_norm.parameters(),
            "ffn_norm": self.ffn_norm.parameters(),
            "layer_scale": self.layer_scale.parameters(),
        }
        if self.use_moe:
            params["ffn"] = self.ffn.parameters()
        else:
            params["ffn"] = self.ffn.parameters()
        return params


class StandardFFN(nn.Module):
    """
    Standard feed-forward network with SiLU gating.
    """

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        """Apply SiLU(gate) * up -> down."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(silu_gating(gate, up))

    def parameters(self):
        """Return trainable parameters."""
        return {
            "gate_proj": self.gate_proj.parameters(),
            "up_proj": self.up_proj.parameters(),
            "down_proj": self.down_proj.parameters(),
        }
