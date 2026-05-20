"""
Hybrid Mamba3 + Transformer model with TuckerMoE.
Architecture: 4 Mamba blocks + 1 Transformer block per cycle.
"""

import dataclasses
import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .mamba_block import Mamba3Block
from .transformer_block import TransformerBlock
from .ops import RMSNorm


@dataclasses.dataclass
class Mamba3Config:
    """Configuration for Mamba3LanguageModel."""

    # Model dimensions
    d_model: int = 768
    d_state: int = 64
    d_head: int = 64
    n_groups: int = 1
    mimo_rank: int = 4
    expand: int = 4

    # Architecture
    num_layers: int = 15
    mamba_ratio: float = 0.8  # 80% Mamba, 20% Transformer
    vocab_size: int = 32768

    # Mamba-specific
    use_parallel_scan: bool = True
    chunk_size: int = 64
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    layer_scale_init: float = 1e-2

    # MoE-specific
    use_kmoe: bool = True
    kmoe_num_experts: int = 8
    kmoe_top_k: int = 2
    kmoe_r1: int = 4
    kmoe_r2: int = 1024
    kmoe_r3: int = 256
    ffn_expand: int = 6

    # Transformer-specific
    num_kv_heads: int = 4
    rms_norm_eps: float = 1e-5

    # Training
    use_activation_checkpoint: bool = False

    @property
    def n_heads(self):
        """Compute number of attention heads."""
        return (self.expand * self.d_model) // self.d_head

    @property
    def d_inner(self):
        """Compute inner dimension."""
        return self.expand * self.d_model


class TrueHybridMamba(nn.Module):
    """
    Hybrid model alternating Mamba and Transformer blocks.

    Configuration: (4 Mamba, 1 Transformer) repeating.
    """

    def __init__(self, config: Mamba3Config):
        super().__init__()

        self.config = config
        self.num_mamba_per_cycle = 4
        self.num_transformer_per_cycle = 1

        # Calculate layer distribution
        total_cycles = config.num_layers // (
            self.num_mamba_per_cycle + self.num_transformer_per_cycle
        )
        remainder = config.num_layers % (
            self.num_mamba_per_cycle + self.num_transformer_per_cycle
        )

        layers = []
        for _ in range(total_cycles):
            # Add 4 Mamba blocks
            for _ in range(self.num_mamba_per_cycle):
                layers.append(
                    Mamba3Block(
                        d_model=config.d_model,
                        d_state=config.d_state,
                        d_head=config.d_head,
                        n_groups=config.n_groups,
                        mimo_rank=config.mimo_rank,
                        expand=config.expand,
                        dt_min=config.dt_min,
                        dt_max=config.dt_max,
                        dt_init_floor=config.dt_init_floor,
                        layer_scale_init=config.layer_scale_init,
                        chunk_size=config.chunk_size,
                    )
                )
            # Add 1 Transformer block
            layers.append(
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    num_kv_heads=config.num_kv_heads,
                    use_moe=config.use_kmoe,
                    moe_config={
                        "num_experts": config.kmoe_num_experts,
                        "top_k": config.kmoe_top_k,
                        "r1": config.kmoe_r1,
                        "r2": config.kmoe_r2,
                        "r3": config.kmoe_r3,
                    } if config.use_kmoe else None,
                    ffn_hidden_dim=int(config.ffn_expand * config.d_model),
                    layer_scale_init=config.layer_scale_init,
                )
            )

        # Add remainder layers
        for _ in range(remainder):
            layers.append(
                Mamba3Block(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_head=config.d_head,
                    n_groups=config.n_groups,
                    mimo_rank=config.mimo_rank,
                    expand=config.expand,
                    dt_min=config.dt_min,
                    dt_max=config.dt_max,
                    dt_init_floor=config.dt_init_floor,
                    layer_scale_init=config.layer_scale_init,
                    chunk_size=config.chunk_size,
                )
            )

        self.layers = layers

    def __call__(self, x, states=None, training=False):
        """
        Forward pass.

        Args:
            x: Input (B, L, d_model)
            states: List of state dicts for Mamba layers (for decode)
            training: Training mode flag

        Returns:
            out: Output (B, L, d_model)
            new_states: Updated state dicts
            aux_loss: Auxiliary loss from MoE layers
        """
        if states is None:
            states = [None] * len(self.layers)

        new_states = []
        aux_loss = 0.0

        for i, layer in enumerate(self.layers):
            if isinstance(layer, Mamba3Block):
                x, state = layer(x, state=states[i], training=training)
                new_states.append(state)
            elif isinstance(layer, TransformerBlock):
                x, _, loss = layer(x, kv_cache=None, training=training)
                new_states.append(None)
                aux_loss = aux_loss + loss

        return x, new_states, aux_loss

    def parameters(self):
        """Return trainable parameters."""
        params = {}
        for i, layer in enumerate(self.layers):
            params[f"layer_{i}"] = layer.parameters()
        return params


class Mamba3LanguageModel(nn.Module):
    """
    Full Mamba3 language model with embedding and output projection.

    Supports:
    - Prefill: Process full prompt sequence
    - Decode: Generate tokens one-by-one with state management
    """

    def __init__(self, config: Mamba3Config):
        super().__init__()

        self.config = config

        # Embedding
        self.embed = nn.Linear(config.vocab_size, config.d_model, bias=False)
        self.embed_scale = math.sqrt(config.d_model)

        # Backbone
        self.backbone = TrueHybridMamba(config)

        # Output layers
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Share embedding weight with lm_head
        # This is handled during weight loading

    def __call__(self, token_ids, states=None, training=False):
        """
        Forward pass for both prefill and decode.

        Args:
            token_ids: Token IDs (B, L) or (B, 1) for decode
            states: List of state dicts (for decode)
            training: Training mode flag

        Returns:
            logits: (B, L, vocab_size)
            new_states: Updated state dicts
            aux_loss: Auxiliary loss
        """
        B, L = token_ids.shape

        # Embed tokens using transposed weight for indexing
        # embed.weight is (d_model, vocab_size), transpose to (vocab_size, d_model) for indexing
        embed_weight_t = self.embed.weight.T  # (vocab_size, d_model)
        x = embed_weight_t[token_ids] * self.embed_scale  # (B, L, d_model)

        # Backbone
        x, new_states, aux_loss = self.backbone(x, states=states, training=training)

        # Output
        x = self.norm(x)
        logits = x @ self.lm_head.weight.T  # (B, L, vocab_size)

        return logits, new_states, aux_loss

    def parameters(self):
        """Return trainable parameters."""
        return {
            "embed": {"weight": self.embed.weight},
            "backbone": self.backbone.parameters(),
            "norm": self.norm.parameters(),
            "lm_head": {"weight": self.lm_head.weight},
        }

    @staticmethod
    def from_pretrained(checkpoint_path, config=None):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to .npz or .pt file
            config: Mamba3Config (if None, loads from checkpoint)

        Returns:
            model: Loaded model
            weights: Dictionary of weights
        """
        if checkpoint_path.endswith(".npz"):
            weights = dict(np.load(checkpoint_path))
            weights = {k: mx.array(v) for k, v in weights.items()}
        else:
            raise NotImplementedError("Only .npz format supported for now")

        # Load config from weights metadata if available
        if config is None:
            config = Mamba3Config()

        model = Mamba3LanguageModel(config)
        return model, weights

    @staticmethod
    def save_weights(model, output_path):
        """
        Save model weights to .npz file.

        Args:
            model: Mamba3LanguageModel instance
            output_path: Path to save .npz file
        """
        params = model.parameters()
        # Flatten nested dict and convert to numpy
        flat_params = {}

        def flatten_dict(d, prefix=""):
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    flatten_dict(v, full_key)
                else:
                    flat_params[full_key] = np.array(v)

        flatten_dict(params)
        np.savez(output_path, **flat_params)
