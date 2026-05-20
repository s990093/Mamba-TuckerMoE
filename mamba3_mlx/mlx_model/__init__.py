"""MLX model components for Mamba3 + TuckerMoE."""

from .ops import (
    tanh_approx,
    scaled_tanh,
    silu,
    silu_gating,
    rope,
    RMSNorm,
    LayerScale,
    chunk_parallel_scan,
)
from .tucker_moe import TuckerMoE, MixtralMoEFeedForward
from .mamba_block import Mamba3Block
from .transformer_block import TransformerBlock, GroupedQueryAttention, StandardFFN
from .hybrid_model import Mamba3LanguageModel, TrueHybridMamba, Mamba3Config

__all__ = [
    # Ops
    "tanh_approx",
    "scaled_tanh",
    "silu",
    "silu_gating",
    "rope",
    "RMSNorm",
    "LayerScale",
    "chunk_parallel_scan",
    # MoE
    "TuckerMoE",
    "MixtralMoEFeedForward",
    # Blocks
    "Mamba3Block",
    "TransformerBlock",
    "GroupedQueryAttention",
    "StandardFFN",
    # Models
    "Mamba3LanguageModel",
    "TrueHybridMamba",
    "Mamba3Config",
]
