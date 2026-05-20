"""MLX Mamba3 + TuckerMoE: Compute-bound inference on Apple Silicon."""

from mlx_model import (
    Mamba3LanguageModel,
    Mamba3Config,
    TuckerMoE,
    Mamba3Block,
    TransformerBlock,
)
from inference import AutoregressiveGenerator, TextSampler
from utils import GenerationConfig, ModelConfig, get_generation_args

__version__ = "0.1.0"
__all__ = [
    "Mamba3LanguageModel",
    "Mamba3Config",
    "TuckerMoE",
    "Mamba3Block",
    "TransformerBlock",
    "AutoregressiveGenerator",
    "TextSampler",
    "GenerationConfig",
    "ModelConfig",
    "get_generation_args",
]
