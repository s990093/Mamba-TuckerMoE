"""dLLM (diffusion-LLM) port of the Mamba-TuckerMoE stack — additive package.

Implements DLLM_MLX_PORT.md on top of the existing ``mamba3_mlx.mlx_model``
without touching it: the AR ``Mamba3Block`` / ``TuckerMoE`` are reused as-is;
only the transformer block becomes bidirectional (②), the vocab gains a
``[MASK]`` slot (①), and generation becomes iterative unmasking (④).  The
masked-CE training objective (③) lives in ``loss.py`` for the future training
move.  No checkpoint is loaded — the model self-initialises with random
weights so inference + the high-performance compiled path can be exercised
end-to-end while the real weights are still training.
"""

from .config import (ASSISTANT_HEADER, BASE_VOCAB, MASK_ID, STOP_IDS,
                     DiffusionGenConfig, DLLMConfig)
from .dllm_model import DLLMBackbone, DLLMModel, build_random_dllm
from .generate import (DLLMResult, ban_mask, cosine_fill_schedule,
                       diffusion_generate, diffusion_generate_cached,
                       iterative_unmask, trim_to_stop)
from .samplers import (StableConfidentStop, entropy_bound_select,
                       linear_temperature, token_entropy)
from .self_conditioning import DLLMSelfConditioning, soft_embeddings
from .static_dllm import StaticDLLM

__all__ = [
    "DLLMConfig", "DiffusionGenConfig", "MASK_ID", "BASE_VOCAB", "STOP_IDS",
    "ASSISTANT_HEADER", "DLLMModel", "DLLMBackbone", "build_random_dllm",
    "iterative_unmask", "diffusion_generate", "diffusion_generate_cached",
    "cosine_fill_schedule", "ban_mask", "trim_to_stop", "DLLMResult", "StaticDLLM",
    "token_entropy", "linear_temperature", "entropy_bound_select",
    "StableConfidentStop", "DLLMSelfConditioning", "soft_embeddings",
]
