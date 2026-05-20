"""
Configuration dataclasses for generation and inference.
"""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class GenerationConfig:
    """Configuration for text generation."""

    # Generation limits
    max_tokens: int = 256

    # Sampling hyperparameters
    # Table of all parameters with defaults (per plan section VIII):
    # | Parameter         | Default | Description                       |
    # | temp              | 0.8     | Temperature                       |
    # | top_k             | 40      | Top-k filtering                   |
    # | top_p             | 0.9     | Nucleus filtering                 |
    # | min_p             | 0.05    | Min-p relative threshold          |
    # | rep_pen           | 1.1     | Repetition penalty coefficient    |
    # | pres_pen          | 0.0     | Presence penalty                  |
    # | freq_pen          | 0.02    | Frequency penalty                 |
    # | repeat_last_n     | 64      | Penalty window size               |
    # | dtype             | bf16    | Weight/compute dtype              |
    # | kv_dtype          | auto    | KV cache dtype (auto→dtype)       |
    # | max_tokens        | 256     | Max generation length             |
    # | full-decode-compile | True  | Enable mx.compile for decode      |

    temp: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    min_p: float = 0.05
    rep_pen: float = 1.1
    pres_pen: float = 0.0
    freq_pen: float = 0.02
    repeat_last_n: int = 64

    # Data types
    dtype: str = "bf16"  # Options: fp32, bf16, fp16
    kv_dtype: str = "auto"  # Options: auto, bf16, fp16, fp32

    # Optimization flags
    full_decode_compile: bool = True

    # Speculative decoding
    speculative: bool = False
    speculative_k: int = 5

    # Other
    greedy: bool = False
    eos_token_id: Optional[int] = None


@dataclasses.dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # From Mamba3Config
    d_model: int = 768
    d_state: int = 64
    d_head: int = 64
    n_groups: int = 1
    mimo_rank: int = 4
    expand: int = 4
    num_layers: int = 15
    vocab_size: int = 32768

    # Mamba
    use_parallel_scan: bool = True
    chunk_size: int = 64
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    layer_scale_init: float = 1e-2

    # MoE
    use_kmoe: bool = True
    kmoe_num_experts: int = 8
    kmoe_top_k: int = 2
    kmoe_r1: int = 4
    kmoe_r2: int = 1024
    kmoe_r3: int = 256
    ffn_expand: int = 6

    # Transformer
    num_kv_heads: int = 4
    rms_norm_eps: float = 1e-5

    # Training
    use_activation_checkpoint: bool = False
