# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GenerationConfig:
    # Basic
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    min_p: float = 0.05

    # Penalties
    repetition_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.02
    repetition_window: int = 64

    # Stop tokens (filled from tokenizer at runtime)
    stop_token_ids: List[int] = field(default_factory=list)

    # Router temperature override (None = use schedule)
    router_temperature: Optional[float] = None

    # Speculative decoding
    num_speculative_tokens: int = 5

    # Minimum tokens before stop tokens are honoured
    min_new_tokens: int = 0

    # Compilation
    full_decode_compile: bool = False

    # KV cache dtype
    kv_dtype: str = "auto"   # "auto" | "bf16" | "fp16" | "fp32"

    # If True, don't halt generation when a stop token is sampled.
    # Generation continues until max_new_tokens or external abort.
    no_eos_stop: bool = False

    # Metal optimizations (experimental)
    use_metal_sampling: bool = False  # if True, use Metal-optimized decode sampling
    use_metal_scan: bool = False      # if True, use Metal-optimized SSM scan
    metal_threadgroup_size: int = 256  # Metal threadgroup size for sampling kernels
