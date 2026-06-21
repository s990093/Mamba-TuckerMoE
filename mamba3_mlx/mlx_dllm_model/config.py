"""dLLM (diffusion-LLM) config + shared vocab constants.

Port of sft_cot_bundle/DLLM_MLX_PORT.md to the mamba3_mlx stack.  Only the
4 documented changes apply (§0); everything else reuses the AR architecture:

    ① [MASK] token            → vocab 32007 → 32008 (this file)
    ② bidirectional attention → DLLMTransformerBlock (bidirectional_block.py)
    ③ masked-CE training loss  → loss.py (training only; not used at inference)
    ④ iterative unmasking      → generate.py / static_dllm.py

``DLLMConfig`` subclasses the AR ``Mamba3Config`` so the reused ``Mamba3Block``
and ``TuckerMoE`` read byte-identical hyper-parameters.  Only the vocab size
grows by one and the bidirectional flag is added.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..utils.config import Mamba3Config

# ── Vocab constants — must match BOTH ends (DLLM_MLX_PORT.md §詞表常數) ─────────
MASK_ID = 32007            # appended right after the 32007-id base vocab
PAD_ID = 32006
IM_END = 32001             # <|im_end|>
THINK_OPEN = 32002         # <think>
THINK_CLOSE = 32003        # </think>
FINAL_OPEN = 32004         # <final>
FINAL_CLOSE = 32005        # </final>
# <|im_start|>assistant\n  → response region starts right after these ids
ASSISTANT_HEADER = (32000, 465, 22137, 13)
BASE_VOCAB = 32007         # AR vocab size before the [MASK] slot is added

# tokens that terminate the generated region (used to trim §④ output)
STOP_IDS = (FINAL_CLOSE, IM_END)


@dataclass
class DLLMConfig(Mamba3Config):
    """AR config + the dLLM-specific fields.

    Inherits d_model / n_heads / kmoe_* / chunk_size / … unchanged so the
    reused Mamba3Block and TuckerMoE behave identically to the AR model.
    """
    vocab_size: int = BASE_VOCAB + 1   # 32008  (= 32007 base + [MASK])
    mask_id: int = MASK_ID
    bidirectional: bool = True         # ② flip to False only for parity checks
    # Self-conditioning (DiffusionGemma-style) — adds params, needs training,
    # so OFF for the current AR checkpoint.  See self_conditioning.py.
    self_conditioning: bool = False


@dataclass
class DiffusionGenConfig:
    """Inference-time diffusion controls (DiffusionGemma-inspired defaults).

    ``sampler``: "cosine" = fixed schedule (DLLM_MLX_PORT.md §④);
                 "entropy" = adaptive entropy-bound acceptance (recommended).
    """
    gen_len: int = 64                  # canvas length G (the [MASK] span)
    steps: int = 16                    # max denoising steps T
    sampler: str = "entropy"           # "entropy" | "cosine"
    entropy_bound: float = 0.1         # EntropyBoundSamplerConfig default
    # linear temperature schedule (t_max early → t_min late); t<=0 ⇒ greedy
    t_min: float = 0.4
    t_max: float = 0.8
    # adaptive early stop (StableAndConfidentStoppingCriteria); enable via CLI
    adaptive_stop: bool = False
    stability_threshold: int = 1
    confidence_threshold: float = 0.005
    seed: int = 0
