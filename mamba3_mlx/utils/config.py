from dataclasses import dataclass


@dataclass
class Mamba3Config:
    d_model: int = 768
    d_state: int = 64
    d_head: int = 64
    n_groups: int = 1
    mimo_rank: int = 4
    expand: int = 2
    num_layers: int = 6
    mamba_ratio: int = 4
    num_kv_heads: int = 4
    chunk_size: int = 64
    rms_norm_eps: float = 1e-5
    vocab_size: int = 32007
    kmoe_num_experts: int = 8
    kmoe_top_k: int = 2
    kmoe_r1: int = 32
    kmoe_r2: int = 512
    kmoe_r3: int = 256
    ffn_expand: int = 6
    layer_scale_init: float = 1e-2

    @property
    def d_inner(self):
        return self.expand * self.d_model

    @property
    def n_heads(self):
        return self.d_inner // self.d_head


@dataclass
class GenerationConfig:
    max_tokens: int = 256
    temperature: float = 0.426   # fallback; per-mode defaults live in utils/mode_configs.py
    top_k: int = 20
    top_p: float = 0.981
    min_p: float = 0.067
    rep_pen: float = 1.146
    pres_pen: float = 0.143
    freq_pen: float = 0.133
    repeat_last_n: int = 128
    seed: int = 0
