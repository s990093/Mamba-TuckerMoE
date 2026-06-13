"""Architecture constants — mirror of mamba3_mlx/utils/config.py."""
from dataclasses import dataclass, field


@dataclass
class Mamba3Config:
    d_model:     int = 768
    expand:      int = 2
    d_head:      int = 64
    d_state:     int = 64
    n_groups:    int = 1
    mimo_rank:   int = 4
    num_layers:  int = 6         # transformer blocks
    mamba_ratio: int = 4         # mamba blocks per transformer
    chunk_size:  int = 64
    vocab_size:  int = 32007
    ffn_expand:  int = 6
    rms_norm_eps: float = 1e-5
    # MoE
    kmoe_num_experts: int = 8
    kmoe_top_k:       int = 2
    kmoe_r1:          int = 32
    kmoe_r2:          int = 512
    kmoe_r3:          int = 256

    @property
    def d_inner(self): return self.expand * self.d_model
    @property
    def n_heads(self): return self.d_inner // self.d_head
    @property
    def n_total(self): return self.num_layers * (self.mamba_ratio + 1)
    @property
    def n_mamba(self): return self.num_layers * self.mamba_ratio

    def block_types(self):
        """Return list of 'mamba'/'transformer' for each of n_total blocks."""
        types = []
        for _ in range(self.num_layers):
            for _ in range(self.mamba_ratio):
                types.append("mamba")
            types.append("transformer")
        return types
