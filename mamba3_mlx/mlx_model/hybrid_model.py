import math

import mlx.core as mx
import mlx.nn as nn

from .ops import RMSNorm, scaled_tanh
from .mamba_block import Mamba3Block
from .transformer_block import TransformerBlock


class TrueHybridMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        for _ in range(config.num_layers):
            for _ in range(config.mamba_ratio):
                layers.append(Mamba3Block(config))
            layers.append(TransformerBlock(config))
        self.layers = layers

    def __call__(self, x, states=None):
        new_states = []
        # During multi-token prefill (L > 1) each block's G_selected tensor in
        # TuckerMoE is (L, top_k, r3, r2). Without an explicit eval boundary,
        # MLX keeps every block's intermediate tensors alive simultaneously in
        # the lazy graph, causing 3-4 GB spikes even for short sequences.
        # Evaluating after each block forces the graph to be freed block-by-block.
        prefill = (x.shape[1] > 1)
        for i, blk in enumerate(self.layers):
            st = states[i] if states is not None else None
            x, new_st = blk(x, state=st)
            if prefill and new_st is not None:
                state_vals = [v for v in new_st.values() if v is not None]
                mx.eval(x, *state_vals)
            new_states.append(new_st)
        return x, new_states


class Mamba3LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.backbone = TrueHybridMamba(config)
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        # head weight is tied to embed.weight (set at load time)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.inv_sqrt_d = 1.0 / math.sqrt(config.d_model)

    def precompute(self) -> None:
        """Precompute all cached tensors after weight loading.

        For each Mamba3Block:
          - Calls precompute_G_experts() on its two TuckerMoE layers.
          - Calls block.precompute() to cache D_expand and compiled decode.
        For each TransformerBlock:
          - Calls precompute_G_experts() on its three FFN TuckerMoE layers.
          - Calls block.precompute() to compile decode sub-functions.
        Model-level:
          - Compiles norm+head+scaled_tanh for the fixed-shape decode path.
        """
        for layer in self.backbone.layers:
            if isinstance(layer, Mamba3Block):
                layer.x_up_proj.precompute_G_experts()
                layer.out_proj.precompute_G_experts()
                layer.precompute()
            elif isinstance(layer, TransformerBlock):
                layer.ffn.gate_proj.precompute_G_experts()
                layer.ffn.up_proj.precompute_G_experts()
                layer.ffn.down_proj.precompute_G_experts()
                layer.precompute()

        # Compile norm + head + scaled_tanh for decode (fixed shape (1,1,d_model)).
        self._compiled_head = mx.compile(self._head_forward)
        _dummy = mx.zeros((1, 1, self.config.d_model), dtype=mx.bfloat16)
        mx.eval(self._compiled_head(_dummy))

    def _head_forward(self, x):
        """norm + head projection + scaled_tanh — fixed shape for decode."""
        h = self.norm(x)
        logits = self.head(h * self.inv_sqrt_d).astype(mx.float32)
        return scaled_tanh(logits, 30.0)

    def __call__(self, input_ids, states=None):
        x = self.embed(input_ids)
        x, new_states = self.backbone(x, states=states)
        # Use compiled head for decode (L=1, fixed shape).
        if x.shape[1] == 1 and getattr(self, '_compiled_head', None) is not None:
            logits = self._compiled_head(x)
        else:
            h = self.norm(x)
            logits = self.head(h * self.inv_sqrt_d).astype(mx.float32)
            logits = scaled_tanh(logits, 30.0)
        return logits, new_states
