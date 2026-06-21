"""Self-conditioning (cross-step) — MLX port of DiffusionGemmaSelfConditioning.

Feeds the previous denoising step's prediction back into the current forward so
the model "remembers" what it last guessed:

    soft_emb = softmax(prev_logits) @ embed_table      # expected embedding
    fused    = post_norm( inputs_embeds + gated_mlp(pre_norm(soft_emb)) )

Reference: transformers/models/diffusion_gemma/modeling_diffusion_gemma.py
  · DiffusionGemmaSelfConditioning (gated MLP + pre/post RMS norm)
  · DiffusionGemmaDecoderModel.forward (soft_embeddings = softmax(logits) @ W_e)

NOTE: this adds parameters (gate/up/down). The current AR checkpoint does not
contain them, so self-conditioning is OFF by default (``DLLMConfig.self_
conditioning = False``).  It is wired and ready for when a self-conditioned
dLLM is trained into this stack — the inference loop will then feed each step's
logits into the next forward automatically.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..mlx_model.ops import RMSNorm, silu_gating


class DLLMSelfConditioning(nn.Module):
    def __init__(self, d_model: int, d_ff: int, eps: float = 1e-5):
        super().__init__()
        self.pre_norm = RMSNorm(d_model, eps=eps)
        self.post_norm = RMSNorm(d_model, eps=eps)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, inputs_embeds, sc_signal):
        """inputs_embeds, sc_signal: (B, L, d_model)."""
        normed = self.pre_norm(sc_signal)
        sc = self.down_proj(silu_gating(self.gate_proj(normed), self.up_proj(normed)))
        return self.post_norm(inputs_embeds + sc)


def soft_embeddings(prev_logits, embed_weight):
    """softmax(prev_logits) @ embed_table → expected (soft) embedding per pos.

    prev_logits: (B, L, V) float; embed_weight: (V, d_model).
    """
    p = mx.softmax(prev_logits.astype(mx.float32), axis=-1).astype(embed_weight.dtype)
    return mx.matmul(p, embed_weight)
