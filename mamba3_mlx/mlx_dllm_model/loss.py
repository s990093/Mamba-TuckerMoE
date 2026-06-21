"""dLLM change ③ — absorbing-diffusion masked-CE training objective.

Training only; not used by any inference path.  Included so the training loop
(once the dLLM weights move here) has the loss alongside the model.  Mirrors
DLLM_MLX_PORT.md §③ exactly:

  * sample a per-sequence noise ratio t ∈ (0, 1];
  * mask response tokens where uniform < t (prompt is never masked);
  * CE only on masked positions, weighted by 1/t (fewer masks ⇒ each worth
    more), averaged over the response length.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import MASK_ID


def dllm_loss(model, x, resp_mask, *, mask_id: int = MASK_ID, eps: float = 1e-3):
    """x:(B,T) int ids,  resp_mask:(B,T) bool (True over the response region)."""
    B, T = x.shape
    t = mx.random.uniform(shape=(B, 1)) * (1 - eps) + eps           # (B,1)
    noise = mx.random.uniform(shape=(B, T))
    masked = resp_mask & (noise < t)                                # (B,T) bool
    noisy = mx.where(masked, mask_id, x)

    logits = model(noisy)                                           # (B,T,V) bidirectional
    ce = nn.losses.cross_entropy(logits, x, reduction="none")      # (B,T)
    ce_masked = (ce * masked).sum(axis=1)                          # (B,)
    per_seq = (1.0 / t.squeeze(1)) * ce_masked / mx.maximum(resp_mask.sum(1), 1)
    return per_seq.mean()
