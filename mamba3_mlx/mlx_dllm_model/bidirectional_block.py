"""dLLM change ② — bidirectional (non-causal) transformer block.

Subclasses the AR ``TransformerBlock`` and overrides only ``__call__``: the
attention mask is dropped (``mask=None``) so every position attends to the
whole sequence.  All weights / projections / the TuckerMoE FFN are inherited
unchanged.

The Mamba scan stays unidirectional — this is the deliberate "partial
bidirectional" design from DLLM_MLX_PORT.md §0 (full-bidirectional would need
a BiMamba scan, which this port does not do).

There is no KV cache and no L=1 decode fast path here: a dLLM forward always
re-reads the entire (prompt + masked-response) sequence each unmasking
iteration, so the parent's incremental-decode machinery is never used.
"""

from __future__ import annotations

import mlx.core as mx

from ..mlx_model.transformer_block import TransformerBlock


class DLLMTransformerBlock(TransformerBlock):
    def __call__(self, x, state=None):
        B, L, D = x.shape
        residual = x
        nx = self.norm_attn(x)
        q = self.q_proj(nx).reshape(B, L, self.num_heads, 64).transpose(0, 2, 1, 3)
        k = self.k_proj(nx).reshape(B, L, self.num_kv_heads, 64).transpose(0, 2, 1, 3)
        v = self.v_proj(nx).reshape(B, L, self.num_kv_heads, 64).transpose(0, 2, 1, 3)

        # ② bidirectional: mask=None.  GQA (q heads > kv heads) is handled
        # natively by the Metal SDPA kernel, same as the AR block.
        if getattr(self.config, "bidirectional", True):
            attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        else:
            # causal path kept only for the §驗證(A) parity check — lets the
            # validator compare unidirectional vs bidirectional on one model.
            mask = mx.triu(mx.full((L, L), -mx.inf, dtype=q.dtype), k=1)
            attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        attn = attn.transpose(0, 2, 1, 3).reshape(B, L, D)
        x = residual + self.ls_attn(self.o_proj(attn))

        h = self.norm_ffn(x)
        ffn_out = self.ffn(h)
        out = x + self.ls_ffn(ffn_out)
        # state is None — dLLM carries no per-block cache across iterations.
        return out, None

    # ── prefix-cache (encoder/decoder split) — DiffusionGemma block-diffusion ──
    # The prompt is encoded ONCE (it never changes and, under prefix-LM, does
    # not attend to the canvas), then each denoising step forwards only the
    # G-token canvas attending to the cached prompt KV.  This avoids re-reading
    # the prompt T times and keeps the canvas scan to a single chunk (G≤64).

    def _qkv(self, x):
        B, L, _ = x.shape
        nx = self.norm_attn(x)
        q = self.q_proj(nx).reshape(B, L, self.num_heads, 64).transpose(0, 2, 1, 3)
        k = self.k_proj(nx).reshape(B, L, self.num_kv_heads, 64).transpose(0, 2, 1, 3)
        v = self.v_proj(nx).reshape(B, L, self.num_kv_heads, 64).transpose(0, 2, 1, 3)
        return q, k, v

    def _post(self, attn, residual):
        B, H, L, _ = attn.shape
        D = self.config.d_model
        attn = attn.transpose(0, 2, 1, 3).reshape(B, L, D)
        x = residual + self.ls_attn(self.o_proj(attn))
        h = self.norm_ffn(x)
        return x + self.ls_ffn(self.ffn(h))

    def encode(self, x):
        """Encode the prompt: bidirectional within the prompt; return its KV
        cache for the denoiser to attend to.  (prefix-LM: the prompt does NOT
        see the canvas, so this is computed once and reused every step.)"""
        q, k, v = self._qkv(x)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        return self._post(attn, x), {"k": k, "v": v}

    def denoise(self, x, kv):
        """Forward G canvas tokens attending to [prompt KV ⊕ canvas KV],
        bidirectional (no mask — every canvas position sees the whole prompt
        and the whole canvas)."""
        q, kc, vc = self._qkv(x)
        k = mx.concatenate([kv["k"], kc], axis=2)
        v = mx.concatenate([kv["v"], vc], axis=2)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        return self._post(attn, x)
