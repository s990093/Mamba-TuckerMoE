# -*- coding: utf-8 -*-
import mlx.core as mx
import mlx.nn as nn

from .ops import RMSNorm, LayerScale, silu_gating
from .tucker_moe import TuckerMoE


class MixtralMoEFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        import math
        d_ff = int(math.ceil(config.ffn_expand * config.d_model / 256) * 256)
        kw = dict(
            num_experts=config.kmoe_num_experts,
            top_k=config.kmoe_top_k,
            r1=config.kmoe_r1,
            r2=config.kmoe_r2,
            r3=config.kmoe_r3,
        )
        self.gate_proj = TuckerMoE(config.d_model, d_ff, **kw)
        self.up_proj   = TuckerMoE(config.d_model, d_ff, **kw)
        self.down_proj = TuckerMoE(d_ff, config.d_model, **kw)

    def __call__(self, x, step=None):
        gate, lb_g, _ = self.gate_proj(x, step=step)
        feat, lb_u, _ = self.up_proj(x, step=step)
        y,    lb_d, _ = self.down_proj(silu_gating(gate, feat), step=step)
        return y, lb_g + lb_u + lb_d, 0.0


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model      = config.d_model
        head_dim     = 64
        num_heads    = d_model // head_dim       # 12
        num_kv_heads = config.num_kv_heads       # 4

        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self._scale       = head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, num_heads    * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)

        self.norm_attn = RMSNorm(d_model, eps=config.rms_norm_eps)
        self.norm_ffn  = RMSNorm(d_model, eps=config.rms_norm_eps)
        self.ffn       = MixtralMoEFFN(config)

        self.ls_attn = LayerScale(d_model, init_value=config.layer_scale_init)
        self.ls_ffn  = LayerScale(d_model, init_value=config.layer_scale_init)

    def __call__(self, x, step=None, kv_cache=None):
        """
        kv_cache: dict {'k': (B, kv_heads, T_past, D), 'v': same} | None
        Returns : (out, lb, z_loss, new_kv_cache)

        All ops remain lazy (no mx.eval inside).
        """
        B, L, D = x.shape
        residual = x
        nx = self.norm_attn(x)

        # ── Projections ────────────────────────────────────────────────────────
        # Layout after transpose: (B, heads, T, D)
        q = self.q_proj(nx).reshape(B, L, self.num_heads,    self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(nx).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(nx).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # ── KV cache append ────────────────────────────────────────────────────
        if kv_cache is not None and kv_cache.get("k") is not None:
            k = mx.concatenate([kv_cache["k"], k], axis=2)
            v = mx.concatenate([kv_cache["v"], v], axis=2)

        new_kv_cache = {"k": k, "v": v}

        # ── Fast SDPA (GQA-aware, fused softmax) ───────────────────────────────
        # mx.fast.scaled_dot_product_attention natively handles GQA when
        # k/v have fewer heads than q (num_heads // num_kv_heads groups).
        T_q = q.shape[2]
        T_k = k.shape[2]

        if T_q == 1:
            # Decode: single query token attends to all cached keys — no mask.
            mask = None
        elif T_k == T_q:
            # Pure prefill (no prior cache): standard causal.
            mask = "causal"
        else:
            # Prefill on top of an existing KV prefix: build explicit mask.
            # Q positions [0..T_q-1] can attend to full prefix + causal within Q.
            offset = T_k - T_q
            causal_part = mx.tril(mx.ones((T_q, T_q), dtype=mx.bool_))
            prefix_part = mx.ones((T_q, offset), dtype=mx.bool_)
            mask = mx.concatenate([prefix_part, causal_part], axis=1)   # (T_q, T_k)
            # Convert bool → additive float mask expected by fast sdpa
            mask = mx.where(mask, mx.zeros(mask.shape, dtype=q.dtype),
                            mx.full(mask.shape, float("-inf"), dtype=q.dtype))
            mask = mask[None, None]   # (1, 1, T_q, T_k)

        attn_out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self._scale, mask=mask
        )  # (B, num_heads, T_q, head_dim)

        # ── Merge heads & output projection ───────────────────────────────────
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, D)
        x = residual + self.ls_attn(self.o_proj(attn_out))

        # ── FFN ───────────────────────────────────────────────────────────────
        residual = x
        ffn_out, lb, z = self.ffn(self.norm_ffn(x), step=step)
        out = residual + self.ls_ffn(ffn_out)

        return out, lb, z, new_kv_cache
