import math

import mlx.core as mx
import mlx.nn as nn

from .ops import RMSNorm, LayerScale, silu_gating
from .tucker_moe import TuckerMoE


class MixtralMoEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_ff = int(math.ceil(config.ffn_expand * config.d_model / 256) * 256)
        kw = dict(num_experts=config.kmoe_num_experts, top_k=config.kmoe_top_k,
                  r1=config.kmoe_r1, r2=config.kmoe_r2, r3=config.kmoe_r3,
                  eps=config.rms_norm_eps)
        self.gate_proj = TuckerMoE(config.d_model, d_ff, **kw)
        self.up_proj = TuckerMoE(config.d_model, d_ff, **kw)
        self.down_proj = TuckerMoE(d_ff, config.d_model, **kw)

    def __call__(self, x):
        gate = self.gate_proj(x)
        feat = self.up_proj(x)
        return self.down_proj(silu_gating(gate, feat))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = 64
        self.num_heads = config.d_model // 64
        self.num_kv_heads = config.num_kv_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.d_model, self.num_heads * 64, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.num_kv_heads * 64, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.num_kv_heads * 64, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.norm_attn = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm_ffn = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn = MixtralMoEFeedForward(config)
        self.ls_attn = LayerScale(config.d_model, init_value=config.layer_scale_init)
        self.ls_ffn = LayerScale(config.d_model, init_value=config.layer_scale_init)

        # Compiled decode sub-functions — set by precompute().
        self._compiled_pre:  object | None = None
        self._compiled_post: object | None = None

    def precompute(self) -> None:
        """Compile the decode-path sub-functions that have fixed output shapes.

        For L=1 decode the KV cache grows each step, preventing full-block
        compilation.  We split into two compiled pieces around the SDPA call:
          _decode_pre : norm_attn + q/k/v projections  (fixed shapes)
          _decode_post: o_proj + residual + norm_ffn + full inlined FFN (fixed shapes)

        The non-compilable middle section (KV concat + SDPA) is only 2 Metal
        kernel dispatches so its overhead is negligible.
        """
        self._compiled_pre  = mx.compile(self._decode_pre)
        self._compiled_post = mx.compile(self._decode_post)
        D   = self.config.d_model
        H   = self.num_heads
        kvH = self.num_kv_heads
        _dx   = mx.zeros((1, 1, D),            dtype=mx.bfloat16)
        _dattn = mx.zeros((1, H, 1, 64),        dtype=mx.bfloat16)
        q_w, k_w, v_w = self._compiled_pre(_dx)
        mx.eval(q_w, k_w, v_w)
        _out = self._compiled_post(_dattn, _dx)
        mx.eval(_out)

    def _decode_pre(self, x):
        """Compile: norm_attn + q/k/v projections (fixed shapes for decode)."""
        B, L, D = x.shape
        nx = self.norm_attn(x)
        q = self.q_proj(nx).reshape(B, L, self.num_heads, 64).transpose(0, 2, 1, 3)
        k = self.k_proj(nx).reshape(B, L, self.num_kv_heads, 64).transpose(0, 2, 1, 3)
        v = self.v_proj(nx).reshape(B, L, self.num_kv_heads, 64).transpose(0, 2, 1, 3)
        return q, k, v

    def _decode_post(self, attn, x):
        """Compile: o_proj + residual + norm_ffn + inlined FFN (fixed shapes).

        attn: (B, H, 1, 64) output of SDPA — fixed shape for decode.
        x:    (B, 1, D)     residual before attention.
        """
        B, L, D = x.shape
        attn_out = attn.transpose(0, 2, 1, 3).reshape(B, L, D)
        x = x + self.ls_attn(self.o_proj(attn_out))
        h = self.norm_ffn(x)
        gate = self.ffn.gate_proj._forward(h)
        feat = self.ffn.up_proj._forward(h)
        ffn_out = self.ffn.down_proj._forward(silu_gating(gate, feat))
        return x + self.ls_ffn(ffn_out)

    def __call__(self, x, state=None):
        B, L, D = x.shape

        # Fast decode path: compiled pre/post around the non-compilable KV concat.
        if (L == 1 and state is not None
                and self._compiled_pre is not None
                and state.get("k") is not None):
            q, k_new, v_new = self._compiled_pre(x)
            past_k, past_v = state["k"], state["v"]
            k_full = mx.concatenate([past_k, k_new], axis=2)
            v_full = mx.concatenate([past_v, v_new], axis=2)
            attn = mx.fast.scaled_dot_product_attention(q, k_full, v_full, scale=self.scale)
            out = self._compiled_post(attn, x)
            return out, {"k": k_full, "v": v_full}

        # Prefill / first-token path.
        residual = x
        nx = self.norm_attn(x)
        q = self.q_proj(nx).reshape(B, L, self.num_heads, 64).transpose(0, 2, 1, 3)
        k = self.k_proj(nx).reshape(B, L, self.num_kv_heads, 64).transpose(0, 2, 1, 3)
        v = self.v_proj(nx).reshape(B, L, self.num_kv_heads, 64).transpose(0, 2, 1, 3)

        if state is not None and state.get("k") is not None:
            past_k = state["k"]          # (B, kv_h, S_past, 64)
            past_v = state["v"]
            k = mx.concatenate([past_k, k], axis=2)
            v = mx.concatenate([past_v, v], axis=2)
        new_k_cache = k
        new_v_cache = v

        # mx.fast.scaled_dot_product_attention supports native GQA:
        # q (B, H, L, D) and k/v (B, kv_H, S, D) where H % kv_H == 0.
        # No need to expand k/v via mx.repeat — the Metal kernel handles group
        # broadcasting internally, saving a (B, H, S, D) allocation per step.
        S = k.shape[2]
        # causal mask: query positions are [S-L .. S-1], keys are [0 .. S-1]
        if L == 1:
            attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            mask = mx.full((L, S), -mx.inf, dtype=q.dtype)
            # row i (query) attends to keys j where j <= (S - L + i)
            i_idx = mx.arange(L)[:, None]
            j_idx = mx.arange(S)[None, :]
            allow = j_idx <= (S - L + i_idx)
            mask = mx.where(allow, mx.zeros_like(mask), mask)
            attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        attn = attn.transpose(0, 2, 1, 3).reshape(B, L, D)
        x = residual + self.ls_attn(self.o_proj(attn))

        residual = x
        h = self.norm_ffn(x)
        ffn_out = self.ffn(h)
        out = residual + self.ls_ffn(ffn_out)

        new_state = {"k": new_k_cache, "v": new_v_cache}
        return out, new_state
