"""
Pure Flash-Attention Transformer draft model for speculative decoding.

Architecture:
  d_model=256, n_layers=6, n_heads=8, n_kv_heads=2 (GQA 4:1)
  SwiGLU FFN, RMSNorm pre-norm, RoPE with position offset, tied embed+head
  ~12.7M params, all active.

Attention:
  F.scaled_dot_product_attention(enable_gqa=True)  →  FA2 on Ampere (RTX 3090)

KV Cache:
  forward(input_ids, past_key_values=None, use_cache=False)
  - prefill  (past_key_values=None, T>1): is_causal=True, returns (logits, kvs)
  - decode   (past_key_values=kvs, T=1):  is_causal=False, O(T) per step → O(1) per token
  This is required for speculative decoding; without it each step is O(T²).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

KVCache = list[tuple[torch.Tensor, torch.Tensor]]   # [(k, v), ...] per layer


@dataclass
class DraftConfig:
    d_model:    int = 256
    n_layers:   int = 6
    n_heads:    int = 8
    n_kv_heads: int = 2      # GQA: 4 query heads per KV head
    ffn_mult:   int = 3      # SwiGLU hidden = d_model * ffn_mult
    max_seq:    int = 2048
    vocab_size: int = 32007
    rope_base:  int = 10000


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq: int, base: int):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t   = torch.arange(max_seq).float()
        emb = torch.cat([torch.outer(t, inv)] * 2, dim=-1)   # (max_seq, head_dim)
        # Pre-computed; persistent=False so not saved in state_dict
        self.register_buffer("cos", emb.cos(), persistent=False)   # (max_seq, hd)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: (B, H, T, head_dim)
        T   = x.shape[2]
        h   = x.shape[-1] // 2
        cos = self.cos[offset : offset + T][None, None]   # (1,1,T,hd)
        sin = self.sin[offset : offset + T][None, None]
        return x * cos + torch.cat([-x[..., h:], x[..., :h]], dim=-1) * sin


class GQAttention(nn.Module):
    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.nh  = cfg.n_heads
        self.nkv = cfg.n_kv_heads
        self.hd  = cfg.d_model // cfg.n_heads
        kv_dim   = self.nkv * self.hd

        self.q    = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k    = nn.Linear(cfg.d_model, kv_dim,      bias=False)
        self.v    = nn.Linear(cfg.d_model, kv_dim,      bias=False)
        self.o    = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RoPE(self.hd, cfg.max_seq, cfg.rope_base)

    def forward(
        self,
        x:       torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        offset:  int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh,  self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv = (k, v)

        # is_causal=True only during prefill (T>1, no past); False during decode (T=1)
        is_causal = (past_kv is None) and (T > 1)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, enable_gqa=True
        )
        return self.o(out.transpose(1, 2).contiguous().view(B, T, -1)), new_kv


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: int):
        super().__init__()
        h = d_model * mult
        self.gate = nn.Linear(d_model, h, bias=False)
        self.up   = nn.Linear(d_model, h, bias=False)
        self.down = nn.Linear(h, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = GQAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = SwiGLU(cfg.d_model, cfg.ffn_mult)

    def forward(
        self,
        x:       torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        offset:  int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.norm1(x), past_kv=past_kv, offset=offset)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


class DraftTransformer(nn.Module):
    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm   = nn.RMSNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight   # weight tying
        self._init()

    def _init(self) -> None:
        std = 0.02
        nn.init.normal_(self.embed.weight, std=std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight is not self.embed.weight:
                nn.init.normal_(m.weight, std=std / math.sqrt(2 * self.cfg.n_layers))

    def forward(
        self,
        input_ids:        torch.Tensor,
        labels:           Optional[torch.Tensor] = None,
        past_key_values:  Optional[KVCache]      = None,
        use_cache:        bool                   = False,
    ):
        # offset = number of tokens already in KV cache
        offset = past_key_values[0][0].shape[2] if past_key_values else 0

        x        = self.embed(input_ids)
        new_kvs: KVCache = []
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values else None
            x, kv = layer(x, past_kv=past, offset=offset)
            new_kvs.append(kv)

        logits = self.head(self.norm(x))

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                labels.view(-1),
            )
            return (loss, logits, new_kvs) if use_cache else (loss, logits)

        return (logits, new_kvs) if use_cache else logits
