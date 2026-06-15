"""MLX port of the Flash-Attention draft transformer (speculative decoding).

This is an MLX re-implementation of the PyTorch ``DraftTransformer`` defined in
``sft_cot_bundle/scripts/draft/transformer_draft.py`` and documented in
``sft_cot_bundle/scripts/draft/ARCHITECTURE.md``.

Architecture (DraftConfig defaults)::

    d_model=256, n_layers=6, n_heads=8, n_kv_heads=2 (GQA 4:1),
    head_dim=32, SwiGLU ffn_mult=3, RMSNorm pre-norm, RoPE with offset,
    tied embed+head, vocab=32007, ~12.7M params.

The model exposes a KV-cache forward identical in spirit to the PyTorch
version so it can serve as the draft model in classic (target-verified)
speculative decoding against ``Mamba3LanguageModel``.

Usage::

    from mamba3_mlx.speculative.draft_transformer import load_draft_model, DraftGuesser
    model = load_draft_model("checkpoints/draft_model/draft_tf_s10000.pt")
    g = DraftGuesser(model)
    g.prefill(prompt_ids)            # KV covers up-to-but-excluding next token
    guesses = g.draft(prev_token, K - 1)
    g.commit([prev_token, *accepted[:-1]])

Nothing in this module mutates existing repo files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

# KV cache: list (len n_layers) of (k, v); each (B, n_kv_heads, T, head_dim).
KVCache = list[tuple[mx.array, mx.array]]


@dataclass
class DraftConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2       # GQA: 4 query heads per KV head
    ffn_mult: int = 3         # SwiGLU hidden = d_model * ffn_mult
    max_seq: int = 4096       # RoPE precompute length (decode buffer)
    vocab_size: int = 32007
    rope_base: int = 10000
    rms_eps: float = 1e-6     # PyTorch training default (see ARCHITECTURE.md §4.1)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RoPE(nn.Module):
    """Rotary embedding with KV-cache position offset (non-persistent tables)."""

    def __init__(self, head_dim: int, max_seq: int, base: int):
        super().__init__()
        inv = 1.0 / (base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
        t = mx.arange(max_seq).astype(mx.float32)
        emb = mx.concatenate([mx.outer(t, inv)] * 2, axis=-1)   # (max_seq, head_dim)
        # Stored as plain (non-parameter) attributes so load_weights ignores them.
        self._cos = mx.cos(emb)
        self._sin = mx.sin(emb)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        # x: (B, H, T, head_dim)
        T = x.shape[2]
        h = x.shape[-1] // 2
        cos = self._cos[offset:offset + T][None, None].astype(x.dtype)
        sin = self._sin[offset:offset + T][None, None].astype(x.dtype)
        x_rot = mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)
        return x * cos + x_rot * sin


class GQAttention(nn.Module):
    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        kv_dim = cfg.n_kv_heads * cfg.head_dim

        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RoPE(cfg.head_dim, cfg.max_seq, cfg.rope_base)
        self.scale = cfg.head_dim ** -0.5

    def __call__(self, x, past_kv=None, offset=0):
        B, T, _ = x.shape
        q = self.q(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if past_kv is not None:
            k = mx.concatenate([past_kv[0], k], axis=2)
            v = mx.concatenate([past_kv[1], v], axis=2)

        is_causal = (past_kv is None) and (T > 1)
        mask = None
        if is_causal:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(k.shape[2]).astype(x.dtype)

        # mx.fast.scaled_dot_product_attention broadcasts KV across query groups.
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o(out), (k, v)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: int):
        super().__init__()
        h = d_model * mult
        self.gate = nn.Linear(d_model, h, bias=False)
        self.up = nn.Linear(d_model, h, bias=False)
        self.down = nn.Linear(h, d_model, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.attn = GQAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_mult)

    def __call__(self, x, past_kv=None, offset=0):
        attn_out, new_kv = self.attn(self.norm1(x), past_kv=past_kv, offset=offset)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


class DraftTransformer(nn.Module):
    def __init__(self, cfg: Optional[DraftConfig] = None):
        super().__init__()
        cfg = cfg or DraftConfig()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        # head shares embed.weight (tied); we matmul against embed directly so
        # no separate Linear is needed and load_weights stays clean.

    def __call__(self, input_ids, past_key_values: Optional[KVCache] = None):
        """Returns (logits (B,T,V) float32, new KVCache)."""
        offset = past_key_values[0][0].shape[2] if past_key_values else 0
        x = self.embed(input_ids)
        new_kvs: KVCache = []
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values else None
            x, kv = layer(x, past_kv=past, offset=offset)
            new_kvs.append(kv)
        x = self.norm(x)
        logits = self.embed.as_linear(x).astype(mx.float32)   # tied head
        return logits, new_kvs


# ── Checkpoint loading ──────────────────────────────────────────────────────

def _torch_state_to_mlx(sd, dtype: mx.Dtype) -> dict:
    """Convert a PyTorch state_dict (already on CPU) to flat MLX weight dict.

    Drops ``head.weight`` (tied to ``embed.weight``). Keys map directly:
    ``layers.{i}.attn.q.weight`` etc. already match this module's attribute
    tree, with the one rename that the tied head has no separate parameter.
    """
    import numpy as np
    weights: dict[str, mx.array] = {}
    for k, v in sd.items():
        if k == "head.weight":
            continue   # tied to embed.weight
        arr = v.to(getattr(__import__("torch"), "float32")).numpy().astype(np.float32)
        weights[k] = mx.array(arr).astype(dtype)
    return weights


def load_draft_model(path: str | Path, *, dtype: mx.Dtype = mx.bfloat16,
                     write_npz_sidecar: bool = True) -> DraftTransformer:
    """Load the draft model from a ``.pt`` (torch) or ``.npz`` (fast) file.

    On first ``.pt`` load, a ``.npz`` sidecar is written next to it (plus a
    small ``.cfg.json``) so subsequent loads need no torch.
    """
    path = Path(path)
    if path.suffix == ".npz":
        return _load_from_npz(path, dtype)

    import json
    import torch
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg_d = ckpt.get("config", {})
    cfg = DraftConfig(
        d_model=cfg_d.get("d_model", 256),
        n_layers=cfg_d.get("n_layers", 6),
        n_heads=cfg_d.get("n_heads", 8),
        n_kv_heads=cfg_d.get("n_kv_heads", 2),
        ffn_mult=cfg_d.get("ffn_mult", 3),
        vocab_size=cfg_d.get("vocab_size", 32007),
    )
    weights = _torch_state_to_mlx(ckpt["model"], dtype)

    model = DraftTransformer(cfg)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    if write_npz_sidecar:
        npz = path.with_suffix(".mlx.npz")
        mx.savez(str(npz), **weights)
        npz.with_suffix(".cfg.json").write_text(json.dumps({
            "d_model": cfg.d_model, "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
            "ffn_mult": cfg.ffn_mult, "vocab_size": cfg.vocab_size,
        }))
    return model


def _load_from_npz(path: Path, dtype: mx.Dtype) -> DraftTransformer:
    import json
    cfg_path = path.with_suffix(".cfg.json")
    cfg = DraftConfig()
    if cfg_path.exists():
        d = json.loads(cfg_path.read_text())
        cfg = DraftConfig(**{k: d[k] for k in (
            "d_model", "n_layers", "n_heads", "n_kv_heads", "ffn_mult",
            "vocab_size") if k in d})
    raw = mx.load(str(path))
    weights = {k: v.astype(dtype) for k, v in raw.items()}
    model = DraftTransformer(cfg)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model


# ── Draft guesser with rollback KV ──────────────────────────────────────────

class DraftGuesser:
    """Maintains the draft model's KV cache for speculative rounds.

    Invariant: between rounds the KV cache covers exactly the confirmed
    sequence *up to but excluding* the next ``prev_token`` to be fed.  Each
    round we draft ``n`` tokens (growing the KV), then ``commit`` rolls the KV
    back to the pre-draft checkpoint and re-feeds the actually-accepted tokens
    in a single batched forward — making partial-accept rollback exact and
    cheap.
    """

    def __init__(self, model: DraftTransformer):
        self.model = model
        self.kv: Optional[KVCache] = None
        self._ckpt_len = 0

    def reset(self) -> None:
        self.kv = None
        self._ckpt_len = 0

    def _kv_len(self) -> int:
        return 0 if self.kv is None else int(self.kv[0][0].shape[2])

    def prefill(self, ids: list[int]) -> int:
        """Prefill the draft on ``ids``; returns greedy next-token argmax."""
        arr = mx.array([ids], dtype=mx.int32)
        logits, self.kv = self.model(arr, past_key_values=None)
        mx.eval(logits, *[t for kv in self.kv for t in kv])
        return int(mx.argmax(logits[0, -1]).item())

    def draft(self, prev_token: int, n: int) -> list[int]:
        """Autoregressively produce ``n`` greedy guesses starting at prev_token.

        Feeds ``prev_token`` then each generated token; KV grows by ``n`` rows
        (prev_token + first n-1 guesses).  Records a checkpoint so ``commit``
        can roll back.
        """
        self._ckpt_len = self._kv_len()
        guesses: list[int] = []
        tok = int(prev_token)
        for _ in range(n):
            arr = mx.array([[tok]], dtype=mx.int32)
            logits, self.kv = self.model(arr, past_key_values=self.kv)
            tok = int(mx.argmax(logits[0, -1]).item())
            guesses.append(tok)
        return guesses

    def commit(self, real_tokens: list[int]) -> None:
        """Roll KV back to the pre-draft checkpoint and feed the confirmed
        tokens (``[prev_token, *accepted[:-1]]``) so the cache is correct for
        the next round.
        """
        L = self._ckpt_len
        self.kv = [(k[:, :, :L, :], v[:, :, :L, :]) for (k, v) in self.kv]
        if real_tokens:
            arr = mx.array([real_tokens], dtype=mx.int32)
            _, self.kv = self.model(arr, past_key_values=self.kv)
        mx.eval(*[t for kv in self.kv for t in kv])


# ── Compiled static-KV draft guesser (low-overhead path) ─────────────────────

class CompiledDraftGuesser:
    """Fast draft guesser: whole single-step forward compiled over a *static*
    KV buffer, with free partial-accept rollback.

    Techniques layered on top of :class:`DraftGuesser`:

    * **Whole-step ``mx.compile``** — the entire 6-layer step is one compiled
      program (vs. eager per-op dispatch), the dominant lever on Apple Silicon
      where decode is GPU-kernel-launch bound.
    * **Static KV via ``mx.slice_update``** — a fixed ``(1, n_kv, S, hd)``
      buffer written at ``write_pos``; no per-step ``concatenate`` realloc and
      the graph shape never changes so the compile cache always hits.
    * **Free commit** — accepted draft tokens are already the correct tokens at
      their KV slots, so rollback is just ``write_pos = ckpt + m`` (no re-feed
      forward — saves a whole batched draft pass per round).
    * **RoPE rows passed as inputs** — ``cos``/``sin`` for the current absolute
      position are sliced eagerly and fed in, so the compiled graph needs no
      dynamic table indexing.
    * **Deferred eval** — the ``n`` chained steps build one lazy graph and sync
      once per round instead of ``n`` times.
    """

    def __init__(self, model: DraftTransformer, *, max_seq: int = 1024):
        self.model = model
        self.cfg = model.cfg
        self.hd = model.cfg.head_dim
        self.S = max_seq
        rope = model.layers[0].attn.rope
        self._cos = rope._cos          # (max_seq_rope, hd)
        self._sin = rope._sin
        self.kv: Optional[list[mx.array]] = None   # flat [k0,v0,k1,v1,...]
        self.wp = 0
        self._ckpt = 0
        self._step = mx.compile(self._step_impl)

    def reset(self) -> None:
        self.kv = None
        self.wp = 0
        self._ckpt = 0

    def _step_impl(self, token, cos_row, sin_row, write_pos, kvs):
        m = self.model
        nh, nkv, hd = self.cfg.n_heads, self.cfg.n_kv_heads, self.hd
        h = hd // 2
        x = m.embed(token)                                   # (1,1,d)
        S = kvs[0].shape[2]
        allow = mx.arange(S)[None, :] <= write_pos[0]
        mask = mx.where(allow, mx.array(0.0, dtype=x.dtype),
                        mx.array(-mx.inf, dtype=x.dtype)).reshape(1, 1, 1, S)
        new = list(kvs)
        for li, layer in enumerate(m.layers):
            nx = layer.norm1(x)
            q = layer.attn.q(nx).reshape(1, 1, nh, hd).transpose(0, 2, 1, 3)
            k = layer.attn.k(nx).reshape(1, 1, nkv, hd).transpose(0, 2, 1, 3)
            v = layer.attn.v(nx).reshape(1, 1, nkv, hd).transpose(0, 2, 1, 3)
            q = q * cos_row + mx.concatenate([-q[..., h:], q[..., :h]], axis=-1) * sin_row
            k = k * cos_row + mx.concatenate([-k[..., h:], k[..., :h]], axis=-1) * sin_row
            kc = mx.slice_update(new[2 * li], k, write_pos, axes=(2,))
            vc = mx.slice_update(new[2 * li + 1], v, write_pos, axes=(2,))
            attn = mx.fast.scaled_dot_product_attention(
                q, kc, vc, scale=layer.attn.scale, mask=mask)
            attn = attn.transpose(0, 2, 1, 3).reshape(1, 1, -1)
            x = x + layer.attn.o(attn)
            x = x + layer.ffn(layer.norm2(x))
            new[2 * li], new[2 * li + 1] = kc, vc
        logits = m.embed.as_linear(m.norm(x))                # (1,1,V)
        return mx.argmax(logits[0, -1]).astype(mx.int32), new

    def _rope_row(self, pos: int):
        cos = self._cos[pos:pos + 1].reshape(1, 1, 1, self.hd)
        sin = self._sin[pos:pos + 1].reshape(1, 1, 1, self.hd)
        return cos.astype(self.model.embed.weight.dtype), sin.astype(self.model.embed.weight.dtype)

    def prefill(self, ids: list[int]) -> int:
        """Eager full prefill, then scatter KV into the static buffer."""
        arr = mx.array([ids], dtype=mx.int32)
        logits, kv = self.model(arr, past_key_values=None)
        L = len(ids)
        pad = self.S - L
        flat: list[mx.array] = []
        for (k, v) in kv:
            kb = mx.pad(k, ((0, 0), (0, 0), (0, pad), (0, 0)))
            vb = mx.pad(v, ((0, 0), (0, 0), (0, pad), (0, 0)))
            flat += [kb, vb]
        self.kv = flat
        self.wp = L
        mx.eval(logits, *self.kv)
        return int(mx.argmax(logits[0, -1]).item())

    def draft(self, prev_token: int, n: int) -> list[int]:
        self._ckpt = self.wp
        tok_arr = mx.array([[prev_token]], dtype=mx.int32)
        outs: list[mx.array] = []
        for _ in range(n):
            cos_row, sin_row = self._rope_row(self.wp)
            wp_arr = mx.array([self.wp], dtype=mx.int32)
            out_tok, self.kv = self._step(tok_arr, cos_row, sin_row, wp_arr, self.kv)
            outs.append(out_tok)
            tok_arr = out_tok.reshape(1, 1)
            self.wp += 1
        mx.eval(*outs, *self.kv)             # one sync for the whole chain
        return [int(t.item()) for t in outs]

    def commit(self, m_accepted: int) -> None:
        """Free rollback: keep the first ``m`` written KV slots (already the
        correct tokens), drop the rejected tail by rewinding ``write_pos``."""
        self.wp = self._ckpt + m_accepted
