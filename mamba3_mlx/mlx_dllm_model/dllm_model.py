"""dLLM model — AR Mamba-Tucker backbone with a bidirectional attention head.

Reuses, unchanged, from ``mamba3_mlx.mlx_model``:
  * ``Mamba3Block``      — unidirectional SSM (the prefill / chunk_scan path,
                           which already runs on the Metal scan kernel);
  * ``TuckerMoE``        — low-rank expert MoE + its G-cache precompute;
  * ``RMSNorm`` / ``scaled_tanh`` ops.

The only swapped component is the transformer block → ``DLLMTransformerBlock``
(bidirectional, change ②).  The embedding/head grow by one row for [MASK]
(change ①).

No checkpoint is loaded here (the dLLM weights are still training).  The
model self-initialises with random weights so the inference + high-performance
paths can be exercised and benchmarked end-to-end on shape-correct tensors.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ..mlx_model.mamba_block import Mamba3Block
from ..mlx_model.ops import RMSNorm, scaled_tanh
from .bidirectional_block import DLLMTransformerBlock
from .config import BASE_VOCAB, DLLMConfig
from .self_conditioning import DLLMSelfConditioning, soft_embeddings


class DLLMBackbone(nn.Module):
    """Same layer pattern as the AR ``TrueHybridMamba`` (mamba_ratio Mamba
    blocks then one transformer block, × num_layers) but with the
    bidirectional transformer block."""

    def __init__(self, config: DLLMConfig):
        super().__init__()
        self.config = config
        layers = []
        for _ in range(config.num_layers):
            for _ in range(config.mamba_ratio):
                layers.append(Mamba3Block(config))
            layers.append(DLLMTransformerBlock(config))
        self.layers = layers

    def __call__(self, x, *, eval_boundary: bool = True):
        # eval_boundary frees each Mamba block's chunk-scan intermediates
        # block-by-block (same trick as TrueHybridMamba) — but mx.eval cannot
        # run inside a compiled graph, so the static path passes False.
        for blk in self.layers:
            x, st = blk(x, state=None)
            if eval_boundary and st is not None:
                vals = [v for v in st.values() if v is not None]
                mx.eval(x, *vals)
        return x


class DLLMModel(nn.Module):
    def __init__(self, config: DLLMConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.backbone = DLLMBackbone(config)
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.inv_sqrt_d = 1.0 / math.sqrt(config.d_model)

        # Optional cross-step self-conditioning (DiffusionGemma-style). Adds
        # params → off unless trained; see self_conditioning.py.
        self.self_conditioning = None
        if getattr(config, "self_conditioning", False):
            d_ff = int(math.ceil(config.ffn_expand * config.d_model / 256) * 256)
            self.self_conditioning = DLLMSelfConditioning(
                config.d_model, d_ff, eps=config.rms_norm_eps)

    # ── weight setup (no checkpoint — model is still training) ────────────────

    def tie_weights(self) -> None:
        """① head/embed share one array (same as the AR loader)."""
        self.head.weight = self.embed.weight

    def init_mask_embedding(self) -> None:
        """① initialise the [MASK] row = mean(base rows) + 0.02·N(0,1).

        No-op-safe: when real dLLM weights are later loaded the embedding is
        already (32008, D) and this is skipped by the caller.
        """
        emb = self.embed.weight
        if emb.shape[0] <= BASE_VOCAB:
            return
        D = emb.shape[1]
        base_mean = emb[:BASE_VOCAB].mean(axis=0)
        emb[BASE_VOCAB] = base_mean + 0.02 * mx.random.normal((D,)).astype(emb.dtype)
        self.tie_weights()

    def init_experts_random(self, scale: float = 0.02) -> None:
        """Fill the zero-initialised TuckerMoE factors with small randoms.

        TuckerMoE constructs U_in/U_out/U_expert/core as zeros (they are meant
        to be loaded).  With no checkpoint that makes every MoE emit only its
        bias, so the iterative-unmasking demo never moves a token.  Seeding
        them with small randoms makes the untrained model produce varied
        (still meaningless) output so the full pipeline is exercised.
        """
        from ..mlx_model.tucker_moe import TuckerMoE

        def _seed(moe: TuckerMoE):
            moe.U_in = scale * mx.random.normal(moe.U_in.shape).astype(moe.U_in.dtype)
            moe.U_out = scale * mx.random.normal(moe.U_out.shape).astype(moe.U_out.dtype)
            moe.U_expert = scale * mx.random.normal(moe.U_expert.shape).astype(moe.U_expert.dtype)
            moe.core = scale * mx.random.normal(moe.core.shape).astype(moe.core.dtype)

        for layer in self.backbone.layers:
            if isinstance(layer, Mamba3Block):
                _seed(layer.x_up_proj)
                _seed(layer.out_proj)
            elif isinstance(layer, DLLMTransformerBlock):
                _seed(layer.ffn.gate_proj)
                _seed(layer.ffn.up_proj)
                _seed(layer.ffn.down_proj)

    def precompute(self) -> None:
        """Cache the TuckerMoE G tensors (and Mamba D_expand) for fast forward.

        Only the bits the dLLM full-sequence forward actually uses — it never
        runs the AR L=1 decode path, so the per-block decode-graph compilation
        from the AR loader is skipped.
        """
        for layer in self.backbone.layers:
            if isinstance(layer, Mamba3Block):
                layer.x_up_proj.precompute_G_experts()
                layer.out_proj.precompute_G_experts()
                layer._D_expand = mx.repeat(layer.D, layer.P, axis=0)
                mx.eval(layer._D_expand)
            elif isinstance(layer, DLLMTransformerBlock):
                layer.ffn.gate_proj.precompute_G_experts()
                layer.ffn.up_proj.precompute_G_experts()
                layer.ffn.down_proj.precompute_G_experts()

    # ── forward ───────────────────────────────────────────────────────────────

    # ── prefix-cache (encoder/decoder split) — high-performance path ──────────

    def encode_prefix(self, prompt_ids):
        """Encode the prompt ONCE into a per-layer cache (Mamba state + TF KV).

        prefix-LM semantics: the prompt is processed without the canvas, so the
        cache is reused unchanged across every denoising step.  Mamba is already
        unidirectional, so its prompt state is exact; the TF blocks attend only
        within the prompt here.  Returns ``(cache, prompt_x)``.
        """
        x = self.embed(prompt_ids)
        cache: list = []
        for blk in self.backbone.layers:
            if isinstance(blk, Mamba3Block):
                x, st = blk(x, state=None)
                cache.append(st)
            else:                                   # DLLMTransformerBlock
                x, kv = blk.encode(x)
                cache.append(kv)
        return cache, x

    def denoise(self, canvas_ids, cache, *, self_conditioning_logits=None):
        """Forward only the G-token canvas using a prompt ``cache`` → logits
        (1, G, V).  Each Mamba block continues from the cached prompt state;
        each TF block attends to the cached prompt KV ⊕ fresh canvas KV."""
        x = self.embed(canvas_ids)
        if self.self_conditioning is not None and self_conditioning_logits is not None:
            x = self.self_conditioning(x, soft_embeddings(self_conditioning_logits, self.embed.weight))
        for blk, st in zip(self.backbone.layers, cache):
            if isinstance(blk, Mamba3Block):
                x, _ = blk(x, state=st)             # continue from prompt state
            else:
                x = blk.denoise(x, st)              # attend cached prompt KV
        h = self.norm(x)
        logits = self.head(h * self.inv_sqrt_d).astype(mx.float32)
        return scaled_tanh(logits, 30.0)

    def __call__(self, input_ids, *, self_conditioning_logits=None,
                 eval_boundary: bool = True):
        """Bidirectional full-sequence forward → logits (B, L, V) float32.

        ``input_ids`` may contain ``mask_id`` (= 32007) at positions to be
        predicted.  No causal mask, no state: every call re-reads the whole
        sequence (the dLLM diffusion forward).

        ``self_conditioning_logits`` (B, L, V): the previous denoising step's
        logits, fused as a soft embedding when self-conditioning is enabled.
        """
        x = self.embed(input_ids)
        if self.self_conditioning is not None and self_conditioning_logits is not None:
            sc = soft_embeddings(self_conditioning_logits, self.embed.weight)
            x = self.self_conditioning(x, sc)
        x = self.backbone(x, eval_boundary=eval_boundary)
        h = self.norm(x)
        logits = self.head(h * self.inv_sqrt_d).astype(mx.float32)
        return scaled_tanh(logits, 30.0)


def build_random_dllm(config: DLLMConfig | None = None, *, seed: int = 0,
                      seed_experts: bool = True) -> DLLMModel:
    """Build a fully-initialised untrained DLLMModel (no checkpoint).

    Order matters: experts are seeded BEFORE ``precompute`` so the cached G
    tensors reflect the seeded factors.
    """
    mx.random.seed(seed)
    cfg = config or DLLMConfig()
    model = DLLMModel(cfg)
    if seed_experts:
        model.init_experts_random()
    model.init_mask_embedding()
    model.tie_weights()
    model.precompute()
    mx.eval(model.parameters())
    return model
