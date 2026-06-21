"""High-performance dLLM forward — one compiled graph reused across all
unmasking iterations (batch = 1).

Mirrors ``mlx_model.static_decode.StaticDecoder``'s single-graph philosophy,
but for the dLLM diffusion forward instead of AR L=1 decode:

  * The sequence length (P + G) is constant for every one of the ``T``
    iterations of a single ``generate`` call, so ``mx.compile`` traces the
    whole ``embed → 24×Mamba3Block → 6×bidirectional-TF → head`` graph ONCE
    and every subsequent step reuses it (≈50 → 1 dispatch per forward).
  * The Mamba blocks run their chunk-scan prefill path, which already executes
    on the Metal scan kernel (``mlx_model/scan_metal.py``); the TuckerMoE
    G-cache is reused from ``model.precompute()``.
  * ``eval_boundary=False`` is required: ``mx.eval`` cannot run inside a
    compiled graph.

The per-step selection/scheduling stay eager (argsort/entropy over G≈64 is
negligible and the commit count is data-dependent).
"""

from __future__ import annotations

import time

import mlx.core as mx

from .config import MASK_ID, DiffusionGenConfig
from .generate import (DLLMResult, diffusion_generate,
                       diffusion_generate_cached, iterative_unmask)


class StaticDLLM:
    def __init__(self, model):
        self._model = model
        self._use_sc = getattr(model, "self_conditioning", None) is not None
        self._compiled = None

    def _build_forward(self):
        """Compiled forward with a uniform ``forward(x, sc)`` signature.

        When self-conditioning is on, ``sc`` (the previous step's full-seq
        logits) is a compiled input (fixed shape across steps); when off it is
        ignored so the trace stays single-input.
        """
        model = self._model
        if self._use_sc:
            def _fwd(x, sc):
                return model(x, self_conditioning_logits=sc, eval_boundary=False)
            return mx.compile(_fwd)
        def _fwd(x, sc=None):
            return model(x, eval_boundary=False)
        return mx.compile(_fwd)

    def _warm(self, prompt_ids, gen_len) -> float:
        if self._compiled is None:
            self._compiled = self._build_forward()
        P, G, V = len(prompt_ids), int(gen_len), self._model.config.vocab_size
        t_c = time.perf_counter()
        x0 = mx.zeros((1, P + G), dtype=mx.int32)
        if self._use_sc:
            warm = self._compiled(x0, mx.zeros((1, P + G, V), dtype=mx.float32))
        else:
            warm = self._compiled(x0, None)
        mx.eval(warm)
        return time.perf_counter() - t_c

    # ── cosine baseline (DLLM_MLX_PORT.md §④) ─────────────────────────────────
    def generate(self, prompt_ids, gen_len, *, steps=16, temperature=0.0,
                 seed=0, mask_id=MASK_ID, on_commit=None) -> DLLMResult:
        compile_s = self._warm(prompt_ids, gen_len)
        compiled = self._compiled
        fwd = (lambda x: compiled(x, None))
        res = iterative_unmask(self._model, prompt_ids, gen_len, steps=steps,
                               temperature=temperature, seed=seed,
                               mask_id=mask_id, forward=fwd, on_commit=on_commit)
        return res._replace(compile_s=compile_s)

    # ── optimized path (entropy sampler / temp schedule / adaptive stop / SC) ──
    def diffusion(self, prompt_ids, dgcfg: DiffusionGenConfig, *,
                  mask_id=MASK_ID, on_step=None) -> DLLMResult:
        compile_s = self._warm(prompt_ids, dgcfg.gen_len)
        compiled = self._compiled
        fwd = (lambda x, sc: compiled(x, sc)) if self._use_sc else (lambda x, sc: compiled(x, None))
        res = diffusion_generate(self._model, prompt_ids, dgcfg,
                                 forward=fwd, mask_id=mask_id, on_step=on_step)
        return res._replace(compile_s=compile_s)

    # ── prefix-cache path (encoder/decoder split — fastest) ───────────────────
    def diffusion_cached(self, prompt_ids, dgcfg: DiffusionGenConfig, *,
                         mask_id=MASK_ID, on_step=None) -> DLLMResult:
        """Encode the prompt once, then run a compiled canvas-only denoiser for
        every diffusion step.  ~chunk-cliff + prompt-reread savings vs the full
        forward (see dllm_infer.py --prefix-cache)."""
        model = self._model
        t_enc = time.perf_counter()
        cache, _ = model.encode_prefix(mx.array(prompt_ids, dtype=mx.int32)[None])
        flat = [v for st in cache if isinstance(st, dict) for v in st.values() if v is not None]
        mx.eval(*flat)

        if self._use_sc:
            def _den(canvas, sc):
                return model.denoise(canvas, cache, self_conditioning_logits=sc)[0]
        else:
            def _den(canvas, sc=None):
                return model.denoise(canvas, cache)[0]
        compiled = mx.compile(_den)

        G, V = int(dgcfg.gen_len), model.config.vocab_size
        c0 = mx.zeros((1, G), dtype=mx.int32)
        warm = compiled(c0, mx.zeros((1, G, V), dtype=mx.float32)) if self._use_sc else compiled(c0, None)
        mx.eval(warm)
        compile_s = time.perf_counter() - t_enc

        res = diffusion_generate_cached(model, prompt_ids, dgcfg,
                                        denoise=compiled, mask_id=mask_id, on_step=on_step)
        return res._replace(compile_s=compile_s)
