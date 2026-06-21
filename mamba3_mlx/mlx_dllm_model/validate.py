"""dLLM validation harness (DLLM_MLX_PORT.md §驗證).

Three layers, ported to run self-contained against the MLX model:

  (A) forward parity
        · bidirectional vs causal — proves change ② is actually wired in
          (outputs must differ; both must be finite);
        · eager vs compiled (StaticDLLM) — proves the high-performance path
          is numerically faithful (bf16 max-abs-diff ~1e-2).
  (B) fixed-ratio reconstruction — top-1 accuracy on randomly-masked response
        positions.  On an untrained model this is ~chance (the doc's baseline);
        it becomes meaningful once real dLLM weights are loaded.
  (C) iterative reconstruction — run §④ from all-[MASK] and compare to a gold
        sequence (token accuracy + exact-match).  Also ~chance untrained.

The CUDA reference numbers to diff against live in
``sft_cot_bundle/scripts/dllm_validate.py``; this harness produces the MLX-side
figures in the same shapes.
"""

from __future__ import annotations

import mlx.core as mx

from .config import BASE_VOCAB, MASK_ID
from .generate import iterative_unmask
from .static_dllm import StaticDLLM


# ── (A) forward parity ─────────────────────────────────────────────────────────

def check_bidirectional(model, *, seq_len: int = 48, seed: int = 0) -> dict:
    """Bidirectional vs causal on the same input — change ② sanity."""
    mx.random.seed(seed)
    x = mx.random.randint(0, BASE_VOCAB, (1, seq_len)).astype(mx.int32)

    prev = model.config.bidirectional
    model.config.bidirectional = True
    lg_bi = model(x)
    model.config.bidirectional = False
    lg_ca = model(x)
    model.config.bidirectional = prev
    mx.eval(lg_bi, lg_ca)

    max_diff = float(mx.max(mx.abs(lg_bi - lg_ca)).item())
    finite = bool(mx.all(mx.isfinite(lg_bi)).item() and mx.all(mx.isfinite(lg_ca)).item())
    return {
        "max_abs_diff_bi_vs_causal": max_diff,
        "outputs_differ": max_diff > 1e-3,
        "all_finite": finite,
    }


def check_static_parity(model, *, seq_len: int = 48, seed: int = 0,
                        tol: float = 2e-2) -> dict:
    """Eager forward vs the compiled StaticDLLM forward — high-perf fidelity."""
    mx.random.seed(seed)
    x = mx.random.randint(0, model.config.vocab_size, (1, seq_len)).astype(mx.int32)

    lg_eager = model(x, eval_boundary=True)
    sd = StaticDLLM(model)
    compiled = sd._build_forward()
    lg_static = compiled(x, None) if sd._use_sc else compiled(x)
    mx.eval(lg_eager, lg_static)

    max_diff = float(mx.max(mx.abs(lg_eager - lg_static)).item())
    return {"max_abs_diff_eager_vs_static": max_diff, "within_tol": max_diff <= tol, "tol": tol}


# ── (B) fixed-ratio reconstruction ─────────────────────────────────────────────

def fixed_ratio_reconstruction(
    model, *, seq_len: int = 128, resp_start: int | None = None,
    ratios=(0.1, 0.2, 0.3, 0.5, 0.7, 0.9), seed: int = 0,
    mask_id: int = MASK_ID,
) -> dict:
    """Top-1 accuracy on masked response positions at fixed mask ratios."""
    mx.random.seed(seed)
    T = seq_len
    rs = resp_start if resp_start is not None else T // 2
    x = mx.random.randint(0, BASE_VOCAB, (1, T)).astype(mx.int32)
    resp_mask = (mx.arange(T)[None, :] >= rs)

    out: dict[float, float] = {}
    for ratio in ratios:
        masked = resp_mask & (mx.random.uniform(shape=(1, T)) < ratio)
        n = float(masked.sum().item())
        if n == 0:
            out[ratio] = float("nan")
            continue
        pred = mx.argmax(model(mx.where(masked, mask_id, x)), axis=-1)
        acc = float((((pred == x) & masked).sum() / masked.sum()).item())
        out[ratio] = acc
    return out


# ── (C) iterative reconstruction vs gold ───────────────────────────────────────

def iterative_reconstruction(
    model, *, n_prompt: int = 16, gen_len: int = 48, steps: int = 16,
    temperature: float = 0.0, seed: int = 0, static: bool = False,
) -> dict:
    """Generate from all-[MASK] and compare to a random gold sequence."""
    mx.random.seed(seed)
    full = mx.random.randint(0, BASE_VOCAB, (n_prompt + gen_len,)).astype(mx.int32).tolist()
    prompt_ids = [int(t) for t in full[:n_prompt]]
    gold = [int(t) for t in full[n_prompt:]]

    if static:
        res = StaticDLLM(model).generate(prompt_ids, gen_len, steps=steps,
                                         temperature=temperature, seed=seed)
    else:
        res = iterative_unmask(model, prompt_ids, gen_len, steps=steps,
                               temperature=temperature, seed=seed)
    gen = res.response_ids
    matches = sum(1 for a, b in zip(gen, gold) if a == b)
    return {
        "token_accuracy": matches / max(1, gen_len),
        "exact_match": gen == gold,
        "tokens_per_s": res.tokens_per_s,
        "forwards_per_s": res.forwards_per_s,
        "compile_s": res.compile_s,
    }
