# -*- coding: utf-8 -*-
"""
Autoregressive generation loop — official MLX LLM inference pattern.

Lazy-evaluation strategy (matches mlx-lm / mlx-examples/llms):
────────────────────────────────────────────────────────────────
MLX builds a compute graph lazily; no GPU work happens until mx.eval().
We follow the same discipline as the official MLX Llama example:

  1. model() forward  → lazy graph (no eval)
  2. sample_token()   → mx.eval(logits) inside via .item(), returns Python int
  3. _eval_states()   → eval Mamba h/angles + KV k/v to cap graph growth

mx.compile wraps the inner decode forward so graph compilation amortises
over the full generation, matching official mlx-lm compile behaviour.
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Generator, Callable, NamedTuple

import mlx.core as mx
import numpy as np

from ..mlx_model.hybrid_model import Mamba3LanguageModel
from ..mlx_model.state_utils import MAMBA_STATE_KEYS as _MAMBA_STATE_KEYS
from ..utils.config import GenerationConfig
from .sampler import (
    apply_repetition_penalty,
    apply_presence_frequency_penalty,
    sample_token,
)

# Optional Metal sampler (enabled via config.use_metal_sampling)
try:
    from .sampler_metal import sample_token_metal_full
    _METAL_SAMPLING_AVAILABLE = True
except Exception as e:
    _METAL_SAMPLING_AVAILABLE = False
    _METAL_IMPORT_ERROR = str(e)

def _assert_state_schema(mamba_states: list, label: str = "") -> None:
    """
    Fail fast if any Mamba state dict is missing required keys.
    Catches schema drift (e.g. a new field added in mamba_block but not
    in init_empty_states or _eval_states) before it silently produces zeros.
    """
    for i, s in enumerate(mamba_states):
        missing = _MAMBA_STATE_KEYS - set(s)
        extra   = set(s) - _MAMBA_STATE_KEYS
        if missing:
            raise RuntimeError(
                f"[{label}] mamba_states[{i}] missing keys: {missing}. "
                f"Update init_empty_states and _eval_states to match mamba_block."
            )
        if extra:
            raise RuntimeError(
                f"[{label}] mamba_states[{i}] has unexpected keys: {extra}. "
                f"Update _MAMBA_STATE_KEYS if this is intentional."
            )


# ── ChatML wrapper ─────────────────────────────────────────────────────────────

def wrap_prompt_chatml(user_text: str) -> str:
    return (
        "<|im_start|>user\n"
        f"{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ── State eval helper ──────────────────────────────────────────────────────────

def _eval_states(mamba_states: list, kv_caches: list) -> None:
    """
    Materialise Mamba hidden states and KV caches at the decode loop boundary.
    Caps the compute graph to a single step rather than growing unboundedly.
    """
    arrays = []
    for s in mamba_states:
        arrays.append(s["h"])
        arrays.append(s["angles"])
        arrays.append(s["last_ip"])
    for kv in kv_caches:
        if kv.get("k") is not None:
            arrays.append(kv["k"])
            arrays.append(kv["v"])
    if arrays:
        mx.eval(*arrays)


# ── Prefill ────────────────────────────────────────────────────────────────────

def prefill(
    model: Mamba3LanguageModel,
    input_ids: mx.array,        # (1, L)
    step: Optional[int] = None,
):
    """
    Full-sequence forward pass (prefill).
    Returns lazy arrays — caller decides when to eval.
    """
    mamba_states, kv_caches = model.init_empty_states(batch_size=1)
    logits, _, new_mamba, new_kv = model(
        input_ids, step=step,
        mamba_states=mamba_states,
        kv_caches=kv_caches,
    )
    logits_last = logits[:, -1, :]     # (1, V) — still lazy
    return logits_last, new_mamba, new_kv


# ── Continuation prefill (with existing states) ────────────────────────────────

def prefill_continuation(
    model: Mamba3LanguageModel,
    input_ids: mx.array,        # (1, L) — suffix after the primed system block
    mamba_states: list,
    kv_caches: list,
    step: Optional[int] = None,
):
    """
    Prefill a suffix on top of pre-existing states (e.g. a primed system prefix).
    Avoids re-running the system block for every chat turn.
    Returns lazy arrays — caller decides when to eval.
    """
    logits, _, new_mamba, new_kv = model(
        input_ids, step=step,
        mamba_states=mamba_states,
        kv_caches=kv_caches,
    )
    logits_last = logits[:, -1, :]
    return logits_last, new_mamba, new_kv


# ── Single decode step ─────────────────────────────────────────────────────────

def decode_step(
    model: Mamba3LanguageModel,
    token_id: int,
    mamba_states: list,
    kv_caches: list,
    step: Optional[int] = None,
):
    """
    Single-token decode forward.
    Returns lazy arrays — no mx.eval inside.
    """
    x = mx.array([[token_id]])          # (1, 1)
    logits, _, new_mamba, new_kv = model(
        x, step=step,
        mamba_states=mamba_states,
        kv_caches=kv_caches,
    )
    return logits[:, -1, :], new_mamba, new_kv   # (1, V), lazy


# ── Per-step output (official mlx-lm pattern) ─────────────────────────────────

class GenerationStep(NamedTuple):
    token: int
    logprobs: Optional[mx.array]   # (V,) lazy — caller evals if needed


# ── Sampling selector (Metal vs MLX) ───────────────────────────────────────────

def _sample_token_wrapper(
    logits_1d: mx.array,
    gen_cfg: GenerationConfig,
    generated: List[int],
) -> int:
    """
    Choose between Metal and MLX sampler based on config.

    Metal sampler requires token counts for penalty application;
    we reconstruct from generated list.
    """
    if gen_cfg.use_metal_sampling and _METAL_SAMPLING_AVAILABLE:
        # Count token occurrences for penalty application
        vocab_size = logits_1d.shape[0]
        token_counts = mx.zeros((vocab_size,))

        # Build count array from generated tokens (within repetition window)
        recent = generated[-gen_cfg.repetition_window:] if gen_cfg.repetition_window else generated
        counts_np = np.zeros(vocab_size, dtype=np.float32)
        for tid in recent:
            if 0 <= tid < vocab_size:
                counts_np[tid] += 1.0
        token_counts = mx.array(counts_np)

        # Metal sampler: pass logits directly (already penalized in stream_generate)
        # Create a mock args object with sampling parameters
        class _MetalArgs:
            pass

        args = _MetalArgs()
        args.temp = gen_cfg.temperature
        args.top_k = gen_cfg.top_k
        args.top_p = gen_cfg.top_p
        args.min_p = gen_cfg.min_p
        args.rep_pen = gen_cfg.repetition_penalty
        args.pres_pen = gen_cfg.presence_penalty
        args.freq_pen = gen_cfg.frequency_penalty
        args.fast_sample = False

        token = int(sample_token_metal_full(
            logits_1d, token_counts, args,
            threadgroup_size=gen_cfg.metal_threadgroup_size
        ).item())
        return token
    else:
        # Standard MLX sampler
        return sample_token(
            logits_1d, gen_cfg.temperature, gen_cfg.top_k,
            gen_cfg.top_p, gen_cfg.min_p
        )


# ── Core token stream (official pattern) ──────────────────────────────────────

def stream_generate(
    model: Mamba3LanguageModel,
    prompt_ids: List[int],
    gen_cfg: GenerationConfig,
    step_offset: int = 0,
    return_logprobs: bool = False,
) -> Generator[GenerationStep, None, None]:
    """
    Low-level streaming generator that yields one GenerationStep per token.

    Follows the official MLX LLM example pattern:
      • All ops inside the decode step are wrapped in mx.compile when
        gen_cfg.full_decode_compile is True (matches mlx-lm behaviour).
      • Graph is evaluated once per token at the sample boundary.
      • States are evaluated once per token to cap graph growth.

    The caller decides when to .item() / decode tokens — we yield lazily.
    """
    prompt_tensor = mx.array([prompt_ids])
    generated: List[int] = list(prompt_ids)

    # Optionally compile the decode step.
    #
    # LIMITATION: mx.compile requires stable shapes across calls.  Our Transformer
    # KV cache grows by 1 every decode step (T → T+1), which triggers retracing
    # on every step — nullifying the compile benefit.  Until the KV cache is
    # pre-allocated to a fixed max size, full_decode_compile gives no speedup
    # for this hybrid model and is kept as a forward-compatibility hook only.
    #
    # IMPORTANT: state must always be passed as function arguments, never stored
    # on self — mx.compile captures arrays it sees at trace time as constants.
    # Any mutable state on self would be "frozen" in the compiled graph.
    if gen_cfg.full_decode_compile:
        def _decode_fn(model, token_id, mamba_states, kv_caches, step=None):
            return decode_step(model, token_id, mamba_states, kv_caches, step=step)
        _decode_fn = mx.compile(_decode_fn)
    else:
        _decode_fn = decode_step

    # ── Prefill ───────────────────────────────────────────────────────────────
    logits, mamba_states, kv_caches = prefill(model, prompt_tensor, step=step_offset)
    _assert_state_schema(mamba_states, label="prefill")  # fail fast on schema drift

    # Apply penalties on the still-lazy logit tensor (pure graph ops)
    logits_1d = logits[0]
    logits_1d = apply_repetition_penalty(
        logits_1d, generated, gen_cfg.repetition_penalty, gen_cfg.repetition_window)
    logits_1d = apply_presence_frequency_penalty(
        logits_1d, generated,
        gen_cfg.presence_penalty, gen_cfg.frequency_penalty, gen_cfg.repetition_window)

    # sample_token triggers mx.eval internally — first GPU sync
    # Uses either Metal or MLX sampler based on config
    token = _sample_token_wrapper(logits_1d, gen_cfg, generated)

    # Eval states to bound graph before entering decode loop
    _eval_states(mamba_states, kv_caches)

    logprobs_out = mx.log_softmax(logits_1d) if return_logprobs else None
    generated.append(token)
    n_generated = 1
    yield GenerationStep(token=token, logprobs=logprobs_out)

    if token in gen_cfg.stop_token_ids and n_generated >= gen_cfg.min_new_tokens:
        return

    # ── Decode loop ───────────────────────────────────────────────────────────
    _first_decode = True
    for i in range(1, gen_cfg.max_new_tokens):
        step = step_offset + i

        # ① Build graph lazily
        logits, mamba_states, kv_caches = _decode_fn(
            model, token, mamba_states, kv_caches, step=step)
        if _first_decode:
            _assert_state_schema(mamba_states, label="decode[0]")
            _first_decode = False

        # ② Penalty ops — still lazy MLX graph nodes
        logits_1d = logits[0]
        logits_1d = apply_repetition_penalty(
            logits_1d, generated, gen_cfg.repetition_penalty, gen_cfg.repetition_window)
        logits_1d = apply_presence_frequency_penalty(
            logits_1d, generated,
            gen_cfg.presence_penalty, gen_cfg.frequency_penalty, gen_cfg.repetition_window)

        # ③ sample_token: single GPU sync for logits (Metal or MLX)
        token = _sample_token_wrapper(logits_1d, gen_cfg, generated)

        # ④ Eval states: caps graph size at one step
        _eval_states(mamba_states, kv_caches)

        logprobs_out = mx.log_softmax(logits_1d) if return_logprobs else None
        generated.append(token)
        n_generated += 1
        yield GenerationStep(token=token, logprobs=logprobs_out)

        if token in gen_cfg.stop_token_ids and n_generated >= gen_cfg.min_new_tokens:
            break


# ── High-level generate (back-compat, yields plain ints) ──────────────────────

def generate(
    model: Mamba3LanguageModel,
    prompt_ids: List[int],
    gen_cfg: GenerationConfig,
    step_offset: int = 0,
    verbose: bool = False,
) -> Generator[int, None, None]:
    """
    Streaming token generator.  Yields one integer token id at a time.

    Thin wrapper over stream_generate that adds optional verbose timing and
    maintains the original int-yielding interface for backward compatibility.
    """
    t0 = time.perf_counter()
    n_generated = 0
    t_prefill_end: Optional[float] = None

    for i, step in enumerate(stream_generate(model, prompt_ids, gen_cfg, step_offset)):
        if i == 0:
            t_prefill_end = time.perf_counter()
            if verbose:
                t_pre = t_prefill_end - t0
                print(f"[prefill] {len(prompt_ids)} tokens in {t_pre*1000:.0f} ms "
                      f"({len(prompt_ids)/t_pre:.0f} tok/s)")

        yield step.token
        n_generated += 1

    if verbose and n_generated > 1 and t_prefill_end is not None:
        elapsed = time.perf_counter() - t_prefill_end
        print(f"[decode] {n_generated} tokens in {elapsed:.2f}s "
              f"→ {n_generated/elapsed:.1f} tok/s")


# ── Speculative decoding scaffold ─────────────────────────────────────────────

def generate_speculative(
    target_model: Mamba3LanguageModel,
    draft_fn: Callable,
    prompt_ids: List[int],
    gen_cfg: GenerationConfig,
    step_offset: int = 0,
    verbose: bool = False,
) -> Generator[int, None, None]:
    """
    Speculative decoding.  draft_fn has same signature as decode_step.
    Evaluation discipline: same as generate() — eval per accepted token.
    """
    import random
    K = gen_cfg.num_speculative_tokens
    prompt_tensor = mx.array([prompt_ids])
    generated: List[int] = list(prompt_ids)

    logits, t_mamba, t_kv = prefill(target_model, prompt_tensor, step=step_offset)
    _, d_mamba, d_kv = prefill(target_model, prompt_tensor, step=step_offset)
    _eval_states(t_mamba, t_kv)
    _eval_states(d_mamba, d_kv)

    token = _sample_token_wrapper(logits[0], gen_cfg, generated)
    generated.append(token)
    yield token
    if token in gen_cfg.stop_token_ids:
        return

    step = step_offset + 1
    n_generated = 1

    while n_generated < gen_cfg.max_new_tokens:
        # Draft: K tokens
        draft_tokens: List[int] = []
        draft_probs_np: List[np.ndarray] = []
        cur = token
        cur_d_mamba, cur_d_kv = d_mamba, d_kv

        for _ in range(K):
            d_logits, cur_d_mamba, cur_d_kv = draft_fn(cur, cur_d_mamba, cur_d_kv, step)
            d_prob = mx.softmax(d_logits[0].astype(mx.float32))
            mx.eval(d_prob)
            d_prob_np = np.array(d_prob, dtype=np.float64)
            d_prob_np /= d_prob_np.sum()
            dt = int(np.random.choice(len(d_prob_np), p=d_prob_np))
            draft_tokens.append(dt)
            draft_probs_np.append(d_prob_np)
            _eval_states(cur_d_mamba, cur_d_kv)
            cur = dt
            step += 1

        # Verify with target
        t_logits_all: List[np.ndarray] = []
        cur_t_mamba, cur_t_kv = t_mamba, t_kv
        for tok in [token] + draft_tokens:
            t_logits, cur_t_mamba, cur_t_kv = decode_step(target_model, tok,
                                                           cur_t_mamba, cur_t_kv)
            t_prob = mx.softmax(t_logits[0].astype(mx.float32))
            mx.eval(t_prob)
            t_logits_all.append(np.array(t_prob, dtype=np.float64))
            _eval_states(cur_t_mamba, cur_t_kv)

        # Accept / reject
        accepted = 0
        for k in range(K):
            dt = draft_tokens[k]
            r = min(1.0, t_logits_all[k][dt] / (draft_probs_np[k][dt] + 1e-8))
            if random.random() < r:
                generated.append(dt)
                yield dt
                n_generated += 1
                accepted += 1
                if dt in gen_cfg.stop_token_ids:
                    return
            else:
                corrected = np.maximum(t_logits_all[k] - draft_probs_np[k], 0.0)
                corrected /= corrected.sum() + 1e-8
                bonus = int(np.random.choice(len(corrected), p=corrected))
                generated.append(bonus)
                yield bonus
                n_generated += 1
                break
        else:
            bonus_probs = t_logits_all[-1]
            bonus_probs /= bonus_probs.sum()
            bonus = int(np.random.choice(len(bonus_probs), p=bonus_probs))
            generated.append(bonus)
            yield bonus
            n_generated += 1

        t_mamba, t_kv = cur_t_mamba, cur_t_kv
        d_mamba, d_kv = cur_d_mamba, cur_d_kv
        token = generated[-1]
