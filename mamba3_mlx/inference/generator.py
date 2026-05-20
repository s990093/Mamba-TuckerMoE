"""Prefill / decode loop for Mamba3LanguageModel."""

from __future__ import annotations

import time
from typing import Callable, NamedTuple

import mlx.core as mx

from .sampler import sample_logits, apply_repetition_penalty, apply_freq_presence_penalty


# ── Return type ────────────────────────────────────────────────────────────────

class GenerateResult(NamedTuple):
    tokens: list[int]           # all generated token ids (including warmup tokens)
    stop_reason: str            # "max_tokens" | "eos" | "stop_string"
    n_prompt: int               # prompt length in tokens
    n_warmup: int               # compile-warmup steps (real tokens, not timed)
    elapsed_prefill: float      # prefill wall-clock (seconds)
    elapsed_decode: float       # timed decode wall-clock (seconds, warmup excluded)
    prefill_tps: float          # n_prompt / elapsed_prefill
    decode_tps: float           # (len(tokens) - n_warmup) / elapsed_decode


# ── State helpers ──────────────────────────────────────────────────────────────

def _eval_states(states: list) -> None:
    """Force-evaluate all state arrays so lazy computation is flushed."""
    for st in states:
        if st is None:
            continue
        for v in st.values():
            if v is not None:
                mx.eval(v)


def _iter_state_arrays(states: list) -> list:
    out = []
    for st in states:
        if st is None:
            continue
        for v in st.values():
            if v is not None:
                out.append(v)
    return out


# ── Compile helpers ────────────────────────────────────────────────────────────

def compile_model(model) -> object:
    """Return an mx.compile-wrapped version of the model.

    The compiled callable has the same signature as model.__call__:
        (input_ids, states=...) -> (logits, new_states)

    On the first call (warm-up), MLX traces and compiles the computation graph.
    All subsequent calls with the same shapes reuse the compiled graph.
    """
    return mx.compile(model)


def warmup(compiled_model, last_logits, states, n_steps: int):
    """Run ``n_steps`` real decode steps to trigger JIT compilation.

    Uses greedy sampling (no RNG) so the graph is deterministically traced.
    Returns (last_logits, states) after the warmup steps.
    The generated tokens are included in the caller's generated list so the
    conversation state stays consistent.
    """
    logits = last_logits
    cur_states = states
    warmup_ids: list[int] = []

    for _ in range(n_steps):
        tok_id = int(mx.argmax(logits).item())
        warmup_ids.append(tok_id)
        ids = mx.array([[tok_id]], dtype=mx.int32)
        logits_out, cur_states = compiled_model(ids, states=cur_states)
        logits = logits_out[0, -1]
        mx.eval(logits, *_iter_state_arrays(cur_states))

    return logits, cur_states, warmup_ids


# ── Public API ─────────────────────────────────────────────────────────────────

def prefill(model, prompt_ids: list[int]):
    """Full-sequence prefill. Returns (logits_at_last_pos, states, elapsed_s)."""
    ids = mx.array(prompt_ids, dtype=mx.int32)[None, :]
    t0 = time.perf_counter()
    logits, states = model(ids, states=None)
    mx.eval(logits, *_iter_state_arrays(states))
    elapsed = time.perf_counter() - t0
    return logits[0, -1], states, elapsed


def generate(
    model,
    prompt_ids: list[int],
    gen_config,
    *,
    stop_token_ids: list[int] | None = None,
    stop_strings: list[str] | None = None,
    no_eos_stop: bool = False,
    full_decode_compile: bool = False,
    warmup_steps: int = 3,
    tokenizer=None,
    on_token: Callable[[int], None] | None = None,
) -> GenerateResult:
    """Autoregressive generation.

    Parameters
    ----------
    full_decode_compile:
        Compile the full model graph with ``mx.compile`` before the decode
        loop.  ``warmup_steps`` greedy-sampled steps are run first to pay the
        compilation cost, then actual timed generation begins.
    warmup_steps:
        Number of real decode steps used solely for JIT compilation warm-up.
        These tokens appear in the output but are excluded from ``decode_tps``.
    stop_token_ids:
        Token ids that halt generation.  Ignored when ``no_eos_stop`` is True.
    stop_strings:
        Decoded substrings that halt generation (multi-token safe).
        Requires ``tokenizer``.
    no_eos_stop:
        Ignore ``stop_token_ids`` — run until ``max_tokens`` or a
        ``stop_strings`` match.  Useful for inspecting model output past EOS.
    """
    stop_set: set[int] = set() if no_eos_stop else set(stop_token_ids or [])
    stop_strings = stop_strings or []
    if stop_strings and tokenizer is None:
        raise ValueError("stop_strings requires tokenizer to be passed")
    _stop_str_window = max((len(s) * 4 for s in stop_strings), default=0)

    key = mx.random.key(gen_config.seed)
    window = max(1, gen_config.repeat_last_n)

    # ── Prefill ────────────────────────────────────────────────────────────────
    last_logits, states, elapsed_prefill = prefill(model, prompt_ids)
    prefill_tps = len(prompt_ids) / max(elapsed_prefill, 1e-6)

    # ── Compile + warmup ───────────────────────────────────────────────────────
    decode_model = model
    n_warmup = 0
    generated: list[int] = []

    if full_decode_compile:
        decode_model = compile_model(model)
        actual_warmup = min(warmup_steps, gen_config.max_tokens)
        if actual_warmup > 0:
            last_logits, states, warmup_ids = warmup(
                decode_model, last_logits, states, actual_warmup)
            n_warmup = len(warmup_ids)
            generated.extend(warmup_ids)
            if on_token is not None:
                for tid in warmup_ids:
                    on_token(tid)

    # ── Timed decode loop ──────────────────────────────────────────────────────
    stop_reason = "max_tokens"
    t_decode_start = time.perf_counter()
    remaining = gen_config.max_tokens - n_warmup

    for _step in range(remaining):
        z = last_logits.astype(mx.float32)
        recent = (list(prompt_ids) + generated)[-window:]
        z = apply_repetition_penalty(z, recent, gen_config.rep_pen)
        z = apply_freq_presence_penalty(z, recent, gen_config.pres_pen, gen_config.freq_pen)

        tok_arr, key = sample_logits(
            z, gen_config.temperature, gen_config.top_k,
            gen_config.top_p, gen_config.min_p, key,
        )
        mx.eval(tok_arr)
        tok_id = int(tok_arr.item())
        generated.append(tok_id)

        if on_token is not None:
            on_token(tok_id)

        if tok_id in stop_set:
            stop_reason = "eos"
            break

        if stop_strings:
            tail_text = tokenizer.decode(
                generated[-max(_stop_str_window, 1):],
                skip_special_tokens=False,
            )
            if any(s in tail_text for s in stop_strings):
                stop_reason = "stop_string"
                break

        ids = mx.array([[tok_id]], dtype=mx.int32)
        logits_out, states = decode_model(ids, states=states)
        last_logits = logits_out[0, -1]
        mx.eval(last_logits, *_iter_state_arrays(states))

    elapsed_decode = time.perf_counter() - t_decode_start
    timed_tokens = len(generated) - n_warmup
    decode_tps = timed_tokens / max(elapsed_decode, 1e-6)

    return GenerateResult(
        tokens=generated,
        stop_reason=stop_reason,
        n_prompt=len(prompt_ids),
        n_warmup=n_warmup,
        elapsed_prefill=elapsed_prefill,
        elapsed_decode=elapsed_decode,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
    )


# Legacy alias (used by scripts that call decode_step directly)
def decode_step(model, last_token: int, states):
    ids = mx.array([[int(last_token)]], dtype=mx.int32)
    logits, states = model(ids, states=states)
    return logits[0, -1], states
