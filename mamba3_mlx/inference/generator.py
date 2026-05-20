"""Prefill / decode loop for Mamba3LanguageModel."""

from __future__ import annotations

import time
from typing import Callable, NamedTuple

import mlx.core as mx

from .sampler import sample_logits, apply_repetition_penalty, apply_freq_presence_penalty


# ── Return type ────────────────────────────────────────────────────────────────

class GenerateResult(NamedTuple):
    tokens: list[int]           # generated token ids (excluding prompt)
    stop_reason: str            # "max_tokens" | "eos" | "stop_string"
    n_prompt: int               # prompt length in tokens
    elapsed: float              # wall-clock seconds (excludes model load)
    tps: float                  # tokens / second


# ── Internals ─────────────────────────────────────────────────────────────────

def _iter_state_arrays(states: list) -> list:
    out = []
    for st in states:
        if st is None:
            continue
        for v in st.values():
            if v is not None:
                out.append(v)
    return out


def _make_decode_fn(model, full_compile: bool):
    """Return a callable (ids, states) -> (logits_1d, states).

    When full_compile=True the model forward is wrapped in mx.compile so the
    computation graph is reused across steps (shapes are fixed at B=1, L=1).
    """
    if full_compile:
        _compiled = mx.compile(model)

        def _fn(last_token: int, states):
            ids = mx.array([[last_token]], dtype=mx.int32)
            logits, new_states = _compiled(ids, states=states)
            return logits[0, -1], new_states
    else:
        def _fn(last_token: int, states):
            ids = mx.array([[last_token]], dtype=mx.int32)
            logits, new_states = model(ids, states=states)
            return logits[0, -1], new_states

    return _fn


def prefill(model, prompt_ids: list[int]):
    """Full-sequence prefill. Returns (logits_at_last_pos, states)."""
    ids = mx.array(prompt_ids, dtype=mx.int32)[None, :]
    logits, states = model(ids, states=None)
    return logits[0, -1], states


# ── Public generate ───────────────────────────────────────────────────────────

def generate(
    model,
    prompt_ids: list[int],
    gen_config,
    *,
    stop_token_ids: list[int] | None = None,
    stop_strings: list[str] | None = None,
    no_eos_stop: bool = False,
    full_decode_compile: bool = False,
    tokenizer=None,
    on_token: Callable[[int], None] | None = None,
) -> GenerateResult:
    """Autoregressive generation with full stop-condition control.

    Parameters
    ----------
    stop_token_ids:
        Token ids that trigger an immediate stop.  Ignored when ``no_eos_stop``
        is True.  Default: empty (caller should pass im_end / eos ids).
    stop_strings:
        Decoded strings that trigger a stop when they appear in the generated
        text.  Requires ``tokenizer`` to be passed.  Handles multi-token stop
        sequences (e.g. ``["</final>", "<|im_end|>"]``).
    no_eos_stop:
        When True, ``stop_token_ids`` are NOT checked — generation runs until
        ``max_tokens`` or a ``stop_strings`` match.  Useful for spec-decode
        debugging or inspecting model behaviour past the EOS boundary.
    full_decode_compile:
        Wrap the model forward in ``mx.compile`` for the decode loop.  The
        first decode step pays a compilation cost; subsequent steps reuse the
        graph.
    tokenizer:
        A ``tokenizers.Tokenizer`` instance.  Required when ``stop_strings`` is
        non-empty.
    on_token:
        Called with each sampled token id *before* deciding whether to stop.
        Suitable for streaming output.
    """
    stop_set: set[int] = set()
    if not no_eos_stop and stop_token_ids:
        stop_set = set(stop_token_ids)

    stop_strings = stop_strings or []
    if stop_strings and tokenizer is None:
        raise ValueError("stop_strings requires tokenizer to be passed")

    key = mx.random.key(gen_config.seed)
    window = max(1, gen_config.repeat_last_n)

    # ── Prefill ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    last_logits, states = prefill(model, prompt_ids)
    mx.eval(last_logits, *_iter_state_arrays(states))

    decode_fn = _make_decode_fn(model, full_decode_compile)
    generated: list[int] = []
    stop_reason = "max_tokens"

    # For stop_strings: check a rolling window of decoded text.
    # We keep the last max(len(s) for s in stop_strings) * 4 tokens decoded,
    # which is enough to detect any multi-token stop phrase.
    _stop_str_window = max((len(s) * 4 for s in stop_strings), default=0)

    # ── Decode loop ────────────────────────────────────────────────────────────
    for _step in range(gen_config.max_tokens):
        z = last_logits.astype(mx.float32)

        recent = (list(prompt_ids) + generated)[-window:]
        z = apply_repetition_penalty(z, recent, gen_config.rep_pen)
        z = apply_freq_presence_penalty(z, recent, gen_config.pres_pen, gen_config.freq_pen)

        tok_arr, key = sample_logits(
            z, gen_config.temperature, gen_config.top_k,
            gen_config.top_p, gen_config.min_p, key
        )
        mx.eval(tok_arr)
        tok_id = int(tok_arr.item())

        generated.append(tok_id)

        # Stream callback fires before stop check so stop token appears in output
        if on_token is not None:
            on_token(tok_id)

        # ── Stop conditions ────────────────────────────────────────────────────
        if tok_id in stop_set:
            stop_reason = "eos"
            break

        if stop_strings:
            tail = generated[-max(_stop_str_window, 1):]
            tail_text = tokenizer.decode(tail, skip_special_tokens=False)
            if any(s in tail_text for s in stop_strings):
                stop_reason = "stop_string"
                break

        last_logits, states = decode_fn(tok_id, states)
        mx.eval(last_logits, *_iter_state_arrays(states))

    elapsed = time.perf_counter() - t0
    n = len(generated)
    return GenerateResult(
        tokens=generated,
        stop_reason=stop_reason,
        n_prompt=len(prompt_ids),
        elapsed=elapsed,
        tps=n / max(elapsed, 1e-6),
    )


# Backwards-compatible alias used by run.py before the result namedtuple existed
def decode_step(model, last_token: int, states):
    ids = mx.array([[int(last_token)]], dtype=mx.int32)
    logits, states = model(ids, states=states)
    return logits[0, -1], states
