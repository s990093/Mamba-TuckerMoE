import mlx.core as mx

from .sampler import sample_logits, apply_repetition_penalty, apply_freq_presence_penalty


def prefill(model, prompt_ids):
    """Run prefill on the whole prompt. Returns (last_logits, states)."""
    ids = mx.array(prompt_ids, dtype=mx.int32)[None, :]   # (1, L)
    logits, states = model(ids, states=None)
    return logits[0, -1], states


def decode_step(model, last_token: int, states):
    ids = mx.array([[int(last_token)]], dtype=mx.int32)
    logits, states = model(ids, states=states)
    return logits[0, -1], states


def generate(model, prompt_ids, gen_config, stop_token_ids=None, on_token=None):
    """Autoregressive generation. Returns list[int] of generated token ids (excluding prompt)."""
    stop_set = set(stop_token_ids or [])
    key = mx.random.key(gen_config.seed)

    last_logits, states = prefill(model, prompt_ids)
    mx.eval(last_logits, *_iter_states(states))

    generated = []
    # repeat window
    window = max(1, gen_config.repeat_last_n)

    for step in range(gen_config.max_tokens):
        # Penalties
        z = last_logits.astype(mx.float32)
        recent = (prompt_ids + generated)[-window:] if window > 0 else (prompt_ids + generated)
        z = apply_repetition_penalty(z, recent, gen_config.rep_pen)
        z = apply_freq_presence_penalty(z, recent, gen_config.pres_pen, gen_config.freq_pen)

        tok, key = sample_logits(z,
                                 gen_config.temperature,
                                 gen_config.top_k,
                                 gen_config.top_p,
                                 gen_config.min_p,
                                 key)
        mx.eval(tok)
        tok_id = int(tok.item())

        if on_token is not None:
            on_token(tok_id)

        generated.append(tok_id)
        if tok_id in stop_set:
            break

        last_logits, states = decode_step(model, tok_id, states)
        mx.eval(last_logits, *_iter_states(states))

    return generated


def _iter_states(states):
    out = []
    for st in states:
        if st is None:
            continue
        for v in st.values():
            if v is not None:
                out.append(v)
    return out
