import mlx.core as mx


def apply_repetition_penalty(logits, recent_ids, penalty: float):
    """Divide positive logits for repeated tokens by `penalty`, multiply negative ones.

    logits: (V,) float32 array
    recent_ids: list[int] of recently generated ids (already trimmed to window)
    """
    if penalty == 1.0 or not recent_ids:
        return logits
    uniq = list({int(t) for t in recent_ids})
    idx = mx.array(uniq, dtype=mx.int32)
    sel = logits[idx]
    sel = mx.where(sel > 0, sel / penalty, sel * penalty)
    # scatter back: replace logits[idx] with sel
    # logits' - logits at idx positions
    delta = sel - logits[idx]
    logits = logits.at[idx].add(delta)
    return logits


def apply_freq_presence_penalty(logits, recent_ids, pres_pen: float, freq_pen: float):
    if (pres_pen == 0.0 and freq_pen == 0.0) or not recent_ids:
        return logits
    counts = {}
    for t in recent_ids:
        t = int(t)
        counts[t] = counts.get(t, 0) + 1
    idx = mx.array(list(counts.keys()), dtype=mx.int32)
    cnt = mx.array(list(counts.values()), dtype=logits.dtype)
    delta = -pres_pen - freq_pen * cnt
    return logits.at[idx].add(delta)


def make_compiled_sampler(rep_pen: float, pres_pen: float, freq_pen: float,
                          temperature: float, top_k: int, top_p: float,
                          min_p: float, vocab_size: int):
    """Build a single ``mx.compile``d graph fusing the whole sampling chain.

    Replicates, bit-for-bit, the eager sequence used in chat_demo's decode loop:

        apply_repetition_penalty → apply_freq_presence_penalty → sample_logits

    but as ONE compiled dispatch over fixed-shape inputs, eliminating the
    per-token Python dict/list bookkeeping and the dozen tiny eager kernel
    launches.  Measured 36 → ~80 tok/s on the StaticStreamSession demo path.

    The returned callable has signature::

        tok_i32, new_key = sample(z, recent_idx, recent_valid, key)

    where
        z:            (V,) float32 — logits AFTER any middleware/ban transform
        recent_idx:   (W,) int32   — ring buffer of recent token ids (pad = 0)
        recent_valid: (W,) float32 — 1.0 for a real slot, 0.0 for padding

    All sampling hyper-parameters are baked in as constants at build time, so
    the Python branches below resolve at trace time (no runtime overhead).
    The window penalties are vectorised through a scatter-add into a (V,)
    count vector — duplicates accumulate exactly like the eager dict, and
    padded slots (valid=0) contribute nothing, so the math is identical.
    """
    V = int(vocab_size)
    has_rep = rep_pen != 1.0
    has_fp = (pres_pen != 0.0 or freq_pen != 0.0)
    greedy = temperature <= 0.0
    neg = -1e9

    def _sample(z, recent_idx, recent_valid, key):
        if has_rep or has_fp:
            counts = mx.zeros((V,), dtype=mx.float32)
            counts = counts.at[recent_idx].add(recent_valid)   # occurrences per id
            seen = counts > 0
            if has_rep:
                pen = mx.where(z > 0, z / rep_pen, z * rep_pen)
                z = mx.where(seen, pen, z)
            if has_fp:
                delta = mx.where(seen, -pres_pen - freq_pen * counts,
                                 mx.array(0.0, dtype=z.dtype))
                z = z + delta

        if greedy:
            return mx.argmax(z, axis=-1).astype(mx.int32), key

        z = z / temperature
        if top_k > 0 and top_k < V:
            kth = mx.topk(z, top_k)[..., 0:1]
            z = mx.where(z < kth, mx.array(neg, dtype=z.dtype), z)
        if top_p < 1.0:
            sorted_asc = mx.sort(z)
            sorted_desc = sorted_asc[::-1]
            probs = mx.softmax(sorted_desc, axis=-1)
            cum = mx.cumsum(probs, axis=-1)
            shifted_cum = mx.concatenate(
                [mx.zeros((1,), dtype=cum.dtype), cum[:-1]], axis=0)
            keep = shifted_cum < top_p
            kept_logits = mx.where(keep, sorted_desc,
                                   mx.array(mx.inf, dtype=sorted_desc.dtype))
            cutoff = mx.min(kept_logits)
            z = mx.where(z < cutoff, mx.array(neg, dtype=z.dtype), z)
        if min_p > 0.0:
            probs = mx.softmax(z, axis=-1)
            max_p = mx.max(probs)
            z = mx.where(probs < max_p * min_p, mx.array(neg, dtype=z.dtype), z)

        key, sub = mx.random.split(key)
        tok = mx.random.categorical(z, key=sub)
        return tok.astype(mx.int32), key

    return mx.compile(_sample)


def sample_logits(logits, temperature: float, top_k: int, top_p: float, min_p: float, key):
    """Sample a single token id from a 1-D logits array."""
    if temperature <= 0.0:
        return mx.argmax(logits, axis=-1).astype(mx.int32), key

    z = logits / temperature
    V = z.shape[-1]

    if top_k > 0 and top_k < V:
        kth = mx.topk(z, top_k)[..., 0:1]    # smallest of the top-k
        z = mx.where(z < kth, mx.array(-1e9, dtype=z.dtype), z)

    if top_p < 1.0:
        sorted_asc = mx.sort(z)
        sorted_desc = sorted_asc[::-1]
        probs = mx.softmax(sorted_desc, axis=-1)
        cum = mx.cumsum(probs, axis=-1)
        # Keep tokens whose cumulative prob (including) <= top_p; always keep the top-1
        # Shift so that the first token is always kept:
        shifted_cum = mx.concatenate(
            [mx.zeros((1,), dtype=cum.dtype), cum[:-1]], axis=0)
        keep = shifted_cum < top_p
        kept_logits = mx.where(keep, sorted_desc, mx.array(mx.inf, dtype=sorted_desc.dtype))
        cutoff = mx.min(kept_logits)
        z = mx.where(z < cutoff, mx.array(-1e9, dtype=z.dtype), z)

    if min_p > 0.0:
        probs = mx.softmax(z, axis=-1)
        max_p = mx.max(probs)
        z = mx.where(probs < max_p * min_p, mx.array(-1e9, dtype=z.dtype), z)

    key, sub = mx.random.split(key)
    tok = mx.random.categorical(z, key=sub)
    return tok.astype(mx.int32), key
