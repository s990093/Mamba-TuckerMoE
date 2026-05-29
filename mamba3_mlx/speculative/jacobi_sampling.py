"""SJD (Speculative Jacobi Decoding) for sampling mode.

Greedy ``jacobi_decode`` requires ``pred[i] == guess[i]`` (exact match) to
accept a draft token — under greedy decoding this yields byte-equal output
to AR-greedy.  But the model's argmax often disagrees with even a top-1
n-gram guess on a single rare token, capping ARL at ~2.

For *sampling* decoding the criterion can be relaxed:
    accept guess[i] with probability  p_T(guess[i] | filtered logits_{i-1})

If u ~ Uniform(0,1) < p_T(guess[i]), the draft token is accepted as the
sample from the target model's distribution; otherwise we resample from the
residual distribution and stop.  This is the rule from:

    Teng et al., "Accelerating Auto-regressive Text-to-Image Generation
    with Training-free Speculative Jacobi Decoding" (ICLR 2025, arXiv
    2410.01699), Algorithm 1 and §3.2.

It is the same rejection-sampling trick used in classic speculative
decoding (Leviathan et al. 2023) — *just with a deterministic draft*
(n-gram / suffix retrieval) so no draft model is needed.  Output is
distribution-equivalent to AR sampling with the same filters; it is **not**
byte-equal to AR greedy.

Why this can hit ARL ≥ 7 where greedy is stuck at ~3
----------------------------------------------------
The model's probability mass on top-1 is typically 0.5–0.95 once the
context is structured (CoT, lists, code).  At each position we accept
with that probability, so

    E[ARL] ≈ K · E[p_T(guess)]

For K=12 and a draft that averages p_T ≈ 0.6 (very achievable with n-gram
+ retrieval on repetitive text), expected ARL ≈ 7.2.
"""
from __future__ import annotations

import time
from typing import Callable, NamedTuple, Optional

import mlx.core as mx

from ..inference.generator import _iter_state_arrays, prefill
from .cot_cache import (
    CoTCachesArg,
    CoTPhaseTracker,
    infer_initial_phase,
    load_cot_caches,
)
from .drafts import SuffixRetriever, build_hybrid_branches  # noqa: F401
from .forward import extract_state_at, model_verify_forward
from .forward_metal import model_verify_forward_metal
from .jacobi import _build_guesses
from .ngram_cache import NGramCache


class JacobiSamplingResult(NamedTuple):
    tokens: list[int]
    stop_reason: str
    n_prompt: int
    elapsed_prefill: float
    elapsed_decode: float
    prefill_tps: float
    decode_tps: float
    n_rounds: int
    n_accepted: int          # tokens emitted from verify rounds (excl. first_token)
    arl: float               # n_accepted / n_rounds
    n_draft_hits: int        # # of draft sources that supplied a candidate
    K: int
    n_full_accepts: int      # rounds where draft accepted through all K positions
    K_history: list[int]


# ── Sampling utilities ──────────────────────────────────────────────────────

def _apply_filters(logits, temperature: float, top_k: int, top_p: float,
                   min_p: float):
    """Apply temperature + top-k + top-p + min-p filters; return *filtered
    probabilities* with the same shape as ``logits`` (last dim sums to 1).

    Works for any leading dim — pass shape ``(K, V)`` to filter K positions
    in one GPU dispatch.
    """
    if temperature <= 0.0:
        # Degenerate one-hot at argmax along the last axis.
        probs = mx.zeros_like(logits)
        idx = mx.argmax(logits, axis=-1, keepdims=True)
        # Scatter 1.0 at argmax position.  ``put_along_axis``-style:
        ones = mx.ones_like(idx).astype(probs.dtype)
        # Build via where: probs[..., j] = 1 if j == idx else 0.
        rng = mx.arange(logits.shape[-1])
        # Broadcast: idx has shape (..., 1), rng has shape (V,)
        probs = (rng[None, ...] == idx).astype(probs.dtype) if logits.ndim == 1 \
            else (rng == idx).astype(probs.dtype)
        return probs

    z = (logits / temperature).astype(mx.float32)
    V = z.shape[-1]

    if top_k > 0 and top_k < V:
        # mx.topk returns top-k values along last axis (sorted ascending).
        kth = mx.topk(z, top_k)[..., 0:1]
        z = mx.where(z < kth, mx.array(-1e9, dtype=z.dtype), z)

    if top_p < 1.0:
        # Sort along last axis (ascending → reverse to descending).
        sorted_desc = mx.sort(z, axis=-1)[..., ::-1]
        probs = mx.softmax(sorted_desc, axis=-1)
        cum = mx.cumsum(probs, axis=-1)
        # Shift right by 1 along last axis.
        zeros_shape = list(cum.shape)
        zeros_shape[-1] = 1
        shifted = mx.concatenate(
            [mx.zeros(zeros_shape, dtype=cum.dtype), cum[..., :-1]],
            axis=-1,
        )
        keep = shifted < top_p
        kept_logits = mx.where(
            keep, sorted_desc, mx.array(mx.inf, dtype=sorted_desc.dtype),
        )
        cutoff = mx.min(kept_logits, axis=-1, keepdims=True)
        z = mx.where(z < cutoff, mx.array(-1e9, dtype=z.dtype), z)

    if min_p > 0.0:
        p_pre = mx.softmax(z, axis=-1)
        max_p = mx.max(p_pre, axis=-1, keepdims=True)
        z = mx.where(p_pre < max_p * min_p, mx.array(-1e9, dtype=z.dtype), z)

    return mx.softmax(z, axis=-1)


def _categorical_from_probs(probs, key):
    """Sample one token id from a 1-D prob vector. Returns (tok, new_key)."""
    key, sub = mx.random.split(key)
    # mx.random.categorical takes logits; recover via log.
    tok = mx.random.categorical(mx.log(probs + 1e-20), key=sub)
    return tok.astype(mx.int32), key


# ── Compiled SJD acceptance kernel ──────────────────────────────────────────
# Folds (filter logits → filtered probs → index at draft tokens → uniform
# draw) into ONE compiled graph.  Filter hyper-parameters (temp, top_k,
# top_p, min_p) are baked in at build time as Python closures, so each
# (K, hyperparams) combo gets its own compiled graph cached by MLX.

def _build_compiled_accept(K: int, V: int, temperature: float,
                           top_k: int, top_p: float, min_p: float):
    """Return ``fn(logits_K_V, guesses, key) -> (draft_probs, us, all_probs)``
    where all three outputs evaluate together in one Metal dispatch."""

    arange_K1 = mx.arange(K - 1)

    def _inner(logits, guesses, sub_key):
        # Inline filter (avoids the Python-side branching of _apply_filters
        # so the JIT can fuse everything).
        z = (logits / temperature).astype(mx.float32)
        if top_k > 0 and top_k < V:
            kth = mx.topk(z, top_k)[..., 0:1]
            z = mx.where(z < kth, mx.array(-1e9, dtype=z.dtype), z)
        if top_p < 1.0:
            sorted_desc = mx.sort(z, axis=-1)[..., ::-1]
            ps = mx.softmax(sorted_desc, axis=-1)
            cum = mx.cumsum(ps, axis=-1)
            zeros_shape = list(cum.shape)
            zeros_shape[-1] = 1
            shifted = mx.concatenate(
                [mx.zeros(zeros_shape, dtype=cum.dtype), cum[..., :-1]],
                axis=-1,
            )
            keep = shifted < top_p
            kept = mx.where(keep, sorted_desc,
                            mx.array(mx.inf, dtype=sorted_desc.dtype))
            cutoff = mx.min(kept, axis=-1, keepdims=True)
            z = mx.where(z < cutoff, mx.array(-1e9, dtype=z.dtype), z)
        if min_p > 0.0:
            p_pre = mx.softmax(z, axis=-1)
            max_p = mx.max(p_pre, axis=-1, keepdims=True)
            z = mx.where(p_pre < max_p * min_p,
                         mx.array(-1e9, dtype=z.dtype), z)
        all_probs = mx.softmax(z, axis=-1)             # (K, V)
        # draft_probs: probs at guesses for positions 0..K-2.
        draft_probs = all_probs[arange_K1, guesses]    # (K-1,)
        us = mx.random.uniform(shape=(K - 1,), key=sub_key)
        return draft_probs, us, all_probs

    return mx.compile(_inner)


# ── Public API ──────────────────────────────────────────────────────────────

def jacobi_decode_sampling(
    model,
    prompt_ids: list[int],
    gen_config,
    *,
    K: int = 12,
    use_ngram: bool = True,
    ngram_n: int = 4,
    use_retrieval: bool = True,
    retrieval_max_suffix: int = 8,
    seed: int = 0,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    cache_warmup_tokens: Optional[list[int]] = None,
    preloaded_ngram: Optional[NGramCache] = None,
    preloaded_retriever: Optional[SuffixRetriever] = None,
    cot_caches: CoTCachesArg = None,
    cot_bucket: Optional[str] = None,
    metal_verify: bool = False,
    compile_verify: bool = False,
    stop_token_ids: Optional[list[int]] = None,
    stop_strings: Optional[list[str]] = None,
    no_eos_stop: bool = False,
    tokenizer=None,
    on_token: Optional[Callable[[int], None]] = None,
    on_emit: Optional[Callable[[int, bool], None]] = None,
    verbose: bool = False,
    adaptive_K: bool = False,
    K_min: int = 4,
    K_max: int = 24,
) -> JacobiSamplingResult:
    """SJD: probabilistic-acceptance Jacobi decoder.

    Output is distribution-equivalent to standard AR sampling with the same
    ``temperature/top_k/top_p/min_p`` filters.  It is **not** byte-equal to
    AR greedy — that's the whole point: relaxed acceptance gives much higher
    ARL on the same drafts.

    Parameters that default to ``None`` are read from ``gen_config``.
    """
    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")
    if adaptive_K and not (K_min <= K <= K_max):
        raise ValueError(
            f"adaptive_K requires K_min <= K <= K_max "
            f"(got K_min={K_min}, K={K}, K_max={K_max})"
        )
    if temperature is None:
        temperature = gen_config.temperature
    if top_k is None:
        top_k = gen_config.top_k
    if top_p is None:
        top_p = gen_config.top_p
    if min_p is None:
        min_p = gen_config.min_p

    stop_set: set[int] = set() if no_eos_stop else set(stop_token_ids or [])
    stop_strings = list(stop_strings or [])
    if stop_strings and tokenizer is None:
        raise ValueError("stop_strings requires tokenizer")
    _stop_str_window = max((len(s) * 4 for s in stop_strings), default=0)

    # ── Build verify forward (optionally with per-Mamba mx.compile) ─────────
    if metal_verify and compile_verify:
        raise ValueError("metal_verify and compile_verify are mutually exclusive")
    if compile_verify:
        from .forward_compiled import make_compiled_model_verify_forward
        _verify_fwd_fn = make_compiled_model_verify_forward(model)
    elif metal_verify:
        _verify_fwd_fn = model_verify_forward_metal
    else:
        _verify_fwd_fn = model_verify_forward

    # ── Prefill ──
    last_logits, state, elapsed_prefill = prefill(model, prompt_ids)
    prefill_tps = len(prompt_ids) / max(elapsed_prefill, 1e-6)

    key = mx.random.key(seed)
    generated: list[int] = []
    stop_reason = "max_tokens"

    # First token: SAMPLED from prefill logits (not greedy).
    probs0 = _apply_filters(last_logits, temperature, top_k, top_p, min_p)
    tok_arr, key = _categorical_from_probs(probs0, key)
    mx.eval(tok_arr)
    first_token = int(tok_arr.item())
    generated.append(first_token)
    if on_token is not None:
        on_token(first_token)
    if first_token in stop_set:
        return _empty(generated, "eos", prompt_ids, elapsed_prefill,
                      prefill_tps, K)
    if len(generated) >= gen_config.max_tokens:
        return _empty(generated, "max_tokens", prompt_ids, elapsed_prefill,
                      prefill_tps, K)

    ngram: Optional[NGramCache] = None
    if use_ngram:
        if preloaded_ngram is not None:
            ngram = preloaded_ngram
        else:
            ngram = NGramCache(n=ngram_n)
        # Pre-populate from prior tokens if supplied (multi-turn / warmed
        # session), then from the current prompt.  This is *pure cache
        # data* — it does NOT lengthen the model's KV cache or affect AR
        # baseline timing.
        if cache_warmup_tokens:
            ngram.update_sequence(list(cache_warmup_tokens))
        ngram.update_sequence(list(prompt_ids))
    retriever: Optional[SuffixRetriever] = None
    if use_retrieval:
        if preloaded_retriever is not None:
            retriever = preloaded_retriever
        else:
            retriever = SuffixRetriever(max_suffix=retrieval_max_suffix)
        if cache_warmup_tokens:
            retriever.extend(list(cache_warmup_tokens))
        retriever.extend(list(prompt_ids))
        retriever.extend([first_token])

    # COT-dataset cache (optional).  When supplied, the phase-aware n-gram
    # cache feeds the ``cot_ngram`` slot and the bucket-specific retriever
    # feeds the ``cot_retriever`` slot in build_hybrid_branches — both pre-
    # warmed from the training corpus so we get useful drafts from round 0.
    cot_tracker: Optional[CoTPhaseTracker] = None
    if cot_caches is not None:
        bundle = load_cot_caches(cot_caches)
        think_cache, final_cache, cot_retr = bundle.get_caches(cot_bucket)
        # Retriever is opt-in: it fires only when the caller routes the
        # prompt to a bucket explicitly.  Without ``cot_bucket`` we behave
        # like the v1 "cot ngram only" path so the existing draft cost
        # profile is preserved.
        if cot_bucket is None:
            cot_retr = None
        if (think_cache is not None and final_cache is not None
                and think_cache.n != final_cache.n):
            raise ValueError("COT think and final caches must share the same n")
        init_phase = infer_initial_phase(
            list(prompt_ids) + [first_token], bundle.markers,
        )
        cot_tracker = CoTPhaseTracker(
            bundle.markers,
            think_cache, final_cache, cot_retr,
            initial=init_phase,
        )

    prev_token = first_token
    fallback_seed = first_token
    K_cur = K

    V = int(last_logits.shape[-1])
    # Compiled accept kernels are cached per K value — adaptive_K may produce
    # different K_cur values across rounds so we build on demand.
    _compiled_accepts: dict[int, object] = {}

    # Adaptive-K EWMA state (mirrors jacobi_decode).
    arl_ema: float = 0.0
    EMA_ALPHA = 0.3
    BUMP_HI = 0.55
    BUMP_LO = 0.30
    K_STEP = 2

    n_rounds = 0
    n_accepted = 0
    n_draft_hits = 0
    n_full = 0
    K_history: list[int] = []

    t_decode_start = time.perf_counter()

    while len(generated) < gen_config.max_tokens:
        cot_active: Optional[NGramCache] = (
            cot_tracker.active_cache() if cot_tracker is not None else None
        )
        cot_retr_active: Optional[SuffixRetriever] = (
            cot_tracker.active_retriever() if cot_tracker is not None else None
        )

        history = list(prompt_ids) + generated[:-1]
        # Build a single draft chain (best-of-sources).
        if (use_retrieval or cot_tracker is not None
                or (use_ngram and ngram is not None)):
            branches, hits = build_hybrid_branches(
                K_cur, prev_token, history, ngram, retriever, fallback_seed,
                num_branches=1,
                cot_ngram=cot_active,
                cot_retriever=cot_retr_active,
            )
            guesses = branches[0]
        else:
            guesses, hits = _build_guesses(K_cur, prev_token, history, ngram,
                                           fallback_seed)
        n_draft_hits += hits
        K_history.append(K_cur)

        # Verify forward.
        verify_ids = mx.array([[prev_token] + guesses], dtype=mx.int32)
        logits, perpos_payload = _verify_fwd_fn(model, verify_ids, state)
        # logits: (1, K, V) fp32
        mx.eval(logits)

        # ── SJD acceptance, *compiled & batched* ───────────────────────────
        # Compiled accept kernel is cached per K_cur value — look up or build.
        if K_cur not in _compiled_accepts:
            _compiled_accepts[K_cur] = _build_compiled_accept(
                K_cur, V, temperature, top_k, top_p, min_p,
            )
        compiled_accept = _compiled_accepts[K_cur]

        guesses_arr = mx.array(guesses, dtype=mx.int32)         # (K_cur-1,)
        key, sub = mx.random.split(key)
        draft_probs, us, all_probs = compiled_accept(
            logits[0], guesses_arr, sub,
        )

        # Single sync barrier — pull only the two small arrays we need to
        # walk in CPU.  all_probs stays lazy and only materialises if we
        # actually hit the resample/bonus paths below.
        mx.eval(draft_probs, us)
        accept_vec = (us < draft_probs).tolist()

        # ``accept_vec[i]`` = True iff draft at draft-position i (=
        # guesses[i]) is accepted as y_{i+1}.

        # 5. CPU loop: find first rejection.
        first_reject = -1
        for i in range(K_cur - 1):
            if not accept_vec[i]:
                first_reject = i
                break

        emitted: list[int] = []
        emit_flags: list[bool] = []
        accepted_all = (first_reject < 0)

        if accepted_all:
            # Every draft accepted: emit them all + 1 bonus sampled from
            # logits[K_cur-1] (the model's prediction at position K_cur+1).
            for g in guesses:
                emitted.append(int(g))
                emit_flags.append(True)   # draft — guessed correctly
            tok_arr, key = _categorical_from_probs(all_probs[K_cur - 1], key)
            mx.eval(tok_arr)
            emitted.append(int(tok_arr.item()))
            emit_flags.append(False)      # bonus — not a draft guess
            n_full += 1
        else:
            # Drafts before ``first_reject`` accepted; draft at ``first_reject``
            # rejected — resample from the residual of all_probs[first_reject].
            for j in range(first_reject):
                emitted.append(int(guesses[j]))
                emit_flags.append(True)   # draft — guessed correctly
            g_rej = int(guesses[first_reject])
            probs_rej = all_probs[first_reject]
            mass = probs_rej[g_rej]
            resid = probs_rej.at[
                mx.array([g_rej], dtype=mx.int32)
            ].add(-mass)
            total = mx.sum(resid)
            resid = mx.where(
                total > 1e-9,
                resid / mx.maximum(total, 1e-9),
                probs_rej,                                     # safety
            )
            tok_arr, key = _categorical_from_probs(resid, key)
            mx.eval(tok_arr)
            emitted.append(int(tok_arr.item()))
            emit_flags.append(False)      # resample — guess was wrong

        accepted_chain = accepted_all  # alias used by verbose log below
        m = len(emitted)
        # State extraction.
        # Input positions consumed by the verify forward: [prev, g_0, ..,
        # g_{K-2}] (K positions).  After accepting the first n drafts and
        # resampling at position n+1, the running sequence after this round
        # is [prev, g_0, .., g_{n-1}, resample].  State after these tokens
        # should reflect *only* the accepted-prefix portion — n+1 positions
        # in the verify window (positions 0..n, since the resampled token
        # is *not* one of the verify inputs and there's nothing to do for
        # it past picking it).  However, position n in the input is
        # g_{n-1}, which equals the n-th emitted token — so state after
        # ``n+1`` inputs is what we want, capped at K.
        m_state = min(len(emitted), K_cur)

        # ── Emit + stop checks ──
        emit_done = False
        for i, tok in enumerate(emitted):
            generated.append(int(tok))
            if on_emit is not None:
                on_emit(int(tok), emit_flags[i])
            elif on_token is not None:
                on_token(int(tok))
            if int(tok) in stop_set:
                stop_reason = "eos"
                emit_done = True
                break
            if len(generated) >= gen_config.max_tokens:
                stop_reason = "max_tokens"
                emit_done = True
                break
            if stop_strings:
                tail = tokenizer.decode(
                    generated[-max(_stop_str_window, 1):],
                    skip_special_tokens=False,
                )
                if any(s in tail for s in stop_strings):
                    stop_reason = "stop_string"
                    emit_done = True
                    break

        # State extraction (position m_state - 1 in the verify window).
        state = extract_state_at(perpos_payload, m_state, branch=0)
        mx.eval(*_iter_state_arrays(state))

        n_rounds += 1
        n_accepted += m
        prev_token = int(emitted[-1])
        fallback_seed = prev_token

        if retriever is not None:
            retriever.extend(emitted)
        if ngram is not None:
            full = list(prompt_ids) + generated
            start = max(0, len(full) - m - ngram.key_len)
            ngram.update_sequence(full, start_idx=start)
        # Phase tracker advance — flips think→other→final on the marker tokens.
        if cot_tracker is not None:
            cot_tracker.observe(emitted)

        # ── Adaptive K (GammaTune-style EWMA over ARL/K_cur) ────────────────
        if adaptive_K:
            ratio = m / K_cur
            arl_ema = (1 - EMA_ALPHA) * arl_ema + EMA_ALPHA * ratio
            if arl_ema > BUMP_HI and K_cur + K_STEP <= K_max:
                K_cur += K_STEP
            elif arl_ema < BUMP_LO and K_cur - K_STEP >= K_min:
                K_cur -= K_STEP

        if verbose:
            K_tag = f"K={K_cur:2d}*" if adaptive_K else f"K={K_cur:2d}"
            tag = "FULL" if accepted_chain else f"part@{len(emitted) - 1}"
            print(
                f"[sjd] round={n_rounds:4d} m={m:2d}/{K_tag} "
                f"{tag} arl={n_accepted / max(n_rounds, 1):.2f} "
                f"emitted={len(generated):4d}",
                flush=True,
            )
        if emit_done:
            break

    elapsed_decode = time.perf_counter() - t_decode_start
    timed_tokens = max(len(generated) - 1, 0)
    decode_tps = timed_tokens / max(elapsed_decode, 1e-6)

    return JacobiSamplingResult(
        tokens=generated, stop_reason=stop_reason,
        n_prompt=len(prompt_ids),
        elapsed_prefill=elapsed_prefill,
        elapsed_decode=elapsed_decode,
        prefill_tps=prefill_tps, decode_tps=decode_tps,
        n_rounds=n_rounds, n_accepted=n_accepted,
        arl=n_accepted / max(n_rounds, 1),
        n_draft_hits=n_draft_hits, K=K,
        n_full_accepts=n_full, K_history=K_history,
    )


def _empty(generated, stop_reason, prompt_ids, elapsed_prefill, prefill_tps, K):
    return JacobiSamplingResult(
        tokens=generated, stop_reason=stop_reason,
        n_prompt=len(prompt_ids),
        elapsed_prefill=elapsed_prefill, elapsed_decode=1e-6,
        prefill_tps=prefill_tps, decode_tps=0.0,
        n_rounds=0, n_accepted=0, arl=0.0, n_draft_hits=0, K=K,
        n_full_accepts=0, K_history=[],
    )
