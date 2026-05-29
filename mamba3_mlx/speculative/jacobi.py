"""Jacobi (parallel-verify) decoder for Mamba3LanguageModel.

How it works
------------
A single Jacobi *round* feeds K tokens through the model in one forward
pass — the most recently confirmed token at position 0 and ``K-1``
*guesses* at positions 1..K-1.  Greedy argmax of the logits at each
position gives the model's own prediction for the *next* token.

We then walk left-to-right:
    accepted[0]     = argmax(logits[0])   -- always accepted
    accepted[i+1]   = argmax(logits[i+1]) if accepted[i] == guesses[i]

The accepted prefix has length ``m ∈ [1, K]``.  ``m`` tokens are
emitted; on partial accept the model is re-run with the m-token
prefix to obtain the correct state (one batched call instead of m
serial steps), and the carry/N-gram seeds for the next round are
derived from ``accepted[-1]``.

Correctness
-----------
With greedy decoding, every token Jacobi emits is the same token a
pure autoregressive greedy loop would have produced **given identical
numerical paths**.  The verify path uses ``Mamba3Block``'s
``chunk_parallel_scan`` (L > 1) whereas AR uses the single-step path
(L = 1); in fp32 these are mathematically equivalent and the streams
match bit-for-bit.  In bf16 the two paths agree to within rounding
noise and the streams may diverge once the cumulative noise exceeds
the logit margin (the verify harness reports this).

This module imports from existing mamba3_mlx files but does **not**
modify any of them.
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
from .drafts import SuffixRetriever, build_hybrid_branches
from .forward import extract_state_at, model_verify_forward, replicate_state
from .forward_metal import model_verify_forward_metal
from .ngram_cache import NGramCache


# ── Result container ──────────────────────────────────────────────────────────

class JacobiResult(NamedTuple):
    tokens: list[int]
    stop_reason: str        # "max_tokens" | "eos" | "stop_string"
    n_prompt: int
    elapsed_prefill: float
    elapsed_decode: float
    prefill_tps: float
    decode_tps: float
    n_rounds: int           # Jacobi rounds executed
    n_accepted: int         # tokens emitted from verify rounds (excludes first_token)
    arl: float              # n_accepted / n_rounds (average run length)
    n_ngram_hits: int       # guesses that came from any draft source
    K: int
    tree_B: int             # number of branches per round (1 = single-branch)
    branch_wins: list[int]  # branch_wins[b] = times branch b had the longest accept
    K_history: list[int]    # K used at each round (for adaptive-K diagnostics)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_tree_guesses(
    K: int,
    prev_token: int,
    history_tail: list[int],
    ngram: Optional[NGramCache],
    fallback_seed: int,
    num_branches: int,
) -> tuple[list[list[int]], int]:
    """Build ``num_branches`` independent guess sequences (length K-1 each).

    Branch ``b`` uses the **b-th MRU** n-gram continuation as the seed at
    position 1 (the only position where branches differ); positions 2..K-1
    use the top-1 continuation chain on top of that seed.  When there aren't
    ``num_branches`` distinct continuations available we backfill with the
    carry seed and (if needed) pad with duplicates of the top candidate so
    the batched verify always sees ``num_branches`` rows.
    """
    if num_branches < 1:
        raise ValueError("num_branches must be >= 1")
    if num_branches == 1 or ngram is None or ngram.key_len <= 0:
        branch, hits = _build_guesses(
            K, prev_token, history_tail, ngram, fallback_seed,
        )
        return [branch] * num_branches, hits

    L = ngram.key_len
    ctx_base: list[int] = list(history_tail[-L:])
    ctx_base.append(int(prev_token))
    if len(ctx_base) > L:
        ctx_base = ctx_base[-L:]

    # Top-N candidates for position 1 (= guesses[0] of each branch).
    top = ngram.query_topk(tuple(ctx_base), num_branches) if len(ctx_base) == L else []
    # Pad to exactly num_branches distinct seeds (fallback if cache short).
    seen: set[int] = set()
    pos0_seeds: list[int] = []
    for t in top:
        if t not in seen:
            seen.add(t)
            pos0_seeds.append(int(t))
        if len(pos0_seeds) >= num_branches:
            break
    while len(pos0_seeds) < num_branches:
        # Backfill with carry seed (or just duplicate top-1 if no carry).
        pos0_seeds.append(int(fallback_seed) if fallback_seed not in seen else int(top[0]) if top else int(fallback_seed))
        seen.add(pos0_seeds[-1])

    branches: list[list[int]] = []
    hits_total = 0
    for b in range(num_branches):
        guesses = [pos0_seeds[b]]
        if pos0_seeds[b] in (top or []):
            hits_total += 1
        ctx = list(ctx_base) + [pos0_seeds[b]]
        if len(ctx) > L:
            ctx = ctx[-L:]
        for _ in range(K - 2):
            tok: Optional[int] = None
            if len(ctx) == L:
                tok = ngram.query(tuple(ctx))
            if tok is None:
                tok = int(fallback_seed)
            else:
                hits_total += 1
            guesses.append(int(tok))
            ctx.append(int(tok))
            if len(ctx) > L:
                ctx = ctx[-L:]
        branches.append(guesses)
    return branches, hits_total


def _build_guesses(
    K: int,
    prev_token: int,
    history_tail: list[int],
    ngram: Optional[NGramCache],
    fallback_seed: int,
) -> tuple[list[int], int]:
    """Fill the K-1 guess positions.

    Priority per position:
      1. n-gram(last n-1 tokens) MRU continuation
      2. carry/fallback seed (``fallback_seed``)

    Returns (guesses, n_hits).
    """
    n_minus_1 = K - 1
    if ngram is None or ngram.key_len <= 0:
        return [int(fallback_seed)] * n_minus_1, 0

    L = ngram.key_len
    # Working context = [history_tail ... prev_token (guesses appended)] truncated to L.
    ctx: list[int] = list(history_tail[-L:])
    ctx.append(int(prev_token))
    if len(ctx) > L:
        ctx = ctx[-L:]

    guesses: list[int] = []
    hits = 0
    fb = int(fallback_seed)
    for _ in range(n_minus_1):
        tok: Optional[int] = None
        if len(ctx) == L:
            tok = ngram.query(tuple(ctx))
        if tok is None:
            tok = fb
        else:
            hits += 1
        guesses.append(int(tok))
        ctx.append(int(tok))
        if len(ctx) > L:
            ctx = ctx[-L:]
    return guesses, hits


# ── Public API ────────────────────────────────────────────────────────────────

def jacobi_decode(
    model,
    prompt_ids: list[int],
    gen_config,
    *,
    K: int = 8,
    use_ngram: bool = True,
    ngram_n: int = 3,
    tree_B: int = 1,
    use_retrieval: bool = False,
    retrieval_max_suffix: int = 8,
    adaptive_K: bool = False,
    K_min: int = 4,
    K_max: int = 16,
    cot_caches: CoTCachesArg = None,
    cot_bucket: Optional[str] = None,
    metal_verify: bool = False,
    compile_verify: bool = False,
    preloaded_ngram: Optional[NGramCache] = None,
    preloaded_retriever: Optional[SuffixRetriever] = None,
    cache_warmup_tokens: Optional[list[int]] = None,
    stop_token_ids: Optional[list[int]] = None,
    stop_strings: Optional[list[str]] = None,
    no_eos_stop: bool = False,
    tokenizer=None,
    on_token: Optional[Callable[[int], None]] = None,
    verbose: bool = False,
    compile_graphs: bool = True,
    warmup_rounds: int = 1,
) -> JacobiResult:
    """Greedy Jacobi decoder.

    Sampling penalties / temperature from ``gen_config`` are NOT applied —
    Jacobi acceptance requires deterministic argmax so the verifier is
    consistent with the per-step model.  ``gen_config.max_tokens`` and the
    stop conditions (``stop_token_ids``, ``stop_strings``, ``no_eos_stop``)
    are honored.

    Parameters
    ----------
    model : Mamba3LanguageModel
        Model whose ``__call__`` accepts ``(input_ids[B,L], states=...)``.
    prompt_ids : list[int]
        Pre-tokenized prompt.
    gen_config : GenerationConfig
        Only ``max_tokens`` is honored for now.
    K : int
        Jacobi window (≥ 2).  K-1 guesses are issued per round.
    use_ngram : bool
        Enable n-gram seeding of guess positions 1..K-1.
    ngram_n : int
        Total n-gram length; key length is ``ngram_n - 1``.

    Returns
    -------
    JacobiResult
    """
    if K < 2:
        raise ValueError(f"Jacobi K must be >= 2, got {K}")
    if tree_B < 1:
        raise ValueError(f"tree_B must be >= 1, got {tree_B}")
    if adaptive_K and not (K_min <= K <= K_max):
        raise ValueError(
            f"adaptive_K requires K_min <= K <= K_max "
            f"(got K_min={K_min}, K={K}, K_max={K_max})"
        )

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

    # ── Prefill ──────────────────────────────────────────────────────────────
    last_logits, state, elapsed_prefill = prefill(model, prompt_ids)
    prefill_tps = len(prompt_ids) / max(elapsed_prefill, 1e-6)

    # The verify path uses ``model_verify_forward`` from ``.forward``, which
    # records per-position state — this lets us skip the replay forward on
    # partial accept by indexing into the per-position payload.

    generated: list[int] = []
    stop_reason: str = "max_tokens"

    # First decoded token: greedy from prefill logits
    first_tok_arr = mx.argmax(last_logits).astype(mx.int32)
    mx.eval(first_tok_arr)
    first_token = int(first_tok_arr.item())
    generated.append(first_token)
    if on_token is not None:
        on_token(first_token)
    if first_token in stop_set:
        return JacobiResult(
            tokens=generated, stop_reason="eos",
            n_prompt=len(prompt_ids),
            elapsed_prefill=elapsed_prefill, elapsed_decode=1e-6,
            prefill_tps=prefill_tps, decode_tps=0.0,
            n_rounds=0, n_accepted=0, arl=0.0, n_ngram_hits=0, K=K,
            tree_B=tree_B, branch_wins=[0] * tree_B, K_history=[],
        )
    if len(generated) >= gen_config.max_tokens:
        return JacobiResult(
            tokens=generated, stop_reason="max_tokens",
            n_prompt=len(prompt_ids),
            elapsed_prefill=elapsed_prefill, elapsed_decode=1e-6,
            prefill_tps=prefill_tps, decode_tps=0.0,
            n_rounds=0, n_accepted=0, arl=0.0, n_ngram_hits=0, K=K,
            tree_B=tree_B, branch_wins=[0] * tree_B, K_history=[],
        )

    # N-gram cache (optional); pre-warmed if ``preloaded_ngram`` was supplied,
    # otherwise built from the prompt.  Either way ``cache_warmup_tokens`` are
    # ingested first (multi-turn session warm-up).
    ngram: Optional[NGramCache] = None
    if use_ngram:
        if preloaded_ngram is not None:
            ngram = preloaded_ngram
        else:
            ngram = NGramCache(n=ngram_n)
        if cache_warmup_tokens:
            ngram.update_sequence(list(cache_warmup_tokens))
        ngram.update_sequence(list(prompt_ids))

    # Suffix retriever (optional); same pre-warm + extend pattern.
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
    K_cur = K                                # current K under adaptive-K policy

    n_rounds = 0
    n_accepted = 0
    n_ngram_hits = 0
    branch_wins: list[int] = [0] * tree_B
    K_history: list[int] = []

    # Adaptive-K state: EWMA over ARL/K ratio.
    arl_ema: float = 0.0
    EMA_ALPHA = 0.3
    BUMP_HI = 0.55          # if ARL/K_cur > this for a while, grow K_cur
    BUMP_LO = 0.30          # if ARL/K_cur < this for a while, shrink K_cur
    K_STEP = 2

    t_decode_start = time.perf_counter()

    while len(generated) < gen_config.max_tokens:
        cot_active: Optional[NGramCache] = (
            cot_tracker.active_cache() if cot_tracker is not None else None
        )
        cot_retr_active: Optional[SuffixRetriever] = (
            cot_tracker.active_retriever() if cot_tracker is not None else None
        )

        # ── 1. Construct guess buffer(s) ─────────────────────────────────────
        history = list(prompt_ids) + generated[:-1]  # everything before prev_token

        # Use hybrid (multi-source) builder when retrieval/COT cache is
        # enabled or when tree_B>1 with at least one extra source; otherwise
        # the simple single-source builders.
        use_hybrid = (use_retrieval or cot_tracker is not None
                      or (tree_B > 1 and (use_ngram or use_retrieval)))
        if use_hybrid:
            branches, hits = build_hybrid_branches(
                K_cur, prev_token, history,
                ngram, retriever, fallback_seed, tree_B,
                cot_ngram=cot_active,
                cot_retriever=cot_retr_active,
            )
        elif tree_B == 1:
            branch_guesses, hits = _build_guesses(
                K_cur, prev_token, history, ngram, fallback_seed,
            )
            branches = [branch_guesses]
        else:
            branches, hits = _build_tree_guesses(
                K_cur, prev_token, history, ngram, fallback_seed, tree_B,
            )
        n_ngram_hits += hits
        K_history.append(K_cur)

        # ── 2. Verify forward (batch = tree_B, length = K_cur) ───────────────
        verify_ids = mx.array(
            [[prev_token] + br for br in branches],   # shape (tree_B, K_cur)
            dtype=mx.int32,
        )
        batched_state = replicate_state(state, tree_B)
        logits, perpos_payload = _verify_fwd_fn(model, verify_ids, batched_state)
        preds = mx.argmax(logits, axis=-1).astype(mx.int32)   # (tree_B, K)
        mx.eval(preds)
        preds_per_branch = [list(row) for row in preds.tolist()]

        # ── 3. Greedy Jacobi accept loop — pick longest-accepting branch ────
        best_branch = 0
        best_accepted: list[int] = [int(preds_per_branch[0][0])]
        for b in range(tree_B):
            pred_list = preds_per_branch[b]
            guesses_b = branches[b]
            acc_b: list[int] = [int(pred_list[0])]
            for i in range(K_cur - 1):
                if int(pred_list[i]) == int(guesses_b[i]):
                    acc_b.append(int(pred_list[i + 1]))
                else:
                    break
            if len(acc_b) > len(best_accepted):
                best_accepted = acc_b
                best_branch = b
        accepted = best_accepted
        m = len(accepted)
        branch_wins[best_branch] += 1

        # ── 4. Emit accepted tokens (with stop checks) ──────────────────────
        emit_done = False
        for tok in accepted:
            generated.append(int(tok))
            if on_token is not None:
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
                tail_ids = generated[-max(_stop_str_window, 1):]
                tail = tokenizer.decode(tail_ids, skip_special_tokens=False)
                if any(s in tail for s in stop_strings):
                    stop_reason = "stop_string"
                    emit_done = True
                    break

        # ── 5. Update model state — pluck (best_branch, m-1) from per-pos ──
        # No replay forward!  Per-position payload was recorded by verify;
        # we slice both the winning branch and the m-th position out of it.
        state = extract_state_at(perpos_payload, m, branch=best_branch)
        mx.eval(*_iter_state_arrays(state))

        n_rounds += 1
        n_accepted += m
        # The next round will feed accepted[-1] (the most recent emitted token,
        # which has NOT been fed to the model yet).
        prev_token = int(accepted[-1])
        fallback_seed = prev_token

        # ── 6a. Update suffix retriever with newly accepted tokens ──────────
        if retriever is not None:
            # The first accepted token of this round was already known to the
            # retriever (it was the prev_token of THIS round, appended last
            # round).  So we append accepted[0..m-1] minus the one we already
            # added? No — first_token / prev_token after prefill was the
            # only token added before the first round; from round 1 onwards
            # ``prev_token`` is set from the previous round's accepted[-1],
            # which we already appended.  Append the *full* accepted list.
            retriever.extend(accepted)

        # ── 6. Update n-gram cache with newly accepted tokens ───────────────
        if ngram is not None:
            full = list(prompt_ids) + generated
            # Re-insert n-grams whose suffix includes any of the m new tokens.
            start = max(0, len(full) - m - ngram.key_len)
            ngram.update_sequence(full, start_idx=start)

        # ── 6b. Advance the COT phase tracker so the next round queries the
        # right cache (think → final transition fires on </think>).
        if cot_tracker is not None:
            cot_tracker.observe(accepted)

        # ── 7. Adaptive K (GammaTune-style, EWMA over ARL/K_cur) ────────────
        if adaptive_K:
            ratio = m / K_cur
            arl_ema = (1 - EMA_ALPHA) * arl_ema + EMA_ALPHA * ratio
            if arl_ema > BUMP_HI and K_cur + K_STEP <= K_max:
                K_cur += K_STEP
            elif arl_ema < BUMP_LO and K_cur - K_STEP >= K_min:
                K_cur -= K_STEP

        if verbose:
            tag = f"b{best_branch}/{tree_B}" if tree_B > 1 else "b0"
            K_tag = (f"K={K_cur:2d}*" if adaptive_K else f"K={K_cur:2d}")
            print(
                f"[jacobi] round={n_rounds:4d}  m={m:2d}/{K_tag}  "
                f"win={tag}  next_in={prev_token:6d}  "
                f"emitted={len(generated):4d}  "
                f"arl={n_accepted / max(n_rounds, 1):.2f}",
                flush=True,
            )

        if emit_done:
            break

    elapsed_decode = time.perf_counter() - t_decode_start
    # decode tps excludes the first_token (which came from prefill logits).
    timed_tokens = max(len(generated) - 1, 0)
    decode_tps = timed_tokens / max(elapsed_decode, 1e-6)

    return JacobiResult(
        tokens=generated,
        stop_reason=stop_reason,
        n_prompt=len(prompt_ids),
        elapsed_prefill=elapsed_prefill,
        elapsed_decode=elapsed_decode,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        n_rounds=n_rounds,
        n_accepted=n_accepted,
        arl=n_accepted / max(n_rounds, 1),
        n_ngram_hits=n_ngram_hits,
        K=K,
        tree_B=tree_B,
        branch_wins=branch_wins,
        K_history=K_history,
    )
