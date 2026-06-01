#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jacobi decoding with training-free enhancements for MLX Mamba inference.

v4 additions over v3 (2.82× peak):
  P0 — Tree Attention + Multi-Candidate Parallel Verification
    - NGramCache.query_topk(): return top-k distinct next-token candidates
    - build_guess_tree(): build B independent draft branches of depth D each
    - _find_best_tree_branch(): verify B branches sequentially, pick longest match
    - use_tree=True enables tree mode; B and D are configurable
    - Branches verified with the SAME original caches (no mutation between branches)
    - Full-accept path reuses verify caches directly; partial-accept uses _batch_replay

  P1 — Entropy-Based Adaptive K (SJD-PAC inspired)
    - _entropy(): Shannon entropy of current token distribution (low = high confidence)
    - use_entropy_k=True adapts K and branch count each round based on entropy
    - Low entropy → K_max, B=1 (aggressive single-path, maximize ARL in easy segments)
    - Medium entropy → K_mid, B=2 (balanced)
    - High entropy → K_min, B=1 (conservative, avoid ARL collapse)
    - Prevents the "wasted K" problem where large K fails in diverse/uncertain segments

v3 improvements (kept):
  Multi-order NGramCache (n=4→3→2→1), prompt pre-warming, merged Lookahead,
  vectorized comparison, dynamic K via ARL EMA.

Architecture (unchanged):
  - Per-layer _compiled_verify[K] kernels via attach_verify_compilation()
  - build_compiled_verify_fns() for outer mx.compile per K
  - Dynamic K via ARL EMA (use_dynamic_K) or entropy (use_entropy_k)
  - chunk_parallel_scan_with_init: O(1) Mamba scan for K>1 verify

Integration:
    from jacobi_enhanced import enhanced_jacobi_stream, build_compiled_verify_fns
    from mlx_hybrid_infer import attach_verify_compilation
    attach_verify_compilation(model, K_values=[8,16,32], max_cache_len=max_cache_len)
    compiled_verify_fns = build_compiled_verify_fns(model, router_temp,
        K_values=(8,16,32), caches_for_warmup=caches)
    token_iter = enhanced_jacobi_stream(model=model, ...,
        compiled_verify_fns=compiled_verify_fns,
        use_tree=True, num_tree_branches=3,
        use_entropy_k=True, entropy_low=0.5, entropy_high=1.5)
"""
from __future__ import annotations

import sys
import numpy as _np
from collections import Counter
from typing import Any, Iterator


def _deep_copy_caches(node: Any) -> Any:
    """Return a truly independent copy of a cache tree via numpy round-trip.

    MLX's `a + zeros(())` is optimized to identity, so materialize_cache_tree_fn
    can alias the original GPU buffers — any in-place mx.slice_update (KV) or
    cumulative SSM state mutation from the verify call will corrupt the snapshot.
    numpy round-trip forces a new allocation disconnected from the compute graph.

    Optimized: collect all leaf arrays → ONE mx.eval (one GPU sync) → then
    numpy-copy in Python (CPU memcpy only, no further GPU sync per array).
    """
    import mlx.core as _mx

    # --- Step 1: collect all leaf mx.arrays in tree order ---
    leaves: list = []

    def _collect(n: Any) -> None:
        if isinstance(n, _mx.array):
            leaves.append(n)
        elif isinstance(n, (list, tuple)):
            for x in n:
                _collect(x)

    _collect(node)

    if not leaves:
        return node  # no arrays to copy

    # --- Step 2: ONE batch GPU sync for all leaves ---
    _mx.eval(*leaves)

    # --- Step 3: numpy-copy each leaf (CPU only, no GPU sync per array) ---
    copies: list[_mx.array] = []
    for arr in leaves:
        orig_dtype = arr.dtype
        if orig_dtype == _mx.bfloat16:
            np_arr = _np.array(arr.astype(_mx.float32), copy=True)
            copies.append(_mx.array(np_arr).astype(orig_dtype))
        else:
            copies.append(_mx.array(_np.array(arr, copy=True)))

    # --- Step 4: reconstruct tree with copied arrays ---
    _it = iter(copies)

    def _rebuild(n: Any) -> Any:
        if isinstance(n, _mx.array):
            return next(_it)
        if isinstance(n, list):
            return [_rebuild(x) for x in n]
        if isinstance(n, tuple):
            return tuple(_rebuild(x) for x in n)
        return n

    return _rebuild(node)

# ---------------------------------------------------------------------------
# Multi-order N-gram Cache
# ---------------------------------------------------------------------------

class NGramCache:
    """
    Multi-order N-gram with adaptive fallback (n → min_n).

    Stores n-grams for every order in [min_n, n]. query() tries the longest
    context first and falls back to shorter orders on cache miss.
    query_topk() returns the k most-frequent next tokens (for tree attention).

    update() registers all orders in one pass with per-order LRU eviction.
    Pre-warm from prompt tokens before generation: ngram_cache.update(prompt_ids).
    """

    def __init__(self, n: int = 4, min_n: int = 1, max_size: int = 15_000) -> None:
        self.n = n
        self.min_n = min_n
        self.max_size = max_size
        n_orders = max(n - min_n + 1, 1)
        per_order = max(max_size // n_orders, 1)
        self._caches: dict[int, dict[tuple[int, ...], list[int]]] = {
            o: {} for o in range(min_n, n + 1)
        }
        self._fifos: dict[int, list[tuple[int, ...]]] = {
            o: [] for o in range(min_n, n + 1)
        }
        self._cap: dict[int, int] = {o: per_order for o in range(min_n, n + 1)}

    @property
    def cache(self) -> dict:
        """Backward compat: expose highest-order cache as .cache."""
        return self._caches[self.n]

    def update(self, tokens: list[int]) -> None:
        """Register all n-gram sub-sequences (all orders) found in tokens."""
        for order in range(self.min_n, self.n + 1):
            c = self._caches[order]
            fifo = self._fifos[order]
            cap = self._cap[order]
            for i in range(len(tokens) - order):
                key = tuple(tokens[i: i + order])
                nxt = tokens[i + order]
                if key not in c:
                    if len(c) >= cap:
                        evict = fifo.pop(0)
                        c.pop(evict, None)
                    c[key] = []
                    fifo.append(key)
                c[key].append(nxt)

    def query(self, context: list[int]) -> int | None:
        """Try longest-to-shortest n-gram; return most-frequent next token or None."""
        for order in range(min(len(context), self.n), self.min_n - 1, -1):
            key = tuple(context[-order:])
            vals = self._caches[order].get(key)
            if vals:
                return max(set(vals), key=vals.count)
        return None

    def query_topk(self, context: list[int], k: int = 3) -> list[int]:
        """
        Try longest-to-shortest n-gram; return up to k most-frequent next tokens.

        Used by build_guess_tree() to generate diverse alternative branches.
        Returns fewer than k tokens if the cache has fewer distinct candidates.
        """
        for order in range(min(len(context), self.n), self.min_n - 1, -1):
            key = tuple(context[-order:])
            vals = self._caches[order].get(key)
            if vals:
                freq = Counter(vals)
                return [tok for tok, _ in freq.most_common(k)]
        return []


# ---------------------------------------------------------------------------
# Lookahead Cache (unchanged from v3)
# ---------------------------------------------------------------------------

class LookaheadCache:
    """
    Stores multi-step continuations observed during generation.

    query_draft() finds the longest suffix of context that exists in the
    cache, then greedily extends it up to max_draft tokens.  In v3, the
    result is used as a GUESS BUFFER PREFIX rather than a separate verify
    draft: it is verified together with the regular Jacobi K-token call,
    adding zero extra cost when a hit is found.
    """

    def __init__(self, max_len: int = 8, max_size: int = 5_000) -> None:
        self.max_len = max_len
        self.cache: dict[tuple[int, ...], list[int]] = {}
        self.max_size = max_size
        self._keys_fifo: list[tuple[int, ...]] = []

    def update(self, tokens: list[int]) -> None:
        """Register all n-gram keys (length 1..max_len) present in tokens."""
        for start in range(len(tokens)):
            for l in range(1, self.max_len + 1):
                if start + l >= len(tokens):
                    break
                key = tuple(tokens[start: start + l])
                nxt = tokens[start + l]
                if key not in self.cache:
                    if len(self.cache) >= self.max_size:
                        evict = self._keys_fifo.pop(0)
                        self.cache.pop(evict, None)
                    self.cache[key] = []
                    self._keys_fifo.append(key)
                self.cache[key].append(nxt)

    def query_draft(self, context: list[int], max_draft: int = 8) -> list[int] | None:
        """
        Try from longest to shortest suffix match; greedily extend on hit.
        Returns a list of draft token ids or None when no match.
        """
        for l in range(min(len(context), self.max_len), 0, -1):
            key = tuple(context[-l:])
            vals = self.cache.get(key)
            if not vals:
                continue
            seed = max(set(vals), key=vals.count)
            draft = [seed]
            ext_ctx = list(context) + [seed]
            for _ in range(max_draft - 1):
                found = False
                for el in range(min(len(ext_ctx), self.max_len), 0, -1):
                    ev = self.cache.get(tuple(ext_ctx[-el:]))
                    if ev:
                        nxt = max(set(ev), key=ev.count)
                        draft.append(nxt)
                        ext_ctx.append(nxt)
                        found = True
                        break
                if not found:
                    break
            return draft
        return None


# ---------------------------------------------------------------------------
# N-gram + Lookahead seeded guess initialization (linear mode)
# ---------------------------------------------------------------------------

def _ngram_init_guesses(
    prefix: list[int],
    K: int,
    ngram_cache: NGramCache,
    carry_draft: list[int] | None = None,
    la_prefix: list[int] | None = None,
    ban_ids: frozenset[int] = frozenset({0}),
) -> list[int]:
    """
    Build K initial guesses with priority chain:
      1. Lookahead prefix (la_prefix[i])   — positions 0..la_len-1
      2. Carry-over (carry_draft[i])        — model's own predictions from prev round
      3. Multi-order N-gram lookup          — historical token co-occurrences
      4. Repeat-last-token                  — final fallback (skips banned IDs)
    """
    guesses: list[int] = []
    context = list(prefix)
    la_len = len(la_prefix) if la_prefix else 0
    # Last non-banned context token for safe fallback
    safe_fallback = next((t for t in reversed(context) if t not in ban_ids), 1)
    for i in range(K):
        tok: int | None = None
        if i < la_len:
            tok = la_prefix[i]
            if tok in ban_ids:
                tok = None
        elif carry_draft is not None and i < len(carry_draft):
            tok = carry_draft[i]
            if tok in ban_ids:
                tok = None
        if tok is None:
            tok = ngram_cache.query(context)
            if tok is not None and tok in ban_ids:
                tok = None
        if tok is None:
            tok = safe_fallback
        guesses.append(tok)
        context.append(tok)
        if tok not in ban_ids:
            safe_fallback = tok
    return guesses


# ---------------------------------------------------------------------------
# P0: Tree Attention helpers
# ---------------------------------------------------------------------------

def build_guess_tree(
    carry_token: int,
    ngram_cache: NGramCache,
    prefix: list[int],
    branch_depth: int,
    num_branches: int = 3,
) -> list[list[int]]:
    """
    Build num_branches independent draft sequences, each of length branch_depth.

    Branch 0 starts with carry_token (model's argmax prediction at the current
    position — guaranteed to pass depth-0 verification).  Branches 1+ start with
    the top-(i+1) N-gram candidate that differs from carry_token, then extend
    greedily with N-gram from depth 1 onward.

    When fewer than num_branches distinct candidates exist, later branches
    duplicate carry_token as their first token (they will score identically to
    branch 0 at depth 0 and diverge at depth 1+ via N-gram extensions).
    """
    # Get top-(num_branches+1) N-gram candidates for first position
    topk = ngram_cache.query_topk(prefix, k=num_branches + 1)

    # Branch 0 always uses carry_token; subsequent branches use N-gram alternatives
    first_tokens: list[int] = [carry_token]
    for t in topk:
        if t != carry_token and len(first_tokens) < num_branches:
            first_tokens.append(t)
    # Pad with carry_token if not enough distinct N-gram candidates
    while len(first_tokens) < num_branches:
        first_tokens.append(carry_token)

    branches: list[list[int]] = []
    for first in first_tokens:
        branch = [first]
        ctx = list(prefix) + [first]
        for _ in range(1, branch_depth):
            nxt = ngram_cache.query(ctx)
            if nxt is None:
                nxt = ctx[-1]
            branch.append(nxt)
            ctx.append(nxt)
        branches.append(branch)

    return branches


def _entropy(logit_row: Any) -> float:
    """
    Shannon entropy of the token probability distribution.

    Low value → model is confident (peaked distribution).
    High value → model is uncertain (flat distribution).

    Cast to float32 before computation: bf16 cannot represent 1e-9 (underflows to 0),
    which causes log(0) = -inf → entropy = NaN.  float32 handles the epsilon correctly.
    Returns 0.0 on NaN/inf to fail-safe to K_max (high-confidence assumption).
    """
    import mlx.core as mx
    import math
    p = mx.softmax(logit_row.astype(mx.float32), axis=-1)
    ent = -mx.sum(p * mx.log(p + 1e-7))
    mx.eval(ent)
    val = float(ent.item())
    return val if math.isfinite(val) else 0.0


def _find_best_tree_branch(
    tree_tokens: list[list[int]],
    logits_branches: list[Any],
    current_logit_row: Any,
) -> tuple[int, int, int]:
    """
    Evaluate all verified branches and return the best one.

    Branch 0 always has carry_token (= argmax of current_logit_row) as its
    first token, so it ALWAYS passes depth-0 verification (accepted >= 1).
    Branches 1+ may fail at depth 0 if their first token ≠ carry_token.

    Returns:
        (best_branch_idx, accepted_len, correction_token)
        where correction_token is the model's next prediction after accepted_len tokens.
        For full-accept (accepted_len == D), correction_token is the next-step seed.
    """
    import mlx.core as mx

    D = len(tree_tokens[0])
    pred_at_0 = int(mx.argmax(current_logit_row, axis=-1).item())

    best_branch = 0
    best_accepted = 0
    best_correction = pred_at_0  # fallback: accepted=0, correction=pred_at_0

    for b, (branch, logits_b) in enumerate(zip(tree_tokens, logits_branches)):
        # Depth 0: does model predict branch[0]?
        if pred_at_0 != branch[0]:
            # This branch fails at depth 0 (accepted=0), cannot beat branch 0
            continue

        accepted = 1
        correction: int | None = None
        for d in range(1, D):
            # logits_b shape: [1, D, V]; logits_b[0, d-1, :] predicts token at depth d
            pred_d = int(mx.argmax(logits_b[0, d - 1, :], axis=-1).item())
            if pred_d != branch[d]:
                correction = pred_d
                break
            accepted += 1

        if correction is None:
            # Full branch accept: correction = model's next seed after D tokens
            correction = int(mx.argmax(logits_b[0, D - 1, :], axis=-1).item())

        if accepted > best_accepted:
            best_accepted = accepted
            best_branch = b
            best_correction = correction

    return best_branch, best_accepted, best_correction


# ---------------------------------------------------------------------------
# P1: Entropy-based adaptive K selection
# ---------------------------------------------------------------------------

def _entropy_decide_K_B(
    ent: float,
    entropy_low: float,
    entropy_high: float,
    K_min: int,
    K_max: int,
    K_values: tuple[int, ...],
    num_tree_branches: int,
    use_tree: bool,
) -> tuple[int, int]:
    """
    Return (K_this, effective_branches) based on current entropy.

    Low entropy (model confident) → large K, 1 branch (fast linear)
    Medium entropy → medium K, 2 branches (balanced)
    High entropy (model uncertain) → small K, 1 branch (conservative)

    Tree branches are only enabled in the medium-entropy zone.
    """
    if ent < entropy_low:
        # High confidence: push as many tokens as possible
        K_this = K_max
        B_eff = 1
    elif ent < entropy_high:
        # Medium confidence: balance K and branching
        K_mid_raw = (K_max + K_min) // 2
        valid = sorted(v for v in K_values if K_min <= v <= K_max)
        K_this = min(valid, key=lambda v: abs(v - K_mid_raw)) if valid else K_mid_raw
        B_eff = min(2, num_tree_branches) if use_tree else 1
    else:
        # Low confidence: small K, no tree (avoid wasted compute)
        K_this = K_min
        B_eff = 1

    return K_this, B_eff


# ---------------------------------------------------------------------------
# Enhanced Jacobi stream (v4)
# ---------------------------------------------------------------------------

def enhanced_jacobi_stream(
    *,
    model: Any,
    run_prefill: Any,
    x_prefill: Any,
    prompt_ids: list[int],
    max_cache_len: int,
    router_temp: Any,
    max_new_tokens: int,
    sample_args: Any,
    stop_token_ids: frozenset[int] | None = None,
    prefill_outputs: tuple[Any, Any] | None = None,
    # Jacobi window
    K: int = 32,
    K_min: int = 4,
    K_max: int = 32,
    K_values: tuple[int, ...] = (4, 8, 16, 32),
    # Feature flags
    use_ngram: bool = True,
    use_lookahead: bool = True,
    use_dynamic_K: bool = False,
    # P0: Tree Attention
    use_tree: bool = False,
    num_tree_branches: int = 3,
    tree_branch_depth: int | None = None,   # None → K // num_tree_branches
    tree_early_stop: bool = True,            # skip remaining branches if branch 0 accepts >= threshold
    tree_early_stop_depth: int = 16,         # min accepted tokens to trigger early stop
    # P1: Entropy-based adaptive K
    use_entropy_k: bool = False,
    entropy_low: float = 0.5,               # below: high-confidence zone, use K_max
    entropy_high: float = 1.5,              # above: uncertain zone, use K_min
    # N-gram settings
    ngram_n: int = 4,
    min_ngram_n: int = 1,
    ngram_max_size: int = 15_000,
    warm_prompt: bool = True,
    # Lookahead settings
    lookahead_max_len: int = 8,
    lookahead_max_size: int = 5_000,
    min_lookahead_hit: int = 4,
    # Forced single-step fallback when ARL stays near 0
    max_stall_rounds: int = 3,
    # Autoregressive warmup before Jacobi: first N tokens generated greedily (1-by-1)
    # so the N-gram cache has real history before speculative guessing begins.
    ar_warmup_tokens: int = 20,
    # Token IDs to never generate (e.g. <unk>=0 which floods carry_draft when quantized)
    ban_token_ids: frozenset[int] = frozenset({0}),
    # Stats dict (written in-place)
    stats: dict | None = None,
    # Pre-built compiled functions dict: {K: compiled_model_forward}
    compiled_verify_fns: dict | None = None,
    # Optional: {K: compiled forward_with_mamba_intra} for batch_replay-free partial-accept
    compiled_verify_with_intra_fns: dict | None = None,
    # Shared compiled single-step fn
    compiled_single_fn: Any = None,
    # Legacy compat aliases
    compiled_verify_fn: Any = None,
    # Helpers
    materialize_cache_tree_fn: Any = None,
    pad_transformer_caches_fn: Any = None,
) -> Iterator[Any]:
    """
    Enhanced Jacobi decode stream v4.

    New in v4:
    - P0 (use_tree=True): Multi-candidate tree attention. Builds B independent
      draft branches of depth D each. Verifies each branch sequentially against
      the SAME original caches (no inter-branch mutation). Selects the branch with
      the longest accepted prefix. Falls back to _batch_replay for cache update.
    - P1 (use_entropy_k=True): Entropy-based adaptive K. Computes Shannon entropy
      of the current logit distribution and selects K (and branch count) based on
      model confidence. Prevents wasted compute in uncertain/diverse segments.

    Backcompat: use_tree=False and use_entropy_k=False reproduce v3 behavior exactly.
    """
    import mlx.core as mx

    if max_new_tokens <= 0:
        return

    # --- Resolve helpers ---
    if materialize_cache_tree_fn is None or pad_transformer_caches_fn is None:
        import os
        _inf = os.path.dirname(os.path.abspath(__file__))
        for _p in (_inf, os.path.join(_inf, "lib")):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        from benchmark_mlx import _materialize_cache_tree, _pad_transformer_caches
        if materialize_cache_tree_fn is None:
            materialize_cache_tree_fn = _materialize_cache_tree
        if pad_transformer_caches_fn is None:
            pad_transformer_caches_fn = _pad_transformer_caches

    # --- Prefill ---
    if prefill_outputs is None:
        logits, caches = run_prefill(x_prefill, router_temp)
        mx.eval(logits, caches)
    else:
        logits, caches = prefill_outputs

    no_mat = getattr(sample_args, "no_materialize_caches", True)
    if not no_mat:
        caches = materialize_cache_tree_fn(caches)
        mx.eval(caches)
    caches = pad_transformer_caches_fn(caches, max_cache_len)
    mx.eval(caches)

    # --- Build / resolve compiled verify functions ---
    if compiled_verify_fns is not None:
        _vfns: dict = compiled_verify_fns
    else:
        _vfns = {}
        all_K = set(K_values) | {K, 1, 2, 3, 4, 5, 6, 7, 8}
        for _k in sorted(all_K):
            def _mk(k):
                def _fn(x_k, c, pos):
                    return model(x_k, caches=c, seq_pos=pos, router_temp=router_temp)
                return mx.compile(_fn)
            _vfns[_k] = _mk(_k)
        _dummy_caches = pad_transformer_caches_fn(
            run_prefill(mx.array([[0]], dtype=mx.int32), router_temp)[1], max_cache_len,
        )
        mx.eval(_dummy_caches)
        for _k, _fn in _vfns.items():
            _lw, _cw = _fn(mx.array([[0] * _k], dtype=mx.int32), _dummy_caches,
                           mx.array(0, dtype=mx.int32))
            mx.eval(_lw, _cw)
        del _dummy_caches, _lw, _cw

    def _call_verify(tokens_list: list[int], c: Any, p: int) -> tuple:
        n = len(tokens_list)
        fn = _vfns.get(n)
        x = mx.array([tokens_list], dtype=mx.int32)
        if fn is not None:
            return fn(x, c, mx.array(p, dtype=mx.int32))
        return model(x, caches=c, seq_pos=mx.array(p, dtype=mx.int32), router_temp=router_temp)

    _witrfns: dict = compiled_verify_with_intra_fns or {}
    _model_layers = list(getattr(getattr(model, "backbone", None), "layers", []))

    def _call_verify_with_intra(tokens_list: list[int], c: Any, p: int) -> tuple:
        """Returns (logits, caches, mamba_intra_per_layer).  Falls back to regular verify."""
        n = len(tokens_list)
        fn = _witrfns.get(n)
        x = mx.array([tokens_list], dtype=mx.int32)
        if fn is not None:
            return fn(x, c, mx.array(p, dtype=mx.int32))
        # Fallback: regular verify (no mamba_intra → batch_replay fallback)
        logits, caches_out = _call_verify(tokens_list, c, p)
        return logits, caches_out, None

    if compiled_single_fn is not None:
        compiled_single = compiled_single_fn
    else:
        compiled_single = _vfns.get(1) or mx.compile(
            lambda x, c, pos: model(x, caches=c, seq_pos=pos, router_temp=router_temp)
        )

    # --- Initialize caches ---
    ngram_cache = NGramCache(n=ngram_n, min_n=min_ngram_n, max_size=ngram_max_size)
    lookahead_cache = LookaheadCache(max_len=lookahead_max_len, max_size=lookahead_max_size)

    if use_ngram and warm_prompt and prompt_ids:
        ngram_cache.update(prompt_ids)

    pos = len(prompt_ids)
    generated_ids: list[int] = []
    prefix: list[int] = list(prompt_ids)

    # --- Banned-token helpers (prevents <unk>=0 flooding carry_draft) ---
    _ban_set = frozenset(ban_token_ids) if ban_token_ids else frozenset()
    # Last real (non-banned) token seen — used as fallback seed
    _last_real_seed: int = next((t for t in reversed(prompt_ids) if t not in _ban_set), 1)

    # Precompute 1-D ban mask per (V, dtype) lazily — built once, reused every round.
    # For the common case of banning only token 0 we use fast slice-concat to avoid
    # a 32 k-element Python list; general case falls through to the list path.
    _ban_mask_cache: dict[tuple, Any] = {}

    def _get_ban_mask(logit_row: Any) -> Any:
        V = logit_row.shape[-1]
        key = (V, str(logit_row.dtype))
        if key not in _ban_mask_cache:
            if _ban_set == frozenset({0}):
                # Fast path: [-inf, 0, 0, …]
                _ban_mask_cache[key] = mx.concatenate([
                    mx.array([float('-inf')], dtype=logit_row.dtype),
                    mx.zeros((V - 1,), dtype=logit_row.dtype),
                ])
            else:
                mask_vals = [float('-inf') if i in _ban_set else 0.0 for i in range(V)]
                _ban_mask_cache[key] = mx.array(mask_vals, dtype=logit_row.dtype)
        return _ban_mask_cache[key]

    def _safe_argmax(logit_row: Any) -> int:
        """Argmax skipping banned IDs; falls back to last real seed if all banned."""
        tok = int(mx.argmax(logit_row, axis=-1).item())
        if not _ban_set or tok not in _ban_set:
            return tok
        tok2 = int(mx.argmax(logit_row + _get_ban_mask(logit_row), axis=-1).item())
        return tok2 if tok2 not in _ban_set else _last_real_seed

    def _safe_preds(pred_logits_batch: Any) -> Any:
        """Vectorized argmax [K, V] with banned IDs forced to -inf (broadcast)."""
        if not _ban_set:
            return mx.argmax(pred_logits_batch, axis=-1)
        # _get_ban_mask returns shape [V]; broadcast over [K, V] automatically.
        return mx.argmax(pred_logits_batch + _get_ban_mask(pred_logits_batch[0]), axis=-1)

    # Sampling helpers — keeps AR warmup bit-identical to stream_mlx.py.
    # sample_decode_token honours fast_sample / temp / top_k / penalties and does NOT
    # ban token 0, eliminating the divergence caused by _safe_argmax's ban_token_ids.
    from benchmark_mlx import sample_decode_token as _sample_decode_token, _init_token_counts as _init_tc
    import types as _types
    # Build a fully-populated sample_args for the AR warmup phase with safe defaults
    # so callers that only pass a minimal sample_args (e.g. compare_ar_jacobi.py) work.
    _sa_warmup = _types.SimpleNamespace(
        rep_pen   = getattr(sample_args, "rep_pen",   1.0),
        pres_pen  = getattr(sample_args, "pres_pen",  0.0),
        freq_pen  = getattr(sample_args, "freq_pen",  0.0),
        fast_sample = getattr(sample_args, "fast_sample", True),
        temp      = getattr(sample_args, "temp",      0.0),
        top_k     = getattr(sample_args, "top_k",     0),
        top_p     = getattr(sample_args, "top_p",     1.0),
        min_p     = getattr(sample_args, "min_p",     0.0),
        no_penalties = getattr(sample_args, "no_penalties", True),
        fused_sample_metal    = getattr(sample_args, "fused_sample_metal",    False),
        fused_sample_metal_v2 = getattr(sample_args, "fused_sample_metal_v2", False),
    )
    _vocab_size = int(logits.shape[-1])
    _token_counts = _init_tc(list(prompt_ids), _vocab_size)

    # --- First token from prefill logits ---
    row0 = logits[0, -1, :]
    _tok0_arr = _sample_decode_token(row0, _token_counts, _sa_warmup)
    mx.eval(_tok0_arr)
    tok0 = int(_tok0_arr.item())
    if not _sa_warmup.no_penalties:
        _token_counts[tok0] = _token_counts[tok0] + 1
    if tok0 not in _ban_set:
        _last_real_seed = tok0
    yield mx.array(tok0, dtype=mx.int32)
    generated_ids.append(tok0)
    prefix.append(tok0)
    if stop_token_ids and tok0 in stop_token_ids:
        return

    logits_1, caches = _call_verify([tok0], caches, pos)
    mx.eval(logits_1, caches)
    # Detach caches from compiled graph output buffer before switching K sizes.
    # Without this, subsequent K!=1 verify calls read NaN from aliased buffers.
    caches = materialize_cache_tree_fn(caches)
    mx.eval(caches)
    current_logit_row = logits_1[0, -1, :]
    pos += 1

    carry_draft: list[int] = [tok0 if tok0 not in _ban_set else _last_real_seed]

    # --- Stats and adaptive-K state ---
    _rounds = 0
    _accepted_total = 0
    _proposed_total = 0
    _lookahead_assists = 0
    _stall_rounds = 0
    _arl_ema: float = float(K)
    _ema_alpha: float = 0.25
    _current_K: int = K
    # P0 tree stats
    _tree_branch_wins: list[int] = [0] * num_tree_branches
    _tree_rounds = 0
    # P1 entropy stats
    _entropy_sum = 0.0
    _entropy_rounds = 0

    # Per-round correctness metrics (written live into stats dict)
    if stats is not None:
        stats.setdefault("per_round_arl", [])
        stats.setdefault("ngram_hit_rounds", 0)
        stats.setdefault("intra_rounds", 0)
        stats.setdefault("replay_rounds", 0)

    def _next_K(arl: float, cur_K: int) -> int:
        nonlocal _arl_ema
        _arl_ema = _ema_alpha * arl + (1.0 - _ema_alpha) * _arl_ema
        if not use_dynamic_K:
            return cur_K
        if _arl_ema >= cur_K * 0.85 and cur_K < K_max:
            new_k = cur_K + 4
        elif _arl_ema < cur_K * 0.4 and cur_K > K_min:
            new_k = cur_K - 4
        else:
            new_k = cur_K
        valid = sorted(kv for kv in K_values if K_min <= kv <= K_max)
        return min(valid, key=lambda kv: abs(kv - new_k)) if valid else cur_K

    def _batch_replay(consume: list[int], start_caches: Any, start_pos: int) -> tuple:
        """Replay consume tokens in ONE batch verify call.

        No redundant materialize+eval: mx.eval(logits_r, new_caches) already
        forces GPU computation; _deep_copy_caches at the next round start will
        handle any aliasing by creating fresh numpy-backed buffers.
        """
        logits_r, new_caches = _call_verify(consume, start_caches, start_pos)
        mx.eval(logits_r, new_caches)
        return logits_r[0, -1, :], new_caches, start_pos + len(consume)

    # --- Autoregressive warmup (fills N-gram cache before Jacobi guessing starts) ---
    # tok0 is already generated; generate ar_warmup_tokens-1 more tokens one-by-one.
    _warmup_target = max(0, ar_warmup_tokens - 1)  # tok0 counts as first warmup token
    _warmup_done = 0
    while _warmup_done < _warmup_target and len(generated_ids) < max_new_tokens:
        _pred_arr = _sample_decode_token(current_logit_row, _token_counts, _sa_warmup)
        mx.eval(_pred_arr)
        pred = int(_pred_arr.item())
        if not _sa_warmup.no_penalties:
            _token_counts[pred] = _token_counts[pred] + 1
        if pred not in _ban_set:
            _last_real_seed = pred
        yield mx.array(pred, dtype=mx.int32)
        generated_ids.append(pred)
        prefix.append(pred)
        if stop_token_ids and pred in stop_token_ids:
            _log_stats(_rounds, _accepted_total, _proposed_total,
                       _lookahead_assists, _tree_branch_wins, _tree_rounds,
                       _entropy_sum, _entropy_rounds, stats=stats)
            return
        logits_w, caches = _call_verify([pred], caches, pos)
        mx.eval(logits_w, caches)
        caches = materialize_cache_tree_fn(caches)
        mx.eval(caches)
        current_logit_row = logits_w[0, -1, :]
        pos += 1
        carry_draft = [pred if pred not in _ban_set else _last_real_seed]
        _warmup_done += 1

    # Seed N-gram cache with warmup history before Jacobi begins.
    # Include `ngram_n` tokens of prompt context so transition n-grams are captured.
    if use_ngram and _warmup_done > 0:
        overlap = ngram_cache.n
        update_start = max(0, len(prompt_ids) - overlap)
        ngram_cache.update(prefix[update_start:])

    # --- Main generation loop ---
    while len(generated_ids) < max_new_tokens:
        remaining = max_new_tokens - len(generated_ids)
        K_this = min(_current_K, remaining)

        # ── Forced single-step fallback when Jacobi is stuck ─────────────────
        if _stall_rounds >= max_stall_rounds:
            fallback_n = min(K_this, remaining)
            for _ in range(fallback_n):
                pred = _safe_argmax(current_logit_row)
                if pred not in _ban_set:
                    _last_real_seed = pred
                logits_s, caches = _call_verify([pred], caches, pos)
                mx.eval(logits_s, caches)
                current_logit_row = logits_s[0, -1, :]
                pos += 1
                yield mx.array(pred, dtype=mx.int32)
                generated_ids.append(pred)
                prefix.append(pred)
                carry_draft = [pred if pred not in _ban_set else _last_real_seed]
                if stop_token_ids and pred in stop_token_ids:
                    _log_stats(_rounds, _accepted_total, _proposed_total,
                               _lookahead_assists, _tree_branch_wins, _tree_rounds,
                               _entropy_sum, _entropy_rounds, stats=stats)
                    return
                if len(generated_ids) >= max_new_tokens:
                    _log_stats(_rounds, _accepted_total, _proposed_total,
                               _lookahead_assists, _tree_branch_wins, _tree_rounds,
                               _entropy_sum, _entropy_rounds, stats=stats)
                    return
            _stall_rounds = 0
            _arl_ema = float(_current_K) * 0.5
            continue

        # ── P1: Entropy-based K and branch adaptation ─────────────────────────
        if use_entropy_k:
            ent = _entropy(current_logit_row)
            _entropy_sum += ent
            _entropy_rounds += 1
            K_this_adj, B_eff = _entropy_decide_K_B(
                ent, entropy_low, entropy_high,
                K_min, K_max, K_values,
                num_tree_branches, use_tree,
            )
            K_this = min(K_this_adj, remaining)
        else:
            B_eff = num_tree_branches if use_tree else 1

        # ── P0: Tree Attention path ───────────────────────────────────────────
        if use_tree and B_eff > 1 and K_this >= 2:
            D = max(1, tree_branch_depth if tree_branch_depth is not None else K_this // B_eff)
            D = min(D, remaining)
            B_actual = B_eff

            tree = build_guess_tree(
                carry_draft[0], ngram_cache, prefix, D, B_actual
            )

            _rounds += 1
            _proposed_total += B_actual * D
            _tree_rounds += 1

            # Snapshot BEFORE any branch verify — _call_verify may do in-place
            # mx.slice_update on KV/SSM buffers even though we pass the same caches.
            # Partial-accept must replay from the pre-verify state.
            tree_caches_snapshot = _deep_copy_caches(caches)

            # Verify each branch against the SAME original caches (no mutation)
            logits_branches: list[Any] = []
            verify_caches_branches: list[Any] = []

            for b_idx, branch in enumerate(tree):
                logits_b, caches_b = _call_verify(branch, caches, pos)
                logits_branches.append(logits_b)
                verify_caches_branches.append(caches_b)

                # Early stopping: if branch 0 already accepts enough, skip rest
                if tree_early_stop and b_idx == 0:
                    # Quick check: how many depth-1+ positions match for branch 0
                    # (depth 0 always matches since carry_token = argmax of current_logit_row)
                    # We don't fully evaluate yet — check after eval below
                    pass

            # Evaluate all branch logits at once (potential Metal-level parallelism)
            mx.eval(*logits_branches)

            # Find best branch
            best_b, accepted, correction = _find_best_tree_branch(
                tree, logits_branches, current_logit_row
            )
            if best_b < len(_tree_branch_wins):
                _tree_branch_wins[best_b] += 1
            best_branch = tree[best_b]
            _accepted_total += accepted

            # Accept tokens
            if accepted == D:
                # Full branch accept: reuse verify caches directly
                consume = best_branch
                caches = verify_caches_branches[best_b]
                mx.eval(caches)
                caches = materialize_cache_tree_fn(caches)
                mx.eval(caches)
                current_logit_row = logits_branches[best_b][0, D - 1, :]
                if correction not in _ban_set:
                    _last_real_seed = correction
                carry_draft = [correction if correction not in _ban_set else _last_real_seed]
                _stall_rounds = 0

                for t in consume:
                    yield mx.array(t, dtype=mx.int32)
                    generated_ids.append(t)
                    prefix.append(t)
                    if stop_token_ids and t in stop_token_ids:
                        _log_stats(_rounds, _accepted_total, _proposed_total,
                                   _lookahead_assists, _tree_branch_wins, _tree_rounds,
                                   _entropy_sum, _entropy_rounds, stats=stats)
                        return
                    if len(generated_ids) >= max_new_tokens:
                        _log_stats(_rounds, _accepted_total, _proposed_total,
                                   _lookahead_assists, _tree_branch_wins, _tree_rounds,
                                   _entropy_sum, _entropy_rounds, stats=stats)
                        return
                pos += D
                _update_caches(prefix, ngram_cache, lookahead_cache, D)

            else:
                # Partial accept: replay against the pre-verify snapshot (caches may be
                # mutated in-place by the branch _call_verify loop above).
                consume = best_branch[:accepted] + [correction]
                current_logit_row, caches, pos = _batch_replay(consume, tree_caches_snapshot, pos)
                next_seed = _safe_argmax(current_logit_row)
                if next_seed not in _ban_set:
                    _last_real_seed = next_seed
                carry_draft = [next_seed if next_seed not in _ban_set else _last_real_seed]
                _stall_rounds = 0 if accepted > 0 else _stall_rounds + 1

                for t in consume:
                    yield mx.array(t, dtype=mx.int32)
                    generated_ids.append(t)
                    prefix.append(t)
                    if stop_token_ids and t in stop_token_ids:
                        _log_stats(_rounds, _accepted_total, _proposed_total,
                                   _lookahead_assists, _tree_branch_wins, _tree_rounds,
                                   _entropy_sum, _entropy_rounds, stats=stats)
                        return
                    if len(generated_ids) >= max_new_tokens:
                        _log_stats(_rounds, _accepted_total, _proposed_total,
                                   _lookahead_assists, _tree_branch_wins, _tree_rounds,
                                   _entropy_sum, _entropy_rounds, stats=stats)
                        return
                if accepted > 0:
                    _update_caches(prefix, ngram_cache, lookahead_cache, len(consume))

            _current_K = _next_K(accepted, _current_K)
            continue

        # ── Linear path (existing v3 logic) ──────────────────────────────────
        # Lookahead seeds guess buffer prefix (zero extra cost on hit)
        la_prefix = None
        if use_lookahead:
            la_draft = lookahead_cache.query_draft(prefix, max_draft=K_this)
            if la_draft and len(la_draft) >= min_lookahead_hit:
                la_prefix = la_draft[:K_this]
                _lookahead_assists += 1

        # Build K_this-token guess buffer
        if use_ngram:
            if stats is not None and ngram_cache.query(prefix):
                stats["ngram_hit_rounds"] += 1
            guesses = _ngram_init_guesses(
                prefix, K_this, ngram_cache, carry_draft, la_prefix, ban_ids=_ban_set,
            )
        else:
            seed = carry_draft[0] if carry_draft else prefix[-1]
            if seed in _ban_set:
                seed = _last_real_seed
            if la_prefix:
                la_len = len(la_prefix)
                guesses = list(la_prefix[:K_this]) + [seed] * max(K_this - la_len, 0)
            else:
                guesses = [seed] * K_this

        _rounds += 1
        _proposed_total += K_this

        # Snapshot caches BEFORE the K>1 verify call.
        # mx.compile may perform mx.slice_update in-place on the KV cache buffers,
        # meaning `caches` may be mutated to reflect pos+K after the call.
        # If partial accept occurs, _batch_replay must start from the ORIGINAL pos,
        # so we detach a fresh copy now rather than trying to recover after the fact.
        if K_this > 1:
            # numpy round-trip guarantees a new GPU buffer — materialize_cache_tree_fn
            # uses a+zeros(()) which MLX folds to identity and aliases the original.
            # Without a true copy, K-step verify's in-place SSM state mutation corrupts
            # the snapshot, causing _batch_replay to start from the wrong Mamba state.
            caches_snapshot = _deep_copy_caches(caches)
        else:
            caches_snapshot = caches

        # One verify call for the full guess buffer.
        # Use verify+intra only when partial-accept is likely (ARL EMA << K), because
        # the intra path adds ~24ms of graph overhead that hurts full-accept rounds.
        # When ARL EMA ≥ 90% of K we expect full-accept → prefer regular verify.
        _expect_partial = _arl_ema < K_this * 0.9
        if _witrfns and K_this > 1 and _expect_partial:
            logits_v, verify_caches, _mamba_intra = _call_verify_with_intra(guesses, caches, pos)
        else:
            logits_v, verify_caches = _call_verify(guesses, caches, pos)
            _mamba_intra = None
        mx.eval(logits_v, verify_caches)

        # Vectorized comparison: batch all K predictions in one mx.eval
        if K_this == 1:
            pred_logits = current_logit_row[None, :]
        else:
            pred_logits = mx.concatenate(
                [current_logit_row[None, :], logits_v[0, :K_this - 1, :]], axis=0
            )
        preds = _safe_preds(pred_logits)
        guesses_arr = mx.array(guesses, dtype=mx.int32)
        matches = (preds == guesses_arr)
        mx.eval(preds, matches)

        matches_list = matches.tolist()
        accepted = 0
        for m in matches_list:
            if m:
                accepted += 1
            else:
                break
        correction = int(preds[accepted].item()) if accepted < K_this else None

        # Track per-round ARL for metrics
        if stats is not None:
            stats["per_round_arl"].append(accepted)

        # Full accept
        if accepted == K_this:
            consume = guesses
            caches = verify_caches   # already eval'd above; _deep_copy at next round handles aliasing
            current_logit_row = logits_v[0, K_this - 1, :]
            next_seed = _safe_argmax(current_logit_row)
            if next_seed not in _ban_set:
                _last_real_seed = next_seed
            carry_draft = [next_seed if next_seed not in _ban_set else _last_real_seed]
            _stall_rounds = 0
            _accepted_total += accepted

            for t in consume:
                if t not in _ban_set:
                    _last_real_seed = t
                yield mx.array(t, dtype=mx.int32)
                generated_ids.append(t)
                prefix.append(t)
                if stop_token_ids and t in stop_token_ids:
                    _log_stats(_rounds, _accepted_total, _proposed_total,
                               _lookahead_assists, _tree_branch_wins, _tree_rounds,
                               _entropy_sum, _entropy_rounds, stats=stats)
                    return
                if len(generated_ids) >= max_new_tokens:
                    _log_stats(_rounds, _accepted_total, _proposed_total,
                               _lookahead_assists, _tree_branch_wins, _tree_rounds,
                               _entropy_sum, _entropy_rounds, stats=stats)
                    return
            pos += len(consume)
            _update_caches(prefix, ngram_cache, lookahead_cache, len(consume))

        # Partial accept
        else:
            assert correction is not None
            consume = guesses[:accepted] + [correction]
            if _mamba_intra is not None and accepted > 0 and _model_layers:
                # Fast path: extract Mamba state at verify position `accepted-1`,
                # reuse Transformer KV from verify_caches (future entries are masked),
                # then run ONE K=1 correction step instead of full batch_replay.
                if stats is not None:
                    stats["intra_rounds"] += 1
                partial_caches = _extract_mamba_caches_at(
                    accepted, verify_caches, _mamba_intra, _model_layers
                )
                mx.eval(partial_caches)
                logits_corr, caches = _call_verify([correction], partial_caches, pos + accepted)
                mx.eval(logits_corr, caches)
                caches = materialize_cache_tree_fn(caches)
                mx.eval(caches)
                current_logit_row = logits_corr[0, -1, :]
                pos = pos + accepted + 1
            else:
                if stats is not None:
                    stats["replay_rounds"] += 1
                current_logit_row, caches, pos = _batch_replay(consume, caches_snapshot, pos)
            next_seed = _safe_argmax(current_logit_row)
            if next_seed not in _ban_set:
                _last_real_seed = next_seed
            carry_draft = [next_seed if next_seed not in _ban_set else _last_real_seed]
            _stall_rounds = 0 if accepted > 0 else _stall_rounds + 1
            _accepted_total += accepted

            for t in consume:
                if t not in _ban_set:
                    _last_real_seed = t
                yield mx.array(t, dtype=mx.int32)
                generated_ids.append(t)
                prefix.append(t)
                if stop_token_ids and t in stop_token_ids:
                    _log_stats(_rounds, _accepted_total, _proposed_total,
                               _lookahead_assists, _tree_branch_wins, _tree_rounds,
                               _entropy_sum, _entropy_rounds, stats=stats)
                    return
                if len(generated_ids) >= max_new_tokens:
                    _log_stats(_rounds, _accepted_total, _proposed_total,
                               _lookahead_assists, _tree_branch_wins, _tree_rounds,
                               _entropy_sum, _entropy_rounds, stats=stats)
                    return
            if accepted > 0:
                _update_caches(prefix, ngram_cache, lookahead_cache, len(consume))

        _current_K = _next_K(accepted, _current_K)

    _log_stats(_rounds, _accepted_total, _proposed_total, _lookahead_assists,
               _tree_branch_wins, _tree_rounds, _entropy_sum, _entropy_rounds, stats=stats)


# ---------------------------------------------------------------------------
# Build compiled verify functions dict (call once, reuse across runs)
# ---------------------------------------------------------------------------

def build_compiled_verify_fns(
    model: Any,
    router_temp: Any,
    K_values: tuple[int, ...],
    caches_for_warmup: Any,
    prompt_len: int = 1,
    tree_branch_depth: int | None = None,
    num_tree_branches: int = 3,
) -> dict:
    """
    Build and warmup outer-compiled verify functions for each K in K_values.

    Includes dense small-K coverage (1..8) so batch_replay always finds a
    compiled fn for replay sizes up to 8.  Also includes K+1 and tree branch
    depth D for tree attention mode.

    Build ONCE before the generation loop and reuse across all prompts/configs.
    """
    import mlx.core as mx
    fns: dict = {}
    dense_small = set(range(1, 9))
    replay_sizes = {k + 1 for k in K_values}
    # Include tree branch depth if specified
    tree_sizes: set[int] = set()
    if tree_branch_depth is not None:
        tree_sizes.add(tree_branch_depth)
    for k in K_values:
        d = max(1, k // num_tree_branches)
        tree_sizes.add(d)
    all_K = sorted(dense_small | set(K_values) | replay_sizes | tree_sizes)
    for k in all_K:
        def _mk(kk):
            def _fn(x_k, c, pos):
                return model(x_k, caches=c, seq_pos=pos, router_temp=router_temp)
            return mx.compile(_fn)
        fns[k] = _mk(k)
    # Warmup
    for k, fn in fns.items():
        _lw, _cw = fn(
            mx.array([[0] * k], dtype=mx.int32),
            caches_for_warmup,
            mx.array(prompt_len, dtype=mx.int32),
        )
        mx.eval(_lw, _cw)
    return fns


# ---------------------------------------------------------------------------
# Compiled verify-with-intra functions (replaces batch_replay for partial-accept)
# ---------------------------------------------------------------------------

def build_compiled_verify_with_intra_fns(
    model: Any,
    router_temp: Any,
    K_values: tuple[int, ...],
    caches_for_warmup: Any,
    prompt_len: int = 1,
) -> dict:
    """Build compiled functions that call model.forward_with_mamba_intra.

    Each function returns (logits, caches, mamba_intra_per_layer).
    mamba_intra_per_layer is a list over Mamba layers of (h_all_corrected, input_signal, angles):
      h_all_corrected : (b, K, h, n, p) — corrected SSM state after each verify token
      input_signal    : (b, K, h, n, p) — SSM input at each position
      angles          : (b, K, h, n//2) — cumulative RoPE angles at each position

    Use the state at position `accepted-1` (0-indexed) to reconstruct Mamba caches
    at the accept boundary without running a separate batch_replay.
    """
    import mlx.core as mx
    fns: dict = {}
    # Only need to cover verify sizes (not small-K replay — those still use regular fns).
    for k in sorted(set(K_values)):
        def _mk(kk):
            def _fn(x_k, c, pos):
                return model.forward_with_mamba_intra(
                    x_k, caches=c, seq_pos=pos, router_temp=router_temp
                )
            return mx.compile(_fn)
        fns[k] = _mk(k)
    # Warmup
    for k, fn in fns.items():
        _lw, _cw, _mi = fn(
            mx.array([[0] * k], dtype=mx.int32),
            caches_for_warmup,
            mx.array(prompt_len, dtype=mx.int32),
        )
        mx.eval(_lw, _cw)
        for entry in _mi:
            if entry is not None:
                mx.eval(*entry)
    return fns


def _extract_mamba_caches_at(
    accepted: int,
    verify_caches: Any,
    mamba_intra_per_layer: list,
    layers: list,
) -> list:
    """Build partial caches: Mamba state at verify position `accepted-1`, Transformer KV unchanged.

    Args:
        accepted: number of guesses accepted (>= 1).  The Mamba state to extract
                  is after processing the (accepted-1)-th token (0-indexed).
        verify_caches: full cache list after K-token verify (Mamba at pos+K).
        mamba_intra_per_layer: list of (h_all, inp_sig, angs) per Mamba layer,
                               in the same order as mamba layers appear in `layers`.
        layers: list of model backbone layers (to identify Mamba vs Transformer).

    Returns:
        new_caches: same structure as verify_caches but with Mamba states
                    rolled back to position accepted-1.
    """
    import mlx.core as mx
    j = accepted - 1  # 0-indexed position after accepted guesses
    new_caches = []
    mamba_idx = 0
    for layer, cache in zip(layers, verify_caches):
        lt = getattr(layer, "l_type", None)
        if lt == "mamba":
            entry = mamba_intra_per_layer[mamba_idx] if mamba_idx < len(mamba_intra_per_layer) else None
            if entry is not None:
                h_all, inp_sig, angs = entry
                # h_all: (b, K, h, n, p) — state AFTER processing position j
                # inp_sig: (b, K, h, n, p)
                # angs: (b, K, h, n//2)
                h_at  = h_all[:, j, :, :, :]            # (b, h, n, p)
                i_at  = inp_sig[:, j:j+1, :, :, :]      # (b, 1, h, n, p)
                a_at  = angs[:, j:j+1, :, :]             # (b, 1, h, n//2)
                new_caches.append((h_at, i_at, a_at))
            else:
                new_caches.append(cache)  # fallback (shouldn't happen in verify mode)
            mamba_idx += 1
        else:
            # Transformer KV: reuse verify_caches directly.
            # The decode attention mask (j > q_pos → -1e9) blocks positions > pos+accepted
            # even though entries up to pos+K-1 are written, so no correction needed.
            new_caches.append(cache)
    return new_caches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _update_caches(
    prefix: list[int],
    ngram_cache: NGramCache,
    lookahead_cache: LookaheadCache,
    n_new: int,
) -> None:
    """Incremental cache update: only process the slice containing new n-grams."""
    overlap = ngram_cache.n + n_new
    ngram_cache.update(prefix[-overlap:])
    overlap_la = lookahead_cache.max_len + n_new + 1
    lookahead_cache.update(prefix[-overlap_la:])


def _log_stats(
    rounds: int,
    accepted: int,
    proposed: int,
    la_assists: int,
    tree_branch_wins: list[int],
    tree_rounds: int,
    entropy_sum: float,
    entropy_rounds: int,
    stats: dict | None = None,
) -> None:
    rate = 100.0 * accepted / max(proposed, 1)
    arl = accepted / max(rounds, 1)
    la_rate = la_assists / max(rounds, 1)
    mean_ent = entropy_sum / max(entropy_rounds, 1)

    tree_info = ""
    if tree_rounds > 0:
        wins_str = "/".join(str(w) for w in tree_branch_wins)
        tree_info = f"  tree_rounds={tree_rounds}  branch_wins=[{wins_str}]"

    entropy_info = f"  mean_entropy={mean_ent:.3f}" if entropy_rounds > 0 else ""

    print(
        f"\n[jacobi-v4] rounds={rounds}  accepted={accepted}/{proposed}"
        f"  ({rate:.1f}%)  ARL={arl:.2f}  la_assists={la_assists} ({100*la_rate:.1f}%)"
        f"{tree_info}{entropy_info}",
        flush=True,
    )
    if stats is not None:
        stats["rounds"] = rounds
        stats["accepted"] = accepted
        stats["proposed"] = proposed
        stats["lookahead_hits"] = la_assists
        stats["arl"] = arl
        stats["la_hit_rate"] = la_rate
        stats["tree_rounds"] = tree_rounds
        stats["tree_branch_wins"] = list(tree_branch_wins)
        stats["mean_entropy"] = mean_ent
        # Derived rate for per-round ngram hit
        stats["ngram_hit_rate"] = stats.get("ngram_hit_rounds", 0) / max(rounds, 1)


# ---------------------------------------------------------------------------
# Self-contained tests
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    print("=" * 60)
    print("jacobi_enhanced.py v4 — unit tests")
    print("=" * 60)
    failures: list[str] = []

    # --- NGramCache (multi-order) ---
    print("\n[1] NGramCache (multi-order, n=2, min_n=1)")
    ng = NGramCache(n=2, min_n=1, max_size=200)
    tokens = [1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2, 3]
    ng.update(tokens)

    result = ng.query([1, 2])
    assert result == 3, f"expected 3 got {result}"
    print("  query([1,2]) == 3  ✓")

    result2 = ng.query([99, 99])
    assert result2 is None, f"expected None got {result2}"
    print("  query([99,99]) == None  ✓")

    # query_topk test
    topk = ng.query_topk([1, 2], k=3)
    assert 3 in topk and 4 in topk, f"expected 3 and 4 in topk: {topk}"
    assert topk[0] == 3, f"expected 3 as most common: {topk}"
    print(f"  query_topk([1,2], k=3) = {topk}  ✓")

    topk_none = ng.query_topk([99, 99], k=3)
    assert topk_none == [], f"expected [] for miss: {topk_none}"
    print("  query_topk([99,99]) == []  ✓")

    # --- build_guess_tree ---
    print("\n[2] build_guess_tree")
    ng2 = NGramCache(n=2, min_n=1, max_size=200)
    # "the" most common after [5], "a" second
    ng2.update([5, 10, 5, 10, 5, 10, 5, 20, 5, 30])  # 10 appears 3×, 20 and 30 once each
    tree = build_guess_tree(
        carry_token=10, ngram_cache=ng2, prefix=[5], branch_depth=3, num_branches=3
    )
    assert len(tree) == 3, f"expected 3 branches: {len(tree)}"
    assert len(tree[0]) == 3, f"expected depth 3: {len(tree[0])}"
    assert tree[0][0] == 10, f"branch 0 should start with carry_token=10: {tree[0]}"
    print(f"  tree[0]={tree[0]}, tree[1]={tree[1]}, tree[2]={tree[2]}  ✓")

    # --- _entropy ---
    print("\n[3] _entropy")
    try:
        import mlx.core as mx
        # Peaked distribution → low entropy
        peaked = mx.array([10.0, 0.0, 0.0, 0.0])
        ent_low = _entropy(peaked)
        # Flat distribution → high entropy
        flat = mx.array([0.0, 0.0, 0.0, 0.0])
        ent_high = _entropy(flat)
        assert ent_low < ent_high, f"peaked entropy {ent_low:.3f} should be < flat {ent_high:.3f}"
        print(f"  peaked entropy={ent_low:.3f} < flat entropy={ent_high:.3f}  ✓")
    except ImportError:
        print("  MLX not available — skipping entropy test")

    # --- _entropy_decide_K_B ---
    print("\n[4] _entropy_decide_K_B")
    K_vals = (4, 8, 16, 32)
    k, b = _entropy_decide_K_B(0.2, 0.5, 1.5, 4, 32, K_vals, 3, True)
    assert k == 32 and b == 1, f"low entropy: expected K=32, B=1, got K={k}, B={b}"
    print(f"  low entropy → K={k}, B={b}  ✓")

    k, b = _entropy_decide_K_B(1.0, 0.5, 1.5, 4, 32, K_vals, 3, True)
    assert b == 2, f"medium entropy: expected B=2, got B={b}"
    print(f"  medium entropy → K={k}, B={b}  ✓")

    k, b = _entropy_decide_K_B(2.0, 0.5, 1.5, 4, 32, K_vals, 3, True)
    assert k == 4 and b == 1, f"high entropy: expected K=4, B=1, got K={k}, B={b}"
    print(f"  high entropy → K={k}, B={b}  ✓")

    # --- LookaheadCache ---
    print("\n[5] LookaheadCache")
    la = LookaheadCache(max_len=3, max_size=500)
    seq = [10, 20, 30, 40, 50, 10, 20, 30, 40, 60]
    la.update(seq)

    draft = la.query_draft([10, 20], max_draft=3)
    assert draft is not None and draft[0] == 30, f"expected 30 as first draft got {draft}"
    print(f"  query_draft([10,20]) starts with 30  ✓  full={draft}")

    # --- _ngram_init_guesses ---
    print("\n[6] _ngram_init_guesses")
    ng3 = NGramCache(n=2, min_n=1, max_size=200)
    ng3.update([5, 6, 7, 5, 6, 8, 5, 6, 7])
    guesses = _ngram_init_guesses([5, 6], K=3, ngram_cache=ng3, carry_draft=None)
    assert len(guesses) == 3 and guesses[0] == 7, f"expected [7,...] got {guesses}"
    print(f"  guesses [5,6] K=3: {guesses}  ✓")

    guesses_la = _ngram_init_guesses([5, 6], K=4, ngram_cache=ng3,
                                     carry_draft=[99, 99, 99, 99], la_prefix=[11, 22])
    assert guesses_la[0] == 11 and guesses_la[1] == 22, f"la_prefix not honored: {guesses_la}"
    assert guesses_la[2] == 99, f"carry not used at pos 2: {guesses_la}"
    print(f"  la_prefix + carry fallback: {guesses_la}  ✓")

    # --- MLX integration tests ---
    print("\n[7] enhanced_jacobi_stream v4 (MLX mock model — linear mode)")
    try:
        import mlx.core as mx
        _run_mlx_mock_test(mx, failures, use_tree=False)
    except ImportError:
        print("  MLX not available — skipping stream integration tests")

    print("\n[8] enhanced_jacobi_stream v4 (MLX mock model — tree mode)")
    try:
        import mlx.core as mx
        _run_mlx_mock_test(mx, failures, use_tree=True)
    except ImportError:
        print("  MLX not available — skipping tree stream test")

    print("\n[9] enhanced_jacobi_stream v4 (entropy-based K)")
    try:
        import mlx.core as mx
        _run_mlx_mock_test(mx, failures, use_tree=False, use_entropy_k=True)
    except ImportError:
        print("  MLX not available — skipping entropy-K test")

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} test(s)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All tests PASSED")


def _run_mlx_mock_test(
    mx, failures: list[str],
    use_tree: bool = False,
    use_entropy_k: bool = False,
) -> None:
    """
    Integration test: mock model always predicts (token + 1) % VOCAB.
    Starting from prompt [1,2,3], expected 8-token output: [4,5,6,7,8,9,10,11].
    """
    VOCAB = 20
    vocab_range = mx.arange(VOCAB)

    class MockModel:
        def __call__(self, x: Any, *, caches: Any, seq_pos: Any, router_temp: Any) -> tuple:
            targets = (x + 1) % VOCAB
            flat = targets.reshape(-1)
            one_hot = (flat[:, None] == vocab_range).astype(mx.float32) * 100.0
            logits = one_hot.reshape(x.shape[0], -1, VOCAB)
            return logits, None

    model = MockModel()
    router_temp = mx.array(0.5)
    prompt_ids = [1, 2, 3]
    K = 4
    tag = f"tree={use_tree}, entropy_k={use_entropy_k}"

    last_row = [0.0] * VOCAB
    last_row[4] = 100.0
    dummy_logits = mx.array([[[0.0] * VOCAB] * (len(prompt_ids) - 1) + [last_row]])

    class MockArgs:
        no_materialize_caches = True
        fast_sample = True
        temp = 0.0; top_k = 0; top_p = 1.0; min_p = 0.0
        rep_pen = 1.0; pres_pen = 0.0; freq_pen = 0.0; no_penalties = True

    try:
        _vfns_mock: dict = {}
        for _k in sorted(set(range(1, K + 4)) | {K}):
            def _mk(kk):
                def _fn(xk, c, pos):
                    return model(xk, caches=c, seq_pos=pos, router_temp=router_temp)
                return mx.compile(_fn)
            _vfns_mock[_k] = _mk(_k)
        for _k, _fn in _vfns_mock.items():
            _lw, _cw = _fn(mx.array([[0] * _k], dtype=mx.int32), None, mx.array(0, dtype=mx.int32))
            mx.eval(_lw, _cw)

        token_iter = enhanced_jacobi_stream(
            model=model, run_prefill=None, x_prefill=None,
            prompt_ids=prompt_ids, max_cache_len=512, router_temp=router_temp,
            max_new_tokens=8, sample_args=MockArgs(), stop_token_ids=None,
            prefill_outputs=(dummy_logits, None),
            K=K, K_min=1, K_max=K, K_values=(1, 2, K),
            ngram_n=2, min_ngram_n=1, ngram_max_size=100,
            lookahead_max_len=3, lookahead_max_size=100, min_lookahead_hit=2,
            warm_prompt=True,
            use_tree=use_tree,
            num_tree_branches=2 if use_tree else 1,
            use_entropy_k=use_entropy_k,
            entropy_low=0.3,
            entropy_high=2.0,
            compiled_verify_fns=_vfns_mock,
            materialize_cache_tree_fn=lambda x: x,
            pad_transformer_caches_fn=lambda c, _: c,
        )
        collected: list[int] = []
        for tok in token_iter:
            mx.eval(tok)
            collected.append(int(tok.item()))

        print(f"  [{tag}] Generated: {collected}")
        assert len(collected) == 8, f"expected 8 tokens got {len(collected)}"
        expected = [(4 + i) % VOCAB for i in range(8)]
        assert collected == expected, f"expected {expected} got {collected}"
        print(f"  [{tag}] Token sequence matches {expected}  ✓")
    except Exception as e:
        msg = f"MLX mock test [{tag}] failed: {e}"
        failures.append(msg)
        print(f"  FAIL: {msg}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="jacobi_enhanced v4 — test mode")
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    if args.test:
        _run_tests()
    else:
        p.print_help()
