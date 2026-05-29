"""Draft-token sources for Jacobi-style speculative decoding.

Three independent draft sources are provided here, all training-free and all
preserving byte-equality with AR-greedy when verified through ``jacobi_decode``:

* :class:`NGramCache` (imported from ``.ngram_cache``)
    LRU map from ``(n-1)``-token contexts → MRU continuation tokens.  Tight
    locality, fast updates, low memory.  See *The N-Grammys* (arXiv 2411.03786)
    §3.2 (model-weights / context-based drafts).

* :class:`SuffixRetriever`
    Sliding buffer over prompt+generated tokens.  On query, finds the longest
    suffix of the current context that appears earlier in the buffer and
    returns the subsequent K-1 tokens as a draft chain.  This is the *PLD*
    / N-Grammys context-retrieval scheme and the building block exploited by
    REST, RACER, and Graft for their retrieval branches.

* :func:`build_hybrid_branches`
    Composes the draft sources above + the carry seed into a *multi-source*
    tree of branches.  Heterogeneous branches (retrieval / n-gram / carry)
    decorrelate the chance of any one source matching the target model's
    argmax, lifting effective ARL more than top-K of the same source can.

All sources only emit ``int`` token IDs; the verify stage is responsible for
checking each draft against the target model and only accepting tokens that
match the model's argmax.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .ngram_cache import NGramCache


# ── Suffix retrieval (PLD / N-Grammys context source) ──────────────────────

class SuffixRetriever:
    """Longest-suffix match over a rolling buffer of accepted tokens.

    The buffer initially holds the prompt; ``update`` appends accepted tokens
    after each round.  ``query(K)`` returns up to ``K-1`` continuation tokens
    drawn from the earliest occurrence of the *longest* recent-suffix match.

    Parameters
    ----------
    max_window : int
        Maximum buffer size — older tokens are dropped.  Bounds the cost of
        the suffix search.  4096 is plenty for chat-scale generations.
    min_suffix : int
        Shortest suffix length we'll trust as a match.  2 is the empirical
        sweet spot; 1 produces too many false matches (every common token
        becomes an anchor), and 3+ misses early repetitions before the
        buffer has warmed up.
    max_suffix : int
        Cap on suffix scan length per query — also bounds query cost.
    """

    __slots__ = (
        "buf", "max_window", "min_suffix", "max_suffix",
        "_np_buf", "_np_dirty",
    )

    def __init__(self, max_window: int = 4096,
                 min_suffix: int = 2, max_suffix: int = 8):
        if min_suffix < 1:
            raise ValueError("min_suffix must be >= 1")
        if max_suffix < min_suffix:
            raise ValueError("max_suffix must be >= min_suffix")
        self.buf: list[int] = []
        self.max_window = int(max_window)
        self.min_suffix = int(min_suffix)
        self.max_suffix = int(max_suffix)
        # NumPy mirror for fast vectorised queries — rebuilt lazily after
        # any mutation.  Large buckets baked once at startup query many
        # times; this avoids redoing the np.array() conversion every call.
        self._np_buf: Optional[np.ndarray] = None
        self._np_dirty: bool = True

    # ── Buffer maintenance ──────────────────────────────────────────────────

    def reset(self) -> None:
        self.buf.clear()
        self._np_dirty = True

    def _ensure_np(self) -> np.ndarray:
        if self._np_dirty or self._np_buf is None:
            self._np_buf = np.asarray(self.buf, dtype=np.int32)
            self._np_dirty = False
        return self._np_buf

    # ── Persistence ────────────────────────────────────────────────────────

    def to_state(self) -> dict:
        """Serialize buffer to a plain dict (JSON / pickle friendly)."""
        return {
            "buf": list(self.buf),
            "max_window": self.max_window,
            "min_suffix": self.min_suffix,
            "max_suffix": self.max_suffix,
        }

    @classmethod
    def from_state(cls, state: dict) -> "SuffixRetriever":
        out = cls(
            max_window=state.get("max_window", 4096),
            min_suffix=state.get("min_suffix", 2),
            max_suffix=state.get("max_suffix", 8),
        )
        out.buf = list(state.get("buf", []))
        out._np_dirty = True
        return out

    def extend(self, tokens: list[int]) -> None:
        self.buf.extend(int(t) for t in tokens)
        if len(self.buf) > self.max_window:
            # Trim from the head; preserves the recency-biased part of the
            # buffer where matches are most useful.
            self.buf = self.buf[-self.max_window:]
        self._np_dirty = True

    # ── Query ───────────────────────────────────────────────────────────────

    def query(self, current_tail: list[int], k_continuation: int
              ) -> Optional[list[int]]:
        """Return up to ``k_continuation`` tokens that historically followed
        the longest matching suffix of ``current_tail``, or ``None`` on miss.

        Vectorised PLD search.  For each candidate end-position ``i`` we
        walk backwards token-by-token counting how many tokens of
        ``current_tail`` align with the buffer leading up to ``i`` — this
        gives the per-position *suffix match length*.  We then pick the
        rightmost position whose match length is ≥ ``min_suffix``, capped
        at ``max_suffix`` (longer-match-first, with right-bias on ties).

        The work is done with NumPy on the cached ``_np_buf`` mirror — the
        previous pure-Python triple loop dominated decode wall-time once
        buffers grew past a few thousand tokens (see
        ``LOOKAHEAD_COT_RESULTS.md`` §13).
        """
        if not current_tail or k_continuation <= 0 or not self.buf:
            return None

        n_buf = len(self.buf)
        # We need at least one continuation token after the match end.
        max_L = min(self.max_suffix, len(current_tail), n_buf - 1)
        if max_L < self.min_suffix:
            return None

        buf = self._ensure_np()
        # `prefix` walks the suffix of current_tail from the most recent
        # token outward: prefix[0] = last tail token, prefix[1] = second-
        # to-last, etc.  We need it as an int32 ndarray for == comparisons.
        prefix = np.asarray(current_tail[-max_L:][::-1], dtype=np.int32)

        # Match-length array: match[i] = number of consecutive tokens
        # immediately ending at position i that agree with the most-recent
        # tokens of current_tail.  We build it via shifted equality scans.
        #
        # Position i is "the index of the last buffer token consumed by a
        # candidate match", so the continuation starts at i+1.  We bound
        # i to ``[max_L-1, n_buf-2]`` so (a) we have ``max_L`` preceding
        # tokens to inspect and (b) we have room for ≥ 1 continuation tok.

        # Candidate end-position range:
        last = n_buf - 1                # last valid end-pos (need 1 cont tok)
        first = max_L - 1               # need max_L tokens before+including end
        if last < first:
            return None
        n_pos = last - first + 1        # positions to score

        # cur_match[k]: count of matches at end-pos (first+k) up to current
        # comparison depth d.  We use uint16 (max_suffix ≤ 64K which is huge
        # for what this thing is used for).
        cur_match = np.zeros(n_pos, dtype=np.uint16)
        active = np.ones(n_pos, dtype=bool)
        # Best match length found per position; equals cur_match when active.
        best = np.zeros(n_pos, dtype=np.uint16)

        # End-positions vector (i values).
        end_pos = np.arange(first, last + 1, dtype=np.int64)

        for d in range(max_L):
            # Token at depth d (counted backward from end position) is
            # buf[end_pos - d].  Compare to prefix[d].
            tok = buf[end_pos - d]
            eq = (tok == prefix[d])
            # Positions still on a streak.
            active &= eq
            cur_match += active.astype(np.uint16)
            # Once we've gone past min_suffix, every active streak's depth
            # equals d+1 ≥ min_suffix → tracked in best.
            best = np.where(active, cur_match, best)
            if not active.any():
                # No streak can grow further; pruning saves work when the
                # buffer has few first-token matches.
                break

        # Filter to positions whose best match meets min_suffix.
        good = best >= self.min_suffix
        if not good.any():
            return None

        # Among those, the *longest* match wins; ties broken by recency
        # (largest end_pos).  argmax on (best, end_pos) ordering:
        good_idx = np.flatnonzero(good)
        best_vals = best[good_idx]
        max_len = best_vals.max()
        # Subset to longest-match positions, then pick the latest.
        winners = good_idx[best_vals == max_len]
        i_pick = int(end_pos[winners[-1]])
        cont = self.buf[i_pick + 1: i_pick + 1 + k_continuation]
        if not cont:
            return None
        return [int(t) for t in cont]


# ── Multi-source branch composition ─────────────────────────────────────────

def _ngram_chain(
    cache: NGramCache,
    history_tail: list[int],
    prev_token: int,
    K_minus_1: int,
    fallback_seed: int,
) -> tuple[Optional[list[int]], int]:
    """Build a length-``K-1`` chain by walking ``cache`` greedily.

    Returns ``(chain, hits)`` or ``(None, 0)`` when the very first key misses
    (cold cache or insufficient history).  Subsequent misses inside the chain
    are filled with ``fallback_seed`` and do *not* abort the chain — the
    verifier will simply truncate at the first mismatch.
    """
    L = cache.key_len
    if L <= 0:
        return None, 0
    ctx: list[int] = list(history_tail[-L:])
    ctx.append(int(prev_token))
    if len(ctx) > L:
        ctx = ctx[-L:]
    if len(ctx) < L:
        return None, 0
    first = cache.query(tuple(ctx))
    if first is None:
        return None, 0
    chain: list[int] = [int(first)]
    hits = 1
    ctx.append(int(first))
    if len(ctx) > L:
        ctx = ctx[-L:]
    for _ in range(K_minus_1 - 1):
        tok = cache.query(tuple(ctx)) if len(ctx) == L else None
        if tok is None:
            tok = int(fallback_seed)
        else:
            hits += 1
        chain.append(int(tok))
        ctx.append(int(tok))
        if len(ctx) > L:
            ctx = ctx[-L:]
    return chain, hits


def build_hybrid_branches(
    K: int,
    prev_token: int,
    history_tail: list[int],
    ngram: Optional[NGramCache],
    retriever: Optional[SuffixRetriever],
    fallback_seed: int,
    num_branches: int,
    *,
    cot_ngram: Optional[NGramCache] = None,
    cot_retriever: Optional[SuffixRetriever] = None,
    cot_retriever_max_len: int = 5,
) -> tuple[list[list[int]], int]:
    """Build ``num_branches`` heterogeneous guess sequences (length K-1 each).

    Branch priority order (when sources are available):
        0. Runtime suffix retriever (longest-suffix over prompt+generated)
        1. COT suffix retriever     (first ``cot_retriever_max_len`` tokens)
           completed by COT n-gram for the remaining positions — combines
           long-match precision of retrieval with n-gram's adaptive chain.
        2. COT n-gram top-1 chain   (pre-warmed, frequency-ranked)
        3. N-gram top-1 chain       (history-derived MRU)
        4. Carry-only (repeat fallback_seed)
        5..  N-gram top-2..top-k chains

    When a source returns ``None`` (cold cache / no match), we fall through
    to the next source.  Any leftover branches duplicate branch 0 (a no-op
    extra verify is cheap relative to a fresh forward, and at least gives
    us per-round "is the cache enough?" signal in ``branch_wins``).
    """
    if num_branches < 1:
        raise ValueError("num_branches must be >= 1")

    K_minus_1 = K - 1
    if K_minus_1 == 0:
        return [[]], 0

    # Slot generators in priority order.
    candidates: list[list[int]] = []
    hits = 0

    full_tail: Optional[list[int]] = None

    def _ensure_tail() -> list[int]:
        nonlocal full_tail
        if full_tail is None:
            full_tail = list(history_tail) + [int(prev_token)]
        return full_tail

    # 0. Runtime suffix retriever
    if retriever is not None:
        cont = retriever.query(_ensure_tail(), K_minus_1)
        if cont is not None:
            while len(cont) < K_minus_1:
                cont.append(int(fallback_seed))
            candidates.append(cont)
            hits += len(cont)

    # 1. COT suffix retriever — first ``cot_retriever_max_len`` tokens from a
    # longest-suffix match, then completed by COT n-gram for positions beyond
    # that length.  Asking the retriever for only N tokens (not K-1) keeps it
    # precise; the n-gram's adaptive chain fills the tail so the overall branch
    # degrades gracefully to pure n-gram when there is no retriever match.
    if cot_retriever is not None:
        _rmax = min(cot_retriever_max_len, K_minus_1)
        rtr_cont = cot_retriever.query(_ensure_tail(), _rmax)
        if rtr_cont is not None:
            hits += len(rtr_cont)
            # Extend with cot_ngram from the last retriever token if available.
            if cot_ngram is not None and len(rtr_cont) < K_minus_1:
                L = cot_ngram.key_len
                ng_ctx: list[int] = (_ensure_tail() + rtr_cont)[-L:]
                if len(ng_ctx) == L:
                    for _ in range(K_minus_1 - len(rtr_cont)):
                        tok = cot_ngram.query(tuple(ng_ctx))
                        if tok is None:
                            tok = int(fallback_seed)
                        else:
                            hits += 1
                        rtr_cont.append(int(tok))
                        ng_ctx = (ng_ctx + [int(tok)])[-L:]
            while len(rtr_cont) < K_minus_1:
                rtr_cont.append(int(fallback_seed))
            candidates.append(rtr_cont)

    # 2. COT n-gram chain — pre-warmed from the training corpus and
    # frequency-ranked; tried after retrievers so longest-match wins first.
    if cot_ngram is not None:
        cot_chain, cot_hits = _ngram_chain(
            cot_ngram, history_tail, prev_token, K_minus_1, fallback_seed,
        )
        if cot_chain is not None:
            candidates.append(cot_chain)
            hits += cot_hits

    # 3. N-gram top-1 chain
    if ngram is not None and ngram.key_len > 0:
        ng_chain, ng_hits = _ngram_chain(
            ngram, history_tail, prev_token, K_minus_1, fallback_seed,
        )
        if ng_chain is None:
            # First key missed — still seed something so verify isn't empty.
            ng_chain = [int(fallback_seed)] * K_minus_1
        else:
            hits += ng_hits
        candidates.append(ng_chain)

    # 4. Carry-only
    candidates.append([int(fallback_seed)] * K_minus_1)

    # 5..  N-gram top-2..top-k chains (if num_branches > len(candidates))
    if ngram is not None and ngram.key_len > 0 and num_branches > len(candidates):
        L = ngram.key_len
        ctx_base: list[int] = list(history_tail[-L:])
        ctx_base.append(int(prev_token))
        if len(ctx_base) > L:
            ctx_base = ctx_base[-L:]
        top = ngram.query_topk(tuple(ctx_base), num_branches) if len(ctx_base) == L else []
        # Skip top[0] (already used by branch 3).
        for seed in top[1:]:
            if len(candidates) >= num_branches:
                break
            chain = [int(seed)]
            hits += 1
            ctx = list(ctx_base) + [int(seed)]
            if len(ctx) > L:
                ctx = ctx[-L:]
            for _ in range(K_minus_1 - 1):
                tok = ngram.query(tuple(ctx)) if len(ctx) == L else None
                if tok is None:
                    tok = int(fallback_seed)
                else:
                    hits += 1
                chain.append(int(tok))
                ctx.append(int(tok))
                if len(ctx) > L:
                    ctx = ctx[-L:]
            candidates.append(chain)

    # Trim to num_branches; pad short by repeating branch 0 (cheapest filler).
    branches = candidates[:num_branches]
    while len(branches) < num_branches:
        branches.append(list(branches[0]))
    return branches, hits
