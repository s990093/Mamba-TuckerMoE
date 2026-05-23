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

    __slots__ = ("buf", "max_window", "min_suffix", "max_suffix")

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

    # ── Buffer maintenance ──────────────────────────────────────────────────

    def reset(self) -> None:
        self.buf.clear()

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
        return out

    def extend(self, tokens: list[int]) -> None:
        self.buf.extend(int(t) for t in tokens)
        if len(self.buf) > self.max_window:
            # Trim from the head; preserves the recency-biased part of the
            # buffer where matches are most useful.
            self.buf = self.buf[-self.max_window:]

    # ── Query ───────────────────────────────────────────────────────────────

    def query(self, current_tail: list[int], k_continuation: int
              ) -> Optional[list[int]]:
        """Return up to ``k_continuation`` tokens that historically followed
        the longest matching suffix of ``current_tail``, or ``None`` on miss.

        Strategy: try descending suffix lengths from ``max_suffix`` down to
        ``min_suffix``.  At each length, scan the buffer *from the right* —
        the most recent matching occurrence is likely the most relevant
        continuation (recency-biased PLD).  First hit wins.
        """
        if not current_tail or k_continuation <= 0 or not self.buf:
            return None

        max_L = min(self.max_suffix, len(current_tail), len(self.buf) - 1)
        for L in range(max_L, self.min_suffix - 1, -1):
            needle = current_tail[-L:]
            # Search from the right; need at least one continuation token,
            # so stop at len(buf) - L - 1.
            i = len(self.buf) - L - 1
            while i >= 0:
                # Manual tight loop beats list slicing here for small L.
                hit = True
                for j in range(L):
                    if self.buf[i + j] != needle[j]:
                        hit = False
                        break
                if hit:
                    end = i + L
                    cont = self.buf[end: end + k_continuation]
                    if cont:
                        return [int(t) for t in cont]
                i -= 1
        return None


# ── Multi-source branch composition ─────────────────────────────────────────

def build_hybrid_branches(
    K: int,
    prev_token: int,
    history_tail: list[int],
    ngram: Optional[NGramCache],
    retriever: Optional[SuffixRetriever],
    fallback_seed: int,
    num_branches: int,
) -> tuple[list[list[int]], int]:
    """Build ``num_branches`` heterogeneous guess sequences (length K-1 each).

    Branch priority order (when sources are available):
        0. Suffix retriever (longest-suffix continuation)
        1. N-gram top-1 chain
        2. Carry-only (repeat fallback_seed)
        3..  N-gram top-2..top-k chains

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

    # 0. Suffix retriever
    if retriever is not None:
        # Use the entire history (incl. prev_token) as the "tail" we want
        # to extend — query needs to find continuations of that tail.
        full_tail = list(history_tail) + [int(prev_token)]
        cont = retriever.query(full_tail, K_minus_1)
        if cont is not None:
            # Pad with fallback if shorter than K-1.
            while len(cont) < K_minus_1:
                cont.append(int(fallback_seed))
            candidates.append(cont)
            hits += len(cont)

    # 1. N-gram top-1 chain
    if ngram is not None and ngram.key_len > 0:
        L = ngram.key_len
        ctx: list[int] = list(history_tail[-L:])
        ctx.append(int(prev_token))
        if len(ctx) > L:
            ctx = ctx[-L:]
        chain: list[int] = []
        for _ in range(K_minus_1):
            tok: Optional[int] = None
            if len(ctx) == L:
                tok = ngram.query(tuple(ctx))
            if tok is None:
                tok = int(fallback_seed)
            else:
                hits += 1
            chain.append(int(tok))
            ctx.append(int(tok))
            if len(ctx) > L:
                ctx = ctx[-L:]
        candidates.append(chain)

    # 2. Carry-only
    candidates.append([int(fallback_seed)] * K_minus_1)

    # 3..  N-gram top-2..top-k chains (if num_branches > len(candidates))
    if ngram is not None and ngram.key_len > 0 and num_branches > len(candidates):
        L = ngram.key_len
        ctx_base: list[int] = list(history_tail[-L:])
        ctx_base.append(int(prev_token))
        if len(ctx_base) > L:
            ctx_base = ctx_base[-L:]
        top = ngram.query_topk(tuple(ctx_base), num_branches) if len(ctx_base) == L else []
        # Skip top[0] (already used by branch 1).
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
