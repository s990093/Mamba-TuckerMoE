"""LRU n-gram continuation cache for Jacobi guess seeding.

Maps the last (n-1) accepted tokens to a list of recently-observed
"next" tokens, MRU-first.  Used to seed positions 1..K-1 of the
Jacobi guess buffer; cache misses fall back to the carry seed.

The cache is purely a *guess source* — wrong predictions are silently
rejected by the verify step, so the cache never affects correctness,
only acceptance rate.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional


class NGramCache:
    """Bounded LRU n-gram cache.

    Parameters
    ----------
    n : int
        Total n-gram length.  Key is the last ``n-1`` tokens; the value
        is the token that historically followed.  Must be ≥ 2.
    max_entries : int
        Maximum number of distinct keys; oldest evicted first.
    max_continuations : int
        Maximum number of (MRU-ordered) continuations stored per key.
    """

    __slots__ = ("n", "key_len", "max_entries", "max_cont", "_d")

    def __init__(
        self,
        n: int = 3,
        max_entries: int = 8192,
        max_continuations: int = 4,
    ):
        if n < 2:
            raise ValueError(f"NGramCache requires n >= 2 (got {n})")
        self.n = int(n)
        self.key_len = int(n) - 1
        self.max_entries = int(max_entries)
        self.max_cont = int(max_continuations)
        self._d: "OrderedDict[tuple[int, ...], list[int]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._d)

    # ── Query ───────────────────────────────────────────────────────────────

    def query(self, key: tuple[int, ...]) -> Optional[int]:
        """Return the MRU continuation for ``key``, or ``None`` on miss."""
        lst = self._d.get(key)
        if not lst:
            return None
        self._d.move_to_end(key)
        return int(lst[0])

    def query_topk(self, key: tuple[int, ...], k: int = 4) -> list[int]:
        """Return up to ``k`` recent continuations for ``key`` (MRU first)."""
        lst = self._d.get(key)
        if not lst:
            return []
        self._d.move_to_end(key)
        return [int(x) for x in lst[:k]]

    # ── Update ──────────────────────────────────────────────────────────────

    def update_sequence(self, seq: list[int], start_idx: int = 0) -> None:
        """Insert all n-grams covering positions ``[start_idx, len(seq))``.

        The (n-1)-token context window must be fully available, so the actual
        first inserted position is ``max(start_idx, key_len)``.
        """
        L = self.key_len
        if L == 0:
            return
        start = max(int(start_idx), L)
        for i in range(start, len(seq)):
            key = tuple(int(t) for t in seq[i - L: i])
            tok = int(seq[i])
            lst = self._d.get(key)
            if lst is None:
                lst = []
                self._d[key] = lst
            if tok in lst:
                lst.remove(tok)
            lst.insert(0, tok)
            if len(lst) > self.max_cont:
                del lst[self.max_cont:]
            self._d.move_to_end(key)
            if len(self._d) > self.max_entries:
                self._d.popitem(last=False)

    def update_ngrams(self, ngrams) -> None:
        """Insert pre-formed length-``n`` n-grams directly.

        Each n-gram is split into its (n-1)-token key and continuation token,
        then merged into the cache with the same MRU/LRU bookkeeping as
        :meth:`update_sequence`.  N-grams of the wrong length are skipped.

        Used by the offline COT baker (``bake_cot_caches.py``) to push
        frequency-sorted n-grams into the cache in reverse order so the
        highest-frequency continuation ends up MRU.
        """
        L = self.key_len
        if L == 0:
            return
        for ng in ngrams:
            if len(ng) != self.n:
                continue
            key = tuple(int(t) for t in ng[:L])
            tok = int(ng[L])
            lst = self._d.get(key)
            if lst is None:
                lst = []
                self._d[key] = lst
            if tok in lst:
                lst.remove(tok)
            lst.insert(0, tok)
            if len(lst) > self.max_cont:
                del lst[self.max_cont:]
            self._d.move_to_end(key)
            if len(self._d) > self.max_entries:
                self._d.popitem(last=False)

    def reset(self) -> None:
        self._d.clear()

    # ── Persistence ────────────────────────────────────────────────────────

    def to_state(self) -> dict:
        """Serialize cache to a plain dict (JSON / pickle friendly)."""
        return {
            "n": self.n,
            "max_entries": self.max_entries,
            "max_continuations": self.max_cont,
            # Keys are tuples → convert to lists of lists for JSON.
            "entries": [(list(k), list(v)) for k, v in self._d.items()],
        }

    @classmethod
    def from_state(cls, state: dict) -> "NGramCache":
        out = cls(
            n=state.get("n", 3),
            max_entries=state.get("max_entries", 8192),
            max_continuations=state.get("max_continuations", 4),
        )
        for k, v in state.get("entries", []):
            out._d[tuple(int(x) for x in k)] = [int(x) for x in v]
        return out
