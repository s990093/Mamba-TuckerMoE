"""Runtime glue for the COT-dataset-derived caches.

The :mod:`bake_cot_caches` script writes a pickle containing per-bucket,
phase-aware caches:

* ``think[bucket]`` / ``final[bucket]`` — :class:`NGramCache` snapshots for
  the ``<think>`` and ``<final>`` slices of every training sample in that
  bucket;
* ``retrievers[bucket]`` — :class:`SuffixRetriever` snapshots holding the
  bucket's concatenated assistant-side token stream (think+final), so the
  decoder can pull whole K-1-token continuations on a longest-suffix match;
* ``markers``  — the four special-token ids that delimit the assistant turn.

This module reconstructs that bundle and exposes :class:`CoTPhaseTracker`,
a tiny observer that watches the emitted token stream, flips the active
phase between ``think`` / ``final`` / ``other`` and returns the n-gram
cache + retriever that should currently feed the draft slots.

Both decoders (greedy ``jacobi`` and SJD ``jacobi_sampling``) accept the
``cot_caches`` argument plus an optional ``cot_bucket`` that selects which
per-bucket cache to draw from.  The legacy single-bucket ``v1`` pkl layout
is still recognised: it loads as a single ``__default__`` bucket and
``active_retriever`` always returns ``None``.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Optional, Union

from .drafts import SuffixRetriever
from .ngram_cache import NGramCache


CoTCachesArg = Union[str, Path, dict, "CoTCacheBundle", None]

# Bucket name used as the single namespace when loading a v1 (pre-bucket)
# pickle, or when the caller doesn't supply a routing bucket.
_LEGACY_BUCKET = "__default__"


class CoTCacheBundle:
    """In-memory representation of a baked COT-cache pickle.

    Holds:

    * ``think_all`` / ``final_all`` — *global* n-gram caches aggregated over
      every training sample.  These are the primary draft source: their
      key coverage (~300k–800k entries) dwarfs any single bucket's, so
      smaller buckets (e.g. math_drill with 200 samples) don't suffer from
      per-bucket sparsity.
    * ``think`` / ``final`` — per-bucket n-gram caches, available for
      callers that explicitly want bucket-local stats.  Not used by the
      default draft path.
    * ``retrievers`` — per-bucket :class:`SuffixRetriever` snapshots,
      populated with the bucket's concatenated assistant token stream.
      Returned via :meth:`get_caches`.
    * ``markers`` — special-token IDs delimiting think/final blocks.

    Use :meth:`get_caches` or :meth:`make_tracker` to pull the runtime
    objects you need; construct via :func:`load_cot_caches`.
    """

    __slots__ = (
        "think", "final", "retrievers", "think_all", "final_all",
        "markers", "buckets", "default_bucket",
    )

    def __init__(
        self,
        think: dict[str, NGramCache],
        final: dict[str, NGramCache],
        retrievers: dict[str, SuffixRetriever],
        markers: dict[str, int],
        buckets: list[str],
        default_bucket: str,
        think_all: Optional[NGramCache] = None,
        final_all: Optional[NGramCache] = None,
    ):
        self.think = think
        self.final = final
        self.retrievers = retrievers
        self.think_all = think_all
        self.final_all = final_all
        self.markers = markers
        self.buckets = list(buckets)
        self.default_bucket = default_bucket

    def resolve_bucket(self, bucket: Optional[str]) -> str:
        """Map a caller-supplied bucket name to one we actually hold.

        Falls back to ``self.default_bucket`` (and then to the first known
        bucket) when the requested name isn't baked.
        """
        if bucket is not None and bucket in self.think:
            return bucket
        if self.default_bucket in self.think:
            return self.default_bucket
        if self.buckets:
            return self.buckets[0]
        return _LEGACY_BUCKET

    def get_caches(
        self,
        bucket: Optional[str],
        prefer_global_ngram: bool = True,
    ) -> tuple[Optional[NGramCache], Optional[NGramCache], Optional[SuffixRetriever]]:
        """Return ``(think_cache, final_cache, retriever)`` for ``bucket``.

        When ``prefer_global_ngram`` is true and the bundle was baked with
        global ngram caches available, those override the per-bucket
        ngrams.  The retriever is always bucket-specific.
        """
        b = self.resolve_bucket(bucket)
        if prefer_global_ngram and self.think_all is not None and self.final_all is not None:
            think_cache: Optional[NGramCache] = self.think_all
            final_cache: Optional[NGramCache] = self.final_all
        else:
            think_cache = self.think.get(b)
            final_cache = self.final.get(b)
        return (
            think_cache,
            final_cache,
            self.retrievers.get(b),
        )

    def make_tracker(
        self,
        bucket: Optional[str],
        initial_phase: str = "other",
    ) -> "CoTPhaseTracker":
        think, final, retr = self.get_caches(bucket)
        return CoTPhaseTracker(
            markers=self.markers,
            think_cache=think,
            final_cache=final,
            retriever=retr,
            initial=initial_phase,
        )


def _load_v2(state: dict) -> CoTCacheBundle:
    required = {"think", "final", "retrievers", "markers"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"COT cache v2 bundle missing keys: {missing}")
    think_states = state["think"]
    final_states = state["final"]
    retr_states  = state["retrievers"]
    if not (isinstance(think_states, dict)
            and isinstance(final_states, dict)
            and isinstance(retr_states, dict)):
        raise ValueError(
            "COT cache v2 expects dict-valued 'think', 'final', 'retrievers'."
        )

    buckets = list(state.get("buckets") or sorted(think_states.keys()))
    default_bucket = state.get("default_bucket") or (
        buckets[0] if buckets else _LEGACY_BUCKET
    )

    think  = {b: NGramCache.from_state(think_states[b]) for b in buckets if b in think_states}
    final  = {b: NGramCache.from_state(final_states[b]) for b in buckets if b in final_states}
    retr   = {b: SuffixRetriever.from_state(retr_states[b]) for b in buckets if b in retr_states}

    # Global aggregated think/final caches are baked alongside the per-
    # bucket ones; they are used as the *primary* ngram source because the
    # aggregate has much higher key coverage than any single small bucket.
    # The per-bucket caches remain available via ``CoTCacheBundle.per_bucket_*``
    # for callers that explicitly want bucket-local statistics.
    think_all = (
        NGramCache.from_state(state["think_all"])
        if "think_all" in state else None
    )
    final_all = (
        NGramCache.from_state(state["final_all"])
        if "final_all" in state else None
    )

    markers = {k: int(v) for k, v in state["markers"].items()}
    _validate_markers(markers)
    return CoTCacheBundle(
        think=think, final=final, retrievers=retr,
        think_all=think_all, final_all=final_all,
        markers=markers, buckets=buckets, default_bucket=default_bucket,
    )


def _load_v1(state: dict) -> CoTCacheBundle:
    try:
        think = NGramCache.from_state(state["think"])
        final = NGramCache.from_state(state["final"])
        markers = {k: int(v) for k, v in state["markers"].items()}
    except KeyError as exc:
        raise ValueError(
            f"COT cache bundle missing required key: {exc}"
        ) from None
    _validate_markers(markers)
    return CoTCacheBundle(
        think={_LEGACY_BUCKET: think},
        final={_LEGACY_BUCKET: final},
        retrievers={},
        think_all=think,
        final_all=final,
        markers=markers,
        buckets=[_LEGACY_BUCKET],
        default_bucket=_LEGACY_BUCKET,
    )


def _validate_markers(markers: dict[str, int]) -> None:
    required = {"think_open", "think_close", "final_open", "final_close"}
    if set(markers) != required:
        raise ValueError(
            f"COT cache markers must be exactly {required}, got {set(markers)}"
        )


def load_cot_caches(source: CoTCachesArg) -> CoTCacheBundle:
    """Load a baked COT-cache bundle (v1 or v2 layout).

    Accepts a path to a pickle written by :mod:`bake_cot_caches`, an
    already-deserialised dict in the same shape, or an existing
    :class:`CoTCacheBundle` (in which case it is returned unchanged).
    """
    if source is None:
        raise ValueError("source must not be None")
    if isinstance(source, CoTCacheBundle):
        return source
    if isinstance(source, dict):
        state = source
    else:
        path = Path(source)
        with open(path, "rb") as f:
            state = pickle.load(f)

    is_v2 = (
        state.get("version") == 2
        or isinstance(state["think"], dict) and "n" not in state["think"]
    )
    if is_v2:
        return _load_v2(state)
    return _load_v1(state)


class CoTPhaseTracker:
    """Observe the emitted token stream and surface the active caches.

    A turn moves through these states:

      * ``other``  — outside any think / final block (e.g. before the first
        ``<think>`` opens or after ``</final>`` closes).
      * ``think``  — between ``<think>`` and ``</think>``.
      * ``final``  — between ``<final>`` and ``</final>``.

    On construction the tracker is given an *initial* phase guess (usually
    derived from the prompt — see :func:`infer_initial_phase`).  After each
    decoded token batch, call :meth:`observe` with the freshly emitted ids
    so the tracker can flip phases on the marker tokens.

    :meth:`active_cache` returns the phase-appropriate n-gram cache (or
    ``None`` when no cache is registered for the current phase, in which
    case the cot_ngram draft slot stays empty for that round).  The
    retriever is phase-independent (it holds the bucket's concatenated
    assistant stream) and exposed via :meth:`active_retriever`.
    """

    __slots__ = ("_markers", "_think", "_final", "_retriever", "phase")

    def __init__(
        self,
        markers: dict[str, int],
        think_cache: Optional[NGramCache],
        final_cache: Optional[NGramCache],
        retriever: Optional[SuffixRetriever] = None,
        initial: str = "other",
    ):
        _validate_markers(markers)
        if initial not in ("think", "final", "other"):
            raise ValueError(f"unknown initial phase: {initial!r}")
        self._markers = markers
        self._think = think_cache
        self._final = final_cache
        self._retriever = retriever
        self.phase = initial

    def active_cache(self) -> Optional[NGramCache]:
        if self.phase == "think":
            return self._think
        if self.phase == "final":
            return self._final
        return None

    def active_retriever(self) -> Optional[SuffixRetriever]:
        return self._retriever

    def observe(self, tokens: Iterable[int]) -> None:
        """Advance phase given freshly emitted token ids."""
        tho = self._markers["think_open"]
        thc = self._markers["think_close"]
        fno = self._markers["final_open"]
        fnc = self._markers["final_close"]
        for raw in tokens:
            t = int(raw)
            if t == thc:
                self.phase = "other"
            elif t == fnc:
                self.phase = "other"
            elif t == tho:
                self.phase = "think"
            elif t == fno:
                self.phase = "final"


def infer_initial_phase(
    prompt_ids: Iterable[int],
    markers: dict[str, int],
) -> str:
    """Walk the prompt and return the phase the *next* decoded token starts in.

    The standard CoT prompt for assisted decoding is::

        <|im_start|>assistant
        <think>

    so the very first decoded token is inside the think block.  Callers may
    supply prompts that pre-seed past ``</think>`` / into ``<final>`` — this
    helper handles both transparently by scanning for the last marker.
    """
    tho = markers["think_open"]
    thc = markers["think_close"]
    fno = markers["final_open"]
    fnc = markers["final_close"]
    phase = "other"
    for raw in prompt_ids:
        t = int(raw)
        if t == thc or t == fnc:
            phase = "other"
        elif t == tho:
            phase = "think"
        elif t == fno:
            phase = "final"
    return phase
