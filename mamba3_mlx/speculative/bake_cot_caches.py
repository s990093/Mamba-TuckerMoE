"""Offline COT-dataset n-gram + retriever baker.

Differs from :mod:`bake_cache` (which runs the *model* to produce sample
sequences and ingests them into a single cache) — this script directly
reads the training-format Chain-of-Thought corpus and produces **per-bucket,
phase-aware** caches:

* ``think[bucket]``       — n-grams collected from every sample's ``cot``
  text (between ``<think>\\n`` and ``\\n</think>``) within ``bucket``.
* ``final[bucket]``       — n-grams collected from every sample's ``output``
  text (between ``<final>\\n`` and ``\\n</final>``) within ``bucket``.
* ``retrievers[bucket]``  — :class:`SuffixRetriever` populated with the
  concatenated assistant token streams (think + final) of every sample in
  ``bucket``.  Right-to-left longest-suffix matching pulls back **whole
  K-1 token continuations** in one shot.  This is the *PLD / REST / RACER*
  retrieval scheme and is the main ARL booster on structured outputs.

At decode time the runner switches between them based on (a) which
special-token phase the generator is currently in and (b) which bucket the
caller has tagged the prompt with (see ``jacobi.py`` /
``jacobi_sampling.py`` ``cot_caches`` + ``cot_bucket`` parameters).

Data source
-----------
JSON mode is the default — we need each sample's ``sys_bucket`` to route
think/final/retriever entries to the correct bucket cache.  ``--src-dir``
points at the directory containing the canonical CoT JSON files (default:
repo ``cot_dataset/``).  Bin mode is preserved for the legacy
single-bucket layout but is mutually exclusive with ``--per_bucket``.

Frequency-weighted n-gram selection
-----------------------------------
Per (n-1)-token key we count *every* observed continuation across the whole
bucket, then insert the top ``max_continuations`` of them in reverse order
so the most-frequent continuation ends up MRU.

Retriever buffer policy
-----------------------
The retriever stores a rolling token window.  When the corpus is larger
than ``--retriever_max_window``, we drop *earlier* samples so the buffer
ends with the most-recently-seen samples — but for this offline baker, all
samples are equivalent and we simply keep the tail of the concatenation
(NB: order doesn't materially affect query quality because the recency
prior runs against insertion order, and all corpus samples share structure).

Output pickle (v2)
------------------
``{
    "version": 2,
    "think":      {bucket: NGramCache.to_state(), ...},
    "final":      {bucket: NGramCache.to_state(), ...},
    "retrievers": {bucket: SuffixRetriever.to_state(), ...},
    "markers": {
        "think_open": int, "think_close": int,
        "final_open": int, "final_close": int,
    },
    "buckets":        [list of bucket names],
    "default_bucket": "daily_conversation",
    "args": {ngram_n, max_entries, n_samples, ...},
}``

Run (JSON mode + per-bucket — recommended)::

    .venv/bin/python3 -m mamba3_mlx.speculative.bake_cot_caches \\
        --ngram_n 4 --max_continuations 4 \\
        --retriever_max_window 32768 \\
        --retriever_min_suffix 4 --retriever_max_suffix 12 \\
        --out mamba3_mlx/speculative/cot_caches_v2.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
COT_DIR = REPO_ROOT / "cot_dataset"

# Default .bin path — produced by stf_cot_to_bin.py
_DEFAULT_BIN = (
    REPO_ROOT
    / "pre-train"
    / "sft_cot_bundle"
    / "dataset"
    / "stf_cot_train.bin"
)
_DEFAULT_TOKENIZER = COT_DIR / "tokenizer.json"
_DEFAULT_BUCKET = "daily_conversation"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(COT_DIR))

from mamba3_mlx.speculative.drafts import SuffixRetriever  # noqa: E402
from mamba3_mlx.speculative.ngram_cache import NGramCache  # noqa: E402

_MARKER_TOKENS = ("<think>", "</think>", "<final>", "</final>")


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def _resolve_markers(tok: Tokenizer) -> dict[str, int]:
    """Return a mapping from marker string to token id."""
    out: dict[str, int] = {}
    for name in _MARKER_TOKENS:
        tid = tok.token_to_id(name)
        if tid is None:
            raise SystemExit(
                f"Tokenizer is missing special token {name!r}. "
                "This baker expects the CoT special tokens to be present."
            )
        out[name] = int(tid)
    return out


# ---------------------------------------------------------------------------
# Slice helpers — shared between bin-mode and JSON-mode
# ---------------------------------------------------------------------------

def _slice_by_markers(
    ids: list[int] | np.ndarray,
    markers: dict[str, int],
) -> tuple[list[int] | None, list[int] | None]:
    """Return (think_ids, final_ids) for a tokenised assistant turn.

    Both slices include their closing marker so the cache learns the
    transition tokens; the opening marker is dropped (the decoder is
    already past it when the corresponding phase begins).
    """
    ids = list(ids)
    try:
        to = ids.index(markers["<think>"])
        tc = ids.index(markers["</think>"], to + 1)
        fo = ids.index(markers["<final>"], tc + 1)
        fc = ids.index(markers["</final>"], fo + 1)
    except ValueError:
        return None, None
    return ids[to + 1: tc + 1], ids[fo + 1: fc + 1]


# ---------------------------------------------------------------------------
# N-gram counting / cache building
# ---------------------------------------------------------------------------

def _count_ngrams(
    seq: list[int],
    n: int,
    counts: dict[tuple[int, ...], dict[int, int]],
) -> None:
    """Accumulate ``counts[key][tok] += 1`` for every length-``n`` n-gram."""
    L = n - 1
    if L <= 0 or len(seq) < n:
        return
    for i in range(L, len(seq)):
        key = tuple(int(t) for t in seq[i - L: i])
        tok = int(seq[i])
        bucket = counts.get(key)
        if bucket is None:
            counts[key] = {tok: 1}
        else:
            bucket[tok] = bucket.get(tok, 0) + 1


def _build_cache(
    counts: dict[tuple[int, ...], dict[int, int]],
    n: int,
    max_continuations: int,
) -> NGramCache:
    """Materialise the counted n-grams into an NGramCache.

    For each key we keep the top ``max_continuations`` tokens by frequency
    and insert them in *reverse* frequency order so the most-frequent
    continuation ends up MRU (the slot returned by ``cache.query``).
    ``max_entries`` is sized to fit every observed key with a small slack.
    """
    cache = NGramCache(
        n=n,
        max_entries=max(len(counts) + 256, 8192),
        max_continuations=max_continuations,
    )
    for key, tok_counts in counts.items():
        top = sorted(tok_counts.items(), key=lambda kv: -kv[1])[:max_continuations]
        for tok, _freq in reversed(top):
            cache.update_ngrams([key + (tok,)])
    return cache


def _build_retriever(
    bucket_token_stream: list[int],
    max_window: int,
    min_suffix: int,
    max_suffix: int,
) -> SuffixRetriever:
    """Populate a SuffixRetriever with the per-bucket assistant-side stream.

    ``extend`` already trims to ``max_window`` from the head — the tail
    (most recent samples) is preserved, which is also the part the
    right-to-left scan visits first.
    """
    r = SuffixRetriever(
        max_window=max_window,
        min_suffix=min_suffix,
        max_suffix=max_suffix,
    )
    r.extend(bucket_token_stream)
    return r


# ---------------------------------------------------------------------------
# Data-ingestion modes
# ---------------------------------------------------------------------------

def _iter_slices_from_bin(
    bin_path: Path,
    markers: dict[str, int],
    max_samples: int,
) -> tuple[list[tuple[list[int], list[int]]], int]:
    """Read a flat uint16 token stream and extract (think, final) slice pairs.

    Bin mode has no sys_bucket information — every slice is assigned to the
    default bucket downstream when ``--per_bucket`` is on.
    """
    print(
        f"[bake-cot] reading bin: {bin_path}  "
        f"({bin_path.stat().st_size / 1024 / 1024:.1f} MB)",
        file=sys.stderr,
    )
    arr: np.ndarray = np.fromfile(str(bin_path), dtype=np.uint16)
    print(f"[bake-cot] loaded {len(arr):,} tokens from bin", file=sys.stderr)

    think_open  = markers["<think>"]
    think_close = markers["</think>"]
    final_open  = markers["<final>"]
    final_close = markers["</final>"]

    ids = arr.tolist()  # list[int] for fast .index()
    total = len(ids)
    slices: list[tuple[list[int], list[int]]] = []
    n_skipped = 0
    pos = 0

    while pos < total:
        try:
            to = ids.index(think_open, pos)
        except ValueError:
            break
        try:
            tc = ids.index(think_close, to + 1)
        except ValueError:
            n_skipped += 1
            pos = to + 1
            continue
        try:
            fo = ids.index(final_open, tc + 1)
        except ValueError:
            n_skipped += 1
            pos = tc + 1
            continue
        try:
            fc = ids.index(final_close, fo + 1)
        except ValueError:
            n_skipped += 1
            pos = fo + 1
            continue

        think_ids = ids[to + 1: tc + 1]
        final_ids = ids[fo + 1: fc + 1]

        if think_ids or final_ids:
            slices.append((think_ids, final_ids))

        pos = fc + 1

        if max_samples > 0 and len(slices) >= max_samples:
            print(
                f"[bake-cot] reached --max_samples {max_samples}, stopping early",
                file=sys.stderr,
            )
            break

        if len(slices) % 2000 == 0 and len(slices) > 0:
            print(
                f"[bake-cot] scanned {len(slices)} samples  pos={pos}/{total}",
                file=sys.stderr,
            )

    return slices, n_skipped


def _iter_slices_from_json(
    src_dir: Path,
    files_arg: str,
    tok: Tokenizer,
    markers: dict[str, int],
    allow_missing: bool,
    duplicate_policy: str,
    dedupe_by_content: bool,
    max_samples: int,
) -> tuple[list[tuple[str, list[int], list[int]]], int]:
    """JSON mode — tag every slice with its ``sys_bucket``.

    Returns a list of ``(bucket, think_ids, final_ids)`` tuples.
    """
    from cot_dataset.export_hf_dataset import (  # type: ignore[import]
        discover_json_specs,
        iter_records,
    )

    if files_arg.strip().lower() == "auto":
        file_list = discover_json_specs(src_dir.resolve())
    else:
        file_list = [x.strip() for x in files_arg.split(",") if x.strip()]
    print(f"[bake-cot] {len(file_list)} JSON specs", file=sys.stderr)

    rows, missing, _, invalid_msgs, dup_content = iter_records(
        src_dir.resolve(),
        file_list,
        allow_missing=allow_missing,
        duplicate_policy=duplicate_policy,
        dedupe_by_content=dedupe_by_content,
    )
    if missing:
        print(f"[bake-cot] WARNING: missing specs skipped: {missing}", file=sys.stderr)
    if invalid_msgs:
        print(
            f"[bake-cot] WARNING: {len(invalid_msgs)} rows dropped due to "
            f"forbidden special tokens (first: {invalid_msgs[0]})",
            file=sys.stderr,
        )
    if not rows:
        raise SystemExit("[bake-cot] no rows after filtering")
    if max_samples > 0:
        rows = rows[:max_samples]
    print(
        f"[bake-cot] {len(rows)} samples (dedupe_by_content removed {dup_content})",
        file=sys.stderr,
    )

    slices: list[tuple[str, list[int], list[int]]] = []
    n_skipped = 0
    for i, row in enumerate(rows):
        ids = tok.encode(row["text"], add_special_tokens=False).ids
        think_ids, final_ids = _slice_by_markers(ids, markers)
        if think_ids is None or final_ids is None:
            n_skipped += 1
            continue
        bucket = str(row.get("sys_bucket") or _DEFAULT_BUCKET)
        slices.append((bucket, think_ids, final_ids))
        if (i + 1) % 2000 == 0:
            print(f"[bake-cot] encoded {i + 1}/{len(rows)}", file=sys.stderr)

    return slices, n_skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Build per-bucket phase-aware CoT n-gram + retriever caches.",
    )

    # --- data source ---
    src_grp = p.add_argument_group("data source (choose one)")
    src_grp.add_argument(
        "--bin",
        type=Path,
        default=_DEFAULT_BIN,
        metavar="PATH",
        help=(
            "Pre-tokenised uint16 flat token stream produced by stf_cot_to_bin.py. "
            "Used only when --json-mode is OFF and --per_bucket is OFF. "
            f"Default: {_DEFAULT_BIN}"
        ),
    )
    src_grp.add_argument(
        "--json-mode",
        action="store_true",
        help=(
            "Use JSON + re-tokenise path (required for --per_bucket). "
            "Default behaviour when --per_bucket is on."
        ),
    )
    src_grp.add_argument(
        "--src-dir",
        type=Path,
        default=COT_DIR,
        metavar="DIR",
        help="(JSON mode) directory containing CoT JSON files.",
    )
    src_grp.add_argument(
        "--files",
        type=str,
        default="auto",
        help="(JSON mode) 'auto' or comma-separated file/dir/glob list.",
    )

    # --- tokenizer ---
    p.add_argument(
        "--tokenizer_path",
        type=Path,
        default=_DEFAULT_TOKENIZER,
        help=f"tokenizers Tokenizer JSON. Default: {_DEFAULT_TOKENIZER}",
    )

    # --- output ---
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "mamba3_mlx" / "speculative" / "cot_caches_v2.pkl",
    )

    # --- per-bucket / retriever toggles ---
    p.add_argument(
        "--per_bucket",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build per-sys_bucket caches (JSON mode required). "
             "Disable with --no-per_bucket to write a single-bucket v1-shape pkl.",
    )

    # --- cache hyperparams ---
    p.add_argument(
        "--ngram_n", type=int, default=4,
        help="Total n-gram length; key = n-1 tokens.",
    )
    p.add_argument(
        "--max_continuations", type=int, default=4,
        help="Continuations stored per (n-1)-token key (top-K by frequency).",
    )
    p.add_argument(
        "--max_samples", type=int, default=0,
        help="0 = use all samples; else stop after this many.",
    )

    # --- retriever hyperparams ---
    p.add_argument(
        "--retriever_max_window", type=int, default=32768,
        help="Maximum buffer size (tokens) per bucket retriever.",
    )
    p.add_argument(
        "--retriever_min_suffix", type=int, default=4,
        help="Shortest suffix length the retriever will trust as a match.",
    )
    p.add_argument(
        "--retriever_max_suffix", type=int, default=12,
        help="Maximum suffix length the retriever scans for per query.",
    )
    p.add_argument(
        "--default_bucket", type=str, default=_DEFAULT_BUCKET,
        help="Bucket used when the caller hasn't routed the prompt to one.",
    )

    # --- JSON-mode-only flags ---
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument(
        "--duplicate-policy", default="keep-first",
        choices=("error", "keep-first", "keep-last", "keep-all"),
    )
    p.add_argument("--dedupe-by-content", action="store_true")

    args = p.parse_args()

    if args.per_bucket and not args.json_mode:
        print(
            "[bake-cot] --per_bucket implies JSON mode (need sys_bucket per sample); "
            "switching to JSON mode.",
            file=sys.stderr,
        )
        args.json_mode = True

    # ---- tokenizer ----
    print(f"[bake-cot] loading tokenizer {args.tokenizer_path}", file=sys.stderr)
    tok = Tokenizer.from_file(str(args.tokenizer_path))
    markers = _resolve_markers(tok)
    print(f"[bake-cot] marker ids: {markers}", file=sys.stderr)

    # ---- ingest ----
    t0 = time.perf_counter()

    bucketed_slices: list[tuple[str, list[int], list[int]]]

    if args.json_mode:
        print("[bake-cot] mode: JSON (re-tokenise)", file=sys.stderr)
        bucketed_slices, n_skipped = _iter_slices_from_json(
            src_dir=args.src_dir,
            files_arg=args.files,
            tok=tok,
            markers=markers,
            allow_missing=args.allow_missing,
            duplicate_policy=args.duplicate_policy,
            dedupe_by_content=args.dedupe_by_content,
            max_samples=args.max_samples,
        )
    else:
        bin_path: Path = args.bin
        if not bin_path.is_file():
            raise SystemExit(
                f"[bake-cot] bin file not found: {bin_path}\n"
                "Use --bin <path> to specify the stf_cot_train.bin file, "
                "or pass --json-mode to use the JSON data path."
            )
        print(f"[bake-cot] mode: bin  ({bin_path})", file=sys.stderr)
        flat, n_skipped = _iter_slices_from_bin(
            bin_path=bin_path,
            markers=markers,
            max_samples=args.max_samples,
        )
        bucketed_slices = [(args.default_bucket, t, f) for (t, f) in flat]

    if not bucketed_slices:
        raise SystemExit("[bake-cot] no valid (think, final) pairs found — nothing to bake")

    print(
        f"[bake-cot] {len(bucketed_slices)} valid samples, {n_skipped} skipped",
        file=sys.stderr,
    )

    # ---- count per-bucket n-grams + concatenate retriever streams ----
    think_counts_b: dict[str, dict[tuple[int, ...], dict[int, int]]] = defaultdict(dict)
    final_counts_b: dict[str, dict[tuple[int, ...], dict[int, int]]] = defaultdict(dict)
    retr_streams: dict[str, list[int]] = defaultdict(list)

    n_think_tokens = 0
    n_final_tokens = 0
    samples_per_bucket: dict[str, int] = defaultdict(int)

    for bucket, think_ids, final_ids in bucketed_slices:
        _count_ngrams(think_ids, args.ngram_n, think_counts_b[bucket])
        _count_ngrams(final_ids, args.ngram_n, final_counts_b[bucket])
        retr_streams[bucket].extend(think_ids)
        retr_streams[bucket].extend(final_ids)
        n_think_tokens += len(think_ids)
        n_final_tokens += len(final_ids)
        samples_per_bucket[bucket] += 1

    buckets = sorted(samples_per_bucket.keys())
    if args.default_bucket not in buckets:
        # Fall back to the most-represented bucket if the requested default
        # isn't present in the corpus.
        default_bucket = max(samples_per_bucket.items(), key=lambda kv: kv[1])[0]
        print(
            f"[bake-cot] WARNING: requested default_bucket={args.default_bucket!r} "
            f"absent; falling back to {default_bucket!r}",
            file=sys.stderr,
        )
    else:
        default_bucket = args.default_bucket

    think_caches: dict[str, NGramCache] = {}
    final_caches: dict[str, NGramCache] = {}
    retrievers:   dict[str, SuffixRetriever] = {}

    for b in buckets:
        think_caches[b] = _build_cache(
            think_counts_b[b], args.ngram_n, args.max_continuations,
        )
        final_caches[b] = _build_cache(
            final_counts_b[b], args.ngram_n, args.max_continuations,
        )
        retrievers[b] = _build_retriever(
            retr_streams[b],
            max_window=args.retriever_max_window,
            min_suffix=args.retriever_min_suffix,
            max_suffix=args.retriever_max_suffix,
        )

    elapsed = time.perf_counter() - t0
    print(
        f"[bake-cot] built in {elapsed:.1f}s — buckets:",
        file=sys.stderr,
    )
    for b in buckets:
        rb = retrievers[b]
        tc = think_caches[b]
        fc = final_caches[b]
        print(
            f"           {b:>22s}  samples={samples_per_bucket[b]:>5d}  "
            f"think={len(tc):>6d} keys  final={len(fc):>6d} keys  "
            f"retr_buf={len(rb.buf):>6d} tok",
            file=sys.stderr,
        )

    # ---- build global aggregated n-gram caches (used as fallback for
    # small buckets where per-bucket counts are too sparse to draft well).
    think_counts_all: dict[tuple[int, ...], dict[int, int]] = {}
    final_counts_all: dict[tuple[int, ...], dict[int, int]] = {}
    for b in buckets:
        for k, v in think_counts_b[b].items():
            tgt = think_counts_all.setdefault(k, {})
            for tok_id, cnt in v.items():
                tgt[tok_id] = tgt.get(tok_id, 0) + cnt
        for k, v in final_counts_b[b].items():
            tgt = final_counts_all.setdefault(k, {})
            for tok_id, cnt in v.items():
                tgt[tok_id] = tgt.get(tok_id, 0) + cnt
    think_all = _build_cache(think_counts_all, args.ngram_n, args.max_continuations)
    final_all = _build_cache(final_counts_all, args.ngram_n, args.max_continuations)
    print(
        f"[bake-cot] global think/final caches: "
        f"think={len(think_all):>7d} keys, final={len(final_all):>7d} keys",
        file=sys.stderr,
    )

    # ---- serialise ----
    if args.per_bucket:
        state = {
            "version": 2,
            "think":      {b: think_caches[b].to_state() for b in buckets},
            "final":      {b: final_caches[b].to_state() for b in buckets},
            "think_all":  think_all.to_state(),
            "final_all":  final_all.to_state(),
            "retrievers": {b: retrievers[b].to_state()   for b in buckets},
            "markers": {
                "think_open":  markers["<think>"],
                "think_close": markers["</think>"],
                "final_open":  markers["<final>"],
                "final_close": markers["</final>"],
            },
            "buckets":        buckets,
            "default_bucket": default_bucket,
            "args": {
                "ngram_n":              args.ngram_n,
                "max_continuations":    args.max_continuations,
                "retriever_max_window": args.retriever_max_window,
                "retriever_min_suffix": args.retriever_min_suffix,
                "retriever_max_suffix": args.retriever_max_suffix,
                "n_samples":            len(bucketed_slices),
                "n_skipped":            n_skipped,
                "n_think_tokens":       n_think_tokens,
                "n_final_tokens":       n_final_tokens,
                "samples_per_bucket":   dict(samples_per_bucket),
                "tokenizer_path":       str(args.tokenizer_path),
                "data_source":          (
                    str(args.src_dir) if args.json_mode else str(args.bin)
                ),
                "mode":                 "json" if args.json_mode else "bin",
            },
        }
    else:
        # Legacy v1 shape: single global cache (sum all buckets together).
        state = {
            "think": think_all.to_state(),
            "final": final_all.to_state(),
            "markers": {
                "think_open":  markers["<think>"],
                "think_close": markers["</think>"],
                "final_open":  markers["<final>"],
                "final_close": markers["</final>"],
            },
            "args": {
                "ngram_n":        args.ngram_n,
                "max_continuations": args.max_continuations,
                "n_samples":      len(bucketed_slices),
                "n_skipped":      n_skipped,
                "n_think_tokens": n_think_tokens,
                "n_final_tokens": n_final_tokens,
                "tokenizer_path": str(args.tokenizer_path),
                "data_source":    (
                    str(args.src_dir) if args.json_mode else str(args.bin)
                ),
                "mode":           "json" if args.json_mode else "bin",
            },
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(state, f)
    size_kb = args.out.stat().st_size / 1024
    print(
        f"[bake-cot] wrote {args.out}  ({size_kb:.1f} KB)\n"
        f"[bake-cot] summary: {json.dumps(state['args'], ensure_ascii=False)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
