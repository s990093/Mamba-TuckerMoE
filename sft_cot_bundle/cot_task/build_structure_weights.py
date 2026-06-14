#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2 — Structure Weights Builder (Tiered, Mask-Aware, Batch-Tokenized).

What this does
--------------
For every sample in the CoT HuggingFace dataset, scan the **assistant region**
(after `<|im_start|>assistant\\n`) of the rendered ChatML `text` field with a set
of regular expressions corresponding to TASK2_LOSS_ENGINEERING §4.2 (R1-R6).
Each pattern is assigned a *tier* (see fix.md §3.A) and a per-token weight
vector aligned with `tok.encode(text, add_special_tokens=False)` is produced:

    tier-1 (logical backbone)  : R1 step, R2 pipe          → w = 2.8
    tier-2 (formatting aid)    : R4 bold, R5 heading, R6 ``` → w = 1.3
    tier-3 (deprecated)        : R3 separator (strict)      → w = 1.0   (no boost)

Mask awareness
--------------
*   Tokens before `<|im_start|>assistant\\n` keep w = 1.0 (they are -100 anyway).
*   Special tokens reserved for FCP / structure_token_ce_weighting
    (`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, `<final>`, `</final>`,
    `[PAD]`) are **always** forced to w = 1.0 to avoid double weighting.

Output
------
*   `reports/structure_weights/<sample_id>.npz`  — per-sample weights (compat).
*   `reports/structure_weights_bundle.pt`        — single pickle with
        {"ids": List[str], "weights": List[np.float32 array], "config": {...}}
    suitable for fast load at training time.
*   `reports/structure_weights_metadata.json`    — aggregated statistics.

Usage
-----
    python build_structure_weights.py                 # all samples, default tiers
    python build_structure_weights.py --sample-count 50
    python build_structure_weights.py --no-npz        # only bundle.pt + metadata
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # llm/
SFT_COT_BUNDLE = PROJECT_ROOT / "sft_cot_bundle"
DEFAULT_HF_PATH = SFT_COT_BUNDLE / "dataset" / "stf_cot_hf"
DEFAULT_TOKENIZER = SFT_COT_BUNDLE / "dataset" / "tokenizer"
DEFAULT_OUT_DIR = SFT_COT_BUNDLE / "cot_task" / "reports"

# ============================================================================
# Regex patterns (mapped to fix.md §3.A tier system)
# ============================================================================
# (pattern, flags, tier_name)
RE_PATTERNS: Dict[str, Tuple[str, int, str]] = {
    # Tier 1 — logical backbone
    "step":         (r"Step\s*\d+[:：]?", re.IGNORECASE | re.MULTILINE, "tier1"),  # R1
    "pipe":         (r"\|",               0,                            "tier1"),  # R2
    # Tier 2 — formatting aid
    "bold":         (r"\*\*[^*\n]+\*\*",  0,                            "tier2"),  # R4
    "heading":      (r"^#+\s",            re.MULTILINE,                 "tier2"),  # R5
    "fenced_code":  (r"```",              0,                            "tier2"),  # R6
    # Tier 3 — deprecated (strict version per fix.md to avoid R2 overlap)
    "separator":    (r"\|[-:\s]+\|",      re.MULTILINE,                 "tier3"),  # R3 strict
}

DEFAULT_TIER_WEIGHTS = {
    "tier1": 2.8,
    "tier2": 1.3,
    "tier3": 1.0,
}

# Special tokens reserved for FCP / structure_token_ce_weighting (force w=1.0)
# These IDs come from the tokenizer config and are verified at load time.
RESERVED_TOKEN_IDS = {32000, 32001, 32002, 32003, 32004, 32005, 32006}


# ============================================================================
# Helpers
# ============================================================================
def load_hf_dataset(hf_path: Path) -> Dataset:
    hf_path = Path(hf_path)
    if not hf_path.exists():
        raise FileNotFoundError(f"HF dataset not found: {hf_path}")
    ds = load_from_disk(str(hf_path))
    if isinstance(ds, dict):
        if "train" in ds:
            return ds["train"]
        return ds[next(iter(ds))]
    return ds


def load_tokenizer(tok_dir: Path) -> PreTrainedTokenizerFast:
    tok = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)
    if len(tok) != 32007:
        raise ValueError(f"tokenizer total vocab should be 32007, got {len(tok)}")
    if not tok.is_fast:
        raise ValueError("tokenizer must be a fast tokenizer (offset_mapping required)")
    # Sanity check reserved IDs
    for marker, expect in [("<|im_start|>", 32000), ("<|im_end|>", 32001),
                            ("<think>", 32002), ("</think>", 32003),
                            ("<final>", 32004), ("</final>", 32005)]:
        ids = tok.encode(marker, add_special_tokens=False)
        if len(ids) != 1 or ids[0] != expect:
            raise ValueError(f"{marker} did not map to single id {expect}: {ids}")
    return tok


def find_assistant_start_char(text: str) -> int:
    """Return char offset of the first token AFTER `<|im_start|>assistant\\n`.

    Returns 0 (whole sequence) if the marker is not found.
    """
    marker = "<|im_start|>assistant\n"
    p = text.find(marker)
    if p < 0:
        return 0
    return p + len(marker)


def char_span_to_token_indices_vectorized(
    starts: np.ndarray, ends: np.ndarray, c_lo: int, c_hi: int
) -> np.ndarray:
    """All token indices whose [start,end) overlaps [c_lo, c_hi).

    starts/ends: arrays of shape [T] from offset_mapping.
    """
    # overlap: not (token_end <= c_lo or token_start >= c_hi)
    mask = ~((ends <= c_lo) | (starts >= c_hi))
    return np.nonzero(mask)[0]


# ============================================================================
# Core: per-sample weight computation
# ============================================================================
def compute_one(
    text: str,
    input_ids: np.ndarray,
    offsets: np.ndarray,            # [T, 2] of (char_start, char_end)
    patterns: Dict[str, Tuple[str, int, str]],
    tier_weights: Dict[str, float],
    w_min: float,
    w_max: float,
) -> Tuple[np.ndarray, Dict[str, int], int]:
    """Return (weights [T], pattern_hits, assistant_start_token)."""
    T = len(input_ids)
    weights = np.ones(T, dtype=np.float32)

    # 1) Locate assistant region
    asst_char = find_assistant_start_char(text)
    starts = offsets[:, 0]
    ends = offsets[:, 1]
    # token whose start >= asst_char is in assistant region
    asst_tok_arr = np.where(starts >= asst_char)[0]
    asst_start_token = int(asst_tok_arr[0]) if asst_tok_arr.size > 0 else T

    pattern_hits: Dict[str, int] = {name: 0 for name in patterns}

    # 2) For each pattern, apply tier weight to overlapping tokens in assistant region only
    for name, (pat, flags, tier) in patterns.items():
        w_tier = tier_weights.get(tier, 1.0)
        if w_tier <= 1.0:
            # Still count hits so we have statistics, but skip the weight update
            pattern_hits[name] = sum(1 for _ in re.finditer(pat, text, flags=flags))
            continue
        n = 0
        for m in re.finditer(pat, text, flags=flags):
            n += 1
            c_lo, c_hi = m.start(), m.end()
            idx = char_span_to_token_indices_vectorized(starts, ends, c_lo, c_hi)
            if idx.size == 0:
                continue
            # Only tokens in assistant region get weighted
            idx = idx[idx >= asst_start_token]
            if idx.size == 0:
                continue
            # Use max(current, tier) instead of multiplication — prevents double
            # weighting from overlapping patterns (e.g. step + bold overlap).
            cur = weights[idx]
            weights[idx] = np.maximum(cur, w_tier)
        pattern_hits[name] = n

    # 3) Force reserved special-token positions to 1.0
    reserved = np.isin(input_ids, list(RESERVED_TOKEN_IDS))
    weights[reserved] = 1.0

    # 4) Clamp
    weights = np.clip(weights, w_min, w_max).astype(np.float32)

    return weights, pattern_hits, asst_start_token


# ============================================================================
# Driver
# ============================================================================
def build_all(
    dataset: Dataset,
    tok: PreTrainedTokenizerFast,
    sample_indices: Optional[List[int]],
    tier_weights: Dict[str, float],
    w_min: float,
    w_max: float,
    batch_size: int,
    text_field: str = "text",
    id_field: str = "id",
) -> Dict[str, dict]:
    """Process all samples. Returns dict[sample_id] -> result dict."""
    if sample_indices is None:
        sample_indices = list(range(len(dataset)))

    results: Dict[str, dict] = {}
    total = len(sample_indices)
    t0 = time.time()

    for i in range(0, total, batch_size):
        batch_idx = sample_indices[i:i + batch_size]
        rows = [dataset[int(j)] for j in batch_idx]
        texts = [r[text_field] for r in rows]
        sids = [r.get(id_field, f"sample_{j:06d}") for r, j in zip(rows, batch_idx)]

        # Batch tokenize with offsets
        enc = tok(
            texts,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            padding=False,
        )

        for sid, text, ids_list, offsets_list in zip(
            sids, texts, enc["input_ids"], enc["offset_mapping"]
        ):
            input_ids = np.asarray(ids_list, dtype=np.int64)
            offsets = np.asarray(offsets_list, dtype=np.int64)
            if offsets.ndim != 2 or offsets.shape[1] != 2:
                # Some tokenizers may return flat list; reshape
                offsets = offsets.reshape(-1, 2)
            weights, hits, asst_start = compute_one(
                text=text,
                input_ids=input_ids,
                offsets=offsets,
                patterns=RE_PATTERNS,
                tier_weights=tier_weights,
                w_min=w_min,
                w_max=w_max,
            )
            results[sid] = {
                "weight": weights,
                "pattern_hits": hits,
                "text_length": len(text),
                "token_length": int(input_ids.size),
                "assistant_start_token": int(asst_start),
            }

        if (i // batch_size) % 20 == 0:
            done = i + len(batch_idx)
            tps = done / max(1e-6, time.time() - t0)
            eta = (total - done) / max(1e-6, tps)
            print(f"  [{done:>6d}/{total}]  {tps:6.1f} samples/s  ETA {eta:6.1f}s",
                  flush=True)

    print(f"  done {total} samples in {time.time()-t0:.1f}s", flush=True)
    return results


def save_outputs(
    results: Dict[str, dict],
    out_dir: Path,
    save_npz_per_sample: bool,
    tier_weights: Dict[str, float],
) -> Dict:
    out_dir = Path(out_dir)
    weights_dir = out_dir / "structure_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 1) Per-sample npz (optional)
    if save_npz_per_sample:
        # Clean stale files first to avoid mixing tiered + uniform weight files
        for old in weights_dir.glob("*.npz"):
            try:
                old.unlink()
            except OSError:
                pass
        for sid, r in results.items():
            np.savez_compressed(
                weights_dir / f"{sid}.npz",
                weight=r["weight"].astype(np.float32),
            )

    # 2) Bundle for fast training-time load
    ids = list(results.keys())
    weights_list = [results[s]["weight"].astype(np.float32) for s in ids]
    asst_starts = [results[s]["assistant_start_token"] for s in ids]
    bundle = {
        "ids": ids,
        "weights": weights_list,
        "assistant_start_tokens": asst_starts,
        "config": {
            "tier_weights": dict(tier_weights),
            "reserved_token_ids": sorted(list(RESERVED_TOKEN_IDS)),
            "patterns": {k: {"pattern": v[0], "flags": v[1], "tier": v[2]}
                         for k, v in RE_PATTERNS.items()},
        },
    }
    bundle_path = out_dir / "structure_weights_bundle.pt"
    torch.save(bundle, bundle_path)

    # 3) Aggregated metadata
    n = len(results)
    weight_max = [float(w.max()) for w in weights_list]
    weight_mean = [float(w.mean()) for w in weights_list]
    weight_above_one = [float((w > 1.0).sum()) for w in weights_list]
    token_len = [int(r["token_length"]) for r in results.values()]
    asst_len = [int(r["token_length"] - r["assistant_start_token"]) for r in results.values()]

    pattern_distribution: Dict[str, Dict] = {}
    for name in RE_PATTERNS:
        counts = [r["pattern_hits"].get(name, 0) for r in results.values()]
        pattern_distribution[name] = {
            "total_hits": int(sum(counts)),
            "mean_per_sample": float(np.mean(counts) if counts else 0.0),
            "samples_with_hits": int(sum(1 for c in counts if c > 0)),
        }

    meta = {
        "total_samples": n,
        "tier_weights": dict(tier_weights),
        "reserved_token_ids": sorted(list(RESERVED_TOKEN_IDS)),
        "token_length_stats": {
            "mean": float(np.mean(token_len)) if token_len else 0.0,
            "p50":  float(np.percentile(token_len, 50)) if token_len else 0.0,
            "p95":  float(np.percentile(token_len, 95)) if token_len else 0.0,
            "max":  int(np.max(token_len)) if token_len else 0,
        },
        "assistant_length_stats": {
            "mean": float(np.mean(asst_len)) if asst_len else 0.0,
            "p50":  float(np.percentile(asst_len, 50)) if asst_len else 0.0,
            "p95":  float(np.percentile(asst_len, 95)) if asst_len else 0.0,
            "max":  int(np.max(asst_len)) if asst_len else 0,
        },
        "weight_stats": {
            "max_mean":  float(np.mean(weight_max)) if weight_max else 0.0,
            "mean_mean": float(np.mean(weight_mean)) if weight_mean else 0.0,
            "tokens_above_1_per_sample": {
                "mean": float(np.mean(weight_above_one)) if weight_above_one else 0.0,
                "p50": float(np.percentile(weight_above_one, 50)) if weight_above_one else 0.0,
            },
        },
        "pattern_distribution": pattern_distribution,
        "bundle_path": str(bundle_path.relative_to(out_dir.parent.parent)),
    }

    meta_path = out_dir / "structure_weights_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


# ============================================================================
# Verification helpers
# ============================================================================
def quick_verify(results: Dict[str, dict]) -> None:
    """Light sanity checks."""
    n = len(results)
    if n == 0:
        raise RuntimeError("no results to verify")
    # 1) weights should all be in [1.0, w_max]
    bad = []
    for sid, r in results.items():
        w = r["weight"]
        if not np.all(np.isfinite(w)):
            bad.append((sid, "non-finite"))
        elif w.min() < 0.0:
            bad.append((sid, f"min<0 ({w.min()})"))
    if bad:
        print(f"⚠️  {len(bad)} samples failed sanity check (first 3): {bad[:3]}")
    else:
        print(f"✅ Sanity OK on {n} samples (all weights finite, ≥0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-path", type=Path, default=DEFAULT_HF_PATH)
    ap.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--sample-count", type=int, default=None,
                    help="Process first N samples (default: all).")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Tokenization batch size.")
    ap.add_argument("--w-min", type=float, default=1.0)
    ap.add_argument("--w-max", type=float, default=10.0)
    ap.add_argument("--w-tier1", type=float, default=DEFAULT_TIER_WEIGHTS["tier1"])
    ap.add_argument("--w-tier2", type=float, default=DEFAULT_TIER_WEIGHTS["tier2"])
    ap.add_argument("--w-tier3", type=float, default=DEFAULT_TIER_WEIGHTS["tier3"])
    ap.add_argument("--no-npz", action="store_true",
                    help="Skip per-sample .npz output; only bundle.pt + metadata.")
    args = ap.parse_args()

    tier_weights = {
        "tier1": float(args.w_tier1),
        "tier2": float(args.w_tier2),
        "tier3": float(args.w_tier3),
    }

    print(f"📂 HF dataset: {args.hf_path}")
    print(f"📂 Tokenizer:  {args.tokenizer_dir}")
    print(f"📂 Output:     {args.output_dir}")
    print(f"⚙️  Tier weights: {tier_weights}")
    print(f"⚙️  Clamp: [{args.w_min}, {args.w_max}]")
    print(f"⚙️  Reserved special token IDs (force w=1.0): {sorted(RESERVED_TOKEN_IDS)}")
    print()

    print("Loading dataset…")
    ds = load_hf_dataset(args.hf_path)
    print(f"  → {len(ds)} samples")

    print("Loading tokenizer…")
    tok = load_tokenizer(args.tokenizer_dir)
    print(f"  → len(tok)={len(tok)}  eos_id={tok.eos_token_id}")

    sample_indices = None
    if args.sample_count and args.sample_count > 0:
        sample_indices = list(range(min(args.sample_count, len(ds))))

    print("Building weights (batched)…")
    results = build_all(
        dataset=ds,
        tok=tok,
        sample_indices=sample_indices,
        tier_weights=tier_weights,
        w_min=args.w_min,
        w_max=args.w_max,
        batch_size=args.batch_size,
    )

    quick_verify(results)

    print("Saving outputs…")
    meta = save_outputs(
        results=results,
        out_dir=args.output_dir,
        save_npz_per_sample=not args.no_npz,
        tier_weights=tier_weights,
    )

    print("\n=== Summary ===")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n✅ Outputs written under: {args.output_dir}")


if __name__ == "__main__":
    main()
