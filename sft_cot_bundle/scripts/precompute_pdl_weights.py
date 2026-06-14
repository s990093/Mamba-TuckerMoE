#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute PDL (Probabilistic Down-weighting by token frequency) weights.

What this does
--------------
Reads the tokenised SFT-CoT training stream (`dataset/stf_cot_train.bin`,
uint16 mmap), counts per-token frequency with `np.bincount`, then maps the
frequency to a static per-token CE multiplier:

    pdl[v] = ((freq[v] + 1.0) / freq.max()) ** (-alpha)

Rare tokens get a multiplier > 1.0 (up-weighted), frequent tokens get < 1.0.

Special / structure tokens (32000..32006: <|im_start|>, <|im_end|>, <think>,
</think>, <final>, </final>, [PAD]) have their frequency forced to 0 so their
weight collapses to exactly 1.0 — they are already handled by SFT-GO / FCP /
structure_token_ce_weighting and must not be double-weighted here.

Output
------
*   `cot_task/reports/pdl_weights.pt`   — torch.float32 tensor [vocab_size].
*   `cot_task/reports/pdl_weights_meta.json` — summary statistics.

Usage
-----
    python precompute_pdl_weights.py                 # alpha=0.4, vocab=32007
    python precompute_pdl_weights.py --alpha 0.5
    python precompute_pdl_weights.py --bin dataset/stf_cot_train.bin
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Special / structure token ids that must never be PDL-weighted (force w=1.0).
SPECIAL_TOKEN_IDS = (32000, 32001, 32002, 32003, 32004, 32005, 32006)


def _resolve(p: str) -> Path:
    """Resolve a path relative to the bundle root (parent of scripts/)."""
    pp = Path(p)
    if pp.is_absolute() or pp.exists():
        return pp
    bundle_root = Path(__file__).resolve().parent.parent
    return bundle_root / p


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute PDL token-frequency weights.")
    ap.add_argument("--bin", default="dataset/stf_cot_train.bin",
                    help="uint16 mmap of training token ids")
    ap.add_argument("--vocab-size", type=int, default=32007)
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="down-weighting exponent; 0 disables (all weights=1.0)")
    ap.add_argument("--no-normalize", action="store_true",
                    help="skip occurrence-weighted normalisation (raw >=1 weights)")
    ap.add_argument("--out", default="cot_task/reports/pdl_weights.pt")
    args = ap.parse_args()

    bin_path = _resolve(args.bin)
    if not bin_path.exists():
        raise FileNotFoundError(f"training bin not found: {bin_path}")

    mm = np.memmap(bin_path, dtype=np.uint16, mode="r")
    freq = np.bincount(mm, minlength=args.vocab_size).astype(np.float64)
    if freq.shape[0] > args.vocab_size:
        # ids beyond declared vocab should not exist; fold defensively
        freq = freq[: args.vocab_size]

    # Force special / structure tokens to freq=0 → weight exactly 1.0.
    for tid in SPECIAL_TOKEN_IDS:
        if 0 <= tid < freq.shape[0]:
            freq[tid] = 0.0

    fmax = float(freq.max()) if freq.max() > 0 else 1.0
    alpha = float(args.alpha)
    # Base shape: ((freq+1)/fmax) ** (-alpha). This is monotonically *decreasing*
    # in freq (rare → larger), but is always >= 1.0, so on its own it only ever
    # up-weights and inflates the overall loss scale (mean can be tens of x).
    pdl = (((freq + 1.0) / fmax) ** (-alpha)).astype(np.float64)

    used = freq > 0.0
    # Occurrence-weighted normalisation: divide by E_token[pdl] so that the
    # *expected* multiplier over the real token stream is exactly 1.0. This keeps
    # the CE loss / gradient scale unchanged while still up-weighting rare tokens
    # relative to frequent ones (frequent → <1, rare → >1). Without this the mean
    # multiplier would be ~50x and training would diverge.
    if not args.no_normalize and used.any():
        occ_mean = float((freq[used] * pdl[used]).sum() / freq[used].sum())
        if occ_mean > 0:
            pdl = pdl / occ_mean

    pdl = pdl.astype(np.float32)
    # Special / structure tokens → exactly 1.0 (handled by SFT-GO / FCP).
    for tid in SPECIAL_TOKEN_IDS:
        if 0 <= tid < pdl.shape[0]:
            pdl[tid] = 1.0
    # Tokens never seen in training → 1.0 (don't blindly up-weight the unknown).
    pdl[~used] = 1.0

    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(pdl), out_path)

    # Occurrence-weighted mean of the final weights (should be ~1.0 when normalised).
    occ_mean_final = (
        float((freq[used] * pdl[used]).sum() / freq[used].sum()) if used.any() else 1.0
    )
    meta = {
        "bin": str(bin_path),
        "vocab_size": int(pdl.shape[0]),
        "alpha": alpha,
        "normalized": (not args.no_normalize),
        "total_tokens": int(mm.shape[0]),
        "max_freq": fmax,
        "seen_tokens": int(used.sum()),
        "pdl_min": float(pdl.min()),
        "pdl_max": float(pdl.max()),
        "pdl_mean": float(pdl.mean()),
        "pdl_mean_used": float(pdl[used].mean()) if used.any() else 1.0,
        "pdl_occurrence_weighted_mean": occ_mean_final,
        "special_token_ids_forced_to_1": list(SPECIAL_TOKEN_IDS),
    }
    meta_path = out_path.with_name("pdl_weights_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ PDL weights saved: {out_path}  (vocab={pdl.shape[0]}, alpha={alpha})")
    print(f"   pdl range [{pdl.min():.4f}, {pdl.max():.4f}]  "
          f"mean(used)={meta['pdl_mean_used']:.4f}  seen={meta['seen_tokens']}")
    print(f"   meta: {meta_path}")


if __name__ == "__main__":
    main()
