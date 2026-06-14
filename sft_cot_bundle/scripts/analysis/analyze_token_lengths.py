#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掃描 HF dataset 的 text 欄位，統計每筆 token 長度分佈，
輸出 SEQ_LEN 建議（讓 p95 不被截斷）。

用法：
    python3 sft_cot_bundle/scripts/analyze_token_lengths.py \
        --hf-dir cot_dataset/stf_cot_hf_final \
        --tokenizer-dir cot_dataset \
        --out output/stf_cot_token_len_report.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="Token 長度統計")
    ap.add_argument("--hf-dir", type=Path, required=True)
    ap.add_argument("--tokenizer-dir", type=Path, required=True)
    ap.add_argument("--text-column", type=str, default="text")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ds = load_from_disk(str(args.hf_dir))
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True)
    tok.model_max_length = 1_000_000

    lens: list[int] = []
    bucket_lens: dict[str, list[int]] = {}
    for i in range(len(ds)):
        row = ds[i]
        text = str(row[args.text_column])
        ids = tok.encode(text, add_special_tokens=False)
        n = len(ids)
        lens.append(n)
        bucket = row.get("sys_bucket", "unknown")
        bucket_lens.setdefault(bucket, []).append(n)

    arr = np.asarray(lens, dtype=np.int64)
    pct = lambda q: float(np.percentile(arr, q)) if len(arr) else 0.0

    candidates = [256, 512, 768, 1024, 1536, 2048]
    trunc: dict[str, dict] = {}
    for seq in candidates:
        over = int(np.sum(arr > seq + 1))
        trunc[str(seq)] = {
            "n_truncated": over,
            "truncation_rate": f"{100 * over / max(1, len(arr)):.1f}%",
        }

    p95 = pct(95)
    suggested = None
    for seq in candidates:
        if p95 <= seq + 1:
            suggested = seq
            break
    if suggested is None:
        suggested = max(candidates)

    per_bucket: dict[str, dict] = {}
    for bk, bl in sorted(bucket_lens.items()):
        ba = np.asarray(bl, dtype=np.int64)
        per_bucket[bk] = {
            "count": len(bl),
            "min": int(ba.min()),
            "mean": round(float(ba.mean()), 1),
            "max": int(ba.max()),
            "p50": round(float(np.percentile(ba, 50)), 0),
            "p95": round(float(np.percentile(ba, 95)), 0),
            "p99": round(float(np.percentile(ba, 99)), 0),
        }

    report = {
        "total_examples": len(arr),
        "token_len_min": int(arr.min()) if len(arr) else 0,
        "token_len_mean": round(float(arr.mean()), 1) if len(arr) else 0,
        "token_len_max": int(arr.max()) if len(arr) else 0,
        "p50": round(pct(50), 0),
        "p90": round(pct(90), 0),
        "p95": round(pct(95), 0),
        "p99": round(pct(99), 0),
        "suggested_seq_len": suggested,
        "truncation_by_seq_len": trunc,
        "per_bucket_stats": per_bucket,
    }

    print("\n=== Token Length Report ===")
    print(f"  Total examples: {report['total_examples']:,}")
    print(f"  Min / Mean / Max: {report['token_len_min']} / {report['token_len_mean']} / {report['token_len_max']}")
    print(f"  p50={report['p50']:.0f}  p90={report['p90']:.0f}  p95={report['p95']:.0f}  p99={report['p99']:.0f}")
    print(f"\n  ✨ 建議 SEQ_LEN = {suggested} (p95 = {p95:.0f} tokens)")
    print(f"\n  Truncation rates:")
    for seq, info in trunc.items():
        print(f"    SEQ_LEN={seq:>5s}: {info['n_truncated']:>5d} truncated ({info['truncation_rate']})")
    print(f"\n  Per-bucket stats:")
    for bk, info in per_bucket.items():
        print(f"    {bk:>22s}: n={info['count']:>5d}  mean={info['mean']:>6.1f}  p95={info['p95']:>5.0f}  max={info['max']:>5d}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n  Report saved → {args.out}")


if __name__ == "__main__":
    main()
