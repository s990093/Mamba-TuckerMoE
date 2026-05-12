#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版 mask 抽查：從 HF dataset 隨機抽樣，輸出每個位置的 x → y 對照，
特別標記監督邊界（-100 → supervised 與 supervised → -100 的轉換點）。

用法：
    python3 sft_cot_bundle/scripts/verify_mask_xy.py \
        --hf-dir dataset/stf_cot_hf \
        --tokenizer-dir dataset/tokenizer \
        --seq-len 1024 \
        --num-samples 3 \
        --out output/mask_xy_check.txt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer


def build_labels(ids: list[int], tok: object) -> list[int]:
    """與 train_sft._build_xy_masked 核心邏輯一致。"""
    t = len(ids)
    labels: list[int] = [-100] * t

    header_ids: list[int] = list(
        tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    )
    header_len = len(header_ids)

    stop_seqs: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for end_s in ("<|im_end|>",):
        seq = list(tok.encode(end_s, add_special_tokens=False))
        if seq:
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                stop_seqs.append(seq)

    def match_stop(j: int) -> int:
        for sq in stop_seqs:
            ls = len(sq)
            if ls and j + ls <= t and ids[j : j + ls] == sq:
                return ls
        return 0

    i = 0
    while i < t and header_len > 0:
        if i + header_len <= t and ids[i : i + header_len] == header_ids:
            start = i + header_len
            end = start
            while end < t:
                m = match_stop(end)
                if m:
                    end += m
                    break
                end += 1
            for j in range(start, end):
                labels[j] = ids[j]
            i = end
        else:
            i += 1
    return labels


def format_sample(
    sample_idx: int,
    row: dict,
    tok: object,
    seq_len: int,
    pad_id: int,
) -> str:
    text = str(row.get("text", ""))
    sid = row.get("id", "?")
    cat = row.get("category", "?")
    bucket = row.get("sys_bucket", "?")

    ids = list(tok.encode(text, add_special_tokens=False))
    labels = build_labels(ids, tok)
    orig_len = len(ids)

    if len(ids) > seq_len + 1:
        ids = ids[: seq_len + 1]
        labels = labels[: seq_len + 1]

    need = seq_len + 1
    pad = need - len(ids)
    if pad > 0:
        ids = ids + [pad_id] * pad
        labels = labels + [-100] * pad

    x_ids = ids[:seq_len]
    y = labels[1 : seq_len + 1]

    lines: list[str] = []
    lines.append(f"{'='*80}")
    lines.append(f"Sample #{sample_idx}  id={sid}  category={cat}  bucket={bucket}")
    lines.append(f"Original token length: {orig_len}  |  SEQ_LEN: {seq_len}")
    if orig_len > seq_len + 1:
        lines.append(f"⚠️  截斷: {orig_len} → {seq_len + 1}")
    lines.append(f"{'='*80}")

    # ChatML 預覽
    lines.append("")
    lines.append("--- ChatML preview (first 800 chars) ---")
    lines.append(text[:800] + ("..." if len(text) > 800 else ""))

    # 監督區間
    runs: list[tuple[int, int]] = []
    j = 0
    lbl_len = min(len(labels), seq_len + 1)
    while j < lbl_len:
        if labels[j] != -100:
            k = j
            while k < lbl_len and labels[k] != -100:
                k += 1
            runs.append((j, k))
            j = k
        else:
            j += 1

    sup_count = sum(b - a for a, b in runs)
    lines.append("")
    lines.append(f"--- Supervised spans: {len(runs)} segments, {sup_count} tokens ---")
    for a, b in runs:
        start_dec = tok.decode(ids[a : a + 3], skip_special_tokens=False)
        end_dec = tok.decode(ids[max(a, b - 3) : b], skip_special_tokens=False)
        lines.append(f"  [{a:>4d}, {b:>4d}): {b-a:>4d} tokens  start={start_dec!r}  end={end_dec!r}")

    # 完整 x → y 對照（邊界附近 + supervised 區間）
    lines.append("")
    lines.append("--- x → y 對照（邊界 🔑 + supervised 全列）---")
    lines.append(f"{'k':>5s}  {'x_tok':<25s}  →  {'y_tok':<25s}  {'status'}")
    lines.append("-" * 90)

    CONTEXT = 2
    prev_supervised = False
    show_positions: set[int] = set()

    for k in range(len(x_ids)):
        is_sup = y[k] != -100
        if is_sup != prev_supervised:
            for c in range(max(0, k - CONTEXT), min(len(x_ids), k + CONTEXT + 1)):
                show_positions.add(c)
        if is_sup:
            show_positions.add(k)
        prev_supervised = is_sup

    last_shown = -2
    prev_sup_for_display = False
    for k in range(len(x_ids)):
        is_sup = y[k] != -100
        if k not in show_positions:
            if last_shown == k - 1 and k > 0:
                lines.append("  ···  (masked region)")
            prev_sup_for_display = is_sup
            continue

        if k > last_shown + 1 and last_shown >= 0:
            lines.append("  ···")

        x_dec = tok.decode([x_ids[k]], skip_special_tokens=False)
        y_dec = "-100" if y[k] == -100 else tok.decode([y[k]], skip_special_tokens=False)

        boundary = ""
        if is_sup and not prev_sup_for_display:
            boundary = "  ← 🔑 監督開始"
        elif not is_sup and prev_sup_for_display:
            boundary = "  ← 🔑 監督結束"

        status = "[SUP]" if is_sup else "[MSK]"
        lines.append(f"{k:>5d}  {x_dec!r:<25s}  →  {y_dec!r:<25s}  {status}{boundary}")
        last_shown = k
        prev_sup_for_display = is_sup

    lines.append("")
    lines.append(f"Summary: {sup_count} supervised / {len(x_ids)} total tokens")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="增強版 mask x→y 抽查")
    ap.add_argument("--hf-dir", type=Path, required=True)
    ap.add_argument("--tokenizer-dir", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--indices", type=str, default=None,
                    help="Comma-separated sample indices (overrides --num-samples)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ds = load_from_disk(str(args.hf_dir))
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True)
    tok.model_max_length = 1_000_000
    pad_id = tok.convert_tokens_to_ids("[PAD]")
    if pad_id is None or pad_id < 0:
        pad_id = 0

    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",")]
    else:
        rng = random.Random(args.seed)
        buckets: dict[str, list[int]] = {}
        for i in range(len(ds)):
            bk = ds[i].get("sys_bucket", "unknown")
            buckets.setdefault(bk, []).append(i)

        indices = []
        bucket_names = sorted(buckets.keys())
        per = max(1, args.num_samples // max(1, len(bucket_names)))
        for bk in bucket_names:
            pool = buckets[bk]
            rng.shuffle(pool)
            indices.extend(pool[:per])
        rng.shuffle(indices)
        indices = indices[: args.num_samples]

    output_parts: list[str] = []
    output_parts.append(f"Mask x→y Verification Report")
    output_parts.append(f"HF: {args.hf_dir}  |  Tokenizer: {args.tokenizer_dir}  |  SEQ_LEN: {args.seq_len}")
    output_parts.append(f"Samples: {len(indices)}  indices={indices}")
    output_parts.append("")

    for rank, idx in enumerate(indices, 1):
        row = ds[idx]
        part = format_sample(rank, row, tok, args.seq_len, pad_id)
        output_parts.append(part)
        print(part[:500] + "...\n" if len(part) > 500 else part)

    full = "\n".join(output_parts)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(full, encoding="utf-8")
        print(f"\n✅ Full report → {args.out}")
    else:
        print(full)


if __name__ == "__main__":
    main()
