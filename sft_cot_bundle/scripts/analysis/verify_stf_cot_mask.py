#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
不依賴 PyTorch：複製 train_sft._build_xy_masked 的 **id 層級**邏輯（見 mask.md），
對 stf CoT ChatML 樣本檢查：

  - 能匹配 `<|im_start|>assistant\\n` 後之監督區間
  - 結尾對齊 `<|im_end|>`（整段子序列）
  - y[k] 對應「預測 ids[k+1]」：首個正文 token、結尾後皆為 -100

若已安裝 torch，可再加 `--compare-torch` 與 train_sft 實作逐元素比對。

用法（專案根目錄）::

    python scripts/verify_stf_cot_mask.py
    python scripts/verify_stf_cot_mask.py --tokenizer-dir dataset/tokenizer_tiny_llm_cot
    python scripts/verify_stf_cot_mask.py --sample-index 0 --seq-len 256
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def _add_scripts_to_path() -> None:
    s = str(ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)


def build_labels_masked(
    ids: list[int],
    tok: object,
) -> list[int]:
    """與 train_sft._build_xy_masked 核心一致（labels 與 ids 等長）。"""
    t = len(ids)
    labels: list[int] = [-100] * t

    header_ids: list[int] = list(
        tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    )
    header_len = len(header_ids)
    stop_seqs: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    _stop_redacted = "<|" + "redacted" + "_" + "im" + "_" + "end" + "|" + ">"
    _stop_im_end = "<|" + "im" + "_" + "end" + "|" + ">"
    for end_s in (_stop_redacted, _stop_im_end):
        seq = list(tok.encode(end_s, add_special_tokens=False))
        if not seq:
            continue
        key = tuple(seq)
        if key in seen:
            continue
        seen.add(key)
        stop_seqs.append(seq)

    def _match_stop_len_at(j: int) -> int:
        best = 0
        for sq in stop_seqs:
            Lsq = len(sq)
            if Lsq and j + Lsq <= t and ids[j : j + Lsq] == sq:
                best = max(best, Lsq)
        return best

    i = 0
    while i < t and header_len > 0:
        if i + header_len <= t and ids[i : i + header_len] == header_ids:
            start_idx = i + header_len
            end_exclusive = start_idx
            while end_exclusive < t:
                mlen = _match_stop_len_at(end_exclusive)
                if mlen:
                    end_exclusive += mlen
                    break
                end_exclusive += 1
            for j in range(start_idx, end_exclusive):
                labels[j] = ids[j]
            i = end_exclusive
        else:
            i += 1
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stf-json", type=Path, default=PROJECT_ROOT / "dataset" / "stf.json")
    ap.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "tokenizer_tiny_llm_cot",
    )
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument(
        "--compare-torch",
        action="store_true",
        help="若可 import torch，與 train_sft._build_xy_masked 比對",
    )
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "output" / "stf_cot_mask_verify.txt")
    args = ap.parse_args()

    _add_scripts_to_path()
    from lima_to_bin import SPECIAL_TOKENS
    from stf_cot_to_bin import format_stf_row_as_chatml, load_tokenizer

    stf_path = Path(args.stf_json)
    raw = json.loads(stf_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise SystemExit("stf 需為非空陣列")
    idx = int(args.sample_index)
    if idx < 0 or idx >= len(raw):
        raise SystemExit(f"sample-index 越界：{idx}（共 {len(raw)} 筆）")
    row = raw[idx]
    if not isinstance(row, dict):
        raise SystemExit("該筆不是 object")

    td = Path(args.tokenizer_dir)
    if td.is_dir():
        tok = load_tokenizer(td, PROJECT_ROOT / "third_party" / "Tiny-LLM")
    else:
        print(f"警告：{td} 不存在，改用 Tiny-LLM 基底即時 add_tokens", file=sys.stderr)
        tok = load_tokenizer(None, PROJECT_ROOT / "third_party" / "Tiny-LLM")

    text = format_stf_row_as_chatml(row, special_tokens=SPECIAL_TOKENS)
    if not text:
        raise SystemExit("格式化後為空（缺 input/output？）")

    ids = list(tok.encode(text, add_special_tokens=False))
    labels = build_labels_masked(ids, tok)
    seq_len = int(args.seq_len)
    pad_id = tok.convert_tokens_to_ids(SPECIAL_TOKENS[6])

    t = len(ids)
    if t > seq_len + 1:
        ids = ids[: seq_len + 1]
        labels = labels[: seq_len + 1]
        t = len(ids)
    need = seq_len + 1
    pad = need - t
    if pad > 0:
        ids = ids + [pad_id] * pad
        labels = labels + [-100] * pad

    x_ids = ids[:seq_len]
    y = labels[1 : seq_len + 1]

    lines: list[str] = []
    lines.append(f"sample_index={idx} id={row.get('id', '')}\n")
    lines.append("--- chatml preview (first 1200 chars) ---\n")
    lines.append(text[:1200] + ("\n...\n" if len(text) > 1200 else "\n"))
    lines.append("\n--- first supervised spans in labels (indices where label==id, contiguous) ---\n")
    runs: list[tuple[int, int]] = []
    j = 0
    while j < len(labels):
        if labels[j] != -100:
            k = j
            while k < len(labels) and labels[k] != -100:
                k += 1
            runs.append((j, k))
            j = k
        else:
            j += 1
    for a, b in runs[:5]:
        lines.append(f"  [{a}, {b}): {b - a} tokens\n")
    lines.append("\n--- decode check: first 3 (x[k], y[k]) pairs (y targets next id) ---\n")
    for k in range(min(3, len(x_ids))):
        lines.append(
            f"  k={k} x={x_ids[k]} -> y={y[k]} | "
            f"x_dec={tok.decode([x_ids[k]], skip_special_tokens=False)!r} | "
            f"y_dec={('_100' if y[k] == -100 else repr(tok.decode([y[k]], skip_special_tokens=False)))}\n"
        )

    hdr = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    assert hdr, "header encode 為空"
    # 確認至少有一段 supervision
    sup_count = sum(1 for la in labels[:t] if la != -100)
    assert sup_count > 0, "沒有任何監督位置（header 對不齊？）"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f" supervised_token_positions={sup_count} (within unpadded prefix up to truncation)")

    if args.compare_torch:
        try:
            import torch
            from train_sft import _build_xy_masked as torch_xy
        except Exception as e:
            print("compare_torch 略過:", e)
            return
        tx, ty = torch_xy(text, tok, seq_len, pad_id)
        tx_l = tx.tolist()
        ty_l = ty.tolist()
        assert tx_l == x_ids, "x 與 torch 版不一致"
        assert ty_l == y, "y 與 torch 版不一致"
        print("OK: pure-python mask matches train_sft._build_xy_masked")


if __name__ == "__main__":
    main()
