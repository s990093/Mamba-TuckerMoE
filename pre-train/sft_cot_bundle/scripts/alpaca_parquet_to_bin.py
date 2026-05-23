# -*- coding: utf-8 -*-
"""
將 tatsu-lab/alpaca 結構的 Parquet（instruction / input / output，可選 text）轉成：

1) HuggingFace 資料集目錄（每列一則 `text`，內容為與 lima_to_bin 相同的 ChatML 串接），供 train_sft.py
   `SFT_DATA_SOURCE="hf_text"`、`SFT_TEXT_COLUMN="text"` 使用。
2) uint16 串流 .bin（與 train.py / lima_to_bin 相同），供 validate_config 與串流長度統計。

Alpaca prompt 轉成單輪 user，再包成：

  <s> <|im_start|>user\\n{alpaca_prefix}<|im_end|>
      <|im_start|>assistant\\n{output}<|im_end|> </s>

特殊 token 與詞表與 lima_to_bin.SPECIAL_TOKENS / build_tokenizer() 一致。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT.parent / "dataset"
DEFAULT_PARQUET = ROOT.parent / "train-00000-of-00001-a09b74b3ef9c3b56.parquet"

# 與官方 Alpaca 常見模板一致（與你提供的樣本相同）
ALPACA_PROMPT_PREFIX = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
)


def _add_scripts_to_path() -> None:
    s = str(ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)


def _format_alpaca_user_turn(instruction: str, input_text: str) -> str:
    inst = (instruction or "").strip()
    inp = (input_text or "").strip()
    if inp:
        return (
            f"{ALPACA_PROMPT_PREFIX}### Instruction:\n{inst}\n\n"
            f"### Input:\n{inp}\n\n### Response:\n"
        )
    return f"{ALPACA_PROMPT_PREFIX}### Instruction:\n{inst}\n\n### Response:\n"


def build_tokenizer_custom(tokenizer_dir: Path | None) -> PreTrainedTokenizerFast:
    """
    優先使用 --tokenizer-dir；否則沿用 lima_to_bin.build_tokenizer()（scripts/llama2_tokenizer）。
    """
    _add_scripts_to_path()
    if tokenizer_dir is not None:
        from lima_to_bin import SPECIAL_TOKENS

        tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        n0 = len(tok)
        if n0 not in (32000, 32007):
            raise SystemExit(
                f"預期 tokenizer len 為 32000 或 32007，目前 {n0}（{tokenizer_dir}）"
            )
        if n0 == 32000:
            added = tok.add_tokens(SPECIAL_TOKENS)
            if added != 7 or len(tok) != 32007:
                raise SystemExit(
                    f"需成功新增 7 個 special tokens；added={added} len={len(tok)}"
                )
        if tok.pad_token is None:
            tok.pad_token = SPECIAL_TOKENS[6]
        tok.model_max_length = 1_000_000
        return tok  # type: ignore[return-value]

    from lima_to_bin import build_tokenizer

    return build_tokenizer()


def _token_len_analysis(
    lens: list[int],
    *,
    candidate_seq_lens: list[int],
) -> dict:
    """
    與 train_sft.py 一致：單筆長度 > SEQ_LEN+1 時會前向截斷，尾部 assistant/</s> 可能丟失。
    candidate_seq_lens: 建議比對的 SEQ_LEN 候選（如 256, 512, 1024）。
    """
    if not lens:
        return {
            "n_examples": 0,
            "error": "no examples",
        }
    arr = np.asarray(lens, dtype=np.int64)
    n = len(arr)
    pct = lambda q: float(np.percentile(arr, q)) if n else 0.0

    # 每個 SEQ_LEN：會被截斷的樣本比例（長度 > SEQ_LEN+1）
    trunc_by_seq: dict[str, dict] = {}
    for seq in sorted(set(candidate_seq_lens)):
        thr = int(seq) + 1
        over = int(np.sum(arr > thr))
        trunc_by_seq[str(seq)] = {
            "max_len_no_truncation_token_count": thr,
            "n_truncated": over,
            "truncation_rate": float(over / max(1, n)),
        }

    p95 = pct(95)
    p99 = pct(99)
    # 粗建議：p95 納入視窗的「最小」SEQ_LEN（满足 p95 <= seq+1）
    suggested = None
    for seq in sorted(candidate_seq_lens):
        if p95 <= seq + 1:
            suggested = seq
            break
    if suggested is None:
        suggested = max(candidate_seq_lens)

    rationale = (
        f"依 p95={p95:.0f} tokens：若目標是讓約 95% 樣本**不被**長度截斷，"
        f"應設 SEQ_LEN>={int(np.ceil(p95 - 1))}（即 seq_len+1>={int(np.ceil(p95))}）。"
    )
    if p99 > suggested + 1:
        rationale += (
            f" 但 p99={p99:.0f}，若仍用 SEQ_LEN={suggested}，約 {(100 * np.mean(arr > suggested + 1)):.1f}% "
            "樣本仍會截斷；要覆蓋更多長樣本可升 SEQ_LEN 或接受截斷。"
        )

    return {
        "n_examples": n,
        "per_example_token_len_min": int(arr.min()),
        "per_example_token_len_max": int(arr.max()),
        "per_example_token_len_mean": float(arr.mean()),
        "per_example_token_len_std": float(arr.std()),
        "p50": pct(50),
        "p90": pct(90),
        "p95": p95,
        "p99": p99,
        "truncation_if_seq_len": trunc_by_seq,
        "suggested_seq_len_for_p95": suggested,
        "note_sft": (
            "train_sft 單筆需 seq_len+1 個 token 才不需截斷；表格中 truncation_rate = P(len > SEQ_LEN+1)。"
        ),
        "recommendation": rationale,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Alpaca Parquet → HF snapshot + train.bin")
    ap.add_argument(
        "parquet",
        nargs="?",
        default=str(DEFAULT_PARQUET),
        help="Parquet 路徑（預設專案根目錄下 train-00000-of-00001-*.parquet）",
    )
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="輸出目錄（預設 ../dataset）",
    )
    ap.add_argument(
        "--hf-out",
        type=str,
        default="alpaca_hf",
        help="HF 快照子目錄名（在 dataset-dir 下）",
    )
    ap.add_argument(
        "--bin-name",
        type=str,
        default="train_alpaca.bin",
        help="輸出 .bin 檔名（勿覆蓋既有 lima train.bin 時可改檔名）",
    )
    ap.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="已含 32007 詞表或需 +7 special 的目錄；省略則用 lima_to_bin 的 llama2_tokenizer",
    )
    ap.add_argument(
        "--from-text-column",
        action="store_true",
        help="直接使用 Parquet 的「text」欄位（須已是 ChatML 結構，否則 SFT label mask 不會對）",
    )
    ap.add_argument(
        "--seq-candidates",
        type=str,
        default="256,512,768,1024,2048",
        help="逗號分隔，用於截斷率與 SEQ_LEN 建議（預設 256,512,768,1024,2048）",
    )
    args = ap.parse_args()

    pq_path = Path(args.parquet).resolve()
    if not pq_path.is_file():
        raise SystemExit(f"找不到 Parquet: {pq_path}")

    ds_dir: Path = args.dataset_dir
    ds_dir.mkdir(parents=True, exist_ok=True)
    hf_path = ds_dir / args.hf_out
    bin_path = ds_dir / args.bin_name
    tok_out = ds_dir / "tokenizer"
    meta_path = ds_dir / "alpaca_bin_meta.json"

    print(f"Reading {pq_path} …")
    df = pd.read_parquet(pq_path)

    _add_scripts_to_path()
    from lima_to_bin import SPECIAL_TOKENS, _format_conversation

    texts: list[str] = []
    n_skip = 0

    if args.from_text_column:
        if "text" not in df.columns:
            raise SystemExit("使用 --from-text-column 需要欄位 'text'")
        for _, row in df.iterrows():
            t = str(row["text"]).strip()
            if not t:
                n_skip += 1
                continue
            texts.append(t)
    else:
        need = {"instruction", "output"}
        miss = need - set(df.columns)
        if miss:
            raise SystemExit(f"Parquet 缺少欄位 {miss}；可用 --from-text-column 改讀 text 欄")
        has_input = "input" in df.columns
        for _, row in df.iterrows():
            ins = row.get("instruction", "")
            outp = row.get("output", "")
            inp = row.get("input", "") if has_input else ""
            if not str(outp).strip():
                n_skip += 1
                continue
            u = _format_alpaca_user_turn(str(ins), str(inp))
            full = _format_conversation([u, str(outp).strip()])
            texts.append(full)

    print("Building tokenizer …")
    tok = build_tokenizer_custom(args.tokenizer_dir)
    tok_out.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(tok_out))

    seq_candidates = [int(x.strip()) for x in args.seq_candidates.split(",") if x.strip()]

    all_ids: list[int] = []
    per_len: list[int] = []
    for text in texts:
        ids = tok.encode(text, add_special_tokens=False)
        per_len.append(len(ids))
        for wid in ids:
            if not (0 <= wid < 65_535):
                raise SystemExit(f"token id 超出 uint16: {wid}")
        all_ids.extend(ids)

    len_stats = _token_len_analysis(per_len, candidate_seq_lens=seq_candidates)
    stats_path = ds_dir / "alpaca_token_len_stats.json"
    stats_path.write_text(
        json.dumps(len_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== Token 長度分析（單筆樣本 = 一則 ChatML 全文）===")
    if len_stats.get("n_examples"):
        print(
            f"  筆數 {len_stats['n_examples']:_} | "
            f"每筆 token 數 min/mean/max = "
            f"{len_stats['per_example_token_len_min']}/{len_stats['per_example_token_len_mean']:.1f}/"
            f"{len_stats['per_example_token_len_max']}"
        )
        print(
            f"  p50={len_stats['p50']:.0f}  p90={len_stats['p90']:.0f}  "
            f"p95={len_stats['p95']:.0f}  p99={len_stats['p99']:.0f}"
        )
        print("  各 SEQ_LEN 下「長度 > SEQ_LEN+1」比例（會被前向截斷）：")
        for sk, row in sorted(
            len_stats.get("truncation_if_seq_len", {}).items(), key=lambda x: int(x[0])
        ):
            tr = 100.0 * float(row["truncation_rate"])
            print(f"    SEQ_LEN={sk:>4}  →  截斷率 {tr:5.2f}%  ({row['n_truncated']:_} / {len_stats['n_examples']:_})")
        print(
            f"\n  粗建議：若優先覆蓋 p95，可先試 SEQ_LEN = {len_stats.get('suggested_seq_len_for_p95')}\n"
            f"  {len_stats.get('recommendation', '')}"
        )
        print(f"  詳細 JSON -> {stats_path}\n")

    n_tokens = len(all_ids)
    arr = np.asarray(all_ids, dtype=np.uint16)
    print(f"Writing {n_tokens} tokens -> {bin_path}")
    arr.tofile(str(bin_path))

    ds_hf = Dataset.from_dict({"text": texts})
    print(f"Saving HF dataset ({len(texts)} rows) -> {hf_path}")
    ds_hf.save_to_disk(str(hf_path))

    meta = {
        "format": "alpaca_to_chatml_im_v1",
        "source_parquet": str(pq_path),
        "hf_snapshot": str(hf_path.resolve()),
        "train_bin": str(bin_path.resolve()),
        "tokenizer_dir": str(tok_out.resolve()),
        "num_tokens": n_tokens,
        "num_examples": len(texts),
        "num_skipped": n_skip,
        "vocab_size": len(tok),
        "special_tokens": list(SPECIAL_TOKENS),
        "token_length_stats_file": str(stats_path.resolve()),
        "token_length_stats": len_stats,
        "train_sft_notes": (
            'train_sft.py 建議：SFT_DATA_SOURCE="hf_text", '
            f'LIMA_HF="{hf_path}", DATA_PATH="{bin_path}", '
            'TOKENIZER_DIR="dataset/tokenizer", SFT_TEXT_COLUMN="text"；'
            "SEQ_LEN 請參考 alpaca_token_len_stats.json"
        ),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Meta -> {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
