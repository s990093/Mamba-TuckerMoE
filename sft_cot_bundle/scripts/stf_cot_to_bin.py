#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 dataset/stf.json（欄位 input / cot / output / category）轉為 ChatML + CoT 結構，並輸出：

  - HF 快照（預設 dataset/stf_cot_hf；欄位 `id` / `category` / `sys_bucket` / `system` / `text`）
  - uint16 串流 .bin（預設 dataset/stf_cot_train.bin）
  - meta JSON（長度統計、`train_sft.py` 參數提示）

Assistant 區塊（與 lima_to_bin.SPECIAL_TOKENS[2:6] 對齊）::

    <think>
    {cot}
    </think>
    <final>
    {output}
    </final>

再接 <|im_end|>（見 _format_conversation）。

**System（demo）**：依 `category` 對照三種人设——對話（daily、meta）、任務（task）、
總結（knowledge），在 user 區塊前插入 `<|im_start|>system\\n…<|im_end|>\\n`。
可用 `--no-system` 關閉。

監督 mask 與 mask.md / train_sft._build_xy_masked 一致：自
`<|im_start|>assistant\\n` 後直至（含）`<|im_end|>` 序列皆計 CE
（含 thinking / final / 結尾符）。

Tokenizer：預設載入 `--tokenizer-dir`（須為已含 32007 之目錄）；若省略則自
`--tokenizer-base`（例如 third_party/Tiny-LLM）自動 add_tokens(SPECIAL_TOKENS)。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"


def _add_scripts_to_path() -> None:
    s = str(ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)


def load_tokenizer(tokenizer_dir: Path | None, tokenizer_base: Path) -> object:
    _add_scripts_to_path()
    from lima_to_bin import SPECIAL_TOKENS

    base = Path(tokenizer_base)
    if tokenizer_dir is not None:
        td = Path(tokenizer_dir)
        tok = AutoTokenizer.from_pretrained(str(td), local_files_only=True)
        if len(tok) not in (32000, 32007):
            raise SystemExit(f"預期 tokenizer len 32000 或 32007，目前 {len(tok)}（{td}）")
        if len(tok) == 32000:
            added = tok.add_tokens(SPECIAL_TOKENS)
            if added != 7:
                raise SystemExit(f"需在 {td} 上成功 add 7 個 special；added={added}")
    else:
        if not base.is_dir():
            raise SystemExit(f"找不到 tokenizer 基底目錄: {base}")
        tok = AutoTokenizer.from_pretrained(str(base), local_files_only=False)
        if len(tok) != 32000:
            raise SystemExit(
                f"--tokenizer-base 預期詞表長 32000（Tiny-LLM），目前 {len(tok)}；"
                "或改傳已合併的 --tokenizer-dir"
            )
        added = tok.add_tokens(SPECIAL_TOKENS)
        if added != 7 or len(tok) != 32007:
            raise SystemExit(f"add_tokens 失敗：added={added} len={len(tok)}")
    if tok.pad_token is None:
        tok.pad_token = SPECIAL_TOKENS[6]
    tok.model_max_length = 1_000_000
    return tok


@dataclass
class StfCotFormatted:
    text: str
    sample_id: str
    category: str
    sys_bucket: str
    system: str


def format_stf_cot_record(
    row: dict,
    *,
    special_tokens: list[str],
    use_system: bool,
    prompts: dict[str, str] | None,
    category_to_bucket: dict[str, str] | None,
) -> StfCotFormatted | None:
    from lima_to_bin import _format_conversation
    from stf_cot_sysprompt import prepend_system_chatml, system_for_row

    t_open, t_close = special_tokens[2], special_tokens[3]
    f_open, f_close = special_tokens[4], special_tokens[5]
    round_end = special_tokens[1]
    u = str(row.get("input", "")).strip()
    cot = str(row.get("cot", "")).strip()
    out = str(row.get("output", "")).strip()
    if not u or not out:
        return None
    if not cot:
        cot = ""
    assistant = f"{t_open}\n{cot}\n{t_close}\n{f_open}\n{out}\n{f_close}"
    body = _format_conversation([u, assistant])
    cat = str(row.get("category", "")).strip()
    bucket, sys_txt = system_for_row(
        row, prompts=prompts, category_to_bucket=category_to_bucket
    )
    if use_system and sys_txt:
        full = prepend_system_chatml(sys_txt, body, round_end_token=round_end)
    else:
        full = body
    sid = str(row.get("id", "")).strip()
    return StfCotFormatted(
        text=full,
        sample_id=sid,
        category=cat,
        sys_bucket=bucket,
        system=sys_txt if use_system else "",
    )


def format_stf_row_as_chatml(
    row: dict,
    *,
    special_tokens: list[str],
    use_system: bool = True,
    prompts: dict[str, str] | None = None,
    category_to_bucket: dict[str, str] | None = None,
) -> str:
    """相容舊呼叫：只回傳 `text`；預設帶 system。"""
    r = format_stf_cot_record(
        row,
        special_tokens=special_tokens,
        use_system=use_system,
        prompts=prompts,
        category_to_bucket=category_to_bucket,
    )
    return r.text if r else ""


def _token_len_analysis(
    lens: list[int],
    *,
    candidate_seq_lens: list[int],
) -> dict:
    if not lens:
        return {"n_examples": 0, "error": "no examples"}
    arr = np.asarray(lens, dtype=np.int64)
    n = len(arr)
    pct = lambda q: float(np.percentile(arr, q)) if n else 0.0

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
    suggested = None
    for seq in sorted(candidate_seq_lens):
        if p95 <= seq + 1:
            suggested = seq
            break
    if suggested is None:
        suggested = max(candidate_seq_lens)

    rationale = (
        f"依 p95={p95:.0f} tokens：若目標是讓約 95% 樣本**不被**長度截斷，"
        f"應設 SEQ_LEN>={int(np.ceil(p95 - 1))}。"
    )
    if p99 > suggested + 1:
        rationale += (
            f" 但 p99={p99:.0f}，若仍用 SEQ_LEN={suggested}，約 "
            f"{(100 * float(np.mean(arr > suggested + 1))):.1f}% 樣本仍可能截斷。"
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
        "recommendation": rationale,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="stf.json → ChatML-CoT HF + .bin")
    ap.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_DATASET_DIR / "stf.json",
        help="來源 JSON 陣列（input / cot / output）",
    )
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--hf-out", type=str, default="stf_cot_hf")
    ap.add_argument("--bin-name", type=str, default="stf_cot_train.bin")
    ap.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "tokenizer",
        help="已含 32007 或 32000+可 add 的 tokenizer 目錄（預設 dataset/tokenizer）；",
    )
    ap.add_argument(
        "--tokenizer-base",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "Tiny-LLM",
        help="基底 32000（預設 Tiny-LLM），會自動加 7 個 special",
    )
    ap.add_argument(
        "--save-tokenizer",
        type=Path,
        default=None,
        help="若設定，將本次使用的 tokenizer（含 32007）存到此目錄",
    )
    ap.add_argument("--seq-candidates", type=str, default="256,512,768,1024,1536,2048")
    ap.add_argument(
        "--no-system",
        action="store_true",
        help="不加 <|im_start|>system 區塊（舊行為相容）",
    )
    ap.add_argument(
        "--category-map-json",
        type=Path,
        default=None,
        help=r'選填 JSON：{"daily":"dialogue","task":"task",...} 覆寫內建 category→bucket',
    )
    ap.add_argument(
        "--sys-prompts-json",
        type=Path,
        default=None,
        help=r'選填 JSON：{"dialogue":"...","task":"...","summary":"..."} 覆寫三種 system',
    )
    args = ap.parse_args()

    src = Path(args.src).resolve()
    if not src.is_file():
        raise SystemExit(f"找不到 {src}")

    raw = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("stf.json 頂層須為陣列")

    _add_scripts_to_path()
    from lima_to_bin import SPECIAL_TOKENS
    from stf_cot_sysprompt import (
        DEFAULT_STF_CATEGORY_TO_BUCKET,
        DEFAULT_SYS_PROMPTS_BY_BUCKET,
    )

    category_map: dict[str, str] = dict(DEFAULT_STF_CATEGORY_TO_BUCKET)
    if args.category_map_json is not None:
        p = Path(args.category_map_json)
        extra = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(extra, dict):
            raise SystemExit("--category-map-json 須為物件")
        for k, v in extra.items():
            category_map[str(k).strip().lower()] = str(v).strip()

    prompts: dict[str, str] = dict(DEFAULT_SYS_PROMPTS_BY_BUCKET)
    if args.sys_prompts_json is not None:
        p = Path(args.sys_prompts_json)
        extra = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(extra, dict):
            raise SystemExit("--sys-prompts-json 須為物件")
        for k, v in extra.items():
            prompts[str(k).strip()] = str(v).strip()

    use_system = not args.no_system

    tok = load_tokenizer(
        Path(args.tokenizer_dir) if args.tokenizer_dir is not None else None,
        Path(args.tokenizer_base),
    )

    ds_dir: Path = args.dataset_dir
    ds_dir.mkdir(parents=True, exist_ok=True)
    hf_path = ds_dir / args.hf_out
    bin_path = ds_dir / args.bin_name
    meta_path = ds_dir / "stf_cot_meta.json"

    if args.save_tokenizer is not None:
        out_tok = Path(args.save_tokenizer)
        out_tok.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(out_tok))
        print(f"Tokenizer saved -> {out_tok}")

    texts: list[str] = []
    row_ids: list[str] = []
    row_categories: list[str] = []
    row_buckets: list[str] = []
    row_systems: list[str] = []
    n_skip = 0
    for row in raw:
        if not isinstance(row, dict):
            n_skip += 1
            continue
        rec = format_stf_cot_record(
            row,
            special_tokens=SPECIAL_TOKENS,
            use_system=use_system,
            prompts=prompts,
            category_to_bucket=category_map,
        )
        if rec is None or not rec.text.strip():
            n_skip += 1
            continue
        texts.append(rec.text)
        row_ids.append(rec.sample_id)
        row_categories.append(rec.category)
        row_buckets.append(rec.sys_bucket)
        row_systems.append(rec.system)

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
    stats_path = ds_dir / "stf_cot_token_len_stats.json"
    stats_path.write_text(
        json.dumps(len_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== stf CoT → ChatML token 長度 ===")
    if len_stats.get("n_examples"):
        print(
            f"  筆數 {len_stats['n_examples']:_} | "
            f"min/mean/max = {len_stats['per_example_token_len_min']}/"
            f"{len_stats['per_example_token_len_mean']:.1f}/"
            f"{len_stats['per_example_token_len_max']}"
        )
        print(
            f"  p50={len_stats['p50']:.0f} p95={len_stats['p95']:.0f} p99={len_stats['p99']:.0f}"
        )
        print(f"  建議 SEQ_LEN（p95）≈ {len_stats.get('suggested_seq_len_for_p95')}")
        print(f"  詳細 -> {stats_path}\n")

    arr = np.asarray(all_ids, dtype=np.uint16)
    print(f"Writing {len(all_ids)} tokens -> {bin_path}")
    arr.tofile(str(bin_path))

    ds_hf = Dataset.from_dict(
        {
            "id": row_ids,
            "category": row_categories,
            "sys_bucket": row_buckets,
            "system": row_systems,
            "text": texts,
        }
    )
    print(f"Saving HF ({len(texts)} rows, columns id/category/sys_bucket/system/text) -> {hf_path}")
    ds_hf.save_to_disk(str(hf_path))

    tok_ref = str(Path(args.save_tokenizer).resolve()) if args.save_tokenizer else (
        str(Path(args.tokenizer_dir).resolve()) if args.tokenizer_dir else str(args.tokenizer_base)
    )

    meta = {
        "format": "stf_cot_chatml_v2_sysprompt" if use_system else "stf_cot_chatml_v1_no_system",
        "use_system_prompt": use_system,
        "category_to_bucket": category_map,
        "sys_prompts": prompts,
        "source_json": str(src),
        "hf_snapshot": str(hf_path.resolve()),
        "train_bin": str(bin_path.resolve()),
        "tokenizer_ref": tok_ref,
        "vocab_size": len(tok),
        "special_tokens": list(SPECIAL_TOKENS),
        "num_examples": len(texts),
        "num_skipped": n_skip,
        "num_tokens": len(all_ids),
        "token_length_stats_file": str(stats_path.resolve()),
        "token_length_stats": len_stats,
        "train_sft_notes": (
            'SFT_DATA_SOURCE="hf_text", '
            f'LIMA_HF="{hf_path}", DATA_PATH="{bin_path}", '
            'SFT_TEXT_COLUMN="text", VOCAB_SIZE=32007；'
            "TOKENIZER_DIR 請指到已存檔的 32007 tokenizer（build_tiny_llm_tokenizer.py）。"
        ),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Meta -> {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
