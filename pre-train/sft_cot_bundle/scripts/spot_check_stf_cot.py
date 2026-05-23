#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽查 dataset/stf.json 與 stf→ChatML（含三路徑 system）是否一致：

  - 列出 category／sys_bucket 筆數
  - 每個 bucket（dialogue／task／summary）各印幾則縮排預覽（system / user 首段 / 有無 think+final）

用法::

    python scripts/spot_check_stf_cot.py
    python scripts/spot_check_stf_cot.py --per-bucket 3 --seed 42
    python scripts/spot_check_stf_cot.py --no-system --per-bucket 2
    python scripts/spot_check_stf_cot.py --full-three-types --out output/stf_spotcheck_full.txt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def _add_scripts_to_path() -> None:
    s = str(ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)


def _preview(s: str, max_chars: int) -> str:
    s = s.replace("\r\n", "\n")
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n…(truncated)…"


def main() -> None:
    _add_scripts_to_path()
    from lima_to_bin import SPECIAL_TOKENS
    from stf_cot_sysprompt import DEFAULT_STF_CATEGORY_TO_BUCKET
    from stf_cot_to_bin import format_stf_cot_record

    ap = argparse.ArgumentParser(description="抽查 stf.json → ChatML+system")
    ap.add_argument(
        "--src",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "stf.json",
    )
    ap.add_argument("--per-bucket", type=int, default=2, help="每個 sys_bucket 抽幾則縮印")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=900)
    ap.add_argument("--no-system", action="store_true")
    ap.add_argument(
        "--full-three-types",
        action="store_true",
        help="完整輸出三個 bucket（dialogue/task/summary）所有樣本到 --out",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="若設定，將輸出同時寫入 txt 檔",
    )
    args = ap.parse_args()

    src = Path(args.src)
    rows = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit("頂層須為陣列")

    use_system = not args.no_system
    cat_ctr: Counter[str] = Counter()
    bucket_ctr: Counter[str] = Counter()
    by_bucket: dict[str, list[int]] = {"dialogue": [], "task": [], "summary": []}

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        c = str(row.get("category", "")).strip() or "(empty)"
        cat_ctr[c] += 1
        rec = format_stf_cot_record(
            row,
            special_tokens=SPECIAL_TOKENS,
            use_system=use_system,
            prompts=None,
            category_to_bucket=dict(DEFAULT_STF_CATEGORY_TO_BUCKET),
        )
        if rec is None:
            continue
        bucket_ctr[rec.sys_bucket] += 1
        if rec.sys_bucket in by_bucket:
            by_bucket[rec.sys_bucket].append(i)

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s + "\n")

    emit(f"來源: {src}\n")
    emit("=== category（原始標籤）筆數 ===")
    for k, v in sorted(cat_ctr.items(), key=lambda x: (-x[1], x[0])):
        b = DEFAULT_STF_CATEGORY_TO_BUCKET.get(k.lower(), "dialogue (fallback)")
        emit(f"  {k:12} {v:5}  → bucket: {b}")
    emit("\n=== sys_bucket（對話／任務／總結）筆數 ===")
    for k in ("dialogue", "task", "summary"):
        emit(f"  {k:10} {bucket_ctr.get(k, 0)}")
    for k in sorted(bucket_ctr.keys()):
        if k not in ("dialogue", "task", "summary"):
            emit(f"  {k:10} {bucket_ctr[k]} (非預設三類，請檢查對照表)")

    rng = random.Random(args.seed)
    emit("\n=== 分层抽查（只看 text 前段）===\n")
    emit("內建對照：" + repr(DEFAULT_STF_CATEGORY_TO_BUCKET))

    per = max(0, args.per_bucket)
    for bucket in ("dialogue", "task", "summary"):
        idxs = list(by_bucket[bucket])
        if args.full_three_types:
            pick = idxs
        else:
            rng.shuffle(idxs)
            pick = idxs[:per]
        mode = "全量" if args.full_three_types else f"抽 {len(pick)} 則"
        emit(f"--- bucket={bucket!r} {mode} ---")
        for ii in pick:
            row = rows[ii]
            rec = format_stf_cot_record(
                row,
                special_tokens=SPECIAL_TOKENS,
                use_system=use_system,
                prompts=None,
                category_to_bucket=dict(DEFAULT_STF_CATEGORY_TO_BUCKET),
            )
            assert rec is not None
            cid = row.get("id", ii)
            emit(f"  [id={cid}] category={row.get('category')!r}")
            if use_system:
                emit(f"    system={_preview(rec.system, 240)!r}")
            chk = []
            chk.append("<|im_start|>system" in rec.text if use_system else True)
            chk.append("<|im_start|>user" in rec.text)
            chk.append("<think>" in rec.text)
            chk.append("<final>" in rec.text)
            chk.append("<|im_end|>" in rec.text)
            emit(f"    structure_ok: system?={chk[0]} user={chk[1]} think={chk[2]} final={chk[3]} im_end={chk[4]}")
            emit(_preview(rec.text, args.max_chars))
            emit()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(lines), encoding="utf-8")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
