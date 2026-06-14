#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
準備 **完整 SFT-CoT .pt checkpoint** 給 Kaggle Dataset 上傳（與 npz 那包分開）。

- 自動選 ``output/checkpoint_sft_cot_s*.pt`` 中 **step 最大** 者（或 ``--checkpoint`` 指定）
- 在 ``output/kaggle_upload_sft_cot_checkpoint/`` 建立 **hard link**：
  ``latest_sft_cot_model.pt`` → 該檔（不複製、省空間；須與來源同檔案系統）
- 若目錄內尚無 ``dataset-metadata.json``，寫入預設（``--kaggle-user`` / ``--dataset-slug`` 可改）

請在 **sft_cot_bundle 根目錄** 執行::

    python3 scripts/prepare_kaggle_sft_cot_checkpoint_upload.py

**第一次**建立 Dataset::

    export KAGGLE_API_TOKEN='…'
    cd output/kaggle_upload_sft_cot_checkpoint
    kaggle datasets create -p . -r zip

**之後**只更新版本::

    kaggle datasets version -p . -m "s1200 full ckpt" -r zip

**背景上傳**（nohup + 日誌；讀 ``.api``）::

    ./scripts/kaggle_upload_sft_cot_bg.sh checkpoint "訊息"
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

_RE_SFT_COT_STEP = re.compile(r"^checkpoint_sft_cot_s(\d+)$")


def _bundle_root() -> Path:
    # scripts/export/ → bundle 根為上兩層
    return Path(__file__).resolve().parent.parent.parent


def find_latest_sft_cot_checkpoint_pt(output_dir: Path) -> Path:
    best_step = -1
    best_path: Path | None = None
    for p in output_dir.glob("checkpoint_sft_cot_s*.pt"):
        m = _RE_SFT_COT_STEP.match(p.stem)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_path = p
    if best_path is None:
        raise FileNotFoundError(
            f"在 {output_dir} 找不到 checkpoint_sft_cot_s*.pt（請先訓練或傳 --checkpoint）"
        )
    return best_path


def _write_default_metadata(
    kaggle_dir: Path,
    *,
    kaggle_user: str,
    dataset_slug: str,
    title: str,
) -> None:
    meta = {
        "title": title,
        "id": f"{kaggle_user}/{dataset_slug}",
        "licenses": [{"name": "unknown"}],
    }
    path = kaggle_dir / "dataset-metadata.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已寫入 {path}")


def main() -> None:
    root = _bundle_root()
    ap = argparse.ArgumentParser(description="準備 SFT-CoT 完整 .pt 給 Kaggle 上傳目錄")
    ap.add_argument(
        "--bundle-root",
        type=Path,
        default=root,
        help="sft_cot_bundle 根目錄",
    )
    ap.add_argument("--checkpoint", type=Path, default=None, help="指定 .pt；省略則選 step 最大")
    ap.add_argument(
        "--kaggle-user",
        default="s990093",
        help="Kaggle 使用者名（metadata id 前半）",
    )
    ap.add_argument(
        "--dataset-slug",
        default="latest-sft-cot-model-pt",
        help="Dataset slug（metadata id 後半）",
    )
    ap.add_argument(
        "--title",
        default="Latest SFT-CoT full checkpoint (.pt)",
        help="Dataset 標題",
    )
    ap.add_argument(
        "--force-metadata",
        action="store_true",
        help="覆寫 dataset-metadata.json（慎用，會改掉既有 id）",
    )
    args = ap.parse_args()
    bundle = args.bundle_root.expanduser().resolve()
    out_dir = bundle / "output"
    if not out_dir.is_dir():
        raise SystemExit(f"找不到 output 目錄: {out_dir}")

    src = args.checkpoint.expanduser().resolve() if args.checkpoint else find_latest_sft_cot_checkpoint_pt(out_dir)
    if not src.is_file():
        raise SystemExit(f"找不到 checkpoint: {src}")

    kaggle_dir = out_dir / "kaggle_upload_sft_cot_checkpoint"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    meta_path = kaggle_dir / "dataset-metadata.json"
    if args.force_metadata or not meta_path.is_file():
        _write_default_metadata(
            kaggle_dir,
            kaggle_user=args.kaggle_user,
            dataset_slug=args.dataset_slug,
            title=args.title,
        )
    else:
        print(f"沿用既有 {meta_path}（若要重寫请加 --force-metadata）")

    dst = kaggle_dir / "latest_sft_cot_model.pt"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.link(src, dst)
    mb = dst.stat().st_size / 1e6
    print(f"來源: {src.name} ({mb:.0f} MB)")
    print(f"Hard link: {dst} -> 同上 inode")

    print()
    print("下一步（勿把 token 貼到聊天）：")
    print("  export KAGGLE_API_TOKEN='…'")
    print(f"  cd {kaggle_dir}")
    print("  # 第一次：kaggle datasets create -p . -r zip")
    print("  # 之後：  kaggle datasets version -p . -m \"…\" -r zip")


if __name__ == "__main__":
    main()
