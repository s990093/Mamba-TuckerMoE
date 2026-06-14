#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自動選 **step 最大** 的 ``output/checkpoint_sft_cot_s*.pt``，匯出權重為固定檔名：

- ``output/latest_sft_cot_model.npz``（與 ``export_ckpt_to_npz.py`` 相同：僅 ``model`` 張量）
- 可選：``output/latest_sft_cot_model.pt`` → 符號連結指向該次使用的 .pt（方便腳本固定路徑）

請在 **sft_cot_bundle 根目錄** 執行::

    python3 scripts/export_latest_sft_cot_model.py

或指定輸出目錄::

    python3 scripts/export_latest_sft_cot_model.py --output-dir /path/to/output
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# 與 `python3 scripts/xxx.py` 搭配：同目錄模組可 import
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from export_ckpt_to_npz import export_pt_to_npz  # noqa: E402

_RE_SFT_COT_STEP = re.compile(r"^checkpoint_sft_cot_s(\d+)$")


def _default_bundle_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def find_latest_sft_cot_checkpoint(output_dir: Path) -> Path:
    """``checkpoint_sft_cot_s{step}.pt`` 中取 step 最大者。"""
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
            f"在 {output_dir} 找不到 checkpoint_sft_cot_s*.pt（請先完成訓練或指定 --checkpoint）"
        )
    return best_path


def _refresh_symlink(link_path: Path, target: Path) -> None:
    """在與 target 同目錄下建立相對連結 link_path -> target。"""
    link_path = link_path.resolve()
    target = target.resolve()
    if link_path.parent != target.parent:
        raise ValueError("連結與目標須在同一目錄，請改用 --output-dir 將兩者放在一起。")
    rel = os.path.relpath(target, start=link_path.parent)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    os.symlink(rel, link_path)


def main() -> None:
    root = _default_bundle_root()
    ap = argparse.ArgumentParser(description="最新 SFT-CoT checkpoint → latest_sft_cot_model.npz")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"checkpoint / 輸出目錄（預設: {root / 'output'})",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="不自動偵測時，手動指定 .pt",
    )
    ap.add_argument(
        "--weights-only",
        action="store_true",
        help="傳給 torch.load(..., weights_only=True)",
    )
    ap.add_argument(
        "--no-pt-symlink",
        action="store_true",
        help="不要建立 latest_sft_cot_model.pt 符號連結",
    )
    args = ap.parse_args()
    out_dir = (args.output_dir or (root / "output")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    src = args.checkpoint.expanduser().resolve() if args.checkpoint else find_latest_sft_cot_checkpoint(out_dir)
    if not src.is_file():
        raise SystemExit(f"找不到 checkpoint: {src}")

    dst_npz = out_dir / "latest_sft_cot_model.npz"
    print(f"來源 checkpoint (step 檔): {src.name}")
    export_pt_to_npz(src, dst_npz, weights_only=args.weights_only)

    if not args.no_pt_symlink:
        link_pt = out_dir / "latest_sft_cot_model.pt"
        _refresh_symlink(link_pt, src)
        print(f"已建立符號連結: {link_pt.name} -> {src.name}")


if __name__ == "__main__":
    main()
