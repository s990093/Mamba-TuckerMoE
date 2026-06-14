#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從完整 SFT checkpoint（含 optimizer）取出 model，另存為輕量「僅權重」.pt，
供 `train_sft` 以 `SFT_INIT_WEIGHTS_ONLY=True` 從 step=0 在新資料集上開訓。

格式與 repo 根目錄 `export_checkpoint_weights.py` 一致：state_dict + config + step（僅日誌）。
不含 `sft: True`，載入時會走 weights_only → 新 optimizer/scheduler。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT full ckpt → model-only .pt for fresh SFT")
    ap.add_argument(
        "checkpoint",
        type=Path,
        help="例如 output/checkpoint_sft_cot_s1240.pt",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("dataset/sft_cot_base_model.pt"),
        help="輸出路徑（預設 dataset/sft_cot_base_model.pt）",
    )
    args = ap.parse_args()
    src = args.checkpoint.expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"找不到檔案: {src}")
    print(f"Loading {src} …")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise SystemExit("checkpoint 需含 'model' 鍵（完整 SFT 存檔）。")
    out = {
        "state_dict": ckpt["model"],
        "step": int(ckpt.get("step", -1)),
        "vocab_size": 32007,
        "source_checkpoint": str(src),
        "config": ckpt.get("config"),
    }
    dst = args.out
    if not dst.is_absolute():
        dst = Path.cwd() / dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    mb = dst.stat().st_size / 1e6
    print(f"Saved -> {dst} ({mb:.1f} MB)")
    print("Next: 更新資料後跑 stf_cot_to_bin，再訓練；train_sft_cot 會優先載入此檔（見 train_sft_cot.py）。")


if __name__ == "__main__":
    main()
