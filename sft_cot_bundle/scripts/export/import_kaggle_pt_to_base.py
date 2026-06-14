#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從 Kaggle ``latest-sft-cot-model-pt`` 下載 checkpoint，轉成僅權重的 ``dataset/sft_cot_base_model.pt``。

用法（bundle 根目錄）::

    python3 scripts/export/import_kaggle_pt_to_base.py --download
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

KAGGLE_DATASET = "s990093/latest-sft-cot-model-pt"
DEFAULT_OUT = Path("dataset/sft_cot_base_model.pt")


def _bundle_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def download_pt() -> Path:
    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    root = Path(path)
    candidates = sorted(root.rglob("*.pt"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"在 {root} 找不到 .pt")
    for prefer in ("checkpoint_sft_cot", "latest_sft_cot", "checkpoint"):
        for p in candidates:
            if prefer in p.name.lower():
                return p.resolve()
    return candidates[0].resolve()


def ckpt_to_base(src: Path, out: Path) -> None:
    print(f"Loading {src} …")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
        step = int(ckpt.get("global_step", ckpt.get("step", -1)))
        config = ckpt.get("config")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        step = int(ckpt.get("step", -1))
        config = ckpt.get("config")
    else:
        raise SystemExit("不支援的 checkpoint 格式（需含 model 或 state_dict）")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": sd,
            "step": step,
            "vocab_size": 32007,
            "source_checkpoint": str(src.resolve()),
            "source_kaggle": KAGGLE_DATASET,
            "config": config,
        },
        out,
    )
    mb = out.stat().st_size / 1e6
    print(f"Saved -> {out} ({mb:.0f} MB, {len(sd)} tensors, logged step={step})")
    print("訓練時會以 SFT_INIT_WEIGHTS_ONLY 從 step=0 開跑（見 train_sft_cot.py）。")


def main() -> None:
    ap = argparse.ArgumentParser(description="Kaggle PT → dataset/sft_cot_base_model.pt")
    ap.add_argument("--download", action="store_true", help="kagglehub 下載後匯入")
    ap.add_argument("--pt", type=Path, default=None, help="本機 .pt（省略則 --download）")
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bundle-root", type=Path, default=None)
    args = ap.parse_args()
    bundle = (args.bundle_root or _bundle_root()).resolve()

    if args.download or args.pt is None:
        print(f"Downloading {KAGGLE_DATASET} …")
        src = download_pt()
        print(f"Downloaded: {src}")
    else:
        src = args.pt.expanduser().resolve()
        if not src.is_file():
            raise SystemExit(f"找不到: {src}")

    out = args.out if args.out.is_absolute() else bundle / args.out
    ckpt_to_base(src, out.resolve())


if __name__ == "__main__":
    main()
