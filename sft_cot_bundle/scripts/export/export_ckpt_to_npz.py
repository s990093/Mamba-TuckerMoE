#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 PyTorch checkpoint（.pt / .pth / .bin）中的權重張量匯出為 .npz（每鍵一個 numpy 陣列）。

邏輯對齊常見的「僅讀權重」載入：若頂層為 dict 且含 ``model`` 或 ``state_dict``，只匯出該子 dict
的浮點張量（``bfloat16`` / ``float16`` 先轉 float32 再 ``.numpy()``）。

用法（在 bundle 根目錄，或任意目錄搭配絕對路徑）::

    python3 scripts/export_ckpt_to_npz.py output/checkpoint_sft_cot_s40.pt \\
        -o output/checkpoint_sft_cot_s40_weights.npz
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _state_dict_from_checkpoint(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "model" in obj:
        raw = obj["model"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        raw = obj["state_dict"]
    elif isinstance(obj, dict):
        raw = obj
    else:
        raise TypeError(f"無法從型別 {type(obj)} 取得 state_dict")
    if not isinstance(raw, dict):
        raise TypeError("預期 state_dict 為 dict")
    out: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if torch.is_tensor(v):
            out[str(k)] = v
    return out


def export_pt_to_npz(
    src: Path,
    dst: Path,
    *,
    weights_only: bool = False,
) -> None:
    src = src.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"找不到檔案: {src}")

    print(f"Loading {src} (weights_only={weights_only}) …")
    ckpt = torch.load(str(src), map_location="cpu", weights_only=weights_only)
    sd = _state_dict_from_checkpoint(ckpt)
    del ckpt

    np_dict: dict[str, np.ndarray] = {}
    for name, t in sd.items():
        np_dict[name] = t.detach().cpu().float().numpy()
    del sd

    dst = dst.expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(np_dict)} tensors -> {dst} …")
    np.savez_compressed(str(dst), **np_dict)
    mb = dst.stat().st_size / 1e6
    print(f"Done. Size ≈ {mb:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description="PyTorch checkpoint -> compressed .npz (model weights)")
    ap.add_argument("checkpoint", type=Path, help="例如 output/checkpoint_sft_cot_s40.pt")
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="輸出 .npz（預設：與來源同目錄，副檔名改為 _weights.npz）",
    )
    ap.add_argument(
        "--weights-only",
        action="store_true",
        help="以 torch.load(..., weights_only=True) 載入（若遇錯可改不加此旗標）",
    )
    args = ap.parse_args()
    out = args.out
    if out is None:
        out = args.checkpoint.with_name(args.checkpoint.stem + "_weights.npz")
    export_pt_to_npz(args.checkpoint, out, weights_only=args.weights_only)


if __name__ == "__main__":
    main()
