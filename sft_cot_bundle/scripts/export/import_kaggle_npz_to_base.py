#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從 Kaggle ``latest-sft-cot-model-npz``（或本機 .npz）匯入權重，寫成 ``dataset/sft_cot_base_model.pt``。

用法（bundle 根目錄）::

    python3 scripts/export/import_kaggle_npz_to_base.py
    python3 scripts/export/import_kaggle_npz_to_base.py --npz /path/to/latest_sft_cot_model.npz --step 3600

或先下載再匯入::

    python3 scripts/export/import_kaggle_npz_to_base.py --download
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

KAGGLE_DATASET = "s990093/latest-sft-cot-model-npz"
DEFAULT_OUT = Path("dataset/sft_cot_base_model.pt")
_META_KEYS = frozenset(
    {"config_json", "global_step", "sft", "router_temperature", "step", "vocab_size"}
)


def _bundle_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def download_npz() -> Path:
    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    root = Path(path)
    candidates = list(root.rglob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"在 {root} 找不到 .npz")
    # 優先固定檔名
    for name in ("latest_sft_cot_model.npz",):
        for p in candidates:
            if p.name == name:
                return p.resolve()
    return max(candidates, key=lambda p: p.stat().st_size).resolve()


def _load_config_from_npz(data: np.lib.npyio.NpzFile) -> dict | None:
    if "config_json" not in data.files:
        return None
    raw = data["config_json"]
    if raw.dtype == object:
        s = raw.item()
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        if isinstance(s, str):
            return json.loads(s)
    return None


def _load_config_fallback(bundle: Path) -> dict | None:
    for rel in (
        "dataset/sft_cot_base_model.pt",
        "output/checkpoint_sft_cot_s3600.pt",
    ):
        p = bundle / rel
        if not p.is_file():
            continue
        try:
            raw = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(raw, dict) and raw.get("config"):
                return raw["config"]
            if isinstance(raw, dict) and "model" in raw:
                return raw.get("config")
        except Exception:
            continue
    return None


def npz_to_state_dict(npz_path: Path) -> dict[str, torch.Tensor]:
    data = np.load(npz_path, allow_pickle=True)
    weight_keys = [k for k in data.files if k not in _META_KEYS]
    strip_model_prefix = bool(weight_keys) and all(k.startswith("model_") for k in weight_keys)
    sd: dict[str, torch.Tensor] = {}
    for key in weight_keys:
        arr = data[key]
        if not isinstance(arr, np.ndarray) or arr.dtype == object:
            continue
        name = key[6:] if strip_model_prefix and key.startswith("model_") else key
        t = torch.from_numpy(np.asarray(arr))
        if t.dtype in (torch.float64, torch.float16, torch.bfloat16):
            t = t.float()
        sd[name] = t
    data.close()
    if not sd:
        raise ValueError(f"{npz_path} 內沒有可用的權重陣列")
    return sd


def save_base_model(
    npz_path: Path,
    out_path: Path,
    *,
    step: int,
    bundle: Path,
    source: str,
) -> None:
    sd = npz_to_state_dict(npz_path)
    config = _load_config_from_npz(np.load(npz_path, allow_pickle=True)) or _load_config_fallback(
        bundle
    )
    out = {
        "state_dict": sd,
        "step": int(step),
        "vocab_size": 32007,
        "source_checkpoint": source,
        "source_npz": str(npz_path.resolve()),
        "config": config,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    mb = out_path.stat().st_size / 1e6
    print(f"Saved {out_path} ({mb:.0f} MB, {len(sd)} tensors, step={step})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Kaggle NPZ → dataset/sft_cot_base_model.pt")
    ap.add_argument("--download", action="store_true", help="先用 kagglehub 下載 dataset")
    ap.add_argument("--npz", type=Path, default=None, help="本機 .npz 路徑（省略則 --download 或預設 cache）")
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--step", type=int, default=3600, help="寫入檔內 step（僅日誌；SFT 新開仍從 0）")
    ap.add_argument("--bundle-root", type=Path, default=None)
    args = ap.parse_args()
    bundle = (args.bundle_root or _bundle_root()).resolve()

    if args.download or args.npz is None:
        print(f"Downloading {KAGGLE_DATASET} …")
        npz_path = download_npz()
        print(f"Downloaded: {npz_path}")
    else:
        npz_path = args.npz.expanduser().resolve()
        if not npz_path.is_file():
            raise SystemExit(f"找不到: {npz_path}")

    out = args.out
    if not out.is_absolute():
        out = bundle / out
    save_base_model(
        npz_path,
        out.resolve(),
        step=args.step,
        bundle=bundle,
        source=KAGGLE_DATASET,
    )
    print("train_sft_cot 在無 output/checkpoint_sft_cot_s*.pt 時會優先載入此 base。")


if __name__ == "__main__":
    main()
