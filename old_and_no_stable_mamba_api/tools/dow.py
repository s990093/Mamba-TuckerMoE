#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download HyMamba3 PyTorch weights from Kaggle Hub and optionally install them as the repo checkpoint.

  python dow.py                    # download only, print path
  python dow.py --install          # copy to ./checkpoint.pt and remove ./checkpoint.npz (MLX sidecar)

Requires Kaggle credentials for private models (see https://www.kaggle.com/docs/api).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _ensure_kagglehub() -> object:
    try:
        import kagglehub  # type: ignore
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub", "-q"])
        import kagglehub  # type: ignore
    return kagglehub


def _find_checkpoint_pt(root: Path) -> Optional[Path]:
    for p in sorted(root.rglob("checkpoint.pt")):
        if p.is_file():
            return p
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Kaggle Hub → HyMamba3 checkpoint.pt")
    p.add_argument(
        "--model",
        default="mannyhsu/hymamba3/pyTorch/default",
        help="kagglehub.model_download() handle (owner/slug/framework/variation)",
    )
    p.add_argument(
        "--install",
        action="store_true",
        help=f"Copy weights to {REPO_ROOT / 'checkpoint.pt'} and delete {REPO_ROOT / 'checkpoint.npz'} if present",
    )
    p.add_argument(
        "--backup",
        action="store_true",
        help="If ./checkpoint.pt exists, rename to checkpoint.pt.bak before install",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Project root for --install (default: directory containing this script)",
    )
    args = p.parse_args()

    kagglehub = _ensure_kagglehub()
    print(f"Downloading model: {args.model!r} …")
    path_str = kagglehub.model_download(args.model)
    base = Path(path_str)
    print("Path to model folder:", base)

    ck = _find_checkpoint_pt(base)
    if ck is None:
        print("ERROR: no checkpoint.pt under download path.", file=sys.stderr)
        return 1
    print("Found weights:", ck, f"({ck.stat().st_size / 1e9:.2f} GB)")

    if not args.install:
        print("\nDone (download only). To use as repo default weights, run:")
        print(f"  python {Path(__file__).name} --install")
        return 0

    repo = args.repo_root.resolve()
    repo.mkdir(parents=True, exist_ok=True)
    dest = repo / "checkpoint.pt"
    npz_sidecar = repo / "checkpoint.npz"

    if dest.exists() and args.backup:
        bak = repo / "checkpoint.pt.bak"
        if bak.exists():
            bak.unlink()
        dest.rename(bak)
        print("Backed up existing checkpoint.pt → checkpoint.pt.bak")

    shutil.copy2(ck, dest)
    print("Installed:", dest)

    if npz_sidecar.is_file():
        npz_sidecar.unlink()
        print("Removed stale MLX sidecar:", npz_sidecar, "(will be regenerated on first .pt load if applicable)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
