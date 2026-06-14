#!/usr/bin/env python3
"""Pack sft_cot_bundle（僅程式碼）為 zip：不含 dataset、權重、structure_weights、output。"""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

SKIP_SUFFIXES = {".pt", ".npz", ".pyc", ".log", ".png", ".html", ".bin", ".arrow"}
SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", "output", "dataset"}
SKIP_FILES = {".api", ".DS_Store"}
SKIP_NAME_SUBSTRINGS = ("structure_weights_bundle",)


def _bundle_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def should_skip(root: Path, path: Path) -> bool:
    rel = path.relative_to(root)
    if path.name in SKIP_FILES:
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    if any(s in path.name for s in SKIP_NAME_SUBSTRINGS):
        return True
    parts = rel.parts
    if not parts:
        return False
    if parts[0] in ("output", "dataset"):
        return True
    if "scripts" in parts and "output" in parts:
        return True
    if any(p in SKIP_DIR_NAMES for p in parts):
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="sft_cot_bundle 純程式碼 zip（無 dataset / 權重）")
    ap.add_argument("--bundle", type=Path, default=None, help="bundle 根目錄")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="輸出 zip（預設: ../sft_cot_bundle_code_only.zip）",
    )
    args = ap.parse_args()
    bundle = (args.bundle or _bundle_root()).resolve()
    out = args.output or (bundle.parent / "sft_cot_bundle_code_only.zip")
    out = out.resolve()

    n_files = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for dirpath, dirnames, filenames in os.walk(bundle):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
            for name in filenames:
                fp = Path(dirpath) / name
                if should_skip(bundle, fp):
                    continue
                arc = fp.relative_to(bundle.parent)
                zf.write(fp, arc.as_posix())
                n_files += 1

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Wrote {out}")
    print(f"  files: {n_files}  size: {size_mb:.2f} MiB")
    print("  含: scripts/, cot_task/（.py/.sh/.md/.json 報告，無 bundle.pt）")
    print("  不含: dataset/, output/, *.pt/*.npz/*.bin, structure_weights_bundle")


if __name__ == "__main__":
    main()
