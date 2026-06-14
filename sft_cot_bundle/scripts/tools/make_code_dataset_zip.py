#!/usr/bin/env python3
"""Pack sft_cot_bundle (code + dataset only) into a zip; skip weights/logs/checkpoints."""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

SKIP_SUFFIXES = {".pt", ".npz", ".pyc", ".log", ".png", ".html"}
SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", "output"}
SKIP_FILES = {".api", ".DS_Store"}


def should_skip(root: Path, path: Path) -> bool:
    rel = path.relative_to(root)
    if path.name in SKIP_FILES:
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    parts = rel.parts
    if parts and parts[0] == "output":
        return True
    if "scripts" in parts and "output" in parts:
        return True
    if any(p in SKIP_DIR_NAMES for p in parts):
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bundle",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="sft_cot_bundle root",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="zip path (default: parent/sft_cot_bundle_code_dataset.zip)",
    )
    args = ap.parse_args()
    bundle = args.bundle.resolve()
    out = args.output or (bundle.parent / "sft_cot_bundle_code_dataset.zip")
    out = out.resolve()

    n_files = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for dirpath, dirnames, filenames in os.walk(bundle):
            dirnames[:] = [
                d
                for d in dirnames
                if d not in SKIP_DIR_NAMES and d != "output"
            ]
            for name in filenames:
                fp = Path(dirpath) / name
                if should_skip(bundle, fp):
                    continue
                arc = fp.relative_to(bundle.parent)
                zf.write(fp, arc.as_posix())
                n_files += 1

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Wrote {out}")
    print(f"  files: {n_files}  size: {size_mb:.1f} MiB")


if __name__ == "__main__":
    main()
