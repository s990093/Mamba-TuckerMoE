#!/bin/bash
# 下載 latest-sft-cot-model-pt，連結 checkpoint + train/val CSV 到 output/ 供完整續跑
set -euo pipefail
BUNDLE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BUNDLE"
mkdir -p output

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "torch310" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate torch310
fi

export PYTHONNOUSERSITE=1
python3 - <<'PY'
import re
from pathlib import Path

import kagglehub

KAGGLE_DATASET = "s990093/latest-sft-cot-model-pt"
bundle = Path(".").resolve()
out_dir = bundle / "output"

print(f"Downloading {KAGGLE_DATASET} …")
root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
candidates = sorted(root.rglob("*.pt"), key=lambda p: p.stat().st_size, reverse=True)
if not candidates:
    raise SystemExit(f"在 {root} 找不到 .pt")
src = candidates[0]
for prefer in ("checkpoint_sft_cot", "latest_sft_cot", "checkpoint"):
    for p in candidates:
        if prefer in p.name.lower():
            src = p
            break

step = -1
m = re.search(r"checkpoint_sft_cot_s(\d+)", src.name)
if m:
    step = int(m.group(1))
else:
  # 嘗試從 checkpoint 內讀 step
    import torch
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        step = int(ckpt.get("global_step", ckpt.get("step", -1)))

if step >= 0:
    dst_name = f"checkpoint_sft_cot_s{step}.pt"
else:
    dst_name = "checkpoint_sft_cot_s_kaggle.pt"
dst = out_dir / dst_name
if dst.exists() and dst.samefile(src):
    print(f"Already linked: {dst}")
elif dst.exists():
    dst.unlink()
    dst.symlink_to(src.resolve())
    print(f"Updated symlink: {dst} -> {src}")
else:
    dst.symlink_to(src.resolve())
    print(f"Linked: {dst} -> {src}")

ver_dir = src.parent
for csv_name in ("train_sft_cot_log.csv", "val_sft_cot_log.csv"):
    csv_src = ver_dir / csv_name
    if not csv_src.is_file():
        print(f"⚠️  略過（不存在）: {csv_src}")
        continue
    csv_dst = out_dir / csv_name
    if csv_dst.exists() or csv_dst.is_symlink():
        csv_dst.unlink()
    csv_dst.symlink_to(csv_src.resolve())
    print(f"Linked: {csv_dst} -> {csv_src}")

print(f"Resume with: SFT_COT_RESUME_CKPT={dst.relative_to(bundle)} ./scripts/training/start_training.sh")
PY
