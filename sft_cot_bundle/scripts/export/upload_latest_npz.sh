#!/usr/bin/env bash
# 一鍵將 SFT-CoT checkpoint 轉 .npz 並上傳 Kaggle。
#
# 用法（在 sft_cot_bundle 根目錄）:
#
#   ./scripts/export/upload_latest_npz.sh "版本說明"
#   ./scripts/export/upload_latest_npz.sh --ckpt checkpoint_sft_cot_s450.pt "版本說明"
#
# 憑證：自動讀取 bundle/.api 或上層 repo 根目錄的 .api（KAGGLE_API_TOKEN）

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "$SCRIPT_DIR/../.." && pwd)"
HANDLE="s990093/latest-sft-cot-model-npz"

# ---- 參數解析 ----------------------------------------------------------
CKPT_ARG=""
MSG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt|-c)
      CKPT_ARG="$2"; shift 2 ;;
    *)
      MSG="$*"; break ;;
  esac
done
MSG="${MSG:-upload $(date -u +%Y-%m-%dT%H:%MZ)}"
CKPT_PATH="$BUNDLE/output/${CKPT_ARG##*/}"
[[ -n "$CKPT_ARG" && ! -f "$CKPT_PATH" ]] && CKPT_PATH="$CKPT_ARG"

# ---- 找 .api ----------------------------------------------------------
API_FILE=""
for candidate in "$BUNDLE/.api" "$BUNDLE/../.api"; do
  if [[ -f "$candidate" ]]; then API_FILE="$candidate"; break; fi
done
if [[ -z "$API_FILE" ]]; then
  echo "❌ 找不到 .api（$BUNDLE/.api 或 $BUNDLE/../.api）" >&2; exit 1
fi
set -a; source "$API_FILE"; set +a
echo "已載入 $API_FILE"

# ---- Conda torch310 ---------------------------------------------------
conda_init() {
  for f in "${HOME}/miniforge3/etc/profile.d/conda.sh" \
           "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$f" ]]; then source "$f"; return 0; fi
  done
  return 1
}

# ---- Step 1: .pt → .npz (torch310 env) ---------------------------------
echo "=== Step 1: 匯出 checkpoint 為 .npz ==="
cd "$BUNDLE"

CKPT_INFO_FILE="$BUNDLE/output/.latest_ckpt_info.json"
if [[ "${CONDA_DEFAULT_ENV:-}" != "torch310" ]]; then
  set +u; conda_init; conda activate torch310; set -u
fi

if [[ -n "$CKPT_ARG" ]]; then
  python3 "$SCRIPT_DIR/export_ckpt_to_npz.py" "$CKPT_PATH" \
    -o "$BUNDLE/output/latest_sft_cot_model.npz"
else
  python3 "$SCRIPT_DIR/export_latest_sft_cot_model.py" --no-pt-symlink
fi

# 寫入 checkpoint 來源資訊
python3 -c "
import json, re, os
from pathlib import Path

out_dir = Path('$BUNDLE/output')

if '$CKPT_ARG':
    p = Path('$CKPT_PATH')
    best_path = p
    best_step = 0
    m = re.match(r'^checkpoint_sft_cot_s(\d+)$', p.stem)
    if m: best_step = int(m.group(1))
else:
    best_step = -1; best_path = None
    for p in out_dir.glob('checkpoint_sft_cot_s*.pt'):
        m = re.match(r'^checkpoint_sft_cot_s(\d+)$', p.stem)
        if m:
            s = int(m.group(1))
            if s > best_step:
                best_step = s
                best_path = p

if best_path and best_path.is_file():
    info = {
        'source_checkpoint': best_path.name,
        'source_step': best_step,
        'source_size_mb': round(best_path.stat().st_size / 1e6, 1),
        'npz_size_mb': round((out_dir / 'latest_sft_cot_model.npz').stat().st_size / 1e6, 1),
    }
    Path('$CKPT_INFO_FILE').write_text(json.dumps(info, indent=2))
    print(f'來源: {best_path.name} ({info[\"source_size_mb\"]} MB) → npz {info[\"npz_size_mb\"]} MB')
"

# ---- Step 2: 準備 Kaggle 上傳目錄 --------------------------------------
KAGGLE_DIR="$BUNDLE/output/kaggle_upload_latest_sft_cot"
mkdir -p "$KAGGLE_DIR"

rm -f "$KAGGLE_DIR/latest_sft_cot_model.npz"
ln "$BUNDLE/output/latest_sft_cot_model.npz" "$KAGGLE_DIR/latest_sft_cot_model.npz"

# ---- Step 3: 上傳（base python3 + kagglehub，不能是 torch310！）--------
echo "=== Step 3: 上傳 Kaggle ($HANDLE) ==="
conda deactivate 2>/dev/null || true

"${HOME}/miniforge3/bin/python3" <<PY
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

TZ = timezone(timedelta(hours=8))
bundle     = Path("$BUNDLE")
kaggle_dir = bundle / "output" / "kaggle_upload_latest_sft_cot"

ckpt_info = {}
info_file = bundle / "output" / ".latest_ckpt_info.json"
if info_file.is_file():
    ckpt_info = json.loads(info_file.read_text())

now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

title_parts = ["Latest SFT-CoT model weights (.npz)"]
if ckpt_info:
    title_parts.append(f"[step={ckpt_info.get('source_step', '?')}]")

meta = {
    "title": " ".join(title_parts),
    "id": "$HANDLE",
    "licenses": [{"name": "unknown"}],
    "source": ckpt_info,
    "uploaded_at": now,
}
meta_path = kaggle_dir / "dataset-metadata.json"
meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"已寫入 metadata: source={ckpt_info.get('source_checkpoint', '?')}")

npz = kaggle_dir / "latest_sft_cot_model.npz"
print(f"上傳中: $HANDLE  ({npz.stat().st_size / 1e6:.0f} MB)")
import kagglehub
kagglehub.dataset_upload(handle="$HANDLE", local_dataset_dir=str(kaggle_dir), version_notes="$MSG")
print("=== 上傳完成 ===")
PY