#!/usr/bin/env bash
# 背景執行 Kaggle Dataset 上傳（nohup），避免大檔上傳佔住前景終端。
#
# 用法（建議在 sft_cot_bundle 根目錄執行本腳本）::
#
#   ./scripts/kaggle_upload_sft_cot_bg.sh checkpoint "s1240 full ckpt"
#   ./scripts/kaggle_upload_sft_cot_bg.sh npz "refresh npz"
#
# 憑證：讀取 bundle 根目錄 ``.api``（``export KAGGLE_API_TOKEN=...``）。
# 日誌：output/logs/kaggle_upload_<mode>_YYYYMMDD_HHMMSS.log
# PID ：output/logs/kaggle_upload_<mode>.pid

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "$SCRIPT_DIR/../.." && pwd)"
API_FILE="$BUNDLE/.api"

MODE="${1:-checkpoint}"
shift || true
MSG="${*:-upload $(date -u +%Y-%m-%dT%H:%MZ)}"

if [[ ! -f "$API_FILE" ]]; then
  echo "❌ 找不到 $API_FILE（請建立並寫入 export KAGGLE_API_TOKEN=...）" >&2
  exit 1
fi

mkdir -p "$BUNDLE/output/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$BUNDLE/output/logs/kaggle_upload_${MODE}_${TS}.log"
PIDFILE="$BUNDLE/output/logs/kaggle_upload_${MODE}.pid"

case "$MODE" in
  checkpoint|npz) ;;
  *)
    echo "用法: $0 {checkpoint|npz} [版本說明…]" >&2
    exit 2
    ;;
esac

KAGGLE_BIN="$(command -v kaggle 2>/dev/null || true)"
if [[ -z "$KAGGLE_BIN" && -x "${HOME}/anaconda3/bin/kaggle" ]]; then
  KAGGLE_BIN="${HOME}/anaconda3/bin/kaggle"
fi
if [[ -z "$KAGGLE_BIN" ]]; then
  echo "❌ 找不到 kaggle CLI（請 conda activate mamba 或安裝 kaggle）" >&2
  exit 1
fi

nohup env BUNDLE="$BUNDLE" MODE="$MODE" MSG="$MSG" KAGGLE_BIN="$KAGGLE_BIN" bash -c '
  set -euo pipefail
  set -a
  # shellcheck source=/dev/null
  source "$BUNDLE/.api"
  set +a
  echo "=== $(date -Is) start mode=$MODE ==="
  if [[ "$MODE" == "checkpoint" ]]; then
    python3 "$BUNDLE/scripts/export/prepare_kaggle_sft_cot_checkpoint_upload.py"
    cd "$BUNDLE/output/kaggle_upload_sft_cot_checkpoint"
  else
    cd "$BUNDLE"
    python3 scripts/export/export_latest_sft_cot_model.py --no-pt-symlink
    KAGGLE_NPZ_DIR="$BUNDLE/output/kaggle_upload_latest_sft_cot"
    mkdir -p "$KAGGLE_NPZ_DIR"
    META="$KAGGLE_NPZ_DIR/dataset-metadata.json"
    if [[ ! -f "$META" ]]; then
      python3 -c "
import json
from pathlib import Path
p = Path(\"$META\")
p.write_text(json.dumps({
  \"title\": \"Latest SFT-CoT model weights (.npz)\",
  \"id\": \"s990093/latest-sft-cot-model-npz\",
  \"licenses\": [{\"name\": \"unknown\"}],
}, ensure_ascii=False, indent=2) + \"\\n\", encoding=\"utf-8\")
print(f\"已寫入 {p}\")
"
    fi
    rm -f "$KAGGLE_NPZ_DIR/latest_sft_cot_model.npz"
    ln -f "$BUNDLE/output/latest_sft_cot_model.npz" \
      "$KAGGLE_NPZ_DIR/latest_sft_cot_model.npz"
    cd "$KAGGLE_NPZ_DIR"
  fi
  if "$KAGGLE_BIN" datasets version -p . -m "$MSG" -r zip; then
    echo "=== $(date -Is) version OK ==="
  else
    echo "=== $(date -Is) version failed, try create ==="
    "$KAGGLE_BIN" datasets create -p . -r zip
    echo "=== $(date -Is) create finished ==="
  fi
' >>"$LOG" 2>&1 &

echo $! >"$PIDFILE"
echo "背景上傳已啟動  PID=$(cat "$PIDFILE")"
echo "模式          $MODE"
echo "日誌          $LOG"
echo "追蹤          tail -f $LOG"
