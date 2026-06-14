#!/bin/bash
# ===========================================================================
# plot.sh — 自動畫訓練曲線（支援 terminal 模式 / 互動視窗 / PNG 輸出）
#
# 用法：
#   bash scripts/tools/plot.sh                    # 預設：儲存 PNG
#   bash scripts/tools/plot.sh -t                 # 終端機 ASCII 圖表
#   bash scripts/tools/plot.sh -s                 # 互動視窗
#   bash scripts/tools/plot.sh --ma 200 -dpi 300  # 自訂參數
#   bash scripts/tools/plot.sh -o my_plots.png    # 自訂輸出
#
# 環境變數（可覆蓋預設值）：
#   TRAIN_CSV  — 訓練 CSV 路徑
#   VAL_CSV    — 驗證 CSV 路徑
#   OUTPUT     — 輸出 PNG 路徑
#   DPI        — 輸出 DPI
#   SCRIPT     — 指定 plot script（預設 enhanced 版，設 simple 用精簡版）
# ===========================================================================
set -e

BUNDLE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BUNDLE_ROOT"

# ---- conda env ----
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "torch310" ]]; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate torch310 2>/dev/null || {
        echo "[!] 無法啟動 conda env torch310，嘗試使用系統 python ..."
    }
fi

# ---- 選擇 script ----
SCRIPT_CHOICE="${SCRIPT:-enhanced}"
if [[ "$SCRIPT_CHOICE" == "simple" ]]; then
    PY_SCRIPT="${BUNDLE_ROOT}/scripts/tools/plot_sft_train_val.py"
else
    PY_SCRIPT="${BUNDLE_ROOT}/scripts/tools/plot_sft_train_val_enhanced.py"
fi

# ---- 偵測使用者是否已指定 CSV 路徑（避免跟 env var 重複） ----
HAS_CSV_ARG=false
for a in "$@"; do
    if [[ "$a" == *.csv ]]; then
        HAS_CSV_ARG=true
        break
    fi
done

if ! $HAS_CSV_ARG; then
    TRAIN_CSV="${TRAIN_CSV:-output/train_sft_cot_log.csv}"
    VAL_CSV="${VAL_CSV:-output/val_sft_cot_log.csv}"

    ARGS=()
    [[ -f "$TRAIN_CSV" ]] || { echo "❌ 找不到 $TRAIN_CSV"; exit 1; }
    ARGS+=("$TRAIN_CSV")
    [[ -f "$VAL_CSV" ]] && ARGS+=("$VAL_CSV")
else
    ARGS=()
fi

echo "============================================"
echo "  SFT-CoT Plot"
echo "  Script : $(basename "$PY_SCRIPT")"
echo "  Env    : ${CONDA_DEFAULT_ENV:-system}"
echo "============================================"

PYTHONPATH="${BUNDLE_ROOT}/scripts" python3 "$PY_SCRIPT" "${ARGS[@]}" "$@"
