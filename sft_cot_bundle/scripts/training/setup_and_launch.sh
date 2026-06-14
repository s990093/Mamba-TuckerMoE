#!/bin/bash
# ==============================================================================
#  setup_and_launch.sh — 一鍵生成前置檔案 + 啟動 3-GPU 訓練
#
#  會依序：
#    1. conda activate torch310
#    2. 生成 SFT-GO 結構權重 (structure_weights_bundle.pt)
#    3. 生成 PDL 權重 (pdl_weights.pt)
#    4. 設定環境變數 (ENABLE_PDL=true, ENABLE_SFTGO=true)
#    5. 啟動 3 卡訓練 (GPU 1,2,3, grad_accum=2)
#    6. 自動續跑最新 checkpoint
#
#  用法：
#    ./scripts/training/setup_and_launch.sh
#    ./scripts/training/setup_and_launch.sh --force-rebuild  (強制重建)
# ==============================================================================

set -e
export TZ=Asia/Taipei

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

FORCE_REBUILD=false
[[ "${1:-}" == "--force-rebuild" ]] && FORCE_REBUILD=true

# ── conda env ──────────────────────────────────────────────────────────
echo "🔧 Activating torch310 ..."
eval "$(conda shell.bash hook)"
conda activate torch310
export PYTHONPATH="$PROJECT_ROOT/scripts:$PROJECT_ROOT/scripts/data:${PYTHONPATH:-}"

# ── Step 1: SFT-GO 結構權重 ───────────────────────────────────────────
BUNDLE="$PROJECT_ROOT/cot_task/reports/structure_weights_bundle.pt"
mkdir -p "$(dirname "$BUNDLE")"

if [[ "$FORCE_REBUILD" == true || ! -f "$BUNDLE" ]]; then
    echo ""
    echo "════════════════════════════════════════════"
    echo "  Step 1/2: 生成 SFT-GO 結構權重"
    echo "════════════════════════════════════════════"
    python cot_task/build_structure_weights.py \
        --hf-path dataset/stf_cot_hf \
        --tokenizer-dir dataset/tokenizer \
        --output-dir cot_task/reports
    echo "✅ SFT-GO: $BUNDLE"
else
    echo "⏭   SFT-GO bundle already exists → skip"
fi

# ── Step 2: PDL 權重 ───────────────────────────────────────────────────
PDL="$PROJECT_ROOT/cot_task/reports/pdl_weights.pt"

if [[ "$FORCE_REBUILD" == true || ! -f "$PDL" ]]; then
    echo ""
    echo "════════════════════════════════════════════"
    echo "  Step 2/2: 生成 PDL 權重"
    echo "════════════════════════════════════════════"
    python scripts/precompute_pdl_weights.py
    echo "✅ PDL: $PDL"
else
    echo "⏭   PDL weights already exist → skip"
fi

# ── Step 3: 啟動訓練 ───────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  Step 3/3: 啟動 3-GPU 訓練"
echo "════════════════════════════════════════════"

# 啟用 PDL + SFT-GO + FCP + SCALe
export SFT_COT_ENABLE_PDL=true
export SFT_COT_ENABLE_SFTGO=true
export SFT_COT_ENABLE_FCP=true
export SFT_COT_ENABLE_SCALE=true
export SFT_COT_ENABLE_FINAL_ENHANCE=true
export SFT_COT_ENABLE_FINAL_SW=true
export SFT_COT_AUTO_RESUME=1

# 3 卡配置
export GPU_DEVICES="1,2,3"
export NUM_GPUS=3
export GRAD_ACCUM=2

# 呼叫 start_training.sh (它現在讀 GPU_DEVICES/NUM_GPUS/GRAD_ACCUM 環境變數)
bash "$PROJECT_ROOT/scripts/training/start_training.sh"

echo ""
echo "════════════════════════════════════════════"
echo "  ✅ Setup & Launch complete"
echo "════════════════════════════════════════════"
