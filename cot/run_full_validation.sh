#!/bin/bash
# -*- coding: utf-8 -*-
#
# 一鍵運行完整驗證流程（Step 1-4）
#
# 用法：
#   bash run_full_validation.sh
#   bash run_full_validation.sh --sample-count 100  # 快速驗證模式
#
# 環境變數可覆蓋預設值：
#   HF_PATH, TOKENIZER_DIR, OUTPUT_DIR, W_STRUCT, SEQ_LEN

set -e

# ===== 激活虛擬環境 =====
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# ===== 預設配置 =====
HF_PATH="${HF_PATH:-$PROJECT_ROOT/sft_cot_bundle/dataset/stf_cot_hf}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$PROJECT_ROOT/sft_cot_bundle/dataset/tokenizer}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/cot/reports}"
COT_DIR="$PROJECT_ROOT/cot"
W_STRUCT="${W_STRUCT:-3.0}"
SEQ_LEN="${SEQ_LEN:-512}"
SAMPLE_COUNT="${SAMPLE_COUNT:-}"
TOTAL_STEPS="${TOTAL_STEPS:-10000}"

# 解析命令列引數
while [[ $# -gt 0 ]]; do
    case $1 in
        --sample-count)
            SAMPLE_COUNT="$2"
            shift 2
            ;;
        --w-struct)
            W_STRUCT="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --help)
            echo "用法: bash run_full_validation.sh [options]"
            echo ""
            echo "選項："
            echo "  --sample-count N    只處理前 N 筆樣本（快速驗證）"
            echo "  --w-struct W        結構權重乘數（預設 3.0）"
            echo "  --seq-len L         序列長度（預設 512）"
            echo "  --help              顯示此幫助信息"
            exit 0
            ;;
        *)
            echo "未知選項: $1"
            exit 1
            ;;
    esac
done

# ===== 檢查前置條件 =====
echo "=========================================="
echo "CoT 驗證流程：前置檢查"
echo "=========================================="

if [ ! -d "$HF_PATH" ]; then
    echo "❌ HF dataset 不存在: $HF_PATH"
    exit 1
fi
echo "✓ HF dataset: $HF_PATH"

if [ ! -d "$TOKENIZER_DIR" ]; then
    echo "❌ Tokenizer 不存在: $TOKENIZER_DIR"
    exit 1
fi
echo "✓ Tokenizer: $TOKENIZER_DIR"

mkdir -p "$OUTPUT_DIR"
echo "✓ Output dir: $OUTPUT_DIR"

# ===== Step 1: 生成結構權重 =====
echo ""
echo "=========================================="
echo "Step 1: 生成結構權重 (build_structure_weights.py)"
echo "=========================================="

CMD="python $COT_DIR/build_structure_weights.py \
  --hf-path $HF_PATH \
  --tokenizer-dir $TOKENIZER_DIR \
  --output-dir $OUTPUT_DIR/structure_weights \
  --w-struct $W_STRUCT"

if [ -n "$SAMPLE_COUNT" ]; then
    CMD="$CMD --sample-count $SAMPLE_COUNT"
fi

echo "執行: $CMD"
cd "$COT_DIR"
eval "$CMD"

# ===== Step 2: 可視化驗證 =====
echo ""
echo "=========================================="
echo "Step 2: 可視化驗證 (visualize_structure_weights.py)"
echo "=========================================="

SAMPLE_VIS="${SAMPLE_COUNT:-20}"
CMD="python $COT_DIR/visualize_structure_weights.py \
  --hf-path $HF_PATH \
  --weights-dir $OUTPUT_DIR/structure_weights \
  --tokenizer-dir $TOKENIZER_DIR \
  --output-dir $OUTPUT_DIR \
  --sample-count $SAMPLE_VIS"

echo "執行: $CMD"
eval "$CMD"

# ===== Step 3: 統計圖表 =====
echo ""
echo "=========================================="
echo "Step 3: 統計圖表 (validate_and_plot.py)"
echo "=========================================="

CMD="python $COT_DIR/validate_and_plot.py \
  --hf-path $HF_PATH \
  --weights-dir $OUTPUT_DIR/structure_weights \
  --tokenizer-dir $TOKENIZER_DIR \
  --output-dir $OUTPUT_DIR \
  --seq-len $SEQ_LEN \
  --total-steps $TOTAL_STEPS"

echo "執行: $CMD"
eval "$CMD"

# ===== Step 4: 單元測試 =====
echo ""
echo "=========================================="
echo "Step 4: 單元測試 (test_loss_weights.py)"
echo "=========================================="

if command -v pytest &> /dev/null; then
    CMD="pytest $COT_DIR/test_loss_weights.py -v --tb=short"
    echo "執行: $CMD"
    eval "$CMD" || true  # 不中斷流程
else
    echo "⚠️  pytest 未安裝，跳過單元測試"
    echo "  安裝: pip install pytest"
fi

# ===== 總結 =====
echo ""
echo "=========================================="
echo "✓ 驗證流程完成"
echo "=========================================="
echo ""
echo "輸出檔案："
echo "  結構權重:  $OUTPUT_DIR/structure_weights/"
echo "  元數據:    $OUTPUT_DIR/structure_weights_metadata.json"
echo "  可視化:    $OUTPUT_DIR/structure_samples.html"
echo "             $OUTPUT_DIR/structure_samples.txt"
echo "  統計圖表:  $OUTPUT_DIR/plot_*.png"
echo "  驗證報告:  $OUTPUT_DIR/validation_report.json"
echo ""
echo "後續步驟："
echo "  1. 檢查 HTML: open $OUTPUT_DIR/structure_samples.html"
echo "  2. 檢查圖表: open $OUTPUT_DIR/plot_*.png"
echo "  3. 閱讀指南: cat $COT_DIR/VALIDATION_PIPELINE.md"
echo ""

# ===== 顯示統計摘要 =====
if [ -f "$OUTPUT_DIR/structure_weights_metadata.json" ]; then
    echo "統計摘要:"
    python3 -c "
import json
from pathlib import Path
json_path = Path('$OUTPUT_DIR/structure_weights_metadata.json')
if json_path.exists():
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f\"  總樣本數: {data.get('total_samples', '?')}\")
    struct_info = data.get('structure_tokens_per_sample', {})
    print(f\"  結構 token 平均: {struct_info.get('mean', '?'):.1f}\")
    dist = data.get('pattern_distribution', {})
    if dist:
        print('  Pattern 分佈:')
        for name, info in dist.items():
            hits = info.get('total_hits', 0)
            print(f'    - {name}: {hits} hits')
"
fi

echo ""
echo "完成！"
