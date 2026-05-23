#!/bin/bash
set -euo pipefail

###############################################################################
# prepare_data.sh
#
# 一鍵完成 SFT-CoT 資料準備：
#   1. 掃描 cot_dataset/ 所有 JSON，export 完整 7-bucket HF dataset
#   2. 跑 token 長度統計 → 建議最佳 SEQ_LEN
#   3. 產生 tokenized .bin
#   4. 把 HF dataset + tokenizer + .bin 複製到 dataset/
#   5. 抽樣驗證 mask (x → y)
#
# 用法：
#   cd <project_root>
#   bash sft_cot_bundle/scripts/prepare_data.sh
###############################################################################

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
COT_DIR="${PROJECT_ROOT}/cot_dataset"
BUNDLE_DIR="${PROJECT_ROOT}/sft_cot_bundle"
SCRIPTS_DIR="${BUNDLE_DIR}/scripts"
DATASET_DIR="${BUNDLE_DIR}/dataset"
OUTPUT_DIR="${BUNDLE_DIR}/output"

echo "=========================================="
echo " SFT-CoT 資料準備腳本"
echo " PROJECT_ROOT = ${PROJECT_ROOT}"
echo "=========================================="

# ── Step 0: 自動掃描所有 JSON 檔案 ─────────────────────────────────────────
echo ""
echo ">>> [Step 0] 動態掃描 cot_dataset/ 所有 JSON 資料檔..."
FILES=""

# 1. 掃描根目錄所有 .json 檔案（排除特定檔案）
echo "   掃描根目錄 JSON 檔案..."
for f in "${COT_DIR}"/*.json; do
    if [[ -f "$f" ]]; then
        basename_f="$(basename "$f")"
        # 排除 tokenizer 相關檔案
        if [[ "$basename_f" != "tokenizer.json" && "$basename_f" != "tokenizer_config.json" ]]; then
            FILES="${FILES:+${FILES},}${basename_f}"
            echo "     + ${basename_f}"
        fi
    fi
done

# 2. 動態掃描所有子目錄（使用通配符模式）
echo "   掃描子目錄 JSON 檔案..."
# 定義要掃描的目錄模式
DIR_PATTERNS=(
    "emotion*"
    "self*"
    "noise*"
    "system_call*"
    "movie*"
    "deep*"
    "mail*"
)

for pattern in "${DIR_PATTERNS[@]}"; do
    for subdir in "${COT_DIR}"/${pattern}; do
        if [[ -d "$subdir" ]]; then
            subdir_name="$(basename "$subdir")"
            echo "     掃描 ${subdir_name}/ ..."
            for f in "${subdir}"/*.json; do
                if [[ -f "$f" ]]; then
                    rel="${subdir_name}/$(basename "$f")"
                    FILES="${FILES:+${FILES},}${rel}"
                    echo "       + ${rel}"
                fi
            done
        fi
    done
done

# 3. 掃描其他可能的子目錄（不在上述模式中的）
echo "   掃描其他子目錄..."
for subdir in "${COT_DIR}"/*/; do
    if [[ -d "$subdir" ]]; then
        subdir_name="$(basename "$subdir")"
        # 跳過特殊目錄
        if [[ "$subdir_name" == "__pycache__" || "$subdir_name" == ".git" || "$subdir_name" == ".bob" || "$subdir_name" == "scripts" || "$subdir_name" == "metadata_sft_tiny_llm" || "$subdir_name" == "metal" || "$subdir_name" == "cot" || "$subdir_name" =~ ^stf_cot_hf ]]; then
            continue
        fi
        # 檢查是否已經被上面的模式掃描過
        already_scanned=false
        for pattern in "${DIR_PATTERNS[@]}"; do
            if [[ "$subdir_name" == ${pattern} ]]; then
                already_scanned=true
                break
            fi
        done
        if [[ "$already_scanned" == false ]]; then
            echo "     掃描 ${subdir_name}/ ..."
            for f in "${subdir}"/*.json; do
                if [[ -f "$f" ]]; then
                    rel="${subdir_name}/$(basename "$f")"
                    FILES="${FILES:+${FILES},}${rel}"
                    echo "       + ${rel}"
                fi
            done
        fi
    fi
done

if [[ -z "$FILES" ]]; then
    echo "   ❌ 錯誤：未找到任何 JSON 檔案！"
    exit 1
fi

echo ""
echo "   ✅ 找到檔案清單: ${FILES}"

# ── Step 1: Export 完整 7-bucket HF dataset ─────────────────────────────────
HF_FINAL="${DATASET_DIR}/stf_cot_hf_final"
echo ""
mkdir -p "${DATASET_DIR}" "${OUTPUT_DIR}"
echo ">>> [Step 1] Export 7-bucket HF dataset → ${HF_FINAL}"
python3 "${COT_DIR}/export_hf_dataset.py" \
    --src-dir "${COT_DIR}" \
    --files "${FILES}" \
    --out "${HF_FINAL}" \
    --duplicate-policy keep-all \
    --dedupe-by-content \
    --invalid-row-policy skip \
    --rewrite-id-prefix train_ \
    --shuffle --seed 42

echo ""
echo "   ✅ HF dataset 匯出完成"

# ── Step 2: Token 長度統計 ──────────────────────────────────────────────────
echo ""
echo ">>> [Step 2] Token 長度統計..."
TOKENIZER_DIR="${COT_DIR}"
mkdir -p "${OUTPUT_DIR}"
python3 "${SCRIPTS_DIR}/analyze_token_lengths.py" \
    --hf-dir "${HF_FINAL}" \
    --tokenizer-dir "${TOKENIZER_DIR}" \
    --out "${OUTPUT_DIR}/stf_cot_token_len_report.json"

# ── Step 3: 產生 tokenized .bin ─────────────────────────────────────────────
echo ""
echo ">>> [Step 3] 產生 tokenized .bin..."
BIN_PATH="${DATASET_DIR}/stf_cot_train.bin"
python3 -c "
import json, sys, numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer

hf = load_from_disk('${HF_FINAL}')
tok = AutoTokenizer.from_pretrained('${TOKENIZER_DIR}', local_files_only=True)
tok.model_max_length = 1_000_000

all_ids = []
for i in range(len(hf)):
    text = hf[i]['text']
    ids = tok.encode(text, add_special_tokens=False)
    for wid in ids:
        if not (0 <= wid < 65535):
            print(f'WARNING: token id {wid} out of uint16 range at row {i}', file=sys.stderr)
    all_ids.extend(ids)

arr = np.array(all_ids, dtype=np.uint16)
arr.tofile('${BIN_PATH}')
print(f'   Written {len(all_ids):,} tokens -> ${BIN_PATH}')
"

# ── Step 4: 整理輸出 ─────────────────────────────────────────────────────────
echo ""
echo ">>> [Step 4] 整理輸出到 sft_cot_bundle/dataset/..."

if [[ -d "${DATASET_DIR}/stf_cot_hf" ]]; then
    rm -rf "${DATASET_DIR}/stf_cot_hf"
fi
mv "${HF_FINAL}" "${DATASET_DIR}/stf_cot_hf"
echo "   ✅ HF dataset → sft_cot_bundle/dataset/stf_cot_hf"

mkdir -p "${DATASET_DIR}/tokenizer"
cp -f "${COT_DIR}/tokenizer.json" "${DATASET_DIR}/tokenizer/tokenizer.json"
cp -f "${COT_DIR}/tokenizer_config.json" "${DATASET_DIR}/tokenizer/tokenizer_config.json"
echo "   ✅ Tokenizer → sft_cot_bundle/dataset/tokenizer"
echo "   ✅ .bin → sft_cot_bundle/dataset/stf_cot_train.bin"

# ── Step 5: 抽樣驗證 mask ──────────────────────────────────────────────────
echo ""
echo ">>> [Step 5] 抽樣驗證 mask (x → y)..."
python3 "${SCRIPTS_DIR}/verify_mask_xy.py" \
    --hf-dir "${DATASET_DIR}/stf_cot_hf" \
    --tokenizer-dir "${DATASET_DIR}/tokenizer" \
    --seq-len 1024 \
    --num-samples 5 \
    --out "${OUTPUT_DIR}/mask_xy_check.txt"

echo ""
echo "=========================================="
echo " 全部完成！"
echo ""
echo " 資料位置："
echo "   HF dataset:  sft_cot_bundle/dataset/stf_cot_hf"
echo "   Tokenizer:   sft_cot_bundle/dataset/tokenizer"
echo "   Token bin:   sft_cot_bundle/dataset/stf_cot_train.bin"
echo ""
echo " 報告："
echo "   Token 長度:  sft_cot_bundle/output/stf_cot_token_len_report.json"
echo "   Mask 抽查:   sft_cot_bundle/output/mask_xy_check.txt"
echo ""
echo " 訓練："
echo "   python3 sft_cot_bundle/scripts/train_sft_cot.py"
echo "=========================================="
