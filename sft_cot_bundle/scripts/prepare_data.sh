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
echo ">>> [Step 0] 掃描 cot_dataset/ 所有 JSON 資料檔..."
FILES=""
for f in emotion.json self_awareness.json email_summary.json movie_intro.json noise.json; do
    if [[ -f "${COT_DIR}/${f}" ]]; then
        FILES="${FILES:+${FILES},}${f}"
    fi
done
for subdir in emotion self noise system_call movie_intro deep_dive; do
    if [[ -d "${COT_DIR}/${subdir}" ]]; then
        for f in "${COT_DIR}/${subdir}"/*.json; do
            if [[ -f "$f" ]]; then
                rel="${subdir}/$(basename "$f")"
                FILES="${FILES:+${FILES},}${rel}"
            fi
        done
    fi
done
echo "   檔案清單: ${FILES}"

# ── Step 1: Export 完整 7-bucket HF dataset ─────────────────────────────────
HF_FINAL="${DATASET_DIR}/stf_cot_hf_final"
echo ""
mkdir -p "${DATASET_DIR}" "${OUTPUT_DIR}"
echo ">>> [Step 1] Export 7-bucket HF dataset → ${HF_FINAL}"
python3 "${COT_DIR}/export_hf_dataset.py" \
    --src-dir "${COT_DIR}" \
    --files "${FILES}" \
    --out "${HF_FINAL}" \
    --duplicate-policy keep-last \
    --dedupe-by-content \
    --invalid-row-policy skip \
    --rewrite-id-prefix train_

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
