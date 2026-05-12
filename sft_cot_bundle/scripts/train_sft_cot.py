# -*- coding: utf-8 -*-
"""
SFT-CoT（stf）：資料來自 `dataset/stf.json` 經 `scripts/stf_cot_to_bin.py` 物化為
`dataset/stf_cot_hf` + `dataset/stf_cot_train.bin`；詞表請用 Tiny-LLM + 7 special：

1. 使用既有 tokenizer：`dataset/tokenizer`（需含 32007 與 7 個 special）
2. `python3 scripts/stf_cot_to_bin.py`（預設依 category 插入三種 system：對話／任務／總結；可 `--no-system`）
3. 抽查：`python3 scripts/spot_check_stf_cot.py`；mask smoke：`python3 scripts/verify_stf_cot_mask.py`
4. SFT：`python3 scripts/train_sft_cot.py`（須在 repo 根目錄，`PYTHONPATH=scripts` 或由 nohup 腳本啟動）

與 GAIR/LIMA SFT 分檔；訓練邏輯見 `train_sft.train_sft`（hf_text + mask.md）。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from train_sft import train_sft


def _find_latest_sft_checkpoint() -> str:
    """自動找 /ssd1/hungwei/checkpoint_sft_s*.pt 中 step 最大的檔案。"""
    override = os.environ.get("SFT_COT_CHECKPOINT_LOAD")
    if override:
        return override
    best_step, latest = -1, None
    for p in Path("/ssd1/hungwei").glob("checkpoint_sft_s*.pt"):
        if "model_only" in p.name:
            continue
        m = re.search(r"_s(\d+)\.pt$", p.name)
        if m and int(m.group(1)) > best_step:
            best_step = int(m.group(1))
            latest = p
    ckpt = str(latest) if latest else "dataset/pre_train_base_model.pt"
    print(f"ℹ️  COT checkpoint load = {ckpt}", flush=True)
    return ckpt


if __name__ == "__main__":
    train_sft(
        SFT_DATA_SOURCE="hf_text",
        LIMA_HF="dataset/stf_cot_hf",
        DATA_PATH="dataset/stf_cot_train.bin",
        CHECKPOINT_LOAD=_find_latest_sft_checkpoint(),
        SFT_INIT_WEIGHTS_ONLY=True,
        CHECKPOINT_SAVE="/ssd1/hungwei/checkpoint_sft_cot.pt",
        LOG_FILE="output/train_sft_cot_log.csv",
        VAL_LOG_FILE="output/val_sft_cot_log.csv",
        TOKENIZER_DIR="dataset/tokenizer",
        VOCAB_SIZE=32007,
        SEQ_LEN=1024,
        BATCH_SIZE=8,
        GRADIENT_ACCUMULATION_STEPS=4,
        LR=1e-5,
        EPOCHS=100,
        SAVE_EVERY_STEPS=100,
        SFT_TEST_PROMPT="<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n",
        SFT_TEST_EVERY_STEPS=200,
        SFT_TEST_LOG_FILE="output/sft_cot_test_gen.log",
        SFT_TEST_TEMPERATURE=0.2,
        SFT_TEST_TOP_P=0.85,
        SFT_TEST_MAX_NEW=128,
        VAL_EVERY_STEPS=15,
    )
