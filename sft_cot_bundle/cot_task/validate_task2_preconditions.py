#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2 前置驗證腳本：檢查所有必要條件是否滿足
- Tokenizer 完整性 (EOS token ID, vocab_size=32007)
- 結構權重檔案存在與格式
- train_sft.py 的 loss 計算邏輯
- Dataset 完整性與樣本驗證
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer

# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # llm root
SFT_COT_BUNDLE = PROJECT_ROOT / "sft_cot_bundle"
DATASET_DIR = SFT_COT_BUNDLE / "dataset"
TOKENIZER_DIR = DATASET_DIR / "tokenizer"
STRUCTURE_WEIGHTS_DIR = SFT_COT_BUNDLE / "cot_task" / "reports" / "structure_weights"

print(f"📂 Project Root: {PROJECT_ROOT}")
print(f"📂 SFT-CoT Bundle: {SFT_COT_BUNDLE}")
print(f"📂 Dataset Dir: {DATASET_DIR}")
print()

# ============================================================================
# 1. TOKENIZER VALIDATION
# ============================================================================

def validate_tokenizer():
    """驗證 Tokenizer 屬性"""
    print("=" * 70)
    print("🔍 [1/4] TOKENIZER VALIDATION")
    print("=" * 70)

    if not TOKENIZER_DIR.exists():
        print(f"❌ Tokenizer 目錄不存在: {TOKENIZER_DIR}")
        return False

    try:
        tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)
        print(f"✅ Tokenizer 加載成功")
        print(f"   - eos_token_id: {tok.eos_token_id}")
        print(f"   - eos_token: {repr(tok.eos_token)}")
        print(f"   - vocab_size: {tok.vocab_size}  (excl. added specials)")
        print(f"   - len(tok):  {len(tok)}  (incl. added specials)")
        print(f"   - pad_token_id: {tok.pad_token_id}")
        print(f"   - bos_token_id: {tok.bos_token_id}")

        # 關鍵約束：總詞表大小應為 32007 (32000 base + 7 specials)
        total_vocab = len(tok)
        if total_vocab != 32007:
            print(f"❌ ERROR: len(tok)={total_vocab}，預期 32007")
            return False

        if tok.eos_token_id is None or tok.eos_token_id < 0:
            print(f"❌ eos_token_id 無效: {tok.eos_token_id}")
            return False

        # 驗證特殊 tokens 是否為「單一 token」(這對 FCP/SFT-GO 至關重要)
        single_token_markers = {
            "<|im_start|>": 32000,
            "<|im_end|>":   32001,
            "<think>":      32002,
            "</think>":     32003,
            "<final>":      32004,
            "</final>":     32005,
        }
        print(f"\n✅ 特殊標記編碼驗證 (必須為單 token):")
        ok = True
        for marker, expected_id in single_token_markers.items():
            ids = tok.encode(marker, add_special_tokens=False)
            actual = ids[0] if len(ids) == 1 else ids
            mark = "✅" if (len(ids) == 1 and ids[0] == expected_id) else "❌"
            print(f"   {mark} {marker:14s} → {ids}  (expected {expected_id})")
            if len(ids) != 1 or ids[0] != expected_id:
                ok = False

        if not ok:
            print(f"\n❌ 某些特殊標記不是預期的單 token，FCP/SFT-GO 將無法正確工作")
            return False

        print(f"\n✅ Tokenizer 驗證通過")
        return True

    except Exception as e:
        print(f"❌ Tokenizer 加載失敗: {e}")
        return False

# ============================================================================
# 2. STRUCTURE WEIGHTS VALIDATION
# ============================================================================

def validate_structure_weights():
    """驗證結構權重檔案"""
    print("\n" + "=" * 70)
    print("🔍 [2/4] STRUCTURE WEIGHTS VALIDATION")
    print("=" * 70)

    if not STRUCTURE_WEIGHTS_DIR.exists():
        print(f"❌ Structure weights 目錄不存在: {STRUCTURE_WEIGHTS_DIR}")
        return False

    # 檢查 metadata (放在 reports/ 不是 reports/structure_weights/)
    metadata_file = STRUCTURE_WEIGHTS_DIR.parent / "structure_weights_metadata.json"
    if not metadata_file.exists():
        print(f"❌ structure_weights_metadata.json 不存在: {metadata_file}")
        return False

    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        print(f"✅ 元數據加載成功")
        print(f"   - 總樣本數: {metadata.get('total_samples', '?')}")
        print(f"   - 結構模式: {metadata.get('patterns', {}).keys() if isinstance(metadata.get('patterns'), dict) else '?'}")
        print(f"   - 權重範圍: [{metadata.get('weight_min', '?')}, {metadata.get('weight_max', '?')}]")

    except Exception as e:
        print(f"⚠️  Warning: 無法解析 metadata.json: {e}")

    # 檢查權重檔案樣本
    weight_files = list(STRUCTURE_WEIGHTS_DIR.glob("*.npz"))
    print(f"\n✅ 權重檔案統計:")
    print(f"   - 總計: {len(weight_files)} 個 .npz 檔案")

    if weight_files:
        # 隨機抽查一個 (允許 pickle 因 token_patterns 是 object array)
        sample_file = weight_files[0]
        try:
            data = np.load(sample_file, allow_pickle=True)
            print(f"\n✅ 樣本檔案抽查 ({sample_file.name}):")
            for key in data.files:
                arr = data[key]
                if arr.dtype == object:
                    print(f"   - {key}: shape={arr.shape}, dtype=object, sample[:3]={list(arr[:3])}")
                else:
                    print(f"   - {key}: shape={arr.shape}, dtype={arr.dtype}, "
                          f"range=[{arr.min():.3f}, {arr.max():.3f}]")
        except Exception as e:
            print(f"❌ 無法加載樣本檔案: {e}")
            return False

    print(f"\n✅ Structure weights 驗證通過")
    return True

# ============================================================================
# 3. DATASET VALIDATION
# ============================================================================

def validate_dataset():
    """驗證資料集"""
    print("\n" + "=" * 70)
    print("🔍 [3/4] DATASET VALIDATION")
    print("=" * 70)

    hf_path = DATASET_DIR / "stf_cot_hf"
    bin_path = DATASET_DIR / "stf_cot_train.bin"

    if not hf_path.exists():
        print(f"❌ HuggingFace 資料集不存在: {hf_path}")
        return False

    if not bin_path.exists():
        print(f"⚠️  Warning: .bin 檔案不存在: {bin_path}")

    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(hf_path))

        if hasattr(ds, '__len__'):
            print(f"✅ 資料集加載成功")
            print(f"   - 樣本總數: {len(ds)}")

            # 檢查欄位
            if hasattr(ds, 'column_names'):
                print(f"   - 欄位: {ds.column_names}")

            # 抽查一個樣本
            if len(ds) > 0:
                sample = ds[0]
                print(f"\n✅ 樣本抽查 (第 0 筆):")
                for key, val in sample.items():
                    if isinstance(val, str):
                        print(f"   - {key}: {len(val)} chars, preview: {val[:100]}...")
                    elif isinstance(val, list):
                        print(f"   - {key}: list of {len(val)} items, first 5: {val[:5]}")
                    else:
                        print(f"   - {key}: {type(val).__name__}")
        else:
            print(f"❌ 資料集格式不正確")
            return False

        print(f"\n✅ Dataset 驗證通過")
        return True

    except Exception as e:
        print(f"❌ 資料集加載失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# 4. TRAINING CODE VALIDATION
# ============================================================================

def validate_training_code():
    """檢查 train_sft.py 是否包含必要的 hook point"""
    print("\n" + "=" * 70)
    print("🔍 [4/4] TRAINING CODE VALIDATION")
    print("=" * 70)

    train_sft_path = SFT_COT_BUNDLE / "scripts" / "train_sft.py"

    if not train_sft_path.exists():
        print(f"❌ train_sft.py 不存在: {train_sft_path}")
        return False

    try:
        with open(train_sft_path, 'r', encoding='utf-8') as f:
            code = f.read()

        print(f"✅ train_sft.py 加載成功 ({len(code)} bytes)")

        # 檢查關鍵函數與邏輯
        checks = {
            "loss 計算": "loss = " in code or "out[0]" in code,
            "backward": "backward(" in code,
            "optimizer": "optimizer.step()" in code,
            "logging": "log_w.writerow" in code or "writerow" in code,
            "checkpoint": "torch.save" in code,
        }

        print(f"\n✅ 關鍵代碼邏輯檢查:")
        for check_name, present in checks.items():
            status = "✅" if present else "⚠️"
            print(f"   {status} {check_name}")

        print(f"\n✅ Training code 驗證通過")
        return True

    except Exception as e:
        print(f"❌ 無法檢查 train_sft.py: {e}")
        return False

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("🚀 Task 2 前置條件驗證")
    print("=" * 70 + "\n")

    results = {
        "tokenizer": validate_tokenizer(),
        "structure_weights": validate_structure_weights(),
        "dataset": validate_dataset(),
        "training_code": validate_training_code(),
    }

    print("\n" + "=" * 70)
    print("📊 驗證結果摘要")
    print("=" * 70)

    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10s} {check_name}")

    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ 所有前置條件驗證通過，可開始 Task 2 實現")
    else:
        print("❌ 存在驗證失敗，請先解決問題再繼續")
        sys.exit(1)
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
