# CoT 推理品質下降 — 問題診斷與修復報告

## 🎯 問題摘要

`cot_format_fsm.py` 和 `cot_middleware.py` 加入後，生成品質明顯下降。根本原因已識別並修復。

---

## 🔍 診斷過程

### 第一層：組件單元測試 ✅
```bash
python diagnose_cot.py --test all
```
**結果：** 所有測試通過  
**含義：** FSM 和中間件邏輯本身正確

---

### 第二層：根本原因分析

#### 發現的問題
在 `build_format_guard()` 中，CoT token IDs 無法正確識別：

```python
# 預期：
</think>  → ID 32003 ✓
</final>  → ID 32005 ✓

# 實際（修復前）：
</think>  → ID 829 ❌  (fallback to </)
</final>  → ID 829 ❌  (fallback to </)
```

#### 根本原因
分詞器報告的 `vocab_size=32000`，但實際後端詞彙表大小是 32007：

```
Tokenizer vocab_size:    32000
Backend actual vocab:    32007
                         ^^^^^^ Mismatch!

CoT tokens:
  32002 (<think>)   → Outside range (> 31999)
  32003 (</think>)  → Outside range (> 31999)
  32004 (<final>)   → Outside range (> 31999)
  32005 (</final>)  → Outside range (> 31999)
```

**為什麼這造成問題：**

1. `build_format_guard()` 中的檢查：
   ```python
   if tid is not None and 0 <= tid < vocab_size:  # 32000
       close_map[mode] = tid
   ```

2. 當 `tid=32003` 時，條件失敗（32003 ≥ 32000）
3. 代碼退回到 `</` (ID 829)
4. Close bias 現在針對錯誤的 token

**結果：**
- 模型被鼓勵生成 `</` 而不是 `</think>` 或 `</final>`
- FSM 永遠看不到正確的關閉標籤
- 推理模式轉換失敗
- 生成品質下降

---

## ✅ 修復方案

### 修改：`mamba3_mlx/cot_format_fsm.py`

#### 問題 1：`build_format_guard()` 中的 vocab_size 檢查

**修復：** 使用實際的後端詞彙大小
```python
def build_format_guard(tokenizer, *, vocab_size: int, cfg):
    # 獲取實際的詞彙大小
    actual_vocab_size = vocab_size
    if hasattr(tokenizer, "backend_tokenizer"):
        try:
            backend_vocab = tokenizer.backend_tokenizer.get_vocab()
            if backend_vocab:
                actual_vocab_size = max(actual_vocab_size, 
                                       max(backend_vocab.values()) + 1)
        except Exception:
            pass
    
    # 使用 actual_vocab_size 進行所有檢查
    # 而不是原來的 vocab_size
```

#### 問題 2：`merge_format_guard_stop_ids()` 中同樣的問題

**修復：** 應用相同的 vocab_size 調整邏輯

---

## 📊 驗證結果

### 修復前
```
close_bias(+4.0→+16.0)=[think→829 `</`, final→829 `</`]
                              ^^^^              ^^^^
                         錯誤的 token（後備）
```

### 修復後
```
close_bias(+4.0→+16.0)=[
  think→32003 `</think>`,
  between→32004 `<final>`,
  final→32005 `</final>`
]
             ^^^^^^^^^^^^
        正確的 CoT token
```

### 診斷測試結果
```
✓ Format Guard: 正確識別 3 個禁止 token + 3 個關閉目標
✓ Stream Splitter: 所有模式轉換測試通過
✓ Middleware: 預算和狀態跟蹤正確
✓ All tests passed
```

---

## 🔧 實現細節

### 受影響的函數
1. `build_format_guard()` - 構建格式保護（主要修復點）
2. `merge_format_guard_stop_ids()` - 合併停止 ID
3. `diagnose_cot.py` - 診斷腳本（調整以使用正確的 vocab_size）

### 修改行數
- `cot_format_fsm.py`: +20 lines
- `diagnose_cot.py`: +12 lines

### 向後兼容性
✅ **完全向後兼容** — 修復只影響正確 token 的識別，不改變 API

---

## 📈 預期改進

### 修復前的症狀
- ❌ 生成品質下降
- ❌ Close bias 應用於錯誤的 token
- ❌ FSM 模式轉換失敗

### 修復後的預期
- ✅ 正確應用 close_bias 到 `</think>` 和 `</final>`
- ✅ FSM 正確轉換模式
- ✅ 推理預算生效
- ✅ 推理 vs 最終答案正確分離

---

## 🧪 測試清單

- [x] 單元測試（`diagnose_cot.py --test all`）
- [x] 流分流器測試（所有標籤解析場景）
- [x] 中間件配置測試
- [x] Format guard token 識別

---

## 💾 提交信息

```
fix: CoT token ID resolution for extended vocabulary

PROBLEM: CoT closing tokens (32002-32005) were outside vocab_size
boundary and fell back to '</' (ID 829), preventing proper FSM
transitions and degrading generation quality.

FIX: Use actual backend vocabulary size instead of tokenizer.vocab_size
for token validation checks.

RESULT: ✓ All CoT tokens now correctly identified and biased.
```

---

## 📝 後續步驟

1. ✅ 修復已提交
2. ⏳ 等待模型測試驗證生成品質改進
3. 📊 對比修復前後的推理輸出

---

## 🔍 關鍵學習

**為什麼會出現這個 bug：**
- 分詞器報告的 `vocab_size` 與實際後端詞彙大小不一致
- 代碼假設 `tokenizer.vocab_size` 是準確的邊界
- 特殊 CoT token（作為 SFT 訓練的一部分加入）超出了假設邊界

**防止類似問題：**
1. 始終檢查 `backend_tokenizer.get_vocab()` 以獲得真實邊界
2. 對特殊 token 進行驗證測試
3. 有單元測試來驗證 token 標識符的正確性

---

## ✅ 修復驗證 Checklist

- [x] 診斷腳本通過
- [x] 所有 CoT token 正確識別
- [x] 無廣播大小錯誤
- [x] Guard 配置正確
- [x] 中間件初始化成功
- [x] 提交消息完整

**狀態：修復已完成，待推理測試驗證**
