# Task 2 實現檢查清單：FCP + SFT-GO

## 🎬 前置準備

### 代碼檢查與驗證

**關鍵檢查項目**：
- [ ] 定位 `train_sft.py` 的 loss 計算部分
- [ ] **確認 EOS Token ID**：
  ```bash
  # 在 train_sft.py 或互動環境執行
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained("path_to_tokenizer")
  print(f"eos_token_id: {tok.eos_token_id}")
  print(f"eos_token: {tok.eos_token}")
  print(f"vocab_size: {tok.vocab_size}")
  # 記錄正確的 EOS_ID（不要假設為 2 或 32001）
  ```
- [ ] 找出 `MaterializedSftDataset` 類定義
- [ ] 找出 think 區的 start/end markers（`<think>` 和 `</think>`）
- [ ] **確認分隔符定義**（僅限以下 2 個）：
  - `</think>` — think 區結束
  - `<|im_end|>` — message 結束
  - ✅ 不包含：`<final>`, `</final>`, `<|im_start|>` (由正常訓練學習)
- [ ] 檢查 `build_structure_weights.py` 中分隔符位置是否已設為 w=1.0（讓 FCP 單獨處理）

### 數據準備
- [ ] ✅ 結構權重已生成 (`cot/reports/structure_weights/`)
- [ ] ✅ 元數據已準備 (`structure_weights_metadata.json`)
- [ ] ✅ 權重向量已驗證（mask-aware，僅包含 assistant 區域）

---

## 🔴 Phase 1：FCP (Format/EOS Penalty)

### Step 1.1：實現 EOS 檢測 (1 小時)
```python
# train_sft.py 中新增
def find_think_region(input_ids, tok):
    """找出 <think>...</think> 區間"""
    # 回傳 (start_idx, end_idx)
```

**任務**：
- [ ] 實現 find_think_region() 函數
- [ ] 處理邊界情況（嵌套、多個 think 區）
- [ ] 單元測試

### Step 1.2：實現 EOS Penalty 計算 (2 小時)
```python
def compute_eos_penalty(logits, input_ids, think_spans, 
                        lambda_eos=0.2, delta=0.01):
    """計算 think 區內的 EOS penalty"""
    # 回傳 scalar penalty
```

**任務**：
- [ ] 實現 compute_eos_penalty() 函數
- [ ] 驗證 penalty 計算正確性
- [ ] 測試邊界值（δ 值調整）

### Step 1.3：分隔符權重應用 (1.5 小時)
```python
def apply_separator_weights(loss_ce, separator_indices, 
                           weight=3.0, labels=None):
    """對分隔符位置應用固定權重"""
    # 回傳加權後的 loss
```

**任務**：
- [ ] 實現 apply_separator_weights() 函數
- [ ] 確保不與 SFT-GO 重疊
- [ ] 驗證分隔符位置識別

### Step 1.4：集成到訓練迴圈 (1.5 小時)
```python
# 在訓練迴圈中修改 loss 計算（完整公式）
loss_ce_weighted = criterion(logits_flat, labels_flat)  # 已應用 SFT-GO 結構權重
penalty_eos = compute_eos_penalty(...)                   # Σ max(0, p_eos - δ)²
loss_separators = apply_separator_weights(...)            # 分隔符額外權重損失

loss = loss_ce_weighted + 0.2 * penalty_eos + loss_separators
```

**任務**：
- [ ] 修改 loss 計算邏輯（三項相加）
- [ ] 檢查各項損失的量級（見 Loss 幅度指南）
- [ ] 確保 backward() 能正確計算梯度
- [ ] 測試無 NaN/Inf

### Step 1.5：監控日誌 (1 小時)
```python
# 每 N steps 打印
if step % 100 == 0:
    print(f"[FCP] EOS prob: {eos_prob:.4f}, Penalty: {loss_fcp:.4f}")
```

**任務**：
- [ ] 新增 FCP 監控日誌
- [ ] 追蹤 EOS probability 趨勢
- [ ] 設置 tensorboard logging

**小計：FCP = 7-8 小時**

---

## 🟡 Phase 2：SFT-GO (Structure Token Group Optimization)

### Step 2.1：加載結構權重 (1 小時)
```python
# MaterializedSftDataset 中新增
def load_structure_weights(self, sample_id):
    """從 .npz 檔加載預計算的權重"""
    # 回傳 weight 向量 [seq_len]
```

**任務**：
- [ ] 實現 load_structure_weights() 函數
- [ ] 處理缺失的權重檔（回退到 w=1.0）
- [ ] 驗證權重向量形狀

### Step 2.2：應用結構權重 (1.5 小時)
```python
def apply_structure_weights(loss, structure_weights, labels):
    """對結構 tokens 應用 w_struct = 2.5
    
    注意：分隔符位置（</think>, <|im_end|>）應在 structure_weights 中設為 1.0
         讓 FCP 的 apply_separator_weights() 單獨處理它們
    """
    weighted_loss = loss * structure_weights
    return weighted_loss[labels != -100].mean()
```

**任務**：
- [ ] 實現 apply_structure_weights() 函數（w_struct = 2.5）
- [ ] 確保只對有效位置應用（labels != -100）
- [ ] **驗證分隔符位置是否被設為 1.0**（不要與 FCP 重疊）
- [ ] 驗證權重乘法的廣播正確性

### Step 2.3：集成到訓練迴圈 (1.5 小時)
```python
# SFT-GO 應用於 loss 計算的第一步（在 FCP 之前）
loss_ce = criterion(logits_flat, labels_flat)
loss_ce_weighted = apply_structure_weights(loss_ce,    # 應用 w_struct = 2.5
                                            structure_weights, 
                                            labels)
# 然後在主 loss 計算中使用 loss_ce_weighted（見 Step 1.4）
```

**任務**：
- [ ] 修改 loss 計算順序（SFT-GO 先應用）
- [ ] **確保與 FCP 不衝突**：
  - SFT-GO 應用於全部 tokens（除了分隔符設為 1.0）
  - FCP 在分隔符位置再額外加權
- [ ] 檢查損失量級（見 Loss 幅度指南）
- [ ] 測試梯度計算（無 NaN）

### Step 2.4：監控日誌 (1 小時)
```python
if step % 100 == 0:
    struct_loss = (loss_ce * structure_weights).mean()
    print(f"[SFT-GO] Struct loss: {struct_loss:.4f}")
```

**任務**：
- [ ] 新增 SFT-GO 監控日誌
- [ ] 追蹤結構 loss 與總 loss 的比例
- [ ] 設置 tensorboard logging

**小計：SFT-GO = 5-6 小時**

---

## 🧪 Phase 3：驗證與測試

### Step 3.1：單元測試 (2 小時)
```python
# test_fcp_sftgo.py
def test_find_think_region():
def test_compute_eos_penalty():
def test_apply_separator_weights():
def test_apply_structure_weights():
```

**任務**：
- [ ] 編寫 4 個核心函數的單元測試
- [ ] 測試邊界情況
- [ ] 驗證數值結果

### Step 3.2：小規模訓練 (4 小時)
```bash
# 100-500 steps on 1 GPU
# 監控：loss 曲線、EOS prob、format accuracy
```

**任務**：
- [ ] 準備最小化訓練集（10 個樣本）
- [ ] 運行 500 steps
- [ ] 檢查 loss 是否平穩下降
- [ ] 驗證無 NaN/Inf/CUDA 錯誤

### Step 3.3：推理驗證 (1.5 小時)
```python
# 生成樣本，檢查格式正確性
# 指標：
# - Think 區是否在正確位置結束
# - Step 1/2/3/... 是否完整
# - 分隔符是否正確
```

**任務**：
- [ ] 實現推理驗證腳本
- [ ] 手動檢查 5-10 個生成樣本
- [ ] 記錄格式正確率

**小計：驗證 = 7.5 小時**

---

## 📏 Loss 幅度與監控指南

實現前需要理解三項損失的預期範圍：

| 損失項 | 典型範圍 | 預期趨勢 | 異常信號 |
|--------|--------|--------|---------|
| Loss_CE_weighted | 2.5-3.5 | ↓ 逐步下降 | >5.0 或無下降 |
| penalty_EOS | 0.01-0.1 | ↓ 逐步下降 | >0.5 表示 think 區未學會避免 EOS |
| Loss_separator | 0.1-0.5 | ↓ 逐步下降 | >1.0 表示分隔符識別有誤 |
| Loss_total | 2.6-4.1 | ↓ 平穩下降 | >10 或出現 NaN（實現錯誤） |

**實現後的監控**：
- 前 100 steps：確認各項損失都在預期範圍
- 500 steps 時：penalty_EOS 應下降至 < 0.05
- 格式評估：手動檢查 5-10 個生成樣本的 think 區是否完整

**不應出現的情況**：
- ❌ Loss_total > 10（檢查係數或公式有誤）
- ❌ penalty_EOS 無下降趨勢（λ_fcp 是否過小）
- ❌ NaN 或 Inf（檢查 max() 函數或 softmax 溢出）

---

## 📊 總時間估計

| 階段 | 時間 | 狀態 |
|------|------|------|
| Phase 1 (FCP) | 7-8 小時 | ⏳ 待開始 |
| Phase 2 (SFT-GO) | 5-6 小時 | ⏳ 待開始 |
| Phase 3 (驗證) | 7.5 小時 | ⏳ 待開始 |
| **總計** | **20-21 小時** | ⏳ **~2.5 工作天** |

---

## 🔑 關鍵檢查點

### 前置檢查
- [ ] **EOS Token ID 確認**：執行 Step 1.1 中的驗證代碼，記錄實際值
- [ ] **分隔符列表確認**：僅限 `</think>` 和 `<|im_end|>` 兩個
- [ ] **權重值確認**：w_struct = 2.5, w_sep = 3.0（無衝突）
- [ ] **分隔符位置檢查**：`build_structure_weights.py` 中分隔符是否設為 w=1.0

### FCP 檢查點
- [ ] EOS penalty 能正確計算（數值 > 0）
- [ ] 在 think 區內有作用（其他區域無 penalty）
- [ ] Separator 權重生效（loss ↓）
- [ ] 損失量級正常（penalty_EOS ~ 0.01-0.1）
- [ ] 訓練曲線平穩（無抖動）

### SFT-GO 檢查點
- [ ] 權重向量形狀正確 [seq_len]
- [ ] 權重乘法無廣播錯誤
- [ ] 只對有效位置應用（labels != -100）
- [ ] **分隔符位置未被 w_struct 加權**（應為 1.0）
- [ ] 結構 loss 趨勢向下
- [ ] 損失量級正常（loss_ce_weighted ~ 2.5-3.5）

### 集成檢查點
- [ ] FCP + SFT-GO 不相互干擾（分隔符由 FCP 單獨處理）
- [ ] Loss 計算無重複計算
- [ ] 三項損失相加：loss = loss_ce_weighted + 0.2*penalty_eos + loss_sep
- [ ] Loss_total 量級 ~ 2.6-4.1（見 Loss 幅度指南）
- [ ] Backward pass 無梯度異常（無 NaN）
- [ ] Checkpoint save/load 正常

---

## 🚀 快速開始

```bash
# 1. 查看實現計畫
cat /Users/hungwei/Desktop/Proj/Mamba3-XR/cot/TASK2_FOCUSED_PLAN.md

# 2. 檢查 train_sft.py 位置
find /Users/hungwei/Desktop/Proj -name "train_sft.py"

# 3. 檢查結構權重是否已生成
ls -lh /Users/hungwei/Desktop/Proj/Mamba3-XR/cot/reports/structure_weights/ | head -5

# 4. 準備開始
# 下一步：確認開始 Phase 1 (FCP)
```

---

## 📝 注意事項

### 避免的陷阱
- ❌ **不要在 FCP 和 SFT-GO 中都對同一位置加權** — 分隔符 (`</think>`, `<|im_end|>`) 必須在 structure_weights 中設為 1.0，讓 FCP 單獨處理
- ❌ 不要忘記檢查 labels != -100（masked positions，應該跳過）
- ❌ 不要在 loss 上使用 `.item()` 過於頻繁（降低效率，每 100 steps 記一次即可）
- ❌ **不要假設 EOS Token ID**（必須在實現前驗證正確的 ID，見前置準備）
- ❌ 不要混淆分隔符定義（僅 `</think>` 和 `<|im_end|>`，不含其他）

### 性能優化
- ✅ 預計算 separator 位置（不要每次計算）
- ✅ 使用 vectorized 操作（避免 for 迴圈）
- ✅ 批量加載權重（不要逐個樣本加載）

---

**準備好開始實現 Phase 1 (FCP) 了嗎？** 
👉 確認後我就開始編寫代碼。
