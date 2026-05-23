# Task 2：聚焦實現計畫（FCP + SFT-GO）

## 🎯 核心目標

只實現 **2 個技術**，達成小模型訓練的「最小可行集合」：

1. **FCP** (Format/EOS Penalty) — 防止 think 區提早輸出 EOS
2. **SFT-GO** (Structure Token Group Optimization) — 讓模型學會整齊的 CoT 格式

---

## 📋 技術規格

### 1️⃣ FCP (Format/EOS Penalty)

**問題**：55M 參數小模型容易在 `<think>` 區內提早輸出 `<|im_end|>`，導致推理鏈斷裂。

**解決方案**（含完整 Loss 計算）：
```
Loss_total = Loss_CE_weighted + λ_fcp · penalty_EOS + w_sep · Loss_separator

where:
  Loss_CE_weighted = CE(logits, labels)              # Cross-entropy loss (standard)
  λ_fcp = 0.2                                        # EOS penalty coefficient
  δ = 0.01                                           # EOS probability threshold
  penalty_EOS = Σ_i∈[think_start, think_end] max(0, p_EOS[i] - δ)²  # 只在 think 區內
  w_sep = 3.0                                        # Separator weight multiplier
  Loss_separator = CE(logits[sep_indices], labels[sep_indices]) × w_sep
```

**關鍵點**：
1. `Loss_CE_weighted` 是標準交叉熵，已應用 SFT-GO 結構權重（見 §2）
2. `penalty_EOS` 只在 think 區內計算，其他區域不計算
3. `Loss_separator` 是分隔符位置的額外權重損失
4. 三項損失相加（無論 magnitude 大小）

**實現位置**：
- `train_sft.py` 中新增 `compute_fcp_penalty()` 和 `apply_separator_weights()`
- 在訓練迴圈中計算 EOS 機率
- 對分隔符位置加權重

**EOS Token ID 說明**：
在實現前，確認 tokenizer 中的 EOS Token：
```python
# 檢查：
tokenizer.eos_token_id        # 通常為 2 (Qwen) 或 32001 (GPT-style)
tokenizer.eos_token           # 顯示實際符號 (</s> 或其他)
# 在 train_sft.py 中必須使用正確的 EOS_ID，不要假設
```

**分隔符定義**（限定於這 2 個）：
- `</think>` — think 區結束標記
- `<|im_end|>` — message 結束標記

❌ 不包含：`<final>`, `</final>`, `<|im_start|>` (這些不是終止符，由正常訓練學習)

**核心邏輯**：
```python
# 在 think 區內檢測 EOS 機率
for pos in range(think_start, think_end):
    eos_prob = F.softmax(logits[pos], dim=-1)[eos_token_id]
    if eos_prob > delta:
        penalty += lambda_eos * (eos_prob - delta) ** 2
```

---

### 2️⃣ SFT-GO (Structure Token Group Optimization)

**問題**：結構 tokens（Step、\n、|、**、#） 容易被模型當成「低優先級」而忽略，導致輸出格式亂掉。

**解決方案**：
```
Loss_CE_weighted = Loss_CE * w_struct    # 在計算 Loss_total 之前

where:
  w_struct = 2.5                         # 結構 token 的權重乘數（避免與 w_sep=3.0 衝突）
  適用於: Step、豎線|、分隔行、粗體、標題、代碼塊（R1-R6）
  不適用於: 分隔符（</think>, <|im_end|>） — 已由 FCP 單獨處理
```

**與 FCP 的權重邊界**：
- FCP 分隔符權重 `w_sep = 3.0` — 處理 `</think>` 和 `<|im_end|>`
- SFT-GO 結構權重 `w_struct = 2.5` — 處理 R1-R6 模式（不重疊）
- 在 `build_structure_weights.py` 生成的權重向量中，分隔符位置應該設為 1.0（由 FCP 單獨處理）

**實現位置**：
- 已在 `build_structure_weights.py` 實現
- `train_sft.py` 中應用權重
- Loss 計算時乘入權重

**與 FCP 的邊界**：
```
FCP 權重: 分隔符（</think>, <|im_end|>, <|im_start|>） = 3.0
SFT-GO 權重: 結構 tokens (Step, |, -, **, #, ```) = 2.5~3.0

關鍵：不重疊！分隔符由 FCP 單獨處理，
      SFT-GO 只處理內容結構。
```

---

## 🗂️ 實現步驟

### Phase 1：FCP 實現 (1-2 天)

#### 1.1 識別分隔符位置
```python
# 在 train_sft.py 中新增函數
def find_separator_tokens(input_ids, tok):
    """找出所有分隔符位置：</think>, <|im_end|>, <|im_start|>"""
    separators = ["</think>", "<|im_end|>", "<|im_start|>"]
    separator_ids = [tok.encode(s, add_special_tokens=False) for s in separators]
    # 回傳 separator 在 input_ids 中的位置列表
```

#### 1.2 計算 EOS Penalty
```python
def compute_fcp_penalty(logits, input_ids, think_start, think_end, 
                        lambda_eos=0.2, delta=0.01, eos_id=32003):
    """
    在 think 區內，對 EOS 機率 > δ 的位置加懲罰
    """
    penalty = 0.0
    for pos in range(think_start, think_end):
        eos_prob = torch.softmax(logits[pos], dim=-1)[eos_id]
        if eos_prob > delta:
            penalty += lambda_eos * (eos_prob - delta) ** 2
    return penalty
```

#### 1.3 集成到訓練迴圈
```python
# 在 train_sft.py 的 loss 計算部分
loss_ce = criterion(logits.view(-1, vocab_size), labels.view(-1))
penalty_fcp = compute_fcp_penalty(logits, input_ids, think_start, think_end)
loss_separator = apply_separator_weights(loss_ce, separator_indices, w=3.0)

loss_total = loss_ce + penalty_fcp + loss_separator
```

---

### Phase 2：SFT-GO 集成 (1 天)

#### 2.1 應用結構權重
```python
# 在 train_sft.py 中
def apply_structure_weights(loss, structure_weights, labels):
    """
    對結構 tokens 應用 w_struct ≈ 2.5~3.0
    使用已在 build_structure_weights.py 生成的權重向量
    """
    # loss: [batch_size, seq_len]
    # structure_weights: [seq_len]
    # 只對有效 tokens 應用（labels != -100）
    
    weighted_loss = loss * structure_weights
    return weighted_loss[labels != -100].mean()
```

#### 2.2 加載預生成的權重
```python
# 在 dataset 初始化時
def load_structure_weights(weight_dir, sample_id):
    """從 build_structure_weights.py 生成的 .npz 檔載入"""
    weight_file = f"{weight_dir}/structure_weights/{sample_id}.npz"
    data = np.load(weight_file)
    return data['weight']  # shape: [seq_len]
```

---

## ⚖️ Loss 幅度與監控指南

三項損失的典型值範圍（供訓練監控）：

```
Loss_CE_weighted:     ~2.5 - 3.5   (使用結構權重後的交叉熵)
penalty_EOS:          ~0.01 - 0.1  (EOS 機率懲罰，逐步下降)
Loss_separator:       ~0.1 - 0.5   (分隔符權重損失)
─────────────────────────────────
Loss_total:           ~2.6 - 4.1   (三項之和)
```

**監控建議**：
- 若 `Loss_total` 無下降趨勢 → 檢查 λ_fcp 是否過大
- 若 `penalty_EOS` 持續 > 0.1 → think 區未學會避免 EOS，增加 λ_fcp
- 若結構準確率未提升 → w_struct=2.5 可能過小，檢查 structure_weights 是否正確應用
- **不應出現**：Loss 值 > 10 或 NaN（表示實現有誤）

---

## 📊 驗證方案

### FCP 驗證
```bash
# 監控指標
- EOS probability in think region (should ↓ from ~0.1 to ~0.01)
- Think region completion rate (should ↑)
- Early termination count (should ↓)

# Checkpoint test
# 每 500 steps 打印：
# [Step 500] Think EOS prob: 0.032 | Separator loss: 0.234
```

### SFT-GO 驗證
```bash
# 監控指標
- Structure token loss (should ↓ faster)
- Output format accuracy (human eval: correct Step/| usage)
- Structure token prediction (token-level accuracy on R1-R6)

# Checkpoint test
# 每 500 steps 打印：
# [Step 500] Struct loss: 0.156 | Format score: 8.2/10
```

---

## 🔧 代碼修改清單

### 需要修改的文件

| 文件 | 修改內容 | 優先級 |
|------|--------|--------|
| `train_sft.py` | 新增 FCP 計算函數、EOS penalty | 🔴 必須 |
| `train_sft.py` | 新增分隔符權重應用 | 🔴 必須 |
| `train_sft.py` | 集成 SFT-GO 結構權重 | 🔴 必須 |
| `train_sft.py` | 修改 loss 計算邏輯 | 🔴 必須 |
| `train_sft.py` | 新增 FCP/SFT-GO 監控日誌 | 🟡 可選 |
| `MaterializedSftDataset` | 加載結構權重 | 🔴 必須 |

### 不修改/不需要的文件

- ❌ IPS (Inverse Probability Sampling) — 跳過
- ❌ Focal Loss — 跳過
- ❌ SCALe (Schedule Loss Annealing) — 跳過
- ✅ build_structure_weights.py — 已完成
- ✅ validate_and_plot.py — 已完成
- ✅ visualize_structure_weights.py — 已完成

---

## 📅 時間估計

| 任務 | 時間 | 備註 |
|------|------|------|
| FCP 實現 | 4-6 小時 | 包括 EOS 檢測、penalty 計算、集成 |
| SFT-GO 集成 | 2-3 小時 | 加載權重、應用乘法、驗證 |
| 單元測試 | 2-3 小時 | 測試 FCP penalty 計算、權重應用 |
| 小規模訓練驗證 | 4-6 小時 | 100-500 steps on 1 GPU |
| **總計** | **12-18 小時** | 約 1.5-2 個工作天 |

---

## 🎯 成功標準

### FCP 達成
- [ ] EOS probability in think 區 < 0.01（訓練中期）
- [ ] Think 區完成率 > 95%（無提早中斷）
- [ ] Separator 識別正確率 = 100%

### SFT-GO 達成
- [ ] 結構 token 的 per-token accuracy ↑ 5-10%（vs baseline）
- [ ] 生成的 Step 格式正確率 > 90%
- [ ] 表格豎線、粗體等結構保留率 > 85%

### 集成達成
- [ ] 無 CUDA 報錯
- [ ] Loss 曲線平穩下降
- [ ] Inference 時能輸出格式正確的 CoT

---

## 📝 文檔清單

待新增：
- [ ] `FCP_IMPLEMENTATION.md` — FCP 技術細節
- [ ] `SFT_GO_INTEGRATION.md` — SFT-GO 集成指南
- [ ] `TRAINING_LOG_FORMAT.md` — 監控日誌格式說明

---

## 🚀 執行流程

```
1. 準備 train_sft.py
   ├─ 新增 find_separator_tokens()
   ├─ 新增 compute_fcp_penalty()
   └─ 新增 apply_structure_weights()

2. 集成 FCP 到 loss 計算
   ├─ 修改 loss = criterion(...) 
   ├─ 加入 + penalty_fcp
   └─ 加入 + loss_separator

3. 集成 SFT-GO 到 loss 計算
   ├─ 加載 structure_weights
   ├─ 乘入 loss * w_struct
   └─ 驗證權重應用

4. 監控與驗證
   ├─ 新增訓練日誌（FCP penalty、結構 loss）
   ├─ 100-step 檢查點
   └─ 半小時訓練驗證

5. 測試 & 迭代
   ├─ 調整 λ、δ 參數
   ├─ 調整 w_struct 數值
   └─ A/B 對比（有無 FCP/SFT-GO）
```

---

**下一步**：確認開始實現 FCP，還是先做充分的代碼準備？
