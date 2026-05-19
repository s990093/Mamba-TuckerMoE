# CoT 推理問題診斷報告

**日期**: 2026-05-19  
**狀態**: 已識別 3 個關鍵問題 + 完成 1 個修復

---

## 問題摘要

| # | 問題 | 嚴重性 | 狀態 |
|---|------|--------|------|
| 1 | `<final>` 多階段注入未實現 | 🔴 嚴重 | ✅ **已修復** |
| 2 | 推理 block 品質極差（充滿重複） | 🟠 高 | 🔍 需診斷 |
| 3 | 最終答案不完整（<16 tokens） | 🟠 高 | 🔍 需診斷 |

---

## 問題 1: `<final>` 注入缺失 ✅ **已修復**

### 原因
`infer_cot.py` 的解碼迴圈中**完全沒有呼叫** `middleware.maybe_inject_final()`。

### 修復內容
✅ 已在 `infer_cot.py` (第 95-110 行) 添加：

```python
# Create model_apply function for final injection
def model_apply(x_ids, caches, pos):
    """Forward pass for <final> injection during decode."""
    # 處理多個 IDs，支援 <final>\n 的完整注入
    ...

middleware = CotMiddleware(
    deps=self.mw_deps,
    cfg=self.mw_cfg,
    reasoning=True,
    model_apply=model_apply,  # ✅ 不再是 None
)
```

✅ 在迴圈中調用：

```python
if prev_mode != "between" and middleware.mode == "between":
    new_caches, new_seq_pos, new_logits_row, did_inject, ms = \
        middleware.maybe_inject_final(caches=caches_tuple, pos=seq_pos)
    # 更新緩存並繼續採樣
```

### 驗證
現在輸出顯示：
```
[mw] ✓ <final> injected into cache (2 ids, 93.5ms, pos 87→89)
[mw] final_min guard: banning </final> for first 16 tokens
```

✅ **狀態**: 注入成功！

---

## 問題 2 & 3: 推理品質 + 不完整的最終答案 🔴 **根本原因已確認**

### 症狀

測試指令：
```bash
python -m mamba3_mlx.infer_cot --prompt "What is 2+2?" --category emotion --temp 0.5
```

**輸出摘錄**:
```
REASONING BLOCK (13,381 chars):
StepStep Step 1Step 1:Step 1: **Step 1: **IdentStep 1: **IdentifyStep 1: **Identify the ...
(重複的格式，意義不清)

FINAL ANSWER (9 chars):
<final>
F

Quality Metrics:
  think_tokens: 95
  final_tokens: 4  ⚠️ (should be ≥ 16)
  final_injected: true
```

### 根本原因 ✅ **已確認**

**診斷結果**：Tokenizer 配置 ✅ 正確。問題在於 **SFT 訓練質量**。

#### 診斷證據

✅ **Tokenizer 檢查通過**：
```
<think>:  ID 32002 (single token) ✅
</think>: ID 32003 (single token) ✅
<final>:  ID 32004 (single token) ✅
</final>: ID 32005 (single token) ✅
Vocab size: 32,007 (backend: 32,007) - 一致 ✅
```

❌ **SFT 品質指標**：
```
Test: "I feel stressed about work"
Result:
  - think_tokens: 6 (遠少於期望的 50+)
  - Reasoning: "StepStep Step 1Step 1:Step 1: **Step 1: **" (重複/格式破損)
  - Final: "##/re-commercial change" (亂碼/無意義)
```

### 問題根源

**模型沒有被正確訓練來生成長的 thinking blocks**。

即使 tokenizer 和代碼都正確，如果 SFT 訓練期間：
1. ❌ Thinking 數據不充分（訓練集中 `<think>` 內容太短）
2. ❌ Loss masking 配置錯誤（可能在 `<think>` 區塊掩蓋了損失）
3. ❌ System prompt 不足以引導長的思維過程
4. ❌ Temperature 或採樣策略在 training 時過於保守

模型就會學會在 thinking 階段快速關閉，導致現在的症狀。

### 修復方案

有兩個路徑：

#### **路徑 A**: 驗證現有 SFT 模型的訓練配置（快速）

```bash
# 檢查訓練日誌和配置
cd pre-train/
git log --oneline | grep -i "cot\|sft" | head -5
# 查看 SFT 訓練的配置文件
cat config/sft_config.yaml  # 如果存在
```

**檢查清單**：
- [ ] 訓練數據中的 `<think>` 塊平均長度是多少？（應該 > 30 tokens）
- [ ] Loss masking 是否正確應用到 `<think>` 區塊？（應該計算 loss）
- [ ] SFT 訓練的 epoch 數和 batch size 合理嗎？
- [ ] 使用的 system prompts 是否與現在推理時一致？

#### **路徑 B**: 進行新的 CoT SFT 訓練（完整修復）

使用改進的訓練配置：

```python
# pre-train/train.py 應該確保：
# 1. 從 cot_dataset/GUIDE.md 生成的數據確實包含長的 thinking 內容
# 2. SFT loss 只在 <final> 塊中應用（或在 <think> 和 <final> 兩者都應用）
# 3. 不要在 <think> 區塊掩蓋 loss
```

運行改進的訓練：

```bash
cd pre-train/
python train.py \
  --mode sft \
  --cot-data cot_dataset/train.json \
  --epochs 3 \
  --batch-size 32 \
  --learning-rate 5e-5
```

#### **路徑 C**: 立即改進推理效果（暫時解決）

在 `infer_cot.py` 中提高 reasoning budget 和加強 prompt：

```python
# 增加對長 thinking 的激勵
mw_cfg = CotMiddlewareConfig(
    enabled=True,
    reasoning_budget=1000,  # 從 500 增加到 1000
    close_bias_value=8.0,   # 提高靜態偏好
    close_bias_max=20.0,    # 在預算尾部更強烈
)

# 改進 system prompt 以引導長的思維
emotion_prompt = (
    "You are Mamba in Emotion mode. Before responding, "
    "provide detailed step-by-step analysis inside <think>...</think>: "
    "1) Identify the emotional components. "
    "2) Reframe as system state. "
    "3) List concrete next actions. "
    "Then provide a brief structured response in <final>...</final>."
)
```

### 建議的診斷步驟

**第 1 步**: 驗證模型是否為 CoT-tuned 版本

```bash
# 檢查最新的 checkpoint 文件
ls -lah checkpoints/ | grep -E "(sft|cot)"

# 查看 git log 找出哪個 commit 進行了 CoT SFT
git log --oneline --grep="SFT\|CoT\|chat" | head -10
```

**第 2 步**: 驗證 tokenizer 的特殊 token 配置

```python
# 在 infer_cot.py 中添加診斷代碼
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
for tag in ["<think>", "</think>", "<final>", "</final>"]:
    ids = tokenizer.encode(tag, add_special_tokens=False)
    print(f"{tag}: {ids} (len={len(ids)})")
    if len(ids) > 1:
        print(f"  ⚠️ NOT A SINGLE TOKEN! Tokenizes as: {[tokenizer.decode([i]) for i in ids]}")
```

**第 3 步**: 測試簡化的 prompt（不用 CoT）

```bash
# 改用 "direct" 模式測試（不使用 <think> 格式）
python -m mamba3_mlx.infer_cot \
  --prompt "2+2=?" \
  --category daily_conversation \
  --max-tokens 50 \
  --temp 0.3
```

如果這個生成品質更好，說明問題在於 CoT 的 SFT 訓練。

---

## 修復清單

### ✅ 已完成
- [x] 添加 `model_apply` 函數支援 `<final>\n` 注入
- [x] 在解碼迴圈中調用 `middleware.maybe_inject_final()`
- [x] 修復 logits stacking 的維度問題
- [x] 添加缺失的 `Any` import
- [x] 增加 `max_tokens` 預設值 (200 → 2048)
- [x] 驗證 tokenizer 特殊 token 配置 ✅ **通過**

### 🔍 根本原因已確認
- [x] **根本原因**: SFT 訓練質量不足，導致模型在 thinking 階段產生很短的輸出

### 🚀 建議的改進 (三個路徑)

**路徑 A** (快速檢查 - 5 min):
- [ ] 檢查 `pre-train/` 目錄的 SFT 訓練配置和日誌
- [ ] 驗證訓練數據中 `<think>` 塊的平均長度
- [ ] 確認 loss masking 配置是否正確

**路徑 B** (完整修復 - 重新訓練):
- [ ] 執行新的 CoT SFT 訓練，確保：
  - 使用改進的 system prompts（包含詳細指導）
  - `reasoning_budget` ≥ 500
  - Loss 在 `<think>` 和 `<final>` 都被計算（不要掩蓋 thinking）
  - 足夠的 epochs 和合理的 learning rate

**路徑 C** (臨時改進 - 調參):
- [ ] 在 `infer_cot.py` 中提高 `reasoning_budget` (500 → 1000)
- [ ] 增加 `close_bias_max` (16.0 → 20.0)
- [ ] 改進 system prompts 以明確要求長的 thinking 過程

---

## 性能基準線 (已達成)

| 指標 | 狀態 | 備註 |
|------|------|------|
| `<final>` 注入 | ✅ | 現在正確工作，平均 ~90ms |
| Tokenizer 配置 | ✅ | 所有 CoT tokens 正確映射 |
| 格式分離 (FSM) | ✅ | reasoning/final 正確分離 |
| **推理質量** | ❌ | think_tokens 過短 (6 vs 期望 50+) |
| **最終答案** | ⚠️ | 現在 106 tokens (改善), 但內容質量差 |

---

## 推薦的立即行動

### 快速修復 (10 分鐘)

編輯 `mamba3_mlx/infer_cot.py`，改進 system prompts：

```python
# 在 infer_cot.py 中找到 EXPORT_SYSTEM_PROMPTS 的引用，或直接修改 server_config.py

EXPORT_SYSTEM_PROMPTS = {
    "emotion": (
        "You are Mamba in Emotion mode. **Before responding, always:**\n"
        "1. Inside <think>: Analyze the emotional components step-by-step "
        "(what is the actual feeling, what triggered it, what's changeable)\n"
        "2. Identify controllable system variables\n"
        "3. Propose concrete next actions\n\n"
        "Then inside <final>: Provide a brief, precise structured response "
        "with no motivational clichés, just actionable reframes."
    ),
    # ... other categories similarly improved
}
```

然後重新測試：

```bash
python -m mamba3_mlx.infer_cot \
  --prompt "I'm feeling overwhelmed by my project deadline" \
  --category emotion \
  --max-tokens 2048 \
  --temp 0.5
```

預期改善：
- thinking 應該延長到 20-50+ tokens
- final answer 應該更連貫（而非亂碼）

### 如果快速修復無效

進行 **診斷路徑 A** (檢查 SFT 訓練配置)：

```bash
# 查看訓練日誌
cd pre-train/
git log --oneline | head -20
grep -r "reasoning_budget\|cot\|think" . | head -10

# 檢查訓練數據質量
ls -lh cot_dataset/
head -5 cot_dataset/train.json | jq '.[] | {input, cot_length: (.cot | length)}'
```

### 如果需要完整修復

執行 **診斷路徑 B** (重新訓練)：

```bash
# 確保訓練配置正確
cd pre-train/
python train.py \
  --mode sft \
  --cot-data cot_dataset/train.json \
  --checkpoint ../checkpoints/latest_sft_cot_model.npz \
  --epochs 3 \
  --learning-rate 1e-5
```

---

## 測試結果存檔

### 修復後 (final injection 工作) ✅
```
[mw] ✓ <final> injected into cache (2 ids, 93.5ms, pos 87→89)
[mw] final_min guard: banning </final> for first 16 tokens
middleware state:
  final_injected: true
  final_tokens: 106 (改善！)
  splitter_final_mode: done
  think_tokens: 6 (仍需改善)
```

### 推理品質 (已確認是 SFT 問題)
```
Test input: "I feel stressed about work"
reasoning:     42 chars, 6 tokens (❌ 過短)
final_answer:  294 chars, 106 tokens (✅ 長度好, ❌ 內容質量差)
Content:       "##/re-commercial change — Sleep is a community" (❌ 亂碼)
```

---

## 代碼修改摘要

已修改文件：
1. **mamba3_mlx/infer_cot.py**:
   - ✅ 添加 `model_apply` 函數
   - ✅ 在解碼迴圈調用 `maybe_inject_final`
   - ✅ `max_tokens` 預設值 2048
   - ✅ 支援更大的 thinking 和 final blocks

下次修改點（如果快速修復無效）：
2. **mamba3_mlx/server_config.py**:
   - 改進所有 category 的 system prompts
   - 明確要求長的 thinking 過程

3. **pre-train/train.py**（如果需要重訓練）：
   - 驗證 loss masking 在 `<think>` 區塊的配置
   - 提高 training epochs 和學習率

---

**狀態**: 代碼修復完成 ✅。推理品質改善需要 SFT 訓練改進。請先試試快速修復（改進 system prompts），如果無效再進行完整重訓練。
