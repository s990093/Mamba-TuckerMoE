# CoT (Chain-of-Thought) Diagnosis Guide

生成品質下降的診斷腳本集合。用這三個腳本來隔離問題：

## 🔍 三層診斷

### 1️⃣ **diagnose_cot.py** — 組件單元測試
檢查 `cot_format_fsm.py` 和 `cot_middleware.py` 的基礎功能。

```bash
python diagnose_cot.py --test all
```

**檢查項目：**
- ✓ `FormatGuard` 初始化（禁止列表、關閉偏差）
- ✓ `CotStreamSplitter` 標籤解析（`<think>`, `</think>`, `<final>`, `</final>`）
- ✓ 模式轉換（`head` → `think` → `between` → `final` → `done`）
- ✓ 中間件配置和狀態管理

**預期結果：** `✓ All tests passed`

**如果失敗：** 檢查 FSM 的標籤識別邏輯或中間件配置。

---

### 2️⃣ **validate_cot_simple.py** — 簡單推理測試
用 `"who are you?"` 問題運行實際推理，驗證：
- 推理內容是否被正確分離
- 最終答案是否存在
- CoT 標籤是否正確生成

```bash
python validate_cot_simple.py --max-tokens 150 --temp 0.7
```

**檢查項目：**
- ✓ 是否有 `<think>…</think>` 塊
- ✓ 是否有 `<final>…</final>` 塊
- ✓ 流分割器是否正確分離（`reasoning` vs `final` 事件）
- ✓ 中間件狀態轉換日誌

**預期結果：**
```
✓ Has reasoning: True
✓ Has final answer: True
✓ Reached final mode: True
✓ CoT separation working: both reasoning and final answer present
```

**如果失敗的症狀：**

| 症狀 | 可能原因 |
|------|--------|
| `No reasoning: True` `No final answer: True` | 標籤完全未生成；檢查 SFT 訓練格式 |
| `Has reasoning: True` `No final answer: False` | `</think>` 或 `<final>` 未生成；預算不足 |
| `Has reasoning: False` `Has final answer: True` | 模型跳過思考直接生成答案；檢查 `force_final_inject` 或系統提示 |
| `Splitter mode: 'between'` | 模型卡在 `</think>` 後，未生成 `<final>`；檢查 `final_min_tokens` |

---

### 3️⃣ **test_cot_inference.py** — 端到端測試框架
針對特定系統提示（如 `self_awareness`, `reasoning`）的多輪推理。

```bash
python test_cot_inference.py --test-case self_awareness --max-tokens 300
```

**系統提示：**
```python
"self_awareness": "You are Mamba in Self-Awareness mode. Answer identity and capability..."
"reasoning": "You are in Reasoning mode. For complex questions, think through step-by-step..."
```

**檢查項目：**
- ✓ 不同系統提示下的一致性
- ✓ 推理預算（`reasoning_budget`）是否尊重
- ✓ 最終答案質量（長度、相關性、準確性）
- ✓ 性能指標（tok/s, 時間）

---

## 🔧 常見問題診斷

### 問題 1：生成質量變差，但沒有明顯錯誤
**可能原因：** 中間件干擾了解碼過程

```bash
# 步驟 1：禁用所有中間件功能
python validate_cot_simple.py --config no-guard

# 步驟 2：逐個啟用功能
python validate_cot_simple.py --config ban-only
python validate_cot_simple.py --config close-bias-only
python validate_cot_simple.py --config all
```

---

### 問題 2：模型卡在 `<think>` 裡面
**可能原因：**
1. `close_bias` 太弱（無法逼迫 `</think>`）
2. `reasoning_budget` 太低（觸發硬停止，但未正確生成標籤）
3. 分詞器未識別 `</think>` 為單個 token

**診斷：**
```bash
# 檢查 </think> token ID
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('checkpoints/tokenizer')
print(f'</think> ID: {tok.convert_tokens_to_ids(\"</think>\")}')
print(f'Encoded: {tok.encode(\"</think>\")}')
"

# 運行並查看中間件報告
python validate_cot_simple.py 2>&1 | grep -A 20 "MIDDLEWARE STATE"
```

**可能的修復：**
```python
middleware_cfg = CotMiddlewareConfig(
    close_bias_value=8.0,      # 增加偏差強度
    close_bias_max=24.0,       # 增加峰值
    close_bias_start=50,       # 更早開始
    reasoning_budget=1500,     # 給更多空間
)
```

---

### 問題 3：生成答案但缺少推理
**可能原因：**
1. SFT 數據未正確掩蓋（丟失 `<think>` 塊訓練信號）
2. `force_final_inject` 太激進（跳過思考）

**診斷：**
```bash
# 檢查數據集格式
python -c "
import json
with open('cot_dataset/data.json') as f:
    sample = json.load(f)[0]
    print('Format:', sample.keys())
    print('CoT:', sample.get('cot', '')[:100])
"
```

---

### 問題 4：標籤解析不正確
**症狀：** 分流器卡在 `head` 或 `between` 模式

**診斷：**
```bash
# 運行分流器單元測試
python diagnose_cot.py --test splitter

# 查看具體失敗的測試
python -c "
from mamba3_mlx.cot_format_fsm import CotStreamSplitter

# 測試你的特定輸出
splitter = CotStreamSplitter(start_in_think=True)
test_output = '<think>test</think><final>answer</final>'
events = splitter.feed(test_output)
print('Events:', events)
print('Final mode:', splitter.mode)
"
```

---

## 📊 效能檢查清單

運行後應查看：

```bash
python validate_cot_simple.py | tee /tmp/cot_test.log
```

**關鍵指標：**
- ✓ `Generated tokens > 100`（模型有足夠時間思考）
- ✓ `Has reasoning: True` 和 `Has final answer: True`
- ✓ `Reasoning/Final ratio` 在 40/60 或 50/50 之間
- ✓ `Time < 5000ms`（推理速度合理）

**閾值：**
| 指標 | 好 | 警告 ⚠️ | 不好 ✗ |
|------|-----|--------|--------|
| 推理 tokens | > 50 | 20-50 | < 20 |
| 最終答案 tokens | > 100 | 50-100 | < 50 |
| 推理到期時間 | < 2s | 2-5s | > 5s |

---

## 🚀 使用 CoT 的核心檢查

### ✅ 格式必須正確
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>
{reasoning_text_here}
</think>
<final>
{final_answer_here}
</final><|im_end|>
```

### ✅ SFT 數據必須包含 CoT
```json
{
  "input": "Who are you?",
  "cot": "Let me think about my architecture...",
  "output": "I am Mamba...",
  "category": "self_awareness"
}
```

### ✅ 分詞器必須正確編碼標籤
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("checkpoints/tokenizer")

# 這些必須都是有效的 token
for tag in ["<think>", "</think>", "<final>", "</final>"]:
    ids = tok.encode(tag, add_special_tokens=False)
    print(f"{tag}: {ids}")  # 應該是非空的
```

---

## 💡 逐步調試流程

1. **運行 `diagnose_cot.py`**
   - 確認組件本身工作正常
   - 如果失敗，修復 FSM 或中間件代碼

2. **運行 `validate_cot_simple.py`**
   - 用簡單的問題測試實際推理
   - 如果分離不工作，問題在生成或格式化

3. **對比輸出**
   ```bash
   # 保存參考輸出
   python validate_cot_simple.py --temp 0.7 > /tmp/baseline.txt
   
   # 修改配置並對比
   python validate_cot_simple.py --temp 0.5 > /tmp/modified.txt
   diff /tmp/baseline.txt /tmp/modified.txt
   ```

4. **隔離問題**
   - 問題在組件？→ 修復 `cot_format_fsm.py`
   - 問題在中間件？→ 調整 `CotMiddlewareConfig` 參數
   - 問題在訓練？→ 檢查 `cot_dataset/` 格式和掩蓋

---

## 📝 關鍵參數

### `CotMiddlewareConfig`
```python
CotMiddlewareConfig(
    enabled=True,                    # 啟用所有 CoT 保護
    ban_im_start=True,               # 禁止 <|im_start|> 在答案中
    close_bias_value=4.0,            # 基礎關閉偏差
    close_bias_max=16.0,             # 峰值關閉偏差（在預算末尾）
    close_bias_start=0,              # 何時開始應用偏差
    reasoning_budget=2000,           # 硬 cap：思考 tokens
    force_final_inject=True,         # `</think>` 後自動注入 `<final>\n`
    final_min_tokens=16,             # 最終答案的最小長度
)
```

---

## 🎯 成功的標誌

運行後看到：
```
✓ All tests passed
✓ CoT separation working: both reasoning and final answer present
```

以及：
```json
{
  "reasoning_len": 150,
  "final_len": 200,
  "think_tokens": 60,
  "final_tokens": 80
}
```

---

## 📞 如果仍然有問題

1. 檢查 git 日誌中最近的 CoT 相關變更
   ```bash
   git log --oneline --all | grep -i cot | head -10
   ```

2. 對比之前工作的版本
   ```bash
   git diff HEAD~5 -- mamba3_mlx/cot_*.py
   ```

3. 運行所有診斷腳本並保存輸出
   ```bash
   ./diagnose_cot.py > /tmp/diag.txt 2>&1
   ./validate_cot_simple.py >> /tmp/diag.txt 2>&1
   ```
