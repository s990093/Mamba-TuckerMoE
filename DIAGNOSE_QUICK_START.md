# CoT 診斷 — 快速開始

## 🚀 5 分鐘快速診斷

生成品質變差的問題來自 `cot_format_fsm.py` 或 `cot_middleware.py` 兩個新模組。這三個診斷腳本可以快速隔離問題。

### 步驟 1：驗證組件本身
```bash
python diagnose_cot.py --test all
```

**預期：** 最後一行顯示 `✓ All tests passed`

✓ 表示 `cot_format_fsm.py` 和 `cot_middleware.py` 邏輯正確。

---

### 步驟 2：驗證實際推理
```bash
python validate_cot_simple.py --max-tokens 150 --temp 0.7
```

**查看：**
- 是否有 `[REASONING BLOCK]` 內容？
- 是否有 `[FINAL ANSWER]` 內容？
- 最後是否說 `✓ CoT separation working`？

**如果✓：** 中間件在推理中正常工作。問題可能在更高層（如 UI/API 集成）。

**如果✗：** 見下面的 [故障排查](#故障排查)。

---

### 步驟 3：測試 self_awareness 提示
```bash
python validate_cot_simple.py --max-tokens 200 --temp 0.5
```

最後查看 `QUALITY CHECK` 部分：

```
✓ Has reasoning: True
✓ Has final answer: True
✓ Reached final mode: True
✓ Thinking blocks properly formatted: True

✓ CoT separation working: both reasoning and final answer present
```

---

## 🔧 故障排查

### 症狀 1：`Has reasoning: False` + `Has final answer: False`
**問題：** 模型根本沒有生成 `<think>…</think>` 或 `<final>…</final>`

**檢查：**
```bash
# 1. 查看原始生成（帶標籤）
python validate_cot_simple.py 2>&1 | grep -A 5 "RAW OUTPUT"

# 2. 確認分詞器認識這些標籤
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('checkpoints/tokenizer')
for tag in ['<think>', '</think>', '<final>', '</final>']:
    ids = tok.encode(tag, add_special_tokens=False)
    print(f'{tag}: {ids}')
"
```

**可能原因：**
- ❌ SFT 訓練未包含 CoT 格式
- ❌ 分詞器不認識這些標籤
- ❌ 模型權重不是 CoT-aware 版本

---

### 症狀 2：`Has reasoning: True` + `Has final answer: False`
**問題：** 模型生成了 `<think>` 但卡住，沒有生成 `</think>` 或 `<final>`

**檢查：**
```bash
python validate_cot_simple.py 2>&1 | grep -A 3 "MIDDLEWARE STATE"
```

查看：
- `reasoning_budget: 1000` — 預算是否耗盡？
- `splitter_final_mode: "think"` — 仍卡在思考？
- `think_tokens: 1000+` — 達到或超過預算？

**修復：**
```python
# 在 validate_cot_simple.py 中：
middleware_cfg = CotMiddlewareConfig(
    close_bias_value=8.0,      # 增加到 8（從 4）
    close_bias_max=24.0,       # 增加到 24（從 16）
    close_bias_start=50,       # 提早開始
    reasoning_budget=3000,     # 增加預算
)
```

---

### 症狀 3：`Has reasoning: False` + `Has final answer: True`
**問題：** 模型完全跳過了思考，直接生成答案

**檢查：**
```bash
python validate_cot_simple.py 2>&1 | grep "final_injected"
```

**修復方案：**

選項 A：強制進入思考模式（在 validate_cot_simple.py 中）
```python
full_prompt = (
    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
    f"<|im_start|>assistant\n"
    f"<think>\n"  # ← 強制進入思考
)
```

選項 B：啟用 `force_final_inject`
```python
middleware_cfg = CotMiddlewareConfig(
    force_final_inject=True,  # 自動注入 <final>\n
)
```

---

### 症狀 4：分流器卡在 `"between"` 模式
**問題：** 模型生成了 `</think>` 但未生成 `<final>`

**檢查：**
```bash
python validate_cot_simple.py 2>&1 | grep "splitter_final_mode"
```

**修復：**
增加 `close_bias_max`：
```python
middleware_cfg = CotMiddlewareConfig(
    close_bias_max=32.0,  # 增加峰值（吸引 <final>）
)
```

---

## 📋 完整診斷流程

如果上面的快速檢查沒解決問題，運行完整流程：

```bash
# 1. 單元測試
python diagnose_cot.py --test guard
python diagnose_cot.py --test splitter
python diagnose_cot.py --test deps
python diagnose_cot.py --test orchestration

# 2. 推理驗證
python validate_cot_simple.py --max-tokens 150 --temp 0.5
python validate_cot_simple.py --max-tokens 150 --temp 0.8

# 3. 儲存完整報告
(python diagnose_cot.py --test all && \
 python validate_cot_simple.py --max-tokens 200) > /tmp/cot_diagnosis.txt 2>&1
cat /tmp/cot_diagnosis.txt | grep -E "(✓|✗|error|ERROR)"
```

---

## 🎯 預期結果

成功的診斷看起來像：

```
╔════════════════════════════════════════════════════════════════════════════╗
║                      CoT FORMAT & MIDDLEWARE DIAGNOSTIC                    ║
╚════════════════════════════════════════════════════════════════════════════╝

Loading tokenizer... ✓
Vocab size: 32000

================================================================================
TEST: Format Guard
================================================================================
✓ Guard initialized: 1 banned IDs
✓ Close bias targets: {'think': 829, 'final': 829}

================================================================================
TEST: Stream Splitter
================================================================================
✓ Complete reasoning + final
✓ Reasoning + final injection
✓ No reasoning (head mode)
✓ Incomplete: between mode EOS before final

================================================================================
SUMMARY
================================================================================
✓ All tests passed


╔════════════════════════════════════════════════════════════════════════════╗
║                      SIMPLE CoT VALIDATION (who are you?)                 ║
╚════════════════════════════════════════════════════════════════════════════╝

System prompt: You are Mamba in Self-Awareness mode...
User prompt: Who are you?

Running inference...

[REASONING BLOCK] (342 chars)
Let me think about this carefully. I'm a hybrid architecture combining...

[FINAL ANSWER] (521 chars)
I am Mamba, a hybrid Mamba-TuckerMoE language model...

================================================================================
QUALITY CHECK
================================================================================
✓ Has reasoning: True
✓ Has final answer: True
✓ Reached final mode: True
✓ Thinking blocks properly formatted: True

✓ CoT separation working: both reasoning and final answer present
```

---

## 📞 仍然有問題？

1. **檢查最近的改動**
   ```bash
   git diff HEAD~3 -- mamba3_mlx/cot_*.py
   ```

2. **查看完整診斷指南**
   ```bash
   cat COT_DIAGNOSIS_GUIDE.md
   ```

3. **收集所有信息**
   ```bash
   python diagnose_cot.py --test all > /tmp/d1.txt 2>&1
   python validate_cot_simple.py > /tmp/d2.txt 2>&1
   echo "=== diagnose_cot.py ===" && cat /tmp/d1.txt
   echo "=== validate_cot_simple.py ===" && cat /tmp/d2.txt
   ```

---

## ✅ 最小可行檢查

如果你只有 5 分鐘：

```bash
# 1. 組件是否工作？
python diagnose_cot.py --test all | tail -10

# 2. 推理是否產生 CoT？
python validate_cot_simple.py 2>&1 | grep -E "(QUALITY CHECK|✓|✗)" | head -10
```

如果都是 ✓，問題不在這兩個模組。

如果有 ✗，見上面的故障排查。

---

## 三個診斷腳本

| 腳本 | 用途 | 運行時間 |
|------|------|--------|
| `diagnose_cot.py` | 單元測試 FSM + 中間件 | < 1s |
| `validate_cot_simple.py` | 單個推理 ("who are you?") | 10-30s |
| `test_cot_inference.py` | 多輪推理框架 | 1-3m |

**開始：** `python diagnose_cot.py --test all`

---

## 核心問題檢查清單

- [ ] `diagnose_cot.py` 所有測試都通過？
- [ ] `validate_cot_simple.py` 顯示 `✓ CoT separation working`？
- [ ] `raw_text` 包含 `<think>`, `</think>`, `<final>`, `</final>` 標籤？
- [ ] `reasoning` 和 `final` 文本都非空？
- [ ] `middleware_state.enabled == true`？

全部✓ → 組件正常，問題在使用層

有✗ → 見故障排查
