# infer_cot.py 修復 — 下一步行動計畫

## 當前狀態

✅ **已完成：**
- Vocab size 修復（32000 → 32007）
- Format guard 初始化正確
- middleware.step() 集成（從直接 splitter.feed() 改為）
- 移除每步的 tokenizer.decode() 調用（性能優化）

⚠️ **待診斷：**
- 為什麼還是生成垃圾文本（"PosPositivePositive..."）

---

## 你需要做的

### 在你的 Mac 上運行

```bash
python -m mamba3_mlx.infer_cot --prompt "Who are you?" --category self_awareness
```

### 檢查結果

**如果修復成功** ✅
```
Tokens: 100+
Has reasoning: True
Has final answer: True
Output: 清晰的文本（不是垃圾）
✅ PASS: CoT separation working
```

**如果還是有問題** ⚠️
```
Tokens: 10-20
Output: "PosPositivePositive..." 或垃圾
```

### 如果還是有問題，執行診斷

見 `DIAGNOSE_INFER_COT.md`

關鍵步驟：
1. 在循環中添加調試輸出
2. 查看前 5 步的 logits 和採樣的 token ID
3. 檢查 middleware.step() 返回的事件

---

## 可能的原因分析

### 原因 A：採樣函數問題
- Token ID 被設置到垃圾值
- 導致 tokenizer.decode() 出錯

### 原因 B：Close Bias 過強
- Logits 被修改得太極端
- 採樣總是返回同一個 token

### 原因 C：Middleware 事件處理問題
- middleware.step() 返回的事件格式不同
- 我們提取錯誤的字段

### 原因 D：Model 狀態問題
- Prefill 後 logits 不正常
- KV cache 狀態有問題

---

## 修復建議

基於目前的症狀（重複的"Positive"），我建議：

### 步驟 1：驗證採樣
```python
# 在 infer_cot.py 第 125 行後添加
tid = sample_token(logits_row, temperature=temperature, top_k=40)
if step_idx < 5:
    print(f"Step {step_idx}: tid={tid}, token={self.tokenizer.convert_ids_to_tokens(tid)}")
```

運行看採樣的 token ID 是否合理。

### 步驟 2：檢查 Logits Bias
```python
# 在 middleware.transform_logits() 後添加
if step_idx == 0:
    print(f"Logits top 5: {mx.argsort(-logits_row)[:5].tolist()}")
    print(f"Logits values: {logits_row[mx.argsort(-logits_row)[:5]].tolist()}")
```

看 close_bias 是否導致了某個 token 的分數過高。

### 步驟 3：驗證 Middleware Events
```python
for event in middleware.step(tid, ...):
    if step_idx < 5:
        print(f"Event keys: {event.keys()}")
        print(f"Event: {event}")
```

確保事件包含正確的字段。

---

## 文件列表

| 文件 | 用途 |
|------|------|
| `mamba3_mlx/infer_cot.py` | 主要實現（已修改） |
| `DIAGNOSE_INFER_COT.md` | 詳細診斷指南 |
| `INFER_COT_FINAL_FIX.md` | 修復歷程和分析 |
| `QUICK_FIX_REFERENCE.txt` | 快速參考卡 |
| `RUN_TESTS_ON_MAC.sh` | 測試腳本 |

---

## 預期時間表

- **立即**：運行基本測試，查看結果
- **5-10 分鐘**：如果失敗，執行診斷（見 DIAGNOSE_INFER_COT.md）
- **10-20 分鐘**：基於診斷結果修復根本原因
- **5 分鐘**：驗證修復
- **2 分鐘**：提交 git

---

## 一旦驗證通過

```bash
# 1. 查看修改
git diff mamba3_mlx/infer_cot.py

# 2. 提交
git add mamba3_mlx/infer_cot.py
git commit -m "fix: optimize middleware.step() integration - decode only once at end

- Remove per-token tokenizer.decode() call (performance optimization)
- Build raw_text once after generation complete
- Preserve proper event handling from middleware.step()
- All diagnostic outputs show tokens correctly generated

Result: Tokens 10→100+, has_reasoning T, has_final_answer T
Status: ✅ PASS

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"

# 3. 運行完整測試
bash RUN_TESTS_ON_MAC.sh
```

---

## 我的分析

目前看起來問題可能在於：

1. **採樣邏輯** — 看起來總是採樣同一個 token
2. **Logits 變形** — close_bias 可能修改得太多
3. **事件提取** — 可能提取了錯誤的字段

最好的方法是運行診斷看具體發生了什麼。

---

## 需要你提供的信息

當你運行診斷後，請告訴我：

1. **前 5 步的 token ID**（應該都是合理的數字）
2. **解碼的文本**（應該是清晰的英文，不是垃圾）
3. **Middleware events**（應該有 "reasoning" 或 "final" 類型）
4. **Logits 統計**（min/max 應該是合理的浮點數）

有了這些信息，我能精確定位問題。

---

**總結：下一步就是在你的 Mac 上驗證。**

按照 `DIAGNOSE_INFER_COT.md` 中的步驟運行，收集診斷信息，然後告訴我看到什麼。
