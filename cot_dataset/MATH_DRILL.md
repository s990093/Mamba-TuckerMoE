# Math Drill 類別規格（`math_drill.json`）

> **少量、低同質**的基礎算術補強。目標是讓模型「會算」，而不是用 2000 筆把權重綁死在同一套日誌格式上。
>
> ⚠️ 與 `noise.json` 內的 `math_basic` 不同：noise 可帶購物情境；本檔只做**稀疏**裸算式，且 **CoT 必須像人在黑板上推導**。

---

## 1. 總量（嚴格上限）

| 項目 | 值 |
| ---- | --- |
| 檔案 | `math_drill.json` |
| **建議總筆數** | **≤ 200**（腳本預設 **200**） |
| ID | `mat_0001` ~ `mat_0200` |
| 語言 | 全英文 |
| 產生 | `python3 cot_dataset/scripts/generate_math_drill.py` |

**不要**擴到 2000。同一規則（例如 ×100）在一個位數 tier 內 **20~50 筆已夠**；其餘 SFT 配額留給 Emotion / noise / Email 等多元任務。

---

## 2. 子分類配額（預設 200）

| `category` | 筆數 | 抽樣方式 |
| ---------- | ---- | -------- |
| `arith_add_units` | 40 | 從 0~9 加法池（100 組）隨機抽 |
| `arith_mul_table` | 48 | 從 1~12 乘法表（144 組）隨機抽 |
| `arith_mul_teens` | 24 | 從 13~19 池（49 組）隨機抽 |
| `arith_add_mixed` | 36 | 從 10~99 加法池隨機抽 |
| `arith_mul_extended` | 36 | 從 20~99 乘法池隨機抽 |
| `arith_mul_hundred` | **16** | 從 `n×100`（100 組）隨機抽 — **禁止**寫滿 100 筆 |

---

## 3. System Prompt（bucket：`math_drill`）

```text
You are Mamba answering a quick arithmetic question in English.
Reason briefly in plain language (no "Step 1" labels, no Parse operands / Emit answer phrasing).
Then give the numeric result only in the final line — digits, no extra words.
```

---

## 4. CoT 與 output 規則（防格式中毒）

### 禁止（會污染全模型）

- ❌ `Step 1:` / `Step 2:` …
- ❌ `**Parse operands**` / `**Execute operation**` / `**Emit answer**`
- ❌ `**Answer: 7600**` 在 `output` 欄（訓練腳本會包 `<final>`，勿雙重包裝）

### 正確風格

**`cot`**：1~3 句自然英文，解釋「為什麼」或心算路徑，像黑板草稿。

**`output`**：**僅數字**字串，例如 `"7600"`。

### 訓練時組裝（由 `stf_cot_to_bin.py` 自動完成，協作者勿手寫）

```
<|im_start|>user
What is 76 times 100?<|im_end|>
<|im_start|>assistant
<think>
Multiplying 76 by 100 appends two zeros — the digits shift two places left.
76 → 7600.
</think>
<final>
7600
</final><|im_end|>
```

---

## 5. JSON 範例

```json
{
  "id": "mat_0001",
  "category": "arith_add_units",
  "input": "What is 1 plus 1?",
  "cot": "Adding 1 and 1 gives 2.",
  "output": "2"
}
```

```json
{
  "id": "mat_0185",
  "category": "arith_mul_hundred",
  "input": "What is 76 times 100?",
  "cot": "Multiplying 76 by 100 is the same as appending two zeros — the digits shift two places to the left.\n76 → 7600.",
  "output": "7600"
}
```

---

## 6. 三個為什麼不能堆量

| 風險 | 說明 |
| ---- | ---- |
| **Over-representation** | 2000 筆同質算術 → 權重偏向數字/乘法，General 能力被擠壓 |
| **Format poisoning** | 死板 Step CoT → 情緒/天氣題也輸出 `Parse operands` |
| **容量浪費** | 417M 模型應學多元 CoT；算術只需「嘗試味道」的少量樣本 |

---

## 7. 產生與檢查

```bash
python3 cot_dataset/scripts/generate_math_drill.py
python3 -m json.tool cot_dataset/math_drill.json > /dev/null
# 確認無 Step 1: 、無 Answer: 前綴
rg 'Step [0-9]:|Parse operands|Emit answer|\\*\\*Answer:' cot_dataset/math_drill.json
```

`rg` 應 **無匹配**。
