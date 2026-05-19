# CoT Format FSM 架構與運作邏輯

完整的 Chain-of-Thought 推理格式控制系統，包含流分流器、格式保護和 token 識別。

---

## 📐 整體架構

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          推理生成流程 (Decode Loop)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  第 1 層：Logits 轉換 (FormatGuard)                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ • 應用禁止列表 (ban_mask)：禁止 <|im_start|>, </s>, <|im_end|>      │ │
│  │ • 應用關閉偏差 (close_bias)：鼓勵生成 </think>, <final>, </final>   │ │
│  │ • 方式依賴於 FSM 當前模式 (think/final/etc)                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                   │                                         │
│  輸入：logits[vocab_size]  →  輸出：biased_logits[vocab_size]                │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  採樣層                                                                      │
│  ├─ sample_token(biased_logits) → token_id                                  │
│  └─ 根據溫度、top_k 等採樣                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  第 2 層：Token 解碼 & 流分流                                                 │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ tokenizer.decode([token_id]) → "text chunk"                          │ │
│  │                                 │                                     │ │
│  │                                 ▼                                     │ │
│  │ CotStreamSplitter.feed("text chunk")                                │ │
│  │  ├─ 檢測標籤: <think>, </think>, <final>, </final>, <|im_end|>     │ │
│  │  ├─ 管理 FSM 狀態                                                    │ │
│  │  └─ 發出事件: (kind, text)                                          │ │
│  │      └─ kind ∈ {reasoning, final, stop}                            │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  輸入：token_id  →  輸出：[(kind, text), ...]                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  第 3 層：中間件編排 (CotMiddleware)                                         │
│  ├─ 追蹤思考 token 數量                                                     │
│  ├─ 監控推理預算                                                            │
│  ├─ 計算動態 close_bias 坡度                                                │
│  └─ 發出 UI 事件                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 CotStreamSplitter — 狀態機詳解

### 狀態圖

```
                         [初始化]
                            │
                 start_in_think=False
                            │
                            ▼
                    ┌─────────────────┐
                    │     head        │  (無標籤或在 <think> 前)
                    └─────────────────┘
                            │
                    [發現 <think>]
                            │
                            ▼
                    ┌─────────────────┐
         ┌─────────│     think       │◄─────────┐
         │         └─────────────────┘          │
         │                 │                    │
         │        [發現 </think>]                │
         │                 │                    │ start_in_think=True
         │                 ▼                    │  (初始化直接進入)
         │         ┌─────────────────┐          │
         │         │    between      │──────────┘
         │         └─────────────────┘
         │                 │
         │        [發現 <final> 或 <|im_end|>]
         │                 │
         ├─────────────────┼─────────────────┐
         │ <final>         │ <|im_end|>      │ [超時/EOS]
         │                 │                  │
         ▼                 ▼                  ▼
    ┌──────────┐   ┌──────────────┐   ┌──────────────┐
    │  final   │   │     done     │   │     done     │
    └──────────┘   └──────────────┘   └──────────────┘
         │
    [發現 </final> 或 <|im_end|>]
         │
         ▼
    ┌──────────────┐
    │     done     │  (終止)
    └──────────────┘
```

### 詳細狀態轉換表

| 當前模式 | 觸發事件 | 下一狀態 | 輸出事件 | 說明 |
|---------|---------|---------|---------|------|
| `head` | 發現 `<think>` | `think` | - | 進入推理塊 |
| `head` | 發現 `<final>` | `final` | `("final", prefix)` | 直接進入答案 |
| `head` | 發現 `<\|im_end\|>` | `done` | `("stop", "")` | 提前結束 |
| `head` | 其他文本 | `head` | `("final", text)` | 無標籤答案 |
| `think` | 發現 `</think>` | `between` | `("reasoning", text)` | 推理結束 |
| `think` | 發現 `<\|im_end\|>` | `done` | `("reasoning", text), ("stop", "")` | 推理中斷 |
| `think` | 其他文本 | `think` | `("reasoning", text)` | 繼續推理 |
| `between` | 發現 `<final>` | `final` | - | 進入答案塊 |
| `between` | 發現 `<\|im_end\|>` | `done` | `("stop", "")` | 提前結束 |
| `between` | 發現 `</final>` | `done` | `("stop", "")` | 異常路徑 |
| `between` | 其他文本 | `between` | `("final", text)` | 過渡文本 |
| `final` | 發現 `</final>` | `done` | `("final", text), ("stop", "")` | 答案結束 |
| `final` | 發現 `<\|im_end\|>` | `done` | `("final", text), ("stop", "")` | 強制結束 |
| `final` | 其他文本 | `final` | `("final", text)` | 繼續答案 |
| `done` | 任何 | `done` | - | 終止 (無操作) |

---

## 📝 Feed 函數運作邏輯

### 輸入：分塊文本 (Streamed Text Chunks)

```python
chunk = tokenizer.decode([token_id])  # e.g., "I need"
events = splitter.feed(chunk)         # → [(kind, text), ...]
```

### 演算法：逐字符掃描 (Greedy Tag Matching)

```
┌────────────────────────────────────────────────────────┐
│ INPUT: chunk = "I need to <think> "                    │
│        buffer was "" → new buffer: "I need to <think> " │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │  尋找所有已知標籤的位置            │
        │  - <think>   @ index 11           │
        │  - </think>  @ -1 (未找到)        │
        │  - <final>   @ -1 (未找到)        │
        │  - </final>  @ -1 (未找到)        │
        │  - <|im_end|> @ -1 (未找到)      │
        └───────────────────────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │  選擇最早的標籤位置                │
        │  → min([11]) = 11 @ "<think>"    │
        └───────────────────────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │  提取標籤前的文本 (如果相關)       │
        │  if mode=="head" && loose_final:   │
        │    emit ("final", "I need to ")   │
        └───────────────────────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │  狀態轉換                         │
        │  buffer := buffer[11+7:]          │
        │         = " "                     │
        │  mode := "think"                  │
        │  progressed := True               │
        └───────────────────────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │  循環直到無進展 (progressed=False) │
        │  或 buffer 無標籤                 │
        └───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ OUTPUT: events = [("final", "I need to ")]             │
│         mode = "think"                                 │
│         buffer = " "                                   │
└────────────────────────────────────────────────────────┘
```

### 核心安全機制：部分標籤緩衝

```python
def _safe_tail_cut(self) -> str:
    """保護部分標籤不被割斷"""
    
    # 例如：buffer = "...text</th"
    #       標籤 </think> 被割成兩個 chunk
    
    buffer = "...text</th"
    
    # 向後檢查：是否有標籤前綴匹配？
    # "</th" 是否匹配任何標籤開始？
    #   - "</think>" 的前 4 字符是 "</th" ✓
    
    # 保留最後 4 字符，發出前面的
    emit = buffer[:-4]        # "...text</"
    keep = buffer[-4:]        # "</th"
    
    return emit, keep
```

---

## 🛡️ FormatGuard — Logit 轉換系統

### 架構

```
┌─────────────────────────────────────────────────────────────┐
│            FormatGuard: Logit Transformation                │
└─────────────────────────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
    ┌──────────────┐          ┌──────────────────┐
    │  Ban Masks   │          │  Close Bias      │
    │  (靜態)      │          │  (動態)          │
    └──────────────┘          └──────────────────┘
          │                             │
          │                             │
    - 全局禁止                 - 依賴於 mode
    - 應用於所有非終止模式      - think: 線性坡度 (0→max)
    - IDs: <|im_start|>        - between/final: 常數值
            </s>
            <|im_end|>

          │                             │
          └──────────────┬──────────────┘
                         ▼
            ┌─────────────────────────┐
            │  apply(logits, mode)    │
            │  ┌───────────────────┐  │
            │  │ logits += ban_mask│  │
            │  └───────────────────┘  │
            │  ┌───────────────────┐  │
            │  │ logits += close   │  │
            │  │   _bias_oneshot   │  │
            │  └───────────────────┘  │
            └─────────────────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │  biased_logits[vocab]   │
            │  (準備採樣)             │
            └─────────────────────────┘
```

### Token ID 解析流程（修復前後對比）

#### ❌ 修復前

```
┌──────────────────────────────────────────────────────────┐
│ tokenizer.vocab_size = 32000 (不完整的報告)              │
└──────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
   ┌──────────────┐            ┌──────────────────┐
   │ ban_ids 檢查 │            │ close_map 檢查   │
   │ IDs: 2       │            │ 尋找 </think>    │
   │ (OK)         │            │                  │
   └──────────────┘            └──────────────────┘
                                        │
                            ┌───────────┘
                            │
                            ▼
                    ┌────────────────┐
                    │ 32003 ≥ 32000? │
                    │ YES → 拒絕 ✗   │
                    └────────────────┘
                            │
                            ▼
                    ┌────────────────┐
                    │ 退回到備選項   │
                    │ try '</'       │
                    │ → ID 829 ✗     │
                    └────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │ close_map = {                 │
            │   "think": 829 ❌             │
            │   "final": 829 ❌             │
            │ }                             │
            │ (應用於錯誤的 token!)         │
            └───────────────────────────────┘
```

#### ✅ 修復後

```
┌──────────────────────────────────────────────────────────┐
│ 檢測實際詞彙大小 via backend_tokenizer.get_vocab()       │
│ actual_vocab_size = 32007 (完整)                         │
└──────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
   ┌──────────────┐            ┌──────────────────┐
   │ ban_ids 檢查 │            │ close_map 檢查   │
   │ IDs: 2,      │            │ 尋找 </think>    │
   │      32000,  │            │                  │
   │      32001   │            │ 32003 < 32007?  │
   │ (OK)         │            │ YES → 接受 ✓    │
   └──────────────┘            │                  │
                               │ close_map = {   │
                               │  "think": 32003 │
                               │  "final": 32005 │
                               │ } ✓             │
                               └──────────────────┘
```

---

## 🔢 Vocab Size 問題深度分析

### Token ID 分佈

```
┌─────────────────────────────────────────────────────────────┐
│ Tokenizer Vocabulary Distribution                           │
│                                                              │
│ Range          │ Purpose              │ Count               │
│ ───────────────┼──────────────────────┼─────────────        │
│ 0 - 31999      │ 基礎詞彙               │ 32000 token        │
│                │ (字、詞、子詞)        │                     │
│                │                      │                     │
│ 32000          │ Special: <|im_start|>│ 1 token             │
│ 32001          │ Special: <|im_end|>  │ 1 token             │
│ 32002          │ CoT: <think>         │ 1 token             │
│ 32003          │ CoT: </think>        │ 1 token  ← 問題!   │
│ 32004          │ CoT: <final>         │ 1 token  ← 問題!   │
│ 32005          │ CoT: </final>        │ 1 token  ← 問題!   │
│ 32006          │ (保留)               │ 1 token             │
│                │                      │ ───────────         │
│ TOTAL:         │                      │ 32007 token         │
└─────────────────────────────────────────────────────────────┘

問題邊界檢查：
  if 0 <= tid < vocab_size:  # 32000
      # 32003 失敗！ (32003 ≥ 32000)
```

### 為什麼 tokenizer.vocab_size 報告不完整？

```
HuggingFace Tokenizer API:
  ├─ tokenizer.vocab_size  
  │  └─ 報告: "標準字典"大小 (32000)
  │     (不包括動態添加的特殊 token)
  │
  └─ tokenizer.backend_tokenizer.get_vocab()
     └─ 報告: "實際"詞彙表 (32007)
        (包括所有特殊 token 和 CoT token)

SFT 訓練時添加的 CoT token:
  special_tokens_dict = {
    "additional_special_tokens": [
      "<think>",    # 作為 special token 添加
      "</think>",   # 而不是從預訓練詞彙中
      "<final>",    # 這導致 ID > vocab_size
      "</final>"
    ]
  }
```

---

## 🎯 Build FormatGuard 流程

### 詳細步驟

```
輸入：tokenizer, vocab_size=32000, config
   │
   ├─ STEP 1: 檢測實際詞彙大小
   │   ├─ 嘗試取得 backend_tokenizer.get_vocab()
   │   ├─ 如果存在，計算 actual_vocab_size = max(values) + 1
   │   └─ 否則，使用傳入的 vocab_size
   │
   ├─ STEP 2: 識別禁止 token (ban_ids)
   │   ├─ 如果 ban_im_start 啟用：
   │   │   ├─ 解析 "<|im_start|>"  → 32000
   │   │   ├─ 解析 "</s>"          → 2
   │   │   └─ 解析 "<|im_end|>"    → 32001
   │   └─ 檢查: 0 <= tid < actual_vocab_size ✓✓✓
   │
   ├─ STEP 3: 識別關閉 token (close_map)
   │   ├─ for mode in {think, between, final}:
   │   │   ├─ 按優先順序嘗試變體
   │   │   │   think:   (</think>, </,  </think)
   │   │   │   between: (<final>,  <final)
   │   │   │   final:   (</final>, </,  <|im_end|>)
   │   │   │
   │   │   └─ 第一個匹配被使用 (break)
   │   │
   │   ├─ think:   </think>  → 32003 ✓
   │   ├─ between: <final>   → 32004 ✓
   │   └─ final:   </final>  → 32005 ✓
   │
   ├─ STEP 4: 構建 MLX mask 陣列
   │   ├─ _ban_mask[actual_vocab_size]
   │   │   └─ 在禁止 ID 處設置 -inf
   │   │
   │   ├─ _think_ban_mask[actual_vocab_size]
   │   │   └─ (目前未使用，全為 0)
   │   │
   │   ├─ _final_ban_mask[actual_vocab_size]
   │   │   └─ 在 </final> 位置設置 -inf
   │   │
   │   └─ _close_one_hot_by_mode[mode]
   │       └─ one_hot[actual_vocab_size] @ close_id
   │
   └─ OUTPUT: FormatGuard 實例
      ├─ vocab_size = 32007 ✓ (實際值)
      ├─ ban_ids = (2, 32000, 32001)
      ├─ close_first_id_by_mode = {
      │    "think": 32003,
      │    "between": 32004,
      │    "final": 32005
      │  }
      └─ 所有 mask 已編譯到 MLX 設備上
```

---

## ⚡ Apply 函數 (逐步執行)

### 執行流程

```
輸入：logits[vocab_size=32007], mode="think", close_bias_scalar=4.5

STEP 1: 檢查是否應用任何轉換
  if not cfg.enabled or mode == "done":
      return logits  (跳過，無轉換)
  
  → mode="think", cfg.enabled=True → 繼續

STEP 2: 應用全局禁止 mask
  if self._ban_mask is not None:
      out = logits + ban_mask.astype(logits.dtype)
      
  效果：
    logits[2]     -= inf  (禁止 </s>)
    logits[32000] -= inf  (禁止 <|im_start|>)
    logits[32001] -= inf  (禁止 <|im_end|>)
    其他 ID 保持不變

STEP 3: 應用模式特定 mask (think)
  if mode == "think" and self._think_ban_mask is not None:
      out = out + think_ban_mask.astype(out.dtype)
  
  → think_ban_mask 為 0 (未使用)，無效果

STEP 4: 應用關閉偏差 (close bias)
  if close_bias_scalar > 0:
      oh = self._close_one_hot_by_mode.get("think")
      if oh is not None:
          out = out + (oh * close_bias_scalar).astype(out.dtype)
  
  one_hot @ 32003 = [0,0,...,1,...,0]  # index 32003 處為 1
  one_hot * 4.5 = [0,0,...,4.5,...,0]
  
  效果：
    logits[32003] += 4.5  (鼓勵 </think>)
    其他 ID 保持不變

輸出：biased_logits[32007]
  ├─ 禁止 token (2, 32000, 32001) 分數 -= inf
  ├─ </think> (32003) 分數 += 4.5 ← 最優先
  └─ 其他 token 分數保持不變
```

### 視覺化：Logits 變換

```
原始 logits (片段):
  [1.2, 0.5, ..., 2.3, ..., 3.1, ..., 1.8, ..., 0.9]
   (0)  (1)       (2)       (32000)(32001)(32003)  (32005)

禁止 </s> @ 2:
  [1.2, 0.5, ..., -inf, ..., 3.1, ..., 1.8, ..., 0.9]

禁止 <|im_start|> @ 32000:
  [1.2, 0.5, ..., -inf, ..., -inf, ..., 1.8, ..., 0.9]

禁止 <|im_end|> @ 32001:
  [1.2, 0.5, ..., -inf, ..., -inf, ..., -inf, ..., 0.9]

加上 close_bias @ 32003 (+4.5):
  [1.2, 0.5, ..., -inf, ..., -inf, ..., -inf, ..., 0.9 + 4.5]
                                                  └─ 5.9 ← 最高!

採樣：
  softmax(biased_logits)
  argmax → token_id = 32003 (</think>)
```

---

## 🧪 Diagnose_cot.py 測試架構

```
┌─────────────────────────────────────────────────────────────┐
│                    diagnose_cot.py                          │
│              完整的單元測試套件                              │
└─────────────────────────────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐
    │TEST: Guard  │  │TEST: Splitter│  │TEST: Middleware │
    │             │  │              │  │                 │
    │✓ Token ID   │  │✓ Mode Trans  │  │✓ Budget Track   │
    │✓ Ban Mask   │  │✓ Tag Parse   │  │✓ Bias Ramp      │
    │✓ Close Bias │  │✓ Buffering   │  │✓ Health Report  │
    └─────────────┘  └──────────────┘  └─────────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  JSON Report    │
                    │  + 詳細日誌      │
                    └─────────────────┘

測試覆蓋：
  1. 詞彙邊界 (32000 vs 32007) ✓
  2. Token 識別 (ban_ids, close_map) ✓
  3. FSM 狀態轉換 (head→think→between→final→done) ✓
  4. 部分標籤安全 (緩衝檢查) ✓
  5. 預算和偏差邏輯 ✓
```

---

## 📊 完整數據流範例

### 推理序列：簡單 "Who are you?" 回應

```
┌──────────────────────────────────────────────────────────────┐
│ INPUT: prompt = "Who are you?"                               │
│        system = "Answer identity with consistency"           │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────┐
│ Prefill: 前綴編碼 + KV 快取初始化                 │
│ prompt_ids = [user_id, ..., ass_id, <think>, \n] │
└─────────────────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    STEP│STEP 1             │STEP 2              │STEP 3...
       │Token 1             │Token 2             │
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐
    │logits @ i=0 │  │logits @ i=1  │  │logits @ i=2     │
    │  ↓          │  │  ↓           │  │  ↓              │
    │apply Guard  │  │apply Guard   │  │apply Guard      │
    │ (mode=think)│  │(mode=think)  │  │(mode=think)     │
    │  ↓          │  │  ↓           │  │  ↓              │
    │sample "I"   │  │sample "am"   │  │sample "a"       │
    │  ↓          │  │  ↓           │  │  ↓              │
    │decode "I"   │  │decode " am"  │  │decode " a"      │
    │  ↓          │  │  ↓           │  │  ↓              │
    │splitter✓    │  │splitter✓     │  │splitter.       │
    │mode=think   │  │mode=think    │  │mode=think      │
    │event:none   │  │event:none    │  │...
    │             │  │              │  │
    └─────────────┘  └──────────────┘  └─────────────────┘
         │                │                     │
         └────────────────┼─────────────────────┘
                          │
                          ▼
    ┌──────────────────────────────────────────┐
    │STEP N: ...thinking...                    │
    │   logits: model(token_n)                 │
    │   ↓                                      │
    │   apply Guard: close_bias ramps up       │
    │   (as think_tokens → reasoning_budget)   │
    │   ↓                                      │
    │   sample "</think>" (32003)              │
    │   decode "</think>"                      │
    │   ↓                                      │
    │   splitter.feed("</think>")              │
    │   ├─ 發現標籤 @ index 0                 │
    │   ├─ emit ("reasoning", "I am...")       │
    │   ├─ mode: think → between               │
    │   └─ progressed=True                     │
    └──────────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────┐
    │STEP N+1: 進入 between 模式                │
    │   logits: model(</think>)                │
    │   ↓                                      │
    │   apply Guard:                           │
    │   ├─ ban_mask (仍然活躍)                │
    │   ├─ close_bias target → <final> (32004)│
    │   ↓                                      │
    │   sample "<final>" (32004)               │
    │   decode "<final>"                       │
    │   ↓                                      │
    │   splitter.feed("<final>")               │
    │   ├─ 發現標籤                           │
    │   ├─ mode: between → final               │
    │   └─ progressed=True                     │
    └──────────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────┐
    │STEP N+2: 進入 final 模式                 │
    │   logits: model(<final>)                 │
    │   ↓                                      │
    │   apply Guard:                           │
    │   ├─ ban_mask (仍然活躍)                │
    │   ├─ close_bias target → </final> (32005)
    │   ├─ final_ban_mask (禁止 0-content)    │
    │   ↓                                      │
    │   sample "\n"                            │
    │   decode "\n"                            │
    │   ↓                                      │
    │   splitter.feed("\n")                    │
    │   ├─ 發現: 無標籤                       │
    │   ├─ emit ("final", "\n")                │
    │   └─ mode: final                         │
    └──────────────────────────────────────────┘
         │
    ┌────┴───────────────────────────────┐
    │  迴圈: sample → emit "final" event   │
    │  直到看到 </final> 或 <|im_end|>    │
    │  (最多 remaining budget tokens)     │
    └────┬───────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────┐
    │STEP N+K: 採樣 "</final>" 或 "<|im_end|>"│
    │   logits: model(...)                     │
    │   ↓                                      │
    │   sample "</final>" (32005)              │
    │   decode "</final>"                      │
    │   ↓                                      │
    │   splitter.feed("</final>")              │
    │   ├─ 發現標籤                           │
    │   ├─ emit ("final", "...")               │
    │   ├─ emit ("stop", "")                   │
    │   ├─ mode: final → done                  │
    │   └─ progressed=True                     │
    └──────────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────┐
    │ OUTPUT: done=True, 生成結束               │
    │                                          │
    │ 事件序列:                                │
    │  ✓ ("reasoning", "I am Mamba...")        │
    │  ✓ ("final", "\nMamba is a hybrid...")   │
    │  ✓ ("stop", "")                         │
    └──────────────────────────────────────────┘
```

---

## 🔍 關鍵洞察

### 為什麼這個設計有效？

1. **分層責任**
   - FormatGuard: 低層 logit 操作 (常數時間)
   - CotStreamSplitter: 純文本 FSM (無依賴)
   - CotMiddleware: 高層編排 (預算、坡度)

2. **零分配設計**
   - Ban mask 和 close_bias one_hot 預編譯
   - 每個 decode 步驟只需 1 個向量加法
   - 無需重新編譯或重新分配

3. **標籤邊界安全**
   - Buffer 保留部分標籤 (_safe_tail_cut)
   - 跨 chunk 標籤安全處理
   - 無 regex 開銷

4. **詞彙邊界修復**
   - 使用實際的後端詞彙大小 (32007)
   - 不依賴可能不完整的 tokenizer.vocab_size
   - 向後兼容（無 API 變更）

---

## 📋 快速參考表

### Token ID 映射表

| Token | ID | 用途 | 狀態 |
|-------|----|----|------|
| `<\|im_start\|>` | 32000 | 用戶/系統開始 | 禁止 |
| `<\|im_end\|>` | 32001 | 回應結束 | 禁止 + 停止 |
| `<think>` | 32002 | 推理開始 | 普通 |
| `</think>` | 32003 | 推理結束 | **關閉目標** (think) |
| `<final>` | 32004 | 答案開始 | **關閉目標** (between) |
| `</final>` | 32005 | 答案結束 | **關閉目標** (final) |
| `</s>` | 2 | 結束符 | 禁止 |

### FSM 模式速查

| 模式 | 含義 | 輸入來源 | 輸出事件 |
|------|------|---------|---------|
| `head` | 無標籤區域 | 起始或 `<final>` 前 | `final` (如有文本) |
| `think` | 推理區域 | 在 `<think>…</think>` 中 | `reasoning` |
| `between` | 過渡區域 | 在 `</think>…<final>` 中 | `final` (如有文本) |
| `final` | 答案區域 | 在 `<final>…</final>` 中 | `final` |
| `done` | 終止 | 任何終止信號 | 無 |

---

## ✅ 驗證檢查清單

- [x] 詞彙邊界正確 (32007 not 32000)
- [x] 所有 CoT token 可識別
- [x] Ban mask 適用於所有禁止 ID
- [x] Close bias 應用於正確目標
- [x] FSM 狀態轉換完整
- [x] 部分標籤安全
- [x] 診斷測試全過

---

**這就是 CoT Format FSM 的完整工作原理！**
