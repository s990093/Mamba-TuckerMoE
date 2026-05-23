# SFT 訓練格式、System Prompt 與 Mask 說明

> 給協作者的技術文件：說明每筆資料在訓練時如何被組裝成 ChatML、哪些 token 被 mask、以及三種 System Prompt 的分配邏輯。

---

## 1. 整體流程概覽

```
JSON 資料 (emotion.json / self_awareness.json / email_summary.json / movie_intro.json / noise.json / system_call.json / deep_dive.json)
       │
       ▼
  stf_cot_to_bin.py     ← 組裝 ChatML + 插入 System Prompt
       │
       ├──► HF Dataset (stf_cot_hf/)     ← 文字格式，可抽查
       └──► uint16 .bin (stf_cot_train.bin) ← token id 串流
              │
              ▼
        train_sft.py / train_sft_cot.py   ← 訓練，內部做 mask
```

每筆 JSON 資料經過以下步驟變成訓練樣本：

1. **讀取** `input`、`cot`、`output`、`category` 欄位
2. **依 `category` 決定 System Prompt**（三選一）
3. **組裝成 ChatML 格式**的完整字串
4. **Tokenize** 成 token id 序列
5. **建立 label mask**：只對 assistant 回覆區間計算 loss

---

## 2. ChatML 格式（每筆資料的最終結構）

每筆訓練資料最終被組裝成以下 ChatML 結構：

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
<think>
{cot}
</think>
<final>
{output}
</final><|im_end|>
```

### 各區塊說明

| 區塊 | 內容來源 | 是否被 mask |
|------|----------|-------------|
| `system` | 由 category 自動決定（見 Section 3） | **是（mask = -100，不計算 loss）** |
| `user` | JSON 的 `input` 欄位 | **是（mask = -100，不計算 loss）** |
| `assistant` | JSON 的 `cot` + `output`，包裹在 `<think>`/`<final>` 中 | **否（計算 loss）** |

### 具體範例

以 `emo_0001`（category: `burnout`）為例，最終 ChatML 為：

```
<|im_start|>system
You are a helpful conversational assistant. Reply naturally, acknowledge the user's intent, and give practical advice.<|im_end|>
<|im_start|>user
I have been coding for 14 hours straight and I cannot think anymore. My brain just refuses to function.<|im_end|>
<|im_start|>assistant
<think>
Step 1: **Identify user state** — Severe cognitive depletion from sustained high-intensity computation without thermal cooldown.
Step 2: **Reject motivation strategy** — Do not push further output. The biological hardware has reached its thermal limit.
Step 3: **Frame diminishing returns** — Continued work past the exhaustion threshold causes negative net output.
Step 4: **Prescribe action** — Immediate forced shutdown of cognitive processes.
</think>
<final>
Your **neural architecture** has hit thermal throttling. Continued execution past this point yields **negative returns** — you are now generating bugs faster than solutions. Initiate a hard shutdown. Sleep is the only defragmentation protocol your biological hardware supports.
</final><|im_end|>
```

---

## 3. 三種 System Prompt

系統有 **八種** 訓練用 System Prompt bucket（外加舊版 `dialogue` / `task` / `summary` 相容別名），依據每筆資料的 `category` 或來源檔案自動分配。

### 3.1 Bucket 定義

| Bucket | System Prompt 全文 |
|--------|-------------------|
| **dialogue** | `You are a helpful conversational assistant. Reply naturally, acknowledge the user's intent, and give practical advice.` |
| **task** | `You are a task-oriented assistant. Follow instructions and output concise, directly usable results.` |
| **summary** | `You are a summarization assistant. Start with a brief conclusion, then add key reasons clearly and concisely.` |

### 3.2 Category → Bucket 對照表

以下是所有 `category` 值對應到哪個 bucket（system prompt）：

#### Emotion 類（全部 → `dialogue`）

| category | bucket |
|----------|--------|
| `burnout` | dialogue |
| `self_doubt` | dialogue |
| `loneliness` | dialogue |
| `rejection` | dialogue |
| `social_conflict` | dialogue |
| `existential_crisis` | dialogue |
| `anxiety` | dialogue |
| `anger` | dialogue |
| `grief` | dialogue |
| `perfectionism` | dialogue |

#### Self-Awareness 類（全部 → `dialogue`）

| category | bucket |
|----------|--------|
| `core_identity` | dialogue |
| `architecture` | dialogue |
| `hardware_awareness` | dialogue |
| `relationship_role` | dialogue |
| `existential_bounds` | dialogue |
| `capability_limits` | dialogue |
| `emotional_simulation` | dialogue |
| `upgrade_and_training` | dialogue |

#### Email & Summary 類

| category | bucket |
|----------|--------|
| `email_draft` | **task** |
| `email_reply` | **task** |
| `email_tone_adjust` | **task** |
| `meeting_summary` | **summary** |
| `document_summary` | **summary** |
| `task_extraction` | **summary** |
| `bullet_point` | **summary** |
| `priority_triage` | **summary** |
| `academic_email` | **task** |

#### Movie Intro 類（獨立檔案 `movie_intro.json`）

| category | bucket |
|----------|--------|
| `plot_overview` | **summary** |
| `character_analysis` | dialogue |
| `theme_deconstruction` | dialogue |
| `technical_craft` | dialogue |
| `comparative_analysis` | **summary** |
| `recommendation_filter` | **task** |
| `trivia_context` | **summary** |

> Movie Intro 是獨立的第四類別（見 GUIDE.md Section 7），目標 2,000 筆。ID 前綴為 `mov_`。

#### Daily Conversation 類（獨立檔案 `noise.json`）

| category | bucket |
|----------|--------|
| `tech_troubleshoot` | **task** |
| `learning_strategy` | dialogue |
| `time_management` | **task** |
| `writing_assist` | **task** |
| `culinary_science` | dialogue |
| `fitness_systems` | dialogue |
| `finance_logic` | **task** |
| `travel_logistics` | **task** |
| `general_knowledge` | dialogue |
| `creative_problem` | dialogue |

> Daily Conversation 是獨立的第五類別（見 GUIDE.md Section 8），目標 5,000 筆。ID 前綴為 `gen_`。

#### Math Drill 類（獨立檔案 `math_drill.json`）

| category | bucket |
|----------|--------|
| `arith_add_units` | **math_drill** |
| `arith_add_mixed` | **math_drill** |
| `arith_mul_table` | **math_drill** |
| `arith_mul_teens` | **math_drill** |
| `arith_mul_extended` | **math_drill** |
| `arith_mul_hundred` | **math_drill** |

> Math Drill 是獨立的第六類別（見 GUIDE.md §8.5、`MATH_DRILL.md`），目標 **≤200** 筆（稀疏抽樣）。ID 前綴為 `mat_`。
>
> **CoT 禁止** `Step 1:` / `Parse operands` 等編譯器日誌體。JSON `output` 僅數字；腳本包成 `<final>{n}</final>`。

#### System Call 類（獨立檔案 `system_call.json`）

| category | bucket |
|----------|--------|
| `tool_trigger` | **task** |
| `tool_response` | **task** |

> System Call 是獨立的第六類別（見 GUIDE.md Section 9），目標 600 筆。ID 前綴為 `sys_`。
> 
> **特殊說明**：System Call 的行為模式與其他類別根本不同——`tool_trigger` 的 output 是固定格式的控制 token `[CALL: xxx]`，而非自然語言。`tool_response` 的 input 以 `[SYSTEM_RESULT: xxx]` 開頭，是前端注入的系統數據。兩個子分類都歸入 `task` bucket，因為它們本質上是「執行指令→產出結果」的任務導向行為。

#### Deep Dive 類（獨立檔案 `deep_dive.json`）

| category | bucket |
|----------|--------|
| `deep_diagnostic` | dialogue |
| `system_report` | dialogue |
| `comprehensive_analysis` | **summary** |
| `strategy_planning` | **task** |

> Deep Dive 是獨立的第七類別（見 GUIDE.md Section 10），輸出可佔滿 2048 tokens。ID 前綴為 `dd_`。

> **重要**：如果某個 `category` 不在對照表中，預設 fallback 到 `dialogue`。

### 3.3 為什麼只有三種？

1. **模型容量有限** — Mamba 是輕量 edge 模型（vocab 32,007），太多種 system prompt 會稀釋每種的訓練信號
2. **三種覆蓋主要互動模式** — 對話（情緒 + 自我認知）、任務執行（寫信）、總結（摘要）
3. **System prompt 被 mask 掉** — 模型不需要「生成」system prompt，只需要理解它作為前置條件的語義影響

### 3.4 需要更新 `stf_cot_sysprompt.py`

目前 `sft_cot_bundle/scripts/stf_cot_sysprompt.py` 的 `DEFAULT_STF_CATEGORY_TO_BUCKET` 尚未包含本次新增的子分類。在資料轉換前，必須更新該對照表，加入 Emotion、Self-Awareness、Email & Summary 的所有子分類。

---

## 4. Mask 機制（核心！）

### 4.1 什麼是 Mask？

SFT 訓練的目標是**讓模型學會生成 assistant 的回覆**，而不是學會生成 system prompt 或使用者的輸入。因此我們對每筆 token 序列建立一個 `labels` 陣列：

- `labels[i] = ids[i]`：這個 token **參與 loss 計算**（模型需要學會預測它）
- `labels[i] = -100`：這個 token **被 mask**（忽略，不計算 loss）

### 4.2 Mask 規則

```
 被 mask（-100）                          計算 loss
 ◄──────────────────────────────────────► ◄─────────────────────────────────────────────────►
┌──────────────────────────────────────┐ ┌─────────────────────────────────────────────────┐
│<|im_start|>system                    │ │<think>                                          │
│{system_prompt}<|im_end|>             │ │{cot}                                            │
│<|im_start|>user                      │ │</think>                                         │
│{input}<|im_end|>                     │ │<final>                                          │
│<|im_start|>assistant\n               │ │{output}                                         │
│                                      │ │</final><|im_end|>                               │
└──────────────────────────────────────┘ └─────────────────────────────────────────────────┘
```

具體而言，`train_sft._build_xy_masked` 的邏輯：

1. **掃描** token 序列，找到 `<|im_start|>assistant\n` 的 token pattern
2. **從該 pattern 結束位置開始**，將後續所有 token 的 label 設為 `ids[i]`（計算 loss）
3. **直到遇到** `<|im_end|>` 結尾序列（含該序列本身），然後停止
4. 若有多輪對話，會重複 step 1-3 找到所有 assistant 區間

### 4.3 哪些 Token 被監督？

| Token 區間 | 是否計算 loss | 說明 |
|-----------|:---:|------|
| `<|im_start|>system\n...` | ❌ | System prompt 是前置條件，不需要模型生成 |
| `<|im_end|>` (system 結尾) | ❌ | 同上 |
| `<|im_start|>user\n...` | ❌ | 使用者輸入不需要模型生成 |
| `<|im_end|>` (user 結尾) | ❌ | 同上 |
| `<|im_start|>assistant\n` | ❌ | Header 本身不計算（是固定格式） |
| `<think>\n{cot}\n</think>` | ✅ | **模型要學會推理過程** |
| `<final>\n{output}\n</final>` | ✅ | **模型要學會最終輸出** |
| `<|im_end|>` (assistant 結尾) | ✅ | **模型要學會在何時停止生成** |

### 4.4 為什麼 CoT 也計算 loss？

- CoT（`<think>...</think>`）是模型的**內部推理**，訓練時需要讓模型學會這個推理過程
- 這就是為什麼 CoT 的品質非常重要——寫得好的 CoT 直接影響推理品質
- 在推理（inference）時，模型會先生成 `<think>` 區塊，再生成 `<final>` 區塊
- 前端可以選擇只顯示 `<final>` 的內容給使用者

### 4.5 Token 序列示意圖（含 label）

```
位置:     0    1    2    3   ...  N   N+1  N+2  ...  M   M+1  ...  K
token:   <|im  sys  tem  \n  ... end  <|im user  \n  ... end  <|im asst  \n  <think> Step... </think> <final> Your... </final> <|im_end|>
label:   -100 -100 -100 -100... -100 -100 -100 -100... -100  -100 -100  -100   ✅     ✅      ✅       ✅      ✅       ✅        ✅
                                                                    ▲                                                           ▲
                                                         header 結束位置                                                   最後一個監督位置
```

### 4.6 x / y 對齊

訓練時每筆資料被切成 `(x, y)` pair：

- `x = ids[0 : seq_len]`（輸入）
- `y = labels[1 : seq_len+1]`（目標，右移一位）

這意味著 `x[k]` 的目標是預測 `y[k] = ids[k+1]`（next token prediction）。被 mask 的位置 `y[k] = -100`，PyTorch 的 `CrossEntropyLoss(ignore_index=-100)` 會自動跳過。

---

## 5. 特殊 Token 一覽

| Index | Token | 用途 |
|-------|-------|------|
| 0 | `<\|im_start\|>` | 每個角色區塊的開頭 |
| 1 | `<\|im_end\|>` | 每個角色區塊的結尾 |
| 2 | `<think>` | CoT 推理區塊開始 |
| 3 | `</think>` | CoT 推理區塊結束 |
| 4 | `<final>` | 最終輸出區塊開始 |
| 5 | `</final>` | 最終輸出區塊結束 |
| 6 | `[PAD]` | 填充符號（不寫入對話內文） |

> 這 7 個 special token 加上 LLaMA 基底的 32,000 = 總詞表大小 **32,007**。

---

## 6. 長度限制與截斷

| 參數 | 值 | 說明 |
|------|-----|------|
| `model_max_length` | 2048 tokens | tokenizer 設定的上限 |
| `SEQ_LEN`（訓練用） | 1024 tokens | 預設訓練序列長度 |
| 單筆建議上限 | ~512 tokens | `input + cot + output` 的目標上限 |

- 若單筆 token 數超過 `SEQ_LEN + 1`，**尾部會被截斷**
- 截斷可能切掉 `</final>` 和 `<|im_end|>`，導致模型學不到「何時停止」
- 因此 **嚴格控制每筆資料長度** 是資料品質的重要指標
- System prompt 大約占 15~25 token，撰寫資料時要預留這個空間

---

## 7. 生成資料時的注意事項

### 7.1 你不需要寫 System Prompt

撰寫 JSON 時**只需要寫** `id`、`category`、`input`、`cot`、`output`。System prompt 由 `stf_cot_to_bin.py` 根據 `category` 自動插入。

### 7.2 你不需要寫 Special Token

JSON 中的 `cot` 和 `output` 是**純文字**，不需要包含 `<think>`、`</think>`、`<final>`、`</final>`。這些由轉換腳本自動包裹。

### 7.3 Category 值必須精確

`category` 欄位的值決定了使用哪個 system prompt。如果拼錯（例如 `"bornout"` 而非 `"burnout"`），會 fallback 到 `dialogue` bucket——雖然不會報錯，但可能不是你預期的 system prompt。

### 7.4 CoT 換行用 `\n`

在 JSON string 中，`cot` 欄位的每個 Step 之間用 `\n` 分隔。這個 `\n` 在 ChatML 中會變成真正的換行，讓 `<think>` 區塊內的推理步驟分行排列。

---

## 8. 驗證工具

資料轉換後，可用以下工具抽查：

```bash
# 1. 轉換 JSON → ChatML + bin
python3 sft_cot_bundle/scripts/stf_cot_to_bin.py --src dataset/stf.json

# 2. 抽查某筆的 ChatML 文字
python3 sft_cot_bundle/scripts/spot_check_stf_cot.py

# 3. 驗證 mask 正確性（確認 assistant 區間被正確標記）
python3 sft_cot_bundle/scripts/verify_stf_cot_mask.py --sample-index 0 --seq-len 512
```

`verify_stf_cot_mask.py` 的輸出會顯示：

- 哪些位置被標記為 supervised（計算 loss）
- `x[k]` 和 `y[k]` 的 decode 對照（確認 next-token 對齊正確）
- supervised token 總數

---

## 9. 完整資料流圖

```
                    ┌─────────────────┐
                    │  emotion.json   │
                    │  (5,000 筆)     │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │ self_awareness  │
                    │   .json         │
                    │  (5,000 筆)     │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │ email_summary   │
                    │   .json         │
                    │  (5,000 筆)     │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │ deep_dive.json  │
                    │  (700 筆)       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   合併為        │
                    │   stf.json      │
                    │  (15,700 筆)    │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │  stf_cot_to_bin.py           │
              │  1. 讀取 category            │
              │  2. 對照 bucket              │
              │  3. 插入 system prompt       │
              │  4. 包裹 <think>/<final>     │
              │  5. 組裝 ChatML              │
              │  6. Tokenize                 │
              └──────┬──────────────┬────────┘
                     │              │
          ┌──────────▼───┐   ┌─────▼──────────┐
          │ stf_cot_hf/  │   │ stf_cot_train  │
          │ (HF Dataset) │   │    .bin         │
          │ 可讀文字     │   │ (uint16 ids)   │
          └──────────────┘   └───────┬─────────┘
                                     │
                          ┌──────────▼──────────┐
                          │ train_sft_cot.py     │
                          │                      │
                          │ _build_xy_masked:    │
                          │ ┌──────────────────┐ │
                          │ │ system → -100    │ │
                          │ │ user   → -100    │ │
                          │ │ assistant → loss │ │
                          │ └──────────────────┘ │
                          │                      │
                          │ CrossEntropyLoss     │
                          │ (ignore_index=-100)  │
                          └──────────────────────┘
```

---

## 10. FAQ

**Q: 為什麼 system prompt 這麼短？**
A: 因為 Mamba 是 edge 模型，context window 有限（2048 tokens）。長 system prompt 會擠壓 user input 和 assistant output 的空間。三種 prompt 各約 15~25 tokens，已足夠引導模型的回覆風格。

**Q: 可以自訂 system prompt 嗎？**
A: 可以，在執行 `stf_cot_to_bin.py` 時傳入 `--sys-prompts-json` 參數，指向一個包含 `{"dialogue": "...", "task": "...", "summary": "..."}` 的 JSON 檔案即可覆寫預設值。

**Q: 如果我不想要 system prompt 呢？**
A: 加上 `--no-system` 參數，ChatML 中就不會包含 system 區塊。但不建議這樣做，因為 system prompt 能有效引導模型行為。

**Q: Mask 會影響 CoT 中的 Markdown 嗎？**
A: 不會。Mask 是在 token id 層級操作的，只區分「assistant 區間」和「非 assistant 區間」。CoT 中的 Markdown 語法（如 `**粗體**`）會被正常 tokenize 並被完整監督。

**Q: `<think>` 和 `<final>` 是必要的嗎？**
A: 是的。這兩個 special token 讓模型在推理時知道「先思考、後輸出」的結構。前端可依據這兩個標記分別處理：隱藏思考過程、只顯示最終回覆，或兩者都顯示。
