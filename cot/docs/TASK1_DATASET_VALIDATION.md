# Task 1：資料集與訓練前驗證規格

> **單一權威**：在實作任何客製 `compute_loss`、權重排程或離線權重（IPS 等）之前，訓練資料與格式契約須先滿足本文件。  
> **上層規範**：[`cot_dataset/GUIDE.md`](../../cot_dataset/GUIDE.md)（撰寫規則）、[`cot_dataset/SFT_FORMAT.md`](../../cot_dataset/SFT_FORMAT.md)（組裝與 mask）。本文件不重複全文，只對齊章節並收斂為可執行的驗證清單。

---

## 1. 範圍與非範圍

### 1.1 在範圍內（Task 1 必須涵蓋）

- 各 JSON 檔之 **schema**、**欄位**、**內容規則**（含 `cot` / `output` 結構、Markdown 約束）。
- **英文品質**、**風格禁令**、**類別配額與 id 規則**（見 GUIDE）。
- **Token 長度預算**（協作者自查或抽樣估算；自動化腳本屬 Task 1b，非本文件前提）。
- **預處理後** 與訓練管線的 **label / mask 契約**（概念與對照 SFT_FORMAT；實際位元組層驗證可用 SFT_FORMAT §8 所列腳本抽查）。

### 1.2 不在範圍內（Task 2+）

- `compute_loss` 內 Focal / SCALe / FCP、IPS 離線權重、SFT-GO 分組、Step-DPO 樣本構造等。  
- 將 RE 命中位置映射到 `char_to_token` / `loss_weight_mask` 的實作與數值實驗。  

→ 見 [`TASK2_LOSS_ENGINEERING.md`](TASK2_LOSS_ENGINEERING.md)。

---

## 2. 欄位與 Schema 完整性

對照 **GUIDE §3**。

| 檢查項 | 規則摘要 | 失敗範例 |
|--------|----------|----------|
| 必填欄位 | 每筆含 `id`, `category`, `input`, `cot`, `output` | 缺欄 |
| `history` | 選填；第一版可 `[]` 或省略 | — |
| `id` 格式 | `{類別縮寫}_{四位數}`，與 GUIDE 表格一致（`emo_`…`dd_`） | `emo_1`、`emotion_0001` |
| `id` 唯一 | 全庫不重複 | 重複 id |
| 檔案頂層 | 合法 **JSON array**：`[ {...}, ... ]` | 單物件無 array |
| JSON 合法性 | 可通過 `python -m json.tool <file>`（GUIDE §19） | 語法錯誤 |
| Special token | **禁止**在 JSON 內手寫 `<think>`、`</think>`、`<final>`、`</final>`、`<|im_start|>`、`<|im_end|>` 等（GUIDE「預處理自動包裝」） | 雙重包裝污染 |

---

## 3. CoT 與 Output 結構規範

對照 **GUIDE §3**（轉義）、**§18**（品質清單內與 `cot`/`output` 相關條）、各類別專節（如 Deep Dive §10）。

| 檢查項 | 規則摘要 |
|--------|----------|
| `cot` 步驟數 | 一般 **3~5** 步；Deep Dive 依 GUIDE **5~7** 步 |
| 步驟前綴 | 每步以 **`Step N:`** 開頭（`N` 正整數），以 `\n` 分行 |
| 第一步語意 | 第一步為分析使用者意圖／情緒／請求類型（GUIDE §18） |
| 粗體標記 | 每步使用 `**粗體**` 標記核心判斷（如 `Step 1: **Identify context** — ...`） |
| Deep Dive | `input` 觸發關鍵字、`cot` 含規定步驟、`output` 四段式等（GUIDE §10、§18） |
| Markdown | Email/Summary 等需結構化排版處依 GUIDE；禁止 H1/H2、圖片、連結、HTML、**大型表格**（> 5 欄 × 10 列）等（GUIDE §18） |
| JSON 內換行 | 使用 `\n`，勿在 string 內直接換行（GUIDE §3「Markdown 轉義提醒」） |
| `output` | 全英文、類別字數／風格依 GUIDE；System Call 觸發格式等依 §9 / §18 |

### 3.1 選讀：RE 輔助（非取代 GUIDE）

Task 1 **通過與否**以 GUIDE 文字規則為準。下列正則僅供人工掃描或未來腳本參考（與舊 `step/task1.md` 想法對齊）：

| 用途 | 建議模式（Python raw string） |
|------|--------------------------------|
| 步驟導引 | `r"Step\s*\d+[:：]?"` |
| 表格豎線 | `r"\|"` |
| 粗體片段 | 以 `**` 成對為準，需人工或輔助邏輯檢查 |

若 RE 與 GUIDE 衝突，**以 GUIDE 為準**。

---

## 4. 類別與 System Prompt bucket 一致性

對照 **GUIDE §2、各類別章節** 與 **SFT_FORMAT §3**（`category` → `dialogue` / `task` / `summary`）。

| 檢查項 | 規則摘要 |
|--------|----------|
| `category` 白名單 | 必須為 GUIDE 中該檔案所列**子分類之一** |
| Bucket 對照 | 須存在於 **SFT_FORMAT §3.2** 對照表，才能對應到預期 system prompt |

**重要（SFT_FORMAT §3.2 末段、§7.3）**：若 `category` 拼錯或不在表內，管線可能 **fallback 到 `dialogue`** 而不報錯。Task 1 應將「不在 GUIDE + SFT_FORMAT 所定義集合內的 `category`」視為 **驗證失敗**（或至少 **必須修正的警告**），避免語意與預期 bucket 不一致。

---

## 5. 內容與英文品質

對照 **GUIDE §17**（英文品質）、**§18**（清單中含拼字、縮寫、標點、禁止語氣）、**§19**（退件標準）。

| 檢查項 | 規則摘要 |
|--------|----------|
| 拼字／文法 | 零拼字錯誤；主動一致、冠詞、時態等（§17 表） |
| 縮寫 | Mamba 風格：**不用** `don't` 等，改用 `do not`（§17–§18） |
| 禁止語氣 | 無雞湯、無指定模糊語（§18）；**禁止語句**與**轉化**見 GUIDE **附錄 B / D**（驗證時對照原文，本處不重貼） |
| emoji | 依 §18（含 Priority Triage 例外說明） |

---

## 6. Token 與長度預算

對照 **GUIDE §14**（依類別之總 token 預算）、**§18**（字數與 token 勾選項）。

| 類別情境 | 總 token 預算（摘要） |
|----------|------------------------|
| Emotion / Self-Awareness（常規） | ≤ **512** |
| Daily Conversation | ≤ **512** |
| System Call — trigger | ≤ **256** |
| System Call — response | ≤ **512** |
| Movie Intro | ≤ **768** |
| Email & Summary（常規） | ≤ **768** |
| Deep Dive | ≤ **2048** |

**實務**：Task 1 可要求協作者以專案 tokenizer（**vocab 32,007**、與訓練相同版本，GUIDE §14）抽樣估算；若超過預算或逼近 **SFT_FORMAT §6** 所述截斷風險，視為未通過長度驗證。

---

## 7. 預處理後序列與 Label／Mask 契約

對照 **SFT_FORMAT §1–§2、§4**（以 repo 內實作為準；文件名見 SFT_FORMAT：`stf_cot_to_bin.py`、`train_sft` / `_build_xy_masked`）。

### 7.1 概念規則（預設管線）

| 區段 | `labels` / loss |
|------|-----------------|
| `system` 全文 | **不計算 loss**（`-100`） |
| `user` 全文 | **不計算 loss**（`-100`） |
| `<|im_start|>assistant` header 等固定前綴 | 依 SFT_FORMAT §4.2–§4.3（**不**監督生成該前綴本身） |
| `<think>` … `</think>`（含其中 `cot`） | **計算 loss** |
| `<final>` … `</final>`（含 `output`） | **計算 loss** |
| assistant 區段結尾標記（見 SFT_FORMAT §4.3 表，含對話結束序列） | **計算 loss**（模型須學會適時結束） |

### 7.2 Task 1 要確認的命題

- **預設 SFT**：在「未改 Trainer」前提下，`cot` / `output` 內容（含 Step、Markdown）位於 **assistant 監督區間**，不應被全域 mask 成 `-100`。  
- **若日後加入** 區段加權或二次 mask：不得破壞「user/system 忽略、assistant 內容可學」與 SFT_FORMAT 的一致性；細節留待 Task 2 設計。

### 7.3 抽查工具（可選但建議）

見 **SFT_FORMAT §8**（`verify_stf_cot_mask.py` 等）。通過 Task 1 並進入訓練前，至少對合併後資料跑一輪 GUIDE／SFT_FORMAT 建議的轉換與 mask 抽查。

---

## 8. 品質檢查清單與提交命令

### 8.1 驗證執行清單（對齊 GUIDE §18）

將下列項目逐筆或抽樣勾選（原文細節以 GUIDE §18 為準）：

- [ ] `id` 格式正確且不重複  
- [ ] `category` 為定義內子分類，且落在 SFT_FORMAT §3.2 對照表  
- [ ] `input` 自然英文口語  
- [ ] `cot` 步驟數與 `Step N:` 格式符合類別要求  
- [ ] Deep Dive 專條（若適用）  
- [ ] `output` 風格、字數、System Call 格式等符合類別  
- [ ] 無不當 emoji／模糊語／禁止語句（附錄 B/D）  
- [ ] Markdown 與 JSON 轉義正確  
- [ ] 未手寫 special token  
- [ ] 拼字／文法／標點／縮寫政策  
- [ ] Token 預算與 JSON 語法  

### 8.2 提交前命令（對齊 GUIDE §19）

以下為 GUIDE 所列之**最低限度**人工步驟（檔名請替換為實際檔案）：

```bash
# 1. 驗證 JSON 格式
python -m json.tool emotion.json > /dev/null

# 2. 檢查拼字（需安裝 aspell 或其他工具）
cat emotion.json | aspell list | sort -u

# 3. 檢查常見縮寫（Mamba 不用縮寫）
grep -i "dont\|wont\|cant\|im \|youre\|theyre" emotion.json
```

---

## 9. Definition of Done（Task 1 通過準則）

在進入 Task 2（Loss 工程）或大規模訓練前，建議同時滿足：

1. **Schema**：§2 表格無任一失敗；所有目標檔均為合法 JSON array。  
2. **內容與結構**：§3–§5 符合 GUIDE；§8.1 清單對該次交付批次全部勾選（或依團隊定義之抽樣比例全過）。  
3. **長度**：§6 預算無系統性超標；Deep Dive 等長序列已單獨複查截斷風險（SFT_FORMAT §6）。  
4. **Mask 契約**：至少依 SFT_FORMAT §8 完成一次轉換後 mask 抽查，結果與 §7 一致。  
5. **提交命令**：§8.2 相關檢查無未解釋錯誤（grep 命中需改寫至符合風格）。

---

## 10. 與後續 Task 的接口

- **Task 2**：在 Task 1 全綠後，才導入 RE → token 索引、`loss_weight_mask`、IPS / SFT-GO / 排程 loss 等；見 [`TASK2_LOSS_ENGINEERING.md`](TASK2_LOSS_ENGINEERING.md)。  
- **研究筆記**：`cot/1.md`–`3.md` 不納入 Task 1 必讀；實作 Task 2 時與該目錄及訓練程式對齊即可。

---

## 文件修訂

| 版本 | 說明 |
|------|------|
| 1.0 | 初版；章節對齊 GUIDE §3、§14、§17–§19 與 SFT_FORMAT §1–§4、§6–§8。 |
