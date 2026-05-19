# Task 2+：Loss 工程、參數與 RE／Token 標註規格

> 本文件說明 **Task 1 完成後** 的 Loss 加權、資料預處理標註與訓練整合；與 [`TASK1_DATASET_VALIDATION.md`](TASK1_DATASET_VALIDATION.md) 分離。  
> **權威**：資料格式／ChatML／預設 mask 以 [`cot_dataset/SFT_FORMAT.md`](../../cot_dataset/SFT_FORMAT.md) 為準；本 repo SFT 迴圈預設常數以 [`sft_cot_bundle/scripts/train_sft.py`](../../sft_cot_bundle/scripts/train_sft.py) 內 `validate_config` 為準（以下表格數值摘自該檔，若程式修改請以程式為準）。

---

## 目錄

1. [本 repo SFT 訓練相關常數與路徑](#1-本-repo-sft-訓練相關常數與路徑)
2. [資料管線與 mask（ASCII）](#2-資料管線與-maskascii)
3. [Next-token 對齊（x／y）](#3-next-token-對齊xy)
4. [資料預處理：RE 與 token 級標記（詳細規格）](#4-資料預處理re-與-token-級標記詳細規格)
5. [各方案計算時機與建議參數表](#5-各方案計算時機與建議參數表)
6. [訓練時 loss 整合（概念與 ASCII）](#6-訓練時-loss-整合概念與-ascii)
7. [IPS／SFT-GO 階段規劃與組合公式](#7-ipssft-go-階段規劃與組合公式)
8. [架構示意：FCP、SCALe、SFT-GO、Focal](#8-架構示意fcpscale-sft-go-focal)
9. [延伸閱讀](#9-延伸閱讀)
10. [文件修訂](#10-文件修訂)

---

## 1. 本 repo SFT 訓練相關常數與路徑

### 1.1 `train_sft.py` → `validate_config` 預設（摘要）

| 符號／鍵                          | 預設值                                             | 說明                                                                   |
| --------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------- |
| `VOCAB_SIZE`                      | `32007`                                            | 與 GUIDE／SFT_FORMAT 一致（7 個 special + 32000）                      |
| `SEQ_LEN`                         | `512`                                              | 單步訓練序列長度（`x` 長度）；註解建議可試 `1024`／`2048` 減少尾端截斷 |
| `BATCH_SIZE`                      | `4`                                                | 每裝置 micro-batch                                                     |
| `GRADIENT_ACCUMULATION_STEPS`     | `8`                                                | 梯度累積步數                                                           |
| **有效 batch（概念）**            | `BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS × GPU數` | 由 Accelerate 管理多 GPU                                               |
| `LR`                              | `1e-5`                                             | 學習率                                                                 |
| `WARMUP`                          | `200`                                              | warmup steps 上限之一                                                  |
| `WARMUP_FRAC`                     | `0.08`                                             | 與新步數結合的 warmup 計算（見程式註解）                               |
| `EPOCHS`                          | `3`                                                | 非 `None` 時會影響總 steps 推算                                        |
| `STEPS_MAX`                       | `100_000`                                          | 上限護欄                                                               |
| `ROUTER_T_START` / `ROUTER_T_END` | `2.0` / `0.5`                                      | MoE router 溫度退火端點                                                |
| `ROUTER_WARMUP` / `ROUTER_TOTAL`  | `500` / `10000`                                    | Router 退火步數尺度                                                    |
| `SFT_FIXED_ROUTER_T`              | `0.5`                                              | SFT 預設固定 router 溫度（`None` 則重新退火）                          |
| `VAL_FRAC` / `VAL_EVERY_STEPS`    | `0.05` / `100`                                     | 驗證子集比例與間隔                                                     |
| `SFT_TEST_TEMPERATURE` 等         | 見程式                                             | 週期解碼測試用                                                         |

**CrossEntropy**：標準 SFT 路線使用 `ignore_index=-100`（與 `labels` 對齊）；見 `train_sft.py` 內說明與 [`verify_stf_cot_mask.py`](../../sft_cot_bundle/scripts/verify_stf_cot_mask.py)。

### 1.2 與 SFT_FORMAT 對齊的長度語意

| 來源          | 參數               | 值       | 說明                                                                                             |
| ------------- | ------------------ | -------- | ------------------------------------------------------------------------------------------------ |
| SFT_FORMAT §6 | `model_max_length` | 2048     | Tokenizer 設計上限                                                                               |
| SFT_FORMAT §6 | 建議單筆           | ~512     | 撰寫資料時 `input+cot+output` 目標                                                               |
| `train_sft`   | `SEQ_LEN`          | 預設 512 | **訓練張量**時間維度；若資料常超過 `SEQ_LEN+1` tokens 會**從頭保留、截斷尾部**（易切掉關閉標籤） |

### 1.3 關鍵腳本與模組路徑

| 路徑                                                                                                   | 用途                                                   |
| ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| [`sft_cot_bundle/scripts/stf_cot_to_bin.py`](../../sft_cot_bundle/scripts/stf_cot_to_bin.py)           | JSON → ChatML → tokenize → HF / `.bin`                 |
| [`sft_cot_bundle/scripts/train_sft.py`](../../sft_cot_bundle/scripts/train_sft.py)                     | `_build_xy_masked`、`MaterializedSftDataset`、訓練迴圈 |
| [`sft_cot_bundle/scripts/verify_stf_cot_mask.py`](../../sft_cot_bundle/scripts/verify_stf_cot_mask.py) | 與 `_build_xy_masked` 一致的 mask 抽查                 |
| [`sft_cot_bundle/scripts/verify_mask_xy.py`](../../sft_cot_bundle/scripts/verify_mask_xy.py)           | 輔助比對 x/y                                           |

---

## 2. 資料管線與 mask（ASCII）

### 2.1 自 JSON 到訓練 batch（邏輯流）

```text
  emotion.json / self_awareness.json / ... / merged stf.json
              |
              v
  +-----------+-----------+
  | stf_cot_to_bin.py     |
  | - category -> bucket  |
  | - wrap cot / output   |  見 cot_dataset/SFT_FORMAT.md
  +-----------+-----------+
              |
              v
       完整 ChatML 字串（每筆一列 text 欄位）
              |
              v
  +-----------+-----------+
  | train_sft.py          |
  | _build_xy_masked       |
  | - encode 成 ids[]      |
  | - 找 <|im_start|>assistant\n 後至結尾序列 |  supervised 區間 labels[j]=ids[j]
  | - 其餘 labels=-100     |
  | - 截斷 / pad 至 SEQ_LEN+1 -> x, y        |
  +-----------+-----------+
              |
              v
       DataLoader -> model(logits) -> CE(logits, y)
```

### 2.2 單筆序列上「誰被監督」（概念條）

```text
  ids index:    0 --------|---- user/sys masked ----|---- assistant ----|--- pad ---|
  labels:      -100      -100                     id_k..id_end          -100..
  loss:          不計算                     計算 CE（next-token 對 y）     不計算
```

結尾序列的實際 byte 與 tokenizer 版本以 **`train_sft._build_xy_masked`** 與 SFT_FORMAT §4 為準（支援多種 `im_end` 變體比對）。

---

## 3. Next-token 對齊（x／y）

`MaterializedSftDataset` 回傳：

- `x = ids[0:SEQ_LEN]`
- `y = labels[1:SEQ_LEN+1]`（長度 `SEQ_LEN`）

因此位置 `k` 的 logits 預測目標為 **下一個 token** `y[k] == labels[k+1]`；`y[k] == -100` 的位置不參與 loss 聚合。

```text
  index:     ...  k-1    k     k+1   ...
  x (input): ... id[k-1] id[k] ...
  y (target):... id[k]   id[k+1] ...   僅在 labels 非 -100 處有效
```

---

## 4. 資料預處理：RE 與 token 級標記（詳細規格）

本章規範 **如何在「字串層」找結構、再映射到「token 層」權重**，供 SFT-GO、IPS 權重欄位、或自訂 `compute_loss` 使用。  
**前提**：Task 1 已通過；tokenizer **必須**與訓練相同目錄／版本（`TOKENIZER_DIR`），`VOCAB_SIZE=32007`。

### 4.1 在「哪一段字串」上跑 RE？

| 策略          | 字串來源                                                                                      | 優點                                           | 注意                                                                  |
| ------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------- | ------------------------------ |
| **A（建議）** | 僅 **assistant 正文**：組裝後的 `cot` + `output` 純文字（與 JSON 欄位一致、不含 ChatML 標籤） | 與 GUIDE 撰寫規則一致；不受 system prompt 干擾 | 與 `input_ids` 對齊時需先定位該段在 **完整 ChatML** 中的 **字元偏移** |
| **B**         | 完整 ChatML 字串                                                                              | 與磁碟上 `text` 欄位一致                       | RE 易誤觸 system/user 內的 `Step`、`                                  | ` 等子字串；需更嚴格前後文錨定 |
| **C**         | 僅 `<think>`…`</think>` 內子字串                                                              | 專注推理段                                     | 需先切子區間再算全域 offset                                           |

**規格建議**：實作採 **A 或 C**；若採 **B**，每個 RE 必須帶 **左側錨點**（例如已越過第一個 `assistant` header 之後）以避免誤匹配。

### 4.2 RE 模式表（建議與範圍）

以下模式以 **Python `re`** 表示；預設建議開啟 `re.MULTILINE`（`^`／`$` 對齊行首）若使用行首類模式。

| ID  | 名稱                   | 模式                        | 常用 flags  | 匹配意義                                               |
| --- | ---------------------- | --------------------------- | ----------- | ------------------------------------------------------ |
| R1  | 推理步驟導引           | `r"Step\s*\d+[:：]?"`       | 無或 `re.I` | `Step 1:`、`Step 2：`                                  |
| R1b | 中文步驟（若資料出現） | `r"步驟\s*\d+[:：]?"`       | —           | 與 GUIDE「全英文」衝突時應 **Task 1 退件**，此列僅防呆 |
| R2  | Markdown 豎線          | `r"\|"`                     | —           | 表格欄分隔                                             |
| R3  | 表格分隔行（鬆弛）     | `r"\|?[\s\-]*\|[\s\-]*\|?"` | —           | 僅作弱提示；易誤報需人工複核                           |
| R4  | 粗體片段（輔助）       | `r"\*\*[^*]+\*\*"`          | —           | 與 GUIDE「每步粗體」對照；非強制權重依賴               |
| R5  | 行首標題（若未禁止）   | `r"^#+\s"`                  | `re.M`      | GUIDE 禁止 H1/H2 時 **不應**在合法資料出現             |
| R6  | fenced code            | `r"```"`                    | —           | 出現頻率低；多用於 Email／技術類                       |

**規則**：Task 1 以 GUIDE 為準；RE 僅為 **檢測／加權輔助**。若 R5 在 Emotion 類大量命中 → 先修資料而非加權。

### 4.3 字元 span → token index（演算法規格）

**輸入**：完整用於訓練的 UTF-8 字串 `full_text`（與 `tok.encode(full_text, add_special_tokens=False)` 一致）、以及在某子字串上的每個 match `(s, e)`（**半開區間** `[s, e)` 或 Python `match.start()`/`match.end()`）。

**輸出**：長度為 `T = len(input_ids)` 的 `loss_weight`（`float32`），或二元 `structure_mask`（`0/1`）再與純量相乘。

**步驟**：

1. **Tokenize（必開 offset mapping）**
   - 對 `PreTrainedTokenizerFast`：  
     `enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)`
   - 得到 `input_ids`、`offset_mapping`，其中 `offset_mapping[i] = (char_start_i, char_end_i)` 對第 `i` 個 token。

2. **子字串偏移還原（若 RE 跑在 cot-only）**
   - 令 `base` = `cot` 在 `full_text` 中的起始字元 index（由組裝函式保證或自行 `full_text.find(cot_snippet)`）。
   - RE 在 `cot` 上得到 `(s_rel, e_rel)` 後，全域 `(s, e) = (base + s_rel, base + e_rel)`。

3. **Token 索引集合（兩種政策，擇一寫死）**

   | 政策        | 定義                                                                     | 用途                                                      |
   | ----------- | ------------------------------------------------------------------------ | --------------------------------------------------------- |
   | **P-union** | 所有滿足 `not (char_end_i <= s or char_start_i >= e)` 的 token index `i` | 整個子詞片段都加權；適合 BPE 切很碎時仍覆蓋完整視覺 token |
   | **P-first** | 僅取滿足上式的 **最小** `i`                                              | 權重集中；可能只落在第一個子詞                            |

4. **寫入權重（mask 感知）**
   
   **關鍵**：權重乘法 `w_struct` 應**只應用於 assistant 部分**，user/system 部分保持 `1.0`，以確保與訓練時的 mask 邏輯一致。
   
   實作步驟：
   - 初始化 `loss_weight[i] = 1.0`（全序列）。
   - **定位 assistant 邊界**：在 `full_text` 中查找 `<|im_start|>assistant\n` 的位置，計算其後第一個 token 的索引 `assistant_start_token`。
   - 對每個結構命中集合 `S`，**僅對滿足** `i >= assistant_start_token` **的 token** 乘權：
     ```
     for i in S:
         if i >= assistant_start_token:
             loss_weight[i] *= w_struct
     ```
   - **Clamp**：`loss_weight[i] = min(w_max, max(w_min, loss_weight[i]))`，建議 `w_min ∈ [0.25, 1.0]`、`w_max ∈ [2.0, 10.0]`。
   - **結果**：user/system tokens 的權重始終為 `1.0`（無加權）；assistant tokens 內的結構才被加權。

5. **與 `labels` 對齊**
   - 訓練時，CE loss 只在 **`labels[j] != -100`** 的位置計算（assistant 部分）。
   - 權重向量 `loss_weight` 包含全序列，但只有 assistant 部分的權重會被實際使用（因為 user/system 的 labels 為 `-100`）。
   - **確保一致性**：由 `find_assistant_start_token` 識別的邊界必須與訓練管線中 `_build_xy_masked` 使用的邊界一致。詳見 [`train_sft.py` §_build_xy_masked](../../sft_cot_bundle/scripts/train_sft.py)。

6. **Sanity check（必做）**
   - 隨機抽 `N` 筆（建議 `N ≥ 20`），對每個命中 token `i`：將 `input_ids[i]` 或 `input_ids[i-k:i+k]` decode 回字串，目視是否落在預期 `Step`／`|`／粗體內。
   - 統計：`structure_hit_rate`、平均每筆權重 `>1` 的 token 數，避免全序列被乘爆。

### 4.4 張量形狀與 dtype（規格）

| 名稱                 | 形狀              | dtype     | 說明                                        |
| -------------------- | ----------------- | --------- | ------------------------------------------- |
| `input_ids`          | `[B, T]`          | `int64`   | 與現有訓練一致                              |
| `labels` / `y`       | `[B, T]`          | `int64`   | `-100` 為忽略；與 `train_sft` 對齊          |
| `loss_weight`        | `[B, T]` 或 `[T]` | `float32` | 與 CE **逐元素**相乘後再 `sum/mean`         |
| `ips_weight`（離線） | `[B, T]`          | `float32` | 由參考模型算完存檔；建議同時存 `valid_mask` |

### 4.5 離線產物（建議 schema，實作可調）

若 IPS 或結構權重與資料 **分檔** 儲存：

```text
  sample_uid 或 (file, row_index)  ->  npz / parquet:
    - token_ids_ref: optional
    - ips_w: float32[T] 或壓縮稀疏
    - loss_weight: float32[T]
    - tokenizer_hash: string  (版本防呆)
```

---

## 5. 各方案計算時機與建議參數表

### 5.1 時機總表

| 方案               | 計算時機                 | 說明                                     |
| ------------------ | ------------------------ | ---------------------------------------- |
| **IPS**            | 事前（離線 forward）     | 參考模型 \(p*{\text{ref}}(y_t\|x*{<t})\) |
| **SFT-GO**         | 事前結構 mask + 訓練聚合 | 分組 CE／worst-group max                 |
| **Step-DPO／ORPO** | 事前造偏好對             | 非單純 CE                                |
| **SPHL**           | 事前標註 + 訓練輔助頭    | 需額外 head／標籤                        |
| **Focal／TOFU**    | 訓練中                   | 用當前 `p_t`                             |
| **FCP（EOS）**     | 訓練中                   | 在 cot 區間對 EOS logit 加罰             |
| **SCALe**          | 訓練中                   | `global_step` 排程                       |
| **DFT**            | 訓練中                   | stop_gradient 於高信心 token             |

### 5.2 建議超參數（起點值；需實驗驗證）

#### IPS（逆概率加權）

| 參數       | 符號          | 建議範圍                | 說明                                                         |
| ---------- | ------------- | ----------------------- | ------------------------------------------------------------ |
| 參考模型   | —             | 凍結 Base / 上一版 SFT  | 與當前 \(\theta\) 分離                                       |
| 平滑       | \(\epsilon\)  | `1e-6` ~ `1e-4`         | \(p*{\text{ref}} \leftarrow \max(p*{\text{ref}}, \epsilon)\) |
| 原始權重   | \(w_t\)       | \(1 / p\_{\text{ref}}\) | 低機率 → 高權重                                              |
| Clamp 下限 | \(w\_{\min}\) | `0.5` ~ `1.0`           | 避免過小梯度                                                 |
| Clamp 上限 | \(w\_{\max}\) | `3.0` ~ `10.0`          | 防爆                                                         |

#### Focal／TOFU（token 級）

| 參數                 | 符號       | 建議範圍       | 說明                       |
| -------------------- | ---------- | -------------- | -------------------------- |
| Focusing             | \(\gamma\) | `1.0` ~ `3.0`  | \((1-p_t)^\gamma\)         |
| \(\alpha_t\)（可選） | 平衡類別   | `0.25` ~ `1.0` | 若採用 class-balanced 變體 |

#### SCALe（think vs answer 排程）

對 think 段權重（概念，與舊筆記一致）：

\[
w*{\text{think}}(s) = \eta*{\min} + \tfrac{1}{2}(\eta*{\max}-\eta*{\min})\bigl(1 + \cos(\pi s/S)\bigr)
\]

| 參數                   | 建議範圍                              | 說明              |
| ---------------------- | ------------------------------------- | ----------------- |
| \(s\)                  | `global_step`                         | 當前優化步        |
| \(S\)                  | `max_steps` 或 `total_training_steps` | 與 scheduler 一致 |
| \(\eta\_{\max}\)       | `1.0`                                 | 初期 think 權重   |
| \(\eta\_{\min}\)       | `0.3` ~ `0.7`                         | 末期 think 權重   |
| \(w\_{\text{answer}}\) | 常數 `1.0` 或略升                     | 維持答案區訊號    |

**區段判定**：以 special token 邊界界定 `<think>`…`</think>` vs `<final>`…（見 SFT_FORMAT）。

#### FCP（EOS 懲罰，概念）

| 參數                      | 建議範圍       | 說明                            |
| ------------------------- | -------------- | ------------------------------- |
| \(\lambda\_{\text{EOS}}\) | `0.05` ~ `0.5` | 懲罰項係數                      |
| 最小 cot token 長度       | \(L\_{\min}\)  | 依資料分位數設定，例如 p25 長度 |

#### SFT-GO（分組）

| 參數     | 符號                   | 建議範圍      | 說明                        |
| -------- | ---------------------- | ------------- | --------------------------- |
| 混合係數 | \(\lambda\)            | `0.1` ~ `0.5` | 與 IPS 或基礎 CE 線性組合時 |
| 結構權重 | \(w\_{\text{struct}}\) | `2.0` ~ `5.0` | 乘在結構組 token            |

#### Step-DPO（離線偏好）

| 參數      | 建議範圍      | 說明                                         |
| --------- | ------------- | -------------------------------------------- |
| \(\beta\) | `0.1` ~ `0.5` | DPO 溫度                                     |
| 破壞策略  | —             | 刪步、亂序、截斷等；需固定 RNG seed 以利復現 |

---

## 6. 訓練時 loss 整合（概念與 ASCII）

### 6.1 加權 CE（最簡整合形）

```text
  logits:     [B, T, V]
  y:          [B, T]     with -100 ignored
  ce_t:       [B, T]     per-position CE (reduction='none')
  w:          [B, T]     loss_weight * ips_weight * zone_schedule ...
  loss = sum( ce_t * w * (y != -100) ) / sum( (y != -100) * (w > 0) )
```

### 6.2 前向到反傳（ASCII）

```text
        input_ids
             |
             v
    +--------+---------+
    |   LM forward      |
    +--------+---------+
             |
        logits [B,T,V]
             |
     +-------+--------+
     | CE per token   |<----- y (next-token shifted)
     +-------+--------+
             |
     * loss_weight, ips, scale...
             |
     +-------+--------+
     | masked mean    |
     +-------+--------+
             v
          backward
```

---

## 7. IPS／SFT-GO 階段規劃與組合公式

### 7.1 階段

1. **結構可定位**：RE + §4 映射通過 sanity。
2. **IPS 離線**：ref model 前向 → 存 `ips_w`。
3. **SFT-GO**：`structure_mask` 分組 → worst-group loss。
4. **訓練**：例如  
   \[
   \mathcal{L} = (1-\lambda)\,\mathbb{E}[\text{CE}_t \cdot w^{\text{IPS}}_t] + \lambda \,\mathcal{L}_{\text{worst_group}}
   \]  
   並可對 \(\lambda\) 與 think 區排程權重 \(w_{\text{think}}(s)\) 再做餘弦調度。

### 7.2 取捨（與舊稿一致）

- 格式亂（漏 `\|`）→ 偏向 **SFT-GO**／結構權重。
- 推理中途發散 → **IPS** 或 **Focal**。

---

## 8. 架構示意：FCP、SCALe、SFT-GO、Focal

以下圖中「Think／Answer」對應 ChatML 的 `<think>`／`<final>` 區段（名稱以 SFT_FORMAT 為準）。

```text
===============================================================================
                     Loss 工程：推理鏈 (CoT) 完整性強化架構
===============================================================================

[ 訓練序列分段 (Training Sequence) ]

   <user+sys masked> | <think 區>              | <final 區>      | <EOS / 結尾 special>
   ------------------|-------------------------|-----------------|----------------------
        -100         |   邏輯核心（可排程加權） |  答案（常駐高權重） |  學會停止

[ 1. FCP（格式與 EOS 懲罰）]
   * 對區段 delimiter 的 token 施以 w_delim >> 1（若與 SFT-GO 並用須避免重複乘算）
   * 在 think 區間內對「提早 EOS」加額外 penalty（實作須讀取 EOS token id）

[ 2. SCALe（動態排程）]
   * w_think(s) 餘弦由 eta_max -> eta_min；w_answer 常近 1

[ 3. SFT-GO（分組）]
   * 結構組：Step、|、換行等 vs 文本組 -> worst-group max 或加權和

[ 4. Focal / TOFU]
   * (1 - p_t)^gamma 放大低信心正確 token 的 CE

===============================================================================





================================================================================
      Loss 工程：推理鏈完整性強化架構（基於 ChatML <think> / <final>）
================================================================================

                          訓練序列 (input_ids) 與預設 mask
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │  <|im_start|>system...<|im_end|>  │  <|im_start|>assistant<|im_start|>think  │
 │         labels = -100 (忽略)       │          labels = -100 (忽略)             │
 ├────────────────────────────────────┼──────────────────────────────────────────┤
 │           繼續 <think> 區段        │ <|im_end|>think<|im_start|>final        │
 │      labels = token_id (計算損失)  │  labels = token_id (計算損失)             │
 ├────────────────────────────────────┼──────────────────────────────────────────┤
 │  <|im_end|>final<|im_end|>assistant│  <|endoftext|> (EOS)                     │
 │         labels = -100 (忽略)       │  labels = token_id (可選是否計算)         │
 └────────────────────────────────────┴──────────────────────────────────────────┘
   ↑                                 ↑                              ↑
   │                                 │                              │
   │    [訓練中動態讀取]              │                              │
   │    pos_think_start, pos_think_end, pos_final_start, pos_final_end (由 mask 推算)

================================================================================
                              四項 Loss 工程技術
================================================================================

┌───────────────────────────────────────────────────────────────────────────────┐
│ 1. FCP (Format & EOS Penalty)                                                 │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  作用範圍            │  邏輯                                                  │
│  ───────────────────┼─────────────────────────────────────────────────────── │
│  <think> 結束邊界   │  Delimiter weighting:                                   │
│  (例如              │  對 <|im_end|>think 等特殊 token 給予固定權重 w_delim   │
│  <|im_end|>think)   │  (例：3.0)。確保模型學會輸出正確區段分隔符。            │
│                     │  ⚠️ 若與 SFT-GO 共用相同 token，避免重覆加權。         │
│  ───────────────────┼─────────────────────────────────────────────────────── │
│  <think> 區段內     │  Early EOS penalty:                                     │
│  所有位置           │  loss += λ_eos * relu( log P(EOS) - log(δ) )           │
│                     │  強制模型在推理過程中壓低 EOS 機率，防止提早結束。      │
│                     │  超參數：門檻 δ = 0.01, λ_eos = 0.1~0.5                 │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│ 2. SCALe (Scheduled Loss Annealing for CoT)                                   │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  作用範圍          │  公式                                                    │
│  ────────────────┼─────────────────────────────────────────────────────────── │
│  <think> 區段     │  w_think(s) = η_min + 0.5*(η_max - η_min)*(1+cos(π·s/S)) │
│                   │  s: 目前訓練步數, S: 總排程步數                            │
│                   │  典型值：η_max=1.0, η_min=0.3, S=total_steps              │
│  <final> 區段     │  w_final(s) = 1.0（常數，或輕微對稱變化）                 │
│  ────────────────┼─────────────────────────────────────────────────────────── │
│  目的             │  初期集中學習推理結構，後期強化答案正確性。                 │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│ 3. SFT-GO (Structure Token Group Weighting)                                    │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  作用範圍                │  流程                                              │
│  ──────────────────────┼────────────────────────────────────────────────────  │
│  預先定義的結構 token   │  1. 以 Regex 掃描原始文本 (cot 區段)：               │
│  例：Step, |, \n,       │     定位 Step\s*\d+, 表格豎線, 換行等符號           │
│  <|im_end|>think 等     │  2. 用 tokenizer char_to_token() 映射至 token 索引  │
│                         │  3. 建立 structure_mask (boolean tensor)             │
│                         │  4. 為這些位置指定高權重 w_struct (例：3.0~5.0)     │
│  ──────────────────────┼────────────────────────────────────────────────────  │
│  與 Focal 結合方式      │  最終 token 權重 = w_focal * w_struct (相乘)         │
│                         │  或 = w_focal + w_struct (相加)                      │
│                         │  (相乘較穩定，避免權重爆炸)                           │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│ 4. Focal Loss (Adaptive Hard-Token Focusing)                                   │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  作用範圍          │  公式                                                    │
│  ────────────────┼───────────────────────────────────────────────────────────  │
│  <think> +        │  對每個 token t：                                         │
│  <final> 全部     │  若 label ≠ -100：                                        │
│  (可限制在         │      p_t = softmax(logits[t])[label[t]]                  │
│  結構關鍵區)       │      w_focal = (1 - p_t)^γ                               │
│                   │      γ = 2.0 (典型值)                                      │
│                   │  CE_focal = - w_focal * log(p_t)                           │
│                   │                                                            │
│                   │  效果：低機率 token（困難）梯度放大；高機率 token 梯度縮小 │
└───────────────────────────────────────────────────────────────────────────────┘

================================================================================
                           權重合併與最終 Loss
================================================================================

針對每個 token t (忽略 labels=-100 的位置)：

  1. 基礎權重 w_base = 1.0

  2. 根據位置應用 SCALe：
     if t in think_region:  w_sched = w_think(global_step)
     elif t in final_region: w_sched = w_final(global_step)
     else: w_sched = 1.0

  3. 結構加權 SFT-GO：
     if structure_mask[t]: w_struct = 3.0 (或 5.0)
     else: w_struct = 1.0

  4. 焦點加權 Focal：
     由當前模型輸出動態計算 w_focal = (1-p_t)^γ

  5. Token 總權重：
     w_total[t] = w_sched * w_struct * w_focal
     (可選 clamp 防止梯度爆炸，例如 max=10.0)

  6. 加權交叉熵：
     loss_token[t] = w_total[t] * CE(p_t, label[t])

  7. 加入 FCP 懲罰項（僅對 think 區域）：
     loss_eos_penalty = λ_eos * sum_{t in think} relu( log P(EOS)[t] - log(δ) )

  8. 最終 Loss：
     L = mean(loss_token) + loss_eos_penalty / num_tokens

================================================================================
                         超參數設定建議 (快速起步)
================================================================================

  γ (Focal)         : 2.0
  w_struct (SFT-GO) : 3.0 ~ 5.0
  w_delim (FCP)     : 3.0（若與 SFT-GO 重疊則以 SFT-GO 為主）
  δ (EOS門檻)       : 0.01
  λ_eos             : 0.1 ~ 0.2
  SCALe η_max       : 1.0
  SCALe η_min       : 0.3
  SCALe 排程步數    : 與總訓練步數相同（或 80% 步數進行退火，後期固定）
  clamp max weight  : 10.0

================================================================================
```
