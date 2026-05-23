# CoT 與資料集文件索引

本目錄整理 **訓練前驗證（Task 1）** 與 **後續 Loss 工程（Task 2+）** 的文件邊界，避免與 [`cot/step/task1.md`](../step/task1.md) 舊稿中「驗證 + Loss」混寫造成重複解讀。

## 權威來源（寫資料與跑管線必讀）

| 文件 | 角色 |
|------|------|
| [`cot_dataset/GUIDE.md`](../../cot_dataset/GUIDE.md) | 協作者撰寫 JSON 的格式、類別、品質、提交與退件標準。 |
| [`cot_dataset/SFT_FORMAT.md`](../../cot_dataset/SFT_FORMAT.md) | ChatML 組裝、`category` → system prompt bucket、`labels` / mask 規則與抽查工具說明。 |

## 本目錄（驗證與延伸）

| 文件 | 角色 |
|------|------|
| [`TASK1_DATASET_VALIDATION.md`](TASK1_DATASET_VALIDATION.md) | **Task 1 唯一權威**：在訓練與 Loss 修改前，資料集須通過的檢查項、與 GUIDE / SFT_FORMAT 的對照表、Definition of Done。 |
| [`TASK2_LOSS_ENGINEERING.md`](TASK2_LOSS_ENGINEERING.md) | **Task 2+ 規格**：`train_sft` 常數、資料管線／mask／next-token ASCII、**RE→token 完整步驟與張量約定**、IPS／Focal／SCALe／FCP／SFT-GO／Step-DPO **建議超參表**、訓練 loss 整合圖（自舊 `step/task1.md` 擴充）。 |

## `cot/` 根目錄其他檔案

| 路徑 | 角色 |
|------|------|
| [`../1.md`](../1.md)、[`../2.md`](../2.md)、[`../3.md`](../3.md) | Loss 與結構化輸出的**研究筆記／彙整**，非 Task 1 必讀；實作 Loss 時再對齊 TASK2 與訓練程式。 |
| [`../step/task1.md`](../step/task1.md) | **索引**：指向本目錄 TASK1 / TASK2，避免在單一檔內混寫驗證與 Loss。 |

## 閱讀順序建議

1. 撰寫或審資料：`GUIDE.md` → 依 [`TASK1_DATASET_VALIDATION.md`](TASK1_DATASET_VALIDATION.md) 勾選驗證。
2. 確認訓練時長相：`SFT_FORMAT.md`（與 TASK1 第 7 節一致）。
3. 僅在 Task 1 全綠後：再讀 [`TASK2_LOSS_ENGINEERING.md`](TASK2_LOSS_ENGINEERING.md) 與 `cot/1.md`–`3.md`。
