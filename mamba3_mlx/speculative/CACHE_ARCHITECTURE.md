# Speculative Decoding 與 Cache 系統架構解析

這份文件旨在快速解釋 `mamba3_mlx/speculative/` 目錄下各個核心元件的**原理**、**用途**與**意義**，幫助開發者快速理解這個投機解碼（Speculative Decoding）系統是如何運作並達成加速的。

---

## 1. 核心解碼迴圈 (The Decoders)

### `jacobi.py` (Greedy Jacobi) & `jacobi_sampling.py` (SJD)
*   **原理**：傳統的自迴歸（Auto-Regressive）模型是一次生成一個 token。Jacobi 解碼則是「猜測」接下來的 $K-1$ 個 tokens，並將「已確認的最後一個 token + 猜測的 $K-1$ 個 tokens」組成長度為 $K$ 的序列，一次性送入模型（Forward Pass）進行平行驗證（Parallel Verification）。
*   **意義**：將原本需要 $K$ 次的前向傳播（Forward Pass）壓縮成 $1$ 次批次運算。只要猜測命中率夠高，就能大幅提升每秒生成的 token 數（TPS, Tokens Per Second）。
*   **差異**：`jacobi.py` 要求猜測的 token 必須與模型的 Greedy Argmax **完全一致**才接受；`jacobi_sampling.py` (SJD) 則是基於機率分佈進行拒絕採樣（Rejection Sampling），允許在 Sampling 模式下接受高品質但不一定是 Argmax 的猜測，從而達到更高的接受率（ARL, Average Run Length）。

---

## 2. 猜測來源元件 (Draft Sources)

為了讓 Jacobi 解碼發揮作用，我們需要快速且準確地產生那 $K-1$ 個猜測 tokens。我們採用了「無額外模型（Training-free）」的多來源混合草稿策略。

### 2.1 `SuffixRetriever` (Prompt Lookup Decoding / PLD)
*   **位置**：`drafts.py`
*   **原理**：維護一個包含 Prompt 和已生成 tokens 的滑動視窗（Buffer）。當需要猜測時，拿當前剛生成的最後幾個 tokens 作為「後綴（Suffix）」，去 Buffer 裡尋找歷史上是否出現過相同的片段，如果有的話，就把歷史片段後面的 tokens 抓過來當作猜測。
*   **意義**：非常適合處理**重複性高**的文本（例如程式碼縮排、特定格式的 JSON、或是文章中的特定名詞與句型）。不需要訓練，只需要比對歷史紀錄。

### 2.2 `NGramCache` (Runtime N-Gram 快取)
*   **位置**：`ngram_cache.py`
*   **原理**：一個基於 LRU（最近最少使用）的字典。記錄每當看到特定的連續 $N-1$ 個 tokens 時，接下來出現的 token 是什麼（MRU, 最近最常出現）。在推論過程中，每確認一個 token，就會動態更新這個快取。
*   **意義**：模型在當前對話中會表現出短期的局部規律性。Runtime N-Gram 能快速捕捉並重用這些短期規律。

### 2.3 `cot_ngram` (離線訓練語料快取 / CoT Cache)
*   **位置**：`bake_cot_caches.py` (生成) / `cot_cache.py` (載入)
*   **原理**：這是一個**預先算好（Pre-baked）**並設定為唯讀（READ-ONLY）的 `NGramCache`。我們將訓練模型用的完整 Chain-of-Thought (CoT) 語料集，離線掃描一遍，統計所有 $N-1$ 組合最常接的 token。
*   **意義**：這是**系統最大的加速來源**。因為模型在生成 CoT 時有強烈的特定用語習慣（例如 "Step 1:", "Therefore,"）。預先載入這個快取等於給了解碼器一本「常見句型字典」，讓它在第一步就能給出極高命中率的猜測。

---

## 3. 狀態追蹤與整合 (Integration)

有了多個猜測來源，我們需要有機制來決定「什麼時候該用誰」。

### 3.1 `CoTPhaseTracker` (階段追蹤器)
*   **位置**：`cot_cache.py`
*   **原理**：監聽目前已經生成的 token stream。當看到特定的標記（Marker tokens，如 `<think>`, `</think>`, `<final>`, `</final>`）時，會切換內部的狀態機（`think` -> `other` -> `final`）。
*   **意義**：由於模型在思考階段（think）和最終回答階段（final）的用字遣詞習慣截然不同。Tracker 負責在不同的階段，將對應的 `cot_ngram` 快取（think cache 或是 final cache）餵給猜測系統，確保猜測的上下文完全吻合。

### 3.2 `build_hybrid_branches` (混合分支建構器)
*   **位置**：`drafts.py`
*   **原理**：作為猜測系統的「大腦 / 排程器」。它會向上述的所有元件要猜測結果，並依照優先權（Priority）組合成多個不同的分支（Branches）讓模型驗證。
*   **優先權順序**：
    1. **Suffix Retriever** (最長後綴匹配，針對當前對話的長篇重複)
    2. **CoT N-Gram** (訓練語料的高頻句型，命中率極高)
    3. **Runtime N-Gram** (當前對話的短期規律)
    4. **Carry Fallback** (最差情況，重複最後一個 token)
*   **意義**：單靠一種猜測方式容易遇到瓶頸。混合多種來源可以達到**互補（Decorrelation）**的效果。當 N-Gram 找不到時，Retriever 可能找得到；當前對話沒出現過的，訓練語料快取可能知道，從而最大化整體的平均接受長度（ARL）。

---

## 總結工作流 (Workflow Summary)

```text
[Start Round]
     │
     ▼
[CoTPhaseTracker] ──(判斷階段: think/final/other)──┐
                                                   ▼
                                         [build_hybrid_branches]
                                         索取草稿 (Drafts)
                                          ├─ SuffixRetriever (長篇重複)
                                          ├─ CoT 快取 (對應階段的高頻句型)
                                          └─ Runtime N-Gram (短期規律)
                                                   │
                                                   ▼
                                        組合成 K 長度的猜測序列
                                                   │
                                                   ▼
                                         [Jacobi Forward Pass]
                                        模型進行 1 次平行驗證
                                                   │
                                                   ▼
[輸出給使用者] ◄──(驗證通過的 tokens)── [比對 Argmax 決定接受長度]
     │                                             │
     ▼                                             ▼
[下一輪生成] ◄────────────────────── (更新 Suffix & Runtime N-Gram)
```
