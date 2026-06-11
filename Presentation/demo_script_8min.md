# Hybrid Mamba-TuckerMoE · 8 分鐘 Demo 逐字稿

> **版本：** 8 分鐘精簡版 · 13 張投影片  
> **節奏：** ~230 字/分鐘（繁體中文口說速度）  
> `[→]` = 換頁，`[指]` = 指向投影片特定區域

---

## 📌 時間分配總覽

| 投影片 | 主題 | 時間 | 累計 |
|--------|------|------|------|
| S1 | Title | 0:15 | 0:15 |
| S2 | Outline | 0:10 | 0:25 |
| S3 | Motivation | 0:55 | 1:20 |
| S4 | System Architecture | 1:00 | 2:20 |
| S5 | Mamba-3 SSM | 0:50 | 3:10 |
| S6 | Tucker Decomp | 0:50 | 4:00 |
| S7 | TuckerMoE vs Standard MoE | 0:50 | 4:50 |
| S8 | Multi-Stage Training & CoT | 0:50 | 5:40 |
| S9 | Speculative Jacobi Decoding | 0:50 | 6:30 |
| S10 | MLX Inference on Apple Silicon | 0:25 | 6:55 |
| S11 | Results | 0:40 | 7:35 |
| S12 | Conclusion | 0:15 | 7:50 |
| S13 | Demo + Q&A | 0:10 | 8:00 |

---

## ▶ 逐字稿

---

### S1 · Title（0:00 → 0:15）

大家好，我是賴弘偉。今天要分享的研究叫做 **Breaking the Memory Wall**——我們如何讓 TuckerMoE 在 Hybrid State Space Model 上，把記憶體瓶頸翻轉成算力受限。核心成果：82.87% 參數壓縮、32K token 長上下文、3.32 倍解碼加速，全部跑在 16 GB 的 M2 Pro 上。

`[→ S2]`

---

### S2 · Outline（0:15 → 0:25）

今天分成七個主題，八分鐘完成。我們從動機、架構、Mamba-3 與 Tucker 分解，一路講到訓練策略、SJD 加速，最後看結果和 Demo。

`[→ S3]`

---

### S3 · Motivation（0:25 → 1:20）

問題出在哪？在 16 GB 邊緣設備上做 LLM 推論，同時撞上三道牆。

`[指左卡片]` 第一，Attention 的複雜度是 O(N²)。到了 32K token，VRAM 直接爆掉，dense FFN OOM。

`[指中間卡片]` 第二，傳統 KV Cache 線性增長。每多 1024 個 token，你就要多搬一批 key-value pairs，帶寬壓力線性累積。

`[指右卡片]` 第三，Sparse MoE 是 bandwidth-bound 的。每個 decode step，router 選完 expert 之後，你要把那些 expert 的 weight 從 DRAM 搬進來——而 M2 Pro 的帶寬上限只有 200 GB/s，算力完全閒置在等資料。

這三個問題在 32K context 同時爆發。我們要做的，就是一次解掉全部三道牆。

`[→ S4]`

---

### S4 · System Architecture（1:20 → 2:20）

我們的解法叫 **Hybrid Mamba-TuckerMoE**。`[指架構圖]` 每個 Macro Block 由四個 Mamba-3 Block 加上一個 GQA Attention 組成。Mamba 負責長序列的 O(1) 狀態記憶，GQA 補充局部 attention 精度。

整個模型 24 個 Macro Block，嵌入維度 768，最關鍵的數字：`[指右側統計]` **total 417M 參數**，但等效 dense 容量是 **2.4B**，壓縮率 82.87%。

傳統 FFN 在這裡被 Tucker MoE 替代。每個 expert 的 weight matrix 做 Tucker 分解，共享 factor matrices 常駐 L2 cache，expert dispatch 從 memory-bound 變成 compute-bound——這是整個架構最核心的系統設計。

`[→ S5]`

---

### S5 · Mamba-3 SSM（2:20 → 3:10）

Mamba-3 最重要的改動是**梯形離散化**。`[指公式框]` 傳統 Mamba 用 Euler 法，B-bar 就是 Δ×B。Mamba-3 改成梯形中點法：

$$\bar{B}_t = \tfrac{\Delta_t}{2}(\bar{A}_t + I)B_t$$

這個修改讓離散化誤差更小，長序列的 receptive field 更紮實。

`[指四個特性卡]` 四個關鍵設計：O(1) state memory、32K context、RoPE 直接加在 B 和 C 矩陣上、訓練用 chunk-wise parallel scan 而推論用 sequential recurrence。

結果是：`[指底部統計條]` 狀態大小固定 14.1 MiB，不管你跑 1K 還是 32K token 都一樣。KV cache 線性增長的問題徹底解掉了。

`[→ S6]`

---

### S6 · Tucker Decomposition（3:10 → 4:00）

Tucker 分解的直觉非常簡單。`[指左側 SVG 圖]` 傳統做法：每個 expert 是一個完整的 W_e 矩陣，存 E×d² 個參數。Tucker 把 W_e 分解成：

$$W_e \approx U_{\text{in}} \cdot G_e \cdot U_{\text{out}}$$

`[指分解等式]` U_in 和 U_out 是**所有 expert 共享**的 factor matrices，只有核心張量 G_e 是 expert 專屬的。

效果：`[指壓縮數字]` 每個 expert 的參數從 75.5M 降到 7.39M，少了 90.2%。而且因為 U_in、U_out 共享並常駐 cache，等一下做 forward 時幾乎不需要 DRAM access。

精度方面：`[指 MSE 比較]` Tucker 的重建 MSE 是 0.00135，SVD 是 0.00247，Tucker 不只更省，精度還更好。

`[→ S7]`

---

### S7 · TuckerMoE vs Standard MoE（4:00 → 4:50）

現在來看兩者的計算流程對比。

`[指左半]` Standard MoE：Router 選出 top-K expert，每個 decode step 都要把那些 expert 的完整 W_e 從 DRAM 搬進 SRAM——每次 forward 都是一次大搬運。這就是為什麼標準 MoE 是 bandwidth-bound。

`[指右半]` TuckerMoE：U_in 和 U_out **只載入一次**，長期駐留在 L2 cache。每個 expert forward 只需要做兩步小矩陣乘——先把 x 壓到 rank 空間，再用 G_e 做映射。G_e 很小，搬運代價幾乎可以忽略。

這樣整個 expert dispatch 就從 **memory-bandwidth-limited 變成 matrix-multiply-limited**，M2 Pro 的 GPU compute 終於能充分利用。這是我們突破 memory wall 的核心手段。

`[→ S8]`

---

### S8 · Multi-Stage Training & CoT Loss（4:50 → 5:40）

訓練分三個階段。`[指三個階段箭頭]` 第一階段用 FineWeb-Edu 做語言模型預訓練，PPL 24.6；第二階段 UltraChat 200k SFT，PPL 降到 18.3；第三階段引入**客製化 5 項 CoT Loss**，PPL 再降到 12.1——從頭到尾下降了 51%。

`[指損失函數卡片]` CoT Loss 包含五個加權項：SCALe 評分、SFT-GO 正確率、Final-SW 滑動窗口、PDL 差分損失、InfoEntropy 資訊熵，再加上 λ·FCP 懲罰項來防止 expert 負載不均。

`[指右側 before/after 比較]` 沒有自定義損失的模型，CoT 輸出會跳過思考直接回答，邏輯跳躍明顯；有了這套損失，模型學會了先 think 再 final，推理鏈完整且連貫。

`[→ S9]`

---

### S9 · Speculative Jacobi Decoding（5:40 → 6:30）

傳統自迴歸解碼，每次只能生成一個 token。`[指 Jacobi 流程圖]` SJD 的想法是：一次猜測未來 K=16 個 token 當成 draft，再用原始模型做**平行 verify**，一次可以確認多個位置。

Draft 從三個來源填充：`[指三個 source 卡片]` ① NGramCache——從已生成文字找頻繁 4-gram；② SuffixRetriever——從 suffix array 找最相似的歷史片段；③ CoTPhaseTracker——狀態機偵測目前在 \<think\> 還是 \<final\> 階段，動態切換 cache 策略。

結果：`[指加速數字]` 平均接受長度 ARL=2.82，整體解碼加速 **3.32×**，從 42 tok/s 提升到實測峰值 139 tok/s。這是完全不改模型架構、純靠解碼演算法拿到的加速。

`[→ S10]`

---

### S10 · MLX Inference on Apple Silicon（6:30 → 6:55）

MLX 推論棧：`[指三個 panel]` prefill 達到約 3,800 tok/s、decode bf16 基線從 47 提升到 68 tok/s（靠 mx.compile）、KV 狀態 14.1 MiB、整個模型只佔 3.1 GB unified memory。

16 GB 的 M2 Pro 跑起來還有超過 12 GB 餘裕，不需要量化就能全速推論。

`[→ S11]`

---

### S11 · Results（6:55 → 7:35）

`[指四個大數字]` 四個核心成果：

- **82.87%** 參數壓縮——2.4B dense-equivalent 用 417M 實現
- **3.32×** 解碼加速——SJD K=16，ARL=2.82
- **3.1 GB** 記憶體佔用——16 GB 設備輕鬆跑，無需量化
- **PPL 12.1**——從預訓練的 24.6 降低 51%，三階段訓練的成效

`[指右側 Tucker vs SVD 比較]` Tucker 分解的重建精度比 SVD 更好（MSE 0.00135 vs 0.00247），驗證了在 MoE 場景下 Tucker 是比 SVD 更合適的分解方法。

`[→ S12]`

---

### S12 · Conclusion（7:35 → 7:50）

四個核心貢獻快速總結：`[指四個卡片]`

① **Tucker Is Viable**——Tucker 分解讓 MoE 從 memory-bound 走向 compute-bound  
② **32K Long Context**——Mamba-3 梯形離散化，O(1) state，16 GB 跑 32K  
③ **Embedded CoT**——五項自定義損失，CoT 推理能力從訓練階段嵌入  
④ **End-to-End MVP**——從訓練到推論到 Jacobi 加速，完整可運行系統

`[→ S13]`

---

### S13 · Demo + Q&A（7:50 → 8:00）

讓我快速 demo 一下。

`[開啟 terminal，跑指令]`

```bash
cd mamba3_mlx && make run
```

模型載入後直接問它一個需要思考的問題——你會看到 \<think\> tag 出現，CoT 推理鏈展開，最後輸出 \<final\> 答案。

謝謝。歡迎提問。

---

## 📝 備用補充說明（問答 Q&A 參考）

**Q: Tucker 分解 vs LoRA 有什麼差異？**  
Tucker 是三階張量分解，U_in/U_out 是 expert 間**共享**的 factor，不是個別 adapter。LoRA 是加 delta 而不是替換 weight。Tucker 直接替換 expert weight，在推論時不需要任何 merge 操作。

**Q: SJD 的 3.32× 是和什麼比較？**  
和 autoregressive baseline（Mamba3 bf16 compile, 42 tok/s）比較。SJD 在 K=16、ARL=2.82 時的實測峰值是 139 tok/s。

**Q: 訓練資料量多少？**  
三階段共約 2B tokens。FineWeb-Edu ~1.5B，UltraChat 200k ~300M，CoT fine-tune ~200M。

**Q: 和 Mamba2 有什麼差別？**  
Mamba-3 新增了梯形離散化（更精準的 B-bar 計算）和 RoPE 在 B、C 矩陣上的注入，以及 MIMO projection，讓 long-range 建模更穩。

**Q: 為什麼不直接用量化（4-bit）？**  
4-bit 量化在 MoE expert 上會有明顯質量退化。我們先在 bf16 baseline 驗證 Tucker 的有效性，量化是下一步的 orthogonal optimization。

---

*逐字稿版本：2026-06-11 · Hybrid Mamba-TuckerMoE ICLR 2026*
