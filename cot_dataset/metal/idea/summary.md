# 混合 Mamba-TuckerMoE 架構於 Apple Silicon 之 Metal 極致優化完整報告

> **涵蓋範圍**：MoE 路由機制演進 × Tucker 分解數學推導 × Apple M3 微架構分析 × Metal 內核實作 × 效能模型與驗證

---

## 目錄

1. [執行摘要](#1-執行摘要)
2. [架構背景與問題定義](#2-架構背景與問題定義)
3. [TuckerMoE 數學機制與 Tucker 分解](#3-tuckermoe-數學機制與-tucker-分解)
4. [MoE 路由機制深度解析](#4-moe-路由機制深度解析)
5. [Apple M3 GPU 微架構與記憶體模型](#5-apple-m3-gpu-微架構與記憶體模型)
6. [Bank 衝突病理學：GEMV 中的序列化根源](#6-bank-衝突病理學gemv-中的序列化根源)
7. [記憶體佈局優化策略](#7-記憶體佈局優化策略)
8. [Metal 3 非同步記憶體複製與 TMA 對等替代](#8-metal-3-非同步記憶體複製與-tma-對等替代)
9. [AMX 協同處理器與 SIMD-Group Matrix API](#9-amx-協同處理器與-simd-group-matrix-api)
10. [Fused Latent MoE Kernel 完整實作](#10-fused-latent-moe-kernel-完整實作)
11. [MLX JIT 動態編譯整合](#11-mlx-jit-動態編譯整合)
12. [效能模型與量化估算](#12-效能模型與量化估算)
13. [Tucker-MoE 推論優化的系統效益](#13-tucker-moe-推論優化的系統效益)
14. [Tucker-MoE 底層工程挑戰](#14-tucker-moe-底層工程挑戰)
15. [測試、驗證與診斷計畫](#15-測試驗證與診斷計畫)
16. [進化演算法優化展望](#16-進化演算法優化展望)
17. [風險評估與替代策略](#17-風險評估與替代策略)
18. [結論與最佳實踐藍圖](#18-結論與最佳實踐藍圖)

---

## 1. 執行摘要

本報告整合四份研究文件，針對在 **Apple M3 GPU**（19 核心，200 GB/s 統一記憶體頻寬）上執行 **Hybrid Mamba-TuckerMoE** 混合架構模型的極致推論優化進行全面深度解析。

### 核心問題

| 參數          | 數值                  |
| ------------- | --------------------- |
| 模型總參數    | 550M                  |
| 單步啟用參數  | 230M                  |
| 推論精度      | bfloat16 (BF16)       |
| Batch Size    | 1（自迴歸推論）       |
| 目標硬體      | Apple M3 GPU，19 核心 |
| 記憶體頻寬    | 200 GB/s              |
| Tucker 分解秩 | r3=256，r2=1024       |

### 核心病理

在 BS=1 的自迴歸推論中，系統面臨三層疊加瓶頸：

1. **記憶體牆（Memory Wall）**：模型處於絕對 Memory-bound 狀態，230M 活躍參數以 BF16 存儲，單次 Token 生成需讀取約 460MB 權重資料
2. **動態 Gather 序列化**：MoE 動態路由產生不可預測的非連續記憶體存取，破壞 GPU L2 快取預取機制
3. **SRAM Bank 衝突**：在 Threadgroup Memory 執行 GEMV 時引發 **32-way 極端 Bank 衝突**，記憶體存取延遲飆升 32 倍

### 核心優化方向

```
資料佈局重構（消除 Bank 衝突）
    ↓
非同步記憶體複製（隱藏搬移延遲）
    ↓
AMX 暫存器層級計算（消除 Threadgroup 往返）
    ↓
Fused Kernel（消除全域記憶體往返）
    ↓
JIT 動態編譯（降低執行期開銷）
```

---

## 2. 架構背景與問題定義

### 2.1 混合架構概述

Hybrid Mamba-TuckerMoE 結合了兩種先進技術路徑：

- **Mamba（SSM）**：基於狀態空間模型（State Space Models）的序列建模，依賴選擇性掃描（Selective Scan）
- **TuckerMoE**：基於 Tucker 分解的混合專家系統，透過張量分解極大壓縮專家權重規模

### 2.2 遷移背景

該模型正從 **NVIDIA 架構**（Triton 編譯器）向 **Apple Silicon 統一記憶體架構**（MLX 框架）遷移。這不是單純的 API 語法轉換，而是底層硬體哲學的範式轉移：

| 維度       | NVIDIA 架構                      | Apple Silicon                      |
| ---------- | -------------------------------- | ---------------------------------- |
| 記憶體架構 | 獨立 HBM + SRAM                  | 統一記憶體（LPDDR5）               |
| 內核語言   | CUDA / Triton                    | Metal Shading Language (MSL)       |
| 矩陣加速   | Tensor Cores                     | AMX（Apple Matrix Extensions）     |
| 共享記憶體 | Shared Memory（可達數十 MB）     | Threadgroup Memory（M3 限 32 KiB） |
| 非同步搬移 | TMA（Tensor Memory Accelerator） | simdgroup_async_copy               |

### 2.3 推論階段的效能瓶頸

大型語言模型推論分為兩個階段：

**Prefill 階段**（算力受限）：

- 並行處理所有輸入 Token
- 矩陣乘法擁有大內部維度，能填滿 GPU 算力
- 屬於 **Compute-bound** 狀態

**Decode 階段**（記憶體受限）：

- 每次僅生成一個 Token（BS=1）
- 需搬移龐大權重但浮點運算次數極少
- 深陷 **Memory-bound** 泥淖

當升級為 Top-K MoE 後，碎任務效應（Fragmentation Effect）進一步惡化：假設 Batch Size=128、Top-2 路由、64 個專家，每個專家僅獲分配 **4 個 Token**，算術強度趨近於零。

---

## 3. TuckerMoE 數學機制與 Tucker 分解

### 3.1 Tucker 分解的數學基礎

傳統稀疏 MoE 系統中，每個專家包含獨立的密集權重矩陣，導致記憶體佔用極大且存在結構性冗餘。TuckerMoE（基於 TD-MoE 理論）採用跨專家張量化與聯合三維分解技術。

設某層所有專家權重整合為三階張量：

$$T \in \mathbb{R}^{E \times d_{in} \times d_{out}}$$

其中 $E$ 為專家數量，$d_{in}$ 與 $d_{out}$ 為輸入與輸出維度。透過 Tucker 分解與多線性白化，該張量被近似分解為：

$$T \approx G \times_1 U_{expert} \times_2 U_{in} \times_3 U_{out}$$

各因子定義：

| 符號                                                          | 形狀                             | 說明                                        |
| ------------------------------------------------------------- | -------------------------------- | ------------------------------------------- |
| $U_{in} \in \mathbb{R}^{d_{in} \times r_{in}}$                | $d_{in} \times r_{in}$           | 所有專家共享的降維投影矩陣                  |
| $U_{out} \in \mathbb{R}^{d_{out} \times r_{out}}$             | $d_{out} \times r_{out}$         | 所有專家共享的升維投影矩陣                  |
| $G_{experts} \in \mathbb{R}^{E \times r_{in} \times r_{out}}$ | $E \times r_{in} \times r_{out}$ | 各專家專屬的核心特徵轉換矩陣（Core Tensor） |

其中 $r_{in}, r_{out} \ll d$（例如 $d=4096$，$r=256$），實現了參數的極大壓縮。

### 3.2 BS=1 推論前向傳播流程

在單一 Token 生成階段，推論邏輯分為五個步驟：

```
步驟 1：共享輸入投影
  x_shared = x × U_in
  （輸入特徵降維至 r_in 的低維空間）

步驟 2：動態路由運算
  e = Router(x_shared)
  （路由器判定激活的專家索引）

步驟 3：核心權重讀取（Gather）← 主要瓶頸
  G_active = G_experts[e]
  （從全域記憶體提取激活的專家核心矩陣）

步驟 4：專家收縮運算
  y_shared = x_shared × G_active

步驟 5：共享輸出投影
  y = y_shared × U_out
  （投影回原高維空間）
```

### 3.3 Tucker-MoE「降維後再路由」排程優化

Tucker-MoE 的結構允許實施「先降維、再路由（Route-after-Reduction）」策略，將推論排程重構為四個階段：

**階段一：全局稠密降維（Dense Reduction）**
$$Z = XU_{in} \quad (X \in \mathbb{R}^{N \times d} \rightarrow Z \in \mathbb{R}^{N \times r})$$

**階段二：基於低維特徵的路由（Latent Space Routing）**

- Router 利用降維後的輕量特徵 $Z$ 計算 Gating Scores
- 路由器計算量依比例大幅縮減

**階段三：核心專家運算（Low-Rank Grouped GEMM）**
$$H_i = Z_i C_i \quad (Z \in \mathbb{R}^{N \times r},\; C_i \in \mathbb{R}^{r \times r})$$

**階段四：全局稠密升維（Dense Expansion）**
$$Y = HV \quad (\text{恢復至原始 } d \text{ 維度空間})$$

---

## 4. MoE 路由機制深度解析

### 4.1 Token Choice Routing（傳統方案）

代表模型：Google Switch Transformer（Top-1）、Mistral Mixtral 8x7B（Top-2）

**運作原理**：

1. 對每個 Token 計算與所有專家的親和力分數（Affinity Scores）
2. 應用 Softmax 轉換為機率分佈
3. 挑選機率最高的 $K$ 個專家

**優勢**：

- 高度動態性與語意契合度
- 完美支援自迴歸因果律
- 適合生成式任務逐字解碼

**致命缺陷**：

- **負載不均（Load Imbalance）**：訓練動態產生馬太效應，多數 Token 被送往少數「明星專家」
- **路由崩塌（Routing Collapse）**：多數專家處於閒置或未充分訓練狀態
- **Token Dropping**：超過容量限制的 Token 被丟棄，損害模型精度

### 4.2 Expert Choice Routing（Google 2022 提案）

**運作原理**：決策權反轉，讓每個專家根據全局親和力分數主動挑選 Top-K 個 Token

**優勢**：

- 演算法層面完美硬體負載平衡
- 徹底根除 Padding 造成的算力浪費
- 每個 Token 被分配的專家數量動態可變

**致命缺陷**：破壞因果律（Non-causality），需預知所有 Token 才能排序，**無法用於自迴歸文本生成推論**

### 4.3 現代 Token Choice 演進：DeepSeek-V3 ALF-LB

DeepSeek-V3 提出「無輔助損失負載平衡（Auxiliary-Loss-Free Load Balancing，ALF-LB）」策略：

**核心機制**：

- 為每個專家 $i$ 維護動態偏差值 $b_i$
- 路由時：$s'_{i,t} = s_{i,t} + b_i$，Router 根據 $s'_{i,t}$ 進行 Top-K 選擇
- 訓練中持續監控負載：過載則降低偏差值，閒置則調升

**革命性優勢**：

1. **無梯度干擾**：偏差值不參與反向傳播，完全解放語意學習能力
2. **保證不掉字**：100% 無 Token Dropping，確保所有語意資訊完整傳遞

### 4.4 三種路由機制對比

| 路由機制核心比較 | 傳統 Token Choice         | Expert Choice            | 現代 Token Choice (DeepSeek-V3) |
| ---------------- | ------------------------- | ------------------------ | ------------------------------- |
| 分配決策發起者   | Token 獨立挑選 Top-K 專家 | 專家主動挑選 Top-K Token | Token 獨立挑選 Top-K 專家       |
| 硬體負載平衡     | 極差（嚴重依賴 Padding）  | 完美（無 Padding 浪費）  | 極優（無損失動態演算法調節）    |
| 推論階段因果律   | 完美支援                  | **無法支援**             | 完美支援                        |
| Token 掉字現象   | 頻繁發生                  | 解決掉字                 | **徹底根除**                    |

### 4.5 推論框架的 Token Sorting 流程

以 vLLM `fused_moe` 實作為例，Token 到 Grouped GEMM 的轉換步驟：

1. **專家索引計算**：執行 Top-K 選取，生成 `topk_ids` 與 `topk_weights`
2. **Token 排序**：呼叫 `moe_align_block_size`，依專家 ID 進行基數排序（Radix Sort）
3. **記憶體重排（Gather）**：從 HBM 非連續 Hidden States 中收集資料，拼湊為依專家排序的連續特徵矩陣
4. **區塊對齊補零**：Tensor Core 需求 `BLOCK_SIZE_M` 對齊（16/32/64），不足者填入零值 Token
5. **Grouped GEMM 核心執行**：讀取 `expert_ids` 指標，映射至各 Thread Blocks
6. **反向重排（Scatter Add）**：乘上 `topk_weights`，將結果寫回原始 Token 位置

> **效能瓶頸**：最嚴重的瓶頸在於 Permute/Unpermute 的**非連續記憶體存取（Uncoalesced Memory Access）**，而非矩陣乘法本身

---

## 5. Apple M3 GPU 微架構與記憶體模型

### 5.1 統一記憶體架構（UMA）

Apple Silicon 採用統一記憶體架構，CPU、GPU 與神經網路引擎共享同一塊高速 LPDDR5 記憶體。理論頻寬 200 GB/s，但實際效能高度依賴存取模式的連續性。

### 5.2 兩級本地記憶體模型

Apple GPU 具備獨特的「兩級本地記憶體模型（Two-Tier Local Memory Model）」：

#### 第一級：暫存器檔案與 SIMD 群組記憶體（Tier 1）

- **容量**：每核心約 **208 KiB**（逆向工程估算）
- **存取速度**：以核心最高時脈速度，延遲趨近於零
- **存取單位**：SIMD-group（32 個執行緒，鎖步執行）**獨佔**存取
- **特殊能力**：可直接與 AMX 硬體矩陣加速器對接

#### 第二級：執行緒群組記憶體（Threadgroup Memory / Tier 2）

- **容量**：M3 架構限制每執行緒群組 **32 KiB**
- **相當於**：CUDA 中的 Shared Memory
- **危險**：具有複雜 Bank 結構，存取不當導致效能崩潰

### 5.3 Bank 衝突機制

Threadgroup Memory 被切分為 **32 個獨立記憶體庫（Banks）**，Bank 映射公式：

$$\text{Bank ID} = \left\lfloor \frac{\text{Byte Address}}{4} \right\rfloor \pmod{32}$$

由於 32 Banks × 4 bytes = **128 bytes 精確循環**，當 SIMD 群組的 32 個執行緒落入相同 Bank 時，硬體強制序列化（Serialization），引發 **N-way Bank Conflict**，最糟情況下延遲飆升 **32 倍**。

### 5.4 M3（G15）與 M1/M2（G13）的架構差異

M3 晶片的 Threadgroup Memory 被硬體研究者形容為「極度不規則的混亂狀態（A hot mess）」：

| 微架構特徵                   | Apple M1/M2 (G13)      | Apple M3 (G15)                        |
| ---------------------------- | ---------------------- | ------------------------------------- |
| 記憶體庫數量                 | 32 個獨立記憶體庫      | 不規則佈局（半週期性特徵）            |
| 連續存取（Stride 1）         | 最佳效能（無衝突）     | 最佳效能（無衝突）                    |
| 偶數步幅（Stride 2）         | 效能衰退約 50%         | **無明顯效能懲罰**                    |
| 極端步幅（Stride 32）        | 最嚴重 32 路序列化衝突 | **效能異常優異（悖離傳統認知）**      |
| 合併寫入（Coalesced Stores） | 支援並鼓勵             | **強烈偏好（Stride 1-4 內表現極佳）** |
| 步幅懲罰區間                 | 所有 32 倍數步幅       | 步幅 8、16、24、32                    |

**關鍵結論**：M3 對完美合併儲存操作（步長 ≤ 4）展現出絕對偏好，內核開發者需針對此特性重新設計資料佈局策略。

---

## 6. Bank 衝突病理學：GEMV 中的序列化根源

### 6.1 衝突根源推導

以 $r_{out} = 64$ 的核心矩陣 $G_{active} \in \mathbb{R}^{r_{in} \times r_{out}}$ 為例，在 Threadgroup Memory 中執行 GEMV：

- 每個執行緒 $t$ 負責計算結果向量的一個元素
- 執行緒需沿著矩陣「行（Column）」方向讀取，計算內積
- 對於 float16（2 bytes/元素），相鄰執行緒（執行緒 0 與執行緒 1）的存取跨度：

$$\text{Stride} = r_{out} \times 2\,\text{bytes} = 64 \times 2 = 128\,\text{bytes}$$

代入 Bank 映射公式：

$$\text{Bank ID}(t) = \left\lfloor \frac{t \times 128}{4} \right\rfloor \pmod{32} = (t \times 32) \pmod{32} = 0$$

**結果**：32 個執行緒（$t = 0, 1, \ldots, 31$）計算出的 Bank ID **全部都是 0**，引發最極端的 **32-way Bank 衝突**。

### 6.2 衝突影響量化

| 情境                | 有效頻寬     | 延遲倍增 |
| ------------------- | ------------ | -------- |
| 零衝突（理想）      | 200 GB/s     | 1×       |
| 32-way 衝突（最糟） | 約 6.25 GB/s | 32×      |
| M3 嚴重散亂存取     | 約 67 GB/s   | 約 3×    |

---

## 7. 記憶體佈局優化策略

針對 $G_{experts}$ 的 Coalesced Load 問題，提出三種主要策略：

### 7.1 策略一：離線張量轉置（Offline Tensor Transposition）

**原理**：在推論前，將 $G_{experts}$ 的儲存形狀從 `[E, r_in, r_out]` 轉換為 `[E, r_out, r_in]`

**效果**：相鄰執行緒存取步長從 $r_{out}$（=64 元素）變為 **1 個元素（2 bytes）**

**零衝突數學保證**：
$$\gcd(\text{Stride}, \text{Banks}) = \gcd(1, 32) = 1$$

32 個執行緒完美映射到 16 個不同 Bank，達成零衝突合併載入。

**注意事項**：轉置必須在**模型權重轉換腳本中離線完成**，避免推論時的 MLX 動態轉置觸發額外 Kernel Dispatch。

### 7.2 策略二：數學互質填充（Mathematical Coprime Padding）

**原理**：在 Threadgroup Memory 每一列末尾填充虛擬元素，強行改變每列的實體記憶體寬度，使步長不再是 32 的倍數。

**核心數學**：32 = $2^5$，任何**奇數**必然與 32 互質。

**具體公式**（以 float32 為例，原寬度 64，填充後寬度 65）：

$$\text{Address}(t) = t \times 65 \times 4 + c \times 4$$

$$\text{Bank ID}(t) = \left\lfloor \frac{t \times 260 + c \times 4}{4} \right\rfloor \pmod{32} = (t \times 65 + c) \pmod{32} = (t + c) \pmod{32}$$

由於 $65 \pmod{32} = 1$，當執行緒索引 $t$ 從 0 遞增至 31 時，Bank ID 精確從 $c$ 循環至 $(c + 31) \pmod{32}$，**無任何重複**。

**BF16 精度的 Padding 策略**（本專案具體實作）：

令 $\Delta_{pad} = 8$，邏輯 Row Stride 從 32 變更為：

$$\text{Stride}_{row} = T_K + \Delta_{pad} = 32 + 8 = 40\,\text{elements}$$

執行緒 $T_{idx}$ 的目標寫入位址偏移量：

$$\text{Offset}_{dst}(T_{idx}) = \left(\left\lfloor \frac{T_{idx}}{8} \right\rfloor \times \text{Stride}_{row}\right) + (T_{idx} \bmod 8) \times 4$$

**優點**：空間開銷微乎其微（增加約 1.5%），實作相對簡單

### 7.3 策略三：XOR 位元交錯映射（XOR-Based Memory Swizzling）

**原理**：在邏輯座標與物理記憶體索引之間建立非線性映射

**公式**：
$$\text{Physical Index} = \text{row} \times \text{width} + (\text{col} \oplus (\text{row} \bmod 32))$$

透過 XOR 運算將資料偽隨機打散到不同 Bank，保證無論循列或循行讀取均避免嚴重衝突。

**優點**：不增加任何空間開銷，保持陣列大小不變

**缺點**：實作複雜，需額外幾個 ALU 週期計算位址

### 7.4 策略對比總表

| 優化策略                 | 邏輯形狀                        | Threadgroup 存取步長 | Bank 衝突狀態       | 實作複雜度                       |
| ------------------------ | ------------------------------- | -------------------- | ------------------- | -------------------------------- |
| Naive Baseline（未優化） | `[E, r_in, r_out]`              | $r_{out}$（例如 64） | **32-way 極端衝突** | 極低（現有代碼）                 |
| Offline Transpose        | `[E, r_out, r_in]`              | 1（完美連續）        | **0 衝突**          | 低（需修改上游 Triton 導出模型） |
| Mathematical Padding     | `[E, r_in, r_out+1]`（例如 65） | 奇數步長             | **0 衝突**          | 中（需修改 Metal Shader 索引）   |
| XOR Swizzling            | `[E, r_in, r_out]`              | 1（邏輯上）          | **0 衝突**          | 高（需實作 MSL 位元邏輯運算）    |

---

## 8. Metal 3 非同步記憶體複製與 TMA 對等替代

### 8.1 NVIDIA TMA 背景

NVIDIA Hopper 架構（H100）的 TMA（Tensor Memory Accelerator）允許透過單一指令非同步地將多維張量切片從 Global Memory 搬移至 Shared Memory，且**不需要 SM 的 ALU 參與**，實現計算與記憶體傳輸的完美重疊（Overlapping）。

### 8.2 Metal 3 的 simdgroup_async_copy

在 Apple Silicon 的 Metal 3 框架中，`metal::simdgroup_async_copy` 指令允許整個 SIMD-group（32 個執行緒）協同發起 DMA 請求，將資料從 device 記憶體空間高效串流至 threadgroup 記憶體空間。

**特性**：

- 非同步執行，ALU 可在等待資料期間繼續計算前一個資料區塊
- 搭配 `simdgroup_barrier(mem_flags::mem_threadgroup)` 確保記憶體一致性
- `simdgroup_barrier` 迫使 SIMD-group 所有執行緒暫停，直到所有非同步寫入操作完成

### 8.3 M3/M4 的特殊考量

| 架構世代          | 非同步複製實作方式                                      | 建議                          |
| ----------------- | ------------------------------------------------------- | ----------------------------- |
| M1/M2（Apple7/8） | 專屬硬體加速單元，極大提升頻寬利用率                    | 積極使用 simdgroup_async_copy |
| M3/M4（Apple9+）  | **透過微碼模擬（Emulation）實現**，延遲特性可能不如預期 | 謹慎評估使用時機              |

**M3 架構建議**：

- 需反覆重用的權重矩陣（如 $G_{experts}$）：**使用** `simdgroup_async_copy` 預先載入
- 單次讀取的向量（如 BS=1 時的 $x_{shared}$）：**繞過** Threadgroup Memory，直接從 Device Memory 發起向量化載入，避免模擬指令的額外開銷

### 8.4 雙重緩衝軟體流水線

```
Threadgroup Memory 中分配兩個 Stage 的空間：
Stage[0] 和 Stage[1]

主迴圈：
  k = 當前 Tile 索引
  1. AMX 計算 Stage[k % 2] 中的 Tile k
  2. 同時非同步預載 Tile k+1 至 Stage[(k+1) % 2]
  3. simdgroup_barrier 確保同步
  4. 切換 stage_idx
```

---

## 9. AMX 協同處理器與 SIMD-Group Matrix API

### 9.1 AMX 硬體概述

Apple Silicon 內建 AMX（Apple Matrix Extensions）協同處理器，等同於 NVIDIA 架構中的 Tensor Cores：

- 專為密集矩陣乘加運算（MMA）設計
- 透過 `<metal_simdgroup_matrix>` 頭檔調用
- 暴露 `simdgroup_matrix` 特殊資料型態

### 9.2 simdgroup_matrix 的硬體語意

宣告 `simdgroup_matrix<bfloat, 8, 8>` **不會**在普通暫存器檔案或 Threadgroup Memory 中分配空間，而是將 64 個 bfloat 元素**打散分佈於 SIMD-group 32 個執行緒的隱藏暫存器中**。

> **重要限制**：所有對 simdgroup_matrix 的操作，必須由 SIMD-group 內 32 個執行緒在**一致的控制流（Uniform Control-Flow）**下協同執行，否則引發未定義行為。

### 9.3 BS=1 GEMV 映射至 AMX 的技術

AMX 為 GEMM（矩陣-矩陣乘法）設計，為將 GEMV 映射到 AMX 上，需對向量 $x_{shared}$ 進行**廣播（Broadcasting）**：

**向量提升流程**：

1. 在 MSL 中宣告 $8 \times 8$ 的 `simdgroup_matrix`
2. 利用 `simdgroup_load` 配合特定步長，將 $x_{shared}$ 的 8 元素切片載入矩陣，並強制在矩陣的 8 個列中**完全複製**
3. 原本的 $1 \times 8$ 向量被轉化為所有列相同的 $8 \times 8$ 矩陣
4. 呼叫 `simdgroup_multiply_accumulate` 執行矩陣乘加運算

**延遲消除的決定性優勢**：
透過 `simdgroup_matrix`，整個 Tucker 收縮運算被強制侷限在每核心 **208 KiB 的 Tier 1 暫存器**內，完全繞過 Tier 2 Threadgroup Memory 的 Bank 衝突風險與額外存取開銷。

**傳統 SIMD FMA 路徑** vs **AMX 暫存器路徑**：

```
傳統路徑：
全域記憶體 → XOR Swizzling 寫入 Threadgroup Memory
           → threadgroup_barrier 同步
           → 讀取至 ALU
           → 乘加運算

AMX 暫存器路徑：
全域記憶體 → simdgroup_load（自動對齊 128-byte 事務邊界）
           → simdgroup_multiply_accumulate（直接在 Tier 1 執行）
           → 無任何 Threadgroup 往返
```

---

## 10. Fused Latent MoE Kernel 完整實作

### 10.1 設計目標

在標準 MLX 模型中，投影、路由、Gather、收縮、輸出投影為獨立算子，每個算子都需將中間結果寫回全域 LPDDR5 記憶體。深度融合能徹底消除這些全域記憶體往返。

### 10.2 硬體常數定義

```c
// 為完美映射至 AMX 的 8x8 區塊，設定 TILE 尺寸為 32x32
// 允許每個 SIMD-group (32 threads) 計算 4x4 個 simdgroup_matrix 區塊
constant uint TILE_M = 32;
constant uint TILE_K = 32;

// Bank Conflict 消除策略：加入 8 個 bfloat (16 Bytes) 的 Padding
// 使 ROW_STRIDE = 32 + 8 = 40，打破 32-Bank 記憶體的週期對齊
constant uint PAD_BFLOAT = 8;
constant uint ROW_STRIDE = TILE_K + PAD_BFLOAT; // = 40
```

### 10.3 非同步張量搬移函數

```c
// 每個 Thread 負責搬移 1 個 bfloat4 (8 Bytes)
// 32 Threads * 8 Bytes = 256 Bytes = 兩條 128-Byte Cachelines
inline void async_load_expert_tile(
    threadgroup bfloat* dst,
    const device bfloat* src,
    uint row_offset, uint col_offset,
    uint ld_src, uint thread_id
) {
    #pragma unroll
    for (uint i = 0; i < TILE_M; i += 4) {
        uint local_row = (thread_id / 8) + i;
        uint local_col = (thread_id % 8) * 4;
        uint src_idx = (row_offset + local_row) * ld_src + (col_offset + local_col);
        uint dst_idx = local_row * ROW_STRIDE + local_col;

        // 呼叫 Metal 3 的 simdgroup_async_copy，模擬 NVIDIA TMA 行為
        simdgroup_async_copy(
            (threadgroup bfloat4*)(&dst[dst_idx]),
            (const device bfloat4*)(&src[src_idx]),
            1 // 複製 1 個 bfloat4 向量
        );
    }
}
```

### 10.4 主內核：Fused Latent MoE GEMV

```c
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_compute>
using namespace metal;

[[kernel]] void fused_latent_moe_gemv(
    const device bfloat* x_shared      [[buffer(0)]],
    const device bfloat* G_experts     [[buffer(1)]],
    const device uint*   expert_indices [[buffer(2)]],
    device bfloat*       out_hidden    [[buffer(3)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint  thread_position_in_threadgroup [[thread_position_in_threadgroup]]
) {
    // 解析動態路由結果
    uint expert_id = expert_indices[threadgroup_position_in_grid.z];
    uint R3 = 256;  // r3 維度
    uint R2 = 1024; // r2 維度

    const device bfloat* expert_base = G_experts + (expert_id * R3 * R2);

    // 在 Threadgroup Memory 中宣告雙重緩衝區（含 Padding 消除 Bank Conflict）
    threadgroup bfloat stages[2][TILE_M * ROW_STRIDE];

    // 宣告 AMX 累加暫存器（float32 精度防止溢出）
    simdgroup_matrix<float, 8, 8> acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        acc[i] = make_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    uint thread_id  = thread_position_in_threadgroup;
    uint row_block  = threadgroup_position_in_grid.x * TILE_M;
    uint num_tiles  = R2 / TILE_K; // 1024 / 32 = 32 個 Tiles
    uint stage_idx  = 0;

    // ── 軟體流水線 Prologue ──────────────────────────────────────
    async_load_expert_tile(stages[0], expert_base, row_block, 0, R2, thread_id);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // ── 軟體流水線 Main Loop ─────────────────────────────────────
    for (uint k = 0; k < num_tiles; k++) {
        uint next_k         = k + 1;
        uint next_stage_idx = (stage_idx + 1) % 2;

        // 1. 非同步預載下一個 Tile（與下方 AMX 計算完全重疊）
        if (next_k < num_tiles) {
            async_load_expert_tile(
                stages[next_stage_idx], expert_base,
                row_block, next_k * TILE_K, R2, thread_id
            );
        }

        // 2. 載入向量 x_shared 並廣播至 AMX 暫存器
        //    M3 架構：繞過 SRAM 直接從 Device Memory 載入，避免 Emulation 開銷
        simdgroup_matrix<bfloat, 8, 8> x_mat[4];
        // （實務上使用特定 thread_id 擷取 x_shared 並廣播填入矩陣）

        // 3. 從無 Bank Conflict 的 Threadgroup Memory 載入 G_experts
        simdgroup_matrix<bfloat, 8, 8> g_mat[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                // 使用 ROW_STRIDE = 40，完美錯開 32-Bank 對齊衝突
                simdgroup_load(
                    g_mat[i][j],
                    stages[stage_idx] + (i * 8 * ROW_STRIDE) + (j * 8),
                    ROW_STRIDE
                );
            }
        }

        // 4. 調用 AMX 硬體進行高吞吐量矩陣乘加（FP32 累加 BF16 輸入）
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                simdgroup_multiply_accumulate(acc[i], x_mat[j], g_mat[i][j], acc[i]);
            }
        }

        // 5. 等待下一個 Tile 非同步搬移完成
        if (next_k < num_tiles) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        stage_idx = next_stage_idx;
    }

    // ── Epilogue：寫回結果 ───────────────────────────────────────
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint out_row = row_block + (i * 8);
        simdgroup_store(
            acc[i],
            out_hidden + out_row,
            1,   // Column Stride = 1（輸出向量）
            threadgroup_position_in_grid.x,
            true // Row-major
        );
    }
}
```

### 10.5 融合內核的架構管線設計

```
執行管線（Pipeline）邏輯：

1. 環境初始化
   └─ 宣告含 Padding/Swizzling 邏輯的 Tier 2 記憶體空間

2. 共享輸入投影融合（x → x_shared）
   └─ U_in 從全域記憶體串流載入，直接在 Tier 1 暫存器中完成
   └─ 結果 x_shared 留在 Tier 1，絕對不寫回全域記憶體

3. 動態路由與 Gather 決策
   └─ 內核內部根據路由邏輯即時計算激活專家索引 e

4. 無衝突 Gather 載入
   └─ 結合 Padding 或 Swizzling 邏輯
   └─ G_active 以完全合併且零衝突方式寫入 Threadgroup Memory

5. 核心專家收縮運算（x_shared → y_shared）
   └─ x_shared 在暫存器中，與無衝突 G_active 進行 GEMV
   └─ 結果 y_shared 保留於暫存器

6. 共享輸出投影與結果回寫（y_shared → y）
   └─ 載入 U_out，與暫存器內 y_shared 進行最後內積
   └─ 最終結果寫回全域記憶體
```

此架構徹底消除三次全域記憶體讀寫延遲，確保 200 GB/s 頻寬專心用於串流模型權重。

---

## 11. MLX JIT 動態編譯整合

### 11.1 mx.fast.metal_kernel 介面

MLX 框架提供 `mx.fast.metal_kernel` API，允許開發者注入自定義 MSL 源碼字串。框架底層機制自動根據 `input_names` 與 `output_names` 補齊 C++ 函數簽名，並智慧綁定 `[[buffer(n)]]` 索引。

### 11.2 JIT 常數注入策略

為降低暫存器壓力（Register Pressure）並減少執行期條件分支：

```python
# Python 端在推論前根據當前層維度動態生成帶硬編碼常數的 MSL 字串
def build_kernel_source(r3: int, r2: int) -> str:
    return f"""
    constant uint R3 = {r3};  // 硬編碼，消除執行期讀取
    constant uint R2 = {r2};  // 硬編碼，消除條件分支
    constant uint TILE_K = 32;
    constant uint PAD_BFLOAT = 8;
    constant uint ROW_STRIDE = TILE_K + PAD_BFLOAT;  // = 40
    // ... 完整內核代碼 ...
    """
```

> **建議**：利用 Metal 4 的非同步編譯 API，在**模型載入階段**預先構建 Pipeline State，避免 Token 生成時的編譯卡頓（Stuttering）

### 11.3 Occupancy Heuristics（佔用率啟發式演算法）

要讓 BF16 精度的運算填滿 AMX 單元管線，每一顆物理 GPU 核心必須被分配**至少 9 個 Threadgroups**：

$$\text{最小 Threadgroup 總數} = 19 \times 9 = 171$$

當佔用率不足（模型維度過小），動態調整 Tile 大小：

| 情況       | Block Size     | 策略                             |
| ---------- | -------------- | -------------------------------- |
| 正常       | $48 \times 48$ | 標準配置                         |
| 佔用率低   | $32 \times 32$ | 縮小以增加 Threadgroup 總數      |
| 佔用率極低 | $16 \times 16$ | 進一步縮小，隱藏 Cache Miss 延遲 |

---

## 12. 效能模型與量化估算

### 12.1 理論效能上限

Apple M3 19 核心 GPU 估算（以 $f \approx 3$ GHz）：

$$\text{FP32 峰值算力} \approx 19 \times 256 \times 3 \times 10^9 \approx 14.6\,\text{TFLOP/s}$$

記憶體頻寬：200 GB/s

$$\text{Roofline 平衡點} = \frac{14.6 \times 10^{12}}{2 \times 10^{11}} \approx 73\,\text{FLOP/Byte}$$

### 12.2 記憶體需求估算

230M 活躍參數（BF16）：

$$\text{單次推論記憶體讀取} = 230 \times 10^6 \times 2\,\text{bytes} = 460\,\text{MB}$$

$$\text{理論最短延遲} = \frac{460\,\text{MB}}{200\,\text{GB/s}} = 2.3\,\text{ms}$$

### 12.3 Bank 衝突對效能的量化影響

| 存取模式           | 有效頻寬             | 每 Token 延遲（估算） |
| ------------------ | -------------------- | --------------------- |
| 完美合併（零衝突） | 200 GB/s             | 2.3 ms                |
| 中等衝突           | 约 67 GB/s（1/3）    | 约 6.9 ms             |
| 32-way 衝突        | 约 6.25 GB/s（1/32） | 约 73.6 ms            |

### 12.4 Tucker-MoE 記憶體流量縮減

以 $d=4096,\; r=256$ 為例，Tucker-MoE 的 Permute 階段記憶體流量縮減：

$$\text{縮減比} = \frac{r}{d} = \frac{256}{4096} = \frac{1}{16} \quad (93.75\%\,\text{流量縮減})$$

從 HBM 讀取 $C_i$ 核心矩陣所消耗的記憶體頻寬縮減比例：

$$\left(\frac{d}{r}\right)^2 = \left(\frac{4096}{256}\right)^2 = 256\,\text{倍縮減}$$

### 12.5 目標效能指標

| 指標             | 目標值   |
| ---------------- | -------- |
| GFLOPS 利用率    | 60–80%   |
| 記憶體頻寬利用率 | 70–90%   |
| 每 Token 延遲    | < 1–2 ms |
| Bank 衝突率      | 趨近 0   |

---

## 13. Tucker-MoE 推論優化的系統效益

### 13.1 消滅大維度碎任務：最大化 Tensor Core 飽和度

在 Decode 階段（Batch Size 極小），計算 $XU$ 與 $HV$ 屬於**無分支的稠密矩陣乘法（Dense GEMM）**：

- 可將全 Batch Token 集中，以統一 Kernel 呼叫填滿所有 SM 資源
- 徹底消除在 $d$ 維度空間中因專家分流導致的算術強度崩塌
- 僅有中間層小維度矩陣 $C_i \in \mathbb{R}^{r \times r}$ 需面臨碎任務挑戰

### 13.2 數量級縮減 Token Sorting 與 Permute 記憶體讀寫

| 操作                    | 傳統 MoE         | Tucker-MoE       |
| ----------------------- | ---------------- | ---------------- |
| Permute/Gather 資料維度 | $d$（例如 4096） | $r$（例如 256）  |
| HBM 記憶體流量          | 100%（基準）     | 6.25%（1/16）    |
| Radix Sort 耗時         | 瓶頸             | 可被計算指令隱藏 |

### 13.3 顯著簡化 Router 計算開銷

Router 直接以低維度特徵 $Z \in \mathbb{R}^{N \times r}$ 進行 Top-K 評分，計算量依 $r/d$ 比例大幅縮減，並在系統架構上實現「特徵抽取」與「路由分配」的底層統一。

---

## 14. Tucker-MoE 底層工程挑戰

### 14.1 挑戰一：Kernel Fusion 的極限與暫存器溢出

嘗試將 `[Dense-U] → [Grouped GEMM-C] → [Dense-V]` 融合為巨型單一 Kernel 時：

**問題**：

- Token Sorting 依賴全域同步（Global Synchronization）或 Warp Shuffle 指令
- 這類指令與 GEMM 的非同步 TMA 在資源排程上易產生衝突
- 強制融合導致**暫存器溢出（Register Spilling）**，SM 駐留率（Occupancy）直線下滑

**建議解法**：拆解為兩個高駐留率的特化 Kernel

```
Kernel 1：Dense_U_and_Permute_Kernel
  ├─ 核心計算 XU
  └─ 融合 Dense GEMM 與 Scatter Write，將低秩特徵 Z 寫入 HBM 連續位置

Kernel 2：Grouped_C_and_Dense_V_Kernel
  ├─ 執行 Z × C_i 分組計算（Grouped GEMM）
  ├─ 中間結果保留在 Shared Memory
  ├─ 同一 Thread Block 直接載入 V 完成升維
  └─ Scatter Add 寫回輸出張量
```

### 14.2 挑戰二：低維特徵的記憶體對齊與量化複雜度

**問題**：主流框架（DeepGEMM、vLLM）已全面轉向 FP8/INT8 區塊級量化（Block-wise Quantization）。在 Tucker-MoE 資料流中，$U$ 矩陣的輸出 $Z$ 既是 $C_i$ 的輸入，又牽涉量化縮放因子的重新校準。

若低秩維度 $r$ 設定過小（例如小於 128），無法滿足 NVIDIA SM90（Hopper）與 SM100（Blackwell）對 FP8 TMA 載入的連續記憶體對齊要求。

**建議解法**：

- 強制確保低秩維度 $r$ 是硬體對齊邊界（128 或 256）的整數倍
- 由於所有 $C_i$ 矩陣形狀完全相同（皆為 $r \times r$），採用 **Batched GEMM** 策略替代繁重的 Grouped GEMM 泛用實作，以最直接的記憶體映射方式榨乾硬體效能

---

## 15. 測試、驗證與診斷計畫

### 15.1 Microbenchmarks 設計

針對不同資料佈局（行主序、列主序、Padding、XOR Swizzling）設計微基準：

1. 不同步幅（Stride）下的純載入測試：測量吞吐與延遲
2. 硬編碼 32 執行緒存取模式，觀察不同 Stride 下的效能曲線
3. 比較固定模式下的執行時間與載入次數，評估 Bank 衝突影響

### 15.2 Profile 指標收集

使用 **Xcode GPU Frame Capture** 與 **Metal System Trace** 收集：

| 指標                              | 工具                           | 目的                             |
| --------------------------------- | ------------------------------ | -------------------------------- |
| Bank 衝突計數                     | Metal Trace（或效能熱圖）      | 確認衝突是否消除                 |
| 記憶體交易數                      | Metal System Trace             | 對照理論最小值，分析瓶頸         |
| SIMD 利用率                       | Xcode GPU Frame Capture        | 確認 simdgroup_matrix 指令活躍度 |
| Execution Stall vs Compute Active | GPU Counters                   | 判斷是否記憶體或 ALU 瓶頸        |
| GPU 時間線與記憶體活動            | Instruments Metal System Trace | 全面資源使用分析                 |

### 15.3 驗證結果

- 確保在測試例上實現的吞吐量與模型規模相稱
- 若可能，與 CPU 或 M2 做基準比較，確定 M3 的相對提升
- 驗證每 Token 延遲目標（< 1–2 ms）

---

## 16. 進化演算法優化展望

### 16.1 OpenEvolve 自動化 GPU 內核優化

基於進化程式設計（Evolutionary Programming）的 OpenEvolve 系統已被應用於自動發掘 MLX 框架下針對 Apple Silicon 的最佳化 Metal 內核。

**LLM 驅動的優化循環**：

1. 變異：大型語言模型生成候選內核修改
2. 評估：實際執行效能測量
3. 選擇：保留最優個體作為下一代基礎

**自動發現的優化策略**：

- 完美的 SIMD 向量化長度（例如發掘出 `vec<T, 8>` 是特定注意力頭維度的最優解）
- 兩階段線上 Softmax 的記憶體融合
- 針對特定硬體架構的非正規記憶體存取模式

**實驗結果**：在針對 Qwen3-0.6B 等模型的優化實驗中，自動演化的內核在生成任務上展現超過 **100% 的效能提升幅度**。

### 16.2 面向未來硬體的展望

對於 Hybrid Mamba-TuckerMoE 這類依賴狀態空間掃描與動態路由的複雜架構：

- 結合本報告提出的 AMX 手動流水線設計作為**進化演算法的基準種子（Baseline Seed）**
- 在未來 Apple Silicon（如預計搭載更強大神經網路加速器的 **M5 系列**）上，探索連硬體工程師都未曾預料到的非傳統高效存取模式

---

## 17. 風險評估與替代策略

### 17.1 風險矩陣

| 風險                                     | 發生機率 | 影響程度 | 緩解策略                        |
| ---------------------------------------- | -------- | -------- | ------------------------------- |
| M3 simdgroup_async_copy 模擬開銷高於預期 | 中       | 高       | 對單次讀取向量直接繞過 SRAM     |
| Kernel Fusion 引發 Register Spilling     | 高       | 高       | 拆解為兩個特化 Kernel           |
| 低秩維度不滿足 FP8 對齊要求              | 中       | 中       | 強制 r 為 128/256 整數倍        |
| Occupancy 不足（維度過小）               | 中       | 中       | 動態縮小 Tile 大小              |
| M3 記憶體控制器異常行為影響 Padding 效果 | 低       | 中       | 同時使用 XOR Swizzling 作為備案 |

### 17.2 替代策略

**預取（Prefetching）**：

- 提前將下一批權重從 Device Memory 載入 Threadgroup
- 遮蓋記憶體延遲，配合雙緩衝技術使用

**雙緩衝（Double Buffering）**：

- 維護兩組緩衝區，一組計算、一組預載，交替使用
- 在一次 `threadgroup_barrier` 中同時準備下一個資料塊

**專家權重複製**：

- 若少數專家頻繁被訪問，複製其小塊權重供多個子組獨立運算
- 雖增加 Threadgroup 記憶體使用，但降低競爭

**稀疏打包（Sparse Packing）**：

- 僅存儲非零專家權重，使用 CSR/CSC 表示
- 減少記憶體傳輸，但增加索引計算開銷

**回退至標準向量化**：

- 若 simdgroup_matrix 無法使用，回退至手寫向量化內積
- 使用 `float4` 向量操作作為兼容方案

---

## 18. 結論與最佳實踐藍圖

### 18.1 三層優化策略總結

```
第一層：資料佈局改造（消除 Bank 衝突）
┌─────────────────────────────────────────┐
│ 優先：離線張量轉置 [E, r_out, r_in]      │
│ 次選：數學互質填充（+8 bfloat16 元素）    │
│ 備選：XOR 位元交錯映射                    │
└─────────────────────────────────────────┘

第二層：深度內核融合（消除全域記憶體往返）
┌─────────────────────────────────────────┐
│ 透過 MLX mx.fast.metal_kernel JIT 機制   │
│ 將「共享投影→路由→Gather→收縮→輸出」     │
│ 融合進單一 Metal Shader Dispatch         │
└─────────────────────────────────────────┘

第三層：暫存器層級 AMX 計算（消除 Threadgroup 延遲）
┌─────────────────────────────────────────┐
│ 利用 simdgroup_matrix 將向量廣播為矩陣    │
│ Tucker 降階運算完全侷限在 208 KiB Tier 1 │
│ 完全迴避 Tier 2 Threadgroup Memory 開銷  │
└─────────────────────────────────────────┘
```

### 18.2 關鍵技術決策清單

- [x] `G_experts` 採用 Padding（ROW_STRIDE = 40）或離線轉置消除 Bank 衝突
- [x] `simdgroup_async_copy` + 雙重緩衝軟體流水線隱藏 Gather 延遲
- [x] `simdgroup_matrix` 將 GEMV 映射至 AMX 暫存器層級計算
- [x] `mx.fast.metal_kernel` JIT 注入帶硬編碼常數的融合 Kernel
- [x] Metal 4 非同步編譯 API 在模型載入時預構建 Pipeline State
- [x] Occupancy Heuristics 動態調整 Tile 大小（目標 ≥171 Threadgroups）
- [x] BF16 精度，float32 內部累加防止數值溢出

### 18.3 預期效能收益

| 優化項目                           | 預期收益                                           |
| ---------------------------------- | -------------------------------------------------- |
| 消除 32-way Bank 衝突              | 最高 **32×** 記憶體存取延遲改善                    |
| Tucker 降維後路由                  | Permute 記憶體流量降低 **93.75%**（d=4096, r=256） |
| Fused Kernel（消除全域記憶體往返） | 消除 3 次 HBM 讀寫往返                             |
| AMX 暫存器計算                     | 消除 Threadgroup 往返延遲，提升核心佔用率          |
| JIT 常數注入                       | 降低暫存器壓力，減少條件分支開銷                   |
| **綜合預期**                       | **每 Token 延遲 < 2 ms，GFLOPS 利用率 60–80%**     |

### 18.4 最終結語

將處於嚴峻 Memory-bound 狀態的 Hybrid Mamba-TuckerMoE 架構移植至 Apple Silicon，其核心挑戰在於跨越純粹演算邏輯的範疇，深入探究硬體的物理極限。

本報告詳盡解構了 Apple M2 與 M3 世代在 Threadgroup Memory 佈局上的致命差異，提出透過引入 16 Bytes（$\Delta_{pad} = 8$ bfloat16 元素）Padding 打破記憶體庫對齊週期的通用數學解法。透過精確使用 `metal::simdgroup_async_copy` 與 `simdgroup_barrier` 構建的雙重緩衝軟體流水線，成功在 Apple 平台上複製了 NVIDIA Hopper TMA 的非同步資料預取能力。最後，藉由將資料廣播並強行匯入 AMX 協同處理器（透過 `<metal_simdgroup_matrix>` API），確保消除 Bank Conflict 後的每一筆 BF16 資料，都能以硬體所能支援的最高乘加運算吞吐量進行消化。

這種結合底層硬體特性、精確物理記憶體轉置與高階 JIT 動態編譯的綜合性內核優化策略，為在統一記憶體架構上運行 BS=1 的大型 MoE 模型，樹立了全新的效能標竿。

---

## 附錄：資料流程圖

```
輸入 x  ──[U_in × Tucker 降維]──→  x_shared (Tier 1 暫存器)
                                          │
                                    [Router 路由]
                                          │
                                    ┌─────▼─────┐
                                    │  Gather    │ ← G_experts (Padding 佈局)
                                    │  (無衝突)  │
                                    └─────┬─────┘
                                          │
                            [Async Load + 雙重緩衝]
                                          │
                                    ┌─────▼─────┐
                                    │ Threadgroup│
                                    │  Memory   │ (ROW_STRIDE=40，零衝突)
                                    └─────┬─────┘
                                          │
                              [simdgroup_matrix × AMX]
                                          │
                                    ┌─────▼─────┐
                                    │ Tier 1 暫存│ y_shared (float32 累加)
                                    │    器     │
                                    └─────┬─────┘
                                          │
                                    [U_out × 升維]
                                          │
                                     輸出 y  ──→ 全域記憶體
```

---

_本報告整合自四份研究文件：《Metal 3 內核優化與記憶體搬移》、《MoE Router 推論優化與 Tucker-MoE》、《執行摘要》、《Metal 內核優化 Mamba-TuckerMoE》，涵蓋所有關鍵技術細節與實作建議。_
