# Tucker-MoE 有效性論證報告

> 主題：從原本的 dense MoE 出發，逐步推導出本專案的 Tucker 分解 MoE，論證它為何有效；
> 並回答核心疑問：**論文是「先有 dense → 做 Tucker → 截斷」，我們反過來「直接訓練因子」可以嗎？**
>
> 對應程式碼：[`pre-train/sft_cot_bundle/scripts/model.py`](../pre-train/sft_cot_bundle/scripts/model.py)（`TritonTuckerMoE`，含反向傳播）、
> [`mamba3_mlx/mlx_model/tucker_moe.py`](../mamba3_mlx/mlx_model/tucker_moe.py)（MLX 推論版）。

---

## 0. TL;DR

1. Dense MoE 的參數隨「專家數 E × 輸入維 × 輸出維」線性爆炸；本質上是一個 3D 權重張量 `W ∈ ℝ^(E × dᵢₙ × dₒᵤₜ)`。
2. 把 `W` 做 **Tucker 分解**（三個 mode 各配一個因子矩陣 + 一個小 core），就得到 `U_expert / U_in / core / U_out`——也就是你說的「三個矩陣乘積」。
3. 關鍵收益：**只有 `U_expert (E×r1)` 和 router 隨 E 成長，三個大張量 `U_in / core / U_out` 被所有專家共享**。多加一個專家的邊際成本只有 `r1`（本專案 = 4）個參數。這正是「417M 參數 ≈ 2.4B dense-equivalent、壓縮 82.87%」的機制。
4. **「反過來」(from-scratch 直接訓因子) 是可以的，而且在「從零訓練、受記憶體預算限制」的情境下比 post-hoc 壓縮更好**——因為你直接對「任務損失」做受限最佳化，而不是對某個既有 dense 權重做「重建誤差」最佳化。代價是：失去 warm-start、優化地形非凸，必須靠 **正交初始化 + 中間 RMSNorm + router 溫度退火 + 輔助損失** 來補（本專案全部都做了）。

---

## 1. 從原本的 Dense MoE 出發

一個標準（dense）的 MoE 專家層，對輸入 `x ∈ ℝ^dᵢₙ`：

```
router:   p = softmax(x · W_router)          # W_router ∈ ℝ^(dᵢₙ × E)
top-k:    選出 k 個專家及其機率 p_k
expert e: yₑ = x · Wₑ                         # Wₑ ∈ ℝ^(dᵢₙ × dₒᵤₜ)，每個專家一塊獨立全秩權重
output:   out = Σ_k p_k · y_{eₖ}
```

把所有專家權重疊起來，就是一個三維張量：

```
W ∈ ℝ^(E × dᵢₙ × dₒᵤₜ)
```

**問題：參數隨 E 線性爆炸。** 以本專案 Transformer 區塊裡的 `gate_proj` 為例（`E=8, dᵢₙ=d_model=768, dₒᵤₜ=d_ff=4608`）：

```
P_dense(experts) = E · dᵢₙ · dₒᵤₜ = 8 · 768 · 4608 = 28,311,552 ≈ 28.3M
```

而且這只是「一個投影」。要靠「加專家」換容量，每多一個專家就要付 `dᵢₙ·dₒᵤₜ = 3.5M` 參數——在 16GB M2 上很快撐爆。

**觀察：這些專家彼此高度相關。** 它們處理同一個語意空間，輸入/輸出方向大量重疊，沒必要每個都用一塊獨立全秩矩陣。這就是低秩分解的切入點。

---

## 2. 三步壓縮：為什麼正好是 Tucker

我們對 `W ∈ ℝ^(E × dᵢₙ × dₒᵤₜ)` 的三個 mode 分別做低秩約束。

### Step 1 — 輸入 mode 低秩：共享輸入子空間 `U_in`
所有專家其實只用到輸入空間的一個 `r3` 維子空間。先投影：
```
x_shared = inner_norm(x · U_in)              # U_in ∈ ℝ^(dᵢₙ × r3)，r3=256
```
> dᵢₙ=768 → r3=256，立刻把後續所有運算的輸入維壓掉 3×。`inner_norm`(RMSNorm) 是關鍵的數值穩定點（見 §3.4）。

### Step 2 — 輸出 mode 低秩：共享輸出子空間 `U_out`
所有專家的輸出也落在一個 `r2` 維子空間，最後再升回 `dₒᵤₜ`：
```
out = x_core · U_out + bias                  # U_out ∈ ℝ^(r2 × dₒᵤₜ)，r2=1024
```

### Step 3 — 專家 mode 低秩：core 共享、專家只用 `r1` 係數區分
中間的 `r3 → r2` 變換是「專家特定」的，但不讓每個專家自由：它們**共用一個 core**，每個專家只用一個 `r1` 維係數向量去組合 core：
```
G[e] = Σ_{a=1}^{r1} U_expert[e, a] · core[a]    # core ∈ ℝ^(r1 × r3 × r2)，U_expert ∈ ℝ^(E × r1)，r1=4
x_core = x_shared · G[e]                          # (r3) · (r3 × r2) → (r2)
```

### 合起來：這就是 Tucker 分解
把三步代回，每個專家的等效 dense 權重是
```
Wₑ = U_in · G[e] · U_out
```
逐元素寫出來，整個張量 `W` 就是標準 Tucker：

```
W[e, i, j] = Σ_{a,b,c} U_expert[e,a] · U_in[i,b] · core[a,b,c] · U_out[c,j]
            └ mode-1 因子 ┘ └ mode-2 因子 ┘ └─ core ─┘ └ mode-3 因子 ┘
multilinear rank = (r1, r3, r2) = (4, 256, 1024)
```

| 對應 | Tucker 名稱 | 本專案張量 | 形狀 |
|---|---|---|---|
| mode-1 (專家軸) 因子 | `U⁽¹⁾` | `U_expert` | `E × r1` = 8×4 |
| mode-2 (輸入軸) 因子 | `U⁽²⁾` | `U_in` | `dᵢₙ × r3` = 768×256 |
| mode-3 (輸出軸) 因子 | `U⁽³⁾` | `U_out` | `r2 × dₒᵤₜ` = 1024×4608 |
| 核心張量 | `𝒢` | `core` | `r1 × r3 × r2` |

> 你說「只是三個矩乘積」——正確，而且這正是它的優點：Tucker 把一個本來 O(E·dᵢₙ·dₒᵤₜ) 的張量，拆成幾個小因子的連乘。**唯一偏離純線性 Tucker 的地方是中間插了 `inner_norm`**，這是刻意加入的非線性/正規化（見 §3.4），不影響上面的分解直覺。

---

## 3. 為什麼這樣很好（逐步論證）

### 3.1 邊際專家成本 = r1
把 Tucker 版參數逐項列出（`gate_proj`，E=8）：

| 張量 | 公式 | 參數量 | 隨 E 成長？ |
|---|---|---:|:---:|
| `U_in` | dᵢₙ·r3 = 768·256 | 196,608 | ✗ 共享 |
| `U_out` | r2·dₒᵤₜ = 1024·4608 | 4,718,592 | ✗ 共享 |
| `core` | r1·r3·r2 = 4·256·1024 | 1,048,576 | ✗ 共享 |
| `U_expert` | E·r1 = 8·4 | 32 | ✓ **線性，但係數只有 r1=4** |
| `router` | dᵢₙ·E = 768·8 | 6,144 | ✓ |
| `bias` | dₒᵤₜ | 4,608 | ✗ |
| **合計** | | **≈ 5.97M** | |

對比 dense 28.3M → **壓縮 4.74× (78.9%)**。**重點不在這個倍率，而在「斜率」**：多加一個專家只要 `r1 + dᵢₙ = 4 + 768` 個參數，幾乎免費。

### 3.2 dense-equivalent 隨 E 放大而暴增（417M ↔ 2.4B 的來源）
把專家數想成 E=64（更高容量）：

```
Dense  : 64 · 768 · 4608 = 226,492,416 ≈ 226M
Tucker : (U_in+U_out+core 不變 5.96M) + U_expert(64·4=256) + router(49,152) ≈ 6.02M
壓縮倍率 ≈ 37.6×
```

「dense-equivalent」= 若把全模型所有 `W[e,i,j]` 真的展開成 dense MoE 的總參數量。本專案報告 **417M 實際參數 ≈ 2,434M dense-equivalent（壓縮 82.87% = 1 − 417/2434）**。機制就是：**容量隨 E、隨各層投影累加而成長，但實際存的因子幾乎不動。**

### 3.3 直擊 decode 瓶頸：記憶體頻寬
本專案的核心結論是「單流 decode 是 GPU-dispatch / 記憶體頻寬 bound」（見 `memory/project_decode_wall.md`）。Tucker 的收益**首要是參數/權重的記憶體佔用與每 token 需串流的權重量**，而不是 FLOP：

- FLOP 上，top_k=2 的 dense 與 Tucker 同數量級（都要做幾次矩乘）。
- 但 **權重足跡縮小 5–38×**，模型才塞得進 16GB；且共享的 `core/U_out` 常駐，避免 dense MoE「存 E 塊全秩權重」的記憶體壓力。

也就是說 Tucker 不是用來省算力，是用來**省記憶體與頻寬**——剛好打在你的瓶頸上。

### 3.4 數值與優化穩定性（這是 from-scratch 能成立的前提）
程式碼 [model.py:305-310](../pre-train/sft_cot_bundle/scripts/model.py#L305-L310) 不是隨便初始化：
- `U_in`, `U_out` 用 **正交初始化** → 因子矩陣保持等距，避免連乘造成的梯度爆炸/消失。
- `inner_norm`（RMSNorm 在 `x_shared` 上）→ 控制 `r3` 維瓶頸的尺度，讓 core 的條件數可控。
- `core`, `U_expert` 用 xavier → 維持前向方差。

這些都是為了讓「直接訓練因子連乘」這件非凸的事在數值上站得住腳。

---

## 4. 核心問題：Post-hoc 壓縮 vs. From-scratch（反過來可以嗎？）

你看到的論文流程是 **post-hoc 壓縮**：
```
(1) 訓練好一個 dense W
(2) HOSVD / HOOI 做 Tucker 分解
(3) 依奇異值大小截斷 rank
(4) 微調補回精度
```
你做的是 **from-scratch / 結構化參數化**：一開始就把模型參數定義成 `U_in / core / U_out / U_expert`，隨機（正交）初始化，**直接用任務損失訓練這些因子，從不展開 dense W**。

### 結論先講：可以，而且在你的情境更好。理由如下。

### 4.1 兩者的假設空間（hypothesis class）完全相同
不論你是「分解+截斷一個 dense W」還是「梯度下降訓練因子」，能表示的 `W` 都恰好是
```
ℳ_r = { 所有 multilinear rank ≤ (r1, r3, r2) 的張量 }
```
**可達集合一模一樣**。所以表達力沒有損失——from-scratch 不會因為「沒先有 dense」而少了什麼能表示的函數。

### 4.2 兩者最佳化的「目標」不同 → from-scratch 反而更對
- Post-hoc 截斷（HOSVD/HOOI）最佳化的是 **重建誤差** `‖W − Ŵ‖_F`（且 Tucker 的 HOSVD 只是 *準*最優，要 HOOI 迭代才更好；不像矩陣 SVD 有 Eckart–Young 的精確最優）。
- 但 **重建誤差 ≠ 任務損失**。一個 dense `W` 裡有些方向 norm 很大卻對任務無用，也有些 norm 小卻關鍵；按奇異值截斷會砍掉「低能量但任務重要」的方向。
- **From-scratch 直接對任務損失在 `ℳ_r` 上做受限最佳化**，找的是「對任務最好的低秩 W」，而不是「對某個既有 dense W 最好的低秩近似」。**你把對 proxy 的最優保證，換成對真正目標的直接最優化——這是划算的交換。**

### 4.3 理論上站得住腳
- 這正是「低秩因子化訓練」的標準設定，可視為在固定 multilinear rank 流形上的（黎曼）最佳化。
- 過參數化的因子化形式，梯度下降有**隱式低秩/最小範數偏好**；在類 RIP 條件下低秩矩陣分解「無虛假局部極小」(Burer–Monteiro 系列結果)。雖然張量情形理論更弱，但實務上配合好初始化與正規化普遍可訓。

### 4.4 真正的代價與風險（你要守的條件）
| 風險 | 說明 | 本專案的緩解 |
|---|---|---|
| **失去 warm-start** | post-hoc 從已知好的 dense 出發；from-scratch 冷啟動，更依賴優化找到好盆地 | 正交初始化 + RMSNorm + LR warmup |
| **非凸 + gauge 冗餘** | `(U_in, core)` 與 `(U_in R, R⁻¹·core)` 給同一個 W → 平坦方向；weight decay 作用在因子 ≠ 作用在 W | 影響有限；必要時對 W 而非因子做正則 |
| **rank 不足會欠擬合** | 只有當任務最優的低秩 W 真的存在於 `ℳ_r`，from-scratch 才追得到 | 需做 rank ablation（§5）確認 r1/r2/r3 夠用 |
| **MoE routing 要從零學** | router 與專家分化都得自己長出來 | router **溫度退火 2.0→0.5**（[model.py:113-120](../pre-train/sft_cot_bundle/scripts/model.py#L113-L120)）+ **load-balance / z-loss**（[model.py:321-332](../pre-train/sft_cot_bundle/scripts/model.py#L321-L332)） |
| **topk 不可微** | 哪些專家被選沒有梯度（見反向傳播：只回 `dx_shared, dG, dprobs`） | 靠 lb/z 輔助損失提供 router 訓練訊號 |

### 4.5 什麼時候反而該用 post-hoc？
若你**已經有**一個訓練好的強 dense 模型、只想壓縮部署，那 post-hoc（warm-start）比較省事。
但你的情境是「**在 16GB 記憶體預算下從零訓練自己的模型**」——根本不該、也付不起先去訓一個 2.4B dense 再壓。**From-scratch 因子化是正解，而且省掉了整個最貴的步驟。**

---

## 5. 如何「證明」它真的有效（建議的實驗 / ablation）

要把上面的論證變成可呈現的證據（適合放 paper §ablation）：

1. **Rank–容量 Pareto 前緣**：掃 `r1 ∈ {2,4,8}`、`r2 ∈ {512,1024,2048}`、`r3 ∈ {128,256,512}`，畫「驗證損失 vs 參數量」。證明在固定參數預算下 rank 分配是有效率的，且現有 (4,256,1024) 不是欠擬合。
2. **From-scratch vs Post-hoc 對照**：
   - (a) 同參數預算的真・dense MoE；
   - (b) 訓練好 (a) 後做 Tucker 截斷 + 微調；
   - (c) 本專案 from-scratch 因子化。
   預期 **(c) ≥ (b)**（§4.2 的直接論點），且 (c) 完全不需要訓練 (a)。
3. **有效秩檢查**：對學到的 `G[e] = U_expert[e]·core` 做 SVD，看奇異值譜是否「用滿」了 r2——若大量奇異值≈0 代表 rank 給太多（可再壓）；若尾巴仍肥代表沒浪費容量。
4. **專家分化檢查**：畫 router 對各 category 的負載分布，確認 lb-loss 真的讓專家分工（否則 MoE 退化成單專家）。
5. **端到端指標**：壓縮率 + perplexity + CoT 任務正確率，對齊 `CLAUDE.md` 報告的 417M/2.4B/82.87%。

---

## 6. 結論

- Dense MoE 是一個 `(E × dᵢₙ × dₒᵤₜ)` 張量，參數隨專家數爆炸。
- 對三個 mode 各做低秩約束 = **Tucker 分解**，得到你看到的「三矩陣 + core」連乘；大張量被所有專家共享，**邊際專家成本只有 r1**，這就是 417M↔2.4B 的來源，且直擊 decode 的記憶體頻寬瓶頸。
- **「反過來」直接訓練因子是合法且更優的**：假設空間與 post-hoc 相同，但它直接最佳化任務損失而非重建誤差，並省掉先訓 dense 的天價步驟。
- 代價是冷啟動 + 非凸地形 + routing 要從零學——靠**正交初始化、inner-norm、溫度退火、lb/z 輔助損失**補齊，這些本專案都已內建。
- 要「證明」，跑 §5 的 rank ablation 與 from-scratch vs post-hoc 對照即可。
