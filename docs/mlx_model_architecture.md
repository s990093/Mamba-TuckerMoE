# `mamba3_mlx/mlx_model/` 完整技術文件

> 版本：2026-06-14  
> 適用對象：第一次閱讀這份 codebase 的開發者

---

## 目錄

1. [整體架構圖](#1-整體架構圖)
2. [ops.py — 基礎工具函式](#2-opspy--基礎工具函式)
3. [tucker_moe.py — Tucker 分解 MoE 線性層](#3-tucker_moepy--tucker-分解-moe-線性層)
4. [mamba_block.py — Mamba-3 SSM Block](#4-mamba_blockpy--mamba-3-ssm-block)
5. [transformer_block.py — Transformer 注意力塊](#5-transformer_blockpy--transformer-注意力塊)
6. [scan_metal.py — Chunk-Parallel SSM 掃描](#6-scan_metalpy--chunk-parallel-ssm-掃描)
7. [fused_ssm_step.py — Metal GPU Kernel 深度解說](#7-fused_ssm_steppy--metal-gpu-kernel-深度解說)
8. [hybrid_model.py — 最頂層模型](#8-hybrid_modelpy--最頂層模型)
9. [static_decode.py — 靜態圖 Decode 外殼](#9-static_decodepy--靜態圖-decode-外殼)
10. [weights.py — Checkpoint 載入](#10-weightspy--checkpoint-載入)
11. [推理速度累積效果對照](#11-推理速度累積效果對照)
12. [Decode 單步完整資料流](#12-decode-單步完整資料流)

---

## 1. 整體架構圖

```
輸入 token ids
      │
      ▼
┌─────────────────────────────────────────┐
│           hybrid_model.py               │
│  embed → TrueHybridMamba → norm → head  │
│                                         │
│  TrueHybridMamba 內部層堆疊：            │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │ Mamba3Block  │  │ TransformerBlock │ │
│  │ (×24 層)     │  │ (×6 層，交錯)     │ │
│  └──────┬───────┘  └────────┬─────────┘ │
│         │                   │            │
│         ▼                   ▼            │
│  ┌────────────┐    ┌────────────────┐   │
│  │ tucker_moe │    │  tucker_moe    │   │
│  │  (MoE 層)  │    │   (FFN 層)     │   │
│  └────────────┘    └────────────────┘   │
│         │                               │
│         ▼                               │
│  ┌────────────┐                         │
│  │ scan_metal │                         │
│  │ (SSM 掃描) │                         │
│  └────────────┘                         │
└─────────────────────────────────────────┘
      │
      ▼  (推理加速外殼)
┌─────────────────┐
│ static_decode   │  ← 把上面全部 compile 成單一 GPU graph
└─────────────────┘
      │
      ▼
  fused_ssm_step.py  ← Metal GPU kernel（最底層加速）
  ops.py             ← 基礎工具函式
  weights.py         ← checkpoint 載入
```

**層數統計（預設 config）：**
- `num_layers = 6`，`mamba_ratio = 4`
- 每個 cycle：4 個 Mamba + 1 個 Transformer = 5 層
- 共 6 cycles = **24 個 Mamba block + 6 個 Transformer block = 30 層**

---

## 2. `ops.py` — 基礎工具函式

所有模組共用的數學積木。

### 激活函式

```python
silu(x)        = x * sigmoid(x)       # Mamba gate 和 FFN 用
softplus(x)    = log(1 + e^x)         # 確保 dt > 0（時間步長必須正數）
scaled_tanh(x, s) = tanh(x/s) * s    # 壓縮 logits，防數值爆炸
```

`scaled_tanh` 用 `2*sigmoid(2x)-1` 近似 tanh，速度更快且數值穩定。

### RMSNorm

Root Mean Square Layer Normalization，比 LayerNorm 更快（不算 mean）：

```
rms(x) = sqrt( mean(x²) + ε )
output = x / rms(x) * weight
```

實作優先使用 `mx.fast.rms_norm`（MLX 內建 Metal kernel），舊版 MLX 退回手動 float32。

### LayerScale

每層輸出前乘一個可學習的縮放係數 γ，初始值極小（1e-2），讓訓練初期像是恆等映射，穩定梯度：

```python
output = x * gamma   # gamma 初始 ≈ 0.01
```

### RoPE（旋轉位置編碼）

給向量加入「位置資訊」，不直接加，而是旋轉：

```python
apply_rope_with_sc(x, sin_a, cos_a)
# x: (..., N, R)  → 把 N 維兩兩一組，各自旋轉 angle 度
# 輸入預計算好的 sin/cos，避免同一組角度算兩次（B 和 C 共用）
```

---

## 3. `tucker_moe.py` — Tucker 分解 MoE 線性層

**取代模型中所有普通 Linear 層**，用張量分解 + 混合專家同時達到壓縮和靈活。

### 直覺理解

想像你有 8 位廚師（experts），每位廚師有自己的拿手菜。
給你一道菜（token），Router 評估這道菜「最像哪 2 位廚師的風格」，
然後把這 2 位廚師的食譜按比例混合，產生這道菜的專屬食譜。

普通 Linear 用一個固定食譜，Tucker MoE 為每個 token 動態客製食譜。

### 數學公式

```
原本 Linear：y = x @ W   (W 是固定的大矩陣)

Tucker MoE：
  1. Router:  logits = x @ router.weight.T   → (E,)
  2. capped   = scaled_tanh(logits, 10.0)
  3. probs    = softmax(capped / temp)
  4. top_k    = argpartition(-probs)[:2]       → top-2 expert 索引
  
  5. 合成每位專家的特徵矩陣 G：
     G[e]  = einsum("r,rst->st", U_expert[e], core)   (r3, r2)
  
  6. 加權混合：
     G_w   = Σ_k prob[k] * G[top_k[k]]                (r3, r2)
  
  7. 投影：
     x_shared  = inner_norm(x @ U_in)                  (r3,)
     x_core    = x_shared @ G_w                        (r2,)
     out       = x_core @ U_out + bias                 (dim_out,)
```

**關鍵壓縮比**：原本需要 `(dim_in, dim_out)` 個參數，
現在只需要 `E*r1 + dim_in*r3 + r2*dim_out + r1*r3*r2`，壓縮約 82.87%。

### Prefill 路徑（長序列，B_flat 大）

```
問題：對每個 token 都 gather G[top_k] → 中間量 1 GB
解法：一次大矩陣乘把所有 expert 全算完再 gather

G_flat = G.transpose(1,0,2).reshape(r3, E*r2)  ← 預計算，load 後一次
xG_flat = x_shared @ G_flat                     → (B, E*r2)
然後按 top_k 索引取出，記憶體峰值從 1 GB → 32 MB
```

### Decode 路徑（單 token，B_flat=1）

```
直接取 top-2 expert，對這 2 個 G 加權
G_weighted = G[e0] * p0 + G[e1] * p1
x_core = x_shared @ G_weighted   → (r2,)
```

搭配 `mx.compile`，約 1.8-2.2× 加速。

### 預計算（`precompute_G_experts`）

```python
# 在 load checkpoint 後呼叫一次
G = einsum("er,rst->est", U_expert, core)   # (E, r3, r2)
self._G_experts_cache_bf16 = G              # decode 用
self._G_flat_cache_bf16 = G.T.reshape(...)  # prefill 用
self._compiled_call = mx.compile(self._forward)  # compile forward
```

之後每個 token 不需要重算 G，直接查表。

---

## 4. `mamba_block.py` — Mamba-3 SSM Block

SSM（State Space Model）是 Transformer Attention 的替代方案：
- **Attention**：每個 token 看所有其他 token，計算量 O(L²)，但表達力強
- **SSM**：用固定大小的「記憶狀態」h 累積過去資訊，計算量 O(L)，記憶體固定

### SSM 基本公式

```
h[t] = A * h[t-1] + B * u[t]    ← 狀態更新（A=衰減，B=輸入投影）
y[t] = C * h[t]                  ← 輸出
```

`h` 就是模型的「記憶」，大小固定為 `(H, N, P)`，不管序列多長。

### Mamba-3 的增強（相比 Mamba-2）

| 特性 | Mamba-1/2 | Mamba-3 |
|------|-----------|---------|
| B、C 矩陣 | 向量 (N,) | MIMO 矩陣 (N, R)，R>1 |
| 輸入混合 | 無 | λ 混合（學習信任當前 vs 前一步） |
| 位置編碼 | 無 | RoPE 旋轉 B 和 C |
| 離散化 | ZOH | 梯形（更準確） |

### 完整 Forward 流程

```
輸入 x: (B, L, d_model)
         │
         ├─ residual_mamba = x   ← 殘差連接保存
         │
         ▼
    norm_mamba(x)
         │
         ▼
    in_proj  → 切出 7 個子向量：
    ┌───────────────────────────────────────────┐
    │  z        (B, L, H*P)    gating 信號       │
    │  x_prime  (B, L, H*P)    主要信號           │
    │  B_param  (B, L, G*N*R)  輸入投影           │
    │  C_param  (B, L, G*N*R)  輸出投影           │
    │  dt_p     (B, L, G)      → softplus → dt  │
    │  A_p      (B, L, G)      → -exp() → A     │
    │  lam      (B, L, G)      → sigmoid → λ    │
    └───────────────────────────────────────────┘
         │
         ▼
    計算 RoPE 角度：
    delta_angle = dt * theta         (time-dependent rotation)
    angles_cum = cumsum(delta_angle) (累積相位)
    B_rot = RoPE(B_param, angles)
    C_rot = RoPE(C_param, angles)
         │
         ▼
    x_up = TuckerMoE(x_prime)     (H*P → H*P*R，升維)
    x_ssm = reshape(x_up, H, P, R)
         │
         ▼
    input_signal = einsum(B_rot, x_ssm)   (B, L, H, N, P)
         │
         ▼
    λ 混合（梯形離散化）：
    u_ssm = λ*dt*input_signal + (1-λ)*dt*av*prev_input_signal
         │
         ▼
    SSM 遞推：
    ┌──────────────────────────────────────┐
    │  L=1: h_new = av * h_prev + u_ssm   │  ← 單步，O(1)
    │  L>1: chunk_scan(u_ssm, ...)        │  ← 長序列，平行化
    └──────────────────────────────────────┘
         │
         ▼
    y = einsum(h_new, C_rot)   (B, L, H, P, R)
    y = y_down_proj(y)         (H*P*R → H*P，降維)
    y = y + x_prime * D        (skip connection，D 是可學習 scalar)
         │
         ▼
    gated = pre_gate_norm(y) * silu(z)
    mamba_out = mamba_dense_proj(gated)
    mid = residual_mamba + ls_mamba(mamba_out)
         │
         ▼
    proj_out = TuckerMoE(norm_out_proj(mid))
    out = mid + ls_out_proj(proj_out)
```

### State（狀態）格式

```python
state = {
    "h_prev":            (B, H, N, P)    # SSM 隱藏狀態，核心「記憶」
    "prev_input_signal": (B, H, N, P)    # 上一步輸入，λ 混合需要
    "angles_cum":        (B, H, N//2)    # 累積 RoPE 相位
}
```

### 兩條執行路徑

```python
def __call__(self, x, state=None):
    if L == 1 and state is not None and self._compiled_decode:
        # ── 快速路徑：mx.compile 後的 decode ──
        # 繞過所有 Python 條件判斷，直接執行 compiled graph
        return self._compiled_decode(x, h_prev, prev_input_signal, angles_cum)
    else:
        # ── 一般路徑：Prefill（L > 1）或第一個 token ──
        # 使用 chunk_scan 處理長序列
```

---

## 5. `transformer_block.py` — Transformer 注意力塊

標準 GQA（Grouped Query Attention）+ TuckerMoE FFN，
每隔 4 個 Mamba block 插一個，補充 Mamba 的全局注意力能力。

### GQA（Grouped Query Attention）

```
普通 MHA：每個 head 都有自己的 K、V（記憶體大）
GQA：      多個 Q head 共享同一組 K、V（記憶體節省 kv_groups 倍）

本模型：num_heads = d_model/64，num_kv_heads = num_heads / kv_groups
```

### Forward 流程

```
輸入 x: (B, L, d_model)
    │
    ▼ norm_attn
    ├── q_proj → Q: (B, H,   L, 64)
    ├── k_proj → K: (B, kvH, L, 64)    kvH < H，節省 KV cache
    └── v_proj → V: (B, kvH, L, 64)
    │
    ▼ 拼接過去的 KV cache
    K_full = [past_K, K]    (decode 時每步 +1)
    V_full = [past_V, V]
    │
    ▼ mx.fast.scaled_dot_product_attention
    (Apple Metal 原生 GQA kernel，自動處理 group broadcasting)
    │
    ▼ o_proj + ls_attn residual
    │
    ▼ norm_ffn
    │
    ▼ TuckerMoE FFN（SwiGLU 結構）
    gate = gate_proj(x)
    feat = up_proj(x)
    ffn_out = down_proj(silu(gate) * feat)
    │
    ▼ ls_ffn residual
    out
```

### Decode 時的特殊處理

KV cache 每步都在 append（形狀變動），**無法整塊 compile**，所以切成兩段：

```
_compiled_pre:  norm_attn + q/k/v proj       ← 輸入/輸出形狀固定，可 compile
中間：          KV concat + SDPA             ← 形狀動態，只有 2 個 kernel launch
_compiled_post: o_proj + FFN               ← 輸入/輸出形狀固定，可 compile
```

---

## 6. `scan_metal.py` — Chunk-Parallel SSM 掃描

Mamba block 在長序列（Prefill）時使用的平行計算演算法。

### 問題：SSM 遞推天生是串行的

```
h[0] = u[0]
h[1] = α[1]*h[0] + u[1]
h[2] = α[2]*h[1] + u[2]
...
h[L] = α[L]*h[L-1] + u[L]
```

每步依賴前一步，無法直接平行。L=4096 時非常慢。

### 解法：Chunk 平行掃描

把 L 切成 nc 個 chunk，每個長 Lc：

**第一階段：區塊內平行（GPU 計算）**

```
對每個 chunk c，計算所有位置對的衰減比例矩陣 M：
  la_cum[i] = cumsum(la[0..i])
  M[i,j] = exp(la_cum[i] - la_cum[j])  （i≥j 才有意義，j<i 的才影響 i）

h_intra[i] = Σ_{j≤i} M[i,j] * u[j]    ← 這個 einsum 完全平行
y_diag = einsum(h_intra, C)             ← 再平行
```

**第二階段：區塊間傳遞（CPU Python loop，nc 次，很快）**

```python
for c in range(nc):
    h_inter_list.append(h_prev)
    h_prev = h_prev * decay[c] + h_intra[c, -1]
    # decay[c] = exp(sum(la[c*Lc:(c+1)*Lc]))  ← 每個 chunk 的總衰減
```

**第三階段：合併**

```
y_off = einsum(h_inter, C * 衰減比例)   ← 前面 chunk 的貢獻
y = y_diag + y_off                      ← 最終輸出
```

### 純 MLX 版本 vs Metal 加速版本

| | `chunk_scan` | `chunk_scan_metal` |
|-|---|---|
| 實作 | einsum（MLX 自動用 GEMM） | Metal kernel transpose + matmul |
| 瓶頸 | H 維在最後，GEMM 有 stride | H 移到最前，記憶體連續 |
| 速度 | 基準 | ~2× 快 |

Metal 版本的核心技巧：
```
原本：M (B, nc, Lc, Lc, H)  — H 在最後，GEMM stride 讀取
轉換：M_hf (B*nc*H, Lc, Lc) — H 在最前，連續記憶體
→ 普通 batched matmul，速度翻倍
```

---

## 7. `fused_ssm_step.py` — Metal GPU Kernel 深度解說

> 這是整個推理加速最底層也最難讀的部分，從「GPU 是什麼」開始解釋。

---

### 7.1 前置知識：GPU 和 Kernel 是什麼

#### CPU vs GPU

```
CPU：
  - 少量核心（如 8 core）
  - 每個核心很聰明，可執行複雜邏輯、分支判斷
  - 適合串行、複雜的任務

GPU（如 Apple M2 Pro 的 GPU）：
  - 大量核心（如 3584 個 shader core）
  - 每個核心很簡單，只做加減乘除
  - 適合「同樣的運算重複幾千次」的任務
```

#### Kernel（GPU 核心程式）是什麼

Kernel 是**一段在 GPU 上執行的程式**，你可以想像成：

```
你告訴 GPU：「幫我把這個函式，同時對 10000 個資料點執行」
GPU 把 10000 個執行緒一起啟動，每個 thread 負責一個資料點
```

一個 Kernel Launch（一次 GPU 呼叫）的開銷約 5-10 μs，
即使計算本身只要 1 μs，也要花 5-10 μs 啟動。

#### Metal 是什麼

Metal 是 Apple 的低層 GPU API（類似 CUDA 但只有 Apple 裝置用）。
用 Metal 可以直接寫 GPU 程式（用類似 C++ 的語法），
比 PyTorch/MLX 的高階介面更底層、更靈活、開銷更小。

---

### 7.2 問題：Mamba Decode 的 Kernel 碎片化

每個 Mamba block 的 decode 步驟，在 MLX 中是一連串**小而獨立**的操作：

```
softplus(dt_p)           ← 1 kernel launch
-exp(A_p)                ← 1 kernel launch
la = dt * A              ← 1 kernel launch
av = exp(la)             ← 1 kernel launch
sigmoid(lam)             ← 1 kernel launch
delta_angle = dt * theta ← 1 kernel launch
angles_cum + delta       ← 1 kernel launch
sin(angles)              ← 1 kernel launch
cos(angles)              ← 1 kernel launch
B * cos - B * sin  ×2    ← 4 kernel launches (RoPE for B)
C * cos - C * sin  ×2    ← 4 kernel launches (RoPE for C)
input_signal einsum      ← 1 kernel launch
lv * dv * is             ← 3 kernel launches
(1-lv)*dv*av*ip          ← 4 kernel launches
u_ssm = t1 + t2          ← 1 kernel launch
h_new = av*h_prev + u    ← 2 kernel launches
y einsum                 ← 1 kernel launch
...共約 20+ 個 launches
```

每個 launch 5-10 μs，乘以 24 個 Mamba block = **2880-5760 μs = 2.9-5.8 ms** 純 launch overhead，
但這些計算本身只要 < 0.5 ms。**overhead 遠大於實際計算**。

---

### 7.3 解法：把所有計算合進一個 Kernel

`fused_ssm_step.py` 的核心想法：

```
把上面 20 個操作全部寫進一個 Metal C++ 函式
→ 只需要 1 個 kernel launch（5-10 μs）
→ 節省 95% 的 launch overhead
```

---

### 7.4 Kernel 的 Grid 設計

**Grid** = 你要啟動多少個 thread（執行緒）。

本模型的 SSM Kernel grid 設計：

```
grid = (N, P, H * B)

意思：
  N 個 thread 在 x 方向  → 對應狀態維度 N
  P 個 thread 在 y 方向  → 對應投影維度 P
  H*B 個 thread 在 z 方向 → 對應 head 數 H × batch B
```

每個 thread 有唯一的座標 `(n, p, h*B + b)`，
負責計算狀態張量 `h_new[b, h, n, p]` 的一個元素。

**直覺**：把一個 `(B, H, N, P)` 的張量鋪平，每個 thread 負責一個格子。

---

### 7.5 SSM Kernel 原始碼逐行解說

```metal
const uint n = thread_position_in_grid.x;   // 這個 thread 的 n 座標
const uint p = thread_position_in_grid.y;   // 這個 thread 的 p 座標
const uint zz = thread_position_in_grid.z;  // 對應到哪個 (b, h)
const uint h = zz % H;
const uint b = zz / H;
```

**步驟 1：計算 dt、A、av、lv（scalar，每個 thread 都算，因為每個 b 不同）**

```metal
// softplus(dt_p) = log(1 + exp(dt_p))
// 穩定版本：max(x, 0) + log(1 + exp(-|x|))
bfloat16_t xb   = dt_p[b];
bfloat16_t maxv = metal::max(xb, zero_b);
bfloat16_t minv = metal::min(xb, zero_b);
bfloat16_t dt_b = maxv + log1p(metal::exp(minv - maxv));
// ↑ 精確複製 MLX 的 binary_ops.h LogAddExp，保持 bf16 精度一致

// A = -exp(A_p)，使用 precise exp（比 fast::exp 精度高）
bfloat16_t eA  = metal::precise::exp(A_p[b]);
bfloat16_t A_b = static_cast<bfloat16_t>(-static_cast<float>(eA));

// la = float(bf16(dt * A))  → 注意 cast 順序！這是刻意的
// bf16 截斷發生在進入 float32 之前，這樣數值與 MLX 逐步算完全一致
bfloat16_t dtA = static_cast<bfloat16_t>(float(dt_b) * float(A_b));
bfloat16_t av  = static_cast<bfloat16_t>(metal::precise::exp(float(dtA)));

// sigmoid(lam) 穩定版本（避免 exp overflow）
bfloat16_t lb = lam[b];
bfloat16_t y0 = one_b / (one_b + metal::exp(metal::abs(lb)));
bfloat16_t lv = (lb < zero_b) ? y0 : one_b - y0;
```

為什麼要這麼仔細複製 MLX 的 cast 順序？

> 因為這個 kernel 必須和 MLX 的 unfused 版本**逐 token bit-exact**（完全一樣的輸出），
> 才能讓使用者在 metal_fuse=True 和 False 之間切換時，輸出不變。
> 任何一個 float32↔bf16 的轉換順序不一樣，都會差幾個 ULP（最後一位精度）。

**步驟 2：計算 RoPE 角度（每個 n 有自己的 angle）**

```metal
// 每個 n 對應 pair k = n/2
const uint k = n >> 1;     // n=0,1 → k=0；n=2,3 → k=1；...

// delta = dt * theta[k]  (dt 是當前步的，theta 是可學習頻率)
float delta = float(dt_b) * theta[k];   // theta 是 float32

// 累積角度：ac = delta + 上一步的角度
float ac = delta + ac_prev[(b * H + h) * K2 + k];

// 轉 bf16 做 sin/cos（mlx 的 angles.astype(bf16) → sin/cos）
bfloat16_t ang = static_cast<bfloat16_t>(ac);
bfloat16_t sin_a = static_cast<bfloat16_t>(metal::precise::sin(float(ang)));
bfloat16_t cos_a = static_cast<bfloat16_t>(metal::precise::cos(float(ang)));
```

**步驟 3：旋轉 B 和 C（RoPE）**

```metal
// n 的奇偶決定 RoPE 的哪半邊
const uint j = n & 1u;   // j=0 → cos部分，j=1 → sin部分

// 讀取 B[n, r] 和 B[n^1, r]（相鄰的 pair）
bfloat16_t b1 = Bp[b * NR + (2*k)   * R + r];  // n 偶數那個
bfloat16_t b2 = Bp[b * NR + (2*k+1) * R + r];  // n 奇數那個

// RoPE 公式：
// 新 x[2k]   = x[2k]*cos - x[2k+1]*sin
// 新 x[2k+1] = x[2k+1]*cos + x[2k]*sin
Brot[r] = (j == 0u)
    ? bf16(float(b1)*float(cos_a) - float(b2)*float(sin_a))
    : bf16(float(b2)*float(cos_a) + float(b1)*float(sin_a));
// C 同理
```

**步驟 4：計算 input_signal（跨 r 維度做 dot product）**

```metal
// input_signal[h,n,p] = Σ_r Brot[n,r] * x_ssm[h,p,r]
// 每個 thread (n,p,h) 負責算自己那個元素
float acc = 0.0f;
for (uint r = 0; r < R; ++r) {
    acc += float(Brot[r]) * float(x_ssm[(b*H + h) * PR + p*R + r]);
}
bfloat16_t is_ = static_cast<bfloat16_t>(acc);  // 歸約到 bf16
```

**步驟 5：λ 混合 + SSM 遞推**

```metal
// u_ssm = lv*dv*is + (1-lv)*dv*av*ip
// 每個二元運算都在 bf16 下進行（與 MLX 逐步算一致）
bfloat16_t t1 = bf16(float(lv) * float(dv));     // lv * dt
t1 = bf16(float(t1) * float(is_));                // * input_signal
bfloat16_t t2 = bf16(float(1-lv) * float(dv));   // (1-lv) * dt
t2 = bf16(float(t2) * float(av));                 // * exp(la)
t2 = bf16(float(t2) * float(ip[idx]));            // * prev_input_signal
bfloat16_t u = bf16(float(t1) + float(t2));       // 合計

// h_new = av * h_prev + u_ssm
bfloat16_t ah = bf16(float(av) * float(h_prev[idx]));
bfloat16_t hn = bf16(float(ah) + float(u));
h_new[idx] = hn;
new_ip[idx] = is_;   // 保存 input_signal 給下一步用
```

**步驟 6：計算 y（需要在 N 維度做歸約，用 Threadgroup Shared Memory）**

```metal
// y[h,p,r] = Σ_n h_new[h,n,p] * Crot[n,r]
// 問題：每個 thread 只算一個 n，但 y 需要所有 n 加起來
// 解法：用 threadgroup shared memory，讓同一 threadgroup 的 N 個 thread 協作

threadgroup float sh[N * R];   // 共享記憶體（整個 threadgroup 都看得到）

// 每個 thread 寫入自己的貢獻
for (uint r = 0; r < R; ++r) {
    sh[n * R + r] = float(hn) * float(Crot[r]);
}

threadgroup_barrier(mem_flags::mem_threadgroup);  // 等所有 thread 都寫完

// 只有 n==0 的 thread 負責加總（串行，N 次加法）
if (n == 0u) {
    for (uint r = 0; r < R; ++r) {
        float s = 0.0f;
        for (uint nn = 0; nn < N; ++nn) {
            s += sh[nn * R + r];
        }
        y[(b*H + h) * PR + p*R + r] = static_cast<bfloat16_t>(s);
    }
}
```

> `threadgroup_barrier` = 全組 thread 的同步點，確保所有 thread 寫完才繼續讀。

---

### 7.6 Tucker G_w Kernel

把 TuckerMoE 的 routing chain 做成一個 kernel，每個 thread 負責 G_w 的一個 `(r, s)` 元素：

```
grid = (R2, R3, B)  — 一個 thread 對應一個 G_w[b, r, s]

每個 thread 執行：
1. 讀 logits[b, 0..E]（8 個，很小）
2. scaled_tanh → softmax → top-2 選擇（在 register 裡算，不需 shared memory）
3. G_w[b, r, s] = G[e0, r, s]*p0 + G[e1, r, s]*p1
```

這 8 個 E 的 routing 邏輯在每個 thread 都重複算了一遍（R2*R3 次），
看起來浪費，但因為 routing 只有 8 個數字（L1 cache 熱），
比先算 routing 再 scatter 快得多。

---

### 7.7 norm_fold 變體（Norm 折疊進 SSM Kernel）

把 `norm_B`、`norm_C` 的 RMS 正規化計算折進 SSM kernel，
省去 4 個額外的 launch（2 個 rms_norm + 2 個 bias add）。

RMS 正規化需要對整個向量做歸約（求平方和），
需要 threadgroup shared memory 和同步：

```
# N 個 thread，每個讀 4 個元素（N_READS = 4，即 NR = 4N）
# simd_sum：同一 simdgroup（32 thread）的 simd 加速歸約
# 結果：inv = rsqrt(mean(x²) + ε)
```

此變體僅在 `mimo_rank == 4`（即 R == 4）時啟用，
因為 N_READS 假設每個 thread 恰好讀 4 個元素。

---

### 7.8 pregate Kernel

把 `pre_gate_norm(y) * silu(z)` 合成一個 kernel：

```
輸入：y (B, 1, D)、z (B, 1, D)、w (D,) 正規化權重
輸出：gated (B, 1, D)

每個 threadgroup 負責一個 batch row（D 個元素）
1. 歸約求 rms(y) → inv
2. nrm = w * bf16(float(y) * inv)
3. silu(z) = z * sigmoid(z)
4. gated = nrm * silu(z)
```

---

### 7.9 整體 Kernel Launch 節省

| | Kernel Launches（每 Mamba block，每 token） |
|-|---|
| 原始 MLX eager | ~20 個 |
| mx.compile（per-block） | ~5-8 個（MLX 自動合併部分） |
| metal_fuse=True | **1 個**（SSM chain） + 1 個（G_w）+ 1 個（pregate） + matmuls |
| norm_fold=True | 再省 4 個（norm_B/C 折進 SSM kernel） |

24 個 Mamba block × 節省 15+ launches × 5μs/launch ≈ **1.8 ms 節省**，
在 ~10 ms 的 decode step 裡佔 18%，是可量測的加速。

---

## 8. `hybrid_model.py` — 最頂層模型

### 架構

```python
class Mamba3LanguageModel:
    embed      # nn.Embedding(vocab_size, d_model)：token id → 向量
    backbone   # TrueHybridMamba：30 層 block
    norm       # 最後 RMSNorm
    head       # nn.Linear(d_model, vocab_size)（權重與 embed 共享）
    inv_sqrt_d # 1/sqrt(d_model)，在 head 前縮放（類似 Attention 的 scaling）
```

### head 的 scaled_tanh

```python
logits = self.head(h * inv_sqrt_d)    # 縮放後投影
logits = scaled_tanh(logits, 30.0)    # 壓縮到 [-30, 30]
```

壓縮 logits 是為了防止 softmax 數值爆炸，scale=30 讓大部分值不受影響，
只有極端值（>30）才被壓縮。

### Prefill 記憶體管理

```python
# 在 TrueHybridMamba.__call__ 裡
if prefill:
    mx.eval(x, *state_vals)  # 每個 block 後立刻 eval，釋放 lazy graph
```

**為什麼需要這個？**

MLX 是 lazy evaluation——計算不會立刻執行，而是建 graph，最後一起執行。
Prefill 時 L=4096，24 個 Mamba block 的中間 TuckerMoE 張量如果都留在 graph 裡，
會同時佔用 3-4 GB。每個 block eval 一次，讓 graph 逐步釋放，peak 記憶體從 4 GB 降到 < 1 GB。

### `precompute()` 方法

```python
def precompute(self):
    # 在 load_checkpoint 之後呼叫，做一次性的昂貴計算
    for layer in backbone.layers:
        if Mamba3Block:
            layer.x_up_proj.precompute_G_experts()   # Tucker cache
            layer.out_proj.precompute_G_experts()
            layer.precompute()                        # compile decode
        if TransformerBlock:
            layer.ffn.*.precompute_G_experts()        # Tucker cache
            layer.precompute()                        # compile pre/post

    # compile head（固定形狀 (1,1,d_model)）
    self._compiled_head = mx.compile(self._head_forward)
```

---

## 9. `static_decode.py` — 靜態圖 Decode 外殼

### 問題：per-block compile 的瓶頸

即使每個 block 都 compile 了，Python 還是要：
- 遍歷 30 個 layer（30 次 Python loop）
- 每個 layer 調用一次 compiled function（30 次 Python dispatch）
- 每個 Transformer block 做 KV concat（動態形狀，無法消除）

合計每步仍需 ~50 個 Python dispatch。

### 解法：靜態 KV Cache + 全模型編譯

```python
# 預先分配固定大小的 KV cache
S = ceil((n_prompt + max_tokens) / kv_round) * kv_round   # 向上對齊
K_buf = zeros((B, kvH, S, 64))                             # 固定大小
V_buf = zeros((B, kvH, S, 64))

# 每步用 slice_update 在固定位置寫入
K_buf = mx.slice_update(K_buf, k_new, write_pos, axes=(2,))

# 所有張量形狀固定 → mx.compile 整個 step function
compiled_step = mx.compile(one_token_step)
# one_token_step：embed → 30 layers → head → sampler → next token
# 全部在一個 compiled graph 裡
```

**效果**：Python dispatch 從 ~50 次 → **1 次**（呼叫 compiled_step）。

### decode 迴圈設計

```python
# 先算第一個 token（從 prefill 的 logits）
tok0, key = sample(prefill_logits)
x_tok = tok0.reshape(1, 1)

# 主迴圈
while not done:
    toks, write_pos, m_flat, kvs, ... = step_fn(x_tok, ...)
    x_tok = toks[-1].reshape(1, 1)   # 最後一個 token 給下一輪用
    mx.async_eval(toks)               # 非同步：讓 GPU 繼續跑
    if pending:
        drain(pending)                # 同時 CPU 處理上一輪的結果
    pending = toks
```

`mx.async_eval` 讓 GPU 和 CPU **流水線**執行：

```
時間軸：
GPU: [step 1] [step 2] [step 3] ...
CPU:           [print 1] [print 2] ...
```

### `unroll > 1`

```python
def step(x_tok, ...):
    toks = []
    for _ in range(unroll):
        row, ... = core(x_tok, ...)         # 整個模型 forward
        tok, key = sample(row, ...)
        toks.append(tok)
        x_tok = tok.reshape(1, 1)           # 立刻作為下一個 token！
    return mx.stack(toks), ...              # unroll 個 token 一起輸出
```

`unroll=4` 時，4 個 token 在同一個 compiled graph 裡依序產生，
完全不需要 Python 介入，消除 4-1=3 個 Python-GPU 同步點。

### 量化選項

```python
StaticDecoder(
    quant_moe_bits=8,    # MoE 的 U_in、U_out 矩陣量化
                         # 效果：頻寬節省 2×，速度 +35%
                         # 代價：非 bit-exact（但品質差異小）

    quant_proj_bits=8,   # in_proj 主體部分量化
                         # 注意：dt、A、λ 的 tail (3列) 保持 bf16
                         # 因為 SSM 對這些 scalar 的精度敏感

    quant_head_bits=8,   # head 投影量化
)
```

### `StaticStreamSession`

```python
# 適用於 chat_demo 的 CoT middleware
sess = decoder.start_stream(states, len(prompt_ids), max_new=512)

# 每個 token：
row = sess.step(tid)   # model forward only，返回原始 logits
# 然後 Python 端決定如何取樣（可以注入 token、修改 logits 等）
```

---

## 10. `weights.py` — Checkpoint 載入

### 雙路載入策略

```
第一次載入（.pt 或原始 .npz）：
    1. NumPy 讀入所有權重
    2. Key remapping：
       "backbone.layers.0.block.in_proj.weight"
       → "backbone.layers.0.in_proj.weight"
    3. 轉成 bf16 的 MLX array
    4. 用 mx.savez 寫出 .mlx_bf16.npz sidecar

第二次起（sidecar 存在）：
    model.load_weights(sidecar_path, strict=False)
    → MLX 用 mmap + Apple Unified Memory 零拷貝
    → 載入時間 < 5 ms
```

**為什麼 Unified Memory 快？**

Apple Silicon 的 CPU 和 GPU 共用同一塊 RAM。
`mmap` 讓權重直接映射到這塊 RAM，GPU 可以直接讀取，
完全不需要 CPU → GPU 的資料搬運（傳統 PCIe GPU 需要此步驟）。

---

## 11. 推理速度累積效果對照

| 優化層次 | 速度 | 關鍵技術 |
|----------|------|----------|
| 基礎 MLX eager | 23 tok/s | — |
| + mx.compile（per-block） | 47 tok/s | 消除 Python overhead |
| + StaticDecoder（靜態 KV） | 62 tok/s | 全模型一個 graph，+30% |
| + metal_fuse SSM + Tucker | 98-101 tok/s | 自定 Metal kernel，~2× |
| + 8-bit MoE 量化 | 131-144 tok/s | 頻寬節省，+35% |
| + norm_fold | 135-144 tok/s | 折疊 norm 進 kernel，+7.5% |

---

## 12. Decode 單步完整資料流

```
token_id: int
      │
      ▼ embed (lookup table)
  x: (B, 1, d_model) bf16
      │
  ┌───┤ 24× Mamba3Block
  │   │
  │   ├─ norm_mamba(x)
  │   ├─ in_proj → z, x_prime, B, C, dt, A, λ
  │   │
  │   ├─ [metal_fuse=False] MLX 逐步算（~20 launches）
  │   │   softplus(dt) → A → la → av → sigmoid(λ)
  │   │   RoPE 角度 → sin/cos → 旋轉 B, C
  │   │   input_signal einsum
  │   │   λ 混合 → u_ssm
  │   │   h_new = av * h_prev + u_ssm
  │   │   y einsum
  │   │
  │   ├─ [metal_fuse=True] 一個 Metal kernel（1 launch）
  │   │   （上面全部在 kernel 裡）
  │   │
  │   ├─ y_down_proj + D skip
  │   ├─ pre_gate_norm(y) * silu(z)  [可 pregate kernel]
  │   ├─ mamba_dense_proj
  │   ├─ 更新 state: h_new, input_signal, angles_cum
  │   └─ TuckerMoE out_proj（1 launch or fused G_w kernel）
  │
  ├───┤ 6× TransformerBlock（穿插於 Mamba 之間）
  │   │
  │   ├─ norm_attn + q/k/v proj（compiled）
  │   ├─ slice_update(K_buf, k_new, pos)   ← 靜態 KV cache 寫入
  │   ├─ mx.fast.SDPA(q, K_buf, V_buf)    ← Metal GQA kernel
  │   └─ o_proj + TuckerMoE FFN（compiled）
  │
  ▼
  x: (B, 1, d_model) bf16
      │
      ▼ norm + head（compiled，inv_sqrt_d scaling + scaled_tanh）
  logits: (B, vocab_size) f32
      │
      ▼ 取樣（rep penalty → presence penalty → top-k → top-p → categorical）
  next_token_id: (B,) int32
      │
      ▼ 更新 repeat window (ring buffer)
  [回到迴圈頂端]
```

---

*文件由 Claude 生成，對應 commit: main 分支 2026-06-14*
