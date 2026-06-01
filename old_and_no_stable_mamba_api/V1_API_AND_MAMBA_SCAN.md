# V1 API 設計重點 + Mamba 平行掃描 & 加速優化

本文整理 `old_and_no_stable_mamba_api/` 中的 **v1 推論 API 設計模式**，以及 **Mamba 相關的加速、平行掃描、數值精度優化**。這是被 `mamba3_mlx/` 取代前的舊版實現。

---

## 一、V1 API 設計模式

### 1.1 集中式 Config 物件

全部配置集中在 `Mamba3Config`，既是模型架構參數也是推論/效能開關（`mlx_hybrid_infer.py:27-86`）：

```python
class Mamba3Config:
    # 模型架構（與 train.py 對齊）
    d_model=768, d_state=64, d_head=64, expand=4, num_layers=6
    chunk_size=64, use_parallel_scan=True
    kmoe_num_experts=8, kmoe_top_k=2, r1=32, r2=512, r3=256

    # 推論效能開關（寫在 config 內，不是 cli args）
    lookahead_router=False
    tucker_einsum_fuse=False
    tucker_full_fuse=False
    tucker_amx_fuse=False
    tucker_scalar_fuse=False
    fused_mamba_mixer=False
    fused_mamba_mixer_v3=False
```

**設計重點**：不是 CLI-arg dict，而是**物件屬性**。推論開關混在架構參數中，便於 per-layer 內部讀取（如 `self.config.tucker_einsum_fuse`）。

---

### 1.2 三層模型結構

```
Mamba3LanguageModel
├── embed: nn.Embedding
├── backbone: TrueHybridMamba       ← 30 層 (24 Mamba + 6 Transformer)
├── norm: nn.RMSNorm
└── head: nn.Linear (weight-tied with embed)
```

**Macro-layer 設計**（`TrueHybridMamba.__init__`, line 1155-1167）：
```
for _ in range(num_layers=6):
    for _ in range(mamba_ratio=4):
        Mamba3Block     # tagged l_type="mamba"
    TransformerBlock    # tagged l_type="transformer"
```
得出 6×5 = 30 層：`MMMM T MMMM T MMMM T MMMM T MMMM T MMMM T`。

每層透過 `l_type` 標籤區分行為，forward 時統一 iter，但按 l_type 分支。

---

### 1.3 Cache 介面：block 級別的類型化 cache tuple

**Mamba block cache**（`Mamba3Block.__call__`, line 790-791）：
```python
cache: Optional[Tuple[h_state_mx, prev_input_mx, prev_angle_sum_mx]]
# shape: (B, H, N, P), (B, 1, H, N, P), (B, 1, H, N//2)
```

**Transformer block cache**（`TransformerBlock.__call__`, line 1083-1090）：
```python
cache: Optional[Tuple[k_cache_mx, v_cache_mx]]
# shape: (B, num_heads, max_seq_len, head_dim)
# + seq_pos: mx.array — 當前要寫入的 KV 位置
```

**設計重點**：
- 非 dict，而是 **固定長度的 tuple**（Mamba=3 項, Transformer=2 項）
- decode 時 `seq_pos` 用 `mx.slice_update` 做定長 cache 就地更新，而非 `mx.concatenate`（避免每次 alloc）
- 編譯 decode/verify 附加在 layer attribute 上（見 1.5 節）

---

### 1.4 Lookahead Router（隱藏路由延遲）

`TrueHybridMamba.__call__`（line 1169-1248）在每層 forward 前提取上一層的輸入 hidden 作為 `router_anchor`：

```python
layer_in: List[mx.array] = []
for li, (layer, cache) in enumerate(zip(self.layers, caches)):
    anchor = layer_in[-1] if lookahead and li >= 1 else None
    x_in = x
    ...
    layer_in.append(x_in)
```

`_router_anchor_matching_dim()` 做維度匹配檢查，因為 Mamba `x_up_proj` 路由輸入維度是 `h*p` 而非 `d_model`。

---

### 1.5 Compile Attachment 模式（post-init JIT）

**核心思路**：`mx.compile` 不在 layer 內部定義，而是透過外部函數附加到 layer attribute。

**decode**（`attach_decode_compilation`, line 1375-1458）：
```python
# 外部函數為每個 layer 建立 mx.compile wrapper，掛到 layer._compiled_decode
layer._compiled_decode = make_mamba_compiled(layer, use_la=la)
```

**verify**（`attach_verify_compilation`, line 1461-1558）：
```python
# 為每個 K 值建立獨立的 compiled function，存入 dict
layer._compiled_verify: dict[int, Callable] = {
    4:  mx.compile(_verify_step_K4),
    6:  mx.compile(_verify_step_K6),
    ...
}
```

**Forward 內的 dispatch 優先級**（`TrueHybridMamba.__call__`, line 1182-1248）：
```
if l==1 and has _compiled_decode:
    → layer._compiled_decode(...)          # 最快的單 token decode
elif has _compiled_verify and l in it:
    → layer._compiled_verify[l](...)       # Jacobi K-token verify
else:
    → layer(...)                            # fallback: eager forward
```

**設計重點**：
- JIT 編譯完全在模型外完成，model 本身不依賴 compile
- 每個 K 值獨立 compile（避免 graph 重編譯）
- 有 `forward_with_mamba_intra()` 專用路徑（回傳 per-layer 中間態，用於 Jacobi 部分 accept 時重建 Mamba cache）

---

### 1.6 TuckerMoE 可切換多路徑

`TuckerMoE.__call__`（line 542-685）根據 config flag 走不同路徑：

```
1. tucker_full_fuse    → _full_fused_dense (U_in+RMSNorm+routedG+U_out 一發 Metal kernel)
2. tucker_scalar_fuse  → 外部 scalar kernel (decode b=1 專用)
3. tucker_amx_fuse     → 外部 AMX kernel (decode b=1 bf16，最快)
4. tucker_einsum_fuse  → einsum 融合 Metal kernel (SRAM cache x_shared)
5. fallback            → mx.einsum("br,bkrs->bks") + sum
```

每個路徑是獨立 Metal kernel（inline source 或外部 `_ultimate_kernels` library）。

---

### 1.7 Weight Loading 慣例

- `resolve_mlx_checkpoint()` → `.npz` > `.pt`，首次載入 `.pt` 後自動產生 `<stem>.npz` sidecar
- `strict_load_and_convert()`: key 重映射 `.block.` → `.`，`model_` prefix 移除
- Tucker weight 自動 transpose（舊格式 `U_in` 是 `(dim_in, r3)`，轉 `nn.Linear.weight` 的 `(r3, dim_in)`）

---

## 二、Mamba 平行掃描 & 加速優化

### 2.1 核心：chunk_parallel_scan_mlx

**檔案位置**：`lib/mlx_hybrid_infer.py:140-281`

**演算法**（與 `train.py` chunk_parallel_scan 數學對齊）：
```
輸入: u(B,L,H,N,P), dt_b(B,L,H), a_b(B,L,H), C_rotated(B,L,G,N,R)
輸出: y(B,L,H,P,R), h_final(B,H,N,P)

1. 補齊到 chunk_size 的倍數
2. 分成 nc 個 chunks
3. Intra-chunk: 自訂 Metal kernel（每個 thread 負責一個 channel 的 sequential scan）
   - 3D grid: (d_inner, H, B*nc), threadgroup=(32,1,1)
   - 逐 t 迭代: h_val = exp(la_clamped) * h_val + u
   - 輸出 h_intra: (B, nc, Lc, H, N, P)
4. y_diag = einsum("bclhnp, bclhnr -> bclhpr", h_intra, C_c)
5. Inter-chunk: 自訂 Metal kernel（單一 dispatch，nc 迴圈在寄存器內跑）
   - 3D grid: (d_inner, H, B), threadgroup=(min(d_inner,32), 1, 1)
   - 輸出 h_inter(進入每個 chunk 時的 prefix state) + h_prev(最終 state)
   - 用 float32 避免 bf16 累積誤差
6. decay_intra = exp(clip(cumsum(la_c, axis=2)))  # 所有 cumsum 在 float32
7. y_off = einsum("bchnp, bclhnr -> bclhpr", h_inter, C_c * decay_intra)
8. y = y_diag + y_off → reshape → return
```

**為什麼不用 Kogge-Stone**：Metal kernel 的串行 nc 迴圈在 real GPU 上比 Kogge-Stone 的 log-N 步驟更快（避免了 Python dispatch 開銷和多次 global memory 讀寫）。benchmark 顯示 Metal kernel 比序列 Python loop 快 **3-8×**（`benchmark_tree_scan.py:85-106`）。

---

### 2.2 增量掃描：chunk_parallel_scan_with_init

**檔案位置**：`lib/mlx_hybrid_infer.py:284-358`

**用途**：Jacobi K-token verify 時，Mamba cache 有非零初始狀態 `h_init`。

**數學分解**：
```
h_t = h_t^(zero)  +  h_init * exp(cumsum(dt * a))[t]
```

**計算步驟**（O(1) 平行深度）：
```python
# 1. 跑一次完整的 chunk_parallel_scan_mlx（h_init=0）
y_zero, h_final_zero = chunk_parallel_scan_mlx(u, dt_b, a_b, c_rotated, chunk_size)

# 2. 初始狀態修正（全部 float32，避免 bf16 cumsum drift）
log_alpha_all = dt_b * a_b                  # 只算一次，復用於 alpha_cum 和 alpha_total
alpha_cum = exp(cumsum(log_alpha_all))      # (B, L, H)
h_init_t = h_init * alpha_cum               # (B, L, H, N, P)
y_init = einsum("blhnp, blhnr -> blhpr", h_init_t, c_rotated)

# 3. 合併
y = y_zero + y_init
h_final = h_final_zero + h_init * exp(sum(log_alpha_all))
```

**關鍵**：不用 Python `for t in range(L)` 的 L 次 MLX dispatch（會阻塞 Metal pipeline），而是用 **1 個 Metal scan + 1 個 cumsum + 1 個 einsum 修正** 完成。

---

### 2.3 Float32 SSM 狀態更新（解碼精度保護）

**位置**：`mlx_hybrid_infer.py:983-990`

```python
# 單步 decode 時強制用 float32 中間精度
av0_f32 = av[:, 0].astype(mx.float32)
h_final = (
    prev_h.astype(mx.float32) * av0_f32
    + u_ssm[:, 0].astype(mx.float32)
).astype(x.dtype)
```

**原因**：bfloat16 每步 decode 損失約 8 mantissa bits，512 步後累積誤差會顯著漂移。float32 中間精度消滅此問題。

**benchmark 驗證**（`benchmark_tree_scan.py:253-275`）：bf16 單步 vs f32 中間的誤差對比，f32 中間可防止長序列 mantissa 腐蝕。

---

### 2.4 Fused Mamba Mixer（Metal 融合 kernel）

**位置**：`mlx_hybrid_infer.py:858-952`

#### Fused Mamba Mixer（基礎版）
單個 Metal dispatch 融合以下操作：
```
Norm(B, C) → RoPE(B, C, x_ssm) → Einsum(B×x_ssm) → Lambda gating → SSM update → Einsum(C×h) (=y)
```
從外部 kernel library `_ultimate_kernels.mamba_mixer.run()` 執行。節省多個 kernel launch overhead。

#### Fused Mamba Mixer v3（進階版）
額外融合：
```
y_down_proj  +  D_skip  (= x_prime * D_rep)
```
在同一個 Metal dispatch 內，節省 2 個 kernel/layer（~20 layers）。

**y_down_proj weight 預轉置**（`mlx_hybrid_infer.py:867-885`）：
```python
# 原始 shape: (P_out=64, P_R=256)
# 轉置後: (P_R=256, P_out=64)
# 讓 Phase 3 的 SIMD lanes 相鄰 stride 從 512 bytes 降到 2 bytes
# 32 個 cache-line 讀取合併為 1 個
self._yd_weight_dense_T = mx.contiguous(self._yd_weight_dense.T)
```

---

### 2.5 三種 Inter-Chunk Scan 對比

`benchmark_tree_scan.py` 比較了三種實現：

| 方法 | 複雜度 | 實現 | 速度 |
|------|--------|------|------|
| **Serial** | O(nc) | Python for loop | 基準 |
| **Kogge-Stone** | O(log nc) | MLX + mx.where broadcast | 1-2×（Python overhead 在小 nc 時主導） |
| **Metal kernel** | O(nc) in GPU registers | 單一 mx.fast.metal_kernel dispatch | **3-8×** |

**結論**：Metal kernel 最快，因為 nc 迴圈完全在 GPU 寄存器內跑，**零 Python dispatch 開銷**。

---

### 2.6 Sampled SSM 的 Metal 實現細節

#### Intra-chunk scan kernel（`mlx_hybrid_infer.py:175-211`）

```metal
uint d = thread_position_in_grid.x;   // d_inner 維
uint h = thread_position_in_grid.y;   // head 維
uint b_c = thread_position_in_grid.z; // batch * nc 維

T h_val = T(0.0);
for (uint t = 0; t < Lc; ++t) {
    T la = la_c[b_c * Lc * H + t * H + h];
    T u  = u_c[b_c * Lc * H * D + t * H * D + h * D + d];
    la = clamp(la, -40.0, 40.0);       // 防止 exp 溢出
    h_val = metal::exp(la) * h_val + u;
    out[t] = h_val;
}
```

每個 thread 負責一個 `(channel, head, batch*chunk)` 三元組的完整串行掃描。

#### Inter-chunk prefix scan kernel（`mlx_hybrid_infer.py:222-253`）

```metal
float h_accum = 0.0f;
for (uint c = 0; c < NC; ++c) {
    float decay = metal::exp(clamp(log_d[c], -88.0f, 88.0f));
    float x_val = float(x_in[c]);     // 每個 chunk 的最後一個 intra state
    h_inter[c] = h_accum;            // 進入 chunk c 之前的 prefix state
    h_accum = h_accum * decay + x_val;
}
h_prev = h_accum;                    // 最終 state
```

輸出兩個陣列：`h_inter`（exclusive prefix，shape `(B,nc,H,NP)`）和 `h_prev`（最終 state，shape `(B,H,NP)`）。**零 cross-thread 通訊**。

---

### 2.7 Intra State 提取（Jacobi partial-accept 優化）

**位置**：`mlx_hybrid_infer.py:266-279, 1001-1052`

`chunk_parallel_scan_mlx` 支援 `return_h_full_zero=True`，返回每個位置的 zero-init 隱藏態：
```
h_zero[c, t] = h_intra[c, t] + h_inter[c] * decay_intra[c, t]
```

`Mamba3Block` 的 `return_intra=True` 路徑（`__call__`）會把這個傳給 caller：
```python
return out, new_cache, (_h_all_corrected, input_signal, angles)
```

`TrueHybridMamba.forward_with_mamba_intra()` 收集所有 Mamba 層的中間態，供 Jacobi 在接受位置 `m` 重建 cache，**省去整個 `_batch_replay(m+1)` 的 forward pass**（~61ms → ~12ms）。

---

### 2.8 Float32 Cumsum 在 inter-chunk 中的應用

```python
# 在 chunk_parallel_scan_mlx 中（line 258-259）：
decay_intra = mx.exp(mx.clip(
    mx.cumsum(la_c.astype(mx.float32), axis=2),  # float32 cumsum
    -88.0, 88.0
))
c_dec = c_c * mx.expand_dims(decay_intra.astype(u.dtype), (-1, -2))
```

`chunk_parallel_scan_with_init` 也全部在 float32 做 cumsum（line 330-336），避免 bf16 在長序列（nc>64）時的累加誤差。

---

## 三、Sampling 加速（補充）

### 3.1 Unnormalized Probability Shortcut（v2 sampling kernel）

`lib/fused_sampling_metal_v2.py:196-206`：

```metal
// 傳統: p_i >= min_p * p_max, 其中 p_max = max(softmax(x))
// 但 softmax 歸一化: p_max = e^(gmax) / Z
// 展開: e^(xi - gmax) / Z >= min_p * (1/Z)
//       e^(xi - gmax) >= min_p  ← 不需要算 Z！不需要歸一化！
float e = metal::exp(work[base + i] - gmax);
if (MIN_P > 0.0f && e < MIN_P) {
    e = 0.0f;
}
work[base + i] = e;  // unnormalized probs, 直接當累積密度用
```

後續 top-p binary search 和 inverse CDF sampling 全部在未歸一化概率上進行，省去了一次全域除法。

### 3.2 Single-Dispatch Stochastic Sampling

v2 的貪婪 kernel 把 penalties + argmax 合併在一次 Metal dispatch（v1 要 6+ 次）。
Stochastic kernel 把 penalties + temperature + exp + min-p + top-p binary search (5 iter) + CDF prefix-sum + inverse-CDF sample **全部合在一次 dispatch**。

---

## 四、Key Takeaways

| 層面 | V1 核心模式 |
|------|------------|
| **Config** | 集中式物件，架構參數 + 推論 flag 一起存 |
| **Macro 層** | 4 Mamba + 1 Transformer 循環，`l_type` tag 標記 |
| **Cache** | 固定 tuple（Mamba=3 項, Transformer=2 項）；`mx.slice_update` 就地在 pre-allocated buffer 更新 |
| **Compile** | 外部 `attach_*_compilation` 附加到 `layer._compiled_decode/verify`；dispatch 優先級：compiled_decode > compiled_verify > eager |
| **Lookahead** | 上一層 hidden 作為 router anchor，隱藏 MoE routing 延遲 |
| **MoE 路徑** | Tucker full_fuse → scalar → AMX → einsum_fuse → fallback |
| **Parallel scan** | Intra-chunk: Metal kernel sequential scan per thread；Inter-chunk: Metal kernel 單 dispatch nc loop；所有 cumsum 在 float32 |
| **增量 scan** | `h_init * alpha_cum` 分解，O(1) parallel depth，取代 L 次 Python dispatch |
| **精度** | SSM state decode 用 float32 中間值；cumsum 全 f32；bf16 只在最終寫出時轉回 |
| **Fused mixer** | v3: Norm + RoPE + Lambda + SSM + y_down_proj + D_skip 一發 Metal dispatch；預轉置權重減少 32× cache-line 讀取 |
| **Intra extraction** | 回傳 per-position Mamba 隱藏態，讓 Jacobi partial-accept 直接重建 cache，省 batch_replay |
