# Metal Kernel 設計準則 — Hybrid Mamba-TuckerMoE 推論優化

> 適用範圍：Apple Silicon (M1/M2 Pro/Max)，MLX 框架，Batch=1 自回歸 Decode 模式  
> 版本：2026-05，基於 88.9 tok/s → 100 tok/s 突破目標的實戰分析

---

## 1. Apple Silicon GPU 記憶體階層

### 1.1 統一記憶體架構（Unified Memory Architecture）

Apple Silicon 最重要的特性是 CPU、GPU、Neural Engine 共用同一塊 DRAM（M2 Pro 16GB）。  
這個設計消除了 PCIe 搬運開銷，但也意味著 GPU 帶寬（~200 GB/s）必須與 CPU 共享。

```
┌─────────────────────────────────────────────┐
│           LPDDR5 Unified Memory (16 GB)      │
│                  ~200 GB/s                   │
└────────┬──────────────────────┬──────────────┘
         │                      │
    ┌────▼────┐            ┌────▼────┐
    │  CPU    │            │  GPU    │
    │  L1: 192KB/core      │  L1 SRAM (threadgroup)
    │  L2: 共享            │  L2: ~4-8 MB (chip cache)
    └─────────┘            └─────────┘
```

**關鍵數字（M2 Pro GPU）：**

| 層級 | 大小 | 延遲 | 帶寬 |
|------|------|------|------|
| Threadgroup SRAM（L1） | 32 KB/threadgroup（上限） | ~1 cycle | ~數 TB/s |
| L2 Chip Cache | ~4-8 MB（估計） | ~10-20 cycles | ~數 TB/s |
| DRAM（Unified Memory） | 16 GB | ~100+ cycles | ~200 GB/s |

### 1.2 Threadgroup 資源限制

每個 Threadgroup 的硬限制：

- **最大執行緒數**：1024 threads/threadgroup
- **SRAM 大小**：32 KB threadgroup memory（超過會 spill 到 L2/DRAM）
- **SIMD group size**：32 threads（固定，Metal 規範保證）
- **暫存器（register）**：每 thread 約 256 個 32-bit register，超過會 spill

### 1.3 SIMD Group 操作

Metal 提供 simd_* 內建函式，在 32 個 lane 之間做 warp-level reduction：

```metal
// 在 SIMD group 內對 partial 做求和
float result = simd_sum(partial);  // 每個 lane 得到相同的 group total

// 其他 simd 操作
float v = simd_shuffle_down(x, 1);  // lane i 得到 lane i+1 的值
float v = simd_broadcast(x, 0);     // lane 0 廣播給所有 lane
```

**關鍵限制**：`simd_sum` 在 SIMD group 內對**所有 32 個 lane 的 partial 值加總**，得到一個相同的 scalar 結果廣播給所有 lane。若 32 個 lane 本來計算的是**不同輸出元素**的 partial sum，`simd_sum` 只會把它們錯誤地加在一起。

---

## 2. Batch=1 Decode 的核心特性

### 2.1 為何 Decode 比 Prefill 難優化

自回歸 Decode 每步只處理 1 個 token（L=1）。這意味著：

- **算術強度（Arithmetic Intensity）極低**：每個權重元素只使用 1-2 次
- **記憶體帶寬瓶頸**：幾乎所有時間都花在從 DRAM 載入權重，而不是計算
- **Batch=1 無法並行**：無法利用批次維度分攤記憶體讀取

以 SSM 掃描為例：

```
Arithmetic Intensity = FLOPs / Bytes
= (H × N × P × 2) / (H × N × P × sizeof(bf16) × 2)
= 1 FLOP/Byte  ← 完全 memory-bound
```

M2 Pro GPU roofline：
- Peak compute：~11 TFLOPS（fp16）
- Peak bandwidth：~200 GB/s
- Roofline crossover：~55 FLOP/Byte
- SSM decode AI ~1 → **永遠 memory-bound**，除非藉由 kernel fusion 消除 intermediate tensor 的讀寫

### 2.2 Kernel Launch 開銷的重要性

在 `mx.compile` 模式下（throughput 模式），MLX 把整個 decode step 編入一個 Metal command buffer。  
此時**沒有 Python 開銷**，但每個 kernel dispatch 仍有固定成本：

- **Metal dispatch overhead**（估計）：~5-15 µs/dispatch（driver 端）
- **24 Mamba layers × 2 extra dispatches（y_down + D_skip）= 48 dispatches**
- 潛在節省：48 × 10 µs = ~0.48 ms（理論上限）

在**非編譯模式（safe/eager）**下，每個 dispatch 還有額外 Python 與 MLX lazy evaluation 開銷：
- 每個 dispatch 約 5-20 µs 額外
- 48 dispatches = 0.24-0.96 ms 可節省

### 2.3 V1 Fused Mamba Mixer 的設計決策

V1 Kernel 融合了以下操作：
1. **RMSNorm**（B、C 兩條線）
2. **Bias 加法**
3. **RoPE 旋轉**（使用 prev_angle + dt×theta）
4. **Einsum1**（B × x_ssm → input_signal）
5. **Lambda-trapezoid SSM 掃描**
6. **Einsum2**（h_state × C → y_out）

每層省去 6 個 MLX dispatch，合計 24 layers × 6 = 144 個 dispatch 消除。

**測量結果**：V1 比完全 unfused 路徑快 ~0.84 ms/layer（~20 ms/step total improvement）。

---

## 3. Fusion 決策準則：何時融合，何時不融合

### 3.1 融合有益的條件

```
融合有益 ⟺ 以下條件成立：

(1) Intermediate tensor 大小 × 2（讀+寫）× 帶寬倒數 > Kernel launch overhead
    即：省下的記憶體來回時間 > 多出的計算開銷

(2) 融合後的 threadgroup SRAM 使用量 ≤ 32 KB

(3) 融合後的 register 使用量不導致 register spill

(4) 融合操作之間不需要改變 threadgroup 維度（grid geometry 相容）
```

### 3.2 融合無益的條件

```
融合無益（甚至有害）⟺

(1) 在 mx.compile 模式下：中間 tensor 不需寫到 DRAM，
    Intermediate buffers 由 MLX runtime 在 L2 cache 存留。
    此時融合只增加 compute，卻不消除實際的 DRAM roundtrip。

(2) 融合後的額外計算 > 節省的記憶體讀寫時間

(3) 融合後 threadgroup 結構改變，導致原本的 vectorized load 退化
```

### 3.3 本專案具體判斷表

| 融合目標 | 已融合 | 是否有益 | 原因 |
|---------|--------|---------|------|
| RMSNorm + Bias + RoPE | V1 | ✅ 有益 | 省去 3 個 dispatch，intermediate 跨 dispatch |
| Einsum1 + SSM + Einsum2 | V1 | ✅ 有益 | 核心計算，state 必須在 register 中完成 |
| V1 + y_down_proj | V3 | ❌ 無益 | 見第 4 節詳細分析 |
| V1 + D_skip | V3 | ❌ 無益 | 同上，compute 增加但 DRAM 未減少 |
| Tucker U_in + RMSNorm + Core + U_out | full_fuse | ❌ 無益 | 32 KB SRAM 不足，register spill |

---

## 4. V3 分析：為何 Phase 3 輸給 MLX 原生 Matmul（已確認）

### 4.1 V3 的設計意圖

V3 在 V1 的 Phase 2 結束後新增 Phase 3，在同一個 Metal kernel 內繼續計算：

```
y_combined[h, p] = dot(y_out[h, :], yd_weight[p, :]) + x_prime[h*P+p] * D[h*P+p]
```

其中 `y_out` 為 `(H, P, R)` = `(24, 64, 4)` shaped，`yd_weight` 為 `(P_out=64, P_in*R=256)` shaped。

目標：消除 2 個 dispatch（`y_down_proj` + `D_skip`），節省 dispatch overhead。

### 4.2 實測結果（M2 Pro, 4-bit, mx.compile throughput, warmup=3, 256 tokens）

```
V1 only:                             98.0 tok/s  ← 最快
V3 original (non-coalesced reads):   80.8 tok/s  ← REGRESSION −17.2
V3 + transposed yd_weight (fix):     83.3 tok/s  ← REGRESSION −14.7

Kernel-level (benchmark_mixer_v3.py):
  V1:  0.254 ms/kernel
  V3 (original, stride 512 bytes): ~0.339 ms/kernel (+0.085 ms)
  V3 (transposed, stride 2 bytes):  0.275 ms/kernel (+0.021 ms) ← 75% 改善
  Phase 3 overhead × 24 layers: 0.021 × 24 = +0.504 ms/step
```

**Note**: 早期記憶體中的「102-103 tok/s with V3」是錯誤測量（不同 warmup / token count 條件）。

### 4.3 根本原因：Phase 3 Sequential Loop vs AMX Matmul

```
V1 path:  V1 kernel (0.254 ms) + MLX native y_down_proj matmul
V3 path:  V3 kernel (0.275 ms, Phase 3 sequential loop)

MLX native matmul for y_down_proj (64×256 weight × (H=24) vectors):
  → 使用 simdgroup_matrix (AMX)
  → 資料保留在 Tier-1 register file (208 KiB)，完全不碰 device memory
  → 原生 tile 大小 8×8，可以同時計算多個 output elements

V3 Phase 3 sequential loop (即使 transposed 後 coalesced):
  → 每個 thread p 獨立計算 dot(y_sram[256], yd_weight_T[:, p])
  → 雖然 reads 已 coalesced (stride=2 bytes)，仍需 L2 cache 往返
  → 無法利用 AMX register-level computation
```

**結論**：V3 Phase 3 無論如何優化 memory access pattern，都無法超越 MLX 內部使用 AMX 的 matmul。  
在 `mx.compile` 模式下，y_down_proj 的 dispatch overhead 幾乎為零（在同一 command buffer 內），  
因此 V3 只有增加 compute cost，沒有節省 dispatch cost。

### 4.4 Transposed yd_weight Fix（已實施，但不足）

```
原始 Phase 3（非 coalesced）：
  yd_weight[p * P_R + pr]
  SIMD lanes 間 stride = P_R × sizeof(T) = 512 bytes → 32 cache lines per iteration

修復後（coalesced）：
  yd_weight_T[pr * P_VAL + p]  （yd_weight_T 為 offline 轉置：(P_R=256, P_VAL=64)）
  SIMD lanes 間 stride = sizeof(T) = 2 bytes → 1 cache line per iteration

效果：
  Phase 3 overhead: 0.085 ms/layer → 0.021 ms/layer（75% 改善）
  端到端: 80.8 → 83.3 tok/s（+2.5 tok/s 改善，但仍輸 V1 98.0 tok/s）
```

Fix 已保留在 code 中（`ultimate_kernel_lib.py` V3 kernel + `mlx_hybrid_infer.py`），  
但 `run_fast_stream.sh` 中 V3 維持 DISABLED。

### 4.5 為何 simd_sum 無法修復 Phase 3

一個直覺的優化想法是：用 `simd_sum` 做 cooperative reduction，把 P_R=256 的 dot product 分配給 32 個 SIMD lane，每個 lane 計算 8 個元素的 partial sum，再用 `simd_sum` reduce。

**問題所在**：本 kernel 的 threadgroup geometry 是 `(P=64, 1, 1)`，分成 2 個 SIMD group（lane 0-31 = thread p=0..31，lane 0-31 = thread p=32..63）。

```
SIMD group 0：
  lane 0 → thread p=0，計算 y_down[p=0] 的 partial，讀 yd_weight[0, :]
  lane 1 → thread p=1，計算 y_down[p=1] 的 partial，讀 yd_weight[1, :]
  ...
  lane 31 → thread p=31，計算 y_down[p=31] 的 partial，讀 yd_weight[31, :]

simd_sum(partial) → 把所有 lane 的 partial 加總 = y_down[0]+y_down[1]+...+y_down[31]
                    這是一個沒有意義的純量，不是任何一個 y_down[p] 的正確值！
```

要讓 `simd_sum` 正確地計算 y_down[p]，需要**32 個 lane 全部計算同一個 p 的不同 segment**。  
這需要把 threadgroup 重構為 `(P_reduction=32, P_out=64, 1)` = 2048 threads → 超過 Metal 1024 上限。

**結論**：在目前的 threadgroup geometry 下，simd_sum 無法應用於 Phase 3。  
這是架構性限制，不是程式碼問題。

---

## 5. V3 Phase 3 的正確設計（若要重構 threadgroup）

### 5.1 理論上可行的方案

若將 Phase 3 獨立為一個新 kernel，可採用以下設計：

```
grid：(P_out=64, H=24, 1)
threadgroup：(P_in_partial=16, P_out=64, 1) = 1024 threads（Metal 上限）

每個 threadgroup 負責一個 head (h_idx)：
  - 16 個 reduction thread 共同計算同一個 p 的 y_down[p]
  - SIMD group（32 lanes）跨越相鄰 2 個 p 的 reduction threads
  - 無法直接用 simd_sum；需要 threadgroup reduction
```

但此方案需要：
1. 一次額外的 kernel launch（開銷 ~10 µs）
2. 完全重寫 kernel，與 V1 無共用
3. y_out 仍需寫到 threadgroup SRAM（32 KB），幾乎佔滿 SRAM 配額

**結論**：理論可行但不值得實施。即使 Phase 3 用更好的 reduction，核心問題是：  
MLX 的 native matmul 已經是 AMX-accelerated，Phase 3 的最佳實作也只是追平而不會超越。  
真正的路徑是使用 `simdgroup_matrix` 重寫 Phase 3，但這要求重構整個 kernel geometry。

### 5.2 simdgroup_matrix 方案（未來方向，高難度）

若要讓 V3 Phase 3 與 MLX AMX matmul 競爭，需要：

```
Phase 3 用 simdgroup_float8x8 計算 yd_weight.T @ y_sram:
  y_sram:    (256,) in threadgroup memory
  yd_weight_T: (256, 64) in device memory
  輸出:      (64,) = y_down

simdgroup_matrix tiling:
  每個 SIMD group 計算一個 (8, 256) × (256, 8) → (8, 8) 的 tile
  需要 threadgroup geometry 能支援 (8, 8) 的 output tile
  與 Phases 1&2 的 (P=64, 1, 1) geometry 不相容
  → 需要完全重寫為 Phase 1+2+3 分開的 multi-pass kernel
```

**難度**：極高。建議先實施 speculative decoding（預期 +20-40%）。

---

## 6. 記憶體優化：Paged State Cache 設計

### 6.1 動機

目前每次 decode 前，MLX 可能需要為 intermediate tensors 分配記憶體。  
預先分配所有快取（pre-allocate）並強制 `mx.eval()` 可以：

1. **消除首 token 分配延遲**：~5-15 ms（一次性）
2. **防止 page fault**：確保 GPU 頁面在 Metal 記憶體中是熱的
3. **連續記憶體佈局**：改善 cache locality

### 6.2 Mamba State Cache 結構

每個 Mamba block 的 decode cache：

```python
# h_state:   (B=1, H=24, N=64, P=64)  → 24×64×64×2B = 196 KB
# prev_input: (B=1, 1, H=24, N=64, P=64) → 196 KB
# prev_angle: (B=1, 1, H=24, N//2=32)   → 24×32×2B = 1.5 KB

# 共 24 個 Mamba blocks → 24 × (196+196+1.5) KB ≈ 9.4 MB
```

### 6.3 Transformer KV Cache 結構

每個 Transformer block：

```python
# k_cache: (B=1, num_heads, max_seq_len, head_dim=64) → heads×seq×64×2B
# v_cache: 同上

# num_heads = d_model // 64（需按 config 計算）
# 6 個 Transformer blocks，max_seq_len=512 → ~6 × 2 × heads × 512 × 64 × 2B
```

### 6.4 預分配實作要點

```python
mx.eval(buffer)  # 強制 Metal 分配，消除首次延遲
```

`mx.eval` 把 lazy tensor 強制 evaluate，確保 Metal buffer 分配完成。  
不呼叫此函式，MLX 可能在首次 decode 時才真正分配記憶體，引入突發延遲。

---

## 7. 失敗實驗摘要

### 7.1 Full Tucker Fusion（--full-fuse）

**嘗試**：把 TuckerMoE 的 U_in（768→R1=32）+ RMSNorm + Core（R1×R2×R3）+ U_out（R2=512→768）融合成單一 kernel。

**結果**：比 scalar_fuse 慢。

**原因**：
- Tucker Core 矩陣大小 = E×R1×R2 = 8×32×512 = 131,072 元素 = 256 KB（超過 L2）
- 融合後 threadgroup 需要同時存放 U_in、Core、U_out，遠超 32 KB SRAM
- Register spill 使每個 thread 頻繁存取 device memory

### 7.2 Tucker AMX Front Fusion（tucker_front）

**嘗試**：用 simdgroup_matrix（AMX 等效）加速 U_in 矩陣乘法（768×32）。

**結果**：無顯著提升（batch=1）。

**原因**：
- simdgroup_matrix 設計用於 batch≥8 的矩陣乘法
- Batch=1 時，AMX 的 8×8 tile 大多是 padding，利用率 <13%
- Metal 的 `mx.fast.matmul` 已有同等優化

### 7.3 V3 Naive Phase 3（目前版本）

**嘗試**：在 V1 kernel 末尾加入 Phase 3，用 threadgroup SRAM 共享 y_out，再做 dot product。

**結果**：每層 +0.085 ms，全系統 +2.04 ms（88.9 tok/s → ~86.8 tok/s）。

**根本原因**：如第 4 節所述，mx.compile 模式下 intermediate tensor 不走 DRAM，融合只帶來計算開銷而無記憶體節省。

### 7.4 TuckerMoE XOR Swizzle Kernel

**嘗試**：用 XOR swizzling 消除 G_experts Bank Conflict，避免離線轉置。

**結果**：與 scalar_fuse 相當，但程式碼複雜度高。

**原因**：M2 Pro 的 L2 cache 足夠大，離線轉置後 G_experts 在 cache 中，bank conflict 實際影響有限。

---

## 8. 未來優化方向（現實可行性評估）

### 8.1 高可行性

#### A. 投機解碼（Speculative Decoding）調優

- **目前**：draft 8 層，target 24+6 層
- **可優化**：draft acceptance rate 提升（beam search draft）
- **預期收益**：若 acceptance rate 從 0.7 提升到 0.85，解碼速度可提升 20-30%

#### B. 4-bit Asymmetric MoE 量化

- **目前**：8-bit 量化 → 68 tok/s；4-bit → 88.9 tok/s（含 V1）
- **可優化**：僅對 G_experts 做 4-bit（佔模型 60% 以上），U_in/U_out 保持 8-bit
- **預期收益**：帶寬節省 2× for G_experts，估計 5-10 tok/s 提升

#### C. KV Cache 量化

- **目前**：KV cache 為 bf16
- **可優化**：fp8（Metal 不原生支援，需軟體模擬）或 int8 KV
- **預期收益**：KV 記憶體從 14 MB → 7 MB @512 steps，帶寬節省 2×

### 8.2 中等可行性

#### D. SSM State 量化（8-bit Mamba State）

- Mamba h_state 為 (H, N, P) = (24, 64, 64) 的浮點張量
- 每個 decode step 讀寫一次，9.4 MB 帶寬/step
- 量化為 int8：帶寬節省 2×，但 SSM 動態範圍大，量化誤差需謹慎

#### E. 融合 Mamba + TuckerMoE（跨 block fusion）

- 把整個 Mamba block（含 MoE）融合成 2-3 個 kernel
- 需要完全重寫，開發成本極高
- 在 compile 模式下效益不確定

### 8.3 低可行性（目前不建議）

#### F. Neural Engine (ANE) 卸載

- ANE 僅支援 CoreML 格式，無法直接使用 MLX
- 需要完整模型轉換，與動態 SSM state 不相容
- 開發成本 > 1 個月

#### G. Multi-Token Prediction（MTP）

- 需要修改模型結構，額外訓練 draft head
- 對 Mamba 的 stateful 性質有挑戰（state 必須 rollback on rejection）

---

## 9. 實戰工作流程：如何正確評估一個新的 Kernel Fusion

```
1. 確認目標（benchmark BEFORE）
   make mlx-bench DECODE_TOK=256

2. 在 safe 模式驗證數值正確性
   --inference-type safe --no-tucker-einsum-fuse --quantize 0
   max_abs_err < 0.05 for bf16

3. 理解 bottleneck 位置
   python inference/tools/profile_mlx_infer.py --profile-decode-steps 32
   → 哪個 layer/operation 佔最多時間？

4. 計算理論節省
   - Intermediate tensor 大小 × 2（讀+寫）× (1/帶寬) = 記憶體節省時間
   - 是否超過 kernel launch overhead（~10-20 µs）？
   - 在 mx.compile 模式下 intermediate 是否真的走 DRAM？

5. 實作並用 benchmark_fused_mamba_mixer.py 做 micro-benchmark

6. 端對端測試
   make mlx-bench DECODE_TOK=256

7. 與 V1 基準對比，報告 delta（正為改善，負為 regression）
```

---

## 10. 關鍵設計原則總結

1. **在 mx.compile 模式下，intermediate tensors 不一定走 DRAM**  
   → 融合不一定省記憶體帶寬，但一定增加 compute

2. **Batch=1 Decode 的瓶頸是帶寬，而非 compute**  
   → 有意義的融合必須消除真正的 DRAM roundtrip

3. **simd_sum 僅在 32 個 lane 計算同一輸出的 partial sum 時有效**  
   → 若各 lane 計算不同輸出，simd_sum 會產生錯誤結果

4. **Threadgroup SRAM 32 KB 上限是硬限制**  
   → 超過後 register spill 會嚴重降低效能

5. **Metal Kernel Launch Overhead 約 10-20 µs（compiled 模式）**  
   → 融合的目標必須節省遠超此值的時間，否則不值得

6. **V1 Fused Mamba Mixer 是目前的最優解**  
   → 在 mx.compile throughput 模式下，V1+scalar_fuse+4bit 為最快組合（88.9 tok/s）

---

*最後更新：2026-05-16*  
*作者：Mamba3-XR 研究團隊*  
*參考：implementation_plan.md, metal/ultimate_kernel_lib.py, inference/lib/mlx_hybrid_infer.py*
