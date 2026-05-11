# Ultimate Metal Kernels — 使用說明

> **Mamba3-XR 極致 Metal Kernel 算子融合套件**  
> 針對 Apple M3 GPU（19 cores, ~200 GB/s）設計的最高效能 BF16 推論 Kernel 集合

---

## 快速開始

```bash
# 1. 確保在 Mamba3-XR 目錄
cd /path/to/Mamba3-XR

# 2. 激活 Python 環境
source .venv/bin/activate

# 3. 執行完整 Benchmark（4 個實驗）
cd metal
python benchmark_ultimate_kernels.py --r3 256 --r2 1024 --e 8 --k 2 --dtype bf16

# 4. 查看結果
cat results/ultimate_experiment_log.md
```

---

## Kernel 架構總覽

### 三層優化策略（來自 metal/idea/ PDF 研究）

```
L1: 資料佈局改造（消除 Bank Conflict）
  ├─ 離線轉置：G_experts [E,R3,R2] → [E,R2,R3]（Python 端一次性執行）
  ├─ Padding ROW_STRIDE=40（= TILE_K+8，打破 32-bank 對齊）
  └─ XOR Swizzling（備選方案，無需離線轉置）

L2: 深度 Kernel 融合（消除全域記憶體往返）
  ├─ TuckerMoE：五算子融合 → 單一 dispatch
  ├─ SSM Scan：la_clamp + exp + 遞迴更新 → 單一 dispatch
  └─ RMSNorm：norm + linear 融合，x_norm 不寫 HBM

L3: BF16 精度 + FP32 累加
  ├─ bfloat16 存儲（節省 2x 記憶體頻寬）
  ├─ float32 內部累加（防止精度損失）
  └─ 向量化 bfloat4 讀寫（每次 8 bytes）
```

---

## Kernel 檔案說明

### `ultimate_tucker_moe_bf16.metal`

TuckerMoE 完整融合 Kernel，包含 5 個函式：

| 函式名 | 功能 | 適用場景 |
|--------|------|---------|
| `tucker_project_u_in` | x @ U_in → x_shared | 輸入降維 |
| `tucker_moe_gemv_amx` | x_shared @ G_e（AMX + 雙重緩衝）| 核心計算 |
| `tucker_project_u_out` | y_shared @ U_out → y | 輸出升維 |
| `tucker_moe_full_fused_small` | 全融合小秩版（R3≤256）| 端對端融合 |
| `tucker_moe_xor_swizzle` | XOR Swizzling 版 | 無需離線轉置時 |

### `ultimate_ssm_scan_bf16.metal`

SSM Parallel Scan，5 個函式：

| 函式名 | 功能 | 複雜度 |
|--------|------|--------|
| `ssm_decode_step_bf16` | 單步 decode（BS=1, L=1）| O(1) |
| `ssm_prefill_scan_bf16` | Prefill 序列掃描 | O(L) per channel |
| `ssm_prefill_scan_vec4_bf16` | 向量化版（bfloat4）| O(L/4) per channel |
| `ssm_fused_inproj_scan_bf16` | 融合 dt softplus + ZOH 離散化 | 完整融合 |
| `ssm_chunked_scan_bf16` | Chunked 版（避免 L2 thrash）| 長序列適用 |

### `ultimate_rms_norm_linear_bf16.metal`

RMSNorm + Linear 融合，4 個函式：

| 函式名 | 功能 |
|--------|------|
| `rms_norm_linear_bf16` | 單投影版（simd_sum RMS）|
| `rms_norm_qkv_bf16` | 三路 QKV 投影（共享一次 RMS）|
| `rms_norm_linear_amx_bf16` | AMX 向量化版（bfloat4）|
| `rms_norm_head_logits_bf16` | 融合 lm_head + scaled_tanh(30)|

### `ultimate_gate_silu_bf16.metal`

GateSiLU + LayerScale 融合，6 個函式（純量/向量 × 有無殘差）

### `ultimate_tucker_moe_v2_split.metal`

**備案 Kernel（Register Spilling 時使用）**  
拆解為 A+B+C+D 四個協作 Kernel，每個暫存器壓力極低，可達最高 GPU 佔用率

---

## Python 整合

### `ultimate_kernel_lib.py`

```python
from metal.ultimate_kernel_lib import UltimateMambaKernels

# 初始化
kernels = UltimateMambaKernels()

# 1. 離線轉置（只做一次，在模型載入時）
g_experts_T = kernels.prepare_g_experts(g_experts)  # [E,R3,R2] → [E,R2,R3]

# 2. 預熱（避免首次 decode 卡頓）
kernels.warmup(r3=256, r2=1024, e=8, k=2)

# 3. TuckerMoE 推論
amx_fn = kernels.tucker.build_amx(r3=256, r2=1024, e=8, k=2)
y = kernels.tucker.run_amx(amx_fn, x_shared, g_experts_T, expert_ids, r2=1024, k=2)

# 4. SSM decode
ssm_fn = kernels.ssm.build_decode_step(H=64, D=64)

# 5. 採樣（V3）
from metal.ultimate_sampling_v3 import sample_token_v3
token = sample_token_v3(logits, counts, args)
```

---

## 實驗結果（M3 GPU，BF16）

| 實驗 | 測試場景 | 加速比 |
|------|---------|--------|
| Bank Conflict 消除 | Tucker r3=256, r2=1024 | **3.51x** vs MLX |
| SSM Scan | L=64, H=64, D=64 | **10.79x** vs MLX |
| 融合效果 | Tucker 五算子 | **2.80x** vs MLX |

完整報告見：`results/ultimate_experiment_log.md`

---

## Benchmark 命令

```bash
# 全部實驗（約 30 秒）
python benchmark_ultimate_kernels.py --r3 256 --r2 1024 --e 8 --k 2

# 只跑 SSM 實驗（最快）
python benchmark_ultimate_kernels.py --exp exp3 --seq-len 128

# 精度更高（更多 trials）
python benchmark_ultimate_kernels.py --warmup 30 --trials 200

# 指定 FP32 精度對比
python benchmark_ultimate_kernels.py --dtype fp32
```

---

## 已知限制

| 限制 | 說明 | 解決方案 |
|------|------|---------|
| AMX 在 GEMV 無明顯優勢 | batch=1 時計算量太小 | prefill 或 batch≥8 時再啟用 |
| SSM max_err=144 | BF16 長序列累積誤差 | 使用 FP32 輸入版本 |
| r3/r2 需為 32 的倍數 | AMX tile 對齊要求 | Scalar 版本無此限制 |
| Register Spilling 風險 | 完整融合暫存器壓力大 | 使用 v2_split 版本 |

---

## 檔案結構

```
metal/
├── ultimate_tucker_moe_bf16.metal    # 最強 TuckerMoE kernel（5 變體）
├── ultimate_tucker_moe_v2_split.metal # 拆解版備案（4 kernel 協作）
├── ultimate_ssm_scan_bf16.metal       # SSM Scan（5 變體）
├── ultimate_rms_norm_linear_bf16.metal # RMSNorm+Linear（4 變體）
├── ultimate_gate_silu_bf16.metal      # GateSiLU+LayerScale（6 變體）
├── ultimate_sampling_v3.py            # 採樣 Kernel V3
├── ultimate_kernel_lib.py             # Python 統一管理層
├── benchmark_ultimate_kernels.py      # 完整 4 實驗 Benchmark
└── results/
    ├── ultimate_experiment_log.md     # 完整實驗記錄
    └── ultimate_experiment_results.json
```
