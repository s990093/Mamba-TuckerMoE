# Mamba3-XR Inference Optimizations

本目錄包含 Mamba3-XR 模型在 Apple Silicon (MLX) 上的極致推論優化實作。

## 📁 檔案說明

| 檔案名稱 | 功能描述 |
| :--- | :--- |
| **`stream_mlx.py`** | 主要串流推論程式。支援互動式 Rich UI 介面與 token-by-token 輸出。 |
| **`benchmark_mlx.py`** | 綜合效能評測工具。支援 Speculative Decoding、各種算子編譯策略與吞吐量分析。 |
| **`fused_sampling_metal_v2.py`** | **[核心]** 最新 V3 單核心融合取樣核心。合併 Softmax、Min-P、Top-P 與抽樣邏輯。 |
| **`mlx_hybrid_infer.py`** | 模型架構核心定義。包含 Mamba3 與 Transformer 的混合層實作與緩存管理。 |
| **`mlx_mixed_quant.py`** | 混合精度量化引擎。針對 TuckerMoE 的 router 與權重進行非對稱位元量化。 |
| **`stream_fast_metal.sh`** | **[推薦]** 極速啟動腳本。預設開啟所有優化旗標以達到最高 tok/s。 |
| **`bench_pure_metal.sh`** | 硬體吞吐量極限測試。排除 I/O 與 Python 顯示開銷的純推論基準測試。 |
| **`custom_metal_ssm.py`** | 自定義 Metal SSM 並行掃描 (Parallel Scan) 算子介面。 |
| **`custom_metal_tucker.py`** | 自定義 Metal TuckerMoE 權重分解融合算子介面。 |
| **`bench_optimizations_ab.py`** | 自動化優化驗證工具。用於比較不同優化階段的性能收益 (A/B Test)。 |

## 🚀 核心優化：Fused Metal Sampling v3

我們實作了全新的單次調度 (Single-Dispatch) Metal 取樣核心 (`fused_sampling_metal_v2.py`)，將原本分散的 Logits 修正、Softmax、Top-P/Min-P 過濾與隨機抽樣全部合併為一個 GPU Kernel。

### 1. 技術關鍵：未歸一化概率捷徑 (Unnormalized Probability Shortcut)
這是性能突破的核心。利用數學特性：
- 設 $p_{max} = 1/Z$（$Z$ 為配分函數）。
- $min\_p$ 過濾條件 $p_i < min\_p \cdot p_{max}$ 可簡化為 $exp(x_i - gmax) < min\_p$。
- **優勢**：完全消除了浮點除法與全域歸一化同步的開銷。我們直接在未歸一化的累積分布函數 (CDF) 上進行取樣。

### 2. 取樣技巧與優化
- **幾何 Top-P 掃描**：使用 5 步幾何遞減掃描代替二分查找，在不影響取樣質量的前提下大幅降低核心延遲。
- **貪婪模式回退 (Greedy Fallback)**：當檢測到 `--fast-sample` 且無 Penalty 時，自動切換至 MLX 原生的 `mx.argmax`（網格歸約優化），確保在最簡單場景下也能達到硬體極限。
- **單次指令派發**：從 Logits 到最終 Token ID 僅需一次 GPU Command Buffer 提交。

## 📊 實驗結果 (Apple Silicon)

測試條件：`bf16` + `4-bit Quant` + `Full Decode Compile` + `Tucker Einsum Fuse`

| 模式 | 採樣參數 | 吞吐量 (tok/s) | 備註 |
| :--- | :--- | :--- | :--- |
| **Pure MLX (Greedy)** | default | 54.1 | 基準線 |
| **Fused Metal v3 (Greedy)** | --fast-sample | **58.4** | 超越原生 MLX |
| **Fused Metal v3 (Stochastic)** | top_p=0.9, min_p=0.05 | **49.2** | 較 v1 提升 12%+ |
| **Pure MLX (Stochastic)** | top_p=0.9, min_p=0.05 | 41.2 | 傳統方式較慢 |

> [!TIP]
> 目前瓶頸已轉移至記憶體頻寬 (Memory Bandwidth)。要突破 100+ tok/s，建議開啟 **Speculative Decoding** 路徑。

## 🛠️ 使用說明

### 快速啟動最快串流
```bash
sh inference/stream_fast_metal.sh --prompt "Your prompt here"
```

### 效能基準測試
```bash
sh inference/bench_pure_metal.sh --decode-tokens 2048
```

### 關鍵參數
- `--fused-sample-metal-v2`: 啟用 v3 單核心優化取樣。
- `--fast-sample`: 啟用極速貪婪解碼路徑。
- `--tucker-einsum-fuse`: 啟用 TuckerMoE 算子融合。
