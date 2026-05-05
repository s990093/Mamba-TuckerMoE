# Mamba3-XR System-Level Metal Optimizations

本資料夾包含了專為 Apple Silicon 打造的極致 Metal Shader 最佳化核心程式碼。

## 1. 核心實作邏輯：Fused TuckerMoE (SRAM 最佳化)

在 `fused_tucker_moe.metal` 中，我們實作了徹底消滅記憶體瓶頸的四步魔法，專為解決 MoE 動態路由造成的 Gather/Scatter 記憶體碎片化而生：

- **Step 1: 載入共用字典 (Threadgroup Memory)**
  將共享的 $U_{in}$ 和 $U_{out}$ 載入到 Threadgroup Memory（GPU 內最快的 SRAM 快取）。因為這兩個矩陣很小，大家可以一起看，大幅減少 DRAM 讀取。
  
- **Step 2: 各自領取任務**
  將 Token 映射到每一個 Thread 上，每個 Thread 專門負責一個 Token 的端到端處理。

- **Step 3: 晶片內連乘（核心魔法）**
  - Thread 讀取自己的 Token，先乘上 SRAM 裡的 $U_{in}$，得到降維後的 $x_s$。**注意：這裡絕對不能把它寫回主記憶體。**
  - 根據 Router 的結果，去主記憶體把這個 Token 專屬的 $G_e$ 拉進來（這是唯一一次去主記憶體拿動態資料）。
  - 立刻把 $x_s \times G_e$ 算完。
  - 算完的結果立刻再乘上 SRAM 裡的 $U_{out}$，得到最終的 $y$。

- **Step 4: 功德圓滿才放行**
  直到整個運算管線結束，才把最終的 $y$ 寫回主記憶體（Unified Memory）。

## 2. 核心實作邏輯：Mamba SSM Parallel Scan ($O(N)$ 優化)

針對 Mamba 架構中的 SSM (State Space Model) Parallel Scan 運算：
MLX 原生為了平行化，採用了擴張為矩陣的方法，導致 $O(N^2)$ 的記憶體與計算複雜度。
我們寫了原生的 Metal Kernel，讓每個 Thread 負責一個獨立的通道，利用 GPU 的平行度直接進行純量遞迴 (Sequential Scan)，將複雜度降回 $O(N)$。

在微基準測試中，這個作法讓 SSM 的計算速度提升了 **2.59 倍**！
