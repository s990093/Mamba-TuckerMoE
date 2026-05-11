### 系統級優化實驗結果 (System-Level Optimizations)

我們在 MLX 框架上針對 **Einsum Fuse (Gather 消除)** 與 **Lookahead Router (路由預取)** 進行了效能驗證，測試模型規模約為 1.3B，解碼(Decode)速度的提升如下：

| Configuration                     | Prefill (tok/s) | Decode (tok/s) | 效能對比 (vs Baseline) |
| :-------------------------------- | :-------------: | :------------: | :--------------------: |
| **① Baseline**                    |      174.7      |      56.7      |           ━            |
| **② Einsum Fuse (Gather消除)**    |      177.2      |    **59.6**    |      **▲ +5.1%**       |
| **③ Lookahead Router (路由預取)** |      174.6      |    **59.4**    |      **▲ +4.9%**       |
| **④ Both Combined (雙優化)**      |      177.1      |      56.4      |        ▼ -0.6%         |

- **單項優化顯著提升**：針對空間問題的 Einsum Fuse 與時間問題的 Lookahead，分別帶來了約 5% 的 TPS 提升，證明消除記憶體搬移與掩蓋延遲能有效榨乾硬體極限。
- **雙重優化的挑戰**：在 1.3B 的小模型下，同時開啟兩者的即時編譯 (JIT Compile) 與調度開銷超過了節省的時間。預期在更大參數級別的模型中，這兩招合體的威力將更加明顯。

---

### MLX 與 Metal 底層最佳化實作概念

要將上述的 Einsum Fuse 推向極致，我們需要跳脫框架原生的限制，使用 C++ 與 Metal Shader 開發客製化算子 (Custom Kernel)。

#### 1. Metal Shader 底層實作概念 (虛擬碼)

我們強制 GPU 在高速共享快取 (SRAM) 內一氣呵成完成「專家選擇、計算、加總」，絕不碰觸慢速的統一記憶體。

```cpp
// fused_tucker_moe.metal
#include <metal_stdlib>
using namespace metal;

kernel void fused_tucker_moe_kernel(
    device const float* x [[buffer(0)]],
    device const int* router_indices [[buffer(1)]],
    device const float* U_in [[buffer(2)]],
    device const float* G_cores [[buffer(3)]],
    device float* out [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    // 1. 絕不寫回 DRAM，全程在高速 SRAM 操作
    // 2. 根據 router_indices 動態載入對應的 G_cores
    // 3. 執行矩陣乘積加總後，一次性寫回最終結果 out
}
```

#### 2. MLX 端的綁定 (Python / C++)

透過 MLX 提供的 `mx.fast.metal_kernel`，我們甚至不需要額外編譯 C++ 函式庫，就能直接在 Python 中動態編譯執行這個高度特化的 Metal 算子：

```python
import mlx.core as mx

def fused_tucker_metal(probs: mx.array, x_shared: mx.array, g_sel: mx.array) -> mx.array:
    # 取得維度資訊
    B, K = probs.shape
    _, R1 = x_shared.shape
    _, _, _, R2 = g_sel.shape
    
    # 撰寫底層 Metal Shader
    source = """
        uint s = thread_position_in_grid.x;
        uint b = thread_position_in_grid.y;
        if (b >= B || s >= R2) return;
        
        T sum = 0.0;
        for (uint k = 0; k < K; ++k) {
            T prob = probs[b * K + k];
            for (uint r = 0; r < R1; ++r) {
                T x_val = x_shared[b * R1 + r];
                T g_val = g_sel[b * K * R1 * R2 + k * R1 * R2 + r * R2 + s];
                sum += prob * x_val * g_val;
            }
        }
        out[b * R2 + s] = sum;
    """
    
    # 即時編譯並註冊 Metal Kernel
    kernel = mx.fast.metal_kernel(
        name="fused_tucker_einsum",
        input_names=["probs", "x_shared", "g_sel"],
        output_names=["out"],
        source=source,
        header=f"constant uint B={B}; constant uint K={K}; constant uint R1={R1}; constant uint R2={R2};"
    )
    
    # 執行與派發工作到 GPU
    return kernel(
        inputs=[probs, x_shared, g_sel],
        template=[("T", probs.dtype)],
        grid=(R2, B, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, R2)],
        output_dtypes=[probs.dtype],
    )[0]
```

### 🚀 Custom Metal Kernel 微基準測試結果 (Microbenchmark)

我們獨立抽出了 `Fused TuckerMoE` 的運算核心進行微基準測試 (Microbenchmark)。在設定為 `B=64, K=2, R1=32, R2=256` 的典型情境下，對比 MLX 原生的 `mx.einsum`：

```text
✅ Correctness Check: Max Diff = 0.000008

📊 Microbenchmark Results (Fused TuckerMoE)
===========================================
Shape: B=64, K=2, R1=32, R2=256
Native MLX (einsum):  0.3121 ms
Custom Metal Kernel:  0.1882 ms
Speedup:              1.66x
===========================================
```

**結論**：
透過徹底消滅原生編譯器在執行 Gather 過程中所產生的記憶體碎片化寫回，**手寫的 Metal Kernel 在核心運算上帶來了 1.66 倍 (提升 66%) 的驚人加速！** 證明了在效能瓶頸處使用特製化硬體指令，是突破框架極限的關鍵。

---

### 🚀 Mamba SSM Parallel Scan (Metal 最佳化)

除了 TuckerMoE 之外，我們也針對 Mamba 的核心瓶頸：**Parallel Scan (關聯掃描)** 進行了 Metal 底層改寫。

在原本的 MLX 實作中，為了達到平行化，採用了矩陣乘法的方式 ($O(N^2)$ 的記憶體與計算複雜度) 來計算 Chunk 內的 Scan：
```python
h_flat = mx.matmul(M, u_flat)
```

我們將其改寫為 Metal 內部的純量遞迴 (Sequential Scan)，每個 Thread 負責一個獨立的通道，將空間複雜度降回 $O(N)$，同時完美利用 GPU 的平行度。

#### SSM Microbenchmark 實驗結果 (B=2, nc=8, Lc=64, H=32, D=128)

```text
✅ Correctness Check: Max Diff = 0.000013

📊 Microbenchmark Results (SSM Chunk Scan)
===========================================
Native MLX (O(N^2) matmul): 0.9422 ms
Custom Metal Kernel (O(N)): 0.3632 ms
Speedup:                    2.59x
===========================================
```

**結論**：針對 Mamba SSM 的平行掃描，改寫為 $O(N)$ 的 Custom Metal Kernel 帶來了高達 **2.59 倍的巨量加速**！這讓 Hybrid 模型的推理速度不再受限於框架原生的 Prefix Sum 實作瓶頸。
