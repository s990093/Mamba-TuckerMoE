使用 GEMV
計算定義：GEMV 是通用矩陣乘法 (GEMM) 的一個特例，即 GEMM 中當 \(M=1\) 或 \(N=1\) 的情況。深度學習應用：在深度學習，特別是大語言模型（LLM）的推論（Inference）階段，GEMV 是最關鍵的運算之一，用於線性層（Linear Layer）的計算。效能瓶頸：由於 GEMV 操作通常是記憶體頻寬受限（Memory-bound）而非計算受限（Compute-bound），因此在 GPU 上進行優化時（如提升記憶體存取效率）至關重要。實現與庫：BLAS 規範中定義了此操作，許多高效能函式庫如 Intel MKL、cuBLAS、昇騰 CANN 都對其進行了優
