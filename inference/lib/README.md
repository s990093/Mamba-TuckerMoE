# `inference/lib/`

推論用 **Python 模型實作與取樣輔助**。由 `benchmark_mlx.py`／`stream_mlx.py` 透過 `sys.path` 載入，請勿當成獨立套件安裝。

- `mlx_hybrid_infer.py`：Mamba3 + TuckerMoE、`mx.compile` 附件、checkpoint 載入  
- `mlx_mixed_quant.py`：MoE 非對稱量化（實驗）  
- `fused_sampling_metal.py`、`fused_sampling_metal_v2.py`：自訂 Metal 取樣（實驗）
