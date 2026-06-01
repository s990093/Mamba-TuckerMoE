# `inference/tools/`

開發與分析用腳本（**非**日常生成入口；生成請用上一層的 `stream_mlx.py` 或 `run_stable_stream.sh`）。

| 腳本 | 說明 |
|------|------|
| `profile_mlx_infer.py` | MLX 層級／Metal 瓶頸剖析 |
| `mlx_profile_components.py`、`mlx_fine_decode_profile.py` | 由上者匯入的細部計時模組 |
| `bench_optimizations_ab.py` | Tucker fuse / lookahead router 等 A/B |
| `plot_decode_compile_comparison.py` | 呼叫 `benchmark_mlx.py` 比較 compile 並寫入 `results/` |
| `analyze_kv_cache_sizes.py` | KV / hybrid cache 記憶體估算 |
| `test_profile_mem_check.py` | 隨機權重的記憶體煙霧測 |
| `custom_metal_ssm.py`、`custom_metal_tucker.py` | Metal 原型實驗 |
