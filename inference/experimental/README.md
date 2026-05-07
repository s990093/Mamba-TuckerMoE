# 實驗與極速路徑（非預設穩定行為）

此目錄放**效能極限／自訂 Metal 取樣**相關的 shell。模型與 fused 核心實作在上一層的 [`../lib/`](../lib/)。

若你關心輸出品質、與參考實作對齊、或避免 compile+cache 邊角案例，請使用：

- `inference/run_stable_stream.sh`
- `inference/run_stable_benchmark.sh`

並閱讀 `inference/INFERENCE_STACK.md`。

## 腳本

| 腳本 | 說明 |
|------|------|
| `stream_fast_metal.sh` | 高吞吐量串流：`bf16`、4-bit、einsum fuse、full decode compile、`--fused-sample-metal-v2`、greedy |
| `bench_pure_metal.sh` | 同理念之 `benchmark_mlx.py` 壓力測（預設 `--no-materialize-caches`） |

從 repo 根目錄：

```bash
sh inference/experimental/stream_fast_metal.sh --prompt "Your prompt"
sh inference/experimental/bench_pure_metal.sh --decode-tokens 2048
```
