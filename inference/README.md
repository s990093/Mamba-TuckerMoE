# Mamba3-XR Inference（Apple Silicon / MLX）

## 目錄一覽

| 路徑 | 用途 |
|------|------|
| `stream_mlx.py`、`benchmark_mlx.py` | **主要入口**（串流／基準） |
| `run_stable_stream.sh`、`run_stable_benchmark.sh` | 品質優先捷徑 |
| [`lib/`](./lib/) | 模型與 fused 取樣實作 |
| [`tools/`](./tools/) | 剖析、A/B、繪圖、分析腳本 |
| [`results/`](./results/) | 工具腳本輸出的圖表／JSON |
| [`experimental/`](./experimental/) | 極速 shell 與實驗說明 |
| `tokenizer/` | 本地 tokenizer |

## 穩定版 vs 實驗

| 目標 | 入口 | 備註 |
|------|------|------|
| **穩定、品質優先** | `sh inference/run_stable_stream.sh` | 預設 `--checkpoint checkpoint_sft_s27510_model_only.pt`（同目錄 `.npz` 優先） |
| **穩定 + 對話 REPL** | `sh inference/run_stable_stream.sh --interactive`（或 `-i`） | 自動加 `--stop-on-eos`、`--plain-output` |
| **穩定基準測試** | `sh inference/run_stable_benchmark.sh` | 預設 128 decode tokens |
| **技術細節與失真原因** | [`INFERENCE_STACK.md`](./INFERENCE_STACK.md) | 建議讀過再調旗標 |
| **極速組合** | [`experimental/`](./experimental/) | 高 TPS；行為可能與保守路徑不同 |

## 手動呼叫範例

Repo 根目錄、`./.venv` 已建前提下：

```bash
# --prompt is plain user text (ChatML-wrapped automatically); use --raw-prompt for a literal string.
python inference/stream_mlx.py --prompt "Hello" ...
python inference/benchmark_mlx.py --decode-tokens 64 ...
python inference/tools/profile_mlx_infer.py --help
```

## Fused Metal 取樣

`lib/fused_sampling_metal*.py` 為實驗路徑；與 MLX 預設取樣在數值細節上可能不同，說明見 **`INFERENCE_STACK.md`**。
