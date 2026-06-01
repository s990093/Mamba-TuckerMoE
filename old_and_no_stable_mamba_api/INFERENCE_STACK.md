# Mamba3-XR：`inference/` 技術說明與穩定版指引

本文整理目前 `inference/` 目錄的架構、技術元件、已知數值／行為風險，以及**建議的穩定執行路徑**（純 MLX 取樣與較保守的圖編譯設定）。極致效能用的自訂 Metal 取樣、不對稱量化等見 `lib/` 與 `experimental/`。

---

## 0. 目錄結構（整理後）

| 路徑 | 用途 |
|------|------|
| `benchmark_mlx.py`、`stream_mlx.py` | 主要 CLI 入口 |
| `run_stable_*.sh` | 穩定優先捷徑 |
| `lib/` | 模型與 fused 取樣核心（`mlx_hybrid_infer`、量化與 fused sampling） |
| `tools/` | 剖析、A/B benchmark、繪圖、KV 分析、Metal 原型腳本 |
| `results/` | 報告用輸出的 JSON／PNG（可從 `--help` 改路徑） |
| `tokenizer/` | 本地 tokenizer |
| `experimental/` | 極速 shell 腳本說明 |

---

## 1. 整體架構（資料流）

1. **權重**：`resolve_mlx_checkpoint` / `strict_load_and_convert`（`lib/mlx_hybrid_infer.py`）  
   - 支援 `.npz`、`.pt`（需 PyTorch）；首次成功從 `.pt` 載入後會寫入同檔名 `.npz` 側車，之後執行會**優先讀 `.npz`**（較快）。  
   - 本倉庫預設 SFT 權重檔名：`checkpoint_sft_s27510_model_only.pt`（`run_stable_*.sh` 已預設此路徑）。
2. **模型**：`Mamba3LanguageModel` + `Mamba3Config`  
   - Mamba 區塊：chunk 化 parallel scan（與訓練 `train.py` 數學對齊）。  
   - Transformer 區塊：GQA + `mx.fast.scaled_dot_product_attention`。  
   - MoE：`TuckerMoE`（router + 低秩核心）；可選 **einsum 融合**、**full fuse**（皆為自訂 Metal kernel 路徑）。
3. **Prefill**：整段 prompt 一次或編譯後前向，`caches` 填入 KV / Mamba 狀態。  
4. **Decode**：每步 `(B=1, T=1)` + 更新 caches；可 **外層單步 `mx.compile`**（throughput）或 **逐層 compile**／**eager**。  
5. **取樣**：`benchmark_mlx.sample_decode_token`  
   - 預設路徑：penalty（若開）+ MLX `argmax` / `mx.random.categorical`。  
   - 選用 `--fused-sample-metal` / `--fused-sample-metal-v2` 時改走**自訂 Metal kernel**（與純 MLX 路徑在浮點與演算法細節上可能略有差異）。

---

## 2. 目錄內檔案職責

| 檔案 | 角色 | 穩定版建議 |
|------|------|------------|
| `lib/mlx_hybrid_infer.py` | 模型、checkpoint、compile 附件、Mamba scan、Tucker 融合 kernel | **核心依賴**；若要最大一致性可關閉 `tucker_einsum_fuse` |
| `benchmark_mlx.py` | Prefill/decode 基準、`sample_decode_token`、inference-type 預設、工具函式（cache materialize、padding 等） | **核心依賴**；穩定跑法見下方腳本 |
| `stream_mlx.py` | 互動／Rich 串流生成，重用 `benchmark_mlx` 的取樣與 cache 邏輯 | **建議入口**；預設偏 throughput，請用 `run_stable_stream.sh` 覆寫 |
| `lib/mlx_mixed_quant.py` | MoE 非對稱量化（router int4、部分 int8） | **實驗**（數值誤差較大風險） |
| `lib/fused_sampling_metal.py` | 取樣 Metal v1（多 dispatch） | **實驗** |
| `lib/fused_sampling_metal_v2.py` | 取樣 Metal v2（單 dispatch、未歸一化 CDF 等） | **實驗** |
| `tools/custom_metal_ssm.py`、`tools/custom_metal_tucker.py` | Metal 原型／研究腳本 | **實驗** |
| `tools/profile_mlx_infer.py`、`mlx_profile_components.py`、`mlx_fine_decode_profile.py` | 剖析 | 開發用 |
| `tools/bench_optimizations_ab.py`、`plot_decode_compile_comparison.py` | A/B 與繪圖 | 實驗／報告 |
| `tools/analyze_kv_cache_sizes.py`、`test_profile_mem_check.py` | 記憶體／簡單檢查 | 工具 |
| `tokenizer/` | Hugging Face 相容 tokenizer 資產 | 必填（路徑可 `--tokenizer`） |
| `results/` | 基準輸出圖表與 JSON | 可刪除再生 |
| `experimental/` | 極速 shell、實驗說明 | 見該目錄 `README.md` |

---

## 2b. ChatML 推論前綴（與 SFT 對齊）

`benchmark_mlx.py` / `stream_mlx.py` 預設將 `--prompt` 當成**純使用者內文**，在 tokenize 前先包成：

`<|im_start|>user\n{內文}<|im_end|>\n<|im_start|>assistant\n`

並使用 `tokenizer.encode(..., add_special_tokens=False)`，與常見 SFT CLI（如 `sft_cli.py` → `lima_to_bin._format_conversation`）的假設一致：**不要**再把整包已含 `<|im_start|>assistant\n` 的 ChatML 貼進 `--prompt`。若必須沿用舊的「字串原樣 encode」行為，請加 **`--raw-prompt`**。

---

## 3. 為什麼「開很多優化」容易覺得輸出偏差大？

以下為程式碼與設計上**已知的失真來源**，可疊加放大主觀「胡言亂語感」或與 PyTorch 參考實作不一致：

1. **編譯圖 + 未物化的 cache**  
   `benchmark_mlx.py` 註解說明：prefill 在編譯圖輸出下若不做 cache materialize，decode 可能異常（例如 argmax 常落到 token 0）。**Throughput 預設在 benchmark 會 materialize；`stream_mlx` 預設為追求速度會跳過**，較不穩。
2. **4-bit／8-bit 權重量化**  
   大幅降低頻寬，但 logits 分佈與 fp16/bf16 權重不同，取樣後文字品質下降是預期現象。
3. **Tucker einsum fuse / full fuse**  
   與分段 matmul+einsum 的參考路徑在累加順序與 fp 捨入上可能不同；多數情況接近，但不保證 bit-identical。
4. **自訂 Metal 取樣（v1 / v2）**  
   溫度、min-p、top-p、penalty 的實作與 MLX 預設路徑在細節（排序、mask、隨機數）上不完全相同；追求可重現性或論文對齊時應關閉。
5. **`--lookahead-router`、`--kmoe-no-gather`**  
   明確標為實驗，改變路由或計算圖，與訓練時行為可能不一致。
6. **`--inference-type throughput`**  
   全圖 compile + parallel scan，除錯與「逐步對齊 PyTorch」時應改用 `safe` 或 `eager`。

---

## 4. 建議的「穩定版」定義（本 repo 約定）

- **取樣**：只用 MLX（**不要**加 `--fused-sample-metal` / `--fused-sample-metal-v2`）。  
- **圖行為**：`--inference-type safe`（compiled prefill + **eager decode**，減少 decode 端 compile 與 cache 耦合問題）。  
- **Cache**：串流時加 `--materialize-caches`。`benchmark_mlx.py` 預設即會物化 caches（除非你顯式傳 `--no-materialize-caches`）。  
- **Tucker**：若仍懷疑融合 kernel，加 `--no-tucker-einsum-fuse` 走參考分解路徑。  
- **量化**：穩定優先時使用 `--quantize 0`；要省頻寬再試 8-bit，最後才 4-bit。

實際一鍵指令：

- 串流：`sh inference/run_stable_stream.sh --prompt "..."`  
- 基準：`sh inference/run_stable_benchmark.sh`（可外加參數）

---

## 5. 與論文／簡報的關係

- 高 **tok/s** 數字通常來自：`bf16`、`--quantize 4`、`--full-decode-compile`、`--tucker-einsum-fuse`、greedy、無 penalty、不 materialize cache 等組合（見 `experimental/` 內腳本）。  
- 該組合**偏效能展示**，與「最像訓練時精確行為」不是同一目標；比對品質或除錯請切回本文件第 4 節。

---

## 6. 依賴（簡列）

- 執行：`mlx`、`numpy`、`transformers`（tokenizer）。  
- 載入 `.pt`：另需 `torch`。  
- 可選 UI：`rich`（`stream_mlx` 美化輸出）。

---

*最後更新：對應目前倉庫 `inference/` 布局；若你新增算子或改預設旗標，請同步更新本檔「目錄內檔案職責」一節。*
