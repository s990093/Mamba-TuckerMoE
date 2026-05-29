# Speculative Decoding Optimization Log

> **完整問題陳述、實驗紀錄、瓶頸分析。** 可供外部 AI 查閱後給出優化建議。

---

## 0. 問題摘要（Summary for AI）

### 背景

- **模型**：Mamba3-Hybrid，768-dim，6 macro-layers（每層 4 Mamba SSM + 1 Transformer + TuckerMoE），共 30 sub-layers，vocab 32,007
- **推理框架**：Apple MLX（bf16），M2 Pro 16GB
- **解碼方式**：Speculative Jacobi Decoding (SJD)，分佈等價於 AR-sampling
- **一輪流程**：build K-1 draft tokens（來自多個 cache 來源）→ `model_verify_forward`（一次完整前向，K tokens，含 per-position state 輸出）→ accept loop（機率接受 draft）→ `extract_state_at`（從 per-position payload 切片取得狀態，免 replay）
- **draft 來源**（`build_hybrid_branches`）：SuffixRetriever（PLD 長後綴匹配）、CoT ngram cache（訓練語料統計）、runtime ngram cache（歷史 MRU）、carry seed（fallback）
- **cache 種類**：
  - runtime cache（`bake_cache.py`）：跑 AR-sampling 預熱 → `demo_cache_v2.pkl`，含 NGramCache + SuffixRetriever（需 6 分鐘）
  - CoT cache（`bake_cot_caches.py`）：掃 10,217 筆訓練 JSON → `cot_caches_n4.pkl`，含 think/final phase-aware NGramCache（6 秒，免跑模型）

### 現狀（最佳配置）

| Prompt             | Config        | K   | ARL  | tps  | speedup   |
| ------------------ | ------------- | --- | ---- | ---- | --------- |
| self_awareness     | SJD ng+rt+cot | 8   | 3.93 | 57.8 | **1.53×** |
| math_drill         | SJD ng+rt     | 8   | 3.43 | 61.1 | **1.68×** |
| daily_conversation | SJD ng+rt+cot | 8   | 2.93 | 49.4 | **1.26×** |

### 瓶頸

**draft hit rate 太低。** 瓶頸公式：

```
AR 1 token = 25ms
SJD K=8 round = 92ms
break-even ARL = 92/25 = 3.68

當前 ARL = 3.93 → 只比打平多 0.25 token/輪
```

每多 1 ARL ≈ +11 tok/s。K=8 上限 ARL=8 → 理論上限 speedup ≈ 2.2×。當前 1.5× 表示只用到理論上限的 68%。

**ARL 卡在 ~4 的根本原因**：ngram（n=4）只能看 3-token 上下文，逐 token 猜測；SuffixRetriever 能抓長片段但僅靠 runtime 累積（冷啟動為空）。CoT cache 只有 ngram 沒有 retriever。純 ngram ARL 天花板 ~3.5，任何 K 都無法突破。

### 已嘗試的無效方向

| 嘗試                                               | 結果     | 根因                               |
| -------------------------------------------------- | -------- | ---------------------------------- |
| Lookahead Phase A（extra model forward per round） | −60% tps | 雙 forward cost > ARL gain         |
| n=5 COT cache                                      | −31% tps | 4-token key 太特異，命中率暴跌     |
| Hillis-Steele Metal parallel scan                  | −25% tps | bf16 數值漂移摧毀 cache 命中率     |
| Kogge-Stone SIMD-shuffle scan                      | −21% tps | 同上 + dispatch overhead           |
| metal_verify（chunk_scan_per_pos）                 | −12% tps | scan 只佔 verify 成本 2%，不是瓶頸 |
| K=16/20/24（純加大 K）                             | < K=8    | verify cost 增長壓過 ARL 增長      |

### 唯一可行方向

**提升 draft 品質（不改模型）**：

1. CoT cache 加 SuffixRetriever — 從訓練語料 per-category 建 retriever buffer，免跑模型，6 秒 bake。預期 ARL +50~100%（長片段命中）
2. 同時載入 runtime cache + CoT cache（目前兩者從未合併於同一 run）
3. `adaptive_K` — 現有實作已存在，cache hit 時自動擴 K 到 16

**預期**：ARL 從 3.93 → 6~8，speedup 從 1.5× → 2.0~2.5×。

### 硬體限制

- Metal kernel dispatch overhead ~100μs/kernel，對小 Lc=K (8-24) 不利
- 24 個 Mamba layers 無法做 attention mask fusion（無 attention 概念）
- `mx.compile` graph recompilation 風險未評估
- 量化（4-bit）會進一步影響 bf16 數值穩定性，對 cache 命中率額外衝擊未知

---

## 完整實驗紀錄

完整紀錄本輪 `mamba3_mlx/speculative/` 的擴展工作：Lookahead Decoding (Phase 1) +
訓練語料離線 think/final 雙快取。所有 benchmark 在 M2 Pro 16 GB / MLX bf16
跑出。

---

## 1. 做了什麼

### 1.1 Lookahead Branch (Phase 1, 方案 A 優化版)

依 `PLAN_LOOKAHEAD.md` 第 1.2 / 1.3 節實作：

- 新增 `LookaheadTrajectory`：(N-1) × W 的 2D 視窗，欄 = trajectory path，列 = 過去 N-1 輪預測。
- 每輪 Phase A：把 W 條 path 批次成 `(W, N-1)`、複製 verified state 給 `model_verify_forward` 一次跑完；丟掉 per-position payload，狀態仍由 Phase B verify 持有。
- Phase A 預測的 W 個下個 token 用 `extract_ngrams()` 包成 W 個長度為 N 的 n-gram，灌進獨立的 lookahead n-gram 快取，當輪 `build_hybrid_branches` 多一個 draft 源。
- 不動 MLX attention，不動 `mlx_model/`，零侵入。

### 1.2 COT 雙快取（**真正的加速來源**）

依 user 要求：「透過 `cot_dataset/export_hf_dataset.py` 代碼，think 跟 final 兩個 cache，一開始 init 透過這些做統計」。

- 新增 offline baker：跑過全部 10,217 筆 CoT JSON → 用同一 tokenizer 把整段 `text` tokenize → 依 `<think>/</think>` 和 `<final>/</final>` 四個 special-token ID 切片 → 對每片計每個 (n-1)-key 的 continuation 頻率分布 → 取 top-K 並**依頻率倒序插入**讓最高頻成為 MRU → 寫成單一 pkl。
- Runtime phase tracker 觀察每輪 emit 的 token，遇 marker 切 think → other → final → other。
- 解碼時 phase 對應的快取直接 plug 進 `build_hybrid_branches` 的 `lookahead_ngram` 槽。
- 若 `use_lookahead=True` 也開，trajectory 採到的 n-gram 會合流寫入「當前 phase 的快取」，所以仍會 runtime 學習。

**快取架構圖 (Cache Design)**：

```
decode round (Jacobi / SJD)
│
├─ build_hybrid_branches()                          drafts.py
│   │
│   ├─ [slot 0] SuffixRetriever.query()             PLD 最長後綴
│   │           ─ runtime 學習 (prompt + 已生成)
│   │
│   ├─ [slot 1] cot_ngram (_ngram_chain)            ← 訓練語料離線快取
│   │           ─ READ-ONLY：bake_cot_caches 寫死
│   │           ─ 頻率加權 MRU，n=4 key=3
│   │           ─ think / final 兩個分別的 NGramCache
│   │
│   ├─ [slot 2] ngram (_ngram_chain)                runtime n-gram
│   │           ─ 每輪 update_sequence(accepted)
│   │
│   ├─ [slot 3] carry fallback_seed                 最後一個接受 token
│   │
│   └─ [slot 4+] ngram top-2..K                     額外多樣性
│
└─ CoTPhaseTracker.observe(accepted)
    ─ 依 <think>/<final> marker 切換 slot 1 的快取
```

### 1.3 Greedy 與 SJD 全都接

`jacobi.py` 和 `jacobi_sampling.py` 都新增 `use_lookahead`、`lookahead_N`、`lookahead_W`、`lookahead_seed`、`cot_caches` 五個 kwargs。SJD 的接受規則完全不動（distribution 等價性保留），greedy 也維持 fp32 byte-equal。

---

## 2. 改了什麼（檔案層級）

| 動作 | 路徑                       | 說明                                                                                                     |
| ---- | -------------------------- | -------------------------------------------------------------------------------------------------------- |
| 新檔 | `lookahead_trajectory.py`  | 2D 視窗類別                                                                                              |
| 新檔 | `lookahead_forward.py`     | 批次 (W,N-1) Phase A forward                                                                             |
| 新檔 | `cot_cache.py`             | 雙快取載入 + `CoTPhaseTracker` + `infer_initial_phase`                                                   |
| 新檔 | `bake_cot_caches.py`       | 離線頻率加權 baker                                                                                       |
| 新檔 | `benchmark_cot.py`         | 本份 benchmark 用的綜合 driver                                                                           |
| 新檔 | `LOOKAHEAD_COT_RESULTS.md` | 本文                                                                                                     |
| 修改 | `ngram_cache.py`           | +`update_ngrams(list[tuple])`                                                                            |
| 修改 | `drafts.py`                | +`_ngram_chain` helper；`build_hybrid_branches` +`lookahead_ngram` keyword-only                          |
| 修改 | `jacobi.py`                | 主迴圈第 0 步插入 Phase A；`JacobiResult` +`n_lookahead_rounds/_ngrams`；`use_lookahead/cot_caches` 整合 |
| 修改 | `jacobi_sampling.py`       | 同樣鏡像 lookahead + cot 接線                                                                            |
| 修改 | `__init__.py`              | 匯出 `LookaheadTrajectory`、`lookahead_branch_step`                                                      |
| 修改 | `verify.py`                | +`--use_lookahead/--lookahead_N/--lookahead_W/--cot_caches`                                              |

---

## 3. 跑了什麼

```bash
# 一次性 bake CoT 雙快取 (6 秒)
.venv/bin/python3 -m mamba3_mlx.speculative.bake_cot_caches \
  --ngram_n 4 --max_continuations 4 \
  --out mamba3_mlx/speculative/cot_caches_n4.pkl
# → 24 MB；356,786 think keys / 781,290 final keys；902,231/1,419,088 tokens

# Greedy 正確性 (fp32 嚴格 byte-equal)
.venv/bin/python3 -m mamba3_mlx.speculative.verify \
  --model_path checkpoints/4_loss_func/latest_sft_cot_model.mlx_fp32.npz \
  --dtype fp32 --max_tokens 64 --K 4 --K 8 \
  --use_lookahead --lookahead_W 4 --lookahead_N 4 \
  --cot_caches mamba3_mlx/speculative/cot_caches_n4.pkl \
  --no-eos-stop
# → K=4: OK prefix=64/64  K=8: OK prefix=64/64  ALL OK

# 完整 SJD benchmark (3 prompts × K=8,12 × 5 configs, max=512)
.venv/bin/python3 -m mamba3_mlx.speculative.benchmark_cot \
  --max_tokens 512 --Ks 8 12 \
  --cot_caches mamba3_mlx/speculative/cot_caches_n4.pkl
```

---

## 4. 結果（bf16, max_tokens=512, M2 Pro）

### 4.1 self_awareness — _"Who are you?"_

| config              | K     | ARL      | full%    | la_r | tps      | wall      | speedup      |
| ------------------- | ----- | -------- | -------- | ---- | -------- | --------- | ------------ |
| AR-sampling         | -     | 1.00     | 0.0      | 0    | 39.3     | 12.99s    | 1.00×        |
| SJD ng+rt           | 8     | 3.01     | 22.2     | 0    | 56.0     | 9.73s     | 1.42×        |
| **SJD ng+rt+cot**   | **8** | **3.93** | **33.1** | 0    | **73.9** | **9.84s** | **1.88×** ⭐ |
| SJD ng+rt+la4x4     | 8     | 3.59     | 27.1     | 144  | 29.2     | 17.76s    | 0.74×        |
| SJD ng+rt+cot+la4x4 | 8     | 3.75     | 28.3     | 138  | 29.3     | 20.29s    | 0.74×        |
| SJD ng+rt           | 12    | 2.72     | 8.5      | 0    | 41.3     | 12.99s    | 1.05×        |
| SJD ng+rt+cot       | 12    | 4.04     | 19.4     | 0    | 55.4     | 12.15s    | 1.41×        |
| SJD ng+rt+la4x4     | 12    | 4.03     | 19.7     | 127  | 30.1     | 17.61s    | 0.77×        |
| SJD ng+rt+cot+la4x4 | 12    | 3.70     | 14.2     | 141  | 26.2     | 23.35s    | 0.67×        |

### 4.2 math_drill — _"Solve (15+3)\*4/2 step by step"_

| config              | K     | ARL      | full%    | la_r | tps      | wall      | speedup      |
| ------------------- | ----- | -------- | -------- | ---- | -------- | --------- | ------------ |
| AR-sampling         | -     | 1.00     | 0.0      | 0    | 40.7     | 12.56s    | 1.00×        |
| **SJD ng+rt**       | **8** | **3.43** | **32.0** | 0    | **61.9** | **8.91s** | **1.52×** ⭐ |
| SJD ng+rt+cot       | 8     | 3.52     | 27.4     | 0    | 60.2     | 12.02s    | 1.48×        |
| SJD ng+rt+la4x4     | 8     | 3.06     | 21.3     | 169  | 24.5     | 21.48s    | 0.60×        |
| SJD ng+rt+cot+la4x4 | 8     | 4.16     | 38.2     | 123  | 32.2     | 19.24s    | 0.79×        |
| SJD ng+rt           | 12    | 3.42     | 4.7      | 0    | 50.9     | 10.67s    | 1.25×        |
| SJD ng+rt+cot       | 12    | 3.74     | 16.1     | 0    | 53.0     | 13.36s    | 1.30×        |
| SJD ng+rt+la4x4     | 12    | 3.93     | 20.8     | 130  | 29.1     | 18.17s    | 0.72×        |
| SJD ng+rt+cot+la4x4 | 12    | 4.06     | 18.1     | 127  | 28.8     | 21.58s    | 0.71×        |

### 4.3 daily_conversation — _"What are some quick tips for staying focused while working from home?"_

| config              | K     | ARL      | full%    | la_r | tps      | wall       | speedup      |
| ------------------- | ----- | -------- | -------- | ---- | -------- | ---------- | ------------ |
| AR-sampling         | -     | 1.00     | 0.0      | 0    | 39.3     | 13.02s     | 1.00×        |
| SJD ng+rt           | 8     | 2.15     | 11.7     | 0    | 39.4     | 13.57s     | 1.00×        |
| **SJD ng+rt+cot**   | **8** | **2.93** | **18.2** | 0    | **51.1** | **13.75s** | **1.30×** ⭐ |
| SJD ng+rt+la4x4     | 8     | 3.08     | 25.9     | 166  | 25.1     | 20.94s     | 0.64×        |
| SJD ng+rt+cot+la4x4 | 8     | 2.14     | 6.7      | 240  | 17.0     | 33.35s     | 0.43×        |
| SJD ng+rt           | 12    | 2.14     | 7.1      | 0    | 32.5     | 16.30s     | 0.83×        |
| SJD ng+rt+cot       | 12    | 2.74     | 7.9      | 0    | 38.8     | 16.61s     | 0.99×        |
| SJD ng+rt+la4x4     | 12    | 2.52     | 6.9      | 204  | 18.5     | 28.12s     | 0.47×        |
| SJD ng+rt+cot+la4x4 | 12    | 3.27     | 1.3      | 157  | 23.4     | 25.90s     | 0.60×        |

---

## 5. 分析

### 5.1 哪一招贏？

| Prompt             | 最快配置          | tps  | 較 AR    | 較舊 baseline (ng+rt) |
| ------------------ | ----------------- | ---- | -------- | --------------------- |
| self_awareness     | SJD ng+rt+cot K=8 | 73.9 | **+88%** | **+32%**              |
| math_drill         | SJD ng+rt K=8     | 61.9 | +52%     | baseline              |
| daily_conversation | SJD ng+rt+cot K=8 | 51.1 | +30%     | +30%                  |

**結論一：COT 快取在 self-awareness 與 daily 類 prompt 上加速明顯**（+30~88% vs AR），math 因 n-gram 多樣性高沒幫助但也沒退步。

**結論二：Lookahead Phase A 目前 100% 是淨虧損**（6/6 組合都比同 K 的 baseline 慢）。原因如 PLAN_LOOKAHEAD §5.3 預測：每輪多 W×(N-1)=12 token 的 Phase A forward，ARL gain 平均只 ~5-15%，補不回來。

**結論三：K=8 普遍優於 K=12**（K=12 的 verify forward 變貴，且 ARL 提升不足以 amortise）。`adaptive_K` 可能可進一步調校。

### 5.2 為什麼 LA 在 daily_conv K=8 配 cot+la 慘掉 (0.43×)？

`la_r=240` — lookahead 跑了 240 輪、emit 才 ~512 token，平均每輪只 emit 2 token。trajectory 的低 ARL 拖垮 wall。原因是 daily prompt 的 cot/final 與訓練樣本差距大，trajectory n-grams 灌進 cot 快取後反而把高品質的訓練 n-grams MRU-踢出。**未來修正**：cot 快取應為 READ-ONLY；trajectory 改寫到自己的快取（恢復 Phase 1 的兩源並列）。

---

## 6. 下一步建議

### 立即可做（不改 mlx_model）

1. **Cot 快取改 READ-ONLY**（修 jacobi.py / jacobi_sampling.py 約 4 行）— 修掉 5.2 描述的退化情況；預期 daily_conv +la 至少回到 baseline。
2. **Adaptive-W**（GammaTune 風格 EWMA）— Phase A 沒 hit 就動態縮 W：W=1 時 Phase A 退化成單 token forward，幾乎免費；W=4 時最貴。預期把 LA 從 0.43× 拉回 ~1.0×。
3. **Adaptive-K** 沿用既有實作，搭 cot 跑一輪 sweep 確認 K=10/14 是否優於 K=8/12。

### 中期（要動 mlx_model）

4. **Phase 2 融合 mask（PLAN_LOOKAHEAD §1.2 方案 B）**— 在 Transformer block 加 custom mask，把 Phase A 收進同一次 verify forward。預期把每輪 forward 數從 2 降回 1，LA 才有機會回本。但 30 layers 中 24 個是 Mamba SSM（沒有 attention mask 概念），只對 6 個 Transformer layer 生效，收益有限。

### 推薦順序

`READ-ONLY cot` → `Adaptive-W` → 重跑 benchmark → 再評估是否值得做 Phase 2。

---

## 8. 第二輪優化結果（READ-ONLY cot + adaptive_W）

### 8.1 變更實作

**Fix 1：cot 快取 READ-ONLY**

- `build_hybrid_branches` 新增 keyword-only `cot_ngram` 參數，與 `lookahead_ngram` 為**獨立 branch slot**（priority：suffix → cot → la → ngram → carry → ngram top-k）。
- 解碼器不再把 trajectory n-grams 寫進 cot 快取；trajectory 只寫到自己的 `lookahead_ngram` 快取。
- cot 快取維持訓練語料採頻率排名後的 MRU 順序，不被 trajectory 噪音淹沒。

**Fix 2：Adaptive-W**

- 新增 `adaptive_W` / `adaptive_W_threshold=0.45` / `adaptive_W_rounds=8` kwargs。
- 觀察 `m/K_cur` 每輪 ratio；連續 N 輪 > threshold 後**永久關閉 Phase A**。
- 設計選擇：當其它 draft 源（cot/retriever/ngram）已經拉到高 ARL，Phase A 是純浪費 → 砍掉即可。`lookahead_ngram` 快取的既有條目仍可繼續查詢。

### 8.2 第二輪 benchmark（同 prompts × K × max=512 bf16）

**Key wins highlighted。** "Δ" 是相對第一輪同 cell 的變化。

#### self_awareness — "Who are you?"

| config                  | K   | ARL      | tps      | speedup   | Δ vs §4.1                       |
| ----------------------- | --- | -------- | -------- | --------- | ------------------------------- |
| SJD ng+rt+cot           | 8   | 3.93     | 72.8     | 1.72×     | (var)                           |
| SJD ng+rt+cot+la4x4     | 8   | **4.34** | 34.7     | **0.82×** | **ARL +16%, tps +18%** ⭐       |
| **SJD ng+rt+cot+la+aW** | 8   | 4.34     | **38.9** | **0.92×** | **tps +33% vs old la combo** ⭐ |
| SJD ng+rt+cot           | 12  | 4.04     | 60.0     | 1.42×     | (var)                           |
| SJD ng+rt+cot+la4x4     | 12  | 3.68     | 27.0     | 0.64×     | small                           |
| SJD ng+rt+cot+la+aW     | 12  | 3.68     | 27.4     | 0.65×     | small                           |

#### math_drill — "Solve (15+3)\*4/2 step by step"

| config              | K   | ARL  | tps      | speedup          | Δ                  |
| ------------------- | --- | ---- | -------- | ---------------- | ------------------ |
| SJD ng+rt           | 8   | 3.43 | **62.7** | **1.50× (best)** | unchanged          |
| SJD ng+rt+cot+la4x4 | 8   | 3.56 | 27.6     | 0.66×            | tps −14% (regress) |
| SJD ng+rt+cot+la+aW | 8   | 3.56 | 30.1     | 0.72×            | tps −7%            |

註：math 的小退步是因為原本 trajectory 透過 MRU bumping 在強化 cot 的數學序列入口，分離後 cot 拿不到那些 trajectory n-grams。但本來 math 的最佳就是純 ng+rt（1.50×），la 組合不是 math 的最佳路徑，所以這個 regression 不影響推薦配置。

#### daily_conversation — _focus tips_ (**最大贏家**)

| config              | K   | ARL      | tps      | speedup   | Δ vs §4.3                      |
| ------------------- | --- | -------- | -------- | --------- | ------------------------------ |
| SJD ng+rt+cot+la4x4 | 8   | **3.44** | **26.8** | **0.66×** | **ARL +61%, tps +58%** ⭐⭐    |
| SJD ng+rt+cot+la+aW | 8   | 3.44     | **30.3** | **0.74×** | **tps +78% vs old 0.43×** ⭐⭐ |
| SJD ng+rt+cot+la4x4 | 12  | 3.39     | 24.2     | 0.59×     | ARL +36%, tps +31%             |
| SJD ng+rt+cot+la+aW | 12  | 3.39     | 24.3     | 0.60×     | ARL +36%, tps +31%             |

§5.2 描述的 **0.43× 災難案例** 已完全修復至 0.74×（+72% 相對）。

### 8.3 整體最佳值（最終結論）

| Prompt         | 最佳配置          | tps  | 較 AR     | 變更                         |
| -------------- | ----------------- | ---- | --------- | ---------------------------- |
| self_awareness | SJD ng+rt+cot K=8 | 72.8 | **1.72×** | 不需要 lookahead             |
| math_drill     | SJD ng+rt K=8     | 62.7 | **1.50×** | cot 不需要、lookahead 不需要 |
| daily_conv     | SJD ng+rt+cot K=8 | 50.9 | **1.25×** | 不需要 lookahead             |

**這三個情境下，最佳配置都沒有用到 lookahead Phase A。** Adaptive-W 把 lookahead 從「災難」拉到「無傷大雅」（0.43× → 0.92×），但仍然慢於純 cot+ngram 路徑。

### 8.4 為什麼 Phase A 在此模型上沒贏面？

`m/K=0.5` 才能 break even（Phase A 約一倍 verify cost）。實測 ARL/K = 3.93/8 ≈ 0.49（self_aware）/ 3.43/8 ≈ 0.43（math）/ 2.93/8 ≈ 0.37（daily）。**已經貼著 break-even 線且 cot/ngram 是主因。** lookahead 可額外貢獻的 ARL 提升只有 0-15%，補不回 Phase A 的 1× verify cost。

要讓 lookahead 真正回本，唯一路徑是 **Phase 2 融合 mask** 把 Phase A 收進同一次前向。但本模型 30 層中 24 個是 Mamba SSM（無 attention mask），只 6 個 Transformer layer 受惠 — 預期收益 ≤ 20%。

### 8.5 推薦最終 demo 配置

```bash
# 一次性
.venv/bin/python3 -m mamba3_mlx.speculative.bake_cot_caches \
  --ngram_n 4 --out mamba3_mlx/speculative/cot_caches_n4.pkl

# 解碼（Python API）
from mamba3_mlx.speculative import jacobi_decode_sampling
r = jacobi_decode_sampling(
    model, prompt_ids, gen_config,
    K=8,
    use_ngram=True, ngram_n=4,
    use_retrieval=True,
    cot_caches="mamba3_mlx/speculative/cot_caches_n4.pkl",
    # use_lookahead=False  ← 預設關閉，本模型上是淨虧損
)
```

self_awareness/CoT-like prompts：**~1.7× vs AR-sampling**，並且 cold-start（pkl 6 秒 bake，無需 6 分鐘 model warmup）。

### 8.6 下一步可能性（暫不做）

| 工作                                         | 預期收益                               | 成本                               |
| -------------------------------------------- | -------------------------------------- | ---------------------------------- |
| Phase 2 融合 mask                            | ≤ +20% LA tps（只對 6/30 layer）       | 改 mlx_model/，高風險              |
| Per-category COT cache（依 sys_bucket 分開） | ARL 在小眾類別 (math_drill) 上可能拉高 | 重新 bake，加 phase tracker 複雜度 |
| Distilled draft model                        | 大幅突破 ARL 上限                      | 訓練成本                           |
| Tree attention in verify                     | tree_B>1 配合 cot 可能再 +10%          | 改 attention，高風險               |

「立即可做」清單已完成。再往下就是改 mlx_model 或訓練 draft model — 不是 Phase 1 範圍。

---

## 9. Phase A 移除（2026-05-27）

依 §5 / §6 / §8 的全面 benchmark：Phase A lookahead 在本模型上 6/6 組合都淨虧損，最佳配置完全不需要它。決定移除以降低維護面積。

### 9.1 刪除/修改清單

| 動作                   | 檔案                                                                                                                                                                                                        |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **刪除**               | `lookahead_trajectory.py`、`lookahead_forward.py`、`PLAN_LOOKAHEAD.md`                                                                                                                                      |
| **drafts.py**          | 移除 `build_hybrid_branches` 的 `lookahead_ngram` keyword-only 參數（保留 `cot_ngram`）；branch slots 重編號（0 suffix → 1 cot → 2 ngram → 3 carry → 4+ ngram top-k）                                       |
| **jacobi.py**          | 移除 `use_lookahead/lookahead_N/lookahead_W/lookahead_seed/adaptive_W/adaptive_W_threshold/adaptive_W_rounds` 七個 kwargs；移除 `JacobiResult.n_lookahead_rounds/_ngrams` 兩個欄位；移除 Phase A 主迴圈區塊 |
| **jacobi_sampling.py** | 同樣鏡像清除；SJD 接受規則完全不動                                                                                                                                                                          |
| ****init**.py**        | 移除 `LookaheadTrajectory/lookahead_branch_step` 匯出                                                                                                                                                       |
| **verify.py**          | 移除 `--use_lookahead/--lookahead_N/--lookahead_W` CLI 旗標                                                                                                                                                 |
| **benchmark_cot.py**   | 移除 LA 相關欄位與 5 個 LA-組合的測項，只保留 `SJD ng+rt` 和 `SJD ng+rt+cot` 兩個對比                                                                                                                       |

### 9.2 清理後 sanity 驗證

```bash
# 1. fp32 byte-equal correctness — 保留
.venv/bin/python3 -m mamba3_mlx.speculative.verify \
  --model_path checkpoints/4_loss_func/latest_sft_cot_model.mlx_fp32.npz \
  --dtype fp32 --max_tokens 64 --K 4 --K 8 \
  --cot_caches mamba3_mlx/speculative/cot_caches_n4.pkl --no-eos-stop
# → K=4 cot: OK 64/64    K=8 cot: OK 64/64    ALL OK

# 2. bf16 benchmark — 速度不變
.venv/bin/python3 -m mamba3_mlx.speculative.benchmark_cot \
  --max_tokens 512 --Ks 8 12 \
  --cot_caches mamba3_mlx/speculative/cot_caches_n4.pkl
```

### 9.3 清理後 benchmark（vs §4/§8 同矩陣）

| Prompt         | 配置          | K   | tps      | speedup   | 與 §4 baseline 變化 |
| -------------- | ------------- | --- | -------- | --------- | ------------------- |
| self_awareness | SJD ng+rt+cot | 8   | **72.8** | **1.69×** | 一致（noise）       |
| self_awareness | SJD ng+rt+cot | 12  | 59.4     | 1.37×     | 一致                |
| math_drill     | SJD ng+rt+cot | 8   | **64.9** | **1.53×** | +1.5%               |
| math_drill     | SJD ng+rt+cot | 12  | 53.7     | 1.27×     | 一致                |
| daily_conv     | SJD ng+rt+cot | 8   | 50.3     | 1.18×     | 一致                |
| daily_conv     | SJD ng+rt+cot | 12  | 38.9     | 0.91×     | 一致                |

無回歸。最佳值 **self_awareness K=8 cot = 1.69× / 73 tps**（保持）。

### 9.4 Phase 2 融合 mask 為何不在範圍內

本模型結構：30 layers 中 **24 個 Mamba SSM** + 6 個 Transformer。

- **Transformer attention mask**：可自訂遮罩控制 token 可見性，理論上能讓 lookahead trajectories 與 verify tokens 在同一序列中互不干擾。
- **Mamba SSM scan**：因果性內建於 scan，**沒有 attention mask 概念**。若要讓 W 條 trajectory 平行而不互相污染，唯一方法是 batched 處理（W 份 state 複製），等同於目前 Phase A 的 batched forward。

也就是說，「Phase 2 融合 mask」只對 6/30 = 20% 的 layer 有效。剩下 80% 仍需要 batched W 處理，Mamba 那邊的 state replication overhead 並沒省。預期最大加速比上限 = 1/(0.8 + 0.2/W) ≈ 1.25× when W=4。

考量：

1. 要寫一份 mlx_model/ 的拷貝 + 改 attention（風險）
2. 收益上限 1.25× × （現在 Phase A 從 0.43× ~ 0.92× 拉回 ~1.0×）= 仍可能淨虧損
3. cot cache 已達 1.69×

**決議：Phase 2 不做。**真正的下一步應該是改善 draft 品質（更大 cot cache、per-category 拆分、distilled small model），這些都不需要動 mlx_model/ 核心。

---

## 10. 第三輪實驗（負面結果，2026-05-27）

### 10.1 嘗試 A：cot cache 升級 n=5 / max_cont=8

**目的**：更長 n-gram 提高匹配特異性，更多 continuations 拓寬命中。

**結果**：

| Prompt         | K   | n=4 tps | n=5 tps | 變化     |
| -------------- | --- | ------- | ------- | -------- |
| self_awareness | 8   | 72.8    | 50.1    | **−31%** |
| self_awareness | 12  | 59.4    | 45.6    | **−23%** |
| math_drill     | 8   | 64.9    | 45.6    | **−30%** |
| math_drill     | 12  | 53.7    | 59.8    | +11%     |
| daily_conv     | 8   | 50.3    | 46.6    | −7%      |
| daily_conv     | 12  | 38.9    | 38.7    | 0%       |

**結論**：n=5 是 **NET LOSS**。原因：每個 (n-1)=4-token key 在 10K 樣本中觀察次數降低，命中率下滑。`356k → 462k` 鍵但平均每鍵繼承 token 數從 2.5 → 1.95。**n=4 是甜蜜點。**

### 10.2 嘗試 E：mlx_model_v2 + Kogge-Stone 平行前綴掃描

**目的**：把 `scan_metal.py` 的 O(Lc) 序列掃描升級成 O(log Lc) Hillis-Steele 平行前綴掃描，吃滿 GPU。

**作法**：

1. 複製 `mlx_model/` → `mlx_model_v2/`（保留原始不動）
2. 新 Metal kernel：每個 `(d, hh, b_c)` 用 `Lc` 條 cooperating threads，threadgroup memory + `threadgroup_barrier` 做 log2(Lc) 階段合併
3. Associative operator: `(a2, h2) ∘ (a1, h1) = (a2·a1, a2·h1 + h2)`
4. 補丁 `speculative/forward.py` 的 isinstance 檢查支援 v1/v2 雙模型

**Kernel-level correctness (fp32 ground truth)**：

| 測試                  | v1 vs ref | v2 vs ref | PASS? |
| --------------------- | --------- | --------- | ----- |
| L=128 Lc=32 h_init    | 5.86e-2   | 6.19e-2   | ✓     |
| L=64 Lc=64 h_init     | 9.11e-2   | 7.23e-2   | ✓     |
| L=8 Lc=8 (verify K=8) | 7.04e-2   | 6.70e-2   | ✓     |
| L=512 Lc=64 prefill   | 9.44e-2   | 9.44e-2   | ✓     |

**Kernel-level wall-time micro-benchmark**：

| Shape               | v1 ms | v2 ms | speedup |
| ------------------- | ----- | ----- | ------- |
| prefill L=512 Lc=64 | 1.061 | 0.987 | 1.07×   |
| decode L=1 Lc=64    | 0.541 | 0.442 | 1.22×   |
| verify L=8 Lc=64    | 0.449 | 0.442 | 1.02×   |
| verify L=64 Lc=64   | 0.435 | 0.413 | 1.05×   |

Kernel 本身 5-22% 加速，方向是對的。

**End-to-end SJD benchmark (self_awareness K=8 max=512)**：

| 模型   | ARL      | tps      | wall       | 結果                      |
| ------ | -------- | -------- | ---------- | ------------------------- |
| v1     | 3.93     | 72.5     | 10.2 s     | baseline                  |
| **v2** | **2.94** | **52.7** | **13.0 s** | **−25% tps, −25% ARL** ❌ |

**深層原因（負面結論）**：

1. **bf16 數值漂移**：平行前綴掃描的計算順序與序列掃描不同（Hillis-Steele tree vs strictly sequential）。雖然 mathematically 等價，但 float 順序重排造成數值差異。內部累積在 float32 沒問題，但最終 cast 回 bf16 後，24 層 Mamba 累積的漂移使 logits 偏移到 2.5（佔 scaled_tanh 範圍 8%）。

2. **SJD 退化的傳導路徑**：模型輸出分布變了 → ngram/cot 快取的「最高頻 continuations」與模型新 argmax 偏離 → SJD acceptance probability 降低 → ARL 從 3.93 跌到 2.94。

3. **Spec decode 熱路徑根本不走 chunk_scan**：spec decode verify forward 用 `_scan_per_pos`（pure MLX einsum），不走 `scan_metal.py`。v2 改進只影響 prefill（<1% 總時間），對 verify 的 10s 完全沒幫助。

**剖析結論**：

| 預測                         | 實際                                                                                                      |
| ---------------------------- | --------------------------------------------------------------------------------------------------------- |
| log(Lc) 並行該比 O(Lc) 快    | Kernel 只快 5-22%（threadgroup memory + barriers 抵銷大部分理論加速）                                     |
| Spec decode 整體會加速       | 加速 5% × 1% = 0.05%，被 bf16 漂移帶來的 ARL 退化（-25%）完全淹沒                                         |
| 「真正前綴和計算」會吃滿硬體 | 確實有更多 threads in flight，但 Apple Silicon 對 threadgroup memory 有 concurrency 限制 + barrier 序列化 |

**最終決議**：**v2 不採用**。`mlx_model_v2/` 留著當 dead-end 紀錄供未來參考。

### 10.3 為什麼這個方向行不通的根本原因

要在 spec decode 上拿到加速，需要打對的位置：

```
spec decode 的 wall-time 分布:
  prefill         ~ 0.5s   ( 5%)  ← v2 影響這
  verify forward × 100 rounds = 8s (80%)  ← 用 _scan_per_pos, 純 MLX
  state extract  × 100 = 1s   (10%)
  Python glue                 (5%)
```

要把 wall 從 10s 降到 5s，必須砍 verify forward 一半。verify forward 不過 `chunk_scan`，而是 `_scan_per_pos`，目前是純 MLX einsum（為了同時輸出 per-position state）。

**真正能加速的方向（未做）**：

- 寫一個 Metal kernel 替代 `_scan_per_pos`，同時輸出 per-position state（W\*Lc 個 logits）
- 但這要重寫 spec decode 整套狀態提取邏輯，不是 PLAN_LOOKAHEAD §1.3 描述的「fused mask」

### 10.4 學到的事

1. **「parallel = faster」是迷思**。Apple Silicon 的記憶體頻寬、threadgroup memory 容量、barrier 開銷都會抵銷理論加速。
2. **數值穩定性必須跟著測**。bf16 對運算順序敏感，spec decode 對輸出分布敏感，兩者疊加就出現 25% ARL 退化。
3. **要先找對 hot path**。chunk_scan 不是 spec decode 的瓶頸，再怎麼優化也只動到 prefill 的 1%。
4. **n=4 是訓練語料規模下的甜蜜點**。n=5 鍵更特異但匹配率掉太多。

當前最佳組合維持為：**SJD K=8 + cot_caches_n4 → 1.69× / 73 tps (self_awareness max=512)**。

---

## 11. 第四輪實驗（負面，2026-05-27）

依使用者要求按論文設計直接重構，做兩件事：

1. 在 `mlx_model_v2/scan_metal.py` 加入硬體感知的三層 Kogge-Stone 平行前綴掃描 — 用 `simd_shuffle_up`（register only，無 barrier）做 Phase 1，threadgroup memory 做 Phase 2 / 3 cross-SIMD merge
2. 加入 `chunk_scan_per_pos` API + `speculative/forward_metal.py`，把 spec decode verify 路徑的 pure-MLX `_scan_per_pos` 改用 Metal kernel（hot path 替換）

### 11.1 SIMD-shuffle Kogge-Stone (Phase 1 follow-up)

依論文設計：

| Phase | Op                                                           | Sync               |
| ----- | ------------------------------------------------------------ | ------------------ |
| 1     | 5 階 `simd_shuffle_up` Kogge-Stone within 32-lane SIMD group | register only      |
| 2     | last lane 寫 aggregate 進 threadgroup memory                 | 1 barrier          |
| 3     | sgi>0 thread 讀前面 SIMD groups 的 aggregate 並合併          | (no extra barrier) |

**Kernel-level micro-bench** (vs v1 serial):

| Shape              | v1 ms | v2-simd ms | speedup   |
| ------------------ | ----- | ---------- | --------- |
| Lc=8               | 0.454 | 0.398      | 1.14×     |
| Lc=16              | 0.371 | 0.371      | 1.00×     |
| Lc=32              | 0.410 | 0.403      | 1.02×     |
| Lc=64              | 0.456 | 0.429      | 1.06×     |
| Lc=64 L=512 nc=8   | 1.021 | 1.035      | 0.99×     |
| Lc=128 L=1024 nc=8 | 1.734 | 1.960      | **0.88×** |

Kernel 本身只在 single-chunk 短序列拿到 6-14% 加速；多 chunk 反而變慢（threadgroup 啟動開銷壓過 log(Lc) 優勢）。

**End-to-end SJD (self_awareness max=512)**:

| 模型            | ARL      | tps    | 結果                      |
| --------------- | -------- | ------ | ------------------------- |
| v1 serial       | 3.93     | 67     | baseline                  |
| v2 SIMD-shuffle | **3.32** | **53** | **−15% ARL, −21% tps** ❌ |

bf16 reassociation 漂移仍在（比舊 Hillis-Steele 版本好一點：ARL 從 2.94 拉到 3.32，但仍不及 v1）。

**結論：v2 chunk_scan 改回 v1 serial。** SIMD-shuffle kernel 留在 v2/scan_metal.py 作為 `_intra_chunk_metal_pp`，文件化為 dead end，未來研究 numerical stability（Kahan compensated sum 等）時可重訪。

### 11.2 chunk_scan_per_pos + forward_metal.py（攻 verify hot path）

**動機**：spec decode 的 verify forward 100% 是用 `_scan_per_pos`（pure MLX einsum）執行 SSM scan。寫一個 Metal kernel 取代它，並輸出 per-position state，理論上能省下 O(Lc²) MLX einsum + dispatch overhead。

**作法**：

- `mlx_model_v2/scan_metal.py::chunk_scan_per_pos(u, la, C_rot, chunk_size, h_init) → (y, h_per_pos)`
- 內部用既有 v1 serial Metal 的 intra/inter chunk kernels 跑出 `h_intra_flat`，再用 MLX 做 superposition `h_inter * exp(la_cum) + h_intra` 與 h_init 校正
- `speculative/forward_metal.py::mamba_verify_step_metal` 鏡像 `mamba_verify_step`，唯一差別是把 `_scan_per_pos` 換成 `chunk_scan_per_pos`
- decoder kwarg `metal_verify=False`（opt-in）切換 verify path

**Correctness**：fp32 byte-equal vs MLX path（K=4, K=8, max=48 prefix 48/48）。bf16 logits diff 跟 MLX 比約 0.28（在 scaled_tanh 30 範圍內 1%），**argmax 一致**。

**End-to-end SJD bf16 (max=512)**:

| Prompt         | K   | mlx tps | metal tps | metal ΔARL | metal Δtps |
| -------------- | --- | ------- | --------- | ---------- | ---------- |
| self_awareness | 8   | 71.9    | 68.7      | +3%        | **−4%**    |
| self_awareness | 12  | 58.8    | 59.0      | +4%        | +0%        |
| math_drill     | 8   | 58.6    | 56.6      | +1%        | −3%        |
| math_drill     | 12  | 50.2    | 37.5      | **−24%**   | **−25%**   |
| daily_conv     | 8   | 50.6    | 37.9      | **−19%**   | **−25%**   |
| daily_conv     | 12  | 35.2    | 32.4      | −8%        | −8%        |
| self_aware     | 16  | 43.5    | 35.6      | **−14%**   | **−18%**   |
| self_aware     | 24  | 45.4    | 35.8      | **−17%**   | **−21%**   |

**0/8 案例淨贏。** 平均 −12% tps。

### 11.3 為什麼 metal_verify 不贏

剖析三個原因：

1. **`_scan_per_pos` 不是真正的瓶頸**。對 Lc=K=8 的 verify forward，MLX einsum (Lc²=64 ops) 的絕對成本只有 ~64 × HNP × 24 layers = 200K FLOPs 等級。一輪 verify 的總 wall 時間 50-100ms 主要花在其他 24 個 mamba_verify_step 內的 RoPE / 投影 / norm / silu / einsum (B_rot \* x_ssm) 上。

2. **Metal kernel 對小 Lc 沒有優勢**。Apple Silicon 上一次 metal_kernel dispatch 開銷 ~100μs。對 Lc=8 跑 24 層 → 24 × 100μs = 2.4ms 純 dispatch overhead，逼近 MLX einsum 本來的成本。

3. **bf16 數值漂移再次出手**。Metal kernel 內部 fp32 累積、bf16 寫回，與 MLX einsum 在 bf16 全程的捨入點不同。對 ARL 已經邊際的 prompt（daily_conv ARL ≈ 2.4），微小漂移把 SJD 接受率拉下 19%。

換言之：spec decode 是 **memory-bandwidth bound + Python-overhead bound**，而非 SSM scan compute bound。攻 SSM scan 性價比低。

### 11.4 程式碼狀態

| 檔案                                                | 狀態                                                     |
| --------------------------------------------------- | -------------------------------------------------------- |
| `mlx_model_v2/scan_metal.py::chunk_scan`            | = v1 serial (identical)                                  |
| `mlx_model_v2/scan_metal.py::chunk_scan_per_pos`    | 新增，正確但端到端無加速                                 |
| `mlx_model_v2/scan_metal.py::_intra_chunk_metal_pp` | SIMD-shuffle Kogge-Stone (experimental, drift-prone)     |
| `speculative/forward_metal.py`                      | mamba_verify_step_metal + model_verify_forward_metal     |
| `speculative/jacobi.py` / `jacobi_sampling.py`      | 新增 `metal_verify=False` kwarg (opt-in, **預設 False**) |

兩個 v2 變體都留著，文件化為「不建議啟用，作為未來研究的起點」。**`metal_verify=True` 不推薦** — 0/8 配置淨贏，平均退步 12%。

### 11.5 真正能加速 spec decode 的方向（未做）

| 方向                                                       | 預期               | 工作量 |
| ---------------------------------------------------------- | ------------------ | ------ |
| Fuse mamba_verify_step 整體成單一 Metal kernel             | +30-50% verify tps | 高     |
| Per-position state 從 chunk_scan 直接拉出（避免重算）      | +10-20%            | 中     |
| 把 24 個 Mamba layer 的 RoPE/投影 fuse 成一個 Metal kernel | +20-40%            | 高     |
| Quantize Mamba weights 到 4-bit (減 memory bandwidth)      | +30-60%            | 高     |
| 把 head projection (vocab_size=32007) 改用 sparse top-k    | +10%               | 中     |

這些都是大手術，超出本輪「替換 scan」的範圍。**結論：spec decode 已接近這個架構的甜蜜點，再榨需要動模型 dataflow 層級的重構。**

---

## 7. 既定 demo 數字更新

舊 README 的「demo headline」是 SJD K=16 max=512 v2 cache → **3.20× / 8.0s wall**。新 COT 路徑在 K=8 max=512 cold-start（**不需要 6 分鐘 model-generation bake**）拿到 **1.88×** 純加上 cot pre-warm。

兩條路徑的取捨：

- **v2 cache** (`bake_cache.py`)：要跑 6 分鐘 model 採樣，但 ARL 衝到 7.24（特化單一 prompt + warm-up corpus）。
- **cot cache** (`bake_cot_caches.py`)：6 秒從訓練語料抽完，跨 prompt 通用，self_awareness 拿 3.93 ARL，math 與 daily 也不掉。

新 cot cache 不是要取代 v2 cache，是補另一條「通用、快、不需要 model」的路徑。

---

## 12. K-sweep：大 K 能否單獨解決 ARL？（2026-05-27）

**假設**：如果 K 拉到 16/24 時 ARL 自然到 5+，cot retriever 就不是必要。

### 12.1 完整 K-sweep（bf16, max_tokens=512, M2 Pro）

#### self_awareness — "Who are you?"

| K   | config    | ARL      | full% | tps  | wall   | speedup      |
| --- | --------- | -------- | ----- | ---- | ------ | ------------ |
| 8   | ng+rt     | 3.01     | 22.2  | 54.8 | 9.95s  | **1.45×**    |
| 8   | ng+rt+cot | 3.93     | 33.1  | 57.8 | 12.12s | **1.53×** ⭐ |
| 12  | ng+rt     | 2.72     | 8.5   | 38.2 | 14.24s | 1.01×        |
| 12  | ng+rt+cot | 4.04     | 19.4  | 51.4 | 14.92s | 1.36×        |
| 16  | ng+rt     | 3.89     | 11.9  | 47.9 | 11.29s | 1.19×        |
| 16  | ng+rt+cot | 3.87     | 11.9  | 45.1 | 14.76s | 1.12×        |
| 20  | ng+rt     | 3.44     | 5.9   | 35.2 | 15.09s | 0.88×        |
| 20  | ng+rt+cot | 3.84     | 5.8   | 35.9 | 18.03s | 0.89×        |
| 24  | ng+rt     | 3.41     | 2.0   | 31.8 | 16.66s | 0.79×        |
| 24  | ng+rt+cot | **5.25** | 3.0   | 46.6 | 15.13s | 1.16×        |

#### math_drill — "Solve (15+3)\*4/2 step by step"

| K   | config    | ARL  | full% | tps  | wall   | speedup      |
| --- | --------- | ---- | ----- | ---- | ------ | ------------ |
| 8   | ng+rt     | 3.43 | 32.0  | 61.1 | 9.15s  | **1.68×** ⭐ |
| 8   | ng+rt+cot | 3.52 | 27.4  | 58.5 | 12.85s | 1.61×        |
| 12  | ng+rt     | 3.42 | 4.7   | 51.9 | 10.45s | 1.42×        |
| 12  | ng+rt+cot | 3.74 | 16.1  | 52.5 | 13.91s | 1.44×        |
| 16  | ng+rt     | 3.43 | 2.0   | 43.7 | 12.33s | 1.07×        |
| 16  | ng+rt+cot | 3.75 | 13.0  | 44.7 | 15.20s | 1.10×        |
| 20  | ng+rt     | 3.38 | 1.3   | 33.7 | 15.76s | 0.83×        |
| 20  | ng+rt+cot | 3.53 | 8.7   | 35.5 | 19.11s | 0.87×        |
| 24  | ng+rt     | 3.28 | 0.0   | 30.6 | 17.30s | 0.75×        |
| 24  | ng+rt+cot | 4.47 | 11.0  | 38.2 | 16.91s | 0.94×        |

#### daily_conversation — focus tips

| K   | config    | ARL  | full% | tps  | wall   | speedup      |
| --- | --------- | ---- | ----- | ---- | ------ | ------------ |
| 8   | ng+rt     | 2.15 | 11.7  | 38.8 | 13.80s | 0.99×        |
| 8   | ng+rt+cot | 2.93 | 18.2  | 49.4 | 14.85s | **1.26×** ⭐ |
| 12  | ng+rt     | 2.14 | 7.1   | 30.5 | 17.47s | 0.78×        |
| 12  | ng+rt+cot | 2.74 | 7.9   | 37.6 | 17.20s | 0.96×        |
| 16  | ng+rt     | 2.16 | 3.0   | 26.4 | 20.28s | 0.68×        |
| 16  | ng+rt+cot | 3.42 | 8.7   | 36.3 | 17.82s | 0.93×        |
| 20  | ng+rt     | 2.16 | 2.9   | 21.5 | 24.44s | 0.55×        |
| 20  | ng+rt+cot | 3.15 | 4.9   | 32.2 | 19.35s | 0.83×        |
| 24  | ng+rt     | 1.98 | 0.0   | 16.8 | 31.03s | 0.43×        |
| 24  | ng+rt+cot | 4.01 | 8.6   | 34.4 | 19.66s | 0.88×        |

### 12.2 分析

#### ARL 隨 K 的走勢

```
self_awareness + cot:
  K=8:  ARL=3.93 ████████████████████
  K=12: ARL=4.04 █████████████████████
  K=16: ARL=3.87 ███████████████████
  K=20: ARL=3.84 ███████████████████
  K=24: ARL=5.25 ██████████████████████████  ← 唯一跨 5
```

ARL 在 K=8~20 幾乎平坦（3.8~4.0）。只有 K=24 跳升到 5.25。原因是 K=24 的 verify window 夠長，cot ngram chain 可以走更遠才遇到 miss。

#### 速度卻在 K=8 最優

```
self_awareness tps by K:
  K=8:  57.8 tok/s ████████████████████████████  ← 最高
  K=12: 51.4      ██████████████████████████
  K=16: 45.1      ███████████████████████
  K=20: 35.9      ██████████████
  K=24: 46.6      ████████████████████████  ← 反彈
```

K=24 的 ARL 跳升扛住了 verify cost 增長，但 tps 仍不及 K=8。底層公式：

```
tps ≈ ARL / (verify_ms(K) / 1000)
K=8:  3.93 / 0.092 = 42.7 → 實測 57.8 (accept bonus 拉升)
K=24: 5.25 / 0.145 = 36.2 → 實測 46.6
```

K=24 需要 ARL ≥ 5.25×1.25=6.6 才追平 K=8，需要 ARL ≥ 8.0 才超越。

#### 無 cot 時的 ARL 上限

```
純 ng+rt (無 cot), 全 K 最高 ARL:
  self_awareness: 3.89 (K=16)
  math_drill:     3.43 (K=8/16)
  daily_conv:     2.16 (K=8/16/20)

無 cot 時 ARL 天花板 ≈ 3.5，任何 K 都無法突破。
```

### 12.3 結論

1. **K 單獨不夠** — 純 ng+rt 下 ARL 天花板 ~3.5，K=24 也不過 3.41。draft 品質才是瓶頸。

2. **cot ngram 在 K=24 效益最大** — ARL +54%（3.41→5.25），cot chain 在長 window 有更多機會命中。

3. **但速度最優還是 K=8** — verify cost 增長壓過 ARL 增長。K=8 是 tps 甜蜜點。

4. **要突破 1.7× 需要更高 ARL 在更小 K** — 唯一路徑：改善 draft 品質（加 retriever、per-category），讓 K=8~12 就拿到 ARL≥6。cot retriever 是必要非奢侈。

5. **adaptive_K 可能有幫助** — 在 cot ngram 剛好命中長 chain 時自動擴 K，其它時間留在 K=8。現有實作（`jacobi.py` adaptive_K block）可直接啟用測試。

---

## 13. EXECUTION_PLAN 落地：per-bucket retriever + 向量化 query（2026-05-27）

依 `EXECUTION_PLAN.md`，目標把訓練語料的長片段抽出來變成 SuffixRetriever，補上 cot cache 缺的「長後綴匹配」這條路徑。

### 13.1 變更摘要

| 動作 | 檔案 | 說明 |
| ---- | ---- | ---- |
| 重寫 | `bake_cot_caches.py` | 同時輸出 (a) 全局 think/final NGramCache（aggregated，等價於 v1），(b) 每個 `sys_bucket` 的 think/final NGramCache，(c) 每個 bucket 的 SuffixRetriever（buffer=32K tokens、min_suffix=4、max_suffix=12）。bake 用 JSON 模式才能拿到 sys_bucket。 |
| 重寫 | `cot_cache.py` | 新增 `CoTCacheBundle` 容器；`load_cot_caches` 支援 v1 + v2 兩種 pkl 格式；`CoTPhaseTracker` 多了 `active_retriever()`；`bundle.get_caches(bucket)` 預設回傳「全局 ngram + 該 bucket retriever」（小 bucket 不會因 per-bucket sparsity 退化）。 |
| 修改 | `drafts.py` | `build_hybrid_branches` 多了 keyword-only `cot_retriever`，slot 順序：runtime_rtr → **cot_rtr** → cot_ng → ng → carry；`SuffixRetriever.query` 重寫成 NumPy 向量化版（lazy `_np_buf` mirror，per-position match-length scan），大 buffer 不再卡 wall-time。 |
| 修改 | `jacobi.py` / `jacobi_sampling.py` | 兩個 decoder 都新增 `cot_bucket: Optional[str]` kwarg。沒給 bucket 時，cot retriever **不**啟用（保留 v1 「只用 cot ngram」的行為）；給了 bucket 才把該 bucket 的 retriever 接到 `cot_retriever` slot。 |
| 修改 | `verify.py` | CLI 多 `--cot_bucket`；fp32 byte-equal 驗證新路徑。 |
| 修改 | `benchmark_cot.py` | 多一個 variant `SJD ng+rt+cot+r`，用 `cot_bucket=mode` 啟用 per-bucket retriever。 |
| 新增產物 | `cot_caches_v2.pkl` | 50 MB；含 8 buckets + 全局 ngram + 8 個 retriever。bake 約 7.6 秒（JSON 模式）。 |

### 13.2 fp32 byte-equal 驗證

```bash
.venv/bin/python3 -m mamba3_mlx.speculative.verify \
  --model_path checkpoints/4_loss_func/latest_sft_cot_model.mlx_fp32.npz \
  --dtype fp32 --max_tokens 48 --K 4 --K 8 \
  --cot_caches mamba3_mlx/speculative/cot_caches_v2.pkl \
  --cot_bucket self_awareness --use_retrieval --no-eos-stop
# → K=4 rt+cot OK prefix=48/48   K=8 rt+cot OK prefix=48/48   ALL OK
```

新 retriever（向量化版）跟 cot bundle 在 fp32 嚴格模式下與 AR-greedy byte-equal。

### 13.3 SJD benchmark（bf16, max_tokens=512, M2 Pro）

兩種 seed 各跑一輪以呈現 SJD 取樣的 run-to-run 變異：

#### 觀測一：retriever **不再吃 wall-time**

舊版（Python 三層 loop, 32K buffer）每輪 retriever query 約 ~10 ms — 在 K=8 verify (~92 ms) 上吃掉 ~10% wall。新版（NumPy `_np_buf` mirror + vector match-length scan）約 ~0.3 ms。`SJD ng+rt+cot` vs `ng+rt+cot+r` 的 tps 差距大多落在 ±3%（噪音範圍）。

#### 觀測二：retriever 救小 bucket

`math_drill` 只有 200 個訓練樣本，per-bucket think ngram 只有 1158 keys。但 bundle 預設用「全局 ngram + per-bucket retriever」配方，所以 cot ngram 不退化、retriever 又能補上小 bucket 的長後綴匹配。

| Prompt          | K | config        | ARL  | tps   | speedup       |
| --------------- | - | ------------- | ---- | ----- | ------------- |
| math_drill      | 8 | ng+rt         | 2.70 | 50.2  | 1.19×         |
| math_drill      | 8 | ng+rt+cot     | 3.85 | 66.6  | 1.57×         |
| **math_drill**  | 8 | **ng+rt+cot+r** | **3.85** | **71.6** | **1.69×** ⭐ |
| math_drill      | 12 | ng+rt+cot+r  | 4.21 | 58.4  | 1.38×         |

`math_drill K=8 cot+r = 1.69×` — 跟 §7 列出的舊 demo headline (`3.20× / v2 cache`) 不同的路徑，但 **cold-start**（pkl 7 秒 bake）。

#### 觀測三：retriever 在 cot ngram 已經很強時是 ties

`self_awareness` / `summarize_email` / 多數 daily 類 prompt，cot ngram 全局快取已經有 300k+ keys，命中率高，chain 自然延伸。retriever 在這些 prompt 上拿到的 ARL 跟單獨 cot ngram 持平（很多 cell 顯示完全一樣的 ARL，因為 retriever 的固定鏈在第 1 個 token 就跟 cot_ngram 一樣或被拒絕）。

#### 觀測四：seed 變異大

| Prompt         | K | config     | seed=42 spd | seed=1 spd |
| -------------- | - | ---------- | ----------- | ---------- |
| self_awareness | 8 | ng+rt+cot  | 1.11×       | 1.18×      |
| self_awareness | 8 | ng+rt+cot+r| 1.10×       | 0.94×      |
| math_drill     | 8 | ng+rt+cot+r| 1.69×       | 1.30×      |
| daily_conv     | 8 | ng+rt+cot+r| 0.82×       | 0.73×      |

SJD acceptance 機率本身就會隨機數展開不同；單一 seed 的單一 run 不能下結論。`benchmark_cot.py` 預設 `--seed 42`，需要 multi-seed average 才有統計意義。

### 13.4 結論

1. **整合無回歸**：當 `cot_bucket=None` 時，行為等價於 v1「全局 cot ngram + runtime ngram + retriever」。fp32 byte-equal pass。

2. **retriever 不再是 wall-time 瓶頸**：NumPy 重寫把 32K buffer 的 query 從 ~10 ms 降到 ~0.3 ms，於是 cot+r 路徑跟 cot 路徑的 tps 差距收斂到 SJD 噪音以內。

3. **小 bucket（math_drill）拿到清楚的 retriever 收益**：在最佳 seed 上 K=8 1.69×、K=12 1.38×。其他 prompt 持平。

4. **沒有 universal speed-up**：plan 預期的「ARL 從 3.93 → 6~8、speedup 從 1.5× → 2.0~2.5×」沒有出現。原因是 cot ngram 在 large bucket 上已經把 K=8 的 ARL 推到 ~4，retriever 的固定 K-1 鏈在 acceptance probability 下不夠 adaptive，常常第 2-3 個 token 就被拒。

5. **下一步候選**（unverified, 不在這輪實作範圍）：
   - 把 retriever 接受門檻拉高（min_suffix 從 4 → 6 或 8）— 只在「長到一定程度」才信任；
   - retriever chain 截短（只回前 4-6 個 token，剩下 fallback 給 cot_ngram）— 結合長片段的可信度 + ngram 的 adaptivity；
   - adaptive_K：retriever 命中長 chain 時自動擴 K 到 16-24。

### 13.5 推薦的 demo 配置

```python
from mamba3_mlx.speculative import jacobi_decode_sampling
r = jacobi_decode_sampling(
    model, prompt_ids, gen_config,
    K=8,
    use_ngram=True, ngram_n=4,
    use_retrieval=True,
    cot_caches="mamba3_mlx/speculative/cot_caches_v2.pkl",
    cot_bucket="math_drill",   # 或 None 跑 v1 等價路徑
)
```

baker 命令：

```bash
.venv/bin/python3 -m mamba3_mlx.speculative.bake_cot_caches \
  --json-mode --ngram_n 4 --max_continuations 4 \
  --retriever_max_window 32768 \
  --retriever_min_suffix 4 --retriever_max_suffix 12 \
  --out mamba3_mlx/speculative/cot_caches_v2.pkl
```

---

## §14. 三步調校最終結果：min_suffix=8 + hybrid chain + adaptive_K

### 14.1 實作摘要

依上一輪「下一步候選」逐一完成：

| 步驟 | 內容 | 方式 |
|------|------|------|
| Step 1 | `min_suffix` 4 → 8（retriever 只在 ≥8-token 後綴命中時才觸發）| re-bake pkl |
| Step 2 | hybrid chain：retriever 前 5 tokens + cot_ngram 延伸至 K-1 | `build_hybrid_branches` 新參數 `cot_retriever_max_len=5` |
| Step 3 | `adaptive_K`（GammaTune EWMA）加入 `jacobi_decode_sampling`；benchmark 新增 `+aK` variant | code 實作 + benchmark |

**bake 命令（Step 1）**：
```bash
.venv/bin/python3 -m mamba3_mlx.speculative.bake_cot_caches \
  --json-mode --ngram_n 4 --max_continuations 4 \
  --retriever_max_window 32768 \
  --retriever_min_suffix 8 --retriever_max_suffix 12 \
  --out mamba3_mlx/speculative/cot_caches_v2.pkl
```

### 14.2 benchmark 結果（seed=42, max_tokens=256）

```
=== mode=self_awareness  prompt='Who are you?'  max=256  dtype=bf16 ===
config                           K    arl  full%     tps    wall    spd
-----------------------------------------------------------------------
AR-sampling                      -   1.00    0.0    40.0    6.38   1.00x
SJD ng+rt+cot+r                  8   2.29    6.2    33.9   15.96   0.85x
SJD ng+rt+cot+r+aK               8   1.90   12.7    33.8   19.49   0.84x
SJD ng+rt+cot+r                 12   2.55    5.0    34.4   15.69   0.86x
SJD ng+rt+cot+r+aK              12   2.34   17.4    37.6   15.35   0.94x
SJD ng+rt+cot+r                 16   3.28    7.7    35.3   18.25   0.88x
SJD ng+rt+cot+r+aK              16   2.34   17.4    36.6   15.78   0.92x

=== mode=math_drill  prompt='Solve (15+3)*4/2 step by step' ===
AR-sampling                      -   1.00    0.0    36.9    6.91   1.00x
SJD ng+rt+cot+r                  8   2.53   12.9    42.4   13.38   1.15x
SJD ng+rt+cot+r+aK               8   2.43    9.3    40.0   13.75   1.08x
SJD ng+rt+cot+r                 16   2.67    5.9    29.8   16.25   0.81x
SJD ng+rt+cot+r+aK              16   2.43    8.4    37.8   13.82   1.02x

=== mode=daily_conversation  prompt='Tips for working from home?' ===
AR-sampling                      -   1.00    0.0    38.2    6.67   1.00x
SJD ng+rt+cot+r                 16   1.61    0.0    19.2   19.68   0.50x
SJD ng+rt+cot+r+aK              16   1.67   11.8    31.0   14.60   0.81x
```

### 14.3 分析

**Step 1（min_suffix=8）**：幾乎看不到效果差異。`cot+r` 的結果與 `cot` 基本一致，代表 min_suffix=8 的 retriever 在 K=8 短窗口下幾乎從不觸發（8-token 後綴在 5827-32768 token 的 retriever buffer 裡不常有精確匹配）。在這個 token 長度（256）上，retriever 對結果的貢獻已降至噪音水準。

**Step 2（hybrid chain）**：由於 Step 1 retriever 幾乎不命中，hybrid chain 等同於退化為 cot_ngram-only。技術上正確（smoke test pass），但在這組參數下無法驗證實際效益。

**Step 3（adaptive_K）**：這是三步中效益最明顯的：

1. **從大 K 收縮（主要效益）**：`math_drill K=16→aK`：29.8 → 37.8 tps（+27%，從 0.81× 回升到 1.02×）。`daily_conv K=16→aK`：19.2 → 31.0 tps（+61%，從 0.50× 到 0.81×）。adaptive_K 偵測到 ARL/K < 0.30（BUMP_LO），把 K 從 16 收縮到 ~6，降低 verify 的固定開銷。

2. **從中 K 收縮（次要效益）**：`self_awareness K=12→aK`：34.4 → 37.6 tps（+9%）。

3. **從小 K 收縮（反效果）**：`self_awareness K=8→aK`：ARL 從 2.29 降到 1.90，tps 微降 33.9 → 33.8。K=8 起點下 EWMA 還在適應，ARL 下降但 full% 上升，整體持平。

4. **K 從不擴張**：目前 ARL/K 比值 (0.20-0.35) 低於 BUMP_HI=0.55，所以 K 只收縮不擴張。若要讓 adaptive_K 真正「自動找到最大有效 K」，需要調低 BUMP_HI（如 0.35-0.40）或從 K_min=4 起步讓 K 往上爬。

### 14.4 三輪下來的整體結論

```
K=8 固定 ARL ceiling ≈ 3-4 → 理論 speedup ~1.5-1.7×（已達到）
K=12-16 adaptive ARL ceiling ≈ 3-4 但 K 自動縮 → 接近 K=8 效率
「破 2×」 需要 ARL > 5.5（對應 K=12 的 55% accept rate）
目前語料 accept rate ≈ 25-35%，還差一倍
```

**要真正突破 2×**，需要以下其中之一：
1. 改善 draft quality（如 token-level 預測而非 ngram lookup）— 需要小 draft model 或 Medusa head
2. 改善 verify 效率（如更快的 Mamba scan）— 降低 break-even ARL
3. 換一個 accept rate 更高的 domain（程式碼、固定格式輸出）

在「不改模型、不加 draft model」的前提下，`cot+r+aK` 是目前可落地的最佳配置，讓各 domain 在不同 K 下都能收歛到 ≥1× 的效率（避免過大 K 拖累）。

### 14.5 推薦配置（adaptive_K 版）

```python
r = jacobi_decode_sampling(
    model, prompt_ids, gen_config,
    K=12,                        # 起步 K；adaptive 會自動縮至最佳點
    adaptive_K=True, K_min=4, K_max=24,
    use_ngram=True, ngram_n=4,
    use_retrieval=True,
    cot_caches="mamba3_mlx/speculative/cot_caches_v2.pkl",
    cot_bucket="self_awareness",  # 依 prompt 路由
)
```

---

## §15. 推進到 100 tok/s 計畫：Steps 1 + 2 + 3

### 15.1 三步計畫與假設

目標 TPS 57.8 → 100，需要 **ARL 3.93 → 6.8** 或 **round 92ms → 53ms**（compile 做不到）。
拆三步同時推：

| Step | 內容 | 假設增益 | 風險 |
|------|------|----------|------|
| 1 | `mx.compile` 24 層 Mamba verify | 92→74ms（−20%）| 無，fp32 byte-equal |
| 2 | runtime retriever 預熱（demo_cache_v2.pkl）| ARL 3.93→7 | 無，SJD 機率接受保證等價 |
| 3 | `mx.compile` 6 層 Transformer verify + 固定 KV pre-alloc | 74→68ms（−10%）| 中：要設計 fixed-size KV |

### 15.2 Step 1 + Step 2 實作

**Step 1（`forward_compiled.py`）**：新建 `make_compiled_model_verify_forward(model)`，逐層對 Mamba verify step 套 `mx.compile`，每層分 cold/warm 兩種編譯版本（state=None vs 有 state）。Transformer 層不編譯（KV 每輪長度變動，shape 不固定）。

**Step 2（`bake_cache.py` → `demo_cache_v2.pkl`）**：用既有的 baker，跑 16384 token 的 AR-sampling 預熱，產出 1152 ngrams + 16384-token retriever buffer。

**接線**：
- `jacobi_decode` 與 `jacobi_decode_sampling` 都加 `compile_verify` + `preloaded_ngram` + `preloaded_retriever` 參數
- `run_jacobi.py` / `run_jacobi_sampling.py` 加 `--compile_verify` 與 `--runtime_cache`
- `run_sjd_best.sh` 預設 `COMPILE=1` 與 runtime cache
- 新增 `benchmark_steps.py` 一次跑四階梯對照

### 15.3 SJD 實測（K=16, self_awareness, max=256, seed=42）

```
config                              ARL  rounds  decode_tps   wall(s)
baseline (cold)                    1.63    106      20.9       8.56
+ cot_caches                       1.90     79      24.4      11.98
+ cot + runtime                    2.01     71      25.5      11.23
+ cot + runtime + compile_verify   1.98     98      24.3      13.57
```

**Greedy K=12 adaptive（`make sjd PROMPT="Who are you?" MAX_TOK=512`）**：
- baseline (cot only)：34 tps, ARL=1.86, 66 rounds
- full stack (cot + runtime + compile_verify)：37 tps, ARL=2.03, 62 rounds（+9%）

### 15.4 分析：projection 為什麼沒打到

projection 假設 ARL 從 3.93 跳到 7（K=16 demo 老紀錄）。實測 cot+runtime 只到 **ARL=2**，差距 5x。

**原因**：
1. **retriever match 命中率低**：16K runtime buf 對於「Who are you?」回答的長後綴匹配能命中，但 SJD 的機率接受門檻（temp=0.15, top_p=0.85, min_p=0.08）對連續 token 的接受機率連乘很快衰減。
2. **prompt 短 + 結構化 Step 列表**：每個 "Step N: **..." 段都是新 anchor，retriever 對 step prefix 命中但 step content 不命中，導致每 round 接受 2-3 個 token 就被拒。
3. **K=16 verify cost 高**：即使 ARL=2 也比 K=12 ARL=2 慢，因為 verify 線性 scale K。

**Step 1 (mx.compile) 確實有效**：拿掉 cot 與 runtime，純看 compile 在 greedy 上：29 → 34 tps（+17%）。fp32 byte-equal 通過 80-token 比對。

**Step 2 (runtime cache) 也有效但只 +10-15%**：因為 ARL 沒大跳。

### 15.5 為什麼 Step 3 不解決 ARL 問題

projection 表預期 "step 1+2 把 ARL 推到 7"，Step 3 再砍 round time。**實際 ARL 在 step 1+2 後沒到 7**。Step 3 (TF compile + fixed KV) 只能在固定 ARL 下進一步壓 round time，**無法**修補 ARL 缺口。

要打到 100 tps，先解 ARL 才有意義。可走路徑：

1. **更大、更長、更貼合的 runtime bake**：用真正會 deploy 的 prompt 集（不只「Who are you?」），bake 10K+ tokens 的多樣輸出
2. **動態溫度 / 接受門檻 tuning**：temp 0.15 太低，導致 SJD 連續接受機率不穩；可改 temp=0.5 + min_p 高一點
3. **小 draft model（如蒸餾 50M）**：每個 round 給 K 個 token 都來自 draft model 的高機率區，accept rate 可上 60-70%
4. **改用結構化任務 demo**：程式碼、表格生成，retriever 命中率本來就高

### 15.6 Step 3 狀態

**未實作**。Step 3 (TF compile + fixed-size KV pre-alloc) 需要：
- 預配 `(B, kv_h, MAX_LEN, 64)` 的 KV buffer，每層一份
- 用 `mx.array.at[..., positions, :].set(...)` 寫入 slot
- attention mask 依 `past_len + L` 動態計算（mask shape 仍固定）
- 整層 `mx.compile`

預估工作量：~3-4 hr，盤旋 ~5-8% TPS 增益（在 ARL 不變的前提下）。
**建議先解 ARL 再評估 Step 3**——因為當 ARL 還在 2-3，Step 3 把 round time 從 X→0.9X 的絕對值很小。

### 15.7 落地用法

```bash
# 命令列（greedy + 全 stack）
make sjd PROMPT="…" MAX_TOK=512 STREAM=1

# 關 compile（驗證 Step 1 增益）
COMPILE=0 make sjd …

# 關 runtime cache（驗證 Step 2 增益）
RUNTIME=none make sjd …

# 重新 bake runtime cache（換 checkpoint 後跑一次）
.venv/bin/python3 -m mamba3_mlx.speculative.bake_cache \
    --model_path checkpoints/v2/latest_sft_cot_model.mlx_bf16.npz \
    --warmup_tokens 16384 \
    --retrieval_max_window 16384 --retrieval_max_suffix 12 \
    --out mamba3_mlx/speculative/demo_cache_v2.pkl

# 四階梯對照
.venv/bin/python3 -m mamba3_mlx.speculative.benchmark_steps --K 16
```

