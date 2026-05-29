# Execution Plan: Speculative Decoding 加速 — 資料驅動草稿品質提升

**目標**：將 SJD speedup 從 1.5× 提升至 2.0~2.5×，**不改模型**。

---

## 1. 問題陳述

### 1.1 現狀

- **最佳配置**：SJD K=8 + cot_caches_n4.pkl → self_awareness 1.53×, math_drill 1.68×
- **ARL 瓶頸**：最佳 ARL = 3.93。純 ngram（n=4）天花板 ~3.5，任何 K 都無法突破
- **CoT cache 缺口**：`cot_caches_n4.pkl` 只有 think/final NGramCache，**沒有 SuffixRetriever**

### 1.2 為什麼不改模型的所有嘗試都死了

| 嘗試 | 結果 | 根因 |
|------|------|------|
| K=24 純 ng+rt | ARL=3.41 | ngram 天花板，K 無關 |
| n=5 cot cache | −31% tps | 4-token key 命中率暴跌 |
| Hillis-Steele Metal scan | −25% tps | bf16 數值漂移 + scan 只佔 2% wall time |
| Kogge-Stone SIMD scan | −21% tps | 同上 |
| metal_verify forward | −12% tps | dispatch overhead 壓過 kernel 收益 |
| Lookahead Phase A | −60% tps | 雙 forward cost >> ARL gain |

### 1.3 唯一可行的方向

**提升 draft 品質**。每多 1 ARL ≈ +11 tok/s（K=8）。retriever 是 ARL 暴漲的唯一來源 — v2 demo 的 ARL=7.24 就是 retriever 抓到的長片段。

---

## 2. 技術原理

### 2.1 為什麼 retriever 是關鍵（PLD / REST 原理）

```
ngram (n=4):  看 3 個前導 token，逐 token MRU 猜測 → 天花板 ~3.5
retriever:    在整個 corpus buffer 做最長後綴匹配 → 回傳整段後續 token

範例:
  buffer 存過: "I am a language model trained by..."
  當前尾:       "I am a language model"
  retriever 找到匹配 → 回傳 [trained, by, ..., ..., ..., ...]
  整段 K-1 token 被 SJD 接受 → ARL 一次暴漲
```

**原理**（PLD / REST 架構，arXiv 2503.11238, 2411.03786）：
- 訓練語料是輸出的統計鏡像。模型生成的結構性輸出（intro, body, conclusion）在訓練集中反覆出現
- SuffixRetriever 建構滑動視窗 buffer，query 時搜最長後綴 → 回傳後續 token
- 搜尋用右向左掃描（recency bias），優先取最近出現的匹配
- 對高度結構化的輸出（think→step→conclusion 定型句式），命中率極高

### 2.2 為什麼現在沒效果

當前 `benchmark_cot.py` 的 `ng+rt+cot` 組態：
- `ng` = runtime ngram（從 prompt + generated 累積，冷啟動為空）
- `rt` = runtime retriever（同上）
- `cot` = CoT ngram（有數據但只有逐 token 猜測）

**retriever 是空的**，沒有預熱數據。v2 demo 的 ARL=7.24 來自 `demo_cache_v2.pkl`（6 分鐘 model 預熱的 retriever buffer）。

### 2.3 解法：從訓練集建 retriever

訓練資料有 10,217 筆完整 CoT JSON。每筆 `text` 欄位已是 `<think>...</think><final>...</final>` 的 token 序列。直接把這些塞進 SuffixRetriever buffer — **不需要跑模型**，純 tokenize 即可。

```
bake_cot_caches.py 現有:   tokenize → 切 think/final → 計 ngram 頻率
bake_cot_caches.py 新增:   tokenize → 整段塞進 per-category SuffixRetriever
```

### 2.4 為什麼 per-category（8 buckets）

訓練資料 `sys_bucket` 欄位有 8 類（summarize_email、math_drill、self_awareness...）。混在一起的問題：

- summarize_email (30%) 佔據 MRU 排序，擠出 math_drill (2%)
- 跑 math prompt 時 retriever buffer 裡幾乎都是 email 數據 → 長後綴永遠命中 email，不是 math

**Per-category**：跑 math prompt 只查 math bucket → 200 筆數學題在 buffer → 命中同類題目的完整解題過程。

### 2.5 Draft source decorrelation（多來源並存）

`build_hybrid_branches` 的 5 個 slot 同時工作：

```
slot 0: runtime retriever (demo_cache_v2.pkl)    — 模型輸出行為
slot 1: cot ngram (think/final, per-category)     — 訓練集語法結構
slot 2: cot retriever (per-category buffer)        — 訓練集長片段 / 同類題目
slot 3: runtime ngram (history MRU)                — 當前 prompt 歷史
slot 4: carry seed
```

關鍵：這五個來源的**失敗模式彼此獨立**。retriever 依賴長後綴重合，ngram 只需 3-token key。一個 miss ≠ 另一個也 miss。等效命中率 = 1 − Π(1−p_i)，遠高於單一來源。

---

## 3. 執行計畫

### Phase 1: COT cache 加 retriever（不改 decoder）

**檔案**：`bake_cot_caches.py`、`cot_cache.py`

#### 3.1 修改 `bake_cot_caches.py`

1. 對每個 `sys_bucket` 建立獨立的 `SuffixRetriever(max_window=???, max_suffix=16)`
2. 對該 bucket 的所有樣本，把 `text` tokenize 後的整個序列塞進 retriever
3. 輸出 pkl 加 `"retrievers": {bucket: retriever.to_state()}` 欄位
4. Think/final ngram 也改 per-category

#### 3.2 修改 `cot_cache.py`

1. `load_cot_caches()` 回傳新增的 `retriever_dict`
2. `CoTPhaseTracker` 不變 — phase tracking 邏輯同上，但加 `active_retriever()` 方法

#### 3.3 輸出格式

```python
cot_caches_v2.pkl:
{
    "think":    {bucket: NGramCache_state, ...},   # 既有的 per-category 版
    "final":    {bucket: NGramCache_state, ...},
    "retrievers": {bucket: SuffixRetriever_state, ...},  # 新增
    "markers":  {think_open, think_close, final_open, final_close},
    "bucket_map": {sys_prompt_prefix: bucket, ...},      # 新增：分類依據
    "default_bucket": "daily_conversation",
}
```

### Phase 2: 合併 runtime cache + COT cache（改 decoder）

**檔案**：`drafts.py`、`jacobi_sampling.py`、`run_sjd_demo.py`（或 benchmark_cot.py）

#### 3.4 修改 `build_hybrid_branches`

加 keyword-only `cot_retriever: Optional[SuffixRetriever] = None`，slot 2（cot ngram 之後）。

#### 3.5 修改 `jacobi_sampling.py`

1. 接受 `cot_caches` 時也載入 retriever_dict
2. 每輪用 `cot_tracker.active_retriever()` 取當前 phase 對應 category 的 retriever
3. 傳入 `build_hybrid_branches(cot_retriever=...)`

#### 3.6 修改 runner

```bash
# 同時載入兩種 cache
python -m mamba3_mlx.speculative.benchmark_cot \
  --cache mamba3_mlx/speculative/demo_cache_v2.pkl \     # runtime
  --cot_caches mamba3_mlx/speculative/cot_caches_v2.pkl   # training
```

### Phase 3: adaptive_K（啟用現有實作）

`jacobi.py:377-382` 已有 adaptive_K 實作（GammaTune EWMA）。啟用參數：

```python
adaptive_K=True, K=8, K_min=4, K_max=24
```

當 cache hit rate 高時自動擴 K → ARL 上限從 8 擴到 24。

---

## 4. 預期結果

### 4.1 Per-category retriever 命中率估算

| Category | Samples | 預期 retriever 命中率 | 預期 ARL |
|----------|---------|----------------------|----------|
| self_awareness | 1,357 | 高（intro 句型重複） | **6~8** |
| summarize_email | 3,052 | 高（固定格式） | **7~10** |
| math_drill | 200 | 中（題目變化大但解題結構固定） | **5~7** |
| daily_conversation | 2,280 | 中（自由文本） | **4~6** |
| emotion | 1,609 | 高（固定套路） | **6~8** |

### 4.2 預期 speedup（K=8, adaptive_K 關閉）

| Prompt | 目前 ARL | 目前 speedup | 預期 ARL | 預期 speedup |
|--------|----------|-------------|----------|-------------|
| self_awareness | 3.93 | 1.53× | 6~8 | **2.0~2.4×** |
| math_drill | 3.52 | 1.61× | 5~7 | **1.8~2.2×** |
| daily_conversation | 2.93 | 1.26× | 4~6 | **1.5~1.9×** |

### 4.3 預期 speedup（adaptive_K 啟用）

當 retriever 命中長片段 → K 自動擴到 16~24 → 單輪 ARL 可達 10~15。理論上限：

```
ARL=12, K=auto, verify=130ms, AR=25ms/tok
tps = 12/0.130 = 92 → speedup = 92/40 = 2.3×
```

---

## 5. 成本與風險

| | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| 程式碼改動 | ~80 行（baker） | ~40 行（drafts + decoder） | ~5 行（啟用參數） |
| 離線 bake | 6 秒（現有）→ ~30 秒（加 retriever） | 不需重 bake | 不需 |
| Pkl 大小 | 24MB → ~50MB（含 retriever buffers） | - | - |
| 風險 | 零（不改 decoder, 不改變既有輸出格式） | 低（slot 優先級不會破壞既有邏輯） | 零（現有實作已驗證） |
| 模型改動 | 無 | 無 | 無 |

---

## 6. 不要做的事（已證偽）

- ❌ Metal kernel 改寫（scan 只佔 2% wall time，bf16 漂移毀 ARL）
- ❌ n=5 ngram（命中率暴跌）
- ❌ K 單獨加大（verify cost 壓過 ARL 增長）
- ❌ Lookahead Phase A（雙 forward 永遠虧損）
- ❌ Phase 2 fused mask（24/30 層是 Mamba，無 attention mask 概念）
- ❌ `mx.compile` 全模型（KV cache 動態增長 → 頻繁 recompile，未評估）
