# Final-SW + PDL + InfoEntropy 實作計畫

> 不重複貼程式碼，只說改哪裡、改什麼、以及事前驗證步驟。

---

## 0. 前置：先讀這些檔案（依序）

| 順序 | 檔案 | 為什麼要讀 |
|---|---|---|
| 1 | `scripts/model.py:782-892` | forward() 的 loss 計算區段，FCP/SCALe/SFT-GO 都在這 |
| 2 | `scripts/model.py:1-50` | Mamba3LanguageModel.__init__，看 model attribute 的 pattern |
| 3 | `scripts/train_sft.py:1510-1575` | FCP/SCALe/SFT-GO config setup，把參數掛到 model 上的邏輯 |
| 4 | `scripts/train_sft.py:1640-1820` | 訓練迴圈中累積 metric、寫 CSV 的段落 |
| 5 | `scripts/train_sft.py:1664-1746` | SCALe scale_w 計算 + 傳入 model 的流程 |
| 6 | `scripts/train_sft.py:860-900` | train_sft() 函式簽名，決定新參數加在哪 |
| 7 | `scripts/train_sft.py:1580-1620` | val CSV 寫入段落 |
| 8 | `scripts/train_sft_cot.py` | 入口，設預設值 |

---

## 1. 事前資料驗證（先做，不改程式碼）

### 1.1 確認 token id 對應

用 conda env 跑：

```bash
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('dataset/tokenizer')
print('eos_token_id:', tok.eos_token_id)
print('<|im_end|> id:', tok.convert_tokens_to_ids('<|im_end|>'))
print('<|im_start|> id:', tok.convert_tokens_to_ids('<|im_start|>'))
print('<think> id:', tok.convert_tokens_to_ids('<think>'))
print('</think> id:', tok.convert_tokens_to_ids('</think>'))
print('<final> id:', tok.convert_tokens_to_ids('<final>'))
print('</final> id:', tok.convert_tokens_to_ids('</final>'))
"
```

預期輸出：
```
eos_token_id: 2
<|im_end|> id: 32001
<think> id: 32002
</think> id: 32003
<final> id: 32004
</final> id: 32005
```

### 1.2 確認 train 用哪個 tokenizer 路徑

讀 `scripts/train_sft.py:1524-1532`，確認 `_ce_tok` 從哪個路徑載入（應是 `TOKENIZER_DIR` 參數的 `dataset/tokenizer`）。

### 1.3 驗證 training data 中該 token 是否存在

```bash
conda run -n torch310 python3 -c "
import numpy as np
mm = np.memmap('dataset/stf_cot_train.bin', dtype=np.uint16, mode='r')
# 驗證特殊 token 存在性
for tid in [2, 32000, 32001, 32002, 32003, 32004, 32005]:
    c = (mm == tid).sum()
    print(f'id={tid}: {c}')
# 確認 final 區域存在（32004 後到 32005 前有東西）
"
```

預期：32001 大量出現（終止符），32002/32003/32004/32005 各約 40956 次。

### 1.4 統計 token 頻率（為 PDL 做準備）

```bash
conda run -n torch310 python3 -c "
import numpy as np
mm = np.memmap('dataset/stf_cot_train.bin', dtype=np.uint16, mode='r')
freq = np.bincount(mm, minlength=32007)
print(f'vocab seen tokens: {(freq>0).sum()}')
print(f'max freq: {freq.max()} (id={freq.argmax()})')
print(f'min freq among used: {freq[freq>0].min()}')
# 特殊 token 頻率
for tid in [2, 32000, 32001, 32002, 32003, 32004, 32005]:
    print(f'id={tid}: freq={freq[tid]}')
np.save('cot_task/reports/token_freq.npy', freq)
print('token_freq.npy saved')
"
```

---

## 2. 修改清單

### A. FCP token 修正（BUG FIX）

**問題**：FCP 監控 `tok.eos_token_id` (id=2)，但 ChatML 資料用 `<|im_end|>` (id=32001) 做終止。`</s>` 只出現 2 次/13.3M tokens。

**檔案**：`scripts/train_sft.py`

**改法**：在 FCP setup 段落（約 line 1550），把：
```python
_um_ce._fcp_eos_id = int(eos_id_run)
```
改成從 tokenizer 解析 `<|im_end|>` 的 id 並傳給 FCP。維持保留 `eos_id_run` 用於 inference stopping，僅 FCP 的 `_fcp_eos_id` 改用 `32001`。

**具體位置**：`train_sft.py:1516-1554` 區段。新增一個變數 `chatml_end_id_run`（從 `_ce_tok.convert_tokens_to_ids('<|im_end|>')` 取得），在 `SFT_USE_FCP` 段落把 `_um_ce._fcp_eos_id` 設成這個值而非 `eos_id_run`。

### B. model.py：forward() 新增 final_mask + 三層權重

**檔案**：`scripts/model.py`

**位置**：`Mamba3LanguageModel.forward()` (lines 782-892)

**修改點**：

1. **新增 model attributes**（在 `__init__` 裡加，約 line 1047 之後）：

```python
# Final Enhance flags & params (預設關閉，由 train_sft.py 啟動時設定)
self._enable_final_sw = False   # bool
self._final_sw_eta = 0.1        # float
self._final_sw_lambda = 0.8     # float
self._enable_pdl = False        # bool
self._pdl_weights = None        # torch.Tensor [vocab_size] 或 None
self._enable_ie = False         # bool
self._ie_gamma = 0.5            # float
self._ie_on_think = False       # bool
```

2. **forward() 內新增 final_mask 計算**（在 FCP 的 cumsum mask 之後，約 line 861 之前）：

```python
# final region mask (32004..32005)
f_fs = (input_ids == 32004).to(torch.int32)
f_fe = (input_ids == 32005).to(torch.int32)
f_diff = f_fs.cumsum(dim=1) - f_fe.cumsum(dim=1)
final_mask = (f_diff > 0).to(logits.dtype)
final_mask = final_mask * valid_mask  # 只用於有 label 的位置
```

3. **在 ce_weighted 計算段落（line 805-837）新增三層權重乘積**：

   不改原結構，在 `ce_weighted` 算出後、FCP 計算前（約 line 843 之前），插入：

```python
# ---- Final Enhance: Final-SW + PDL + IE ----
_any_enhance = self._enable_final_sw or self._enable_pdl or self._enable_ie
if _any_enhance:
    enhance_w = torch.ones_like(labels_flat, dtype=raw.dtype)

    # Final-SW: 前綴加權，僅 final 區
    if self._enable_final_sw:
        ... (用 final_mask + final_offset 算出 final_weights)
        enhance_w = enhance_w * final_weights.reshape(-1)

    # PDL: 靜態查表
    if self._enable_pdl and self._pdl_weights is not None:
        pdl_w = self._pdl_weights[labels_flat.clamp(0, vocab_size-1)]
        enhance_w = enhance_w * pdl_w.to(raw.device).to(raw.dtype)

    # IE: 動態 entropy 權重，僅 final 區
    if self._enable_ie and _probs is not None:
        ... (從 _probs 算 entropy → ie_w)
        enhance_w = enhance_w * ie_w

    # 乘回 ce_weighted（注意：不能直接改 ce_weighted 破壞 logging）
    ce_weighted_enhanced = (raw * enhance_w).sum() / n_sup
    # 保留原始 ce_weighted 用於 ce_loss 欄位 logging
```

   **關鍵**：`ce_weighted` 保留原始值寫入 CSV 的 `ce_loss` 欄；`ce_weighted_enhanced` 只用於 `loss` 計算。這樣曲線可比對新權重的效果。

   **注意**：這裡 `final_offset` 需要在 reshape 前用 cumsum 算出（每個 batch item 內 final 區域的位置偏移）。

4. **複用 FCP 的 softmax**：FCP 已做 `F.softmax(logits.float(), dim=-1)`（line 863），InfoEntropy 可以直接拿這個結果來算 entropy，省一次 softmax。把 softmax 結果存成變數，先算 IE，再從中取 eos_probs。

### C. train_sft.py：參數傳遞（無觸發、無 warmup、直接啟用）

> **設計決策**：三項新權重預設全部啟動，不做 PPL 觸發 / 漸進 warmup。loss spike 來自權重改變（非模型崩潰），屬一次性跳躍，訓練會自行收斂。

**檔案**：`scripts/train_sft.py`

**位置**：

1. **函式簽名** (line ~860)：新增參數（無 PPL 相關）
   - `ENABLE_FINAL_ENHANCE: bool = True` ← 預設開
   - `ENABLE_FINAL_SW: bool = True`       ← 個別開關
   - `ENABLE_PDL: bool = True`            ← 個別開關
   - `ENABLE_IE: bool = True`             ← 個別開關
   - `FINAL_SW_ETA: float = 0.1`
   - `FINAL_SW_LAMBDA: float = 0.8`
   - `PDL_ALPHA: float = 0.4`
   - `PDL_WEIGHTS_PATH: str = "cot_task/reports/pdl_weights.pt"`
   - `IE_GAMMA: float = 0.5`
   - `IE_ON_THINK: bool = False`

2. **參數 setup 段落**（約 line 1303 之後，接在 SCALe 設定之後）：
   - 解析環境變數
   - 載入 `pdl_weights.pt`（若不存在則 warn 並自動關閉 PDL）
   - **不需要** trigger / warmup / alpha 邏輯 ← 刪除原計劃此部分

3. **傳入 model**（約 line 1664 之後，在訓練迴圈外、一次性掛上去就可以了）：
   ```python
   if ENABLE_FINAL_ENHANCE:
       _um_ce._enable_final_sw = ENABLE_FINAL_SW
       _um_ce._final_sw_eta = FINAL_SW_ETA
       _um_ce._final_sw_lambda = FINAL_SW_LAMBDA
       _um_ce._enable_pdl = ENABLE_PDL and (_pdl_weights is not None)
       _um_ce._pdl_weights = _pdl_weights.to(device)
       _um_ce._enable_ie = ENABLE_IE
       _um_ce._ie_gamma = IE_GAMMA
       _um_ce._ie_on_think = IE_ON_THINK
   ```
   不需要每個 batch 更新（權重是靜態或動態計算的）。

4. **監控指標**：在訓練 step log（約 line 1770-1830）新增印出 `pdl_mean, ie_mean`

5. **CSV 表頭擴充**（約 line 1623-1637）：新增 2 欄（不需要 alpha）：
   ```
   pdl_weight_mean, ie_weight_mean
   ```

6. **Val 同擴**（約 line 556-650 + val CSV header 約 line 1583-1600）

### D. train_sft_cot.py：入口預設值（全部預設啟動）

**檔案**：`scripts/train_sft_cot.py`

在 `train_sft()` 呼叫（line 171-201）中加入新參數：

```python
ENABLE_FINAL_ENHANCE=True,   # 預設開
ENABLE_FINAL_SW=True,
ENABLE_PDL=True,
ENABLE_IE=True,
FINAL_SW_ETA=0.1,
FINAL_SW_LAMBDA=0.8,
PDL_ALPHA=0.4,
PDL_WEIGHTS_PATH="cot_task/reports/pdl_weights.pt",
IE_GAMMA=0.5,
IE_ON_THINK=False,
```

對應環境變數（用於手動關閉個別組件）：
- `SFT_COT_ENABLE_FINAL_ENHANCE` → 總開關
- `SFT_COT_ENABLE_FINAL_SW` → 單獨關 Final-SW
- `SFT_COT_ENABLE_PDL` → 單獨關 PDL
- `SFT_COT_ENABLE_IE` → 單獨關 IE
- `SFT_COT_FINAL_SW_ETA` / `SFT_COT_FINAL_SW_LAMBDA`
- `SFT_COT_PDL_ALPHA` / `SFT_COT_IE_GAMMA`

**無** PPL threshold / warmup 相關環境變數。

### E. 預計算腳本（獨立新檔）

**新檔案**：`scripts/precompute_pdl_weights.py`

邏輯：
1. 讀 `dataset/stf_cot_train.bin`（mmap）
2. `np.bincount` 算出 freq[32007]
3. 特殊 token (32000-32006) 的 freq 強制設為 0（讓權重=1.0，不干擾 SFT-GO）
4. `pdl = ((freq + 1.0) / freq.max()) ** (-alpha)`
5. 存成 `cot_task/reports/pdl_weights.pt`

---

## 2.5. 效能關鍵：softmax 複用 + 條件計算

新機制最大的計算瓶頸是 **InfoEntropy 需要對 32007-dim logits 做 softmax + entropy**。如果不處理好，訓練速度會掉 10-20%。

### 當前 FCP 的 softmax 狀況

`model.py:863`：
```python
eos_probs = F.softmax(logits.float(), dim=-1)[..., int(fcp_eos_id)]
```

這行已經 materialize 了一顆 `[B, T, 32007]` 的 float32 tensor（約 1.5 GB for effective batch），但只取其中一個 channel。InfoEntropy 需要整個分佈，所以**必須複用這顆 softmax 結果，不能做兩次**。

### 改法：softmax 提前到條件判斷外面

```python
# 提前做一次 softmax，不管 FCP/IE 是否啟用
_probs = None
_need_probs = (fcp_eos_id is not None and fcp_lambda > 0.0) or \
              self._enable_ie
if _need_probs:
    _probs = F.softmax(logits.float(), dim=-1)  # [B, T, V]

# FCP 從 _probs 取 eos channel
if fcp_eos_id is not None and fcp_lambda > 0.0:
    eos_probs = _probs[..., int(fcp_eos_id)].to(logits.dtype)
    ...

# IE 從 _probs 算 entropy
if self._enable_ie:
    _H = -(_probs * torch.log(_probs + 1e-12)).sum(dim=-1)  # [B, T]
    ...
```

**關鍵點**：
- 僅在需要時才做 softmax（條件判斷在外面）
- softmax 只做一次，FCP 和 IE 共用
- 不啟用任何機制時，零額外開銷

### 其他計算的效能特點

| 機制 | 計算量 | 記憶體 | 可最佳化 |
|---|---|---|---|
| Final-SW | O(BT) cumsum + exp | 可忽略 | cumsum 在 int32 做，exp 僅對 final 區域 |
| PDL | O(BT) 查表 | 0（index 操作） | `self._pdl_weights[labels]` 是 single kernel |
| InfoEntropy | O(BTV) softmax + O(BTV)entropy | ~1.5GB (_probs) | **複用 FCP softmax**，只在 final 區算 entropy |
| 三者乘積 | O(BT) element-wise mul | 可忽略 | 一次性 broadcast 乘 |

### 計算順序（減少記憶體峰值）

```
1. ce_loss_per_token 已經算出 (raw, [B*T])
2. 算 final_mask + final_offset (int32 cumsum, ~0 開銷)
3. 如果需要，做一次 softmax → _probs [B,T,V]  ← 最大 tensor
4. 從 _probs 取 eos channel → FCP
5. 從 _probs 算 entropy → IE weight [B,T]     ← 可以立刻 reshape 成 flat
6. del _probs 釋放 1.5GB
7. PDL 查表 → pdl_w [B*T]                      ← 已 flat
8. Final-SW → final_w [B*T]                    ← 已 flat
9. 五層乘積: raw * scale_w * sftgo_w * final_w * pdl_w * ie_w
```

**原則**：在 `_probs` 之後立刻算出 IE weight 並 reshape 成 flat，然後 `del _probs`，後續 PDL/Final-SW 都在 `[B*T]` 維度上操作，不接觸 `[B,T,V]`。

### torch.compile 注意

- cumsum mask 用 `torch.int32`（非 python bool），compile 友好
- `torch.where` 優於 `mask.float() * value`
- 避免在 forward 內用 `.item()` 或 Python `if`（已透過 `self._xxx > 0.0` 的 tensor 條件避開）

### 預期開銷總結

| 場景 | 額外 VRAM | 額外時間 |
|---|---|---|
| FCP only（原樣） | 0 | 0 |
| FCP + IE 全關 | 0 | 0 |
| FCP + IE 啟用 | 0（複用同一顆 _probs） | ~2-3%（entropy 計算） |
| 三者全啟用 | ~0（PDL/SW 可忽略） | ~3-5% 總訓練時間 |

---

## 2.6. 記錄與繪圖變更

### 2.6.1 CSV 新增欄位

**train CSV** (`output/train_sft_cot_log.csv`) 新增 2 欄，插入在 `scale_w` 之後、`lr` 之前：

```
step, loss, ce_loss, fcp_penalty, eos_prob, eos_prob_max, sftgo_loss, scale_w,
pdl_weight_mean,   ← 新增：batch 內 final 區 PDL 權重均值（PDL 關閉時=1.0）
ie_weight_mean,    ← 新增：batch 內 final 區 IE 權重均值（IE 關閉時=1.0）
lr, grad_norm, router_temp, tokens_seen, step_time_s
```

**val CSV** (`output/val_sft_cot_log.csv`) 新增 2 欄，接在 `val_batches` 之後：

```
step, val_ce_loss, val_loss_mean, val_fcp_penalty, val_eos_prob, val_eos_prob_max,
val_batches,
val_pdl_weight_mean,    ← 新增
val_ie_weight_mean      ← 新增
```

> 注意：不需要 `final_sw_alpha` 欄（無 trigger/warmup，值恆為 1.0 或 0）。

**實作方式**：沿用 train_sft.py 既有 CSV migrate pattern（`_migrate_sft_train_csv_add_eos_prob_max`、`_migrate_sft_val_csv_fcp_columns`），新增對應的 migrate 函式，舊 CSV 自動升級、歷史列新欄留空。

### 2.6.2 訓練 log 終端輸出

在現有 `L_tot ... P(EOS)_max ... scale ... ` 行尾追加：

```
pdl 0.82  ie 1.15
```

### 2.6.3 plot_sft_train_val_enhanced.py 修改

**檔案**：`scripts/tools/plot_sft_train_val_enhanced.py`

| 改動 | 位置 | 說明 |
|---|---|---|
| `detect_features()` 擴充 | ~line 109-124 | 新增 `final_enhance` key：偵測 `pdl_weight_mean` in cols 或 `ie_weight_mean` in cols |
| 新增 `panel_final_enhance()` | ~line 488 之後 | 同一個 panel：pdl_mean（左軸）+ ie_mean（右軸），觀察權重變化趨勢 |
| 顏色擴充 | ~line 62-75 | `_C` 陣列加 2 色給 pdl/ie |
| layout 計算 | ~line 586-590 | `n_optional += 1 if features["final_enhance"] else 0` |
| panel 呼叫 | ~line 648 之後 | 在 SCALe panel 之後插入 `panel_final_enhance()` |

### 2.6.4 plot.sh / plot_sft_train_val.py

**`scripts/tools/plot_sft_train_val.py`**（非 enhanced 版）：只需確認不會因未知欄位 crash（目前用 csv.DictReader + _f()，未知 key 回 None，不會 crash，不需改）。

**`scripts/tools/plot.sh`**：預設呼叫 plot 指令，確認路徑指向 latest CSV。

### 2.6.5 cot_task/validate_and_plot.py

**不需要改**。此腳本讀的是 `structure_weights_bundle.pt`，畫結構權重分布／SCALe schedule，不讀訓練 CSV。新機制無交集。

---

## 3. 測試步驟（依序）

1. **跑 1.1-1.4 事前驗證** — 確認 token id、頻率
2. **跑 precompute_pdl_weights.py** — 產生 pdl_weights.pt
3. **修改 FCP token** — 先用現有 checkpoint 跑幾步，確認 `P(EOS)` 和 `fcp_penalty` 有變化
4. **加入 Final-SW + PDL + IE** — 預設全開，跑 50 步確認不 crash、metric 有值、loss 未爆炸
5. **觀察 loss jump** — 首次啟用時 loss 會因加權改變而跳躍（非模型崩潰），確認後續逐步下降即可
6. **跑繪圖** — `plot_sft_train_val_enhanced.py` 確認曲線

---

## 4. 風險點

| 風險 | 緩解 |
|---|---|
| PDL 權重把結構 token 壓太低 | 特殊 token (32000-32006) 權重設為 1.0 |
| InfoEntropy softmax 兩次 | 複用 FCP 的 softmax |
| final_offset cumsum 在 batch 維度亂掉 | 先 reshape 成 flat，每個樣本獨立算 mask |
| 啟用時 loss 跳躍（非崩潰） | 預期行為：權重改變導致 loss 值平移，不影響收斂 |
| IE 權重過大導致不穩 | 可設 `SFT_COT_ENABLE_IE=false` 單獨關閉 |
| CSV column 不相容歷史檔 | train_sft.py 已有 migrate 邏輯，沿用其 pattern |


