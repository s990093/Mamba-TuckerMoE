# Loss 組成說明 & 訓練診斷指南

> 對應 CSV 欄位 `output/train_sft_cot_log.csv` 和終端 log 輸出行

---

## 1. 總 Loss 結構

```
L_total = (weighted CE) / N_valid + L_lb × 0.1/66 + L_z × 0.005/66 + L_fcp
```

其中 weighted CE = base CE × **五層權重乘積**：

```
W(t) = w_scale(t) × w_sftgo(t) × w_final(t) × w_pdl(t) × w_ie(t)
```

---

## 2. 各組件說明

### 2.1 CE loss（基礎）

| CSV 欄 | 終端顯示 | 說明 |
|---|---|---|
| `ce_loss` | `CE` | **未加權的原始 cross-entropy**，保留作為基準比較線 |

- 越低越好，反映模型對下一個 token 的預測準確度
- PPL = exp(ce_loss)，例如 ce_loss=2.77 → PPL≈16
- 這個值是「未加權」的，不會因為 Final-SW/PDL/IE 權重而改變，用於公平對比前後版本

### 2.2 total loss（最終）

| CSV 欄 | 終端顯示 | 說明 |
|---|---|---|
| `loss` | `L_tot` | **所有加權 + aux loss 後的總 loss**，optimizer 真正優化的值 |

- 啟用 Final Enhance 後，`loss` 會**大於** `ce_loss`（因為 final 區被加權放大）
- loss - ce_loss ≈ fcp + aux + 加權增額
- 這是 optimizer 看到的梯度來源

---

### 2.3 SCALe（Think 區域權重衰減）

| 機制 | CSV 欄 | 終端顯示 | 範圍 |
|---|---|---|---|
| SCALe | `scale_w` | `scale 0.85` | think: 1→0.3, 其他: 1.0 |

**做什麼**：隨訓練進度，把 `<think>` 區的 CE 權重從 1.0 餘弦退火到 0.3

**為什麼**：訓練初期需要學習推理格式，後期應聚焦 `<final>` 答案區。SCALe 逐步把計算資源從 think 轉移到 final

**怎麼看**：
- 早期 0.99、中期 ~0.85、後期 ~0.3
- 如果 think 消失太快（模型不會推理了），降 `SFT_COT_SCALE_ETA_MIN`（從 0.3 改 0.5）
- 如果 answer 品質沒提升，可能 SCALe 衰減太慢，降 `SFT_COT_SCALE_ETA_MIN`

---

### 2.4 SFT-GO（結構 Token 加權）

| 機制 | CSV 欄 | 終端顯示 | 說明 |
|---|---|---|---|
| SFT-GO | `sftgo_loss` | 無獨立顯示 | 僅在 FCP panel 對比 |

**做什麼**：對三個結構 token（`</think>`=32003, `</final>`=32005, `<|im_end|>`=32001）的 CE loss 做 **8x 加權**

**為什麼**：這些 token 決定輸出格式的正確性，忘了其中一個就會格式崩壞

**怎麼看**：
- `sftgo_loss` 應略大於 `ce_loss`（因為結構 token 權重較高）
- 如果 `sftgo_loss ≈ ce_loss`（一模一樣），表示 `structure_weights_bundle.pt` 可能未載入或 SFT-GO 未生效
- 格式錯誤（少 `</think>` 等）→ 確認 SFT-GO 有啟用

---

### 2.5 FCP（Format / EOS Penalty — 修正後）

| 機制 | CSV 欄 | 終端顯示 | 說明 |
|---|---|---|---|
| FCP penalty | `fcp_penalty` | `FCP 0.0000` | `<think>` 內過早終止的懲罰 |
| 平均 EOS 機率 | `eos_prob` | `P(EOS) 0.0000` | think 區內 `<|im_end|>` 的平均機率 |
| 最大 EOS 機率 | `eos_prob_max` | `P(EOS)_max 0.0002` | think 區內 `<|im_end|>` 的單點最大機率 |

**做什麼**：監控 `<think>` 區域內 `<|im_end|>` 的預測機率，超過 δ=0.01 (1%) 時施加平方懲罰

**為什麼**：防止模型在推理未完時提前輸出結束標記。修正後監控 `<|im_end|>` 而非 `</s>`

**修正前 vs 修正後**：
| | 修正前 | 修正後 |
|---|---|---|
| 監控 token | `</s>` (id=2) | `<|im_end|>` (id=32001) |
| 訓練資料出現次數 | 2 / 13.3M | 122,868 / 13.3M |
| 預期 P(EOS) | ~0（model 沒學過） | 有意義的數值 |
| fcp_penalty | 永遠 0（bug） | 偶爾觸發 |

**怎麼看**：
- `P(EOS)_max` 應該比修正前顯著提高（因為 `<|im_end|>` 是模型學過的 token）
- `fcp_penalty > 0` 表示模型在某些 think 位置有提前結束的傾向，FCP 在壓制
- `P(EOS)_max` 持續上升 → 模型逐漸忘記 think 區不該結束，需觀察
- 正常情況 `fcp_penalty` 應該很小但非零，偶爾有 spike

---

### 2.6 Final-SW（答案前綴自適應加權）🆕

| 機制 | CSV 欄 | 終端顯示 | 說明 |
|---|---|---|---|
| Final-SW | 無獨立欄位 | 合併在 loss 中 | `<final>` 區位置權重 |

**做什麼**：在 `<final>` 區域內，越靠近開頭權重越高，指數+線性混合衰減

```
w_final(i) = 0.8 × exp(-i/τ) + 0.2 × (1 - i/L)
τ = 0.1 × L_final
```

**為什麼**：答案的前幾個 token 決定整句方向（主詞、動詞、語態），錯了後面全歪。集中學習資源在前綴

**預期效果**：
- final 區開頭 token 的 loss 貢獻變大
- 答案首 token 準確率提升（尤其長答案）
- Demo 時第一句不容易歪掉

---

### 2.7 PDL（Power-Law Decay 頻率反加權）🆕

| 機制 | CSV 欄 | 終端顯示 | 說明 |
|---|---|---|---|
| PDL | `pdl_weight_mean` | `pdl 0.82` | batch 內 final 區 PDL 權重均值 |

**做什麼**：對高頻 token（的、了、是、and、the）降權，對低頻關鍵詞升權

```
w_pdl(token) = (freq(token) / freq_max)^(-0.4)
```

**為什麼**：標準 CE loss 被高頻功能詞主導，語義關鍵詞（名詞、動詞）的學習信號被淹沒

**預期效果**：
- `pdl_weight_mean` < 1.0 表示 batch 含較多高頻詞
- `pdl_weight_mean` > 1.0 表示 batch 含較多低頻關鍵詞
- 長期看，稀有詞的正確率提升，生成內容更豐富

---

### 2.8 InfoEntropy（不確定性聚焦）🆕

| 機制 | CSV 欄 | 終端顯示 | 說明 |
|---|---|---|---|
| InfoEntropy | `ie_weight_mean` | `ie 1.15` | batch 內 final 區 IE 權重均值 |

**做什麼**：模型對某位置越不確定（entropy 高），該位置的 loss 權重越大

```
w_ie = (1 - H/H_max)^0.5
H = -Σ p_j × log(p_j)
H_max = log(32007) ≈ 10.37
```

**為什麼**：與其均勻學習所有位置，不如集中火力在模型還搞不清楚的地方

**預期效果**：
- 訓練早期 entropy 高 → `ie_weight_mean` 較高（~1.2-1.5）
- 訓練後期模型變確定 → `ie_weight_mean` 趨近 1.0
- **這是唯一會隨訓練進度自然下降的權重**，可作為「學習進度指標」
- 若 `ie_weight_mean` 始終很高不降 → 模型學不動，可能 LR 太低或資料太難

---

## 3. Aux Loss（輔助損失）

| 組件 | 來源 | 權重 | 說明 |
|---|---|---|---|
| L_lb (load balance) | TuckerMoE router | × 0.1/66 | 鼓勵 expert 均勻使用，防止少數 expert 被過度依賴 |
| L_z (z-loss) | TuckerMoE router | × 0.005/66 | 防止 router logits 過大導致梯度不穩 |

兩個值都很小（經過 1/66 縮放），不會主導訓練，主要從 `train.py:1056-1060` 加入。

---

## 4. 其他監控指標

| CSV 欄 | 終端顯示 | 正常範圍 | 異常信號 |
|---|---|---|---|
| `lr` | `lr 8.00e-06` | 1e-6 ~ 1e-5 | 突然歸零或暴漲 |
| `grad_norm` | `\|grad\| 3.5` | 2~6 | >10 可能梯度爆炸；長期 <0.5 學習停滯 |
| `router_temp` | `T_router 0.500` | SFT 固定 0.5 | 不應變化（SFT 固定） |
| `step_time_s` | `20.0s/step` | 19~21s（雙卡） | >30s 可能 GPU throttling |

---

## 5. 訓練診斷流程

### 5.1 正常訓練曲線

```
step   ce_loss  loss   P(EOS)_max  scale  pdl   ie    lr
100    3.20     3.40   0.002       0.99   0.85  1.25  1e-5
500    3.05     3.30   0.005       0.95   0.84  1.18  1e-5
2000   2.85     3.10   0.003       0.75   0.83  1.08  8e-6
5000   2.80     3.00   0.002       0.45   0.83  1.03  3e-6
```

- ce_loss 緩降、loss 緩降、P(EOS)_max 控制在 <1%
- scale_w 從 1.0 退火到 ~0.3
- ie_weight_mean 從 >1.2 降到 ~1.0
- pdl_weight_mean 穩定在 0.8-0.9

### 5.2 首次啟用 Final Enhance 的預期跳躍

從舊 checkpoint resume 時，新權重首次生效：

```
step    ce_loss   loss     說明
1320    2.85      3.10     ← 舊 checkpoint，無新權重
1321    2.85      3.60     ← 新權重生效，loss 跳躍 ~0.5
1322    2.84      3.55     ← 開始收斂
...
1400    2.82      3.20     ← 回到正常軌道
```

- ce_loss 不變（它是未加權基準）
- loss 跳躍 ~0.3-0.8 合理（取決於 PDL 和 IE 的權重幅度）
- **只要 ce_loss 沒暴漲、grad_norm 沒異常，就是正常的**

### 5.3 異常診斷表

| 症狀 | 可能原因 | 診斷方式 |
|---|---|---|
| ce_loss 突然暴漲 | 模型崩潰 | 看 grad_norm 是否 >20，恢復上個 checkpoint |
| loss 不降但 ce_loss 降 | 加權過強 | 降 PDL_ALPHA 或 IE_GAMMA |
| P(EOS)_max 持續 >0.01 | 模型在 think 內頻繁預測結束 | FCP 應懲罰，確認 fcp_penalty >0 |
| ie_weight_mean 永遠 1.0 | IE 未啟用 | 檢查 `SFT_COT_ENABLE_IE` |
| pdl_weight_mean 永遠 1.0 | PDL 未載入 | 檢查 `cot_task/reports/pdl_weights.pt` 存在 |
| scale_w 不下降 | SCALe 未啟用 | 檢查 `SFT_COT_ENABLE_SCALE` |
| sftgo_loss = loss | SFT-GO 未生效 | 檢查 `cot_task/reports/structure_weights_bundle.pt` |

### 5.4 快速健康檢查指令

```bash
# 看最後 5 步
tail -5 output/train_sft_cot_log.csv | column -s, -t

# 看 FCP 是否有觸發過
rg "FCP [^0]" output/logs/train_sft_cot_*.log

# 看 val loss 趨勢（最後 10 點）
tail -10 output/val_sft_cot_log.csv | column -s, -t

# 繪圖
python3 scripts/tools/plot_sft_train_val_enhanced.py \
    output/train_sft_cot_log.csv output/val_sft_cot_log.csv --ma 200
```
