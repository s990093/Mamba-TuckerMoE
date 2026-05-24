以下是將 **InfoEntropy Loss** 與 **Power‑Law Decay (PDL)** 無縫疊加到「比例自適應前綴權重衰減（Final‑SW）」方案上的完整計劃。三個機制彼此正交，可直接相乘於 per‑token CE loss 上，並同樣在驗證 PPL ≈ 16 後啟用，讓你的模型在 Demo 階段同時擁有前綴聚焦、難度自適應與反高頻詞偏置三重增強。

---

## 1. 設計目標（更新）

- 保留原 Final‑SW 的「長度自適應前綴加權」。
- 加入 **InfoEntropy Loss**：讓模型後期自動聚焦於「仍然不確定的 token」（熵高），避免在已學會的位置浪費容量。
- 加入 **Power‑Law Decay**：根據訓練語料中 token 的頻率進行靜態反加權，抑制高頻功能詞的主導，提升低頻關鍵詞的學習信號。
- 三者僅作用於 **`<final>` 區域**（InfoEntropy 可選擇僅在 final 區或全域；為聚焦答案品質，建議限於 final 區），與 SCALe、SFT‑GO、FCP、MoE 輔助損失完全正交。
- 啟用時機：監控驗證 PPL，當其連續低於閾值（≈ 16）後，同時漸進引入 Final‑SW、InfoEntropy 與 PDL 三項加權。

---

## 2. 完整損失函數

最終 per‑token CE loss 乘上 **五個獨立權重**，再與輔助損失相加：

\[
\begin{aligned}
L*{\text{total}} = \frac{1}{N*{\text{valid}}} \sum*{t \in \text{valid}}
\Big[ & w*{\text{scale}}(t) \cdot w*{\text{sftgo}}(t) \cdot w*{\text{final}}(t) \cdot
w*{\text{pdl}}(y_t) \cdot w*{\text{ie}}(t) \cdot \ell*{\text{CE}}(t) \Big] \\
&+ \frac{0.1}{66} L*{\text{lb}} + \frac{0.005}{66} L*{\text{z}} + \lambda*{\text{FCP}} L\_{\text{FCP}}
\end{aligned}
\]

其中：

- \( w\_{\text{scale}}(t) \)：SCALe，在 `<think>` 區域由 1.0 餘弦退火至 0.3，其餘為 1。
- \( w\_{\text{sftgo}}(t) \)：結構 token（`</think>`, `</final>`, `<|im_end|>`）為 8.0，其餘為 1。
- \( w*{\text{final}}(t; L*{\text{final}}) \)：自適應前綴衰減權重（指數＋線性混合），僅在 `<final>` 區生效，詳見第 3 節。
- \( w\_{\text{pdl}}(y_t) \)：Power‑Law Decay 權重，僅取決於該 token 的 ID，由訓練集頻率預先計算（詳見第 4 節）。
- \( w\_{\text{ie}}(t) \)：InfoEntropy 權重，\( (1 - H(\mathbf{p}\_t))^{\gamma} \)，\( \mathbf{p}\_t \) 為該位置的輸出機率分佈（詳見第 5 節）。
- \( N\_{\text{valid}} \)：batch 中有效 token 總數（token‑centric 歸一化）。

三項新權重（\( w*{\text{final}}, w*{\text{pdl}}, w\_{\text{ie}} \)）均只在 `<final>` 區域不為 1，非 final 區皆為 1，因此不干擾 think 區域的訓練。若想讓 InfoEntropy 也作用於 think 區，可單獨控制，但為簡化與聚焦 Demo，建議限於 final。

---

## 3. 自適應前綴權重 \( w*{\text{final}}(i; L*{\text{final}}) \)（沿用原設計）

保持不變，詳見前案，此處僅列出關鍵公式：

\[
\tau(L*{\text{final}}) = \eta \cdot L*{\text{final}},\quad
w*{\text{exp}}(i) = \exp(-i/\tau),\quad
w*{\text{lin}}(i) = \max(1 - i/L*{\text{final}}, 0)
\]
\[
w*{\text{final}}(i) = \lambda \, w*{\text{exp}}(i) + (1-\lambda) \, w*{\text{lin}}(i)
\]

漸進啟用係數 \( \alpha \in [0,1] \) 使 \( w*{\text{eff}} = 1 + \alpha (w*{\text{final}} - 1) \)。

---

## 4. Power‑Law Decay 靜態權重 \( w\_{\text{pdl}}(y_t) \)

**論文**：_Power‑Law Decay Loss for Text Generation Finetuning_ (2025)

**動機**：標準 CE 損失對高頻 token（如 “的”、“了”、“。”）給予過多信號，導致模型被這些簡單 token 主導，忽略真正決定語義的低頻詞。

**設計**：統計 SFT 訓練集中每個 token \( t \) 的出現次數 \( f_t \)，計算平滑後的權重向量：

\[
w*{\text{pdl}}(t) = \left( \frac{f_t + \epsilon}{f*{\text{max}}} \right)^{-\alpha}
\]

- \( \epsilon \)：平滑項（如 1.0），避免除以零。
- \( f*{\text{max}} \)：最大頻率，用於歸一化，使權重範圍約在 \( [1, (f*{\text{max}}/\epsilon)^{\alpha}] \)。
- \( \alpha \)：衰減指數，建議 0.3~0.5。值越大，對低頻詞的強化越強。

**預計算**：只需在訓練前掃描一次 SFT 資料集，得到一個長度為 \( V=32007 \) 的浮點數組 `pdl_weights`。儲存為 `.pt` 檔案，訓練時載入。

**注意**：特殊 token（如 `<think>`, `</final>` 等）頻率極高，權重會變小，但 SFT‑GO 的 8× 加權會對沖，因此無需單獨處理。

**程式碼示例**（頻率統計腳本）：

```python
import torch
from collections import Counter

freq = Counter()
for sample in sft_dataset:
    freq.update(sample['input_ids'].tolist())

freq_tensor = torch.zeros(vocab_size)
for t, c in freq.items():
    freq_tensor[t] = c

f_max = freq_tensor.max()
epsilon = 1.0
alpha = 0.4
pdl_weights = ((freq_tensor + epsilon) / f_max) ** (-alpha)
torch.save(pdl_weights, 'cot_task/reports/pdl_weights.pt')
```

---

## 5. InfoEntropy 自適應難度權重 \( w\_{\text{ie}}(t) \)

**論文**：_Mitigating the Bias of Learning Difficulties in Language Model Pretraining with Focal Loss & InfoEntropy Loss_ (2024)

**動機**：Focal Loss 僅用預測機率 \( p\_{t_i} \) 來判斷難度，但語言中許多位置有多個合理 token。InfoEntropy 改用整個輸出分佈的熵 \( H(\mathbf{p}) \) 來衡量「不確定性」，更適合語言生成。

**權重公式**：

\[
H(\mathbf{p}) = -\sum*j p_j \log p_j \quad (\text{對數底 } e)
\]
\[
w*{\text{ie}} = (1 - H\_{\text{norm}})^{\gamma}
\]

其中 \( H\_{\text{norm}} = H / \log V \)（歸一化至 \([0,1]\)），\( \gamma \) 為聚焦強度（建議 0.5~1.0）。

- 當模型對某位置很混淆（熵高），\( H\_{\text{norm}} \to 1 \)，權重變大，強化學習。
- 當模型已很確定（熵低），\( H\_{\text{norm}} \to 0 \)，權重變小，減少過擬合。

**作用範圍**：為集中資源，建議 **僅在 `<final>` 區域** 內計算並施加 InfoEntropy 權重，其他區域設為 1.0。若想額外提升 think 區的推理品質，可再加一個開關單獨控制。

**計算效率**：每個位置需計算一次 softmax 與熵。對於 550M 模型、batch size 適中時，增加的開銷約 5~10%，可接受。

---

## 6. 啟用機制（統一觸發）

所有新加權（Final‑SW、PDL、InfoEntropy）均採用相同的 **驗證 PPL 觸發 + 線性漸進啟用** 策略：

- 監控 `val_ce_loss`，當其連續 3 個驗證點 ≤ \( \ln(16) \approx 2.77 \) 時，設置 `triggered = True`。
- 從觸發步數開始，在 `warmup_steps`（例如 500 步）內，讓三項權重從「等效為 1」線性過渡到完整權重：
  \[
  \alpha = \min\left(1, \frac{\text{step} - \text{trigger_step}}{\text{warmup_steps}}\right)
  \]
  \[
  w\_{\text{eff}}(t) = 1 + \alpha \cdot (w(t) - 1)
  \]
- PDL 與 InfoEntropy 也可全程開啟，但為了完全避免早期干擾，採用統一觸發更安全。

若希望手動啟動，可設定環境變數 `SFT_COT_ENABLE_FINAL_ENHANCE=true` 強制開啟三項。

---

## 7. 超參數總表（更新）

| 參數                       | 環境變數                           | 預設值                            | 說明                                  |
| -------------------------- | ---------------------------------- | --------------------------------- | ------------------------------------- |
| `ENABLE_FINAL_ENHANCE`     | `SFT_COT_ENABLE_FINAL_ENHANCE`     | `false`                           | 是否啟用 Final‑SW + PDL + InfoEntropy |
| `FINAL_SW_ETA`             | `SFT_COT_FINAL_SW_ETA`             | `0.1`                             | 前綴權重 τ = η × L_final              |
| `FINAL_SW_LAMBDA`          | `SFT_COT_FINAL_SW_LAMBDA`          | `0.8`                             | 指數/線性混合比                       |
| `PDL_ALPHA`                | `SFT_COT_PDL_ALPHA`                | `0.4`                             | 頻率衰減指數                          |
| `PDL_WEIGHTS_PATH`         | `SFT_COT_PDL_WEIGHTS_PATH`         | `cot_task/reports/pdl_weights.pt` | 預計算權重檔                          |
| `IE_GAMMA`                 | `SFT_COT_IE_GAMMA`                 | `0.5`                             | InfoEntropy 聚焦強度                  |
| `IE_ON_THINK`              | `SFT_COT_IE_ON_THINK`              | `false`                           | 是否也在 think 區啟用 IE              |
| `FINAL_ENHANCE_PPL_THRESH` | `SFT_COT_FINAL_ENHANCE_PPL_THRESH` | `16`                              | 觸發 PPL 閾值                         |
| `FINAL_ENHANCE_WARMUP`     | `SFT_COT_FINAL_ENHANCE_WARMUP`     | `500`                             | 漸進啟用步數                          |
| `FINAL_PREFIX_P`           | `SFT_COT_FINAL_PREFIX_P`           | `0.2`                             | 前綴準確率比例                        |

---

## 8. 程式碼整合步驟

### 8.1 模型 forward 修改（model.py）

在計算 per‑token CE loss 的段落，依序乘上權重。確保已取得 `final_mask` 與 `final_offset`。

```python
# 假設已有:
# ce_loss_per_token: [B, T] 未歸一化的 CE loss
# scale_w: [B, T] (SCALe)
# sftgo_w: [B, T] (SFT-GO)
# final_mask: [B, T] bool
# final_offset: [B, T] long
# labels: [B, T]

# --- 加載靜態 PDL 權重 ---
if self.pdl_weights is not None:
    pdl_w = self.pdl_weights[labels]            # [B, T]，根據 label 的 token id 取值
else:
    pdl_w = torch.ones_like(labels, dtype=torch.float32)

# --- 計算 InfoEntropy 權重 ---
if self.ie_gamma is not None and final_sw_alpha > 0:
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)       # [B, T, V]
        # 計算熵並歸一化
        H = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)   # [B, T]
        H_norm = H / math.log(vocab_size)
        ie_w = (1.0 - H_norm) ** self.ie_gamma    # [B, T]
    # 若只在 final 區使用
    if not self.ie_on_think:
        ie_w = torch.where(final_mask, ie_w, torch.ones_like(ie_w))
    # 漸進混合
    ie_w = 1.0 + final_sw_alpha * (ie_w - 1.0)
else:
    ie_w = torch.ones_like(labels, dtype=torch.float32)

# --- 計算 Final‑SW 權重 (與之前相同) ---
if final_sw_alpha > 0:
    final_len = final_mask.sum(dim=1).float().clamp(min=1.0)
    tau = self.final_sw_eta * final_len
    i = final_offset.float()
    w_exp = torch.exp(-i / tau.unsqueeze(1))
    w_lin = torch.clamp(1.0 - i / final_len.unsqueeze(1), min=0.0)
    w_final = self.final_sw_lambda * w_exp + (1 - self.final_sw_lambda) * w_lin
    final_weights = 1.0 + final_sw_alpha * (w_final - 1.0)
    final_weights = torch.where(final_mask, final_weights, torch.ones_like(final_weights))
else:
    final_weights = torch.ones_like(labels, dtype=torch.float32)

# --- 全部乘入 ---
weighted_ce = ce_loss_per_token * scale_w * sftgo_w * final_weights * pdl_w * ie_w
loss_ce = weighted_ce.sum() / valid_mask.sum()
```

### 8.2 訓練腳本觸發邏輯（train_sft.py）

```python
# 配置
enable_enhance = os.environ.get('SFT_COT_ENABLE_FINAL_ENHANCE', 'false').lower() == 'true'
enhance_ppl_thresh = float(os.environ.get('SFT_COT_FINAL_ENHANCE_PPL_THRESH', '16'))
ce_thresh = math.log(enhance_ppl_thresh)
warmup_steps = int(os.environ.get('SFT_COT_FINAL_ENHANCE_WARMUP', '500'))

triggered = False
trigger_step = None

# 訓練迴圈中，每個驗證後檢查
if enable_enhance and not triggered:
    if val_ce_loss <= ce_thresh:
        triggered = True
        trigger_step = global_step
        print(f"[Enhance] All final enhancements triggered at step {global_step}")

if triggered:
    alpha = min(1.0, (global_step - trigger_step) / warmup_steps)
else:
    alpha = 0.0

# 傳入模型
loss = model(input_ids, labels=labels, final_sw_alpha=alpha, ...)
```

### 8.3 預計算 PDL 權重腳本（獨立執行）

```python
# scripts/precompute_pdl.py
import torch
from collections import Counter
from dataset import get_sft_dataset   # 你現有的資料集載入

dataset = get_sft_dataset()
freq = Counter()
for sample in dataset:
    freq.update(sample['input_ids'].tolist())

vocab_size = 32007
freq_tensor = torch.zeros(vocab_size)
for t, c in freq.items():
    freq_tensor[t] = c

torch.save(freq_tensor, 'cot_task/reports/token_freq.pt')

# 產生 PDL 權重
alpha = 0.4
epsilon = 1.0
f_max = freq_tensor.max()
pdl_weights = ((freq_tensor + epsilon) / f_max) ** (-alpha)
torch.save(pdl_weights, 'cot_task/reports/pdl_weights.pt')
print("PDL weights saved.")
```

### 8.4 監控指標擴充

除原有 `final_prefix_acc`，新增以下記錄：

- `pdl_weight_mean`：batch 中 final 區域 pdl 權重的平均值（反映當前 batch 的低頻詞密度）。
- `ie_weight_mean`：final 區域 InfoEntropy 權重均值，可看出模型對答案區的「不確定性」。
- `final_ce_weighted`：經過所有五項加權後的 final 區域 CE loss（便於觀察權重是否過激）。

在驗證/訓練日誌中加入這些欄位，並修改繪圖腳本支援。

---

## 9. 預期效果與調參方向

- **綜合效果**：Final‑SW 確保答案開頭正確，PDL 讓低頻關鍵詞不被淹沒，InfoEntropy 讓模型集中解決仍不確定的位置。三者在 final 區內相乘，形成一個「聚焦前綴、壓制高頻、強化難點」的協同作用。
- **Demo 穩定性**：由於 PDL 是靜態權重，不會因梯度變化而抖動；InfoEntropy 雖動態但變化平緩，兩者都能平穩改善，不會引入新的不穩因素。
- **若出現不穩**：
  - `val_ce_loss` 突然上升：可能 InfoEntropy 的 \( \gamma \) 過大，先降低至 0.3；或暫時關閉 IE 只留 Final‑SW + PDL。
  - `final_prefix_acc` 未提升：增加 PDL 的 \( \alpha \)（如 0.6）強化低頻詞，或減小 Final‑SW 的 \( \eta \) 使前綴聚焦更集中。
  - 產生重複或退化：考慮稍微降低 Final‑SW 的 \( \lambda \)（少用指數尖峰），避免過度獎勵少數開頭 token。

---

## 10. 快速啟動指令

準備好 PDL 權重檔後，當你看到 PPL 降至 ~16 時執行：

```bash
SFT_COT_ENABLE_FINAL_ENHANCE=true \
SFT_COT_FINAL_SW_ETA=0.1 \
SFT_COT_FINAL_SW_LAMBDA=0.8 \
SFT_COT_PDL_ALPHA=0.4 \
SFT_COT_IE_GAMMA=0.5 \
python3 scripts/train_sft_cot.py
```

所有新增機制將在觸發後自動漸進啟用，無需中斷訓練。

---

這份計劃將 InfoEntropy 與 Power‑Law Decay 無縫融入你原有的 Final‑SW 方案，三層加權共同作用於 `<final>` 區域，讓你的模型在 Demo 時展現更精準、更穩定的答案生成能力。如需調整權重作用範圍（例如讓 InfoEntropy 也覆蓋 think 區）或提供更具體的程式碼插入位置，我可以進一步協助。

以下是以 ASCII 描繪的完整損失組合架構圖，展示五層權重如何疊加於基礎 CE loss，以及與輔助損失的匯總。

```
   input_ids, labels            logits (from Mamba3-TuckerMoE)
        |                              |
        |   +--------------------------+
        |   |
        v   v
  ┌─────────────────────────────────────┐
  │  per-token CE loss ℓ_CE(t)           │
  │  shape [B, T]                       │
  └──────────────┬──────────────────────┘
                 |
                 |  (按順序逐層乘上權重，正交無衝突)
                 v
  ┌──────────────────────────────────────┐
  │ ① SCALe  w_scale(t)                  │
  │   think 區域: 1.0 ──餘弦退火──► 0.3  │
  │   其他區域: 1.0                       │
  └──────────────┬───────────────────────┘
                 |
                 v
  ┌──────────────────────────────────────┐
  │ ② SFT-GO  w_sftgo(t)                 │
  │   </think>  </final>  <|im_end|>     │
  │   三個結構 token → × 8.0              │
  │   其餘 token → × 1.0                  │
  └──────────────┬───────────────────────┘
                 |
                 v
  ┌──────────────────────────────────────┐  final 區域 only
  │ ③ Final-SW  w_final(t; L_final)      │
  │   τ = η·L_final                      │
  │   w = λ·exp(-i/τ) + (1-λ)·(1-i/L)   │
  │   漸進 α: 0 → 1 (線性 warmup)        │
  └──────────────┬───────────────────────┘
                 |
                 v
  ┌──────────────────────────────────────┐  final 區域 only (可選全域)
  │ ④ Power‑Law Decay  w_pdl(y_t)        │
  │   靜態: ∝ freq(token)⁻⁰·⁴             │
  │   依 label 的 token ID 查表給權重     │
  └──────────────┬───────────────────────┘
                 |
                 v
  ┌──────────────────────────────────────┐  final 區域 only (可選全域)
  │ ⑤ InfoEntropy  w_ie(t)               │
  │   H = -Σ p log p                     │
  │   w_ie = (1 - H/log V)^γ             │
  │   漸進 α 混合 (同 Final‑SW)          │
  └──────────────┬───────────────────────┘
                 |
                 v
  ┌──────────────────────────────────────┐
  │  Weighted CE = Σ_{t} (product_of_5_w) · ℓ_CE(t)  │
  │  L_main = Weighted_CE / N_valid      │  (token‑centric 平均)
  └──────────────┬───────────────────────┘
                 |
                 v
  ┌──────────────────────────────────────┐
  │  + MoE auxiliary losses              │
  │    L_lb  (load balance) × 0.1/66     │
  │    L_z   (router z-loss) × 0.005/66  │
  └──────────────┬───────────────────────┘
                 |
                 v
  ┌──────────────────────────────────────┐
  │  + FCP penalty (僅 think 區)          │
  │    λ_fcp · max(P(EOS|t)-δ, 0)²       │
  └──────────────┬───────────────────────┘
                 |
                 v
              L_total
```

**區域作用說明**

- **① SCALe**：僅降低 `<think>` 區的整體權重（隨訓練進度），讓後期更重視 `<final>`。
- **② SFT‑GO**：全序列範圍，強迫模型記住三個關鍵格式 token。
- **③ Final‑SW**：只在 `<final>` 內部，根據答案長度自適應衰減，聚焦前綴 token。
- **④ PDL**：靜態查表，對高頻詞降權、低頻詞升權，主要用於 `<final>` 區提升語義 token 的影響。
- **⑤ InfoEntropy**：動態計算，對模型仍不確定的位置（高熵）賦予更高權重，同樣集中在 `<final>` 區。

五層權重在 `<final>` 區內 **直接相乘**，產生「前綴聚焦 × 低頻強化 × 難點放大」的協同效果；在 `<think>` 區僅有 ① 與 ② 作用，並由 FCP 獨立壓制過早的 EOS。三項輔助損失（MoE load‑balance / z‑loss / FCP）則以加法形式加入，確保訓練穩定。
