# SUMMARY — sft_cot_bundle 專案總覽

## 這個 Bundle 是什麼？

針對 **550M 參數 Mamba3-TuckerMoE 混合模型** 的 CoT（Chain-of-Thought）監督微調（SFT）管線，
在標準 CE loss 上疊加三種 Loss 工程機制，改善推理鏈的完整性與格式品質。

---

## 模型架構

```
Mamba3LanguageModel (550M)
├── Embedding  [V=32007, D=768]
├── 6× Hybrid Block
│   ├── Mamba3 Scan  (selective SSM, parallel scan)
│   └── TuckerMoE FFN
│       ├── 8 專家 (top-2 routing)
│       └── Tucker 分解: U_expert ⊗ core (r1×r2×r3)
└── LM Head  [D=768 → V=32007]
```

**詞表特殊 token**（32007 詞）：

| Token | ID | 用途 |
|-------|----|------|
| `<|im_end|>` | 32001 | ChatML message 結尾 |
| `<think>` | 32002 | 推理區開始 |
| `</think>` | 32003 | 推理區結尾 |
| `<final>` | 32004 | 答案區開始 |
| `</final>` | 32005 | 答案區結尾 |

---

## 完整損失函數

$$L_{\text{total}} = \underbrace{L_{\text{CE,weighted,scaled}}}_{\text{主項 ≈ 92\%}} + \underbrace{\frac{0.1}{66} L_{\text{lb}} + \frac{0.005}{66} L_{\text{z}}}_{\text{MoE 輔助 ≈ 0.1\%}} + \underbrace{\lambda L_{\text{FCP}}}_{\text{CoT 格式 ≈ 8\%}}$$

| 項目 | 作用 | 權重係數 |
|------|------|---------|
| $L_{\text{CE}}$ + SFT-GO + SCALe | 學習正確 token 序列 | 1.0 |
| $L_{\text{lb}}$ | 防止 MoE 專家負載不均 | 0.1 / 66 ≈ 0.0015 |
| $L_{\text{z}}$ | 防止路由器 logit 崩潰 | 0.005 / 66 ≈ 7.6e-5 |
| $L_{\text{FCP}}$ | 抑制 think 區提前 EOS | λ = 0.2（可調）|

---

## 三個 Loss 機制

### 1. FCP（Format & EOS Penalty）

**作用**：在 `<think>` 區內，當 EOS 概率超過閾值 $\delta = 0.01$ 時施加懲罰。

$$L_{\text{FCP}} = \frac{\lambda}{|M_{\text{think}}|} \sum_t M_{\text{think}}[t] \cdot \max\!\left(\text{P(EOS}|t) - \delta,\ 0\right)^2$$

- **CoT 預設**：啟用，$\lambda = 0.2$，$\delta = 0.01$
- **效果**：`eos_prob` 從 ~0.2 降至 ~0.01–0.05

### 2. SFT-GO（Structure Token Weighting）

**作用**：對 `</think>`、`</final>`、`<|im_end|>` 三個關鍵 token 的 CE loss 乘以 8.0。

$$w[t] = \begin{cases} 8.0 & t \in \{\texttt{</think>},\ \texttt{</final>},\ \texttt{<|im\_end|>}\} \\ w_{\text{bundle}}[t] & \text{otherwise} \end{cases}$$

- **CoT 預設**：結構倍數 8.0 恆啟用；bundle 加權需 `SFT_COT_ENABLE_SFTGO=true`

### 3. SCALe（Scheduled Cosine Loss Annealing）

**作用**：`<think>` 區域 CE 權重從 1.0 餘弦退火至 0.3，使模型訓練後期更重視 `<final>` 輸出。

$$\eta_{\text{think}}(s) = 0.3 + 0.35 \cdot \left(1 + \cos\frac{\pi s}{S}\right) \in [0.3,\ 1.0]$$

- **CoT 預設**：啟用，$\eta_{\max} = 1.0$，$\eta_{\min} = 0.3$

---

## 目錄結構

```
sft_cot_bundle/
├── scripts/
│   ├── model.py                 # 模型 + Triton kernel + Loss 計算
│   ├── train_sft.py             # SFT 主訓練迴圈（匯出 train_sft()）
│   ├── train_sft_cot.py         # CoT wrapper（啟用 FCP + SCALe）
│   └── tools/
│       └── plot_sft_train_val_enhanced.py   # 訓練曲線繪圖（MA + PPL）
├── cot_task/
│   └── reports/
│       └── structure_weights_bundle.pt      # SFT-GO 預計算權重（非必須）
├── output/
│   ├── train_sft_cot_log.csv    # 訓練日誌（12 欄）
│   └── val_sft_cot_log.csv      # 驗證日誌
├── TASK.md                      # 任務規格 + 實作細節
└── SUMMARY.md                   # 本文件
```

---

## 快速上手

### 啟動訓練

```bash
cd sft_cot_bundle
# FCP + SCALe 已為 CoT 預設啟用
python3 scripts/train_sft_cot.py
```

### 覆蓋參數

```bash
# 增強 FCP 懲罰
SFT_COT_FCP_LAMBDA=0.5 python3 scripts/train_sft_cot.py

# 關閉所有新機制（純基線）
SFT_COT_ENABLE_FCP=false \
SFT_COT_ENABLE_SCALE=false \
python3 scripts/train_sft_cot.py

# 全開（需 bundle）
SFT_COT_ENABLE_SFTGO=true python3 scripts/train_sft_cot.py
```

### 繪製訓練曲線

```bash
python3 scripts/tools/plot_sft_train_val_enhanced.py \
    output/train_sft_cot_log.csv \
    output/val_sft_cot_log.csv \
    --ma 200 -dpi 200 -o output/plots.png
```

---

## 健康訓練的指標

| 指標 | 預期走勢 | 異常信號 |
|------|---------|---------|
| `ce_loss` | 單調遞減 | 停滯 > 200 步 |
| `eos_prob` | 0.2 → 0.02 | 長期 > 0.1 → 增大 `FCP_LAMBDA` |
| `scale_w` | 1.0 → 0.3 | 無變化 → 檢查 token ID |
| `fcp_penalty` | 先升後降 | 一直為 0 → FCP 未啟用 |
| `router_temp` | 2.0 → 0.5 | 急降 → 檢查 checkpoint |
| `val_ce_loss` | ≈ `ce_loss` | 差距 > 20% → 可能過擬合 |

---

## 關鍵程式碼位置

| 功能 | 檔案 | 行號 |
|------|------|------|
| z_loss 計算 | `model.py` | L322 |
| lb_loss 計算 | `model.py` | L333 |
| FCP 懲罰計算 | `model.py` | L844–866 |
| 四項總和 | `model.py` | L868 |
| SCALe 計算 | `train_sft.py` | L1550–1558 |
| SCALe mask 生成 | `train_sft.py` | L1583–1596 |
| FCP/SFT-GO/SCALe 參數（函式） | `train_sft.py` | L789–806 |
| CoT 預設設定 | `train_sft_cot.py` | L110–141 |
