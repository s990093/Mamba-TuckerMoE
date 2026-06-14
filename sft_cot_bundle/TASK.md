# TASK — FCP + SFT-GO + SCALe Loss 工程實作

## 任務目標

在現有 SFT-CoT 訓練管線中加入三種互補的 Loss 工程機制，改善 CoT 推理品質：

| 機制 | 全名 | 目的 |
|------|------|------|
| **FCP** | Format & EOS Penalty | 抑制 `<think>` 內提前終止 |
| **SFT-GO** | Structure Token Weighting | 對關鍵結構 token 加權 |
| **SCALe** | Scheduled Cosine Loss Annealing | 動態調整 think 區 CE 權重 |

---

## 完整損失函數

### 數學定義（4 項）

$$L_{\text{total}} = L_{\text{CE,weighted,scaled}} + \underbrace{\frac{0.1}{n} \cdot L_{\text{lb}}}_{\text{MoE 負載}} + \underbrace{\frac{0.005}{n} \cdot L_{\text{z}}}_{\text{MoE 正則}} + \underbrace{\lambda \cdot L_{\text{FCP}}}_{\text{CoT 格式}}$$

其中 $n = \texttt{num\_layers} \times 11$（Tucker MoE 權重張量數，預設 $6 \times 11 = 66$）

### 各項定義

**① CE Loss（主項，含 SFT-GO + SCALe）**

$$L_{\text{CE}} = -\frac{1}{N} \sum_{t} \underbrace{\eta[t](s)}_{\text{SCALe}} \cdot \underbrace{w[t]}_{\text{SFT-GO}} \cdot \mathbb{1}[\text{label}_t \neq -100] \cdot \log P(\text{label}_t \mid \mathbf{x}_{<t})$$

**② SCALe 余弦退火**（針對 think 區）

$$\eta_{\text{think}}(s) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi s}{S}\right)$$

- $\eta_{\max} = 1.0$（訓練初），$\eta_{\min} = 0.3$（訓練末）
- final 區：$\eta_{\text{final}} = 1.0$（恆定）

**③ SFT-GO 結構加權**

$$w[t] = \begin{cases} \alpha_{\text{struct}} = 8.0 & \text{if } \text{label}_t \in \{\texttt{</think>}, \texttt{</final>}, \texttt{<|im\_end|>}\} \\ w_{\text{bundle}}[t] & \text{otherwise} \end{cases}$$

**④ FCP EOS 懲罰**（只作用在 `<think>` 區）

$$L_{\text{FCP}} = \frac{1}{|M_{\text{think}}|} \sum_{t} M_{\text{think}}[t] \cdot \lambda \cdot \max\!\left(\log \frac{P(\text{EOS}|t)}{\delta}, 0\right)$$

- $\lambda = 0.2$（懲罰強度），$\delta = 0.01$（EOS 機率閾值）

**⑤ MoE 負載均衡（lb_loss）**

$$L_{\text{lb}} = K \sum_{i=1}^{E} f_i \cdot P_i, \quad K = 8 \text{（專家數）}$$

- $f_i$ = 專家 $i$ 的實際選中率，$P_i$ = router 概率

**⑥ MoE 路由正則（z_loss）**

$$L_{\text{z}} = \mathbb{E}\left[(\log\sum_i e^{\text{capped\_logit}_i})^2\right]$$

---

## 修改的檔案

### `scripts/train_sft.py`

**新增 9 個函式參數**（在 `SFT_STRUCTURE_TOKEN_CE_MULT` 之後）：

```python
def train_sft(
    ...
    SFT_STRUCTURE_TOKEN_CE_MULT: float = 8.0,
    # ---- FCP + SFT-GO + SCALe 顯式參數（None → 讀環境變數）----
    ENABLE_SFTGO: Optional[bool] = None,
    ENABLE_FCP: Optional[bool] = None,
    ENABLE_SCALE: Optional[bool] = None,
    STRUCT_BUNDLE_PATH: Optional[str] = None,
    FCP_LAMBDA: Optional[float] = None,
    FCP_DELTA: Optional[float] = None,
    SCALE_ETA_MAX: Optional[float] = None,
    SCALE_ETA_MIN: Optional[float] = None,
    SCALE_W_FINAL: Optional[float] = None,
):
```

**修改讀取邏輯**（函式參數優先 > 環境變數 > 內建預設）：

```python
# 舊版：只讀環境變數
SFT_USE_FCP = _env_flag("SFT_USE_FCP", False)

# 新版：函式參數覆蓋環境變數
SFT_USE_FCP = ENABLE_FCP if ENABLE_FCP is not None else _env_flag("SFT_USE_FCP", False)
```

### `scripts/train_sft_cot.py`

**新增 4 個 helper 函式**：

```python
def _sft_cot_enable_fcp()   -> bool   # 預設 True（可用 SFT_COT_ENABLE_FCP=false 關閉）
def _sft_cot_enable_scale() -> bool   # 預設 True
def _sft_cot_enable_sftgo() -> bool   # 預設 False（需 bundle）
def _env_float_cot(name, default) -> float
```

**更新 `train_sft()` 呼叫**，CoT 預設啟用 FCP + SCALe：

```python
train_sft(
    ...
    ENABLE_FCP=_sft_cot_enable_fcp(),        # True
    FCP_LAMBDA=_env_float_cot("SFT_COT_FCP_LAMBDA", 0.2),
    FCP_DELTA=_env_float_cot("SFT_COT_FCP_DELTA", 0.01),
    ENABLE_SCALE=_sft_cot_enable_scale(),    # True
    SCALE_ETA_MAX=_env_float_cot("SFT_COT_SCALE_ETA_MAX", 1.0),
    SCALE_ETA_MIN=_env_float_cot("SFT_COT_SCALE_ETA_MIN", 0.3),
    SCALE_W_FINAL=_env_float_cot("SFT_COT_SCALE_W_FINAL", 1.0),
    ENABLE_SFTGO=_sft_cot_enable_sftgo(),    # False
)
```

### `scripts/tools/plot_sft_train_val_enhanced.py`

完整重寫，新增：
- **移動平均**：`--ma 200`（預設 window=200），原始訊號淡化、MA 粗線
- **PPL 副軸**：exp(CE) 困惑度顯示在右 y 軸
- **論文排版**：serif 字體、無上/右邊框、200 DPI
- **修正 `-dpi` 參數解析**（支援 `-dpi 150` 和 `--dpi 150`）

---

## 環境變數一覽

### CoT 專用（`train_sft_cot.py`）

| 變數 | 預設 | 說明 |
|------|------|------|
| `SFT_COT_ENABLE_FCP` | `true` | FCP 開關 |
| `SFT_COT_FCP_LAMBDA` | `0.2` | EOS 懲罰強度 |
| `SFT_COT_FCP_DELTA` | `0.01` | EOS 機率閾值 |
| `SFT_COT_ENABLE_SCALE` | `true` | SCALe 開關 |
| `SFT_COT_SCALE_ETA_MAX` | `1.0` | think 初始權重 |
| `SFT_COT_SCALE_ETA_MIN` | `0.3` | think 最終權重 |
| `SFT_COT_SCALE_W_FINAL` | `1.0` | final 區恆定權重 |
| `SFT_COT_ENABLE_SFTGO` | `false` | SFT-GO 開關（需 bundle）|

### 通用（`train_sft.py`，可被 CoT 繼承覆蓋）

| 變數 | 預設 | 說明 |
|------|------|------|
| `SFT_USE_FCP` | `false` | 通用 FCP 開關 |
| `SFT_USE_SCALE` | `false` | 通用 SCALe 開關 |
| `SFT_USE_SFTGO` | `false` | 通用 SFT-GO 開關 |
| `SFT_STRUCTURE_TOKEN_CE_MULT` | `8.0` | 結構 token 倍數 |
| `SFT_STRUCT_BUNDLE_PATH` | `auto` | bundle 路徑 |

---

## 訓練指令

```bash
# 標準 CoT 訓練（FCP + SCALe 自動啟用）
python3 scripts/train_sft_cot.py

# 關閉 FCP
SFT_COT_ENABLE_FCP=false python3 scripts/train_sft_cot.py

# 啟用全部（含 SFT-GO，需 bundle）
SFT_COT_ENABLE_SFTGO=true \
SFT_STRUCT_BUNDLE_PATH=cot_task/reports/structure_weights_bundle.pt \
python3 scripts/train_sft_cot.py

# 繪圖
python3 scripts/tools/plot_sft_train_val_enhanced.py \
    output/train_sft_cot_log.csv output/val_sft_cot_log.csv \
    --ma 200 -dpi 200
```

---

## CSV 日誌欄位

`output/train_sft_cot_log.csv` 的各欄含義：

| 欄位 | 對應 | 說明 |
|------|------|------|
| `step` | 全域步數 | — |
| `loss` | $L_{\text{total}}$ | 四項總和（主指標）|
| `ce_loss` | $L_{\text{CE}}$（未加權）| 基準 CE，方便對照歷史曲線 |
| `fcp_penalty` | $\lambda L_{\text{FCP}}$ | FCP 貢獻，0 = 未啟用 |
| `eos_prob` | $\mathbb{E}[P(\text{EOS}) \mid M_{\text{think}}]$ | think 區 EOS 概率（應降低）|
| `sftgo_loss` | $L_{\text{total}} - L_{\text{FCP}}$ | CE+MoE 的代理值 |
| `scale_w` | $\eta_{\text{think}}(s)$ | 當前 SCALe 權重（1.0→0.3）|
| `lr` | 學習率 | — |
| `grad_norm` | $\|\nabla\|_2$ | 梯度範數 |
| `router_temp` | $T_{\text{router}}(s)$ | MoE 路由溫度 |
| `tokens_seen` | 累計 token 數 | — |
| `step_time_s` | 本步耗時（秒）| — |

---

## 實作細節

### Think 遮罩計算（無 CPU sync）

```python
is_start = (input_ids == think_start_id).to(torch.int32)
is_end   = (input_ids == think_end_id).to(torch.int32)
think_mask = (is_start.cumsum(dim=1) - is_end.cumsum(dim=1)) > 0
```

### FCP 懲罰計算（model.py line 850–866）

```python
eos_probs = F.softmax(logits.float(), dim=-1)[..., eos_id]
excess    = F.relu(eos_probs - fcp_delta)           # (·)+ = max(·, 0)
denom     = region_mask.sum().clamp(min=1.0)
fcp_penalty = (excess * excess * region_mask).sum() / denom * fcp_lambda
```

> 注意：model.py 實際用的是 $(\text{excess})^2$ 而非 $\text{excess}$，比 log-space 版本更平滑。

### SCALe 計算（train_sft.py line 1550–1596）

```python
_progress = min(1.0, global_step / max(1, STEPS))
scale_w_think = SFT_SCALE_ETA_MIN + 0.5 * (SFT_SCALE_ETA_MAX - SFT_SCALE_ETA_MIN) \
                * (1.0 + math.cos(math.pi * _progress))
```

---

## 待辦 / 未來工作

- [ ] 提供 `structure_weights_bundle.pt` 生成腳本（才能完整啟用 SFT-GO）
- [ ] 加入 val loss 的 FCP / SCALe breakdown 欄位（目前 val 為未加權 CE）
- [ ] 提供超參敏感度實驗（λ vs eos_prob 收斂速度）
