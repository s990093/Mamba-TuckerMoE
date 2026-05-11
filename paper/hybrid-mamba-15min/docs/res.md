# Ablation Results Log

本檔用來記錄「相同訓練算力預算下」的核心命題驗證：

- Dense-Small（baseline）
- TuckerMoE-Small（proposed）
- Dense-Large（larger dense reference）

目標是檢驗：在相同訓練 budget 下，`TuckerMoE-Small` 是否能接近或超過 `Dense-Large` 的生成能力代理指標（先用 `val_ce` / `loss`）。

---

## 1) 實驗固定條件（Controlled Variables）

- Dataset: `fineweb_tokenized.bin`
- Vocab size: `32007`
- Sequence length: `512`
- Global batch proxy: `BATCH_SIZE=1`, `GRADIENT_ACCUMULATION_STEPS=16`
- Optimizer/LR:
  - `LR=1.2e-4`
  - `WARMUP=400`
  - `STEPS=10000`（可用 `--steps` 覆寫）
- Dataloader:
  - `DATALOADER_WORKERS=None`
  - `DATALOADER_PIN_MEMORY=None`
  - `DATALOADER_PREFETCH_FACTOR=4`
  - `DATALOADER_PERSISTENT_WORKERS=True`
  - `SHUFFLE_BUFFER_SIZE=2048`
- Regularization:
  - `DROPOUT=0.1`
  - `LABEL_SMOOTHING=0.05`
  - `Z_LOSS_COEFF=1.5e-2`
  - `LB_LOSS_COEFF=0.1`
  - `WEIGHT_DECAY=0.1`
- Runtime:
  - `TRAIN_MODE=True`
  - `COMPILE_ENABLED=False`
- Validation:
  - `VAL_ENABLED=True`
  - `VAL_EVERY_STEPS=500`
  - `VAL_FRACTION=0.005`
  - `VAL_MAX_BATCHES=64`

---

## 2) 消融變因（Independent Variables）

### `dense_small`

- `D_MODEL=512`
- `NUM_LAYERS=6`
- `USE_KMOE=False`
- 其他主幹: `D_STATE=64`, `D_HEAD=64`, `EXPAND=2`, `MIMO_RANK=4`, `NUM_KV_HEADS=4`, `CHUNK_SIZE=64`

### `tuckermoe_small`

- `D_MODEL=512`
- `NUM_LAYERS=6`
- `USE_KMOE=True`
- `KMOE_NUM_EXPERTS=8`
- `KMOE_TOP_K=2`
- `KMOE_R1=16`, `KMOE_R2=384`, `KMOE_R3=192`
- 其他主幹同上

### `dense_large`

- `D_MODEL=640`
- `NUM_LAYERS=7`
- `USE_KMOE=False`
- 其他主幹: `D_STATE=64`, `D_HEAD=64`, `EXPAND=2`, `MIMO_RANK=4`, `NUM_KV_HEADS=4`, `CHUNK_SIZE=64`

---

## 3) 執行指令（Repro Commands）

```powershell
python .\run_ablation.py --preset dense_small --steps 10000
python .\run_ablation.py --preset tuckermoe_small --steps 10000
python .\run_ablation.py --preset dense_large --steps 10000
python .\plot_ablation.py --root output/ablation --out output/ablation/ablation_compare.png
```

一鍵版：

```powershell
.\run_ablation_all.ps1
```

16~18 小時完整版（3 組 × 1500 steps）：

```powershell
python .\run_ablation.py --preset dense_small --steps 1500
python .\run_ablation.py --preset tuckermoe_small --steps 1500
python .\run_ablation.py --preset dense_large --steps 1500
python .\plot_ablation.py --root output/ablation --out output/ablation/ablation_compare.png
python .\summarize_ablation.py --root output/ablation --md docs/ABLATION_RESULTS.md --figure output/ablation/ablation_compare.png
```

---

## 5) 本次消融結果記錄（Auto Summary）

_Last updated: 2026-04-24 22:44:49_

| Preset          | Final step | Best val_ce | Best step | Final loss | Avg step_time_s |
| --------------- | ---------: | ----------: | --------: | ---------: | --------------: |
| Dense-Small     |        N/A |         N/A |       N/A |        N/A |             N/A |
| TuckerMoE-Small |        N/A |         N/A |       N/A |        N/A |             N/A |
| Dense-Large     |        N/A |         N/A |       N/A |        N/A |             N/A |

### Figure

- Curve path: `output/ablation/ablation_compare.png`
- Metric priority: `val_ce` (main), `loss` (secondary)

---

## 6) 對照結論（Auto Draft）

- `TuckerMoE-Small` vs `Dense-Small`:
  - N/A（資料未齊）
- `TuckerMoE-Small` vs `Dense-Large`:
  - N/A（資料未齊）

---

## 5) 本次消融結果記錄（Auto Summary）

_Last updated: 2026-04-24 22:56:51_

| Preset          | Final step | Best val_ce | Best step | Final loss | Avg step_time_s |
| --------------- | ---------: | ----------: | --------: | ---------: | --------------: |
| Dense-Small     |         37 |         N/A |       N/A |    10.3658 |          13.347 |
| TuckerMoE-Small |        N/A |         N/A |       N/A |        N/A |             N/A |
| Dense-Large     |        N/A |         N/A |       N/A |        N/A |             N/A |

### Figure

- Curve path: `output/ablation/ablation_compare.png`
- Metric priority: `val_ce` (main), `loss` (secondary)

---

## 6) 對照結論（Auto Draft）

- `TuckerMoE-Small` vs `Dense-Small`:
  - N/A（資料未齊）
- `TuckerMoE-Small` vs `Dense-Large`:
  - N/A（資料未齊）
