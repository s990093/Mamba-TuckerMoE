# -*- coding: utf-8 -*-
"""
SFT-CoT（stf）：資料來自 `dataset/stf.json` 經 `scripts/stf_cot_to_bin.py` 物化為
`dataset/stf_cot_hf` + `dataset/stf_cot_train.bin`；詞表請用 Tiny-LLM + 7 special：

1. 使用既有 tokenizer：`dataset/tokenizer`（需含 32007 與 7 個 special）
2. `python3 scripts/stf_cot_to_bin.py`（預設依 category 插入三種 system：對話／任務／總結；可 `--no-system`）
3. 抽查：`python3 scripts/spot_check_stf_cot.py`；mask smoke：`python3 scripts/verify_stf_cot_mask.py`
4. SFT：`python3 scripts/train_sft_cot.py`（須在 bundle 根目錄，`PYTHONPATH=scripts` 或由 `scripts/training/start_training.sh` 啟動）
   - **預設（base 新開）**：若存在 `dataset/sft_cot_base_model.pt`（可由 `scripts/export/import_kaggle_pt_to_base.py --download` 從 Kaggle 匯入），**優先**只載權重、**step=0** 重跑，不會自動續跑 `output/checkpoint_sft_cot_s*.pt`。
   - **完整續跑**：`SFT_COT_RESUME_CKPT=output/checkpoint_sft_cot_s560.pt ./scripts/training/start_training.sh`，或 `SFT_COT_AUTO_RESUME=1`（有 base 時仍選 output 最大 step）。
   - 手動指定：`SFT_COT_CHECKPOINT_LOAD=...`；僅權重：`SFT_COT_INIT_WEIGHTS_ONLY=true`。

與 GAIR/LIMA SFT 分檔；訓練邏輯見 `train_sft.train_sft`（hf_text + mask.md）。

## FCP + SFT-GO + SCALe 三合一 Loss（CoT 特定設定）

預設：FCP=True、SCALe=True、SFT-GO=False（需提供 bundle 檔才啟用）。
可透過環境變數覆寫（1/true/yes/on = 開，0/false/no/off = 關）：

  SFT_COT_ENABLE_FCP=true       FCP：<think> 區域 EOS 懲罰（λ=0.2, δ=0.01）
  SFT_COT_ENABLE_SCALE=true     SCALe：<think> 區域 CE 餘弦退火（η_max=1.0, η_min=0.3）
  SFT_COT_ENABLE_SFTGO=true     SFT-GO：結構 token 加權（需 structure_weights_bundle.pt；設 false 關閉）

  SFT_COT_FCP_LAMBDA=0.2        FCP λ（懲罰強度）
  SFT_COT_FCP_DELTA=0.01        FCP δ（EOS 機率閾值）
  SFT_COT_SCALE_ETA_MAX=1.0     SCALe η_max（訓練初始 think 權重）
  SFT_COT_SCALE_ETA_MIN=0.3     SCALe η_min（訓練尾端 think 權重）
  SFT_COT_SCALE_W_FINAL=1.0     SCALe final 區域固定權重

## Final Enhance：Final-SW + PDL + InfoEntropy（預設全開，無觸發/暖機）

需先產生 PDL 權重：`python3 scripts/precompute_pdl_weights.py`（缺檔自動關 PDL）。

  SFT_COT_ENABLE_FINAL_ENHANCE=false   總開關（關閉三者）
  SFT_COT_ENABLE_FINAL_SW=false        單獨關 Final-SW（final 區前綴加權）
  SFT_COT_ENABLE_PDL=false             單獨關 PDL（token 頻率靜態加權）
  SFT_COT_ENABLE_IE=false              單獨關 InfoEntropy（final 區熵動態加權）
  SFT_COT_FINAL_SW_ETA=0.1             Final-SW 衰減率
  SFT_COT_FINAL_SW_LAMBDA=0.8          Final-SW 前綴增益
  SFT_COT_PDL_ALPHA=0.4                PDL 指數（僅供記錄；權重由 precompute 決定）
  SFT_COT_IE_GAMMA=0.5                 InfoEntropy 權重幅度
  SFT_COT_IE_ON_THINK=false            IE 是否也作用於 <think> 區
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from train_sft import train_sft


def _sft_cot_enable_fcp() -> bool:
    """FCP 預設啟用（CoT 訓練需抑制 <think> 內的提前 EOS）。設 SFT_COT_ENABLE_FCP=false 關閉。"""
    v = (os.environ.get("SFT_COT_ENABLE_FCP") or "true").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def _sft_cot_enable_scale() -> bool:
    """SCALe 預設啟用（動態調整 <think> 區域 CE 權重）。設 SFT_COT_ENABLE_SCALE=false 關閉。"""
    v = (os.environ.get("SFT_COT_ENABLE_SCALE") or "true").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def _sft_cot_enable_sftgo() -> bool:
    """SFT-GO 預設啟用（需 cot_task/reports/structure_weights_bundle.pt）。設 SFT_COT_ENABLE_SFTGO=false 關閉。"""
    v = (os.environ.get("SFT_COT_ENABLE_SFTGO") or "true").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def _env_float_cot(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    try:
        return float(v) if v else default
    except ValueError:
        return default


_RE_SFT_COT_CKPT_STEP = re.compile(r"^checkpoint_sft_cot_s(\d+)$")


def _find_max_step_sft_cot_pt(output_dir: Path) -> Path | None:
    """`output/checkpoint_sft_cot_s*.pt` 中取 step 最大者。"""
    best_step = -1
    best: Path | None = None
    for p in output_dir.glob("checkpoint_sft_cot_s*.pt"):
        m = _RE_SFT_COT_CKPT_STEP.match(p.stem)
        if not m:
            continue
        s = int(m.group(1))
        if s > best_step:
            best_step = s
            best = p
    return best


def _parse_sft_cot_init_weights_only_env() -> bool | None:
    """若未設定 `SFT_COT_INIT_WEIGHTS_ONLY` 則回傳 None。"""
    v = os.environ.get("SFT_COT_INIT_WEIGHTS_ONLY")
    if v is None:
        return None
    v = str(v).strip()
    if not v:
        return None
    return v.lower() not in ("0", "false", "no", "off")


def _env_auto_resume() -> bool:
    """設 `SFT_COT_AUTO_RESUME=1` 時，即使有 base 也會先選 output 最大 step 完整續跑。"""
    v = (os.environ.get("SFT_COT_AUTO_RESUME") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _resolve_sft_cot_checkpoint() -> tuple[str, bool]:
    """回傳 (`CHECKPOINT_LOAD`, `SFT_INIT_WEIGHTS_ONLY`)。

    優先序：
    1) `SFT_COT_CHECKPOINT_LOAD`
    2) `dataset/sft_cot_base_model.pt`（預設僅權重、step=0 新開；除非設 `SFT_COT_AUTO_RESUME=1` 跳過）
    3) `output/checkpoint_sft_cot_s*.pt` step 最大（`SFT_COT_AUTO_RESUME=1` 或無 base 時；預設完整續跑）
    4) Kaggle SFT model-only
    5) `dataset/pre_train_base_model.pt`
    """
    override = os.environ.get("SFT_COT_CHECKPOINT_LOAD")
    init_explicit = _parse_sft_cot_init_weights_only_env()

    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"SFT_COT_CHECKPOINT_LOAD 不是有效檔案: {override}")
        print(f"ℹ️  COT checkpoint load (env) = {p}", flush=True)
        init_only = init_explicit if init_explicit is not None else True
        return str(p), init_only

    bundle_base = Path("dataset/sft_cot_base_model.pt")
    if bundle_base.is_file() and not _env_auto_resume():
        print(
            f"ℹ️  COT checkpoint load = {bundle_base.resolve()} "
            f"(base model-only，從 step=0 新開；完整續跑請設 SFT_COT_AUTO_RESUME=1 或 SFT_COT_RESUME_CKPT)",
            flush=True,
        )
        init_only = init_explicit if init_explicit is not None else True
        return str(bundle_base.resolve()), init_only

    out = Path("output")
    latest = _find_max_step_sft_cot_pt(out)
    if latest is not None:
        if init_explicit is not None:
            init_only = init_explicit
        else:
            init_only = False
        mode = "完整續跑" if not init_only else "僅權重"
        _m = _RE_SFT_COT_CKPT_STEP.match(latest.stem)
        _step_s = _m.group(1) if _m else "?"
        print(
            f"ℹ️  COT auto checkpoint load = {latest.resolve()} ({mode}; step={_step_s})",
            flush=True,
        )
        return str(latest.resolve()), init_only

    if bundle_base.is_file():
        print(f"ℹ️  COT checkpoint load = {bundle_base.resolve()} (model-only base)", flush=True)
        init_only = init_explicit if init_explicit is not None else True
        return str(bundle_base.resolve()), init_only

    KAGGLE_BASE = os.path.expanduser(
        "~/.cache/kagglehub/datasets/s990093/sft-step-27510-model-only/versions/3"
        "/checkpoint_sft_s53280_model_only.pt"
    )
    fallback = KAGGLE_BASE if Path(KAGGLE_BASE).is_file() else "dataset/pre_train_base_model.pt"
    print(f"ℹ️  COT checkpoint load = {fallback}", flush=True)
    init_only = init_explicit if init_explicit is not None else True
    return fallback, init_only


if __name__ == "__main__":
    _ckpt_path, _init_w_only = _resolve_sft_cot_checkpoint()
    train_sft(
        SFT_DATA_SOURCE="hf_text",
        LIMA_HF="dataset/stf_cot_hf",
        DATA_PATH="dataset/stf_cot_train.bin",
        CHECKPOINT_LOAD=_ckpt_path,
        SFT_INIT_WEIGHTS_ONLY=_init_w_only,
        CHECKPOINT_SAVE="output/checkpoint_sft_cot.pt",
        LOG_FILE="output/train_sft_cot_log.csv",
        VAL_LOG_FILE="output/val_sft_cot_log.csv",
        TOKENIZER_DIR="dataset/tokenizer",
        VOCAB_SIZE=32007,
        SEQ_LEN=1024,
        BATCH_SIZE=4,  # 2卡 GA=6 → eff_batch=4×2×6=48
        GRADIENT_ACCUMULATION_STEPS=6,
        LR=3e-6,
        EPOCHS=10,
        WARMUP=0,
        WARMUP_FRAC=0.0,
        WEIGHT_DECAY=0.0,
        MAX_GRAD_NORM=5.0,
        SAVE_EVERY_STEPS=50,
        VAL_EVERY_STEPS=200,
        # FCP: 關閉 think 區 EOS penalty
        ENABLE_FCP=False,
        # SCALe: think=1.0 final=1.0（普通加權，不退火）
        ENABLE_SCALE=_sft_cot_enable_scale(),
        SCALE_ETA_MAX=_env_float_cot("SFT_COT_SCALE_ETA_MAX", 1.0),
        SCALE_ETA_MIN=_env_float_cot("SFT_COT_SCALE_ETA_MIN", 1.0),
        SCALE_W_FINAL=_env_float_cot("SFT_COT_SCALE_W_FINAL", 1.0),
        # SFT-GO: 結構 token 加權（預設開啟；需提供 structure_weights_bundle.pt）
        ENABLE_SFTGO=_sft_cot_enable_sftgo(),
        # ---- Final Enhance: Final-SW + PDL + InfoEntropy（預設全開，無觸發/暖機）----
        # 個別關閉用環境變數：SFT_COT_ENABLE_FINAL_ENHANCE / _FINAL_SW / _PDL / _IE
        # 超參環境變數：SFT_COT_FINAL_SW_ETA / _FINAL_SW_LAMBDA / _IE_GAMMA / _IE_ON_THINK
        # PDL 權重需先跑 scripts/precompute_pdl_weights.py 產生 pdl_weights.pt（缺檔自動關閉）
        ENABLE_FINAL_ENHANCE=True,
        ENABLE_FINAL_SW=True,
        ENABLE_PDL=True,
        ENABLE_IE=False,
        FINAL_SW_ETA=_env_float_cot("SFT_COT_FINAL_SW_ETA", 0.1),
        FINAL_SW_LAMBDA=_env_float_cot("SFT_COT_FINAL_SW_LAMBDA", 0.8),
        PDL_ALPHA=_env_float_cot("SFT_COT_PDL_ALPHA", 0.2),
        PDL_WEIGHTS_PATH="cot_task/reports/pdl_weights.pt",
        IE_GAMMA=_env_float_cot("SFT_COT_IE_GAMMA", 0.5),
        IE_ON_THINK=(os.environ.get("SFT_COT_IE_ON_THINK", "").strip().lower()
                     in ("1", "true", "yes", "on")),
    )
