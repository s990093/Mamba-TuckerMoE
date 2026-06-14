# -*- coding: utf-8 -*-
"""
SFT 訓練（獨立於 train.py，不修改原檔）：
- 一筆樣本 = 一則完整對話（預設）或 **`SFT_SINGLE_TURN_EXPAND=True` 時** 將多輪拆成多筆「單輪 user→assistant」；不做多列合併 / packing。
- 若單筆 token 長度 > SEQ_LEN+1 會**保留前段、截掉後段**。尾部通常含 `</s>`、`<|im_end|>` 等關閉符——截斷過多會讓模型少學到「何時停」；請看啟動時的長度分佈與 `sft_lima_len_stats.json`。
- **SEQ_LEN 建議**：預設 **512**；要覆蓋長對話可試 **1024/2048** 並觀察 VRAM 與截斷率。
- 本管線的 LIMA 樣本為 **instruction 格式**（`im_start` / `im_end`），你若有另加 CoT/`<final>` 結構，更需長 `SEQ_LEN` 以免切尾。
- 啟動時掃 HF，輸出長度分佈到 stdout 與 `output/sft_*_len_stats*.json`。粒度／過濾不同時檔名含 `_single`、`_noweak`、`_instr` 等後綴；見 `sft_instruction_filter.py`。
- Val：自 train 隨機劃 5%（可調 VAL_FRAC），用與訓練**相同** label mask 算 val_ce / val_loss，寫入 `VAL_LOG_FILE`（預設 `output/val_sft_log.csv`）；表頭可含 `val_fcp_penalty` / `val_eos_prob` / `val_eos_prob_max`（與訓練同理之 FCP 區內平均，啟動時自動從舊 4 欄升級）。
- **CSV**：訓練 log 含 `fcp_penalty`、`eos_prob`；若表頭尚無 `eos_prob_max`，**下次啟動主行程**會自動在 `eos_prob` 後插入該欄（歷史列該欄留空）。與 `P(EOS)_max` 終端欄對照可看 think 內最大 EOS 機率是否超過 `FCP_DELTA`。
- **SFT_STRUCTURE_TOKEN_CE_MULT**（預設 8）：對 `</think>`、`</final>`、`<|im_end|>` 等「關閉／轉折」label 位置的 CE 加權（5～10 倍區間）；**CSV 的 ce_loss 仍為未加權平均**以利對照舊曲線。環境變數 `SFT_STRUCTURE_TOKEN_CE_MULT` 可覆寫；設 ≤1 關閉。
- 可設 **SFT_TEST_PROMPT**（非空字串）：與 `val_every_eff` 同節律做續寫實驗性 decode；`SFT_TEST_TEMPERATURE=0` 為 **greedy**，>0 為 multinomial+可選 `SFT_TEST_TOP_P` (nucleus)。另寫入 `SFT_TEST_LOG_FILE`。
- 訓練長度預設依 **EPOCHS**（例如 3）與訓練筆數、batch、grad_accum 推算總 `STEPS`；`EPOCHS=None` 時改用上界 **STEPS_MAX**。
- **WARMUP** 不會再「min(本段新步, 舊 200)」壓到近全程；改為新步數的 **WARMUP_FRAC**（預設 8%）並受 **WARMUP** 作步數上限。
- **SFT 計步**：只載入預訓 **.pt（state_dict）** 或無 `sft` 的完整檔時，**一律從 global_step=0** 開始；僅當載入 **含 `sft: True` 的完整 checkpoint** 且 **未** 設 `SFT_INIT_WEIGHTS_ONLY=True` 時，才用檔內 step 續跑並可還原 optimizer/scheduler。`SFT_INIT_WEIGHTS_ONLY=True` 僅取 `model`，用於從他輪 SFT 權重**新開**另一資料集之 SFT（如 e3 → sft_cot）。

預設載入 `dataset/pre_train_base_model.pt`（可由 `export_checkpoint_weights.py` 從預訓練 checkpoint 匯出）；續跑 SFT 指向含 `sft: True` 的 `checkpoint_sft_sK.pt`。**Checkpoint** 預設每 100 step 寫一檔：`CHECKPOINT_SAVE` 去副檔名加 `_s100`、`_s200`…。環境變數 **`SFT_CHECKPOINT_LOAD`** 可覆寫載入路徑。

**函式預設資料**：`dataset/mix_a25_u75_ins_hf` + `dataset/mix_a25_u75_ins.bin`（`build_mix_alpaca_ultrachat.py --single-turn --instruction-filter ins_strict` 產物；列上已單輪，勿再開 `SFT_SINGLE_TURN_EXPAND`／`ins_strict`，改 `off`）。**Mixed CoT**：可改指向 `dataset/mix_a25_u75_cot_hf` + `.bin`。純 Alpaca：`alpaca_parquet_to_bin.py`。**LIMA**：`lima_to_bin.py`。**SFT-CoT**：`SFT_DATA_SOURCE="hf_text"`、`LIMA_HF=dataset/sft_cot_hf` 等。
- **T_router（MoE）**：預訓練退火終點為 `ROUTER_T_END=0.5`。新開 SFT 預設 `SFT_FIXED_ROUTER_T=0.5`，固定該溫度，不會因 global_step 從 0 而又用 2.0 暖 500 step。設 `SFT_FIXED_ROUTER_T=None` 則依 `ROUTER_T_START/END` 與 `ROUTER_WARMUP/ROUTER_TOTAL` 重新退火。續跑 `sft=True` 斷點時沿用 checkpoint 內 router 欄位。
"""
from __future__ import annotations

import csv
import gc
import inspect 
import json
import math
import os
import re
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from accelerate import Accelerator

# 導入 train 會執行其 CUDA/精度偵測（與 train.py 行為一致）
import train as train_mod
from model import Mamba3Config, Mamba3LanguageModel

from train import (
    get_lr_scheduler,
    get_router_temperature,
    print_model_analysis,
    resolve_dataloader_settings,
    validate_config,
    unwrap_model,
)
from lima_to_bin import (
    SPECIAL_TOKENS,
    _format_conversation,
    expand_conversation_turns_to_single_turn_chatml,
    split_multiturn_chatml_text_into_single_turns,
)
from sft_instruction_filter import (
    instruction_filter_keeps_chatml,
    normalize_instruction_filter_mode,
)


def _load_hf_train_split(hf_path: str | Path) -> Dataset:
    """
    `load_from_disk` 可為 DatasetDict（GAIR/lima 常含 'train'）
    或單一表（SFT-CoT：`Dataset.from_dict(...).save_to_disk`；型別可為
    `datasets.Dataset` 或 `datasets.arrow_dataset.Dataset`）。
    一律回傳單一可索引的 Dataset。
    """
    raw = load_from_disk(str(hf_path))
    if isinstance(raw, DatasetDict):
        if "train" in raw:
            return raw["train"]
        k = next(iter(raw))
        return raw[k]
    if isinstance(raw, Dataset):
        return raw
    # 單一表（例如 arrow_dataset.Dataset）與 `from datasets import Dataset` 可非同類
    if hasattr(raw, "column_names") and hasattr(raw, "__len__"):
        return raw  # type: ignore[return-value]
    raise TypeError(f"不預期 load_from_disk 回傳型別: {type(raw)!r} ({hf_path!r})")


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
MIXED_PRECISION = train_mod.MIXED_PRECISION

warnings.filterwarnings(
    "ignore",
    message=".*Online softmax is disabled on the fly.*",
    category=UserWarning,
)

def _load_pretrain_file(path: str) -> dict:
    """
    支援：
    - 輕量 `*_32007.pt`（export）：'state_dict'、'config'、'step'（僅供日誌；SFT 訓練計步應從 0）
    - 完整 checkpoint：'model'、可含 'sft': True 與 optimizer/scheduler — 僅 **sft=True** 時才視為 SFT 斷點續跑
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"不支援的權重檔: {path!r}（需為 dict）")
    sft_flag = bool(raw.get("sft", False))
    if "state_dict" in raw and "model" not in raw:
        return {
            "model": raw["state_dict"],
            "config": raw.get("config"),
            "step": int(raw.get("step", 0)),
            "optimizer": None,
            "scheduler": None,
            "kind": "weights_only",
            "sft": sft_flag,
        }
    if "model" in raw:
        return {
            "model": raw["model"],
            "config": raw.get("config"),
            "step": int(raw.get("step", 0)),
            "optimizer": raw.get("optimizer") if isinstance(raw.get("optimizer"), dict) else None,
            "scheduler": raw.get("scheduler") if isinstance(raw.get("scheduler"), dict) else None,
            "kind": "full",
            "sft": sft_flag,
            "sft_warmup_steps": raw.get("sft_warmup_steps"),
            "sft_total_steps": raw.get("sft_total_steps"),
        }
    raise ValueError(f"權重檔需含 'model' 或 'state_dict'：{path}")


def _mamba3_config_from_ckpt(cd: dict | None, fb: dict) -> Mamba3Config:
    if not isinstance(cd, dict):
        cd = {}
    base_kwargs = dict(
        d_model=cd.get("d_model", fb["D_MODEL"]),
        d_state=cd.get("d_state", fb["D_STATE"]),
        d_head=cd.get("d_head", fb["D_HEAD"]),
        expand=cd.get("expand", fb["EXPAND"]),
        num_layers=cd.get("num_layers", fb["NUM_LAYERS"]),
        use_parallel_scan=cd.get("use_parallel_scan", True),
        chunk_size=cd.get("chunk_size", fb["CHUNK_SIZE"]),
        use_kmoe=cd.get("use_kmoe", True),
        kmoe_num_experts=cd.get("kmoe_num_experts", fb["KMOE_NUM_EXPERTS"]),
        kmoe_top_k=cd.get("kmoe_top_k", fb["KMOE_TOP_K"]),
        kmoe_r1=cd.get("kmoe_r1", fb["KMOE_R1"]),
        kmoe_r2=cd.get("kmoe_r2", fb["KMOE_R2"]),
        kmoe_r3=cd.get("kmoe_r3", fb["KMOE_R3"]),
        ffn_expand=cd.get("ffn_expand", fb["FFN_EXPAND"]),
        mimo_rank=cd.get("mimo_rank", fb["MIMO_RANK"]),
        num_kv_heads=cd.get("num_kv_heads", fb["NUM_KV_HEADS"]),
        layer_scale_init=cd.get("layer_scale_init", 1e-2),
    )
    extra_kwargs = dict(
        router_warmup=cd.get("router_warmup", fb.get("ROUTER_WARMUP", 500)),
        router_total=cd.get("router_total", fb.get("ROUTER_TOTAL", 10_000)),
        router_t_start=cd.get("router_t_start", fb.get("ROUTER_T_START", 2.0)),
        router_t_end=cd.get("router_t_end", fb.get("ROUTER_T_END", 0.5)),
    )
    # 相容不同版本的 Mamba3Config：有些版本（model.py）不接受 router_* 參數。
    try:
        sig = inspect.signature(Mamba3Config.__init__)
        allowed = set(sig.parameters.keys())
    except (TypeError, ValueError):
        allowed = set(base_kwargs.keys()) | set(extra_kwargs.keys())
    kwargs = {k: v for k, v in {**base_kwargs, **extra_kwargs}.items() if k in allowed}
    return Mamba3Config(**kwargs)


def _build_xy_masked(
    text: str,
    tok: "object",
    seq_len: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ChatML SFT：在 token id 序列上找 `<|im_start|>assistant\\n`，對其後直到（含）任一結尾符
    的連續片段計算 CE；`labels[j]=ids[j]` 再與 `y[i]=labels[i+1]` 對齊，使標頭最後一顆
    token 仍監督「預測首個正文 token」。結尾符以完整子序列比對（支援多 token 的 special）。
    """
    ids: list[int] = list(tok.encode(text, add_special_tokens=False))
    t = len(ids)
    if t == 0:
        z = torch.full((seq_len,), pad_id, dtype=torch.long)
        y = torch.full((seq_len,), -100, dtype=torch.long)
        return z, y

    labels: list[int] = [-100] * t

    header_ids: list[int] = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    header_len = len(header_ids)
    stop_seqs: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    _stop_redacted = "<|" + "redacted" + "_" + "im" + "_" + "end" + "|" + ">"
    _stop_im_end = "<|" + "im" + "_" + "end" + "|" + ">"
    for end_s in (_stop_redacted, _stop_im_end):
        seq = tok.encode(end_s, add_special_tokens=False)
        if not seq:
            continue
        key = tuple(seq)
        if key in seen:
            continue
        seen.add(key)
        stop_seqs.append(seq)

    def _match_stop_len_at(j: int) -> int:
        """若 ids[j:] 以某結尾序列開頭則回傳該序列長度，否則 0。"""
        best = 0
        for sq in stop_seqs:
            Lsq = len(sq)
            if Lsq and j + Lsq <= t and ids[j : j + Lsq] == sq:
                best = max(best, Lsq)
        return best

    i = 0
    while i < t and header_len > 0:
        if i + header_len <= t and ids[i : i + header_len] == header_ids:
            start_idx = i + header_len
            end_exclusive = start_idx
            while end_exclusive < t:
                mlen = _match_stop_len_at(end_exclusive)
                if mlen:
                    end_exclusive += mlen
                    break
                end_exclusive += 1
            for j in range(start_idx, end_exclusive):
                labels[j] = ids[j]
            i = end_exclusive
        else:
            i += 1

    if t > seq_len + 1:
        ids = ids[: seq_len + 1]
        labels = labels[: seq_len + 1]
        t = len(ids)

    need = seq_len + 1
    pad = need - t
    if pad > 0:
        ids = ids + [pad_id] * pad
        labels = labels + [-100] * pad

    x = torch.tensor(ids[:seq_len], dtype=torch.long)
    y = torch.tensor(labels[1 : seq_len + 1], dtype=torch.long)
    return x, y


def materialize_lima_hf_examples(
    full_rows: Dataset,
    row_indices: list[int],
    single_turn_expand: bool,
    instruction_filter: str = "off",
) -> tuple[list[str], dict]:
    out: list[str] = []
    origin: list[int] = []
    n_cand = 0
    n_drop = 0
    for i in row_indices:
        row = full_rows[int(i)]
        conv = row.get("conversations")
        if not isinstance(conv, list) or not conv:
            continue
        turns = [str(x) for x in conv]
        if single_turn_expand:
            frags = expand_conversation_turns_to_single_turn_chatml(turns)
        else:
            txt = _format_conversation(turns)
            frags = [txt] if txt.strip() else []
        for frag in frags:
            if not frag.strip():
                continue
            n_cand += 1
            if instruction_filter != "off" and not instruction_filter_keeps_chatml(
                frag, instruction_filter
            ):
                n_drop += 1
                continue
            out.append(frag)
            origin.append(int(i))
    meta = {
        "instruction_filter": instruction_filter,
        "n_instruction_candidates": n_cand,
        "n_instruction_dropped": n_drop,
        "n_kept": len(out),
        "row_origin": origin,
    }
    return out, meta


def materialize_hf_text_examples(
    full_rows: Dataset,
    row_indices: list[int],
    text_column: str,
    single_turn_expand: bool,
    instruction_filter: str = "off",
) -> tuple[list[str], dict]:
    out: list[str] = []
    origin: list[int] = []  # parallel: original HF row index for each fragment
    n_cand = 0
    n_drop = 0
    for i in row_indices:
        raw = str(full_rows[int(i)][text_column]).strip()
        if not raw:
            continue
        if single_turn_expand:
            frags = split_multiturn_chatml_text_into_single_turns(raw)
        else:
            frags = [raw]
        for frag in frags:
            if not frag.strip():
                continue
            n_cand += 1
            if instruction_filter != "off" and not instruction_filter_keeps_chatml(
                frag, instruction_filter
            ):
                n_drop += 1
                continue
            out.append(frag)
            origin.append(int(i))
    meta = {
        "instruction_filter": instruction_filter,
        "n_instruction_candidates": n_cand,
        "n_instruction_dropped": n_drop,
        "n_kept": len(out),
        "row_origin": origin,
    }
    return out, meta


class MaterializedSftDataset(Dataset):
    """一字串一個 supervised ChatML（已由 lima `_format_conversation` 格式化）。

    Always returns a 3-tuple `(x, y, sw)`:
      * `x`, `y`: input_ids and target labels, shape `[seq_len]`.
      * `sw`: per-token SFT-GO structure weight, shape `[seq_len]`, float32.
              All ones when the optional bundle is not provided — i.e. a no-op
              multiplier in the loss path, preserving baseline behaviour.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer_path: str | Path,
        seq_len: int,
        *,
        structure_weights: Optional[torch.Tensor] = None,  # [N_rows_total, seq_len]
        row_origin: Optional[list[int]] = None,            # local i → HF row idx
    ):
        self.texts = list(texts)
        self.tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        if self.tok.pad_token is None and SPECIAL_TOKENS:
            self.tok.pad_token = SPECIAL_TOKENS[6]
        self.pad_id = int(self.tok.convert_tokens_to_ids(self.tok.pad_token))
        self.seq_len = int(seq_len)
        self.tok.model_max_length = 1_000_000
        if structure_weights is not None and row_origin is not None:
            if len(row_origin) != len(self.texts):
                raise ValueError(
                    f"row_origin len {len(row_origin)} != texts len {len(self.texts)}"
                )
            self._structure_weights = structure_weights
            self._row_origin = list(row_origin)
        else:
            self._structure_weights = None
            self._row_origin = None
        self._ones = torch.ones(self.seq_len, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = _build_xy_masked(self.texts[i], self.tok, self.seq_len, self.pad_id)
        if self._structure_weights is not None and self._row_origin is not None:
            row = int(self._row_origin[i])
            if 0 <= row < int(self._structure_weights.shape[0]):
                sw = self._structure_weights[row].clone()
            else:
                sw = self._ones.clone()
        else:
            sw = self._ones.clone()
        return x, y, sw


def _lima_token_len_stats(
    lima_hf_root: str | Path,
    tokenizer_path: str | Path,
    seq_len: int,
    *,
    single_turn_expand: bool = False,
    instruction_filter: str = "off",
) -> dict:
    """
    跑完整 train 的原始 token 長度（不截斷），用來檢查 SEQ_LEN 是否足夠。
    """
    rows = _load_hf_train_split(lima_hf_root)
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    tok.model_max_length = 1_000_000
    lens: list[int] = []
    for i in range(len(rows)):
        conv = rows[i].get("conversations")
        if not isinstance(conv, list) or not conv:
            continue
        turns = [str(x) for x in conv]
        if single_turn_expand:
            frags = expand_conversation_turns_to_single_turn_chatml(turns)
        else:
            t0 = _format_conversation(turns)
            frags = [t0] if t0.strip() else []
        for frag in frags:
            if instruction_filter != "off" and not instruction_filter_keeps_chatml(
                frag, instruction_filter
            ):
                continue
            lens.append(len(tok.encode(frag, add_special_tokens=False)))
    if not lens:
        return {
            "single_turn_expand": bool(single_turn_expand),
            "instruction_filter": instruction_filter,
            "n_examples": 0,
            "error": "no_examples",
        }
    arr = np.asarray(lens, dtype=np.int64)
    need = int(np.sum(arr > seq_len + 1))
    short = int(np.sum(arr < 2))  # 幾乎空
    return {
        "single_turn_expand": bool(single_turn_expand),
        "instruction_filter": instruction_filter,
        "n_examples": int(len(lens)),
        "token_len_min": int(arr.min()) if len(arr) else 0,
        "token_len_max": int(arr.max()) if len(arr) else 0,
        "token_len_mean": float(arr.mean()) if len(arr) else 0.0,
        "p50": float(np.percentile(arr, 50)) if len(arr) else 0.0,
        "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
        "p95": float(np.percentile(arr, 95)) if len(arr) else 0.0,
        "p99": float(np.percentile(arr, 99)) if len(arr) else 0.0,
        "seq_len_setting": int(seq_len),
        "n_examples_truncated_len_gt_seq_plus_1": need,
        "truncation_rate": float(need / max(1, len(lens))),
        "n_trivially_short": short,
        "recommendation": (
            f"尾端常含關閉符，截斷率 {100 * need / max(1, len(lens)):.1f}% 時不建議用此 SEQ_LEN 長期訓練。"
            f" 若 p90/p95 > {seq_len + 1}，可先試 SEQ_LEN=1024 實測，再升 2048 覆蓋多數樣本。"
        ),
    }


def _hf_text_token_len_stats(
    hf_root: str | Path,
    tokenizer_path: str | Path,
    seq_len: int,
    text_column: str = "text",
    *,
    single_turn_expand: bool = False,
    instruction_filter: str = "off",
) -> dict:
    rows = _load_hf_train_split(hf_root)
    if text_column not in rows.column_names:
        return {
            "error": f"缺欄位 {text_column!r}，有 {rows.column_names}",
            "instruction_filter": instruction_filter,
            "single_turn_expand": single_turn_expand,
        }
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    tok.model_max_length = 1_000_000
    lens: list[int] = []
    for i in range(len(rows)):
        raw = str(rows[i][text_column]).strip()
        if not raw:
            continue
        if single_turn_expand:
            frags = split_multiturn_chatml_text_into_single_turns(raw)
        else:
            frags = [raw]
        for frag in frags:
            if instruction_filter != "off" and not instruction_filter_keeps_chatml(
                frag, instruction_filter
            ):
                continue
            lens.append(len(tok.encode(frag, add_special_tokens=False)))
    if not lens:
        return {
            "single_turn_expand": bool(single_turn_expand),
            "instruction_filter": instruction_filter,
            "n_examples": 0,
            "error": "no_examples_or_empty_rows",
        }
    arr = np.asarray(lens, dtype=np.int64)
    need = int(np.sum(arr > seq_len + 1))
    short = int(np.sum(arr < 2))
    return {
        "single_turn_expand": bool(single_turn_expand),
        "instruction_filter": instruction_filter,
        "n_examples": int(len(lens)),
        "token_len_min": int(arr.min()) if len(arr) else 0,
        "token_len_max": int(arr.max()) if len(arr) else 0,
        "token_len_mean": float(arr.mean()) if len(arr) else 0.0,
        "p50": float(np.percentile(arr, 50)) if len(arr) else 0.0,
        "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
        "p95": float(np.percentile(arr, 95)) if len(arr) else 0.0,
        "p99": float(np.percentile(arr, 99)) if len(arr) else 0.0,
        "seq_len_setting": int(seq_len),
        "n_examples_truncated_len_gt_seq_plus_1": need,
        "truncation_rate": float(need / max(1, len(lens))),
        "n_trivially_short": short,
        "recommendation": (
            f"尾端常含關閉符，截斷率 {100 * need / max(1, len(lens)):.1f}% 時不建議用此 SEQ_LEN 長期訓練。"
            f" 若 p90/p95 > {seq_len + 1}，可先試 SEQ_LEN=1024 實測，再升 2048 覆蓋多數樣本。"
        ),
    }


def _split_train_val_indices(
    n: int, val_frac: float, seed: int
) -> tuple[list[int], list[int]]:
    if n <= 1 or val_frac <= 0.0:
        return list(range(n)), []
    n_val = max(1, int(round(n * val_frac)))
    n_val = min(n_val, n - 1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def _migrate_sft_train_csv_add_eos_prob_max(path: str) -> bool:
    """
    在既有訓練 CSV 的 `eos_prob` 欄後插入 `eos_prob_max`（舊列填空白）。
    成功改寫回傳 True。
    """
    if not os.path.isfile(path) or os.stat(path).st_size == 0:
        return False
    with open(path, "r", newline="", encoding="utf-8") as rf:
        rows = list(csv.reader(rf))
    if not rows:
        return False
    hdr = rows[0]
    if "eos_prob_max" in hdr:
        return False
    try:
        ei = hdr.index("eos_prob")
    except ValueError:
        return False
    new_hdr = hdr[: ei + 1] + ["eos_prob_max"] + hdr[ei + 1 :]
    new_rows: list[list[str]] = [new_hdr]
    for r in rows[1:]:
        if len(r) <= ei:
            new_rows.append(r)
            continue
        new_rows.append(r[: ei + 1] + [""] + r[ei + 1 :])
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as wf:
        csv.writer(wf).writerows(new_rows)
    os.replace(tmp, path)
    return True


def _migrate_sft_train_csv_add_final_enhance(path: str) -> bool:
    """
    在既有訓練 CSV 的 `scale_w` 欄後插入 `pdl_weight_mean`, `ie_weight_mean`
    （舊列填空白）。成功改寫回傳 True。需在 add_eos_prob_max 之後呼叫。
    """
    if not os.path.isfile(path) or os.stat(path).st_size == 0:
        return False
    with open(path, "r", newline="", encoding="utf-8") as rf:
        rows = list(csv.reader(rf))
    if not rows:
        return False
    hdr = rows[0]
    if "pdl_weight_mean" in hdr and "ie_weight_mean" in hdr:
        return False
    try:
        si = hdr.index("scale_w")
    except ValueError:
        return False
    new_cols = ["pdl_weight_mean", "ie_weight_mean"]
    new_hdr = hdr[: si + 1] + new_cols + hdr[si + 1 :]
    new_rows: list[list[str]] = [new_hdr]
    for r in rows[1:]:
        if len(r) <= si:
            new_rows.append(r)
            continue
        new_rows.append(r[: si + 1] + ["", ""] + r[si + 1 :])
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as wf:
        csv.writer(wf).writerows(new_rows)
    os.replace(tmp, path)
    return True


_VAL_CSV_HEADER_LEGACY_4 = ["step", "val_ce_loss", "val_loss_mean", "val_batches"]
_VAL_CSV_HEADER_FULL = [
    "step",
    "val_ce_loss",
    "val_loss_mean",
    "val_fcp_penalty",
    "val_eos_prob",
    "val_eos_prob_max",
    "val_batches",
    "val_pdl_weight_mean",
    "val_ie_weight_mean",
]


def _migrate_sft_val_csv_fcp_columns(path: str) -> bool:
    """
    將僅 4 欄的 val CSV 擴成含 FCP / P(EOS) / P(EOS)_max + Final-Enhance（舊列新欄留空）。
    """
    if not os.path.isfile(path) or os.stat(path).st_size == 0:
        return False
    with open(path, "r", newline="", encoding="utf-8") as rf:
        rows = list(csv.reader(rf))
    if not rows:
        return False
    hdr = rows[0]
    if "val_fcp_penalty" in hdr:
        return False
    if hdr != _VAL_CSV_HEADER_LEGACY_4:
        return False
    new_rows: list[list[str]] = [_VAL_CSV_HEADER_FULL]
    for r in rows[1:]:
        if len(r) >= 4:
            # legacy: step, ce, loss, batches → insert 3 FCP blanks + 2 FE blanks
            new_rows.append([r[0], r[1], r[2], "", "", "", r[3], "", ""])
        else:
            new_rows.append(r)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as wf:
        csv.writer(wf).writerows(new_rows)
    os.replace(tmp, path)
    return True


def _migrate_sft_val_csv_add_final_enhance(path: str) -> bool:
    """
    在既有 7 欄 val CSV（含 FCP）尾端追加 val_pdl_weight_mean, val_ie_weight_mean
    （舊列新欄留空）。需在 fcp_columns 之後呼叫。
    """
    if not os.path.isfile(path) or os.stat(path).st_size == 0:
        return False
    with open(path, "r", newline="", encoding="utf-8") as rf:
        rows = list(csv.reader(rf))
    if not rows:
        return False
    hdr = rows[0]
    if "val_pdl_weight_mean" in hdr and "val_ie_weight_mean" in hdr:
        return False
    if "val_fcp_penalty" not in hdr:
        return False  # not an FCP-era file; fcp migrate handles it
    new_rows: list[list[str]] = [hdr + ["val_pdl_weight_mean", "val_ie_weight_mean"]]
    for r in rows[1:]:
        new_rows.append(r + ["", ""])
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as wf:
        csv.writer(wf).writerows(new_rows)
    os.replace(tmp, path)
    return True


@torch.no_grad()
def _sft_eval_batches(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int,
    global_step: int,
    amp_dtype: torch.dtype,
) -> dict | None:
    """同訓練：labels 含 -100；CE 的 ignore_index 預設 -100 與 train 一致。"""
    model.eval()
    total_ce = 0.0
    total_loss = 0.0
    total_fcp = 0.0
    total_eos = 0.0
    total_eos_max = 0.0
    total_pdl = 0.0
    total_ie = 0.0
    n_batches = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        # MaterializedSftDataset now yields 3-tuples (x, y, sw). For val we
        # compute *unweighted* CE (matches old behaviour) by passing
        # `structure_weights=None`.
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            xb, yb, _sw = batch
        else:
            xb, yb = batch
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True, dtype=torch.long)
        if (yb != -100).sum().item() == 0:
            continue
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model(xb, labels=yb, step=global_step)
        ce = out[2].item() if isinstance(out[2], torch.Tensor) else float(out[2])
        loss_mean = out[0].mean().item()
        total_ce += ce
        total_loss += loss_mean
        if len(out) >= 8:
            total_fcp += out[5].item() if isinstance(out[5], torch.Tensor) else float(out[5])
            total_eos += out[6].item() if isinstance(out[6], torch.Tensor) else float(out[6])
            total_eos_max += out[7].item() if isinstance(out[7], torch.Tensor) else float(out[7])
        if len(out) >= 10:
            total_pdl += out[8].item() if isinstance(out[8], torch.Tensor) else float(out[8])
            total_ie += out[9].item() if isinstance(out[9], torch.Tensor) else float(out[9])
        n_batches += 1
    model.train()
    if n_batches == 0:
        return None
    return {
        "val_ce_loss": total_ce / n_batches,
        "val_loss_mean": total_loss / n_batches,
        "val_fcp_penalty": total_fcp / n_batches,
        "val_eos_prob": total_eos / n_batches,
        "val_eos_prob_max": total_eos_max / n_batches,
        "val_batches": n_batches,
        "val_pdl_weight_mean": total_pdl / n_batches,
        "val_ie_weight_mean": total_ie / n_batches,
    }


def _next_token_from_logits(
    logits_last: torch.Tensor,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    repetition_penalty: float,
    generated_ids: list[int],
    gen: torch.Generator,
) -> int:
    """
    logits_last: [V] 浮點；
    generated_ids: 已經生成的 token id 列表 (用於計算重複懲罰)
    """
    x = logits_last.float()

    # 1. 執行 Repetition Penalty
    if repetition_penalty != 1.0 and generated_ids:
        unique_ids = list(set(generated_ids))
        idx_tensor = torch.tensor(unique_ids, dtype=torch.long, device=x.device)
        score = torch.gather(x, 0, idx_tensor)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        x.scatter_(0, idx_tensor, score)

    # 2. Temperature (<=0 為 greedy)
    if temperature is None or temperature <= 0.0:
        return int(x.argmax().item())

    x = x / max(float(temperature), 1e-8)

    # 3. Top-k
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(x, min(top_k, x.size(-1)))
        x[x < v[-1]] = -float("Inf")

    probs = F.softmax(x, dim=-1)

    # 4. Top-p
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        s_probs, s_idx = torch.sort(probs, descending=True, dim=0)
        c = torch.cumsum(s_probs, dim=0)
        rm = c > float(top_p)
        if int(rm.numel()) > 1:
            rm[1:] = rm[:-1].clone()
        rm[0] = False
        s_masked = s_probs * (~rm).to(s_probs.dtype)
        p2 = torch.zeros_like(probs)
        p2[s_idx] = s_masked
        ssum = p2.sum()
        if ssum > 0:
            probs = p2 / ssum
    p_cpu = probs.detach().cpu()
    return int(torch.multinomial(p_cpu, 1, generator=gen).item())


@torch.no_grad()
def _sft_sample_test_reply(
    model: nn.Module,
    tok: object,
    user_turn: str,
    device: torch.device,
    global_step: int,
    max_new: int,
    seq_len_cap: int,
    amp_dtype: torch.dtype,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = 50,
    repetition_penalty: float = 1.15,
    sample_seed: int = 0,
) -> str:
    """
    與訓練相同 _format_conversation 前綴，自最後位置續寫；一產生 **EOS（</s>）** 或 **<|im_end|>** 即停，或達 max_new 上限。
    temperature=0 為 greedy；>0 為 multinomial，可併用 top_p nucleus（0~1）。
    """
    text0 = _format_conversation([user_turn.strip()])
    ids: list[int] = tok.encode(text0, add_special_tokens=False)
    if not ids:
        return ""
    n0 = len(ids)
    eos_id = int(tok.eos_token_id) if getattr(tok, "eos_token_id", None) is not None else 2
    stop_ids: set[int] = {eos_id}
    if len(SPECIAL_TOKENS) > 1:
        try:
            raw = tok.convert_tokens_to_ids(SPECIAL_TOKENS[1])
            _ie = int(raw[0] if isinstance(raw, (list, tuple)) and raw else raw)
            if _ie >= 0:
                stop_ids.add(_ie)
        except (TypeError, ValueError, IndexError):
            pass
    out: list[int] = list(ids)
    m = unwrap_model(model)
    was_training = model.training
    model.eval()
    gen = torch.Generator()
    gen.manual_seed(int(sample_seed) + int(global_step) * 1_000_003)
    for _ in range(max_new):
        use = out[-seq_len_cap:] if len(out) > seq_len_cap else out
        t = torch.tensor([use], device=device, dtype=torch.long)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = m(t, labels=None, step=global_step)
        else:
            logits = m(t, labels=None, step=global_step)
        last = logits[0, -1, :]
        nxt = _next_token_from_logits(
            logits_last=last,
            temperature=temperature,
            top_p=top_p,
            top_k=(50 if top_k is None else top_k),
            repetition_penalty=float(repetition_penalty),
            generated_ids=out,
            gen=gen,
        )
        out.append(nxt)
        if nxt in stop_ids:
            break
    if was_training:
        model.train()
    new_ids = out[n0:]
    return tok.decode(new_ids, skip_special_tokens=False) if new_ids else ""


def train_sft(
    D_MODEL=768,
    D_STATE=64,
    D_HEAD=64,
    EXPAND=2,
    NUM_LAYERS=6,
    MIMO_RANK=4,
    NUM_KV_HEADS=4,
    CHUNK_SIZE=64,
    KMOE_NUM_EXPERTS=8,
    KMOE_TOP_K=2,
    KMOE_R1=4,
    KMOE_R2=1024,
    KMOE_R3=256,
    FFN_EXPAND=6,
    # 用於 validate_config（路徑必須存在）
    DATA_PATH="dataset/mix_a25_u75_ins.bin",
    # hf_text：見 build_mix_* / alpaca_parquet_to_bin；lima：dataset/lima_hf
    LIMA_HF="dataset/mix_a25_u75_ins_hf",
    # "lima" = GAIR/lima 對話欄位；"hf_text" = 每列有「text」（Alpaca / CoT 等）
    SFT_DATA_SOURCE: str = "hf_text",
    SFT_TEXT_COLUMN: str = "text",
    TOKENIZER_DIR="dataset/tokenizer",
    OUTPUT_DIR="output/",
    LOG_FILE="output/train_sft_log.csv",
    # 預設輕量權重；要含 optimizer/scheduler 續訓可改 output/checkpoint.pt
    CHECKPOINT_LOAD="dataset/pre_train_base_model.pt",
    # True：若檔為 sft 整包，仍只取 model、不續 step/optimizer/scheduler（新開一輪 SFT，自 step=0）
    SFT_INIT_WEIGHTS_ONLY: bool = False,
    # 實際寫出 output/checkpoint_sft_s100.pt、_s200.pt、…（每 SAVE_EVERY_STEPS 儲存）
    CHECKPOINT_SAVE="output/checkpoint_sft.pt",
    SAVE_EVERY_STEPS: int = 30,
    PRETRAINED_EMBED_PATH="",
    VOCAB_SIZE=32007,
    # 預設 512；上調 1024/2048 可減少尾端截斷（見 sft_*_len_stats）
    # Effective batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * n_gpu
    SEQ_LEN=512,
    BATCH_SIZE=4,
    GRADIENT_ACCUMULATION_STEPS=8,
    LR=1e-5,
    # 實際 warmup = min(ceil(新步數 * WARMUP_FRAC), WARMUP, 新步數-1)，避免短程 SFT 幾乎全程在爬升 LR
    WARMUP_FRAC: float = 0.08,
    WARMUP=200,
    WEIGHT_DECAY: float = 0.1,
    MAX_GRAD_NORM: float = 1.0,
    # EPOCHS 不為 None 時：STEPS = start_step + EPOCHS * steps_per_epoch
    EPOCHS: int | None = 3,
    STEPS_MAX=100_000,
    DATALOADER_WORKERS=2,
    DATALOADER_PIN_MEMORY=None,
    DATALOADER_PREFETCH_FACTOR=2,
    DATALOADER_PERSISTENT_WORKERS=True,
    ROUTER_T_START=2.0,
    ROUTER_T_END=0.5,
    # 與 train.py 預訓練退火長度相同（僅 SFT_FIXED_ROUTER_T 為 None 時生效）
    ROUTER_WARMUP=500,
    ROUTER_TOTAL=10_000,
    # 預訓練收斂後 T_router=ROUTER_T_END（0.5）；SFT 若從 step=0 續訓，預設**固定**該溫度，不會再從 2.0 暖起。設為 None 則依上列重新退火。
    SFT_FIXED_ROUTER_T: float | None = 0.5,
    TRAIN_MODE=True,
    COMPILE_ENABLED=False,
    COMPILE_MODE="default",
    COMPILE_FULLGRAPH=False,
    
    GRAD_CHECK_INTERVAL=50,
    
    RESUME_FAST_SKIP_DATALOADER=True,
    # 1 = 每個 optimizer step 印訓練行；>1 則每 N 步印一次
    PRINT_EVERY_STEPS: int = 1,
    # 與訓練相同 mask 的 val CE（從 train 劃一筆子集，非 train.py 的 bin 尾段）
    VAL_ENABLED=True,
    VAL_FRAC=0.05,
    VAL_EVERY_STEPS=100,
    VAL_MAX_BATCHES=32,
    VAL_LOG_FILE="output/val_sft_log.csv",
    VAL_SEED=42,
    # 非空字串則週期做測試解碼（前綴與訓練相同）；None/"" 關閉
    SFT_TEST_PROMPT: str | None = None,
    # None 表示與 val 相同間隔 val_every_eff；有設數字則 min(該值, train_len//2) 同 val
    SFT_TEST_EVERY_STEPS: int | None = None,
    SFT_TEST_MAX_NEW: int = 64,
    SFT_TEST_LOG_FILE: str = "output/sft_test_gen.log",
    # 0.0 = greedy；建議 0.7～0.9 搭配 top_p=0.9
    SFT_TEST_TEMPERATURE: float = 0.0,
    SFT_TEST_TOP_P: float | None = None,
    SFT_TEST_TOP_K: int | None = None,
    SFT_TEST_REPETITION_PENALTY: float = 1.0,
    SFT_TEST_SAMPLE_SEED: int = 42,
    # True：若 output 下已有長度統計檔，直接讀取重用，避免每次啟動重掃全資料（可大幅加速啟動）
    REUSE_LEN_STATS_IF_EXISTS: bool = True,
    # True：多輪 [u1,a1,u2,a2,...] 拆成多筆單輪樣本；訓練列數／每 epoch steps 會變多（通常對小模型更穩）。
    SFT_SINGLE_TURN_EXPAND: bool = False,
    # user 區塊過濾：drop_weak=刪問候／Tell me something；ins_strict=偏 What/Why/How/Explain／Alpaca ### Instruction
    SFT_INSTRUCTION_FILTER: str = "off",
    # 對結構關閉符（</think>、</final>、<|im_end|>）的 label 位置 CE 加權；≤1 關閉。環境變數 SFT_STRUCTURE_TOKEN_CE_MULT 可覆寫。
    SFT_STRUCTURE_TOKEN_CE_MULT: float = 8.0,
    # ---- FCP + SFT-GO + SCALe explicit params (None → fall back to env var / built-in default) ----
    # Set to True/False to override env vars SFT_USE_SFTGO / SFT_USE_FCP / SFT_USE_SCALE.
    ENABLE_SFTGO: Optional[bool] = None,
    ENABLE_FCP: Optional[bool] = None,
    ENABLE_SCALE: Optional[bool] = None,
    # Path to the structure-weight bundle (.pt); None → env var SFT_STRUCT_BUNDLE_PATH.
    STRUCT_BUNDLE_PATH: Optional[str] = None,
    # FCP hyper-params; None → read from env vars SFT_FCP_LAMBDA / SFT_FCP_DELTA.
    FCP_LAMBDA: Optional[float] = None,
    FCP_DELTA: Optional[float] = None,
    # SCALe cosine-annealing range; None → env vars SFT_SCALE_ETA_MAX / SFT_SCALE_ETA_MIN.
    SCALE_ETA_MAX: Optional[float] = None,
    SCALE_ETA_MIN: Optional[float] = None,
    SCALE_W_FINAL: Optional[float] = None,
    # ---- Final Enhance: Final-SW + PDL + InfoEntropy (default ON, no trigger/warmup) ----
    # Master + per-component toggles; env vars SFT_COT_ENABLE_* override these.
    ENABLE_FINAL_ENHANCE: bool = True,
    ENABLE_FINAL_SW: bool = True,
    ENABLE_PDL: bool = True,
    ENABLE_IE: bool = True,
    FINAL_SW_ETA: float = 0.1,
    FINAL_SW_LAMBDA: float = 0.8,
    PDL_ALPHA: float = 0.4,
    PDL_WEIGHTS_PATH: str = "cot_task/reports/pdl_weights.pt",
    IE_GAMMA: float = 0.5,
    IE_ON_THINK: bool = False,
):
    # 將相對路徑固定錨到專案根目錄，避免從不同 cwd 啟動時重複重算 len stats。
    def _to_project_path(p: str) -> str:
        pp = Path(p)
        return str(pp if pp.is_absolute() else (PROJECT_ROOT / pp))

    OUTPUT_DIR = _to_project_path(OUTPUT_DIR)
    LOG_FILE = _to_project_path(LOG_FILE)
    VAL_LOG_FILE = _to_project_path(VAL_LOG_FILE)
    SFT_TEST_LOG_FILE = _to_project_path(SFT_TEST_LOG_FILE)

    inst_filter_mode = normalize_instruction_filter_mode(SFT_INSTRUCTION_FILTER)

    if SFT_DATA_SOURCE not in ("lima", "hf_text"):
        raise ValueError("SFT_DATA_SOURCE 需為 'lima' 或 'hf_text'")
    for p, name in ((LIMA_HF, "LIMA_HF"), (TOKENIZER_DIR, "TOKENIZER_DIR")):
        if not os.path.isdir(p):
            _hint = (
                "需先執行 lima_to_bin"
                if SFT_DATA_SOURCE == "lima"
                else "需先執行 alpaca_parquet_to_bin.py（或 sft_cot_to_bin）產生 HF 快照目錄"
            )
            raise FileNotFoundError(f"{name} 不存在: {p}（{_hint}）")
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"validate / 長度用 DATA_PATH 不存在: {DATA_PATH}")

    # 盡早建立 Accelerator，才能用 is_main_process 避免多行程 / accelerate 重複印出整段 log
    accelerator = Accelerator(
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )

    if accelerator.is_main_process:
        n_bin = 0
        if os.path.isfile(DATA_PATH):
            n_bin = len(np.memmap(DATA_PATH, dtype="uint16", mode="r"))
        n_hf = 0
        if os.path.isdir(LIMA_HF):
            n_hf = len(_load_hf_train_split(LIMA_HF))
        _src_lbl = "lima train" if SFT_DATA_SOURCE == "lima" else "hf_text（text 欄）"
        _st_stem = "sft_lima_len_stats" if SFT_DATA_SOURCE == "lima" else "sft_hf_text_len_stats"
        _st_stem = f"{_st_stem}_single" if SFT_SINGLE_TURN_EXPAND else _st_stem
        if inst_filter_mode == "drop_weak":
            _st_stem = f"{_st_stem}_noweak"
        elif inst_filter_mode == "ins_strict":
            _st_stem = f"{_st_stem}_instr"
        _gran = "單輪 user→assistant" if SFT_SINGLE_TURN_EXPAND else "每 HF 列一則對話"
        _ifmsg = (
            f"；user 過濾={inst_filter_mode!r}"
            if inst_filter_mode != "off"
            else ""
        )
        print(
            f"[SFT] data .bin 串流長度 = {n_bin:_} | {_src_lbl} 列數 = {n_hf} | 詞表 = {VOCAB_SIZE}\n"
            f"      ※ 粒度：{_gran}{_ifmsg}；不做多列合併/packing；長句僅**前向截斷**（啟動統計→ {_st_stem}.json）\n",
            flush=True,
        )

    if accelerator.is_main_process:
        _stem = "sft_lima_len_stats" if SFT_DATA_SOURCE == "lima" else "sft_hf_text_len_stats"
        if SFT_SINGLE_TURN_EXPAND:
            _stem = f"{_stem}_single"
        if inst_filter_mode == "drop_weak":
            _stem = f"{_stem}_noweak"
        elif inst_filter_mode == "ins_strict":
            _stem = f"{_stem}_instr"
        _stats_name = f"{_stem}.json"
        stats_path = os.path.join(OUTPUT_DIR, _stats_name)
        st = None
        if REUSE_LEN_STATS_IF_EXISTS and os.path.isfile(stats_path):
            try:
                with open(stats_path, "r", encoding="utf-8") as fp:
                    st = json.load(fp)
                if isinstance(st, dict):
                    mism = []
                    if bool(st.get("single_turn_expand", False)) != bool(
                        SFT_SINGLE_TURN_EXPAND
                    ):
                        mism.append("single_turn_expand")
                    if str(st.get("instruction_filter", "off")) != inst_filter_mode:
                        mism.append("instruction_filter")
                    if mism:
                        print(
                            f"      [len] 統計檔與目前設定不符（{', '.join(mism)}），將重算：{stats_path}"
                        )
                        st = None
                    else:
                        print(f"      [len] 重用既有統計：{stats_path}")
                else:
                    st = None
            except (OSError, json.JSONDecodeError) as e:
                print(f"      [len] 讀取既有統計失敗，改為重算：{e}")
                st = None
        if st is None:
            if SFT_DATA_SOURCE == "hf_text":
                st = _hf_text_token_len_stats(
                    LIMA_HF,
                    TOKENIZER_DIR,
                    SEQ_LEN,
                    text_column=SFT_TEXT_COLUMN,
                    single_turn_expand=SFT_SINGLE_TURN_EXPAND,
                    instruction_filter=inst_filter_mode,
                )
            else:
                st = _lima_token_len_stats(
                    LIMA_HF,
                    TOKENIZER_DIR,
                    SEQ_LEN,
                    single_turn_expand=SFT_SINGLE_TURN_EXPAND,
                    instruction_filter=inst_filter_mode,
                )
            try:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                with open(stats_path, "w", encoding="utf-8") as fp:
                    json.dump(st, fp, ensure_ascii=False, indent=2)
                print(f"      已寫入 {stats_path}")
            except OSError as e:
                print(f"      無法寫入 {stats_path}: {e}")
        for k, v in st.items():
            print(f"      [len] {k}: {v}")
        print("")

    fb = {
        "D_MODEL": D_MODEL,
        "D_STATE": D_STATE,
        "D_HEAD": D_HEAD,
        "EXPAND": EXPAND,
        "NUM_LAYERS": NUM_LAYERS,
        "CHUNK_SIZE": CHUNK_SIZE,
        "KMOE_NUM_EXPERTS": KMOE_NUM_EXPERTS,
        "KMOE_TOP_K": KMOE_TOP_K,
        "KMOE_R1": KMOE_R1,
        "KMOE_R2": KMOE_R2,
        "KMOE_R3": KMOE_R3,
        "FFN_EXPAND": FFN_EXPAND,
        "MIMO_RANK": MIMO_RANK,
        "NUM_KV_HEADS": NUM_KV_HEADS,
        "ROUTER_T_START": ROUTER_T_START,
        "ROUTER_T_END": ROUTER_T_END,
        "ROUTER_WARMUP": ROUTER_WARMUP,
        "ROUTER_TOTAL": ROUTER_TOTAL,
    }
    ckpt = None
    if not os.path.isfile(CHECKPOINT_LOAD):
        raise FileNotFoundError(
            f"找不到權重：{CHECKPOINT_LOAD!r}（可跑 export_checkpoint_weights.py 產生 dataset/pre_train_base_model.pt；"
                f"續跑 SFT 請指定含 sft: True 的 output/checkpoint_sft_sK.pt 等）"
        )
    ckpt = _load_pretrain_file(CHECKPOINT_LOAD)
    _file_step = int(ckpt["step"])
    sft_resume = bool(ckpt.get("sft")) and ckpt.get("kind") == "full"
    if SFT_INIT_WEIGHTS_ONLY:
        sft_resume = False
    if sft_resume:
        start_step = _file_step
    else:
        start_step = 0
    config = _mamba3_config_from_ckpt(ckpt.get("config"), fb)
    if SFT_FIXED_ROUTER_T is not None:
        # 不論新開或續跑，只要指定固定值就強制覆寫 router 溫度設定。
        v = float(SFT_FIXED_ROUTER_T)
        router_warmup = 0
        router_total = 1
        router_t_start = v
        router_t_end = v
    else:
        if not sft_resume:
            config.router_warmup = int(ROUTER_WARMUP)
            config.router_total = int(ROUTER_TOTAL)
            config.router_t_start = float(ROUTER_T_START)
            config.router_t_end = float(ROUTER_T_END)
        router_warmup = int(getattr(config, "router_warmup", ROUTER_WARMUP))
        router_total = int(getattr(config, "router_total", ROUTER_TOTAL))
        router_t_start = float(getattr(config, "router_t_start", ROUTER_T_START))
        router_t_end = float(getattr(config, "router_t_end", ROUTER_T_END))

    if accelerator.is_main_process and not sft_resume:
        if SFT_FIXED_ROUTER_T is not None:
            print(
                f"   🌡️  Router 溫度**固定** T={SFT_FIXED_ROUTER_T!r}（與預訓練 T_end=ROUTER_T_END 一致，"
                f" 不從 {ROUTER_T_START} 重新暖起前 500 step）"
            )
        else:
            print(
                f"   🌡️  Router 退火：{ROUTER_T_START} → {ROUTER_T_END}，"
                f"warmup={ROUTER_WARMUP}，total={ROUTER_TOTAL}"
            )
    if accelerator.is_main_process:
        if SFT_INIT_WEIGHTS_ONLY and _file_step > 0:
            print(
                f"ℹ️  SFT_INIT_WEIGHTS_ONLY：僅載入 {CHECKPOINT_LOAD} 的 **model**，"
                f"檔內 step={_file_step} 不沿用 — SFT **從 step=0** 起（新 optimizer/scheduler）。\n"
                f"   Mamba3Config 自檔內 config 還原。"
            )
        elif SFT_INIT_WEIGHTS_ONLY and _file_step == 0:
            print(
                "ℹ️  SFT_INIT_WEIGHTS_ONLY：僅載入權重；Mamba3Config 自檔內 config 還原。"
            )
        if sft_resume:
            print(
                f"ℹ️  SFT 斷點續跑 {CHECKPOINT_LOAD}（sft=True，從 step={start_step}）"
                f"；Mamba3Config 自檔內 config 還原。LR/scheduler 沿用斷點，不從 0/rewarm 重排。"
            )
        elif not SFT_INIT_WEIGHTS_ONLY:
            ign = f" 檔內預訓 step={_file_step} 不計入 SFT" if _file_step else ""
            if ckpt.get("kind") == "weights_only":
                print(
                    f"ℹ️  僅權重 {CHECKPOINT_LOAD}（state_dict）；SFT 從 **step=0** 起算。{ign}"
                )
            else:
                print(
                    f"ℹ️  完整權重 {CHECKPOINT_LOAD}（非 SFT 斷點，無 sft: True）；SFT 從 **step=0** 起算。{ign}"
                )
            print("   Mamba3Config 自檔內 config 還原。")

    hf_table = _load_hf_train_split(LIMA_HF)
    n_rows = len(hf_table)
    if VAL_ENABLED:
        tr_idx, va_idx = _split_train_val_indices(n_rows, VAL_FRAC, VAL_SEED)
    else:
        tr_idx, va_idx = list(range(n_rows)), []
    if SFT_DATA_SOURCE == "lima":
        tr_materialized, tr_inst_meta = materialize_lima_hf_examples(
            hf_table, tr_idx, SFT_SINGLE_TURN_EXPAND, inst_filter_mode
        )
        if va_idx:
            va_materialized, va_inst_meta = materialize_lima_hf_examples(
                hf_table, va_idx, SFT_SINGLE_TURN_EXPAND, inst_filter_mode
            )
        else:
            va_materialized, va_inst_meta = [], {}
    else:
        tr_materialized, tr_inst_meta = materialize_hf_text_examples(
            hf_table, tr_idx, SFT_TEXT_COLUMN, SFT_SINGLE_TURN_EXPAND, inst_filter_mode
        )
        if va_idx:
            va_materialized, va_inst_meta = materialize_hf_text_examples(
                hf_table, va_idx, SFT_TEXT_COLUMN, SFT_SINGLE_TURN_EXPAND, inst_filter_mode
            )
        else:
            va_materialized, va_inst_meta = [], {}
    n_train = len(tr_materialized)
    if n_train <= 0:
        raise ValueError(
            "展開後訓練樣本數為 0：請檢查 HF 資料、欄位，或關閉 SFT_SINGLE_TURN_EXPAND / 放寬 SFT_INSTRUCTION_FILTER。"
        )
    # DDP：accelerate.prepare 會用 DistributedSampler，每 epoch 每卡只見 n_train/n_gpu 筆；
    # 有效 batch = BATCH × grad_accum × n_gpu，故 steps/epoch 須依此計算（勿用單卡公式×2）。
    n_gpus = max(1, int(accelerator.num_processes))
    eff_batch = int(BATCH_SIZE) * int(GRADIENT_ACCUMULATION_STEPS) * n_gpus
    steps_per_epoch = max(1, (n_train + eff_batch - 1) // eff_batch)
    n_batches_one_epoch = steps_per_epoch * GRADIENT_ACCUMULATION_STEPS  # 供 log 對照
    if EPOCHS is not None:
        STEPS = start_step + int(EPOCHS) * steps_per_epoch
    else:
        STEPS = int(STEPS_MAX)
    if STEPS <= start_step:
        raise ValueError(
            f"沒有訓練步可跑：STEPS={STEPS} start_step={start_step}，請調 EPOCHS 或 start_step / checkpoint。"
        )
    train_len = STEPS - start_step
    if train_len <= 0:
        raise ValueError("train_len=STEPS-start_step 必須 > 0")
    # 預設為新步數的 8% 左右（5%~10% 區間），再與參數 WARMUP（上限步數）取 min，避免短程 SFT 近全程 warmup
    if train_len == 1:
        warmup_eff = 0
    else:
        cap = train_len - 1
        from_pct = max(1, int(round(train_len * float(WARMUP_FRAC))))
        warmup_eff = int(min(from_pct, WARMUP, cap))
        warmup_eff = max(0, warmup_eff)
    val_every_eff = min(VAL_EVERY_STEPS, max(1, train_len // 2))
    test_every_eff = (
        val_every_eff
        if SFT_TEST_EVERY_STEPS is None
        else min(int(SFT_TEST_EVERY_STEPS), max(1, train_len // 2))
    )

    if accelerator.is_main_process:
        if inst_filter_mode != "off":
            print(
                f"      🧹 指令過濾 `{inst_filter_mode}`："
                f"train 候選={tr_inst_meta.get('n_instruction_candidates', 0):_} "
                f"保留={tr_inst_meta.get('n_kept', 0):_} "
                f"丟棄={tr_inst_meta.get('n_instruction_dropped', 0):_} | "
                f"val 候選={va_inst_meta.get('n_instruction_candidates', 0)} "
                f"保留={va_inst_meta.get('n_kept', 0)} "
                f"丟棄={va_inst_meta.get('n_instruction_dropped', 0)}",
                flush=True,
            )
        if VAL_ENABLED and len(va_idx) > 0:
            print(
                f"[SFT] train/val: HF 列 train={len(tr_idx)} val={len(va_idx)} | "
                f"展開後 samples train={len(tr_materialized)} val={len(va_materialized)}\n"
                f"           有效 batch={eff_batch} ({BATCH_SIZE}×{GRADIENT_ACCUMULATION_STEPS}×{n_gpus} GPU) "
                f"→ optimizer steps/epoch ≈ {steps_per_epoch}"
            )
        else:
            print(
                f"[SFT] 訓練 HF 列={len(tr_idx)} 展開後 samples={len(tr_materialized)}，無 val 子集。"
            )
        if EPOCHS is not None:
            print(
                f"📅 依 EPOCHS={EPOCHS}：從 start_step={start_step} 訓到 STEPS={STEPS} "
                f"（新步數 {train_len}，WARMUP={warmup_eff} 步 ≈{WARMUP_FRAC * 100:.0f}% 新步、上限 {WARMUP}）"
            )
        else:
            print(
                f"📅 依 STEPS_MAX：從 {start_step} 訓到 {STEPS}，WARMUP={warmup_eff} "
                f"（≈{WARMUP_FRAC * 100:.0f}% 新步、上限 {WARMUP}）"
            )
        if int(SAVE_EVERY_STEPS) > 0:
            cks, cke = os.path.splitext(CHECKPOINT_SAVE)
            cke = cke or ".pt"
            print(
                f"💾 每 {int(SAVE_EVERY_STEPS)} step 存 checkpoint 一次 → "
                f"{os.path.basename(cks)}_s{int(SAVE_EVERY_STEPS)}{cke}、"
                f"{os.path.basename(cks)}_s{2 * int(SAVE_EVERY_STEPS)}{cke}、…"
            )
        if SFT_TEST_PROMPT and str(SFT_TEST_PROMPT).strip():
            ppr = str(SFT_TEST_PROMPT).replace("\n", " ")
            if len(ppr) > 100:
                ppr = ppr[:100] + "…"
            _dec = (
                f"greedy T=0"
                if (SFT_TEST_TEMPERATURE is None or float(SFT_TEST_TEMPERATURE) <= 0)
                else (
                    f"sample T={SFT_TEST_TEMPERATURE!r} top_p={SFT_TEST_TOP_P!r} "
                    f"top_k={SFT_TEST_TOP_K!r} rep_pen={SFT_TEST_REPETITION_PENALTY!r}"
                )
            )
            print(
                f"🧪 SFT 測試解碼：每 {test_every_eff} 步（{_dec}、max_new={SFT_TEST_MAX_NEW}、seed+step）→ {SFT_TEST_LOG_FILE}\n"
                f"   prompt: {ppr!r}\n"
                f"   VAL 間隔={val_every_eff}；PRINT_EVERY_STEPS={PRINT_EVERY_STEPS}",
                flush=True,
            )

    if accelerator.is_main_process:
        validate_config(
            D_MODEL=D_MODEL,
            D_STATE=D_STATE,
            D_HEAD=D_HEAD,
            EXPAND=EXPAND,
            NUM_LAYERS=NUM_LAYERS,
            MIMO_RANK=MIMO_RANK,
            NUM_KV_HEADS=NUM_KV_HEADS,
            KMOE_NUM_EXPERTS=KMOE_NUM_EXPERTS,
            KMOE_TOP_K=KMOE_TOP_K,
            CHUNK_SIZE=CHUNK_SIZE,
            SEQ_LEN=SEQ_LEN,
            BATCH_SIZE=BATCH_SIZE,
            VOCAB_SIZE=VOCAB_SIZE,
            DATA_PATH=DATA_PATH,
            OUTPUT_DIR=OUTPUT_DIR,
            PRETRAINED_EMBED_PATH=PRETRAINED_EMBED_PATH,
            LR=LR,
            STEPS=STEPS,
            WARMUP=warmup_eff,
        )
    accelerator.wait_for_everyone()

    model = Mamba3LanguageModel(config, VOCAB_SIZE)
    if accelerator.is_main_process:
        print_model_analysis(unwrap_model(model), config, VOCAB_SIZE)

    if TRAIN_MODE and COMPILE_ENABLED:
        from train import resolve_compile_settings

        mode, copt, _note = resolve_compile_settings(COMPILE_MODE)
        try:
            model = torch.compile(
                model, mode=mode, fullgraph=COMPILE_FULLGRAPH, options=copt
            )
        except Exception as e:
            accelerator.print(f"compile 失敗，改 eager: {e}")

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(
            k in name
            for k in ["U_expert", "U_in", "U_out", "core", "bias", "norm", "LayerScale"]
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LR,
        betas=(0.9, 0.95),
        fused=True,
    )
    # SFT 斷點續訓：已載入 optimizer/scheduler 的步進與學率，勿用 resume_step+rewarmup
    # （LambdaLR 的 last_epoch 比 global_step 小 1，下一個 step 若 resume_step==start_step
    #  會觸發 rewarmup，把 LR 乘 0.1 起，看起來像從 0 重排）。
    # 同時從斷點讀 sft_warmup_steps / sft_total_steps，讓 get_lr 曲線與存檔時一致。
    if sft_resume:
        _w = ckpt.get("sft_warmup_steps")
        _t = ckpt.get("sft_total_steps")
        w_s = int(_w) if _w is not None else int(warmup_eff)
        t_saved = int(_t) if _t is not None else int(STEPS)
        t_s = max(int(STEPS), t_saved)
        scheduler = get_lr_scheduler(
            optimizer, warmup_steps=w_s, total_steps=t_s, resume_step=0, rewarmup_steps=0
        )
    else:
        scheduler = get_lr_scheduler(
            optimizer, warmup_steps=warmup_eff, total_steps=STEPS, resume_step=start_step, rewarmup_steps=100
        )

    # =========================================================================
    # Task 2 (FCP + SFT-GO + SCALe) — config from env vars; default: all disabled
    # =========================================================================
    def _env_flag(name: str, default: bool = False) -> bool:
        v = os.environ.get(name, "").strip().lower()
        if not v:
            return default
        return v in ("1", "true", "yes", "on")

    def _env_float(name: str, default: float) -> float:
        v = os.environ.get(name, "").strip()
        try:
            return float(v) if v else default
        except ValueError:
            return default

    # Function params take priority; env vars are the fallback.
    SFT_USE_SFTGO = ENABLE_SFTGO if ENABLE_SFTGO is not None else _env_flag("SFT_USE_SFTGO", False)
    SFT_USE_FCP = ENABLE_FCP if ENABLE_FCP is not None else _env_flag("SFT_USE_FCP", False)
    SFT_USE_SCALE = ENABLE_SCALE if ENABLE_SCALE is not None else _env_flag("SFT_USE_SCALE", False)
    _sbp_env_default = str(
        Path(__file__).resolve().parent.parent / "cot_task" / "reports" / "structure_weights_bundle.pt"
    )
    SFT_STRUCT_BUNDLE_PATH = (
        STRUCT_BUNDLE_PATH if STRUCT_BUNDLE_PATH is not None
        else os.environ.get("SFT_STRUCT_BUNDLE_PATH", _sbp_env_default)
    )
    SFT_FCP_LAMBDA = FCP_LAMBDA if FCP_LAMBDA is not None else _env_float("SFT_FCP_LAMBDA", 0.2)
    SFT_FCP_DELTA = FCP_DELTA if FCP_DELTA is not None else _env_float("SFT_FCP_DELTA", 0.01)
    SFT_SCALE_ETA_MAX = SCALE_ETA_MAX if SCALE_ETA_MAX is not None else _env_float("SFT_SCALE_ETA_MAX", 1.0)
    SFT_SCALE_ETA_MIN = SCALE_ETA_MIN if SCALE_ETA_MIN is not None else _env_float("SFT_SCALE_ETA_MIN", 0.3)
    SFT_SCALE_W_FINAL = SCALE_W_FINAL if SCALE_W_FINAL is not None else _env_float("SFT_SCALE_W_FINAL", 1.0)

    # ---- Final Enhance (Final-SW + PDL + InfoEntropy) — default ON ----------
    def _env_flag_default(name: str, default: bool) -> bool:
        v = os.environ.get(name, "").strip().lower()
        if not v:
            return default
        return v in ("1", "true", "yes", "on")

    FE_MASTER = _env_flag_default("SFT_COT_ENABLE_FINAL_ENHANCE", bool(ENABLE_FINAL_ENHANCE))
    FE_FINAL_SW = FE_MASTER and _env_flag_default("SFT_COT_ENABLE_FINAL_SW", bool(ENABLE_FINAL_SW))
    FE_PDL = FE_MASTER and _env_flag_default("SFT_COT_ENABLE_PDL", bool(ENABLE_PDL))
    FE_IE = FE_MASTER and _env_flag_default("SFT_COT_ENABLE_IE", bool(ENABLE_IE))
    FE_SW_ETA = _env_float("SFT_COT_FINAL_SW_ETA", float(FINAL_SW_ETA))
    FE_SW_LAMBDA = _env_float("SFT_COT_FINAL_SW_LAMBDA", float(FINAL_SW_LAMBDA))
    FE_IE_GAMMA = _env_float("SFT_COT_IE_GAMMA", float(IE_GAMMA))
    FE_IE_ON_THINK = _env_flag_default("SFT_COT_IE_ON_THINK", bool(IE_ON_THINK))

    # Load static PDL weights [vocab_size]; auto-disable PDL if missing/mismatched.
    _pdl_weights: Optional[torch.Tensor] = None
    if FE_PDL:
        _pdl_path = Path(_to_project_path(PDL_WEIGHTS_PATH))
        try:
            if not _pdl_path.exists():
                if accelerator.is_main_process:
                    print(f"⚠️  PDL weights not found at {_pdl_path}; disabling PDL. "
                          f"(run scripts/precompute_pdl_weights.py)", flush=True)
                FE_PDL = False
            else:
                _pw = torch.load(str(_pdl_path), weights_only=False)
                _pw = torch.as_tensor(_pw, dtype=torch.float32).reshape(-1)
                if _pw.numel() != int(VOCAB_SIZE):
                    if accelerator.is_main_process:
                        print(f"⚠️  PDL weights size {_pw.numel()} != VOCAB_SIZE {VOCAB_SIZE}; "
                              f"disabling PDL.", flush=True)
                    FE_PDL = False
                else:
                    _pdl_weights = _pw
        except Exception as e:
            if accelerator.is_main_process:
                print(f"⚠️  PDL weights load failed: {e}; disabling PDL.", flush=True)
            FE_PDL = False
            _pdl_weights = None

    # Resolve SFT-GO bundle into a padded [N_total, SEQ_LEN] tensor
    struct_weights_tensor: Optional[torch.Tensor] = None
    struct_id_to_idx: dict[str, int] = {}
    if SFT_USE_SFTGO:
        try:
            bp = Path(SFT_STRUCT_BUNDLE_PATH)
            if not bp.exists():
                if accelerator.is_main_process:
                    print(f"⚠️  SFT-GO bundle not found at {bp}; falling back to no SFT-GO.",
                          flush=True)
                SFT_USE_SFTGO = False
            else:
                _bundle = torch.load(str(bp), weights_only=False)
                _ids = list(_bundle["ids"])
                _weights = _bundle["weights"]
                struct_id_to_idx = {sid: i for i, sid in enumerate(_ids)}
                N_total = len(_ids)
                struct_weights_tensor = torch.ones((N_total, int(SEQ_LEN)), dtype=torch.float32)
                for i, w_np in enumerate(_weights):
                    w_np = np.asarray(w_np, dtype=np.float32)
                    k = min(int(w_np.size), int(SEQ_LEN))
                    if k > 0:
                        struct_weights_tensor[i, :k] = torch.from_numpy(w_np[:k])
                if accelerator.is_main_process:
                    print(
                        f"🧮 SFT-GO: loaded {N_total} sample weights from {bp.name}; "
                        f"mean(>1) tokens/sample ≈ "
                        f"{float((struct_weights_tensor > 1.0).float().sum(dim=1).mean()):.1f}",
                        flush=True,
                    )
        except Exception as e:
            if accelerator.is_main_process:
                print(f"⚠️  SFT-GO bundle load failed: {e}; disabling SFT-GO.", flush=True)
            SFT_USE_SFTGO = False
            struct_weights_tensor = None

    # Map local materialized index → HF row index, when bundle is active
    def _ds_row_origin(meta: dict) -> Optional[list[int]]:
        ro = meta.get("row_origin") if isinstance(meta, dict) else None
        return list(ro) if (ro is not None and len(ro) > 0) else None

    train_row_origin = _ds_row_origin(tr_inst_meta) if SFT_USE_SFTGO else None
    val_row_origin = _ds_row_origin(va_inst_meta) if SFT_USE_SFTGO else None
    # Bundle is keyed by dataset row index — only valid when row_origin matches it.
    # If not provided (e.g. legacy paths), fall back to disabled SFT-GO for safety.
    if SFT_USE_SFTGO and (train_row_origin is None or struct_weights_tensor is None):
        if accelerator.is_main_process:
            print("⚠️  SFT-GO: row_origin missing — disabling SFT-GO for this run.",
                  flush=True)
        SFT_USE_SFTGO = False
        struct_weights_tensor = None

    train_ds = MaterializedSftDataset(
        tr_materialized, TOKENIZER_DIR, SEQ_LEN,
        structure_weights=struct_weights_tensor if SFT_USE_SFTGO else None,
        row_origin=train_row_origin if SFT_USE_SFTGO else None,
    )
    val_ds = (
        MaterializedSftDataset(
            va_materialized, TOKENIZER_DIR, SEQ_LEN,
            structure_weights=struct_weights_tensor if SFT_USE_SFTGO else None,
            row_origin=val_row_origin if SFT_USE_SFTGO else None,
        )
        if (VAL_ENABLED and len(va_materialized) > 0)
        else None
    )

    dl_cfg = resolve_dataloader_settings(
        train_mode=TRAIN_MODE,
        num_workers=DATALOADER_WORKERS,
        pin_memory=DATALOADER_PIN_MEMORY,
        prefetch_factor=DATALOADER_PREFETCH_FACTOR,
        persistent_workers=DATALOADER_PERSISTENT_WORKERS,
    )
    dataloader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        prefetch_factor=dl_cfg["prefetch_factor"] if dl_cfg["num_workers"] > 0 else None,
        persistent_workers=dl_cfg["persistent_workers"] if dl_cfg["num_workers"] > 0 else False,
    )
    # val 僅主程序跑 forward，不經 DDP 包裝的 DataLoader
    val_loader: DataLoader | None = None
    if val_ds is not None and VAL_ENABLED and accelerator.is_main_process:
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=dl_cfg["pin_memory"] and torch.cuda.is_available(),
        )

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader,
    )

    accelerator.print(f"📂 將權重載入模型 ← {CHECKPOINT_LOAD} ({ckpt.get('kind', '?')})")
    incomp = unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
    if hasattr(incomp, "missing_keys") and (incomp.missing_keys or incomp.unexpected_keys):
        raise RuntimeError(f"load_state_dict: {incomp}")
    if sft_resume:
        if ckpt.get("optimizer") is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                accelerator.print(f"⚠️ optimizer 未載入: {e}")
        if ckpt.get("scheduler") is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                accelerator.print(f"⚠️ scheduler 未載入: {e}")
        for pg in optimizer.param_groups:
            pg["lr"] = float(LR)
        if hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs = [float(LR) for _ in scheduler.base_lrs]
    else:
        accelerator.print("ℹ️  新開 SFT：不載入非 SFT 斷點內的 optimizer/scheduler，避免與新 LR 排程衝突。")
    del ckpt
    gc.collect()

    _tok_dir_ce = _to_project_path(TOKENIZER_DIR)
    _ce_tok = AutoTokenizer.from_pretrained(_tok_dir_ce, local_files_only=True)
    _struct_strings = ("</think>", "</final>", "<|im_end|>")
    _struct_ids: list[int] = []
    _struct_skipped: list[str] = []
    _unk_id = getattr(_ce_tok, "unk_token_id", None)
    for _s in _struct_strings:
        _tid = _ce_tok.convert_tokens_to_ids(_s)
        if isinstance(_tid, (list, tuple)):
            _tid = int(_tid[0]) if _tid else -1
        else:
            _tid = int(_tid)
        _bad = _tid < 0 or _tid >= int(VOCAB_SIZE)
        if not _bad and _unk_id is not None and _tid == int(_unk_id):
            _bad = True
        if _bad:
            _struct_skipped.append(_s)
            continue
        _struct_ids.append(_tid)
    _struct_ids = sorted(set(_struct_ids))

    _ce_mult = float(SFT_STRUCTURE_TOKEN_CE_MULT)
    _ce_mult_env = os.environ.get("SFT_STRUCTURE_TOKEN_CE_MULT")
    if _ce_mult_env is not None and str(_ce_mult_env).strip() != "":
        try:
            _ce_mult = float(_ce_mult_env)
        except ValueError:
            if accelerator.is_main_process:
                print(
                    f"⚠️  無效的 SFT_STRUCTURE_TOKEN_CE_MULT={_ce_mult_env!r}，改用函式參數值 {SFT_STRUCTURE_TOKEN_CE_MULT}",
                    flush=True,
                )
            _ce_mult = float(SFT_STRUCTURE_TOKEN_CE_MULT)

    _um_ce = unwrap_model(model)
    if _ce_mult > 1.0 and _struct_ids:
        _um_ce.enable_structure_token_ce_weighting(_struct_ids, _ce_mult)
        if accelerator.is_main_process:
            _pairs = ", ".join(
                f"{_s}→{_ce_tok.convert_tokens_to_ids(_s)}"
                for _s in _struct_strings
                if _s not in _struct_skipped
            )
            print(
                f"🎯 Structure-token CE weighting: mult={_ce_mult}  ({_pairs})",
                flush=True,
            )
    else:
        _um_ce.disable_structure_token_ce_weighting()
        if accelerator.is_main_process:
            if _ce_mult <= 1.0:
                print(
                    "ℹ️  Structure-token CE weighting 關閉（SFT_STRUCTURE_TOKEN_CE_MULT<=1）",
                    flush=True,
                )
            elif not _struct_ids:
                print(
                    "⚠️  Structure-token CE weighting 未啟用：無有效 token id"
                    + (f"（略過字串: {_struct_skipped}）" if _struct_skipped else ""),
                    flush=True,
                )

    # =========================================================================
    # Task 2 — FCP attribute setup on the underlying model
    # =========================================================================
    # We resolve token IDs dynamically from the tokenizer — never hardcoded.
    eos_id_run = -1
    chatml_end_id_run = -1
    think_start_id_run = -1
    think_end_id_run = -1
    final_start_id_run = -1
    final_end_id_run = -1
    try:
        eos_id_run = int(_ce_tok.eos_token_id) if _ce_tok.eos_token_id is not None else -1
        # ChatML data terminates assistant turns with `<|im_end|>` (id≈32001),
        # NOT `</s>` (eos, id=2 — appears ~2×/13M tokens). FCP must monitor the
        # token the model actually emits to stop. Resolve it from the tokenizer.
        _ce_id = _ce_tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(_ce_id, (list, tuple)):
            _ce_id = int(_ce_id[0]) if _ce_id else -1
        else:
            _ce_id = int(_ce_id)
        _unk = getattr(_ce_tok, "unk_token_id", None)
        if _ce_id is not None and _ce_id >= 0 and (_unk is None or _ce_id != int(_unk)):
            chatml_end_id_run = _ce_id
        # `<think>` / `</think>` are single tokens in this tokenizer; verify.
        _ts = _ce_tok.encode("<think>", add_special_tokens=False)
        _te = _ce_tok.encode("</think>", add_special_tokens=False)
        _fs = _ce_tok.encode("<final>", add_special_tokens=False)
        _fe = _ce_tok.encode("</final>", add_special_tokens=False)
        think_start_id_run = int(_ts[0]) if len(_ts) == 1 else -1
        think_end_id_run = int(_te[0]) if len(_te) == 1 else -1
        final_start_id_run = int(_fs[0]) if len(_fs) == 1 else -1
        final_end_id_run = int(_fe[0]) if len(_fe) == 1 else -1
    except Exception as e:
        if accelerator.is_main_process:
            print(f"⚠️  FCP setup: tokenizer ID resolution failed: {e}", flush=True)

    # FCP monitors the ChatML terminator (<|im_end|>); fall back to </s> only if
    # the tokenizer somehow lacks it.
    fcp_eos_id_run = chatml_end_id_run if chatml_end_id_run >= 0 else eos_id_run
    if SFT_USE_FCP and (fcp_eos_id_run < 0 or think_start_id_run < 0 or think_end_id_run < 0):
        if accelerator.is_main_process:
            print(
                "⚠️  FCP: required token ids missing "
                f"(fcp_eos={fcp_eos_id_run}, think_start={think_start_id_run}, "
                f"think_end={think_end_id_run}); disabling FCP.",
                flush=True,
            )
        SFT_USE_FCP = False

    if SFT_USE_FCP:
        _um_ce._fcp_eos_id = int(fcp_eos_id_run)
        _um_ce._fcp_think_start_id = int(think_start_id_run)
        _um_ce._fcp_think_end_id = int(think_end_id_run)
        _um_ce._fcp_lambda = float(SFT_FCP_LAMBDA)
        _um_ce._fcp_delta = float(SFT_FCP_DELTA)
        if accelerator.is_main_process:
            print(
                f"🎯 FCP enabled: fcp_eos_id={fcp_eos_id_run} (<|im_end|>={chatml_end_id_run}, "
                f"</s>={eos_id_run})  think=[{think_start_id_run},{think_end_id_run})  "
                f"λ={SFT_FCP_LAMBDA}  δ={SFT_FCP_DELTA}",
                flush=True,
            )
    else:
        # Ensure attributes are reset/zero so legacy resumes don't carry residue
        _um_ce._fcp_lambda = 0.0

    # ---- Final Enhance: mount Final-SW + PDL + InfoEntropy on the model ------
    # Static / dynamic weights — set once, no per-batch update needed.
    if FE_MASTER:
        _um_ce._enable_final_sw = bool(FE_FINAL_SW)
        _um_ce._final_sw_eta = float(FE_SW_ETA)
        _um_ce._final_sw_lambda = float(FE_SW_LAMBDA)
        _um_ce._enable_pdl = bool(FE_PDL and _pdl_weights is not None)
        _um_ce._pdl_weights = (
            _pdl_weights.to(accelerator.device) if (_pdl_weights is not None) else None
        )
        _um_ce._enable_ie = bool(FE_IE)
        _um_ce._ie_gamma = float(FE_IE_GAMMA)
        _um_ce._ie_on_think = bool(FE_IE_ON_THINK)
        if final_start_id_run >= 0:
            _um_ce._final_start_id = int(final_start_id_run)
        if final_end_id_run >= 0:
            _um_ce._final_end_id = int(final_end_id_run)
        if accelerator.is_main_process:
            print(
                f"✨ Final Enhance: Final-SW={_um_ce._enable_final_sw} "
                f"(η={FE_SW_ETA}, λ={FE_SW_LAMBDA})  "
                f"PDL={_um_ce._enable_pdl}  "
                f"IE={_um_ce._enable_ie} (γ={FE_IE_GAMMA}, on_think={FE_IE_ON_THINK})  "
                f"final=[{final_start_id_run},{final_end_id_run})",
                flush=True,
            )
    else:
        _um_ce._enable_final_sw = False
        _um_ce._enable_pdl = False
        _um_ce._enable_ie = False
        _um_ce._pdl_weights = None
        if accelerator.is_main_process:
            print("ℹ️  Final Enhance disabled (SFT_COT_ENABLE_FINAL_ENHANCE=0)", flush=True)

    if accelerator.is_main_process:
        print(
            f"🧩 Task 2 toggles: SFT-GO={SFT_USE_SFTGO}  FCP={SFT_USE_FCP}  "
            f"SCALe={SFT_USE_SCALE}  "
            f"(ηmax={SFT_SCALE_ETA_MAX}, ηmin={SFT_SCALE_ETA_MIN}, w_final={SFT_SCALE_W_FINAL})",
            flush=True,
        )

    if start_step > 0 and not RESUME_FAST_SKIP_DATALOADER:
        batches_to_skip = start_step * GRADIENT_ACCUMULATION_STEPS
        dataloader = accelerator.skip_first_batches(dataloader, num_batches=batches_to_skip)
    elif start_step > 0:
        accelerator.print("⚡ SFT: 已略過 DataLoader 快進（從下個 epoch 隨機抽樣）。")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    val_log_fp = None
    val_w = None
    _val_log_has_fcp = False
    _val_log_cols: list[str] = list(_VAL_CSV_HEADER_FULL)
    if VAL_ENABLED and val_loader is not None:
        vpath = os.path.abspath(VAL_LOG_FILE)
        os.makedirs(os.path.dirname(vpath) or ".", exist_ok=True)
        if accelerator.is_main_process:
            if _migrate_sft_val_csv_fcp_columns(vpath):
                accelerator.print(
                    f"ℹ️  已升級 val CSV 表頭（加入 val_fcp_penalty / val_eos_prob / val_eos_prob_max）: {vpath}",
                    flush=True,
                )
            if _migrate_sft_val_csv_add_final_enhance(vpath):
                accelerator.print(
                    f"ℹ️  已升級 val CSV 表頭（加入 val_pdl_weight_mean / val_ie_weight_mean）: {vpath}",
                    flush=True,
                )
        v_new = not os.path.exists(vpath) or os.stat(vpath).st_size == 0
        val_log_fp = open(VAL_LOG_FILE, "a", newline="", encoding="utf-8")
        val_w = csv.writer(val_log_fp)
        _val_log_has_fcp = True
        if not v_new:
            with open(VAL_LOG_FILE, "r", encoding="utf-8") as _vrf:
                _vh = _vrf.readline().rstrip("\n")
            _val_log_cols = [c for c in csv.reader([_vh])][0] if _vh else list(_VAL_CSV_HEADER_FULL)
            _val_log_has_fcp = "val_fcp_penalty" in _val_log_cols
        if v_new:
            val_w.writerow(_VAL_CSV_HEADER_FULL)
            val_log_fp.flush()

    gen_tok = None
    test_log_fp = None
    if SFT_TEST_PROMPT and str(SFT_TEST_PROMPT).strip() and accelerator.is_main_process:
        gen_tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
        if gen_tok.pad_token is None and SPECIAL_TOKENS:
            gen_tok.pad_token = SPECIAL_TOKENS[6]
        gen_tok.model_max_length = 1_000_000
        tpa = os.path.abspath(SFT_TEST_LOG_FILE)
        os.makedirs(os.path.dirname(tpa) or ".", exist_ok=True)
        test_log_fp = open(SFT_TEST_LOG_FILE, "a", encoding="utf-8")
        test_log_fp.write(
            f"\n# --- SFT 測試解碼 每 {test_every_eff} 步 T={SFT_TEST_TEMPERATURE!r} "
            f"top_p={SFT_TEST_TOP_P!r} top_k={SFT_TEST_TOP_K!r} "
            f"rep_pen={SFT_TEST_REPETITION_PENALTY!r} max_new={SFT_TEST_MAX_NEW} ---\n"
        )
        test_log_fp.flush()

    # Extended CSV header (backward-compat: detect existing header without eos_prob_max)
    _EXTENDED_CSV_HEADER_FULL = [
        "step",
        "loss",           # total post-FCP/SFT-GO loss
        "ce_loss",        # plain (unweighted) CE, mirrors old column
        "fcp_penalty",    # FCP component (0 if disabled)
        "eos_prob",       # mean p(EOS) inside think (0 if disabled)
        "eos_prob_max",   # max p(EOS) inside think (diagnostic vs FCP δ)
        "sftgo_loss",     # CE under SFT-GO weights only (logging-only proxy)
        "scale_w",        # SCALe scalar for `<think>` at this step (1.0 if off)
        "pdl_weight_mean",   # mean PDL weight over final region (1.0 if off)
        "ie_weight_mean",    # mean InfoEntropy weight over final region (1.0 if off)
        "think_end_acc",  # top-1 acc at </think> label positions (NaN if absent)
        "final_end_acc",  # top-1 acc at </final> label positions (NaN if absent)
        "im_end_acc",     # top-1 acc at <|im_end|> label positions (NaN if absent)
        "lr",
        "grad_norm",
        "router_temp",
        "tokens_seen",
        "step_time_s",
    ]

    if accelerator.is_main_process:
        _train_csv_abs = os.path.abspath(LOG_FILE)
        if _migrate_sft_train_csv_add_eos_prob_max(_train_csv_abs):
            accelerator.print(
                f"ℹ️  已升級訓練 CSV（插入 eos_prob_max；舊列該欄留空）: {_train_csv_abs}",
                flush=True,
            )
        if _migrate_sft_train_csv_add_final_enhance(_train_csv_abs):
            accelerator.print(
                f"ℹ️  已升級訓練 CSV（插入 pdl_weight_mean / ie_weight_mean；舊列該欄留空）: {_train_csv_abs}",
                flush=True,
            )

    _log_cols: list[str] = list(_EXTENDED_CSV_HEADER_FULL)
    if os.path.exists(LOG_FILE) and os.stat(LOG_FILE).st_size > 0:
        with open(LOG_FILE, "r", encoding="utf-8") as _rf:
            _h = _rf.readline().rstrip("\n")
        if _h:
            _log_cols = [c for c in csv.reader([_h])][0]

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as log_fp:
        log_w = csv.writer(log_fp)
        if not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0:
            log_w.writerow(_EXTENDED_CSV_HEADER_FULL)
            _log_cols = list(_EXTENDED_CSV_HEADER_FULL)

        global_step = start_step
        model.train()
        data_iter = iter(dataloader)
        t0 = time.time()
        tokens_seen = global_step * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * SEQ_LEN

        # ---- Task 2: cached SCALe think/final IDs once ----------------------
        _scale_ts = int(think_start_id_run) if think_start_id_run > 0 else -1
        _scale_te = int(think_end_id_run) if think_end_id_run > 0 else -1
        _scale_fs = int(final_start_id_run) if final_start_id_run > 0 else -1
        _scale_fe = int(final_end_id_run) if final_end_id_run > 0 else -1
        _scale_active = bool(SFT_USE_SCALE and _scale_ts > 0 and _scale_te > 0)
        _do_graceful_exit = False

        while global_step < STEPS:
            t_step = time.time()
            acc_loss = 0.0
            acc_ce = 0.0
            acc_fcp = 0.0
            acc_eos = 0.0
            acc_eos_max = 0.0
            acc_scale_w = 0.0
            acc_pdl = 0.0
            acc_ie = 0.0
            # structural-token accuracy accumulators (NaN-aware averaging)
            acc_think_end_acc_sum = 0.0
            acc_think_end_acc_n   = 0
            acc_final_end_acc_sum = 0.0
            acc_final_end_acc_n   = 0
            acc_im_end_acc_sum    = 0.0
            acc_im_end_acc_n      = 0
            optimizer.zero_grad(set_to_none=True)
            gnorm = 0.0
            # Compute SCALe scalar once per macro-step
            if _scale_active:
                _progress = min(1.0, max(0.0, global_step / max(1, int(STEPS))))
                scale_w_think = float(
                    SFT_SCALE_ETA_MIN
                    + 0.5 * (SFT_SCALE_ETA_MAX - SFT_SCALE_ETA_MIN)
                    * (1.0 + math.cos(math.pi * _progress))
                )
            else:
                scale_w_think = 1.0
            scale_w_final = float(SFT_SCALE_W_FINAL) if _scale_active else 1.0
            for _ in range(GRADIENT_ACCUMULATION_STEPS):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    xb, yb, sw_b = batch
                else:
                    xb, yb = batch
                    sw_b = None
                with accelerator.accumulate(model):
                    _amp = torch.bfloat16 if MIXED_PRECISION == "bf16" else torch.float16
                    with torch.autocast(device_type="cuda", dtype=_amp):
                        yb = yb.to(torch.long)
                        # SFT-GO per-token weights (only pass when bundle active)
                        sw_for_model = None
                        if SFT_USE_SFTGO and isinstance(sw_b, torch.Tensor):
                            sw_for_model = sw_b.to(xb.device)
                        # SCALe per-token weights (cumsum mask × scalar)
                        scale_for_model = None
                        if _scale_active and (scale_w_think != 1.0 or scale_w_final != 1.0):
                            with torch.no_grad():
                                is_ts = (xb == _scale_ts).to(torch.int32)
                                is_te = (xb == _scale_te).to(torch.int32)
                                mt = (is_ts.cumsum(1) - is_te.cumsum(1)) > 0
                                if _scale_fs > 0 and _scale_fe > 0:
                                    is_fs = (xb == _scale_fs).to(torch.int32)
                                    is_fe = (xb == _scale_fe).to(torch.int32)
                                    mf = (is_fs.cumsum(1) - is_fe.cumsum(1)) > 0
                                else:
                                    mf = torch.zeros_like(mt)
                                sc = torch.ones_like(xb, dtype=torch.float32)
                                sc = torch.where(mt, torch.full_like(sc, scale_w_think), sc)
                                if scale_w_final != 1.0:
                                    sc = torch.where(mf, torch.full_like(sc, scale_w_final), sc)
                                # Fix 1: protect structural boundary labels from
                                # SCALe downweighting.  The position that predicts
                                # </think> lies inside the think-region mask and
                                # would otherwise get scale_w_think (≥0.3).
                                # Restore it to 1.0 so boundary tokens are always
                                # trained at full strength regardless of schedule.
                                if _scale_te > 0:
                                    sc = torch.where(
                                        yb == _scale_te,
                                        torch.ones_like(sc), sc)
                                if _scale_fe > 0:
                                    sc = torch.where(
                                        yb == _scale_fe,
                                        torch.ones_like(sc), sc)
                                scale_for_model = sc
                        out = model(
                            xb, labels=yb, step=global_step,
                            structure_weights=sw_for_model,
                            scale_weights=scale_for_model,
                        )
                    loss = out[0].mean()
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    accelerator.backward(loss)
                    acc_loss += loss.detach().float()
                    if len(out) >= 3:
                        acc_ce += out[2].item() if isinstance(out[2], torch.Tensor) else float(out[2])
                    if len(out) >= 7:
                        acc_fcp += out[5].item() if isinstance(out[5], torch.Tensor) else float(out[5])
                        acc_eos += out[6].item() if isinstance(out[6], torch.Tensor) else float(out[6])
                    if len(out) >= 8:
                        acc_eos_max += out[7].item() if isinstance(out[7], torch.Tensor) else float(out[7])
                    if len(out) >= 10:
                        acc_pdl += out[8].item() if isinstance(out[8], torch.Tensor) else float(out[8])
                        acc_ie += out[9].item() if isinstance(out[9], torch.Tensor) else float(out[9])
                    if len(out) >= 13:
                        def _acc_update(acc_sum, acc_n, val):
                            v = val.item() if isinstance(val, torch.Tensor) else float(val)
                            import math
                            if not math.isnan(v):
                                return acc_sum + v, acc_n + 1
                            return acc_sum, acc_n
                        acc_think_end_acc_sum, acc_think_end_acc_n = _acc_update(
                            acc_think_end_acc_sum, acc_think_end_acc_n, out[10])
                        acc_final_end_acc_sum, acc_final_end_acc_n = _acc_update(
                            acc_final_end_acc_sum, acc_final_end_acc_n, out[11])
                        acc_im_end_acc_sum, acc_im_end_acc_n = _acc_update(
                            acc_im_end_acc_sum, acc_im_end_acc_n, out[12])
                    acc_scale_w += float(scale_w_think)
                # 必須在「本輪最後一個」accumulate 的 backward 剛完成後讀取；若放在
                # 8 次迴圈外，sync_gradients 往往已變 False，|grad| 會誤顯 0.0
                if accelerator.sync_gradients:
                    nv = accelerator.clip_grad_norm_(
                        unwrap_model(model).parameters(), max_norm=MAX_GRAD_NORM
                    )
                    gnorm = float(nv) if not isinstance(nv, torch.Tensor) else nv.item()
            if not (math.isnan(gnorm) or math.isinf(gnorm)):
                optimizer.step()
            scheduler.step()

            global_step += 1
            tokens_seen += BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * SEQ_LEN

            if accelerator.is_main_process:
                rtemp = get_router_temperature(
                    global_step,
                    warmup=router_warmup,
                    total=router_total,
                    t_start=router_t_start,
                    t_end=router_t_end,
                )
                _al = acc_loss.item() if torch.is_tensor(acc_loss) else float(acc_loss)
                avg_loss = _al / GRADIENT_ACCUMULATION_STEPS
                avg_ce = acc_ce / max(1, GRADIENT_ACCUMULATION_STEPS)
                avg_fcp = acc_fcp / max(1, GRADIENT_ACCUMULATION_STEPS)
                avg_eos = acc_eos / max(1, GRADIENT_ACCUMULATION_STEPS)
                avg_eos_max = acc_eos_max / max(1, GRADIENT_ACCUMULATION_STEPS)
                avg_scale_w = acc_scale_w / max(1, GRADIENT_ACCUMULATION_STEPS)
                avg_pdl = acc_pdl / max(1, GRADIENT_ACCUMULATION_STEPS)
                avg_ie = acc_ie / max(1, GRADIENT_ACCUMULATION_STEPS)
                import math as _math
                avg_think_end_acc = (acc_think_end_acc_sum / acc_think_end_acc_n
                                     if acc_think_end_acc_n > 0 else float("nan"))
                avg_final_end_acc = (acc_final_end_acc_sum / acc_final_end_acc_n
                                     if acc_final_end_acc_n > 0 else float("nan"))
                avg_im_end_acc    = (acc_im_end_acc_sum / acc_im_end_acc_n
                                     if acc_im_end_acc_n > 0 else float("nan"))
                # SFT-GO "loss" proxy = total loss - FCP penalty (so we can see the
                # weighted CE contribution alone in logs).  No extra compute.
                sftgo_loss = max(0.0, avg_loss - avg_fcp)
                lr0 = scheduler.get_last_lr()[0]
                st_el = time.time() - t_step
                tok1 = float(BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * SEQ_LEN)
                tps = tok1 / max(st_el, 1e-6)
                done = global_step - start_step
                pct = 100.0 * done / max(1, train_len)
                if EPOCHS is not None and steps_per_epoch > 0 and done > 0:
                    ei = min(int(EPOCHS), (done - 1) // steps_per_epoch + 1)
                    pi = (done - 1) % steps_per_epoch + 1
                    # 「本輪 ep」= 依本次啟動的 EPOCHS 排程（續跑時仍從 1 起算），非全域從頭計的 epoch。
                    ep_s = f" | 本輪 ep {ei}/{int(EPOCHS)} ({pi}/{steps_per_epoch} step/ep)"
                else:
                    ep_s = ""
                try:
                    ppl = math.exp(avg_ce) if avg_ce < 20.0 else float("inf")
                except OverflowError:
                    ppl = float("inf")
                def _fmt_acc(v: float) -> str:
                    return "nan" if _math.isnan(v) else f"{v:.3f}"

                if PRINT_EVERY_STEPS <= 1 or (global_step % max(1, PRINT_EVERY_STEPS) == 0):
                    print(
                        f"[SFT] 累計 step {global_step}/{STEPS}  本輪進度 {done}/{train_len} ({pct:.1f}%){ep_s} | "
                        f"L_tot {avg_loss:.4f}  CE {avg_ce:.4f}  FCP {avg_fcp:.4f}  "
                        f"P(EOS) {avg_eos:.4f}  P(EOS)_max {avg_eos_max:.4f}  scale {avg_scale_w:.2f}  "
                        f"pdl {avg_pdl:.2f}  ie {avg_ie:.2f}  ppl {ppl:.2f} | "
                        f"</think>acc {_fmt_acc(avg_think_end_acc)}  </final>acc {_fmt_acc(avg_final_end_acc)}  "
                        f"<|im_end|>acc {_fmt_acc(avg_im_end_acc)} | "
                        f"lr {lr0:.2e}  |grad| {gnorm:.3f}  T_router {rtemp:.3f} | "
                        f"{tps:,.0f} tok/s  {st_el:.2f}s/step  cum_tok {tokens_seen:,}",
                        flush=True,
                    )
                # Header-driven row: write exactly the columns this CSV has, in
                # its order. Robust to legacy files missing eos_prob_max / the
                # new final-enhance columns.
                _row_vals = {
                    "step": global_step,
                    "loss": f"{avg_loss:.5f}",
                    "ce_loss": f"{avg_ce:.5f}",
                    "fcp_penalty": f"{avg_fcp:.6f}",
                    "eos_prob": f"{avg_eos:.6f}",
                    "eos_prob_max": f"{avg_eos_max:.6f}",
                    "sftgo_loss": f"{sftgo_loss:.5f}",
                    "scale_w": f"{avg_scale_w:.4f}",
                    "pdl_weight_mean": f"{avg_pdl:.4f}",
                    "ie_weight_mean": f"{avg_ie:.4f}",
                    "think_end_acc": "" if _math.isnan(avg_think_end_acc) else f"{avg_think_end_acc:.4f}",
                    "final_end_acc": "" if _math.isnan(avg_final_end_acc) else f"{avg_final_end_acc:.4f}",
                    "im_end_acc":    "" if _math.isnan(avg_im_end_acc)    else f"{avg_im_end_acc:.4f}",
                    "lr": f"{lr0:.2e}",
                    "grad_norm": f"{gnorm:.4f}",
                    "router_temp": f"{rtemp:.4f}",
                    "tokens_seen": tokens_seen,
                    "step_time_s": f"{st_el:.3f}",
                }
                log_w.writerow(
                    [_row_vals.get(c, "") for c in _log_cols]
                )
                log_fp.flush()

            accelerator.wait_for_everyone()
            if (
                VAL_ENABLED
                and val_loader is not None
                and val_w is not None
                and global_step > 0
                and global_step % val_every_eff == 0
                and accelerator.is_main_process
            ):
                _amp = torch.bfloat16 if MIXED_PRECISION == "bf16" else torch.float16
                t_val = time.time()
                # 必須用 unwrap 後的模組做 val：DDP wrapper 僅 rank0 跑 forward 會讓下一步 allreduce 死鎖
                stats = _sft_eval_batches(
                    unwrap_model(model),
                    val_loader,
                    accelerator.device,
                    VAL_MAX_BATCHES,
                    global_step,
                    _amp,
                )
                if stats and val_log_fp is not None:
                    # Header-driven row matching this val CSV's actual columns.
                    _vrow = {
                        "step": global_step,
                        "val_ce_loss": f"{stats['val_ce_loss']:.8f}",
                        "val_loss_mean": f"{stats['val_loss_mean']:.8f}",
                        "val_fcp_penalty": f"{stats['val_fcp_penalty']:.8f}",
                        "val_eos_prob": f"{stats['val_eos_prob']:.8f}",
                        "val_eos_prob_max": f"{stats['val_eos_prob_max']:.8f}",
                        "val_batches": stats["val_batches"],
                        "val_pdl_weight_mean": f"{stats['val_pdl_weight_mean']:.6f}",
                        "val_ie_weight_mean": f"{stats['val_ie_weight_mean']:.6f}",
                    }
                    val_w.writerow([_vrow.get(c, "") for c in _val_log_cols])
                    val_log_fp.flush()
                    if _val_log_has_fcp:
                        print(
                            f"  📈 [SFT val] step={global_step} ce={stats['val_ce_loss']:.4f} "
                            f"loss={stats['val_loss_mean']:.4f} "
                            f"fcp={stats['val_fcp_penalty']:.6f} "
                            f"P(EOS)={stats['val_eos_prob']:.6f} P(EOS)_max={stats['val_eos_prob_max']:.6f} "
                            f"pdl={stats['val_pdl_weight_mean']:.3f} ie={stats['val_ie_weight_mean']:.3f} "
                            f"batches={stats['val_batches']} t={time.time() - t_val:.2f}s",
                            flush=True,
                        )
                    else:
                        print(
                            f"  📈 [SFT val] step={global_step} ce={stats['val_ce_loss']:.4f} "
                            f"loss={stats['val_loss_mean']:.4f} "
                            f"batches={stats['val_batches']} t={time.time() - t_val:.2f}s",
                            flush=True,
                        )
            if (
                gen_tok is not None
                and test_log_fp is not None
                and global_step > 0
                and global_step % test_every_eff == 0
            ):
                _ampg = torch.bfloat16 if MIXED_PRECISION == "bf16" else torch.float16
                t0g = time.time()
                uq = str(SFT_TEST_PROMPT).strip()
                reply = _sft_sample_test_reply(
                    model,
                    gen_tok,
                    uq,
                    accelerator.device,
                    global_step,
                    SFT_TEST_MAX_NEW,
                    SEQ_LEN,
                    _ampg,
                    temperature=float(SFT_TEST_TEMPERATURE or 0.0),
                    top_p=SFT_TEST_TOP_P,
                    top_k=SFT_TEST_TOP_K,
                    repetition_penalty=float(SFT_TEST_REPETITION_PENALTY),
                    sample_seed=SFT_TEST_SAMPLE_SEED,
                )
                rshort = reply.replace("\n", " ")
                if len(rshort) > 500:
                    rshort = rshort[:500] + "…"
                _dm = (
                    f"T={SFT_TEST_TEMPERATURE!r} top_p={SFT_TEST_TOP_P!r} "
                    f"top_k={SFT_TEST_TOP_K!r} rep_pen={SFT_TEST_REPETITION_PENALTY!r}"
                    if (SFT_TEST_TEMPERATURE and float(SFT_TEST_TEMPERATURE) > 0)
                    else "greedy"
                )
                print(
                    f"  🧪 [SFT test] step={global_step}  ({time.time() - t0g:.1f}s)  [{_dm}]\n"
                    f"      user: {uq[:200]!r}{'…' if len(uq) > 200 else ''}\n"
                    f"      reply: {rshort!r}",
                    flush=True,
                )
                test_log_fp.write(
                    f"\n### step {global_step}  T={SFT_TEST_TEMPERATURE!r} top_p={SFT_TEST_TOP_P!r} "
                    f"top_k={SFT_TEST_TOP_K!r} rep_pen={SFT_TEST_REPETITION_PENALTY!r}\n"
                    f"user:\n{uq}\nreply:\n{reply}\n"
                )
                test_log_fp.flush()
            accelerator.wait_for_everyone()

            done = global_step - start_step
            if (
                int(SAVE_EVERY_STEPS) > 0
                and global_step > 0
                and global_step % int(SAVE_EVERY_STEPS) == 0
                and accelerator.is_main_process
            ):
                ck = {
                    "step": global_step,
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config.__dict__,
                    "sft": True,
                    "sft_warmup_steps": int(warmup_eff),
                    "sft_total_steps": int(STEPS),
                }
                cks, cke = os.path.splitext(CHECKPOINT_SAVE)
                cke = cke or ".pt"
                ck_path = f"{cks}_s{int(global_step)}{cke}"
                d = os.path.dirname(os.path.abspath(ck_path))
                if d:
                    os.makedirs(d, exist_ok=True)
                torch.save(ck, ck_path)
                del ck
                print(
                    f"💾 SFT checkpoint -> {ck_path} (step {global_step})",
                    flush=True,
                )

                # 每累積 10 個 checkpoint，就刪除舊的 9 個（只保留最新 1 個）
                existing_ckpts = []
                base_name = os.path.basename(cks)
                for p in Path(d).glob(f"{base_name}_s*{cke}"):
                    m = re.search(r"_s(\d+)\.pt$", p.name)
                    if m:
                        existing_ckpts.append((int(m.group(1)), p))
                existing_ckpts.sort(key=lambda x: x[0])
                if len(existing_ckpts) >= 10:
                    for _, p in existing_ckpts[:-1]:
                        try:
                            os.remove(p)
                            print(f"🧹 刪除舊版 checkpoint: {p.name}", flush=True)
                        except OSError:
                            pass

                # 優雅停止：檢查 scheduler 是否要求在下一個 checkpoint 後退出
                _flag_path = os.path.join(
                    os.path.dirname(os.path.abspath(OUTPUT_DIR)),
                    "logs", ".scheduler", "request_graceful_stop"
                )
                if os.path.isfile(_flag_path):
                    try:
                        os.remove(_flag_path)
                    except OSError:
                        pass
                    print(
                        f"🛑 優雅停止：在 step {global_step} 收到停止信號，"
                        f"checkpoint 已儲存 → 安全退出",
                        flush=True,
                    )
                    _do_graceful_exit = True  # type: ignore[name-defined]  # noqa: F821

            if _do_graceful_exit:
                break

        if accelerator.is_main_process:
            print(f"🎉 SFT 結束，step={global_step}，log={LOG_FILE}", flush=True)
    if val_log_fp is not None:
        val_log_fp.close()
    if test_log_fp is not None:
        test_log_fp.close()


if __name__ == "__main__":
    # 載入優先序：
    # 1) 環境變數 SFT_CHECKPOINT_LOAD（手動指定）
    # 2) output/checkpoint_sft_s*.pt 中 step 最大者（自動續跑）
    # 3) dataset/pre_train_base_model.pt（從預訓權重新開）
    _ckpt_load = os.environ.get("SFT_CHECKPOINT_LOAD")
    if not _ckpt_load:
        latest = None
        best_step = -1
        for p in Path("output").glob("checkpoint_sft_s*.pt"):
            m = re.search(r"_s(\d+)\.pt$", p.name)
            if not m:
                continue
            s = int(m.group(1))
            if s > best_step:
                best_step = s
                latest = p
        _ckpt_load = str(latest) if latest is not None else "dataset/pre_train_base_model.pt"
        print(f"ℹ️  auto checkpoint load = {_ckpt_load}", flush=True)

    train_sft(
        SFT_DATA_SOURCE="hf_text",
        SFT_TEXT_COLUMN="text",
        LIMA_HF="dataset/mix_a25_u75_ins_hf",
        TOKENIZER_DIR="dataset/tokenizer",
        DATA_PATH="dataset/mix_a25_u75_ins.bin",
        # mix_ins 已在建資料時 --single-turn，列上已是單輪，勿在此再展開
        SFT_SINGLE_TURN_EXPAND=False,
        # 建檔時已 --instruction-filter ins_strict，訓練端關過濾以免重複
        SFT_INSTRUCTION_FILTER="off",
        CHECKPOINT_LOAD=_ckpt_load,
        CHECKPOINT_SAVE="output/checkpoint_sft.pt",
        LOG_FILE="output/train_sft_log.csv",
        VAL_LOG_FILE="output/val_sft_log.csv",
        VOCAB_SIZE=32007,
        SEQ_LEN=768,
        BATCH_SIZE=4,
        GRADIENT_ACCUMULATION_STEPS=4,
        LR=1e-5,
        EPOCHS=1,
        # 週期解碼 smoke test；關閉設 SFT_TEST_PROMPT=None
        SFT_TEST_PROMPT="<|im_start|>user\nWhat is 2+2? Answer with one number.<|im_end|>\n<|im_start|>assistant\n",
        SFT_TEST_EVERY_STEPS=200,
        SFT_TEST_TEMPERATURE=0.8,
        SFT_TEST_TOP_P=0.85,
        SFT_TEST_TOP_K=40,
        SFT_TEST_REPETITION_PENALTY=1.2,
        SFT_TEST_MAX_NEW=64,
    )
