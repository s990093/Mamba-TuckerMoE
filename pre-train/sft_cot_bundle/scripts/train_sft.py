# -*- coding: utf-8 -*-
"""
SFT 訓練（獨立於 train.py，不修改原檔）：
- 一筆樣本 = 一則完整對話（預設）或 **`SFT_SINGLE_TURN_EXPAND=True` 時** 將多輪拆成多筆「單輪 user→assistant」；不做多列合併 / packing。
- 若單筆 token 長度 > SEQ_LEN+1 會**保留前段、截掉後段**。尾部通常含 `</s>`、`<|im_end|>` 等關閉符——截斷過多會讓模型少學到「何時停」；請看啟動時的長度分佈與 `sft_lima_len_stats.json`。
- **SEQ_LEN 建議**：預設 **512**；要覆蓋長對話可試 **1024/2048** 並觀察 VRAM 與截斷率。
- 本管線的 LIMA 樣本為 **instruction 格式**（`im_start` / `im_end`），你若有另加 CoT/`<final>` 結構，更需長 `SEQ_LEN` 以免切尾。
- 啟動時掃 HF，輸出長度分佈到 stdout 與 `output/sft_*_len_stats*.json`。粒度／過濾不同時檔名含 `_single`、`_noweak`、`_instr` 等後綴；見 `sft_instruction_filter.py`。
- Val：自 train 隨機劃 5%（可調 VAL_FRAC），用與訓練**相同** label mask 算 val_ce / val_loss，寫入 output/val_sft_log.csv。
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
    meta = {
        "instruction_filter": instruction_filter,
        "n_instruction_candidates": n_cand,
        "n_instruction_dropped": n_drop,
        "n_kept": len(out),
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
    meta = {
        "instruction_filter": instruction_filter,
        "n_instruction_candidates": n_cand,
        "n_instruction_dropped": n_drop,
        "n_kept": len(out),
    }
    return out, meta


class MaterializedSftDataset(Dataset):
    """一字串一個 supervised ChatML（已由 lima `_format_conversation` 格式化）。"""

    def __init__(self, texts: list[str], tokenizer_path: str | Path, seq_len: int):
        self.texts = list(texts)
        self.tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        if self.tok.pad_token is None and SPECIAL_TOKENS:
            self.tok.pad_token = SPECIAL_TOKENS[6]
        self.pad_id = int(self.tok.convert_tokens_to_ids(self.tok.pad_token))
        self.seq_len = int(seq_len)
        self.tok.model_max_length = 1_000_000

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _build_xy_masked(self.texts[i], self.tok, self.seq_len, self.pad_id)


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
    n_batches = 0
    for i, (xb, yb) in enumerate(val_loader):
        if i >= max_batches:
            break
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
        n_batches += 1
    model.train()
    if n_batches == 0:
        return None
    return {
        "val_ce_loss": total_ce / n_batches,
        "val_loss_mean": total_loss / n_batches,
        "val_batches": n_batches,
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
    n_batches_one_epoch = max(1, (n_train + BATCH_SIZE - 1) // BATCH_SIZE)
    steps_per_epoch = max(
        1, (n_batches_one_epoch + GRADIENT_ACCUMULATION_STEPS - 1) // GRADIENT_ACCUMULATION_STEPS
    )
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
                f"           每 epoch 微批次 {n_batches_one_epoch} → optimizer steps/epoch ≈ {steps_per_epoch}"
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
            {"params": decay_params, "weight_decay": 0.1},
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

    train_ds = MaterializedSftDataset(tr_materialized, TOKENIZER_DIR, SEQ_LEN)
    val_ds = (
        MaterializedSftDataset(va_materialized, TOKENIZER_DIR, SEQ_LEN)
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

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
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
    else:
        accelerator.print("ℹ️  新開 SFT：不載入非 SFT 斷點內的 optimizer/scheduler，避免與新 LR 排程衝突。")
    del ckpt
    gc.collect()

    if start_step > 0 and not RESUME_FAST_SKIP_DATALOADER:
        batches_to_skip = start_step * GRADIENT_ACCUMULATION_STEPS
        dataloader = accelerator.skip_first_batches(dataloader, num_batches=batches_to_skip)
    elif start_step > 0:
        accelerator.print("⚡ SFT: 已略過 DataLoader 快進（從下個 epoch 隨機抽樣）。")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    val_log_fp = None
    val_w = None
    if VAL_ENABLED and val_loader is not None:
        vpath = os.path.abspath(VAL_LOG_FILE)
        os.makedirs(os.path.dirname(vpath) or ".", exist_ok=True)
        v_new = not os.path.exists(vpath) or os.stat(vpath).st_size == 0
        val_log_fp = open(VAL_LOG_FILE, "a", newline="", encoding="utf-8")
        val_w = csv.writer(val_log_fp)
        if v_new:
            val_w.writerow(["step", "val_ce_loss", "val_loss_mean", "val_batches"])
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

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as log_fp:
        log_w = csv.writer(log_fp)
        if not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0:
            log_w.writerow(
                [
                    "step",
                    "loss",
                    "ce_loss",
                    "lr",
                    "grad_norm",
                    "router_temp",
                    "tokens_seen",
                    "step_time_s",
                ]
            )

        global_step = start_step
        model.train()
        data_iter = iter(dataloader)
        t0 = time.time()
        tokens_seen = global_step * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * SEQ_LEN

        while global_step < STEPS:
            t_step = time.time()
            acc_loss = 0.0
            acc_ce = 0.0
            optimizer.zero_grad(set_to_none=True)
            gnorm = 0.0
            for _ in range(GRADIENT_ACCUMULATION_STEPS):
                try:
                    xb, yb = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    xb, yb = next(data_iter)
                with accelerator.accumulate(model):
                    _amp = torch.bfloat16 if MIXED_PRECISION == "bf16" else torch.float16
                    with torch.autocast(device_type="cuda", dtype=_amp):
                        yb = yb.to(torch.long)
                        out = model(xb, labels=yb, step=global_step)
                    loss = out[0].mean()
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    accelerator.backward(loss)
                    acc_loss += loss.detach().float()
                    if len(out) >= 3:
                        acc_ce += out[2].item() if isinstance(out[2], torch.Tensor) else float(out[2])
                # 必須在「本輪最後一個」accumulate 的 backward 剛完成後讀取；若放在
                # 8 次迴圈外，sync_gradients 往往已變 False，|grad| 會誤顯 0.0
                if accelerator.sync_gradients:
                    nv = accelerator.clip_grad_norm_(
                        unwrap_model(model).parameters(), max_norm=1.0
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
                lr0 = scheduler.get_last_lr()[0]
                st_el = time.time() - t_step
                tok1 = float(BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * SEQ_LEN)
                tps = tok1 / max(st_el, 1e-6)
                done = global_step - start_step
                pct = 100.0 * done / max(1, train_len)
                if EPOCHS is not None and steps_per_epoch > 0 and done > 0:
                    ei = min(int(EPOCHS), (done - 1) // steps_per_epoch + 1)
                    pi = (done - 1) % steps_per_epoch + 1
                    ep_s = f" | ep {ei}/{int(EPOCHS)} ({pi}/{steps_per_epoch} step/ep)"
                else:
                    ep_s = ""
                try:
                    ppl = math.exp(avg_ce) if avg_ce < 20.0 else float("inf")
                except OverflowError:
                    ppl = float("inf")
                if PRINT_EVERY_STEPS <= 1 or (global_step % max(1, PRINT_EVERY_STEPS) == 0):
                    print(
                        f"[SFT] step {global_step}  新步 {done}/{train_len} ({pct:.1f}%){ep_s} | "
                        f"loss {avg_loss:.4f}  ce {avg_ce:.4f}  ppl {ppl:.2f} | "
                        f"lr {lr0:.2e}  |grad| {gnorm:.3f}  T_router {rtemp:.3f} | "
                        f"{tps:,.0f} tok/s  {st_el:.2f}s/step  cum_tok {tokens_seen:,}",
                        flush=True,
                    )
                log_w.writerow(
                    [
                        global_step,
                        f"{avg_loss:.5f}",
                        f"{avg_ce:.5f}",
                        f"{lr0:.2e}",
                        f"{gnorm:.4f}",
                        f"{rtemp:.4f}",
                        tokens_seen,
                        f"{st_el:.3f}",
                    ]
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
                stats = _sft_eval_batches(
                    model, val_loader, accelerator.device, VAL_MAX_BATCHES, global_step, _amp
                )
                if stats and val_log_fp is not None:
                    val_w.writerow(
                        [
                            global_step,
                            f"{stats['val_ce_loss']:.5f}",
                            f"{stats['val_loss_mean']:.5f}",
                            stats["val_batches"],
                        ]
                    )
                    val_log_fp.flush()
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
        for p in Path("/ssd1/hungwei").glob("checkpoint_sft_s*.pt"):
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
        CHECKPOINT_SAVE="/ssd1/hungwei/checkpoint_sft.pt",
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
