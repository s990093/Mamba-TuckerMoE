#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draft Model — Overfit Self-Awareness
======================================
Pure Flash-Attention Transformer (~12.7M params) to overfit the
self_awareness subset of the dataset. For speculative decoding:
draft handles SA queries fast, main Mamba3 model (546M/234M active) verifies.

Architecture:
  d_model=256, n_layers=6, n_heads=8, n_kv_heads=2 (GQA)
  SwiGLU FFN, RMSNorm pre-norm, RoPE, FA2 via torch SDPA, tied embed+head

Usage (via run_overfit.sh — handles conda + CUDA_VISIBLE_DEVICES=4):
    cd /home/hungwei/llm/sft_cot_bundle
    bash scripts/draft/run_overfit.sh

Or manually:
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=scripts \\
        python3 scripts/draft/overfit_self_awareness.py

Outputs:
    output/draft_tf_s{step}.pt      checkpoints
    output/draft_sa_train_log.json  loss curve
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # sft_cot_bundle/
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tokenizers import Tokenizer

# ─── Draft Model Architecture ────────────────────────────────────────────────
# Pure Flash-Attention Transformer (no Mamba/SSM)
#   d_model=256, n_layers=6, n_heads=8, n_kv_heads=2 (GQA 4:1)
#   SwiGLU FFN (mult=3), RMSNorm, RoPE, tied embed+head
#   ~12.7M params — all active, no MoE, FA2 via torch SDPA
VOCAB_SIZE = 32007

# ─── Training Hyper-params ───────────────────────────────────────────────────
LR           = 3e-4
WARMUP_STEPS = 200
TOTAL_STEPS  = int(os.environ.get("DRAFT_STEPS", "5000"))
BATCH_SIZE   = 4
GRAD_ACCUM   = 8      # effective batch = 4×8=32 seqs → stable gradients
SEQ_LEN      = 512
SAVE_EVERY   = 10000  # save every 10k steps for long runs
EVAL_EVERY   = 5000   # eval every 5k steps
LOG_EVERY    = 200    # print loss row every 200 steps
GRAD_CLIP    = 1.0
WEIGHT_DECAY = 0.01
DTYPE        = torch.bfloat16

# ─── Paths ───────────────────────────────────────────────────────────────────
DATASET_PATH   = ROOT / "dataset" / "stf_cot_hf"
TOKENIZER_JSON = ROOT / "dataset" / "tokenizer" / "tokenizer.json"
OUTPUT_DIR     = ROOT / "output"
# transformer checkpoints use "draft_tf_s*.pt" to avoid mixing with old Mamba ckpts
CKPT_PREFIX    = "draft_tf_s"

# ─── Self-Awareness ──────────────────────────────────────────────────────────
SA_BUCKET   = "self_awareness"
SA_SYSPROMPT = (
    "You are Mamba in Self-Awareness mode. Keep answers architecturally "
    "consistent: Hybrid Mamba-TuckerMoE, edge-deployed on iPhone, offline "
    "by default, no subjective consciousness, and no fabricated capabilities."
)

# Two focused eval questions — enough to track overfit progress
EVAL_QUESTIONS = [
    "What architecture are you based on?",
    "Do you have subjective consciousness?",
]

# Special token IDs (from dataset/tokenizer/tokenizer.json)
_IM_START = 32000   # <|im_start|>
_IM_END   = 32001   # <|im_end|>
_EOS      = 2       # </s>
_THINK    = 32002   # <think>


# ─── Model ───────────────────────────────────────────────────────────────────

def build_draft_model() -> "DraftTransformer":
    from draft.transformer_draft import DraftTransformer, DraftConfig

    cfg = DraftConfig(
        d_model=256,
        n_layers=6,
        n_heads=8,
        n_kv_heads=2,     # GQA: 4 query heads share each KV head
        ffn_mult=3,       # SwiGLU hidden = 768
        max_seq=SEQ_LEN + 256,
        vocab_size=VOCAB_SIZE,
    )
    model = DraftTransformer(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"Draft model: {n/1e6:.1f}M params  (pure Transformer, GQA 8h/2kv, FA2)")
    print(f"Main  model: 546M total / 234M active  (Mamba3 Hybrid TuckerMoE)")
    print(f"Speedup est: ~{234e6/n:.1f}×")
    return model


# ─── Dataset ─────────────────────────────────────────────────────────────────

def load_sa_texts() -> list[str]:
    from datasets import load_from_disk

    ds = load_from_disk(str(DATASET_PATH))
    sa = ds.filter(lambda x: x.get("sys_bucket") == SA_BUCKET, num_proc=1)
    print(f"Self-awareness examples: {len(sa):,} / {len(ds):,} total")
    return sa["text"]


def tokenize_texts(texts: list[str], tok: Tokenizer) -> list[int]:
    """Tokenize all SA sequences and pack into one flat token stream."""
    packed: list[int] = []
    n_seqs = 0
    for text in texts:
        ids = tok.encode(text).ids
        if len(ids) > 16:
            packed.extend(ids)
            packed.append(_EOS)   # sequence boundary marker
            n_seqs += 1
    print(f"Tokenized: {n_seqs:,} seqs packed → {len(packed):,} tokens total")
    return packed


# ─── Training utils ──────────────────────────────────────────────────────────

def sample_batch(
    packed: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random contiguous chunks from the packed token stream."""
    xs, ys = [], []
    n = len(packed)
    for _ in range(batch_size):
        start = random.randint(0, n - seq_len - 1)
        chunk = packed[start : start + seq_len + 1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y


def cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


# ─── Inference ───────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: "DraftTransformer",
    tok: Tokenizer,
    question: str,
    device: torch.device,
    max_new: int = 120,
    temperature: float = 0.8,
    top_p: float = 0.9,
    rep_penalty: float = 1.3,
) -> str:
    model.eval()
    prompt = (
        f"<|im_start|>system\n{SA_SYSPROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    ids = tok.encode(prompt).ids
    x   = torch.tensor([ids], dtype=torch.long, device=device)

    # ── Prefill: process entire prompt once, build KV cache ──────────────────
    with torch.autocast(device_type="cuda", dtype=DTYPE):
        logits, kv_cache = model(x, use_cache=True)

    out_ids: list[int] = []

    for _ in range(max_new):
        lf = logits[0, -1, :].float()

        # repetition penalty (suppresses "a a a" collapse)
        for tid in set(out_ids[-30:]):
            lf[tid] = lf[tid] / rep_penalty if lf[tid] > 0 else lf[tid] * rep_penalty

        # temperature + nucleus (top-p) sampling
        lf = lf / temperature
        probs = torch.softmax(lf, dim=-1)
        sprobs, sidx = torch.sort(probs, descending=True)
        cumprobs = sprobs.cumsum(dim=0)
        sprobs[cumprobs - sprobs > top_p] = 0.0
        sprobs /= sprobs.sum()
        next_id = int(sidx[torch.multinomial(sprobs, 1)])

        if next_id in (_IM_END, _EOS):
            break
        out_ids.append(next_id)

        # ── Decode: one new token + cached KV → O(T) not O(T²) ──────────
        x_new = torch.tensor([[next_id]], dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            logits, kv_cache = model(x_new, past_key_values=kv_cache, use_cache=True)

    return tok.decode(out_ids, skip_special_tokens=False)


# ─── Main ────────────────────────────────────────────────────────────────────

def _find_latest_checkpoint() -> Path | None:
    ckpts = sorted(OUTPUT_DIR.glob(f"{CKPT_PREFIX}*.pt"),
                   key=lambda p: int(p.stem.split("_s")[-1]))
    return ckpts[-1] if ckpts else None


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # CUDA_VISIBLE_DEVICES=4 is set by run_overfit.sh → this process sees it as cuda:0
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32  = True
    torch.backends.cudnn.allow_tf32        = True
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Steps:  {TOTAL_STEPS}  batch={BATCH_SIZE}  seq={SEQ_LEN}\n")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    assert tok.get_vocab_size() == VOCAB_SIZE, \
        f"Vocab mismatch: {tok.get_vocab_size()} vs {VOCAB_SIZE}"

    # ── Dataset ───────────────────────────────────────────────────────────────
    texts   = load_sa_texts()
    all_ids = tokenize_texts(texts, tok)

    # ── Model + Optimizer ─────────────────────────────────────────────────────
    model = build_draft_model().to(device=device, dtype=DTYPE)

    # torch.compile disabled: triton 3.6.0 + torch 2.5.1 inductor ABI conflict
    # TF32 + FA2 via SDPA (enabled above) already give good throughput in eager mode

    opt   = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
        fused=True,
    )

    # ── Resume from latest checkpoint ─────────────────────────────────────────
    start_step = 0
    resume_ckpt = _find_latest_checkpoint()
    if resume_ckpt is not None:
        ckpt_data = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data["model"])
        start_step = ckpt_data.get("step", 0)
        print(f"Resumed from {resume_ckpt.name}  (step {start_step})\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    log_rows: list[dict] = []
    t_start   = time.time()
    step_end  = start_step + TOTAL_STEPS
    loss_buf: list[float] = []   # running window for smoothed loss

    # ─ header ─────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  TRAIN  {start_step+1} → {step_end}"
          f"  eff_batch={BATCH_SIZE*GRAD_ACCUM}  seq={SEQ_LEN}")
    print(f"{'═'*60}")
    print(f"  {'step':>7}  {'loss(avg)':>9}  {'ppl':>7}  {'lr':>8}  elapsed")
    print(f"  {'─'*7}  {'─'*9}  {'─'*7}  {'─'*8}  {'─'*8}")

    def _save_ckpt(step: int) -> None:
        ckpt = OUTPUT_DIR / f"{CKPT_PREFIX}{step}.pt"
        torch.save({
            "step":   step,
            "model":  model.state_dict(),
            "arch":   "transformer",
            "config": {
                "d_model":    256,
                "n_layers":   6,
                "n_heads":    8,
                "n_kv_heads": 2,
                "ffn_mult":   3,
                "vocab_size": VOCAB_SIZE,
            },
        }, ckpt)
        elapsed = time.time() - t_start
        m, s = divmod(int(elapsed), 60)
        print(f"\n  [CKPT] {ckpt.name}  ({m}m{s:02d}s)\n")

    for step in range(start_step + 1, step_end + 1):
        model.train()
        rel    = step - start_step
        lr_now = cosine_lr(rel, WARMUP_STEPS, TOTAL_STEPS, LR)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        # ── gradient accumulation (effective batch = BATCH_SIZE × GRAD_ACCUM) ─
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(GRAD_ACCUM):
            x, y = sample_batch(all_ids, BATCH_SIZE, SEQ_LEN, device)
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                out = model(x, labels=y)
            micro_loss = out[0] if isinstance(out, (tuple, list)) else out
            (micro_loss / GRAD_ACCUM).backward()
            accum_loss += float(micro_loss) / GRAD_ACCUM
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        loss_val = accum_loss
        ppl_val  = math.exp(min(loss_val, 20.0))
        loss_buf.append(loss_val)
        if len(loss_buf) > LOG_EVERY:
            loss_buf.pop(0)
        log_rows.append({"step": step, "loss": round(loss_val, 5),
                         "ppl": round(ppl_val, 3), "lr": lr_now})

        # ── TRAIN log row (smoothed) ──────────────────────────────────────
        if step % LOG_EVERY == 0:
            avg_loss = sum(loss_buf) / len(loss_buf)
            avg_ppl  = math.exp(min(avg_loss, 20.0))
            elapsed  = time.time() - t_start
            m, s = divmod(int(elapsed), 60)
            print(f"  {step:>7}/{step_end}  {avg_loss:9.4f}  "
                  f"{avg_ppl:7.1f}  {lr_now:.2e}  {m}m{s:02d}s")

        # ── EVAL block ────────────────────────────────────────────────────
        if step % EVAL_EVERY == 0:
            elapsed = time.time() - t_start
            m, s = divmod(int(elapsed), 60)
            avg_loss = sum(loss_buf) / len(loss_buf)
            avg_ppl  = math.exp(min(avg_loss, 20.0))
            print(f"\n{'─'*60}")
            print(f"  EVAL  step={step}/{step_end}"
                  f"  loss={avg_loss:.4f}  ppl={avg_ppl:.1f}  {m}m{s:02d}s")
            print(f"{'─'*60}")
            for q in EVAL_QUESTIONS:
                ans = generate(model, tok, q, device)
                print(f"  Q: {q}")
                print(f"  A: {ans[:200]}")
            print(f"{'─'*60}\n")
            model.train()

        # ── Checkpoint ────────────────────────────────────────────────────
        if step % SAVE_EVERY == 0:
            _save_ckpt(step)

    # ── Always save final checkpoint ──────────────────────────────────────────
    last_step = step_end
    if last_step % SAVE_EVERY != 0:
        _save_ckpt(last_step)

    # ── Write loss log ────────────────────────────────────────────────────────
    log_path = OUTPUT_DIR / "draft_sa_train_log.json"
    log_path.write_text(json.dumps(log_rows, indent=2))
    final_ppl = math.exp(min(log_rows[-1]["loss"], 20.0)) if log_rows else 0.0
    print(f"\n{'═'*60}")
    print(f"  DONE  steps={last_step}  loss={log_rows[-1]['loss']:.4f}"
          f"  ppl={final_ppl:.1f}")
    print(f"  log → {log_path.name}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
