"""Autoregressive generation for Mamba3LM (pure PyTorch)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.nn.functional as F

from .model import Mamba3LM


@dataclass
class GenConfig:
    max_tokens:   int   = 256
    temperature:  float = 0.426
    top_k:        int   = 20
    top_p:        float = 0.981
    min_p:        float = 0.067
    rep_pen:      float = 1.146
    pres_pen:     float = 0.143
    freq_pen:     float = 0.133
    repeat_last_n: int  = 128
    seed:         int   = 0


def _sample(logits: torch.Tensor, cfg: GenConfig, key: torch.Generator,
            counts: torch.Tensor, recent: list[int]) -> int:
    """Temperature + top-k/p/min-p + repetition penalties."""
    v = logits.float()

    # Repetition penalty on recent tokens
    if cfg.rep_pen != 1.0 and recent:
        for tid in set(recent):
            if v[tid] > 0:
                v[tid] /= cfg.rep_pen
            else:
                v[tid] *= cfg.rep_pen

    # Presence + frequency penalty from counts
    if (cfg.pres_pen != 0 or cfg.freq_pen != 0) and counts.any():
        v -= cfg.pres_pen * (counts > 0).float()
        v -= cfg.freq_pen * counts

    if cfg.temperature == 0:
        return int(v.argmax())

    v /= cfg.temperature

    # Top-k
    if cfg.top_k > 0:
        topk_v, _ = torch.topk(v, min(cfg.top_k, v.shape[-1]))
        v[v < topk_v[-1]] = float("-inf")

    probs = F.softmax(v, dim=-1)

    # Min-p
    if cfg.min_p > 0:
        p_max = float(probs.max())
        thresh = cfg.min_p * p_max
        probs[probs < thresh] = 0
        probs /= probs.sum().clamp(min=1e-8)

    # Top-p (nucleus)
    if cfg.top_p < 1.0:
        sorted_p, sorted_idx = torch.sort(probs, descending=True)
        cumsum = sorted_p.cumsum(dim=-1)
        remove = cumsum - sorted_p > cfg.top_p
        sorted_p[remove] = 0
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_p)
        probs /= probs.sum().clamp(min=1e-8)

    return int(torch.multinomial(probs, 1, generator=key))


@dataclass
class GenerateResult:
    tokens:         list[int]
    stop_reason:    str
    n_prompt:       int
    elapsed_prefill: float
    elapsed_decode: float
    prefill_tps:    float
    decode_tps:     float


def generate(
    model:       Mamba3LM,
    prompt_ids:  list[int],
    cfg:         GenConfig,
    tokenizer,                         # tokenizers.Tokenizer
    stop_token_ids: tuple[int, ...] = (),
    on_token:    Callable[[int], None] | None = None,
    kv_len:      int = 512,
) -> GenerateResult:
    """Full autoregressive generation."""
    model.eval()
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    key = torch.Generator(device="cpu")
    key.manual_seed(cfg.seed)

    # ── Prefill ──────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, states = model.prefill(prompt_ids, kv_len=kv_len)
    _sync(device)
    t_prefill = time.perf_counter() - t0

    # Seed first token from prefill logits
    counts  = torch.zeros(model.cfg.vocab_size, device="cpu")
    recent: list[int] = list(prompt_ids[-cfg.repeat_last_n:])

    tid     = _sample(logits.cpu(), cfg, key, counts, recent)
    tokens  = [tid]
    recent.append(tid)
    counts[tid] += 1
    if on_token:
        on_token(tid)

    stop = tid in stop_token_ids

    # ── Decode loop ───────────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(cfg.max_tokens - 1):
            if stop:
                break
            logits, states = model.decode_step(tid, states)
            _sync(device)
            tid = _sample(logits.cpu(), cfg, key, counts, recent[-cfg.repeat_last_n:])
            tokens.append(tid)
            recent.append(tid)
            if len(recent) > cfg.repeat_last_n + len(prompt_ids):
                recent.pop(0)
            counts[tid] += 1
            if on_token:
                on_token(tid)
            if tid in stop_token_ids:
                stop = True
    elapsed_decode = time.perf_counter() - t1

    stop_reason = "eos" if stop else "max_tokens"

    return GenerateResult(
        tokens=tokens,
        stop_reason=stop_reason,
        n_prompt=len(prompt_ids),
        elapsed_prefill=t_prefill,
        elapsed_decode=elapsed_decode,
        prefill_tps=len(prompt_ids) / t_prefill if t_prefill > 0 else 0,
        decode_tps=len(tokens) / elapsed_decode if elapsed_decode > 0 else 0,
    )


def stream(
    model:       Mamba3LM,
    prompt_ids:  list[int],
    cfg:         GenConfig,
    tokenizer,
    stop_token_ids: tuple[int, ...] = (),
    kv_len:      int = 512,
) -> Iterator[dict]:
    """
    Yield dicts: {'type': 'token', 'id': int, 'text': str, 'tok_s': float}
    and finally: {'type': 'done', 'total_tokens': int, 'tok_s': float}
    """
    model.eval()
    device = next(model.parameters()).device
    key    = torch.Generator(device="cpu")
    key.manual_seed(cfg.seed)

    # Prefill
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, states = model.prefill(prompt_ids, kv_len=kv_len)
    _sync(device)
    t_prefill = time.perf_counter() - t0
    yield {"type": "meta", "prefill_tps": len(prompt_ids) / t_prefill,
           "n_prompt": len(prompt_ids)}

    counts  = torch.zeros(model.cfg.vocab_size, device="cpu")
    recent  = list(prompt_ids[-cfg.repeat_last_n:])
    tid     = _sample(logits.cpu(), cfg, key, counts, recent)
    tokens  = [tid]
    recent.append(tid)
    counts[tid] += 1

    # Streaming decode — maintain incremental decoded text
    seen_ids: list[int] = []
    _w_start  = 0
    _prev_win = ""

    def _emit_text(new_tid: int) -> str:
        nonlocal _w_start, _prev_win
        WINDOW = 32
        seen_ids.append(new_tid)
        n = len(seen_ids)
        if n - _w_start > WINDOW * 2:
            new_ws = n - WINDOW
            _w_start = new_ws
            _prev_win = tokenizer.decode(seen_ids[_w_start:-1], skip_special_tokens=False)
        cur = tokenizer.decode(seen_ids[_w_start:], skip_special_tokens=False)
        new_text = cur[len(_prev_win):]
        _prev_win = cur
        return new_text

    t_start = time.perf_counter()
    text    = _emit_text(tid)
    tok_s   = 1 / max(time.perf_counter() - t_start, 1e-9)
    yield {"type": "token", "id": tid, "text": text, "tok_s": 0.0}

    stop = tid in stop_token_ids
    with torch.no_grad():
        for _ in range(cfg.max_tokens - 1):
            if stop:
                break
            t_step = time.perf_counter()
            logits, states = model.decode_step(tid, states)
            _sync(device)
            tid = _sample(logits.cpu(), cfg, key, counts, recent[-cfg.repeat_last_n:])
            elapsed_step = time.perf_counter() - t_step
            tokens.append(tid)
            recent.append(tid)
            counts[tid] += 1

            text  = _emit_text(tid)
            tok_s = 1 / max(elapsed_step, 1e-6)
            yield {"type": "token", "id": tid, "text": text, "tok_s": round(tok_s, 1)}

            if tid in stop_token_ids:
                stop = True

    total_s = time.perf_counter() - t_start
    yield {
        "type":         "done",
        "total_tokens": len(tokens),
        "tok_s":        round(len(tokens) / total_s, 1),
    }


def _sync(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
