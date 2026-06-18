#!/usr/bin/env python3
"""Verify chunked prefill bounds memory on long prompts (up to 16k tokens).

Loads the real model and runs _prefill_chunked-style prefill at increasing
sequence lengths, reporting MLX peak memory + throughput. Confirms the
chunked path keeps peak roughly flat instead of growing with L.

  python -m mamba3_mlx.tools.verify_long_prefill
  python -m mamba3_mlx.tools.verify_long_prefill --lengths 2048,8192,16384 --chunk 256
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from mamba3_mlx import chat_config as cfg
from mamba3_mlx.utils.config import Mamba3Config
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint

_REPO = Path(__file__).resolve().parents[2]


def _iter_state_arrays(states):
    if not states:
        return
    for st in states:
        if isinstance(st, dict):
            for v in st.values():
                if v is not None:
                    yield v


def prefill_chunked(model, ids, chunk):
    states = None
    n = len(ids)
    for start in range(0, n, chunk):
        sl = ids[start:start + chunk]
        x = mx.array([sl], dtype=mx.int32)
        logits, states = model(x, states=states)
        # Force per-chunk evaluation so intermediates are freed (mirrors
        # chat_demo._prefill_chunked + hybrid_model block-by-block eval).
        if start + chunk >= n:
            mx.eval(logits[0, -1], *_iter_state_arrays(states))
        else:
            mx.eval(*_iter_state_arrays(states))
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(_REPO / cfg.CHECKPOINT_RELPATH))
    ap.add_argument("--tokenizer", default=str(_REPO / cfg.TOKENIZER_RELPATH))
    ap.add_argument("--lengths", default="1024,2048,4096,8192,16384")
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()

    tok_path = args.tokenizer
    if tok_path.endswith(".json"):
        tok_path = str(Path(tok_path).parent)
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    vocab = len(tok)

    print(f"[verify] loading model ({Path(args.checkpoint).name}, vocab={vocab}) …", flush=True)
    t0 = time.perf_counter()
    model = Mamba3LanguageModel(Mamba3Config(vocab_size=vocab))
    load_checkpoint(model, args.checkpoint, dtype=mx.bfloat16)
    print(f"[verify] loaded in {time.perf_counter()-t0:.1f}s  chunk={args.chunk}\n", flush=True)

    lengths = [int(x) for x in args.lengths.split(",") if x]
    print(f"{'tokens':>8} | {'peak GB':>8} | {'time s':>7} | {'tok/s':>7} | status")
    print("-" * 52)
    for n in lengths:
        ids = [(i * 131 + 7) % vocab for i in range(n)]   # deterministic dummy
        mx.reset_peak_memory()
        t = time.perf_counter()
        try:
            prefill_chunked(model, ids, args.chunk)
            dt = time.perf_counter() - t
            peak = mx.get_peak_memory() / 1e9
            print(f"{n:8d} | {peak:8.2f} | {dt:7.2f} | {n/dt:7.0f} | ok", flush=True)
        except Exception as e:
            print(f"{n:8d} | {'--':>8} | {'--':>7} | {'--':>7} | FAILED: {type(e).__name__}: {e}", flush=True)
            break


if __name__ == "__main__":
    main()
