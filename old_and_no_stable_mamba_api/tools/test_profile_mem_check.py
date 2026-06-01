#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight memory smoke check for mlx_hybrid_infer + profile-style prefill.

Run (from repo root, with MLX available):
  python inference/tools/test_profile_mem_check.py
  python inference/tools/test_profile_mem_check.py --dtype fp32
  python inference/tools/test_profile_mem_check.py --dtype int8

``--dtype`` can be fp32 / bf16 / fp16 (float weights) or int4 / int8 (``nn.quantize`` on top of bf16
weights, same idea as ``benchmark_mlx --quantize``).

Peak/active use ``mx.get_peak_memory`` / ``mx.get_active_memory`` and ``mx.clear_cache``.

Does not load a checkpoint (random weights).
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_INF_ROOT = os.path.dirname(_TOOLS_DIR)
_LIB_DIR = os.path.join(_INF_ROOT, "lib")
for _p in (_LIB_DIR, _INF_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> None:
    p = argparse.ArgumentParser(description="MLX memory smoke test (random weights)")
    p.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=("fp32", "bf16", "fp16", "int4", "int8"),
        help="Float dtype for weights/KV, or int4/int8 for weight-only quantization (bf16 base + nn.quantize).",
    )
    args = p.parse_args()

    import mlx.core as mx
    import mlx.nn as nn

    from benchmark_mlx import _invalidate_tucker_caches, _pad_transformer_caches
    from mlx_hybrid_infer import Mamba3Config, Mamba3LanguageModel, attach_decode_compilation

    compute_map = {
        "fp32": (mx.float32, 0),
        "bf16": (mx.bfloat16, 0),
        "fp16": (mx.float16, 0),
        "int4": (mx.bfloat16, 4),
        "int8": (mx.bfloat16, 8),
    }
    compute_dt, qbits = compute_map[args.dtype]

    config = Mamba3Config(
        d_model=768,
        d_state=64,
        d_head=64,
        expand=2,
        num_layers=6,
        mimo_rank=4,
        num_kv_heads=4,
        use_parallel_scan=True,
        chunk_size=64,
        use_kmoe=True,
        kmoe_num_experts=8,
        kmoe_top_k=2,
        kmoe_r1=32,
        kmoe_r2=512,
        kmoe_r3=256,
        ffn_expand=6,
    )
    model = Mamba3LanguageModel(config, 32000)

    model.apply(lambda x: x.astype(compute_dt))
    mx.eval(model.parameters())
    if qbits > 0:
        nn.quantize(model, group_size=64, bits=qbits)
        mx.eval(model.parameters())
    _invalidate_tucker_caches(model)

    # KV / router: match compute dtype for float runs; after quantize keep bf16 KV like benchmark_mlx
    kv_dtype = mx.bfloat16 if qbits > 0 else compute_dt
    router_temp = mx.array(0.5, dtype=compute_dt if qbits == 0 else mx.bfloat16)

    seq_len = 128
    max_cache_len = seq_len + 8
    attach_decode_compilation(model, max_cache_len=max_cache_len, kv_dtype=kv_dtype, compile_decode=False)
    model.set_lm_head_compile(False)

    x = mx.zeros((1, seq_len), dtype=mx.int32)
    mx.reset_peak_memory()

    logits, caches = model(x, caches=None, seq_pos=None, router_temp=router_temp)
    mx.eval(logits, caches)
    caches = _pad_transformer_caches(caches, max_cache_len)
    mx.eval(caches)

    pos = seq_len
    one = mx.zeros((1, 1), dtype=mx.int32)
    logits2, caches = model(one, caches=caches, seq_pos=mx.array(pos, dtype=mx.int32), router_temp=router_temp)
    mx.eval(logits2, caches)

    del logits, logits2, caches, x, one
    gc.collect()
    try:
        mx.clear_cache()
    except Exception as e:
        print(f"test_profile_mem_check: mx.clear_cache() failed: {e}")

    peak = mx.get_peak_memory() / (1024**2)
    active = mx.get_active_memory() / (1024**2)
    tag = args.dtype if qbits == 0 else f"{args.dtype}(q{qbits})"
    print(f"test_profile_mem_check: OK  mode={tag}  peak≈{peak:.1f} MB  active≈{active:.1f} MB")


if __name__ == "__main__":
    main()
