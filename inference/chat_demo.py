#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mamba3-XR Chat Demo — WebSocket real-time streaming chat.

Launches a local FastAPI server with:
  - True WebSocket bidirectional connection (persistent, no reconnect per turn)
  - Multi-turn ChatML conversation history
  - Token-by-token streaming with live metrics
  - Beautiful chat interface for demo

Usage:
  python inference/chat_demo.py                           # defaults
  python inference/chat_demo.py --port 8080               # custom port
  python inference/chat_demo.py --checkpoint path/to.pt   # custom weights

Requires: pip install fastapi uvicorn websockets
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

_INF_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_INF_DIR, "lib")
_REPO_ROOT = os.path.abspath(os.path.join(_INF_DIR, ".."))
for _p in (_LIB_DIR, _INF_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx.core as mx
import mlx.nn as nn

from benchmark_mlx import (
    _apply_inference_type,
    _build_prompt_ids,
    _init_token_counts,
    _invalidate_tucker_caches,
    _materialize_cache_tree,
    _pad_transformer_caches,
    sample_decode_token,
)
from mlx_hybrid_infer import (
    Mamba3Config,
    Mamba3LanguageModel,
    attach_decode_compilation,
    maybe_export_npz_sidecar_after_pt_load,
    resolve_mlx_checkpoint,
    strict_load_and_convert,
)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_model: Mamba3LanguageModel | None = None
_tokenizer: Any = None
_run_prefill: Any = None
_router_temp: mx.array | None = None
_max_cache_len: int = 0
_args: Any = None
_config: Any = None
_stop_ids: frozenset[int] | None = None
_model_ready = False
_vocab_size: int = 0

# Inference lock — model is single-threaded on Metal GPU
_infer_lock = asyncio.Lock()


def _load_model(args: argparse.Namespace) -> dict[str, Any]:
    global _model, _tokenizer, _run_prefill, _router_temp
    global _max_cache_len, _args, _config, _stop_ids, _model_ready, _vocab_size

    _args = args
    timings: dict[str, Any] = {}
    t0 = time.perf_counter()

    compute_dtype_map = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
    target_dtype = compute_dtype_map[args.dtype]
    kv_map = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}
    kv_dtype = target_dtype if args.kv_dtype == "auto" else kv_map[args.kv_dtype]

    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    _vocab_size = len(_tokenizer) if args.vocab_size <= 0 else args.vocab_size
    timings["tokenizer_ms"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    _config = Mamba3Config(
        d_model=768, d_state=64, d_head=64, expand=2, num_layers=6,
        mimo_rank=4, num_kv_heads=4, use_parallel_scan=args.use_parallel_scan,
        chunk_size=64, use_kmoe=True, kmoe_num_experts=8, kmoe_top_k=2,
        kmoe_r1=32, kmoe_r2=512, kmoe_r3=256, ffn_expand=6,
    )
    _config.lookahead_router = False
    _config.tucker_einsum_fuse = bool(args.tucker_einsum_fuse)
    _config.tucker_amx_fuse = False
    _model = Mamba3LanguageModel(_config, _vocab_size)
    timings["model_init_ms"] = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    resolved, kind = resolve_mlx_checkpoint(
        args.checkpoint, repo_root=_REPO_ROOT,
        npz_cache=args.npz_cache, force_pt=args.force_pt,
    )
    if resolved is None or kind == "none":
        print("[chat_demo] No checkpoint — random weights (smoke test).")
    else:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            strict_load_and_convert(_model, resolved)
        if kind == "pt":
            maybe_export_npz_sidecar_after_pt_load(_model, resolved, force_refresh=args.force_pt)
    timings["weights_ms"] = (time.perf_counter() - t2) * 1000
    timings["checkpoint"] = resolved or "(random init)"
    timings["kind"] = kind

    _model.apply(lambda x: x.astype(target_dtype))
    if args.quantize > 0:
        nn.quantize(_model, group_size=64, bits=args.quantize)
    mx.eval(_model.parameters())
    _invalidate_tucker_caches(_model)
    _router_temp = mx.array(args.router_temp, dtype=target_dtype)

    pre_budget = int(args.seq_len) if args.seq_len > 0 else 4096
    _max_cache_len = pre_budget + args.max_new_tokens + 8

    attach_decode_compilation(
        _model, max_cache_len=_max_cache_len, kv_dtype=kv_dtype,
        compile_decode=False,
    )

    def prefill_forward(x: mx.array, rt: mx.array):
        return _model(x, caches=None, seq_pos=None, router_temp=rt)

    _run_prefill = mx.compile(prefill_forward)

    from stream_mlx import _build_stream_stop_token_ids
    _stop_ids = _build_stream_stop_token_ids(_tokenizer, enabled=True)

    t3 = time.perf_counter()
    warmup_ids = _build_prompt_ids(_tokenizer, "warmup", args.seq_len, chatml_user=True)
    x_warm = mx.array([warmup_ids], dtype=mx.int32)
    for _ in range(args.warmup):
        lo, ca = _run_prefill(x_warm, _router_temp)
        mx.eval(lo, ca)
    timings["warmup_ms"] = (time.perf_counter() - t3) * 1000
    timings["total_ms"] = (time.perf_counter() - t0) * 1000

    _model_ready = True
    return timings


# ---------------------------------------------------------------------------
# Multi-turn ChatML prompt builder
# ---------------------------------------------------------------------------
def _encode_plain(text: str) -> list[int]:
    try:
        return list(_tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(_tokenizer.encode(text))


def _build_multiturn_ids(history: list[dict], seq_len: int) -> list[int]:
    """
    Build token ids from multi-turn conversation in ChatML format.
    history: [{"role": "user"|"assistant", "content": "..."}, ...]
    The last entry must be role=user; we append <|im_start|>assistant\n for generation.
    """
    parts: list[str] = []
    for msg in history:
        role = msg["role"]
        content = msg["content"].strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    full_prompt = "".join(parts)
    ids = _encode_plain(full_prompt)
    if seq_len > 0 and len(ids) > seq_len:
        ids = ids[-seq_len:]
    if not ids:
        ids = _encode_plain("<|im_start|>assistant\n")
    return ids


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------
def _stream_generate(history: list[dict], max_tokens: int = 512) -> Iterator[dict]:
    """
    Yields dicts sent as WebSocket JSON frames:
      {"type": "meta",  ...}
      {"type": "token", "text": "...", ...}
      {"type": "done",  ...}
    """
    assert _model_ready

    prompt_ids = _build_multiturn_ids(history, _args.seq_len)

    x_prefill = mx.array([prompt_ids], dtype=mx.int32)
    t_pre0 = time.perf_counter()
    logits, caches = _run_prefill(x_prefill, _router_temp)
    mx.eval(logits, caches)
    prefill_ms = (time.perf_counter() - t_pre0) * 1000

    if not _args.no_materialize_caches:
        caches = _materialize_cache_tree(caches)
        mx.eval(caches)
    caches = _pad_transformer_caches(caches, _max_cache_len)
    mx.eval(caches)

    yield {
        "type": "meta",
        "prefill_ms": round(prefill_ms, 2),
        "prompt_tokens": len(prompt_ids),
        "turns": len([m for m in history if m["role"] == "user"]),
    }

    from stream_mlx import make_compiled_decode_step
    decode_fn = make_compiled_decode_step(_model, _router_temp)

    pos = len(prompt_ids)
    generated_ids: list[int] = []
    special_ids = set(getattr(_tokenizer, "all_special_ids", []) or [])
    prev_decoded_text = ""

    row = logits[0, -1, :]
    token_counts = _init_token_counts(prompt_ids, int(row.shape[0]))
    last = sample_decode_token(row, token_counts, _args)
    token_counts[last] = token_counts[last] + 1
    mx.eval(last)

    t_dec0 = time.perf_counter()
    ttft_ms: float | None = None
    n_out = 0

    def _decode_chunk(tid: int) -> str:
        nonlocal prev_decoded_text
        if tid in special_ids:
            try:
                tok = str(_tokenizer.convert_ids_to_tokens(tid)).strip().lower().replace(" ", "")
            except Exception:
                tok = ""
            if tok in {"<s>", "</s>", "<eos>", "<|eot_id|>", "<|endoftext|>", "<|im_end|>"}:
                return "\n"
            return ""
        generated_ids.append(tid)
        try:
            full_text = _tokenizer.decode(
                generated_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if full_text.startswith(prev_decoded_text):
                chunk = full_text[len(prev_decoded_text):]
            else:
                chunk = _tokenizer.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            prev_decoded_text = full_text
        except Exception:
            chunk = f"<{tid}>"
        return chunk

    tid = int(last.item())
    n_out += 1
    ttft_ms = (time.perf_counter() - t_dec0) * 1000
    chunk = _decode_chunk(tid)
    if chunk:
        elapsed = time.perf_counter() - t_dec0
        yield {"type": "token", "text": chunk, "n": n_out, "tok_s": round(n_out / max(elapsed, 1e-9), 1)}

    if _stop_ids and tid in _stop_ids:
        elapsed = time.perf_counter() - t_dec0
        yield _done_event(n_out, elapsed, ttft_ms, prefill_ms)
        return

    x_one = last.reshape(1, 1)
    for _ in range(max_tokens - 1):
        seq_pos = mx.array(pos, dtype=mx.int32)
        logits_d, caches = decode_fn(x_one, caches, seq_pos)
        row = logits_d[0, -1, :]
        last = sample_decode_token(row, token_counts, _args)
        token_counts[last] = token_counts[last] + 1
        mx.eval(last, caches, token_counts)

        tid = int(last.item())
        n_out += 1
        chunk = _decode_chunk(tid)
        if chunk:
            elapsed = time.perf_counter() - t_dec0
            yield {"type": "token", "text": chunk, "n": n_out, "tok_s": round(n_out / max(elapsed, 1e-9), 1)}

        if _stop_ids and tid in _stop_ids:
            break
        x_one = last.reshape(1, 1)
        pos += 1

    elapsed = time.perf_counter() - t_dec0
    yield _done_event(n_out, elapsed, ttft_ms, prefill_ms)


def _done_event(n_out, elapsed, ttft_ms, prefill_ms):
    return {
        "type": "done",
        "total_tokens": n_out,
        "total_ms": round(elapsed * 1000, 2),
        "tok_s": round(n_out / max(elapsed, 1e-9), 1),
        "ttft_ms": round(ttft_ms or 0, 2),
        "prefill_ms": round(prefill_ms, 2),
    }


# ---------------------------------------------------------------------------
# FastAPI + WebSocket
# ---------------------------------------------------------------------------
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Mamba3-XR Chat Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_load_timings: dict[str, Any] = {}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "chat_demo.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def status():
    return JSONResponse({
        "ready": _model_ready,
        "load_timings": _load_timings,
        "config": {
            "d_model": _config.d_model if _config else 0,
            "num_layers": _config.num_layers if _config else 0,
            "kmoe_num_experts": _config.kmoe_num_experts if _config else 0,
            "kmoe_top_k": _config.kmoe_top_k if _config else 0,
            "vocab_size": _vocab_size,
            "quantize": _args.quantize if _args else 0,
            "dtype": _args.dtype if _args else "",
            "max_new_tokens": _args.max_new_tokens if _args else 0,
        },
    })


@app.websocket("/ws")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "connected", "ready": _model_ready})

    conversation: list[dict] = []
    loop = asyncio.get_event_loop()

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            action = msg.get("action")

            if action == "ping":
                await ws.send_json({"type": "pong", "ready": _model_ready})
                continue

            if action == "clear":
                conversation.clear()
                await ws.send_json({"type": "cleared"})
                continue

            if action == "chat":
                prompt = msg.get("prompt", "").strip()
                max_tokens = min(msg.get("max_tokens", 512), _args.max_new_tokens if _args else 2048)
                if not prompt:
                    await ws.send_json({"type": "error", "message": "Empty prompt"})
                    continue
                if not _model_ready:
                    await ws.send_json({"type": "error", "message": "Model not loaded"})
                    continue

                conversation.append({"role": "user", "content": prompt})
                history_snapshot = list(conversation)

                def _send_now(ev: dict):
                    """Fire ws.send_json from generator thread — blocks until sent."""
                    fut = asyncio.run_coroutine_threadsafe(ws.send_json(ev), loop)
                    fut.result()

                def run_gen() -> str:
                    text = ""
                    try:
                        for ev in _stream_generate(history_snapshot, max_tokens=max_tokens):
                            _send_now(ev)
                            if ev["type"] == "token":
                                text += ev.get("text", "")
                    except Exception as exc:
                        try:
                            _send_now({"type": "error", "message": str(exc)})
                        except Exception:
                            pass
                    return text

                async with _infer_lock:
                    assistant_text = await loop.run_in_executor(None, run_gen)

                conversation.append({"role": "assistant", "content": assistant_text})
                continue

            await ws.send_json({"type": "error", "message": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mamba3-XR Chat Demo Server")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--npz-cache", type=str, default="")
    p.add_argument("--force-pt", action="store_true")
    p.add_argument("--tokenizer", type=str, default=os.path.join(_INF_DIR, "tokenizer"))
    p.add_argument("--inference-type", type=str, default="safe",
                    choices=("throughput", "safe", "eager", "sequential-ssm", "custom"))
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--vocab-size", type=int, default=32007)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--kv-dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--quantize", type=int, choices=[0, 4, 8], default=4)
    p.add_argument("--router-temp", type=float, default=0.5)
    p.add_argument("--temp", type=float, default=0.3)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--min_p", type=float, default=0.05)
    p.add_argument("--rep_pen", type=float, default=1.0)
    p.add_argument("--pres_pen", type=float, default=0.0)
    p.add_argument("--freq_pen", type=float, default=0.0)
    p.add_argument("--no-fast-sample", dest="fast_sample", action="store_false", default=False)
    p.add_argument("--fast-sample", dest="fast_sample", action="store_true")
    p.add_argument("--tucker-einsum-fuse", dest="tucker_einsum_fuse", action="store_true", default=True)
    p.add_argument("--no-tucker-einsum-fuse", dest="tucker_einsum_fuse", action="store_false")
    p.add_argument("--materialize-caches", action="store_true", default=True)
    p.add_argument("--no-compile-prefill", action="store_true", default=False)
    p.add_argument("--eager-decode", action="store_true", default=False)
    p.add_argument("--full-decode-compile", action="store_true", default=True)
    args = p.parse_args()

    _apply_inference_type(args)
    args.no_materialize_caches = not bool(getattr(args, "materialize_caches", True))
    args.no_penalties = True
    args.fused_sample_metal = False
    args.fused_sample_metal_v2 = False
    args.interactive = False
    args.lookahead_router = False
    args.no_eos_stop = False
    args.stop_on_eos = True
    return args


def main():
    args = parse_args()
    global _load_timings

    print(f"\n{'='*60}")
    print(f"  Mamba3-XR Chat Demo (WebSocket)")
    print(f"  Loading model...")
    print(f"{'='*60}\n")

    _load_timings = _load_model(args)

    print(f"\n  Model loaded in {_load_timings['total_ms']:.0f} ms")
    print(f"  Checkpoint: {_load_timings.get('checkpoint', 'N/A')}")
    print(f"  Quantize: {args.quantize}-bit | dtype: {args.dtype}")
    print(f"\n  Server: http://localhost:{args.port}")
    print(f"  WebSocket: ws://localhost:{args.port}/ws")
    print(f"{'='*60}\n")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
