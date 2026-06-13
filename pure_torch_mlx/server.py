#!/usr/bin/env python3
"""
Pure-PyTorch inference server for Mamba3-TuckerMoE.

Runs on a separate port (default 7861) so it can be accessed alongside
the MLX chat_demo server for side-by-side speed comparison.

Endpoints:
  GET  /api/status        — model load status + last benchmark
  POST /api/bench         — run N decode steps, return tok/s JSON
  POST /api/stream        — SSE streaming token generation
  GET  /api/config        — model config (vocab, layers, etc.)

Usage:
  cd /path/to/Mamba3-XR
  .venv/bin/python3 -m pure_torch_mlx.server [--port 7861] [--device mps]
  .venv/bin/python3 -m pure_torch_mlx.server --mock   # UI only, no weights
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except ImportError:
    sys.exit("PyTorch not installed — pip install torch")

try:
    from tokenizers import Tokenizer
except ImportError:
    sys.exit("tokenizers not installed — pip install tokenizers")

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError:
    sys.exit("fastapi not installed — pip install fastapi uvicorn")

from .config  import Mamba3Config
from .model   import Mamba3LM
from .weights import load_checkpoint
from .generate import GenConfig, generate, stream as gen_stream

# ── Globals ─────────────────────────────────────────────────────────────────────

_model:     Mamba3LM | None = None
_tokenizer: Tokenizer | None = None
_device:    torch.device | None = None
_dtype:     torch.dtype | None = None
_cfg:       Mamba3Config | None = None
_ready      = False
_load_info: dict = {}
_args = None
_MOCK_MODE = False

DEFAULT_CKPT = str(REPO_ROOT / "checkpoints" / "v6" / "latest_sft_cot_model.npz")
DEFAULT_TOK  = str(REPO_ROOT / "cot_dataset" / "tokenizer.json")
DEFAULT_SYS  = "You are Mamba, an AI assistant built on a Hybrid Mamba-3 architecture with Tucker-decomposed Mixture-of-Experts, running locally on Apple Silicon."

# ── FastAPI ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="Mamba3-XR PyTorch Reference Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/status")
async def status():
    return JSONResponse({
        "ready":     _ready,
        "mock":      _MOCK_MODE,
        "device":    str(_device) if _device else None,
        "dtype":     str(_dtype)  if _dtype  else None,
        "load_info": _load_info,
    })


@app.get("/api/config")
async def config():
    if not _cfg:
        return JSONResponse({"error": "not loaded"}, status_code=503)
    return JSONResponse({
        "d_model":     _cfg.d_model,
        "n_heads":     _cfg.n_heads,
        "n_total":     _cfg.n_total,
        "n_mamba":     _cfg.n_mamba,
        "num_layers":  _cfg.num_layers,
        "vocab_size":  _cfg.vocab_size,
        "kmoe_num_experts": _cfg.kmoe_num_experts,
    })


@app.post("/api/bench")
async def bench(request: Request):
    """Run N decode steps (greedy) and return tok/s."""
    if not _ready:
        return JSONResponse({"error": "model not ready"}, status_code=503)
    body  = await request.json()
    steps = int(body.get("steps", 32))

    def _run():
        import time
        states  = _model.init_states(kv_len=256)
        tok_ids = [1] * 4                       # short dummy prompt
        with torch.no_grad():
            logits, states = _model.prefill(tok_ids, kv_len=256)
        _sync()
        tid = int(logits.argmax())
        t0  = time.perf_counter()
        for _ in range(steps):
            logits, states = _model.decode_step(tid, states)
            tid = int(logits.argmax())
        _sync()
        return steps / (time.perf_counter() - t0)

    loop = asyncio.get_event_loop()
    tps  = await loop.run_in_executor(None, _run)
    return JSONResponse({
        "backend":    "pytorch",
        "device":     str(_device),
        "dtype":      str(_dtype),
        "steps":      steps,
        "decode_tps": round(tps, 1),
    })


@app.post("/api/stream")
async def stream_generate(request: Request):
    """
    Server-Sent Events streaming generation.

    Body JSON: { prompt, max_tokens, temperature, top_k, category_key, seed }

    SSE events: token | done | error
    """
    if _MOCK_MODE:
        return _mock_stream()

    if not _ready:
        return JSONResponse({"error": "model not ready"}, status_code=503)

    body  = await request.json()
    prompt      = body.get("prompt", "Who are you?")
    max_tokens  = int(body.get("max_tokens", 256))
    temperature = float(body.get("temperature", 0.426))
    top_k       = int(body.get("top_k", 20))
    top_p       = float(body.get("top_p", 0.981))
    min_p       = float(body.get("min_p", 0.067))
    rep_pen     = float(body.get("rep_pen", 1.146))
    pres_pen    = float(body.get("pres_pen", 0.143))
    freq_pen    = float(body.get("freq_pen", 0.133))
    seed        = int(body.get("seed", 0))
    sys_prompt  = body.get("system_prompt", DEFAULT_SYS)

    full_prompt = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    bos    = _tokenizer.token_to_id("<s>") or 1
    ids    = [bos] + _tokenizer.encode(full_prompt, add_special_tokens=False).ids
    stop_t = []
    for n in ("<|im_end|>", "</s>"):
        t = _tokenizer.token_to_id(n)
        if t is not None:
            stop_t.append(t)

    gen_cfg = GenConfig(
        max_tokens=max_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, min_p=min_p,
        rep_pen=rep_pen, pres_pen=pres_pen, freq_pen=freq_pen, seed=seed,
    )

    def _sse_gen():
        for ev in gen_stream(_model, ids, gen_cfg, _tokenizer,
                             stop_token_ids=tuple(stop_t)):
            yield f"data: {json.dumps(ev)}\n\n"
        yield "data: [DONE]\n\n"

    # Run blocking generation in thread pool
    queue: asyncio.Queue = asyncio.Queue()

    def _worker():
        for ev in gen_stream(_model, ids, gen_cfg, _tokenizer,
                             stop_token_ids=tuple(stop_t)):
            asyncio.run_coroutine_threadsafe(queue.put(ev), loop)
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    loop = asyncio.get_event_loop()
    threading.Thread(target=_worker, daemon=True).start()

    async def _sse():
        while True:
            ev = await queue.get()
            if ev is None:
                yield "data: [DONE]\n\n"
                break
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(
        _sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _mock_stream():
    import asyncio
    mock_text = (
        "I'm Mamba — a local AI assistant built on a Hybrid Mamba-3 architecture "
        "with Tucker-decomposed Mixture-of-Experts. I run entirely on your device "
        "using Apple Silicon, with no cloud routing or network latency."
    )
    words = mock_text.split()

    async def _gen():
        yield f"data: {json.dumps({'type':'meta','prefill_tps':0,'n_prompt':10})}\n\n"
        await asyncio.sleep(0.05)
        for i, w in enumerate(words):
            text = w + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type':'token','id':i,'text':text,'tok_s':18.0})}\n\n"
            await asyncio.sleep(1 / 18)
        yield f"data: {json.dumps({'type':'done','total_tokens':len(words),'tok_s':18.0})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


# ── Model loading ─────────────────────────────────────────────────────────────

def _sync():
    if _device and _device.type == "mps":
        torch.mps.synchronize()
    elif _device and _device.type == "cuda":
        torch.cuda.synchronize()


def _load_model(ckpt: str, tok_path: str, device_str: str):
    global _model, _tokenizer, _device, _dtype, _cfg, _ready, _load_info

    if device_str == "auto":
        if torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device_str)

    # MPS: float16; CPU/CUDA: float32
    dt = torch.float16 if dev.type == "mps" else torch.float32

    print(f"[pt-server] device={dev}  dtype={dt}")
    _device = dev
    _dtype  = dt

    print(f"[pt-server] loading tokenizer …", flush=True)
    _tokenizer = Tokenizer.from_file(tok_path)

    print(f"[pt-server] building model …", flush=True)
    t0  = time.perf_counter()
    cfg = Mamba3Config(vocab_size=_tokenizer.get_vocab_size())
    _cfg = cfg
    mdl = Mamba3LM(cfg).to(device=dev, dtype=dt)

    print(f"[pt-server] loading weights …", flush=True)
    tw = time.perf_counter()
    load_checkpoint(mdl, ckpt, dtype=dt)
    tw = time.perf_counter() - tw

    # Warmup
    print(f"[pt-server] warming up …", flush=True)
    mdl.eval()
    with torch.no_grad():
        _, st = mdl.prefill([1, 2, 3], kv_len=256)
        _, st = mdl.decode_step(4, st)
    _sync()

    total = time.perf_counter() - t0
    _model = mdl
    _ready = True
    _load_info = {
        "device":    str(dev),
        "dtype":     str(dt),
        "weight_s":  round(tw, 2),
        "total_s":   round(total, 2),
    }
    print(f"[pt-server] ready in {total:.1f}s  (weights {tw:.1f}s)", flush=True)


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser(description="PyTorch Mamba3 server")
    ap.add_argument("--port",       type=int, default=7861)
    ap.add_argument("--device",     default="auto")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer",  default=DEFAULT_TOK)
    ap.add_argument("--mock",       action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = _parse_args()
    _MOCK_MODE = args.mock

    if not _MOCK_MODE:
        t = threading.Thread(
            target=_load_model,
            args=(args.checkpoint, args.tokenizer, args.device),
            daemon=True,
        )
        t.start()
    else:
        _ready = True
        print("[pt-server] mock mode — no weights loaded")

    print(f"[pt-server] starting on http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
