#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mamba3-XR Chat Demo — WebSocket real-time streaming chat (mamba3_mlx native stack).

Rewritten on top of the native MLX inference stack at ``mamba3_mlx/``:
  * Model      → ``mamba3_mlx.mlx_model.hybrid_model.Mamba3LanguageModel``
  * Weights    → ``mamba3_mlx.mlx_model.weights.load_checkpoint`` (.npz)
  * Sampling   → ``mamba3_mlx.inference.sampler``
  * Middleware → ``mamba3_mlx/mv/cot_middleware`` (FSM + format guard + budget +
                 multi-stage ``<final>`` injection)

Compared to the old version this drops the KV-padding / primed-prefix cache
machinery (those required fixed-shape compiled graphs from the
``inference/lib/`` stack that doesn't exist under ``mamba3_mlx/``).  Single-
turn correctness is preserved; multi-turn re-runs prefill each call.

Usage:
  python -m mamba3_mlx.chat_demo                                # defaults
  python -m mamba3_mlx.chat_demo --port 8080                    # custom port
  python -m mamba3_mlx.chat_demo --checkpoint path/to.npz       # weights
  python -m mamba3_mlx.chat_demo --mock                         # UI-only

Requires: fastapi, uvicorn, transformers, mlx, tokenizers.
"""


from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import random
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

# ── Tier-1 perf: faster JSON encoder + libuv event loop ───────────────────────
# orjson is ~3-4x faster than stdlib json.dumps and emits bytes directly
# (no UTF-8 round-trip). uvloop replaces asyncio's selector loop with one
# backed by libuv, cutting per-await scheduling overhead.  Both fall back
# gracefully if the package is not installed.
try:
    import orjson  # type: ignore
    def _ws_dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode("utf-8")
except Exception:
    orjson = None  # type: ignore[assignment]
    def _ws_dumps(obj: Any) -> str:
        return json.dumps(obj, separators=(",", ":"))

try:
    import uvloop  # type: ignore
    _UVLOOP_AVAILABLE = True
except Exception:
    uvloop = None  # type: ignore[assignment]
    _UVLOOP_AVAILABLE = False

_INF_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_INF_DIR, ".."))
# Ensure `from mamba3_mlx.*` works when launched directly.
for _p in (_REPO_ROOT, os.path.join(_INF_DIR, "mv")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx.core as mx

from mamba3_mlx import chat_config as _cfg
from mamba3_mlx.utils.config import Mamba3Config, GenerationConfig
from mamba3_mlx.utils.system_prompts import resolve_system_prompt
from mamba3_mlx.inference.generator import generate as _gen_pure
from mamba3_mlx.utils.mode_configs import get_mode_gen_config
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint, _sidecar_path
from mamba3_mlx.inference.sampler import (
    sample_logits,
    apply_repetition_penalty,
    apply_freq_presence_penalty,
    make_compiled_sampler,
)
from cot_middleware import (  # noqa: E402 (sys.path tweak above)
    CotMiddleware,
    CotMiddlewareConfig,
    CotMiddlewareDeps,
    render_health_line,
)
# profiler bridge intentionally not imported here — runs as a separate process
# to avoid any impact on inference throughput.

# ── Load-progress tracking (for browser overlay) ─────────────────────────────
_load_stage:    str = "Initialising…"
_load_progress: int = 0          # 0-100 integer
_static_bench_tps: float | None = None   # cached pure-compute decode speed


def _set_stage(stage: str, pct: int, *, bar: bool = True) -> None:
    """Update load stage + progress. Also prints a terminal progress bar."""
    global _load_stage, _load_progress
    _load_stage    = stage
    _load_progress = max(0, min(100, pct))
    if bar:
        filled = _load_progress // 5          # 20-char bar
        empty  = 20 - filled
        print(f"\r  [{'█' * filled}{'░' * empty}] {_load_progress:3d}%  {stage:<40}", end="", flush=True)


# ── PyTorch reference backend (for /speed-race) ──────────────────────────────
_pt_model    = None
_pt_tok      = None
_pt_ready    = False
_pt_info:    dict = {}


def _load_pytorch_backend(checkpoint: str, tokenizer_path: str) -> None:
    """Load PyTorch reference model in background (non-blocking startup)."""
    global _pt_model, _pt_tok, _pt_ready, _pt_info
    try:
        import torch
        from mamba3_mlx.pure_torch_mlx.config  import Mamba3Config as PTConfig
        from mamba3_mlx.pure_torch_mlx.model   import Mamba3LM
        from mamba3_mlx.pure_torch_mlx.weights import load_checkpoint as pt_load
        from tokenizers import Tokenizer as HFTok
        import time as _time

        # Resolve tokenizer: accept a .json file OR a directory containing tokenizer.json
        tok_path = Path(tokenizer_path)
        if tok_path.is_dir():
            tok_json = tok_path / "tokenizer.json"
        else:
            tok_json = tok_path
        if not tok_json.exists():
            raise FileNotFoundError(f"tokenizer.json not found at {tok_json}")

        # Use the original .npz (not MLX sidecar) for numpy loading
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists() or ckpt_path.is_dir():
            # Fall back to looking for the base .npz alongside the sidecar
            for candidate in ckpt_path.parent.glob("*.npz"):
                if "mlx_bf16" not in candidate.name:
                    ckpt_path = candidate
                    break
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        dev   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if dev.type == "mps" else torch.float32
        print(f"[pt-ref] loading weights on {dev} ({dtype})…", flush=True)
        t0   = _time.perf_counter()
        ptok = HFTok.from_file(str(tok_json))
        cfg  = PTConfig(vocab_size=ptok.get_vocab_size())
        mdl  = Mamba3LM(cfg).to(device=dev, dtype=dtype)
        pt_load(mdl, str(ckpt_path), dtype=dtype)
        mdl.eval()
        # Warm-up: 3 rounds to fully settle MPS command-buffer JIT
        print(f"[pt-ref] warming up MPS pipeline (3 rounds)…", flush=True)
        with torch.no_grad():
            for _wi in range(3):
                _, _st = mdl.prefill([1, 2, 3, 4, 5], kv_len=128)
                for _ in range(8):
                    _, _st = mdl.decode_step(1, _st)
                if dev.type == "mps":
                    torch.mps.synchronize()
        print(f"[pt-ref] warmup done", flush=True)
        elapsed = _time.perf_counter() - t0
        _pt_model = mdl
        _pt_tok   = ptok
        _pt_ready = True
        _pt_info  = {"device": str(dev), "dtype": str(dtype), "load_s": round(elapsed, 1)}
        print(f"[pt-ref] ready in {elapsed:.1f}s  ({dev}, {dtype})", flush=True)
    except Exception as exc:
        print(f"[pt-ref] WARN: could not load PyTorch reference: {exc}", flush=True)


_CATEGORY_PROMPTS_MERGE: Any = None


def _category_system_prompts_map(data: dict[str, Any]) -> dict[str, str]:
    """Per sidebar category key → short SFT/export string."""
    global _CATEGORY_PROMPTS_MERGE
    if _CATEGORY_PROMPTS_MERGE is None:
        try:
            _cp = os.path.join(_REPO_ROOT, "cot_dataset")
            if _cp not in sys.path:
                sys.path.insert(0, _cp)
            from category_system_prompts import merged_category_prompts_for_api as _m

            _CATEGORY_PROMPTS_MERGE = _m
        except Exception:
            _CATEGORY_PROMPTS_MERGE = False
    if callable(_CATEGORY_PROMPTS_MERGE):
        return dict(_CATEGORY_PROMPTS_MERGE(data))
    return {}


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_model: Mamba3LanguageModel | None = None
_tokenizer: Any = None
_args: Any = None
_config: Any = None
_stop_ids: frozenset[int] | None = None
_mw_deps: CotMiddlewareDeps | None = None
_mw_cfg: CotMiddlewareConfig | None = None
_extra_ban_mask: mx.array | None = None     # </final> ban for think/between
_final_ban_mask: mx.array | None = None     # <think>/<final> ban inside final block
_think_early_ban: mx.array | None = None    # </think> ban for first N think tokens
_direct_head_ban: mx.array | None = None    # <think> ban in head mode when reasoning=False
_model_ready = False
_vocab_size: int = 0
_MOCK_MODE: bool = False
# Compiled-sampler cache keyed by (rep,pres,freq,temp,top_k,top_p,min_p,V) so
# repeated requests with the same sampling params reuse one mx.compile graph
# instead of re-tracing per generation.
_SAMPLER_CACHE: dict = {}


def _get_compiled_sampler(rep, pres, freq, temp, topk, topp, minp, V):
    k = (rep, pres, freq, temp, topk, topp, minp, V)
    s = _SAMPLER_CACHE.get(k)
    if s is None:
        s = make_compiled_sampler(rep, pres, freq, temp, topk, topp, minp, V)
        _SAMPLER_CACHE[k] = s
    return s
# Fast forward path: StaticDecoder streaming session (metal_fuse + selective
# q8 + norm_fold) — replaces only the per-token model forward; sampling /
# penalties / CoT middleware stay on the exact same Python code path.
_static_decoder: Any = None

# Rich console for live terminal output (token streaming + injection markers).
try:
    from rich.console import Console as _RichConsole
    _console = _RichConsole(highlight=False)
except ImportError:
    _console = None


def _cprint(text: str, style: str = "", end: str = "") -> None:
    if _console is not None:
        _console.print(text, style=style, end=end, highlight=False)
    else:
        print(text, end=end, flush=True)


# Inference lock — MLX Metal is single-stream.
_infer_lock = asyncio.Lock()

# Default decode ceiling — set in chat_config.py.
DEFAULT_MAX_NEW_TOKENS = _cfg.MAX_NEW_TOKENS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_ui_config() -> dict[str, Any]:
    try:
        p = Path(__file__).parent / "ui" / "mock_config.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _iter_state_arrays(states: Any) -> list:
    out: list = []
    if states is None:
        return out
    for st in states:
        if st is None:
            continue
        for v in st.values():
            if v is not None:
                out.append(v)
    return out


def _free_metal_cache() -> None:
    fn = getattr(mx, "clear_cache", None)
    if fn is None:
        fn = getattr(getattr(mx, "metal", None), "clear_cache", None)
    if fn is None:
        return
    try:
        fn()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------
def _tok_id(tokenizer: Any, name: str) -> int:
    try:
        t = tokenizer.convert_tokens_to_ids(name)
        return int(t) if isinstance(t, int) and t >= 0 else -1
    except Exception:
        return -1


def _build_ban_mask(token_ids: list[int], vocab_size: int) -> mx.array | None:
    ids = [t for t in token_ids if t >= 0]
    if not ids:
        return None
    m = mx.zeros((vocab_size,), dtype=mx.float32)
    for t in ids:
        m[t] = -1e9
    mx.eval(m)
    return m


def _build_extra_ban_mask(tokenizer: Any, vocab_size: int) -> mx.array | None:
    """Bans structural tags in think/between modes.

    * </final> — prevents early loop exit before final mode
    * <final>  — prevents model jumping to final while still in think
    * <think>  — prevents model re-opening a think block inside think
    """
    ids = [_tok_id(tokenizer, t) for t in ("</final>", "<final>", "<think>")]
    return _build_ban_mask(ids, vocab_size)


def _build_final_ban_mask(tokenizer: Any, vocab_size: int) -> mx.array | None:
    """Bans <think> and <final> inside the final block.

    After the middleware injects <final>, the model sometimes regenerates
    **<think> or a second <final> tag as literal text.  Banning those token
    IDs in final mode prevents that re-entry loop.
    """
    ids = [_tok_id(tokenizer, n) for n in ("<think>", "<final>", "</think>")]
    return _build_ban_mask(ids, vocab_size)


def _prewarm_identity(n: int = 5) -> float:
    """Run N short generations to stabilise Metal GPU bf16 arithmetic for identity mode.

    5 prior sequences (max_tokens=20) is enough to warm the Metal JIT/GPU power
    state so seed=26/temp=0.25 → "I am Mamba" on the first real request (~3s).
    """
    sys_prompt = resolve_system_prompt("self_awareness", "")
    prompt_text = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\nWho are you?<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    bos_id = _tokenizer.convert_tokens_to_ids("<s>")
    ids: list[int] = _tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if isinstance(bos_id, int) and bos_id >= 0:
        ids = [bos_id] + ids

    stop_ids_pw: list[int] = []
    for nm in ("<|im_end|>", "</s>"):
        tid = _tokenizer.convert_tokens_to_ids(nm)
        if isinstance(tid, int) and tid >= 0:
            stop_ids_pw.append(tid)

    t0 = time.perf_counter()
    print(f"[chat_demo] identity pre-warm: {n} sequences ...", end="", flush=True)
    for i in range(n):
        gc = GenerationConfig(
            max_tokens=20, temperature=0.25, top_k=60,
            top_p=0.856, min_p=0.122, rep_pen=1.243,
            pres_pen=0.306, freq_pen=0.031,
            repeat_last_n=256, seed=i,
        )
        _gen_pure(_model, ids, gc, stop_token_ids=stop_ids_pw, no_eos_stop=True)
    elapsed = time.perf_counter() - t0
    print(f" done ({elapsed:.1f}s)")
    return elapsed * 1000


def _load_model(args: argparse.Namespace) -> dict[str, Any]:
    global _model, _tokenizer, _args, _config, _stop_ids
    global _mw_deps, _mw_cfg, _extra_ban_mask, _final_ban_mask, _think_early_ban, _direct_head_ban, _model_ready, _vocab_size

    _args = args
    timings: dict[str, Any] = {}
    t0 = time.perf_counter()

    print()   # newline so the bar starts on its own line

    # ── Stage 1: Tokenizer ────────────────────────────────────────────────────
    _set_stage("Loading tokenizer…", 5)
    dtype_map = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
    target_dtype = dtype_map[args.dtype]

    from transformers import AutoTokenizer
    _tok_path = args.tokenizer
    if _tok_path.endswith(".json"):
        _tok_path = str(Path(_tok_path).parent)
    _tokenizer = AutoTokenizer.from_pretrained(_tok_path, trust_remote_code=True)
    _vocab_size = len(_tokenizer) if args.vocab_size <= 0 else args.vocab_size
    timings["tokenizer_ms"] = (time.perf_counter() - t0) * 1000

    # ── Stage 2: Model init ───────────────────────────────────────────────────
    _set_stage("Initialising model graph…", 12)
    t1 = time.perf_counter()
    _config = Mamba3Config(vocab_size=_vocab_size)
    _model = Mamba3LanguageModel(_config)
    timings["model_init_ms"] = (time.perf_counter() - t1) * 1000

    # ── Stage 3: Weights ──────────────────────────────────────────────────────
    t2 = time.perf_counter()
    if args.checkpoint:
        sidecar = _sidecar_path(args.checkpoint, target_dtype)
        if sidecar.exists():
            load_kind = "mmap-instant"
            _set_stage(f"Loading weights (mmap)…  {sidecar.name}", 20)
        else:
            load_kind = "convert+save"
            _set_stage(f"Converting weights → {Path(sidecar).name}…", 20)

        # Animate progress during the blocking load (20 → 70 %)
        _weight_done = threading.Event()
        def _weight_anim():
            pct = 20
            while not _weight_done.is_set():
                time.sleep(0.4)
                if pct < 68:
                    pct += 2
                _set_stage(_load_stage, pct)
        threading.Thread(target=_weight_anim, daemon=True).start()

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            load_checkpoint(_model, args.checkpoint, dtype=target_dtype)
        _weight_done.set()

        timings["checkpoint"] = args.checkpoint
        timings["kind"] = load_kind
        args.checkpoint_label = Path(args.checkpoint).stem
    else:
        _set_stage("Random weights (smoke test)", 20)
        timings["checkpoint"] = "(random init)"
        timings["kind"] = "none"
        args.checkpoint_label = "Mamba3-XR"

    _set_stage("Evaluating parameters…", 72)
    mx.eval(_model.parameters())
    timings["weights_ms"] = (time.perf_counter() - t2) * 1000

    # ── Stage 4: Middleware ───────────────────────────────────────────────────
    _set_stage("Setting up CoT middleware…", 76)

    # Middleware deps (tokenizer ids + bias vectors).
    existing_stop: list[int] = []
    for name in ("<|im_end|>", "</s>"):
        try:
            tid = _tokenizer.convert_tokens_to_ids(name)
        except Exception:
            tid = -1
        if isinstance(tid, int) and tid >= 0:
            existing_stop.append(int(tid))

    _mw_cfg = CotMiddlewareConfig.config_from_args(args)
    args.close_bias_start = _mw_cfg.close_bias_start
    _mw_deps = CotMiddlewareDeps.build(
        tokenizer=_tokenizer,
        vocab_size=_vocab_size,
        existing_stop_ids=existing_stop,
        cfg=_mw_cfg,
    )
    _stop_ids = _mw_deps.stop_ids
    _extra_ban_mask = _build_extra_ban_mask(_tokenizer, _vocab_size)
    _final_ban_mask = _build_final_ban_mask(_tokenizer, _vocab_size)
    # Ban </think> for the first THINK_MIN_TOKENS think tokens so the model
    # can't prematurely close its chain-of-thought mid-sentence.
    _think_close_id = -1
    _think_open_id = -1
    try:
        _think_close_id = int(_tokenizer.convert_tokens_to_ids("</think>"))
        _think_open_id  = int(_tokenizer.convert_tokens_to_ids("<think>"))
    except Exception:
        pass
    if 0 <= _think_close_id < _vocab_size:
        _think_early_ban = mx.zeros((_vocab_size,), dtype=mx.float32).at[_think_close_id].add(-1e9)
    # Ban <think> in head mode when reasoning=False so the model can't
    # spontaneously enter a think block (SFT training makes this happen).
    if 0 <= _think_open_id < _vocab_size:
        _direct_head_ban = mx.zeros((_vocab_size,), dtype=mx.float32).at[_think_open_id].add(-1e9)
    print(f"[chat_demo] {_mw_deps.describe()}")
    if _extra_ban_mask is not None:
        print("[chat_demo] ban[think/between]: </final> <final> <think>")
    if _final_ban_mask is not None:
        print("[chat_demo] ban[final]:         <think> </think> <final>  (prevents re-entry loop)")
    if _think_early_ban is not None:
        print(f"[chat_demo] ban[think-early]:   </think> for first {_cfg.THINK_MIN_TOKENS} think tokens")
    if _direct_head_ban is not None:
        print("[chat_demo] ban[direct-head]:   <think> in head mode when reasoning=False")

    # ── Stage 5: Warmup ───────────────────────────────────────────────────────
    # Skipped when static_stream is on: this reference-path forward perturbs
    # Metal bf16 state and breaks tuned seeds. `make self-s` (run.py --static)
    # skips it too — keeping chat's identity fastpath numerically identical.
    _set_stage("JIT warmup…", 84)
    t3 = time.perf_counter()
    warm = mx.array([[0]], dtype=mx.int32)
    if not getattr(args, "static_stream", False):
        for _ in range(max(0, int(args.warmup))):
            lo, st = _model(warm, states=None)
            mx.eval(lo, *_iter_state_arrays(st))
    timings["warmup_ms"] = (time.perf_counter() - t3) * 1000

    # ── Stage 6: StaticDecoder compile + benchmark ───────────────────────────
    global _static_decoder, _static_bench_tps
    _static_decoder = None
    if getattr(args, "static_stream", False):
        _q_on = any(int(getattr(args, k, 0) or 0) for k in ("quant_moe", "quant_proj", "quant_head"))
        _set_stage(f"Compiling StaticDecoder (metal_fuse{' + q8' if _q_on else ', bf16 no-q8'})…", 88)
        t4 = time.perf_counter()
        try:
            from mamba3_mlx.mlx_model.static_decode import StaticDecoder
            from mamba3_mlx.utils.config import GenerationConfig as _GC
            _static_decoder = StaticDecoder(
                _model, metal_fuse=True,
                quant_moe_bits=int(getattr(args, "quant_moe", 0) or 0),
                quant_proj_bits=int(getattr(args, "quant_proj", 0) or 0),
                quant_head_bits=int(getattr(args, "quant_head", 0) or 0),
            )
            # NOTE: NO startup warmup / bench / start_stream pre-compile here.
            # Those accumulate Metal/MPS bf16 numeric state that shifts low-temp
            # logits and breaks tuned seeds (seed=26 → "I am Mamba"). The identity
            # fastpath uses decoder.generate on a CLEAN decoder — exactly like
            # `make self-s` (clean process). First request pays the one-time
            # compile cost (slower first token), matching self-s behaviour.
            timings["static_compile_ms"] = (time.perf_counter() - t4) * 1000
            _static_bench_tps = None
            print(f"\n[chat_demo] static-stream ready (clean, no warmup): "
                  f"metal_fuse + quant(moe={args.quant_moe},proj={args.quant_proj},"
                  f"head={args.quant_head}) — matches make self-s inference path")

        except Exception as exc:
            _static_decoder = None
            print(f"[chat_demo] WARN: static-stream unavailable, falling back "
                  f"to reference decode ({exc})")
    timings["total_ms"] = (time.perf_counter() - t0) * 1000

    _set_stage("Ready ✓", 100)
    print()   # newline after the progress bar
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


def _get_bos_id() -> int | None:
    """Return BOS token id if the tokenizer has one (run.py prepends it)."""
    try:
        bos = _tokenizer.bos_token_id
        if bos is not None and int(bos) > 0:
            return int(bos)
    except Exception:
        pass
    return None


def _build_multiturn_ids(
    history: list[dict],
    seq_len: int,
    system_prompt: str | None = None,
    reasoning: bool = False,
) -> list[int]:
    parts: list[str] = []
    sys_text = (system_prompt or "").strip()
    if sys_text:
        parts.append(f"<|im_start|>system\n{sys_text}<|im_end|>\n")
    for msg in history:
        role = msg["role"]
        content = msg["content"].strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    if reasoning:
        parts.append("<think>\n")
    full_prompt = "".join(parts)
    ids = _encode_plain(full_prompt)
    if seq_len > 0 and len(ids) > seq_len:
        ids = ids[-seq_len:]
    if not ids:
        tail = "<|im_start|>assistant\n" + ("<think>\n" if reasoning else "")
        ids = _encode_plain(tail)
    # Prepend BOS token to match run.py's build_chatml_prompt (id=1).
    # Without BOS the Mamba SSM prefill state diverges, breaking tuned seeds.
    bos = _get_bos_id()
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids
    return ids


# ---------------------------------------------------------------------------
# Chunked prefill — processes prompt in slices to cap peak activation memory.
#
# Root cause of OOM on long prompts:
#   chunk_scan builds h_intra of shape (B, nc, Lc, H, N, P) where nc=L/chunk_size.
#   At L=2048 this is ~400 MB per Mamba block.  By splitting the prefill into
#   _PREFILL_CHUNK-token slices we bound nc to _PREFILL_CHUNK/chunk_size regardless
#   of total prompt length, and we avoid materialising the full [1, L, vocab]
#   logits tensor (only last_logits[vocab] is needed).
# ---------------------------------------------------------------------------
_PREFILL_CHUNK = 256  # tokens per model call during prefill


def _prefill_chunked(prompt_ids: list[int]) -> "tuple[mx.array, Any]":
    """Run prefill in _PREFILL_CHUNK-token slices and return (last_logits, states)."""
    states = None
    n = len(prompt_ids)
    for start in range(0, n, _PREFILL_CHUNK):
        chunk = prompt_ids[start : start + _PREFILL_CHUNK]
        x = mx.array([chunk], dtype=mx.int32)
        logits, states = _model(x, states=states)
        is_last = (start + _PREFILL_CHUNK) >= n
        if is_last:
            last_row = logits[0, -1]  # only materialise the single vocab row
            mx.eval(last_row, *_iter_state_arrays(states))
            del logits, x
            return last_row, states
        # intermediate chunk: eval states, discard logits immediately
        mx.eval(*_iter_state_arrays(states))
        del logits, x
    raise RuntimeError("_prefill_chunked: empty prompt_ids")


# ---------------------------------------------------------------------------
# Identity fast-path — mirror `make self-s` (decoder.generate) exactly so
# tuned seeds reproduce in chat. Runs decoder.generate in a worker thread and
# streams tokens through a queue (decoder.generate is blocking + callback-based).
# ---------------------------------------------------------------------------
def _stream_identity_fastpath(
    prompt_ids: list[int],
    sample_args: Any,
    max_tokens: int,
    abort_event: "threading.Event | None" = None,
) -> Iterator[dict]:
    import queue as _queue
    from mamba3_mlx.utils.config import GenerationConfig as _GC

    gc = _GC(
        # KV buffer must stay ≤768 (max_tokens≤512) — larger buffers change the
        # StaticDecoder bf16 path and break tuned seeds (see project memory).
        max_tokens=min(int(max_tokens), 512),
        temperature=float(getattr(sample_args, "temp", 0.25)),
        top_k=int(getattr(sample_args, "top_k", 60)),
        top_p=float(getattr(sample_args, "top_p", 0.856)),
        min_p=float(getattr(sample_args, "min_p", 0.122)),
        rep_pen=float(getattr(sample_args, "rep_pen", 1.243)),
        pres_pen=float(getattr(sample_args, "pres_pen", 0.306)),
        freq_pen=float(getattr(sample_args, "freq_pen", 0.031)),
        # Force 256 to match `make self-s` (REPEAT_LAST_N=256). chat_config
        # defaults to 128 → different penalty window → tuned seeds break.
        repeat_last_n=256,
        seed=int(getattr(sample_args, "seed", 26) or 0),
    )

    # Frontend expects a meta event before tokens (prefill is done inside
    # decoder.generate; emit an approximate meta up-front).
    yield {
        "type": "meta",
        "prefill_ms": 0.0,
        "prefill_tps": 0.0,
        "prompt_tokens": len(prompt_ids),
        "turns": 1,
    }

    q: "_queue.Queue" = _queue.Queue()
    _SENTINEL = object()
    holder: dict[str, Any] = {}

    def _on_tok(tid: int) -> None:
        q.put(int(tid))

    def _run() -> None:
        try:
            holder["r"] = _static_decoder.generate(
                prompt_ids, gc, stop_token_ids=tuple(_stop_ids or ()),
                on_token=_on_tok,
            )
        except Exception as exc:  # noqa: BLE001
            holder["err"] = exc
        finally:
            q.put(_SENTINEL)

    t_dec = time.perf_counter()
    th = threading.Thread(target=_run, daemon=True)
    th.start()

    seen: list[int] = []
    prev_text = ""
    ttft_ms: float | None = None
    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t_dec) * 1000
        seen.append(item)
        try:
            full = _tokenizer.decode(seen, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        except Exception:
            continue
        delta = full[len(prev_text):]
        prev_text = full
        if delta:
            el = max(time.perf_counter() - t_dec, 1e-9)
            _cprint(delta, style="green")
            yield {"type": "token", "text": delta, "n": len(seen),
                   "tok_s": round(len(seen) / el, 1)}
        if abort_event is not None and abort_event.is_set():
            break
    th.join(timeout=1.0)

    r = holder.get("r")
    if holder.get("err") is not None:
        yield {"type": "error", "message": f"identity fastpath: {holder['err']}"}
    dec_ms = (time.perf_counter() - t_dec) * 1000
    n_out = len(seen)
    yield {
        "type": "done",
        "total_tokens": n_out,
        "total_ms": round(dec_ms, 2),
        "tok_s": round(n_out / max(dec_ms / 1000, 1e-9), 1),
        "prefill_tps": round(getattr(r, "prefill_tps", 0.0), 1) if r else 0.0,
        "ttft_ms": round(ttft_ms or 0.0, 2),
        "prefill_ms": round(getattr(r, "elapsed_prefill", 0.0) * 1000, 2) if r else 0.0,
    }


# ---------------------------------------------------------------------------
# Streaming generator (native MLX + middleware)
# ---------------------------------------------------------------------------
def _stream_generate(
    history: list[dict],
    max_tokens: int = 512,
    system_prompt: str | None = None,
    reasoning: bool = False,
    args_for_call: Any = None,
    no_eos_stop: bool = False,
    abort_event: "threading.Event | None" = None,
) -> Iterator[dict]:
    """Yield WS-shaped event dicts (meta / reasoning / token / done)."""
    assert _model_ready and _model is not None and _mw_deps is not None

    sample_args = args_for_call if args_for_call is not None else _args
    sys_text = (system_prompt or "").strip()

    # ── Prefill ─────────────────────────────────────────────────────────────
    prompt_ids = _build_multiturn_ids(history, _args.seq_len, sys_text, reasoning)

    # ── Identity fast-path: EXACT decoder.generate call as `make self-s` ──────
    # raw_sampling modes (self_awareness) are tuned against decoder.generate
    # (single-shot prefill + compiled loop + _make_sampler). The chat's chunked
    # prefill + start_stream + custom sampling is a different bf16 path and
    # breaks tuned seeds (e.g. seed=26 → "I am Mamba"). For these modes we run
    # the identical decoder.generate and stream tokens via on_token, so the chat
    # reproduces self-s exactly. (Bypasses middleware — fine, SFT emits <final>.)
    if (bool(getattr(sample_args, "raw_sampling", False))
            and _static_decoder is not None):
        yield from _stream_identity_fastpath(prompt_ids, sample_args,
                                             max_tokens, abort_event)
        return

    t_pre = time.perf_counter()
    last_logits, states = _prefill_chunked(prompt_ids)
    prefill_ms = (time.perf_counter() - t_pre) * 1000

    _prefill_tps = round(len(prompt_ids) / max(prefill_ms / 1000, 1e-6), 1)
    yield {
        "type": "meta",
        "prefill_ms": round(prefill_ms, 2),
        "prefill_tps": _prefill_tps,
        "prompt_tokens": len(prompt_ids),
        "turns": len([m for m in history if m["role"] == "user"]),
    }

    # ── Static streaming session (fast forward path) ────────────────────────
    # Wraps ONLY the model forward in the compiled static graph; the token
    # loop below (masks, penalties, sampling, middleware FSM) is unchanged.
    _sess = None
    if _static_decoder is not None:
        try:
            _sess = _static_decoder.start_stream(
                states, len(prompt_ids),
                max_new=int(max_tokens) + 64)
        except Exception as exc:
            print(f"[chat_demo] WARN: static session failed, reference path ({exc})")

    # ── Middleware (per-turn) ───────────────────────────────────────────────
    mw_cfg = _mw_cfg or CotMiddlewareConfig.config_from_args(_args)
    if getattr(sample_args, "format_guard_call_override", None) is False:
        mw_cfg = replace(mw_cfg, enabled=False)
    if getattr(sample_args, "force_final_inject_call_override", None) is False:
        mw_cfg = replace(mw_cfg, force_final_inject=False)
    def _model_apply(x: mx.array, ca: Any, sp: mx.array):
        # mamba3_mlx model ignores seq_pos / router_temp.
        if _sess is not None:
            # Feed the injected ids through the static session one by one;
            # middleware only consumes logits[0, -1, :].  ``ca`` is passed
            # through untouched (the session owns the real state).
            rows = [_sess.step(int(t)) for t in x[0].tolist()]
            return mx.stack(rows)[None, :], ca
        return _model(x, states=ca)

    mw = CotMiddleware(
        deps=_mw_deps,
        cfg=mw_cfg,
        reasoning=bool(reasoning),
        model_apply=_model_apply,
    )

    pos = len(prompt_ids)
    generated: list[int] = []
    all_tids: list[int] = []
    key = mx.random.key(int(getattr(sample_args, "seed", 0) or 0))
    recent_window = max(1, int(getattr(sample_args, "repeat_last_n", 64) or 64))

    # ── Compiled sampling chain ──────────────────────────────────────────────
    # penalties + temp/top-k/top-p/min-p + categorical fused into ONE compiled
    # dispatch (vs ~12 eager kernel launches + Python dict bookkeeping per
    # token).  Bit-identical to the eager path; ~2.2× faster on the stream demo.
    _samp_rep = float(getattr(sample_args, "rep_pen", 1.281))
    _samp_pres = float(getattr(sample_args, "pres_pen", 0.298))
    _samp_freq = float(getattr(sample_args, "freq_pen", 0.168))
    _samp_temp = float(getattr(sample_args, "temp", 0.236))
    _samp_topk = int(getattr(sample_args, "top_k", 30))
    _samp_topp = float(getattr(sample_args, "top_p", 0.959))
    _samp_minp = float(getattr(sample_args, "min_p", 0.122))
    _compiled_sampler = _get_compiled_sampler(
        _samp_rep, _samp_pres, _samp_freq, _samp_temp,
        _samp_topk, _samp_topp, _samp_minp, int(_vocab_size))
    # Fixed-shape ring buffer mirroring generated[-recent_window:] for the
    # vectorised window penalties (pad slots carry valid=0 → no effect).
    _ring_idx = mx.zeros((recent_window,), dtype=mx.int32)
    _ring_valid = mx.zeros((recent_window,), dtype=mx.float32)
    _ring_one = mx.ones((1,), dtype=mx.float32)
    # Warm the graph on a throwaway key so first-token latency isn't hit by the
    # one-time trace (the real `key` is left untouched).
    _wt, _ = _compiled_sampler(
        mx.zeros((int(_vocab_size),), dtype=mx.float32),
        _ring_idx, _ring_valid, mx.random.key(0))
    mx.eval(_wt)

    t_dec = time.perf_counter()
    elapsed_s_fn = lambda: time.perf_counter() - t_dec  # noqa: E731
    ttft_ms: float | None = None
    n_out = 0
    stop_after = False
    _raw_prev_text = ""   # cumulative decoded string for raw_sampling delta tracking

    # ── Optional per-section timing (set MAMBA_PROFILE_LOOP=1 to enable) ──
    _prof = os.environ.get("MAMBA_PROFILE_LOOP", "") in ("1", "true", "yes")
    _t_logits = _t_sample = _t_text = _t_route = _t_fwd = 0.0

    # When no_eos_stop is on, after the splitter enters "done" we keep decoding
    # tokens but bypass the splitter — text streams through `_nes_decode_and_yield`.
    _nes_bypass = False
    _nes_ids: list[int] = []
    _nes_prev_text = ""

    def _nes_decode_and_yield(tid: int) -> dict | None:
        nonlocal _nes_prev_text
        _nes_ids.append(tid)
        try:
            full = _tokenizer.decode(_nes_ids, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
            chunk = full[len(_nes_prev_text):] if full.startswith(_nes_prev_text) else \
                _tokenizer.decode([tid], skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
            _nes_prev_text = full
        except Exception:
            chunk = ""
        if not chunk:
            return None
        el = elapsed_s_fn()
        return {"type": "token", "text": chunk, "n": n_out,
                "tok_s": round(n_out / max(el, 1e-9), 1)}

    # ── Decode loop ─────────────────────────────────────────────────────────
    for _step in range(max_tokens):
        if _prof: _ts0 = time.perf_counter()
        # 1) Logit transform: middleware + script-level mode-specific bans.
        #    raw_sampling=True: skip ALL logit engineering so the path matches run.py
        #    (used for self_awareness where seed=27 is tuned against run.py's path).
        _raw = getattr(sample_args, "raw_sampling", False)
        if _raw:
            row = last_logits  # no ban masks, no close_bias
        else:
            #    think/between: ban </final> to prevent early loop exit.
            #    final:         ban <think>/<final>/</think> to prevent re-entry loop
            #                   where the model restarts the CoT structure after injection.
            row = mw.transform_logits(last_logits)
            if _extra_ban_mask is not None and mw.mode in ("think", "between"):
                row = row + _extra_ban_mask.astype(row.dtype)
            if _final_ban_mask is not None and mw.mode == "final":
                row = row + _final_ban_mask.astype(row.dtype)
            think_min = int(getattr(sample_args, "think_min_tokens", _cfg.THINK_MIN_TOKENS))
            if _think_early_ban is not None and mw.mode == "think" and mw._think_tokens < think_min:
                row = row + _think_early_ban.astype(row.dtype)
        # When reasoning is off, ban <think> in head mode so the SFT-trained model
        # can't spontaneously open a think block and consume the whole token budget.
        if not _raw and _direct_head_ban is not None and not reasoning and mw.mode == "head":
            row = row + _direct_head_ban.astype(row.dtype)
        z = row.astype(mx.float32)
        if _prof: _ts1 = time.perf_counter(); _t_logits += _ts1 - _ts0
        tok_arr, key = _compiled_sampler(z, _ring_idx, _ring_valid, key)
        # Advance the penalty ring (stays on GPU, no sync) — the window the
        # NEXT step sees, matching the eager `generated[-recent_window:]`.
        _ring_idx = mx.concatenate([_ring_idx[1:], tok_arr.reshape(1)])
        _ring_valid = mx.concatenate([_ring_valid[1:], _ring_one])
        # NOTE: no explicit ``mx.eval(tok_arr)`` — ``.item()`` below already
        # forces a CPU sync.  The redundant eval added ~50 µs/tok of
        # scheduling overhead.
        tid = int(tok_arr.item())
        generated.append(tid)
        all_tids.append(tid)
        n_out += 1
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t_dec) * 1000
        if _prof: _ts2 = time.perf_counter(); _t_sample += _ts2 - _ts1

        # ── Kick off the NEXT forward immediately (async) so the GPU runs
        #    in parallel with the Python-side text decode / FSM / WS push
        #    below.  ``_sess.step`` calls ``mx.async_eval`` internally; we
        #    deliberately do NOT call ``mx.eval(last_logits)`` afterwards —
        #    the next iteration's ``_compiled_sampler`` will materialise it
        #    lazily.  This is the difference between ~70 tok/s (serial
        #    GPU→Python→GPU) and ~85 tok/s (overlapped).
        #    Costs one wasted forward on the stop iteration (≈11 ms/turn),
        #    negligible vs the ~1 ms/tok overlap saving across the run.
        if _sess is not None:
            last_logits = _sess.step(tid)        # async-kicked compiled dispatch
            pos += 1
        else:
            x = mx.array([[tid]], dtype=mx.int32)
            logits_d, states = _model(x, states=states)
            last_logits = logits_d[0, -1]
            pos += 1
        if _prof: _ts_fwd = time.perf_counter(); _t_fwd += _ts_fwd - _ts2

        # 2a) Compute display text — raw uses cumulative decode for correct spacing;
        #     non-raw uses single-token decode (middleware handles its own tracking).
        prev_mode = mw.mode
        if _raw:
            try:
                _raw_full = _tokenizer.decode(
                    all_tids, skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                _raw_text = _raw_full[len(_raw_prev_text):]
                _raw_prev_text = _raw_full
            except Exception:
                _raw_text = ""
        else:
            try:
                _raw_text = _tokenizer.decode([tid], skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False)
            except Exception:
                _raw_text = ""

        # Live console: stream decoded token coloured by FSM mode.
        #   think/head  → dim grey  (reasoning in progress)
        #   final/done  → green     (answer tokens)
        #   between     → yellow    (transition zone)
        if _raw_text:
            _mode_now = mw.mode
            if _mode_now in ("head", "think"):
                _cprint(_raw_text, style="dim")
            elif _mode_now == "between":
                _cprint(_raw_text, style="yellow")
            else:
                _cprint(_raw_text, style="green")
        if _prof: _ts3 = time.perf_counter(); _t_text += _ts3 - _ts_fwd

        # 2b) Token routing — raw yields the cumulative delta directly.
        if _raw:
            if _raw_text:
                el = elapsed_s_fn()
                yield {
                    "type": "token",
                    "text": _raw_text,
                    "n": n_out,
                    "tok_s": round(n_out / max(el, 1e-9), 1),
                }
            if tid in set(_stop_ids or []):
                stop_after = True
        elif _nes_bypass:
            ev = _nes_decode_and_yield(tid)
            if ev:
                yield ev
        else:
            # Stream individual CoT tokens so the frontend can display them live.
            if mw.mode == "think":
                try:
                    _cot_chunk = _tokenizer.decode(
                        [tid], skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if _cot_chunk:
                        yield {"type": "reasoning_token", "text": _cot_chunk, "n": n_out}
                except Exception:
                    pass
            for evt in mw.step(tid, n_out=n_out, elapsed_s_fn=elapsed_s_fn):
                if evt.get("__stop__"):
                    if not no_eos_stop:
                        stop_after = True
                    else:
                        _nes_bypass = True
                else:
                    yield evt

        # 3) Stop checks.
        if abort_event and abort_event.is_set():
            break
        if stop_after:
            break
        if not no_eos_stop and mw.should_break(tid):
            stop_after = True
            break
        if _prof: _t_route += time.perf_counter() - _ts3

        # 5) Multi-stage <final> injection — fires at most once per turn.
        # raw_sampling: model generates <final> naturally (SFT), no injection needed.
        if not _raw and prev_mode == "think" and mw.mode == "between":
            states, pos, inj_row, did_inject, inj_ms = mw.maybe_inject_final(
                caches=states, pos=pos,
            )
            if did_inject and inj_row is not None:
                last_logits = inj_row
                prefill_ms += inj_ms
                mx.eval(last_logits)
                _cprint(" [mw:inject<final>] ", style="bold magenta")
                yield {
                    "type": "mw_inject",
                    "what": "<final>",
                    "n_out": n_out,
                    "inject_ms": round(inj_ms, 1),
                }

    for evt in mw.flush(n_out=n_out, elapsed_s_fn=elapsed_s_fn):
        yield evt
    elapsed = time.perf_counter() - t_dec
    streaming_tps = round(n_out / max(elapsed, 1e-9), 1)

    if _prof and n_out > 0:
        _total_us = (_t_logits + _t_sample + _t_text + _t_route + _t_fwd) * 1e6 / n_out
        print(f"\n[prof] per-token µs (n={n_out}): "
              f"logits={_t_logits*1e6/n_out:.0f} "
              f"sample={_t_sample*1e6/n_out:.0f} "
              f"text={_t_text*1e6/n_out:.0f} "
              f"route={_t_route*1e6/n_out:.0f} "
              f"fwd={_t_fwd*1e6/n_out:.0f} "
              f"sum={_total_us:.0f} | wall={elapsed/n_out*1e6:.0f} "
              f"({streaming_tps} tok/s)", flush=True)

    # ── bench_tps: use pre-measured cached value (measured at server startup) ──
    # _static_bench_tps is set once after StaticDecoder compiles using
    # generate() with full compiled loop — no JSON/asyncio overhead.
    done_evt = {
        "type": "done",
        "total_tokens": n_out,
        "total_ms": round(elapsed * 1000, 2),
        "tok_s": streaming_tps,
        "prefill_tps": _prefill_tps if '_prefill_tps' in dir() else 0,
        "ttft_ms": round(ttft_ms or 0.0, 2),
        "prefill_ms": round(prefill_ms, 2),
    }
    if getattr(sample_args, "bench_mode", False) and _static_bench_tps is not None:
        done_evt["bench_tps"] = _static_bench_tps
    yield done_evt

    _cprint("\n")  # end the streaming token line before the summary rule
    try:
        _print_turn_summary(
            history=history,
            system_prompt=sys_text,
            reasoning=bool(reasoning),
            prompt_tokens=len(prompt_ids),
            output_token_count=n_out,
            elapsed_s=elapsed,
            ttft_ms=ttft_ms,
            prefill_ms=prefill_ms,
            mw_health=mw.health_report(),
        )
    except Exception as _exc:
        print(f"[chat_demo] WARN: turn-summary print failed: {_exc}")

    try:
        del states
    except (NameError, UnboundLocalError):
        pass
    _free_metal_cache()


# ---------------------------------------------------------------------------
# Post-turn console dump (slim)
# ---------------------------------------------------------------------------
def _print_turn_summary(
    *,
    history: list[dict],
    system_prompt: str,
    reasoning: bool,
    prompt_tokens: int,
    output_token_count: int,
    elapsed_s: float,
    ttft_ms: float | None,
    prefill_ms: float,
    mw_health: dict[str, Any] | None = None,
) -> None:
    try:
        from rich.console import Console
    except Exception:
        return
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    con = Console(highlight=False)
    con.rule("[bold dim]chat turn complete[/]")
    con.print(
        f'  user: "{(last_user or "")[:80]}…"   reasoning={reasoning}\n'
        f'  metrics: prompt={prompt_tokens}t  out={output_token_count}t  '
        f'prefill={prefill_ms:.0f}ms  ttft={(ttft_ms or 0.0):.0f}ms  '
        f'total={elapsed_s * 1000:.0f}ms'
    )
    if mw_health is not None:
        con.print(f"  middleware: [ {render_health_line(mw_health)} ]", highlight=False)


# ---------------------------------------------------------------------------
# FastAPI + WebSocket
# ---------------------------------------------------------------------------
import pydantic
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Mamba3-XR Chat Demo (mamba3_mlx native)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_UI_DIR = Path(__file__).parent / "ui"
if _UI_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_UI_DIR)), name="static")

_load_timings: dict[str, Any] = {}


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (_UI_DIR / "templates" / "chat_demo.html").read_text(encoding="utf-8")
    try:
        js_v = int((_UI_DIR / "scripts" / "chat_demo.js").stat().st_mtime)
    except OSError:
        js_v = 0
    try:
        css_v = int((_UI_DIR / "styles" / "chat_demo.css").stat().st_mtime)
    except OSError:
        css_v = 0
    html = html.replace("/static/scripts/chat_demo.js", f"/ui/chat_demo.js?v={js_v}")
    html = html.replace("/static/styles/chat_demo.css", f"/ui/chat_demo.css?v={css_v}")
    html = html.replace("/static/scripts/perf_float.js", f"/ui/perf_float.js?v={js_v}")
    html = html.replace("/static/styles/perf_float.css", f"/ui/perf_float.css?v={css_v}")
    resp = HTMLResponse(html)
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _serve_no_store(path: Path, media_type: str):
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    body = path.read_bytes()
    from fastapi.responses import Response
    return Response(
        content=body,
        media_type=media_type,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/ui/eyes.js")
async def ui_eyes_js():
    return _serve_no_store(_UI_DIR / "scripts" / "eyes.js", "application/javascript")


@app.get("/ui/eyes.css")
async def ui_eyes_css():
    return _serve_no_store(_UI_DIR / "styles" / "eyes.css", "text/css")


@app.get("/eyes", response_class=HTMLResponse)
async def eyes_page():
    html = (_UI_DIR / "templates" / "eyes.html").read_text(encoding="utf-8")
    try:
        js_v = int((_UI_DIR / "scripts" / "eyes.js").stat().st_mtime)
    except OSError:
        js_v = 0
    try:
        css_v = int((_UI_DIR / "styles" / "eyes.css").stat().st_mtime)
    except OSError:
        css_v = 0
    html = html.replace("/static/scripts/eyes.js", f"/ui/eyes.js?v={js_v}")
    html = html.replace("/static/styles/eyes.css", f"/ui/eyes.css?v={css_v}")
    html = html.replace("/static/scripts/perf_float.js", f"/ui/perf_float.js?v={js_v}")
    html = html.replace("/static/styles/perf_float.css", f"/ui/perf_float.css?v={css_v}")
    resp = HTMLResponse(html)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/compare", response_class=HTMLResponse)
async def compare_page():
    html = (_UI_DIR / "templates" / "compare.html").read_text(encoding="utf-8")
    try:
        js_v  = int((_UI_DIR / "scripts" / "compare.js").stat().st_mtime)
        css_v = int((_UI_DIR / "styles" / "compare.css").stat().st_mtime)
    except OSError:
        js_v = css_v = 0
    html = html.replace("/static/scripts/compare.js",  f"/ui/compare.js?v={js_v}")
    html = html.replace("/static/styles/compare.css", f"/ui/compare.css?v={css_v}")
    resp = HTMLResponse(html)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/ui/compare.js")
async def ui_compare_js():
    return _serve_no_store(_UI_DIR / "scripts" / "compare.js", "application/javascript")


@app.get("/ui/compare.css")
async def ui_compare_css():
    return _serve_no_store(_UI_DIR / "styles" / "compare.css", "text/css")


@app.get("/speed-race", response_class=HTMLResponse)
async def speed_race_page():
    html = (_UI_DIR / "templates" / "speed_race.html").read_text(encoding="utf-8")
    try:
        js_v  = int((_UI_DIR / "scripts" / "speed_race.js").stat().st_mtime)
        css_v = int((_UI_DIR / "styles"  / "speed_race.css").stat().st_mtime)
    except OSError:
        js_v = css_v = 0
    html = html.replace("/static/scripts/speed_race.js", f"/ui/speed_race.js?v={js_v}")
    html = html.replace("/static/styles/speed_race.css",  f"/ui/speed_race.css?v={css_v}")
    resp = HTMLResponse(html)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/ui/speed_race.js")
async def ui_speed_race_js():
    return _serve_no_store(_UI_DIR / "scripts" / "speed_race.js", "application/javascript")


@app.get("/ui/speed_race.css")
async def ui_speed_race_css():
    return _serve_no_store(_UI_DIR / "styles" / "speed_race.css", "text/css")


@app.get("/api/load-status")
async def load_status():
    """Live loading progress — poll this until ready:true."""
    return JSONResponse({
        "ready":    _model_ready,
        "stage":    _load_stage,
        "progress": _load_progress,
    })


@app.get("/api/pytorch-status")
async def pytorch_status():
    return JSONResponse({
        "ready": _pt_ready,
        "info":  _pt_info,
    })


@app.get("/api/bench-tps")
async def bench_tps_endpoint():
    """Return the pre-measured pure-compute decode speed for MLX and PyTorch."""
    return JSONResponse({
        "mlx_bench_tps": _static_bench_tps,
        "pt_ready":      _pt_ready,
    })


@app.websocket("/ws/pytorch")
async def pytorch_ws(ws: WebSocket):
    """
    PyTorch reference inference WebSocket (for /speed-race page).
    Protocol mirrors /ws: accepts {action:'chat', prompt, max_tokens, category_key, seed}
    Sends: {type:'meta'}, {type:'token', text, tok_s}, {type:'done', total_tokens, tok_s}
    """
    await ws.accept()
    await ws.send_json({"type": "connected", "ready": _pt_ready})

    if not _pt_ready or _pt_model is None:
        await ws.send_json({"type": "error", "message": "PyTorch model not ready yet — loading weights in background"})
        await ws.close()
        return

    try:
        data = await ws.receive_json()
    except Exception:
        await ws.close()
        return

    if data.get("action") != "chat":
        await ws.close()
        return

    prompt_text  = data.get("prompt", "Who are you?")
    # The PyTorch reference has a hard positional limit (model_max_length=2048);
    # cap generation so prompt + output stays inside it (avoids the indexing-error
    # freeze on long dropped files).
    max_tokens   = min(int(data.get("max_tokens", 256)), 512)
    category_key = data.get("category_key", "self_awareness")
    seed         = int(data.get("seed", 0))
    samp         = data.get("sampling") or {}
    temperature  = float(samp.get("temperature", 0.426))
    top_k        = int(samp.get("top_k", 20))

    # Build ChatML prompt (same logic as MLX path)
    sys_prompt = resolve_system_prompt(category_key, None)
    full_prompt = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    bos  = _pt_tok.token_to_id("<s>") or 1
    ids  = [bos] + _pt_tok.encode(full_prompt, add_special_tokens=False).ids
    # Hard-truncate to the PyTorch model's positional limit, keeping BOS + the
    # tail (document body + assistant marker). Prevents 2610>2048 indexing errors.
    _PT_MAX_CTX = 2048
    _budget = max(16, _PT_MAX_CTX - max_tokens - 8)
    if len(ids) > _budget:
        ids = [ids[0]] + ids[-(_budget - 1):]
    stop = []
    for n in ("<|im_end|>", "</s>"):
        t = _pt_tok.token_to_id(n)
        if t is not None:
            stop.append(t)

    import torch
    from mamba3_mlx.pure_torch_mlx.generate import GenConfig, stream as pt_stream

    gen_cfg = GenConfig(
        max_tokens=max_tokens, temperature=temperature, top_k=top_k,
        top_p=0.981, min_p=0.067, rep_pen=1.146, pres_pen=0.143,
        freq_pen=0.133, seed=seed,
    )

    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _worker():
        try:
            for ev in pt_stream(_pt_model, ids, gen_cfg, _pt_tok,
                                 stop_token_ids=tuple(stop)):
                loop.call_soon_threadsafe(queue.put_nowait, ev)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(exc)})
        loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=_worker, daemon=True).start()
    await ws.send_json({"type": "meta", "n_prompt": len(ids)})

    try:
        while True:
            ev = await queue.get()
            if ev is None:
                break
            await ws.send_json(ev)
    except Exception:
        pass
    finally:
        await ws.close()


@app.get("/ui/chat_demo.js")
async def ui_js():
    return _serve_no_store(_UI_DIR / "scripts" / "chat_demo.js", "application/javascript")


@app.get("/ui/chat_demo.css")
async def ui_css():
    return _serve_no_store(_UI_DIR / "styles" / "chat_demo.css", "text/css")


@app.get("/ui/perf_float.js")
async def ui_perf_float_js():
    return _serve_no_store(_UI_DIR / "scripts" / "perf_float.js", "application/javascript")


@app.get("/ui/perf_float.css")
async def ui_perf_float_css():
    return _serve_no_store(_UI_DIR / "styles" / "perf_float.css", "text/css")


@app.get("/api/status")
async def status():
    if _MOCK_MODE:
        data = _read_ui_config().get("status") or {}
        return JSONResponse(data)
    return JSONResponse({
        "ready": _model_ready,
        "load_timings": _load_timings,
        # Pure-compute decode TPS measured at startup with the StaticDecoder
        # (metal_fuse + q8 + norm_fold + in-graph sampler), 5 warmup rounds +
        # 1 timed round.  This is the same path ``make self-s`` runs, so the
        # frontend can show backend tok/s as a percentage of this CLI peak.
        "cli_peak_tps": _static_bench_tps,
        "config": {
            "d_model": _config.d_model if _config else 0,
            "num_layers": _config.num_layers if _config else 0,
            "kmoe_num_experts": getattr(_config, "kmoe_num_experts", 0) if _config else 0,
            "kmoe_top_k": getattr(_config, "kmoe_top_k", 0) if _config else 0,
            "vocab_size": _vocab_size,
            "dtype": _args.dtype if _args else "",
            "max_new_tokens": _args.max_new_tokens if _args else 0,
        },
    })


def _sampling_defaults_dict() -> dict[str, Any]:
    if _args is None:
        return {"temperature": 0.236, "top_k": 30, "top_p": 0.959, "min_p": 0.122,
                "repetition_penalty": 1.281, "presence_penalty": 0.298,
                "frequency_penalty": 0.168}
    return {
        "temperature": float(getattr(_args, "temp", 0.236)),
        "top_k": int(getattr(_args, "top_k", 30)),
        "top_p": float(getattr(_args, "top_p", 0.959)),
        "min_p": float(getattr(_args, "min_p", 0.122)),
        "repetition_penalty": float(getattr(_args, "rep_pen", 1.281)),
        "presence_penalty": float(getattr(_args, "pres_pen", 0.298)),
        "frequency_penalty": float(getattr(_args, "freq_pen", 0.168)),
    }


def _sampling_mode_configs_dict() -> dict[str, Any]:
    """Return per-mode sampling configs keyed by mode name for frontend consumption.
    Also exposes category-key aliases (e.g. 'email_summary' alongside 'summarize_email')
    so the chat UI can look up by category_key directly.
    """
    from mamba3_mlx.utils.mode_configs import MODE_GEN_CONFIGS
    from mamba3_mlx.utils.system_prompts import MODE_ALIASES
    out: dict[str, Any] = {}
    for mode_key, cfg in MODE_GEN_CONFIGS.items():
        entry = {
            "temperature": cfg.get("temperature", 0.426),
            "top_k": cfg.get("top_k", 20),
            "top_p": cfg.get("top_p", 0.981),
            "min_p": cfg.get("min_p", 0.067),
            "repetition_penalty": cfg.get("rep_pen", 1.146),
            "presence_penalty": cfg.get("pres_pen", 0.143),
            "frequency_penalty": cfg.get("freq_pen", 0.133),
            "seed": cfg.get("seed", 0),
            "reasoning": cfg.get("reasoning", True),
        }
        out[mode_key] = entry
        # Also register under every alias that maps to this mode_key
        for alias, canonical in MODE_ALIASES.items():
            if canonical == mode_key and alias != mode_key:
                out[alias] = entry
    return out


def _format_guard_status_dict() -> dict[str, Any]:
    if _args is None:
        return {
            "enabled": True, "ban_im_start": True, "close_bias": 4.0,
            "close_bias_max": 16.0, "close_bias_start": 0,
            "force_final_inject": True, "reasoning_budget": 0,
        }
    return {
        "enabled": bool(getattr(_args, "format_guard", True)),
        "ban_im_start": bool(getattr(_args, "ban_im_start", True)),
        "close_bias": float(getattr(_args, "close_bias", 4.0)),
        "close_bias_max": float(getattr(_args, "close_bias_max", 16.0)),
        "close_bias_start": int(getattr(_args, "close_bias_start", 0)),
        "force_final_inject": bool(getattr(_args, "force_final_inject", True)),
        "reasoning_budget": int(getattr(_args, "reasoning_budget", 0)),
    }


@app.get("/api/demo-config")
async def demo_config():
    data = _read_ui_config()
    flat: list[dict[str, Any]] = []
    for cat in data.get("categories") or []:
        ck = cat.get("key") or ""
        title = cat.get("title") or ""
        for ex in cat.get("examples") or []:
            flat.append({
                "example_id": ex.get("id"),
                "label": f"{title} · {ex.get('subcategory', '')}",
                "prompt": ex.get("user", ""),
                "category_key": ck,
            })
    legacy = data.get("welcome_examples") or []
    return JSONResponse({
        "mock": _MOCK_MODE,
        "system_prompt_markdown": data.get("system_prompt_markdown", ""),
        "category_system_prompts": _category_system_prompts_map(data),
        "style_constraints": data.get("style_constraints", {}),
        "tool_registry": data.get("tool_registry", []),
        "categories": data.get("categories", []),
        "examples": flat if flat else legacy,
        "sampling_defaults": _sampling_defaults_dict(),
        "sampling_mode_configs": _sampling_mode_configs_dict(),
        "max_new_tokens_cap": int(getattr(_args, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS)) if _args else DEFAULT_MAX_NEW_TOKENS,
        "reasoning_budget": int(getattr(_args, "reasoning_budget", 0)) if _args else 0,
        "format_guard": _format_guard_status_dict(),
    })


class TokenCountRequest(pydantic.BaseModel):
    system_prompt: str = ""
    user_message:  str = ""
    history:       list[dict] = []
    reasoning:     bool = False


@app.post("/api/token_count")
async def token_count(req: TokenCountRequest) -> JSONResponse:
    """Return exact token counts for a prompt without running the model.
    Pure tokenisation only — zero inference cost."""
    if _tokenizer is None:
        # Model not loaded yet — return character-based estimate
        sys_est  = len(req.system_prompt) // 4
        user_est = len(req.user_message)  // 4
        return JSONResponse({"sys_tokens": sys_est, "user_tokens": user_est,
                             "total_tokens": sys_est + user_est, "estimated": True})
    hist = list(req.history)
    if req.user_message.strip():
        hist.append({"role": "user", "content": req.user_message.strip()})
    ids = _build_multiturn_ids(
        hist,
        seq_len=int(getattr(_args, "seq_len", 2048)) if _args else 2048,
        system_prompt=req.system_prompt or None,
        reasoning=req.reasoning,
    )
    # Count system-only portion for breakdown
    sys_ids = _encode_plain(
        f"<|im_start|>system\n{req.system_prompt.strip()}<|im_end|>\n"
    ) if req.system_prompt.strip() else []
    total = len(ids)
    sys_t = len(sys_ids)
    return JSONResponse({"sys_tokens": sys_t, "user_tokens": max(0, total - sys_t),
                         "total_tokens": total, "estimated": False})


def _apply_sampling_override(base_args: Any, sampling: dict[str, Any] | None) -> Any:
    import copy as _copy

    if not isinstance(sampling, dict) or not sampling:
        return base_args
    a = _copy.copy(base_args)
    if "temperature" in sampling:
        with contextlib.suppress(Exception):
            a.temp = max(0.0, min(5.0, float(sampling["temperature"])))
    if "top_k" in sampling:
        with contextlib.suppress(Exception):
            a.top_k = max(0, min(2048, int(sampling["top_k"])))
    if "top_p" in sampling:
        with contextlib.suppress(Exception):
            a.top_p = max(0.0, min(1.0, float(sampling["top_p"])))
    if "min_p" in sampling:
        with contextlib.suppress(Exception):
            a.min_p = max(0.0, min(1.0, float(sampling["min_p"])))
    if "repetition_penalty" in sampling:
        with contextlib.suppress(Exception):
            a.rep_pen = max(1.0, min(2.0, float(sampling["repetition_penalty"])))
    if "presence_penalty" in sampling:
        with contextlib.suppress(Exception):
            a.pres_pen = max(0.0, min(2.0, float(sampling["presence_penalty"])))
    if "frequency_penalty" in sampling:
        with contextlib.suppress(Exception):
            a.freq_pen = max(0.0, min(2.0, float(sampling["frequency_penalty"])))
    if "seed" in sampling:
        with contextlib.suppress(Exception):
            a.seed = int(sampling["seed"])
    return a


def _apply_format_guard_override(base_args: Any, override: Any) -> Any:
    import copy as _copy

    if override is None:
        return base_args
    enabled: bool | None = None
    inject: bool | None = None
    if isinstance(override, bool):
        enabled = override
    elif isinstance(override, dict):
        if "enabled" in override:
            with contextlib.suppress(Exception):
                enabled = bool(override["enabled"])
        if "force_final_inject" in override:
            with contextlib.suppress(Exception):
                inject = bool(override["force_final_inject"])
    if enabled is False or inject is False:
        a = _copy.copy(base_args)
        if enabled is False:
            a.format_guard_call_override = False
        if inject is False:
            a.force_final_inject_call_override = False
        return a
    return base_args


# ---------------------------------------------------------------------------
# Mock streaming (unchanged from the previous chat_demo.py — UI-only)
# ---------------------------------------------------------------------------
_TOKEN_RE = __import__("re").compile(r"\S+|\s+")


def _chunk_stream_text(text: str, max_chunk: int = 40) -> list[str]:
    if not text:
        return []
    parts = _TOKEN_RE.findall(text)
    chunks: list[str] = []
    buf = ""
    for p in parts:
        if buf and len(buf) + len(p) > max_chunk:
            chunks.append(buf)
            buf = p
        else:
            buf += p
    if buf:
        chunks.append(buf)
    return chunks if chunks else [text]


def _flatten_mock_examples(data: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cat in data.get("categories") or []:
        ck = str(cat.get("key") or "")
        title = str(cat.get("title") or "")
        for ex in cat.get("examples") or []:
            if not isinstance(ex, dict):
                continue
            row = dict(ex)
            row["category_key"] = ck
            row["category_title"] = title
            out.append(row)
    return out


def _find_mock_example(data: dict[str, Any], example_id: str | None, prompt: str) -> dict[str, Any] | None:
    examples = _flatten_mock_examples(data)
    if example_id:
        eid = str(example_id).strip()
        for ex in examples:
            if ex.get("id") == eid:
                return ex
    pn = " ".join(prompt.split())
    for ex in examples:
        u = str(ex.get("user", "")).strip()
        if u and pn == " ".join(u.split()):
            return ex
    return None


async def _mock_stream_category_system_prompt(ws: WebSocket, category_key: str) -> None:
    data = _read_ui_config()
    stream_cfg = data.get("mock_stream") or {}
    jitter = float(stream_cfg.get("chunk_delay_jitter", 0.12))
    intro_tok_s = float(stream_cfg.get("intro_tok_s", 60.0))
    pause_intro = float(stream_cfg.get("pause_after_intro_ms", 120)) / 1000.0
    ck = str(category_key or "").strip()
    prompts = _category_system_prompts_map(data)
    text = prompts.get(ck, "").strip()
    if not text:
        await ws.send_json({
            "type": "error",
            "message": f"No training system prompt for category {ck!r}.",
        })
        return
    cat_title = ck
    for c in data.get("categories") or []:
        if str(c.get("key") or "") == ck:
            cat_title = str(c.get("title") or ck)
            break
    await ws.send_json({"type": "intro_start", "category_key": ck, "category_title": cat_title})
    intro_n = 0
    intro_per = 1.0 / max(intro_tok_s, 1.0)
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    for ch in _chunk_stream_text(text, 56):
        j = 1.0 + random.uniform(-jitter, jitter)
        await asyncio.sleep(max(0.003, intro_per * j))
        intro_n += 1
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t0) * 1000.0
        el = time.perf_counter() - t0
        await ws.send_json({"type": "token", "text": ch, "n": intro_n,
                            "tok_s": round(intro_n / max(el, 1e-9), 1)})
    elapsed = time.perf_counter() - t0
    await asyncio.sleep(max(0.0, pause_intro))
    await ws.send_json({
        "type": "done",
        "total_tokens": intro_n,
        "total_ms": round(elapsed * 1000.0, 2),
        "tok_s": round(intro_n / max(elapsed, 1e-9), 1),
        "ttft_ms": round(ttft_ms or 0.0, 2),
        "prefill_ms": 0.0,
        "play_system_only": True,
    })


async def _mock_chat_stream(ws: WebSocket, prompt: str, max_tokens: int, turns: int,
                             example_id: str | None = None) -> str:
    data = _read_ui_config()
    stream_cfg = data.get("mock_stream") or {}
    target_tok_s = float(stream_cfg.get("target_tok_s", 40.0)) or 40.0
    jitter = float(stream_cfg.get("chunk_delay_jitter", 0.12))
    base_per_chunk = 1.0 / target_tok_s
    pause_meta = float(stream_cfg.get("pause_after_meta_ms", 160)) / 1000.0
    pause_reason = float(stream_cfg.get("pause_after_reasoning_ms", 400)) / 1000.0

    ex = _find_mock_example(data, example_id, prompt)

    async def sleep_chunk() -> None:
        j = 1.0 + random.uniform(-jitter, jitter)
        await asyncio.sleep(max(0.003, base_per_chunk * j))

    prefill_ms = round(6.0 + (len(prompt) % 11) * 0.45, 2)
    prompt_tokens = min(16 + len(prompt) // 2, 600)
    meta: dict[str, Any] = {"type": "meta", "prefill_ms": prefill_ms,
                            "prompt_tokens": prompt_tokens, "turns": turns}
    if ex:
        meta["category"] = ex.get("category_key")
        meta["example_id"] = ex.get("id")
        meta["subcategory"] = ex.get("subcategory")
    await ws.send_json(meta)
    await asyncio.sleep(max(0.0, pause_meta))

    n_out = 0
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    assembled: list[str] = []

    async def emit(ch: str) -> None:
        nonlocal n_out, ttft_ms
        if n_out >= max_tokens:
            return
        await sleep_chunk()
        n_out += 1
        assembled.append(ch)
        elapsed = time.perf_counter() - t0
        if ttft_ms is None:
            ttft_ms = elapsed * 1000.0
        await ws.send_json({"type": "token", "text": ch, "n": n_out,
                            "tok_s": round(n_out / max(elapsed, 1e-9), 1)})

    cot = (ex.get("cot_markdown") or ex.get("cot") or "") if ex else ""
    cot = str(cot).strip()
    if cot:
        await ws.send_json({"type": "reasoning", "markdown": cot})
        await asyncio.sleep(max(0.0, pause_reason))

    body = (ex.get("assistant_markdown") or ex.get("output") or "") if ex else "[mock] ok."
    for ch in _chunk_stream_text(str(body).strip(), 44):
        if n_out >= max_tokens:
            break
        await emit(ch)

    elapsed = time.perf_counter() - t0
    await ws.send_json({
        "type": "done",
        "total_tokens": n_out,
        "total_ms": round(elapsed * 1000.0, 2),
        "tok_s": round(n_out / max(elapsed, 1e-9), 1),
        "ttft_ms": round(ttft_ms or 0.0, 2),
        "prefill_ms": prefill_ms,
    })
    return "".join(assembled)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "connected", "ready": _model_ready, "mock": _MOCK_MODE})

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
                if not _MOCK_MODE:
                    _free_metal_cache()
                await ws.send_json({"type": "cleared"})
                continue

            if action == "play_system_prompt":
                ck = str(msg.get("category_key", "")).strip()
                if not ck:
                    await ws.send_json({"type": "error", "message": "category_key is required"})
                    continue
                if _MOCK_MODE:
                    await _mock_stream_category_system_prompt(ws, ck)
                    continue
                if not _model_ready:
                    await ws.send_json({"type": "error", "message": "Model not loaded"})
                    continue
                # Native stack: no prefix-prime cache; just stream the text.
                ui_data = _read_ui_config()
                cat_map = _category_system_prompts_map(ui_data)
                sys_text = (cat_map.get(ck) or "").strip()
                if not sys_text:
                    await ws.send_json({"type": "error", "message": f"No system prompt for category {ck!r}"})
                    continue
                cat_title = ck
                for c in ui_data.get("categories") or []:
                    if str(c.get("key") or "") == ck:
                        cat_title = str(c.get("title") or ck)
                        break
                stream_cfg = ui_data.get("mock_stream") or {}
                intro_tok_s = float(stream_cfg.get("intro_tok_s", 64.0)) or 64.0
                jitter = float(stream_cfg.get("chunk_delay_jitter", 0.12))
                intro_per = 1.0 / max(intro_tok_s, 1.0)
                await ws.send_json({"type": "intro_start", "category_key": ck, "category_title": cat_title})
                t0 = time.perf_counter()
                ttft_ms: float | None = None
                n_out = 0
                for ch in _chunk_stream_text(sys_text, 56):
                    j = 1.0 + random.uniform(-jitter, jitter)
                    await asyncio.sleep(max(0.003, intro_per * j))
                    n_out += 1
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                    el = time.perf_counter() - t0
                    await ws.send_json({"type": "token", "text": ch, "n": n_out,
                                        "tok_s": round(n_out / max(el, 1e-9), 1)})
                elapsed = time.perf_counter() - t0
                await ws.send_json({
                    "type": "done",
                    "total_tokens": n_out,
                    "total_ms": round(elapsed * 1000.0, 2),
                    "tok_s": round(n_out / max(elapsed, 1e-9), 1),
                    "ttft_ms": round(ttft_ms or 0.0, 2),
                    "prefill_ms": 0.0,
                    "play_system_only": True,
                    "primed_ok": False,
                })
                continue

            if action == "chat":
                prompt = msg.get("prompt", "").strip()
                # Hard server-side ceiling (防呆): no client request can ask for a
                # runaway generation that pins the GPU for minutes and freezes the
                # whole machine. 2048 is a no-op for normal turns (frontend sends
                # ≤2048) but clamps pathological values like 12864.
                max_tokens = min(
                    int(msg.get("max_tokens", 512) or 512),
                    int(getattr(_args, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS)) if _args else DEFAULT_MAX_NEW_TOKENS,
                    2048,
                )
                if not prompt:
                    await ws.send_json({"type": "error", "message": "Empty prompt"})
                    continue
                if not _model_ready:
                    await ws.send_json({"type": "error", "message": "Model not loaded"})
                    continue

                if _MOCK_MODE:
                    conversation.append({"role": "user", "content": prompt})
                    turns = len([m for m in conversation if m["role"] == "user"])
                    ex_id = msg.get("example_id")
                    assistant_text = await _mock_chat_stream(ws, prompt, max_tokens, turns, example_id=ex_id)
                    conversation.append({"role": "assistant", "content": assistant_text})
                    continue

                # Real inference.
                category_key = str(msg.get("category_key") or "").strip()
                ui_data = _read_ui_config()
                cat_map = _category_system_prompts_map(ui_data)
                fallback_key = "daily_conversation" if "daily_conversation" in cat_map \
                    else (next(iter(cat_map.keys()), ""))
                effective_key = category_key if category_key in cat_map else fallback_key
                system_prompt = cat_map.get(effective_key, "") if effective_key else ""

                # ── Console turn header ──────────────────────────────────────
                _cprint(f"\n{'─'*64}\n", style="dim")
                _cprint(f"[turn] cat={effective_key!r}  "
                        f"max_tokens={max_tokens}\n", style="bold cyan")
                if system_prompt:
                    _cprint(f"[sys]  {system_prompt[:300]}\n", style="cyan dim")
                _cprint(f"[user] {prompt[:200]}\n", style="bold white")
                _cprint("[gen]  ", style="dim")
                # ────────────────────────────────────────────────────────────

                # Merge mode-default sampling → frontend sampling override.
                mode_sampling = get_mode_gen_config(effective_key)
                # reasoning: mode config is the hard default; mode config = False means
                # <think> injection is suppressed regardless of the frontend toggle.
                # The JS applyModeConfig() syncs the UI toggle, so in normal use the
                # frontend will already send the correct value — this is the safety net.
                mode_reasoning = mode_sampling.get("reasoning")
                frontend_reasoning = msg.get("reasoning")
                if mode_reasoning is not None:
                    reasoning_flag = bool(mode_reasoning)
                elif frontend_reasoning is None:
                    reasoning_flag = bool(getattr(_args, "reasoning", True))
                else:
                    reasoning_flag = bool(frontend_reasoning)
                frontend_sampling = msg.get("sampling") if isinstance(msg.get("sampling"), dict) else {}
                merged_sampling = {**mode_sampling, **(frontend_sampling or {})}
                # Convert mode_configs key names to chat_demo field names.
                # "reasoning" is handled separately above — exclude from sampling override.
                _SAMPLING_ONLY_KEYS = {"rep_pen", "pres_pen", "freq_pen", "top_k", "top_p",
                                       "min_p", "temperature", "seed", "repetition_penalty",
                                       "presence_penalty", "frequency_penalty"}
                _key_map = {"rep_pen": "repetition_penalty", "pres_pen": "presence_penalty",
                            "freq_pen": "frequency_penalty", "top_k": "top_k",
                            "top_p": "top_p", "min_p": "min_p",
                            "temperature": "temperature", "seed": "seed"}
                sampling_override = {
                    (_key_map.get(k, k)): v for k, v in merged_sampling.items()
                    if k in _SAMPLING_ONLY_KEYS
                } if merged_sampling else None
                args_for_call = _apply_sampling_override(_args, sampling_override)
                args_for_call = _apply_format_guard_override(args_for_call, msg.get("format_guard"))
                # raw_sampling: disable ALL logit engineering (ban masks, close_bias,
                # extra bans) so chat path matches run.py's unbiased sampling exactly.
                # Used for self_awareness where seed=27 is tuned against run.py's path.
                if mode_sampling.get("raw_sampling"):
                    import copy as _copy
                    args_for_call = _copy.copy(args_for_call)
                    args_for_call.format_guard_call_override = False
                    args_for_call.raw_sampling = True
                # bench_mode: after generation run a short compute-only timing loop
                if msg.get("bench_mode"):
                    import copy as _copy
                    args_for_call = _copy.copy(args_for_call)
                    args_for_call.bench_mode = True
                no_eos_stop_flag = bool(msg.get("no_eos_stop", False))

                conversation.append({"role": "user", "content": prompt})
                history_snapshot = list(conversation)
                abort_event = threading.Event()

                # ── Non-blocking event pipeline (worker thread → drainer task) ──
                # The MLX worker thread MUST NOT wait on the event loop to flush
                # each token's WS frame — that round-trip costs ~1–2 ms/tok and
                # serialises decode behind JSON encode + socket write.
                # Worker only schedules events onto an asyncio.Queue (call_soon_
                # threadsafe is ~tens of µs); a dedicated drainer coroutine in
                # the loop thread does ws.send_json in order.
                out_queue: asyncio.Queue = asyncio.Queue()

                def _send_now(ev: dict):
                    # Called from the worker thread. Fire-and-forget.
                    loop.call_soon_threadsafe(out_queue.put_nowait, ev)

                async def _drainer():
                    # Opportunistic frame merge: when several events were
                    # queued while we were sending the previous frame, bundle
                    # them into a single "batch" envelope so we pay the WS
                    # write/syscall cost once instead of N times.  Single-event
                    # bursts still ship as the original event shape (no
                    # protocol change for the common case).
                    while True:
                        ev = await out_queue.get()
                        if ev is None:
                            return
                        batch = [ev]
                        sentinel = False
                        # Drain anything already sitting in the queue without
                        # yielding back to the loop.
                        while True:
                            try:
                                e2 = out_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                            if e2 is None:
                                sentinel = True
                                break
                            batch.append(e2)
                        try:
                            if len(batch) == 1:
                                await ws.send_text(_ws_dumps(batch[0]))
                            else:
                                await ws.send_text(_ws_dumps(
                                    {"type": "batch", "events": batch}
                                ))
                        except Exception:
                            # WS closed mid-stream — drop remaining events.
                            return
                        if sentinel:
                            return

                def run_gen() -> str:
                    global _static_bench_tps
                    text_parts: list[str] = []
                    try:
                        for ev in _stream_generate(
                            history_snapshot, max_tokens=max_tokens,
                            system_prompt=system_prompt, reasoning=reasoning_flag,
                            args_for_call=args_for_call, no_eos_stop=no_eos_stop_flag,
                            abort_event=abort_event,
                        ):
                            _send_now(ev)
                            if ev.get("type") == "token":
                                text_parts.append(ev.get("text", ""))
                    except Exception as exc:
                        try:
                            _send_now({"type": "error", "message": str(exc)})
                        except Exception:
                            pass
                    # Fresh benchmark after streaming. Two triggers:
                    #   (a) bench_mode (explicit /speed-race) — always re-runs.
                    #   (b) lazy first-touch — fires once per server lifetime when
                    #       _static_bench_tps is None, so /api/status.cli_peak_tps
                    #       gets populated and the perf-float panel can show
                    #       "X% of CLI peak" from the second user message onward.
                    # Skip lazy trigger for self_awareness mode: the cold-boot
                    # seed=26 → "I am Mamba" guarantee depends on a clean Metal
                    # bf16 state, and even a tiny post-stream bench shifts logits.
                    # Subsequent self_awareness turns aren't seed-critical, so the
                    # bench can fire after a non-self_awareness generation.
                    _is_self = (effective_key == "self_awareness")
                    _explicit = bool(getattr(args_for_call, "bench_mode", False))
                    _lazy = (_static_bench_tps is None) and (not _is_self)
                    if _static_decoder is not None and (_explicit or _lazy):
                        try:
                            from mamba3_mlx.utils.config import GenerationConfig as _GC2
                            _fresh_gc = _GC2(
                                max_tokens=128, temperature=0.426, top_k=20, top_p=0.981,
                                min_p=0.067, rep_pen=1.146, pres_pen=0.143, freq_pen=0.133,
                                seed=0)
                            _fresh_ids = [1, 2, 3, 4, 5, 6, 7, 8]
                            _static_decoder.generate(_fresh_ids, _fresh_gc, stop_token_ids=())
                            _fresh_br = _static_decoder.generate(_fresh_ids, _fresh_gc,
                                                                  stop_token_ids=())
                            _static_bench_tps = round(_fresh_br.decode_tps, 1)
                            _send_now({"type": "bench", "bench_tps": _static_bench_tps})
                            _label = "post-race" if _explicit else "lazy post-gen"
                            print(f"\n[bench] fresh {_label}: {_static_bench_tps} tok/s",
                                  flush=True)
                        except Exception as _be:
                            print(f"\n[bench] WARN: fresh bench failed: {_be}", flush=True)
                    return "".join(text_parts)

                async def _abort_listener():
                    try:
                        while isGenerating_flag[0]:
                            try:
                                raw2 = await asyncio.wait_for(ws.receive_text(), timeout=0.5)
                            except asyncio.TimeoutError:
                                continue
                            except Exception:
                                break
                            try:
                                m2 = json.loads(raw2)
                            except Exception:
                                continue
                            if m2.get("action") == "abort":
                                abort_event.set()
                                break
                            if m2.get("action") == "ping":
                                await ws.send_json({"type": "pong", "ready": _model_ready})
                    except Exception:
                        pass

                isGenerating_flag = [True]
                abort_listener_task = asyncio.create_task(_abort_listener())
                drainer_task = asyncio.create_task(_drainer())

                async with _infer_lock:
                    assistant_text = await loop.run_in_executor(None, run_gen)

                # Signal the drainer to flush any tail events then exit.
                out_queue.put_nowait(None)
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(drainer_task, timeout=10.0)

                isGenerating_flag[0] = False
                abort_listener_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await abort_listener_task

                conversation.append({"role": "assistant", "content": assistant_text})
                continue

            if action == "abort":
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
    p = argparse.ArgumentParser(description="Mamba3-XR Chat Demo Server (mamba3_mlx native)")
    p.add_argument("--host", type=str, default=_cfg.HOST)
    p.add_argument("--port", type=int, default=_cfg.PORT)
    p.add_argument("--mock", action="store_true",
                   help="Skip MLX load; serve mock /api/status and WebSocket stream from ui/mock_config.json")
    # Paths
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(_REPO_ROOT, _cfg.CHECKPOINT_RELPATH))
    p.add_argument("--tokenizer", type=str,
                   default=os.path.join(_REPO_ROOT, _cfg.TOKENIZER_RELPATH))
    # Sizing
    p.add_argument("--vocab-size", dest="vocab_size", type=int, default=_cfg.VOCAB_SIZE,
                   help="0 = derive from tokenizer (recommended).")
    p.add_argument("--seq-len", dest="seq_len", type=int, default=_cfg.SEQ_LEN)
    p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=_cfg.MAX_NEW_TOKENS)
    p.add_argument("--warmup", type=int, default=_cfg.WARMUP_STEPS)
    # Precision
    p.add_argument("--dtype", type=str, default=_cfg.DTYPE, choices=["fp32", "bf16", "fp16"])
    # Sampling
    p.add_argument("--temp", type=float, default=_cfg.TEMPERATURE)
    p.add_argument("--top_k", type=int, default=_cfg.TOP_K)
    p.add_argument("--top_p", type=float, default=_cfg.TOP_P)
    p.add_argument("--min_p", type=float, default=_cfg.MIN_P)
    p.add_argument("--rep_pen", type=float, default=_cfg.REP_PEN)
    p.add_argument("--pres_pen", type=float, default=_cfg.PRES_PEN)
    p.add_argument("--freq_pen", type=float, default=_cfg.FREQ_PEN)
    p.add_argument("--repeat_last_n", type=int, default=_cfg.REPEAT_LAST_N)
    p.add_argument("--seed", type=int, default=_cfg.SEED)
    # Middleware
    p.add_argument("--reasoning", dest="reasoning", action="store_true", default=_cfg.REASONING)
    p.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    p.add_argument("--reasoning-budget", dest="reasoning_budget", type=int,
                   default=_cfg.REASONING_BUDGET,
                   help="0 = no cap; model decides when to close <think>.")
    p.add_argument("--format-guard", dest="format_guard", action="store_true",
                   default=_cfg.FORMAT_GUARD)
    p.add_argument("--no-format-guard", dest="format_guard", action="store_false")
    p.add_argument("--ban-im-start", dest="ban_im_start", action="store_true",
                   default=_cfg.BAN_IM_START)
    p.add_argument("--no-ban-im-start", dest="ban_im_start", action="store_false")
    p.add_argument("--close-bias", dest="close_bias", type=float, default=_cfg.CLOSE_BIAS)
    p.add_argument("--close-bias-max", dest="close_bias_max", type=float,
                   default=_cfg.CLOSE_BIAS_MAX)
    p.add_argument("--close-bias-start", dest="close_bias_start", type=int,
                   default=_cfg.CLOSE_BIAS_START)
    p.add_argument("--force-final-inject", dest="force_final_inject",
                   action="store_true", default=_cfg.FORCE_FINAL_INJECT)
    p.add_argument("--no-force-final-inject", dest="force_final_inject", action="store_false")
    p.add_argument("--final-min-tokens", dest="final_min_tokens", type=int,
                   default=_cfg.FINAL_MIN_TOKENS)
    p.add_argument("--static-stream", dest="static_stream", action="store_true",
                   default=True,
                   help="Per-token forward through the StaticDecoder compiled "
                        "graph (metal_fuse + q8 + norm_fold). ~2x decode tok/s.")
    p.add_argument("--no-static-stream", dest="static_stream", action="store_false")
    # q8 DISABLED by default: q8 shifts logits and destroys the tuned
    # self_awareness "I am Mamba" phrase (verified). bf16 keeps demo quality
    # at ~98 tok/s. Re-enable with --quant-moe 8 --quant-proj 8 --quant-head 8.
    p.add_argument("--quant-moe", dest="quant_moe", type=int, default=0,
                   help="TuckerMoE U_in/U_out quant bits for the static path "
                        "(0 = bf16 bit-exact; default 0 to preserve tuned seeds).")
    p.add_argument("--quant-proj", dest="quant_proj", type=int, default=0,
                   help="in_proj/dense/qkv/o quant bits (dt/A/λ stays bf16; default 0).")
    p.add_argument("--quant-head", dest="quant_head", type=int, default=0,
                   help="Head projection quant bits (default 0).")
    return p.parse_args()


def main():
    args = parse_args()
    global _load_timings, _model_ready, _config, _vocab_size, _args, _MOCK_MODE

    print(f"\n{'='*60}")
    print("  Mamba3-XR Chat Demo (WebSocket, mamba3_mlx native)")
    print(f"{'='*60}\n")

    _args = args
    _MOCK_MODE = bool(args.mock)

    if _MOCK_MODE:
        print("  MOCK MODE — no MLX weights (edit mamba3_mlx/ui/mock_config.json)")
        data = _read_ui_config()
        st = data.get("status") or {}
        _load_timings = st.get("load_timings") or {"total_ms": 0.0}
        cfg = dict(st.get("config") or {})
        defaults = {"d_model": 768, "num_layers": 6, "kmoe_num_experts": 8,
                    "kmoe_top_k": 2, "vocab_size": 32007, "dtype": "bf16",
                    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS}
        defaults.update(cfg)
        _config = SimpleNamespace(**defaults)
        _vocab_size = int(getattr(_config, "vocab_size", 32007))
        _model_ready = True
    else:
        print("  Loading model...")
        _load_timings = _load_model(args)
        print(f"\n  Model loaded in {_load_timings['total_ms']:.0f} ms")
        print(f"  Checkpoint: {_load_timings.get('checkpoint', 'N/A')}")
        print(f"  dtype: {args.dtype}")

    print(f"\n  Server: http://localhost:{args.port}")
    print(f"  WebSocket: ws://localhost:{args.port}/ws")
    print(f"  Speed Race: http://localhost:{args.port}/speed-race")
    print(f"{'='*60}\n")

    # Load PyTorch reference model in background (for /speed-race page)
    if not _MOCK_MODE:
        _pt_thread = threading.Thread(
            target=_load_pytorch_backend,
            args=(args.checkpoint, args.tokenizer),
            daemon=True,
        )
        _pt_thread.start()

    import uvicorn
    # Prefer uvloop when available — measurable on hot WS streaming path.
    loop_kw = {"loop": "uvloop"} if _UVLOOP_AVAILABLE else {}
    if _UVLOOP_AVAILABLE:
        print("[chat_demo] uvloop enabled (libuv event loop)")
    if orjson is not None:
        print("[chat_demo] orjson enabled (fast JSON encoder for WS frames)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", **loop_kw)


if __name__ == "__main__":
    main()
