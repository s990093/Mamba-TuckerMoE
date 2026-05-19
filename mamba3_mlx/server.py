# -*- coding: utf-8 -*-
"""
Mamba Edge Chat Server
======================
FastAPI + WebSocket server that wraps the mamba3_mlx inference stack and
serves the static UI from mamba3_mlx/ui/.

Routes
------
GET  /                      → chat_demo.html (JS/CSS src gets ?v=<mtime> buster)
GET  /static/<file>         → ui/ assets with no-store headers
GET  /api/status            → model load timings + config
GET  /api/demo-config       → system prompts, categories, sampling defaults
WS   /ws                    → streaming chat protocol (see spec below)

WS protocol (C→S)
-----------------
  {action:"chat",       prompt, max_tokens, reasoning, sampling, format_guard?, no_eos_stop?,
                        category_key?, example_id?, reasoning_budget?, final_min_tokens?}
  {action:"play_system_prompt", category_key}
  {action:"clear"}
  {action:"abort"}
  {action:"ping"}

WS protocol (S→C)
-----------------
  {type:"connected"}
  {type:"meta",          prefill_ms, prompt_tokens, turns}
  {type:"reasoning",     markdown}
  {type:"assistant_split"}
  {type:"tool_action",   call, system_result, phase}
  {type:"token",         text, n, tok_s}
  {type:"done",          tok_s, ttft_ms, prefill_ms, total_tokens, total_ms, play_system_only?}
  {type:"error",         message}
  {type:"cleared"}
  {type:"pong"}
"""
import argparse
import asyncio
import collections
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Generator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).parent
_UI_DIR       = _HERE / "ui"
_REPO_ROOT    = _HERE.parent
_DEFAULT_CKPT = str(_REPO_ROOT / "checkpoints" / "latest_sft_cot_model.npz")
_DEFAULT_TOK  = str(_REPO_ROOT / "checkpoints" / "tokenizer")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mamba.server")

# ── Global server state ────────────────────────────────────────────────────────
_STATE: dict[str, Any] = {
    "model":            None,
    "tokenizer":        None,
    "ready":            False,
    "loading":          True,
    "error":            None,
    "mock":             False,
    "mock_data":        None,
    "load_timings":     {},
    "model_config":     {},
    "stop_ids":         [],
    "final_inject_ids": [],
    "mw_deps":          None,   # CotMiddlewareDeps — built once, shared across turns
    "mw_cfg":           None,   # CotMiddlewareConfig — default knobs
}

_CFG: dict[str, Any] = {}

# ── Primed system-prefix cache (LRU, up to _PRIME_MAX entries) ─────────────────
_primed_prefix: collections.OrderedDict = collections.OrderedDict()
_prime_lock = threading.Lock()
_PRIME_MAX  = 4


def _prime_get(sys_text: str) -> dict | None:
    with _prime_lock:
        entry = _primed_prefix.get(sys_text)
        if entry is not None:
            _primed_prefix.move_to_end(sys_text)
        return entry


def _prime_put(sys_text: str, state: dict) -> None:
    with _prime_lock:
        _primed_prefix[sys_text] = state
        _primed_prefix.move_to_end(sys_text)
        while len(_primed_prefix) > _PRIME_MAX:
            _primed_prefix.popitem(last=False)


def _prime_system_prefix_sync(sys_text: str) -> dict | None:
    """Blocking: prefill the system block only, return a cacheable state snapshot."""
    import mlx.core as mx
    from .inference.generator import prefill, _eval_states

    model     = _STATE["model"]
    tokenizer = _STATE["tokenizer"]
    if model is None or tokenizer is None:
        return None
    sys_block = f"<|im_start|>system\n{sys_text}<|im_end|>\n"
    ids       = tokenizer.encode(sys_block)
    tensor    = mx.array([ids])
    try:
        _, ms, kv = prefill(model, tensor)
        _eval_states(ms, kv)
        return {"ids": ids, "mamba_states": ms, "kv_caches": kv, "pos": len(ids)}
    except Exception as exc:
        log.warning("System prefix prime failed: %s", exc)
        return None


def _do_prime(sys_text: str) -> None:
    """Background-safe: prime and cache the system prefix if not already done."""
    if _prime_get(sys_text) is not None:
        return
    state = _prime_system_prefix_sync(sys_text)
    if state is not None:
        _prime_put(sys_text, state)
        log.info("Primed system prefix (%.40s…)", sys_text)


def _free_metal_cache() -> None:
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if _CFG.get("mock"):
        from .server_config import load_mock_config
        _STATE["mock"]      = True
        _STATE["mock_data"] = load_mock_config()
        _STATE["ready"]     = True
        _STATE["loading"]   = False
        log.info("Mock mode — no weights loaded")
    else:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _load_model_sync)
    yield


app = FastAPI(lifespan=lifespan)


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model_sync() -> None:
    """Blocking model load — runs in the default thread-pool executor."""
    import mlx.core as mx
    from .mlx_model.hybrid_model import build_model
    from transformers import AutoTokenizer

    t0 = time.perf_counter()
    try:
        log.info("Loading tokenizer …")
        t1 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(_CFG["tokenizer"])
        t2 = time.perf_counter()
        tok_ms = (t2 - t1) * 1000

        log.info("Loading model weights from %s …", _CFG["checkpoint"])
        dtype = {"fp32": mx.float32, "fp16": mx.float16}.get(_CFG.get("dtype", "bf16"), mx.bfloat16)
        model = build_model(_CFG["checkpoint"], dtype=dtype)
        t3 = time.perf_counter()
        weights_ms = (t3 - t2) * 1000

        log.info("Warming up …")
        _warmup(model, tokenizer)
        _free_metal_cache()   # discard compilation scratch buffers from warmup
        t4 = time.perf_counter()
        warmup_ms = (t4 - t3) * 1000

        stop_ids: list[int] = []
        for tok_str in ("<|im_end|>", "<|im_start|>", "</s>", "<|endoftext|>"):
            try:
                tid = tokenizer.convert_tokens_to_ids(tok_str)
                if isinstance(tid, int) and tid > 0:
                    stop_ids.append(tid)
            except Exception:
                pass

        final_inject_ids = tokenizer.encode("<final>\n", add_special_tokens=False) or []

        # Build per-process middleware deps (tokenizer ids resolved once, shared across turns)
        try:
            from .cot_middleware import CotMiddlewareDeps, CotMiddlewareConfig
            # Use model config vocab_size if available; fall back to len(tokenizer)
            # which includes added special tokens (tokenizer.vocab_size is base-only).
            vocab_size = (
                getattr(getattr(model, "config", None), "vocab_size", None)
                or len(tokenizer)
            )
            # Detect actual backend vocabulary size (fixes issue where CoT tokens like 32003-32005
            # are outside tokenizer.vocab_size=32000 but within actual backend vocab=32007)
            if hasattr(tokenizer, "backend_tokenizer") and hasattr(tokenizer.backend_tokenizer, "get_vocab"):
                try:
                    backend_vocab = tokenizer.backend_tokenizer.get_vocab()
                    if backend_vocab:
                        vocab_size = max(vocab_size, max(backend_vocab.values()) + 1)
                except Exception:
                    pass

            # Build middleware config with defaults
            # (can be overridden per-turn via WS message)
            mw_cfg  = CotMiddlewareConfig(
                enabled=True,
                reasoning_budget=500,
                final_min_tokens=16,
            )
            mw_deps = CotMiddlewareDeps.build(
                tokenizer=tokenizer,
                vocab_size=int(vocab_size),
                existing_stop_ids=stop_ids,
                cfg=mw_cfg,
            )
            # Use merged stop_ids (adds single-token </final> id if resolvable)
            stop_ids = list(mw_deps.stop_ids)
            _STATE["mw_deps"] = mw_deps
            _STATE["mw_cfg"]  = mw_cfg
            log.info("Format guard: %s", mw_deps.describe())
        except Exception as _mw_exc:
            log.warning("CotMiddlewareDeps build failed (format guard disabled): %s", _mw_exc)

        cfg = model.config if hasattr(model, "config") else None
        mcfg = {
            "d_model":          getattr(cfg, "d_model", "—"),
            "num_layers":       getattr(cfg, "num_layers", "—"),
            "kmoe_num_experts": getattr(cfg, "kmoe_num_experts", 8),
            "kmoe_top_k":       getattr(cfg, "kmoe_top_k", 2),
            "quantize":         _CFG.get("quantize"),
            "dtype":            _CFG.get("dtype", "bf16"),
            "max_new_tokens":   _CFG.get("max_new_tokens", 2048),
        }

        _STATE.update({
            "model":            model,
            "tokenizer":        tokenizer,
            "ready":            True,
            "loading":          False,
            "stop_ids":         stop_ids,
            "final_inject_ids": final_inject_ids,
            "model_config":     mcfg,
            "load_timings": {
                "total_ms":      (t4 - t0) * 1000,
                "tokenizer_ms":  tok_ms,
                "model_init_ms": 0,
                "weights_ms":    weights_ms,
                "warmup_ms":     warmup_ms,
                "checkpoint":    Path(_CFG["checkpoint"]).name,
                "kind":          "mlx",
            },
        })
        log.info("Model ready  (total %.0f ms)", (t4 - t0) * 1000)

        # Auto-prime the default category so first chat turn is fast
        from .server_config import resolve_system_prompt
        _do_prime(resolve_system_prompt("daily_conversation"))

    except Exception as exc:
        _STATE["loading"] = False
        _STATE["error"]   = str(exc)
        log.exception("Model load failed: %s", exc)


def _warmup(model, tokenizer) -> None:
    import mlx.core as mx
    from .inference.generator import prefill
    dummy = mx.array([[tokenizer.encode("hello", add_special_tokens=False)[0]]])
    prefill(model, dummy)


# ── Static files ───────────────────────────────────────────────────────────────

def _ui_file(name: str) -> Path:
    return _UI_DIR / name


def _cache_buster(p: Path) -> str:
    try:
        return str(int(p.stat().st_mtime))
    except Exception:
        return "0"


@app.get("/")
async def root() -> HTMLResponse:
    html_path = _ui_file("chat_demo.html")
    if not html_path.exists():
        return HTMLResponse("<h1>UI not found</h1>", status_code=404)
    html = html_path.read_text(encoding="utf-8")
    js_path  = _ui_file("chat_demo.js")
    css_path = _ui_file("chat_demo.css")
    html = html.replace(
        'src="/static/chat_demo.js"',
        f'src="/static/chat_demo.js?v={_cache_buster(js_path)}"',
    )
    html = html.replace(
        'href="/static/chat_demo.css"',
        f'href="/static/chat_demo.css?v={_cache_buster(css_path)}"',
    )
    return HTMLResponse(html)


@app.get("/static/{filename:path}")
async def static_file(filename: str) -> FileResponse:
    target = _ui_file(filename)
    if not target.exists() or not target.is_file():
        from fastapi.responses import Response
        return Response(status_code=404)
    return FileResponse(
        str(target),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status() -> dict:
    from .server_config import load_mock_config
    if _STATE["mock"]:
        md = load_mock_config()
        return md.get("status", {
            "ready": True,
            "load_timings": {"kind": "mock"},
            "config": _STATE["model_config"],
        })
    return {
        "ready":        _STATE["ready"],
        "loading":      _STATE["loading"],
        "error":        _STATE["error"],
        "config":       _STATE["model_config"],
        "load_timings": _STATE["load_timings"],
    }


@app.get("/api/demo-config")
async def api_demo_config() -> dict:
    from .server_config import (
        category_prompts_for_api, STYLE_CONSTRAINTS,
        TOOL_REGISTRY, SAMPLING_DEFAULTS, load_mock_config,
    )
    base = {
        "mock":                    _STATE["mock"],
        "category_system_prompts": category_prompts_for_api(),
        "style_constraints":        STYLE_CONSTRAINTS,
        "tool_registry":            TOOL_REGISTRY,
        "sampling_defaults":        SAMPLING_DEFAULTS,
        "format_guard": {
            "enabled":          True,
            "ban_im_start":     True,
            "close_bias":       4.0,
            "close_bias_start": 0,
        },
        "max_new_tokens_cap": _CFG.get("max_new_tokens", 4096),
    }
    if _STATE["mock"]:
        md = load_mock_config()
        base["system_prompt_markdown"] = md.get("system_prompt_markdown", "")
        base["categories"]             = md.get("categories", [])
        base["tool_registry"]          = md.get("tool_registry", TOOL_REGISTRY)
        base["style_constraints"]      = md.get("style_constraints", STYLE_CONSTRAINTS)
    else:
        from .server_config import CATEGORY_TITLES
        # Build category list in the same order the UI expects
        _UI_CATS = [
            "daily_conversation", "emotion", "self_awareness",
            "email_summary", "movie_intro", "system_call", "deep_dive",
        ]
        base["system_prompt_markdown"] = ""
        base["categories"] = [
            {"key": k, "title": CATEGORY_TITLES.get(k, k), "examples": []}
            for k in _UI_CATS
        ]
    return base


# ── WebSocket handler ──────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_json({"type": "connected"})

    turn_count   = 0
    abort_event  = threading.Event()
    conversation: list[dict] = []     # multi-turn history: [{role, content}, ...]
    category_key = "daily_conversation"
    sys_text     = ""

    # Schedule auto-prime of the default system prefix so first turn is fast
    if _STATE["ready"] and not _STATE["mock"]:
        from .server_config import resolve_system_prompt as _rsp
        _default_sys = _rsp("daily_conversation")
        asyncio.get_event_loop().run_in_executor(None, lambda: _do_prime(_default_sys))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            action = msg.get("action", "")

            if action == "ping":
                await ws.send_json({"type": "pong"})

            elif action == "clear":
                abort_event.set()
                abort_event  = threading.Event()
                turn_count   = 0
                conversation = []
                await ws.send_json({"type": "cleared"})

            elif action == "abort":
                abort_event.set()

            elif action == "play_system_prompt":
                abort_event.set()
                abort_event = threading.Event()
                new_cat = msg.get("category_key", "daily_conversation")
                if new_cat != category_key:
                    category_key = new_cat
                    conversation = []   # reset history on category switch
                from .server_config import resolve_system_prompt
                sys_text = resolve_system_prompt(category_key)
                await _handle_play_system(ws, msg, abort_event)

            elif action == "chat":
                abort_event.set()
                abort_event = threading.Event()
                turn_count += 1

                # Resolve system text — may change if category differs
                turn_cat = msg.get("category_key", category_key)
                if turn_cat != category_key or not sys_text:
                    if turn_cat != category_key:
                        conversation = []   # reset history when switching category
                    category_key = turn_cat
                    from .server_config import resolve_system_prompt
                    sys_text = resolve_system_prompt(category_key)

                user_text = msg.get("prompt", "").strip()
                conversation.append({"role": "user", "content": user_text})

                response_acc: list[str] = []
                await _handle_chat(
                    ws, msg, conversation, sys_text,
                    turn_count, abort_event, response_acc,
                )

                # Append assistant response to history for next turn
                asst_text = response_acc[0] if response_acc else ""
                if asst_text:
                    conversation.append({"role": "assistant", "content": asst_text})

                _free_metal_cache()

    except WebSocketDisconnect:
        abort_event.set()
    except Exception as exc:
        log.exception("WS error: %s", exc)
        abort_event.set()


# ── Action handlers ────────────────────────────────────────────────────────────

async def _stream_events(
    ws: WebSocket,
    gen_fn,
    abort_event: threading.Event,
    turn_count: int | None = None,
) -> None:
    """Run a blocking generator in a thread; stream events over ws.

    A concurrent async task listens for abort/clear messages so the user
    can interrupt mid-generation without waiting for the next yield.
    """
    loop  = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=512)

    def worker():
        try:
            for ev in gen_fn():
                loop.call_soon_threadsafe(queue.put_nowait, ev)
        except Exception as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "error", "message": str(exc)}
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()

    async def _listen_abort():
        while not abort_event.is_set():
            try:
                raw  = await asyncio.wait_for(ws.receive_text(), timeout=0.05)
                data = json.loads(raw)
                if data.get("action") in ("abort", "clear"):
                    abort_event.set()
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

    abort_task = asyncio.create_task(_listen_abort())
    try:
        while True:
            ev = await queue.get()
            if ev is None:
                break
            if turn_count is not None and isinstance(ev, dict) and ev.get("type") == "meta":
                ev = {**ev, "turns": turn_count}
            try:
                await ws.send_json(ev)
            except Exception:
                break
    finally:
        abort_task.cancel()
        try:
            await abort_task
        except asyncio.CancelledError:
            pass


async def _handle_chat(ws, msg, conversation, sys_text,
                       turn_count, abort_event, response_acc) -> None:
    if _STATE["mock"]:
        await _stream_events(
            ws, lambda: _gen_mock_chat(msg, abort_event), abort_event, turn_count
        )
    elif not _STATE["ready"]:
        await ws.send_json({"type": "error", "message": "Model is still loading, please wait."})
    else:
        await _stream_events(
            ws,
            lambda: _gen_real_chat(
                msg, conversation, sys_text, abort_event, response_acc, turn_count
            ),
            abort_event,
            turn_count,
        )


async def _handle_play_system(ws, msg, abort_event) -> None:
    if _STATE["mock"]:
        await _stream_events(
            ws, lambda: _gen_mock_play_system(msg, abort_event), abort_event
        )
    elif not _STATE["ready"]:
        await ws.send_json({"type": "error", "message": "Model is still loading."})
    else:
        await _stream_events(
            ws, lambda: _gen_real_play_system(msg, abort_event), abort_event
        )


# ── Mock generators ────────────────────────────────────────────────────────────

def _gen_mock_chat(msg: dict, abort_event: threading.Event) -> Generator:
    from .server_config import find_mock_example

    example_id   = msg.get("example_id")
    category_key = msg.get("category_key", "daily_conversation")
    reasoning    = msg.get("reasoning", True)
    mock_data    = _STATE["mock_data"] or {}

    ex        = find_mock_example(mock_data, example_id, category_key)
    cot_md    = ex.get("cot_markdown", "")
    asst_md   = ex.get("assistant_markdown", "")
    tool_flow = ex.get("tool_flow")

    yield {"type": "meta", "prefill_ms": 45.0, "prompt_tokens": 24}

    if reasoning and cot_md:
        acc   = ""
        words = cot_md.split(" ")
        for i, w in enumerate(words):
            if abort_event.is_set():
                return
            acc += ("" if i == 0 else " ") + w
            time.sleep(0.035)
            yield {"type": "reasoning", "markdown": acc}
        yield {"type": "assistant_split"}

    if tool_flow:
        if tool_flow.get("trigger_output"):
            time.sleep(0.08)
            yield {"type": "tool_action",
                   "call": tool_flow["trigger_output"],
                   "system_result": None, "phase": "request"}
        if tool_flow.get("system_result"):
            time.sleep(0.08)
            yield {"type": "tool_action",
                   "call": None,
                   "system_result": tool_flow["system_result"], "phase": "response"}

    n   = 0
    t0  = time.perf_counter()
    words = asst_md.split(" ")
    for i, w in enumerate(words):
        if abort_event.is_set():
            break
        text  = ("" if i == 0 else " ") + w
        n    += 1
        time.sleep(0.028)
        tok_s = n / max(time.perf_counter() - t0, 1e-9)
        yield {"type": "token", "text": text, "n": n, "tok_s": round(tok_s, 1)}

    t_total = (time.perf_counter() - t0) * 1000
    yield {"type": "done",
           "tok_s": round(n / max(t_total / 1000, 1e-9), 1),
           "ttft_ms": 45.0, "prefill_ms": 45.0,
           "total_tokens": n, "total_ms": round(t_total, 1)}


def _gen_mock_play_system(msg: dict, abort_event: threading.Event) -> Generator:
    from .server_config import resolve_system_prompt, CATEGORY_TITLES

    category_key   = msg.get("category_key", "daily_conversation")
    sys_text       = resolve_system_prompt(category_key)
    category_title = CATEGORY_TITLES.get(category_key, category_key)

    yield {"type": "intro_start", "category_title": category_title}

    n   = 0
    t0  = time.perf_counter()
    words = sys_text.split(" ")
    for i, w in enumerate(words):
        if abort_event.is_set():
            break
        text  = ("" if i == 0 else " ") + w
        n    += 1
        time.sleep(0.018)
        tok_s = n / max(time.perf_counter() - t0, 1e-9)
        yield {"type": "token", "text": text, "n": n, "tok_s": round(tok_s, 1)}

    t_total = (time.perf_counter() - t0) * 1000
    yield {"type": "done", "play_system_only": True,
           "tok_s": round(n / max(t_total / 1000, 1e-9), 1),
           "ttft_ms": 18.0, "prefill_ms": 18.0,
           "total_tokens": n, "total_ms": round(t_total, 1)}


# ── Real-model generators ──────────────────────────────────────────────────────

def _build_prompt_from_history(history: list[dict], sys_text: str, reasoning: bool) -> str:
    """Build full ChatML string from conversation history."""
    parts = [f"<|im_start|>system\n{sys_text}<|im_end|>\n"]
    for turn in history:
        parts.append(f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    if reasoning:
        parts.append("<think>\n")
    return "".join(parts)


def _build_suffix_from_history(history: list[dict], reasoning: bool) -> str:
    """Build the suffix after the system block (for continuation prefill)."""
    parts = []
    for turn in history:
        parts.append(f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    if reasoning:
        parts.append("<think>\n")
    return "".join(parts)


def _gen_real_chat(
    msg: dict,
    conversation: list[dict],
    sys_text: str,
    abort_event: threading.Event,
    response_acc: list | None = None,
    turn_count: int = 0,
) -> Generator:
    import mlx.core as mx
    from .inference.generator import prefill, prefill_continuation, decode_step, _eval_states
    from .mlx_model.state_utils import clone_mamba_states, clone_kv_caches
    from .inference.sampler import (
        apply_repetition_penalty, apply_presence_frequency_penalty, sample_token,
    )
    from .utils.config import GenerationConfig
    from .cot_middleware import CotMiddleware, CotMiddlewareConfig, render_health_line 
    
    model     = _STATE["model"]
    tokenizer = _STATE["tokenizer"]
    mw_deps   = _STATE.get("mw_deps")
    mw_cfg    = _STATE.get("mw_cfg") or CotMiddlewareConfig()

    if mw_deps is None:
        yield {"type": "error", "message": "Model middleware not initialized — restart server."}
        return

    stop_ids = list(mw_deps.stop_ids)   # merged: original EOS + single-token </final>

    sampling    = msg.get("sampling", {})
    no_eos_stop = bool(msg.get("no_eos_stop", False))
    reasoning   = bool(msg.get("reasoning", True))
    max_tokens  = int(msg.get("max_tokens", _CFG.get("max_new_tokens", 4096)))
    enable_cot_mw = msg.get("enable_cot_middleware", True)

    # Middleware config overrides (safety limit: max_tokens is still checked)
    reasoning_budget = int(msg.get("reasoning_budget", mw_cfg.reasoning_budget or 500))
    final_min_tokens = int(msg.get("final_min_tokens", mw_cfg.final_min_tokens or 16))

    gen_cfg = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=float(sampling.get("temperature", 0.8)),
        top_k=int(sampling.get("top_k", 40)),
        top_p=float(sampling.get("top_p", 0.9)),
        min_p=float(sampling.get("min_p", 0.05)),
        repetition_penalty=float(sampling.get("repetition_penalty", 1.1)),
        stop_token_ids=stop_ids,
        no_eos_stop=no_eos_stop,
    )

    # ── Prefill: continuation from primed system prefix when available ────────
    t0     = time.perf_counter()
    primed = _prime_get(sys_text)
    if primed and conversation:
        m_states = clone_mamba_states(primed["mamba_states"])
        k_caches = clone_kv_caches(primed["kv_caches"])
        suffix   = _build_suffix_from_history(conversation, reasoning)
        suf_ids  = tokenizer.encode(suffix)
        prompt_ids = primed["ids"] + suf_ids
        pf_logits, mamba_states, kv_caches = prefill_continuation(
            model, mx.array([suf_ids]), m_states, k_caches, step=primed["pos"]
        )
    else:
        full_prompt = _build_prompt_from_history(conversation, sys_text, reasoning)
        prompt_ids  = tokenizer.encode(full_prompt)
        pf_logits, mamba_states, kv_caches = prefill(model, mx.array([prompt_ids]))

    generated = list(prompt_ids)   # all token IDs including prompt (used for rep. penalty)
    n_text = 0  # Initialize before first _sample() call

    def _sample(logits_1d, debug_label=""):
        import mlx.core as mx
        logits_min = float(mx.min(logits_1d))
        logits_max = float(mx.max(logits_1d))
        logits_mean = float(mx.mean(logits_1d))
        if debug_label and (n_text < 5 or n_text % 20 == 0):
            log.info(f"[logits] {debug_label}: min={logits_min:.2f}, max={logits_max:.2f}, mean={logits_mean:.4f}")

        logits_1d = apply_repetition_penalty(
            logits_1d, generated, gen_cfg.repetition_penalty, gen_cfg.repetition_window)
        logits_1d = apply_presence_frequency_penalty(
            logits_1d, generated,
            gen_cfg.presence_penalty, gen_cfg.frequency_penalty, gen_cfg.repetition_window)
        return sample_token(
            logits_1d, gen_cfg.temperature, gen_cfg.top_k, gen_cfg.top_p, gen_cfg.min_p)

    # ── Model apply wrapper for <final>\n injection ───────────────────────────
    # Matches the signature CotMiddleware.maybe_inject_final expects:
    # (x[1,N], caches=(ms, kv), seq_pos_scalar) → (logits[1,N,V], new_caches)
    def _model_apply(x, caches, seq_pos_arr):
        ms, kv = caches
        logits_all, _, new_ms, new_kv = model(
            x, step=int(seq_pos_arr), mamba_states=ms, kv_caches=kv)
        return logits_all, (new_ms, new_kv)

    # ── Per-turn middleware ────────────────────────────────────────────────────
    # Create per-turn config with client-provided overrides
    turn_mw_cfg = CotMiddlewareConfig(
        enabled=mw_cfg.enabled and enable_cot_mw,
        ban_im_start=mw_cfg.ban_im_start,
        close_bias_value=mw_cfg.close_bias_value,
        close_bias_max=mw_cfg.close_bias_max,
        close_bias_start=mw_cfg.close_bias_start,
        reasoning_budget=reasoning_budget,
        final_min_tokens=final_min_tokens,
        force_final_inject=mw_cfg.force_final_inject,
    )
    mw = CotMiddleware(
        deps=mw_deps,
        cfg=turn_mw_cfg,
        reasoning=reasoning,
        model_apply=_model_apply,
    )

    # Apply format guard to prefill logits before sampling the first token
    first_logits = mw.transform_logits(pf_logits[0])
    first_tok    = _sample(first_logits, debug_label="first_logits(after transform)")
    _eval_states(mamba_states, kv_caches)
    generated.append(first_tok)

    t_first    = time.perf_counter()
    prefill_ms = (t_first - t0) * 1000
    yield {"type": "meta", "prefill_ms": round(prefill_ms, 1), "prompt_tokens": len(prompt_ids)}

    t_decode_start  = t_first
    asst_split_sent = False
    answer_parts: list[str] = []
    past_final      = False

    def elapsed_s() -> float:
        return max(time.perf_counter() - t_decode_start, 1e-9)

    def _process_mw_events(it) -> tuple[list, bool]:
        """Consume middleware events; patch n/tok_s; return (ws_events, stop_flag)."""
        nonlocal n_text
        ws_evs: list = []
        stop_now = False
        for ev in it:
            if ev.get("__stop__"):
                stop_now = True
            elif ev.get("type") == "reasoning":
                ws_evs.append(ev)
            elif ev.get("type") == "token":
                n_text += 1
                text = ev.get("text", "")
                answer_parts.append(text)
                ws_evs.append({
                    "type": "token", "text": text,
                    "n": n_text, "tok_s": round(n_text / elapsed_s(), 1),
                })
        return ws_evs, stop_now

    def _check_asst_split() -> bool:
        nonlocal asst_split_sent
        if reasoning and not asst_split_sent and mw.mode in ("between", "final", "done"):
            asst_split_sent = True
            return True
        return False

    # ── Process first token ───────────────────────────────────────────────────
    log.info(f"[mw] first_tok={first_tok}, mw.enabled={turn_mw_cfg.enabled}")
    first_evs, stop_now = _process_mw_events(
        mw.step(first_tok, n_out=n_text, elapsed_s_fn=elapsed_s))

    if _check_asst_split():
        yield {"type": "assistant_split"}
    for ev in first_evs:
        yield ev

    if not no_eos_stop and mw.should_break(first_tok):
        stop_now = True

    if stop_now:
        flush_evs, _ = _process_mw_events(mw.flush(n_out=n_text, elapsed_s_fn=elapsed_s))
        for ev in flush_evs:
            yield ev
        t_total = (time.perf_counter() - t0) * 1000
        _print_turn_summary(turn_count, [mw.reasoning_markdown], answer_parts, n_text, t_total, prefill_ms)
        if response_acc is not None:
            response_acc.append("".join(answer_parts))
        done_ev = _done_event(n_text, t_total, prefill_ms)
        if enable_cot_mw:
            done_ev["cot_health_report"] = render_health_line(mw.health_report())
        yield done_ev
        return

    # ── Decode loop ───────────────────────────────────────────────────────────
    # Note: max_tokens is a safety limit; actual stopping controlled by middleware
    # (reasoning_budget, final_min_tokens, </final>, <|im_end|>)
    prev_tok      = first_tok
    decode_step_n = 0
    inj_logits: mx.array | None = None
    use_inj       = False
    tokens_generated = 1  # first_tok already generated

    while tokens_generated < max_tokens:
        if abort_event.is_set():
            break
        tokens_generated += 1

        if use_inj:
            logits_1d = inj_logits
            use_inj   = False
            inj_logits = None
            tok = _sample(logits_1d, debug_label="injected_logits(no_transform)")
            log.info(f"[inj] injected tok={tok}")
        else:
            decode_step_n += 1
            logits, mamba_states, kv_caches = decode_step(
                model, prev_tok, mamba_states, kv_caches, step=decode_step_n)
            _eval_states(mamba_states, kv_caches)
            logits_1d = logits[0]

            # Apply format guard (ban mask + dynamic close-bias) before sampling
            logits_1d = mw.transform_logits(logits_1d)
            tok = _sample(logits_1d, debug_label="decode_logits(after_transform)")

        generated.append(tok)
        prev_tok = tok

        if mw.should_break(tok) and not no_eos_stop:
            break

        if past_final:
            # no_eos_stop: continue emitting after </final> — middleware is done
            raw = tokenizer.decode([tok], skip_special_tokens=False)
            if raw:
                n_text += 1
                answer_parts.append(raw)
                yield {"type": "token", "text": raw,
                       "n": n_text, "tok_s": round(n_text / elapsed_s(), 1)}
            continue

        # Feed token to middleware (decodes internally via decode-all-diff)
        evs, stop_now = _process_mw_events(
            mw.step(tok, n_out=n_text, elapsed_s_fn=elapsed_s))

        # <final>\n injection — fires once when splitter just moved to "between"
        caches = (mamba_states, kv_caches)
        caches, new_pos, last_row, did_inject, _ = mw.maybe_inject_final(
            caches=caches, pos=len(generated))
        if False and did_inject:  # FIX H1: Disable injection entirely
            import mlx.core as mx
            mamba_states, kv_caches = caches
            # Track injected ids in generated so repetition penalty stays accurate
            for inj_id in mw.deps.final_inject_ids:
                generated.append(inj_id)
            inj_logits = last_row
            inj_min = float(mx.min(inj_logits))
            inj_max = float(mx.max(inj_logits))
            inj_mean = float(mx.mean(inj_logits))
            log.info(f"[inj] injected_logits: min={inj_min:.2f}, max={inj_max:.2f}, mean={inj_mean:.4f}")
            use_inj    = True
            if not asst_split_sent:
                asst_split_sent = True
                yield {"type": "assistant_split"}

        if _check_asst_split():
            yield {"type": "assistant_split"}

        for ev in evs:
            yield ev

        if stop_now:
            if no_eos_stop:
                past_final = True
            else:
                break

    # Check if we hit max_tokens safety limit (should be rare with proper middleware budgets)
    if tokens_generated >= max_tokens:
        log.info("Decode loop: max_tokens safety limit reached (%d tokens)", max_tokens)

    # Flush any remaining splitter buffer
    flush_evs, _ = _process_mw_events(mw.flush(n_out=n_text, elapsed_s_fn=elapsed_s))
    for ev in flush_evs:
        yield ev

    log.info("[mw] %s", render_health_line(mw.health_report()))

    t_total = (time.perf_counter() - t0) * 1000
    r_parts = [mw.reasoning_markdown] if mw.reasoning_markdown else []
    _print_turn_summary(turn_count, r_parts, answer_parts, n_text, t_total, prefill_ms)
    if response_acc is not None:
        response_acc.append("".join(answer_parts))
    done_ev = _done_event(n_text, t_total, prefill_ms)
    if enable_cot_mw:
        done_ev["cot_health_report"] = render_health_line(mw.health_report())
    yield done_ev


def _gen_real_play_system(msg: dict, abort_event: threading.Event) -> Generator:
    """Prime system prefix and type-animate its text for the UI (real mode)."""
    from .server_config import resolve_system_prompt, CATEGORY_TITLES

    category_key = msg.get("category_key", "daily_conversation")
    sys_text     = resolve_system_prompt(category_key)
    cat_title    = CATEGORY_TITLES.get(category_key, category_key)

    yield {"type": "intro_start", "category_title": cat_title}

    # Prime the system prefix in a background thread so the next chat turn is fast.
    # Start before the animation so priming overlaps with the type-out delay.
    if _prime_get(sys_text) is None:
        threading.Thread(target=_do_prime, args=(sys_text,), daemon=True).start()

    n   = 0
    t0  = time.perf_counter()
    for i, w in enumerate(sys_text.split(" ")):
        if abort_event.is_set():
            break
        text  = ("" if i == 0 else " ") + w
        n    += 1
        time.sleep(0.016)
        tok_s = n / max(time.perf_counter() - t0, 1e-9)
        yield {"type": "token", "text": text, "n": n, "tok_s": round(tok_s, 1)}

    t_total = (time.perf_counter() - t0) * 1000
    yield {"type": "done", "play_system_only": True,
           "tok_s": round(n / max(t_total / 1000, 1e-9), 1),
           "ttft_ms": 18.0, "prefill_ms": 18.0,
           "total_tokens": n, "total_ms": round(t_total, 1)}


# ── Utilities ──────────────────────────────────────────────────────────────────

def _print_turn_summary(
    turn: int,
    reasoning_parts: list[str],
    answer_parts: list[str],
    n_tokens: int,
    total_ms: float,
    prefill_ms: float,
) -> None:
    tok_s  = n_tokens / max(total_ms / 1000, 1e-9)
    r_text = "".join(reasoning_parts)
    a_text = "".join(answer_parts)
    snippet = lambda s, n: s[:n] + ("…" if len(s) > n else "")
    try:
        from rich.console import Console
        c = Console(stderr=True)
        c.print(
            f"[bold cyan]Turn {turn}[/bold cyan] | "
            f"[green]{n_tokens} tok[/green] | "
            f"[yellow]{tok_s:.1f} tok/s[/yellow] | "
            f"{total_ms:.0f}ms (prefill {prefill_ms:.0f}ms)"
        )
        if r_text:
            c.print(f"  [dim]<think>[/dim] {snippet(r_text, 160)}")
        if a_text:
            c.print(f"  [white]{snippet(a_text, 240)}[/white]")
    except Exception:
        log.info(
            "Turn %d | %d tok @ %.1f tok/s | %.0fms | answer: %s",
            turn, n_tokens, tok_s, total_ms, snippet(a_text, 120),
        )


def _done_event(n_text: int, t_total_ms: float, prefill_ms: float) -> dict:
    tok_s = n_text / max(t_total_ms / 1000, 1e-9)
    return {
        "type":         "done",
        "tok_s":        round(tok_s, 1),
        "ttft_ms":      round(prefill_ms, 1),
        "prefill_ms":   round(prefill_ms, 1),
        "total_tokens": n_text,
        "total_ms":     round(t_total_ms, 1),
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    global _CFG

    parser = argparse.ArgumentParser(description="Mamba Edge Chat Server")
    parser.add_argument("--mock",           action="store_true",   help="Use mock responses (no model)")
    parser.add_argument("--checkpoint",     default=_DEFAULT_CKPT, help="Path to .npz checkpoint")
    parser.add_argument("--tokenizer",      default=_DEFAULT_TOK,  help="HuggingFace tokenizer directory")
    parser.add_argument("--host",           default="0.0.0.0",     help="Bind host")
    parser.add_argument("--port",           type=int, default=8000, help="Bind port")
    parser.add_argument("--dtype",          default="bf16",        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--quantize",       type=int, default=0,   choices=[0, 4, 8])
    args = parser.parse_args()

    _CFG.update(vars(args))
    _CFG["max_new_tokens"] = args.max_new_tokens

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
