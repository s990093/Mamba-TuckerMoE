"""Thin hooks for chat_demo / run.py to publish LLM telemetry."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .llm_state import reset_llm_idle, set_inference_state, update_llm_telemetry

_PUBLISH_URL = os.environ.get("MAMBA_PROFILER_URL", "").rstrip("/")


def _publish_remote(llm: dict | None = None, state: dict | None = None) -> None:
    if not _PUBLISH_URL:
        return
    payload: dict = {}
    if llm:
        payload["llm"] = llm
    if state:
        payload["state"] = state
    if not payload:
        return
    req = urllib.request.Request(
        f"{_PUBLISH_URL}/telemetry",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=0.15)
    except (urllib.error.URLError, TimeoutError, OSError):
        pass


def _sync(llm: dict | None = None, state: dict | None = None) -> None:
    _publish_remote(llm=llm, state=state)


def bind_model_name(name: str) -> None:
    set_inference_state(model=name)
    _sync(state={"model": name})


def on_turn_start(*, model: str | None = None, context_length: int | None = None) -> None:
    if model:
        bind_model_name(model)
    set_inference_state(running=True, phase="prefill")
    llm = {
        "tokens_per_sec": None,
        "prompt_tps": None,
        "decode_tps": None,
        "latency_ms": None,
        "context_length": context_length,
        "batch_size": 1,
    }
    update_llm_telemetry(**llm)
    _sync(llm=llm, state={"running": True, "phase": "prefill", "model": model})


def on_prefill_done(
    *,
    prompt_tokens: int,
    prefill_ms: float,
    context_length: int | None = None,
) -> None:
    prompt_tps = prompt_tokens / max(prefill_ms / 1000.0, 1e-6)
    llm = {
        "prompt_tps": round(prompt_tps, 1),
        "context_length": context_length or prompt_tokens,
    }
    update_llm_telemetry(**llm)
    set_inference_state(phase="decode")
    _sync(llm=llm, state={"phase": "decode", "running": True})


def on_decode_tick(
    *,
    n_out: int,
    elapsed_s: float,
    ttft_ms: float | None = None,
    context_length: int | None = None,
) -> None:
    decode_tps = n_out / max(elapsed_s, 1e-6)
    llm = {
        "tokens_per_sec": round(decode_tps, 1),
        "decode_tps": round(decode_tps, 1),
        "latency_ms": ttft_ms,
        "context_length": context_length,
    }
    update_llm_telemetry(**llm)
    _sync(llm=llm, state={"phase": "decode", "running": True})


def on_turn_done(
    *,
    total_tokens: int,
    total_ms: float,
    ttft_ms: float | None,
    prefill_ms: float,
    prompt_tokens: int,
    context_length: int | None = None,
) -> None:
    decode_s = max((total_ms - prefill_ms) / 1000.0, 1e-6)
    decode_tps = total_tokens / decode_s
    prompt_tps = prompt_tokens / max(prefill_ms / 1000.0, 1e-6)
    llm = {
        "tokens_per_sec": round(total_tokens / max(total_ms / 1000.0, 1e-6), 1),
        "prompt_tps": round(prompt_tps, 1),
        "decode_tps": round(decode_tps, 1),
        "latency_ms": ttft_ms,
        "context_length": context_length,
    }
    update_llm_telemetry(**llm)
    set_inference_state(running=False, phase="idle")
    _sync(llm=llm, state={"running": False, "phase": "idle"})


def on_turn_abort(model: str | None = None) -> None:
    reset_llm_idle(model=model)
    _sync(
        llm={
            "tokens_per_sec": None,
            "prompt_tps": None,
            "decode_tps": None,
            "latency_ms": None,
            "context_length": None,
            "batch_size": 1,
        },
        state={"running": False, "phase": "idle", "model": model},
    )
