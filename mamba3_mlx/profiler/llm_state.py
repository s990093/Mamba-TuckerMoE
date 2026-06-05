"""Thread-safe LLM telemetry published by chat_demo / run.py."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class LlmTelemetry:
    tokens_per_sec: float | None = None
    prompt_tps: float | None = None
    decode_tps: float | None = None
    latency_ms: float | None = None
    kv_cache_gb: float | None = None
    context_length: int | None = None
    batch_size: int = 1


@dataclass
class InferenceState:
    running: bool = False
    model: str | None = None
    phase: str = "idle"  # idle | prefill | decode


_lock = threading.Lock()
_llm = LlmTelemetry()
_state = InferenceState()


def update_llm_telemetry(**kwargs) -> None:
    with _lock:
        for key, value in kwargs.items():
            if hasattr(_llm, key):
                setattr(_llm, key, value)


def set_inference_state(**kwargs) -> None:
    with _lock:
        for key, value in kwargs.items():
            if hasattr(_state, key):
                setattr(_state, key, value)


def get_llm_telemetry() -> LlmTelemetry:
    with _lock:
        return LlmTelemetry(**vars(_llm))


def get_inference_state() -> InferenceState:
    with _lock:
        return InferenceState(**vars(_state))


def reset_llm_idle(model: str | None = None) -> None:
    with _lock:
        _llm.tokens_per_sec = None
        _llm.prompt_tps = None
        _llm.decode_tps = None
        _llm.latency_ms = None
        _llm.kv_cache_gb = None
        _llm.context_length = None
        _llm.batch_size = 1
        _state.running = False
        _state.phase = "idle"
        if model is not None:
            _state.model = model
