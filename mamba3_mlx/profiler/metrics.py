"""Aggregate system + LLM metrics into the fixed profiler schema."""

from __future__ import annotations

import time

from .collectors import (
    get_cpu_metrics,
    get_gpu_metrics,
    get_memory_metrics,
    get_swap_metrics,
    get_thermal_metrics,
)
from .llm_state import get_inference_state, get_llm_telemetry
from .schema import LlmMetrics, MetricsSnapshot, InferenceStateMetrics, round_float


def _get_llm_metrics() -> LlmMetrics:
    t = get_llm_telemetry()
    return {
        "tokens_per_sec": round_float(t.tokens_per_sec),
        "prompt_tps": round_float(t.prompt_tps),
        "decode_tps": round_float(t.decode_tps),
        "latency_ms": round_float(t.latency_ms, 0),
        "kv_cache_gb": round_float(t.kv_cache_gb, 2),
        "context_length": t.context_length,
        "batch_size": t.batch_size,
    }


def _get_state_metrics() -> InferenceStateMetrics:
    s = get_inference_state()
    return {
        "running": s.running,
        "model": s.model,
        "phase": s.phase,
    }


def collect_metrics() -> MetricsSnapshot:
    return {
        "timestamp": time.time(),
        "gpu": get_gpu_metrics(),
        "cpu": get_cpu_metrics(),
        "memory": get_memory_metrics(),
        "swap": get_swap_metrics(),
        "thermal": get_thermal_metrics(),
        "llm": _get_llm_metrics(),
        "state": _get_state_metrics(),
    }
