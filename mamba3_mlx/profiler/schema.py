"""Fixed JSON schema for the profiler WebSocket stream.

Frontend should only read nested fields (e.g. ``data.snapshot.gpu.usage_percent``).
``schema_version`` bumps only on breaking field renames/removals.
"""

from __future__ import annotations

from typing import Any, TypedDict


SCHEMA_VERSION = 1


class GpuMetrics(TypedDict, total=False):
    usage_percent: float | None
    idle_percent: float | None
    frequency_mhz: int | None
    power_mw: int | None
    available: bool


class CpuMetrics(TypedDict, total=False):
    usage_percent: float | None
    per_core_percent: list[float]


class MemoryMetrics(TypedDict, total=False):
    used_gb: float | None
    available_gb: float | None
    total_gb: float | None
    percent: float | None


class SwapMetrics(TypedDict, total=False):
    used_gb: float | None
    percent: float | None


class ThermalMetrics(TypedDict, total=False):
    pressure: str | None
    throttling: bool | None


class LlmMetrics(TypedDict, total=False):
    tokens_per_sec: float | None
    prompt_tps: float | None
    decode_tps: float | None
    latency_ms: float | None
    kv_cache_gb: float | None
    context_length: int | None
    batch_size: int | None


class InferenceStateMetrics(TypedDict, total=False):
    running: bool
    model: str | None
    phase: str | None


class MetricsSnapshot(TypedDict):
    timestamp: float
    gpu: GpuMetrics
    cpu: CpuMetrics
    memory: MemoryMetrics
    swap: SwapMetrics
    thermal: ThermalMetrics
    llm: LlmMetrics
    state: InferenceStateMetrics


class WsEnvelope(TypedDict):
    schema_version: int
    snapshot: MetricsSnapshot
    history: list[MetricsSnapshot]


def _null_gpu() -> GpuMetrics:
    return {
        "usage_percent": None,
        "idle_percent": None,
        "frequency_mhz": None,
        "power_mw": None,
        "available": False,
    }


def _null_cpu() -> CpuMetrics:
    return {"usage_percent": None, "per_core_percent": []}


def _null_memory() -> MemoryMetrics:
    return {
        "used_gb": None,
        "available_gb": None,
        "total_gb": None,
        "percent": None,
    }


def _null_swap() -> SwapMetrics:
    return {"used_gb": None, "percent": None}


def _null_thermal() -> ThermalMetrics:
    return {"pressure": None, "throttling": None}


def _null_llm() -> LlmMetrics:
    return {
        "tokens_per_sec": None,
        "prompt_tps": None,
        "decode_tps": None,
        "latency_ms": None,
        "kv_cache_gb": None,
        "context_length": None,
        "batch_size": 1,
    }


def _null_state() -> InferenceStateMetrics:
    return {"running": False, "model": None, "phase": "idle"}


def empty_snapshot(ts: float | None = None) -> MetricsSnapshot:
    import time

    return {
        "timestamp": ts if ts is not None else time.time(),
        "gpu": _null_gpu(),
        "cpu": _null_cpu(),
        "memory": _null_memory(),
        "swap": _null_swap(),
        "thermal": _null_thermal(),
        "llm": _null_llm(),
        "state": _null_state(),
    }


def make_envelope(snapshot: MetricsSnapshot, history: list[MetricsSnapshot]) -> WsEnvelope:
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot": snapshot,
        "history": history,
    }


def round_float(v: float | None, ndigits: int = 1) -> float | None:
    if v is None:
        return None
    return round(float(v), ndigits)


def round_gb(v: float | None) -> float | None:
    if v is None:
        return None
    return round(float(v), 2)
