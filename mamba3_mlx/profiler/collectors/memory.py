"""RAM and swap metrics via psutil."""

from __future__ import annotations

from ..schema import MemoryMetrics, SwapMetrics, round_gb

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


def get_memory_metrics() -> MemoryMetrics:
    if psutil is None:
        return {
            "used_gb": None,
            "available_gb": None,
            "total_gb": None,
            "percent": None,
        }
    mem = psutil.virtual_memory()
    return {
        "used_gb": round_gb(mem.used / 1024**3),
        "available_gb": round_gb(mem.available / 1024**3),
        "total_gb": round_gb(mem.total / 1024**3),
        "percent": round(mem.percent, 1),
    }


def get_swap_metrics() -> SwapMetrics:
    if psutil is None:
        return {"used_gb": None, "percent": None}
    swap = psutil.swap_memory()
    return {
        "used_gb": round_gb(swap.used / 1024**3),
        "percent": round(swap.percent, 1),
    }
