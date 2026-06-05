"""CPU utilization via psutil."""

from __future__ import annotations

from ..schema import CpuMetrics, round_float

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

_first_sample = True


def get_cpu_metrics() -> CpuMetrics:
    global _first_sample
    if psutil is None:
        return {"usage_percent": None, "per_core_percent": []}
    per_core = psutil.cpu_percent(percpu=True)
    if _first_sample:
        # Prime the rolling average window.
        psutil.cpu_percent()
        per_core = psutil.cpu_percent(percpu=True)
        _first_sample = False
    usage = sum(per_core) / max(len(per_core), 1)
    return {
        "usage_percent": round_float(usage),
        "per_core_percent": [round_float(x, 0) or 0.0 for x in per_core],
    }
