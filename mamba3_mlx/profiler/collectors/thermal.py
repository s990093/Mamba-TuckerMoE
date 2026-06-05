"""macOS thermal + memory pressure."""

from __future__ import annotations

import re
import shutil
import subprocess

from ..schema import ThermalMetrics
from .gpu import get_gpu_metrics, get_gpu_peak_frequency_mhz

_PRESSURE_RE = re.compile(
    r"System-wide memory free percentage:\s*(\d+)%|"
    r"memory pressure:\s*(\w+)",
    re.IGNORECASE,
)


def _run_memory_pressure() -> str | None:
    if shutil.which("memory_pressure") is None:
        return None
    try:
        out = subprocess.run(
            ["memory_pressure"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (out.stdout or "") + (out.stderr or "")


def _parse_pressure(text: str) -> str | None:
    lower = text.lower()
    if "warn" in lower:
        return "warn"
    if "critical" in lower:
        return "critical"
    if "normal" in lower or "nominal" in lower:
        return "nominal"
    m = re.search(r"System-wide memory free percentage:\s*(\d+)%", text)
    if m:
        free_pct = int(m.group(1))
        if free_pct < 10:
            return "critical"
        if free_pct < 25:
            return "warn"
        return "nominal"
    return None


def _detect_throttling() -> bool | None:
    gpu = get_gpu_metrics()
    usage = gpu.get("usage_percent")
    freq = gpu.get("frequency_mhz")
    peak = get_gpu_peak_frequency_mhz()
    if usage is None or freq is None or peak is None:
        return None
    if usage >= 95.0 and freq < peak * 0.85:
        return True
    if usage >= 95.0:
        return False
    return False


def get_thermal_metrics() -> ThermalMetrics:
    raw = _run_memory_pressure()
    pressure = _parse_pressure(raw) if raw else None
    return {
        "pressure": pressure,
        "throttling": _detect_throttling(),
    }
