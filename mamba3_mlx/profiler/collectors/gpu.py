"""Apple Silicon GPU metrics via ioreg (no sudo required).

Primary source: ioreg -r -d 1 -c AGXAccelerator → PerformanceStatistics dict.
Falls back to null if ioreg is unavailable or returns no data.

Polled in a background thread every sample_ms milliseconds.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from typing import Any

from ..schema import GpuMetrics

# ── Regex patterns against the ioreg PerformanceStatistics line ──────────────
_RE_DEVICE  = re.compile(r'"Device Utilization %"\s*=\s*([\d.]+)')
_RE_RENDER  = re.compile(r'"Renderer Utilization %"\s*=\s*([\d.]+)')
_RE_TILER   = re.compile(r'"Tiler Utilization %"\s*=\s*([\d.]+)')
_RE_MEM_USE = re.compile(r'"In use system memory"\s*=\s*(\d+)')       # without "(driver)"
_RE_MEM_ALL = re.compile(r'"Alloc system memory"\s*=\s*(\d+)')

_lock = threading.Lock()
_gpu: dict[str, Any] = {
    "usage_percent":    None,
    "idle_percent":     None,
    "frequency_mhz":    None,   # not available from ioreg; stays None
    "power_mw":         None,   # not available without powermetrics; stays None
    "memory_used_mb":   None,
    "memory_alloc_mb":  None,
}

_monitor_thread: threading.Thread | None = None
_stop_event = threading.Event()


def _sample_once() -> None:
    """Run one ioreg query and update _gpu metrics."""
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "AGXAccelerator"],
            capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return

    for line in result.stdout.splitlines():
        if "PerformanceStatistics" not in line:
            continue
        m_dev  = _RE_DEVICE.search(line)
        m_rnd  = _RE_RENDER.search(line)
        m_til  = _RE_TILER.search(line)
        m_muse = _RE_MEM_USE.search(line)
        m_mall = _RE_MEM_ALL.search(line)

        with _lock:
            if m_dev:
                pct = float(m_dev.group(1))
                _gpu["usage_percent"] = pct
                _gpu["idle_percent"]  = round(100.0 - pct, 1)
            if m_muse:
                _gpu["memory_used_mb"]  = round(int(m_muse.group(1)) / 1024**2, 1)
            if m_mall:
                _gpu["memory_alloc_mb"] = round(int(m_mall.group(1)) / 1024**2, 1)
        break   # only one AGXAccelerator entry expected


def _monitor_loop(sample_ms: int) -> None:
    interval = sample_ms / 1000.0
    _stop_event.clear()
    while not _stop_event.is_set():
        _sample_once()
        _stop_event.wait(timeout=interval)


def start_gpu_monitor(sample_ms: int = 1000) -> bool:
    """Start background ioreg poller. Returns False if ioreg unavailable."""
    global _monitor_thread
    if shutil.which("ioreg") is None:
        return False
    if _monitor_thread is not None and _monitor_thread.is_alive():
        return True
    _stop_event.clear()
    _monitor_thread = threading.Thread(
        target=_monitor_loop,
        args=(sample_ms,),
        name="profiler-gpu-ioreg",
        daemon=True,
    )
    _monitor_thread.start()
    return True


def stop_gpu_monitor() -> None:
    global _monitor_thread
    _stop_event.set()
    _monitor_thread = None


def get_gpu_peak_frequency_mhz() -> int | None:
    return None   # not available from ioreg


def get_gpu_metrics() -> GpuMetrics:
    with _lock:
        has = _gpu["usage_percent"] is not None
        return {
            "usage_percent":   _gpu["usage_percent"],
            "idle_percent":    _gpu["idle_percent"],
            "frequency_mhz":  _gpu["frequency_mhz"],
            "power_mw":       _gpu["power_mw"],
            "available":      has,
        }
