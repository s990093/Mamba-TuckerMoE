"""Thread-safe rolling buffer of metric snapshots."""

from __future__ import annotations

import threading
from collections import deque

from .schema import MetricsSnapshot


class RollingBuffer:
    def __init__(self, maxlen: int) -> None:
        self._maxlen = max(1, int(maxlen))
        self._items: deque[MetricsSnapshot] = deque(maxlen=self._maxlen)
        self._lock = threading.Lock()

    def push(self, snapshot: MetricsSnapshot) -> None:
        with self._lock:
            self._items.append(snapshot)

    def list(self) -> list[MetricsSnapshot]:
        with self._lock:
            return list(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
