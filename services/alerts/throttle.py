"""
Alert Throttling and Temporal Smoothing.
Prevents alert flapping in live streams and deduplicates repeated alerts.
"""

from __future__ import annotations

import time
from collections import defaultdict


class AlertThrottle:
    def __init__(self, window_seconds: int = 30, max_alerts_per_window: int = 3) -> None:
        self.window_seconds = window_seconds
        self.max_alerts_per_window = max_alerts_per_window
        self._history: dict[str, list[float]] = defaultdict(list)

    def should_suppress(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        valid = [t for t in self._history[key] if t > cutoff]
        if len(valid) >= self.max_alerts_per_window:
            return True  # Suppress to prevent flapping
        valid.append(now)
        self._history[key] = valid
        return False
