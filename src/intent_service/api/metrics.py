"""A minimal, dependency-free request counter.

Deliberately not Prometheus/OpenTelemetry -- pulling in an observability
stack for a course capstone would be its own architectural decision. This
gives /metrics something real to report while keeping the surface small
enough to unit-test in one file.
"""

import threading
from dataclasses import dataclass, field


@dataclass
class MetricsRegistry:
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _total_requests: int = 0
    _total_errors: int = 0
    _by_intent: dict[str, int] = field(default_factory=dict)

    def record_prediction(self, intent: str) -> None:
        with self._lock:
            self._total_requests += 1
            self._by_intent[intent] = self._by_intent.get(intent, 0) + 1

    def record_error(self) -> None:
        with self._lock:
            self._total_errors += 1

    def snapshot(self) -> tuple[int, dict[str, int], int]:
        with self._lock:
            return self._total_requests, dict(self._by_intent), self._total_errors
