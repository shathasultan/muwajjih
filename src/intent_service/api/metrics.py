"""A minimal, dependency-free counter registry.

Deliberately not Prometheus/OpenTelemetry -- pulling in an observability
stack for a capstone would be its own architectural decision with its own
operational weight (see DECISIONS.md). This gives /v1/metrics something real
to report while keeping the surface small enough to unit-test in one file.

Thread-safe because FastAPI runs sync endpoints in a worker thread pool:
concurrent requests genuinely do mutate these counters in parallel, and
`self._x += 1` is not atomic in CPython once a dict lookup is involved.
"""

import threading
from dataclasses import dataclass, field


@dataclass
class MetricsRegistry:
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _total_requests: int = 0
    _total_errors: int = 0
    _cache_hits: int = 0
    _by_action: dict[str, int] = field(default_factory=dict)

    def record_decision(self, action: str, *, cache_hit: bool) -> None:
        with self._lock:
            self._total_requests += 1
            self._by_action[action] = self._by_action.get(action, 0) + 1
            if cache_hit:
                self._cache_hits += 1

    def record_error(self) -> None:
        with self._lock:
            self._total_errors += 1

    def snapshot(self) -> tuple[int, dict[str, int], int, int]:
        """(total_requests, decisions_by_action, total_errors, cache_hits)."""
        with self._lock:
            return (
                self._total_requests,
                dict(self._by_action),
                self._total_errors,
                self._cache_hits,
            )
