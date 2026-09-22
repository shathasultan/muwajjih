"""Redis-backed decision cache -- the project's extension.

Why a cache is worth building here: triage traffic is extremely repetitive.
Support queues receive the same boilerplate sentences hundreds of times a
day ("وين طلبي؟"), and a TF-IDF + SVM inference on every duplicate is pure
waste. See BENCHMARKS.md for the measured effect.

Why it is a SEPARATE ADAPTER rather than a few redis calls inside the API
layer: the service layer depends on the `DecisionCache` Protocol, so the
entire feature can be switched off (cache=None), swapped for the in-memory
implementation in tests, or replaced with Memcached, without the service
knowing. This file is the only one in the project allowed to import redis.

DEGRADATION IS THE DESIGN. Every public method swallows backend errors and
reports a miss. A Redis outage makes the service slower, never unavailable
-- which is exactly the property that justifies adding an external
dependency to a request path at all.
"""

import logging

from redis import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class RedisDecisionCache:
    """Adapter satisfying the `DecisionCache` port."""

    def __init__(self, client: Redis, ttl_seconds: int) -> None:
        self._client = client
        self._ttl = ttl_seconds

    @classmethod
    def connect(cls, url: str, ttl_seconds: int, timeout_seconds: float) -> "RedisDecisionCache":
        """Build a client. Does NOT connect -- redis-py is lazy, and a
        connection attempt here would make startup fail on a cache outage.

        Both timeouts are set and deliberately short: an unbounded socket
        read would let a hung Redis stall every prediction request, turning
        the optional dependency into a total outage. This is the single most
        important line in the file.
        """
        client = Redis.from_url(
            url,
            socket_timeout=timeout_seconds,
            socket_connect_timeout=timeout_seconds,
            decode_responses=True,
        )
        return cls(client, ttl_seconds)

    def get(self, key: str) -> str | None:
        try:
            value = self._client.get(key)
        except RedisError:
            # debug, not error: a cache miss caused by an outage is expected
            # behaviour for this component, and logging it at error level on
            # every request would bury real failures.
            logger.debug("cache_get_failed key=%s", key, exc_info=True)
            return None
        return value if isinstance(value, str) else None

    def set(self, key: str, value: str) -> None:
        try:
            # Always with a TTL. An unbounded cache of customer-derived keys
            # grows without limit and retains message-derived data forever.
            self._client.setex(key, self._ttl, value)
        except RedisError:
            logger.debug("cache_set_failed key=%s", key, exc_info=True)

    def healthy(self) -> bool:
        try:
            return bool(self._client.ping())
        except RedisError:
            return False

    def close(self) -> None:
        """Release the connection pool on shutdown, so a stopped container
        does not leave sockets open on the Redis side."""
        try:
            self._client.close()
        except RedisError:  # pragma: no cover - defensive
            logger.debug("cache_close_failed", exc_info=True)


class InMemoryDecisionCache:
    """Process-local fallback satisfying the same port.

    Used when no Redis URL is configured, and in tests. Bounded by
    `max_entries` with simple insertion-order eviction: an unbounded dict
    keyed by user input is a memory-exhaustion vector, not a cache.
    """

    def __init__(self, max_entries: int = 1024) -> None:
        self._store: dict[str, str] = {}
        self._max_entries = max_entries

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        if key not in self._store and len(self._store) >= self._max_entries:
            # dicts preserve insertion order, so this evicts the oldest.
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[key] = value

    def healthy(self) -> bool:
        return True
