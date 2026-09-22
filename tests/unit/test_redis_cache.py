"""Unit tests for the Redis adapter's degradation contract.

The adapter's entire value proposition is that a Redis outage makes the
service slower, never unavailable. That claim is only true if every method
swallows RedisError -- so that is what these tests pin.
"""

from redis.exceptions import ConnectionError as RedisConnectionError

from intent_service.adapters.redis_cache import RedisDecisionCache


class BrokenClient:
    """Every call fails, the way a client against a dead Redis does."""

    def get(self, key: str) -> str:
        raise RedisConnectionError("redis is down")

    def setex(self, key: str, ttl: int, value: str) -> None:
        raise RedisConnectionError("redis is down")

    def ping(self) -> bool:
        raise RedisConnectionError("redis is down")

    def close(self) -> None:
        raise RedisConnectionError("redis is down")


class RecordingClient:
    def __init__(self) -> None:
        self.store: dict[str, tuple[int, str]] = {}
        self.closed = False

    def get(self, key: str) -> str | None:
        entry = self.store.get(key)
        return entry[1] if entry else None

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = (ttl, value)

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


def cache(client: object) -> RedisDecisionCache:
    return RedisDecisionCache(client, ttl_seconds=60)  # type: ignore[arg-type]


def test_get_reports_a_miss_when_redis_is_down() -> None:
    assert cache(BrokenClient()).get("k") is None


def test_set_is_silent_when_redis_is_down() -> None:
    cache(BrokenClient()).set("k", "v")  # must not raise


def test_healthy_is_false_when_redis_is_down() -> None:
    assert cache(BrokenClient()).healthy() is False


def test_close_is_silent_when_redis_is_down() -> None:
    cache(BrokenClient()).close()  # must not raise


def test_values_round_trip_with_the_configured_ttl() -> None:
    client = RecordingClient()
    adapter = cache(client)
    adapter.set("k", "payload")
    assert adapter.get("k") == "payload"
    # An entry without a TTL is a leak: keys derived from customer messages
    # would be retained forever.
    assert client.store["k"][0] == 60


def test_healthy_and_close_on_a_working_client() -> None:
    client = RecordingClient()
    adapter = cache(client)
    assert adapter.healthy() is True
    adapter.close()
    assert client.closed is True


def test_connect_sets_socket_timeouts() -> None:
    """A hung Redis must not stall a prediction. Without both timeouts, an
    optional dependency becomes a total outage."""
    adapter = RedisDecisionCache.connect("redis://localhost:6379/0", 60, 0.25)
    kwargs = adapter._client.get_connection_kwargs()
    assert kwargs["socket_timeout"] == 0.25
    assert kwargs["socket_connect_timeout"] == 0.25
