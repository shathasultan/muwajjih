"""Unit tests for the service layer: orchestration and caching.

Uses `FakeIntentModel` throughout -- these tests never load an artifact, so
they run in milliseconds and fail only when the orchestration is wrong.
"""

from intent_service.adapters.redis_cache import InMemoryDecisionCache
from intent_service.domain.entities import CustomerMessage
from intent_service.domain.policy import PolicyThresholds
from intent_service.service.interfaces import DecisionCache, IntentModel
from intent_service.service.triage import TriageService
from tests.conftest import FakeIntentModel

T = PolicyThresholds(auto_route_floor=0.45, reject_floor=0.20)


def test_fake_satisfies_the_port() -> None:
    """Guards the seam itself: if `IntentModel` grows a method, this fails
    here rather than as a confusing AttributeError deep in a later test."""
    assert isinstance(FakeIntentModel(), IntentModel)
    assert isinstance(InMemoryDecisionCache(), DecisionCache)


def test_model_output_flows_into_the_decision() -> None:
    service = TriageService(model=FakeIntentModel("price_inquiry", 0.8), thresholds=T)
    decision = service.triage(CustomerMessage(text="كم السعر")).decision
    assert (decision.intent, decision.action, decision.department) == (
        "price_inquiry",
        "auto_route",
        "sales",
    )
    assert decision.model_version == "test-v1"


def test_second_identical_call_is_served_from_cache() -> None:
    model = FakeIntentModel()
    service = TriageService(model=model, thresholds=T, cache=InMemoryDecisionCache())
    message = CustomerMessage(text="نفس الرسالة")

    first = service.triage(message)
    second = service.triage(message)

    assert (first.cache_hit, second.cache_hit) == (False, True)
    # The real assertion: the model was invoked once, not twice.
    assert model.calls == 1
    assert second.decision == first.decision


def test_cache_is_bypassed_when_not_configured() -> None:
    model = FakeIntentModel()
    service = TriageService(model=model, thresholds=T, cache=None)
    service.triage(CustomerMessage(text="نفس الرسالة"))
    service.triage(CustomerMessage(text="نفس الرسالة"))
    assert model.calls == 2


def test_cache_key_changes_with_model_version() -> None:
    """A retrain must invalidate the cache. Without this, a deploy would keep
    serving decisions made by the previous model for the whole TTL."""
    old = TriageService(model=FakeIntentModel(), thresholds=T)
    new_model = FakeIntentModel()
    new_model.model_version = "test-v2"
    new = TriageService(model=new_model, thresholds=T)
    assert old.cache_key("نص") != new.cache_key("نص")


def test_cache_key_changes_with_thresholds() -> None:
    """Same reasoning for a policy change: the stored decision was made under
    rules that no longer apply."""
    a = TriageService(model=FakeIntentModel(), thresholds=T)
    b = TriageService(
        model=FakeIntentModel(),
        thresholds=PolicyThresholds(auto_route_floor=0.7, reject_floor=0.2),
    )
    assert a.cache_key("نص") != b.cache_key("نص")


def test_cache_key_never_contains_the_raw_message() -> None:
    """Customer text must not leak into backend monitoring tools via key
    names."""
    secret = "رقم بطاقتي 4111111111111111"
    assert secret not in TriageService(model=FakeIntentModel(), thresholds=T).cache_key(secret)


class CorruptCache:
    """A cache holding a payload written by an older, incompatible version."""

    def get(self, key: str) -> str | None:
        return '{"not":"a decision"}'

    def set(self, key: str, value: str) -> None:
        pass

    def healthy(self) -> bool:
        return True


def test_unparseable_cached_payload_falls_through_to_the_model() -> None:
    """A stale schema in the cache must degrade to a recomputation, not a
    500. This is the failure mode of every cache that outlives a deploy."""
    model = FakeIntentModel()
    service = TriageService(model=model, thresholds=T, cache=CorruptCache())
    result = service.triage(CustomerMessage(text="رسالة"))
    assert result.cache_hit is False
    assert model.calls == 1


def test_in_memory_cache_evicts_instead_of_growing_without_bound() -> None:
    cache = InMemoryDecisionCache(max_entries=2)
    cache.set("a", "1")
    cache.set("b", "2")
    cache.set("c", "3")
    assert cache.get("a") is None  # oldest evicted
    assert (cache.get("b"), cache.get("c")) == ("2", "3")
