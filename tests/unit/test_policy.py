"""Unit tests for the decision policy -- the pure core of the service.

No model, no HTTP, no cache: `decide()` is a total function of its arguments,
so every branch can be pinned exactly. These are the tests that matter most,
because a bug here is a wrong decision shipped to a customer, not a 500.
"""

import pytest
from pydantic import ValidationError

from intent_service.domain.policy import (
    DEPARTMENT_BY_INTENT,
    SAFETY_DEPARTMENT,
    PolicyThresholds,
    decide,
    detect_urgency,
    normalise,
)

T = PolicyThresholds(auto_route_floor=0.45, reject_floor=0.20)


def run(text: str = "رسالة عادية", intent: str = "complaint", confidence: float = 0.9):
    return decide(
        text=text,
        intent=intent,  # type: ignore[arg-type]
        confidence=confidence,
        model_version="test-v1",
        thresholds=T,
    )


# --- the three bands --------------------------------------------------------


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (1.00, "auto_route"),
        (0.46, "auto_route"),
        (0.45, "auto_route"),  # boundary: >= is inclusive
        (0.44, "human_review"),
        (0.20, "human_review"),  # boundary: >= is inclusive
        (0.19, "reject"),
        (0.00, "reject"),
    ],
)
def test_confidence_maps_to_the_documented_band(confidence: float, expected: str) -> None:
    """Both boundaries are tested from both sides. An off-by-one in a policy
    threshold is invisible in production until an auditor asks why a 0.45
    message went to a human."""
    assert run(confidence=confidence).action == expected


def test_every_intent_has_an_owning_department() -> None:
    """A missing entry here would be a KeyError at request time, on exactly
    the intent nobody tested."""
    for intent in DEPARTMENT_BY_INTENT:
        assert run(intent=intent).department == DEPARTMENT_BY_INTENT[intent]


def test_reason_is_populated_for_every_band() -> None:
    for confidence in (0.9, 0.3, 0.05):
        decision = run(confidence=confidence)
        assert decision.reason, "an unexplained decision is not auditable"
        assert f"{confidence:.3f}" in decision.reason


# --- urgency escalation -----------------------------------------------------


def test_urgency_term_forces_urgent_priority_and_auto_route() -> None:
    decision = run(text="في تسرب غاز في المطبخ", confidence=0.05)
    assert decision.priority == "urgent"
    # Escalated despite a confidence that would otherwise be rejected.
    assert decision.action == "auto_route"
    assert "غاز" in decision.urgency_signals


@pytest.mark.parametrize("intent", sorted(DEPARTMENT_BY_INTENT))
def test_escalation_overrides_the_department_for_every_intent(intent: str) -> None:
    """The bug this pins: escalating on urgency while still taking the
    DEPARTMENT from the model sends a misread message to the wrong team --
    urgently. A real gas-leak message scored 0.37 as `praise`, so it was
    correctly marked urgent and then routed to customer relations.

    A message the classifier does not understand is exactly where its opinion
    is worth least, so the override is total."""
    decision = run(text="في تسرب غاز في المطبخ", intent=intent, confidence=0.05)
    assert decision.department == SAFETY_DEPARTMENT
    assert decision.department != DEPARTMENT_BY_INTENT[intent] or intent is None


def test_the_safety_department_is_reachable_only_by_escalation() -> None:
    """It must not be something the model can route to on its own -- otherwise
    a classifier error could send ordinary traffic to the emergency team and
    bury a real one."""
    assert SAFETY_DEPARTMENT not in DEPARTMENT_BY_INTENT.values()
    for intent in DEPARTMENT_BY_INTENT:
        for confidence in (0.99, 0.5, 0.01):
            assert run(intent=intent, confidence=confidence).department != SAFETY_DEPARTMENT


def test_the_reason_names_the_department_it_overrode() -> None:
    """An operator seeing a `praise` message in the safety queue needs the
    decision to say why, or it reads as a bug."""
    decision = run(text="صار حريق", intent="praise", confidence=0.37)
    assert "safety" in decision.reason
    assert "customer_relations" in decision.reason


def test_urgency_signals_are_sorted_and_deduplicated() -> None:
    first = run(text="حريق و غاز")
    second = run(text="غاز و حريق")
    assert first.urgency_signals == second.urgency_signals == ["حريق", "غاز"]


def test_no_urgency_term_means_normal_priority() -> None:
    decision = run(text="ابغى اعرف سعر المنتج")
    assert decision.priority == "normal"
    assert decision.urgency_signals == []


# --- normalisation ----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "حريق",
        "حَرِيق",  # diacritics
        "حــريق",  # tatweel
        "حريق!!!",  # punctuation
        "  حريق  ",  # surrounding whitespace
    ],
)
def test_urgency_detection_survives_presentational_variation(text: str) -> None:
    """These five strings are the same word to a reader. If the escalation
    rule disagreed, a real emergency written with diacritics would be
    silently downgraded."""
    assert detect_urgency(text) == ["حريق"]


def test_normalise_does_not_strip_arabic_letters() -> None:
    assert "منتج" in normalise("المنتج، تالف!")


# --- configuration guardrails -----------------------------------------------


def test_inverted_thresholds_are_rejected_at_construction() -> None:
    """With reject_floor above auto_route_floor the human_review band is
    empty and the service would silently never ask for a human."""
    with pytest.raises(ValidationError, match="must not exceed"):
        PolicyThresholds(auto_route_floor=0.2, reject_floor=0.5)


def test_equal_thresholds_are_allowed() -> None:
    """A deliberate 'no human review' configuration is legitimate; only an
    inverted one is incoherent."""
    assert PolicyThresholds(auto_route_floor=0.5, reject_floor=0.5).reject_floor == 0.5
