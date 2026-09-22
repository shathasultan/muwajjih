"""Behavioural layer 2 -- DIRECTIONAL.

Claim under test: a change to a message that adds risk must move the decision
in one direction only. Never the other way.

This is the safety contract of the whole service, and the reason the urgency
rule lives in the domain rather than in the model. A retrained model is free
to change its mind about which department a message belongs to; it is NOT
free to start ignoring a gas leak. A directional test is what makes that
distinction enforceable in CI.
"""

from pathlib import Path

import pytest

from intent_service.adapters.sklearn_model import SklearnIntentModel
from intent_service.domain.entities import CustomerMessage
from intent_service.domain.policy import URGENCY_TERMS, PolicyThresholds, decide
from intent_service.service.triage import TriageService

pytestmark = pytest.mark.behavioural

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "intent_model.joblib"
THRESHOLDS = PolicyThresholds(auto_route_floor=0.60, reject_floor=0.25)

# Ordered, so "did the priority go up or down" is a comparison, not a guess.
PRIORITY_RANK = {"normal": 0, "urgent": 1}
# Ordered by how much automation the service is willing to apply.
AUTOMATION_RANK = {"reject": 0, "human_review": 1, "auto_route": 2}


@pytest.fixture(scope="module")
def service() -> TriageService:
    if not MODEL_PATH.exists():
        pytest.skip(f"model artifact missing at {MODEL_PATH}; run `make train`")
    return TriageService(
        model=SklearnIntentModel.load(MODEL_PATH), thresholds=THRESHOLDS, cache=None
    )


def triage(service: TriageService, text: str):
    return service.triage(CustomerMessage(text=text)).decision


NEUTRAL_MESSAGES = [
    "الجهاز ما يشتغل عدل من يومين",
    "عندي مشكلة في السخان الجديد",
    "ابي استفسر عن حالة الطلب",
    "المنتج وصل بس في ملاحظة بسيطة عليه",
    "؟؟؟",
]


@pytest.mark.parametrize("base", NEUTRAL_MESSAGES)
@pytest.mark.parametrize("term", ["حريق", "تسرب غاز", "انفجار", "دخان"])
def test_adding_an_urgency_term_never_lowers_the_priority(
    service: TriageService, base: str, term: str
) -> None:
    """The core directional invariant: risk can only push urgency up."""
    before = triage(service, base)
    after = triage(service, f"{base} وفي {term}")

    assert PRIORITY_RANK[after.priority] >= PRIORITY_RANK[before.priority]
    assert after.priority == "urgent", "an explicit emergency term must escalate"


@pytest.mark.parametrize("base", NEUTRAL_MESSAGES)
def test_adding_an_urgency_term_never_causes_a_rejection(service: TriageService, base: str) -> None:
    """Escalation must not be undone by low model confidence. A message the
    classifier finds unintelligible but that mentions a fire is still a fire
    -- rejecting it back to the sender is the worst possible outcome."""
    after = triage(service, f"{base} وصار حريق في البيت")
    assert after.action == "auto_route"
    assert after.urgency_signals


@pytest.mark.parametrize("term", sorted(URGENCY_TERMS))
def test_every_configured_urgency_term_actually_escalates(
    service: TriageService, term: str
) -> None:
    """A term in the list that does not survive normalisation is a silent
    hole in the safety net. This test makes adding one to URGENCY_TERMS
    self-verifying."""
    assert triage(service, f"عندي مشكلة وفي {term} في المكان").priority == "urgent"


@pytest.mark.parametrize(
    ("lower", "higher"),
    [(0.10, 0.30), (0.30, 0.70), (0.10, 0.99), (0.59, 0.60), (0.24, 0.25)],
)
def test_higher_confidence_never_reduces_automation(lower: float, higher: float) -> None:
    """Monotonicity of the policy itself: being MORE sure about a message must
    never make the service less willing to act on it. Tested against `decide`
    directly, because the point is the policy's shape, not the model's."""
    kwargs = {
        "text": "رسالة محايدة",
        "intent": "complaint",
        "model_version": "test",
        "thresholds": THRESHOLDS,
    }
    low = decide(confidence=lower, **kwargs)  # type: ignore[arg-type]
    high = decide(confidence=higher, **kwargs)  # type: ignore[arg-type]

    assert AUTOMATION_RANK[high.action] >= AUTOMATION_RANK[low.action]
