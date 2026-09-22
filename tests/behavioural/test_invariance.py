"""Behavioural layer 1 -- INVARIANCE.

Claim under test: a change to a message that does not change its meaning must
not change the decision.

This is the property that separates a model from a lookup table. It is also
the one that breaks silently: a tokenizer change, a normalisation regression,
or an over-fit character n-gram can make "حريق" and "حَريق" route differently,
and nothing in a unit suite would notice.

Runs against the REAL trained artifact, not a fake.
"""

from pathlib import Path

import pytest

from intent_service.adapters.sklearn_model import SklearnIntentModel
from intent_service.domain.entities import CustomerMessage
from intent_service.domain.policy import PolicyThresholds
from intent_service.service.triage import TriageService

pytestmark = pytest.mark.behavioural

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "intent_model.joblib"
THRESHOLDS = PolicyThresholds(auto_route_floor=0.60, reject_floor=0.25)


@pytest.fixture(scope="module")
def service() -> TriageService:
    if not MODEL_PATH.exists():
        pytest.skip(f"model artifact missing at {MODEL_PATH}; run `make train`")
    return TriageService(
        model=SklearnIntentModel.load(MODEL_PATH), thresholds=THRESHOLDS, cache=None
    )


def decide(service: TriageService, text: str):
    return service.triage(CustomerMessage(text=text)).decision


BASE_CASES = [
    "وصلني الجهاز مكسور ومب شغال نهائيا",
    "كم سعر الشاحن الاصلي مع الشحن؟",
    "لسه ما وصلني الطلب من ثلاثة ايام",
    "في تسرب غاز من السخان في المطبخ",
]

# Each transform is meaning-preserving by construction.
TRANSFORMS = {
    "trailing_whitespace": lambda t: f"   {t}   ",
    "collapsed_spacing": lambda t: t.replace(" ", "  "),
    "extra_punctuation": lambda t: f"{t}!!!...",
    "trailing_newline": lambda t: f"{t}\n",
}


@pytest.mark.parametrize("text", BASE_CASES)
@pytest.mark.parametrize("name", list(TRANSFORMS))
def test_meaning_preserving_edits_do_not_change_the_decision(
    service: TriageService, text: str, name: str
) -> None:
    baseline = decide(service, text)
    perturbed = decide(service, TRANSFORMS[name](text))

    assert perturbed.action == baseline.action, f"{name} changed the action"
    assert perturbed.department == baseline.department, f"{name} changed the department"
    assert perturbed.priority == baseline.priority, f"{name} changed the priority"


@pytest.mark.parametrize(
    "variant",
    [
        "في تسرب غاز في المطبخ",
        "في تسرب غَاز في المطبخ",  # diacritic
        "في تسرب غـــاز في المطبخ",  # tatweel
        "في تسرب غاز في المطبخ!!!",  # punctuation
    ],
)
def test_urgency_escalation_is_invariant_to_arabic_orthography(
    service: TriageService, variant: str
) -> None:
    """A real emergency written with diacritics must not be silently
    downgraded. This is the highest-consequence invariance in the system."""
    assert decide(service, variant).priority == "urgent"


def test_the_decision_is_deterministic(service: TriageService) -> None:
    """Same input, same output, every time. A non-deterministic decision
    system cannot be audited after the fact."""
    text = "ابي ارجع المنتج لان المقاس غلط"
    decisions = {decide(service, text).model_dump_json() for _ in range(10)}
    assert len(decisions) == 1
