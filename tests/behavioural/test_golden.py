"""Behavioural layer 3 -- GOLDEN REFERENCE.

Claim under test: the decisions this service made yesterday are the decisions
it makes today, unless somebody deliberately changed them.

Invariance and directional tests pin PROPERTIES. This file pins ACTUAL
OUTPUTS for a fixed corpus, which is the only thing that catches a silent
drift no property happens to cover -- a retrain that quietly moves a whole
intent into a different department, say.

HOW TO HANDLE A FAILURE HERE
----------------------------
A red test in this file is a BEHAVIOUR CHANGE, not a broken test. Read the
diff it prints and decide whether the new behaviour is better. If it is not,
fix the code. If it is, run:

    python -m scripts.regenerate_golden --reviewed-by "Name" --reason "..."

and commit the resulting diff for review. Regenerating the file to make the
suite green is the single fastest way to end up with a test suite that
asserts nothing -- which is why the generator refuses to run without a named
reviewer and a stated reason.
"""

import json
from pathlib import Path

import pytest

from intent_service.adapters.sklearn_model import SklearnIntentModel
from intent_service.domain.entities import CustomerMessage
from intent_service.domain.policy import PolicyThresholds
from intent_service.service.triage import TriageService

pytestmark = pytest.mark.behavioural

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "intent_model.joblib"
GOLDEN_PATH = Path(__file__).parent / "golden_decisions.json"
CASES_PATH = Path(__file__).parent / "golden_cases.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.skip("golden file missing; run `python -m scripts.regenerate_golden`")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def service(golden: dict) -> TriageService:
    if not MODEL_PATH.exists():
        pytest.skip(f"model artifact missing at {MODEL_PATH}; run `make train`")
    # Thresholds come FROM the golden file, not from production config: a
    # recorded "human_review" means nothing unless the band that produced it
    # is known. This also makes a threshold change fail loudly here.
    return TriageService(
        model=SklearnIntentModel.load(MODEL_PATH),
        thresholds=PolicyThresholds(**golden["thresholds"]),
        cache=None,
    )


def test_the_golden_file_covers_every_declared_case(golden: dict) -> None:
    """Stops the corpus from silently shrinking: dropping a case from the
    golden file would make this suite pass by testing less."""
    cases = set(json.loads(CASES_PATH.read_text(encoding="utf-8")))
    assert set(golden["decisions"]) == cases


def test_the_golden_file_records_who_approved_it(golden: dict) -> None:
    """Provenance is the control that makes an arbitrary regeneration visible
    in review: `git blame` on this file must name a person and a reason."""
    assert golden.get("reviewed_by"), "golden file has no named reviewer"
    assert golden.get("reason"), "golden file has no stated reason"


def _cases() -> list[str]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("text", _cases())
def test_decision_matches_the_approved_reference(
    service: TriageService, golden: dict, text: str
) -> None:
    expected = golden["decisions"][text]
    decision = service.triage(CustomerMessage(text=text)).decision
    actual = {
        "action": decision.action,
        "department": decision.department,
        "priority": decision.priority,
        "intent": decision.intent,
        "urgency_signals": decision.urgency_signals,
    }
    assert actual == expected, (
        f"\nBEHAVIOUR CHANGE for {text!r}\n"
        f"  approved: {expected}\n"
        f"  current:  {actual}\n"
        "This is a behaviour change, not a broken test. See this file's "
        "docstring before regenerating."
    )
