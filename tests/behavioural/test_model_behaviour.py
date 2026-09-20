"""Behavioural tests: verify the trained model's actual quality, not just
that the code runs.

These are deliberately different from the unit/integration suites:

- test_entities.py / test_classifier.py never touch the real model.
- test_api.py exercises the real model, but only checks the API contract
  (status codes, response shape).
- THIS file is the only place that asks "is the model any good?" -- and it
  asks with sentences that do NOT appear verbatim in train/generate_dataset.py's
  templates, so a passing suite means genuine generalisation, not memorised
  training strings.

Marked `behavioural` (see pyproject.toml) so CI can run them as a distinct,
slower stage after the fast unit suite.
"""

import csv
from pathlib import Path

import pytest

from intent_service.adapters.sklearn_model import SklearnIntentModel

pytestmark = pytest.mark.behavioural

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "intent_model.joblib"
TEST_CSV = ROOT / "data" / "test.csv"

# Hand-written probes -- deliberately reworded, not copy-pasted from any
# template in train/generate_dataset.py, with product names and phrasing
# the generator never used. This is the generalisation check.
HELD_OUT_PROBES: list[tuple[str, str]] = [
    ("الجهاز اللي طلبته وصل مكسور تماماً ومب شغال خالص", "complaint"),
    ("ياليت تخبروني كم تكلفة الشاحن مع الشحن للدمام", "price_inquiry"),
    ("ما عرفت اربط الجهاز بالبلوتوث، أي خطوة أسوي أول؟", "support_request"),
    ("والله خدمتكم ممتازة والمنتج نظيف وبجودة عالية، تسلمون", "praise"),
    ("لسه ما وصلني الطلب مع إنه مكتوب إنه بيوصل من ثلاثة أيام", "order_status"),
    ("أبي أبدل المقاس لأن اللي وصلني صغير علي", "return_refund"),
]


@pytest.fixture(scope="module")
def model() -> SklearnIntentModel:
    if not MODEL_PATH.exists():
        pytest.skip(f"model artifact missing at {MODEL_PATH}; run `python -m train.train_model`")
    return SklearnIntentModel.load(MODEL_PATH)


def test_minimum_accuracy_on_held_out_test_split(model: SklearnIntentModel) -> None:
    """Regression gate: catches a training change that quietly tanks
    quality, without hardcoding the exact number train_model.py reported."""
    if not TEST_CSV.exists():
        pytest.skip(f"{TEST_CSV} missing; run `python -m train.generate_dataset`")

    with TEST_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    correct = sum(1 for row in rows if model.predict(row["text"])[0] == row["label"])
    accuracy = correct / len(rows)

    assert accuracy >= 0.90, (
        f"test-split accuracy dropped to {accuracy:.2%} ({correct}/{len(rows)})"
    )


def test_generalises_to_unseen_phrasing(model: SklearnIntentModel) -> None:
    """The real generalisation check: none of these sentences were emitted
    by any template in the training data."""
    misses = []
    for text, expected in HELD_OUT_PROBES:
        predicted, _ = model.predict(text)
        if predicted != expected:
            misses.append((text, expected, predicted))

    # Allow at most one miss out of six hand-written probes -- a hard 100%
    # bar on tiny hand-written examples is brittle; zero tolerance on a
    # majority miss is the actual regression signal.
    assert len(misses) <= 1, f"too many misses on held-out phrasing: {misses}"


def test_predictions_are_deterministic(model: SklearnIntentModel) -> None:
    for text, _ in HELD_OUT_PROBES:
        assert model.predict(text) == model.predict(text)


def test_confidence_is_always_bounded(model: SklearnIntentModel) -> None:
    """Fuzz-ish check over adversarial/degenerate inputs: the confidence
    score must stay in [0, 1] even for text nothing like the training
    distribution -- this is what SklearnIntentModel's softmax conversion
    exists to guarantee."""
    long_text = "أ" * 500
    edge_cases = ["", "a", "١٢٣٤٥", "😀😀😀", long_text, "text with English mixed in فقط"]
    for text in edge_cases:
        if not text:
            continue  # empty text is rejected at the domain layer, not the model
        _, confidence = model.predict(text)
        assert 0.0 <= confidence <= 1.0
