"""Behavioural layer 4 -- MODEL QUALITY on held-out phrasing.

The other behavioural files test the SYSTEM's properties. This one asks the
narrower question the other three deliberately avoid: is the model any good?

Every probe below is hand-written and does NOT appear verbatim in
train/generate_dataset.py's templates -- different product nouns, different
phrasing, different dialect register. A passing suite therefore means genuine
generalisation rather than memorised training strings, which is exactly the
failure a synthetic dataset invites.
"""

import csv
from pathlib import Path

import pytest

from intent_service.adapters.sklearn_model import SklearnIntentModel

pytestmark = pytest.mark.behavioural

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "intent_model.joblib"
TEST_CSV = ROOT / "data" / "test.csv"

HELD_OUT_PROBES: list[tuple[str, str]] = [
    ("الجهاز اللي طلبته وصل مكسور تماماً ومب شغال خالص", "complaint"),
    ("ياليت تخبروني كم تكلفة الشاحن مع الشحن للدمام", "price_inquiry"),
    ("ما عرفت اربط الجهاز بالبلوتوث، أي خطوة أسوي أول؟", "support_request"),
    ("والله خدمتكم ممتازة والمنتج نظيف وبجودة عالية، تسلمون", "praise"),
    ("لسه ما وصلني الطلب مع إنه مكتوب إنه بيوصل من ثلاثة أيام", "order_status"),
    ("أبي أبدل المقاس لأن اللي وصلني صغير علي", "return_refund"),
]

# Not 100%: a gate that demands perfection on held-out phrasing gets lowered
# the first time it fails for a good reason. 5/6 is a bar the team will
# actually defend.
MIN_PROBE_ACCURACY = 5 / 6
MIN_TEST_ACCURACY = 0.85


@pytest.fixture(scope="module")
def model() -> SklearnIntentModel:
    if not MODEL_PATH.exists():
        pytest.skip(f"model artifact missing at {MODEL_PATH}; run `make train`")
    return SklearnIntentModel.load(MODEL_PATH)


def test_model_generalises_to_unseen_phrasing(model: SklearnIntentModel) -> None:
    wrong = [
        (text, expected, model.predict(text)[0])
        for text, expected in HELD_OUT_PROBES
        if model.predict(text)[0] != expected
    ]
    accuracy = 1 - len(wrong) / len(HELD_OUT_PROBES)
    assert accuracy >= MIN_PROBE_ACCURACY, f"accuracy {accuracy:.2f}; misclassified: {wrong}"


def test_accuracy_on_the_held_out_split(model: SklearnIntentModel) -> None:
    if not TEST_CSV.exists():
        pytest.skip("data/test.csv missing; run `make train`")
    rows = list(csv.DictReader(TEST_CSV.open(encoding="utf-8")))
    correct = sum(model.predict(row["text"])[0] == row["label"] for row in rows)
    accuracy = correct / len(rows)
    assert accuracy >= MIN_TEST_ACCURACY, f"test accuracy {accuracy:.3f} < {MIN_TEST_ACCURACY}"


def test_confidence_is_a_bounded_number(model: SklearnIntentModel) -> None:
    """LinearSVC has no predict_proba; the adapter derives a pseudo-confidence
    from decision_function. If that maths broke, the policy's thresholds would
    be comparing against nonsense."""
    for text, _ in HELD_OUT_PROBES:
        _, confidence = model.predict(text)
        assert 0.0 <= confidence <= 1.0


def test_predicted_labels_stay_inside_the_declared_set(model: SklearnIntentModel) -> None:
    for text, _ in HELD_OUT_PROBES:
        assert model.predict(text)[0] in model.labels
