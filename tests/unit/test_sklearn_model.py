"""Unit tests for the sklearn adapter, using a tiny real pipeline trained
in-memory (no need for the full artifact on disk)."""

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from intent_service.adapters.sklearn_model import SklearnIntentModel

TEXTS = ["ممتاز جداً", "ممتاز جداً", "منتج رائع", "تالف ومكسور", "وصل مكسور", "سيء جداً"]
LABELS = ["praise", "praise", "praise", "complaint", "complaint", "complaint"]


def _fit_tiny_pipeline() -> Pipeline:
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))),
            # Mirrors the real pipeline's classifier family: the adapter calls
            # predict_proba, so a tiny stand-in that lacks it would pass a test
            # the production artifact fails. Which is exactly what happened
            # when this project moved off LinearSVC.
            ("clf", LogisticRegression(random_state=42)),
        ]
    )
    pipeline.fit(TEXTS, LABELS)
    return pipeline


@pytest.fixture
def artifact_path(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(
        {"pipeline": _fit_tiny_pipeline(), "version": "v9.9.9", "labels": ["complaint", "praise"]},
        path,
    )
    return path


def test_load_reads_pipeline_version_and_labels(artifact_path) -> None:
    model = SklearnIntentModel.load(artifact_path)

    assert model.model_version == "v9.9.9"
    assert model.labels == ["complaint", "praise"]


def test_load_reports_a_missing_artifact_clearly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        SklearnIntentModel.load(tmp_path / "nope.joblib")


def test_load_rejects_label_mismatch(tmp_path) -> None:
    """A stale `labels` list in the artifact (e.g. after adding an intent
    without retraining) must fail loudly at load time, not silently
    mis-map predictions at serving time."""
    path = tmp_path / "mismatched.joblib"
    stale_labels = ["complaint", "praise", "extra"]
    joblib.dump(
        {"pipeline": _fit_tiny_pipeline(), "version": "v1", "labels": stale_labels},
        path,
    )

    with pytest.raises(ValueError, match="label mismatch"):
        SklearnIntentModel.load(path)


def test_predict_returns_label_and_bounded_confidence(artifact_path) -> None:
    model = SklearnIntentModel.load(artifact_path)

    label, confidence = model.predict("منتج رائع فعلاً")

    assert label in {"complaint", "praise"}
    assert 0.0 <= confidence <= 1.0


def test_confidence_is_the_probability_of_the_returned_label(artifact_path) -> None:
    """The label and the confidence must describe the same class. Reading one
    from `predict` and the other from `predict_proba` would let them disagree
    -- handing the policy a number that belongs to a different label."""
    model = SklearnIntentModel.load(artifact_path)
    label, confidence = model.predict("وصل تالف ومكسور")

    probabilities = model._pipeline.predict_proba(["وصل تالف ومكسور"])[0]
    index = list(model._pipeline.classes_).index(label)
    assert confidence == pytest.approx(probabilities[index])
    assert confidence == pytest.approx(max(probabilities))


def test_predict_is_deterministic(artifact_path) -> None:
    model = SklearnIntentModel.load(artifact_path)

    first = model.predict("تجربة سيئة ووصل تالف")
    second = model.predict("تجربة سيئة ووصل تالف")

    assert first == second
