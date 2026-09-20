"""Unit tests for the orchestration layer, using a test double.

None of these tests loads the real model artifact or imports sklearn --
that is the entire point of the `IntentModel` protocol. If this file
needed either, the seam would have failed.
"""

from typing import ClassVar

import pytest

from intent_service.domain.entities import CustomerMessage
from intent_service.service.classifier import IntentClassifier


class ConstantModel:
    """Test double satisfying the `IntentModel` protocol in a few lines."""

    def __init__(self, label: str, confidence: float, version: str = "test-v0") -> None:
        self.model_version = version
        self.labels = ["complaint", "praise"]
        self._label = label
        self._confidence = confidence

    def predict(self, text: str) -> tuple[str, float]:
        return self._label, self._confidence


class ExplodingModel:
    model_version = "broken-v0"
    labels: ClassVar[list[str]] = []

    def predict(self, text: str) -> tuple[str, float]:
        raise RuntimeError("model backend unreachable")


MESSAGE = CustomerMessage(text="test message")


def test_classifier_returns_model_label_and_confidence() -> None:
    classifier = IntentClassifier(model=ConstantModel("praise", 0.87))

    result = classifier.classify(MESSAGE)

    assert result.intent == "praise"
    assert result.confidence == 0.87


def test_model_version_is_carried_into_the_result() -> None:
    classifier = IntentClassifier(model=ConstantModel("complaint", 0.5, "custom-v9"))

    assert classifier.classify(MESSAGE).model_version == "custom-v9"


def test_model_failure_is_not_swallowed() -> None:
    """A broken model must raise, not fall back to a fake default intent --
    the API layer decides how to respond (503/500/alert)."""
    classifier = IntentClassifier(model=ExplodingModel())

    with pytest.raises(RuntimeError, match="model backend unreachable"):
        classifier.classify(MESSAGE)


def test_unknown_label_from_model_is_rejected_by_domain_validation() -> None:
    """If an adapter ever returns a label outside the domain's Intent set,
    pydantic must reject it at construction time rather than let a typo
    silently become a "valid" intent."""
    classifier = IntentClassifier(model=ConstantModel("not_a_real_intent", 0.9))

    with pytest.raises(Exception, match="not_a_real_intent"):
        classifier.classify(MESSAGE)
