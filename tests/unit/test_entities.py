"""Unit tests for domain entities -- validation at the boundary."""

import pytest
from pydantic import ValidationError

from intent_service.domain.entities import CustomerMessage, IntentPrediction


def test_message_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        CustomerMessage(text="")


def test_message_rejects_oversized_text() -> None:
    with pytest.raises(ValidationError):
        CustomerMessage(text="a" * 2001)


def test_message_is_immutable() -> None:
    message = CustomerMessage(text="hello")
    with pytest.raises(ValidationError):
        message.text = "changed"  # type: ignore[misc]


def test_prediction_rejects_unknown_intent() -> None:
    with pytest.raises(ValidationError):
        IntentPrediction(intent="not_a_real_intent", confidence=0.5, model_version="v1")  # type: ignore[arg-type]


def test_prediction_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValidationError):
        IntentPrediction(intent="praise", confidence=1.5, model_version="v1")
