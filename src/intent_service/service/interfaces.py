"""Service-layer ports. The service depends on THESE, never on a concrete
ML framework.

A single Protocol is the seam that lets `IntentClassifier` be unit-tested
with a six-line fake instead of a real TF-IDF + SVM pipeline, and lets the
model family change (SVM -> transformer -> hosted API) by writing one new
adapter, with zero changes to the service or API layers.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class IntentModel(Protocol):
    """Anything that can score a raw message against the label set."""

    model_version: str
    labels: list[str]

    def predict(self, text: str) -> tuple[str, float]:
        """Return (predicted_label, confidence) for one message."""
        ...
