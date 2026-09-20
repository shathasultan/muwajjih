"""Use-case orchestration: classify one customer message.

Note what is ABSENT here: no FastAPI, no sklearn, no file paths, no
try/except. Pure orchestration = trivially testable (construct with a fake
IntentModel in one line, no mocking framework needed).

A broken model must surface as an exception here, not be swallowed --
the caller (the API layer) decides how to respond (503, alert, retry).
"""

from dataclasses import dataclass
from typing import cast

from intent_service.domain.entities import CustomerMessage, Intent, IntentPrediction
from intent_service.service.interfaces import IntentModel


@dataclass(frozen=True)
class IntentClassifier:
    """Constructor dependency injection: the classifier is handed a model,
    it never goes looking for one."""

    model: IntentModel

    def classify(self, message: CustomerMessage) -> IntentPrediction:
        label, confidence = self.model.predict(message.text)
        # The adapter guarantees `label` is one of `self.model.labels`, which
        # must equal the domain's Intent set (checked at load time in
        # SklearnIntentModel.load) -- pydantic still validates it for real
        # on the next line, this cast only satisfies the type checker.
        return IntentPrediction(
            intent=cast(Intent, label),
            confidence=confidence,
            model_version=self.model.model_version,
        )
