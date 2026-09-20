"""Domain entities: the vocabulary of the intent-classification problem.

Rules for this file:
- imports from stdlib + pydantic ONLY
- no I/O, no framework types, no ML library types (no sklearn, no joblib)

Keeping this layer free of ML types means the business vocabulary (what a
"message" and a "classification" are) survives a future swap of model
family (SVM -> transformer -> hosted LLM) untouched.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Closed set of intents the service recognises. Adding a seventh intent is a
# one-line change here plus a retrain -- not a schema migration.
Intent = Literal[
    "complaint",
    "order_status",
    "praise",
    "price_inquiry",
    "return_refund",
    "support_request",
]


class CustomerMessage(BaseModel):
    """A single inbound message to be classified."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1, max_length=2000)


class IntentPrediction(BaseModel):
    """The result of classifying one message."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str
