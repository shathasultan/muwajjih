"""API-layer request/response contracts.

Deliberately separate from `domain.entities`: the domain's `IntentPrediction`
is the internal result shape, while these are the HTTP wire shapes. They
happen to look similar today, but keeping them distinct means a future HTTP
concern (pagination, an API version field, a wrapping envelope) never
leaks into the domain layer -- this is the "schema leakage" mistake in
reverse: the API depends on the domain, the domain never depends on the API.
"""

from pydantic import BaseModel, ConfigDict, Field

from intent_service.domain.entities import Intent


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000, description="Customer message in Arabic")


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    low_confidence: bool = Field(
        description="True when confidence fell below the configured floor -- "
        "callers should consider routing this message to a human."
    )
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    labels: list[str]


class MetricsResponse(BaseModel):
    total_requests: int
    requests_by_intent: dict[str, int]
    total_errors: int
