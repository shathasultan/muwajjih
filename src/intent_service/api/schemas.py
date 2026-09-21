"""API-layer request/response contracts.

Deliberately separate from `domain.entities`: the domain's `IntentPrediction`
is the internal result shape, while these are the HTTP wire shapes. They
happen to look similar today, but keeping them distinct means a future HTTP
concern (pagination, an API version field, a wrapping envelope) never
leaks into the domain layer -- the API depends on the domain, the domain
never depends on the API.
"""

from pydantic import BaseModel, ConfigDict, Field

from intent_service.domain.entities import Intent


class PredictRequest(BaseModel):
    # `extra="forbid"` turns an unknown or misspelled field into a 422 that
    # names the offending key, instead of silently ignoring it. A typo in a
    # client integration becomes a loud error at the boundary rather than a
    # request that quietly does the wrong thing.
    model_config = ConfigDict(extra="forbid")

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
    trace_id: str = Field(
        description="Unique id for this request, also written to the logs -- "
        "quote it in a bug report to find the exact request."
    )


class HealthResponse(BaseModel):
    """Liveness only. Answering this must involve no I/O and no model access:
    a liveness probe asks 'is the process alive', not 'can it serve traffic'."""

    status: str


class ReadyResponse(BaseModel):
    """Readiness: is this instance able to serve prediction traffic?

    Reports 503 until the model is loaded AND the warm-up prediction has
    completed, so a load balancer never routes the first (slow) request from
    a real user.
    """

    model_config = ConfigDict(protected_namespaces=())

    ready: bool
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    labels: list[str]


class MetricsResponse(BaseModel):
    total_requests: int
    requests_by_intent: dict[str, int]
    total_errors: int
