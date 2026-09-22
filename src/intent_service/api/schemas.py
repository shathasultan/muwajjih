"""API-layer request/response contracts, including the unified envelope.

Deliberately separate from `domain.entities` and `domain.policy`: those are
the internal result shapes, these are the HTTP wire shapes. They happen to
carry similar fields today, but keeping them distinct means a future HTTP
concern (pagination, an envelope version, a deprecation header) never leaks
into the domain layer -- the API depends on the domain, never the reverse.

THE ENVELOPE
------------
Every response from this service -- success, validation failure, rate limit,
internal error -- has the same three top-level keys:

    {"data": <payload|null>, "error": <error|null>, "meta": {"trace_id": ...}}

A client therefore writes one parser, not two, and `trace_id` is always in
the same place. The usual alternative (200s return the payload at the top
level, errors return FastAPI's `{"detail": ...}`) forces every caller to
branch on the status code before it can even find the correlation id, which
is precisely when you most need it.
"""

from pydantic import BaseModel, ConfigDict, Field

from intent_service.domain.entities import Intent
from intent_service.domain.policy import Action, Department, Priority


class Meta(BaseModel):
    """Carried by every response, successful or not."""

    trace_id: str = Field(
        description="Unique id for this request, also written to the logs -- "
        "quote it in a bug report to find the exact request."
    )


class ErrorDetail(BaseModel):
    """The error half of the envelope.

    `code` is a stable machine-readable string that clients may branch on;
    `message` is for humans and may be reworded at any time. Keeping the two
    apart is what lets the wording improve without breaking integrations.
    """

    code: str
    message: str
    # Field-level validation failures, when the error came from the request
    # body. Empty for everything else.
    fields: list[str] = Field(default_factory=list)


class Envelope[PayloadT](BaseModel):
    """`data` xor `error` -- exactly one is populated, never both."""

    data: PayloadT | None = None
    error: ErrorDetail | None = None
    meta: Meta


class TriageRequest(BaseModel):
    # `extra="forbid"` turns an unknown or misspelled field into a 422 that
    # names the offending key, instead of silently ignoring it. A typo in a
    # client integration becomes a loud error at the boundary rather than a
    # request that quietly does the wrong thing.
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000, description="Customer message, in Arabic")


class TriageData(BaseModel):
    """The decision, as it appears on the wire.

    `action` is the field a caller integrates against; everything else is
    there so a human can audit why that action was chosen.
    """

    model_config = ConfigDict(protected_namespaces=())

    action: Action = Field(
        description="auto_route: act on it automatically. human_review: an "
        "operator confirms first. reject: return to sender for clarification."
    )
    department: Department
    priority: Priority
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    urgency_signals: list[str] = Field(
        description="Safety terms found in the message. Non-empty means the "
        "priority was escalated by policy, overriding model confidence."
    )
    reason: str = Field(description="Human-readable justification for `action`.")
    model_version: str
    cached: bool = Field(description="True when served from the decision cache.")


class HealthData(BaseModel):
    """Liveness only. Answering this must involve no I/O and no model access:
    a liveness probe asks 'is the process alive', not 'can it serve traffic'."""

    status: str


class ReadyData(BaseModel):
    """Readiness: is this instance able to serve prediction traffic?

    Reports 503 until the model is loaded AND the warm-up decision has
    completed, so a load balancer never routes the first (slow) request from
    a real user.

    `cache_healthy` is reported but deliberately NOT part of the readiness
    verdict: the cache is an optimisation, and failing readiness on a Redis
    outage would take down a service that is perfectly able to serve.
    """

    model_config = ConfigDict(protected_namespaces=())

    ready: bool
    model_loaded: bool
    model_version: str | None = None
    cache_backend: str
    cache_healthy: bool


class PolicyData(BaseModel):
    """The live decision policy, exposed so operators can confirm which
    thresholds a running instance is actually using -- rather than trusting
    that the deployment picked up the intended environment variables."""

    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    labels: list[str]
    auto_route_floor: float
    reject_floor: float
    departments: dict[Intent, Department]
    urgency_terms: list[str]


class MetricsData(BaseModel):
    total_requests: int
    decisions_by_action: dict[str, int]
    total_errors: int
    cache_hits: int
