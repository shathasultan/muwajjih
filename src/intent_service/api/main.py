"""FastAPI entrypoint -- the composition root for the HTTP service.

This is the ONE place concrete implementations get wired together (the model
adapter, the cache adapter, the settings, the service, the security
controls). Every layer below this file depends on abstractions only; this
file is where reality is injected.

NOTE: there is deliberately no module-level `app = create_app()`. Building
the app reads configuration, so a module-level instance would do real work at
import time -- importing this module would fail whenever the environment is
incomplete, which breaks test collection and makes the failure look like a
code bug rather than a missing variable. `create_app` is a factory, launched
with:

    uvicorn intent_service.api.main:create_app --factory

This is the same principle the model adapter follows: nothing expensive or
failable happens because someone imported a module.

Routes live under /v1 so a future breaking change can ship as /v2 while
existing clients keep working.
"""

import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Starlette's base class, not FastAPI's subclass. The router raises the BASE
# class for an unmatched route, so a handler registered against FastAPI's
# HTTPException alone would miss every 404 -- and 404s would be the one
# response shape that escaped the envelope.
from starlette.exceptions import HTTPException as StarletteHTTPException

from intent_service.adapters.redis_cache import InMemoryDecisionCache, RedisDecisionCache
from intent_service.adapters.sklearn_model import SklearnIntentModel
from intent_service.api.logging_config import configure_logging
from intent_service.api.metrics import MetricsRegistry
from intent_service.api.schemas import (
    Envelope,
    ErrorDetail,
    HealthData,
    Meta,
    MetricsData,
    PolicyData,
    ReadyData,
    TriageData,
    TriageRequest,
)
from intent_service.api.security import (
    SECURITY_HEADERS,
    SlidingWindowRateLimiter,
    caller_identity,
    require_api_key,
)
from intent_service.config import Settings
from intent_service.domain.entities import CustomerMessage
from intent_service.domain.policy import DEPARTMENT_BY_INTENT, URGENCY_TERMS, PolicyThresholds
from intent_service.service.interfaces import DecisionCache
from intent_service.service.triage import TriageService

logger = logging.getLogger(__name__)

API_PREFIX = "/v1"

# Probe endpoints: never rate-limited, never authenticated. An orchestrator
# polls these without credentials, and throttling a probe turns a healthy
# instance into a falsely-unhealthy one.
PROBE_PATHS = frozenset({f"{API_PREFIX}/health", f"{API_PREFIX}/ready"})

# Methods that may carry a request body, and therefore must declare its size.
BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})

# Text used to warm the model at startup. Its result is discarded -- the point
# is paying the first-call cost (lazy numpy/BLAS initialisation, page faults
# on the unpickled arrays) before any real user does.
WARMUP_TEXT = "رسالة تهيئة للتأكد من جاهزية النموذج"

TRACE_HEADER = "X-Trace-Id"


@dataclass
class AppState:
    """Everything a request handler needs, built once at startup."""

    service: TriageService | None
    settings: Settings
    metrics: MetricsRegistry
    rate_limiter: SlidingWindowRateLimiter
    cache: DecisionCache | None
    cache_backend: str
    # Stays False until the warm-up decision has actually returned. Setting it
    # any earlier would make /v1/ready lie: the instance would be sent traffic
    # while the first request still has to pay initialisation cost.
    ready: bool = field(default=False)


def _trace_id(request: Request) -> str:
    """The id for this request.

    Generated once by the middleware and stashed on `request.state`, so the
    success path, every exception handler and every log line all quote the
    same value. Falling back to a fresh uuid keeps this total: a handler that
    somehow runs before the middleware still gets an id rather than raising.
    """
    existing = getattr(request.state, "trace_id", None)
    return existing if isinstance(existing, str) else str(uuid.uuid4())


def _envelope(
    request: Request,
    *,
    data: Any = None,
    error: ErrorDetail | None = None,
    status_code: int = status.HTTP_200_OK,
) -> JSONResponse:
    """Build the one response shape this service ever returns.

    Used by the error handlers. The success path returns a typed `Envelope`
    directly so FastAPI still generates an accurate OpenAPI schema for it.
    """
    trace_id = _trace_id(request)
    body = Envelope[Any](data=data, error=error, meta=Meta(trace_id=trace_id))
    return JSONResponse(
        body.model_dump(mode="json"),
        status_code=status_code,
        headers={**SECURITY_HEADERS, TRACE_HEADER: trace_id},
    )


def build_cache(settings: Settings) -> tuple[DecisionCache, str]:
    """Pick a cache backend. Redis when configured, in-process otherwise.

    Returns the backend name alongside the adapter so /v1/ready can report
    which one is actually in use -- "is Redis configured" is exactly the kind
    of thing that is assumed true in production and turns out not to be.
    """
    if settings.redis_url:
        return (
            RedisDecisionCache.connect(
                settings.redis_url,
                ttl_seconds=settings.cache_ttl_seconds,
                timeout_seconds=settings.cache_timeout_seconds,
            ),
            "redis",
        )
    return InMemoryDecisionCache(), "memory"


def load_app_state(settings: Settings | None = None) -> AppState:
    """Build the real dependencies. Isolated from `lifespan` so tests can call
    it directly with a custom Settings instance."""
    settings = settings or Settings()
    cache, cache_backend = build_cache(settings)
    thresholds = PolicyThresholds(
        auto_route_floor=settings.auto_route_floor,
        reject_floor=settings.reject_floor,
    )
    try:
        model = SklearnIntentModel.load(settings.model_path)
        service: TriageService | None = TriageService(
            model=model, thresholds=thresholds, cache=cache
        )
        logger.info("model_loaded", extra={"event": "model_loaded"})
    except FileNotFoundError:
        # Fail visibly through /v1/ready rather than crash-looping with no
        # explanation. Unlike a missing API key (which is a misconfiguration
        # we refuse to start on), a missing artifact is recoverable by
        # mounting the right volume.
        logger.exception("model artifact missing", extra={"event": "model_missing"})
        service = None
    return AppState(
        service=service,
        settings=settings,
        metrics=MetricsRegistry(),
        rate_limiter=SlidingWindowRateLimiter(
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        ),
        cache=cache,
        cache_backend=cache_backend,
    )


def warm_up(state: AppState) -> None:
    """Run one throwaway decision, then mark the instance ready.

    Order matters: `ready` is set only after `triage` returns. Marking it
    before would defeat the purpose of the warm-up entirely.
    """
    if state.service is None:
        return
    state.service.triage(CustomerMessage(text=WARMUP_TEXT))
    state.ready = True
    logger.info("warmup complete", extra={"event": "warmup_complete"})


def shut_down(state: AppState) -> None:
    """Release resources on SIGTERM.

    `docker stop` sends SIGTERM, uvicorn turns it into the shutdown half of
    the lifespan, and this runs. Without it the container still exits -- but
    only after Redis times out the abandoned connections, which shows up as
    error noise on the Redis side of an otherwise clean deploy.
    """
    state.ready = False
    closer = getattr(state.cache, "close", None)
    if callable(closer):
        closer()
    logger.info("shutdown complete", extra={"event": "shutdown_complete"})


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    `settings` is injected by tests; production passes nothing and the
    configuration is read from the environment here -- at call time, never at
    import time.
    """
    resolved = settings or Settings()
    # Do this first: everything below may log, and an unconfigured logger
    # drops those lines silently.
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Closes over `resolved`, so the running app always uses exactly the
        # settings it was created with. Re-reading the environment here would
        # silently ignore injected test configuration.
        state = load_app_state(resolved)
        app.state.app_state = state
        warm_up(state)
        try:
            yield
        finally:
            shut_down(state)

    app = FastAPI(
        title=resolved.api_title,
        version="2.0.0",
        lifespan=lifespan,
        # Interactive docs enumerate every endpoint and schema. Harmless for a
        # course demo, worth switching off on a public deployment.
        docs_url="/docs" if resolved.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if resolved.enable_docs else None,
    )

    if resolved.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.cors_origin_list,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key"],
        )

    # -- error handlers: the envelope contract, enforced ---------------------
    # Without these three, FastAPI answers validation failures and unhandled
    # exceptions with its own `{"detail": ...}` shape, and a client would have
    # to parse two different formats depending on the status code -- losing
    # the trace id exactly when it is most needed.

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # `loc` is a tuple like ("body", "text"); joined, it names the offending
        # key so a client sees *which* field failed, not just that one did.
        fields = [".".join(str(part) for part in err["loc"]) for err in exc.errors()]
        logger.warning(
            "request validation failed",
            extra={"event": "validation_error", "trace_id": _trace_id(request)},
        )
        return _envelope(
            request,
            error=ErrorDetail(
                code="validation_error",
                message="the request body failed validation",
                fields=fields,
            ),
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        response = _envelope(
            request,
            error=ErrorDetail(
                code=_ERROR_CODES.get(exc.status_code, "error"), message=str(exc.detail)
            ),
            status_code=exc.status_code,
        )
        # Preserve WWW-Authenticate and friends that the raiser attached.
        for header, value in (exc.headers or {}).items():
            response.headers[header] = value
        return response

    @app.exception_handler(Exception)
    async def on_unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        # The caller never sees the traceback: internals stay in the logs,
        # correlated by the trace id that IS returned.
        #
        # `getattr` rather than a direct attribute read: this handler is the
        # last line of defence, and an app whose lifespan never ran has no
        # app_state. Raising HERE would replace a described failure with an
        # undescribed one, which is the worst possible moment to do it.
        state = getattr(request.app.state, "app_state", None)
        if state is not None:
            state.metrics.record_error()
        logger.exception(
            "unhandled error", extra={"event": "unhandled_error", "trace_id": _trace_id(request)}
        )
        return _envelope(
            request,
            error=ErrorDetail(code="internal_error", message="internal server error"),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    @app.middleware("http")
    async def harden(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        state: AppState = request.app.state.app_state
        # Assigned before anything can fail, so every downstream handler --
        # including the error handlers above -- has an id to report.
        request.state.trace_id = str(uuid.uuid4())

        # 1. Reject oversized bodies before reading them into memory.
        #    A Content-Length check alone is bypassable: a chunked request
        #    carries no such header, so it would skip the limit entirely. This
        #    API therefore *requires* Content-Length on any request with a
        #    body, and answers 411 otherwise -- which is what that code is for.
        if request.method in BODY_METHODS:
            declared = request.headers.get("content-length")
            if declared is None or not declared.isdigit():
                return _envelope(
                    request,
                    error=ErrorDetail(
                        code="length_required", message="Content-Length header is required"
                    ),
                    status_code=status.HTTP_411_LENGTH_REQUIRED,
                )
            if int(declared) > state.settings.max_request_bytes:
                return _envelope(
                    request,
                    error=ErrorDetail(code="payload_too_large", message="request body too large"),
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                )

        # 2. Rate-limit everything except the probes.
        if request.url.path not in PROBE_PATHS:
            caller = caller_identity(request, state.settings.trust_proxy_headers)
            allowed, retry_after = state.rate_limiter.check(caller)
            if not allowed:
                state.metrics.record_error()
                limited = _envelope(
                    request,
                    error=ErrorDetail(code="rate_limited", message="rate limit exceeded"),
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                )
                limited.headers["Retry-After"] = str(retry_after)
                return limited

        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value
        response.headers[TRACE_HEADER] = request.state.trace_id
        return response

    router = APIRouter(prefix=API_PREFIX)

    @router.get("/health", response_model=Envelope[HealthData])
    def health(request: Request) -> Envelope[HealthData]:
        """Liveness. Touches no state, performs no I/O, and never depends on
        the model -- so it answers instantly even while the model is loading or
        after it has failed. Restarting on a failed liveness probe is correct
        only when the process itself is broken."""
        return Envelope[HealthData](
            data=HealthData(status="ok"), meta=Meta(trace_id=_trace_id(request))
        )

    @router.get("/ready", response_model=Envelope[ReadyData])
    def ready(request: Request, response: Response) -> Envelope[ReadyData]:
        """Readiness. 503 until the model is loaded and warmed up, so a load
        balancer holds traffic back instead of handing a user the slow first
        request."""
        state: AppState = request.app.state.app_state
        if not state.ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return Envelope[ReadyData](
            data=ReadyData(
                ready=state.ready,
                model_loaded=state.service is not None,
                model_version=state.service.model.model_version if state.service else None,
                cache_backend=state.cache_backend,
                # Informational only -- see ReadyData's docstring.
                cache_healthy=state.cache.healthy() if state.cache else False,
            ),
            meta=Meta(trace_id=_trace_id(request)),
        )

    @router.get(
        "/policy", response_model=Envelope[PolicyData], dependencies=[Depends(require_api_key)]
    )
    def policy(request: Request) -> Envelope[PolicyData]:
        state: AppState = request.app.state.app_state
        if state.service is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
        return Envelope[PolicyData](
            data=PolicyData(
                model_version=state.service.model.model_version,
                labels=state.service.model.labels,
                auto_route_floor=state.service.thresholds.auto_route_floor,
                reject_floor=state.service.thresholds.reject_floor,
                departments=dict(DEPARTMENT_BY_INTENT),
                urgency_terms=sorted(URGENCY_TERMS),
            ),
            meta=Meta(trace_id=_trace_id(request)),
        )

    @router.get(
        "/metrics", response_model=Envelope[MetricsData], dependencies=[Depends(require_api_key)]
    )
    def metrics(request: Request) -> Envelope[MetricsData]:
        state: AppState = request.app.state.app_state
        total, by_action, errors, cache_hits = state.metrics.snapshot()
        return Envelope[MetricsData](
            data=MetricsData(
                total_requests=total,
                decisions_by_action=by_action,
                total_errors=errors,
                cache_hits=cache_hits,
            ),
            meta=Meta(trace_id=_trace_id(request)),
        )

    # NOTE: a plain `def`, deliberately not `async def`. Inference is
    # CPU-bound; declaring it async would run it on the event loop and block
    # every other request for its duration. As a sync def, FastAPI runs it in a
    # worker thread and the server stays responsive under load.
    @router.post(
        "/predict", response_model=Envelope[TriageData], dependencies=[Depends(require_api_key)]
    )
    def predict(request: Request, payload: TriageRequest) -> Envelope[TriageData]:
        state: AppState = request.app.state.app_state
        trace_id = _trace_id(request)

        if state.service is None:
            state.metrics.record_error()
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")

        # No try/except here: the `Exception` handler above owns the 500 path,
        # in one place, for every route. Catching it locally would duplicate
        # that logic and make the two drift.
        result = state.service.triage(CustomerMessage(text=payload.text))
        decision = result.decision

        state.metrics.record_decision(decision.action, cache_hit=result.cache_hit)
        # Note what is NOT in `extra`: the message text. See logging_config.
        logger.info(
            "decision made",
            extra={
                "event": "decision",
                "trace_id": trace_id,
                "action": decision.action,
                "intent": decision.intent,
                "priority": decision.priority,
                "confidence": round(decision.confidence, 4),
                "cached": result.cache_hit,
            },
        )
        return Envelope[TriageData](
            data=TriageData(
                action=decision.action,
                department=decision.department,
                priority=decision.priority,
                intent=decision.intent,
                confidence=decision.confidence,
                urgency_signals=decision.urgency_signals,
                reason=decision.reason,
                model_version=decision.model_version,
                cached=result.cache_hit,
            ),
            meta=Meta(trace_id=trace_id),
        )

    app.include_router(router)
    return app


# Stable machine-readable codes for the status codes this service raises.
# Clients branch on these strings, never on the prose in `message`.
_ERROR_CODES = {
    status.HTTP_401_UNAUTHORIZED: "unauthorized",
    status.HTTP_404_NOT_FOUND: "not_found",
    status.HTTP_411_LENGTH_REQUIRED: "length_required",
    status.HTTP_413_CONTENT_TOO_LARGE: "payload_too_large",
    status.HTTP_429_TOO_MANY_REQUESTS: "rate_limited",
    status.HTTP_500_INTERNAL_SERVER_ERROR: "internal_error",
    status.HTTP_503_SERVICE_UNAVAILABLE: "service_unavailable",
}
