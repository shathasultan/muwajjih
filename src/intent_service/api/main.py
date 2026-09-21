"""FastAPI entrypoint -- the composition root for the HTTP service.

This is the ONE place concrete implementations get wired together (the
model adapter, the settings, the classifier, the security controls). Every
layer below this file depends on abstractions only; this file is where
reality is injected.

NOTE: there is deliberately no module-level `app = create_app()`. Building
the app reads configuration, so a module-level instance would do real work
at import time -- importing this module would fail whenever the environment
is incomplete, which breaks test collection and makes the failure look like
a code bug rather than a missing variable. `create_app` is a factory,
launched with:

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

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from intent_service.adapters.sklearn_model import SklearnIntentModel
from intent_service.api.logging_config import configure_logging
from intent_service.api.metrics import MetricsRegistry
from intent_service.api.schemas import (
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)
from intent_service.api.security import (
    SECURITY_HEADERS,
    SlidingWindowRateLimiter,
    caller_identity,
    require_api_key,
)
from intent_service.config import Settings
from intent_service.domain.entities import CustomerMessage
from intent_service.service.classifier import IntentClassifier

logger = logging.getLogger(__name__)

API_PREFIX = "/v1"

# Probe endpoints: never rate-limited, never authenticated. An orchestrator
# polls these without credentials, and throttling a probe turns a healthy
# instance into a falsely-unhealthy one.
PROBE_PATHS = frozenset({f"{API_PREFIX}/health", f"{API_PREFIX}/ready"})

# Methods that may carry a request body, and therefore must declare its size.
BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})

# Text used to warm the model at startup. Its result is discarded -- the
# point is paying the first-call cost (lazy numpy/BLAS initialisation, page
# faults on the unpickled arrays) before any real user does.
WARMUP_TEXT = "رسالة تهيئة للتأكد من جاهزية النموذج"


@dataclass
class AppState:
    """Everything a request handler needs, built once at startup."""

    classifier: IntentClassifier | None
    settings: Settings
    metrics: MetricsRegistry
    rate_limiter: SlidingWindowRateLimiter
    # Stays False until the warm-up prediction has actually returned. Setting
    # it any earlier would make /v1/ready lie: the instance would be sent
    # traffic while the first request still has to pay initialisation cost.
    ready: bool = field(default=False)


def load_app_state(settings: Settings | None = None) -> AppState:
    """Build the real dependencies. Isolated from `lifespan` so tests can
    call it directly with a custom Settings instance."""
    settings = settings or Settings()
    metrics = MetricsRegistry()
    rate_limiter = SlidingWindowRateLimiter(
        max_requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    try:
        model = SklearnIntentModel.load(settings.model_path)
        classifier: IntentClassifier | None = IntentClassifier(model=model)
        logger.info("model_loaded version=%s", model.model_version)
    except FileNotFoundError:
        # Fail visibly through /v1/ready rather than crash-looping with no
        # explanation. Unlike a missing API key (which is a misconfiguration
        # we refuse to start on), a missing artifact is recoverable by
        # mounting the right volume.
        logger.exception("Model artifact missing; service will report not ready")
        classifier = None
    return AppState(
        classifier=classifier, settings=settings, metrics=metrics, rate_limiter=rate_limiter
    )


def warm_up(state: AppState) -> None:
    """Run one throwaway prediction, then mark the instance ready.

    Order matters: `ready` is set only after `classify` returns. Marking it
    before would defeat the purpose of the warm-up entirely.
    """
    if state.classifier is None:
        return
    state.classifier.classify(CustomerMessage(text=WARMUP_TEXT))
    state.ready = True
    logger.info("warmup_complete ready=true")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    `settings` is injected by tests; production passes nothing and the
    configuration is read from the environment here -- at call time, never
    at import time.
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
        yield

    app = FastAPI(
        title=resolved.api_title,
        version="1.0.0",
        lifespan=lifespan,
        # Interactive docs enumerate every endpoint and schema. Harmless for
        # a course demo, worth switching off on a public deployment.
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

    @app.middleware("http")
    async def harden(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        state: AppState = request.app.state.app_state

        # 1. Reject oversized bodies before reading them into memory.
        #    A Content-Length check alone is bypassable: a chunked request
        #    carries no such header, so it would skip the limit entirely.
        #    This API therefore *requires* Content-Length on any request with
        #    a body, and answers 411 otherwise -- which is exactly what that
        #    status code is for.
        if request.method in BODY_METHODS:
            declared = request.headers.get("content-length")
            if declared is None or not declared.isdigit():
                return JSONResponse(
                    {"detail": "Content-Length header is required"},
                    status_code=status.HTTP_411_LENGTH_REQUIRED,
                    headers=SECURITY_HEADERS,
                )
            if int(declared) > state.settings.max_request_bytes:
                return JSONResponse(
                    {"detail": "request body too large"},
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    headers=SECURITY_HEADERS,
                )

        # 2. Rate-limit everything except the probes.
        if request.url.path not in PROBE_PATHS:
            caller = caller_identity(request, state.settings.trust_proxy_headers)
            allowed, retry_after = state.rate_limiter.check(caller)
            if not allowed:
                state.metrics.record_error()
                return JSONResponse(
                    {"detail": "rate limit exceeded"},
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    headers={**SECURITY_HEADERS, "Retry-After": str(retry_after)},
                )

        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value
        return response

    router = APIRouter(prefix=API_PREFIX)

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness. Touches no state, performs no I/O, and never depends on
        the model -- so it answers instantly even while the model is loading
        or after it has failed. Restarting on a failed liveness probe is
        correct only when the process itself is broken."""
        return HealthResponse(status="ok")

    @router.get("/ready", response_model=ReadyResponse)
    def ready(request: Request, response: Response) -> ReadyResponse:
        """Readiness. 503 until the model is loaded and warmed up, so a load
        balancer holds traffic back instead of handing a user the slow first
        request."""
        state: AppState = request.app.state.app_state
        model_loaded = state.classifier is not None
        if not state.ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(
            ready=state.ready,
            model_loaded=model_loaded,
            model_version=state.classifier.model.model_version if state.classifier else None,
        )

    @router.get(
        "/model-info", response_model=ModelInfoResponse, dependencies=[Depends(require_api_key)]
    )
    def model_info(request: Request) -> ModelInfoResponse:
        state: AppState = request.app.state.app_state
        if state.classifier is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
        model = state.classifier.model
        return ModelInfoResponse(model_version=model.model_version, labels=model.labels)

    @router.get("/metrics", response_model=MetricsResponse, dependencies=[Depends(require_api_key)])
    def metrics(request: Request) -> MetricsResponse:
        state: AppState = request.app.state.app_state
        total, by_intent, errors = state.metrics.snapshot()
        return MetricsResponse(
            total_requests=total, requests_by_intent=by_intent, total_errors=errors
        )

    # NOTE: a plain `def`, deliberately not `async def`. Inference is
    # CPU-bound; declaring it async would run it on the event loop and block
    # every other request for its duration. As a sync def, FastAPI runs it in
    # a worker thread and the server stays responsive under load.
    @router.post(
        "/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)]
    )
    def predict(request: Request, payload: PredictRequest) -> PredictResponse:
        state: AppState = request.app.state.app_state
        trace_id = str(uuid.uuid4())

        if state.classifier is None:
            state.metrics.record_error()
            logger.error("predict_unavailable trace_id=%s", trace_id)
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")

        try:
            message = CustomerMessage(text=payload.text)
            prediction = state.classifier.classify(message)
        except Exception:
            # No bare `except: pass` -- log the failure with its trace_id,
            # report a generic 500, and keep the error counter honest. The
            # caller never sees the traceback: internals stay in the logs.
            state.metrics.record_error()
            logger.exception("predict_failed trace_id=%s", trace_id)
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, "prediction failed"
            ) from None

        state.metrics.record_prediction(prediction.intent)
        logger.info(
            "predict_ok trace_id=%s intent=%s confidence=%.4f",
            trace_id,
            prediction.intent,
            prediction.confidence,
        )
        return PredictResponse(
            intent=prediction.intent,
            confidence=prediction.confidence,
            low_confidence=prediction.confidence < state.settings.confidence_floor,
            model_version=prediction.model_version,
            trace_id=trace_id,
        )

    app.include_router(router)
    return app
