"""FastAPI entrypoint -- the composition root for the HTTP service.

This is the ONE place concrete implementations get wired together (the
model adapter, the settings, the classifier, the security controls). Every
layer below this file depends on abstractions only; this file is where
reality is injected.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from intent_service.adapters.sklearn_model import SklearnIntentModel
from intent_service.api.metrics import MetricsRegistry
from intent_service.api.schemas import (
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
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


@dataclass
class AppState:
    """Everything a request handler needs, built once at startup."""

    classifier: IntentClassifier | None
    settings: Settings
    metrics: MetricsRegistry
    rate_limiter: SlidingWindowRateLimiter


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
        logger.info("Loaded model version %s", model.model_version)
    except FileNotFoundError:
        # Fail visibly through /health rather than crash-looping with no
        # explanation. Unlike a missing API key (which is a misconfiguration
        # we refuse to start on), a missing artifact is recoverable by
        # mounting the right volume.
        logger.exception("Model artifact missing; service will report unhealthy")
        classifier = None
    return AppState(
        classifier=classifier, settings=settings, metrics=metrics, rate_limiter=rate_limiter
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.app_state = load_app_state()
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings()

    app = FastAPI(
        title=resolved.api_title,
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
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > state.settings.max_request_bytes:
            return JSONResponse(
                {"detail": "request body too large"},
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                headers=SECURITY_HEADERS,
            )

        # 2. Rate-limit everything except the liveness probe, so an
        #    orchestrator's polling can never exhaust a caller's budget.
        if request.url.path != "/health":
            allowed, retry_after = state.rate_limiter.check(caller_identity(request))
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

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Unauthenticated by design: liveness probes carry no credentials,
        and this exposes no business data."""
        state: AppState = request.app.state.app_state
        loaded = state.classifier is not None
        return HealthResponse(status="ok" if loaded else "unavailable", model_loaded=loaded)

    @app.get(
        "/model-info", response_model=ModelInfoResponse, dependencies=[Depends(require_api_key)]
    )
    def model_info(request: Request) -> ModelInfoResponse:
        state: AppState = request.app.state.app_state
        if state.classifier is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
        model = state.classifier.model
        return ModelInfoResponse(model_version=model.model_version, labels=model.labels)

    @app.get("/metrics", response_model=MetricsResponse, dependencies=[Depends(require_api_key)])
    def metrics(request: Request) -> MetricsResponse:
        state: AppState = request.app.state.app_state
        total, by_intent, errors = state.metrics.snapshot()
        return MetricsResponse(
            total_requests=total, requests_by_intent=by_intent, total_errors=errors
        )

    @app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
    def predict(request: Request, payload: PredictRequest) -> PredictResponse:
        state: AppState = request.app.state.app_state
        if state.classifier is None:
            state.metrics.record_error()
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")

        try:
            message = CustomerMessage(text=payload.text)
            prediction = state.classifier.classify(message)
        except Exception:
            # No bare `except: pass` -- log the failure, report it as a proper
            # 500, and keep the error counter honest. The caller never sees
            # the traceback: internal details stay in the logs.
            state.metrics.record_error()
            logger.exception("Prediction failed")
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, "prediction failed"
            ) from None

        state.metrics.record_prediction(prediction.intent)
        return PredictResponse(
            intent=prediction.intent,
            confidence=prediction.confidence,
            low_confidence=prediction.confidence < state.settings.confidence_floor,
            model_version=prediction.model_version,
        )

    return app


app = create_app()
