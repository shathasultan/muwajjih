"""FastAPI entrypoint -- the composition root for the HTTP service.

This is the ONE place concrete implementations get wired together (the
model adapter, the settings, the classifier). Every layer below this file
depends on abstractions only; this file is where reality is injected --
same principle as `batch.py` in the offline pipeline, applied to a
long-lived server instead of a one-shot script.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, status

from intent_service.adapters.sklearn_model import SklearnIntentModel
from intent_service.api.metrics import MetricsRegistry
from intent_service.api.schemas import (
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
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


def load_app_state() -> AppState:
    """Build the real dependencies. Isolated from `lifespan` so tests can
    call it directly, or substitute a state with `classifier=None` to
    exercise the not-ready path without touching the filesystem."""
    settings = Settings()
    metrics = MetricsRegistry()
    try:
        model = SklearnIntentModel.load(settings.model_path)
        classifier: IntentClassifier | None = IntentClassifier(model=model)
        logger.info("Loaded model version %s", model.model_version)
    except FileNotFoundError:
        # Fail visibly through /health rather than crashing the process --
        # a container orchestrator can still start the container, see it
        # report unhealthy, and surface a clear signal instead of a
        # crash-loop with no explanation.
        logger.exception("Model artifact missing; service will report unhealthy")
        classifier = None
    return AppState(classifier=classifier, settings=settings, metrics=metrics)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.app_state = load_app_state()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title=Settings().api_title, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        state: AppState = app.state.app_state
        loaded = state.classifier is not None
        return HealthResponse(status="ok" if loaded else "unavailable", model_loaded=loaded)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        state: AppState = app.state.app_state
        if state.classifier is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
        model = state.classifier.model
        return ModelInfoResponse(model_version=model.model_version, labels=model.labels)

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        state: AppState = app.state.app_state
        total, by_intent, errors = state.metrics.snapshot()
        return MetricsResponse(
            total_requests=total, requests_by_intent=by_intent, total_errors=errors
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        state: AppState = app.state.app_state
        if state.classifier is None:
            state.metrics.record_error()
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")

        try:
            message = CustomerMessage(text=request.text)
            prediction = state.classifier.classify(message)
        except Exception:
            # No bare `except: pass` -- log the failure, report it as a
            # proper 500, and keep the error counter honest. Swallowing a
            # classification failure into a fake default intent is exactly
            # the "except Exception: pass" mistake this project is built to
            # avoid.
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
