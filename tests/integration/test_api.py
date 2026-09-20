"""Integration tests: exercise every endpoint through FastAPI's TestClient
against the real, trained model artifact -- these need
`python -m train.generate_dataset && python -m train.train_model` to have
run first (the CI pipeline runs both before this suite)."""

import pytest
from fastapi.testclient import TestClient

from intent_service.api.main import create_app, load_app_state
from intent_service.config import Settings

TEST_KEY = "test-key-abc123"
AUTH = {"X-API-Key": TEST_KEY}

LABELS = {
    "complaint",
    "order_status",
    "praise",
    "price_inquiry",
    "return_refund",
    "support_request",
}


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"api_keys": TEST_KEY, "rate_limit_requests": 1000}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture
def client():
    settings = _settings()
    app = create_app(settings)
    with TestClient(app) as c:
        # The lifespan builds state from the environment; override it with
        # the test settings so the suite never depends on ambient env vars.
        c.app.state.app_state = load_app_state(settings)
        yield c


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_health_needs_no_credentials(client: TestClient) -> None:
    """Liveness probes carry no API key -- this must stay open."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_lists_the_trained_labels(client: TestClient) -> None:
    response = client.get("/model-info", headers=AUTH)

    assert response.status_code == 200
    assert set(response.json()["labels"]) == LABELS


def test_predict_returns_a_well_formed_prediction(client: TestClient) -> None:
    response = client.post(
        "/predict", json={"text": "وصلني المنتج تالف ومكسور من الصندوق"}, headers=AUTH
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "complaint"
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["low_confidence"], bool)
    assert body["model_version"]


@pytest.mark.parametrize(
    "payload",
    [{"text": ""}, {}, {"text": "a" * 2001}],
    ids=["empty", "missing-field", "too-long"],
)
def test_invalid_payloads_are_rejected_with_422(client: TestClient, payload: dict) -> None:
    assert client.post("/predict", json=payload, headers=AUTH).status_code == 422


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/predict", "/model-info", "/metrics"])
def test_protected_endpoints_reject_a_missing_key(client: TestClient, path: str) -> None:
    response = client.request("POST" if path == "/predict" else "GET", path, json={"text": "مرحبا"})

    assert response.status_code == 401


def test_protected_endpoint_rejects_a_wrong_key(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "مرحبا"}, headers={"X-API-Key": "not-the-key"})

    assert response.status_code == 401


def test_auth_failure_does_not_reveal_which_part_failed(client: TestClient) -> None:
    """Both failures must look identical to the caller."""
    missing = client.post("/predict", json={"text": "مرحبا"})
    wrong = client.post("/predict", json={"text": "مرحبا"}, headers={"X-API-Key": "nope"})

    assert missing.json()["detail"] == wrong.json()["detail"]


def test_auth_runs_before_the_model(client: TestClient) -> None:
    """An unauthenticated caller must not be able to spend model compute,
    even with a payload that would otherwise be valid."""
    before = client.get("/metrics", headers=AUTH).json()["total_requests"]

    client.post("/predict", json={"text": "وين طلبي"})

    after = client.get("/metrics", headers=AUTH).json()["total_requests"]
    assert after == before


# --------------------------------------------------------------------------
# Hardening
# --------------------------------------------------------------------------


def test_security_headers_are_present_on_every_response(client: TestClient) -> None:
    response = client.get("/health")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def test_security_headers_are_present_on_error_responses(client: TestClient) -> None:
    """Headers must not be lost on the error path."""
    response = client.post("/predict", json={"text": "مرحبا"})

    assert response.status_code == 401
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_oversized_body_is_rejected_with_413(client: TestClient) -> None:
    huge = "a" * 20_000

    response = client.post("/predict", json={"text": huge}, headers=AUTH)

    assert response.status_code == 413


def test_rate_limit_returns_429_with_retry_after() -> None:
    settings = _settings(rate_limit_requests=2, rate_limit_window_seconds=60)
    app = create_app(settings)
    with TestClient(app) as c:
        c.app.state.app_state = load_app_state(settings)

        c.get("/model-info", headers=AUTH)
        c.get("/model-info", headers=AUTH)
        blocked = c.get("/model-info", headers=AUTH)

    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1


def test_health_is_exempt_from_rate_limiting() -> None:
    """An orchestrator polling /health must never be throttled."""
    settings = _settings(rate_limit_requests=1, rate_limit_window_seconds=60)
    app = create_app(settings)
    with TestClient(app) as c:
        c.app.state.app_state = load_app_state(settings)

        codes = [c.get("/health").status_code for _ in range(5)]

    assert codes == [200] * 5


def test_cors_is_denied_by_default(client: TestClient) -> None:
    """With no configured origins, no cross-origin allowance is emitted."""
    response = client.get("/health", headers={"Origin": "https://evil.example"})

    assert "access-control-allow-origin" not in {k.lower() for k in response.headers}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_metrics_increments_after_a_prediction(client: TestClient) -> None:
    before = client.get("/metrics", headers=AUTH).json()["total_requests"]

    client.post("/predict", json={"text": "كم سعر الطابعة؟"}, headers=AUTH)

    after = client.get("/metrics", headers=AUTH).json()
    assert after["total_requests"] == before + 1
    assert after["requests_by_intent"].get("price_inquiry", 0) >= 1


def test_validation_errors_are_not_counted_as_predictions(client: TestClient) -> None:
    before = client.get("/metrics", headers=AUTH).json()["total_requests"]

    client.post("/predict", json={"text": ""}, headers=AUTH)

    assert client.get("/metrics", headers=AUTH).json()["total_requests"] == before


# --------------------------------------------------------------------------
# Degraded mode
# --------------------------------------------------------------------------


def test_service_reports_unhealthy_when_the_artifact_is_missing(tmp_path) -> None:
    """A missing model must degrade visibly, not crash the process."""
    settings = _settings(model_path=tmp_path / "absent.joblib")
    app = create_app(settings)
    with TestClient(app) as c:
        c.app.state.app_state = load_app_state(settings)

        health = c.get("/health")
        predict = c.post("/predict", json={"text": "مرحبا"}, headers=AUTH)

    assert health.status_code == 200
    assert health.json()["model_loaded"] is False
    assert predict.status_code == 503
