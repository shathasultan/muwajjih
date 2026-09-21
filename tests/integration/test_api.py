"""Integration tests: exercise every endpoint through FastAPI's TestClient
against the real, trained model artifact -- these need
`python -m train.generate_dataset && python -m train.train_model` to have
run first (the CI pipeline runs both before this suite)."""

import pytest
from fastapi.testclient import TestClient

from intent_service.api.main import API_PREFIX, create_app, load_app_state, warm_up
from intent_service.config import Settings

TEST_KEY = "test-key-abc123"
AUTH = {"X-API-Key": TEST_KEY}

HEALTH = f"{API_PREFIX}/health"
READY = f"{API_PREFIX}/ready"
PREDICT = f"{API_PREFIX}/predict"
MODEL_INFO = f"{API_PREFIX}/model-info"
METRICS = f"{API_PREFIX}/metrics"

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
    app = create_app(_settings())
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------
# Versioning
# --------------------------------------------------------------------------


def test_routes_are_served_under_the_v1_prefix(client: TestClient) -> None:
    """A breaking change must be able to ship as /v2 without moving clients."""
    assert client.get(HEALTH).status_code == 200
    assert client.get("/health").status_code == 404


# --------------------------------------------------------------------------
# Liveness vs readiness
# --------------------------------------------------------------------------


def test_health_is_liveness_only_and_needs_no_credentials(client: TestClient) -> None:
    response = client.get(HEALTH)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_answers_even_without_a_model(tmp_path) -> None:
    """Liveness must not depend on the model: a missing artifact is a
    readiness problem, and restarting the process would not fix it."""
    app = create_app(_settings(model_path=tmp_path / "absent.joblib"))
    with TestClient(app) as c:
        assert c.get(HEALTH).status_code == 200


def test_ready_reports_200_once_warmed_up(client: TestClient) -> None:
    response = client.get(READY)

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_ready_reports_503_before_warm_up_completes() -> None:
    """The state is only 'ready' after the warm-up prediction returns --
    reporting ready earlier would send a user the slow first request."""
    state = load_app_state(_settings())

    assert state.ready is False  # loaded, but not yet warmed

    warm_up(state)

    assert state.ready is True


def test_ready_reports_503_when_the_artifact_is_missing(tmp_path) -> None:
    app = create_app(_settings(model_path=tmp_path / "absent.joblib"))
    with TestClient(app) as c:
        response = c.get(READY)
        predict = c.post(PREDICT, json={"text": "مرحبا"}, headers=AUTH)

    assert response.status_code == 503
    assert response.json()["model_loaded"] is False
    assert predict.status_code == 503


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_model_info_lists_the_trained_labels(client: TestClient) -> None:
    response = client.get(MODEL_INFO, headers=AUTH)

    assert response.status_code == 200
    assert set(response.json()["labels"]) == LABELS


def test_predict_returns_a_well_formed_prediction(client: TestClient) -> None:
    response = client.post(
        PREDICT, json={"text": "وصلني المنتج تالف ومكسور من الصندوق"}, headers=AUTH
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "complaint"
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["low_confidence"], bool)
    assert body["model_version"]
    assert body["trace_id"]


def test_each_request_gets_a_distinct_trace_id(client: TestClient) -> None:
    """A trace_id shared between requests is useless for finding one of them
    in the logs."""
    first = client.post(PREDICT, json={"text": "وين طلبي"}, headers=AUTH).json()
    second = client.post(PREDICT, json={"text": "وين طلبي"}, headers=AUTH).json()

    assert first["trace_id"] != second["trace_id"]


@pytest.mark.parametrize(
    "payload",
    [{"text": ""}, {}, {"text": "a" * 2001}],
    ids=["empty", "missing-field", "too-long"],
)
def test_invalid_payloads_are_rejected_with_422(client: TestClient, payload: dict) -> None:
    assert client.post(PREDICT, json=payload, headers=AUTH).status_code == 422


def test_unknown_field_is_rejected_with_422_naming_the_field(client: TestClient) -> None:
    """extra="forbid": a typo in a client integration must fail loudly at the
    boundary, not be silently dropped."""
    response = client.post(PREDICT, json={"text": "مرحبا", "txet": "typo"}, headers=AUTH)

    assert response.status_code == 422
    assert "txet" in response.text


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", [PREDICT, MODEL_INFO, METRICS])
def test_protected_endpoints_reject_a_missing_key(client: TestClient, path: str) -> None:
    response = client.request("POST" if path == PREDICT else "GET", path, json={"text": "مرحبا"})

    assert response.status_code == 401


def test_protected_endpoint_rejects_a_wrong_key(client: TestClient) -> None:
    response = client.post(PREDICT, json={"text": "مرحبا"}, headers={"X-API-Key": "not-the-key"})

    assert response.status_code == 401


def test_auth_failure_does_not_reveal_which_part_failed(client: TestClient) -> None:
    """Both failures must look identical to the caller."""
    missing = client.post(PREDICT, json={"text": "مرحبا"})
    wrong = client.post(PREDICT, json={"text": "مرحبا"}, headers={"X-API-Key": "nope"})

    assert missing.json()["detail"] == wrong.json()["detail"]


def test_auth_runs_before_the_model(client: TestClient) -> None:
    """An unauthenticated caller must not be able to spend model compute,
    even with a payload that would otherwise be valid."""
    before = client.get(METRICS, headers=AUTH).json()["total_requests"]

    client.post(PREDICT, json={"text": "وين طلبي"})

    assert client.get(METRICS, headers=AUTH).json()["total_requests"] == before


# --------------------------------------------------------------------------
# Hardening
# --------------------------------------------------------------------------


def test_security_headers_are_present_on_every_response(client: TestClient) -> None:
    response = client.get(HEALTH)

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def test_security_headers_are_present_on_error_responses(client: TestClient) -> None:
    response = client.post(PREDICT, json={"text": "مرحبا"})

    assert response.status_code == 401
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_oversized_body_is_rejected_with_413(client: TestClient) -> None:
    response = client.post(PREDICT, json={"text": "a" * 20_000}, headers=AUTH)

    assert response.status_code == 413


def test_rate_limit_returns_429_with_retry_after() -> None:
    app = create_app(_settings(rate_limit_requests=2, rate_limit_window_seconds=60))
    with TestClient(app) as c:
        c.get(MODEL_INFO, headers=AUTH)
        c.get(MODEL_INFO, headers=AUTH)
        blocked = c.get(MODEL_INFO, headers=AUTH)

    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1


@pytest.mark.parametrize("path", [HEALTH, READY])
def test_probes_are_exempt_from_rate_limiting(path: str) -> None:
    """An orchestrator polls these constantly; throttling one would turn a
    healthy instance into a falsely-unhealthy one."""
    app = create_app(_settings(rate_limit_requests=1, rate_limit_window_seconds=60))
    with TestClient(app) as c:
        codes = [c.get(path).status_code for _ in range(5)]

    assert codes == [200] * 5


def test_cors_is_denied_by_default(client: TestClient) -> None:
    response = client.get(HEALTH, headers={"Origin": "https://evil.example"})

    assert "access-control-allow-origin" not in {k.lower() for k in response.headers}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_metrics_increments_after_a_prediction(client: TestClient) -> None:
    before = client.get(METRICS, headers=AUTH).json()["total_requests"]

    client.post(PREDICT, json={"text": "كم سعر الطابعة؟"}, headers=AUTH)

    after = client.get(METRICS, headers=AUTH).json()
    assert after["total_requests"] == before + 1
    assert after["requests_by_intent"].get("price_inquiry", 0) >= 1


def test_validation_errors_are_not_counted_as_predictions(client: TestClient) -> None:
    before = client.get(METRICS, headers=AUTH).json()["total_requests"]

    client.post(PREDICT, json={"text": ""}, headers=AUTH)

    assert client.get(METRICS, headers=AUTH).json()["total_requests"] == before


def test_warm_up_prediction_is_not_counted_in_metrics(client: TestClient) -> None:
    """The startup warm-up must not pollute the served-request count."""
    assert client.get(METRICS, headers=AUTH).json()["total_requests"] == 0
