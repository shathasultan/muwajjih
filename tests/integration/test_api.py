"""Integration tests: every endpoint through FastAPI's TestClient against the
real, trained model artifact.

These need `python -m train.generate_dataset && python -m train.train_model`
to have run first (CI runs both before this suite, and `make train` does it
locally). Unlike the unit suite these exercise the wiring: middleware order,
dependency resolution, exception handlers, and the envelope contract.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from intent_service.api.main import API_PREFIX, create_app, load_app_state, warm_up
from intent_service.config import Settings

TEST_KEY = "test-key-abc123"
AUTH = {"X-API-Key": TEST_KEY}

HEALTH = f"{API_PREFIX}/health"
READY = f"{API_PREFIX}/ready"
PREDICT = f"{API_PREFIX}/predict"
POLICY = f"{API_PREFIX}/policy"
METRICS = f"{API_PREFIX}/metrics"

ACTIONS = {"auto_route", "human_review", "reject"}
DEPARTMENTS = {
    "quality_assurance",
    "sales",
    "technical_support",
    "customer_relations",
    "logistics",
    "returns",
}


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "api_keys": TEST_KEY,
        "rate_limit_requests": 1000,
        "redis_url": "",
        "log_level": "WARNING",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture
def client():
    app = create_app(_settings())
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# The envelope contract -- the property every other test leans on
# ---------------------------------------------------------------------------


def assert_envelope(body: dict) -> None:
    """`data` xor `error`, and `meta.trace_id` always present."""
    assert set(body) == {"data", "error", "meta"}
    assert "trace_id" in body["meta"] and body["meta"]["trace_id"]
    assert (body["data"] is None) != (body["error"] is None), "data xor error"


@pytest.mark.parametrize(
    ("method", "path", "headers", "payload", "expected_status"),
    [
        ("get", HEALTH, {}, None, 200),
        ("get", READY, {}, None, 200),
        ("get", POLICY, AUTH, None, 200),
        ("get", METRICS, AUTH, None, 200),
        ("post", PREDICT, AUTH, {"text": "الجهاز تالف"}, 200),
        ("post", PREDICT, {}, {"text": "الجهاز تالف"}, 401),  # no key
        ("post", PREDICT, AUTH, {"txt": "typo"}, 422),  # unknown field
        ("post", PREDICT, AUTH, {"text": ""}, 422),  # empty text
        ("get", f"{API_PREFIX}/nope", AUTH, None, 404),  # unknown route
    ],
)
def test_every_response_uses_the_same_envelope(
    client: TestClient, method, path, headers, payload, expected_status
) -> None:
    """Success AND failure. A client that parses one shape must never meet a
    second one -- which is exactly what FastAPI's default `{"detail": ...}`
    would do on the error paths above."""
    response = (
        client.get(path, headers=headers)
        if method == "get"
        else client.post(path, json=payload, headers=headers)
    )
    assert response.status_code == expected_status
    assert_envelope(response.json())


def test_trace_id_is_echoed_in_a_header_and_matches_the_body(client: TestClient) -> None:
    response = client.post(PREDICT, json={"text": "وين طلبي"}, headers=AUTH)
    assert response.headers["X-Trace-Id"] == response.json()["meta"]["trace_id"]


def test_trace_ids_are_unique_per_request(client: TestClient) -> None:
    seen = {
        client.post(PREDICT, json={"text": "وين طلبي"}, headers=AUTH).json()["meta"]["trace_id"]
        for _ in range(5)
    }
    assert len(seen) == 5


def test_validation_errors_name_the_offending_field(client: TestClient) -> None:
    """A typo in a client integration must be a loud, specific error -- not a
    silently ignored key."""
    error = client.post(PREDICT, json={"txt": "typo"}, headers=AUTH).json()["error"]
    assert error["code"] == "validation_error"
    assert any("txt" in field for field in error["fields"])


# ---------------------------------------------------------------------------
# Versioning, liveness and readiness
# ---------------------------------------------------------------------------


def test_routes_are_served_under_the_v1_prefix(client: TestClient) -> None:
    """A breaking change must be able to ship as /v2 without moving clients."""
    assert client.get(HEALTH).status_code == 200
    assert client.get("/health").status_code == 404


def test_health_is_liveness_only_and_needs_no_credentials(client: TestClient) -> None:
    assert client.get(HEALTH).json()["data"] == {"status": "ok"}


def test_ready_reports_a_loaded_warm_model(client: TestClient) -> None:
    data = client.get(READY).json()["data"]
    assert data["ready"] is True
    assert data["model_loaded"] is True
    assert data["model_version"]


def test_ready_is_503_before_warm_up_completes() -> None:
    """The distinction that makes readiness worth having: the process is
    alive (200 on /health) while it is still unable to serve (503 on /ready).
    Built without the lifespan so the warm-up genuinely has not run."""
    app = create_app(_settings())
    app.state.app_state = load_app_state(_settings())  # loaded, NOT warmed
    client = TestClient(app)

    assert client.get(HEALTH).status_code == 200
    ready = client.get(READY)
    assert ready.status_code == 503
    assert ready.json()["data"]["ready"] is False


def test_ready_reports_the_cache_backend(client: TestClient) -> None:
    data = client.get(READY).json()["data"]
    assert data["cache_backend"] == "memory"
    assert data["cache_healthy"] is True


def test_missing_artifact_degrades_to_503_instead_of_crashing(tmp_path: Path) -> None:
    """A missing model is recoverable by mounting the right volume, so the
    service reports it through /ready rather than crash-looping."""
    settings = _settings(model_path=tmp_path / "absent.joblib")
    app = create_app(settings)
    app.state.app_state = load_app_state(settings)
    client = TestClient(app)

    assert client.get(HEALTH).status_code == 200
    assert client.get(READY).json()["data"]["model_loaded"] is False
    response = client.post(PREDICT, json={"text": "مرحبا"}, headers=AUTH)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"


# ---------------------------------------------------------------------------
# The decision endpoint
# ---------------------------------------------------------------------------


def test_predict_returns_a_complete_decision(client: TestClient) -> None:
    data = client.post(PREDICT, json={"text": "وصلني الجهاز مكسور ومب شغال"}, headers=AUTH).json()[
        "data"
    ]

    assert data["action"] in ACTIONS
    assert data["department"] in DEPARTMENTS
    assert data["priority"] in {"normal", "urgent"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["reason"]
    assert data["model_version"]


def test_urgency_escalates_priority_end_to_end(client: TestClient) -> None:
    data = client.post(
        PREDICT, json={"text": "في تسرب غاز من السخان والرائحة قوية"}, headers=AUTH
    ).json()["data"]
    assert data["priority"] == "urgent"
    assert data["urgency_signals"]


def test_repeated_text_is_served_from_the_cache(client: TestClient) -> None:
    payload = {"text": "رسالة مكررة للتأكد من التخزين المؤقت"}
    assert client.post(PREDICT, json=payload, headers=AUTH).json()["data"]["cached"] is False
    assert client.post(PREDICT, json=payload, headers=AUTH).json()["data"]["cached"] is True


def test_text_longer_than_the_limit_is_rejected(client: TestClient) -> None:
    assert client.post(PREDICT, json={"text": "ا" * 2001}, headers=AUTH).status_code == 422


# ---------------------------------------------------------------------------
# Security controls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [PREDICT, POLICY, METRICS])
def test_business_endpoints_require_a_key(client: TestClient, path: str) -> None:
    response = client.get(path) if path != PREDICT else client.post(path, json={"text": "x"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_a_wrong_key_is_rejected(client: TestClient) -> None:
    response = client.post(PREDICT, json={"text": "x"}, headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_rate_limit_returns_429_with_retry_after() -> None:
    app = create_app(_settings(rate_limit_requests=2, rate_limit_window_seconds=60))
    with TestClient(app) as client:
        for _ in range(2):
            assert client.post(PREDICT, json={"text": "مرحبا"}, headers=AUTH).status_code == 200
        limited = client.post(PREDICT, json={"text": "مرحبا"}, headers=AUTH)

    assert limited.status_code == 429
    assert limited.json()["error"]["code"] == "rate_limited"
    assert int(limited.headers["Retry-After"]) >= 1


def test_probes_are_never_rate_limited() -> None:
    """Throttling a probe turns a healthy instance into a falsely-unhealthy
    one, which an orchestrator answers by restarting it."""
    app = create_app(_settings(rate_limit_requests=1, rate_limit_window_seconds=60))
    with TestClient(app) as client:
        for _ in range(10):
            assert client.get(HEALTH).status_code == 200
            assert client.get(READY).status_code == 200


def test_oversized_body_is_rejected_before_it_is_read(client: TestClient) -> None:
    response = client.post(PREDICT, json={"text": "ا" * 40_000}, headers=AUTH)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_body_without_content_length_is_rejected(client: TestClient) -> None:
    """A chunked request carries no Content-Length, so a size check alone
    would be bypassable."""
    response = client.post(
        PREDICT,
        content=iter([b'{"text":"hi"}']),
        headers={**AUTH, "Content-Type": "application/json"},
    )
    assert response.status_code == 411


def test_security_headers_are_present_on_every_response(client: TestClient) -> None:
    headers = client.get(HEALTH).headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Cache-Control"] == "no-store"


def test_docs_are_disabled_by_default(client: TestClient) -> None:
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


# ---------------------------------------------------------------------------
# Operator endpoints
# ---------------------------------------------------------------------------


def test_policy_endpoint_exposes_the_live_thresholds(client: TestClient) -> None:
    """Operators must be able to confirm what a running instance is actually
    using, rather than trusting that the deployment picked it up."""
    data = client.get(POLICY, headers=AUTH).json()["data"]
    assert data["auto_route_floor"] == 0.60
    assert data["reject_floor"] == 0.25
    assert set(data["departments"].values()) <= DEPARTMENTS
    assert "حريق" in data["urgency_terms"]


def test_metrics_count_real_decisions(client: TestClient) -> None:
    before = client.get(METRICS, headers=AUTH).json()["data"]["total_requests"]
    client.post(PREDICT, json={"text": "ابغى ارجع المنتج"}, headers=AUTH)
    after = client.get(METRICS, headers=AUTH).json()["data"]

    assert after["total_requests"] == before + 1
    assert sum(after["decisions_by_action"].values()) == after["total_requests"]


def test_warm_up_is_a_no_op_without_a_model(tmp_path: Path) -> None:
    state = load_app_state(_settings(model_path=tmp_path / "absent.joblib"))
    warm_up(state)
    assert state.ready is False
