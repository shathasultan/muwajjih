"""Integration tests: exercise every endpoint through FastAPI's TestClient
against the real, trained model artifact -- these need
`python -m train.generate_dataset && python -m train.train_model` to have
run first (the CI pipeline runs both before this suite)."""

import pytest
from fastapi.testclient import TestClient

from intent_service.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_model_loaded(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_lists_the_trained_labels(client: TestClient) -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"]
    assert set(body["labels"]) == {
        "complaint",
        "order_status",
        "praise",
        "price_inquiry",
        "return_refund",
        "support_request",
    }


def test_predict_returns_a_well_formed_prediction(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "وصلني المنتج تالف ومكسور من الصندوق"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "complaint"
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["low_confidence"], bool)
    assert body["model_version"]


def test_predict_rejects_empty_text_with_422(client: TestClient) -> None:
    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 422


def test_predict_rejects_missing_field_with_422(client: TestClient) -> None:
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_predict_rejects_oversized_text_with_422(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "a" * 2001})

    assert response.status_code == 422


def test_metrics_increments_after_a_prediction(client: TestClient) -> None:
    before = client.get("/metrics").json()["total_requests"]

    client.post("/predict", json={"text": "كم سعر الطابعة؟"})

    after = client.get("/metrics").json()
    assert after["total_requests"] == before + 1
    assert after["requests_by_intent"].get("price_inquiry", 0) >= 1


def test_metrics_counts_validation_errors_separately_from_predictions(client: TestClient) -> None:
    """A 422 is a client error caught before the model runs -- it must not
    be counted as a served prediction."""
    before = client.get("/metrics").json()["total_requests"]

    client.post("/predict", json={"text": ""})

    after = client.get("/metrics").json()["total_requests"]
    assert after == before
