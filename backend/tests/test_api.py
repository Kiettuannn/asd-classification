"""Tests for API routes and schemas."""

from fastapi.testclient import TestClient

from asd_classification.api.app import create_app
from asd_classification.api.schemas import HealthResponse, PredictRequest, PredictResponse


def test_schema_validations():
    """Verify API schema defaults and fields."""
    health = HealthResponse()
    assert health.status == "ok"
    assert health.atlas == "cc200"


def test_api_routes_preserve_their_http_contracts():
    """The relocated app still exposes health, prediction, and explanation routes."""
    client = TestClient(create_app())

    health = client.get("/api/v1/health")
    prediction = client.post("/api/v1/predict", json={})
    explanation = client.get("/api/v1/explain/subject-1")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert prediction.status_code == 501
    assert explanation.status_code == 501
