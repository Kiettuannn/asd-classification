"""Tests for API routes and schemas."""

from src.api.schemas import HealthResponse, PredictRequest, PredictResponse


def test_schema_validations():
    """Verify API schema defaults and fields."""
    health = HealthResponse()
    assert health.status == "ok"
    assert health.atlas == "cc200"
