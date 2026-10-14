"""Health check endpoint."""

from typing import Any
from asd_classification.api.schemas import HealthResponse

try:
    from fastapi import APIRouter
    router = APIRouter(tags=["Health"])

    @router.get("/health", response_model=HealthResponse)
    async def get_health() -> HealthResponse:
        """Returns service status and active model configuration."""
        return HealthResponse(status="ok")
except ImportError:
    router = None  # type: ignore
