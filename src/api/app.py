"""FastAPI application factory and middleware configuration."""

from typing import Any
from src.config.settings import get_settings

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from src.api.routes import health_router, predict_router, explain_router

    def create_app() -> FastAPI:
        """Constructs and configures the FastAPI application."""
        settings = get_settings()
        app = FastAPI(
            title=settings.api_title,
            version=settings.api_version,
            description="ASD Neuroimaging Classification API using Functional Connectivity and GNNs",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.include_router(health_router, prefix="/api/v1")
        app.include_router(predict_router, prefix="/api/v1")
        app.include_router(explain_router, prefix="/api/v1")

        return app

except ImportError:
    def create_app() -> Any:
        raise ImportError(
            "FastAPI is not installed. Install optional dependencies: pip install '.[api]'"
        )
