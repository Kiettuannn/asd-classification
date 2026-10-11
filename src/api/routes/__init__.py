from .health import router as health_router
from .predict import router as predict_router
from .explain import router as explain_router

__all__ = ["health_router", "predict_router", "explain_router"]
