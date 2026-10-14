"""Inference prediction endpoint."""

from asd_classification.api.schemas import PredictRequest, PredictResponse

try:
    from fastapi import APIRouter, HTTPException
    router = APIRouter(tags=["Inference"])

    @router.post("/predict", response_model=PredictResponse)
    async def predict_asd(request: PredictRequest) -> PredictResponse:
        """Classifies subject connectivity into ASD vs TD."""
        raise HTTPException(
            status_code=501, detail="Inference implementation will be added in implementation phase."
        )
except ImportError:
    router = None  # type: ignore
