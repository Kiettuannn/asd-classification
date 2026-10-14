"""Pydantic schemas for the ASD Classification API."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check status and loaded model information."""
    status: str = "ok"
    version: str = "0.1.0"
    active_model: Optional[str] = None
    atlas: str = "cc200"
    num_rois: int = 200


class PredictRequest(BaseModel):
    """Prediction request with FC matrix or subject ID."""
    subject_id: Optional[str] = None
    fc_matrix: Optional[List[List[float]]] = Field(
        None, description="200x200 Functional Connectivity Pearson correlation matrix"
    )


class PredictResponse(BaseModel):
    """Prediction output containing diagnosis probabilities."""
    subject_id: Optional[str] = None
    diagnosis: str = Field(..., description="'ASD' or 'TD'")
    asd_probability: float = Field(..., ge=0.0, le=1.0)
    td_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class ConnectionWeight(BaseModel):
    """An individual salient brain connection."""
    roi_source: int
    roi_target: int
    weight: float


class ExplainResponse(BaseModel):
    """Explainability report with top contributing connections."""
    subject_id: Optional[str] = None
    top_connections: List[ConnectionWeight]
    model_name: str
