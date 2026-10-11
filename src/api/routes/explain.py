"""Model explainability & salient brain connections endpoint."""

from src.api.schemas import ExplainResponse

try:
    from fastapi import APIRouter, HTTPException
    router = APIRouter(tags=["Explainability"])

    @router.get("/explain/{subject_id}", response_model=ExplainResponse)
    async def explain_subject(subject_id: str, top_k: int = 15) -> ExplainResponse:
        """Returns top influential brain connections or GAT attention weights."""
        raise HTTPException(
            status_code=501, detail="Explainability implementation will be added in implementation phase."
        )
except ImportError:
    router = None  # type: ignore
