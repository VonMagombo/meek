from fastapi import APIRouter, HTTPException

from app.schemas.moderation import HealthResponse, ModerationRequest, ModerationResult
from app.services import moderation_service

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", models=moderation_service.available_models())


@router.post("/moderate", response_model=ModerationResult)
def moderate(request: ModerationRequest) -> ModerationResult:
    try:
        return moderation_service.moderate(request.text, request.language)
    except moderation_service.ShonaModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
