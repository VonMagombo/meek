from fastapi import APIRouter, HTTPException

from app.schemas.moderation import (
    HealthResponse,
    ModerationBatchRequest,
    ModerationBatchResult,
    ModerationRequest,
    ModerationResult,
)
from app.services import moderation_service

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", models=moderation_service.available_models())


@router.post("/moderate", response_model=ModerationResult)
def moderate(request: ModerationRequest) -> ModerationResult:
    try:
        return moderation_service.moderate(
            text=request.text,
            language=request.language,
            custom_thresholds=request.thresholds,
        )
    except moderation_service.ShonaModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/moderate/batch", response_model=ModerationBatchResult)
def moderate_batch(request: ModerationBatchRequest) -> ModerationBatchResult:
    try:
        return moderation_service.moderate_batch(
            texts=request.texts,
            language=request.language,
            custom_thresholds=request.thresholds,
        )
    except moderation_service.ShonaModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

