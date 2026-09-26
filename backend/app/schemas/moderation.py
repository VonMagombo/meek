from typing import Literal

from pydantic import BaseModel, Field

from app.core.config import get_settings

_SETTINGS = get_settings()
_MAX_TEXT_LENGTH = _SETTINGS.max_text_length
_MAX_BATCH_SIZE = _SETTINGS.max_batch_size

Language = Literal["en", "sn", "auto"]


class ModerationRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=_MAX_TEXT_LENGTH, description="Comment text to score."
    )
    language: Language = Field(
        "en", description="'en' (English), 'sn' (Shona), or 'auto' (automatic detection)."
    )
    thresholds: dict[str, float] | None = Field(
        None, description="Optional per-label threshold overrides."
    )


class ModerationResult(BaseModel):
    text: str
    language: Language
    model: str
    scores: dict[str, float]
    flagged_labels: list[str]
    is_toxic: bool
    thresholds_used: dict[str, float] = Field(
        default_factory=dict, description="Thresholds applied to evaluate toxicity."
    )


class ModerationBatchRequest(BaseModel):
    texts: list[str] = Field(
        ..., min_length=1, max_length=_MAX_BATCH_SIZE, description="List of comments to score."
    )
    language: Language = Field(
        "en", description="'en' (English), 'sn' (Shona), or 'auto' (automatic detection)."
    )
    thresholds: dict[str, float] | None = Field(
        None, description="Optional per-label threshold overrides."
    )


class ModerationBatchResult(BaseModel):
    results: list[ModerationResult]
    total: int
    flagged_count: int


class HealthResponse(BaseModel):
    status: str
    models: dict[str, str]

