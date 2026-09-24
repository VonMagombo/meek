from typing import Literal

from pydantic import BaseModel, Field

from app.core.config import get_settings

_MAX_TEXT_LENGTH = get_settings().max_text_length

Language = Literal["en", "sn"]


class ModerationRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=_MAX_TEXT_LENGTH, description="Comment text to score."
    )
    language: Language = Field(
        "en", description="'en' (English, unitary/toxic-bert) or 'sn' (Shona, fine-tuned)."
    )


class ModerationResult(BaseModel):
    text: str
    language: Language
    model: str
    scores: dict[str, float]
    flagged_labels: list[str]
    is_toxic: bool


class HealthResponse(BaseModel):
    status: str
    models: dict[str, str]
