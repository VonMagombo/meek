import os
from functools import lru_cache

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "meek"
    toxicity_threshold: float = float(os.environ.get("MEEK_TOXICITY_THRESHOLD", 0.5))
    max_text_length: int = int(os.environ.get("MEEK_MAX_TEXT_LENGTH", 5000))
    cors_origins: list[str] = os.environ.get("MEEK_CORS_ORIGINS", "*").split(",")


@lru_cache
def get_settings() -> Settings:
    return Settings()
