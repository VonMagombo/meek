import os
from functools import lru_cache

from pydantic import BaseModel


LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

DEFAULT_THRESHOLDS: dict[str, float] = {
    "threat": 0.35,
    "severe_toxic": 0.35,
    "identity_hate": 0.40,
    "toxic": 0.50,
    "insult": 0.50,
    "obscene": 0.60,
}


class Settings(BaseModel):
    app_name: str = "meek"
    toxicity_threshold: float = float(os.environ.get("MEEK_TOXICITY_THRESHOLD", 0.5))
    max_text_length: int = int(os.environ.get("MEEK_MAX_TEXT_LENGTH", 5000))
    max_batch_size: int = int(os.environ.get("MEEK_MAX_BATCH_SIZE", 100))
    cors_origins: list[str] = os.environ.get("MEEK_CORS_ORIGINS", "*").split(",")

    def get_thresholds(self, overrides: dict[str, float] | None = None) -> dict[str, float]:
        # If global MEEK_TOXICITY_THRESHOLD is explicitly set in env, use it as fallback base
        global_override = os.environ.get("MEEK_TOXICITY_THRESHOLD")
        base = (
            {label: float(global_override) for label in LABELS}
            if global_override is not None
            else dict(DEFAULT_THRESHOLDS)
        )

        # Apply any per-category env vars (e.g. MEEK_THRESHOLD_THREAT)
        for label in LABELS:
            env_val = os.environ.get(f"MEEK_THRESHOLD_{label.upper()}")
            if env_val is not None:
                base[label] = float(env_val)

        # Apply any per-request overrides
        if overrides:
            for label, val in overrides.items():
                if label in LABELS and val is not None:
                    base[label] = float(val)

        return base


@lru_cache
def get_settings() -> Settings:
    return Settings()

