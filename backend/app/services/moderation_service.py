from ml.inference import shona_toxicity_classifier
from ml.inference import toxicity_classifier as en_classifier

from app.core.config import get_settings
from app.schemas.moderation import ModerationResult

ShonaModelUnavailableError = shona_toxicity_classifier.ModelNotFineTunedError


def moderate(text: str, language: str = "en") -> ModerationResult:
    settings = get_settings()
    if language == "sn":
        result = shona_toxicity_classifier.classify(text, threshold=settings.toxicity_threshold)
        model = shona_toxicity_classifier.BASE_MODEL + " (fine-tuned)"
    else:
        result = en_classifier.classify(text, threshold=settings.toxicity_threshold)
        model = en_classifier.MODEL_NAME

    return ModerationResult(language=language, model=model, **result.to_dict())


def available_models() -> dict:
    return {
        "en": en_classifier.MODEL_NAME,
        "sn": (
            shona_toxicity_classifier.BASE_MODEL + " (fine-tuned)"
            if shona_toxicity_classifier.is_available()
            else "not fine-tuned yet"
        ),
    }


def warm_up_model() -> None:
    en_classifier.warm_up()
    if shona_toxicity_classifier.is_available():
        shona_toxicity_classifier.warm_up()
