from ml.inference import shona_toxicity_classifier
from ml.inference import toxicity_classifier as en_classifier
from ml.preprocessing.language_detector import detect_language

from app.core.config import get_settings
from app.schemas.moderation import ModerationBatchResult, ModerationResult

ShonaModelUnavailableError = shona_toxicity_classifier.ModelNotFineTunedError


def resolve_thresholds(custom_thresholds: dict[str, float] | None = None) -> dict[str, float]:
    return get_settings().get_thresholds(overrides=custom_thresholds)


def moderate(
    text: str,
    language: str = "en",
    custom_thresholds: dict[str, float] | None = None,
) -> ModerationResult:
    thresholds = resolve_thresholds(custom_thresholds)
    resolved_lang = detect_language(text, default="en") if language == "auto" else language

    if resolved_lang == "sn":
        result = shona_toxicity_classifier.classify(text, threshold=thresholds)
        model = shona_toxicity_classifier.BASE_MODEL + " (fine-tuned)"
    else:
        result = en_classifier.classify(text, threshold=thresholds)
        model = en_classifier.MODEL_NAME

    return ModerationResult(language=resolved_lang, model=model, **result.to_dict())


def moderate_batch(
    texts: list[str],
    language: str = "en",
    custom_thresholds: dict[str, float] | None = None,
) -> ModerationBatchResult:
    thresholds = resolve_thresholds(custom_thresholds)

    if language == "auto":
        # Group by detected language to batch efficiently while preserving order
        en_indices, en_texts = [], []
        sn_indices, sn_texts = [], []

        for idx, text in enumerate(texts):
            lang = detect_language(text, default="en")
            if lang == "sn":
                sn_indices.append(idx)
                sn_texts.append(text)
            else:
                en_indices.append(idx)
                en_texts.append(text)

        results: list[ModerationResult | None] = [None] * len(texts)

        if en_texts:
            en_outputs = en_classifier.classify_batch(en_texts, threshold=thresholds)
            for orig_idx, out in zip(en_indices, en_outputs):
                results[orig_idx] = ModerationResult(
                    language="en",
                    model=en_classifier.MODEL_NAME,
                    **out.to_dict(),
                )

        if sn_texts:
            sn_outputs = shona_toxicity_classifier.classify_batch(sn_texts, threshold=thresholds)
            for orig_idx, out in zip(sn_indices, sn_outputs):
                results[orig_idx] = ModerationResult(
                    language="sn",
                    model=shona_toxicity_classifier.BASE_MODEL + " (fine-tuned)",
                    **out.to_dict(),
                )

        final_results = [r for r in results if r is not None]

    elif language == "sn":
        sn_outputs = shona_toxicity_classifier.classify_batch(texts, threshold=thresholds)
        final_results = [
            ModerationResult(
                language="sn",
                model=shona_toxicity_classifier.BASE_MODEL + " (fine-tuned)",
                **out.to_dict(),
            )
            for out in sn_outputs
        ]
    else:
        en_outputs = en_classifier.classify_batch(texts, threshold=thresholds)
        final_results = [
            ModerationResult(
                language="en",
                model=en_classifier.MODEL_NAME,
                **out.to_dict(),
            )
            for out in en_outputs
        ]

    total = len(final_results)
    flagged_count = sum(1 for r in final_results if r.is_toxic)

    return ModerationBatchResult(
        results=final_results,
        total=total,
        flagged_count=flagged_count,
    )


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

