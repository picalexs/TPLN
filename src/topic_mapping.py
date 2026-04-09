"""Topic normalization and mapping utilities."""

from .text_processing import strip_diacritics

TOPIC_MAPPING = {
    "politic": "politica",
    "politica": "politica",
    "guvern": "politica",

    "externe": "international",
    "extern": "international",
    "international": "international",
    "stiri-externe": "international",
    "razboi": "international",

    "economie": "economie",
    "economic": "economie",
    "financiar": "economie",
    "stiri-economice": "economie",
    "bani": "economie",

    "social": "social",
    "societate": "social",
    "stiri-sociale": "social",
    "viata": "social",

    "justitie": "justitie",
    "stiri-justitie": "justitie",

    "sanatate": "sanatate",
    "sport": "sport",
    "educatie": "educatie",

    "stiinta": "stiinta",
    "it-stiinta": "stiinta",
    "tehnologie": "stiinta",

    "diverse": "diverse",
    "stiri-diverse": "diverse",

    "cultura": "cultura",
}


def normalize_topic(topic: str) -> str:
    """Normalize topic to standard group names.
    
    Handles diacritics and maps variants to canonical forms.
    """
    topic = str(topic).strip().lower()
    topic_ascii = strip_diacritics(topic)

    # Try ASCII version first (handles politică→politica, cultură→cultura)
    if topic_ascii in TOPIC_MAPPING:
        return TOPIC_MAPPING[topic_ascii]
    
    # Then try original
    if topic in TOPIC_MAPPING:
        return TOPIC_MAPPING[topic]

    return topic if topic != "" else "necunoscut"
