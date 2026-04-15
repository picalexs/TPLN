"""Topic normalization and mapping utilities."""

from __future__ import annotations

import re
from typing import Any

from .text_processing import strip_diacritics

CANONICAL_TOPICS = {
    "politica",
    "international",
    "economie",
    "social",
    "justitie",
    "sanatate",
    "sport",
    "educatie",
    "stiinta",
    "diverse",
    "cultura",
    "necunoscut",
}

# Alias labels are normalized to ASCII and lower-case before lookup.
TOPIC_MAPPING = {
    # Core political and election labels.
    "politic": "politica",
    "politica": "politica",
    "guvern": "politica",
    "politic intern": "politica",
    "politica interna": "politica",
    "alegeri": "politica",
    "parlamentare 2019": "politica",
    "anticipate 2021": "politica",
    "locale 2023": "politica",
    "locale noi 21": "politica",

    # Foreign affairs and geopolitics.
    "extern": "international",
    "externe": "international",
    "international": "international",
    "stiri externe": "international",
    "razboi": "international",
    "nato ue": "international",
    "in lume": "international",
    "geopolitical futures": "international",

    # Economy and finance.
    "economie": "economie",
    "economic": "economie",
    "economica": "economie",
    "financiar": "economie",
    "stiri economice": "economie",
    "bani": "economie",
    "business": "economie",

    # Society / public-interest life.
    "social": "social",
    "societate": "social",
    "stiri sociale": "social",
    "viata": "social",
    "life": "social",
    "oameni": "social",

    # Justice / investigations.
    "justitie": "justitie",
    "stiri justitie": "justitie",
    "juridic": "justitie",
    "ancheta": "justitie",
    "investigatii": "justitie",

    # Health.
    "sanatate": "sanatate",
    "coronavirus": "sanatate",
    "covid": "sanatate",
    "covid 19": "sanatate",
    "sars cov2": "sanatate",

    "sport": "sport",

    # Education.
    "educatie": "educatie",
    "invatamant": "educatie",

    # Science / technology.
    "stiinta": "stiinta",
    "it stiinta": "stiinta",
    "tehnologie": "stiinta",
    "sci tech": "stiinta",

    # Culture / entertainment.
    "cultura": "cultura",
    "cultura media": "cultura",
    "entertainment": "cultura",
    "showbiz": "cultura",

    # Section or format labels that are not reliable topical buckets.
    # These were checked against publisher navigation or sample articles and are
    # intentionally folded into a neutral fallback bucket until the pipeline
    # grows a separate content_type/content_format field.
    "diverse": "diverse",
    "stiri diverse": "diverse",
    "evenimente": "diverse",
    "meteo": "diverse",
    "comunicate": "diverse",
    "revista presei": "diverse",
    "editorial": "diverse",
    "editoriale": "diverse",
    "reportaj": "diverse",
    "reportaje": "diverse",
    "interviu": "diverse",
    "interviuri": "diverse",
    "emisiune": "diverse",
    "emisiuni": "diverse",
    "advertorial": "diverse",
    "advertoriale": "diverse",
    "toate stirile": "diverse",
    "opinii": "diverse",
    "istorii": "diverse",
    "stop fals": "diverse",
    "revolutie": "diverse",

    "root": "necunoscut",
    "necunoscut": "necunoscut",
    "uncategorized": "necunoscut",
    "echipa radio chisinau": "necunoscut",
    "dosar": "justitie",
    "diaspora": "social",
}

_MISSING_TOPIC_VALUES = {
    "",
    "-",
    "na",
    "n/a",
    "none",
    "null",
    "nan",
    "unknown",
    "necunoscut",
}

_COMPOSITE_SPLIT_RE = re.compile(r"[,/|;&+]+|\bsi\b|\bsau\b")
_WHITESPACE_RE = re.compile(r"\s+")


def _is_missing_topic(topic: Any) -> bool:
    """Return True for blank, placeholder, or NaN-like topic values."""
    if topic is None:
        return True
    if isinstance(topic, float) and topic != topic:
        return True

    normalized = strip_diacritics(str(topic)).strip().lower()
    return normalized in _MISSING_TOPIC_VALUES


def _normalize_key(topic: Any) -> str:
    """Normalize a raw topic label to a comparison-friendly ASCII key."""
    ascii_topic = strip_diacritics(str(topic)).lower().strip()
    ascii_topic = re.sub(r"[\-_./:]+", " ", ascii_topic)
    ascii_topic = _WHITESPACE_RE.sub(" ", ascii_topic).strip()
    return ascii_topic


def _lookup_topic(topic_key: str) -> str | None:
    """Return a canonical topic if the key is a known alias."""
    if not topic_key:
        return None

    if topic_key in TOPIC_MAPPING:
        return TOPIC_MAPPING[topic_key]

    compact_key = topic_key.replace(" ", "")
    if compact_key in TOPIC_MAPPING:
        return TOPIC_MAPPING[compact_key]

    if topic_key in CANONICAL_TOPICS:
        return topic_key
    if compact_key in CANONICAL_TOPICS:
        return compact_key

    return None


def _resolve_composite_topic(topic_text: str) -> str | None:
    """Resolve labels that mix multiple values into a single canonical bucket."""
    for part in _COMPOSITE_SPLIT_RE.split(topic_text):
        part_key = _normalize_key(part)
        resolved = _lookup_topic(part_key)
        if resolved is not None:
            return resolved

    for part in topic_text.split(" "):
        part_key = _normalize_key(part)
        resolved = _lookup_topic(part_key)
        if resolved is not None:
            return resolved

    return None


def normalize_topic(topic: Any) -> str:
    """Normalize a topic label to a stable bucket name.

    Missing or blank labels become ``necunoscut``. Known aliases are mapped to
    canonical topic buckets. Composite labels use the first resolvable topical
    part. Format-like labels are collapsed into ``diverse``. Anything else is
    kept as a normalized ASCII fallback so genuinely novel labels remain visible.
    """
    if _is_missing_topic(topic):
        return "necunoscut"

    topic_key = _normalize_key(topic)
    resolved = _lookup_topic(topic_key)
    if resolved is not None:
        return resolved

    composite_resolved = _resolve_composite_topic(topic_key)
    if composite_resolved is not None:
        return composite_resolved

    return topic_key or "necunoscut"


def normalize_topic_with_reason(topic: Any) -> tuple[str, str]:
    """Normalize a topic label and explain how the bucket was chosen."""
    if _is_missing_topic(topic):
        return "necunoscut", "missing_topic"

    topic_key = _normalize_key(topic)
    if topic_key == "root":
        return "necunoscut", "metadata_topic"

    resolved = _lookup_topic(topic_key)
    if resolved is not None:
        if resolved == topic_key and topic_key in CANONICAL_TOPICS:
            return resolved, "canonical_topic"
        return resolved, "mapped_topic_alias"

    composite_resolved = _resolve_composite_topic(topic_key)
    if composite_resolved is not None:
        return composite_resolved, "composite_topic"

    return topic_key or "necunoscut", "unmapped_topic"
