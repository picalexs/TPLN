"""Date extraction utilities from URLs and text."""

import re
from typing import Any, TypeAlias
from urllib.parse import parse_qs, unquote, urlparse
import pandas as pd
from .config import (
    ROMANIAN_MONTHS,
    TIMESTAMP_MAX_FUTURE_DAYS_TEXT,
    TIMESTAMP_MAX_FUTURE_DAYS_URL,
    TIMESTAMP_QUALITY_HIGH,
    TIMESTAMP_QUALITY_LOW,
    TIMESTAMP_QUALITY_MEDIUM,
    TIMESTAMP_QUALITY_MISSING,
    TIMESTAMP_QUALITY_REJECTED_FUTURE,
    TIMESTAMP_SOURCE_TEXT,
    TIMESTAMP_SOURCE_MISSING,
    TIMESTAMP_SOURCE_URL,
    TIMESTAMP_TEXT_LOW_CONFIDENCE_BEFORE_YEAR,
)

TimestampOrNaT: TypeAlias = Any

TEXT_DATE_SCAN_CHARS = 220
TEXT_DATE_MAX_POSITION = 120
TEXT_DATE_MIN_YEAR = 2015
TEXT_DATE_MAX_YEAR_FUTURE_OFFSET = 1

URL_DATE_PATTERNS = (
    re.compile(
        r"(?<!\d)(20\d{2})[\/._-](0?[1-9]|1[0-2])[\/._-](0?[1-9]|[12]\d|3[01])(?!\d)"
    ),
    re.compile(
        r"(?<!\d)(0?[1-9]|[12]\d|3[01])[\/._-](0?[1-9]|1[0-2])[\/._-](20\d{2})(?!\d)"
    ),
)
URL_QUERY_DATE_KEYS = {
    "date",
    "datetime",
    "published",
    "pubdate",
    "publish_date",
    "publishdate",
    "timestamp",
    "time",
    "created",
    "updated",
    "article_date",
    "news_date",
}

TEXT_DATE_CUES = (
    "publicat",
    "publicata",
    "publicată",
    "actualizat",
    "actualizata",
    "actualizată",
    "updated",
    "data",
    "ora",
)


def _build_timestamp(year: str, month: str, day: str) -> TimestampOrNaT:
    """Build a validated timestamp from date parts."""
    ts = pd.to_datetime(f"{year}-{month}-{day}", errors="coerce")
    if pd.isna(ts):
        return pd.NaT

    max_year = pd.Timestamp.today().year + TEXT_DATE_MAX_YEAR_FUTURE_OFFSET
    if TEXT_DATE_MIN_YEAR <= ts.year <= max_year:
        return ts
    return pd.NaT


def validate_extracted_timestamp(
    timestamp: TimestampOrNaT,
    source: str,
) -> tuple[TimestampOrNaT, str]:
    """Validate an extracted timestamp and assign a quality label."""
    if pd.isna(timestamp):
        return pd.NaT, TIMESTAMP_QUALITY_MISSING

    source = str(source).strip().lower()
    today = pd.Timestamp.today().normalize()
    max_future_days = (
        TIMESTAMP_MAX_FUTURE_DAYS_TEXT
        if source == TIMESTAMP_SOURCE_TEXT
        else TIMESTAMP_MAX_FUTURE_DAYS_URL
    )
    if timestamp.normalize() > today + pd.Timedelta(days=max_future_days):
        return pd.NaT, TIMESTAMP_QUALITY_REJECTED_FUTURE

    if source == TIMESTAMP_SOURCE_URL:
        return timestamp, TIMESTAMP_QUALITY_HIGH

    if source == TIMESTAMP_SOURCE_TEXT:
        if timestamp.year < TIMESTAMP_TEXT_LOW_CONFIDENCE_BEFORE_YEAR:
            return timestamp, TIMESTAMP_QUALITY_LOW
        return timestamp, TIMESTAMP_QUALITY_MEDIUM

    return timestamp, TIMESTAMP_QUALITY_MISSING


def _looks_like_metadata_date(text: str, match_start: int) -> bool:
    """Keep only dates that appear near the article header or date cues."""
    if match_start <= TEXT_DATE_MAX_POSITION:
        return True

    context_start = max(0, match_start - 35)
    context = text[context_start:match_start].lower()
    return any(cue in context for cue in TEXT_DATE_CUES)


def _extract_timestamp_from_url_candidate(candidate: str) -> TimestampOrNaT:
    """Extract a timestamp from a URL path or a date-like query value."""
    if not candidate:
        return pd.NaT

    for pattern in URL_DATE_PATTERNS:
        match = pattern.search(candidate)
        if match:
            if pattern.pattern.startswith("(?<!\\d)(20"):
                return _build_timestamp(match.group(1), match.group(2), match.group(3))
            return _build_timestamp(match.group(3), match.group(2), match.group(1))

    compact_match = re.search(
        r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)",
        candidate,
    )
    if compact_match:
        return _build_timestamp(
            compact_match.group(1),
            compact_match.group(2),
            compact_match.group(3),
        )

    return pd.NaT


def _extract_timestamp_from_query(query: str) -> TimestampOrNaT:
    """Extract a timestamp from explicit date query parameters."""
    if not query:
        return pd.NaT

    params = parse_qs(query, keep_blank_values=True)

    year_values = params.get("year")
    month_values = params.get("month")
    day_values = params.get("day")
    if year_values and month_values and day_values:
        return _build_timestamp(
            year_values[0],
            month_values[0].zfill(2),
            day_values[0].zfill(2),
        )

    for key, values in params.items():
        if key.lower() in URL_QUERY_DATE_KEYS:
            for value in values:
                ts = _extract_timestamp_from_url_candidate(unquote(value))
                if not pd.isna(ts):
                    return ts

    return pd.NaT


def extract_date_from_url_with_source(url) -> tuple[TimestampOrNaT, str]:
    """Extract date from a URL and return the extraction source."""
    if pd.isna(url):
        return pd.NaT, TIMESTAMP_SOURCE_MISSING

    url = unquote(str(url))
    parsed = urlparse(url if "://" in url else f"http://{url}")

    for candidate in (parsed.path, parsed.fragment):
        timestamp = _extract_timestamp_from_url_candidate(candidate)
        if not pd.isna(timestamp):
            return timestamp, TIMESTAMP_SOURCE_URL

    timestamp = _extract_timestamp_from_query(parsed.query)
    if not pd.isna(timestamp):
        return timestamp, TIMESTAMP_SOURCE_URL

    timestamp = _extract_timestamp_from_url_candidate(url)
    if not pd.isna(timestamp):
        return timestamp, TIMESTAMP_SOURCE_URL

    return pd.NaT, TIMESTAMP_SOURCE_MISSING


def extract_date_from_url(url) -> TimestampOrNaT:
    """Extract date from URL patterns like /2021/03/23/ or /2021-03-23/."""
    timestamp, _ = extract_date_from_url_with_source(url)
    return timestamp


def extract_date_from_text(text: str) -> TimestampOrNaT:
    """Extract likely publication date from article header text."""
    if pd.isna(text):
        return pd.NaT

    text = str(text)[:TEXT_DATE_SCAN_CHARS]

    # Pattern 1: "23 martie 2021" or "23 Martie 2021"
    for month_name, month_num in ROMANIAN_MONTHS.items():
        pattern = rf"\b(\d{{1,2}})\s+{month_name}\s+(20\d{{2}})\b"
        match = re.search(pattern, text, re.IGNORECASE)
        if match and _looks_like_metadata_date(text, match.start()):
            return _build_timestamp(match.group(2), month_num, match.group(1).zfill(2))

    # Pattern 2: "23.03.2021" or "23/03/2021" (day.month.year, European format)
    match = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](20\d{2})\b", text)
    if match and _looks_like_metadata_date(text, match.start()):
        return _build_timestamp(match.group(3), match.group(2).zfill(2), match.group(1).zfill(2))

    # Pattern 3: "2021-03-23" (ISO format in text)
    match = re.search(r"\b(20\d{2})[-.](\d{2})[-.](\d{2})\b", text)
    if match and _looks_like_metadata_date(text, match.start()):
        return _build_timestamp(match.group(1), match.group(2), match.group(3))

    return pd.NaT
