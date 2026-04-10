"""Date extraction utilities from URLs and text."""

import re
import pandas as pd
from pandas import Timestamp
from .config import ROMANIAN_MONTHS

TimestampOrNaT = Timestamp | type(pd.NaT)

TEXT_DATE_SCAN_CHARS = 220
TEXT_DATE_MAX_POSITION = 120
TEXT_DATE_MIN_YEAR = 2015
TEXT_DATE_MAX_YEAR_FUTURE_OFFSET = 1

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


def _looks_like_metadata_date(text: str, match_start: int) -> bool:
    """Keep only dates that appear near the article header or date cues."""
    if match_start <= TEXT_DATE_MAX_POSITION:
        return True

    context_start = max(0, match_start - 35)
    context = text[context_start:match_start].lower()
    return any(cue in context for cue in TEXT_DATE_CUES)


def extract_date_from_url(url) -> TimestampOrNaT:
    """Extract date from URL patterns like /2021/03/23/ or /2021-03-23/"""
    if pd.isna(url):
        return pd.NaT
    
    url = str(url)
    match = re.search(r"(20\d{2})[-/](\d{2})[-/](\d{2})", url)
    
    if match:
        return _build_timestamp(match.group(1), match.group(2), match.group(3))
    return pd.NaT


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
