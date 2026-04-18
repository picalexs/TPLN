"""Text processing utilities for Romanian language data."""

import re
import threading
import unicodedata
import pandas as pd
from .paths import STOPWORDS_PATH

_STOPWORDS_CACHE = None
_STOPWORDS_LOCK = threading.Lock()


def strip_emojis(text: str) -> str:
    """Remove emoji and pictographic symbols while preserving regular text."""
    return "".join(
        char for char in text
        if unicodedata.category(char) != "So"
    )


def strip_diacritics(text: str) -> str:
    """Remove Romanian diacritics: ă→a, î→i, â→a, ș→s, ț→t"""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def load_stopwords():
    """Load Romanian stopwords from file. Cached after first load. Thread-safe."""
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None:
        return _STOPWORDS_CACHE
    with _STOPWORDS_LOCK:
        if _STOPWORDS_CACHE is None:  # double-checked locking
            with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
                stopwords = {line.strip().lower() for line in f if line.strip()}
            stopwords_nodiac = {strip_diacritics(w) for w in stopwords}
            _STOPWORDS_CACHE = stopwords | stopwords_nodiac
    return _STOPWORDS_CACHE


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, normalize whitespace."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = strip_emojis(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deep_clean_text(text: str) -> str:
    """Extended cleaning: removes HTML, URLs, punctuation from boundaries."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = strip_emojis(text)
    text = text.replace("\n", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords_and_clean(text: str) -> str:
    """Remove stopwords, strip diacritics, filter short/numeric tokens.
    
    Produces clean ASCII text suitable for TF-IDF.
    """
    if pd.isna(text):
        return ""
    
    stopwords = load_stopwords()
    text = str(text)
    
    # Split hyphenated words (e.g. "într-un" -> "într" "un")
    text = text.replace("-", " ").replace("–", " ").replace("—", " ")
    words = text.split()
    filtered = []
    
    for w in words:
        clean_w = w.strip(".,;:!?\"'()…»«/\\")
        if not clean_w:
            continue
        
        clean_w = strip_diacritics(clean_w)
        
        # Skip short tokens
        if len(clean_w) <= 2:
            continue
        
        # Skip numeric-only tokens
        if clean_w.isdigit():
            continue
        
        # Skip stopwords
        if clean_w in stopwords:
            continue
        
        filtered.append(clean_w)
    
    return " ".join(filtered)
