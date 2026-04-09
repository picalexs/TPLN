"""Configuration constants for TPLN pipeline."""

# =========================================================================
# MODEL CONFIGURATION
# =========================================================================
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# =========================================================================
# EMBEDDING & CLUSTERING
# =========================================================================
TEXT_COLUMN = "short_document"
MIN_TOPIC_SIZE = 200
MAX_ROWS = 15000

# HDBSCAN configurations for hyperparameter sweep
HDBSCAN_DEFAULT_CONFIGS = [
    {"min_cluster_size": 8,  "min_samples": 3},
    {"min_cluster_size": 10, "min_samples": 4},
    {"min_cluster_size": 12, "min_samples": 6},
    {"min_cluster_size": 15, "min_samples": 8},
]

# Topic-specific HDBSCAN overrides
HDBSCAN_TOPIC_CONFIGS = {
    "politica": [
        {"min_cluster_size": 10, "min_samples": 4},
        {"min_cluster_size": 12, "min_samples": 6},
        {"min_cluster_size": 15, "min_samples": 8},
    ]
}

# =========================================================================
# TF-IDF CONFIGURATION
# =========================================================================
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM = (1, 2)
TOP_TERMS_PER_CLUSTER = 10

# =========================================================================
# TEXT COLUMN NAMES
# =========================================================================
COLUMNS_TO_CLEAN = [
    "title", "text", "summary", "keywords", "topics",
    "dialect", "url", "author"
]

COLUMNS_TO_PRESERVE = [
    "title", "text", "summary", "keywords", "topics",
    "dialect", "url", "author", "document", "short_document",
    "document_nostop", "timestamp"
]

# =========================================================================
# GDELT CONFIGURATION
# =========================================================================
GKG_COLUMN_INDICES = {
    "GKGRECORDID": 0,
    "DATE": 1,
    "SRCLC": 3,
    "SRCURL": 4,
    "V1THEMES": 7,
    "V1LOCATIONS": 9,
    "V1PERSONS": 11,
    "V1ORGS": 13,
    "V2TONE": 15,
    "V2GCAM": 16,
    "V2EXTRASXML": 27,
}

ROMANIAN_LANG_CODES = {"rum", "ron", "ro", "romanian", "moldavian", "mol"}

# =========================================================================
# DATE EXTRACTION
# =========================================================================
ROMANIAN_MONTHS = {
    "ianuarie": "01", "februarie": "02", "martie": "03", "aprilie": "04",
    "mai": "05", "iunie": "06", "iulie": "07", "august": "08",
    "septembrie": "09", "octombrie": "10", "noiembrie": "11", "decembrie": "12",
}
