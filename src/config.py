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
UMAP_N_NEIGHBORS = 30

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
        {"min_cluster_size": 12, "min_samples": 4, "cluster_selection_method": "leaf"},
        {"min_cluster_size": 15, "min_samples": 6, "cluster_selection_method": "leaf"},
        {"min_cluster_size": 20, "min_samples": 8, "cluster_selection_method": "leaf", "cluster_selection_epsilon": 0.02},
        {"min_cluster_size": 25, "min_samples": 10, "cluster_selection_method": "eom", "cluster_selection_epsilon": 0.05},
    ],
    "social": [
        {"min_cluster_size": 10, "min_samples": 3},
        {"min_cluster_size": 15, "min_samples": 6, "cluster_selection_method": "leaf"},
        {"min_cluster_size": 20, "min_samples": 8, "cluster_selection_method": "leaf", "cluster_selection_epsilon": 0.02},
        {"min_cluster_size": 25, "min_samples": 10, "cluster_selection_method": "eom", "cluster_selection_epsilon": 0.05},
    ],
    "international": [
        {"min_cluster_size": 10, "min_samples": 3},
        {"min_cluster_size": 15, "min_samples": 6, "cluster_selection_method": "leaf"},
        {"min_cluster_size": 20, "min_samples": 8, "cluster_selection_method": "leaf", "cluster_selection_epsilon": 0.02},
        {"min_cluster_size": 25, "min_samples": 10, "cluster_selection_method": "eom", "cluster_selection_epsilon": 0.05},
    ],
    "economie": [
        {"min_cluster_size": 8, "min_samples": 3},
        {"min_cluster_size": 12, "min_samples": 4, "cluster_selection_method": "leaf"},
        {"min_cluster_size": 15, "min_samples": 6, "cluster_selection_method": "leaf", "cluster_selection_epsilon": 0.02},
        {"min_cluster_size": 20, "min_samples": 8, "cluster_selection_method": "eom", "cluster_selection_epsilon": 0.05},
    ],
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
    "dialect", "author"
]

COLUMNS_TO_PRESERVE = [
    "title", "text", "summary", "keywords", "topics",
    "dialect", "url", "author", "document", "short_document",
    "document_nostop", "timestamp", "timestamp_source", "timestamp_quality"
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

# =========================================================================
# TIMESTAMP COVERAGE
# =========================================================================
TIMESTAMP_SOURCE_COLUMN = "timestamp_source"
TIMESTAMP_SOURCE_URL = "url"
TIMESTAMP_SOURCE_TEXT = "text"
TIMESTAMP_SOURCE_MISSING = "missing"
TIMESTAMP_QUALITY_COLUMN = "timestamp_quality"
TIMESTAMP_QUALITY_HIGH = "high"
TIMESTAMP_QUALITY_MEDIUM = "medium"
TIMESTAMP_QUALITY_LOW = "low"
TIMESTAMP_QUALITY_MISSING = "missing"
TIMESTAMP_QUALITY_REJECTED_FUTURE = "rejected_future"
TIMESTAMP_SOURCE_VALUES = (
    TIMESTAMP_SOURCE_URL,
    TIMESTAMP_SOURCE_TEXT,
    TIMESTAMP_SOURCE_MISSING,
)
TIMESTAMP_QUALITY_VALUES = (
    TIMESTAMP_QUALITY_HIGH,
    TIMESTAMP_QUALITY_MEDIUM,
    TIMESTAMP_QUALITY_LOW,
    TIMESTAMP_QUALITY_MISSING,
    TIMESTAMP_QUALITY_REJECTED_FUTURE,
)
TIMESTAMP_DOMAIN_REPORT_TOP_N = 15
TIMESTAMP_DOMAIN_MIN_ROWS_FOR_WEAKNESS = 20
TIMESTAMP_TEXT_LOW_CONFIDENCE_BEFORE_YEAR = 2018
TIMESTAMP_MAX_FUTURE_DAYS_URL = 2
TIMESTAMP_MAX_FUTURE_DAYS_TEXT = 0
