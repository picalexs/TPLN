"""Configuration constants for TPLN pipeline."""

from __future__ import annotations

import math

# =========================================================================
# MODEL CONFIGURATION
# =========================================================================
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# =========================================================================
# EMBEDDING & CLUSTERING
# =========================================================================
TEXT_COLUMN = "short_document"
MIN_TOPIC_SIZE = 200

# Number of leading characters (title + body excerpt) fed into SBERT.
# Bumped from 500 to 1500 to give the embedder enough context to
# discriminate between articles that share a templated lede.
SHORT_DOCUMENT_MAX_CHARS = 1500

# Fallback only. Use compute_umap_n_neighbors(topic_size) in practice so
# the UMAP manifold scales with the number of articles in the topic.
UMAP_N_NEIGHBORS = 30


def compute_umap_n_neighbors(topic_size: int) -> int:
    """Return a UMAP `n_neighbors` value scaled to the topic size.

    Small neighborhoods preserve only micro-structure, which makes HDBSCAN
    fragment the space into thousands of tiny clusters on large topics.
    Scaling with ~sqrt(N) keeps the manifold smooth while staying within
    UMAP's practical performance envelope.
    """
    if topic_size <= 0:
        return UMAP_N_NEIGHBORS
    scaled = int(round(math.sqrt(topic_size) * 1.2))
    return max(30, min(200, scaled))


def build_hdbscan_configs(topic_size: int) -> list[dict]:
    """Return a size-aware HDBSCAN hyperparameter sweep.

    Two axes matter and the previous sweep only varied one of them:

    * `min_cluster_size` sets the smallest cluster the algorithm will
      keep. Too small on a 100k-article topic and HDBSCAN fragments into
      thousands of micro-clusters; too large and the condensed tree
      collapses into a single giant bucket.
    * `min_samples` controls how conservative the cluster boundary is.
      High values (~= `min_cluster_size` / 4) treat many borderline
      articles as noise. Low values (~= `min_cluster_size` / 8-10) let
      more articles into clusters at the cost of looser coherence.
    """
    if topic_size < 500:
        base = 10
    elif topic_size < 2_000:
        base = 15
    elif topic_size < 8_000:
        base = 25
    elif topic_size < 25_000:
        base = 40
    elif topic_size < 75_000:
        base = 60
    elif topic_size < 150_000:
        base = 80
    else:
        base = 100

    # Loose vs tight min_samples ratios. "loose" pulls borderline
    # articles into clusters (low noise); "tight" enforces cohesion
    # (higher noise, cleaner clusters).
    loose_samples = max(3, base // 8)
    tight_samples = max(5, base // 4)

    return [
        # Anchor at the base size with loose samples to minimize noise.
        {
            "min_cluster_size": base,
            "min_samples": loose_samples,
        },
        # Same size with tighter samples. If the topic has genuine dense
        # cores, this is where silhouette peaks.
        {
            "min_cluster_size": base,
            "min_samples": tight_samples,
        },
        # Slightly larger clusters with epsilon smoothing so nearby
        # dense regions merge instead of being split.
        {
            "min_cluster_size": int(base * 1.25),
            "min_samples": max(5, base // 6),
            "cluster_selection_epsilon": 0.02,
        },
        # A conservative safety net for very noisy topics.
        {
            "min_cluster_size": int(base * 1.5),
            "min_samples": tight_samples,
            "cluster_selection_epsilon": 0.04,
        },
    ]


# Backwards-compatible fallbacks. Prefer build_hdbscan_configs(topic_size).
HDBSCAN_DEFAULT_CONFIGS = build_hdbscan_configs(5_000)
HDBSCAN_TOPIC_CONFIGS: dict[str, list[dict]] = {}

# =========================================================================
# UMAP PARAMETERS TUNED FOR DENSITY-BASED CLUSTERING
# =========================================================================
# umap-learn's clustering guide recommends `min_dist=0.0` and a low
# `n_components` (5-10) when the downstream consumer is HDBSCAN. The
# defaults (`min_dist=0.1`, `n_components=2`) are tuned for
# human-friendly visualization and leave HDBSCAN with a manifold that is
# too spread out to form clean density cores.
UMAP_MIN_DIST = 0.0
UMAP_N_COMPONENTS = 10

# =========================================================================
# NEAR-DUPLICATE FILTERING AND NOISE REASSIGNMENT
# =========================================================================
# Romanian news has heavy wire syndication: identical press releases
# reprinted across dozens of portals. Those near-duplicates destabilize
# UMAP (exact-collision singularities) and fragment HDBSCAN clusters.
# Pre-grouping them by cosine similarity stabilizes the pipeline.
NEAR_DUPLICATE_COSINE_THRESHOLD = 0.97
NEAR_DUPLICATE_SEARCH_K = 5

# After HDBSCAN we reassign noise points whose nearest neighbors in the
# original embedding space strongly agree on a single cluster label.
# This mirrors BERTopic's `reduce_outliers(strategy="embeddings")` and
# HDBSCAN's soft-cluster reassignment; in practice it drops global noise
# from ~50% to ~15-20% without meaningfully hurting cluster coherence.
NOISE_REASSIGN_K = 10
NOISE_REASSIGN_MIN_AGREEMENT = 0.6

# =========================================================================
# TF-IDF CONFIGURATION
# =========================================================================
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM = (1, 2)
TOP_TERMS_PER_CLUSTER = 10

# =========================================================================
# CLUSTER INTERPRETABILITY (c-TF-IDF labels + representative titles)
# =========================================================================
# Class-based TF-IDF treats every cluster as a single "document" and
# extracts terms that distinguish it from all other clusters. This is
# the BERTopic formulation for topic labels; see MaartenGr/BERTopic.
CLUSTER_LABEL_TOP_TERMS = 10
CLUSTER_REPRESENTATIVE_TITLES = 5
# Text column used to build each cluster's class document. Falls back
# to `short_document` or `document` when the preferred column is
# missing from the parquet.
CLUSTER_LABEL_TEXT_COLUMN = "document_nostop"

# =========================================================================
# SINGLE-SOURCE CLUSTER FLAGGING
# =========================================================================
# Clusters whose articles come overwhelmingly from one outlet are almost
# always scraper/aggregator artefacts, not coordinated campaigns across
# the Romanian news ecosystem. We flag them so the dashboard can show a
# separate "multi-source only" suspicion ranking.
SINGLE_SOURCE_TOP_DOMAIN_SHARE = 0.90
SINGLE_SOURCE_MAX_DOMAIN_COUNT = 2

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
