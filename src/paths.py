"""Path management and data directory setup."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


def setup_data_dirs():
    """Create all required data subdirectories."""
    subdirs = [
        "embeddings",
        "faiss",
        "umap",
        "clusters",
        "temporal",
        "gdelt",
    ]
    for subdir in subdirs:
        path = DATA_DIR / subdir
        path.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


# Inline setup on module import
DATA_DIR_READY = setup_data_dirs()

# =========================================================================
# MAIN DATA FILES
# =========================================================================
RAW_PARQUET = DATA_DIR / "rolargesum_raw.parquet"
CLEAN_CSV = DATA_DIR / "rolargesum_train_clean.csv"

# Alias for backward compatibility
INPUT_CSV = CLEAN_CSV

# =========================================================================
# EMBEDDINGS & CLUSTERING
# =========================================================================
EMB_DIR = DATA_DIR / "embeddings"
FAISS_DIR = DATA_DIR / "faiss"
UMAP_DIR = DATA_DIR / "umap"
CLUSTER_DIR = DATA_DIR / "clusters"

# =========================================================================
# TEMPORAL & GDELT
# =========================================================================
TEMPORAL_DIR = DATA_DIR / "temporal"
GDELT_DIR = DATA_DIR / "gdelt"

# =========================================================================
# RESULTS
# =========================================================================
CLUSTERED_CSV = CLUSTER_DIR / "clustered_data.csv"
HDBSCAN_CONFIG_RESULTS = CLUSTER_DIR / "hdbscan_config_results.csv"
TEMPORAL_STATS = TEMPORAL_DIR / "cluster_temporal_stats.csv"
TFIDF_CLUSTERS_CSV = DATA_DIR / "tfidf_kmeans_clusters.csv"
TFIDF_ABLATION_REPORT = DATA_DIR / "tfidf_ablation_report.csv"

# =========================================================================
# STOPWORDS
# =========================================================================
STOPWORDS_PATH = BASE_DIR / "stopwords-ro.txt"
