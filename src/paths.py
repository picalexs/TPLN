"""Path management and data directory setup."""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


def setup_data_dirs():
    """Create all required data subdirectories."""
    subdirs = [
        "embeddings",
        "clusters",
        "temporal",
        "gdelt",
        "dashboard",
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
CLEAN_PARQUET = DATA_DIR / "rolargesum_train_clean.parquet"

# =========================================================================
# EMBEDDINGS & CLUSTERING
# =========================================================================
EMB_DIR = DATA_DIR / "embeddings"
CLUSTER_DIR = DATA_DIR / "clusters"

# =========================================================================
# TEMPORAL & GDELT
# =========================================================================
TEMPORAL_DIR = DATA_DIR / "temporal"
GDELT_DIR = DATA_DIR / "gdelt"
DASHBOARD_DIR = DATA_DIR / "dashboard"

# =========================================================================
# RESULTS
# =========================================================================
CLUSTERED_PARQUET = CLUSTER_DIR / "clustered_data.parquet"
HDBSCAN_CONFIG_RESULTS = CLUSTER_DIR / "hdbscan_config_results.parquet"
TEMPORAL_STATS = TEMPORAL_DIR / "cluster_temporal_stats.parquet"
EVALUATION_REPORT = DATA_DIR / "evaluation_report.parquet"
TFIDF_ABLATION_REPORT = DATA_DIR / "tfidf_ablation_report.parquet"

# =========================================================================
# DASHBOARD ASSETS
# =========================================================================
DASHBOARD_META_PARQUET = DASHBOARD_DIR / "dashboard_meta.parquet"
DASHBOARD_TOPIC_SUMMARY_PARQUET = DASHBOARD_DIR / "topic_summary.parquet"
DASHBOARD_CLUSTER_OVERVIEW_PARQUET = DASHBOARD_DIR / "cluster_overview.parquet"
DASHBOARD_CLUSTER_DAILY_PARQUET = DASHBOARD_DIR / "cluster_daily_counts.parquet"
DASHBOARD_CLUSTER_ARTICLES_PARQUET = DASHBOARD_DIR / "cluster_articles.parquet"
DASHBOARD_SCATTER_SAMPLE_PARQUET = DASHBOARD_DIR / "scatter_sample.parquet"
DASHBOARD_TEMPORAL_PARQUET = DASHBOARD_DIR / "temporal_stats.parquet"
DASHBOARD_CONFIG_PARQUET = DASHBOARD_DIR / "config_results.parquet"
DASHBOARD_EVAL_PARQUET = DASHBOARD_DIR / "evaluation_report.parquet"
DASHBOARD_ABLATION_PARQUET = DASHBOARD_DIR / "tfidf_ablation.parquet"

# Backward-friendly aliases for newer dashboard builder code.
DASHBOARD_GLOBAL_META = DASHBOARD_META_PARQUET
DASHBOARD_TOPIC_SUMMARY = DASHBOARD_TOPIC_SUMMARY_PARQUET
DASHBOARD_CLUSTER_OVERVIEW = DASHBOARD_CLUSTER_OVERVIEW_PARQUET
DASHBOARD_CLUSTER_DAILY_COUNTS = DASHBOARD_CLUSTER_DAILY_PARQUET
DASHBOARD_ARTICLE_DETAIL = DASHBOARD_CLUSTER_ARTICLES_PARQUET
DASHBOARD_UMAP_SAMPLE = DASHBOARD_SCATTER_SAMPLE_PARQUET

# =========================================================================
# STOPWORDS
# =========================================================================
STOPWORDS_PATH = BASE_DIR / "stopwords-ro.txt"
