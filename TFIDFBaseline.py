"""
TF-IDF Baseline & Ablation Study
=============================================
Ablation comparison required by project spec:
    TF-IDF + KMeans  vs  Sentence-BERT + HDBSCAN

Also runs KMeans on SBERT embeddings (for HDBSCAN vs KMeans comparison).

Outputs:
    data/tfidf_kmeans_clusters.csv     article-level cluster labels
    data/tfidf_ablation_report.csv     per-topic metrics comparison

Run AFTER:
    python DataCuration.py
    python EmbeddingsClustering.py
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.config import (
    MAX_ROWS, MIN_TOPIC_SIZE,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM, TOP_TERMS_PER_CLUSTER
)
from src.paths import (
    CLEAN_CSV, HDBSCAN_CONFIG_RESULTS, EMB_DIR,
    TFIDF_CLUSTERS_CSV, TFIDF_ABLATION_REPORT
)
from src.io_utils import load_clean_csv
from src.topic_mapping import normalize_topic

warnings.filterwarnings("ignore")


def build_metric_space(X_sparse):
    """Create a compact dense representation for dense-only clustering metrics."""
    max_components = min(100, X_sparse.shape[0] - 1, X_sparse.shape[1] - 1)
    if max_components < 2:
        print("  Metric space: skipped TruncatedSVD (matrix too small).")
        return None

    svd = TruncatedSVD(n_components=max_components, random_state=42)
    X_reduced = np.asarray(svd.fit_transform(X_sparse), dtype=np.float32)
    explained = float(svd.explained_variance_ratio_.sum())
    print(
        f"  Metric space via TruncatedSVD: {X_reduced.shape} "
        f"(explained variance={explained:.2%})"
    )
    return X_reduced


# =========================================================================
# LOAD DATA
# =========================================================================
if not os.path.exists(CLEAN_CSV):
    raise FileNotFoundError(f"Cleaned CSV not found at {CLEAN_CSV}. Run DataCuration.py first.")

df = load_clean_csv(nrows=MAX_ROWS)

df["document_nostop"] = df["document_nostop"].fillna("").astype(str)
df["topics"] = df["topics"].fillna("").astype(str)

df["topic_group"] = df["topics"].apply(normalize_topic)

topic_counts = df["topic_group"].value_counts()
eligible_topics = topic_counts[topic_counts >= MIN_TOPIC_SIZE].index.tolist()

print(f"Eligible topics: {eligible_topics}")
print(f"Total rows: {df.shape[0]}")


# =========================================================================
# INFER K FROM HDBSCAN RESULTS
# =========================================================================
hdbscan_k_per_topic = {}

if os.path.exists(HDBSCAN_CONFIG_RESULTS):
    hdb_df = pd.read_csv(HDBSCAN_CONFIG_RESULTS)
    best_hdb = (
        hdb_df.sort_values("selection_score", ascending=False)
        .groupby("topic_group")
        .first()
        .reset_index()
    )
    for _, row in best_hdb.iterrows():
        hdbscan_k_per_topic[row["topic_group"]] = max(int(row["num_clusters"]), 3)
    print(f"\nK from HDBSCAN: {hdbscan_k_per_topic}")
else:
    print("No HDBSCAN results found; using K=10 default.")


# =========================================================================
# ABLATION A: TF-IDF + KMEANS PER TOPIC
# =========================================================================
all_ablation = []
final_chunks = []

print("\n" + "=" * 60)
print("ABLATION A: TF-IDF + KMeans")
print("=" * 60)

for topic in eligible_topics:
    print(f"\n--- Topic: {topic} ---")

    sub = df[df["topic_group"] == topic].copy().reset_index(drop=True)
    corpus = sub["document_nostop"].tolist()

    k = hdbscan_k_per_topic.get(topic, 10)
    print(f"  Articles: {len(sub)} | K={k}")

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM,
        min_df=2,
        sublinear_tf=True,
    )
    X_sparse = vectorizer.fit_transform(corpus)
    print(f"  TF-IDF matrix: {X_sparse.shape}")
    X_metric = build_metric_space(X_sparse)

    # KMeans
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        max_iter=300,
        batch_size=min(len(sub), max(256, k * 10)),
    )
    labels = km.fit_predict(X_sparse)

    # Silhouette (sampled for speed)
    try:
        sample_matrix = X_metric if X_metric is not None else X_sparse
        sample_size = min(5000, len(labels))
        idx_sample = np.random.RandomState(42).choice(len(labels), sample_size, replace=False)
        sil = silhouette_score(sample_matrix[idx_sample], labels[idx_sample], metric="cosine")
    except Exception:
        sil = None

    try:
        db = davies_bouldin_score(X_metric, labels) if X_metric is not None else None
    except Exception:
        db = None

    print(f"  Silhouette (cosine): {sil}")
    print(f"  Davies-Bouldin:      {db}")

    # Top terms per cluster
    feature_names = vectorizer.get_feature_names_out()
    cluster_terms = {}
    for c in range(k):
        center = km.cluster_centers_[c]
        top_idx = center.argsort()[::-1][:TOP_TERMS_PER_CLUSTER]
        cluster_terms[c] = ", ".join(feature_names[top_idx])

    print(f"\n  Top terms (first 3 clusters):")
    for c in list(cluster_terms.keys())[:3]:
        print(f"    Cluster {c}: {cluster_terms[c]}")

    sub["tfidf_cluster"] = labels
    sub["tfidf_top_terms"] = sub["tfidf_cluster"].map(cluster_terms)
    final_chunks.append(sub)

    all_ablation.append({
        "topic_group": topic,
        "n_articles": len(sub),
        "k": k,
        "method": "tfidf_kmeans",
        "silhouette": sil,
        "davies_bouldin": db,
        "num_clusters": k,
    })


# =========================================================================
# ABLATION B: SBERT EMBEDDINGS + KMEANS (vs HDBSCAN)
# =========================================================================
print("\n" + "=" * 60)
print("ABLATION B: SBERT + KMeans (same K)")
print("=" * 60)

for topic in eligible_topics:
    safe_topic = topic.replace("/", "_").replace(" ", "_")
    emb_path = EMB_DIR / f"{safe_topic}_embeddings.npy"

    if not emb_path.exists():
        print(f"\n--- Topic: {topic} - embeddings not found, skipping ---")
        continue

    print(f"\n--- Topic: {topic} ---")

    embeddings = np.asarray(np.load(emb_path), dtype=np.float32)
    k = hdbscan_k_per_topic.get(topic, 10)

    print(f"  Embeddings: {embeddings.shape} | K={k}")

    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(embeddings)

    try:
        sample_size = min(5000, len(labels))
        idx_sample = np.random.RandomState(42).choice(len(labels), sample_size, replace=False)
        sil = silhouette_score(embeddings[idx_sample], labels[idx_sample], metric="cosine")
    except Exception:
        sil = None

    try:
        db = davies_bouldin_score(embeddings, labels)
    except Exception:
        db = None

    print(f"  Silhouette (cosine): {sil}")
    print(f"  Davies-Bouldin:      {db}")

    all_ablation.append({
        "topic_group": topic,
        "n_articles": len(embeddings),
        "k": k,
        "method": "sbert_kmeans",
        "silhouette": sil,
        "davies_bouldin": db,
        "num_clusters": k,
    })


# =========================================================================
# ADD HDBSCAN RESULTS FOR COMPARISON
# =========================================================================
if os.path.exists(HDBSCAN_CONFIG_RESULTS):
    hdb_df = pd.read_csv(HDBSCAN_CONFIG_RESULTS)
    best_hdb = (
        hdb_df.sort_values("selection_score", ascending=False)
        .groupby("topic_group")
        .first()
        .reset_index()
    )
    for _, row in best_hdb.iterrows():
        all_ablation.append({
            "topic_group": row["topic_group"],
            "n_articles": row["topic_size"],
            "k": None,
            "method": "sbert_hdbscan",
            "silhouette": row.get("silhouette"),
            "davies_bouldin": None,
            "num_clusters": row["num_clusters"],
        })


# =========================================================================
# SAVE
# =========================================================================
if final_chunks:
    out_df = pd.concat(final_chunks, ignore_index=True)
    out_df.to_csv(TFIDF_CLUSTERS_CSV, index=False)
    print(f"\nTF-IDF cluster labels saved to: {TFIDF_CLUSTERS_CSV}")

ablation_df = pd.DataFrame(all_ablation)
ablation_df.to_csv(TFIDF_ABLATION_REPORT, index=False)
print(f"Ablation report saved to: {TFIDF_ABLATION_REPORT}")

print("\n" + "=" * 60)
print("ABLATION SUMMARY")
print("=" * 60)
print(ablation_df.to_string(index=False))
