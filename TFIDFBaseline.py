"""
TF-IDF baseline and ablation study.

Compares:
    TF-IDF + KMeans
    SBERT + KMeans
    SBERT + HDBSCAN

Output:
    data/tfidf_ablation_report.parquet
"""

from __future__ import annotations

import argparse
from time import perf_counter
from typing import Any, cast
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, spmatrix
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.config import MIN_TOPIC_SIZE, TFIDF_MAX_FEATURES, TFIDF_NGRAM, TOP_TERMS_PER_CLUSTER
from src.io_utils import load_clean_data
from src.paths import EMB_DIR, HDBSCAN_CONFIG_RESULTS, TFIDF_ABLATION_REPORT
from src.runtime_profile import apply_runtime_profile, detect_runtime_profile, format_runtime_profile
from src.topic_mapping import normalize_topic
from tqdm import tqdm

warnings.filterwarnings("ignore")

SILHOUETTE_SAMPLE_SIZE = 5_000
SBERT_MINIBATCH_THRESHOLD = 50_000


def build_metric_space(x_sparse) -> np.ndarray | None:
    """Create a compact dense representation for dense-only clustering metrics."""
    max_components = min(100, x_sparse.shape[0] - 1, x_sparse.shape[1] - 1)
    if max_components < 2:
        print("  Metric space: skipped TruncatedSVD (matrix too small).")
        return None

    svd = TruncatedSVD(n_components=max_components, random_state=42)
    x_reduced = np.asarray(svd.fit_transform(x_sparse), dtype=np.float32)
    explained = float(svd.explained_variance_ratio_.sum())
    print(
        f"  Metric space via TruncatedSVD: {x_reduced.shape} "
        f"(explained variance={explained:.2%})"
    )
    return x_reduced


def load_hdbscan_topic_k() -> dict[str, int]:
    """Infer topic-level K values from the best HDBSCAN config per topic."""
    if not HDBSCAN_CONFIG_RESULTS.exists():
        print("No HDBSCAN results found; using K=10 default.")
        return {}

    hdb_df = pd.read_parquet(HDBSCAN_CONFIG_RESULTS)
    best_hdb = (
        hdb_df.sort_values("selection_score", ascending=False)
        .groupby("topic_group")
        .first()
        .reset_index()
    )
    topic_k = {
        str(row["topic_group"]): max(int(row["num_clusters"]), 3)
        for _, row in best_hdb.iterrows()
    }
    print(f"\nK from HDBSCAN: {topic_k}")
    return topic_k


def effective_k(requested_k: int, topic_size: int, max_k: int | None) -> int:
    """Optionally cap K for speed while keeping it valid for the topic size."""
    safe_k = max(2, min(requested_k, topic_size - 1))
    if max_k is None:
        return safe_k
    return max(2, min(safe_k, max_k))


def sampled_silhouette(x_matrix, labels: np.ndarray, metric: str) -> float | None:
    """Compute a bounded silhouette score for speed and stability."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None

    try:
        sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(labels))
        idx_sample = np.random.RandomState(42).choice(len(labels), sample_size, replace=False)
        return float(silhouette_score(x_matrix[idx_sample], labels[idx_sample], metric=metric))
    except Exception:
        return None


def build_topic_indices(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build one index lookup per topic to avoid rescanning the full frame repeatedly."""
    return {
        str(topic): np.asarray(indices, dtype=np.int64)
        for topic, indices in df.groupby("topic_group", sort=False).indices.items()
    }


def build_sbert_kmeans(topic_size: int, k: int):
    """Use faster MiniBatchKMeans for very large topics, plain KMeans otherwise."""
    if topic_size >= SBERT_MINIBATCH_THRESHOLD:
        return MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            n_init=5,
            max_iter=200,
            batch_size=min(topic_size, max(2048, k * 64)),
            init_size=min(topic_size, max(4096, k * 8)),
            max_no_improvement=20,
            reassignment_ratio=0.01,
            verbose=1,
        )

    return KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        max_iter=300,
        verbose=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TF-IDF and SBERT ablation experiments from parquet artifacts.",
    )
    parser.add_argument("--nrows", type=int, default=None, help="Optional row cap for experiments")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Override CPU thread count for local linear algebra workloads",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=None,
        help="Optional cap for KMeans cluster count to speed up very large topics",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = detect_runtime_profile(device="cpu", cpu_threads=args.cpu_threads)
    apply_runtime_profile(profile)
    print(format_runtime_profile(profile))

    load_start = perf_counter()
    df = load_clean_data(
        nrows=args.nrows,
        columns=["document_nostop", "topics"],
    )
    df["document_nostop"] = df["document_nostop"].fillna("").astype(str)
    df["topics"] = df["topics"].fillna("").astype(str)
    df["topic_group"] = df["topics"].map(normalize_topic)
    print(f"Loaded baseline input in {perf_counter() - load_start:.1f}s")

    topic_indices = build_topic_indices(df)
    topic_counts = pd.Series({topic: len(indices) for topic, indices in topic_indices.items()}).sort_values(ascending=False)
    eligible_topics = [topic for topic, count in topic_counts.items() if count >= MIN_TOPIC_SIZE]
    print(f"Eligible topics: {eligible_topics}")
    print(f"Total rows: {df.shape[0]}")

    hdbscan_k_per_topic = load_hdbscan_topic_k()
    all_ablation: list[dict[str, object]] = []

    print("\n" + "=" * 60)
    print("ABLATION A: TF-IDF + KMeans")
    print("=" * 60)
    tfidf_progress = tqdm(eligible_topics, desc="TF-IDF topics", unit="topic")
    for topic_idx, topic in enumerate(tfidf_progress, start=1):
        topic_start = perf_counter()
        topic_name = str(topic)
        print(f"\n--- Topic {topic_idx}/{len(eligible_topics)}: {topic_name} ---")
        sub = df.take(topic_indices[topic_name]).reset_index(drop=True)
        corpus = sub["document_nostop"].tolist()
        requested_k = hdbscan_k_per_topic.get(topic_name, 10)
        k = effective_k(requested_k, len(sub), args.max_k)
        print(f"  Articles: {len(sub)} | K={k}" + (f" (requested {requested_k})" if k != requested_k else ""))

        tfidf_start = perf_counter()
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM,
            min_df=2,
            sublinear_tf=True,
            dtype=np.float32,
        )
        x_sparse = vectorizer.fit_transform(corpus)
        x_sparse_base = cast(spmatrix, x_sparse)
        x_sparse_csr = csr_matrix(x_sparse_base)
        print(f"  TF-IDF matrix: {x_sparse.shape} ({perf_counter() - tfidf_start:.1f}s)")

        metric_start = perf_counter()
        x_metric = build_metric_space(x_sparse)
        print(f"  Metric-space step: {perf_counter() - metric_start:.1f}s")

        cluster_start = perf_counter()
        cluster_matrix = x_metric if x_metric is not None else x_sparse_csr
        if x_metric is not None:
            feature_count = int(x_metric.shape[1])
        else:
            sparse_shape = cast(tuple[int, int], cast(Any, x_sparse_csr).shape)
            feature_count = int(sparse_shape[1])
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            n_init=5,
            max_iter=80,
            batch_size=min(len(sub), max(2048, k * 16)),
            init_size=min(len(sub), max(4096, k * 4)),
            max_no_improvement=10,
            verbose=1,
        )
        print(f"  TF-IDF clustering started on {feature_count} features...")
        labels = km.fit_predict(cluster_matrix)
        print(f"  TF-IDF clustering: {perf_counter() - cluster_start:.1f}s")

        metric_eval_start = perf_counter()
        sil = sampled_silhouette(
            x_metric if x_metric is not None else x_sparse_csr,
            labels,
            metric="cosine",
        )
        try:
            db = float(davies_bouldin_score(x_metric, labels)) if x_metric is not None else None
        except Exception:
            db = None
        print(f"  Metric evaluation: {perf_counter() - metric_eval_start:.1f}s")

        print(f"  Silhouette (cosine): {sil}")
        print(f"  Davies-Bouldin:      {db}")

        feature_names = vectorizer.get_feature_names_out()
        cluster_terms: dict[int, str] = {}
        preview_cluster_ids = range(min(3, k))
        for cluster_id in preview_cluster_ids:
            member_idx = np.flatnonzero(labels == cluster_id)
            if member_idx.size == 0:
                cluster_terms[cluster_id] = "<empty>"
                continue
            mean_vector = np.asarray(x_sparse_csr[member_idx].mean(axis=0)).ravel()
            top_idx = mean_vector.argsort()[::-1][:TOP_TERMS_PER_CLUSTER]
            cluster_terms[cluster_id] = ", ".join(feature_names[top_idx])

        print("\n  Top terms (first 3 clusters):")
        for cluster_id in preview_cluster_ids:
            print(f"    Cluster {cluster_id}: {cluster_terms[cluster_id]}")

        all_ablation.append(
            {
                "topic_group": topic_name,
                "n_articles": len(sub),
                "k": k,
                "requested_k": requested_k,
                "method": "tfidf_kmeans",
                "silhouette": sil,
                "davies_bouldin": db,
                "num_clusters": k,
            }
        )
        topic_elapsed = perf_counter() - topic_start
        tfidf_progress.set_postfix_str(f"last={topic_name} {topic_elapsed:.1f}s")
        print(f"  Topic total: {topic_elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("ABLATION B: SBERT + KMeans (same K)")
    print("=" * 60)
    sbert_progress = tqdm(eligible_topics, desc="SBERT topics", unit="topic")
    for topic_idx, topic in enumerate(sbert_progress, start=1):
        topic_name = str(topic)
        safe_topic = topic_name.replace("/", "_").replace(" ", "_")
        emb_path = EMB_DIR / f"{safe_topic}_embeddings.npy"
        if not emb_path.exists():
            print(f"\n--- Topic: {topic_name} - embeddings not found, skipping ---")
            continue

        topic_start = perf_counter()
        print(f"\n--- Topic {topic_idx}/{len(eligible_topics)}: {topic_name} ---")

        load_emb_start = perf_counter()
        embeddings = np.asarray(np.load(emb_path), dtype=np.float32)
        requested_k = hdbscan_k_per_topic.get(topic_name, 10)
        k = effective_k(requested_k, len(embeddings), args.max_k)
        print(
            f"  Embeddings: {embeddings.shape} | K={k}"
            + (f" (requested {requested_k})" if k != requested_k else "")
            + f" ({perf_counter() - load_emb_start:.1f}s)"
        )

        cluster_start = perf_counter()
        km = build_sbert_kmeans(len(embeddings), k)
        print(f"  SBERT clustering started with {km.__class__.__name__}...")
        labels = km.fit_predict(embeddings)
        print(f"  SBERT clustering: {perf_counter() - cluster_start:.1f}s ({km.__class__.__name__})")

        metric_eval_start = perf_counter()
        sil = sampled_silhouette(embeddings, labels, metric="cosine")
        try:
            db = float(davies_bouldin_score(embeddings, labels))
        except Exception:
            db = None
        print(f"  Metric evaluation: {perf_counter() - metric_eval_start:.1f}s")

        print(f"  Silhouette (cosine): {sil}")
        print(f"  Davies-Bouldin:      {db}")

        all_ablation.append(
            {
                "topic_group": topic_name,
                "n_articles": len(embeddings),
                "k": k,
                "requested_k": requested_k,
                "method": "sbert_kmeans",
                "silhouette": sil,
                "davies_bouldin": db,
                "num_clusters": k,
            }
        )
        topic_elapsed = perf_counter() - topic_start
        sbert_progress.set_postfix_str(f"last={topic_name} {topic_elapsed:.1f}s")
        print(f"  Topic total: {topic_elapsed:.1f}s")

    if HDBSCAN_CONFIG_RESULTS.exists():
        hdb_df = pd.read_parquet(HDBSCAN_CONFIG_RESULTS)
        best_hdb = (
            hdb_df.sort_values("selection_score", ascending=False)
            .groupby("topic_group")
            .first()
            .reset_index()
        )
        for _, row in best_hdb.iterrows():
            all_ablation.append(
                {
                    "topic_group": row["topic_group"],
                    "n_articles": int(row["topic_size"]),
                    "k": None,
                    "requested_k": int(row["num_clusters"]),
                    "method": "sbert_hdbscan",
                    "silhouette": row.get("silhouette"),
                    "davies_bouldin": None,
                    "num_clusters": int(row["num_clusters"]),
                }
            )

    ablation_df = pd.DataFrame(all_ablation)
    ablation_df.to_parquet(TFIDF_ABLATION_REPORT, index=False, compression="zstd")
    print(f"Ablation report saved to: {TFIDF_ABLATION_REPORT}")

    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(ablation_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
