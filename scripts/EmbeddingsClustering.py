"""
Embedding generation and semantic clustering.

Reads the cleaned parquet dataset, generates Sentence-BERT embeddings
(with per-topic caching), reduces them with a single 15-D UMAP pass,
and clusters them with HDBSCAN.

Outputs:
    data/embeddings/{topic}_embeddings.npy        per-topic raw embeddings
    data/clusters/clustered_data.parquet          main labelled dataset
    data/clusters/hdbscan_config_results.parquet  config sweep results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from time import perf_counter
from pathlib import Path
from typing import Any, cast

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import faiss
import hdbscan
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.config import (
    COLUMNS_TO_PRESERVE,
    HDBSCAN_DEFAULT_CONFIGS,
    HDBSCAN_TOPIC_CONFIGS,
    MIN_TOPIC_SIZE,
    SBERT_MODEL,
    TEXT_COLUMN,
    UMAP_N_NEIGHBORS,
)
from src.io_utils import load_clean_data
from src.paths import (
    CLEAN_PARQUET,
    CLUSTERED_PARQUET,
    EMB_DIR,
    FAISS_DIR,
    FAISS_KNN_DIR,
    HDBSCAN_CONFIG_RESULTS,
    RUNTIME_OBSERVABILITY_PARQUET,
)
from src.runtime_profile import (
    apply_runtime_profile,
    detect_runtime_profile,
    format_runtime_profile,
)
from src.topic_mapping import normalize_topic_with_reason

stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(encoding="utf-8", errors="replace")

SILHOUETTE_SAMPLE_SIZE = 5000
MIN_UMAP_RECURSION_LIMIT = 20_000
FAISS_METADATA_VERSION = 1
GRAPH_COHESION_SAMPLE_SIZE = 5000


def build_topic_metadata(topics: pd.Series) -> pd.DataFrame:
    """Normalize repeated topic labels once, then map back by value."""
    sentinel = "__MISSING_TOPIC__"
    lookup_series = topics.astype("object").where(topics.notna(), sentinel)
    unique_topics = lookup_series.drop_duplicates()
    metadata_map = {
        raw_topic: normalize_topic_with_reason(None if raw_topic == sentinel else raw_topic)
        for raw_topic in unique_topics.tolist()
    }
    normalized = lookup_series.map(metadata_map)
    return pd.DataFrame(
        normalized.tolist(),
        index=topics.index,
        columns=["topic_group", "topic_group_reason"],
    )


def load_input_data(nrows: int | None = None) -> pd.DataFrame:
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {CLEAN_PARQUET}.\n"
            "Run scripts/DataCuration.py first."
        )

    load_columns = [col for col in COLUMNS_TO_PRESERVE if col in {"title", "topics", "short_document", "document"}]
    df = load_clean_data(nrows=nrows, columns=load_columns)

    print("Shape initial:", df.shape)
    print("Columns:", df.columns.tolist())

    required = ["title", "document", "short_document", "topics"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")

    topic_meta_df = build_topic_metadata(df["topics"])
    df["topic_group"] = topic_meta_df["topic_group"]
    df["topic_group_reason"] = topic_meta_df["topic_group_reason"]
    print("\nDistribution of topic_group:")
    print(df["topic_group"].value_counts().head(20))
    return df


def get_eligible_topics(df: pd.DataFrame) -> list[str]:
    topic_counts = df["topic_group"].value_counts()
    eligible_topics = topic_counts[topic_counts >= MIN_TOPIC_SIZE].index.tolist()
    print("\nEligible topics for clustering:")
    print(topic_counts[topic_counts >= MIN_TOPIC_SIZE])
    return eligible_topics


def safe_topic_name(topic_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", topic_name).strip("_")


def encode_documents(
    model: SentenceTransformer,
    documents: list[str],
    batch_size: int,
    device_name: str,
) -> np.ndarray:
    if device_name == "dml":
        import torch

        # SentenceTransformer.encode is wrapped in inference_mode(). On DirectML,
        # this can fail inside transformer layers with a version_counter error.
        encode_impl = getattr(type(model).encode, "__wrapped__", None)
        if callable(encode_impl):
            with torch.no_grad():
                return cast(np.ndarray, encode_impl(
                    model,
                    documents,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )).astype(np.float32)

        with torch.no_grad():
            return model.encode(
                documents,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

    return model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def load_or_create_embeddings(
    model: SentenceTransformer,
    documents: list[str],
    topic_name: str,
    batch_size: int,
    device_name: str,
) -> np.ndarray:
    emb_path = EMB_DIR / f"{safe_topic_name(topic_name)}_embeddings.npy"

    if emb_path.exists():
        print(f"Loading cached embeddings from {emb_path}...")
        embeddings = np.load(emb_path).astype(np.float32)
        if embeddings.shape[0] == len(documents):
            print(f"Embeddings shape: {embeddings.shape}")
            return embeddings
        print(f"  Cache mismatch ({embeddings.shape[0]} vs {len(documents)}), regenerating...")
    else:
        print("Generating embeddings...")

    embeddings = encode_documents(
        model,
        documents,
        batch_size=batch_size,
        device_name=device_name,
    )
    np.save(emb_path, embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def resolve_sentence_transformer_device(device_name: str):
    if device_name == "dml":
        import torch_directml

        return torch_directml.device()
    return device_name


def _compute_documents_fingerprint(documents: list[str]) -> str:
    """Create a stable fingerprint so FAISS indexes can be reused safely."""
    hasher = hashlib.sha256()
    for doc in documents:
        hasher.update(str(doc).encode("utf-8", errors="ignore"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _faiss_paths(topic_name: str) -> tuple[Path, Path, Path]:
    safe_name = safe_topic_name(topic_name)
    index_path = FAISS_DIR / f"{safe_name}.faiss"
    metadata_path = FAISS_DIR / f"{safe_name}.json"
    knn_path = FAISS_KNN_DIR / f"{safe_name}_knn.parquet"
    return index_path, metadata_path, knn_path


def _build_faiss_index(
    embeddings: np.ndarray,
    *,
    index_type: str,
    ivf_nlist: int,
    ivf_nprobe: int,
) -> tuple[faiss.Index, dict[str, Any]]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    dim = int(embeddings.shape[1])
    n_vec = int(embeddings.shape[0])

    if index_type == "ivfflat":
        if n_vec < 200:
            print("  FAISS ivfflat requested, but topic is too small; falling back to flat index.")
        else:
            nlist = min(max(1, ivf_nlist), max(1, n_vec // 40))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            cast(Any, index).train(embeddings)
            cast(Any, index).add(embeddings)
            cast(Any, index).nprobe = min(max(1, ivf_nprobe), nlist)
            return index, {
                "index_type": "ivfflat",
                "ivf_nlist": int(nlist),
                "ivf_nprobe": int(cast(Any, index).nprobe),
            }

    index = faiss.IndexFlatIP(dim)
    cast(Any, index).add(embeddings)
    return index, {
        "index_type": "flat",
        "ivf_nlist": None,
        "ivf_nprobe": None,
    }


def _load_faiss_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _save_faiss_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)


def get_or_create_faiss_index(
    topic_name: str,
    embeddings: np.ndarray,
    documents: list[str],
    *,
    index_type: str,
    ivf_nlist: int,
    ivf_nprobe: int,
    rebuild: bool,
) -> tuple[faiss.Index, dict[str, Any], bool]:
    """Load a compatible persisted FAISS index or build and save a new one."""
    index_path, metadata_path, _ = _faiss_paths(topic_name)
    doc_fingerprint = _compute_documents_fingerprint(documents)
    expected_count = int(len(documents))
    expected_dim = int(embeddings.shape[1])

    if not rebuild and index_path.exists() and metadata_path.exists():
        metadata = _load_faiss_metadata(metadata_path)
        if metadata is not None:
            is_compatible = (
                int(metadata.get("schema_version", -1)) == FAISS_METADATA_VERSION
                and str(metadata.get("topic_group", "")) == str(topic_name)
                and int(metadata.get("vector_count", -1)) == expected_count
                and int(metadata.get("vector_dim", -1)) == expected_dim
                and str(metadata.get("document_fingerprint", "")) == doc_fingerprint
            )
            if is_compatible:
                try:
                    index = faiss.read_index(str(index_path))
                    if int(index.ntotal) == expected_count and int(index.d) == expected_dim:
                        if metadata.get("index_type") == "ivfflat":
                            desired_nprobe = int(metadata.get("ivf_nprobe") or ivf_nprobe)
                            try:
                                cast(Any, index).nprobe = max(1, desired_nprobe)
                            except Exception:
                                pass
                        print(f"Reusing persisted FAISS index: {index_path}")
                        return index, metadata, False
                except Exception:
                    pass

    print("Building persisted FAISS index...")
    index, faiss_info = _build_faiss_index(
        embeddings,
        index_type=index_type,
        ivf_nlist=ivf_nlist,
        ivf_nprobe=ivf_nprobe,
    )
    faiss.write_index(index, str(index_path))

    metadata = {
        "schema_version": FAISS_METADATA_VERSION,
        "topic_group": topic_name,
        "vector_count": expected_count,
        "vector_dim": expected_dim,
        "document_fingerprint": doc_fingerprint,
        **faiss_info,
    }
    _save_faiss_metadata(metadata_path, metadata)
    print(f"Persisted FAISS index: {index_path}")
    return index, metadata, True


def maybe_export_knn(
    topic_name: str,
    index: faiss.Index,
    embeddings: np.ndarray,
    *,
    k: int,
) -> None:
    if k < 1:
        return
    _, _, knn_path = _faiss_paths(topic_name)
    q = np.asarray(embeddings, dtype=np.float32)
    distances, indices = cast(Any, index).search(q, k + 1)

    records: list[dict[str, Any]] = []
    for source_idx in range(len(q)):
        rank = 0
        for neighbor_idx, score in zip(indices[source_idx], distances[source_idx]):
            if int(neighbor_idx) == int(source_idx) or int(neighbor_idx) < 0:
                continue
            rank += 1
            if rank > k:
                break
            records.append(
                {
                    "topic_group": topic_name,
                    "source_idx": int(source_idx),
                    "neighbor_idx": int(neighbor_idx),
                    "rank": int(rank),
                    "score": float(score),
                }
            )

    pd.DataFrame(records).to_parquet(knn_path, index=False)
    print(f"Persisted FAISS kNN edges: {knn_path} ({len(records):,} rows)")


def print_neighbor_examples(index: faiss.Index, embeddings: np.ndarray, df_topic: pd.DataFrame) -> None:
    distances, indices = cast(
        Any,
        index,
    ).search(np.asarray(embeddings[:3], dtype=np.float32), 5)
    print("\nNearest neighbor examples:")
    for i in range(min(3, len(df_topic))):
        print(f"\n  Document {i}: {str(df_topic.iloc[i]['title'])[:80]}")
        for rank, idx in enumerate(indices[i]):
            print(
                f"    Neighbor {rank}: idx={idx}, score={distances[i][rank]:.4f}, "
                f"title={str(df_topic.iloc[idx]['title'])[:60]}"
            )


def ensure_umap_recursion_limit() -> None:
    """Raise Python's recursion limit enough for large PyNNDescent builds."""
    current_limit = sys.getrecursionlimit()
    if current_limit < MIN_UMAP_RECURSION_LIMIT:
        sys.setrecursionlimit(MIN_UMAP_RECURSION_LIMIT)
        print(
            f"Raised Python recursion limit from {current_limit} "
            f"to {MIN_UMAP_RECURSION_LIMIT} for UMAP neighbor graph construction."
        )


def fit_umap_for_clustering(embeddings: np.ndarray, cpu_threads: int) -> np.ndarray:
    ensure_umap_recursion_limit()
    low_memory = cpu_threads < 16

    base_kwargs = {
        "n_neighbors": UMAP_N_NEIGHBORS,
        "n_components": 15,
        "metric": "cosine",
        "low_memory": low_memory,
        "n_jobs": cpu_threads,
    }

    try:
        reducer = umap.UMAP(**base_kwargs)
        return np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)
    except RecursionError:
        print("UMAP hit Python's recursion limit during NNDescent; retrying with a safer fallback.")
    except RuntimeError as exc:
        error_text = str(exc).lower()
        if "recursion" not in error_text:
            raise
        print(
            "UMAP hit a recursion-related runtime error during NNDescent; "
            "retrying with a safer fallback."
        )

    fallback_kwargs = {
        **base_kwargs,
        "init": "random",
        "angular_rp_forest": False,
    }
    reducer = umap.UMAP(**fallback_kwargs)
    return np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)


def build_scatter_projection(embeddings_reduced: np.ndarray) -> np.ndarray:
    if embeddings_reduced.shape[1] < 2:
        raise ValueError("UMAP reduction must expose at least two components for scatter plotting.")
    return np.asarray(embeddings_reduced[:, :2], dtype=np.float32)


def create_silhouette_sample_indices(topic_size: int) -> np.ndarray | None:
    if topic_size <= 1:
        return None
    if topic_size <= SILHOUETTE_SAMPLE_SIZE:
        return np.arange(topic_size, dtype=np.int64)

    rng = np.random.default_rng(42)
    return np.sort(
        rng.choice(topic_size, size=SILHOUETTE_SAMPLE_SIZE, replace=False).astype(np.int64)
    )


def compute_clustering_metrics(
    labels: np.ndarray,
    embeddings_full: np.ndarray,
    embeddings_reduced: np.ndarray,
    silhouette_sample_idx: np.ndarray | None,
) -> tuple[int, int, float, int, float | None, float | None, float | None]:
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = int((labels == -1).sum())
    noise_percent = 100 * num_noise / len(labels)

    real_cluster_counts = pd.Series(labels[labels != -1]).value_counts()
    largest_real = int(real_cluster_counts.iloc[0]) if not real_cluster_counts.empty else 0

    sil_score: float | None = None
    db_score: float | None = None
    ch_score: float | None = None
    if silhouette_sample_idx is not None:
        sample_idx = silhouette_sample_idx[labels[silhouette_sample_idx] != -1]
        if sample_idx.size > 1 and len(set(labels[sample_idx])) > 1:
            try:
                sil_score = float(
                    silhouette_score(
                        embeddings_full[sample_idx],
                        labels[sample_idx],
                        metric="cosine",
                    )
                )
            except Exception:
                sil_score = None

            try:
                db_score = float(davies_bouldin_score(embeddings_reduced[sample_idx], labels[sample_idx]))
            except Exception:
                db_score = None

            try:
                ch_score = float(calinski_harabasz_score(embeddings_reduced[sample_idx], labels[sample_idx]))
            except Exception:
                ch_score = None

    return num_clusters, num_noise, noise_percent, largest_real, sil_score, db_score, ch_score


def compute_graph_cohesion(
    labels: np.ndarray,
    embeddings_full: np.ndarray,
    *,
    k: int = 10,
) -> tuple[float | None, float | None]:
    """Estimate cluster cohesion with FAISS neighbor agreement ratios."""
    n_rows = int(len(labels))
    if n_rows < 3:
        return None, None

    sample_size = min(n_rows, GRAPH_COHESION_SAMPLE_SIZE)
    if sample_size < 3:
        return None, None

    rng = np.random.default_rng(42)
    if sample_size == n_rows:
        sample_idx = np.arange(n_rows, dtype=np.int64)
    else:
        sample_idx = np.sort(rng.choice(n_rows, size=sample_size, replace=False).astype(np.int64))

    sampled_vectors = np.asarray(embeddings_full[sample_idx], dtype=np.float32)
    sampled_labels = np.asarray(labels[sample_idx])
    if len(set(sampled_labels)) < 2:
        return None, None

    index = faiss.IndexFlatIP(sampled_vectors.shape[1])
    cast(Any, index).add(sampled_vectors)
    distances, neighbors = cast(Any, index).search(sampled_vectors, k + 1)
    _ = distances

    same_cluster_ratios: list[float] = []
    same_cluster_real_only: list[float] = []

    for i in range(sample_size):
        src_label = int(sampled_labels[i])
        neighbor_labels: list[int] = []
        for nb in neighbors[i]:
            nb = int(nb)
            if nb < 0 or nb == i:
                continue
            neighbor_labels.append(int(sampled_labels[nb]))
            if len(neighbor_labels) >= k:
                break
        if not neighbor_labels:
            continue

        agree = float(sum(1 for nb_label in neighbor_labels if nb_label == src_label)) / float(len(neighbor_labels))
        same_cluster_ratios.append(agree)
        if src_label != -1:
            same_cluster_real_only.append(agree)

    if not same_cluster_ratios:
        return None, None

    graph_cohesion = float(np.mean(same_cluster_ratios))
    graph_cohesion_real = float(np.mean(same_cluster_real_only)) if same_cluster_real_only else None
    return graph_cohesion, graph_cohesion_real


def compute_selection_score(
    num_clusters: int,
    noise_percent: float,
    largest_real: int,
    sil_score: float | None,
    topic_size: int,
) -> float:
    penalty = largest_real / topic_size
    sil_bonus = sil_score * 20 if sil_score is not None else 0
    cluster_bonus = min(num_clusters, 100) / 10

    if topic_size < 1000:
        score = (100 - noise_percent) - (penalty * 120) + (sil_bonus * 1.5) + cluster_bonus
    else:
        score = (100 - noise_percent) - (penalty * 150) + sil_bonus + cluster_bonus

    if num_clusters < 3:
        score -= 100

    return score


def evaluate_topic_configs(
    topic_name: str,
    topic_size: int,
    embeddings_reduced: np.ndarray,
    embeddings_full: np.ndarray,
    silhouette_sample_idx: np.ndarray | None,
    core_dist_n_jobs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    configs = HDBSCAN_TOPIC_CONFIGS.get(topic_name, HDBSCAN_DEFAULT_CONFIGS)
    best_result: dict[str, Any] | None = None
    topic_results: list[dict[str, Any]] = []

    print("\nTesting HDBSCAN configs...")
    for cfg in configs:
        print(f"\n  Config: {cfg}")

        clusterer_kwargs = {
            "min_cluster_size": cfg["min_cluster_size"],
            "min_samples": cfg["min_samples"],
            "metric": "euclidean",
            "prediction_data": True,
            # On Windows, joblib multiprocessing uses Win32 IPC pipes that overflow
            # (WinError 1450) when pickling large KD-tree payloads.  Force single-
            # threaded core-distance computation; the tree build itself is fast enough.
            "core_dist_n_jobs": 1 if sys.platform == "win32" else core_dist_n_jobs,
        }
        if "cluster_selection_epsilon" in cfg:
            clusterer_kwargs["cluster_selection_epsilon"] = cfg["cluster_selection_epsilon"]
        if "cluster_selection_method" in cfg:
            clusterer_kwargs["cluster_selection_method"] = cfg["cluster_selection_method"]

        clusterer = hdbscan.HDBSCAN(**clusterer_kwargs)
        labels = clusterer.fit_predict(embeddings_reduced)

        (
            num_clusters,
            num_noise,
            noise_percent,
            largest_real,
            sil_score,
            db_score,
            ch_score,
        ) = compute_clustering_metrics(
            labels=labels,
            embeddings_full=embeddings_full,
            embeddings_reduced=embeddings_reduced,
            silhouette_sample_idx=silhouette_sample_idx,
        )

        graph_cohesion, graph_cohesion_real = compute_graph_cohesion(
            labels=labels,
            embeddings_full=embeddings_full,
            k=10,
        )

        print(
            f"    Clusters: {num_clusters}, Noise: {num_noise} ({noise_percent:.1f}%), "
            f"Largest: {largest_real}, Silhouette: {sil_score}, "
            f"DB: {db_score}, CH: {ch_score}, Cohesion: {graph_cohesion}"
        )

        selection_score = compute_selection_score(
            num_clusters=num_clusters,
            noise_percent=noise_percent,
            largest_real=largest_real,
            sil_score=sil_score,
            topic_size=topic_size,
        )

        result = {
            "cfg": cfg,
            "labels": labels,
            "num_clusters": num_clusters,
            "num_noise": num_noise,
            "noise_percent": noise_percent,
            "largest_real_cluster": largest_real,
            "silhouette": sil_score,
            "davies_bouldin": db_score,
            "calinski_harabasz": ch_score,
            "graph_cohesion": graph_cohesion,
            "graph_cohesion_real": graph_cohesion_real,
            "selection_score": selection_score,
            "clusterer": clusterer,
        }
        topic_results.append(result)

        if best_result is None or result["selection_score"] > best_result["selection_score"]:
            best_result = result

    if best_result is None:
        raise RuntimeError(f"No HDBSCAN result produced for topic '{topic_name}'")

    return best_result, topic_results


def to_results_rows(topic_name: str, topic_size: int, topic_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in topic_results:
        cfg = result["cfg"]
        rows.append(
            {
                "topic_group": topic_name,
                "topic_size": topic_size,
                "min_cluster_size": cfg["min_cluster_size"],
                "min_samples": cfg["min_samples"],
                "cluster_selection_method": cfg.get("cluster_selection_method", "eom"),
                "cluster_selection_epsilon": cfg.get("cluster_selection_epsilon", 0.0),
                "num_clusters": result["num_clusters"],
                "num_noise": result["num_noise"],
                "noise_percent": result["noise_percent"],
                "largest_real_cluster": result["largest_real_cluster"],
                "silhouette": result["silhouette"],
                "davies_bouldin": result["davies_bouldin"],
                "calinski_harabasz": result["calinski_harabasz"],
                "graph_cohesion": result["graph_cohesion"],
                "graph_cohesion_real": result["graph_cohesion_real"],
                "selection_score": result["selection_score"],
            }
        )
    return rows


def apply_best_labels(
    df_topic: pd.DataFrame,
    best_result: dict[str, Any],
    scatter_projection: np.ndarray,
    global_cluster_offset: int,
) -> tuple[pd.DataFrame, int]:
    print(f"\nBest config for {df_topic['topic_group'].iloc[0]}: {best_result['cfg']}")
    print(
        f"  Clusters: {best_result['num_clusters']}, "
        f"Noise: {best_result['num_noise']} ({best_result['noise_percent']:.1f}%), "
        f"Silhouette: {best_result['silhouette']}"
    )

    df_topic = df_topic.copy()
    df_topic["best_min_cluster_size"] = best_result["cfg"]["min_cluster_size"]
    df_topic["best_min_samples"] = best_result["cfg"]["min_samples"]
    df_topic["best_cluster_selection_method"] = best_result["cfg"].get("cluster_selection_method", "eom")
    df_topic["best_cluster_selection_epsilon"] = best_result["cfg"].get("cluster_selection_epsilon", 0.0)
    df_topic["topic_is_eligible"] = True

    local_labels = best_result["labels"]
    df_topic["cluster"] = [-1 if label == -1 else label + global_cluster_offset for label in local_labels]
    clusterer = best_result["clusterer"]
    membership_strength = getattr(clusterer, "probabilities_", None)
    if membership_strength is None or len(membership_strength) != len(df_topic):
        membership_strength = np.full(len(df_topic), np.nan, dtype=np.float32)
    outlier_score = getattr(clusterer, "outlier_scores_", None)
    if outlier_score is None or len(outlier_score) != len(df_topic):
        outlier_score = np.full(len(df_topic), np.nan, dtype=np.float32)
    df_topic["cluster_membership_strength"] = np.asarray(membership_strength, dtype=np.float32)
    df_topic["cluster_outlier_score"] = np.asarray(outlier_score, dtype=np.float32)
    df_topic["cluster_assignment_reason"] = np.where(np.asarray(local_labels) == -1, "hdbscan_noise", "clustered")

    scatter_projection = np.asarray(scatter_projection, dtype=np.float32)
    df_topic["umap_x"] = scatter_projection[:, 0]
    df_topic["umap_y"] = scatter_projection[:, 1]

    real_local_clusters = len(set(local_labels)) - (1 if -1 in local_labels else 0)
    return df_topic, global_cluster_offset + max(real_local_clusters, 0)


def process_topic(
    df: pd.DataFrame,
    topic_name: str,
    model: SentenceTransformer,
    batch_size: int,
    core_dist_n_jobs: int,
    global_cluster_offset: int,
    device_name: str,
    faiss_enabled: bool,
    faiss_index_type: str,
    faiss_ivf_nlist: int,
    faiss_ivf_nprobe: int,
    faiss_rebuild: bool,
    faiss_write_knn: bool,
    faiss_knn_k: int,
    debug_neighbors: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int, dict[str, Any]]:
    topic_wall_start = perf_counter()
    print("\n" + "=" * 60)
    print(f"TOPIC_GROUP: {topic_name}")
    print("=" * 60)

    df_topic = df[df["topic_group"] == topic_name].copy().reset_index(drop=True)
    df_topic["topic_row_idx"] = np.arange(len(df_topic), dtype=np.int64)
    print(f"Articles in topic: {len(df_topic)}")

    documents = df_topic[TEXT_COLUMN].tolist()
    embed_start = perf_counter()
    embeddings = load_or_create_embeddings(
        model=model,
        documents=documents,
        topic_name=topic_name,
        batch_size=batch_size,
        device_name=device_name,
    )
    embedding_seconds = perf_counter() - embed_start

    faiss_seconds = 0.0
    if faiss_enabled:
        faiss_start = perf_counter()
        faiss_index, faiss_meta, created = get_or_create_faiss_index(
            topic_name=topic_name,
            embeddings=embeddings,
            documents=documents,
            index_type=faiss_index_type,
            ivf_nlist=faiss_ivf_nlist,
            ivf_nprobe=faiss_ivf_nprobe,
            rebuild=faiss_rebuild,
        )
        print(
            "FAISS ready: "
            f"type={faiss_meta.get('index_type')} vectors={int(faiss_index.ntotal)} "
            f"created_now={'yes' if created else 'no'}"
        )
        if faiss_write_knn:
            maybe_export_knn(
                topic_name=topic_name,
                index=faiss_index,
                embeddings=embeddings,
                k=faiss_knn_k,
            )
        if debug_neighbors:
            print_neighbor_examples(faiss_index, embeddings, df_topic)
        faiss_seconds = perf_counter() - faiss_start
    elif debug_neighbors:
        tmp_index = faiss.IndexFlatIP(embeddings.shape[1])
        cast(Any, tmp_index).add(np.asarray(embeddings, dtype=np.float32))
        print_neighbor_examples(tmp_index, embeddings, df_topic)

    umap_start = perf_counter()
    print("\nReducing dimensions with UMAP 15-D for clustering and scatter...")
    embeddings_reduced = fit_umap_for_clustering(embeddings, cpu_threads=core_dist_n_jobs)
    print(f"UMAP 15-D shape: {embeddings_reduced.shape}")
    scatter_projection = build_scatter_projection(embeddings_reduced)
    umap_seconds = perf_counter() - umap_start

    silhouette_sample_idx = create_silhouette_sample_indices(len(df_topic))
    hdbscan_start = perf_counter()
    best_result, topic_results = evaluate_topic_configs(
        topic_name=topic_name,
        topic_size=len(df_topic),
        embeddings_reduced=embeddings_reduced,
        embeddings_full=embeddings,
        silhouette_sample_idx=silhouette_sample_idx,
        core_dist_n_jobs=core_dist_n_jobs,
    )
    hdbscan_seconds = perf_counter() - hdbscan_start
    label_start = perf_counter()
    labelled_topic, next_offset = apply_best_labels(
        df_topic=df_topic,
        best_result=best_result,
        scatter_projection=scatter_projection,
        global_cluster_offset=global_cluster_offset,
    )
    label_seconds = perf_counter() - label_start

    topic_total_seconds = perf_counter() - topic_wall_start
    timing_row = {
        "topic_group": topic_name,
        "topic_size": int(len(df_topic)),
        "embedding_seconds": round(embedding_seconds, 4),
        "faiss_seconds": round(faiss_seconds, 4),
        "umap_seconds": round(umap_seconds, 4),
        "hdbscan_sweep_seconds": round(hdbscan_seconds, 4),
        "label_apply_seconds": round(label_seconds, 4),
        "topic_total_seconds": round(topic_total_seconds, 4),
        "rows_per_second": round(len(df_topic) / max(topic_total_seconds, 1e-6), 4),
    }
    return labelled_topic, to_results_rows(topic_name, len(df_topic), topic_results), next_offset, timing_row


def build_small_topics_noise(df: pd.DataFrame, eligible_topics: list[str]) -> pd.DataFrame:
    small_topics_df = df[~df["topic_group"].isin(eligible_topics)].copy()
    if small_topics_df.empty:
        return small_topics_df

    small_topics_df["best_min_cluster_size"] = np.nan
    small_topics_df["best_min_samples"] = np.nan
    small_topics_df["best_cluster_selection_method"] = np.nan
    small_topics_df["best_cluster_selection_epsilon"] = np.nan
    small_topics_df["topic_is_eligible"] = False
    small_topics_df["cluster"] = -1
    small_topics_df["cluster_membership_strength"] = np.nan
    small_topics_df["cluster_outlier_score"] = np.nan
    small_topics_df["cluster_assignment_reason"] = "small_topic_noise"
    small_topics_df["umap_x"] = np.nan
    small_topics_df["umap_y"] = np.nan
    return small_topics_df


def finalize_results(final_chunks: list[pd.DataFrame]) -> pd.DataFrame:
    final_df = pd.concat(final_chunks, ignore_index=True)
    cluster_sizes = final_df["cluster"].value_counts().to_dict()
    final_df["cluster_size"] = final_df["cluster"].map(cluster_sizes)
    return final_df


def print_global_summary(final_df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("GLOBAL RESULTS")
    print("=" * 60)

    num_clusters_global = len(set(final_df["cluster"])) - (1 if -1 in final_df["cluster"].values else 0)
    num_noise_global = int((final_df["cluster"] == -1).sum())
    noise_pct_global = 100 * num_noise_global / len(final_df)

    print(f"Global clusters:  {num_clusters_global}")
    print(f"Global noise:     {num_noise_global} ({noise_pct_global:.1f}%)")
    print("\nCluster distribution (top 20):")
    print(final_df["cluster"].value_counts().head(20))

    top_real = final_df[final_df["cluster"] != -1]["cluster"].value_counts().head(10)
    print("\nTop real clusters:")
    print(top_real)

    cluster_counts = final_df[final_df["cluster"] != -1]["cluster"].value_counts()
    if not cluster_counts.empty:
        largest = cluster_counts.idxmax()
        print(f"\nLargest cluster ({largest}) - sample titles:")
        for _, row in final_df[final_df["cluster"] == largest].head(10).iterrows():
            print(f"  - {str(row['title'])[:100]}")

    print("\nSample from first 5 clusters:")
    shown = 0
    for cluster_id in sorted(final_df["cluster"].unique()):
        if cluster_id == -1:
            continue
        subset = final_df[final_df["cluster"] == cluster_id].head(5)
        print(f"\n  === Cluster {cluster_id} ===")
        for _, row in subset.iterrows():
            print(f"    - {str(row['title'])[:100]}")
        shown += 1
        if shown >= 5:
            break


def save_outputs(
    final_df: pd.DataFrame,
    all_results: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
) -> None:
    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(HDBSCAN_CONFIG_RESULTS, index=False)
    print(f"\nConfig results saved to: {HDBSCAN_CONFIG_RESULTS}")

    final_df.to_parquet(CLUSTERED_PARQUET, index=False)
    print(f"Clustered data saved to: {CLUSTERED_PARQUET}")
    print(f"Final shape: {final_df.shape}")

    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df.to_parquet(RUNTIME_OBSERVABILITY_PARQUET, index=False)
    print(f"Runtime observability profile saved to: {RUNTIME_OBSERVABILITY_PARQUET}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SBERT embeddings and run HDBSCAN clustering on the cleaned parquet dataset."
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps", "dml"])
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--embed-batch-size", type=int, default=None)
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for quick tests")
    parser.add_argument(
        "--disable-faiss",
        action="store_true",
        default=False,
        help="Skip FAISS index build/reuse for this run.",
    )
    parser.add_argument(
        "--faiss-index-type",
        choices=["flat", "ivfflat"],
        default="ivfflat",
        help="FAISS index family to build and persist per topic.",
    )
    parser.add_argument(
        "--faiss-ivf-nlist",
        type=int,
        default=1024,
        help="Target IVF list count for ivfflat indexes.",
    )
    parser.add_argument(
        "--faiss-ivf-nprobe",
        type=int,
        default=32,
        help="Probe count for ivfflat indexes at query time.",
    )
    parser.add_argument(
        "--faiss-rebuild",
        action="store_true",
        default=False,
        help="Force rebuilding persisted FAISS indexes even when metadata matches.",
    )
    parser.add_argument(
        "--faiss-write-knn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist top-k nearest-neighbor edges per topic in parquet format.",
    )
    parser.add_argument(
        "--faiss-knn-k",
        type=int,
        default=10,
        help="Neighbor count per row when --faiss-write-knn is enabled.",
    )
    parser.add_argument(
        "--debug-neighbors",
        action="store_true",
        default=False,
        help="Build FAISS index and print nearest-neighbor examples (high RAM; off by default)",
    )
    return parser.parse_args()


def main() -> int:
    wall_start = perf_counter()
    args = parse_args()
    profile = detect_runtime_profile(
        device=args.device,
        cpu_threads=args.cpu_threads,
        embed_batch_size=args.embed_batch_size,
    )
    apply_runtime_profile(profile)
    if hasattr(faiss, "omp_set_num_threads"):
        try:
            faiss.omp_set_num_threads(profile.cpu_threads)
        except Exception:
            pass
    print(format_runtime_profile(profile))

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_KNN_DIR.mkdir(parents=True, exist_ok=True)

    df = load_input_data(nrows=args.nrows)
    eligible_topics = get_eligible_topics(df)

    print(f"\nText column for embeddings: {TEXT_COLUMN}")
    print(f"Loading SentenceTransformer model: {SBERT_MODEL} on {profile.device}...")
    model_device = resolve_sentence_transformer_device(profile.device)
    model = SentenceTransformer(SBERT_MODEL, device=model_device)

    all_results: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    final_chunks: list[pd.DataFrame] = []
    global_cluster_offset = 0

    for topic_name in eligible_topics:
        labelled_topic, topic_results, global_cluster_offset, timing_row = process_topic(
            df=df,
            topic_name=topic_name,
            model=model,
            batch_size=profile.embedding_batch_size,
            core_dist_n_jobs=profile.cpu_threads,
            global_cluster_offset=global_cluster_offset,
            device_name=profile.device,
            faiss_enabled=not args.disable_faiss,
            faiss_index_type=args.faiss_index_type,
            faiss_ivf_nlist=max(1, args.faiss_ivf_nlist),
            faiss_ivf_nprobe=max(1, args.faiss_ivf_nprobe),
            faiss_rebuild=args.faiss_rebuild,
            faiss_write_knn=args.faiss_write_knn,
            faiss_knn_k=max(1, args.faiss_knn_k),
            debug_neighbors=args.debug_neighbors,
        )
        final_chunks.append(labelled_topic)
        all_results.extend(topic_results)
        runtime_rows.append(timing_row)

    small_topics_df = build_small_topics_noise(df, eligible_topics)
    if not small_topics_df.empty:
        final_chunks.append(small_topics_df)

    final_df = finalize_results(final_chunks)
    print_global_summary(final_df)
    total_seconds = perf_counter() - wall_start
    runtime_rows.append(
        {
            "topic_group": "__GLOBAL__",
            "topic_size": int(len(df)),
            "embedding_seconds": None,
            "faiss_seconds": None,
            "umap_seconds": None,
            "hdbscan_sweep_seconds": None,
            "label_apply_seconds": None,
            "topic_total_seconds": round(total_seconds, 4),
            "rows_per_second": round(len(df) / max(total_seconds, 1e-6), 4),
        }
    )
    save_outputs(final_df, all_results, runtime_rows)

    runtime_topic_df = pd.DataFrame(runtime_rows)
    runtime_topic_df = runtime_topic_df[runtime_topic_df["topic_group"] != "__GLOBAL__"].copy()
    if not runtime_topic_df.empty:
        print("\nSlowest topics by total runtime:")
        print(
            runtime_topic_df.sort_values("topic_total_seconds", ascending=False)[
                [
                    "topic_group",
                    "topic_size",
                    "embedding_seconds",
                    "faiss_seconds",
                    "umap_seconds",
                    "hdbscan_sweep_seconds",
                    "topic_total_seconds",
                ]
            ].head(10).to_string(index=False)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
