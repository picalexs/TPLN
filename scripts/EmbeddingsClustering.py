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
from collections import Counter
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
    NEAR_DUPLICATE_COSINE_THRESHOLD,
    NEAR_DUPLICATE_SEARCH_K,
    NOISE_REASSIGN_K,
    NOISE_REASSIGN_MIN_AGREEMENT,
    SBERT_MODEL,
    SHORT_DOCUMENT_MAX_CHARS,
    TEXT_COLUMN,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
    build_hdbscan_configs,
    compute_umap_n_neighbors,
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


def _documents_fingerprint(documents: list[str]) -> str:
    """Compact hash over document count plus sampled content.

    We cannot hash every row (too slow on 100k+ topics), but we want to
    detect the common "text column was regenerated" case (e.g. the
    SHORT_DOCUMENT_MAX_CHARS slice length changed). A handful of sampled
    documents combined with the total count is sufficient in practice.
    """
    if not documents:
        return hashlib.sha1(b"empty").hexdigest()
    sample_indices = [0, len(documents) // 2, len(documents) - 1]
    hasher = hashlib.sha1()
    hasher.update(str(len(documents)).encode("utf-8"))
    for idx in sample_indices:
        hasher.update(b"|")
        hasher.update(documents[idx].encode("utf-8", errors="replace"))
    return hasher.hexdigest()


def _embedding_fingerprint(documents: list[str]) -> dict[str, Any]:
    return {
        "model": SBERT_MODEL,
        "text_column": TEXT_COLUMN,
        "short_doc_chars": SHORT_DOCUMENT_MAX_CHARS,
        "count": len(documents),
        "documents_hash": _documents_fingerprint(documents),
    }


def _cache_is_fresh(meta_path: Path, expected: dict[str, Any]) -> bool:
    if not meta_path.exists():
        return False
    try:
        with meta_path.open("r", encoding="utf-8") as meta_file:
            cached = json.load(meta_file)
    except (OSError, json.JSONDecodeError):
        return False
    return all(cached.get(key) == value for key, value in expected.items())


def load_or_create_embeddings(
    model: SentenceTransformer,
    documents: list[str],
    topic_name: str,
    batch_size: int,
    device_name: str,
) -> np.ndarray:
    emb_path = EMB_DIR / f"{safe_topic_name(topic_name)}_embeddings.npy"
    meta_path = emb_path.with_suffix(".meta.json")
    fingerprint = _embedding_fingerprint(documents)

    if emb_path.exists() and _cache_is_fresh(meta_path, fingerprint):
        print(f"Loading cached embeddings from {emb_path}...")
        embeddings = np.load(emb_path).astype(np.float32)
        if embeddings.shape[0] == len(documents):
            print(f"Embeddings shape: {embeddings.shape}")
            return embeddings
        print(f"  Cache mismatch ({embeddings.shape[0]} vs {len(documents)}), regenerating...")
    elif emb_path.exists():
        print(
            f"  Cached embeddings at {emb_path} are stale (model, slice length, "
            "or source text changed); regenerating..."
        )
    else:
        print("Generating embeddings...")

    embeddings = encode_documents(
        model,
        documents,
        batch_size=batch_size,
        device_name=device_name,
    )
    np.save(emb_path, embeddings)
    with meta_path.open("w", encoding="utf-8") as meta_file:
        json.dump(fingerprint, meta_file, indent=2, sort_keys=True)
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


def _build_flat_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Return an in-memory cosine (inner-product) FAISS index.

    Used as a fallback when persistent FAISS is disabled so that the
    dedup and noise-reassignment steps always have a nearest-neighbor
    backend available.
    """
    vectors = np.asarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    cast(Any, index).add(vectors)
    return index


def detect_near_duplicates(
    embeddings: np.ndarray,
    faiss_index: faiss.Index,
    *,
    threshold: float,
    top_k: int,
) -> np.ndarray:
    """Group near-identical articles and return each row's canonical index.

    Romanian news outlets heavily syndicate wire copy, so the embedding
    space contains many near-exact duplicates that collide under UMAP
    and waste density budget in HDBSCAN. We union rows whose cosine
    similarity exceeds ``threshold`` into a single group and nominate
    the lowest-index row as the canonical.

    Returns an array ``canonical_of`` where ``canonical_of[i]`` is the
    row index of the canonical member of row ``i``'s duplicate group
    (``canonical_of[i] == i`` when ``i`` is itself canonical). Inputs
    are assumed to be L2-normalized so inner product equals cosine
    similarity, which the SBERT pipeline already guarantees.
    """
    n = int(embeddings.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    query = np.asarray(embeddings, dtype=np.float32)
    # +1 accounts for the self-match that FAISS always returns first.
    k = min(top_k + 1, n)
    sims, neighbors = cast(Any, faiss_index).search(query, k)

    canonical = np.arange(n, dtype=np.int64)
    for i in range(n):
        for nb, sim in zip(neighbors[i], sims[i]):
            nb = int(nb)
            if nb < 0 or nb == i:
                continue
            if sim >= threshold and nb > i and canonical[nb] > i:
                canonical[nb] = i

    # Path-compress so `canonical[i]` always holds the group's root in
    # a single hop, even when chains (i -> j -> k) formed above.
    for i in range(n):
        root = int(canonical[i])
        while int(canonical[root]) != root:
            root = int(canonical[root])
        canonical[i] = root

    return canonical


def reassign_noise_via_knn(
    labels: np.ndarray,
    embeddings: np.ndarray,
    faiss_index: faiss.Index,
    *,
    k: int,
    min_agreement: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Promote HDBSCAN noise points to the cluster their neighbors vote for.

    For each point currently labelled ``-1``, we look at its ``k``
    nearest already-clustered neighbors in the full SBERT embedding
    space. When a single cluster captures at least ``min_agreement`` of
    those votes, the noise point inherits that cluster's label. This
    mirrors BERTopic's ``reduce_outliers(strategy="embeddings")`` and
    HDBSCAN's soft-clustering reassignment, but uses the FAISS index
    the pipeline already builds so it stays fast on 100k-article topics.

    Returns ``(new_labels, reassigned_mask)`` where ``reassigned_mask[i]``
    is ``True`` iff row ``i`` was promoted from noise to a cluster.
    """
    labels = np.asarray(labels)
    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx) == 0:
        return labels, np.zeros(len(labels), dtype=bool)

    search_k = k + 1  # +1 for the self-match
    queries = np.asarray(embeddings[noise_idx], dtype=np.float32)
    _, neighbors = cast(Any, faiss_index).search(queries, search_k)

    new_labels = labels.copy()
    reassigned = np.zeros(len(labels), dtype=bool)
    for local_i, global_i in enumerate(noise_idx):
        neighbor_labels: list[int] = []
        for nb in neighbors[local_i]:
            nb = int(nb)
            if nb < 0 or nb == int(global_i):
                continue
            lab = int(labels[nb])
            if lab == -1:
                continue
            neighbor_labels.append(lab)
            if len(neighbor_labels) >= k:
                break
        if not neighbor_labels:
            continue
        best_label, best_count = Counter(neighbor_labels).most_common(1)[0]
        if best_count / len(neighbor_labels) >= min_agreement:
            new_labels[global_i] = best_label
            reassigned[global_i] = True

    return new_labels, reassigned


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


def fit_umap_for_clustering(
    embeddings: np.ndarray,
    cpu_threads: int,
    n_neighbors: int | None = None,
) -> np.ndarray:
    ensure_umap_recursion_limit()
    low_memory = cpu_threads < 16
    resolved_neighbors = n_neighbors if n_neighbors is not None else UMAP_N_NEIGHBORS
    # Clamp to the number of points minus one so small topics still fit.
    resolved_neighbors = max(5, min(resolved_neighbors, embeddings.shape[0] - 1))

    base_kwargs = {
        "n_neighbors": resolved_neighbors,
        "n_components": UMAP_N_COMPONENTS,
        "min_dist": UMAP_MIN_DIST,
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
    """Score an HDBSCAN config, preferring balanced cluster layouts.

    The score rewards low-noise configurations but heavily penalizes
    two degenerate failure modes:

    * Collapse: a single cluster swallows most of the topic
      (`largest_real / topic_size > 0.5`).
    * Under-segmentation: fewer than three real clusters.

    Without these guards the raw `(100 - noise_percent)` term tempts the
    selector to pick "1 giant cluster + 1% noise" over any meaningful
    partition.
    """
    largest_share = largest_real / topic_size if topic_size else 0.0
    sil_bonus = sil_score * 20 if sil_score is not None else 0
    cluster_bonus = min(num_clusters, 100) / 10

    if topic_size < 1000:
        score = (100 - noise_percent) - (largest_share * 120) + (sil_bonus * 1.5) + cluster_bonus
    else:
        score = (100 - noise_percent) - (largest_share * 150) + sil_bonus + cluster_bonus

    if num_clusters < 3:
        score -= 100

    # Hard disqualification for collapse: >50% of the topic in one cluster
    # is never a good clustering, regardless of how low the noise looks.
    if largest_share > 0.50:
        score -= 200

    return score


def evaluate_topic_configs(
    topic_name: str,
    topic_size: int,
    embeddings_reduced: np.ndarray,
    embeddings_full: np.ndarray,
    silhouette_sample_idx: np.ndarray | None,
    core_dist_n_jobs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Topic-specific overrides win when present; otherwise fall back to a
    # size-aware sweep so large topics do not get micro min_cluster_size.
    configs = HDBSCAN_TOPIC_CONFIGS.get(topic_name) or build_hdbscan_configs(topic_size)
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
    full_labels: np.ndarray,
    full_reasons: np.ndarray,
    full_membership_strength: np.ndarray,
    full_outlier_score: np.ndarray,
    full_scatter: np.ndarray,
    global_cluster_offset: int,
) -> tuple[pd.DataFrame, int]:
    """Attach cluster labels and metadata to the per-topic dataframe.

    All array inputs are pre-computed in ``process_topic`` at full-topic
    length. This function only worries about offsetting labels into the
    global cluster namespace, persisting the best-config hyperparameters,
    and returning the updated offset.
    """
    print(f"\nBest config for {df_topic['topic_group'].iloc[0]}: {best_result['cfg']}")
    print(
        f"  Canonical clusters: {best_result['num_clusters']}, "
        f"canonical noise: {best_result['num_noise']} "
        f"({best_result['noise_percent']:.1f}%), "
        f"Silhouette: {best_result['silhouette']}"
    )

    df_topic = df_topic.copy()
    df_topic["best_min_cluster_size"] = best_result["cfg"]["min_cluster_size"]
    df_topic["best_min_samples"] = best_result["cfg"]["min_samples"]
    df_topic["best_cluster_selection_method"] = best_result["cfg"].get("cluster_selection_method", "eom")
    df_topic["best_cluster_selection_epsilon"] = best_result["cfg"].get("cluster_selection_epsilon", 0.0)
    df_topic["topic_is_eligible"] = True

    offset_labels = np.where(
        full_labels == -1,
        -1,
        full_labels.astype(np.int64) + int(global_cluster_offset),
    )
    df_topic["cluster"] = offset_labels
    df_topic["cluster_membership_strength"] = np.asarray(full_membership_strength, dtype=np.float32)
    df_topic["cluster_outlier_score"] = np.asarray(full_outlier_score, dtype=np.float32)
    df_topic["cluster_assignment_reason"] = np.asarray(full_reasons, dtype=object)

    full_scatter = np.asarray(full_scatter, dtype=np.float32)
    df_topic["umap_x"] = full_scatter[:, 0]
    df_topic["umap_y"] = full_scatter[:, 1]

    canonical_labels = np.asarray(best_result["labels"])
    real_local_clusters = len(set(canonical_labels.tolist())) - (1 if -1 in canonical_labels else 0)
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

    n_total = int(len(df_topic))

    # --- FAISS: build or reuse. Always keep a handle around because the
    #     dedup and noise-reassignment steps need a nearest-neighbor
    #     backend even when persistence is disabled via --disable-faiss.
    faiss_seconds = 0.0
    faiss_index: faiss.Index | None = None
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
    else:
        print("  Building in-memory FAISS index for dedup and noise reassignment...")
        faiss_index = _build_flat_faiss_index(embeddings)
        if debug_neighbors:
            print_neighbor_examples(faiss_index, embeddings, df_topic)

    # --- Near-duplicate detection: reduce to canonical embeddings so
    #     UMAP doesn't see cosine ~1.0 point collisions from syndicated
    #     wire copy.
    dedup_start = perf_counter()
    canonical_of = detect_near_duplicates(
        embeddings,
        faiss_index,
        threshold=NEAR_DUPLICATE_COSINE_THRESHOLD,
        top_k=NEAR_DUPLICATE_SEARCH_K,
    )
    canonical_unique_idx = np.unique(canonical_of)
    n_canonical = int(len(canonical_unique_idx))
    n_duplicate = n_total - n_canonical
    dedup_seconds = perf_counter() - dedup_start
    print(
        f"\nNear-duplicates (cosine >= {NEAR_DUPLICATE_COSINE_THRESHOLD}): "
        f"{n_duplicate:,} duplicates collapsed into {n_canonical:,} canonicals "
        f"({100.0 * n_duplicate / max(n_total, 1):.1f}% dup share)"
    )

    # Reduce to canonical embeddings for UMAP and HDBSCAN.
    embeddings_canon = np.asarray(embeddings[canonical_unique_idx], dtype=np.float32)

    umap_start = perf_counter()
    umap_n_neighbors = compute_umap_n_neighbors(n_canonical)
    print(
        f"\nReducing dimensions with UMAP {UMAP_N_COMPONENTS}-D "
        f"(n_neighbors={umap_n_neighbors}, min_dist={UMAP_MIN_DIST}, "
        f"canonical rows={n_canonical:,})..."
    )
    embeddings_reduced_canon = fit_umap_for_clustering(
        embeddings_canon,
        cpu_threads=core_dist_n_jobs,
        n_neighbors=umap_n_neighbors,
    )
    print(f"UMAP {UMAP_N_COMPONENTS}-D shape: {embeddings_reduced_canon.shape}")
    scatter_canon = build_scatter_projection(embeddings_reduced_canon)
    umap_seconds = perf_counter() - umap_start

    # --- HDBSCAN sweep runs on canonicals. Metrics in topic_results
    #     therefore describe the canonical clustering; duplicates and
    #     noise-reassigned rows are book-keeping after the fact.
    silhouette_sample_idx = create_silhouette_sample_indices(n_canonical)
    hdbscan_start = perf_counter()
    best_result, topic_results = evaluate_topic_configs(
        topic_name=topic_name,
        topic_size=n_canonical,
        embeddings_reduced=embeddings_reduced_canon,
        embeddings_full=embeddings_canon,
        silhouette_sample_idx=silhouette_sample_idx,
        core_dist_n_jobs=core_dist_n_jobs,
    )
    hdbscan_seconds = perf_counter() - hdbscan_start

    # --- Propagate canonical labels / UMAP coords / HDBSCAN stats back
    #     to every row in the topic.
    canonical_labels = np.asarray(best_result["labels"], dtype=np.int64)
    canon_to_local_idx = {
        int(canonical_unique_idx[i]): i for i in range(n_canonical)
    }
    local_indices_per_row = np.asarray(
        [canon_to_local_idx[int(c)] for c in canonical_of],
        dtype=np.int64,
    )
    full_labels = canonical_labels[local_indices_per_row].copy()
    full_scatter = scatter_canon[local_indices_per_row].copy()

    clusterer = best_result["clusterer"]
    canonical_strength = getattr(clusterer, "probabilities_", None)
    canonical_outlier = getattr(clusterer, "outlier_scores_", None)
    if canonical_strength is None or len(canonical_strength) != n_canonical:
        canonical_strength = np.full(n_canonical, np.nan, dtype=np.float32)
    if canonical_outlier is None or len(canonical_outlier) != n_canonical:
        canonical_outlier = np.full(n_canonical, np.nan, dtype=np.float32)
    full_membership_strength = np.asarray(
        canonical_strength, dtype=np.float32
    )[local_indices_per_row].copy()
    full_outlier_score = np.asarray(
        canonical_outlier, dtype=np.float32
    )[local_indices_per_row].copy()

    # --- Noise reassignment via k-NN vote on the full (canonical +
    #     duplicate) label array. Duplicate rows already carry their
    #     canonical's label, so they also vote.
    reassign_start = perf_counter()
    n_noise_before = int((full_labels == -1).sum())
    reassigned_labels, reassigned_mask = reassign_noise_via_knn(
        full_labels,
        embeddings,
        faiss_index,
        k=NOISE_REASSIGN_K,
        min_agreement=NOISE_REASSIGN_MIN_AGREEMENT,
    )
    n_noise_after = int((reassigned_labels == -1).sum())
    reassign_seconds = perf_counter() - reassign_start
    print(
        f"\nNoise reassignment ({NOISE_REASSIGN_K}-NN, "
        f"min_agreement={NOISE_REASSIGN_MIN_AGREEMENT}): "
        f"{n_noise_before:,} -> {n_noise_after:,} "
        f"(recovered {int(reassigned_mask.sum()):,})"
    )
    # Rows promoted from noise get NaN membership / outlier values:
    # HDBSCAN never saw them as cluster members.
    full_membership_strength[reassigned_mask] = np.nan
    full_outlier_score[reassigned_mask] = np.nan

    # --- Final per-row cluster_assignment_reason classification.
    is_canonical_row = canonical_of == np.arange(n_total, dtype=np.int64)
    reasons = np.full(n_total, "hdbscan_noise", dtype=object)
    clustered_canonical_mask = is_canonical_row & (full_labels != -1)
    clustered_duplicate_mask = (~is_canonical_row) & (full_labels != -1)
    reasons[clustered_canonical_mask] = "clustered"
    reasons[clustered_duplicate_mask] = "duplicate_of_clustered"
    reasons[reassigned_mask] = "reassigned_from_noise"

    label_start = perf_counter()
    labelled_topic, next_offset = apply_best_labels(
        df_topic=df_topic,
        best_result=best_result,
        full_labels=reassigned_labels,
        full_reasons=reasons,
        full_membership_strength=full_membership_strength,
        full_outlier_score=full_outlier_score,
        full_scatter=full_scatter,
        global_cluster_offset=global_cluster_offset,
    )
    label_seconds = perf_counter() - label_start

    final_noise_pct = 100.0 * n_noise_after / max(n_total, 1)
    print(
        f"  Final topic composition: "
        f"{int(clustered_canonical_mask.sum()):,} canonical-clustered | "
        f"{int(clustered_duplicate_mask.sum()):,} duplicate-of-clustered | "
        f"{int(reassigned_mask.sum()):,} reassigned-from-noise | "
        f"{n_noise_after:,} final-noise ({final_noise_pct:.1f}%)"
    )

    topic_total_seconds = perf_counter() - topic_wall_start
    timing_row = {
        "topic_group": topic_name,
        "topic_size": n_total,
        "canonical_size": n_canonical,
        "duplicate_rows": n_duplicate,
        "reassigned_rows": int(reassigned_mask.sum()),
        "final_noise_rows": n_noise_after,
        "final_noise_percent": round(final_noise_pct, 4),
        "embedding_seconds": round(embedding_seconds, 4),
        "faiss_seconds": round(faiss_seconds, 4),
        "dedup_seconds": round(dedup_seconds, 4),
        "umap_seconds": round(umap_seconds, 4),
        "hdbscan_sweep_seconds": round(hdbscan_seconds, 4),
        "reassign_seconds": round(reassign_seconds, 4),
        "label_apply_seconds": round(label_seconds, 4),
        "topic_total_seconds": round(topic_total_seconds, 4),
        "rows_per_second": round(n_total / max(topic_total_seconds, 1e-6), 4),
    }
    return labelled_topic, to_results_rows(topic_name, n_canonical, topic_results), next_offset, timing_row


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
            "canonical_size": None,
            "duplicate_rows": None,
            "reassigned_rows": None,
            "final_noise_rows": None,
            "final_noise_percent": None,
            "embedding_seconds": None,
            "faiss_seconds": None,
            "dedup_seconds": None,
            "umap_seconds": None,
            "hdbscan_sweep_seconds": None,
            "reassign_seconds": None,
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
