"""
Embedding Generation, FAISS Indexing & HDBSCAN Clustering
=======================================================================
Reads the cleaned CSV, generates Sentence-BERT embeddings
(with caching), indexes with FAISS, reduces with UMAP, and clusters
with HDBSCAN. Saves all artefacts to data/ subdirectories.

Outputs:
    data/embeddings/{topic}_embeddings.npy     per-topic raw embeddings
    data/umap/{topic}_umap15d.npy              15-D UMAP (for clustering)
    data/umap/{topic}_umap2d.npy               2-D UMAP (for dashboard scatter)
    data/faiss/{topic}.faiss                   FAISS index per topic
    data/clusters/clustered_data.csv           main labelled dataset
    data/clusters/hdbscan_config_results.csv   config sweep results
"""

import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import faiss
import hdbscan
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score

from src.config import (
    HDBSCAN_DEFAULT_CONFIGS,
    HDBSCAN_TOPIC_CONFIGS,
    MAX_ROWS,
    MIN_TOPIC_SIZE,
    SBERT_MODEL,
    TEXT_COLUMN,
)
from src.io_utils import load_clean_csv
from src.paths import CLUSTER_DIR, EMB_DIR, FAISS_DIR, INPUT_CSV, UMAP_DIR
from src.topic_mapping import normalize_topic


def load_input_data() -> pd.DataFrame:
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Cleaned CSV not found at {INPUT_CSV}.\n"
            "Run DataCuration.py first."
        )

    df = load_clean_csv(nrows=MAX_ROWS)
    print("Shape initial:", df.shape)
    print("Columns:", df.columns.tolist())

    required = ["title", "document", "short_document", "topics"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")

    df["topic_group"] = df["topics"].apply(normalize_topic)
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
    return topic_name.replace("/", "_").replace(" ", "_")


def encode_documents(model: SentenceTransformer, documents: list[str]) -> np.ndarray:
    return model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def load_or_create_embeddings(
    model: SentenceTransformer,
    documents: list[str],
    topic_name: str,
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

    embeddings = encode_documents(model, documents)
    np.save(emb_path, embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray, topic_name: str) -> faiss.Index:
    print("Building FAISS index...")
    embeddings = np.asarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"Vectors indexed: {index.ntotal}")

    faiss_path = FAISS_DIR / f"{safe_topic_name(topic_name)}.faiss"
    faiss.write_index(index, str(faiss_path))
    print(f"FAISS index saved to: {faiss_path}")
    return index


def print_neighbor_examples(index: faiss.Index, embeddings: np.ndarray, df_topic: pd.DataFrame) -> None:
    distances, indices = index.search(np.asarray(embeddings[:3], dtype=np.float32), 5)
    print("\nNearest neighbor examples:")
    for i in range(min(3, len(df_topic))):
        print(f"\n  Document {i}: {str(df_topic.iloc[i]['title'])[:80]}")
        for rank, idx in enumerate(indices[i]):
            print(
                f"    Neighbor {rank}: idx={idx}, score={distances[i][rank]:.4f}, "
                f"title={str(df_topic.iloc[idx]['title'])[:60]}"
            )


def fit_umap(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=30,
        n_components=n_components,
        metric="cosine",
        random_state=42,
    )
    return np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)


def load_or_create_umap(
    embeddings: np.ndarray,
    topic_name: str,
    n_components: int,
) -> np.ndarray:
    suffix = "umap15d" if n_components == 15 else "umap2d"
    path = UMAP_DIR / f"{safe_topic_name(topic_name)}_{suffix}.npy"
    label = f"UMAP {n_components}-D"

    if path.exists():
        print(f"Loading cached {label} from {path}...")
        reduced = np.load(path).astype(np.float32)
        if reduced.shape[0] == len(embeddings):
            print(f"{label} shape: {reduced.shape}")
            return reduced
        print("  Cache mismatch, regenerating...")
    else:
        if n_components == 15:
            print(f"\nReducing dimensions with {label} for clustering...")
        else:
            print(f"Reducing dimensions with {label} for visualization...")

    reduced = fit_umap(embeddings, n_components=n_components)
    np.save(path, reduced)
    print(f"{label} shape: {reduced.shape}")
    return reduced


def compute_clustering_metrics(labels: np.ndarray, embeddings_reduced: np.ndarray) -> tuple[int, int, float, int, float | None]:
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = int((labels == -1).sum())
    noise_percent = 100 * num_noise / len(labels)

    real_cluster_counts = pd.Series(labels[labels != -1]).value_counts()
    largest_real = int(real_cluster_counts.iloc[0]) if not real_cluster_counts.empty else 0

    sil_score = None
    mask = labels != -1
    if mask.sum() > 1 and len(set(labels[mask])) > 1:
        try:
            sil_score = silhouette_score(
                embeddings_reduced[mask],
                labels[mask],
                metric="euclidean",
            )
        except Exception:
            sil_score = None

    return num_clusters, num_noise, noise_percent, largest_real, sil_score


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
) -> tuple[dict, list[dict]]:
    configs = HDBSCAN_TOPIC_CONFIGS.get(topic_name, HDBSCAN_DEFAULT_CONFIGS)
    best_result: dict | None = None
    topic_results: list[dict] = []

    print("\nTesting HDBSCAN configs...")
    for cfg in configs:
        print(f"\n  Config: {cfg}")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg["min_cluster_size"],
            min_samples=cfg["min_samples"],
            metric="euclidean",
            prediction_data=True,
        )
        labels = clusterer.fit_predict(embeddings_reduced)

        num_clusters, num_noise, noise_percent, largest_real, sil_score = compute_clustering_metrics(
            labels,
            embeddings_reduced,
        )

        print(
            f"    Clusters: {num_clusters}, Noise: {num_noise} ({noise_percent:.1f}%), "
            f"Largest: {largest_real}, Silhouette: {sil_score}"
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
            "selection_score": selection_score,
        }
        topic_results.append(result)

        if best_result is None or result["selection_score"] > best_result["selection_score"]:
            best_result = result

    if best_result is None:
        raise RuntimeError(f"No HDBSCAN result produced for topic '{topic_name}'")

    return best_result, topic_results


def to_results_rows(topic_name: str, topic_size: int, topic_results: list[dict]) -> list[dict]:
    rows = []
    for result in topic_results:
        cfg = result["cfg"]
        rows.append({
            "topic_group": topic_name,
            "topic_size": topic_size,
            "min_cluster_size": cfg["min_cluster_size"],
            "min_samples": cfg["min_samples"],
            "num_clusters": result["num_clusters"],
            "num_noise": result["num_noise"],
            "noise_percent": result["noise_percent"],
            "largest_real_cluster": result["largest_real_cluster"],
            "silhouette": result["silhouette"],
            "selection_score": result["selection_score"],
        })
    return rows


def apply_best_labels(
    df_topic: pd.DataFrame,
    best_result: dict,
    embeddings_2d: np.ndarray,
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

    local_labels = best_result["labels"]
    df_topic["cluster"] = [
        -1 if label == -1 else label + global_cluster_offset
        for label in local_labels
    ]

    embeddings_2d = np.asarray(embeddings_2d, dtype=np.float32)
    df_topic["umap_x"] = embeddings_2d[:, 0]
    df_topic["umap_y"] = embeddings_2d[:, 1]

    real_local_clusters = len(set(local_labels)) - (1 if -1 in local_labels else 0)
    return df_topic, global_cluster_offset + max(real_local_clusters, 0)


def process_topic(
    df: pd.DataFrame,
    topic_name: str,
    model: SentenceTransformer,
    global_cluster_offset: int,
) -> tuple[pd.DataFrame, list[dict], int]:
    print("\n" + "=" * 60)
    print(f"TOPIC_GROUP: {topic_name}")
    print("=" * 60)

    df_topic = df[df["topic_group"] == topic_name].copy().reset_index(drop=True)
    print(f"Articles in topic: {len(df_topic)}")

    documents = df_topic[TEXT_COLUMN].tolist()
    embeddings = load_or_create_embeddings(model, documents, topic_name)
    index = build_faiss_index(embeddings, topic_name)
    print_neighbor_examples(index, embeddings, df_topic)

    embeddings_reduced = load_or_create_umap(embeddings, topic_name, n_components=15)
    embeddings_2d = load_or_create_umap(embeddings, topic_name, n_components=2)

    best_result, topic_results = evaluate_topic_configs(
        topic_name=topic_name,
        topic_size=len(df_topic),
        embeddings_reduced=embeddings_reduced,
    )
    labelled_topic, next_offset = apply_best_labels(
        df_topic=df_topic,
        best_result=best_result,
        embeddings_2d=embeddings_2d,
        global_cluster_offset=global_cluster_offset,
    )
    return labelled_topic, to_results_rows(topic_name, len(df_topic), topic_results), next_offset


def build_small_topics_noise(df: pd.DataFrame, eligible_topics: list[str]) -> pd.DataFrame:
    small_topics_df = df[~df["topic_group"].isin(eligible_topics)].copy()
    if small_topics_df.empty:
        return small_topics_df

    small_topics_df["best_min_cluster_size"] = np.nan
    small_topics_df["best_min_samples"] = np.nan
    small_topics_df["cluster"] = -1
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


def save_outputs(final_df: pd.DataFrame, all_results: list[dict]) -> None:
    results_df = pd.DataFrame(all_results)
    results_path = CLUSTER_DIR / "hdbscan_config_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nConfig results saved to: {results_path}")

    output_path = CLUSTER_DIR / "clustered_data.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Clustered data saved to: {output_path}")
    print(f"Final shape: {final_df.shape}")


def main() -> None:
    df = load_input_data()
    eligible_topics = get_eligible_topics(df)

    print(f"\nText column for embeddings: {TEXT_COLUMN}")
    print(f"Loading SentenceTransformer model: {SBERT_MODEL}...")
    model = SentenceTransformer(SBERT_MODEL)

    all_results: list[dict] = []
    final_chunks: list[pd.DataFrame] = []
    global_cluster_offset = 0

    for topic_name in eligible_topics:
        labelled_topic, topic_results, global_cluster_offset = process_topic(
            df=df,
            topic_name=topic_name,
            model=model,
            global_cluster_offset=global_cluster_offset,
        )
        final_chunks.append(labelled_topic)
        all_results.extend(topic_results)

    small_topics_df = build_small_topics_noise(df, eligible_topics)
    if not small_topics_df.empty:
        final_chunks.append(small_topics_df)

    final_df = finalize_results(final_chunks)
    print_global_summary(final_df)
    save_outputs(final_df, all_results)


if __name__ == "__main__":
    main()
