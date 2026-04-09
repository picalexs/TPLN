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

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss
import hdbscan
import umap
from sklearn.metrics import silhouette_score

from src.config import (
    SBERT_MODEL, TEXT_COLUMN, MIN_TOPIC_SIZE,
    MAX_ROWS, HDBSCAN_DEFAULT_CONFIGS, HDBSCAN_TOPIC_CONFIGS
)
from src.paths import (
    INPUT_CSV, EMB_DIR, UMAP_DIR, FAISS_DIR, CLUSTER_DIR
)
from src.io_utils import load_clean_csv
from src.topic_mapping import normalize_topic




# =========================================================================
# LOAD DATA (assumes already cleaned by DataCuration.py)
# =========================================================================
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"Cleaned CSV not found at {INPUT_CSV}.\n"
        "Run DataCuration.py first."
    )

df = load_clean_csv(nrows=MAX_ROWS)

print("Shape initial:", df.shape)
print("Columns:", df.columns.tolist())

# No re-cleaning; data is already clean from DataCuration.py
# Just ensure required columns exist
for col in ["title", "document", "short_document", "topics"]:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in DataFrame")


# =========================================================================
# NORMALIZE TOPICS
# =========================================================================
df["topic_group"] = df["topics"].apply(normalize_topic)

print("\nDistribution of topic_group:")
print(df["topic_group"].value_counts().head(20))


# =========================================================================
# LOAD SBERT MODEL
# =========================================================================
print(f"\nText column for embeddings: {TEXT_COLUMN}")
print(f"Loading SentenceTransformer model: {SBERT_MODEL}...")

model = SentenceTransformer(SBERT_MODEL)


# =========================================================================
# HDBSCAN CONFIG SWEEP & CLUSTERING
# =========================================================================
all_results = []
final_chunks = []
global_cluster_offset = 0

topic_counts = df["topic_group"].value_counts()
eligible_topics = topic_counts[topic_counts >= MIN_TOPIC_SIZE].index.tolist()

print("\nEligible topics for clustering:")
print(topic_counts[topic_counts >= MIN_TOPIC_SIZE])
for topic_name in eligible_topics:
    safe_topic = topic_name.replace("/", "_").replace(" ", "_")

    print("\n" + "=" * 60)
    print(f"TOPIC_GROUP: {topic_name}")
    print("=" * 60)

    df_topic = df[df["topic_group"] == topic_name].copy().reset_index(drop=True)
    print(f"Articles in topic: {len(df_topic)}")

    documents = df_topic[TEXT_COLUMN].tolist()

    # ---- EMBEDDINGS (with caching) ----
    emb_path = EMB_DIR / f"{safe_topic}_embeddings.npy"

    if emb_path.exists():
        print(f"Loading cached embeddings from {emb_path}...")
        embeddings = np.load(emb_path).astype(np.float32)
        if embeddings.shape[0] != len(df_topic):
            print(f"  Cache mismatch ({embeddings.shape[0]} vs {len(df_topic)}), regenerating...")
            embeddings = model.encode(
                documents, batch_size=32, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=True
            ).astype(np.float32)
            np.save(emb_path, embeddings)
    else:
        print("Generating embeddings...")
        embeddings = model.encode(
            documents, batch_size=32, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        np.save(emb_path, embeddings)

    print(f"Embeddings shape: {embeddings.shape}")

    # ---- FAISS INDEX (save to disk) ----
    print("Building FAISS index...")
    embeddings = np.asarray(embeddings, dtype=np.float32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"Vectors indexed: {index.ntotal}")

    faiss_path = FAISS_DIR / f"{safe_topic}.faiss"
    faiss.write_index(index, str(faiss_path))
    print(f"FAISS index saved to: {faiss_path}")

    # Quick nearest-neighbor check
    k = 5
    distances, indices = index.search(np.asarray(embeddings[:3], dtype=np.float32), k)
    print("\nNearest neighbor examples:")
    for i in range(min(3, len(df_topic))):
        print(f"\n  Document {i}: {str(df_topic.iloc[i]['title'])[:80]}")
        for rank, idx in enumerate(indices[i]):
            print(
                f"    Neighbor {rank}: idx={idx}, score={distances[i][rank]:.4f}, "
                f"title={str(df_topic.iloc[idx]['title'])[:60]}"
            )

    # ---- UMAP 15-D (for clustering) - with caching ----
    umap15d_path = UMAP_DIR / f"{safe_topic}_umap15d.npy"

    if umap15d_path.exists():
        print(f"\nLoading cached UMAP 15-D from {umap15d_path}...")
        embeddings_reduced = np.load(umap15d_path).astype(np.float32)
        if embeddings_reduced.shape[0] != len(df_topic):
            print(f"  Cache mismatch, regenerating...")
            reducer = umap.UMAP(n_neighbors=30, n_components=15, metric="cosine", random_state=42)
            embeddings_reduced = np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)
            np.save(umap15d_path, embeddings_reduced)
    else:
        print("\nReducing dimensions with UMAP (15-D for clustering)...")
        reducer = umap.UMAP(n_neighbors=30, n_components=15, metric="cosine", random_state=42)
        embeddings_reduced = np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)
        np.save(umap15d_path, embeddings_reduced)

    print(f"UMAP 15-D shape: {embeddings_reduced.shape}")

    # ---- UMAP 2-D (for dashboard scatter) - with caching ----
    umap2d_path = UMAP_DIR / f"{safe_topic}_umap2d.npy"

    if umap2d_path.exists():
        print(f"Loading cached UMAP 2-D from {umap2d_path}...")
        embeddings_2d = np.load(umap2d_path).astype(np.float32)
        if embeddings_2d.shape[0] != len(df_topic):
            print(f"  Cache mismatch, regenerating...")
            reducer_2d = umap.UMAP(n_neighbors=30, n_components=2, metric="cosine", random_state=42)
            embeddings_2d = np.asarray(reducer_2d.fit_transform(embeddings), dtype=np.float32)
            np.save(umap2d_path, embeddings_2d)
    else:
        print("Reducing dimensions with UMAP (2-D for visualization)...")
        reducer_2d = umap.UMAP(n_neighbors=30, n_components=2, metric="cosine", random_state=42)
        embeddings_2d = np.asarray(reducer_2d.fit_transform(embeddings), dtype=np.float32)
        np.save(umap2d_path, embeddings_2d)

    print(f"UMAP 2-D shape: {embeddings_2d.shape}")

    # ---- HDBSCAN CONFIG SWEEP ----
    topic_specific_configs = HDBSCAN_TOPIC_CONFIGS.get(topic_name, HDBSCAN_DEFAULT_CONFIGS)

    best_result = None

    print("\nTesting HDBSCAN configs...")
    for cfg in topic_specific_configs:
        print(f"\n  Config: {cfg}")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg["min_cluster_size"],
            min_samples=cfg["min_samples"],
            metric="euclidean",
            prediction_data=True
        )
        labels = clusterer.fit_predict(embeddings_reduced)

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = int((labels == -1).sum())
        noise_percent = 100 * num_noise / len(labels)

        real_cluster_counts = pd.Series(labels[labels != -1]).value_counts()
        largest_real = int(real_cluster_counts.iloc[0]) if not real_cluster_counts.empty else 0

        # Silhouette on non-noise
        sil_score = None
        mask = labels != -1
        if mask.sum() > 1 and len(set(labels[mask])) > 1:
            try:
                sil_score = silhouette_score(
                    embeddings_reduced[mask], labels[mask], metric="euclidean"
                )
            except Exception:
                sil_score = None

        print(f"    Clusters: {num_clusters}, Noise: {num_noise} ({noise_percent:.1f}%), "
              f"Largest: {largest_real}, Silhouette: {sil_score}")

        penalty = largest_real / len(labels)
        sil_bonus = sil_score * 20 if sil_score is not None else 0
        cluster_bonus = min(num_clusters, 100) / 10

        if len(df_topic) < 1000:
            selection_score = (100 - noise_percent) - (penalty * 120) + (sil_bonus * 1.5) + cluster_bonus
        else:
            selection_score = (100 - noise_percent) - (penalty * 150) + sil_bonus + cluster_bonus

        if num_clusters < 3:
            selection_score -= 100

        result = {
            "cfg": cfg, "labels": labels,
            "num_clusters": num_clusters, "num_noise": num_noise,
            "noise_percent": noise_percent, "largest_real_cluster": largest_real,
            "silhouette": sil_score, "selection_score": selection_score,
        }

        all_results.append({
            "topic_group": topic_name, "topic_size": len(df_topic),
            "min_cluster_size": cfg["min_cluster_size"],
            "min_samples": cfg["min_samples"],
            "num_clusters": num_clusters, "num_noise": num_noise,
            "noise_percent": noise_percent, "largest_real_cluster": largest_real,
            "silhouette": sil_score, "selection_score": selection_score,
        })

        if best_result is None or result["selection_score"] > best_result["selection_score"]:
            best_result = result

    if best_result is not None:
        print(f"\nBest config for {topic_name}: {best_result['cfg']}")
        print(f"  Clusters: {best_result['num_clusters']}, "
              f"Noise: {best_result['num_noise']} ({best_result['noise_percent']:.1f}%), "
              f"Silhouette: {best_result['silhouette']}")

        df_topic["best_min_cluster_size"] = best_result["cfg"]["min_cluster_size"]
        df_topic["best_min_samples"] = best_result["cfg"]["min_samples"]

        local_labels = best_result["labels"]
        global_labels = []
        for lbl in local_labels:
            if lbl == -1:
                global_labels.append(-1)
            else:
                global_labels.append(lbl + global_cluster_offset)

        df_topic["cluster"] = global_labels
        embeddings_2d_arr = np.asarray(embeddings_2d, dtype=np.float32)
        df_topic["umap_x"] = embeddings_2d_arr[:, 0]
        df_topic["umap_y"] = embeddings_2d_arr[:, 1]

        real_local_clusters = len(set(local_labels)) - (1 if -1 in local_labels else 0)
        if real_local_clusters > 0:
            global_cluster_offset += real_local_clusters

        final_chunks.append(df_topic)


# Small topics go to noise
small_topics_df = df[~df["topic_group"].isin(eligible_topics)].copy()
if not small_topics_df.empty:
    small_topics_df["best_min_cluster_size"] = np.nan
    small_topics_df["best_min_samples"] = np.nan
    small_topics_df["cluster"] = -1
    small_topics_df["umap_x"] = np.nan
    small_topics_df["umap_y"] = np.nan
    final_chunks.append(small_topics_df)


# ===========================================================================
# 5. GLOBAL RESULTS
# ===========================================================================
final_df = pd.concat(final_chunks, ignore_index=True)

cluster_sizes = final_df["cluster"].value_counts().to_dict()
final_df["cluster_size"] = final_df["cluster"].map(cluster_sizes)

print("\n" + "=" * 60)
print("GLOBAL RESULTS")
print("=" * 60)

num_clusters_global = len(set(final_df["cluster"])) - (1 if -1 in final_df["cluster"].values else 0)
num_noise_global = int((final_df["cluster"] == -1).sum())
noise_pct_global = 100 * num_noise_global / len(final_df)

print(f"Global clusters:  {num_clusters_global}")
print(f"Global noise:     {num_noise_global} ({noise_pct_global:.1f}%)")
print(f"\nCluster distribution (top 20):")
print(final_df["cluster"].value_counts().head(20))

top_real = final_df[final_df["cluster"] != -1]["cluster"].value_counts().head(10)
print(f"\nTop real clusters:")
print(top_real)

cluster_counts = final_df[final_df["cluster"] != -1]["cluster"].value_counts()
if not cluster_counts.empty:
    largest = cluster_counts.idxmax()
    print(f"\nLargest cluster ({largest}) - sample titles:")
    for _, row in final_df[final_df["cluster"] == largest].head(10).iterrows():
        print(f"  - {str(row['title'])[:100]}")

print("\nSample from first 5 clusters:")
shown = 0
for cid in sorted(final_df["cluster"].unique()):
    if cid == -1:
        continue
    subset = final_df[final_df["cluster"] == cid].head(5)
    print(f"\n  === Cluster {cid} ===")
    for _, row in subset.iterrows():
        print(f"    - {str(row['title'])[:100]}")
    shown += 1
    if shown >= 5:
        break


# =========================================================================
# SAVE
# =========================================================================
results_df = pd.DataFrame(all_results)
results_path = CLUSTER_DIR / "hdbscan_config_results.csv"
results_df.to_csv(results_path, index=False)
print(f"\nConfig results saved to: {results_path}")

output_path = CLUSTER_DIR / "clustered_data.csv"
final_df.to_csv(output_path, index=False)
print(f"Clustered data saved to: {output_path}")
print(f"Final shape: {final_df.shape}")