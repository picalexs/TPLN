"""
Evaluation Report
=============================
Loads pre-computed artefacts (clustered CSV, embeddings, config results,
temporal stats) and produces a consolidated evaluation report.

Metrics:
  - Silhouette scores per topic x config (from HDBSCAN config sweep)
  - Intra-cluster cosine similarity (from .npy embeddings)
  - Burst-score vs cluster-size analysis (from temporal stats)

Outputs:
    data/evaluation_report.csv

Run AFTER:
    python DataCuration.py
    python EmbeddingsClustering.py
    python TemporalAnalysis.py
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")


# ===========================================================================
# 1. SILHOUETTE SCORES FROM CONFIG SWEEP
# ===========================================================================
print("=" * 60)
print("1. SILHOUETTE SCORES PER TOPIC x CONFIG")
print("=" * 60)

config_path = os.path.join(data_dir, "clusters", "hdbscan_config_results.csv")
if os.path.exists(config_path):
    config_df = pd.read_csv(config_path)
    print(f"Config file: {config_path}")
    print(config_df[["topic_group", "min_cluster_size", "min_samples",
                      "num_clusters", "noise_percent", "silhouette",
                      "selection_score"]].to_string(index=False))

    best_per_topic = (
        config_df.sort_values("selection_score", ascending=False)
        .groupby("topic_group")
        .first()
        .reset_index()
    )
    print("\nBest config per topic:")
    print(best_per_topic[["topic_group", "min_cluster_size", "min_samples",
                           "num_clusters", "silhouette"]].to_string(index=False))
else:
    print(f"Config file not found at {config_path}")
    config_df = pd.DataFrame()
    best_per_topic = pd.DataFrame()


# ===========================================================================
# 2. INTRA-CLUSTER COSINE SIMILARITY
# ===========================================================================
print("\n" + "=" * 60)
print("2. INTRA-CLUSTER COSINE SIMILARITY")
print("=" * 60)

cluster_csv = os.path.join(data_dir, "clusters", "clustered_data.csv")
cosine_records = []

if os.path.exists(cluster_csv):
    clustered_df = pd.read_csv(cluster_csv, low_memory=False)

    emb_dir = os.path.join(data_dir, "embeddings")
    emb_files = glob.glob(os.path.join(emb_dir, "*_embeddings.npy"))

    for emb_file in sorted(emb_files):
        # e.g. politica_embeddings.npy
        fname = os.path.basename(emb_file)
        topic_name = fname.replace("_embeddings.npy", "")

        print(f"\nTopic: {topic_name}")

        embeddings = np.load(emb_file)

        if "topic_group" not in clustered_df.columns:
            print("  Column 'topic_group' not found in clustered CSV.")
            continue

        topic_rows = clustered_df[
            clustered_df["topic_group"] == topic_name
        ].copy().reset_index(drop=True)

        if len(topic_rows) == 0:
            print("  No rows for this topic.")
            continue

        if len(embeddings) != len(topic_rows):
            usable_len = min(len(embeddings), len(topic_rows))
            print(
                "  WARNING: Embedding row count does not match topic rows "
                f"({len(embeddings)} != {len(topic_rows)}). "
                f"Using first {usable_len} aligned rows."
            )
            embeddings = embeddings[:usable_len]
            topic_rows = topic_rows.iloc[:usable_len].reset_index(drop=True)

        real_topic_rows = topic_rows[topic_rows["cluster"] != -1]
        if len(real_topic_rows) == 0:
            print("  No rows with real clusters.")
            continue

        for cluster_id in sorted(topic_rows["cluster"].unique()):
            idx = np.flatnonzero(topic_rows["cluster"].to_numpy() == cluster_id)
            if len(idx) < 2 or cluster_id == -1:
                continue

            cluster_emb = embeddings[idx]
            sim_matrix = cosine_similarity(cluster_emb)
            np.fill_diagonal(sim_matrix, np.nan)
            mean_sim = float(np.nanmean(sim_matrix))

            cosine_records.append({
                "topic_group": topic_name,
                "cluster": cluster_id,
                "cluster_size": len(idx),
                "mean_intra_cosine": round(mean_sim, 4),
            })

        # Print summary for this topic
        topic_cosines = [r for r in cosine_records if r["topic_group"] == topic_name]
        if topic_cosines:
            weights = np.array([r["cluster_size"] for r in topic_cosines], dtype=float)
            values = np.array([r["mean_intra_cosine"] for r in topic_cosines], dtype=float)
            mean_all = np.mean(values)
            weighted_all = float(np.average(values, weights=weights)) if weights.sum() > 0 else mean_all
            print(
                f"  Clusters: {len(topic_cosines)}, "
                f"Mean cosine: {mean_all:.4f}, "
                f"Weighted cosine: {weighted_all:.4f}"
            )
else:
    print(f"Clustered CSV not found at {cluster_csv}")

cosine_df = pd.DataFrame(cosine_records)
if not cosine_df.empty:
    print(f"\nGlobal mean intra-cluster cosine: {cosine_df['mean_intra_cosine'].mean():.4f}")


# ===========================================================================
# 3. BURST SCORE VS CLUSTER SIZE
# ===========================================================================
print("\n" + "=" * 60)
print("3. BURST SCORE VS CLUSTER SIZE")
print("=" * 60)

temporal_path = os.path.join(data_dir, "temporal", "cluster_temporal_stats.csv")

if os.path.exists(temporal_path):
    temporal_df = pd.read_csv(temporal_path)

    # Merge cosine data
    merged = temporal_df.copy()
    if not cosine_df.empty:
        merged = merged.merge(
            cosine_df[["topic_group", "cluster", "mean_intra_cosine"]],
            on=["topic_group", "cluster"],
            how="left",
        )

    suspicious = merged[merged["burst_score"] > 0].sort_values(
        "suspicion_score", ascending=False
    )

    print(f"Clusters with burst detected: {len(suspicious)}")
    print("\nTop 15 suspicious clusters:")
    cols = ["cluster", "topic_group", "article_count", "burst_score",
            "burst_duration_days", "span_days", "suspicion_score"]
    if "mean_intra_cosine" in suspicious.columns:
        cols.append("mean_intra_cosine")
    print(suspicious[cols].head(15).to_string(index=False))

    # Correlation
    corr = temporal_df[["burst_score", "article_count", "span_days",
                         "temporal_spread_days"]].corr()
    print("\nCorrelation matrix:")
    print(corr.to_string())
else:
    print(f"Temporal stats not found at {temporal_path}")
    merged = pd.DataFrame()


# ===========================================================================
# 4. SAVE EVALUATION REPORT
# ===========================================================================
report_rows = []

# Config summary rows
if not config_df.empty:
    for _, row in best_per_topic.iterrows():
        report_rows.append({
            "section": "silhouette",
            "topic_group": row.get("topic_group"),
            "cluster": None,
            "method": "sbert_hdbscan",
            "silhouette": row.get("silhouette"),
            "noise_percent": row.get("noise_percent"),
            "num_clusters": row.get("num_clusters"),
            "mean_intra_cosine": None,
            "burst_score": None,
            "suspicion_score": None,
        })

# Cosine per cluster rows
for _, row in cosine_df.iterrows():
    report_rows.append({
        "section": "intra_cosine",
        "topic_group": row.get("topic_group"),
        "cluster": row.get("cluster"),
        "method": "sbert_hdbscan",
        "silhouette": None,
        "noise_percent": None,
        "num_clusters": None,
        "mean_intra_cosine": row.get("mean_intra_cosine"),
        "burst_score": None,
        "suspicion_score": None,
    })

# Burst per cluster rows
if os.path.exists(temporal_path) and not merged.empty:
    for _, row in merged.iterrows():
        report_rows.append({
            "section": "burst",
            "topic_group": row.get("topic_group"),
            "cluster": row.get("cluster"),
            "method": "sbert_hdbscan",
            "silhouette": None,
            "noise_percent": None,
            "num_clusters": None,
            "mean_intra_cosine": row.get("mean_intra_cosine"),
            "burst_score": row.get("burst_score"),
            "suspicion_score": row.get("suspicion_score"),
        })

report_df = pd.DataFrame(report_rows)
report_path = os.path.join(data_dir, "evaluation_report.csv")
report_df.to_csv(report_path, index=False)
print(f"\nEvaluation report saved to: {report_path}")
print(f"Total rows: {len(report_df)}")
