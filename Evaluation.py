"""
Evaluation report builder.

Loads pre-computed parquet artefacts and the saved embedding cache, then
produces a consolidated evaluation report.

Output:
    data/evaluation_report.parquet
"""

from __future__ import annotations

import argparse
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.paths import CLUSTERED_PARQUET, EMB_DIR, EVALUATION_REPORT, HDBSCAN_CONFIG_RESULTS, TEMPORAL_STATS
from src.runtime_profile import apply_runtime_profile, detect_runtime_profile, format_runtime_profile

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an evaluation report from parquet pipeline artefacts.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Override CPU thread count for pairwise similarity work.",
    )
    return parser.parse_args()


def load_config_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load full config sweep plus the best config per topic."""
    print("=" * 60)
    print("1. SILHOUETTE SCORES PER TOPIC x CONFIG")
    print("=" * 60)

    if not HDBSCAN_CONFIG_RESULTS.exists():
        print(f"Config file not found at {HDBSCAN_CONFIG_RESULTS}")
        return pd.DataFrame(), pd.DataFrame()

    config_df = pd.read_parquet(HDBSCAN_CONFIG_RESULTS)
    print(f"Config file: {HDBSCAN_CONFIG_RESULTS}")
    print(
        config_df[
            [
                "topic_group",
                "min_cluster_size",
                "min_samples",
                "num_clusters",
                "noise_percent",
                "silhouette",
                "selection_score",
            ]
        ].to_string(index=False)
    )

    best_per_topic = (
        config_df.sort_values("selection_score", ascending=False)
        .groupby("topic_group")
        .first()
        .reset_index()
    )
    print("\nBest config per topic:")
    print(
        best_per_topic[
            ["topic_group", "min_cluster_size", "min_samples", "num_clusters", "silhouette"]
        ].to_string(index=False)
    )
    return config_df, best_per_topic


def compute_intra_cluster_cosine() -> pd.DataFrame:
    """Compute mean intra-cluster cosine similarity using cached embeddings."""
    print("\n" + "=" * 60)
    print("2. INTRA-CLUSTER COSINE SIMILARITY")
    print("=" * 60)

    if not CLUSTERED_PARQUET.exists():
        print(f"Clustered parquet not found at {CLUSTERED_PARQUET}")
        return pd.DataFrame()

    clustered_df = pd.read_parquet(CLUSTERED_PARQUET)
    emb_files = glob.glob(str(EMB_DIR / "*_embeddings.npy"))
    cosine_records: list[dict[str, object]] = []

    for emb_file_str in sorted(emb_files):
        emb_file = Path(emb_file_str)
        topic_name = emb_file.name.replace("_embeddings.npy", "")
        print(f"\nTopic: {topic_name}")

        embeddings = np.asarray(np.load(emb_file), dtype=np.float32)
        if "topic_group" not in clustered_df.columns:
            print("  Column 'topic_group' not found in clustered parquet.")
            continue

        topic_rows = clustered_df[clustered_df["topic_group"] == topic_name].copy().reset_index(drop=True)
        if topic_rows.empty:
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
        if real_topic_rows.empty:
            print("  No rows with real clusters.")
            continue

        cluster_values = topic_rows["cluster"].to_numpy()
        for cluster_id in sorted(topic_rows["cluster"].unique()):
            if int(cluster_id) == -1:
                continue

            idx = np.flatnonzero(cluster_values == cluster_id)
            if len(idx) < 2:
                continue

            cluster_emb = embeddings[idx]
            sim_matrix = cosine_similarity(cluster_emb)
            np.fill_diagonal(sim_matrix, np.nan)
            mean_sim = float(np.nanmean(sim_matrix))

            cosine_records.append(
                {
                    "topic_group": topic_name,
                    "cluster": int(cluster_id),
                    "cluster_size": len(idx),
                    "mean_intra_cosine": round(mean_sim, 4),
                }
            )

        topic_cosines = [record for record in cosine_records if record["topic_group"] == topic_name]
        if topic_cosines:
            weights = np.array([record["cluster_size"] for record in topic_cosines], dtype=float)
            values = np.array([record["mean_intra_cosine"] for record in topic_cosines], dtype=float)
            mean_all = float(np.mean(values))
            weighted_all = float(np.average(values, weights=weights)) if weights.sum() > 0 else mean_all
            print(
                f"  Clusters: {len(topic_cosines)}, "
                f"Mean cosine: {mean_all:.4f}, "
                f"Weighted cosine: {weighted_all:.4f}"
            )

    cosine_df = pd.DataFrame(cosine_records)
    if not cosine_df.empty:
        print(f"\nGlobal mean intra-cluster cosine: {cosine_df['mean_intra_cosine'].mean():.4f}")
    return cosine_df


def load_temporal_with_cosine(cosine_df: pd.DataFrame) -> pd.DataFrame:
    """Load temporal stats and enrich them with cosine data when available."""
    print("\n" + "=" * 60)
    print("3. BURST SCORE VS CLUSTER SIZE")
    print("=" * 60)

    if not TEMPORAL_STATS.exists():
        print(f"Temporal stats not found at {TEMPORAL_STATS}")
        return pd.DataFrame()

    temporal_df = pd.read_parquet(TEMPORAL_STATS)
    merged = temporal_df.copy()
    if not cosine_df.empty:
        merged = merged.merge(
            cosine_df[["topic_group", "cluster", "mean_intra_cosine"]],
            on=["topic_group", "cluster"],
            how="left",
        )

    suspicious = merged[merged["burst_score"] > 0].sort_values("suspicion_score", ascending=False)
    print(f"Clusters with burst detected: {len(suspicious)}")
    print("\nTop 15 suspicious clusters:")

    cols = [
        "cluster",
        "topic_group",
        "article_count",
        "burst_score",
        "burst_duration_days",
        "span_days",
        "suspicion_score",
    ]
    if "mean_intra_cosine" in suspicious.columns:
        cols.append("mean_intra_cosine")
    print(suspicious[cols].head(15).to_string(index=False))

    corr = temporal_df[["burst_score", "article_count", "span_days", "temporal_spread_days"]].corr()
    print("\nCorrelation matrix:")
    print(corr.to_string())
    return merged


def build_report(
    config_df: pd.DataFrame,
    best_per_topic: pd.DataFrame,
    cosine_df: pd.DataFrame,
    merged_temporal_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the final report table."""
    report_rows: list[dict[str, object]] = []

    if not config_df.empty:
        for _, row in best_per_topic.iterrows():
            report_rows.append(
                {
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
                }
            )

    for _, row in cosine_df.iterrows():
        report_rows.append(
            {
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
            }
        )

    if not merged_temporal_df.empty:
        for _, row in merged_temporal_df.iterrows():
            report_rows.append(
                {
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
                }
            )

    return pd.DataFrame(report_rows)


def main() -> int:
    args = parse_args()
    profile = detect_runtime_profile(device="cpu", cpu_threads=args.cpu_threads)
    apply_runtime_profile(profile)
    print(format_runtime_profile(profile))

    config_df, best_per_topic = load_config_results()
    cosine_df = compute_intra_cluster_cosine()
    merged_temporal_df = load_temporal_with_cosine(cosine_df)
    report_df = build_report(config_df, best_per_topic, cosine_df, merged_temporal_df)

    report_df.to_parquet(EVALUATION_REPORT, index=False, compression="zstd")
    print(f"\nEvaluation report saved to: {EVALUATION_REPORT}")
    print(f"Total rows: {len(report_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
