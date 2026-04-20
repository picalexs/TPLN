"""Summarize cluster quality from the parquet artefacts.

Reads:
    data/clusters/clustered_data.parquet
    data/clusters/hdbscan_config_results.parquet
    data/temporal/cluster_temporal_stats.parquet

Prints:
    - topic-group distribution
    - eligibility breakdown
    - global and per-topic noise shares
    - cluster size distribution + top 15 largest clusters
    - HDBSCAN config sweep table
    - temporal summary (burst counts, median coverage, suspicion score spread)

Run after EmbeddingsClustering.py (and optionally TemporalAnalysis.py) to
confirm the clustering hyperparameters produced a healthy run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paths import CLUSTERED_PARQUET, HDBSCAN_CONFIG_RESULTS, TEMPORAL_STATS


def main() -> None:
    df = pd.read_parquet(CLUSTERED_PARQUET)
    print(f"Loaded clustered_data.parquet: {len(df):,} rows")
    print(f"Columns: {df.columns.tolist()}")
    print()

    print("=" * 70)
    print("TOPIC GROUPS (post-normalization)")
    print("=" * 70)
    print(df["topic_group"].value_counts(dropna=False).head(30).to_string())
    print()

    print("=" * 70)
    print("ELIGIBILITY (topics >= MIN_TOPIC_SIZE participated in clustering)")
    print("=" * 70)
    print(df["topic_is_eligible"].value_counts(dropna=False).to_string())
    elig = df.groupby("topic_is_eligible").size()
    total = len(df)
    for k, v in elig.items():
        print(f"  {k!r:>15}: {v:>8,} ({100*v/total:.1f}%)")
    print()

    print("=" * 70)
    print("GLOBAL NOISE SHARE")
    print("=" * 70)
    noise = (df["cluster"] == -1).sum()
    clustered = (df["cluster"] != -1).sum()
    print(f"  total:           {total:>8,}")
    print(f"  noise (-1):      {noise:>8,} ({100*noise/total:.1f}%)")
    print(f"  real clusters:   {clustered:>8,} ({100*clustered/total:.1f}%)")
    print()

    print("=" * 70)
    print("CLUSTER ASSIGNMENT REASON BREAKDOWN")
    print("=" * 70)
    print(df["cluster_assignment_reason"].value_counts(dropna=False).to_string())
    print()

    print("=" * 70)
    print("NOISE SHARE PER ELIGIBLE TOPIC (only rows that went through HDBSCAN)")
    print("=" * 70)
    eligible = df[df["topic_is_eligible"] == True].copy()
    per_topic = eligible.groupby("topic_group", group_keys=False).apply(
        lambda g: pd.Series({
            "rows": len(g),
            "noise": (g["cluster"] == -1).sum(),
            "noise_pct": round(100 * (g["cluster"] == -1).mean(), 1),
            "num_clusters": g.loc[g["cluster"] != -1, "cluster"].nunique(),
        }),
        include_groups=False,
    ).sort_values("rows", ascending=False)
    print(per_topic.to_string())
    print()

    print("=" * 70)
    print("CLUSTER SIZE DISTRIBUTION (real clusters only)")
    print("=" * 70)
    real = df[df["cluster"] != -1].copy()
    sizes = real["cluster"].value_counts()
    print(f"  number of real clusters: {len(sizes):,}")
    print(f"  min size:  {int(sizes.min()) if len(sizes) else 0}")
    print(f"  p25 size:  {int(sizes.quantile(0.25)) if len(sizes) else 0}")
    print(f"  median:    {int(sizes.median()) if len(sizes) else 0}")
    print(f"  p75 size:  {int(sizes.quantile(0.75)) if len(sizes) else 0}")
    print(f"  p95 size:  {int(sizes.quantile(0.95)) if len(sizes) else 0}")
    print(f"  max:       {int(sizes.max()) if len(sizes) else 0}")
    print(f"  mean:      {sizes.mean():.1f}")
    print()
    print("  Top 15 largest clusters (size, topic):")
    for cid, sz in sizes.head(15).items():
        topic = real.loc[real["cluster"] == cid, "topic_group"].mode()
        topic = topic.iloc[0] if not topic.empty else "?"
        print(f"    cluster {int(cid):>4d}  size={int(sz):>6d}  topic={topic}")
    print()

    print("=" * 70)
    print("MEMBERSHIP STRENGTH / OUTLIER SCORE (on real clusters)")
    print("=" * 70)
    ms = real["cluster_membership_strength"].dropna()
    os_ = real["cluster_outlier_score"].dropna()
    if len(ms):
        print(f"  membership_strength: mean={ms.mean():.3f} median={ms.median():.3f} p10={ms.quantile(0.1):.3f} p90={ms.quantile(0.9):.3f}")
    if len(os_):
        print(f"  outlier_score:       mean={os_.mean():.3f} median={os_.median():.3f} p10={os_.quantile(0.1):.3f} p90={os_.quantile(0.9):.3f}")
    print()

    if HDBSCAN_CONFIG_RESULTS.exists():
        print("=" * 70)
        print("HDBSCAN CONFIG SWEEP (per topic)")
        print("=" * 70)
        cfg = pd.read_parquet(HDBSCAN_CONFIG_RESULTS)
        cols = [
            "topic_group", "topic_size", "min_cluster_size", "min_samples",
            "cluster_selection_method", "cluster_selection_epsilon",
            "num_clusters", "num_noise", "noise_percent",
            "largest_real_cluster", "silhouette", "selection_score",
        ]
        cols = [c for c in cols if c in cfg.columns]
        print(cfg[cols].sort_values(["topic_group", "selection_score"], ascending=[True, False]).to_string(index=False))
        print()

    if TEMPORAL_STATS.exists():
        print("=" * 70)
        print("TEMPORAL STATS SUMMARY")
        print("=" * 70)
        t = pd.read_parquet(TEMPORAL_STATS)
        print(f"  total real clusters in temporal stats: {len(t):,}")
        print(f"  daily burst > 0:   {(t['burst_score_daily'] > 0).sum():,}")
        print(f"  weekly burst > 0:  {(t['burst_score_weekly'] > 0).sum():,}")
        print(f"  burst_stable == 1: {(t['burst_stable'] == 1).sum():,}")
        print(f"  median span_days:  {t['span_days'].median():.1f}")
        print(f"  median active_day_ratio: {t['active_day_ratio'].median():.3f}")
        print(f"  median timestamp_coverage_ratio: {t['timestamp_coverage_ratio'].median():.3f}")
        print(f"  median domain_count: {t['domain_count'].median():.1f}")
        print(f"  median top_domain_share: {t['top_domain_share'].median():.3f}")
        print()
        print("  Suspicion-score distribution:")
        print(f"    min={t['suspicion_score'].min():.2f} "
              f"p25={t['suspicion_score'].quantile(0.25):.2f} "
              f"median={t['suspicion_score'].median():.2f} "
              f"p75={t['suspicion_score'].quantile(0.75):.2f} "
              f"p95={t['suspicion_score'].quantile(0.95):.2f} "
              f"max={t['suspicion_score'].max():.2f}")


if __name__ == "__main__":
    main()
