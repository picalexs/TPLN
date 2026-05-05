"""Repair broad suspicion scores for robust multi-year recurring clusters.

This script is optional and post-hoc. It recomputes the long sparse-span
penalty from the columns already stored in
``data/temporal/cluster_temporal_stats.parquet``.

The important guardrail is that long-span relief is only given to robust
recurring clusters, not tiny two-burst incidents:

    burst_periods_daily >= 5
    total_articles >= 50
    active_days >= 20
    domain_count >= 3

For report case studies, prefer ``campaign_candidate_score`` from
``scripts/apply_campaign_candidate_scoring.py``. That score also filters obvious
organic news events by title.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(encoding="utf-8", errors="replace")

from src.campaign_scoring import add_campaign_candidate_columns
from src.paths import TEMPORAL_STATS

MIN_PERIODS_DEFAULT = 5
MIN_ARTICLES_DEFAULT = 50
MIN_ACTIVE_DAYS_DEFAULT = 20
MIN_DOMAINS_DEFAULT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-periods", type=int, default=MIN_PERIODS_DEFAULT)
    parser.add_argument("--min-articles", type=int, default=MIN_ARTICLES_DEFAULT)
    parser.add_argument("--min-active-days", type=int, default=MIN_ACTIVE_DAYS_DEFAULT)
    parser.add_argument("--min-domains", type=int, default=MIN_DOMAINS_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _recomputed_long_sparse_penalty(df: pd.DataFrame) -> pd.Series:
    """Recompute the long-span penalty without trusting stored patched values."""
    return (
        np.log1p(df["span_days"].fillna(0))
        * np.maximum(0.0, df["span_days"].fillna(0) - 90.0)
        / 90.0
        * (1.0 - df["active_day_ratio"].fillna(0))
        * (1.0 - df["timestamp_coverage_ratio"].fillna(0))
        * 1.8
    )


def main() -> int:
    args = parse_args()
    if not TEMPORAL_STATS.exists():
        print(f"ERROR: {TEMPORAL_STATS} not found. Run scripts/TemporalAnalysis.py first.")
        return 1

    df = pd.read_parquet(TEMPORAL_STATS)
    print(f"Loaded {len(df):,} clusters from {TEMPORAL_STATS}")

    robust_recurring = (
        (df["burst_periods_daily"].fillna(0) >= args.min_periods)
        & (df["total_articles"].fillna(0) >= args.min_articles)
        & (df["active_days"].fillna(0) >= args.min_active_days)
        & (df["domain_count"].fillna(0) >= args.min_domains)
    )

    recomputed_long_penalty = _recomputed_long_sparse_penalty(df)
    other_penalties = df["suspicion_penalty_total"].fillna(0) - df["long_sparse_span_penalty"].fillna(0)
    adjusted_long_penalty = recomputed_long_penalty.mask(robust_recurring, 0.0)
    adjusted_penalty_total = other_penalties + adjusted_long_penalty

    weights_product = (
        df["support_weight"].fillna(0)
        * df["coverage_weight"].fillna(0)
        * df["source_weight"].fillna(0)
        * df["domain_weight"].fillna(0)
    )
    adjusted_score = (
        df["suspicion_score_raw"].fillna(0) * weights_product
        - adjusted_penalty_total
    ).clip(lower=0).round(3)

    print(f"Robust recurring clusters: {int(robust_recurring.sum()):,}")
    print(f"Score changes:             {int((adjusted_score != df['suspicion_score']).sum()):,}")
    print(f"Previous max score:        {df['suspicion_score'].max():.3f}")
    print(f"Adjusted max score:        {adjusted_score.max():.3f}")

    preview = df.copy()
    preview["adjusted_score"] = adjusted_score
    preview = preview.sort_values("adjusted_score", ascending=False)
    print("\nTop 12 adjusted broad-suspicion clusters:")
    print(
        preview.head(12)[
            [
                "topic_group",
                "total_articles",
                "active_days",
                "burst_periods_daily",
                "domain_count",
                "suspicion_score",
                "adjusted_score",
                "representative_title",
            ]
        ].to_string(index=False, max_colwidth=80)
    )

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return 0

    df["long_sparse_span_penalty"] = adjusted_long_penalty.round(4)
    df["suspicion_penalty_total"] = adjusted_penalty_total.round(4)
    df["suspicion_score"] = adjusted_score
    df = add_campaign_candidate_columns(df)
    df = df.sort_values("suspicion_score", ascending=False).reset_index(drop=True)
    df.to_parquet(TEMPORAL_STATS, index=False, compression="zstd")
    print(f"\nSaved patched stats to: {TEMPORAL_STATS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
