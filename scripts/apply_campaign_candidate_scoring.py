"""Add report-oriented campaign-candidate scores to temporal stats.

This is a lightweight post-hoc step. It reads the existing temporal parquet,
adds the stricter campaign-candidate columns, and writes the same file back.
No embedding, clustering, or temporal aggregation is rerun.

Run:
    python scripts/apply_campaign_candidate_scoring.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(encoding="utf-8", errors="replace")

from src.campaign_scoring import add_campaign_candidate_columns
from src.paths import TEMPORAL_STATS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resulting top candidates without overwriting the parquet.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not TEMPORAL_STATS.exists():
        print(f"ERROR: {TEMPORAL_STATS} not found. Run scripts/TemporalAnalysis.py first.")
        return 1

    df = pd.read_parquet(TEMPORAL_STATS)
    scored = add_campaign_candidate_columns(df)

    print(f"Loaded clusters:              {len(scored):,}")
    print(f"Organic-event titles:         {int(scored['organic_event_title'].sum()):,}")
    print(f"Public-affairs signals:       {int(scored['public_affairs_signal'].sum()):,}")
    print(f"Campaign candidates > 0:      {int((scored['campaign_candidate_score'] > 0).sum()):,}")
    print(f"Max campaign candidate score: {scored['campaign_candidate_score'].max():.3f}")

    preview_cols = [
        "topic_group",
        "total_articles",
        "active_days",
        "burst_periods_daily",
        "domain_count",
        "suspicion_score",
        "campaign_candidate_score",
        "representative_title",
    ]
    print("\nTop 12 campaign candidates:")
    print(
        scored.sort_values("campaign_candidate_score", ascending=False)
        .head(12)[preview_cols]
        .to_string(index=False, max_colwidth=80)
    )

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return 0

    scored.to_parquet(TEMPORAL_STATS, index=False, compression="zstd")
    print(f"\nSaved updated temporal stats to: {TEMPORAL_STATS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
