"""Figure D13 — Daily timelines of three highly suspicious clusters.

Shows the daily article-count timeline for the top clusters by
``suspicion_score_multi_source``, which makes the burst pattern visible.
Reads ``data/dashboard/cluster_daily_counts.parquet`` and
``data/temporal/cluster_temporal_stats.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import pandas as pd

from _common import DASH, OKABE_ITO, TEMPORAL, save, setup_style, shorten, topic_label

N_PANELS = 3


def main() -> None:
    setup_style()
    ts = pd.read_parquet(TEMPORAL / "cluster_temporal_stats.parquet",
                         columns=["topic_group", "cluster", "suspicion_score",
                                  "domain_count", "burst_score_daily", "representative_title"])
    ts["score"] = ts["suspicion_score"].where(ts["domain_count"] > 1, 0.0)
    daily = pd.read_parquet(DASH / "cluster_daily_counts.parquet")
    daily["date"] = pd.to_datetime(daily["date"])

    chosen = ts.sort_values("score", ascending=False).head(N_PANELS)

    fig, axes = plt.subplots(N_PANELS, 1, figsize=(10.5, 2.6 * N_PANELS + 0.6))
    if N_PANELS == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, chosen.iterrows()):
        sub = daily[(daily["topic_group"] == row["topic_group"]) &
                    (daily["cluster"] == row["cluster"])]
        sub = sub.sort_values("date")
        ax.fill_between(sub["date"], sub["article_count"],
                        color=OKABE_ITO[0], alpha=0.45)
        ax.plot(sub["date"], sub["article_count"], color=OKABE_ITO[0], linewidth=1.0)
        ax.set_ylabel("Articles / day")
        title = (f"[{topic_label(row['topic_group'])}] "
                 f"{shorten(str(row['representative_title']), 80)}  "
                 f"(suspicion = {row['score']:.2f}, "
                 f"daily burst = {row['burst_score_daily']:.2f})")
        ax.set_title(title, fontsize=9.5, loc="left")
    axes[-1].set_xlabel("Date")
    fig.suptitle("Daily timelines of top suspicious clusters", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "D13_top_cluster_timelines")


if __name__ == "__main__":
    main()
