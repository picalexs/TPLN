"""Figure D12 — Top-K most suspicious clusters.

Horizontal bar chart of the highest-scoring clusters by
``suspicion_score_multi_source`` (so single-source spikes are filtered out).
Reads ``data/temporal/cluster_temporal_stats.parquet`` and
``data/dashboard/cluster_overview.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import pandas as pd

from _common import OKABE_ITO, TEMPORAL, save, setup_style, shorten, topic_label

TOP_K = 15


def main() -> None:
    setup_style()
    ts = pd.read_parquet(TEMPORAL / "cluster_temporal_stats.parquet",
                         columns=["topic_group", "cluster", "suspicion_score",
                                  "domain_count", "representative_title", "article_count"])
    # restrict to multi-source clusters (domain_count > 1) before ranking
    ts["score"] = ts["suspicion_score"].where(ts["domain_count"] > 1, 0.0)
    df = ts.sort_values("score", ascending=False).head(TOP_K).iloc[::-1]

    labels = [
        f"[{topic_label(t)}] {shorten(title, 70)}"
        for t, title in zip(df["topic_group"], df["representative_title"].fillna(""))
    ]
    fig, ax = plt.subplots(figsize=(10, 0.45 * TOP_K + 1.0))
    bars = ax.barh(labels, df["score"], color=OKABE_ITO[3])
    for bar, value, n in zip(bars, df["score"], df["article_count"]):
        ax.text(value, bar.get_y() + bar.get_height() / 2,
                f"  {value:.2f}  ({int(n) if pd.notna(n) else 0} art.)",
                va="center", fontsize=8)
    ax.set_xlabel("Suspicion score (multi-source clusters)")
    ax.set_title(f"Top {TOP_K} most suspicious clusters")
    save(fig, "D12_top_suspicious_clusters")


if __name__ == "__main__":
    main()
