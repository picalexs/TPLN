"""Figure A2 — Timestamp source breakdown per topic.

Shows the share of timestamps coming from URL parsing vs. article text vs.
htmldate vs. missing. Reads ``data/dashboard/cluster_articles.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import pandas as pd

from _common import DASH, OKABE_ITO, TIMESTAMP_SOURCE_LABELS, save, setup_style, topic_label


def main() -> None:
    setup_style()
    df = pd.read_parquet(DASH / "cluster_articles.parquet",
                         columns=["topic_group", "timestamp_source"])
    df["timestamp_source"] = df["timestamp_source"].fillna("missing")
    pivot = (df.groupby(["topic_group", "timestamp_source"]).size()
               .unstack(fill_value=0))
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    order_cols = [c for c in ["url", "htmldate", "text", "missing"] if c in pivot.columns]
    pivot = pivot[order_cols]
    pivot = pivot.loc[pivot["url"].sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = pd.Series(0.0, index=pivot.index)
    color_map = {"url": OKABE_ITO[0], "htmldate": OKABE_ITO[2],
                 "text": OKABE_ITO[1], "missing": "#bbbbbb"}
    for src in pivot.columns:
        ax.barh([topic_label(t) for t in pivot.index], pivot[src], left=bottom,
                color=color_map.get(src, OKABE_ITO[3]),
                label=TIMESTAMP_SOURCE_LABELS.get(src, src))
        bottom = bottom + pivot[src].values
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of articles")
    ax.set_title("Timestamp provenance by topic")
    ax.legend(loc="lower right", ncols=4)
    save(fig, "A2_timestamp_source_breakdown")


if __name__ == "__main__":
    main()
