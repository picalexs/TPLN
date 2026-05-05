"""Figure E17 — Single-source vs multi-source cluster mix per topic.

Stacked bar showing how many real clusters in each topic are dominated by
a single domain vs. supported by multiple domains. Single-source clusters
are typically discounted from the suspicion ranking.
Reads ``data/temporal/cluster_temporal_stats.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import pandas as pd

from _common import OKABE_ITO, TEMPORAL, save, setup_style, topic_label


def main() -> None:
    setup_style()
    df = pd.read_parquet(TEMPORAL / "cluster_temporal_stats.parquet",
                         columns=["topic_group", "domain_count"])
    df["is_single_source"] = df["domain_count"] == 1
    counts = (df.groupby(["topic_group", "is_single_source"]).size()
                .unstack(fill_value=0))
    counts.columns = ["multi-source", "single-source"][:len(counts.columns)] \
        if False else ["multi-source" if not c else "single-source" for c in counts.columns]
    counts = counts.assign(total=lambda d: d.sum(axis=1)).sort_values("total")
    counts = counts.drop(columns=["total"])

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bottom = pd.Series(0, index=counts.index)
    color_map = {"multi-source": OKABE_ITO[2], "single-source": OKABE_ITO[3]}
    for col in ["multi-source", "single-source"]:
        if col not in counts.columns:
            continue
        ax.barh([topic_label(t) for t in counts.index], counts[col], left=bottom,
                color=color_map[col], label=col)
        bottom = bottom + counts[col].values
    ax.set_xlabel("Number of real clusters")
    ax.set_title("Single- vs. multi-source clusters per topic")
    ax.legend(loc="lower right")
    save(fig, "E17_single_vs_multi_source")


if __name__ == "__main__":
    main()
