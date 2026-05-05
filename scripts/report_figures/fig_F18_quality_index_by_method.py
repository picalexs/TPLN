"""Figure F18 — Quality index by method, grouped by topic.

The ablation pipeline computes a normalized 0-1 ``quality_index`` per
(method, topic). This grouped bar shows how each clustering method
performs across topics. Reads ``data/tfidf_ablation_report.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import DATA, OKABE_ITO, method_label, save, setup_style, topic_label

METHODS = ["tfidf_kmeans", "sbert_kmeans", "sbert_hdbscan"]


def main() -> None:
    setup_style()
    df = pd.read_parquet(DATA / "tfidf_ablation_report.parquet")
    df = df[df["method"].isin(METHODS)].copy()
    df = df.dropna(subset=["quality_index"])
    pivot = df.pivot_table(index="topic_group", columns="method",
                           values="quality_index")
    pivot = pivot[METHODS]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    topics = pivot.index.tolist()
    x = np.arange(len(topics))
    width = 0.27
    colors = [OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[0]]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (m, c) in enumerate(zip(METHODS, colors)):
        ax.bar(x + (i - 1) * width, pivot[m].values, width=width, color=c,
               label=method_label(m))
    ax.set_xticks(x)
    ax.set_xticklabels([topic_label(t) for t in topics], rotation=30, ha="right")
    ax.set_ylabel("Quality index (per-topic, 0–1 normalized)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Clustering quality by method, per topic")
    ax.legend(loc="upper right", ncols=3)
    save(fig, "F18_quality_index_by_method")


if __name__ == "__main__":
    main()
