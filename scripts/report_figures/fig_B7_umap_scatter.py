"""Figure B7 — UMAP scatter for two illustrative topics.

Two-panel scatter of the pre-sampled UMAP projection (anchors + edge cases +
noise) for the two largest topics. Noise points are gray; clusters get
discrete colors. Reads ``data/dashboard/scatter_sample.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import DASH, OKABE_ITO, save, setup_style, topic_label


def panel(ax, df: pd.DataFrame, title: str) -> None:
    noise = df[df["cluster"] < 0]
    real = df[df["cluster"] >= 0]
    ax.scatter(noise["umap_x"], noise["umap_y"], s=6, c="#cccccc",
               alpha=0.55, linewidths=0, label="noise")

    clusters = real["cluster"].value_counts().head(10).index.tolist()
    palette = OKABE_ITO + ["#999999", "#7570b3", "#e7298a", "#66a61e"]
    for i, cl in enumerate(clusters):
        sub = real[real["cluster"] == cl]
        ax.scatter(sub["umap_x"], sub["umap_y"], s=10,
                   c=palette[i % len(palette)], alpha=0.85, linewidths=0)
    other = real[~real["cluster"].isin(clusters)]
    if len(other):
        ax.scatter(other["umap_x"], other["umap_y"], s=6, c="#888888",
                   alpha=0.55, linewidths=0, label="other clusters")
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    setup_style()
    df = pd.read_parquet(DASH / "scatter_sample.parquet")
    topics = ["politica", "international"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, t in zip(axes, topics):
        panel(ax, df[df["topic_group"] == t], topic_label(t))
    fig.suptitle("UMAP projection of SBERT embeddings (top 10 clusters per topic)")
    save(fig, "B7_umap_scatter")


if __name__ == "__main__":
    main()
