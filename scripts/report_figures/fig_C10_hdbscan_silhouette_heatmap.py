"""Figure C10 — HDBSCAN silhouette heatmap (min_cluster_size x min_samples).

One panel per topic with silhouette plotted as a heatmap over the two main
HDBSCAN hyperparameters. Reads ``data/clusters/hdbscan_config_results.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import CLUSTERS, save, setup_style, topic_label


def main() -> None:
    setup_style()
    df = pd.read_parquet(CLUSTERS / "hdbscan_config_results.parquet")
    topics = sorted(df["topic_group"].unique().tolist())
    cols = 2
    rows = (len(topics) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(11.5, 4.1 * rows))
    axes = axes.flatten()

    vmin = float(df["silhouette"].min())
    vmax = float(df["silhouette"].max())
    for ax, t in zip(axes, topics):
        sub = df[df["topic_group"] == t]
        pivot = sub.pivot_table(index="min_samples", columns="min_cluster_size",
                                values="silhouette", aggfunc="max")
        pivot = pivot.sort_index(ascending=False)
        im = ax.imshow(pivot.values, cmap="viridis", vmin=vmin, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("min_cluster_size", labelpad=8)
        ax.set_ylabel("min_samples", labelpad=8)
        ax.set_title(topic_label(t))
        ax.grid(False)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="white" if v < (vmin + vmax) / 2 else "black",
                            fontsize=7)
    for ax in axes[len(topics):]:
        ax.set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.95, bottom=0.06, hspace=0.42, wspace=0.28)
    cbar_ax = fig.add_axes([0.91, 0.16, 0.018, 0.70])
    fig.colorbar(im, cax=cbar_ax, label="Silhouette")
    fig.suptitle("HDBSCAN silhouette across hyperparameter sweep", y=0.985)
    save(fig, "C10_hdbscan_silhouette_heatmap")


if __name__ == "__main__":
    main()
