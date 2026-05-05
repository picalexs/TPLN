"""Figure C9 — HDBSCAN sweep: selection score vs. noise rate.

One panel per topic with hyperparameter sweep results. Marker size encodes
``min_cluster_size``; the chosen-best config (highest selection_score) is
highlighted with a black ring. Reads ``data/clusters/hdbscan_config_results.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import pandas as pd

from _common import CLUSTERS, OKABE_ITO, save, setup_style, topic_label


def main() -> None:
    setup_style()
    df = pd.read_parquet(CLUSTERS / "hdbscan_config_results.parquet")
    topics = sorted(df["topic_group"].unique().tolist())
    n = len(topics)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.3 * rows), sharey=True)
    axes = axes.flatten()

    method_colors = {"eom": OKABE_ITO[0], "leaf": OKABE_ITO[1]}
    for ax, t in zip(axes, topics):
        sub = df[df["topic_group"] == t]
        for method in sub["cluster_selection_method"].unique():
            d = sub[sub["cluster_selection_method"] == method]
            ax.scatter(d["noise_percent"], d["selection_score"],
                       s=d["min_cluster_size"] * 4,
                       c=method_colors.get(method, OKABE_ITO[2]),
                       alpha=0.75, edgecolor="white", linewidth=0.7,
                       label=str(method))
        best = sub.loc[sub["selection_score"].idxmax()]
        ax.scatter([best["noise_percent"]], [best["selection_score"]],
                   s=best["min_cluster_size"] * 4 + 30,
                   facecolors="none", edgecolors="black", linewidth=1.5)
        ax.set_title(topic_label(t))
        ax.set_xlabel("Noise %")
        ax.set_ylabel("Selection score")
    for ax in axes[len(topics):]:
        ax.set_visible(False)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                          markerfacecolor=method_colors[m], markeredgecolor="white",
                          label=f"{m}") for m in method_colors]
    handles.append(plt.Line2D([0], [0], marker="o", linestyle="", markersize=10,
                              markerfacecolor="none", markeredgecolor="black",
                              label="chosen best"))
    fig.legend(handles=handles, loc="lower center", ncols=3,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("HDBSCAN sweep: selection score vs. noise rate (marker size = min_cluster_size)",
                 y=1.0)
    fig.tight_layout()
    save(fig, "C9_hdbscan_sweep_scatter")


if __name__ == "__main__":
    main()
