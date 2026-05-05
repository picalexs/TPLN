"""Figure B5 — Cluster size distribution.

Histogram of real cluster sizes (excluding noise / `cluster = -1`) on a
log-log scale, with a vertical line at the median.
Reads ``data/dashboard/cluster_overview.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import DASH, OKABE_ITO, save, setup_style


def main() -> None:
    setup_style()
    df = pd.read_parquet(DASH / "cluster_overview.parquet")
    sizes = df.loc[df["cluster"] >= 0, "article_count"].astype(int)

    bins = np.logspace(np.log10(max(sizes.min(), 2)), np.log10(sizes.max()), 40)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(sizes, bins=bins, color=OKABE_ITO[0], edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    median = float(np.median(sizes))
    ax.axvline(median, color=OKABE_ITO[3], linestyle="--",
               label=f"median = {median:.0f}")
    ax.set_xlabel("Cluster size (articles)")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Cluster size distribution (real clusters)")
    ax.legend(loc="upper right")
    save(fig, "B5_cluster_size_distribution")


if __name__ == "__main__":
    main()
