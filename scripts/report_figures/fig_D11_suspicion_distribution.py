"""Figure D11 — Distribution of suspicion scores across clusters.

Histogram of ``suspicion_score`` and ``suspicion_score_multi_source`` from the
temporal stage. Most clusters get score zero; the long right tail is what we
care about. Reads ``data/temporal/cluster_temporal_stats.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import OKABE_ITO, TEMPORAL, save, setup_style


def main() -> None:
    setup_style()
    df = pd.read_parquet(TEMPORAL / "cluster_temporal_stats.parquet",
                         columns=["suspicion_score", "domain_count"])
    # multi-source = clusters with more than one domain (single-domain bursts
    # are usually one outlet republishing, not coordinated)
    df["suspicion_score_multi_source"] = df["suspicion_score"].where(df["domain_count"] > 1, 0.0)
    s_all = df["suspicion_score"].dropna().values
    s_multi = df["suspicion_score_multi_source"].dropna().values

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bins = np.linspace(0, max(s_all.max(), s_multi.max()) + 0.5, 40)
    ax.hist(s_all, bins=bins, color=OKABE_ITO[0], alpha=0.55,
            label="All clusters", edgecolor="white", linewidth=0.4)
    ax.hist(s_multi, bins=bins, color=OKABE_ITO[3], alpha=0.7,
            label="Multi-source only", edgecolor="white", linewidth=0.4)
    ax.set_yscale("symlog")
    ax.set_xlabel("Suspicion score")
    ax.set_ylabel("Number of clusters (symlog)")
    ax.set_title("Suspicion score distribution across clusters")
    ax.legend(loc="upper right")
    save(fig, "D11_suspicion_distribution")


if __name__ == "__main__":
    main()
