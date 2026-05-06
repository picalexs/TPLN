"""Figure E16 — Domain entropy vs. suspicion score.

Scatter showing how source-domain diversity correlates with the multi-source
suspicion ranking. High entropy (diverse outlets) + high score is the most
interesting signal — it implies many outlets are publishing similar content.
Reads ``data/temporal/cluster_temporal_stats.parquet``.
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
                         columns=["domain_entropy", "suspicion_score",
                                  "domain_count", "article_count"])
    df = df.dropna(subset=["domain_entropy", "suspicion_score"])
    df["is_single_source"] = df["domain_count"] == 1

    fig, ax = plt.subplots(figsize=(7.5, 5))
    sizes = np.clip(df["article_count"].fillna(10) ** 0.5, 4, 50)
    single = df[df["is_single_source"]]
    multi = df[~df["is_single_source"]]
    ax.scatter(single["domain_entropy"], single["suspicion_score"],
               s=sizes.loc[single.index], c="#bbbbbb", alpha=0.6, linewidths=0,
               label="single-source")
    ax.scatter(multi["domain_entropy"], multi["suspicion_score"],
               s=sizes.loc[multi.index], c=OKABE_ITO[3], alpha=0.85,
               edgecolor="white", linewidth=0.4, label="multi-source")
    ax.set_xlabel("Domain entropy (Shannon, base e)")
    ax.set_ylabel("Suspicion score")
    ax.set_title("Domain entropy vs. suspicion score")
    ax.legend(loc="upper left")
    save(fig, "E16_domain_entropy_vs_suspicion")


if __name__ == "__main__":
    main()
