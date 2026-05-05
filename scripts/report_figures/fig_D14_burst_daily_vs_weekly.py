"""Figure D14 — Daily vs weekly burst score scatter.

Stable bursts show up on both daily and weekly windows; ephemeral spikes
appear only daily. Color encodes ``burst_stable``; size encodes article count.
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
                         columns=["burst_score_daily", "burst_score_weekly",
                                  "burst_stable", "article_count"])
    df = df.dropna(subset=["burst_score_daily", "burst_score_weekly"])

    fig, ax = plt.subplots(figsize=(7, 5.3))
    sizes = np.clip(df["article_count"].fillna(10) ** 0.5, 4, 40)
    stable = df[df["burst_stable"].astype(bool)]
    other = df[~df["burst_stable"].astype(bool)]
    ax.scatter(other["burst_score_daily"], other["burst_score_weekly"],
               s=sizes.loc[other.index], c="#bbbbbb", alpha=0.6, linewidths=0,
               label="not stable")
    ax.scatter(stable["burst_score_daily"], stable["burst_score_weekly"],
               s=sizes.loc[stable.index], c=OKABE_ITO[3], alpha=0.85,
               edgecolor="white", linewidth=0.4, label="burst_stable = True")
    lim = max(df["burst_score_daily"].max(), df["burst_score_weekly"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Daily burst score")
    ax.set_ylabel("Weekly burst score")
    ax.set_title("Daily vs. weekly burst score (marker size = article count)")
    ax.legend(loc="lower right")
    save(fig, "D14_burst_daily_vs_weekly")


if __name__ == "__main__":
    main()
