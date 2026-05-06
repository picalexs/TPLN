"""Figure B6 — HDBSCAN noise rate per topic.

Bar chart of noise rate (share of unclustered articles) per topic.
Reads ``data/dashboard/topic_summary.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import pandas as pd

from _common import DASH, OKABE_ITO, save, setup_style, topic_label


def main() -> None:
    setup_style()
    df = pd.read_parquet(DASH / "topic_summary.parquet")
    df = df.sort_values("noise_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.barh([topic_label(t) for t in df["topic_group"]],
                   df["noise_rate"], color=OKABE_ITO[3])
    ax.set_xlim(0, max(df["noise_rate"].max() * 1.15, 0.5))
    ax.set_xlabel("Fraction labeled noise by HDBSCAN")
    ax.set_title("Noise rate per topic")
    for bar, value in zip(bars, df["noise_rate"]):
        ax.text(value, bar.get_y() + bar.get_height() / 2,
                f" {value:.0%}", va="center", fontsize=8)
    save(fig, "B6_noise_rate_per_topic")


if __name__ == "__main__":
    main()
