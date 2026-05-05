"""Figure A1 — Article counts per topic group.

Shows how the corpus is distributed across the 12 normalized topic buckets.
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
    df = df.sort_values("total_rows", ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = [topic_label(t) for t in df["topic_group"]]
    bars = ax.barh(labels, df["total_rows"], color=OKABE_ITO[0])
    ax.set_xscale("log")
    ax.set_xlabel("Number of articles (log scale)")
    ax.set_title("Article distribution across topic groups")
    for bar, value in zip(bars, df["total_rows"]):
        ax.text(value, bar.get_y() + bar.get_height() / 2,
                f" {int(value):,}", va="center", fontsize=8)
    ax.set_axisbelow(True)
    save(fig, "A1_topic_distribution")


if __name__ == "__main__":
    main()
