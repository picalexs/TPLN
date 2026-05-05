"""Figure A3 — Timestamp coverage ratio per topic.

How many articles in each topic actually carry a usable timestamp.
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
    df = df.sort_values("timestamp_coverage_ratio", ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.barh([topic_label(t) for t in df["topic_group"]],
                   df["timestamp_coverage_ratio"], color=OKABE_ITO[2])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction with timestamp")
    ax.set_title("Timestamp coverage by topic")
    for bar, value in zip(bars, df["timestamp_coverage_ratio"]):
        ax.text(value, bar.get_y() + bar.get_height() / 2,
                f" {value:.0%}", va="center", fontsize=8)
    save(fig, "A3_timestamp_coverage")


if __name__ == "__main__":
    main()
