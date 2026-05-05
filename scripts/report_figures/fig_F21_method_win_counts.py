"""Figure F21 — Number of topics each method wins on quality_index.

Counts how many topics report each method as the ``topic_winner_method``.
Reads ``data/tfidf_ablation_report.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import pandas as pd

from _common import DATA, OKABE_ITO, method_label, save, setup_style


def main() -> None:
    setup_style()
    df = pd.read_parquet(DATA / "tfidf_ablation_report.parquet")
    winners = (df.dropna(subset=["topic_winner_method"])
                 .drop_duplicates(subset=["topic_group"])["topic_winner_method"]
                 .value_counts())

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [OKABE_ITO[1] if m == "tfidf_kmeans"
              else OKABE_ITO[2] if m == "sbert_kmeans"
              else OKABE_ITO[0] for m in winners.index]
    bars = ax.bar([method_label(m) for m in winners.index], winners.values,
                  color=colors)
    for bar, value in zip(bars, winners.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value,
                f"{int(value)}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Topics where method wins on quality_index")
    ax.set_title("Method win counts across topics")
    ax.set_ylim(0, max(winners.values) * 1.18)
    save(fig, "F21_method_win_counts")


if __name__ == "__main__":
    main()
