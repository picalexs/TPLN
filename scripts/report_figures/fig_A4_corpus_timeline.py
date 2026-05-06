"""Figure A4 — Corpus-wide publication timeline (monthly).

Shows monthly article counts for the whole corpus, restricted to
timestamped articles. Reads ``data/dashboard/cluster_articles.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import pandas as pd

from _common import DASH, OKABE_ITO, save, setup_style


def main() -> None:
    setup_style()
    df = pd.read_parquet(DASH / "cluster_articles.parquet",
                         columns=["timestamp_date"])
    df = df.dropna(subset=["timestamp_date"])
    df["timestamp_date"] = pd.to_datetime(df["timestamp_date"], errors="coerce")
    df = df.dropna(subset=["timestamp_date"])
    monthly = df.set_index("timestamp_date").resample("MS").size()
    monthly = monthly[(monthly.index.year >= 2010) & (monthly.index.year <= 2025)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(monthly.index, monthly.values, color=OKABE_ITO[0], alpha=0.6)
    ax.plot(monthly.index, monthly.values, color=OKABE_ITO[0], linewidth=1.0)
    ax.set_xlabel("Month")
    ax.set_ylabel("Articles per month")
    ax.set_title("Corpus publication timeline (timestamped articles only)")
    save(fig, "A4_corpus_timeline")


if __name__ == "__main__":
    main()
