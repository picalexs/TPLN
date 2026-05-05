"""Figure G23 — Topic size vs total runtime.

Log-log scatter showing how clustering runtime scales with topic size,
annotated with topic names. Reads ``data/clusters/runtime_observability.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import CLUSTERS, OKABE_ITO, save, setup_style, topic_label


def main() -> None:
    setup_style()
    df = pd.read_parquet(CLUSTERS / "runtime_observability.parquet")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(df["topic_size"], df["topic_total_seconds"], s=80,
               c=OKABE_ITO[0], alpha=0.85, edgecolor="white", linewidth=0.6)
    for _, row in df.iterrows():
        ax.annotate(topic_label(row["topic_group"]),
                    (row["topic_size"], row["topic_total_seconds"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    x = np.array([df["topic_size"].min(), df["topic_size"].max()])
    log_x = np.log(df["topic_size"])
    log_y = np.log(df["topic_total_seconds"])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    ax.plot(x, np.exp(intercept) * x ** slope, color=OKABE_ITO[3],
            linestyle="--", linewidth=1, label=f"power-law fit (slope ≈ {slope:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Topic size (articles, log scale)")
    ax.set_ylabel("Total clustering time (seconds, log scale)")
    ax.set_title("Clustering runtime scaling with topic size")
    ax.legend(loc="lower right")
    save(fig, "G23_topic_size_vs_runtime")


if __name__ == "__main__":
    main()
