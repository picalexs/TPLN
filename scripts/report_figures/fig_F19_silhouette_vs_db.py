"""Figure F19 — Silhouette vs Davies-Bouldin scatter across methods.

Each (method, topic) becomes a point. Higher silhouette is better; lower
Davies-Bouldin is better - so the upper-left is best. Reads
``data/tfidf_ablation_report.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import pandas as pd

from _common import DATA, OKABE_ITO, method_label, save, setup_style

METHODS = ["tfidf_kmeans", "sbert_kmeans", "sbert_hdbscan"]
COLORS = {"tfidf_kmeans": OKABE_ITO[1],
          "sbert_kmeans": OKABE_ITO[2],
          "sbert_hdbscan": OKABE_ITO[0]}
MARKERS = {"tfidf_kmeans": "o", "sbert_kmeans": "s", "sbert_hdbscan": "^"}


def main() -> None:
    setup_style()
    df = pd.read_parquet(DATA / "tfidf_ablation_report.parquet")
    df = df[df["method"].isin(METHODS)].dropna(
        subset=["silhouette", "davies_bouldin"])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in METHODS:
        sub = df[df["method"] == m]
        ax.scatter(sub["silhouette"], sub["davies_bouldin"],
                   s=70, c=COLORS[m], marker=MARKERS[m], alpha=0.85,
                   edgecolor="white", linewidth=0.6, label=method_label(m))
    ax.invert_yaxis()
    ax.set_xlabel("Silhouette (higher better →)")
    ax.set_ylabel("Davies-Bouldin (lower better ↓)")
    ax.set_title("Silhouette vs. Davies-Bouldin per (method, topic)")
    ax.legend(loc="lower right")
    save(fig, "F19_silhouette_vs_db")


if __name__ == "__main__":
    main()
