"""Figure F20 — Runtime vs silhouette tradeoff.

For TF-IDF / SBERT × KMeans / HDBSCAN, plot runtime against silhouette to
visualize the cost-quality tradeoff. SBERT+HDBSCAN runtime is taken from the
``runtime_observability`` artefact (HDBSCAN sweep + UMAP + reassign times).
Reads ``data/tfidf_ablation_report.parquet`` and
``data/clusters/runtime_observability.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import pandas as pd

from _common import CLUSTERS, DATA, OKABE_ITO, method_label, save, setup_style

METHODS = ["tfidf_kmeans", "sbert_kmeans", "sbert_hdbscan"]
COLORS = {"tfidf_kmeans": OKABE_ITO[1],
          "sbert_kmeans": OKABE_ITO[2],
          "sbert_hdbscan": OKABE_ITO[0]}
MARKERS = {"tfidf_kmeans": "o", "sbert_kmeans": "s", "sbert_hdbscan": "^"}


def main() -> None:
    setup_style()
    df = pd.read_parquet(DATA / "tfidf_ablation_report.parquet")
    df = df[df["method"].isin(METHODS)].copy()

    obs = pd.read_parquet(CLUSTERS / "runtime_observability.parquet",
                          columns=["topic_group", "umap_seconds",
                                   "hdbscan_sweep_seconds", "label_apply_seconds"])
    obs["sbert_hdbscan_runtime"] = (obs["umap_seconds"].fillna(0)
                                    + obs["hdbscan_sweep_seconds"].fillna(0)
                                    + obs["label_apply_seconds"].fillna(0))
    df = df.merge(obs[["topic_group", "sbert_hdbscan_runtime"]],
                  on="topic_group", how="left")
    df["runtime"] = df["runtime_seconds"].copy()
    mask = df["method"] == "sbert_hdbscan"
    df.loc[mask, "runtime"] = df.loc[mask, "sbert_hdbscan_runtime"]
    df = df.dropna(subset=["silhouette", "runtime"])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in METHODS:
        sub = df[df["method"] == m]
        ax.scatter(sub["runtime"], sub["silhouette"], s=70, c=COLORS[m],
                   marker=MARKERS[m], alpha=0.85, edgecolor="white",
                   linewidth=0.6, label=method_label(m))
    ax.set_xscale("log")
    ax.set_xlabel("Runtime per topic (seconds, log scale)")
    ax.set_ylabel("Silhouette")
    ax.set_title("Runtime vs. silhouette tradeoff")
    ax.legend(loc="lower right")
    save(fig, "F20_runtime_vs_silhouette")


if __name__ == "__main__":
    main()
