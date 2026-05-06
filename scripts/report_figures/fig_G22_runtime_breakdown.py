"""Figure G22 — Per-topic stage runtime breakdown.

Stacked horizontal bar showing seconds spent in each pipeline stage
(embedding, FAISS, dedup, UMAP, HDBSCAN sweep, reassignment, label apply)
per topic. Reads ``data/clusters/runtime_observability.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import pandas as pd

from _common import CLUSTERS, OKABE_ITO, save, setup_style, topic_label

STAGES = [
    ("embedding_seconds", "Embedding"),
    ("faiss_seconds", "FAISS sanity check"),
    ("umap_seconds", "UMAP"),
    ("hdbscan_sweep_seconds", "HDBSCAN sweep"),
    ("label_apply_seconds", "Label apply"),
]


def main() -> None:
    setup_style()
    df = pd.read_parquet(CLUSTERS / "runtime_observability.parquet")
    df = df.sort_values("topic_total_seconds", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    labels = [topic_label(t) for t in df["topic_group"]]
    bottom = pd.Series(0.0, index=range(len(df)))
    palette = OKABE_ITO + ["#888888"]
    for (col, lbl), color in zip(STAGES, palette):
        vals = df[col].fillna(0).values
        ax.barh(labels, vals, left=bottom, color=color, label=lbl)
        bottom = bottom + vals
    ax.set_xlabel("Seconds")
    ax.set_title("Per-topic clustering stage runtime breakdown")
    ax.legend(loc="lower right", ncols=2, fontsize=8)
    save(fig, "G22_runtime_breakdown")


if __name__ == "__main__":
    main()
