"""Figure D15 — Suspicion score waterfall + gating weights.

The temporal stage computes
``suspicion = max(0, raw × support_w × coverage_w × source_w × domain_w) − penalties``
so the score is multiplicative on the four 0-1 gating weights and additive on
the penalties. This figure shows both views for the top-K clusters by
``suspicion_score_multi_source``:

  Left  - waterfall: raw burst score, after-gating value, and final score.
  Right - heatmap of the four multiplicative gating weights and three
          penalty terms.

Reads ``data/temporal/cluster_temporal_stats.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import OKABE_ITO, TEMPORAL, save, setup_style, shorten, topic_label

TOP_K = 12

WEIGHT_COLS = [
    ("support_weight", "Support"),
    ("coverage_weight", "Coverage"),
    ("source_weight", "Source"),
    ("domain_weight", "Domain"),
]
PENALTY_COLS = [
    ("long_sparse_span_penalty", "Sparse span"),
    ("single_domain_penalty", "Single domain"),
    ("source_reliability_penalty", "Source reliability"),
]


def main() -> None:
    setup_style()
    df = pd.read_parquet(TEMPORAL / "cluster_temporal_stats.parquet")
    df["score"] = df["suspicion_score"].where(df["domain_count"] > 1, 0.0)
    df = (df.sort_values("score", ascending=False)
            .head(TOP_K).iloc[::-1].reset_index(drop=True))

    weights_prod = np.ones(len(df))
    for col, _ in WEIGHT_COLS:
        weights_prod = weights_prod * df[col].fillna(1.0).values
    after_gate = df["suspicion_score_raw"].fillna(0).values * weights_prod
    final = df["score"].fillna(0).values
    raw = df["suspicion_score_raw"].fillna(0).values

    labels = [
        f"[{topic_label(t)}] {shorten(title, 50)}"
        for t, title in zip(df["topic_group"], df["representative_title"].fillna(""))
    ]
    y = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 0.55 * TOP_K + 1.6),
        gridspec_kw={"width_ratios": [1.6, 1.0]},
    )

    height = 0.27
    ax1.barh(y - height, raw, height=height, color=OKABE_ITO[5], alpha=0.85,
             label="Raw burst score")
    ax1.barh(y, after_gate, height=height, color=OKABE_ITO[2], alpha=0.85,
             label="After multiplicative gating (weights)")
    ax1.barh(y + height, final, height=height, color=OKABE_ITO[3], alpha=0.95,
             label="Final score (after penalties)")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Score (higher = more suspicious)")
    ax1.set_title("Waterfall: raw → gated → final")
    ax1.legend(loc="lower right", fontsize=8)

    heat = np.column_stack([
        df[c].values for c, _ in WEIGHT_COLS
    ] + [
        df[c].values for c, _ in PENALTY_COLS
    ])
    im = ax2.imshow(heat, cmap="viridis", aspect="auto",
                    vmin=0, vmax=max(np.nanmax(heat), 1.0))
    ax2.set_yticks(y)
    ax2.set_yticklabels([""] * len(y))
    cols = [lbl for _, lbl in WEIGHT_COLS] + [lbl for _, lbl in PENALTY_COLS]
    ax2.set_xticks(range(len(cols)))
    ax2.set_xticklabels(cols, rotation=30, ha="right")
    ax2.axvline(len(WEIGHT_COLS) - 0.5, color="white", linewidth=2)
    ax2.set_title("Gating weights (0-1) and penalty terms")
    ax2.grid(False)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat[i, j]
            if not np.isnan(v):
                ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=7,
                         color="white" if v < 0.5 * np.nanmax(heat) else "black")
    fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02, label="Value")

    fig.suptitle(f"Suspicion score breakdown for top {TOP_K} clusters", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "D15_suspicion_components")


if __name__ == "__main__":
    main()
