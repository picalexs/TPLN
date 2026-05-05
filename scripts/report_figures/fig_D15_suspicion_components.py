"""Figure D15 - Suspicion and campaign-candidate score breakdown.

The temporal stage first computes a broad ``suspicion_score``. The report then
applies campaign-candidate weights that require support, recurrence, active
days, source diversity, and a public-affairs narrative signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import OKABE_ITO, ROOT, TEMPORAL, save, setup_style, shorten, topic_label

sys.path.insert(0, str(ROOT))
from src.campaign_scoring import add_campaign_candidate_columns

TOP_K = 12

TEMPORAL_WEIGHT_COLS = [
    ("support_weight", "Temporal support"),
    ("coverage_weight", "Timestamp coverage"),
    ("source_weight", "Timestamp source"),
    ("domain_weight", "Domain balance"),
]
CAMPAIGN_WEIGHT_COLS = [
    ("campaign_support_weight", "Articles"),
    ("campaign_recurrence_weight", "Recurrence"),
    ("campaign_active_days_weight", "Active days"),
    ("campaign_source_diversity_weight", "Sources"),
    ("campaign_span_weight", "Span"),
    ("campaign_narrative_weight", "Narrative"),
]


def main() -> None:
    setup_style()
    df = pd.read_parquet(TEMPORAL / "cluster_temporal_stats.parquet")
    if "campaign_candidate_score" not in df.columns:
        df = add_campaign_candidate_columns(df)
    df = (
        df.sort_values("campaign_candidate_score", ascending=False)
        .head(TOP_K)
        .iloc[::-1]
        .reset_index(drop=True)
    )

    temporal_weights = np.ones(len(df))
    for col, _ in TEMPORAL_WEIGHT_COLS:
        temporal_weights = temporal_weights * df[col].fillna(1.0).values
    after_temporal_gate = df["suspicion_score_raw"].fillna(0).values * temporal_weights
    raw = df["suspicion_score_raw"].fillna(0).values
    suspicion = df["suspicion_score"].fillna(0).values
    campaign = df["campaign_candidate_score"].fillna(0).values

    labels = [
        f"[{topic_label(t)}] {shorten(title, 50)}"
        for t, title in zip(df["topic_group"], df["representative_title"].fillna(""))
    ]
    y = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(13.5, 0.58 * TOP_K + 1.6),
        gridspec_kw={"width_ratios": [1.7, 1.05]},
    )

    height = 0.20
    ax1.barh(y - 1.5 * height, raw, height=height, color=OKABE_ITO[5], alpha=0.85, label="Raw burst score")
    ax1.barh(y - 0.5 * height, after_temporal_gate, height=height, color=OKABE_ITO[2], alpha=0.85, label="After temporal gates")
    ax1.barh(y + 0.5 * height, suspicion, height=height, color=OKABE_ITO[3], alpha=0.85, label="Suspicion score")
    ax1.barh(y + 1.5 * height, campaign, height=height, color=OKABE_ITO[0], alpha=0.95, label="Campaign candidate score")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Score")
    ax1.set_title("Broad burst score vs. stricter campaign score")
    ax1.legend(loc="lower right", fontsize=8)

    heat_cols = TEMPORAL_WEIGHT_COLS + CAMPAIGN_WEIGHT_COLS
    heat = np.column_stack([df[c].fillna(0).values for c, _ in heat_cols])
    im = ax2.imshow(heat, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels([""] * len(y))
    ax2.set_xticks(range(len(heat_cols)))
    ax2.set_xticklabels([label for _, label in heat_cols], rotation=30, ha="right")
    ax2.axvline(len(TEMPORAL_WEIGHT_COLS) - 0.5, color="white", linewidth=2)
    ax2.set_title("Temporal gates and campaign filters")
    ax2.grid(False)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat[i, j]
            ax2.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if v < 0.5 else "black",
            )
    fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02, label="Weight")

    fig.suptitle(f"Score breakdown for top {TOP_K} compact campaign candidates", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "D15_suspicion_components")


if __name__ == "__main__":
    main()
