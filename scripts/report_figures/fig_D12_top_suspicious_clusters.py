"""Figure D12 - Top-K campaign candidates.

Horizontal bar chart of the highest-scoring clusters by
``campaign_candidate_score``. This stricter report score keeps the temporal
burst signal but down-ranks obvious organic news events.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import pandas as pd

from _common import OKABE_ITO, ROOT, TEMPORAL, save, setup_style, shorten, topic_label

sys.path.insert(0, str(ROOT))
from src.campaign_scoring import add_campaign_candidate_columns

TOP_K = 15


def main() -> None:
    setup_style()
    ts = pd.read_parquet(TEMPORAL / "cluster_temporal_stats.parquet")
    if "campaign_candidate_score" not in ts.columns:
        ts = add_campaign_candidate_columns(ts)
    ts["score"] = ts["campaign_candidate_score"]
    df = ts.sort_values("score", ascending=False).head(TOP_K).iloc[::-1]

    labels = [
        f"[{topic_label(t)}] {shorten(title, 70)}"
        for t, title in zip(df["topic_group"], df["representative_title"].fillna(""))
    ]
    fig, ax = plt.subplots(figsize=(10, 0.45 * TOP_K + 1.0))
    bars = ax.barh(labels, df["score"], color=OKABE_ITO[3])
    for bar, value, n, span in zip(bars, df["score"], df["article_count"], df["span_days"]):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f"  {value:.2f}  ({int(n) if pd.notna(n) else 0} art., {int(span)}d)",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("Campaign candidate score")
    ax.set_title(f"Top {TOP_K} compact campaign candidates")
    save(fig, "D12_top_suspicious_clusters")


if __name__ == "__main__":
    main()
