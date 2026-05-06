"""Figure D13 - Daily timelines of top non-COVID compact campaign candidates.

Shows the daily article-count timeline for the top clusters by
``campaign_candidate_score`` so the burst pattern remains visible while the
case studies avoid COVID dominating the examples.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import pandas as pd

from _common import DASH, OKABE_ITO, ROOT, TEMPORAL, save, setup_style, shorten, topic_label

sys.path.insert(0, str(ROOT))
from src.campaign_scoring import add_campaign_candidate_columns

N_PANELS = 3
COVID_MARKERS = (
    "covid",
    "coronavirus",
    "vaccin",
    "anticovid",
    "carantin",
    "pcr",
    "pandemie",
    "imuniz",
    "certificat covid",
    "certificat verde",
)


def _normalize(value: object) -> str:
    text = "" if value is None else str(value).lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _is_covid_candidate(title: object) -> bool:
    text = _normalize(title)
    return any(marker in text for marker in COVID_MARKERS)


def main() -> None:
    setup_style()
    ts = pd.read_parquet(TEMPORAL / "cluster_temporal_stats.parquet")
    if "campaign_candidate_score" not in ts.columns:
        ts = add_campaign_candidate_columns(ts)
    ts["score"] = ts["campaign_candidate_score"]
    ts = ts[(ts["score"] > 0) & ~ts["representative_title"].map(_is_covid_candidate)]
    daily = pd.read_parquet(DASH / "cluster_daily_counts.parquet")
    daily["date"] = pd.to_datetime(daily["date"])

    chosen = ts.sort_values("score", ascending=False).head(N_PANELS)

    fig, axes = plt.subplots(N_PANELS, 1, figsize=(10.5, 2.6 * N_PANELS + 0.6))
    if N_PANELS == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, chosen.iterrows()):
        sub = daily[
            (daily["topic_group"] == row["topic_group"])
            & (daily["cluster"] == row["cluster"])
        ].sort_values("date")
        ax.bar(sub["date"], sub["article_count"], width=1.0, color=OKABE_ITO[0], alpha=0.75)
        ax.set_ylabel("Articles / day")
        title = (
            f"[{topic_label(row['topic_group'])}] "
            f"{shorten(str(row['representative_title']), 80)}  "
            f"(campaign = {row['score']:.2f}, "
            f"suspicion = {row['suspicion_score']:.2f}, "
            f"span = {int(row['span_days'])}d)"
        )
        ax.set_title(title, fontsize=9.5, loc="left")
    axes[-1].set_xlabel("Date")
    fig.suptitle("Daily timelines of top non-COVID compact campaign candidates", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "D13_top_cluster_timelines")


if __name__ == "__main__":
    main()
