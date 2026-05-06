"""Figure B8 — Per-topic silhouette and per-cluster intra-cluster cosine.

Two panels:
  Left  - global silhouette per topic (one value per topic).
  Right - boxplot of mean intra-cluster cosine per real cluster, by topic.
Reads ``data/evaluation_report.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import pandas as pd

from _common import DATA, OKABE_ITO, save, setup_style, topic_label


def main() -> None:
    setup_style()
    df = pd.read_parquet(DATA / "evaluation_report.parquet")

    sil = df[df["section"] == "silhouette"][["topic_group", "silhouette"]].dropna()
    sil = sil.sort_values("silhouette", ascending=True)

    cos = df[df["section"] == "intra_cosine"][["topic_group", "mean_intra_cosine"]].dropna()
    topic_med = (cos.groupby("topic_group")["mean_intra_cosine"]
                    .median().sort_values(ascending=False))
    topics = topic_med.index.tolist()
    box_data = [cos.loc[cos["topic_group"] == t, "mean_intra_cosine"].values for t in topics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    bars = axes[0].barh([topic_label(t) for t in sil["topic_group"]],
                        sil["silhouette"], color=OKABE_ITO[0])
    axes[0].set_xlabel("Silhouette")
    axes[0].set_title("Per-topic silhouette (best HDBSCAN config)")
    for bar, value in zip(bars, sil["silhouette"]):
        axes[0].text(value, bar.get_y() + bar.get_height() / 2,
                     f" {value:.3f}", va="center", fontsize=8)

    bp = axes[1].boxplot(box_data, vert=True, patch_artist=True, widths=0.6,
                         showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(OKABE_ITO[2])
        patch.set_alpha(0.7)
        patch.set_edgecolor("#333333")
    for med in bp["medians"]:
        med.set_color("#222222")
    axes[1].set_xticks(range(1, len(topics) + 1))
    axes[1].set_xticklabels([topic_label(t) for t in topics], rotation=35, ha="right")
    axes[1].set_ylabel("Mean intra-cluster cosine")
    axes[1].set_title("Per-cluster intra-cluster cosine, by topic")

    save(fig, "B8_silhouette_intracosine")


if __name__ == "__main__":
    main()
