"""Figure D16 - Campaign candidate narrative types.

Groups compact campaign candidates into transparent keyword-based narrative
types. The categories are heuristic, intended for report exploration rather
than ground-truth labeling.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import OKABE_ITO, TEMPORAL, save, setup_style, shorten


TYPE_RULES = [
    (
        "COVID / vaccination",
        (
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
        ),
    ),
    (
        "Ukraine / Russia war",
        (
            "ucraina",
            "ucrainei",
            "ucrainean",
            "rusia",
            "rus",
            "kiev",
            "zelenski",
            "mariupol",
            "razboi",
            "negocieri",
            "criza alimentara",
            "cereale",
            "wagner",
        ),
    ),
    (
        "Energy / economy",
        (
            "gaz",
            "energie",
            "carburan",
            "tarif",
            "compensati",
            "taxa de 9 euro",
            "curse avia",
            "wizz air",
            "cardurile sociale",
            "termoelectrica",
        ),
    ),
    (
        "Governance / elections",
        (
            "aleger",
            "electoral",
            "partid",
            "parlament",
            "guvern",
            "presed",
            "maia sandu",
            "dodon",
            "campanie",
            "cec",
            "anticipate",
            "referendum",
            "licentelor tv",
            "cnesp",
            "stare de urgenta",
        ),
    ),
    (
        "Justice / corruption",
        (
            "procur",
            "anticorup",
            "dosar",
            "arest",
            "judec",
            "urmaririi penale",
            "stoianoglo",
            "tariceanu",
            "tate",
            "avocati",
            "greva",
        ),
    ),
    (
        "Security / propaganda",
        (
            "propagand",
            "dezinform",
            "securitate",
            "supraveghere aeriana",
            "talibanii",
            "mercenari",
            "refugiati",
            "frontiera",
        ),
    ),
    (
        "Public health, non-COVID",
        (
            "sanatate",
            "medicina",
            "universitatile de medicina",
            "spital",
        ),
    ),
    (
        "Travel / mobility",
        (
            "trafic",
            "zbor",
            "aeroport",
            "intrare",
            "acces",
            "repatriere",
            "calatorie",
            "conditiile de acces",
        ),
    ),
]


def _normalize(value: object) -> str:
    text = "" if value is None else str(value).lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def classify_candidate_type(title: object) -> str:
    text = _normalize(title)
    for label, needles in TYPE_RULES:
        if any(needle in text for needle in needles):
            return label
    return "Other / mixed public narratives"


def main() -> None:
    setup_style()
    df = pd.read_parquet(TEMPORAL / "cluster_temporal_stats.parquet")
    df = df[df["campaign_candidate_score"].fillna(0) > 0].copy()
    df["candidate_type"] = df["representative_title"].map(classify_candidate_type)

    summary = (
        df.groupby("candidate_type", as_index=False)
        .agg(
            candidate_count=("cluster", "size"),
            total_score=("campaign_candidate_score", "sum"),
            max_score=("campaign_candidate_score", "max"),
        )
        .sort_values(["candidate_count", "total_score"], ascending=False)
    )

    top_examples = (
        df.sort_values("campaign_candidate_score", ascending=False)
        .drop_duplicates("candidate_type")
        .set_index("candidate_type")["representative_title"]
    )
    summary["example"] = summary["candidate_type"].map(top_examples).fillna("")
    summary = summary.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(summary))
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(13.5, max(4.8, 0.62 * len(summary) + 1.2)),
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )

    colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(summary))]
    ax1.barh(y, summary["candidate_count"], color=colors, alpha=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels(summary["candidate_type"])
    ax1.set_xlabel("Number of compact candidates")
    ax1.set_title("Candidate types by count")
    for yi, count in zip(y, summary["candidate_count"]):
        ax1.text(count, yi, f"  {int(count)}", va="center", fontsize=8)

    ax2.barh(y, summary["total_score"], color=colors, alpha=0.9)
    ax2.set_yticks(y)
    ax2.set_yticklabels([""] * len(summary))
    ax2.set_xlabel("Total campaign-candidate score")
    ax2.set_title("Same types weighted by score")
    for yi, row in summary.iterrows():
        label = shorten(str(row["example"]), 58)
        ax2.text(row["total_score"], yi, f"  e.g. {label}", va="center", fontsize=7.5)

    fig.suptitle("Compact campaign candidates by narrative type", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, "D16_candidate_types")


if __name__ == "__main__":
    main()
