"""Shared style, palette, and IO helpers for report figures.

All figure scripts in this folder import from this module so the figures share
a consistent look. Outputs go to ``<repo>/report/figures``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DASH = DATA / "dashboard"
TEMPORAL = DATA / "temporal"
CLUSTERS = DATA / "clusters"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

OKABE_ITO = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

TOPIC_LABELS_EN = {
    "social": "Social",
    "politica": "Politics",
    "international": "International",
    "sport": "Sport",
    "justitie": "Justice",
    "economie": "Economy",
    "sanatate": "Health",
    "diverse": "Diverse",
    "cultura": "Culture",
    "stiinta": "Science",
    "educatie": "Education",
    "necunoscut": "Unknown",
}

METHOD_LABELS = {
    "tfidf_kmeans": "TF-IDF + KMeans",
    "sbert_kmeans": "SBERT + KMeans",
    "sbert_hdbscan": "SBERT + HDBSCAN",
}

TIMESTAMP_SOURCE_LABELS = {
    "url": "URL",
    "text": "Article text",
    "htmldate": "htmldate",
    "missing": "Missing",
}


def setup_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": ["DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
    })


def save(fig, name: str) -> Path:
    setup_style()
    out_png = OUT / f"{name}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png


def topic_label(t: str) -> str:
    return TOPIC_LABELS_EN.get(str(t), str(t).title())


def method_label(m: str) -> str:
    return METHOD_LABELS.get(str(m), str(m))


def topic_order(df: pd.DataFrame, col: str = "topic_group", by: str | None = None) -> list[str]:
    """Return topics sorted descending by ``by`` column (default: row count)."""
    if by is None:
        return df[col].value_counts().index.tolist()
    return df.sort_values(by, ascending=False)[col].tolist()


def shorten(text: str, n: int = 60) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"
