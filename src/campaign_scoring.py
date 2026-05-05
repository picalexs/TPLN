"""Post-hoc campaign-candidate scoring helpers.

The temporal score is intentionally broad: it surfaces bursty multi-source
clusters. That includes organic news events such as accidents, weather alerts,
routine health counters, and police briefs. For report headline figures we need
a stricter score that asks: does this look like a compact public narrative?
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd


HARD_EVENT_TERMS = (
    "accident",
    "decedat",
    "murit",
    "moarta",
    "mort",
    "ucis",
    "omor",
    "omorat",
    "spital",
    "victim",
    "ranit",
    "incendiu",
    "ars",
    "inecat",
    "cadavru",
    "violat",
    "violata",
    "cutit",
    "batut",
    "bataie",
    "jefui",
    "hot",
    "hoti",
    "furt",
    "sparger",
    "meteo",
    "cod galben",
    "ploi",
    "ninso",
    "furtun",
    "cutremur",
    "zapada",
    "cazuri noi",
    "noi cazuri",
    "bilant",
    "decese",
    "prezenta la vot",
    "alerta cu bomba",
    "bomba a explodat",
    "alcool metilic",
    "gaze lacrimogene",
    "scandal nocturn",
    "drog",
    "focar",
    "infectie",
)

ROUTINE_HEALTH_TERMS = (
    "cazuri",
    "decese",
    "decedat",
    "bilant",
    "infectari",
    "infectare",
    "internati",
    "pacienti",
    "respiratie asistata",
    "stare grava",
    "donat plasma",
    "doneze plasma",
    "donare",
    "plasma",
    "doze",
    "lot",
    "punctele mobile",
    "amplasate",
    "programate",
    "prezentat pentru vaccinare",
    "campania de vaccinare",
    "maratonul de vaccinare",
    "test pcr",
    "diamond princess",
    "nava",
    "croaziera",
    "pasagerii",
)


PUBLIC_AFFAIRS_TERMS = (
    "guvern",
    "parlament",
    "presed",
    "minister",
    "ministr",
    "partid",
    "aleger",
    "electoral",
    "candidat",
    "campanie",
    "opozit",
    "socialist",
    "pas",
    "sor",
    "dodon",
    "sandu",
    "tauber",
    "nastase",
    "ambasad",
    "nato",
    "rusia",
    "rus",
    "ucraina",
    "gaz",
    "energie",
    "carburan",
    "tarif",
    "pret",
    "protest",
    "sancti",
    "propagand",
    "dezinform",
    "justit",
    "procur",
    "anticorup",
    "lege",
    "constitut",
    "securitate",
    "transnistr",
    "gagauz",
    "concesionare",
    "covid",
    "coronavirus",
    "vaccin",
    "vaccinare",
    "certificat",
    "carantina",
    "restricti",
    "pandemie",
    "sanatate",
)


PUBLIC_AFFAIRS_TOPICS = {"politica", "economie", "justitie"}


def _normalize_text(value: object) -> str:
    text = "" if value is None else str(value).lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    padded = f" {text} "
    return any(f" {term} " in padded for term in terms)


def _is_routine_health_update(text: str) -> bool:
    """Identify routine COVID/health status updates without blocking narratives."""
    if not any(marker in text for marker in ("covid", "coronavirus", "vaccin", "carantin", "pcr")):
        return False
    return _contains_any(text, ROUTINE_HEALTH_TERMS)


def add_campaign_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with report-oriented campaign-candidate columns.

    Required columns are the existing temporal-analysis outputs:
    ``suspicion_score``, ``total_articles``, ``active_days``,
    ``burst_periods_daily``, ``domain_count``, ``span_days``,
    ``topic_group``, and ``representative_title``.
    """
    out = df.copy()
    titles = out.get("representative_title", pd.Series("", index=out.index)).map(_normalize_text)
    topics = out.get("topic_group", pd.Series("", index=out.index)).fillna("").astype(str)

    out["organic_event_title"] = titles.map(
        lambda value: (
            _contains_any(value, HARD_EVENT_TERMS)
            or _is_routine_health_update(value)
        )
    )
    out["public_affairs_title"] = titles.map(lambda value: _contains_any(value, PUBLIC_AFFAIRS_TERMS))
    out["public_affairs_signal"] = out["public_affairs_title"] | topics.isin(PUBLIC_AFFAIRS_TOPICS)

    support_weight = (out["total_articles"].fillna(0) / 50.0).clip(upper=1.0)
    recurrence_weight = (out["burst_periods_daily"].fillna(0) / 5.0).clip(upper=1.0)
    active_days_weight = (out["active_days"].fillna(0) / 20.0).clip(upper=1.0)
    source_diversity_weight = (
        0.65 + 0.35 * ((out["domain_count"].fillna(0) - 1.0) / 5.0).clip(lower=0.0, upper=1.0)
    )
    # Report case studies should be compact episodes, not semantic themes that
    # reappear across multiple years. Full weight up to 180 days, zero at 365.
    span_weight = (
        1.0 - ((out["span_days"].fillna(10_000) - 180.0).clip(lower=0.0) / (365.0 - 180.0))
    ).clip(lower=0.0, upper=1.0)

    narrative_weight = pd.Series(0.55, index=out.index, dtype="float64")
    narrative_weight.loc[out["public_affairs_signal"]] = 1.0
    narrative_weight.loc[out["organic_event_title"]] = 0.0
    narrative_weight.loc[out["domain_count"].fillna(0) <= 1] = 0.0

    out["campaign_support_weight"] = support_weight.round(4)
    out["campaign_recurrence_weight"] = recurrence_weight.round(4)
    out["campaign_active_days_weight"] = active_days_weight.round(4)
    out["campaign_source_diversity_weight"] = source_diversity_weight.round(4)
    out["campaign_span_weight"] = span_weight.round(4)
    out["campaign_narrative_weight"] = narrative_weight.round(4)

    score = (
        out["suspicion_score"].fillna(0)
        * support_weight
        * recurrence_weight
        * active_days_weight
        * source_diversity_weight
        * span_weight
        * narrative_weight
    )
    out["campaign_candidate_score"] = np.maximum(score, 0).round(3)
    out["campaign_candidate"] = out["campaign_candidate_score"] > 0
    return out
