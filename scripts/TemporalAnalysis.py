"""
Temporal Analysis & Burst Detection
==============================================
Reads the clustered parquet, builds time-series for every real cluster,
applies Kleinberg burst detection at DAILY and WEEKLY granularity,
and computes temporal concentration metrics with improved scoring.

Outputs:
    data/temporal/cluster_temporal_stats.parquet
"""

import argparse
import sys
import math
from pathlib import Path
import warnings
from urllib.parse import urlparse
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paths import CLUSTERED_PARQUET, TEMPORAL_DIR, TEMPORAL_STATS
from src.io_utils import load_clean_data
from src.runtime_profile import apply_runtime_profile, detect_runtime_profile, format_runtime_profile

warnings.filterwarnings("ignore")

# Ensure cluster titles with non-ASCII characters do not crash Windows console output.
stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temporal burst analysis over clustered_data.parquet.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Override CPU thread count for CPU-bound work.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS (safe to keep at top level)
# ---------------------------------------------------------------------------
TEMPORAL_STATS_COLUMNS = [
    "cluster",
    "topic_group",
    "total_articles",
    "timestamped_articles",
    "timestamp_coverage_ratio",
    "timestamp_source_url_share",
    "timestamp_source_text_share",
    "timestamp_source_missing_share",
    "timestamp_source_reliability",
    "domain_count",
    "top_domain_share",
    "timestamped_domain_count",
    "top_timestamp_domain_share",
    "domain_entropy",
    "first_seen",
    "last_seen",
    "span_days",
    "temporal_spread_days",
    "active_days",
    "active_day_ratio",
    "median_gap_days",
    "max_gap_days",
    "days_per_timestamped_article",
    "compactness_index",
    "peak_day_count",
    "peak_day_share",
    "peak_to_baseline_ratio",
    "burst_score_daily",
    "burst_duration_daily",
    "burst_periods_daily",
    "burst_duration_share_daily",
    "burst_score_weekly",
    "burst_duration_weekly",
    "burst_periods_weekly",
    "burst_duration_share_weekly",
    "burst_stable",
    "concentration",
    "support_weight",
    "coverage_weight",
    "source_weight",
    "domain_weight",
    "long_sparse_span_penalty",
    "single_domain_penalty",
    "source_reliability_penalty",
    "suspicion_score_raw",
    "suspicion_penalty_total",
    "suspicion_score",
    "representative_title",
    "burst_score",
    "burst_duration_days",
    "article_count",
    "num_burst_periods",
]

# Kleinberg parameters
BURST_S = 2.0
BURST_GAMMA = 1.0

# Minimum articles for meaningful burst detection
MIN_ARTICLES_FOR_BURST = 3


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _normalize_domain(url_value):
    if pd.isna(url_value):
        return "unknown_domain"

    parsed = urlparse(str(url_value))
    domain = (parsed.netloc or "").strip().lower()
    if not domain:
        return "unknown_domain"

    if "@" in domain:
        domain = domain.split("@", 1)[-1]

    if ":" in domain:
        domain = domain.split(":", 1)[0]

    if domain.startswith("www."):
        domain = domain[4:]

    return domain or "unknown_domain"


def _normalized_entropy(series):
    if series.empty:
        return 0.0

    counts = series.value_counts()
    total = counts.sum()
    if total <= 0:
        return 0.0

    probs = counts / total
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 0.0
    if max_entropy <= 0:
        return 0.0
    return entropy / max_entropy


def _ensure_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp", "timestamp_source", "url"]
    missing = [column for column in required if column not in df.columns]
    if not missing:
        return df

    key_columns = ["title", "document", "short_document", "topics"]
    if not all(column in df.columns for column in key_columns):
        for column in missing:
            if column == "timestamp":
                df[column] = pd.NaT
            elif column == "timestamp_source":
                df[column] = "missing"
            else:
                df[column] = ""
        return df

    clean_columns = key_columns + missing
    clean_df = load_clean_data(columns=clean_columns)
    clean_subset = clean_df[key_columns + missing].drop_duplicates(subset=key_columns, keep="first")
    merged = df.merge(clean_subset, on=key_columns, how="left", suffixes=("", "_clean"))

    for column in missing:
        clean_column = f"{column}_clean"
        if column in merged.columns and clean_column in merged.columns:
            merged[column] = merged[column].combine_first(merged[clean_column])
            merged = merged.drop(columns=[clean_column])
        elif clean_column in merged.columns:
            merged[column] = merged[clean_column]
            merged = merged.drop(columns=[clean_column])

    return merged


# ---------------------------------------------------------------------------
# KLEINBERG BURST DETECTION
# ---------------------------------------------------------------------------
def kleinberg_burst(event_times, s=2.0, gamma=1.0, freq="D"):
    """
    Simplified Kleinberg burst detector.

    freq: 'D' for daily, 'W' for weekly.

    Uses 'W-MON' period anchoring and '7D' date_range increments for weekly
    to avoid the deprecated 'W' alias (removed in pandas 2.2) and to ensure
    consistent 7-day alignment between floored timestamps and the full index.

    Returns: (burst_score, burst_periods)
    """
    if len(event_times) < MIN_ARTICLES_FOR_BURST:
        return 0, []

    # Fixed freq strings: '7D' for weekly, 'D' for daily.
    rng_freq = "7D" if freq == "W" else "D"

    # Try library first
    try:
        import burst_detection as bd  # type: ignore[import-not-found]
        dates = event_times.sort_values()
        if freq == "W":
            floored = dates.dt.to_period("W-MON").apply(lambda p: p.start_time)
        else:
            floored = dates.dt.floor("D")
        full_index = pd.date_range(floored.min(), floored.max(), freq=rng_freq)
        counts = floored.value_counts().sort_index().reindex(full_index, fill_value=0)
        r = counts.values.astype(float)
        n = len(r)
        d = len(full_index)

        if n < 2 or d < 2:
            return 0, []

        q = bd.burst_detection(r, d, s=s, gamma=gamma)
        level_max = int(np.max(q)) if len(q) > 0 else 0

        periods = []
        in_burst = False
        burst_start = None
        burst_level = 0
        for i, lv in enumerate(q):
            if i >= len(full_index):
                break
            date = full_index[i]
            if lv >= 1 and not in_burst:
                in_burst = True
                burst_start = date
                burst_level = int(lv)
            elif lv >= 1 and in_burst:
                burst_level = max(burst_level, int(lv))
            elif lv < 1 and in_burst:
                periods.append((burst_start, full_index[i - 1], burst_level))
                in_burst = False
        if in_burst:
            periods.append((burst_start, full_index[min(len(q), len(full_index)) - 1], burst_level))

        return level_max, periods
    except (ImportError, Exception):
        pass

    # Manual Kleinberg (two-state)
    dates = event_times.sort_values()

    if freq == "W":
        floored = dates.dt.to_period("W-MON").apply(lambda p: p.start_time)
    else:
        floored = dates.dt.floor("D")

    first_day = floored.min()
    last_day = floored.max()

    all_periods = pd.date_range(first_day, last_day, freq=rng_freq)
    T = len(all_periods)

    if T < 2:
        return 0, []

    n = len(dates)
    rate_base = n / T
    rate_burst = rate_base * s

    daily = floored.value_counts().sort_index()
    counts = daily.reindex(all_periods, fill_value=0).values.astype(float)

    levels = []
    for c in counts:
        if rate_burst > 0 and rate_base > 0:
            ll_burst = c * math.log(rate_burst + 1e-10) - rate_burst
            ll_base = c * math.log(rate_base + 1e-10) - rate_base
            score = ll_burst - ll_base - gamma
            levels.append(1 if score > 0 else 0)
        else:
            levels.append(0)

    burst_score = max(levels) if levels else 0

    periods = []
    in_burst = False
    burst_start = None
    for i, lv in enumerate(levels):
        date = all_periods[i]
        if lv == 1 and not in_burst:
            in_burst = True
            burst_start = date
        elif lv == 0 and in_burst:
            periods.append((burst_start, all_periods[i - 1], 1))
            in_burst = False
    if in_burst:
        periods.append((burst_start, all_periods[-1], 1))

    return burst_score, periods


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    profile = detect_runtime_profile(device="cpu", cpu_threads=args.cpu_threads)
    apply_runtime_profile(profile)
    print(format_runtime_profile(profile))

    TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)

    if not CLUSTERED_PARQUET.exists():
        raise FileNotFoundError(
            f"Clustered dataset not found at {CLUSTERED_PARQUET}.\n"
            "Run scripts/EmbeddingsClustering.py first."
        )

    df = pd.read_parquet(CLUSTERED_PARQUET)
    print(f"Loaded {len(df)} rows from {CLUSTERED_PARQUET}")

    df = _ensure_temporal_columns(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp_source"] = (
        df.get("timestamp_source", pd.Series(index=df.index, dtype="object"))
        .fillna("missing")
        .astype(str)
        .str.lower()
    )
    df["domain"] = df["url"].apply(_normalize_domain) if "url" in df.columns else "unknown_domain"

    # Filter: valid timestamp AND real cluster
    df_ts = df[(df["timestamp"].notna()) & (df["cluster"] != -1)].copy()

    print(f"\nTimestamp + cluster coverage:")
    print(f"  Total rows:                    {len(df)}")
    print(f"  Rows with valid timestamp:     {df['timestamp'].notna().sum()}")
    print(f"  Rows in real clusters:         {(df['cluster'] != -1).sum()}")
    print(f"  Rows with BOTH (for analysis): {len(df_ts)}")

    if len(df_ts) == 0:
        print("\nWARNING: No rows with both timestamp and cluster.")
        pd.DataFrame(columns=TEMPORAL_STATS_COLUMNS).to_parquet(
            TEMPORAL_STATS,
            index=False,
            compression="zstd",
        )
        print(f"Wrote empty stats file with expected columns to: {TEMPORAL_STATS}")
        return 0

    cluster_ids = sorted(df_ts["cluster"].unique())
    print(f"Real clusters with timestamps: {len(cluster_ids)}")

    clustered_df = df[df["cluster"] != -1].copy()
    grouped_all = clustered_df.groupby("cluster", sort=True)
    grouped_ts = df_ts.groupby("cluster", sort=True)

    # ---------------------------------------------------------------------------
    # COMPUTE STATS PER CLUSTER
    # ---------------------------------------------------------------------------
    print("\nComputing temporal stats per cluster...")
    records = []

    for i, cluster_id in enumerate(cluster_ids):
        sub_all = grouped_all.get_group(cluster_id)
        sub = grouped_ts.get_group(cluster_id)
        times = sub["timestamp"]

        article_count_ts = len(sub)  # articles WITH timestamp
        total_articles = len(sub_all)
        timestamp_coverage_ratio = article_count_ts / max(total_articles, 1)
        first_seen = times.min()
        last_seen = times.max()
        span_days = max((last_seen - first_seen).days, 1)

        # Temporal spread
        temporal_spread = times.apply(lambda t: (t - first_seen).days).std()
        if pd.isna(temporal_spread):
            temporal_spread = 0.0

        daily_counts = times.dt.floor("D").value_counts().sort_index()
        active_days = int(len(daily_counts))
        active_day_ratio = _safe_div(active_days, span_days)
        peak_day_count = int(daily_counts.max()) if active_days > 0 else 0
        peak_day_share = peak_day_count / max(article_count_ts, 1)
        baseline_per_day = article_count_ts / max(span_days, 1)
        peak_to_baseline_ratio = peak_day_count / max(baseline_per_day, 1e-9)

        sorted_days = daily_counts.index.sort_values()
        day_gaps = [max((sorted_days[j] - sorted_days[j - 1]).days - 1, 0) for j in range(1, len(sorted_days))]
        median_gap_days = float(np.median(day_gaps)) if day_gaps else 0.0
        max_gap_days = float(max(day_gaps)) if day_gaps else 0.0
        days_per_timestamped_article = _safe_div(span_days, article_count_ts)
        compactness_index = active_day_ratio * timestamp_coverage_ratio

        all_domain_counts = sub_all["domain"].fillna("unknown_domain").value_counts()
        ts_domain_counts = sub["domain"].fillna("unknown_domain").value_counts()
        domain_count = int(all_domain_counts.size)
        top_domain_share = _safe_div(int(all_domain_counts.iloc[0]) if not all_domain_counts.empty else 0, total_articles)
        timestamped_domain_count = int(ts_domain_counts.size)
        top_timestamp_domain_share = _safe_div(int(ts_domain_counts.iloc[0]) if not ts_domain_counts.empty else 0, article_count_ts)
        domain_entropy = round(_normalized_entropy(sub_all["domain"].fillna("unknown_domain")), 4)

        source_counts = sub_all["timestamp_source"].fillna("missing").astype(str).str.lower().value_counts()
        url_count = int(source_counts.get("url", 0))
        text_count = int(source_counts.get("text", 0))
        missing_count = int(source_counts.get("missing", 0))
        timestamp_source_url_share = _safe_div(url_count, total_articles)
        timestamp_source_text_share = _safe_div(text_count, total_articles)
        timestamp_source_missing_share = _safe_div(missing_count, total_articles)
        timestamp_source_reliability = timestamp_source_url_share + (0.35 * timestamp_source_text_share)

        # ---- DAILY burst detection ----
        burst_daily, burst_periods_daily = kleinberg_burst(times, s=BURST_S, gamma=BURST_GAMMA, freq="D")
        burst_dur_daily = sum((end - start).days + 1 for start, end, _ in burst_periods_daily)

        # ---- WEEKLY burst detection ----
        burst_weekly, burst_periods_weekly = kleinberg_burst(times, s=BURST_S, gamma=BURST_GAMMA, freq="W")
        burst_dur_weekly = sum(max((end - start).days // 7 + 1, 1) for start, end, _ in burst_periods_weekly)

        # Burst stability: burst found at BOTH granularities = more suspicious
        burst_stable = 1 if (burst_daily > 0 and burst_weekly > 0) else 0

        # Topic
        topic = "unknown"
        if "topic_group" in sub.columns:
            mode = sub["topic_group"].mode()
            topic = mode.iloc[0] if not mode.empty else "unknown"

        # Representative title — safe column check
        rep_title = ""
        if len(sub_all) > 0 and "title" in sub_all.columns:
            rep_title = str(sub_all.iloc[0]["title"])[:200]

        # Concentration: timestamped articles / span_days
        concentration = article_count_ts / max(span_days, 1)

        # Confidence weights keep sparse / low-coverage clusters from over-ranking.
        # Calibrated for the observed corpus where median timestamp coverage is
        # only ~14% and many clusters span a long calendar range. The previous
        # calibration (coverage base 0.35, linear slope 0.65) drove the median
        # suspicion_score to zero because coverage_weight alone started at ~0.44.
        support_weight = min(1.0, article_count_ts / 10.0)
        coverage_weight = min(1.0, 0.55 + (0.45 * timestamp_coverage_ratio))
        source_weight = 0.55 + (0.45 * min(1.0, timestamp_source_reliability))
        domain_weight = 0.60 + (0.40 * min(1.0, 1.0 - max(0.0, top_domain_share - 0.5)))
        burst_duration_share_daily = burst_dur_daily / max(span_days, 1)
        burst_duration_share_weekly = min((burst_dur_weekly * 7) / max(span_days, 1), 1.0)

        # Explicit penalties for cases that were over-ranking before.
        # The long-sparse penalty previously blew up at ~46 points for a median
        # cluster (span ~530 days, active_day_ratio ~0.01, coverage ~0.14),
        # overwhelming any suspicion signal. We cap the (span_days - 90)/90
        # ratio and halve the multiplier so the penalty still discourages
        # sprawling clusters without annihilating the score.
        long_sparse_span_penalty = (
            math.log1p(span_days)
            * min(3.0, max(0.0, span_days - 90) / 90.0)
            * (1.0 - active_day_ratio)
            * (1.0 - timestamp_coverage_ratio)
            * 0.6
        )
        single_domain_penalty = (
            max(0.0, top_domain_share - 0.5) * 4.0
            + (1.75 if domain_count == 1 else 0.0)
            + max(0.0, top_timestamp_domain_share - 0.6) * 2.0
        )
        source_reliability_penalty = (1.0 - timestamp_source_reliability) * 1.2
        compactness_penalty = max(0.0, 0.20 - active_day_ratio) * 3.0
        penalty_total = long_sparse_span_penalty + single_domain_penalty + source_reliability_penalty + compactness_penalty

        burst_signal = (
            (burst_daily * 4.0)
            + (burst_weekly * 3.0)
            + (burst_stable * 4.0)
        )
        concentration_signal = math.log1p(concentration) * 2.0
        peak_signal = (math.log1p(peak_to_baseline_ratio) * 2.5) + (peak_day_share * 3.0)
        duration_signal = (burst_duration_share_daily * 2.5) + (burst_duration_share_weekly * 1.5)
        size_signal = math.log1p(total_articles) * 0.75
        support_penalty = max(0, 6 - article_count_ts) * 0.75

        suspicion_raw = (
            burst_signal
            + concentration_signal
            + peak_signal
            + duration_signal
            + size_signal
        )
        suspicion_score = max(
            0.0,
            (suspicion_raw * support_weight * coverage_weight * source_weight * domain_weight)
            - support_penalty
            - penalty_total,
        )
        suspicion_score = round(suspicion_score, 3)

        records.append({
            "cluster": cluster_id,
            "topic_group": topic,
            "total_articles": total_articles,
            "timestamped_articles": article_count_ts,
            "timestamp_coverage_ratio": round(timestamp_coverage_ratio, 4),
            "timestamp_source_url_share": round(timestamp_source_url_share, 4),
            "timestamp_source_text_share": round(timestamp_source_text_share, 4),
            "timestamp_source_missing_share": round(timestamp_source_missing_share, 4),
            "timestamp_source_reliability": round(timestamp_source_reliability, 4),
            "domain_count": domain_count,
            "top_domain_share": round(top_domain_share, 4),
            "timestamped_domain_count": timestamped_domain_count,
            "top_timestamp_domain_share": round(top_timestamp_domain_share, 4),
            "domain_entropy": domain_entropy,
            "first_seen": first_seen.date() if pd.notna(first_seen) else None,
            "last_seen": last_seen.date() if pd.notna(last_seen) else None,
            "span_days": span_days,
            "temporal_spread_days": round(temporal_spread, 2),
            "active_days": active_days,
            "active_day_ratio": round(active_day_ratio, 4),
            "median_gap_days": round(median_gap_days, 2),
            "max_gap_days": round(max_gap_days, 2),
            "days_per_timestamped_article": round(days_per_timestamped_article, 4),
            "compactness_index": round(compactness_index, 4),
            "peak_day_count": peak_day_count,
            "peak_day_share": round(peak_day_share, 4),
            "peak_to_baseline_ratio": round(peak_to_baseline_ratio, 4),
            "burst_score_daily": burst_daily,
            "burst_duration_daily": burst_dur_daily,
            "burst_periods_daily": len(burst_periods_daily),
            "burst_duration_share_daily": round(burst_duration_share_daily, 4),
            "burst_score_weekly": burst_weekly,
            "burst_duration_weekly": burst_dur_weekly,
            "burst_periods_weekly": len(burst_periods_weekly),
            "burst_duration_share_weekly": round(burst_duration_share_weekly, 4),
            "burst_stable": burst_stable,
            "concentration": round(concentration, 4),
            "support_weight": round(support_weight, 4),
            "coverage_weight": round(coverage_weight, 4),
            "source_weight": round(source_weight, 4),
            "domain_weight": round(domain_weight, 4),
            "long_sparse_span_penalty": round(long_sparse_span_penalty, 4),
            "single_domain_penalty": round(single_domain_penalty, 4),
            "source_reliability_penalty": round(source_reliability_penalty, 4),
            "suspicion_score_raw": round(suspicion_raw, 4),
            "suspicion_penalty_total": round(penalty_total + support_penalty, 4),
            "suspicion_score": suspicion_score,
            "representative_title": rep_title,
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(cluster_ids)} clusters...")

    stats_df = pd.DataFrame(records).sort_values(
        "suspicion_score", ascending=False
    ).reset_index(drop=True)

    # Backward-compatible column aliases for dashboard
    stats_df["burst_score"] = stats_df["burst_score_daily"]
    stats_df["burst_duration_days"] = stats_df["burst_duration_daily"]
    stats_df["article_count"] = stats_df["total_articles"]
    stats_df["num_burst_periods"] = stats_df["burst_periods_daily"]

    # ---------------------------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEMPORAL STATS - TOP 20 MOST SUSPICIOUS CLUSTERS")
    print("=" * 60)
    for _, row in stats_df.head(20).iterrows():
        title = str(row.get("representative_title", ""))[:60]
        print(
            f"  Cluster {int(row['cluster']):4d} | {str(row['topic_group']):15s} "
            f"| total={int(row['total_articles']):5d} | ts={int(row['timestamped_articles']):4d} "
            f"| cov={row['timestamp_coverage_ratio']:.2f} | act={row['active_day_ratio']:.3f} "
            f"| doms={int(row['domain_count']):2d} topdom={row['top_domain_share']:.2f} "
            f"| src_rel={row['timestamp_source_reliability']:.2f} "
            f"| burst_d={int(row['burst_score_daily'])} burst_w={int(row['burst_score_weekly'])} "
            f"| peakx={row['peak_to_baseline_ratio']:.1f} "
            f"| pen={row['suspicion_penalty_total']:.1f} "
            f"| susp={row['suspicion_score']:.1f} | {title}"
        )

    print(f"\nTotal clusters analyzed:        {len(stats_df)}")
    print(f"Clusters with daily burst > 0:  {(stats_df['burst_score_daily'] > 0).sum()}")
    print(f"Clusters with weekly burst > 0: {(stats_df['burst_score_weekly'] > 0).sum()}")
    print(f"Clusters stable (both):         {stats_df['burst_stable'].sum()}")
    print(f"Max suspicion score:            {stats_df['suspicion_score'].max()}")
    print(f"Median active day ratio:        {stats_df['active_day_ratio'].median():.4f}")
    print(f"Median source reliability:      {stats_df['timestamp_source_reliability'].median():.4f}")

    # ---------------------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------------------
    stats_df.to_parquet(TEMPORAL_STATS, index=False, compression="zstd")
    print(f"\nSaved to: {TEMPORAL_STATS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
