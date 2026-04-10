"""
Temporal Analysis & Burst Detection
==============================================
Reads the clustered CSV, builds time-series for every real cluster,
applies Kleinberg burst detection at DAILY and WEEKLY granularity,
and computes temporal concentration metrics with improved scoring.

Outputs:
    data/temporal/cluster_temporal_stats.csv
"""

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS & CONFIG
# ---------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")

CLUSTER_CSV = os.path.join(data_dir, "clusters", "clustered_data.csv")
OUTPUT_DIR  = os.path.join(data_dir, "temporal")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cluster_temporal_stats.csv")

TEMPORAL_STATS_COLUMNS = [
    "cluster",
    "topic_group",
    "total_articles",
    "timestamped_articles",
    "first_seen",
    "last_seen",
    "span_days",
    "temporal_spread_days",
    "burst_score_daily",
    "burst_duration_daily",
    "burst_periods_daily",
    "burst_score_weekly",
    "burst_duration_weekly",
    "burst_periods_weekly",
    "burst_stable",
    "concentration",
    "suspicion_score",
    "representative_title",
    "burst_score",
    "burst_duration_days",
    "article_count",
    "num_burst_periods",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Kleinberg parameters
BURST_S = 2.0
BURST_GAMMA = 1.0

# Minimum articles for meaningful burst detection
MIN_ARTICLES_FOR_BURST = 3


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
if not os.path.exists(CLUSTER_CSV):
    raise FileNotFoundError(
        f"Clustered CSV not found at {CLUSTER_CSV}.\n"
        "Run EmbeddingsClustering.py first."
    )

df = pd.read_csv(CLUSTER_CSV, low_memory=False)
print(f"Loaded {len(df)} rows from {CLUSTER_CSV}")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Filter: valid timestamp AND real cluster
df_ts = df[(df["timestamp"].notna()) & (df["cluster"] != -1)].copy()

print(f"\nTimestamp + cluster coverage:")
print(f"  Total rows:                    {len(df)}")
print(f"  Rows with valid timestamp:     {df['timestamp'].notna().sum()}")
print(f"  Rows in real clusters:         {(df['cluster'] != -1).sum()}")
print(f"  Rows with BOTH (for analysis): {len(df_ts)}")

if len(df_ts) == 0:
    print("\nWARNING: No rows with both timestamp and cluster.")
    pd.DataFrame(columns=TEMPORAL_STATS_COLUMNS).to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote empty stats file with expected columns to: {OUTPUT_PATH}")
    sys.exit(0)

cluster_ids = sorted(df_ts["cluster"].unique())
print(f"Real clusters with timestamps: {len(cluster_ids)}")


# ---------------------------------------------------------------------------
# KLEINBERG BURST DETECTION
# ---------------------------------------------------------------------------
def kleinberg_burst(event_times, s=2.0, gamma=1.0, freq="D"):
    """
    Simplified Kleinberg burst detector.
    freq: 'D' for daily, 'W' for weekly
    Returns: (burst_score, burst_periods)
    """
    if len(event_times) < MIN_ARTICLES_FOR_BURST:
        return 0, []

    # Try library first
    try:
        import burst_detection as bd  # type: ignore[import-not-found]
        dates = event_times.sort_values()
        if freq == "W":
            floored = dates.dt.to_period("W").apply(lambda p: p.start_time)
        else:
            floored = dates.dt.floor("D")
        daily = floored.value_counts().sort_index()
        r = daily.values.astype(float)
        n = len(r)
        rng_freq = "7D" if freq == "W" else "D"
        d = len(pd.date_range(daily.index.min(), daily.index.max(), freq=rng_freq))

        if n < 2 or d < 2:
            return 0, []

        q = bd.burst_detection(r, d, s=s, gamma=gamma)
        level_max = int(np.max(q)) if len(q) > 0 else 0

        periods = []
        in_burst = False
        burst_start = None
        burst_level = 0
        for i, lv in enumerate(q):
            date = daily.index[i]
            if lv >= 1 and not in_burst:
                in_burst = True
                burst_start = date
                burst_level = int(lv)
            elif lv >= 1 and in_burst:
                burst_level = max(burst_level, int(lv))
            elif lv < 1 and in_burst:
                periods.append((burst_start, daily.index[i - 1], burst_level))
                in_burst = False
        if in_burst:
            periods.append((burst_start, daily.index[-1], burst_level))

        return level_max, periods
    except (ImportError, Exception):
        pass

    # Manual Kleinberg (two-state)
    dates = event_times.sort_values()

    if freq == "W":
        # For weekly: bin by ISO week
        floored = dates.dt.to_period("W").apply(lambda p: p.start_time)
    else:
        floored = dates.dt.floor("D")

    first_day = floored.min()
    last_day = floored.max()

    # Build complete range
    if freq == "W":
        all_periods = pd.date_range(first_day, last_day, freq="7D")
    else:
        all_periods = pd.date_range(first_day, last_day, freq="D")
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
# COMPUTE STATS PER CLUSTER
# ---------------------------------------------------------------------------
print("\nComputing temporal stats per cluster...")
records = []

# Get all cluster sizes (including articles without timestamps) for scoring
all_cluster_sizes = df[df["cluster"] != -1]["cluster"].value_counts().to_dict()

for i, cluster_id in enumerate(cluster_ids):
    sub = df_ts[df_ts["cluster"] == cluster_id]
    times = sub["timestamp"]

    article_count_ts = len(sub)  # articles WITH timestamp
    total_articles = all_cluster_sizes.get(cluster_id, article_count_ts)
    first_seen = times.min()
    last_seen = times.max()
    span_days = max((last_seen - first_seen).days, 1)

    # Temporal spread
    temporal_spread = times.apply(lambda t: (t - first_seen).days).std()
    if pd.isna(temporal_spread):
        temporal_spread = 0.0

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

    # Representative title
    rep_title = str(sub.iloc[0].get("title", ""))[:200] if len(sub) > 0 else ""

    # Concentration: timestamped articles / span_days
    concentration = article_count_ts / max(span_days, 1)

    # Improved suspicion score:
    #   - Higher for burst at both granularities
    #   - Penalize very small clusters (< 5 timestamped articles)
    #   - Factor in total cluster size
    size_factor = min(total_articles / 20, 5.0)  # caps at 5x for clusters >= 100
    small_penalty = max(0, 5 - article_count_ts) * 2  # penalty for < 5 timestamped articles

    suspicion_score = round(
        (burst_daily * 10)
        + (burst_weekly * 5)
        + (burst_stable * 10)
        + (concentration * 2)
        + (burst_dur_daily * 0.5)
        + (size_factor * 1.5)
        - small_penalty,
        3
    )

    records.append({
        "cluster": cluster_id,
        "topic_group": topic,
        "total_articles": total_articles,
        "timestamped_articles": article_count_ts,
        "first_seen": first_seen.date() if pd.notna(first_seen) else None,
        "last_seen": last_seen.date() if pd.notna(last_seen) else None,
        "span_days": span_days,
        "temporal_spread_days": round(temporal_spread, 2),
        "burst_score_daily": burst_daily,
        "burst_duration_daily": burst_dur_daily,
        "burst_periods_daily": len(burst_periods_daily),
        "burst_score_weekly": burst_weekly,
        "burst_duration_weekly": burst_dur_weekly,
        "burst_periods_weekly": len(burst_periods_weekly),
        "burst_stable": burst_stable,
        "concentration": round(concentration, 4),
        "suspicion_score": suspicion_score,
        "representative_title": rep_title,
    })

    if (i + 1) % 50 == 0:
        print(f"  Processed {i + 1}/{len(cluster_ids)} clusters...")

stats_df = pd.DataFrame(records).sort_values(
    "suspicion_score", ascending=False
).reset_index(drop=True)

# Also keep backward-compatible column names for dashboard
stats_df["burst_score"] = stats_df["burst_score_daily"]
stats_df["burst_duration_days"] = stats_df["burst_duration_daily"]
stats_df["article_count"] = stats_df["total_articles"]
stats_df["num_burst_periods"] = stats_df["burst_periods_daily"]


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEMPORAL STATS — TOP 20 MOST SUSPICIOUS CLUSTERS")
print("=" * 60)
for _, row in stats_df.head(20).iterrows():
    title = str(row.get("representative_title", ""))[:60]
    print(
        f"  Cluster {int(row['cluster']):4d} | {str(row['topic_group']):15s} "
        f"| total={int(row['total_articles']):5d} | ts={int(row['timestamped_articles']):4d} "
        f"| burst_d={int(row['burst_score_daily'])} burst_w={int(row['burst_score_weekly'])} "
        f"| stable={int(row['burst_stable'])} "
        f"| susp={row['suspicion_score']:.1f} | {title}"
    )

print(f"\nTotal clusters analyzed:        {len(stats_df)}")
print(f"Clusters with daily burst > 0:  {(stats_df['burst_score_daily'] > 0).sum()}")
print(f"Clusters with weekly burst > 0: {(stats_df['burst_score_weekly'] > 0).sum()}")
print(f"Clusters stable (both):         {stats_df['burst_stable'].sum()}")
print(f"Max suspicion score:            {stats_df['suspicion_score'].max()}")


# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
stats_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to: {OUTPUT_PATH}")
