"""
Streamlit Dashboard
==============================
Interactive dashboard for exploring coordinated campaign detection results.
Loads parquet-backed dashboard assets prepared by PrepareDashboardData.py.

Run with:
    python PrepareDashboardData.py
    streamlit run Dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.paths import (
    DASHBOARD_ABLATION_PARQUET,
    DASHBOARD_CLUSTER_ARTICLES_PARQUET,
    DASHBOARD_CLUSTER_DAILY_PARQUET,
    DASHBOARD_CLUSTER_OVERVIEW_PARQUET,
    DASHBOARD_CONFIG_PARQUET,
    DASHBOARD_EVAL_PARQUET,
    DASHBOARD_META_PARQUET,
    DASHBOARD_SCATTER_SAMPLE_PARQUET,
    DASHBOARD_TEMPORAL_PARQUET,
    DASHBOARD_TOPIC_SUMMARY_PARQUET,
)


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Coordinated Campaign Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
ESSENTIAL_ASSETS = [
    DASHBOARD_META_PARQUET,
    DASHBOARD_TOPIC_SUMMARY_PARQUET,
    DASHBOARD_CLUSTER_OVERVIEW_PARQUET,
    DASHBOARD_CLUSTER_DAILY_PARQUET,
    DASHBOARD_CLUSTER_ARTICLES_PARQUET,
    DASHBOARD_SCATTER_SAMPLE_PARQUET,
    DASHBOARD_TEMPORAL_PARQUET,
]

OPTIONAL_ASSETS = [
    DASHBOARD_CONFIG_PARQUET,
    DASHBOARD_EVAL_PARQUET,
    DASHBOARD_ABLATION_PARQUET,
]

CLUSTER_SUMMARY_COLUMNS = [
    "cluster",
    "topic_group",
    "article_count",
    "burst_score",
    "burst_duration_days",
    "span_days",
    "suspicion_score",
    "representative_title",
]

EXTRA_TEMPORAL_COLUMNS = [
    "burst_score_daily",
    "burst_score_weekly",
    "burst_stable",
    "total_articles",
    "timestamped_articles",
    "timestamp_coverage_ratio",
    "peak_day_share",
    "peak_to_baseline_ratio",
    "support_weight",
    "coverage_weight",
    "source_weight",
    "domain_weight",
    "active_day_ratio",
    "top_domain_share",
    "timestamp_source_reliability",
]

PAGE_OPTIONS = [
    "Cluster Explorer",
    "Timeline and Bursts",
    "Top Campaigns",
    "Evaluation and Ablation",
]


def ensure_dataframe(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Return an empty DataFrame when an optional loader returns None."""
    return frame if frame is not None else pd.DataFrame()


def assets_missing() -> list[Path]:
    return [path for path in ESSENTIAL_ASSETS if not path.exists()]


def format_cluster_option(row: pd.Series) -> str:
    return (
        f"Cluster {int(row['cluster'])} - {row['topic_group']} "
        f"(articles={int(row['articles_shown'])}, size={int(row['cluster_size'])})"
    )


def filter_temporal_df(
    temporal_df: pd.DataFrame,
    selected_topics: list[str],
    min_cluster_size: int,
    min_burst: int,
) -> pd.DataFrame:
    if temporal_df.empty:
        return temporal_df

    burst_col = "burst_score" if "burst_score" in temporal_df.columns else "burst_score_daily"
    size_col = "article_count" if "article_count" in temporal_df.columns else "total_articles"
    filtered = temporal_df[
        temporal_df["topic_group"].isin(selected_topics)
        & (temporal_df[size_col] >= min_cluster_size)
        & (temporal_df[burst_col] >= min_burst)
    ].copy()
    if "suspicion_score" in filtered.columns:
        filtered = filtered.sort_values("suspicion_score", ascending=False)
    return filtered


def build_visible_clusters(
    cluster_overview_df: pd.DataFrame,
    cluster_daily_df: pd.DataFrame,
    selected_topics: list[str],
    min_cluster_size: int,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> pd.DataFrame:
    visible = cluster_overview_df[
        cluster_overview_df["topic_group"].isin(selected_topics)
        & (cluster_overview_df["cluster_size"] >= min_cluster_size)
    ].copy()

    if visible.empty:
        visible["in_range_timestamped"] = []
        visible["articles_shown"] = []
        return visible

    if date_range is None:
        visible["in_range_timestamped"] = visible["timestamped_articles"]
        visible["articles_shown"] = visible["total_articles"]
        return visible

    date_start, date_end = date_range
    in_range = cluster_daily_df[
        cluster_daily_df["topic_group"].isin(selected_topics)
        & (cluster_daily_df["date"] >= date_start)
        & (cluster_daily_df["date"] <= date_end)
    ].copy()
    in_range_counts = (
        in_range.groupby("cluster")["article_count"]
        .sum()
        .reset_index(name="in_range_timestamped")
    )

    visible = visible.merge(in_range_counts, on="cluster", how="left")
    visible["in_range_timestamped"] = visible["in_range_timestamped"].fillna(0).astype(int)
    visible["articles_shown"] = visible["undated_articles"] + visible["in_range_timestamped"]
    visible = visible[visible["articles_shown"] > 0].copy()
    return visible


def filter_sample_for_display(
    scatter_df: pd.DataFrame,
    cluster_overview_df: pd.DataFrame,
    selected_topics: list[str],
    min_cluster_size: int,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> pd.DataFrame:
    if scatter_df.empty:
        return scatter_df

    sample = scatter_df.copy()
    if "cluster_size" not in sample.columns:
        cluster_sizes = cluster_overview_df[["cluster", "cluster_size"]].drop_duplicates()
        sample = sample.merge(cluster_sizes, on="cluster", how="left")

    if "cluster_size" not in sample.columns:
        sample["cluster_size"] = 0
    sample["cluster_size"] = sample["cluster_size"].fillna(0)

    sample = sample[
        sample["topic_group"].isin(selected_topics)
        & (sample["cluster_size"] >= min_cluster_size)
    ].copy()
    if sample.empty:
        return sample

    if date_range is not None:
        date_start, date_end = date_range
        mask = sample["timestamp"].isna() | (
            (sample["timestamp"] >= date_start) & (sample["timestamp"] <= date_end)
        )
        sample = sample[mask].copy()

    sample["cluster_str"] = sample["cluster"].astype(str)
    return sample


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dashboard_meta() -> dict | None:
    if not DASHBOARD_META_PARQUET.exists():
        return None
    meta_df = pd.read_parquet(DASHBOARD_META_PARQUET)
    if meta_df.empty:
        return None
    meta = meta_df.iloc[0].to_dict()
    for key in ("min_timestamp", "max_timestamp"):
        if key in meta and pd.notna(meta[key]):
            meta[key] = pd.to_datetime(meta[key])
    return meta


@st.cache_data(show_spinner=False)
def load_topic_summary() -> pd.DataFrame | None:
    if not DASHBOARD_TOPIC_SUMMARY_PARQUET.exists():
        return None
    return pd.read_parquet(DASHBOARD_TOPIC_SUMMARY_PARQUET)


@st.cache_data(show_spinner=False)
def load_cluster_overview() -> pd.DataFrame | None:
    if not DASHBOARD_CLUSTER_OVERVIEW_PARQUET.exists():
        return None
    df = pd.read_parquet(DASHBOARD_CLUSTER_OVERVIEW_PARQUET)
    for col in ("first_seen", "last_seen"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_cluster_daily() -> pd.DataFrame | None:
    if not DASHBOARD_CLUSTER_DAILY_PARQUET.exists():
        return None
    df = pd.read_parquet(DASHBOARD_CLUSTER_DAILY_PARQUET)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_scatter_sample() -> pd.DataFrame | None:
    if not DASHBOARD_SCATTER_SAMPLE_PARQUET.exists():
        return None
    df = pd.read_parquet(DASHBOARD_SCATTER_SAMPLE_PARQUET)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_temporal() -> pd.DataFrame | None:
    if not DASHBOARD_TEMPORAL_PARQUET.exists():
        return None
    return pd.read_parquet(DASHBOARD_TEMPORAL_PARQUET)


@st.cache_data(show_spinner=False)
def load_config() -> pd.DataFrame | None:
    if not DASHBOARD_CONFIG_PARQUET.exists():
        return None
    return pd.read_parquet(DASHBOARD_CONFIG_PARQUET)


@st.cache_data(show_spinner=False)
def load_ablation() -> pd.DataFrame | None:
    if not DASHBOARD_ABLATION_PARQUET.exists():
        return None
    return pd.read_parquet(DASHBOARD_ABLATION_PARQUET)


@st.cache_data(show_spinner=False)
def load_eval() -> pd.DataFrame | None:
    if not DASHBOARD_EVAL_PARQUET.exists():
        return None
    return pd.read_parquet(DASHBOARD_EVAL_PARQUET)


@st.cache_data(show_spinner=False)
def load_articles_for_cluster(cluster_id: int) -> pd.DataFrame:
    if not DASHBOARD_CLUSTER_ARTICLES_PARQUET.exists():
        return pd.DataFrame()

    filters = [("cluster", "==", int(cluster_id))]
    try:
        df = pd.read_parquet(DASHBOARD_CLUSTER_ARTICLES_PARQUET, filters=filters)
    except Exception:
        df = pd.read_parquet(DASHBOARD_CLUSTER_ARTICLES_PARQUET)
        df = df[df["cluster"] == int(cluster_id)].copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "timestamp_date" not in df.columns and "timestamp" in df.columns:
        df["timestamp_date"] = df["timestamp"].dt.strftime("%Y-%m-%d").fillna("")
    if "topics" not in df.columns:
        df["topics"] = ""
    return df


@st.cache_data(show_spinner=False)
def load_daily_for_cluster(cluster_id: int) -> pd.DataFrame:
    if not DASHBOARD_CLUSTER_DAILY_PARQUET.exists():
        return pd.DataFrame()

    filters = [("cluster", "==", int(cluster_id))]
    try:
        df = pd.read_parquet(DASHBOARD_CLUSTER_DAILY_PARQUET, filters=filters)
    except Exception:
        df = pd.read_parquet(DASHBOARD_CLUSTER_DAILY_PARQUET)
        df = df[df["cluster"] == int(cluster_id)].copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# LOAD CORE DATA
# ---------------------------------------------------------------------------
missing_assets = assets_missing()
dashboard_meta = load_dashboard_meta() if not missing_assets else None
topic_summary_df = ensure_dataframe(load_topic_summary() if not missing_assets else None)
cluster_overview_df = ensure_dataframe(load_cluster_overview() if not missing_assets else None)
cluster_daily_df = ensure_dataframe(load_cluster_daily() if not missing_assets else None)
scatter_sample_df = ensure_dataframe(load_scatter_sample() if not missing_assets else None)
temporal_df = ensure_dataframe(load_temporal() if not missing_assets else None)
config_df = ensure_dataframe(load_config() if not missing_assets else None)
ablation_df = ensure_dataframe(load_ablation() if not missing_assets else None)
eval_df = ensure_dataframe(load_eval() if not missing_assets else None)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.title("Campaign Detector")
st.sidebar.markdown("---")
page = st.sidebar.radio("Page", PAGE_OPTIONS, index=0)

if missing_assets:
    st.title("Coordinated Campaign Detection Dashboard")
    st.error(
        "Fast dashboard assets are missing.\n\n"
        "Run this once after your pipeline finishes:\n\n"
        "```bash\n"
        "python PrepareDashboardData.py\n"
        "streamlit run Dashboard.py\n"
        "```"
    )
    with st.expander("Missing assets"):
        for path in missing_assets:
            st.write(str(path))
    st.info(
        "The dashboard now expects parquet assets on purpose so it does not need "
        "to read the full clustering parquet at startup."
    )
    st.stop()

topics = sorted(topic_summary_df["topic_group"].dropna().unique()) if "topic_group" in topic_summary_df.columns else []
selected_topics = st.sidebar.multiselect("Filter by topic", topics, default=topics)

max_cluster_size = int(cluster_overview_df["cluster_size"].max()) if not cluster_overview_df.empty else 100
min_cluster_size = st.sidebar.slider("Min cluster size", 1, max(max_cluster_size, 1), 5)

if not temporal_df.empty:
    burst_col = "burst_score" if "burst_score" in temporal_df.columns else "burst_score_daily"
    max_burst = int(temporal_df[burst_col].max()) if len(temporal_df) > 0 else 1
    min_burst = st.sidebar.slider("Min burst score", 0, max(max_burst, 1), 0)
else:
    min_burst = 0

if dashboard_meta and pd.notna(dashboard_meta.get("min_timestamp")) and pd.notna(dashboard_meta.get("max_timestamp")):
    date_min = pd.Timestamp(dashboard_meta["min_timestamp"]).date()
    date_max = pd.Timestamp(dashboard_meta["max_timestamp"]).date()
    date_range_input = st.sidebar.date_input(
        "Date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    if isinstance(date_range_input, (tuple, list)) and len(date_range_input) == 2:
        date_range = (
            pd.Timestamp(date_range_input[0]),
            pd.Timestamp(date_range_input[1]),
        )
    else:
        date_range = None
else:
    date_range = None

st.sidebar.markdown("---")
st.sidebar.caption(
    "Metrics, tables, and timelines use full parquet assets. Only the UMAP "
    "scatter uses a stratified sample for rendering speed."
)


# ---------------------------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------------------------
st.title("Coordinated Campaign Detection Dashboard")
st.markdown(
    "Semantic clustering + temporal burst analysis on Romanian news (RoLargeSum)"
)

visible_clusters_df = build_visible_clusters(
    cluster_overview_df=cluster_overview_df,
    cluster_daily_df=cluster_daily_df,
    selected_topics=selected_topics,
    min_cluster_size=min_cluster_size,
    date_range=date_range,
)

filtered_temporal_df = filter_temporal_df(
    temporal_df=temporal_df,
    selected_topics=selected_topics,
    min_cluster_size=min_cluster_size,
    min_burst=min_burst,
)

scatter_display_df = filter_sample_for_display(
    scatter_df=scatter_sample_df,
    cluster_overview_df=cluster_overview_df,
    selected_topics=selected_topics,
    min_cluster_size=min_cluster_size,
    date_range=date_range,
)


# ===========================================================================
# PAGE: CLUSTER EXPLORER
# ===========================================================================
if page == "Cluster Explorer":
    st.header("Cluster Explorer")

    articles_shown = int(visible_clusters_df["articles_shown"].sum()) if not visible_clusters_df.empty else 0
    timestamped_shown = int(visible_clusters_df["in_range_timestamped"].sum()) if not visible_clusters_df.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Articles shown", articles_shown)
    col2.metric("Unique clusters", int(visible_clusters_df["cluster"].nunique()) if not visible_clusters_df.empty else 0)
    col3.metric("Topics", int(visible_clusters_df["topic_group"].nunique()) if not visible_clusters_df.empty else 0)
    ts_pct = 100 * timestamped_shown / max(articles_shown, 1)
    col4.metric("With timestamp", f"{ts_pct:.0f}%")

    st.caption(
        f"UMAP scatter is rendered from a stratified sample of {len(scatter_display_df):,} "
        "points so the plot stays responsive without changing the underlying metrics."
    )

    if not scatter_display_df.empty:
        fig_scatter = px.scatter(
            scatter_display_df,
            x="umap_x",
            y="umap_y",
            color="topic_group",
            hover_data=["title", "cluster", "topic_group"],
            title="UMAP 2-D Projection (colored by topic)",
            opacity=0.65,
            height=600,
        )
        fig_scatter.update_layout(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            legend_title="Topic",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        fig_scatter_cluster = px.scatter(
            scatter_display_df,
            x="umap_x",
            y="umap_y",
            color="cluster_str",
            hover_data=["title", "cluster", "topic_group"],
            title="UMAP 2-D Projection (colored by cluster ID)",
            opacity=0.65,
            height=600,
        )
        fig_scatter_cluster.update_layout(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            showlegend=False,
        )
        st.plotly_chart(fig_scatter_cluster, use_container_width=True)
    else:
        st.info("No UMAP sample rows match the selected filters.")

    st.subheader("Cluster Summary")
    if not filtered_temporal_df.empty:
        available_cols = [c for c in CLUSTER_SUMMARY_COLUMNS if c in filtered_temporal_df.columns]
        for col in EXTRA_TEMPORAL_COLUMNS:
            if col in filtered_temporal_df.columns and col not in available_cols:
                available_cols.insert(-1, col)

        st.dataframe(
            filtered_temporal_df[available_cols].head(50),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No clusters match the current filters.")

    st.subheader("Article Viewer")
    if not visible_clusters_df.empty:
        option_df = visible_clusters_df.sort_values(["articles_shown", "cluster"], ascending=[False, True]).reset_index(drop=True)
        selected_idx = st.selectbox(
            "Select a cluster to view articles",
            range(len(option_df)),
            format_func=lambda idx: format_cluster_option(option_df.iloc[idx]),
        )
        selected_cluster = int(option_df.iloc[selected_idx]["cluster"])
        cluster_articles = load_articles_for_cluster(selected_cluster)

        if date_range is not None and not cluster_articles.empty:
            date_start, date_end = date_range
            mask = cluster_articles["timestamp"].isna() | (
                (cluster_articles["timestamp"] >= date_start)
                & (cluster_articles["timestamp"] <= date_end)
            )
            cluster_articles = cluster_articles[mask].copy()

        st.write(f"**{len(cluster_articles)} articles** in cluster {selected_cluster}")
        if not cluster_articles.empty:
            st.dataframe(
                cluster_articles[["title", "topics", "timestamp_date", "url"]]
                .rename(columns={"timestamp_date": "date"})
                .head(30),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No articles remain in this cluster after the current date filter.")
    else:
        st.info("No visible clusters match the current filters.")


# ===========================================================================
# PAGE: TIMELINE & BURSTS
# ===========================================================================
elif page == "Timeline and Bursts":
    st.header("Timeline & Burst Analysis")

    if not filtered_temporal_df.empty:
        burst_col = "burst_score" if "burst_score" in filtered_temporal_df.columns else "burst_score_daily"
        size_col = "article_count" if "article_count" in filtered_temporal_df.columns else "total_articles"
        cluster_options = filtered_temporal_df.apply(
            lambda r: (
                f"Cluster {int(r['cluster'])} - {r['topic_group']} "
                f"(burst={int(r[burst_col])}, arts={int(r[size_col])})"
            ),
            axis=1,
        ).tolist()
        cluster_ids_list = filtered_temporal_df["cluster"].tolist()

        selected_idx = st.selectbox(
            "Select cluster",
            range(len(cluster_options)),
            format_func=lambda i: cluster_options[i],
        )
        sel_cluster_id = int(cluster_ids_list[selected_idx])

        sel_stats = filtered_temporal_df[filtered_temporal_df["cluster"] == sel_cluster_id]
        if not sel_stats.empty:
            row = sel_stats.iloc[0]
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Total articles", int(row.get("total_articles", row.get("article_count", 0))))
            mcol2.metric("Timestamped", int(row.get("timestamped_articles", 0)))
            mcol3.metric("Suspicion", f"{row.get('suspicion_score', 0):.1f}")

            burst_d = row.get("burst_score_daily", row.get("burst_score", 0))
            burst_w = row.get("burst_score_weekly", 0)
            stable = "Yes" if row.get("burst_stable", 0) == 1 else "No"
            mcol4.metric("Burst (D/W/Stable)", f"{int(burst_d)}/{int(burst_w)}/{stable}")

            coverage_ratio = row.get("timestamp_coverage_ratio")
            peak_ratio = row.get("peak_to_baseline_ratio")
            support_weight = row.get("support_weight")
            coverage_weight = row.get("coverage_weight")
            source_weight = row.get("source_weight")
            domain_weight = row.get("domain_weight")

            st.markdown(f"""
            **Cluster {sel_cluster_id}** - {row.get('topic_group', 'N/A')}

            | Metric | Value |
            |---|---|
            | Date range | {row.get('first_seen', '?')} to {row.get('last_seen', '?')} |
            | Span (days) | {row.get('span_days', 'N/A')} |
            | Burst daily | {int(burst_d)} |
            | Burst weekly | {int(burst_w)} |
            | Burst stable | {stable} |
            | Timestamp coverage | {f"{coverage_ratio:.1%}" if pd.notna(coverage_ratio) else "N/A"} |
            | Peak / baseline | {f"{peak_ratio:.2f}" if pd.notna(peak_ratio) else "N/A"} |
            | Support / coverage weight | {f"{support_weight:.2f} / {coverage_weight:.2f}" if pd.notna(support_weight) and pd.notna(coverage_weight) else "N/A"} |
            | Source / domain weight | {f"{source_weight:.2f} / {domain_weight:.2f}" if pd.notna(source_weight) and pd.notna(domain_weight) else "N/A"} |
            | Representative | {str(row.get('representative_title', ''))[:120]} |
            """)

        cluster_daily = load_daily_for_cluster(sel_cluster_id)
        if not cluster_daily.empty:
            fig_timeline = px.bar(
                cluster_daily,
                x="date",
                y="article_count",
                title=f"Daily article count - Cluster {sel_cluster_id}",
            )
            fig_timeline.update_layout(
                xaxis_title="Date",
                yaxis_title="Articles per day",
                height=400,
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

            weekly = (
                cluster_daily.set_index("date")["article_count"]
                .resample("W")
                .sum()
                .reset_index()
                .rename(columns={"article_count": "count", "date": "week"})
            )
            fig_weekly = px.bar(
                weekly,
                x="week",
                y="count",
                title=f"Weekly article count - Cluster {sel_cluster_id}",
            )
            fig_weekly.update_layout(
                xaxis_title="Week",
                yaxis_title="Articles per week",
                height=350,
            )
            st.plotly_chart(fig_weekly, use_container_width=True)
        else:
            st.warning(f"No timestamped articles in cluster {sel_cluster_id}.")

        st.subheader("Burst Score Distribution")
        fig_burst_dist = px.histogram(
            filtered_temporal_df,
            x=burst_col,
            title="Distribution of daily burst scores across filtered clusters",
            nbins=max(int(filtered_temporal_df[burst_col].max()) + 1, 5),
        )
        st.plotly_chart(fig_burst_dist, use_container_width=True)

        st.subheader("Burst Score vs Cluster Size")
        if "suspicion_score" in filtered_temporal_df.columns:
            susp = filtered_temporal_df[filtered_temporal_df["suspicion_score"] > 0].copy()
        else:
            susp = filtered_temporal_df.copy()
        if not susp.empty:
            fig_burst_size = px.scatter(
                susp,
                x=size_col,
                y=burst_col,
                color="topic_group",
                size="suspicion_score" if "suspicion_score" in susp.columns else None,
                hover_data=["cluster", "representative_title"],
                title="Cluster size vs Burst score (bubble size = suspicion)",
                height=500,
            )
            st.plotly_chart(fig_burst_size, use_container_width=True)
    else:
        st.info("No temporal clusters match the selected filters.")


# ===========================================================================
# PAGE: TOP CAMPAIGNS
# ===========================================================================
elif page == "Top Campaigns":
    st.header("Top Suspected Coordinated Campaigns")
    st.markdown(
        "Clusters ranked by suspicion score - combining semantic cohesion, "
        "temporal burst strength, and cluster size."
    )

    if not filtered_temporal_df.empty:
        top_campaigns = filtered_temporal_df.head(15)

        for rank, (_, camp) in enumerate(top_campaigns.iterrows(), 1):
            cid = int(camp["cluster"])
            topic = camp.get("topic_group", "?")
            score = camp.get("suspicion_score", 0)
            total = int(camp.get("total_articles", camp.get("article_count", 0)))
            burst_d = int(camp.get("burst_score_daily", camp.get("burst_score", 0)))
            burst_w = int(camp.get("burst_score_weekly", 0))
            stable = camp.get("burst_stable", 0)
            first = camp.get("first_seen", "?")
            last = camp.get("last_seen", "?")
            rep = str(camp.get("representative_title", ""))[:150]

            severity = "HIGH" if score > 20 else "MEDIUM" if score > 10 else "LOW"

            with st.expander(
                f"[{severity}] #{rank} - Cluster {cid} ({topic}) - Score: {score:.1f}",
                expanded=(rank <= 3),
            ):
                ecol1, ecol2, ecol3 = st.columns(3)
                ecol1.metric("Total articles", total)
                ecol2.metric("Burst (D/W)", f"{burst_d}/{burst_w}")
                ecol3.metric("Stable burst?", "Yes" if stable else "No")

                st.markdown(f"**Period:** {first} to {last}")
                st.markdown(f"**Representative:** {rep}")

                arts = load_articles_for_cluster(cid)
                if not arts.empty:
                    st.dataframe(
                        arts[["title", "topics", "timestamp_date", "url"]]
                        .rename(columns={"timestamp_date": "date"})
                        .head(10),
                        use_container_width=True,
                        hide_index=True,
                    )
    else:
        st.info("No campaigns match the current filters.")


# ===========================================================================
# PAGE: EVALUATION & ABLATION
# ===========================================================================
else:
    st.header("Evaluation & Ablation Studies")

    if not config_df.empty:
        st.subheader("HDBSCAN Config Sweep Results")
        st.dataframe(config_df, use_container_width=True, hide_index=True)

        best = (
            config_df.sort_values("selection_score", ascending=False)
            .groupby("topic_group")
            .first()
            .reset_index()
        )

        fig_sil = px.bar(
            best,
            x="topic_group",
            y="silhouette",
            color="topic_group",
            title="Best Silhouette Score per Topic (SBERT + HDBSCAN)",
            text="silhouette",
        )
        fig_sil.update_traces(texttemplate="%{text:.3f}", textposition="auto")
        st.plotly_chart(fig_sil, use_container_width=True)

        fig_noise = px.bar(
            best,
            x="topic_group",
            y="noise_percent",
            color="topic_group",
            title="Noise % per Topic (best config)",
            text="noise_percent",
        )
        fig_noise.update_traces(texttemplate="%{text:.1f}%", textposition="auto")
        st.plotly_chart(fig_noise, use_container_width=True)

    if not ablation_df.empty:
        st.subheader("Ablation: TF-IDF/KMeans vs SBERT/KMeans vs SBERT/HDBSCAN")
        st.dataframe(ablation_df, use_container_width=True, hide_index=True)

        sil_df = ablation_df[ablation_df["silhouette"].notna()].copy()
        if not sil_df.empty:
            fig_ablation = px.bar(
                sil_df,
                x="topic_group",
                y="silhouette",
                color="method",
                barmode="group",
                title="Silhouette Score Comparison by Method & Topic",
                text="silhouette",
            )
            fig_ablation.update_traces(texttemplate="%{text:.3f}", textposition="auto")
            st.plotly_chart(fig_ablation, use_container_width=True)

        db_df = ablation_df[ablation_df["davies_bouldin"].notna()].copy()
        if not db_df.empty:
            fig_db = px.bar(
                db_df,
                x="topic_group",
                y="davies_bouldin",
                color="method",
                barmode="group",
                title="Davies-Bouldin Index Comparison (lower = better)",
                text="davies_bouldin",
            )
            fig_db.update_traces(texttemplate="%{text:.3f}", textposition="auto")
            st.plotly_chart(fig_db, use_container_width=True)
    else:
        st.info("Ablation report not found. Run TFIDFBaseline.py and then PrepareDashboardData.py.")

    if not eval_df.empty:
        st.subheader("Detailed Evaluation Report")
        st.dataframe(eval_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pipeline**: DataCuration -> Embeddings -> Clustering -> Temporal -> Evaluation -> PrepareDashboardData -> Dashboard"
)
st.sidebar.markdown("Built with Streamlit + Plotly")
