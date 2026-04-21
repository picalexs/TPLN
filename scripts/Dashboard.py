"""
Streamlit Dashboard
==============================
Interactive dashboard for exploring coordinated campaign detection results.
Loads parquet-backed dashboard assets prepared by PrepareDashboardData.py.

Run with:
    python scripts/PrepareDashboardData.py
    streamlit run scripts/Dashboard.py
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.paths import (
    DASHBOARD_ABLATION_PARQUET,
    DASHBOARD_CLUSTER_ARTICLES_PARQUET,
    DASHBOARD_CLUSTER_DAILY_PARQUET,
    DASHBOARD_CLUSTER_SIMILARITY_PARQUET,
    DASHBOARD_CLUSTER_OVERVIEW_PARQUET,
    DASHBOARD_CONFIG_PARQUET,
    DASHBOARD_EVAL_PARQUET,
    DASHBOARD_META_PARQUET,
    DASHBOARD_NEIGHBORS_PARQUET,
    DASHBOARD_RUNTIME_PARQUET,
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
    DASHBOARD_NEIGHBORS_PARQUET,
    DASHBOARD_CLUSTER_SIMILARITY_PARQUET,
    DASHBOARD_RUNTIME_PARQUET,
]

CLUSTER_SUMMARY_COLUMNS = [
    "cluster",
    "topic_group",
    "article_count",
    "burst_score",
    "burst_duration_days",
    "span_days",
    "suspicion_score",
    "suspicion_score_multi_source",
    "is_single_source",
    "top_terms_joined",
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
    "is_single_source",
    "suspicion_score_multi_source",
    "top_terms",
    "top_terms_joined",
    "representative_titles",
]

PAGE_OPTIONS = [
    "Cluster Explorer",
    "Timeline and Bursts",
    "Top Campaigns",
    "Similarity Map",
    "Topic Health",
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
def load_neighbors() -> pd.DataFrame | None:
    if not DASHBOARD_NEIGHBORS_PARQUET.exists():
        return None
    df = pd.read_parquet(DASHBOARD_NEIGHBORS_PARQUET)
    for col in ("source_timestamp", "neighbor_timestamp"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_cluster_similarity() -> pd.DataFrame | None:
    if not DASHBOARD_CLUSTER_SIMILARITY_PARQUET.exists():
        return None
    return pd.read_parquet(DASHBOARD_CLUSTER_SIMILARITY_PARQUET)


@st.cache_data(show_spinner=False)
def load_runtime_profile() -> pd.DataFrame | None:
    if not DASHBOARD_RUNTIME_PARQUET.exists():
        return None
    return pd.read_parquet(DASHBOARD_RUNTIME_PARQUET)


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
    if "topic_row_idx" in df.columns:
        df["topic_row_idx"] = pd.to_numeric(df["topic_row_idx"], errors="coerce")
    else:
        df["topic_row_idx"] = pd.Series([pd.NA] * len(df))
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
        "python scripts/PrepareDashboardData.py\n"
        "streamlit run scripts/Dashboard.py\n"
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
    max_susp = float(temporal_df["suspicion_score"].max()) if "suspicion_score" in temporal_df.columns and len(temporal_df) > 0 else 0.0
    min_suspicion = st.sidebar.slider("Min suspicion score", 0.0, max(max_susp, 0.0), 0.0, 0.5)
    exclude_single_source = (
        st.sidebar.checkbox(
            "Exclude single-source clusters",
            value=True,
            help=(
                "Drop clusters dominated by one domain "
                "(top_domain_share >= 0.9 AND domain_count <= 2). "
                "They are almost always scraper/aggregator artefacts "
                "rather than multi-outlet coordinated campaigns."
            ),
        )
        if "is_single_source" in temporal_df.columns
        else False
    )
else:
    min_burst = 0
    min_suspicion = 0.0
    exclude_single_source = False

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

if not filtered_temporal_df.empty and "suspicion_score" in filtered_temporal_df.columns:
    filtered_temporal_df = filtered_temporal_df[
        filtered_temporal_df["suspicion_score"] >= float(min_suspicion)
    ].copy()

single_source_removed = 0
if exclude_single_source and "is_single_source" in filtered_temporal_df.columns:
    before_rows = len(filtered_temporal_df)
    filtered_temporal_df = filtered_temporal_df[
        ~filtered_temporal_df["is_single_source"].fillna(False).astype(bool)
    ].copy()
    single_source_removed = before_rows - len(filtered_temporal_df)

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
        st.plotly_chart(fig_scatter, width="stretch")

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
        st.plotly_chart(fig_scatter_cluster, width="stretch")
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
            width="stretch",
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
                width="stretch",
                hide_index=True,
            )

            st.markdown("### Semantic Neighbor Evidence")
            neighbors_df = ensure_dataframe(load_neighbors())
            if not neighbors_df.empty and "topic_row_idx" in cluster_articles.columns:
                neighbor_candidates = cluster_articles.dropna(subset=["topic_row_idx"]).copy()
                if not neighbor_candidates.empty:
                    neighbor_candidates["topic_row_idx"] = pd.to_numeric(
                        neighbor_candidates["topic_row_idx"],
                        errors="coerce",
                    )
                    neighbor_candidates = neighbor_candidates.dropna(subset=["topic_row_idx"]).copy()
                if not neighbor_candidates.empty:
                    neighbor_candidates = neighbor_candidates.reset_index(drop=True)
                    article_idx = st.selectbox(
                        "Choose article for neighbor lookup",
                        range(len(neighbor_candidates)),
                        format_func=lambda i: str(neighbor_candidates.iloc[i]["title"])[:110],
                    )
                    src = neighbor_candidates.iloc[article_idx]
                    src_topic = str(src.get("topic_group", ""))
                    src_row_idx = int(src.get("topic_row_idx"))
                    src_neighbors = neighbors_df[
                        (neighbors_df["topic_group"] == src_topic)
                        & (pd.to_numeric(neighbors_df["source_idx"], errors="coerce") == src_row_idx)
                    ].copy()
                    if not src_neighbors.empty:
                        src_neighbors = src_neighbors.sort_values("rank").head(20)
                        st.dataframe(
                            src_neighbors[
                                [
                                    "rank",
                                    "score",
                                    "neighbor_cluster",
                                    "neighbor_title",
                                    "neighbor_domain",
                                    "neighbor_url",
                                ]
                            ],
                            width="stretch",
                            hide_index=True,
                        )
                    else:
                        st.info("No FAISS neighbors found for this article. Re-run clustering if needed.")
            else:
                st.info("Neighbor asset not available. Re-run embeddings with FAISS kNN export enabled.")
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
            | Single-source cluster | {'Yes (top_domain >= 0.9)' if bool(row.get('is_single_source', False)) else 'No'} |
            | Multi-source suspicion | {f"{row.get('suspicion_score_multi_source', 0):.1f}" if pd.notna(row.get('suspicion_score_multi_source')) else 'N/A'} |
            | Top terms (c-TF-IDF) | {str(row.get('top_terms_joined', '')) or 'N/A'} |
            | Representative | {str(row.get('representative_title', ''))[:120]} |
            """)

            rep_titles_raw = row.get("representative_titles", [])
            if isinstance(rep_titles_raw, np.ndarray):
                rep_titles_raw = rep_titles_raw.tolist()
            rep_titles_list = [str(t) for t in rep_titles_raw or [] if str(t).strip()]
            if rep_titles_list:
                with st.expander("More representative titles", expanded=False):
                    for rep_idx, rep_title in enumerate(rep_titles_list, start=1):
                        st.markdown(f"{rep_idx}. {rep_title}")

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
            st.plotly_chart(fig_timeline, width="stretch")

            weekly = (
                cluster_daily.set_index("date")["article_count"]
                .resample("W-SUN")
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
            st.plotly_chart(fig_weekly, width="stretch")
        else:
            st.warning(f"No timestamped articles in cluster {sel_cluster_id}.")

        st.subheader("Burst Score Distribution")
        fig_burst_dist = px.histogram(
            filtered_temporal_df,
            x=burst_col,
            title="Distribution of daily burst scores across filtered clusters",
            nbins=max(int(filtered_temporal_df[burst_col].max()) + 1, 5),
        )
        st.plotly_chart(fig_burst_dist, width="stretch")

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
            st.plotly_chart(fig_burst_size, width="stretch")
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
    if exclude_single_source:
        st.caption(
            f"Single-source clusters excluded from this view: {single_source_removed}. "
            "Toggle 'Exclude single-source clusters' in the sidebar to include them."
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
            terms = str(camp.get("top_terms_joined", ""))
            is_single_source_flag = bool(camp.get("is_single_source", False))

            severity = "HIGH" if score > 20 else "MEDIUM" if score > 10 else "LOW"
            single_tag = " [SINGLE-SOURCE]" if is_single_source_flag else ""

            with st.expander(
                f"[{severity}] #{rank} - Cluster {cid} ({topic}){single_tag} - Score: {score:.1f}",
                expanded=(rank <= 3),
            ):
                ecol1, ecol2, ecol3 = st.columns(3)
                ecol1.metric("Total articles", total)
                ecol2.metric("Burst (D/W)", f"{burst_d}/{burst_w}")
                ecol3.metric("Stable burst?", "Yes" if stable else "No")

                if is_single_source_flag:
                    st.warning(
                        "Single-source cluster: dominated by one domain. "
                        "Likely a scraper/aggregator artefact rather than a "
                        "cross-outlet coordinated campaign."
                    )
                if terms:
                    st.markdown(f"**Top terms (c-TF-IDF):** {terms}")
                st.markdown(f"**Period:** {first} to {last}")
                st.markdown(f"**Representative:** {rep}")

                explain_cols = [
                    "burst_score_daily",
                    "burst_score_weekly",
                    "burst_stable",
                    "timestamp_coverage_ratio",
                    "peak_to_baseline_ratio",
                    "top_domain_share",
                    "timestamp_source_reliability",
                    "support_weight",
                    "coverage_weight",
                    "source_weight",
                    "domain_weight",
                    "suspicion_penalty_total",
                ]
                explain_payload = {c: camp.get(c) for c in explain_cols if c in camp.index}
                if explain_payload:
                    st.json(explain_payload, expanded=False)

                arts = load_articles_for_cluster(cid)
                if not arts.empty:
                    st.dataframe(
                        arts[["title", "topics", "timestamp_date", "url"]]
                        .rename(columns={"timestamp_date": "date"})
                        .head(10),
                        width="stretch",
                        hide_index=True,
                    )

                storyline = load_daily_for_cluster(cid)
                if not storyline.empty:
                    story_fig = px.area(
                        storyline.sort_values("date"),
                        x="date",
                        y="article_count",
                        title=f"Campaign storyline for cluster {cid}",
                    )
                    story_fig.update_layout(height=260)
                    st.plotly_chart(story_fig, width="stretch")
    else:
        st.info("No campaigns match the current filters.")


# ===========================================================================
# PAGE: SIMILARITY MAP
# ===========================================================================
elif page == "Similarity Map":
    st.header("Cross-Cluster Similarity Map")
    st.caption("Cluster proximity graph based on representative UMAP coordinates.")
    cluster_similarity_df = ensure_dataframe(load_cluster_similarity())

    if cluster_similarity_df.empty:
        st.info("No cluster similarity asset found. Re-run PrepareDashboardData.")
    else:
        graph_df = cluster_similarity_df.copy()
        graph_df = graph_df[graph_df["topic_group"].isin(selected_topics)].copy()
        if graph_df.empty:
            st.info("No cluster-similarity edges match selected topic filters.")
        else:
            # Keep a fixed cap for responsiveness and deterministic rendering.
            graph_df = graph_df.sort_values("similarity", ascending=False).head(2000)
            st.dataframe(
                graph_df[["topic_group", "cluster", "neighbor_cluster", "similarity", "distance", "rank"]],
                width="stretch",
                hide_index=True,
            )

            # Use full anchor sample (not date-filtered display sample) so edge drawing
            # remains stable even when sidebar date/size filters are restrictive.
            scatter_anchor_df = scatter_sample_df.copy()
            if not scatter_anchor_df.empty and "sample_type" in scatter_anchor_df.columns:
                scatter_anchor_df = scatter_anchor_df[
                    scatter_anchor_df["sample_type"] == "cluster_anchor"
                ].copy()
            if not scatter_anchor_df.empty:
                scatter_anchor_df = scatter_anchor_df[
                    scatter_anchor_df["topic_group"].isin(selected_topics)
                ].copy()

            cluster_positions = (
                scatter_anchor_df.dropna(subset=["umap_x", "umap_y"])
                .groupby(["topic_group", "cluster"], as_index=False)[["umap_x", "umap_y"]]
                .mean()
            )
            if not cluster_positions.empty:
                pos_lookup = {
                    (str(row.topic_group), int(row.cluster)): (float(row.umap_x), float(row.umap_y))
                    for row in cluster_positions.itertuples(index=False)
                }

                edge_x: list[float | None] = []
                edge_y: list[float | None] = []
                drawn_edges = 0
                for row in graph_df.itertuples(index=False):
                    key_a = (str(row.topic_group), int(row.cluster))
                    key_b = (str(row.topic_group), int(row.neighbor_cluster))
                    if key_a not in pos_lookup or key_b not in pos_lookup:
                        continue
                    ax, ay = pos_lookup[key_a]
                    bx, by = pos_lookup[key_b]
                    edge_x.extend([ax, bx, None])
                    edge_y.extend([ay, by, None])
                    drawn_edges += 1

                fig_graph = go.Figure()
                if edge_x:
                    fig_graph.add_trace(
                        go.Scatter(
                            x=edge_x,
                            y=edge_y,
                            mode="lines",
                            line=dict(width=1.2, color="rgba(40,40,40,0.55)"),
                            hoverinfo="skip",
                            name="similarity edges",
                        )
                    )

                for topic_name in sorted(cluster_positions["topic_group"].astype(str).unique().tolist()):
                    topic_points = cluster_positions[cluster_positions["topic_group"] == topic_name]
                    fig_graph.add_trace(
                        go.Scatter(
                            x=topic_points["umap_x"],
                            y=topic_points["umap_y"],
                            mode="markers",
                            marker=dict(size=7),
                            name=topic_name,
                            text=[f"cluster={int(c)}" for c in topic_points["cluster"]],
                            hovertemplate="%{text}<extra>" + topic_name + "</extra>",
                        )
                    )

                fig_graph.update_layout(
                    title=f"Cluster similarity graph",
                    xaxis_title="UMAP-1",
                    yaxis_title="UMAP-2",
                    height=700,
                )
                st.plotly_chart(fig_graph, width="stretch")
                if drawn_edges == 0:
                    st.warning(
                        "No edges could be projected onto current cluster anchors. "
                        "Re-run PrepareDashboardData.py if this persists."
                    )


# ===========================================================================
# PAGE: TOPIC HEALTH
# ===========================================================================
elif page == "Topic Health":
    st.header("Topic Health Board")
    config_df = ensure_dataframe(load_config())
    runtime_df = ensure_dataframe(load_runtime_profile())

    if topic_summary_df.empty:
        st.info("Topic summary is unavailable.")
    else:
        health_df = topic_summary_df.copy()
        if not config_df.empty:
            best_cfg = (
                config_df.sort_values("selection_score", ascending=False)
                .groupby("topic_group")
                .first()
                .reset_index()[
                    [
                        "topic_group",
                        "silhouette",
                        "noise_percent",
                        "davies_bouldin",
                        "calinski_harabasz",
                        "graph_cohesion",
                        "graph_cohesion_real",
                    ]
                ]
            )
            health_df = health_df.merge(best_cfg, on="topic_group", how="left")

        health_cols = [
            "topic_group",
            "total_rows",
            "real_cluster_count",
            "noise_rate",
            "timestamp_coverage_ratio",
            "mean_real_cluster_size",
            "silhouette",
            "noise_percent",
            "davies_bouldin",
            "calinski_harabasz",
            "graph_cohesion",
            "graph_cohesion_real",
        ]
        health_cols = [col for col in health_cols if col in health_df.columns]
        st.dataframe(health_df[health_cols], width="stretch", hide_index=True)

        if not runtime_df.empty:
            st.subheader("Runtime Observability")
            timed = runtime_df[runtime_df["topic_group"] != "__GLOBAL__"].copy()
            if not timed.empty:
                fig_runtime = px.bar(
                    timed.sort_values("topic_total_seconds", ascending=False),
                    x="topic_group",
                    y="topic_total_seconds",
                    color="topic_group",
                    title="Per-topic total runtime",
                )
                st.plotly_chart(fig_runtime, width="stretch")

                runtime_cols = [
                    "topic_group",
                    "topic_size",
                    "embedding_seconds",
                    "faiss_seconds",
                    "umap_seconds",
                    "hdbscan_sweep_seconds",
                    "topic_total_seconds",
                    "rows_per_second",
                ]
                runtime_cols = [col for col in runtime_cols if col in timed.columns]
                st.dataframe(
                    timed[runtime_cols].sort_values("topic_total_seconds", ascending=False),
                    width="stretch",
                    hide_index=True,
                )


# ===========================================================================
# PAGE: EVALUATION & ABLATION
# ===========================================================================
else:
    st.header("Evaluation & Ablation Studies")
    config_df = ensure_dataframe(load_config())
    ablation_df = ensure_dataframe(load_ablation())
    eval_df = ensure_dataframe(load_eval())

    if not config_df.empty:
        st.subheader("HDBSCAN Config Sweep Results")
        st.dataframe(config_df, width="stretch", hide_index=True)

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
        st.plotly_chart(fig_sil, width="stretch")

        fig_noise = px.bar(
            best,
            x="topic_group",
            y="noise_percent",
            color="topic_group",
            title="Noise % per Topic (best config)",
            text="noise_percent",
        )
        fig_noise.update_traces(texttemplate="%{text:.1f}%", textposition="auto")
        st.plotly_chart(fig_noise, width="stretch")

    if not ablation_df.empty:
        st.subheader("Ablation: TF-IDF/KMeans vs SBERT/KMeans vs SBERT/HDBSCAN")
        ab_cols = st.columns(3)
        ab_topics = ["All"] + sorted(ablation_df["topic_group"].dropna().astype(str).unique().tolist())
        ab_methods = ["All"] + sorted(ablation_df["method"].dropna().astype(str).unique().tolist())
        ab_topic = ab_cols[0].selectbox("Ablation topic", ab_topics, index=0)
        ab_method = ab_cols[1].selectbox("Ablation method", ab_methods, index=0)
        ab_max_rows = int(ab_cols[2].slider("Rows", min_value=20, max_value=2000, value=300, step=20))

        ablation_view = ablation_df.copy()
        if ab_topic != "All":
            ablation_view = ablation_view[ablation_view["topic_group"] == ab_topic]
        if ab_method != "All":
            ablation_view = ablation_view[ablation_view["method"] == ab_method]

        st.dataframe(ablation_view.head(ab_max_rows), width="stretch", hide_index=True)

        winner_col = "topic_winner_method"
        if winner_col in ablation_view.columns:
            st.markdown("#### Topic-level winners")
            winner_df = (
                ablation_view[["topic_group", winner_col]]
                .dropna()
                .drop_duplicates()
            )
            if not winner_df.empty:
                winner_counts = winner_df[winner_col].value_counts().reset_index()
                winner_counts.columns = ["method", "wins"]
                fig_winners = px.bar(
                    winner_counts,
                    x="method",
                    y="wins",
                    color="method",
                    title="How many topics each method wins",
                )
                st.plotly_chart(fig_winners, width="stretch")

        sil_df = ablation_view[ablation_view["silhouette"].notna()].copy()
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
            st.plotly_chart(fig_ablation, width="stretch")

        db_df = ablation_view[ablation_view["davies_bouldin"].notna()].copy()
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
            st.plotly_chart(fig_db, width="stretch")

        qi_df = ablation_view[ablation_view.get("quality_index").notna()].copy() if "quality_index" in ablation_view.columns else pd.DataFrame()
        if not qi_df.empty:
            fig_qi = px.bar(
                qi_df,
                x="topic_group",
                y="quality_index",
                color="method",
                barmode="group",
                title="Composite quality index by method and topic",
                text="quality_index",
            )
            fig_qi.update_traces(texttemplate="%{text:.3f}", textposition="auto")
            st.plotly_chart(fig_qi, width="stretch")

        if "runtime_seconds" in ablation_view.columns:
            rt_df = ablation_view[ablation_view["runtime_seconds"].notna()].copy()
            if not rt_df.empty:
                fig_rt = px.bar(
                    rt_df,
                    x="topic_group",
                    y="runtime_seconds",
                    color="method",
                    barmode="group",
                    title="Ablation runtime by method and topic",
                )
                st.plotly_chart(fig_rt, width="stretch")

        burst_view = pd.DataFrame()
        if "ablation_family" in ablation_view.columns:
            burst_view = ablation_view[ablation_view["ablation_family"] == "burst_scoring"].copy()
        if not burst_view.empty:
            st.markdown("#### Burst Scoring On vs Off")

            if "mean_cluster_suspicion" in burst_view.columns:
                fig_burst_mean = px.bar(
                    burst_view,
                    x="topic_group",
                    y="mean_cluster_suspicion",
                    color="method",
                    barmode="group",
                    title="Mean cluster suspicion with vs without burst scoring",
                )
                st.plotly_chart(fig_burst_mean, width="stretch")

            if "topk_mean_suspicion" in burst_view.columns:
                fig_burst_topk = px.bar(
                    burst_view,
                    x="topic_group",
                    y="topk_mean_suspicion",
                    color="method",
                    barmode="group",
                    title="Top-k mean suspicion with vs without burst scoring",
                )
                st.plotly_chart(fig_burst_topk, width="stretch")
    else:
        st.info(
            "Ablation report not found. Run scripts/TFIDFBaseline.py and then "
            "scripts/PrepareDashboardData.py."
        )

    if not eval_df.empty:
        st.subheader("Detailed Evaluation Report")
        st.dataframe(eval_df, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pipeline**: DataCuration -> Embeddings -> Clustering -> Temporal -> Evaluation -> PrepareDashboardData -> Dashboard"
)
st.sidebar.markdown("Built with Streamlit + Plotly")
