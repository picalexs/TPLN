"""
Streamlit Dashboard
==============================
Interactive dashboard for exploring coordinated campaign detection results.
Loads only pre-computed CSV/NPY files.

Run with:
    streamlit run Dashboard.py
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Coordinated Campaign Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")

CLUSTER_CSV   = os.path.join(data_dir, "clusters", "clustered_data.csv")
CONFIG_CSV    = os.path.join(data_dir, "clusters", "hdbscan_config_results.csv")
TEMPORAL_CSV  = os.path.join(data_dir, "temporal", "cluster_temporal_stats.csv")
ABLATION_CSV  = os.path.join(data_dir, "tfidf_ablation_report.csv")
EVAL_CSV      = os.path.join(data_dir, "evaluation_report.csv")


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data
def load_clustered():
    if not os.path.exists(CLUSTER_CSV):
        return None
    df = pd.read_csv(CLUSTER_CSV, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp_date"] = df["timestamp"].dt.strftime("%Y-%m-%d").fillna("")
    return df

@st.cache_data
def load_temporal():
    if not os.path.exists(TEMPORAL_CSV):
        return None
    return pd.read_csv(TEMPORAL_CSV)

@st.cache_data
def load_config():
    if not os.path.exists(CONFIG_CSV):
        return None
    return pd.read_csv(CONFIG_CSV)

@st.cache_data
def load_ablation():
    if not os.path.exists(ABLATION_CSV):
        return None
    return pd.read_csv(ABLATION_CSV)


clustered_df = load_clustered()
temporal_df  = load_temporal()
config_df    = load_config()
ablation_df  = load_ablation()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.title("Campaign Detector")
st.sidebar.markdown("---")

if clustered_df is not None:
    topics = sorted(clustered_df["topic_group"].dropna().unique())
    selected_topics = st.sidebar.multiselect(
        "Filter by topic", topics, default=topics
    )

    min_cluster_size = st.sidebar.slider(
        "Min cluster size", 1, 100, 5
    )

    if temporal_df is not None and "burst_score" in temporal_df.columns:
        max_burst = int(temporal_df["burst_score"].max()) if len(temporal_df) > 0 else 1
        min_burst = st.sidebar.slider(
            "Min burst score", 0, max(max_burst, 1), 0
        )
    else:
        min_burst = 0

    valid_dates = clustered_df["timestamp"].dropna()
    if len(valid_dates) > 0:
        date_min = valid_dates.min().date()
        date_max = valid_dates.max().date()
        date_range = st.sidebar.date_input(
            "Date range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
        )
    else:
        date_range = None
else:
    selected_topics = []
    min_cluster_size = 5
    min_burst = 0
    date_range = None


# ---------------------------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------------------------
st.title("Coordinated Campaign Detection Dashboard")
st.markdown(
    "Semantic clustering + temporal burst analysis on Romanian news (RoLargeSum)"
)

if clustered_df is None:
    st.error(
        "No data found. Run the pipeline first:\n\n"
        "```\n"
        "python DataCuration.py\n"
        "python EmbeddingsClustering.py\n"
        "python TemporalAnalysis.py\n"
        "python Evaluation.py\n"
        "python TFIDFBaseline.py\n"
        "```"
    )
    st.stop()


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Cluster Explorer",
    "Timeline and Bursts",
    "Top Campaigns",
    "Evaluation and Ablation",
])


# ===========================================================================
# TAB 1: CLUSTER EXPLORER
# ===========================================================================
with tab1:
    st.header("Cluster Explorer")

    view_df = clustered_df[clustered_df["topic_group"].isin(selected_topics)].copy()
    view_df = view_df[view_df["cluster"] != -1].copy()

    if "cluster_size" in view_df.columns:
        view_df = view_df[view_df["cluster_size"] >= min_cluster_size]

    if date_range and len(date_range) == 2:
        d_start, d_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        mask = view_df["timestamp"].notna()
        view_df = view_df[~mask | ((view_df["timestamp"] >= d_start) & (view_df["timestamp"] <= d_end))]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Articles shown", len(view_df))
    col2.metric("Unique clusters", view_df["cluster"].nunique())
    col3.metric("Topics", view_df["topic_group"].nunique())
    ts_pct = 100 * view_df["timestamp"].notna().sum() / max(len(view_df), 1)
    col4.metric("With timestamp", f"{ts_pct:.0f}%")

    # UMAP scatter plot
    if "umap_x" in view_df.columns and "umap_y" in view_df.columns:
        scatter_df = view_df[view_df["umap_x"].notna()].copy()
        scatter_df["cluster_str"] = scatter_df["cluster"].astype(str)

        if len(scatter_df) > 0:
            fig_scatter = px.scatter(
                scatter_df,
                x="umap_x", y="umap_y",
                color="topic_group",
                hover_data=["title", "cluster", "topic_group"],
                title="UMAP 2-D Projection (colored by topic)",
                opacity=0.6,
                height=600,
            )
            fig_scatter.update_layout(
                xaxis_title="UMAP-1", yaxis_title="UMAP-2",
                legend_title="Topic",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            fig_scatter2 = px.scatter(
                scatter_df,
                x="umap_x", y="umap_y",
                color="cluster_str",
                hover_data=["title", "cluster", "topic_group"],
                title="UMAP 2-D Projection (colored by cluster ID)",
                opacity=0.6,
                height=600,
            )
            fig_scatter2.update_layout(
                xaxis_title="UMAP-1", yaxis_title="UMAP-2",
                showlegend=False,
            )
            st.plotly_chart(fig_scatter2, use_container_width=True)
        else:
            st.info("No UMAP data available for the selected filters.")
    else:
        st.info("UMAP coordinates not found. Re-run EmbeddingsClustering.py.")

    # Cluster table
    st.subheader("Cluster Summary")
    if temporal_df is not None:
        display_cols = ["cluster", "topic_group", "article_count",
                        "burst_score", "burst_duration_days", "span_days",
                        "suspicion_score", "representative_title"]
        # Add new columns if they exist
        for col in ["burst_score_daily", "burst_score_weekly", "burst_stable",
                     "total_articles", "timestamped_articles"]:
            if col in temporal_df.columns and col not in display_cols:
                display_cols.insert(-1, col)  # before representative_title

        available_cols = [c for c in display_cols if c in temporal_df.columns]

        filtered_temporal = temporal_df[
            temporal_df["topic_group"].isin(selected_topics) &
            (temporal_df["article_count"] >= min_cluster_size) &
            (temporal_df["burst_score"] >= min_burst)
        ].sort_values("suspicion_score", ascending=False)

        if len(filtered_temporal) > 0:
            st.dataframe(
                filtered_temporal[available_cols].head(50),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No clusters match the current filters.")
    else:
        st.info("Temporal stats not available. Run TemporalAnalysis.py.")

    # Article viewer
    st.subheader("Article Viewer")
    available_clusters = sorted(view_df["cluster"].unique())
    if available_clusters:
        selected_cluster = st.selectbox(
            "Select a cluster to view articles",
            available_clusters,
        )
        cluster_articles = view_df[view_df["cluster"] == selected_cluster]
        st.write(f"**{len(cluster_articles)} articles** in cluster {selected_cluster}")
        st.dataframe(
            cluster_articles[["title", "topics", "timestamp_date", "url"]]
            .rename(columns={"timestamp_date": "date"})
            .head(30),
            use_container_width=True,
            hide_index=True,
        )


# ===========================================================================
# TAB 2: TIMELINE & BURSTS
# ===========================================================================
with tab2:
    st.header("Timeline & Burst Analysis")

    if temporal_df is not None and len(temporal_df) > 0:
        burst_clusters = temporal_df[
            temporal_df["topic_group"].isin(selected_topics)
        ].sort_values("suspicion_score", ascending=False)

        if len(burst_clusters) > 0:
            cluster_options = burst_clusters.apply(
                lambda r: (
                    f"Cluster {int(r['cluster'])} - {r['topic_group']} "
                    f"(burst={r['burst_score']}, arts={r['article_count']})"
                ),
                axis=1
            ).tolist()
            cluster_ids_list = burst_clusters["cluster"].tolist()

            selected_idx = st.selectbox(
                "Select cluster", range(len(cluster_options)),
                format_func=lambda i: cluster_options[i],
            )
            sel_cluster_id = int(cluster_ids_list[selected_idx])

            # Show stats card
            sel_stats = temporal_df[temporal_df["cluster"] == sel_cluster_id]
            if len(sel_stats) > 0:
                row = sel_stats.iloc[0]

                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Total articles", row.get("total_articles", row.get("article_count", "?")))
                mcol2.metric("Timestamped", row.get("timestamped_articles", "?"))
                mcol3.metric("Suspicion", f"{row.get('suspicion_score', 0):.1f}")

                burst_d = row.get("burst_score_daily", row.get("burst_score", "?"))
                burst_w = row.get("burst_score_weekly", "?")
                stable = "Yes" if row.get("burst_stable", 0) == 1 else "No"
                mcol4.metric("Burst (D/W/Stable)", f"{burst_d}/{burst_w}/{stable}")

                st.markdown(f"""
                **Cluster {sel_cluster_id}** - {row.get('topic_group', 'N/A')}

                | Metric | Value |
                |---|---|
                | Date range | {row.get('first_seen', '?')} to {row.get('last_seen', '?')} |
                | Span (days) | {row.get('span_days', 'N/A')} |
                | Burst daily | {burst_d} |
                | Burst weekly | {burst_w} |
                | Burst stable | {stable} |
                | Representative | {str(row.get('representative_title', ''))[:120]} |
                """)

            # Build daily timeline
            cluster_data = clustered_df[
                (clustered_df["cluster"] == sel_cluster_id) &
                (clustered_df["timestamp"].notna())
            ].copy()

            if len(cluster_data) > 0:
                daily = cluster_data.set_index("timestamp").resample("D").size()
                daily = daily.reset_index()
                daily.columns = ["date", "count"]

                fig_timeline = px.bar(
                    daily, x="date", y="count",
                    title=f"Daily article count - Cluster {sel_cluster_id}",
                )
                fig_timeline.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Articles per day",
                    height=400,
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

                # Weekly view
                weekly = cluster_data.set_index("timestamp").resample("W").size()
                weekly = weekly.reset_index()
                weekly.columns = ["week", "count"]

                fig_weekly = px.bar(
                    weekly, x="week", y="count",
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
        else:
            st.info("No clusters match the selected topics.")

        # Global burst overview
        st.subheader("Burst Score Distribution")
        fig_burst_dist = px.histogram(
            temporal_df, x="burst_score",
            title="Distribution of daily burst scores across all clusters",
            nbins=max(int(temporal_df["burst_score"].max()) + 1, 5),
        )
        st.plotly_chart(fig_burst_dist, use_container_width=True)

        # Burst vs size scatter
        st.subheader("Burst Score vs Cluster Size")
        size_col = "suspicion_score"
        susp = temporal_df[temporal_df[size_col] > 0].copy()
        if len(susp) > 0:
            fig_burst_size = px.scatter(
                susp,
                x="article_count", y="burst_score",
                color="topic_group",
                size=size_col,
                hover_data=["cluster", "representative_title"],
                title="Cluster size vs Burst score (bubble size = suspicion)",
                height=500,
            )
            st.plotly_chart(fig_burst_size, use_container_width=True)
    else:
        st.info("Temporal stats not available. Run TemporalAnalysis.py first.")


# ===========================================================================
# TAB 3: TOP CAMPAIGNS
# ===========================================================================
with tab3:
    st.header("Top Suspected Coordinated Campaigns")
    st.markdown(
        "Clusters ranked by suspicion score - combining semantic cohesion, "
        "temporal burst strength, and cluster size."
    )

    if temporal_df is not None and clustered_df is not None:
        top_campaigns = temporal_df.sort_values("suspicion_score", ascending=False).head(15)

        for rank, (_, camp) in enumerate(top_campaigns.iterrows(), 1):
            cid = int(camp["cluster"])
            topic = camp.get("topic_group", "?")
            score = camp.get("suspicion_score", 0)
            total = int(camp.get("total_articles", camp.get("article_count", 0)))
            ts_count = int(camp.get("timestamped_articles", 0))
            burst_d = int(camp.get("burst_score_daily", camp.get("burst_score", 0)))
            burst_w = int(camp.get("burst_score_weekly", 0))
            stable = camp.get("burst_stable", 0)
            first = camp.get("first_seen", "?")
            last = camp.get("last_seen", "?")
            rep = str(camp.get("representative_title", ""))[:150]

            severity = "HIGH" if score > 20 else "MEDIUM" if score > 10 else "LOW"

            with st.expander(f"[{severity}] #{rank} - Cluster {cid} ({topic}) - Score: {score:.1f}", expanded=(rank <= 3)):
                ecol1, ecol2, ecol3 = st.columns(3)
                ecol1.metric("Total articles", total)
                ecol2.metric("Burst (D/W)", f"{burst_d}/{burst_w}")
                ecol3.metric("Stable burst?", "Yes" if stable else "No")

                st.markdown(f"**Period:** {first} to {last}")
                st.markdown(f"**Representative:** {rep}")

                # Show sample articles from this cluster
                arts = clustered_df[clustered_df["cluster"] == cid]
                if len(arts) > 0:
                    st.dataframe(
                        arts[["title", "topics", "timestamp_date", "url"]]
                        .rename(columns={"timestamp_date": "date"})
                        .head(10),
                        use_container_width=True,
                        hide_index=True,
                    )
    else:
        st.info("Run the full pipeline first.")


# ===========================================================================
# TAB 4: EVALUATION & ABLATION
# ===========================================================================
with tab4:
    st.header("Evaluation & Ablation Studies")

    # HDBSCAN Config Results
    if config_df is not None:
        st.subheader("HDBSCAN Config Sweep Results")
        st.dataframe(config_df, use_container_width=True, hide_index=True)

        best = (
            config_df.sort_values("selection_score", ascending=False)
            .groupby("topic_group")
            .first()
            .reset_index()
        )
        fig_sil = px.bar(
            best, x="topic_group", y="silhouette",
            color="topic_group",
            title="Best Silhouette Score per Topic (SBERT + HDBSCAN)",
            text="silhouette",
        )
        fig_sil.update_traces(texttemplate="%{text:.3f}", textposition="auto")
        st.plotly_chart(fig_sil, use_container_width=True)

        fig_noise = px.bar(
            best, x="topic_group", y="noise_percent",
            color="topic_group",
            title="Noise % per Topic (best config)",
            text="noise_percent",
        )
        fig_noise.update_traces(texttemplate="%{text:.1f}%", textposition="auto")
        st.plotly_chart(fig_noise, use_container_width=True)

    # Ablation Comparison
    if ablation_df is not None:
        st.subheader("Ablation: TF-IDF/KMeans vs SBERT/KMeans vs SBERT/HDBSCAN")

        st.dataframe(ablation_df, use_container_width=True, hide_index=True)

        sil_df = ablation_df[ablation_df["silhouette"].notna()].copy()
        if len(sil_df) > 0:
            fig_ablation = px.bar(
                sil_df,
                x="topic_group", y="silhouette",
                color="method", barmode="group",
                title="Silhouette Score Comparison by Method & Topic",
                text="silhouette",
            )
            fig_ablation.update_traces(texttemplate="%{text:.3f}", textposition="auto")
            st.plotly_chart(fig_ablation, use_container_width=True)

        db_df = ablation_df[ablation_df["davies_bouldin"].notna()].copy()
        if len(db_df) > 0:
            fig_db = px.bar(
                db_df,
                x="topic_group", y="davies_bouldin",
                color="method", barmode="group",
                title="Davies-Bouldin Index Comparison (lower = better)",
                text="davies_bouldin",
            )
            fig_db.update_traces(texttemplate="%{text:.3f}", textposition="auto")
            st.plotly_chart(fig_db, use_container_width=True)
    else:
        st.info("Ablation report not found. Run TFIDFBaseline.py first.")

    # Evaluation Report
    if os.path.exists(EVAL_CSV):
        st.subheader("Detailed Evaluation Report")
        eval_df = pd.read_csv(EVAL_CSV)
        st.dataframe(eval_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pipeline**: DataCuration -> Embeddings -> Clustering -> Temporal -> Evaluation -> Dashboard"
)
st.sidebar.markdown("Built with Streamlit + Plotly")
