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

from html import escape
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.campaign_scoring import add_campaign_candidate_columns
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
    "representative_title",
    "cluster",
    "topic_group",
    "article_count",
    "burst_score",
    "burst_duration_days",
    "span_days",
    "suspicion_score",
    "campaign_candidate_score",
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
    "organic_event_title",
    "public_affairs_signal",
    "campaign_support_weight",
    "campaign_recurrence_weight",
    "campaign_active_days_weight",
    "campaign_source_diversity_weight",
    "campaign_span_weight",
    "campaign_narrative_weight",
]

PAGE_OPTIONS = [
    "Cluster Explorer",
    "Timeline and Bursts",
    "Top Campaigns",
    "Similarity Map",
    "Topic Health",
    "Evaluation and Ablation",
]

COLOR_SEQUENCE = [
    "#60a5fa",
    "#2dd4bf",
    "#f59e0b",
    "#a78bfa",
    "#fb7185",
    "#a3e635",
    "#38bdf8",
    "#facc15",
    "#818cf8",
    "#34d399",
    "#fb923c",
    "#94a3b8",
]

PLOT_LAYOUT = {
    "template": "plotly_dark",
    "font": {"family": "Inter, Segoe UI, sans-serif", "size": 13, "color": "#dbeafe"},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "hoverlabel": {
        "bgcolor": "#020617",
        "font": {"color": "#f8fafc", "family": "Inter, Segoe UI, sans-serif"},
        "bordercolor": "#334155",
    },
    "title": {"font": {"color": "#e5edf8", "size": 16}},
    "legend": {
        "orientation": "v",
        "yanchor": "top",
        "y": 1,
        "xanchor": "left",
        "x": 1.02,
        "font": {"color": "#cbd5e1", "size": 11},
        "title": {"font": {"color": "#94a3b8", "size": 11}},
        "bgcolor": "rgba(15, 23, 42, 0.72)",
        "bordercolor": "rgba(148, 163, 184, 0.18)",
        "borderwidth": 1,
    },
    "margin": {"l": 12, "r": 170, "t": 82, "b": 28},
}

METRIC_HELP = {
    "campaign_candidate_score": (
        "Report-facing rank for compact campaign candidates. It starts from broad "
        "suspicion and rewards support, recurrence, active days, source diversity, "
        "compact span, and public-affairs narrative signal."
    ),
    "suspicion_score": (
        "Broad temporal suspiciousness. Useful for exploration, but intentionally "
        "generous because organic news events can also be bursty."
    ),
    "burst_score_daily": "Number of daily burst intervals detected for a semantic cluster.",
    "burst_score_weekly": "Weekly burst score used as a stability check for daily spikes.",
    "timestamp_coverage_ratio": "Share of articles in a group with a usable timestamp.",
    "peak_to_baseline_ratio": "How large the highest daily peak is relative to the cluster baseline.",
    "top_domain_share": "Share of timestamped articles from the most frequent source domain.",
    "noise_rate": "Share of topic articles that HDBSCAN left as noise rather than assigning to a real cluster.",
    "silhouette": "Higher is better. Measures how well separated clusters are in embedding space.",
    "davies_bouldin": "Lower is better. Measures cluster compactness and separation.",
    "graph_cohesion": "Nearest-neighbor cohesion summary from the semantic similarity graph.",
}

LINK_COLUMN_NAMES = {
    "url": "URL",
    "source_url": "Source URL",
    "neighbor_url": "Neighbor URL",
    "representative_url": "Representative URL",
}


def inject_app_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --dash-bg: #070b14;
            --dash-surface: #0f172a;
            --dash-surface-2: #111c31;
            --dash-border: rgba(148, 163, 184, 0.22);
            --dash-ink: #e5edf8;
            --dash-muted: #94a3b8;
            --dash-accent: #60a5fa;
            --dash-accent-strong: #38bdf8;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.18), transparent 32rem),
                linear-gradient(180deg, #070b14 0%, #0b1220 48%, #090d16 100%);
            color: var(--dash-ink);
        }
        header[data-testid="stHeader"] {
            background: rgba(7, 11, 20, 0.86);
            border-bottom: 1px solid rgba(148, 163, 184, 0.14);
        }
        .stApp p, .stApp li, .stApp label, .stMarkdown, .stCaptionContainer {
            color: var(--dash-muted);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #070b14 0%, #0f172a 100%);
            color: #e5e7eb;
            border-right: 1px solid rgba(148, 163, 184, 0.16);
        }
        [data-testid="stSidebar"] * {
            color: inherit;
        }
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stMultiSelect label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stDateInput label {
            color: #cbd5e1;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"] {
            background-color: #172554;
            border: 1px solid rgba(148, 163, 184, 0.35);
            color: #e0f2fe;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"] span {
            color: #e0f2fe;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            border-color: rgba(148, 163, 184, 0.30);
            background-color: #111827;
        }
        [data-testid="stSidebar"] [data-baseweb="radio"] span {
            color: #e5edf8;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            background: #111827;
            color: #e5edf8;
        }
        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }
        h1, h2, h3 {
            color: var(--dash-ink);
            letter-spacing: 0;
        }
        h1 {
            font-size: 2.2rem;
            line-height: 1.1;
            margin-bottom: 0.25rem;
        }
        h2 {
            margin-top: 1.1rem;
        }
        [data-testid="stMetric"] {
            background:
                linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(15, 23, 42, 0.84));
            border: 1px solid var(--dash-border);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.24);
        }
        [data-testid="stMetricLabel"] {
            color: #9fb1c9;
            font-weight: 700;
        }
        [data-testid="stMetricValue"] {
            color: #f8fafc;
            font-size: 1.65rem;
            line-height: 1.15;
        }
        .dash-intro {
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(17, 28, 49, 0.92));
            border: 1px solid var(--dash-border);
            border-left: 4px solid var(--dash-accent);
            border-radius: 8px;
            padding: 1rem 1.15rem;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.22);
            margin: 0.4rem 0 1.15rem 0;
        }
        .dash-intro strong {
            display: block;
            color: #f8fafc;
            font-size: 0.95rem;
            margin-bottom: 0.2rem;
        }
        .dash-intro span {
            color: #b6c4d7;
            font-size: 0.92rem;
            line-height: 1.55;
        }
        .stDataFrame {
            border: 1px solid var(--dash-border);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.22);
        }
        .dark-table-wrap {
            max-height: 620px;
            overflow: auto;
            border: 1px solid var(--dash-border);
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.92);
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.22);
        }
        .dark-table {
            width: 100%;
            border-collapse: collapse;
            min-width: 820px;
            font-size: 0.84rem;
            color: #dbeafe;
        }
        .dark-table th {
            position: sticky;
            top: 0;
            z-index: 1;
            background: #111c31;
            color: #bfdbfe;
            text-align: left;
            font-weight: 800;
            border-bottom: 1px solid rgba(148, 163, 184, 0.22);
            padding: 0.62rem 0.7rem;
            white-space: nowrap;
        }
        .dark-table td {
            border-bottom: 1px solid rgba(148, 163, 184, 0.13);
            padding: 0.54rem 0.7rem;
            vertical-align: top;
            color: #dbeafe;
            max-width: 420px;
        }
        .dark-table tr:nth-child(even) td {
            background: rgba(30, 41, 59, 0.34);
        }
        .dark-table tr:hover td {
            background: rgba(96, 165, 250, 0.10);
        }
        .dark-table a {
            color: #7dd3fc;
            font-weight: 700;
            text-decoration: underline;
            text-underline-offset: 2px;
        }
        .table-note {
            color: #94a3b8;
            font-size: 0.82rem;
            margin: 0.35rem 0 0.8rem 0;
        }
        .stPlotlyChart {
            background:
                linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.86));
            border: 1px solid var(--dash-border);
            border-radius: 8px;
            padding: 0.75rem;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.24);
        }
        div[data-testid="stExpander"] {
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid var(--dash-border);
            border-radius: 8px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.20);
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] p {
            color: #dbeafe;
        }
        div[data-testid="stAlert"] {
            background: rgba(15, 23, 42, 0.95);
            color: #dbeafe;
            border: 1px solid var(--dash-border);
        }
        div[data-testid="stMarkdownContainer"] code {
            background: rgba(96, 165, 250, 0.14);
            color: #bfdbfe;
            border: 1px solid rgba(96, 165, 250, 0.20);
            border-radius: 4px;
        }
        .small-note {
            color: #94a3b8;
            font-size: 0.86rem;
            line-height: 1.45;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background-color: #111827;
            border-color: rgba(148, 163, 184, 0.26);
            color: #e5edf8;
        }
        div[data-baseweb="popover"] {
            background: #0f172a;
            border: 1px solid var(--dash-border);
        }
        div[role="listbox"],
        ul[role="listbox"],
        [data-baseweb="menu"] {
            background: #0f172a;
            border: 1px solid var(--dash-border);
            color: #e5edf8 !important;
        }
        div[role="option"],
        li[role="option"],
        div[role="option"] *,
        li[role="option"] *,
        div[data-baseweb="popover"] * {
            background: #0f172a;
            color: #e5edf8 !important;
        }
        div[role="option"]:hover,
        li[role="option"]:hover {
            background: rgba(96, 165, 250, 0.18);
            color: #f8fafc !important;
        }
        div[data-baseweb="select"] input,
        div[data-baseweb="input"] input {
            color: #e5edf8;
        }
        button[kind="secondary"],
        button[data-testid="baseButton-secondary"] {
            background: #111827;
            border: 1px solid rgba(148, 163, 184, 0.28);
            color: #e5edf8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_intro(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="dash-intro">
            <strong>{title}</strong>
            <span>{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_value(value: object, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, (int, float)):
        if abs(float(value)) >= 1000:
            return f"{float(value):,.0f}{suffix}"
        return f"{float(value):.2f}{suffix}" if isinstance(value, float) and not float(value).is_integer() else f"{int(value)}{suffix}"
    return f"{value}{suffix}"


def percent_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def apply_plot_style(fig: go.Figure, *, height: int | None = None, showlegend: bool | None = None) -> go.Figure:
    fig.update_layout(**PLOT_LAYOUT)
    if height is not None:
        fig.update_layout(height=height)
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    if showlegend is False:
        fig.update_layout(margin={"l": 12, "r": 24, "t": 72, "b": 28})
    fig.update_layout(
        coloraxis_colorbar={
            "tickfont": {"color": "#94a3b8"},
            "title": {"font": {"color": "#bfdbfe"}},
        }
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.16)",
        zeroline=False,
        title_font={"size": 12, "color": "#bfdbfe"},
        tickfont={"color": "#94a3b8"},
        linecolor="rgba(148, 163, 184, 0.18)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.16)",
        zeroline=False,
        title_font={"size": 12, "color": "#bfdbfe"},
        tickfont={"color": "#94a3b8"},
        linecolor="rgba(148, 163, 184, 0.18)",
    )
    return fig


def plot_chart(fig: go.Figure, *, height: int | None = None, showlegend: bool | None = None) -> None:
    st.plotly_chart(
        apply_plot_style(fig, height=height, showlegend=showlegend),
        width="stretch",
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )


def explain_metrics(metrics: dict[str, str]) -> None:
    with st.expander("How to read these metrics"):
        for label, description in metrics.items():
            st.markdown(f"**{label}:** {description}")


def available_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def dataframe_with_links(frame: pd.DataFrame, **kwargs: object) -> None:
    """Render a compact dark table with known URL columns as clickable links."""
    display_frame = frame.copy()
    for column in LINK_COLUMN_NAMES:
        if column not in display_frame.columns:
            continue
        display_frame[column] = display_frame[column].fillna("").astype(str).str.strip()
        has_value = display_frame[column].ne("")
        has_scheme = display_frame[column].str.startswith(("http://", "https://"))
        display_frame.loc[has_value & ~has_scheme, column] = "https://" + display_frame.loc[has_value & ~has_scheme, column]

    max_rows = int(kwargs.pop("max_rows", 500))
    if "width" in kwargs:
        kwargs.pop("width")
    if "hide_index" in kwargs:
        kwargs.pop("hide_index")
    visible_frame = display_frame.head(max_rows)

    def format_cell(column: str, value: object) -> str:
        if value is None or pd.isna(value):
            return ""
        if column in LINK_COLUMN_NAMES:
            url = str(value).strip()
            if not url:
                return ""
            return f'<a href="{escape(url)}" target="_self">Open</a>'
        if isinstance(value, pd.Timestamp):
            return escape(value.strftime("%Y-%m-%d"))
        if isinstance(value, float):
            return escape(f"{value:.4g}")
        text = str(value)
        if len(text) > 220:
            text = text[:217] + "..."
        return escape(text)

    header_html = "".join(f"<th>{escape(str(column))}</th>" for column in visible_frame.columns)
    row_html = []
    for _, row in visible_frame.iterrows():
        cells = "".join(
            f"<td>{format_cell(str(column), row[column])}</td>"
            for column in visible_frame.columns
        )
        row_html.append(f"<tr>{cells}</tr>")

    st.markdown(
        f"""
        <div class="dark-table-wrap">
            <table class="dark-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{''.join(row_html)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if len(display_frame) > len(visible_frame):
        st.markdown(
            f'<div class="table-note">Showing {len(visible_frame):,} of {len(display_frame):,} rows.</div>',
            unsafe_allow_html=True,
        )


def ensure_dataframe(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Return an empty DataFrame when an optional loader returns None."""
    return frame if frame is not None else pd.DataFrame()


def ensure_campaign_candidate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add report-facing campaign-candidate columns for older dashboard assets."""
    if frame.empty or "campaign_candidate_score" in frame.columns:
        return frame

    required = {
        "suspicion_score",
        "total_articles",
        "active_days",
        "burst_periods_daily",
        "domain_count",
        "span_days",
        "topic_group",
        "representative_title",
    }
    if not required.issubset(frame.columns):
        return frame

    try:
        return add_campaign_candidate_columns(frame)
    except Exception:
        return frame


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


def limit_plot_rows(frame: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    if frame.empty or len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)


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
    return ensure_campaign_candidate_columns(pd.read_parquet(DASHBOARD_TEMPORAL_PARQUET))


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
def load_neighbors_for_source(topic_group: str, source_idx: int) -> pd.DataFrame:
    if not DASHBOARD_NEIGHBORS_PARQUET.exists():
        return pd.DataFrame()

    columns = [
        "topic_group",
        "source_idx",
        "neighbor_idx",
        "rank",
        "score",
        "neighbor_cluster",
        "neighbor_title",
        "neighbor_domain",
        "neighbor_url",
    ]
    filters = [
        ("topic_group", "==", str(topic_group)),
        ("source_idx", "==", int(source_idx)),
    ]
    try:
        df = pd.read_parquet(DASHBOARD_NEIGHBORS_PARQUET, columns=columns, filters=filters)
    except Exception:
        # Fallback keeps memory down by reading only the columns required for the UI.
        df = pd.read_parquet(DASHBOARD_NEIGHBORS_PARQUET, columns=columns)
        df = df[
            (df["topic_group"] == str(topic_group))
            & (pd.to_numeric(df["source_idx"], errors="coerce") == int(source_idx))
        ].copy()
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
temporal_df = ensure_dataframe(load_temporal() if not missing_assets else None)

inject_app_css()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.title("Campaign Detector")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Page",
    PAGE_OPTIONS,
    index=0,
    help="Switch between exploration, temporal evidence, campaign candidates, similarity, topic health, and evaluation views.",
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")

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
selected_topics = st.sidebar.multiselect(
    "Filter by topic",
    topics,
    default=topics,
    help="Restricts every chart and table to the selected normalized topic groups.",
)

max_cluster_size = int(cluster_overview_df["cluster_size"].max()) if not cluster_overview_df.empty else 100
min_cluster_size = st.sidebar.slider(
    "Min cluster size",
    1,
    max(max_cluster_size, 1),
    5,
    help="Hide very small semantic clusters when reviewing patterns or campaign candidates.",
)

if not temporal_df.empty:
    burst_col = "burst_score" if "burst_score" in temporal_df.columns else "burst_score_daily"
    max_burst = int(temporal_df[burst_col].max()) if len(temporal_df) > 0 else 1
    min_burst = st.sidebar.slider(
        "Min burst score",
        0,
        max(max_burst, 1),
        0,
        help=METRIC_HELP["burst_score_daily"],
    )
    max_susp = float(temporal_df["suspicion_score"].max()) if "suspicion_score" in temporal_df.columns and len(temporal_df) > 0 else 0.0
    min_suspicion = st.sidebar.slider(
        "Min suspicion score",
        0.0,
        max(max_susp, 0.0),
        0.0,
        0.5,
        help=METRIC_HELP["suspicion_score"],
    )
else:
    min_burst = 0
    min_suspicion = 0.0

if dashboard_meta and pd.notna(dashboard_meta.get("min_timestamp")) and pd.notna(dashboard_meta.get("max_timestamp")):
    date_min = pd.Timestamp(dashboard_meta["min_timestamp"]).date()
    date_max = pd.Timestamp(dashboard_meta["max_timestamp"]).date()
    date_range_input = st.sidebar.date_input(
        "Date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
        help="Limits timestamped articles and daily timelines. Undated articles remain available in article views.",
    )
    if isinstance(date_range_input, (tuple, list)) and len(date_range_input) == 2:
        selected_start = pd.Timestamp(date_range_input[0])
        selected_end = pd.Timestamp(date_range_input[1])
        if selected_start.date() == date_min and selected_end.date() == date_max:
            date_range = None
        else:
            date_range = (selected_start, selected_end)
    else:
        date_range = None
else:
    date_range = None

st.sidebar.markdown("---")
st.sidebar.caption(
    "Metrics, tables, and timelines use full parquet assets. Only the UMAP "
    "scatter uses a stratified sample for rendering speed."
)

if st.sidebar.button(
    "Warm page caches",
    help="Preload medium dashboard assets into Streamlit's cache. This is synchronous, not a true browser background preload, and it skips the very large neighbor asset.",
):
    with st.sidebar:
        with st.spinner("Warming page caches..."):
            ensure_dataframe(load_scatter_sample())
            ensure_dataframe(load_cluster_daily())
            ensure_dataframe(load_cluster_similarity())
            ensure_dataframe(load_config())
            ensure_dataframe(load_runtime_profile())
            ensure_dataframe(load_ablation())
            ensure_dataframe(load_eval())
    st.sidebar.success("Page caches warmed for this Streamlit session.")


# ---------------------------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------------------------
visible_clusters_df = build_visible_clusters(
    cluster_overview_df=cluster_overview_df,
    cluster_daily_df=ensure_dataframe(load_cluster_daily()) if date_range is not None else pd.DataFrame(),
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


# ===========================================================================
# PAGE: CLUSTER EXPLORER
# ===========================================================================
if page == "Cluster Explorer":
    st.title("Coordinated Campaign Detection Dashboard")
    st.markdown(
        "Semantic clustering + temporal burst analysis on Romanian news (RoLargeSum)"
    )

    if dashboard_meta:
        st.caption(
            f"Dashboard assets generated at {dashboard_meta.get('generated_at', 'unknown')} "
            f"from {dashboard_meta.get('total_rows', 0):,} articles."
        )

        meta_cols = st.columns(5)
        meta_cols[0].metric(
            "Articles",
            f"{int(dashboard_meta.get('total_rows', 0)):,}",
            help="Total article rows included in the prepared dashboard assets.",
        )
        meta_cols[1].metric(
            "Real clusters",
            f"{int(dashboard_meta.get('real_cluster_count', 0)):,}",
            help="HDBSCAN clusters excluding noise points.",
        )
        meta_cols[2].metric(
            "Noise rate",
            percent_value(dashboard_meta.get("noise_rate", 0)),
            help=METRIC_HELP["noise_rate"],
        )
        meta_cols[3].metric(
            "Timestamp coverage",
            percent_value(dashboard_meta.get("timestamp_coverage_ratio", 0)),
            help=METRIC_HELP["timestamp_coverage_ratio"],
        )
        meta_cols[4].metric(
            "Topics",
            f"{int(dashboard_meta.get('topic_count', 0)):,}",
            help="Number of normalized topic groups in the dashboard data.",
        )

    with st.expander("Dashboard reading guide", expanded=False):
        st.markdown(
            """
            This dashboard is an evidence browser, not an automatic verdict. Start with
            campaign candidates for ranked cases, then open timeline evidence to inspect
            whether bursts are compact, recurring, and supported by reliable timestamps.
            Use the cluster explorer for article-level evidence and the evaluation pages
            to check whether the semantic clustering layer is healthy.
            """
        )

    st.header("Cluster Explorer")
    page_intro(
        "Explore semantic clusters and article evidence",
        "Use this page to move from a corpus-level map to a specific cluster, then inspect the articles and nearest-neighbor evidence behind it.",
    )

    articles_shown = int(visible_clusters_df["articles_shown"].sum()) if not visible_clusters_df.empty else 0
    timestamped_shown = int(visible_clusters_df["in_range_timestamped"].sum()) if not visible_clusters_df.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Articles shown", f"{articles_shown:,}", help="Articles visible after topic, size, and date filters.")
    col2.metric(
        "Unique clusters",
        f"{int(visible_clusters_df['cluster'].nunique()):,}" if not visible_clusters_df.empty else "0",
        help="Distinct HDBSCAN clusters passing the current filters.",
    )
    col3.metric(
        "Topics",
        int(visible_clusters_df["topic_group"].nunique()) if not visible_clusters_df.empty else 0,
        help="Normalized topic groups represented in the filtered result set.",
    )
    ts_pct = 100 * timestamped_shown / max(articles_shown, 1)
    col4.metric("With timestamp", f"{ts_pct:.0f}%", help=METRIC_HELP["timestamp_coverage_ratio"])

    scatter_sample_df = ensure_dataframe(load_scatter_sample())
    scatter_display_df = filter_sample_for_display(
        scatter_df=scatter_sample_df,
        cluster_overview_df=cluster_overview_df,
        selected_topics=selected_topics,
        min_cluster_size=min_cluster_size,
        date_range=date_range,
    )
    max_umap_points = st.slider(
        "Max UMAP points",
        min_value=2_000,
        max_value=max(2_000, min(25_000, len(scatter_display_df))),
        value=min(8_000, max(2_000, len(scatter_display_df))),
        step=1_000,
        help="Lower values make the scatter render faster. Metrics and tables still use the full prepared assets.",
    )
    scatter_plot_df = limit_plot_rows(scatter_display_df, max_umap_points)

    st.caption(
        f"UMAP scatter is rendering {len(scatter_plot_df):,} of {len(scatter_display_df):,} sampled points "
        "so the plot stays responsive without changing the underlying metrics."
    )

    if not visible_clusters_df.empty:
        topic_mix = (
            visible_clusters_df.groupby("topic_group", as_index=False)
            .agg(articles=("articles_shown", "sum"), clusters=("cluster", "nunique"))
            .sort_values("articles", ascending=False)
        )
        mix_col, health_col = st.columns([1.05, 1])
        with mix_col:
            fig_topic_mix = px.bar(
                topic_mix,
                x="articles",
                y="topic_group",
                color="topic_group",
                orientation="h",
                color_discrete_sequence=COLOR_SEQUENCE,
                title="Filtered articles by topic",
                hover_data={"articles": ":,", "clusters": ":,", "topic_group": False},
            )
            fig_topic_mix.update_layout(yaxis={"categoryorder": "total ascending"})
            plot_chart(fig_topic_mix, height=360, showlegend=False)
        with health_col:
            cluster_health = visible_clusters_df.copy()
            fig_cluster_health = px.scatter(
                cluster_health,
                x="cluster_size",
                y="timestamp_coverage_ratio",
                color="topic_group",
                size="peak_day_count" if "peak_day_count" in cluster_health.columns else None,
                color_discrete_sequence=COLOR_SEQUENCE,
                title="Cluster size vs timestamp coverage",
                hover_data=["cluster", "representative_title", "active_days", "peak_day_share"],
            )
            fig_cluster_health.update_layout(
                xaxis_title="Cluster size",
                yaxis_title="Timestamp coverage",
                yaxis_tickformat=".0%",
            )
            plot_chart(fig_cluster_health, height=360, showlegend=False)

    if not scatter_plot_df.empty:
        fig_scatter = px.scatter(
            scatter_plot_df,
            x="umap_x",
            y="umap_y",
            color="topic_group",
            color_discrete_sequence=COLOR_SEQUENCE,
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
        fig_scatter.update_traces(
            marker={"size": 7, "line": {"width": 0}},
            hovertemplate="<b>%{customdata[0]}</b><br>Cluster %{customdata[1]}<br>Topic %{customdata[2]}<extra></extra>",
        )
        plot_chart(fig_scatter, height=560, showlegend=False)

        if st.toggle(
            "Show cluster-colored UMAP",
            value=False,
            help="This can be slower because it creates one color group per visible cluster.",
        ):
            fig_scatter_cluster = px.scatter(
                scatter_plot_df,
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
            fig_scatter_cluster.update_traces(marker={"size": 7, "line": {"width": 0}})
            plot_chart(fig_scatter_cluster, height=560, showlegend=False)
    else:
        st.info("No UMAP sample rows match the selected filters.")

    st.subheader("Cluster Summary")
    explain_metrics(
        {
            "Burst score": METRIC_HELP["burst_score_daily"],
            "Suspicion score": METRIC_HELP["suspicion_score"],
            "Candidate score": METRIC_HELP["campaign_candidate_score"],
            "Peak to baseline": METRIC_HELP["peak_to_baseline_ratio"],
        }
    )
    if not filtered_temporal_df.empty:
        available_cols = [c for c in CLUSTER_SUMMARY_COLUMNS if c in filtered_temporal_df.columns]
        for col in EXTRA_TEMPORAL_COLUMNS:
            if col in filtered_temporal_df.columns and col not in available_cols:
                available_cols.append(col)

        dataframe_with_links(
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
            dataframe_with_links(
                cluster_articles[["title", "topics", "timestamp_date", "url"]]
                .rename(columns={"timestamp_date": "date"})
                .head(30),
                width="stretch",
                hide_index=True,
            )

            st.markdown("### Semantic Neighbor Evidence")
            if DASHBOARD_NEIGHBORS_PARQUET.exists() and "topic_row_idx" in cluster_articles.columns:
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
                    src_url = str(src.get("url", "")).strip()
                    if src_url.startswith(("http://", "https://")):
                        st.markdown(
                            f'<a href="{escape(src_url)}" target="_self">Open selected source article</a>',
                            unsafe_allow_html=True,
                        )
                    if st.button("Load semantic neighbors", key=f"load_neighbors_{src_topic}_{src_row_idx}"):
                        with st.spinner("Loading semantic neighbors for the selected article..."):
                            src_neighbors = load_neighbors_for_source(src_topic, src_row_idx)
                        if not src_neighbors.empty:
                            src_neighbors = src_neighbors.sort_values("rank").head(20)
                            dataframe_with_links(
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
    page_intro(
        "Inspect whether semantic clusters are temporally concentrated",
        "A campaign candidate should show more than a large cluster. Look for compact peaks, repeated burst intervals, adequate timestamp coverage, and source patterns that support the score.",
    )
    explain_metrics(
        {
            "Daily burst": METRIC_HELP["burst_score_daily"],
            "Weekly burst": METRIC_HELP["burst_score_weekly"],
            "Peak to baseline": METRIC_HELP["peak_to_baseline_ratio"],
            "Timestamp coverage": METRIC_HELP["timestamp_coverage_ratio"],
            "Suspicion score": METRIC_HELP["suspicion_score"],
        }
    )

    if not filtered_temporal_df.empty:
        burst_col = "burst_score" if "burst_score" in filtered_temporal_df.columns else "burst_score_daily"
        size_col = "article_count" if "article_count" in filtered_temporal_df.columns else "total_articles"

        has_candidate_score = (
            "campaign_candidate_score" in filtered_temporal_df.columns
            and (filtered_temporal_df["campaign_candidate_score"].fillna(0) > 0).any()
        )
        ranking_choices = [
            ("Campaign candidate score", "campaign_candidate_score"),
            ("Suspicion score", "suspicion_score"),
            ("Daily burst score", burst_col),
            ("Article count", size_col),
            ("Peak to baseline", "peak_to_baseline_ratio"),
            ("Active days", "active_days"),
        ]
        ranking_choices = [(label, col) for label, col in ranking_choices if col in filtered_temporal_df.columns]
        rank_labels = [label for label, _ in ranking_choices]
        preferred_rank = "Campaign candidate score" if has_candidate_score else "Suspicion score"
        default_rank_idx = rank_labels.index(preferred_rank) if preferred_rank in rank_labels else 0

        rank_cols = st.columns([1, 1])
        rank_label = rank_cols[0].selectbox(
            "Rank clusters by",
            rank_labels,
            index=default_rank_idx,
            help="The first row in this ranking becomes the default selected cluster.",
        )
        rank_col = dict(ranking_choices)[rank_label]

        timeline_df = filtered_temporal_df.copy()
        only_candidates = False
        if has_candidate_score:
            only_candidates = rank_cols[1].checkbox(
                "Only campaign candidates",
                value=True,
                help="Keeps the timeline selector focused on compact report-facing candidates. Turn this off to inspect broad-suspicion outliers.",
            )
            if only_candidates:
                candidate_view = timeline_df[timeline_df["campaign_candidate_score"].fillna(0) > 0].copy()
                if not candidate_view.empty:
                    timeline_df = candidate_view

        timeline_df["_rank_value"] = pd.to_numeric(timeline_df[rank_col], errors="coerce").fillna(0)
        timeline_df = timeline_df.sort_values(["_rank_value", size_col], ascending=[False, False]).reset_index(drop=True)
        timeline_df["_rank"] = timeline_df.index + 1

        st.info(
            "Cluster selection is deterministic. The default is the top row after the current sidebar filters, "
            f"ranked by {rank_label.lower()}. Broad suspicion can surface long, sparse news clusters; "
            "campaign-candidate score is stricter and is the better default for report-facing review."
        )

        cluster_options = timeline_df.apply(
            lambda r: (
                f"#{int(r['_rank'])} Cluster {int(r['cluster'])} - {r['topic_group']} "
                f"({rank_label.lower()}={float(r['_rank_value']):.2f}, "
                f"candidate={float(r.get('campaign_candidate_score', 0) or 0):.2f}, "
                f"burst={int(r[burst_col])}, active={int(r.get('active_days', 0) or 0)}, "
                f"span={int(r.get('span_days', 0) or 0)}d, arts={int(r[size_col])})"
            ),
            axis=1,
        ).tolist()
        cluster_ids_list = timeline_df["cluster"].tolist()

        selected_idx = st.selectbox(
            "Select cluster",
            range(len(cluster_options)),
            format_func=lambda i: cluster_options[i],
        )
        sel_cluster_id = int(cluster_ids_list[selected_idx])
        st.caption(
            f"The selected cluster is rank #{int(timeline_df.iloc[selected_idx]['_rank'])} "
            f"by {rank_label.lower()} under the current sidebar filters."
        )

        sel_stats = timeline_df[timeline_df["cluster"] == sel_cluster_id]
        if not sel_stats.empty:
            row = sel_stats.iloc[0]
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric(
                "Total articles",
                f"{int(row.get('total_articles', row.get('article_count', 0))):,}",
                help="Total articles assigned to the selected semantic cluster.",
            )
            mcol2.metric(
                "Timestamped",
                f"{int(row.get('timestamped_articles', 0)):,}",
                help=METRIC_HELP["timestamp_coverage_ratio"],
            )
            mcol3.metric("Suspicion", f"{row.get('suspicion_score', 0):.1f}", help=METRIC_HELP["suspicion_score"])

            burst_d = row.get("burst_score_daily", row.get("burst_score", 0))
            burst_w = row.get("burst_score_weekly", 0)
            stable = "Yes" if row.get("burst_stable", 0) == 1 else "No"
            mcol4.metric(
                "Burst D/W/Stable",
                f"{int(burst_d)}/{int(burst_w)}/{stable}",
                help="Daily burst count, weekly burst count, and whether both granularities agree.",
            )

            coverage_ratio = row.get("timestamp_coverage_ratio")
            peak_ratio = row.get("peak_to_baseline_ratio")
            support_weight = row.get("support_weight")
            coverage_weight = row.get("coverage_weight")
            source_weight = row.get("source_weight")
            domain_weight = row.get("domain_weight")
            candidate_score = float(row.get("campaign_candidate_score", 0) or 0)
            span_days = float(row.get("span_days", 0) or 0)
            active_days = float(row.get("active_days", 0) or 0)

            if has_candidate_score and candidate_score <= 0:
                st.warning(
                    "This cluster is not a campaign candidate. It appears here only because the current filters "
                    "also allow broad temporal suspicion, which can be high for long, sparse clusters."
                )
            elif span_days > 365 and active_days > 0 and span_days / max(active_days, 1) > 30:
                st.info(
                    "This cluster is spread across a long calendar span with relatively few active days. "
                    "Use the active-day view and component weights before treating it as coordinated activity."
                )

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

            weight_cols = [
                "support_weight",
                "coverage_weight",
                "source_weight",
                "domain_weight",
                "campaign_support_weight",
                "campaign_recurrence_weight",
                "campaign_active_days_weight",
                "campaign_source_diversity_weight",
                "campaign_span_weight",
                "campaign_narrative_weight",
            ]
            weight_df = pd.DataFrame(
                [
                    {"component": col.replace("_", " "), "value": float(row.get(col))}
                    for col in weight_cols
                    if col in row.index and pd.notna(row.get(col))
                ]
            )
            if not weight_df.empty:
                fig_weights = px.bar(
                    weight_df,
                    x="value",
                    y="component",
                    orientation="h",
                    title="Score component weights for selected cluster",
                    color="value",
                    color_continuous_scale=["#1e293b", "#38bdf8"],
                    hover_data={"value": ":.3f", "component": False},
                )
                fig_weights.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
                plot_chart(fig_weights, height=380, showlegend=False)

        cluster_daily = load_daily_for_cluster(sel_cluster_id)
        if not cluster_daily.empty:
            cluster_daily = cluster_daily.sort_values("date").copy()
            peak_row = cluster_daily.sort_values("article_count", ascending=False).iloc[0]
            active_dates = cluster_daily["date"].dropna().drop_duplicates().sort_values()
            date_gaps = active_dates.diff().dt.days.dropna()
            median_gap = float(date_gaps.median()) if not date_gaps.empty else 0.0
            active_metric_cols = st.columns(4)
            active_metric_cols[0].metric("Active days", f"{cluster_daily['date'].nunique():,}")
            active_metric_cols[1].metric("Peak articles in one day", f"{int(peak_row['article_count']):,}")
            active_metric_cols[2].metric("Peak date", pd.Timestamp(peak_row["date"]).strftime("%Y-%m-%d"))
            active_metric_cols[3].metric(
                "Median gap",
                f"{median_gap:.0f} days",
                help="Median number of days between active publication dates for this cluster.",
            )

            timeline_granularity = st.radio(
                "Timeline view",
                ["Active publication days", "Weekly aggregation", "Monthly aggregation"],
                horizontal=True,
                help="Active publication days removes empty dates from the visual emphasis. Weekly and monthly views show broader bursts.",
            )

            if timeline_granularity == "Active publication days":
                fig_timeline = go.Figure()
                fig_timeline.add_trace(
                    go.Bar(
                        x=cluster_daily["date"],
                        y=cluster_daily["article_count"],
                        name="Articles",
                        marker_color="#60a5fa",
                        hovertemplate="%{x|%Y-%m-%d}<br>Articles: %{y}<extra></extra>",
                    )
                )
                fig_timeline.add_trace(
                    go.Scatter(
                        x=cluster_daily["date"],
                        y=cluster_daily["article_count"].rolling(3, min_periods=1).mean(),
                        name="3-active-day average",
                        mode="lines",
                        line={"color": "#2dd4bf", "width": 2},
                        hovertemplate="%{x|%Y-%m-%d}<br>Average: %{y:.2f}<extra></extra>",
                    )
                )
                fig_timeline.update_layout(
                    title=f"Active publication days - Cluster {sel_cluster_id}",
                    xaxis_title="Date",
                    yaxis_title="Articles on active day",
                    bargap=0.22,
                )
                fig_timeline.update_xaxes(rangeslider_visible=True)
                fig_timeline.add_hline(
                    y=float(cluster_daily["article_count"].mean()),
                    line_dash="dash",
                    line_color="#94a3b8",
                    annotation_text="Active-day mean",
                    annotation_position="top left",
                )
                plot_chart(fig_timeline, height=460, showlegend=True)
            else:
                rule = "W-SUN" if timeline_granularity == "Weekly aggregation" else "ME"
                label = "week" if timeline_granularity == "Weekly aggregation" else "month"
                aggregated = (
                    cluster_daily.set_index("date")["article_count"]
                    .resample(rule)
                    .sum()
                    .reset_index()
                    .rename(columns={"article_count": "count", "date": label})
                )
                aggregated = aggregated[aggregated["count"] > 0].copy()
                fig_aggregated = px.bar(
                    aggregated,
                    x=label,
                    y="count",
                    title=f"{timeline_granularity} - Cluster {sel_cluster_id}",
                    color_discrete_sequence=["#2dd4bf"],
                )
                fig_aggregated.update_layout(
                    xaxis_title=label.title(),
                    yaxis_title=f"Articles per {label}",
                )
                fig_aggregated.update_xaxes(rangeslider_visible=True)
                plot_chart(fig_aggregated, height=430, showlegend=False)
        else:
            st.warning(f"No timestamped articles in cluster {sel_cluster_id}.")

        st.subheader("Filtered Burst Landscape")
        st.caption(
            "These plots use the same filtered set as the selector above. If campaign-only is enabled, "
            "broad-suspicion artifacts are hidden from the landscape."
        )
        landscape_df = timeline_df.copy()
        landscape_col1, landscape_col2 = st.columns([1, 1])
        with landscape_col1:
            fig_burst_dist = px.histogram(
                landscape_df,
                x=burst_col,
                title="Distribution of daily burst scores",
                nbins=max(int(landscape_df[burst_col].max()) + 1, 5),
                color_discrete_sequence=["#60a5fa"],
            )
            fig_burst_dist.update_layout(xaxis_title="Burst score", yaxis_title="Clusters")
            plot_chart(fig_burst_dist, height=360, showlegend=False)
        with landscape_col2:
            top_bursts = landscape_df.sort_values([burst_col, rank_col], ascending=[False, False]).head(15).copy()
            if not top_bursts.empty:
                top_bursts["cluster_label"] = top_bursts["cluster"].map(lambda value: f"Cluster {int(value)}")
                fig_top_burst = px.bar(
                    top_bursts,
                    x=burst_col,
                    y="cluster_label",
                    color="topic_group",
                    orientation="h",
                    color_discrete_sequence=COLOR_SEQUENCE,
                    title="Top burst-heavy clusters",
                    hover_data=["representative_title", "suspicion_score"],
                )
                fig_top_burst.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Burst score")
                plot_chart(fig_top_burst, height=360, showlegend=False)

        landscape_df = landscape_df.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=[size_col, burst_col])
        if not landscape_df.empty:
            score_color_col = "campaign_candidate_score" if has_candidate_score else "suspicion_score"
            if {"active_days", "peak_day_count"}.issubset(landscape_df.columns):
                hover_cols = [
                    col for col in [
                        "cluster",
                        "topic_group",
                        size_col,
                        burst_col,
                        "suspicion_score",
                        "campaign_candidate_score",
                        "peak_to_baseline_ratio",
                        "median_gap_days",
                        "representative_title",
                    ]
                    if col in landscape_df.columns
                ]
                fig_burst_size = px.scatter(
                    landscape_df,
                    x="active_days",
                    y="peak_day_count",
                    color=score_color_col if score_color_col in landscape_df.columns else None,
                    size=size_col,
                    color_continuous_scale=["#334155", "#38bdf8", "#facc15"],
                    hover_data=hover_cols,
                    title="Temporal concentration landscape (bubble size = articles)",
                    height=500,
                )
                fig_burst_size.update_layout(
                    xaxis_title="Active publication days",
                    yaxis_title="Peak articles in one day",
                )
            else:
                fig_burst_size = px.scatter(
                    landscape_df,
                    x=size_col,
                    y=burst_col,
                    color="topic_group",
                    size="suspicion_score" if "suspicion_score" in landscape_df.columns else None,
                    color_discrete_sequence=COLOR_SEQUENCE,
                    hover_data=["cluster", "representative_title"],
                    title="Cluster size vs burst score (fallback view)",
                    height=500,
                )
                fig_burst_size.update_layout(xaxis_title="Cluster size", yaxis_title="Burst score")
            plot_chart(fig_burst_size, height=500, showlegend=False)
    else:
        st.info("No temporal clusters match the selected filters.")


# ===========================================================================
# PAGE: TOP CAMPAIGNS
# ===========================================================================
elif page == "Top Campaigns":
    st.header("Top Compact Campaign Candidates")
    page_intro(
        "Review report-facing campaign candidates",
        "This view ranks compact, evidence-supported candidates rather than every bursty news story. Use the component charts and expanders to see why a cluster moved up or down.",
    )
    explain_metrics(
        {
            "Candidate score": METRIC_HELP["campaign_candidate_score"],
            "Broad suspicion": METRIC_HELP["suspicion_score"],
            "Top domain share": METRIC_HELP["top_domain_share"],
            "Public-affairs signal": "Narrative weight indicating whether the representative title looks relevant to public affairs.",
            "Organic-event filter": "Penalty marker for obvious organic news events that can create legitimate publication bursts.",
        }
    )

    if not filtered_temporal_df.empty:
        campaign_df = ensure_campaign_candidate_columns(filtered_temporal_df)
        if "campaign_candidate_score" in campaign_df.columns:
            campaign_df = campaign_df[campaign_df["campaign_candidate_score"].fillna(0) > 0].copy()
            campaign_df = campaign_df.sort_values("campaign_candidate_score", ascending=False)
            score_col = "campaign_candidate_score"
            score_label = "Candidate score"
        else:
            campaign_df = campaign_df.sort_values("suspicion_score", ascending=False)
            score_col = "suspicion_score"
            score_label = "Suspicion score"
            st.warning(
                "Campaign-candidate columns are missing from the temporal asset, "
                "so this view is falling back to broad suspicion score. Re-run "
                "`scripts/TemporalAnalysis.py` and `scripts/PrepareDashboardData.py` "
                "to refresh the dashboard data."
            )

        top_campaigns = campaign_df.head(15)
        if not top_campaigns.empty:
            chart_df = top_campaigns.copy()
            chart_df["cluster_label"] = chart_df.apply(
                lambda r: f"#{int(r['cluster'])} {str(r.get('topic_group', ''))}",
                axis=1,
            )
            score_col1, score_col2 = st.columns([1.05, 1])
            with score_col1:
                fig_ranked = px.bar(
                    chart_df,
                    x=score_col,
                    y="cluster_label",
                    color="topic_group",
                    orientation="h",
                    color_discrete_sequence=COLOR_SEQUENCE,
                    title=f"Top candidates by {score_label.lower()}",
                    hover_data=["representative_title", "suspicion_score", "total_articles"],
                )
                fig_ranked.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title=score_label)
                plot_chart(fig_ranked, height=450, showlegend=False)
            with score_col2:
                component_cols = available_columns(
                    chart_df,
                    [
                        "campaign_support_weight",
                        "campaign_recurrence_weight",
                        "campaign_active_days_weight",
                        "campaign_source_diversity_weight",
                        "campaign_span_weight",
                        "campaign_narrative_weight",
                    ],
                )
                if component_cols:
                    component_df = chart_df[["cluster_label"] + component_cols].melt(
                        id_vars="cluster_label",
                        var_name="component",
                        value_name="weight",
                    )
                    component_df["component"] = component_df["component"].str.replace("_", " ", regex=False)
                    fig_components = px.bar(
                        component_df,
                        x="weight",
                        y="cluster_label",
                        color="component",
                        orientation="h",
                        title="Candidate score component profile",
                        hover_data={"weight": ":.3f", "cluster_label": False},
                    )
                    fig_components.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Component weight")
                    plot_chart(fig_components, height=450)

        for rank, (_, camp) in enumerate(top_campaigns.iterrows(), 1):
            cid = int(camp["cluster"])
            topic = camp.get("topic_group", "?")
            score = float(camp.get(score_col, 0) or 0)
            suspicion_score = float(camp.get("suspicion_score", 0) or 0)
            total = int(camp.get("total_articles", camp.get("article_count", 0)))
            burst_d = int(camp.get("burst_score_daily", camp.get("burst_score", 0)))
            burst_w = int(camp.get("burst_score_weekly", 0))
            stable = camp.get("burst_stable", 0)
            first = camp.get("first_seen", "?")
            last = camp.get("last_seen", "?")
            rep = str(camp.get("representative_title", ""))[:150]

            if score_col == "campaign_candidate_score":
                severity = "HIGH" if score >= 5 else "MEDIUM" if score >= 2 else "LOW"
            else:
                severity = "HIGH" if score > 20 else "MEDIUM" if score > 10 else "LOW"

            with st.expander(
                f"[{severity}] #{rank} - Cluster {cid} ({topic}) - {score_label}: {score:.2f}",
                expanded=(rank <= 3),
            ):
                ecol1, ecol2, ecol3, ecol4 = st.columns(4)
                ecol1.metric(score_label, f"{score:.2f}", help=METRIC_HELP.get(score_col, METRIC_HELP["suspicion_score"]))
                ecol2.metric("Broad suspicion", f"{suspicion_score:.1f}", help=METRIC_HELP["suspicion_score"])
                ecol3.metric("Articles", f"{total:,}", help="Article support behind this candidate cluster.")
                ecol4.metric("Burst D/W", f"{burst_d}/{burst_w}", help="Daily and weekly burst interval counts.")

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
                    "campaign_support_weight",
                    "campaign_recurrence_weight",
                    "campaign_active_days_weight",
                    "campaign_source_diversity_weight",
                    "campaign_span_weight",
                    "campaign_narrative_weight",
                    "organic_event_title",
                    "public_affairs_signal",
                ]
                explain_payload = {c: camp.get(c) for c in explain_cols if c in camp.index}
                if explain_payload and st.button("Show score details", key=f"campaign_score_details_{cid}"):
                    explain_df = pd.DataFrame(
                        [
                            {"metric": key, "value": value}
                            for key, value in explain_payload.items()
                        ]
                    )
                    dataframe_with_links(explain_df, width="stretch", hide_index=True)

                detail_cols = st.columns(2)
                if detail_cols[0].button("Load article examples", key=f"campaign_articles_{cid}"):
                    with st.spinner(f"Loading article examples for cluster {cid}..."):
                        arts = load_articles_for_cluster(cid)
                    if not arts.empty:
                        dataframe_with_links(
                            arts[["title", "topics", "timestamp_date", "url"]]
                            .rename(columns={"timestamp_date": "date"})
                            .head(10),
                            width="stretch",
                            hide_index=True,
                        )

                if detail_cols[1].button("Load storyline", key=f"campaign_storyline_{cid}"):
                    storyline = load_daily_for_cluster(cid)
                    if not storyline.empty:
                        story_fig = px.area(
                            storyline.sort_values("date"),
                            x="date",
                            y="article_count",
                            title=f"Campaign storyline for cluster {cid}",
                            color_discrete_sequence=["#60a5fa"],
                        )
                        story_fig.update_layout(height=260)
                        plot_chart(story_fig, height=280, showlegend=False)
        if top_campaigns.empty:
            st.info("No compact campaign candidates match the current filters.")
    else:
        st.info("No campaigns match the current filters.")


# ===========================================================================
# PAGE: SIMILARITY MAP
# ===========================================================================
elif page == "Similarity Map":
    st.header("Cross-Cluster Similarity Map")
    page_intro(
        "Find clusters that sit close together in semantic space",
        "Edges connect representative cluster anchors that are close in the UMAP projection. Treat this as a navigation aid for related narratives, not as a final campaign score.",
    )
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
            sim_cols = st.columns(4)
            sim_cols[0].metric("Edges shown", f"{len(graph_df):,}", help="Highest-similarity cluster-neighbor pairs after topic filtering.")
            sim_cols[1].metric("Median similarity", f"{graph_df['similarity'].median():.3f}", help="Similarity is computed as 1 / (1 + representative UMAP distance).")
            sim_cols[2].metric("Source clusters", f"{graph_df['cluster'].nunique():,}", help="Distinct source clusters participating in the edge table.")
            sim_cols[3].metric("Topics", f"{graph_df['topic_group'].nunique():,}", help="Filtered topic groups represented in the similarity map.")

            sim_chart_col, edge_chart_col = st.columns([1, 1])
            with sim_chart_col:
                fig_sim_dist = px.histogram(
                    graph_df,
                    x="similarity",
                    nbins=40,
                    title="Similarity distribution across retained edges",
                    color_discrete_sequence=["#60a5fa"],
                )
                fig_sim_dist.update_layout(xaxis_title="Similarity", yaxis_title="Edges")
                plot_chart(fig_sim_dist, height=340, showlegend=False)
            with edge_chart_col:
                top_edge_topics = (
                    graph_df.groupby("topic_group", as_index=False)
                    .agg(edges=("cluster", "size"), median_similarity=("similarity", "median"))
                    .sort_values("edges", ascending=False)
                )
                fig_edge_topics = px.bar(
                    top_edge_topics,
                    x="edges",
                    y="topic_group",
                    color="median_similarity",
                    orientation="h",
                    color_continuous_scale=["#1e293b", "#38bdf8"],
                    title="Similarity edges by topic",
                    hover_data={"median_similarity": ":.3f", "edges": ":,"},
                )
                fig_edge_topics.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
                plot_chart(fig_edge_topics, height=340, showlegend=False)

            # Use full anchor sample (not date-filtered display sample) so edge drawing
            # remains stable even when sidebar date/size filters are restrictive.
            scatter_anchor_df = ensure_dataframe(load_scatter_sample())
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
            if not cluster_positions.empty and st.toggle(
                "Render cluster similarity graph",
                value=False,
                help="Graph rendering is heavier than the summary charts. Turn this on when you need the network view.",
            ):
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
                            line=dict(width=1.1, color="rgba(71,85,105,0.30)"),
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
                            marker=dict(size=8, line=dict(width=0.8, color="#e2e8f0")),
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
                plot_chart(fig_graph, height=700, showlegend=False)
                if drawn_edges == 0:
                    st.warning(
                        "No edges could be projected onto current cluster anchors. "
                        "Re-run PrepareDashboardData.py if this persists."
                    )

            if st.button("Load similarity edge table"):
                dataframe_with_links(
                    graph_df[["topic_group", "cluster", "neighbor_cluster", "similarity", "distance", "rank"]],
                    width="stretch",
                    hide_index=True,
                )


# ===========================================================================
# PAGE: TOPIC HEALTH
# ===========================================================================
elif page == "Topic Health":
    st.header("Topic Health Board")
    page_intro(
        "Check data coverage, clustering quality, and runtime by topic",
        "This page helps diagnose whether a topic has enough timestamp coverage, reasonable noise, and acceptable clustering behavior before using it for campaign interpretation.",
    )
    explain_metrics(
        {
            "Noise rate": METRIC_HELP["noise_rate"],
            "Timestamp coverage": METRIC_HELP["timestamp_coverage_ratio"],
            "Silhouette": METRIC_HELP["silhouette"],
            "Davies-Bouldin": METRIC_HELP["davies_bouldin"],
            "Graph cohesion": METRIC_HELP["graph_cohesion"],
        }
    )
    config_df = ensure_dataframe(load_config())
    runtime_df = ensure_dataframe(load_runtime_profile())

    if topic_summary_df.empty:
        st.info("Topic summary is unavailable.")
    else:
        health_df = topic_summary_df[topic_summary_df["topic_group"].isin(selected_topics)].copy()
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

        health_metric_cols = st.columns(4)
        health_metric_cols[0].metric("Topic rows", f"{int(health_df['total_rows'].sum()):,}", help="Total source articles across visible topic groups.")
        health_metric_cols[1].metric("Real clusters", f"{int(health_df['real_cluster_count'].sum()):,}", help="Total non-noise clusters across visible topic groups.")
        health_metric_cols[2].metric("Mean noise", percent_value(health_df["noise_rate"].mean()), help=METRIC_HELP["noise_rate"])
        health_metric_cols[3].metric("Mean timestamp coverage", percent_value(health_df["timestamp_coverage_ratio"].mean()), help=METRIC_HELP["timestamp_coverage_ratio"])

        health_chart_col1, health_chart_col2 = st.columns([1, 1])
        with health_chart_col1:
            fig_topic_rows = px.bar(
                health_df.sort_values("total_rows", ascending=False),
                x="total_rows",
                y="topic_group",
                color="real_cluster_count",
                orientation="h",
                color_continuous_scale=["#1e293b", "#38bdf8"],
                title="Corpus size and cluster count by topic",
                hover_data={"total_rows": ":,", "real_cluster_count": ":,"},
            )
            fig_topic_rows.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            plot_chart(fig_topic_rows, height=390, showlegend=False)
        with health_chart_col2:
            fig_health_scatter = px.scatter(
                health_df,
                x="noise_rate",
                y="timestamp_coverage_ratio",
                size="total_rows",
                color="topic_group",
                color_discrete_sequence=COLOR_SEQUENCE,
                title="Noise rate vs timestamp coverage",
                hover_data=["real_cluster_count", "mean_real_cluster_size"],
            )
            fig_health_scatter.update_layout(
                xaxis_title="Noise rate",
                yaxis_title="Timestamp coverage",
                xaxis_tickformat=".0%",
                yaxis_tickformat=".0%",
            )
            plot_chart(fig_health_scatter, height=390, showlegend=False)

        if {"silhouette", "davies_bouldin"}.issubset(health_df.columns):
            quality_df = health_df.dropna(subset=["silhouette", "davies_bouldin"]).copy()
            if not quality_df.empty:
                fig_quality = px.scatter(
                    quality_df,
                    x="davies_bouldin",
                    y="silhouette",
                    size="real_cluster_count",
                    color="topic_group",
                    color_discrete_sequence=COLOR_SEQUENCE,
                    title="Clustering quality tradeoff by topic",
                    hover_data=["noise_percent", "graph_cohesion", "graph_cohesion_real"],
                )
                fig_quality.update_layout(
                    xaxis_title="Davies-Bouldin (lower is better)",
                    yaxis_title="Silhouette (higher is better)",
                )
                plot_chart(fig_quality, height=430, showlegend=False)

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
        dataframe_with_links(health_df[health_cols], width="stretch", hide_index=True)

        if not runtime_df.empty:
            st.subheader("Runtime Observability")
            timed = runtime_df[runtime_df["topic_group"] != "__GLOBAL__"].copy()
            if not timed.empty:
                stage_cols = available_columns(
                    timed,
                    ["embedding_seconds", "faiss_seconds", "umap_seconds", "hdbscan_sweep_seconds", "label_apply_seconds"],
                )
                if stage_cols:
                    runtime_long = timed[["topic_group"] + stage_cols].melt(
                        id_vars="topic_group",
                        var_name="stage",
                        value_name="seconds",
                    )
                    runtime_long["stage"] = runtime_long["stage"].str.replace("_seconds", "", regex=False).str.replace("_", " ", regex=False)
                    fig_runtime = px.bar(
                        runtime_long,
                        x="seconds",
                        y="topic_group",
                        color="stage",
                        orientation="h",
                        title="Runtime breakdown by topic",
                        hover_data={"seconds": ":.2f"},
                    )
                    fig_runtime.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Seconds")
                    plot_chart(fig_runtime, height=460)

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
                dataframe_with_links(
                    timed[runtime_cols].sort_values("topic_total_seconds", ascending=False),
                    width="stretch",
                    hide_index=True,
                )


# ===========================================================================
# PAGE: EVALUATION & ABLATION
# ===========================================================================
else:
    st.header("Evaluation & Ablation Studies")
    page_intro(
        "Compare model choices and clustering quality",
        "Use this page to validate the semantic layer and baseline comparisons before treating temporal signals as meaningful evidence.",
    )
    explain_metrics(
        {
            "Selection score": "Internal HDBSCAN config-selection score balancing cluster quality and noise behavior.",
            "Silhouette": METRIC_HELP["silhouette"],
            "Davies-Bouldin": METRIC_HELP["davies_bouldin"],
            "Quality index": "Composite ablation metric combining normalized silhouette and inverse Davies-Bouldin.",
            "Runtime": "Wall-clock seconds recorded for each method or pipeline stage.",
        }
    )
    config_df = ensure_dataframe(load_config())
    ablation_df = ensure_dataframe(load_ablation())
    eval_df = ensure_dataframe(load_eval())

    if not config_df.empty:
        st.subheader("HDBSCAN Config Sweep Results")
        cfg_view = config_df[config_df["topic_group"].isin(selected_topics)].copy()
        if cfg_view.empty:
            cfg_view = config_df.copy()

        best = (
            cfg_view.sort_values("selection_score", ascending=False)
            .groupby("topic_group")
            .first()
            .reset_index()
        )

        cfg_col1, cfg_col2 = st.columns([1, 1])
        with cfg_col1:
            fig_selection = px.scatter(
                cfg_view,
                x="noise_percent",
                y="silhouette",
                color="topic_group",
                size="selection_score",
                color_discrete_sequence=COLOR_SEQUENCE,
                title="Config sweep: noise vs silhouette",
                hover_data=["min_cluster_size", "min_samples", "selection_score", "num_clusters"],
            )
            fig_selection.update_layout(
                xaxis_title="Noise percent",
                yaxis_title="Silhouette",
                xaxis_ticksuffix="%",
            )
            plot_chart(fig_selection, height=430, showlegend=False)
        with cfg_col2:
            fig_selection_score = px.bar(
                best.sort_values("selection_score", ascending=False),
                x="selection_score",
                y="topic_group",
                color="topic_group",
                orientation="h",
                color_discrete_sequence=COLOR_SEQUENCE,
                title="Best config selection score by topic",
                hover_data=["silhouette", "noise_percent", "davies_bouldin"],
            )
            fig_selection_score.update_layout(yaxis={"categoryorder": "total ascending"})
            plot_chart(fig_selection_score, height=430, showlegend=False)

        if st.button("Load HDBSCAN config sweep table"):
            dataframe_with_links(cfg_view, width="stretch", hide_index=True)

        fig_sil = px.bar(
            best,
            x="topic_group",
            y="silhouette",
            color="topic_group",
            color_discrete_sequence=COLOR_SEQUENCE,
            title="Best Silhouette Score per Topic (SBERT + HDBSCAN)",
            text="silhouette",
        )
        fig_sil.update_traces(texttemplate="%{text:.3f}", textposition="auto")
        plot_chart(fig_sil, height=390, showlegend=False)

        fig_noise = px.bar(
            best,
            x="topic_group",
            y="noise_percent",
            color="topic_group",
            color_discrete_sequence=COLOR_SEQUENCE,
            title="Noise % per Topic (best config)",
            text="noise_percent",
        )
        fig_noise.update_traces(texttemplate="%{text:.1f}%", textposition="auto")
        plot_chart(fig_noise, height=390, showlegend=False)

    if not ablation_df.empty:
        st.subheader("Ablation: TF-IDF/KMeans vs SBERT/KMeans vs SBERT/HDBSCAN")
        ab_cols = st.columns(3)
        ab_topics = ["All"] + sorted(ablation_df["topic_group"].dropna().astype(str).unique().tolist())
        ab_methods = ["All"] + sorted(ablation_df["method"].dropna().astype(str).unique().tolist())
        ab_topic = ab_cols[0].selectbox("Ablation topic", ab_topics, index=0)
        ab_method = ab_cols[1].selectbox("Ablation method", ab_methods, index=0)
        ab_max_rows = int(ab_cols[2].slider("Rows", min_value=20, max_value=2000, value=300, step=20))

        ablation_view = ablation_df.copy()
        ablation_view = ablation_view[ablation_view["topic_group"].isin(selected_topics)].copy()
        if ab_topic != "All":
            ablation_view = ablation_view[ablation_view["topic_group"] == ab_topic]
        if ab_method != "All":
            ablation_view = ablation_view[ablation_view["method"] == ab_method]

        if st.button("Load ablation table"):
            dataframe_with_links(ablation_view.head(ab_max_rows), width="stretch", hide_index=True)

        if st.toggle(
            "Render detailed ablation charts",
            value=False,
            help="These charts are useful for analysis but add several Plotly renders.",
        ):
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
                    plot_chart(fig_winners, height=330)

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
                plot_chart(fig_ablation, height=430)

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
                plot_chart(fig_db, height=430)

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
                plot_chart(fig_qi, height=430)

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
                    plot_chart(fig_rt, height=430)

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
                    plot_chart(fig_burst_mean, height=390)

                if "topk_mean_suspicion" in burst_view.columns:
                    fig_burst_topk = px.bar(
                        burst_view,
                        x="topic_group",
                        y="topk_mean_suspicion",
                        color="method",
                        barmode="group",
                        title="Top-k mean suspicion with vs without burst scoring",
                    )
                    plot_chart(fig_burst_topk, height=390)
    else:
        st.info(
            "Ablation report not found. Run scripts/TFIDFBaseline.py and then "
            "scripts/PrepareDashboardData.py."
        )

    if not eval_df.empty:
        st.subheader("Detailed Evaluation Report")
        eval_view = eval_df.copy()
        if "topic_group" in eval_view.columns:
            eval_view = eval_view[eval_view["topic_group"].isin(selected_topics)].copy()
        if {"section", "topic_group", "silhouette", "noise_percent"}.issubset(eval_view.columns):
            sil_summary = eval_view[eval_view["section"] == "silhouette"].dropna(subset=["silhouette"]).copy()
            if not sil_summary.empty:
                fig_eval = px.scatter(
                    sil_summary,
                    x="noise_percent",
                    y="silhouette",
                    color="topic_group",
                    color_discrete_sequence=COLOR_SEQUENCE,
                    title="Evaluation summary: silhouette vs noise",
                    hover_data=["num_clusters", "method"],
                )
                fig_eval.update_layout(xaxis_title="Noise percent", yaxis_title="Silhouette")
                plot_chart(fig_eval, height=390, showlegend=False)
        if st.button("Load detailed evaluation table"):
            dataframe_with_links(eval_view, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pipeline**: DataCuration -> Embeddings -> Clustering -> Temporal -> Evaluation -> PrepareDashboardData -> Dashboard"
)
st.sidebar.markdown("Built with Streamlit + Plotly")
