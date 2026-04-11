"""Build parquet-backed dashboard assets from clustered pipeline artefacts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.paths import (
    CLUSTERED_PARQUET,
    DASHBOARD_ABLATION_PARQUET,
    DASHBOARD_ARTICLE_DETAIL,
    DASHBOARD_CLUSTER_DAILY_PARQUET,
    DASHBOARD_CLUSTER_OVERVIEW_PARQUET,
    DASHBOARD_CONFIG_PARQUET,
    DASHBOARD_DIR,
    DASHBOARD_EVAL_PARQUET,
    DASHBOARD_META_PARQUET,
    DASHBOARD_SCATTER_SAMPLE_PARQUET,
    DASHBOARD_TEMPORAL_PARQUET,
    DASHBOARD_TOPIC_SUMMARY_PARQUET,
    EVALUATION_REPORT,
    HDBSCAN_CONFIG_RESULTS,
    TEMPORAL_STATS,
    TFIDF_ABLATION_REPORT,
)
from src.runtime_profile import apply_runtime_profile, detect_runtime_profile, format_runtime_profile


DETAIL_OUTPUT_COLUMNS = [
    "title",
    "summary",
    "topics",
    "url",
    "author",
    "timestamp",
    "timestamp_date",
    "timestamp_source",
    "timestamp_quality",
    "source_domain",
    "topic_group",
    "topic_group_reason",
    "topic_is_eligible",
    "cluster",
    "cluster_size",
    "cluster_assignment_reason",
    "cluster_membership_strength",
    "cluster_outlier_score",
    "umap_x",
    "umap_y",
]

SCATTER_OUTPUT_COLUMNS = [
    "sample_type",
    "topic_group",
    "cluster",
    "cluster_size",
    "title",
    "url",
    "timestamp",
    "source_domain",
    "umap_x",
    "umap_y",
    "cluster_membership_strength",
    "cluster_outlier_score",
]

READ_COLUMNS = [
    "title",
    "summary",
    "topics",
    "url",
    "author",
    "timestamp",
    "timestamp_source",
    "timestamp_quality",
    "topic_group",
    "topic_group_reason",
    "topic_is_eligible",
    "cluster",
    "cluster_size",
    "cluster_assignment_reason",
    "cluster_membership_strength",
    "cluster_outlier_score",
    "umap_x",
    "umap_y",
]


def _extract_domain(url: Any) -> str:
    raw = "" if url is None else str(url).strip().lower()
    if not raw or raw == "nan":
        return ""
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    domain = parsed.netloc or parsed.path.split("/")[0]
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _asset_path(output_dir: Path, template_path: Path) -> Path:
    return output_dir / template_path.name


def _write_parquet_atomic(df: pd.DataFrame, final_path: Path) -> None:
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(tmp_path), compression="zstd")
    tmp_path.replace(final_path)


def _copy_parquet_atomic(source_path: Path, dest_path: Path) -> None:
    if not source_path.exists():
        return
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    shutil.copyfile(source_path, tmp_path)
    tmp_path.replace(dest_path)


def _ensure_columns(frame: pd.DataFrame, columns: list[str], fill_value: Any = np.nan) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = fill_value
    return output.loc[:, columns].copy()


def _iter_cluster_input_frames(
    input_path: Path,
    read_columns: list[str],
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    if input_path.suffix.lower() != ".parquet":
        raise ValueError("PrepareDashboardData.py only supports parquet inputs.")

    parquet_file = pq.ParquetFile(str(input_path))
    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=read_columns):
        yield batch.to_pandas()


def _aggregate_count_frames(count_frames: list[pd.DataFrame], count_column: str) -> pd.DataFrame:
    if not count_frames:
        return pd.DataFrame(columns=["topic_group", count_column])

    combined = pd.concat(count_frames, ignore_index=True)
    grouped = combined.groupby("topic_group", as_index=False)[count_column].sum()
    grouped = grouped.loc[grouped["topic_group"].sort_values().index].reset_index(drop=True)
    return grouped


def _build_mode_frame(
    count_frames: list[pd.DataFrame],
    value_column: str,
    output_column: str,
) -> pd.DataFrame:
    if not count_frames:
        return pd.DataFrame(columns=["topic_group", "cluster", output_column])

    combined = pd.concat(count_frames, ignore_index=True)
    combined = combined.groupby(["topic_group", "cluster", value_column], as_index=False)["count"].sum()
    combined = combined.loc[combined[value_column].sort_values(kind="stable").index]
    combined = combined.loc[combined["count"].sort_values(ascending=False, kind="stable").index]
    combined = combined.loc[combined["cluster"].sort_values(kind="stable").index]
    combined = combined.loc[combined["topic_group"].sort_values(kind="stable").index]
    combined = combined.drop_duplicates(["topic_group", "cluster"], keep="first").reset_index(drop=True)
    combined[output_column] = combined[value_column]
    return combined[["topic_group", "cluster", output_column]].reset_index(drop=True)


def _combine_candidate_rows(candidate_frames: list[pd.DataFrame], best: bool) -> pd.DataFrame:
    if not candidate_frames:
        return pd.DataFrame()

    combined = pd.concat(candidate_frames, ignore_index=True)
    combined["_membership_rank"] = pd.to_numeric(
        combined["cluster_membership_strength"],
        errors="coerce",
    ).fillna(-np.inf if best else np.inf)
    combined["_outlier_rank"] = pd.to_numeric(
        combined["cluster_outlier_score"],
        errors="coerce",
    ).fillna(np.inf if best else -np.inf)

    sort_ascending = [True, True, False, True] if best else [True, True, True, False]
    combined = (
        combined.sort_values(
            by=["topic_group", "cluster", "_membership_rank", "_outlier_rank"],
            ascending=sort_ascending,
        )
        .drop_duplicates(["topic_group", "cluster"], keep="first")
        .drop(columns=["_membership_rank", "_outlier_rank"], errors="ignore")
        .reset_index(drop=True)
    )
    return combined


def build_dashboard_assets(
    input_path: Path,
    output_dir: Path,
    chunk_size: int,
    scatter_cap: int,
    noise_per_topic_cap: int,
) -> dict[str, Any]:
    """Stream clustered parquet once and materialize fast dashboard parquet assets."""

    output_dir.mkdir(parents=True, exist_ok=True)
    article_detail_path = _asset_path(output_dir, DASHBOARD_ARTICLE_DETAIL)
    cluster_overview_path = _asset_path(output_dir, DASHBOARD_CLUSTER_OVERVIEW_PARQUET)
    topic_summary_path = _asset_path(output_dir, DASHBOARD_TOPIC_SUMMARY_PARQUET)
    global_meta_path = _asset_path(output_dir, DASHBOARD_META_PARQUET)
    cluster_daily_path = _asset_path(output_dir, DASHBOARD_CLUSTER_DAILY_PARQUET)
    scatter_sample_path = _asset_path(output_dir, DASHBOARD_SCATTER_SAMPLE_PARQUET)

    rng = np.random.default_rng(42)
    total_rows = 0
    real_rows = 0
    noise_rows = 0
    timestamped_rows = 0

    topic_total_frames: list[pd.DataFrame] = []
    topic_real_frames: list[pd.DataFrame] = []
    topic_noise_frames: list[pd.DataFrame] = []
    topic_timestamp_frames: list[pd.DataFrame] = []
    topic_cluster_frames: list[pd.DataFrame] = []
    cluster_agg_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    topic_reason_count_frames: list[pd.DataFrame] = []
    assignment_reason_count_frames: list[pd.DataFrame] = []
    best_candidate_frames: list[pd.DataFrame] = []
    edge_candidate_frames: list[pd.DataFrame] = []
    noise_samples: dict[str, pd.DataFrame] = {}

    detail_writer: pq.ParquetWriter | None = None
    detail_tmp_path = article_detail_path.with_suffix(article_detail_path.suffix + ".tmp")

    try:
        for chunk_index, chunk in enumerate(
            _iter_cluster_input_frames(input_path, READ_COLUMNS, chunk_size),
            start=1,
        ):
            if chunk.empty:
                continue

            chunk = chunk.copy()
            chunk["topic_group"] = (
                chunk["topic_group"]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace("", "necunoscut")
            )
            chunk["cluster"] = pd.to_numeric(chunk["cluster"], errors="coerce").fillna(-1).astype(int)
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
            chunk["timestamp_date"] = chunk["timestamp"].dt.floor("D")
            chunk["source_domain"] = chunk["url"].map(_extract_domain)
            for text_column in (
                "title",
                "summary",
                "topics",
                "url",
                "author",
                "timestamp_source",
                "timestamp_quality",
                "topic_group_reason",
                "cluster_assignment_reason",
                "source_domain",
            ):
                chunk[text_column] = chunk[text_column].fillna("").astype(str)
            chunk["topic_is_eligible"] = chunk["topic_is_eligible"].fillna(False).astype(bool)
            chunk["cluster_membership_strength"] = pd.to_numeric(
                chunk["cluster_membership_strength"],
                errors="coerce",
            )
            chunk["cluster_outlier_score"] = pd.to_numeric(
                chunk["cluster_outlier_score"],
                errors="coerce",
            )
            if "cluster_size" in chunk.columns:
                chunk["cluster_size"] = pd.to_numeric(chunk["cluster_size"], errors="coerce")
            else:
                chunk["cluster_size"] = np.nan

            total_rows += len(chunk)
            timestamped_rows += int(chunk["timestamp"].notna().sum())

            topic_total_frames.append(
                chunk.groupby("topic_group").size().rename("total_rows").reset_index()
            )
            timestamped_chunk = chunk[chunk["timestamp"].notna()]
            if not timestamped_chunk.empty:
                topic_timestamp_frames.append(
                    timestamped_chunk.groupby("topic_group").size().rename("timestamped_rows").reset_index()
                )

            real_chunk = chunk[chunk["cluster"] != -1].copy()
            noise_chunk = chunk[chunk["cluster"] == -1].copy()
            real_rows += len(real_chunk)
            noise_rows += len(noise_chunk)

            if not real_chunk.empty:
                topic_real_frames.append(
                    real_chunk.groupby("topic_group").size().rename("real_rows").reset_index()
                )
                topic_cluster_frames.append(
                    real_chunk[["topic_group", "cluster"]].drop_duplicates().reset_index(drop=True)
                )

                detail_frame = _ensure_columns(real_chunk, DETAIL_OUTPUT_COLUMNS, fill_value="")
                detail_table = pa.Table.from_pandas(detail_frame, preserve_index=False)
                if detail_writer is None:
                    detail_writer = pq.ParquetWriter(
                        str(detail_tmp_path),
                        detail_table.schema,
                        compression="zstd",
                    )
                detail_writer.write_table(detail_table)

                real_chunk["_timestamped"] = real_chunk["timestamp"].notna().astype(np.int64)
                real_chunk["_membership_filled"] = real_chunk["cluster_membership_strength"].fillna(0.0)
                real_chunk["_outlier_filled"] = real_chunk["cluster_outlier_score"].fillna(0.0)
                cluster_agg_frames.append(
                    real_chunk.groupby(["topic_group", "cluster"], as_index=False).agg(
                        article_count=("cluster", "size"),
                        timestamped_count=("_timestamped", "sum"),
                        sum_membership=("_membership_filled", "sum"),
                        sum_outlier=("_outlier_filled", "sum"),
                        min_timestamp=("timestamp", "min"),
                        max_timestamp=("timestamp", "max"),
                    )
                )

                dated_real = real_chunk[real_chunk["timestamp_date"].notna()]
                if not dated_real.empty:
                    daily_frames.append(
                        dated_real.groupby(["topic_group", "cluster", "timestamp_date"], as_index=False)
                        .size()
                        .rename(columns={"size": "article_count", "timestamp_date": "date"})
                    )

                topic_reason_chunk = real_chunk[
                    real_chunk["topic_group_reason"].fillna("").astype(str).str.strip() != ""
                ][["topic_group", "cluster", "topic_group_reason"]]
                if not topic_reason_chunk.empty:
                    topic_reason_count_frames.append(
                        topic_reason_chunk.groupby(
                            ["topic_group", "cluster", "topic_group_reason"],
                            as_index=False,
                        ).size().rename(columns={"size": "count"})
                    )

                assignment_reason_chunk = real_chunk[
                    real_chunk["cluster_assignment_reason"].fillna("").astype(str).str.strip() != ""
                ][["topic_group", "cluster", "cluster_assignment_reason"]]
                if not assignment_reason_chunk.empty:
                    assignment_reason_count_frames.append(
                        assignment_reason_chunk.groupby(
                            ["topic_group", "cluster", "cluster_assignment_reason"],
                            as_index=False,
                        ).size().rename(columns={"size": "count"})
                    )

                coords_chunk = real_chunk[real_chunk["umap_x"].notna() & real_chunk["umap_y"].notna()].copy()
                if not coords_chunk.empty:
                    best_candidate_frames.append(
                        coords_chunk.sort_values(
                            ["topic_group", "cluster", "cluster_membership_strength", "cluster_outlier_score"],
                            ascending=[True, True, False, True],
                        ).drop_duplicates(["topic_group", "cluster"], keep="first")
                    )
                    edge_candidate_frames.append(
                        coords_chunk.sort_values(
                            ["topic_group", "cluster", "cluster_membership_strength", "cluster_outlier_score"],
                            ascending=[True, True, True, False],
                        ).drop_duplicates(["topic_group", "cluster"], keep="first")
                    )

            if not noise_chunk.empty:
                topic_noise_frames.append(
                    noise_chunk.groupby("topic_group").size().rename("noise_rows").reset_index()
                )
                noise_coords = noise_chunk[noise_chunk["umap_x"].notna() & noise_chunk["umap_y"].notna()].copy()
                if not noise_coords.empty:
                    noise_coords["sample_rand"] = rng.random(len(noise_coords))
                    noise_sample_cols = SCATTER_OUTPUT_COLUMNS + ["sample_rand"]
                    noise_coords["sample_type"] = "noise_topic_reservoir"
                    topic_groups = [str(topic_group) for topic_group in noise_coords["topic_group"].dropna().unique().tolist()]
                    for topic_group in topic_groups:
                        topic_noise = _ensure_columns(
                            noise_coords[noise_coords["topic_group"] == topic_group],
                            noise_sample_cols,
                        )
                        existing = noise_samples.get(topic_group)
                        combined = (
                            pd.concat([existing, topic_noise], ignore_index=True)
                            if existing is not None
                            else topic_noise
                        )
                        noise_samples[topic_group] = combined.nsmallest(
                            noise_per_topic_cap,
                            "sample_rand",
                        ).reset_index(drop=True)

            if chunk_index % 10 == 0:
                print(
                    f"Processed {total_rows:,} rows | "
                    f"real={real_rows:,} noise={noise_rows:,} timestamped={timestamped_rows:,}"
                )

    finally:
        if detail_writer is not None:
            detail_writer.close()
            if detail_tmp_path.exists():
                detail_tmp_path.replace(article_detail_path)

    if detail_writer is None:
        _write_parquet_atomic(pd.DataFrame(columns=DETAIL_OUTPUT_COLUMNS), article_detail_path)

    if cluster_agg_frames:
        cluster_overview_df = (
            pd.concat(cluster_agg_frames, ignore_index=True)
            .groupby(["topic_group", "cluster"], as_index=False)
            .agg(
                article_count=("article_count", "sum"),
                timestamped_articles=("timestamped_count", "sum"),
                sum_membership=("sum_membership", "sum"),
                sum_outlier=("sum_outlier", "sum"),
                first_seen=("min_timestamp", "min"),
                last_seen=("max_timestamp", "max"),
            )
        )
    else:
        cluster_overview_df = pd.DataFrame(
            columns=[
                "topic_group",
                "cluster",
                "article_count",
                "timestamped_articles",
                "sum_membership",
                "sum_outlier",
                "first_seen",
                "last_seen",
            ]
        )

    if daily_frames:
        cluster_daily_df = pd.concat(daily_frames, ignore_index=True)
        cluster_daily_df = cluster_daily_df.groupby(
            ["topic_group", "cluster", "date"],
            as_index=False,
        )["article_count"].sum()
        cluster_daily_df = (
            cluster_daily_df
            .set_index(["topic_group", "cluster", "date"])
            .sort_index()
            .reset_index()
        )
    else:
        cluster_daily_df = pd.DataFrame(columns=["topic_group", "cluster", "date", "article_count"])

    if not cluster_daily_df.empty:
        daily_summary_df = (
            cluster_daily_df.groupby(["topic_group", "cluster"], as_index=False)
            .agg(active_days=("date", "nunique"), peak_day_count=("article_count", "max"))
        )
    else:
        daily_summary_df = pd.DataFrame(columns=["topic_group", "cluster", "active_days", "peak_day_count"])

    topic_reason_mode_df = _build_mode_frame(
        topic_reason_count_frames,
        value_column="topic_group_reason",
        output_column="topic_group_reason_mode",
    )
    assignment_reason_mode_df = _build_mode_frame(
        assignment_reason_count_frames,
        value_column="cluster_assignment_reason",
        output_column="cluster_assignment_reason_mode",
    )
    best_rows_df = _combine_candidate_rows(best_candidate_frames, best=True)
    edge_rows_df = _combine_candidate_rows(edge_candidate_frames, best=False)

    if not cluster_overview_df.empty:
        cluster_overview_df = cluster_overview_df.merge(
            daily_summary_df,
            on=["topic_group", "cluster"],
            how="left",
        )
        cluster_overview_df = cluster_overview_df.merge(
            topic_reason_mode_df,
            on=["topic_group", "cluster"],
            how="left",
        )
        cluster_overview_df = cluster_overview_df.merge(
            assignment_reason_mode_df,
            on=["topic_group", "cluster"],
            how="left",
        )

        if not best_rows_df.empty:
            cluster_overview_df = cluster_overview_df.merge(
                best_rows_df[
                    [
                        "topic_group",
                        "cluster",
                        "title",
                        "url",
                        "timestamp",
                        "source_domain",
                        "cluster_membership_strength",
                        "cluster_outlier_score",
                    ]
                ].rename(
                    columns={
                        "title": "representative_title",
                        "url": "representative_url",
                        "timestamp": "representative_timestamp",
                        "source_domain": "representative_domain",
                        "cluster_membership_strength": "representative_membership_strength",
                        "cluster_outlier_score": "representative_outlier_score",
                    }
                ),
                on=["topic_group", "cluster"],
                how="left",
            )

        cluster_overview_df["total_articles"] = cluster_overview_df["article_count"]
        cluster_overview_df["cluster_size"] = cluster_overview_df["article_count"]
        cluster_overview_df["undated_articles"] = (
            cluster_overview_df["article_count"] - cluster_overview_df["timestamped_articles"]
        )
        cluster_overview_df["timestamp_coverage_ratio"] = (
            cluster_overview_df["timestamped_articles"] / cluster_overview_df["article_count"]
        ).fillna(0.0)
        cluster_overview_df["active_days"] = cluster_overview_df["active_days"].fillna(0).astype(int)
        cluster_overview_df["peak_day_count"] = cluster_overview_df["peak_day_count"].fillna(0).astype(int)
        cluster_overview_df["peak_day_share"] = (
            cluster_overview_df["peak_day_count"] / cluster_overview_df["timestamped_articles"]
        ).fillna(0.0)
        cluster_overview_df["mean_membership_strength"] = (
            cluster_overview_df["sum_membership"] / cluster_overview_df["article_count"]
        ).replace([np.inf, -np.inf], np.nan)
        cluster_overview_df["mean_outlier_score"] = (
            cluster_overview_df["sum_outlier"] / cluster_overview_df["article_count"]
        ).replace([np.inf, -np.inf], np.nan)
        cluster_overview_df["span_days"] = (
            cluster_overview_df["last_seen"] - cluster_overview_df["first_seen"]
        ).dt.days.fillna(0).astype(int)
        cluster_overview_df = cluster_overview_df.sort_values(
            ["article_count", "timestamped_articles"],
            ascending=[False, False],
        ).reset_index(drop=True)
    else:
        cluster_overview_df = pd.DataFrame(
            columns=[
                "topic_group",
                "cluster",
                "article_count",
                "total_articles",
                "cluster_size",
                "timestamped_articles",
                "undated_articles",
                "timestamp_coverage_ratio",
                "first_seen",
                "last_seen",
                "span_days",
                "active_days",
                "peak_day_count",
                "peak_day_share",
                "mean_membership_strength",
                "mean_outlier_score",
                "topic_group_reason_mode",
                "cluster_assignment_reason_mode",
                "representative_title",
                "representative_url",
                "representative_timestamp",
                "representative_domain",
                "representative_membership_strength",
                "representative_outlier_score",
            ]
        )

    _write_parquet_atomic(cluster_overview_df, cluster_overview_path)
    _write_parquet_atomic(cluster_daily_df, cluster_daily_path)

    topic_total_df = _aggregate_count_frames(topic_total_frames, "total_rows")
    topic_real_df = _aggregate_count_frames(topic_real_frames, "real_rows")
    topic_noise_df = _aggregate_count_frames(topic_noise_frames, "noise_rows")
    topic_timestamp_df = _aggregate_count_frames(topic_timestamp_frames, "timestamped_rows")
    if topic_cluster_frames:
        topic_cluster_count_df = (
            pd.concat(topic_cluster_frames, ignore_index=True)
            .drop_duplicates()
            .groupby("topic_group", as_index=False)
            .size()
            .rename(columns={"size": "real_cluster_count"})
        )
    else:
        topic_cluster_count_df = pd.DataFrame(columns=["topic_group", "real_cluster_count"])

    topic_summary_df = topic_total_df.merge(topic_real_df, on="topic_group", how="left")
    topic_summary_df = topic_summary_df.merge(topic_noise_df, on="topic_group", how="left")
    topic_summary_df = topic_summary_df.merge(topic_timestamp_df, on="topic_group", how="left")
    topic_summary_df = topic_summary_df.merge(topic_cluster_count_df, on="topic_group", how="left")
    for column in ("real_rows", "noise_rows", "timestamped_rows", "real_cluster_count"):
        topic_summary_df[column] = topic_summary_df[column].fillna(0).astype(int)

    if not cluster_overview_df.empty:
        topic_cluster_stats_df = (
            cluster_overview_df.groupby("topic_group", as_index=False)
            .agg(
                mean_real_cluster_size=("article_count", "mean"),
                median_real_cluster_size=("article_count", "median"),
                min_timestamp=("first_seen", "min"),
                max_timestamp=("last_seen", "max"),
            )
        )
        topic_top_clusters_df = (
            cluster_overview_df.sort_values(
                ["topic_group", "article_count", "cluster"],
                ascending=[True, False, True],
            )
            .drop_duplicates("topic_group", keep="first")
            .rename(columns={"cluster": "top_cluster", "article_count": "top_cluster_size"})[
                ["topic_group", "top_cluster", "top_cluster_size"]
            ]
        )
        topic_summary_df = topic_summary_df.merge(topic_cluster_stats_df, on="topic_group", how="left")
        topic_summary_df = topic_summary_df.merge(topic_top_clusters_df, on="topic_group", how="left")
    else:
        topic_summary_df["mean_real_cluster_size"] = 0.0
        topic_summary_df["median_real_cluster_size"] = 0.0
        topic_summary_df["min_timestamp"] = pd.NaT
        topic_summary_df["max_timestamp"] = pd.NaT
        topic_summary_df["top_cluster"] = np.nan
        topic_summary_df["top_cluster_size"] = 0

    topic_summary_df["article_count"] = topic_summary_df["real_rows"]
    topic_summary_df["unique_clusters"] = topic_summary_df["real_cluster_count"]
    topic_summary_df["timestamp_coverage_ratio"] = (
        topic_summary_df["timestamped_rows"] / topic_summary_df["total_rows"]
    ).fillna(0.0)
    topic_summary_df["noise_rate"] = (
        topic_summary_df["noise_rows"] / topic_summary_df["total_rows"]
    ).fillna(0.0)
    topic_summary_df["mean_real_cluster_size"] = topic_summary_df["mean_real_cluster_size"].fillna(0.0).round(3)
    topic_summary_df["median_real_cluster_size"] = topic_summary_df["median_real_cluster_size"].fillna(0.0).round(3)
    topic_summary_df["top_cluster_size"] = topic_summary_df["top_cluster_size"].fillna(0).astype(int)
    topic_summary_df = topic_summary_df.sort_values(
        ["total_rows", "real_cluster_count"],
        ascending=[False, False],
    ).reset_index(drop=True)
    _write_parquet_atomic(topic_summary_df, topic_summary_path)

    scatter_frames: list[pd.DataFrame] = []
    if not best_rows_df.empty:
        anchors = _ensure_columns(best_rows_df, SCATTER_OUTPUT_COLUMNS[1:], fill_value=np.nan)
        anchors.insert(0, "sample_type", "cluster_anchor")
        scatter_frames.append(anchors)

    if not edge_rows_df.empty:
        edges = _ensure_columns(edge_rows_df, SCATTER_OUTPUT_COLUMNS[1:], fill_value=np.nan)
        edges.insert(0, "sample_type", "cluster_edge")
        scatter_frames.append(edges)

    for noise_df in noise_samples.values():
        scatter_frames.append(
            _ensure_columns(noise_df.drop(columns=["sample_rand"], errors="ignore"), SCATTER_OUTPUT_COLUMNS)
        )

    if scatter_frames:
        scatter_df = pd.concat(scatter_frames, ignore_index=True)
        scatter_df = scatter_df.drop_duplicates(
            subset=["sample_type", "topic_group", "cluster", "title", "url", "timestamp", "umap_x", "umap_y"]
        )
        if len(scatter_df) > scatter_cap:
            anchors = scatter_df[scatter_df["sample_type"] == "cluster_anchor"].copy()
            remainder = scatter_df[scatter_df["sample_type"] != "cluster_anchor"].copy()
            if len(anchors) > scatter_cap:
                scatter_df = anchors.sample(n=scatter_cap, random_state=42)
            else:
                remaining_budget = max(scatter_cap - len(anchors), 0)
                if len(remainder) > remaining_budget:
                    remainder = remainder.sample(n=remaining_budget, random_state=42)
                scatter_df = pd.concat([anchors, remainder], ignore_index=True)
        scatter_df = scatter_df.reset_index(drop=True)
    else:
        scatter_df = pd.DataFrame(columns=SCATTER_OUTPUT_COLUMNS)
    _write_parquet_atomic(scatter_df, scatter_sample_path)

    min_timestamp = (
        cluster_overview_df["first_seen"].dropna().min()
        if not cluster_overview_df.empty and "first_seen" in cluster_overview_df.columns
        else pd.NaT
    )
    max_timestamp = (
        cluster_overview_df["last_seen"].dropna().max()
        if not cluster_overview_df.empty and "last_seen" in cluster_overview_df.columns
        else pd.NaT
    )
    global_meta_df = pd.DataFrame(
        [
            {
                "source_parquet": str(input_path),
                "generated_dir": str(output_dir),
                "generated_at": pd.Timestamp.utcnow(),
                "total_rows": total_rows,
                "real_rows": real_rows,
                "noise_rows": noise_rows,
                "noise_rate": round(noise_rows / total_rows, 6) if total_rows else 0.0,
                "timestamped_rows": timestamped_rows,
                "timestamp_coverage_ratio": round(timestamped_rows / total_rows, 6) if total_rows else 0.0,
                "real_cluster_count": len(cluster_overview_df),
                "topic_count": topic_summary_df["topic_group"].nunique() if not topic_summary_df.empty else 0,
                "cluster_overview_rows": len(cluster_overview_df),
                "topic_summary_rows": len(topic_summary_df),
                "daily_count_rows": len(cluster_daily_df),
                "article_detail_rows": real_rows,
                "scatter_sample_rows": len(scatter_df),
                "scatter_cap": scatter_cap,
                "noise_per_topic_cap": noise_per_topic_cap,
                "min_timestamp": min_timestamp,
                "max_timestamp": max_timestamp,
            }
        ]
    )
    _write_parquet_atomic(global_meta_df, global_meta_path)

    _copy_parquet_atomic(TEMPORAL_STATS, output_dir / DASHBOARD_TEMPORAL_PARQUET.name)
    _copy_parquet_atomic(HDBSCAN_CONFIG_RESULTS, output_dir / DASHBOARD_CONFIG_PARQUET.name)
    _copy_parquet_atomic(EVALUATION_REPORT, output_dir / DASHBOARD_EVAL_PARQUET.name)
    _copy_parquet_atomic(TFIDF_ABLATION_REPORT, output_dir / DASHBOARD_ABLATION_PARQUET.name)

    return {
        "output_dir": output_dir,
        "total_rows": total_rows,
        "real_rows": real_rows,
        "noise_rows": noise_rows,
        "timestamped_rows": timestamped_rows,
        "real_cluster_count": len(cluster_overview_df),
        "topic_count": topic_summary_df["topic_group"].nunique() if not topic_summary_df.empty else 0,
        "cluster_overview_rows": len(cluster_overview_df),
        "topic_summary_rows": len(topic_summary_df),
        "daily_count_rows": len(cluster_daily_df),
        "scatter_sample_rows": len(scatter_df),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build parquet-backed dashboard assets from clustered_data.parquet.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=CLUSTERED_PARQUET,
        help="Path to clustered_data.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DASHBOARD_DIR,
        help="Output directory for parquet dashboard assets",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Override CPU thread count for parquet batching and pandas work",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override parquet batch size",
    )
    parser.add_argument(
        "--scatter-cap",
        type=int,
        default=25_000,
        help="Maximum number of rows to keep in the UMAP scatter sample",
    )
    parser.add_argument(
        "--noise-per-topic-cap",
        type=int,
        default=50,
        help="Maximum number of noise rows to keep per topic in the scatter sample",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = detect_runtime_profile(
        device="cpu",
        cpu_threads=args.cpu_threads,
        chunk_size=args.chunk_size,
    )
    apply_runtime_profile(profile)
    print(format_runtime_profile(profile))

    if not args.input.exists():
        raise FileNotFoundError(
            f"Clustered parquet not found at {args.input}. Run EmbeddingsClustering.py first."
        )

    result = build_dashboard_assets(
        input_path=args.input,
        output_dir=args.output_dir,
        chunk_size=profile.chunk_size,
        scatter_cap=args.scatter_cap,
        noise_per_topic_cap=args.noise_per_topic_cap,
    )

    print("\nDashboard assets built successfully.")
    print(f"Output directory: {result['output_dir']}")
    print(f"Rows processed: {result['total_rows']:,}")
    print(f"Real clusters: {result['real_cluster_count']:,}")
    print(f"Scatter sample rows: {result['scatter_sample_rows']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
