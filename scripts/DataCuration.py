"""
Data curation and cleaning for the coordinated-campaign pipeline.

Outputs:
    data/rolargesum_raw.parquet
    data/rolargesum_train_clean.parquet
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import sys
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    COLUMNS_TO_CLEAN,
    COLUMNS_TO_PRESERVE,
    TIMESTAMP_DOMAIN_MIN_ROWS_FOR_WEAKNESS,
    TIMESTAMP_DOMAIN_REPORT_TOP_N,
    TIMESTAMP_MAX_FUTURE_DAYS_TEXT,
    TIMESTAMP_MAX_FUTURE_DAYS_URL,
    TIMESTAMP_QUALITY_COLUMN,
    TIMESTAMP_QUALITY_HIGH,
    TIMESTAMP_QUALITY_LOW,
    TIMESTAMP_QUALITY_MEDIUM,
    TIMESTAMP_QUALITY_MISSING,
    TIMESTAMP_QUALITY_REJECTED_FUTURE,
    TIMESTAMP_SOURCE_COLUMN,
    TIMESTAMP_SOURCE_MISSING,
    TIMESTAMP_SOURCE_TEXT,
    TIMESTAMP_SOURCE_URL,
    TIMESTAMP_TEXT_LOW_CONFIDENCE_BEFORE_YEAR,
)
from src.date_extraction import extract_date_from_text, extract_date_from_url_with_source
from src.paths import CLEAN_PARQUET, RAW_PARQUET
from src.runtime_profile import (
    apply_runtime_profile,
    detect_runtime_profile,
    format_runtime_profile,
)
from src.text_processing import clean_text, remove_stopwords_and_clean

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(encoding="utf-8", errors="replace")


def _extract_domain_from_url(url: str) -> str:
    """Return a normalized domain for coverage reporting."""
    if pd.isna(url):
        return TIMESTAMP_SOURCE_MISSING

    raw_url = str(url).strip()
    if not raw_url:
        return TIMESTAMP_SOURCE_MISSING

    parsed = urlparse(raw_url if "://" in raw_url else f"http://{raw_url}")
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain.removeprefix("www.")
    return domain or TIMESTAMP_SOURCE_MISSING


def _clean_stopword_chunk(values: list[str]) -> list[str]:
    """Worker-safe chunk transform for stopword removal."""
    return [remove_stopwords_and_clean(value) for value in values]


def build_document_nostop(series: pd.Series, cpu_threads: int, chunk_size: int) -> pd.Series:
    """Build the stopword-free document column with conservative parallelism."""
    values = series.fillna("").astype(str).tolist()
    if not values:
        return pd.Series(dtype="object", index=series.index)

    if cpu_threads <= 1 or len(values) < max(chunk_size, 50_000):
        cleaned = [remove_stopwords_and_clean(value) for value in values]
        return pd.Series(cleaned, index=series.index, dtype="object")

    worker_chunk_size = max(5_000, min(chunk_size, len(values) // max(cpu_threads * 4, 1)))
    worker_chunks = [
        values[start:start + worker_chunk_size]
        for start in range(0, len(values), worker_chunk_size)
    ]

    cleaned_values: list[str] = []
    with ProcessPoolExecutor(max_workers=cpu_threads) as executor:
        for cleaned_chunk in executor.map(_clean_stopword_chunk, worker_chunks):
            cleaned_values.extend(cleaned_chunk)

    return pd.Series(cleaned_values, index=series.index, dtype="object")


def load_or_download_dataset() -> pd.DataFrame:
    """Load the raw parquet cache or download the dataset from HuggingFace."""
    if RAW_PARQUET.exists():
        print(f"Loading dataset from local cache: {RAW_PARQUET}")
        train_df = pd.read_parquet(RAW_PARQUET)
        print(f"Loaded {len(train_df):,} rows from cache.")
        return train_df

    print("First run - downloading from HuggingFace...")
    from dotenv import load_dotenv
    from datasets import load_dataset
    from huggingface_hub import login

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN not set. Add it to your .env before the first run.")

    login(token=hf_token)
    dataset = load_dataset("avramandrei/rolargesum")
    print(dataset)
    train_df = cast(pd.DataFrame, dataset["train"].to_pandas())
    train_df.to_parquet(RAW_PARQUET, index=False, compression="zstd")
    print(f"Dataset cached at: {RAW_PARQUET} ({len(train_df):,} rows)")
    return train_df


def clean_base_columns(train_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the core textual columns used throughout the pipeline."""
    train_df = train_df.copy()
    for column in COLUMNS_TO_CLEAN:
        if column not in train_df.columns:
            train_df[column] = ""
        train_df[column] = train_df[column].map(clean_text)
    return train_df


def attach_timestamps(train_df: pd.DataFrame) -> pd.DataFrame:
    """Extract timestamps from URLs first, then from article text."""
    train_df = train_df.copy()
    print("\nExtracting timestamps...")

    url_series = cast(pd.Series, train_df["url"])
    url_extraction = url_series.apply(extract_date_from_url_with_source)
    url_extraction_df = pd.DataFrame(
        url_extraction.tolist(),
        index=train_df.index,
        columns=["timestamp", TIMESTAMP_SOURCE_COLUMN],
    )
    train_df["timestamp"] = pd.to_datetime(url_extraction_df["timestamp"], errors="coerce")
    train_df[TIMESTAMP_SOURCE_COLUMN] = url_extraction_df[TIMESTAMP_SOURCE_COLUMN]

    url_found = int((train_df[TIMESTAMP_SOURCE_COLUMN] == TIMESTAMP_SOURCE_URL).sum())
    print(f"  From URL:  {url_found:,} ({100 * url_found / len(train_df):.1f}%)")

    missing_mask = train_df["timestamp"].isna()
    if int(missing_mask.sum()) > 0:
        print(f"  Scanning text body for {int(missing_mask.sum()):,} remaining rows...")
        text_dates = cast(pd.Series, train_df.loc[missing_mask, "text"]).apply(extract_date_from_text)
        train_df.loc[missing_mask, "timestamp"] = pd.to_datetime(text_dates, errors="coerce")
        text_found_mask = text_dates.notna()
        train_df.loc[text_dates.index[text_found_mask], TIMESTAMP_SOURCE_COLUMN] = TIMESTAMP_SOURCE_TEXT
        text_found = int(text_found_mask.sum())
        print(f"  From text: {text_found:,} ({100 * text_found / len(train_df):.1f}%)")

    train_df[TIMESTAMP_SOURCE_COLUMN] = (
        train_df[TIMESTAMP_SOURCE_COLUMN]
        .fillna(TIMESTAMP_SOURCE_MISSING)
        .astype(str)
        .str.lower()
    )

    total_found = int(train_df["timestamp"].notna().sum())
    print(f"  TOTAL:     {total_found:,} / {len(train_df):,} ({100 * total_found / len(train_df):.1f}%)")
    print(f"  Missing:   {int(train_df['timestamp'].isna().sum()):,}")
    return train_df


def validate_timestamp_quality(train_df: pd.DataFrame) -> pd.DataFrame:
    """Apply vectorized quality labels and reject impossible future timestamps."""
    train_df = train_df.copy()
    timestamps = pd.to_datetime(train_df["timestamp"], errors="coerce")
    sources = (
        train_df[TIMESTAMP_SOURCE_COLUMN]
        .fillna(TIMESTAMP_SOURCE_MISSING)
        .astype(str)
        .str.lower()
    )
    quality = pd.Series(TIMESTAMP_QUALITY_MISSING, index=train_df.index, dtype="object")

    today = pd.Timestamp.today().normalize()
    url_future_mask = (
        timestamps.notna()
        & (sources == TIMESTAMP_SOURCE_URL)
        & (timestamps.dt.normalize() > today + pd.Timedelta(days=TIMESTAMP_MAX_FUTURE_DAYS_URL))
    )
    text_future_mask = (
        timestamps.notna()
        & (sources == TIMESTAMP_SOURCE_TEXT)
        & (timestamps.dt.normalize() > today + pd.Timedelta(days=TIMESTAMP_MAX_FUTURE_DAYS_TEXT))
    )
    rejected_mask = url_future_mask | text_future_mask
    timestamps = timestamps.mask(rejected_mask)
    quality.loc[rejected_mask] = TIMESTAMP_QUALITY_REJECTED_FUTURE

    url_mask = timestamps.notna() & (sources == TIMESTAMP_SOURCE_URL)
    text_low_mask = (
        timestamps.notna()
        & (sources == TIMESTAMP_SOURCE_TEXT)
        & (timestamps.dt.year < TIMESTAMP_TEXT_LOW_CONFIDENCE_BEFORE_YEAR)
    )
    text_medium_mask = timestamps.notna() & (sources == TIMESTAMP_SOURCE_TEXT) & ~text_low_mask

    quality.loc[url_mask] = TIMESTAMP_QUALITY_HIGH
    quality.loc[text_low_mask] = TIMESTAMP_QUALITY_LOW
    quality.loc[text_medium_mask] = TIMESTAMP_QUALITY_MEDIUM

    train_df["timestamp"] = timestamps
    train_df[TIMESTAMP_QUALITY_COLUMN] = quality

    print("  Source breakdown:")
    source_counts = (
        train_df[TIMESTAMP_SOURCE_COLUMN]
        .value_counts()
        .reindex(
            [TIMESTAMP_SOURCE_URL, TIMESTAMP_SOURCE_TEXT, TIMESTAMP_SOURCE_MISSING],
            fill_value=0,
        )
    )
    for source, count in source_counts.items():
        print(f"    {source:<8} {int(count):>6} ({100 * count / len(train_df):.1f}%)")

    print("  Quality breakdown:")
    quality_counts = train_df[TIMESTAMP_QUALITY_COLUMN].value_counts().to_dict()
    for quality_name in (
        TIMESTAMP_QUALITY_HIGH,
        TIMESTAMP_QUALITY_MEDIUM,
        TIMESTAMP_QUALITY_LOW,
        TIMESTAMP_QUALITY_MISSING,
        TIMESTAMP_QUALITY_REJECTED_FUTURE,
    ):
        count = int(quality_counts.get(quality_name, 0))
        print(f"    {quality_name:<16} {count:>6} ({100 * count / len(train_df):.1f}%)")

    return train_df


def print_domain_report(train_df: pd.DataFrame) -> None:
    """Summarize timestamp coverage by source domain."""
    domain_report_df = train_df.copy()
    domain_report_df["domain"] = domain_report_df["url"].map(_extract_domain_from_url)

    domain_summary = (
        domain_report_df.groupby("domain", dropna=False)
        .agg(
            rows=("domain", "size"),
            timestamped=("timestamp", lambda s: int(s.notna().sum())),
            url_source=(TIMESTAMP_SOURCE_COLUMN, lambda s: int((s == TIMESTAMP_SOURCE_URL).sum())),
            text_source=(TIMESTAMP_SOURCE_COLUMN, lambda s: int((s == TIMESTAMP_SOURCE_TEXT).sum())),
            missing_source=(TIMESTAMP_SOURCE_COLUMN, lambda s: int((s == TIMESTAMP_SOURCE_MISSING).sum())),
        )
        .sort_values(["rows", "timestamped"], ascending=[False, False])
    )
    domain_summary["coverage_pct"] = 100 * domain_summary["timestamped"] / domain_summary["rows"]

    print("\nTop domains by volume:")
    print(
        domain_summary.head(TIMESTAMP_DOMAIN_REPORT_TOP_N)[
            ["rows", "timestamped", "coverage_pct", "url_source", "text_source", "missing_source"]
        ].to_string(float_format=lambda value: f"{value:.1f}")
    )

    weak_domains = domain_summary[
        domain_summary["rows"] >= TIMESTAMP_DOMAIN_MIN_ROWS_FOR_WEAKNESS
    ].sort_values(["coverage_pct", "rows"], ascending=[True, False])
    if weak_domains.empty:
        return

    print(f"\nWeakest domains with at least {TIMESTAMP_DOMAIN_MIN_ROWS_FOR_WEAKNESS} rows:")
    print(
        weak_domains.head(TIMESTAMP_DOMAIN_REPORT_TOP_N)[
            ["rows", "timestamped", "coverage_pct", "url_source", "text_source", "missing_source"]
        ].to_string(float_format=lambda value: f"{value:.1f}")
    )


def build_documents(train_df: pd.DataFrame, cpu_threads: int, chunk_size: int) -> pd.DataFrame:
    """Create the text columns used downstream by embeddings and TF-IDF."""
    train_df = train_df.copy()
    train_df["document"] = (train_df["title"] + ". " + train_df["text"]).map(clean_text)
    train_df["short_document"] = (
        train_df["title"] + ". " + train_df["text"].fillna("").astype(str).str.slice(0, 500)
    ).map(clean_text)

    print("\nBuilding stopword-free documents...")
    train_df["document_nostop"] = build_document_nostop(
        train_df["document"],
        cpu_threads=cpu_threads,
        chunk_size=chunk_size,
    )
    return train_df


def filter_rows(train_df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty rows and duplicate documents."""
    print("\nShape before filtering:", train_df.shape)
    filtered_df = train_df[train_df["title"].str.strip() != ""].copy()
    filtered_df = filtered_df[filtered_df["document"].str.strip() != ""].copy()
    filtered_df = filtered_df.drop_duplicates(subset=["document"]).reset_index(drop=True)
    print("Shape after filtering:", filtered_df.shape)
    return filtered_df


def print_stats(train_df: pd.DataFrame) -> None:
    """Show a compact summary of the cleaned dataset."""
    print("\nMissing values per column:")
    print(train_df.isna().sum())

    print("\nTop topic values:")
    print(train_df["topics"].value_counts(dropna=False).head(20))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/cache RoLargeSum, clean it, and write parquet outputs.",
    )
    parser.add_argument("--nrows", type=int, default=None, help="Optional row cap for faster experiments")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Override CPU worker count for preprocessing",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override preprocessing chunk size",
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

    train_df = load_or_download_dataset()
    if args.nrows is not None:
        train_df = train_df.head(args.nrows).copy()
        print(f"Using first {len(train_df):,} rows due to --nrows.")

    print("Shape initial:", train_df.shape)
    print("Columns:", train_df.columns.tolist())
    print(train_df.head(3))

    train_df = clean_base_columns(train_df)
    train_df = attach_timestamps(train_df)
    train_df = validate_timestamp_quality(train_df)
    print_domain_report(train_df)
    train_df = build_documents(
        train_df,
        cpu_threads=profile.cpu_threads,
        chunk_size=profile.chunk_size,
    )
    train_df = filter_rows(train_df)
    print_stats(train_df)

    for column in COLUMNS_TO_PRESERVE:
        if column not in train_df.columns:
            train_df[column] = ""
    train_df = train_df.loc[:, COLUMNS_TO_PRESERVE].copy()
    train_df.to_parquet(CLEAN_PARQUET, index=False, compression="zstd")

    print(f"\nShape final: {train_df.shape}")
    print(f"Saved to: {CLEAN_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
