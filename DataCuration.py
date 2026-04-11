"""
Data Curation & Cleaning
===================================
Downloads the RoLargeSum dataset from HuggingFace (first run only),
caches it locally as a parquet file, then cleans the text and saves
a processed CSV to data/.

Outputs:
    data/rolargesum_raw.parquet         (raw cache - skips HF API on re-run)
    data/rolargesum_train_clean.csv     (cleaned dataset)
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
from typing import cast
from urllib.parse import urlparse
import pandas as pd
from src.config import (
    TIMESTAMP_DOMAIN_MIN_ROWS_FOR_WEAKNESS,
    TIMESTAMP_DOMAIN_REPORT_TOP_N,
    TIMESTAMP_SOURCE_COLUMN,
    TIMESTAMP_SOURCE_MISSING,
    TIMESTAMP_SOURCE_TEXT,
    TIMESTAMP_SOURCE_URL,
)
from src.paths import BASE_DIR, DATA_DIR, RAW_PARQUET, CLEAN_CSV
from src.text_processing import (
    clean_text, remove_stopwords_and_clean
)
from src.date_extraction import extract_date_from_text, extract_date_from_url_with_source

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _extract_domain_from_url(url: str) -> str:
    """Return a normalized domain for coverage reporting."""
    if pd.isna(url):
        return TIMESTAMP_SOURCE_MISSING

    url = str(url).strip()
    if not url:
        return TIMESTAMP_SOURCE_MISSING

    parsed = urlparse(url if "://" in url else f"http://{url}")
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain.removeprefix("www.")
    return domain or TIMESTAMP_SOURCE_MISSING


# =========================================================================
# LOAD DATASET (with local cache)
# =========================================================================
if os.path.exists(RAW_PARQUET):
    print(f"Loading dataset from local cache: {RAW_PARQUET}")
    train_df: pd.DataFrame = pd.read_parquet(RAW_PARQUET)
    print(f"Loaded {len(train_df)} rows from cache.")
else:
    print("First run - downloading from HuggingFace...")

    from dotenv import load_dotenv
    from datasets import load_dataset
    from huggingface_hub import login

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN not set!")

    login(token=hf_token)
    dataset = load_dataset("avramandrei/rolargesum")
    print(dataset)
    train_df: pd.DataFrame = cast(pd.DataFrame, dataset["train"].to_pandas())

    train_df.to_parquet(RAW_PARQUET, index=False)
    print(f"Dataset cached at: {RAW_PARQUET} ({len(train_df)} rows)")

print("Shape initial:", train_df.shape)
print("Columns:", train_df.columns.tolist())
print(train_df.head(3))


# =========================================================================
# CLEAN ALL COLUMNS
# =========================================================================
train_df["title"]    = train_df["title"].apply(clean_text)
train_df["text"]     = train_df["text"].apply(clean_text)
train_df["summary"]  = train_df["summary"].apply(clean_text)
train_df["keywords"] = train_df["keywords"].apply(clean_text)
train_df["topics"]   = train_df["topics"].apply(clean_text)
train_df["dialect"]  = train_df["dialect"].apply(clean_text)
train_df["url"]      = train_df["url"].apply(clean_text)
train_df["author"]   = train_df["author"].apply(clean_text)


# =========================================================================
# TIMESTAMP EXTRACTION (layered: URL first, then text body)
# =========================================================================
print("\nExtracting timestamps...")

# Layer 1: from URL
url_series = cast(pd.Series, train_df["url"])
url_extraction = url_series.apply(extract_date_from_url_with_source)  # type: ignore
url_extraction_df = pd.DataFrame(
    url_extraction.tolist(),
    index=train_df.index,
    columns=["timestamp", TIMESTAMP_SOURCE_COLUMN],
)
train_df["timestamp"] = url_extraction_df["timestamp"]
train_df[TIMESTAMP_SOURCE_COLUMN] = url_extraction_df[TIMESTAMP_SOURCE_COLUMN]
url_found = (train_df[TIMESTAMP_SOURCE_COLUMN] == TIMESTAMP_SOURCE_URL).sum()
print(f"  From URL:  {url_found} ({100 * url_found / len(train_df):.1f}%)")

# Layer 2: from article text (for rows still missing timestamp)
missing_mask = train_df["timestamp"].isna()
if missing_mask.sum() > 0:
    print(f"  Scanning text body for {missing_mask.sum()} remaining rows...")
    text_series = cast(pd.Series, train_df.loc[missing_mask, "text"])
    text_dates = text_series.apply(extract_date_from_text)  # type: ignore
    train_df.loc[missing_mask, "timestamp"] = text_dates
    text_found_mask = text_dates.notna()
    train_df.loc[text_dates.index[text_found_mask], TIMESTAMP_SOURCE_COLUMN] = TIMESTAMP_SOURCE_TEXT
    text_found = text_found_mask.sum()
    print(f"  From text: {text_found} ({100 * text_found / len(train_df):.1f}%)")

total_found = train_df["timestamp"].notna().sum()
print(f"  TOTAL:     {total_found} / {len(train_df)} ({100 * total_found / len(train_df):.1f}%)")
print(f"  Missing:   {train_df['timestamp'].isna().sum()}")

print("  Source breakdown:")
source_counts = (
    train_df[TIMESTAMP_SOURCE_COLUMN]
    .fillna(TIMESTAMP_SOURCE_MISSING)
    .value_counts()
    .reindex(
        [TIMESTAMP_SOURCE_URL, TIMESTAMP_SOURCE_TEXT, TIMESTAMP_SOURCE_MISSING],
        fill_value=0,
    )
)
for source, count in source_counts.items():
    print(f"    {source:<8} {count:>6} ({100 * count / len(train_df):.1f}%)")

domain_report_df = train_df.copy()
domain_report_df["domain"] = domain_report_df["url"].apply(_extract_domain_from_url)
domain_summary = (
    domain_report_df.groupby("domain", dropna=False)
    .agg(
        rows=("domain", "size"),
        timestamped=("timestamp", lambda s: s.notna().sum()),
        url_source=("timestamp_source", lambda s: (s == TIMESTAMP_SOURCE_URL).sum()),
        text_source=("timestamp_source", lambda s: (s == TIMESTAMP_SOURCE_TEXT).sum()),
        missing_source=("timestamp_source", lambda s: (s == TIMESTAMP_SOURCE_MISSING).sum()),
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

weak_domains = domain_summary[domain_summary["rows"] >= TIMESTAMP_DOMAIN_MIN_ROWS_FOR_WEAKNESS].sort_values(
    ["coverage_pct", "rows"], ascending=[True, False]
)
if not weak_domains.empty:
    print(
        f"\nWeakest domains with at least {TIMESTAMP_DOMAIN_MIN_ROWS_FOR_WEAKNESS} rows:"
    )
    print(
        weak_domains.head(TIMESTAMP_DOMAIN_REPORT_TOP_N)[
            ["rows", "timestamped", "coverage_pct", "url_source", "text_source", "missing_source"]
        ].to_string(float_format=lambda value: f"{value:.1f}")
    )


# =========================================================================
# BUILD DOCUMENTS
# =========================================================================
train_df["document"] = train_df["title"] + ". " + train_df["text"]
train_df["document"] = train_df["document"].apply(clean_text)

# Short document = title + first 500 chars of text (for SBERT embeddings)
train_df["short_document"] = train_df["title"] + ". " + train_df["text"].str.slice(0, 500)
train_df["short_document"] = train_df["short_document"].apply(clean_text)


# =========================================================================
# FILTER
# =========================================================================
print("\nShape before filtering:", train_df.shape)

train_df = train_df[train_df["title"].str.strip() != ""].copy()
train_df = train_df[train_df["document"].str.strip() != ""].copy()
train_df = train_df.drop_duplicates(subset=["document"]).reset_index(drop=True)

# Stopword-free version (for TF-IDF baseline) — diacritics stripped, short tokens removed
train_df["document_nostop"] = train_df["document"].apply(remove_stopwords_and_clean)

print("Shape after filtering:", train_df.shape)


# =========================================================================
# STATS
# =========================================================================
print("\nMissing values per column:")
print(train_df.isna().sum())

print("\nTop topic values:")
print(train_df["topics"].value_counts(dropna=False).head(20))


# =========================================================================
# SAVE
# =========================================================================
train_df = train_df[[
    "title",
    "text",
    "summary",
    "keywords",
    "topics",
    "dialect",
    "url",
    "author",
    "document",
    "short_document",
    "document_nostop",
    "timestamp",
    TIMESTAMP_SOURCE_COLUMN,
]].copy()

train_df.to_csv(CLEAN_CSV, index=False)

print(f"\nShape final: {train_df.shape}")
print(f"Saved to: {CLEAN_CSV}")
