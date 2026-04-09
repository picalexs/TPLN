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

from typing import cast
import pandas as pd
from src.paths import BASE_DIR, DATA_DIR, RAW_PARQUET, CLEAN_CSV
from src.text_processing import (
    clean_text, remove_stopwords_and_clean
)
from src.date_extraction import extract_date_from_url, extract_date_from_text


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
train_df["timestamp"] = url_series.apply(extract_date_from_url)  # type: ignore
url_found = train_df["timestamp"].notna().sum()
print(f"  From URL:  {url_found} ({100 * url_found / len(train_df):.1f}%)")

# Layer 2: from article text (for rows still missing timestamp)
missing_mask = train_df["timestamp"].isna()
if missing_mask.sum() > 0:
    print(f"  Scanning text body for {missing_mask.sum()} remaining rows...")
    text_series = cast(pd.Series, train_df.loc[missing_mask, "text"])
    text_dates = text_series.apply(extract_date_from_text)  # type: ignore
    train_df.loc[missing_mask, "timestamp"] = text_dates
    text_found = text_dates.notna().sum()
    print(f"  From text: {text_found} ({100 * text_found / len(train_df):.1f}%)")

total_found = train_df["timestamp"].notna().sum()
print(f"  TOTAL:     {total_found} / {len(train_df)} ({100 * total_found / len(train_df):.1f}%)")
print(f"  Missing:   {train_df['timestamp'].isna().sum()}")


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
    "timestamp"
]].copy()

train_df.to_csv(CLEAN_CSV, index=False)

print(f"\nShape final: {train_df.shape}")
print(f"Saved to: {CLEAN_CSV}")
