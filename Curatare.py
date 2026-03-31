"""
Data Cleaning
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

import pandas as pd
import re

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)

RAW_PARQUET = os.path.join(data_dir, "rolargesum_raw.parquet")
CLEAN_CSV   = os.path.join(data_dir, "rolargesum_train_clean.csv")

'''
https://huggingface.co/datasets/avramandrei/rolargesum
https://github.com/avramandrei/rolargesum

Schema:
{
  "text": "This is the main text of the article",
  "summary": "This is the summary",
  "title": "Title of article",
  "keywords": "keyword1,keyword2,keyword3",
  "dialect": "romanian",
  "topics": "politica",
  "url": "www.example.com",
  "author": "John Doe"
}
'''


# LOAD DATASET (with local cache)
if os.path.exists(RAW_PARQUET):
    print(f"Loading dataset from local cache: {RAW_PARQUET}")
    train_df = pd.read_parquet(RAW_PARQUET)
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
    train_df = dataset["train"].to_pandas()

    train_df.to_parquet(RAW_PARQUET, index=False)
    print(f"Dataset cached at: {RAW_PARQUET} ({len(train_df)} rows)")

print("Shape initial:", train_df.shape)
print("Columns:", train_df.columns.tolist())
print(train_df.head(3))


# LOAD STOPWORDS
stopwords_path = os.path.join(base_dir, "stopwords-ro.txt")
with open(stopwords_path, "r", encoding="utf-8") as f:
    romanian_stopwords = {line.strip().lower() for line in f if line.strip()}


# CLEANING FUNCTIONS
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in romanian_stopwords]
    return " ".join(words)


def extract_date_from_url(url):
    if pd.isna(url):
        return pd.NaT

    url = str(url)
    match = re.search(r"(20\d{2})[-/](\d{2})[-/](\d{2})", url)
    if match:
        return pd.to_datetime(
            f"{match.group(1)}-{match.group(2)}-{match.group(3)}",
            errors="coerce"
        )
    return pd.NaT


# CLEAN ALL COLUMNS
train_df["title"]    = train_df["title"].apply(clean_text)
train_df["text"]     = train_df["text"].apply(clean_text)
train_df["summary"]  = train_df["summary"].apply(clean_text)
train_df["keywords"] = train_df["keywords"].apply(clean_text)
train_df["topics"]   = train_df["topics"].apply(clean_text)
train_df["dialect"]  = train_df["dialect"].apply(clean_text)
train_df["url"]      = train_df["url"].apply(clean_text)
train_df["author"]   = train_df["author"].apply(clean_text)

# Extract timestamps from URLs
train_df["timestamp"] = train_df["url"].apply(extract_date_from_url)
print("\nTimestamp coverage:")
print(f"  Found:   {train_df['timestamp'].notna().sum()}")
print(f"  Missing: {train_df['timestamp'].isna().sum()}")
print(train_df[["url", "timestamp"]].head(10))

# Construire documente
train_df["document"] = train_df["title"] + ". " + train_df["text"]
train_df["document"] = train_df["document"].apply(clean_text)

# Short document = title + first 500 chars of text
train_df["short_document"] = train_df["title"] + ". " + train_df["text"].str.slice(0, 500)
train_df["short_document"] = train_df["short_document"].apply(clean_text)


# FILTER
print("\nShape before filtering:", train_df.shape)

# Remove empty titles/documents
train_df = train_df[train_df["title"].str.strip() != ""].copy()
train_df = train_df[train_df["document"].str.strip() != ""].copy()

# Remove exact duplicates
train_df = train_df.drop_duplicates(subset=["document"]).reset_index(drop=True)

# Stopword-free version (for TF-IDF baseline)
train_df["document_nostop"] = train_df["document"].apply(remove_stopwords)

print("Shape after filtering:", train_df.shape)


# STATS
print("\nMissing values per column:")
print(train_df.isna().sum())

print("\nTop topic values:")
print(train_df["topics"].value_counts(dropna=False).head(20))

print("\nDialect values:")
print(train_df["dialect"].value_counts(dropna=False))

# SAVE
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