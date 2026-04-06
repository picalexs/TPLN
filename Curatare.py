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
import unicodedata

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


# =========================================================================
# LOAD DATASET (with local cache)
# =========================================================================
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


# =========================================================================
# LOAD STOPWORDS
# =========================================================================
stopwords_path = os.path.join(base_dir, "stopwords-ro.txt")
with open(stopwords_path, "r", encoding="utf-8") as f:
    romanian_stopwords = {line.strip().lower() for line in f if line.strip()}

def strip_diacritics(text):
    """Remove Romanian diacritics: ă→a, î→i, â→a, ș→s, ț→t"""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

romanian_stopwords_nodiac = {strip_diacritics(w) for w in romanian_stopwords}
romanian_stopwords = romanian_stopwords | romanian_stopwords_nodiac


# =========================================================================
# CLEANING FUNCTIONS
# =========================================================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deep_clean_text(text):
    """Extended cleaning: removes punctuation from word boundaries,
    numeric-only tokens, single chars, HTML remnants."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("\n", " ")
    # Remove HTML tags/entities
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)
    # Strip punctuation from word boundaries (keep internal hyphens/apostrophes)
    text = re.sub(r"[^\w\s'-]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords_and_clean(text):
    """Remove stopwords, strip diacritics, filter short/numeric tokens.
    Produces clean ASCII text suitable for TF-IDF."""
    if pd.isna(text):
        return ""
    text = str(text)
    # Split hyphenated words (e.g. "într-un" -> "într" "un")
    text = text.replace("-", " ").replace("–", " ").replace("—", " ")
    words = text.split()
    filtered = []
    for w in words:
        # Strip punctuation from edges
        clean_w = w.strip(".,;:!?\"'()…»«/\\")
        if not clean_w:
            continue
        # Strip diacritics for normalization
        clean_w = strip_diacritics(clean_w)
        # Skip short tokens (<=2 chars are usually noise like "ii", "sa", "ma")
        if len(clean_w) <= 2:
            continue
        # Skip numeric-only tokens
        if clean_w.isdigit():
            continue
        # Skip stopwords
        if clean_w in romanian_stopwords:
            continue
        filtered.append(clean_w)
    return " ".join(filtered)


def extract_date_from_url(url):
    """Extract date from URL pattern like /2021/03/23/ or /2021-03-23/"""
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


RO_MONTHS = {
    "ianuarie": "01", "februarie": "02", "martie": "03", "aprilie": "04",
    "mai": "05", "iunie": "06", "iulie": "07", "august": "08",
    "septembrie": "09", "octombrie": "10", "noiembrie": "11", "decembrie": "12",
}

def extract_date_from_text(text):
    """Extract date from article text body using multiple patterns."""
    if pd.isna(text):
        return pd.NaT
    text = str(text)[:500]

    # Pattern 1: "23 martie 2021" or "23 Martie 2021"
    for month_name, month_num in RO_MONTHS.items():
        pattern = rf"(\d{{1,2}})\s+{month_name}\s+(20\d{{2}})"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return pd.to_datetime(
                f"{match.group(2)}-{month_num}-{match.group(1).zfill(2)}",
                errors="coerce"
            )

    # Pattern 2: "23.03.2021" or "23/03/2021" (day.month.year, European format)
    match = re.search(r"(\d{1,2})[./](\d{1,2})[./](20\d{2})", text)
    if match:
        return pd.to_datetime(
            f"{match.group(3)}-{match.group(2).zfill(2)}-{match.group(1).zfill(2)}",
            errors="coerce"
        )

    # Pattern 3: "2021-03-23" (ISO format in text)
    match = re.search(r"(20\d{2})[-.](\d{2})[-.](\d{2})", text)
    if match:
        return pd.to_datetime(
            f"{match.group(1)}-{match.group(2)}-{match.group(3)}",
            errors="coerce"
        )

    return pd.NaT


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
train_df["timestamp"] = train_df["url"].apply(extract_date_from_url)
url_found = train_df["timestamp"].notna().sum()
print(f"  From URL:  {url_found} ({100 * url_found / len(train_df):.1f}%)")

# Layer 2: from article text (for rows still missing timestamp)
missing_mask = train_df["timestamp"].isna()
if missing_mask.sum() > 0:
    print(f"  Scanning text body for {missing_mask.sum()} remaining rows...")
    text_dates = train_df.loc[missing_mask, "text"].apply(extract_date_from_text)
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