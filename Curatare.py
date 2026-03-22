import os
import pandas as pd
import re

from datasets import load_dataset
from huggingface_hub import login

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

login(token="REMOVED_API_TOKEN")  # TOKEN DE ACCES: NU MODIFICATI
dataset = load_dataset("avramandrei/rolargesum")
print(dataset)

# Folderul in care se afla scriptul
base_dir = os.path.dirname(os.path.abspath(__file__))

stopwords_path = os.path.join(base_dir, "stopwords-ro.txt")
with open(stopwords_path, "r", encoding="utf-8") as f:
    romanian_stopwords = {line.strip().lower() for line in f if line.strip()}

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

train_df = dataset["train"].to_pandas()

print("Shape initial:", train_df.shape)
print("Coloane:", train_df.columns.tolist())
print(train_df.head(3))

# Curatare coloane text
train_df["title"] = train_df["title"].apply(clean_text)
train_df["text"] = train_df["text"].apply(clean_text)
train_df["summary"] = train_df["summary"].apply(clean_text)
train_df["keywords"] = train_df["keywords"].apply(clean_text)
train_df["topics"] = train_df["topics"].apply(clean_text)
train_df["dialect"] = train_df["dialect"].apply(clean_text)
train_df["url"] = train_df["url"].apply(clean_text)
train_df["author"] = train_df["author"].apply(clean_text)
train_df["timestamp"] = train_df["url"].apply(extract_date_from_url)
print("\nValori timestamp lipsa:", train_df["timestamp"].isna().sum())
print("Valori timestamp gasite:", train_df["timestamp"].notna().sum())
print(train_df[["url", "timestamp"]].head(10))

print(train_df[["title", "text", "summary", "keywords"]].head(3))

print("Shape inainte de filtrare:", train_df.shape)

# Construire documente
train_df["document"] = train_df["title"] + ". " + train_df["text"]
train_df["document"] = train_df["document"].apply(clean_text)

train_df["short_document"] = train_df["title"] + ". " + train_df["text"].str.slice(0, 500)
train_df["short_document"] = train_df["short_document"].apply(clean_text)

# Eliminare randuri problematice
train_df = train_df[train_df["title"].str.strip() != ""].copy()
train_df = train_df[train_df["document"].str.strip() != ""].copy()

# Eliminare duplicate
train_df = train_df.drop_duplicates(subset=["document"]).reset_index(drop=True)

# Varianta fara stopwords - utila doar pentru baseline / TF-IDF
train_df["document_nostop"] = train_df["document"].apply(remove_stopwords)

print("Shape dupa filtrare:", train_df.shape)
print(train_df[["title", "document", "document_nostop"]].head(3))

print("\nValori lipsa pe coloane:")
print(train_df.isna().sum())

print("\nTop valori in topics:")
print(train_df["topics"].value_counts(dropna=False).head(20))

print("\nValori in dialect:")
print(train_df["dialect"].value_counts(dropna=False))

print(train_df[["document", "document_nostop"]].head(3))

# Pastrez coloanele importante
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

clean_path = os.path.join(base_dir, "rolargesum_train_clean.csv")
train_df.to_csv(clean_path, index=False)

print("\nShape final:", train_df.shape)
print("Fisier salvat la:", clean_path)