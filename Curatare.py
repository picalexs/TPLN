import os
import pandas as pd
import re

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

# Folderul in care se afla scriptul
base_dir = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(base_dir, "rolargesum_train.csv")
test_path = os.path.join(base_dir, "rolargesum_test.csv")

print("Train path:", train_path)
print("Test path:", test_path)
print("Train exists:", os.path.exists(train_path))
print("Test exists:", os.path.exists(test_path))

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Curatare simpla
train_df["text"] = train_df["text"].apply(clean_text)
test_df["text"] = test_df["text"].apply(clean_text)

# Scoatem texte goale daca exista
train_df = train_df[train_df["text"].str.strip() != ""].copy()
test_df = test_df[test_df["text"].str.strip() != ""].copy()

# Scoatem duplicatele dupa text
train_df = train_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
test_df = test_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

train_out = os.path.join(base_dir, "rolargesum_train_clean.csv")
test_out = os.path.join(base_dir, "rolargesum_test_clean.csv")

train_df.to_csv(train_out, index=False)
test_df.to_csv(test_out, index=False)

print("Train clean shape:", train_df.shape)
print("Test clean shape:", test_df.shape)
print("Files saved:", train_out, test_out)