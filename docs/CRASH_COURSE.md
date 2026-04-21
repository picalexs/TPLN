# TPLN Crash Course

A plain-language guide to what this project does, the terminology you will
run into, and how the code in this repository fits together. Read this
first, then use [`PIPELINE_GUIDE.md`](./PIPELINE_GUIDE.md) as a deeper
reference.

---

## 1. What is this project, in one minute?

The goal is to **detect coordinated information campaigns** in Romanian news.

A "coordinated campaign" is a situation where many articles push the same
narrative at roughly the same time, often from a small set of sources. It
looks different from normal news because:

- the **content** is unusually similar across many articles
- the **timing** is unusually concentrated (a burst of publications)
- the **sources** are often less diverse than you would expect

So the project combines two signals:

1. **Semantic similarity**: articles that say the same thing cluster
   together in a numeric "meaning" space.
2. **Temporal concentration**: clusters whose articles are published in
   sudden bursts are flagged as suspicious.

Everything else in the codebase exists to compute one or both of those
signals reliably at scale.

---

## 2. Core terminology (read once, refer back forever)

### Natural Language Processing (NLP)

The field that deals with making computers work with human language.
Everything you are doing here is NLP.

### Corpus / Dataset

A large collection of text documents. In this project the corpus is
**RoLargeSum**: a dataset of Romanian news articles (title, body, summary,
URL, etc.).

### Tokenization

Splitting text into smaller units ("tokens"): words, sub-words, or
characters. Done implicitly by the models and libraries here. You do not
call a tokenizer directly.

### Stopwords

Extremely common words ("the", "is", "și", "de", "la") that carry little
topical meaning. Removed before TF-IDF so the vocabulary focuses on
content words. The list lives in `stopwords-ro.txt`.

### Embedding

A fixed-size vector of numbers that represents a piece of text. Two
articles with similar meaning get vectors that are close to each other in
this "embedding space". This is the backbone of the whole pipeline.

- Dimension in this project: **384** floats per article.
- Model used: **Sentence-BERT (multilingual MiniLM)**.

### Sentence-BERT (SBERT)

A specific family of transformer models trained to produce **one vector
per sentence or short document** such that cosine similarity between
vectors is a good proxy for semantic similarity.

Why not plain BERT? Plain BERT gives you one vector per *token*, and
comparing full sentences with it is awkward and slow. SBERT solves that
by design.

### Cosine similarity

A number in `[-1, 1]` that measures how similar two vectors are in
*direction*, ignoring their lengths. In this project, two articles are
"semantically similar" if their embedding vectors have high cosine
similarity (close to `1`).

### FAISS

A library (by Meta) for very fast nearest-neighbor search over vectors.
Given one article's embedding, FAISS can instantly retrieve the most
semantically similar articles. Here it is used mostly as a **debugging
and sanity-check tool** (`--debug-neighbors`), not a core pipeline step.

### Dimensionality reduction

Taking high-dimensional vectors (384-D) and squashing them down to a
smaller number of dimensions (15-D or 2-D) while trying to preserve
structure. Helps clustering and lets you plot points in 2-D.

### UMAP

A specific dimensionality-reduction algorithm. Used twice here:

- **15-D UMAP** is fed into HDBSCAN for clustering.
- The first two UMAP dimensions (**umap_x, umap_y**) are reused for the
  dashboard scatter plot.

### Clustering

Grouping items so that items in the same group are more similar to each
other than to items in other groups. We cluster article embeddings to
discover groups of articles that talk about the same topic or event.

### HDBSCAN

A **density-based** clustering algorithm. Key properties:

- You do **not** have to pick `k` (the number of clusters) up front.
- Some points are labelled `-1` = **noise** (not assigned to any cluster).
- Handles uneven cluster sizes and non-spherical shapes.

This is the main clustering algorithm in the project.

### KMeans

A simpler clustering algorithm used only as a **baseline** in
`scripts/TFIDFBaseline.py`:

- You must pick `k` up front.
- It forces every point into some cluster (no "noise" bucket).
- It assumes roughly spherical clusters.

### TF-IDF

"Term Frequency — Inverse Document Frequency". A classical way to turn a
document into a sparse vector based on how often each word appears, down-
weighted by how common that word is in the whole corpus. Interpretable,
fast, but less semantic than SBERT. Used here only as an **ablation
baseline** to prove SBERT is worth the effort.

### Silhouette score

A metric that tells you how well points fit their assigned cluster
(`+1` = excellent, `0` = borderline, `-1` = likely in the wrong cluster).
Used to compare HDBSCAN configurations and as an evaluation metric in the
ablation.

### Burst detection

Given a timeline of events (here: "an article was published on day X"),
burst detection identifies time windows where the event rate is
unusually high compared to the overall average.

### Kleinberg's algorithm

A specific, classical way to detect bursts, based on a two-state model
("normal rate" vs "burst rate"). Implemented in
`scripts/TemporalAnalysis.py` with a manual fallback so the code does not
depend on an external burst library being installed.

### Suspicion score

A single number per cluster, combining:

- burst strength (daily + weekly)
- temporal concentration (how peaky is the activity)
- cluster size
- reliability weights (timestamp quality, domain diversity, etc.)
- penalties (sparse coverage, single-domain dominance, low-quality
  timestamps)

Higher `suspicion_score` = this cluster looks more like a coordinated
campaign. Computed in `scripts/TemporalAnalysis.py`.

### Parquet

A columnar binary file format for tabular data. Faster and smaller than
CSV. The pipeline uses parquet everywhere except for embeddings, which
stay as `.npy` NumPy arrays.

---

## 3. Repository layout

```
TPLN/
├── scripts/              <- one entry point per pipeline stage (run these)
│   ├── DataCuration.py
│   ├── EmbeddingsClustering.py
│   ├── TemporalAnalysis.py
│   ├── Evaluation.py
│   ├── TFIDFBaseline.py
│   ├── PrepareDashboardData.py
│   └── Dashboard.py
│
├── src/                  <- reusable library code (imported by scripts)
│   ├── config.py           (constants, hyperparameters, SBERT model name)
│   ├── paths.py            (where every artefact lives on disk)
│   ├── io_utils.py         (parquet loading helpers)
│   ├── text_processing.py  (Romanian-aware cleaning + stopword removal)
│   ├── date_extraction.py  (timestamps from URLs and article bodies)
│   ├── topic_mapping.py    (noisy topic labels -> canonical buckets)
│   └── runtime_profile.py  (auto-detect CPU/GPU, threads, batch sizes)
│
├── data/                 <- everything generated by the pipeline
├── docs/                 <- this guide and the pipeline guide
├── stopwords-ro.txt      <- Romanian stopwords (with and without diacritics)
├── requirements.txt      <- pinned Python dependencies
└── README.md             <- how to install and run
```

**Rule of thumb**: `scripts/` are *verbs* (things you run), `src/` is
*library code* (things that get imported). The repo root deliberately
stays thin.

---

## 4. The pipeline, end to end

Run in this order (from `README.md`):

```bash
python scripts/DataCuration.py          # Stage 1
python scripts/EmbeddingsClustering.py  # Stage 2 + 3
python scripts/TemporalAnalysis.py      # Stage 4
python scripts/Evaluation.py            # Stage 5
python scripts/PrepareDashboardData.py  # Stage 6
streamlit run scripts/Dashboard.py      # Stage 7
python scripts/TFIDFBaseline.py         # Optional ablation
```

Each stage reads the previous stage's parquet output and writes a new one.
Later stages never go back to re-download or re-clean the raw data.

### Stage 1 — Data Curation (`scripts/DataCuration.py`)

**Input**: RoLargeSum from Hugging Face, or cached
`data/rolargesum_raw.parquet` if it already exists.

**What it does**:

1. Downloads (or loads cached) raw dataset.
2. Cleans text columns (lowercasing, whitespace, emoji stripping).
3. Extracts timestamps in two passes:
   - First from the article URL (most reliable).
   - If no URL date, from date-like patterns in the first ~220 chars of
     the article body.
4. Validates timestamps and labels each one with:
   - `timestamp_source`: `url`, `text`, or `missing`
   - `timestamp_quality`: `high`, `medium`, `low`, `missing`, or
     `rejected_future`
5. Builds three text views of each article:
   - `document`: title + full body (best for deduplication).
   - `short_document`: title + first 500 chars (input to SBERT — smaller
     is faster, and most news meaning lives early in the article).
   - `document_nostop`: diacritics stripped, stopwords removed, punctuation
     gone. Used only by TF-IDF.
6. Drops duplicates on `document`.

**Output**: `data/rolargesum_train_clean.parquet`.

Supporting code used here:

- `src/text_processing.py` — `clean_text`, `remove_stopwords_and_clean`.
- `src/date_extraction.py` — regex-based URL and text date extraction.
- `src/config.py` — which columns to preserve, which quality labels
  exist.

### Stage 2 — Topic Normalization (`src/topic_mapping.py`)

Raw topic labels in the corpus are messy (aliases, composite labels,
missing values, format labels like "editorial"). This module folds them
into a small set of stable buckets:

```
politica, social, international, economie, sanatate, justitie,
sport, educatie, stiinta, cultura, diverse, necunoscut
```

Each article gets:

- `topic_group`: the canonical bucket.
- `topic_group_reason`: why that bucket was picked
  (`canonical_topic`, `mapped_topic_alias`, `composite_topic`,
  `missing_topic`, `unmapped_topic`).

This stage is not a separate script: it runs inside Stage 3 as the first
thing `scripts/EmbeddingsClustering.py` does with the cleaned data.

### Stage 3 — Embeddings and Clustering (`scripts/EmbeddingsClustering.py`)

The semantic core of the project.

1. **Filter** to topics with at least `MIN_TOPIC_SIZE` = 200 articles.
   Smaller topics are marked noise and skipped (they do not have enough
   data to cluster meaningfully).

2. **Embed** each topic separately:
   - Input text: `short_document`.
   - Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
   - Output: `(N, 384)` float32 matrix, saved to
     `data/embeddings/<topic>_embeddings.npy`.
   - Cached, so reruns are fast.

3. **(Optional) FAISS sanity check**: build an in-memory FAISS index and
   print nearest neighbors for the first few articles. Controlled by
   `--debug-neighbors`. Not persisted.

4. **Reduce dimensions with UMAP** from 384-D to 15-D (for clustering)
   and keep the first 2 dimensions for the dashboard scatter plot.

5. **Sweep HDBSCAN configurations** per topic. Each config tries
   different values of `min_cluster_size`, `min_samples`, and sometimes
   `cluster_selection_method` / `cluster_selection_epsilon`. Political
   and social topics get custom configs in `src/config.py`.

6. **Score each config** with:
   - noise percentage (lower is better, but only up to a point)
   - number of clusters (penalize <3 heavily)
   - largest-cluster share (penalize "one giant cluster eats everything")
   - silhouette score on a 5000-row sample
   Then pick the best config per topic.

7. **Write** the final labelled dataset with per-article fields such as:
   - `cluster`, `cluster_size`, `cluster_membership_strength`,
     `cluster_outlier_score`
   - `umap_x`, `umap_y` (for dashboard)
   - `best_min_cluster_size`, `best_min_samples`, etc.
   - `topic_is_eligible` (False for small topics marked noise)

**Outputs**:

- `data/clusters/clustered_data.parquet` — article-level labels.
- `data/clusters/hdbscan_config_results.parquet` — every config tried.
- `data/embeddings/*.npy` — one file per topic bucket.

### Stage 4 — Temporal Analysis (`scripts/TemporalAnalysis.py`)

Given semantic clusters, ask "which ones *also* look suspicious over time?"

For every real cluster (cluster != -1) it computes:

- **Coverage**: how many articles actually have timestamps, and how
  reliable those timestamps are (URL vs text vs missing).
- **Domain stats**: how many unique domains contribute, how dominant the
  top domain is, domain entropy.
- **Time spread**: first/last article dates, active days, median and max
  gap between active days.
- **Peak measures**: `peak_day_count`, `peak_day_share`,
  `peak_to_baseline_ratio`.
- **Burst scores**:
  - `burst_score_daily` via Kleinberg's model with `s=2.0`, `gamma=1.0`.
  - `burst_score_weekly` via the same algorithm on 7-day windows.
  - `burst_stable = 1` if both daily and weekly bursts fire (more
    trustworthy signal).
- **Weights** (multiplicative confidence factors, each in `[0, 1]`):
  `support_weight`, `coverage_weight`, `source_weight`, `domain_weight`.
- **Penalties**:
  - `long_sparse_span_penalty` — big calendar spans with very few active
    days look like background noise, not campaigns.
  - `single_domain_penalty` — one outlet hammering one topic is
    suspicious but often mundane repetition.
  - `source_reliability_penalty` — punishes clusters where most
    timestamps came from weak sources.
- **Final**:

  ```text
  suspicion_score = max(
      0,
      raw_signal * support_weight * coverage_weight
                 * source_weight   * domain_weight
      - support_penalty
      - (long_sparse + single_domain + source_reliability + compactness)
  )
  ```

Clusters are sorted by `suspicion_score` descending and written to
`data/temporal/cluster_temporal_stats.parquet`.

**Mental model**: suspicion_score asks
*"is this cluster compact in time, from a healthy mix of sources, with
enough well-timestamped articles, and does it burst?"*

### Stage 5 — Evaluation (`scripts/Evaluation.py`)

Builds a single consolidated report by stitching together:

1. **Best HDBSCAN config per topic** (from the config sweep parquet).
2. **Mean intra-cluster cosine similarity** per real cluster (uses
   cached embeddings). Higher = more semantically tight cluster.
3. **Temporal burst data** joined on `(topic_group, cluster)`.

Output: `data/evaluation_report.parquet` with a `section` column
distinguishing silhouette rows, intra-cosine rows, and burst rows.

### Stage 6 — Ablation baseline (`scripts/TFIDFBaseline.py`, optional)

Compares three approaches on the same eligible topics:

1. **TF-IDF + KMeans** — classical baseline.
2. **SBERT + KMeans** — isolates the effect of the representation
   (keeps KMeans constant, changes the vectors).
3. **SBERT + HDBSCAN** — the main pipeline, pulled from the config sweep.

`k` for KMeans is set from the number of HDBSCAN clusters per topic so
the comparison is fair. Silhouette and Davies-Bouldin scores are
reported. Output: `data/tfidf_ablation_report.parquet`.

This is how you answer "is all this SBERT + UMAP + HDBSCAN machinery
actually worth it compared to a simple TF-IDF + KMeans setup?"

### Stage 7 — Dashboard assets (`scripts/PrepareDashboardData.py`)

Precomputes lean parquet files so Streamlit does not have to process the
full clustered dataset on every page load. Among other things it writes:

- `dashboard/cluster_overview.parquet` — one row per cluster.
- `dashboard/topic_summary.parquet` — one row per topic bucket.
- `dashboard/cluster_daily_counts.parquet` — per-cluster time series.
- `dashboard/cluster_articles.parquet` — per-article detail.
- `dashboard/scatter_sample.parquet` — anchor + edge + noise samples for
  the scatter plot (plotting all points would be slow and messy).
- Plus copies of temporal, config, evaluation, and ablation parquets
  into `data/dashboard/`.

### Stage 8 — Streamlit UI (`scripts/Dashboard.py`)

A read-only web app with four pages:

- **Cluster Explorer** — pick a topic and a cluster, see articles and
  the UMAP scatter.
- **Timeline and Bursts** — daily and weekly activity plus burst flags.
- **Top Campaigns** — clusters ranked by `suspicion_score`.
- **Evaluation and Ablation** — silhouette tables, cosine tables, and
  the TF-IDF vs SBERT comparison.

All heavy work is already done by Stage 7 — the dashboard only reads
parquet and renders.

---

## 5. The shared infrastructure (`src/`)

You will import from these modules often. Here is what each owns.

### `src/config.py`

All tunable constants and hyperparameters. Worth skimming end-to-end.
Highlights:

- `SBERT_MODEL` — which Sentence-BERT checkpoint to use.
- `TEXT_COLUMN = "short_document"` — what we feed into SBERT.
- `MIN_TOPIC_SIZE = 200` — topics smaller than this are not clustered.
- `UMAP_N_NEIGHBORS = 30` — UMAP neighborhood size.
- `HDBSCAN_DEFAULT_CONFIGS` + `HDBSCAN_TOPIC_CONFIGS` — the config sweep.
- `TFIDF_MAX_FEATURES`, `TFIDF_NGRAM` — baseline vectorizer settings.
- `ROMANIAN_MONTHS`, timestamp quality/source constants.

### `src/paths.py`

Every file path used anywhere in the pipeline. If you wonder "where does
X live on disk?", the answer is here. Also creates all needed
sub-directories under `data/` on import.

### `src/io_utils.py`

`load_clean_data(...)` — the canonical way to read
`rolargesum_train_clean.parquet` with schema validation.

### `src/text_processing.py`

- `clean_text` — lowercase + whitespace + emoji-strip.
- `deep_clean_text` — also strips HTML, URLs, and boundary punctuation.
- `strip_diacritics` — Romanian-aware (`ă→a`, `î→i`, `ș→s`, `ț→t`).
- `remove_stopwords_and_clean` — diacritic-folded, stopword-stripped,
  short-token-filtered text for TF-IDF.
- `load_stopwords` — thread-safe cached loader for `stopwords-ro.txt`.

### `src/date_extraction.py`

- `extract_date_from_url_with_source` — tries URL path, URL query
  params (`?date=...`, `?year=...&month=...`), compact `YYYYMMDD`, etc.
- `extract_date_from_text` — Romanian month names ("23 martie 2021"),
  European dotted format ("23.03.2021"), ISO format ("2021-03-23").
- `validate_extracted_timestamp` — assigns quality labels and rejects
  implausible future dates.

### `src/topic_mapping.py`

- `CANONICAL_TOPICS` — the 12 stable buckets.
- `TOPIC_MAPPING` — alias → canonical bucket lookup table.
- `normalize_topic` — raw label → bucket name.
- `normalize_topic_with_reason` — same, plus the reason code.

### `src/runtime_profile.py`

Auto-detects the machine and picks sensible defaults for:

- `device`: `cuda`, `mps`, `dml`, or `cpu`.
- `cpu_threads`: for OMP/MKL/torch/numba.
- `embedding_batch_size`: scales with GPU memory.
- `chunk_size`: parquet chunking.

The only thing currently GPU-accelerated is **Sentence-BERT inference**.
Everything else (UMAP, HDBSCAN, temporal analysis, TF-IDF) runs on CPU
by design.

---

## 6. Data artefacts at a glance

Everything below is in `data/`. The pipeline is parquet-first.

| File                                                  | Produced by                 | Contents                                   |
| ----------------------------------------------------- | --------------------------- | ------------------------------------------ |
| `rolargesum_raw.parquet`                              | Stage 1                     | Raw download, kept for fast reruns         |
| `rolargesum_train_clean.parquet`                      | Stage 1                     | Cleaned articles + timestamps              |
| `embeddings/<topic>_embeddings.npy`                   | Stage 3                     | SBERT vectors per topic                    |
| `clusters/clustered_data.parquet`                     | Stage 3                     | Article-level cluster labels + UMAP coords |
| `clusters/hdbscan_config_results.parquet`             | Stage 3                     | Every config tried in the sweep            |
| `temporal/cluster_temporal_stats.parquet`             | Stage 4                     | Per-cluster burst + suspicion scores       |
| `evaluation_report.parquet`                           | Stage 5                     | Silhouette + cosine + burst summary        |
| `tfidf_ablation_report.parquet`                       | Stage 6                     | TF-IDF vs SBERT comparison                 |
| `dashboard/*.parquet`                                 | Stage 7                     | Precomputed assets for Streamlit           |

---

## 7. Mental model — how to think about this pipeline

Think of it as a **funnel**:

1. Start with ~hundreds of thousands of raw articles.
2. Clean them → a usable dataset with timestamps and topic buckets.
3. Embed every article → now every article is a point in 384-D space
   where "close" means "similar meaning".
4. Cluster per topic → groups of articles about the same thing.
5. Look at the timeline of each cluster → which clusters burst?
6. Combine burst + size + reliability + source diversity → a single
   `suspicion_score` per cluster.
7. Sort by that score → top candidates for coordinated campaigns.

The **semantic layer** (embeddings + clustering) answers *"what are they
talking about?"*. The **temporal layer** answers *"when are they talking
about it, and does that pattern look natural?"*. Neither is sufficient
alone: a famous real event naturally produces a semantic cluster *and* a
burst, so you need the extra weights and penalties to separate genuine
campaigns from ordinary spikes of news.

---

## 8. How to run things, quick reference

First-time setup:

```bash
pip install -r requirements.txt
# Add HF_TOKEN to .env for the first DataCuration run
```

Optional NVIDIA GPU (for faster SBERT only):

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Full pipeline:

```bash
python scripts/DataCuration.py
python scripts/EmbeddingsClustering.py
python scripts/TemporalAnalysis.py
python scripts/Evaluation.py
python scripts/PrepareDashboardData.py
streamlit run scripts/Dashboard.py
```

Useful flags:

- `--nrows N` on `DataCuration.py`, `EmbeddingsClustering.py`,
  `TFIDFBaseline.py` — cap the dataset for quick experiments.
- `--device {auto,cpu,cuda,mps,dml}` on `EmbeddingsClustering.py` —
  force a specific backend.
- `--cpu-threads N` on most scripts — override thread count.
- `--debug-neighbors` on `EmbeddingsClustering.py` — print FAISS nearest
  neighbors to sanity check embeddings.
- `--max-k N` on `TFIDFBaseline.py` — cap KMeans cluster count on huge
  topics for speed.

---

## 9. Common questions

### Why `short_document` instead of the full article for SBERT?

Most of a news article's semantic identity is in the title and the first
paragraph. Using a shorter input cuts embedding time and memory use
without losing much meaning.

### Why HDBSCAN instead of KMeans?

KMeans forces every article into a cluster, needs `k` up front, and
prefers spherical clusters. Real news data has outliers, uneven cluster
sizes, and a variable number of true "stories". HDBSCAN handles all
three.

### Why a separate TF-IDF baseline if SBERT is clearly better?

To *prove* it is better on this data, not just assume it. The ablation
tests two dimensions at once:

- representation (TF-IDF vs SBERT)
- clustering method (KMeans vs HDBSCAN)

### Why keep `.npy` embeddings around?

They are the most expensive artefact to produce. Evaluation, the TF-IDF
ablation, and any re-clustering all reuse the same embeddings directly.

### Why topic-first instead of global clustering?

Clustering every Romanian article in one huge embedding space is slow,
wasteful, and blurs obvious topical boundaries (politics articles should
not be clustered together with sports articles just because they share a
few function words). Splitting by normalized topic bucket first lets each
topic use its own HDBSCAN config.

### Why is suspicion score "not just the biggest burst"?

Because the biggest burst is often just a real, big news event. The
score deliberately combines burst strength with size, domain diversity,
timestamp quality, and explicit penalties so that genuinely suspicious
clusters (compact, multi-source, well-timestamped, bursty) rank above
large-but-boring ones.

---

## 10. Short glossary (for fast look-ups)

- **SBERT**: Sentence-BERT. Produces one embedding per short document;
  cosine similarity between embeddings ≈ semantic similarity.
- **Embedding**: a fixed-length vector of floats representing a piece of
  text.
- **Cosine similarity**: angle-based similarity of two vectors, in
  `[-1, 1]`, where `1` means identical direction.
- **UMAP**: nonlinear dimensionality reducer; used for 15-D clustering
  input and 2-D visualization coordinates.
- **HDBSCAN**: density-based clustering; labels outliers as noise
  (`cluster = -1`); main algorithm in this project.
- **KMeans**: centroid clustering baseline; requires `k`; forces every
  point into a cluster.
- **FAISS**: fast nearest-neighbor search library; used here for
  debugging, not persisted.
- **TF-IDF**: sparse word-frequency representation, corpus-weighted; the
  baseline input for the ablation study.
- **Silhouette score**: metric telling you how well points fit their
  assigned cluster, in `[-1, 1]`.
- **Kleinberg burst detection**: algorithm for finding time windows with
  an unusually high event rate.
- **Suspicion score**: project-specific combined metric for "how much
  does this cluster look like a coordinated campaign?"
- **Parquet**: columnar binary tabular file format used for every
  tabular artefact here.
- **Topic bucket**: one of the canonical normalized topic labels
  (`politica`, `social`, `international`, ...).
- **Noise (cluster = -1)**: articles HDBSCAN did not confidently assign
  to any cluster.
- **Suspicion weights**: multiplicative factors in `[0, 1]` that reduce
  the score when evidence is weak (coverage, sources, domains, etc.).
- **Suspicion penalties**: additive deductions that further reduce the
  score for known "false positive" shapes (single-domain dominance,
  sparse long spans, low timestamp reliability).
