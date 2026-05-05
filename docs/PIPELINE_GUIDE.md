# Pipeline Guide

This document explains how the coordinated-campaign detection pipeline works,
which files own each stage, what data flows between stages, and what the main
models and algorithms are doing.

## Quick File Map

- Main pipeline entrypoints:
  - [`scripts/DataCuration.py`](../scripts/DataCuration.py)
  - [`scripts/EmbeddingsClustering.py`](../scripts/EmbeddingsClustering.py)
  - [`scripts/TemporalAnalysis.py`](../scripts/TemporalAnalysis.py)
  - [`scripts/Evaluation.py`](../scripts/Evaluation.py)
  - [`scripts/TFIDFBaseline.py`](../scripts/TFIDFBaseline.py)
  - [`scripts/PrepareDashboardData.py`](../scripts/PrepareDashboardData.py)
  - [`scripts/Dashboard.py`](../scripts/Dashboard.py)
- Optional maintenance and report entrypoints:
  - [`scripts/apply_campaign_candidate_scoring.py`](../scripts/apply_campaign_candidate_scoring.py)
  - [`scripts/fix_multiyear_suspicion.py`](../scripts/fix_multiyear_suspicion.py)
  - [`scripts/report_figures/run_all.py`](../scripts/report_figures/run_all.py)
  - [`report/figures/FIGURES.md`](../report/figures/FIGURES.md)
  - [`report/figures/TOP_5_COMPACT_CAMPAIGN_CANDIDATES.md`](../report/figures/TOP_5_COMPACT_CAMPAIGN_CANDIDATES.md)
- Shared infrastructure:
  - [`src/config.py`](../src/config.py)
  - [`src/paths.py`](../src/paths.py)
  - [`src/io_utils.py`](../src/io_utils.py)
  - [`src/runtime_profile.py`](../src/runtime_profile.py)
  - [`src/topic_mapping.py`](../src/topic_mapping.py)
  - [`src/text_processing.py`](../src/text_processing.py)
  - [`src/date_extraction.py`](../src/date_extraction.py)
  - [`src/campaign_scoring.py`](../src/campaign_scoring.py)
- Entry points are grouped in `scripts/` so the repo root stays focused on
  shared code, docs, and configuration.

## High-Level Goal

The project tries to detect coordinated or suspicious information campaigns in a
large Romanian news corpus.

It does that by combining two signals:

1. Semantic similarity: articles that talk about the same thing should cluster
   together in embedding space.
2. Temporal concentration: coordinated campaigns often publish many similar
   articles in short time windows or in bursty patterns.

The pipeline first finds semantic clusters, then asks which of those clusters
also look temporally suspicious.

## Architecture

```text
                        +----------------------------------+
                        | RoLargeSum dataset (HF)          |
                        +----------------+-----------------+
                                         |
                                         v
                      +--------------------------------------+
                      | scripts/DataCuration.py              |
                      | - clean text                         |
                      | - extract timestamps                 |
                      | - build document fields              |
                      | - remove duplicates                  |
                      +----------------+---------------------+
                                       |
                                       v
                     data/rolargesum_train_clean.parquet
                                       |
                                       v
                  +---------------------------------------------+
                  | scripts/EmbeddingsClustering.py             |
                  | - normalize topic labels                    |
                  | - encode short_document with SBERT          |
                  | - cache embeddings per topic                |
                  | - persist FAISS indexes + KNN edges         |
                  | - UMAP 15-D reduction                       |
                  | - HDBSCAN config sweep per topic            |
                  | - choose best clustering config             |
                  +------------------+--------------------------+
                                     |
                  +------------------+--------------------------+
                  |                                             |
                  v                                             v
 data/embeddings/*.npy                         data/clusters/clustered_data.parquet
 data/faiss/*.faiss                            data/clusters/runtime_observability.parquet
 data/faiss/knn/*.parquet
                                               data/clusters/hdbscan_config_results.parquet
                                                              |
                             +--------------------------------+--------------------------------+
                             |                                                                 |
                             v                                                                 v
             +----------------------------------+                         +----------------------------------+
             | scripts/TemporalAnalysis.py      |                         | scripts/TFIDFBaseline.py         |
             | - daily/weekly burst detection   |                         | - TF-IDF + KMeans                |
             | - coverage/domain/source stats   |                         | - SBERT + KMeans                 |
             | - suspicion score                |                         | - compare with SBERT+HDBSCAN     |
             | - campaign candidate score       |                         | - burst on/off ablation          |
             +----------------+-----------------+                         +----------------+-----------------+
                              |                                                              |
                              v                                                              v
     data/temporal/cluster_temporal_stats.parquet                        data/tfidf_ablation_report.parquet
                              |
                              v
                  +-------------------------------+
                  | scripts/Evaluation.py         |
                  | - best-config summary         |
                  | - intra-cluster cosine        |
                  | - burst/size relationships    |
                  +-------------------------------+
                              |
                              v
                  data/evaluation_report.parquet
                              |
                              v
                +--------------------------------------+
                | scripts/PrepareDashboardData.py      |
                | - build dashboard parquet assets     |
                | - cluster overview                   |
                | - article detail                     |
                | - daily counts                       |
                | - scatter sample                     |
                | - semantic neighbor evidence         |
                | - cross-cluster similarity           |
                +----------------+---------------------+
                                 |
                                 v
                       data/dashboard/*.parquet
                                 |
                                 v
                         scripts/Dashboard.py (Streamlit)
                                 |
                                 v
                 scripts/report_figures/run_all.py
                                 |
                                 v
                       report/figures/*.png
```

## Data Artefacts

The pipeline is parquet-first for tabular data. The important outputs are:

- [`data/rolargesum_raw.parquet`](../data/rolargesum_raw.parquet)
- [`data/rolargesum_train_clean.parquet`](../data/rolargesum_train_clean.parquet)
- [`data/embeddings/*.npy`](../data/embeddings/)
- [`data/faiss/*.faiss`](../data/faiss/)
- [`data/faiss/*.json`](../data/faiss/)
- [`data/faiss/knn/*_knn.parquet`](../data/faiss/knn/)
- [`data/clusters/clustered_data.parquet`](../data/clusters/clustered_data.parquet)
- [`data/clusters/hdbscan_config_results.parquet`](../data/clusters/hdbscan_config_results.parquet)
- [`data/clusters/runtime_observability.parquet`](../data/clusters/runtime_observability.parquet)
- [`data/temporal/cluster_temporal_stats.parquet`](../data/temporal/cluster_temporal_stats.parquet)
- [`data/evaluation_report.parquet`](../data/evaluation_report.parquet)
- [`data/tfidf_ablation_report.parquet`](../data/tfidf_ablation_report.parquet)
- [`data/dashboard/*.parquet`](../data/dashboard/)
- [`report/figures/*.png`](../report/figures/)

Why this matters:

- parquet is much faster than CSV for large intermediate tables
- embeddings stay as `.npy` because they are dense numeric arrays and are reused
  directly by multiple scripts
- FAISS indexes and KNN edge parquet files are persisted so the dashboard can
  show semantic-neighbor evidence without rebuilding nearest-neighbor search

## Stage 1: Data Curation

Code:

- [`scripts/DataCuration.py`](../scripts/DataCuration.py)
- [`src/text_processing.py`](../src/text_processing.py)
- [`src/date_extraction.py`](../src/date_extraction.py)
- [`src/config.py`](../src/config.py)

### What Data Curation does

This stage prepares the raw article dataset for everything downstream.

Main responsibilities:

1. Load the cached raw parquet or download RoLargeSum from Hugging Face.
2. Clean the basic text columns:
   - `title`
   - `text`
   - `summary`
   - `keywords`
   - `topics`
   - `dialect`
   - `url`
   - `author`
3. Extract timestamps in three passes:
   - first from URL structure
   - then from article text if URL extraction fails
   - then from `htmldate` as a final URL-based fallback
4. Validate timestamp quality and reject impossible future dates.
5. Build:
   - `document`: title + full article text
   - `short_document`: title + first ~500 chars of text
   - `document_nostop`: cleaned stopword-free text for TF-IDF
6. Drop duplicates on `document`.

### Why there are three text fields

- `document`
  - the most complete version
  - useful for deduplication and general textual analysis
- `short_document`
  - shorter input for Sentence-BERT
  - reduces embedding cost without losing most semantic signal
- `document_nostop`
  - more aggressively cleaned
  - optimized for bag-of-words / TF-IDF style methods

### Timestamp extraction logic

Timestamp extraction is layered on purpose:

1. URL date patterns are often the most reliable because many publishers encode
   publication dates in the article URL.
2. If that fails, the pipeline scans the article text for publication-like date
   snippets near the article header.
3. If both local passes fail, the pipeline tries `htmldate` against the URL as a
   final fallback.

Each timestamp also gets:

- `timestamp_source`: `url`, `text`, `htmldate`, or `missing`
- `timestamp_quality`: `high`, `medium`, `low`, `missing`, or `rejected_future`

That information is later reused by the temporal stage to avoid trusting weak
time signals too much.

## Stage 2: Topic Normalization

Code:

- [`src/topic_mapping.py`](../src/topic_mapping.py)

### What Topic Normalization does

Raw topic labels in the corpus are noisy. Some are canonical, some are aliases,
some are formatting labels, some are mixed strings, and some are missing.

This module maps raw labels into stable buckets such as:

- `politica`
- `social`
- `international`
- `economie`
- `sanatate`
- `justitie`
- `sport`
- `educatie`
- `stiinta`
- `cultura`
- `diverse`
- `necunoscut`

It also returns a reason code such as:

- `canonical_topic`
- `mapped_topic_alias`
- `composite_topic`
- `missing_topic`
- `unmapped_topic`

That makes the pipeline more explainable: later stages know not only the chosen
topic bucket, but also how that bucket was chosen.

## Stage 3: Embeddings and Clustering

Code:

- [`scripts/EmbeddingsClustering.py`](../scripts/EmbeddingsClustering.py)
- [`src/runtime_profile.py`](../src/runtime_profile.py)
- [`src/config.py`](../src/config.py)

This is the core semantic stage.

### What Sentence-BERT is doing

The project uses:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

This is a Sentence-BERT model, often abbreviated SBERT.

Plain BERT is good at token-level contextual representations, but it is not
ideal as-is for directly comparing two entire sentences or documents with a
simple cosine similarity.

Sentence-BERT changes that workflow by producing a fixed-size embedding vector
for the whole input text. In this project:

- input: `short_document`
- output: a 384-dimensional dense vector
- similarity metric: cosine / inner-product style similarity

Why SBERT is a good fit here:

- semantically similar articles end up close in vector space
- multilingual support helps with Romanian-language news
- once cached, embeddings can be reused by clustering, evaluation, and ablation
  stages

### Why embeddings are cached per topic

The script writes one `.npy` file per topic bucket in `data/embeddings/`.

That gives two benefits:

1. reruns are much faster if topic membership did not change
2. later scripts like [`scripts/Evaluation.py`](../scripts/Evaluation.py) and
   [`scripts/TFIDFBaseline.py`](../scripts/TFIDFBaseline.py) can reuse the same embeddings

### Why FAISS is persisted

The script now builds and persists one FAISS index per eligible topic in
`data/faiss/`, together with a small JSON metadata file. The metadata includes a
document fingerprint so stale indexes can be rebuilt safely when topic contents
change.

By default the script also writes top-k nearest-neighbor edges to
`data/faiss/knn/*_knn.parquet`. Those edges are consumed by
[`scripts/PrepareDashboardData.py`](../scripts/PrepareDashboardData.py) and then
shown in [`scripts/Dashboard.py`](../scripts/Dashboard.py) as semantic-neighbor
evidence for selected articles.

FAISS has three roles here:

- reuse nearest-neighbor indexes across runs
- provide dashboard evidence for "these articles are semantically close"
- support optional debugging with `--debug-neighbors`

### What UMAP is doing

HDBSCAN tends to work better after a moderate dimensionality reduction step.

UMAP takes the dense SBERT embedding space and projects it into a smaller space.
In this project:

- input: 384-D SBERT vectors
- output: 15-D reduced vectors for clustering

The first two UMAP dimensions are also reused for dashboard scatter plotting.

Why UMAP is useful here:

- preserves local neighborhood structure reasonably well
- reduces noise and computational cost before HDBSCAN
- gives a compact 2-D projection for visualization

### What HDBSCAN is doing

HDBSCAN is a density-based clustering algorithm.

Very roughly:

- regions with many nearby points become clusters
- isolated points become noise (`cluster = -1`)
- it does not force every point into a cluster
- it does not require pre-specifying a single global `k`

That makes it a strong fit for news clustering, because:

- some topics naturally contain many coherent sub-stories
- some articles are outliers and should remain unlabeled noise
- cluster shapes in embedding space are not necessarily spherical

### Why there is a config sweep

HDBSCAN has important hyperparameters:

- `min_cluster_size`
- `min_samples`
- sometimes `cluster_selection_method`
- sometimes `cluster_selection_epsilon`

Instead of trusting one global setting, the script tries several configs per
topic bucket and picks the best one using:

- noise percentage
- cluster structure
- largest-cluster penalty
- silhouette score on a fixed sample

That logic lives in:

- `evaluate_topic_configs`
- `compute_clustering_metrics`
- `compute_selection_score`

### Final output of this stage

The final clustered parquet contains article-level labels and metadata such as:

- `topic_group`
- `topic_group_reason`
- `cluster`
- `cluster_size`
- `cluster_membership_strength`
- `cluster_outlier_score`
- `topic_row_idx`
- `umap_x`
- `umap_y`
- best HDBSCAN config fields

The stage also writes:

- persisted FAISS indexes and metadata in `data/faiss/`
- top-k neighbor edges in `data/faiss/knn/`
- per-topic timing rows in `data/clusters/runtime_observability.parquet`

Small topics below `MIN_TOPIC_SIZE` are not clustered; they are marked as noise
with `topic_is_eligible = False`.

## Stage 4: Temporal Analysis

Code:

- [`scripts/TemporalAnalysis.py`](../scripts/TemporalAnalysis.py)

This stage asks: “which semantic clusters also behave suspiciously over time?”

### Core idea

A semantic cluster alone is not enough to call something coordinated. Many real
stories naturally generate many similar articles. The temporal stage looks for
patterns such as:

- unusually concentrated publication bursts
- very peaky short-term activity
- abnormal time concentration relative to cluster span
- weak timestamp provenance
- suspicious domain concentration

### What Kleinberg burst detection is

The script uses Kleinberg-style burst detection at:

- daily granularity
- weekly granularity

Burst detection tries to identify intervals where event frequency becomes
significantly more intense than a baseline rate.

In this project, each “event” is an article timestamp belonging to a semantic
cluster.

Why daily and weekly:

- daily catches short coordinated spikes
- weekly reduces noise and checks whether the signal is stable across a coarser
  timeline

### More than just burst scores

The script also computes:

- timestamp coverage
- domain diversity
- top-domain dominance
- active-day ratio
- peak-to-baseline ratio
- concentration
- penalties for sparse, low-quality, single-domain behavior

Those are combined into `suspicion_score`.

That means the ranking is not simply:

- “highest burst wins”

It is closer to:

- “strong burst signal, with enough timestamp support, decent source quality,
  and not just one low-diversity domain dominating everything”

### Campaign-candidate scoring

[`scripts/TemporalAnalysis.py`](../scripts/TemporalAnalysis.py) now also calls
[`src/campaign_scoring.py`](../src/campaign_scoring.py) before writing
`data/temporal/cluster_temporal_stats.parquet`.

The broad `suspicion_score` is intentionally generous: it can surface organic
bursty events, recurring themes, and noisy cases worth inspecting. The stricter
`campaign_candidate_score` is report-facing. It starts from `suspicion_score`
and applies additional weights for:

- article support
- recurring daily burst periods
- active-day support
- source diversity
- compact temporal span
- public-affairs narrative signal
- obvious organic-event title filtering

Use [`scripts/apply_campaign_candidate_scoring.py`](../scripts/apply_campaign_candidate_scoring.py)
only when an existing temporal parquet needs to be refreshed after scoring logic
changes. A normal fresh temporal run already includes those columns.

## Stage 5: Evaluation

Code:

- [`scripts/Evaluation.py`](../scripts/Evaluation.py)

This stage summarizes the quality of the discovered clusters.

It does three main things:

1. Reads the HDBSCAN config sweep and surfaces the best config per topic.
2. Computes intra-cluster cosine similarity using the saved embedding cache.
3. Joins in temporal burst information for a combined evaluation report.

### Why intra-cluster cosine matters

If a cluster is good semantically, points inside that cluster should be fairly
close to each other in embedding space.

This stage therefore computes the mean pairwise cosine similarity within each
real cluster. That gives an additional quality signal beyond the HDBSCAN sweep.

## Stage 6: TF-IDF Baseline and Ablation

Code:

- [`scripts/TFIDFBaseline.py`](../scripts/TFIDFBaseline.py)

This script is used to answer:

- “How much better is the main semantic pipeline than a simpler baseline?”

It compares:

1. TF-IDF + KMeans
2. SBERT + KMeans
3. SBERT + HDBSCAN
4. SBERT + HDBSCAN with burst scoring enabled vs disabled

### What TF-IDF is

TF-IDF is a classical sparse text representation.

Instead of learning deep semantic embeddings, it represents a document using
weighted word and n-gram frequencies. It is fast and interpretable, but it
usually captures meaning less robustly than SBERT.

### Why KMeans is included

KMeans is a standard clustering baseline:

- simple
- fast
- requires choosing `k`
- assumes more spherical cluster shapes

That gives you two useful comparisons:

- representation comparison: TF-IDF vs SBERT
- clustering-method comparison: KMeans vs HDBSCAN

## Stage 7: Dashboard Asset Builder

Code:

- [`scripts/PrepareDashboardData.py`](../scripts/PrepareDashboardData.py)

This script prepares fast dashboard inputs from the heavier pipeline artefacts.

### Why it exists

The full clustering parquet is too large and too detailed to make Streamlit
pleasant if the dashboard has to rebuild everything on startup.

So this script precomputes:

- cluster overview table
- topic summary table
- per-cluster daily counts
- article detail parquet
- sampled scatter parquet
- semantic neighbor parquet from FAISS KNN edges
- cluster-similarity parquet for the dashboard similarity map
- copied temporal/config/evaluation/ablation assets
- copied runtime observability asset
- one global dashboard metadata file

### Why there is a scatter sample

The dashboard does not need to plot every article point to be useful.

Instead, it keeps:

- anchor examples per cluster
- edge examples per cluster
- a bounded sample of noise points

This keeps the scatter responsive while the metric tables still use full-data
aggregations.

## Stage 8: Streamlit Dashboard

Code:

- [`scripts/Dashboard.py`](../scripts/Dashboard.py)

The dashboard is a read-only exploration layer over the prepared parquet assets.

Pages:

- Cluster Explorer
- Timeline and Bursts
- Top Campaigns
- Similarity Map
- Topic Health
- Evaluation and Ablation

Main design choice:

- expensive work is moved out of the dashboard and into
  [`scripts/PrepareDashboardData.py`](../scripts/PrepareDashboardData.py)

That keeps the UI responsive and startup time manageable.

## Stage 9: Report Figures

Code and docs:

- [`scripts/report_figures/run_all.py`](../scripts/report_figures/run_all.py)
- [`scripts/report_figures/_common.py`](../scripts/report_figures/_common.py)
- [`report/figures/FIGURES.md`](../report/figures/FIGURES.md)
- [`report/figures/TOP_5_COMPACT_CAMPAIGN_CANDIDATES.md`](../report/figures/TOP_5_COMPACT_CAMPAIGN_CANDIDATES.md)

The report figure scripts read existing parquet artefacts and write static PNGs
to `report/figures/`. They do not rerun data curation, embeddings, clustering,
or temporal analysis.

Run all figures with:

```bash
python scripts/report_figures/run_all.py
```

Run one figure with its matching script, for example:

```bash
python scripts/report_figures/fig_A1_topic_distribution.py
```

## Hardware / Performance Layer

Code:

- [`src/runtime_profile.py`](../src/runtime_profile.py)

The project has a shared runtime-profile layer that detects hardware and picks
reasonable defaults.

### What it auto-detects

- `cuda`, `mps`, `dml`, or `cpu`
- GPU name and memory when available
- CPU thread count
- embedding batch size
- parquet chunk size

### What currently uses GPU

Only Sentence-BERT embedding inference is GPU-accelerated in the main pipeline.

That means:

- [`scripts/EmbeddingsClustering.py`](../scripts/EmbeddingsClustering.py) can run SBERT on CUDA, MPS, or DirectML when the local PyTorch runtime supports it
- FAISS, UMAP, HDBSCAN, temporal analysis, TF-IDF, evaluation, dashboard prep,
  and report figure generation remain CPU-based

### Why this split is intentional

The embedding stage is the clearest high-value GPU target and the safest to
accelerate without changing model behavior or adding major platform complexity.

## Important Design Choices

### 1. Parquet-first tabular pipeline

The code used to rely heavily on CSV. It now uses parquet for normal execution
because parquet is:

- faster to read
- faster to write
- more compact
- better for repeated downstream loading

Dense vectors and similarity-search assets are kept in their native formats:
`.npy` for embeddings, `.faiss` plus `.json` metadata for FAISS indexes, and
parquet for KNN edges.

### 2. Topic-first clustering

The corpus is not clustered globally in one shot. It is split by normalized
topic bucket first, then clustered topic by topic.

Why:

- it reduces computational cost
- it avoids forcing unrelated topics into the same neighborhood structure
- it lets each topic use its own HDBSCAN config sweep

### 3. Noise is allowed

Not every article gets forced into a cluster. This is important. In real news
data, many items are off-topic, one-off, mixed, or simply not dense enough to
justify a reliable cluster.

### 4. Temporal ranking is multi-factor

The final suspiciousness ranking is not just “largest cluster” or “largest
burst”. It is a combination of:

- semantic grouping
- burst behavior
- timestamp reliability
- domain diversity
- penalties for weak evidence
- report-facing campaign-candidate filters when ranking compact case studies

## Common Questions

### Why use `short_document` for SBERT instead of the full article?

Because most of the semantic identity of a news article is present in the title
and early body. Using a shorter input lowers runtime and memory cost.

### Why not use TF-IDF as the main method?

TF-IDF is fast, but it is lexical rather than deeply semantic. It struggles more
with paraphrases, wording variation, and semantically similar stories phrased in
different ways.

### Why HDBSCAN instead of KMeans as the main clustering algorithm?

Because the data is messy. HDBSCAN:

- does not require a fixed global `k`
- can mark outliers as noise
- handles uneven cluster densities better

### Why keep `.npy` embeddings?

Because they are reused by multiple later stages and are faster to reload than
recomputing SBERT embeddings every time.

## Short Glossary

- SBERT: Sentence-BERT. A sentence/document embedding model that maps text into
  dense vectors that can be compared with cosine similarity.
- UMAP: A nonlinear dimensionality-reduction method used here to compress
  embedding vectors before clustering and for 2-D visualization coordinates.
- HDBSCAN: A density-based clustering algorithm that can discover clusters of
  different densities and label outliers as noise.
- FAISS: A fast similarity-search library used here for persisted per-topic
  indexes and dashboard nearest-neighbor evidence.
- TF-IDF: A sparse bag-of-words representation based on word importance in a
  corpus.
- KMeans: A centroid-based clustering baseline.
- Burst detection: A temporal method that detects time intervals with unusually
  high event rate.
