# Detecting Coordinated Campaigns: Semantic Clustering & Temporal Analysis

**Cotea Carla, Ginghina Mihai, Hriscu Cosmin, Picioroaga Alexandru, Smau
Robert**

> Project 5 — TPLN (Tehnici de Prelucrare a Limbajului Natural)

---

## Abstract

This project presents a robust pipeline for detecting coordinated information
campaigns within continuous text streams by combining semantic similarity with
temporal analysis. Utilizing the **RoLargeSum** dataset of Romanian news
articles, the pipeline:

1. **Cleans and preprocesses** the raw corpus with tokenization and stopword
   removal
2. **Generates multilingual Sentence-BERT embeddings** to capture deep semantic
   meaning
3. **Indexes vectors with FAISS** for rapid nearest-neighbor retrieval
4. **Clusters with HDBSCAN** to automatically identify semantic communities
5. **Applies Kleinberg burst detection** to identify unnatural temporal spikes
   per cluster
6. **Evaluates** cluster quality and runs ablation studies (TF-IDF vs SBERT,
   KMeans vs HDBSCAN)
7. **Visualizes** everything in an interactive Streamlit dashboard

By unifying semantic cohesion with temporal synchronization, this methodology
reliably isolates highly suspicious clusters, providing a scalable unsupervised
foundation for detecting coordinated inauthentic behavior.

---

## Pipeline Architecture

```text
RoLargeSum Dataset (HuggingFace)
        │
        ▼
┌─────────────────┐
│ DataCuration.py │  Download + Clean + Cache
│   (Cleaning)    │  → data/rolargesum_raw.parquet
│                 │  → data/rolargesum_train_clean.csv
└────────┬────────┘
         │
         ▼
┌──────────────────────────┐
│ EmbeddingsClustering.py  │  SBERT + FAISS + UMAP + HDBSCAN
│ (Embeddings & Clusters)  │  → data/embeddings/*.npy
│                          │  → data/faiss/*.faiss
│                          │  → data/umap/*.npy
│                          │  → data/clusters/clustered_data.csv
└────────┬─────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────────────┐  ┌─────────────────┐
│ TemporalAnaly- │  │ TFIDFBaseline.py│  Ablation
│ sis.py         │  │ (Ablation)      │  → data/tfidf_ablation_report.csv
│ Burst          │  └────────┬────────┘
│ → cluster_     │           │
│   temporal_    │           │
│   stats.csv    │           │
└────────┬───────┘           │
         │                   │
         ▼                   │
┌─────────────────┐          │
│ Evaluation.py   │◄─────────┘  Metrics
│ (Evaluation)    │  → data/evaluation_report.csv
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Dashboard.py   │  Streamlit + Plotly
│  (Dashboard)    │  → streamlit run Dashboard.py
└─────────────────┘
```

---

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/picalexs/TPLN
cd TPLN
pip install -r requirements.txt
```

### 2. HuggingFace Token

Create a `.env` file in the project root (see `.env.example`):

```text
HF_TOKEN=your_huggingface_token_here
```

> This token is only needed for the **first run** of `DataCuration.py`. After
> that, the dataset is cached locally.

### 3. Run the Pipeline

Run the entire pipeline in sequence. Each script saves intermediate results to
the `data/` directory, so you can run them independently after the first
execution.

```bash
python DataCuration.py
python EmbeddingsClustering.py
python TemporalAnalysis.py
python Evaluation.py
python TFIDFBaseline.py
streamlit run Dashboard.py
```

---

## Technologies

| Component                | Technology                                              |
| ------------------------ | ------------------------------------------------------- |
| Language                 | Python                                                  |
| Embeddings               | Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Vector search            | FAISS (`IndexFlatIP`)                                   |
| Clustering               | HDBSCAN (auto K, density-based)                         |
| Dimensionality reduction | UMAP                                                    |
| Burst detection          | Kleinberg automaton model                               |
| Baseline                 | TF-IDF + KMeans (ablation)                              |
| Dashboard                | Streamlit + Plotly                                      |
| Dataset                  | RoLargeSum (Romanian news)                              |

---

## Ablation Studies

Three ablation experiments as required by the project specification:

| #   | Comparison                         | Variable                            |
| --- | ---------------------------------- | ----------------------------------- |
| A   | TF-IDF + KMeans vs SBERT + HDBSCAN | Representation + clustering method  |
| B   | SBERT + KMeans vs SBERT + HDBSCAN  | Clustering method (same embeddings) |
| C   | With burst scoring vs without      | Impact of temporal analysis         |

Results are saved to `data/tfidf_ablation_report.csv` and visualized in the
dashboard's **Evaluation & Ablation** tab.
