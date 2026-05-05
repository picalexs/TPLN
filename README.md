# Detecting Coordinated Campaigns in Romanian News

This repository detects suspicious coordinated news campaigns by combining
semantic clustering, temporal burst analysis, campaign-candidate scoring, and a
parquet-backed Streamlit dashboard.

Start here:

- [Pipeline guide](docs/PIPELINE_GUIDE.md) - detailed architecture and
  stage-by-stage data flow.
- [Report figures guide](report/figures/FIGURES.md) - generated figures, source
  data, and figure scripts.
- [Top 5 compact campaign candidates](report/figures/TOP_5_COMPACT_CAMPAIGN_CANDIDATES.md) -
  report-facing examples ranked by campaign-candidate score.
- [Requirements](requirements.txt) - pinned Python dependencies.
- [Environment template](.env.example) - Hugging Face token placeholder for the
  first data download.

## Dashboard Preview

The Streamlit dashboard is a dark-mode evidence browser for moving from corpus
overview to cluster-level articles. It can inspect topic coverage, semantic
cluster structure, temporal bursts, ranked campaign candidates, similarity
links, and quality/runtime diagnostics.

<!--markdownlint-disable MD033-->
<table>
  <tr>
    <td><img src="docs/images/dashboard-cluster-explorer.png" alt="Cluster explorer with topic counts, timestamp coverage, and UMAP projection"></td>
    <td><img src="docs/images/dashboard-timeline-bursts.png" alt="Timeline and burst view with active publication days"></td>
  </tr>
  <tr>
    <td><img src="docs/images/dashboard-top-campaigns.png" alt="Top campaign candidates with score components and storyline controls"></td>
    <td><img src="docs/images/dashboard-similarity-map.png" alt="Cross-cluster similarity map and edge diagnostics"></td>
  </tr>
</table>
<!--markdownlint-enable MD033-->

Additional diagnostic views:
[topic health](docs/images/dashboard-topic-health.png),
[runtime observability](docs/images/dashboard-runtime-observability.png), and
[evaluation/config sweep](docs/images/dashboard-evaluation-config.png).

## What The Pipeline Does

```mermaid
flowchart TD
    raw["RoLargeSum dataset"]:::input

    subgraph prep["Prepare Corpus"]
        direction TB
        curate["Data curation"]:::prepStep
        timestamps["URL, text, htmldate timestamps"]:::prepStep
        clean["Clean parquet corpus"]:::artefact
    end

    subgraph semantic["Semantic Layer"]
        direction TB
        sbert["SBERT embeddings"]:::semanticStep
        hdbscan["UMAP + HDBSCAN clusters"]:::semanticStep
        faiss["FAISS neighbor index"]:::semanticArtefact
    end

    subgraph temporal["Temporal Layer"]
        direction TB
        bursts["Daily and weekly bursts"]:::temporalStep
        sources["Domain and source diversity"]:::temporalStep
        scores["Suspicion + campaign scores"]:::temporalArtefact
    end

    subgraph qa["Quality Checks"]
        direction TB
        eval["Cluster evaluation"]:::qaStep
        ablation["TF-IDF / SBERT ablations"]:::qaStep
        reports["Evaluation reports"]:::qaArtefact
    end

    subgraph publish["Publish"]
        direction TB
        assets["Dashboard parquet assets"]:::publishStep
        dashboard["Streamlit dashboard"]:::output
        figures["Report figures"]:::output
    end

    raw --> curate
    curate --> timestamps
    timestamps --> clean
    clean --> sbert
    sbert --> hdbscan
    hdbscan --> bursts
    bursts --> sources
    sources --> scores
    scores --> eval
    eval --> reports
    reports --> assets
    assets --> dashboard
    assets --> figures

    sbert -. semantic cache .-> faiss
    faiss -. neighbor evidence .-> assets
    sbert -. baseline input .-> ablation
    ablation --> reports
    hdbscan -. cluster labels .-> eval
    scores -. campaign metrics .-> assets
    scores -. candidate ranking .-> figures

    style prep fill:#172554,stroke:#60a5fa,stroke-width:1.5px,color:#ffffff
    style semantic fill:#312e81,stroke:#a5b4fc,stroke-width:1.5px,color:#ffffff
    style temporal fill:#451a03,stroke:#fbbf24,stroke-width:1.5px,color:#ffffff
    style qa fill:#052e16,stroke:#34d399,stroke-width:1.5px,color:#ffffff
    style publish fill:#0f172a,stroke:#94a3b8,stroke-width:1.5px,color:#ffffff

    classDef input fill:#020617,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    classDef prepStep fill:#1e3a8a,stroke:#93c5fd,stroke-width:1.4px,color:#ffffff
    classDef semanticStep fill:#3730a3,stroke:#c7d2fe,stroke-width:1.4px,color:#ffffff
    classDef temporalStep fill:#92400e,stroke:#fde68a,stroke-width:1.4px,color:#ffffff
    classDef qaStep fill:#065f46,stroke:#a7f3d0,stroke-width:1.4px,color:#ffffff
    classDef publishStep fill:#334155,stroke:#cbd5e1,stroke-width:1.4px,color:#ffffff
    classDef artefact fill:#1d4ed8,stroke:#bfdbfe,stroke-width:1.2px,color:#ffffff
    classDef semanticArtefact fill:#4338ca,stroke:#ddd6fe,stroke-width:1.2px,color:#ffffff
    classDef temporalArtefact fill:#b45309,stroke:#fde68a,stroke-width:1.2px,color:#ffffff
    classDef qaArtefact fill:#047857,stroke:#bbf7d0,stroke-width:1.2px,color:#ffffff
    classDef output fill:#020617,stroke:#f8fafc,stroke-width:1.8px,color:#ffffff
```

The project works in seven main stages:

1. [Data curation](scripts/DataCuration.py) downloads or loads RoLargeSum,
   cleans text, extracts timestamps from URL/text/htmldate, and writes cleaned
   parquet.
2. [Embeddings and clustering](scripts/EmbeddingsClustering.py) normalizes
   topics, creates SBERT embeddings, persists FAISS indexes and KNN edges, runs
   UMAP + HDBSCAN, and records runtime metrics.
3. [Temporal analysis](scripts/TemporalAnalysis.py) computes daily/weekly burst
   features, source/domain concentration, broad suspicion scores, and stricter
   campaign-candidate columns.
4. [Evaluation](scripts/Evaluation.py) combines HDBSCAN sweep results,
   intra-cluster cosine similarity, and temporal metrics.
5. [TF-IDF baseline](scripts/TFIDFBaseline.py) compares TF-IDF + KMeans, SBERT +
   KMeans, SBERT + HDBSCAN, and burst on/off ablations.
6. [Dashboard asset prep](scripts/PrepareDashboardData.py) builds fast parquet
   assets for Streamlit, including article details, timelines, neighbor
   evidence, cluster similarity, and runtime views.
7. [Dashboard](scripts/Dashboard.py) provides an interactive explorer for
   clusters, bursts, campaign candidates, similarity maps, topic health, and
   ablation results.

Shared code lives in [src/](src), including [configuration](src/config.py),
[paths](src/paths.py), [runtime profiling](src/runtime_profile.py),
[topic mapping](src/topic_mapping.py),
[date extraction](src/date_extraction.py), and
[campaign scoring](src/campaign_scoring.py).

## Setup

Use Python 3.11+ if possible. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the first run only, create `.env` from [.env.example](.env.example) and add
your Hugging Face token:

```text
HF_TOKEN=your_huggingface_token_here
```

The token is only needed when
[data/rolargesum_raw.parquet](data/rolargesum_raw.parquet) is absent and
[DataCuration.py](scripts/DataCuration.py) must download the dataset.

## Run The Full Pipeline

Run these commands from the repository root:

```powershell
python scripts/DataCuration.py
python scripts/EmbeddingsClustering.py
python scripts/TemporalAnalysis.py
python scripts/Evaluation.py
python scripts/TFIDFBaseline.py
python scripts/PrepareDashboardData.py
python -m streamlit run scripts/Dashboard.py
```

If you already have the parquet artefacts and only want to open the dashboard:

```powershell
python scripts/PrepareDashboardData.py
python -m streamlit run scripts/Dashboard.py
```

To regenerate report figures:

```powershell
python scripts/report_figures/run_all.py
```

## Useful Run Options

Small experiment run:

```powershell
python scripts/DataCuration.py --nrows 5000
python scripts/EmbeddingsClustering.py --nrows 5000
```

Force CPU or select a GPU backend for SBERT inference:

```powershell
python scripts/EmbeddingsClustering.py --device cpu
python scripts/EmbeddingsClustering.py --device cuda
python scripts/EmbeddingsClustering.py --device mps
python scripts/EmbeddingsClustering.py --device dml
```

Tune clustering/runtime helpers:

```powershell
python scripts/EmbeddingsClustering.py --cpu-threads 8 --embed-batch-size 128
python scripts/EmbeddingsClustering.py --disable-faiss
python scripts/EmbeddingsClustering.py --faiss-rebuild
python scripts/PrepareDashboardData.py --scatter-cap 25000 --noise-per-topic-cap 50
```

Optional post-hoc maintenance scripts:

```powershell
python scripts/apply_campaign_candidate_scoring.py --dry-run
python scripts/fix_multiyear_suspicion.py --dry-run
```

[TemporalAnalysis.py](scripts/TemporalAnalysis.py) already writes the current
campaign-candidate columns, so
[apply_campaign_candidate_scoring.py](scripts/apply_campaign_candidate_scoring.py)
is mainly useful when refreshing an existing temporal parquet after scoring
logic changes.

## Main Artefacts

Generated data is intentionally ignored by git. The important outputs are:

- [data/rolargesum_raw.parquet](data/rolargesum_raw.parquet) - cached Hugging
  Face dataset.
- [data/rolargesum_train_clean.parquet](data/rolargesum_train_clean.parquet) -
  cleaned text, timestamp fields, and document variants.
- [data/embeddings/](data/embeddings/) - per-topic SBERT `.npy` embedding
  caches.
- [data/faiss/](data/faiss/) - persisted per-topic FAISS indexes and metadata.
- [data/faiss/knn/](data/faiss/knn/) - per-topic nearest-neighbor edge parquet
  files.
- [data/clusters/clustered_data.parquet](data/clusters/clustered_data.parquet) -
  article-level cluster labels and UMAP coordinates.
- [data/clusters/hdbscan_config_results.parquet](data/clusters/hdbscan_config_results.parquet) -
  HDBSCAN sweep results.
- [data/clusters/runtime_observability.parquet](data/clusters/runtime_observability.parquet) -
  per-topic runtime breakdown.
- [data/temporal/cluster_temporal_stats.parquet](data/temporal/cluster_temporal_stats.parquet) -
  burst, suspicion, and campaign-candidate metrics.
- [data/evaluation_report.parquet](data/evaluation_report.parquet) -
  consolidated evaluation rows.
- [data/tfidf_ablation_report.parquet](data/tfidf_ablation_report.parquet) -
  baseline and ablation metrics.
- [data/dashboard/](data/dashboard/) - Streamlit-ready parquet assets.
- [report/figures/](report/figures/) - generated report PNGs and figure
  documentation.

Older CSV artefacts may exist locally from previous runs, but the current normal
pipeline is parquet-first.

## Optional NVIDIA GPU Support

The default dependency install may provide CPU-only PyTorch through
`sentence-transformers`. For NVIDIA acceleration, install a CUDA-enabled PyTorch
build that matches your system:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` prints `True`,
[EmbeddingsClustering.py](scripts/EmbeddingsClustering.py) will use CUDA
automatically with `--device auto`. Use the official selector if you need a
different build: <https://pytorch.org/get-started/locally/>

## Notes

- GPU acceleration is limited to SentenceTransformer embedding inference; UMAP,
  HDBSCAN, FAISS, temporal analysis, evaluation, ablations, dashboard prep, and
  figure generation are CPU-based.
- Runtime settings are auto-tuned by
  [src/runtime_profile.py](src/runtime_profile.py) and can be overridden with
  script flags.
- The dashboard expects prepared parquet assets by design, so run
  [PrepareDashboardData.py](scripts/PrepareDashboardData.py) after changing
  clustering, temporal, evaluation, or ablation outputs.
