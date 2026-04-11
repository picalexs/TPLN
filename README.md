# Detecting Coordinated Campaigns: Semantic Clustering and Temporal Analysis

This project detects suspicious coordinated news campaigns by combining:

1. semantic clustering with Sentence-BERT embeddings and HDBSCAN
2. temporal burst analysis with daily and weekly burst scoring
3. evaluation and ablation studies
4. an interactive Streamlit dashboard

Detailed architecture guide: [`docs/PIPELINE_GUIDE.md`](docs/PIPELINE_GUIDE.md)

## Pipeline

Run the scripts in this order:

```bash
python DataCuration.py
python EmbeddingsClustering.py
python TemporalAnalysis.py
python Evaluation.py
python PrepareDashboardData.py
streamlit run Dashboard.py
```

Optionally, run the TF-IDF baseline ablation, for comparison (note: this takes
some time to run):

```bash
python TFIDFBaseline.py
```

## Artefacts

The pipeline is parquet-only for normal execution.

- `data/rolargesum_raw.parquet`
- `data/rolargesum_train_clean.parquet`
- `data/embeddings/*.npy`
- `data/clusters/clustered_data.parquet`
- `data/clusters/hdbscan_config_results.parquet`
- `data/temporal/cluster_temporal_stats.parquet`
- `data/evaluation_report.parquet`
- `data/tfidf_ablation_report.parquet`
- `data/dashboard/*.parquet`

The GDELT downloader also writes parquet outputs in `data/gdelt/`.

## Runtime Behavior

- SentenceTransformer inference uses GPU automatically when the installed
  PyTorch runtime supports `cuda` or `mps`.
- UMAP, HDBSCAN, temporal analysis, TF-IDF, and dashboard preparation stay
  CPU-based.
- CPU threads, embedding batch size, and parquet chunk size are auto-tuned per
  machine.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional GPU support for embeddings on NVIDIA:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` prints `True`, the pipeline will use GPU
automatically for SentenceTransformer embeddings. If you need a different CUDA
build, use the official selector: <https://pytorch.org/get-started/locally/>

For the first `DataCuration.py` run, add your Hugging Face token to `.env`:

```text
HF_TOKEN=your_huggingface_token_here
```

## Main Components

- `DataCuration.py`: downloads or loads RoLargeSum, cleans text, extracts
  timestamps, writes cleaned parquet
- `EmbeddingsClustering.py`: creates or reuses embedding caches, clusters per
  topic, writes clustering outputs
- `TemporalAnalysis.py`: computes burst and temporal concentration features per
  cluster
- `Evaluation.py`: combines clustering, embedding, and temporal metrics into one
  evaluation report
- `TFIDFBaseline.py`: runs TF-IDF and SBERT ablation baselines
- `PrepareDashboardData.py`: builds fast parquet dashboard assets from the
  clustering outputs
- `Dashboard.py`: Streamlit explorer for clusters, timelines, campaigns, and
  evaluation

## Notes

- GPU acceleration in this repo is intentionally limited to SentenceTransformer
  inference.
- Saved embedding caches are kept because later scripts reuse them directly.
- FAISS indexes, persisted UMAP arrays, and CSV pipeline outputs are not
  generated anymore.
