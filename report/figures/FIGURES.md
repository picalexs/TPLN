# Report figures

23 figures generated from the existing parquet artefacts. Each figure has a
matching script in `scripts/report_figures/fig_<id>.py`. Static PNG at 300 DPI,
sized for a Word page. Pick whichever you want to include — the IDs (A1, B5, …)
are independent of report ordering.

Related documentation:

- [Project README](../../README.md)
- [Pipeline guide](../../docs/PIPELINE_GUIDE.md)
- [Top 5 compact campaign candidates](TOP_5_COMPACT_CAMPAIGN_CANDIDATES.md)
- [Figure-generation script folder](../../scripts/report_figures/)
- [Shared figure helpers](../../scripts/report_figures/_common.py)

To regenerate everything: `python scripts/report_figures/run_all.py`. To
regenerate one figure, run its matching script from the table below, for example
`python scripts/report_figures/fig_A1_topic_distribution.py`.

Generated figure index:

| ID | PNG | Script |
|---|---|---|
| A1 | [Topic distribution](A1_topic_distribution.png) | [fig_A1_topic_distribution.py](../../scripts/report_figures/fig_A1_topic_distribution.py) |
| A2 | [Timestamp source breakdown](A2_timestamp_source_breakdown.png) | [fig_A2_timestamp_source_breakdown.py](../../scripts/report_figures/fig_A2_timestamp_source_breakdown.py) |
| A3 | [Timestamp coverage](A3_timestamp_coverage.png) | [fig_A3_timestamp_coverage.py](../../scripts/report_figures/fig_A3_timestamp_coverage.py) |
| A4 | [Corpus timeline](A4_corpus_timeline.png) | [fig_A4_corpus_timeline.py](../../scripts/report_figures/fig_A4_corpus_timeline.py) |
| B5 | [Cluster size distribution](B5_cluster_size_distribution.png) | [fig_B5_cluster_size_distribution.py](../../scripts/report_figures/fig_B5_cluster_size_distribution.py) |
| B6 | [Noise rate per topic](B6_noise_rate_per_topic.png) | [fig_B6_noise_rate_per_topic.py](../../scripts/report_figures/fig_B6_noise_rate_per_topic.py) |
| B7 | [UMAP scatter](B7_umap_scatter.png) | [fig_B7_umap_scatter.py](../../scripts/report_figures/fig_B7_umap_scatter.py) |
| B8 | [Silhouette and intra-cosine](B8_silhouette_intracosine.png) | [fig_B8_silhouette_intracosine.py](../../scripts/report_figures/fig_B8_silhouette_intracosine.py) |
| C9 | [HDBSCAN sweep scatter](C9_hdbscan_sweep_scatter.png) | [fig_C9_hdbscan_sweep_scatter.py](../../scripts/report_figures/fig_C9_hdbscan_sweep_scatter.py) |
| C10 | [HDBSCAN silhouette heatmap](C10_hdbscan_silhouette_heatmap.png) | [fig_C10_hdbscan_silhouette_heatmap.py](../../scripts/report_figures/fig_C10_hdbscan_silhouette_heatmap.py) |
| D11 | [Suspicion distribution](D11_suspicion_distribution.png) | [fig_D11_suspicion_distribution.py](../../scripts/report_figures/fig_D11_suspicion_distribution.py) |
| D12 | [Top suspicious clusters](D12_top_suspicious_clusters.png) | [fig_D12_top_suspicious_clusters.py](../../scripts/report_figures/fig_D12_top_suspicious_clusters.py) |
| D13 | [Top cluster timelines](D13_top_cluster_timelines.png) | [fig_D13_top_cluster_timelines.py](../../scripts/report_figures/fig_D13_top_cluster_timelines.py) |
| D15 | [Suspicion components](D15_suspicion_components.png) | [fig_D15_suspicion_components.py](../../scripts/report_figures/fig_D15_suspicion_components.py) |
| D16 | [Candidate types](D16_candidate_types.png) | [fig_D16_candidate_types.py](../../scripts/report_figures/fig_D16_candidate_types.py) |
| E16 | [Domain entropy vs suspicion](E16_domain_entropy_vs_suspicion.png) | [fig_E16_domain_entropy_vs_suspicion.py](../../scripts/report_figures/fig_E16_domain_entropy_vs_suspicion.py) |
| E17 | [Single vs multi-source](E17_single_vs_multi_source.png) | [fig_E17_single_vs_multi_source.py](../../scripts/report_figures/fig_E17_single_vs_multi_source.py) |
| F18 | [Quality index by method](F18_quality_index_by_method.png) | [fig_F18_quality_index_by_method.py](../../scripts/report_figures/fig_F18_quality_index_by_method.py) |
| F19 | [Silhouette vs Davies-Bouldin](F19_silhouette_vs_db.png) | [fig_F19_silhouette_vs_db.py](../../scripts/report_figures/fig_F19_silhouette_vs_db.py) |
| F20 | [Runtime vs silhouette](F20_runtime_vs_silhouette.png) | [fig_F20_runtime_vs_silhouette.py](../../scripts/report_figures/fig_F20_runtime_vs_silhouette.py) |
| F21 | [Method win counts](F21_method_win_counts.png) | [fig_F21_method_win_counts.py](../../scripts/report_figures/fig_F21_method_win_counts.py) |
| G22 | [Runtime breakdown](G22_runtime_breakdown.png) | [fig_G22_runtime_breakdown.py](../../scripts/report_figures/fig_G22_runtime_breakdown.py) |
| G23 | [Topic size vs runtime](G23_topic_size_vs_runtime.png) | [fig_G23_topic_size_vs_runtime.py](../../scripts/report_figures/fig_G23_topic_size_vs_runtime.py) |

Conventions

- Topic labels are translated from Romanian to English (e.g. `politica` →
  "Politics").
- The Okabe-Ito colorblind-safe palette is used throughout.
- "Real cluster" means HDBSCAN cluster id ≥ 0 (i.e. excludes noise label `-1`).
- **Multi-source** clusters have `domain_count > 1` (articles sourced from more
  than one domain). Single-source clusters (`domain_count == 1`) are grayed out
  or scored as zero in the suspicion figures, because a single outlet republishing
  its own content is not a coordinated-campaign signal.

---

## A. Corpus overview

### A1 — Article distribution across topic groups

**Source:** `data/dashboard/topic_summary.parquet`. Horizontal bars (log scale)
of article count per normalized topic bucket. The corpus is heavily skewed toward
Social (89 629), International (60 587), and Politics (44 296), which together
make up ~70% of all articles. The four smallest topics (Science, Education,
Unknown, Diverse) each have under 10 000 articles. This size imbalance explains
why the HDBSCAN config sweep is restricted to the four largest topics — the
smaller ones do not provide enough data for a meaningful grid search.
**Use as:** the first orientation figure of the data section.

### A2 — Timestamp provenance by topic

**Source:** `data/dashboard/cluster_articles.parquet`. Stacked bar showing, per
topic, the share of articles whose timestamp came from URL parsing, htmldate,
article-text scanning, or is missing entirely. `htmldate` is the dominant source
for most topics (60–97%), while Science (79%) and Education (68%) rely heavily on
"missing" because their URLs tend to lack date slugs and their content is sparse.
The "Unknown" topic is unusual: 89% of its timestamps come from URL parsing,
reflecting a different site structure. Justifies the layered timestamp-extraction
design and the `timestamp_source` penalty in the suspicion score.
**Use as:** support for the "data quality" subsection.

### A3 — Timestamp coverage per topic

**Source:** `data/dashboard/topic_summary.parquet`. Fraction of articles in each
topic that ended up with a usable timestamp. Coverage ranges from only 20% for
Science to 99% for Unknown. The big topics (Social 91%, International 89%,
Politics 88%) are well-covered; the tail topics (Science 20%, Education 33%) are
significantly under-timestamped, which limits the temporal analysis for those
categories. Mean corpus-wide coverage is 71%.
**Use as:** alongside A2 in the data quality subsection.

### A4 — Corpus publication timeline (timestamped articles only)

**Source:** `data/dashboard/cluster_articles.parquet`. Monthly counts of
timestamped articles for the whole corpus. Shows the dataset spans many years
with non-uniform density — useful to motivate the topic-first clustering and to
caveat any temporal interpretation. **Use as:** dataset characterization.

---

## B. Clustering quality

### B5 — Cluster size distribution

**Source:** `data/dashboard/cluster_overview.parquet`. Log-log histogram of real
cluster sizes. There are 6 882 real clusters with a median of 20 articles and a
mean of 41. The distribution has a heavy right tail: the largest cluster contains
2 374 articles. The heavy right tail is typical of news corpora — a handful of
large, coherent stories (elections, major disasters) plus a long tail of small,
niche ones. **Use as:** characterizing the structure HDBSCAN found.

### B6 — Noise rate per topic

**Source:** `data/dashboard/topic_summary.parquet`. Fraction of each topic's
articles that HDBSCAN labeled as noise (`cluster = -1`) after noise reassignment.
Politics has the highest noise rate (66%) and Science the lowest (35%).
High noise in Politics and Justice reflects the diversity of political discourse —
many one-off articles that do not cluster with anything. The pipeline's
noise-reassignment step reduces raw HDBSCAN noise, so these are final
post-reassignment rates.
**Use as:** justifies HDBSCAN's "noise is allowed" design choice.

### B7 — UMAP projection for two illustrative topics

**Source:** `data/dashboard/scatter_sample.parquet`. 2-panel scatter for
Politics and International, using the pre-sampled UMAP coordinates (anchors +
edges + bounded noise). Top-10 clusters get distinct Okabe-Ito colors, others
are gray. Do not expect crisp blobs — the sample is deliberately small for
dashboard responsiveness — but the colored islands make it visually clear that
the embedding space has structure.
**Use as:** qualitative evidence that the embedding space has structure.

### B8 — Per-topic silhouette and per-cluster intra-cluster cosine

**Source:** `data/evaluation_report.parquet`. Two panels:

- Left: global silhouette per topic for the chosen HDBSCAN config. Values range
  from 0.016 (Politics) to 0.128 (Science/Education). These are modest but
  expected for short-text SBERT embeddings on heterogeneous news corpora.
- Right: distribution of mean intra-cluster cosine across real clusters. Even
  modest global silhouette produces tight individual clusters (high mean
  intra-cluster cosine), confirming that HDBSCAN is finding genuinely coherent
  story groups.

**Use as:** the headline cluster-quality figure.

---

## C. HDBSCAN hyperparameter sweep

### C9 — Selection score vs. noise rate per topic

**Source:** `data/clusters/hdbscan_config_results.parquet`. One panel per topic:
each dot is one (`min_cluster_size`, `min_samples`, `cluster_selection_method`)
configuration. Marker size encodes `min_cluster_size`, color encodes selection
method (eom/leaf), and the chosen best config is ringed in black. Shows there is
a genuine tradeoff between noise rate and the composite selection score — the
best config is never at a noise extreme.
**Use as:** evidence that the per-topic config sweep is doing real work.

### C10 — Silhouette heatmap over the sweep

**Source:** `data/clusters/hdbscan_config_results.parquet`. Per-topic heatmap of
silhouette score across `min_cluster_size` × `min_samples`. Smaller topics
(Education, Science) have only a handful of cells; the sweep grid is denser for
the four big topics. The heatmap makes it visually obvious which corner of the
grid HDBSCAN prefers and that the chosen config sits in or near the high-silhouette
region.
**Use as:** complement to C9 showing a different quality metric.

---

## D. Temporal / burst analysis (the headline finding)

### D11 — Suspicion score distribution

**Source:** `data/temporal/cluster_temporal_stats.parquet`. Histogram (symlog
y-axis) of `suspicion_score` for all clusters and for multi-source clusters only
(those with `domain_count > 1`). Of 6 813 real clusters, 1 643 receive a
non-zero suspicion score, and only 287 are single-source clusters (excluded from
the multi-source histogram). The distribution has a strong zero-mass spike
(~5 170 clusters score exactly zero) and a long right tail reaching ~22, which is
what the pipeline is designed to surface.
**Use as:** motivation for ranking by suspicion rather than treating it as a
binary label.

### D12 — Top 15 compact campaign candidates

**Source:** `data/temporal/cluster_temporal_stats.parquet`. Horizontal bars of
the highest-scoring clusters by `campaign_candidate_score`. This score starts
from `suspicion_score`, then applies stricter report-facing filters: enough
article support, recurring burst periods, enough active days, source diversity,
compact temporal span, and a public-affairs narrative signal. The span gate gives
full credit up to 180 days and reaches zero at one year, so multi-year semantic
themes cannot be presented as campaign episodes. Obvious organic event titles
(accidents, weather alerts, thefts, deaths, police briefs, and routine COVID
case/death/vaccination-logistics updates) are zeroed out. COVID policy,
restriction, certificate, and controversy narratives are still allowed.
**Use as:** the headline candidate figure, with the caveat that these are
ranked candidates rather than verified coordinated campaigns.

### D13 — Daily timelines of the top non-COVID compact campaign candidates

**Source:** `data/dashboard/cluster_daily_counts.parquet` and
`data/temporal/cluster_temporal_stats.parquet`. Three stacked panels showing the
daily article counts for the top-3 **non-COVID** clusters by
`campaign_candidate_score`. Each title shows both the campaign-candidate score
and the underlying suspicion score. The plot uses daily bars rather than a
filled line so sparse publication dates are not visually connected across long
gaps.
**Use as:** qualitative case study showing that the pipeline also surfaces
non-COVID compact narratives.

### D15 — Suspicion and campaign-candidate score breakdown

**Source:** `data/temporal/cluster_temporal_stats.parquet`. Two panels for the
top 12 clusters by `campaign_candidate_score`:

- Left bars: `raw burst score`, `after temporal gates`, broad
  `suspicion_score`, and final `campaign_candidate_score`.
- Right heatmap: the original temporal gates plus the campaign-candidate gates
  for article support, recurrence, active days, source diversity, compact span,
  and narrative signal.

This figure explains why the earlier top-ranked false positives disappeared:
small organic incidents and multi-year semantic themes can still have high burst
suspicion, but their campaign candidate score collapses when the
support/recurrence/span/narrative filters are applied. **Use as:** the concrete
explanation of the stricter headline ranking.

### D16 — Campaign candidate narrative types

**Source:** `data/temporal/cluster_temporal_stats.parquet`. Two-panel summary of
all compact campaign candidates grouped by heuristic narrative type. The left
panel counts candidates by type; the right panel sums `campaign_candidate_score`
by the same type and annotates the highest-scoring example in each bucket. Main
buckets include COVID/vaccination, Ukraine/Russia war, energy/economy,
governance/elections, justice/corruption, security/propaganda, travel/mobility,
public-health non-COVID, and an explicit "Other / mixed public narratives"
bucket for titles that do not match the simple keyword rules.
**Use as:** high-level characterization of what kinds of narratives the
candidate ranking surfaces, while being transparent that these type labels are
heuristic.

---

## E. Source / domain concentration

### E16 — Domain entropy vs. suspicion score

**Source:** `data/temporal/cluster_temporal_stats.parquet`. Scatter of
cluster-level Shannon entropy of source-domain distribution against suspicion
score. Single-source clusters (`domain_count == 1`, 287 clusters) are shown in
gray; multi-source clusters are colored. Mean domain entropy is 0.74 (std 0.22).
High entropy + high suspicion score is the most interesting quadrant — it means
many distinct outlets are publishing near-identical stories simultaneously, the
textbook coordinated-campaign signature.
**Use as:** support for the "multi-source filter is meaningful" claim.

### E17 — Single- vs. multi-source clusters per topic

**Source:** `data/temporal/cluster_temporal_stats.parquet`. Stacked bar showing
how many clusters in each topic are dominated by a single domain
(`domain_count == 1`). Single-source clusters range from 0% (Science) to 46%
(Unknown) and 36% (Diverse). The large political topics (Politics 0.6%,
International 0.5%) have very few single-source clusters, so the multi-source
filter has minimal effect there — all their high-scoring clusters are genuine
multi-outlet stories.
**Use as:** characterization of the cluster pool the temporal stage operates on.

---

## F. TF-IDF / SBERT × KMeans / HDBSCAN ablation

### F18 — Quality index by method, per topic

**Source:** `data/tfidf_ablation_report.parquet`. Grouped bars of normalized 0–1
`quality_index` for three methods per topic. `quality_index` is min-max
normalized within each topic: SBERT+KMeans consistently loses on both silhouette
(mean 0.041) and Davies-Bouldin, so it is pinned near 0 on every topic — that is
expected, not a bug. TF-IDF+KMeans (mean quality_index 0.643) and SBERT+HDBSCAN
(mean 0.615) trade wins across topics.
**Use as:** the headline ablation figure.

### F19 — Silhouette vs. Davies-Bouldin scatter

**Source:** `data/tfidf_ablation_report.parquet`. Each (method, topic) is one
point. Higher silhouette is better; lower DB is better, so the y-axis is inverted
and the upper-left is best. SBERT+HDBSCAN sits clearly to the upper-left on
Davies-Bouldin while TF-IDF+KMeans wins on raw silhouette (mean 0.088 vs 0.056).
This is the underlying tension F18 reflects: the two metrics reward different
things.
**Use as:** lets the reader judge how much each metric matters.

### F20 — Runtime vs. silhouette tradeoff

**Source:** `data/tfidf_ablation_report.parquet` and
`data/clusters/runtime_observability.parquet`. SBERT+HDBSCAN runtime is taken
from `runtime_observability` (UMAP + HDBSCAN sweep + label-apply seconds). Shows
the SBERT pipeline is one to two orders of magnitude more expensive than
TF-IDF+KMeans per topic for similar or slightly lower silhouette.
**Use as:** cost-quality discussion.

### F21 — Topic-level method win counts

**Source:** `data/tfidf_ablation_report.parquet`. Counts how many of the 12
topics each method wins on `quality_index`. Provides a single-number summary of
F18.
**Use as:** ablation "headline number" alongside F18.

---

## G. Runtime / scaling

### G22 — Per-topic clustering stage runtime breakdown

**Source:** `data/clusters/runtime_observability.parquet`. Stacked horizontal bar
of seconds spent in each stage per topic (embedding, FAISS sanity check, UMAP,
HDBSCAN sweep, label apply). Embedding is fast on GPU (< 0.2 s per topic). UMAP
and the HDBSCAN sweep dominate: Social takes 86 s for UMAP and 107 s for the
sweep; the four large topics (Social, Politics, International, Economy) account
for the vast majority of total runtime.
**Use as:** support for any "where does the time go?" discussion.

### G23 — Topic size vs. total clustering runtime

**Source:** `data/clusters/runtime_observability.parquet`. Log-log scatter with a
power-law fit. The slope (included in the legend automatically) characterizes the
scaling regime of the clustering stage — sub-linear if FAISS/UMAP dominate,
super-linear if HDBSCAN sweep iterations grow with topic size.
**Use as:** scaling claim.

---

## Notes on the data

- **The pipeline was not re-run.** Every figure reads existing parquet artefacts.
- **`quality_index` normalization quirk.** `quality_index` is min-max normalized
  per topic across methods. SBERT+KMeans is worst on every topic, so it is pinned
  to 0 — expected and not a bug.
- **Suspicion vs. campaign candidates.** `suspicion_score` is a broad burst score
  and can surface organic news events or multi-year semantic themes. Figures D12,
  D13, and D15 now use `campaign_candidate_score`, which adds support,
  recurrence, compact-span, source-diversity, and title-based organic-event
  filters for report case studies. COVID is not blanket-filtered; only routine
  health counters and logistics are treated as organic updates.
- **Multi-source vs. all-cluster suspicion.** Figures D11, E16, and E17 use a
  derived multi-source score: `suspicion_score` is kept as-is for clusters with
  `domain_count > 1` and treated as zero for single-domain clusters.
- **Removed pipeline stages.** `dedup_seconds` and `reassign_seconds` were
  present in an earlier pipeline version but are no longer written to
  `runtime_observability.parquet`. G22 shows only the five stages that remain:
  Embedding, FAISS, UMAP, HDBSCAN sweep, and Label apply.
