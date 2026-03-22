import os
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss
import hdbscan
import umap

from sklearn.metrics import silhouette_score

# Instalare:
# python -m pip install sentence-transformers faiss-cpu hdbscan umap-learn scikit-learn

# Folder curent
base_dir = os.path.dirname(os.path.abspath(__file__))

# Fisier curatat
input_path = os.path.join(base_dir, "rolargesum_train_clean.csv")

# Numar maxim de linii
MAX_ROWS = 15000

# Coloana folosita pentru embeddings
TEXT_COLUMN = "title"

# Topicurile sub acest prag merg direct in noise
MIN_TOPIC_SIZE = 200

# Citire date
df = pd.read_csv(input_path, nrows=MAX_ROWS)

print("Shape initial:", df.shape)
print("Coloane:", df.columns.tolist())
print(df[["title", "topics", "document"]].head(3))

# Curatare minima dupa citire
df["title"] = df["title"].fillna("").astype(str)
df["document"] = df["document"].fillna("").astype(str)
df["short_document"] = df["short_document"].fillna("").astype(str)
df["topics"] = df["topics"].fillna("").astype(str)

# Eliminare randuri fara titlu sau document
df = df[df["title"].str.strip() != ""].copy().reset_index(drop=True)
df = df[df["document"].str.strip() != ""].copy().reset_index(drop=True)

print("Shape dupa eliminarea documentelor/titlurilor goale:", df.shape)


# =========================
# 1. NORMALIZARE TOPICS
# =========================
def normalize_topic(topic: str) -> str:
    topic = str(topic).strip().lower()

    mapping = {
        "politic": "politica",
        "politică": "politica",
        "politica": "politica",
        "guvern": "politica",

        "externe": "international",
        "international": "international",
        "stiri-externe": "international",

        "economie": "economie",
        "economic": "economie",
        "financiar": "economie",
        "stiri-economice": "economie",
        "bani": "economie",

        "social": "social",
        "societate": "social",
        "stiri-sociale": "social",
        "viata": "social",

        "justitie": "justitie",
        "stiri-justitie": "justitie",

        "sanatate": "sanatate",
        "sport": "sport",
        "educatie": "educatie",

        "stiinta": "stiinta",
        "it-stiinta": "stiinta",
        "tehnologie": "stiinta",

        "diverse": "diverse",
        "stiri-diverse": "diverse",

        "cultura": "cultura",
        "cultură": "cultura",

        "razboi": "international"
    }

    return mapping.get(topic, topic if topic != "" else "necunoscut")


# Topic normalizat
df["topic_group"] = df["topics"].apply(normalize_topic)

print("\nDistributia topic_group:")
print(df["topic_group"].value_counts().head(20))


# =========================
# 2. MODEL EMBEDDINGS
# =========================
print(f"\nReprezentare folosita pentru embeddings: {TEXT_COLUMN}")
print("\nIncarc modelul SentenceTransformer...")

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# =========================
# 3. CONFIGURATII DE BAZA
# =========================
default_configs = [
    {"min_cluster_size": 8, "min_samples": 3},
    {"min_cluster_size": 10, "min_samples": 4},
    {"min_cluster_size": 12, "min_samples": 6},
    {"min_cluster_size": 15, "min_samples": 8},
]

# Rezultate pentru raport
all_results = []

# Bucati finale care vor fi unite
final_chunks = []

# Offset pentru clustere globale unice
global_cluster_offset = 0

# Topicuri suficient de mari
topic_counts = df["topic_group"].value_counts()
eligible_topics = topic_counts[topic_counts >= MIN_TOPIC_SIZE].index.tolist()

print("\nTopicuri eligibile pentru clusterizare:")
print(topic_counts[topic_counts >= MIN_TOPIC_SIZE])


# =========================
# 4. CLUSTERIZARE PE TOPIC
# =========================
for topic_name in eligible_topics:
    print("\n" + "=" * 60)
    print(f"TOPIC_GROUP: {topic_name}")
    print("=" * 60)

    # Subset pe topic
    df_topic = df[df["topic_group"] == topic_name].copy().reset_index(drop=True)

    print("Numar articole in topic:", len(df_topic))

    # Alegere text pentru embeddings
    documents = df_topic[TEXT_COLUMN].tolist()

    # Embeddings
    print("Generez embeddings...")
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("Embeddings shape:", embeddings.shape)

    # Salvare embeddings pe topic
    safe_topic = topic_name.replace("/", "_").replace(" ", "_")
    embeddings_path = os.path.join(
        base_dir,
        f"rolargesum_embeddings_{TEXT_COLUMN}_{safe_topic}_{len(df_topic)}.npy"
    )
    np.save(embeddings_path, embeddings)

    # FAISS
    print("Construiesc indexul FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print("Numar vectori indexati:", index.ntotal)

    # Vecini apropiati pentru verificare
    k = 5
    distances, indices = index.search(embeddings[:3], k)

    print("\nExemple vecini apropiati:")
    for i in range(min(3, len(df_topic))):
        print(f"\nDocument {i}:")
        print("Titlu:", str(df_topic.iloc[i]["title"]))
        for rank, idx in enumerate(indices[i]):
            print(
                f"  Vecin {rank}: idx={idx}, scor={distances[i][rank]:.4f}, titlu={str(df_topic.iloc[idx]['title'])}"
            )

    # UMAP
    print("\nReduc dimensionalitatea cu UMAP...")
    reducer = umap.UMAP(
        n_neighbors=30,
        n_components=15,
        metric="cosine",
        random_state=42
    )

    embeddings_reduced = reducer.fit_transform(embeddings)
    print("Shape dupa UMAP:", embeddings_reduced.shape)

    # Salvare embeddings reduse
    reduced_path = os.path.join(
        base_dir,
        f"rolargesum_umap_{TEXT_COLUMN}_{safe_topic}_{len(df_topic)}.npy"
    )
    np.save(reduced_path, embeddings_reduced)

    # Configuratii specifice pe topic
    topic_specific_configs = default_configs

    # Pentru politica, fortez variante mai stricte
    if topic_name == "politica":
        topic_specific_configs = [
            {"min_cluster_size": 10, "min_samples": 4},
            {"min_cluster_size": 12, "min_samples": 6},
            {"min_cluster_size": 15, "min_samples": 8},
        ]

    best_result = None

    print("\nTestez configuratii HDBSCAN...")
    for cfg in topic_specific_configs:
        print(f"\nConfig testata: {cfg}")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg["min_cluster_size"],
            min_samples=cfg["min_samples"],
            metric="euclidean",
            prediction_data=True
        )

        labels = clusterer.fit_predict(embeddings_reduced)

        # Numar clustere reale
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Numar noise
        num_noise = int((labels == -1).sum())
        noise_percent = 100 * num_noise / len(labels)

        # Cel mai mare cluster real
        real_cluster_counts = pd.Series(labels[labels != -1]).value_counts()
        largest_real_cluster = int(real_cluster_counts.iloc[0]) if not real_cluster_counts.empty else 0

        # Silhouette doar pe non-noise
        sil_score = None
        mask = labels != -1
        if mask.sum() > 1 and len(set(labels[mask])) > 1:
            try:
                sil_score = silhouette_score(
                    embeddings_reduced[mask],
                    labels[mask],
                    metric="euclidean"
                )
            except Exception:
                sil_score = None

        print("Numar clustere:", num_clusters)
        print("Noise:", num_noise, f"({noise_percent:.2f}%)")
        print("Cel mai mare cluster real:", largest_real_cluster)
        print("Silhouette:", sil_score)

        # Penalizare pentru cluster dominant
        penalty = largest_real_cluster / len(labels)

        # Bonus pentru silhouette
        sil_bonus = sil_score * 20 if sil_score is not None else 0

        # Bonus mic pentru numar bun de clustere
        cluster_bonus = min(num_clusters, 100) / 10

        # Pentru topicuri mici, pun mai mult accent pe coeziune
        if len(df_topic) < 1000:
            selection_score = (100 - noise_percent) - (penalty * 120) + (sil_bonus * 1.5) + cluster_bonus
        else:
            selection_score = (100 - noise_percent) - (penalty * 150) + sil_bonus + cluster_bonus

        # Prea putine clustere = penalizare
        if num_clusters < 3:
            selection_score -= 100

        result = {
            "cfg": cfg,
            "labels": labels,
            "num_clusters": num_clusters,
            "num_noise": num_noise,
            "noise_percent": noise_percent,
            "largest_real_cluster": largest_real_cluster,
            "silhouette": sil_score,
            "selection_score": selection_score,
        }

        # Salvare pentru raport
        all_results.append({
            "topic_group": topic_name,
            "topic_size": len(df_topic),
            "min_cluster_size": cfg["min_cluster_size"],
            "min_samples": cfg["min_samples"],
            "num_clusters": num_clusters,
            "num_noise": num_noise,
            "noise_percent": noise_percent,
            "largest_real_cluster": largest_real_cluster,
            "silhouette": sil_score,
            "selection_score": selection_score
        })

        # Pastram configuratia cea mai buna
        if best_result is None or result["selection_score"] > best_result["selection_score"]:
            best_result = result

    print("\nConfiguratia aleasa pentru topic:", topic_name)
    print(best_result["cfg"])
    print("Numar clustere:", best_result["num_clusters"])
    print("Numar puncte noise:", best_result["num_noise"])
    print("Procent noise:", round(best_result["noise_percent"], 2), "%")
    print("Cel mai mare cluster real:", best_result["largest_real_cluster"])
    print("Silhouette:", best_result["silhouette"])

    # Salvam parametrii alesi
    df_topic["best_min_cluster_size"] = best_result["cfg"]["min_cluster_size"]
    df_topic["best_min_samples"] = best_result["cfg"]["min_samples"]

    # Labeluri locale
    local_labels = best_result["labels"]

    # Transformare in labeluri globale unice
    global_labels = []
    for lbl in local_labels:
        if lbl == -1:
            global_labels.append(-1)
        else:
            global_labels.append(lbl + global_cluster_offset)

    df_topic["cluster"] = global_labels

    # Crestere offset pentru topicul urmator
    real_local_clusters = len(set(local_labels)) - (1 if -1 in local_labels else 0)
    if real_local_clusters > 0:
        global_cluster_offset += real_local_clusters

    # Adaugam rezultatul topicului
    final_chunks.append(df_topic)


# Topicurile prea mici merg in noise
small_topics_df = df[~df["topic_group"].isin(eligible_topics)].copy()
if not small_topics_df.empty:
    small_topics_df["best_min_cluster_size"] = np.nan
    small_topics_df["best_min_samples"] = np.nan
    small_topics_df["cluster"] = -1
    final_chunks.append(small_topics_df)


# =========================
# 5. REZULTAT FINAL GLOBAL
# =========================
final_df = pd.concat(final_chunks, ignore_index=True)

# Marime cluster pentru fiecare rand
cluster_sizes = final_df["cluster"].value_counts().to_dict()
final_df["cluster_size"] = final_df["cluster"].map(cluster_sizes)

print("\n" + "=" * 60)
print("REZULTAT FINAL GLOBAL")
print("=" * 60)

# Statistici globale
num_clusters_global = len(set(final_df["cluster"])) - (1 if -1 in final_df["cluster"].values else 0)
num_noise_global = int((final_df["cluster"] == -1).sum())
noise_percent_global = 100 * num_noise_global / len(final_df)

print("Numar clustere globale:", num_clusters_global)
print("Numar puncte noise globale:", num_noise_global)
print("Procent noise global:", round(noise_percent_global, 2), "%")

print("\nDistributia clusterelor globale:")
print(final_df["cluster"].value_counts().head(20))

# Top clustere reale
top_real_clusters = final_df[final_df["cluster"] != -1]["cluster"].value_counts().head(10)
print("\nTop clustere reale globale:")
print(top_real_clusters)

# Inspectare cluster global mare
cluster_counts = final_df[final_df["cluster"] != -1]["cluster"].value_counts()
if not cluster_counts.empty:
    largest_cluster = cluster_counts.idxmax()
    print("\nCel mai mare cluster real global este:", largest_cluster)

    subset_big = final_df[final_df["cluster"] == largest_cluster].head(20)
    for _, row in subset_big.iterrows():
        print("-", str(row["title"]))
else:
    print("\nNu exista clustere reale diferite de -1.")

# Exemple din primele clustere globale
print("\nExemple din primele clustere globale:")
shown = 0
for cluster_id in sorted(final_df["cluster"].unique()):
    if cluster_id == -1:
        continue

    subset = final_df[final_df["cluster"] == cluster_id].head(5)
    print(f"\n=== Cluster global {cluster_id} ===")
    for _, row in subset.iterrows():
        print("-", str(row["title"]))

    shown += 1
    if shown >= 5:
        break


# =========================
# 6. SALVARE REZULTATE
# =========================
results_df = pd.DataFrame(all_results)
results_path = os.path.join(base_dir, f"hdbscan_results_by_topic_{TEXT_COLUMN}_{MAX_ROWS}.csv")
results_df.to_csv(results_path, index=False)
print("\nRezultatele configuratiilor salvate la:", results_path)

output_path = os.path.join(base_dir, f"rolargesum_with_clusters_by_topic_{TEXT_COLUMN}_{MAX_ROWS}.csv")
final_df.to_csv(output_path, index=False)

print("Fisier salvat la:", output_path)
print("Shape final:", final_df.shape)