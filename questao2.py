import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.neighbors import NearestNeighbors
from minisom import MiniSom

# ===========================================
# CONFIGURAÇÕES
# ===========================================
DATA_PATH = "creditcard.csv"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ===========================================
# CARREGAR DADOS
# ===========================================
df = pd.read_csv(DATA_PATH)
print("Shape original:", df.shape)
print(df["Class"].value_counts())

# Separar rótulo (sem usar para clustering)
y_true = df["Class"].values

# Remover coluna Class
X = df.drop(columns=["Class"])

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA para visualização (2D)
pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca2 = pca2.fit_transform(X_scaled)

# PCA para DBSCAN (reduzir custo)
pca10 = PCA(n_components=10, random_state=RANDOM_STATE)
X_pca10 = pca10.fit_transform(X_scaled)

# ===========================================
# FUNÇÃO AUXILIAR DE MÉTRICAS
# ===========================================
def compute_metrics(X_emb, labels, y_true):
    unique = set(labels) - {-1}
    n_clusters = len(unique)
    metrics = {"n_clusters": n_clusters}

    if n_clusters >= 2:
        metrics["silhouette"] = silhouette_score(X_emb, labels)
        metrics["calinski"] = calinski_harabasz_score(X_emb, labels)
        metrics["davies"] = davies_bouldin_score(X_emb, labels)
    else:
        metrics["silhouette"] = np.nan
        metrics["calinski"] = np.nan
        metrics["davies"] = np.nan

    # métricas externas (comparando com Class)
    metrics["ARI"] = adjusted_rand_score(y_true, labels)
    metrics["NMI"] = normalized_mutual_info_score(y_true, labels)

    return metrics

# ===========================================
# 1) KMEANS (k=2)
# ===========================================
print("\n===== KMEANS =====")
kmeans = KMeans(n_clusters=2, n_init=20, random_state=RANDOM_STATE)
k_labels = kmeans.fit_predict(X_scaled)
k_metrics = compute_metrics(X_scaled, k_labels, y_true)
print(k_metrics)

plt.figure()
plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=k_labels, s=5, cmap="tab10")
plt.title("KMEANS (k=2)")
plt.savefig("kmeans_pca.png", dpi=300)
plt.close()

# ===========================================
# 2) DBSCAN (com amostragem + PCA)
# ===========================================
print("\n===== DBSCAN =====")

# Amostragem para estimar eps
sample_size = 20000
idx = np.random.choice(len(X_pca10), sample_size, replace=False)
X_sample = X_pca10[idx]

min_samples = 10
neigh = NearestNeighbors(n_neighbors=min_samples)
nbrs = neigh.fit(X_sample)
distances, _ = nbrs.kneighbors(X_sample)

k_distances = np.sort(distances[:, -1])

# (se quiser ver o gráfico, descomente)
# plt.plot(k_distances)
# plt.title("k-distance (sampled)")
# plt.show()

# Vamos escolher eps aproximado pelo percentil 98
eps_candidate = np.percentile(k_distances, 98)
print("eps escolhido aproximadamente:", eps_candidate)

# Rodar DBSCAN completo com PCA(10)
db = DBSCAN(eps=eps_candidate, min_samples=min_samples, n_jobs=-1)
db_labels = db.fit_predict(X_pca10)

db_metrics = compute_metrics(X_scaled, db_labels, y_true)
print(db_metrics)

plt.figure()
plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=db_labels, s=5, cmap="tab10")
plt.title(f"DBSCAN eps={eps_candidate:.3f}")
plt.savefig("dbscan_pca.png", dpi=300)
plt.close()

# ===========================================
# 3) SOM (10x10 -> KMeans(2))
# ===========================================
print("\n===== SOM =====")

som_x, som_y = 10, 10
som = MiniSom(som_x, som_y, X_scaled.shape[1],
              sigma=1.0, learning_rate=0.5,
              random_seed=RANDOM_STATE)

som.train_random(X_scaled, 5000)

# Best Matching Units
bmus = np.array([som.winner(x) for x in X_scaled])
bmus_flat = np.array([u * som_y + v for u, v in bmus])

# Clusterizar os pesos do SOM com k=2
weights = som.get_weights().reshape(som_x * som_y, -1)
som_kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=20)
weight_labels = som_kmeans.fit_predict(weights)

som_labels = np.array([weight_labels[i] for i in bmus_flat])

som_metrics = compute_metrics(X_scaled, som_labels, y_true)
print(som_metrics)

plt.figure()
plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=som_labels, s=5, cmap="tab10")
plt.title("SOM (10x10 -> KMEANS 2)")
plt.savefig("som_pca.png", dpi=300)
plt.close()

# ===========================================
# 4) SALVAR RESUMO
# ===========================================
summary = pd.DataFrame([
    {
        "Método": "KMEANS",
        **k_metrics
    },
    {
        "Método": f"DBSCAN eps={eps_candidate:.3f}",
        **db_metrics
    },
    {
        "Método": "SOM 10x10 -> KMEANS 2",
        **som_metrics
    }
])

summary.to_csv("clustering_summary.csv", index=False)
print("\nResumo salvo em clustering_summary.csv")
print(summary)
