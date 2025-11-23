# clustering_fast.py
# Rápido, amigável à RAM — amostra estratificada + MiniBatchKMeans + DBSCAN (PCA) + fallback SOM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CSV_PATH = "creditcard.csv"  # coloque aqui o caminho do seu csv
OUT_DIR = "clustering_out"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Carregar e amostrar estratificado (até 20k amostras)
df = pd.read_csv(CSV_PATH)
sample_n = min(20000, len(df))
df_sample, _ = train_test_split(df, train_size=sample_n, stratify=df["Class"], random_state=RANDOM_STATE)
print("Amostra shape:", df_sample.shape)

y = df_sample["Class"].values
X = df_sample.drop(columns=["Class"])

# downcast para economizar RAM
for col in X.columns:
    if pd.api.types.is_float_dtype(X[col]) or pd.api.types.is_integer_dtype(X[col]):
        X[col] = pd.to_numeric(X[col], downcast="float")

# normalizar Amount, padronizar demais
if "Amount" in X.columns:
    X["Amount"] = MinMaxScaler().fit_transform(X[["Amount"]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA para visualização (2) e para reduzir custo (10)
pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)
pca10 = PCA(n_components=min(10, X_scaled.shape[1]), random_state=RANDOM_STATE).fit_transform(X_scaled)

def compute_metrics(X_emb, labels, y_true):
    unique = set(labels) - {-1}
    n_clusters = len(unique)
    metrics = {"n_clusters": n_clusters}
    if n_clusters >= 2:
        try:
            metrics["silhouette"] = float(silhouette_score(X_emb, labels))
            metrics["calinski"] = float(calinski_harabasz_score(X_emb, labels))
            metrics["davies"] = float(davies_bouldin_score(X_emb, labels))
        except:
            metrics["silhouette"] = metrics["calinski"] = metrics["davies"] = float("nan")
    else:
        metrics["silhouette"] = metrics["calinski"] = metrics["davies"] = float("nan")
    # externas
    try:
        metrics["ARI"] = float(adjusted_rand_score(y_true, labels))
        metrics["NMI"] = float(normalized_mutual_info_score(y_true, labels))
    except:
        metrics["ARI"] = metrics["NMI"] = float("nan")
    return metrics

# KMeans rápido (MiniBatch) no PCA10
mbk = MiniBatchKMeans(n_clusters=2, random_state=RANDOM_STATE, batch_size=2000, n_init=5)
k_labels = mbk.fit_predict(pca10)
k_metrics = compute_metrics(pca10, k_labels, y)
print("KMeans:", k_metrics)
plt.figure(); plt.scatter(pca2[:,0], pca2[:,1], c=k_labels, s=5); plt.title("MiniBatchKMeans k=2"); plt.savefig(os.path.join(OUT_DIR,"kmeans_pca.png"), dpi=300); plt.close()

# DBSCAN: estimar eps via k-distance (subamostra)
ss = min(5000, pca10.shape[0])
idx = np.random.choice(len(pca10), ss, replace=False)
nbrs = NearestNeighbors(n_neighbors=10).fit(pca10[idx])
distances, _ = nbrs.kneighbors(pca10[idx])
k_dist = np.sort(distances[:, -1])
eps = float(np.percentile(k_dist, 98))
db = DBSCAN(eps=eps, min_samples=10, n_jobs=-1)
db_labels = db.fit_predict(pca10)
db_metrics = compute_metrics(pca10, db_labels, y)
print("DBSCAN (eps≈{:.4f}):".format(eps), db_metrics)
plt.figure(); plt.scatter(pca2[:,0], pca2[:,1], c=db_labels, s=5); plt.title(f"DBSCAN eps={eps:.3f}"); plt.savefig(os.path.join(OUT_DIR,"dbscan_pca.png"), dpi=300); plt.close()

# SOM (tentar MiniSom; se não tiver, fallback para KMeans sobre PCA10)
try:
    from minisom import MiniSom
    som_ok = True
except:
    som_ok = False

if som_ok:
    som = MiniSom(10,10,X_scaled.shape[1], sigma=1.0, learning_rate=0.5, random_seed=RANDOM_STATE)
    som.train_random(X_scaled, 1000)
    bmus = np.array([som.winner(x) for x in X_scaled])
    bmus_flat = np.array([u*10 + v for u,v in bmus])
    weights = som.get_weights().reshape(100, -1)
    som_k = MiniBatchKMeans(n_clusters=2, random_state=RANDOM_STATE, batch_size=500, n_init=5)
    wlabels = som_k.fit_predict(weights)
    som_labels = np.array([wlabels[i] for i in bmus_flat])
else:
    som_labels = MiniBatchKMeans(n_clusters=2, random_state=RANDOM_STATE, batch_size=2000, n_init=5).fit_predict(pca10)

som_metrics = compute_metrics(pca10, som_labels, y)
print("SOM (ou fallback):", som_metrics)
plt.figure(); plt.scatter(pca2[:,0], pca2[:,1], c=som_labels, s=5); plt.title("SOM -> KMeans 2 (fallback ok)"); plt.savefig(os.path.join(OUT_DIR,"som_pca.png"), dpi=300); plt.close()

# salvar resumo
summary = pd.DataFrame([
    {"Método":"MiniBatchKMeans (PCA10)", **k_metrics},
    {"Método":f"DBSCAN eps={eps:.3f}", **db_metrics},
    {"Método":"SOM 10x10 -> KMeans2", **som_metrics}
])
summary.to_csv(os.path.join(OUT_DIR,"clustering_summary.csv"), index=False)
print("Resumo salvo em", os.path.join(OUT_DIR,"clustering_summary.csv"))
print("Figuras em", OUT_DIR)
