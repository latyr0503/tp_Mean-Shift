import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import metrics

file_path = 'sales.csv'
names = []
data = []

with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            names = row
        else:
            # Ignore la première colonne (row[0]), convertit le reste en float
            data.append([float(val) for val in row[1:]])

X = np.array(data)

print("--- PARTIE 2 ---")
print(f"Forme de X : {X.shape}")
print(f"Nombre de magasins : {X.shape[0]}")
print(f"Nombre de variables : {X.shape[1]}")

# Partie 3 — Entraînement du modèle Mean Shift
quantile = 0.8
bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=len(X))
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("\n--- PARTIE 3 ---")
print(f"Quantile utilisé: {quantile}")
print(f"Nombre de clusters estimés: {n_clusters_}")
print("\nCentres des clusters :")
print(", ".join(names[1:]))
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i}: " + ", ".join([f"{val:.2f}" for val in center]))

# Partie 4 — Visualisation
cluster_centers_2d = cluster_centers[:, 1:3]

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], 
            marker='x', s=169, linewidths=3, color='red', zorder=10)
plt.title(f'Centers of 2D clusters (Tank top vs Halter top)')
plt.xlabel(names[2])
plt.ylabel(names[3])
plt.grid(True)
plt.savefig('clusters_2d.png')
print("\n--- PARTIE 4 ---")
print("Graphique sauvegardé sous 'clusters_2d.png'")

# Partie 6 — Pour aller plus loin (bonus)
silhouette = metrics.silhouette_score(X, labels)
print("\n--- PARTIE 6 ---")
print(f"Silhouette Score (quantile={quantile}) : {silhouette:.4f}")

print("\n--- Test des différents quantiles (Questions 4 et 11) ---")
quantiles_to_test = [0.3, 0.8, 0.95]
for q in quantiles_to_test:
    bw = estimate_bandwidth(X, quantile=q, n_samples=len(X))
    ms_q = MeanShift(bandwidth=bw, bin_seeding=True)
    ms_q.fit(X)
    n_clus = len(np.unique(ms_q.labels_))
    if 1 < n_clus < len(X):
        sil = metrics.silhouette_score(X, ms_q.labels_)
        print(f"Quantile: {q} | Clusters: {n_clus} | Silhouette Score: {sil:.4f}")
    else:
        print(f"Quantile: {q} | Clusters: {n_clus} | Silhouette Score: Non calculable")
