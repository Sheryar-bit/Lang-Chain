## Clustering Example (K-Means)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#  Generate sample data (age vs income)
X = np.array([
    [25, 40000], [27, 48000], [23, 35000], [45, 80000],
    [40, 75000], [50, 90000], [60, 95000]
])

# Applying KMeans (let's find 2 clusters)
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

#  Get results
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Customers')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centers')
plt.xlabel("Age")
plt.ylabel("Income")
plt.legend()
plt.show()

print("Cluster Centers:\n", centers)


## Dimensionality Reduction Example (PCA)

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load Iris dataset (4 features)
data = load_iris()
X = data.data

# Reduce to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Visualize
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=data.target, cmap='viridis')
plt.title("PCA - Dimensionality Reduction")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
