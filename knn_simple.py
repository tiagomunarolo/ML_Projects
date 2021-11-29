import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

NUM_CLUSTERS = 3

# Set seaborn as default
sns.set()
# Clear canvas
plt.clf()

# Create 2 subplots in current figure
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
# Create 3 clusters
X, _ = make_blobs(n_samples=5000, centers=[[-5, -5], [0, 0], [5, 5]], cluster_std=0.9)

# Plot X
sns.scatterplot(X[:, 0], X[:, 1], ax=axes[0])

# Use K-means to algorithm to identify clusters
k_means = KMeans(n_clusters=NUM_CLUSTERS)
k_means.fit(X=X)
labels = k_means.labels_

for cluster_label, color in zip(range(0, NUM_CLUSTERS), ['g', 'r', 'y']):
    x_cluster = X[labels == cluster_label]
    sns.scatterplot(x_cluster[:, 0], x_cluster[:, 1], color=color, ax=axes[1])

plt.show(block=True)
