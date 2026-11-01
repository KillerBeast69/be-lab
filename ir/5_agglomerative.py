from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Load Iris dataset (150 flowers, 4 measurements each: sepal length, sepal width, petal length, petal width)
X, y_true = load_iris(return_X_y=True)  # X=measurements, y_true=known labels (0,1,2)

# Build the clustering model:
# n_clusters=3  → stop merging when 3 groups remain (we know there are 3 species)
# linkage='ward' → merge the pair whose combination increases internal variance the least
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X)  # trains on all data and returns cluster label for each flower

# Print cluster sizes and how well they match known labels (ARI=1 is perfect)
print("Cluster sizes:", X.shape[0])
print("Adjusted Rand Index:", round(adjusted_rand_score(y_true, labels), 3))

# Scatter plot: x=petal length, y=petal width (these separate species best)
plt.scatter(X[:, 2], X[:, 3], c=labels, cmap='viridis', s=40)
plt.xlabel('Petal length'); plt.ylabel('Petal width')
plt.title('Agglomerative Clustering on Iris')
plt.show()

# Dendrogram: tree diagram of every merge step (hallmark of hierarchical clustering)
Z = linkage(X, method='ward')  # recompute all merges from scratch (not stopped at 3)
plt.figure()
dendrogram(Z, truncate_mode='lastp', p=10)  # show last 10 branches for readability
plt.title('Dendrogram')
plt.show()