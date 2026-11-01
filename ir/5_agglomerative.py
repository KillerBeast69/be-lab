# ---------------------------------------------------------------
# IMPORTS: pull in ready-made tools so we don't rewrite everything
# ---------------------------------------------------------------
from sklearn.datasets import load_iris        # gives us the Iris dataset
from sklearn.cluster import AgglomerativeClustering  # the algorithm itself
from sklearn.metrics import adjusted_rand_score      # grades our clustering
import matplotlib.pyplot as plt               # draws the graphs
from scipy.cluster.hierarchy import linkage, dendrogram  # builds the tree diagram
import numpy as np                            # basic number/array helpers

# ---------------------------------------------------------------
# STEP 1: LOAD THE DATASET
# ---------------------------------------------------------------
X, y = load_iris(return_X_y=True)
# X = the 150 flowers' measurements, shape (150, 4)
#     each row = [sepal length, sepal width, petal length, petal width]
# y = the TRUE species of each flower (0=setosa, 1=versicolor, 2=virginica)
#     we already know these answers - normally clustering has no answers,
#     but here we keep them ONLY to check how well we did

# ---------------------------------------------------------------
# STEP 2: RUN AGGLOMERATIVE CLUSTERING
# ---------------------------------------------------------------
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
# n_clusters=3 -> stop merging when 3 groups remain (we know there are 3 species)
# linkage='ward' -> rule for deciding which two clusters are "closest":
#                   pick the merge that increases internal messiness the least

labels = model.fit_predict(X)
# fit_predict does the whole job in one call:
#   1. treats all 150 flowers as 150 separate clusters
#   2. repeatedly merges the two closest clusters (using ward's rule)
#   3. stops at 3 clusters
# returns 'labels': one number (0, 1 or 2) per flower = which group it landed in

# ---------------------------------------------------------------
# STEP 3: CHECK HOW GOOD THE RESULT IS
# ---------------------------------------------------------------
print("Cluster sizes:", np.bincount(labels))
# counts how many flowers ended up in group 0, group 1, group 2

score = adjusted_rand_score(y, labels)
print("Adjusted Rand Index:", round(score, 3))
# compares our 'labels' against the true 'y'
# 1.0 = perfect grouping, 0 = no better than random guessing
# using all 4 features typically gives ~0.74 here

# ---------------------------------------------------------------
# STEP 4: PLOT THE CLUSTERS (picture of the result)
# ---------------------------------------------------------------
plt.scatter(X[:, 2], X[:, 3], c=labels, cmap='viridis', s=40)
# scatter = dot plot
# X[:, 2] -> column 2 of X = petal length (x-axis)
# X[:, 3] -> column 3 of X = petal width  (y-axis)
#            (petals chosen because they separate the species best)
# c=labels -> color each dot by its cluster number
# cmap='viridis' -> color scheme; s=40 -> dot size
plt.xlabel('Petal length'); plt.ylabel('Petal width')
plt.title('Agglomerative Clustering on Iris')
plt.show()   # pops open the graph window

# ---------------------------------------------------------------
# STEP 5: DRAW THE DENDROGRAM (the merge history tree)
# ---------------------------------------------------------------
Z = linkage(X, method='ward')
# recomputes ALL merges from start to finish (not stopped at 3 clusters)
# Z = record of every merge: who joined whom, and at what distance
dendrogram(Z, truncate_mode='lastp', p=10)
# draws the tree; 'truncate_mode/lastp/p=10' keeps it readable
# by showing only the last 10 branches instead of all 150 twigs
plt.title('Dendrogram')
plt.show()