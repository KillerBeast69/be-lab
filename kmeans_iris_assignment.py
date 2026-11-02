"""
====================================================================
ASSIGNMENT: K-Means Clustering on Iris Dataset
====================================================================

1. ASSIGNMENT DETAILS
---------------------
Objective: Implement the K-Means clustering algorithm on the Iris.csv 
dataset. Determine the optimal number of clusters using the Elbow Method 
and visualize the resulting clusters.
Dataset Link: https://www.kaggle.com/datasets/uciml/iris

2. VIVA THEORY
--------------
- What is K-Means Clustering?
  It is an unsupervised machine learning algorithm that groups unlabeled 
  data into 'K' distinct clusters based on feature similarity.
- How does K-Means work?
  1. Initializes K random centroids.
  2. Assigns each data point to the closest centroid.
  3. Recalculates the centroids as the mean of all points in the cluster.
  4. Repeats steps 2 and 3 until convergence (centroids stop moving).
- What is the Elbow Method?
  A heuristic used to determine the optimal number of clusters (K). It 
  plots the WCSS against the number of clusters. The "elbow" of the curve 
  represents the point where adding more clusters yields diminishing returns.
- What is WCSS (Within-Cluster Sum of Squares) / Inertia?
  The sum of the squared distances between each data point and its 
  corresponding cluster centroid. We want to minimize this, but not overfit.
- Why is it Unsupervised?
  Because it doesn't use target labels (like the 'Species' column) during 
  training; it solely finds hidden patterns in the features.

3. CODE DETAILS
---------------
- pd.read_csv(): Loads the Iris.csv file.
- Data Preprocessing: Drops the 'Id' and 'Species' columns because K-Means 
  only works on numerical features and doesn't need labels.
- KMeans(n_clusters=i): Scikit-learn's clustering model. We loop this from 
  i=1 to 10 to calculate the inertia (WCSS) for the Elbow Method.
- kmeans.inertia_: Extracts the WCSS value for the fitted model.
- plt.plot(): Visualizes the Elbow curve.
- fit_predict(): Trains the final KMeans model with the optimal K (which 
  is 3 for the Iris dataset) and assigns a cluster label to each row.
- plt.scatter(): Plots the clustered data points using the first two 
  features (Sepal Length and Sepal Width) for 2D visualization.

4. IMPLEMENTATION & EXECUTION
-----------------------------
Prerequisites (Necessary Packages):
- pandas, numpy, scikit-learn, matplotlib, seaborn

How to run:
1. Download 'Iris.csv' from the Kaggle link and place it in the same 
   directory as this script.
2. Open your terminal or command prompt.
3. Install required packages:
   pip install pandas numpy scikit-learn matplotlib seaborn
4. Save this code in a file named `kmeans_iris.py`.
5. Execute the script:
   python kmeans_iris.py
====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import datasets
import os
import warnings
warnings.filterwarnings('ignore') # Suppress sklearn memory leak warning on Windows for KMeans

def main():
    print("1. Loading the Iris dataset...")
    # Attempt to load Iris.csv from local directory
    if os.path.exists('Iris.csv'):
        print("-> Found 'Iris.csv' locally.")
        df = pd.read_csv('Iris.csv')
        # Drop the 'Id' column if it exists (Kaggle version usually has this)
        if 'Id' in df.columns:
            df.drop('Id', axis=1, inplace=True)
        # Extract features (drop the target 'Species' column for unsupervised learning)
        X = df.drop('Species', axis=1).values if 'Species' in df.columns else df.values
    else:
        print("-> 'Iris.csv' not found locally. Loading standard Iris dataset from scikit-learn as fallback...")
        iris = datasets.load_iris()
        X = iris.data

    print("\n2. Finding optimal number of clusters using the Elbow Method...")
    wcss = [] # Within-Cluster Sum of Squares (Inertia)
    
    # Test K from 1 to 10
    for i in range(1, 11):
        # init='k-means++' ensures smart initialization of centroids to avoid local optima
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
    plt.title('The Elbow Method for Optimal K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('WCSS (Inertia)')
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('elbow_curve.png')
    print("-> Saved Elbow curve plot as 'elbow_curve.png'")
    
    print("\n3. Applying K-Means to the dataset...")
    # From the elbow curve, we typically observe the elbow at K=3 for Iris
    optimal_k = 3 
    print(f"-> Training final K-Means model with K={optimal_k} clusters.")
    
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    print("\n4. Visualizing the Clusters...")
    # Plotting the clusters for the first two features: Sepal Length (0) and Sepal Width (1)
    plt.figure(figsize=(10, 6))
    
    # Scatter plots for each cluster
    colors = ['red', 'blue', 'green']
    labels = ['Cluster 1 (Iris-setosa)', 'Cluster 2 (Iris-versicolor)', 'Cluster 3 (Iris-virginica)']
    for i in range(optimal_k):
        plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], 
                    s=100, c=colors[i], label=labels[i], edgecolors='k', alpha=0.7)

    # Plot the centroids
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                s=250, c='yellow', label='Centroids', edgecolors='k', marker='*')

    plt.title('K-Means Clustering of Iris Dataset (Sepal Features)')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('kmeans_clusters.png')
    print("-> Saved clusters visualization as 'kmeans_clusters.png'")
    
    print("\nExecution Complete. Please check the directory for generated image plots.")

if __name__ == "__main__":
    main()
