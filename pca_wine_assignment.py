"""
====================================================================
ASSIGNMENT: PCA for Dimensionality Reduction on Wine Dataset
====================================================================

1. ASSIGNMENT DETAILS
---------------------
Objective: Apply the Principal Component Analysis (PCA) algorithm to the 
Wine dataset to reduce its dimensionality. The goal is to capture most 
of the variations in the measurements (alcohol, ash, magnesium, etc.) 
using a small number of principal components. This makes it easier to 
visualize the data and distinguish between different wine classes.
Dataset Link: https://media.geeksforgeeks.org/wp-content/uploads/Wine.csv

2. VIVA THEORY
--------------
- What is PCA? 
  Principal Component Analysis is an unsupervised linear dimensionality 
  reduction technique. It transforms a large set of correlated variables 
  into a smaller number of uncorrelated variables called Principal Components.
- What are Principal Components (PCs)?
  New orthogonal axes that capture the maximum variance in the data. PC1 
  captures the most variance, PC2 captures the second most, and so on.
- Why Standardize? 
  PCA is highly sensitive to the scale of features. Since measurements 
  like 'magnesium' and 'alcohol' have completely different scales, we 
  must standardize them (mean=0, variance=1) so they contribute equally.
- What are Eigenvalues and Eigenvectors?
  Eigenvectors determine the direction of the new feature space (the PCs), 
  while Eigenvalues determine the magnitude (variance) captured by each PC.

3. CODE DETAILS
---------------
- pandas (pd): Used to load the CSV dataset from the URL.
- StandardScaler: Scales the features so they have a mean of 0 and std of 1.
- PCA(n_components=2): Initializes the PCA algorithm to reduce the 13 
  original features down to just 2 principal components.
- fit_transform: Computes the principal components and applies the 
  transformation to the dataset.
- matplotlib.pyplot (plt): Used to create a 2D scatter plot of the new 
  components, making it easy to visually distinguish the wine classes.

4. IMPLEMENTATION & EXECUTION
-----------------------------
Prerequisites (Necessary Packages):
- pandas
- scikit-learn
- matplotlib

How to run:
1. Open your terminal or command prompt.
2. Install the required packages by running:
   pip install pandas scikit-learn matplotlib
3. Save this code in a file named `pca_wine.py`.
4. Execute the script by running:
   python pca_wine.py
====================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    # 1. Load the dataset
    url = "https://media.geeksforgeeks.org/wp-content/uploads/Wine.csv"
    print(f"Loading dataset from {url}...")
    df = pd.read_csv(url)

    # 2. Separate features (X) and target class (y)
    # Fallback applied in case the target column name differs
    if 'Customer_Segment' in df.columns:
        X = df.drop('Customer_Segment', axis=1)
        y = df['Customer_Segment']
    else:
        X = df.iloc[:, 1:] if df.columns[0].lower() in ['class', 'target'] else df.iloc[:, :-1]
        y = df.iloc[:, 0] if df.columns[0].lower() in ['class', 'target'] else df.iloc[:, -1]

    # 3. Standardize the features (Crucial step for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Apply PCA (reduce to 2 components for 2D visualization)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Calculate variance explained by these 2 components
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    print(f"Variance captured by 2 principal components: {explained_variance:.2f}%")

    # 5. Visualize the result
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', alpha=0.7)
    
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('PCA of Wine Dataset (2D Projection)')
    plt.colorbar(scatter, label='Wine Class')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save or show the plot
    print("Plotting the results... Close the plot window to exit.")
    plt.show()

if __name__ == "__main__":
    main()
