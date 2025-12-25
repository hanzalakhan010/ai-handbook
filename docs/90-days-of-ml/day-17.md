---
id: day-17
title: 'Day 17: Gaussian Mixture Models (GMM) and PCA'
---

## Day 17: Gaussian Mixture Models (GMM) and PCA

Today, we'll explore a more flexible and probabilistic approach to clustering called **Gaussian Mixture Models (GMM)**. We'll also learn how to visualize high-dimensional clusters using **Principal Component Analysis (PCA)**.

### What are Gaussian Mixture Models (GMM)?

While K-Means performs "hard clustering" (assigning each data point to exactly one cluster), GMM performs "soft clustering". It's a probabilistic model that assumes the data points are generated from a mixture of a finite number of Gaussian distributions (bell curves).

Instead of just assigning a point to a cluster, GMM provides the **probability** that a data point belongs to each of the clusters. This can be very useful when clusters overlap or when data points could plausibly belong to multiple groups.

### The Workflow: GMM on the Wine Dataset

Let's apply GMM to the `load_wine` dataset and compare its performance to other clustering algorithms.

#### 1. Finding the Optimal Number of Clusters

First, we use the Elbow Method with K-Means on the scaled wine dataset to determine the optimal number of clusters. The plot clearly shows an elbow at **K=3**, which matches the actual number of wine classes in the dataset.

#### 2. Training and Comparing Clustering Models

Now, we train four different clustering models with `n_clusters=3` (or `n_components=3` for GMM) and compare their Silhouette Scores.

```python
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

# Load and scale data
data = load_wine()
X_scaled = StandardScaler().fit_transform(data.data)

# --- Train Models ---
# Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=2)
gmm_labels = gmm.fit_predict(X_scaled)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=3)
km_labels = kmeans.fit_predict(X_scaled)

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo.fit_predict(X_scaled)

# --- Evaluate ---
gmm_score = silhouette_score(X_scaled, gmm_labels)
km_score = silhouette_score(X_scaled, km_labels)
agglo_score = silhouette_score(X_scaled, agglo_labels)

print(f'GMM Silhouette Score: {gmm_score:.3f}')
print(f'K-Means Silhouette Score: {km_score:.3f}')
print(f'Agglomerative Silhouette Score: {agglo_score:.3f}')
```
The results show that all three models perform similarly on this dataset, with scores around 0.28.

### Visualizing High-Dimensional Clusters with PCA

The wine dataset has 13 features, so we can't just plot it on a 2D scatter plot. To visualize the clusters, we need to reduce the number of dimensions.

**Principal Component Analysis (PCA)** is a dimensionality reduction technique that transforms the data into a new coordinate system, where the first few coordinates (the "principal components") capture the most variance in the data.

By reducing our 13-dimensional data to 2 principal components, we can create a 2D scatter plot to visualize the clusters.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce dimensions to 2 using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agglo_labels, cmap='plasma')
plt.title('Clusters Visualized with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
```
This plot allows us to visually inspect the separation of the clusters found by our algorithm.

### Key Takeaways for Day 17

*   **Gaussian Mixture Models (GMM)** provide a "soft clustering" approach, giving probabilities of a data point belonging to each cluster.
*   **Principal Component Analysis (PCA)** is a powerful technique for dimensionality reduction, which is essential for visualizing high-dimensional data.
*   Comparing multiple clustering algorithms and their evaluation scores is a good practice to find the best model for your specific problem.

## Small Project: Segmenting Satellite Images with GMM

**Objective:** Use a Gaussian Mixture Model (GMM) to cluster different types of terrain from satellite image data and use PCA to visualize the results.

**Dataset:** The [Statlog (Landsat Satellite)](https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)) dataset from the UCI repository. This dataset consists of pixel data from a satellite image, and the goal is to classify the terrain type.

**Steps:**

1.  **Load and Prepare the Data:**
    *   You can find the data file `sat.trn` in the repository. Load it using `pd.read_csv('.../sat.trn', sep=' ', header=None)`.
    *   The features are in columns 0-35, and the target label (terrain type) is in column 36.
    *   Separate your features (X) and true labels (y).
    *   **Scale your features** using `StandardScaler`.

2.  **Find the Optimal Number of Components with BIC:**
    *   While the Elbow Method works for K-Means, a more common method for GMM is to use information criteria like the **Bayesian Information Criterion (BIC)** or Akaike Information Criterion (AIC). A lower BIC score generally indicates a better model.
    *   Create a loop that fits a `GaussianMixture` model for a range of `n_components` (e.g., from 1 to 10).
    *   In the loop, calculate and store the `bic()` of each model.
    *   Plot the BIC scores against the number of components. The value of `n_components` that gives the minimum BIC is a good choice.

3.  **Train and Evaluate the GMM:**
    *   Train a `GaussianMixture` model using the optimal number of components you found.
    *   Predict the cluster labels for your data.
    *   Even though this is unsupervised, you have the true labels. Use `pd.crosstab(y_true, gmm_labels)` to see how well your clusters correspond to the actual terrain types. Does each cluster primarily contain one type of terrain?

4.  **Visualize with PCA:**
    *   The dataset has 36 features. Use `PCA` to reduce the dimensionality to 2 components.
    *   Create a scatter plot of the two principal components.
    *   Color the points in the scatter plot using the cluster labels from your GMM. This will give you a visual representation of your segmentation.

**Key Takeaway:** This project will give you experience using GMM on a real-world dataset. You will learn a new technique (BIC) for model selection with GMMs and reinforce your ability to use PCA for visualizing high-dimensional clusters. You'll also learn how to assess the quality of your clustering by comparing it to ground truth labels.
