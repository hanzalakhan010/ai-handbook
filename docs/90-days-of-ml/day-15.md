---
id: day-15
title: 'Day 15: Hierarchical Clustering'
---

## Day 15: Hierarchical Clustering

Today we'll explore another type of clustering algorithm called **Hierarchical Clustering**. Unlike K-Means, it doesn't require you to specify the number of clusters beforehand. Instead, it builds a hierarchy of clusters.

### What is Hierarchical Clustering?

There are two main types of hierarchical clustering:

1.  **Agglomerative (Bottom-Up):** This is the more common approach. It starts with each data point as its own cluster and then iteratively merges the closest pairs of clusters until only one cluster (containing all data points) is left.
2.  **Divisive (Top-Down):** This approach starts with all data points in a single cluster and recursively splits the clusters until each data point is its own cluster.

We will focus on **Agglomerative Clustering**.

### Visualizing Clusters with a Dendrogram

A **dendrogram** is a tree-like diagram that records the sequence of merges or splits. It's the primary tool for visualizing hierarchical clustering and helping us decide on the optimal number of clusters.

The y-axis of the dendrogram represents the distance between clusters. To find the optimal number of clusters, we look for the longest vertical line that doesn't cross any horizontal lines and count the number of vertical lines it crosses.

#### Example with the Iris Dataset

Let's apply this to the Iris dataset.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Load and scale the data
data = load_iris()
X_scaled = StandardScaler().fit_transform(data.data)

# Perform hierarchical clustering
linked = linkage(X_scaled, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title("Dendrogram for Iris Dataset")
plt.show()
```
By examining the dendrogram, we can see that cutting the tree where the vertical lines are longest suggests that 2 or 3 clusters would be a good choice.

### Evaluating Clustering Performance

Once we have our clusters, how do we know if they are good? We can use evaluation metrics like:

*   **Silhouette Score:** Measures how similar an object is to its own cluster compared to other clusters. Higher is better.
*   **Davies-Bouldin Score:** Measures the average similarity ratio of each cluster with its most similar cluster. Lower is better.

#### Example with the Wine Quality Dataset

Let's apply this to a more complex dataset, `WineQT.csv`, and programmatically find the best number of clusters.

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ... (load and scale the WineQT data) ...

# Iterate through different numbers of clusters and calculate scores
for i in range(2, 11):
    model = AgglomerativeClustering(n_clusters=i)
    labels = model.fit_predict(X_scaled)
    
    sil_score = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)
    
    print(f"n_clusters = {i}, Silhouette Score = {sil_score:.2f}, Davies-Bouldin Score = {db_score:.2f}")

```

By analyzing the output of these scores for different `n_clusters`, you can make a more informed decision about the optimal number of clusters for your data. For example, you might choose the `n_clusters` that gives you the highest Silhouette Score and the lowest Davies-Bouldin Score.

### Key Takeaways for Day 15

*   **Hierarchical Clustering** builds a hierarchy of clusters, which can be visualized with a **dendrogram**.
*   The dendrogram is a useful tool for deciding on the number of clusters.
*   Metrics like the **Silhouette Score** and **Davies-Bouldin Score** can be used to programmatically evaluate the quality of your clusters.

## Small Project: Clustering Wholesale Customers

**Objective:** Apply Hierarchical Clustering to segment wholesale customers and compare its results to K-Means.

**Dataset:** The [Wholesale customers data Set](https://archive.ics.uci.edu/ml/datasets/wholesale+customers) from the UCI Machine Learning Repository.

**Steps:**

1.  **Load and Prepare the Data:**
    *   Load the `Wholesale customers data.csv` data.
    *   The data contains features like 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'. It's all numerical, but the scales are very different.
    *   Drop the 'Channel' and 'Region' columns for simplicity.
    *   **Scale the data** using `StandardScaler`.

2.  **Hierarchical Clustering:**
    *   **Create a Dendrogram:** Use `scipy.cluster.hierarchy.dendrogram` and `linkage` to create a dendrogram of your data.
    *   Look at the dendrogram. What would be a reasonable number of clusters to choose? (Try to find the longest vertical line that doesn't cross a horizontal line). Let's say you choose K=3 for this example.
    *   **Train the Model:** Train an `AgglomerativeClustering` model with your chosen number of clusters.

3.  **K-Means Clustering:**
    *   Now, train a `KMeans` model on the same data. Use the same number of clusters you chose for the hierarchical model to ensure a fair comparison.

4.  **Compare the Results:**
    *   You now have two sets of labels for your data, one from each clustering algorithm.
    *   A simple way to compare them is to use `pd.crosstab()`. This will create a table showing how many data points from a K-Means cluster ended up in each Hierarchical cluster.
    *   `pd.crosstab(kmeans_labels, hierarchical_labels)`
    *   Do the clusters from the two methods align well? Or are there significant differences?

5.  **Interpret the Clusters:**
    *   Choose one of the clustering results (e.g., from K-Means).
    *   Just like in the Day 14 project, add the cluster labels back to the original DataFrame and use `groupby()` to calculate the mean of the features for each cluster.
    *   What do the clusters represent? Is there a "general retailer" cluster (high in all categories)? A "cafe" cluster (high in 'Milk' and 'Grocery')?

**Key Takeaway:** This project will give you a practical comparison between the two main clustering methods. You will learn that different algorithms can produce different groupings and see how a dendrogram can help guide your choice of K for hierarchical clustering.
