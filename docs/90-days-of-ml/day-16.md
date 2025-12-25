---
id: day-16
title: 'Day 16: Density-Based Clustering with DBSCAN'
---

## Day 16: Density-Based Clustering with DBSCAN

In the last few days, we've explored clustering algorithms like K-Means and Hierarchical Clustering. Today, we'll look at a different approach: **density-based clustering**, using an algorithm called **DBSCAN**.

### What is DBSCAN?

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** is a clustering algorithm that groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions.

This is very different from K-Means, which creates spherical clusters. DBSCAN can find arbitrarily shaped clusters and is also robust to outliers.

DBSCAN has two key parameters:
1.  **`eps` (epsilon):** The maximum distance between two samples for one to be considered as in the neighborhood of the other.
2.  **`min_samples`:** The number of samples in a neighborhood for a point to be considered as a core point.

### DBSCAN vs. K-Means on a Non-Linear Dataset

To see the power of DBSCAN, let's compare it to K-Means on the `make_moons` dataset, which has a non-linear shape that is challenging for K-Means.

```python
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt

# Generate the dataset
X, y = make_moons(n_samples=20000, noise=0.3, random_state=2)

# --- DBSCAN ---
dbscan = DBSCAN(eps=0.175, min_samples=15)
db_labels = dbscan.fit_predict(X)

# --- K-Means ---
kmeans = KMeans(n_clusters=2, random_state=2)
km_labels = kmeans.fit_predict(X)

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(X[:,0], X[:,1], c=db_labels, cmap='Accent')
ax1.set_title('DBSCAN')
ax2.scatter(X[:,0], X[:,1], c=km_labels, cmap='Accent')
ax2.set_title('K-Means')
plt.show()
```

When you visualize the results, you'll see that DBSCAN is able to correctly identify the two moon-shaped clusters, while K-Means, trying to create spherical clusters, fails to separate them properly.

### The Importance of Parameter Tuning in DBSCAN

DBSCAN is powerful, but its performance is highly dependent on the choice of `eps` and `min_samples`.

Let's see what happens when we apply it to the `Wholesale customers data.csv` dataset. In the notebook for this example, the chosen parameters for DBSCAN resulted in all data points being classified as noise (a label of `-1`).

This is a critical learning point: **if your parameters are not set correctly for your data's density, DBSCAN may fail to find any clusters at all.**

When this happens, you need to go back and tune the parameters. There are methods to help you find good values for `eps` and `min_samples`, such as the **K-distance graph**.

### Key Takeaways for Day 16

*   **DBSCAN** is a density-based clustering algorithm that can find arbitrarily shaped clusters and is robust to outliers.
*   It excels at clustering non-linear datasets where K-Means fails.
*   The performance of DBSCAN is highly sensitive to the `eps` and `min_samples` parameters.
*   Unlike K-Means, you don't need to specify the number of clusters for DBSCAN beforehand. It determines the number of clusters based on the data's density.

## Small Project: Finding Optimal `eps` for DBSCAN

**Objective:** Learn a practical technique to estimate a good value for the `eps` parameter in DBSCAN using a k-distance plot.

**Dataset:** Continue using the `Wholesale customers data.csv` dataset.

**The Challenge:** As noted in the lesson, running DBSCAN with default or poorly chosen parameters can result in all points being labeled as noise (`-1`). The choice of `eps` is critical.

**The Technique: K-Distance Plot**
A common heuristic for choosing `eps` is to analyze the distance to the *k*-th nearest neighbor for all points. The "knee" or "elbow" in the sorted plot of these distances can be a good estimate for `eps`.

**Steps:**

1.  **Load and Scale the Data:**
    *   Load the `Wholesale customers data.csv` and scale it using `StandardScaler`.

2.  **Calculate Nearest Neighbor Distances:**
    *   Use `NearestNeighbors` from `sklearn.neighbors` to calculate the distance of each point to its neighbors.
    *   A good starting point for `min_samples` is often related to the dimensionality of your data. A common rule of thumb is `min_samples = 2 * n_features`. For this dataset, that would be `2 * 6 = 12`. Let's use `n_neighbors=12`.
    *   Fit the `NearestNeighbors` model to your scaled data.
    *   Use `kneighbors()` to get the distances to the 12 nearest neighbors for each point.

3.  **Create the K-Distance Plot:**
    *   The `kneighbors()` method returns two arrays: distances and indices. We only need the distances.
    *   We are interested in the distance to the 12th neighbor, so we take the last column of the distances array.
    *   Sort these distances in ascending order and plot them.

    ```python
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    
    # After scaling your data to X_scaled...
    neighbors = NearestNeighbors(n_neighbors=12)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)
    
    # Sort the distances to the k-th neighbor
    distances = np.sort(distances[:, 11]) 
    
    plt.plot(distances)
    plt.title('K-Distance Plot')
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel('Distance to 12th Nearest Neighbor (eps)')
    plt.show()
    ```

4.  **Choose `eps` and Run DBSCAN:**
    *   Look at the plot. Find the "elbow" of the curve - the point where the distances start to rise sharply. The y-value at this point is a good candidate for `eps`.
    *   Now, run `DBSCAN` again on your data, but this time use your estimated `eps` and `min_samples=12`.
    *   How many clusters did you find this time? How many points are considered noise? The result should be much more meaningful than before.

**Key Takeaway:** This project teaches you a crucial, practical skill for using DBSCAN effectively. You will learn that hyperparameter tuning for DBSCAN is not just guesswork, and techniques like the k-distance plot can provide a strong, data-driven starting point.
