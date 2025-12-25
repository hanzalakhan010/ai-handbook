---
id: day-18
title: 'Day 18: Dimensionality Reduction - PCA, t-SNE, and UMAP'
---

## Day 18: Dimensionality Reduction - PCA, t-SNE, and UMAP

Today, we'll dive into **dimensionality reduction**, a crucial step in machine learning for dealing with high-dimensional datasets. Reducing the number of features can help with visualization, reduce computational cost, and prevent the curse of dimensionality. We'll explore three popular techniques: **Principal Component Analysis (PCA)**, **t-Distributed Stochastic Neighbor Embedding (t-SNE)**, and **Uniform Manifold Approximation and Projection (UMAP)**.

### Overview of Dimensionality Reduction Techniques

| Aspect               | PCA                     | t-SNE                     | UMAP                      |
| :------------------- | :---------------------- | :------------------------ | :------------------------ |
| **Method**           | Linear                  | Non-linear                | Non-linear                |
| **Goal**             | Maximize variance       | Preserve local structure  | Preserve local + global structure |
| **Speed**            | Fast                    | Slow                      | Faster than t-SNE         |
| **Scalability**      | Scales well             | Limited to ~10,000 samples | Handles millions of samples |
| **Output Dimensions**| Any                     | Typically 2 or 3          | Typically 2 or 3          |
| **Hyperparameters**  | Minimal                 | Sensitive (e.g., perplexity) | Intuitive (e.g., `n_neighbors`, `min_dist`) |
| **Global Structure** | Good                    | Poor                      | Better than t-SNE         |
| **Local Structure**  | Weak                    | Excellent                 | Excellent                 |
| **Interpretability** | High                    | Low                       | Moderate                  |

### When to Use Each Technique:

*   **PCA:** Use for general dimensionality reduction, preprocessing for ML pipelines, or when interpretability of components is important. It's fast and scales well.
*   **t-SNE:** Ideal for visualizing smaller datasets where you want to highlight local groupings or separations. It excels at preserving local structures.
*   **UMAP:** A modern alternative to t-SNE, offering better speed and scalability, and often better preservation of both local and global structures. Great for exploring large, complex datasets.

---

### 1. Principal Component Analysis (PCA)

PCA is a linear technique that transforms your data into a new set of orthogonal (uncorrelated) components, called principal components. The first principal component captures the most variance in the data, the second captures the second most, and so on.

#### Impact of Scaling on PCA

PCA is sensitive to the scale of the features. We'll demonstrate this using the Iris dataset.

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load and prepare data
df = load_iris(as_frame=True).frame
y = df.target
X = df.drop(columns=['target'])

# --- Scaling ---
# Standard Scaler
x_std_scaled = StandardScaler().fit_transform(X)
# Min-Max Scaler
x_mm_scaled = MinMaxScaler().fit_transform(X)

# --- PCA on Standard Scaled Data ---
pca_std = PCA(n_components=2)
xy_std = pca_std.fit_transform(x_std_scaled)
plt.figure(figsize=(6, 5))
plt.scatter(xy_std[:, 0], xy_std[:, 1], c=y, cmap='Accent')
plt.title('PCA on Standard Scaled Iris Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# --- PCA on Min-Max Scaled Data ---
pca_mm = PCA(n_components=2)
xy_mm = pca_mm.fit_transform(x_mm_scaled)
plt.figure(figsize=(6, 5))
plt.scatter(xy_mm[:, 0], xy_mm[:, 1], c=y, cmap='Accent')
plt.title('PCA on Min-Max Scaled Iris Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
```
The visualizations will show how different scaling methods can influence the spread and separation of clusters after PCA.

---

### 2. t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE is a non-linear technique particularly well-suited for visualizing high-dimensional datasets. It maps high-dimensional data to a lower-dimensional space (typically 2D or 3D) while trying to preserve the local relationships between data points.

#### Code Example: t-SNE on Breast Cancer Data

Let's use the breast cancer dataset for this example, as it has more features.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load and scale data
df_bc = load_breast_cancer(as_frame=True).frame
y_bc = df_bc.target
X_bc = df_bc.drop(columns=['target'])
X_bc_scaled = StandardScaler().fit_transform(X_bc)

# --- t-SNE ---
tsne = TSNE(n_components=2, random_state=3, perplexity=30, n_iter=10000)
X_tsne = tsne.fit_transform(X_bc_scaled)

plt.figure(figsize=(6, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_bc, cmap='Accent')
plt.title('t-SNE on Breast Cancer Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
```
t-SNE is excellent for revealing clusters in your data that might not be apparent with linear methods like PCA.

---

### <mark>Things I Didn't Go Through (But You Should Explore)</mark>

*   **UMAP Implementation:** While conceptually discussed, UMAP was not implemented. You should try to implement UMAP and compare its results (speed, cluster separation) with PCA and t-SNE on these datasets. The `umap-learn` library is easy to use.
*   **Hyperparameter Tuning for t-SNE and UMAP:** Both t-SNE (especially `perplexity`) and UMAP (`n_neighbors`, `min_dist`) have hyperparameters that significantly affect the visualization. Experiment with different values to understand their impact.
*   **Interpretation of Components:** For PCA, you can analyze the `components_` attribute to understand which original features contribute most to each principal component. This can offer insights into your data.
*   **Combining Techniques:** Often, PCA is used as a preprocessing step to reduce noise and initial dimensionality before applying t-SNE or UMAP to a smaller number of components.

## Small Project: Visualizing Handwritten Digits

**Objective:** Apply and compare PCA, t-SNE, and UMAP for visualizing the MNIST handwritten digits dataset.

**Dataset:** The famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/) of handwritten digits. It's readily available in many libraries, including `sklearn.datasets.fetch_openml`.

**Steps:**

1.  **Load the Data:**
    *   Loading MNIST can take a moment. It's a good practice to work with a smaller subset for faster iteration.
    *   `from sklearn.datasets import fetch_openml`
    *   `mnist = fetch_openml('mnist_784', version=1)`
    *   `X, y = mnist['data'], mnist['target']`
    *   Let's take a random sample of 5000 data points to speed things up:
        *   `import numpy as np`
        *   `sample_indices = np.random.choice(X.shape[0], 5000, replace=False)`
        *   `X_sample, y_sample = X.iloc[sample_indices], y.iloc[sample_indices]`

2.  **Scale the Data:**
    *   Scale the pixel values using `StandardScaler`.

3.  **Apply Dimensionality Reduction Techniques:**
    *   **PCA:** Apply `PCA` with `n_components=2` to your scaled data.
    *   **t-SNE:** Apply `TSNE` with `n_components=2`. A `perplexity` of 30-50 is a good starting point.
    *   **UMAP:** You'll need to `pip install umap-learn`. Then, import `umap.UMAP` and apply it with `n_components=2`. Good starting parameters are `n_neighbors=15` and `min_dist=0.1`.

4.  **Visualize and Compare:**
    *   Create a figure with three subplots, one for each technique.
    *   In each subplot, create a scatter plot of the 2D-reduced data.
    *   Color the points according to their true digit label (`y_sample`). You'll need to convert `y_sample` to integers (`y_sample.astype(int)`).
    *   Add a title to each subplot indicating the method used (PCA, t-SNE, or UMAP).
    *   How do the visualizations compare?
        *   Does PCA show any separation?
        *   How well do t-SNE and UMAP separate the different digit clusters?
        *   Is there a noticeable difference between the t-SNE and UMAP embeddings?

**Key Takeaway:** This project provides a powerful visual comparison of linear vs. non-linear dimensionality reduction. You will see firsthand why t-SNE and UMAP are so popular for data visualization, and you will gain experience using `umap-learn`, a state-of-the-art dimensionality reduction library.
