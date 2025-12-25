---
id: day-14
title: 'Day 14: Customer Segmentation with K-Means Clustering'
---

## Day 14: Customer Segmentation with K-Means Clustering

Today, we'll dive into **unsupervised learning** with one of the most popular clustering algorithms: **K-Means**. Our goal is to perform customer segmentation on the `Mall_Customers.csv` dataset to identify different groups of customers based on their spending habits.

### What is K-Means Clustering?

K-Means is an algorithm that aims to partition a set of data points into 'K' distinct, non-overlapping clusters. It works by:
1.  Initializing 'K' cluster centroids randomly.
2.  Assigning each data point to the nearest centroid.
3.  Recalculating the centroids as the mean of all data points assigned to that cluster.
4.  Repeating steps 2 and 3 until the cluster assignments no longer change.

Like KNN, K-Means is distance-based, so **feature scaling is important**.

### The Workflow

#### 1. Data Preprocessing

First, we load the data and prepare it for clustering. This involves one-hot encoding the 'Gender' column and scaling the numerical features.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./datasets/Mall_Customers.csv')

# One-hot encode the 'Gender' column
df = pd.get_dummies(df, columns=['Gender'])

# Scale numerical features
features_to_scale = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
```

#### 2. Finding the Optimal Number of Clusters (K)

How do we choose the right value for 'K'? Two common methods are the Elbow Method and the Silhouette Score.

##### The Elbow Method

The Elbow Method calculates the **Within-Cluster Sum of Squares (WCSS)** for different values of K. WCSS is the sum of the squared distances between each data point and its centroid. We plot WCSS against K, and the "elbow" of the curve is a good estimate for the optimal K.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []
features = df[['Annual Income (k$)', 'Spending Score (1-100)']]

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_) # kmeans.inertia_ calculates WCSS

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
```
The plot shows a clear elbow at **K=5**.

##### Silhouette Score

The Silhouette Score measures how similar a data point is to its own cluster compared to other clusters. The score ranges from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.

```python
from sklearn.metrics import silhouette_score

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(features)
    score = silhouette_score(features, labels)
    print(f"For n_clusters = {k}, Silhouette Score = {score:.2f}")
```
The output shows the highest silhouette score is for **K=5**, which confirms our finding from the Elbow Method.

#### 3. Training and Visualizing the Model

Now that we know the optimal number of clusters is 5, we can train our final model and visualize the results.

```python
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(features)
labels = kmeans.labels_

plt.scatter(features['Annual Income (k$)'], features['Spending Score (1-100)'], 
            c=labels, cmap='viridis')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

## Small Project: Segmenting Universities

**Objective:** Use K-Means clustering to segment universities into different groups based on their characteristics.

**Dataset:** A modified version of the [College Scorecard](https://collegescorecard.ed.gov/data/) dataset is available on Kaggle. For this project, we'll use a simplified version. Let's find a suitable one on Kaggle. A good alternative is the ["Private or Public University"](https://www.kaggle.com/datasets/fatemehmehrparvar/private-or-public-university) dataset.

**Steps:**

1.  **Load and Prepare the Data:**
    *   Load the `Private or Public University.csv` data.
    *   You'll need to encode the 'Private' column (`Yes`/`No`) into `1`/`0`.
    *   All features should be scaled using `StandardScaler` before clustering.

2.  **Find the Optimal Number of Clusters (K):**
    *   Use the **Elbow Method** to plot the WCSS for a range of K values (e.g., 1 to 10). Where does the "elbow" appear?
    *   Use the **Silhouette Score** to confirm your choice of K.

3.  **Train the K-Means Model:**
    *   Train a `KMeans` model with your chosen number of clusters.
    *   Add the cluster labels back to your original (unscaled) DataFrame. This is important for interpretation.

4.  **Interpret the Clusters:**
    *   This is the most important step. Now that you have your clusters, what do they represent?
    *   Use `groupby()` on your DataFrame to group by the new 'Cluster' column.
    *   For each cluster, calculate the mean of the other features (e.g., `Apps`, `Accept`, `Top10perc`, `PhD`, `Grad.Rate`).
    *   Analyze the results. For example:
        *   Is there a cluster for "elite, highly selective private schools"? (High `Top10perc`, low `Accept` rate, high `PhD`, high `Grad.Rate`).
        *   Is there a cluster for "large, open-enrollment public schools"? (High `Apps` and `Enroll`, lower `Top10perc`).
        *   Describe each of your clusters in a sentence or two based on their average characteristics.

**Key Takeaway:** This project teaches you that the goal of clustering is not just to create groups, but to *understand* them. You will learn how to analyze the results of a clustering algorithm to extract meaningful insights and create "personas" for each cluster.

The scatter plot clearly shows the 5 distinct customer segments based on their income and spending score. This is valuable information for a business that wants to create targeted marketing campaigns.
