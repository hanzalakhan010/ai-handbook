---
id: day-19
title: 'Day 19: Anomaly Detection and Association Rule Learning'
---

## Day 19: Anomaly Detection and Association Rule Learning

Today's session is packed with two distinct yet powerful machine learning techniques: **Anomaly Detection**, which helps us find unusual patterns in data, and **Association Rule Learning**, which uncovers interesting relationships between items.

---

### Part 1: Anomaly Detection

Anomaly detection is the process of identifying rare items, events, or observations that deviate significantly from the majority of the data. Anomalies can indicate critical incidents like fraud, system malfunctions, or rare diseases.

#### The Problem: Credit Card Fraud

We'll use the `creditcard.csv` dataset, a classic example of an imbalanced dataset where fraudulent transactions (anomalies) are rare. For unsupervised anomaly detection, we often remove the `Class` label and let the algorithm find the anomalies.

#### Techniques Explored: Isolation Forest and One-Class SVM

1.  **Isolation Forest:** This algorithm is built on the principle that anomalies are "few and different," making them easier to isolate than normal observations. It works by randomly selecting features and then randomly selecting a split point between the maximum and minimum values of the selected feature. This process is repeated to create "isolation trees." Anomalies typically require fewer splits to be isolated.

2.  **One-Class SVM:** A variant of Support Vector Machines, One-Class SVM learns a decision boundary that encapsulates the "normal" data points. Any data point falling outside this boundary is considered an anomaly.

#### Code Example: Isolation Forest

We'll first reduce the dimensionality of the data using PCA for visualization purposes, then scale it.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM # Used for One-Class SVM
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load data and prepare
df = pd.read_csv('./datasets/creditcard.csv')
# For unsupervised anomaly detection, we typically don't use the 'Class' label
# but we'll keep it for visualization to see how well we detect fraud.
y_true = df['Class']
X = df.drop(columns=['Class'])

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# --- Isolation Forest ---
# 'contamination' parameter estimates the proportion of outliers in the data
iso_forest = IsolationForest(n_estimators=100, contamination=0.001, random_state=2)
iso_forest_pred = iso_forest.fit_predict(X_scaled)

# Separate anomalies and normal points for visualization
anomalies_iso = X_scaled[iso_forest_pred == -1]
normals_iso = X_scaled[iso_forest_pred == 1]

plt.figure(figsize=(10, 6))
plt.scatter(normals_iso[:, 0], normals_iso[:, 1], c='green', label='Normal Points', s=5, alpha=0.6)
plt.scatter(anomalies_iso[:, 0], anomalies_iso[:, 1], c='red', label='Anomalies', s=15, marker='x')
plt.title('Isolation Forest Anomaly Detection (PCA-reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# --- One-Class SVM ---
# The 'nu' parameter is an upper bound on the fraction of training errors
# and a lower bound of the fraction of support vectors.
one_class_svm = OneClassSVM(kernel='linear', gamma=0.1, nu=0.4)
one_class_svm.fit(X_scaled)
one_class_svm_pred = one_class_svm.predict(X_scaled)

# You can similarly visualize and evaluate One-Class SVM results.
```

---

### Part 2: Association Rule Learning

Association rule learning is a rule-based machine learning method for discovering interesting relations between variables in large databases. It is intended to identify strong rules discovered in databases using some measures of "interestingness."

#### The Problem: Market Basket Analysis (Groceries)

We'll use the `Groceries_dataset.csv` to find out what items are frequently purchased together. This is a classic "Market Basket Analysis" problem.

#### The Apriori Algorithm

The **Apriori Algorithm** is a popular algorithm for frequent itemset mining and association rule learning. It works in two steps:
1.  **Frequent Itemset Generation:** Find all itemsets that occur in at least a user-specified minimum number of transactions (support).
2.  **Rule Generation:** From the frequent itemsets, generate strong association rules.

#### Key Metrics for Association Rules:

*   **Support:** How frequently an itemset appears in the dataset.
    `Support(X) = (Number of transactions containing X) / (Total number of transactions)`
*   **Confidence:** The probability that a customer will buy item Y given that they have already bought item X.
    `Confidence(X -> Y) = Support(X and Y) / Support(X)`
*   **Lift:** How much more likely item Y is purchased when item X is purchased, relative to its baseline probability. A lift value greater than 1 suggests a positive correlation.
    `Lift(X -> Y) = Confidence(X -> Y) / Support(Y)`

#### Code Example: Apriori Algorithm

First, we need to preprocess the data into a "basket" format.

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import numpy as np

df_groceries = pd.read_csv('./datasets/Groceries_dataset.csv')

# --- Data Preprocessing: Convert to Basket Format ---
# Group by Member_number and itemDescription, then count occurrences
baskets = df_groceries.groupby(['Member_number', 'itemDescription'])['itemDescription'].count().unstack().fillna(0)

# Convert counts to binary (1 if present, 0 if not)
def one_hot_encoder(k):
    return 1 if k >= 1 else 0

baskets_final = baskets.applymap(one_hot_encoder)

# --- Frequent Itemset Generation ---
# min_support: The minimum support value for an itemset to be considered frequent
frequent_items = apriori(baskets_final, min_support=0.025, use_colnames=True, max_len=2)

# --- Rule Generation ---
# min_threshold: The minimum threshold for the specified metric (here, 'lift')
rules = association_rules(frequent_items, metric='lift', min_threshold=1)
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
rules = rules.sort_values('lift', ascending=False)

print("Top 25 Association Rules:")
print(rules.head(25))
```
The output will show rules like `{yogurt} -> {whole milk}` with their respective support, confidence, and lift values, revealing which products are often bought together.

---

### <mark>Things I Didn't Go Through (But You Should Explore)</mark>

*   **Advanced Anomaly Detection:** Explore other anomaly detection algorithms like DBSCAN (which can also detect noise points as anomalies), Local Outlier Factor (LOF), or clustering-based anomaly detection methods.
*   **Evaluation of Anomaly Detection:** For unsupervised anomaly detection, evaluating performance can be tricky without true labels. Learn about techniques like precision-recall curves if some labeled anomalies are available.
*   **FP-growth Algorithm:** An alternative to Apriori for frequent itemset mining that can be more efficient for very large datasets.
*   **Visualizing Association Rules:** There are various ways to visualize association rules, such as network graphs or scatter plots of support, confidence, and lift.

## Small Project: Expanding Your Toolkit

**Objective:** Explore alternative algorithms for both anomaly detection and association rule learning.

---

### Part A: Anomaly Detection with Local Outlier Factor (LOF)

**LOF** is another powerful anomaly detection algorithm that works by measuring the local density deviation of a data point with respect to its neighbors. It can perform well in datasets where anomalies might form their own small clusters.

**Dataset:** Continue using the `creditcard.csv` dataset.

**Steps:**

1.  **Load and Prepare Data:** Load and scale the data as you did for the Isolation Forest example.
2.  **Train LOF Model:**
    *   `from sklearn.neighbors import LocalOutlierFactor`
    *   Create a `LocalOutlierFactor` model. Key parameters are `n_neighbors` and `contamination`. Set `contamination` to the same value you used for Isolation Forest (e.g., `0.001`).
    *   Use `fit_predict()` to get the anomaly labels (`-1` for outliers, `1` for inliers).
3.  **Compare Results:**
    *   How many outliers did LOF find compared to Isolation Forest?
    *   Since you have the true labels, you can create a `classification_report` for both models' predictions to see how their precision and recall for the fraud class (`1`) compare. (Note: you will need to convert the models' -1/1 output to the 0/1 format of the true labels).

---

### Part B: Association Rules with FP-Growth

The **FP-Growth** algorithm is an improvement over Apriori for finding frequent itemsets. It avoids the costly step of generating and testing a large number of candidate itemsets by using a compact tree-based data structure (the FP-tree).

**Dataset:** Continue using the `Groceries_dataset.csv`.

**Steps:**

1.  **Load and Prepare Data:** Prepare the data in the one-hot encoded "basket" format, just as you did for Apriori.
2.  **Run FP-Growth:**
    *   `from mlxtend.frequent_patterns import fpgrowth`
    *   Use `fpgrowth()` on your one-hot encoded DataFrame. It takes the same `min_support` parameter as `apriori`.
    *   Use the `time` module to measure how long `fpgrowth` takes to run.
3.  **Compare to Apriori:**
    *   Run `apriori()` with the same `min_support`. Measure how long it takes.
    *   Is there a noticeable difference in speed? (For this dataset it may be small, but for larger datasets, FP-Growth is often much faster).
    *   Do both algorithms produce the exact same set of frequent itemsets? (They should!).
4.  **Generate Rules:**
    *   You can feed the frequent itemsets generated by `fpgrowth` directly into the `association_rules` function, just like you did with the output from `apriori`.

**Key Takeaway:** This project broadens your toolkit by introducing you to `LOF` and `FP-Growth`, powerful alternatives to Isolation Forest and Apriori. You'll learn that there are often multiple algorithms to solve the same problem, each with its own trade-offs in performance and efficiency.
