---
id: day-13
title: 'Day 13: K-Nearest Neighbors (KNN)'
---

## Day 13: K-Nearest Neighbors (KNN)

Today, we'll explore the **K-Nearest Neighbors (KNN)** algorithm, a simple yet powerful non-parametric method used for both classification and regression. We'll see how to apply it to two different datasets and how to use essential tools like `Pipelines` and `GridSearchCV` to improve our models.

### How Does KNN Work?

The core idea behind KNN is simple:
*   To predict the label of a new data point, KNN looks at the 'K' closest data points (its "neighbors") in the training set.
*   **For classification:** It assigns the new data point the most common class among its K neighbors (majority vote).
*   **For regression:** It assigns the new data point the average of the values of its K neighbors.

Because KNN relies on the distance between data points, it's crucial to **scale the features** before training the model.

---

### Part 1: KNN for Regression

In this part, we'll use KNN to predict house prices from the `USA Housing Dataset.csv`.

#### The Challenge with Unscaled Data

First, we tried using `KNeighborsRegressor` directly on the unscaled data.

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# ... (load and split data) ...

knn_reg = KNeighborsRegressor(n_neighbors=10)
knn_reg.fit(x_train, y_train)
y_pred = knn_reg.predict(x_test)

r2 = r2_score(y_test, y_pred) 
# The R2 score was negative, indicating the model performed very poorly.
```
A negative R2 score means the model is worse than just predicting the mean of the target variable. This is a clear sign that something is wrong, and in the case of KNN, the issue is almost always feature scaling.

#### Using Pipelines and GridSearchCV

To fix this, we build a `Pipeline` that first scales the data using `StandardScaler` and then trains the KNN model. We also use `GridSearchCV` to find the best hyperparameters.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Create a pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=5))
])

# Define the hyperparameter grid to search
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

# Create and fit the GridSearchCV object
grid = GridSearchCV(pipe, param_grid, cv=10, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid.fit(x_train, y_train)

print("Best Parameters:", grid.best_params_)
```
This process automates the scaling and hyperparameter tuning, leading to a much better model.

---

### Part 2: KNN for Classification

Now, let's use KNN for a classification task on the classic `Iris.csv` dataset.

#### The Workflow

The workflow is very similar to the regression task. We use a `Pipeline` to combine scaling and classification.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=4))
])

# Use cross-validation to get a robust estimate of performance
cv_scores = cross_val_score(pipeline, X, y, cv=10, scoring='accuracy')
print("Average Accuracy:", cv_scores.mean())
# Average Accuracy: 0.993
```

#### Hyperparameter Tuning

We can use `GridSearchCV` again to find the best number of neighbors (`n_neighbors`) and the best `weights` strategy.

```python
param_grid = {
    "knn__n_neighbors": [2, 3, 4, 5, 6, 7],
    'knn__weights': ['uniform', 'distance']
}

grid = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid.fit(x_train, y_train)

print("Best Parameters:", grid.best_params_)

# Evaluate the best model on the test set
best_model = grid.best_estimator_
y_pred = best_model.predict(x_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
# Test Accuracy: 1.0
```

### Key Takeaways for Day 13

*   **KNN** is a simple, distance-based algorithm for both regression and classification.
*   **Feature Scaling** (e.g., with `StandardScaler`) is **essential** for KNN to work correctly.
*   **Pipelines** are an incredibly useful tool for chaining together data preprocessing steps and a final estimator.
*   **GridSearchCV** helps you automate the process of finding the best hyperparameters for your model.

## Small Project: Glass Identification

**Objective:** Use the K-Nearest Neighbors algorithm to identify the type of glass based on its chemical composition.

**Dataset:** The [Glass Identification](https://archive.ics.uci.edu/ml/datasets/glass+identification) dataset from the UCI Machine Learning Repository. You can load it directly from the web: `pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data', header=None)`.

**Steps:**

1.  **Load and Prepare the Data:**
    *   Load the data using the URL above. You'll need to manually add column names. The columns are: `Id, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe, Type`.
    *   The 'Id' column is not a feature and should be dropped.
    *   The target variable is 'Type'.

2.  **Build a KNN Pipeline:**
    *   Just like in the lesson, it's crucial to scale the data.
    *   Create a `Pipeline` that contains two steps:
        1.  `StandardScaler`
        2.  `KNeighborsClassifier`

3.  **Find the Optimal K:**
    *   Use `GridSearchCV` to find the best hyperparameters for your KNN model.
    *   You should search for the best `n_neighbors` (e.g., from 1 to 20).
    *   You can also search for the best `weights` ('uniform' vs. 'distance') and `metric` ('euclidean' vs. 'manhattan').
    *   Use `cv=5` for cross-validation.

4.  **Evaluate the Best Model:**
    *   After the grid search is complete, get the `best_estimator_`.
    *   Split your data into a training and testing set.
    *   Fit the best model on the training data and evaluate its performance on the test data using `accuracy_score` and a `classification_report`.

5.  **Interpret the Results:**
    *   Look at the `classification_report`. Are there certain types of glass that the model is better or worse at predicting?
    *   Why might this be the case? (Hint: Look at the class distribution of the 'Type' column using `df['Type'].value_counts()`).

**Key Takeaway:** This project will solidify your understanding of the complete KNN workflow, including the critical steps of scaling and hyperparameter tuning. It will also introduce you to the important step of analyzing the results to understand the model's strengths and weaknesses on a per-class basis.
