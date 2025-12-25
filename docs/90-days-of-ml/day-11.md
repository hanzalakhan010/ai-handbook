---
id: day-11
title: 'Day 11: Fraud Detection with Ensemble Models'
---

## Day 11: Fraud Detection with Ensemble Models

### A Quick Recap

Today's focus is on a real-world classification problem: **credit card fraud detection**. We'll use powerful ensemble models, Random Forests and Gradient Boosting, to tackle this challenge and compare their performance.

### The Dataset: Credit Card Fraud

We are using the `creditcard.csv` dataset, a common dataset for this task. A key characteristic of this dataset is that it is **highly imbalanced**. This means that the number of fraudulent transactions is very small compared to the number of legitimate transactions.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load the dataset
df = pd.read_csv('./datasets/creditcard.csv')

# Check class distribution
print(df['Class'].value_counts())
# Output:
# 0    284315
# 1       492
# Name: Class, dtype: int64
```
As you can see, there are far more `0`s (legit) than `1`s (fraud).

### The Workflow

1.  **Data Splitting:** We split the data into a training set and a testing set.
    ```python
    X = df.drop(columns=['Class'])
    y = df['Class']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

2.  **Model Training:** We train two different ensemble models:
    *   A `RandomForestClassifier`.
    *   A `GradientBoostingClassifier`.

    ```python
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)

    # Train Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(x_train, y_train)
    ```

3.  **Evaluation:** We evaluate both models on the test set using a few key metrics:
    *   **Accuracy:** The proportion of correctly classified transactions.
    *   **F1 Score:** The harmonic mean of precision and recall. This is often a better metric than accuracy for imbalanced datasets.
    *   **ROC AUC Score:** A measure of the model's ability to distinguish between classes.

### Performance Comparison

After training and evaluating both models, we get the following performance report:

| Model                | Accuracy | F1 Score | ROC AUC Score |
| -------------------- | -------- | -------- | ------------- |
| **Random Forest**    | 1.000    | 1.000    | 0.957         |
| **Gradient Boosting**| 0.998    | 0.999    | 0.767         |

From this report, the Random Forest model appears to perform exceptionally well on this dataset.

## Small Project: Improving Fraud Detection with SMOTE

**Objective:** Properly handle the imbalanced nature of the credit card fraud dataset using SMOTE and evaluate the model's performance with more suitable metrics.

**Dataset:** Continue using the `creditcard.csv` dataset.

**Steps:**

1.  **Prepare the Data:**
    *   Load the data.
    *   The 'Amount' feature has a very wide range. It's a good practice to scale it. Use `StandardScaler` to scale the 'Amount' column. You can drop the 'Time' column as it's unlikely to be useful without further feature engineering.
    *   Split the data into training and testing sets as before.

2.  **Apply SMOTE (Synthetic Minority Over-sampling TEchnique):**
    *   Use the `imblearn` library (you might need to `pip install imblearn`).
    *   Create a `SMOTE` object.
    *   **Important:** Apply SMOTE **only to the training data**. You want to train your model on a balanced dataset, but test it on the original, imbalanced test set to see how it performs in a real-world scenario.
    *   Use `smote.fit_resample(x_train, y_train)` to get your new, balanced training set.

3.  **Train a Model:**
    *   Train a `RandomForestClassifier` on the **new, resampled training data**.

4.  **Evaluate with Better Metrics:**
    *   Make predictions on the **original `x_test`**.
    *   Accuracy is misleading here. Instead, focus on:
        *   **`classification_report`:** Pay close attention to the **recall** for class 1 (fraud). In fraud detection, it's very important to catch as many frauds as possible.
        *   **`confusion_matrix`:** How many frauds did you miss (False Negatives)?
        *   **Precision-Recall Curve:** This is a much better visualization than the ROC curve for imbalanced datasets. Plot the Precision-Recall curve and calculate the **Average Precision Score** (`average_precision_score`).

**Key Takeaway:** This project will teach you the correct way to handle imbalanced datasets. You will learn that simply training on imbalanced data can lead to misleading results, and that techniques like SMOTE combined with appropriate evaluation metrics are essential for building effective models for real-world problems like fraud detection.

---

### <mark>Things I Didn't Go Through (But You Should Explore)</mark>

*   **Handling Imbalanced Data:** The notebook doesn't explicitly handle the class imbalance. While the models performed well, in many cases, you need to use techniques to address imbalance. You could explore:
    *   **Oversampling the minority class:** Using techniques like **SMOTE** (Synthetic Minority Over-sampling TEchnique) to create more examples of the fraudulent transactions.
    *   **Undersampling the majority class:** Randomly removing examples from the legitimate transactions.
    *   **Using different evaluation metrics:** For imbalanced data, accuracy can be misleading. Metrics like **Precision-Recall AUC** are often more informative.
*   **Feature Importance:** Both Random Forest and Gradient Boosting models can provide information about which features were most important for making predictions. You could visualize the feature importances to understand what factors are most indicative of fraud.
*   **Hyperparameter Tuning:** We used the default parameters for our models. You could likely improve performance further by using techniques like `GridSearchCV` or `RandomizedSearchCV` to find the optimal hyperparameters for each model.
