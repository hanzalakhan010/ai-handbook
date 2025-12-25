---
sidebar_position: 12
---

# Day 9: Advanced Classification with Ensemble Methods

On my ninth day, I applied the ensemble techniques I learned on Day 8 to a real-world dataset: the Heart Disease Dataset from UCI. This was a great opportunity to practice my skills and gain more experience with advanced classification models.

## The Machine Learning Workflow

I followed a complete machine learning workflow to train and evaluate two ensemble models: **Random Forest** and **Gradient Boosting**.

### 1. Data Loading and Preprocessing

I started by loading the dataset and checking for any missing values or duplicates. The dataset was clean, so no major preprocessing was needed.

```python
import pandas as pd

df = pd.read_csv('./datasets/heart.csv')
```

### 2. Train-Test Split

I then split the data into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['target'])
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. Model Training and Evaluation

I trained both a `RandomForestClassifier` and a `GradientBoostingClassifier` on the training data. I evaluated each model using a comprehensive set of metrics:

*   **Accuracy:** The proportion of correctly classified instances.
*   **F1 Score:** The harmonic mean of precision and recall.
*   **ROC-AUC Score:** The area under the ROC curve, which measures the model's ability to distinguish between classes.
*   **Recall Score:** The proportion of actual positives that were correctly identified.
*   **Precision Score:** The proportion of positive predictions that were actually correct.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, recall_score, precision_score

# ... (code for training and evaluating each model)
```

### 4. ROC Curve Visualization

I plotted the ROC curves for both models to visualize their performance.

```python
import matplotlib.pyplot as plt

# ... (code for plotting the ROC curves)
```

## Performance Report

Here is a summary of the performance of the two models:

- **Random Forest**
    - Accuracy : 0.985
    - F1 Score : 0.985
    - ROC AUC Score : 0.986
    - Recall Score : 1.0
    - Precision Score : 0.971
- **Gradient Boosting**
    - Accuracy : 0.932
    - F1 Score : 0.932
    - ROC AUC Score : 0.932
    - Recall Score : 0.916
    - Precision Score : 0.951

As you can see, the **Random Forest** model performed better across all metrics.

## Reflections

Day 9 was a great exercise in applying and comparing different classification models on a real-world dataset. I'm now more comfortable with using a variety of evaluation metrics to assess model performance. I'm also starting to get a better understanding of the strengths and weaknesses of different ensemble methods.

## Small Project: Customer Churn Prediction

**Objective:** Predict whether a customer will churn (leave a service) using advanced ensemble methods. This is a common and important business problem.

**Dataset:** The [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) dataset from Kaggle is perfect for this task. It contains information about a fictional telco company's customers and whether they churned or not.

**Steps:**

1.  **Load and Prepare the Data:**
    *   Load the `WA_Fn-UseC_-Telco-Customer-Churn.csv` data.
    *   The 'TotalCharges' column has missing values for new customers. You'll need to handle these. A reasonable approach is to set them to 0, or to the 'MonthlyCharges' value. You might also notice that 'TotalCharges' is an object type. You'll need to convert it to a numeric type.
    *   You'll need to encode the categorical features. For binary features (like 'gender', 'Partner', 'Dependents'), you can use `LabelEncoder`. For multi-class categorical features (like 'InternetService', 'Contract'), `pd.get_dummies` is a good choice.
    *   The 'customerID' column should be dropped.

2.  **Train-Test Split:** Split your data into training and testing sets.

3.  **Model Training and Comparison:**
    *   Train the following advanced ensemble models:
        *   `RandomForestClassifier`
        *   `XGBClassifier` (from the `xgboost` library)
        *   `LGBMClassifier` (from the `lightgbm` library, you might need to `pip install lightgbm`)
    *   `LightGBM` is another gradient boosting framework that is known for its speed and efficiency, especially on large datasets.

4.  **Model Evaluation:**
    *   This is a slightly imbalanced dataset, so it's a good idea to use a variety of metrics.
    *   For each model, evaluate its performance on the test set using:
        *   `accuracy_score`
        *   `roc_auc_score`
        *   `classification_report` (pay attention to precision and recall for the 'Yes' churn class)
        *   `confusion_matrix`

5.  **Hyperparameter Tuning:**
    *   Choose your best performing model (it will likely be `XGBoost` or `LightGBM`) and use `RandomizedSearchCV` or `GridSearchCV` to find the best hyperparameters.
    *   For these models, important parameters to tune include `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`.

6.  **Analyze Feature Importance:**
    *   For your final tuned model, inspect the `feature_importances_`.
    *   What are the top factors that lead to customer churn? Is it the contract type? The monthly charges?

**Key Takeaway:** This project will introduce you to two of the most popular and powerful machine learning libraries in use today: `XGBoost` and `LightGBM`. You will gain experience in solving a real-world business problem and learn how to interpret your model to provide actionable insights.
