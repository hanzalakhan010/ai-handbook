---
sidebar_position: 11
---

# Day 8: Ensemble Methods for Classification

On my eighth day, I dove into the world of ensemble methods for classification. Ensemble methods combine the predictions of multiple models to produce a more accurate and robust prediction.

I focused on three popular ensemble methods:

*   **Random Forest:** An ensemble of decision trees that are trained on different subsets of the data.
*   **AdaBoost:** A boosting algorithm that sequentially trains a series of weak learners, where each learner focuses on the mistakes of the previous one.
*   **Gradient Boosting:** Another boosting algorithm that builds trees sequentially, where each tree tries to correct the errors of the previous one.

## Model Comparison on the Iris Dataset

I started by comparing the performance of these three models on the Iris dataset. I evaluated each model using accuracy, F1-score, precision, and recall.

```python
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Creating and evaluating the models
# ... (code for training and evaluating each model)
```

The performance of the three models on the Iris dataset was very similar. To better evaluate their performance, I decided to create a custom imbalanced dataset.

## Model Comparison on an Imbalanced Dataset

I created a custom imbalanced dataset using `make_classification` with a 90/10 class distribution. This is a more realistic scenario for many real-world classification problems.

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=2,
    n_redundant=10,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],  # Class imbalance (90% class 0, 10% class 1)
    flip_y=0,
    random_state=42
)
```

I then trained and evaluated the three models on this imbalanced dataset.

```python
# Creating and evaluating the models on the imbalanced dataset
# ... (code for training and evaluating each model)
```

## Reflections

Day 8 was a great introduction to ensemble methods. I learned how these methods can be used to improve the performance of classification models. I also learned the importance of evaluating models on imbalanced datasets, as accuracy can be a misleading metric in these cases.

I'm excited to continue exploring ensemble methods and to apply them to more complex classification problems.

## Small Project: Credit Card Fraud Detection

**Objective:** Build and compare several ensemble models to detect fraudulent credit card transactions. This is a classic example of an imbalanced classification problem.

**Dataset:** The [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset on Kaggle is a great dataset for this task. It contains transactions made by European cardholders, and the classes are highly imbalanced.

**Steps:**

1.  **Load and Prepare the Data:**
    *   Load the `creditcard.csv` data.
    *   The data is already anonymized and scaled. The only features that are not scaled are 'Time' and 'Amount'. It's a good practice to scale 'Amount' using `StandardScaler` or `RobustScaler` (which is less sensitive to outliers). You can probably drop the 'Time' feature for this exercise.
    *   The target variable is 'Class', where 1 represents a fraudulent transaction.

2.  **Address the Imbalance (Sampling):**
    *   Because the dataset is highly imbalanced, training a model on the raw data might lead to poor performance on the minority class (fraud).
    *   Before training, you should resample the data. A common technique is **undersampling**, where you randomly remove samples from the majority class. You can use the `imblearn` library for this, specifically `RandomUnderSampler`.
    *   **Important:** Apply undersampling only to the **training data**, not the test data.

3.  **Train-Test Split:** Split your original (pre-sampling) data into training and testing sets. Then apply the undersampling to the `x_train` and `y_train`.

4.  **Model Training and Comparison:**
    *   Train the following ensemble models on the **resampled training data**:
        *   `RandomForestClassifier`
        *   `GradientBoostingClassifier`
        *   `XGBClassifier` (from the `xgboost` library, you might need to `pip install xgboost`)
    *   `XGBoost` is a very popular and powerful implementation of gradient boosting that often performs better than the one in scikit-learn.

5.  **Model Evaluation (on the original Test Set):**
    *   Evaluate your models on the **original, imbalanced test set**.
    *   **Do not use accuracy** as your primary metric. It will be misleadingly high.
    *   Focus on the **`classification_report`** and the **`confusion_matrix`**.
    *   Pay close attention to the **recall** for the minority class (class 1). In fraud detection, it's very important to catch as many fraudulent transactions as possible (high recall).
    *   Another great metric for imbalanced classification is the **Area Under the Precision-Recall Curve (AUPRC)**. You can calculate it using `average_precision_score`.

**Key Takeaway:** This project will give you practical experience with a real-world imbalanced classification problem. You will learn how to handle imbalanced data using sampling techniques, how to use a powerful new library (`xgboost`), and how to choose the right evaluation metrics for imbalanced classes.
