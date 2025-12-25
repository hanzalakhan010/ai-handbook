---
sidebar_position: 13
---

# Day 10: Loan Approval Prediction with Random Forest

On my tenth day, I built a `RandomForestClassifier` to predict loan approval. This was a great opportunity to apply my knowledge of ensemble methods to a real-world problem.

## The Machine Learning Workflow

I followed a complete machine learning workflow, from data cleaning to model evaluation and feature importance visualization.

### 1. Data Loading and Cleaning

I started by loading the loan approval dataset and encoding the categorical features using `LabelEncoder`.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./datasets/loan_approval_dataset.csv')

# Encoding categorical features
encoder = LabelEncoder()
df[' education'] = encoder.fit_transform(df[' education'])
df[' self_employed'] = encoder.fit_transform(df[' self_employed'])
df[' loan_status'] = encoder.fit_transform(df[' loan_status'])
```

### 2. Train-Test Split

I then split the data into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=[' loan_status'])
y = df[' loan_status']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. Model Training and Evaluation

I trained a `RandomForestClassifier` on the training data and evaluated its performance using a comprehensive set of metrics.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, recall_score, precision_score

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

y_pred = rf_model.predict(x_test)

# ... (code for calculating performance metrics)
```

### 4. ROC Curve Visualization

I plotted the ROC curve for the model to visualize its performance.

```python
import matplotlib.pyplot as plt

# ... (code for plotting the ROC curve)
```

### 5. Feature Importance Visualization

I visualized the feature importances of the model to understand which features were most important for predicting loan approval.

```python
# ... (code for plotting feature importances)
```

## Performance Report

Here is a summary of the model's performance:

- **Random Forest**
    - Accuracy : 0.977
    - F1 Score : 0.977
    - ROC AUC Score : 0.976
    - Recall Score : 0.972
    - Precision Score : 0.965

## Reflections

Day 10 was a great exercise in building and evaluating a classification model. I'm now more comfortable with the entire machine learning workflow, from data cleaning to model evaluation and interpretation. I'm also starting to appreciate the power of ensemble methods for building high-performing models.

## Small Project: The Business Impact of Loan Default Prediction

**Objective:** Go beyond simple accuracy metrics and evaluate your loan prediction model based on its potential business impact. This involves understanding the costs of different types of model errors.

**Dataset:** You can use the same Loan Approval dataset or, for a more challenging task, the [Home Equity Default dataset](https://www.kaggle.com/datasets/ajaymanwani/home-equity-loan-default-dataset) from Kaggle. Let's stick with the one you've already used for simplicity.

**Scenario:**

Imagine you are a bank. When you approve a loan, there are two possible outcomes:
*   **The customer repays the loan:** You make a profit (e.g., from interest).
*   **The customer defaults on the loan:** You lose a significant amount of money (the principal).

When you reject a loan, there are two "what if" outcomes:
*   **The customer would have repaid:** You lose out on potential profit (opportunity cost).
*   **The customer would have defaulted:** You correctly avoided a loss.

**The Cost of Errors:**
The key insight is that the cost of a **False Negative** (predicting 'Approved' when they will actually 'Default') is much higher than the cost of a **False Positive** (predicting 'Default' when they would have 'Repaid').

**Steps:**

1.  **Define a Cost Matrix:**
    *   Let's assign some hypothetical costs to each outcome in the confusion matrix.
    *   **True Positive (Predicted Approved, Actually Approved):** Benefit = +$5,000 (e.g., interest profit)
    *   **True Negative (Predicted Rejected, Actually Rejected):** Benefit = $0 (you correctly avoided a loss)
    *   **False Positive (Predicted Rejected, Actually Approved):** Cost = -$5,000 (opportunity cost of lost profit)
    *   **False Negative (Predicted Approved, Actually Rejected):** Cost = -$100,000 (the average loss on a defaulted loan)

2.  **Re-train Your Model:**
    *   Train your `RandomForestClassifier` as you did before.

3.  **Evaluate with the Cost Matrix:**
    *   Make predictions on your test set.
    *   Get the `confusion_matrix` from your predictions.
    *   Now, create a function that takes your confusion matrix and the cost matrix and calculates the total profit or loss of your model on the test set.

    ```python
    import numpy as np
    
    # Assuming your cost matrix is a 2x2 numpy array
    # and your confusion matrix is also a 2x2 numpy array
    total_value = np.sum(confusion_matrix * cost_matrix)
    print(f"Total business value of the model: ${total_value:,.2f}")
    ```

4.  **Optimize for Business Value (Threshold Tuning):**
    *   A classification model doesn't just predict `0` or `1`. It predicts a *probability*. By default, the threshold is `0.5`. If the probability is > 0.5, it predicts `1`.
    *   You can change this threshold to make the model more or less conservative.
    *   To reduce the number of costly False Negatives, you might want to be more strict. What happens if you only approve loans where the model's predicted probability of being 'Approved' is very high (e.g., > 0.8)?
    *   Use `model.predict_proba(x_test)` to get the predicted probabilities.
    *   Create a loop that tries different thresholds (e.g., from 0.1 to 0.9). For each threshold, calculate the new confusion matrix and the new total business value.
    *   Plot the business value vs. the threshold. What threshold maximizes your profit?

**Key Takeaway:** This project will teach you a crucial lesson in applied machine learning: accuracy is often not the best metric. The best model is the one that provides the most value to the business. You will learn how to frame a machine learning problem in terms of its business impact and how to optimize a model for a specific business objective, not just a statistical one.
