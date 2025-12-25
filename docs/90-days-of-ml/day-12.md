---
id: day-12
title: 'Day 12: A Deep Dive into Ensemble Learning'
---

## Day 12: A Deep Dive into Ensemble Learning

### A Quick Recap

In Day 11, we used ensemble models like Random Forest and Gradient Boosting. Today, we'll take a closer look at the theory behind these techniques and explore three main types of ensemble learning: **Bagging**, **Boosting**, and **Stacking**.

We'll be using the `heart.csv` dataset to predict the presence of heart disease.

### What is Ensemble Learning?

Ensemble learning is a technique where you combine the predictions of multiple machine learning models to produce a more accurate and robust prediction than any single model.

### 1. Bagging (Bootstrap Aggregating)

*   **Key Idea:** Train multiple models (usually of the same type) in parallel on different random subsets of the training data. The subsets are created using **bootstrap sampling** (sampling with replacement).
*   **Goal:** To reduce the variance of the model and prevent overfitting.
*   **Popular Examples:** `BaggingClassifier`, `RandomForestClassifier`.

#### Code Example: `BaggingClassifier`

Here, we create a bagging classifier that uses 100 Decision Tree models.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Train BaggingClassifier with Decision Trees
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)
bagging_model.fit(x_train, y_train)
y_pred = bagging_model.predict(x_test)
print(f'Accuracy Score: {accuracy_score(y_pred, y_test)}')
# Accuracy Score: 0.985
```

### 2. Boosting

*   **Key Idea:** Train multiple models sequentially. Each new model tries to correct the errors made by the previous models. It does this by paying more attention to the training instances that were misclassified by the previous models.
*   **Goal:** To reduce the bias of the model and achieve higher accuracy.
*   **Popular Examples:** `GradientBoostingClassifier`, `AdaBoostClassifier`, `XGBoost`, `LightGBM`.

#### Code Example: `GradientBoostingClassifier` and `AdaBoostClassifier`

```python
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

# Train Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(x_train, y_train)
# ... (evaluation)

# Train AdaBoost
ada_model = AdaBoostClassifier(estimator=RandomForestClassifier(max_depth=10), n_estimators=50, learning_rate=0.1, random_state=42)
ada_model.fit(x_train, y_train)
# ... (evaluation)
```

### 3. Stacking

*   **Key Idea:** Train multiple different models (called "base learners") on the same data. Then, train a final "meta-learner" model that takes the predictions of the base learners as input and learns to make the final prediction.
*   **Goal:** To combine the strengths of different types of models.
*   **Popular Example:** `StackingClassifier`.

#### Code Example: `StackingClassifier`

Here, we use a Random Forest and a Decision Tree as our base learners, and a Gradient Boosting model as our meta-learner.

```python
from sklearn.ensemble import StackingClassifier

stacking_model = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('dt', DecisionTreeClassifier(max_depth=2, random_state=42))
    ],
    final_estimator=GradientBoostingClassifier(n_estimators=100, random_state=42)
)
stacking_model.fit(x_train, y_train)
# ... (evaluation)
```

### Cross-Validation

To get a more reliable estimate of our model's performance, we can use **cross-validation**. This involves splitting the training data into multiple "folds" and training and evaluating the model multiple times, using a different fold as the validation set each time.

```python
from sklearn.model_selection import cross_val_score

# Perform 10-fold cross-validation on the stacking model
scores = cross_val_score(stacking_model, x_train, y_train, cv=10, scoring='f1')
print(f'Cross Validation F1 Scores: {scores}')
```

## Small Project: Predicting Employee Attrition

**Objective:** Apply and compare Bagging, Boosting, and Stacking to predict whether an employee will leave a company.

**Dataset:** The [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-performance) dataset on Kaggle is a classic classification problem.

**Steps:**

1.  **Load and Prepare the Data:**
    *   Load the dataset.
    *   You will need to encode several categorical features. `LabelEncoder` can be used for binary features, and `pd.get_dummies` for multi-class features.
    *   The dataset is fairly clean, but make sure all features are numeric before training.

2.  **Train-Test Split:** Split the data into training and testing sets.

3.  **Train and Compare Ensemble Models:**
    *   **Bagging:** Train a `BaggingClassifier` with `DecisionTreeClassifier` as the base estimator.
    *   **Boosting:** Train an `XGBClassifier` (from the `xgboost` library). `XGBoost` is a powerful and popular boosting algorithm.
    *   **Stacking:** Train a `StackingClassifier`.
        *   For your `estimators`, try a combination of a `LogisticRegression` and a `RandomForestClassifier`.
        *   For your `final_estimator`, you could use a `DecisionTreeClassifier`.

4.  **Evaluate the Models:**
    *   For each of the three models (Bagging, XGBoost, Stacking), evaluate its performance on the test set.
    *   Use `accuracy_score` and `classification_report` to compare them. Which ensemble method performs the best for this problem?

5.  **Cross-Validation:**
    *   For your best performing model, use `cross_val_score` with `cv=5` to get a more robust estimate of its performance.

**Key Takeaway:** This project will give you hands-on experience in implementing and comparing the three main types of ensemble learning on a single problem. You will learn how to build a stacking pipeline and get to use `XGBoost`, one of the most widely used machine learning algorithms.

---

### <mark>Things I Didn't Go Through (But You Should Explore)</mark>

*   **XGBoost and LightGBM:** The notebook imports libraries for XGBoost and LightGBM but doesn't use them. These are highly optimized and popular implementations of gradient boosting that are often much faster and more accurate than scikit-learn's `GradientBoostingClassifier`. You should try implementing them and comparing their performance.
*   **Hyperparameter Tuning:** For each of these ensemble methods, there are many important hyperparameters to tune (e.g., `n_estimators`, `learning_rate`, `max_depth`). Using techniques like `GridSearchCV` can help you find the best combination of parameters for your problem.
*   **Feature Importance:** Most ensemble models can provide insights into which features were most important for making predictions. This is a crucial step for understanding your model and your data.
