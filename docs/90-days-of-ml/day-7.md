---
sidebar_position: 8
---

# Day 7: Advanced Regression Models

On my seventh day, I delved deeper into regression by exploring more advanced models. I used the California Housing dataset for this exercise, which is a classic dataset for regression tasks.

## The Machine Learning Workflow

I followed a complete machine learning workflow, from data preparation to model evaluation and hyperparameter tuning.

### 1. Data Preparation and Scaling

I started by loading the dataset and splitting it into training and testing sets. I then scaled the features using `StandardScaler` to ensure that all features had a mean of 0 and a standard deviation of 1.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features and target
X = data.drop('MedHouseVal', axis=1)
y = data.MedHouseVal

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# Normalizing the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

**Important Note:** It's crucial to use `fit_transform` on the training data and `transform` on the testing data to prevent data leakage.

### 2. Model Training and Evaluation

I trained and evaluated four different regression models:

*   **Linear Regression:** A simple and interpretable model.
*   **Decision Tree Regressor:** A tree-based model that can capture non-linear relationships.
*   **Random Forest Regressor:** An ensemble of decision trees that is more robust than a single decision tree.
*   **Gradient Boosting Regressor:** Another powerful ensemble method that builds trees sequentially.

I evaluated each model using Mean Squared Error (MSE) and R-squared (R²).

```python
# Evaluate each model
for model_name, y_pred in zip(
    ['Decision Tree', 'Random Forest', 'Gradient Boosting'],
    [dT_model_pred, rF_model_pred, gB_Model_pred]
):
    print(f"{model_name} - MSE: {mean_squared_error(y_test, y_pred)}, R²: {r2_score(y_test, y_pred)}")
```

### 3. Cross-Validation

To get a more robust estimate of the models' performance, I used 5-fold cross-validation.

```python
from sklearn.model_selection import cross_val_score

models = [dT_model, rF_model, gB_Model]
model_names = ['Decision Tree', 'Random Forest', 'Gradient Boosting']
for model, name in zip(models, model_names):
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"{name} - Average MSE: {-scores.mean()}")
```

### 4. Hyperparameter Tuning

I used `GridSearchCV` to find the best hyperparameters for the `RandomForestRegressor`.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 200],
    'max_depth': [None, 20],
    'min_samples_split': [6, 10]
}
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=2, scoring='neg_mean_squared_error')
grid_search.fit(X, y)
print(f"Best parameters: {grid_search.best_params_}")
```

### 5. Visualization

Finally, I visualized the performance of the best model by plotting the true vs. predicted house prices.

```python
best_model = grid_search.best_estimator_
y_best_pred = best_model.predict(x_test)

plt.scatter(y_test, y_best_pred, alpha=0.7)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs. Predicted House Prices")
plt.show()
```

## Reflections

Day 7 was a fantastic learning experience. I now have a solid understanding of how to build and evaluate different regression models. I also learned the importance of cross-validation and hyperparameter tuning for building robust and high-performing models.

## Small Project: Comparing Regression Models for Bike Sharing Demand

**Objective:** Build, compare, and tune several regression models to predict the demand for bike sharing.

**Dataset:** The [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand/data) dataset from Kaggle is a popular dataset for regression and feature engineering. The goal is to predict the total number of bikes rented in a given hour.

**Steps:**

1.  **Load and Prepare the Data:**
    *   Load the `train.csv` data.
    *   **Feature Engineering:** The 'datetime' column is the most important one. Create new features from it, such as 'hour', 'dayofweek', 'month', and 'year'.
    *   The features 'season', 'holiday', 'workingday', and 'weather' are categorical. Decide how to handle them. Since they are already encoded as numbers, you could treat them as is for tree-based models, but you might want to one-hot encode them for linear models.
    *   The target variable is 'count'. Notice that 'casual' and 'registered' add up to 'count'. You should drop 'casual' and 'registered' from your training data, as they won't be available in the test set.

2.  **Train-Test Split:** Split your data into training and testing sets.

3.  **Model Training and Comparison:**
    *   Train the following regression models:
        *   `LinearRegression`
        *   `Ridge` (a regularized linear model)
        *   `DecisionTreeRegressor`
        *   `RandomForestRegressor`
        *   `GradientBoostingRegressor`
    *   For each model, calculate the Root Mean Squared Log Error (RMSLE), which is the evaluation metric for this Kaggle competition. You can calculate it as `np.sqrt(mean_squared_log_error(y_true, y_pred))`.
    *   Compare the performance of the models. Which one performs best out of the box?

4.  **Cross-Validation:**
    *   Use 5-fold cross-validation to get a more reliable performance estimate for your top 2-3 models.

5.  **Hyperparameter Tuning:**
    *   Choose your best performing model (likely `RandomForestRegressor` or `GradientBoostingRegressor`) and use `GridSearchCV` or `RandomizedSearchCV` to find the best hyperparameters.
    *   Focus on tuning parameters like `n_estimators`, `max_depth`, and `learning_rate` (for Gradient Boosting).

6.  **Analyze Feature Importance:**
    *   For your final tuned model, inspect the `feature_importances_` attribute.
    *   What are the most important features for predicting bike sharing demand? Does 'hour' play a big role? What about the weather?

**Key Takeaway:** This project provides a complete workflow for a real-world regression problem. You will practice feature engineering, model comparison, hyperparameter tuning, and interpreting model results. You will also learn about a new evaluation metric (RMSLE) which is common for skewed target variables.
