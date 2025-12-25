---
sidebar_position: 10
---

# Day 7 (Part 3): Regularized Linear Regression

In the third part of my seventh day, I started exploring regularized linear regression models. These models are useful for preventing overfitting, especially when dealing with datasets that have a large number of features.

I focused on three types of regularized linear regression:

*   **Ridge Regression:** This model adds a penalty to the loss function that is proportional to the square of the magnitude of the coefficients.
*   **Lasso Regression:** This model adds a penalty to the loss function that is proportional to the absolute value of the magnitude of the coefficients. This can lead to some coefficients being set to zero, which can be useful for feature selection.
*   **ElasticNet Regression:** This model is a combination of Ridge and Lasso regression.

## Ridge Regression

I started by experimenting with Ridge regression on the California Housing dataset. I trained a Ridge model with different values of the regularization parameter `alpha` and evaluated its performance using Mean Absolute Error (MAE).

### 1. Data Preparation

I loaded the dataset and split it into training and testing sets.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housings = fetch_california_housing(as_frame=True)
housings = housings.frame

X = housings.drop(columns=['MedHouseVal'])
y = housings['MedHouseVal']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### 2. Ridge Regression with Different Alphas

I then trained a Ridge model with different values of `alpha` and printed the MAE for each value.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import numpy as np

for i in np.linspace(0, 10, num=10):
    ridgeModel = Ridge(alpha=i)
    ridgeModel.fit(x_train, y_train)
    y_pred = ridgeModel.predict(x_test)
    print(f'Alpha = {i}')
    print(f'mean_absolute_error = {mean_absolute_error(y_pred, y_test)}')
```

## Future Work

This notebook is incomplete. I plan to continue this exercise by:

*   Implementing Lasso and ElasticNet regression.
*   Comparing the performance of the regularized models with a simple linear regression model.
*   Visualizing the effect of the regularization parameter `alpha` on the model coefficients.

## Reflections

This was a good introduction to regularized linear regression. I'm starting to understand how these models can be used to prevent overfitting and improve the generalization performance of a model.

## Small Project: Ridge vs. Lasso vs. ElasticNet

**Objective:** Compare the effects of Ridge, Lasso, and ElasticNet regularization on a linear model, and visualize how they affect model coefficients.

**Dataset:** You can continue using the California Housing dataset as you have already started.

**Steps:**

1.  **Scale your Features:**
    *   It is **very important** to scale your features before using regularized regression models. The penalty is applied to the coefficients, and if features are on different scales, the penalty will affect them differently.
    *   Use `StandardScaler` to scale your `x_train` and `x_test` data.

2.  **Train and Evaluate Models:**
    *   Train four different models on the **scaled** training data:
        *   `LinearRegression` (as a baseline)
        *   `Ridge` (try `alpha=1.0`)
        *   `Lasso` (try `alpha=1.0`)
        *   `ElasticNet` (try `alpha=1.0` and `l1_ratio=0.5`)
    *   For each model, calculate the Mean Absolute Error (MAE) or Mean Squared Error (MSE) on the test set. How do they compare to the baseline `LinearRegression`?

3.  **Visualize the Effect of Alpha on Coefficients (Lasso):**
    *   The most interesting property of Lasso is that it can shrink coefficients to exactly zero. Let's visualize this.
    *   Create a loop that trains a `Lasso` model for different values of `alpha` (e.g., from `0.001` to `10.0` on a log scale).
    *   In the loop, store the coefficients (`model.coef_`) for each `alpha`.
    *   After the loop, create a plot where the y-axis is the coefficient value and the x-axis is the `alpha` value. Each line on the plot will represent a feature's coefficient. You should see the coefficients getting smaller and eventually becoming zero as `alpha` increases.

4.  **Visualize the Effect of Alpha on Coefficients (Ridge):**
    *   Repeat the same process as in step 3, but this time for a `Ridge` model.
    *   What is the main difference you observe in the plot compared to the Lasso plot? (Hint: Do the coefficients ever become exactly zero?)

5.  **Analyze the Lasso Coefficients:**
    *   Train a `Lasso` model with a moderate `alpha` (e.g., `alpha=0.1`).
    *   Print the coefficients of the trained model.
    *   How many of the coefficients are zero? This is a form of automatic feature selection. The features with non-zero coefficients are the ones the model considers most important.

**Key Takeaway:** This project will give you a deep, practical understanding of how regularization works. You will learn the key difference between Ridge and Lasso, see how Lasso can be used for feature selection, and understand the importance of scaling features when using these models.
