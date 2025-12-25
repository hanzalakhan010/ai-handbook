---
sidebar_position: 7
---

# Day 6: Introduction to Supervised Learning

On my sixth day, I started my journey into supervised learning, one of the two main types of machine learning. Supervised learning involves training a model on a labeled dataset, where the correct output is known for each input.

I explored two types of supervised learning tasks:

*   **Regression:** Predicting a continuous output (e.g., house prices, temperatures).
*   **Classification:** Predicting a categorical outcome (e.g., spam email detection, loan approval).

## Regression: Predicting House Prices

I started with a regression task: predicting house prices using the "USA Housing Dataset." I used a linear regression model for this task.

### 1. Data Preparation

First, I loaded the dataset and dropped the categorical columns that I wouldn't be using for this initial model.

```python
import pandas as pd

housings = pd.read_csv('./datasets/USA_Housing.csv')
housings.drop(columns=['Date', 'Address', 'Street', 'City', 'State', 'Zip'], axis=1, inplace=True)
```

### 2. Train-Test Split

Next, I split the data into training and testing sets. The training set is used to train the model, and the testing set is used to evaluate its performance.

```python
from sklearn.model_selection import train_test_split

y = housings.price
X = housings.drop(columns=['price'])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### 3. Model Training and Evaluation

Finally, I trained a linear regression model on the training data and evaluated its performance on the testing data using the mean squared error (MSE) metric.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
```

## Classification: Kickstarter Project Success

I then moved on to a classification task: predicting whether a Kickstarter project will be successful. I started by exploring the "Kickstarter Projects" dataset.

My goal is to build a logistic regression model to predict the 'state' of a project (e.g., 'successful', 'failed', 'canceled'). I started by loading the data and identifying the unique values in the 'state' column.

```python
import pandas as pd

df = pd.read_csv('./datasets/ks-projects-201801.csv')
df['state'].unique()
```

I will continue working on this classification problem in the coming days.

## Reflections

Day 6 was a great introduction to supervised learning. I'm excited to have built my first machine learning model, and I'm looking forward to diving deeper into both regression and classification in the coming days.

## Small Project: Your First Supervised Learning Models

**Objective:** Build and evaluate your first regression and classification models on classic datasets.

---

### Part 1: Regression with the Boston Housing Dataset

**Dataset:** The Boston Housing dataset is a classic dataset for regression tasks. It's available in `sklearn.datasets`.

**Steps:**

1.  **Load the dataset:** Load the Boston Housing dataset from `sklearn.datasets`.
2.  **Data Preparation:** The dataset is already clean, so you can directly proceed to the next step.
3.  **Train-Test Split:** Split the data into training and testing sets.
4.  **Model Training:** Train a `LinearRegression` model from `scikit-learn` on the training data.
5.  **Model Evaluation:**
    *   Make predictions on the test set.
    *   Evaluate your model using metrics like Mean Squared Error (MSE) and R-squared (`r2_score`).
    *   Print the coefficients of your model (`model.coef_`). What do they tell you about the importance of each feature?

---

### Part 2: Classification with the Breast Cancer Wisconsin Dataset

**Dataset:** The Breast Cancer Wisconsin dataset is a classic binary classification dataset, also available in `sklearn.datasets`. The goal is to predict whether a tumor is malignant or benign.

**Steps:**

1.  **Load the dataset:** Load the Breast Cancer dataset from `sklearn.datasets`.
2.  **Data Preparation:** The data is clean. However, it's good practice to scale your data before feeding it to a logistic regression model. Use `StandardScaler` to scale the features.
3.  **Train-Test Split:** Split the scaled data into training and testing sets.
4.  **Model Training:** Train a `LogisticRegression` model on the training data.
5.  **Model Evaluation:**
    *   Make predictions on the test set.
    *   Evaluate your model using metrics like:
        *   **Accuracy Score:** The proportion of correct predictions.
        *   **Confusion Matrix:** To see the number of true positives, true negatives, false positives, and false negatives.
        *   **Classification Report:** Which provides precision, recall, and F1-score for each class.

**Key Takeaway:** This project will give you hands-on experience in building and evaluating both regression and classification models using `scikit-learn`. You will learn the basic workflow for supervised learning tasks and how to interpret common evaluation metrics.
