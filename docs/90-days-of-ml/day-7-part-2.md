---
sidebar_position: 9
---

# Day 7 (Part 2): Classification with Decision Trees

In the second part of my seventh day, I continued my journey into supervised learning by building a classification model. I used the preprocessed Titanic dataset from Day 4 to predict whether a passenger survived or not.

## The Machine Learning Workflow

I followed a standard machine learning workflow for classification.

### 1. Data Loading and Cleaning

I started by loading the preprocessed Titanic dataset and dropping the columns that were not needed for the model.

```python
import pandas as pd

df = pd.read_csv('./datasets/filtered_datasets/titanic.csv')
df.drop(columns=['Name', 'Unnamed: 0', 'PassengerId', 'Ticket'], inplace=True)
```

### 2. Feature and Target Separation

I then separated the data into features (X) and the target variable (y), which is 'Survived'.

```python
X = df.drop(columns=['Survived'], axis=0)
y = df['Survived']
```

### 3. Train-Test Split

I split the data into training and testing sets to train and evaluate the model.

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
```

### 4. Model Training and Evaluation

I trained a `DecisionTreeClassifier` on the training data. I chose a decision tree because it is an interpretable model that is easy to understand.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
```

I then evaluated the model's performance using accuracy and a classification report. The classification report provides more detailed metrics, such as precision, recall, and f1-score.

```python
print('Classification Report:')
print(classification_report(y_pred, y_test))
```

### 5. Hyperparameter Tuning (Future Work)

I noted that I could use `GridSearchCV` to find the optimal hyperparameters for the decision tree, but I deferred this to a later time due to the computational resources required.

## Reflections

This was a great exercise in building a classification model from start to finish. I'm starting to get a good feel for the machine learning workflow, and I'm excited to explore more advanced classification models in the future.

## Small Project: Classifying Penguin Species

**Objective:** Build, evaluate, and compare different classification models to predict the species of a penguin from the Palmer Penguins dataset.

**Dataset:** The [Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/) is a wonderful dataset for classification tasks. You can load it easily using `seaborn.load_dataset('penguins')`.

**Steps:**

1.  **Load and Prepare the Data:**
    *   Load the penguins dataset.
    *   Handle the few missing values. Dropping the rows with missing data is a reasonable approach here.
    *   The 'sex' column has one missing value that you might need to handle separately if you choose to use it. For simplicity, you can start by dropping this column.
    *   The features are numerical, so you're almost ready to go. For models like Logistic Regression, it's a good practice to scale the features using `StandardScaler`.
    *   The target variable is 'species'.

2.  **Train-Test Split:** Split your data into training and testing sets.

3.  **Model Training and Comparison:**
    *   Train the following classification models:
        *   `LogisticRegression`
        *   `DecisionTreeClassifier`
        *   `RandomForestClassifier`
        *   `SVC` (Support Vector Classifier)
    *   For each model, evaluate its performance on the test set using:
        *   `accuracy_score`
        *   `classification_report` (to see precision, recall, f1-score)
        *   `confusion_matrix`
    *   Compare the performance of the models. Which one performs best?

4.  **Visualize the Decision Tree:**
    *   For the `DecisionTreeClassifier` you trained, you can visualize the tree to understand how it makes decisions.
    *   Use `sklearn.tree.plot_tree` to create a visual representation of your decision tree. This is a powerful way to interpret the model.

    ```python
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    plt.figure(figsize=(20,10))
    plot_tree(your_decision_tree_model, feature_names=your_feature_names, class_names=your_class_names, filled=True)
    plt.show()
    ```

5.  **Hyperparameter Tuning:**
    *   Choose your best performing model and use `GridSearchCV` to find the optimal hyperparameters. For a `RandomForestClassifier`, you could tune `n_estimators` and `max_depth`.

**Key Takeaway:** This project will give you experience in comparing different classification algorithms and choosing the best one for a given task. You will also learn a powerful and important technique for model interpretability: visualizing a decision tree.
