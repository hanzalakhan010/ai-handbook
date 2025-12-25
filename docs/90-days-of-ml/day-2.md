---
sidebar_position: 3
---

# Day 2: Scaling and Normalization

On my second day, I delved into the crucial concepts of data scaling and normalization. These techniques are essential for preparing data for many machine learning algorithms, as they can help to improve model performance and stability.

## Key Learnings

### 1. Scaling vs. Normalization

I learned the difference between scaling and normalization:

*   **Scaling:** This involves transforming the data to a specific range, such as 0 to 1. This is useful when the features have different scales, which can cause issues for some algorithms.
*   **Normalization:** This involves transforming the data to have a mean of 0 and a standard deviation of 1. This is useful when the data has a Gaussian distribution and you want to use algorithms that assume this distribution.

### 2. Min-Max Scaling

I practiced min-max scaling, which scales the data to a fixed range (usually 0 to 1). I used the `mlxtend.preprocessing.minmax_scaling` function for this.

```python
# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# scaling the data
scaled_data = minmax_scaling(original_data, columns=['feature_name'])
```

### 3. Box-Cox Transformation

I also learned about the Box-Cox transformation, which is a powerful technique for transforming non-normally distributed data into a normal distribution. I used the `scipy.stats.boxcox` function for this.

```python
# for Box-Cox Transformation
from scipy import stats

# transforming the data
normalized_data = stats.boxcox(original_data)
```

## Pandas Practice

I also spent a significant amount of time practicing my `pandas` skills. I worked through a series of exercises that covered:

*   **Basic Slicing:** Selecting rows and columns.
*   **Conditional Filtering:** Filtering data based on conditions.
*   **Combination Filters:** Combining multiple conditions.
*   **Indexing:** Setting and resetting the index.
*   **Advanced Slicing:** Using `.loc` and `.iloc` for more complex selections.
*   **Groupby and Aggregations:** Grouping data and calculating aggregate statistics.
*   **Sorting:** Sorting data by one or more columns.

## Reflections

Day 2 was a deep dive into data preprocessing. I now have a much better understanding of scaling and normalization techniques and when to use them. I also feel more confident in my `pandas` skills, which will be invaluable for future projects.

## Small Project: Scaling and Normalization on the Iris Dataset

**Objective:** Apply scaling and normalization techniques to the Iris dataset and observe their effect on a simple classification model.

**Dataset:** The Iris dataset is a classic and is included in the `scikit-learn` library.

**Steps:**

1.  **Load the dataset:** Load the Iris dataset from `sklearn.datasets`.
2.  **Split the data:** Split the data into training and testing sets.
3.  **Train a baseline model:** Train a simple logistic regression model on the original (unscaled) training data and evaluate its performance on the test set.
4.  **Apply Min-Max Scaling:**
    *   Scale the training data using `MinMaxScaler` from `scikit-learn`.
    *   Train a new logistic regression model on the scaled data.
    *   Evaluate the model on the scaled test data and compare the performance to the baseline model.
5.  **Apply Standardization (Z-score normalization):**
    *   Scale the training data using `StandardScaler` from `scikit-learn`.
    *   Train another logistic regression model on the standardized data.
    *   Evaluate the model on the standardized test data and compare the performance to the other two models.

**Key Takeaway:** This project will provide hands-on experience with `scikit-learn`'s scaling tools and demonstrate the impact of feature scaling on model performance. You'll learn how to build a simple machine learning pipeline and compare the results of different preprocessing techniques.

## Topics Not Covered

*   **Standardization (Z-score normalization):** While I touched upon normalization, I didn't explicitly practice standardization, which is another common normalization technique.
*   **Robust Scaling:** This is a scaling technique that is robust to outliers. I will need to explore this in the future.

I'm excited to continue my journey and learn more about feature engineering and selection in the coming days.
