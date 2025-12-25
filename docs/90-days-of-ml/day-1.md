---
sidebar_position: 2
---

# Day 1: Data Cleaning and Preparation

On the first day of my 90-day machine learning journey, I focused on the fundamental task of data cleaning and preparation. A crucial first step in any machine learning project, this process ensures the quality and reliability of the data used to train a model.

I worked with the "Building Permits" dataset from San Francisco. The primary tool for this task was the `pandas` library, a powerful and versatile tool for data manipulation in Python.

## Key Learnings

### 1. Handling Missing Data

A common challenge in real-world datasets is the presence of missing values. I learned how to identify and handle these missing data points.

First, I calculated the percentage of missing values in the dataset:

```python
import pandas as pd
import numpy as np

sf_permit = pd.read_csv('datasets/Building_Permits.csv')

# 2) How many missing data points do we have?
total_missing = sf_permit.isnull().sum().sum()
total_cells  = sf_permit.shape[0] * sf_permit.shape[1]
percentage_missing = (total_missing/total_cells)
percentage_missing
```

Then, I explored different strategies for handling these missing values, such as filling them with the next or previous value (`bfill` or `ffill`) or dropping columns with a high percentage of missing data.

```python
#filling null values with 
sf_permit.fillna(method='bfill',axis = 0)
```

### 2. Dropping Unnecessary Columns

Not all columns in a dataset are relevant for a machine learning model. I learned how to drop unnecessary columns to simplify the dataset and improve model performance.

```python
sf_permit.drop(columns=["Permit Number",'Permit Type Definition'],inplace=True,axis=1)
```

## Reflections

This first day was a great introduction to the practical side of machine learning. I learned that data cleaning and preparation is a critical and often time-consuming part of the process. I also gained a better understanding of how to use `pandas` to manipulate and clean data.

## Small Project: Cleaning a Real-World Dataset

**Objective:** Apply the data cleaning techniques learned today to a new, messy dataset.

**Dataset:** You can find many messy datasets on platforms like Kaggle. A good starting point would be the [Titanic dataset](https://www.kaggle.com/c/titanic/data), as it's a classic and contains various data quality issues like missing values and a mix of data types.

**Steps:**

1.  **Load the dataset:** Use `pandas` to load the Titanic dataset into a DataFrame.
2.  **Inspect the data:** Use `.info()`, `.describe()`, and `.isnull().sum()` to understand the dataset's structure, identify missing values, and find other inconsistencies.
3.  **Handle missing values:**
    *   For the 'Age' column, try filling missing values with the mean or median age.
    *   For the 'Embarked' column, fill missing values with the most frequent port of embarkation.
    *   The 'Cabin' column has too many missing values; it might be best to drop this column.
4.  **Drop unnecessary columns:** Decide which columns are not relevant for predicting survival (e.g., 'PassengerId', 'Name', 'Ticket') and drop them.
5.  **Save the cleaned dataset:** Save the cleaned DataFrame to a new CSV file.

**Key Takeaway:** This project will give you hands-on experience in making data cleaning decisions and applying `pandas` functions to a real-world problem.

## Topics Not Covered

*   **Imputation with more advanced techniques:** I used simple forward and backward fill methods. More advanced techniques like mean/median/mode imputation or even model-based imputation could be explored.
*   **Handling categorical missing data:** The dataset contains categorical features with missing values. I need to learn specific techniques for handling these.

I'm looking forward to diving deeper into data exploration and visualization on Day 2!
