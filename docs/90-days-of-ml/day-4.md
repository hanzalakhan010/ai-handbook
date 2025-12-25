---
sidebar_position: 5
---

# Day 4: A Deep Dive into Data Preprocessing

On my fourth day, I performed a comprehensive data preprocessing workflow on the Titanic dataset. This was a great opportunity to apply and solidify the concepts I've learned over the past three days.

## Key Learnings

### 1. Handling Missing Values

I started by checking for missing values in the dataset. I found that the 'Cabin' column had a large number of missing values, so I decided to drop it. For the 'Age' column, I filled the missing values with the median age.

```python
# dropping the 'Cabin' column
df.drop(columns=['Cabin'], axis=1, inplace=True)

# filling missing 'Age' values with the median
df['Age'] = df['Age'].fillna(df['Age'].median())
```

### 2. Encoding Categorical Variables

I then encoded the categorical variables. I used `LabelEncoder` for the 'Sex' column, as it only has two categories. For the 'Embarked' column, I used one-hot encoding with `pd.get_dummies`.

```python
from sklearn.preprocessing import LabelEncoder

# encoding the 'Sex' column
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

# one-hot encoding the 'Embarked' column
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
```

### 3. Handling Outliers

I noticed that the 'Fare' column had some outliers. I used the `winsorize` function from `scipy.stats.mstats` to handle these outliers by clipping the values at the 5th and 95th percentiles.

```python
from scipy.stats.mstats import winsorize

# handling outliers in the 'Fare' column
df["Fare"] = winsorize(df['Fare'], limits=[0.05, 0.05])
```

### 4. Feature Engineering

I created a new feature called 'FamilySize' by combining the 'SibSp' and 'Parch' columns. This could be a useful feature for predicting survival, as passengers with families might have had different survival rates.

```python
# creating the 'FamilySize' feature
df['FamilySize'] = df['Parch'] + df['SibSp']
```

### 5. Scaling Numerical Features

Finally, I scaled the numerical features ('Age' and 'Fare') using `MinMaxScaler` to bring them to a common scale.

```python
from sklearn.preprocessing import MinMaxScaler

# scaling the numerical features
df[['Age','Fare']] = MinMaxScaler().fit_transform(df[['Age','Fare']])
```

## Reflections

Day 4 was a very productive day. I feel much more confident in my ability to perform a complete data preprocessing workflow. I also learned the importance of feature engineering and how it can be used to create new and potentially more informative features.

I'm excited to start building machine learning models with this preprocessed data in the coming days.

## Small Project: Preprocessing the House Prices Dataset

**Objective:** Apply a complete data preprocessing workflow to the House Prices dataset to prepare it for a regression model.

**Dataset:** The [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) dataset on Kaggle is a great choice for this project. It has a wide variety of features and many data quality issues to address.

**Steps:**

1.  **Load the dataset:** Load the `train.csv` data.
2.  **Handle missing values:**
    *   This dataset has many columns with missing values. You'll need to decide on a strategy for each one.
    *   For some columns, like 'Alley', a missing value might have a specific meaning (e.g., no alley access). You could fill these with a 'None' string.
    *   For numerical features like 'LotFrontage', you could fill missing values with the mean or median.
    *   Some columns might have too many missing values to be useful, so you might decide to drop them.
3.  **Feature Engineering:**
    *   Combine features to create new ones. For example, you could create 'TotalSF' by adding 'TotalBsmtSF', '1stFlrSF', and '2ndFlrSF'.
    *   You could also create features that represent the age of the house ('YrSold' - 'YearBuilt').
4.  **Handle categorical features:**
    *   The dataset has many categorical features. Use one-hot encoding for features with a small number of categories.
    *   For features with many categories (like 'Neighborhood'), you might want to explore other encoding techniques or group some categories together.
5.  **Handle outliers:**
    *   The 'GrLivArea' (Above Ground Living Area) is known to have some extreme outliers. Identify and remove them.
6.  **Scale numerical features:**
    *   Use `StandardScaler` or `MinMaxScaler` to scale all numerical features.
7.  **Save the preprocessed data:** Save the cleaned and engineered DataFrame to a new CSV file.

**Key Takeaway:** This project will challenge you to apply all the preprocessing techniques you've learned to a large and complex dataset. It will give you a realistic experience of what it's like to prepare data for a machine learning competition.
