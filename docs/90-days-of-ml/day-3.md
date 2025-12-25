---
sidebar_position: 4
---

# Day 3: Parsing Dates and Categorical Data

On the third day of my machine learning journey, I focused on two important data preprocessing tasks: parsing dates and handling categorical data. These are common challenges when working with real-world datasets, and mastering them is essential for building effective machine learning models.

## Key Learnings

### 1. Parsing Dates

I learned how to parse date columns in a dataset using `pandas.to_datetime`. This function can automatically infer the date format in many cases, but I also learned how to specify the format explicitly. Once the dates are parsed, I can easily extract features like the month or day of the week.

```python
import pandas as pd

# parsing the date column
df['date_parsed'] = pd.to_datetime(df['date'], format="%m/%d/%y")

# extracting the month
month_of_landslides = df['date_parsed'].dt.month
```

### 2. One-Hot Encoding

I also learned how to handle categorical data using one-hot encoding. This technique creates a new binary column for each category in a categorical feature. I used the `pandas.get_dummies` function to perform one-hot encoding.

```python
# one-hot encoding the 'Species' column
encoded_df = pd.get_dummies(df, columns=['Species'])
```

## Titanic Dataset: A Case Study

I applied these techniques to the Titanic dataset. I handled missing values in the 'Age' column by filling them with the median age. I also performed one-hot encoding on the 'Embarked' column.

```python
df['Age'] = df['Age'].fillna(df['Age'].median())
df_encoded = pd.get_dummies(df, columns=['Embarked'])
```

## Reflections

### Challenges Faced

*   While working on the volcanoes dataset, I encountered dates in BCE and CE format. I realized that I would need to filter these dates first before parsing them.

### Handling Missing Values

*   I used `fillna(df['Age'].median())` to fill missing ages with the median of the 'Age' column. I could have also used the mean.
*   To remove outliers, I would need to first find the lower and upper bounds using the interquartile range (IQR) method (q1 - 1.5*iqr and q3 + 1.5*iqr) and then filter the values in the 'Age' column.

I'm getting more comfortable with data preprocessing, and I'm excited to move on to more advanced topics.

## Small Project: Feature Engineering on a Sales Dataset

**Objective:** Apply date parsing and categorical data encoding to a sales dataset to prepare it for a sales forecasting model.

**Dataset:** You can find many sales datasets online. A good example is the [Rossmann Store Sales dataset](https://www.kaggle.com/c/rossmann-store-sales/data) on Kaggle.

**Steps:**

1.  **Load the dataset:** Load the sales data into a `pandas` DataFrame.
2.  **Parse dates:**
    *   The 'Date' column is in a string format. Use `pd.to_datetime` to convert it to a datetime object.
    *   Extract useful features from the date, such as 'Year', 'Month', 'Day', 'DayOfWeek', and 'WeekOfYear'. These new features can be very helpful for a forecasting model.
3.  **Handle categorical features:**
    *   The dataset has several categorical columns like 'StoreType', 'Assortment', and 'StateHoliday'.
    *   Use `pd.get_dummies` to one-hot encode these columns. Pay attention to the 'StateHoliday' column, which has a '0' as a string and 0 as a number. You might need to clean this up first.
4.  **Explore the new features:**
    *   Analyze the newly created date features. For example, plot the average sales per month to see if there are any seasonal patterns.
    *   Check the correlation between the new features and the 'Sales' column.
5.  **Save the preprocessed data:** Save the DataFrame with the new features to a new CSV file.

**Key Takeaway:** This project will teach you how to create valuable new features from existing columns (feature engineering), which is a crucial skill in machine learning. You'll see how parsing dates and encoding categorical variables can make a dataset much more suitable for building predictive models.
