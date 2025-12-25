---
sidebar_position: 6
---

# Day 5: Exploratory Data Analysis (EDA)

On my fifth day, I dove into the world of Exploratory Data Analysis (EDA). EDA is a crucial step in the machine learning workflow, as it allows you to understand the characteristics of your data, identify patterns and anomalies, and formulate hypotheses.

I used the classic Iris dataset for this exercise. I followed a set of guidelines to ensure a thorough and systematic EDA process.

## EDA Guidelines

*   **Why Explore the Data?**
    *   To identify patterns, trends, or anomalies.
    *   To understand the distribution of data.
    *   To identify relationships or correlations between features.
*   **Key Questions to Ask:**
    *   What are the central tendencies and spreads (mean, median, variance)?
    *   Are there any outliers or anomalies?
    *   Are features correlated with one another?
    *   Are there patterns across categorical or time-based features?

## Key Learnings

### 1. Central Tendencies and Spreads

I started by calculating the central tendencies and spreads of the numerical features, grouped by species. This gave me a good initial understanding of the data.

```python
df.groupby('Species').agg({
    'SepalLengthCm':['min','max','std','var'],
    'SepalWidthCm':['min','max','std'],
    'PetalLengthCm':['min','max','std'],
    'PetalWidthCm':['min','max','std'],
}).transpose()
```

### 2. Identifying Outliers

I used box plots to visualize the distribution of each feature and identify any potential outliers.

```python
import seaborn as sns

sns.boxplot(x=df['PetalLengthCm'])
```

### 3. Identifying Correlations

I used scatter plots to visualize the relationships between pairs of features and a heatmap to visualize the correlation matrix of the numerical features. This helped me to identify which features were correlated with each other.

```python
# scatter plot
plt.scatter(df['PetalLengthCm'], df['PetalWidthCm'])

# heatmap
numerical = df.drop(columns=['Species']).corr()
sns.heatmap(numerical, annot=True, cmap='coolwarm')
```

### 4. Exploring Relationships

I used bar plots and violin plots to explore the relationship between the categorical 'Species' feature and the other numerical features.

```python
# bar plot
sns.barplot(x="Species", y='SepalLengthCm', data=df)

# violin plot
sns.violinplot(x='Species', y='SepalLengthCm', data=df)
```

### 5. Pairwise Relationships

Finally, I used a pair plot to visualize the pairwise relationships between all the features in the dataset. This gave me a comprehensive overview of the data.

```python
sns.pairplot(df)
```

## Reflections

Day 5 was a great learning experience. I now have a solid understanding of the EDA process and how to use various visualization techniques to gain insights into the data. I'm excited to apply these skills to more complex datasets in the future.

## Small Project: EDA on the Palmer Penguins Dataset

**Objective:** Perform a comprehensive Exploratory Data Analysis (EDA) on the Palmer Penguins dataset and present your findings.

**Dataset:** The [Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/) is a great alternative to the Iris dataset for data exploration and visualization. It contains data for three penguin species observed on three islands in the Palmer Archipelago, Antarctica. You can easily load it using the `seaborn` library (`seaborn.load_dataset('penguins')`).

**Steps:**

1.  **Load the dataset:** Load the penguins dataset using `seaborn`.
2.  **Initial Inspection:**
    *   Use `.info()`, `.describe()`, and `.isnull().sum()` to get a first look at the data.
    *   Handle the few missing values. A simple approach is to drop the rows with missing data, as there are only a few.
3.  **Univariate Analysis (Analyzing single variables):**
    *   Create histograms and box plots for the numerical features (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`) to understand their distributions and identify outliers.
    *   Create count plots for the categorical features (`species`, `island`, `sex`) to see the frequency of each category.
4.  **Bivariate Analysis (Analyzing relationships between two variables):**
    *   Use scatter plots to explore the relationship between `bill_length_mm` and `bill_depth_mm`. Color the points by `species`. What do you observe?
    *   Create box plots to see the distribution of numerical features for each species (e.g., `flipper_length_mm` vs. `species`).
    *   Use a `FacetGrid` to create separate scatter plots for each species.
5.  **Multivariate Analysis (Analyzing relationships between multiple variables):**
    *   Create a `pairplot` of the entire dataset, with the points colored by `species`. This will give you a great overview of the relationships.
    *   Calculate and visualize the correlation matrix for the numerical features using a heatmap.
6.  **Document your findings:**
    *   Write a summary of your key findings. What are the main differences between the penguin species? Are there any interesting correlations? What are the key characteristics of the dataset?

**Key Takeaway:** This project will solidify your EDA skills and teach you how to systematically explore a dataset to uncover insights. You will also get more practice with `seaborn` and `matplotlib` for creating informative visualizations.
