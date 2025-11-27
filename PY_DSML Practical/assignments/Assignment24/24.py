"""
Perform the following operations using Python on a suitable data set,
counting unique values of data, format of each column, converting
variable data type (e.g. from long to short, vice versa), identifying missing
values and filling in the missing values.
"""


import pandas as pd
import numpy as np

# Load the dataset
# Ensure 'Titanic.csv' is in your working directory
df = pd.read_csv('Titanic.csv')

print("--- Data Loaded ---")
print(df.head())

# Check the format (data type) of each column
print("--- Column Data Types ---")
print(df.dtypes)

# Count unique values for a specific categorical column (e.g., 'Pclass')
print("--- Count of Unique Values in 'Pclass' ---")
print(df['Pclass'].value_counts())

# Check number of unique values in all columns
print("\n--- Number of Unique Values per Column ---")
print(df.nunique())

# Identify missing values (NaN/Null) in each column
print("--- Missing Values Count ---")
missing_values = df.isnull().sum()
print(missing_values)

# 1. Fill numeric missing values (Age) with the Mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# 2. Fill categorical missing values (Cabin) with a placeholder string
df['Cabin'] = df['Cabin'].fillna('Unknown')

# 3. Fill numeric missing values (Fare) with the Median
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

print("--- Missing Values After Filling ---")
print(df.isnull().sum())

# Example: Convert 'PassengerId' from int64 (Long) to int16 (Short) to save memory
print(f"Original 'PassengerId' Type: {df['PassengerId'].dtype}")

# Converting to int16
df['PassengerId'] = df['PassengerId'].astype('int16')

print(f"Converted 'PassengerId' Type: {df['PassengerId'].dtype}")

# Example: Convert 'Survived' (0/1) to Boolean
print(f"Original 'Survived' Type: {df['Survived'].dtype}")
df['Survived'] = df['Survived'].astype('bool')
print(f"Converted 'Survived' Type: {df['Survived'].dtype}") 