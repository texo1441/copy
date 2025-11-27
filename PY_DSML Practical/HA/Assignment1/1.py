"""
Perform the following operations using R/Python on suitable data sets, read data from 
different formats(like csv, xls),indexing and selecting data, sort data, describe attributes of data, 
checking data types of each column, counting unique values of data, format of each column, 
converting variable data type (e.g. from long to short, vice versa), identifying missing values 
and fill in the missing values
"""

import pandas as pd
import numpy as np

# --- 1. Read Data from Different Formats ---
# Reading CSV
# Ensure 'Titanic.csv' is in your working directory
df = pd.read_csv('Titanic.csv')

# Reading Excel (Example code, commented out as we don't have the file)
# df_excel = pd.read_excel('filename.xlsx')

print("--- Data Loaded ---")
print(df.head())

# --- 2. Indexing and Selecting Data ---
print("\n--- Selecting Specific Columns (Name, Age) ---")
print(df[['Name', 'Age']].head())

print("\n--- Indexing with .iloc (Rows 0-4, Cols 0-2) ---")
print(df.iloc[0:5, 0:3])

# --- 3. Sort Data ---
print("\n--- Sorting by Age (Descending) ---")
sorted_df = df.sort_values(by='Age', ascending=False)
print(sorted_df[['Name', 'Age']].head())

# --- 4. Describe Attributes of Data ---
print("\n--- Statistical Description ---")
print(df.describe())

# --- 5 & 7. Check Data Types and Format ---
print("\n--- Data Types (Format of each column) ---")
print(df.dtypes)

# --- 6. Counting Unique Values ---
print("\n--- Unique Values Count per Column ---")
print(df.nunique())

print("\n--- Count of Unique Values in 'Pclass' ---")
print(df['Pclass'].value_counts())

# --- 8. Converting Variable Data Type ---
# Example: Convert 'PassengerId' from int64 (long) to int16 (short)
print(f"\nOriginal 'PassengerId' Type: {df['PassengerId'].dtype}")
df['PassengerId'] = df['PassengerId'].astype('int16')
print(f"Converted 'PassengerId' Type: {df['PassengerId'].dtype}")

# Example: Convert 'Survived' to boolean
df['Survived'] = df['Survived'].astype('bool')
print(f"Converted 'Survived' Type: {df['Survived'].dtype}")

# --- 9. Identifying Missing Values ---
print("\n--- Missing Values Before Filling ---")
print(df.isnull().sum())

# --- 10. Fill in Missing Values ---
# Fill Age (numeric) with Mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill Cabin (categorical) with 'Unknown'
df['Cabin'] = df['Cabin'].fillna('Unknown')

# Fill Fare (numeric) with Median
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

print("\n--- Missing Values After Filling ---")
print(df.isnull().sum())