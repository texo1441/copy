"""
Perform Data Cleaning, Data transformation using Python on any data
set.
"""


import pandas as pd
import numpy as np

# Load the dataset
# Ensure 'Titanic.csv' is in your current directory
df = pd.read_csv('Titanic.csv')

print("--- Initial Data ---")
print(df.head())

# 1. Handle Missing Values
# Fill Age with Median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill Fare with Median
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Fill Embarked with Mode (Most common value)
if 'Embarked' in df.columns:
    mode_embarked = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(mode_embarked)

# 2. Drop Irrelevant Columns
# We remove columns that are not useful for general analysis
cols_to_drop = ['Cabin', 'PassengerId', 'Name', 'Ticket']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

print("--- Data Cleaned (Missing Values Handled) ---")
print(df.isnull().sum())

# 1. Numerical Transformation (Mapping)
# Convert 'Sex' to numbers (0=Male, 1=Female)
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 2. Binning (Creating Categories)
# Create Age Groups: Child (0-12), Teen (12-18), Adult (18-60), Senior (60+)
bins = [0, 12, 18, 60, 100]
labels = ['Child', 'Teenager', 'Adult', 'Senior']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Note: We skipped One-Hot Encoding for 'Embarked', so it remains as text (S, C, Q).

print("\n--- Final Transformed Data ---")
print(df.head())
print("\nFinal Data Types:")
print(df.dtypes)