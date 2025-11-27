# Perform the following operations using Python on a data set : read data
# from different formats(like csv, xls),indexing and selecting data, sort data,
# describe attributes of data, checking data types of each column. (Use
# Titanic Dataset).

import pandas as pd

# Read data from CSV format
# Ensure 'Titanic.csv' is in the same directory or provide the full path
df = pd.read_csv('Titanic.csv')

# Read data from Excel format (Example code)
# Note: This requires 'openpyxl' or 'xlrd' library and an actual .xlsx file
# df_excel = pd.read_excel('Titanic.xlsx')

print("Data loaded successfully.")
print(df.head()) # Display first 5 rows to verify

# 1. Selecting specific columns
# Select a single column
ages = df['Age']
# Select multiple columns
subset = df[['Name', 'Sex', 'Age', 'Survived']]

# 2. Indexing with iloc (Integer Location)
# Select first 5 rows and first 3 columns
rows_iloc = df.iloc[:5, :3]
print(rows_iloc)
print('\n')

# 3. Indexing with loc (Label/Condition based)
# Select data where Age is greater than 50
older_passengers = df.loc[df['Age'] > 76]
print("hi ",older_passengers)
print('\n')

print("Indexing examples executed.")
print(subset.head())

# Sort data by 'Age' in ascending order
sorted_by_age = df.sort_values(by='Age', ascending=True)

# Sort data by 'Fare' in descending order
sorted_by_fare = df.sort_values(by='Fare', ascending=False)

print("Data sorted by Fare (Top 5):")
print(sorted_by_fare[['Name', 'Fare']].head())

# Generate descriptive statistics (count, mean, std, min, max, etc.)
description = df.describe()

print("Statistical Description of Numerical Columns:")
print(description)

# Check the shape of the dataset (rows, columns)
print(f"\nDataset Shape: {df.shape}")

# List attributes (column names)
print(f"\nAttributes: {df.columns.tolist()}")

# Check data types of each column
print("Data Types of Each Column:")
print(df.dtypes)

# Detailed information including non-null counts
print("\nDetailed Info:")
df.info()