"""
Perform the following operations using Python on the Telecom_Churn
dataset. Compute and display summary statistics for each feature available
in the dataset using separate commands for each statistic. (e.g. minimum
value, maximum value, mean, range, standard deviation, variance and
percentiles).
"""


import pandas as pd
import numpy as np

# Set display option to show all rows if needed
pd.set_option('display.max_rows', None)

# Load the dataset
# Ensure 'Telecom Churn.csv' is in the same directory
df = pd.read_csv('Telecom Churn.csv')

# Display first 5 rows to verify
print("Dataset Head:")
print(df.head())

# Select only numerical columns for statistical calculations
# This excludes text columns like 'state', 'phone number', etc.
numeric_df = df.select_dtypes(include=['number'])
print("\nNumerical Columns selected for analysis:")
print(numeric_df.columns.tolist())

# Calculate Minimum value for each numerical feature
min_vals = numeric_df.min()

print("--- Minimum Values ---")
print(min_vals)

# Calculate Maximum value for each numerical feature
max_vals = numeric_df.max()

print("--- Maximum Values ---")
print(max_vals)

# Calculate Mean (Average) for each numerical feature
mean_vals = numeric_df.mean()

print("--- Mean Values ---")
print(mean_vals)

# Calculate Range (Max - Min) for each numerical feature
# We use the previously calculated min and max variables
range_vals = max_vals - min_vals

print("--- Range (Max - Min) ---")
print(range_vals)

# Calculate Standard Deviation for each numerical feature
std_vals = numeric_df.std()

print("--- Standard Deviation ---")
print(std_vals)

# Calculate Variance for each numerical feature
var_vals = numeric_df.var()

print("--- Variance ---")
print(var_vals)

# Calculate Percentiles (25%, 50%, 75%) for each numerical feature
# 25th Percentile
p25 = numeric_df.quantile(0.25)

# 50th Percentile (Median)
p50 = numeric_df.quantile(0.50)

# 75th Percentile
p75 = numeric_df.quantile(0.75)

print("--- 25th Percentiles ---")
print(p25)
print("\n--- 50th Percentiles (Median) ---")
print(p50)
print("\n--- 75th Percentiles ---")
print(p75)