"""
Perform the following operations using Python on the data set
House_Price Prediction dataset. Compute standard deviation, variance and
percentiles using separate commands, for each feature. Create a histogram
for each feature in the dataset to illustrate the feature distributions.
"""


import pandas as pd
import matplotlib.pyplot as plt

# Optional: Display floats with 2 decimal places
pd.options.display.float_format = '{:.2f}'.format

# Load the dataset
df = pd.read_csv('Housing.csv')

print(df.head())

# Select only numerical columns for statistics and histograms
# (Standard deviation and variance are not applicable to text columns like 'yes'/'no')
numeric_df = df.select_dtypes(include=['number'])

print("Data Loaded Successfully.")
print(numeric_df.head())

print(numeric_df.describe())

# 1. Standard Deviation
print("--- Standard Deviation ---")
print(numeric_df.std())

# 2. Variance
print("\n--- Variance ---")
print(numeric_df.var())

# 3. Percentiles
# We calculate the 25th, 50th (Median), and 75th percentiles
print("\n--- 25th Percentile ---")
print(numeric_df.quantile(0.25))

print("\n--- 50th Percentile (Median) ---")
print(numeric_df.quantile(0.50))

print("\n--- 75th Percentile ---")
print(numeric_df.quantile(0.75))

# Create histograms for each numerical feature 
#A histogram is a bar chart that visualizes the frequency distribution of numerical
#data by grouping values into continuous ranges called "bins."
# The x-axis represents the value ranges (like price buckets), and the y-axis shows 
# the count of items falling into each bucket.
# bins=20 defines how many bars are in the chart
# figsize defines the image size (width, height)
numeric_df.hist(figsize=(12, 10), bins=20, edgecolor='black')

# Add a main title
plt.suptitle('Feature Distributions (Histograms)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to fit title

# Show the plot
plt.show()