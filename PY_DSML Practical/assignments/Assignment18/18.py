"""
Use House_Price prediction dataset. Provide summary statistics (mean,
median, minimum, maximum, standard deviation) of variables (categorical
vs quantitative) such as- For example, if categorical variable is age groups
and quantitative variable is income, then provide summary statistics of
income grouped by the age groups.
"""


import pandas as pd

# Load the dataset
# Ensure 'Housing.csv' is in the same directory as your script
df = pd.read_csv('Housing.csv')

# Display basic info to help choose variables
print("--- Columns in Dataset ---")
print(df.columns.tolist())
print("\n--- First 5 Rows ---")
print(df.head())

# We will perform two examples of grouping.

# Example 1: Analyze 'price' (Quantitative) grouped by 'furnishingstatus' (Categorical)
# This answers: "Does a furnished house cost more on average?"
print("\n=== Analysis 1: House Price by Furnishing Status ===")

# .groupby('Category')['Quantitative'].agg([...list of stats...])
stats_price = df.groupby('furnishingstatus')['price'].agg(['mean', 'median', 'min', 'max', 'std'])

# Format the output for better readability (optional)
pd.options.display.float_format = '{:,.2f}'.format
print(stats_price)


# Example 2: Analyze 'area' (Quantitative) grouped by 'mainroad' (Categorical)
# This answers: "Are houses on the main road generally bigger?"
print("\n=== Analysis 2: House Area by Main Road Access ===")

stats_area = df.groupby('mainroad')['area'].agg(['mean', 'median', 'min', 'max', 'std'])
print(stats_area)