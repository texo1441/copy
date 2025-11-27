"""
Use Iris flower dataset and perform following :
1. List down the features and their types (e.g., numeric, nominal)
available in the dataset. 2. Create a histogram for each feature in the
dataset to illustrate the feature distributions.
"""


import pandas as pd
import matplotlib.pyplot as plt

# Optional: Set plot style for better aesthetics
plt.style.use('ggplot')

# Load the dataset
# Ensure 'IRIS.csv' is in the same directory
df = pd.read_csv('IRIS.csv')

# Display first 5 rows to check data
print("Data Loaded Successfully:")
print(df.head())

print("--- Features and their Types ---")
# df.dtypes returns the type of each column
# float64 = Numeric (Continuous)
# object = Nominal (Categorical/Text)
print(df.dtypes)

print("\n--- Summary ---")
for col in df.columns:
    # Determine type based on pandas dtype
    if pd.api.types.is_numeric_dtype(df[col]):
        print(f"Feature '{col}': Numeric")
    else:
        print(f"Feature '{col}': Nominal (Categorical)")

# Select only numeric columns for histograms
# The 'species' column is categorical, so we don't plot a histogram for it
numeric_features = df.select_dtypes(include=['number'])

# Create histograms
# layout=(2, 2) arranges the 4 plots in a 2x2 grid
# figsize=(10, 8) sets the image size
numeric_features.hist(bins=20, edgecolor='black', figsize=(10, 8))

# Add a main title
plt.suptitle('Feature Distributions (Histograms)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to fit title

# Show the plot
plt.show()