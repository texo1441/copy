"""
Use Iris flower dataset and perform following :
1. Create a box plot for each feature in the dataset.
2. Identify and discuss distributions and identify outliers from them.
"""


import pandas as pd
import matplotlib.pyplot as plt

# Optional: Set a nice style for the plots
plt.style.use('ggplot')

# Load the dataset
df = pd.read_csv('IRIS.csv')

# Display first few rows to ensure data is loaded correctly
print("Data Sample:")
print(df.head())

# Select only numeric columns for the box plot
numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Create the plot
plt.figure(figsize=(10, 6))
df.boxplot(column=numeric_cols)

# Add titles and labels
plt.title('Distribution of Iris Features (Box Plot)', fontsize=16)
plt.ylabel('Length/Width (cm)', fontsize=12)
plt.grid(True) # Add grid lines for easier reading

# Show the plot
plt.show()

# Function to detect outliers using the Interquartile Range (IQR) method
print("--- Outlier Analysis ---")

for col in numeric_cols:
    # 1. Calculate Quartiles
    Q1 = df[col].quantile(0.25) # 25th percentile
    Q3 = df[col].quantile(0.75) # 75th percentile
    IQR = Q3 - Q1               # Interquartile Range
    
    # 2. Define Bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 3. Find values outside these bounds
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    
    # 4. Print Results
    print(f"\nFeature: {col}")
    if not outliers.empty:
        print(f"  Number of Outliers: {len(outliers)}")
        print(f"  Outlier Values: {outliers.values}")
    else:
        print("  No outliers detected.")