"""
Write a Python program to display some basic statistical details like
percentile, mean, standard deviation etc (Use python and pandas
commands) the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris-versicolor’
of iris.csv dataset.
"""


import pandas as pd

# Load the dataset
# Ensure 'IRIS.csv' is in the same directory as your script
df = pd.read_csv('IRIS.csv')

# Display the first few rows to verify loading
print("First 5 rows of the dataset:")
print(df.head())

# List of species to analyze
# The dataset typically contains these three species
species_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

print("\n--- Statistical Details by Species ---")

for species in species_names:
    print(f"\nSpecies: {species}")
    
    # 1. Filter the dataframe to get only rows for the current species
    species_data = df[df['species'] == species]
    
    # Check if the filtered data is not empty
    if not species_data.empty:
        # 2. Calculate statistics using describe()
        # describe() computes count, mean, std, min, 25%, 50%, 75%, max
        stats = species_data.describe()
        print(stats)
    else:
        print("No data found for this species.")