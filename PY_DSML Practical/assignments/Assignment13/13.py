"""
Use the covid_vaccine_statewise.csv dataset and perform the
following analytics.
#
a. Describe the dataset
b. Number of persons state wise vaccinated for first dose in India
c. Number of persons state wise vaccinated for second dose in India
"""


import pandas as pd

# Load the dataset
df = pd.read_csv('Covid_Vaccine_Statewise.csv')

# Display the first 5 rows
print("Data Loaded Successfully.")
print(df.head())

# a. Describe the dataset
print("--- Dataset Information ---")
print(df.info())

print("\n--- Statistical Description ---")
print(df.describe())

print("\n--- List of States ---")
print(df['State'].unique())

# b. Number of persons state wise vaccinated for first dose
# Since the data is time-series cumulative, we take the maximum value for each state
# to get the total count.

first_dose_stats = df.groupby('State')['First Dose Administered'].max().reset_index()

# Rename columns for clarity
first_dose_stats.columns = ['State', 'Total First Doses']

print("--- Number of Persons Vaccinated for First Dose (State-wise) ---")
print(first_dose_stats)

# c. Number of persons state wise vaccinated for second dose
# Similarly, we take the maximum value to get the latest cumulative count.

second_dose_stats = df.groupby('State')['Second Dose Administered'].max().reset_index()

# Rename columns for clarity
second_dose_stats.columns = ['State', 'Total Second Doses']

print("--- Number of Persons Vaccinated for Second Dose (State-wise) ---")
print(second_dose_stats)