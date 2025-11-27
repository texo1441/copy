"""
Use the covid_vaccine_statewise.csv dataset and perform the
following analytics.
#
A. Describe the dataset.
B. Number of Males vaccinated
C.. Number of females vaccinated
"""


import pandas as pd

# Load the dataset
# Ensure the file 'Covid Vaccine Statewise.csv' is in your current directory
df = pd.read_csv('Covid_Vaccine_Statewise.csv')

# Display the first few rows to verify loading
print("Data loaded successfully.")
print(df.head())

# A. Describe the dataset
print("--- Dataset Information ---")
print(df.info())

print("\n--- Statistical Description ---")
print(df.describe())

# B. Number of Males vaccinated
# Since the data is cumulative, we calculate the maximum value for each state 
# to get the total count for that state.

# 'Male (Doses Administered)' tracks the total doses given to males.
male_vaccinated_statewise = df.groupby('State')['Male (Doses Administered)'].max().reset_index()

# Rename the column for clarity
male_vaccinated_statewise.columns = ['State', 'Total Males Vaccinated']

print("--- Number of Males Vaccinated (State-wise) ---")
print(male_vaccinated_statewise)

# To find the total for India, we can filter for the 'India' row if it exists,
# or sum the states (excluding 'India' to avoid double counting).
# Here we check the 'India' row which usually aggregates the data.
india_total_males = df[df['State'] == 'India']['Male (Doses Administered)'].max()
print(f"\nTotal Males Vaccinated in India: {india_total_males}")

# C. Number of Females vaccinated
# Similarly, we find the maximum cumulative value for females per state.

female_vaccinated_statewise = df.groupby('State')['Female (Doses Administered)'].max().reset_index()

# Rename the column for clarity
female_vaccinated_statewise.columns = ['State', 'Total Females Vaccinated']

print("--- Number of Females Vaccinated (State-wise) ---")
print(female_vaccinated_statewise)

# Total for India (from the 'India' row)
india_total_females = df[df['State'] == 'India']['Female (Doses Administered)'].max()
print(f"\nTotal Females Vaccinated in India: {india_total_females}")