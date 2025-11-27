"""
Use the dataset &#39;titanic&#39;. The dataset contains 891 rows and contains
information about the passengers who boarded the unfortunate Titanic
ship. Use the Seaborn library to see if we can find any patterns in the data.
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Load the dataset
df = pd.read_csv('Titanic.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize the overall survival count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Survived', palette='pastel')
plt.title('Overall Survival Count (0 = No, 1 = Yes)')
plt.show()

# Visualize survival based on Gender
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Survived', hue='Sex', palette='coolwarm')
plt.title('Survival by Gender')
plt.show()

# Visualize survival based on Passenger Class (Pclass)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Survived', hue='Pclass', palette='viridis')
plt.title('Survival by Passenger Class')
plt.show()

# Visualize the age distribution of survivors vs non-survivors
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Survived', kde=True, element="step", palette='autumn')
plt.title('Age Distribution by Survival Status')
plt.show()