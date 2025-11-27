"""
Use the inbuilt dataset &#39;titanic&#39;. The dataset contains 891 rows and
contains information about the passengers who boarded the unfortunate
Titanic ship. Write a code to check how the price of the ticket (column
name: &#39;fare&#39;) for each passenger is distributed by plotting a histogram.
"""


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd # Optional, but good practice

# Load the inbuilt 'titanic' dataset from seaborn
# This dataset contains 891 rows as specified
df = sns.load_dataset('titanic')

# Display first few rows to verify
print("Dataset Shape:", df.shape)
print(df.head())

# Set the figure size
plt.figure(figsize=(10, 6))

# Create the histogram
# kde=True adds a Kernel Density Estimate line (the smooth curve)
# bins=30 controls the number of bars
sns.histplot(data=df, x='fare', kde=True, bins=30, color='blue')

# Add titles and labels
plt.title('Distribution of Ticket Prices (Fare)', fontsize=16)
plt.xlabel('Fare', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Show the plot
plt.show()