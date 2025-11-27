"""
With reference to Table , obtain the Frequency table for the
attribute age. From the frequency table you have obtained, calculate
the information gain of the frequency table while splitting on Age. (Use
step by step Python/Pandas commands)

| Age    | Income | Married | Health | Class |
|--------|--------|---------|--------|-------|
| Young  | High   | No      | Fair   | No    |
| young  | High   | No      | Good   | No    |
| Middle | High   | No      | Fair   | Yes   |
| Old    | Medium | No      | Fair   | Yes   |
| Old    | Low    | Yes     | Fair   | Yes   |
| Old    | Low    | Yes     | Good   | Yes   |
| Middle | Low    | No      | Good   | Yes   |
| Young  | Medium | No      | Fair   | No    |
| Young  | Low    | Yes     | Fair   | Yes   |
| Old    | Medium | Yes     | Fair   | Yes   |
| Young  | Medium | Yes     | Good   | Yes   |
| Middle | Medium | No      | Good   | Yes   |
| Middle | High   | Yes     | Fair   | Yes   |
| Old    | Medium | No      | Good   | No    |

"""


import pandas as pd
import numpy as np

# Create the dataset directly from the provided table
data = {
    'Age': ['Young', 'young', 'Middle', 'Old', 'Old', 'Old', 'Middle', 'Young', 'Young', 'Old', 'Young', 'Middle', 'Middle', 'Old'],
    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium'],
    'Married': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No'],
    'Health': ['Fair', 'Good', 'Fair', 'Fair', 'Fair', 'Good', 'Good', 'Fair', 'Fair', 'Fair', 'Good', 'Good', 'Fair', 'Good'],
    'Class': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Normalize 'Age' column (fix 'young' vs 'Young' inconsistency)
df['Age'] = df['Age'].str.capitalize()

print("--- Dataset ---")
print(df)

# 1. Simple Frequency Table (Counts of each Age)
age_counts = df['Age'].value_counts()
print("\n--- Frequency Table (Attribute: Age) ---")
print(age_counts)

# 2. Contingency Table (Age vs Class)
# This shows how many Yes/No classes are in each Age group
freq_table = pd.crosstab(df['Age'], df['Class'])
print("\n--- Frequency Table (Age vs Class) ---")
print(freq_table)

def calculate_entropy(series):
    """Calculates the Shannon Entropy of a target series."""
    counts = series.value_counts()
    probabilities = counts / len(series)
    # Formula: H(S) = - sum(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Step 1: Calculate Total Entropy of the system (Parent Node)
total_entropy = calculate_entropy(df['Class'])
print(f"\n1. Total Entropy (S): {total_entropy:.4f}")

# Step 2: Calculate Weighted Entropy after splitting by Age
weighted_entropy = 0
total_samples = len(df)

print("\n2. Entropy per Age Group:")
for age_group in df['Age'].unique():
    # Filter data for this specific age group
    subset = df[df['Age'] == age_group]
    
    # Calculate entropy for this subset
    subset_entropy = calculate_entropy(subset['Class'])
    
    # Calculate weight (Probability of this age group occurring)
    weight = len(subset) / total_samples
    
    # Add to weighted sum
    weighted_entropy += weight * subset_entropy
    
    print(f"   - Entropy({age_group}): {subset_entropy:.4f} (Weight: {weight:.4f})")

print(f"   -> Weighted Entropy (S|Age): {weighted_entropy:.4f}")

# Step 3: Calculate Information Gain
# Formula: Gain(S, A) = Entropy(S) - Weighted_Entropy(S|A)
info_gain = total_entropy - weighted_entropy

print(f"\n3. Information Gain for 'Age': {info_gain:.4f}")