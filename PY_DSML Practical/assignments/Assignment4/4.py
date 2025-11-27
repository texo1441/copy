"""
Write a program to do: A dataset collected in a cosmetics shop showing
details of customers and whether or not they responded to a special offer
to buy a new lip-stick is shown in table below. (Implement step by step
using commands - Dont use library) Use this dataset to build a decision
tree, with Buys as the target variable, to help in buying lipsticks in the
future. Find the root node of the decision tree.
"""


import pandas as pd
import numpy as np
import math

# Load the dataset
df = pd.read_csv('Lipstick.csv')

# Drop 'Id' column as it is not a feature
df = df.drop('Id', axis=1)

print("Dataset:")
print(df)
print("\nColumns:", df.columns.tolist())

def calculate_entropy(data, target_col):
    """
    Calculates the Shannon Entropy of a dataset for a given target column.
    Formula: H(S) = -sum(p_i * log2(p_i))
    """
    # Get all unique values in the target column (e.g., 'Yes', 'No')
    values = data[target_col].unique()
    entropy = 0
    total_count = len(data)
    
    for value in values:
        # Count how many times this value appears
        count = len(data[data[target_col] == value])
        # Calculate probability p_i
        probability = count / total_count
        # Add to entropy formula
        entropy += -probability * math.log2(probability)
        
    return entropy

def calculate_information_gain(data, attribute, target_col):
    """
    Calculates the Information Gain of a specific attribute.
    Formula: Gain(S, A) = Entropy(S) - Sum((|Sv|/|S|) * Entropy(Sv))
    """
    # 1. Calculate Total Entropy of the entire set
    total_entropy = calculate_entropy(data, target_col)
    
    # 2. Calculate Weighted Entropy of the attribute
    values = data[attribute].unique()
    weighted_entropy = 0
    total_count = len(data)
    
    # Loop through each unique value of the attribute (e.g., 'High', 'Medium', 'Low')
    for value in values:
        # Create a subset of data where the attribute equals this value
        subset = data[data[attribute] == value]
        
        # Calculate the weight (|Sv|/|S|)
        weight = len(subset) / total_count
        
        # Calculate entropy of this subset
        subset_entropy = calculate_entropy(subset, target_col)
        
        # Add to the weighted entropy sum
        weighted_entropy += weight * subset_entropy
        
    # 3. Calculate Information Gain
    information_gain = total_entropy - weighted_entropy
    return information_gain, total_entropy

# Define the target variable and features
target_col = 'Buys'
features = [col for col in df.columns if col != target_col]

best_gain = -1
root_node = None

print(f"--- Calculating Information Gain for Target: '{target_col}' ---\n")

# Loop through each feature to calculate its Information Gain
for feature in features:
    gain, total_entropy = calculate_information_gain(df, feature, target_col)
    
    print(f"Feature: {feature}")
    print(f"  Total Entropy:    {total_entropy:.4f}")
    print(f"  Information Gain: {gain:.4f}")
    print("-" * 30)
    
    # Check if this is the best gain so far
    if gain > best_gain:
        best_gain = gain
        root_node = feature

print(f"\nRESULT: The Root Node is '{root_node}' with the highest Information Gain of {best_gain:.4f}")

def build_tree(data, features, target_col, level=0):
    """
    Recursive function to build the full decision tree structure.
    """
    indent = "  " * level
    
    # Base Case 1: If all target values are the same, return that value (Leaf Node)
    unique_targets = data[target_col].unique()
    if len(unique_targets) == 1:
        print(f"{indent}Leaf: {unique_targets[0]}")
        return
    
    # Base Case 2: If no features left, return the most common target value
    if len(features) == 0:
        most_common = data[target_col].mode()[0]
        print(f"{indent}Leaf: {most_common}")
        return

    # Find the best feature to split on
    best_gain = -1
    best_feature = None
    
    for feature in features:
        gain, _ = calculate_information_gain(data, feature, target_col)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
            
    print(f"{indent}Node: {best_feature} (Gain: {best_gain:.4f})")
    
    # Recursively build tree for each value of the best feature
    remaining_features = [f for f in features if f != best_feature]
    
    for value in data[best_feature].unique():
        print(f"{indent}-> Branch: {value}")
        subset = data[data[best_feature] == value]
        build_tree(subset, remaining_features, target_col, level + 1)

print("--- Full Decision Tree Structure ---")
build_tree(df, features, target_col)