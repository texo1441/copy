"""
Perform the following operations using R/Python on the data sets Compute and display 
summary statistics for each feature available in the dataset. (e.g. minimum value, maximum 
value, mean, range, standard deviation, variance and percentiles · Data Visualization-Create a 
histogram for each feature in the dataset to illustrate the feature distributions. · Data cleaning · 
Data integration · Data transformation · Data model building(e.g. Classification)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Data Integration (Simulated) ---
# To demonstrate data integration, we'll simulate loading two separate datasets
# and merging them based on a common key ('PassengerId').

print("--- 1. Data Integration ---")
# Load the full dataset first to simulate the split
try:
    full_df = pd.read_csv('Titanic.csv')
    
    # Split into two datasets
    # Dataset A: Personal Information
    df_personal = full_df[['PassengerId', 'Name', 'Sex', 'Age']]
    
    # Dataset B: Travel & Survival Information
    df_travel = full_df[['PassengerId', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    
    print("Dataset A (Personal) Columns:", df_personal.columns.tolist())
    print("Dataset B (Travel) Columns:", df_travel.columns.tolist())
    
    # Merge them back together (Integration)
    df = pd.merge(df_personal, df_travel, on='PassengerId')
    print("Merged Dataset Shape:", df.shape)
    
except FileNotFoundError:
    print("Error: 'Titanic.csv' not found. Please upload the file.")
    # Stop execution if file is missing (in a real script)
    df = pd.DataFrame() 

if not df.empty:
    # --- 2. Data Cleaning ---
    print("\n--- 2. Data Cleaning ---")
    print("Missing Values Before Cleaning:\n", df.isnull().sum())
    
    # Fill missing Age with Median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Fill missing Embarked with Mode
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        
    # Fill missing Fare with Median
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Drop columns that won't be used for the model
    df.drop(columns=['Name'], inplace=True)
    
    print("Missing Values After Cleaning:\n", df.isnull().sum())

    # --- 3. Data Transformation ---
    print("\n--- 3. Data Transformation ---")
    # Convert 'Sex' to numeric (0=male, 1=female)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Convert 'Embarked' to numeric using One-Hot Encoding or Mapping
    # Here using Mapping for simplicity: S=0, C=1, Q=2
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    df['Embarked'] = df['Embarked'].map(embarked_map)
    
    print("Data Types After Transformation:\n", df.dtypes)
    print(df.head())

    # --- 4. Summary Statistics ---
    print("\n--- 4. Summary Statistics ---")
    # Select numerical columns
    num_cols = df.select_dtypes(include=np.number).columns
    
    # Calculate stats
    stats = df[num_cols].agg(['min', 'max', 'mean', 'std', 'var'])
    
    # Calculate Range (Max - Min)
    range_row = df[num_cols].max() - df[num_cols].min()
    stats.loc['range'] = range_row
    
    # Calculate Percentiles
    percentiles = df[num_cols].quantile([0.25, 0.50, 0.75])
    
    print("Basic Statistics:")
    print(stats)
    print("\nPercentiles:")
    print(percentiles)

    # --- 5. Data Visualization ---
    print("\n--- 5. Data Visualization (Histograms) ---")
    # Plot histograms for all numerical features
    df[num_cols].hist(figsize=(12, 10), bins=20, edgecolor='black')
    plt.suptitle('Feature Distributions', fontsize=16)
    plt.tight_layout()
    plt.show()

    # --- 6. Data Model Building (Classification) ---
    print("\n--- 6. Data Model Building (Logistic Regression) ---")
    
    # Define Features (X) and Target (y)
    # Target: Survived, Features: All others except PassengerId
    X = df.drop(columns=['Survived', 'PassengerId'])
    y = df['Survived']
    
    # Split into Train and Test sets (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and Train Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))