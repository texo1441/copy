"""
Write a program to do: A dataset collected in a cosmetics shop showing
details of customers and whether or not they responded to a special offer
to buy a new lip-stick is shown in table below. (Use library commands)
According to the decision tree you have made from the previous training
data set, what is the decision for the test data: [Age > 35, Income =
Medium, Gender = Female, Marital Status = Married]?
"""


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv('Lipstick.csv')

# Drop 'Id' column if it exists
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

print("Training Data Sample:")
print(df.head())

# --- Preprocessing ---
# Initialize encoders for each column
le_age = LabelEncoder()
le_income = LabelEncoder()
le_gender = LabelEncoder()
le_ms = LabelEncoder()
le_buys = LabelEncoder()

# Fit and transform the training data
df['Age_encoded'] = le_age.fit_transform(df['Age'])
df['Income_encoded'] = le_income.fit_transform(df['Income'])
df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])
df['Ms_encoded'] = le_ms.fit_transform(df['Ms'])
df['Buys_encoded'] = le_buys.fit_transform(df['Buys'])

# Define Features (X) and Target (y)
X = df[['Age_encoded', 'Income_encoded', 'Gender_encoded', 'Ms_encoded']]
y = df['Buys_encoded']

print("\nData encoded successfully.")

# Initialize and Train Decision Tree Model
# criterion='entropy' uses Information Gain (matching ID3 logic)
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y)

print("Model trained successfully.")

# The Test Data: [Age > 35, Income = Medium, Gender = Female, Marital Status = Married]
test_age = '>35'
test_income = 'Medium'
test_gender = 'Female'
test_ms = 'Married'

print(f"Testing for: Age={test_age}, Income={test_income}, Gender={test_gender}, Ms={test_ms}")

# We must encode the test data using the SAME encoders we trained with
# Note: Inputs to transform must be a list (e.g., [test_age])
try:
    encoded_input = [
        le_age.transform([test_age])[0],
        le_income.transform([test_income])[0],
        le_gender.transform([test_gender])[0],
        le_ms.transform([test_ms])[0]
    ]
    
    # Predict (input must be a 2D array, so we wrap our list in another list)
    prediction_n = clf.predict([encoded_input])
    
    # Convert the numerical prediction (0 or 1) back to text ('Yes' or 'No')
    prediction_text = le_buys.inverse_transform(prediction_n)
    
    print(f"\nPrediction: {prediction_text[0]}")

except ValueError as e:
    print(f"\nError: {e}")
    print("Ensure test values match the categories in the training data exactly.")