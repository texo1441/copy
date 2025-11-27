"""
Compute Accuracy, Error rate, Precision, Recall for following confusion
matrix ( Use formula for each)

True Positives (TPs): 1 False Positives (FPs): 1
False Negatives (FNs): 8 True Negatives (TNs): 90
"""


# Define the values from the confusion matrix
TP = 1   # True Positives
FP = 1   # False Positives
FN = 8   # False Negatives
TN = 90  # True Negatives

print("Confusion Matrix:")
print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")

# Calculate Total Predictions
total = TP + FP + FN + TN

# 1. Accuracy
# Formula: (TP + TN) / Total
accuracy = (TP + TN) / total

# 2. Error Rate
# Formula: (FP + FN) / Total  OR  1 - Accuracy
error_rate = (FP + FN) / total

# 3. Precision
# Formula: TP / (TP + FP)
# Measures: Out of all predicted positives, how many were actually positive?
precision = TP / (TP + FP)

# 4. Recall (Sensitivity)
# Formula: TP / (TP + FN)
# Measures: Out of all actual positives, how many did we correctly identify?
recall = TP / (TP + FN)

print("--- Calculated Metrics ---")
print(f"Accuracy:   {accuracy:.4f}")
print(f"Error Rate: {error_rate:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")