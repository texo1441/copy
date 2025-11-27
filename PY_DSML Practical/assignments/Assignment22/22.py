"""
Compute Accuracy, Error rate, Precision, Recall for the following
confusion matrix.
### Actual Class\Predicted class     cancer=yes    cancer = no Total

### cancer = yes   90 210 300
### cancer = no   140 9560 9700
### Total         230 9770 10000
"""


# Define the values based on the provided table
# "Cancer = Yes" is taken as the Positive Class
TP = 90    # True Positives: Actual Yes, Predicted Yes
FN = 210   # False Negatives: Actual Yes, Predicted No
FP = 140   # False Positives: Actual No, Predicted Yes
TN = 9560  # True Negatives: Actual No, Predicted No

total = TP + FN + FP + TN

print("Confusion Matrix:")
print(f"TP={TP}, FN={FN}, FP={FP}, TN={TN}")
print(f"Total Samples: {total}")

# 1. Accuracy
# Formula: (TP + TN) / Total
accuracy = (TP + TN) / total

# 2. Error Rate
# Formula: (FP + FN) / Total  OR  (1 - Accuracy)
error_rate = (FP + FN) / total

# 3. Precision
# Formula: TP / (TP + FP)
# Measures: When it predicts cancer, how often is it correct?
precision = TP / (TP + FP)

# 4. Recall (Sensitivity)
# Formula: TP / (TP + FN)
# Measures: Out of all actual cancer cases, how many did we catch?
recall = TP / (TP + FN)

print("--- Performance Metrics ---")
print(f"Accuracy:   {accuracy:.4f}")
print(f"Error Rate: {error_rate:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")