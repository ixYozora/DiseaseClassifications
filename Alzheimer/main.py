from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, recall_score, confusion_matrix, f1_score, \
    classification_report
from sklearn.model_selection import train_test_split
import AlzheimerModel
import pandas as pd
import matplotlib.pyplot as plt




data = pd.read_csv('data/alzheimers_preprocessed_non_invasive.csv', delimiter=',')

X = data.drop('Diagnosis', axis=1).values
y = data['Diagnosis'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

predictor = AlzheimerModel.AlzheimerPredictor(model_path='pth/AlzheimerModel.pth')

y_pred = predictor.predict(X_test)
y_pred_proba = predictor.predict_proba(X_test)


flat_list = [val[0] for val in y_pred_proba]

# Print the flattened list
print(flat_list)

y_test_int = [int(val) for val in y_test]

# Print the integers comma-separated
print(", ".join(map(str, y_test_int)))


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

sensitivity = recall_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')
print(f'F1 Score: {f1}')
print(f'ROC AUC: {roc_auc}')
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: {confusion_matrix(y_test, y_pred)}')
print(f'Classification Report: {classification_report(y_test, y_pred)}')

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
