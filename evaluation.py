# visualize_metrics.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# Sample data (replace with real model predictions)
# -------------------------------
# Ground truth (actual labels from annotated video)
y_true = [
    "Alert", "Alert", "Drowsy - Eyes Closed", "Alert", "Drowsy - Yawning",
    "Alert", "Drowsy - Eyes Closed", "Drowsy - Yawning", "Alert", "Alert"
]

# Model predictions (from CNN + EAR/MAR logic)
y_pred = [
    "Alert", "Drowsy - Eyes Closed", "Drowsy - Eyes Closed", "Alert", "Drowsy - Yawning",
    "Alert", "Alert", "Drowsy - Yawning", "Alert", "Drowsy - Eyes Closed"
]

# -------------------------------
# Compute metrics
# -------------------------------
labels = ["Alert", "Drowsy - Eyes Closed", "Drowsy - Yawning"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, labels=labels, average=None)
recall = recall_score(y_true, y_pred, labels=labels, average=None)
f1 = f1_score(y_true, y_pred, labels=labels, average=None)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_true, y_pred, labels=labels))

# -------------------------------
# Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Driver Fatigue Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# Bar plot of precision, recall, f1
# -------------------------------
import numpy as np

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(8,5))
plt.bar(x - width, precision, width, label="Precision", color="#4caf50")
plt.bar(x, recall, width, label="Recall", color="#2196f3")
plt.bar(x + width, f1, width, label="F1-Score", color="#ff9800")

plt.xticks(x, labels)
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title("Evaluation Metrics per Class")
plt.legend()
plt.show()
