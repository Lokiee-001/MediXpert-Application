import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Set style for plots
sns.set_style("whitegrid")

# Example data
y_true = [0, 1, 0, 1, 0, 1]  # True labels (0 = Healthy, 1 = Affected)
y_pred = [0, 1, 0, 0, 0, 1]  # Predicted labels
y_scores = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]  # Predicted probabilities

# Metrics
accuracy = 0.92
precision = 0.90
recall = 0.80
f1_score = 0.85

# Class distribution
class_labels = ['Healthy', 'Affected']
class_sizes = [70, 30]  # Percentage of each class

# Training history
epochs = range(1, 11)
train_accuracy = [0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.93, 0.94]
val_accuracy = [0.55, 0.65, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87, 0.88, 0.89]

# Create a figure with subplots
plt.figure(figsize=(18, 12))

# 1. Bar Plot for Metrics
plt.subplot(2, 3, 1)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1_score]
plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.title('Model Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, value in enumerate(values):
    plt.text(i, value + 0.02, f'{value:.2f}', ha='center')

# 2. Confusion Matrix Heatmap
plt.subplot(2, 3, 2)
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# 3. ROC Curve
plt.subplot(2, 3, 3)
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# 4. Precision-Recall Curve
plt.subplot(2, 3, 4)
precision, recall, _ = precision_recall_curve(y_true, y_scores)
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')

# 5. Pie Chart for Class Distribution
plt.subplot(2, 3, 5)
plt.pie(class_sizes, labels=class_labels, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
plt.title('Class Distribution')

# 6. Line Plot for Training and Validation Accuracy
plt.subplot(2, 3, 6)
plt.plot(epochs, train_accuracy, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)

# Adjust layout and display
plt.tight_layout()
plt.show()