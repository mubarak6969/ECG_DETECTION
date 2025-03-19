import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and predictions
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

# Load models
cnn_model = load_model('cnn_model.h5')
rcnn_model = load_model('rcnn_model.h5')

# Generate predictions
cnn_pred = (cnn_model.predict(X_val) > 0.5).astype(int)
rcnn_pred = (rcnn_model.predict(X_val) > 0.5).astype(int)

# Metrics
cnn_report = classification_report(y_val, cnn_pred, output_dict=True)
rcnn_report = classification_report(y_val, rcnn_pred, output_dict=True)
cnn_cm = confusion_matrix(y_val, cnn_pred)
rcnn_cm = confusion_matrix(y_val, rcnn_pred)

# Print comparison
print("CNN vs. RCNN Comparison:")
print(f"CNN Accuracy: {cnn_report['accuracy']:.4f}")
print(f"RCNN Accuracy: {rcnn_report['accuracy']:.4f}")
print(f"CNN Macro Avg F1: {cnn_report['macro avg']['f1-score']:.4f}")
print(f"RCNN Macro Avg F1: {rcnn_report['macro avg']['f1-score']:.4f}")

# Plot comparison bar chart
metrics = ['Accuracy', 'Macro F1']
cnn_values = [cnn_report['accuracy'], cnn_report['macro avg']['f1-score']]
rcnn_values = [rcnn_report['accuracy'], rcnn_report['macro avg']['f1-score']]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, cnn_values, width, label='CNN', color='skyblue')
ax.bar(x + width/2, rcnn_values, width, label='RCNN', color='salmon')
ax.set_ylabel('Score')
ax.set_title('CNN vs. RCNN Performance')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.savefig('model_comparison.png')
plt.close()

# Save results to text file
with open('model_comparison.txt', 'w') as f:
    f.write("CNN Classification Report:\n")
    f.write(classification_report(y_val, cnn_pred))
    f.write("\nCNN Confusion Matrix:\n")
    f.write(str(cnn_cm))
    f.write("\n\nRCNN Classification Report:\n")
    f.write(classification_report(y_val, rcnn_pred))
    f.write("\nRCNN Confusion Matrix:\n")
    f.write(str(rcnn_cm))