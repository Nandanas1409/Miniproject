import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
import numpy as np

def plot_roc_curve(y_true, y_prob, model_name, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    plt.tight_layout()

    path = f"{output_dir}/{model_name}_roc_curve.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved ROC curve: {path}")
    return path


def plot_confusion_matrix(y_true, y_pred, model_name, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()

    path = f"{output_dir}/{model_name}_confusion_matrix.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved Confusion Matrix: {path}")
    return path
