import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import numpy as np
import os

# ==========================
# Paths
# ==========================
MODEL_PATH = "models/depression_rf_model.pkl"
SCALER_PATH = "models/depression_scaler.pkl"
DATA_PATH = "data/raw/depression_dataset.csv"

os.makedirs("reports", exist_ok=True)

print("Loading model and data...")

# ==========================
# Load Model & Scaler
# ==========================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv(DATA_PATH)

# Filter Depressive vs Healthy only
df = df[df["specific.disorder"].isin(["Depressive disorder", "Healthy control"])]

df["label"] = (df["specific.disorder"] == "Depressive disorder").astype(int)

# Drop non-numeric columns
non_numeric = df.select_dtypes(include=["object"]).columns
df = df.drop(columns=non_numeric)

X = df.drop("label", axis=1)
y = df["label"]

# ==========================
# Clean Data
# ==========================
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.dropna(axis=1, how='all')
X = X.fillna(X.median())

# ==========================
# Scale Data
# ==========================
X_scaled = scaler.transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# ==========================
# ROC Curve
# ==========================
y_prob = model.predict_proba(X_scaled)[:, 1]

fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Depression ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("reports/depression_roc_curve.png", dpi=300)
plt.close()

print("Saved ROC curve.")

# ==========================
# Confusion Matrix
# ==========================
y_pred = model.predict(X_scaled)
cm = confusion_matrix(y, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Depression Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/depression_confusion_matrix.png", dpi=300)
plt.close()

print("Saved confusion matrix.")

# ==========================
# SHAP Summary (Top 20 Features)
# ==========================
print("Generating SHAP summary...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled_df)

# Handle different SHAP output formats safely
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    shap_vals = shap_values[:, :, 1]
else:
    shap_vals = shap_values

# Force 2D shape
shap_vals = np.array(shap_vals)
if shap_vals.ndim > 2:
    shap_vals = shap_vals.reshape(shap_vals.shape[0], -1)

# Compute mean absolute SHAP importance
mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

# Ensure 1D
mean_abs_shap = np.array(mean_abs_shap).flatten()

# Get top 20 indices
top_indices = np.argsort(mean_abs_shap)[-20:]
top_indices = np.array(top_indices).flatten()

# Select safely
X_top = X_scaled_df.iloc[:, top_indices]
shap_top = shap_vals[:, top_indices]

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_top, X_top, show=False)
plt.tight_layout()
plt.savefig("reports/depression_shap_summary.png", dpi=300)
plt.close()

print("Saved SHAP summary (top 20 features).")

print("\nDepression analysis complete.")
