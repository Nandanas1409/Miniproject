import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ==========================
# Configuration
# ==========================
INPUT_PATH = "data/raw/depression_dataset.csv"
MODEL_PATH = "models/depression_rf_model.pkl"

os.makedirs("models", exist_ok=True)

# ==========================
# Load Dataset
# ==========================
print("Loading dataset...")
df = pd.read_csv(INPUT_PATH)

# Keep only Depressive disorder & Healthy control
df = df[df["specific.disorder"].isin(["Depressive disorder", "Healthy control"])]

print("Filtered shape:", df.shape)

# Create binary label
df["label"] = (df["specific.disorder"] == "Depressive disorder").astype(int)

print("\nClass distribution:")
print(df["label"].value_counts())

# Drop non-numeric columns
non_numeric_cols = df.select_dtypes(include=["object"]).columns
df = df.drop(columns=non_numeric_cols)

X = df.drop("label", axis=1)
y = df["label"]

# Handle NaN & infinite
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.dropna(axis=1, how='all')
X = X.fillna(X.median())

print("Feature shape after cleaning:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\nModel Evaluation\n")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, "models/depression_scaler.pkl")

print("\nModel saved successfully.")
