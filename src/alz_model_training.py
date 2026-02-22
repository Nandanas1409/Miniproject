import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os
import numpy as np

# ==========================
# Configuration
# ==========================
INPUT_PATH = "data/processed/alz_window_features.csv"
MODEL_PATH = "models/alz_rf_model.pkl"
THRESHOLD = 0.4   # Custom probability threshold

os.makedirs("models", exist_ok=True)

# ==========================
# Load Processed Data
# ==========================
print("Loading processed features...")
df = pd.read_csv(INPUT_PATH)

print("Total windows:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())

# ==========================
# Chronological Split
# ==========================
df_class0 = df[df["label"] == 0]
df_class1 = df[df["label"] == 1]

split0 = int(0.7 * len(df_class0))
split1 = int(0.7 * len(df_class1))

train_df = pd.concat([
    df_class0.iloc[:split0],
    df_class1.iloc[:split1]
])

test_df = pd.concat([
    df_class0.iloc[split0:],
    df_class1.iloc[split1:]
])

X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]

X_test = test_df.drop("label", axis=1)
y_test = test_df["label"]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================
# Train Improved Random Forest
# ==========================
print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================
# Evaluation with Custom Threshold
# ==========================
print("\nModel Evaluation\n")

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > THRESHOLD).astype(int)

print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
from evaluation_plots import plot_roc_curve, plot_confusion_matrix

plot_roc_curve(y_test, y_prob, "alz")
plot_confusion_matrix(y_test, y_pred, "alz")

# ==========================
# Save Model
# ==========================
joblib.dump(model, MODEL_PATH)

print("\nModel saved at:", MODEL_PATH)