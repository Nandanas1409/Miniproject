import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# ==========================
# Configuration
# ==========================
INPUT_PATH = "data/processed/alz_window_features.csv"
MODEL_PATH = "models/alz_rf_model.pkl"
SCALER_PATH = "models/alz_scaler.pkl"
THRESHOLD_PATH = "models/alz_threshold.pkl"

os.makedirs("models", exist_ok=True)

# ==========================
# Load Data
# ==========================
df = pd.read_csv(INPUT_PATH)

X = df.drop("label", axis=1)
y = df["label"]

# ==========================
# Chronological Class-wise Split
# ==========================
df0 = df[df["label"] == 0]
df1 = df[df["label"] == 1]

split0 = int(0.7 * len(df0))
split1 = int(0.7 * len(df1))

train_df = pd.concat([df0.iloc[:split0], df1.iloc[:split1]])
test_df  = pd.concat([df0.iloc[split0:], df1.iloc[split1:]])

X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]

X_test = test_df.drop("label", axis=1)
y_test = test_df["label"]

# ==========================
# Scaling
# ==========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, SCALER_PATH)

# ==========================
# Random Forest
# ==========================
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ==========================
# Calibration
# ==========================
model = CalibratedClassifierCV(rf, method="isotonic")
model.fit(X_train_scaled, y_train)

# ==========================
# Evaluation
# ==========================
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ----- Optimal Threshold (Youden J) -----
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
youden_index = tpr - fpr
best_threshold = thresholds[np.argmax(youden_index)]

print("Optimal Threshold:", round(best_threshold, 3))

y_pred_optimal = (y_prob > best_threshold).astype(int)

print("\nClassification Report (Optimal Threshold)")
print(classification_report(y_test, y_pred_optimal))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# ==========================
# Save Model + Threshold
# ==========================
joblib.dump(model, MODEL_PATH)
joblib.dump(best_threshold, THRESHOLD_PATH)

print("\nModel, scaler, and threshold saved successfully.")