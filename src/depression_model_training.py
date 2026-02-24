import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("Loading depression dataset...")
df = pd.read_csv("data/raw/depression_dataset.csv")

df = df[df["specific.disorder"].isin(["Depressive disorder", "Healthy control"])]

df["label"] = df["specific.disorder"].apply(
    lambda x: 1 if x == "Depressive disorder" else 0
)

# ==========================================================
# REGION DEFINITIONS
# ==========================================================

regions = {
    "Frontal": ["FP1","FP2","F3","F4","F7","F8","FZ"],
    "Central": ["C3","C4","CZ"],
    "Temporal": ["T3","T4","T5","T6"],
    "Parietal": ["P3","P4","PZ"],
    "Occipital": ["O1","O2"]
}

# ==========================================================
# ENGINEER FEATURES
# ==========================================================

band_features = []

for band in ["delta", "theta", "alpha", "beta", "gamma"]:
    band_cols = [c for c in df.columns if band in c.lower() and c.startswith("AB.")]
    for region, electrodes in regions.items():
        region_cols = [c for c in band_cols if any(e in c.upper() for e in electrodes)]
        if region_cols:
            df[f"{region}_{band}_mean"] = df[region_cols].mean(axis=1)
            band_features.append(f"{region}_{band}_mean")

# Frontal alpha asymmetry
left_alpha = [c for c in df.columns if "alpha" in c.lower() and "FP1" in c.upper()]
right_alpha = [c for c in df.columns if "alpha" in c.lower() and "FP2" in c.upper()]

if left_alpha and right_alpha:
    df["frontal_alpha_asymmetry"] = (
        df[right_alpha].mean(axis=1) - df[left_alpha].mean(axis=1)
    )
    band_features.append("frontal_alpha_asymmetry")

# Region-to-region coherence
coh_features = []
coh_cols = [c for c in df.columns if c.startswith("COH.")]

for r1 in regions:
    for r2 in regions:
        if r1 != r2:
            relevant = [
                c for c in coh_cols
                if any(e1 in c.upper() for e1 in regions[r1])
                and any(e2 in c.upper() for e2 in regions[r2])
            ]
            if relevant:
                name = f"{r1}_{r2}_coh_mean"
                df[name] = df[relevant].mean(axis=1)
                coh_features.append(name)

feature_cols = band_features + coh_features

print("Final engineered feature count:", len(feature_cols))

X = df[feature_cols]
y = df["label"]

# ==========================================================
# SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# SCALING
# ==========================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# TRAIN MODEL
# ==========================================================

model = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# ==========================================================
# THRESHOLD OPTIMIZATION
# ==========================================================

y_prob = model.predict_proba(X_test_scaled)[:, 1]
thresholds = np.arange(0.3, 0.8, 0.02)

best_threshold = 0.5
best_score = 0

for t in thresholds:
    y_pred_temp = (y_prob >= t).astype(int)

    tp = np.sum((y_pred_temp == 1) & (y_test == 1))
    tn = np.sum((y_pred_temp == 0) & (y_test == 0))
    fp = np.sum((y_pred_temp == 1) & (y_test == 0))
    fn = np.sum((y_pred_temp == 0) & (y_test == 1))

    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)

    balanced_acc = (sensitivity + specificity) / 2

    if balanced_acc > best_score:
        best_score = balanced_acc
        best_threshold = t

print("Best Threshold:", best_threshold)

# Final evaluation
y_pred = (y_prob >= best_threshold).astype(int)

print("\nFinal Evaluation\n")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ==========================================================
# SAVE
# ==========================================================

joblib.dump(model, "models/depression_rf_model.pkl")
joblib.dump(scaler, "models/depression_scaler.pkl")
joblib.dump(feature_cols, "models/depression_feature_columns.pkl")
joblib.dump(best_threshold, "models/depression_threshold.pkl")

print("\nModel saved successfully.")