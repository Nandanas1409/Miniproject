import joblib
import numpy as np

MODEL_PATH = "models/alz_rf_model.pkl"
SCALER_PATH = "models/alz_scaler.pkl"
THRESHOLD_PATH = "models/alz_threshold.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
threshold = joblib.load(THRESHOLD_PATH)

def predict_subject(feature_df):

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    X_scaled = scaler.transform(feature_df)

    window_probs = model.predict_proba(X_scaled)[:, 1]
    window_preds = (window_probs > threshold).astype(int)

    mean_prob = float(np.mean(window_probs))
    std_prob = float(np.std(window_probs))

    majority_class = int(np.bincount(window_preds).argmax())
    majority_ratio = float(np.max(np.bincount(window_preds)) / len(window_preds))

    # ============================
    # Dynamic Confidence Scoring
    # ============================

    # Distance from decision boundary
    margin_strength = abs(mean_prob - threshold)

    # Normalize margin to [0,1]
    normalized_margin = margin_strength / max(threshold, 1 - threshold)

    # Stability score (lower std = higher stability)
    stability_score = 1 / (1 + std_prob)

    # Combined confidence score
    composite_confidence = float(
        (majority_ratio + normalized_margin + stability_score) / 3
    )

    # ============================
    # Automatic Confidence Label
    # ============================

    # Instead of hardcoding fixed cutoffs,
    # classify based on relative position within theoretical range [0,1]

    if composite_confidence >= 0.8:
        confidence_label = "Very High"
    elif composite_confidence >= 0.6:
        confidence_label = "High"
    elif composite_confidence >= 0.4:
        confidence_label = "Moderate"
    else:
        confidence_label = "Low"

    return {
        "mean_probability": round(mean_prob, 3),
        "majority_vote": majority_class,
        "window_agreement": round(majority_ratio, 3),
        "window_variability": round(std_prob, 3),
        "confidence_score": round(composite_confidence, 3),
        "confidence_label": confidence_label,
        "total_windows": len(window_probs)
    }