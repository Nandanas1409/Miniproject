import pandas as pd
import joblib

def preprocess_depression_data(df, scaler):

    trained_features = joblib.load("models/depression_feature_columns.pkl")
    
    # Clone the dataframe to avoid SettingWithCopyWarning
    df_feat = df.copy()

    regions = {
        "Frontal": ["FP1","FP2","F3","F4","F7","F8","FZ"],
        "Central": ["C3","C4","CZ"],
        "Temporal": ["T3","T4","T5","T6"],
        "Parietal": ["P3","P4","PZ"],
        "Occipital": ["O1","O2"]
    }

    # Band features
    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
        band_cols = [c for c in df_feat.columns if band in c.lower() and c.startswith("AB.")]
        for region, electrodes in regions.items():
            region_cols = [c for c in band_cols if any(e in c.upper() for e in electrodes)]
            if region_cols:
                df_feat[f"{region}_{band}_mean"] = df_feat[region_cols].mean(axis=1)

    # Frontal alpha asymmetry
    left_alpha = [c for c in df_feat.columns if "alpha" in c.lower() and "FP1" in c.upper()]
    right_alpha = [c for c in df_feat.columns if "alpha" in c.lower() and "FP2" in c.upper()]
    
    if left_alpha and right_alpha:
        df_feat["frontal_alpha_asymmetry"] = (
            df_feat[right_alpha].mean(axis=1) - df_feat[left_alpha].mean(axis=1)
        )

    # Coherence features
    coh_cols = [c for c in df_feat.columns if c.startswith("COH.")]
    
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
                    df_feat[name] = df_feat[relevant].mean(axis=1)

    # Construct the final feature matrix
    # Fill missing features with 0 to prevent errors
    for col in trained_features:
        if col not in df_feat.columns:
            df_feat[col] = 0
            
    X = df_feat[trained_features]

    # Scale
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=trained_features)

    return X_scaled