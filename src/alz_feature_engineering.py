import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import os

# ==========================
# Configuration
# ==========================
INPUT_PATH = "data/raw/alzheimers_dataset.csv"
OUTPUT_PATH = "data/processed/alz_window_features.csv"

WINDOW_SIZE = 512
STEP = WINDOW_SIZE // 2   # 50% overlap
FS = 256

os.makedirs("data/processed", exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(INPUT_PATH)

channels = df.columns[:-1]

features = []
labels = []

print("Extracting enhanced spectral + ratio features...")

for start in range(0, len(df) - WINDOW_SIZE, STEP):
    window = df.iloc[start:start + WINDOW_SIZE]
    feature_vector = []

    for ch in channels:
        signal = window[ch].values

        # Statistical features
        feature_vector.extend([
            np.mean(signal),
            np.std(signal),
            np.var(signal),
            skew(signal),
            kurtosis(signal)
        ])

        # Welch PSD
        freqs, psd = welch(signal, fs=FS)

        def bandpower(fmin, fmax):
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            return np.sum(psd[idx])

        delta = bandpower(0.5, 4)
        theta = bandpower(4, 8)
        alpha = bandpower(8, 13)
        beta = bandpower(13, 30)

        feature_vector.extend([delta, theta, alpha, beta])

        def safe_ratio(a, b):
            return a / b if b != 0 else 0

        feature_vector.extend([
            safe_ratio(theta, alpha),
            safe_ratio(delta, alpha),
            safe_ratio(delta, beta),
            safe_ratio(theta, beta),
            safe_ratio(delta + theta, alpha + beta),
            safe_ratio(alpha, beta)
        ])

    features.append(feature_vector)
    labels.append(window["status"].mode()[0])

feature_names = []
for ch in channels:
    feature_names.extend([
        f"{ch}_mean",
        f"{ch}_std",
        f"{ch}_var",
        f"{ch}_skew",
        f"{ch}_kurt",
        f"{ch}_delta",
        f"{ch}_theta",
        f"{ch}_alpha",
        f"{ch}_beta",
        f"{ch}_theta_alpha",
        f"{ch}_delta_alpha",
        f"{ch}_delta_beta",
        f"{ch}_theta_beta",
        f"{ch}_slow_fast_ratio",
        f"{ch}_alpha_beta"
    ])

features_df = pd.DataFrame(features, columns=feature_names)
features_df["label"] = labels

features_df.to_csv(OUTPUT_PATH, index=False)

print("Feature extraction completed.")
print("Total windows:", len(features_df))
print("Features per window:", len(feature_names))