import pandas as pd
import os
from alz_feature_pipeline import extract_alz_features

# ==========================
# Configuration
# ==========================
INPUT_PATH = "data/raw/alzheimers_dataset.csv"
OUTPUT_PATH = "data/processed/alz_window_features.csv"

os.makedirs("data/processed", exist_ok=True)

print("Loading raw dataset...")
df = pd.read_csv(INPUT_PATH)

print("Extracting updated features...")
feature_df = extract_alz_features(df)

# Attach label column (assuming raw has 'status')
if "status" in df.columns:
    # Align window labels
    labels = []
    WINDOW_SIZE = 512
    STEP = WINDOW_SIZE // 2

    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP):
        window = df.iloc[start:start + WINDOW_SIZE]
        labels.append(window["status"].mode()[0])

    feature_df["label"] = labels

else:
    raise ValueError("Raw dataset must contain 'status' column")

feature_df.to_csv(OUTPUT_PATH, index=False)

print("New feature file saved.")
print("Feature shape:", feature_df.shape)