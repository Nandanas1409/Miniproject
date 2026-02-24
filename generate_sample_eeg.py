import numpy as np
import pandas as pd

# EEG channels (must match your Alzheimer model training)
channels = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "T3","C3","Cz","C4","T4",
    "T5","P3","Pz","P4"
]

# Sampling frequency
FS = 256

# Generate 2048 samples (~8 seconds)
samples = 2048

# Create synthetic EEG signal (mix of rhythms + noise)
time = np.arange(samples) / FS

data = []

for ch in channels:
    signal = (
        20 * np.sin(2 * np.pi * 2 * time) +     # delta (2 Hz)
        10 * np.sin(2 * np.pi * 6 * time) +     # theta (6 Hz)
        5  * np.sin(2 * np.pi * 10 * time) +    # alpha (10 Hz)
        np.random.normal(0, 5, samples)         # noise
    )
    data.append(signal)

# Transpose to shape (samples, channels)
data = np.array(data).T

df = pd.DataFrame(data, columns=channels)

df.to_csv("sample_alz_raw_eeg_2048.csv", index=False)

print("Sample EEG file created: sample_alz_raw_eeg_2048.csv")