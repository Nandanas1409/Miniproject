import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch

WINDOW_SIZE = 512
FS = 256

def extract_alz_features(raw_df):
    """
    Takes raw EEG dataframe and returns window-level feature dataframe.
    """

    channels = raw_df.columns[:-1] if "status" in raw_df.columns else raw_df.columns

    features = []
    for start in range(0, len(raw_df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = raw_df.iloc[start:start + WINDOW_SIZE]
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

            def safe_ratio(a, b):
                return a / b if b != 0 else 0

            feature_vector.extend([
                delta,
                theta,
                alpha,
                beta,
                safe_ratio(theta, alpha),
                safe_ratio(delta, alpha),
                safe_ratio(delta, beta),
                safe_ratio(theta, beta),
                safe_ratio(delta + theta, alpha + beta),
                safe_ratio(alpha, beta)
            ])

        features.append(feature_vector)

    feature_df = pd.DataFrame(features)

    return feature_df
