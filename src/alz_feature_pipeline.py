import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch

WINDOW_SIZE = 512
STEP = WINDOW_SIZE // 2  # 50% overlap
FS = 256

def extract_alz_features(raw_df):

    channels = raw_df.columns[:-1] if "status" in raw_df.columns else raw_df.columns
    features = []

    for start in range(0, len(raw_df) - WINDOW_SIZE + 1, STEP):
        window = raw_df.iloc[start:start + WINDOW_SIZE]
        feature_vector = []

        for ch in channels:
            signal = window[ch].values

            # ---------- Statistical ----------
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            var_val = np.var(signal)
            skew_val = skew(signal)
            kurt_val = kurtosis(signal)

            feature_vector.extend([
                mean_val, std_val, var_val, skew_val, kurt_val
            ])

            # ---------- Welch PSD ----------
            freqs, psd = welch(signal, fs=FS)

            def bandpower(fmin, fmax):
                idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                return np.sum(psd[idx])

            delta = bandpower(0.5, 4)
            theta = bandpower(4, 8)
            alpha = bandpower(8, 13)
            beta  = bandpower(13, 30)

            # Log power
            feature_vector.extend([
                np.log(delta + 1e-6),
                np.log(theta + 1e-6),
                np.log(alpha + 1e-6),
                np.log(beta + 1e-6)
            ])

            total_power = delta + theta + alpha + beta
            if total_power == 0:
                delta_rel = theta_rel = alpha_rel = beta_rel = 0
            else:
                delta_rel = delta / total_power
                theta_rel = theta / total_power
                alpha_rel = alpha / total_power
                beta_rel  = beta / total_power

            feature_vector.extend([
                delta_rel, theta_rel, alpha_rel, beta_rel
            ])

            # ---------- Spectral Entropy ----------
            psd_norm = psd / (np.sum(psd) + 1e-12)
            spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
            feature_vector.append(spectral_entropy)

            # ---------- Hjorth ----------
            first_deriv = np.diff(signal)
            second_deriv = np.diff(first_deriv)

            activity = np.var(signal)
            mobility = np.sqrt(np.var(first_deriv) / activity) if activity != 0 else 0
            complexity = (
                np.sqrt(np.var(second_deriv) / np.var(first_deriv)) / mobility
                if mobility != 0 and np.var(first_deriv) != 0 else 0
            )

            feature_vector.extend([activity, mobility, complexity])

        features.append(feature_vector)

    return pd.DataFrame(features)