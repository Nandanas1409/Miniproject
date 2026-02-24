import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.alz_feature_pipeline import extract_alz_features
from src.depression_feature_pipeline import preprocess_depression_data
from src.eeg_topomap import plot_topomap

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Explainable EEG Clinical Decision Support",
    layout="wide"
)

st.title("🧠 Explainable EEG Clinical Decision Support System")
st.markdown("### Early Alzheimer’s and Depression Detection using EEG-based Machine Learning")

st.sidebar.warning(
    "⚠️ Academic Demonstration Only.\n\n"
    "This system does NOT replace clinical diagnosis."
)

# ==========================================================
# Load Models
# ==========================================================
@st.cache_resource
def load_models():
    alz_model = joblib.load("models/alz_rf_model.pkl")
    dep_model = joblib.load("models/depression_rf_model.pkl")
    dep_scaler = joblib.load("models/depression_scaler.pkl")
    return alz_model, dep_model, dep_scaler

alz_model, dep_model, dep_scaler = load_models()

# ==========================================================
# Tabs
# ==========================================================
tab1, tab2 = st.tabs(["Alzheimer Detection", "Depression Detection"])

# ==========================================================
# ================= ALZHEIMER SECTION ======================
# ==========================================================
with tab1:

    st.header("Alzheimer’s Detection from Raw EEG")

    alz_file = st.file_uploader(
        "Upload Raw EEG CSV (16 channels, ≥512 samples)",
        type=["csv"],
        key="alz_upload"
    )

    if alz_file is not None:

        raw_df = pd.read_csv(alz_file)

        st.subheader("Raw Data Preview")
        st.dataframe(raw_df.head())

        if len(raw_df) < 512:
            st.error("Minimum 512 samples required for spectral windowing.")
        else:

            # ================= Feature Extraction =================
            feature_df = extract_alz_features(raw_df)

            # ================= Prediction =================
            probabilities = alz_model.predict_proba(feature_df)[:, 1]
            patient_score = np.mean(probabilities)

            st.subheader("Prediction Result")
            st.metric("Alzheimer Probability", f"{patient_score:.2f}")

            if patient_score > 0.7:
                st.error("High Likelihood of Alzheimer’s Pattern")
            elif patient_score > 0.4:
                st.warning("Moderate Risk Pattern Detected")
            else:
                st.success("Low Alzheimer Pattern Likelihood")

            confidence = abs(patient_score - 0.5) * 2
            st.write(f"Model Confidence: {confidence:.2f}")
            st.write(f"Total EEG Windows Analyzed: {len(probabilities)}")

            # ======================================================
            # ========== CHANNEL-LEVEL SLOW WAVE INDEX =============
            # ======================================================

            channels = [
                "Fp1","Fp2","F7","F3","Fz","F4","F8",
                "T3","C3","Cz","C4","T4",
                "T5","P3","Pz","P4"
            ]

            BLOCK_SIZE = 15
            channel_slow_index = {}

            for idx, ch in enumerate(channels):
                base = idx * BLOCK_SIZE
                delta_vals = feature_df.iloc[:, base + 5]
                alpha_vals = feature_df.iloc[:, base + 7]
                slow_index = np.mean(delta_vals) - np.mean(alpha_vals)
                channel_slow_index[ch] = slow_index

            st.subheader("EEG Slow-Wave Topographic Map")
            fig = plot_topomap(channel_slow_index, title="Delta-Alpha Slow-Wave Dominance")
            st.pyplot(fig)

            # ======================================================
            # ========== GLOBAL BANDPOWER ANALYSIS =================
            # ======================================================

            delta_vals = []
            theta_vals = []
            alpha_vals = []

            for col in range(0, feature_df.shape[1], BLOCK_SIZE):
                delta_vals.extend(feature_df.iloc[:, col + 5])
                theta_vals.extend(feature_df.iloc[:, col + 6])
                alpha_vals.extend(feature_df.iloc[:, col + 7])

            avg_delta = np.mean(delta_vals)
            avg_theta = np.mean(theta_vals)
            avg_alpha = np.mean(alpha_vals)
            theta_alpha_ratio = avg_theta / avg_alpha if avg_alpha != 0 else 0

            st.subheader("Neurophysiological Spectral Analysis")
            st.write(f"Average Delta Power: {avg_delta:.2f}")
            st.write(f"Average Theta Power: {avg_theta:.2f}")
            st.write(f"Average Alpha Power: {avg_alpha:.2f}")
            st.write(f"Theta/Alpha Ratio: {theta_alpha_ratio:.2f}")

            interpretation = []

            if avg_delta > avg_alpha:
                interpretation.append("Dominance of slow-wave Delta activity detected.")
            if avg_theta > avg_alpha:
                interpretation.append("Elevated Theta activity relative to Alpha rhythm.")
            if theta_alpha_ratio > 1:
                interpretation.append("High Theta/Alpha ratio indicating cortical slowing.")
            if avg_alpha < (avg_theta * 0.8):
                interpretation.append("Reduced Alpha rhythm amplitude observed.")

            if interpretation:
                st.subheader("EEG Interpretation")
                for line in interpretation:
                    st.write(f"• {line}")
            else:
                st.write("No significant spectral imbalance detected.")

            # ======================================================
            # ========== REGION-WISE ANALYSIS ======================
            # ======================================================

            region_map = {
                "Frontal": ["Fp1","Fp2","F7","F3","Fz","F4","F8"],
                "Temporal": ["T3","T4","T5"],
                "Central": ["C3","C4","Cz"],
                "Parietal": ["P3","P4","Pz"]
            }

            st.subheader("Region-wise Slow-Wave Analysis")

            for region, ch_list in region_map.items():
                region_delta = []
                region_alpha = []

                for ch in ch_list:
                    ch_index = channels.index(ch)
                    base = ch_index * BLOCK_SIZE
                    region_delta.extend(feature_df.iloc[:, base + 5])
                    region_alpha.extend(feature_df.iloc[:, base + 7])

                if np.mean(region_delta) > np.mean(region_alpha):
                    st.write(f"• {region}: Relative slow-wave dominance detected.")
                else:
                    st.write(f"• {region}: No significant slowing.")

            # ======================================================
            # ========== MODEL EXPLANATION =========================
            # ======================================================

            st.subheader("Model Decision Explanation")

            st.write(f"""
The model predicted an Alzheimer probability of {patient_score:.2f} 
based on spectral feature patterns across EEG windows.

Primary contributing factors:

• Dominance of low-frequency bands (Delta & Theta)
• Reduced Alpha rhythm strength
• Elevated Theta/Alpha ratio
• Consistency of these patterns across windows
""")

            st.subheader("Clinical Recommendation")

            st.write("""
Findings suggest increased likelihood of Alzheimer’s disease.  
Clinical correlation and neurological evaluation are recommended.

Suggested next steps:

• Cognitive screening (MMSE / MoCA)  
• Structural MRI (hippocampal assessment)  
• Laboratory tests to rule out reversible causes  
• Neurology referral  
""")

# ==========================================================
# ================= DEPRESSION SECTION =====================
# ==========================================================
with tab2:

    st.header("Depression Detection (Feature Dataset Required)")

    dep_file = st.file_uploader(
        "Upload Depression EEG Feature CSV",
        type=["csv"],
        key="dep_upload"
    )

    if dep_file is not None:

        raw_df = pd.read_csv(dep_file)
        X_scaled = preprocess_depression_data(raw_df, dep_scaler)

        probabilities = dep_model.predict_proba(X_scaled)[:, 1]
        patient_score = np.mean(probabilities)

        st.subheader("Prediction Result")
        st.metric("Depression Probability", f"{patient_score:.2f}")

        if patient_score > 0.7:
            st.error("High Likelihood of Depressive EEG Pattern")
        elif patient_score > 0.4:
            st.warning("Moderate Risk Pattern Detected")
        else:
            st.success("Low Depression Pattern Likelihood")

        confidence = abs(patient_score - 0.5) * 2
        st.write(f"Model Confidence: {confidence:.2f}")
        st.write(f"Total Feature Rows Analyzed: {len(probabilities)}")

        # ======================================================
        # 1️⃣ BAND-LEVEL SPECTRAL ANALYSIS
        # ======================================================

        delta_cols = [col for col in raw_df.columns if "delta" in col.lower()]
        theta_cols = [col for col in raw_df.columns if "theta" in col.lower()]
        alpha_cols = [col for col in raw_df.columns if "alpha" in col.lower()]
        beta_cols  = [col for col in raw_df.columns if "beta" in col.lower()]

        st.subheader("Neurophysiological Spectral Analysis")

        if delta_cols and theta_cols and alpha_cols:

            avg_delta = raw_df[delta_cols].mean().mean()
            avg_theta = raw_df[theta_cols].mean().mean()
            avg_alpha = raw_df[alpha_cols].mean().mean()

            st.write(f"Average Delta Activity: {avg_delta:.4f}")
            st.write(f"Average Theta Activity: {avg_theta:.4f}")
            st.write(f"Average Alpha Activity: {avg_alpha:.4f}")

            if avg_alpha != 0:
                ratio = avg_theta / avg_alpha
                st.write(f"Theta/Alpha Ratio: {ratio:.4f}")

        # ======================================================
        # 2️⃣ FRONTAL ALPHA ASYMMETRY (Depression Marker)
        # ======================================================

        st.subheader("Frontal Alpha Asymmetry Analysis")

        fp1_alpha = [col for col in alpha_cols if "fp1" in col.lower()]
        fp2_alpha = [col for col in alpha_cols if "fp2" in col.lower()]

        if fp1_alpha and fp2_alpha:

            left_alpha = raw_df[fp1_alpha].mean().mean()
            right_alpha = raw_df[fp2_alpha].mean().mean()

            asymmetry = right_alpha - left_alpha

            st.write(f"Left Frontal Alpha: {left_alpha:.4f}")
            st.write(f"Right Frontal Alpha: {right_alpha:.4f}")
            st.write(f"Frontal Alpha Asymmetry Index: {asymmetry:.4f}")

            if asymmetry < 0:
                st.write("Relative left-frontal hypoactivation detected.")
            else:
                st.write("No significant left-frontal hypoactivation detected.")
        # ======================================================
        # EEG TOPOGRAPHIC MAP (ALPHA POWER DISTRIBUTION)
        # ======================================================

        st.subheader("EEG Alpha Power Topographic Map")

        alpha_cols = [col for col in raw_df.columns if "alpha" in col.lower()]

        channels = [
            "Fp1","Fp2","F7","F3","Fz","F4","F8",
            "T3","C3","Cz","C4","T4",
            "T5","P3","Pz","P4",
            "O1","O2"
        ]

        channel_alpha = {}

        for ch in channels:
            relevant = [c for c in alpha_cols if ch.upper() in c.upper()]
            if relevant:
                channel_alpha[ch] = raw_df[relevant].mean().mean()
            else:
                channel_alpha[ch] = 0

        fig = plot_topomap(channel_alpha, title="Alpha Power Distribution")
        st.pyplot(fig)
        st.subheader("Topographic Analysis")

        # Classify electrodes based on label pattern
        frontal = []
        posterior = []

        for ch, val in channel_alpha.items():
            if ch.startswith("F") or ch.startswith("FP"):
                frontal.append(val)
            elif ch.startswith("P") or ch.startswith("O"):
                posterior.append(val)

        if frontal and posterior:
            frontal_mean = np.mean(frontal)
            posterior_mean = np.mean(posterior)

            st.write(f"Frontal Mean Alpha: {frontal_mean:.4f}")
            st.write(f"Posterior Mean Alpha: {posterior_mean:.4f}")

            ratio = posterior_mean / (frontal_mean + 1e-9)
            st.write(f"Posterior / Frontal Ratio: {ratio:.4f}")

            if ratio > 1.5:
                st.success("Posterior dominant alpha pattern observed (physiologically typical).")
            elif ratio > 1.0:
                st.info("Mild posterior dominance.")
            else:
                st.warning("Reduced posterior dominance or elevated frontal alpha detected.")
        else:
            st.info("Insufficient regional distribution for anterior–posterior comparison.")
        # ======================================================
        # 3️⃣ HEMISPHERIC ALPHA ASYMMETRY TOPOGRAPHIC MAP
        # ======================================================

        st.subheader("Hemispheric Alpha Asymmetry Topographic Map (Right − Left)")

        alpha_cols = [col for col in raw_df.columns if "alpha" in col.lower()]

        # Symmetric channel pairs
        pairs = [
            ("Fp1", "Fp2"),
            ("F3", "F4"),
            ("F7", "F8"),
            ("C3", "C4"),
            ("P3", "P4"),
            ("T3", "T4"),
            ("O1", "O2")
        ]

        channel_asymmetry = {}

        for left, right in pairs:
            left_cols = [c for c in alpha_cols if left.upper() in c.upper()]
            right_cols = [c for c in alpha_cols if right.upper() in c.upper()]

            if left_cols and right_cols:
                left_val = raw_df[left_cols].mean().mean()
                right_val = raw_df[right_cols].mean().mean()

                asym = right_val - left_val

                # Assign asymmetry value to both channels
                channel_asymmetry[left] = -asym
                channel_asymmetry[right] = asym
            else:
                channel_asymmetry[left] = 0
                channel_asymmetry[right] = 0

        fig = plot_topomap(channel_asymmetry, title="Alpha Asymmetry (Right − Left)")
        st.pyplot(fig) 
        st.subheader("Topographic Interpretation")

        mean_asym = np.mean([abs(v) for v in channel_asymmetry.values()])

        frontal_keys = ["Fp1","Fp2","F3","F4","F7","F8"]
        frontal_asym = np.mean([abs(channel_asymmetry[k]) for k in frontal_keys if k in channel_asymmetry])

        st.write(f"Mean Hemispheric Asymmetry Magnitude: {mean_asym:.4f}")
        st.write(f"Mean Frontal Asymmetry Magnitude: {frontal_asym:.4f}")

        if frontal_asym > 0.02:
            st.warning("Significant frontal alpha asymmetry detected, consistent with depressive EEG patterns.")
        elif frontal_asym > 0.01:
            st.info("Mild frontal hemispheric imbalance observed.")
        else:
            st.success("No significant frontal alpha asymmetry detected.")
        # ======================================================
        # 4️⃣ TOP MODEL FEATURES
        # ======================================================

        st.subheader("Top Contributing EEG Features")

        importances = dep_model.feature_importances_
        feature_names = X_scaled.columns

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        top_features = importance_df.head(10)
        st.bar_chart(top_features.set_index("Feature"))

        # ======================================================
        # 5️⃣ MODEL DECISION EXPLANATION
        # ======================================================

        st.subheader("Model Decision Explanation")

        st.write(f"""
The model predicted a depression probability of {patient_score:.2f}.

The decision was primarily influenced by variations in:

{', '.join(top_features['Feature'].head(5))}

These features represent changes in EEG spectral power
and functional connectivity patterns associated with depressive states.
""")

        # ======================================================
        # 6️⃣ CLINICAL RECOMMENDATION
        # ======================================================

        st.subheader("Clinical Recommendation")

        st.write("""
Findings suggest deviation in EEG connectivity and/or
frontal alpha asymmetry patterns.

Clinical psychiatric evaluation and symptom correlation
are recommended.

Suggested next steps:

• Structured depression assessment (PHQ-9 / HAM-D)  
• Clinical psychiatric consultation  
• Sleep and stress evaluation  
• Consider neuropsychological testing if cognitive symptoms present  
""")