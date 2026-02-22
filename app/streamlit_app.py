import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# ==========================================================
# Add src folder to system path
# ==========================================================
sys.path.append(os.path.abspath("src"))

from alz_feature_pipeline import extract_alz_features
from depression_feature_pipeline import preprocess_depression_data

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Explainable EEG Clinical Decision Support",
    layout="wide"
)

st.title("🧠 Explainable EEG Clinical Decision Support System")
st.markdown("### Early Alzheimer’s and Depression Detection using Machine Learning")

st.sidebar.warning(
    "⚠️ This system is for academic demonstration only.\n\n"
    "It is NOT intended for clinical diagnosis."
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
# Create Tabs
# ==========================================================
tab1, tab2 = st.tabs(["Alzheimer Detection", "Depression Detection"])

# ==========================================================
# ================= ALZHEIMER SECTION ======================
# ==========================================================
with tab1:

    st.header("Alzheimer’s Detection from Raw EEG")

    uploaded_file = st.file_uploader(
        "Upload Raw EEG CSV (Alzheimer Dataset Format)",
        type=["csv"],
        key="alz_upload"
    )

    if uploaded_file is not None:

        raw_df = pd.read_csv(uploaded_file)

        st.subheader("Raw Data Preview")
        st.dataframe(raw_df.head())

        try:
            st.info("Extracting EEG features...")

            feature_df = extract_alz_features(raw_df)

            if feature_df.shape[0] == 0:
                st.error("Not enough data for windowing. Ensure EEG length > 512 samples.")
            else:
                probabilities = alz_model.predict_proba(feature_df)[:, 1]
                patient_score = np.mean(probabilities)

                st.subheader("Prediction Result")

                st.metric("Alzheimer Probability", f"{patient_score:.2f}")

                # Risk Interpretation
                if patient_score > 0.7:
                    st.error("High Risk of Alzheimer’s")
                elif patient_score > 0.4:
                    st.warning("Moderate Risk of Alzheimer’s")
                else:
                    st.success("Low Risk of Alzheimer’s")

                # Confidence Score
                confidence = abs(patient_score - 0.5) * 2
                st.write(f"Model Confidence: {confidence:.2f}")

                # Additional Info
                st.write(f"Total EEG Windows Analyzed: {len(probabilities)}")

        except Exception as e:
            st.error(f"Error during processing: {e}")

# ==========================================================
# ================= DEPRESSION SECTION =====================
# ==========================================================
with tab2:

    st.header("Depression Detection (Depressive vs Healthy Control)")

    uploaded_file = st.file_uploader(
        "Upload Depression EEG Feature CSV",
        type=["csv"],
        key="dep_upload"
    )

    if uploaded_file is not None:

        raw_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(raw_df.head())

        try:
            X_scaled = preprocess_depression_data(raw_df, dep_scaler)

            probabilities = dep_model.predict_proba(X_scaled)[:, 1]
            patient_score = np.mean(probabilities)

            st.subheader("Prediction Result")

            st.metric("Depression Probability", f"{patient_score:.2f}")

            # Risk Interpretation
            if patient_score > 0.7:
                st.error("High Risk of Depressive Disorder")
            elif patient_score > 0.4:
                st.warning("Moderate Risk of Depressive Disorder")
            else:
                st.success("Low Risk of Depressive Disorder")

            confidence = abs(patient_score - 0.5) * 2
            st.write(f"Model Confidence: {confidence:.2f}")

            st.write(f"Total Samples Analyzed: {len(probabilities)}")

        except Exception as e:
            st.error(f"Error during processing: {e}")
