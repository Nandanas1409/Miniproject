import pandas as pd
import joblib
from explainability import generate_shap_plots

MODEL_PATH = "models/alz_rf_model.pkl"
DATA_PATH = "data/processed/alz_window_features.csv"

print("Loading Alzheimer model...")
model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
X = df.drop("label", axis=1)

generate_shap_plots(model, X, model_name="alz")
