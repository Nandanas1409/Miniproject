import shap
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_shap_plots(model, X, model_name, output_dir="reports"):
    """
    Generates SHAP summary plot for tree-based models.
    Safe for script execution (no IPython dependency).
    """

    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating SHAP plots for {model_name}...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle binary classifier output formats
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_vals = shap_values[:, :, 1]
    else:
        shap_vals = shap_values

    # Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X, show=False)
    plt.tight_layout()

    summary_path = f"{output_dir}/{model_name}_shap_summary.png"
    plt.savefig(summary_path, dpi=300)
    plt.close()

    print(f"Saved SHAP summary plot: {summary_path}")

    return summary_path
