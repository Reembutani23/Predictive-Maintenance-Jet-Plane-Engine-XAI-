# WEEK 3: INTERPRETABILITY & TRUST (XAI)
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

def run_week3_xai():
    # PATHS
    MODEL_PATH = "output/xgboost_week2_final.joblib"
    DATA_PATH = "output/week1_feature_engineered_dataset.csv"
    OUTPUT_DIR = "output"

    # Check if Week 2 outputs exist
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Week 2 model not found at {MODEL_PATH}. Please run Week 2 code first.")
        return

    # 1. LOAD MODEL AND DATA
    print("--- Loading Model and Data ---")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    # Prepare features (same logic as Week 2)
    drop_cols = ["dataset_id", "engine_id", "cycle", "RUL", "failure_24h"]
    X = df.drop(columns=drop_cols)
    
    # We will use a subset of the test data for SHAP to speed up computation
    # Taking the last 500 rows (most recent cycles)
    X_test_subset = X.tail(500)

    # 2. INITIALIZE SHAP EXPLAINER
    print("--- Calculating SHAP Values (This may take a moment) ---")
    # TreeExplainer is optimized for XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_subset)

    # 3. SUMMARY PLOT
    # Shows the global importance of features and their impact direction
    print("--- Generating Summary Plot ---")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_subset, show=False)
    plt.title("SHAP Summary Plot: Feature Impact on Failure Prediction")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary_plot.png")
    plt.close()

    # 4. FORCE PLOT (Individual Prediction Explanation)
    # Explain the very last observation in the dataset
    print("--- Generating Force Plot for a Single Prediction ---")
    # We choose the last row , The last row represents the most recent engine cycle
    instance_index = -1 
    
    # Generate the force plot
    # Note: Force plots are usually HTML-based, we save the explanation data
    expected_value = explainer.expected_value
    if isinstance(expected_value, list): # Handle multi-class output if necessary
        expected_value = expected_value[1]
        
    shap.force_plot(
        expected_value, 
        shap_values[instance_index], 
        X_test_subset.iloc[instance_index], 
        matplotlib=True, 
        show=False
    )
    plt.title(f"Force Plot for Last Data Point")
    plt.savefig(f"{OUTPUT_DIR}/shap_force_plot_sample.png", bbox_inches='tight')
    plt.close()

    # 5. VALIDATE MODEL LOGIC (Project Requirement)
    # Checking correlation of top features with high risk
    print("\n--- Validating Model Logic ---")
    # Get mean absolute SHAP values to find top features
    top_features_idx = np.abs(shap_values).mean(0).argsort()[-3:][::-1]
    top_features = X.columns[top_features_idx].tolist()
    
    print(f"Top 3 influential sensors: {top_features}")
    for feature in top_features:
        correlation = df[feature].corr(df['failure_24h'])
        print(f"Validation: Correlation of {feature} with failure is {correlation:.2f}")

    print(f"\nWeek 3 completed. SHAP plots saved to '{OUTPUT_DIR}' folder.")

# ENTRY POINT
if __name__ == "__main__":
    # Ensure shap is installed: pip install shap
    try:
        run_week3_xai()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Tip: Ensure you have 'shap' installed (pip install shap).")