import os
import joblib
import pandas as pd
import time
import shap
from flask import Flask, request, jsonify

app = Flask(__name__)

# CONFIGURATION
MODEL_PATH = "output/xgboost_week2_final.joblib"
# Load the model once at startup for performance
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    # Initialize SHAP explainer once to save time during requests
    explainer = shap.TreeExplainer(model)
else:
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run Week 2 first.")

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()#Measure response time
    
    # 1. Get JSON data
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    
    # 2. Convert to DataFrame
    # Expects a dictionary of features matching the model training columns
    input_df = pd.DataFrame([data])
    
    # 3. Generate Prediction
    # Probability of class 1 failure
    prob = model.predict_proba(input_df)[0][1]
    prediction = int(model.predict(input_df)[0])
    
    # 4. Generate SHAP Explanation
    # We get the SHAP values for this specific instance
    shap_values = explainer.shap_values(input_df)
    
    # Create a dictionary of feature impacts for the API response
    feature_names = input_df.columns.tolist()
    # Using [0] because we are explaining a single prediction
    explanations = dict(zip(feature_names, shap_values[0].tolist()))
    
    # Sort explanations by absolute impact to show the top 3 drivers
    top_drivers = sorted(explanations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    latency_ms = (time.time() - start_time) * 1000

    # 5. Build Response
    response = {
        "failure_prediction": prediction,
        "failure_probability": round(float(prob), 4),
        "top_3_explanation_drivers": dict(top_drivers),
        "api_latency_ms": round(latency_ms, 2),
        "status": "High Risk" if prob > 0.5 else "Stable"
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, port=5000)