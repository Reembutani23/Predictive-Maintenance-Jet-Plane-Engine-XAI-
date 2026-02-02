import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")

st.title("🛠️ Jet Engine Health Monitoring Dashboard")
st.markdown("### Real-time Failure Prediction & Explainable AI (XAI)")

# 1. Sidebar for Input
st.sidebar.header("Manual Sensor Override")
# We'll load a sample for the default values
DATA_PATH = "output/week1_feature_engineered_dataset.csv"
df = pd.read_csv(DATA_PATH)
sample_row = df.iloc[-1]

# Create sliders for the top 3 drivers identified in your test
s12 = st.sidebar.slider("Sensor 12", 500.0, 600.0, float(sample_row['sensor_12']))
s13 = st.sidebar.slider("Sensor 13", 2300.0, 2400.0, float(sample_row['sensor_13']))
lag2 = st.sidebar.slider("Sensor 4 Lag 2", 1000.0, 1500.0, float(sample_row['sensor_4_lag_2']))

# Prepare data for API (using sample_row as base)
input_data = sample_row.drop(["dataset_id", "engine_id", "cycle", "RUL", "failure_24h"]).to_dict()
input_data['sensor_12'] = s12
input_data['sensor_13'] = s13
input_data['sensor_4_lag_2'] = lag2

# 2. API Call Logic
if st.button("Analyze Engine Health"):
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
        res = response.json()

        # Row 1: Key Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            color = "inverse" if res['status'] == "High Risk" else "normal"
            st.metric("Status", res['status'], delta=None, delta_color=color)
        with col2:
            st.metric("Failure Probability", f"{res['failure_probability']*100:.2f}%")
        with col3:
            st.metric("API Latency", f"{res['api_latency_ms']}ms")

        # Row 2: Explanations
        st.subheader("Why this prediction?")
        drivers = res['top_3_explanation_drivers']
        
        # Plotting the SHAP Drivers
        fig, ax = plt.subplots()
        ax.barh(list(drivers.keys()), list(drivers.values()), color=['#ff4b4b' if x > 0 else '#0068c9' for x in drivers.values()])
        ax.set_title("Top Feature Impacts (SHAP)")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not connect to Flask API. Ensure week4_app.py is running! Error: {e}")