import requests
import pandas as pd
import json

def test_api():
    # Load your engineered dataset to get a sample row
    DATA_PATH = "output/week1_feature_engineered_dataset.csv"
    df = pd.read_csv(DATA_PATH)
    
    # Drop non-feature columns to match model input
    drop_cols = ["dataset_id", "engine_id", "cycle", "RUL", "failure_24h"]
    X = df.drop(columns=drop_cols)
    
    # Take a sample row, the last one which is likely high risk
    sample_data = X.iloc[-1].to_dict()
    
    url = "http://127.0.0.1:5000/predict"
    
    print("--- Sending Request to Flask API ---")
    try:
        response = requests.post(url, json=sample_data) #Sending data to the API
        result = response.json() #Convert API response to Python dictionary
        
        print(f"Status Code: {response.status_code}")
        print(f"Prediction: {result['status']}")
        print(f"Probability: {result['failure_probability']}")
        print(f"Latency: {result['api_latency_ms']}ms")
        print(f"Top Drivers: {result['top_3_explanation_drivers']}")
        
        if result['api_latency_ms'] < 50:
            print("SUCCESS: Latency is within target (<50ms).")
        else:
            print("WARNING: Latency exceeds 50ms target.")
            
    except Exception as e:
        print(f"Error connecting to API: {e}")

if __name__ == "__main__":
    test_api()