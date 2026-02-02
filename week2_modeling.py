# WEEK 2: MODELING & HYPERPARAMETER TUNING (RF Baseline + XGBoost High-Performance)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. CLEAN OUTPUT: Suppress library warnings for Python 3.14 stability
warnings.filterwarnings('ignore')

def run_week2_modeling():
    # PATHS
    INPUT_PATH = "output/week1_feature_engineered_dataset.csv"
    OUTPUT_DIR = "output"

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Week 1 output not found. Please run Week 1 code first.")
        return

    # 2. LOAD DATA
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.") 

    # 3. PREPARE DATA
    drop_cols = ["dataset_id", "engine_id", "cycle", "RUL", "failure_24h"]#this column not should be model input
    X = df.drop(columns=drop_cols)
    y = df["failure_24h"]

    # Time-series aware split (No shuffle to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False #data leakage prevent
    )

    # 4. TRAIN RANDOM FOREST (Baseline)
    # Required to establish a baseline before high-performance modeling
    print("\n--- Training Random Forest Baseline ---")
    rf_baseline = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_baseline.fit(X_train, y_train)
    rf_pred = rf_baseline.predict(X_test)
    print(classification_report(y_test, rf_pred))

    # 5. TRAIN HIGH-PERFORMANCE XGBOOST (Hyperparameter Tuning)
    print("\n--- Tuning High-Performance XGBoost ---")
    # scale_pos_weight is critical for handling the rare failure class
    param_dist_xgb = {
        'n_estimators': [100, 200, 300], #no of boosting tree
        'max_depth': [3, 5, 7], #depth of each tree
        'learning_rate': [0.01, 0.05, 0.1], #how much fast to model learn
        'subsample': [0.8, 1.0], 
        'scale_pos_weight': [sum(y==0) / sum(y==1)] 
    }

    xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

    # RandomizedSearchCV focused on F1-Score 
    random_search_xgb = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist_xgb,
        n_iter=5, 
        scoring='f1', 
        cv=3, 
        verbose=1, 
        n_jobs=1, # Fixed at 1 for Python 3.14 / Windows stability
        random_state=42
    )

    random_search_xgb.fit(X_train, y_train)
    best_xgb = random_search_xgb.best_estimator_

    # 6. FINAL EVALUATION
    # Metrics focus: F1-Score and Recall (minimizing false negatives)
    y_pred_xgb = best_xgb.predict(X_test)#predict best model
    print("\n" + "="*45)
    print("WEEK 2 FINAL RESULTS: XGBOOST (F1 & RECALL FOCUS)")
    print("="*45)
    print(classification_report(y_test, y_pred_xgb))

    # 7. SAVE OUTPUTS
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', cmap='Blues')
    plt.title("XGBoost Confusion Matrix")
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_week2.png")
    
    # Save high-performance model for Week 3 SHAP analysis
    joblib.dump(best_xgb, f"{OUTPUT_DIR}/xgboost_week2_final.joblib")
    print(f"\nModel and evaluation saved to '{OUTPUT_DIR}' folder.")

# 8. ENTRY POINT GUARD
if __name__ == "__main__":
    run_week2_modeling()