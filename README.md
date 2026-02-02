<div align="center">

# ✈️ Predictive Maintenance with Explainable AI (XAI)
### Remaining Useful Life (RUL) Prediction on NASA CMAPSS Data

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-ff0055?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-API-lightgrey?style=for-the-badge&logo=flask)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)

<p align="center">
  <strong>An end-to-end industrial AI pipeline to predict jet engine failures <br>
  and provide transparent, sensor-level explanations using a real-time monitoring dashboard.</strong>
</p>

</div>

---

## 📖 Project Overview

This repository implements a comprehensive four-stage pipeline for **Predictive Maintenance**. By analyzing the **NASA CMAPSS Jet Engine dataset**, the system predicts imminent failures (within 24 hours/cycles) and utilizes **Explainable AI (XAI)** to solve the "black box" problem. This allows maintenance engineers to understand exactly which sensor patterns (e.g., Sensor 12, 13) are driving a high-risk alert.

## 🛠️ Multi-Week Pipeline

### **Week 1: Data Engineering**
* **Ingestion**: Automated loading of `FD001` through `FD004` datasets.
* **Cleaning**: Removes non-informative constant sensors and handles missing values via forward-fill.
* **Feature Engineering**: Creates temporal context using **Lag Features** and **Rolling Statistics** (Mean/Std Dev) to capture degradation over time.

### **Week 2: High-Performance Modeling**
* **Advanced Architecture**: Utilizes a tuned **XGBoost Classifier** optimized for F1-Score and Recall.
* **Class Imbalance**: Implements `scale_pos_weight` to handle the rare nature of failure events in industrial data.
* **Time-Series Split**: Uses non-shuffled splitting to prevent data leakage.

### **Week 3: Explainable AI (SHAP)**
* **Global Interpretability**: Summary plots identify the most critical sensors across the entire fleet.
* **Local Interpretability**: Force plots explain specific individual predictions, showing which sensor values increased or decreased the risk.
* **Logic Validation**: Cross-references top SHAP features with Pearson correlation to ensure physical model reliability.

### **Week 4: Real-Time Deployment**
* **Backend API**: A **Flask** server that loads the model and SHAP explainer once for low-latency inference (<50ms).
* **Interactive Dashboard**: A **Streamlit** UI allowing users to manually override sensor values via sliders to simulate engine conditions.

---

## 📂 Directory Structure

```text
├── CMaps/                     # 📥 Input: Raw NASA .txt files
├── output/                    # 📤 Output: Models, CSVs, and Visualizations
│   ├── correlation_matrix.png
│   ├── confusion_matrix_week2.png
│   ├── shap_summary_plot.png
│   ├── shap_force_plot_sample.png
│   ├── xgboost_week2_final.joblib    # 🧠 Production Model
│   └── week1_feature_engineered_dataset.csv
├── week1.py                   # 📜 Data & Feature Engineering
├── week2_modeling.py          # 📜 XGBoost & Tuning
├── week3_xai.py               # 📜 SHAP Interpretability
├── week4_app.py               # 🔌 Flask API (Backend)
├── week4_frontend.py          # 📊 Streamlit Dashboard (UI)
├── week4_test.py              # 🧪 API Integration Test
├── requirements.txt           # 📦 Project Dependencies
└── README.md                  # 📄 Documentation
