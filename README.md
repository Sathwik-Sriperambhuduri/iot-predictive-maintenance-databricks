# IoT Predictive Maintenance Pipeline using Databricks

## Project Overview

This project is an end-to-end predictive maintenance pipeline built using Databricks, PySpark, Delta Lake, MLflow, Unity Catalog, and machine learning models.

The goal of the project is to predict the Remaining Useful Life (RUL) of aircraft engines and classify failure risk using sensor data. The final output helps maintenance teams identify high-risk engines early and prioritize inspections before failure occurs.

The project uses the NASA C-MAPSS FD001 turbofan engine dataset.

---

## Business Problem

Industrial equipment can fail unexpectedly, causing downtime, production delays, and increased maintenance costs.

Instead of relying only on fixed maintenance schedules or waiting for failure, this project uses historical sensor readings to estimate engine health and predict which engines are likely to fail soon.

This allows maintenance teams to make data-driven decisions, reduce unplanned downtime, and prioritize urgent maintenance work.

---

## Tech Stack

- Databricks
- PySpark
- Spark SQL
- Delta Lake
- MLflow
- Unity Catalog
- Python
- Scikit-learn / Spark ML
- Random Forest Regression
- Logistic Regression
- Databricks Notebooks
- Delta Tables

---

## Dataset

Dataset used:

- NASA C-MAPSS FD001 Turbofan Engine Degradation Dataset

The raw dataset is not uploaded directly to this repository. The dataset can be downloaded from the NASA Prognostics Center of Excellence data repository.

---

## Project Architecture

The project follows this end-to-end flow:

Raw Sensor Data  
↓  
Data Cleaning and Validation  
↓  
Feature Engineering  
↓  
Model-Ready Delta Tables  
↓  
Regression and Classification Model Training  
↓  
MLflow Experiment Tracking  
↓  
Unity Catalog Model Registration  
↓  
Champion Model Inference  
↓  
Risk Classification and Maintenance Recommendation  
↓  
AI Dashboard and Decision Report  

---

## Key Features

- Loaded raw turbofan engine sensor data into Databricks
- Created raw, cleaned, feature, model-ready, inference, and dashboard Delta tables
- Performed data validation and exploratory analysis
- Engineered rolling averages, sensor differences, and cycle-based features
- Built a Remaining Useful Life regression model
- Built a failure-risk classification model
- Tracked experiments and metrics using MLflow
- Registered best models in Unity Catalog
- Loaded Champion models for deployment-style inference
- Generated engine-level maintenance recommendations
- Built an AI-style predictive maintenance dashboard

---

## Model Results

### RUL Regression Model

Selected model: Tuned Random Forest Regressor

| Metric | Value |
|---|---:|
| RMSE | 36.40 |
| MAE | 26.23 |
| R² | 0.7215 |

The regression model predicts the Remaining Useful Life of each engine based on sensor behavior and engineered features.

### Failure-Risk Classification Model

Selected model: Logistic Regression

| Metric | Value |
|---|---:|
| Accuracy | 0.9497 |
| Precision | 0.9492 |
| Recall | 0.9497 |
| F1 Score | 0.9494 |
| AUC | 0.9838 |

The classification model identifies whether an engine is at high risk of failure.

---

## AI Dashboard Results

The final dashboard monitored 100 engines.

| Dashboard Metric | Value |
|---|---:|
| Total Engines Monitored | 100 |
| High Risk Engines | 18 |
| Medium Risk Engines | 15 |
| Low Risk Engines | 67 |
| Average Predicted RUL | 87.17 |
| Lowest Predicted RUL | 7.52 |
| Highest Predicted RUL | 191.23 |

For Engine 35, the dashboard showed:

- Predicted RUL: 8.76 cycles
- Failure Risk Probability: 99.59%
- Current Risk Category: High Risk
- First Medium Risk Cycle: 133
- First High Risk Cycle: 161
- Recommended Action: Immediate maintenance inspection recommended

---

## Repository Structure

    iot-predictive-maintenance-databricks/
    │
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    │
    ├── notebooks/
    │   ├── 01_databricks_setup_and_data_loading.ipynb
    │   ├── 02_raw_data_exploration.ipynb
    │   ├── 03_data_cleaning_and_validation.ipynb
    │   ├── 04_delta_table_creation.ipynb
    │   ├── 05_feature_engineering.ipynb
    │   ├── 06_model_ready_dataset_preparation.ipynb
    │   ├── 07_baseline_rul_regression_model.ipynb
    │   ├── 08_tuned_rul_regression_model.ipynb
    │   ├── 09_failure_risk_classification_model.ipynb
    │   ├── 10_mlflow_experiment_tracking.ipynb
    │   ├── 11_model_comparison_and_selection.ipynb
    │   ├── 12_unity_catalog_model_registration.ipynb
    │   ├── 13_registered_model_validation.ipynb
    │   ├── 14_model_serving_and_inference_testing.ipynb
    │   ├── 15_ai_prediction_application_dashboard.ipynb
    │   └── 16_project_documentation_and_architecture.ipynb
    │
    ├── docs/
    │   ├── architecture.md
    │   ├── delta_tables.md
    │   ├── model_results.md
    │   └── project_summary.md
    │
    ├── screenshots/
    │   ├── mlflow/
    │   ├── model_registry/
    │   ├── dashboard/
    │   └── results/
    │
    ├── presentation/
    │   └── IoT_Predictive_Maintenance_Training_Presentation.pptx
    │
    └── data/
        └── README.md

---

## Notebooks Completed

1. Databricks setup and dataset loading
2. Raw data exploration
3. Data cleaning and validation
4. Delta table creation
5. Feature engineering
6. Model-ready dataset preparation
7. Baseline RUL regression model
8. Tuned RUL regression model
9. Failure-risk classification model
10. MLflow experiment tracking
11. Model comparison and selection
12. Unity Catalog model registration
13. Registered model validation
14. Model serving and inference testing
15. AI prediction application dashboard
16. Final project documentation and architecture

---

## Final Output

The project produces:

- RUL predictions
- Failure-risk predictions
- Risk categories
- Maintenance recommendations
- Engine-level priority queue
- Single-engine AI decision report
- Dashboard-ready Delta tables

---

## Future Improvements

Future enhancements can include:

- Databricks model serving endpoint
- API-based real-time prediction
- Streamlit or web-based dashboard
- Automated model retraining workflow
- CI/CD integration using GitHub and Jenkins
- Monitoring for model drift and data drift

---

## Project Status

Completed initial end-to-end Databricks training project with model development, MLflow tracking, model registration, inference testing, and AI-style dashboard outputs.
