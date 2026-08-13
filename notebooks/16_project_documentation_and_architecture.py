# Databricks notebook source
# MAGIC %md
# MAGIC # IoT Predictive Maintenance Pipeline - Final Project Documentation
# MAGIC
# MAGIC ## Project Overview
# MAGIC
# MAGIC This project builds an end-to-end predictive maintenance pipeline using Databricks, PySpark, Delta tables, MLflow, and machine learning models.
# MAGIC
# MAGIC The goal is to predict Remaining Useful Life (RUL) for aircraft engines and classify failure risk so maintenance teams can identify high-risk engines before failure occurs.
# MAGIC
# MAGIC The project uses the NASA C-MAPSS FD001 turbofan engine dataset. The pipeline starts from raw sensor data, performs cleaning and feature engineering, trains regression and classification models, tracks experiments using MLflow, registers the best models in Unity Catalog, and creates an AI-style dashboard for maintenance decision support.
# MAGIC
# MAGIC ## Business Problem
# MAGIC
# MAGIC Industrial equipment can fail unexpectedly, causing downtime, production delays, and higher maintenance costs.
# MAGIC
# MAGIC Instead of waiting for equipment to fail or following only fixed maintenance schedules, this project uses sensor data to estimate how many cycles an engine has remaining and whether it is at high risk of failure.
# MAGIC
# MAGIC This helps maintenance teams prioritize inspections, reduce unplanned downtime, and make data-driven maintenance decisions.
# MAGIC
# MAGIC ## Main Outcomes
# MAGIC
# MAGIC - Built raw, cleaned, feature, model-ready, inference, and dashboard Delta tables
# MAGIC - Created RUL regression model
# MAGIC - Created failure-risk classification model
# MAGIC - Tracked experiments and metrics using MLflow
# MAGIC - Registered Champion models in Unity Catalog
# MAGIC - Loaded registered models for deployment-style inference
# MAGIC - Generated risk categories and maintenance recommendations
# MAGIC - Built an AI-style dashboard for engine-level decision support

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Model Results
# MAGIC
# MAGIC ### RUL Regression Model
# MAGIC
# MAGIC The selected regression model was a tuned Random Forest Regressor.
# MAGIC
# MAGIC Final regression performance:
# MAGIC
# MAGIC - RMSE: 36.40
# MAGIC - MAE: 26.23
# MAGIC - R²: 0.7215
# MAGIC
# MAGIC This model predicts the Remaining Useful Life of each engine based on sensor patterns, rolling averages, sensor differences, and cycle age.
# MAGIC
# MAGIC ### Failure-Risk Classification Model
# MAGIC
# MAGIC The selected classification model was Logistic Regression.
# MAGIC
# MAGIC Final classification performance:
# MAGIC
# MAGIC - Accuracy: 0.9497
# MAGIC - Precision: 0.9492
# MAGIC - Recall: 0.9497
# MAGIC - F1 Score: 0.9494
# MAGIC - AUC: 0.9838
# MAGIC
# MAGIC This model classifies whether an engine is at high risk of failure based on engineered sensor features.

# COMMAND ----------

# MAGIC %md
# MAGIC ## End-to-End Architecture Flow
# MAGIC
# MAGIC The project follows this pipeline:
# MAGIC
# MAGIC Raw Sensor Data  
# MAGIC         ↓  
# MAGIC Data Cleaning and Validation  
# MAGIC         ↓  
# MAGIC Feature Engineering  
# MAGIC         ↓  
# MAGIC Model-Ready Delta Tables  
# MAGIC         ↓  
# MAGIC Regression and Classification Model Training  
# MAGIC         ↓  
# MAGIC MLflow Experiment Tracking  
# MAGIC         ↓  
# MAGIC Unity Catalog Model Registration  
# MAGIC         ↓  
# MAGIC Champion Model Inference  
# MAGIC         ↓  
# MAGIC Risk Classification and Maintenance Recommendation  
# MAGIC         ↓  
# MAGIC AI Dashboard and Decision Report
# MAGIC
# MAGIC ## Architecture Explanation
# MAGIC
# MAGIC The raw NASA C-MAPSS sensor dataset was first loaded into Databricks and saved as raw Delta tables. After that, the data was cleaned, validated, and prepared for analysis.
# MAGIC
# MAGIC Feature engineering was performed using cycle age, selected sensor readings, rolling averages, and sensor difference features. These features were used to create model-ready datasets for both regression and classification.
# MAGIC
# MAGIC The regression model predicted Remaining Useful Life, while the classification model predicted whether an engine was at high risk of failure. Model training and evaluation were tracked using MLflow.
# MAGIC
# MAGIC The best models were registered in Unity Catalog and assigned Champion aliases. These registered models were then loaded again in a separate inference notebook to simulate deployment-style prediction.
# MAGIC
# MAGIC The final prediction outputs were used to build an AI-style dashboard that shows engine risk levels, maintenance priority, prediction history, and a human-readable maintenance decision report.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta Table Layers
# MAGIC
# MAGIC ### Raw Layer
# MAGIC
# MAGIC - `workspace.default.raw_sensor_training_data`
# MAGIC - `workspace.default.raw_sensor_test_data`
# MAGIC - `workspace.default.raw_sensor_test_rul_summary`
# MAGIC
# MAGIC ### Cleaned Layer
# MAGIC
# MAGIC - `workspace.default.cleaned_sensor_training_data`
# MAGIC - `workspace.default.cleaned_sensor_test_data`
# MAGIC - `workspace.default.cleaned_sensor_test_rul_summary`
# MAGIC
# MAGIC ### Feature Layer
# MAGIC
# MAGIC - `workspace.default.feature_sensor_training_data`
# MAGIC - `workspace.default.feature_sensor_test_data`
# MAGIC
# MAGIC ### Model-Ready Layer
# MAGIC
# MAGIC - `workspace.default.model_ready_sensor_train_regression`
# MAGIC - `workspace.default.model_ready_sensor_train_classification`
# MAGIC - `workspace.default.model_ready_sensor_test`
# MAGIC
# MAGIC ### Model Result Tables
# MAGIC
# MAGIC - `workspace.default.best_rul_prediction_results`
# MAGIC - `workspace.default.best_failure_risk_prediction_results`
# MAGIC - `workspace.default.feature_importance_rul_random_forest`
# MAGIC - `workspace.default.model_comparison_rul_regression_baseline`
# MAGIC - `workspace.default.model_comparison_rul_regression_tuning`
# MAGIC - `workspace.default.model_comparison_failure_risk_classification`
# MAGIC
# MAGIC ### Inference and Dashboard Tables
# MAGIC
# MAGIC - `workspace.default.iot_model_serving_inference_output`
# MAGIC - `workspace.default.iot_latest_engine_status_dashboard`
# MAGIC - `workspace.default.iot_ai_dashboard_kpi_summary`
# MAGIC - `workspace.default.iot_ai_risk_distribution`
# MAGIC - `workspace.default.iot_ai_maintenance_priority_queue`
# MAGIC - `workspace.default.iot_ai_selected_engine_history`
# MAGIC - `workspace.default.iot_ai_selected_engine_report`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Deliverables
# MAGIC
# MAGIC The project was completed through the following notebook flow:
# MAGIC
# MAGIC 1. Databricks setup and dataset loading
# MAGIC 2. Raw data exploration
# MAGIC 3. Data cleaning and validation
# MAGIC 4. Delta table creation
# MAGIC 5. Feature engineering
# MAGIC 6. Model-ready dataset preparation
# MAGIC 7. Baseline RUL regression model
# MAGIC 8. Tuned RUL regression model
# MAGIC 9. Failure-risk classification model
# MAGIC 10. MLflow experiment tracking
# MAGIC 11. Model comparison and selection
# MAGIC 12. Unity Catalog model registration
# MAGIC 13. Registered model validation
# MAGIC 14. Model serving and inference testing
# MAGIC 15. AI prediction application dashboard
# MAGIC 16. Final project documentation and architecture
# MAGIC
# MAGIC Each notebook represents one stage of the end-to-end machine learning lifecycle, from raw data ingestion to AI-style decision support.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Project Completion Summary
# MAGIC
# MAGIC This project successfully demonstrates an end-to-end predictive maintenance solution using Databricks.
# MAGIC
# MAGIC The final pipeline can:
# MAGIC
# MAGIC - Load and manage raw sensor data
# MAGIC - Clean and validate operational records
# MAGIC - Engineer machine learning features
# MAGIC - Train regression and classification models
# MAGIC - Track experiments using MLflow
# MAGIC - Register best models in Unity Catalog
# MAGIC - Load Champion models for inference
# MAGIC - Predict Remaining Useful Life
# MAGIC - Classify failure risk
# MAGIC - Generate maintenance recommendations
# MAGIC - Create AI-style dashboard outputs
# MAGIC
# MAGIC The final dashboard monitored 100 engines. Out of these, 18 engines were classified as High Risk, 15 as Medium Risk, and 67 as Low Risk.
# MAGIC
# MAGIC The average predicted Remaining Useful Life was 87.17 cycles. The lowest predicted Remaining Useful Life was 7.52 cycles, showing that the dashboard can identify engines requiring urgent maintenance attention.

# COMMAND ----------

