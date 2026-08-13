# Project Architecture

## End-to-End Architecture Flow

The IoT Predictive Maintenance pipeline follows this flow:

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

## Architecture Explanation

The project starts with NASA C-MAPSS turbofan engine sensor data. The raw training, test, and RUL label files are loaded into Databricks and stored as Delta tables.

The data is then cleaned and validated to remove inconsistencies and prepare it for feature engineering. Feature engineering includes cycle age, selected sensor readings, rolling averages, and sensor difference features.

The model-ready datasets are used for two machine learning tasks:

- RUL regression to predict Remaining Useful Life
- Failure-risk classification to classify high-risk engines

MLflow is used to track model experiments, metrics, and results. The best models are registered in Unity Catalog and assigned Champion aliases.

The registered Champion models are loaded again for deployment-style inference. The inference output is then used to generate risk categories, maintenance recommendations, priority queues, and AI-style dashboard outputs.

## Main Components

- Databricks workspace
- PySpark data processing
- Delta Lake tables
- Feature engineering pipeline
- Spark ML models
- MLflow tracking
- Unity Catalog model registry
- Inference workflow
- AI dashboard tables
