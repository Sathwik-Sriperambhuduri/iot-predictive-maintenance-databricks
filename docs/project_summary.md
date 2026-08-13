# Project Summary

## IoT Predictive Maintenance Pipeline

This project builds an end-to-end predictive maintenance pipeline using Databricks, PySpark, Delta Lake, MLflow, Unity Catalog, and machine learning models.

The goal is to predict the Remaining Useful Life of aircraft engines and classify failure risk using NASA C-MAPSS turbofan engine sensor data.

## Business Problem

Industrial equipment can fail unexpectedly, causing downtime, higher maintenance costs, and operational delays.

This project helps maintenance teams identify engines that are likely to fail soon so they can plan inspections before failure occurs.

## Final Results

The final dashboard monitored 100 engines.

| Metric | Value |
|---|---:|
| Total Engines | 100 |
| High Risk Engines | 18 |
| Medium Risk Engines | 15 |
| Low Risk Engines | 67 |
| Average Predicted RUL | 87.17 |
| Lowest Predicted RUL | 7.52 |
| Highest Predicted RUL | 191.23 |

## Model Results

### RUL Regression

| Metric | Value |
|---|---:|
| RMSE | 36.40 |
| MAE | 26.23 |
| R² | 0.7215 |

### Failure-Risk Classification

| Metric | Value |
|---|---:|
| Accuracy | 0.9497 |
| Precision | 0.9492 |
| Recall | 0.9497 |
| F1 Score | 0.9494 |
| AUC | 0.9838 |

## Final Output

The project produces:

- RUL predictions
- Failure-risk predictions
- Risk category labels
- Maintenance recommendations
- Engine-level dashboard outputs
- AI maintenance decision report
