# Delta Tables

This project uses Delta tables across multiple pipeline layers.

## Raw Layer

Raw source data loaded into Databricks.

- `workspace.default.raw_sensor_training_data`
- `workspace.default.raw_sensor_test_data`
- `workspace.default.raw_sensor_test_rul_summary`

## Cleaned Layer

Cleaned and validated sensor data.

- `workspace.default.cleaned_sensor_training_data`
- `workspace.default.cleaned_sensor_test_data`
- `workspace.default.cleaned_sensor_test_rul_summary`

## Feature Layer

Feature-engineered datasets.

- `workspace.default.feature_sensor_training_data`
- `workspace.default.feature_sensor_test_data`

## Model-Ready Layer

Final datasets prepared for model training and inference.

- `workspace.default.model_ready_sensor_train_regression`
- `workspace.default.model_ready_sensor_train_classification`
- `workspace.default.model_ready_sensor_test`

## Model Result Tables

Tables created from model evaluation, comparison, and output tracking.

- `workspace.default.best_rul_prediction_results`
- `workspace.default.best_failure_risk_prediction_results`
- `workspace.default.feature_importance_rul_random_forest`
- `workspace.default.model_comparison_rul_regression_baseline`
- `workspace.default.model_comparison_rul_regression_tuning`
- `workspace.default.model_comparison_failure_risk_classification`

## Inference Tables

Tables created during deployment-style inference testing.

- `workspace.default.iot_model_serving_inference_output`
- `workspace.default.iot_latest_engine_status_dashboard`

## AI Dashboard Tables

Tables created for dashboard and decision-support outputs.

- `workspace.default.iot_ai_dashboard_kpi_summary`
- `workspace.default.iot_ai_risk_distribution`
- `workspace.default.iot_ai_maintenance_priority_queue`
- `workspace.default.iot_ai_selected_engine_history`
- `workspace.default.iot_ai_selected_engine_report`
