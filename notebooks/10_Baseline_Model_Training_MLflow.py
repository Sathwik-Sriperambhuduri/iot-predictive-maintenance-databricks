# Databricks notebook source
# MAGIC %md
# MAGIC Objective:
# MAGIC
# MAGIC The objective of this notebook is to begin Week 3 model training for the IoT Predictive Maintenance Pipeline. This notebook will load the model-ready regression dataset, create feature vectors, train baseline RUL regression models, evaluate model performance, and track experiments using MLflow.

# COMMAND ----------

model_ready_regression_df = spark.table("model_ready_sensor_train_regression")

print("Rows:", model_ready_regression_df.count())
print("Columns:", len(model_ready_regression_df.columns))

model_ready_regression_df.show(5)
model_ready_regression_df.printSchema()

# COMMAND ----------

# Define feature columns
selected_sensor_cols = [
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_7", "sensor_11", "sensor_12", "sensor_15"
]

base_feature_cols = [
    "cycle_age"
]

rolling_feature_cols = [
    f"{sensor}_rolling_avg_5" for sensor in selected_sensor_cols
]

diff_feature_cols = [
    f"{sensor}_diff" for sensor in selected_sensor_cols
]

feature_cols = base_feature_cols + selected_sensor_cols + rolling_feature_cols + diff_feature_cols

label_col = "rul"

print("Total feature columns:", len(feature_cols))
print(feature_cols)

# COMMAND ----------

from pyspark.sql.functions import col, sum as spark_sum, when

missing_feature_df = model_ready_regression_df.select([
    spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in feature_cols + [label_col]
])

missing_feature_df.show()

# COMMAND ----------

# Create Feature Vector
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

model_input_df = assembler.transform(model_ready_regression_df).select(
    "unit_number",
    "time_in_cycles",
    "features",
    label_col
)

model_input_df.show(5)

# COMMAND ----------

# Split training and validation data
train_data, validation_data = model_input_df.randomSplit([0.8, 0.2], seed=42)
print("Training rows:", train_data.count())
print("Validation rows:", validation_data.count())

# COMMAND ----------

import mlflow
mlflow.set_experiment("/Shared/iot_predictive_maintenance_week3")

# COMMAND ----------

# Train Model 1 — Linear Regression baseline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import mlflow.spark
import os

mlflow_tmp_path = "/Volumes/workspace/default/iot_predictive_maintenance_volume/mlflow_tmp"
os.environ["MLFLOW_DFS_TMP"] = mlflow_tmp_path

if mlflow.active_run() is not None:
    mlflow.end_run()


lr = LinearRegression(
    featuresCol="features",
    labelCol="rul",
    predictionCol="prediction"
)

# Define evaluators
rmse_evaluator = RegressionEvaluator(
    labelCol="rul",
    predictionCol="prediction",
    metricName="rmse"
)

mae_evaluator = RegressionEvaluator(
    labelCol="rul",
    predictionCol="prediction",
    metricName="mae"
)

r2_evaluator = RegressionEvaluator(
    labelCol="rul",
    predictionCol="prediction",
    metricName="r2"
)

# Train, evaluate, and log model
with mlflow.start_run(run_name="baseline_linear_regression"):
    lr_model = lr.fit(train_data)
    lr_predictions = lr_model.transform(validation_data)

    lr_rmse = rmse_evaluator.evaluate(lr_predictions)
    lr_mae = mae_evaluator.evaluate(lr_predictions)
    lr_r2 = r2_evaluator.evaluate(lr_predictions)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("feature_count", len(feature_cols))
    mlflow.log_param("label_column", "rul")
    mlflow.log_param("prediction_column", "prediction")

    mlflow.log_metric("rmse", lr_rmse)
    mlflow.log_metric("mae", lr_mae)
    mlflow.log_metric("r2", lr_r2)

    mlflow.spark.log_model(
        lr_model,
        "model",
        dfs_tmpdir=mlflow_tmp_path
    )

print("Linear Regression RMSE:", lr_rmse)
print("Linear Regression MAE:", lr_mae)
print("Linear Regression R2:", lr_r2)

# COMMAND ----------

# Train Model 2 — Random Forest Regression baseline with MLflow logging

from pyspark.ml.regression import RandomForestRegressor
import mlflow
import mlflow.spark
import os

mlflow_tmp_path = "/Volumes/workspace/default/iot_predictive_maintenance_volume/mlflow_tmp"
os.environ["MLFLOW_DFS_TMP"] = mlflow_tmp_path

if mlflow.active_run() is not None:
    mlflow.end_run()

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="rul",
    predictionCol="prediction",
    numTrees=50,
    maxDepth=5,
    seed=42
)

with mlflow.start_run(run_name="baseline_random_forest_regression"):
    rf_model = rf.fit(train_data)
    rf_predictions = rf_model.transform(validation_data)

    rf_rmse = rmse_evaluator.evaluate(rf_predictions)
    rf_mae = mae_evaluator.evaluate(rf_predictions)
    rf_r2 = r2_evaluator.evaluate(rf_predictions)

    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("feature_count", len(feature_cols))
    mlflow.log_param("label_column", "rul")
    mlflow.log_param("prediction_column", "prediction")
    mlflow.log_param("numTrees", 50)
    mlflow.log_param("maxDepth", 5)

    mlflow.log_metric("rmse", rf_rmse)
    mlflow.log_metric("mae", rf_mae)
    mlflow.log_metric("r2", rf_r2)

    mlflow.spark.log_model(
        rf_model,
        "model",
        dfs_tmpdir=mlflow_tmp_path
    )

print("Random Forest RMSE:", rf_rmse)
print("Random Forest MAE:", rf_mae)
print("Random Forest R2:", rf_r2)

# COMMAND ----------

comparison_data = [
    ("Linear Regression", float(lr_rmse), float(lr_mae), float(lr_r2)),
    ("Random Forest Regression", float(rf_rmse), float(rf_mae), float(rf_r2))
]

comparison_df = spark.createDataFrame(
    comparison_data,
    ["model_name", "rmse", "mae", "r2"]
)

comparison_df.show()

# COMMAND ----------

comparison_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    "model_comparison_rul_regression_baseline"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Model Result Summary:
# MAGIC
# MAGIC Trained two baseline RUL regression models: Linear Regression and Random Forest Regression. Random Forest performed better than Linear Regression because it achieved lower RMSE and MAE and a higher R2 score.
# MAGIC
# MAGIC Linear Regression achieved an RMSE of 40.88, MAE of 31.47, and R2 of 0.6488. Random Forest achieved an RMSE of 37.17, MAE of 26.94, and R2 of 0.7096.
# MAGIC
# MAGIC Based on these baseline results, Random Forest is the stronger model for the current feature set. The next step is to tune the Random Forest model and review whether additional features can improve RUL prediction performance.

# COMMAND ----------

