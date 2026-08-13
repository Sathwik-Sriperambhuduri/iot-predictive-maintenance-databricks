# Databricks notebook source
# Notebook 14: Model Serving and Inference Testing
# Load registered models and prepare for RUL + failure-risk prediction

import mlflow
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.functions import vector_to_array
from datetime import datetime

# Unity Catalog model names
RUL_MODEL_NAME = "workspace.default.iot_rul_random_forest_model"
FAILURE_MODEL_NAME = "workspace.default.iot_failure_risk_logistic_regression_model"

# Use Champion alias if available
RUL_MODEL_URI = f"models:/{RUL_MODEL_NAME}@Champion"
FAILURE_MODEL_URI = f"models:/{FAILURE_MODEL_NAME}@Champion"

print("Notebook 14 setup completed successfully.")
print("RUL Model URI:", RUL_MODEL_URI)
print("Failure Risk Model URI:", FAILURE_MODEL_URI)
print("Current Time:", datetime.now())

# COMMAND ----------

# Cell 2: Create UC volume path and load registered models

import os
import mlflow

# Create a Unity Catalog volume for MLflow temporary Spark model files
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.mlflow_tmp")

# UC volume path
MLFLOW_TMP_PATH = "/Volumes/workspace/default/mlflow_tmp"

# Set MLflow temporary path
os.environ["MLFLOW_DFS_TMP"] = MLFLOW_TMP_PATH

print("MLflow temp path set to:", MLFLOW_TMP_PATH)

# Load registered models using the UC volume temp path
rul_model = mlflow.spark.load_model(
    RUL_MODEL_URI,
    dfs_tmpdir=MLFLOW_TMP_PATH
)

failure_model = mlflow.spark.load_model(
    FAILURE_MODEL_URI,
    dfs_tmpdir=MLFLOW_TMP_PATH
)

print("Both registered models loaded successfully.")
print("RUL model type:", type(rul_model))
print("Failure-risk model type:", type(failure_model))

# COMMAND ----------

# Inspect model pipeline stages and required feature columns

def inspect_pipeline_model(model, model_name):
    print("=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    feature_columns = None

    for i, stage in enumerate(model.stages):
        stage_type = stage.__class__.__name__
        print(f"\nStage {i + 1}: {stage_type}")

        # Print input column if available
        try:
            print("Input Col:", stage.getInputCol())
        except:
            pass

        # Print multiple input columns if available
        try:
            input_cols = stage.getInputCols()
            print("Input Cols:", input_cols)

            # Usually VectorAssembler contains actual feature input columns
            if stage_type == "VectorAssembler":
                feature_columns = input_cols
        except:
            pass

        # Print output column if available
        try:
            print("Output Col:", stage.getOutputCol())
        except:
            pass

        # Print features column if available
        try:
            print("Features Col:", stage.getFeaturesCol())
        except:
            pass

        # Print label column if available
        try:
            print("Label Col:", stage.getLabelCol())
        except:
            pass

    print("\nFinal required feature columns:")
    print(feature_columns)

    return feature_columns


rul_feature_columns = inspect_pipeline_model(rul_model, "RUL Regression Model")
failure_feature_columns = inspect_pipeline_model(failure_model, "Failure Risk Classification Model")

# COMMAND ----------

# Check available tables in workspace.default

tables_df = spark.sql("SHOW TABLES IN workspace.default")

display(tables_df)

print("Available tables in workspace.default:")
for row in tables_df.collect():
    print(row.tableName)

# COMMAND ----------

# Load model-ready test data for inference

inference_input_df = spark.table("workspace.default.model_ready_sensor_test")

print("Model-ready test data loaded successfully.")
print("Total records:", inference_input_df.count())

print("Schema:")
inference_input_df.printSchema()

display(inference_input_df.limit(5))

# COMMAND ----------

# Cell 6: Recreate features column for inference

from pyspark.ml.feature import VectorAssembler

# Same feature columns used during model training
feature_columns = [
    "cycle_age",
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15",
    "sensor_2_rolling_avg_5", "sensor_3_rolling_avg_5", "sensor_4_rolling_avg_5",
    "sensor_7_rolling_avg_5", "sensor_11_rolling_avg_5", "sensor_12_rolling_avg_5",
    "sensor_15_rolling_avg_5",
    "sensor_2_diff", "sensor_3_diff", "sensor_4_diff", "sensor_7_diff",
    "sensor_11_diff", "sensor_12_diff", "sensor_15_diff"
]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features",
    handleInvalid="keep"
)

inference_features_df = assembler.transform(inference_input_df)

print("Features column created successfully.")
print("Number of feature columns:", len(feature_columns))

# Check expected feature count from models
rul_expected_features = rul_model.stages[-1].numFeatures
failure_expected_features = failure_model.stages[-1].numFeatures

print("RUL model expected features:", rul_expected_features)
print("Failure-risk model expected features:", failure_expected_features)

display(
    inference_features_df
    .select("unit_number", "time_in_cycles", "features")
    .limit(5)
)

# COMMAND ----------

# Generate Remaining Useful Life predictions

rul_predictions_df = rul_model.transform(inference_features_df)

# Rename prediction column for clarity
rul_predictions_df = rul_predictions_df.withColumnRenamed(
    "prediction",
    "predicted_rul"
)

print("RUL predictions generated successfully.")

display(
    rul_predictions_df
    .select(
        "unit_number",
        "time_in_cycles",
        "cycle_age",
        "predicted_rul"
    )
    .orderBy("unit_number", "time_in_cycles")
    .limit(10)
)

# COMMAND ----------

# Cell 8: Generate failure-risk predictions

failure_predictions_df = failure_model.transform(inference_features_df)

print("Failure-risk predictions generated successfully.")

display(
    failure_predictions_df
    .select(
        "unit_number",
        "time_in_cycles",
        "cycle_age",
        "prediction",
        "probability"
    )
    .withColumnRenamed("prediction", "failure_risk_prediction")
    .orderBy("unit_number", "time_in_cycles")
    .limit(10)
)

# COMMAND ----------

#Combine RUL prediction, failure-risk prediction, and maintenance recommendation

from pyspark.ml.functions import vector_to_array

# Step 1: Run RUL model
rul_output_df = rul_model.transform(inference_features_df) \
    .withColumnRenamed("prediction", "predicted_rul")

# Step 2: Run failure-risk model on same inference data
failure_output_df = failure_model.transform(rul_output_df) \
    .withColumnRenamed("prediction", "failure_risk_prediction")

# Step 3: Extract high-risk probability from probability vector
final_inference_df = failure_output_df.withColumn(
    "failure_risk_probability",
    vector_to_array(F.col("probability"))[1]
)

# Step 4: Add readable risk category
final_inference_df = final_inference_df.withColumn(
    "risk_category",
    F.when((F.col("failure_risk_prediction") == 1) | (F.col("predicted_rul") <= 30), "High Risk")
     .when((F.col("predicted_rul") > 30) & (F.col("predicted_rul") <= 60), "Medium Risk")
     .otherwise("Low Risk")
)

# Step 5: Add maintenance recommendation
final_inference_df = final_inference_df.withColumn(
    "maintenance_recommendation",
    F.when(
        F.col("risk_category") == "High Risk",
        "Immediate maintenance inspection recommended"
    ).when(
        F.col("risk_category") == "Medium Risk",
        "Schedule maintenance soon and monitor sensor trends"
    ).otherwise(
        "Continue normal operation and routine monitoring"
    )
)

# Step 6: Select final serving-style output
final_serving_output_df = final_inference_df.select(
    "unit_number",
    "time_in_cycles",
    "cycle_age",
    F.round("predicted_rul", 2).alias("predicted_rul"),
    F.round("failure_risk_probability", 4).alias("failure_risk_probability"),
    "failure_risk_prediction",
    "risk_category",
    "maintenance_recommendation"
)

print("Final inference output created successfully.")

display(
    final_serving_output_df
    .orderBy("unit_number", "time_in_cycles")
    .limit(20)
)

# COMMAND ----------

# Create latest engine-level maintenance view

from pyspark.sql.window import Window

# Pick the latest available cycle for each engine
latest_cycle_window = Window.partitionBy("unit_number").orderBy(F.desc("time_in_cycles"))

latest_engine_status_df = final_serving_output_df \
    .withColumn("row_num", F.row_number().over(latest_cycle_window)) \
    .filter(F.col("row_num") == 1) \
    .drop("row_num")

print("Latest engine-level inference view created successfully.")
print("Total engines:", latest_engine_status_df.count())

print("Risk category distribution:")
display(
    latest_engine_status_df
    .groupBy("risk_category")
    .count()
    .orderBy("risk_category")
)

print("Engines with lowest predicted RUL:")
display(
    latest_engine_status_df
    .orderBy("predicted_rul")
    .limit(20)
)

# COMMAND ----------

# Save final inference output as Delta table

final_serving_output_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.default.iot_model_serving_inference_output")

latest_engine_status_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.default.iot_latest_engine_status_dashboard")

print("Inference output tables saved successfully.")
print("Saved table 1: workspace.default.iot_model_serving_inference_output")
print("Saved table 2: workspace.default.iot_latest_engine_status_dashboard")

# COMMAND ----------

# Verify saved inference tables

print("Full inference output table:")
display(spark.table("workspace.default.iot_model_serving_inference_output").limit(10))

print("Latest engine dashboard table:")
display(spark.table("workspace.default.iot_latest_engine_status_dashboard").orderBy("predicted_rul").limit(10))

# COMMAND ----------

display(
    spark.table("workspace.default.iot_latest_engine_status_dashboard")
    .orderBy("predicted_rul")
    .limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook 14 Summary
# MAGIC
# MAGIC In this notebook, I completed the deployment-style inference workflow for the IoT Predictive Maintenance project.
# MAGIC
# MAGIC The registered Champion models from Unity Catalog were loaded successfully and used to generate predictions on model-ready sensor data. The workflow produced Remaining Useful Life predictions, failure-risk predictions, risk categories, and maintenance recommendations.
# MAGIC
# MAGIC Two Delta tables were created for downstream use:
# MAGIC
# MAGIC - `workspace.default.iot_model_serving_inference_output`
# MAGIC - `workspace.default.iot_latest_engine_status_dashboard`
# MAGIC
# MAGIC The latest engine-level dashboard table will be used in the next notebook to build the AI application/dashboard.