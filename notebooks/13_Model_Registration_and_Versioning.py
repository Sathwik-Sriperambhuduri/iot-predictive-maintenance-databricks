# Databricks notebook source
# MAGIC %md
# MAGIC Model Registration and Versioning
# MAGIC
# MAGIC This notebook registers the best RUL regression model and the best failure-risk classification model from the Week 3 MLflow experiments.

# COMMAND ----------

import mlflow
import mlflow.spark
from mlflow import MlflowClient
from mlflow.models import infer_signature
import os

mlflow.set_experiment("/Shared/iot_predictive_maintenance_week3")

# Use Unity Catalog Model Registry
mlflow.set_registry_uri("databricks-uc")

client = MlflowClient()

catalog_name = "workspace"
schema_name = "default"

rul_model_name = (
    f"{catalog_name}.{schema_name}."
    "iot_rul_random_forest_model"
)

classification_model_name = (
    f"{catalog_name}.{schema_name}."
    "iot_failure_risk_logistic_regression_model"
)

mlflow_tmp_path = (
    "/Volumes/workspace/default/"
    "iot_predictive_maintenance_volume/mlflow_tmp"
)

os.environ["MLFLOW_DFS_TMP"] = mlflow_tmp_path

print("RUL model:", rul_model_name)
print("Classification model:", classification_model_name)

# COMMAND ----------

# Find the existing best model runs
def get_latest_run_id(run_name):
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1
    )

    if runs.empty:
        raise ValueError(f"No MLflow run found for: {run_name}")

    return runs.iloc[0]["run_id"]


rul_run_id = get_latest_run_id(
    "best_tuned_random_forest_regression"
)

classification_run_id = get_latest_run_id(
    "baseline_logistic_regression_failure_classification"
)

print("RUL Regression Run ID:", rul_run_id)
print("Classification Run ID:", classification_run_id)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

selected_sensor_cols = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_11",
    "sensor_12",
    "sensor_15"
]

base_feature_cols = ["cycle_age"]

rolling_feature_cols = [
    f"{sensor}_rolling_avg_5"
    for sensor in selected_sensor_cols
]

diff_feature_cols = [
    f"{sensor}_diff"
    for sensor in selected_sensor_cols
]

feature_cols = (
    base_feature_cols
    + selected_sensor_cols
    + rolling_feature_cols
    + diff_feature_cols
)

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

rul_df = spark.table(
    "model_ready_sensor_train_regression"
).fillna(0, subset=feature_cols)

classification_df = spark.table(
    "model_ready_sensor_train_classification"
).fillna(0, subset=feature_cols)

rul_input_df = assembler.transform(rul_df).select("features")

classification_input_df = (
    assembler
    .transform(classification_df)
    .select("features")
)

print("Model input datasets prepared.")

# COMMAND ----------

# Load existing models and create signatures
rul_source_uri = f"runs:/{rul_run_id}/model"

classification_source_uri = (
    f"runs:/{classification_run_id}/model"
)

rul_model = mlflow.spark.load_model(rul_source_uri)

classification_model = mlflow.spark.load_model(
    classification_source_uri
)

rul_sample = rul_input_df.limit(10)

classification_sample = (
    classification_input_df.limit(10)
)

rul_output = (
    rul_model
    .transform(rul_sample)
    .select("prediction")
)

classification_output = (
    classification_model
    .transform(classification_sample)
    .select("prediction")
)

rul_signature = infer_signature(
    rul_sample,
    rul_output
)

classification_signature = infer_signature(
    classification_sample,
    classification_output
)

print("Model signatures created.")
print("RUL signature:", rul_signature)
print(
    "Classification signature:",
    classification_signature
)

# COMMAND ----------

# Re-log and register the RUL model

if mlflow.active_run() is not None:
    mlflow.end_run()

with mlflow.start_run(
    run_name="register_best_rul_model_uc"
):
    mlflow.log_param(
        "source_run_id",
        rul_run_id
    )

    rul_model_info = mlflow.spark.log_model(
        spark_model=rul_model,
        artifact_path="model",
        signature=rul_signature,
        dfs_tmpdir=mlflow_tmp_path
    )

rul_registered_model = mlflow.register_model(
    model_uri=rul_model_info.model_uri,
    name=rul_model_name
)

print(
    "RUL model registered as version:",
    rul_registered_model.version
)

# COMMAND ----------

# Re-log and register the classification model

if mlflow.active_run() is not None:
    mlflow.end_run()

with mlflow.start_run(
    run_name="register_best_classification_model_uc"
):
    mlflow.log_param(
        "source_run_id",
        classification_run_id
    )

    classification_model_info = (
        mlflow.spark.log_model(
            spark_model=classification_model,
            artifact_path="model",
            signature=classification_signature,
            dfs_tmpdir=mlflow_tmp_path
        )
    )

classification_registered_model = mlflow.register_model(
    model_uri=classification_model_info.model_uri,
    name=classification_model_name
)

print(
    "Classification model registered as version:",
    classification_registered_model.version
)

# COMMAND ----------

# Add champion Aliases
client.set_registered_model_alias(
    name=rul_model_name,
    alias="Champion",
    version=str(rul_registered_model.version)
)

client.set_registered_model_alias(
    name=classification_model_name,
    alias="Champion",
    version=str(
        classification_registered_model.version
    )
)

print("Champion aliases added successfully.")

# COMMAND ----------

# Verify the models
rul_champion = client.get_model_version_by_alias(
    rul_model_name,
    "Champion"
)

classification_champion = (
    client.get_model_version_by_alias(
        classification_model_name,
        "Champion"
    )
)

print(
    "RUL model version:",
    rul_champion.version
)

print(
    "Classification model version:",
    classification_champion.version
)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

