# Databricks notebook source
# MAGIC %md
# MAGIC Objective:
# MAGIC
# MAGIC The objective of this notebook is to continue Week 3 model training by tuning the Random Forest regression model for Remaining Useful Life prediction. This notebook compares multiple Random Forest parameter settings, identifies the best-performing model, reviews feature importance, and saves tuning outputs for model review.

# COMMAND ----------

model_ready_regression_df = spark.table("model_ready_sensor_train_regression")
print("Rows:", model_ready_regression_df.count())
print("Columns:", len(model_ready_regression_df.columns))
model_ready_regression_df.show(5)
model_ready_regression_df.printSchema()

# COMMAND ----------

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

# Prepare model input
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, sum as spark_sum, when

missing_feature_df = model_ready_regression_df.select([
    spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in feature_cols + [label_col]
])

missing_feature_df.show()

model_ready_regression_df = model_ready_regression_df.fillna(0, subset=feature_cols)

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

train_data, validation_data = model_input_df.randomSplit([0.8, 0.2], seed=42)

print("Training rows:", train_data.count())
print("Validation rows:", validation_data.count())

# COMMAND ----------

import mlflow
import mlflow.spark
import os

mlflow.set_experiment("/Shared/iot_predictive_maintenance_week3")

mlflow_tmp_path = "/Volumes/workspace/default/iot_predictive_maintenance_volume/mlflow_tmp"
os.environ["MLFLOW_DFS_TMP"] = mlflow_tmp_path

if mlflow.active_run() is not None:
    mlflow.end_run()

# COMMAND ----------

# Define evaluators
from pyspark.ml.evaluation import RegressionEvaluator

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

# COMMAND ----------

# Tune Random Forest model
from pyspark.ml.regression import RandomForestRegressor

tuning_configs = [
    {"numTrees": 50, "maxDepth": 5},
    {"numTrees": 75, "maxDepth": 5},
    {"numTrees": 50, "maxDepth": 7},
    {"numTrees": 75, "maxDepth": 7}
]

tuning_results = []

best_rmse = None
best_model = None
best_predictions = None
best_config = None
best_mae = None
best_r2 = None

for config in tuning_configs:
    run_name = f"rf_tuning_trees_{config['numTrees']}_depth_{config['maxDepth']}"

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="rul",
        predictionCol="prediction",
        numTrees=config["numTrees"],
        maxDepth=config["maxDepth"],
        seed=42
    )

    with mlflow.start_run(run_name=run_name):
        rf_model = rf.fit(train_data)
        rf_predictions = rf_model.transform(validation_data)

        rf_rmse = rmse_evaluator.evaluate(rf_predictions)
        rf_mae = mae_evaluator.evaluate(rf_predictions)
        rf_r2 = r2_evaluator.evaluate(rf_predictions)

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("numTrees", config["numTrees"])
        mlflow.log_param("maxDepth", config["maxDepth"])
        mlflow.log_param("feature_count", len(feature_cols))

        mlflow.log_metric("rmse", rf_rmse)
        mlflow.log_metric("mae", rf_mae)
        mlflow.log_metric("r2", rf_r2)

        tuning_results.append((
            "Random Forest",
            config["numTrees"],
            config["maxDepth"],
            float(rf_rmse),
            float(rf_mae),
            float(rf_r2)
        ))

        if best_rmse is None or rf_rmse < best_rmse:
            best_rmse = rf_rmse
            best_mae = rf_mae
            best_r2 = rf_r2
            best_model = rf_model
            best_predictions = rf_predictions
            best_config = config

print("Best Random Forest Config:", best_config)
print("Best RMSE:", best_rmse)
print("Best MAE:", best_mae)
print("Best R2:", best_r2)

# COMMAND ----------

tuning_comparison_df = spark.createDataFrame(
    tuning_results,
    ["model_name", "numTrees", "maxDepth", "rmse", "mae", "r2"]
)

tuning_comparison_df.orderBy("rmse").show()

tuning_comparison_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    "model_comparison_rul_regression_tuning"
)

# COMMAND ----------

# Log the best tuned model
if mlflow.active_run() is not None:
    mlflow.end_run()

with mlflow.start_run(run_name="best_tuned_random_forest_regression"):
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("best_numTrees", best_config["numTrees"])
    mlflow.log_param("best_maxDepth", best_config["maxDepth"])
    mlflow.log_param("feature_count", len(feature_cols))

    mlflow.log_metric("best_rmse", best_rmse)
    mlflow.log_metric("best_mae", best_mae)
    mlflow.log_metric("best_r2", best_r2)

    mlflow.spark.log_model(
        best_model,
        "model",
        dfs_tmpdir=mlflow_tmp_path
    )

print("Best tuned model logged successfully.")

# COMMAND ----------

# Save feature importance table
feature_importance_values = best_model.featureImportances.toArray().tolist()

feature_importance_data = list(zip(feature_cols, feature_importance_values))

feature_importance_df = spark.createDataFrame(
    feature_importance_data,
    ["feature_name", "importance_score"]
)

feature_importance_df.orderBy(col("importance_score").desc()).show(25, truncate=False)

feature_importance_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    "feature_importance_rul_random_forest"
)

# COMMAND ----------

# Save best model predictions

best_predictions_output_df = best_predictions.select(
    "unit_number",
    "time_in_cycles",
    "rul",
    "prediction"
)

best_predictions_output_df.show(20)

best_predictions_output_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    "best_rul_prediction_results"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Day 2 Learning:
# MAGIC
# MAGIC Today I continued Week 3 model training by tuning the Random Forest regression model for Remaining Useful Life prediction. Since Random Forest performed better than Linear Regression during baseline training, I tested multiple Random Forest configurations using different values for numTrees and maxDepth.
# MAGIC
# MAGIC I evaluated each model using RMSE, MAE, and R2, tracked the tuning runs in MLflow, and saved the tuning comparison results as a Delta table. I selected the best Random Forest model based on the lowest RMSE and logged the best tuned model in MLflow.
# MAGIC
# MAGIC I also generated a feature importance table to understand which sensor and engineered features contributed most to the RUL prediction. Finally, I saved the best model prediction results for review and future model deployment steps.

# COMMAND ----------

