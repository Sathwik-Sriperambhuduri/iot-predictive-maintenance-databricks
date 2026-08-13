# Databricks notebook source
# MAGIC %md
# MAGIC Objective:
# MAGIC
# MAGIC The objective of this notebook is to train baseline classification models for failure-risk prediction in the IoT Predictive Maintenance Pipeline. This notebook uses the model-ready classification dataset, trains classification models, evaluates performance, and tracks the results in MLflow.

# COMMAND ----------

print("hello")

# COMMAND ----------

classification_df = spark.table("model_ready_sensor_train_classification")

print("Rows:", classification_df.count())
print("Columns:", len(classification_df.columns))

classification_df.show(5)
classification_df.printSchema()

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

label_col = "failure_risk_binary"

print("Total feature columns:", len(feature_cols))
print(feature_cols)

# COMMAND ----------

# Label Distribution
from pyspark.sql.functions import col, count

classification_df.groupBy(label_col).agg(
    count("*").alias("record_count")
).show()

# COMMAND ----------

# prepare feature vector
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import sum as spark_sum, when

missing_feature_df = classification_df.select([
    spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in feature_cols + [label_col]
])

missing_feature_df.show()

classification_df = classification_df.fillna(0, subset=feature_cols)

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

model_input_df = assembler.transform(classification_df).select(
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

# Set MLflow experiment and volume path
import mlflow
import mlflow.spark
import os

mlflow.set_experiment("/Shared/iot_predictive_maintenance_week3")

mlflow_tmp_path = "/Volumes/workspace/default/iot_predictive_maintenance_volume/mlflow_tmp"
os.environ["MLFLOW_DFS_TMP"] = mlflow_tmp_path

if mlflow.active_run() is not None:
    mlflow.end_run()

# COMMAND ----------

# Define classification evaluators
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="accuracy"
)

precision_evaluator = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="weightedRecall"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="f1"
)

auc_evaluator = BinaryClassificationEvaluator(
    labelCol=label_col,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

# COMMAND ----------

# Train Logistic Regression classifier
from pyspark.ml.classification import LogisticRegression

lr_classifier = LogisticRegression(
    featuresCol="features",
    labelCol=label_col,
    predictionCol="prediction",
    maxIter=20
)

with mlflow.start_run(run_name="baseline_logistic_regression_failure_classification"):
    lr_class_model = lr_classifier.fit(train_data)
    lr_class_predictions = lr_class_model.transform(validation_data)

    lr_accuracy = accuracy_evaluator.evaluate(lr_class_predictions)
    lr_precision = precision_evaluator.evaluate(lr_class_predictions)
    lr_recall = recall_evaluator.evaluate(lr_class_predictions)
    lr_f1 = f1_evaluator.evaluate(lr_class_predictions)
    lr_auc = auc_evaluator.evaluate(lr_class_predictions)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("feature_count", len(feature_cols))
    mlflow.log_param("label_column", label_col)

    mlflow.log_metric("accuracy", lr_accuracy)
    mlflow.log_metric("precision", lr_precision)
    mlflow.log_metric("recall", lr_recall)
    mlflow.log_metric("f1", lr_f1)
    mlflow.log_metric("auc", lr_auc)

    mlflow.spark.log_model(
        lr_class_model,
        "model",
        dfs_tmpdir=mlflow_tmp_path
    )

print("Logistic Regression Accuracy:", lr_accuracy)
print("Logistic Regression Precision:", lr_precision)
print("Logistic Regression Recall:", lr_recall)
print("Logistic Regression F1:", lr_f1)
print("Logistic Regression AUC:", lr_auc)

# COMMAND ----------

# Train Random Forest classifier
from pyspark.ml.classification import RandomForestClassifier

rf_classifier = RandomForestClassifier(
    featuresCol="features",
    labelCol=label_col,
    predictionCol="prediction",
    numTrees=75,
    maxDepth=7,
    seed=42
)

if mlflow.active_run() is not None:
    mlflow.end_run()

with mlflow.start_run(run_name="baseline_random_forest_failure_classification"):
    rf_class_model = rf_classifier.fit(train_data)
    rf_class_predictions = rf_class_model.transform(validation_data)

    rf_accuracy = accuracy_evaluator.evaluate(rf_class_predictions)
    rf_precision = precision_evaluator.evaluate(rf_class_predictions)
    rf_recall = recall_evaluator.evaluate(rf_class_predictions)
    rf_f1 = f1_evaluator.evaluate(rf_class_predictions)
    rf_auc = auc_evaluator.evaluate(rf_class_predictions)

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("feature_count", len(feature_cols))
    mlflow.log_param("label_column", label_col)
    mlflow.log_param("numTrees", 75)
    mlflow.log_param("maxDepth", 7)

    mlflow.log_metric("accuracy", rf_accuracy)
    mlflow.log_metric("precision", rf_precision)
    mlflow.log_metric("recall", rf_recall)
    mlflow.log_metric("f1", rf_f1)
    mlflow.log_metric("auc", rf_auc)

    mlflow.spark.log_model(
        rf_class_model,
        "model",
        dfs_tmpdir=mlflow_tmp_path
    )

print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Precision:", rf_precision)
print("Random Forest Recall:", rf_recall)
print("Random Forest F1:", rf_f1)
print("Random Forest AUC:", rf_auc)

# COMMAND ----------

classification_comparison_data = [
    ("Logistic Regression", float(lr_accuracy), float(lr_precision), float(lr_recall), float(lr_f1), float(lr_auc)),
    ("Random Forest Classification", float(rf_accuracy), float(rf_precision), float(rf_recall), float(rf_f1), float(rf_auc))
]

classification_comparison_df = spark.createDataFrame(
    classification_comparison_data,
    ["model_name", "accuracy", "precision", "recall", "f1", "auc"]
)

classification_comparison_df.orderBy(col("f1").desc()).show()

classification_comparison_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    "model_comparison_failure_risk_classification"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Both classification models performed almost equally. Logistic Regression was selected as the best classification model because it produced a slightly higher F1 score and precision while maintaining the same accuracy and recall as Random Forest. Random Forest had a slightly higher AUC, but the difference was minimal.

# COMMAND ----------

# Save best classification predictions
if rf_f1 >= lr_f1:
    best_classification_predictions = rf_class_predictions
    best_classification_model_name = "Random Forest Classification"
else:
    best_classification_predictions = lr_class_predictions
    best_classification_model_name = "Logistic Regression"

print("Best Classification Model:", best_classification_model_name)

best_classification_predictions_output_df = best_classification_predictions.select(
    "unit_number",
    "time_in_cycles",
    label_col,
    "prediction",
    "probability"
)

best_classification_predictions_output_df.show(20, truncate=False)

best_classification_predictions_output_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    "best_failure_risk_prediction_results"
)