# Databricks notebook source
# MAGIC %md
# MAGIC Objective:
# MAGIC
# MAGIC The objective of this notebook is to validate the feature engineering layer and prepare model-ready datasets for the IoT Predictive Maintenance Pipeline. This notebook checks feature completeness, null values, label distribution, selected feature columns, and saves final model-ready Delta tables for Week 3 model training and MLflow tracking.

# COMMAND ----------

train_features_df = spark.table("feature_sensor_training_data")
test_features_df = spark.table("feature_sensor_test_data")

print("Training feature rows:", train_features_df.count())
print("Test feature rows:", test_features_df.count())

print("Training feature columns:", len(train_features_df.columns))
print("Test feature columns:", len(test_features_df.columns))

# COMMAND ----------

train_features_df.printSchema()
test_features_df.printSchema()

# COMMAND ----------

# Check missing values from feature Columns

from pyspark.sql.functions import col, sum as spark_sum, when

missing_train_df = train_features_df.select([
    spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in train_features_df.columns
])

missing_train_df.show()

# COMMAND ----------

missing_test_df = test_features_df.select([
    spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in test_features_df.columns
])

missing_test_df.show()

# COMMAND ----------

# Check failure-risk label distribution
from pyspark.sql.functions import count

train_features_df.groupBy("failure_risk").agg(
    count("*").alias("record_count")
).show()

train_features_df.groupBy("failure_risk_binary").agg(
    count("*").alias("record_count")
).show()

# COMMAND ----------

# Select final model features
base_feature_cols = [
    "cycle_age"
]

selected_sensor_cols = [
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_7", "sensor_11", "sensor_12", "sensor_15"
]

rolling_feature_cols = [
    f"{sensor}_rolling_avg_5" for sensor in selected_sensor_cols
]

diff_feature_cols = [
    f"{sensor}_diff" for sensor in selected_sensor_cols
]

model_feature_cols = base_feature_cols + selected_sensor_cols + rolling_feature_cols + diff_feature_cols

# COMMAND ----------

# Create model-ready training data

# For RUL regression
model_ready_train_regression_df = train_features_df.select(
    ["unit_number", "time_in_cycles"] + model_feature_cols + ["rul"]
)

# for failure-risk classification
model_ready_train_classification_df = train_features_df.select(
    ["unit_number", "time_in_cycles"] + model_feature_cols + ["failure_risk_binary"]
)

# COMMAND ----------

# Create Model- ready test data
model_ready_test_df = test_features_df.select(
    ["unit_number", "time_in_cycles"] + model_feature_cols
)

# COMMAND ----------

# Validate final row counts
print("Regression training rows:", model_ready_train_regression_df.count())
print("Classification training rows:", model_ready_train_classification_df.count())
print("Model-ready test rows:", model_ready_test_df.count())

print("Regression columns:", len(model_ready_train_regression_df.columns))
print("Classification columns:", len(model_ready_train_classification_df.columns))
print("Test columns:", len(model_ready_test_df.columns))

# COMMAND ----------

model_ready_train_regression_df.show(5)
model_ready_train_classification_df.show(5)
model_ready_test_df.show(5)

# COMMAND ----------

# Save

model_ready_train_regression_df.write.format("delta").mode("overwrite").saveAsTable(
    "model_ready_sensor_train_regression"
)

model_ready_train_classification_df.write.format("delta").mode("overwrite").saveAsTable(
    "model_ready_sensor_train_classification"
)

model_ready_test_df.write.format("delta").mode("overwrite").saveAsTable(
    "model_ready_sensor_test"
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM model_ready_sensor_train_regression;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM model_ready_sensor_train_classification;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM model_ready_sensor_test;