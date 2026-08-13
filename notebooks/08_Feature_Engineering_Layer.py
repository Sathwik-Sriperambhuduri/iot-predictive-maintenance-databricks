# Databricks notebook source
# MAGIC %md
# MAGIC Objective:
# MAGIC
# MAGIC The objective of this notebook is to create feature-level datasets for the IoT Predictive Maintenance Pipeline using the cleaned Delta tables. This notebook will create cycle-based features, sensor summary features, sensor trend features, and model-ready labels for RUL regression and failure-risk classification.

# COMMAND ----------

train_clean_df = spark.table("cleaned_sensor_training_data")
test_clean_df = spark.table("cleaned_sensor_test_data")
test_rul_summary_df = spark.table("cleaned_sensor_test_rul_summary")

print("Training rows:", train_clean_df.count())
print("Test rows:", test_clean_df.count())
print("Test RUL summary rows:", test_rul_summary_df.count())

# COMMAND ----------

from pyspark.sql.functions import col, avg, max, min, when, lag
from pyspark.sql.window import Window

# COMMAND ----------

#Define sensor columns
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

selected_sensor_cols = [
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_7", "sensor_11", "sensor_12", "sensor_15"
]

print("Total sensor columns:", len(sensor_cols))
print("Selected sensor columns:", selected_sensor_cols)

# COMMAND ----------

#Create cycle-based features
train_features_df = train_clean_df.withColumn(
    "cycle_age",
    col("time_in_cycles")
)

test_features_df = test_clean_df.withColumn(
    "cycle_age",
    col("time_in_cycles")
)

# COMMAND ----------

# Create RUL-based classification label

train_features_df = train_features_df.withColumn(
    "failure_risk_binary",
    when(col("rul") <= 30, 1).otherwise(0)
)

# COMMAND ----------

# Create rolling average features
rolling_window = Window.partitionBy("unit_number").orderBy("time_in_cycles").rowsBetween(-5, 0)

for sensor in selected_sensor_cols:
    train_features_df = train_features_df.withColumn(
        f"{sensor}_rolling_avg_5",
        avg(col(sensor)).over(rolling_window)
    )

    test_features_df = test_features_df.withColumn(
        f"{sensor}_rolling_avg_5",
        avg(col(sensor)).over(rolling_window)
    )

# COMMAND ----------

# Create sensor trend/difference features
lag_window = Window.partitionBy("unit_number").orderBy("time_in_cycles")

for sensor in selected_sensor_cols:
    train_features_df = train_features_df.withColumn(
        f"{sensor}_previous",
        lag(col(sensor), 1).over(lag_window)
    ).withColumn(
        f"{sensor}_diff",
        col(sensor) - col(f"{sensor}_previous")
    )

    test_features_df = test_features_df.withColumn(
        f"{sensor}_previous",
        lag(col(sensor), 1).over(lag_window)
    ).withColumn(
        f"{sensor}_diff",
        col(sensor) - col(f"{sensor}_previous")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC sensor difference features show whether a sensor value is increasing or decreasing from the previous cycle.

# COMMAND ----------

# Fill null values created by lag
diff_cols = [f"{sensor}_diff" for sensor in selected_sensor_cols]
train_features_df = train_features_df.fillna(0, subset=diff_cols)
test_features_df = test_features_df.fillna(0, subset=diff_cols)

# COMMAND ----------

# Drop temporary previous sensor columns
previous_cols = [f"{sensor}_previous" for sensor in selected_sensor_cols]
train_features_df = train_features_df.drop(*previous_cols)
test_features_df = test_features_df.drop(*previous_cols)

# COMMAND ----------

# Check final feature columns
print("Training feature columns:", len(train_features_df.columns))
print("Test feature columns:", len(test_features_df.columns))

train_features_df.select(
    "unit_number", "time_in_cycles", "cycle_age", "rul", "failure_risk", "failure_risk_binary"
).show(10)

# COMMAND ----------

# Save feature-level Delta tables
train_features_df.write.format("delta").mode("overwrite").saveAsTable("feature_sensor_training_data")
test_features_df.write.format("delta").mode("overwrite").saveAsTable("feature_sensor_test_data")

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM feature_sensor_training_data;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM feature_sensor_test_data;