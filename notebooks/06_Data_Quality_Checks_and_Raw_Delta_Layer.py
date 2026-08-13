# Databricks notebook source
# MAGIC %md
# MAGIC Week-2 
# MAGIC Day-1
# MAGIC Today’s Objective:
# MAGIC
# MAGIC The objective of this notebook is to continue the IoT Predictive Maintenance Pipeline by performing deeper data-quality checks on the NASA C-MAPSS FD001 dataset and preparing the raw Delta layer in Databricks.
# MAGIC
# MAGIC This notebook will validate row counts, schema consistency, missing values, duplicate records, sensor ranges, low-variance sensors, and create reusable raw-layer tables for future cleaning and feature engineering.

# COMMAND ----------

from pyspark.sql.functions import col, split, trim, max, min, avg, count, when, row_number, monotonically_increasing_id
from pyspark.sql.window import Window

# COMMAND ----------

dataset_name = "nasa_cmapss_fd001"

entity_id_col = "unit_number"
time_col = "time_in_cycles"

operational_setting_cols = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3"
]

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

all_columns = [entity_id_col, time_col] + operational_setting_cols + sensor_cols

print("Dataset:", dataset_name)
print("Total columns:", len(all_columns))

# COMMAND ----------

train_path = "/Volumes/workspace/default/iot_predictive_maintenance_volume/train_FD001.txt"
test_path = "/Volumes/workspace/default/iot_predictive_maintenance_volume/test_FD001.txt"
rul_path = "/Volumes/workspace/default/iot_predictive_maintenance_volume/RUL_FD001.txt"

# COMMAND ----------

def load_cmapss_sensor_file(file_path, column_names):
    raw_text_df = spark.read.text(file_path)

    parsed_df = raw_text_df.select(
        split(trim(col("value")), "\\s+").alias("cols")
    )

    for index, column_name in enumerate(column_names):
        parsed_df = parsed_df.withColumn(
            column_name,
            col("cols")[index].cast("double")
        )

    final_df = parsed_df.drop("cols")

    final_df = final_df.withColumn("unit_number", col("unit_number").cast("int")) \
                       .withColumn("time_in_cycles", col("time_in_cycles").cast("int"))

    return final_df

# COMMAND ----------

train_df = load_cmapss_sensor_file(train_path, all_columns)
test_df = load_cmapss_sensor_file(test_path, all_columns)

print("Train rows:", train_df.count())
print("Test rows:", test_df.count())

train_df.show(5)
test_df.show(5)

# COMMAND ----------

max_cycle_df = train_df.groupBy("unit_number").agg(
    max("time_in_cycles").alias("max_cycle")
)

train_with_rul_df = train_df.join(
    max_cycle_df,
    on="unit_number",
    how="left"
).withColumn(
    "rul",
    col("max_cycle") - col("time_in_cycles")
)

train_labeled_df = train_with_rul_df.withColumn(
    "failure_risk",
    when(col("rul") <= 30, "high")
    .when((col("rul") > 30) & (col("rul") <= 80), "medium")
    .otherwise("low")
)

train_labeled_df.select("unit_number", "time_in_cycles", "max_cycle", "rul", "failure_risk").show(10)

# COMMAND ----------

rul_raw_df = spark.read.text(rul_path)

rul_clean_df = rul_raw_df.select(
    trim(col("value")).cast("int").alias("true_rul_at_last_cycle")
)

window_spec = Window.orderBy(monotonically_increasing_id())

rul_df = rul_clean_df.withColumn(
    "unit_number",
    row_number().over(window_spec)
).select(
    "unit_number",
    "true_rul_at_last_cycle"
)

rul_df.show(10)

# COMMAND ----------

test_last_cycle_df = test_df.groupBy("unit_number").agg(
    max("time_in_cycles").alias("last_observed_cycle")
)

test_summary_df = test_last_cycle_df.join(
    rul_df,
    on="unit_number",
    how="left"
)

test_summary_df = test_summary_df.withColumn(
    "estimated_failure_cycle",
    col("last_observed_cycle") + col("true_rul_at_last_cycle")
)

test_summary_df.orderBy("unit_number").show(10)

# COMMAND ----------

print("Train rows:", train_df.count())
print("Test rows:", test_df.count())
print("Training labeled rows:", train_labeled_df.count())
print("Test summary rows:", test_summary_df.count())

print("Train columns:", len(train_df.columns))
print("Test columns:", len(test_df.columns))
print("Training labeled columns:", len(train_labeled_df.columns))

# COMMAND ----------

train_columns = set(train_df.columns)
test_columns = set(test_df.columns)

missing_in_test = train_columns - test_columns
missing_in_train = test_columns - train_columns

print("Columns missing in test:", missing_in_test)
print("Columns missing in train:", missing_in_train)

# COMMAND ----------

from pyspark.sql.functions import col, sum as spark_sum, when

missing_train_df = train_labeled_df.select([
    spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in train_labeled_df.columns
])

missing_train_df.show()

# COMMAND ----------

missing_test_df = test_df.select([
    spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in test_df.columns
])

missing_test_df.show()

# COMMAND ----------

train_total = train_labeled_df.count()
train_distinct = train_labeled_df.distinct().count()
train_duplicates = train_total - train_distinct

test_total = test_df.count()
test_distinct = test_df.distinct().count()
test_duplicates = test_total - test_distinct

print("Train duplicate rows:", train_duplicates)
print("Test duplicate rows:", test_duplicates)

# COMMAND ----------

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
train_labeled_df.select(sensor_cols).describe().show()

# COMMAND ----------

from pyspark.sql.functions import stddev
sensor_variance_df = train_labeled_df.select([
    stddev(col(c)).alias(c)
    for c in sensor_cols
])
sensor_variance_df.show()

# COMMAND ----------

train_labeled_df.write.format("delta").mode("overwrite").saveAsTable("raw_sensor_training_data")
test_df.write.format("delta").mode("overwrite").saveAsTable("raw_sensor_test_data")
test_summary_df.write.format("delta").mode("overwrite").saveAsTable("raw_sensor_test_rul_summary")

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM raw_sensor_training_data;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM raw_sensor_test_data;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM raw_sensor_test_rul_summary;