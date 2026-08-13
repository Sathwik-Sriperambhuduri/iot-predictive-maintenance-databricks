# Databricks notebook source
# Day - 4
from pyspark.sql.functions import col, split, trim, max, min, avg, count, when
from pyspark.sql.window import Window

# COMMAND ----------

dataset_name = "nasa_cmapss_fd001"
entity_id_col = "unit_number"
time_col = "time_in_cycles"

# COMMAND ----------

operational_setting_cols = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3"
]
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
all_columns = [entity_id_col, time_col] + operational_setting_cols + sensor_cols
print("Dataset:", dataset_name)
print("Expected columns:", len(all_columns))

# COMMAND ----------

train_path = "/Volumes/workspace/default/iot_predictive_maintenance_volume/train_FD001.txt"
test_path = "/Volumes/workspace/default/iot_predictive_maintenance_volume/test_FD001.txt"
rul_path = "/Volumes/workspace/default/iot_predictive_maintenance_volume/RUL_FD001.txt"

# COMMAND ----------

# Create a reusable loading function

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

# COMMAND ----------

print("Train rows:", train_df.count())
print("Test rows:", test_df.count())

# COMMAND ----------

train_df.show(5)
test_df.show(5)

# COMMAND ----------

# Compare train and test schema
train_df.printSchema()
test_df.printSchema()

print("Train columns:", len(train_df.columns))
print("Test columns:", len(test_df.columns))

# COMMAND ----------

train_cycle_summary_df = train_df.groupBy("unit_number").agg(
    min("time_in_cycles").alias("start_cycle"),
    max("time_in_cycles").alias("failure_cycle"),
    count("*").alias("total_cycles")
).orderBy("unit_number")
train_cycle_summary_df.show(10)

# COMMAND ----------

# Calculate RUL for training data
max_cycle_df = train_df.groupBy("unit_number").agg(
    max("time_in_cycles").alias("max_cycle")
)

train_with_rul_df = train_df.join(max_cycle_df, on="unit_number", how="left") \
    .withColumn("rul", col("max_cycle") - col("time_in_cycles"))

train_with_rul_df.select(
    "unit_number", "time_in_cycles", "max_cycle", "rul"
).orderBy("unit_number", "time_in_cycles").show(20)

# COMMAND ----------

# Add simple failure-risk category
train_labeled_df = train_with_rul_df.withColumn(
    "failure_risk",
    when(col("rul") <= 30, "high")
    .when((col("rul") > 30) & (col("rul") <= 80), "medium")
    .otherwise("low")
)

train_labeled_df.select(
    "unit_number", "time_in_cycles", "rul", "failure_risk"
).orderBy("unit_number", "time_in_cycles").show(20)

# COMMAND ----------

# Check failure-risk distribution
train_labeled_df.groupBy("failure_risk").agg(
    count("*").alias("record_count")
).show()

# COMMAND ----------

from pyspark.sql.functions import col, trim, row_number, monotonically_increasing_id
from pyspark.sql.window import Window

# Read RUL file as text using DataFrame API
rul_raw_df = spark.read.text(rul_path)

# Clean and convert the RUL value
rul_clean_df = rul_raw_df.select(
    trim(col("value")).cast("int").alias("true_rul_at_last_cycle")
)

# Add unit_number based on row order
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

# teast engine summary
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

train_labeled_df.createOrReplaceTempView("sensor_training_data")
test_df.createOrReplaceTempView("sensor_test_data")
test_summary_df.createOrReplaceTempView("sensor_test_rul_summary")

# COMMAND ----------

spark.sql("""
SELECT unit_number, time_in_cycles, rul, failure_risk
FROM sensor_training_data
ORDER BY unit_number, time_in_cycles
LIMIT 10
""").show()

# COMMAND ----------

selected_sensor_cols = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15"]
train_labeled_df.select(selected_sensor_cols).describe().show()

# COMMAND ----------

train_labeled_df.groupBy("failure_risk").agg(
    avg("sensor_2").alias("avg_sensor_2"),
    avg("sensor_3").alias("avg_sensor_3"),
    avg("sensor_4").alias("avg_sensor_4"),
    avg("sensor_11").alias("avg_sensor_11"),
    avg("sensor_15").alias("avg_sensor_15")
).show()