# Databricks notebook source
# MAGIC %md
# MAGIC Day 3 Objective:
# MAGIC
# MAGIC The objective of this notebook is to load the NASA C-MAPSS raw sensor data into Databricks, inspect the schema, understand the column structure, and prepare the first raw-data layer for the IoT Predictive Maintenance Pipeline.
# MAGIC
# MAGIC The pipeline will use generic sensor-data naming conventions so the dataset can be changed later if needed.

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

# COMMAND ----------

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
all_columns = [entity_id_col, time_col] + operational_setting_cols + sensor_cols
print("Dataset:", dataset_name)
print("Total expected columns:", len(all_columns))
print(all_columns)

# COMMAND ----------

train_path = "/Volumes/workspace/default/iot_predictive_maintenance_volume/train_FD001.txt"
test_path = "/Volumes/workspace/default/iot_predictive_maintenance_volume/test_FD001.txt"
rul_path = "/Volumes/workspace/default/iot_predictive_maintenance_volume/RUL_FD001.txt"

# COMMAND ----------

raw_train_text_df = spark.read.text(train_path)
raw_train_text_df.show(5, truncate=False)

# COMMAND ----------

from pyspark.sql.functions import col, split, trim

# COMMAND ----------

parsed_train_df = raw_train_text_df.select(
    split(trim(col("value")), "\\s+").alias("cols")
)

for index, column_name in enumerate(all_columns):
    parsed_train_df = parsed_train_df.withColumn(
        column_name,
        col("cols")[index].cast("double")
    )

# COMMAND ----------

train_df = parsed_train_df.drop("cols")
train_df.show(5)
train_df.printSchema()

# COMMAND ----------

train_df = train_df.withColumn("unit_number", col("unit_number").cast("int")) \
                   .withColumn("time_in_cycles", col("time_in_cycles").cast("int"))

train_df.printSchema()

# COMMAND ----------

row_count = train_df.count()
column_count = len(train_df.columns)
engine_count = train_df.select("unit_number").distinct().count()

# COMMAND ----------

print("Row count:", row_count)
print("Column count:", column_count)
print("Number of engines:", engine_count)

# COMMAND ----------

train_df.groupBy("unit_number").count().orderBy("unit_number").show(10)

# COMMAND ----------

# Check cycle range by engine
from pyspark.sql.functions import min, max

train_df.groupBy("unit_number").agg(
    min("time_in_cycles").alias("min_cycle"),
    max("time_in_cycles").alias("max_cycle")
).orderBy("unit_number").show(10)

# COMMAND ----------

# check missing values
from pyspark.sql.functions import sum as spark_sum, when

missing_summary_df = train_df.select([
    spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in train_df.columns
])

missing_summary_df.show()

# COMMAND ----------

# Check duplicate records
total_rows = train_df.count()
distinct_rows = train_df.distinct().count()
duplicate_rows = total_rows - distinct_rows

# COMMAND ----------

print("Total rows:", total_rows)
print("Distinct rows:", distinct_rows)
print("Duplicate rows:", duplicate_rows)

# COMMAND ----------

# Create a temporary SQL view
train_df.createOrReplaceTempView("raw_sensor_data")

spark.sql("""
SELECT *
FROM raw_sensor_data
LIMIT 10
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC Week 1 Day 3 Learning:
# MAGIC
# MAGIC Today I started working with the actual NASA C-MAPSS dataset in Databricks. I uploaded the FD001 raw files, created a flexible dataset configuration, loaded the training data as raw text, parsed the sensor records into structured columns, and performed initial schema and data-quality checks.
# MAGIC
# MAGIC I checked row count, column count, engine count, cycle ranges, missing values, and duplicate records. I also created a temporary view named raw_sensor_data so the project can support SQL-based exploration.
# MAGIC
# MAGIC I used generic naming so the pipeline can later support another predictive maintenance dataset if needed. This keeps the project flexible while continuing with NASA C-MAPSS as the current dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC # Day - 4

# COMMAND ----------

train_df.show(5)
train_df.printSchema()
train_df.count()