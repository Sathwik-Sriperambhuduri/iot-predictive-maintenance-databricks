# Databricks notebook source
# week 2 Day 2
train_raw_df = spark.table("raw_sensor_training_data")
test_raw_df = spark.table("raw_sensor_test_data")
test_rul_summary_df = spark.table("raw_sensor_test_rul_summary")

# COMMAND ----------

print("Training rows:", train_raw_df.count())
print("Test rows:", test_raw_df.count())
print("Test RUL summary rows:", test_rul_summary_df.count())

# COMMAND ----------

train_raw_df.printSchema()
test_raw_df.printSchema()
test_rul_summary_df.printSchema()

# COMMAND ----------

# Remove duplicate rows
train_clean_df = train_raw_df.dropDuplicates()
test_clean_df = test_raw_df.dropDuplicates()
test_rul_clean_df = test_rul_summary_df.dropDuplicates()

print("Train raw rows:", train_raw_df.count())
print("Train clean rows:", train_clean_df.count())

print("Test raw rows:", test_raw_df.count())
print("Test clean rows:", test_clean_df.count())

# COMMAND ----------

#Check missing values
from pyspark.sql.functions import col, sum as spark_sum, when

missing_train_df = train_clean_df.select([
    spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in train_clean_df.columns
])

missing_train_df.show()

# COMMAND ----------

missing_test_df = test_clean_df.select([
    spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in test_clean_df.columns
])

missing_test_df.show()

# COMMAND ----------

# Validate important columns
required_train_columns = ["unit_number", "time_in_cycles", "rul", "failure_risk"]
required_test_columns = ["unit_number", "time_in_cycles"]

for column_name in required_train_columns:
    print(column_name, column_name in train_clean_df.columns)

for column_name in required_test_columns:
    print(column_name, column_name in test_clean_df.columns)

# COMMAND ----------

#Check invalid RUL values
train_clean_df.filter(col("rul") < 0).show()

# COMMAND ----------

from pyspark.sql.functions import min, max, avg

train_clean_df.select(
    min("rul").alias("min_rul"),
    max("rul").alias("max_rul"),
    avg("rul").alias("avg_rul")
).show()

# COMMAND ----------

# Check failure-risk distribution
from pyspark.sql.functions import count

train_clean_df.groupBy("failure_risk").agg(
    count("*").alias("record_count")
).show()

# COMMAND ----------

#Check sensor ranges
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
train_clean_df.select(sensor_cols).describe().show()

# COMMAND ----------

# Identify low-variance sensors
from pyspark.sql.functions import stddev

sensor_stddev_df = train_clean_df.select([
    stddev(col(c)).alias(c)
    for c in sensor_cols
])
sensor_stddev_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Low-variance sensors will be reviewed before feature engineering. Sensors with very little variation may not add much predictive value, but I will not remove them yet until further analysis.

# COMMAND ----------

# Add cleaned-data validation flags
train_clean_df = train_clean_df.withColumn(
    "is_valid_record",
    when(
        (col("unit_number").isNotNull()) &
        (col("time_in_cycles").isNotNull()) &
        (col("rul").isNotNull()) &
        (col("rul") >= 0),
        1
    ).otherwise(0)
)

test_clean_df = test_clean_df.withColumn(
    "is_valid_record",
    when(
        (col("unit_number").isNotNull()) &
        (col("time_in_cycles").isNotNull()),
        1
    ).otherwise(0)
)

# COMMAND ----------

train_clean_df.groupBy("is_valid_record").count().show()
test_clean_df.groupBy("is_valid_record").count().show()

# COMMAND ----------

train_clean_df.write.format("delta").mode("overwrite").saveAsTable("cleaned_sensor_training_data")
test_clean_df.write.format("delta").mode("overwrite").saveAsTable("cleaned_sensor_test_data")
test_rul_clean_df.write.format("delta").mode("overwrite").saveAsTable("cleaned_sensor_test_rul_summary")

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM cleaned_sensor_training_data;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM cleaned_sensor_test_data;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM cleaned_sensor_test_rul_summary;