# Databricks notebook source
from pyspark.sql import Row
from pyspark.sql.functions import col, avg, max, min

# COMMAND ----------

data = [
    Row(device_id="D1", temperature=72.5, vibration=0.32, status="normal"),
    Row(device_id="D2", temperature=88.1, vibration=0.75, status="warning"),
    Row(device_id="D3", temperature=95.4, vibration=1.20, status="failure"),
    Row(device_id="D1", temperature=74.0, vibration=0.35, status="normal")
]

# COMMAND ----------

df = spark.createDataFrame(data)
df.show()
df.printSchema()

# COMMAND ----------

high_temp_df = df.filter(col("temperature") > 80)
high_temp_df.show()

# COMMAND ----------

df.groupBy("device_id").agg(
    avg("temperature").alias("avg_temperature"),
    max("vibration").alias("max_vibration"),
    min("vibration").alias("min_vibration")
).show()

# COMMAND ----------

# Day-2

from pyspark.sql import Row
from pyspark.sql.functions import col, when, avg, max, min, count

# COMMAND ----------

data = [
    Row(device_id="M1", temperature=72.5, vibration=0.32, pressure=30.1, machine_status="normal"),
    Row(device_id="M2", temperature=88.1, vibration=0.75, pressure=34.8, machine_status="warning"),
    Row(device_id="M3", temperature=95.4, vibration=1.20, pressure=39.2, machine_status="failure"),
    Row(device_id="M1", temperature=74.0, vibration=0.35, pressure=31.0, machine_status="normal"),
    Row(device_id="M2", temperature=84.6, vibration=0.68, pressure=33.9, machine_status="warning")
]

df = spark.createDataFrame(data)

df.show()
df.printSchema()

# COMMAND ----------

# Failure risk Logic

risk_df = df.withColumn(
    "failure_risk",
    when((col("temperature") > 90) | (col("vibration") >= 1.0), "high")
    .when((col("temperature") > 80) | (col("vibration") >= 0.7), "medium")
    .otherwise("low")
)

risk_df.show()

# COMMAND ----------

# alert flag

alert_df = risk_df.withColumn(
    "alert_flag",
    when((col("failure_risk") == "high") | (col("machine_status") == "failure"), 1)
    .otherwise(0)
)

alert_df.show()

# COMMAND ----------

# Filter high-risk machines
high_risk_df = alert_df.filter(col("failure_risk") == "high")
high_risk_df.show()

# COMMAND ----------

# Create device-level summary
device_summary_df = alert_df.groupBy("device_id").agg(
    avg("temperature").alias("avg_temperature"),
    max("vibration").alias("max_vibration"),
    min("pressure").alias("min_pressure"),
    count("*").alias("record_count")
)

device_summary_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Day 2 PySpark Learning:
# MAGIC
# MAGIC Today I practiced PySpark DataFrame operations using predictive sensor data. I created sample machine records with temperature, vibration, pressure, and machine status fields.
# MAGIC
# MAGIC I used withColumn and when conditions to create failure risk levels and alert flags. I also practiced filtering high-risk machines and creating device-level summaries using groupBy and aggregation functions.
# MAGIC
# MAGIC These operations are important for the IoT Predictive Maintenance Pipeline because real sensor data needs to be cleaned, classified, summarized, and prepared before machine learning model training.

# COMMAND ----------

