# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC Day 1 - Spark Architecture Notes
# MAGIC
# MAGIC Project: Production-Oriented IoT Data Pipeline on Databricks
# MAGIC
# MAGIC Spark Driver:
# MAGIC The main program that controls the Spark application. It creates the SparkSession and sends tasks to executors.
# MAGIC
# MAGIC Executor:
# MAGIC A process that runs tasks and stores data for the Spark application.
# MAGIC
# MAGIC Worker Node:
# MAGIC A machine where executors run.
# MAGIC
# MAGIC Cluster Manager:
# MAGIC The system that allocates resources to Spark applications.
# MAGIC
# MAGIC Transformation:
# MAGIC An operation such as select, filter, withColumn, or groupBy. Transformations are lazy and do not execute immediately.
# MAGIC
# MAGIC Action:
# MAGIC An operation such as show, count, collect, or write. Actions trigger Spark execution.
# MAGIC
# MAGIC Lazy Evaluation:
# MAGIC Spark waits until an action is called before executing transformations. This helps Spark optimize the execution plan.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC How Spark Architecture Applies to My IoT Predictive Maintenance Project:
# MAGIC
# MAGIC In this project, the Spark Driver will control the main application logic and coordinate the execution of the IoT data pipeline.
# MAGIC
# MAGIC Executors will run tasks such as reading sensor data, filtering records, creating failure-risk labels, and aggregating machine-level summaries.
# MAGIC
# MAGIC Transformations such as filter, withColumn, and groupBy will remain lazy until an action is called.
# MAGIC
# MAGIC Actions such as show, count, or write will trigger Spark execution.
# MAGIC
# MAGIC This matters for predictive maintenance because large sensor datasets can be processed in distributed steps instead of handling all records manually in memory.

# COMMAND ----------

