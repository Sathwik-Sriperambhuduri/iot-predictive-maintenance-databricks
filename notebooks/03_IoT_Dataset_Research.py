# Databricks notebook source
# MAGIC %md
# MAGIC IoT Dataset Research
# MAGIC
# MAGIC Final Project Direction: IoT Predictive Maintenance Pipeline on Databricks
# MAGIC
# MAGIC Project Goal:
# MAGIC Build a production-oriented IoT predictive maintenance pipeline using Databricks, PySpark, Delta Lake, and MLflow. The project will ingest sensor or machine-monitoring data, clean and transform it, create predictive maintenance features, train machine learning models, track experiments, and prepare a final prediction workflow.
# MAGIC
# MAGIC Final Use Case:
# MAGIC Predict equipment failure risk, anomaly behavior, or remaining useful life based on sensor readings and device-level operating conditions.
# MAGIC
# MAGIC Dataset Selection Criteria:
# MAGIC
# MAGIC 1. Must support predictive maintenance or machine monitoring.
# MAGIC 2. Should contain sensor readings such as temperature, vibration, pressure, speed, torque, or similar machine signals.
# MAGIC 3. Should have device ID, machine ID, engine ID, or equipment identifier.
# MAGIC 4. Should have time, cycle, timestamp, or sequence information.
# MAGIC 5. Should support failure prediction, anomaly detection, or remaining useful life prediction.
# MAGIC 6. Should be suitable for PySpark processing in Databricks.
# MAGIC
# MAGIC | Dataset                  | Use Case                         | Strength                                         | Weakness                      | Decision     |
# MAGIC | ------------------------ | -------------------------------- | ------------------------------------------------ | ----------------------------- | ------------ |
# MAGIC | NASA C-MAPSS             | Remaining Useful Life prediction | Strong time-series sensor data                   | More complex preprocessing    | First choice |
# MAGIC | UCI AI4I 2020            | Failure classification           | Simple and direct predictive maintenance dataset | Synthetic and smaller dataset | Backup       |
# MAGIC | Other IoT Sensor Dataset | Anomaly detection                | Flexible IoT use case                            | May not have failure labels   | Optional     |
# MAGIC
# MAGIC
# MAGIC Current Dataset Decision:
# MAGIC
# MAGIC First Choice:
# MAGIC NASA C-MAPSS Turbofan Engine Dataset
# MAGIC
# MAGIC Reason:
# MAGIC This dataset fits the predictive maintenance project because it contains machine degradation data, time-series style sensor records, engine-level monitoring, and supports remaining useful life prediction.
# MAGIC
# MAGIC Backup Dataset:
# MAGIC UCI AI4I 2020 Predictive Maintenance Dataset
# MAGIC
# MAGIC Reason:
# MAGIC This dataset is easier to use and directly supports failure classification, but it is synthetic and smaller than NASA C-MAPSS.

# COMMAND ----------

