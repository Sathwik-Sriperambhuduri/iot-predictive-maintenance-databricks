# Databricks notebook source
# Notebook 15: AI Prediction Application Dashboard
# Goal: Build a simple AI-style dashboard for predictive maintenance decisions

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime

# Input tables created from Notebook 14
FULL_INFERENCE_TABLE = "workspace.default.iot_model_serving_inference_output"
LATEST_ENGINE_TABLE = "workspace.default.iot_latest_engine_status_dashboard"

# Load inference outputs
full_inference_df = spark.table(FULL_INFERENCE_TABLE)
latest_engine_df = spark.table(LATEST_ENGINE_TABLE)

print("Notebook 15 setup completed successfully.")
print("Full inference table:", FULL_INFERENCE_TABLE)
print("Latest engine dashboard table:", LATEST_ENGINE_TABLE)
print("Total inference records:", full_inference_df.count())
print("Total engines:", latest_engine_df.count())
print("Current time:", datetime.now())

display(latest_engine_df.orderBy("predicted_rul").limit(10))

# COMMAND ----------

# Dashboard KPI Summary

total_engines = float(latest_engine_df.count())

high_risk_count = float(latest_engine_df.filter(F.col("risk_category") == "High Risk").count())
medium_risk_count = float(latest_engine_df.filter(F.col("risk_category") == "Medium Risk").count())
low_risk_count = float(latest_engine_df.filter(F.col("risk_category") == "Low Risk").count())

avg_predicted_rul = float(latest_engine_df.select(F.round(F.avg("predicted_rul"), 2)).collect()[0][0])
min_predicted_rul = float(latest_engine_df.select(F.round(F.min("predicted_rul"), 2)).collect()[0][0])
max_predicted_rul = float(latest_engine_df.select(F.round(F.max("predicted_rul"), 2)).collect()[0][0])

kpi_data = [
    ("Total Engines Monitored", total_engines),
    ("High Risk Engines", high_risk_count),
    ("Medium Risk Engines", medium_risk_count),
    ("Low Risk Engines", low_risk_count),
    ("Average Predicted RUL", avg_predicted_rul),
    ("Lowest Predicted RUL", min_predicted_rul),
    ("Highest Predicted RUL", max_predicted_rul)
]

kpi_df = spark.createDataFrame(kpi_data, ["metric", "value"])

print("Dashboard KPI summary created successfully.")

display(kpi_df)

# COMMAND ----------

# Risk Category Distribution

risk_distribution_df = latest_engine_df \
    .groupBy("risk_category") \
    .count() \
    .withColumn("percentage", F.round((F.col("count") / F.lit(total_engines)) * 100, 2)) \
    .orderBy(
        F.when(F.col("risk_category") == "High Risk", 1)
         .when(F.col("risk_category") == "Medium Risk", 2)
         .otherwise(3)
    )

print("Risk category distribution created successfully.")

display(risk_distribution_df)

# COMMAND ----------

# High-Risk Engine Dashboard View

high_risk_engines_df = latest_engine_df \
    .filter(F.col("risk_category") == "High Risk") \
    .select(
        "unit_number",
        "time_in_cycles",
        "predicted_rul",
        "failure_risk_probability",
        "failure_risk_prediction",
        "risk_category",
        "maintenance_recommendation"
    ) \
    .orderBy("predicted_rul")

print("High-risk engine dashboard view created successfully.")
print("Total high-risk engines:", high_risk_engines_df.count())

display(high_risk_engines_df)

# COMMAND ----------

# Medium-Risk Engine Dashboard View

medium_risk_engines_df = latest_engine_df \
    .filter(F.col("risk_category") == "Medium Risk") \
    .select(
        "unit_number",
        "time_in_cycles",
        "predicted_rul",
        "failure_risk_probability",
        "failure_risk_prediction",
        "risk_category",
        "maintenance_recommendation"
    ) \
    .orderBy("predicted_rul")

print("Medium-risk engine dashboard view created successfully.")
print("Total medium-risk engines:", medium_risk_engines_df.count())

display(medium_risk_engines_df)

# COMMAND ----------

# Maintenance Priority Queue

maintenance_priority_df = latest_engine_df \
    .withColumn(
        "risk_level_score",
        F.when(F.col("risk_category") == "High Risk", 3)
         .when(F.col("risk_category") == "Medium Risk", 2)
         .otherwise(1)
    ) \
    .withColumn(
        "rul_urgency_score",
        F.when(F.col("predicted_rul") <= 30, 3)
         .when((F.col("predicted_rul") > 30) & (F.col("predicted_rul") <= 60), 2)
         .otherwise(1)
    ) \
    .withColumn(
        "maintenance_priority_score",
        F.round(
            (F.col("risk_level_score") * 30) +
            (F.col("rul_urgency_score") * 20) +
            (F.col("failure_risk_probability") * 50),
            2
        )
    ) \
    .select(
        "unit_number",
        "time_in_cycles",
        "predicted_rul",
        "failure_risk_probability",
        "risk_category",
        "maintenance_priority_score",
        "maintenance_recommendation"
    ) \
    .orderBy(F.desc("maintenance_priority_score"), F.asc("predicted_rul"))

print("Maintenance priority queue created successfully.")

display(maintenance_priority_df.limit(25))

# COMMAND ----------

# Single Engine AI Lookup

# Change this engine number anytime to check another engine
selected_engine_id = 35

selected_engine_status_df = latest_engine_df \
    .filter(F.col("unit_number") == selected_engine_id) \
    .select(
        "unit_number",
        "time_in_cycles",
        "predicted_rul",
        "failure_risk_probability",
        "failure_risk_prediction",
        "risk_category",
        "maintenance_recommendation"
    )

print(f"AI maintenance lookup for Engine {selected_engine_id}")

display(selected_engine_status_df)

# COMMAND ----------

# Selected Engine Prediction Trend
# This shows how predicted RUL and failure risk changed across cycles for one engine

selected_engine_history_df = full_inference_df \
    .filter(F.col("unit_number") == selected_engine_id) \
    .select(
        "unit_number",
        "time_in_cycles",
        "predicted_rul",
        "failure_risk_probability",
        "failure_risk_prediction",
        "risk_category",
        "maintenance_recommendation"
    ) \
    .orderBy("time_in_cycles")

print(f"Prediction history created for Engine {selected_engine_id}")
print("Total cycle records:", selected_engine_history_df.count())

display(selected_engine_history_df)

# COMMAND ----------

#AI Maintenance Decision Report

# Get latest engine status
latest_engine_row = selected_engine_status_df.collect()[0]

# Find first Medium Risk cycle
first_medium_df = selected_engine_history_df \
    .filter(F.col("risk_category") == "Medium Risk") \
    .orderBy("time_in_cycles") \
    .limit(1)

# Find first High Risk cycle
first_high_df = selected_engine_history_df \
    .filter(F.col("risk_category") == "High Risk") \
    .orderBy("time_in_cycles") \
    .limit(1)

first_medium_cycle = first_medium_df.collect()[0]["time_in_cycles"] if first_medium_df.count() > 0 else "Not reached"
first_high_cycle = first_high_df.collect()[0]["time_in_cycles"] if first_high_df.count() > 0 else "Not reached"

# Create AI-style maintenance report
report_data = [
    ("Engine ID", str(latest_engine_row["unit_number"])),
    ("Latest Cycle", str(latest_engine_row["time_in_cycles"])),
    ("Predicted Remaining Useful Life", str(latest_engine_row["predicted_rul"]) + " cycles"),
    ("Failure Risk Probability", str(round(latest_engine_row["failure_risk_probability"] * 100, 2)) + "%"),
    ("Current Risk Category", latest_engine_row["risk_category"]),
    ("First Medium Risk Cycle", str(first_medium_cycle)),
    ("First High Risk Cycle", str(first_high_cycle)),
    ("Recommended Action", latest_engine_row["maintenance_recommendation"])
]

ai_report_df = spark.createDataFrame(report_data, ["report_field", "report_value"])

print("AI maintenance decision report created successfully.")

display(ai_report_df)

print("\nAI Summary:")
print(
    f"Engine {latest_engine_row['unit_number']} is currently classified as {latest_engine_row['risk_category']}. "
    f"The model predicts only {latest_engine_row['predicted_rul']} cycles of remaining useful life, "
    f"with a failure-risk probability of {round(latest_engine_row['failure_risk_probability'] * 100, 2)}%. "
    f"The engine first entered Medium Risk at cycle {first_medium_cycle} and High Risk at cycle {first_high_cycle}. "
    f"Recommended action: {latest_engine_row['maintenance_recommendation']}."
)

# COMMAND ----------

# Save AI dashboard outputs as Delta tables

kpi_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.iot_ai_dashboard_kpi_summary")

risk_distribution_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.iot_ai_risk_distribution")

maintenance_priority_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.iot_ai_maintenance_priority_queue")

selected_engine_history_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.iot_ai_selected_engine_history")

ai_report_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.iot_ai_selected_engine_report")

print("AI dashboard output tables saved successfully.")

print("Saved tables:")
print("workspace.default.iot_ai_dashboard_kpi_summary")
print("workspace.default.iot_ai_risk_distribution")
print("workspace.default.iot_ai_maintenance_priority_queue")
print("workspace.default.iot_ai_selected_engine_history")
print("workspace.default.iot_ai_selected_engine_report")

# COMMAND ----------

# Verify saved AI dashboard tables

print("KPI Summary:")
display(spark.table("workspace.default.iot_ai_dashboard_kpi_summary"))

print("Risk Distribution:")
display(spark.table("workspace.default.iot_ai_risk_distribution"))

print("Maintenance Priority Queue:")
display(
    spark.table("workspace.default.iot_ai_maintenance_priority_queue")
    .orderBy(F.desc("maintenance_priority_score"))
    .limit(10)
)

print("Selected Engine AI Report:")
display(spark.table("workspace.default.iot_ai_selected_engine_report"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook 15 Summary
# MAGIC
# MAGIC In this notebook, I built a simple AI-style predictive maintenance dashboard using the inference outputs from the registered models.
# MAGIC
# MAGIC The dashboard includes:
# MAGIC
# MAGIC - KPI summary for monitored engines
# MAGIC - Risk category distribution
# MAGIC - High-risk and medium-risk engine views
# MAGIC - Maintenance priority queue
# MAGIC - Single engine AI lookup
# MAGIC - Prediction history for a selected engine
# MAGIC - AI maintenance decision report
# MAGIC
# MAGIC For Engine 35, the dashboard showed that the engine entered Medium Risk at cycle 133 and High Risk at cycle 161. At the latest cycle, the model predicted 8.76 cycles of remaining useful life with a 99.59% failure-risk probability, recommending immediate maintenance inspection.
# MAGIC
# MAGIC The final dashboard outputs were saved as Delta tables for reuse in reporting, visualization, or future deployment.