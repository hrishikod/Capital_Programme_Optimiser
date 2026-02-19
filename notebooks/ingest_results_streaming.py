# Databricks notebook source
# MAGIC %md
# MAGIC # Stream Optimization Results (Auto Loader)
# MAGIC
# MAGIC This notebook continuously monitors the MLflow experiment directory for new result files (`schedule.csv` and `cash_flow.csv`) and ingests them into Delta tables.
# MAGIC It automatically extracts the `run_id` from the file path.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import logging
from pyspark.sql.functions import col, lit, current_timestamp, input_file_name, regexp_extract

# Set up logging
logger = logging.getLogger("StreamIngest")
logging.basicConfig(level=logging.INFO)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

# Default Experiment ID
default_experiment_id = "101f7e40d5334a8f994da4404408481d"

dbutils.widgets.text("experiment_id", default_experiment_id, "Experiment ID")
dbutils.widgets.text("schedule_table", "optimiser_schedule", "Target Schedule Table")
dbutils.widgets.text("cash_flow_table", "optimiser_cash_flow", "Target Cash Flow Table")
dbutils.widgets.text("config_table", "optimiser_config", "Target Config Table")
dbutils.widgets.text("checkpoint_dir", "/dbfs/FileStore/capital_optimizer/checkpoints", "Checkpoint Directory")

experiment_id = dbutils.widgets.get("experiment_id")
schedule_table = dbutils.widgets.get("schedule_table")
cash_flow_table = dbutils.widgets.get("cash_flow_table")
config_table = dbutils.widgets.get("config_table")
checkpoint_dir = dbutils.widgets.get("checkpoint_dir")

# MLflow Artifacts Root for the Experiment
# Default DBFS location: dbfs:/databricks/mlflow-tracking/<experiment_id>
source_path = f"dbfs:/databricks/mlflow-tracking/{experiment_id}"

print(f"Monitoring Source: {source_path}")
print(f"Target Schedule Table: {schedule_table}")
print(f"Target Cash Flow Table: {cash_flow_table}")
print(f"Target Config Table: {config_table}")
print(f"Checkpoint Dir: {checkpoint_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stream Logic

# COMMAND ----------

def stream_file_type(file_name, target_table, checkpoint_subdir, format="csv"):
    """
    Sets up a stream for a specific file name (e.g., 'schedule.csv' or 'config.json').
    """
    
    # Path Glob to find specific files deep in directory structure
    # CSV/Output: .../<run_id>/artifacts/output_data/<file_name>
    # JSON/Input: .../<run_id>/artifacts/input_data/<file_name>
    # We use a broader recursive lookup and filter by filename, so specific parent folder matter less 
    # but we can try to be specific if needed. For now, matching filename recursively is robust.
    
    # Checkpoint location for this specific stream
    ckpt_path = f"{checkpoint_dir}/{checkpoint_subdir}"
    
    logger.info(f"Starting stream for {file_name}...")
    
    reader = (spark.readStream
                 .format("cloudFiles")
                 .option("cloudFiles.format", format)
                 .option("cloudFiles.schemaLocation", f"{ckpt_path}/schema")
                 .option("cloudFiles.inferColumnTypes", "true"))
    
    if format == "csv":
        reader = reader.option("header", "true")
                 
    df_stream = (reader
                 .option("pathGlobFilter", file_name)  # Filter for specific file
                 .option("recursiveFileLookup", "true") # Look recursively
                 .load(source_path))

    # Add Metadata
    r_extract = regexp_extract(col("source_file"), r".*/([a-f0-9]{32})/artifacts/.*", 1)

    df_transformed = (df_stream
                      .withColumn("ingestion_timestamp", current_timestamp())
                      .withColumn("source_file", input_file_name())
                      .withColumn("run_id", r_extract)
                      )

    # WriteStream to Delta
    query = (df_transformed.writeStream
             .format("delta")
             .outputMode("append")
             .option("checkpointLocation", ckpt_path)
             .option("mergeSchema", "true")
             .trigger(availableNow=True) # Process all new data then stop (batch-like)
             .table(target_table))
             
    return query

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Streams

# COMMAND ----------

# Stream 1: Schedule
query_schedule = stream_file_type("schedule.csv", schedule_table, "schedule", "csv")

# Stream 2: Cash Flow
query_cash_flow = stream_file_type("cash_flow.csv", cash_flow_table, "cash_flow", "csv")

# Stream 3: Config
query_config = stream_file_type("config.json", config_table, "config", "json")

# Wait for all to finish
query_schedule.awaitTermination()
query_cash_flow.awaitTermination()
query_config.awaitTermination()

print("Ingestion complete.")
