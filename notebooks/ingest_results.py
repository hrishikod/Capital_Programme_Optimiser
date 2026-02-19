# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Optimization Results
# MAGIC
# MAGIC This notebook ingests the output CSVs (schedule and cash flow) from the Capital Programme Optimizer and saves them to Delta tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql.functions import lit, current_timestamp
import logging
import mlflow

# Set up logging
logger = logging.getLogger("IngestResults")
logging.basicConfig(level=logging.INFO)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text("schedule_path", "", "Schedule CSV Path")
dbutils.widgets.text("cash_flow_path", "", "Cash Flow CSV Path")
dbutils.widgets.text("schedule_table", "optimiser_schedule", "Target Schedule Table")
dbutils.widgets.text("cash_flow_table", "optimiser_cash_flow", "Target Cash Flow Table")
dbutils.widgets.text("run_id", "", "Run ID (Optional)")
dbutils.widgets.dropdown("mode", "append", ["append", "overwrite"], "Write Mode")

schedule_path = dbutils.widgets.get("schedule_path")
cash_flow_path = dbutils.widgets.get("cash_flow_path")
schedule_table = dbutils.widgets.get("schedule_table")
cash_flow_table = dbutils.widgets.get("cash_flow_table")
run_id = dbutils.widgets.get("run_id")
mode = dbutils.widgets.get("mode")

if run_id and not (schedule_path and cash_flow_path):
    print(f"Run ID provided ({run_id}), attempting to infer paths from MLflow artifacts...")
    try:
        run_info = mlflow.get_run(run_id).info
        artifact_uri = run_info.artifact_uri
        # artifact_uri is usually dbfs:/... or similar.
        # We know our files are in output_data/
        # Spark can read dbfs:/ paths directly.
        
        if not schedule_path:
            schedule_path = f"{artifact_uri}/output_data/schedule.csv"
            print(f"Inferred Schedule Path: {schedule_path}")
            
        if not cash_flow_path:
            cash_flow_path = f"{artifact_uri}/output_data/cash_flow.csv"
            print(f"Inferred Cash Flow Path: {cash_flow_path}")
            
    except Exception as e:
        logger.error(f"Failed to infer paths from Run ID: {e}")
        print("Please provide explicit paths.")

print(f"Final Schedule Path: {schedule_path}")
print(f"Final Cash Flow Path: {cash_flow_path}")
print(f"Target Schedule Table: {schedule_table}")
print(f"Target Cash Flow Table: {cash_flow_table}")
print(f"Run ID: {run_id}")
print(f"Mode: {mode}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingestion Logic

# COMMAND ----------

def ingest_csv(path, table_name, run_id_val, write_mode):
    if not path:
        logger.info(f"No path provided for {table_name}, skipping.")
        return

    try:
        # Read CSV
        # Assuming header is present
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(path)

        # Add metadata columns
        df = df.withColumn("ingestion_timestamp", current_timestamp())
        if run_id_val:
            df = df.withColumn("run_id", lit(run_id_val))
        
        # Write to Delta
        (df.write
         .format("delta")
         .mode(write_mode)
         .option("mergeSchema", "true")
         .saveAsTable(table_name))
        
        logger.info(f"Successfully wrote {path} to table {table_name}")
        display(df)  # Show sample of written data
        
    except Exception as e:
        logger.error(f"Error ingesting {path} to {table_name}: {str(e)}")
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Ingestion

# COMMAND ----------

if schedule_path:
    ingest_csv(schedule_path, schedule_table, run_id, mode)
else:
    print("No schedule path provided.")

# COMMAND ----------

if cash_flow_path:
    ingest_csv(cash_flow_path, cash_flow_table, run_id, mode)
else:
    print("No cash flow path provided.")
