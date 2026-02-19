# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Ingest Optimization Results
# MAGIC
# MAGIC This notebook queries MLflow for successful runs and ingests their results (`schedule.csv`, `cash_flow.csv`, `config.json`) into Delta tables.
# MAGIC It checks which `run_id`s are already in the tables and only processes new ones.
# MAGIC
# MAGIC ## How to Use
# MAGIC - **Schedule notebook as a Job** (e.g., hourly).
# MAGIC
# MAGIC ## Parameters
# MAGIC - `experiment_id`: The MLflow Experiment ID to monitor.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
import logging
from pyspark.sql.functions import lit, current_timestamp
from pyspark.sql.utils import AnalysisException

# Set up logging
logger = logging.getLogger("BatchIngest")
logger.setLevel(logging.INFO)

# Suppress py4j logging
logging.getLogger("py4j").setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

# Experiment ID
default_experiment_id = "101f7e40d5334a8f994da4404408481d"

dbutils.widgets.text("experiment_id", default_experiment_id, "Experiment ID")
dbutils.widgets.text("schedule_table", "capital_programme_optimiser.schedule", "Target Schedule Table")
dbutils.widgets.text("cash_flow_table", "capital_programme_optimiser.cash_flow", "Target Cash Flow Table")
dbutils.widgets.text("config_table", "capital_programme_optimiser.config", "Target Config Table")

experiment_id = dbutils.widgets.get("experiment_id")
schedule_table = dbutils.widgets.get("schedule_table")
cash_flow_table = dbutils.widgets.get("cash_flow_table")
config_table = dbutils.widgets.get("config_table")

# Ensure schema exists
spark.sql("CREATE SCHEMA IF NOT EXISTS capital_programme_optimiser")

print(f"Monitoring Experiment: {experiment_id}")
print(f"Target Schedule Table: {schedule_table}")
print(f"Target Cash Flow Table: {cash_flow_table}")
print(f"Target Config Table: {config_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def get_processed_run_ids(table_name):
    """
    Returns a set of run_ids that are already in the target table.
    """
    try:
        df = spark.table(table_name)
        # Check if table is empty or doesn't have run_id yet
        if "run_id" not in df.columns:
            return set()
        
        # Collect run_ids distinct
        rows = df.select("run_id").distinct().collect()
        return set(r.run_id for r in rows)
    except AnalysisException:
        # Table doesn't exist yet
        return set()

def ingest_file(run_id, file_path_suffix, target_table, format="csv"):
    """
    Ingests a single file from a run artifact into a Delta table.
    """
    
    try:
        # Download locally because Spark cannot access MLflow artifacts directly in some modes
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=file_path_suffix)
        
        # Read with Spark from local path
        spark_path = f"file://{local_path}"
        
        if format == "csv":
            df = spark.read.option("header", "true").option("inferSchema", "true").csv(spark_path)
        elif format == "json":
            df = spark.read.option("multiLine", "true").json(spark_path)
            
        # Add Metadata
        df = df.withColumn("ingestion_timestamp", current_timestamp()) \
               .withColumn("source_file", lit(file_path_suffix)) \
               .withColumn("run_id", lit(run_id))
        
        # Write to Delta (Append)
        (df.write
         .format("delta")
         .mode("append")
         .option("mergeSchema", "true")
         .saveAsTable(target_table))
         
        logger.info(f"Ingested {file_path_suffix} for run {run_id} into {target_table}")
        return True
        
    except Exception as e:
        # It's possible the file doesn't exist for this run (e.g. failed run or different structure)
        logger.warning(f"Could not ingest {file_path_suffix} for run {run_id}: {e}")
        return False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logic

# COMMAND ----------

# Get all successful runs from MLflow
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["start_time DESC"]
)

if not runs.empty:
    print(f"Found {len(runs)} finished runs.")
    
    # Get set of processed Run IDs for each table to avoid duplicates
    processed_schedule = get_processed_run_ids(schedule_table)
    processed_cash = get_processed_run_ids(cash_flow_table)
    processed_config = get_processed_run_ids(config_table)
    
    # Iterate through runs
    count = 0
    for index, row in runs.iterrows():
        run_id = row["run_id"]
        
        # Ingest Schedule
        if run_id not in processed_schedule:
            ingest_file(run_id, "output_data/schedule.csv", schedule_table, "csv")
            
        # Ingest Cash Flow
        if run_id not in processed_cash:
            ingest_file(run_id, "output_data/cash_flow.csv", cash_flow_table, "csv")
            
        # Ingest Config
        if run_id not in processed_config:
            ingest_file(run_id, "input_data/config.json", config_table, "json")
            
        count += 1
        
    print("Batch ingestion complete.")
else:
    print("No runs found.")
