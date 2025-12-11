# Databricks notebook source
# MAGIC %md
# MAGIC # Capital Programme Optimizer
# MAGIC This notebook runs the optimization and logs the results to MLflow.

# COMMAND ----------

# The model requires ortools to be installed
# %pip install ortools

# COMMAND ----------

import sys
import os
import argparse
import mlflow
import json
from pathlib import Path

# NOTE - Add src to path if not already there
# We dynamically check where 'src' is located to be robust against CWD changes
cwd = Path(os.getcwd())
if (cwd / "src").exists():
    project_root = cwd
elif (cwd.parent / "src").exists():
    project_root = cwd.parent
else:
    # Fallback: assume notebook is in notebooks/ folder
    project_root = cwd.parent

print(f"Current Working Directory: {cwd}")
print(f"Detected Project Root: {project_root}")

# Add src to sys.path so we can import modules directly
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
    
# Import the optimizer
# Since src is in path, we import main directly
from main import run_optimization

# COMMAND ----------

# Define Widgets for Parameters
dbutils.widgets.text("funding_level", "1500.0", "Annual Funding Level ($M)")
valid_dimensions = ["Total", "Inclusive Access", "Healthy and safe people", "Economic Prosperity", "Environmental Sustainability", "Resilience and Security"]
dbutils.widgets.dropdown("dimension", "Total", valid_dimensions, "Dimension")
dbutils.widgets.text("start_year", "2026", "Start Year")
dbutils.widgets.text("horizon", "60", "Horizon (Years)")
dbutils.widgets.text("overflow_tiers", "0.12:1000,0.15:4000,0.20:12000", "Overflow Tiers")
dbutils.widgets.dropdown("optimizer", "cp-sat", ["cp-sat", "optimizer"], "Optimizer Backend")
dbutils.widgets.text("time_limit", "300.0", "Time Limit (s)")
dbutils.widgets.text("workers", "0", "Num Workers")
dbutils.widgets.text("costs_path", "input/costs.csv", "Costs CSV Path")
dbutils.widgets.text("benefits_path", "input/benefits.csv", "Benefits CSV Path")
dbutils.widgets.text("output_dir", "output", "Output Directory")

# COMMAND ----------

# Parse arguments from widgets
class Args:
    pass

args = Args()
args.funding_level = float(dbutils.widgets.get("funding_level"))
args.dimension = dbutils.widgets.get("dimension")
args.start_year = int(dbutils.widgets.get("start_year"))
args.horizon = int(dbutils.widgets.get("horizon"))
args.overflow_tiers = dbutils.widgets.get("overflow_tiers")
args.optimizer = dbutils.widgets.get("optimizer")
args.time_limit = float(dbutils.widgets.get("time_limit"))
args.workers = int(dbutils.widgets.get("workers"))
args.costs_path = dbutils.widgets.get("costs_path")
args.benefits_path = dbutils.widgets.get("benefits_path")
args.output_dir = dbutils.widgets.get("output_dir")
args.generate_only = False
args.relax = False

print(f"Running optimization with config: {vars(args)}")

# COMMAND ----------

# Start MLflow run
# experiment_name = "CapitalProgrammeOptimizer"
# mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=f"opt_{args.dimension}_{args.funding_level}"):
    # Log parameters
    mlflow.log_params(vars(args))
    
    # Log input data files as artifacts
    # Resolve costs and benefits file paths
    if os.path.isabs(args.costs_path):
        costs_file = Path(args.costs_path)
    else:
        costs_file = project_root / args.costs_path
        
    if os.path.isabs(args.benefits_path):
        benefits_file = Path(args.benefits_path)
    else:
        benefits_file = project_root / args.benefits_path
    
    # Log input files if they exist
    if costs_file.exists():
        mlflow.log_artifact(str(costs_file), artifact_path="input_data")
    else:
        print(f"Warning: Costs file not found at {costs_file}, skipping artifact logging")
        
    if benefits_file.exists():
        mlflow.log_artifact(str(benefits_file), artifact_path="input_data")
    else:
        print(f"Warning: Benefits file not found at {benefits_file}, skipping artifact logging")
    
    # Run Optimization
    result = run_optimization(args)
    
    if result:
        # Log Metrics
        mlflow.log_metric("objective_value", result.objective_value)
        mlflow.log_metric("gap", result.gap)
        
        if result.breakdown:
            for k, v in result.breakdown.items():
                mlflow.log_metric(k, v)
        
        # Calculate summary metrics from results
        total_spend = result.spend_profile.iloc[0, :].sum()
        mlflow.log_metric("total_spend", total_spend)
        
        # Log output data artifacts (CSVs)
        # result object doesn't have file paths, but run_optimization writes them to output/
        # We can find them or use the dataframes directly
        
        output_dir = project_root / "output"
        if output_dir.exists():
            mlflow.log_artifacts(str(output_dir), artifact_path="output_data")

        # Also log the log file
        if result.log_file and os.path.exists(result.log_file):
            mlflow.log_artifact(result.log_file, artifact_path="logs")
        else:
            # Fallback logic if log_file not populated (e.g. error) or missing
            log_dir = project_root / "logs"
            if log_dir.exists():
                logs = list(log_dir.glob("*.log"))
                if logs:
                    latest_log = max(logs, key=os.path.getctime)
                    mlflow.log_artifact(str(latest_log), artifact_path="logs")
                
        print(f"Run complete. Metrics and artifacts logged to MLflow.")
    else:
        print("Optimization failed or no solution found.")
        mlflow.log_param("status", "FAILED")
