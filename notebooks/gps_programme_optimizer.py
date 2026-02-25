# Databricks notebook source
# MAGIC %md
# MAGIC # Capital Programme Optimizer
# MAGIC This notebook runs the optimization and logs the results to MLflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up dependencies

# COMMAND ----------

# The model requires ortools to be installed
from mlflow_model import OptimizerPyFuncModel
from main import run_optimization
from mlflow.models.signature import infer_signature
import pandas as pd
import mlflow
from pathlib import Path
import time
import sys
import os
import json
%pip install ortools

# COMMAND ----------


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


# COMMAND ----------

# MAGIC %md
# MAGIC ## Define parameters

# COMMAND ----------

# Define Widgets for Parameters
dbutils.widgets.text("funding_level", "1500.0", "Annual Funding Level ($M)")
valid_dimensions = [
    "Total",
    "Inclusive Access",
    "Healthy and safe people",
    "Economic Prosperity",
    "Environmental Sustainability",
    "Resilience and Security",
]
dbutils.widgets.dropdown("dimension", "Total", valid_dimensions, "Dimension")
dbutils.widgets.text("start_year", "2026", "Start Year")
dbutils.widgets.text("horizon", "60", "Horizon (Years)")
dbutils.widgets.text(
    "overflow_tiers", "0.12:1000,0.15:4000,0.20:12000", "Overflow Tiers")
dbutils.widgets.dropdown("optimizer", "cp-sat",
                         ["cp-sat", "optimizer"], "Optimizer Backend")
dbutils.widgets.text("time_limit", "300.0", "Time Limit (s)")
dbutils.widgets.text("workers", "0", "Num Workers")
dbutils.widgets.text("costs_path", "input/costs.csv", "Costs CSV Path")
dbutils.widgets.text(
    "benefits_path", "input/benefits.csv", "Benefits CSV Path")
dbutils.widgets.text(
    "output_dir",
    "/dbfs/FileStore/capital_optimizer/output",
    "Output Directory (DBFS mount)",
)
dbutils.widgets.text("model_tag", "", "Model Tag (Optional)")

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
model_tag = dbutils.widgets.get("model_tag")
args.generate_only = False
args.relax = False

# Ensure the DBFS-backed output directory exists on the local /dbfs mount
output_dir_path = Path(args.output_dir)
output_dir_path.mkdir(parents=True, exist_ok=True)
# Normalize to string in case downstream expects a string path
args.output_dir = str(output_dir_path)

print(f"Running optimization with config: {vars(args)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Optimisation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trigger Model

# COMMAND ----------

# Start MLflow run
# experiment_name = "CapitalProgrammeOptimizer"
# mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=f"opt_{args.dimension}_{args.funding_level}"):
    # Log parameters
    mlflow.log_params(vars(args))

    # Log input parameters as a JSON artifact
    config_file = output_dir_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=4)
    mlflow.log_artifact(str(config_file), artifact_path="input_data")

    # Log model tag if provided
    if model_tag:
        mlflow.set_tag("model_tag", model_tag)

    # Log input data files as artifacts
    for path_arg, name in [(args.costs_path, "costs"), (args.benefits_path, "benefits")]:
        input_file = Path(path_arg) if os.path.isabs(
            path_arg) else project_root / path_arg
        if input_file.exists():
            mlflow.log_artifact(str(input_file), artifact_path="input_data")
        else:
            print(
                f"Warning: {name.capitalize()} file not found at {input_file}, skipping artifact logging")

    # Run Optimization
    start_time = time.perf_counter()
    result, outputs = run_optimization(args)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    mlflow.log_metric("optimization_time_seconds", elapsed)
    print(f"Optimization time: {elapsed:.2f} seconds")

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

        # Log output data artifacts (CSVs)
        # Explicitly log only the CSVs to avoid duplicating logs and lp files
        # which are subdirectories of output_dir but are logged to their own top-level artifact paths
        schedule_file = outputs.get("schedule")
        if schedule_file and os.path.exists(schedule_file):
            mlflow.log_artifact(schedule_file, artifact_path="output_data")

        cash_flow_file = outputs.get("cash_flow")
        if cash_flow_file and os.path.exists(cash_flow_file):
            mlflow.log_artifact(cash_flow_file, artifact_path="output_data")

        # Log exported model representation alongside other artifacts
        lp_file = outputs.get("lp_file")
        if lp_file and os.path.exists(lp_file):
            mlflow.log_artifact(lp_file, artifact_path="solver_model")

        # Log MLflow pyfunc model for registration/serving in Databricks
        model_input_example = pd.DataFrame(
            [
                {
                    "funding_level": args.funding_level,
                    "dimension": args.dimension,
                    "start_year": args.start_year,
                    "horizon": args.horizon,
                    "overflow_tiers": args.overflow_tiers,
                    "optimizer": args.optimizer,
                    "time_limit": args.time_limit,
                    "workers": args.workers,
                    "costs_path": args.costs_path,
                    "benefits_path": args.benefits_path,
                    "output_dir": args.output_dir,
                    "generate_only": args.generate_only,
                    "relax": args.relax,
                }
            ]
        )
        model_output_example = pd.DataFrame(
            [
                {
                    "status": result.status,
                    "objective_value": float(result.objective_value),
                    "gap": float(result.gap),
                    "total_spend": float(total_spend),
                    "schedule_file": schedule_file,
                    "cash_flow_file": cash_flow_file,
                    "log_file": outputs.get("log_file") or getattr(result, "log_file", None),
                }
            ]
        )
        model_signature = infer_signature(
            model_input_example, model_output_example)
        optimizer_model = OptimizerPyFuncModel(base_args=vars(args))

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=optimizer_model,
            code_paths=[str(src_dir)],
            signature=model_signature,
            input_example=model_input_example,
        )

        # Also log the log file
        log_file = outputs.get("log_file") or getattr(result, "log_file", None)
        if log_file and os.path.exists(log_file):
            mlflow.log_artifact(log_file, artifact_path="logs")
        else:
            # Fallback logic if log_file not populated (e.g. error) or missing
            log_dir = project_root / "logs"
            if log_dir.exists():
                logs = list(log_dir.glob("*.log"))
                if logs:
                    latest_log = max(logs, key=os.path.getctime)
                    mlflow.log_artifact(str(latest_log), artifact_path="logs")

        print("Run complete. Metrics and artifacts logged to MLflow.")
    else:
        print("Optimization failed or no solution found.")
        mlflow.log_param("status", "FAILED")
