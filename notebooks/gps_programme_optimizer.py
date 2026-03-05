# Databricks notebook source
# MAGIC %md
# MAGIC # Capital Programme Optimizer
# MAGIC This notebook runs the optimization and logs the results to MLflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up dependencies

# COMMAND ----------

# The model requires ortools to be installed
%pip install ortools

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load libraries and project modules

# COMMAND ----------

import json
import os
import sys
import time
from pathlib import Path

import cloudpickle
import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature


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
import mlflow_model as mlflow_model_module
from mlflow_model import OptimizerPyFuncModel

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
dbutils.widgets.text("time_limit", "30.0", "Time Limit (s)")
dbutils.widgets.text("workers", "0", "Num Workers")
dbutils.widgets.dropdown("run_permutations", "false", [
                         "false", "true"], "Run Cost/Benefit Permutations")
dbutils.widgets.text("costs_path", "input/costs.csv", "Costs CSV Path")
dbutils.widgets.text(
    "benefits_path", "input/benefits.csv", "Benefits CSV Path")
dbutils.widgets.dropdown(
    "run_mode",
    "both",
    ["both", "optimize_only", "model_only"],
    "Run Mode",
)
dbutils.widgets.text(
    "model_source_run_id",
    "",
    "Model-only Source Run ID (required for model_only)",
)
dbutils.widgets.text(
    "output_dir",
    "/dbfs/FileStore/capital_optimizer/output",
    "Output Directory (DBFS mount)",
)
dbutils.widgets.text("model_tag", "", "Model Tag (Optional)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse widget values

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
args.run_permutations = dbutils.widgets.get(
    "run_permutations").strip().lower() == "true"
args.costs_path = dbutils.widgets.get("costs_path")
args.benefits_path = dbutils.widgets.get("benefits_path")
run_mode = dbutils.widgets.get("run_mode")
model_source_run_id = dbutils.widgets.get("model_source_run_id").strip()
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
print(f"Run mode: {run_mode}")
if model_source_run_id:
    print(f"Model source run ID: {model_source_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve run mode and effective parameters

# COMMAND ----------


valid_modes = {"both", "optimize_only", "model_only"}
if run_mode not in valid_modes:
    raise ValueError(
        f"Invalid run_mode '{run_mode}'. Expected one of {sorted(valid_modes)}")


def _to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if run_mode == "model_only":
    if not model_source_run_id:
        raise ValueError(
            "model_source_run_id is required when run_mode='model_only' to avoid parameter drift.")

    source_run = mlflow.get_run(model_source_run_id)
    source_params = source_run.data.params

    required_params = {
        "funding_level": float,
        "dimension": str,
        "start_year": int,
        "horizon": int,
        "overflow_tiers": str,
        "optimizer": str,
        "time_limit": float,
        "workers": int,
        "costs_path": str,
        "benefits_path": str,
        "output_dir": str,
        "generate_only": _to_bool,
        "relax": _to_bool,
    }

    missing = [key for key in required_params if key not in source_params]
    if missing:
        raise ValueError(
            f"Source run {model_source_run_id} is missing required params for model packaging: {missing}")

    for key, converter in required_params.items():
        setattr(args, key, converter(source_params[key]))

    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir_path)

    print(
        f"Using parameters from source run {model_source_run_id} for model_only packaging.")

effective_run_name = f"{run_mode}_{args.dimension}_{args.funding_level}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute workflow and log to MLflow

# COMMAND ----------

# Start MLflow run
# experiment_name = "CapitalProgrammeOptimizer"
# mlflow.set_experiment(experiment_name)
with mlflow.start_run(run_name=effective_run_name):
    # ---- Run metadata and shared artifact logging ----
    if run_mode == "model_only":
        mlflow.set_tag("model_source_run_id", model_source_run_id)

    # Log parameters after any model_only source-run overrides
    mlflow.log_params(vars(args))
    mlflow.log_param("run_mode", run_mode)

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

    result = None
    outputs = {}
    total_spend = None

    # ---- Optimisation branch (both / optimize_only) ----
    if run_mode in {"both", "optimize_only"}:
        start_time = time.perf_counter()
        result, outputs = run_optimization(args)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        mlflow.log_metric("optimization_time_seconds", elapsed)
        print(f"Optimization time: {elapsed:.2f} seconds")

        if result:
            mlflow.log_metric("objective_value", result.objective_value)
            mlflow.log_metric("gap", result.gap)

            if result.breakdown:
                for k, v in result.breakdown.items():
                    mlflow.log_metric(k, v)

            total_spend = float(result.spend_profile.iloc[0, :].sum())
            mlflow.log_metric("total_spend", total_spend)

            schedule_file = outputs.get("schedule")
            if schedule_file and os.path.exists(schedule_file):
                mlflow.log_artifact(schedule_file, artifact_path="output_data")

            cash_flow_file = outputs.get("cash_flow")
            if cash_flow_file and os.path.exists(cash_flow_file):
                mlflow.log_artifact(
                    cash_flow_file, artifact_path="output_data")

            lp_file = outputs.get("lp_file")
            if lp_file and os.path.exists(lp_file):
                mlflow.log_artifact(lp_file, artifact_path="solver_model")

            log_file = outputs.get("log_file") or getattr(
                result, "log_file", None)
            if log_file and os.path.exists(log_file):
                mlflow.log_artifact(log_file, artifact_path="logs")
            else:
                log_dir = project_root / "logs"
                if log_dir.exists():
                    logs = list(log_dir.glob("*.log"))
                    if logs:
                        latest_log = max(logs, key=os.path.getctime)
                        mlflow.log_artifact(
                            str(latest_log), artifact_path="logs")

            permutation_summary_file = outputs.get("permutation_summary")
            if permutation_summary_file and os.path.exists(permutation_summary_file):
                mlflow.log_artifact(
                    permutation_summary_file, artifact_path="output_data")
        else:
            print("Optimization failed or no solution found.")
            mlflow.log_param("status", "FAILED")

    # ---- Model packaging branch (both / model_only) ----
    if run_mode in {"both", "model_only"}:
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

        schedule_file = outputs.get("schedule")
        cash_flow_file = outputs.get("cash_flow")
        log_file = outputs.get("log_file") or (
            getattr(result, "log_file", None) if result else None)

        if result:
            output_status = result.status
            output_objective = float(result.objective_value)
            output_gap = float(result.gap)
            output_total_spend = float(
                total_spend) if total_spend is not None else None
        else:
            output_status = "NOT_RUN"
            output_objective = None
            output_gap = None
            output_total_spend = None

        model_output_example = pd.DataFrame(
            [
                {
                    "status": output_status,
                    "objective_value": output_objective,
                    "gap": output_gap,
                    "total_spend": output_total_spend,
                    "schedule_file": schedule_file,
                    "cash_flow_file": cash_flow_file,
                    "log_file": log_file,
                }
            ]
        )
        model_signature = infer_signature(
            model_input_example, model_output_example)
        cloudpickle.register_pickle_by_value(mlflow_model_module)
        optimizer_model = OptimizerPyFuncModel(base_args=vars(args))

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=optimizer_model,
            code_paths=[str(src_dir)],
            signature=model_signature,
            input_example=model_input_example,
        )
        print("MLflow pyfunc model artifact logged to 'model'.")

    print("Run complete. Metrics and artifacts logged to MLflow.")
