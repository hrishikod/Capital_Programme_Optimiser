# Databricks notebook source
# MAGIC %md
# MAGIC # Model Registration Notebook
# MAGIC This notebook registers trained optimization models from MLflow runs to the model registry.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up dependencies

# COMMAND ----------

import json
import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure project paths

# COMMAND ----------

# Detect project root
cwd = Path(os.getcwd())
if (cwd / "src").exists():
    project_root = cwd
elif (cwd.parent / "src").exists():
    project_root = cwd.parent
else:
    project_root = cwd.parent

print(f"Current Working Directory: {cwd}")
print(f"Detected Project Root: {project_root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define parameters

# COMMAND ----------

# Define parameters for model registration
# Provide the run_id from the optimization run output

RUN_ID = None  # Set this to the run ID from gps_programme_optimizer.py, e.g., "abc123def456"

# Model registry settings
MODEL_NAME = "capital-programme-optimizer"  # Set a descriptive model name
MODEL_STAGE = "Staging"  # Options: "None", "Staging", "Production", "Archived"

print(f"Model Name: {MODEL_NAME}")
print(f"Target Stage: {MODEL_STAGE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Search for optimization run

# COMMAND ----------

client = MlflowClient()

# Get the run from the provided RUN_ID
if not RUN_ID:
    print("Error: RUN_ID must be set. Copy the run ID from gps_programme_optimizer.py output.")
    run = None
else:
    run = client.get_run(RUN_ID)
    print(f"Using run ID: {RUN_ID}")
    print(f"\nRun Details:")
    print(f"  Status: {run.info.status}")
    print(f"  Params: {run.data.params}")
    print(f"  Metrics: {run.data.metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register model from run

# COMMAND ----------

if run:
    try:
        # Get the model URI from the run
        model_uri = f"runs:/{run.info.run_id}/model"

        # Check if model artifacts exist in the run
        artifacts = client.list_artifacts(run.info.run_id, "model")
        if artifacts:
            print(f"Model artifacts found in run:")
            for artifact in artifacts:
                print(f"  - {artifact.path}")
        else:
            print("Warning: No model artifacts found in 'model' directory")
            # List all artifacts to see what's available
            all_artifacts = client.list_artifacts(run.info.run_id)
            print(f"Available artifacts:")
            for artifact in all_artifacts:
                print(f"  - {artifact.path}")
    except Exception as e:
        print(f"Error checking artifacts: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create and register model version

# COMMAND ----------

if run:
    try:
        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"

        model_version = mlflow.register_model(model_uri, MODEL_NAME)
        print(f"Model registered successfully!")
        print(f"  Model Name: {MODEL_NAME}")
        print(f"  Version: {model_version.version}")
        print(f"  URI: {model_uri}")

        # Update model metadata with run details
        description = f"Registered from run {run.info.run_id}"
        client.update_model_version(
            name=MODEL_NAME,
            version=model_version.version,
            description=description
        )
        print(f"  Description updated")

        # Transition to target stage
        if MODEL_STAGE and MODEL_STAGE != "None":
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=model_version.version,
                stage=MODEL_STAGE,
                archive_existing_versions=False
            )
            print(f"  Transitioned to stage: {MODEL_STAGE}")

    except Exception as e:
        print(f"Error registering model: {e}")
        print(f"Model artifacts may not exist in the run. Ensure gps_programme_optimizer.py logs the model artifacts.")
else:
    print("No run available for registration")

# COMMAND ----------

# MAGIC %md
# MAGIC ## List registered models

# COMMAND ----------

# Show all registered models and their versions
models = client.search_registered_models()
print(f"Registered Models ({len(models)}):")
print()

for model in models:
    print(f"Model: {model.name}")
    print(
        f"  Latest version: {model.latest_versions[0].version if model.latest_versions else 'N/A'}")
    for version in model.latest_versions:
        print(f"    Version {version.version}: {version.current_stage}")
    print()
