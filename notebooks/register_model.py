# Databricks notebook source
# MAGIC %md
# MAGIC # Register Optimiser Model in MLflow
# MAGIC This notebook registers the `model/` artifact from an existing optimiser run.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import time

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text("run_id", "", "Run ID (required)")
dbutils.widgets.text(
    "model_name", "capital-programme-optimizer", "Registered Model Name")
dbutils.widgets.dropdown("target_stage", "Staging", [
                         "None", "Staging", "Production", "Archived"], "Target Stage")
dbutils.widgets.dropdown("archive_existing_versions", "false", [
                         "false", "true"], "Archive existing in target stage")
dbutils.widgets.text("version_description", "",
                     "Version Description (optional)")
dbutils.widgets.dropdown("preflight_load", "true", [
                         "true", "false"], "Preflight model load check")
dbutils.widgets.dropdown("preflight_predict", "false", [
                         "false", "true"], "Optional preflight predict smoke test")

RUN_ID = dbutils.widgets.get("run_id").strip()
MODEL_NAME = dbutils.widgets.get("model_name").strip()
TARGET_STAGE = dbutils.widgets.get("target_stage").strip()
ARCHIVE_EXISTING = dbutils.widgets.get(
    "archive_existing_versions").strip().lower() == "true"
VERSION_DESCRIPTION = dbutils.widgets.get("version_description").strip()
PREFLIGHT_LOAD = dbutils.widgets.get(
    "preflight_load").strip().lower() == "true"
PREFLIGHT_PREDICT = dbutils.widgets.get(
    "preflight_predict").strip().lower() == "true"

if not RUN_ID:
    raise ValueError(
        "run_id is required. Paste the run ID from gps_programme_optimizer output/MLflow run page.")

if not MODEL_NAME:
    raise ValueError("model_name is required.")

MODEL_URI = f"runs:/{RUN_ID}/model"

# Use legacy Workspace Model Registry (non-Unity Catalog)
mlflow.set_registry_uri("databricks")

print(f"Run ID: {RUN_ID}")
print(f"Model URI: {MODEL_URI}")
print(f"Registered Model: {MODEL_NAME}")
print(f"Target Stage: {TARGET_STAGE}")


print("Registry URI: databricks (Will have to change for Unity Catalog)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate run and model artifact

# COMMAND ----------

client = MlflowClient()

try:
    run = client.get_run(RUN_ID)
except Exception as exc:
    raise RuntimeError(f"Unable to fetch run '{RUN_ID}'.") from exc

print("Run found.")
print(f"Run status: {run.info.status}")

model_artifacts = client.list_artifacts(RUN_ID, "model")
if not model_artifacts:
    root_artifacts = client.list_artifacts(RUN_ID)
    available = [a.path for a in root_artifacts]
    raise RuntimeError(
        "No MLflow model artifact found at 'model/'. "
        f"Available top-level artifacts: {available}"
    )

print("Model artifact directory exists:")
for item in model_artifacts:
    print(f"  - {item.path}")

if PREFLIGHT_LOAD:
    print("Running preflight model load check...")
    try:
        loaded_model = mlflow.pyfunc.load_model(MODEL_URI)
        print("Preflight load passed.")

        if PREFLIGHT_PREDICT:
            print("Running optional preflight predict smoke test...")

            def _to_bool(value, default=False):
                if value is None:
                    return default
                return str(value).strip().lower() in {"1", "true", "yes", "y"}

            params = run.data.params
            smoke_input = pd.DataFrame(
                [
                    {
                        "funding_level": float(params.get("funding_level", 1500.0)),
                        "dimension": params.get("dimension", "Total"),
                        "start_year": int(params.get("start_year", 2026)),
                        "horizon": int(params.get("horizon", 2)),
                        "overflow_tiers": params.get(
                            "overflow_tiers", "0.12:1000,0.15:4000,0.20:12000"
                        ),
                        "optimizer": params.get("optimizer", "cp-sat"),
                        "time_limit": float(params.get("time_limit", 1.0)),
                        "workers": int(params.get("workers", 0)),
                        "costs_path": params.get("costs_path", "input/costs.csv"),
                        "benefits_path": params.get("benefits_path", "input/benefits.csv"),
                        "output_dir": params.get("output_dir", "output"),
                        "generate_only": _to_bool(params.get("generate_only"), default=True),
                        "relax": _to_bool(params.get("relax"), default=False),
                    }
                ]
            )

            _ = loaded_model.predict(smoke_input)
            print("Preflight predict smoke test passed.")
    except Exception as exc:
        root_message = str(exc)
        hints = [
            "Model artifact exists but failed to load with mlflow.pyfunc.load_model.",
            f"Root error: {type(exc).__name__}: {root_message}",
        ]

        if "No module named 'mlflow_model'" in root_message:
            hints.append(
                "This run was likely logged before the by-value packaging fix. "
                "Re-run gps_programme_optimizer.py to create a new run, then register that new run ID."
            )

        if "ortools" in root_message.lower() or "uninstalled" in root_message.lower():
            hints.append(
                "Install dependencies from the model artifact environment, or run this notebook in an environment with ortools available."
            )

        if "No module named 'src'" in root_message or "No module named 'main'" in root_message:
            hints.append(
                "The model code import path is incompatible with this environment. Re-log the model with latest src/mlflow_model.py and retry registration."
            )

        raise RuntimeError(" ".join(hints)) from exc

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure registered model exists

# COMMAND ----------

try:
    client.get_registered_model(MODEL_NAME)
    print(f"Registered model already exists: {MODEL_NAME}")
except Exception:
    client.create_registered_model(MODEL_NAME)
    print(f"Created registered model: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register or reuse model version

# COMMAND ----------

existing_versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
matching_version = None
for mv in existing_versions:
    mv_run_id = getattr(mv, "run_id", None)
    mv_source = getattr(mv, "source", "") or ""
    if mv_run_id == RUN_ID or mv_source.endswith(f"/{RUN_ID}/artifacts/model"):
        matching_version = mv
        break

if matching_version:
    version_number = str(matching_version.version)
    print(f"Reusing existing version {version_number} for run {RUN_ID}")
else:
    created = mlflow.register_model(model_uri=MODEL_URI, name=MODEL_NAME)
    version_number = str(created.version)
    print(f"Registered new model version: {version_number}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for READY and apply metadata/stage

# COMMAND ----------

for _ in range(30):
    details = client.get_model_version(name=MODEL_NAME, version=version_number)
    if details.status == "READY":
        break
    time.sleep(1)

details = client.get_model_version(name=MODEL_NAME, version=version_number)
if details.status != "READY":
    raise RuntimeError(
        f"Model version {version_number} not READY after waiting. Current status: {details.status}"
    )

description = VERSION_DESCRIPTION or f"Registered from run {RUN_ID}"
client.update_model_version(
    name=MODEL_NAME,
    version=version_number,
    description=description,
)

client.set_model_version_tag(
    MODEL_NAME, version_number, "source_run_id", RUN_ID)
client.set_model_version_tag(
    MODEL_NAME, version_number, "source_model_uri", MODEL_URI)

if TARGET_STAGE != "None":
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version_number,
        stage=TARGET_STAGE,
        archive_existing_versions=ARCHIVE_EXISTING,
    )
    print(f"Transitioned version {version_number} to stage: {TARGET_STAGE}")
else:
    print("Stage transition skipped (target_stage=None).")

print("Registration pipeline complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

summary = client.get_model_version(name=MODEL_NAME, version=version_number)
print("Registration Summary")
print(f"  Name: {summary.name}")
print(f"  Version: {summary.version}")
print(f"  Status: {summary.status}")
print(f"  Stage: {summary.current_stage}")
print(f"  Run ID: {summary.run_id}")
print(f"  Source: {summary.source}")
