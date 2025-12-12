#!/usr/bin/env python3
"""
Test script to verify MLflow artifact logging for input and output data.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import mlflow

from main import run_optimization

# Add src to path
cwd = Path(os.getcwd())
project_root = cwd
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))




def test_mlflow_artifacts():
    """Test that input and output data are logged as MLflow artifacts."""

    # Create args similar to the notebook
    class Args:
        pass

    args = Args()
    args.funding_level = 1500.0
    args.dimension = "Total"
    args.start_year = 2026
    args.horizon = 10  # Shorter horizon for quick test
    args.overflow_tiers = "0.12:1000,0.15:4000,0.20:12000"
    args.optimizer = "cp-sat"
    args.time_limit = 30.0  # Short time limit for testing
    args.workers = 0
    args.costs_path = "input/costs.csv"
    args.benefits_path = "input/benefits.csv"
    args.output_dir = "output"
    args.generate_only = False
    args.relax = False

    print(f"Testing MLflow artifact logging with config: {vars(args)}")

    # Use a temporary directory for MLflow tracking
    with tempfile.TemporaryDirectory() as temp_dir:
        mlflow.set_tracking_uri(f"file://{temp_dir}/mlruns")

        with mlflow.start_run(run_name=f"test_opt_{args.dimension}_{args.funding_level}"):
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
                print(f"✓ Logged input artifact: {costs_file}")
            else:
                print(f"⚠ Warning: Costs file not found at {costs_file}, skipping artifact logging")

            if benefits_file.exists():
                mlflow.log_artifact(str(benefits_file), artifact_path="input_data")
                print(f"✓ Logged input artifact: {benefits_file}")
            else:
                print(f"⚠ Warning: Benefits file not found at {benefits_file}, skipping artifact logging")

            # Run Optimization
            print("\nRunning optimization...")
            start_time = time.perf_counter()
            result = run_optimization(args)
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
                output_dir = project_root / "output"
                if output_dir.exists():
                    mlflow.log_artifacts(str(output_dir), artifact_path="output_data")
                    print(f"✓ Logged output artifacts from: {output_dir}")

                # Also log the log file
                if result.log_file and os.path.exists(result.log_file):
                    mlflow.log_artifact(result.log_file, artifact_path="logs")
                    print(f"✓ Logged log file: {result.log_file}")
                else:
                    # Fallback logic if log_file not populated
                    log_dir = project_root / "logs"
                    if log_dir.exists():
                        logs = list(log_dir.glob("*.log"))
                        if logs:
                            latest_log = max(logs, key=os.path.getctime)
                            mlflow.log_artifact(str(latest_log), artifact_path="logs")
                            print(f"✓ Logged log file: {latest_log}")

                print("\n✓ Test complete. Metrics and artifacts logged to MLflow.")
                print(f"✓ Status: {result.status}")
                print(f"✓ Objective: {result.objective_value:,.2f}")
                print(f"✓ Gap: {result.gap:.4%}")
                return True
            else:
                print("✗ Optimization failed or no solution found.")
                mlflow.log_param("status", "FAILED")
                return False


if __name__ == "__main__":
    success = test_mlflow_artifacts()
    sys.exit(0 if success else 1)
