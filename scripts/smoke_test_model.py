#!/usr/bin/env python3
"""Smoke test an MLflow pyfunc optimizer model without registering it."""

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd


DEFAULT_ROW = {
    "funding_level": 1500,
    "dimension": "Total",
    "start_year": 2026,
    "horizon": 60,
    "overflow_tiers": "0.12:1000,0.15:4000,0.20:12000",
    "optimizer": "cp-sat",
    "time_limit": 30,
    "workers": 0,
    "costs_path": "input/costs.csv",
    "benefits_path": "input/benefits.csv",
    "output_dir": "/dbfs/FileStore/capital_optimizer/output",
    "generate_only": False,
    "relax": False,
}


def _load_dataframe_from_payload(payload_path: Path) -> pd.DataFrame:
    with payload_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    split = payload.get("dataframe_split")
    if not isinstance(split, dict):
        raise ValueError("Payload must contain a 'dataframe_split' object.")

    columns = split.get("columns")
    data = split.get("data")
    if not isinstance(columns, list) or not isinstance(data, list):
        raise ValueError(
            "'dataframe_split' must include list fields 'columns' and 'data'.")

    return pd.DataFrame(data, columns=columns)


def _resolve_model_uri(run_id: str | None, model_uri: str | None) -> str:
    if model_uri:
        return model_uri
    if run_id:
        return f"runs:/{run_id}/model"
    raise ValueError("Provide either --run-id or --model-uri.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", help="MLflow run ID that logged artifact path 'model'.")
    parser.add_argument(
        "--model-uri", help="Explicit model URI (for example runs:/<run_id>/model).")
    parser.add_argument(
        "--payload-json",
        type=Path,
        help="Optional path to request payload JSON containing dataframe_split.",
    )
    parser.add_argument("--tracking-uri", help="Optional MLflow tracking URI.")
    parser.add_argument("--registry-uri", help="Optional MLflow registry URI.")

    # Overrides when not using --payload-json
    parser.add_argument("--funding-level", type=float,
                        default=DEFAULT_ROW["funding_level"])
    parser.add_argument("--dimension", default=DEFAULT_ROW["dimension"])
    parser.add_argument("--start-year", type=int,
                        default=DEFAULT_ROW["start_year"])
    parser.add_argument("--horizon", type=int, default=DEFAULT_ROW["horizon"])
    parser.add_argument("--overflow-tiers",
                        default=DEFAULT_ROW["overflow_tiers"])
    parser.add_argument("--optimizer", default=DEFAULT_ROW["optimizer"])
    parser.add_argument("--time-limit", type=float,
                        default=DEFAULT_ROW["time_limit"])
    parser.add_argument("--workers", type=int, default=DEFAULT_ROW["workers"])
    parser.add_argument("--costs-path", default=DEFAULT_ROW["costs_path"])
    parser.add_argument("--benefits-path",
                        default=DEFAULT_ROW["benefits_path"])
    parser.add_argument("--output-dir", default=DEFAULT_ROW["output_dir"])
    parser.add_argument("--generate-only", action="store_true",
                        default=DEFAULT_ROW["generate_only"])
    parser.add_argument("--relax", action="store_true",
                        default=DEFAULT_ROW["relax"])

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    if args.registry_uri:
        mlflow.set_registry_uri(args.registry_uri)

    uri = _resolve_model_uri(args.run_id, args.model_uri)

    if args.payload_json:
        df = _load_dataframe_from_payload(args.payload_json)
    else:
        df = pd.DataFrame(
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

    model = mlflow.pyfunc.load_model(uri)
    prediction = model.predict(df)

    print("Model URI:", uri)
    print("Input rows:", len(df))
    print("Prediction:")
    print(prediction.to_json(orient="records", indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
