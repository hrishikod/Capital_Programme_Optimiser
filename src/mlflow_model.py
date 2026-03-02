from types import SimpleNamespace

import mlflow.pyfunc
import pandas as pd


class OptimizerPyFuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, base_args: dict):
        self.base_args = dict(base_args)

    def _coerce_value(self, key, value):
        base_value = self.base_args.get(key)
        if base_value is None:
            return value
        base_type = type(base_value)
        if isinstance(base_value, bool):
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y"}
            return bool(value)
        try:
            return base_type(value)
        except Exception:
            return value

    def predict(self, context, model_input):
        from src.main import run_optimization

        if isinstance(model_input, pd.DataFrame):
            records = model_input.to_dict(orient="records")
        elif isinstance(model_input, dict):
            records = [model_input]
        else:
            records = [{}]

        outputs = []
        for record in records:
            merged = dict(self.base_args)
            for key, value in record.items():
                if pd.isna(value):
                    continue
                merged[key] = self._coerce_value(key, value)

            args = SimpleNamespace(**merged)
            result, written_outputs = run_optimization(args)

            if result:
                total_spend = float(result.spend_profile.iloc[0, :].sum())
                outputs.append(
                    {
                        "status": result.status,
                        "objective_value": float(result.objective_value),
                        "gap": float(result.gap),
                        "total_spend": total_spend,
                        "schedule_file": written_outputs.get("schedule"),
                        "cash_flow_file": written_outputs.get("cash_flow"),
                        "log_file": written_outputs.get("log_file"),
                    }
                )
            else:
                outputs.append(
                    {
                        "status": "FAILED",
                        "objective_value": None,
                        "gap": None,
                        "total_spend": None,
                        "schedule_file": written_outputs.get("schedule"),
                        "cash_flow_file": written_outputs.get("cash_flow"),
                        "log_file": written_outputs.get("log_file"),
                    }
                )

        return pd.DataFrame(outputs)
