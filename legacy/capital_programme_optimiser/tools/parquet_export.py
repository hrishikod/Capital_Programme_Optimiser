"""Utilities for exporting scenario pickles to a single parquet file."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from capital_programme_optimiser.dashboard import data as dashboard_data

MILLION = 1_000_000.0

MBCM_DISCOUNT_RATE_FIRST = 0.02
MBCM_DISCOUNT_RATE_LATER = 0.015
MBCM_DISCOUNT_FIRST_YEARS = 30

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ATTRIBUTE_WORKBOOK = REPO_ROOT / "Cost_benefit_streams.xlsx"


@dataclass
class ExportSummary:
    rows: int
    row_counts: Dict[str, int]
    output_path: Path


@contextmanager
def _benefit_scale_override(overrides: Dict[str, float]) -> Iterable[None]:
    original = dict(getattr(dashboard_data, "_BENEFIT_VALUE_SCALE_OVERRIDES", {}) or {})
    dashboard_data._BENEFIT_VALUE_SCALE_OVERRIDES = dict(overrides)
    try:
        yield
    finally:
        dashboard_data._BENEFIT_VALUE_SCALE_OVERRIDES = original


def _mbcm_discount_divisors(year_offsets: np.ndarray) -> np.ndarray:
    offsets = np.asarray(year_offsets, dtype=int)
    first_years = np.minimum(offsets, MBCM_DISCOUNT_FIRST_YEARS)
    later_years = np.maximum(offsets - MBCM_DISCOUNT_FIRST_YEARS, 0)
    divisors = np.power(1.0 + MBCM_DISCOUNT_RATE_FIRST, first_years) * np.power(
        1.0 + MBCM_DISCOUNT_RATE_LATER, later_years
    )
    divisors[divisors == 0] = 1.0
    return divisors


def _compute_bcr_series(
    years: np.ndarray,
    spend: np.ndarray,
    benefit: np.ndarray,
    *,
    start_year: int,
    horizon_years: Optional[int],
) -> Tuple[np.ndarray, float, float, float]:
    year_values = np.asarray(years, dtype=int)
    spend_vals = np.asarray(spend, dtype=float)
    benefit_vals = np.asarray(benefit, dtype=float)
    if horizon_years:
        limit = int(start_year) + int(horizon_years) - 1
        mask = year_values <= limit
        spend_vals = np.where(mask, spend_vals, 0.0)
        benefit_vals = np.where(mask, benefit_vals, 0.0)
    offsets = np.maximum(year_values - int(start_year), 0)
    multipliers = np.divide(
        1.0,
        _mbcm_discount_divisors(offsets),
        out=np.ones_like(spend_vals, dtype=float),
    )
    spend_pv = spend_vals * multipliers
    benefit_pv = benefit_vals * multipliers
    cum_spend = np.cumsum(spend_pv)
    cum_benefit = np.cumsum(benefit_pv)
    bcr = np.divide(
        cum_benefit,
        cum_spend,
        out=np.full_like(cum_benefit, np.nan, dtype=float),
        where=cum_spend != 0,
    )
    spend_total = float(spend_pv.sum())
    benefit_total = float(benefit_pv.sum())
    bcr_total = benefit_total / spend_total if spend_total else float("nan")
    return bcr, spend_total, benefit_total, bcr_total


def _normalise_project_key(name: str) -> str:
    if name is None:
        return ""
    normalised = str(name).strip().lower()
    return " ".join(normalised.split())


def _clean_mapping_value(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _canonicalise_gps_tier(value: object) -> str:
    text = _clean_mapping_value(value)
    if text is None:
        return "Unknown"
    normalised = " ".join(str(text).replace("_", " ").replace("-", " ").split()).lower()
    if not normalised or normalised == "unknown":
        return "Unknown"
    if "must" in normalised:
        return "Must do"
    if "should" in normalised:
        return "Should do"
    if "could" in normalised:
        return "Could do"
    return str(text).strip()


def _read_excel_safe(path: Path, *, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        return pd.DataFrame()


def _load_project_attributes(workbook_path: Optional[Path]) -> pd.DataFrame:
    if workbook_path is None:
        return pd.DataFrame(
            columns=[
                "project_key",
                "project",
                "region",
                "activity_class",
                "gps_request_tier",
                "strategic_dimension",
            ]
        )
    if not workbook_path.exists():
        return pd.DataFrame(
            columns=[
                "project_key",
                "project",
                "region",
                "activity_class",
                "gps_request_tier",
                "strategic_dimension",
            ]
        )

    def _read_sheet(sheet: str, column_map: Dict[str, str]) -> pd.DataFrame:
        df = _read_excel_safe(workbook_path, sheet_name=sheet)
        if df.empty:
            return pd.DataFrame(columns=["ProjectKey", "Project", *column_map.keys()])
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]
        lookup = {str(col).strip().lower(): col for col in df.columns}
        out = pd.DataFrame()
        project_col = lookup.get("project")
        if project_col is None:
            return pd.DataFrame(columns=["ProjectKey", "Project", *column_map.keys()])
        out["Project"] = df[project_col].map(_clean_mapping_value)
        out = out.dropna(subset=["Project"]).copy()
        out["Project"] = out["Project"].astype(str).str.strip()
        out = out[out["Project"].ne("")]
        out["ProjectKey"] = out["Project"].map(_normalise_project_key)
        for out_col, src_label in column_map.items():
            src_col = lookup.get(src_label.lower())
            out[out_col] = df[src_col].map(_clean_mapping_value) if src_col is not None else None
        out = out.drop_duplicates(subset=["ProjectKey"], keep="first")
        return out[["ProjectKey", "Project", *column_map.keys()]]

    costs = _read_sheet(
        "Costs",
        {
            "Region": "Region",
            "ActivityClass": "Activity Class",
            "GPSTier": "GPS Request Tier",
            "StrategicDimension": "Strategic Dimension",
        },
    )
    benefits = _read_sheet(
        "Benefits Linear 40yrs",
        {
            "Region": "Region",
            "ActivityClass": "Activity Class",
            "StrategicDimension": "Strategic Dimension",
        },
    )

    if costs.empty and benefits.empty:
        return pd.DataFrame(
            columns=[
                "project_key",
                "project",
                "region",
                "activity_class",
                "gps_request_tier",
                "strategic_dimension",
            ]
        )

    merged = costs.set_index("ProjectKey").combine_first(benefits.set_index("ProjectKey")).reset_index()
    for col in ("Project", "Region", "ActivityClass", "GPSTier", "StrategicDimension"):
        if col not in merged.columns:
            merged[col] = None
    merged = merged[["ProjectKey", "Project", "Region", "ActivityClass", "GPSTier", "StrategicDimension"]].copy()
    merged["Region"] = merged["Region"].fillna("Unknown")
    merged["ActivityClass"] = merged["ActivityClass"].fillna("Unknown")
    merged["GPSTier"] = merged["GPSTier"].fillna("Unknown").map(_canonicalise_gps_tier)
    merged["StrategicDimension"] = merged["StrategicDimension"].fillna("Unknown")

    merged = merged.rename(
        columns={
            "ProjectKey": "project_key",
            "Project": "project",
            "Region": "region",
            "ActivityClass": "activity_class",
            "GPSTier": "gps_request_tier",
            "StrategicDimension": "strategic_dimension",
        }
    )
    return merged


def _scenario_meta_table(results: Dict[str, Dict[str, object]], scenarios: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "Code": "scenario_code",
        "ScenarioTitle": "scenario_title",
        "Conf": "scenario_conf",
        "BenSteep": "scenario_ben_steep",
        "BenHorizon": "scenario_ben_horizon",
        "BenLevel": "scenario_ben_level",
        "Mode": "scenario_mode",
        "Envelope": "scenario_envelope_m",
        "EnvelopeFull": "scenario_envelope_full_m",
        "Buffer": "scenario_buffer_m",
        "CashPlus": "scenario_cash_plus_m",
        "ObjectiveDim": "scenario_objective_dim",
        "ObjectiveDimShort": "scenario_objective_dim_short",
        "Profile": "scenario_profile",
        "StartFY": "scenario_start_fy",
        "HorizonYears": "scenario_horizon_years",
        "BenRate": "scenario_ben_rate",
        "Gap": "scenario_gap",
        "IsComp": "scenario_is_comp",
        "CacheStem": "scenario_cache_stem",
        "CacheFile": "scenario_cache_file",
        "OrigStem": "scenario_orig_stem",
    }
    meta = scenarios.rename(columns=rename).copy()
    meta = meta[[col for col in rename.values() if col in meta.columns]]

    status_lookup = {stem: res.get("status") for stem, res in results.items()}
    created_lookup = {stem: res.get("created_at") for stem, res in results.items()}
    name_lookup = {stem: res.get("scenario") for stem, res in results.items()}
    meta["scenario_status"] = meta["scenario_orig_stem"].map(status_lookup)
    meta["scenario_created_at"] = meta["scenario_orig_stem"].map(created_lookup)
    meta["scenario_name"] = meta["scenario_orig_stem"].map(name_lookup)
    return meta


def _scenario_year_table(
    meta: pd.DataFrame, cf: pd.DataFrame, benefit: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cf_norm = cf.rename(
        columns={
            "Code": "scenario_code",
            "Year": "year",
            "Spend": "spend_total_nzd",
            "Envelope": "envelope_nzd",
            "ClosingNet": "closing_net_nzd",
        }
    ).copy()
    cf_norm = cf_norm.drop(columns=[c for c in cf_norm.columns if str(c).lower() == "key"], errors="ignore")
    for col in ("spend_total_nzd", "envelope_nzd", "closing_net_nzd"):
        cf_norm[col] = pd.to_numeric(cf_norm[col], errors="coerce") * MILLION

    ben_norm = benefit.rename(
        columns={"Code": "scenario_code", "Year": "year", "BenefitFlow": "benefit_total_nzd"}
    ).copy()
    ben_norm = ben_norm.drop(columns=[c for c in ben_norm.columns if str(c).lower() == "key"], errors="ignore")
    ben_norm["benefit_total_nzd"] = pd.to_numeric(ben_norm["benefit_total_nzd"], errors="coerce")

    merged = cf_norm.merge(ben_norm, on=["scenario_code", "year"], how="outer")
    merged["record_type"] = "scenario_year"

    merged = merged.merge(meta, on="scenario_code", how="left")
    merged = merged.drop(columns=[c for c in merged.columns if str(c).lower().startswith("key")], errors="ignore")

    bcr_rows: List[pd.DataFrame] = []
    pv_rows: List[Dict[str, object]] = []
    meta_lookup = meta.set_index("scenario_code")

    for scenario_code, group in merged.groupby("scenario_code"):
        years = pd.to_numeric(group["year"], errors="coerce")
        valid_mask = years.notna()
        if not valid_mask.any():
            continue
        years = years[valid_mask].astype(int).to_numpy()
        spend = pd.to_numeric(group.loc[valid_mask, "spend_total_nzd"], errors="coerce").fillna(0.0)
        benefit_vals = pd.to_numeric(
            group.loc[valid_mask, "benefit_total_nzd"], errors="coerce"
        ).fillna(0.0)
        spend = spend.to_numpy(dtype=float)
        benefit_vals = benefit_vals.to_numpy(dtype=float)

        start_year = None
        horizon_years = None
        if scenario_code in meta_lookup.index:
            meta_row = meta_lookup.loc[scenario_code]
            try:
                start_year = int(meta_row.get("scenario_start_fy"))
            except (TypeError, ValueError):
                start_year = None
            try:
                horizon_years = int(meta_row.get("scenario_horizon_years"))
            except (TypeError, ValueError):
                horizon_years = None
        if start_year is None and len(years):
            start_year = int(years.min())

        bcr, spend_pv_total, benefit_pv_total, bcr_pv_total = _compute_bcr_series(
            years,
            spend,
            benefit_vals,
            start_year=int(start_year),
            horizon_years=horizon_years,
        )

        bcr_rows.append(
            pd.DataFrame(
                {
                    "scenario_code": scenario_code,
                    "year": years,
                    "bcr_pv": bcr,
                }
            )
        )
        pv_rows.append(
            {
                "scenario_code": scenario_code,
                "spend_pv_total_nzd": spend_pv_total,
                "benefit_pv_total_nzd": benefit_pv_total,
                "bcr_pv_total": bcr_pv_total,
            }
        )

    bcr_table = pd.concat(bcr_rows, ignore_index=True) if bcr_rows else pd.DataFrame()
    if not bcr_table.empty:
        merged = merged.merge(bcr_table, on=["scenario_code", "year"], how="left")
    pv_table = pd.DataFrame(pv_rows)
    return merged, pv_table


def _project_year_table(
    meta: pd.DataFrame,
    spend_matrix: pd.DataFrame,
    min_value: float,
    project_attrs: pd.DataFrame,
) -> pd.DataFrame:
    if spend_matrix.empty:
        return pd.DataFrame()
    spend = spend_matrix.copy()
    year_cols = [c for c in spend.columns if str(c).isdigit()]
    if not year_cols:
        year_cols = [c for c in spend.columns if isinstance(c, (int, np.integer))]
    spend = spend.rename(columns={"Code": "scenario_code", "Project": "project"})
    spend_long = spend.melt(
        id_vars=["scenario_code", "project"],
        value_vars=year_cols,
        var_name="year",
        value_name="spend_project_nzd",
    )
    spend_long["spend_project_nzd"] = (
        pd.to_numeric(spend_long["spend_project_nzd"], errors="coerce") * MILLION
    )
    spend_long["record_type"] = "project_year"
    spend_long = spend_long[spend_long["spend_project_nzd"].abs() > min_value].copy()
    spend_long["project_key"] = spend_long["project"].map(_normalise_project_key)
    if not project_attrs.empty:
        attr_index = project_attrs.set_index("project_key")
        spend_long["region"] = spend_long["project_key"].map(attr_index["region"])
        spend_long["activity_class"] = spend_long["project_key"].map(attr_index["activity_class"])
        spend_long["gps_request_tier"] = spend_long["project_key"].map(attr_index["gps_request_tier"])
        spend_long["strategic_dimension"] = spend_long["project_key"].map(attr_index["strategic_dimension"])
    spend_long = spend_long.merge(meta, on="scenario_code", how="left")
    return spend_long


def _project_schedule_table(meta: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame()
    sched = schedule.rename(
        columns={
            "Code": "scenario_code",
            "Project": "project",
            "StartFY": "start_fy",
            "EndFY": "end_fy",
            "Dur": "duration",
        }
    ).copy()
    sched["record_type"] = "project_schedule"
    sched = sched.merge(meta, on="scenario_code", how="left")
    return sched


def _project_dimension_year_table(
    meta: pd.DataFrame,
    results: Dict[str, Dict[str, object]],
    min_value: float,
    project_attrs: pd.DataFrame,
) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    stem_to_code = meta.set_index("scenario_orig_stem")["scenario_code"].to_dict()
    frames: List[pd.DataFrame] = []
    for stem, res in results.items():
        code = stem_to_code.get(stem)
        if not code:
            continue
        df = res.get("benefits_by_project_dimension_by_year")
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        working = df.copy()
        if isinstance(working.index, pd.MultiIndex):
            working = working.reset_index()
        if "Project" not in working.columns:
            continue
        if "Dimension" not in working.columns:
            continue
        year_cols = [c for c in working.columns if str(c).isdigit()]
        if not year_cols:
            year_cols = [c for c in working.columns if isinstance(c, (int, np.integer))]
        if not year_cols:
            continue
        melted = working.melt(
            id_vars=["Project", "Dimension"],
            value_vars=year_cols,
            var_name="year",
            value_name="benefit_flow_nzd",
        )
        melted["benefit_flow_nzd"] = pd.to_numeric(melted["benefit_flow_nzd"], errors="coerce")
        melted = melted[melted["benefit_flow_nzd"].abs() > min_value].copy()
        if melted.empty:
            continue
        melted["dimension"] = melted["Dimension"].apply(dashboard_data._canonicalise_dimension_label)
        melted["project"] = melted["Project"].astype(str)
        melted["project_key"] = melted["project"].map(_normalise_project_key)
        if not project_attrs.empty:
            attr_index = project_attrs.set_index("project_key")
            melted["region"] = melted["project_key"].map(attr_index["region"])
            melted["activity_class"] = melted["project_key"].map(attr_index["activity_class"])
            melted["gps_request_tier"] = melted["project_key"].map(attr_index["gps_request_tier"])
            melted["strategic_dimension"] = melted["project_key"].map(attr_index["strategic_dimension"])
        melted["scenario_code"] = code
        melted["record_type"] = "project_dimension_year"
        keep_cols = [
            "record_type",
            "scenario_code",
            "project",
            "project_key",
            "dimension",
            "year",
            "benefit_flow_nzd",
            "region",
            "activity_class",
            "gps_request_tier",
            "strategic_dimension",
        ]
        frames.append(melted[[col for col in keep_cols if col in melted.columns]])
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.merge(meta, on="scenario_code", how="left")
    return combined


def build_gps27_parquet(
    scenario_dir: Path,
    output_path: Path,
    *,
    min_value: float = 1e-6,
    attribute_workbook: Optional[Path] = None,
) -> ExportSummary:
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")
    results = dashboard_data.load_results(scenario_dir)
    with _benefit_scale_override({}):
        data = dashboard_data.prepare_dashboard_data(results)

    meta = _scenario_meta_table(results, data.scenarios)
    scenario_year, pv_totals = _scenario_year_table(meta, data.cf, data.benefit)
    scenario_rows = meta.copy()
    scenario_rows["record_type"] = "scenario"
    if not pv_totals.empty:
        scenario_rows = scenario_rows.merge(pv_totals, on="scenario_code", how="left")

    workbook_path = attribute_workbook or DEFAULT_ATTRIBUTE_WORKBOOK
    project_attrs = _load_project_attributes(workbook_path)

    project_year = _project_year_table(meta, data.spend_matrix, min_value, project_attrs)
    project_schedule = _project_schedule_table(meta, data.schedule)
    project_dim_year = _project_dimension_year_table(meta, results, min_value, project_attrs)

    frames = [scenario_rows, scenario_year, project_year, project_dim_year, project_schedule]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("No data extracted from scenario results.")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    for col in ("year", "scenario_start_fy", "scenario_horizon_years", "start_fy", "end_fy", "duration"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("Int64")
    for col in (
        "spend_project_nzd",
        "spend_total_nzd",
        "benefit_flow_nzd",
        "benefit_total_nzd",
        "envelope_nzd",
        "closing_net_nzd",
        "bcr_pv",
        "spend_pv_total_nzd",
        "benefit_pv_total_nzd",
        "bcr_pv_total",
    ):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    output_path = output_path.with_suffix(".parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False, compression="snappy", engine="pyarrow")

    row_counts = combined["record_type"].value_counts(dropna=False).to_dict()
    return ExportSummary(rows=int(len(combined)), row_counts=row_counts, output_path=output_path)
