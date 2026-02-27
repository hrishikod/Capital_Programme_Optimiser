"""Utilities for exporting scenario pickles to a single parquet file."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from capital_programme_optimiser.dashboard import data as dashboard_data
from capital_programme_optimiser.dashboard import regions as dashboard_regions

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
    boundaries_geojson_path: Optional[Path] = None


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


def _region_scenario_meta(meta: pd.DataFrame) -> pd.DataFrame:
    if meta.empty:
        return pd.DataFrame(columns=["scenario_code"])
    out = meta.copy()
    title_series = out.get("scenario_title")
    name_series = out.get("scenario_name")
    if not isinstance(title_series, pd.Series):
        title_series = pd.Series([None] * len(out), index=out.index, dtype="object")
    if not isinstance(name_series, pd.Series):
        name_series = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["scenario_title_final"] = title_series.where(
        title_series.notna() & title_series.astype(str).str.strip().ne(""),
        name_series.where(name_series.notna() & name_series.astype(str).str.strip().ne(""), out["scenario_code"]),
    )
    preferred_cols = [
        "scenario_code",
        "scenario_title_final",
        "scenario_title",
        "scenario_name",
        "scenario_conf",
        "scenario_ben_steep",
        "scenario_ben_horizon",
        "scenario_ben_level",
        "scenario_mode",
        "scenario_profile",
        "scenario_start_fy",
        "scenario_horizon_years",
        "scenario_ben_rate",
        "scenario_status",
    ]
    cols = [col for col in preferred_cols if col in out.columns]
    return out[cols].drop_duplicates(subset=["scenario_code"], keep="first")


def _region_metrics_table(
    scenario_meta: pd.DataFrame,
    data: dashboard_data.DashboardData,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    if scenario_meta.empty:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for scenario_code in scenario_meta["scenario_code"].dropna().astype(str).unique():
        try:
            region_df = dashboard_regions.compute_region_metrics(data, scenario_code, mapping=mapping_df)
        except Exception:
            continue
        if not isinstance(region_df, pd.DataFrame) or region_df.empty:
            continue
        region_df = region_df.copy()
        region_df["scenario_code"] = scenario_code
        frames.append(region_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.rename(
        columns={
            "Year": "year",
            "Spend_Year": "spend_year_nzd",
            "Spend_National": "spend_national_nzd",
            "Spend_Cum_Region": "spend_cum_region_nzd",
            "Spend_Cum_National": "spend_cum_national_nzd",
            "Share_Year": "share_year",
            "Share_Cum": "share_cum",
            "PerCap_Year": "spend_per_cap_year_nzd",
            "PerCap_Cum": "spend_per_cap_cum_nzd",
            "Pop_Share_Benchmark": "pop_share_benchmark",
            "GDP_Share_Benchmark": "gdp_share_benchmark",
            "OU_vs_Pop": "ou_vs_pop",
            "OU_vs_GDP": "ou_vs_gdp",
            "Ramp_Rate": "ramp_rate",
            "Benefit_Year": "benefit_year_nzd",
            "Benefit_National": "benefit_national_nzd",
            "Benefit_Cum_Region": "benefit_cum_region_nzd",
            "Benefit_Cum_National": "benefit_cum_national_nzd",
            "BenefitShare_Year": "benefit_share_year",
            "BenefitShare_Cum": "benefit_share_cum",
        }
    )
    combined["join_key_norm"] = combined["join_key"].map(dashboard_regions._normalise_region_label)

    spend_cols = [
        "spend_year_nzd",
        "spend_national_nzd",
        "spend_cum_region_nzd",
        "spend_cum_national_nzd",
        "spend_per_cap_year_nzd",
        "spend_per_cap_cum_nzd",
    ]
    for col in spend_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce") * MILLION

    numeric_cols = [
        "benefit_year_nzd",
        "benefit_national_nzd",
        "benefit_cum_region_nzd",
        "benefit_cum_national_nzd",
        "benefit_share_year",
        "benefit_share_cum",
        "share_year",
        "share_cum",
        "pop_share_benchmark",
        "gdp_share_benchmark",
        "ou_vs_pop",
        "ou_vs_gdp",
        "ramp_rate",
        "population",
        "gdp_per_capita",
    ]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined["record_type"] = "region_year"
    combined = combined.merge(scenario_meta, on="scenario_code", how="left")
    return combined


def _region_dimension_table(mapping_df: pd.DataFrame) -> pd.DataFrame:
    catalog, pop_share_map, gdp_share_map = dashboard_regions.region_baselines(mapping_df)
    if catalog.empty:
        return pd.DataFrame()

    region_dim = catalog.rename(
        columns={
            "population": "population_base",
            "gdp_per_capita": "gdp_per_capita_base",
        }
    ).copy()
    region_dim["pop_share_benchmark"] = region_dim["region"].map(pop_share_map)
    region_dim["gdp_share_benchmark"] = region_dim["region"].map(gdp_share_map)

    weights = {
        region: float(weight)
        for region, weight in dashboard_regions.NATIONAL_PROJECT_REGION_WEIGHTS.items()
        if region in dashboard_regions.DISPLAY_REGION_SET and float(weight) > 0.0
    }
    weight_total = float(sum(weights.values()))
    if weight_total > 0 and abs(weight_total - 1.0) > 1e-8:
        weights = {region: value / weight_total for region, value in weights.items()}

    region_dim["national_allocation_weight"] = region_dim["region"].map(weights).fillna(0.0)
    region_dim["is_national_allocation_target"] = region_dim["national_allocation_weight"] > 0
    region_dim["record_type"] = "region_dimension"
    return region_dim


def _iter_lonlat_pairs(node: Any) -> Iterator[Tuple[float, float]]:
    if isinstance(node, (list, tuple)):
        if (
            len(node) >= 2
            and isinstance(node[0], (int, float))
            and isinstance(node[1], (int, float))
        ):
            yield float(node[0]), float(node[1])
            return
        for child in node:
            yield from _iter_lonlat_pairs(child)


def _geometry_bbox(geometry: Dict[str, Any]) -> Tuple[float, float, float, float, int]:
    coords = geometry.get("coordinates")
    pairs = list(_iter_lonlat_pairs(coords))
    if not pairs:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0)
    lons = [pair[0] for pair in pairs]
    lats = [pair[1] for pair in pairs]
    return min(lons), min(lats), max(lons), max(lats), len(pairs)


def _region_boundary_table() -> pd.DataFrame:
    geojson = dashboard_regions.fetch_region_geojson()
    features = geojson.get("features", []) if isinstance(geojson, dict) else []
    if not isinstance(features, list) or not features:
        return pd.DataFrame()

    name_field = dashboard_regions.get_geojson_name_field(geojson)
    rows: List[Dict[str, Any]] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties")
        geometry = feature.get("geometry")
        if not isinstance(properties, dict) or not isinstance(geometry, dict):
            continue

        raw_region = properties.get(name_field)
        canonical_region = dashboard_regions._canonical_region_name(raw_region)
        if not canonical_region:
            continue

        bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat, point_count = _geometry_bbox(geometry)
        centroid_lon = (bbox_min_lon + bbox_max_lon) / 2.0 if point_count else float("nan")
        centroid_lat = (bbox_min_lat + bbox_max_lat) / 2.0 if point_count else float("nan")
        geometry_wkt: Optional[str] = None

        if getattr(dashboard_regions, "_HAS_SHAPELY", False):
            shape_fn = getattr(dashboard_regions, "shape", None)
            if callable(shape_fn):
                try:
                    shape_obj = shape_fn(geometry)
                    geometry_wkt = shape_obj.wkt
                    representative_point = shape_obj.representative_point()
                    centroid_lon = float(representative_point.x)
                    centroid_lat = float(representative_point.y)
                except Exception:
                    geometry_wkt = None

        rows.append(
            {
                "record_type": "region_boundary",
                "region": canonical_region,
                "join_key": dashboard_regions._canonical_join_key(canonical_region),
                "join_key_norm": dashboard_regions._normalise_region_label(canonical_region),
                "geojson_name_field": str(name_field),
                "geojson_name_value": str(raw_region).strip() if raw_region is not None else "",
                "geometry_type": geometry.get("type"),
                "geometry_geojson": json.dumps(geometry, ensure_ascii=False),
                "geometry_wkt": geometry_wkt,
                "bbox_min_lon": bbox_min_lon,
                "bbox_min_lat": bbox_min_lat,
                "bbox_max_lon": bbox_max_lon,
                "bbox_max_lat": bbox_max_lat,
                "bbox_center_lon": (bbox_min_lon + bbox_max_lon) / 2.0 if point_count else float("nan"),
                "bbox_center_lat": (bbox_min_lat + bbox_max_lat) / 2.0 if point_count else float("nan"),
                "longitude": centroid_lon,
                "latitude": centroid_lat,
                "centroid_longitude": centroid_lon,
                "centroid_latitude": centroid_lat,
                "geometry_point_count": int(point_count),
            }
        )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["region"], keep="first")
    return out


def _build_region_boundary_geojson(
    region_boundary: pd.DataFrame,
    region_dim: pd.DataFrame,
) -> Dict[str, Any]:
    if region_boundary.empty or region_dim.empty:
        return {"type": "FeatureCollection", "features": []}

    boundary = region_boundary.copy()
    if "join_key_norm" not in boundary.columns:
        boundary["join_key_norm"] = boundary["join_key"].map(dashboard_regions._normalise_region_label)
    if "region" in boundary.columns:
        boundary = boundary.loc[boundary["region"] != dashboard_regions.AREA_OUTSIDE_REGION].copy()

    boundary = boundary.dropna(subset=["join_key_norm", "geometry_geojson"])
    boundary["join_key_norm"] = boundary["join_key_norm"].map(dashboard_regions._normalise_region_label)
    boundary_lookup = (
        boundary.drop_duplicates(subset=["join_key_norm"], keep="first")
        .set_index("join_key_norm")
        .to_dict("index")
    )

    dim = region_dim.copy()
    if "join_key_norm" not in dim.columns:
        dim["join_key_norm"] = dim["join_key"].map(dashboard_regions._normalise_region_label)
    if "region" in dim.columns:
        dim = dim.loc[dim["region"] != dashboard_regions.AREA_OUTSIDE_REGION].copy()
    dim = dim.dropna(subset=["join_key_norm"]).copy()
    dim["join_key_norm"] = dim["join_key_norm"].map(dashboard_regions._normalise_region_label)
    dim = dim.drop_duplicates(subset=["join_key_norm"], keep="first")

    features: List[Dict[str, Any]] = []
    missing: List[str] = []
    for row in dim.itertuples():
        join_key_norm = str(getattr(row, "join_key_norm", "")).strip()
        if not join_key_norm:
            continue
        boundary_row = boundary_lookup.get(join_key_norm)
        if boundary_row is None:
            missing.append(join_key_norm)
            continue

        geometry_raw = boundary_row.get("geometry_geojson")
        if isinstance(geometry_raw, dict):
            geometry = geometry_raw
        elif isinstance(geometry_raw, str):
            try:
                geometry = json.loads(geometry_raw)
            except json.JSONDecodeError:
                missing.append(join_key_norm)
                continue
        else:
            missing.append(join_key_norm)
            continue

        region_label = getattr(row, "region", None) or boundary_row.get("region") or join_key_norm
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "join_key_norm": join_key_norm,
                    "region": str(region_label),
                },
                "geometry": geometry,
            }
        )

    if missing:
        missing_preview = ", ".join(sorted(set(missing))[:10])
        raise RuntimeError(
            "Region boundary GeoJSON is missing geometry for join_key_norm values: "
            f"{missing_preview}"
        )

    return {"type": "FeatureCollection", "features": features}


def _write_region_boundary_geojson(
    region_boundary: pd.DataFrame,
    region_dim: pd.DataFrame,
    output_path: Path,
) -> Path:
    feature_collection = _build_region_boundary_geojson(region_boundary, region_dim)
    output_path = output_path.with_suffix(".geojson")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(feature_collection, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def _attach_region_coordinates(
    frame: pd.DataFrame,
    boundary: pd.DataFrame,
    *,
    region_column: str,
    join_key_norm_column: Optional[str] = None,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
) -> pd.DataFrame:
    if frame.empty or boundary.empty:
        return frame
    if region_column not in frame.columns and (join_key_norm_column is None or join_key_norm_column not in frame.columns):
        return frame

    lookup = (
        boundary[
            [
                "join_key_norm",
                "latitude",
                "longitude",
                "centroid_latitude",
                "centroid_longitude",
            ]
        ]
        .dropna(subset=["join_key_norm"])
        .drop_duplicates(subset=["join_key_norm"], keep="first")
    )
    lat_map = lookup.set_index("join_key_norm")["latitude"].to_dict()
    lon_map = lookup.set_index("join_key_norm")["longitude"].to_dict()

    out = frame.copy()
    if join_key_norm_column is not None and join_key_norm_column in out.columns:
        key_series = out[join_key_norm_column].map(dashboard_regions._normalise_region_label)
    else:
        key_series = out[region_column].map(dashboard_regions._normalise_region_label)
    out[latitude_column] = key_series.map(lat_map)
    out[longitude_column] = key_series.map(lon_map)
    return out


def _project_region_resolved_table(mapping_df: pd.DataFrame, region_dim: pd.DataFrame) -> pd.DataFrame:
    if mapping_df.empty:
        return pd.DataFrame()

    source = mapping_df.copy()
    source["project"] = source["project"].astype(str).str.strip()
    source["project_key"] = source["project"].map(_normalise_project_key)
    source["source_region"] = source["region"].astype(str).str.strip()
    source["source_join_key"] = source["join_key"].astype(str).str.strip()
    source["source_join_key_norm"] = source["source_join_key"].map(dashboard_regions._normalise_region_label)
    source["source_population"] = pd.to_numeric(source["population"], errors="coerce")
    source["source_gdp_per_capita"] = pd.to_numeric(source["gdp_per_capita"], errors="coerce")
    source["source_is_national"] = source["source_region"].map(dashboard_regions._is_national_region_label)
    source["source_is_unmapped"] = source["source_region"].map(dashboard_regions._is_unmapped_region_label)
    source["source_needs_allocation"] = source["source_is_national"] | source["source_is_unmapped"]

    direct = source.loc[~source["source_needs_allocation"]].copy()
    if not direct.empty:
        direct["resolved_region"] = direct["source_region"].apply(
            lambda value: dashboard_regions._canonical_region_name(value) or str(value).strip()
        )
        direct["allocation_method"] = "direct"
        direct["allocation_weight"] = 1.0

    allocated = source.loc[source["source_needs_allocation"]].copy()
    if not allocated.empty:
        weights = {
            region: float(weight)
            for region, weight in dashboard_regions.NATIONAL_PROJECT_REGION_WEIGHTS.items()
            if region in dashboard_regions.DISPLAY_REGION_SET and float(weight) > 0.0
        }
        weight_total = float(sum(weights.values()))
        if weight_total > 0 and abs(weight_total - 1.0) > 1e-8:
            weights = {region: value / weight_total for region, value in weights.items()}
        weight_df = pd.DataFrame(
            {
                "resolved_region": list(weights.keys()),
                "allocation_weight": list(weights.values()),
            }
        )
        allocated["_merge_key"] = 1
        weight_df["_merge_key"] = 1
        allocated = allocated.merge(weight_df, on="_merge_key", how="inner").drop(columns=["_merge_key"])
        allocated["allocation_method"] = "national_weighted_split"

    combined_frames = [frame for frame in (direct, allocated) if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not combined_frames:
        return pd.DataFrame()

    combined = pd.concat(combined_frames, ignore_index=True, sort=False)
    combined["resolved_join_key"] = combined["resolved_region"].map(dashboard_regions._canonical_join_key)
    combined["resolved_join_key_norm"] = combined["resolved_join_key"].map(dashboard_regions._normalise_region_label)

    if not region_dim.empty:
        resolved_lookup = region_dim.rename(
            columns={
                "region": "resolved_region",
                "population_base": "resolved_population_base",
                "gdp_per_capita_base": "resolved_gdp_per_capita_base",
                "pop_share_benchmark": "resolved_pop_share_benchmark",
                "gdp_share_benchmark": "resolved_gdp_share_benchmark",
                "national_allocation_weight": "resolved_national_allocation_weight",
            }
        )[
            [
                "resolved_region",
                "resolved_population_base",
                "resolved_gdp_per_capita_base",
                "resolved_pop_share_benchmark",
                "resolved_gdp_share_benchmark",
                "resolved_national_allocation_weight",
            ]
        ].drop_duplicates(subset=["resolved_region"], keep="first")
        combined = combined.merge(resolved_lookup, on="resolved_region", how="left")

    combined = combined.drop_duplicates(subset=["project_key", "resolved_region"], keep="first")
    combined["record_type"] = "project_region_resolved"
    return combined[
        [
            "record_type",
            "project",
            "project_key",
            "source_region",
            "source_join_key",
            "source_join_key_norm",
            "source_population",
            "source_gdp_per_capita",
            "source_is_national",
            "source_is_unmapped",
            "source_needs_allocation",
            "resolved_region",
            "resolved_join_key",
            "resolved_join_key_norm",
            "allocation_method",
            "allocation_weight",
            "resolved_population_base",
            "resolved_gdp_per_capita_base",
            "resolved_pop_share_benchmark",
            "resolved_gdp_share_benchmark",
            "resolved_national_allocation_weight",
        ]
    ]


def _economic_year_table(region_year: pd.DataFrame, scenario_meta: pd.DataFrame) -> pd.DataFrame:
    if region_year.empty:
        return pd.DataFrame()
    grouped = (
        region_year.groupby(["scenario_code", "year"], as_index=False)
        .agg(
            spend_national_nzd=("spend_national_nzd", "max"),
            spend_cum_national_nzd=("spend_cum_national_nzd", "max"),
            benefit_national_nzd=("benefit_national_nzd", "max"),
            benefit_cum_national_nzd=("benefit_cum_national_nzd", "max"),
            population_national=("population", "sum"),
            spend_total_regions_nzd=("spend_year_nzd", "sum"),
            benefit_total_regions_nzd=("benefit_year_nzd", "sum"),
        )
    )

    pop_values = grouped["population_national"].to_numpy(dtype=float)
    spend_values = grouped["spend_national_nzd"].to_numpy(dtype=float)
    benefit_values = grouped["benefit_national_nzd"].to_numpy(dtype=float)
    grouped["spend_per_cap_national_nzd"] = np.divide(
        spend_values,
        pop_values,
        out=np.full_like(spend_values, np.nan),
        where=pop_values != 0,
    )
    grouped["benefit_per_cap_national_nzd"] = np.divide(
        benefit_values,
        pop_values,
        out=np.full_like(benefit_values, np.nan),
        where=pop_values != 0,
    )

    grouped["benefit_spend_ratio_year"] = np.divide(
        grouped["benefit_national_nzd"],
        grouped["spend_national_nzd"],
        out=np.full(len(grouped), np.nan),
        where=grouped["spend_national_nzd"] != 0,
    )
    grouped["benefit_spend_ratio_cum"] = np.divide(
        grouped["benefit_cum_national_nzd"],
        grouped["spend_cum_national_nzd"],
        out=np.full(len(grouped), np.nan),
        where=grouped["spend_cum_national_nzd"] != 0,
    )
    grouped["record_type"] = "economic_year"
    grouped = grouped.merge(scenario_meta, on="scenario_code", how="left")
    return grouped


def build_gps27_regions_economic_parquet(
    scenario_dir: Path,
    output_path: Path,
    *,
    mapping_path: Optional[Path] = None,
) -> ExportSummary:
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

    results = dashboard_data.load_results(scenario_dir)
    with _benefit_scale_override({}):
        data = dashboard_data.prepare_dashboard_data(results)

    meta = _scenario_meta_table(results, data.scenarios)
    scenario_meta = _region_scenario_meta(meta)
    mapping_df = dashboard_regions.load_region_mapping(mapping_path)

    region_year = _region_metrics_table(scenario_meta, data, mapping_df)
    region_dim = _region_dimension_table(mapping_df)
    region_boundary = _region_boundary_table()
    project_region_resolved = _project_region_resolved_table(mapping_df, region_dim)
    economic_year = _economic_year_table(region_year, scenario_meta)

    region_year = _attach_region_coordinates(
        region_year,
        region_boundary,
        region_column="region",
        join_key_norm_column="join_key_norm",
        latitude_column="latitude",
        longitude_column="longitude",
    )
    region_dim = _attach_region_coordinates(
        region_dim,
        region_boundary,
        region_column="region",
        join_key_norm_column="join_key_norm",
        latitude_column="latitude",
        longitude_column="longitude",
    )
    project_region_resolved = _attach_region_coordinates(
        project_region_resolved,
        region_boundary,
        region_column="resolved_region",
        join_key_norm_column="resolved_join_key_norm",
        latitude_column="resolved_latitude",
        longitude_column="resolved_longitude",
    )

    scenario_rows = scenario_meta.copy()
    scenario_rows["record_type"] = "region_scenario"

    frames = [scenario_rows, region_year, economic_year, region_dim, region_boundary, project_region_resolved]
    frames = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        raise RuntimeError("No regional/economic data extracted from scenario results.")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    for col in ("year", "scenario_start_fy", "scenario_horizon_years"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("Int64")

    numeric_columns = [
        "spend_year_nzd",
        "spend_national_nzd",
        "spend_cum_region_nzd",
        "spend_cum_national_nzd",
        "benefit_year_nzd",
        "benefit_national_nzd",
        "benefit_cum_region_nzd",
        "benefit_cum_national_nzd",
        "share_year",
        "share_cum",
        "spend_per_cap_year_nzd",
        "spend_per_cap_cum_nzd",
        "benefit_per_cap_national_nzd",
        "spend_per_cap_national_nzd",
        "pop_share_benchmark",
        "gdp_share_benchmark",
        "ou_vs_pop",
        "ou_vs_gdp",
        "ramp_rate",
        "benefit_share_year",
        "benefit_share_cum",
        "population",
        "population_base",
        "population_national",
        "gdp_per_capita",
        "gdp_per_capita_base",
        "allocation_weight",
        "national_allocation_weight",
        "resolved_population_base",
        "resolved_gdp_per_capita_base",
        "resolved_pop_share_benchmark",
        "resolved_gdp_share_benchmark",
        "resolved_national_allocation_weight",
        "benefit_spend_ratio_year",
        "benefit_spend_ratio_cum",
        "spend_total_regions_nzd",
        "benefit_total_regions_nzd",
        "bbox_min_lon",
        "bbox_min_lat",
        "bbox_max_lon",
        "bbox_max_lat",
        "bbox_center_lon",
        "bbox_center_lat",
        "longitude",
        "latitude",
        "centroid_longitude",
        "centroid_latitude",
        "geometry_point_count",
        "resolved_latitude",
        "resolved_longitude",
    ]
    for col in numeric_columns:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    output_path = output_path.with_suffix(".parquet")
    boundaries_geojson_path = output_path.with_name("gps27_region_boundaries.geojson")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False, compression="snappy", engine="pyarrow")
    boundaries_geojson_path = _write_region_boundary_geojson(region_boundary, region_dim, boundaries_geojson_path)

    row_counts = combined["record_type"].value_counts(dropna=False).to_dict()
    return ExportSummary(
        rows=int(len(combined)),
        row_counts=row_counts,
        output_path=output_path,
        boundaries_geojson_path=boundaries_geojson_path,
    )
