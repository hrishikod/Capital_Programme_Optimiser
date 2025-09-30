"""Envelope sizing utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

from capital_programme_optimiser.config import load_settings, Settings


@dataclass
class EnvelopeResult:
    projects_kept: int
    projects_total: int
    total_nominal_b: float
    total_nominal_m: float
    years_needed: int
    annual_envelope_b: float
    annual_envelope_m: float
    buffer_multiplier: float
    raw_years: float
    scarcity: float
    concentration: float


def _norm_name(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ")
    return " ".join(text.split())


def load_envelope_selection(path: Optional[Path] = None) -> Dict[str, bool]:
    if path is None:
        path = (Path(__file__).resolve().parent.parent / 'config' / 'envelope_selection.yaml')
    if not path.exists():
        raise FileNotFoundError(f"Envelope selection config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return {_norm_name(name): bool(flag) for name, flag in raw.items()}


def _load_costs(settings: Settings, sheet: Optional[str] = None) -> pd.DataFrame:
    workbook = settings.scoring_workbook()
    sheet_name = sheet or settings.data.costs_sheet
    return pd.read_excel(workbook, sheet_name=sheet_name, engine="openpyxl")


def compute_envelope(include_flags: Dict[str, bool],
                     baseline_total_b: float,
                     baseline_years: int,
                     settings: Optional[Settings] = None,
                     cost_type_match: str = "P95",
                     nominal_match: str = "Nominal",
                     ) -> EnvelopeResult:
    settings = settings or load_settings()
    df = _load_costs(settings)
    df.columns = [str(c).strip() for c in df.columns]

    project_col = next((c for c in df.columns if c.lower() == "project"), None)
    cost_type_col = next((c for c in df.columns if c.lower() == "cost type"), None)
    if project_col is None or cost_type_col is None:
        raise RuntimeError("Expected 'Project' and 'Cost type' columns in Costs sheet.")

    year_cols = sorted([c for c in df.columns if str(c).isdigit()], key=int)
    if not year_cols:
        raise RuntimeError("No numeric year columns found in Costs sheet.")

    ct_series = df[cost_type_col].astype(str)
    mask = ct_series.str.contains(cost_type_match, case=False, regex=False)
    mask &= ct_series.str.contains(nominal_match, case=False, regex=False)
    if not mask.any():
        mask = ct_series.str.contains(cost_type_match, case=False, regex=False)

    filtered = df.loc[mask].copy()
    filtered["Project_norm"] = filtered[project_col].map(_norm_name)

    for year in year_cols:
        filtered[year] = pd.to_numeric(filtered[year], errors="coerce").fillna(0.0)

    totals = filtered.groupby("Project_norm")[year_cols].sum().sum(axis=1)
    totals = totals[totals > 0]

    include_true = {name for name, flag in include_flags.items() if flag}
    exclude_true = {name for name, flag in include_flags.items() if not flag}

    whitelist_mode = len(include_true) > 0
    all_projects = set(totals.index)

    if whitelist_mode:
        kept = all_projects & include_true
    else:
        kept = all_projects - exclude_true

    if not kept:
        raise RuntimeError("No projects selected after applying include/exclude rules.")

    included_series = totals.loc[sorted(kept)]
    included_total_nzd = float(included_series.sum())
    included_total_B = included_total_nzd / 1e9
    included_total_M = included_total_nzd / 1e6

    baseline_n = len(all_projects)
    included_n = len(kept)

    scarcity = max(0.0, (baseline_n - included_n) / max(1.0, baseline_n))
    slack_scarcity = 1.0 + 0.25 * scarcity

    shares = (included_series / included_series.sum()).to_numpy(dtype=float)
    if included_n > 1:
        hhi = float(np.sum(shares ** 2))
        hhi_min = 1.0 / included_n
        concentration_norm = (hhi - hhi_min) / (1.0 - hhi_min)
    else:
        concentration_norm = 1.0
    slack_concentration = 1.0 + 0.15 * concentration_norm

    buffer_mult = min(1.5, slack_scarcity * slack_concentration)

    raw_years = baseline_years * (included_total_B / baseline_total_b)
    years_needed = max(1, int(np.ceil(raw_years * buffer_mult)))

    annual_envelope_B = included_total_B / years_needed
    annual_envelope_M = included_total_M / years_needed

    return EnvelopeResult(
        projects_kept=included_n,
        projects_total=baseline_n,
        total_nominal_b=included_total_B,
        total_nominal_m=included_total_M,
        years_needed=years_needed,
        annual_envelope_b=annual_envelope_B,
        annual_envelope_m=annual_envelope_M,
        buffer_multiplier=buffer_mult,
        raw_years=raw_years,
        scarcity=slack_scarcity,
        concentration=slack_concentration,
    )


def format_envelope_result(result: EnvelopeResult) -> str:
    return (
        "=== P95 Nominal (Included Projects) ===\n"
        f"Projects kept: {result.projects_kept} / {result.projects_total}\n"
        f"Total P95 Nominal: {result.total_nominal_b:,.3f} $B ({result.total_nominal_m:,.0f} M)\n"
        f"Buffer multiplier: {result.buffer_multiplier:,.3f}  (scarcity={result.scarcity:,.3f}, concentration={result.concentration:,.3f})\n"
        f"Raw proportional years: {result.raw_years:,.2f}\n"
        f"→ Years needed (buffered, ceil): {result.years_needed}\n"
        f"→ Annual envelope: {result.annual_envelope_b:,.3f} $B/yr ({result.annual_envelope_m:,.0f} M $/yr)"
    )


def run_default(baseline_total_b: float = 115.0, baseline_years: int = 50) -> EnvelopeResult:
    settings = load_settings()
    flags = load_envelope_selection()
    return compute_envelope(flags, baseline_total_b, baseline_years, settings=settings)


__all__ = [
    "EnvelopeResult",
    "compute_envelope",
    "format_envelope_result",
    "load_envelope_selection",
    "run_default",
]
