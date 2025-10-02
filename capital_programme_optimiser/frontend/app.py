"""Streamlit front end for exploring optimiser scenarios without Excel."""

from __future__ import annotations

import io
import math
import re
import sys

from dataclasses import dataclass, replace
from datetime import datetime

from pathlib import Path

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import requests

import pandas as pd

import plotly.graph_objects as go
import plotly.colors as plc

if not hasattr(pd.Index, 'clip'):

    def _index_clip(self, lower=None, upper=None):

        arr = self.to_numpy()

        if lower is not None:

            arr = np.maximum(arr, lower)

        if upper is not None:

            arr = np.minimum(arr, upper)

        return pd.Index(arr)

    pd.Index.clip = _index_clip



from plotly.colors import qualitative

import streamlit as st
import streamlit.components.v1 as components

ROOT_CWD = Path.cwd()

ROOT_FILE = Path(__file__).resolve().parents[2]

ROOT_PARENT = ROOT_FILE.parent

for root in {ROOT_CWD, ROOT_FILE, ROOT_PARENT}:

    s = str(root)

    if s not in sys.path:

        sys.path.insert(0, s)

from capital_programme_optimiser.config import Settings, load_settings

from capital_programme_optimiser.dashboard.regions import (
    compute_region_metrics,
    fetch_region_geojson,
    get_geojson_name_field,
    load_region_mapping,
    region_baselines,
)

from capital_programme_optimiser.dashboard.data import (

    DashboardData,

    extract_project_runs,

    find_scenario_code,

    load_results,

    prepare_dashboard_data,

    scenario_metadata,

)

from capital_programme_optimiser.frontend import scenarios as scenario_utils
from capital_programme_optimiser.optimisation import solver_core

PRIMARY_COLOR = "#19456B"

COMPARISON_COLOR = "#C75643"

BAR_OPACITY = 0.75

GANTT_COLOR = "#3A7CA5"

GANTT_OUTLINE_COLOR_BASE = "#98C2DC"
GANTT_OUTLINE_COLOR_ALT = COMPARISON_COLOR
GANTT_OUTLINE_VARIANT_KEY = "gantt_outline_variant"

CLOSING_NET_COLOR = "#F9C80E"

ENVELOPE_COLOR = "#2EC4B6"

CAPACITY_GREEN = "#2E7D32"

CAPACITY_AMBER = "#D18F35"

CAPACITY_RED = "#C44536"

CAPACITY_ZERO = "#94A3B8"

PROJECT_COLOR_POOL = (

    "#704A28",  # warm brown

    "#A2774B",  # muted ochre

    "#B89C73",  # soft tan

    "#6D8C9E",  # slate blue

    "#90AFC0",  # mist blue

    "#C2D3DD",  # pale steel

    "#7B9A88",  # dusty sage

    "#A4C1A3",  # soft green

    "#D2E0C9",  # pale olive

    "#9D8AA5",  # lavender grey

    "#C2B3CF",  # muted lilac

    "#8D6C5A",  # warm taupe

    "#C8A68A",  # sand

    "#DBBC9E",  # pale apricot

    "#8A9DAA",  # blue grey

    "#B0C3D1",  # cool haze

    "#A3B7B3",  # soft teal

    "#CBD8D0",  # frosted mint

    "#B6A899",  # mushroom

    "#D1C3B5",  # linen

    "#9AA3B2",  # muted navy

    "#BFAFBB",  # mauve grey

    "#A19C8D",  # stone

    "#CDBDAA",  # wheat

)


_GANTT_HOTKEY_HTML = """
<script>
(function() {
  const frame = window.frameElement;
  const frameId = frame && frame.id ? frame.id : null;
  if (frame && frame.hasAttribute('data-gantt-hotkey')) {
    return;
  }
  if (frame) {
    frame.setAttribute('data-gantt-hotkey', '1');
  }
  const sendToggle = () => {
    if (!frameId || !window.parent) {
      return;
    }
    window.parent.postMessage({
      type: 'streamlit:setComponentValue',
      id: frameId,
      value: {toggle: Date.now()}
    }, '*');
  };
  const keyHandler = (event) => {
    const key = event.key || event.keyCode;
    if (key === 'z' || key === 'Z' || key === 90 || key === 122) {
      sendToggle();
    }
  };
  const attached = new WeakSet();
  const attach = (target) => {
    if (!target || attached.has(target)) {
      return;
    }
    try {
      target.addEventListener('keydown', keyHandler, true);
      attached.add(target);
    } catch (err) {
      // ignore cross-origin or unsupported targets
    }
  };
  attach(window);
  attach(document);
  if (window.parent && window.parent !== window) {
    attach(window.parent);
    if (window.parent.document) {
      attach(window.parent.document);
    }
  }
})();
</script>
"""


def _inject_gantt_hotkey_listener() -> None:
    if GANTT_OUTLINE_VARIANT_KEY not in st.session_state:
        st.session_state[GANTT_OUTLINE_VARIANT_KEY] = "base"
    response = components.html(_GANTT_HOTKEY_HTML, height=0, width=0)
    if isinstance(response, dict) and response.get("toggle"):
        current = st.session_state.get(GANTT_OUTLINE_VARIANT_KEY, "base")
        st.session_state[GANTT_OUTLINE_VARIANT_KEY] = "alt" if current == "base" else "base"


def _current_gantt_outline_color() -> str:
    variant = st.session_state.get(GANTT_OUTLINE_VARIANT_KEY, "base")
    return GANTT_OUTLINE_COLOR_ALT if variant == "alt" else GANTT_OUTLINE_COLOR_BASE

BRIGHT_PRIMARY_COLOR = "#1976D2"

BRIGHT_COMPARISON_COLOR = "#D81B60"

BENEFIT_SHADE_COLOR = "rgba(25, 118, 210, 0.18)"

WATERFALL_CHART_HEIGHT = 420

PLOTLY_TRANSPARENT_STYLE = """
<style>
div[data-testid="stPlotlyChart"] > div {
    background-color: transparent !important;
}
</style>
"""

CUMULATIVE_OPT_LINE_COLOR = "#2E7D32"
CUMULATIVE_CMP_LINE_COLOR = "#66BB6A"







def is_dark_mode() -> bool:
    """Return True when the Streamlit theme appears to be dark."""
    theme_base = (st.get_option("theme.base") or "").strip().lower()
    if theme_base in {"dark", "light"}:
        return theme_base == "dark"
    background_setting = (st.get_option("theme.backgroundColor") or "").strip()
    luminance = relative_luminance(background_setting) if background_setting else None
    if luminance is not None:
        return luminance < 0.45
    return False


def plotly_template() -> str:
    """Resolve the Plotly template name based on the detected theme."""
    return "plotly_dark" if is_dark_mode() else "plotly_white"

def _hoverlabel_style() -> dict:
    """Consistent hover label styling that respects the active theme."""
    if is_dark_mode():
        return {
            "bgcolor": "rgba(15, 23, 42, 0.92)",
            "bordercolor": "rgba(148, 163, 184, 0.45)",
            "font": dict(color="#E2E8F0", size=12, family="Inter, 'Segoe UI', sans-serif"),
            "namelength": 0,
        }
    return {
        "bgcolor": "rgba(255, 255, 255, 0.96)",
        "bordercolor": "rgba(148, 163, 184, 0.35)",
        "font": dict(color="#0F172A", size=12, family="Inter, 'Segoe UI', sans-serif"),
        "namelength": 0,
    }

def _normalise_project_key(name: str) -> str:
    """Return a canonical project key for matching across scenarios."""
    if name is None:
        return ''
    normalised = str(name).strip().lower()
    return re.sub(r'\s+', ' ', normalised)

def rgba_from_hex(hex_color: str, alpha: float) -> str:
    '''Return an rgba string for a hex colour code or existing rgba string.'''

    colour = hex_color.strip()

    if colour.startswith('rgba'):
        return colour

    if colour.startswith('#'):
        colour = colour[1:]

    if len(colour) == 3:
        colour = ''.join(ch * 2 for ch in colour)

    if len(colour) != 6:
        return hex_color

    try:
        red = int(colour[0:2], 16)
        green = int(colour[2:4], 16)
        blue = int(colour[4:6], 16)
    except ValueError:
        return hex_color

    return f"rgba({red}, {green}, {blue}, {alpha})"



def rgb_components(color: str) -> Optional[Tuple[float, float, float]]:
    """Return RGB components from a hex/rgb/rgba string (0-255 range)."""

    if not color:
        return None

    colour = color.strip()
    if not colour:
        return None

    if colour.startswith('#'):
        hex_color = colour[1:]
        if len(hex_color) == 3:
            hex_color = ''.join(ch * 2 for ch in hex_color)
        if len(hex_color) != 6:
            return None
        try:
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return None

    if colour.startswith('rgb'):
        start = colour.find("(")
        end = colour.find(")", start)
        if start == -1 or end == -1:
            return None
        parts = [segment.strip() for segment in colour[start + 1:end].split(",")]
        if len(parts) < 3:
            return None
        try:
            values = []
            for segment in parts[:3]:
                if segment.endswith('%'):
                    values.append(float(segment[:-1]) * 2.55)
                else:
                    values.append(float(segment))
            return tuple(values)
        except ValueError:
            return None

    return None


def relative_luminance(color: str) -> Optional[float]:
    """Compute relative luminance (0=dark, 1=light) for a colour string."""

    rgb = rgb_components(color)
    if not rgb:
        return None

    def _adjust(component: float) -> float:
        ratio = component / 255.0
        return ratio / 12.92 if ratio <= 0.03928 else ((ratio + 0.055) / 1.055) ** 2.4

    r, g, b = (_adjust(value) for value in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

DEFAULT_PROFILE_LABEL = "Default"

MODE_LABELS = {

    "unconstrained": "Unconstrained",

    "fixed": "Fixed",

    "buffered": "Buffered",

    "cash": "Cash+",

}

@dataclass

class ScenarioSelection:

    """Snapshot of the user-selected scenario inputs."""

    name: str

    code: Optional[str]

    confidence: str

    benefit_steep: str

    benefit_horizon: int

    mode: str

    envelope: Optional[int]

    buffer_value: Optional[int]

    dimension: str

    profile: Optional[str] = None

    metadata: Optional[Dict[str, object]] = None

LOAD_DATA_SCHEMA_VERSION = 2


def _cache_signature(cache_path: Path) -> Tuple[Tuple[str, int, int], ...]:

    """Return a stable signature for the cache directory contents."""

    entries: List[Tuple[str, int, int]] = []

    if not cache_path.exists():

        return tuple()

    for pkl_file in sorted(cache_path.glob("*.pkl")):

        try:

            stat = pkl_file.stat()

        except OSError:

            continue

        entries.append((pkl_file.name, int(stat.st_mtime), int(stat.st_size)))

    return tuple(entries)


@st.cache_resource(show_spinner=False)
def load_dashboard_data(
    cache_dir: str, signature: Tuple[Tuple[str, int, int], ...],
    schema_version: int,
) -> DashboardData:

    """Load dashboard-ready data from the supplied cache directory."""

    cache_path = Path(cache_dir)

    if schema_version != LOAD_DATA_SCHEMA_VERSION:
        raise ValueError('Streamlit cache schema mismatch')

    _ = signature  # consumed by Streamlit cache hashing

    if not cache_path.exists():

        raise FileNotFoundError(f"Cache dir not found: {cache_path}")

    results = load_results(cache_path)

    data = prepare_dashboard_data(results)

    return data

def format_currency(value: float) -> str:

    return f"{value:,.0f} m" if np.isfinite(value) else "-"

def format_large_amount(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}b"
    return f"{value / 1_000_000:.1f}m"

def compute_cash_axis_ticks(values: Iterable[float]) -> Tuple[List[float], List[str], str]:
    finite_values = [val for val in values if np.isfinite(val)]
    if not finite_values:
        return [0.0], ["0"], "millions"

    min_val = min(finite_values)
    max_val = max(finite_values)
    max_abs = max(abs(min_val), abs(max_val))

    unit_value = 1_000_000_000 if max_abs >= 1_000_000_000 else 1_000_000
    unit_suffix = "b" if unit_value == 1_000_000_000 else "m"
    unit_label = "billions" if unit_suffix == "b" else "millions"

    min_index = math.floor(min_val / unit_value)
    max_index = math.ceil(max_val / unit_value)
    approx_ticks = 6
    span_units = max_index - min_index
    if span_units <= 0:
        span_units = 1

    raw_step = span_units / approx_ticks
    if raw_step <= 1:
        step_multiplier = 1
    else:
        magnitude = 10 ** math.floor(math.log10(raw_step))
        step_multiplier = None
        for factor in (1, 2, 5, 10):
            candidate = int(factor * magnitude)
            if candidate < raw_step:
                continue
            step_multiplier = max(1, candidate)
            break
        if step_multiplier is None:
            step_multiplier = int(10 * magnitude)

    tick_indices = range(min_index, max_index + int(step_multiplier), int(step_multiplier))
    tick_vals = [unit_value * idx for idx in tick_indices]
    if 0.0 not in tick_vals:
        tick_vals.append(0.0)
        tick_vals.sort()

    tick_text = []
    for val in tick_vals:
        if math.isclose(val, 0.0, abs_tol=1e-9):
            tick_text.append("0")
        else:
            tick_text.append(f"{val / unit_value:.0f}{unit_suffix}")

    return tick_vals, tick_text, unit_label

def scale_series_to_nzd(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    return series.astype(float) * 1_000_000.0

def series_from_df(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns or "Year" not in df.columns:
        return pd.Series(dtype=float)
    years = pd.to_numeric(df["Year"], errors="coerce")
    values = pd.to_numeric(df[column], errors="coerce")
    mask = years.notna()
    if not mask.any():
        return pd.Series(dtype=float)
    return pd.Series(values[mask].astype(float).to_numpy(), index=years[mask].astype(int).to_numpy())

def sorted_years(*dfs: Optional[pd.DataFrame]) -> List[int]:
    years: Set[int] = set()
    for df in dfs:
        if df is None or df.empty:
            continue
        numeric_years = pd.to_numeric(df.get("Year"), errors="coerce").dropna().astype(int)
        years.update(numeric_years.tolist())
    return sorted(years)



def _create_export_table(years: Iterable[int]) -> Tuple[pd.DataFrame, pd.Index]:
    index = pd.Index([int(year) for year in years], name="Financial year")
    return pd.DataFrame(index=index), index


def pv_to_nominal_index(data: DashboardData, years: Iterable[int]) -> pd.Series:
    """Return discount multipliers to convert PV back to nominal dollars."""
    base_rate = float(getattr(data, "benefit_rate", 0.0) or 0.0)
    start_year = int(getattr(data, "start_fy", 0))
    factor = 1.0 + base_rate
    year_index = pd.Index([int(year) for year in years], name="Financial year")
    offsets = (year_index - start_year).to_numpy()
    multipliers = np.power(factor, np.clip(offsets, a_min=0, a_max=None))
    return pd.Series(multipliers, index=year_index, name="Discount multiplier")

def sanitize_sheet_name(name: str, existing: Set[str]) -> str:
    sanitized = re.sub(r"[\\/*?:\[\]]", "_", name).strip()
    if not sanitized:
        sanitized = "Sheet"
    sanitized = sanitized[:31]
    candidate = sanitized
    suffix = 1
    while candidate.lower() in existing:
        extra = f"_{suffix}"
        candidate = f"{sanitized[:31 - len(extra)]}{extra}"
        suffix += 1
    existing.add(candidate.lower())
    return candidate

def build_export_workbook(tables: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        seen: Set[str] = set()
        for name, df in tables.items():
            sheet_name = sanitize_sheet_name(name, seen)
            (df if df is not None else pd.DataFrame()).to_excel(writer, sheet_name=sheet_name, index=False)
    buffer.seek(0)
    return buffer.getvalue()

def prepare_efficiency_export(
    data: DashboardData,
    opt_df: Optional[pd.DataFrame],
    cmp_df: Optional[pd.DataFrame],
    opt_selection: ScenarioSelection,
    cmp_selection: ScenarioSelection,
) -> Optional[pd.DataFrame]:
    years = sorted_years(opt_df, cmp_df)
    if not years:
        return None
    export, index = _create_export_table(years)
    if opt_df is not None and not opt_df.empty:
        cum_spend = series_from_df(opt_df, "CumSpend")
        if not cum_spend.empty:
            aligned_spend = cum_spend.reindex(index)
            if aligned_spend.notna().any():
                export["Optimised cumulative spend ($)"] = scale_series_to_nzd(aligned_spend)
        series_opt, label_opt = _benefit_series_and_label(opt_df, opt_selection, prefix="Optimised")
        if not series_opt.empty:
            year_values = pd.to_numeric(opt_df.get("Year"), errors="coerce")
            mask = year_values.notna()
            if mask.any():
                aligned = pd.Series(
                    series_opt[mask].astype(float).to_numpy(),
                    index=year_values[mask].astype(int).to_numpy(),
                ).reindex(index)
                if aligned.notna().any():
                    export[f"{label_opt} ($)"] = scale_series_to_nzd(aligned)
    if cmp_df is not None and not cmp_df.empty:
        cum_spend_cmp = series_from_df(cmp_df, "CumSpend")
        if not cum_spend_cmp.empty:
            aligned_cmp_spend = cum_spend_cmp.reindex(index)
            if aligned_cmp_spend.notna().any():
                export["Comparison cumulative spend ($)"] = scale_series_to_nzd(aligned_cmp_spend)
        series_cmp, label_cmp = _benefit_series_and_label(cmp_df, cmp_selection, prefix="Comparison")
        if not series_cmp.empty:
            year_values = pd.to_numeric(cmp_df.get("Year"), errors="coerce")
            mask = year_values.notna()
            if mask.any():
                aligned = pd.Series(
                    series_cmp[mask].astype(float).to_numpy(),
                    index=year_values[mask].astype(int).to_numpy(),
                ).reindex(index)
                if aligned.notna().any():
                    export[f"{label_cmp} ($)"] = scale_series_to_nzd(aligned)
    if export.shape[1] == 0:
        return None
    return export.reset_index()

def prepare_cash_export(
    df: Optional[pd.DataFrame],
    *,
    label_prefix: str,
) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    years = sorted_years(df)
    if not years:
        return None
    export, index = _create_export_table(years)
    column_map = {
        "Spend": "Annual spend ($)",
        "ClosingNet": "Closing net ($)",
        "Envelope": "Envelope ($)",
        "BenefitFlow": "Benefit flow ($)",
        "PVBenefit": "PV benefit ($)",
        "CumSpend": "Cumulative spend ($)",
        "CumBenefit": "Cumulative benefit ($)",
        "CumPVBenefit": "Cumulative PV benefit ($)",
    }
    for source, label in column_map.items():
        if source not in df.columns:
            continue
        series = series_from_df(df, source)
        if series.empty:
            continue
        aligned = series.reindex(index)
        if aligned.notna().any():
            export[f"{label_prefix} {label}"] = scale_series_to_nzd(aligned)
    if export.shape[1] == 0:
        return None
    return export.reset_index()

def prepare_benefit_export(
    opt_df: Optional[pd.DataFrame],
    cmp_df: Optional[pd.DataFrame],
    *,
    opt_label: str,
    cmp_label: str,
) -> Optional[pd.DataFrame]:
    years = sorted_years(opt_df, cmp_df)
    if not years:
        return None
    export, index = _create_export_table(years)
    if opt_df is not None and not opt_df.empty and "PVBenefit" in opt_df.columns:
        series_opt = series_from_df(opt_df, "PVBenefit")
        if not series_opt.empty:
            aligned_opt = series_opt.reindex(index)
            if aligned_opt.notna().any():
                export[f"{opt_label} benefit real ($)"] = scale_series_to_nzd(aligned_opt)
    if cmp_df is not None and not cmp_df.empty and "PVBenefit" in cmp_df.columns:
        series_cmp = series_from_df(cmp_df, "PVBenefit")
        if not series_cmp.empty:
            aligned_cmp = series_cmp.reindex(index)
            if aligned_cmp.notna().any():
                export[f"{cmp_label} benefit real ($)"] = scale_series_to_nzd(aligned_cmp)
    if export.shape[1] == 0:
        return None
    return export.reset_index()

def prepare_benefit_delta_export(
    opt_df: Optional[pd.DataFrame],
    cmp_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if opt_df is None or cmp_df is None or opt_df.empty or cmp_df.empty:
        return None
    merged = opt_df[["Year", "CumBenefit", "CumPVBenefit"]].merge(
        cmp_df[["Year", "CumBenefit", "CumPVBenefit"]],
        on="Year",
        suffixes=("_opt", "_cmp"),
    )
    merged["Year"] = pd.to_numeric(merged["Year"], errors="coerce")
    merged = merged.dropna(subset=["Year"])
    if merged.empty:
        return None
    merged["Year"] = merged["Year"].astype(int)
    export = pd.DataFrame({"Financial year": merged["Year"]})
    export["Delta cumulative benefit real ($)"] = scale_series_to_nzd(
        merged["CumBenefit_opt"] - merged["CumBenefit_cmp"]
    )
    export["Delta benefit real ($)"] = scale_series_to_nzd(
        merged["CumPVBenefit_opt"] - merged["CumPVBenefit_cmp"]
    )
    return export

def prepare_dimension_chart_export(
    pivot: Optional[pd.DataFrame],
    cumulative: bool,
) -> Optional[pd.DataFrame]:
    if pivot is None or pivot.empty:
        return None
    data = pivot.copy()
    data.index = pd.to_numeric(data.index, errors="coerce")
    data = data.dropna()
    if data.empty:
        return None
    data.index = data.index.astype(int)
    if cumulative:
        data = data.cumsum()
    if data.empty:
        return None
    scaled = data.apply(scale_series_to_nzd)
    scaled.insert(0, "Financial year", scaled.index)
    scaled.columns = [scaled.columns[0]] + [f"{col} ($)" for col in scaled.columns[1:]]
    return scaled.reset_index(drop=True)

def prepare_dimension_overlay_export(
    years: Iterable[int],
    pivot_opt: Optional[pd.DataFrame],
    pivot_cmp: Optional[pd.DataFrame],
    dimensions: List[str],
    cumulative: bool,
) -> Optional[pd.DataFrame]:
    selected_dims = [dim for dim in dimensions]
    if not selected_dims:
        return None
    years_list = [int(y) for y in years]
    export = pd.DataFrame({"Financial year": years_list})
    added = False
    if pivot_opt is not None and not pivot_opt.empty:
        opt = pivot_opt.reindex(index=years_list)
        opt.index = pd.Index(years_list, name="Year")
        if cumulative:
            opt = opt.cumsum()
        for dim in selected_dims:
            if dim in opt.columns:
                export[f"Optimised - {dim} ($)"] = scale_series_to_nzd(opt[dim])
                added = True
    if pivot_cmp is not None and not pivot_cmp.empty:
        cmp = pivot_cmp.reindex(index=years_list)
        cmp.index = pd.Index(years_list, name="Year")
        if cumulative:
            cmp = cmp.cumsum()
        for dim in selected_dims:
            if dim in cmp.columns:
                export[f"Comparison - {dim} ($)"] = scale_series_to_nzd(cmp[dim])
                added = True
    if not added:
        return None
    return export

def prepare_waterfall_export(
    data: DashboardData,
    opt_selection: ScenarioSelection,
    cmp_selection: ScenarioSelection,
    *,
    horizon_years: Optional[int] = None,
    pv_opt: Optional[Dict[str, float]] = None,
    pv_cmp: Optional[Dict[str, float]] = None,
) -> Optional[pd.DataFrame]:
    if pv_opt is None:
        pv_opt = pv_by_dimension(data, opt_selection, horizon_years=horizon_years) if opt_selection and opt_selection.code else None
    if pv_cmp is None:
        pv_cmp = pv_by_dimension(data, cmp_selection, horizon_years=horizon_years) if cmp_selection and cmp_selection.code else None
    if not pv_opt or not pv_cmp:
        return None
    dims = set(pv_opt.keys()) | set(pv_cmp.keys())
    ordered_dims = [dim for dim in data.dims if dim in dims]
    remaining_dims = [dim for dim in dims if dim not in ordered_dims]
    non_total_dims = [dim for dim in ordered_dims if str(dim).strip().lower() != "total"]
    non_total_dims += [dim for dim in remaining_dims if str(dim).strip().lower() != "total"]
    rows: List[Dict[str, float]] = []
    for dim in non_total_dims:
        opt_val = float(pv_opt.get(dim, 0.0))
        cmp_val = float(pv_cmp.get(dim, 0.0))
        rows.append(
            {
                "Dimension": str(dim),
                "Optimised NPV ($)": opt_val * 1_000_000.0,
                "Comparison NPV ($)": cmp_val * 1_000_000.0,
                "Delta ($)": (opt_val - cmp_val) * 1_000_000.0,
            }
        )
    total_dim = next((dim for dim in ordered_dims if str(dim).strip().lower() == "total"), None)
    if total_dim is None:
        total_dim = next((dim for dim in dims if str(dim).strip().lower() == "total"), None)
    total_opt = float(pv_opt.get(total_dim, sum(pv_opt.values())))
    total_cmp = float(pv_cmp.get(total_dim, sum(pv_cmp.values())))
    rows.append(
        {
            "Dimension": "Total",
            "Optimised NPV ($)": total_opt * 1_000_000.0,
            "Comparison NPV ($)": total_cmp * 1_000_000.0,
            "Delta ($)": (total_opt - total_cmp) * 1_000_000.0,
        }
    )
    return pd.DataFrame(rows)

def prepare_bridge_export(
    data: DashboardData,
    opt_selection: ScenarioSelection,
    cmp_selection: ScenarioSelection,
    *,
    horizon_years: Optional[int] = None,
    pv_opt: Optional[Dict[str, float]] = None,
    pv_cmp: Optional[Dict[str, float]] = None,
) -> Optional[pd.DataFrame]:
    if pv_opt is None:
        pv_opt = pv_by_dimension(data, opt_selection, horizon_years=horizon_years) if opt_selection and opt_selection.code else None
    if pv_cmp is None:
        pv_cmp = pv_by_dimension(data, cmp_selection, horizon_years=horizon_years) if cmp_selection and cmp_selection.code else None
    if not pv_opt or not pv_cmp:
        return None
    dims = set(pv_opt.keys()) | set(pv_cmp.keys())
    ordered_dims = [dim for dim in data.dims if dim in dims and str(dim).strip().lower() != "total"]
    remaining_dims = sorted(
        [dim for dim in dims if dim not in ordered_dims and str(dim).strip().lower() != "total"],
        key=str,
    )
    dim_sequence = ordered_dims + remaining_dims
    total_dim = next(
        (dim for dim in data.dims if str(dim).strip().lower() == "total" and dim in dims),
        None,
    )
    if total_dim is None:
        total_dim = next((dim for dim in dims if str(dim).strip().lower() == "total"), None)
    total_opt = float(pv_opt.get(total_dim, sum(pv_opt.values())))
    total_cmp = float(pv_cmp.get(total_dim, sum(pv_cmp.values())))
    bridge_diffs = [float(pv_opt.get(dim, 0.0) - pv_cmp.get(dim, 0.0)) for dim in dim_sequence]
    rows = [
        {"Step": "Optimised NPV total", "Value ($)": total_opt * 1_000_000.0, "Measure": "relative"}
    ]
    for dim, delta in zip(dim_sequence, bridge_diffs):
        rows.append(
            {
                "Step": f"{dim} delta NPV",
                "Value ($)": (-delta) * 1_000_000.0,
                "Measure": "relative",
            }
        )
    rows.append(
        {"Step": "Comparison NPV total", "Value ($)": total_cmp * 1_000_000.0, "Measure": "total"}
    )
    return pd.DataFrame(rows)

def prepare_radar_export(
    data: DashboardData,
    opt_selection: ScenarioSelection,
    cmp_selection: ScenarioSelection,
    *,
    pv_opt: Optional[Dict[str, float]] = None,
    pv_cmp: Optional[Dict[str, float]] = None,
) -> Optional[pd.DataFrame]:
    if pv_opt is None:
        pv_opt = pv_by_dimension(data, opt_selection) if opt_selection and opt_selection.code else None
    if pv_cmp is None:
        pv_cmp = pv_by_dimension(data, cmp_selection) if cmp_selection and cmp_selection.code else None
    if not pv_opt and not pv_cmp:
        return None
    dims = [dim for dim in data.dims if str(dim).strip().lower() != "total"]
    extras: Set[str] = set()
    if pv_opt:
        extras.update(pv_opt.keys())
    if pv_cmp:
        extras.update(pv_cmp.keys())
    for dim in sorted(extras, key=str):
        if str(dim).strip().lower() == "total":
            continue
        if dim not in dims:
            dims.append(dim)
    if not dims:
        return None
    rows = []
    for dim in dims:
        rows.append(
            {
                "Dimension": str(dim),
                "Optimised NPV ($)": float(pv_opt.get(dim, 0.0) if pv_opt else 0.0) * 1_000_000.0,
                "Comparison NPV ($)": float(pv_cmp.get(dim, 0.0) if pv_cmp else 0.0) * 1_000_000.0,
            }
        )
    return pd.DataFrame(rows)

def prepare_gantt_export(
    data: DashboardData,
    selection: ScenarioSelection,
    comparison_selection: ScenarioSelection,
) -> Optional[pd.DataFrame]:
    if not selection or not selection.code:
        return None
    runs = extract_project_runs(data, selection.code)
    if not runs:
        return None
    comparison_runs: Dict[str, scenario_utils.ProjectRun] = {}
    if comparison_selection and comparison_selection.code:
        comparison_runs = {
            _normalise_project_key(run.project): run
            for run in extract_project_runs(data, comparison_selection.code)
        }
    rows = []
    for run in runs:
        comp = comparison_runs.get(_normalise_project_key(run.project))
        rows.append(
            {
                "Project": run.project,
                "Start FY": run.start_year,
                "End FY": run.end_year,
                "Duration (years)": run.end_year - run.start_year + 1,
                "Total spend ($)": float(run.total_spend) * 1_000_000.0,
                "Comparison start FY": comp.start_year if comp else None,
                "Comparison end FY": comp.end_year if comp else None,
                "Comparison total spend ($)": float(comp.total_spend) * 1_000_000.0 if comp else None,
            }
        )
    return pd.DataFrame(rows)

def prepare_schedule_export(
    data: DashboardData,
    selection: ScenarioSelection,
) -> Optional[pd.DataFrame]:
    if not selection or not selection.code:
        return None
    runs = extract_project_runs(data, selection.code)
    if not runs:
        return None
    rows = []
    for run in runs:
        for year, value in zip(data.years, run.values):
            if abs(float(value)) <= 1e-9:
                continue
            rows.append(
                {
                    "Project": run.project,
                    "Financial year": int(year),
                    "Annual spend ($)": float(value) * 1_000_000.0,
                    "Total spend ($)": float(run.total_spend) * 1_000_000.0,
                    "Start FY": run.start_year,
                    "End FY": run.end_year,
                }
            )
    if not rows:
        return None
    return pd.DataFrame(rows)

def prepare_capacity_export(series: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if series is None or series.empty or "Spend" not in series or "Year" not in series:
        return None
    df = series[["Year", "Spend"]].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])
    if df.empty:
        return None
    df["Year"] = df["Year"].astype(int)
    spend_b = pd.to_numeric(df["Spend"], errors="coerce").astype(float) / 1000.0
    df["Annual spend ($)"] = scale_series_to_nzd(df["Spend"])
    df["Annual spend ($B)"] = spend_b
    status: List[str] = []
    for value in spend_b:
        if value <= 0.0:
            status.append("No spend recorded")
        elif value >= 3.0:
            status.append("High pressure (>= $3.0B)")
        elif value <= 2.0:
            status.append("Comfortable (<= $2.0B)")
        else:
            status.append("Watch zone ($2.0B-$3.0B)")
    df["Status"] = status
    df = df.drop(columns=["Spend"])
    return df.rename(columns={"Year": "Financial year"})


def format_npv_context(rate: float, *horizon_values: Optional[int]) -> str:
    """Describe the NPV scope including horizon and discount rate."""
    rate_pct = float(rate) * 100.0
    years = sorted({int(value) for value in horizon_values if value is not None})
    if not years:
        return f"NPV @ {rate_pct:.1f}% discount"
    if len(years) == 1:
        return f"{years[0]}-year NPV @ {rate_pct:.1f}% discount"
    horizon_text = '/'.join(str(year) for year in years)
    return f"{horizon_text}-year NPV @ {rate_pct:.1f}% discount"


def npv_context_label(data: DashboardData, *selections: ScenarioSelection, horizon_override: Optional[int] = None) -> str:

    rates: List[float] = []

    horizons: List[int] = []

    for selection in selections:

        if not selection:

            continue

        meta = selection.metadata or {}

        rate = meta.get('BenRate')

        if rate is not None:

            try:

                rates.append(float(rate))

            except (TypeError, ValueError):

                pass

        horizon = meta.get('HorizonYears')

        if horizon is not None:

            try:

                horizons.append(int(horizon))

            except (TypeError, ValueError):

                pass

    rate_value: Optional[float] = rates[0] if rates else getattr(data, 'benefit_rate', None)

    if rate_value is None:

        rate_value = 0.0

    if horizon_override is not None:

        try:

            horizons = [int(horizon_override)]

        except (TypeError, ValueError):

            horizons = []

    if not horizons:

        fallback_horizon = getattr(data, 'model_years', None)

        if fallback_horizon is not None:

            try:

                horizons.append(int(fallback_horizon))

            except (TypeError, ValueError):

                pass

    return format_npv_context(rate_value, *horizons)




def format_eta(seconds: Optional[float]) -> str:

    if seconds is None or not math.isfinite(seconds):

        return "ETA: --"

    total = max(int(round(seconds)), 0)

    minutes, secs = divmod(total, 60)

    hours, minutes = divmod(minutes, 60)

    if hours:

        return f"ETA ~ {hours}h {minutes:02d}m"

    if minutes:

        return f"ETA ~ {minutes}m {secs:02d}s"

    return f"ETA ~ {secs}s"

def _sorted_profile_labels(labels: Iterable[str]) -> List[str]:

    cleaned = {str(label).strip() for label in labels if str(label).strip()}

    if not cleaned:

        return [DEFAULT_PROFILE_LABEL]

    ordered = sorted(cleaned)

    if DEFAULT_PROFILE_LABEL in ordered:

        ordered.remove(DEFAULT_PROFILE_LABEL)

        ordered.insert(0, DEFAULT_PROFILE_LABEL)

    return ordered


def profile_options(

    data: DashboardData,

    *,

    prefer_comparison: Optional[bool] = None,

) -> List[str]:

    df = data.scenarios

    if "Profile" not in df.columns:

        return [DEFAULT_PROFILE_LABEL]

    subset = df

    if prefer_comparison is not None and "IsComp" in subset.columns:

        mask = subset["IsComp"] == (1 if prefer_comparison else 0)

        if mask.any():

            subset = subset[mask]

    return _sorted_profile_labels(subset["Profile"].dropna().astype(str).tolist())

def available_envelopes(

    data: DashboardData,

    *,

    confidence: str,

    steep: str,

    horizon: int,

    mode: str,

    prefer_comparison: Optional[bool] = None,

    scenarios_df: Optional[pd.DataFrame] = None,

) -> List[int]:

    df = scenarios_df if scenarios_df is not None else data.scenarios

    if df.empty:

        return []

    mask = (

        (df["Conf"] == confidence)

        & (df["BenSteep"] == steep)

        & (pd.to_numeric(df["BenHorizon"], errors="coerce") == int(horizon))

        & (df["Mode"] == mode)

    )

    if prefer_comparison is not None and "IsComp" in df.columns:

        mask = mask & (df["IsComp"] == (1 if prefer_comparison else 0))

    subset = df.loc[mask]

    if subset.empty or mode not in {"fixed", "buffered", "cash"}:

        return []

    values = pd.to_numeric(subset["Envelope"], errors="coerce").dropna()

    return sorted({int(v) for v in values.tolist()})



def available_buffer_levels(

    data: DashboardData,

    *,

    confidence: str,

    steep: str,

    horizon: int,

    mode: str,

    prefer_comparison: Optional[bool] = None,

    scenarios_df: Optional[pd.DataFrame] = None,

) -> List[int]:

    df = scenarios_df if scenarios_df is not None else data.scenarios

    if df.empty:

        return []

    mask = (

        (df["Conf"] == confidence)

        & (df["BenSteep"] == steep)

        & (pd.to_numeric(df["BenHorizon"], errors="coerce") == int(horizon))

        & (df["Mode"] == mode)

    )

    if prefer_comparison is not None and "IsComp" in df.columns:

        mask = mask & (df["IsComp"] == (1 if prefer_comparison else 0))

    subset = df.loc[mask]

    if subset.empty:

        return []

    if mode == "buffered":

        values = pd.to_numeric(subset["Buffer"], errors="coerce").dropna()

        return sorted({int(v) for v in values.tolist()})

    if mode == "cash":

        values = pd.to_numeric(subset["CashPlus"], errors="coerce").dropna()

        return sorted({int(v) for v in values.tolist()})

    return []



def scenario_code_options(

    data: DashboardData,

    *,

    prefer_comparison: Optional[bool] = None,

    scenarios_df: Optional[pd.DataFrame] = None,

) -> List[str]:

    df = scenarios_df if scenarios_df is not None else data.scenarios

    if prefer_comparison is not None and "IsComp" in df.columns:

        mask = df["IsComp"] == (1 if prefer_comparison else 0)

        if mask.any():

            df = df[mask]

    if df.empty:

        return []

    return sorted(df["Code"].dropna().astype(str).unique().tolist())




def scenario_selector(

    *,

    name: str,

    data: DashboardData,

    settings: Settings,

    prefer_comparison: bool,

    key_prefix: str,

    profile_name: Optional[str] = None,

) -> ScenarioSelection:

    base_df = data.scenarios

    profile_df = base_df

    profile_filtered = False

    if profile_name and "Profile" in base_df.columns:

        profile_mask = base_df["Profile"] == profile_name

        if profile_mask.any():

            profile_df = base_df.loc[profile_mask].copy()

            profile_filtered = True

    if profile_df.empty:

        profile_df = base_df

        profile_filtered = False

    options_source = replace(data, scenarios=profile_df) if profile_filtered else data

    options = options_source.scenario_options()

    conf_index = options.conf.index(options.conf[0]) if options.conf else 0

    confidence = st.selectbox(

        f"{name} confidence level",

        options.conf,

        index=conf_index,

        key=f"{key_prefix}_conf",

    )

    steep_index = options.benefit_steep.index(options.benefit_steep[0]) if options.benefit_steep else 0

    benefit_steep = st.selectbox(

        f"{name} benefit steepness (aggressive <-> conservative)",

        options.benefit_steep,

        index=steep_index,

        key=f"{key_prefix}_steep",

    )

    horizon_index = 0

    benefit_horizon = st.selectbox(

        f"{name} benefit horizon (years)",

        options.benefit_horizon,

        index=horizon_index,

        key=f"{key_prefix}_horizon",

    )

    preferred_modes_order = ["buffered", "fixed", "cash"]

    available_modes = [m for m in preferred_modes_order if m in options.modes]

    if not available_modes:

        available_modes = [m for m in options.modes if m != "unconstrained"]

    if not available_modes:

        available_modes = list(options.modes or [])

    if not available_modes:

        available_modes = ["buffered"]

    mode_alias = {

        "standard": "buffered",

        "comparison": "buffered",

        "buffer": "buffered",

        "buffered": "buffered",

        "fixed": "fixed",

        "cash": "cash",

        "cash+": "cash",

    }

    preferred_mode = (settings.ui.default_run_mode or "").strip().lower()

    default_mode_key = mode_alias.get(preferred_mode, None)

    if default_mode_key not in available_modes:

        default_mode_key = "buffered" if "buffered" in available_modes else available_modes[0]

    mode_labels = [MODE_LABELS[m] for m in available_modes]

    default_mode_index = available_modes.index(default_mode_key) if default_mode_key in available_modes else 0

    selected_label = st.radio(

        f"{name} optimisation mode",

        mode_labels,

        index=default_mode_index,

        horizontal=True,

        key=f"{key_prefix}_mode",

    )

    mode_key = available_modes[mode_labels.index(selected_label)]

    sources: List[pd.DataFrame] = [profile_df]

    if profile_filtered:

        sources.append(base_df)

    def _collect_values(fetcher, allow_none_pref: bool = True):

        prefs = [prefer_comparison]

        if allow_none_pref:

            prefs.append(None)

        for pref in prefs:

            for src in sources:

                if src.empty:

                    continue

                values = fetcher(pref, src)

                if values:

                    return values

        return []

    envelope_values = _collect_values(

        lambda pref, src: available_envelopes(

            data,

            confidence=confidence,

            steep=benefit_steep,

            horizon=int(benefit_horizon),

            mode=mode_key,

            prefer_comparison=pref,

            scenarios_df=src,

        )

    )

    envelope_choices = sorted({int(v) for v in envelope_values})

    if not envelope_choices:

        envelope_choices = sorted({int(v) for v in options.envelopes if v is not None})

    envelope: Optional[int] = None

    if mode_key in {"fixed", "buffered", "cash"} and envelope_choices:

        envelope = int(

            st.selectbox(

                f"{name} envelope ($m p.a.)",

                envelope_choices,

                index=0,

                key=f"{key_prefix}_envelope",

            )

        )

    elif mode_key in {"fixed", "buffered", "cash"}:

        st.warning("No envelope values available for this combination.")

    buffer_value: Optional[int] = None

    if mode_key == "buffered":

        buffer_values = _collect_values(

            lambda pref, src: available_buffer_levels(

                data,

                confidence=confidence,

                steep=benefit_steep,

                horizon=int(benefit_horizon),

                mode=mode_key,

                prefer_comparison=pref,

                scenarios_df=src,

            )

        )

        buffer_choices = sorted({int(v) for v in buffer_values})

        if not buffer_choices:

            buffer_choices = sorted({int(v) for v in options.buffers if v is not None})

        if buffer_choices:

            buffer_value = int(

                st.selectbox(

                    f"{name} buffer (+/- $m)",

                    buffer_choices,

                    index=0,

                    key=f"{key_prefix}_buffer",

                )

            )

        else:

            st.warning("No buffer levels available for this combination.")

    elif mode_key == "cash":

        cash_values = _collect_values(

            lambda pref, src: available_buffer_levels(

                data,

                confidence=confidence,

                steep=benefit_steep,

                horizon=int(benefit_horizon),

                mode=mode_key,

                prefer_comparison=pref,

                scenarios_df=src,

            )

        )

        cash_choices = sorted({int(v) for v in cash_values})

        if not cash_choices:

            cash_choices = sorted({int(v) for v in options.buffers if v is not None})

        if cash_choices:

            buffer_value = int(

                st.selectbox(

                    f"{name} cash uplift (+ $m)",

                    cash_choices,

                    index=0,

                    key=f"{key_prefix}_cash",

                )

            )

        else:

            st.warning("No cash uplift levels available for this combination.")

    dimension_options = data.dims

    default_dim_index = dimension_options.index("Total") if "Total" in dimension_options else 0

    dimension = st.selectbox(

        f"{name} benefit dimension",

        dimension_options,

        index=default_dim_index,

        key=f"{key_prefix}_dim",

    )

    def _match_mask(df: pd.DataFrame) -> pd.Series:

        mask = (

            (df["Conf"] == confidence)

            & (df["BenSteep"] == benefit_steep)

            & (pd.to_numeric(df["BenHorizon"], errors="coerce") == int(benefit_horizon))

            & (df["Mode"] == mode_key)

        )

        if envelope is not None:

            env_series = pd.to_numeric(df["Envelope"], errors="coerce")

            env_match = np.isclose(env_series.to_numpy(), float(envelope), atol=1e-6, equal_nan=False)

            mask = mask & pd.Series(env_match, index=df.index)

        if buffer_value is not None:

            if mode_key == "buffered":

                buf_series = pd.to_numeric(df["Buffer"], errors="coerce")

            else:

                buf_series = pd.to_numeric(df["CashPlus"], errors="coerce")

            buf_match = np.isclose(buf_series.to_numpy(), float(buffer_value), atol=1e-6, equal_nan=False)

            mask = mask & pd.Series(buf_match, index=df.index)

        if prefer_comparison is not None and "IsComp" in df.columns:

            comp_flag = 1 if prefer_comparison else 0

            comp_mask = df["IsComp"] == comp_flag

            if comp_mask.any():

                mask = mask & comp_mask

        return mask

    subset_codes: List[str] = []

    for src in sources:

        if src.empty:

            continue

        mask = _match_mask(src)

        codes = src.loc[mask, "Code"].dropna().astype(str).tolist()

        if codes:

            subset_codes = codes

            break

    codes_pool = _collect_values(

        lambda pref, src: scenario_code_options(

            data,

            prefer_comparison=pref,

            scenarios_df=src,

        )

    ) or []

    recommended_code: Optional[str] = None

    for idx, src in enumerate(sources):

        profile_filter = profile_name if profile_filtered and idx == 0 else None

        candidate = find_scenario_code(

            data,

            conf=confidence,

            benefit_steep=benefit_steep,

            benefit_horizon=int(benefit_horizon),

            mode=mode_key,

            envelope=float(envelope) if envelope is not None else None,

            buffer_value=float(buffer_value) if buffer_value is not None else None,

            prefer_comparison=prefer_comparison,

            profile=profile_filter,

            scenarios_df=src,

        )

        if candidate:

            recommended_code = candidate

            break

    if recommended_code is None:

        recommended_code = find_scenario_code(

            data,

            conf=confidence,

            benefit_steep=benefit_steep,

            benefit_horizon=int(benefit_horizon),

            mode=mode_key,

            envelope=float(envelope) if envelope is not None else None,

            buffer_value=float(buffer_value) if buffer_value is not None else None,

            prefer_comparison=prefer_comparison,

            profile=profile_name,

        )

    code_choices = subset_codes or codes_pool or scenario_code_options(data)

    if recommended_code is None and code_choices:

        recommended_code = code_choices[0]

    code = recommended_code if (recommended_code and recommended_code in code_choices) else (code_choices[0] if code_choices else None)

    meta = scenario_metadata(data, code) if code else None

    selected_profile = profile_name

    if meta:

        profile_from_meta = meta.get("Profile")

        if profile_from_meta:

            selected_profile = str(profile_from_meta)

    if selected_profile is None and profile_filtered:

        selected_profile = profile_name or DEFAULT_PROFILE_LABEL

    return ScenarioSelection(

        name=name,

        code=code,

        confidence=confidence,

        benefit_steep=benefit_steep,

        benefit_horizon=int(benefit_horizon),

        mode=mode_key,

        envelope=envelope,

        buffer_value=buffer_value,

        dimension=dimension,

        profile=selected_profile,

        metadata=meta,

    )

def build_timeseries(data: DashboardData, selection: ScenarioSelection) -> Optional[pd.DataFrame]:

    if not selection.code:

        return None

    cf = data.cf[data.cf["Code"] == selection.code].copy()

    if cf.empty:

        return None

    cf = cf.sort_values("Year")

    base = pd.DataFrame({"Year": data.years})

    df = base.merge(cf, on="Year", how="left").sort_values("Year").fillna(0.0)

    for col in ("Spend", "ClosingNet", "Envelope"):

        if col in df.columns:

            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        else:

            df[col] = 0.0

    benefit_dim = data.benefit_dim[

        (data.benefit_dim["Code"] == selection.code)

        & (data.benefit_dim["Dimension"].astype(str).str.lower() == selection.dimension.lower())

    ][["Year", "BenefitFlow"]]

    if benefit_dim.empty:

        benefit_dim = data.benefit[data.benefit["Code"] == selection.code][["Year", "BenefitFlow"]]

    df = df.merge(benefit_dim, on="Year", how="left", suffixes=("", "_dim"))

    df["BenefitFlow"] = pd.to_numeric(df["BenefitFlow"], errors="coerce").fillna(0.0)

    year_offsets = (df["Year"].astype(int) - data.start_fy).clip(lower=0)

    discount_base = 1.0 + data.benefit_rate

    discount = np.power(discount_base, year_offsets.to_numpy())

    discount[discount == 0] = 1.0

    df["PVBenefit"] = df["BenefitFlow"] / discount

    df["CumPVBenefit"] = df["PVBenefit"].cumsum()

    df["CumBenefit"] = df["BenefitFlow"].cumsum()

    df["CumSpend"] = df["Spend"].cumsum()

    return df

def pv_by_dimension(data: DashboardData, selection: ScenarioSelection, *, horizon_years: Optional[int] = None) -> Optional[Dict[str, float]]:

    if not selection.code:

        return None

    df = data.benefit_dim[data.benefit_dim["Code"] == selection.code]

    if df.empty:

        return None

    dims_present = df["Dimension"].dropna().astype(str)

    if dims_present.empty:

        return None

    base_years = pd.DataFrame({"Year": data.years})

    rate = 1.0 + data.benefit_rate

    ordered_dims = [dim for dim in data.dims if dim in dims_present.tolist()]

    remaining_dims = [dim for dim in dims_present.unique().tolist() if dim not in ordered_dims]

    dim_sequence = ordered_dims + sorted(remaining_dims, key=str)

    pv: Dict[str, float] = {}

    for dim in dim_sequence:

        mask = df["Dimension"].astype(str).str.lower() == str(dim).lower()

        if not mask.any():

            continue

        dim_rows = df.loc[mask, ["Year", "BenefitFlow"]]

        merged = base_years.merge(dim_rows, on="Year", how="left").fillna(0.0)

        offsets = (merged["Year"].astype(int) - data.start_fy).clip(lower=0)

        if horizon_years is not None:

            try:

                limit = max(int(horizon_years) - 1, 0)

            except (TypeError, ValueError):

                limit = None

            if limit is not None:

                mask = offsets <= limit

                merged = merged.loc[mask].copy()

                offsets = offsets[mask]

        if merged.empty:

            pv[str(dim)] = 0.0

            continue

        discount = np.power(rate, offsets.to_numpy())

        discount[discount == 0] = 1.0

        pv[str(dim)] = float((merged["BenefitFlow"] / discount).sum())

    return pv or None

def scenario_metrics(
    df: pd.DataFrame,
    *,
    start_year: int,
    horizon_years: Optional[int] = None,
) -> Dict[str, float]:

    window = df

    if horizon_years is not None:

        try:

            limit_year = int(start_year) + int(horizon_years) - 1

            window = df[df["Year"].astype(int) <= limit_year]

        except (TypeError, ValueError):

            window = df

    return {

        "total_spend": float(window["Spend"].sum()),

        "total_benefit": float(window["BenefitFlow"].sum()),

        "total_pv": float(window["PVBenefit"].sum()),

    }

def cash_chart(

    df: pd.DataFrame,

    title: str,

    *,

    color: str,

    data: Optional[DashboardData] = None,

    selection: Optional[ScenarioSelection] = None,

    comparison_selection: Optional[ScenarioSelection] = None,

    horizon_override: Optional[int] = None,

) -> go.Figure:

    fig = go.Figure()

    spend_values = df["Spend"].astype(float) * 1_000_000.0
    closing_values = df["ClosingNet"].astype(float) * 1_000_000.0
    envelope_values = df["Envelope"].astype(float) * 1_000_000.0

    axis_samples = [
        spend_values.to_numpy(),
        closing_values.to_numpy(),
        envelope_values.to_numpy(),
    ]

    non_empty_samples = [arr for arr in axis_samples if arr.size]
    tick_vals, tick_text, unit_label = compute_cash_axis_ticks(
        np.concatenate(non_empty_samples) if non_empty_samples else [0.0]
    )

    context_label: Optional[str] = None

    if data is not None and selection is not None:

        selections: List[ScenarioSelection] = [selection]

        if comparison_selection is not None:

            selections.append(comparison_selection)

        context_label = npv_context_label(

            data,

            *selections,

            horizon_override=horizon_override,

        )

    fig.add_trace(

        go.Bar(

            x=df["Year"],

            y=spend_values,

            name="Annual spend",

            marker_color=color,

            opacity=BAR_OPACITY,

            customdata=[format_large_amount(val) for val in spend_values],

            hovertemplate="<b>Annual spend</b><br>FY %{x}: %{customdata}<extra></extra>",

        )

    )

    fig.add_trace(

        go.Scatter(

            x=df["Year"],

            y=closing_values,

            name="Closing net",

            mode="lines",

            line=dict(color=CLOSING_NET_COLOR, width=3),

            customdata=[format_large_amount(val) for val in closing_values],

            hovertemplate="<b>Closing net</b><br>FY %{x}: %{customdata}<extra></extra>",

        )

    )

    fig.add_trace(

        go.Scatter(

            x=df["Year"],

            y=envelope_values,

            name="Envelope",

            mode="lines+markers",

            line=dict(color=ENVELOPE_COLOR, width=3),

            marker=dict(color=ENVELOPE_COLOR, size=6),

            customdata=[format_large_amount(val) for val in envelope_values],

            hovertemplate="<b>Envelope</b><br>FY %{x}: %{customdata}<extra></extra>",

        )

    )

    fig.add_hline(y=0, line=dict(color="#888888", dash="dot", width=1))

    yaxis_context = unit_label

    if context_label:

        yaxis_context = f"{context_label}, {unit_label}"

    yaxis_title_default = f"$ ({yaxis_context})"

    fig.update_layout(

        title=title,

        barmode="overlay",

        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),

        xaxis_title=None,

        yaxis_title=yaxis_title_default,

        template=plotly_template(),

        yaxis=dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text),

        hoverlabel=dict(namelength=-1),

    )

    return fig

def benefit_chart(

    opt_df: Optional[pd.DataFrame],

    cmp_df: Optional[pd.DataFrame],

    *,

    dimension: str,

) -> go.Figure:

    fig = go.Figure()

    if opt_df is not None:

        fig.add_trace(

            go.Scatter(

                x=opt_df["Year"],

                y=opt_df["PVBenefit"],

                name="Optimized Benefit Real",

                mode="lines",

                line=dict(color=CUMULATIVE_OPT_LINE_COLOR, width=3.2),

                opacity=0.85,

                yaxis="y2",

                hovertemplate="<b>Optimized Benefit Real</b><br>FY %{x}: %{y:,.1f}m<extra></extra>",

            )

        )

    if cmp_df is not None:

        fig.add_trace(

            go.Scatter(

                x=cmp_df["Year"],

                y=cmp_df["PVBenefit"],

                name="Comparison Benefit Real",

                mode="lines",

                line=dict(color=CUMULATIVE_CMP_LINE_COLOR, dash="dot", width=2.6),

                opacity=0.85,

                yaxis="y2",

                hovertemplate="<b>Comparison Benefit Real</b><br>FY %{x}: %{y:,.1f}m<extra></extra>",

            )

        )

    fig.update_layout(

        title=f"Real benefit to date ({dimension})",

        xaxis_title=None,

        yaxis_title="Annual benefit ($m)",

        yaxis2=dict(

            title="Benefit to date (real $m)",

            overlaying="y",

            side="right",

        ),

        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),

        template=plotly_template(),

        hoverlabel=dict(namelength=-1),


    )

    return fig

def benefit_delta_chart(

    data: DashboardData,

    opt_df: Optional[pd.DataFrame],

    cmp_df: Optional[pd.DataFrame],

    opt_selection: ScenarioSelection,

    cmp_selection: ScenarioSelection,

    *,

    horizon_years: Optional[int] = None,

) -> go.Figure:

    fig = go.Figure()

    if opt_df is None or cmp_df is None:

        fig.add_annotation(text="Select both scenarios to view the benefit delta", showarrow=False)

        fig.update_layout(template=plotly_template())

        return fig

    context_label = npv_context_label(

        data,

        opt_selection,

        cmp_selection,

        horizon_override=horizon_years,

    )

    merged = opt_df[["Year", "CumBenefit", "CumPVBenefit"]].merge(

        cmp_df[["Year", "CumBenefit", "CumPVBenefit"]],

        on="Year",

        suffixes=("_opt", "_cmp"),

    )

    merged["DeltaBenefit"] = merged["CumBenefit_opt"] - merged["CumBenefit_cmp"]

    merged["DeltaPV"] = merged["CumPVBenefit_opt"] - merged["CumPVBenefit_cmp"]

    fig.add_trace(

        go.Scatter(

            x=merged["Year"],

            y=merged["DeltaBenefit"],

            name="Delta Cumulative Benefit Real",

            mode="lines",

            line=dict(color=CUMULATIVE_OPT_LINE_COLOR, width=3.0),
            hovertemplate="<b>Delta Cumulative Benefit Real</b><br>FY %{x}: %{y:,.1f}m<extra></extra>",



        )

    )

    fig.add_trace(

        go.Scatter(

            x=merged["Year"],

            y=merged["DeltaPV"],

            name="Delta Benefit Real",

            mode="lines",

            line=dict(color=CUMULATIVE_CMP_LINE_COLOR, dash="dot", width=2.6),
            hovertemplate="<b>Delta Benefit Real</b><br>FY %{x}: %{y:,.1f}m<extra></extra>",



        )

    )

    title_text = "Cumulative Benefits Delta Real"

    yaxis_text = f"$ millions ({context_label})" if context_label else "$ millions (real)"

    fig.update_layout(

        title=title_text,

        xaxis_title=None,

        yaxis_title=yaxis_text,

        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),

        template=plotly_template(),

        hoverlabel=dict(namelength=-1),


    )

    return fig


def benefit_radar_chart(
    data: DashboardData,
    opt_selection: ScenarioSelection,
    cmp_selection: ScenarioSelection,
) -> Optional[go.Figure]:
    opt_pv = pv_by_dimension(data, opt_selection) if opt_selection and opt_selection.code else None
    cmp_pv = pv_by_dimension(data, cmp_selection) if cmp_selection and cmp_selection.code else None

    if not opt_pv and not cmp_pv:
        return None

    dims = [dim for dim in data.dims if str(dim).strip().lower() != "total"]

    extras = set()
    if opt_pv:
        extras.update(opt_pv.keys())
    if cmp_pv:
        extras.update(cmp_pv.keys())

    for dim in sorted(extras, key=str):
        if str(dim).strip().lower() == "total":
            continue
        if dim not in dims:
            dims.append(dim)

    if not dims:
        return None

    def value_list(pv_dict: Optional[Dict[str, float]]) -> List[float]:
        if not pv_dict:
            return [0.0 for _ in dims]
        return [float(pv_dict.get(dim, 0.0)) for dim in dims]

    theta_labels = [str(dim) for dim in dims]
    n_dims = max(len(dims), 1)
    theta_positions = [(idx / n_dims) * 360.0 for idx in range(len(dims))]
    theta_closed = theta_positions + theta_positions[:1]

    opt_values = value_list(opt_pv)
    cmp_values = value_list(cmp_pv)

    opt_closed = opt_values + opt_values[:1]
    cmp_closed = cmp_values + cmp_values[:1]

    max_val = max(opt_values + cmp_values + [1.0])
    value_scale = max(max_val, 1.0)

    theme_base = (st.get_option("theme.base") or "").lower()
    background_setting = (st.get_option("theme.backgroundColor") or "").strip()
    text_setting = (st.get_option("theme.textColor") or "").strip()
    dark_toggle_active = is_dark_mode()

    if dark_toggle_active:
        background_color = "#0E1117"
        text_color = "#E2E8F0"
        is_dark_theme = True
    else:
        background_color = background_setting or ("#0E1117" if theme_base == "dark" else "#FFFFFF")
        background_luminance = relative_luminance(background_color)
        is_dark_theme = theme_base == "dark"
        if background_luminance is not None:
            is_dark_theme = background_luminance < 0.45

        text_color = text_setting or ("#E2E8F0" if is_dark_theme else "#1F2933")
        text_luminance = relative_luminance(text_color)
        if text_luminance is not None:
            if is_dark_theme and text_luminance < 0.6:
                text_color = "#E2E8F0"
            elif not is_dark_theme and text_luminance > 0.4:
                text_color = "#1F2933"

    figure_background = "rgba(0, 0, 0, 0)" if is_dark_theme else background_color
    grid_color = "rgba(148, 163, 184, 0.35)" if is_dark_theme else "rgba(148, 163, 184, 0.28)"
    outline_color = "rgba(148, 163, 184, 0.55)" if is_dark_theme else "rgba(100, 116, 139, 0.45)"

    fig = go.Figure()

    if opt_pv:
        opt_name = opt_selection.name if opt_selection and opt_selection.name else "Optimised"
        opt_custom = theta_labels + theta_labels[:1]
        fig.add_trace(
            go.Scatterpolar(
                r=opt_closed,
                theta=theta_closed,
                name=opt_name,
                line=dict(color=CUMULATIVE_OPT_LINE_COLOR, width=3),
                fill="toself",
                fillcolor=rgba_from_hex(CUMULATIVE_OPT_LINE_COLOR, 0.26 if is_dark_theme else 0.18),
                opacity=0.9,
                customdata=opt_custom,
                hovertemplate="<b>%{customdata}</b><br>%{r:,.0f}<extra>%{name}</extra>",
            )
        )

    if cmp_pv:
        cmp_name = cmp_selection.name if cmp_selection and cmp_selection.name else "Comparison"
        cmp_custom = theta_labels + theta_labels[:1]
        fig.add_trace(
            go.Scatterpolar(
                r=cmp_closed,
                theta=theta_closed,
                name=cmp_name,
                line=dict(color=CUMULATIVE_CMP_LINE_COLOR, width=2, dash="dash"),
                fill="toself",
                fillcolor=rgba_from_hex(CUMULATIVE_CMP_LINE_COLOR, 0.22 if is_dark_theme else 0.16),
                opacity=0.78,
                customdata=cmp_custom,
                hovertemplate="<b>%{customdata}</b><br>%{r:,.0f}<extra>%{name}</extra>",
            )
        )

    series_configs = []
    if opt_pv:
        series_configs.append((opt_values, CUMULATIVE_OPT_LINE_COLOR, -1))
    if cmp_pv:
        direction = 1 if opt_pv else -1
        series_configs.append((cmp_values, CUMULATIVE_CMP_LINE_COLOR, direction))

    def _text_anchor(base_angle_deg: float) -> str:
        cos_a = math.cos(math.radians(base_angle_deg))
        sin_a = math.sin(math.radians(base_angle_deg))
        if abs(cos_a) >= 0.4:
            return "middle right" if cos_a >= 0 else "middle left"
        if sin_a >= 0:
            return "bottom center"
        return "top center"

    label_padding = max(value_scale * 0.06, 2.0)
    angle_base = max(10.0, 8.0 * (len(theta_labels) / max(len(theta_labels), 3)))

    for idx, base_angle in enumerate(theta_positions):
        for values, color, direction in series_configs:
            if idx >= len(values):
                continue
            value = float(values[idx])
            if value <= 0:
                continue
            angle_adjust = angle_base * direction
            cos_a = math.cos(math.radians(base_angle))
            if abs(cos_a) < 0.35:
                angle_adjust *= 0.7
            leader_end = min(value_scale * 1.15, value + max(label_padding, value * 0.12))
            final_theta = (base_angle + angle_adjust) % 360.0
            anchor = _text_anchor(final_theta)
            fig.add_trace(
                go.Scatterpolar(
                    r=[value, (value + leader_end) / 2.0, leader_end],
                    theta=[base_angle, base_angle + angle_adjust * 0.4, final_theta],
                    mode="lines",
                    line=dict(color=color, width=1.4, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                    cliponaxis=False,
                )
            )
            fig.add_trace(
                go.Scatterpolar(
                    r=[leader_end],
                    theta=[final_theta],
                    mode="text",
                    text=[f"{value:,.0f}"],
                    textposition=anchor,
                    textfont=dict(color=color, size=11),
                    showlegend=False,
                    hoverinfo="skip",
                    cliponaxis=False,
                )
            )
    fig.update_layout(
        title="Benefit mix radar",
        polar=dict(
            bgcolor="rgba(0, 0, 0, 0)",
            radialaxis=dict(
                visible=True,
                range=[0, value_scale * 1.12],
                title="",
                tickmode="array",
                tickvals=[],
                ticktext=[],
                gridcolor=grid_color,
                linecolor=outline_color,
                gridwidth=1,
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                tickmode="array",
                tickvals=theta_positions,
                ticktext=theta_labels,
                thetaunit="degrees",
                gridcolor=grid_color,
                linecolor=outline_color,
                tickfont=dict(color=text_color, size=11),
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="left",
            x=0.0,
            font=dict(color=text_color, size=11),
            bgcolor="rgba(0, 0, 0, 0)",
        ),
        template=None,
        paper_bgcolor=figure_background,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=40, r=40, t=80, b=50),
        font=dict(color=text_color, size=12),
    )

    fig.layout.paper_bgcolor = figure_background

    return fig

def benefit_waterfall_chart(

    data: DashboardData,

    opt_selection: ScenarioSelection,

    cmp_selection: ScenarioSelection,

    *,

    horizon_years: Optional[int] = None,

) -> go.Figure:

    fig = go.Figure()

    if not opt_selection.code or not cmp_selection.code:

        fig.add_annotation(text="Select both scenarios to view the NPV waterfall", showarrow=False)

        fig.update_layout(template=plotly_template())

        return fig

    context_label = npv_context_label(data, opt_selection, cmp_selection, horizon_override=horizon_years)

    pv_opt = pv_by_dimension(data, opt_selection, horizon_years=horizon_years)

    pv_cmp = pv_by_dimension(data, cmp_selection, horizon_years=horizon_years)

    if not pv_opt or not pv_cmp:

        fig.add_annotation(text="Dimension-level NPV data unavailable for the selected scenarios", showarrow=False)

        fig.update_layout(template=plotly_template())

        return fig

    dims = set(pv_opt.keys()) | set(pv_cmp.keys())

    ordered_dims = [dim for dim in data.dims if dim in dims]

    remaining_dims = [dim for dim in dims if dim not in ordered_dims]

    non_total_dims = [dim for dim in ordered_dims if str(dim).lower() != "total"] + [dim for dim in remaining_dims if str(dim).lower() != "total"]

    waterfall_labels = non_total_dims[:]

    measures = ["relative"] * len(waterfall_labels)

    values = [pv_opt.get(dim, 0.0) - pv_cmp.get(dim, 0.0) for dim in waterfall_labels]

    total_dim = next((dim for dim in ordered_dims if str(dim).lower() == "total"), None)

    if total_dim is None:

        total_dim = next((dim for dim in dims if str(dim).lower() == "total"), None)

    if total_dim:

        total_label = "Total delta" if str(total_dim).strip() == "Total" else f"{total_dim} total"

        total_delta = pv_opt.get(total_dim, 0.0) - pv_cmp.get(total_dim, 0.0)

    else:

        total_label = "Total delta"

        total_delta = sum(values)

    waterfall_labels.append(total_label)

    measures.append("total")

    values.append(total_delta)

    fig.add_trace(

        go.Waterfall(

            name="Delta Benefit Real",

            orientation="v",

            measure=measures,

            x=waterfall_labels,

            y=values,

            increasing={"marker": {"color": PRIMARY_COLOR}},

            decreasing={"marker": {"color": COMPARISON_COLOR}},

            totals={"marker": {"color": "#4C4C4C"}},

            connector={"line": {"color": "#BBBBBB", "width": 0.5}},

            hovertemplate="<b>%{x}</b><br>%{y:,.1f}m<extra></extra>",
        )

    )

    fig.update_layout(

        title=f"{context_label} delta by dimension (optimised minus comparison)",

        yaxis_title=f"$ millions ({context_label})",

        template=plotly_template(),
        hoverlabel=dict(namelength=-1),

        waterfallgap=0.3,

        height=WATERFALL_CHART_HEIGHT,

    )

    return fig

def benefit_bridge_chart(

    data: DashboardData,

    opt_selection: ScenarioSelection,

    cmp_selection: ScenarioSelection,

    *,

    horizon_years: Optional[int] = None,

) -> go.Figure:

    fig = go.Figure()

    context_label = npv_context_label(

        data,

        opt_selection,

        cmp_selection,

        horizon_override=horizon_years,

    )

    if not opt_selection.code or not cmp_selection.code:

        fig.add_annotation(text="Select both scenarios to build the NPV bridge", showarrow=False)

        fig.update_layout(template=plotly_template())

        return fig

    pv_opt = pv_by_dimension(data, opt_selection, horizon_years=horizon_years)

    pv_cmp = pv_by_dimension(data, cmp_selection, horizon_years=horizon_years)

    if not pv_opt or not pv_cmp:

        fig.add_annotation(text="Missing dimension NPV data for bridge", showarrow=False)

        fig.update_layout(template=plotly_template())

        return fig

    dims = set(pv_opt.keys()) | set(pv_cmp.keys())

    ordered_dims = [dim for dim in data.dims if dim in dims and str(dim).strip().lower() != "total"]

    remaining_dims = [dim for dim in dims if dim not in ordered_dims and str(dim).strip().lower() != "total"]

    dim_sequence = ordered_dims + sorted(remaining_dims, key=str)

    total_dim = next((dim for dim in data.dims if str(dim).strip().lower() == "total" and dim in dims), None)

    if total_dim is None:

        total_dim = next((dim for dim in dims if str(dim).strip().lower() == "total"), None)

    total_opt = pv_opt.get(total_dim, sum(pv_opt.values())) if pv_opt else 0.0

    total_cmp = pv_cmp.get(total_dim, sum(pv_cmp.values())) if pv_cmp else 0.0

    bridge_diffs = [pv_opt.get(dim, 0.0) - pv_cmp.get(dim, 0.0) for dim in dim_sequence]

    labels = ["Optimised NPV total"] + [f"{dim} delta NPV" for dim in dim_sequence] + ["Comparison NPV total"]

    measures = ["relative"] + ["relative"] * len(dim_sequence) + ["total"]

    values = [total_opt] + [-delta for delta in bridge_diffs] + [total_cmp]

    hovertexts = [
        f"Optimised NPV total: {total_opt:,.0f} m",
        *[f"{dim} delta NPV: {delta:,.0f} m" for dim, delta in zip(dim_sequence, bridge_diffs)],
        f"Comparison NPV total: {total_cmp:,.0f} m",
    ]

    fig.add_trace(

        go.Waterfall(

            name="NPV bridge",

            orientation="v",

            measure=measures,

            x=labels,

            y=values,

            increasing=dict(marker=dict(color=PRIMARY_COLOR)),

            decreasing=dict(marker=dict(color=COMPARISON_COLOR)),

            totals=dict(marker=dict(color=COMPARISON_COLOR)),

            connector=dict(line=dict(color="#BBBBBB", width=0.5)),

            hovertext=hovertexts,

            hovertemplate="%{hovertext}<extra></extra>",

        )

    )

    fig.update_layout(

        title=f"{context_label} bridge (optimised to comparison)",

        yaxis_title=f"$ millions ({context_label})",

        template=plotly_template(),

        waterfallgap=0.3,

        height=WATERFALL_CHART_HEIGHT,

    )

    return fig

def _benefit_series_and_label(

    df: pd.DataFrame,

    selection: ScenarioSelection,

    *,

    prefix: str,

) -> Tuple[pd.Series, str]:

    confidence = (selection.confidence or "").lower()

    use_real = "real" in confidence

    column = "CumPVBenefit" if use_real and "CumPVBenefit" in df.columns else "CumBenefit"

    series = df.get(column, pd.Series([], dtype=float))

    if series.empty:

        series = df.get("CumBenefit", pd.Series([], dtype=float))

    label = f"{prefix} cumulative {'PV' if column == 'CumPVBenefit' else 'nominal'} benefit"

    return series, label


def efficiency_chart(

    opt_df: Optional[pd.DataFrame],

    cmp_df: Optional[pd.DataFrame],

    opt_selection: ScenarioSelection,

    cmp_selection: ScenarioSelection,

) -> go.Figure:

    fig = go.Figure()

    series_opt = label_opt = None

    if opt_df is not None and not opt_df.empty:

        fig.add_trace(

            go.Bar(

                x=opt_df["Year"],

                y=opt_df["CumSpend"],

                name="Optimised cumulative spend",

                marker_color=BRIGHT_PRIMARY_COLOR,

                opacity=0.55,

                customdata=np.asarray(opt_df["CumSpend"], dtype=float) / 1000.0,

                hovertemplate="<b>Optimised cumulative spend</b><br>FY %{x}: %{customdata:,.1f}b<extra></extra>",

            )

        )

        series_opt, label_opt = _benefit_series_and_label(opt_df, opt_selection, prefix="Optimised")

    series_cmp = label_cmp = None

    if cmp_df is not None and not cmp_df.empty:

        series_cmp, label_cmp = _benefit_series_and_label(cmp_df, cmp_selection, prefix="Comparison")

        fig.add_trace(

            go.Scatter(

                x=cmp_df["Year"],

                y=series_cmp,

                name=label_cmp,

                mode="lines",

                line=dict(color=CUMULATIVE_CMP_LINE_COLOR, width=2.6, dash="dash"),

                customdata=np.asarray(series_cmp, dtype=float) / 1000.0,

                hovertemplate=f"<b>{label_cmp}</b><br>FY %{{x}}: %{{customdata:,.1f}}b<extra></extra>",

            )

        )

    if series_opt is not None:

        fill_kwargs = {}

        if series_cmp is not None:

            fill_kwargs = dict(

                fill="tonexty",

                fillcolor=rgba_from_hex(CUMULATIVE_OPT_LINE_COLOR, 0.22),

            )

        fig.add_trace(

            go.Scatter(

                x=opt_df["Year"],

                y=series_opt,

                name=label_opt,

                mode="lines",

                line=dict(color=CUMULATIVE_OPT_LINE_COLOR, width=3.2),

                customdata=np.asarray(series_opt, dtype=float) / 1000.0,

                hovertemplate=f"<b>{label_opt}</b><br>FY %{{x}}: %{{customdata:,.1f}}b<extra></extra>",

                **fill_kwargs,

            )

        )

    fig.update_layout(

        title="Cumulative spend vs benefit",

        xaxis_title=None,

        yaxis_title="$ millions",

        template=plotly_template(),

        hoverlabel=dict(namelength=-1),

        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),

    )

    return fig

def dimension_timeseries(data: DashboardData, selection: ScenarioSelection) -> Optional[pd.DataFrame]:

    if not selection.code:

        return None

    df = data.benefit_dim[data.benefit_dim['Code'] == selection.code]

    if df.empty:

        return None

    pivot = (

        df.pivot_table(index='Year', columns='Dimension', values='BenefitFlow', aggfunc='sum')

        .reindex(index=data.years, fill_value=0.0)

        .sort_index()

    )

    ordered_cols: List[str] = []

    for dim in data.dims:

        if dim in pivot.columns:

            ordered_cols.append(dim)

    for dim in pivot.columns:

        if dim not in ordered_cols:

            ordered_cols.append(dim)

    if ordered_cols:

        pivot = pivot[ordered_cols]

    return pivot

def benefit_dimension_chart(

    data: DashboardData,

    selection: ScenarioSelection,

    *,

    title: str,

    cumulative: bool = False,

    pivot: Optional[pd.DataFrame] = None,

) -> Optional[go.Figure]:

    if pivot is None:

        pivot = dimension_timeseries(data, selection)

    if pivot is None or pivot.empty:

        return None

    if cumulative:

        pivot = pivot.cumsum()

    dims = [dim for dim in pivot.columns if str(dim).strip().lower() != 'total']

    if not dims:

        return None

    value_format = ',.1f' if cumulative else ',.0f'

    yaxis_label = "Cumulative benefit ($m)" if cumulative else "Annual benefit ($m)"

    fig = go.Figure()

    for idx, dim in enumerate(dims):

        fig.add_trace(

            go.Scatter(

                x=pivot.index,

                y=pivot[dim],

                name=str(dim),

                mode='lines',

                line=dict(width=1.2),

                stackgroup='benefits',

                hovertemplate=f"FY %{{x}}: %{{y:{value_format}}} m<extra>{dim}</extra>",

            )

        )

    fig.update_layout(

        title=title,

        xaxis_title=None,

        yaxis_title=yaxis_label,

        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.12,
            xanchor='center',
            x=0.5,
            title=None,
        ),

        margin=dict(l=40, r=40, t=80, b=60),

        template=plotly_template(),

    )

    return fig


def benefit_dimension_overlay_chart(

    data: DashboardData,

    opt_selection: ScenarioSelection,

    comp_selection: ScenarioSelection,

    *,

    cumulative: bool = False,

    dimensions: Optional[List[str]] = None,

    opt_pivot: Optional[pd.DataFrame] = None,

    cmp_pivot: Optional[pd.DataFrame] = None,

) -> Optional[go.Figure]:

    pivot_opt = opt_pivot if opt_pivot is not None else dimension_timeseries(data, opt_selection)

    pivot_cmp = cmp_pivot if cmp_pivot is not None else dimension_timeseries(data, comp_selection)

    if pivot_opt is None and pivot_cmp is None:

        return None

    if cumulative:

        if pivot_opt is not None:

            pivot_opt = pivot_opt.cumsum()

        if pivot_cmp is not None:

            pivot_cmp = pivot_cmp.cumsum()

    dims_available: List[str] = []

    for dim in data.dims:

        if str(dim).strip().lower() == 'total':

            continue

        in_opt = pivot_opt is not None and dim in pivot_opt.columns

        in_cmp = pivot_cmp is not None and dim in pivot_cmp.columns

        if in_opt or in_cmp:

            dims_available.append(str(dim))

    if not dims_available:

        return None

    if dimensions:

        dims_selected = [dim for dim in dimensions if dim in dims_available]

    else:

        dims_selected = dims_available

    if not dims_selected:

        return None

    value_format = ',.1f' if cumulative else ',.0f'

    yaxis_label = "Cumulative benefit ($m)" if cumulative else "Annual benefit ($m)"

    title_suffix = " (cumulative)" if cumulative else ""

    fig = go.Figure()

    palette = list(qualitative.Plotly) if qualitative.Plotly else ['#1f77b4']

    def series_for(pivot: Optional[pd.DataFrame], dim_label: str) -> Optional[pd.Series]:

        if pivot is None or dim_label not in pivot.columns:

            return None

        return pivot[dim_label]  # type: ignore[index]

    for idx, dim in enumerate(dims_selected):

        color = palette[idx % len(palette)]

        opt_series = series_for(pivot_opt, dim)

        cmp_series = series_for(pivot_cmp, dim)

        if opt_series is not None:

            fig.add_trace(

                go.Scatter(

                    x=opt_series.index,

                    y=opt_series,

                    name=f"{dim} - Optimised",

                    mode='lines',

                    line=dict(color=color, width=2.6),

                    fill='tozeroy',

                    fillcolor=rgba_from_hex(color, 0.28),

                    opacity=0.85,

                    legendgroup=dim,

                    hovertemplate=f"Optimised {dim}<br>FY %{{x}}: %{{y:{value_format}}} m<extra></extra>",

                )

            )

        if cmp_series is not None:

            fig.add_trace(

                go.Scatter(

                    x=cmp_series.index,

                    y=cmp_series,

                    name=f"{dim} - Comparison",

                    mode='lines',

                    line=dict(color=color, width=2.2, dash='dot'),

                    fill='tozeroy',

                    fillcolor=rgba_from_hex(color, 0.16),

                    opacity=0.85,

                    legendgroup=dim,

                    hovertemplate=f"Comparison {dim}<br>FY %{{x}}: %{{y:{value_format}}} m<extra></extra>",

                    showlegend=True,

                )

            )

    if not fig.data:

        return None

    fig.update_layout(

        title=f"Dimension benefit comparison{title_suffix}",

        xaxis_title=None,

        yaxis_title=yaxis_label,

        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.12,
            xanchor='center',
            x=0.5,
            title=None,
        ),

        margin=dict(l=40, r=40, t=80, b=60),

        template=plotly_template(),

        hoverlabel=dict(namelength=-1),

    )

    return fig

def spend_gantt_chart(

    data: DashboardData,

    selection: ScenarioSelection,

    *,

    comparison_selection: ScenarioSelection,

    show_outline: bool,

    title: str,

) -> Optional[go.Figure]:

    if not selection.code:

        return None

    runs = extract_project_runs(data, selection.code)

    if not runs:

        return None

    runs.sort(key=lambda r: (r.start_year, r.project))

    y_labels = [run.project for run in runs]

    y_positions = list(range(len(runs)))

    base_years = [run.start_year for run in runs]

    durations = [max(1, run.end_year - run.start_year + 1) for run in runs]

    comparison_runs: Dict[str, scenario_utils.ProjectRun] = {}
    if comparison_selection and comparison_selection.code:
        comparison_runs = {
            _normalise_project_key(run.project): run
            for run in extract_project_runs(data, comparison_selection.code)
        }

    comparison_available = bool(comparison_runs)
    shift_labels: List[str] = []
    italic_t = "<i>t</i>"
    for run in runs:
        other = comparison_runs.get(_normalise_project_key(run.project))
        if other:
            diff = int(run.start_year - other.start_year)
            if diff > 0:
                shift = f"{italic_t} + {diff}"
            elif diff < 0:
                shift = f"{italic_t} - {abs(diff)}"
            else:
                shift = f"{italic_t} + 0"
        else:
            shift = f"{italic_t} n/a" if comparison_available else f"{italic_t} + 0"
        shift_labels.append(shift)

    custom = [
        [run.project, run.start_year, run.end_year, run.total_spend, shift_labels[idx]]
        for idx, run in enumerate(runs)
    ]

    fig = go.Figure(

        data=go.Bar(

            y=y_positions,

            x=durations,

            base=base_years,

            orientation="h",

            marker=dict(color=GANTT_COLOR, line=dict(width=0)),

            customdata=custom,

            hovertemplate=(

                "Project: %{customdata[0]}<br>"

                "Total spend: %{customdata[3]:,.0f} m<br>"

                "Start FY: %{customdata[1]}<br>"

                "End FY: %{customdata[2]}<br>"

                "Schedule shift: %{customdata[4]}<extra></extra>"

            ),

            hoverlabel=_hoverlabel_style(),

        )

    )

    if show_outline and comparison_runs:

        outline_color = _current_gantt_outline_color()

        bar_half = 0.4

        x_inset = 0.05

        y_inset = 0.08

        for idx, run in enumerate(runs):

            other = comparison_runs.get(_normalise_project_key(run.project))

            if not other:

                continue

            y_center = y_positions[idx]

            y_bottom = y_center - bar_half + y_inset

            y_top = y_center + bar_half - y_inset

            if y_bottom >= y_top:

                continue

            start = float(other.start_year)

            finish = float(other.end_year) + 1.0

            left = start + x_inset

            right = finish - x_inset

            if left >= right:

                midpoint = (start + finish) / 2.0

                left = midpoint - 0.02

                right = midpoint + 0.02

            shift_label = shift_labels[idx]

            hover_lines = [

                f"Project: {run.project}",

                f"Schedule shift: {shift_label}",

                f"Optimised start: {run.start_year}",

                f"Comparison start: {other.start_year}",

            ]

            hover_text = "<br>".join(hover_lines)

            fig.add_trace(

                go.Scatter(

                    x=[left, right, right, left, left],

                    y=[y_bottom, y_bottom, y_top, y_top, y_bottom],

                    mode="lines",

                    line=dict(color=outline_color, width=1.5, dash="dot"),

                    hovertext=[hover_text] * 5,

                    hovertemplate="%{hovertext}<extra></extra>",

                    hoverlabel=_hoverlabel_style(),

                    showlegend=False,

                    cliponaxis=False,

                )

            )

            actual_start = float(run.start_year)
            actual_end = float(run.end_year) + 1.0
            compare_start = float(other.start_year)
            compare_end = float(other.end_year) + 1.0

            segments: list[tuple[float, float]] = []
            if compare_start < actual_start:
                left_end = min(actual_start, compare_end)
                if left_end - compare_start > 1e-6:
                    segments.append((compare_start, left_end))
            if compare_end > actual_end:
                right_start = max(actual_end, compare_start)
                if compare_end - right_start > 1e-6:
                    segments.append((right_start, compare_end))

            for seg_start, seg_end in segments:
                if seg_end <= seg_start:
                    continue
                hover_template = f"{hover_text}<extra></extra>"
                fig.add_trace(
                    go.Scatter(
                        x=[seg_start, seg_end, seg_end, seg_start, seg_start],
                        y=[y_bottom, y_bottom, y_top, y_top, y_bottom],
                        mode="none",
                        fill="toself",
                        fillcolor="rgba(152, 194, 220, 0.02)",
                        hoveron="fills",
                        name="",
                        hovertemplate=hover_template,
                        hoverlabel=_hoverlabel_style(),
                        showlegend=False,
                        cliponaxis=False,
                    )
                )

    fig.update_layout(

        title=title,

        xaxis=dict(title=None, tickmode="linear", dtick=1, tickangle=-45),

        yaxis=dict(

            title="Project",

            autorange="reversed",

            tickmode="array",

            tickvals=y_positions,

            ticktext=y_labels,

        ),

        hoverlabel=dict(

            align="left",

            bgcolor="rgba(0, 0, 0, 0)",

            bordercolor="rgba(0, 0, 0, 0)",

            font=dict(family="Inter, 'Segoe UI', sans-serif", size=12),

            namelength=0,

        ),

        template=plotly_template(),

        height=max(280, 100 + 28 * len(runs)),

        showlegend=False,

    )

    return fig

def project_color_map(data: DashboardData) -> Dict[str, str]:

    palette = PROJECT_COLOR_POOL if PROJECT_COLOR_POOL else ("#1f77b4",)

    seen = set()

    ordered: List[str] = []

    sources: List[List[str]] = []

    if getattr(data, "projects", None):

        sources.append([str(p) for p in data.projects])

    spend_matrix = getattr(data, "spend_matrix", None)

    if spend_matrix is not None and "Project" in spend_matrix.columns:

        sources.append(spend_matrix["Project"].astype(str).tolist())

    for entries in sources:

        for proj in entries:

            if proj and proj not in seen:

                ordered.append(proj)

                seen.add(proj)

    if not ordered:

        ordered = ["Project"]

    mapping: Dict[str, str] = {}

    for idx, proj in enumerate(ordered):

        mapping[proj] = palette[idx % len(palette)]

    return mapping

def project_schedule_area_chart(

    data: DashboardData,

    selection: ScenarioSelection,

    *,

    title: str,

    color_map: Dict[str, str],

) -> Optional[go.Figure]:

    if not selection.code:

        return None

    runs = extract_project_runs(data, selection.code)

    if not runs:

        return None

    years = data.years

    if not years:

        return None

    fig = go.Figure()

    base_palette = PROJECT_COLOR_POOL if PROJECT_COLOR_POOL else ("#1f77b4",)

    tick_step = 1
    if len(years) > 40:
        tick_step = 4
    elif len(years) > 25:
        tick_step = 2
    tick_vals = years[::tick_step]
    tick_text = [str(v) for v in tick_vals]

    for idx, run in enumerate(runs):

        values = [max(0.0, float(v)) for v in run.values]

        if not any(values):

            continue

        color = color_map.get(run.project)

        if color is None:

            color = base_palette[idx % len(base_palette)]

        total_spend = max(0.0, float(run.total_spend))

        customdata = [
            [year, value, total_spend]
            for year, value in zip(years, values)
        ]

        fig.add_trace(

            go.Scatter(

                x=years,

                y=values,

                mode="lines",

                line=dict(width=0, color=color),

                stackgroup="schedule",

                hoveron="fills",

                fillcolor=color,

                name=run.project,

                legendgroup=run.project,

                customdata=customdata,

                hovertemplate=(

                    "<b>%{fullData.name}</b><br>"
                    "Total spend: %{customdata[2]:,.0f} $m<br>"
                    "Year: %{customdata[0]}<br>"
                    "Annual spend: %{customdata[1]:,.0f} $m<extra></extra>"
                ),

                hoverlabel=dict(namelength=-1),

            )

        )

    if not fig.data:

        return None

    min_year = years[0]

    max_year = years[-1]

    fig.update_layout(

        title=title,

        xaxis=dict(

            title=None,

            tickmode="array",

            tickvals=tick_vals,

            ticktext=tick_text,

            tickangle=-35,

            range=[min_year, max_year],

            showgrid=False,

            zeroline=False,

        ),

        yaxis=dict(title="Annual spend ($m)", rangemode="tozero", showgrid=True),

        hovermode="closest",

        legend=dict(

            title="Project",

            orientation="v",

            yanchor="top",

            y=1.0,

            xanchor="left",

            x=1.02,

            tracegroupgap=4,

            font=dict(size=10),

        ),

        margin=dict(l=40, r=220, t=60, b=80),

        template=plotly_template(),

    )

    return fig

def market_capacity_indicator(data: DashboardData, selection: ScenarioSelection) -> Optional[go.Figure]:

    series = build_timeseries(data, selection)

    if series is None:

        return None

    totals = series[["Year", "Spend"]].copy()

    totals["SpendB"] = totals["Spend"] / 1000.0

    runs = extract_project_runs(data, selection.code) if selection.code else []

    max_label_len = max((len(run.project) for run in runs), default=0)

    left_margin = max(48, min(320, 60 + int(max_label_len * 5.5)))

    years = totals["Year"].tolist()

    colors: List[str] = []

    text_values: List[str] = []

    text_colors: List[str] = []

    hover_labels: List[str] = []

    for year, value in totals[["Year", "SpendB"]].itertuples(index=False):

        if value <= 0.0:

            color = CAPACITY_ZERO

            descriptor = "No spend recorded"

            display_value = ""

            text_color = color

        elif value >= 3.0:

            color = CAPACITY_RED

            descriptor = "High pressure (>= $3.0B)"

            display_value = f"{value:.1f}"

            text_color = "#F9FAFB"

        elif value <= 2.0:

            color = CAPACITY_GREEN

            descriptor = "Comfortable (<= $2.0B)"

            display_value = f"{value:.1f}"

            text_color = "#0F172A"

        else:

            color = CAPACITY_AMBER

            descriptor = "Watch zone ($2.0B-$3.0B)"

            display_value = f"{value:.1f}"

            text_color = "#0F172A"

        colors.append(color)

        hover_labels.append(f"Year: {year}<br>Total spend: {value:.1f} $B<br>Status: {descriptor}")

        text_values.append(display_value)

        text_colors.append(text_color)

    gap_ratio = 0.05

    bar_width = max(0.0, 1.0 - gap_ratio)

    fig = go.Figure(

        data=go.Bar(

            x=years,

            y=[1.0] * len(years),

            width=bar_width if years else None,

            marker=dict(color=colors, line=dict(color="#0F172A", width=1.0)),

            hovertext=hover_labels,

            hovertemplate="%{hovertext}<extra></extra>",

            text=text_values,

            textposition="inside",

            texttemplate="%{text}",

        )

    )

    fig.update_traces(textfont=dict(size=14), hoverlabel=_hoverlabel_style())
    if fig.data:
        fig.data[0].textfont.color = text_colors

    fig.update_layout(bargap=gap_ratio, bargroupgap=0.0)

    tick0 = float(totals["Year"].min()) if not totals.empty else 0.0

    last_year = float(totals["Year"].max()) if not totals.empty else tick0

    fig.update_layout(

        title="Market capacity indicator - total spend by year",

        height=110,

        margin=dict(l=left_margin, r=18, t=28, b=4),

        autosize=False,

        xaxis=dict(

            tickmode="linear",

            dtick=1,

            tick0=tick0,

            range=[tick0 - 0.5, last_year + 0.5],

            showticklabels=False,

            showgrid=False,

            zeroline=False,

        ),

        yaxis=dict(visible=False, range=[0, 1], fixedrange=True),

        template=plotly_template(),

        showlegend=False,

    )

    return fig



def summarize_selection(selection: ScenarioSelection) -> pd.DataFrame:

    data = {

        "Metric": [

            "Scenario code",

            "Profile",

            "Mode",

            "Envelope ($m)",

            "Buffer / Cash uplift ($m)",

            "Objective dimension",

            "Cache file",

        ],

        selection.name: [

            selection.code or "",

            selection.profile or DEFAULT_PROFILE_LABEL,

            MODE_LABELS.get(selection.mode, selection.mode.title()),

            selection.metadata.get("Envelope") if selection.metadata else "",

            selection.metadata.get("Buffer") if selection.metadata else "",

            selection.metadata.get("ObjectiveDim") if selection.metadata else "",

            selection.metadata.get("CacheFile") if selection.metadata else "",

        ],

    }

    return pd.DataFrame(data)


# ---------------------------------------------------------------------
# Regional investment view


def _color_to_rgb_tuple(color: str) -> tuple[float, float, float]:
    color = (color or '').strip()
    if not color:
        return 0.0, 0.0, 0.0
    if color.startswith('#'):
        r, g, b = plc.hex_to_rgb(color)
        return float(r), float(g), float(b)
    if color.startswith('rgba') or color.startswith('rgb'):
        start = color.find('(') + 1
        end = color.rfind(')')
        parts = [p.strip() for p in color[start:end].split(',')]
        if len(parts) >= 3:
            return float(parts[0]), float(parts[1]), float(parts[2])
    # Fallback – let Plotly help normalise then strip labels
    try:
        converted = plc.unlabel_rgb(plc.label_rgb(color))
        return tuple(float(c) for c in converted[:3])
    except Exception:
        return 0.0, 0.0, 0.0


def _colorscale_with_opacity(colorscale: Any, opacity: float) -> list[list[Any]]:
    try:
        resolved = plc.get_colorscale(colorscale)
    except Exception:
        resolved = colorscale if isinstance(colorscale, (list, tuple)) else plc.get_colorscale('YlOrRd')
    opacity = float(np.clip(opacity, 0.0, 1.0))
    adjusted = []
    for stop, color in resolved:
        r, g, b = _color_to_rgb_tuple(color)
        adjusted.append([float(stop), f'rgba({int(r)}, {int(g)}, {int(b)}, {opacity:.3f})'])
    return adjusted
# ---------------------------------------------------------------------

def _format_percentage(value: float, decimals: int = 1, *, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    if signed:
        return f"{value:+.{decimals}f}%"
    return f"{value:.{decimals}f}%"

def _format_currency_nzd(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"${value:,.0f}"

def _format_currency_compact(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.1f}m"
    if abs_value >= 1_000:
        return f"${value / 1_000:.1f}k"
    return f"${value:,.0f}"

REGION_METRIC_CONFIG: Dict[str, Dict[str, Any]] = {
    "Share_Cum": {
        "label": "Cumulative share vs national",
        "description": "Share of cumulative spend allocated to each region compared with the national total.",
        "type": "sequential",
        "colorscale": "YlOrRd",
        "multiplier": 100.0,
        "colorbar": "Share (%)",
        "ticksuffix": "%",
        "tickformat": ".1f",
        "force_zero_min": True,
        "formatter": lambda v: _format_percentage(v, 1),
        "sort": "desc",
    },
    "Share_Year": {
        "label": "Annual share vs national",
        "description": "Share of annual spend in the selected year compared with the national total.",
        "type": "sequential",
        "colorscale": "OrRd",
        "multiplier": 100.0,
        "colorbar": "Share (%)",
        "ticksuffix": "%",
        "tickformat": ".1f",
        "force_zero_min": True,
        "formatter": lambda v: _format_percentage(v, 1),
        "sort": "desc",
    },
    "PerCap_Cum": {
        "label": "Cumulative spend per capita",
        "description": "Cumulative spend per resident since the start of the programme.",
        "type": "sequential",
        "colorscale": "Purples",
        "multiplier": 1_000_000.0,
        "colorbar": "$ per person",
        "tickformat": ",.0f",
        "force_zero_min": True,
        "formatter": _format_currency_nzd,
        "table_label": "Per-cap cum",
        "sort": "desc",
    },
    "PerCap_Year": {
        "label": "Annual spend per capita",
        "description": "Annual spend in the selected year per resident.",
        "type": "sequential",
        "colorscale": "Blues",
        "multiplier": 1_000_000.0,
        "colorbar": "$ per person",
        "tickformat": ",.0f",
        "force_zero_min": True,
        "formatter": _format_currency_nzd,
        "table_label": "Per-cap annual",
        "sort": "desc",
    },
    "OU_vs_Pop": {
        "label": "Over / under vs population share",
        "description": "Difference between cumulative spend share and population share (percentage points).",
        "type": "diverging",
        "colorscale": "RdBu_r",
        "multiplier": 100.0,
        "colorbar": "Δ vs pop (pp)",
        "ticksuffix": " pp",
        "tickformat": ".1f",
        "formatter": lambda v: _format_percentage(v, 1, signed=True),
        "sort": "abs_desc",
    },
    "OU_vs_GDP": {
        "label": "Over / under vs GDP share",
        "description": "Difference between cumulative spend share and GDP share (percentage points).",
        "type": "diverging",
        "colorscale": "RdBu_r",
        "multiplier": 100.0,
        "colorbar": "Δ vs GDP (pp)",
        "ticksuffix": " pp",
        "tickformat": ".1f",
        "formatter": lambda v: _format_percentage(v, 1, signed=True),
        "sort": "abs_desc",
    },
    "Ramp_Rate": {
        "label": "Ramp rate (Δ cumulative share)",
        "description": "Year-on-year change in cumulative share (percentage points).",
        "type": "diverging",
        "colorscale": "PuOr",
        "multiplier": 100.0,
        "colorbar": "Δ share (pp)",
        "ticksuffix": " pp",
        "tickformat": ".1f",
        "formatter": lambda v: _format_percentage(v, 1, signed=True),
        "sort": "abs_desc",
    },
}

REGION_METRIC_ORDER = list(REGION_METRIC_CONFIG.keys())

def _scaled_region_metric(df: pd.DataFrame, metric_key: str) -> pd.Series:
    if metric_key not in df.columns:
        return pd.Series(np.nan, index=df.index)
    series = pd.to_numeric(df[metric_key], errors="coerce")
    config = REGION_METRIC_CONFIG.get(metric_key, {})
    multiplier = float(config.get("multiplier", 1.0))
    offset = float(config.get("offset", 0.0))
    return series * multiplier + offset

def _format_region_metric_value(metric_key: str, value: float) -> str:
    config = REGION_METRIC_CONFIG.get(metric_key, {})
    formatter = config.get("formatter")
    if formatter is not None:
        try:
            return formatter(value)
        except Exception:
            pass
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:,.2f}"

def build_region_map_figure(
    map_df: pd.DataFrame,
    metric_key: str,
    *,
    geojson: Dict[str, Any],
    scenario_label: str,
    year: int,
    show_borders: bool,
    fill_opacity: float,
    name_field: str,
) -> go.Figure:
    config = REGION_METRIC_CONFIG[metric_key]
    values = map_df["_metric_value"].to_numpy(dtype=float)
    is_diverging = config.get("type") == "diverging"
    if is_diverging:
        max_abs = float(np.nanmax(np.abs(values))) if values.size else 0.0
        if not np.isfinite(max_abs) or max_abs == 0.0:
            max_abs = 1.0
        zmin, zmax = -max_abs, max_abs
    else:
        zmin = 0.0 if config.get("force_zero_min", False) else float(np.nanmin(values)) if values.size else 0.0
        if not np.isfinite(zmin):
            zmin = 0.0
        zmax = float(np.nanmax(values)) if values.size else 0.0
        if not np.isfinite(zmax) or np.isclose(zmin, zmax):
            zmax = zmin + max(1.0, abs(zmin) * 0.1 + 1.0)
    dark_mode = is_dark_mode()
    line_color = "#1f2937" if dark_mode else "#0b1120"
    coastline_color = "#f1f5f9" if dark_mode else "#1f2937"
    land_color = "rgba(148, 163, 184, 0.32)" if dark_mode else "rgba(148, 163, 184, 0.18)"
    marker_line_width = 1.2 if show_borders else 0.3
    opacity = float(np.clip(fill_opacity, 0.05, 1.0))
    effective_colorscale = _colorscale_with_opacity(config.get("colorscale", "YlOrRd"), opacity)
    map_df = map_df.copy()
    map_df["_metric_display"] = map_df["_metric_value"].apply(lambda v: _format_region_metric_value(metric_key, v))
    map_df["_share_cum_fmt"] = (map_df["Share_Cum"] * 100).map(lambda v: _format_percentage(v, 1))
    map_df["_share_year_fmt"] = (map_df["Share_Year"] * 100).map(lambda v: _format_percentage(v, 1))
    map_df["_percap_cum_fmt"] = (map_df["PerCap_Cum"] * 1_000_000).map(_format_currency_compact)
    map_df["_percap_year_fmt"] = (map_df["PerCap_Year"] * 1_000_000).map(_format_currency_compact)
    map_df["_population_fmt"] = map_df["population"].map(lambda v: f"{v:,.0f}" if np.isfinite(v) else "-")
    map_df["_year_str"] = map_df["Year"].astype(int).astype(str)
    customdata = map_df[
        [
            "region",
            "_year_str",
            "_metric_display",
            "_share_cum_fmt",
            "_share_year_fmt",
            "_percap_cum_fmt",
            "_percap_year_fmt",
            "_population_fmt",
        ]
    ].to_numpy()
    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        "Year: %{customdata[1]}<br>"
        f"{config['label']}: %{customdata[2]}<br>"
        "Cumulative share: %{customdata[3]}<br>"
        "Annual share: %{customdata[4]}<br>"
        "Per-capita cumulative: %{customdata[5]}<br>"
        "Per-capita annual: %{customdata[6]}<br>"
        "Population: %{customdata[7]}<extra></extra>"
    )
    colorbar = dict(
        title=dict(text=config.get("colorbar", config["label"]), side="right"),
        ticksuffix=config.get("ticksuffix"),
        tickformat=config.get("tickformat"),
        len=0.7,
        thickness=16,
        x=1.05,
        y=0.5,
        xpad=12,
        outlinewidth=0,
        bgcolor="rgba(0,0,0,0)",
    )
    trace = go.Choropleth(
        geojson=geojson,
        featureidkey=f"properties.{name_field}",
        locations=map_df["join_key"],
        z=map_df["_metric_value"],
        zmin=zmin,
        zmax=zmax,
        colorscale=effective_colorscale,
        reversescale=bool(config.get("reversescale", False)),
        marker=dict(line=dict(color=line_color, width=marker_line_width)),
        colorbar=colorbar,
        customdata=customdata,
        hovertemplate=hovertemplate,
    )
    fig = go.Figure(data=[trace])
    fig.update_geos(
        visible=False,
        projection_type="mercator",
        projection_scale=2.0,
        center=dict(lat=-41.0, lon=173.0),
        lataxis=dict(range=[-58.0, -24.0]),
        lonaxis=dict(range=[152.0, 195.0]),
        showcountries=False,
        showcoastlines=True,
        coastlinecolor=coastline_color,
        coastlinewidth=1.5,
        showland=True,
        landcolor=land_color,
    )
    fig.update_layout(
        margin=dict(l=0, r=140, t=60, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=520,
        title=dict(
            text=f"{config['label']} — {scenario_label} ({year})",
            x=0.02,
            y=0.96,
            xanchor="left",
            yanchor="top",
            font=dict(size=16),
            pad=dict(b=12),
        ),
    )
    return fig

def build_region_summary_table(df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    config = REGION_METRIC_CONFIG[metric_key]
    table = df.copy()
    table["_metric_value"] = _scaled_region_metric(table, metric_key)
    table["_metric_display"] = table["_metric_value"].apply(lambda v: _format_region_metric_value(metric_key, v))
    table["_share_cum_fmt"] = (table["Share_Cum"] * 100).map(lambda v: _format_percentage(v, 1))
    table["_share_year_fmt"] = (table["Share_Year"] * 100).map(lambda v: _format_percentage(v, 1))
    table["_percap_cum_fmt"] = (table["PerCap_Cum"] * 1_000_000).map(_format_currency_compact)
    table["_percap_year_fmt"] = (table["PerCap_Year"] * 1_000_000).map(_format_currency_compact)
    sort_mode = config.get("sort", "desc")
    if sort_mode == "asc":
        table = table.sort_values("_metric_value", ascending=True)
    elif sort_mode == "abs_desc":
        table = table.assign(_abs=table["_metric_value"].abs()).sort_values("_abs", ascending=False).drop(columns="_abs")
    else:
        table = table.sort_values("_metric_value", ascending=False)
    columns = {
        "region": "Region",
        "_metric_display": config.get("table_label", config["label"]),
        "_share_cum_fmt": "Share (cum)",
        "_share_year_fmt": "Share (annual)",
        "_percap_cum_fmt": "Per-cap cum",
        "_percap_year_fmt": "Per-cap annual",
    }
    formatted = table[list(columns.keys())].rename(columns=columns)
    formatted.reset_index(drop=True, inplace=True)
    return formatted.head(10)

def render_region_map(
    df_year: pd.DataFrame,
    metric_key: str,
    *,
    scenario_label: str,
    year: int,
    show_borders: bool,
    fill_opacity: float,
) -> None:
    geojson = fetch_region_geojson()
    name_field = get_geojson_name_field(geojson)
    valid_regions = set()
    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        value = props.get(name_field)
        if value is None:
            continue
        text_value = str(value).strip()
        if text_value:
            valid_regions.add(text_value)
    df_year = df_year.copy()
    region_mapping = load_region_mapping()
    catalog, _, _ = region_baselines(region_mapping)
    baseline_keys = catalog[["region", "join_key", "population", "gdp_per_capita"]].drop_duplicates(subset=["region"]).copy()
    df_year = baseline_keys.merge(
        df_year,
        on=["region", "join_key"],
        how="left",
        suffixes=("", "_data"),
    )
    if "population_data" in df_year.columns:
        df_year["population"] = df_year["population_data"].fillna(df_year["population"])
    if "gdp_per_capita_data" in df_year.columns:
        df_year["gdp_per_capita"] = df_year["gdp_per_capita_data"].fillna(df_year["gdp_per_capita"])
    df_year.drop(columns=[c for c in ("population_data", "gdp_per_capita_data") if c in df_year.columns], inplace=True)
    df_year["Year"] = df_year["Year"].fillna(int(year))
    numeric_cols = df_year.select_dtypes(include=[np.number]).columns
    df_year[numeric_cols] = df_year[numeric_cols].fillna(0.0)
    df_year["join_key"] = df_year["join_key"].astype(str).str.strip()
    map_df = df_year[df_year["join_key"].isin(valid_regions) & df_year["join_key"].str.len() > 0].copy()
    map_df["join_key"] = map_df["join_key"].astype(str).str.strip()
    if map_df.empty or map_df[metric_key].dropna().empty:
        st.info("No mapped regional spend for the selected inputs.")
        summary = build_region_summary_table(df_year, metric_key)
        st.dataframe(summary, hide_index=True, use_container_width=True)
        return
    map_df["_metric_value"] = _scaled_region_metric(map_df, metric_key)
    if map_df["_metric_value"].dropna().empty:
        st.info("Selected metric has no values for this year.")
        summary = build_region_summary_table(df_year, metric_key)
        st.dataframe(summary, hide_index=True, use_container_width=True)
        return
    map_col, table_col = st.columns([2, 1])
    with map_col:
        fig = build_region_map_figure(
            map_df,
            metric_key,
            geojson=geojson,
            scenario_label=scenario_label,
            year=year,
            show_borders=show_borders,
            fill_opacity=fill_opacity,
            name_field=name_field,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False, "doubleClick": "reset"})
    with table_col:
        summary = build_region_summary_table(df_year, metric_key)
        st.markdown(f"**Top regions ({REGION_METRIC_CONFIG[metric_key]['label']})**")
        st.dataframe(summary, hide_index=True, use_container_width=True, height=420)
        if (df_year["region"] == "Unmapped").any():
            st.caption("Projects without a region mapping are grouped under 'Unmapped'.")

def render_region_map_controls(metrics_df: pd.DataFrame, scenario_label: str) -> None:
    available_years = sorted(int(y) for y in metrics_df["Year"].dropna().unique())
    if not available_years:
        st.info("No spend data available for the selected scenario.")
        return
    default_year = st.session_state.get("region_metric_year", available_years[0])
    if default_year not in available_years:
        default_year = available_years[0]
    col_year, col_metric, col_opts = st.columns([2, 1, 1])
    with col_year:
        selected_year = st.slider(
            "Financial year",
            min_value=int(available_years[0]),
            max_value=int(available_years[-1]),
            value=int(default_year),
            step=1,
            key="region_metric_year_slider",
        )
    metric_options = REGION_METRIC_ORDER
    default_metric = st.session_state.get("region_metric_key", metric_options[0])
    if default_metric not in metric_options:
        default_metric = metric_options[0]
    with col_metric:
        metric_key = st.selectbox(
            "Metric",
            metric_options,
            index=metric_options.index(default_metric),
            format_func=lambda key: REGION_METRIC_CONFIG[key]["label"],
            key="region_metric_select",
        )
    with col_opts:
        show_borders = st.checkbox(
            "Show borders",
            value=st.session_state.get("region_metric_show_borders", True),
            key="region_metric_borders_checkbox",
        )
        fill_opacity = st.slider(
            "Fill opacity",
            min_value=0.2,
            max_value=1.0,
            value=float(st.session_state.get("region_metric_opacity", 0.85)),
            step=0.05,
            key="region_metric_opacity_slider",
        )
    st.session_state["region_metric_year"] = int(selected_year)
    st.session_state["region_metric_key"] = metric_key
    st.session_state["region_metric_show_borders"] = bool(show_borders)
    st.session_state["region_metric_opacity"] = float(fill_opacity)
    config = REGION_METRIC_CONFIG[metric_key]
    st.caption(config.get("description", ""))
    year_df = metrics_df[metrics_df["Year"] == int(selected_year)].copy()
    if year_df.empty:
        st.info(f"No regional spend recorded in {selected_year}.")
        return
    render_region_map(
        year_df,
        metric_key,
        scenario_label=scenario_label,
        year=int(selected_year),
        show_borders=bool(show_borders),
        fill_opacity=float(fill_opacity),
    )

def render_region_section(
    data: DashboardData,
    *,
    opt_selection,
    comp_selection,
    opt_label: str,
    cmp_label: str,
    cache_signature: tuple | None = None,
) -> None:
    with st.container():
        st.subheader("Regional investment overview")
        scenario_options: Dict[str, Any] = {}
        if getattr(opt_selection, "code", None):
            scenario_options[opt_label] = opt_selection
        if getattr(comp_selection, "code", None):
            scenario_options[cmp_label] = comp_selection
        if not scenario_options:
            st.info("Select a scenario to view the regional investment map.")
            return
        labels = list(scenario_options.keys())
        default_label = st.session_state.get("region_metric_scenario", labels[0])
        if default_label not in scenario_options:
            default_label = labels[0]
        selected_label = st.selectbox(
            "Scenario for regional map",
            labels,
            index=labels.index(default_label),
            key="region_metric_scenario_select",
        )
        st.session_state["region_metric_scenario"] = selected_label
        selection = scenario_options[selected_label]
        scenario_code = getattr(selection, "code", None)
        if not scenario_code:
            st.info("Select a scenario with an available cache entry to view the regional investment map.")
            return
        cache_bucket = st.session_state.setdefault("_region_metrics_cache", {})
        cache_key = (cache_signature or "default", scenario_code)
        metrics_df = cache_bucket.get(cache_key)
        if metrics_df is None:
            try:
                metrics_df = compute_region_metrics(data, scenario_code)
            except Exception as exc:
                st.warning(f"Unable to compute regional metrics for {selected_label}: {exc}")
                return
            cache_bucket[cache_key] = metrics_df
        render_region_map_controls(metrics_df, selected_label)

def main() -> None:

    st.set_page_config(page_title="Capital Programme Optimiser", layout="wide")


    st.title("Capital Programme Optimiser")

    if "last_run_summary" in st.session_state:

        info = st.session_state.pop("last_run_summary")

        folder_info = info.get("folder", {})

        elapsed = info.get("elapsed_seconds", 0.0)

        st.success(

            f"Optimiser finished for {folder_info.get('name', 'scenario')} in {elapsed:.1f}s "

            f"({len(info.get('output_files', []))} files saved)."

        )

    if "last_created_folder_name" in st.session_state:

        created_name = st.session_state.pop("last_created_folder_name")

        st.success(f"Scenario folder {created_name} created.")

    settings = load_settings()

    preset_root, saved_root = scenario_utils.ensure_scenario_roots(settings)

    scenario_folders = scenario_utils.list_scenario_folders(settings)

    cache_options = [

        {

            "label": "Configured cache (settings.yaml)",

            "path": settings.cache_dir(),

            "folder": None,

        }

    ]

    for folder in scenario_folders:

        label = f"{folder.kind.title()} / {folder.name}"

        if folder.is_default:

            label += " (default)"

        cache_options.append(

            {

                "label": label,

                "path": folder.path,

                "folder": folder,

            }

        )

    labels = [opt["label"] for opt in cache_options]

    default_index = 0

    if st.session_state.get("selected_cache_label") in labels:

        default_index = labels.index(st.session_state["selected_cache_label"])

    st.sidebar.header("Scenario cache")

    selected_label = st.sidebar.selectbox(

        "Choose scenario bundle",

        labels,

        index=default_index,

    )

    st.session_state["selected_cache_label"] = selected_label

    selected_option = cache_options[labels.index(selected_label)]

    selected_path = Path(selected_option["path"])

    st.sidebar.caption(f"Active cache: {selected_path}")

    manifest = None

    folder_meta = selected_option.get("folder")

    if folder_meta is not None:

        manifest = folder_meta.metadata or scenario_utils.load_manifest(folder_meta.path)

    if manifest and manifest.get("last_run"):

        last_run = manifest["last_run"]

        st.sidebar.write(

            f"Last run finished: {last_run.get('finished_at', 'unknown')}"

        )

        if last_run.get("output_files"):

            st.sidebar.write(f"Files: {len(last_run['output_files'])}")

    else:

        st.sidebar.caption("No recorded manifest details yet for this folder.")

    st.sidebar.divider()

    st.sidebar.header("Configuration")

    st.sidebar.write(f"Dashboard output folder: {settings.dashboard_output_dir()}")

    if st.sidebar.button("Reload cache data"):

        load_dashboard_data.clear()

        st.rerun()

    cache_sig = _cache_signature(selected_path)

    with st.spinner(f"Loading dashboard cache from {selected_path}..."):

        try:

            data = load_dashboard_data(
                str(selected_path), cache_sig, LOAD_DATA_SCHEMA_VERSION
            )

        except Exception as exc:

            st.error(f"Unable to load scenario cache from {selected_path}: {exc}")

            st.stop()

    with st.expander("Scenario filters", expanded=True):
    
        opt_col, cmp_col = st.columns(2)
    
        with opt_col:
    
            st.subheader("Optimised scenario")
    
            opt_profiles = profile_options(data) or [DEFAULT_PROFILE_LABEL]
    
            opt_default_index = opt_profiles.index(DEFAULT_PROFILE_LABEL) if DEFAULT_PROFILE_LABEL in opt_profiles else 0
    
            selected_opt_profile = st.selectbox(
    
                "Optimised profile",
    
                opt_profiles,
    
                index=opt_default_index,
    
                key="opt_profile_select",
    
            )
    
            opt_selection = scenario_selector(
    
                name="Optimised",
    
                data=data,
    
                settings=settings,
    
                prefer_comparison=False,
    
                key_prefix="opt",
    
                profile_name=selected_opt_profile,
    
            )
    
        with cmp_col:
    
            st.subheader("Comparison scenario")
    
            cmp_profiles = profile_options(data) or [DEFAULT_PROFILE_LABEL]
    
            cmp_default_index = next((i for i, label in enumerate(cmp_profiles) if label.lower() == "ncor"), None)
    
            if cmp_default_index is None:
    
                cmp_default_index = cmp_profiles.index(DEFAULT_PROFILE_LABEL) if DEFAULT_PROFILE_LABEL in cmp_profiles else 0
    
            selected_cmp_profile = st.selectbox(
    
                "Comparison profile",
    
                cmp_profiles,
    
                index=cmp_default_index,
    
                key="cmp_profile_select",
    
            )
    
            comp_selection = scenario_selector(
    
                name="Comparison",
    
                data=data,
    
                settings=settings,
    
                prefer_comparison=True,
    
                key_prefix="cmp",
    
                profile_name=selected_cmp_profile,
    
            )
    
    opt_series = build_timeseries(data, opt_selection)

    cmp_series = build_timeseries(data, comp_selection)

    opt_label = opt_selection.name or "Optimised"
    cmp_label = comp_selection.name or "Comparison"

    render_region_section(
        data,
        opt_selection=opt_selection,
        comp_selection=comp_selection,
        opt_label=opt_label,
        cmp_label=cmp_label,
        cache_signature=cache_sig,
    )

    export_tables: Dict[str, pd.DataFrame] = {}

    project_colors = project_color_map(data)

    schedule_opt_fig = project_schedule_area_chart(

        data,

        opt_selection,

        title="Project schedule - optimised",

        color_map=project_colors,

    )

    schedule_cmp_fig = project_schedule_area_chart(

        data,

        comp_selection,

        title="Project schedule - comparison",

        color_map=project_colors,

    )

    summary_horizon_years = int(st.session_state.get("npv_horizon_selection", 50))

    npv_summary_label = npv_context_label(
        data,
        opt_selection,
        comp_selection,
        horizon_override=summary_horizon_years,
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    stats_opt = (
        scenario_metrics(
            opt_series, start_year=data.start_fy, horizon_years=summary_horizon_years
        )
        if opt_series is not None
        else None
    )

    stats_cmp = (
        scenario_metrics(
            cmp_series, start_year=data.start_fy, horizon_years=summary_horizon_years
        )
        if cmp_series is not None
        else None
    )

    with summary_col1:

        if stats_opt is not None:

            st.metric("Optimised - total spend", format_currency(stats_opt["total_spend"]))

            st.metric(f"Optimised total NPV benefit ({npv_summary_label})", format_currency(stats_opt["total_pv"]))

        else:

            st.warning("No optimised scenario data available for the current selection.")

    with summary_col2:

        if stats_cmp is not None:

            st.metric("Comparison - total spend", format_currency(stats_cmp["total_spend"]))

            st.metric(f"Comparison total NPV benefit ({npv_summary_label})", format_currency(stats_cmp["total_pv"]))

        else:

            st.info("Select a comparison scenario to enable side-by-side charts.")

    with summary_col3:

        if stats_opt is not None and stats_cmp is not None:

            st.metric("Delta - total spend", format_currency(stats_opt["total_spend"] - stats_cmp["total_spend"]))

            st.metric(f"Delta total NPV benefit ({npv_summary_label})", format_currency(stats_opt["total_pv"] - stats_cmp["total_pv"]))

        else:

            st.info("Viewing delta requires both scenarios.")

    npv_horizon_options = [50, 35]

    selected_npv_horizon = int(
        st.session_state.get("npv_horizon_selection", npv_horizon_options[0])
    )

    eff_fig = efficiency_chart(opt_series, cmp_series, opt_selection, comp_selection)

    st.plotly_chart(eff_fig, use_container_width=True)

    efficiency_export = prepare_efficiency_export(
        data,
        opt_series,
        cmp_series,
        opt_selection,
        comp_selection,
    )
    if efficiency_export is not None:
        export_tables["Cumulative spend vs benefit"] = efficiency_export

    horizon_index = (
        npv_horizon_options.index(selected_npv_horizon)
        if selected_npv_horizon in npv_horizon_options
        else 0
    )

    selected_npv_horizon = int(
        st.radio(
            "NPV horizon (years)",
            npv_horizon_options,
            index=horizon_index,
            horizontal=True,
            key="npv_horizon_selection",
        )
    )

    pv_opt_horizon = (
        pv_by_dimension(data, opt_selection, horizon_years=selected_npv_horizon)
        if opt_selection and opt_selection.code
        else None
    )
    pv_cmp_horizon = (
        pv_by_dimension(data, comp_selection, horizon_years=selected_npv_horizon)
        if comp_selection and comp_selection.code
        else None
    )

    pv_col1, pv_col2 = st.columns(2)

    with pv_col1:
        st.plotly_chart(
            benefit_waterfall_chart(
                data, opt_selection, comp_selection, horizon_years=selected_npv_horizon
            ),
            use_container_width=True,
        )

        waterfall_export = prepare_waterfall_export(
            data,
            opt_selection,
            comp_selection,
            horizon_years=selected_npv_horizon,
            pv_opt=pv_opt_horizon,
            pv_cmp=pv_cmp_horizon,
        )
        if waterfall_export is not None:
            export_tables["NPV Waterfall"] = waterfall_export

    with pv_col2:
        st.plotly_chart(
            benefit_bridge_chart(
                data, opt_selection, comp_selection, horizon_years=selected_npv_horizon
            ),
            use_container_width=True,
        )

        bridge_export = prepare_bridge_export(
            data,
            opt_selection,
            comp_selection,
            horizon_years=selected_npv_horizon,
            pv_opt=pv_opt_horizon,
            pv_cmp=pv_cmp_horizon,
        )
        if bridge_export is not None:
            export_tables["NPV Bridge"] = bridge_export

    opt_dim_pivot = dimension_timeseries(data, opt_selection)
    cmp_dim_pivot = dimension_timeseries(data, comp_selection)

    available_dimension_labels = [
        str(dim)
        for dim in getattr(data, "dims", [])
        if str(dim).strip().lower() != "total"
        and (
            (opt_dim_pivot is not None and dim in opt_dim_pivot.columns)
            or (cmp_dim_pivot is not None and dim in cmp_dim_pivot.columns)
        )
    ]
    toggle_col1, toggle_col2 = st.columns(2)

    with toggle_col1:
        show_cumulative_benefits = st.checkbox(
            "Show cumulative dimension benefits",
            value=st.session_state.get("show_cumulative_dimension_benefits", False),
            key="show_cumulative_dimension_benefits",
        )

    with toggle_col2:
        compare_dimension_overlays = st.checkbox(
            "Compare dimension overlays",
            value=st.session_state.get("compare_dimension_overlays", False),
            key="compare_dimension_overlays",
        )

    if compare_dimension_overlays:
        if not available_dimension_labels:
            st.info("No dimension data available for comparison.")
        else:
            default_dims = available_dimension_labels[:1]
            selected_dims = st.multiselect(
                "Dimensions to display",
                available_dimension_labels,
                default=default_dims,
                key="dimension_overlay_selection",
            )

            if not selected_dims:
                st.info("Select at least one dimension to compare.")
            else:
                comparison_fig = benefit_dimension_overlay_chart(
                    data,
                    opt_selection,
                    comp_selection,
                    cumulative=show_cumulative_benefits,
                    dimensions=selected_dims,
                    opt_pivot=opt_dim_pivot,
                    cmp_pivot=cmp_dim_pivot,
                )

                if comparison_fig is not None:
                    st.plotly_chart(comparison_fig, use_container_width=True)

                    overlay_export = prepare_dimension_overlay_export(
                        data.years,
                        opt_dim_pivot,
                        cmp_dim_pivot,
                        selected_dims,
                        show_cumulative_benefits,
                    )
                    if overlay_export is not None:
                        export_tables["Dimension overlay comparison"] = overlay_export
                else:
                    st.info("No overlapping dimension data available for comparison.")
    else:
        dim_col1, dim_col2 = st.columns(2)

        with dim_col1:
            opt_dim_fig = benefit_dimension_chart(
                data,
                opt_selection,
                title="Optimised benefit mix by dimension",
                cumulative=show_cumulative_benefits,
                pivot=opt_dim_pivot,
            )

            if opt_dim_fig is not None:
                st.plotly_chart(opt_dim_fig, use_container_width=True)

                opt_dim_export = prepare_dimension_chart_export(opt_dim_pivot, show_cumulative_benefits)
                if opt_dim_export is not None:
                    export_tables["Dimension mix - optimised"] = opt_dim_export

        with dim_col2:
            cmp_dim_fig = benefit_dimension_chart(
                data,
                comp_selection,
                title="Comparison benefit mix by dimension",
                cumulative=show_cumulative_benefits,
                pivot=cmp_dim_pivot,
            )

            if cmp_dim_fig is not None:
                st.plotly_chart(cmp_dim_fig, use_container_width=True)

                cmp_dim_export = prepare_dimension_chart_export(cmp_dim_pivot, show_cumulative_benefits)
                if cmp_dim_export is not None:
                    export_tables["Dimension mix - comparison"] = cmp_dim_export

    st.markdown("### Project delivery schedule")

    controls_col1, controls_col2 = st.columns([3, 1])

    with controls_col1:

        gantt_option = st.radio(

            "Gantt display",

            ["Optimised", "Comparison"],

            horizontal=True,

            key="gantt_choice",

        )

    with controls_col2:

        show_outline = st.checkbox(

            "Show comparison schedule outline",

            value=True,

            key="gantt_outline",

        )

    gantt_selection = opt_selection if gantt_option == "Optimised" else comp_selection

    comparison_selection = comp_selection if gantt_option == "Optimised" else opt_selection

    _inject_gantt_hotkey_listener()

    gantt_fig = spend_gantt_chart(

        data,

        gantt_selection,

        comparison_selection=comparison_selection,

        show_outline=show_outline,

        title=f"Project delivery schedule - {gantt_option.lower()}",

    )

    if gantt_fig is not None:

        st.plotly_chart(gantt_fig, use_container_width=True)

        gantt_export = prepare_gantt_export(
            data,
            gantt_selection,
            comparison_selection,
        )
        if gantt_export is not None:
            export_tables[f"Gantt - {gantt_option}"] = gantt_export

    else:

        st.info("No spend matrix found for the selected scenario.")

    capacity_fig = market_capacity_indicator(data, gantt_selection)

    if capacity_fig is not None:

        st.plotly_chart(capacity_fig, use_container_width=True)

        capacity_export = prepare_capacity_export(
            opt_series if gantt_option == "Optimised" else cmp_series
        )
        if capacity_export is not None:
            export_tables[f"Market capacity - {gantt_option}"] = capacity_export

    schedule_col1, schedule_col2 = st.columns(2)

    with schedule_col1:

        if schedule_opt_fig is not None:

            st.plotly_chart(schedule_opt_fig, use_container_width=True)

            schedule_export = prepare_schedule_export(data, opt_selection)
            if schedule_export is not None:
                export_tables["Project schedule - optimised"] = schedule_export

        elif opt_selection.code:

            st.warning("No project schedule data found for the optimised selection.")

        else:

            st.info("Select an optimised scenario to view the project schedule.")

    with schedule_col2:

        if schedule_cmp_fig is not None:

            st.plotly_chart(schedule_cmp_fig, use_container_width=True)

            schedule_export = prepare_schedule_export(data, comp_selection)
            if schedule_export is not None:
                export_tables["Project schedule - comparison"] = schedule_export

        elif comp_selection.code:

            st.warning("No project schedule data found for the comparison selection.")

        else:

            st.info("Select a comparison scenario to view the project schedule.")

    chart_row1_col1, chart_row1_col2 = st.columns(2)

    with chart_row1_col1:

        if opt_series is not None:

            st.plotly_chart(

                cash_chart(
                    opt_series,
                    "Cash flow - optimised",
                    color=PRIMARY_COLOR,
                    data=data,
                    selection=opt_selection,
                    comparison_selection=comp_selection,
                ),

                use_container_width=True,

            )

            cash_export = prepare_cash_export(
                opt_series,
                label_prefix=opt_label,
            )
            if cash_export is not None:
                export_tables[f"Cash flow - {opt_label}"] = cash_export

    with chart_row1_col2:

        if cmp_series is not None:

            st.plotly_chart(

                cash_chart(
                    cmp_series,
                    "Cash flow - comparison",
                    color=COMPARISON_COLOR,
                    data=data,
                    selection=comp_selection,
                    comparison_selection=opt_selection,
                ),

                use_container_width=True,

            )

            cash_export = prepare_cash_export(
                cmp_series,
                label_prefix=cmp_label,
            )
            if cash_export is not None:
                export_tables[f"Cash flow - {cmp_label}"] = cash_export

    benefit_col1, benefit_col2 = st.columns(2)

    with benefit_col1:

        st.plotly_chart(

            benefit_chart(opt_series, cmp_series, dimension=opt_selection.dimension),

            use_container_width=True,

        )

    with benefit_col2:

        st.plotly_chart(

            benefit_delta_chart(

                data,

                opt_series,

                cmp_series,

                opt_selection,

                comp_selection,

                horizon_years=selected_npv_horizon,

            ),

            use_container_width=True,

        )

    benefit_export = prepare_benefit_export(
        opt_series,
        cmp_series,
        opt_label=opt_label,
        cmp_label=cmp_label,
    )
    if benefit_export is not None:
        export_tables["Benefit trend (real)"] = benefit_export

    benefit_delta_export = prepare_benefit_delta_export(opt_series, cmp_series)
    if benefit_delta_export is not None:
        export_tables["Benefit delta (real)"] = benefit_delta_export

    radar_fig = benefit_radar_chart(data, opt_selection, comp_selection)

    if radar_fig is not None:

        st.plotly_chart(radar_fig, use_container_width=True, theme=None)

        radar_export = prepare_radar_export(
            data,
            opt_selection,
            comp_selection,
        )
        if radar_export is not None:
            export_tables["Benefit mix radar"] = radar_export

    if export_tables:
        export_filename = f"capital_programme_charts_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        export_bytes = build_export_workbook(export_tables)
        st.download_button(
            "Export charts to XLSX",
            data=export_bytes,
            file_name=export_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="charts_export_download",
        )

    st.markdown("---")

    with st.expander("Scenario manager", expanded=False):
    
        st.header("Scenario manager")
    
        st.write("Use this workspace to create scenario folders and run new optimisation batches.")
        st.write(f"Preset scenarios live in {preset_root} | Saved scenarios live in {saved_root}")
    
        table_placeholder = st.empty()
    
        def render_folder_table(folders: List[scenario_utils.ScenarioFolder]) -> None:
            if not folders:
                table_placeholder.info("No scenario folders available yet.")
                return
            rows = []
            for folder in folders:
                meta = folder.metadata or scenario_utils.load_manifest(folder.path)
                last_run = (meta or {}).get("last_run") if meta else None
                rows.append(
                    {
                        "Folder": f"{folder.kind.title()} / {folder.name}",
                        "Default": "Yes" if folder.is_default else "",
                        "Path": str(folder.path),
                        "Last run": last_run.get("finished_at", "") if last_run else "",
                        "Files": len(last_run.get("output_files", [])) if last_run else 0,
                    }
                )
            table_placeholder.dataframe(pd.DataFrame(rows), use_container_width=True)
    
        render_folder_table(scenario_folders)
    
        with st.expander("Create new scenario folder", expanded=False):
            new_name = st.text_input("Scenario name", key="scenario_new_name")
            new_kind = st.radio("Folder type", ("saved", "preset"), index=0, format_func=str.title, horizontal=True, key="scenario_new_kind")
            if st.button("Create folder", key="scenario_create_folder"):
                if new_name.strip():
                    new_folder = scenario_utils.create_scenario_folder(settings, new_name.strip(), new_kind)
                    st.session_state["last_created_folder_name"] = new_folder.name
                    st.session_state["selected_cache_label"] = f"{new_folder.kind.title()} / {new_folder.name}"
                    load_dashboard_data.clear()
                    st.experimental_rerun()
                else:
                    st.warning("Please provide a scenario name before creating a folder.")
    
        st.subheader("Run optimiser")
    
        if not scenario_folders:
            st.info("Create a scenario folder before running the optimiser.")
        else:
            folder_labels = [f"{folder.kind.title()} / {folder.name}" for folder in scenario_folders]
            folder_lookup = {label: folder for label, folder in zip(folder_labels, scenario_folders)}
            default_folder_label = st.session_state.get("selected_cache_label")
            if default_folder_label not in folder_lookup:
                default_folder_label = folder_labels[0]
            with st.form("scenario_run_form"):
                target_label = st.selectbox(
                    "Target scenario folder",
                    folder_labels,
                    index=folder_labels.index(default_folder_label),
                    key="run_target_folder",
                )
                cost_type_options = list(settings.optimisation.cost_types)
                cost_types = st.multiselect("Cost types", cost_type_options, default=cost_type_options)
                scenario_key_options = list(solver_core.BENEFIT_SCENARIOS.keys())
                scenario_keys = st.multiselect("Benefit scenarios", scenario_key_options, default=scenario_key_options)
                dims_options = [str(dim) for dim in getattr(data, "dims", [])]
                if not dims_options:
                    dims_options = ["Total"]
                objective_dims = st.multiselect("Objective dimensions", dims_options, default=dims_options)
                col_cfg1, col_cfg2 = st.columns(2)
                start_fy = col_cfg1.number_input("Start financial year", value=int(settings.optimisation.start_fy), step=1)
                years = col_cfg2.number_input("Planning horizon (years)", value=int(settings.optimisation.years), min_value=1, step=1)
                run_plusminus = st.checkbox("Include buffer +/- levels", value=True)
                st.markdown("Baseline annual envelopes ($m p.a.)")
                envelope_defaults = pd.DataFrame([
                    {"Code": code, "AnnualMillions": value} for code, value in settings.optimisation.surplus_options_m.items()
                ])
                envelopes_editor = st.data_editor(
                    envelope_defaults,
                    num_rows="dynamic",
                    hide_index=True,
                    key="envelope_editor",
                    column_config={
                        "Code": st.column_config.TextColumn("Code", required=True),
                        "AnnualMillions": st.column_config.NumberColumn("Annual $ (millions)", min_value=0.0),
                    },
                )
                st.markdown("+/- levels ($m)")
                plus_defaults = pd.DataFrame({"LevelMillions": settings.optimisation.plusminus_levels_m})
                plus_editor = st.data_editor(
                    plus_defaults,
                    num_rows="dynamic",
                    hide_index=True,
                    key="plus_editor",
                    column_config={
                        "LevelMillions": st.column_config.NumberColumn("Level (+/- $m)", min_value=0.0),
                    },
                )
                st.markdown("Forced start rules")
                forced_rows = []
                for name, rule in settings.forced_start.items():
                    if rule.include is True:
                        include_state = "Include"
                    elif rule.include is False:
                        include_state = "Exclude"
                    else:
                        include_state = "Default"
                    forced_rows.append({"Project": name, "Include": include_state, "StartFY": rule.start})
                forced_defaults = pd.DataFrame(forced_rows)
                forced_editor = st.data_editor(
                    forced_defaults,
                    num_rows="dynamic",
                    hide_index=True,
                    disabled=["Project"],
                    key="forced_editor",
                    column_config={
                        "Include": st.column_config.SelectboxColumn("Include", options=["Default", "Include", "Exclude"]),
                        "StartFY": st.column_config.NumberColumn("Forced start (FY)", help="Leave blank to let the optimiser decide."),
                    },
                )
                run_button = st.form_submit_button("Run optimiser", type="primary")
    
            if run_button:
                proceed = True
                if not cost_types:
                    st.warning("Select at least one cost type.")
                    proceed = False
                if not scenario_keys:
                    st.warning("Select at least one benefit scenario.")
                    proceed = False
                envelopes = {}
                for row in envelopes_editor.to_dict("records"):
                    code = str(row.get("Code", "")).strip()
                    value = row.get("AnnualMillions")
                    if not code or pd.isna(value):
                        continue
                    envelopes[code] = float(value)
                if not envelopes:
                    st.warning("Provide at least one envelope value.")
                    proceed = False
                plus_levels = []
                if isinstance(plus_editor, pd.DataFrame) and "LevelMillions" in plus_editor.columns:
                    for val in plus_editor["LevelMillions"].tolist():
                        if pd.isna(val):
                            continue
                        plus_levels.append(float(val))
                plus_levels = sorted({round(val, 6) for val in plus_levels})
                if run_plusminus and not plus_levels:
                    st.info("No +/- levels supplied; defaulting to [0.0].")
                    plus_levels = [0.0]
                if not objective_dims:
                    objective_dims = dims_options
                forced_inputs = {}
                for row in forced_editor.to_dict("records"):
                    include_state = row.get("Include", "Default")
                    include_val = None
                    if include_state == "Include":
                        include_val = True
                    elif include_state == "Exclude":
                        include_val = False
                    start_val = row.get("StartFY")
                    if pd.isna(start_val):
                        start_val = None
                    else:
                        start_val = int(start_val)
                    forced_inputs[row["Project"]] = scenario_utils.ForcedStartInput(include=include_val, start=start_val)
                if proceed:
                    progress_bar = st.progress(0)
                    status_placeholder = st.empty()
                    eta_placeholder = st.empty()
                    log_placeholder = st.empty()
                    progress_messages: List[str] = []
    
                    def handle_progress(snapshot: scenario_utils.ProgressSnapshot) -> None:
                        total = max(snapshot.total, 1)
                        percent = int(min(snapshot.percent, 1.0) * 100)
                        progress_bar.progress(percent)
                        payload = snapshot.payload
                        if snapshot.stage == "solve_start":
                            message = (
                                f"Solving {payload.get('cost_type')} / {payload.get('scenario_key')} "
                                f"| {payload.get('primary_dim')} | {payload.get('surplus_key')} +/-{payload.get('plus_level')}"
                            )
                        elif snapshot.stage == "solve_complete":
                            message = f"{payload.get('status', 'ok').upper()} - {payload.get('cache_file', '')}"
                        elif snapshot.stage == "run_complete":
                            message = "Run complete."
                        elif snapshot.stage == "run_error":
                            message = "Run aborted."
                        else:
                            message = snapshot.stage.replace('_', ' ').title()
                        status_placeholder.write(f"{snapshot.completed}/{total} - {message}")
                        eta_placeholder.write(format_eta(snapshot.eta_seconds))
                        if snapshot.stage in {"solve_start", "solve_complete"}:
                            progress_messages.append(message)
                            log_placeholder.write("\n".join(progress_messages[-6:]))
    
                    try:
                        run_config = scenario_utils.OptimiserRunConfig(
                            cost_types=cost_types,
                            scenario_keys=scenario_keys,
                            objective_dims=objective_dims,
                            surplus_options_m=envelopes,
                            plusminus_levels_m=plus_levels or [0.0],
                            start_fy=int(start_fy),
                            years=int(years),
                            run_plusminus=run_plusminus,
                            forced_start=forced_inputs,
                            time_limit=int(settings.optimisation.solve_seconds),
                        )
                        summary = scenario_utils.run_optimiser_for_scenario(
                            settings,
                            folder_lookup[target_label],
                            run_config,
                            progress_callback=handle_progress,
                            clean=True,
                        )
                    except Exception as exc:
                        progress_bar.empty()
                        status_placeholder.error(f"Optimiser failed: {exc}")
                    else:
                        st.session_state["last_run_summary"] = summary.serializable()
                        st.session_state["selected_cache_label"] = target_label
                        load_dashboard_data.clear()
                        st.experimental_rerun()
    
if __name__ == "__main__":

    main()







