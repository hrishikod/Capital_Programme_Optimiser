"""Streamlit front end for exploring optimiser scenarios without Excel."""

from __future__ import annotations

import math
import sys

from dataclasses import dataclass, replace

from pathlib import Path

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import pandas as pd

import plotly.graph_objects as go

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

ROOT_CWD = Path.cwd()

ROOT_FILE = Path(__file__).resolve().parents[2]

ROOT_PARENT = ROOT_FILE.parent

for root in {ROOT_CWD, ROOT_FILE, ROOT_PARENT}:

    s = str(root)

    if s not in sys.path:

        sys.path.insert(0, s)

from capital_programme_optimiser.config import Settings, load_settings

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

@st.cache_resource(show_spinner=False)

def load_dashboard_data(cache_dir: str) -> DashboardData:

    """Load dashboard-ready data from the supplied cache directory."""

    cache_path = Path(cache_dir)

    results = load_results(cache_path)

    data = prepare_dashboard_data(results)

    return data

def format_currency(value: float) -> str:

    return f"{value:,.0f} m" if np.isfinite(value) else "-"

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

                f"{name} envelope (NZDm p.a.)",

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

                    f"{name} buffer (+/- NZDm)",

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

                    f"{name} cash uplift (+ NZDm)",

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

            y=df["Spend"],

            name="Annual spend",

            marker_color=color,

            opacity=BAR_OPACITY,

        )

    )

    fig.add_trace(

        go.Scatter(

            x=df["Year"],

            y=df["ClosingNet"],

            name="Closing net",

            mode="lines",

            line=dict(color=CLOSING_NET_COLOR, width=3),

        )

    )

    fig.add_trace(

        go.Scatter(

            x=df["Year"],

            y=df["Envelope"],

            name="Envelope",

            mode="lines+markers",

            line=dict(color=ENVELOPE_COLOR, width=3),

            marker=dict(color=ENVELOPE_COLOR, size=6),

        )

    )

    fig.add_hline(y=0, line=dict(color="#888888", dash="dot", width=1))

    yaxis_title_default = "NZD millions (values already in M)"

    if context_label:

        yaxis_title_default = f"NZD millions ({context_label})"

    fig.update_layout(

        title=title,

        barmode="overlay",

        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),

        xaxis_title="Financial year",

        yaxis_title=yaxis_title_default,

        template="plotly_white",

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

        xaxis_title="Financial year",

        yaxis_title="Annual benefit (NZDm)",

        yaxis2=dict(

            title="Benefit to date (real NZDm)",

            overlaying="y",

            side="right",

        ),

        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),

        template="plotly_white",

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

        fig.update_layout(template="plotly_white")

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

    yaxis_text = f"NZD millions ({context_label})" if context_label else "NZD millions (real)"

    fig.update_layout(

        title=title_text,

        xaxis_title="Financial year",

        yaxis_title=yaxis_text,

        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),

        template="plotly_white",

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
    theta_closed = theta_labels + theta_labels[:1]

    opt_values = value_list(opt_pv)
    cmp_values = value_list(cmp_pv)

    opt_closed = opt_values + opt_values[:1]
    cmp_closed = cmp_values + cmp_values[:1]

    max_val = max(opt_values + cmp_values + [1.0])

    theme_base = (st.get_option("theme.base") or "").lower()
    background_setting = (st.get_option("theme.backgroundColor") or "").strip()
    text_setting = (st.get_option("theme.textColor") or "").strip()

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
        fig.add_trace(
            go.Scatterpolar(
                r=opt_closed,
                theta=theta_closed,
                name=opt_name,
                line=dict(color=CUMULATIVE_OPT_LINE_COLOR, width=3),
                fill="toself",
                fillcolor=rgba_from_hex(CUMULATIVE_OPT_LINE_COLOR, 0.26 if is_dark_theme else 0.18),
                opacity=0.9,
            )
        )

    if cmp_pv:
        cmp_name = cmp_selection.name if cmp_selection and cmp_selection.name else "Comparison"
        fig.add_trace(
            go.Scatterpolar(
                r=cmp_closed,
                theta=theta_closed,
                name=cmp_name,
                line=dict(color=CUMULATIVE_CMP_LINE_COLOR, width=2, dash="dash"),
                fill="toself",
                fillcolor=rgba_from_hex(CUMULATIVE_CMP_LINE_COLOR, 0.22 if is_dark_theme else 0.16),
                opacity=0.78,
            )
        )

    selections_for_label = [sel for sel in (opt_selection, cmp_selection) if sel and sel.code]
    context_label = npv_context_label(data, *selections_for_label) if selections_for_label else None
    radial_title = f"NZD millions ({context_label})" if context_label else "NZD millions"

    fig.update_traces(hovertemplate="<b>%{theta}</b><br>%{r:,.0f} m<extra>%{name}</extra>")

    fig.update_layout(
        title="Benefit mix radar",
        polar=dict(
            bgcolor="rgba(0, 0, 0, 0)",
            radialaxis=dict(
                visible=True,
                range=[0, max_val * 1.05],
                title=radial_title,
                title_font=dict(color=text_color, size=12),
                tickfont=dict(color=text_color, size=11),
                gridcolor=grid_color,
                linecolor=outline_color,
                gridwidth=1,
                ticks="",
                tickformat=",.0f",
                nticks=5,
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
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

        fig.update_layout(template="plotly_white")

        return fig

    context_label = npv_context_label(data, opt_selection, cmp_selection, horizon_override=horizon_years)

    pv_opt = pv_by_dimension(data, opt_selection, horizon_years=horizon_years)

    pv_cmp = pv_by_dimension(data, cmp_selection, horizon_years=horizon_years)

    if not pv_opt or not pv_cmp:

        fig.add_annotation(text="Dimension-level NPV data unavailable for the selected scenarios", showarrow=False)

        fig.update_layout(template="plotly_white")

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

        yaxis_title=f"NZD millions ({context_label})",

        template="plotly_white",
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

        fig.update_layout(template="plotly_white")

        return fig

    pv_opt = pv_by_dimension(data, opt_selection, horizon_years=horizon_years)

    pv_cmp = pv_by_dimension(data, cmp_selection, horizon_years=horizon_years)

    if not pv_opt or not pv_cmp:

        fig.add_annotation(text="Missing dimension NPV data for bridge", showarrow=False)

        fig.update_layout(template="plotly_white")

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

        yaxis_title=f"NZD millions ({context_label})",

        template="plotly_white",

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

        xaxis_title="Financial year",

        yaxis_title="NZD millions",

        template="plotly_white",

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

) -> Optional[go.Figure]:

    pivot = dimension_timeseries(data, selection)

    if pivot is None or pivot.empty:

        return None

    if cumulative:

        pivot = pivot.cumsum()

    dims = [dim for dim in pivot.columns if str(dim).strip().lower() != 'total']

    if not dims:

        return None

    value_format = ',.1f' if cumulative else ',.0f'

    yaxis_label = "Cumulative benefit (NZDm)" if cumulative else "Annual benefit (NZDm)"

    title_text = title

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

        title=title_text,

        xaxis_title='Financial year',

        yaxis_title=yaxis_label,

        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0),

        template='plotly_white',

    )

    return fig

def spend_gantt_chart(

    data: DashboardData,

    selection: ScenarioSelection,

    *,

    comparison_selection: ScenarioSelection,

    show_arrows: bool,

    title: str,

) -> Optional[go.Figure]:

    if not selection.code:

        return None

    runs = extract_project_runs(data, selection.code)

    if not runs:

        return None

    runs.sort(key=lambda r: (r.start_year, r.project))

    y_labels = [run.project for run in runs]

    base_years = [run.start_year for run in runs]

    durations = [max(1, run.end_year - run.start_year + 1) for run in runs]

    custom = [[run.end_year, run.total_spend] for run in runs]

    fig = go.Figure(

        data=go.Bar(

            y=y_labels,

            x=durations,

            base=base_years,

            orientation="h",

            marker=dict(color=GANTT_COLOR),

            hovertemplate="project: %{y}<br>start FY: %{base}<br>end FY: %{customdata[0]}<br>total spend: %{customdata[1]:,.0f} m<extra></extra>",

            customdata=custom,

        )

    )

    if show_arrows and comparison_selection and comparison_selection.code:

        other_runs = {run.project: run for run in extract_project_runs(data, comparison_selection.code)}

        for run in runs:

            other = other_runs.get(run.project)

            if not other:

                continue

            delta = other.start_year - run.start_year

            if delta == 0:

                continue

            arrow_color = CLOSING_NET_COLOR if delta > 0 else COMPARISON_COLOR

            line_x = [other.start_year, run.start_year]

            line_y = [run.project, run.project]

            years = abs(delta)

            direction_label = "T-" if delta > 0 else "T+"

            label = f"{direction_label}{years}"

            fig.add_trace(

                go.Scatter(

                    x=line_x,

                    y=line_y,

                    mode="lines",

                    line=dict(color=arrow_color, width=4),

                    opacity=1.0,

                    hoverinfo="text",

                    text=[f"Schedule shift: {label}" for _ in line_x],

                    showlegend=False,

                )

            )

            head_symbol = "triangle-left" if delta > 0 else "triangle-right"

            fig.add_trace(

                go.Scatter(

                    x=[run.start_year],

                    y=[run.project],

                    mode="markers",

                    marker=dict(symbol=head_symbol, size=12, color=arrow_color),

                    hoverinfo="text",

                    text=[label],

                    showlegend=False,

                )

            )

    fig.update_layout(

        title=title,

        xaxis=dict(title="Financial year", tickmode="linear", dtick=1, tickangle=-45),

        yaxis=dict(title="Project", autorange="reversed"),

        template="plotly_white",

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
                    "Total spend: %{customdata[2]:,.0f} NZDm<br>"
                    "Financial year: %{customdata[0]}<br>"
                    "Annual spend: %{customdata[1]:,.0f} NZDm<extra></extra>"
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

            title="Financial year",

            tickmode="array",

            tickvals=tick_vals,

            ticktext=tick_text,

            tickangle=-35,

            range=[min_year, max_year],

            showgrid=False,

            zeroline=False,

        ),

        yaxis=dict(title="Annual spend (NZDm)", rangemode="tozero", showgrid=True),

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

        template="plotly_white",

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

        hover_labels.append(f"Financial year: {year}<br>Total spend: {value:.1f} B NZD<br>Status: {descriptor}")

        text_values.append(display_value)

        text_colors.append(text_color)

    gap_ratio = 0.05

    shapes = []

    for year, color in zip(years, colors):

        half_gap = gap_ratio / 2.0

        shapes.append(

            dict(

                type="rect",

                xref="x",

                yref="y",

                x0=year - 0.5 + half_gap,

                x1=year + 0.5 - half_gap,

                y0=0.0,

                y1=1.0,

                fillcolor=color,

                line=dict(color="#0F172A", width=1.0),

                layer="below",

            )

        )

    fig = go.Figure(

        data=go.Scatter(

            x=years,

            y=[0.5] * len(years),

            mode="markers+text",

            marker=dict(color="rgba(0,0,0,0)", size=14),

            hovertext=hover_labels,

            hovertemplate="%{hovertext}<extra></extra>",

            text=text_values,

            textposition="middle center",

            texttemplate="%{text}",

            cliponaxis=False,

        )

    )

    fig.update_traces(textfont=dict(size=14, color=text_colors))

    fig.update_layout(shapes=shapes)

    tick0 = float(totals["Year"].min()) if not totals.empty else 0.0

    last_year = float(totals["Year"].max()) if not totals.empty else tick0

    fig.update_layout(

        title="Market capacity indicator - total spend by financial year",

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

        template="plotly_white",

        showlegend=False,

    )

    return fig



def summarize_selection(selection: ScenarioSelection) -> pd.DataFrame:

    data = {

        "Metric": [

            "Scenario code",

            "Profile",

            "Mode",

            "Envelope (NZDm)",

            "Buffer / Cash uplift (NZDm)",

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

def main() -> None:

    st.set_page_config(page_title="Capital Programme Optimiser", layout="wide")
    st.markdown(PLOTLY_TRANSPARENT_STYLE, unsafe_allow_html=True)

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

    with st.spinner(f"Loading dashboard cache from {selected_path}..."):

        try:

            data = load_dashboard_data(str(selected_path))

        except Exception as exc:

            st.error(f"Unable to load scenario cache from {selected_path}: {exc}")

            st.stop()

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

    pv_col1, pv_col2 = st.columns(2)

    with pv_col1:
        st.plotly_chart(
            benefit_waterfall_chart(
                data, opt_selection, comp_selection, horizon_years=selected_npv_horizon
            ),
            use_container_width=True,
        )

    with pv_col2:
        st.plotly_chart(
            benefit_bridge_chart(
                data, opt_selection, comp_selection, horizon_years=selected_npv_horizon
            ),
            use_container_width=True,
        )

    show_cumulative_benefits = st.checkbox(
        "Show cumulative dimension benefits",
        value=st.session_state.get("show_cumulative_dimension_benefits", False),
        key="show_cumulative_dimension_benefits",
    )

    dim_col1, dim_col2 = st.columns(2)

    with dim_col1:
        opt_dim_fig = benefit_dimension_chart(
            data,
            opt_selection,
            title="Optimised benefit mix by dimension",
            cumulative=show_cumulative_benefits,
        )

        if opt_dim_fig is not None:
            st.plotly_chart(opt_dim_fig, use_container_width=True)

    with dim_col2:
        cmp_dim_fig = benefit_dimension_chart(
            data,
            comp_selection,
            title="Comparison benefit mix by dimension",
            cumulative=show_cumulative_benefits,
        )

        if cmp_dim_fig is not None:
            st.plotly_chart(cmp_dim_fig, use_container_width=True)

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

        show_arrows = st.checkbox(

            "Show schedule shift arrows",

            value=True,

            key="gantt_arrows",

        )

    gantt_selection = opt_selection if gantt_option == "Optimised" else comp_selection

    comparison_selection = comp_selection if gantt_option == "Optimised" else opt_selection

    gantt_fig = spend_gantt_chart(

        data,

        gantt_selection,

        comparison_selection=comparison_selection,

        show_arrows=show_arrows,

        title=f"Project delivery schedule - {gantt_option.lower()}",

    )

    if gantt_fig is not None:

        st.plotly_chart(gantt_fig, use_container_width=True)

    else:

        st.info("No spend matrix found for the selected scenario.")

    capacity_fig = market_capacity_indicator(data, gantt_selection)

    if capacity_fig is not None:

        st.plotly_chart(capacity_fig, use_container_width=True)

    schedule_col1, schedule_col2 = st.columns(2)

    with schedule_col1:

        if schedule_opt_fig is not None:

            st.plotly_chart(schedule_opt_fig, use_container_width=True)

        elif opt_selection.code:

            st.warning("No project schedule data found for the optimised selection.")

        else:

            st.info("Select an optimised scenario to view the project schedule.")

    with schedule_col2:

        if schedule_cmp_fig is not None:

            st.plotly_chart(schedule_cmp_fig, use_container_width=True)

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

    radar_fig = benefit_radar_chart(data, opt_selection, comp_selection)

    if radar_fig is not None:

        st.plotly_chart(radar_fig, use_container_width=True, theme=None)


    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:

        st.subheader("Optimised scenario details")

        st.dataframe(summarize_selection(opt_selection), use_container_width=True)

    with detail_col2:

        st.subheader("Comparison scenario details")

        st.dataframe(summarize_selection(comp_selection), use_container_width=True)


    st.markdown("---")

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
            col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
            start_fy = col_cfg1.number_input("Start financial year", value=int(settings.optimisation.start_fy), step=1)
            years = col_cfg2.number_input("Planning horizon (years)", value=int(settings.optimisation.years), min_value=1, step=1)
            time_limit = col_cfg3.number_input("Solver time limit (seconds)", value=int(settings.optimisation.solve_seconds), min_value=60, step=60)
            run_plusminus = st.checkbox("Include buffer +/- levels", value=True)
            st.markdown("Baseline annual envelopes (NZDm p.a.)")
            envelope_defaults = pd.DataFrame([
                {"Code": code, "AnnualNZDm": value} for code, value in settings.optimisation.surplus_options_m.items()
            ])
            envelopes_editor = st.data_editor(
                envelope_defaults,
                num_rows="dynamic",
                hide_index=True,
                key="envelope_editor",
                column_config={
                    "Code": st.column_config.TextColumn("Code", required=True),
                    "AnnualNZDm": st.column_config.NumberColumn("Annual NZD (millions)", min_value=0.0),
                },
            )
            st.markdown("+/- levels (NZDm)")
            plus_defaults = pd.DataFrame({"LevelNZDm": settings.optimisation.plusminus_levels_m})
            plus_editor = st.data_editor(
                plus_defaults,
                num_rows="dynamic",
                hide_index=True,
                key="plus_editor",
                column_config={
                    "LevelNZDm": st.column_config.NumberColumn("Level (+/- NZDm)", min_value=0.0),
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
                value = row.get("AnnualNZDm")
                if not code or pd.isna(value):
                    continue
                envelopes[code] = float(value)
            if not envelopes:
                st.warning("Provide at least one envelope value.")
                proceed = False
            plus_levels = []
            if isinstance(plus_editor, pd.DataFrame) and "LevelNZDm" in plus_editor.columns:
                for val in plus_editor["LevelNZDm"].tolist():
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
                        time_limit=int(time_limit) if time_limit else None,
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

