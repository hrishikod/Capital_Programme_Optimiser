"""Streamlit front end for exploring optimiser scenarios without Excel."""

from __future__ import annotations

import copy
import html
import inspect
import io
import json
import math
import pickle
import re
import shutil
import sys
import tempfile
import textwrap
from dataclasses import dataclass, replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from uuid import uuid4

ROOT_CWD = Path.cwd()
ROOT_FILE = Path(__file__).resolve().parents[2]
ROOT_PARENT = ROOT_FILE.parent

for root in {ROOT_CWD, ROOT_FILE, ROOT_PARENT}:
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)

import numpy as np
import pandas as pd
import plotly.colors as plc
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu

from capital_programme_optimiser.config import Settings, load_settings
from capital_programme_optimiser.dashboard.constants import (
    SCENARIO_COMPARISON_NAME,
    SCENARIO_PAIR_NAME,
    SCENARIO_PRIMARY_NAME,
)
from capital_programme_optimiser.dashboard.data import (
    DashboardData,
    _canonicalise_dimension_label,
    dim_short,
    extract_project_runs,
    find_scenario_code,
    load_results,
    prepare_dashboard_data,
    scenario_metadata,
)
from capital_programme_optimiser.dashboard.regions import (
    _canonical_join_key,
    compute_region_metrics,
    fetch_region_geojson,
    get_geojson_name_field,
    load_region_mapping,
    region_baselines,
)

if not hasattr(pd.Index, "clip"):

    def _index_clip(self, lower=None, upper=None):

        arr = self.to_numpy()

        if lower is not None:

            arr = np.maximum(arr, lower)

        if upper is not None:

            arr = np.minimum(arr, upper)

        return pd.Index(arr)

    pd.Index.clip = _index_clip


PREVIEW_COMPONENT_PATH = (Path(__file__).parent / "components" / "preview_nav").resolve()
_preview_navigation_component = components.declare_component(
    "preview_navigation",
    path=str(PREVIEW_COMPONENT_PATH),
)
ENABLE_PREVIEW_NAVIGATION = False  # Toggle preview pane without removing supporting code.
SHOW_REAL_BENEFIT_CHARTS = False  # Hide real benefit profile charts while keeping logic available.
SHOW_ANALYSIS_CORRELATION = False  # Hide the analysis correlation block (controls + chart).

INTERPOLATED_PROFILES_PATH = ROOT_FILE / "scenario_sets" / "interpolated_profiles_new.pkl"
PCT_BAR_PATH = ROOT_FILE / "scenario_sets" / "Pct_bar.pkl"
COST_BENEFIT_STREAMS_PATH = ROOT_FILE / "Cost_benefit_streams.xlsx"
BCR_MAPPING_PATH = ROOT_FILE / "BCR_mapping.xlsx"
INTERPOLATED_PROFILE_ANIMATION_SECONDS = 16.0

_FIND_SCENARIO_SUPPORTS_DIM = "objective_dim" in inspect.signature(find_scenario_code).parameters
_FIND_SCENARIO_SUPPORTS_BENEFIT_LEVEL = "benefit_level" in inspect.signature(find_scenario_code).parameters

CURRENT_SCENARIO_LABELS: Dict[str, str] = {
    "primary": SCENARIO_PRIMARY_NAME,
    "comparison": SCENARIO_COMPARISON_NAME,
    "pair": SCENARIO_PAIR_NAME,
}


def scenario_primary_label() -> str:
    """Return the display label for the primary scenario."""
    return CURRENT_SCENARIO_LABELS.get("primary", SCENARIO_PRIMARY_NAME)


def scenario_comparison_label() -> str:
    """Return the display label for the comparison scenario."""
    return CURRENT_SCENARIO_LABELS.get("comparison", SCENARIO_COMPARISON_NAME)


def scenario_pair_label() -> str:
    """Return the display label for the primary/comparison pair."""
    return CURRENT_SCENARIO_LABELS.get(
        "pair",
        f"{scenario_primary_label()} - {scenario_comparison_label()}",
    )


def set_scenario_display_labels(primary: str, comparison: str) -> None:
    """Update the global scenario labels used for display."""
    primary_clean = (primary or "").strip() or SCENARIO_PRIMARY_NAME
    comparison_clean = (comparison or "").strip() or SCENARIO_COMPARISON_NAME
    CURRENT_SCENARIO_LABELS["primary"] = primary_clean
    CURRENT_SCENARIO_LABELS["comparison"] = comparison_clean
    CURRENT_SCENARIO_LABELS["pair"] = f"{primary_clean} - {comparison_clean}"


def _slugify_label(label: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", str(label or ""))
    return "_".join(tokens)

from capital_programme_optimiser.frontend import scenarios as scenario_utils
from capital_programme_optimiser.optimisation import solver_core
from capital_programme_optimiser.tools import parquet_export

POWERBI_BLUE = "#19456B"
POWERBI_GREEN = "#AFBD22"
POWERBI_TERTIARY = "#908070"

PRIMARY_COLOR = POWERBI_BLUE

COMPARISON_COLOR = POWERBI_GREEN

WATERFALL_GAIN_COLOR = POWERBI_GREEN
WATERFALL_LOSS_COLOR = "#CA4142"
WATERFALL_TOTAL_COLOR = "#4C4C4C"

BAR_OPACITY = 0.75

GANTT_COLOR = "#225F92"

GANTT_OUTLINE_COLOR_BASE = "#DC2626"
GANTT_OUTLINE_COLOR_ALT = "#B91C1C"
GANTT_OUTLINE_VARIANT_KEY = "gantt_outline_variant"

CLOSING_NET_COLOR = POWERBI_GREEN

ENVELOPE_COLOR = POWERBI_TERTIARY

CAPACITY_GREEN = POWERBI_GREEN

CAPACITY_AMBER = POWERBI_TERTIARY

CAPACITY_RED = POWERBI_BLUE

CAPACITY_ZERO = POWERBI_TERTIARY

BENEFIT_DIMENSION_PALETTE = [
    "#19456B",
    "#008B97",
    "#908070",
    "#AFBD22",
    "#CA4142",
    "#197D5D",
]

# --- Market capacity helpers -------------------------------------------------

SHOW_CAPACITY_COLORBAR = False

def _nice_half_step(value: float, *, min_top: float = 3.25) -> float:
    """Round up to the nearest 0.5 above the provided value and minimum ceiling."""
    top = max(float(value), float(min_top))
    return 0.5 * math.ceil(top / 0.5)

def _capacity_gradient_colorscale(scale_top_b: float) -> list[list[float | str]]:
    """Smooth gradient anchored on key policy thresholds (0B, 2B, ~2.6B, 3B)."""
    def pos(billions: float) -> float:
        return float(np.clip(billions / max(scale_top_b, 1e-6), 0.0, 1.0))

    p0 = 0.0
    p2 = pos(2.0)
    p26 = pos(2.6)
    p3 = pos(3.0)
    p1 = 1.0

    eps = 1e-6
    p26 = max(p26, min(p2 + 0.02, p1 - eps))
    p3 = max(p3, min(p26 + 0.02, p1 - eps))

    return [
        [p0, CAPACITY_HEAT_ZERO],
        [p2, CAPACITY_HEAT_YELLOW],
        [p26, CAPACITY_HEAT_ORANGE],
        [p3, CAPACITY_HEAT_RED],
        [p1, CAPACITY_HEAT_RED],
    ]

def _capacity_step_colorscale() -> list[list[float | str]]:
    """Discrete banded palette (grey, yellow, orange, red)."""
    step_eps = 1e-6
    return [
        [0.00, CAPACITY_HEAT_ZERO],
        [1.0 / 3.0 - step_eps, CAPACITY_HEAT_ZERO],
        [1.0 / 3.0, CAPACITY_HEAT_YELLOW],
        [2.0 / 3.0 - step_eps, CAPACITY_HEAT_YELLOW],
        [2.0 / 3.0, CAPACITY_HEAT_ORANGE],
        [1.0 - step_eps, CAPACITY_HEAT_ORANGE],
        [1.0, CAPACITY_HEAT_RED],
    ]

def _format_hover_spend(billions: float) -> tuple[str, str]:
    """Return (billions text, millions text) for hover readouts."""
    if not np.isfinite(billions):
        return "-", "-"
    return f"{billions:,.1f} B", f"{billions * 1000.0:,.0f} m"

# ---- Market capacity heatmap palette (R-Y-O) ----
# Grey is used only for "no spend recorded".
CAPACITY_HEAT_ZERO   = "#D1D5DB"  # neutral grey
CAPACITY_HEAT_YELLOW = "#FACC15"  # Comfortable  (<= $2.0B)
CAPACITY_HEAT_ORANGE = "#FB923C"  # Watch zone   ($2.0B-$3.0B)
CAPACITY_HEAT_RED    = "#DC2626"  # High pressure (>= $3.0B)


REGION_MAP_ZERO_COLOR = "rgba(148, 163, 184, 0.28)"
MAPLIBRE_FALLBACK_LIGHT = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
MAPLIBRE_FALLBACK_DARK = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
MAPLIBRE_FALLBACK_ATTRIBUTION = "OpenStreetMap contributors | CARTO"
LINZ_TOPO_VECTOR_TILE = "https://basemaps.linz.govt.nz/v1/tiles/topographic/WebMercatorQuad/{z}/{x}/{y}.pbf"
NZ_BOUNDS = [[165.0, -47.5], [180.0, -33.0]]
NZ_CENTER = [173.0, -41.0]

PBI_SEQUENTIAL_SCALE = [
    [0.0, '#2E7D32'],
    [0.5, '#F59E0B'],
    [1.0, '#C62828'],
]

PBI_DIVERGING_SCALE = [
    [0.0, '#2E7D32'],
    [0.5, '#F59E0B'],
    [1.0, '#C62828'],
]


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


_EXPORT_POPOVER_FOCUS_HTML = """
<script>
(() => {
  const POPOVER_SELECTOR = ".pbi-export-popover,[data-testid='stPopover'],[data-testid='stPopoverContent'],[data-baseweb='popover']";
  const attach = (doc) => {
    if (!doc || doc.__pbiExportBlurInstalled) {
      return;
    }
    doc.__pbiExportBlurInstalled = true;
    let hasPopover = false;
    let armUntil = 0;
    const scrubButtons = () => {
      try {
        const roots = doc.querySelectorAll(POPOVER_SELECTOR);
        roots.forEach((root) => {
          root.querySelectorAll("div.stDownloadButton > button").forEach((btn) => {
            btn.tabIndex = -1;
            if (doc.activeElement === btn) {
              btn.blur();
            }
          });
        });
      } catch (err) {
        // ignore query failures
      }
    };
    const arm = () => {
      armUntil = Date.now() + 700;
      const active = doc.activeElement;
      if (active && active.closest && active.closest(POPOVER_SELECTOR)) {
        active.blur();
      }
      scrubButtons();
    };
    const handler = (event) => {
      if (Date.now() > armUntil) {
        return;
      }
      const target = event.target;
      if (target && target.closest && target.closest(POPOVER_SELECTOR) && target.tagName === 'BUTTON') {
        setTimeout(() => {
          try {
            target.blur();
          } catch (err) {
            // ignore focus teardown issues
          }
        }, 0);
      }
    };
    doc.addEventListener('focusin', handler, true);
    if (doc.body && window.MutationObserver) {
      const observer = new MutationObserver(() => {
        const popoverOpen = !!doc.querySelector(POPOVER_SELECTOR);
        if (popoverOpen && !hasPopover) {
          arm();
        }
        hasPopover = popoverOpen;
      });
      observer.observe(doc.body, { childList: true, subtree: true });
    }
  };
  attach(document);
  try {
    if (window.parent && window.parent.document) {
      attach(window.parent.document);
    }
  } catch (err) {
    // ignore cross-origin cases
  }
})();
</script>
"""


def _inject_gantt_hotkey_listener() -> None:
    """
    Mount a tiny HTML shim that listens for the 'Z' key and toggles the
    Programme Schedule outline colour. Works on modern Streamlit; degrades gracefully
    on older builds that don't support `key=` or returning a value.
    """
    # Session defaults
    st.session_state.setdefault(GANTT_OUTLINE_VARIANT_KEY, "base")
    st.session_state.setdefault("_gantt_last_toggle", 0)
    st.session_state.setdefault("_gantt_hotkey_supported", True)

    # Try to mount with `key=` first (newer Streamlit). Fall back if unsupported.
    try:
        response = components.html(
            _GANTT_HOTKEY_HTML,
            height=0,
            width=0,
            key="_gantt_hotkey",
        )
    except TypeError:
        # Older Streamlit: no `key` kwarg and no return value
        components.html(_GANTT_HOTKEY_HTML, height=0, width=0)
        st.session_state["_gantt_hotkey_supported"] = False
        response = None

    # If the HTML shim can post a value back, use it to flip the variant.
    if isinstance(response, dict):
        toggle_value = response.get("toggle")
        if toggle_value and toggle_value != st.session_state["_gantt_last_toggle"]:
            st.session_state["_gantt_last_toggle"] = toggle_value
            current = st.session_state.get(GANTT_OUTLINE_VARIANT_KEY, "base")
            st.session_state[GANTT_OUTLINE_VARIANT_KEY] = "alt" if current == "base" else "base"
            st.rerun()


def _inject_export_popover_focus_guard() -> None:
    """Prevent popover download buttons from auto-focusing on open."""
    try:
        components.html(
            _EXPORT_POPOVER_FOCUS_HTML,
            height=0,
            width=0,
            key="_export_popover_focus_guard",
        )
    except TypeError:
        components.html(_EXPORT_POPOVER_FOCUS_HTML, height=0, width=0)



def _current_gantt_outline_color() -> str:
    variant = st.session_state.get(GANTT_OUTLINE_VARIANT_KEY, "base")
    return GANTT_OUTLINE_COLOR_ALT if variant == "alt" else GANTT_OUTLINE_COLOR_BASE

NAV_TABS = [
    "Cash Flow",
    "Programme Schedule",
    "Analysis",
    "Benefits",
    "Regions",
    "Delivery",
    "Scenario Manager",
    "Demo",
]

NAV_ICON_MAP = {
    "Programme Schedule": "calendar-event",
    "Cash Flow": "cash-coin",
    "Analysis": "bar-chart-line",
    "Benefits": "graph-up-arrow",
    "Regions": "geo-alt",
    "Delivery": "box-seam",
    "Scenario Manager": "kanban",
    "Demo": "speedometer",
}





def inject_kpi_card_theme() -> None:
    """Styles for rounded KPI cards and optional metric facelift."""
    import streamlit as st

    st.markdown(f"""
    <style>
      :root {{
        --pbi-blue: {POWERBI_BLUE};
        --pbi-green: {POWERBI_GREEN};
        --kpi-border: rgba(25,69,107,.18);
        --kpi-shadow: 0 10px 24px rgba(25,69,107,.08);
        --kpi-text-1: #0F172A;
        --kpi-text-2: #475569;
        --kpi-top-grey: rgba(148,163,184,.45);
      }}

      div[data-testid="stMetric"] {{
        position: relative;
        border: 1px solid var(--kpi-border);
        border-radius: 16px;
        padding: 14px 16px;
        background: #fff;
        box-shadow: var(--kpi-shadow);
      }}
      div[data-testid="stMetric"]::before {{
        content: "";
        position: absolute; left: 12px; right: 12px; top: 0;
        height: 6px; border-radius: 999px;
        background: linear-gradient(90deg, var(--pbi-green), var(--kpi-top-grey));
        transform: translateY(-3px);
      }}
      div[data-testid="stMetric"] label {{
        color: var(--pbi-blue);
        font-weight: 600;
        margin-bottom: 2px;
      }}
      div[data-testid="stMetricValue"] {{
        color: var(--kpi-text-1);
        font-weight: 700;
        font-size: 1.9rem;
        line-height: 1.1;
      }}
      div[data-testid="stMetricDelta"] {{
        border-radius: 999px;
        padding: 2px 8px;
        border: 1px solid rgba(25,69,107,.25);
        background: rgba(25,69,107,.08);
        color: var(--pbi-blue);
        font-weight: 600;
      }}

      .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0,1fr));
        gap: 14px;
        padding-bottom: 18px;
      }}
      @media (max-width: 1200px) {{ .kpi-grid {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
      @media (max-width: 800px)  {{ .kpi-grid {{ grid-template-columns: 1fr; }} }}

      .kpi-card {{
        position: relative;
        background: #fff;
        border: 1px solid var(--kpi-border);
        border-radius: 18px;
        box-shadow: var(--kpi-shadow);
        padding: 16px 18px;
        overflow: hidden;
      }}
      .kpi-card::before {{
        content:"";
        position:absolute; left:14px; right:14px; top:0;
        height:6px; border-radius:999px;
        background: var(--kpi-top-grey);
        transform: translateY(-3px);
        z-index: 1;
      }}
      .kpi-card::after {{
        content:"";
        position:absolute; left:14px; right:14px; top:0;
        height:6px; border-radius:999px;
        background: linear-gradient(90deg, var(--pbi-green), var(--kpi-top-grey));
        transform: translateY(-3px) scaleX(0);
        transform-origin: right center;
        z-index: 2;
        animation: none;
      }}
      .kpi-card.kpi-sweep-on::after {{
        animation: kpi-sweep 0.55s ease-out forwards;
        animation-delay: var(--kpi-sweep-delay, 0s);
      }}
      .kpi-card.kpi-sweep-on::after {{
        animation: kpi-sweep 0.55s ease-out forwards;
        animation-delay: var(--kpi-sweep-delay, 0s);
      }}
      @keyframes kpi-sweep {{
        0% {{ transform: translateY(-3px) scaleX(0); }}
        100% {{ transform: translateY(-3px) scaleX(1); }}
      }}
      .kpi-title {{
        color: var(--pbi-blue);
        font-weight: 600;
        font-size: .95rem;
        margin: 0 0 .25rem 0;
      }}
      .kpi-value {{
        color: var(--kpi-text-1);
        font-weight: 700;
        font-size: 2rem;
        line-height: 1.1;
        letter-spacing: -0.01em;
      }}
      .kpi-text {{
        color: var(--kpi-text-1);
        font-weight: 500;
        font-size: 1.3rem;
        line-height: 1.2;
      }}
      .kpi-inline {{
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        align-items: baseline;
      }}
      .kpi-inline .kpi-text {{
        display: inline;
        font-size: 1.05rem;
      }}
      .kpi-inline .kpi-delta {{
        margin-top: 0;
      }}
      .kpi-sub {{
        color: var(--kpi-text-2);
        font-size: .85rem;
        margin-top: .35rem;
      }}
      .kpi-delta {{
        display:inline-flex; align-items:center; gap:6px;
        margin-top:.55rem;
        font-weight:600; font-size:.85rem;
        padding:3px 10px; border-radius:999px;
        border:1px solid transparent;
      }}
      .kpi-delta.lead {{
        font-size:1.5rem;
        padding:9px 18px;
        margin-top:.85rem;
      }}
      .kpi-delta.up {{
        color: var(--pbi-green);
        background: rgba(175,189,34,.12);
        border-color: rgba(175,189,34,.35);
      }}
      .kpi-delta.down {{
        color: #CA4142;
        background: rgba(202,65,66,.10);
        border-color: rgba(202,65,66,.30);
      }}
      .kpi-delta.muted {{
        color: #64748B;
        background: rgba(148,163,184,.18);
        border-color: rgba(148,163,184,.45);
      }}
      .kpi-delta.neutral {{
        color: var(--pbi-blue);
        background: rgba(25,69,107,.10);
        border-color: rgba(25,69,107,.28);
      }}
    </style>
    """, unsafe_allow_html=True)
def inject_powerbi_theme() -> None:
    css = f"""
        <style>
            :root {{
                --pbi-blue: {POWERBI_BLUE};
                --pbi-green: {POWERBI_GREEN};
                --pbi-tertiary: {POWERBI_TERTIARY};
            }}
            body {{
                background-color: #ffffff;
                color: var(--pbi-tertiary);
                font-family: 'Segoe UI', 'Inter', sans-serif;
            }}
            a, .stMarkdown a {{
                color: var(--pbi-green);
            }}
            .pbi-header {{
                color: var(--pbi-blue);
                font-size: 2.2rem;
                font-weight: 600;
                display: inline-block;
                position: relative;
                padding-bottom: 0.45rem;
                margin-bottom: 1.4rem;
            }}
            .pbi-header::after {{
                content: '';
                position: absolute;
                left: 0;
                bottom: 0;
                width: 100%;
                height: 6px;
                border-radius: 999px;
                background: var(--pbi-green);
            }}
            .pbi-header-underline {{
                display: none;
            }}
            .pbi-section-title {{
                color: var(--pbi-blue);
                font-size: 1.35rem;
                font-weight: 600;
                margin: 1.0rem 0 0.5rem;
            }}
            .pbi-help-wrap {{
                display: flex;
                justify-content: flex-end;
                align-items: center;
                margin: 0.1rem 0 0.25rem;
            }}
            .pbi-help-icon {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 22px;
                height: 22px;
                border-radius: 999px;
                border: 1px solid rgba(15, 23, 42, 0.2);
                color: var(--pbi-blue);
                background: #ffffff;
                font-size: 0.78rem;
                font-weight: 700;
                cursor: help;
                box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
            }}
            .pbi-card {{
                background: #ffffff;
                border-radius: 16px;
                padding: 1.1rem 1.3rem;
                box-shadow: 0 10px 24px rgba(25, 69, 107, 0.08);
                border: 1px solid var(--pbi-tertiary);
                margin-bottom: 1.1rem;
            }}
            .pbi-hidden-block {{
                display: none !important;
            }}
            .pbi-table .stDataFrame {{
                border-radius: 12px;
                border: 1px solid var(--pbi-tertiary);
                overflow: hidden;
            }}
            div[data-testid='stDataFrame'] table thead tr th {{
                background: rgba(25, 69, 107, 0.08) !important;
                color: var(--pbi-blue) !important;
                font-weight: 600;
            }}
            .region-summary-table div[data-testid='stDataFrame'] table {{
                font-family: 'Inter', 'Segoe UI', sans-serif !important;
                font-size: 0.95rem !important;
                color: #1f2937 !important;
            }}
            .region-summary-table div[data-testid='stDataFrame'] table thead tr th {{
                background: #ffffff !important;
                color: var(--pbi-blue) !important;
                font-weight: 600 !important;
                position: sticky !important;
                top: 0 !important;
                z-index: 2 !important;
                box-shadow: 0 2px 4px rgba(15, 23, 42, 0.08);
            }}
            .region-summary-table div[data-testid='stDataFrame'] table tbody tr td {{
                background: #ffffff !important;
                color: #475569 !important;
                font-weight: 500 !important;
            }}
            .region-summary-table div[data-testid='stDataFrame'] table tbody tr:nth-child(2n) td {{
                background: rgba(25, 69, 107, 0.04) !important;
            }}
            .region-summary-table div[data-testid='stDataFrame'] table tbody tr:hover td {{
                background: rgba(175, 189, 34, 0.16) !important;
            }}
            .region-summary-table div[data-testid='stDataFrame'] table tbody td:first-child {{
                font-weight: 600 !important;
                color: var(--pbi-blue) !important;
            }}
            div[data-testid='stDataFrame'] table tbody tr:hover td {{
                background-color: rgba(144, 128, 112, 0.15) !important;
            }}
            .region-summary-table div[data-testid='stDataFrame'] table tbody tr:hover td {{
                background: rgba(175, 189, 34, 0.16) !important;
            }}
            div.stButton > button,
            div.stDownloadButton > button,
            div.stFormSubmitButton > button,
            button[kind='primary'],
            button[data-testid='baseButton-primary'] {{
                background: var(--pbi-blue) !important;
                color: #ffffff !important;
                border: 1px solid var(--pbi-blue) !important;
                border-radius: 999px;
                font-weight: 600;
            }}
            div.stButton > button:hover,
            div.stDownloadButton > button:hover,
            div.stFormSubmitButton > button:hover,
            button[kind='primary']:hover,
            button[data-testid='baseButton-primary']:hover {{
                background: var(--pbi-blue) !important;
                border-color: var(--pbi-blue) !important;
                filter: brightness(0.92);
            }}
            div.stButton > button:focus,
            div.stButton > button:active,
            div.stFormSubmitButton > button:focus,
            div.stFormSubmitButton > button:active,
            button[kind='primary']:focus,
            button[kind='primary']:active,
            button[data-testid='baseButton-primary']:focus,
            button[data-testid='baseButton-primary']:active,
            button[aria-pressed='true'] {{
                color: #ffffff !important;
            }}
            div[data-baseweb='segmented-control'] button {{
                background: rgba(144, 128, 112, 0.12);
                border: 1px solid rgba(25, 69, 107, 0.2);
                color: var(--pbi-blue);
                border-radius: 999px;
                transition: all 0.18s ease-in-out;
            }}
            div[data-baseweb='segmented-control'] button:hover {{
                border-color: var(--pbi-blue);
            }}
            div[data-baseweb='segmented-control'] button[aria-pressed='true'],
            div[data-baseweb='segmented-control'] button[aria-pressed='true']:hover {{
                background: var(--pbi-blue) !important;
                color: #ffffff !important;
                border-color: var(--pbi-blue) !important;
                box-shadow: 0 8px 18px rgba(25, 69, 107, 0.22);
                filter: brightness(0.92);
            }}
            div[data-baseweb='segmented-control'] span[data-baseweb='radio-mark'] {{
                display: none !important;
            }}
            div[data-baseweb='segmented-control'] svg {{
                display: none !important;
            }}
            div[data-baseweb='segmented-control'] [data-baseweb='radio-mark'],
            div[data-baseweb='segmented-control'] [data-baseweb='radio-mark']::before,
            div[data-baseweb='segmented-control'] [data-baseweb='radio-mark']::after {{
                display: none !important;
                background: transparent !important;
                box-shadow: none !important;
            }}
            div.stButton > button *,
            div.stDownloadButton > button * {{
                color: inherit !important;
            }}
            div.stButton > button:focus *,
            div.stButton > button:active *,
            button[kind='primary']:focus *,
            button[kind='primary']:active *,
            button[data-testid='baseButton-primary']:focus *,
            button[data-testid='baseButton-primary']:active *,
            button[aria-pressed='true'] * {{
                color: #ffffff !important;
            }}
            .pbi-export-gap {{
                height: 32px;
            }}
            .pbi-export-popover-title {{
                color: var(--pbi-blue);
                font-size: 0.95rem;
                font-weight: 700;
                margin: 0 0 0.15rem;
            }}
            .pbi-export-popover-sub {{
                color: #475569;
                font-size: 0.82rem;
                margin: 0 0 0.6rem;
            }}
            .pbi-export-popover div[data-testid='stRadio'] > div {{
                flex-wrap: nowrap;
                gap: 0.4rem;
            }}
            .pbi-export-popover div[data-testid='stRadio'] label[data-baseweb='radio'] {{
                white-space: nowrap;
                padding: 0.28rem 0.7rem;
                font-size: 0.82rem;
            }}
            .pbi-export-popover div.stDownloadButton > button {{
                width: 100%;
                font-size: 0.82rem;
                padding: 0.45rem 0.7rem;
            }}
            .pbi-export-popover div.stDownloadButton > button:disabled {{
                background: rgba(148, 163, 184, 0.18) !important;
                border-color: rgba(148, 163, 184, 0.45) !important;
                color: #64748B !important;
                opacity: 1 !important;
            }}
            .pbi-export-popover div.stDownloadButton > button:disabled * {{
                color: #64748B !important;
            }}
            .pbi-export-popover div.stDownloadButton > button:not(:disabled) {{
                background: #ffffff !important;
                border: 1px solid rgba(15, 23, 42, 0.16) !important;
                color: #1f2937 !important;
                border-radius: 12px;
            }}
            .pbi-export-popover div.stDownloadButton > button:not(:disabled):hover {{
                border-color: rgba(25, 69, 107, 0.35) !important;
                background: rgba(25, 69, 107, 0.08) !important;
            }}
            .pbi-export-popover div.stDownloadButton > button:not(:disabled):focus,
            .pbi-export-popover div.stDownloadButton > button:not(:disabled):active,
            .pbi-export-popover div.stDownloadButton > button:not(:disabled):focus-visible {{
                color: #1f2937 !important;
            }}
            .pbi-export-popover div.stDownloadButton > button:not(:disabled) *,
            .pbi-export-popover div.stDownloadButton > button:not(:disabled):focus *,
            .pbi-export-popover div.stDownloadButton > button:not(:disabled):active * {{
                color: #1f2937 !important;
            }}
            div[data-testid='stPopover'] div.stDownloadButton > button,
            div[data-testid='stPopoverContent'] div.stDownloadButton > button,
            div[data-baseweb='popover'] div.stDownloadButton > button {{
                background: #ffffff !important;
                border: 1px solid rgba(15, 23, 42, 0.16) !important;
                color: #1f2937 !important;
                border-radius: 12px;
            }}
            div[data-testid='stPopover'] div.stDownloadButton > button:hover,
            div[data-testid='stPopoverContent'] div.stDownloadButton > button:hover,
            div[data-baseweb='popover'] div.stDownloadButton > button:hover {{
                border-color: rgba(25, 69, 107, 0.35) !important;
                background: rgba(25, 69, 107, 0.08) !important;
            }}
            div[data-testid='stPopover'] div.stDownloadButton > button *,
            div[data-testid='stPopoverContent'] div.stDownloadButton > button *,
            div[data-baseweb='popover'] div.stDownloadButton > button * {{
                color: inherit !important;
            }}
            div[data-testid='stPopover'] div.stDownloadButton > button:focus,
            div[data-testid='stPopoverContent'] div.stDownloadButton > button:focus,
            div[data-baseweb='popover'] div.stDownloadButton > button:focus,
            div[data-testid='stPopover'] div.stDownloadButton > button:active,
            div[data-testid='stPopoverContent'] div.stDownloadButton > button:active,
            div[data-baseweb='popover'] div.stDownloadButton > button:active,
            div[data-testid='stPopover'] div.stDownloadButton > button:focus-visible,
            div[data-testid='stPopoverContent'] div.stDownloadButton > button:focus-visible,
            div[data-baseweb='popover'] div.stDownloadButton > button:focus-visible {{
                color: #1f2937 !important;
            }}
            div[data-testid='stPopover'] div.stDownloadButton > button:focus *,
            div[data-testid='stPopoverContent'] div.stDownloadButton > button:focus *,
            div[data-baseweb='popover'] div.stDownloadButton > button:focus *,
            div[data-testid='stPopover'] div.stDownloadButton > button:active *,
            div[data-testid='stPopoverContent'] div.stDownloadButton > button:active *,
            div[data-baseweb='popover'] div.stDownloadButton > button:active *,
            div[data-testid='stPopover'] div.stDownloadButton > button:focus-visible *,
            div[data-testid='stPopoverContent'] div.stDownloadButton > button:focus-visible *,
            div[data-baseweb='popover'] div.stDownloadButton > button:focus-visible * {{
                color: inherit !important;
            }}
            div[data-testid='stPopover'] div.stDownloadButton > button:disabled,
            div[data-testid='stPopoverContent'] div.stDownloadButton > button:disabled,
            div[data-baseweb='popover'] div.stDownloadButton > button:disabled {{
                background: rgba(148, 163, 184, 0.18) !important;
                border-color: rgba(148, 163, 184, 0.45) !important;
                color: #64748B !important;
                opacity: 1 !important;
            }}
            div[data-testid='stPopover'] div.stDownloadButton > button:disabled *,
            div[data-testid='stPopoverContent'] div.stDownloadButton > button:disabled *,
            div[data-baseweb='popover'] div.stDownloadButton > button:disabled * {{
                color: #64748B !important;
            }}
            div[data-testid='stPopover'] > div {{
                background: linear-gradient(180deg, rgba(25, 69, 107, 0.10), rgba(255, 255, 255, 0.98));
                border: 1px solid rgba(25, 69, 107, 0.22);
                border-radius: 14px;
                box-shadow: 0 18px 30px rgba(15, 23, 42, 0.16);
            }}
            div[data-baseweb='segmented-control'] button[aria-pressed='true'] *,
            div[data-baseweb='segmented-control'] button:focus *,
            div[data-baseweb='segmented-control'] button:active * {{
                color: #ffffff !important;
            }}
            div[data-testid='stRadio'] label[data-baseweb='radio'] input:checked ~ * {{
                color: #ffffff !important;
            }}
            [data-baseweb='input'] input,
            [data-testid='stTextInput'] input,
            [data-testid='stTextArea'] textarea,
            [data-testid='stSelectbox'] select {{
                border: 1px solid var(--pbi-tertiary) !important;
                border-radius: 8px;
                color: var(--pbi-blue) !important;
            }}
            [data-baseweb='input'] input:focus,
            [data-testid='stTextInput'] input:focus,
            [data-testid='stTextArea'] textarea:focus,
            [data-testid='stSelectbox'] select:focus {{
                border-color: var(--pbi-green) !important;
                box-shadow: 0 0 0 1px var(--pbi-green) !important;
            }}
            div[data-testid='stCheckbox'] label div[role='checkbox'] {{
                border: 1px solid var(--pbi-tertiary) !important;
            }}
            div[data-testid='stCheckbox'] label div[role='checkbox'][aria-checked='true'] {{
                background-color: var(--pbi-green) !important;
                border-color: var(--pbi-green) !important;
            }}
            div[data-testid='stRadio'] > div {{
                display: flex;
                gap: 0.45rem;
                flex-wrap: wrap;
                align-items: center;
            }}
            div[data-testid='stRadio'] label[data-baseweb='radio'] {{
                background: rgba(144, 128, 112, 0.12);
                border: 1px solid rgba(25, 69, 107, 0.18);
                border-radius: 999px;
                padding: 0.35rem 0.9rem;
                transition: all 0.18s ease-in-out;
                cursor: pointer;
            }}
            div[data-testid='stRadio'] label[data-baseweb='radio']:hover {{
                border-color: var(--pbi-blue);
            }}
            div[data-testid='stRadio'] label[data-baseweb='radio']:has(input:checked) {{
                background: var(--pbi-blue);
                border-color: var(--pbi-blue);
                box-shadow: 0 8px 18px rgba(25, 69, 107, 0.22);
            }}
            div[data-testid='stRadio'] label[data-baseweb='radio'] span {{
                color: var(--pbi-blue);
                font-weight: 600;
            }}
            div[data-testid='stRadio'] label[data-baseweb='radio']:has(input:checked) span {{
                color: #ffffff !important;
            }}
            *[data-baseweb='tag'] {{
                background: var(--pbi-blue) !important;
                color: #ffffff !important;
                border: 1px solid var(--pbi-blue) !important;
                border-radius: 12px;
            }}
            *[data-baseweb='tag'] svg {{
                fill: #ffffff !important;
            }}
            *[data-baseweb='tag']:hover {{
                background: var(--pbi-blue) !important;
                border-color: var(--pbi-blue) !important;
                filter: brightness(0.92);
            }}
            div[data-testid='stMultiSelect'] label {{
                color: var(--pbi-blue);
                font-weight: 600;
            }}
            [data-testid='stSidebar'] *,
            [data-testid='stSidebar'] label {{
                color: var(--pbi-blue);
            }}
            div[data-testid="stVerticalBlock"] ul.nav-pills li a.nav-link {{
                color: var(--pbi-blue) !important;
            }}
            div[data-testid="stVerticalBlock"] ul.nav-pills li a.nav-link .icon {{
                color: var(--pbi-blue) !important;
            }}
            div[data-testid="stVerticalBlock"] ul.nav-pills li a.nav-link.active {{
                color: #ffffff !important;
                background: var(--pbi-blue) !important;
            }}
            div[data-testid="stVerticalBlock"] ul.nav-pills li a.nav-link.active .icon {{
                color: #ffffff !important;
            }}
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_powerbi_navigation(
    active_tab: str,
    *,
    key: str,
    orientation: str = "vertical",
    previews: Dict[str, List[Dict[str, Any]]] | None = None,
) -> str:
    styles_vertical = {
        "container": {
            "padding": "0!important",
            "background": "linear-gradient(180deg, rgba(25, 69, 107, 0.1), rgba(175, 189, 34, 0.12))",
            "border-radius": "20px",
            "border": "1px solid rgba(25, 69, 107, 0.18)",
            "box-shadow": "0 14px 32px rgba(25, 69, 107, 0.16)",
            "width": "100%",
        },
        "menu-title": {
            "font-family": "\'Segoe UI\', \'Inter\', sans-serif",
            "font-size": "0.84rem",
            "font-weight": "600",
            "letter-spacing": "0.08em",
            "text-transform": "uppercase",
            "color": "var(--pbi-blue)",
            "padding": "0.75rem 0.95rem 0.35rem",
            "margin": "0",
        },
        "nav": {
            "display": "flex",
            "flex-direction": "column",
            "gap": "0.55rem",
            "align-items": "stretch",
            "justify-content": "flex-start",
            "padding": "0 0.9rem 0.85rem",
        },
        "icon": {"color": "var(--pbi-blue)", "font-size": "1.05rem"},
        "nav-link": {
            "border-radius": "14px",
            "padding": "0.6rem 0.9rem",
            "background": "rgba(255, 255, 255, 0.92)",
            "color": "var(--pbi-blue)",
            "font-size": "0.96rem",
            "font-weight": "600",
            "border": "1px solid rgba(25, 69, 107, 0.22)",
            "transition": "background 0.2s ease, color 0.2s ease, box-shadow 0.2s ease",
            "text-align": "left",
        },
        "nav-link-selected": {
            "background": "rgba(25, 69, 107, 0.16)",
            "color": "#0F172A",
            "box-shadow": "0 14px 28px rgba(25, 69, 107, 0.28)",
            "border": "1px solid rgba(25, 69, 107, 0.32)",
            "opacity": "1",
        },
    }
    styles_horizontal = {
        "container": {"padding": "0!important", "background-color": "transparent"},
        "menu-title": {
            "font-family": "\'Segoe UI\', \'Inter\', sans-serif",
            "font-size": "0.84rem",
            "font-weight": "600",
            "letter-spacing": "0.08em",
            "text-transform": "uppercase",
            "color": "var(--pbi-blue)",
            "padding": "0.75rem 0.95rem 0.35rem",
            "margin": "0",
        },
        "nav": {"gap": "0.4rem", "justify-content": "center"},
        "icon": {"color": "#ffffff", "font-size": "1rem"},
        "nav-link": {
            "border-radius": "12px",
            "padding": "0.5rem 0.9rem",
            "background": "rgba(255, 255, 255, 0.92)",
            "color": "var(--pbi-blue)",
            "font-size": "0.95rem",
            "font-weight": "600",
            "border": "1px solid rgba(25, 69, 107, 0.16)",
        },
        "nav-link-selected": {
            "background": "rgba(25, 69, 107, 0.16)",
            "color": "#0F172A",
            "box-shadow": "0 8px 18px rgba(25, 69, 107, 0.24)",
            "border": "1px solid rgba(25, 69, 107, 0.26)",
        },
    }
    icon_list = [NAV_ICON_MAP.get(tab, "dot") for tab in NAV_TABS]

    if orientation == "vertical" and previews is not None:
        selection = _render_preview_navigation_component(
            active_tab,
            key=key,
            icon_list=icon_list,
            previews=previews or {},
        )
        if isinstance(selection, str):
            return selection
        return active_tab

    styles = styles_vertical if orientation == "vertical" else styles_horizontal
    container = st.container()
    with container:
        return option_menu(
            "Bookmarks" if orientation == "vertical" else "",
            options=NAV_TABS,
            icons=icon_list,
            menu_icon="",
            default_index=NAV_TABS.index(active_tab) if active_tab in NAV_TABS else 0,
            orientation=orientation,
            key=key,
            styles=styles,
        )

def render_export_download(
    tables: Dict[str, pd.DataFrame],
    *,
    active_tab: str = "",
) -> None:
    if not tables:
        return
    filename = f"capital_programme_dashboard_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    st.markdown("<div class='pbi-export-gap'></div>", unsafe_allow_html=True)

    if active_tab == "Programme Schedule":
        order_key = "gantt_export_order"
        if order_key not in st.session_state:
            st.session_state[order_key] = "project"
        popover = st.popover("Export current tab") if hasattr(st, "popover") else st.expander("Export current tab")
        with popover:
            st.markdown("<div class='pbi-export-popover'>", unsafe_allow_html=True)
            st.markdown("<div class='pbi-export-popover-title'>Export ordering</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='pbi-export-popover-sub'>Order by project name or by the Gantt start year.</div>",
                unsafe_allow_html=True,
            )
            order_choice = st.radio(
                "Order by",
                options=["project", "start_year"],
                format_func=lambda val: "Project order (1, 2, 3)"
                if val == "project"
                else "Project start year (Gantt order)",
                key=order_key,
                horizontal=True,
            )
            ordered_tables = _apply_programme_schedule_export_order(tables, order_choice)
            export_bytes = build_export_workbook(ordered_tables)
            st.download_button(
                "Download Excel",
                data=export_bytes,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_{hash(tuple(ordered_tables.keys())) & 0xffff}_{order_choice}",
            )
            st.markdown("</div>", unsafe_allow_html=True)
    elif active_tab == "Cash Flow":
        _inject_export_popover_focus_guard()
        breakdown_label = st.session_state.get("cash_flow_profile_spend_breakdown", "Total")
        cumulative_tables = {k: v for k, v in tables.items() if k == "Cumulative spend vs benefit"}
        bcr_tables = {k: v for k, v in tables.items() if str(k).startswith("BCR - ")}
        stack_tables = {
            k: v
            for k, v in tables.items()
            if str(k).startswith("Costs vs benefits stack") or str(k).startswith("Costs vs benefits delta")
        }
        cashflow_tables = {k: v for k, v in tables.items() if str(k).startswith("Cash flow - ")}
        popover = st.popover("Export cash flow") if hasattr(st, "popover") else st.expander("Export cash flow")
        with popover:
            st.markdown("<div class='pbi-export-popover'>", unsafe_allow_html=True)
            st.markdown("<div class='pbi-export-popover-title'>Cash flow exports</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='pbi-export-popover-sub'>Breakdown: {html.escape(str(breakdown_label))}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='pbi-export-popover-sub'>Choose what to download:</div>",
                unsafe_allow_html=True,
            )
            row1 = st.columns(2)
            with row1[0]:
                cumulative_bytes = build_export_workbook(cumulative_tables) if cumulative_tables else b""
                st.download_button(
                    "Cumulative chart",
                    data=cumulative_bytes,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_{hash(tuple(cumulative_tables.keys())) & 0xffff}_cumulative",
                    disabled=not cumulative_tables,
                    help="Cumulative spend vs benefit.",
                )
            with row1[1]:
                stack_bytes = build_export_workbook(stack_tables) if stack_tables else b""
                st.download_button(
                    "Stacked costs/benefits",
                    data=stack_bytes,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_{hash(tuple(stack_tables.keys())) & 0xffff}_stack",
                    disabled=not stack_tables,
                    help="Costs vs benefits stack using current settings.",
                )
            row2 = st.columns(2)
            with row2[0]:
                bcr_bytes = build_export_workbook(bcr_tables) if bcr_tables else b""
                st.download_button(
                    "BCR charts",
                    data=bcr_bytes,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_{hash(tuple(bcr_tables.keys())) & 0xffff}_bcr",
                    disabled=not bcr_tables,
                    help="Both BCR charts in one workbook.",
                )
            with row2[1]:
                cashflow_bytes = build_export_workbook(cashflow_tables) if cashflow_tables else b""
                st.download_button(
                    "Cash flow charts",
                    data=cashflow_bytes,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_{hash(tuple(cashflow_tables.keys())) & 0xffff}_cashflow",
                    disabled=not cashflow_tables,
                    help="Cash flow tables for both scenarios.",
                )
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        export_bytes = build_export_workbook(tables)
        st.download_button(
            "Export current tab",
            data=export_bytes,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_{hash(tuple(tables.keys())) & 0xffff}",
        )


def _json_sanitise(value: Any) -> Any:
    """Return a JSON-serialisable structure by coercing NaN/Inf to None."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_json_sanitise(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _json_sanitise(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_json_sanitise(v) for v in value]
    if isinstance(value, (list, tuple)):
        return [_json_sanitise(v) for v in value]
    if isinstance(value, pd.Series):
        return [_json_sanitise(v) for v in value.tolist()]
    if isinstance(value, pd.Index):
        return [_json_sanitise(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [_json_sanitise(row) for row in value.to_dict(orient="records")]
    return value


def _figure_title(fig: go.Figure | None, fallback: str) -> str:
    """Extract a readable title from a Plotly figure."""
    if fig is None:
        return fallback
    title = getattr(getattr(fig, "layout", None), "title", None)
    if isinstance(title, str):
        return title or fallback
    if hasattr(title, "text"):
        return title.text or fallback
    return fallback


def _figure_payload(fig: go.Figure | None) -> Dict[str, Any] | None:
    """Return a JSON-safe payload for a Plotly figure."""
    if fig is None:
        return None
    try:
        json_payload = fig.to_plotly_json()
    except Exception:
        return None
    return _json_sanitise(json_payload)


def _append_preview(
    previews: Dict[str, List[Dict[str, Any]]],
    tab: str,
    *,
    title: str,
    fig: go.Figure | None = None,
    message: str | None = None,
    limit: int = 4,
) -> None:
    """Append a preview item for a navigation tab."""
    entries = previews.setdefault(tab, [])
    if len(entries) >= limit:
        return
    payload: Dict[str, Any] = {"title": title}
    fig_payload = _figure_payload(fig)
    if fig_payload is not None:
        payload["figure"] = fig_payload
    elif message:
        payload["message"] = message
    else:
        return
    entries.append(payload)


def collect_navigation_previews(
    *,
    data: DashboardData,
    opt_selection,
    comp_selection,
    opt_series: pd.DataFrame | None,
    cmp_series: pd.DataFrame | None,
    opt_label: str,
    cmp_label: str,
    settings: Settings,
    cache_signature: tuple | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build a mapping of navigation tab names to preview figure payloads."""
    previews: Dict[str, List[Dict[str, Any]]] = {}
    profile_assets, profile_meta = load_interpolated_profile_assets()

    # Demo tab previews
    if profile_assets:
        order = profile_meta.get("profile_order") or sorted(profile_assets)
        if order:
            asset = profile_assets.get(order[0])
            if asset:
                overview_fig = go.Figure(asset["plot_json"])
                _append_preview(
                    previews,
                    "Demo",
                    title=_figure_title(overview_fig, asset.get("title", f"Cash flow - {asset['label']}")),
                    fig=overview_fig,
                )

    # Cash Flow tab previews
    eff_fig = efficiency_chart(opt_series, cmp_series, opt_selection, comp_selection)
    if eff_fig is not None:
        _append_preview(
            previews,
            "Cash Flow",
            title=_figure_title(eff_fig, "Efficiency preview"),
            fig=eff_fig,
        )
    waterfall_preview = benefit_waterfall_chart(
        data,
        opt_selection,
        comp_selection,
        horizon_years=int(st.session_state.get("npv_horizon_selection", 60)),
    )
    if waterfall_preview is not None:
        _append_preview(
            previews,
            "Cash Flow",
            title=_figure_title(waterfall_preview, "NPV waterfall"),
            fig=waterfall_preview,
        )
    bridge_preview = benefit_bridge_chart(
        data,
        opt_selection,
        comp_selection,
        horizon_years=int(st.session_state.get("npv_horizon_selection", 60)),
    )
    if bridge_preview is not None:
        _append_preview(
            previews,
            "Cash Flow",
            title=_figure_title(bridge_preview, "NPV bridge"),
            fig=bridge_preview,
        )
    radar_preview = benefit_radar_chart(
        data,
        opt_selection,
        comp_selection,
        horizon_years=int(st.session_state.get("npv_horizon_selection", 60)),
    )
    if radar_preview is not None:
        _append_preview(
            previews,
            "Cash Flow",
            title=_figure_title(radar_preview, "Benefit mix"),
            fig=radar_preview,
        )

    # Benefits tab previews
    cumulative_toggle = bool(st.session_state.get("show_cumulative_dimension_benefits", False))
    opt_dim_pivot = dimension_timeseries(data, opt_selection)
    cmp_dim_pivot = dimension_timeseries(data, comp_selection)
    opt_dim_fig = benefit_dimension_chart(
        data,
        opt_selection,
        title=f"{opt_label} dimension mix",
        cumulative=cumulative_toggle,
        pivot=opt_dim_pivot,
    )
    if opt_dim_fig is not None:
        _append_preview(
            previews,
            "Benefits",
            title=_figure_title(opt_dim_fig, f"{opt_label} mix"),
            fig=opt_dim_fig,
        )
    cmp_dim_fig = benefit_dimension_chart(
        data,
        comp_selection,
        title=f"{cmp_label} dimension mix",
        cumulative=cumulative_toggle,
        pivot=cmp_dim_pivot,
    )
    if cmp_dim_fig is not None:
        _append_preview(
            previews,
            "Benefits",
            title=_figure_title(cmp_dim_fig, f"{cmp_label} mix"),
            fig=cmp_dim_fig,
        )
    benefit_fig = benefit_chart(opt_series, cmp_series, dimension=opt_selection.dimension)
    if benefit_fig is not None:
        _append_preview(
            previews,
            "Benefits",
            title=_figure_title(benefit_fig, "Benefit profile"),
            fig=benefit_fig,
        )
    delta_fig = benefit_delta_chart(
        data,
        opt_series,
        cmp_series,
        opt_selection,
        comp_selection,
        horizon_years=int(st.session_state.get("npv_horizon_selection", 60)),
    )
    if delta_fig is not None:
        _append_preview(
            previews,
            "Benefits",
            title=_figure_title(delta_fig, "Benefit delta"),
            fig=delta_fig,
        )

    # Delivery tab previews
    project_colors = project_color_map(data)
    schedule_year_range = _effective_year_range([opt_series, cmp_series], value_column="Spend")
    schedule_opt_fig = project_schedule_area_chart(
        data,
        opt_selection,
        title=f"Schedule - {opt_label}",
        color_map=project_colors,
        display_year_range=schedule_year_range,
    )
    if schedule_opt_fig is not None:
        _append_preview(
            previews,
            "Delivery",
            title=_figure_title(schedule_opt_fig, f"Schedule - {opt_label}"),
            fig=schedule_opt_fig,
        )
    schedule_cmp_fig = project_schedule_area_chart(
        data,
        comp_selection,
        title=f"Schedule - {cmp_label}",
        color_map=project_colors,
        display_year_range=schedule_year_range,
    )
    if schedule_cmp_fig is not None:
        _append_preview(
            previews,
            "Delivery",
            title=_figure_title(schedule_cmp_fig, f"Schedule - {cmp_label}"),
            fig=schedule_cmp_fig,
        )
    cap_opt_fig = market_capacity_indicator(data, opt_selection)
    if cap_opt_fig is not None:
        _append_preview(
            previews,
            "Delivery",
            title=_figure_title(cap_opt_fig, "Market capacity (optimised)"),
            fig=cap_opt_fig,
        )
    cap_cmp_fig = market_capacity_indicator(data, comp_selection)
    if cap_cmp_fig is not None:
        _append_preview(
            previews,
            "Delivery",
            title=_figure_title(cap_cmp_fig, f"Market capacity ({cmp_label})"),
            fig=cap_cmp_fig,
        )

    # Programme schedule preview
    primary_for_gantt = opt_selection if getattr(opt_selection, "code", None) else comp_selection
    comparison_for_gantt = comp_selection if primary_for_gantt is opt_selection else opt_selection
    gantt_fig = spend_gantt_chart(
        data,
        primary_for_gantt,
        comparison_selection=comparison_for_gantt,
        show_outline=True,
        title=f"Programme schedule - {opt_label if primary_for_gantt is opt_selection else cmp_label}",
    )
    if gantt_fig is not None:
        _append_preview(
            previews,
            "Programme Schedule",
            title=_figure_title(gantt_fig, "Programme schedule"),
            fig=gantt_fig,
        )

    # Regions preview
    preview_selection = opt_selection if getattr(opt_selection, "code", None) else comp_selection
    preview_label = opt_label if preview_selection is opt_selection else cmp_label
    scenario_code = getattr(preview_selection, "code", None)
    if scenario_code:
        spread_national_enabled = bool(st.session_state.get("region_spread_national_toggle", True))
        cache_bucket = st.session_state.setdefault("_region_metrics_cache", {})
        cache_key = (cache_signature or "preview", scenario_code, spread_national_enabled, spread_national_enabled)
        metrics_df = cache_bucket.get(cache_key)
        if metrics_df is None:
            try:
                metrics_df = compute_region_metrics(
                    data,
                    scenario_code,
                    spread_national=spread_national_enabled,
                    spread_unmapped=spread_national_enabled,
                )
            except Exception:
                metrics_df = None
            else:
                cache_bucket[cache_key] = metrics_df
        region_preview_added = False
        if metrics_df is not None and not metrics_df.empty:
            default_metric = REGION_METRIC_DEFAULT["Spend share"]
            region_col = next(
                (
                    col
                    for col in metrics_df.columns
                    if str(col).strip().lower()
                    in {"region", "region_name", "region label", "region_label"}
                ),
                None,
            )
            if region_col and default_metric in metrics_df.columns:
                available_years = metrics_df["Year"].dropna()
                if not available_years.empty:
                    latest_year = int(available_years.max())
                    sample_df = (
                        metrics_df.loc[metrics_df["Year"] == latest_year]
                        .nlargest(6, default_metric)
                        [[region_col, default_metric]]
                        .copy()
                    )
                    if not sample_df.empty:
                        sample_df[region_col] = sample_df[region_col].astype(str)
                        bar_fig = go.Figure()
                        bar_fig.add_bar(
                            x=sample_df[default_metric],
                            y=sample_df[region_col],
                            orientation="h",
                            marker=dict(color=POWERBI_BLUE),
                            hovertemplate="%{y}<br>%{x:.1%}<extra></extra>",
                        )
                        bar_fig.update_layout(
                            title=f"{preview_label}: Top regions by spend share",
                            xaxis=dict(tickformat=".0%", title="Share"),
                            yaxis=dict(autorange="reversed", title=""),
                            margin=dict(l=60, r=20, t=40, b=40),
                            template=plotly_template(),
                        )
                        _append_preview(
                            previews,
                            "Regions",
                            title="Regional spend share",
                            fig=bar_fig,
                        )
                        region_preview_added = True
        if not region_preview_added:
            _append_preview(
                previews,
                "Regions",
                title="Regional preview",
                message=f"No regional metrics available for {preview_label}.",
            )
    else:
        _append_preview(
            previews,
            "Regions",
            title="Regional preview",
            message="Select a scenario to view regional metrics.",
        )

    # Scenario manager preview placeholder
    _append_preview(
        previews,
        "Scenario Manager",
        title="Scenario workspace",
        message="Organise optimisation batches and create new scenario folders here.",
    )

    for tab in NAV_TABS:
        entries = previews.setdefault(tab, [])
        if not entries:
            entries.append(
                {
                    "title": f"{tab} preview unavailable",
                    "message": "No preview is available yet for this scenario. Select the bookmark to explore it in full.",
                }
            )
    return previews


def _render_preview_navigation_component(
    active_tab: str,
    *,
    key: str,
    icon_list: List[str],
    previews: Dict[str, List[Dict[str, Any]]],
) -> Optional[str]:
    """Render the React-based navigation panel with bookmark previews."""
    tabs_payload: List[Dict[str, Any]] = []
    default_tabs: List[Dict[str, Any]] = []
    for idx, tab in enumerate(NAV_TABS):
        icon = (icon_list[idx] if idx < len(icon_list) else NAV_ICON_MAP.get(tab, "dot")) or "dot"
        tabs_payload.append(
            {
                "name": tab,
                "icon": icon,
                "previews": previews.get(tab, []),
            }
        )
        default_tabs.append({"name": tab, "icon": icon})

    initial_tab = active_tab if active_tab in NAV_TABS else NAV_TABS[0]
    component_data = _json_sanitise({"tabs": tabs_payload, "activeTab": initial_tab})
    response = _preview_navigation_component(
        data=component_data,
        defaultTabs=_json_sanitise(default_tabs),
        colors={"primary": POWERBI_BLUE, "accent": POWERBI_GREEN},
        key=f"{key}_preview_nav",
        default=initial_tab,
    )
    if isinstance(response, str) and response in NAV_TABS:
        return response
    return active_tab


BRIGHT_PRIMARY_COLOR = POWERBI_BLUE
BRIGHT_COMPARISON_COLOR = POWERBI_GREEN

BENEFIT_SHADE_COLOR = "rgba(25, 69, 107, 0.18)"

WATERFALL_CHART_HEIGHT = 420

PLOTLY_TRANSPARENT_STYLE = """
<style>
div[data-testid="stPlotlyChart"] > div {
    background-color: transparent !important;
}
</style>
"""

CUMULATIVE_OPT_LINE_COLOR = POWERBI_BLUE
CUMULATIVE_CMP_LINE_COLOR = POWERBI_GREEN







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
    _ensure_global_hoverformat()
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


_GLOBAL_HOVERFORMAT = ".2~f"
_HOVERFORMAT_READY = False


def _ensure_global_hoverformat() -> None:
    """Apply a global hoverformat cap to Plotly templates."""
    global _HOVERFORMAT_READY
    if _HOVERFORMAT_READY:
        return
    for template_name in ("plotly_white", "plotly_dark"):
        try:
            template = pio.templates[template_name]
        except Exception:
            continue
        try:
            template.layout.xaxis.hoverformat = _GLOBAL_HOVERFORMAT
            template.layout.yaxis.hoverformat = _GLOBAL_HOVERFORMAT
        except Exception:
            pass
        try:
            template.layout.polar.radialaxis.hoverformat = _GLOBAL_HOVERFORMAT
            template.layout.polar.angularaxis.hoverformat = _GLOBAL_HOVERFORMAT
        except Exception:
            pass
    _HOVERFORMAT_READY = True


DEFAULT_LEGEND_BOTTOM: Dict[str, Any] = {
    "orientation": "h",
    "yanchor": "top",
    "y": -0.18,
    "xanchor": "center",
    "x": 0.5,
    "title": None,
}


def legend_bottom(**overrides: Any) -> Dict[str, Any]:
    """Return a legend config anchored below the chart area."""
    legend = dict(DEFAULT_LEGEND_BOTTOM)
    if overrides:
        legend.update(overrides)
    return legend


def _normalise_project_key(name: str) -> str:
    """Return a canonical project key for matching across scenarios."""
    if name is None:
        return ''
    normalised = str(name).strip().lower()
    return re.sub(r'\s+', ' ', normalised)


def _clean_mapping_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _canonicalise_gps_tier(value: Any) -> str:
    text = _clean_mapping_value(value)
    if text is None:
        return "Unknown"
    normalised = re.sub(r"[\s_-]+", " ", text).strip().lower()
    if not normalised or normalised == "unknown":
        return "Unknown"
    if "must" in normalised:
        return "Must do"
    if "should" in normalised:
        return "Should do"
    if "could" in normalised:
        return "Could do"
    return text.strip()


def _read_excel_safe(path: Path, *, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except Exception:
        pass
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=path.suffix) as tmp:
            tmp_path = Path(tmp.name)
        try:
            shutil.copy2(path, tmp_path)
            return pd.read_excel(tmp_path, sheet_name=sheet_name, engine="openpyxl")
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass
    except Exception:
        return pd.DataFrame()


def _build_project_attribute_mapping(workbook_path: Path) -> pd.DataFrame:
    """Build project-to-attribute mappings used for breakdown charts."""

    def _read_sheet(sheet_name: str, column_map: Dict[str, str]) -> pd.DataFrame:
        if not workbook_path.exists():
            return pd.DataFrame(columns=["ProjectKey", "Project", *column_map.keys()])
        df = _read_excel_safe(workbook_path, sheet_name=sheet_name)
        if df.empty:
            return pd.DataFrame(columns=["ProjectKey", "Project", *column_map.keys()])

        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]
        lookup = {str(col).strip().lower(): col for col in df.columns}
        project_col = lookup.get("project")
        if project_col is None:
            return pd.DataFrame(columns=["ProjectKey", "Project", *column_map.keys()])

        out = pd.DataFrame()
        out["Project"] = df[project_col].map(_clean_mapping_value)
        out = out.dropna(subset=["Project"]).copy()
        out["Project"] = out["Project"].astype(str).str.strip()
        out = out[out["Project"].ne("")]
        out["ProjectKey"] = out["Project"].map(_normalise_project_key)

        for out_col, src_col_name in column_map.items():
            src_col = lookup.get(str(src_col_name).strip().lower())
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
            columns=["ProjectKey", "Project", "Region", "ActivityClass", "GPSTier", "StrategicDimension"]
        )

    merged = costs.set_index("ProjectKey").combine_first(benefits.set_index("ProjectKey")).reset_index()
    for col in ("Project", "Region", "ActivityClass", "GPSTier", "StrategicDimension"):
        if col not in merged.columns:
            merged[col] = None
    merged = merged[["ProjectKey", "Project", "Region", "ActivityClass", "GPSTier", "StrategicDimension"]].copy()
    merged["Region"] = merged["Region"].fillna("Unknown")
    merged["ActivityClass"] = merged["ActivityClass"].fillna("Unknown")
    merged["GPSTier"] = merged["GPSTier"].fillna("Unknown")
    merged["StrategicDimension"] = merged["StrategicDimension"].fillna("Unknown")
    return merged


@st.cache_resource(show_spinner=False)
def load_project_attribute_mapping(
    workbook_path: Path = COST_BENEFIT_STREAMS_PATH,
) -> pd.DataFrame:
    """Load project-to-attribute mappings used for breakdown charts."""
    return _build_project_attribute_mapping(workbook_path)


@st.cache_resource(show_spinner=False)
def load_project_bcr_mapping(
    workbook_path: Path = BCR_MAPPING_PATH,
) -> pd.DataFrame:
    """Return per-project original BCR values from the mapping workbook."""
    if not workbook_path.exists():
        return pd.DataFrame(columns=["ProjectKey", "Project", "BCR (Original)"])
    df = _read_excel_safe(workbook_path, sheet_name=0)
    if df.empty:
        return pd.DataFrame(columns=["ProjectKey", "Project", "BCR (Original)"])

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    lookup = {str(col).strip().lower(): col for col in df.columns}
    project_col = lookup.get("project")
    bcr_col = lookup.get("bcr")
    if project_col is None or bcr_col is None:
        return pd.DataFrame(columns=["ProjectKey", "Project", "BCR (Original)"])

    out = pd.DataFrame()
    out["Project"] = df[project_col].map(_clean_mapping_value)
    out = out.dropna(subset=["Project"]).copy()
    out["Project"] = out["Project"].astype(str).str.strip()
    out = out[out["Project"].ne("")]
    out["ProjectKey"] = out["Project"].map(_normalise_project_key)
    out["BCR (Original)"] = pd.to_numeric(df[bcr_col], errors="coerce")
    out = out.drop_duplicates(subset=["ProjectKey"], keep="first")
    return out[["ProjectKey", "Project", "BCR (Original)"]]


@st.cache_resource(show_spinner=False)
def load_project_new_bcr_mapping(
    workbook_path: Path = COST_BENEFIT_STREAMS_PATH,
    *,
    base_year: Optional[int] = None,
) -> pd.DataFrame:
    """Return per-project BCR from PV'd cost/benefit cashflows (40-year benefits)."""
    if not workbook_path.exists():
        return pd.DataFrame(columns=["ProjectKey", "Project", "BCR (New)"])
    costs = _read_excel_safe(workbook_path, sheet_name="Costs")
    benefits = _read_excel_safe(workbook_path, sheet_name="Benefits Linear 40yrs")
    if costs.empty or benefits.empty:
        return pd.DataFrame(columns=["ProjectKey", "Project", "BCR (New)"])

    if costs is None or costs.empty or "Project" not in costs.columns:
        return pd.DataFrame(columns=["ProjectKey", "Project", "BCR (New)"])
    if benefits is None or benefits.empty or "Project" not in benefits.columns:
        return pd.DataFrame(columns=["ProjectKey", "Project", "BCR (New)"])

    costs = costs.copy()
    costs["Project"] = costs["Project"].map(_clean_mapping_value)
    costs = costs.dropna(subset=["Project"]).copy()
    costs["Project"] = costs["Project"].astype(str).str.strip()
    costs = costs[costs["Project"].ne("")]

    benefits = benefits.copy()
    benefits["Project"] = benefits["Project"].map(_clean_mapping_value)
    benefits = benefits.dropna(subset=["Project"]).copy()
    benefits["Project"] = benefits["Project"].astype(str).str.strip()
    benefits = benefits[benefits["Project"].ne("")]

    year_map: Dict[Any, int] = {}
    for col in costs.columns:
        if isinstance(col, (int, np.integer)):
            year_map[col] = int(col)
        elif isinstance(col, str) and col.strip().isdigit():
            year_map[col] = int(col.strip())
    if not year_map:
        return pd.DataFrame(columns=["ProjectKey", "Project", "BCR (New)"])
    costs = costs.rename(columns=year_map)
    year_cols = sorted(set(year_map.values()))
    costs[year_cols] = costs[year_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if base_year is None:
        base_year = int(min(year_cols))

    t_map: Dict[Any, int] = {}
    for col in benefits.columns:
        if not isinstance(col, str):
            continue
        match = re.match(r"t\\s*\\+\\s*(\\d+)", col.strip().lower())
        if match:
            t_map[col] = int(match.group(1))
    if not t_map:
        return pd.DataFrame(columns=["ProjectKey", "Project", "BCR (New)"])
    benefits = benefits.rename(columns=t_map)
    t_cols = sorted(set(t_map.values()))
    benefits[t_cols] = benefits[t_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if "Dimension" in benefits.columns:
        dim_values = benefits["Dimension"].astype(str).str.strip().str.lower()
        total_mask = dim_values == "total"
        if total_mask.any():
            benefits = benefits[total_mask]

    cost_table = costs.groupby("Project")[year_cols].sum()
    benefit_table = benefits.groupby("Project")[t_cols].sum()

    records: List[Dict[str, Any]] = []
    year_array = np.asarray(year_cols, dtype=int)
    offset_array = np.asarray(t_cols, dtype=int)

    for project, cost_row in cost_table.iterrows():
        cost_vals = cost_row.to_numpy(dtype=float)
        nonzero_mask = np.abs(cost_vals) > 1e-9
        if not nonzero_mask.any():
            continue
        start_year = int(year_array[np.argmax(nonzero_mask)])
        cost_offsets = year_array - int(base_year)
        cost_pv = float(
            np.sum(cost_vals / mbcm_discount_divisors(cost_offsets))
        ) if cost_vals.size else 0.0

        bcr_value = np.nan
        if cost_pv > 0 and project in benefit_table.index and offset_array.size:
            benefit_vals = benefit_table.loc[project].to_numpy(dtype=float)
            benefit_years = start_year + offset_array
            benefit_offsets = benefit_years - int(base_year)
            benefit_pv = float(
                np.sum(benefit_vals / mbcm_discount_divisors(benefit_offsets))
            )
            if np.isfinite(benefit_pv):
                bcr_value = benefit_pv / cost_pv

        records.append(
            {
                "Project": project,
                "ProjectKey": _normalise_project_key(project),
                "BCR (New)": bcr_value,
            }
        )

    if not records:
        return pd.DataFrame(columns=["ProjectKey", "Project", "BCR (New)"])
    out = pd.DataFrame(records)
    out = out.drop_duplicates(subset=["ProjectKey"], keep="first")
    return out[["ProjectKey", "Project", "BCR (New)"]]

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

    benefit_level: str

    mode: str

    envelope: Optional[int]

    buffer_value: Optional[int]

    dimension: str

    profile: Optional[str] = None

    metadata: Optional[Dict[str, object]] = None

    display_label: Optional[str] = None


def resolve_selection_label(
    selection: Optional["ScenarioSelection"],
    *,
    fallback: str,
    profile_choice: Optional[str] = None,
) -> str:
    """Return a human-friendly label for the supplied selection."""
    candidates: List[Optional[str]] = []
    if selection is not None:
        display_override = getattr(selection, "display_label", None)
        candidates.append(display_override)
        candidates.extend(
            [
                selection.profile,
            ]
        )
        meta = selection.metadata or {}
        candidates.extend(
            [
                str(meta.get("Profile")) if meta.get("Profile") is not None else None,
                str(meta.get("ScenarioLabel")) if meta.get("ScenarioLabel") is not None else None,
                str(meta.get("ScenarioName")) if meta.get("ScenarioName") is not None else None,
            ]
        )
        candidates.append(selection.name)
        candidates.append(selection.code)
    if profile_choice is not None:
        candidates.append(profile_choice)
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return fallback

LOAD_DATA_SCHEMA_VERSION = 5


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


@st.cache_resource(show_spinner=False)
def load_interpolated_profile_payload(
    path: Path = INTERPOLATED_PROFILES_PATH,
) -> Tuple[Dict[int, pd.DataFrame], Dict[str, Any]]:
    """Load the interpolated profile states used for the animated cash flow chart."""
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except FileNotFoundError:
        return {}, {}
    except Exception as exc:  # pragma: no cover - defensive guard for unexpected pickle issues
        return {}, {"load_error": str(exc)}

    profiles_df = payload.get("profiles_long")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)

    if not isinstance(profiles_df, pd.DataFrame):
        metadata.setdefault("load_error", "Interpolated profile payload did not contain a DataFrame.")
        return {}, metadata

    profiles_df = profiles_df.copy()
    profiles_df.columns = [str(col).strip() for col in profiles_df.columns]
    profiles_df["Profile"] = pd.to_numeric(profiles_df.get("Profile"), errors="coerce")
    profiles_df["Year"] = pd.to_numeric(profiles_df.get("Year"), errors="coerce")
    profiles_df = profiles_df.dropna(subset=["Profile"]).sort_values(["Profile", "Year"])

    profile_frames: Dict[int, pd.DataFrame] = {}
    for profile_value, group in profiles_df.groupby("Profile"):
        if pd.isna(profile_value):
            continue
        frame = group.copy()
        for column in ("Annual spend", "Envelope", "Closing net"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        profile_frames[int(profile_value)] = frame.reset_index(drop=True)

    label_map = metadata.get("profile_labels")
    if isinstance(label_map, dict):
        metadata["profile_labels"] = {int(k): v for k, v in label_map.items()}
    npv_map = metadata.get("npv_by_profile")
    if isinstance(npv_map, dict):
        cleaned_npv: Dict[int, float] = {}
        for key, value in npv_map.items():
            try:
                key_int = int(key)
            except (TypeError, ValueError):
                continue
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value_float):
                cleaned_npv[key_int] = value_float
        metadata["npv_by_profile"] = cleaned_npv
    metadata.setdefault("n_profiles", len(profile_frames))
    return profile_frames, metadata


@st.cache_resource(show_spinner=False)
def load_interpolated_progress(path: Path = PCT_BAR_PATH) -> List[float]:
    """Load cumulative progress percentages for each interpolated tick."""
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except FileNotFoundError:
        return []
    except Exception:
        return []

    if isinstance(payload, pd.DataFrame):
        df = payload.copy()
    else:
        try:
            df = pd.DataFrame(payload)
        except Exception:
            return []

    df.columns = [str(col).strip() for col in df.columns]
    tick_col = next((col for col in df.columns if str(col).strip().lower() in {"tick", "ticks"}), None)
    pct_col = next(
        (
            col
            for col in df.columns
            if "pct" in str(col).strip().lower()
            or "percent" in str(col).strip().lower()
        ),
        None,
    )
    if pct_col is None:
        return []

    if tick_col is not None:
        df[tick_col] = pd.to_numeric(df[tick_col], errors="coerce")
        df = df.sort_values(tick_col, na_position="last")

    pct_series = pd.to_numeric(df[pct_col], errors="coerce").fillna(method="ffill").fillna(0.0)
    if pct_series.max() > 1.0 + 1e-6:
        pct_series = pct_series / 100.0
    pct_series = pct_series.clip(lower=0.0, upper=1.0)
    return pct_series.tolist()


def _prepare_interpolated_profile_chart(profile_df: pd.DataFrame) -> pd.DataFrame | None:
    """Return a chart-ready DataFrame for the interpolated profile animation."""
    required_cols = {"Year", "Annual spend", "Envelope", "Closing net"}
    if not required_cols.issubset(profile_df.columns):
        return None

    chart_frame = profile_df.loc[:, ["Year", "Annual spend", "Closing net", "Envelope"]].copy()
    chart_frame.rename(
        columns={"Annual spend": "Spend", "Closing net": "ClosingNet"},
        inplace=True,
    )
    chart_frame["Year"] = pd.to_numeric(chart_frame["Year"], errors="coerce")
    for column in ("Spend", "ClosingNet", "Envelope"):
        chart_frame[column] = (
            pd.to_numeric(chart_frame[column], errors="coerce").fillna(0.0) / 1_000_000.0
        )
    chart_frame = chart_frame.dropna(subset=["Year"])
    return chart_frame


@st.cache_resource(show_spinner=False)
def load_interpolated_profile_assets(
    path: Path = INTERPOLATED_PROFILES_PATH,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Prepare chart-ready assets for each interpolated profile state."""
    profile_frames, metadata = load_interpolated_profile_payload(path)
    if not profile_frames:
        return {}, metadata

    label_map = metadata.get("profile_labels")
    if not isinstance(label_map, dict):
        label_map = {}

    progress_values = load_interpolated_progress()
    npv_by_profile = metadata.get("npv_by_profile")
    if not isinstance(npv_by_profile, dict):
        npv_by_profile = {}

    assets: Dict[int, Dict[str, Any]] = {}
    for profile_idx, profile_df in profile_frames.items():
        chart_frame = _prepare_interpolated_profile_chart(profile_df)
        if chart_frame is None or chart_frame.empty:
            continue
        export_frame = profile_df.loc[:, ["Year", "Annual spend", "Envelope", "Closing net"]].copy()
        label = label_map.get(profile_idx)
        if not label and "ProfileLabel" in profile_df.columns:
            label_series = profile_df["ProfileLabel"].dropna()
            if not label_series.empty:
                label = str(label_series.iloc[0])
        label = label or f"Profile {profile_idx}"

        figure = cash_chart(
            chart_frame,
            title="",
            color=PRIMARY_COLOR,
        )
        figure.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            transition={"duration": 50, "easing": "cubic-in-out"},
            uirevision="interpolated_profiles",
        )
        figure.update_layout(title=None)
        figure_json = figure.to_plotly_json()
        status_text = label
        if "–" in status_text:
            status_text = status_text.split("–", 1)[1].strip()
        elif "-" in status_text:
            status_text = status_text.split("-", 1)[1].strip()
        try:
            idx_int = int(profile_idx)
        except (TypeError, ValueError):
            idx_int = None
        progress_value = None
        if idx_int is not None and progress_values:
            pos = idx_int - 1
            if 0 <= pos < len(progress_values):
                try:
                    progress_value = float(progress_values[pos])
                except (TypeError, ValueError):
                    progress_value = None
        npv_value = None
        if "NPV (m)" in profile_df.columns:
            npv_series = pd.to_numeric(profile_df["NPV (m)"], errors="coerce").dropna()
            if not npv_series.empty:
                npv_value = float(npv_series.iloc[0])
        if npv_value is None and npv_by_profile:
            candidate = npv_by_profile.get(profile_idx)
            if candidate is None:
                candidate = npv_by_profile.get(str(profile_idx))
            if candidate is not None:
                try:
                    npv_value = float(candidate)
                except (TypeError, ValueError):
                    npv_value = None
        if npv_value is not None and not math.isfinite(npv_value):
            npv_value = None

        assets[profile_idx] = {
            "label": label,
            "status_text": status_text.strip(),
            "progress": progress_value,
            "chart_frame": chart_frame,
            "figure": go.Figure(figure_json),
            "plot_json": figure_json,
            "title": (
                figure_json.get("layout", {})
                .get("title", {})
                .get("text", "")
            ),
            "export": export_frame,
            "npv": npv_value,
        }

    enriched_metadata = dict(metadata)
    enriched_metadata.setdefault("n_profiles", len(assets))
    enriched_metadata["profile_order"] = sorted(assets)
    return assets, enriched_metadata


def format_currency(value: float) -> str:

    return f"{value:,.0f} m" if np.isfinite(value) else "-"

def format_large_amount(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    return f"{value / 1_000_000:.1f}M"

def compute_cash_axis_ticks(
    values: Iterable[float],
    *,
    force_unit: Optional[str] = None,
) -> Tuple[List[float], List[str], str]:
    forced = (force_unit or "").strip().lower()
    if forced in {"b", "billion", "billions"}:
        forced_suffix = "b"
    elif forced in {"m", "million", "millions"}:
        forced_suffix = "m"
    else:
        forced_suffix = None

    finite_values = [val for val in values if np.isfinite(val)]
    if not finite_values:
        suffix = forced_suffix or "m"
        unit_value = 1_000_000_000 if suffix == "b" else 1_000_000
        unit_label = "billions" if suffix == "b" else "millions"
        return [0.0], ["0"], unit_label

    min_val = min(finite_values)
    max_val = max(finite_values)
    max_abs = max(abs(min_val), abs(max_val))

    if forced_suffix:
        unit_suffix = forced_suffix
        unit_value = 1_000_000_000 if unit_suffix == "b" else 1_000_000
        unit_label = "billions" if unit_suffix == "b" else "millions"
    else:
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

MBCM_DISCOUNT_RATE_FIRST = 0.02
MBCM_DISCOUNT_RATE_LATER = 0.015
MBCM_DISCOUNT_FIRST_YEARS = 30


def mbcm_discount_divisors(offsets: np.ndarray) -> np.ndarray:
    """Return discount divisors per NZTA MBCM (2% for first 30 years, 1.5% thereafter)."""
    year_offsets = np.asarray(offsets)
    if year_offsets.size == 0:
        return year_offsets.astype(float)
    year_offsets = np.clip(year_offsets.astype(int), a_min=0, a_max=None)
    first_years = np.minimum(year_offsets, MBCM_DISCOUNT_FIRST_YEARS)
    later_years = np.maximum(year_offsets - MBCM_DISCOUNT_FIRST_YEARS, 0)
    divisors = np.power(1.0 + MBCM_DISCOUNT_RATE_FIRST, first_years) * np.power(
        1.0 + MBCM_DISCOUNT_RATE_LATER, later_years
    )
    divisors[divisors == 0] = 1.0
    return divisors


def mbcm_discount_label() -> str:
    return (
        f"{MBCM_DISCOUNT_RATE_FIRST * 100:.1f}% for years 1-{MBCM_DISCOUNT_FIRST_YEARS}, "
        f"{MBCM_DISCOUNT_RATE_LATER * 100:.1f}% from year {MBCM_DISCOUNT_FIRST_YEARS + 1}+"
    )


VALUE_BASIS_PV = "PV'd at MBCM rates"
VALUE_BASIS_REAL = "Real $2025"
VALUE_BASIS_NOMINAL = "Nominal"
VALUE_BASIS_OPTIONS = (VALUE_BASIS_PV, VALUE_BASIS_REAL, VALUE_BASIS_NOMINAL)


def _resolve_value_basis(session_key: str, legacy_apply_key: str, *, default: str = VALUE_BASIS_REAL) -> str:
    """Resolve the selected value basis, falling back to legacy checkbox state when present."""
    basis = st.session_state.get(session_key)
    if basis in VALUE_BASIS_OPTIONS:
        return str(basis)
    legacy = st.session_state.get(legacy_apply_key)
    if legacy is not None:
        return VALUE_BASIS_PV if bool(legacy) else VALUE_BASIS_REAL
    return default


def value_basis_multiplier(start_year: int, years: np.ndarray, basis: str) -> np.ndarray:
    """Return per-year multipliers to convert base $2025 flows into the selected basis."""
    year_values = np.asarray(years, dtype=int)
    offsets = np.clip(year_values - int(start_year), a_min=0, a_max=None)
    divisors = mbcm_discount_divisors(offsets)
    basis_norm = (basis or "").strip().lower()
    if basis_norm.startswith("pv"):
        return np.divide(1.0, divisors, out=np.ones_like(divisors, dtype=float), where=divisors != 0)
    if basis_norm.startswith("nom"):
        return divisors.astype(float)
    return np.ones_like(divisors, dtype=float)



def _create_export_table(years: Iterable[int]) -> Tuple[pd.DataFrame, pd.Index]:
    index = pd.Index([int(year) for year in years], name="Financial year")
    return pd.DataFrame(index=index), index


def pv_to_nominal_index(data: DashboardData, years: Iterable[int]) -> pd.Series:
    """Return discount multipliers to convert PV back to nominal dollars."""
    start_year = int(getattr(data, "start_fy", 0))
    year_index = pd.Index([int(year) for year in years], name="Financial year")
    offsets = np.clip((year_index - start_year).to_numpy(dtype=int), a_min=0, a_max=None)
    multipliers = mbcm_discount_divisors(offsets)
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
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        seen: Set[str] = set()
        for name, df in tables.items():
            sheet_name = sanitize_sheet_name(name, seen)
            (df if df is not None else pd.DataFrame()).to_excel(writer, sheet_name=sheet_name, index=False)
            if df is None or df.empty:
                continue
            sheet = writer.sheets.get(sheet_name)
            if sheet is None:
                continue
            weight_cols = [
                idx + 1
                for idx, col in enumerate(df.columns)
                if isinstance(col, str) and col.startswith("Strategic Dimension ")
            ]
            bcr_cols = [
                idx + 1
                for idx, col in enumerate(df.columns)
                if isinstance(col, str) and "bcr" in col.lower()
            ]
            year_cols = []
            for idx, col in enumerate(df.columns):
                if isinstance(col, (int, np.integer)):
                    year_value = int(col)
                elif isinstance(col, str) and col.isdigit():
                    year_value = int(col)
                else:
                    continue
                if year_value >= 2026:
                    year_cols.append(idx + 1)
            format_cols = sorted(set(weight_cols + bcr_cols))
            max_row = sheet.max_row
            for col_idx in format_cols:
                for row_idx in range(2, max_row + 1):
                    sheet.cell(row=row_idx, column=col_idx).number_format = "0.00"
            if year_cols and "Project" in df.columns:
                accounting_format = '_(* #,##0.00_);_(* (#,##0.00);_(* "-"??_);_(@_)'
                for col_idx in year_cols:
                    for row_idx in range(2, max_row + 1):
                        sheet.cell(row=row_idx, column=col_idx).number_format = accounting_format
    buffer.seek(0)
    return buffer.getvalue()

def prepare_efficiency_export(
    data: DashboardData,
    opt_df: Optional[pd.DataFrame],
    cmp_df: Optional[pd.DataFrame],
    opt_selection: ScenarioSelection,
    cmp_selection: ScenarioSelection,
) -> Optional[pd.DataFrame]:
    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()
    cost_basis = _resolve_value_basis(
        "npv_cost_value_basis",
        "npv_apply_cost_discount",
        default=VALUE_BASIS_PV,
    )
    benefit_basis = _resolve_value_basis(
        "npv_benefit_value_basis",
        "npv_apply_discount",
        default=VALUE_BASIS_PV,
    )
    try:
        pv_horizon_years = int(st.session_state.get("npv_horizon_selection", 60))
    except (TypeError, ValueError):
        pv_horizon_years = 60
    start_year = int(getattr(data, "start_fy", 2025) or 2025)

    def _basis_descriptor(basis: str) -> str:
        basis_norm = (basis or "").strip().lower()
        if basis_norm.startswith("pv"):
            return "PV"
        if basis_norm.startswith("nom"):
            return "nominal"
        return "real $2025"

    def _scale_to_dollars(values: pd.Series) -> float:
        values = pd.to_numeric(values, errors="coerce").fillna(0.0)
        magnitude = float(values.abs().max()) if not values.empty else 0.0
        if magnitude >= 10_000_000.0:
            return 1.0
        return 1_000_000.0

    def _cumulative_series_dollars(
        frame: pd.DataFrame,
        *,
        flow_column: str,
        basis: str,
        horizon_years: Optional[int],
    ) -> pd.Series:
        years_raw = pd.to_numeric(frame.get("Year"), errors="coerce")
        flows_raw = pd.to_numeric(frame.get(flow_column), errors="coerce").fillna(0.0)
        mask = years_raw.notna()
        if not mask.any():
            return pd.Series(dtype=float)
        years = years_raw[mask].astype(int)
        flows = flows_raw[mask].astype(float)
        if basis == VALUE_BASIS_PV and horizon_years:
            limit_year = int(start_year) + int(horizon_years) - 1
            flows = flows.where(years <= limit_year, 0.0)
        multipliers = value_basis_multiplier(int(start_year), years.to_numpy(dtype=int), basis)
        adjusted = flows.to_numpy(dtype=float) * multipliers
        cumulative = np.cumsum(adjusted)
        scale = _scale_to_dollars(flows)
        return pd.Series(cumulative * scale, index=years.to_numpy(dtype=int), dtype=float)

    years = sorted_years(opt_df, cmp_df)
    if not years:
        return None
    export, index = _create_export_table(years)
    if opt_df is not None and not opt_df.empty:
        spend_series = _cumulative_series_dollars(
            opt_df,
            flow_column="Spend",
            basis=cost_basis,
            horizon_years=pv_horizon_years if cost_basis == VALUE_BASIS_PV else None,
        )
        if not spend_series.empty:
            aligned_spend = spend_series.reindex(index)
            if aligned_spend.notna().any():
                export[
                    f"{primary_label} cumulative {_basis_descriptor(cost_basis)} spend ($)"
                ] = aligned_spend
        benefit_flow_col = "BenefitFlowTotal" if "BenefitFlowTotal" in opt_df.columns else "BenefitFlow"
        benefit_series = _cumulative_series_dollars(
            opt_df,
            flow_column=benefit_flow_col,
            basis=benefit_basis,
            horizon_years=pv_horizon_years if benefit_basis == VALUE_BASIS_PV else None,
        )
        if not benefit_series.empty:
            aligned = benefit_series.reindex(index)
            if aligned.notna().any():
                export[
                    f"{primary_label} cumulative {_basis_descriptor(benefit_basis)} benefit ($)"
                ] = aligned
    if cmp_df is not None and not cmp_df.empty:
        spend_series_cmp = _cumulative_series_dollars(
            cmp_df,
            flow_column="Spend",
            basis=cost_basis,
            horizon_years=pv_horizon_years if cost_basis == VALUE_BASIS_PV else None,
        )
        if not spend_series_cmp.empty:
            aligned_cmp_spend = spend_series_cmp.reindex(index)
            if aligned_cmp_spend.notna().any():
                export[
                    f"{comparison_label} cumulative {_basis_descriptor(cost_basis)} spend ($)"
                ] = aligned_cmp_spend
        benefit_flow_col = "BenefitFlowTotal" if "BenefitFlowTotal" in cmp_df.columns else "BenefitFlow"
        benefit_series_cmp = _cumulative_series_dollars(
            cmp_df,
            flow_column=benefit_flow_col,
            basis=benefit_basis,
            horizon_years=pv_horizon_years if benefit_basis == VALUE_BASIS_PV else None,
        )
        if not benefit_series_cmp.empty:
            aligned = benefit_series_cmp.reindex(index)
            if aligned.notna().any():
                export[
                    f"{comparison_label} cumulative {_basis_descriptor(benefit_basis)} benefit ($)"
                ] = aligned
    if export.shape[1] == 0:
        return None
    return export.reset_index()


def prepare_bcr_time_series_export(
    data: DashboardData,
    opt_df: Optional[pd.DataFrame],
    cmp_df: Optional[pd.DataFrame],
    opt_selection: ScenarioSelection,
    cmp_selection: ScenarioSelection,
) -> Optional[pd.DataFrame]:
    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()
    years = sorted_years(opt_df, cmp_df)
    if not years:
        return None
    try:
        pv_horizon_years = int(st.session_state.get("npv_horizon_selection", 60))
    except (TypeError, ValueError):
        pv_horizon_years = 60
    start_year = _bcr_start_year(opt_df, cmp_df)

    export, index = _create_export_table(years)
    opt_series = _bcr_series_from_frame(
        opt_df,
        start_year=start_year,
        pv_horizon_years=pv_horizon_years,
    )
    if opt_series is not None and not opt_series.empty:
        aligned = opt_series.reindex(index)
        if aligned.notna().any():
            export[f"{primary_label} BCR (PV)"] = aligned
    cmp_series = _bcr_series_from_frame(
        cmp_df,
        start_year=start_year,
        pv_horizon_years=pv_horizon_years,
    )
    if cmp_series is not None and not cmp_series.empty:
        aligned = cmp_series.reindex(index)
        if aligned.notna().any():
            export[f"{comparison_label} BCR (PV)"] = aligned

    if export.shape[1] == 0:
        return None
    return export.reset_index()

def prepare_cash_export(
    df: Optional[pd.DataFrame],
    *,
    label_prefix: str,
    spend_breakdown: str = "Total",
    data: Optional[DashboardData] = None,
    selection: Optional[ScenarioSelection] = None,
) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    years = sorted_years(df)
    if not years:
        return None
    export, index = _create_export_table(years)
    prefix = (label_prefix or "").strip()

    def _label(text: str) -> str:
        if prefix:
            return f"{prefix} {text}"
        return text

    def _add_series(series: pd.Series, label: str) -> None:
        if series.empty:
            return
        aligned = series.reindex(index)
        if aligned.notna().any():
            export[label] = scale_series_to_nzd(aligned)

    breakdown_key = re.sub(r"\s+", " ", str(spend_breakdown or "Total").strip().lower())
    group_col: Optional[str] = None
    if breakdown_key == "activity class":
        group_col = "ActivityClass"
    elif breakdown_key == "region":
        group_col = "Region"
    elif breakdown_key in {"gps request tier", "gps tier request", "gps tier"}:
        group_col = "GPSTier"

    breakdown_table: Optional[pd.DataFrame] = None
    if group_col and data is not None and selection is not None and getattr(selection, "code", None):
        breakdown_table = _annual_spend_breakdown_by_attribute(
            data,
            selection,
            group_col=group_col,
            years=years,
        )

    if breakdown_table is None or breakdown_table.empty:
        spend_series = series_from_df(df, "Spend")
        _add_series(spend_series, _label("Annual spend ($)"))
    else:
        for label in breakdown_table.columns:
            series = pd.to_numeric(breakdown_table[label], errors="coerce").fillna(0.0)
            series = pd.Series(series.to_numpy(dtype=float), index=breakdown_table.index)
            _add_series(series, _label(f"Spend: {label} ($)"))

    benefit_flow_col = "BenefitFlowTotal" if "BenefitFlowTotal" in df.columns else "BenefitFlow"
    benefit_series = series_from_df(df, benefit_flow_col)
    _add_series(benefit_series, _label("Benefit flow ($)"))

    closing_series = series_from_df(df, "ClosingNet")
    _add_series(closing_series, _label("Closing net ($)"))

    envelope_series = series_from_df(df, "Envelope")
    _add_series(envelope_series, _label("Envelope ($)"))

    if export.shape[1] == 0:
        return None
    return export.reset_index()

def prepare_cumulative_cash_export(
    df: Optional[pd.DataFrame],
    selection: ScenarioSelection,
    *,
    label_prefix: str,
) -> Optional[pd.DataFrame]:
    if df is None or df.empty or not selection:
        return None
    years = sorted_years(df)
    if not years:
        return None
    export, index = _create_export_table(years)

    cost_series = series_from_df(df, "CumSpend")
    if not cost_series.empty:
        aligned_cost = cost_series.reindex(index)
        if aligned_cost.notna().any():
            export[f"{label_prefix} cumulative cost ($)"] = scale_series_to_nzd(aligned_cost)

    benefit_series, benefit_label = _benefit_series_and_label(df, selection, prefix="")
    if isinstance(benefit_series, pd.Series) and not benefit_series.empty:
        aligned_benefit = pd.to_numeric(benefit_series, errors="coerce").reindex(index)
        if aligned_benefit.notna().any():
            is_real = "PV" in (benefit_label or "")
            benefit_label_text = (
                f"{label_prefix} cumulative benefits (real $2025)"
                if is_real
                else f"{label_prefix} cumulative benefits ($)"
            )
            export[benefit_label_text] = scale_series_to_nzd(aligned_benefit)

    if export.shape[1] <= 1:
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
    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()
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
                export[f"{primary_label} - {dim} ($)"] = scale_series_to_nzd(opt[dim])
                added = True
    if pivot_cmp is not None and not pivot_cmp.empty:
        cmp = pivot_cmp.reindex(index=years_list)
        cmp.index = pd.Index(years_list, name="Year")
        if cumulative:
            cmp = cmp.cumsum()
        for dim in selected_dims:
            if dim in cmp.columns:
                export[f"{comparison_label} - {dim} ($)"] = scale_series_to_nzd(cmp[dim])
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
    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()
    try:
        import streamlit as st
    except Exception:
        apply_discount = False
    else:
        apply_discount = bool(st.session_state.get("npv_apply_discount", False))
    if apply_discount:
        primary_col = f"{primary_label} NPV ($)"
        comparison_col = f"{comparison_label} NPV ($)"
        delta_col = "Delta ($)"
    else:
        primary_col = f"{primary_label} Benefits ($2025)"
        comparison_col = f"{comparison_label} Benefits ($2025)"
        delta_col = "Delta ($2025)"
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
                primary_col: opt_val * 1_000_000.0,
                comparison_col: cmp_val * 1_000_000.0,
                delta_col: (opt_val - cmp_val) * 1_000_000.0,
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
            primary_col: total_opt * 1_000_000.0,
            comparison_col: total_cmp * 1_000_000.0,
            delta_col: (total_opt - total_cmp) * 1_000_000.0,
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
    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()
    try:
        import streamlit as st
    except Exception:
        apply_discount = False
    else:
        apply_discount = bool(st.session_state.get("npv_apply_discount", False))
    metric_token = "NPV" if apply_discount else "Benefits ($2025)"
    value_col = "Value ($)" if apply_discount else "Value ($2025)"
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
        {"Step": f"{primary_label} {metric_token} total", value_col: total_opt * 1_000_000.0, "Measure": "relative"}
    ]
    for dim, delta in zip(dim_sequence, bridge_diffs):
        rows.append(
            {
                "Step": f"{dim} delta {metric_token}",
                value_col: (-delta) * 1_000_000.0,
                "Measure": "relative",
            }
        )
    rows.append(
        {"Step": f"{comparison_label} {metric_token} total", value_col: total_cmp * 1_000_000.0, "Measure": "total"}
    )
    return pd.DataFrame(rows)

def prepare_radar_export(
    data: DashboardData,
    opt_selection: ScenarioSelection,
    cmp_selection: ScenarioSelection,
    *,
    pv_opt: Optional[Dict[str, float]] = None,
    pv_cmp: Optional[Dict[str, float]] = None,
    horizon_years: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()
    try:
        import streamlit as st
    except Exception:
        apply_discount = False
    else:
        apply_discount = bool(st.session_state.get("npv_apply_discount", False))
    if apply_discount:
        primary_col = f"{primary_label} NPV ($)"
        comparison_col = f"{comparison_label} NPV ($)"
    else:
        primary_col = f"{primary_label} Benefits ($2025)"
        comparison_col = f"{comparison_label} Benefits ($2025)"
    if pv_opt is None and opt_selection and opt_selection.code:
        kwargs = {}
        if horizon_years is not None:
            kwargs["horizon_years"] = horizon_years
        pv_opt = pv_by_dimension(data, opt_selection, **kwargs)
    if pv_cmp is None and cmp_selection and cmp_selection.code:
        kwargs = {}
        if horizon_years is not None:
            kwargs["horizon_years"] = horizon_years
        pv_cmp = pv_by_dimension(data, cmp_selection, **kwargs)
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
                primary_col: float(pv_opt.get(dim, 0.0) if pv_opt else 0.0) * 1_000_000.0,
                comparison_col: float(pv_cmp.get(dim, 0.0) if pv_cmp else 0.0) * 1_000_000.0,
            }
        )
    return pd.DataFrame(rows)

def _gantt_export_years(data: DashboardData) -> List[int]:
    start_year = getattr(data, "start_fy", None)
    if start_year is None:
        years = [int(y) for y in getattr(data, "years", []) if y is not None]
        start_year = min(years) if years else 2025
    else:
        start_year = int(start_year)
    data_years = [int(y) for y in getattr(data, "years", []) if y is not None]
    if data_years:
        last_year = max(data_years)
    else:
        horizon = getattr(data, "model_years", 0)
        last_year = start_year + int(horizon) - 1 if horizon else start_year
    end_year = max(last_year, 2100)
    return list(range(start_year, end_year + 1))


def _raw_result_for_code(data: DashboardData, code: Optional[str]) -> Optional[Dict[str, Any]]:
    if not code:
        return None
    meta = data.scenario_meta_by_code.get(code)
    if not meta:
        return None
    stem = meta.get("_stem") or meta.get("CacheStem") or meta.get("OrigStem")
    if not stem:
        return None
    return data.raw_results.get(stem)


def _project_activity_class_map() -> Dict[str, str]:
    mapping = load_project_attribute_mapping()
    if mapping is None or mapping.empty:
        return {}
    working = mapping.dropna(subset=["ProjectKey"]).copy()
    if "ActivityClass" not in working.columns:
        return {}
    working["ActivityClass"] = working["ActivityClass"].fillna("Unknown")
    return dict(zip(working["ProjectKey"], working["ActivityClass"]))


def _attach_activity_class_column(
    table: Optional[pd.DataFrame],
    activity_map: Dict[str, str],
) -> Optional[pd.DataFrame]:
    if table is None or table.empty:
        return table
    if "Project" not in table.columns or "Activity Class" in table.columns:
        return table
    if activity_map:
        activity_values = table["Project"].map(
            lambda name: activity_map.get(_normalise_project_key(name), "Unknown")
        )
    else:
        activity_values = pd.Series(["Unknown"] * len(table), index=table.index)
    table = table.copy()
    table.insert(1, "Activity Class", activity_values)
    return table


def _project_dimension_weight_table(
    data: DashboardData,
    selection: ScenarioSelection,
) -> pd.DataFrame:
    if data is None or selection is None or not getattr(selection, "code", None):
        return pd.DataFrame()
    raw = _raw_result_for_code(data, selection.code)
    if not raw:
        return pd.DataFrame()
    benefits = raw.get("benefits_by_project_dimension_by_year")
    if not isinstance(benefits, pd.DataFrame) or benefits.empty:
        return pd.DataFrame()

    working = benefits.copy()
    if isinstance(working.index, pd.MultiIndex):
        working = working.reset_index()
    if "Project" not in working.columns or "Dimension" not in working.columns:
        return pd.DataFrame()

    year_cols = [
        col
        for col in working.columns
        if isinstance(col, (int, np.integer)) or (isinstance(col, str) and str(col).isdigit())
    ]
    if not year_cols:
        return pd.DataFrame()
    col_map = {col: int(str(col)) for col in year_cols}
    working = working.rename(columns=col_map)
    year_cols = [col_map[col] for col in year_cols]

    working["Project"] = working["Project"].astype(str)
    working["Dimension"] = (
        working["Dimension"].apply(_canonicalise_dimension_label).astype(str).str.strip()
    )
    working = working[working["Dimension"].ne("")]
    working = working[working["Dimension"].str.strip().str.lower() != "total"]
    if working.empty:
        return pd.DataFrame()

    working[year_cols] = working[year_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    dim_totals = working.groupby(["Project", "Dimension"], as_index=False)[year_cols].sum()
    dim_totals["Total"] = dim_totals[year_cols].sum(axis=1)
    dim_totals = dim_totals[dim_totals["Total"] > 0]
    if dim_totals.empty:
        return pd.DataFrame()

    project_totals = dim_totals.groupby("Project")["Total"].sum().rename("ProjectTotal")
    dim_totals = dim_totals.merge(project_totals, on="Project", how="left")
    dim_totals = dim_totals[dim_totals["ProjectTotal"] > 0]
    if dim_totals.empty:
        return pd.DataFrame()

    dim_totals["Weight"] = dim_totals["Total"] / dim_totals["ProjectTotal"]
    pivot = dim_totals.pivot_table(
        index="Project", columns="Dimension", values="Weight", aggfunc="first"
    ).fillna(0.0)
    if pivot.empty:
        return pd.DataFrame()

    ordered_dims = [str(dim) for dim in getattr(data, "dims", []) if str(dim).strip().lower() != "total"]
    ordered_cols: List[str] = []
    for dim in ordered_dims:
        if dim in pivot.columns:
            ordered_cols.append(dim)
    for dim in pivot.columns:
        if dim not in ordered_cols:
            ordered_cols.append(dim)
    pivot = pivot[ordered_cols]

    columns = {dim: f"Strategic Dimension {dim_short(dim)}" for dim in pivot.columns}
    pivot = pivot.rename(columns=columns)
    pivot = pivot.reset_index()
    pivot["ProjectKey"] = pivot["Project"].map(_normalise_project_key)
    return pivot.drop(columns=["Project"])


def _project_new_bcr_from_tables(
    cost_table: Optional[pd.DataFrame],
    benefit_table: Optional[pd.DataFrame],
    *,
    base_year: int,
) -> pd.DataFrame:
    if (
        cost_table is None
        or benefit_table is None
        or cost_table.empty
        or benefit_table.empty
        or "Project" not in cost_table.columns
        or "Project" not in benefit_table.columns
    ):
        return pd.DataFrame(columns=["ProjectKey", "BCR (New)"])

    cost_years = sorted(
        [col for col in cost_table.columns if isinstance(col, (int, np.integer))]
    )
    benefit_years = sorted(
        [col for col in benefit_table.columns if isinstance(col, (int, np.integer))]
    )
    if not cost_years or not benefit_years:
        return pd.DataFrame(columns=["ProjectKey", "BCR (New)"])

    cost_df = cost_table[["Project"] + cost_years].copy()
    cost_df["Project"] = cost_df["Project"].astype(str)
    cost_df["ProjectKey"] = cost_df["Project"].map(_normalise_project_key)
    cost_values = cost_df[cost_years].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cost_offsets = np.asarray(cost_years, dtype=int) - int(base_year)
    cost_divisors = mbcm_discount_divisors(cost_offsets)
    cost_pv = cost_values.dot(1.0 / cost_divisors) if cost_divisors.size else np.zeros(len(cost_df))
    cost_map = pd.DataFrame({"ProjectKey": cost_df["ProjectKey"], "CostPV": cost_pv})
    cost_map = cost_map.groupby("ProjectKey", as_index=False)["CostPV"].sum()

    benefit_df = benefit_table[["Project"] + benefit_years].copy()
    benefit_df["Project"] = benefit_df["Project"].astype(str)
    benefit_df["ProjectKey"] = benefit_df["Project"].map(_normalise_project_key)
    benefit_values = (
        benefit_df[benefit_years].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    )
    benefit_offsets = np.asarray(benefit_years, dtype=int) - int(base_year)
    benefit_divisors = mbcm_discount_divisors(benefit_offsets)
    benefit_pv = benefit_values.dot(1.0 / benefit_divisors) if benefit_divisors.size else np.zeros(len(benefit_df))
    benefit_map = pd.DataFrame({"ProjectKey": benefit_df["ProjectKey"], "BenefitPV": benefit_pv})
    benefit_map = benefit_map.groupby("ProjectKey", as_index=False)["BenefitPV"].sum()

    merged = cost_map.merge(benefit_map, on="ProjectKey", how="left")
    merged["BenefitPV"] = merged["BenefitPV"].fillna(0.0)
    merged["BCR (New)"] = np.divide(
        merged["BenefitPV"].to_numpy(dtype=float),
        merged["CostPV"].to_numpy(dtype=float),
        out=np.full(len(merged), np.nan, dtype=float),
        where=merged["CostPV"].to_numpy(dtype=float) > 0,
    )
    return merged[["ProjectKey", "BCR (New)"]]


def _attach_project_attribute_columns(
    table: Optional[pd.DataFrame],
    mapping: pd.DataFrame,
    weight_table: pd.DataFrame,
    *,
    base_year: Optional[int] = None,
    new_bcr_table: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    if table is None or table.empty or "Project" not in table.columns:
        return table
    table = table.copy()
    table["ProjectKey"] = table["Project"].map(_normalise_project_key)

    attrs = pd.DataFrame({"ProjectKey": table["ProjectKey"]})
    if mapping is not None and not mapping.empty:
        attrs = mapping[["ProjectKey", "ActivityClass", "GPSTier", "Region"]].copy()
        attrs = attrs.drop_duplicates(subset=["ProjectKey"]).copy()
        attrs = attrs.rename(
            columns={
                "ActivityClass": "Activity Class",
                "GPSTier": "GPS Request Tier",
                "Region": "Region",
            }
        )
        attrs["Activity Class"] = attrs["Activity Class"].fillna("Unknown")
        attrs["GPS Request Tier"] = attrs["GPS Request Tier"].map(_canonicalise_gps_tier)
        attrs["Region"] = attrs["Region"].fillna("Unknown")
        attrs["GPS Request Tier"] = attrs["GPS Request Tier"].fillna("Unknown")

    merged = table.merge(attrs, on="ProjectKey", how="left")

    bcr_table = load_project_bcr_mapping()
    if bcr_table is not None and not bcr_table.empty:
        merged = merged.merge(
            bcr_table[["ProjectKey", "BCR (Original)"]],
            on="ProjectKey",
            how="left",
        )
    if new_bcr_table is None or new_bcr_table.empty:
        new_bcr_table = load_project_new_bcr_mapping(base_year=base_year)
    if new_bcr_table is not None and not new_bcr_table.empty:
        merged = merged.merge(
            new_bcr_table[["ProjectKey", "BCR (New)"]],
            on="ProjectKey",
            how="left",
        )
    if "BCR (Original)" not in merged.columns:
        merged["BCR (Original)"] = np.nan
    if "BCR (New)" not in merged.columns:
        merged["BCR (New)"] = np.nan

    dim_cols: List[str] = []
    if weight_table is not None and not weight_table.empty:
        merged = merged.merge(weight_table, on="ProjectKey", how="left")
        dim_cols = [col for col in weight_table.columns if col != "ProjectKey"]
        for col in dim_cols:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    for col in ("Activity Class", "GPS Request Tier", "Region"):
        if col in merged.columns:
            merged[col] = merged[col].fillna("Unknown")
        else:
            merged[col] = "Unknown"

    ordered_cols = [
        "Project",
        "Activity Class",
        "GPS Request Tier",
        "Region",
        "BCR (Original)",
        "BCR (New)",
    ] + dim_cols
    remaining = [col for col in merged.columns if col not in ordered_cols and col != "ProjectKey"]
    merged = merged[ordered_cols + remaining]
    return merged


def _scenario_project_cost_table(
    data: DashboardData,
    code: str,
    years: List[int],
) -> Optional[pd.DataFrame]:
    spend_matrix = getattr(data, "spend_matrix", None)
    if spend_matrix is None or spend_matrix.empty or "Code" not in spend_matrix.columns:
        return None
    subset = spend_matrix[spend_matrix["Code"] == code]
    if subset.empty:
        return None
    working = subset.drop(columns=[c for c in ("Key", "Code") if c in subset.columns]).copy()
    if "Project" not in working.columns:
        return None
    working["Project"] = working["Project"].astype(str)
    numeric_cols = [col for col in working.columns if isinstance(col, (int, np.integer))]
    if not numeric_cols:
        return None
    table = (
        working[["Project"] + numeric_cols]
        .set_index("Project")
        .reindex(columns=years, fill_value=0.0)
        .sort_index()
    )
    table.loc[:, years] = table.loc[:, years].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    table = table.reset_index()
    table.columns = ["Project"] + years
    return table


def _scenario_project_benefit_table(
    data: DashboardData,
    code: str,
    years: List[int],
) -> Optional[pd.DataFrame]:
    raw = _raw_result_for_code(data, code)
    if not raw:
        return None
    benefits = raw.get("benefits_by_project_dimension_by_year")
    if not isinstance(benefits, pd.DataFrame) or benefits.empty:
        return None
    working = benefits.copy()
    year_cols = [
        int(str(col))
        for col in working.columns
        if isinstance(col, (int, np.integer)) or (isinstance(col, str) and col.isdigit())
    ]
    if not year_cols:
        return None
    col_map = {col: int(str(col)) for col in working.columns if str(col).isdigit()}
    if col_map:
        working = working.rename(columns=col_map)
    selected_cols = [col for col in working.columns if isinstance(col, int)]
    if not selected_cols:
        return None
    if isinstance(working.index, pd.MultiIndex):
        level_names = list(working.index.names)
        project_level = level_names.index("Project") if "Project" in level_names else 0
        dimension_level = level_names.index("Dimension") if "Dimension" in level_names else None

        if dimension_level is not None:
            try:
                totals_only = working.xs("Total", level=dimension_level, drop_level=True)
            except KeyError:
                totals_only = None
            if totals_only is not None and not totals_only.empty:
                aggregated = totals_only[selected_cols]
            else:
                aggregated = working.groupby(level=project_level)[selected_cols].sum()
        else:
            aggregated = working.groupby(level=project_level)[selected_cols].sum()
    else:
        project_col = None
        dimension_col = None
        for candidate in working.columns:
            normalized = str(candidate).strip().lower()
            if normalized == "project":
                project_col = candidate
            elif normalized == "dimension":
                dimension_col = candidate
        if project_col is None:
            return None
        if dimension_col is not None:
            totals_only = working[
                working[dimension_col].astype(str).str.strip().str.lower() == "total"
            ]
            if not totals_only.empty:
                working = totals_only
        aggregated = working.groupby(project_col)[selected_cols].sum()
    aggregated = aggregated.astype(float)
    aggregated = aggregated.reindex(columns=years, fill_value=0.0)
    aggregated = aggregated.sort_index()
    aggregated = aggregated.reset_index()
    aggregated.columns = ["Project"] + years
    return aggregated


def prepare_gantt_export(
    data: DashboardData,
    opt_selection: ScenarioSelection,
    comp_selection: ScenarioSelection,
    *,
    opt_label: str,
    cmp_label: str,
) -> Dict[str, pd.DataFrame]:
    years = _gantt_export_years(data)
    tables: Dict[str, pd.DataFrame] = {}
    mapping = load_project_attribute_mapping()
    for selection, label in (
        (opt_selection, opt_label),
        (comp_selection, cmp_label),
    ):
        code = getattr(selection, "code", None) if selection else None
        if not code:
            continue
        sheet_label = label or code
        runs = extract_project_runs(data, code)
        order_lookup = {str(run.project): idx for idx, run in enumerate(runs)} if runs else {}
        weight_table = _project_dimension_weight_table(data, selection)
        cost_table = _scenario_project_cost_table(data, code, years)
        benefit_table = _scenario_project_benefit_table(data, code, years)
        base_year = int(getattr(data, "start_fy", 2025) or 2025)
        new_bcr_table = _project_new_bcr_from_tables(
            cost_table,
            benefit_table,
            base_year=base_year,
        )
        cost_table = _attach_project_attribute_columns(
            cost_table,
            mapping,
            weight_table,
            base_year=base_year,
            new_bcr_table=new_bcr_table,
        )
        if cost_table is not None and not cost_table.empty and order_lookup:
            cost_table = cost_table.copy()
            cost_table["_gantt_order_idx"] = (
                cost_table["Project"].astype(str).map(order_lookup).fillna(1_000_000).astype(int)
            )
        if cost_table is not None and not cost_table.empty:
            tables[f"{sheet_label} - Costs"] = cost_table
        benefit_table = _attach_project_attribute_columns(
            benefit_table,
            mapping,
            weight_table,
            base_year=base_year,
            new_bcr_table=new_bcr_table,
        )
        if benefit_table is not None and not benefit_table.empty and order_lookup:
            benefit_table = benefit_table.copy()
            benefit_table["_gantt_order_idx"] = (
                benefit_table["Project"].astype(str).map(order_lookup).fillna(1_000_000).astype(int)
            )
        if benefit_table is not None and not benefit_table.empty:
            tables[f"{sheet_label} - Benefits"] = benefit_table
    return tables


def _split_gantt_export_label(sheet_name: str) -> Tuple[str, str]:
    if sheet_name.endswith(" - Costs"):
        return sheet_name[: -len(" - Costs")], "Costs"
    if sheet_name.endswith(" - Benefits"):
        return sheet_name[: -len(" - Benefits")], "Benefits"
    return sheet_name, ""


def _project_sort_key(name: str) -> Tuple[Tuple[int, Any], ...]:
    text = str(name).strip().lower()
    parts = re.split(r"(\d+)", text)
    key: List[Tuple[int, Any]] = []
    for part in parts:
        if part.isdigit():
            key.append((1, int(part)))
        else:
            key.append((0, part))
    return tuple(key)


def _project_start_year_order(table: pd.DataFrame) -> List[str]:
    if table is None or table.empty or "Project" not in table.columns:
        return []
    year_cols = sorted([col for col in table.columns if isinstance(col, (int, np.integer))])
    if not year_cols:
        return sorted(table["Project"].astype(str).tolist(), key=_project_sort_key)
    values = (
        table[year_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float, copy=True)
    )
    threshold = 1e-6
    order: List[Tuple[int, str, str]] = []
    for idx, row in enumerate(values):
        mask = np.abs(row) > threshold
        if mask.any():
            first_idx = int(np.argmax(mask))
            start_year = int(year_cols[first_idx])
        else:
            start_year = 999999
        project_name = str(table.iloc[idx]["Project"])
        order.append((start_year, _project_sort_key(project_name), project_name))
    order.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in order]


def _project_alpha_order(table: pd.DataFrame) -> List[str]:
    if table is None or table.empty or "Project" not in table.columns:
        return []
    return sorted(table["Project"].astype(str).tolist(), key=_project_sort_key)


def _order_export_table(table: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    if table is None or table.empty or "Project" not in table.columns or not order:
        return table
    table = table.copy()
    existing = table["Project"].astype(str).tolist()
    missing = [name for name in existing if name not in order]
    if missing:
        order = list(order) + sorted(missing, key=_project_sort_key)
    table["_order"] = pd.Categorical(table["Project"], categories=order, ordered=True)
    table = table.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return table


def _apply_programme_schedule_export_order(
    tables: Dict[str, pd.DataFrame],
    order_by: str,
) -> Dict[str, pd.DataFrame]:
    if not tables:
        return tables
    order_by = (order_by or "project").strip().lower()
    scenario_orders: Dict[str, List[str]] = {}
    if order_by == "start_year":
        for name, table in tables.items():
            if not isinstance(table, pd.DataFrame) or "Project" not in table.columns:
                continue
            scenario_label, kind = _split_gantt_export_label(name)
            if scenario_label in scenario_orders:
                continue
            if "_gantt_order_idx" in table.columns:
                ordered = table.sort_values("_gantt_order_idx", kind="stable")
                scenario_orders[scenario_label] = ordered["Project"].astype(str).tolist()
                continue
            if kind == "Costs" and scenario_label not in scenario_orders:
                scenario_orders[scenario_label] = _project_start_year_order(table)
        for name, table in tables.items():
            if not isinstance(table, pd.DataFrame) or "Project" not in table.columns:
                continue
            scenario_label, _ = _split_gantt_export_label(name)
            if scenario_label not in scenario_orders:
                if "_gantt_order_idx" in table.columns:
                    ordered = table.sort_values("_gantt_order_idx", kind="stable")
                    scenario_orders[scenario_label] = ordered["Project"].astype(str).tolist()
                else:
                    scenario_orders[scenario_label] = _project_start_year_order(table)
    else:
        for name, table in tables.items():
            if not isinstance(table, pd.DataFrame) or "Project" not in table.columns:
                continue
            scenario_label, _ = _split_gantt_export_label(name)
            if scenario_label not in scenario_orders:
                scenario_orders[scenario_label] = _project_alpha_order(table)

    ordered_tables: Dict[str, pd.DataFrame] = {}
    for name, table in tables.items():
        if not isinstance(table, pd.DataFrame) or "Project" not in table.columns:
            ordered_tables[name] = table
            continue
        scenario_label, _ = _split_gantt_export_label(name)
        order = scenario_orders.get(scenario_label, [])
        ordered = _order_export_table(table, order)
        ordered_tables[name] = ordered.drop(columns=["_gantt_order_idx"], errors="ignore")
    return ordered_tables

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


def format_npv_context(*horizon_values: Optional[int], apply_discount: bool = True) -> str:
    """Describe the benefit scope including horizon."""
    years = sorted({int(value) for value in horizon_values if value is not None})
    label = "NPV" if apply_discount else "Benefits ($2025)"
    if not years:
        return label
    if len(years) == 1:
        return f"{years[0]}-year {label}"
    horizon_text = "/".join(str(year) for year in years)
    return f"{horizon_text}-year {label}"


def npv_context_label(
    data: DashboardData,
    *selections: ScenarioSelection,
    horizon_override: Optional[int] = None,
    apply_discount: Optional[bool] = None,
) -> str:

    horizons: List[int] = []

    for selection in selections:

        if not selection:

            continue

        meta = selection.metadata or {}

        horizon = meta.get('HorizonYears')

        if horizon is not None:

            try:

                horizons.append(int(horizon))

            except (TypeError, ValueError):

                pass

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

    if apply_discount is None:
        try:
            import streamlit as st
        except Exception:
            apply_discount = False
        else:
            apply_discount = bool(st.session_state.get("npv_apply_discount", False))

    return format_npv_context(*horizons, apply_discount=bool(apply_discount))




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


def _preferred_option_index(options: List[str], preferred: Optional[str]) -> int:
    if not options:
        return 0
    target = str(preferred or "").strip().lower()
    if target:
        for idx, option in enumerate(options):
            if str(option).strip().lower() == target:
                return idx
    return 0


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

def _full_envelope_series(df: pd.DataFrame, mode: str, *, prefer_full: bool) -> pd.Series:

    if df is None or df.empty:

        return pd.Series(dtype=float)

    has_full = prefer_full and "EnvelopeFull" in df.columns

    if has_full:

        env = pd.to_numeric(df["EnvelopeFull"], errors="coerce")

    elif "Envelope" in df.columns:

        env = pd.to_numeric(df["Envelope"], errors="coerce")

    else:

        env = pd.Series(dtype=float)

    if prefer_full and not has_full and mode == "buffered":

        buff = pd.to_numeric(df["Buffer"], errors="coerce") if "Buffer" in df.columns else 0

        env = env.add(buff, fill_value=0)

    elif prefer_full and not has_full and mode == "cash":

        uplift = (

            pd.to_numeric(df["CashPlus"], errors="coerce") if "CashPlus" in df.columns else 0

        )

        env = env.add(uplift, fill_value=0)

    return env

def _base_envelope_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "Envelope" not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df["Envelope"], errors="coerce")

def available_envelopes(
    data: DashboardData,
    *,
    confidence: str,
    steep: str,
    horizon: int,
    benefit_level: str,
    mode: str,
    prefer_comparison: Optional[bool] = None,
    scenarios_df: Optional[pd.DataFrame] = None,
    slug_prefix: Optional[str] = None,
) -> List[int]:
    df = scenarios_df if scenarios_df is not None else data.scenarios
    if df.empty:
        return []
    level = str(benefit_level or "base").strip().lower()
    mask = (
        (df["Conf"] == confidence)
        & (df["BenSteep"] == steep)
        & (pd.to_numeric(df["BenHorizon"], errors="coerce") == int(horizon))
        & (df["Mode"] == mode)
    )
    if "BenLevel" in df.columns:
        mask = mask & (df["BenLevel"].astype(str).str.lower() == level)
    if prefer_comparison is not None and "IsComp" in df.columns:
        mask = mask & (df["IsComp"] == (1 if prefer_comparison else 0))
    if slug_prefix and "CacheStem" in df.columns:
        stems_lower = df["CacheStem"].astype(str).str.lower()
        mask = mask & stems_lower.str.startswith(slug_prefix.lower())
    subset = df.loc[mask]
    if subset.empty or mode not in {"fixed", "buffered", "cash"}:
        return []
    values = _base_envelope_series(subset)
    if values.isna().all():
        values = _full_envelope_series(subset, mode, prefer_full=data.prefer_envelope_full)
    values = values.dropna()
    return sorted({int(v) for v in values.tolist()})




def available_buffer_levels(
    data: DashboardData,
    *,
    confidence: str,
    steep: str,
    horizon: int,
    benefit_level: str,
    mode: str,
    envelope: Optional[int] = None,
    prefer_comparison: Optional[bool] = None,
    scenarios_df: Optional[pd.DataFrame] = None,
    slug_prefix: Optional[str] = None,
) -> List[int]:
    df = scenarios_df if scenarios_df is not None else data.scenarios
    if df.empty:
        return []
    level = str(benefit_level or "base").strip().lower()
    env_base = _base_envelope_series(df)
    if env_base.isna().all():
        env_base = _full_envelope_series(df, mode, prefer_full=data.prefer_envelope_full)
    mask = (
        (df["Conf"] == confidence)
        & (df["BenSteep"] == steep)
        & (pd.to_numeric(df["BenHorizon"], errors="coerce") == int(horizon))
        & (df["Mode"] == mode)
    )
    if "BenLevel" in df.columns:
        mask = mask & (df["BenLevel"].astype(str).str.lower() == level)
    if prefer_comparison is not None and "IsComp" in df.columns:
        mask = mask & (df["IsComp"] == (1 if prefer_comparison else 0))
    if envelope is not None:
        env_match = env_base == float(envelope)
        mask = mask & env_match.fillna(False)
    if slug_prefix and "CacheStem" in df.columns:
        stems_lower = df["CacheStem"].astype(str).str.lower()
        mask = mask & stems_lower.str.startswith(slug_prefix.lower())
    subset = df.loc[mask]
    if subset.empty or mode not in {"buffered", "cash"}:
        return []
    values = pd.to_numeric(
        subset["Buffer"] if mode == "buffered" else subset["CashPlus"],
        errors="coerce",
    ).dropna()
    return sorted({int(v) for v in values.tolist()})



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

    show_benefit_controls: bool = True,

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

    benefit_levels = list(getattr(options, "benefit_levels", []) or [])
    if not benefit_levels:
        benefit_levels = ["base"]
    if "base" not in [str(level).strip().lower() for level in benefit_levels]:
        benefit_levels.insert(0, "base")
    benefit_level_index = _preferred_option_index([str(level) for level in benefit_levels], "base")
    benefit_level = st.selectbox(
        f"{name} benefit level",
        benefit_levels,
        index=benefit_level_index,
        key=f"{key_prefix}_benefit_level",
    )

    if show_benefit_controls and options.benefit_steep:

        steep_index = options.benefit_steep.index(options.benefit_steep[0]) if options.benefit_steep else 0

        benefit_steep = st.selectbox(

            f"{name} benefit steepness (aggressive <-> conservative)",

            options.benefit_steep,

            index=steep_index,

            key=f"{key_prefix}_steep",

        )

    else:

        benefit_steep = options.benefit_steep[0] if options.benefit_steep else ""

    if show_benefit_controls and options.benefit_horizon:

        horizon_index = 0

        benefit_horizon = st.selectbox(

            f"{name} benefit horizon (years)",

            options.benefit_horizon,

            index=horizon_index,

            key=f"{key_prefix}_horizon",

        )

    else:

        benefit_horizon = int(options.benefit_horizon[0]) if options.benefit_horizon else 0

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

    default_mode_index = available_modes.index(default_mode_key) if default_mode_key in available_modes else 0
    mode_key = available_modes[default_mode_index]

    sources: List[pd.DataFrame] = [profile_df]

    if profile_filtered:

        sources.append(base_df)

    profile_prefix = (profile_name or "").strip().lower()

    slug_filter: Optional[str] = None

    if profile_prefix and profile_prefix != DEFAULT_PROFILE_LABEL.lower():

        slug_filter = profile_prefix

    def _collect_values(fetcher, allow_none_pref: bool = True):

        prefs: List[Optional[bool]] = []

        if prefer_comparison is not None:

            prefs.append(prefer_comparison)

            prefs.append(not prefer_comparison)

        if allow_none_pref or not prefs:

            if None not in prefs:

                prefs.append(None)

        for src in sources:

            if src.empty:

                continue

            for pref in prefs:

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

            benefit_level=benefit_level,

            mode=mode_key,

            prefer_comparison=pref,

            scenarios_df=src,

            slug_prefix=slug_filter,

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

                benefit_level=benefit_level,

                mode=mode_key,

                envelope=envelope,

                prefer_comparison=pref,

                scenarios_df=src,

                slug_prefix=slug_filter,

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

                benefit_level=benefit_level,

                mode=mode_key,

                envelope=envelope,

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

    dimension_options: List[str] = []
    dim_source = profile_df if profile_filtered else base_df
    if dim_source.empty:
        dim_source = data.scenarios
    if not dim_source.empty:
        full_env_dim = _base_envelope_series(dim_source)
        if full_env_dim.isna().all():
            full_env_dim = _full_envelope_series(
                dim_source, mode_key, prefer_full=data.prefer_envelope_full
            )
        base_dim_mask = (
            (dim_source["Conf"] == confidence)
            & (dim_source["BenSteep"] == benefit_steep)
            & (pd.to_numeric(dim_source["BenHorizon"], errors="coerce") == int(benefit_horizon))
            & (dim_source["Mode"] == mode_key)
        )
        if "BenLevel" in dim_source.columns:
            level = str(benefit_level or "base").strip().lower()
            base_dim_mask = base_dim_mask & (dim_source["BenLevel"].astype(str).str.lower() == level)
        if mode_key in {"fixed", "buffered", "cash"} and envelope is not None:
            env_match = full_env_dim == float(envelope)
            base_dim_mask = base_dim_mask & env_match.fillna(False)
        if mode_key == "buffered" and buffer_value is not None:
            base_dim_mask = base_dim_mask & (
                pd.to_numeric(dim_source["Buffer"], errors="coerce") == float(buffer_value)
            )
        if mode_key == "cash" and buffer_value is not None:
            base_dim_mask = base_dim_mask & (
                pd.to_numeric(dim_source["CashPlus"], errors="coerce") == float(buffer_value)
            )
        if slug_filter and "CacheStem" in dim_source.columns:
            stems = dim_source["CacheStem"].astype(str).str.lower()
            base_dim_mask = base_dim_mask & stems.str.startswith(slug_filter)

        def _dimension_list(mask) -> List[str]:
            return (
                dim_source.loc[mask, "ObjectiveDim"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

        subset_dims: List[str] = []
        if prefer_comparison is not None and "IsComp" in dim_source.columns:
            pref_mask = base_dim_mask & (dim_source["IsComp"] == (1 if prefer_comparison else 0))
            subset_dims = _dimension_list(pref_mask)
            if not subset_dims:
                alt_mask = base_dim_mask & (dim_source["IsComp"] == (0 if prefer_comparison else 1))
                subset_dims = _dimension_list(alt_mask)
        if not subset_dims:
            subset_dims = _dimension_list(base_dim_mask)
        dimension_options = sorted({dim for dim in subset_dims if dim})
    if not dimension_options:
        dimension_options = [str(dim).strip() for dim in (data.dims or ["Total"]) if str(dim).strip()]
        if not dimension_options:
            dimension_options = ["Total"]
    if dimension_options:
        seen_dim: Set[str] = set()
        deduped: List[str] = []
        for dim in dimension_options:
            label = str(dim).strip()
            key = label.lower()
            if not label or key in seen_dim:
                continue
            seen_dim.add(key)
            deduped.append(label)
        dimension_options = deduped
    try:
        dimension_options = ["Total"] + [dim for dim in dimension_options if dim.lower() != "total"]
    except Exception:
        pass

    default_dim_index = dimension_options.index("Total") if "Total" in dimension_options else 0

    dimension = st.selectbox(

        f"{name} benefit dimension",

        dimension_options,

        index=default_dim_index,

        key=f"{key_prefix}_dim",

    )

    dimension_label = str(dimension or "").strip()
    dimension_lower = dimension_label.lower()
    dimension_short_lower = dim_short(dimension_label).lower() if dimension_label else ""

    def _filter_codes_to_dimension(codes: List[str]) -> List[str]:
        if not codes or not dimension_label:
            return codes or []
        filtered: List[str] = []
        for code_value in codes:
            meta = scenario_metadata(data, code_value) if code_value else None
            if not meta:
                continue
            meta_dim = str(meta.get("ObjectiveDim", "")).strip()
            meta_dim_short = str(meta.get("ObjectiveDimShort", "")).strip()
            if not meta_dim_short and meta_dim:
                meta_dim_short = dim_short(meta_dim)
            if (
                meta_dim.lower() == dimension_lower
                or (meta_dim_short and meta_dim_short.lower() == dimension_short_lower)
            ):
                filtered.append(code_value)
        return filtered

    def _call_find_scenario_code(
        *,
        profile_filter: Optional[str],
        src: Optional[pd.DataFrame],
    ) -> Optional[str]:
        params = {
            "data": data,
            "conf": confidence,
            "benefit_steep": benefit_steep,
            "benefit_horizon": int(benefit_horizon),
            "mode": mode_key,
            "envelope": float(envelope) if envelope is not None else None,
            "buffer_value": float(buffer_value) if buffer_value is not None else None,
            "prefer_comparison": prefer_comparison,
            "profile": profile_filter,
        }
        if _FIND_SCENARIO_SUPPORTS_BENEFIT_LEVEL:
            params["benefit_level"] = benefit_level
        if src is not None:
            params["scenarios_df"] = src
        if _FIND_SCENARIO_SUPPORTS_DIM and dimension_label:
            params["objective_dim"] = dimension
        try:
            return find_scenario_code(**params)
        except TypeError:
            params.pop("objective_dim", None)
            return find_scenario_code(**params)

    def _match_mask(df: pd.DataFrame) -> pd.Series:

        mask = (

            (df["Conf"] == confidence)

            & (df["BenSteep"] == benefit_steep)

            & (pd.to_numeric(df["BenHorizon"], errors="coerce") == int(benefit_horizon))

            & (df["Mode"] == mode_key)

        )

        if "BenLevel" in df.columns:
            level = str(benefit_level or "base").strip().lower()
            mask = mask & (df["BenLevel"].astype(str).str.lower() == level)

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

        if dimension_label and "ObjectiveDim" in df.columns:
            dim_series = df["ObjectiveDim"].astype(str).str.strip()
            dim_mask = dim_series.str.lower() == dimension_lower
            if "ObjectiveDimShort" in df.columns:
                short_series = df["ObjectiveDimShort"].astype(str).str.strip().str.lower()
            else:
                short_series = dim_series.apply(dim_short).str.lower()
            if dimension_short_lower:
                dim_mask = dim_mask | (short_series == dimension_short_lower)
            mask = mask & dim_mask

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

    subset_codes = _filter_codes_to_dimension(subset_codes)
    codes_pool = _collect_values(

        lambda pref, src: scenario_code_options(

            data,

            prefer_comparison=pref,

            scenarios_df=src,

        )

    ) or []

    codes_pool = _filter_codes_to_dimension(codes_pool)
    recommended_code: Optional[str] = None

    for idx, src in enumerate(sources):

        profile_filter = profile_name if profile_filtered and idx == 0 else None

        candidate = _call_find_scenario_code(profile_filter=profile_filter, src=src)

        if candidate:

            recommended_code = candidate

            break

    if recommended_code is None:
        recommended_code = _call_find_scenario_code(profile_filter=profile_name, src=None)

    code_choices_raw = subset_codes or codes_pool or scenario_code_options(data)
    code_choices = _filter_codes_to_dimension(code_choices_raw)

    if recommended_code and recommended_code not in code_choices:
        recommended_code = None

    if recommended_code is None and code_choices:
        recommended_code = code_choices[0]

    if not code_choices:
        st.warning(
            f"No scenarios found for {dimension_label or 'the selected dimension'} with the chosen settings."
        )
        code = None
    else:
        code = recommended_code

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

        benefit_level=str(benefit_level or "base").strip().lower(),

        mode=mode_key,

        envelope=envelope,

        buffer_value=buffer_value,

        dimension=dimension,

        profile=selected_profile,

        metadata=meta,

    )

def build_timeseries(
    data: DashboardData,
    selection: ScenarioSelection,
    *,
    apply_discount: Optional[bool] = None,
) -> Optional[pd.DataFrame]:

    if not selection.code:

        return None

    cf = data.cf[data.cf["Code"] == selection.code].copy()

    if cf.empty:

        return None

    cf = cf.sort_values("Year")

    base = pd.DataFrame({"Year": data.years})

    df = base.merge(cf, on="Year", how="left").sort_values("Year").fillna(0.0)

    dim_label = (selection.dimension or "").strip()
    dim_label_lower = dim_label.lower()

    for col in ("Spend", "ClosingNet", "Envelope"):

        if col in df.columns:

            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        else:

            df[col] = 0.0

    benefit_dim_selected = data.benefit_dim[
        (data.benefit_dim["Code"] == selection.code)
        & (data.benefit_dim["Dimension"].astype(str).str.lower() == dim_label_lower)
    ][["Year", "BenefitFlow"]]

    benefit_total = data.benefit_dim[
        (data.benefit_dim["Code"] == selection.code)
        & (data.benefit_dim["Dimension"].astype(str).str.lower() == "total")
    ][["Year", "BenefitFlow"]]

    if benefit_total.empty:
        benefit_total = data.benefit[data.benefit["Code"] == selection.code][["Year", "BenefitFlow"]]

    benefit_total = benefit_total.rename(columns={"BenefitFlow": "BenefitFlowTotal"})
    df = df.merge(benefit_total, on="Year", how="left")

    if not benefit_dim_selected.empty:
        benefit_dim_selected = benefit_dim_selected.rename(columns={"BenefitFlow": "BenefitFlowDimension"})
        df = df.merge(benefit_dim_selected, on="Year", how="left")
    else:
        df["BenefitFlowDimension"] = np.nan

    df["BenefitFlowTotal"] = pd.to_numeric(df["BenefitFlowTotal"], errors="coerce").fillna(0.0)
    df["BenefitFlowDimension"] = pd.to_numeric(df.get("BenefitFlowDimension"), errors="coerce")
    df["BenefitFlow"] = df["BenefitFlowTotal"]

    if apply_discount is None:
        try:
            import streamlit as st
        except Exception:
            apply_discount = False
        else:
            apply_discount = bool(st.session_state.get("npv_apply_discount", False))

    if apply_discount:
        year_offsets = (df["Year"].astype(int) - data.start_fy).clip(lower=0).to_numpy(dtype=int)
        discount = mbcm_discount_divisors(year_offsets)
        df["PVBenefitTotal"] = df["BenefitFlowTotal"] / discount
    else:
        df["PVBenefitTotal"] = df["BenefitFlowTotal"]
    df["PVBenefit"] = df["PVBenefitTotal"]
    if "BenefitFlowDimension" in df.columns:
        if apply_discount:
            df["PVBenefitDimension"] = df["BenefitFlowDimension"].fillna(0.0) / discount
        else:
            df["PVBenefitDimension"] = df["BenefitFlowDimension"].fillna(0.0)

    df["CumPVBenefitTotal"] = df["PVBenefitTotal"].cumsum()
    df["CumPVBenefit"] = df["CumPVBenefitTotal"]
    if "PVBenefitDimension" in df.columns:
        df["CumPVBenefitDimension"] = df["PVBenefitDimension"].fillna(0.0).cumsum()

    df["CumBenefitTotal"] = df["BenefitFlowTotal"].cumsum()
    df["CumBenefit"] = df["CumBenefitTotal"]
    if "BenefitFlowDimension" in df.columns:
        df["CumBenefitDimension"] = df["BenefitFlowDimension"].fillna(0.0).cumsum()

    df["CumSpend"] = df["Spend"].cumsum()

    return df

def pv_by_dimension(
    data: DashboardData,
    selection: ScenarioSelection,
    *,
    horizon_years: Optional[int] = None,
    apply_discount: Optional[bool] = None,
    value_basis: Optional[str] = None,
) -> Optional[Dict[str, float]]:

    if not selection.code:

        return None

    df = data.benefit_dim[data.benefit_dim["Code"] == selection.code]

    if df.empty:

        return None

    dims_present = df["Dimension"].dropna().astype(str)

    if dims_present.empty:

        return None

    base_years = pd.DataFrame({"Year": data.years})

    if value_basis is None:
        if apply_discount is None:
            try:
                import streamlit as st
            except Exception:
                apply_discount = False
            else:
                apply_discount = bool(st.session_state.get("npv_apply_discount", False))
        value_basis = VALUE_BASIS_PV if apply_discount else VALUE_BASIS_REAL

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

        if horizon_years is not None:

            try:

                limit_year = int(data.start_fy) + int(horizon_years) - 1

            except (TypeError, ValueError):

                limit_year = None

            if limit_year is not None:

                mask = merged["Year"].astype(int) <= limit_year

                merged = merged.loc[mask].copy()

        if merged.empty:

            pv[str(dim)] = 0.0

            continue

        values = pd.to_numeric(merged["BenefitFlow"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        year_values = pd.to_numeric(merged["Year"], errors="coerce").fillna(data.start_fy).astype(int).to_numpy(dtype=int)
        multipliers = value_basis_multiplier(int(data.start_fy), year_values, str(value_basis))
        pv[str(dim)] = float((values * multipliers).sum())

    return pv or None

def scenario_metrics(
    df: pd.DataFrame,
    *,
    start_year: int,
    horizon_years: Optional[int] = None,
) -> Dict[str, float]:

    benefit_col = "BenefitFlowTotal" if "BenefitFlowTotal" in df.columns else "BenefitFlow"
    pv_col = "PVBenefitTotal" if "PVBenefitTotal" in df.columns else "PVBenefit"

    window = df

    if horizon_years is not None:

        try:

            limit_year = int(start_year) + int(horizon_years) - 1

            window = df[df["Year"].astype(int) <= limit_year]

        except (TypeError, ValueError):

            window = df

    spend_values = pd.to_numeric(window.get("Spend"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    year_values = pd.to_numeric(window.get("Year"), errors="coerce").fillna(start_year).astype(int).to_numpy(dtype=int)
    year_offsets = np.clip(year_values - int(start_year), a_min=0, a_max=None)
    spend_pv = float((spend_values / mbcm_discount_divisors(year_offsets)).sum()) if spend_values.size else 0.0

    return {

        "total_spend": float(spend_values.sum()),
        "total_spend_pv": spend_pv,

        "total_benefit": float(pd.to_numeric(window.get(benefit_col), errors="coerce").fillna(0.0).sum()),

        "total_pv": float(pd.to_numeric(window.get(pv_col), errors="coerce").fillna(0.0).sum()),

    }


def _effective_year_range(
    frames: Iterable[Optional[pd.DataFrame]],
    *,
    value_column: str = "Spend",
    threshold: float = 1e-9,
) -> Optional[Tuple[int, int]]:
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    for frame in frames:
        if frame is None or frame.empty:
            continue
        years = pd.to_numeric(frame.get("Year"), errors="coerce")
        if years.isnull().all():
            continue
        valid_years = years.dropna()
        if valid_years.empty:
            continue
        year_min = int(valid_years.min())
        min_year = year_min if min_year is None else min(min_year, year_min)
        values_raw = frame.get(value_column)
        if values_raw is None:
            values = pd.Series(np.zeros(len(frame)), index=frame.index)
        else:
            values = pd.to_numeric(values_raw, errors="coerce").fillna(0.0)
        active_mask = years.notna() & (values.abs() > threshold)
        if active_mask.any():
            year_max = int(years[active_mask].max())
        else:
            year_max = int(valid_years.max())
        max_year = year_max if max_year is None else max(max_year, year_max)
    if max_year is None:
        return None
    if min_year is None:
        min_year = max_year
    return (min_year, max_year)


def _annual_spend_breakdown_by_attribute(
    data: DashboardData,
    selection: ScenarioSelection,
    *,
    group_col: str,
    years: List[int],
) -> Optional[pd.DataFrame]:
    if data is None or selection is None or not getattr(selection, "code", None):
        return None
    spend_matrix = getattr(data, "spend_matrix", None)
    if not isinstance(spend_matrix, pd.DataFrame) or spend_matrix.empty:
        return None
    if group_col not in {"ActivityClass", "GPSTier", "Region", "StrategicDimension"}:
        return None

    subset = spend_matrix[spend_matrix["Code"] == selection.code].copy()
    if subset.empty:
        return None

    year_cols = [int(year) for year in years if year is not None and int(year) in subset.columns]
    if not year_cols:
        return None

    values = subset[year_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    values.insert(0, "Project", subset["Project"].astype(str))
    long = values.melt(id_vars="Project", value_vars=year_cols, var_name="Year", value_name="Spend")
    long["Spend"] = pd.to_numeric(long["Spend"], errors="coerce").fillna(0.0)
    long = long[long["Spend"].abs() > 1e-9]
    if long.empty:
        return None
    long["Year"] = pd.to_numeric(long["Year"], errors="coerce").astype(int)
    long["ProjectKey"] = long["Project"].map(_normalise_project_key)

    mapping = load_project_attribute_mapping()
    if mapping.empty or group_col not in mapping.columns:
        mapping = _build_project_attribute_mapping(COST_BENEFIT_STREAMS_PATH)
    if mapping.empty or group_col not in mapping.columns:
        return None

    merged = long.merge(mapping[["ProjectKey", group_col]], on="ProjectKey", how="left")
    merged[group_col] = merged[group_col].fillna("Unknown").astype(str)
    if group_col == "GPSTier":
        merged[group_col] = merged[group_col].map(_canonicalise_gps_tier)

    pivot = (
        merged.pivot_table(index="Year", columns=group_col, values="Spend", aggfunc="sum", fill_value=0.0)
        .reindex([int(year) for year in years], fill_value=0.0)
    )
    if pivot.empty:
        return None

    pivot = pivot.loc[:, pivot.abs().sum(axis=0) > 1e-9]
    if pivot.empty:
        return None

    totals = pivot.sum(axis=0).sort_values(ascending=False)
    ordered = totals.index.tolist()
    if "Unknown" in ordered:
        ordered = [col for col in ordered if col != "Unknown"] + ["Unknown"]
    pivot = pivot[ordered]

    return pivot


def _annual_benefit_breakdown_by_attribute(
    data: DashboardData,
    selection: ScenarioSelection,
    *,
    group_col: str,
    years: List[int],
) -> Optional[pd.DataFrame]:
    if data is None or selection is None or not getattr(selection, "code", None):
        return None
    if group_col not in {"ActivityClass", "GPSTier", "Region", "StrategicDimension"}:
        return None

    benefit_table = _scenario_project_benefit_table(data, selection.code, years)
    if benefit_table is None or benefit_table.empty:
        return None

    year_cols = [int(year) for year in years if year is not None and int(year) in benefit_table.columns]
    if not year_cols:
        return None

    values = benefit_table[year_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    values.insert(0, "Project", benefit_table["Project"].astype(str))
    long = values.melt(id_vars="Project", value_vars=year_cols, var_name="Year", value_name="Benefit")
    long["Benefit"] = pd.to_numeric(long["Benefit"], errors="coerce").fillna(0.0)
    long = long[long["Benefit"].abs() > 1e-9]
    if long.empty:
        return None
    long["Year"] = pd.to_numeric(long["Year"], errors="coerce").astype(int)
    long["ProjectKey"] = long["Project"].map(_normalise_project_key)

    mapping = load_project_attribute_mapping()
    if mapping.empty or group_col not in mapping.columns:
        mapping = _build_project_attribute_mapping(COST_BENEFIT_STREAMS_PATH)
    if mapping.empty or group_col not in mapping.columns:
        return None

    merged = long.merge(mapping[["ProjectKey", group_col]], on="ProjectKey", how="left")
    merged[group_col] = merged[group_col].fillna("Unknown").astype(str)
    if group_col == "GPSTier":
        merged[group_col] = merged[group_col].map(_canonicalise_gps_tier)

    pivot = (
        merged.pivot_table(index="Year", columns=group_col, values="Benefit", aggfunc="sum", fill_value=0.0)
        .reindex([int(year) for year in years], fill_value=0.0)
    )
    if pivot.empty:
        return None

    pivot = pivot.loc[:, pivot.abs().sum(axis=0) > 1e-9]
    if pivot.empty:
        return None

    totals = pivot.sum(axis=0).sort_values(ascending=False)
    ordered = totals.index.tolist()
    if "Unknown" in ordered:
        ordered = [col for col in ordered if col != "Unknown"] + ["Unknown"]
    pivot = pivot[ordered]

    return pivot


def _annual_cost_breakdown_by_dimension_weights(
    data: DashboardData,
    selection: ScenarioSelection,
    *,
    years: List[int],
) -> Optional[pd.DataFrame]:
    """Allocate spend by strategic dimension using project benefit weights."""
    if data is None or selection is None or not getattr(selection, "code", None):
        return None
    raw = _raw_result_for_code(data, selection.code)
    if not raw:
        return None
    benefits = raw.get("benefits_by_project_dimension_by_year")
    if not isinstance(benefits, pd.DataFrame) or benefits.empty:
        return None

    cost_table = _scenario_project_cost_table(data, selection.code, years)
    if cost_table is None or cost_table.empty:
        return None

    working = benefits.copy()
    if isinstance(working.index, pd.MultiIndex):
        working = working.reset_index()
    if "Project" not in working.columns or "Dimension" not in working.columns:
        return None

    col_map = {col: int(str(col)) for col in working.columns if str(col).isdigit()}
    if col_map:
        working = working.rename(columns=col_map)

    year_cols = [col for col in working.columns if isinstance(col, int)]
    if not year_cols:
        return None
    year_cols = [col for col in year_cols if col in years]
    if not year_cols:
        return None

    working[year_cols] = working[year_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    working["Project"] = working["Project"].astype(str)
    working["Dimension"] = (
        working["Dimension"]
        .apply(_canonicalise_dimension_label)
        .astype(str)
        .str.strip()
    )
    working = working[working["Dimension"].ne("")]
    working = working[working["Dimension"].str.strip().str.lower() != "total"]
    if working.empty:
        return None

    dim_totals = (
        working.groupby(["Project", "Dimension"], as_index=False)[year_cols]
        .sum()
    )
    dim_totals["BenefitTotal"] = dim_totals[year_cols].sum(axis=1)
    dim_totals = dim_totals[dim_totals["BenefitTotal"].abs() > 1e-9]
    if dim_totals.empty:
        return None

    project_totals = dim_totals.groupby("Project")["BenefitTotal"].sum().rename("ProjectTotal")
    dim_totals = dim_totals.merge(project_totals, on="Project", how="left")
    dim_totals = dim_totals[dim_totals["ProjectTotal"] > 0]
    if dim_totals.empty:
        return None
    dim_totals["Weight"] = dim_totals["BenefitTotal"] / dim_totals["ProjectTotal"]
    weights = dim_totals[["Project", "Dimension", "Weight"]]

    cost_year_cols = [col for col in cost_table.columns if isinstance(col, int)]
    if not cost_year_cols:
        return None
    cost_year_cols = [col for col in cost_year_cols if col in years]
    if not cost_year_cols:
        return None

    cost_long = cost_table.melt(
        id_vars="Project",
        value_vars=cost_year_cols,
        var_name="Year",
        value_name="Spend",
    )
    cost_long["Year"] = pd.to_numeric(cost_long["Year"], errors="coerce").astype(int)
    cost_long["Spend"] = pd.to_numeric(cost_long["Spend"], errors="coerce").fillna(0.0)

    merged = cost_long.merge(weights, on="Project", how="left")
    merged["Weight"] = pd.to_numeric(merged["Weight"], errors="coerce").fillna(0.0)
    merged["WeightedSpend"] = merged["Spend"] * merged["Weight"]

    pivot = (
        merged.pivot_table(
            index="Year",
            columns="Dimension",
            values="WeightedSpend",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex([int(year) for year in years], fill_value=0.0)
    )
    if pivot.empty:
        return None

    ordered: List[str] = []
    for dim in getattr(data, "dims", []):
        dim_label = str(dim).strip()
        if not dim_label or dim_label.lower() == "total":
            continue
        if dim_label in pivot.columns and dim_label not in ordered:
            ordered.append(dim_label)
    for dim in pivot.columns:
        dim_label = str(dim).strip()
        if not dim_label or dim_label.lower() == "total":
            continue
        if dim_label not in ordered:
            ordered.append(dim_label)
    if ordered:
        pivot = pivot[ordered]

    return pivot


def cash_chart(

    df: pd.DataFrame,

    title: str,

    *,

    color: str,

    spend_breakdown: str = "Total",

    data: Optional[DashboardData] = None,

    selection: Optional[ScenarioSelection] = None,

    comparison_selection: Optional[ScenarioSelection] = None,

    horizon_override: Optional[int] = None,

    x_axis_year_range: Optional[Tuple[int, int]] = None,

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

    breakdown_key = re.sub(r"\s+", " ", str(spend_breakdown or "Total").strip().lower())
    group_col: Optional[str] = None
    if breakdown_key == "activity class":
        group_col = "ActivityClass"
    elif breakdown_key == "region":
        group_col = "Region"
    elif breakdown_key in {"gps request tier", "gps tier request", "gps tier"}:
        group_col = "GPSTier"

    breakdown_table: Optional[pd.DataFrame] = None
    if group_col and data is not None and selection is not None:
        years = pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int).tolist()
        breakdown_table = _annual_spend_breakdown_by_attribute(
            data,
            selection,
            group_col=group_col,
            years=years,
        )

    if breakdown_table is None or breakdown_table.empty:
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
    else:
        group_labels = [str(col) for col in breakdown_table.columns]
        unknown_mask = [label.strip().lower() == "unknown" for label in group_labels]
        palette_count = len(group_labels) - sum(1 for flag in unknown_mask if flag)
        if palette_count <= 0:
            palette_count = len(group_labels)
        palette_positions = (
            [(0.55 + 0.95) / 2.0]
            if palette_count == 1
            else np.linspace(0.55, 0.95, palette_count).tolist()
        )
        palette = list(reversed(plc.sample_colorscale("Blues", palette_positions, colortype="rgb")))
        palette_iter = iter(palette)
        for label in group_labels:
            if label.strip().lower() == "unknown":
                trace_color = "rgb(156, 163, 175)"
            else:
                trace_color = next(palette_iter, "rgb(59, 130, 246)")
            values_m = pd.to_numeric(breakdown_table[label], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            values_dollars = values_m * 1_000_000.0
            fig.add_trace(
                go.Bar(
                    x=df["Year"],
                    y=values_dollars,
                    name=f"Spend: {label}",
                    marker_color=trace_color,
                    opacity=BAR_OPACITY,
                    customdata=[format_large_amount(val) for val in values_dollars],
                    hovertemplate=f"<b>Spend: {html.escape(label)}</b><br>FY %{{x}}: %{{customdata}}<extra></extra>",
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

    yaxis_title_default = None

    xaxis_config: Dict[str, Any] = {"title": None}
    if x_axis_year_range:
        start_year, end_year = x_axis_year_range
        if start_year is not None and end_year is not None and end_year >= start_year:
            padding = 0.4
            xaxis_config["range"] = [start_year - padding, end_year + padding]

    fig.update_layout(

        title=title,

        barmode="stack" if breakdown_table is not None and not breakdown_table.empty else "overlay",

        legend=legend_bottom(),

        xaxis=xaxis_config,

        yaxis_title=yaxis_title_default,

        template=plotly_template(),

        yaxis=dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text),

        hoverlabel=dict(namelength=-1),

    )

    return fig


def _analysis_activity_classes(
    spend_opt: Optional[pd.DataFrame],
    spend_cmp: Optional[pd.DataFrame],
    benefit_opt: Optional[pd.DataFrame],
    benefit_cmp: Optional[pd.DataFrame],
    *,
    limit: int = 5,
) -> List[str]:
    totals = pd.Series(dtype=float)
    for pivot in (spend_opt, spend_cmp, benefit_opt, benefit_cmp):
        if isinstance(pivot, pd.DataFrame) and not pivot.empty:
            totals = totals.add(pivot.sum(axis=0), fill_value=0.0)
    if totals.empty:
        columns: List[str] = []
        for pivot in (spend_opt, spend_cmp, benefit_opt, benefit_cmp):
            if isinstance(pivot, pd.DataFrame) and not pivot.empty:
                columns.extend([str(col) for col in pivot.columns])
        totals = pd.Series(1.0, index=pd.Index(columns)).groupby(level=0).sum()
    if totals.empty:
        return []
    totals = totals[totals.index.astype(str).str.strip().str.lower() != "total"]
    ordered = totals.sort_values(ascending=False).index.astype(str).tolist()
    if limit > 0:
        ordered = ordered[:limit]
    return ordered


def _analysis_dimension_labels(
    data: DashboardData,
    spend_opt: Optional[pd.DataFrame],
    spend_cmp: Optional[pd.DataFrame],
    benefit_opt: Optional[pd.DataFrame],
    benefit_cmp: Optional[pd.DataFrame],
    *,
    limit: int = 0,
) -> List[str]:
    labels: List[str] = []
    for dim in getattr(data, "dims", []) or []:
        dim_label = str(dim).strip()
        if not dim_label or dim_label.lower() == "total":
            continue
        if dim_label not in labels:
            labels.append(dim_label)

    for pivot in (spend_opt, spend_cmp, benefit_opt, benefit_cmp):
        if isinstance(pivot, pd.DataFrame) and not pivot.empty:
            for col in pivot.columns:
                col_label = str(col).strip()
                if not col_label:
                    continue
                if col_label.lower() in {"total", "unknown"}:
                    continue
                if col_label not in labels:
                    labels.append(col_label)

    if limit > 0 and len(labels) > limit:
        env_label = next(
            (label for label in labels if label.strip().lower() == "environmental sustainability"),
            None,
        )
        trimmed = labels[:limit]
        if env_label and env_label not in trimmed:
            trimmed = labels[: max(limit - 1, 0)]
            trimmed.append(env_label)
        labels = trimmed

    return labels


def _analysis_series_for_class(
    pivot: Optional[pd.DataFrame],
    label: str,
    years: List[int],
) -> pd.Series:
    year_index = pd.Index([int(year) for year in years], dtype=int)
    if pivot is None or pivot.empty or label not in pivot.columns:
        return pd.Series(0.0, index=year_index, dtype=float)
    series = pd.to_numeric(pivot[label], errors="coerce").fillna(0.0)
    series = pd.Series(series.to_numpy(dtype=float), index=pivot.index.astype(int))
    return series.reindex(year_index, fill_value=0.0)


def _analysis_last_nonzero_year(series: pd.Series) -> Optional[int]:
    if series is None or series.empty:
        return None
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    mask = values.abs().to_numpy() > 1e-9
    if not mask.any():
        return None
    years = pd.Index(series.index).astype(int)
    return int(years[mask].max())


def _analysis_max_year_for_classes(
    classes: List[str],
    pivot_opt: Optional[pd.DataFrame],
    pivot_cmp: Optional[pd.DataFrame],
    years: List[int],
) -> Optional[int]:
    if not classes:
        return None
    last_years: List[int] = []
    for label in classes:
        series_opt = _analysis_series_for_class(pivot_opt, label, years)
        series_cmp = _analysis_series_for_class(pivot_cmp, label, years)
        last_opt = _analysis_last_nonzero_year(series_opt)
        last_cmp = _analysis_last_nonzero_year(series_cmp)
        candidates = [year for year in (last_opt, last_cmp) if year is not None]
        if candidates:
            last_years.append(max(candidates))
    if not last_years:
        return max(years) if years else None
    return max(last_years)


def _analysis_max_value_for_classes(
    classes: List[str],
    pivot_opt: Optional[pd.DataFrame],
    pivot_cmp: Optional[pd.DataFrame],
    years: List[int],
) -> float:
    if not classes:
        return 0.0
    max_value = 0.0
    for label in classes:
        series_opt = _analysis_series_for_class(pivot_opt, label, years)
        series_cmp = _analysis_series_for_class(pivot_cmp, label, years)
        max_value = max(
            max_value,
            float(pd.to_numeric(series_opt, errors="coerce").fillna(0.0).max()),
            float(pd.to_numeric(series_cmp, errors="coerce").fillna(0.0).max()),
        )
    if max_value <= 0:
        return 0.0
    return max_value * 1_000_000.0 * 1.05


def _analysis_sparkline_chart(
    label: str,
    years: List[int],
    series_opt: pd.Series,
    series_cmp: pd.Series,
    *,
    opt_label: str,
    cmp_label: str,
    color_opt: str,
    color_cmp: str,
    x_max: Optional[int],
    y_max: Optional[float],
    show_xaxis: bool,
) -> go.Figure:
    year_index = pd.Index([int(year) for year in years], dtype=int)
    opt_values = pd.to_numeric(series_opt, errors="coerce").fillna(0.0).to_numpy(dtype=float) * 1_000_000.0
    cmp_values = pd.to_numeric(series_cmp, errors="coerce").fillna(0.0).to_numpy(dtype=float) * 1_000_000.0

    fig = go.Figure()
    if opt_label:
        fig.add_trace(
            go.Scatter(
                x=year_index,
                y=opt_values,
                name=opt_label,
                mode="lines",
                line=dict(color=color_opt, width=2),
                customdata=[format_large_amount(val) for val in opt_values],
                hovertemplate=f"<b>{html.escape(opt_label)}</b><br>FY %{{x}}: %{{customdata}}<extra></extra>",
            )
        )
    if cmp_label:
        fig.add_trace(
            go.Scatter(
                x=year_index,
                y=cmp_values,
                name=cmp_label,
                mode="lines",
                line=dict(color=color_cmp, width=2),
                customdata=[format_large_amount(val) for val in cmp_values],
                hovertemplate=f"<b>{html.escape(cmp_label)}</b><br>FY %{{x}}: %{{customdata}}<extra></extra>",
            )
        )

    min_year = int(year_index.min()) if len(year_index) else 0
    max_year = int(x_max) if x_max is not None else (int(year_index.max()) if len(year_index) else min_year)
    fig.update_layout(
        title=dict(text=label, x=0.01, y=0.92, xanchor="left", yanchor="top", font=dict(size=12)),
        height=120,
        margin=dict(l=30, r=20, t=26, b=18 if show_xaxis else 6),
        showlegend=False,
        template=plotly_template(),
        hoverlabel=dict(namelength=-1),
    )
    fig.update_xaxes(
        title=None,
        showgrid=True,
        showticklabels=show_xaxis,
        range=[min_year - 0.4, max_year + 0.4],
        zeroline=False,
    )
    yaxis_config: Dict[str, Any] = {
        "title": None,
        "showgrid": True,
        "showticklabels": False,
        "zeroline": False,
    }
    if y_max is not None and y_max > 0:
        pad = float(y_max) * 0.02
        if pad <= 0:
            pad = 1.0
        yaxis_config["range"] = [-pad, float(y_max) + pad]
    fig.update_yaxes(**yaxis_config)
    return fig


def _timing_center_from_pivot(
    pivot: Optional[pd.DataFrame],
) -> Tuple[pd.Series, pd.Series]:
    if pivot is None or pivot.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    values = pivot.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    years = pd.Index(pivot.index).astype(int)
    totals = values.sum(axis=0)
    weighted = values.mul(years, axis=0).sum(axis=0)
    centers = weighted.divide(totals.replace(0.0, np.nan))
    return centers.astype(float), totals.astype(float)


def _overall_timing_center(pivot: Optional[pd.DataFrame]) -> Optional[float]:
    if pivot is None or pivot.empty:
        return None
    values = pivot.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    totals_by_year = values.sum(axis=1)
    total = float(totals_by_year.sum())
    if total <= 0:
        return None
    years = pd.Index(pivot.index).astype(int)
    weighted = (totals_by_year.to_numpy(dtype=float) * years.to_numpy(dtype=int)).sum()
    return float(weighted / total)


def _timing_paddle_chart(
    pivot_opt: Optional[pd.DataFrame],
    pivot_cmp: Optional[pd.DataFrame],
    *,
    breakdown_label: str,
    opt_label: str,
    cmp_label: str,
) -> Optional[go.Figure]:
    centers_opt, totals_opt = _timing_center_from_pivot(pivot_opt)
    centers_cmp, totals_cmp = _timing_center_from_pivot(pivot_cmp)

    combined_totals = totals_opt.add(totals_cmp, fill_value=0.0)
    combined_totals = combined_totals[combined_totals.abs() > 1e-9]
    if combined_totals.empty:
        return None

    total_sum = float(combined_totals.sum())
    categories = combined_totals.sort_values(ascending=False).index.astype(str).tolist()
    share_labels: List[str] = []
    for cat in categories:
        share = (combined_totals.get(cat, 0.0) / total_sum * 100.0) if total_sum else 0.0
        share_labels.append(f"{cat} ({share:.1f}%)")

    line_x: List[Optional[float]] = []
    line_y: List[str] = []
    opt_x: List[float] = []
    opt_y: List[str] = []
    cmp_x: List[float] = []
    cmp_y: List[str] = []
    text_x: List[float] = []
    text_y: List[str] = []
    text_vals: List[str] = []

    for cat, label in zip(categories, share_labels):
        opt_val = centers_opt.get(cat, np.nan)
        cmp_val = centers_cmp.get(cat, np.nan)
        if np.isfinite(opt_val):
            opt_x.append(float(opt_val))
            opt_y.append(label)
        if np.isfinite(cmp_val):
            cmp_x.append(float(cmp_val))
            cmp_y.append(label)
        if np.isfinite(opt_val) and np.isfinite(cmp_val):
            line_x.extend([float(opt_val), float(cmp_val), None])
            line_y.extend([label, label, None])
            delta = float(opt_val) - float(cmp_val)
            text_x.append((float(opt_val) + float(cmp_val)) / 2.0)
            text_y.append(label)
            text_vals.append(f"{delta:+.1f}y")

    overall_opt = _overall_timing_center(pivot_opt)
    overall_cmp = _overall_timing_center(pivot_cmp)
    if overall_opt is not None and overall_cmp is not None:
        overall_shift = overall_opt - overall_cmp
        overall_center = (overall_opt + overall_cmp) / 2.0
    elif overall_opt is not None:
        overall_shift = 0.0
        overall_center = overall_opt
    elif overall_cmp is not None:
        overall_shift = 0.0
        overall_center = overall_cmp
    else:
        overall_shift = 0.0
        overall_center = None

    fig = go.Figure()
    if line_x:
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(color="rgba(71, 85, 105, 0.6)", width=2.0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if opt_x:
        fig.add_trace(
            go.Scatter(
                x=opt_x,
                y=opt_y,
                mode="markers",
                marker=dict(color=PRIMARY_COLOR, size=9, line=dict(color="#ffffff", width=1)),
                name=opt_label,
                customdata=opt_x,
                hovertemplate="<b>%{y}</b><br>Timing center: %{customdata:.1f}y<extra></extra>",
            )
        )
    if cmp_x:
        fig.add_trace(
            go.Scatter(
                x=cmp_x,
                y=cmp_y,
                mode="markers",
                marker=dict(color=CAPACITY_HEAT_RED, size=9, line=dict(color="#ffffff", width=1)),
                name=cmp_label,
                customdata=cmp_x,
                hovertemplate="<b>%{y}</b><br>Timing center: %{customdata:.1f}y<extra></extra>",
            )
        )
    if text_x:
        fig.add_trace(
            go.Scatter(
                x=text_x,
                y=text_y,
                mode="text",
                text=text_vals,
                textfont=dict(color="#0f172a", size=12),
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if overall_center is not None:
        fig.add_vline(
            x=overall_center,
            line=dict(color="rgba(15, 23, 42, 0.45)", width=1.5, dash="dash"),
        )

    title_text = f"{breakdown_label} - overall programme timing shift: {overall_shift:+.2f} yrs"
    height = max(360, 140 + len(categories) * 38)
    fig.update_layout(
        title=title_text,
        xaxis_title="Year (weighted by spend)",
        yaxis_title="Category (share of total spend)",
        template=plotly_template(),
        hoverlabel=dict(namelength=-1),
        legend=legend_bottom(y=-0.28),
        margin=dict(l=200, r=40, t=70, b=120),
        height=height,
    )
    fig.update_xaxes(title_standoff=18)
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=share_labels[::-1],
    )

    return fig


def _analysis_group_totals(pivot: Optional[pd.DataFrame]) -> pd.Series:
    if pivot is None or pivot.empty:
        return pd.Series(dtype=float)
    values = pivot.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return values.sum(axis=0).astype(float)


def _analysis_group_order_and_share(
    totals_opt: pd.Series,
    totals_cmp: pd.Series,
) -> Tuple[List[str], pd.Series]:
    combined = totals_opt.add(totals_cmp, fill_value=0.0)
    combined = combined[combined.abs() > 1e-9]
    if combined.empty:
        return [], pd.Series(dtype=float)
    if totals_opt.abs().sum() > 1e-9:
        base_totals = totals_opt
    elif totals_cmp.abs().sum() > 1e-9:
        base_totals = totals_cmp
    else:
        base_totals = combined
    base_totals = base_totals.reindex(combined.index).fillna(0.0)
    order = base_totals.sort_values(ascending=False).index.astype(str).tolist()
    if "Unknown" in order:
        order = [val for val in order if val != "Unknown"] + ["Unknown"]
    total_sum = float(base_totals.sum())
    share = base_totals / total_sum if total_sum > 0 else base_totals * 0.0
    share = share.reindex(order).fillna(0.0)
    return order, share


def _analysis_matrix_from_pivot(
    pivot: Optional[pd.DataFrame],
    groups: List[str],
    years: List[int],
) -> pd.DataFrame:
    if not groups or not years:
        return pd.DataFrame()
    year_index = pd.Index([int(year) for year in years], dtype=int)
    if pivot is None or pivot.empty:
        return pd.DataFrame(0.0, index=groups, columns=year_index)
    values = (
        pivot.reindex(index=year_index, columns=groups, fill_value=0.0)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    values = values.T
    values.index = values.index.astype(str)
    return values


def _analysis_density_from_values(values: pd.DataFrame) -> pd.DataFrame:
    if values is None or values.empty:
        return pd.DataFrame()
    totals = values.sum(axis=1).replace(0.0, np.nan)
    density = values.divide(totals, axis=0).fillna(0.0)
    return density


def _analysis_heatmap_chart(
    matrix: pd.DataFrame,
    *,
    title: str,
    y_labels: List[str],
    colorscale: List[Any],
    customdata: Optional[List[List[str]]] = None,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    zmid: Optional[float] = None,
    hovertemplate: str,
    show_colorbar: bool,
    colorbar_title: Optional[str],
    height: int,
    left_margin: int,
    right_margin: int,
    show_ylabels: bool,
) -> go.Figure:
    years = matrix.columns.tolist()
    fig = go.Figure(
        go.Heatmap(
            x=years,
            y=y_labels,
            z=matrix.to_numpy(dtype=float),
            customdata=customdata,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            zmid=zmid,
            showscale=show_colorbar,
            xgap=0,
            ygap=1,
            hovertemplate=hovertemplate,
            colorbar=dict(
                title=colorbar_title or "",
                len=0.72,
                thickness=12,
                x=1.02,
                y=0.5,
                ticks="outside",
                outlinewidth=0,
                bgcolor="rgba(0,0,0,0)",
            )
            if show_colorbar
            else None,
        )
    )
    fig.update_traces(hoverlabel=_hoverlabel_style())
    fig.update_layout(
        title=title,
        template=plotly_template(),
        height=height,
        margin=dict(l=left_margin, r=right_margin, t=70, b=50),
    )
    fig.update_xaxes(title="Year", showgrid=False, zeroline=False)
    fig.update_yaxes(
        title="Group (share of total spend)" if show_ylabels else None,
        showgrid=False,
        zeroline=False,
        showticklabels=show_ylabels,
        autorange="reversed",
    )
    return fig


def _analysis_spend_density_charts(
    pivot_opt: Optional[pd.DataFrame],
    pivot_cmp: Optional[pd.DataFrame],
    *,
    years: List[int],
    breakdown_label: str,
    opt_label: str,
    cmp_label: str,
    diff_mode: str,
) -> Optional[Tuple[go.Figure, go.Figure, go.Figure]]:
    totals_opt = _analysis_group_totals(pivot_opt)
    totals_cmp = _analysis_group_totals(pivot_cmp)
    groups, share = _analysis_group_order_and_share(totals_opt, totals_cmp)
    if not groups or not years:
        return None

    year_index = pd.Index([int(year) for year in years], dtype=int)
    last_year: Optional[int] = None
    for pivot in (pivot_opt, pivot_cmp):
        if pivot is None or pivot.empty:
            continue
        values = (
            pivot.reindex(index=year_index, fill_value=0.0)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        active = values.sum(axis=1)
        active = active[active.abs() > 1e-9]
        if not active.empty:
            candidate = int(active.index.max())
            last_year = candidate if last_year is None else max(last_year, candidate)
    if last_year is None:
        last_year = int(year_index.max())
    trimmed_years = [int(year) for year in years if int(year) <= last_year]
    if not trimmed_years:
        trimmed_years = [int(year) for year in years]

    values_opt = _analysis_matrix_from_pivot(pivot_opt, groups, trimmed_years)
    values_cmp = _analysis_matrix_from_pivot(pivot_cmp, groups, trimmed_years)
    density_opt = _analysis_density_from_values(values_opt)
    density_cmp = _analysis_density_from_values(values_cmp)

    diff_values = values_opt.subtract(values_cmp, fill_value=0.0)
    if totals_opt.abs().sum() > 1e-9:
        base_totals = totals_opt
    elif totals_cmp.abs().sum() > 1e-9:
        base_totals = totals_cmp
    else:
        base_totals = totals_opt.add(totals_cmp, fill_value=0.0)
    base_totals = base_totals.reindex(groups).fillna(0.0).replace(0.0, np.nan)
    diff_ratio = diff_values.divide(base_totals, axis=0).fillna(0.0)

    label_share = (share.reindex(groups).fillna(0.0) * 100.0).to_numpy(dtype=float)
    y_labels = [f"{group} ({pct:.1f}%)" for group, pct in zip(groups, label_share)]

    density_max = float(
        max(
            density_opt.to_numpy(dtype=float).max() if not density_opt.empty else 0.0,
            density_cmp.to_numpy(dtype=float).max() if not density_cmp.empty else 0.0,
            0.05,
        )
    )

    diff_matrix = diff_ratio if diff_mode == "relative" else diff_values
    diff_max = float(np.nanmax(np.abs(diff_matrix.to_numpy(dtype=float)))) if not diff_matrix.empty else 0.0
    if diff_max <= 0:
        diff_max = 0.05 if diff_mode == "relative" else 1.0

    height = max(300, int(len(groups) * 26 + 160))
    density_scale = plc.sequential.Blues
    diff_scale = plc.diverging.RdBu

    opt_title = f"{opt_label}: density (row sums to 100%)"
    cmp_title = f"{cmp_label}: density (row sums to 100%)"
    diff_title = (
        f"Differential density ({opt_label} - {cmp_label}), as % of group total"
        if diff_mode == "relative"
        else f"Differential spend ({opt_label} - {cmp_label}), absolute ($m)"
    )

    density_hover = "<b>%{y}</b><br>FY %{x}<br>Density: %{z:.2%}<extra></extra>"
    diff_values_arr = diff_matrix.to_numpy(dtype=float)
    if diff_mode == "relative":
        diff_custom = [
            ["" if not np.isfinite(val) else f"{val:+.2%}" for val in row]
            for row in diff_values_arr
        ]
        diff_hover = "<b>%{y}</b><br>FY %{x}<br>Δ / group total: %{customdata}<extra></extra>"
    else:
        diff_custom = [
            ["" if not np.isfinite(val) else f"{val:+,.2f}m" for val in row]
            for row in diff_values_arr
        ]
        diff_hover = "<b>%{y}</b><br>FY %{x}<br>Δ spend: %{customdata}<extra></extra>"

    fig_opt = _analysis_heatmap_chart(
        density_opt.reindex(index=groups).reindex(columns=trimmed_years, fill_value=0.0),
        title=opt_title,
        y_labels=y_labels,
        colorscale=density_scale,
        zmin=0.0,
        zmax=density_max,
        hovertemplate=density_hover,
        show_colorbar=False,
        colorbar_title=None,
        height=height,
        left_margin=180,
        right_margin=20,
        show_ylabels=True,
    )
    fig_cmp = _analysis_heatmap_chart(
        density_cmp.reindex(index=groups).reindex(columns=trimmed_years, fill_value=0.0),
        title=cmp_title,
        y_labels=y_labels,
        colorscale=density_scale,
        zmin=0.0,
        zmax=density_max,
        hovertemplate=density_hover,
        show_colorbar=False,
        colorbar_title=None,
        height=height,
        left_margin=40,
        right_margin=20,
        show_ylabels=False,
    )
    fig_diff = _analysis_heatmap_chart(
        diff_matrix.reindex(index=groups).reindex(columns=trimmed_years, fill_value=0.0),
        title=diff_title,
        y_labels=y_labels,
        colorscale=diff_scale,
        customdata=diff_custom,
        zmin=-diff_max,
        zmax=diff_max,
        zmid=0.0,
        hovertemplate=diff_hover,
        show_colorbar=True,
        colorbar_title="Δ / group total" if diff_mode == "relative" else "Δ spend ($m)",
        height=height,
        left_margin=40,
        right_margin=60,
        show_ylabels=False,
    )
    return fig_opt, fig_cmp, fig_diff


def _analysis_total_spend_by_year(
    data: DashboardData,
    selection: ScenarioSelection,
    years: pd.Index,
) -> Optional[pd.Series]:
    if data is None or selection is None or not getattr(selection, "code", None):
        return None
    spend_matrix = getattr(data, "spend_matrix", None)
    if not isinstance(spend_matrix, pd.DataFrame) or spend_matrix.empty:
        return None
    subset = spend_matrix[spend_matrix["Code"] == selection.code].copy()
    if subset.empty:
        return None
    year_cols = [int(year) for year in years if int(year) in subset.columns]
    if not year_cols:
        return None
    totals = subset[year_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=0)
    totals.index = pd.Index(year_cols, dtype=int)
    return totals


def _analysis_share_matrix(
    pivot: pd.DataFrame,
    totals_by_year: pd.Series,
    *,
    within_group: bool,
) -> pd.DataFrame:
    if pivot is None or pivot.empty:
        return pd.DataFrame()
    if within_group:
        denom = pivot.sum(axis=1).replace(0.0, np.nan)
    else:
        denom = totals_by_year.replace(0.0, np.nan)
    return pivot.div(denom, axis=0).fillna(0.0)


def _analysis_zscore_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    means = frame.mean(axis=0)
    stds = frame.std(axis=0, ddof=0).replace(0.0, np.nan)
    return frame.sub(means, axis=1).div(stds, axis=1).fillna(0.0)


def _analysis_clr_transform(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
    if frame is None or frame.empty:
        return None
    values = frame.to_numpy(dtype=float)
    positive = values[values > 0]
    if positive.size == 0:
        return None
    pseudocount = float(np.min(positive)) * 0.5
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        pseudocount = 1e-9
    adjusted = np.where(values <= 0, pseudocount, values)
    log_vals = np.log(adjusted)
    mean_log = log_vals.mean(axis=1, keepdims=True)
    clr_vals = log_vals - mean_log
    return pd.DataFrame(clr_vals, index=frame.index, columns=frame.columns)


def _analysis_correlation_help_text(
    mode: str,
    *,
    change_basis: str = "spend",
    zscore: bool = False,
    group_filter: Optional[str] = None,
) -> str:
    if mode == "absolute":
        if zscore:
            return (
                "Pearson correlation of annual spend levels by category.\n"
                "Each category is z-scored before correlation to remove scale effects."
            )
        return "Pearson correlation of annual spend levels by category."
    if mode == "yoy":
        if change_basis == "share":
            return (
                "Pearson correlation of year-over-year changes in spend share by category.\n"
                "Share is measured as a fraction of total annual spend."
            )
        return "Pearson correlation of year-over-year changes in annual spend by category."
    if mode == "clr":
        group_label = group_filter or "selected group"
        return (
            f"Aitchison-style correlation of CLR-transformed spend shares within {group_label}.\n"
            "Uses a small pseudocount for zeros to avoid log(0)."
        )
    group_label = group_filter or "selected group"
    return (
        f"Pearson correlation of spend share time series within {group_label} only.\n"
        "Avoids cross-group mixing (Tier vs Activity vs Region vs Dimension)."
    )


def _analysis_correlation_title(
    mode: str,
    *,
    change_basis: str = "spend",
    zscore: bool = False,
    group_filter: Optional[str] = None,
) -> str:
    if mode == "absolute":
        if zscore:
            return "Correlation matrix - absolute spend (z-scored, Pearson r)"
        return "Correlation matrix - absolute spend (Pearson r)"
    if mode == "yoy":
        basis_label = "spend share" if change_basis == "share" else "spend"
        return f"Correlation matrix - YoY change in {basis_label} (Pearson r)"
    if mode == "clr":
        group_label = group_filter or "group"
        return f"Correlation matrix - compositional CLR/Aitchison ({group_label})"
    group_label = group_filter or "group"
    return f"Correlation matrix - spend share by {group_label} (Pearson r)"


def _analysis_correlation_matrix_data(
    data: DashboardData,
    selection: ScenarioSelection,
    years: List[int],
    *,
    mode: str = "absolute",
    change_basis: str = "spend",
    zscore: bool = False,
    group_filter: Optional[str] = None,
    include_share_pct: bool = True,
) -> Optional[pd.DataFrame]:
    if data is None or selection is None or not getattr(selection, "code", None):
        return None
    year_index = pd.Index([int(year) for year in years], dtype=int)
    totals_by_year = _analysis_total_spend_by_year(data, selection, year_index)
    if totals_by_year is None or totals_by_year.abs().sum() <= 1e-9:
        return None
    active_years = totals_by_year[totals_by_year.abs() > 1e-9]
    if active_years.empty:
        return None
    last_year = int(active_years.index.max())
    year_index = year_index[year_index <= last_year]
    totals_by_year = totals_by_year.reindex(year_index).fillna(0.0)
    overall_total = float(totals_by_year.sum())
    if overall_total <= 0:
        return None
    if mode == "yoy" and len(year_index) < 2:
        return None

    series_entries: List[pd.Series] = []
    label_entries: List[str] = []
    share_entries: List[float] = []
    group_specs = [
        ("Activity", "ActivityClass", False),
        ("Tier", "GPSTier", False),
        ("Region", "Region", False),
        ("Dimension", "StrategicDimension", True),
    ]
    if group_filter:
        group_specs = [spec for spec in group_specs if spec[0] == group_filter]
        if not group_specs:
            return None
    for prefix, group_col, is_dimension in group_specs:
        if is_dimension:
            pivot = _annual_cost_breakdown_by_dimension_weights(
                data,
                selection,
                years=year_index.tolist(),
            )
        else:
            pivot = _annual_spend_breakdown_by_attribute(
                data,
                selection,
                group_col=group_col,
                years=year_index.tolist(),
            )
        if pivot is None or pivot.empty:
            continue
        pivot = pivot.reindex(year_index, fill_value=0.0)
        pivot = pivot.loc[:, pivot.abs().sum(axis=0) > 1e-9]
        if pivot.empty:
            continue
        category_totals = pivot.sum(axis=0)
        group_total = float(category_totals.sum())
        if group_total <= 0:
            continue
        within_group_share = mode in {"group_share", "clr"} or (mode == "yoy" and change_basis == "share" and group_filter)
        share_matrix = None
        if mode in {"group_share", "clr"} or (mode == "yoy" and change_basis == "share"):
            share_matrix = _analysis_share_matrix(
                pivot,
                totals_by_year,
                within_group=within_group_share,
            )
        if mode == "absolute":
            working = pivot
        elif mode == "yoy":
            if change_basis == "share" and share_matrix is not None and not share_matrix.empty:
                working = share_matrix.diff().iloc[1:]
            else:
                working = pivot.diff().iloc[1:]
        elif mode == "group_share":
            working = share_matrix if share_matrix is not None else pd.DataFrame()
        elif mode == "clr":
            if share_matrix is None or share_matrix.empty:
                continue
            active_rows = share_matrix.sum(axis=1) > 1e-9
            clr_source = share_matrix.loc[active_rows]
            clr_matrix = _analysis_clr_transform(clr_source)
            if clr_matrix is None or clr_matrix.empty:
                continue
            working = clr_matrix
        else:
            return None
        if working is None or working.empty:
            continue
        for category in pivot.columns:
            if category not in working.columns:
                continue
            series = working[category].astype(float)
            if series.abs().sum() <= 1e-9:
                continue
            denom_total = group_total if mode in {"group_share", "clr"} or group_filter else overall_total
            share_pct = (category_totals[category] / denom_total) * 100.0
            if include_share_pct:
                label = f"{prefix}: {category} ({share_pct:.1f}%)"
            else:
                label = f"{prefix}: {category}"
            series_entries.append(series.rename(label))
            label_entries.append(label)
            share_entries.append(share_pct)

    if not series_entries:
        return None
    data_matrix = pd.concat(series_entries, axis=1).fillna(0.0)
    if data_matrix.shape[0] < 2 or data_matrix.shape[1] < 2:
        return None
    if mode == "absolute" and zscore:
        data_matrix = _analysis_zscore_frame(data_matrix)
    share_lookup = dict(zip(label_entries, share_entries))
    group_priority = {"Activity": 0, "Tier": 1, "Region": 2, "Dimension": 3}
    def _label_sort_key(label: str) -> Tuple[int, float, str]:
        prefix = label.split(":", 1)[0].strip()
        return (
            group_priority.get(prefix, 99),
            -float(share_lookup.get(label, 0.0)),
            label,
        )
    if group_filter:
        order = sorted(
            data_matrix.columns,
            key=lambda label: (-float(share_lookup.get(label, 0.0)), label),
        )
    else:
        order = sorted(data_matrix.columns, key=_label_sort_key)
    data_matrix = data_matrix[order]
    corr = data_matrix.corr(method="pearson").fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    return corr


def _analysis_correlation_matrix_chart(
    data: DashboardData,
    selection: ScenarioSelection,
    years: List[int],
    *,
    mode: str = "absolute",
    change_basis: str = "spend",
    zscore: bool = False,
    group_filter: Optional[str] = None,
    include_share_pct: bool = True,
    title_suffix: Optional[str] = None,
) -> Optional[go.Figure]:
    corr = _analysis_correlation_matrix_data(
        data,
        selection,
        years,
        mode=mode,
        change_basis=change_basis,
        zscore=zscore,
        group_filter=group_filter,
        include_share_pct=include_share_pct,
    )
    if corr is None or corr.empty:
        return None

    labels = corr.columns.tolist()
    height = max(420, 24 * len(labels) + 120)
    title = _analysis_correlation_title(
        mode,
        change_basis=change_basis,
        zscore=zscore,
        group_filter=group_filter,
    )
    if title_suffix:
        title = f"{title} - {title_suffix}"
    fig = go.Figure(
        go.Heatmap(
            x=labels,
            y=labels,
            z=corr.to_numpy(dtype=float),
            zmin=-1.0,
            zmax=1.0,
            zmid=0.0,
            colorscale=plc.diverging.RdBu,
            colorbar=dict(
                title="Pearson r",
                len=0.75,
                thickness=12,
                x=1.02,
                y=0.5,
                ticks="outside",
                outlinewidth=0,
                bgcolor="rgba(0,0,0,0)",
            ),
            hovertemplate="X=%{x}<br>Y=%{y}<br>r=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_traces(hoverlabel=_hoverlabel_style())
    fig.update_layout(
        title=title,
        template=plotly_template(),
        height=height,
        margin=dict(l=220, r=60, t=70, b=140),
    )
    fig.update_xaxes(title=None, showgrid=False, zeroline=False, tickangle=-45)
    fig.update_yaxes(title=None, showgrid=False, zeroline=False, autorange="reversed")
    return fig


def _analysis_correlation_matrix_delta_chart(
    data: DashboardData,
    selection_x: ScenarioSelection,
    selection_y: ScenarioSelection,
    years: List[int],
    *,
    mode: str = "absolute",
    change_basis: str = "spend",
    zscore: bool = False,
    group_filter: Optional[str] = None,
    label_x: str = "Scenario X",
    label_y: str = "Scenario Y",
) -> Optional[go.Figure]:
    corr_x = _analysis_correlation_matrix_data(
        data,
        selection_x,
        years,
        mode=mode,
        change_basis=change_basis,
        zscore=zscore,
        group_filter=group_filter,
        include_share_pct=False,
    )
    corr_y = _analysis_correlation_matrix_data(
        data,
        selection_y,
        years,
        mode=mode,
        change_basis=change_basis,
        zscore=zscore,
        group_filter=group_filter,
        include_share_pct=False,
    )
    if corr_x is None or corr_y is None or corr_x.empty or corr_y.empty:
        return None
    common = corr_x.index.intersection(corr_y.index)
    if len(common) < 2:
        return None
    order = [label for label in corr_x.columns if label in common]
    corr_x = corr_x.reindex(index=order, columns=order)
    corr_y = corr_y.reindex(index=order, columns=order)
    diff = corr_x - corr_y
    diff_max = float(np.nanmax(np.abs(diff.to_numpy(dtype=float)))) if not diff.empty else 0.0
    if diff_max <= 1e-9:
        diff_max = 1.0

    labels = diff.columns.tolist()
    height = max(420, 24 * len(labels) + 120)
    base_title = _analysis_correlation_title(
        mode,
        change_basis=change_basis,
        zscore=zscore,
        group_filter=group_filter,
    )
    title = f"{base_title} delta ({label_x} - {label_y})"
    fig = go.Figure(
        go.Heatmap(
            x=labels,
            y=labels,
            z=diff.to_numpy(dtype=float),
            zmin=-diff_max,
            zmax=diff_max,
            zmid=0.0,
            colorscale=plc.diverging.RdBu,
            colorbar=dict(
                title="Delta Pearson r",
                len=0.75,
                thickness=12,
                x=1.02,
                y=0.5,
                ticks="outside",
                outlinewidth=0,
                bgcolor="rgba(0,0,0,0)",
            ),
            hovertemplate="X=%{x}<br>Y=%{y}<br>delta r=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_traces(hoverlabel=_hoverlabel_style())
    fig.update_layout(
        title=title,
        template=plotly_template(),
        height=height,
        margin=dict(l=220, r=60, t=70, b=140),
    )
    fig.update_xaxes(title=None, showgrid=False, zeroline=False, tickangle=-45)
    fig.update_yaxes(title=None, showgrid=False, zeroline=False, autorange="reversed")
    return fig


def _analysis_delta_bar_line_chart(
    pivot_opt: Optional[pd.DataFrame],
    pivot_cmp: Optional[pd.DataFrame],
    *,
    years: List[int],
    group_label: str,
    breakdown_label: str,
    opt_label: str,
    cmp_label: str,
) -> Optional[go.Figure]:
    if not years:
        return None
    year_index = pd.Index([int(year) for year in years], dtype=int)
    series_opt = _analysis_series_for_class(pivot_opt, group_label, years)
    series_cmp = _analysis_series_for_class(pivot_cmp, group_label, years)
    if series_opt.abs().sum() <= 1e-9 and series_cmp.abs().sum() <= 1e-9:
        return None

    delta_m = pd.to_numeric(series_opt, errors="coerce").fillna(0.0).reindex(year_index, fill_value=0.0)
    delta_m = delta_m - pd.to_numeric(series_cmp, errors="coerce").fillna(0.0).reindex(year_index, fill_value=0.0)
    delta_nzd = delta_m.to_numpy(dtype=float) * 1_000_000.0
    cumulative_nzd = np.cumsum(delta_nzd)

    max_abs_annual = float(np.nanmax(np.abs(delta_nzd))) if delta_nzd.size else 0.0
    max_abs_cum = float(np.nanmax(np.abs(cumulative_nzd))) if cumulative_nzd.size else 0.0
    if max_abs_annual <= 0:
        max_abs_annual = 1.0
    if max_abs_cum <= 0:
        max_abs_cum = 1.0
    pad_annual = max_abs_annual * 0.05
    pad_cum = max_abs_cum * 0.05
    annual_ticks, annual_text, _ = compute_cash_axis_ticks([-max_abs_annual, max_abs_annual])
    cumulative_ticks, cumulative_text, _ = compute_cash_axis_ticks([-max_abs_cum, max_abs_cum])

    bar_color = POWERBI_GREEN
    line_color = POWERBI_BLUE

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=year_index,
            y=delta_nzd,
            name="Annual Δ spend",
            yaxis="y1",
            marker_color=bar_color,
            opacity=0.85,
            customdata=[format_large_amount(val) for val in delta_nzd],
            hovertemplate="<b>Annual Δ spend</b><br>FY %{x}: %{customdata}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=year_index,
            y=cumulative_nzd,
            name="Cumulative Δ spend",
            mode="lines",
            line=dict(color=line_color, width=2),
            yaxis="y2",
            customdata=[format_large_amount(val) for val in cumulative_nzd],
            hovertemplate="<b>Cumulative Δ spend</b><br>FY %{x}: %{customdata}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color="rgba(15, 23, 42, 0.35)", width=1, dash="dash"))
    fig.update_layout(
        title=f"Annual Δ$ + cumulative Δ$ — {group_label}",
        template=plotly_template(),
        height=320,
        margin=dict(l=70, r=70, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=_hoverlabel_style(),
        yaxis=dict(
            title="Annual Δ spend ($)",
            tickvals=annual_ticks,
            ticktext=annual_text,
            range=[-(max_abs_annual + pad_annual), max_abs_annual + pad_annual],
            zeroline=False,
            showgrid=True,
        ),
        yaxis2=dict(
            title="Cumulative Δ spend ($)",
            tickvals=cumulative_ticks,
            ticktext=cumulative_text,
            range=[-(max_abs_cum + pad_cum), max_abs_cum + pad_cum],
            zeroline=False,
            showgrid=False,
            overlaying="y",
            side="right",
        ),
    )
    min_year = int(year_index.min())
    nonzero_mask = np.abs(delta_nzd) > 1e-9
    if nonzero_mask.any():
        max_year = int(year_index[nonzero_mask].max())
    else:
        max_year = int(year_index.max())
    fig.update_xaxes(
        title="Year",
        showgrid=True,
        zeroline=False,
        range=[min_year - 0.4, max_year + 0.4],
    )
    return fig


def _distribution_cost_chart(
    pivot_opt: Optional[pd.DataFrame],
    pivot_cmp: Optional[pd.DataFrame],
    *,
    breakdown_label: str,
    opt_label: str,
    cmp_label: str,
) -> Optional[go.Figure]:
    if pivot_opt is None or pivot_opt.empty:
        return None
    totals = pivot_opt.apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=0)
    totals = totals[totals.abs() > 1e-9]
    if totals.empty:
        return None

    categories = totals.sort_values(ascending=False).index.astype(str).tolist()
    values = totals.reindex(categories).fillna(0.0) * 1_000_000.0
    total_sum = float(values.sum())

    def _compact_amount_label(value: float) -> str:
        abs_value = abs(value)
        if abs_value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}b"
        if abs_value >= 1_000_000:
            return f"{value / 1_000_000:.2f}m"
        if abs_value >= 1_000:
            return f"{value / 1_000:.2f}k"
        return f"{value:.2f}"

    fig = go.Figure()
    share_labels = []
    for val in values.to_numpy(dtype=float):
        share = (val / total_sum * 100.0) if abs(total_sum) > 1e-9 else 0.0
        share_labels.append(f"{share:.1f}%")
    fig.add_trace(
        go.Bar(
            x=categories,
            y=values.to_numpy(dtype=float),
            name="Total cost",
            marker_color=PRIMARY_COLOR,
            customdata=[
                [_compact_amount_label(val), share_label]
                for val, share_label in zip(values.to_numpy(dtype=float), share_labels)
            ],
            hovertemplate=(
                "<b>%{x}</b><br>Total cost: %{customdata[0]}"
                "<br>% of total: %{customdata[1]}<extra></extra>"
            ),
        )
    )

    sample_values = values.to_numpy(dtype=float)
    tick_vals, tick_text, _ = compute_cash_axis_ticks(sample_values, force_unit="b")

    fig.update_layout(
        title=f"Distribution of Total Cost by {breakdown_label}",
        barmode="group",
        template=plotly_template(),
        showlegend=False,
        margin=dict(l=40, r=30, t=60, b=80),
        xaxis=dict(tickangle=-25, title=None),
        yaxis=dict(title="Total cost (NZD)", tickmode="array", tickvals=tick_vals, ticktext=tick_text),
        hoverlabel=dict(namelength=-1),
    )

    return fig


def _timing_shift_summary(
    pivot_opt: Optional[pd.DataFrame],
    pivot_cmp: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    centers_opt, totals_opt = _timing_center_from_pivot(pivot_opt)
    centers_cmp, totals_cmp = _timing_center_from_pivot(pivot_cmp)
    combined_totals = totals_opt.add(totals_cmp, fill_value=0.0)
    combined_totals = combined_totals[combined_totals.abs() > 1e-9]
    if combined_totals.empty:
        return None
    if totals_opt.abs().sum() > 1e-9:
        base_totals = totals_opt
    elif totals_cmp.abs().sum() > 1e-9:
        base_totals = totals_cmp
    else:
        base_totals = combined_totals
    base_totals = base_totals.reindex(combined_totals.index).fillna(0.0)
    total_sum = float(base_totals.sum())
    if total_sum <= 0:
        return None
    df = pd.DataFrame(
        {
            "Category": combined_totals.index.astype(str),
            "Total": base_totals.to_numpy(dtype=float),
            "SharePct": base_totals.to_numpy(dtype=float) / total_sum * 100.0,
            "CenterOpt": centers_opt.reindex(combined_totals.index).to_numpy(dtype=float),
            "CenterCmp": centers_cmp.reindex(combined_totals.index).to_numpy(dtype=float),
        }
    )
    df["Shift"] = df["CenterOpt"] - df["CenterCmp"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Shift"])
    if df.empty:
        return None
    return df


def _importance_bubble_chart(
    summary: pd.DataFrame,
    *,
    breakdown_label: str,
    opt_label: str,
    cmp_label: str,
) -> Optional[go.Figure]:
    if summary is None or summary.empty:
        return None
    df = summary.sort_values("Total", ascending=False).copy()
    sizes = df["Total"].to_numpy(dtype=float)
    max_size = float(np.nanmax(sizes)) if sizes.size else 0.0
    if max_size <= 0:
        return None
    desired_max = 60.0
    sizeref = 2.0 * max_size / (desired_max**2)
    show_text = len(df) <= 12
    mode = "markers+text" if show_text else "markers"
    text_values = df["Category"].astype(str).tolist() if show_text else None
    text_positions = "top center"
    text_color = "#E2E8F0" if is_dark_mode() else "#0F172A"

    fig = go.Figure(
        go.Scatter(
            x=df["Shift"].to_numpy(dtype=float),
            y=df["SharePct"].to_numpy(dtype=float),
            mode=mode,
            text=text_values,
            textposition=text_positions,
            textfont=dict(color=text_color, size=12),
            marker=dict(
                size=sizes,
                sizemode="area",
                sizeref=sizeref,
                sizemin=8,
                color=PRIMARY_COLOR,
                line=dict(color="#ffffff", width=1),
                opacity=0.78,
            ),
            customdata=np.stack([df["Category"], df["SharePct"], df["Total"]], axis=1),
            hovertemplate=(
                "<b>%{customdata[0]}</b>"
                "<br>Shift: %{x:+.1f}y"
                "<br>Spend share: %{customdata[1]:.1f}%"
                "<br>Total spend: %{customdata[2]:,.1f}m"
                "<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line=dict(color="rgba(15, 23, 42, 0.35)", width=1.2, dash="dash"))
    fig.update_layout(
        title=f"{breakdown_label} - bubble view (shift vs spend share; bubble size = total spend)",
        xaxis_title=f"Raw timing shift (years) [{opt_label} - {cmp_label}]",
        yaxis_title="Spend share (%)",
        template=plotly_template(),
        hoverlabel=dict(namelength=-1),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig


def _importance_rank_chart(
    summary: pd.DataFrame,
    *,
    breakdown_label: str,
    opt_label: str,
    cmp_label: str,
    category_order: Optional[List[str]] = None,
) -> Optional[go.Figure]:
    if summary is None or summary.empty:
        return None
    df = summary.copy()
    if category_order:
        order = [str(cat) for cat in category_order]
        existing = df["Category"].astype(str).tolist()
        remaining = [cat for cat in existing if cat not in order]
        order = order + remaining
        df["Category"] = df["Category"].astype(str)
        df["Category"] = pd.Categorical(df["Category"], categories=order, ordered=True)
        df = df.sort_values("Category")
        df["Category"] = df["Category"].astype(str)
    else:
        df["AbsShift"] = df["Shift"].abs()
        df = df.sort_values("AbsShift", ascending=False)
    labels = [f"{cat} ({share:.1f}%)" for cat, share in zip(df["Category"], df["SharePct"])]
    shifts = df["Shift"].to_numpy(dtype=float)
    fig = go.Figure(
        go.Bar(
            x=shifts,
            y=labels,
            orientation="h",
            marker_color=PRIMARY_COLOR,
            text=[f"{val:+.1f}y" for val in shifts],
            textposition="outside",
            cliponaxis=False,
            customdata=np.round(shifts, 2),
            hovertemplate="<b>%{y}</b><br>Shift: %{customdata:+.2f}y<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color="rgba(15, 23, 42, 0.35)", width=1.2, dash="dash"))
    fig.update_layout(
        title=f"{breakdown_label} - timing shift",
        xaxis_title=f"Raw timing shift (years) [{opt_label} - {cmp_label}]",
        yaxis_title="Category (share of total spend)",
        template=plotly_template(),
        hoverlabel=dict(namelength=-1),
        margin=dict(l=235, r=30, t=60, b=60),
    )
    fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1], ticklabelstandoff=8)
    return fig


def cumulative_benefits_vs_cost_chart(
    df: pd.DataFrame,
    selection: ScenarioSelection,
    *,
    title: str,
    bar_color: str = BRIGHT_PRIMARY_COLOR,
    line_color: str = COMPARISON_COLOR,
) -> go.Figure:
    """
    Show cumulative cost (stack-like bar per FY) vs cumulative benefits as a line.
   - Uses 'CumSpend' for cumulative costs.
    - Uses 'CumPVBenefit' when the selection confidence implies 'real', otherwise 'CumBenefit'.
    - Axis tick labels automatically switch between $m / $b.
    """
    fig = go.Figure()

    # --- Series (convert to nominal $) ---
    cum_cost = pd.to_numeric(df.get("CumSpend", 0.0), errors="coerce").fillna(0.0) * 1_000_000.0

    # Re-use existing helper to pick PV vs nominal series from confidence setting.
    benefit_series, benefit_label = _benefit_series_and_label(df, selection, prefix="")
    cum_benefit = pd.to_numeric(benefit_series, errors="coerce").fillna(0.0) * 1_000_000.0
    is_real = "PV" in (benefit_label or "")

    # --- Axis ticks ($m / $b) ---
    sample_values = np.concatenate([cum_cost.to_numpy(), cum_benefit.to_numpy()]) if len(df) else np.array([0.0])
    tick_vals, tick_text, unit_label = compute_cash_axis_ticks(sample_values, force_unit="b")

    # --- Traces ---
    # Bars = cumulative cost
    fig.add_trace(
        go.Bar(
            x=df["Year"],
            y=cum_cost,
            name="Cumulative cost",
            marker_color=bar_color,
            opacity=BAR_OPACITY,
            customdata=[format_large_amount(v) for v in cum_cost],
            hovertemplate="<b>Cumulative cost</b><br>FY %{x}: %{customdata}<extra></extra>",
        )
    )

    # Line = cumulative benefits
    benefit_display_name = (
        "Cumulative benefits (real $2025)" if is_real else "Cumulative benefits ($)"
    )
    fig.add_trace(
        go.Scatter(
            x=df["Year"],
            y=cum_benefit,
            name=benefit_display_name,
            mode="lines",
            line=dict(color=line_color, width=3.0),
            customdata=[format_large_amount(v) for v in cum_benefit],
            hovertemplate=f"<b>{benefit_display_name}</b><br>FY %{{x}}: %{{customdata}}<extra></extra>",
        )
    )

    fig.add_hline(y=0, line=dict(color="#888888", dash="dot", width=1))

    # --- Layout / styling ---
    yaxis_title_suffix = f"{unit_label}, real" if is_real else unit_label
    fig.update_layout(
        title=title,
        barmode="overlay",  # bars overlay; we're not stacking categories, just showing cumulative height
        legend=legend_bottom(),
        xaxis_title=None,
        yaxis_title=None,
        template=plotly_template(),
        yaxis=dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text, rangemode="tozero"),
        hoverlabel=dict(namelength=-1),
        margin=dict(l=40, r=40, t=60, b=60),
    )

    return fig



def benefit_chart(

    opt_df: Optional[pd.DataFrame],

    cmp_df: Optional[pd.DataFrame],

    *,

    dimension: str,

) -> go.Figure:

    fig = go.Figure()
    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()

    if opt_df is not None:

        series_opt = pd.to_numeric(opt_df["PVBenefit"], errors="coerce").fillna(0.0) * 1_000.0

        fig.add_trace(

            go.Scatter(

                x=opt_df["Year"],

                y=series_opt,

                name=f"{primary_label} Benefit Real",

                mode="lines",

                line=dict(color=CUMULATIVE_CMP_LINE_COLOR, width=3.2),

                opacity=0.85,

                customdata=[format_large_amount(val) for val in series_opt],

                hovertemplate=f"<b>{primary_label} Benefit Real</b><br>FY %{{x}}: %{{customdata}}<extra></extra>",

            )

        )

    if cmp_df is not None:

        series_cmp = pd.to_numeric(cmp_df["PVBenefit"], errors="coerce").fillna(0.0) * 1_000.0

        fig.add_trace(

            go.Scatter(

                x=cmp_df["Year"],

                y=series_cmp,

                name=f"{comparison_label} Benefit Real",

                mode="lines",

                line=dict(color=CUMULATIVE_CMP_LINE_COLOR, dash="dot", width=2.6),

                opacity=0.85,

                customdata=[format_large_amount(val) for val in series_cmp],

                hovertemplate=f"<b>{comparison_label} Benefit Real</b><br>FY %{{x}}: %{{customdata}}<extra></extra>",

            )

        )

    fig.update_layout(

        title=f"Real benefit to date ({dimension})",

        xaxis_title=None,

        legend=legend_bottom(),

        template=plotly_template(),

        hoverlabel=dict(namelength=-1),

        yaxis=dict(

            title=None,

            tickformat="~s",

        ),

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
    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()

    merged = opt_df[["Year", "CumBenefit", "CumPVBenefit"]].merge(

        cmp_df[["Year", "CumBenefit", "CumPVBenefit"]],

        on="Year",

        suffixes=("_opt", "_cmp"),

    )

    benefit_opt = pd.to_numeric(merged["CumBenefit_opt"], errors="coerce").fillna(0.0)
    benefit_cmp = pd.to_numeric(merged["CumBenefit_cmp"], errors="coerce").fillna(0.0)
    pv_opt = pd.to_numeric(merged["CumPVBenefit_opt"], errors="coerce").fillna(0.0)
    pv_cmp = pd.to_numeric(merged["CumPVBenefit_cmp"], errors="coerce").fillna(0.0)

    delta_benefit = (benefit_opt - benefit_cmp) * 1_000_000.0
    delta_pv = (pv_opt - pv_cmp) * 1_000_000.0

    fig.add_trace(

        go.Scatter(

            x=merged["Year"],

            y=delta_benefit,

            name="Delta Cumulative Benefit Real",

            mode="lines",

            line=dict(color=CUMULATIVE_OPT_LINE_COLOR, width=3.0),
            customdata=(delta_benefit / 1_000_000_000.0),
            hovertemplate="<b>Delta Cumulative Benefit Real</b><br>FY %{x}: %{customdata:,.1f}b<extra></extra>",



        )

    )

    fig.add_trace(

        go.Scatter(

            x=merged["Year"],

            y=delta_pv,

            name="Delta Benefit Real",

            mode="lines",

            line=dict(color=CUMULATIVE_CMP_LINE_COLOR, dash="dot", width=2.6),
            customdata=(delta_pv / 1_000_000_000.0),
            hovertemplate="<b>Delta Benefit Real</b><br>FY %{x}: %{customdata:,.1f}b<extra></extra>",



        )

    )

    title_text = "Cumulative Benefits Delta Real"

    fig.update_layout(

        title=title_text,

        xaxis_title=None,

        legend=legend_bottom(),

        template=plotly_template(),

        hoverlabel=dict(namelength=-1),

        yaxis=dict(

            title=None,

            showticklabels=False,

            showgrid=False,

            zeroline=False,

            showline=False,

        ),


    )

    return fig


def benefit_radar_chart(
    data,
    opt_selection,
    cmp_selection,
    *,
    horizon_years: Optional[int] = None,
) -> Optional[go.Figure]:

    # ---------------------------
    # Safe wrappers for app APIs
    # ---------------------------
    def pv_by_dimension_safe(d, sel):
        try:
            if not sel or not getattr(sel, "code", None):
                return None
            kwargs = {}
            if horizon_years is not None:
                kwargs["horizon_years"] = horizon_years
            return pv_by_dimension(d, sel, **kwargs)
        except NameError:
            return (getattr(sel, "pv_by_dim", None) or None) if sel else None

    def resolve_label_safe(sel, fallback):
        try:
            return resolve_selection_label(sel, fallback=fallback)
        except NameError:
            return getattr(sel, "label", None) or fallback

    def scenario_primary_label_safe():
        try:
            return scenario_primary_label()
        except NameError:
            return "Primary"

    def scenario_comparison_label_safe():
        try:
            return scenario_comparison_label()
        except NameError:
            return "Comparison"

    # ---------------------------
    # Theme / color helpers
    # ---------------------------
    def get_theme():
        # Defaults
        bg_light = "#FFFFFF"
        bg_dark = "#0E1117"
        text_light = "#1F2937"  # slate-800
        text_dark = "#E5E7EB"   # gray-200
        try:
            is_dark = is_dark_mode()
        except NameError:
            is_dark = False
        try:
            import streamlit as st  # type: ignore
            theme_base = (st.get_option("theme.base") or "").lower()
            bg = (st.get_option("theme.backgroundColor") or "").strip() or (bg_dark if theme_base == "dark" else bg_light)
            txt = (st.get_option("theme.textColor") or "").strip() or (text_dark if theme_base == "dark" else text_light)
            if theme_base == "dark":
                is_dark = True
        except Exception:
            bg = bg_dark if is_dark else bg_light
            txt = text_dark if is_dark else text_light
        grid = "rgba(148,163,184,0.33)" if is_dark else "rgba(100,116,139,0.28)"
        axis = "rgba(203,213,225,0.80)" if is_dark else "rgba(71,85,105,0.70)"
        return is_dark, bg, txt, grid, axis

    def hex_to_rgba(hex_color: str, alpha: float) -> str:
        hc = hex_color.strip().lstrip("#")
        if len(hc) == 3:
            hc = "".join([c * 2 for c in hc])
        r = int(hc[0:2], 16)
        g = int(hc[2:4], 16)
        b = int(hc[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    def nice_scale(max_value: float, target_steps: int = 5):
        if max_value <= 0:
            return 1.0, [0.2, 0.4, 0.6, 0.8, 1.0]
        raw = float(max_value)
        mag = 10 ** math.floor(math.log10(raw))
        residual = raw / mag
        if residual <= 1.2:
            nice = 1.2
        elif residual <= 1.5:
            nice = 1.5
        elif residual <= 2:
            nice = 2
        elif residual <= 2.5:
            nice = 2.5
        elif residual <= 3:
            nice = 3
        elif residual <= 4:
            nice = 4
        elif residual <= 5:
            nice = 5
        elif residual <= 6:
            nice = 6
        elif residual <= 7.5:
            nice = 7.5
        else:
            nice = 10
        upper = nice * mag
        # choose tick count close to target_steps
        best = None
        err = 1e9
        for n in [4, 5, 6]:
            this_err = abs(n - target_steps)
            if this_err < err:
                err = this_err
                best = n
        ticks = [upper * (i / best) for i in range(1, best + 1)]
        return upper, ticks

    # ---------------------------
    # Data prep
    # ---------------------------
    opt_pv = pv_by_dimension_safe(data, opt_selection)
    cmp_pv = pv_by_dimension_safe(data, cmp_selection)
    if not opt_pv and not cmp_pv:
        return None

    dims: List[str] = [str(dim) for dim in getattr(data, "dims", []) if str(dim).strip().lower() != "total"]
    extras = set()
    if opt_pv:
        extras.update(opt_pv.keys())
    if cmp_pv:
        extras.update(cmp_pv.keys())
    for dim in sorted(extras, key=str):
        s = str(dim)
        if s.strip().lower() == "total":
            continue
        if s not in dims:
            dims.append(s)
    if not dims:
        return None

    def to_values(pv: Optional[Dict[str, float]]) -> List[float]:
        return [float((pv or {}).get(d, 0.0)) for d in dims]

    labels = [str(d) for d in dims]
    n = len(labels)
    thetas = [(i / n) * 360.0 for i in range(n)]
    thetas_closed = thetas + thetas[:1]

    opt_vals = to_values(opt_pv)
    cmp_vals = to_values(cmp_pv)
    opt_closed = opt_vals + opt_vals[:1]
    cmp_closed = cmp_vals + cmp_vals[:1]

    # Colors (keep BLUE + GREEN)
    try:
        blue = CUMULATIVE_OPT_LINE_COLOR
    except NameError:
        blue = "#2563EB"   # blue-600
    try:
        green = CUMULATIVE_CMP_LINE_COLOR
    except NameError:
        green = "#10B981"  # emerald-500

    # Theme
    is_dark, bg_color, text_color, grid_color, axis_color = get_theme()

    # Scale
    max_val = max(opt_vals + cmp_vals + [1.0])
    radial_max, tick_vals = nice_scale(max_val, target_steps=5)
    radial_range = [0, radial_max * 1.18]  # baseline spacing with light padding

    # ---------------------------
    # Figure
    # ---------------------------
    fig = go.Figure()

    # Subtle alternating bands (very light so no "busy" look)
    band_fill = "rgba(148,163,184,0.06)" if is_dark else "rgba(100,116,139,0.05)"
    for i, r in enumerate(tick_vals):
        if i % 2 == 0:
            fig.add_trace(
                go.Scatterpolar(
                    r=[r] * (n + 1),
                    theta=thetas_closed,
                    fill="toself",
                    fillcolor=band_fill,
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
            )
        )

    # Primary (BLUE) â€” crisp, thin stroke; translucent fill
    if any(v > 0 for v in opt_vals):
        primary_label = resolve_label_safe(opt_selection, fallback=scenario_primary_label_safe())
        fig.add_trace(
            go.Scatterpolar(
                r=opt_closed,
                theta=thetas_closed,
                name=primary_label,
                mode="lines",
                line=dict(color=blue, width=2.25),  # thinner, sharper
                fill="toself",
                fillcolor=hex_to_rgba(blue, 0.16 if not is_dark else 0.22),
                hovertemplate="<b>%{customdata}</b><br>%{r:,.0f}<extra>%{fullData.name}</extra>",
                customdata=labels + labels[:1],
            )
        )
        # Vertex markers + small values (above)
        fig.add_trace(
            go.Scatterpolar(
                r=opt_vals,
                theta=thetas,
                mode="markers+text",
                marker=dict(size=7, color=blue, line=dict(width=1.1, color="white" if not is_dark else "#0B1220")),
                text=[f"{v:,.0f}" if v > 0 else "" for v in opt_vals],
                textfont=dict(size=14, color=blue),
                textposition="top center",
                hoverinfo="skip",
                name=f"{primary_label} values",
                showlegend=False,
                cliponaxis=False,
            )
        )

    # Comparison (GREEN) â€” crisp dashed stroke; lighter fill
    if any(v > 0 for v in cmp_vals):
        comparison_label = resolve_label_safe(cmp_selection, fallback=scenario_comparison_label_safe())
        fig.add_trace(
            go.Scatterpolar(
                r=cmp_closed,
                theta=thetas_closed,
                name=comparison_label,
                mode="lines",
                line=dict(color=green, width=2.0, dash="dot"),
                fill="toself",
                fillcolor=hex_to_rgba(green, 0.12 if not is_dark else 0.18),
                hovertemplate="<b>%{customdata}</b><br>%{r:,.0f}<extra>%{fullData.name}</extra>",
                customdata=labels + labels[:1],
            )
        )
        # Vertex markers + values (below)
        fig.add_trace(
            go.Scatterpolar(
                r=cmp_vals,
                theta=thetas,
                mode="markers+text",
                marker=dict(size=6.5, color=green, line=dict(width=1.0, color="white" if not is_dark else "#0B1220")),
                text=[f"{v:,.0f}" if v > 0 else "" for v in cmp_vals],
                textfont=dict(size=13, color=green),
                textposition="bottom center",
                hoverinfo="skip",
                name=f"{comparison_label} values",
                showlegend=False,
                cliponaxis=False,
            )
        )

    # Leader lines and dimension labels (custom, larger text)
    def label_position(angle_deg: float) -> str:
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        if cos_a >= 0.4:
            return "middle left"
        if cos_a <= -0.4:
            return "middle right"
        if sin_a >= 0:
            return "bottom center"
        return "top center"

    for theta_deg, label, opt_val, cmp_val in zip(thetas, labels, opt_vals, cmp_vals):
        anchor_value = max(opt_val, cmp_val, radial_max * 0.12)
        line_end = max(anchor_value * 1.08, radial_max * 0.92)
        line_end = min(line_end, radial_max * 1.05)
        text_radius = min(radial_max * 1.12, line_end * 1.04)
        fig.add_trace(
            go.Scatterpolar(
                r=[anchor_value, line_end, text_radius],
                theta=[theta_deg, theta_deg, theta_deg],
                mode="lines+text",
                line=dict(color=axis_color, width=1.4),
                text=["", "", label],
                textfont=dict(size=16, color=text_color, family="Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif"),
                textposition=label_position(theta_deg),
                hoverinfo="skip",
                showlegend=False,
                cliponaxis=False,
            )
        )

    # Axes & layout â€” stronger outer line, more top spacing to avoid overlap
    tick_texts = [
        f"{v:,.0f}" if idx >= max(len(tick_vals) - 3, 0) else ""
        for idx, v in enumerate(tick_vals)
    ]

    fig.update_layout(
        polar=dict(
            domain=dict(x=[0.0, 1.0], y=[0.12, 0.92]),  # more room for larger canvas
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                range=radial_range,
                tickmode="array",
                tickvals=tick_vals,
                ticktext=tick_texts,
                ticks="outside",
                tickfont=dict(size=11, color=text_color),
                angle=90,
                gridcolor=grid_color,
                gridwidth=0.9,
                showline=True,
                linecolor=axis_color,
                linewidth=2.4,  # thicker, single crisp outer ring
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=thetas,
                ticktext=[""] * len(thetas),  # hide default labels; custom leaders handle text
                direction="clockwise",
                rotation=90,
                tickfont=dict(size=12, color=text_color),
                gridcolor=grid_color,
                gridwidth=0.8,
                linecolor=axis_color,
                linewidth=2.0,
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=80, r=80, t=80, b=90),
        height=900,
        font=dict(color=text_color, size=13, family="Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.06,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            itemclick="toggleothers",
            itemdoubleclick="toggle",
            font=dict(size=12, color=text_color),
        ),
        showlegend=True,
    )

    fig.update_layout(
        title=dict(
            text="<b>Benefit Mix by Dimension</b>",
            x=0.5, xanchor="center",
            y=0.99, yanchor="top",
            font=dict(size=22, color=text_color),
        ),
    )

    return fig















def benefit_waterfall_chart(

    data: DashboardData,

    opt_selection: ScenarioSelection,

    cmp_selection: ScenarioSelection,

    *,

    horizon_years: Optional[int] = None,

) -> go.Figure:

    fig = go.Figure()

    try:
        import streamlit as st
    except Exception:
        apply_discount = False
    else:
        apply_discount = bool(st.session_state.get("npv_apply_discount", False))

    metric_token = "NPV" if apply_discount else "Benefit"
    metric_phrase = "NPV" if apply_discount else "benefit"

    if not opt_selection.code or not cmp_selection.code:

        fig.add_annotation(
            text=f"Select both scenarios to view the {metric_phrase} waterfall",
            showarrow=False,
        )

        fig.update_layout(template=plotly_template())

        return fig

    context_label = npv_context_label(
        data,
        opt_selection,
        cmp_selection,
        horizon_override=horizon_years,
        apply_discount=apply_discount,
    )
    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()

    pv_opt = pv_by_dimension(data, opt_selection, horizon_years=horizon_years)

    pv_cmp = pv_by_dimension(data, cmp_selection, horizon_years=horizon_years)

    if not pv_opt or not pv_cmp:

        fig.add_annotation(
            text=f"Dimension-level {metric_phrase} data unavailable for the selected scenarios",
            showarrow=False,
        )

        fig.update_layout(template=plotly_template())

        return fig

    dims = set(pv_opt.keys()) | set(pv_cmp.keys())

    ordered_non_total = [
        dim for dim in data.dims if dim in dims and str(dim).strip().lower() != "total"
    ]
    remaining_non_total = [
        dim
        for dim in dims
        if dim not in ordered_non_total and str(dim).strip().lower() != "total"
    ]

    dim_sequence = ordered_non_total + sorted(remaining_non_total, key=str)

    total_dim = next(
        (dim for dim in data.dims if dim in dims and str(dim).strip().lower() == "total"),
        None,
    )

    if total_dim is None:

        total_dim = next((dim for dim in dims if str(dim).lower() == "total"), None)

    delta_labels = [f"{dim} delta {metric_token}" for dim in dim_sequence]
    delta_values = [pv_opt.get(dim, 0.0) - pv_cmp.get(dim, 0.0) for dim in dim_sequence]

    net_delta_label = f"Net delta {metric_token}"
    net_delta_value = (
        pv_opt.get(total_dim, sum(pv_opt.values())) - pv_cmp.get(total_dim, sum(pv_cmp.values()))
        if total_dim
        else sum(delta_values)
    )

    waterfall_labels = delta_labels + [net_delta_label]
    measures = ["relative"] * len(delta_labels) + ["total"]
    values = delta_values + [net_delta_value]
    customdata = [[float(val)] for val in values]

    fig.add_trace(

        go.Waterfall(

            name=f"Delta {metric_token}",

            orientation="v",

            measure=measures,

            x=waterfall_labels,

            y=values,

            increasing={"marker": {"color": WATERFALL_GAIN_COLOR}},

            decreasing={"marker": {"color": WATERFALL_LOSS_COLOR}},

            totals={"marker": {"color": WATERFALL_TOTAL_COLOR}},

            connector={"line": {"color": "#BBBBBB", "width": 0.5}},

            customdata=customdata,
            hovertemplate="<b>%{x}</b><br>%{customdata[0]:,.1f}m<extra></extra>",
        )

    )

    fig.update_layout(

        title=f"Delta {context_label} by dimension - {primary_label} vs {comparison_label}",

        template=plotly_template(),
        hoverlabel=dict(namelength=-1),

        waterfallgap=0.3,

        height=WATERFALL_CHART_HEIGHT,

        yaxis=dict(
            title=context_label,
            showticklabels=False,
            ticks="",
        ),

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

    try:
        import streamlit as st
    except Exception:
        apply_discount = False
    else:
        apply_discount = bool(st.session_state.get("npv_apply_discount", False))

    metric_token = "NPV" if apply_discount else "Benefit"
    metric_phrase = "NPV" if apply_discount else "benefit"

    primary_label = scenario_primary_label()

    comparison_label = scenario_comparison_label()

    context_label = npv_context_label(

        data,

        opt_selection,

        cmp_selection,

        horizon_override=horizon_years,
        apply_discount=apply_discount,

    )

    if not opt_selection.code or not cmp_selection.code:

        fig.add_annotation(
            text=f"Select both scenarios to build the {metric_phrase} bridge",
            showarrow=False,
        )

        fig.update_layout(template=plotly_template())

        return fig

    pv_opt = pv_by_dimension(data, opt_selection, horizon_years=horizon_years)

    pv_cmp = pv_by_dimension(data, cmp_selection, horizon_years=horizon_years)

    if not pv_opt or not pv_cmp:

        fig.add_annotation(
            text=f"Missing dimension {metric_phrase} data for bridge",
            showarrow=False,
        )

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

    bridge_labels = [f"{dim} delta {metric_token}" for dim in dim_sequence]
    primary_internal = f"{primary_label} {metric_token} total"
    comparison_internal = f"{comparison_label} {metric_token} total"
    tickvals: Optional[List[str]] = None
    ticktext: Optional[List[str]] = None
    if primary_label == comparison_label:
        primary_internal = f"{primary_label} {metric_token} total (1)"
        comparison_internal = f"{comparison_label} {metric_token} total (2)"
        tickvals = [primary_internal, *bridge_labels, comparison_internal]
        ticktext = [
            f"{primary_label} {metric_token} total",
            *bridge_labels,
            f"{comparison_label} {metric_token} total",
        ]
    labels = [primary_internal] + bridge_labels + [comparison_internal]

    measures = ["absolute"] + ["relative"] * len(dim_sequence) + ["total"]

    values = [total_opt] + [-delta for delta in bridge_diffs] + [total_cmp]

    hovertexts = [
        f"{primary_label} {metric_token} total: {total_opt:,.0f} m",
        *[
            f"{dim} delta {metric_token}: {delta:,.0f} m"
            for dim, delta in zip(dim_sequence, bridge_diffs)
        ],
        f"{comparison_label} {metric_token} total: {total_cmp:,.0f} m",
    ]

    fig.add_trace(

        go.Waterfall(

            name=f"{metric_token} bridge",

            orientation="v",

            measure=measures,

            x=labels,

            y=values,

            increasing=dict(marker=dict(color=BRIGHT_PRIMARY_COLOR)),

            decreasing=dict(marker=dict(color=WATERFALL_LOSS_COLOR)),

            totals=dict(marker=dict(color=WATERFALL_TOTAL_COLOR)),

            connector=dict(line=dict(color="#BBBBBB", width=0.5)),

            hovertext=hovertexts,

            hovertemplate="%{hovertext}<extra></extra>",

        )

    )

    fig.update_layout(

        title=f"Bridge {context_label} - {primary_label} to {comparison_label}",

        template=plotly_template(),

        waterfallgap=0.3,
        height=WATERFALL_CHART_HEIGHT,
        yaxis=dict(
            title=context_label,
            showticklabels=False,
            ticks="",
        ),
        xaxis=dict(
            categoryorder="array",
            categoryarray=tickvals or labels,
            **({"tickmode": "array", "tickvals": tickvals, "ticktext": ticktext} if tickvals else {}),
        ),
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
    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()

    cost_basis = _resolve_value_basis(
        "npv_cost_value_basis",
        "npv_apply_cost_discount",
        default=VALUE_BASIS_PV,
    )
    benefit_basis = _resolve_value_basis(
        "npv_benefit_value_basis",
        "npv_apply_discount",
        default=VALUE_BASIS_PV,
    )
    try:
        pv_horizon_years = int(st.session_state.get("npv_horizon_selection", 60))
    except (TypeError, ValueError):
        pv_horizon_years = 60

    def _start_year() -> int:
        for frame in (opt_df, cmp_df):
            if frame is None or frame.empty:
                continue
            years_raw = pd.to_numeric(frame.get("Year"), errors="coerce").dropna()
            if not years_raw.empty:
                return int(years_raw.min())
        return 2025

    start_year = _start_year()

    def _basis_descriptor(basis: str) -> str:
        basis_norm = (basis or "").strip().lower()
        if basis_norm.startswith("pv"):
            return "PV"
        if basis_norm.startswith("nom"):
            return "nominal"
        return "real $2025"

    def _scale_to_dollars(values: pd.Series) -> float:
        values = pd.to_numeric(values, errors="coerce").fillna(0.0)
        magnitude = float(values.abs().max()) if not values.empty else 0.0
        if magnitude >= 10_000_000.0:
            return 1.0
        return 1_000_000.0

    def _cumulative_flow(
        frame: pd.DataFrame,
        *,
        flow_column: str,
        basis: str,
        horizon_years: Optional[int],
    ) -> Tuple[pd.Series, pd.Series]:
        years = pd.to_numeric(frame.get("Year"), errors="coerce").fillna(start_year).astype(int)
        flows = pd.to_numeric(frame.get(flow_column), errors="coerce").fillna(0.0).astype(float)
        if basis == VALUE_BASIS_PV and horizon_years:
            limit_year = int(start_year) + int(horizon_years) - 1
            flows = flows.where(years <= limit_year, 0.0)
        multipliers = value_basis_multiplier(int(start_year), years.to_numpy(dtype=int), basis)
        adjusted = flows.to_numpy(dtype=float) * multipliers
        cumulative = pd.Series(np.cumsum(adjusted), index=frame.index, dtype=float)
        return years, cumulative

    axis_samples: List[np.ndarray] = []

    series_opt = label_opt = None

    if opt_df is not None and not opt_df.empty:
        cost_years, cost_cum = _cumulative_flow(
            opt_df,
            flow_column="Spend",
            basis=cost_basis,
            horizon_years=pv_horizon_years if cost_basis == VALUE_BASIS_PV else None,
        )
        spend_scale = _scale_to_dollars(pd.to_numeric(opt_df.get("Spend"), errors="coerce").fillna(0.0))
        spend_values = cost_cum * spend_scale
        axis_samples.append(spend_values.to_numpy(dtype=float))

        fig.add_trace(

            go.Bar(

                x=cost_years,

                y=spend_values,

                name=f"{primary_label} cumulative {_basis_descriptor(cost_basis)} spend",

                marker_color=BRIGHT_PRIMARY_COLOR,

                opacity=1.0,

                customdata=spend_values / 1_000_000_000.0,

                hovertemplate=f"<b>{primary_label} cumulative {_basis_descriptor(cost_basis)} spend</b><br>FY %{{x}}: %{{customdata:,.1f}}b<extra></extra>",

            )

        )

        benefit_flow_col = "BenefitFlowTotal" if "BenefitFlowTotal" in opt_df.columns else "BenefitFlow"
        benefit_years_opt, benefit_cum_opt = _cumulative_flow(
            opt_df,
            flow_column=benefit_flow_col,
            basis=benefit_basis,
            horizon_years=pv_horizon_years if benefit_basis == VALUE_BASIS_PV else None,
        )
        benefit_scale = _scale_to_dollars(
            pd.to_numeric(opt_df.get(benefit_flow_col), errors="coerce").fillna(0.0)
        )
        series_opt = benefit_cum_opt * benefit_scale
        label_opt = f"{primary_label} cumulative {_basis_descriptor(benefit_basis)} benefit"
        axis_samples.append(series_opt.to_numpy(dtype=float))

    series_cmp = label_cmp = None

    if cmp_df is not None and not cmp_df.empty:

        benefit_flow_col = "BenefitFlowTotal" if "BenefitFlowTotal" in cmp_df.columns else "BenefitFlow"
        benefit_years_cmp, benefit_cum_cmp = _cumulative_flow(
            cmp_df,
            flow_column=benefit_flow_col,
            basis=benefit_basis,
            horizon_years=pv_horizon_years if benefit_basis == VALUE_BASIS_PV else None,
        )
        benefit_scale = _scale_to_dollars(
            pd.to_numeric(cmp_df.get(benefit_flow_col), errors="coerce").fillna(0.0)
        )
        series_cmp = benefit_cum_cmp * benefit_scale
        label_cmp = f"{comparison_label} cumulative {_basis_descriptor(benefit_basis)} benefit"
        axis_samples.append(series_cmp.to_numpy(dtype=float))

        fig.add_trace(

            go.Scatter(

                x=benefit_years_cmp,

                y=series_cmp,

                name=label_cmp,

                mode="lines",

                line=dict(color=CUMULATIVE_CMP_LINE_COLOR, width=2.6, dash="dash"),

                customdata=np.asarray(series_cmp, dtype=float) / 1_000_000_000.0,

                hovertemplate=f"<b>{label_cmp}</b><br>FY %{{x}}: %{{customdata:,.1f}}b<extra></extra>",

            )

        )

    if series_opt is not None:

        fill_kwargs = {}

        if series_cmp is not None:

            fill_kwargs = dict(

                fill="tonexty",

                fillcolor=rgba_from_hex(CUMULATIVE_CMP_LINE_COLOR, 0.22),

            )

        fig.add_trace(

            go.Scatter(

                x=benefit_years_opt,

                y=series_opt,

                name=label_opt,

                mode="lines",

                line=dict(color=CUMULATIVE_CMP_LINE_COLOR, width=3.2),

                customdata=np.asarray(series_opt, dtype=float) / 1_000_000_000.0,

                hovertemplate=f"<b>{label_opt}</b><br>FY %{{x}}: %{{customdata:,.1f}}b<extra></extra>",

                **fill_kwargs,

            )

        )

    fig.update_layout(

        title="Cumulative spend vs benefit",

        xaxis_title=None,

        template=plotly_template(),

        hoverlabel=dict(namelength=-1),

        legend=legend_bottom(),

    )

    if axis_samples:
        combined = np.concatenate(axis_samples) if axis_samples else np.array([0.0])
        tick_vals, tick_text, _ = compute_cash_axis_ticks(combined, force_unit="b")
        fig.update_yaxes(

            title=None,

            tickmode="array",

            tickvals=tick_vals,

            ticktext=tick_text,

            rangemode="tozero",

        )

    return fig


def _bcr_start_year(opt_df: Optional[pd.DataFrame], cmp_df: Optional[pd.DataFrame]) -> int:
    for frame in (opt_df, cmp_df):
        if frame is None or frame.empty:
            continue
        years_raw = pd.to_numeric(frame.get("Year"), errors="coerce").dropna()
        if not years_raw.empty:
            return int(years_raw.min())
    return 2025


def _bcr_series_from_frame(
    frame: Optional[pd.DataFrame],
    *,
    start_year: int,
    pv_horizon_years: Optional[int],
) -> Optional[pd.Series]:
    if frame is None or frame.empty:
        return None
    years = pd.to_numeric(frame.get("Year"), errors="coerce")
    valid_mask = years.notna()
    if not valid_mask.any():
        return None
    years = years[valid_mask].astype(int)
    spend = pd.to_numeric(frame.get("Spend"), errors="coerce").fillna(0.0)
    spend = spend[valid_mask].astype(float)
    benefit_col = "BenefitFlowTotal" if "BenefitFlowTotal" in frame.columns else "BenefitFlow"
    benefits = pd.to_numeric(frame.get(benefit_col), errors="coerce").fillna(0.0)
    benefits = benefits[valid_mask].astype(float)
    if pv_horizon_years:
        limit_year = int(start_year) + int(pv_horizon_years) - 1
        in_window = years <= limit_year
        spend = spend.where(in_window, 0.0)
        benefits = benefits.where(in_window, 0.0)
    multipliers = value_basis_multiplier(int(start_year), years.to_numpy(dtype=int), VALUE_BASIS_PV)
    spend_pv = spend.to_numpy(dtype=float) * multipliers
    benefit_pv = benefits.to_numpy(dtype=float) * multipliers
    cum_spend = np.cumsum(spend_pv)
    cum_benefit = np.cumsum(benefit_pv)
    bcr = np.divide(
        cum_benefit,
        cum_spend,
        out=np.full_like(cum_benefit, np.nan, dtype=float),
        where=cum_spend != 0,
    )
    return pd.Series(bcr, index=years.to_numpy(dtype=int))


def bcr_time_series_chart(
    opt_df: Optional[pd.DataFrame],
    cmp_df: Optional[pd.DataFrame],
    opt_selection: ScenarioSelection,
    cmp_selection: ScenarioSelection,
) -> Optional[go.Figure]:
    if (opt_df is None or opt_df.empty) and (cmp_df is None or cmp_df.empty):
        return None

    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()

    start_year = _bcr_start_year(opt_df, cmp_df)
    try:
        pv_horizon_years = int(st.session_state.get("npv_horizon_selection", 60))
    except (TypeError, ValueError):
        pv_horizon_years = 60

    series_opt = _bcr_series_from_frame(opt_df, start_year=start_year, pv_horizon_years=pv_horizon_years)
    series_cmp = _bcr_series_from_frame(cmp_df, start_year=start_year, pv_horizon_years=pv_horizon_years)
    if series_opt is None and series_cmp is None:
        return None

    years_index = pd.Index([], dtype=int)
    for series in (series_opt, series_cmp):
        if series is not None:
            years_index = years_index.union(series.index)
    years_index = years_index.sort_values()

    opt_aligned = series_opt.reindex(years_index) if series_opt is not None else None
    cmp_aligned = series_cmp.reindex(years_index) if series_cmp is not None else None

    opt_values = opt_aligned.to_numpy(dtype=float) if opt_aligned is not None else np.array([])
    cmp_values = cmp_aligned.to_numpy(dtype=float) if cmp_aligned is not None else np.array([])

    def _efficiency_sentence(diff_pct: Optional[float]) -> str:
        if diff_pct is None or not np.isfinite(diff_pct):
            return "Efficiency comparison unavailable"
        if math.isclose(diff_pct, 0.0, abs_tol=1e-6):
            diff_pct = 0.0
            direction = "more"
            arrow = ""
        else:
            direction = "more" if diff_pct >= 0 else "less"
            arrow = "&#9650;" if diff_pct >= 0 else "&#9660;"
        return (
            f"{primary_label} is {arrow} {abs(diff_pct):.1f}% {direction} "
            f"efficient than {comparison_label}"
        )

    efficiency_messages: List[str] = []
    if opt_aligned is not None and cmp_aligned is not None:
        for opt_val, cmp_val in zip(opt_values, cmp_values):
            if cmp_val and np.isfinite(opt_val) and np.isfinite(cmp_val):
                diff_pct = ((opt_val / cmp_val) - 1.0) * 100.0
            else:
                diff_pct = None
            efficiency_messages.append(_efficiency_sentence(diff_pct))
    else:
        efficiency_messages = [_efficiency_sentence(None) for _ in range(len(years_index))]

    fig = go.Figure()
    if opt_aligned is not None:
        fig.add_trace(
            go.Scatter(
                x=years_index,
                y=opt_values,
                name=primary_label,
                mode="lines",
                line=dict(color=PRIMARY_COLOR, width=2.8),
                customdata=efficiency_messages,
                hovertemplate=(
                    "<span style='color:#64748B;'>FY %{x}</span>"
                    "<br><span style='display:inline-block;padding-top:4px;font-weight:400;font-size:1.02rem;'>"
                    "%{y:.2f}x</span>"
                    "<br><span style='display:inline-block;padding-top:8px;'>%{customdata}</span>"
                    "<extra></extra>"
                ),
            )
        )
    if cmp_aligned is not None:
        fig.add_trace(
            go.Scatter(
                x=years_index,
                y=cmp_values,
                name=comparison_label,
                mode="lines",
                line=dict(color=CAPACITY_HEAT_RED, width=2.2, dash="dash"),
                customdata=efficiency_messages,
                hovertemplate=(
                    "<span style='color:#64748B;'>FY %{x}</span>"
                    "<br><span style='display:inline-block;padding-top:4px;font-weight:400;font-size:1.02rem;'>"
                    "%{y:.2f}x</span>"
                    "<br><span style='display:inline-block;padding-top:8px;'>%{customdata}</span>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="BCR - Portfolio Weighted Average",
        xaxis_title=None,
        yaxis=dict(title=None, tickformat=".2f", rangemode="tozero"),
        template=plotly_template(),
        hoverlabel=dict(namelength=-1),
        legend=legend_bottom(),
    )

    return fig


def _spend_weighted_bcr_series(
    data: DashboardData,
    selection: ScenarioSelection,
) -> Optional[pd.Series]:
    if data is None or selection is None or not getattr(selection, "code", None):
        return None
    spend_matrix = getattr(data, "spend_matrix", None)
    if not isinstance(spend_matrix, pd.DataFrame) or spend_matrix.empty:
        return None
    subset = spend_matrix[spend_matrix["Code"] == selection.code].copy()
    if subset.empty:
        return None
    years = [int(y) for y in getattr(data, "years", []) if y is not None]
    year_cols = [int(year) for year in years if int(year) in subset.columns]
    if not year_cols:
        return None

    cost_table = _scenario_project_cost_table(data, selection.code, years)
    benefit_table = _scenario_project_benefit_table(data, selection.code, years)
    base_year = int(getattr(data, "start_fy", 2025) or 2025)
    new_bcr = _project_new_bcr_from_tables(
        cost_table,
        benefit_table,
        base_year=base_year,
    )
    if new_bcr is None or new_bcr.empty:
        return None

    subset = subset[["Project"] + year_cols].copy()
    subset["Project"] = subset["Project"].astype(str)
    subset["ProjectKey"] = subset["Project"].map(_normalise_project_key)
    merged = subset.merge(new_bcr, on="ProjectKey", how="left")
    bcr_values = pd.to_numeric(merged.get("BCR (New)"), errors="coerce")
    if bcr_values.isna().all():
        return None

    spend_matrix_vals = (
        merged[year_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    )
    bcr_array = bcr_values.to_numpy(dtype=float)

    series_values: List[float] = []
    for col_idx in range(spend_matrix_vals.shape[1]):
        spend_vals = spend_matrix_vals[:, col_idx]
        mask = np.isfinite(bcr_array) & (spend_vals > 1e-9)
        if not mask.any():
            series_values.append(np.nan)
            continue
        total_spend = float(spend_vals[mask].sum())
        if total_spend <= 0:
            series_values.append(np.nan)
            continue
        weighted = float((spend_vals[mask] * bcr_array[mask]).sum() / total_spend)
        series_values.append(weighted)

    return pd.Series(series_values, index=np.asarray(year_cols, dtype=int), dtype=float)


def _spend_weighted_bcr_time_series_chart(
    data: DashboardData,
    opt_selection: ScenarioSelection,
    cmp_selection: ScenarioSelection,
) -> Optional[go.Figure]:
    if data is None:
        return None

    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()

    series_opt = _spend_weighted_bcr_series(data, opt_selection)
    series_cmp = _spend_weighted_bcr_series(data, cmp_selection)
    if series_opt is None and series_cmp is None:
        return None

    years_index = pd.Index([], dtype=int)
    for series in (series_opt, series_cmp):
        if series is not None:
            years_index = years_index.union(series.index)
    years_index = years_index.sort_values()

    opt_aligned = series_opt.reindex(years_index) if series_opt is not None else None
    cmp_aligned = series_cmp.reindex(years_index) if series_cmp is not None else None

    opt_values = opt_aligned.to_numpy(dtype=float) if opt_aligned is not None else np.array([])
    cmp_values = cmp_aligned.to_numpy(dtype=float) if cmp_aligned is not None else np.array([])

    fig = go.Figure()
    if opt_aligned is not None:
        fig.add_trace(
            go.Scatter(
                x=years_index,
                y=opt_values,
                name=primary_label,
                mode="lines",
                line=dict(color=PRIMARY_COLOR, width=2.8),
                hovertemplate=(
                    "<span style='color:#64748B;'>FY %{x}</span>"
                    "<br><span style='display:inline-block;padding-top:4px;font-weight:400;font-size:1.02rem;'>"
                    "%{y:.2f}x</span>"
                    "<extra></extra>"
                ),
            )
        )
    if cmp_aligned is not None:
        fig.add_trace(
            go.Scatter(
                x=years_index,
                y=cmp_values,
                name=comparison_label,
                mode="lines",
                line=dict(color=CAPACITY_HEAT_RED, width=2.2, dash="dash"),
                hovertemplate=(
                    "<span style='color:#64748B;'>FY %{x}</span>"
                    "<br><span style='display:inline-block;padding-top:4px;font-weight:400;font-size:1.02rem;'>"
                    "%{y:.2f}x</span>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="BCR - Spend-weighted project average",
        xaxis_title=None,
        yaxis=dict(title=None, tickformat=".2f", rangemode="tozero"),
        template=plotly_template(),
        hoverlabel=dict(namelength=-1),
        legend=legend_bottom(),
    )
    def _last_valid_year(series: Optional[pd.Series]) -> Optional[int]:
        if series is None or series.empty:
            return None
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(values)
        if not valid_mask.any():
            return None
        nonzero_mask = valid_mask & (np.abs(values) > 1e-9)
        years = pd.Index(series.index).astype(int)
        if nonzero_mask.any():
            return int(years[nonzero_mask].max())
        return int(years[valid_mask].max())

    min_year = int(years_index.min()) if len(years_index) else None
    max_candidates = [
        year for year in (_last_valid_year(opt_aligned), _last_valid_year(cmp_aligned)) if year is not None
    ]
    max_year = max(max_candidates) if max_candidates else (int(years_index.max()) if len(years_index) else None)
    if min_year is not None and max_year is not None and max_year >= min_year:
        fig.update_xaxes(range=[min_year - 0.4, max_year + 0.4])
    return fig


def prepare_spend_weighted_bcr_export(
    data: DashboardData,
    opt_selection: ScenarioSelection,
    cmp_selection: ScenarioSelection,
) -> Optional[pd.DataFrame]:
    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()
    series_opt = _spend_weighted_bcr_series(data, opt_selection)
    series_cmp = _spend_weighted_bcr_series(data, cmp_selection)
    if series_opt is None and series_cmp is None:
        return None
    years_index = pd.Index([], dtype=int)
    for series in (series_opt, series_cmp):
        if series is not None:
            years_index = years_index.union(series.index)
    years_index = years_index.sort_values()
    export = pd.DataFrame({"Financial year": years_index.astype(int)})
    if series_opt is not None:
        export[f"{primary_label} spend-weighted BCR (New)"] = series_opt.reindex(years_index).to_numpy(dtype=float)
    if series_cmp is not None:
        export[f"{comparison_label} spend-weighted BCR (New)"] = series_cmp.reindex(years_index).to_numpy(dtype=float)
    if export.shape[1] <= 1:
        return None
    return export

COST_BENEFIT_STACK_COST_OPTIONS = (
    "Activity Class",
    "Region",
    "GPS Request Tier",
)

CASH_FLOW_PROFILE_BREAKDOWN_OPTIONS = (
    "Total",
    "Activity Class",
    "Region",
    "GPS Request Tier",
)

COST_BENEFIT_STACK_BENEFIT_OPTIONS = (
    "Strategic Dimension",
    "Region",
    "Activity Class",
)


def _sample_stack_palette(name: str, n: int, *, start: float, end: float) -> List[str]:
    if n <= 0:
        return []
    if n == 1:
        positions = [(start + end) / 2.0]
    else:
        positions = np.linspace(start, end, n).tolist()
    return plc.sample_colorscale(name, positions, colortype="rgb")


def _discounted_table_totals(
    table: pd.DataFrame,
    years: List[int],
    *,
    start_year: int,
    value_basis: str,
) -> pd.Series:
    if table.empty or "Project" not in table.columns:
        return pd.Series(dtype=float)
    safe_years = [int(y) for y in years if y in table.columns]
    if not safe_years:
        return pd.Series(dtype=float)
    values = table[safe_years].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    multipliers = value_basis_multiplier(int(start_year), np.asarray(safe_years, dtype=int), value_basis)
    totals = (values * multipliers).sum(axis=1)
    return pd.Series(totals, index=table["Project"].astype(str).to_numpy())


def _group_totals_by_mapping(
    project_totals: pd.Series,
    mapping: pd.DataFrame,
    *,
    group_col: str,
    preferred_order: Optional[List[str]] = None,
) -> pd.Series:
    if project_totals.empty:
        return pd.Series(dtype=float)
    working = pd.DataFrame({"Project": project_totals.index, "Value": project_totals.to_numpy(dtype=float)})
    working = working[working["Value"].abs() > 1e-9]
    if working.empty:
        return pd.Series(dtype=float)
    working["ProjectKey"] = working["Project"].map(_normalise_project_key)
    merged = working.merge(mapping[["ProjectKey", group_col]], on="ProjectKey", how="left")
    merged[group_col] = merged[group_col].fillna("Unknown").astype(str)
    if group_col == "GPSTier":
        merged[group_col] = merged[group_col].map(_canonicalise_gps_tier)
    grouped = merged.groupby(group_col, dropna=False)["Value"].sum().sort_values(ascending=False)
    grouped = grouped[grouped.abs() > 1e-9]
    if preferred_order:
        lookup = {str(label).strip().lower(): label for label in grouped.index}
        ordered: List[str] = []
        for label in preferred_order:
            key = str(label).strip().lower()
            match = lookup.get(key)
            if match is not None and match not in ordered:
                ordered.append(match)
        extras = [label for label in grouped.index if label not in ordered]
        grouped = grouped.reindex(ordered + extras)
    return grouped


def _compute_cost_benefit_stack_series(
    data: DashboardData,
    selection: ScenarioSelection,
    *,
    pv_horizon_years: Optional[int],
    cost_breakdown: str,
    benefit_breakdown: str,
    cost_value_basis: str,
    benefit_value_basis: str,
    compare_selection: Optional[ScenarioSelection] = None,
) -> Optional[Tuple[pd.Series, pd.Series, Optional[str], int]]:
    if not selection or not selection.code:
        return None

    compare_code = getattr(compare_selection, "code", None) if compare_selection is not None else None
    if compare_code == selection.code:
        compare_code = None

    start_year = int(getattr(data, "start_fy", 0) or 0)
    model_years = int(getattr(data, "model_years", 0) or 0) or 60
    years_full = [
        int(y)
        for y in getattr(data, "years", [])
        if y is not None and start_year <= int(y) <= (start_year + model_years - 1)
    ]
    try:
        pv_horizon_years_int = int(pv_horizon_years) if pv_horizon_years is not None else model_years
    except (TypeError, ValueError):
        pv_horizon_years_int = model_years
    if pv_horizon_years_int <= 0:
        pv_horizon_years_int = model_years
    pv_end_year = start_year + pv_horizon_years_int - 1
    years_pv = [year for year in years_full if year <= pv_end_year]
    cost_years = years_pv if cost_value_basis == VALUE_BASIS_PV else years_full
    benefit_years = years_pv if benefit_value_basis == VALUE_BASIS_PV else years_full

    mapping = load_project_attribute_mapping()

    def _benefit_scale_to_dollars(selection_value: ScenarioSelection, series: pd.Series) -> float:
        if series is None or series.empty:
            return 1.0
        pv_total_raw: Optional[float] = None
        raw = _raw_result_for_code(data, getattr(selection_value, "code", None))
        if isinstance(raw, dict):
            pv_total_raw = raw.get("benefit_pv_total")
            if pv_total_raw is None:
                pv_total_raw = raw.get("pv_total")
        candidates: List[float] = []
        for candidate in (pv_total_raw, series.sum()):
            if candidate is None:
                continue
            try:
                candidates.append(abs(float(candidate)))
            except (TypeError, ValueError):
                continue
        magnitude = max(candidates) if candidates else 0.0
        if magnitude <= 0:
            return 1.0
        return 1_000_000.0 if magnitude < 10_000_000.0 else 1.0

    cost_group_col = "ActivityClass"
    cost_order: Optional[List[str]] = None
    if cost_breakdown == "Region":
        cost_group_col = "Region"
    elif cost_breakdown == "GPS Request Tier":
        cost_group_col = "GPSTier"
        cost_order = ["Must do", "Should do", "Could do", "Unknown"]

    def _cost_series_for(code: str) -> pd.Series:
        cost_table = _scenario_project_cost_table(data, code, cost_years)
        if cost_table is None:
            cost_table = pd.DataFrame()
        cost_project_totals = _discounted_table_totals(
            cost_table,
            cost_years,
            start_year=start_year,
            value_basis=cost_value_basis,
        )
        series = _group_totals_by_mapping(
            cost_project_totals,
            mapping,
            group_col=cost_group_col,
            preferred_order=cost_order,
        )
        return series.astype(float)

    def _benefit_series_for(selection_value: ScenarioSelection) -> pd.Series:
        if benefit_breakdown == "Strategic Dimension":
            pv_map = pv_by_dimension(
                data,
                selection_value,
                horizon_years=pv_horizon_years_int if benefit_value_basis == VALUE_BASIS_PV else None,
                value_basis=benefit_value_basis,
            ) or {}
            series = pd.Series(pv_map, dtype=float)
            series.index = series.index.astype(str)
            series = series[series.index.str.strip().str.lower() != "total"]
            series = series[series.abs() > 1e-9]
            return series.astype(float)

        benefit_group_col = "Region" if benefit_breakdown == "Region" else "ActivityClass"
        benefit_table = _scenario_project_benefit_table(data, selection_value.code, benefit_years)
        if benefit_table is None:
            benefit_table = pd.DataFrame()
        benefit_project_totals = _discounted_table_totals(
            benefit_table,
            benefit_years,
            start_year=start_year,
            value_basis=benefit_value_basis,
        )
        series = _group_totals_by_mapping(
            benefit_project_totals,
            mapping,
            group_col=benefit_group_col,
        )
        return series.astype(float)

    cost_series = _cost_series_for(selection.code)
    benefit_series = _benefit_series_for(selection)
    benefit_series = benefit_series * _benefit_scale_to_dollars(selection, benefit_series)

    if compare_code and compare_selection is not None and compare_selection.code:
        cost_series_cmp = _cost_series_for(compare_selection.code)
        benefit_series_cmp = _benefit_series_for(compare_selection)
        benefit_series_cmp = benefit_series_cmp * _benefit_scale_to_dollars(compare_selection, benefit_series_cmp)

        cost_index = cost_series.index.union(cost_series_cmp.index)
        benefit_index = benefit_series.index.union(benefit_series_cmp.index)
        cost_series = cost_series.reindex(cost_index, fill_value=0.0) - cost_series_cmp.reindex(
            cost_index, fill_value=0.0
        )
        benefit_series = benefit_series.reindex(benefit_index, fill_value=0.0) - benefit_series_cmp.reindex(
            benefit_index, fill_value=0.0
        )
        cost_series = cost_series[cost_series.abs() > 1e-9]
        benefit_series = benefit_series[benefit_series.abs() > 1e-9]

        if not cost_series.empty:
            cost_series = cost_series.reindex(cost_series.abs().sort_values(ascending=False).index)
        if not benefit_series.empty:
            benefit_series = benefit_series.reindex(benefit_series.abs().sort_values(ascending=False).index)
    else:
        if not cost_series.empty:
            cost_series = cost_series.sort_values(ascending=False)
        if not benefit_series.empty:
            benefit_series = benefit_series.sort_values(ascending=False)

    return cost_series, benefit_series, compare_code, pv_horizon_years_int


def cost_benefit_stack_chart(
    data: DashboardData,
    selection: ScenarioSelection,
    *,
    selection_label: str,
    pv_horizon_years: Optional[int],
    cost_breakdown: str,
    benefit_breakdown: str,
    cost_value_basis: str,
    benefit_value_basis: str,
    compare_selection: Optional[ScenarioSelection] = None,
) -> Optional[Tuple[go.Figure, List[Tuple[str, str]], List[Tuple[str, str]]]]:
    series_result = _compute_cost_benefit_stack_series(
        data,
        selection,
        pv_horizon_years=pv_horizon_years,
        cost_breakdown=cost_breakdown,
        benefit_breakdown=benefit_breakdown,
        cost_value_basis=cost_value_basis,
        benefit_value_basis=benefit_value_basis,
        compare_selection=compare_selection,
    )
    if series_result is None:
        return None

    cost_series, benefit_series, compare_code, pv_horizon_years_int = series_result

    cost_total_dollars = float(cost_series.sum() * 1_000_000.0) if not cost_series.empty else 0.0
    benefit_total_dollars = float(benefit_series.sum()) if not benefit_series.empty else 0.0
    cost_abs_total_dollars = float(cost_series.abs().sum() * 1_000_000.0) if not cost_series.empty else 0.0
    benefit_abs_total_dollars = float(benefit_series.abs().sum()) if not benefit_series.empty else 0.0
    cost_positive_dollars = 0.0
    cost_negative_dollars = 0.0
    benefit_positive_dollars = 0.0
    benefit_negative_dollars = 0.0
    if compare_code:
        if not cost_series.empty:
            cost_positive_dollars = float(cost_series[cost_series > 0].sum() * 1_000_000.0)
            cost_negative_dollars = float(cost_series[cost_series < 0].sum() * 1_000_000.0)
        if not benefit_series.empty:
            benefit_positive_dollars = float(benefit_series[benefit_series > 0].sum())
            benefit_negative_dollars = float(benefit_series[benefit_series < 0].sum())
    axis_samples = []
    if not cost_series.empty:
        axis_samples.extend((cost_series.astype(float).to_numpy(dtype=float) * 1_000_000.0).tolist())
    if not benefit_series.empty:
        axis_samples.extend((benefit_series.astype(float).to_numpy(dtype=float)).tolist())
    axis_samples.extend([cost_total_dollars, benefit_total_dollars])
    if compare_code:
        axis_samples.extend(
            [
                cost_positive_dollars,
                cost_negative_dollars,
                benefit_positive_dollars,
                benefit_negative_dollars,
            ]
        )
    axis_samples = [val for val in axis_samples if np.isfinite(val)]
    if not axis_samples or max(abs(val) for val in axis_samples) <= 1e-6:
        return None

    def _segment_text(
        value_dollars: float,
        *,
        total_dollars: float,
        abs_total_dollars: float,
    ) -> str:
        try:
            value_dollars = float(value_dollars)
        except (TypeError, ValueError):
            return ""
        if not math.isfinite(value_dollars):
            return ""

        def _format_billions(value_billions: float) -> str:
            value_billions = abs(float(value_billions))
            if value_billions >= 10:
                return f"{value_billions:.0f}"
            text = f"{value_billions:.1f}"
            return text.rstrip("0").rstrip(".")

        if compare_code:
            denom = float(abs_total_dollars) if abs_total_dollars else 0.0
            if denom <= 0:
                return ""
            share = abs(value_dollars) / denom
            if share < 0.08:
                return ""
            abs_billions = abs(value_dollars) / 1_000_000_000.0
            if abs_billions < 0.5:
                return ""
            sign = "+" if value_dollars > 0 else "-"
            return f"{sign}${_format_billions(abs_billions)}b"

        if total_dollars <= 0 or value_dollars <= 0:
            return ""
        share = value_dollars / total_dollars if total_dollars else 0.0
        if share < 0.08:
            return ""
        billions = value_dollars / 1_000_000_000.0
        return f"${billions:.0f}b" if billions >= 1.0 else ""

    fig = go.Figure()
    gap_color = "#111111" if is_dark_mode() else "#FFFFFF"
    segment_outline = dict(color=gap_color, width=2)

    x_cost = "Costs \u0394" if compare_code else "Costs"
    x_benefit = "Benefits \u0394" if compare_code else "Benefits"
    hover_cost_prefix = "Costs \u0394" if compare_code else "Costs"
    hover_benefit_prefix = "Benefits \u0394" if compare_code else "Benefits"

    cost_labels = cost_series.index.astype(str).tolist()
    cost_colors = list(reversed(_sample_stack_palette("Blues", len(cost_labels), start=0.55, end=0.95)))
    cost_legend_items = [(str(label), str(color)) for label, color in zip(cost_labels, cost_colors)]
    for label, color in zip(cost_labels, cost_colors):
        value_m = float(cost_series.get(label, 0.0))
        value_dollars = value_m * 1_000_000.0
        fig.add_trace(
            go.Bar(
                x=[x_cost],
                y=[value_dollars],
                name=str(label),
                marker=dict(color=color, line=segment_outline),
                text=[
                    _segment_text(
                        value_dollars,
                        total_dollars=cost_total_dollars,
                        abs_total_dollars=cost_abs_total_dollars,
                    )
                ],
                textposition="inside",
                insidetextfont=dict(color="white", size=12),
                constraintext="inside",
                customdata=[format_large_amount(value_dollars)],
                hovertemplate=f"<b>{hover_cost_prefix}: {html.escape(label)}</b><br>%{{customdata}}<extra></extra>",
            )
        )

    benefit_labels = benefit_series.index.astype(str).tolist()
    benefit_colors = list(reversed(_sample_stack_palette("YlGn", len(benefit_labels), start=0.15, end=0.65)))
    benefit_legend_items = [(str(label), str(color)) for label, color in zip(benefit_labels, benefit_colors)]
    for label, color in zip(benefit_labels, benefit_colors):
        value_dollars = float(benefit_series.get(label, 0.0))
        fig.add_trace(
            go.Bar(
                x=[x_benefit],
                y=[value_dollars],
                name=str(label),
                marker=dict(color=color, line=segment_outline),
                text=[
                    _segment_text(
                        value_dollars,
                        total_dollars=benefit_total_dollars,
                        abs_total_dollars=benefit_abs_total_dollars,
                    )
                ],
                textposition="inside",
                insidetextfont=dict(color="#111111", size=12),
                constraintext="inside",
                customdata=[format_large_amount(value_dollars)],
                hovertemplate=f"<b>{hover_benefit_prefix}: {html.escape(label)}</b><br>%{{customdata}}<extra></extra>",
            )
        )

    title_horizon = (
        f" {pv_horizon_years_int}Y"
        if (cost_value_basis == VALUE_BASIS_PV or benefit_value_basis == VALUE_BASIS_PV)
        else ""
    )
    title_prefix = "Costs vs benefits delta" if compare_code else "Costs vs benefits stack"
    fig.update_layout(
        title=f"{title_prefix}{title_horizon} - {selection_label}",
        template=plotly_template(),
        barmode="relative" if compare_code else "stack",
        bargap=0.28,
        barcornerradius=6,
        uniformtext=dict(minsize=10, mode="hide"),
        hoverlabel=dict(namelength=-1),
        showlegend=False,
        margin=dict(l=70, r=20, t=56, b=54, autoexpand=False),
        height=480,
    )
    if any(val < -1e-9 for val in axis_samples):
        fig.add_hline(y=0, line=dict(color="#111111", width=2))

    ticks_vals, ticks_text, _ = compute_cash_axis_ticks(
        axis_samples,
        force_unit="b",
    )
    fig.update_yaxes(
        title=None,
        tickmode="array",
        tickvals=ticks_vals,
        ticktext=ticks_text,
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    fig.update_xaxes(title=None, showgrid=False)
    return fig, cost_legend_items, benefit_legend_items


def prepare_cost_benefit_stack_export(
    data: DashboardData,
    selection: ScenarioSelection,
    *,
    selection_label: str,
    pv_horizon_years: Optional[int],
    cost_breakdown: str,
    benefit_breakdown: str,
    cost_value_basis: str,
    benefit_value_basis: str,
    compare_selection: Optional[ScenarioSelection] = None,
) -> Optional[pd.DataFrame]:
    series_result = _compute_cost_benefit_stack_series(
        data,
        selection,
        pv_horizon_years=pv_horizon_years,
        cost_breakdown=cost_breakdown,
        benefit_breakdown=benefit_breakdown,
        cost_value_basis=cost_value_basis,
        benefit_value_basis=benefit_value_basis,
        compare_selection=compare_selection,
    )
    if series_result is None:
        return None

    cost_series, benefit_series, compare_code, _ = series_result
    if cost_series.empty and benefit_series.empty:
        return None

    rows: List[Dict[str, Any]] = []
    cost_label = "Costs Δ" if compare_code else "Costs"
    benefit_label = "Benefits Δ" if compare_code else "Benefits"
    for label, value in cost_series.items():
        rows.append(
            {
                "Selection": selection_label,
                "Series": cost_label,
                "Breakdown": cost_breakdown,
                "Category": str(label),
                "Value ($)": float(value) * 1_000_000.0,
            }
        )
    for label, value in benefit_series.items():
        rows.append(
            {
                "Selection": selection_label,
                "Series": benefit_label,
                "Breakdown": benefit_breakdown,
                "Category": str(label),
                "Value ($)": float(value),
            }
        )
    return pd.DataFrame(rows)


def render_cost_benefit_stack_legend(
    cost_items: List[Tuple[str, str]],
    benefit_items: List[Tuple[str, str]],
    *,
    columns_per_section: int = 2,
) -> None:
    if not cost_items and not benefit_items:
        return

    divider = "rgba(255, 255, 255, 0.22)" if is_dark_mode() else "rgba(15, 23, 42, 0.14)"

    def _split(items: List[Tuple[str, str]]) -> List[List[Tuple[str, str]]]:
        if columns_per_section <= 1 or len(items) <= 1:
            return [items]
        cols = max(1, int(columns_per_section))
        per_col = int(math.ceil(len(items) / cols))
        return [items[i * per_col : (i + 1) * per_col] for i in range(cols)]

    def _item_html(label: str, color: str) -> str:
        safe_label = html.escape(str(label))
        safe_color = html.escape(str(color))
        return (
            f'<div class="copt-stack-legend-item" title="{safe_label}">'
            f'<span class="copt-stack-legend-swatch" style="background:{safe_color};"></span>'
            f'<span class="copt-stack-legend-label">{safe_label}</span>'
            f"</div>"
        )

    def _section_html(title: str, items: List[Tuple[str, str]]) -> str:
        if not items:
            return ""
        cols = _split(items)
        cols_html = "".join(
            f'<div class="copt-stack-legend-col">{"".join(_item_html(label, color) for label, color in col)}</div>'
            for col in cols
            if col
        )
        safe_title = html.escape(title)
        return (
            '<div class="copt-stack-legend-section">'
            f'<div class="copt-stack-legend-title">{safe_title}</div>'
            f'<div class="copt-stack-legend-columns">{cols_html}</div>'
            "</div>"
        )

    html_block = f"""
<style>
.copt-stack-legend-container {{
  display: flex;
  gap: 18px;
  margin-top: 10px;
}}
.copt-stack-legend-section {{
  flex: 1 1 0;
  min-width: 0;
}}
.copt-stack-legend-section + .copt-stack-legend-section {{
  border-left: 1px solid {divider};
  padding-left: 18px;
}}
.copt-stack-legend-title {{
  font-weight: 600;
  font-size: 0.82rem;
  margin-bottom: 6px;
}}
.copt-stack-legend-columns {{
  display: flex;
  gap: 18px;
}}
.copt-stack-legend-col {{
  flex: 1 1 0;
  min-width: 0;
}}
.copt-stack-legend-item {{
  display: flex;
  align-items: center;
  gap: 7px;
  margin: 2px 0;
  min-width: 0;
}}
.copt-stack-legend-swatch {{
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex: 0 0 10px;
}}
.copt-stack-legend-label {{
  font-size: 0.74rem;
  line-height: 1.25;
  color: inherit;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
</style>
<div class="copt-stack-legend-container">
  {_section_html("Costs", cost_items)}
  {_section_html("Benefits", benefit_items)}
</div>
"""
    st.markdown(html_block, unsafe_allow_html=True)

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

    pivot = (
        pivot.apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(float)
        * 1_000.0
    )

    preferred_order = {
        str(dim): idx
        for idx, dim in enumerate(data.dims)
        if str(dim).strip().lower() != "total"
    }
    dims = [
        dim for dim in pivot.columns if str(dim).strip().lower() != "total"
    ]
    dims.sort(
        key=lambda dim: (
            preferred_order.get(str(dim), len(preferred_order)),
            str(dim),
        )
    )

    if not dims:

        return None

    fig = go.Figure()

    palette = BENEFIT_DIMENSION_PALETTE or ["#19456B"]
    for idx, dim in enumerate(dims):
        dim_label = str(dim)
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=pivot.index,
                y=pivot[dim],
                name=dim_label,
                mode="lines",
                line=dict(color=color, width=1.4),
                stackgroup="benefits",
                fillcolor=rgba_from_hex(color, 0.32),
                opacity=0.9,
                legendgroup=dim_label,
                customdata=[format_large_amount(val) for val in pivot[dim]],
                hovertemplate=f"FY %{{x}}: %{{customdata}}<extra>{dim_label}</extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=None,
        legend=legend_bottom(),
        margin=dict(l=40, r=40, t=80, b=60),
        template=plotly_template(),
        yaxis=dict(
            title=None,
            tickformat="~s",
        ),
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

    pivot_opt_raw = opt_pivot if opt_pivot is not None else dimension_timeseries(data, opt_selection)

    pivot_cmp_raw = cmp_pivot if cmp_pivot is not None else dimension_timeseries(data, comp_selection)

    if pivot_opt_raw is None and pivot_cmp_raw is None:

        return None

    def _prepare_pivot(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:

        if frame is None or frame.empty:

            return None

        numeric = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

        if cumulative:

            numeric = numeric.cumsum()

        return numeric * 1_000.0

    pivot_opt = _prepare_pivot(pivot_opt_raw)

    pivot_cmp = _prepare_pivot(pivot_cmp_raw)

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

    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()

    title_suffix = " (cumulative)" if cumulative else ""

    fig = go.Figure()

    palette = BENEFIT_DIMENSION_PALETTE or ["#19456B"]

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

                    name=f"{dim} - {primary_label}",

                    mode='lines',

                    line=dict(color=color, width=2.6),

                    fill='tozeroy',

                    fillcolor=rgba_from_hex(color, 0.28),

                    opacity=0.85,

                    legendgroup=dim,

                    customdata=[format_large_amount(val) for val in opt_series],

                    hovertemplate=f"{primary_label} {dim}<br>FY %{{x}}: %{{customdata}}<extra></extra>",

                )

            )

        if cmp_series is not None:

            fig.add_trace(

                go.Scatter(

                    x=cmp_series.index,

                    y=cmp_series,

                    name=f"{dim} - {comparison_label}",

                    mode='lines',

                    line=dict(color=color, width=2.2, dash='dot'),

                    fill='tozeroy',

                    fillcolor=rgba_from_hex(color, 0.16),

                    opacity=0.85,

                    legendgroup=dim,

                    customdata=[format_large_amount(val) for val in cmp_series],

                    hovertemplate=f"{comparison_label} {dim}<br>FY %{{x}}: %{{customdata}}<extra></extra>",

                    showlegend=True,

                )

            )

    if not fig.data:

        return None

    fig.update_layout(

        title=f"Dimension benefit {primary_label} vs {comparison_label}{title_suffix}",

        xaxis_title=None,

        legend=legend_bottom(),

        margin=dict(l=40, r=40, t=80, b=60),

        template=plotly_template(),

        hoverlabel=dict(namelength=-1),

        yaxis=dict(

            title=None,

            tickformat="~s",

        ),

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
    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()
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

                f"{primary_label} start: {run.start_year}",

                f"{comparison_label} start: {other.start_year}",

            ]

            hover_text = "<br>".join(hover_lines)

            fig.add_trace(

                go.Scatter(

                    x=[left, right, right, left, left],

                    y=[y_bottom, y_bottom, y_top, y_top, y_bottom],

                    mode="lines",

                    line=dict(color=outline_color, width=1.5, dash="dot"),

                    fill="toself",

                    fillcolor="rgba(0, 0, 0, 0)",

                    fillpattern=dict(shape=".", size=2, solidity=0.4, fgcolor=outline_color, fgopacity=0.6, bgcolor="rgba(0, 0, 0, 0)"),

                    hoveron="fills",

                    hoverinfo="skip",

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
                        fillcolor="rgba(0, 0, 0, 0)",
                        fillpattern=dict(shape=".", size=2, solidity=0.4, fgcolor=outline_color, fgopacity=0.6, bgcolor="rgba(0, 0, 0, 0)"),
                        hoveron="fills",
                        hoverinfo="skip",
                        name="",
                        hoverlabel=_hoverlabel_style(),
                        showlegend=False,
                        cliponaxis=False,
                    )
                )

                span = max(seg_end - seg_start, 1e-6)
                marker_count = max(1, int(math.ceil(span)))
                if marker_count == 1:
                    marker_x = [0.5 * (seg_start + seg_end)]
                else:
                    step = span / marker_count
                    marker_x = list(np.linspace(seg_start + 0.5 * step, seg_end - 0.5 * step, marker_count))
                marker_y = [y_center] * len(marker_x)
                fig.add_trace(
                    go.Scatter(
                        x=marker_x,
                        y=marker_y,
                        mode="markers",
                        marker=dict(color="rgba(0, 0, 0, 0)", size=28, symbol="square"),
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

    display_year_range: Optional[Tuple[int, int]] = None,

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

    display_years = list(years)
    if display_year_range:
        start_year, end_year = display_year_range
        display_years = [
            year for year in years
            if (start_year is None or year >= start_year) and (end_year is None or year <= end_year)
        ]
        if not display_years:
            display_years = list(years)

    tick_step = 1
    if len(display_years) > 40:
        tick_step = 4
    elif len(display_years) > 25:
        tick_step = 2
    tick_vals = display_years[::tick_step]
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

    min_year = display_years[0]

    max_year = display_years[-1]

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

def market_capacity_indicator(
    data: DashboardData,
    selection: ScenarioSelection,
    *,
    style: str = "gradient",
    show_colorbar: bool | None = None,
    height_px: int | None = None,
) -> Optional[go.Figure]:
    """Thin, polished heat strip for market capacity."""
    series = build_timeseries(data, selection)
    if series is None or series.empty:
        return None

    df = series[["Year", "Spend"]].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"]).sort_values("Year")
    if df.empty:
        return None

    df["SpendB"] = pd.to_numeric(df["Spend"], errors="coerce").fillna(0.0) / 1000.0
    years = df["Year"].astype(int).tolist()
    spend_b = df["SpendB"].astype(float).tolist()

    max_b = max(spend_b) if spend_b else 0.0
    scale_top_b = _nice_half_step(max_b, min_top=3.25)

    style_key = (style or "gradient").strip().lower()
    if style_key == "bands":
        def level(vb: float) -> int:
            if vb <= 0.0:
                return 0
            if vb >= 3.0:
                return 3
            if vb > 2.0:
                return 2
            return 1

        z_values = [level(v) for v in spend_b]
        zmin, zmax = 0, 3
        colorscale = _capacity_step_colorscale()
        status_labels = [
            "No spend recorded" if v <= 0.0
            else "High pressure (>= $3.0B)" if v >= 3.0
            else "Watch zone ($2.0B-$3.0B)" if v > 2.0
            else "Comfortable (<= $2.0B)"
            for v in spend_b
        ]
    else:
        z_values = [float(np.clip(v / max(scale_top_b, 1e-6), 0.0, 1.0)) for v in spend_b]
        zmin, zmax = 0.0, 1.0
        colorscale = _capacity_gradient_colorscale(scale_top_b)
        status_labels = [
            "No spend recorded" if v <= 0.0
            else "High pressure" if v >= 3.0
            else "Intensifying" if 2.6 < v < 3.0
            else "Watch zone" if v > 2.0
            else "Comfortable"
            for v in spend_b
        ]

    custom = []
    for y, vb, status in zip(years, spend_b, status_labels):
        if np.isfinite(vb):
            b_txt = f"{vb:,.1f} B"
            m_txt = f"{vb * 1000.0:,.0f} m"
        else:
            b_txt = m_txt = "-"
        custom.append([int(y), float(vb), b_txt, m_txt, status])

    heat = go.Heatmap(
        x=years,
        y=[0],
        z=[z_values],
        zmin=zmin,
        zmax=zmax,
        colorscale=colorscale,
        showscale=bool(SHOW_CAPACITY_COLORBAR if show_colorbar is None else show_colorbar),
        xgap=0,
        ygap=0,
        hoverinfo="text",
        hovertemplate=(
            "<b>FY %{customdata[0]}</b><br>"
            "Total spend: %{customdata[2]} | %{customdata[3]}<br>"
            "Status: %{customdata[4]}<extra></extra>"
        ),
        customdata=[custom],
        colorbar=dict(
            title="Spend scale",
            len=0.4,
            thickness=12,
            x=1.03,
            y=0.5,
            ticks="outside",
            outlinewidth=0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    fig = go.Figure(data=[heat])
    fig.update_traces(hoverlabel=_hoverlabel_style())

    schedule_left_margin = 40
    schedule_right_margin = 220

    max_spend_m = float(pd.to_numeric(series["Spend"], errors="coerce").fillna(0.0).max())
    if max_spend_m > 0:
        nice = int(math.ceil(max_spend_m / 100.0) * 100)
        widest_tick = f"{nice:,}"
    else:
        widest_tick = "0"

    n_years = max(1, len(years))
    if height_px is None:
        height_px = int(max(28, min(44, 44 - 0.12 * (n_years - 20))))
    bottom_margin = max(36, int(height_px * 0.65))
    height_px += max(0, bottom_margin - 2)


    x0 = float(min(years))
    x1 = float(max(years))

    fig.update_layout(
        height=height_px,
        margin=dict(l=schedule_left_margin, r=schedule_right_margin, t=20, b=bottom_margin),
        autosize=False,
        xaxis=dict(
            tickmode="linear",
            dtick=1,
            tick0=x0,
            range=[x0 - 0.5, x1 + 0.5],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            constrain="range",
            fixedrange=True,
        ),
        yaxis=dict(
            automargin=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks="",
            tickmode="array",
            tickvals=[0],
            ticktext=[widest_tick],
            tickfont=dict(color="rgba(0,0,0,0)", size=12),
            title=dict(text="Annual spend ($m)", font=dict(color="rgba(0,0,0,0)", size=12)),
            visible=True,
            range=[-0.5, 0.5],
            fixedrange=True,
        ),
        template=plotly_template(),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.add_hrect(y0=-0.5, y1=-0.49, line_width=0, fillcolor="rgba(0,0,0,0.05)")
    fig.add_hrect(y0=0.49, y1=0.5, line_width=0, fillcolor="rgba(0,0,0,0.05)")

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
    # Fallback - let Plotly help normalise then strip labels
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


def _colorscale_with_zero_base(colorscale: Any, zero_color: str = "#E5E7EB") -> list[list[Any]]:
    """Return a colorscale with a guaranteed light zero stop."""
    try:
        resolved = plc.get_colorscale(colorscale)
    except Exception:
        resolved = colorscale if isinstance(colorscale, (list, tuple)) else plc.get_colorscale('YlOrRd')
    resolved = sorted(resolved, key=lambda entry: float(entry[0]))
    if not resolved:
        fallback_colour = PBI_SEQUENTIAL_SCALE[-1][1] if PBI_SEQUENTIAL_SCALE else "#19456B"
        return [[0.0, zero_color], [1.0, fallback_colour]]
    epsilon = 1e-6
    adjusted: list[list[Any]] = [[0.0, zero_color]]
    for stop, color in resolved:
        adjusted.append([max(float(stop), epsilon), color])
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


def _safe_float(value: Any) -> float | None:
    """Return a finite float or None if coercion fails."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _kpi_card_html(
    title: str,
    value: str | None,
    *,
    subtitle: str | None = None,
    delta_text: str | None = None,
    delta_state: str = "neutral",
    body_html: str | None = None,
) -> str:
    """Return HTML for a single KPI card."""
    parts = [
        '<div class="kpi-card">',
        f'<div class="kpi-title">{title}</div>',
    ]
    if body_html is not None:
        parts.append(body_html)
    else:
        if value is not None:
            parts.append(f'<div class="kpi-value">{value}</div>')
        if subtitle:
            parts.append(f'<div class="kpi-sub">{subtitle}</div>')
        if delta_text:
            parts.append(f'<div class="kpi-delta {delta_state}">{delta_text}</div>')
    parts.append('</div>')
    return "\n".join(parts)


def render_solver_gap_kpis(
    opt_selection: Optional["ScenarioSelection"],
    cmp_selection: Optional["ScenarioSelection"],
) -> None:
    import streamlit as st

    def _fmt_gap(value: float | None) -> str:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return "N/A"
        if not np.isfinite(val):
            return "N/A"
        return f"{val * 100:.2f} %"

    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()
    opt_meta = opt_selection.metadata if opt_selection else {}
    cmp_meta = cmp_selection.metadata if cmp_selection else {}
    opt_gap = opt_meta.get("Gap") if isinstance(opt_meta, dict) else None
    cmp_gap = cmp_meta.get("Gap") if isinstance(cmp_meta, dict) else None

    grid_token = uuid4().hex[:8]
    anim_name = f"kpiSweep_{grid_token}"
    style_block = textwrap.dedent(
        f"""
        <style>
        @keyframes {anim_name} {{
            0% {{ transform: translateY(-3px) scaleX(0); }}
            100% {{ transform: translateY(-3px) scaleX(1); }}
        }}
        [data-kpi-grid-id="{grid_token}"] .kpi-card::after {{
            animation: {anim_name} 0.275s ease-out forwards;
        }}
        [data-kpi-grid-id="{grid_token}"] .kpi-card:nth-of-type(1)::after {{ animation-delay: 0ms; }}
        [data-kpi-grid-id="{grid_token}"] .kpi-card:nth-of-type(2)::after {{ animation-delay: 0ms; }}
        </style>
        """
    )
    cards = [
        f'<div class="kpi-grid" data-kpi-grid-id="{grid_token}">',
        _kpi_card_html(
            f"{primary_label} solver relative gap",
            _fmt_gap(opt_gap),
        ),
        _kpi_card_html(
            f"{comparison_label} solver relative gap",
            _fmt_gap(cmp_gap),
        ),
        "</div>",
    ]
    st.markdown(style_block + "".join(cards), unsafe_allow_html=True)


def render_programme_kpis(
    opt_series: pd.DataFrame | None,
    cmp_series: pd.DataFrame | None,
    *,
    data: DashboardData,
    opt_selection: Optional[ScenarioSelection],
    cmp_selection: Optional[ScenarioSelection],
) -> None:
    """Render the overview KPI card grid."""
    import streamlit as st

    def _fmt(value: float | None) -> str:
        return format_currency(value) if (value is not None and np.isfinite(value)) else "-"

    def _fmt_ratio(value: float | None) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "-"
        if not np.isfinite(numeric):
            return "-"
        return f"{numeric:.2f}"

    def _fmt_abs(value: float | None) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "-"
        if not np.isfinite(numeric):
            return "-"
        return format_currency(abs(numeric))

    primary_label = scenario_primary_label()
    comparison_label = scenario_comparison_label()
    pair_label = scenario_pair_label()

    benefit_basis = _resolve_value_basis("npv_benefit_value_basis", "npv_apply_discount")
    cost_basis = _resolve_value_basis("npv_cost_value_basis", "npv_apply_cost_discount")
    pv_horizon_years = None
    if benefit_basis == VALUE_BASIS_PV or cost_basis == VALUE_BASIS_PV:
        try:
            pv_horizon_years = int(st.session_state.get("npv_horizon_selection", 60))
        except (TypeError, ValueError):
            pv_horizon_years = 60

    def _basis_label(kind: str, basis: str) -> str:
        basis_norm = (basis or "").strip().lower()
        if kind == "benefit":
            if basis_norm.startswith("pv"):
                return "NPV benefits"
            if basis_norm.startswith("nom"):
                return "Benefits (nominal)"
            return "Benefits ($2025)"
        if basis_norm.startswith("pv"):
            return "NPV expenditure"
        if basis_norm.startswith("nom"):
            return "Expenditure (nominal)"
        return "Expenditure ($2025)"

    def _total_from_series(series: pd.DataFrame | None, column: str, basis: str) -> float | None:
        if series is None or series.empty:
            return None
        if "Year" not in series.columns:
            return None
        values = pd.to_numeric(series.get(column), errors="coerce").fillna(0.0)
        years = pd.to_numeric(series.get("Year"), errors="coerce").fillna(data.start_fy).astype(int)
        mask = years.notna()
        if not mask.any():
            return None
        values = values.loc[mask].astype(float)
        years = years.loc[mask].astype(int)
        if basis == VALUE_BASIS_PV and pv_horizon_years:
            limit_year = int(data.start_fy) + int(pv_horizon_years) - 1
            horizon_mask = years <= limit_year
            values = values.loc[horizon_mask]
            years = years.loc[horizon_mask]
        multipliers = value_basis_multiplier(int(data.start_fy), years.to_numpy(dtype=int), basis)
        return float((values.to_numpy(dtype=float) * multipliers).sum())

    benefit_col = "BenefitFlowTotal" if opt_series is not None and "BenefitFlowTotal" in opt_series.columns else "BenefitFlow"
    opt_benefit = _total_from_series(opt_series, benefit_col, benefit_basis)
    cmp_benefit = _total_from_series(cmp_series, benefit_col, benefit_basis)
    delta_benefit = (opt_benefit - cmp_benefit) if (opt_benefit is not None and cmp_benefit is not None) else None

    opt_spend = _total_from_series(opt_series, "Spend", cost_basis)
    cmp_spend = _total_from_series(cmp_series, "Spend", cost_basis)
    delta_spend = (opt_spend - cmp_spend) if (opt_spend is not None and cmp_spend is not None) else None

    def _ratio(numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None:
            return None
        try:
            denom_val = float(denominator)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(denom_val) or math.isclose(denom_val, 0.0, abs_tol=1e-9):
            return None
        try:
            num_val = float(numerator)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(num_val):
            return None
        return num_val / denom_val

    bcr_opt = _ratio(opt_benefit, opt_spend)
    bcr_cmp = _ratio(cmp_benefit, cmp_spend)

    benefit_label = _basis_label("benefit", benefit_basis)
    cost_label = _basis_label("cost", cost_basis)
    benefit_prefix = f"{pv_horizon_years}-year " if (benefit_basis == VALUE_BASIS_PV and pv_horizon_years) else ""
    cost_prefix = f"{pv_horizon_years}-year " if (cost_basis == VALUE_BASIS_PV and pv_horizon_years) else ""

    if delta_benefit is None or not np.isfinite(delta_benefit):
        delta_pv_card = _kpi_card_html(
            f"Delta Total {benefit_prefix}{benefit_label}",
            _fmt(delta_benefit),
            subtitle=pair_label,
        )
    else:
        if math.isclose(delta_benefit, 0.0, abs_tol=1e-6):
            pv_chip_state = "muted"
            pv_chip_text = f"{_fmt_abs(delta_benefit)} vs {comparison_label}"
        else:
            sign = "&#9650;" if delta_benefit > 0 else "&#9660;"
            pv_chip_state = "up" if delta_benefit > 0 else "down"
            pv_chip_text = f"{sign} {_fmt_abs(delta_benefit)} vs {comparison_label}"
        delta_pv_card = _kpi_card_html(
            f"Delta Total {benefit_prefix}{benefit_label}",
            None,
            body_html=f'<div class="kpi-delta lead {pv_chip_state}">{pv_chip_text}</div>',
        )

    if delta_spend is None or not np.isfinite(delta_spend):
        cost_delta_card = _kpi_card_html(
            f"Delta Total {cost_prefix}{cost_label}",
            _fmt(delta_spend),
            subtitle=pair_label,
        )
    else:
        if math.isclose(delta_spend, 0.0, abs_tol=1e-6):
            cost_chip_text = f"{_fmt_abs(delta_spend)} vs {comparison_label}"
            cost_chip_state = "muted"
        else:
            more_spend = delta_spend > 0
            cost_arrow = "&#9650;" if more_spend else "&#9660;"
            cost_chip_state = "down" if more_spend else "up"
            cost_chip_text = f"{cost_arrow} {_fmt_abs(delta_spend)} vs {comparison_label}"
        cost_delta_card = _kpi_card_html(
            f"Delta Total {cost_prefix}{cost_label}",
            None,
            body_html=f'<div class="kpi-delta lead {cost_chip_state}">{cost_chip_text}</div>',
        )

    efficiency_body: str
    if bcr_opt is None or bcr_cmp is None or not np.isfinite(bcr_opt) or not np.isfinite(bcr_cmp):
        efficiency_body = '<div class="kpi-sub">Efficiency comparison unavailable</div>'
    else:
        diff_pct = ((bcr_opt / bcr_cmp) - 1.0) * 100.0 if bcr_cmp else 0.0
        if math.isclose(diff_pct, 0.0, abs_tol=1e-6):
            diff_pct = 0.0
            direction = "more"
            bubble_state = "muted"
            arrow = ""
        else:
            direction = "more" if diff_pct >= 0 else "less"
            bubble_state = "up" if diff_pct >= 0 else "down"
            arrow = "&#9650;" if diff_pct >= 0 else "&#9660;"
        efficiency_body = (
            '<div class="kpi-inline">'
            f'<span class="kpi-text">{html.escape(primary_label)} is</span>'
            f'<span class="kpi-delta {bubble_state}">{arrow} {abs(diff_pct):.1f}%</span>'
            f'<span class="kpi-text"><strong>{direction}</strong> efficient than</span>'
            f'<span class="kpi-text">{html.escape(comparison_label)}</span>'
            "</div>"
        )

    grid_token = uuid4().hex[:8]
    anim_name = f"kpiSweep_{grid_token}"
    style_block = textwrap.dedent(
        f"""
        <style>
        @keyframes {anim_name} {{
            0% {{ transform: translateY(-3px) scaleX(0); }}
            100% {{ transform: translateY(-3px) scaleX(1); }}
        }}
        [data-kpi-grid-id="{grid_token}"] .kpi-card::after {{
            animation: {anim_name} 0.275s ease-out forwards;
        }}
        [data-kpi-grid-id="{grid_token}"] .kpi-card:nth-of-type(1)::after {{ animation-delay: 0ms; }}
        [data-kpi-grid-id="{grid_token}"] .kpi-card:nth-of-type(2)::after {{ animation-delay: 0ms; }}
        [data-kpi-grid-id="{grid_token}"] .kpi-card:nth-of-type(3)::after {{ animation-delay: 0ms; }}
        [data-kpi-grid-id="{grid_token}"] .kpi-card:nth-of-type(4)::after {{ animation-delay: 0ms; }}
        [data-kpi-grid-id="{grid_token}"] .kpi-card:nth-of-type(5)::after {{ animation-delay: 0ms; }}
        [data-kpi-grid-id="{grid_token}"] .kpi-card:nth-of-type(6)::after {{ animation-delay: 0ms; }}
        </style>
        """
    )
    st.markdown(style_block, unsafe_allow_html=True)

    cards = [
        f'<div class="kpi-grid" data-kpi-grid-id="{grid_token}">',
        _kpi_card_html(f"Total {cost_prefix}{cost_label} - {primary_label}", _fmt(opt_spend)),
        _kpi_card_html(f"Total {cost_prefix}{cost_label} - {comparison_label}", _fmt(cmp_spend)),
        cost_delta_card,
        _kpi_card_html(
            f"Total {benefit_prefix}{benefit_label} - {primary_label}",
            _fmt(opt_benefit),
        ),
        _kpi_card_html(
            f"Total {benefit_prefix}{benefit_label} - {comparison_label}",
            _fmt(cmp_benefit),
        ),
        delta_pv_card,
        _kpi_card_html(
            f"BCR - Portfolio Weighted Average - {primary_label}",
            _fmt_ratio(bcr_opt),
        ),
        _kpi_card_html(
            f"BCR - Portfolio Weighted Average - {comparison_label}",
            _fmt_ratio(bcr_cmp),
        ),
        _kpi_card_html(
            "Efficiency",
            None,
            body_html=efficiency_body,
        ),
        '</div>',
    ]
    st.markdown("".join(cards), unsafe_allow_html=True)


REGION_METRIC_CONFIG: Dict[str, Dict[str, Any]] = {
    "Share_Cum": {
        "label": "Cumulative share vs national",
        "description": "Share of cumulative spend allocated to each region compared with the national total.",
        "type": "sequential",
        "colorscale": PBI_SEQUENTIAL_SCALE,
        "multiplier": 100.0,
        "colorbar": "Share (%)",
        "ticksuffix": "%",
        "tickformat": ".1f",
        "force_zero_min": True,
        "formatter": lambda v: _format_percentage(v, 1),
        "sort": "desc",
        "share_columns": {"cum": "Share_Cum", "year": "Share_Year"},
    },
    "Share_Year": {
        "label": "Annual share vs national",
        "description": "Share of annual spend in the selected year compared with the national total.",
        "type": "sequential",
        "colorscale": PBI_SEQUENTIAL_SCALE,
        "multiplier": 100.0,
        "colorbar": "Share (%)",
        "ticksuffix": "%",
        "tickformat": ".1f",
        "force_zero_min": True,
        "formatter": lambda v: _format_percentage(v, 1),
        "sort": "desc",
        "share_columns": {"cum": "Share_Cum", "year": "Share_Year"},
    },
    "PerCap_Cum": {
        "label": "Cumulative spend per capita",
        "description": "Cumulative spend per resident since the start of the programme.",
        "type": "sequential",
        "colorscale": PBI_SEQUENTIAL_SCALE,
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
        "colorscale": PBI_SEQUENTIAL_SCALE,
        "multiplier": 1_000_000.0,
        "colorbar": "$ per person",
        "tickformat": ",.0f",
        "force_zero_min": True,
        "formatter": _format_currency_nzd,
        "table_label": "Per-cap annual",
        "sort": "desc",
    },
    "BenefitShare_Cum": {
        "label": "Cumulative benefit share vs national",
        "description": "Share of cumulative benefit allocated to each region compared with the national total.",
        "type": "sequential",
        "colorscale": PBI_SEQUENTIAL_SCALE,
        "multiplier": 100.0,
        "colorbar": "Benefit share (%)",
        "ticksuffix": "%",
        "tickformat": ".1f",
        "force_zero_min": True,
        "formatter": lambda v: _format_percentage(v, 1),
        "sort": "desc",
        "share_columns": {"cum": "BenefitShare_Cum", "year": "BenefitShare_Year"},
        "table_label": "Benefit share (cum)",
    },
    "BenefitShare_Year": {
        "label": "Annual benefit share vs national",
        "description": "Share of annual benefit in the selected year compared with the national total.",
        "type": "sequential",
        "colorscale": PBI_SEQUENTIAL_SCALE,
        "multiplier": 100.0,
        "colorbar": "Benefit share (%)",
        "ticksuffix": "%",
        "tickformat": ".1f",
        "force_zero_min": True,
        "formatter": lambda v: _format_percentage(v, 1),
        "sort": "desc",
        "share_columns": {"cum": "BenefitShare_Cum", "year": "BenefitShare_Year"},
        "table_label": "Benefit share (annual)",
    },
    "BenefitPerCap_Cum": {
        "label": "Cumulative benefit per capita",
        "description": "Cumulative benefit per resident since the start of the programme.",
        "type": "sequential",
        "colorscale": PBI_SEQUENTIAL_SCALE,
        "multiplier": 1_000_000.0,
        "colorbar": "$ per person",
        "tickformat": ",.0f",
        "force_zero_min": True,
        "formatter": _format_currency_nzd,
        "table_label": "Benefit per-cap cum",
        "sort": "desc",
        "share_columns": {"cum": "BenefitShare_Cum", "year": "BenefitShare_Year"},
    },
    "BenefitPerCap_Year": {
        "label": "Annual benefit per capita",
        "description": "Annual benefit in the selected year per resident.",
        "type": "sequential",
        "colorscale": PBI_SEQUENTIAL_SCALE,
        "multiplier": 1_000_000.0,
        "colorbar": "$ per person",
        "tickformat": ",.0f",
        "force_zero_min": True,
        "formatter": _format_currency_nzd,
        "table_label": "Benefit per-cap annual",
        "sort": "desc",
        "share_columns": {"cum": "BenefitShare_Cum", "year": "BenefitShare_Year"},
    },
}


REGION_METRIC_ORDER = list(REGION_METRIC_CONFIG.keys())
REGION_METRIC_GROUPS = {
    "Spend share": [
        "Share_Cum",
        "Share_Year",
        "PerCap_Cum",
        "PerCap_Year",
        # "OU_vs_Pop",
        # "OU_vs_GDP",
        # "Ramp_Rate",
    ],
    "Benefit share": [
        "BenefitShare_Cum",
        "BenefitShare_Year",
        "BenefitPerCap_Cum",
        "BenefitPerCap_Year",
    ],
}

REGION_METRIC_DEFAULT = {
    "Spend share": "Share_Cum",
    "Benefit share": "BenefitShare_Cum",
}
REGION_METRIC_TOGGLE_PAIRS: Dict[str, Tuple[str, str]] = {
    "Share_Cum": ("Share_Cum", "Share_Year"),
    "Share_Year": ("Share_Cum", "Share_Year"),
    "PerCap_Cum": ("PerCap_Cum", "PerCap_Year"),
    "PerCap_Year": ("PerCap_Cum", "PerCap_Year"),
    "BenefitShare_Cum": ("BenefitShare_Cum", "BenefitShare_Year"),
    "BenefitShare_Year": ("BenefitShare_Cum", "BenefitShare_Year"),
    "BenefitPerCap_Cum": ("BenefitPerCap_Cum", "BenefitPerCap_Year"),
    "BenefitPerCap_Year": ("BenefitPerCap_Cum", "BenefitPerCap_Year"),
}

REGION_METRICS_CACHE_VERSION = 2

_REGION_BOUNDS_EXCLUDE = {"area outside region"}


def _normalise_bound_key(value: Any) -> str:
    return str(value).strip().lower()


def _locations_for_bounds(locations: Iterable[str]) -> list[str]:
    ordered = [str(loc).strip() for loc in locations if str(loc).strip()]
    filtered = [loc for loc in ordered if _normalise_bound_key(loc) not in _REGION_BOUNDS_EXCLUDE]
    return filtered if filtered else ordered



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
    config = REGION_METRIC_CONFIG.get(metric_key, {})
    map_df = map_df.copy()
    map_df["_metric_value"] = pd.to_numeric(map_df.get("_metric_value"), errors="coerce")
    map_df["_has_data"] = map_df["_metric_value"].notna()
    map_df["_metric_value"] = map_df["_metric_value"].fillna(0.0)
    display_series = map_df["_metric_value"].apply(lambda v: _format_region_metric_value(metric_key, v))
    if map_df["_has_data"].isin([False]).any():
        display_series.loc[~map_df["_has_data"]] = "No data"
    map_df["_metric_display"] = display_series

    share_cfg = config.get("share_columns", {"cum": "Share_Cum", "year": "Share_Year"})
    cum_share_col = share_cfg.get("cum")
    year_share_col = share_cfg.get("year")

    def _share_series(column: Optional[str]) -> pd.Series:
        if column and column in map_df.columns:
            return pd.to_numeric(map_df[column], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=map_df.index)

    map_df["_share_cum_fmt"] = (_share_series(cum_share_col) * 100).map(lambda v: _format_percentage(v, 1))
    map_df["_share_year_fmt"] = (_share_series(year_share_col) * 100).map(lambda v: _format_percentage(v, 1))
    map_df["_percap_cum_fmt"] = (pd.to_numeric(map_df.get("PerCap_Cum", 0.0), errors="coerce").fillna(0.0) * 1_000_000).map(_format_currency_compact)
    map_df["_percap_year_fmt"] = (pd.to_numeric(map_df.get("PerCap_Year", 0.0), errors="coerce").fillna(0.0) * 1_000_000).map(_format_currency_compact)
    map_df.loc[~map_df["_has_data"], "_share_cum_fmt"] = '-'
    map_df.loc[~map_df["_has_data"], "_share_year_fmt"] = '-'
    map_df.loc[~map_df["_has_data"], "_percap_cum_fmt"] = '-'
    map_df.loc[~map_df["_has_data"], "_percap_year_fmt"] = '-'
    map_df["_population_fmt"] = map_df["population"].map(lambda v: f"{v:,.0f}" if np.isfinite(v) else "-")
    map_df["_year_str"] = map_df["Year"].astype(int).astype(str)

    values = map_df.loc[map_df["_has_data"], "_metric_value"].to_numpy(dtype=float)
    if values.size == 0:
        values = np.array([0.0], dtype=float)
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
    base_colorscale = _colorscale_with_zero_base(config.get("colorscale", PBI_SEQUENTIAL_SCALE), zero_color=REGION_MAP_ZERO_COLOR)
    effective_colorscale = _colorscale_with_opacity(base_colorscale, opacity)

    active_locations = [
        str(loc)
        for loc, value in zip(map_df["join_key"], map_df["_metric_value"])
        if value is not None and np.isfinite(value)
    ]
    focus_locations = _locations_for_bounds(active_locations or map_df["join_key"].tolist())

    share_prefix = "Benefit share" if (cum_share_col and cum_share_col.startswith("Benefit")) or (year_share_col and year_share_col.startswith("Benefit")) else "Share"

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
        f"{share_prefix} (cum): %{{customdata[3]}}<br>"
        f"{share_prefix} (annual): %{{customdata[4]}}<br>"
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
    z_series = map_df["_metric_value"].where(map_df["_has_data"], np.nan)
    trace = go.Choropleth(
        geojson=geojson,
        featureidkey=f"properties.{name_field}",
        locations=map_df["join_key"],
        z=z_series,
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
            text=f"{config['label']} - {scenario_label} ({year})",
            x=0.02,
            y=0.96,
            xanchor="left",
            yanchor="top",
            font=dict(size=16),
            pad=dict(b=12),
        ),
    )
    return fig



def render_region_map(
    df_year: pd.DataFrame,
    metric_key: str,
    *,
    scenario_label: str,
    year: int,
    show_borders: bool,
    fill_opacity: float,
) -> pd.DataFrame | None:
    geojson = fetch_region_geojson()
    name_field = get_geojson_name_field(geojson)
    valid_regions = set()
    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        value = props.get(name_field)
        if value is None:
            continue
        text_value = _canonical_join_key(value)
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
    df_year["join_key"] = df_year["join_key"].map(_canonical_join_key)
    df_year["region"] = df_year["region"].map(_canonical_join_key)
    map_df = df_year[df_year["join_key"].isin(valid_regions) & (df_year["join_key"].str.len() > 0)].copy()
    if map_df.empty or map_df[metric_key].dropna().empty:
        st.info("No mapped regional spend for the selected inputs.")
        summary = build_region_summary_table(df_year, metric_key, year=int(year))
        st.markdown("<div class='pbi-table region-summary-table'>", unsafe_allow_html=True)
        st.dataframe(summary, hide_index=True, width="stretch", height=420)
        st.markdown("</div>", unsafe_allow_html=True)
        return summary
    map_df["_metric_value"] = _scaled_region_metric(map_df, metric_key)
    if map_df["_metric_value"].dropna().empty:
        st.info("Selected metric has no values for this year.")
        summary = build_region_summary_table(df_year, metric_key, year=int(year))
        st.markdown("<div class='pbi-table region-summary-table'>", unsafe_allow_html=True)
        st.dataframe(summary, hide_index=True, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
        return summary
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
        summary = build_region_summary_table(df_year, metric_key, year=int(year))
        st.markdown(f"**Regions ({REGION_METRIC_CONFIG[metric_key]['label']})**")
        st.markdown("<div class='pbi-table region-summary-table'>", unsafe_allow_html=True)
        st.dataframe(summary, hide_index=True, width="stretch", height=420)
        st.markdown("</div>", unsafe_allow_html=True)
        if (df_year["region"] == "Unmapped").any():
            st.caption("Projects without a region mapping are grouped under 'Unmapped'.")
        return summary

def build_region_summary_table(df: pd.DataFrame, metric_key: str, *, year: int) -> pd.DataFrame:
    """
    Build the region summary table backing the map-side table and the
    server-rendered dataframe. Keeps labels consistent with the rest of the app
    and pre-formats numeric values. Styling (font/colours) is done by CSS.
    """
    config = REGION_METRIC_CONFIG.get(metric_key, {})

    # Working copy
    table = df.copy()

    # Metric value + display text
    raw_metric = pd.to_numeric(_scaled_region_metric(table, metric_key), errors="coerce")
    table["_has_data"] = raw_metric.notna()
    table["_metric_value"] = raw_metric.fillna(0.0)
    display_series = table["_metric_value"].apply(lambda v: _format_region_metric_value(metric_key, v))
    display_series.loc[~table["_has_data"]] = "No data"
    table["_metric_display"] = display_series

    # Which columns represent "share" so we can show both cumulative + annual
    share_cfg = config.get("share_columns", {"cum": "Share_Cum", "year": "Share_Year"})
    cum_share_col = share_cfg.get("cum")
    year_share_col = share_cfg.get("year")

    def _share_series(column: Optional[str]) -> pd.Series:
        if column and column in table.columns:
            return pd.to_numeric(table[column], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=table.index)

    # Pre-formatted share and per-cap columns (so they can be shown in a plain HTML table)
    table["_share_cum_fmt"] = (_share_series(cum_share_col) * 100).map(lambda v: _format_percentage(v, 1))
    table["_share_year_fmt"] = (_share_series(year_share_col) * 100).map(lambda v: _format_percentage(v, 1))
    table["_percap_cum_fmt"] = (
        pd.to_numeric(table.get("PerCap_Cum", 0.0), errors="coerce").fillna(0.0) * 1_000_000
    ).map(_format_currency_compact)
    table["_percap_year_fmt"] = (
        pd.to_numeric(table.get("PerCap_Year", 0.0), errors="coerce").fillna(0.0) * 1_000_000
    ).map(_format_currency_compact)
    table["_benefit_percap_cum_fmt"] = (
        pd.to_numeric(table.get("BenefitPerCap_Cum", 0.0), errors="coerce").fillna(0.0) * 1_000_000
    ).map(_format_currency_compact)
    table["_benefit_percap_year_fmt"] = (
        pd.to_numeric(table.get("BenefitPerCap_Year", 0.0), errors="coerce").fillna(0.0) * 1_000_000
    ).map(_format_currency_compact)
    table.loc[~table["_has_data"], ["_share_cum_fmt", "_share_year_fmt", "_percap_cum_fmt", "_percap_year_fmt"]] = '-'
    table.loc[~table["_has_data"], ["_benefit_percap_cum_fmt", "_benefit_percap_year_fmt"]] = '-'

    # Optional spend / benefit columns (shown when the metric group implies them)
    if "Spend_M" in table.columns:
        table["_spend_year_fmt"] = pd.to_numeric(table["Spend_M"], errors="coerce").fillna(0.0).map(format_currency)
    if "Spend_Cum_Region" in table.columns:
        table["_spend_cum_fmt"] = (
            pd.to_numeric(table["Spend_Cum_Region"], errors="coerce").fillna(0.0).map(format_currency)
        )

    show_benefit_columns = metric_key in REGION_METRIC_GROUPS.get("Benefit share", [])
    if show_benefit_columns:
        if "Benefit_Year" in table.columns:
            table["_benefit_year_fmt"] = (
                pd.to_numeric(table["Benefit_Year"], errors="coerce").fillna(0.0).map(format_currency)
            )
        if "Benefit_Cum_Region" in table.columns:
            table["_benefit_cum_fmt"] = (
                pd.to_numeric(table["Benefit_Cum_Region"], errors="coerce").fillna(0.0).map(format_currency)
            )
    if table["_has_data"].isin([False]).any():
        if "_spend_year_fmt" in table.columns:
            table.loc[~table["_has_data"], "_spend_year_fmt"] = "-"
        if "_spend_cum_fmt" in table.columns:
            table.loc[~table["_has_data"], "_spend_cum_fmt"] = "-"
        if "_benefit_year_fmt" in table.columns:
            table.loc[~table["_has_data"], "_benefit_year_fmt"] = "-"
        if "_benefit_cum_fmt" in table.columns:
            table.loc[~table["_has_data"], "_benefit_cum_fmt"] = "-"

    # Sorting behaviour per metric
    sort_mode = config.get("sort", "desc")
    if sort_mode == "asc":
        table = table.sort_values(["_has_data", "_metric_value"], ascending=[False, True])
    elif sort_mode == "abs_desc":
        table = table.assign(_abs=table["_metric_value"].abs()).sort_values(["_has_data", "_abs"], ascending=[False, False]).drop(columns="_abs")
    else:
        table = table.sort_values(["_has_data", "_metric_value"], ascending=[False, False])

    table.drop(columns=['_has_data'], inplace=True, errors='ignore')

    # Column labels
    share_prefix_is_benefit = (
        (cum_share_col and str(cum_share_col).startswith("Benefit"))
        or (year_share_col and str(year_share_col).startswith("Benefit"))
    )
    share_prefix = "Benefit share" if share_prefix_is_benefit else "Share"

    metric_label = config.get("table_label", config["label"])
    # Avoid confusing duplication if the metric *is* a share column already
    if metric_label in {"Share vs national (cum)", "Share vs national (annual)",
                        "Benefit share (cum)", "Benefit share (annual)"}:
        metric_label = f"{metric_label} value"

    table.drop(columns=['_has_data'], inplace=True, errors='ignore')

    columns = {
        "region": "Region",
        "_metric_display": metric_label,
        "_share_cum_fmt": f"{share_prefix} vs national (cum)",
        "_share_year_fmt": f"{share_prefix} vs national (annual)",
    }

    if share_cfg.get("cum") == metric_key:
        columns.pop("_share_cum_fmt", None)
    if share_cfg.get("year") == metric_key:
        columns.pop("_share_year_fmt", None)

    fy_label = f"FY {int(year)}"
    if "_spend_year_fmt" in table.columns:
        columns["_spend_year_fmt"] = f"Spend in {fy_label}"
    if "_spend_cum_fmt" in table.columns and not show_benefit_columns:
        columns["_spend_cum_fmt"] = f"Cumulative spend to {fy_label}"
    if show_benefit_columns and "_benefit_year_fmt" in table.columns:
        columns["_benefit_year_fmt"] = f"Benefit in {fy_label}"
    if show_benefit_columns and "_benefit_cum_fmt" in table.columns:
        columns["_benefit_cum_fmt"] = f"Cumulative benefit to {fy_label}"
    if show_benefit_columns and "BenefitPerCap_Cum" in df.columns:
        columns["_benefit_percap_cum_fmt"] = "Benefit per-cap cum"
    if show_benefit_columns and "BenefitPerCap_Year" in df.columns:
        columns["_benefit_percap_year_fmt"] = "Benefit per-cap annual"
    if "PerCap_Cum" in df.columns and not show_benefit_columns:
        columns["_percap_cum_fmt"] = "Per-cap cum"
    if "PerCap_Year" in df.columns and not show_benefit_columns:
        columns["_percap_year_fmt"] = "Per-cap annual"

    # Present only columns that actually exist
    present_keys = [key for key in columns if key in table.columns]
    result = table[present_keys].rename(columns=columns).reset_index(drop=True)

    # Return the full set of regions (16 official regions for the NZRC layer).
    return result




@lru_cache(maxsize=1)
def _region_map_template() -> str:
    template_path = Path(__file__).with_name("region_map_template.html")
    if not template_path.exists():
        raise FileNotFoundError(f"Region map template missing: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _linz_topographic_light_style(
    *,
    token: Optional[str],
    min_zoom: Optional[float],
    max_zoom: Optional[float],
) -> dict[str, Any]:
    """Create a lightweight LINZ topographic basemap style for MapLibre."""
    min_zoom_value = 0.0
    max_zoom_value = 14.0
    if min_zoom is not None and math.isfinite(min_zoom):
        min_zoom_value = max(0.0, float(min_zoom))
    if max_zoom is not None and math.isfinite(max_zoom):
        max_zoom_value = max(min_zoom_value, float(max_zoom))
    tile_url = LINZ_TOPO_VECTOR_TILE
    if token:
        tile_url = f"{tile_url}?api={token}"
    return {
        "version": 8,
        "sources": {
            "linz-topographic": {
                "type": "vector",
                "tiles": [tile_url],
                "minzoom": min_zoom_value,
                "maxzoom": max_zoom_value,
            }
        },
        "layers": [
            {
                "id": "background",
                "type": "background",
                "paint": {"background-color": "#f8fafc"},
            },
            {
                "id": "water-fill",
                "type": "fill",
                "source": "linz-topographic",
                "source-layer": "water",
                "paint": {"fill-color": "#bfdbfe", "fill-opacity": 0.85},
            },
            {
                "id": "landcover-fill",
                "type": "fill",
                "source": "linz-topographic",
                "source-layer": "landcover",
                "paint": {"fill-color": "#f1f5f9", "fill-opacity": 0.55},
            },
            {
                "id": "landuse-fill",
                "type": "fill",
                "source": "linz-topographic",
                "source-layer": "landuse",
                "paint": {"fill-color": "#e2e8f0", "fill-opacity": 0.45},
            },
            {
                "id": "transportation-case",
                "type": "line",
                "source": "linz-topographic",
                "source-layer": "transportation",
                "paint": {
                    "line-color": "#cbd5f5",
                    "line-width": [
                        "interpolate",
                        ["linear"],
                        ["zoom"],
                        5,
                        0.6,
                        10,
                        1.4,
                        14,
                        3.0,
                    ],
                    "line-opacity": 0.65,
                },
            },
            {
                "id": "transportation-line",
                "type": "line",
                "source": "linz-topographic",
                "source-layer": "transportation",
                "filter": [
                    "match",
                    ["get", "class"],
                    ["motorway", "trunk", "primary", "secondary"],
                    True,
                    False,
                ],
                "paint": {
                    "line-color": [
                        "match",
                        ["get", "class"],
                        "motorway",
                        "#1d4ed8",
                        "trunk",
                        "#2563eb",
                        "primary",
                        "#0ea5e9",
                        "secondary",
                        "#38bdf8",
                        "#64748b",
                    ],
                    "line-width": [
                        "interpolate",
                        ["linear"],
                        ["zoom"],
                        5,
                        0.4,
                        10,
                        1.1,
                        14,
                        2.4,
                    ],
                    "line-opacity": 0.85,
                },
            },
            {
                "id": "coastline",
                "type": "line",
                "source": "linz-topographic",
                "source-layer": "coastline",
                "paint": {"line-color": "#94a3b8", "line-width": 0.6},
            },
        ],
    }


def _metric_toggle_label(metric_key: str) -> str:
    """Return the label used in the map toggle for a metric key."""
    key_lower = metric_key.lower()
    if "cum" in key_lower:
        return "Show Cumulative"
    if "year" in key_lower:
        return "Show Annual"
    return REGION_METRIC_CONFIG.get(metric_key, {}).get("label", metric_key)


def _build_region_metric_bundle(
    metric_key: str,
    *,
    config: Dict[str, Any],
    year_contexts: Dict[int, Dict[str, pd.DataFrame]],
    locations: List[str],
    initial_year: int,
    opacity: float,
) -> tuple[Dict[str, Any], pd.DataFrame]:
    share_cfg = config.get("share_columns", {"cum": "Share_Cum", "year": "Share_Year"})
    cum_share_col = share_cfg.get("cum")
    year_share_col = share_cfg.get("year")
    feature_states: Dict[str, Dict[str, Dict[str, Any]]] = {}
    table_by_year: Dict[str, List[List[str]]] = {}
    table_headers: List[str] = []
    all_values: List[float] = []
    initial_table_df: pd.DataFrame | None = None
    fill_threshold = float(config.get("map_fill_threshold", 1e-9))
    colorscale = _colorscale_with_zero_base(
        config.get("colorscale", PBI_SEQUENTIAL_SCALE),
        zero_color=REGION_MAP_ZERO_COLOR,
    )
    effective_colorscale = _colorscale_with_opacity(colorscale, opacity)
    reversescale = bool(config.get("reversescale", False))

    def _share_series(column: Optional[str], frame: pd.DataFrame) -> pd.Series:
        if column and column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=frame.index)

    for year, ctx in year_contexts.items():
        df_year = ctx["frame"].copy()
        map_df = ctx["map"].copy()

        raw_metric = _scaled_region_metric(map_df, metric_key)
        map_df["_metric_value"] = pd.to_numeric(raw_metric, errors="coerce")
        map_df["_has_data"] = map_df["_metric_value"].notna()
        map_df["_metric_value"] = map_df["_metric_value"].fillna(0.0)
        map_df["_has_fill"] = map_df["_has_data"] & (map_df["_metric_value"].abs() > fill_threshold)

        display_series = map_df["_metric_value"].apply(
            lambda value: _format_region_metric_value(metric_key, value)
        )
        display_series.loc[~map_df["_has_data"]] = "No data"
        map_df["_metric_display"] = display_series

        map_df["_share_cum_fmt"] = (
            _share_series(cum_share_col, map_df) * 100.0
        ).map(lambda value: _format_percentage(value, 1))
        map_df["_share_year_fmt"] = (
            _share_series(year_share_col, map_df) * 100.0
        ).map(lambda value: _format_percentage(value, 1))
        map_df["_percap_cum_fmt"] = (
            pd.to_numeric(map_df.get("PerCap_Cum", 0.0), errors="coerce").fillna(0.0) * 1_000_000
        ).map(_format_currency_compact)
        map_df["_percap_year_fmt"] = (
            pd.to_numeric(map_df.get("PerCap_Year", 0.0), errors="coerce").fillna(0.0) * 1_000_000
        ).map(_format_currency_compact)
        map_df["_population_fmt"] = map_df["population"].map(
            lambda value: f"{value:,.0f}" if np.isfinite(value) else "-"
        )
        map_df.loc[
            ~map_df["_has_data"],
            ["_share_cum_fmt", "_share_year_fmt", "_percap_cum_fmt", "_percap_year_fmt"],
        ] = "-"

        by_key = map_df.set_index("join_key")
        year_states: Dict[str, Dict[str, Any]] = {}

        for loc in locations:
            if loc in by_key.index:
                row = by_key.loc[loc]
                raw_value = row.get("_metric_value")
                try:
                    value = float(raw_value) if raw_value is not None else None
                except (TypeError, ValueError):
                    value = None
                if value is not None and not math.isfinite(value):
                    value = None
                has_data = bool(row.get("_has_data", False) and value is not None)
                has_fill = bool(row.get("_has_fill", False) and value is not None)
                if has_fill and value is not None:
                    all_values.append(float(value))
                year_states[loc] = {
                    "value": value if has_data else None,
                    "hasData": has_data,
                    "hasFill": has_fill,
                    "region": str(row.get("region", loc)) or loc,
                    "year": int(year),
                    "metricDisplay": str(row.get("_metric_display", "-")),
                    "shareCumulative": str(row.get("_share_cum_fmt", "-")),
                    "shareAnnual": str(row.get("_share_year_fmt", "-")),
                    "perCapitaCumulative": str(row.get("_percap_cum_fmt", "-")),
                    "perCapitaAnnual": str(row.get("_percap_year_fmt", "-")),
                    "population": str(row.get("_population_fmt", "-")),
                }
            else:
                year_states[loc] = {
                    "value": None,
                    "hasData": False,
                    "hasFill": False,
                    "region": loc,
                    "year": int(year),
                    "metricDisplay": "No data",
                    "shareCumulative": "-",
                    "shareAnnual": "-",
                    "perCapitaCumulative": "-",
                    "perCapitaAnnual": "-",
                    "population": "-",
                }

        feature_states[str(year)] = year_states

        summary_df = build_region_summary_table(df_year.copy(), metric_key, year=int(year))
        if not table_headers:
            table_headers = [str(col) for col in summary_df.columns]
        table_by_year[str(year)] = summary_df.astype(str).values.tolist()
        if year == initial_year:
            initial_table_df = summary_df

    if config.get("type") == "diverging":
        max_abs = max((abs(value) for value in all_values), default=1.0)
        if not math.isfinite(max_abs) or max_abs <= 0:
            max_abs = 1.0
        domain_min, domain_max = -max_abs, max_abs
    else:
        if config.get("force_zero_min", False):
            domain_min = 0.0
        else:
            domain_min = min(all_values) if all_values else 0.0
            if not math.isfinite(domain_min):
                domain_min = 0.0
        domain_max = max(all_values) if all_values else (domain_min + 1.0)
        if not math.isfinite(domain_max) or math.isclose(domain_min, domain_max):
            domain_max = domain_min + max(1.0, abs(domain_min) * 0.1 + 1.0)

    span_entries: List[Tuple[float, str]] = []
    for stop, colour in effective_colorscale:
        try:
            fraction = float(stop)
        except (TypeError, ValueError):
            continue
        if reversescale:
            fraction = 1.0 - fraction
        span_entries.append((max(0.0, min(1.0, fraction)), colour))
    span_entries.sort()

    if math.isclose(domain_min, domain_max):
        domain_max = domain_min + 1.0

    span = domain_max - domain_min
    legend_stops: List[Tuple[float, str]] = [
        (domain_min + span * fraction, colour) for fraction, colour in span_entries
    ]

    share_prefix = (
        "Benefit share"
        if (
            (cum_share_col and str(cum_share_col).startswith("Benefit"))
            or (year_share_col and str(year_share_col).startswith("Benefit"))
        )
        else "Share"
    )

    bundle = {
        "metricKey": metric_key,
        "metricLabel": config.get("label", metric_key),
        "legend": {
            "stops": [[float(value), colour] for value, colour in legend_stops],
            "min": float(domain_min),
            "max": float(domain_max),
            "label": config.get("colorbar", config.get("label", metric_key)),
            "suffix": config.get("ticksuffix"),
            "format": config.get("tickformat"),
        },
        "featureStates": feature_states,
        "tableByYear": table_by_year,
        "tableHeaders": table_headers,
        "shareLabel": share_prefix,
    }

    if initial_table_df is None:
        initial_table_df = pd.DataFrame(columns=table_headers)

    return bundle, initial_table_df


def _prepare_region_reactive_payload(
    metrics_df: pd.DataFrame,
    metric_key: str,
    *,
    scenario_label: str,
    initial_year: int,
    show_borders: bool,
    fill_opacity: float,
    settings: Settings,
    basemap_mode: str,
    toggle_metric_keys: Iterable[str] | None = None,
) -> tuple[dict, pd.DataFrame]:
    geojson = fetch_region_geojson()
    name_field = get_geojson_name_field(geojson)

    locations: List[str] = []
    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        value = props.get(name_field)
        if value is None:
            continue
        text_value = _canonical_join_key(value)
        if text_value:
            locations.append(text_value)

    if not locations:
        fetch_region_geojson.cache_clear()
        geojson = fetch_region_geojson()
        name_field = get_geojson_name_field(geojson)
        locations = []
        for feature in geojson.get("features", []):
            props = feature.get("properties") or {}
            value = props.get(name_field)
            if value is None:
                continue
            text_value = str(value).strip()
            if text_value:
                locations.append(text_value)

    if not locations:
        raise ValueError(
            "GeoJSON has no joinable name field. Check ArcGIS out_fields or normalisation."
        )

    region_mapping = load_region_mapping()
    catalog, _, _ = region_baselines(region_mapping)
    baseline_keys = (
        catalog[["region", "join_key", "population", "gdp_per_capita"]]
        .drop_duplicates(subset=["region"])
        .copy()
    )
    baseline_keys["join_key"] = baseline_keys["join_key"].map(_canonical_join_key)
    baseline_keys["region"] = baseline_keys["region"].map(_canonical_join_key)

    working_df = metrics_df.copy()
    working_df["join_key"] = working_df["join_key"].map(_canonical_join_key)
    working_df["region"] = working_df["region"].map(_canonical_join_key)

    years = sorted(int(y) for y in working_df["Year"].dropna().unique())
    initial_year = int(initial_year)
    if initial_year not in years:
        years.append(initial_year)
        years.sort()

    year_contexts: Dict[int, Dict[str, pd.DataFrame]] = {}
    for year in years:
        df_year = working_df[working_df["Year"] == int(year)].copy()
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

        drop_cols = [col for col in ("population_data", "gdp_per_capita_data") if col in df_year.columns]
        if drop_cols:
            df_year.drop(columns=drop_cols, inplace=True)

        df_year["Year"] = int(year)
        numeric_cols = df_year.select_dtypes(include=[np.number]).columns
        df_year[numeric_cols] = df_year[numeric_cols].fillna(0.0)
        df_year["join_key"] = df_year["join_key"].map(_canonical_join_key)
        df_year["region"] = df_year["region"].map(_canonical_join_key)

        map_df = df_year[
            df_year["join_key"].isin(locations) & (df_year["join_key"].str.len() > 0)
        ].copy()

        year_contexts[int(year)] = {"frame": df_year, "map": map_df}

    opacity = float(np.clip(fill_opacity, 0.05, 1.0))
    dark_mode = is_dark_mode()
    line_color = "#93c5fd" if dark_mode else "#1f2937"
    border_width = 1.3 if show_borders else 0.6

    metric_order: List[str] = []
    candidates = [metric_key]
    if toggle_metric_keys:
        candidates.extend(toggle_metric_keys)
    for candidate in candidates:
        if candidate and candidate in REGION_METRIC_CONFIG and candidate not in metric_order:
            metric_order.append(candidate)
    if not metric_order:
        metric_order.append(metric_key)

    metric_bundles: Dict[str, Dict[str, Any]] = {}
    base_key = metric_order[0]
    initial_table_df: pd.DataFrame | None = None
    for key in metric_order:
        config = REGION_METRIC_CONFIG[key]
        bundle, bundle_table = _build_region_metric_bundle(
            key,
            config=config,
            year_contexts=year_contexts,
            locations=locations,
            initial_year=initial_year,
            opacity=opacity,
        )
        metric_bundles[key] = bundle
        if key == base_key:
            initial_table_df = bundle_table

    base_bundle = metric_bundles[base_key]

    def _clean(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _clean_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    toggle_metric_keys = REGION_METRIC_TOGGLE_PAIRS.get(metric_key)

    map_cfg = getattr(getattr(settings, "ui", None), "maplibre", None)
    style_light = _clean(getattr(map_cfg, "style_url", None)) or MAPLIBRE_FALLBACK_LIGHT
    style_dark = _clean(getattr(map_cfg, "dark_style_url", None)) or style_light or MAPLIBRE_FALLBACK_DARK
    style_light_base = style_light
    tile_template = _clean(getattr(map_cfg, "tile_url", None))
    light_style_url = _clean(getattr(map_cfg, "light_style_url", None))
    light_tile_url = _clean(getattr(map_cfg, "light_tile_url", None))
    terrain_label = _clean(getattr(map_cfg, "terrain_label", None)) or "Terrain imagery"
    light_label = _clean(getattr(map_cfg, "light_label", None)) or "Light basemap"
    style_token = _clean(getattr(map_cfg, "token", None))
    style_attr = _clean(getattr(map_cfg, "attribution", None)) or MAPLIBRE_FALLBACK_ATTRIBUTION
    map_min_zoom = _clean_float(getattr(map_cfg, "min_zoom", None))
    map_max_zoom = _clean_float(getattr(map_cfg, "max_zoom", None))
    map_hash_flag = bool(getattr(map_cfg, "hash", False))

    vector_light_style: Optional[dict[str, Any]] = None
    if style_token:
        vector_light_style = _linz_topographic_light_style(
            token=style_token,
            min_zoom=map_min_zoom,
            max_zoom=map_max_zoom,
        )

    selected_tile_template: Optional[str] = None
    selected_style_url: Optional[str] = None
    selected_style_object: Optional[dict[str, Any]] = None
    mode_lower = (basemap_mode or "").strip().lower()
    if mode_lower == "terrain" and tile_template:
        selected_tile_template = tile_template
    elif mode_lower == "light":
        if light_tile_url:
            selected_tile_template = light_tile_url
        elif light_style_url:
            selected_style_url = light_style_url
        elif vector_light_style is not None:
            selected_style_object = vector_light_style
    if (
        selected_tile_template is None
        and selected_style_url is None
        and selected_style_object is None
    ):
        if light_tile_url:
            selected_tile_template = light_tile_url
        elif light_style_url:
            selected_style_url = light_style_url
        elif vector_light_style is not None:
            selected_style_object = vector_light_style
        else:
            selected_style_url = style_light_base or MAPLIBRE_FALLBACK_LIGHT

    if selected_tile_template:
        map_payload = {
            "center": NZ_CENTER,
            "bounds": NZ_BOUNDS,
            "zoom": 4.6,
            "pitch": 0.0,
            "tileTemplate": selected_tile_template,
        }
    else:
        map_payload = {
            "center": NZ_CENTER,
            "bounds": NZ_BOUNDS,
            "zoom": 4.6,
            "pitch": 0.0,
        }
    if map_min_zoom is not None and math.isfinite(map_min_zoom):
        map_payload["minZoom"] = float(map_min_zoom)
    if map_max_zoom is not None and math.isfinite(map_max_zoom):
        map_payload["maxZoom"] = float(map_max_zoom)
    if map_hash_flag:
        map_payload["hash"] = True

    if selected_style_object is not None:
        style_light = selected_style_object
        style_dark = selected_style_object
    elif selected_style_url:
        style_light = selected_style_url
        style_dark = selected_style_url

    payload = {
        "title": base_bundle.get("metricLabel", base_key),
        "metricKey": base_key,
        "metricLabel": base_bundle.get("metricLabel", base_key),
        "scenarioLabel": scenario_label,
        "years": [int(y) for y in years],
        "initialYear": initial_year,
        "geojson": geojson,
        "nameField": name_field,
        "featureIds": locations,
        "featureStates": base_bundle["featureStates"],
        "tableByYear": base_bundle["tableByYear"],
        "tableHeaders": base_bundle["tableHeaders"],
        "legend": base_bundle["legend"],
        "style": {
            "light": style_light,
            "dark": style_dark,
            "token": style_token,
            "attribution": style_attr,
        },
        "map": map_payload,
        "fillOpacity": opacity,
        "lineColor": line_color,
        "borderWidth": float(border_width),
        "noDataColor": REGION_MAP_ZERO_COLOR,
        "shareLabel": base_bundle["shareLabel"],
        "theme": {"mode": "dark" if dark_mode else "light"},
        "metricBundles": metric_bundles,
    }

    if len(metric_order) > 1:
        payload["metricToggle"] = {
            "defaultKey": base_key,
            "options": [
                {"key": key, "label": _metric_toggle_label(key)}
                for key in metric_order
            ],
        }

    if initial_table_df is None:
        initial_table_df = pd.DataFrame(columns=base_bundle["tableHeaders"])

    return payload, initial_table_df


def render_region_map_reactive(
    metrics_df: pd.DataFrame,
    metric_key: str,
    *,
    scenario_label: str,
    initial_year: int,
    show_borders: bool,
    fill_opacity: float,
    settings: Settings,
    basemap_mode: str,
    toggle_metric_keys: Iterable[str] | None = None,
    key: Optional[str] = None,
) -> pd.DataFrame:
    payload, initial_table = _prepare_region_reactive_payload(
        metrics_df,
        metric_key,
        scenario_label=scenario_label,
        initial_year=int(initial_year),
        show_borders=show_borders,
        fill_opacity=fill_opacity,
        settings=settings,
        basemap_mode=basemap_mode,
        toggle_metric_keys=toggle_metric_keys,
    )

    def _sanitise(value: Any):
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, dict):
            return {k: _sanitise(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sanitise(v) for v in value]
        return value

    try:
        payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except ValueError:
        payload_json = json.dumps(_sanitise(payload), ensure_ascii=False, allow_nan=False)

    payload_json = payload_json.replace('</', '<' + chr(92) + '/')
    html = _region_map_template().replace('{DATA_JSON}', payload_json)

    widget_key = key or f"region_map_reactive_{metric_key}"
    state_key = "_region_map_html_supports_key"
    allow_key = st.session_state.get(state_key, True)
    html_kwargs = dict(height=720, scrolling=False)
    if allow_key:
        try:
            components.html(html, key=widget_key, **html_kwargs)
        except TypeError:
            st.session_state[state_key] = False
            components.html(html, **html_kwargs)
    else:
        components.html(html, **html_kwargs)
    return initial_table


def render_region_map_controls(metrics_df: pd.DataFrame, scenario_label: str, *, settings: Settings) -> pd.DataFrame | None:
    metrics_df = metrics_df.copy()
    population_series = pd.to_numeric(metrics_df.get("population"), errors="coerce")
    benefit_year_series = pd.to_numeric(metrics_df.get("Benefit_Year"), errors="coerce").fillna(0.0)
    benefit_cum_series = pd.to_numeric(metrics_df.get("Benefit_Cum_Region"), errors="coerce").fillna(0.0)
    denom = population_series.to_numpy(dtype=float)
    metrics_df["BenefitPerCap_Year"] = np.divide(
        benefit_year_series.to_numpy(dtype=float),
        denom,
        out=np.full(len(metrics_df), np.nan),
        where=denom > 0,
    )
    metrics_df["BenefitPerCap_Cum"] = np.divide(
        benefit_cum_series.to_numpy(dtype=float),
        denom,
        out=np.full(len(metrics_df), np.nan),
        where=denom > 0,
    )

    available_years = sorted(int(y) for y in metrics_df["Year"].dropna().unique())
    if not available_years:
        st.info("No spend data available for the selected scenario.")
        return None

    # Mode group (Spend share vs Benefit share)
    mode_options = list(REGION_METRIC_GROUPS.keys())
    default_mode = st.session_state.get("region_heatmap_mode", mode_options[0])
    if default_mode not in mode_options:
        default_mode = mode_options[0]
    st.session_state.setdefault("region_heatmap_mode", default_mode)

    selected_mode = st.radio(
        "Heatmap focus",
        mode_options,
        index=mode_options.index(st.session_state["region_heatmap_mode"]),
        horizontal=True,
        key="region_heatmap_mode",
        label_visibility="collapsed",
    )
    st.session_state.setdefault("region_heatmap_percap_toggle", False)
    per_capita_heatmap_enabled = st.toggle(
        "Colour heatmap by per-capita values",
        value=bool(st.session_state.get("region_heatmap_percap_toggle", False)),
        key="region_heatmap_percap_toggle",
        help=(
            "When enabled, the heatmap colour scale uses per-person values for the selected focus "
            "(spend or benefit). Turn off to colour by share vs national."
        ),
    )

    # Metric selection UI
    initial_year = int(st.session_state.get("region_metric_year", available_years[0]))
    if initial_year not in available_years:
        initial_year = available_years[0]

    if selected_mode == "Benefit share":
        metric_key = "BenefitPerCap_Cum" if per_capita_heatmap_enabled else "BenefitShare_Cum"
    else:
        metric_key = "PerCap_Cum" if per_capita_heatmap_enabled else "Share_Cum"
    st.session_state["region_metric_key"] = metric_key

    toggle_metric_keys = REGION_METRIC_TOGGLE_PAIRS.get(metric_key)

    map_cfg = getattr(getattr(settings, "ui", None), "maplibre", None)
    terrain_label = getattr(map_cfg, "terrain_label", None) or "Terrain imagery"
    light_label = getattr(map_cfg, "light_label", None) or "Simplified basemap"
    terrain_available = bool(getattr(map_cfg, "tile_url", None))
    st.session_state.setdefault("region_basemap_mode", "light")

    if terrain_available:
        previous_mode = st.session_state.get("region_basemap_mode", "light")
        help_text = f"Turn off to switch to the {light_label.lower()}."
        show_terrain = st.toggle(
            f"{terrain_label} background",
            value=(previous_mode == "terrain"),
            key="region_basemap_toggle",
            help=help_text,
        )
        basemap_mode = "terrain" if show_terrain else "light"
    else:
        basemap_mode = "light"

    st.session_state["region_basemap_mode"] = basemap_mode
    if terrain_available:
        active_background = terrain_label if basemap_mode == "terrain" else light_label
        st.caption(f"Background: {active_background}")
    # Reactive map + table (client-driven)
    initial_table = render_region_map_reactive(
        metrics_df,
        metric_key,
        scenario_label=scenario_label,
        initial_year=initial_year,
        show_borders=False,
        fill_opacity=1.0,
        settings=settings,
        basemap_mode=basemap_mode,
        toggle_metric_keys=toggle_metric_keys,
        key=f"reactive_map_{selected_mode}_{metric_key}",
    )

    # Keep the server-side notion of the "current" year around for exports.
    st.session_state["region_metric_year"] = int(initial_year)

    reconciliation_slice = metrics_df.loc[metrics_df["Year"] == int(initial_year)]
    if not reconciliation_slice.empty:
        row = reconciliation_slice.iloc[0]
        included_cum = float(pd.to_numeric(row.get("Spend_Cum_National"), errors="coerce"))
        full_cum = float(pd.to_numeric(row.get("Spend_Full_Cum"), errors="coerce"))
        excluded_cum = float(pd.to_numeric(row.get("Spend_Excluded_Cum"), errors="coerce"))
        spread_flag = bool(row.get("Spread_National_Enabled", True))
        message = (
            f"Reconciliation (FY {int(initial_year)}): included cumulative {format_currency(included_cum)} "
            f"vs full cumulative {format_currency(full_cum)}."
        )
        if not spread_flag:
            message += f" Excluded national/unmapped cumulative: {format_currency(excluded_cum)}."
        st.caption(message)

    return initial_table if initial_table is not None and not initial_table.empty else None




def render_region_tab(
    data: DashboardData,
    *,
    settings: Settings,
    opt_selection,
    comp_selection,
    opt_label: str,
    cmp_label: str,
    cache_signature: tuple | None = None,
) -> Dict[str, pd.DataFrame]:
    export_tables: Dict[str, pd.DataFrame] = {}
    st.markdown('<div class="pbi-section-title">Regional investment overview</div>', unsafe_allow_html=True)
    scenario_options: Dict[str, Any] = {}
    if getattr(opt_selection, "code", None):
        scenario_options[opt_label] = opt_selection
    if getattr(comp_selection, "code", None):
        scenario_options[cmp_label] = comp_selection
    if not scenario_options:
        st.info("Select a scenario to view the regional investment map.")
        return export_tables
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
        return export_tables
    st.session_state.setdefault("region_spread_national_toggle", True)
    spread_national_enabled = st.toggle(
        "Spread national numbers across regions",
        value=bool(st.session_state.get("region_spread_national_toggle", True)),
        key="region_spread_national_toggle",
        help=(
            "When enabled, national and unmapped project totals are distributed across regions "
            "using national allocation weights. When disabled, they are excluded from regional totals."
        ),
    )
    cache_bucket = st.session_state.setdefault("_region_metrics_cache", {})
    if st.session_state.get("_region_metrics_cache_version") != REGION_METRICS_CACHE_VERSION:
        cache_bucket.clear()
        st.session_state["_region_metrics_cache_version"] = REGION_METRICS_CACHE_VERSION
    cache_key = (
        cache_signature or "default",
        scenario_code,
        bool(spread_national_enabled),
        bool(spread_national_enabled),
    )
    metrics_df = cache_bucket.get(cache_key)
    if isinstance(metrics_df, pd.DataFrame):
        cached_version = metrics_df.attrs.get("cache_version")
        if cached_version != REGION_METRICS_CACHE_VERSION:
            metrics_df = None
            cache_bucket.pop(cache_key, None)
    def _needs_refresh(df: pd.DataFrame | None) -> bool:
        if not isinstance(df, pd.DataFrame):
            return True
        if df.empty:
            return True
        if not {"BenefitShare_Cum", "BenefitShare_Year"}.intersection(df.columns):
            return True
        benefit_cols = [
            col
            for col in ("BenefitShare_Cum", "BenefitShare_Year", "Benefit_Cum_Region", "Benefit_Year")
            if col in df.columns
        ]
        if benefit_cols:
            totals: List[float] = []
            for col in benefit_cols:
                series = pd.to_numeric(df[col], errors="coerce")
                series = series.fillna(0.0)
                totals.append(float(series.abs().sum()))
            benefit_total = max(totals) if totals else 0.0
            if not np.isfinite(benefit_total):
                return True
            if math.isclose(benefit_total, 0.0, abs_tol=1e-6):
                spend_cols = [
                    col for col in ("Spend_National", "Spend_Cum_Region", "Spend_Year") if col in df.columns
                ]
                spend_total = 0.0
                if spend_cols:
                    spend_totals: List[float] = []
                    for col in spend_cols:
                        series = pd.to_numeric(df[col], errors="coerce")
                        series = series.fillna(0.0)
                        spend_totals.append(float(series.abs().sum()))
                    spend_total = max(spend_totals) if spend_totals else 0.0
                if spend_total > 0.0:
                    return True
        return False
    if metrics_df is None:
        try:
            metrics_df = compute_region_metrics(
                data,
                scenario_code,
                spread_national=bool(spread_national_enabled),
                spread_unmapped=bool(spread_national_enabled),
            )
        except Exception as exc:
            st.warning(f"Unable to compute regional metrics for {selected_label}: {exc}")
            return export_tables
        if isinstance(metrics_df, pd.DataFrame):
            metrics_df.attrs["cache_version"] = REGION_METRICS_CACHE_VERSION
            cache_bucket[cache_key] = metrics_df
    elif _needs_refresh(metrics_df):
        try:
            metrics_df = compute_region_metrics(
                data,
                scenario_code,
                spread_national=bool(spread_national_enabled),
                spread_unmapped=bool(spread_national_enabled),
            )
        except Exception as exc:
            st.warning(f"Unable to refresh regional metrics for {selected_label}: {exc}")
            return export_tables
        if isinstance(metrics_df, pd.DataFrame):
            metrics_df.attrs["cache_version"] = REGION_METRICS_CACHE_VERSION
            cache_bucket[cache_key] = metrics_df
    summary = render_region_map_controls(metrics_df, selected_label, settings=settings)
    if summary is not None and not summary.empty:
        export_tables[f"Regional summary - {selected_label}"] = summary
    return export_tables

def render_overview_tab(
    data: DashboardData,
    opt_selection,
    comp_selection,
    opt_series,
    cmp_series,
    *,
    opt_label: str,
    cmp_label: str,
    npv_label: str,
) -> Dict[str, pd.DataFrame]:
    export_tables: Dict[str, pd.DataFrame] = {}

    profile_assets, metadata = load_interpolated_profile_assets()
    load_error = metadata.get("load_error")

    if not profile_assets:
        if load_error:
            st.warning(f"Unable to load interpolated profiles: {load_error}")
        else:
            st.info(
                "Interpolated profile animation is unavailable. "
                f"Place `{INTERPOLATED_PROFILES_PATH.name}` in `{INTERPOLATED_PROFILES_PATH.parent}` to enable it."
            )
        return export_tables

    playback_order: List[int] = list(metadata.get("profile_order") or sorted(profile_assets))
    if not playback_order:
        st.info("No interpolated profiles were found in the supplied data.")
        return export_tables

    total_profiles = len(playback_order)
    first_asset = profile_assets[playback_order[0]]
    final_asset = profile_assets[playback_order[-1]]
    baseline_npv_value = _safe_float(first_asset.get("npv"))

    final_chart_frame = final_asset["chart_frame"]
    spend_actual = (final_chart_frame["Spend"].astype(float).to_numpy() * 1_000_000.0) if not final_chart_frame.empty else np.array([0.0])
    closing_actual = (final_chart_frame["ClosingNet"].astype(float).to_numpy() * 1_000_000.0) if not final_chart_frame.empty else np.array([0.0])
    envelope_actual = (final_chart_frame["Envelope"].astype(float).to_numpy() * 1_000_000.0) if not final_chart_frame.empty else np.array([0.0])
    combined_values = np.concatenate([spend_actual, closing_actual, envelope_actual]) if spend_actual.size or closing_actual.size or envelope_actual.size else np.array([0.0])
    finite_values = combined_values[np.isfinite(combined_values)]
    if finite_values.size == 0:
        finite_values = np.array([0.0])
    fixed_tick_vals, fixed_tick_text, _ = compute_cash_axis_ticks(finite_values)
    if not fixed_tick_vals:
        fixed_tick_vals = [0.0]
        fixed_tick_text = ["0"]
    fixed_y_min = float(min(fixed_tick_vals))
    fixed_y_max = float(max(fixed_tick_vals))
    fixed_y_axis_payload = {
        "values": fixed_tick_vals,
        "labels": fixed_tick_text,
        "range": [fixed_y_min, fixed_y_max],
    }
    final_years = final_chart_frame["Year"].dropna().astype(int).tolist()
    if final_years:
        fixed_x_min = int(min(final_years))
        fixed_x_max = int(max(final_years))
    else:
        fixed_x_min = 0
        fixed_x_max = 1
    fixed_x_axis_payload = [fixed_x_min, fixed_x_max]

    export_tables[f"Cash flow - {final_asset['label']}"] = final_asset["export"].copy()

    st.markdown(
        '<div class="pbi-section-title">Cash flow optimiser</div>',
        unsafe_allow_html=True,
    )

    component_id = f"interpolated_profiles_{uuid4().hex}"
    metrics_grid_token = uuid4().hex[:8]
    metrics_anim_name = f"kpiSweep_{metrics_grid_token}"

    progress_card_html = _kpi_card_html(
        "Optimiser progress",
        None,
        body_html=(
            f'<div class="optimiser-progress">'
            f'  <div class="optimiser-progress__track">'
            f'    <div id="{component_id}-progress-fill" class="optimiser-progress__fill"></div>'
            f'  </div>'
            f'  <div id="{component_id}-progress-label" class="optimiser-progress__label">0%</div>'
            f'</div>'
        ),
    )
    progress_card_html = progress_card_html.replace(
        '<div class="kpi-card">',
        '<div class="kpi-card kpi-sweep-on profile-anim__metric profile-anim__metric--progress">',
        1,
    )

    npv_card_html = _kpi_card_html(
        f"Optimiser {npv_label}",
        value=f'<span id="{component_id}-npv-value">--</span>',
        delta_text="Dartboard delivery plan",
        delta_state="neutral",
    )
    npv_card_html = npv_card_html.replace(
        '<div class="kpi-card">',
        '<div class="kpi-card kpi-sweep-on profile-anim__metric profile-anim__metric--npv">',
        1,
    )
    npv_card_html = npv_card_html.replace(
        '<div class="kpi-delta neutral">',
        f'<div id="{component_id}-npv-delta" class="kpi-delta neutral">',
        1,
    )
    frame_duration_ms = int(
        INTERPOLATED_PROFILE_ANIMATION_SECONDS * 1000.0 / max(total_profiles - 1, 1)
    )
    transition_ms = 80

    data_sequence: List[Dict[str, Any]] = []
    for idx in playback_order:
        asset = profile_assets[idx]
        chart_frame = asset["chart_frame"]
        npv_value = _safe_float(asset.get("npv"))
        data_sequence.append(
            {
                "label": asset["label"],
                "title": "",
                "years": [int(year) if pd.notna(year) else None for year in chart_frame["Year"].tolist()],
                "spend": [float(val) for val in (chart_frame["Spend"] * 1_000_000.0).tolist()],
                "closing": [float(val) for val in (chart_frame["ClosingNet"] * 1_000_000.0).tolist()],
                "envelope": [float(val) for val in (chart_frame["Envelope"] * 1_000_000.0).tolist()],
                "progress": asset.get("progress", None),
                "npv": npv_value,
            }
        )

    data_sequence_json = json.dumps(data_sequence)
    baseline_npv_json = json.dumps(baseline_npv_value)
    npv_label_html = html.escape(npv_label)
    fixed_y_axis_json = json.dumps(fixed_y_axis_payload)
    fixed_x_axis_json = json.dumps(fixed_x_axis_payload)

    component_html = f"""
    <style>
      #{component_id} {{
        --pbi-blue: {POWERBI_BLUE};
        --pbi-green: {POWERBI_GREEN};
        --kpi-border: rgba(25,69,107,.18);
        --kpi-shadow: 0 10px 24px rgba(25,69,107,.08);
        --kpi-text-1: #0F172A;
        --kpi-text-2: #475569;
        --kpi-top-grey: rgba(148,163,184,.45);
        display: flex;
        flex-direction: column;
        gap: 18px;
        padding-bottom: 120px;
      }}
      #{component_id} .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 14px;
        padding-bottom: 18px;
      }}
      #{component_id} .kpi-card {{
        position: relative;
        background: #fff;
        border: 1px solid var(--kpi-border);
        border-radius: 18px;
        box-shadow: var(--kpi-shadow);
        padding: 16px 18px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        gap: 12px;
      }}
      #{component_id} .kpi-card::before {{
        content:"";
        position:absolute; left:14px; right:14px; top:0;
        height:6px; border-radius:999px;
        background: var(--kpi-top-grey);
        transform: translateY(-3px);
        z-index: 1;
      }}
      #{component_id} .kpi-card::after {{
        content:"";
        position:absolute; left:14px; right:14px; top:0;
        height:6px; border-radius:999px;
        background: linear-gradient(90deg, var(--pbi-green), var(--kpi-top-grey));
        transform: translateY(-3px) scaleX(0);
        transform-origin: right center;
        z-index: 2;
        animation: none;
      }}
      @keyframes {metrics_anim_name} {{
        0% {{ transform: translateY(-3px) scaleX(0); }}
        100% {{ transform: translateY(-3px) scaleX(1); }}
      }}
      #{component_id} [data-kpi-grid-id="{metrics_grid_token}"] .kpi-card::after {{
        animation: {metrics_anim_name} 0.55s ease-out forwards;
        animation-delay: var(--kpi-sweep-delay, 0s);
      }}
      #{component_id} .profile-anim__chart-card {{
        border-radius: 18px;
        background: var(--cp-surface-soft, rgba(255,255,255,0.92));
        border: 1px solid var(--cp-outline, rgba(25,69,107,0.18));
        padding: 16px 18px 12px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12);
        margin-bottom: 120px;
      }}
      #{component_id} .profile-anim__metrics {{
        display: grid;
        gap: 16px;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      }}
      #{component_id} .profile-anim__metric {{
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 12px;
      }}
      #{component_id} .kpi-title {{
        color: var(--pbi-blue);
        font-family: 'Segoe UI', 'Inter', sans-serif;
        font-weight: 600;
        font-size: .95rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin: 0 0 .25rem 0;
      }}
      #{component_id} .kpi-value {{
        color: var(--kpi-text-1);
        font-family: 'Segoe UI', 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.9rem;
        line-height: 1.1;
        letter-spacing: -0.01em;
      }}
      #{component_id} .kpi-delta {{
        display:inline-flex; align-items:center; gap:6px;
        margin-top:.55rem;
        font-family: 'Segoe UI', 'Inter', sans-serif;
        font-weight:600; font-size:.85rem;
        padding:3px 10px; border-radius:999px;
        border:1px solid transparent;
      }}
      #{component_id} .kpi-delta.up {{
        color: var(--pbi-green);
        background: rgba(175,189,34,.12);
        border-color: rgba(175,189,34,.35);
      }}
      #{component_id} .kpi-delta.down {{
        color: #CA4142;
        background: rgba(202,65,66,.10);
        border-color: rgba(202,65,66,.30);
      }}
      #{component_id} .kpi-delta.neutral {{
        color: var(--pbi-blue);
        background: rgba(25,69,107,.10);
        border-color: rgba(25,69,107,.28);
      }}
      #{component_id} .optimiser-progress {{
        margin-top: 0.75rem;
        display: flex;
        align-items: center;
        gap: 14px;
      }}
      #{component_id} .optimiser-progress__track {{
        position: relative;
        flex: 1;
        height: 16px;
        border-radius: 999px;
        border: 2px solid rgba(92, 92, 92, 0.85);
        background: #f3f3f3;
        overflow: hidden;
      }}
      #{component_id} .optimiser-progress__fill {{
        position: absolute;
        inset: 0;
        width: 0%;
        background: rgba(160, 160, 160, 0.95);
        border-radius: inherit;
        transition: width 0.28s ease-out;
      }}
      #{component_id} .optimiser-progress__label {{
        font-family: 'Segoe UI', 'Inter', sans-serif;
        font-weight: 700;
        color: rgba(72, 72, 72, 0.95);
        min-width: 60px;
        text-align: right;
      }}
      #{component_id} .profile-anim__controls {{
        display: flex;
        justify-content: flex-end;
        margin-bottom: 10px;
      }}
      #{component_id} .profile-anim__button {{
        background: var(--cp-primary, #19456b);
        color: #fff;
        border: none;
        border-radius: 999px;
        padding: 8px 20px;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease;
        margin-left: auto;
      }}
      #{component_id} .profile-anim__button:hover:not([disabled]) {{
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(25, 69, 107, 0.26);
      }}
      #{component_id} .profile-anim__button[disabled] {{
        opacity: 0.6;
        cursor: not-allowed;
        box-shadow: none;
      }}
      #{component_id} .profile-anim__chart {{
        min-height: 540px;
      }}
    </style>
    <div id="{component_id}" class="profile-anim">
      <div class="profile-anim__metrics kpi-grid" data-kpi-grid-id="{metrics_grid_token}">
        {progress_card_html}
        {npv_card_html}
      </div>
      <div class="profile-anim__chart-card">
        <div class="profile-anim__controls">
          <button id="{component_id}-button" class="profile-anim__button" type="button">Run Optimiser</button>
        </div>
        <div id="{component_id}-chart" class="profile-anim__chart"></div>
      </div>
    </div>
    <script>
      (function() {{
        const PLOTLY_URL = "https://cdn.plot.ly/plotly-2.27.0.min.js";
        const sequence = {data_sequence_json};
        const totalFrames = sequence.length;
        const frameDuration = {frame_duration_ms};
        const transitionDuration = {transition_ms};
        const FIXED_Y_AXIS = {fixed_y_axis_json};
        const FIXED_X_RANGE = {fixed_x_axis_json};
        const component = document.getElementById("{component_id}");
        if (!component) {{
          return;
        }}
        const chart = document.getElementById("{component_id}-chart");
        const progressFillEl = document.getElementById("{component_id}-progress-fill");
        const progressLabelEl = document.getElementById("{component_id}-progress-label");
        const button = document.getElementById("{component_id}-button");
        const npvValueEl = document.getElementById("{component_id}-npv-value");
        const npvDeltaEl = document.getElementById("{component_id}-npv-delta");
        const baselineNpv = {baseline_npv_json};

        const COLORS = {{
          primary: "{PRIMARY_COLOR}",
          closing: "{CLOSING_NET_COLOR}",
          envelope: "{ENVELOPE_COLOR}",
        }};
        const numberFormatter0 = new Intl.NumberFormat(undefined, {{maximumFractionDigits: 0}});
        const TRACE_NAME_TO_KEY = {{
          "Annual spend": "spend",
          Envelope: "envelope",
          "Closing net": "closing",
        }};
        const REVEAL_SEQUENCE = ["spend", "envelope", "closing"];
        const revealState = {{
          spend: false,
          envelope: false,
          closing: false,
        }};
        const appliedVisibility = {{
          spend: false,
          envelope: false,
          closing: false,
        }};
        let revealIndex = 0;
        let revealClickHandlerAttached = false;
        let revealAnimating = false;
        let currentFrameIndex = 0;
        const REVEAL_ANIMATION_DURATION = 400;

        function formatLargeAmount(value) {{
          const absValue = Math.abs(value);
          if (absValue >= 1_000_000_000) {{
            return (value / 1_000_000_000).toFixed(1) + "B";
          }}
          if (absValue >= 1_000_000) {{
            return (value / 1_000_000).toFixed(1) + "M";
          }}
          if (absValue >= 1_000) {{
            return (value / 1_000).toFixed(1) + "K";
          }}
          return value.toFixed(0);
        }}

        function computeAxisTicks() {{
          return FIXED_Y_AXIS;
        }}

        function easeOutCubic(t) {{
          const clamped = Math.min(Math.max(t, 0), 1);
          return 1 - Math.pow(1 - clamped, 3);
        }}

        function getSeriesFromEntry(key, entry) {{
          if (!entry) {{
            return [];
          }}
          if (key === "spend") {{
            return Array.isArray(entry.spend) ? entry.spend.slice() : [];
          }}
          if (key === "envelope") {{
            return Array.isArray(entry.envelope) ? entry.envelope.slice() : [];
          }}
          if (key === "closing") {{
            return Array.isArray(entry.closing) ? entry.closing.slice() : [];
          }}
          return [];
        }}

        function getCurrentEntry() {{
          return sequence[currentFrameIndex] || sequence[0] || null;
        }}

        async function animateTrace(key, traceIndex, entry) {{
          if (!chart || !chart.data || !chart.data[traceIndex]) {{
            await Plotly.restyle(chart, {{visible: true}}, [traceIndex]);
            return;
          }}
          const target = getSeriesFromEntry(key, entry);
          if (!Array.isArray(target) || !target.length) {{
            await Plotly.restyle(chart, {{visible: true}}, [traceIndex]);
            return;
          }}
          const duration = REVEAL_ANIMATION_DURATION;
          if (key === "spend") {{
            const baseline = target.map(() => 0);
            await Plotly.restyle(chart, {{visible: true, y: [baseline]}}, [traceIndex]);
            await new Promise((resolve) => {{
              const start = performance.now();
              function step(now) {{
                const elapsed = now - start;
                const progress = easeOutCubic(elapsed / duration);
                const values = target.map((val) => val * progress);
                chart.data[traceIndex].y = elapsed >= duration ? target.slice() : values;
                Plotly.redraw(chart);
                if (elapsed >= duration) {{
                  resolve();
                }} else {{
                  requestAnimationFrame(step);
                }}
              }}
              requestAnimationFrame(step);
            }});
            return;
          }}
          const initial = target.map(() => null);
          await Plotly.restyle(chart, {{visible: true, y: [initial]}}, [traceIndex]);
          await new Promise((resolve) => {{
            const start = performance.now();
            const lastIndex = target.length - 1;
            function step(now) {{
              const elapsed = now - start;
              if (elapsed >= duration) {{
                chart.data[traceIndex].y = target.slice();
                Plotly.redraw(chart);
                resolve();
                return;
              }}
              const eased = easeOutCubic(elapsed / duration);
              if (target.length === 1) {{
                chart.data[traceIndex].y = [target[0] * eased];
                Plotly.redraw(chart);
                requestAnimationFrame(step);
                return;
              }}
              const limitPos = eased * lastIndex;
              const baseIndex = Math.floor(limitPos);
              const remainder = limitPos - baseIndex;
              const values = new Array(target.length).fill(null);
              for (let idx = 0; idx <= baseIndex && idx < target.length; idx += 1) {{
                values[idx] = target[idx];
              }}
              const nextIndex = Math.min(lastIndex, baseIndex + 1);
              if (nextIndex > baseIndex && nextIndex < target.length) {{
                const startVal = target[baseIndex];
                const endVal = target[nextIndex];
                if (Number.isFinite(startVal) && Number.isFinite(endVal)) {{
                  values[nextIndex] = startVal + (endVal - startVal) * remainder;
                }} else {{
                  values[nextIndex] = endVal;
                }}
              }}
              chart.data[traceIndex].y = values;
              Plotly.redraw(chart);
              requestAnimationFrame(step);
            }}
            requestAnimationFrame(step);
          }});
        }}

        function formatNpvValue(value) {{
          if (!Number.isFinite(value)) {{
            return "--";
          }}
          return `${{numberFormatter0.format(value)}} m`;
        }}

        function updateNpvCard(entry) {{
          if (!npvValueEl && !npvDeltaEl) {{
            return;
          }}
          const npvValue = Number(entry && entry.npv);
          if (npvValueEl) {{
            npvValueEl.textContent = formatNpvValue(npvValue);
          }}
          if (!npvDeltaEl) {{
            return;
          }}
          const baseline =
            typeof baselineNpv === "number" && Number.isFinite(baselineNpv)
              ? baselineNpv
              : NaN;
          npvDeltaEl.classList.remove("up", "down", "neutral");
          if (Number.isFinite(npvValue) && Number.isFinite(baseline)) {{
            const delta = npvValue - baseline;
            if (Math.abs(delta) < 1e-6) {{
            npvDeltaEl.textContent = "Dartboard delivery plan";
              npvDeltaEl.classList.add("neutral");
            }} else {{
              const magnitude = Math.abs(delta);
              const sign = delta > 0 ? "+" : "-";
              npvDeltaEl.textContent = `${{sign}}${{numberFormatter0.format(magnitude)}} m vs start`;
              npvDeltaEl.classList.add(delta > 0 ? "up" : "down");
            }}
          }} else if (Number.isFinite(npvValue)) {{
            npvDeltaEl.textContent = "NPV data available";
            npvDeltaEl.classList.add("neutral");
          }} else {{
            npvDeltaEl.textContent = "No NPV data";
            npvDeltaEl.classList.add("neutral");
          }}
        }}

        function findTraceIndex(key) {{
          if (!chart || !key) {{
            return null;
          }}
          const traces = chart.data || [];
          for (let i = 0; i < traces.length; i += 1) {{
            const trace = traces[i];
            const traceKey =
              (trace && trace.meta && trace.meta.revealKey) ||
              TRACE_NAME_TO_KEY[trace && trace.name ? trace.name : ""];
            if (traceKey === key) {{
              return i;
            }}
          }}
          return null;
        }}

        function applyRevealState(options = {{}}) {{
          if (!chart) {{
            return Promise.resolve();
          }}
          const {{ animateNew = false, entry = getCurrentEntry() }} = options;
          const tasks = [];
          REVEAL_SEQUENCE.forEach((key) => {{
            const traceIndex = findTraceIndex(key);
            if (traceIndex === null || traceIndex === undefined) {{
              return;
            }}
            if (revealState[key]) {{
              if (!appliedVisibility[key]) {{
                appliedVisibility[key] = true;
                if (animateNew) {{
                  tasks.push(animateTrace(key, traceIndex, entry));
                }} else {{
                  tasks.push(Plotly.restyle(chart, {{visible: true}}, [traceIndex]));
                }}
              }} else {{
                tasks.push(Plotly.restyle(chart, {{visible: true}}, [traceIndex]));
              }}
            }} else {{
              appliedVisibility[key] = false;
              tasks.push(Plotly.restyle(chart, {{visible: "legendonly"}}, [traceIndex]));
            }}
          }});
          return Promise.all(tasks);
        }}

        async function handleRevealClick(event) {{
          if (!(event instanceof MouseEvent) || event.button !== 0) {{
            return;
          }}
          if (revealAnimating) {{
            return;
          }}
          if (revealIndex >= REVEAL_SEQUENCE.length) {{
            if (revealClickHandlerAttached) {{
              chart.removeEventListener("click", handleRevealClick);
              revealClickHandlerAttached = false;
            }}
            return;
          }}
          const key = REVEAL_SEQUENCE[revealIndex];
          revealState[key] = true;
          revealIndex += 1;
          revealAnimating = true;
          try {{
            await applyRevealState({{ animateNew: true, entry: getCurrentEntry() }});
          }} finally {{
            revealAnimating = false;
          }}
          if (revealIndex >= REVEAL_SEQUENCE.length && revealClickHandlerAttached) {{
            chart.removeEventListener("click", handleRevealClick);
            revealClickHandlerAttached = false;
          }}
        }}

        function resetRevealWorkflow() {{
          revealIndex = 0;
          revealAnimating = false;
          currentFrameIndex = 0;
          REVEAL_SEQUENCE.forEach((key) => {{
            revealState[key] = false;
            appliedVisibility[key] = false;
          }});
          if (!revealClickHandlerAttached && chart) {{
            chart.addEventListener("click", handleRevealClick);
            revealClickHandlerAttached = true;
          }}
          return applyRevealState({{ animateNew: false, entry: getCurrentEntry() }});
        }}

        function buildFigure(entry) {{
          const spendHover = entry.spend.map(formatLargeAmount);
          const closingHover = entry.closing.map(formatLargeAmount);
          const envelopeHover = entry.envelope.map(formatLargeAmount);
          const axisTicks = computeAxisTicks();

          const data = [
            {{
              type: "bar",
              x: entry.years,
              y: entry.spend,
              name: "Annual spend",
              marker: {{color: COLORS.primary}},
              opacity: 0.75,
              customdata: spendHover,
              hovertemplate: "<b>Annual spend</b><br>FY %{{x}}: %{{customdata}}<extra></extra>",
              visible: "legendonly",
              meta: {{revealKey: "spend"}},
            }},
            {{
              type: "scatter",
              mode: "lines+markers",
              x: entry.years,
              y: entry.envelope,
              name: "Envelope",
              line: {{color: COLORS.envelope, width: 3}},
              marker: {{color: COLORS.envelope, size: 6}},
              customdata: envelopeHover,
              hovertemplate: "<b>Envelope</b><br>FY %{{x}}: %{{customdata}}<extra></extra>",
              visible: "legendonly",
              meta: {{revealKey: "envelope"}},
            }},
            {{
              type: "scatter",
              mode: "lines",
              x: entry.years,
              y: entry.closing,
              name: "Closing net",
              line: {{color: COLORS.closing, width: 3}},
              customdata: closingHover,
              hovertemplate: "<b>Closing net</b><br>FY %{{x}}: %{{customdata}}<extra></extra>",
              visible: "legendonly",
              meta: {{revealKey: "closing"}},
            }},
          ];

          const layout = {{
            title: {{text: entry.title || ""}},
            barmode: "overlay",
            legend: {{orientation: "h", y: -0.22}},
            xaxis: {{
              title: "",
              range: [FIXED_X_RANGE[0] - 2, FIXED_X_RANGE[1] + 1.5],
              automargin: true,
              ticklen: 8
            }},
            yaxis: {{
              tickmode: "array",
              tickvals: axisTicks.values,
              ticktext: axisTicks.labels,
              range: axisTicks.range,
              automargin: true,
              zeroline: true,
              zerolinecolor: "#888888",
              zerolinewidth: 1,
              zerolinedash: "dot",
            }},
            hoverlabel: {{namelength: -1}},
            margin: {{l: 110, r: 40, t: 48, b: 140}},
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
          }};

          return {{data, layout}};
        }}

function updateStatus(idx) {{
          currentFrameIndex = Math.max(0, Math.min(idx, totalFrames - 1));
          const entry = sequence[currentFrameIndex] || {{ progress: 0 }};
          const rawProgress = Number(entry.progress);
          const pct = Number.isFinite(rawProgress) ? Math.min(Math.max(rawProgress, 0), 1) : 0;
          const pctDisplay = Math.round(pct * 1000) / 10;
          if (progressFillEl) {{
            progressFillEl.style.width = `${{pct * 100}}%`;
          }}
          if (progressLabelEl) {{
            progressLabelEl.textContent = `${{pctDisplay.toFixed(1)}}%`;
          }}
          updateNpvCard(entry);
        }}

        function ensurePlotly(callback) {{
          if (window.Plotly) {{
            callback();
            return;
          }}
          const script = document.createElement("script");
          script.src = PLOTLY_URL;
          script.async = true;
          script.onload = callback;
          document.head.appendChild(script);
        }}

        ensurePlotly(() => {{
          const initialEntry = sequence[0];
          const initialFigure = buildFigure(initialEntry);
          const config = {{displayModeBar: false, responsive: true}};
          Plotly.newPlot(chart, initialFigure.data, initialFigure.layout, config).then(async () => {{
            await resetRevealWorkflow();
            updateStatus(0);
            if (totalFrames <= 1) {{
              button.disabled = true;
              return;
            }}
          }});

          button.addEventListener("click", async () => {{
            if (button.disabled) {{
              return;
            }}
            button.disabled = true;
            const start = performance.now();
            for (let i = 1; i < totalFrames; i++) {{
              const entry = sequence[i];
              const figure = buildFigure(entry);
              await Plotly.react(chart, figure.data, figure.layout, config);
              updateStatus(i);
              await applyRevealState({{ entry, animateNew: false }});
              const target = start + frameDuration * i;
              if (i < totalFrames - 1) {{
                const delay = target - performance.now();
                if (delay > 0) {{
                  await new Promise((resolve) => setTimeout(resolve, delay));
                }}
              }}
            }}
            button.disabled = false;
          }});
        }});
      }})();
    </script>
    """

    components.html(component_html, height=780)

    return export_tables


def render_cash_flow_tab(
    data: DashboardData,
    opt_selection,
    comp_selection,
    opt_series,
    cmp_series,
    *,
    opt_label: str,
    cmp_label: str,
) -> Dict[str, pd.DataFrame]:
    export_tables: Dict[str, pd.DataFrame] = {}

    st.markdown('<div class="pbi-section-title">Efficiency & cash flow</div>', unsafe_allow_html=True)
    opt_code = getattr(opt_selection, "code", None) if opt_selection is not None else None
    cmp_code = getattr(comp_selection, "code", None) if comp_selection is not None else None
    show_stack_chart = bool(opt_code or cmp_code)
    control_row_spacer_px = 74
    eff_cols = st.columns([2, 3])
    with eff_cols[0]:
        if show_stack_chart:
            st.markdown(f'<div style="height:{control_row_spacer_px}px;"></div>', unsafe_allow_html=True)
        bcr_view_key = "cashflow_bcr_view"
        bcr_view_options = [
            "Spend-weighted project BCR",
            "Portfolio PV BCR (cumulative)",
        ]
        if st.session_state.get(bcr_view_key) not in bcr_view_options:
            st.session_state[bcr_view_key] = bcr_view_options[0]
        selected_bcr_view = st.radio(
            "BCR view",
            bcr_view_options,
            horizontal=True,
            key=bcr_view_key,
        )

        spend_bcr_export = prepare_spend_weighted_bcr_export(data, opt_selection, comp_selection)
        pv_bcr_export = prepare_bcr_time_series_export(
            data,
            opt_series,
            cmp_series,
            opt_selection,
            comp_selection,
        )

        if selected_bcr_view == bcr_view_options[0]:
            bcr_fig = _spend_weighted_bcr_time_series_chart(data, opt_selection, comp_selection)
            bcr_export = spend_bcr_export
            bcr_export_label = "BCR - Spend-weighted (New)"
        else:
            bcr_fig = bcr_time_series_chart(opt_series, cmp_series, opt_selection, comp_selection)
            bcr_export = pv_bcr_export
            bcr_export_label = "BCR - Portfolio Weighted Average"

        if bcr_fig is not None:
            bcr_fig.update_layout(
                height=420,
                margin=dict(l=60, r=20, t=56, b=80, autoexpand=False),
            )
            st.plotly_chart(
                bcr_fig,
                use_container_width=True,
                key="overview_bcr_time_series_chart",
            )
            if bcr_export is not None:
                export_tables[bcr_export_label] = bcr_export
            if spend_bcr_export is not None:
                export_tables["BCR - Spend-weighted (New)"] = spend_bcr_export
            if pv_bcr_export is not None:
                export_tables["BCR - Portfolio Weighted Average"] = pv_bcr_export
        else:
            st.info("Select scenarios to view the BCR time series.")

    with eff_cols[1]:
        if show_stack_chart:
            stack_view_options: List[str] = []
            stack_view_payload: Dict[
                str, Tuple[ScenarioSelection, Optional[ScenarioSelection], str]
            ] = {}
            opt_label_clean = (opt_label or "").strip() or str(opt_code or scenario_primary_label())
            cmp_label_clean = (cmp_label or "").strip() or str(cmp_code or scenario_comparison_label())
            if opt_code:
                stack_view_options.append(opt_label_clean)
                stack_view_payload[opt_label_clean] = (opt_selection, None, opt_label_clean)
            if cmp_code:
                stack_view_options.append(cmp_label_clean)
                stack_view_payload[cmp_label_clean] = (comp_selection, None, cmp_label_clean)
            if opt_code and cmp_code:
                delta_label = f"NPV [{opt_label_clean}] - NPV [{cmp_label_clean}]"
                stack_view_options.append(delta_label)
                stack_view_payload[delta_label] = (opt_selection, comp_selection, delta_label)

            if not stack_view_options:
                st.info("Select a scenario to view costs vs benefits.")
            else:
                view_key = "npv_stack_view_select"
                if st.session_state.get(view_key) not in stack_view_options:
                    st.session_state[view_key] = stack_view_options[0]

                control_cols = st.columns(3)
                with control_cols[0]:
                    stack_view = st.selectbox(
                        "NPV stack view:",
                        stack_view_options,
                        key=view_key,
                    )
                with control_cols[1]:
                    cost_breakdown = st.selectbox(
                        "Breakdown costs by:",
                        list(COST_BENEFIT_STACK_COST_OPTIONS),
                        index=0,
                        key="cost_benefit_stack_cost_breakdown_select",
                    )
                with control_cols[2]:
                    benefit_breakdown = st.selectbox(
                        "Breakdown benefits by:",
                        list(COST_BENEFIT_STACK_BENEFIT_OPTIONS),
                        index=0,
                        key="cost_benefit_stack_benefit_breakdown_select",
                    )

                selection, compare_selection, selection_label = stack_view_payload.get(
                    stack_view,
                    (opt_selection if opt_code else comp_selection, None, stack_view),
                )

                benefit_value_basis = _resolve_value_basis("npv_benefit_value_basis", "npv_apply_discount")
                cost_value_basis = _resolve_value_basis("npv_cost_value_basis", "npv_apply_cost_discount")
                if benefit_value_basis != cost_value_basis:
                    mismatch_signature = f"{cost_value_basis}||{benefit_value_basis}"
                    dismiss_key = "basis_mismatch_warning_dismissed"
                    if st.session_state.get(dismiss_key) != mismatch_signature:
                        warn_cols = st.columns([0.965, 0.035])
                        with warn_cols[0]:
                            st.warning(
                                f"Costs are shown in `{cost_value_basis}` but benefits are shown in `{benefit_value_basis}`. "
                                "Align the value bases for an apples-to-apples comparison."
                            )
                        with warn_cols[1]:
                            if st.button("✕", key="basis_mismatch_warning_close", help="Dismiss this message"):
                                st.session_state[dismiss_key] = mismatch_signature
                                st.rerun()
                else:
                    st.session_state.pop("basis_mismatch_warning_dismissed", None)
                pv_horizon_years = (
                    int(st.session_state.get("npv_horizon_selection", 60))
                    if (benefit_value_basis == VALUE_BASIS_PV or cost_value_basis == VALUE_BASIS_PV)
                    else None
                )

                stack_result = cost_benefit_stack_chart(
                    data,
                    selection,
                    selection_label=selection_label or str(getattr(selection, "code", "")),
                    pv_horizon_years=pv_horizon_years,
                    cost_breakdown=cost_breakdown,
                    benefit_breakdown=benefit_breakdown,
                    cost_value_basis=cost_value_basis,
                    benefit_value_basis=benefit_value_basis,
                    compare_selection=compare_selection,
                )
                stack_export = prepare_cost_benefit_stack_export(
                    data,
                    selection,
                    selection_label=selection_label or str(getattr(selection, "code", "")),
                    pv_horizon_years=pv_horizon_years,
                    cost_breakdown=cost_breakdown,
                    benefit_breakdown=benefit_breakdown,
                    cost_value_basis=cost_value_basis,
                    benefit_value_basis=benefit_value_basis,
                    compare_selection=compare_selection,
                )
                if stack_export is not None and not stack_export.empty:
                    export_label_text = selection_label or str(getattr(selection, "code", ""))
                    export_label = (
                        f"Costs vs benefits delta - {export_label_text}"
                        if compare_selection is not None and getattr(compare_selection, "code", None)
                        else f"Costs vs benefits stack - {export_label_text}"
                    )
                    export_tables[export_label] = stack_export
                if stack_result is not None:
                    stack_fig, cost_legend_items, benefit_legend_items = stack_result
                    st.plotly_chart(
                        stack_fig,
                        use_container_width=True,
                        key="overview_cost_benefit_stack_chart",
                    )
                    render_cost_benefit_stack_legend(
                        cost_legend_items,
                        benefit_legend_items,
                        columns_per_section=3,
                    )
                else:
                    if compare_selection is not None and getattr(compare_selection, "code", None):
                        st.info("No differences found for this breakdown (delta is zero or data missing).")
                    else:
                        st.info("Costs vs benefits breakdown unavailable for this selection.")
        else:
            st.info("Select a scenario to view costs vs benefits.")
    st.markdown('<div class="pbi-section-title">Cash flow profile</div>', unsafe_allow_html=True)
    breakdown_options = list(CASH_FLOW_PROFILE_BREAKDOWN_OPTIONS)
    breakdown_default_index = (
        breakdown_options.index("GPS Request Tier") if "GPS Request Tier" in breakdown_options else 0
    )
    breakdown_key = "cash_flow_profile_spend_breakdown"
    if breakdown_key in st.session_state and st.session_state[breakdown_key] not in breakdown_options:
        del st.session_state[breakdown_key]
    if breakdown_key in st.session_state:
        spend_breakdown = st.selectbox(
            "Breakdown annual spend by:",
            breakdown_options,
            key=breakdown_key,
        )
    else:
        spend_breakdown = st.selectbox(
            "Breakdown annual spend by:",
            breakdown_options,
            index=breakdown_default_index,
            key=breakdown_key,
        )
    cash_axis_range = _effective_year_range(
        [opt_series, cmp_series],
        value_column="Spend",
    )
    cash_cols = st.columns(2)
    profile_label = "cash flow profile"
    with cash_cols[0]:
        if opt_series is not None:
            opt_fig = cash_chart(
                opt_series,
                f"Cash flow - {opt_label}",
                color=PRIMARY_COLOR,
                spend_breakdown=spend_breakdown,
                data=data,
                selection=opt_selection,
                comparison_selection=comp_selection,
                x_axis_year_range=cash_axis_range,
            )
            st.plotly_chart(
                opt_fig,
                use_container_width=True,
                key="overview_cash_flow_opt_chart",
            )
            export = prepare_cash_export(
                opt_series,
                label_prefix=opt_label,
                spend_breakdown=spend_breakdown,
                data=data,
                selection=opt_selection,
            )
            if export is not None:
                export_name = f"Cash flow - {opt_label}"
                export_tables[export_name] = export
        elif opt_selection.code:
            st.warning(f"Cash flow data unavailable for the {opt_label} selection.")
        else:
            st.info(f"Select {opt_label} to view the {profile_label}.")

    with cash_cols[1]:
        if cmp_series is not None:
            cmp_fig = cash_chart(
                cmp_series,
                f"Cash flow - {cmp_label}",
                color=PRIMARY_COLOR,
                spend_breakdown=spend_breakdown,
                data=data,
                selection=comp_selection,
                comparison_selection=opt_selection,
                x_axis_year_range=cash_axis_range,
            )
            st.plotly_chart(
                cmp_fig,
                use_container_width=True,
                key="overview_cash_flow_cmp_chart",
            )
            export = prepare_cash_export(
                cmp_series,
                label_prefix=cmp_label,
                spend_breakdown=spend_breakdown,
                data=data,
                selection=comp_selection,
            )
            if export is not None:
                export_name = f"Cash flow - {cmp_label}"
                export_tables[export_name] = export
        elif comp_selection.code:
            st.warning(f"Cash flow data unavailable for the {cmp_label} selection.")
        else:
            st.info(f"Select {cmp_label} to view the {profile_label}.")

    st.markdown('<div class="pbi-section-title">Cumulative spend vs benefit</div>', unsafe_allow_html=True)
    eff_fig = efficiency_chart(opt_series, cmp_series, opt_selection, comp_selection)
    if eff_fig is not None:
        eff_fig.update_layout(
            height=460,
            margin=dict(l=70, r=20, t=56, b=100, autoexpand=False),
        )
        st.plotly_chart(
            eff_fig,
            use_container_width=True,
            key="overview_efficiency_chart",
        )
        efficiency_export = prepare_efficiency_export(
            data,
            opt_series,
            cmp_series,
            opt_selection,
            comp_selection,
        )
        if efficiency_export is not None:
            export_tables["Cumulative spend vs benefit"] = efficiency_export
    else:
        st.info("Select scenarios to view cumulative spend vs benefit.")

    return export_tables


def render_analysis_tab(
    data: DashboardData,
    opt_selection,
    comp_selection,
    opt_series,
    cmp_series,
    *,
    opt_label: str,
    cmp_label: str,
) -> Dict[str, pd.DataFrame]:
    export_tables: Dict[str, pd.DataFrame] = {}

    st.markdown('<div class="pbi-section-title">Analysis</div>', unsafe_allow_html=True)
    page_key = "analysis_page_view"
    page_value = st.session_state.get(page_key, 1)
    if page_value not in (1, 2):
        page_value = 1
        st.session_state[page_key] = page_value
    nav_cols = st.columns([1, 1, 1])
    with nav_cols[0]:
        if page_value == 2:
            if st.button("<- Previous page", key="analysis_prev_page"):
                st.session_state[page_key] = 1
                st.rerun()
    with nav_cols[1]:
        st.markdown(
            f"<div style='text-align:center;color:#64748B;font-size:0.85rem;font-weight:600;'>"
            f"Page {page_value} of 2</div>",
            unsafe_allow_html=True,
        )
    with nav_cols[2]:
        if page_value == 1:
            if st.button("Next page ->", key="analysis_next_page"):
                st.session_state[page_key] = 2
                st.rerun()

    opt_code = getattr(opt_selection, "code", None) if opt_selection is not None else None
    cmp_code = getattr(comp_selection, "code", None) if comp_selection is not None else None
    if not (opt_code or cmp_code):
        st.info("Select scenarios to view the analysis.")
        return export_tables

    years = [int(y) for y in getattr(data, "years", []) if y is not None]
    if not years:
        st.info("No annual data available for the analysis view.")
        return export_tables

    breakdown_options = [
        "Region",
        "GPS Request Tier",
        "Activity Class",
        "Strategic Dimension",
    ]
    breakdown_key = "analysis_breakdown"
    if st.session_state.get(breakdown_key) not in breakdown_options:
        st.session_state[breakdown_key] = "Region"
    breakdown_choice = st.radio(
        "Breakdown by",
        breakdown_options,
        horizontal=True,
        key=breakdown_key,
    )
    breakdown_choice_norm = str(breakdown_choice).strip().lower()
    breakdown_label = str(breakdown_choice or "Activity Class").strip()
    group_col: Optional[str] = None
    if breakdown_choice_norm == "activity class":
        group_col = "ActivityClass"
    elif breakdown_choice_norm == "gps request tier":
        group_col = "GPSTier"
    elif breakdown_choice_norm == "region":
        group_col = "Region"
    elif breakdown_choice_norm == "strategic dimension":
        group_col = "StrategicDimension"

    limit = 5
    if breakdown_choice_norm == "region":
        limit = 0

    if group_col == "StrategicDimension":
        spend_opt = (
            _annual_cost_breakdown_by_dimension_weights(
                data,
                opt_selection,
                years=years,
            )
            if opt_code
            else None
        )
        spend_cmp = (
            _annual_cost_breakdown_by_dimension_weights(
                data,
                comp_selection,
                years=years,
            )
            if cmp_code
            else None
        )
        benefit_opt = dimension_timeseries(data, opt_selection) if opt_code else None
        benefit_cmp = dimension_timeseries(data, comp_selection) if cmp_code else None
    else:
        spend_opt = (
            _annual_spend_breakdown_by_attribute(
                data,
                opt_selection,
                group_col=group_col or "ActivityClass",
                years=years,
            )
            if opt_code
            else None
        )
        spend_cmp = (
            _annual_spend_breakdown_by_attribute(
                data,
                comp_selection,
                group_col=group_col or "ActivityClass",
                years=years,
            )
            if cmp_code
            else None
        )
        benefit_opt = (
            _annual_benefit_breakdown_by_attribute(
                data,
                opt_selection,
                group_col=group_col or "ActivityClass",
                years=years,
            )
            if opt_code
            else None
        )
        benefit_cmp = (
            _annual_benefit_breakdown_by_attribute(
                data,
                comp_selection,
                group_col=group_col or "ActivityClass",
                years=years,
            )
            if cmp_code
            else None
        )

    opt_label_text = (opt_label or "").strip() or scenario_primary_label()
    cmp_label_text = (cmp_label or "").strip() or scenario_comparison_label()

    if page_value == 1:
        group_totals_opt = _analysis_group_totals(spend_opt)
        group_totals_cmp = _analysis_group_totals(spend_cmp)
        group_options, _ = _analysis_group_order_and_share(group_totals_opt, group_totals_cmp)

        dist_fig = _distribution_cost_chart(
            spend_opt,
            spend_cmp,
            breakdown_label=breakdown_label,
            opt_label=opt_label_text,
            cmp_label=cmp_label_text,
        )

        inspector_cols = st.columns([1.25, 1.0], gap="large")
        with inspector_cols[0]:
            if group_options:
                st.markdown("**Group inspector**")
                st.caption(f"Select a {breakdown_label} to inspect annual and cumulative deltas.")
                group_key = f"analysis_delta_group_{(group_col or 'group').lower()}"
                current_group = st.session_state.get(group_key)
                if current_group not in group_options:
                    current_group = group_options[0]
                    st.session_state[group_key] = current_group
                selected_group = st.selectbox(
                    f"{breakdown_label} focus",
                    group_options,
                    index=group_options.index(current_group),
                    key=group_key,
                )
                delta_fig = _analysis_delta_bar_line_chart(
                    spend_opt,
                    spend_cmp,
                    years=years,
                    group_label=selected_group,
                    breakdown_label=breakdown_label,
                    opt_label=opt_label_text,
                    cmp_label=cmp_label_text,
                )
                if delta_fig is not None:
                    st.plotly_chart(
                        delta_fig,
                        use_container_width=True,
                        key="analysis_delta_group_chart",
                    )
                else:
                    st.info("No delta data available for the selected group.")
            else:
                st.info("No group data available for the selected breakdown.")
        with inspector_cols[1]:
            if dist_fig is not None:
                st.plotly_chart(
                    dist_fig,
                    use_container_width=True,
                    key="analysis_cost_distribution_chart",
                )
            else:
                st.info("No distribution data available for the selected breakdown.")

        help_text = (
            "Timing center = sum(Year x Spend) / sum(Spend) using annual spend.\n"
            "Dots show each scenario's timing center for the category.\n"
            f"Line label is the shift in years ({opt_label_text} minus {cmp_label_text})."
        )
        help_attr = html.escape(help_text).replace("\n", "&#10;")
        st.markdown(
            f"<div class='pbi-help-wrap'><span class='pbi-help-icon' title='{help_attr}'>?</span></div>",
            unsafe_allow_html=True,
        )
        order_totals_opt = group_totals_opt
        order_totals_cmp = group_totals_cmp
        combined_order = order_totals_opt.add(order_totals_cmp, fill_value=0.0)
        combined_order = combined_order[combined_order.abs() > 1e-9]
        category_order = combined_order.sort_values(ascending=False).index.astype(str).tolist()
        if "Unknown" in category_order:
            category_order = [val for val in category_order if val != "Unknown"] + ["Unknown"]
        importance_summary = _timing_shift_summary(spend_opt, spend_cmp)
        rank_fig = (
            _importance_rank_chart(
                importance_summary,
                breakdown_label=breakdown_label,
                opt_label=opt_label_text,
                cmp_label=cmp_label_text,
                category_order=category_order or None,
            )
            if importance_summary is not None
            else None
        )
        top_cols = st.columns([2.3, 1.2])
        with top_cols[0]:
            paddle_fig = _timing_paddle_chart(
                spend_opt,
                spend_cmp,
                breakdown_label=breakdown_label,
                opt_label=opt_label_text,
                cmp_label=cmp_label_text,
            )
            paddle_height = None
            if paddle_fig is not None:
                paddle_height = paddle_fig.layout.height
                st.plotly_chart(
                    paddle_fig,
                    use_container_width=True,
                    key="analysis_timing_paddle_chart",
                )
            else:
                st.info("No timing data available for the selected breakdown.")
        with top_cols[1]:
            if rank_fig is not None:
                if paddle_height:
                    rank_fig.update_layout(height=paddle_height)
                st.plotly_chart(
                    rank_fig,
                    use_container_width=True,
                    key="analysis_importance_rank_chart",
                )
            else:
                st.info("No rank view data available for the selected breakdown.")

        caption_text = (
            f"{breakdown_label} view. Heatmaps show within-group density (each row sums to 100%). "
            f"Differential is {opt_label_text} minus {cmp_label_text}."
        )
        diff_options = {
            "Δ$ (absolute magnitude)": "absolute",
            "Δ / group total (relative difference)": "relative",
        }
        diff_key = "analysis_density_diff_mode"
        if st.session_state.get(diff_key) not in diff_options:
            st.session_state[diff_key] = list(diff_options.keys())[0]
        header_cols = st.columns([2.6, 1.6])
        with header_cols[0]:
            st.markdown(
                "<div style='padding:0.35rem 0 0.35rem 0;'>"
                "<div style='font-weight:700;color:#0F172A;'>"
                "Spend density & scenario differential explorer"
                "</div>"
                f"<div style='color:#64748B;font-size:0.85rem;margin-top:0.2rem;'>"
                f"{html.escape(caption_text)}"
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        with header_cols[1]:
            st.markdown("**Differential view**")
            diff_choice = st.radio(
                "Differential view",
                list(diff_options.keys()),
                horizontal=True,
                key=diff_key,
                label_visibility="collapsed",
            )
        diff_mode = diff_options.get(diff_choice, "relative")
        density_figs = _analysis_spend_density_charts(
            spend_opt,
            spend_cmp,
            years=years,
            breakdown_label=breakdown_label,
            opt_label=opt_label_text,
            cmp_label=cmp_label_text,
            diff_mode=diff_mode,
        )
        if density_figs is not None:
            dens_cols = st.columns(3)
            with dens_cols[0]:
                st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
                st.plotly_chart(
                    density_figs[0],
                    use_container_width=True,
                    key="analysis_density_opt",
                )
            with dens_cols[1]:
                st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
                st.plotly_chart(
                    density_figs[1],
                    use_container_width=True,
                    key="analysis_density_cmp",
                )
            with dens_cols[2]:
                st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
                st.plotly_chart(
                    density_figs[2],
                    use_container_width=True,
                    key="analysis_density_diff",
                )
        else:
            st.info("No spend density data available for the selected breakdown.")

        if SHOW_ANALYSIS_CORRELATION:
            corr_options = {
                "Compositional (CLR / Aitchison)": "clr",
                "Absolute annual spend": "absolute",
                "Year-over-year change": "yoy",
                "Single-group spend share": "group_share",
            }
            corr_group_options = ["Activity", "Tier", "Region", "Dimension"]
            corr_scenario_key = "analysis_corr_scenario"
            corr_scenario_options: Dict[str, Dict[str, Any]] = {}
            if opt_code:
                corr_scenario_options[f"{opt_label_text} (X)"] = {
                    "kind": "single",
                    "selection": opt_selection,
                    "label": f"{opt_label_text} (X)",
                }
            if cmp_code:
                corr_scenario_options[f"{cmp_label_text} (Y)"] = {
                    "kind": "single",
                    "selection": comp_selection,
                    "label": f"{cmp_label_text} (Y)",
                }
            if opt_code and cmp_code:
                corr_scenario_options[f"Delta ({opt_label_text} - {cmp_label_text})"] = {
                    "kind": "delta",
                    "label": f"Delta ({opt_label_text} - {cmp_label_text})",
                }
            corr_scenario_labels = list(corr_scenario_options.keys())
            if st.session_state.get(corr_scenario_key) not in corr_scenario_options:
                st.session_state[corr_scenario_key] = corr_scenario_labels[0]
            corr_controls = st.columns([0.78, 0.22], gap="small")
            corr_mode = "absolute"
            corr_change_basis = "spend"
            corr_zscore = False
            corr_group = None
            corr_scenario_choice = st.session_state.get(corr_scenario_key, corr_scenario_labels[0])
            with corr_controls[0]:
                corr_choice_default = st.session_state.get("analysis_corr_mode")
                if corr_choice_default not in corr_options:
                    corr_choice_default = list(corr_options.keys())[0]
                corr_mode_preview = corr_options[corr_choice_default]
                if corr_mode_preview in {"clr", "group_share"}:
                    corr_select_cols = st.columns(3)
                    with corr_select_cols[0]:
                        corr_choice = st.selectbox(
                            "Correlation view",
                            list(corr_options.keys()),
                            key="analysis_corr_mode",
                        )
                        corr_mode = corr_options[corr_choice]
                    with corr_select_cols[1]:
                        corr_group = st.selectbox(
                            "Group",
                            corr_group_options,
                            key="analysis_corr_group",
                        )
                    with corr_select_cols[2]:
                        corr_scenario_choice = st.selectbox(
                            "Scenario",
                            corr_scenario_labels,
                            key=corr_scenario_key,
                        )
                else:
                    corr_select_cols = st.columns(2)
                    with corr_select_cols[0]:
                        corr_choice = st.selectbox(
                            "Correlation view",
                            list(corr_options.keys()),
                            key="analysis_corr_mode",
                        )
                        corr_mode = corr_options[corr_choice]
                    with corr_select_cols[1]:
                        corr_scenario_choice = st.selectbox(
                            "Scenario",
                            corr_scenario_labels,
                            key=corr_scenario_key,
                        )
                if corr_mode == "absolute":
                    corr_zscore = st.checkbox(
                        "Z-score per category",
                        value=False,
                        key="analysis_corr_zscore",
                    )
                elif corr_mode == "yoy":
                    basis_label = st.selectbox(
                        "Change basis",
                        ["Spend", "Spend share"],
                        key="analysis_corr_change_basis",
                    )
                    corr_change_basis = "share" if basis_label == "Spend share" else "spend"
                if corr_mode in {"clr", "group_share"} and corr_group is None:
                    corr_group = st.selectbox(
                        "Group",
                        corr_group_options,
                        key="analysis_corr_group",
                    )
                if corr_mode not in {"clr", "group_share"}:
                    corr_group = None
            corr_scenario = corr_scenario_options.get(corr_scenario_choice, {})
            corr_scenario_kind = corr_scenario.get("kind", "single")
            corr_scenario_label = corr_scenario.get("label", corr_scenario_choice)
            help_text = _analysis_correlation_help_text(
                corr_mode,
                change_basis=corr_change_basis,
                zscore=corr_zscore,
                group_filter=corr_group,
            )
            if corr_scenario_kind == "delta":
                help_text = (
                    f"{help_text}\nDelta equals correlation({opt_label_text} (X)) minus "
                    f"correlation({cmp_label_text} (Y))."
                )
            help_attr = html.escape(help_text).replace("\n", "&#10;")
            with corr_controls[1]:
                st.markdown(
                    f"<div class='pbi-help-wrap'><span class='pbi-help-icon' title='{help_attr}'>?</span></div>",
                    unsafe_allow_html=True,
                )
            if corr_scenario_kind == "delta":
                corr_fig = _analysis_correlation_matrix_delta_chart(
                    data,
                    opt_selection,
                    comp_selection,
                    years,
                    mode=corr_mode,
                    change_basis=corr_change_basis,
                    zscore=corr_zscore,
                    group_filter=corr_group,
                    label_x=f"{opt_label_text} (X)",
                    label_y=f"{cmp_label_text} (Y)",
                )
            else:
                corr_selection = corr_scenario.get("selection")
                if corr_selection is None:
                    corr_selection = opt_selection if opt_code else comp_selection
                corr_fig = _analysis_correlation_matrix_chart(
                    data,
                    corr_selection,
                    years,
                    mode=corr_mode,
                    change_basis=corr_change_basis,
                    zscore=corr_zscore,
                    group_filter=corr_group,
                    title_suffix=corr_scenario_label,
                )
            if corr_fig is not None:
                st.plotly_chart(
                    corr_fig,
                    use_container_width=True,
                    key="analysis_correlation_matrix_chart",
                )
            else:
                st.info("No correlation data available for the selected scenarios.")
        return export_tables

    if group_col == "StrategicDimension":
        activity_classes = _analysis_dimension_labels(
            data,
            spend_opt,
            spend_cmp,
            benefit_opt,
            benefit_cmp,
            limit=limit,
        )
    else:
        activity_classes = _analysis_activity_classes(
            spend_opt,
            spend_cmp,
            benefit_opt,
            benefit_cmp,
            limit=limit,
        )
    if not activity_classes:
        st.info("No breakdown data available for the analysis view.")
        return export_tables

    cost_max_year = _analysis_max_year_for_classes(activity_classes, spend_opt, spend_cmp, years)
    benefit_max_year = _analysis_max_year_for_classes(activity_classes, benefit_opt, benefit_cmp, years)
    cost_x_max = cost_max_year if cost_max_year is not None else max(years)
    benefit_x_max = benefit_max_year if benefit_max_year is not None else max(years)
    cost_y_max = _analysis_max_value_for_classes(activity_classes, spend_opt, spend_cmp, years)
    benefit_y_max = _analysis_max_value_for_classes(activity_classes, benefit_opt, benefit_cmp, years)

    legend_items = []
    if opt_code:
        legend_items.append(
            f"<span style='display:flex;gap:0.35rem;align-items:center;color:{PRIMARY_COLOR};'>"
            "<span style='width:8px;height:8px;border-radius:999px;background:"
            f"{PRIMARY_COLOR};display:inline-block;'></span>{html.escape(opt_label_text)}</span>"
        )
    if cmp_code:
        legend_items.append(
            f"<span style='display:flex;gap:0.35rem;align-items:center;color:{CAPACITY_HEAT_RED};'>"
            "<span style='width:8px;height:8px;border-radius:999px;background:"
            f"{CAPACITY_HEAT_RED};display:inline-block;'></span>{html.escape(cmp_label_text)}</span>"
        )
    legend_html = (
        "<div style='display:flex;gap:1rem;align-items:center;margin:0.3rem 0 0.8rem;'>"
        "<span style='font-weight:600;color:#0F172A;'>Scenario comparison (real $):</span>"
        + "".join(legend_items)
        + "</div>"
    )
    st.markdown(legend_html, unsafe_allow_html=True)

    header_cols = st.columns(2)
    with header_cols[0]:
        st.markdown(f"**Spend by {breakdown_label}**")
    with header_cols[1]:
        st.markdown(f"**Benefits by {breakdown_label}**")

    for idx, activity in enumerate(activity_classes):
        show_xaxis = idx == len(activity_classes) - 1
        row_cols = st.columns(2)
        with row_cols[0]:
            series_opt = _analysis_series_for_class(spend_opt, activity, years)
            series_cmp = _analysis_series_for_class(spend_cmp, activity, years)
            cost_fig = _analysis_sparkline_chart(
                activity,
                years,
                series_opt,
                series_cmp,
                opt_label=opt_label_text if opt_code else "",
                cmp_label=cmp_label_text if cmp_code else "",
                color_opt=PRIMARY_COLOR,
                color_cmp=CAPACITY_HEAT_RED,
                x_max=cost_x_max,
                y_max=cost_y_max,
                show_xaxis=show_xaxis,
            )
            st.plotly_chart(
                cost_fig,
                use_container_width=True,
                key=f"analysis_spend_{idx}",
            )
        with row_cols[1]:
            series_opt = _analysis_series_for_class(benefit_opt, activity, years)
            series_cmp = _analysis_series_for_class(benefit_cmp, activity, years)
            benefit_fig = _analysis_sparkline_chart(
                activity,
                years,
                series_opt,
                series_cmp,
                opt_label=opt_label_text if opt_code else "",
                cmp_label=cmp_label_text if cmp_code else "",
                color_opt=PRIMARY_COLOR,
                color_cmp=CAPACITY_HEAT_RED,
                x_max=benefit_x_max,
                y_max=benefit_y_max,
                show_xaxis=show_xaxis,
            )
            st.plotly_chart(
                benefit_fig,
                use_container_width=True,
                key=f"analysis_benefit_{idx}",
            )

    return export_tables


def render_benefits_tab(
    data: DashboardData,
    opt_selection,
    comp_selection,
    opt_series,
    cmp_series,
    *,
    opt_label: str,
    cmp_label: str,
) -> Dict[str, pd.DataFrame]:
    export_tables: Dict[str, pd.DataFrame] = {}
    apply_discount = bool(st.session_state.get("npv_apply_discount", False))
    headline = "Net present value breakdown" if apply_discount else "Benefit breakdown ($2025)"
    st.markdown(f'<div class="pbi-section-title">{headline}</div>', unsafe_allow_html=True)
    npv_horizon_options = [60, 50, 40]
    selected_npv_horizon = st.session_state.get("npv_horizon_selection", npv_horizon_options[0])
    try:
        selected_npv_horizon = int(selected_npv_horizon)
    except (TypeError, ValueError):
        selected_npv_horizon = npv_horizon_options[0]
    if selected_npv_horizon not in npv_horizon_options:
        selected_npv_horizon = npv_horizon_options[0]
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
        waterfall_fig = benefit_waterfall_chart(
            data, opt_selection, comp_selection, horizon_years=selected_npv_horizon
        )
        if waterfall_fig is not None:
            st.plotly_chart(
                waterfall_fig,
                use_container_width=True,
                key="benefits_waterfall_chart",
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
            export_tables["NPV Waterfall" if apply_discount else "Benefits Waterfall ($2025)"] = waterfall_export
    with pv_col2:
        bridge_fig = benefit_bridge_chart(
            data, opt_selection, comp_selection, horizon_years=selected_npv_horizon
        )
        if bridge_fig is not None:
            st.plotly_chart(
                bridge_fig,
                use_container_width=True,
                key="benefits_bridge_chart",
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
            export_tables["NPV Bridge" if apply_discount else "Benefits Bridge ($2025)"] = bridge_export
    radar_fig = benefit_radar_chart(
        data,
        opt_selection,
        comp_selection,
        horizon_years=selected_npv_horizon,
    )
    if radar_fig is not None:
        st.plotly_chart(
            radar_fig,
            use_container_width=True,
            theme=None,
            key="benefits_radar_chart",
        )
        radar_export = prepare_radar_export(
            data,
            opt_selection,
            comp_selection,
            horizon_years=selected_npv_horizon,
        )
        if radar_export is not None:
            export_tables["Benefit mix radar"] = radar_export

    st.markdown('<div class="pbi-section-title">Benefit dimensions</div>', unsafe_allow_html=True)
    opt_dim_pivot = dimension_timeseries(data, opt_selection)
    cmp_dim_pivot = dimension_timeseries(data, comp_selection)
    available_dimension_labels = [
        str(dim)
        for dim in getattr(data, "dims", [])
        if str(dim).strip().lower() != "total"
        and (
            (opt_dim_pivot is not None and dim in getattr(opt_dim_pivot, "columns", []))
            or (cmp_dim_pivot is not None and dim in getattr(cmp_dim_pivot, "columns", []))
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
            st.info(f"No dimension data available for {cmp_label}.")
        else:
            default_dims = available_dimension_labels[:1] or ["Total"]
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
                    st.plotly_chart(
                        comparison_fig,
                        use_container_width=True,
                        key="benefits_dimension_overlay_chart",
                    )
                    overlay_export = prepare_dimension_overlay_export(
                        data.years,
                        opt_dim_pivot,
                        cmp_dim_pivot,
                        selected_dims,
                        show_cumulative_benefits,
                    )
                    if overlay_export is not None:
                        export_tables[f"Dimension overlay - {cmp_label}"] = overlay_export
                else:
                    st.info(f"No overlapping dimension data available for {cmp_label}.")
    else:
        dim_col1, dim_col2 = st.columns(2)
        with dim_col1:
            opt_dim_fig = benefit_dimension_chart(
                data,
                opt_selection,
                title=f"{opt_label} benefit mix by dimension",
                cumulative=show_cumulative_benefits,
                pivot=opt_dim_pivot,
            )
            if opt_dim_fig is not None:
                st.plotly_chart(
                    opt_dim_fig,
                    use_container_width=True,
                    key="benefits_dimension_primary_chart",
                )
                opt_dim_export = prepare_dimension_chart_export(
                    opt_dim_pivot,
                    show_cumulative_benefits,
                )
                if opt_dim_export is not None:
                    export_tables["Dimension mix - optimised"] = opt_dim_export
        with dim_col2:
            cmp_dim_fig = benefit_dimension_chart(
                data,
                comp_selection,
                title=f"{cmp_label} benefit mix by dimension",
                cumulative=show_cumulative_benefits,
                pivot=cmp_dim_pivot,
            )
            if cmp_dim_fig is not None:
                st.plotly_chart(
                    cmp_dim_fig,
                    use_container_width=True,
                    key="benefits_dimension_comparison_chart",
                )
                cmp_dim_export = prepare_dimension_chart_export(
                    cmp_dim_pivot,
                    show_cumulative_benefits,
                )
                if cmp_dim_export is not None:
                    export_tables[f"Dimension mix - {cmp_label}"] = cmp_dim_export
    if SHOW_REAL_BENEFIT_CHARTS:
        st.markdown('<div class="pbi-section-title">Benefit profile</div>', unsafe_allow_html=True)
        horizon_years = int(st.session_state.get("npv_horizon_selection", 60))
        benefit_cols = st.columns(2)
        with benefit_cols[0]:
            benefit_fig = benefit_chart(opt_series, cmp_series, dimension=opt_selection.dimension)
            if benefit_fig is not None:
                st.plotly_chart(
                    benefit_fig,
                    use_container_width=True,
                    key="benefits_profile_chart",
                )
        with benefit_cols[1]:
            delta_fig = benefit_delta_chart(
                data,
                opt_series,
                cmp_series,
                opt_selection,
                comp_selection,
                horizon_years=horizon_years,
            )
            if delta_fig is not None:
                st.plotly_chart(
                    delta_fig,
                    use_container_width=True,
                    key="benefits_delta_chart",
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
    return export_tables


def render_delivery_tab(
    data: DashboardData,
    opt_selection,
    comp_selection,
    opt_series,
    cmp_series,
    *,
    opt_label: str,
    cmp_label: str,
) -> Dict[str, pd.DataFrame]:
    export_tables: Dict[str, pd.DataFrame] = {}
    project_colors = project_color_map(data)
    schedule_year_range = _effective_year_range(
        [opt_series, cmp_series],
        value_column="Spend",
    )
    st.markdown('<div class="pbi-section-title">Delivery schedule</div>', unsafe_allow_html=True)
    schedule_opt_fig = project_schedule_area_chart(
        data,
        opt_selection,
        title=f"Project schedule - {opt_label}",
        color_map=project_colors,
        display_year_range=schedule_year_range,
    )
    schedule_cmp_fig = project_schedule_area_chart(
        data,
        comp_selection,
        title=f"Project schedule - {cmp_label}",
        color_map=project_colors,
        display_year_range=schedule_year_range,
    )
    schedule_col1, schedule_col2 = st.columns(2)
    with schedule_col1:
        if schedule_opt_fig is not None:
            st.plotly_chart(
                schedule_opt_fig,
                use_container_width=True,
                key="project_schedule_opt_chart",
            )
            schedule_export = prepare_schedule_export(data, opt_selection)
            if schedule_export is not None:
                export_tables[f"Project schedule - {opt_label}"] = schedule_export
        elif opt_selection.code:
            st.warning(f"No project schedule data found for the {opt_label} selection.")
        else:
            st.info(f"Select {opt_label} to view the project schedule.")
    with schedule_col2:
        if schedule_cmp_fig is not None:
            st.plotly_chart(
                schedule_cmp_fig,
                use_container_width=True,
                key="project_schedule_cmp_chart",
            )
            schedule_export = prepare_schedule_export(data, comp_selection)
            if schedule_export is not None:
                export_tables[f"Project schedule - {cmp_label}"] = schedule_export
        elif comp_selection.code:
            st.warning(f"No project schedule data found for the {cmp_label} selection.")
        else:
            st.info(f"Select {cmp_label} to view the project schedule.")
    st.markdown('<div class="pbi-section-title">Market capacity</div>', unsafe_allow_html=True)
    cap_cols = st.columns(2)
    with cap_cols[0]:
        cap_fig_opt = market_capacity_indicator(data, opt_selection)
        if cap_fig_opt is not None:
            st.plotly_chart(cap_fig_opt, use_container_width=True, key="market_capacity_opt")
            cap_export = prepare_capacity_export(opt_series)
            if cap_export is not None:
                export_tables[f"Market capacity - {opt_label}"] = cap_export
        elif opt_selection.code:
            st.warning(f"No spend data found for the {opt_label} selection.")
        else:
            st.info(f"Select {opt_label} to view the capacity profile.")
    with cap_cols[1]:
        cap_fig_cmp = market_capacity_indicator(data, comp_selection)
        if cap_fig_cmp is not None:
            st.plotly_chart(cap_fig_cmp, use_container_width=True, key="market_capacity_cmp")
            cap_export = prepare_capacity_export(cmp_series)
            if cap_export is not None:
                export_tables[f"Market capacity - {cmp_label}"] = cap_export
        elif comp_selection.code:
            st.warning(f"No spend data found for the {cmp_label} selection.")
        else:
            st.info(f"Select {cmp_label} to view the capacity profile.")
    return export_tables

def render_gantt_tab(
    data: DashboardData,
    opt_selection,
    comp_selection,
    *,
    opt_label: str,
    cmp_label: str,
) -> Dict[str, pd.DataFrame]:
    export_tables: Dict[str, pd.DataFrame] = {}

    st.markdown('<div class="pbi-section-title">Programme schedule</div>', unsafe_allow_html=True)

    # Work out which scenarios are available for display
    scenario_tokens = []
    token_to_selection: Dict[str, ScenarioSelection] = {}
    token_to_label: Dict[str, str] = {}

    if opt_selection.code:
        scenario_tokens.append(SCENARIO_PRIMARY_NAME)
        token_to_selection[SCENARIO_PRIMARY_NAME] = opt_selection
        token_to_label[SCENARIO_PRIMARY_NAME] = opt_label

    if comp_selection.code:
        scenario_tokens.append(SCENARIO_COMPARISON_NAME)
        token_to_selection[SCENARIO_COMPARISON_NAME] = comp_selection
        token_to_label[SCENARIO_COMPARISON_NAME] = cmp_label

    if not scenario_tokens:
        st.info("Select at least one scenario to display the Programme Schedule chart.")
        return export_tables

    # Keep radio selection stable across reruns
    current_token = st.session_state.get("gantt_choice")
    if current_token not in scenario_tokens:
        st.session_state["gantt_choice"] = scenario_tokens[0]

    control_cols = st.columns([3, 1])
    with control_cols[0]:
        gantt_token = st.radio(
            "Programme schedule display",
            scenario_tokens,
            horizontal=True,
            key="gantt_choice",
            format_func=lambda token: token_to_label[token],
        )

    with control_cols[1]:
        show_outline = st.checkbox(
            f"Show {cmp_label} schedule outline",
            value=True,
            key="gantt_outline",
        )
    primary_selection = token_to_selection[gantt_token]
    primary_label = token_to_label[gantt_token]
    opposite_token = (SCENARIO_COMPARISON_NAME if gantt_token == SCENARIO_PRIMARY_NAME else SCENARIO_PRIMARY_NAME)
    comparison_selection = token_to_selection.get(
        opposite_token,
        comp_selection if gantt_token == SCENARIO_PRIMARY_NAME else opt_selection,
    )

    # Mount hotkey shim (safe across Streamlit versions)
    _inject_gantt_hotkey_listener()

    if st.session_state.get("_gantt_hotkey_supported", False):
        st.caption(f"Press **Z** to toggle the outline colour between {cmp_label.lower()} and baseline styling.")
    else:
        st.caption(f"Outline uses the {cmp_label.lower()} styling by default.")

    gantt_fig = spend_gantt_chart(
        data,
        primary_selection,
        comparison_selection=comparison_selection,
        show_outline=show_outline,
        title=f"Project delivery schedule - {primary_label}",
    )
    if gantt_fig is not None:
        st.plotly_chart(gantt_fig, use_container_width=True)
        gantt_exports = prepare_gantt_export(
            data,
            opt_selection,
            comp_selection,
            opt_label=opt_label,
            cmp_label=cmp_label,
        )
        if gantt_exports:
            export_tables.update(gantt_exports)
    else:
        st.info("No spend matrix found for the selected scenario.")

    return export_tables




def render_scenarios_tab(
    settings: Settings,
    preset_root: Path,
    saved_root: Path,
    scenario_folders: List[scenario_utils.ScenarioFolder],
    data: DashboardData,
    *,
    opt_selection: Optional[ScenarioSelection] = None,
    comp_selection: Optional[ScenarioSelection] = None,
    opt_label: str = "",
    cmp_label: str = "",
    download_tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    st.markdown('<div class="pbi-section-title">Scenario workspace</div>', unsafe_allow_html=True)
    st.write("Use this workspace to create scenario folders and run new optimisation batches.")
    st.write(f"Preset scenarios live in {preset_root} | Saved scenarios live in {saved_root}")
    st.markdown('<div class="pbi-section-title">Solver gap summary</div>', unsafe_allow_html=True)
    render_solver_gap_kpis(opt_selection, comp_selection)
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
        table_placeholder.dataframe(pd.DataFrame(rows), width="stretch")

    render_folder_table(scenario_folders)

    st.markdown('<div class="pbi-section-title">Create new scenario folder</div>', unsafe_allow_html=True)
    new_name = st.text_input("Scenario name", key="scenario_new_name")
    new_kind = st.radio(
        "Folder type",
        ("saved", "preset"),
        index=0,
        format_func=str.title,
        horizontal=True,
        key="scenario_new_kind",
    )
    if st.button("Create folder", key="scenario_create_folder"):
        if new_name.strip():
            new_folder = scenario_utils.create_scenario_folder(settings, new_name.strip(), new_kind)
            st.session_state["last_created_folder_name"] = new_folder.name
            st.session_state["selected_cache_label"] = f"{new_folder.kind.title()} / {new_folder.name}"
            load_dashboard_data.clear()
            st.rerun()
        else:
            st.warning("Please provide a scenario name before creating a folder.")


    st.markdown('<div class="pbi-section-title">Export GPS27 parquet</div>', unsafe_allow_html=True)
    gps27_dir = preset_root / "GPS27"
    parquet_path = gps27_dir / "gps27_scenarios.parquet"
    st.write("Build a single parquet file for Power BI from all GPS27 scenario pickles.")
    st.caption(f"Source: {gps27_dir} | Output: {parquet_path}")
    if st.button("Run parquet build", key="gps27_parquet_build"):
        if not gps27_dir.exists():
            st.error(f"GPS27 folder not found: {gps27_dir}")
        else:
            with st.spinner("Building parquet from GPS27 scenarios..."):
                try:
                    summary = parquet_export.build_gps27_parquet(gps27_dir, parquet_path)
                except Exception as exc:
                    st.error(f"Parquet build failed: {exc}")
                else:
                    st.success(f"Parquet written to {summary.output_path} ({summary.rows:,} rows).")
                    if summary.row_counts:
                        st.write("Row counts:", summary.row_counts)

    st.markdown('<div class="pbi-section-title">Export GPS27 economic + regions parquet</div>', unsafe_allow_html=True)
    regions_parquet_path = gps27_dir / "gps27_regions_economic.parquet"
    regions_geojson_path = gps27_dir / "gps27_region_boundaries.geojson"
    st.write(
        "Build a tidy regional/economic parquet for Power BI "
        "(includes national/unmapped redistribution solved)."
    )
    st.caption(
        f"Source: {gps27_dir} | Outputs: {regions_parquet_path} and {regions_geojson_path}"
    )
    spread_regions_export = st.toggle(
        "Spread national numbers across regions",
        value=True,
        key="gps27_regions_export_spread_national_toggle",
        help=(
            "When enabled, national and unmapped rows are allocated to regions by national weights "
            "in the exported regional/economic parquet."
        ),
    )
    include_both_spread_modes = st.toggle(
        "Include both spread modes in parquet",
        value=False,
        key="gps27_regions_export_include_both_modes_toggle",
        help=(
            "When enabled, exports both `spread_on` and `spread_off` rows with a `spread_mode` column "
            "so Power BI can switch without rebuilding."
        ),
    )
    if st.button("Run economic/regions parquet build", key="gps27_regions_parquet_build"):
        if not gps27_dir.exists():
            st.error(f"GPS27 folder not found: {gps27_dir}")
        else:
            with st.spinner("Building regional/economic parquet from GPS27 scenarios..."):
                try:
                    summary = parquet_export.build_gps27_regions_economic_parquet(
                        gps27_dir,
                        regions_parquet_path,
                        spread_national=bool(spread_regions_export),
                        spread_unmapped=bool(spread_regions_export),
                        include_both_spread_modes=bool(include_both_spread_modes),
                    )
                except Exception as exc:
                    st.error(f"Economic/regions parquet build failed: {exc}")
                else:
                    st.success(f"Parquet written to {summary.output_path} ({summary.rows:,} rows).")
                    if summary.boundaries_geojson_path:
                        st.success(f"GeoJSON written to {summary.boundaries_geojson_path}.")
                    if summary.row_counts:
                        st.write("Row counts:", summary.row_counts)

    st.markdown('<div class="pbi-section-title">Run optimiser</div>', unsafe_allow_html=True)
    if not scenario_folders:
        st.info("Create a scenario folder before running the optimiser.")
        return

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
        selected_folder = folder_lookup[target_label]
        default_run_name = ""
        if selected_folder.metadata:
            default_run_name = str(selected_folder.metadata.get("name") or "")
        if not default_run_name:
            default_run_name = selected_folder.name
        run_name_key = "scenario_run_name"
        run_name_folder_key = "scenario_run_name_folder"
        if (
            run_name_key not in st.session_state
            or st.session_state.get(run_name_folder_key) != selected_folder.name
        ):
            st.session_state[run_name_key] = default_run_name
            st.session_state[run_name_folder_key] = selected_folder.name
        run_name = st.text_input(
            "Run name (used for file prefix)",
            key=run_name_key,
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
                "StartFY": st.column_config.NumberColumn(
                    "Forced start (FY)", help="Leave blank to let the optimiser decide."
                ),
            },
        )
        cost_type_options = list(settings.optimisation.cost_types)
        cost_types = st.multiselect("Cost types", cost_type_options, default=cost_type_options)
        scenario_key_options = list(solver_core.BENEFIT_SCENARIOS.keys())
        # Split combined benefit scenario codes (e.g., "A60") into method and duration parts.
        scenario_method_lookup: Dict[str, str] = {}
        scenario_year_lookup: Dict[str, str] = {}
        scenario_methods: Set[str] = set()
        scenario_years: Set[str] = set()
        scenario_pattern = re.compile(r"(?P<method>[A-Za-z]+)(?P<years>\d+)$")
        for key in scenario_key_options:
            match = scenario_pattern.match(key)
            if match:
                method = match.group("method")
                years = match.group("years")
            else:
                method = key
                years = ""
            scenario_method_lookup[key] = method
            scenario_year_lookup[key] = years
            scenario_methods.add(method)
            if years:
                scenario_years.add(years)
        method_options = sorted(scenario_methods)
        year_sort = lambda val: (0, int(val)) if val.isdigit() else (1, val)
        year_options = sorted(scenario_years, key=year_sort)
        method_labels = {
            "A": "Method A (more aggressive accrual)",
            "B": "Method B (less aggressive accrual)",
        }
        col_method, col_years = st.columns(2)
        selected_methods = col_method.multiselect(
            "Benefits accrual method (A = more aggressive than B)",
            method_options,
            default=method_options,
            key="scenario_benefit_methods",
            format_func=lambda code: method_labels.get(code, f"Method {code}"),
        )
        selected_years = col_years.multiselect(
            "Benefit duration (years)",
            year_options,
            default=year_options,
            key="scenario_benefit_years",
            format_func=lambda years: f"{years} years" if years.isdigit() else years,
        )
        scenario_keys = [
            key
            for key in scenario_key_options
            if scenario_method_lookup[key] in selected_methods
            and (
                not scenario_year_lookup[key]
                or scenario_year_lookup[key] in selected_years
            )
        ]
        dims_options_raw = [str(dim) for dim in getattr(data, "dims", [])] or ["Total"]
        if "Total" in dims_options_raw:
            dims_options = ["Total"] + [dim for dim in dims_options_raw if dim != "Total"]
        else:
            dims_options = dims_options_raw
        objective_dims = st.multiselect("Objective dimensions", dims_options, default=dims_options)
        col_cfg1, col_cfg2 = st.columns(2)
        start_fy = col_cfg1.number_input(
            "Start financial year",
            value=int(settings.optimisation.start_fy),
            step=1,
        )
        years = col_cfg2.number_input(
            "Planning horizon (years)",
            value=int(settings.optimisation.years),
            min_value=1,
            step=1,
        )
        st.markdown("Baseline annual envelopes & buffers ($m p.a.)")
        surplus_items = list(settings.optimisation.surplus_options_m.items())
        plus_defaults = list(settings.optimisation.plusminus_levels_m)
        max_rows = max(len(surplus_items), len(plus_defaults)) or 1
        envelope_rows: List[Dict[str, Optional[float]]] = []
        for idx in range(max_rows):
            code_value: Optional[str] = None
            annual_val: Optional[float] = None
            buffer_val: Optional[float] = None
            if idx < len(surplus_items):
                raw_code, annual_val = surplus_items[idx]
                annual_val = float(annual_val)
                if annual_val is not None:
                    code_value = f"s{int(round(annual_val))}"
                if not code_value and raw_code:
                    code_value = str(raw_code)
            if idx < len(plus_defaults):
                buffer_val = float(plus_defaults[idx])
            envelope_rows.append(
                {
                    "Code": code_value or f"env_{idx}",
                    "AnnualMillions": annual_val,
                    "BufferMillions": buffer_val,
                }
            )
        envelope_defaults = pd.DataFrame(envelope_rows)
        envelopes_editor = st.data_editor(
            envelope_defaults,
            num_rows="dynamic",
            hide_index=True,
            key="envelope_editor",
            column_config={
                "Code": st.column_config.TextColumn("Code", disabled=True),
                "AnnualMillions": st.column_config.NumberColumn("Annual $ (millions)", min_value=0.0),
                "BufferMillions": st.column_config.NumberColumn("Buffer (+/- $m)", min_value=0.0),
            },
            column_order=["AnnualMillions", "BufferMillions"],
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
        envelopes: Dict[str, float] = {}
        envelope_records = envelopes_editor.to_dict("records")
        for idx, row in enumerate(envelope_records):
            value = row.get("AnnualMillions")
            if pd.isna(value):
                continue
            value_f = float(value)
            base_key = f"s{int(round(value_f))}"
            code = base_key
            suffix = 2
            while code in envelopes:
                code = f"{base_key}_{suffix}"
                suffix += 1
            envelopes[code] = value_f
        if not envelopes:
            st.warning("Provide at least one envelope value.")
            proceed = False
        plus_levels = []
        for row in envelope_records:
            buffer_val = row.get("BufferMillions")
            if pd.isna(buffer_val):
                continue
            plus_levels.append(float(buffer_val))
        plus_levels = sorted({round(val, 6) for val in plus_levels})
        if not plus_levels:
            st.info("No buffer values supplied; defaulting to [0.0].")
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
            forced_inputs[row["Project"]] = scenario_utils.ForcedStartInput(
                include=include_val,
                start=start_val,
            )
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
                    run_plusminus=True,
                    forced_start=forced_inputs,
                    time_limit=int(settings.optimisation.solve_seconds),
                    run_prefix=(run_name or "").strip(),
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
                st.rerun()




def main() -> None:
    st.set_page_config(page_title="Capital Programme Optimiser", layout="wide")
    inject_powerbi_theme()
    inject_kpi_card_theme()
    st.markdown('<div class="pbi-header">Capital Programme Optimiser</div>', unsafe_allow_html=True)
    st.markdown('<div class="pbi-header-underline"></div>', unsafe_allow_html=True)

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

    scenario_folders = sorted(
        scenario_folders,
        key=lambda folder: 0
        if str(getattr(folder, "kind", "")).strip().lower() == "preset"
        and str(getattr(folder, "name", "")).strip().upper() == "GPS27"
        else 1,
    )

    configured_cache = {
        "label": "Configured cache (settings.yaml)",
        "path": settings.cache_dir(),
        "folder": None,
    }

    gps27_option = None
    folder_options = []
    for folder in scenario_folders:
        label = f"{str(folder.kind).title()} / {folder.name}"
        if folder.is_default:
            label += " (default)"
        option = {
            "label": label,
            "path": folder.path,
            "folder": folder,
        }
        if str(getattr(folder, "kind", "")).strip().lower() == "preset" and str(getattr(folder, "name", "")).strip().upper() == "GPS27":
            gps27_option = option
        else:
            folder_options.append(option)

    cache_options = []
    if gps27_option is not None:
        cache_options.append(gps27_option)
    cache_options.append(configured_cache)
    cache_options.extend(folder_options)
    # ---- Scenario cache (single instance) ----
    labels = [opt["label"] for opt in cache_options]
    default_index = 0
    if st.session_state.get("selected_cache_label") in labels:
        default_index = labels.index(st.session_state["selected_cache_label"])

    st.sidebar.header("Scenario cache")
    selected_label = st.sidebar.selectbox(
        "Choose scenario bundle",
        labels,
        index=default_index,
        key="cache_bundle_select_sidebar",   # unique key - used only here
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
        st.sidebar.write(f"Last run finished: {last_run.get('finished_at', 'unknown')}")
        if last_run.get("output_files"):
            st.sidebar.write(f"Files: {len(last_run['output_files'])}")
    else:
        st.sidebar.caption("No recorded manifest details yet for this folder.")

    st.sidebar.divider()
    st.sidebar.header("Configuration")
    st.sidebar.write(f"Dashboard output folder: {settings.dashboard_output_dir()}")
    if st.sidebar.button("Reload cache data"):
        load_dashboard_data.clear()
        load_project_attribute_mapping.clear()
        load_project_bcr_mapping.clear()
        load_project_new_bcr_mapping.clear()
        try:
            load_region_mapping.cache_clear()
        except AttributeError:
            pass
        try:
            fetch_region_geojson.cache_clear()
        except AttributeError:
            pass
        st.rerun()
    # ---- end Scenario cache ----


    cache_sig = _cache_signature(selected_path)
    with st.spinner(f"Loading dashboard cache from {selected_path}..."):
        try:
            data = load_dashboard_data(
                str(selected_path), cache_sig, LOAD_DATA_SCHEMA_VERSION
            )
        except Exception as exc:
            st.error(f"Unable to load scenario cache from {selected_path}: {exc}")
            st.stop()

    with st.expander("Scenario selection", expanded=False):
        scenario_cols = st.columns(2)
        with scenario_cols[0]:
            st.subheader(f"{SCENARIO_PRIMARY_NAME} Profile")
            opt_profiles = profile_options(data) or [DEFAULT_PROFILE_LABEL]
            opt_default_index = _preferred_option_index(opt_profiles, "objective best")
            if st.session_state.get("opt_profile_select") not in opt_profiles:
                st.session_state["opt_profile_select"] = opt_profiles[opt_default_index]
            selected_opt_profile = st.selectbox(
                f"{SCENARIO_PRIMARY_NAME} profile",
                opt_profiles,
                key="opt_profile_select",
                label_visibility="collapsed",
            )
        with scenario_cols[1]:
            st.subheader(f"{SCENARIO_COMPARISON_NAME} Profile")
            cmp_profiles = profile_options(data) or [DEFAULT_PROFILE_LABEL]
            cmp_default_index = _preferred_option_index(cmp_profiles, "217 start")
            if st.session_state.get("cmp_profile_select") not in cmp_profiles:
                st.session_state["cmp_profile_select"] = cmp_profiles[cmp_default_index]
            selected_cmp_profile = st.selectbox(
                f"{SCENARIO_COMPARISON_NAME} profile",
                cmp_profiles,
                key="cmp_profile_select",
                label_visibility="collapsed",
            )

    with st.expander("Advanced filters", expanded=False):
        adv_opt_col, adv_cmp_col = st.columns(2)
        with adv_opt_col:
            opt_selection = scenario_selector(
                name=resolve_selection_label(
                    None,
                    fallback=SCENARIO_PRIMARY_NAME,
                    profile_choice=selected_opt_profile,
                ),
                data=data,
                settings=settings,
                prefer_comparison=False,
                key_prefix="opt",
                profile_name=selected_opt_profile,
                show_benefit_controls=False,
            )
        with adv_cmp_col:
            comp_selection = scenario_selector(
                name=resolve_selection_label(
                    None,
                    fallback=SCENARIO_COMPARISON_NAME,
                    profile_choice=selected_cmp_profile,
                ),
                data=data,
                settings=settings,
                prefer_comparison=True,
                key_prefix="cmp",
                profile_name=selected_cmp_profile,
                show_benefit_controls=False,
            )

        st.divider()
        basis_cols = st.columns(2)
        with basis_cols[0]:
            default_cost_basis = _resolve_value_basis(
                "npv_cost_value_basis",
                "npv_apply_cost_discount",
                default=VALUE_BASIS_PV,
            )
            cost_basis = st.selectbox(
                "Costs value basis",
                list(VALUE_BASIS_OPTIONS),
                index=list(VALUE_BASIS_OPTIONS).index(default_cost_basis),
                key="npv_cost_value_basis",
                help=(
                    "Choose how costs are expressed in KPI cards and breakdown charts.\n\n"
                    f"PV: discounted using NZTA MBCM rates ({mbcm_discount_label()})."
                ),
            )
        with basis_cols[1]:
            default_benefit_basis = _resolve_value_basis(
                "npv_benefit_value_basis",
                "npv_apply_discount",
                default=VALUE_BASIS_PV,
            )
            benefit_basis = st.selectbox(
                "Benefits value basis",
                list(VALUE_BASIS_OPTIONS),
                index=list(VALUE_BASIS_OPTIONS).index(default_benefit_basis),
                key="npv_benefit_value_basis",
                help=(
                    "Choose how benefits are expressed in KPI cards and breakdown charts.\n\n"
                    f"PV: discounted using NZTA MBCM rates ({mbcm_discount_label()})."
                ),
            )

        pv_controls_enabled = benefit_basis == VALUE_BASIS_PV or cost_basis == VALUE_BASIS_PV
        npv_horizon_options = [60, 50, 40]
        if pv_controls_enabled:
            default_horizon = st.session_state.get("npv_horizon_selection", npv_horizon_options[0])
            try:
                default_horizon = int(default_horizon)
            except (TypeError, ValueError):
                default_horizon = npv_horizon_options[0]
            if default_horizon not in npv_horizon_options:
                default_horizon = npv_horizon_options[0]
                st.session_state["npv_horizon_selection"] = default_horizon
            st.selectbox(
                "PV horizon (years)",
                npv_horizon_options,
                index=npv_horizon_options.index(default_horizon),
                key="npv_horizon_selection",
            )
        else:
            st.selectbox(
                "PV horizon (years)",
                ["N/A"],
                index=0,
                disabled=True,
                key="npv_horizon_selection_na",
            )

        # Maintain legacy toggle keys for existing charts that expect boolean switches.
        st.session_state["npv_apply_discount"] = benefit_basis == VALUE_BASIS_PV
        st.session_state["npv_apply_cost_discount"] = cost_basis == VALUE_BASIS_PV

    opt_series = build_timeseries(data, opt_selection)
    cmp_series = build_timeseries(data, comp_selection)
    raw_opt_label = resolve_selection_label(
        opt_selection,
        fallback=SCENARIO_PRIMARY_NAME,
        profile_choice=selected_opt_profile,
    )
    raw_cmp_label = resolve_selection_label(
        comp_selection,
        fallback=SCENARIO_COMPARISON_NAME,
        profile_choice=selected_cmp_profile,
    )

    def _normalize_label(value: Optional[str]) -> str:
        return (value or "").strip().lower()

    normalized_opt = _normalize_label(raw_opt_label)
    normalized_cmp = _normalize_label(raw_cmp_label)
    duplicate_labels = bool(normalized_opt) and normalized_opt == normalized_cmp
    if duplicate_labels:
        base_opt_label = (raw_opt_label or "").strip() or SCENARIO_PRIMARY_NAME
        base_cmp_label = (raw_cmp_label or "").strip() or SCENARIO_COMPARISON_NAME
        numbered_opt_label = f"{base_opt_label} (scenario 1)"
        numbered_cmp_label = f"{base_cmp_label} (scenario 2)"
        opt_selection.display_label = numbered_opt_label
        comp_selection.display_label = numbered_cmp_label
        opt_label = numbered_opt_label
        cmp_label = numbered_cmp_label
    else:
        opt_selection.display_label = None
        comp_selection.display_label = None
        opt_label = raw_opt_label
        cmp_label = raw_cmp_label

    set_scenario_display_labels(opt_label, cmp_label)
    nav_previews: Dict[str, List[Dict[str, Any]]] | None = None
    if ENABLE_PREVIEW_NAVIGATION:
        nav_previews = collect_navigation_previews(
            data=data,
            opt_selection=opt_selection,
            comp_selection=comp_selection,
            opt_series=opt_series,
            cmp_series=cmp_series,
            opt_label=opt_label,
            cmp_label=cmp_label,
            settings=settings,
            cache_signature=cache_sig,
        )

    if st.session_state.get("active_tab") not in NAV_TABS:
        fallback_tab = st.session_state.get("pbi_nav")
        st.session_state["active_tab"] = fallback_tab if fallback_tab in NAV_TABS else NAV_TABS[0]
    nav_col, content_col = st.columns((2.2, 7.8), gap="large")

    with nav_col:
        active_tab = render_powerbi_navigation(
            st.session_state["active_tab"],
            key="active_tab",
            orientation="vertical",
            previews=nav_previews,
        )

    if ENABLE_PREVIEW_NAVIGATION:
        st.session_state["active_tab"] = active_tab

    with content_col:
        benefit_basis = _resolve_value_basis("npv_benefit_value_basis", "npv_apply_discount")
        if benefit_basis == VALUE_BASIS_PV:
            try:
                summary_horizon_years = int(st.session_state.get("npv_horizon_selection", 60))
            except (TypeError, ValueError):
                summary_horizon_years = 60
            npv_summary_label = format_npv_context(summary_horizon_years, apply_discount=True)
        elif benefit_basis == VALUE_BASIS_NOMINAL:
            npv_summary_label = "Benefits (nominal)"
        else:
            npv_summary_label = "Benefits ($2025)"

        with st.expander("Programme summary", expanded=False):
            render_programme_kpis(
                opt_series,
                cmp_series,
                data=data,
                opt_selection=opt_selection,
                cmp_selection=comp_selection,
            )

        download_tables: Dict[str, pd.DataFrame] = {}

        if active_tab == "Demo":
            download_tables.update(
                render_overview_tab(
                    data,
                    opt_selection,
                    comp_selection,
                    opt_series,
                    cmp_series,
                    opt_label=opt_label,
                    cmp_label=cmp_label,
                    npv_label=npv_summary_label,
                )
            )
        elif active_tab == "Cash Flow":
            download_tables.update(
                render_cash_flow_tab(
                    data,
                    opt_selection,
                    comp_selection,
                    opt_series,
                    cmp_series,
                    opt_label=opt_label,
                    cmp_label=cmp_label,
                )
            )
        elif active_tab == "Analysis":
            download_tables.update(
                render_analysis_tab(
                    data,
                    opt_selection,
                    comp_selection,
                    opt_series,
                    cmp_series,
                    opt_label=opt_label,
                    cmp_label=cmp_label,
                )
            )
        elif active_tab == "Benefits":
            download_tables.update(
                render_benefits_tab(
                    data,
                    opt_selection,
                    comp_selection,
                    opt_series,
                    cmp_series,
                    opt_label=opt_label,
                    cmp_label=cmp_label,
                )
            )
        elif active_tab == "Regions":
            download_tables.update(
                render_region_tab(
                    data,
                    settings=settings,
                    opt_selection=opt_selection,
                    comp_selection=comp_selection,
                    opt_label=opt_label,
                    cmp_label=cmp_label,
                    cache_signature=cache_sig,
                )
            )
        elif active_tab == "Delivery":
            download_tables.update(
                render_delivery_tab(
                    data,
                    opt_selection,
                    comp_selection,
                    opt_series,
                    cmp_series,
                    opt_label=opt_label,
                    cmp_label=cmp_label,
                )
            )
        elif active_tab == "Programme Schedule":
            download_tables.update(
                render_gantt_tab(
                    data,
                    opt_selection,
                    comp_selection,
                    opt_label=opt_label,
                    cmp_label=cmp_label,
                )
            )
        elif active_tab == "Scenario Manager":
            render_scenarios_tab(
                settings,
                preset_root,
                saved_root,
                scenario_folders,
                opt_selection=opt_selection,
                comp_selection=comp_selection,
                data=data,
                opt_label=opt_label,
                cmp_label=cmp_label,
                download_tables=download_tables,
            )

        if download_tables and active_tab != "Demo":
            render_export_download(download_tables, active_tab=active_tab)



if __name__ == "__main__":
    main()
