"""Optimisation core module extracted from the legacy notebook."""
from __future__ import annotations

from pathlib import Path
import pickle
import time
import re
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import highspy as hs  # type: ignore

from capital_programme_optimiser.config import load_settings

SETTINGS = load_settings()
ROOT = SETTINGS.root
DATA_FILE = SETTINGS.scoring_workbook()
COST_TYPES_RUN = list(SETTINGS.optimisation.cost_types)
BENEFIT_SCENARIOS = dict(SETTINGS.data.benefit_scenarios)
PRIMARY_OBJECTIVE_DIMS_TO_RUN: List[str] = []
FORCED_START = {
    name: {"start": rule.start, "include": rule.include}
    for name, rule in SETTINGS.forced_start.items()
}
SURPLUS_OPTIONS_M = {k: float(v) for k, v in SETTINGS.optimisation.surplus_options_m.items()}
PLUSMINUS_LEVELS_M = [float(v) for v in SETTINGS.optimisation.plusminus_levels_m]
BENEFIT_DISCOUNT_RATE = float(SETTINGS.optimisation.benefit_discount_rate)
SOLVE_SECONDS = int(SETTINGS.optimisation.solve_seconds)
VERBOSE = int(SETTINGS.optimisation.verbose)
CFG = {
    "YEARS": int(SETTINGS.optimisation.years),
    "START_FY": int(SETTINGS.optimisation.start_fy),
    "MAX_STARTS": int(SETTINGS.optimisation.max_starts),
    "CASH_PV_RATE": float(SETTINGS.optimisation.cash_pv_rate),
}
EARLY_STOP_ON = bool(SETTINGS.optimisation.early_stop)
EARLY_STOP_PCT = float(SETTINGS.optimisation.early_stop_pct)
RUN_BENEFIT_PLUSMINUS = bool(SETTINGS.optimisation.run_benefit_plusminus)

CACHE = SETTINGS.cache_dir()
CACHE.mkdir(parents=True, exist_ok=True)

ProgressCallback = Callable[[str, Dict[str, object]], None]
PROGRESS_LISTENER: Optional[ProgressCallback] = None


def set_progress_listener(listener: Optional[ProgressCallback]) -> None:
    """Register a callback for long-run progress updates."""

    global PROGRESS_LISTENER
    PROGRESS_LISTENER = listener


def _notify(stage: str, **payload: object) -> None:
    if PROGRESS_LISTENER is None:
        return
    try:
        PROGRESS_LISTENER(stage, payload)
    except Exception as exc:  # pragma: no cover - best effort logging
        print(f"[warn] progress listener error: {exc}")

# ---------- Dimension tags (used in filenames) ------------------------
_DIM_SHORT = {
    "Total": "TOT",
    "Healthy and safe people": "HSP",
    "Inclusive access": "INC",
    "Environmental sustainability": "ENV",
    "Economic Prosperity": "ECO",
    "Urban Development": "URB",
    "Resilience and Security": "RES",
}
def dim_short(dim: str) -> str:
    if dim in _DIM_SHORT: return _DIM_SHORT[dim]
    tokens = re.findall(r"[A-Za-z0-9]+", dim)
    if not tokens:
        return "DIM"
    code = "".join(t[:3] for t in tokens)[:8].upper()
    return code or "DIM"

# ---------- Normalisation helpers ------------------------------------
def _norm_name(s) -> str:
    if s is None: return ""
    txt = str(s).replace("\xa0", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    low = {str(c).strip().lower(): c for c in df.columns}
    for want in candidates:
        if want in low: return low[want]
    return None

# ⭐ project selection/forcing helpers ---------------------------------
def _keep_and_forced_from_rules(rules: dict, variants: dict) -> tuple[set[str], dict[str, int]]:
    vkeys = {_norm_name(v) for v in variants.keys()}

    include_true = set()
    exclude_true = set()
    start_map_all = {}

    for raw_name, spec in (rules or {}).items():
        pname = _norm_name(raw_name)
        if not isinstance(spec, dict):
            start = spec
            include = None
        else:
            start   = spec.get("start", None)
            include = spec.get("include", None)

        if include is True:
            include_true.add(pname)
        elif include is False:
            exclude_true.add(pname)

        if start is not None:
            try:
                start_map_all[pname] = int(start)
            except Exception:
                raise ValueError(f"Invalid 'start' for project '{raw_name}': {start!r}")

    whitelist_mode = len(include_true) > 0
    if whitelist_mode:
        keep = vkeys & include_true
    else:
        keep = vkeys - exclude_true

    forced_for_kept = {p: y for p, y in start_map_all.items() if p in keep}
    mode = "WHITELIST" if whitelist_mode else "BLACKLIST"
    print(f"   [rules] Mode={mode}; kept={len(keep)}/{len(vkeys)}; forced={len(forced_for_kept)}")
    return keep, forced_for_kept

# ═════════════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════════════
def _calendar_years() -> list[int]:
    start = int(CFG["START_FY"])
    ny    = int(CFG["YEARS"])
    return [start + i for i in range(ny)]

def _pick_project_name_column(df: pd.DataFrame) -> str:
    proj_cols = [c for c in df.columns if str(c).strip().lower() == "project"]
    if not proj_cols:
        raise RuntimeError("No 'Project' column found in Costs sheet.")
    return proj_cols[-1]

def load_costs_and_calendar(cost_type: str):
    """
    Read Costs worksheet and align to CFG window strictly.
    Returns:
      projects: dict[project] -> { "cost":totalM, "dur":dur, "spend":[...M...] }
      variants: dict[variant_id] == { "base":project, "dur":dur, "curve":"raw", "spend": series }
      year_list: list[int] -> equals CFG window
    """
    df = pd.read_excel(DATA_FILE, sheet_name="Costs", engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    all_year_cols = [c for c in df.columns if str(c).isdigit()]
    if not all_year_cols:
        raise RuntimeError("No year columns detected in 'Costs' (numeric headers required).")
    all_year_cols_sorted = sorted(all_year_cols, key=int)

    horizon_years = _calendar_years()
    # Make a map from year to column presence
    have_year = {int(c): c for c in all_year_cols_sorted}
    # Build the exact horizon slice (filling zeros if missing in the sheet)
    yr_cols = [have_year.get(y, None) for y in horizon_years]
    year_list = horizon_years[:]  # strict

    name_col = _pick_project_name_column(df)
    ct_col   = "Cost type"
    if ct_col not in df.columns or "Duration" not in df.columns:
        raise RuntimeError("Missing 'Cost type' or 'Duration' in 'Costs' sheet.")

    cut = df[df[ct_col].astype(str).str.strip() == str(cost_type).strip()].copy()
    if cut.empty:
        raise RuntimeError(f"No rows in 'Costs' for Cost type = '{cost_type}'")

    projects, variants = {}, {}
    for _, row in cut.iterrows():
        p = _norm_name(row[name_col])

        # Build spend vector over the CFG horizon (fill zero where column absent)
        vals = []
        for y, c in zip(horizon_years, yr_cols):
            if c is None:
                vals.append(0.0)
            else:
                vals.append(pd.to_numeric(row[c], errors="coerce"))

        spend_series = pd.Series(vals).fillna(0.0) / 1_000_000.0  # to millions
        nz = spend_series.to_numpy().nonzero()[0]
        if nz.size == 0:
            continue
        series = spend_series.iloc[nz.min(): nz.max() + 1].tolist()
        total_cost = float(sum(series))
        projects[p] = {"cost": total_cost, "dur": int(len(series)), "spend": series}
        variants[p] = {"base": p, "dur": int(len(series)), "curve": "raw", "spend": series}

    return projects, variants, year_list

def load_benefits_table(sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE, sheet_name=sheet_name, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    proj_col = _find_col(df, ["project"])
    if not proj_col:
        raise RuntimeError(f"Benefits sheet '{sheet_name}' must have a 'Project' column.")
    if proj_col != "Project":
        df.rename(columns={proj_col: "Project"}, inplace=True)

    dim_col = _find_col(df, ["dimension", "objective dimension", "objective_dimension"])
    if not dim_col:
        raise RuntimeError(f"Benefits sheet '{sheet_name}' must have a 'Dimension' (or 'Objective dimension') column.")
    if dim_col != "Dimension":
        df.rename(columns={dim_col: "Dimension"}, inplace=True)

    # collect any 't+K' style columns
    tcols_info = []
    for c in df.columns:
        m = re.fullmatch(r"[tT]\s*\+\s*(\d+)", str(c).strip())
        if m:
            k = int(m.group(1))
            tcols_info.append((k, c))
    if not tcols_info:
        raise RuntimeError(f"No 't+K' columns found in Benefits sheet '{sheet_name}'.")
    tcols_info.sort(key=lambda x: x[0])
    for _, c in tcols_info:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df

def extract_dim_flows_per_project(benef_df: pd.DataFrame):
    tcols_info = []
    for c in benef_df.columns:
        m = re.fullmatch(r"[tT]\s*\+\s*(\d+)", str(c).strip())
        if m:
            k = int(m.group(1))
            tcols_info.append((k, c))
    tcols_info.sort(key=lambda x: x[0])
    if not tcols_info:
        raise RuntimeError("No 't+K' columns found in Benefits sheet.")
    tcols_sorted = [c for _, c in tcols_info]

    df = benef_df.copy()
    df["Project"] = df["Project"].map(_norm_name)
    df["Dimension"] = df["Dimension"].map(_norm_name)

    for c in tcols_sorted:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    dims_present = df["Dimension"].dropna().astype(str).unique().tolist()
    flows_by_dim: dict[str, dict[str, list[float]]] = {}

    for dim, grp_dim in df.groupby("Dimension"):
        dim = _norm_name(dim)
        flows_by_dim[dim] = {}
        for project, grp_proj in grp_dim.groupby("Project"):
            row = grp_proj.iloc[0]
            flows_by_dim[dim][_norm_name(project)] = row[tcols_sorted].to_numpy(dtype=float).tolist()

    # Ensure 'Total'
    has_total = any(d.strip().lower() == "total" for d in flows_by_dim.keys())
    if not has_total:
        all_dims = [d for d in flows_by_dim.keys()]
        flows_by_dim["Total"] = {}
        projects = set()
        for d in all_dims:
            projects.update(flows_by_dim[d].keys())
        for p in projects:
            acc = None
            for d in all_dims:
                vec = flows_by_dim[d].get(p, None)
                if vec is None:
                    continue
                acc = vec if acc is None else [a + b for a, b in zip(acc, vec)]
            flows_by_dim["Total"][p] = acc if acc is not None else [0.0] * len(tcols_sorted)

    dims_order = [_norm_name(d) for d in dims_present if _norm_name(d) in flows_by_dim]
    if "Total" in flows_by_dim and "Total" not in dims_order:
        dims_order.append("Total")
    if "Total" in dims_order:
        dims_order = [d for d in dims_order if d != "Total"] + ["Total"]

    return dims_order, flows_by_dim, tcols_sorted

def prepare_kernels_by_dim(variants: dict, flows_by_dim: dict[str, dict[str, list[float]]]) -> dict:
    kernels_by_dim: dict[str, dict] = {}
    for dim, proj2flows in flows_by_dim.items():
        kers = {}
        for v, meta in variants.items():
            base = meta["base"]
            dur  = int(meta["dur"])
            seq  = proj2flows.get(base, None)
            if seq is None:
                kers[v] = [0.0] * dur
                continue
            kers[v] = [0.0] * dur + [float(x) for x in seq]
        kernels_by_dim[dim] = kers
    return kernels_by_dim

# ═════════════════════════════════════════════════════════════════════
#  Helper math for envelope & pruning & warm-start
# ═════════════════════════════════════════════════════════════════════
def envelope_bounds_from_dyn_cfg(ny: int, baseline_surplusM: float, dyn_cfg: dict):
    """Return y_lb, y_ub, conserve_total flag, and total_cap(if conserve)."""
    lb = np.full(ny, float(dyn_cfg["MIN_M"]), dtype=float)
    ub = np.full(ny, float(dyn_cfg["MAX_M"]), dtype=float)
    conserve = bool(dyn_cfg.get("CONSERVE_TOTAL", False))
    total_cap = float(baseline_surplusM) * ny if conserve else np.inf
    return lb, ub, conserve, total_cap

def compute_obj_coeff_map_for_dim(variants: dict, kernels_for_dim: dict, ny: int, r: float) -> dict:
    disc = np.array([(1.0 + r) ** t for t in range(ny)], dtype=float)
    coeff = {}
    for v, meta in variants.items():
        dur = int(meta["dur"])
        ker = kernels_for_dim.get(v, [])
        if not ker:
            continue
        for s in range(0, ny - dur + 1):
            z = 0.0
            for k, f in enumerate(ker):
                if f == 0.0: continue
                t = s + k
                if 0 <= t < ny:
                    z += float(f) / float(disc[t])
            coeff[(v, s)] = z
    return coeff

def _prefix(arr: np.ndarray) -> np.ndarray:
    return np.cumsum(arr, dtype=float)

def prune_impossible_starts(variants: dict,
                            ny: int,
                            y_ub: np.ndarray,
                            conserve_total: bool,
                            total_cap: float,
                            coeff_primary_map: dict,
                            coeff_total_map: dict,
                            forced_map: dict,
                            start_fy: int) -> dict:
    ny = int(ny)
    allowed = {}
    y_ub_prefix = _prefix(y_ub)

    for v, meta in variants.items():
        dur = int(meta["dur"])
        spend = np.asarray(meta["spend"], dtype=float)
        total_cost = float(spend.sum())
        starts = []

        forced_s = None
        if v in forced_map:
            yr = int(forced_map[v])
            forced_s = int(yr - start_fy)

        if conserve_total and total_cost > total_cap + 1e-9:
            if forced_s is not None and 0 <= forced_s <= ny - dur:
                allowed[v] = [forced_s]
            else:
                allowed[v] = []
            continue

        for s in range(0, ny - dur + 1):
            if (coeff_primary_map.get((v, s), 0.0) == 0.0 and
                coeff_total_map.get((v, s), 0.0) == 0.0):
                if forced_s is not None and s == forced_s:
                    pass
                else:
                    continue

            # per-year vs y_ub
            per_year_ok = True
            for i in range(dur):
                t = s + i
                if spend[i] > y_ub[t] + 1e-12:
                    per_year_ok = False
                    break
            if not per_year_ok:
                if forced_s is not None and s == forced_s:
                    starts.append(s)
                continue

            # prefix(project) vs y_ub
            inc = np.zeros(ny, dtype=float)
            inc[s:s+dur] = spend
            if np.any(_prefix(inc) - y_ub_prefix > 1e-12):
                if forced_s is not None and s == forced_s:
                    starts.append(s)
                continue

            starts.append(s)

        if forced_s is not None and 0 <= forced_s <= ny - dur and forced_s not in starts:
            starts.append(forced_s)

        allowed[v] = sorted(starts)

    return allowed

# ---------- GREEDY WARM-START (feasible incumbent via prefix slack) ---
def greedy_warm_start(variants: dict,
                      allowed_starts: dict,
                      ny: int,
                      y_lb: np.ndarray,
                      y_ub: np.ndarray,
                      conserve_total: bool,
                      max_starts_per_year: int,
                      coeff_primary_map: dict):
    ny = int(ny)
    y_lb = np.asarray(y_lb, dtype=float)
    y_ub = np.asarray(y_ub, dtype=float)
    y_lb_prefix = _prefix(y_lb)

    slack = y_ub - y_lb
    slackprefix = _prefix(slack)

    spend_year_sel = np.zeros(ny, dtype=float)
    starts_count = np.zeros(ny, dtype=int)
    selected = {}

    order = []
    for v, meta in variants.items():
        cand = allowed_starts.get(v, [])
        if not cand:
            continue
        best_pv = max(coeff_primary_map.get((v, s), 0.0) for s in cand)
        denom = float(sum(meta["spend"]))
        dens = (best_pv / denom) if denom > 1e-12 else (float('inf') if best_pv > 0 else 0.0)
        order.append((dens, v))
    order.sort(reverse=True)

    for _, v in order:
        meta = variants[v]
        dur = int(meta["dur"])
        spend = np.asarray(meta["spend"], dtype=float)

        starts = allowed_starts.get(v, [])
        starts_sorted = sorted(starts, key=lambda s: coeff_primary_map.get((v, s), 0.0), reverse=True)

        prefix_current = _prefix(spend_year_sel)
        for s in starts_sorted:
            if starts_count[s] >= max_starts_per_year:
                continue
            inc = np.zeros(ny, dtype=float)
            inc[s:s+dur] = spend
            incprefix = _prefix(inc)

            deficit_prev = np.maximum(0.0, prefix_current - y_lb_prefix)
            deficit_new  = np.maximum(0.0, (prefix_current + incprefix) - y_lb_prefix)
            delta_deficit = deficit_new - deficit_prev

            if np.any(delta_deficit - slackprefix > 1e-9):
                continue

            spend_year_sel += inc
            starts_count[s] += 1
            slackprefix -= delta_deficit
            selected[v] = s
            break

    lb_primary = 0.0
    for v, s in selected.items():
        lb_primary += coeff_primary_map.get((v, s), 0.0)

    return selected, float(lb_primary)

# ═════════════════════════════════════════════════════════════════════
#  HiGHS wrapper & model builder
# ═════════════════════════════════════════════════════════════════════
class _Params:
    def __init__(self):
        self.TimeLimit = None
        self.MIPGap = None
        self.OutputFlag = True

class HighsModel:
    def __init__(self, ny: int):
        self.h = hs.Highs()
        self.Params = _Params()
        self.Status = None
        self.SolCount = 0
        self._ny = ny
        self._x_idx = {}      # (v,s) -> col
        self._y_idx = None    # dict[t] -> col
        self._num_cols = 0
        self._y_coeff = None
        self._solution = None

    def _apply_options(self):
        if self.Params.TimeLimit is not None:
            self.h.setOptionValue("time_limit", float(self.Params.TimeLimit))
        if self.Params.MIPGap is not None:
            self.h.setOptionValue("mip_rel_gap", float(self.Params.MIPGap))
        self.h.setOptionValue("output_flag", bool(self.Params.OutputFlag))
        self.h.setOptionValue("log_to_console", bool(self.Params.OutputFlag))

    def optimize(self):
        self._apply_options()
        self.h.run()
        try:
            self._solution = self.h.getSolution()
            self.Status = self.h.getModelStatus()
            self.SolCount = 1 if (self._solution is not None and len(self._solution.col_value) > 0) else 0
        except Exception:
            self.SolCount = 0

    def col_values(self):
        return np.array(self._solution.col_value, dtype=float) if self._solution else np.zeros(self._num_cols, dtype=float)

def _build_yearly_spend_rows(m: HighsModel,
                             variants: dict,
                             ny: int,
                             allowed_starts: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Build sparse rows for Spend[t] = sum_{v,s in allowed[v]} spend_v[t-s] * x[v,s]
    Return per t: (indices, values) arrays.
    """
    rows_idx, rows_val = [], []
    for t in range(ny):
        idx_list, val_list = [], []
        for v, meta in variants.items():
            dur = int(meta["dur"])
            spend = meta["spend"]
            for s in allowed_starts.get(v, []):
                if s <= t < s + dur:
                    idx_list.append(m._x_idx[(v, s)])
                    val_list.append(float(spend[t - s]))
        rows_idx.append(np.asarray(idx_list, dtype=np.int32))
        rows_val.append(np.asarray(val_list, dtype=np.double))
    return rows_idx, rows_val

def _coalesce(idx: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Combine duplicate indices by summation, return sorted unique indices and sums."""
    if idx.size == 0:
        return idx, val
    order = np.argsort(idx)
    idx_sorted = idx[order]
    val_sorted = val[order]
    uniq_idx = [idx_sorted[0]]
    uniq_val = [val_sorted[0]]
    for j in range(1, idx_sorted.size):
        if idx_sorted[j] == uniq_idx[-1]:
            uniq_val[-1] += val_sorted[j]
        else:
            uniq_idx.append(idx_sorted[j])
            uniq_val.append(val_sorted[j])
    return np.asarray(uniq_idx, dtype=np.int32), np.asarray(uniq_val, dtype=np.double)

def build_model(vars_: dict, projects: dict, *,
                baseline_surplusM: float,
                ny: int,
                forced: dict | None = None,
                dyn_cfg: dict | None = None,
                fixed_envelope: bool = False,
                secondary_env_weights: list[float] | None = None,
                allowed_starts: dict | None = None) -> HighsModel:
    """
    Prefix capacity model:
        ∑_{u≤τ} Spend[u] ≤ ∑_{u≤τ} y[u]   ∀τ
    y[t] ∈ [MIN, MAX] from dyn_cfg; TOTAL CAP constraint added explicitly if requested.
    """
    forced = forced or {}
    m = HighsModel(ny=ny)

    # decision variables x[v,s] ∈ {0,1}
    col = 0
    allowed_starts = allowed_starts or {}
    for v, meta in vars_.items():
        dur = int(meta["dur"])
        starts = allowed_starts.get(v, list(range(0, ny - dur + 1)))

        # ensure forced start present
        if v in forced and forced[v] is not None:
            s_forced = int(forced[v] - CFG["START_FY"])
            if 0 <= s_forced <= ny - dur and s_forced not in starts:
                starts = list(starts) + [s_forced]

        for s in sorted(set(starts)):
            m.h.addVar(0.0, 1.0)
            m.h.changeColIntegrality(col, hs.HighsVarType.kInteger)
            m._x_idx[(v, s)] = col
            col += 1

    # forced starts (hard fix to 1)
    for p, yr in (forced or {}).items():
        if yr is None:
            continue
        p_norm = _norm_name(p)
        if p_norm not in vars_:
            continue
        d = int(vars_[p_norm]["dur"])
        s_forced = int(yr - CFG["START_FY"])
        if not (0 <= s_forced <= ny - d):
            raise ValueError(f"Forced start {yr} for '{p_norm}' is outside the model horizon [{CFG['START_FY']}, {CFG['START_FY']+ny-1}].")
        idx = m._x_idx.get((p_norm, s_forced), None)
        if idx is None:
            m.h.addVar(1.0, 1.0)
            m.h.changeColIntegrality(col, hs.HighsVarType.kInteger)
            m._x_idx[(p_norm, s_forced)] = col
            idx = col
            col += 1
        m.h.changeColBounds(idx, 1.0, 1.0)

    # envelope / capacity (dyn_cfg branch)
    if fixed_envelope:
        raise RuntimeError("fixed_envelope path is deprecated; use buffer bounds (±0) instead.")
    elif dyn_cfg is not None:
        y_lb, y_ub, conserve, total_cap = envelope_bounds_from_dyn_cfg(ny, baseline_surplusM, dyn_cfg)

        # y[t] variables within [lb,ub]
        m._y_idx = {}
        for t in range(ny):
            m.h.addVar(float(y_lb[t]), float(y_ub[t]))
            m._y_idx[t] = col
            col += 1

        # Yearly spend rows (sparse) with allowed starts
        spend_rows_idx, spend_rows_val = _build_yearly_spend_rows(m, vars_, ny, allowed_starts)

        # PREFIX constraints: sum_{u≤τ} y[u] - sum_{u≤τ} Spend[u] ≥ 0
        y_idx_list = [m._y_idx[t] for t in range(ny)]
        for tau in range(ny):
            # Build raw pieces
            idx_list = y_idx_list[:tau+1]
            val_list = [1.0] * (tau + 1)

            # Accumulate spends up to tau
            if tau >= 0:
                for u in range(tau + 1):
                    idx_u = spend_rows_idx[u]
                    val_u = spend_rows_val[u]
                    if idx_u.size:
                        idx_list.extend(idx_u.tolist())
                        val_list.extend((-val_u).tolist())

            # Coalesce duplicates for robustness
            idx = np.asarray(idx_list, dtype=np.int32)
            val = np.asarray(val_list, dtype=np.double)
            idx, val = _coalesce(idx, val)

            m.h.addRow(0.0, hs.kHighsInf, len(idx), idx, val)

        # Explicit TOTAL CAP on y if requested
        if conserve and np.isfinite(total_cap):
            idx = np.asarray([m._y_idx[t] for t in range(ny)], dtype=np.int32)
            val = np.ones(ny, dtype=np.double)
            total_rel = str(dyn_cfg.get("TOTAL_REL", "le")).lower()
            if total_rel == "eq":
                m.h.addRow(float(total_cap), float(total_cap), int(len(idx)), idx, val)
            else:
                m.h.addRow(-hs.kHighsInf, float(total_cap), int(len(idx)), idx, val)

        # Late-envelope penalty vector (optional)
        if secondary_env_weights is not None:
            y_coeff = np.zeros(col, dtype=np.double)
            for t, idx_col in m._y_idx.items():
                y_coeff[idx_col] = float(secondary_env_weights[t])
            m._y_coeff = y_coeff

    # Subset selection: at most one start per project
    for p in projects:
        cols = [m._x_idx[(p, s)] for (v, s) in m._x_idx.keys() if v == p]
        if cols:
            idx = np.asarray(cols, dtype=np.int32)
            val = np.ones(len(cols), dtype=np.double)
            m.h.addRow(0.0, 1.0, len(idx), idx, val)

    # Limit starts per year
    for t in range(ny):
        cols_t = [m._x_idx[(v, s)] for (v, s) in m._x_idx.keys() if s == t]
        if cols_t:
            idx = np.asarray(cols_t, dtype=np.int32)
            val = np.ones(len(cols_t), dtype=np.double)
            m.h.addRow(-hs.kHighsInf, float(CFG["MAX_STARTS"]), len(idx), idx, val)

    m._num_cols = col
    return m

# ---------- objective helpers ----------------------------------------
def build_obj_coeff(m: HighsModel, kernels_for_dim: dict, ny: int, r: float) -> np.ndarray:
    disc = np.array([(1.0 + r) ** t for t in range(ny)], dtype=float)
    coeff = np.zeros(m._num_cols, dtype=np.double)
    for (v, s), j in m._x_idx.items():
        ker = kernels_for_dim.get(v, [])
        if not ker:
            continue
        z = 0.0
        for k, f in enumerate(ker):
            if f == 0.0: continue
            t = s + k
            if 0 <= t < ny:
                z += float(f) / float(disc[t])
        coeff[j] = z
    return coeff

def _set_costs_allcols(h: hs.Highs, ncols: int, costs: np.ndarray):
    idx = np.arange(ncols, dtype=np.int32)
    h.changeColsCost(int(ncols), idx, np.asarray(costs, dtype=np.double))

def solve_lexico(m: HighsModel, objectives: list[dict]):
    values = []
    for i, obj in enumerate(objectives):
        sense = obj.get("sense", "max").lower()
        coeff = obj["coeff"]
        tol = obj.get("tol", 1e-9)

        costs = np.zeros(m._num_cols, dtype=np.double)
        if sense == "max":
            costs -= coeff
        else:
            costs += coeff
        _set_costs_allcols(m.h, m._num_cols, costs)

        m.optimize()
        if m.SolCount == 0:
            return values

        col = m.col_values()
        z = float(np.dot(coeff, col))
        values.append((obj.get("name", f"obj{i+1}"), z))

        idx_nz = np.nonzero(coeff)[0].astype(np.int32)
        if idx_nz.size > 0:
            val_nz = coeff[idx_nz].astype(np.double)
            if sense == "max":
                lb = z - max(tol, abs(z) * 1e-9)
                m.h.addRow(lb, hs.kHighsInf, int(len(idx_nz)), idx_nz, val_nz)
            else:
                ub = z + max(tol, abs(z) * 1e-9)
                m.h.addRow(-hs.kHighsInf, ub, int(len(idx_nz)), idx_nz, val_nz)

    return values

# ---------- pickle writer ---------------------------------------------
def dump_pickle(
    m: HighsModel,
    tag: str,
    suffix: str,
    baseline_surplusM: float,
    proj: dict,
    var: dict,
    kernels_by_dim: dict[str, dict],
    benefit_rate: float,
    scenario_name: str,
    primary_dim: str,
    year_list: list[int],
    *,
    cost_type: str,
    surplus_key: str,
    surplus_value: float,
    plus_level: float,
):
    def _complete(status: str, path: Path, message: str = "") -> None:
        _notify(
            "solve_complete",
            status=status,
            cache_file=path.name,
            cache_path=str(path),
            cost_type=cost_type,
            scenario=scenario_name,
            primary_dim=primary_dim,
            surplus_key=surplus_key,
            surplus_value=float(surplus_value),
            plus_level=float(plus_level),
            message=message,
        )

    if m.SolCount == 0:
        fn = CACHE / f"{tag}_{suffix}_noSol.pkl"
        pickle.dump({"status": m.Status, "message": "no feasible solution"}, fn.open("wb"))
        print(f"        ! saved stub {fn.name}")
        _complete("no_solution", fn, "no feasible solution")
        return

    col = m.col_values()

    # Selected starts
    sel = {}
    for (v, s), j in m._x_idx.items():
        if col[j] > 0.5:
            sel[v] = s
    if not sel:
        fn = CACHE / f"{tag}_{suffix}_noSol.pkl"
        pickle.dump({"status": m.Status, "message": "no selected starts"}, fn.open("wb"))
        print(f"        ! saved stub {fn.name}")
        _complete("no_solution", fn, "no selected starts")
        return

    start_fy = CFG["START_FY"]
    ny = m._ny
    fy = [start_fy + i for i in range(ny)]

    # Schedule table
    df_sched = pd.DataFrame([{
        "Project": var[v]["base"],
        "StartFY": start_fy + s,
        "EndFY":   start_fy + s + var[v]["dur"] - 1,
        "Dur":     var[v]["dur"],
        "Curve":   "raw",
        "Scenario": scenario_name,
        "PrimaryDim": primary_dim
    } for v, s in sel.items()]).sort_values(["StartFY","Project"], ignore_index=True)

    # Spend matrix (M NZD)
    df_sp = pd.DataFrame(0.0, index=list(proj.keys()), columns=fy)
    for v, s in sel.items():
        base = var[v]["base"]
        for i, amt in enumerate(var[v]["spend"]):
            t = s + i
            if 0 <= t < ny:
                df_sp.loc[base, fy[t]] += float(amt)
    df_sp.loc["Total Spend"] = df_sp.sum()

    # Envelope y
    if m._y_idx is not None:
        env = [float(col[m._y_idx[i]]) for i in range(ny)]
    else:
        env = [float(baseline_surplusM)] * ny
    df_env = pd.DataFrame({"Year": fy,
                           "Envelope": env,
                           "Delta": [np.nan] + [env[i] - env[i-1] for i in range(1, ny)]})

    # Cash flow (position/debt accounting)
    cash_rows, net_prev = [], 0.0
    for i, yr in enumerate(fy):
        spend = float(df_sp.loc["Total Spend", yr])
        env_i = float(env[i])
        net = net_prev + env_i - spend
        cash_rows.append({
            "Year": yr, "Envelope": env_i,
            "OpeningNet": net_prev, "Spend": spend, "ClosingNet": net,
            "OpeningCash": max(net_prev, 0.0), "OpeningDebt": max(-net_prev, 0.0),
            "ClosingCash": max(net, 0.0), "ClosingDebt": max(-net, 0.0)
        })
        net_prev = net
    df_cash = pd.DataFrame(cash_rows)

    # PV stocks (reporting only)
    disc_c = np.array([(1.0 + float(CFG["CASH_PV_RATE"])) ** i for i in range(ny)], dtype=float)
    pv_cash_val = float(np.sum(df_cash["ClosingCash"].to_numpy(dtype=float) / disc_c))
    pv_debt_val = float(np.sum(df_cash["ClosingDebt"].to_numpy(dtype=float) / disc_c))
    pv_total_val = pv_cash_val + pv_debt_val

    # ── Benefit flows by dimension
    dims = list(kernels_by_dim.keys())
    ben_dim = {d: np.zeros(ny, dtype=float) for d in dims}
    for v, s in sel.items():
        for d in dims:
            ker = kernels_by_dim[d].get(v, [])
            for k, f in enumerate(ker):
                t = s + k
                if 0 <= t < ny:
                    ben_dim[d][t] += float(f)

    # Wide & long tables
    df_ben_wide = pd.DataFrame({"Year": fy})
    for d in dims: df_ben_wide[d] = ben_dim[d]
    long_rows = []
    for d in dims:
        for i, yr in enumerate(fy):
            long_rows.append({"Year": yr, "Dimension": d, "BenefitFlow": float(ben_dim[d][i])})
    df_ben_long = pd.DataFrame(long_rows)

    total_name = "Total" if "Total" in ben_dim else (dims[-1] if dims else "Total")
    df_benefit_total = pd.DataFrame({"Year": fy, "BenefitFlow": ben_dim[total_name]})

    # PV by dimension
    disc_b = np.array([(1.0 + float(benefit_rate)) ** i for i in range(ny)], dtype=float)
    pv_by_dim = {d: float(np.sum(ben_dim[d] / disc_b)) for d in dims}
    ben_pv_total = float(np.sum(ben_dim[total_name] / disc_b))
    ben_pv_primary = pv_by_dim.get(primary_dim, np.nan)

    matched_dims = {d: int(np.sum(np.array(ben_dim[d]) != 0)) for d in dims}
    tot_nonzero = matched_dims.get(total_name, 0)
    print(f"        Benefit flows non‑zero years – {matched_dims} (Total={tot_nonzero})")

    out = {
        "schedule": df_sched,
        "cash_flow": df_cash,
        "spend": df_sp,
        "envelope": df_env,

        # benefits
        "benefit_flow": df_benefit_total,
        "benefit_flow_by_dim_wide": df_ben_wide,
        "benefit_flow_by_dim_long": df_ben_long,
        "benefit_pv_by_dim": pv_by_dim,
        "benefit_pv_total": ben_pv_total,
        "benefit_pv_primary": ben_pv_primary,
        "benefit_rate": float(benefit_rate),

        # PV stocks (reporting)
        "pv_cash": pv_cash_val,
        "pv_debt": pv_debt_val,
        "pv_total": pv_total_val,

        # meta
        "surplus_mode": ("dynamic" if m._y_idx is not None else "fixed"),
        "scenario": scenario_name,
        "objective": {"primary_dim": primary_dim, "secondary": "Total", "tertiary": "LateEnvelope"},
        "calendar": {"start_fy": CFG["START_FY"], "years": ny}
    }
    out["benefit_dim_flow"] = out["benefit_flow_by_dim_wide"]  # back‑compat

    fn = CACHE / f"{tag}_{suffix}.pkl"
    with fn.open("wb") as f:
        pickle.dump(out, f)
    print(f"        ✓ {fn.name}")
    _complete("ok", fn)

# ═════════════════════════════════════════════════════════════════════
#  Scenario runners
# ═════════════════════════════════════════════════════════════════════
def _apply_early_stop(m: HighsModel) -> None:
    if EARLY_STOP_ON and EARLY_STOP_PCT > 0:
        m.Params.MIPGap = EARLY_STOP_PCT / 100.0

def optimise_family_for(ct: str, scenario_key: str, scenario_sheet: str):
    # Load costs & calendar strictly in CFG window
    projects, variants, years = load_costs_and_calendar(ct)
    ny = CFG["YEARS"]
    assert years == _calendar_years(), "Internal: year vector mismatch with CFG window."

    print(f"   [calendar] Horizon: {CFG['START_FY']}..{CFG['START_FY']+ny-1}  (ny={ny})  MAX_STARTS={CFG['MAX_STARTS']}")

    # Apply include/exclude + forced start rules BEFORE benefits/kernels
    keep_set, forced_map = _keep_and_forced_from_rules(FORCED_START, variants)

    # Filter to kept set
    if keep_set != set(variants.keys()):
        variants = {v: meta for v, meta in variants.items() if v in keep_set}
        projects = {p: meta for p, meta in projects.items() if p in keep_set}

    # Load benefits and build per-dimension kernels
    benef_df = load_benefits_table(scenario_sheet)
    dims_order, flows_by_dim, _tcols = extract_dim_flows_per_project(benef_df)

    # Trim benefits to kept projects (avoids zero kernels clutter)
    for d in list(flows_by_dim.keys()):
        flows_by_dim[d] = {p: f for p, f in flows_by_dim[d].items() if p in variants}

    kernels_by_dim = prepare_kernels_by_dim(variants, flows_by_dim)

    # Sanity on benefit mappings
    for dim in dims_order:
        have = sum(1 for p in variants if p in flows_by_dim.get(dim, {}))
        print(f"   [benefit map] Dimension='{dim}': {have}/{len(variants)} projects matched")

    # Determine primary dims to run
    if PRIMARY_OBJECTIVE_DIMS_TO_RUN:
        dims_to_run = [d for d in PRIMARY_OBJECTIVE_DIMS_TO_RUN if d in kernels_by_dim]
    else:
        dims_to_run = list(dims_order)

    _notify(
        "family_start",
        cost_type=ct,
        scenario_key=scenario_key,
        scenario_sheet=scenario_sheet,
        dimensions=[str(d) for d in dims_to_run],
        surplus_keys=list(SURPLUS_OPTIONS_M.keys()),
        plus_levels=[float(p) for p in PLUSMINUS_LEVELS_M],
    )

    for dim_index, primary_dim in enumerate(dims_to_run):
        _notify(
            "dimension_start",
            cost_type=ct,
            scenario_key=scenario_key,
            scenario_sheet=scenario_sheet,
            primary_dim=primary_dim,
            dimension_index=dim_index,
            dimension_total=len(dims_to_run),
        )
        dim_code = dim_short(primary_dim)
        tag_base = f"{ct.replace(' ', '').replace('-', '')}_{scenario_key}_OBJ{dim_code}_BEN"
        print(f"\n>> Objective dimension: {primary_dim} ({dim_code})")

        # Precompute coeff maps for pruning & greedy
        coeff_primary_map = compute_obj_coeff_map_for_dim(variants, kernels_by_dim[primary_dim], ny, BENEFIT_DISCOUNT_RATE)
        total_name = "Total" if "Total" in kernels_by_dim else primary_dim
        coeff_total_map   = compute_obj_coeff_map_for_dim(variants, kernels_by_dim[total_name], ny, BENEFIT_DISCOUNT_RATE)

        # BUFFER runs (± levels), including ±0 (exact envelope)
        if RUN_BENEFIT_PLUSMINUS:
            w = _late_weights(ny)
            for stag, sur in SURPLUS_OPTIONS_M.items():
                print(f"\n  -- Benefit‑optimised (BUFFER around {stag} = {sur} M p.a.) --")
                print(f"     · ± levels tested: {PLUSMINUS_LEVELS_M} (M). "
                      f"Note: ±0 uses exact yearly envelope (emulates 'fixed').")

                for plus in PLUSMINUS_LEVELS_M:
                    if float(plus) == 0.0:
                        dyn_cfg = {"MIN_M": float(sur), "MAX_M": float(sur),
                                   "CONSERVE_TOTAL": True, "TOTAL_REL": "eq",
                                   "ALLOW_BORROWING": False}
                        suffix = f"yoy±{int(plus)}_eq"
                        pretty_line = "     · Using ±0 ⇒ exact yearly envelope (and total equality)."
                    else:
                        dyn_cfg = {"MIN_M": 0.0, "MAX_M": float(sur) + float(plus),
                                   "CONSERVE_TOTAL": True, "TOTAL_REL": "le",
                                   "ALLOW_BORROWING": False}
                        suffix = f"yoy±{int(plus)}_min0_le"
                        pretty_line = f"     · Zero‑floor semantics: min=0, max={sur}+{int(plus)}, total ≤ baseline×years."
                    print(pretty_line)

                    _notify(
                        "solve_start",
                        cost_type=ct,
                        scenario_key=scenario_key,
                        scenario_sheet=scenario_sheet,
                        primary_dim=primary_dim,
                        surplus_key=str(stag),
                        surplus_value=float(sur),
                        plus_level=float(plus),
                        run_mode="buffered",
                    )

                    # Envelope bounds for pruning/greedy
                    y_lb, y_ub, conserve, total_cap = envelope_bounds_from_dyn_cfg(ny, float(sur), dyn_cfg)

                    # Start-time pruning (safe)
                    allowed_starts = prune_impossible_starts(
                        variants, ny, y_ub, conserve, total_cap,
                        coeff_primary_map, coeff_total_map, forced_map, start_fy=CFG["START_FY"]
                    )

                    # Build model using allowed starts only
                    m = build_model(variants, projects,
                                    baseline_surplusM=float(sur), ny=ny,
                                    forced=forced_map, dyn_cfg=dyn_cfg, fixed_envelope=False,
                                    secondary_env_weights=w, allowed_starts=allowed_starts)

                    # Objective vectors
                    coeff_primary = build_obj_coeff(m, kernels_by_dim[primary_dim], ny, BENEFIT_DISCOUNT_RATE)
                    coeff_total   = build_obj_coeff(m, kernels_by_dim[total_name],   ny, BENEFIT_DISCOUNT_RATE)

                    # Greedy warm-start → primary objective lower bound cut
                    sel_greedy, lb_primary = greedy_warm_start(
                        variants, allowed_starts, ny, y_lb, y_ub, conserve,
                        CFG["MAX_STARTS"], coeff_primary_map
                    )
                    if lb_primary > 0.0:
                        idx_nz = np.nonzero(coeff_primary)[0].astype(np.int32)
                        if idx_nz.size > 0:
                            val_nz = coeff_primary[idx_nz].astype(np.double)
                            m.h.addRow(float(lb_primary * (1.0 - 1e-9)), hs.kHighsInf, int(len(idx_nz)), idx_nz, val_nz)
                            print(f"     · Greedy lower bound on primary PV: {lb_primary:,.2f}")

                    # Solve (lexicographic)
                    _apply_early_stop(m)
                    m.Params.TimeLimit = SOLVE_SECONDS
                    m.Params.OutputFlag = (VERBOSE >= 2)

                    objectives = [
                        {"name": f"PV[{primary_dim}]", "coeff": coeff_primary, "sense": "max"},
                        {"name": "PV[Total]",          "coeff": coeff_total,   "sense": "max"},
                    ]
                    if m._y_coeff is not None:
                        objectives.append({"name": "LateEnvelope", "coeff": m._y_coeff, "sense": "min"})

                    solve_lexico(m, objectives)

                    dump_pickle(m, f"{tag_base}_s{int(sur)}_BENBUF",
                                suffix, float(sur),
                                projects, variants, kernels_by_dim, BENEFIT_DISCOUNT_RATE,
                                scenario_key, primary_dim, years,
                                cost_type=ct,
                                surplus_key=str(stag),
                                surplus_value=float(sur),
                                plus_level=float(plus))
        else:
            _notify(
                "dimension_skipped",
                cost_type=ct,
                scenario_key=scenario_key,
                scenario_sheet=scenario_sheet,
                primary_dim=primary_dim,
                reason="RUN_BENEFIT_PLUSMINUS disabled",
            )

        _notify(
            "dimension_complete",
            cost_type=ct,
            scenario_key=scenario_key,
            scenario_sheet=scenario_sheet,
            primary_dim=primary_dim,
        )

    _notify(
        "family_complete",
        cost_type=ct,
        scenario_key=scenario_key,
        scenario_sheet=scenario_sheet,
        dimensions=[str(d) for d in dims_to_run],
    )

# ---------- main ------------------------------------------------------
def main():
    print("MAX_STARTS per FY:", CFG["MAX_STARTS"], "(unit = million NZ$)")
    print(f"CFG window: START_FY={CFG['START_FY']}  YEARS={CFG['YEARS']}")
    if EARLY_STOP_ON:
        print(f">> Early‑stop active: halting when MIP gap ≤ {EARLY_STOP_PCT:.2f}%")

    for ct in COST_TYPES_RUN:
        print(f"\n=== {ct} ===")
        for sc_key, sc_sheet in BENEFIT_SCENARIOS.items():
            print(f"\n>> Benefit Scenario: {sc_key} (sheet: {sc_sheet})")
            optimise_family_for(ct, sc_key, sc_sheet)

# ---------- run -------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    finally:
        print(f"\nRun‑time: {time.time() - t0:.1f}s  → {CACHE}")

