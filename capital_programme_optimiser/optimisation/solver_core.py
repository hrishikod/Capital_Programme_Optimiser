#!/usr/bin/env python
# coding: utf-8

# Fix to the early env. DIP and ALL above changes.WORKING LATEST
# https://chatgpt.com/c/6907cceb-03e4-8324-928a-6d35836e5be4

# In[ ]:


"""
NZTA envelope-first optimiser - COPT multi-objective (PV -> sum net)
v60.3  (DETERMINISTIC TIME-CAP: from a fixed year onward, ClosingNet <= 10% of FULL; fatal validator)

WHAT THIS VERSION GUARANTEES (Proof of Concept)
  - Let CAP_YEAR = START_FY + TIME_CAP_AFTER_YEARS (default: 2025 + 8 = FY2033).
  - For all years Y >= CAP_YEAR: ClosingNet[Y] <= alpha * FULL (alpha = 10%).
  - The cap is hard, applies every year from CAP_YEAR to the horizon, and is validated post-solve.
  - Envelope governance unchanged: flat (FULL) while ON in early/middle years; taper allowed only on a
    contiguous suffix (<= TAPER_SUFFIX_MAX). Once taper starts, y[t] cannot rebound.

New in v60.3 (relative to the original script):
  - Removed draft Total PKL autosaves entirely.
  - Added DIMENSION_INCLUSIONS to choose which non-Total dimensions to run (case-insensitive).
  - Explicit registry reset at the start of main() to avoid stale cross-buffer floors between runs.
  - Otherwise mechanics remain identical to your original script.
"""

from __future__ import annotations

# --- Pin OpenMP/BLAS threads BEFORE any numeric/copt imports ------------------
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Some COPT builds honour this:
os.environ["COPT_NUM_THREADS"] = "1"

import sys, time, re, math, hashlib, random, pickle, json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional, Iterable, Any, Union
from collections import defaultdict

import numpy as np
import pandas as pd

# --- Cardinal Optimizer (COPT) -----------------------------------------------
try:
    import coptpy as co  # type: ignore[import]
except Exception as e:  # pragma: no cover - fallback for non-COPT environments
    co = None  # type: ignore[assignment]

    class _MissingCoptModule:
        """Raise a helpful error whenever the code tries to touch COPT."""

        def __init__(self, err: Exception) -> None:
            self._err = err

        def __getattr__(self, name: str):
            raise RuntimeError(
                "COPT runtime is not available in this environment. "
                "Install the Cardinal Optimizer (coptpy) and configure COPT_HOME. "
                f"Original import error: {self._err}"
            ) from self._err

        def __call__(self, *args, **kwargs):
            return self.__getattr__("__call__")

    _missing_copt = _MissingCoptModule(e)
    co = _missing_copt  # type: ignore[assignment]
    COPT = _missing_copt  # type: ignore[assignment]
    COPT_AVAILABLE = False
    COPT_IMPORT_ERROR = e
else:
    COPT = co.COPT
    COPT_AVAILABLE = True
    COPT_IMPORT_ERROR = None

from capital_programme_optimiser.config import load_settings

SETTINGS = load_settings()

ROOT = SETTINGS.root
DATA_FILE = SETTINGS.scoring_workbook()
CACHE = SETTINGS.cache_dir()
CACHE.mkdir(parents=True, exist_ok=True)
PKL_PREFIX: str | None = None

CFG = {
    "START_FY": int(SETTINGS.optimisation.start_fy),
    "YEARS": int(SETTINGS.optimisation.years),
}

def _refresh_calendar_from_cfg() -> None:
    global START_FY, YEARS, TFIXED, FINAL_YEAR
    START_FY = int(CFG.get("START_FY", SETTINGS.optimisation.start_fy))
    YEARS = int(CFG.get("YEARS", SETTINGS.optimisation.years))
    TFIXED = YEARS
    FINAL_YEAR = START_FY + TFIXED - 1

_refresh_calendar_from_cfg()

MAX_STARTS_PER_FY = int(SETTINGS.optimisation.max_starts)
BENEFIT_DISCOUNT_RATE = float(SETTINGS.optimisation.benefit_discount_rate)

SPEND_SCALE = 100   # 1 unit = 0.01 M
PV_SCALE = 10000    # integer PV scaling

SOLVER_THREADS_DEFAULT = 1
SOLVER_SEED_DEFAULT = 17
VERBOSE = 2
OPTIMISATION_PROFILE = "thorough"  # fast | balanced | thorough | ultra

# QoL knobs
FAST_STOP_GAP = 0.005
CONVERGENCE_REPEAT_N = 2

EFFORT = {
    "fast":      {"MO": 60, "REL_GAP": 0.020},
    "balanced":  {"MO": 80, "REL_GAP": 0.015},
    "thorough":  {"MO": 120, "REL_GAP": 0.010},
    "ultra":     {"MO": 300, "REL_GAP": 0.008},
}[OPTIMISATION_PROFILE]

POOL_PV_TOL = 0.001

USE_ENVELOPE_WINDOW = True
BASE_ON_YEARS_WINDOW = 3
MAX_ON_YEARS_WINDOW = 10

EARLY_PROBE_TIME_S = 10.0
EARLY_GOOD_GAP = 0.010
PRECISION_TOPUP_TIME = 180.0
PRECISION_TOPUP_GAP = 0.001
SLOW_GAP_1 = 0.05
SLOW_GAP_2 = 0.20

ENABLE_ROOT_CUTS = True
ENABLE_STRONG_BRANCHING = True

USE_PEAK_BACKLOG_OBJECTIVE = False  # optional extra smoothing of peaks

ENFORCE_MONOTONE_PV_ACROSS_BUFFERS = True
MONO_REL_EPS = 1e-4
MONO_ABS_EPS = 1e-3

USE_TUNER = False
TUNER_TIME_LIMIT = 900.0
WRITE_TUNED_PAR = True
TUNED_PAR_FILE = CACHE / "nzta_best.par"

# 
# Envelope & antihoarding governance
# 
ALLOW_TAPERED_ENVELOPE = True
DROP_FRAC = 0.25

TAPER_SUFFIX_MAX = 12                        # last  K ON-years may taper
ENVELOPE_NONINCREASING_AFTER_TAPER = True    # once taper starts, no rebound

USE_IDLE_PENALTY = True
IDLE_WEIGHT = 1.0

USE_WEIGHTED_BACKLOG = True
BACKLOG_TAIL_WEIGHT = 2.0

# Soft "overcap"
USE_OVERCAP_PENALTY = True
OVERCAP_WEIGHT = 5.0
BACKLOG_CAP_YEARS = 1.25
BACKLOG_CAP_RAMP_YEARS = 5

# 
# Timebased cap parameters  (CAPS AGAINST FULL, not y[t])
# 
ALPHA_CAP = 0.10      # 10% of FULL after the time-cap year
TIME_CAP_AFTER_YEARS = 8  # from START_FY + 8 onward (i.e., FY2033), cap applies
LOCK_STRICT_EPS_S = 0      # no strict > needed here;  cap is enough

SPIKE_EARLY_TILT = 1e-3    # very small tilt to prefer earlier spike (harmless tiebreak)

# Fatal postsolve validation (stops program if violated)
VALIDATE_SOLUTION = True
VALIDATION_TOL_M = 1e-3  # 0.001 M  1k tolerance

# 
# COST TYPES, SCENARIOS, BUFFERS
# 
COST_TYPES_RUN: List[str] = list(SETTINGS.optimisation.cost_types)
BENEFIT_SCENARIOS: Dict[str, str] = dict(SETTINGS.data.benefit_scenarios)
SURPLUS_OPTIONS_M: Dict[str, float] = {str(k): float(v) for k, v in SETTINGS.optimisation.surplus_options_m.items()}
PLUSMINUS_LEVELS_M: List[float] = [float(v) for v in SETTINGS.optimisation.plusminus_levels_m]
RUN_BENEFIT_PLUSMINUS: bool = bool(SETTINGS.optimisation.run_benefit_plusminus)

if not PLUSMINUS_LEVELS_M:
    PLUSMINUS_LEVELS_M = [0.0]

# 
# FORCED STARTS & RULES
# 
FORCED_START: Dict[str, Dict[str, Optional[Union[int, bool]]]] = {
    name: {"start": rule.start, "include": rule.include}
    for name, rule in SETTINGS.forced_start.items()
}

PROJECT_SELECTION_MODE = "auto"
WHITELIST_FALLBACK_TO_BLACKLIST_IF_EMPTY = True
WARN_ON_UNMATCHED_RULE_NAMES = True

PRIMARY_OBJECTIVE_DIMS_TO_RUN: List[str] = []
DIMENSION_INCLUSIONS: Dict[str, bool] = {"Total": True}
SOLVE_SECONDS = int(SETTINGS.optimisation.solve_seconds)


ProgressCallback = Callable[[str, Dict[str, object]], None]
PROGRESS_LISTENER: Optional[ProgressCallback] = None


def set_progress_listener(listener: Optional[ProgressCallback]) -> None:
    "Register a callback for progress updates."
    global PROGRESS_LISTENER
    PROGRESS_LISTENER = listener


def _notify(stage: str, **payload: object) -> None:
    if PROGRESS_LISTENER is None:
        return
    try:
        PROGRESS_LISTENER(stage, payload)
    except Exception:
        pass


def _sync_dimension_inclusions() -> None:
    if PRIMARY_OBJECTIVE_DIMS_TO_RUN:
        include = {"Total": True}
        for dim in PRIMARY_OBJECTIVE_DIMS_TO_RUN:
            if dim:
                include[dim] = True
        DIMENSION_INCLUSIONS.clear()
        DIMENSION_INCLUSIONS.update(include)
    elif "Total" not in DIMENSION_INCLUSIONS:
        DIMENSION_INCLUSIONS["Total"] = True

# 
# DIMENSION SHORT CODES + INCLUSIONS (case-insensitive matching)
# 
_DIM_SHORT = {
    "Total": "TOT",
    "Healthy and safe people": "HSP",
    "Inclusive access": "INC",
    "Environmental sustainability": "ENV",
    "Economic Prosperity": "ECO",
    "Urban Development": "URB",
    "Resilience and Security": "RES",
}

# Choose which non-Total dimensions to run; Total always runs.
# Names are case-insensitive and matched against your Benefits sheet.
DIMENSION_INCLUSIONS: Dict[str, bool] = {
    "Total": True,                          # Total always runs; this flag is informational
    "Healthy and safe people": True,
    "Inclusive access": False,
    "Environmental sustainability": True,
    "Economic Prosperity": True,
    "Urban Development": True,
    "Resilience and Security": True,
}

def dim_short(dim: str) -> str:
    if dim in _DIM_SHORT:
        return _DIM_SHORT[dim]
    toks = re.findall(r"[A-Za-z0-9]+", dim or "")
    return ("".join(t[:3] for t in toks)[:8] or "DIM").upper()

# 
# Helpers
# 
def cal_years(ny: int) -> List[int]:
    return [START_FY + i for i in range(ny)]

def clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").replace("\xa0"," ")).strip()

def norm(s: str) -> str:
    return clean(s).lower()

def iround(x: float, scale: float) -> int:
    return int(round(float(x) * float(scale)))

def _log(msg: str):
    if VERBOSE >= 1:
        print(msg)

class ParamChangeLogger:
    def __init__(self):
        self._last_params: Dict[str, Any] = {}
    def log_changes(self, params: Dict[str, Any], header: str = "Param changes") -> None:
        changed = {k: v for k, v in params.items()
                   if k not in self._last_params or self._last_params[k] != v}
        if changed:
            self._last_params.update(params)
            changed_str = ", ".join(f"{k}={repr(v)}" for k, v in sorted(changed.items()))
            print(f"{header}: {changed_str}")

_PARAM_LOGGER = ParamChangeLogger()
def _apply_params_logged(m: co.Model, updates: Dict[str, Any], header: str) -> None:
    _PARAM_LOGGER.log_changes(updates, header=header)
    for k, v in updates.items():
        try:
            m.setParam(k, v)
        except Exception:
            pass

def _has_incumbent(m: co.Model) -> bool:
    try:
        hm = m.getAttr(COPT.Attr.HasMipSol)
        if hm is not None:
            return bool(hm)
    except Exception:
        pass
    for attr in (COPT.Attr.HasSolution, COPT.Attr.ObjVal):
        try:
            v = m.getAttr(attr)
            if v is not None:
                if attr == COPT.Attr.ObjVal:
                    float(v)
                return True
        except Exception:
            continue
    return False

def _best_gap(m: co.Model) -> Optional[float]:
    try:
        return float(m.getAttr(COPT.Attr.BestGap))
    except Exception:
        try:
            obj = float(m.getAttr(COPT.Attr.ObjVal))
            bnd = float(m.getAttr(COPT.Attr.BestBnd))
            denom = max(1.0, abs(obj))
            return abs(bnd - obj) / denom
        except Exception:
            return None

RUN_ID = hashlib.sha1(
    f"{OPTIMISATION_PROFILE}|{SOLVER_SEED_DEFAULT}|{SOLVER_THREADS_DEFAULT}".encode("utf-8")
).hexdigest()[:8]

def _val(val_by_id: Optional[Dict[int, float]], var: "co.Var") -> float:
    try:
        if val_by_id is not None:
            v = val_by_id.get(id(var), None)
            if v is not None:
                return float(v)
        return float(var.X)
    except Exception:
        return 0.0

def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def pkl_save(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size = path.stat().st_size if path.exists() else 0
    objective = payload.get("objective", payload.get("primary_dim", "?"))
    best = payload.get("best", {})
    pv = best.get("pv", payload.get("pv_total", "?"))
    gap = best.get("gap", payload.get("gap", "?"))
    print(f"Saved: {path}  ({size} bytes)  objective={objective}  pv={pv}  gap={gap}")

# 
# IO  Costs & Benefits
# 
def load_costs(cost_type: str):
    df = pd.read_excel(DATA_FILE, sheet_name="Costs", engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    proj_col = [c for c in df.columns if c.lower() == "project"]
    if not proj_col:
        raise RuntimeError("Costs sheet needs 'Project'.")
    proj_col = proj_col[-1]
    if "Cost type" not in df.columns or "Duration" not in df.columns:
        raise RuntimeError("Costs sheet must contain 'Cost type' and 'Duration'.")
    horizon_all = [START_FY + i for i in range(YEARS)]
    year_cols = {int(c): c for c in df.columns if str(c).isdigit()}
    use_cols = [year_cols.get(y, None) for y in horizon_all]
    cut = df[df["Cost type"].astype(str).str.strip() == str(cost_type).strip()].copy()
    costs_input = {}
    for _, r in cut.iterrows():
        p = clean(r[proj_col])
        vals = [(pd.to_numeric(r[c], errors="coerce") if c is not None else 0.0) for c in use_cols]
        costs_input[p] = (pd.Series(vals).fillna(0.0) / 1_000_000.0).tolist()  # M
    projects, variants = {}, {}
    for p, seriesM in costs_input.items():
        s = pd.Series(seriesM)
        nz = s.to_numpy().nonzero()[0]
        if nz.size == 0:
            continue
        seg = s.iloc[nz.min(): nz.max()+1].tolist()
        projects[p] = {"cost": float(sum(seg)), "dur": len(seg), "spend": seg}
        variants[p] = {"base": p, "dur": len(seg), "spend": seg, "first_year_idx": int(nz.min())}
    costs_input_df = pd.DataFrame(costs_input, index=horizon_all).T
    costs_input_df.index.name = "Project"
    return projects, variants, costs_input_df

def load_benefits(sheet: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(DATA_FILE, sheet_name=sheet, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    if "Project" not in df.columns:
        raise RuntimeError("Benefits sheet needs 'Project'.")
    dim_col = None
    for c in df.columns:
        if c.lower().startswith("dimension"):
            dim_col = c; break
    if dim_col is None:
        raise RuntimeError("Benefits sheet needs 'Dimension'.")
    if dim_col != "Dimension":
        df.rename(columns={dim_col: "Dimension"}, inplace=True)
    tcols = []
    for c in df.columns:
        m = re.fullmatch(r"[tT]\s*\+\s*(\d+)", str(c))
        if m:
            tcols.append((int(m.group(1)), c))
    tcols.sort(key=lambda x: x[0])
    if not tcols:
        raise RuntimeError("Benefits sheet has no t+K columns.")
    for _, c in tcols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    ben_kernel_df = df.copy()
    ben_kernel_df["Project"] = ben_kernel_df["Project"].map(clean)
    ben_kernel_df["Dimension"] = ben_kernel_df["Dimension"].map(clean)
    ben_kernel_df.set_index(["Project", "Dimension"], inplace=True)
    ben_kernel_df = ben_kernel_df[[c for _, c in tcols]]
    return df, ben_kernel_df

def map_benefit_kernels(benef_df: pd.DataFrame, variants: Dict[str,dict]):
    tcols = [c for _, c in sorted([
        (int(re.fullmatch(r"[tT]\s*\+\s*(\d+)", c).group(1)), c)
        for c in benef_df.columns
        if re.fullmatch(r"[tT]\s*\+\s*(\d+)", c)
    ])]
    df = benef_df.copy()
    df["Project_clean"] = df["Project"].map(clean)
    df["Dimension_clean"] = df["Dimension"].map(clean)

    flows_by_dim: Dict[str, Dict[str, List[float]]] = {}
    order: List[str] = []
    for _, r in df.iterrows():
        p = r["Project_clean"]; d = r["Dimension_clean"]
        seq = r[tcols].to_numpy(dtype=float).tolist()
        flows_by_dim.setdefault(d, {})[p] = seq
        if d not in order:
            order.append(d)

    if "Total" not in flows_by_dim:
        flows_by_dim["Total"] = {}
    all_dims = [d for d in order if d.lower() != "total"]
    projs = set()
    for d in all_dims:
        projs |= set(flows_by_dim[d].keys())
    for p in projs:
        acc = None
        for d in all_dims:
            v = flows_by_dim[d].get(p)
            if v is None:
                continue
            acc = v if acc is None else [a+b for a,b in zip(acc, v)]
        flows_by_dim["Total"][p] = acc or [0.0]*len(tcols)
    order.append("Total")

    keeps = set(variants.keys())
    for d in list(flows_by_dim.keys()):
        flows_by_dim[d] = {p: seq for p, seq in flows_by_dim[d].items() if p in keeps}

    kernels_by_dim: Dict[str, Dict[str, List[float]]] = {}
    for d, mp_ in flows_by_dim.items():
        kernels_by_dim[d] = {}
        for v, meta in variants.items():
            dur = meta["dur"]; ker = mp_.get(v, [])
            kernels_by_dim[d][v] = [0.0]*dur + [float(x) for x in ker]
    return order, kernels_by_dim

# 
# Rules
# 
def apply_forced_rules(variants: Dict[str, dict], rules: Dict[str, Dict]):
    v_norm2canon = {norm(v): v for v in variants.keys()}
    v_norm_set = set(v_norm2canon.keys())

    include_true_norm, exclude_true_norm = set(), set()
    start_map_all_norm = {}

    for raw_name, spec in (rules or {}).items():
        pname_norm = norm(raw_name)
        inc = spec.get("include", None)
        st = spec.get("start", None)
        if inc is True:
            include_true_norm.add(pname_norm)
        if inc is False:
            exclude_true_norm.add(pname_norm)
        if st is not None:
            start_map_all_norm[pname_norm] = int(st)

    matched_includes_norm = include_true_norm & v_norm_set
    mode_req = (PROJECT_SELECTION_MODE or "auto").strip().lower()
    use_whitelist = (mode_req == "whitelist") or (mode_req == "auto" and len(matched_includes_norm) > 0)

    if use_whitelist:
        keep_norm = matched_includes_norm
        if len(keep_norm) == 0 and WHITELIST_FALLBACK_TO_BLACKLIST_IF_EMPTY:
            keep_norm = v_norm_set - (exclude_true_norm & v_norm_set)
        mode = "BLACKLIST (fallback)"
    else:
        keep_norm = v_norm_set - (exclude_true_norm & v_norm_set)
        mode = "BLACKLIST" if mode_req!="auto" else "BLACKLIST (auto)"

    keep_canon = {v_norm2canon[n] for n in keep_norm}
    kept_variants = {v: variants[v] for v in variants if v in keep_canon}

    forced_exact: Dict[str,int] = {}
    for n, yr in start_map_all_norm.items():
        if n in keep_norm:
            forced_exact[v_norm2canon[n]] = int(yr)

    if VERBOSE>=1:
        print(f" [rules] Mode={mode}; kept={len(kept_variants)}/{len(variants)}; forced={len(forced_exact)}")
    if WARN_ON_UNMATCHED_RULE_NAMES:
        unmatched = sorted(list(
            (include_true_norm|exclude_true_norm|set(start_map_all_norm.keys())) - v_norm_set
        ))
        if unmatched:
            print(f" [rules] note: {len(unmatched)} names not found in Costs: {unmatched[:8]}{' ' if len(unmatched)>8 else ''}")

    return kept_variants, forced_exact

# 
# Allowed starts
# 
def allowed_starts_fine(variants: Dict[str,dict], forced_exact: Dict[str,int], Tfine: int) -> Dict[str, List[int]]:
    ny = Tfine
    allowed: Dict[str, List[int]] = {}
    for v, meta in variants.items():
        dur = meta["dur"]
        if v in forced_exact and forced_exact[v] is not None:
            s = forced_exact[v] - START_FY
            allowed[v] = [s] if (0 <= s <= ny - dur) else []
            continue
        s_ear = 0; s_lat = ny - dur
        allowed[v] = list(range(s_ear, s_lat+1))
    return allowed

# 
# PV coefficients (fine)
# 
def coeff_map_for_dim_fine(variants, kernels_for_dim, allowed, Tfine: int, disc_vec: np.ndarray) -> Dict[Tuple[str,int], float]:
    out: Dict[Tuple[str,int], float] = {}
    for v, meta in variants.items():
        dur = meta["dur"]; ker = kernels_for_dim.get(v, [])
        if not ker:
            continue
        for s in allowed.get(v, []):
            val = 0.0
            for k, f in enumerate(ker):
                if f == 0.0:
                    continue
                t = s + k
                if 0 <= t < Tfine:
                    val += float(f) / float(disc_vec[t])
            if val != 0.0:
                out[(v, s)] = val
    return out

def coeff_int(coeff_map: Dict[Tuple[str,int], float], scale: float = PV_SCALE) -> Dict[Tuple[str,int], int]:
    return {k: int(round(v * scale)) for k, v in coeff_map.items() if v != 0.0}

# 
# Weighted-dimension helpers
# 
TOTAL_PV_GUARD_PCT = 85.0

def pv_start0(ker: List[float], r: float) -> float:
    pv = 0.0; denom = 1.0; g = 1.0 + r
    for f in ker:
        pv += float(f)/denom; denom *= g
    return pv

def weights_for_dimension(primary_dim: str, variants: Dict[str, dict], kernels_by_dim: Dict[str, Dict[str, List[float]]], r: float) -> Dict[str, float]:
    alpha=1.0; beta=0.5; wmin=1.0; wmax=2.0; eps=1e-9
    dims = list(kernels_by_dim.keys())
    if primary_dim not in kernels_by_dim:
        return {v: 1.0 for v in variants}

    pv0_primary = {v: pv_start0(kernels_by_dim[primary_dim].get(v,[]), r) for v in variants}
    arr = np.array(list(pv0_primary.values())); order = np.argsort(arr)
    ranks = np.empty_like(arr, dtype=float); ranks[order] = np.arange(1, len(arr)+1, dtype=float)
    pct = {k: float(max(ranks[i]/float(len(arr)), eps)) for i, k in enumerate(pv0_primary.keys())}

    others = [d for d in dims if d != primary_dim and d.lower() != "total"]
    pv0_other_mean = {}
    for v in variants:
        vals = [pv_start0(kernels_by_dim.get(d,{}).get(v,[]), r) for d in others if kernels_by_dim.get(d,{}).get(v)]
        pv0_other_mean[v] = float(np.mean(vals)) if vals else 0.0

    ratio = {}
    for v in variants:
        denom = max(pv0_other_mean[v], eps)
        ratio[v] = pv0_primary[v] / denom if denom > 0 else (wmax if pv0_primary[v] > 0 else 1.0)

    w = {}
    for v in variants:
        wv = (max(ratio[v], eps))**alpha * (max(pct[v], eps))**beta
        w[v] = float(wv)

    m = float(np.mean(list(w.values()))) if w else 1.0
    if m > eps:
        w = {k: v/m for k,v in w.items()}
    return {k: float(min(max(v, wmin), wmax)) for k, v in w.items()}

def weighted_coeff_maps_for_dimension(dim: str, variants: Dict[str,dict], kernels_by_dim: Dict[str, Dict[str, List[float]]],
                                      allowed_fine: Dict[str, List[int]], Tfine: int, disc_vec: np.ndarray) -> Tuple[Dict[Tuple[str,int], float], Dict[Tuple[str,int], int]]:
    w = weights_for_dimension(dim, variants, kernels_by_dim, BENEFIT_DISCOUNT_RATE)
    base = coeff_map_for_dim_fine(variants, kernels_by_dim.get(dim, {}), allowed_fine, Tfine, disc_vec)
    if not base:
        return {}, {}
    weighted = {(v,s): base.get((v,s), 0.0) * float(w.get(v, 1.0)) for (v,s) in base}
    return weighted, coeff_int(weighted)

# 
# Greedy warm start (seed for MO)
# 
def greedy_warm_start(variants: Dict[str,dict], allowed: Dict[str,List[int]], Tfine: int, full_envelope_M: float, max_starts_per_year: int,
                      forced_exact: Dict[str,int], coeff_total_map: Dict[Tuple[str,int], float], reuse_sel: Optional[Dict[Tuple[str,int], int]] = None) -> Dict[Tuple[str,int], int]:
    ny = Tfine
    capacity_prefix = np.array([full_envelope_M*(t+1) for t in range(ny)], dtype=float)
    spend_cum = np.zeros(ny, dtype=float)
    starts_count = np.zeros(ny, dtype=int)
    sel: Dict[Tuple[str,int], int] = {}

    # reuse if compatible
    if reuse_sel:
        ok = True
        tmp_sel, tmp_count = {}, starts_count.copy()
        tmp_cum = spend_cum.copy()
        for (v,s), on in sorted(reuse_sel.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if not on:
                continue
            if s not in set(allowed.get(v, [])):
                ok=False; break
            if tmp_count[s] >= max_starts_per_year:
                ok=False; break
            d = variants[v]["dur"]; vec = np.array(variants[v]["spend"], dtype=float)
            inc = np.zeros(ny, dtype=float); inc[s:s+d] = vec
            if np.any(tmp_cum + np.cumsum(inc) - capacity_prefix > 1e-9):
                ok=False; break
            tmp_cum += np.cumsum(inc); tmp_count[s]+=1; tmp_sel[(v,s)] = 1
        if ok:
            sel = tmp_sel; starts_count = tmp_count; spend_cum = tmp_cum

    # forced
    forced_order = []
    for v, yr in (forced_exact or {}).items():
        if yr is None or v not in variants:
            continue
        s = int(yr - START_FY); d = variants[v]["dur"]
        if s < 0 or s > ny - d:
            continue
        if s not in set(allowed.get(v, [])):
            continue
        if (v,s) not in sel:
            forced_order.append((s, v))
    forced_order.sort()
    for s, v in forced_order:
        if starts_count[s] >= max_starts_per_year:
            continue
        d = variants[v]["dur"]; vec = np.array(variants[v]["spend"], dtype=float)
        inc = np.zeros(ny, dtype=float); inc[s:s+d] = vec
        if np.any(spend_cum + np.cumsum(inc) - capacity_prefix > 1e-9):
            continue
        spend_cum += np.cumsum(inc); starts_count[s]+=1; sel[(v, s)] = 1

    # remaining by PV/cost
    remain = [v for v in variants.keys() if v not in [vv for (vv,_) in sel]]
    order = []
    for v in remain:
        denom = float(sum(variants[v]["spend"])) or 1e-9
        bestpv = max((coeff_total_map.get((v,s),0.0) for s in allowed.get(v, [])), default=0.0)
        order.append((bestpv/denom, v))
    order.sort(reverse=True)
    for _, v in order:
        d = variants[v]["dur"]; vec = np.array(variants[v]["spend"], dtype=float)
        best_s = None
        for s in allowed.get(v, []):
            if starts_count[s] >= max_starts_per_year:
                continue
            inc = np.zeros(ny, dtype=float); inc[s:s+d] = vec
            if np.any(spend_cum + np.cumsum(inc) - capacity_prefix > 1e-9):
                continue
            best_s = s; break
        if best_s is not None:
            inc = np.zeros(ny, dtype=float); inc[best_s:best_s+d] = vec
            spend_cum += np.cumsum(inc); starts_count[best_s] += 1; sel[(v, best_s)] = 1
    return sel

# 
# COPT model  ON/OFF envelope MILP (multi-objective) + TIME-BASED CAP
# 
class CoptOnOffMO:
    """Flat-then-taper envelope; deterministic time-cap: from CAP_YEAR onward, ClosingNet  10%FULL."""
    def __init__(self, variants, allowed, full_envelope_M: float, Tn: int,
                 spend_by_year: Dict[str, List[float]],
                 max_starts_per_year: int,
                 env: Optional[co.Envr] = None,
                 name: str = "onoff_mo",
                 *,
                 fixed_spike: Optional[int] = None,
                 relax_binaries: bool = False,
                 fix_x_to: Optional[Dict[Tuple[str,int], int]] = None):
        self.Tn = int(Tn)
        self.env = env or co.Envr()
        self.m: co.Model = self.env.createModel(name)

        init_params = {
            "RandSeed": int(SOLVER_SEED_DEFAULT),
            "Threads": int(SOLVER_THREADS_DEFAULT),
            "MipStartMode": 2,
            "Logging": 1,
            "Display": 1,
            "Presolve": 1,
            "Scaling": 1,
        }
        if ENABLE_ROOT_CUTS:
            init_params.update({"RootCutLevel": 1, "TreeCutLevel": 1})
        _apply_params_logged(self.m, init_params, header="Model init params")

        self.fullS = iround(full_envelope_M, SPEND_SCALE)  # FULL (scaled S)
        self.thresholdS = int(round(ALPHA_CAP * self.fullS))
        self.drop_cap = int(round(DROP_FRAC * self.fullS)) if DROP_FRAC > 0 else None
        self.BIG_ON = float(self.fullS * self.Tn)  # safe Big-M on net

        # Decision: starts
        self.x: Dict[Tuple[str,int], co.Var] = {}
        self.by_year: Dict[int, List[Tuple[str,int]]] = defaultdict(list)
        x_vtype = COPT.CONTINUOUS if relax_binaries else COPT.BINARY
        for v in variants.keys():
            for s in allowed.get(v, []):
                var = self.m.addVar(lb=0.0, ub=1.0, vtype=x_vtype, name=f"x[{v}|{s}]")
                self.x[(v,s)] = var
                if 0 <= s < self.Tn:
                    self.by_year[s].append((v,s))

        # Envelope ON/OFF and levels
        self.g = [self.m.addVar(vtype=(COPT.CONTINUOUS if relax_binaries else COPT.BINARY), name=f"g[{t}]") for t in range(self.Tn)]
        self.y = [self.m.addVar(lb=0, ub=self.fullS, vtype=COPT.INTEGER, name=f"y[{t}]") for t in range(self.Tn)]
        for t in range(self.Tn-1):
            self.m.addConstr(self.g[t] >= self.g[t+1], name=f"g_monotone[{t}]")

        if ALLOW_TAPERED_ENVELOPE:
            # y  FULLg
            for t in range(self.Tn):
                self.m.addConstr(self.y[t] <= self.fullS * self.g[t], name=f"y_on_cap[{t}]")

            # --- taper suffix: flat early, taper only at the tail (suffix) ---
            self.tail = [self.m.addVar(vtype=(COPT.CONTINUOUS if relax_binaries else COPT.BINARY), name=f"tail[{t}]")
                         for t in range(self.Tn)]
            for t in range(self.Tn):
                self.m.addConstr(self.tail[t] <= self.g[t], name=f"tail_only_when_on[{t}]")
            for t in range(self.Tn-1):
                # suffix monotonicity guarded by ON boundary (prevents infeasibility at g=10)
                self.m.addConstr(self.tail[t] <= self.tail[t+1] + (1.0 - self.g[t+1]), name=f"tail_suffix_mono[{t}]")
                # ensure last ON-year is in tail
                self.m.addConstr(self.tail[t] >= self.g[t] - self.g[t+1], name=f"tail_mark_last_on[{t}]")

            # Tail length cap
            guard_min = 0
            if self.drop_cap is not None and DROP_FRAC > 0:
                guard_min = int(math.ceil(1.0 / float(max(DROP_FRAC, 1e-9))))
            tail_capK = max(int(TAPER_SUFFIX_MAX), guard_min)
            self.m.addConstr(co.quicksum(self.tail) <= float(min(self.Tn, tail_capK)), name="tail_len_cap")

            # FULL while not in tail and ON
            for t in range(self.Tn):
                self.m.addConstr(self.y[t] >= self.fullS * self.g[t] - self.fullS * self.tail[t],
                                 name=f"y_full_when_not_tail[{t}]")

            # Drop cap only inside tail; allow OFF jump
            if self.drop_cap is not None:
                for t in range(1, self.Tn):
                    rhs = float(self.drop_cap) * self.tail[t] + float(self.fullS) * (1.0 - self.g[t])
                    self.m.addConstr(self.y[t-1] - self.y[t] <= rhs, name=f"y_dropcap_tail[{t}]")

            # Non-increasing once taper starts (no rebound)
            if ENVELOPE_NONINCREASING_AFTER_TAPER:
                for t in range(1, self.Tn):
                    self.m.addConstr(self.y[t] <= self.y[t-1] + float(self.fullS) * (1.0 - self.tail[t]) + float(self.fullS) * (1.0 - self.g[t]),
                                     name=f"y_no_rebound[{t}]")
        else:
            for t in range(self.Tn):
                self.m.addConstr(self.y[t] == self.fullS * self.g[t], name=f"y_full_on[{t}]")

        # Spend convolution (scaled)
        spend_terms: List[List[Tuple[co.Var, int]]] = [[] for _ in range(self.Tn)]
        for (v, s), var in self.x.items():
            vec = spend_by_year[v]
            for k, amtM in enumerate(vec):
                if amtM == 0.0:
                    continue
                t = s + k
                if 0 <= t < self.Tn:
                    spend_terms[t].append((var, iround(amtM, SPEND_SCALE)))
        self.spend_expr = []
        for t in range(self.Tn):
            self.spend_expr.append(
                co.quicksum(coeff * var for (var, coeff) in spend_terms[t]) if spend_terms[t] else co.LinExpr(0.0)
            )

        # Net path
        self.net = [self.m.addVar(lb=0.0, vtype=COPT.CONTINUOUS, name=f"net[{t}]") for t in range(self.Tn)]
        self.m.addConstr(self.net[0] == self.y[0] - self.spend_expr[0], name="net0")
        for t in range(1, self.Tn):
            self.m.addConstr(self.net[t] == self.net[t-1] + self.y[t] - self.spend_expr[t], name=f"net[{t}]")

        # Guarantee net is zero whenever envelope is OFF
        for t in range(self.Tn):
            self.m.addConstr(self.net[t] <= self.BIG_ON * self.g[t], name=f"net_off_zero[{t}]")

        # Hard closure at the horizon (no lingering cash)
        self.m.addConstr(self.net[-1] == 0.0, name="final_zero")

        # Starts: exactly once / project; per-year caps
        if not relax_binaries:
            for v in {vv for vv,_ in self.x}:
                cols = [self.x[(vv,s)] for (vv,s) in self.x if vv==v]
                if cols:
                    self.m.addConstr(co.quicksum(cols) == 1, name=f"start_once[{v}]")
            for t in range(self.Tn):
                cols = [self.x[(v,s)] for (v,s) in self.by_year.get(t,[])]
                if cols:
                    self.m.addConstr(co.quicksum(cols) <= int(max_starts_per_year), name=f"cap_starts[{t}]")
        else:
            for v in {vv for vv,_ in self.x}:
                cols = [self.x[(vv,s)] for (vv,s) in self.x if vv==v]
                if cols:
                    self.m.addConstr(co.quicksum(cols) == 1.0, name=f"start_once_relax[{v}]")
            for t in range(self.Tn):
                cols = [self.x[(v,s)] for (v,s) in self.by_year.get(t,[])]
                if cols:
                    self.m.addConstr(co.quicksum(cols) <= float(max_starts_per_year), name=f"cap_starts_relax[{t}]")

        #  net (weighted), final net, peak backlog
        if USE_WEIGHTED_BACKLOG and self.Tn > 1:
            w = np.linspace(1.0, float(BACKLOG_TAIL_WEIGHT), self.Tn, dtype=float).tolist()
        else:
            w = [1.0]*self.Tn
        self.net_weighted_sum = co.quicksum(float(w[t]) * self.net[t] for t in range(self.Tn))
        self.net_last = self.net[self.Tn-1]
        self.u_max = self.m.addVar(lb=0.0, vtype=COPT.CONTINUOUS, name="u_max")
        for t in range(self.Tn):
            self.m.addConstr(self.net[t] <= self.u_max, name=f"u_max_ge_net[{t}]")

        # Soft over-cap penalty
        capS_global = float(BACKLOG_CAP_YEARS) * float(self.fullS)
        rampN = max(1, int(BACKLOG_CAP_RAMP_YEARS))
        self.backlog_cap_vals: List[float] = [float(capS_global * min(1.0, (t+1)/rampN)) for t in range(self.Tn)]
        if USE_OVERCAP_PENALTY:
            self.overcap = [self.m.addVar(lb=0.0, vtype=COPT.CONTINUOUS, name=f"overcap[{t}]") for t in range(self.Tn)]
            for t in range(self.Tn):
                self.m.addConstr(self.overcap[t] >= self.net[t] - self.backlog_cap_vals[t], name=f"overcap_def[{t}]")
            self.overcap_sum = co.quicksum(self.overcap)
        else:
            self.overcap = []
            self.overcap_sum = co.LinExpr(0.0)

        # Local idle funding penalty
        if USE_IDLE_PENALTY:
            self.idle = [self.m.addVar(lb=0.0, vtype=COPT.CONTINUOUS, name=f"idle[{t}]") for t in range(self.Tn)]
            for t in range(self.Tn):
                self.m.addConstr(self.idle[t] >= self.y[t] - self.spend_expr[t], name=f"idle_def[{t}]")
            self.idle_sum = co.quicksum(self.idle)
        else:
            self.idle = []
            self.idle_sum = co.LinExpr(0.0)

        #  Apex diagnostics (optional; harmless to keep) 
        p_vtype = COPT.CONTINUOUS if relax_binaries else COPT.BINARY
        self.p = [self.m.addVar(vtype=p_vtype, name=f"p_spike[{t}]") for t in range(self.Tn)]
        self.m.addConstr(co.quicksum(self.p) == 1.0, name="one_spike")
        for t in range(self.Tn):
            self.m.addConstr(self.p[t] <= self.g[t], name=f"spike_when_on[{t}]")

        self.pre = [self.m.addVar(vtype=(COPT.CONTINUOUS if relax_binaries else COPT.BINARY), name=f"pre[{t}]")
                    for t in range(self.Tn)]
        self.m.addConstr(self.pre[self.Tn-1] == self.p[self.Tn-1], name="pre_last")
        for t in range(self.Tn-2, -1, -1):
            self.m.addConstr(self.pre[t] == self.pre[t+1] + self.p[t], name=f"pre_rec[{t}]")

        self.Peak = self.m.addVar(lb=0.0, vtype=COPT.CONTINUOUS, name="Peak")
        Mpk = self.BIG_ON
        for k in range(self.Tn):
            self.m.addConstr(self.Peak >= self.net[k] - Mpk * (1.0 - self.p[k]), name=f"Peak_ge_net[{k}]")
            self.m.addConstr(self.Peak <= self.net[k] + Mpk * (1.0 - self.p[k]), name=f"Peak_le_net[{k}]")
        for t in range(self.Tn):
            self.m.addConstr(self.net[t] <= self.Peak, name=f"net_le_peak[{t}]")

        #  Deterministic TIME CAP: from CAP_YEAR onward 
        cap_idx = int(min(max(TIME_CAP_AFTER_YEARS, 0), self.Tn-1))
        for t in range(cap_idx, self.Tn):
            self.m.addConstr(self.net[t] <= float(self.thresholdS), name=f"time_cap_full[{t}]")

        # Optional: fixed spike year
        if fixed_spike is not None:
            for t in range(self.Tn):
                self.m.addConstr(self.p[t] == (1.0 if t == int(fixed_spike) else 0.0), name=f"fix_spike[{t}]")

        # Optional: fix some x to 0/1 (for LNS/RINS neighborhoods)
        if fix_x_to:
            for (v,s), val in fix_x_to.items():
                if (v,s) in self.x:
                    self.m.addConstr(self.x[(v,s)] == float(val), name=f"fix_x[{v}|{s}]")

        self._obj_cache: Dict[str, co.LinExpr] = {}
        self._lb_constr: Optional[co.Constr] = None

    # Optional ON-years clamp
    def clamp_on_years_window(self, L_hint: int, w: int):
        lo = max(1, L_hint - w); hi = min(self.Tn, L_hint + w)
        self.m.addConstr(co.quicksum(self.g) >= float(lo), name="gon_lo")
        self.m.addConstr(co.quicksum(self.g) <= float(hi), name="gon_hi")

    def obj_expr(self, coeff_int_map: Dict[Tuple[str,int], int], cache_key: Optional[str]=None) -> co.LinExpr:
        if cache_key and cache_key in self._obj_cache:
            return self._obj_cache[cache_key]
        terms = []
        for (v,s), w in coeff_int_map.items():
            var = self.x.get((v,s))
            if var is not None and int(w)!=0:
                terms.append(w * var)
        expr = co.quicksum(terms) if terms else co.LinExpr(0.0)
        if cache_key:
            self._obj_cache[cache_key] = expr
        return expr

    def add_floor(self, expr: co.LinExpr, target_unscaled: float, name: str):
        self.m.addConstr(expr >= int(math.floor(target_unscaled * PV_SCALE)), name=name)

    def add_hint_from_starts(self, sel: Dict[Tuple[str,int], int]):
        if not sel: return
        try:
            vars_, vals_ = [], []
            for (k, on) in sel.items():
                if on and k in self.x:
                    vars_.append(self.x[k]); vals_.append(1.0)
            if vars_:
                self.m.addMIPStart(vars_, vals_)
        except Exception:
            pass

    def add_hint_on_years(self, L_hint: int):
        try:
            L = int(max(0, min(self.Tn, L_hint)))
            vars_, vals_ = [], []
            for t in range(self.Tn):
                gv = 1.0 if t < L else 0.0
                yv = float(self.fullS) if gv > 0.5 else 0.0
                vars_.append(self.g[t]); vals_.append(gv)
                vars_.append(self.y[t]); vals_.append(yv)
            if vars_:
                self.m.addMIPStart(vars_, vals_)
        except Exception:
            pass

    def add_hint_spike(self, t_star: int):
        try:
            t_star = max(0, min(self.Tn-1, int(t_star)))
            vars_, vals_ = [], []
            for t in range(self.Tn):
                vars_.append(self.p[t]); vals_.append(1.0 if t == t_star else 0.0)
            if vars_:
                self.m.addMIPStart(vars_, vals_)
        except Exception:
            pass

    def add_local_branching_constraint(self, inc_sel: Dict[Tuple[str,int], int], k: int):
        terms = []
        for key, var in self.x.items():
            if inc_sel.get(key, 0) == 1:
                terms.append(1.0 - var)
            else:
                terms.append(var)
        self._lb_constr = self.m.addConstr(co.quicksum(terms) <= float(k), name=f"local_branch_k{int(k)}")

    def add_cumulative_cuts(self, years: Iterable[int], variants: Dict[str,dict]):
        rhs_prefix = [co.quicksum(self.y[k] for k in range(t+1)) for t in range(self.Tn)]
        for t in years:
            if not (0 <= t < self.Tn):
                continue
            lhs_terms = []
            for (v,s), var in self.x.items():
                dur = variants[v]["dur"]; spend = variants[v]["spend"]
                upto = max(0, min(dur, t - s + 1))
                if upto <= 0:
                    continue
                amtM = sum(spend[:upto])
                if amtM != 0.0:
                    lhs_terms.append(iround(amtM, SPEND_SCALE) * var)
            if lhs_terms:
                self.m.addConstr(co.quicksum(lhs_terms) <= rhs_prefix[t], name=f"cumul_cut_t{t}]")

# 
# Adaptive solve helper (legacy, used in dimension solves)
# 
@dataclass
class SolveResult:
    status: int
    has_inc: bool
    gap: Optional[float]
    seconds: float

def solve_mo_adaptive(m: co.Model, stage: str, base_time: float, rel_gap: float, log_name: str, early_probe_time: float = EARLY_PROBE_TIME_S) -> SolveResult:
    _apply_params_logged(m, {"TimeLimit": float(early_probe_time)}, header=f"{stage} probe settings")
    t0 = time.time()
    m.solve()
    has_inc = _has_incumbent(m); gap = _best_gap(m)

    if has_inc and gap is not None and gap <= EARLY_GOOD_GAP:
        _apply_params_logged(m, {"TimeLimit": early_probe_time + PRECISION_TOPUP_TIME,
                                 "RelGap": float(PRECISION_TOPUP_GAP)}, header=f"{stage} top-up settings")
        m.solve()
        return SolveResult(status=int(m.status), has_inc=_has_incumbent(m), gap=_best_gap(m), seconds=time.time()-t0)

    if not has_inc or gap is None or gap > SLOW_GAP_2:
        extras = {"StrongBranching": 1 if ENABLE_STRONG_BRANCHING else 0, "HeurLevel": 1, "RoundingHeurLevel": 1}
        _apply_params_logged(m, extras, header=f"{stage} slow extras")
        extra = max(base_time, 900.0)
    elif gap > SLOW_GAP_1:
        extra = max(0.5*base_time, 360.0)
    else:
        extra = max(0.3*base_time, 240.0)

    total_t = early_probe_time + min(max(base_time, extra), max(base_time, 2400.0))
    _apply_params_logged(m, {"RelGap": float(rel_gap), "TimeLimit": float(total_t)}, header=f"{stage} main settings")
    m.solve()
    return SolveResult(status=int(m.status), has_inc=_has_incumbent(m), gap=_best_gap(m), seconds=time.time()-t0)

# 
# Utilities for selections, PV; solution pool cherry-pick (COPT 8 API safe)
# 
def selection_from_vars(xmap: Dict[Tuple[str,int], co.Var]) -> Dict[Tuple[str,int], int]:
    out = {}
    for (k, var) in xmap.items():
        try:
            val = float(var.X)
        except Exception:
            val = 0.0
        if val > 0.5:
            out[k] = 1
    return out

def selection_from_values(xmap: Dict[Tuple[str,int], co.Var],
                          val_by_id: Optional[Dict[int, float]]) -> Dict[Tuple[str,int], int]:
    out = {}
    for key, var in xmap.items():
        if _val(val_by_id, var) > 0.5:
            out[key] = 1
    return out

def pv_from_selection(coeff_map: Dict[Tuple[str,int], float], sel: Dict[Tuple[str,int], int]) -> float:
    return float(sum(coeff_map.get(k, 0.0) for k, on in sel.items() if on))

def extract_spike_year_from_values(M: CoptOnOffMO,
                                   val_by_id: Optional[Dict[int, float]] = None) -> Optional[int]:
    try:
        vals = [(t, _val(val_by_id, M.net[t])) for t in range(M.Tn)]
        if not vals:
            return None
        return int(max(vals, key=lambda kv: kv[1])[0])
    except Exception:
        return None

def short_solve(M: CoptOnOffMO, time_s: float, rel_gap: float, bundle: str
                ) -> Tuple[bool, Optional[float], Optional[Dict[int, float]]]:
    apply_param_bundle(M.m, bundle)
    _apply_params_logged(M.m, {"RelGap": float(rel_gap), "TimeLimit": float(time_s), "PoolSize": 8}, header=f"short_solve[{bundle}] settings")
    M.m.solve()
    if not _has_incumbent(M.m):
        return False, _best_gap(M.m), None

    val_by_id: Dict[int, float] = {}
    for _, var in M.x.items():     val_by_id[id(var)] = _val(None, var)
    for var in M.net:              val_by_id[id(var)] = _val(None, var)
    for var in M.y:                val_by_id[id(var)] = _val(None, var)
    for var in M.g:                val_by_id[id(var)] = _val(None, var)
    for var in M.p:                val_by_id[id(var)] = _val(None, var)
    for var in M.pre:              val_by_id[id(var)] = _val(None, var)
    if hasattr(M, "tail"):
        for var in M.tail:         val_by_id[id(var)] = _val(None, var)
    val_by_id[id(M.Peak)] = _val(None, M.Peak)

    return True, _best_gap(M.m), val_by_id

def cherry_pick_from_pool(m: co.Model, M: CoptOnOffMO,
                          coeff_total_map_f: Dict[Tuple[str,int], float],
                          best_pv: float, pv_tol_rel: float = POOL_PV_TOL
                          ) -> Optional[Dict[int, float]]:
    try:
        n_pool = int(m.getAttr(COPT.Attr.PoolSols))
    except Exception:
        return None
    if not n_pool or n_pool <= 1:
        return None

    x_items = list(M.x.items())
    x_vars  = [var for _, var in x_items]
    all_vars = []
    all_vars.extend(x_vars)
    all_vars.extend(M.net)
    all_vars.extend(M.y)
    all_vars.extend(M.g)
    all_vars.extend(M.p)
    all_vars.extend(M.pre)
    if hasattr(M, "tail"):
        all_vars.extend(M.tail)
    all_vars.append(M.Peak)

    pos_by_id = {id(v): i for i, v in enumerate(all_vars)}
    net_pos   = [pos_by_id[id(v)] for v in M.net]
    x_kpos    = [ (key, pos_by_id[id(var)]) for (key, var) in x_items ]

    best_vals = None
    best_sigma = None

    for i in range(n_pool):
        vals = None
        try:
            vals = m.getPoolSolution(i, all_vars)
        except Exception:
            try:
                vals = m.getPoolSolution(i)
            except Exception:
                return None
        if vals is None:
            continue

        sel = {}
        for (key, idx) in x_kpos:
            if idx < len(vals) and float(vals[idx]) > 0.5:
                sel[key] = 1

        pv = pv_from_selection(coeff_total_map_f, sel)
        if pv + 1e-12 < (1.0 - pv_tol_rel) * best_pv:
            continue

        sigma = sum(float(vals[j]) for j in net_pos) / SPEND_SCALE
        if best_sigma is None or sigma < best_sigma - 1e-9:
            best_sigma = sigma
            best_vals = vals

    if best_vals is None:
        return None

    return {id(v): float(best_vals[i]) for i, v in enumerate(all_vars) if i < len(best_vals)}

# 
# PKL writer + diagnostics
# 
def dump_pickle_full(M: CoptOnOffMO, tag: str, *,
                     projects, variants, costs_input_df: pd.DataFrame, ben_kernel_df: pd.DataFrame,
                     kernels_by_dim: Dict[str, Dict[str, List[float]]],
                     benefit_rate: float, scenario_name: str, primary_dim: str,
                     Tfine: int, full_envelope_M: float,
                     baseline_envelope_M: Optional[float] = None,
                     sel_override: Optional[Dict[Tuple[str,int], int]] = None,
                     val_override: Optional[Dict[int, float]] = None,
                     extra_diag: Dict[str, str]|None=None, status_override: Optional[str]=None):
    ny = Tfine; fy = cal_years(ny)

    if sel_override is not None:
        sel = sel_override
    elif val_override is not None:
        sel = selection_from_values(M.x, val_override)
    else:
        sel = selection_from_vars(M.x)

    status_text = status_override or "OK"
    if not sel:
        fn = CACHE / f"{(PKL_PREFIX or '')}{tag}_noSol.pkl"
        payload = {"status": "NoSolve", "reason": "no selected starts",
                   "objective": primary_dim, "created_at": _now_stamp()}
        if extra_diag:
            payload["diagnostic"] = extra_diag
        pkl_save(fn, payload)
        return

    # schedule
    rows = []
    for (v,s) in sorted(sel.keys()):
        rows.append({"Project": v,
                     "StartFY": START_FY + s,
                     "EndFY": START_FY + s + variants[v]["dur"] - 1,
                     "Dur": variants[v]["dur"],
                     "Scenario": scenario_name,
                     "PrimaryDim": primary_dim})
    df_sched = pd.DataFrame(rows).sort_values(["StartFY","Project"], ignore_index=True)

    # spend & envelope (PV window)
    df_sp = pd.DataFrame(0.0, index=list(projects.keys()), columns=fy)
    for (v,s) in sel.keys():
        vec = variants[v]["spend"]
        for i, amt in enumerate(vec):
            t = s + i
            if 0 <= t < ny:
                df_sp.loc[v, fy[t]] += float(amt)
    df_sp.loc["Total Spend"] = df_sp.sum()

    env_M = []
    for t in range(ny):
        yv = _val(val_override, M.y[t])
        env_M.append(yv / SPEND_SCALE)

    cash_rows, net_prev = [], 0.0
    for t in range(ny):
        yr = fy[t]
        spend = float(df_sp.loc["Total Spend", yr])
        env = env_M[t]
        net = net_prev + env - spend
        cash_rows.append(dict(
            Year=yr, Envelope=env, OpeningNet=net_prev, Spend=spend, ClosingNet=net,
            OpeningCash=max(net_prev,0.0), OpeningDebt=max(-net_prev,0.0),
            ClosingCash=max(net,0.0), ClosingDebt=max(-net,0.0)
        ))
        net_prev = net
    if len(cash_rows) > 0:
        cash_rows[-1]["ClosingNet"] = 0.0
    df_cash = pd.DataFrame(cash_rows)

    # benefits
    dims = list(kernels_by_dim.keys())
    Tstore = ny
    for (v, s) in sel.keys():
        for d in dims:
            ker = kernels_by_dim.get(d, {}).get(v, [])
            if ker:
                Tstore = max(Tstore, s + len(ker))
    fy_store = cal_years(Tstore)
    ben_total_by_dim_store = {d: np.zeros(Tstore, float) for d in dims}
    proj_dim_year_store = {(p,d): np.zeros(Tstore, float) for d in dims for p in projects.keys()}
    for (v,s) in sel.keys():
        for d in dims:
            ker = kernels_by_dim[d].get(v, [])
            for k, f in enumerate(ker):
                t = s + k
                if 0 <= t < Tstore:
                    ben_total_by_dim_store[d][t] += float(f)
                    proj_dim_year_store[(v,d)][t] += float(f)

    df_ben_year = pd.DataFrame({"Year": fy_store, **{d: ben_total_by_dim_store[d] for d in dims}})
    idx = pd.MultiIndex.from_product([list(projects.keys()), dims], names=["Project","Dimension"])
    df_ben_proj_dim_year = pd.DataFrame(0.0, index=idx, columns=fy_store)
    for (p,d), vec in proj_dim_year_store.items():
        df_ben_proj_dim_year.loc[(p,d), :] = vec

    disc = np.array([(1.0 + benefit_rate) ** t for t in range(ny)], float)
    pv_by_dim = {d: float(np.sum(ben_total_by_dim_store[d][:ny] / disc)) for d in dims}
    pv_by_proj_dim = pd.DataFrame(0.0, index=idx, columns=["PV"])
    for (p,d) in idx:
        vec_full = df_ben_proj_dim_year.loc[(p,d)].to_numpy(dtype=float)
        pv_by_proj_dim.loc[(p,d), "PV"] = float(np.sum(vec_full[:ny] / disc))
    total_pv = pv_by_dim.get("Total", float(np.sum([pv_by_dim[d] for d in dims if d!="Total"])))

    # diagnostics  include cap year
    try:
        net_list = [r["ClosingNet"] for r in cash_rows]
        TH = float(ALPHA_CAP * full_envelope_M)
        cap_idx = int(min(max(TIME_CAP_AFTER_YEARS, 0), ny-1))
        viol_env = []
        for k in range(cap_idx, ny):
            if net_list[k] > TH + 1e-6:
                viol_env.append((fy[k], net_list[k] - TH))
        diag_caps = {
            "cap_year": cal_years(ny)[cap_idx],
            "post_cap_full_violations": len(viol_env),
        }
    except Exception:
        diag_caps = {}

    gap_now = _best_gap(M.m)

    out = dict(
        status=status_text,
        objective=primary_dim,
        created_at=_now_stamp(),
        scenario=scenario_name,
        primary_dim=primary_dim,
        schedule=df_sched,
        spend=df_sp,
        cash_flow=df_cash,
        envelope=pd.DataFrame({"Year": fy, "Envelope": env_M}),
        benefits_by_year=df_ben_year,
        benefits_by_project_dimension_by_year=df_ben_proj_dim_year,
        pv_by_dimension=pv_by_dim,
        pv_total=total_pv,
        gap=gap_now,
        best={"pv": total_pv, "gap": gap_now},
        pv_by_project_and_dimension=pv_by_proj_dim,
        calendar=dict(start_fy=START_FY, years=ny),
        meta=dict(
            full_envelope_M=full_envelope_M,
            baseline_envelope_M=baseline_envelope_M if baseline_envelope_M is not None else full_envelope_M,
            on_years=int(round(sum(_val(val_override, M.g[t]) for t in range(ny)))) if hasattr(M, "g") else None,
            pv_window={"base_year": START_FY, "last_pv_year": FINAL_YEAR},
            alpha_cap=ALPHA_CAP,
            cap_year=cal_years(ny)[int(min(max(TIME_CAP_AFTER_YEARS, 0), ny-1))],
            taper_suffix_max=TAPER_SUFFIX_MAX,
            nonincreasing_after_taper=ENVELOPE_NONINCREASING_AFTER_TAPER,
            overcap_weight=OVERCAP_WEIGHT if USE_OVERCAP_PENALTY else 0.0,
            backlog_cap_years=BACKLOG_CAP_YEARS,
            backlog_cap_ramp_years=BACKLOG_CAP_RAMP_YEARS,
            drop_frac=DROP_FRAC,
            idle_weight=IDLE_WEIGHT if USE_IDLE_PENALTY else 0.0,
            backlog_tail_weight=BACKLOG_TAIL_WEIGHT if USE_WEIGHTED_BACKLOG else 1.0,
            run_id=RUN_ID
        ),
    )
    if extra_diag:
        out.setdefault("diagnostic", {}).update(extra_diag)
    if diag_caps:
        out.setdefault("diagnostic", {}).update(diag_caps)

    fn = CACHE / f"{(PKL_PREFIX or '')}{tag}.pkl"
    if str(fn).endswith("}.pkl"):
        fn = Path(str(fn)[:-5] + ".pkl")
    pkl_save(fn, out)

# 
# Validation that aborts run if time-cap violated
# 
def validate_time_cap_or_die(M: CoptOnOffMO, full_envelope_M: float, val_by_id: Dict[int, float], *, label: str):
    if not VALIDATE_SOLUTION:
        return
    netM = [ _val(val_by_id, M.net[t]) / SPEND_SCALE for t in range(M.Tn) ]
    TH = ALPHA_CAP * full_envelope_M
    tol = VALIDATION_TOL_M
    cap_idx = int(min(max(TIME_CAP_AFTER_YEARS, 0), M.Tn-1))
    for k in range(cap_idx, len(netM)):
        if netM[k] > TH + tol:
            raise RuntimeError(f"[VALIDATION:{label}] Time-cap violated at year {START_FY+k} "
                               f"(t={k}): net={netM[k]:.3f} M > {TH:.3f} M (10% of FULL).")

# 
# Crossbuffer monotonic PV registry
# 
_BEST_PV_BY_ENV: Dict[Tuple[str,str,float], float] = {}
_BEST_SEL_BY_ENV: Dict[Tuple[str,str,float], Dict[Tuple[str,int], int]] = {}

def adaptive_window_params(full_envelope_M: float):
    def min_full_env_M() -> float:
        return float(min(SURPLUS_OPTIONS_M.values())) if SURPLUS_OPTIONS_M else 0.0
    steps = max(0, int(round((full_envelope_M - min_full_env_M()) / 500.0)))
    w = min(MAX_ON_YEARS_WINDOW, BASE_ON_YEARS_WINDOW + steps)
    return w

# 
# Orchestrator for TOTAL runs (no draft autosaves)
# 
@dataclass
class Incumbent:
    sel: Dict[Tuple[str,int], int]
    spike_year: Optional[int]
    pv: float
    sigma_net: float
    val_by_id: Dict[int, float]
    model: CoptOnOffMO
    gap: Optional[float]

def apply_param_bundle(m: co.Model, bundle: str, seed: Optional[int] = None):
    updates: Dict[str, Any] = {}
    if seed is not None:
        updates["RandSeed"] = int(seed)
    if bundle == "baseline":
        updates.update({
            "StrongBranching": 1 if ENABLE_STRONG_BRANCHING else 0,
            "HeurLevel": 1,
            "RoundingHeurLevel": 1,
            "RootCutLevel": 1,
            "TreeCutLevel": 1,
        })
    elif bundle == "intensify":
        updates.update({
            "StrongBranching": 1,
            "HeurLevel": 1,
            "RoundingHeurLevel": 1,
            "RootCutLevel": 2,
            "TreeCutLevel": 2,
        })
    elif bundle == "diversify":
        updates.update({
            "StrongBranching": 0,
            "HeurLevel": 2,
            "RoundingHeurLevel": 2,
            "RootCutLevel": 1,
            "TreeCutLevel": 1,
        })
    if updates:
        _apply_params_logged(m, updates, header=f"Param bundle[{bundle}]")

def build_total_model(variants, allowed, full_envelope_M, Tfine,
                      coeff_total_fine_int, total_floor_target: Optional[float], L_hint: int,
                      *, with_window: bool, fixed_spike: Optional[int] = None,
                      relax_binaries: bool = False,
                      fix_x_to: Optional[Dict[Tuple[str,int], int]] = None,
                      cumulative_cut_years: Optional[List[int]] = None) -> CoptOnOffMO:
    M = CoptOnOffMO(variants, allowed, full_envelope_M, Tfine,
                    spend_by_year={v: variants[v]["spend"] for v in variants},
                    max_starts_per_year=MAX_STARTS_PER_FY, name="MO_TOT",
                    fixed_spike=fixed_spike, relax_binaries=relax_binaries, fix_x_to=fix_x_to)
    M.add_hint_on_years(L_hint)
    if with_window and USE_ENVELOPE_WINDOW:
        w = adaptive_window_params(full_envelope_M)
        M.clamp_on_years_window(L_hint, w)
    expr_tot = M.obj_expr(coeff_total_fine_int, cache_key="TOT")
    try:
        M.m.setObjectiveN(expr_tot, 0, COPT.MAXIMIZE)
        M.m.setObjParamN(0, "MultiObjPriority", 3)
        M.m.setObjParamN(0, "MultiObjRelTol", 0.001)
        M.m.setObjParamN(0, "MultiObjAbsTol", 0.0)
    except Exception:
        M.m.setObjective(expr_tot, COPT.MAXIMIZE)

    if total_floor_target is not None:
        floor_target = total_floor_target - max(MONO_ABS_EPS, MONO_REL_EPS*abs(total_floor_target))
        M.add_floor(expr_tot, floor_target, name="mono_total_floor")

    spike_index = co.quicksum(float(t) * M.p[t] for t in range(Tfine))
    obj1 = M.net_weighted_sum + SPIKE_EARLY_TILT * float(M.fullS) * spike_index
    if USE_OVERCAP_PENALTY: obj1 = obj1 + float(OVERCAP_WEIGHT) * M.overcap_sum
    if USE_IDLE_PENALTY:    obj1 = obj1 + float(IDLE_WEIGHT) * M.idle_sum
    try:
        M.m.setObjectiveN(obj1, 1, COPT.MINIMIZE)
        M.m.setObjParamN(1, "MultiObjPriority", 2)
        M.m.setObjParamN(1, "MultiObjWeight", 1.0)
    except Exception:
        pass
    try:
        M.m.setObjectiveN(M.net[-1], 2, COPT.MINIMIZE)
        M.m.setObjParamN(2, "MultiObjPriority", 2)
        M.m.setObjParamN(2, "MultiObjWeight", float(Tfine))
    except Exception:
        pass
    if USE_PEAK_BACKLOG_OBJECTIVE:
        try:
            M.m.setObjectiveN(M.u_max, 3, COPT.MINIMIZE)
            M.m.setObjParamN(3, "MultiObjPriority", 1)
        except Exception:
            pass

    if cumulative_cut_years:
        M.add_cumulative_cuts(cumulative_cut_years, variants)

    return M

def eval_incumbent(M: CoptOnOffMO, val_by_id: Dict[int, float],
                   coeff_total_fine: Dict[Tuple[str,int], float]) -> Incumbent:
    sel = selection_from_values(M.x, val_by_id)
    pv = pv_from_selection(coeff_total_fine, sel)
    sigma = sum(_val(val_by_id, v) for v in M.net) / SPEND_SCALE
    spike = extract_spike_year_from_values(M, val_by_id)
    gap = _best_gap(M.m)
    return Incumbent(sel=sel, spike_year=spike, pv=pv, sigma_net=sigma, val_by_id=val_by_id, model=M, gap=gap)

def orchestrate_total(variants, allowed_fine, full_envelope_M, Tfine,
                      coeff_total_fine: Dict[Tuple[str,int], float],
                      coeff_total_fine_int: Dict[Tuple[str,int], int],
                      total_floor_target: Optional[float], L_hint: int,
                      *, sel_seed: Optional[Dict[Tuple[str,int], int]],
                      rel_gap_target: float, time_budget_s: float,
                      fast_stop_gap: float = FAST_STOP_GAP,
                      convergence_repeat_n: int = CONVERGENCE_REPEAT_N) -> Tuple[Optional[Incumbent], Optional[float]]:
    t_start = time.time()
    best: Optional[Incumbent] = None
    last_imp_t = time.time()
    last_gap: Optional[float] = None

    last_best_pair: Optional[Tuple[float, float]] = None
    repeat_streak = 0

    def remain():
        return max(0.0, time_budget_s - (time.time() - t_start))

    # Phase 0: quick boot
    M = build_total_model(variants, allowed_fine, full_envelope_M, Tfine,
                          coeff_total_fine_int, total_floor_target, L_hint,
                          with_window=True)
    if sel_seed:
        M.add_hint_from_starts(sel_seed)
    if L_hint is not None:
        M.add_hint_spike(max(0, min(L_hint-1, Tfine-2)))
    ok, gap, valmap = short_solve(M, min(30.0, max(5.0, remain()*0.2)), rel_gap_target, "baseline")
    if ok and valmap is not None:
        cand = eval_incumbent(M, valmap, coeff_total_fine)
        best = cand
        last_imp_t = time.time()
        last_gap = cand.gap
        last_best_pair = (best.pv, best.gap if best.gap is not None else float("inf"))
        if best.gap is not None and best.gap <= fast_stop_gap:
            print(f"[FAST-STOP] baseline gap {best.gap:.4%}  {fast_stop_gap:.2%}.")
            return best, best.gap
    if remain() <= 1.0:
        return best, (best.gap if best else None)

    STALL_PCTPT = 0.002
    STALL_WINDOW_S = 30.0

    while remain() > 1.0:
        # 1) baseline refresh
        M = build_total_model(variants, allowed_fine, full_envelope_M, Tfine,
                              coeff_total_fine_int, total_floor_target, L_hint,
                              with_window=True)
        if best:
            M.add_hint_from_starts(best.sel)
            if best.spike_year is not None:
                M.add_hint_spike(best.spike_year)
        ok, gap, valmap = short_solve(M, min(40.0, max(5.0, remain()*0.25)), rel_gap_target, "baseline")
        if ok and valmap is not None:
            cand = eval_incumbent(M, valmap, coeff_total_fine)
            if (best is None) or (cand.pv > best.pv + 1e-9) or (abs(cand.pv - best.pv) <= 1e-9 and cand.sigma_net < best.sigma_net - 1e-9):
                best = cand; last_imp_t = time.time()
                if best.gap is not None and best.gap <= fast_stop_gap:
                    print(f"[FAST-STOP] gap {best.gap:.4%}  {fast_stop_gap:.2%}.")
                    return best, best.gap
        improved = (best is not None) and (last_gap is None or (gap is not None and (last_gap - gap) >= STALL_PCTPT))
        last_gap = gap

        if best is not None:
            curr_pair = (best.pv, best.gap if best.gap is not None else float("inf"))
            if last_best_pair is not None and math.isclose(curr_pair[0], last_best_pair[0], rel_tol=0, abs_tol=1e-12) \
               and ((math.isclose(curr_pair[1], last_best_pair[1], rel_tol=0, abs_tol=1e-12))):
                repeat_streak += 1
                if repeat_streak >= convergence_repeat_n:
                    print(f"[EARLY FINALIZE] Best pair repeated {repeat_streak} times  finalize TOTAL.")
                    return best, best.gap
            else:
                repeat_streak = 0
                last_best_pair = curr_pair

        if remain() <= 1.0:
            break
        if improved or (time.time() - last_imp_t < STALL_WINDOW_S):
            continue

        # 2) Spike sweep around incumbent
        if best and best.spike_year is not None and remain() > 6.0:
            for t_star in [best.spike_year, max(0, best.spike_year-1), min(Tfine-1, best.spike_year+1)]:
                if remain() <= 2.0: break
                Ms = build_total_model(variants, allowed_fine, full_envelope_M, Tfine,
                                       coeff_total_fine_int, total_floor_target, L_hint,
                                       with_window=False, fixed_spike=int(t_star))
                if best: Ms.add_hint_from_starts(best.sel)
                ok, gap, valmap = short_solve(Ms, min(25.0, max(5.0, remain()*0.20)), rel_gap_target, "baseline")
                if ok and valmap is not None:
                    cand = eval_incumbent(Ms, valmap, coeff_total_fine)
                    if (best is None) or (cand.pv > best.pv + 1e-9) or (abs(cand.pv - best.pv) <= 1e-9 and cand.sigma_net < best.sigma_net - 1e-9):
                        best = cand; last_imp_t = time.time()
                        if best.gap is not None and best.gap <= fast_stop_gap:
                            print(f"[FAST-STOP] gap {best.gap:.4%}  {fast_stop_gap:.2%}.")
                            return best, best.gap
            if time.time() - last_imp_t < STALL_WINDOW_S:
                continue

        if remain() <= 1.0:
            break

        # 3) Local Branching
        if best and remain() > 6.0:
            for k in (10, 18):
                if remain() <= 2.0: break
                Ml = build_total_model(variants, allowed_fine, full_envelope_M, Tfine,
                                       coeff_total_fine_int, total_floor_target, L_hint,
                                       with_window=False)
                Ml.add_local_branching_constraint(best.sel, k=k)
                Ml.add_hint_from_starts(best.sel)
                if best.spike_year is not None:
                    Ml.add_hint_spike(best.spike_year)
                ok, gap, valmap = short_solve(Ml, min(40.0, max(5.0, remain()*0.25)), rel_gap_target, "baseline")
                if ok and valmap is not None:
                    cand = eval_incumbent(Ml, valmap, coeff_total_fine)
                    if (best is None) or (cand.pv > best.pv + 1e-9) or (abs(cand.pv - best.pv) <= 1e-9 and cand.sigma_net < best.sigma_net - 1e-9):
                        best = cand; last_imp_t = time.time()
                        if best.gap is not None and best.gap <= fast_stop_gap:
                            print(f"[FAST-STOP] gap {best.gap:.4%}  {fast_stop_gap:.2%}.")
                            return best, best.gap
                if time.time() - last_imp_t < STALL_WINDOW_S:
                    break
            if time.time() - last_imp_t < STALL_WINDOW_S:
                continue

        if remain() <= 1.0:
            break

        # 4) RINS-lite
        if best and remain() > 8.0:
            Mr = build_total_model(variants, allowed_fine, full_envelope_M, Tfine,
                                   coeff_total_fine_int, total_floor_target, L_hint,
                                   with_window=False, relax_binaries=True)
            if best.spike_year is not None:
                Mr.add_hint_spike(best.spike_year)
            okLP, _, vLP = short_solve(Mr, min(12.0, max(5.0, remain()*0.15)), rel_gap_target, "baseline")
            fix_x: Dict[Tuple[str,int], int] = {}
            if okLP and vLP is not None:
                for key, var in Mr.x.items():
                    xi = 1 if best.sel.get(key, 0) == 1 else 0
                    v = _val(vLP, var)
                    if xi == 1 and v >= 0.95:
                        fix_x[key] = 1
                    elif xi == 0 and v <= 0.05:
                        fix_x[key] = 0
            Mr2 = build_total_model(variants, allowed_fine, full_envelope_M, Tfine,
                                    coeff_total_fine_int, total_floor_target, L_hint,
                                    with_window=False, fix_x_to=fix_x)
            Mr2.add_hint_from_starts(best.sel)
            if best.spike_year is not None:
                Mr2.add_hint_spike(best.spike_year)
            ok, gap, valmap = short_solve(Mr2, min(45.0, max(5.0, remain()*0.30)), rel_gap_target, "baseline")
            if ok and valmap is not None:
                cand = eval_incumbent(Mr2, valmap, coeff_total_fine)
                if (best is None) or (cand.pv > best.pv + 1e-9) or (abs(cand.pv - best.pv) <= 1e-9 and cand.sigma_net < best.sigma_net - 1e-9):
                    best = cand; last_imp_t = time.time()
                    if best.gap is not None and best.gap <= fast_stop_gap:
                        print(f"[FAST-STOP] gap {best.gap:.4%}  {fast_stop_gap:.2%}.")
                        return best, best.gap
            if time.time() - last_imp_t < STALL_WINDOW_S:
                continue

        if remain() <= 1.0:
            break

        # 5) Year-window LNS around spike/hump
        if best and best.val_by_id and remain() > 8.0:
            net_vals = [ _val(best.val_by_id, best.model.net[t]) for t in range(Tfine) ]
            win_size = 6
            centers = set()
            if best.spike_year is not None:
                centers.add(int(best.spike_year))
            centers.add(int(np.argmax(net_vals)))
            windows = []
            for c in sorted(list(centers)):
                lo = max(0, c - win_size//2)
                hi = min(Tfine-1, c + win_size//2)
                windows.append((lo, hi))
            impacted_projects: set[str] = set()
            for (v,s), on in best.sel.items():
                if not on: continue
                dur = variants[v]["dur"]
                span = (s, s+dur-1)
                for (lo,hi) in windows:
                    if not (span[1] < lo or span[0] > hi):
                        impacted_projects.add(v); break
            fix_x: Dict[Tuple[str,int], int] = {}
            for v in variants.keys():
                if v in impacted_projects:
                    continue
                s_star = None
                for (vv,ss), on in best.sel.items():
                    if vv==v and on:
                        s_star = ss; break
                for s in allowed_fine.get(v, []):
                    fix_x[(v,s)] = 1 if (s_star is not None and s==s_star) else 0
            Ml = build_total_model(variants, allowed_fine, full_envelope_M, Tfine,
                                   coeff_total_fine_int, total_floor_target, L_hint,
                                   with_window=False, fix_x_to=fix_x)
            Ml.add_hint_from_starts(best.sel)
            if best.spike_year is not None:
                Ml.add_hint_spike(best.spike_year)
            ok, gap, valmap = short_solve(Ml, min(45.0, max(5.0, remain()*0.30)), rel_gap_target, "baseline")
            if ok and valmap is not None:
                cand = eval_incumbent(Ml, valmap, coeff_total_fine)
                if (best is None) or (cand.pv > best.pv + 1e-9) or (abs(cand.pv - best.pv) <= 1e-9 and cand.sigma_net < best.sigma_net - 1e-9):
                    best = cand; last_imp_t = time.time()
                    if best.gap is not None and best.gap <= fast_stop_gap:
                        print(f"[FAST-STOP] gap {best.gap:.4%}  {fast_stop_gap:.2%}.")
                        return best, best.gap
            if time.time() - last_imp_t < STALL_WINDOW_S:
                continue

        if remain() <= 1.0:
            break

        # 6) Intensify with cumulative cuts around the spike
        if best and remain() > 6.0:
            years = []
            if best.spike_year is not None:
                s = best.spike_year
                years = list(range(max(0, s-3), min(Tfine-1, s+8)+1))
            Mi = build_total_model(variants, allowed_fine, full_envelope_M, Tfine,
                                   coeff_total_fine_int, total_floor_target, L_hint,
                                   with_window=False, cumulative_cut_years=years)
            Mi.add_hint_from_starts(best.sel)
            if best.spike_year is not None:
                Mi.add_hint_spike(best.spike_year)
            ok, gap, valmap = short_solve(Mi, min(60.0, max(5.0, remain()*0.35)), rel_gap_target, "intensify")
            if ok and valmap is not None:
                cand = eval_incumbent(Mi, valmap, coeff_total_fine)
                if (best is None) or (cand.pv > best.pv + 1e-9) or (abs(cand.pv - best.pv) <= 1e-9 and cand.sigma_net < best.sigma_net - 1e-9):
                    best = cand; last_imp_t = time.time()
                    if best.gap is not None and best.gap <= fast_stop_gap:
                        print(f"[FAST-STOP] gap {best.gap:.4%}  {fast_stop_gap:.2%}.")
                        return best, best.gap

    return best, (best.gap if best else None)

# 
# Weighted-dimension run (kept; now filtered by DIMENSION_INCLUSIONS)
# 
def run_dimensions_for_env(ct: str, sc_key: str, sc_sheet: str, sur_key: str, plus_M: float, *,
                           projects: Dict[str,dict], variants: Dict[str,dict],
                           costs_input_df: pd.DataFrame, ben_kernel_df: pd.DataFrame,
                           kernels_by_dim: Dict[str, Dict[str, List[float]]],
                           Tfine: int, full_envelope_M: float,
                           baseline_envelope_M: float,
                           allowed_fine: Dict[str, List[int]],
                           disc_vec: np.ndarray,
                           coeff_total_fine_int: Dict[Tuple[str,int], int],
                           total_best_sel: Dict[Tuple[str,int], int],
                           total_best_pv: float,
                           L_hint: int,
                           tot_model_for_fallback: Optional[CoptOnOffMO],
                           tot_valmap_for_fallback: Optional[Dict[int, float]],
                           prev_dim_floors: Optional[Dict[str, float]] = None,
                           dims_filter: Optional[Iterable[str]] = None) -> Dict[str, float]:
    prev_dim_floors = dict(prev_dim_floors or {})
    # Filter the dimensions: exclude "Total", then apply user filter if provided (case-insensitive handled upstream)
    dims_all = [d for d in kernels_by_dim.keys() if d.lower() != "total"]
    if dims_filter is not None:
        # Keep only dims present in filter (exact names already pre-selected upstream)
        dims = [d for d in dims_all if d in set(dims_filter)]
    else:
        dims = dims_all

    if not dims:
        _log(" [DIM] No weighted-dimension runs requested.")
        return prev_dim_floors

    total_floor = float(total_best_pv) * (TOTAL_PV_GUARD_PCT / 100.0)
    for dim in dims:
        tag_dim = f"{ct.replace(' ','').replace('-', '')}_{sc_key}_{sur_key}_pm{int(plus_M)}_{dim_short(dim)}"
        _log(f" [DIM] {dim} (weighted) with Total-guard  {total_floor:,.6f}")

        coeff_dim_w_f, coeff_dim_w_int = weighted_coeff_maps_for_dimension(
            dim, variants, kernels_by_dim, allowed_fine, Tfine, disc_vec
        )
        if not coeff_dim_w_int:
            if tot_model_for_fallback is not None:
                dump_pickle_full(tot_model_for_fallback, tag_dim,
                                 projects=projects, variants=variants,
                                 costs_input_df=costs_input_df, ben_kernel_df=ben_kernel_df,
                                 kernels_by_dim=kernels_by_dim, benefit_rate=BENEFIT_DISCOUNT_RATE,
                                 scenario_name=sc_key, primary_dim=f"{dim} [WEIGHTED]",
                                 Tfine=Tfine, full_envelope_M=full_envelope_M,
                                 baseline_envelope_M=baseline_envelope_M,
                                 sel_override=total_best_sel, val_override=tot_valmap_for_fallback,
                                 status_override="OK_SUBOPT_NOCOEFF",
                                 extra_diag={"note": "dimension has zero coefficients; reported Total solution"})
            prev_dim_floors[dim] = pv_from_selection(coeff_dim_w_f, total_best_sel)
            continue

        def build_and_solve_dim(with_window: bool, log_suffix: str):
            Md = CoptOnOffMO(variants, allowed_fine, full_envelope_M, Tfine,
                             spend_by_year={v: variants[v]["spend"] for v in variants},
                             max_starts_per_year=MAX_STARTS_PER_FY, name=f"MO_DIM_{dim_short(dim)}")
            Md.add_hint_from_starts(total_best_sel)
            Md.add_hint_on_years(L_hint)
            if with_window and USE_ENVELOPE_WINDOW:
                w = adaptive_window_params(full_envelope_M)
                Md.clamp_on_years_window(L_hint, w)

            expr_dim = Md.obj_expr(coeff_dim_w_int, cache_key=f"DIM_{dim_short(dim)}")
            try:
                Md.m.setObjectiveN(expr_dim, 0, COPT.MAXIMIZE)
                Md.m.setObjParamN(0, "MultiObjPriority", 3)
                Md.m.setObjParamN(0, "MultiObjRelTol", 0.001)
                Md.m.setObjParamN(0, "MultiObjAbsTol", 0.0)
            except Exception:
                Md.m.setObjective(expr_dim, COPT.MAXIMIZE)

            expr_tot_guard = Md.obj_expr(coeff_total_fine_int, cache_key="TOT_GUARD")
            Md.add_floor(expr_tot_guard, total_floor - max(MONO_ABS_EPS, MONO_REL_EPS*abs(total_floor)), name="dim_total_floor")

            spike_index = co.quicksum(float(t) * Md.p[t] for t in range(Tfine))
            obj1 = Md.net_weighted_sum + SPIKE_EARLY_TILT * float(Md.fullS) * spike_index
            if USE_OVERCAP_PENALTY: obj1 = obj1 + float(OVERCAP_WEIGHT) * Md.overcap_sum
            if USE_IDLE_PENALTY:    obj1 = obj1 + float(IDLE_WEIGHT) * Md.idle_sum
            try:
                Md.m.setObjectiveN(obj1, 1, COPT.MINIMIZE)
                Md.m.setObjParamN(1, "MultiObjPriority", 2)
                Md.m.setObjParamN(1, "MultiObjWeight", 1.0)
            except Exception:
                pass
            try:
                Md.m.setObjectiveN(Md.net[-1], 2, COPT.MINIMIZE)
                Md.m.setObjParamN(2, "MultiObjPriority", 2)
                Md.m.setObjParamN(2, "MultiObjWeight", float(Tfine))
            except Exception:
                pass
            if USE_PEAK_BACKLOG_OBJECTIVE:
                try:
                    Md.m.setObjectiveN(Md.u_max, 3, COPT.MINIMIZE)
                    Md.m.setObjParamN(3, "MultiObjPriority", 1)
                except Exception:
                    pass

            res = solve_mo_adaptive(Md.m, stage=f"MO/DIM[{dim_short(dim)}]{'[win]' if with_window else '[relax]'}",
                                    base_time=EFFORT["MO"], rel_gap=EFFORT["REL_GAP"], log_name=f"dim_{dim_short(dim)}_{log_suffix}")
            return Md, res

        Md, resD = build_and_solve_dim(with_window=True, log_suffix="w")
        if not _has_incumbent(Md.m):
            _log(f" [DIM] {dim}: no incumbent with window  relaxing window")
            Md, resD = build_and_solve_dim(with_window=False, log_suffix="relax")

        if _has_incumbent(Md.m):
            sel_model = selection_from_vars(Md.x)
            best_dim_pv_model = pv_from_selection(coeff_dim_w_f, sel_model)

            chosen_valmap = cherry_pick_from_pool(Md.m, Md, coeff_dim_w_f, best_dim_pv_model, POOL_PV_TOL)
            if chosen_valmap is not None:
                sel_pool = selection_from_values(Md.x, chosen_valmap)
                # validate chosen pool solution (time-cap)
                validate_time_cap_or_die(Md, full_envelope_M, chosen_valmap, label=tag_dim+"_pool")
                dump_pickle_full(Md, tag_dim,
                                 projects=projects, variants=variants,
                                 costs_input_df=costs_input_df, ben_kernel_df=ben_kernel_df,
                                 kernels_by_dim=kernels_by_dim, benefit_rate=BENEFIT_DISCOUNT_RATE,
                                 scenario_name=sc_key, primary_dim=f"{dim} [WEIGHTED]",
                                 Tfine=Tfine, full_envelope_M=full_envelope_M,
                                 baseline_envelope_M=baseline_envelope_M,
                                 sel_override=sel_pool, val_override=chosen_valmap)
            else:
                # validate incumbent (time-cap)
                val_by_id = {id(var): float(var.X) for var in Md.net}
                validate_time_cap_or_die(Md, full_envelope_M, val_by_id, label=tag_dim)
                dump_pickle_full(Md, tag_dim,
                                 projects=projects, variants=variants,
                                 costs_input_df=costs_input_df, ben_kernel_df=ben_kernel_df,
                                 kernels_by_dim=kernels_by_dim, benefit_rate=BENEFIT_DISCOUNT_RATE,
                                 scenario_name=sc_key, primary_dim=f"{dim} [WEIGHTED]",
                                 Tfine=Tfine, full_envelope_M=full_envelope_M,
                                 baseline_envelope_M=baseline_envelope_M)
        else:
            if tot_model_for_fallback is not None:
                dump_pickle_full(tot_model_for_fallback, tag_dim,
                                 projects=projects, variants=variants,
                                 costs_input_df=costs_input_df, ben_kernel_df=ben_kernel_df,
                                 kernels_by_dim=kernels_by_dim, benefit_rate=BENEFIT_DISCOUNT_RATE,
                                 scenario_name=sc_key, primary_dim=f"{dim} [WEIGHTED]",
                                 Tfine=Tfine, full_envelope_M=full_envelope_M,
                                 baseline_envelope_M=baseline_envelope_M,
                                 sel_override=total_best_sel, val_override=tot_valmap_for_fallback,
                                 status_override="OK_SUBOPT_LOCKED",
                                 extra_diag={"note": "dimension solver yielded no incumbent; reported Total solution"})

    return prev_dim_floors

# 
# One combo (cost_type, scenario, surplus buffer)
# 
def run_combo(ct: str, sc_key: str, sc_sheet: str, sur_key: str, baseline_surplus_M: float, plus_M: float,
              prev_dim_floors: Optional[Dict[str, float]] = None,
              prev_best_sel: Optional[Dict[Tuple[str,int], int]] = None):
    # ---- Load data, apply rules
    projects_all, variants_all, costs_input_df = load_costs(ct)
    variants, forced_exact = apply_forced_rules(variants_all, FORCED_START)
    projects = {p: projects_all[p] for p in variants.keys() if p in projects_all}
    benef_df, ben_kernel_df = load_benefits(sc_sheet)
    dims_order, kernels_by_dim = map_benefit_kernels(benef_df, variants)
    total_key = "Total"

    # ---- Build the list of non-Total dimensions to run based on DIMENSION_INCLUSIONS (case-insensitive)
    include_norm = {norm(k): bool(v) for k, v in (DIMENSION_INCLUSIONS or {}).items()}
    dims_to_run = [d for d in kernels_by_dim.keys()
                   if d.lower() != "total" and include_norm.get(norm(d), False)]

    # ---- Envelope magnitude
    full_envelope_M = float(baseline_surplus_M) + float(plus_M)
    Tfine = TFIXED
    _log(f" Fixed PV window: 2025{FINAL_YEAR} (Tfine={Tfine})")

    # ---- Structural feasibility
    tot_cost = float(sum(sum(meta["spend"]) for meta in variants.values()))
    tag_tot = f"{ct.replace(' ','').replace('-', '')}_{sc_key}_{sur_key}_pm{int(plus_M)}_{_DIM_SHORT['Total']}"
    if tot_cost > full_envelope_M * Tfine + 1e-9:
        msg = (f"Total project spend {tot_cost:,.1f} M exceeds envelope capacity "
               f"{full_envelope_M*Tfine:,.1f} M (FULLyears). Scenario infeasible.")
        print(" ", msg)
        fn = CACHE / f"{(PKL_PREFIX or '')}{tag_tot}_noSol.pkl"
        pkl_save(fn, {"status":"NoSolve","reason":"mass-balance infeasible","detail":msg,
                      "objective":"Total", "created_at": _now_stamp()})
        return None, prev_dim_floors, prev_best_sel

    # ---- Allowed starts
    allowed_fine = allowed_starts_fine(variants, forced_exact, Tfine)
    missing = [p for p in variants if not allowed_fine.get(p)]
    if missing:
        fn = CACHE / f"{(PKL_PREFIX or '')}{tag_tot}_noSol.pkl"
        pkl_save(fn, {"status":"NoSolve","reason":f"no allowed starts for {missing[:5]}",
                      "diagnostic":{"phase":"screen","Tfine":Tfine}, "objective": "Total", "created_at": _now_stamp()})
        return None, prev_dim_floors, prev_best_sel

    # ---- PV coefficients
    disc_vec = np.array([(1.0 + BENEFIT_DISCOUNT_RATE) ** t for t in range(Tfine)], dtype=float)
    coeff_total_fine = coeff_map_for_dim_fine(variants, kernels_by_dim[total_key], allowed_fine, Tfine, disc_vec)
    coeff_total_fine_int = coeff_int(coeff_total_fine)

    # ---- Greedy seed
    sel_hint = greedy_warm_start(variants, allowed_fine, Tfine, full_envelope_M, MAX_STARTS_PER_FY, forced_exact, coeff_total_fine,
                                 reuse_sel=prev_best_sel)

    # ---- ON-years hint
    L_hint = int(math.ceil(tot_cost / max(full_envelope_M,1e-9))) if full_envelope_M>0 else Tfine//2
    L_hint = min(Tfine, max(1, L_hint))

    # ---- Cross-buffer monotone floor target for Total
    prev_env = max([e for (ctk,sck,e) in _BEST_PV_BY_ENV.keys() if ctk==ct and sck==sc_key and e < full_envelope_M], default=None)
    total_floor_target = None
    if ENFORCE_MONOTONE_PV_ACROSS_BUFFERS and prev_env is not None:
        prev_best = _BEST_PV_BY_ENV[(ct, sc_key, prev_env)]
        total_floor_target = prev_best

    #  Orchestrated Total solve (no draft files; only final Total PKL)
    best_inc, final_gap = orchestrate_total(
        variants, allowed_fine, full_envelope_M, Tfine,
        coeff_total_fine, coeff_total_fine_int,
        total_floor_target, L_hint,
        sel_seed=sel_hint,
        rel_gap_target=EFFORT["REL_GAP"], time_budget_s=float(EFFORT["MO"]),
        fast_stop_gap=FAST_STOP_GAP, convergence_repeat_n=CONVERGENCE_REPEAT_N
    )

    if best_inc is None or best_inc.model is None:
        fn = CACHE / f"{(PKL_PREFIX or '')}{tag_tot}_noSol.pkl"
        pkl_save(fn, {"status":"NoSolve","reason":"orchestrator found no incumbent",
                      "objective":"Total", "created_at": _now_stamp()})
        return None, prev_dim_floors, prev_best_sel

    # ---- VALIDATE PRINCIPLE (abort on violation) ----
    validate_time_cap_or_die(best_inc.model, full_envelope_M, best_inc.val_by_id, label=tag_tot)

    # Persist Total (pool cherry-pick inside)  write TOTAL PKL FIRST
    chosen_valmap = cherry_pick_from_pool(best_inc.model.m, best_inc.model, coeff_total_fine, best_inc.pv, POOL_PV_TOL)
    if chosen_valmap is not None:
        sel_pool = selection_from_values(best_inc.model.x, chosen_valmap)
        # Validate chosen pool solution as well
        validate_time_cap_or_die(best_inc.model, full_envelope_M, chosen_valmap, label=tag_tot + "_pool")
        dump_pickle_full(best_inc.model, tag_tot,
                         projects=projects, variants=variants,
                         costs_input_df=costs_input_df, ben_kernel_df=ben_kernel_df,
                         kernels_by_dim=kernels_by_dim, benefit_rate=BENEFIT_DISCOUNT_RATE,
                         scenario_name=sc_key, primary_dim="Total",
                         Tfine=Tfine, full_envelope_M=full_envelope_M,
                         baseline_envelope_M=float(baseline_surplus_M),
                         sel_override=sel_pool, val_override=chosen_valmap,
                         extra_diag={"orchestrated_gap": f"{(final_gap if final_gap is not None else float('nan')):.4%}"} )
        best_sel_for_next = sel_pool
        valmap_for_fallback = chosen_valmap
    else:
        dump_pickle_full(best_inc.model, tag_tot,
                         projects=projects, variants=variants,
                         costs_input_df=costs_input_df, ben_kernel_df=ben_kernel_df,
                         kernels_by_dim=kernels_by_dim, benefit_rate=BENEFIT_DISCOUNT_RATE,
                         scenario_name=sc_key, primary_dim="Total",
                         Tfine=Tfine, full_envelope_M=full_envelope_M,
                         baseline_envelope_M=float(baseline_surplus_M),
                         sel_override=best_inc.sel, val_override=best_inc.val_by_id,
                         extra_diag={"orchestrated_gap": f"{(final_gap if final_gap is not None else float('nan')):.4%}"} )
        best_sel_for_next = best_inc.sel
        valmap_for_fallback = best_inc.val_by_id

    _BEST_PV_BY_ENV[(ct, sc_key, full_envelope_M)] = pv_from_selection(coeff_total_fine, best_sel_for_next)
    if best_sel_for_next:
        _BEST_SEL_BY_ENV[(ct, sc_key, full_envelope_M)] = best_sel_for_next

    # ---- Weighted-dimension runs controlled by DIMENSION_INCLUSIONS
    run_dimensions_for_env(
        ct, sc_key, sc_sheet, sur_key, plus_M,
        projects=projects, variants=variants,
        costs_input_df=costs_input_df, ben_kernel_df=ben_kernel_df,
        kernels_by_dim=kernels_by_dim, Tfine=Tfine, full_envelope_M=full_envelope_M,
        baseline_envelope_M=float(baseline_surplus_M),
        allowed_fine=allowed_fine, disc_vec=disc_vec,
        coeff_total_fine_int=coeff_total_fine_int,
        total_best_sel=best_sel_for_next, total_best_pv=pv_from_selection(coeff_total_fine, best_sel_for_next),
        L_hint=L_hint, tot_model_for_fallback=best_inc.model, tot_valmap_for_fallback=valmap_for_fallback,
        prev_dim_floors=prev_dim_floors,
        dims_filter=dims_to_run
    )

    return pv_from_selection(coeff_total_fine, best_sel_for_next), prev_dim_floors, best_sel_for_next

def optimise_family_for(ct: str, scenario_key: str, scenario_sheet: str) -> None:
    "Run the optimiser for a single cost type and benefit scenario."
    _refresh_calendar_from_cfg()
    CACHE.mkdir(parents=True, exist_ok=True)
    _sync_dimension_inclusions()

    dims_progress = ["Total"] + [
        str(dim)
        for dim in PRIMARY_OBJECTIVE_DIMS_TO_RUN
        if str(dim).strip() and str(dim).strip().lower() != "total"
    ]
    if not dims_progress:
        dims_progress = ["Total"]

    surplus_entries = sorted(
        ((str(key), float(value)) for key, value in SURPLUS_OPTIONS_M.items()),
        key=lambda item: item[1],
    )

    plus_levels = sorted({float(v) for v in PLUSMINUS_LEVELS_M})
    if not RUN_BENEFIT_PLUSMINUS:
        plus_levels = [0.0]
    if not plus_levels:
        plus_levels = [0.0]
    elif 0.0 not in plus_levels:
        plus_levels = [0.0] + plus_levels

    _notify(
        "family_start",
        cost_type=ct,
        scenario_key=scenario_key,
        scenario_sheet=scenario_sheet,
        dimensions=dims_progress,
        surplus_keys=[key for key, _ in surplus_entries],
        plus_levels=plus_levels,
    )

    prev_dim_floors: Optional[Dict[str, float]] = None
    prev_best_sel: Optional[Dict[Tuple[str, int], int]] = None

    for sur_key, base_m in surplus_entries:
        for plus in plus_levels:
            payload = {
                "cost_type": ct,
                "scenario_key": scenario_key,
                "primary_dim": "Total",
                "surplus_key": sur_key,
                "plus_level": plus,
            }
            _notify("solve_start", **payload)
            try:
                result = run_combo(
                    ct,
                    scenario_key,
                    scenario_sheet,
                    sur_key,
                    float(base_m),
                    float(plus),
                    prev_dim_floors=prev_dim_floors,
                    prev_best_sel=prev_best_sel,
                )
            except Exception as exc:
                payload["status"] = "error"
                payload["error"] = str(exc)
                _notify("solve_complete", **payload)
                raise
            else:
                total_pv, prev_dim_floors, prev_best_sel = result
                tag_tot = f"{ct.replace(' ', '').replace('-', '')}_{scenario_key}_{sur_key}_pm{int(plus)}_{_DIM_SHORT['Total']}"
                payload["status"] = "ok" if total_pv is not None else "nosolve"
                payload["pv"] = total_pv
                payload["cache_file"] = f"{(PKL_PREFIX or '')}{tag_tot}.pkl"
                _notify("solve_complete", **payload)

    _notify("family_complete", cost_type=ct, scenario_key=scenario_key)



# 
# Driver
# 
def main():
    random.seed(SOLVER_SEED_DEFAULT); np.random.seed(SOLVER_SEED_DEFAULT)

    # Ensure clean slate if running multiple times in a warm process
    global _BEST_PV_BY_ENV, _BEST_SEL_BY_ENV
    _BEST_PV_BY_ENV.clear()
    _BEST_SEL_BY_ENV.clear()

    print(f"CFG: START_FY={START_FY} FINAL_YEAR={FINAL_YEAR} PV_WINDOW={TFIXED}y MAX_STARTS/FY={MAX_STARTS_PER_FY}")
    print(f"Effort: {OPTIMISATION_PROFILE} -> {{'MO': {EFFORT['MO']}, 'REL_GAP': {EFFORT['REL_GAP']}}}")
    cap_year = START_FY + int(TIME_CAP_AFTER_YEARS)
    print(f"TIME-BASED CAP: from FY{cap_year} onward, ClosingNet <= {int(ALPHA_CAP*100)}% of FULL.")

    if ALLOW_TAPERED_ENVELOPE:
        print(("Envelope: flat (FULL) while ON in early/middle years; taper allowed only on the last "
               f"{TAPER_SUFFIX_MAX} ON-years; non-increasing after taper; drop cap {int(DROP_FRAC*100)}%/y."))
    else:
        print("Envelope: FULL when ON (no taper).")

    # Show which dimensions (besides Total) will run
    include_norm = {norm(k): bool(v) for k, v in (DIMENSION_INCLUSIONS or {}).items()}
    dims_incl_str = ", ".join([k for k,v in DIMENSION_INCLUSIONS.items() if k.lower()!="total" and v])
    print(f"Dimension inclusions (non-Total): {dims_incl_str if dims_incl_str else 'None'}")

    print(f"RUN_ID: {RUN_ID}\n")

    for ct in COST_TYPES_RUN:
        print(f"\n=== COST: {ct} ===")
        for sc_key, sc_sheet in BENEFIT_SCENARIOS.items():
            print(f">> Scenario {sc_key} (sheet {sc_sheet})")
            prev_sel: Optional[Dict[Tuple[str,int], int]] = None
            prev_dims: Dict[str, float] = {}
            envs_sorted = sorted(SURPLUS_OPTIONS_M.items(), key=lambda kv: kv[1])
            for sur_key, baseM in envs_sorted:
                print(f" -- Surplus {sur_key} = {baseM:.0f} M p.a. --")
                plus_values = sorted({float(v) for v in PLUSMINUS_LEVELS_M}) or [0.0]
                if not RUN_BENEFIT_PLUSMINUS:
                    plus_values = [0.0]
                elif 0.0 not in plus_values:
                    plus_values = [0.0] + plus_values
                for plus in plus_values:
                    print(f" +/-{int(plus)} -> FULL={baseM + plus:,.0f} M")
                    out = run_combo(ct, sc_key, sc_sheet, sur_key, baseM, plus,
                                    prev_dim_floors=prev_dims, prev_best_sel=prev_sel)
                    if out is not None:
                        tot_pv, prev_dims, prev_sel = out

    print("\nDone. Pickles are under:", CACHE)

if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    finally:
        print(f"\nRun-time: {time.time()-t0:.1f}s  {CACHE}")

