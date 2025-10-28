
"""Data preparation utilities for dashboard visualisations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from capital_programme_optimiser.dashboard.constants import (
    SCENARIO_COMPARISON_NAME,
)

_DIM_SHORT = {
    "Total": "TOT",
    "Healthy and safe people": "HSP",
    "Inclusive access": "INC",
    "Environmental sustainability": "ENV",
    "Economic Prosperity": "ECO",
    "Urban Development": "URB",
    "Resilience and Security": "RES",
}

_SHORT_TO_DIM = {code: name for name, code in _DIM_SHORT.items()}


def dim_short(dim: str) -> str:
    if not dim:
        return "DIM"
    d = str(dim).strip()
    if d in _DIM_SHORT:
        return _DIM_SHORT[d]
    tokens = [token for token in pd.Series([d]).str.findall(r"[A-Za-z0-9]+").iloc[0] if token]
    if not tokens:
        return "DIM"
    code = ("".join(token[:3] for token in tokens)[:8]).upper()
    return code or "DIM"


DEFAULT_PROFILE_PREFIXES = (
    "P50REAL",
    "P95REAL",
    "P50NOMINAL",
    "P95NOMINAL",
)

DEFAULT_PROFILE_LABEL = "Default"


def _derive_profile_label(cache_name: str) -> str:
    """Map a cache filename stem to a human-friendly profile label."""
    stem = str(cache_name or "").strip()
    if not stem:
        return DEFAULT_PROFILE_LABEL
    from pathlib import Path

    name = Path(stem).name.split(".", 1)[0]
    upper = name.upper()
    if any(upper.startswith(prefix) for prefix in DEFAULT_PROFILE_PREFIXES):
        return DEFAULT_PROFILE_LABEL
    import re

    match = re.search(r"(P(?:50|95))", name, flags=re.IGNORECASE)
    if match:
        base = name[: match.start()]
    else:
        base = name
    base = base.strip("_- ")
    return base or DEFAULT_PROFILE_LABEL



def _split_prefix(stem: str) -> tuple[str, str]:
    import re

    m = re.search(r"(P50|P95)", stem, flags=re.IGNORECASE)
    if not m or m.start() == 0:
        return "", stem
    return stem[: m.start()], stem[m.start() :]



def _detect_comparison_prefixes(stems: Iterable[str]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for stem in stems:
        pref, _ = _split_prefix(stem)
        if pref:
            counts[pref] = counts.get(pref, 0) + 1

    comparison_counts = {pref: count for pref, count in counts.items() if count >= 2}
    ordered = sorted(comparison_counts.keys(), key=lambda p: (-comparison_counts[p], p))

    import re

    def _label(pref: str) -> str:
        base = pref.strip("_- ").strip()
        if not base:
            return SCENARIO_COMPARISON_NAME
        label = re.sub(r"[_\-]+", " ", base).strip()
        return label if label.isupper() and len(label) <= 6 else label.title()

    labels = {p: _label(p) for p in ordered}
    return {"prefixes": ordered, "label_by_prefix": labels, "count_by_prefix": counts}


def _is_comparison_stem_auto(stem: str, auto_prefixes: List[str]) -> tuple[bool, str]:
    pref, _ = _split_prefix(stem)
    if pref and pref in (auto_prefixes or []):
        return True, pref
    return False, ""


def _strip_detected_prefix(stem: str, prefix: str) -> str:
    return stem[len(prefix) :] if prefix and stem.startswith(prefix) else stem


def _parse_surplus_from_stem(stem: str) -> Optional[float]:
    import re

    m = re.search(r"_s(\d+)", stem, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


def _parse_buffer_from_stem(stem: str) -> Optional[float]:
    import re

    m = re.search(r"_pm(-?\d+)", stem, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"yoy(?:\+/-|\u00B1)(\d+)", stem, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"yoy\+(\d+)", stem, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


def _parse_cash_from_stem(stem: str) -> Optional[float]:
    import re

    m = re.search(r"cash\+(\d+)", stem, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


def _infer_dimension_from_stem(stem: str) -> Optional[str]:
    """Infer the primary objective dimension from the cache filename stem."""
    if not stem:
        return None
    token = stem.split("_")[-1].strip()
    if not token:
        return None
    upper = token.upper()
    if upper in _SHORT_TO_DIM:
        return _SHORT_TO_DIM[upper]
    return None


def _parse_confidence(stem: str) -> str:
    s = stem.upper()
    base = "P50" if "P50" in s else "P95" if "P95" in s else "P50"
    if "REAL" in s:
        return f"{base} - Real"
    if "NOM" in s or "NOMINAL" in s:
        return f"{base} - Nominal"
    return base


def _parse_mode(stem: str) -> str:
    s = stem.lower()
    if "bencb" in s or "cash" in s:
        return "cash"
    if "benbuf" in s or _parse_buffer_from_stem(stem) is not None:
        return "buffered"
    if "fixedenv" in s:
        return "fixed"
    if "unconstrained" in s:
        return "unconstrained"
    return "fixed" if _parse_surplus_from_stem(stem) is not None else "unconstrained"


def _parse_benefit_scenario(stem: str, res: Dict[str, Any]) -> tuple[str, int]:
    import re

    scenario = str(res.get("scenario", "")).strip()
    m = re.match(r"([AB])(\d{2})$", scenario, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"([AB])(\d{2})", stem, flags=re.IGNORECASE)
    return (m.group(1).upper(), int(m.group(2))) if m else ("A", 60)


def _comp_tag(prefix: str, label: str) -> str:
    import re

    base = label.split("+/-")[0].strip() if label else prefix.strip("_-")
    return re.sub(r"[^A-Za-z0-9]+", "", base) or prefix.strip("_-")

def comp_tag(prefix: str, label: str) -> str:
    """Public wrapper around `_comp_tag` for callers that need tag labels."""
    return _comp_tag(prefix, label)



def _build_scenario_code(
    conf: str,
    steep: str,
    horizon: int,
    mode: str,
    env_val: Optional[float],
    yoy_buf: Optional[float],
    cash_buf: Optional[float],
    obj_dim: str,
    comp_prefix: str = "",
    comp_label: str = "",
) -> str:
    parts = [f"{conf}", f"{steep}{horizon}"]
    if mode == "unconstrained":
        parts.append("Unc")
    elif mode == "fixed":
        parts.extend(["Fix", f"{int(round(env_val))}" if env_val is not None else "NA"])
    elif mode == "buffered":
        parts.extend([
            "Buf",
            f"+/-{int(round(yoy_buf))}" if yoy_buf is not None else "+/-NA",
            f"@{int(round(env_val))}" if env_val is not None else "@NA",
        ])
    elif mode == "cash":
        parts.extend([
            "Cash",
            f"+{int(round(cash_buf))}" if cash_buf is not None else "+NA",
            f"@{int(round(env_val))}" if env_val is not None else "@NA",
        ])
    parts.append(f"OBJ{dim_short(obj_dim)}")
    base = "-".join(parts)
    if comp_prefix:
        tag = _comp_tag(comp_prefix, comp_label)
        return f"{tag}/{base}"
    return base


def _build_scenario_title(
    conf: str,
    steep: str,
    horizon: int,
    mode: str,
    env_val: Optional[float],
    yoy_buf: Optional[float],
    cash_buf: Optional[float],
    obj_dim: str,
) -> str:
    bits = [f"{conf} costs", f"{steep}{horizon} benefits"]
    if mode == "unconstrained":
        bits.append("Unconstrained envelope")
    elif mode == "fixed":
        env_text = f"$ {int(round(env_val)):,}m p.a." if env_val is not None else "N/A"
        bits.append(f"Fixed envelope {env_text}")
    elif mode == "buffered":
        buf_text = f"+/-$ {int(round(yoy_buf)):,}m YoY" if yoy_buf is not None else "+/-N/A"
        base = f"$ {int(round(env_val)):,}m p.a." if env_val is not None else "N/A"
        bits.append(f"Buffered envelope {buf_text} around {base}")
    elif mode == "cash":
        cash_text = f"cash +$ {int(round(cash_buf)):,}m" if cash_buf is not None else "cash +N/A"
        base = f"$ {int(round(env_val)):,}m p.a." if env_val is not None else "N/A"
        bits.append(f"Cash-plus envelope ({cash_text}) on {base}")
    bits.append(f"Objective: {obj_dim or 'Total'}")
    return " | ".join(bits)


def _unique_ints(series: pd.Series) -> List[int]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return sorted({int(round(float(v))) for v in vals.tolist()})


def unique_ints(series: pd.Series) -> List[int]:
    """Public wrapper around `_unique_ints` for external callers."""
    return _unique_ints(series)


@dataclass
class ScenarioOptions:
    conf: List[str]
    benefit_steep: List[str]
    benefit_horizon: List[int]
    objective_dims: List[str]
    envelopes: List[int]
    buffers: List[int]
    modes: List[str]


@dataclass
class ProjectRun:
    project: str
    start_year: int
    end_year: int
    total_spend: float
    values: List[float]


@dataclass
class DashboardData:
    raw_results: Dict[str, Dict[str, Any]]
    scenario_meta_by_stem: Dict[str, Dict[str, Any]]
    scenario_meta_by_code: Dict[str, Dict[str, Any]]
    scenarios: pd.DataFrame
    comparison_pairs: pd.DataFrame
    cf: pd.DataFrame
    benefit: pd.DataFrame
    benefit_dim: pd.DataFrame
    schedule: pd.DataFrame
    spend_matrix: pd.DataFrame
    years: List[int]
    dims: List[str]
    projects: List[str]
    start_fy: int
    model_years: int
    benefit_rate: float
    auto_prefixes: List[str]
    auto_labels: Dict[str, str]

    def scenario_options(self) -> ScenarioOptions:
        df = self.scenarios
        envelopes = unique_ints(df["Envelope"]) if "Envelope" in df else []
        buffer_vals = unique_ints(df["Buffer"]) if "Buffer" in df else []
        cash_vals = unique_ints(df["CashPlus"]) if "CashPlus" in df else []
        buffers = sorted(set(buffer_vals + cash_vals))

        if "OrigStem" in df.columns:
            conf_series = df["OrigStem"].astype(str).apply(_parse_confidence)
        else:
            conf_series = df["Conf"].astype(str)

        return ScenarioOptions(
            conf=sorted(conf_series.dropna().unique().tolist()),
            benefit_steep=sorted(df["BenSteep"].dropna().astype(str).unique().tolist()),
            benefit_horizon=sorted(df["BenHorizon"].dropna().drop_duplicates().astype(int).tolist()),
            objective_dims=sorted(df["ObjectiveDim"].dropna().astype(str).unique().tolist()),
            envelopes=envelopes,
            buffers=buffers,
            modes=sorted(df["Mode"].dropna().astype(str).unique().tolist()),
        )


def load_results(cache_dir: Path) -> Dict[str, Dict[str, Any]]:
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache dir not found: {cache_dir}")
    results: Dict[str, Dict[str, Any]] = {}
    files = sorted(
        (p for p in cache_dir.rglob("*.pkl") if p.is_file()),
        key=lambda p: str(p.relative_to(cache_dir)).lower(),
    )
    for f in files:
        if f.stem.endswith("_noSol"):
            continue
        import pickle

        with f.open("rb") as fh:
            res = pickle.load(fh)
        try:
            rel_path = f.relative_to(cache_dir)
        except ValueError:
            rel_path = f.name
        res["_cache_file"] = str(rel_path)
        res["_cache_stem"] = f.stem
        res["_cache_path"] = str(f)
        results[f.stem] = res
    if not results:
        raise RuntimeError("No scenarios found (cache empty).")
    return results


def _normalise_total_benefit(res: Dict[str, Any]) -> pd.DataFrame:
    """Return a Year/BenefitFlow frame from varying solver outputs."""

    def _coerce_year_table(source: pd.DataFrame) -> pd.DataFrame:
        df_alt = source.copy()
        df_alt["Year"] = pd.to_numeric(df_alt["Year"], errors="coerce")
        df_alt = df_alt.dropna(subset=["Year"]).copy()
        df_alt["Year"] = df_alt["Year"].astype(int)
        total_col = None
        for candidate in df_alt.columns:
            if candidate == "Year":
                continue
            if str(candidate).strip().lower() == "total":
                total_col = candidate
                break
        if total_col is None:
            value_cols = [c for c in df_alt.columns if c != "Year"]
            if not value_cols:
                return pd.DataFrame(columns=["Year", "BenefitFlow"])
            df_alt[value_cols] = df_alt[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            df_alt["BenefitFlow"] = df_alt[value_cols].sum(axis=1)
        else:
            df_alt["BenefitFlow"] = pd.to_numeric(df_alt[total_col], errors="coerce").fillna(0.0)
        return df_alt[["Year", "BenefitFlow"]]

    ben = res.get("benefit_flow")
    if isinstance(ben, pd.DataFrame) and {"Year", "BenefitFlow"}.issubset(ben.columns):
        df = ben[["Year", "BenefitFlow"]].copy()
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["BenefitFlow"] = pd.to_numeric(df["BenefitFlow"], errors="coerce").fillna(0.0)
        df = df.dropna(subset=["Year"]).copy()
        df["Year"] = df["Year"].astype(int)
        return df

    full = res.get("benefits_by_year_full")
    if isinstance(full, pd.DataFrame) and "Year" in full.columns:
        return _coerce_year_table(full)

    alt = res.get("benefits_by_year")
    if isinstance(alt, pd.DataFrame) and "Year" in alt.columns:
        return _coerce_year_table(alt)

    return pd.DataFrame(columns=["Year", "BenefitFlow"])


def _benefit_dim_from_project_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convert benefits_by_project_dimension_by_year into a wide Year/Dimension table."""
    if df.empty:
        return pd.DataFrame(columns=["Year"])
    working = df.copy()
    if isinstance(working.index, pd.MultiIndex):
        working = working.reset_index()
    value_cols = [c for c in working.columns if str(c).isdigit()]
    if not value_cols:
        numeric_cols = [c for c in working.columns if pd.api.types.is_numeric_dtype(working[c])]
        value_cols = [c for c in numeric_cols if c != "Year"]
    if not value_cols or "Dimension" not in working.columns:
        return pd.DataFrame(columns=["Year"])
    melted = working.melt(
        id_vars=[c for c in working.columns if c not in value_cols],
        value_vars=value_cols,
        var_name="Year",
        value_name="BenefitFlow",
    )
    melted["Year"] = pd.to_numeric(melted["Year"], errors="coerce")
    melted = melted.dropna(subset=["Year"]).copy()
    melted["Year"] = melted["Year"].astype(int)
    melted["BenefitFlow"] = pd.to_numeric(melted["BenefitFlow"], errors="coerce").fillna(0.0)
    grouped = (
        melted.groupby(["Year", "Dimension"], as_index=False)["BenefitFlow"]
        .sum()
        .pivot_table(index="Year", columns="Dimension", values="BenefitFlow", aggfunc="sum")
        .reset_index()
    )
    return grouped


def _normalise_benefit_dim_wide(res: Dict[str, Any]) -> pd.DataFrame:
    """Return a Year x Dimension wide table from multiple solver outputs."""
    bendim_w = res.get("benefit_flow_by_dim_wide")
    if isinstance(bendim_w, pd.DataFrame) and "Year" in bendim_w.columns:
        return bendim_w.copy()

    bendim_l = res.get("benefit_flow_by_dim_long")
    if isinstance(bendim_l, pd.DataFrame) and {"Year", "Dimension", "BenefitFlow"}.issubset(bendim_l.columns):
        df_l = bendim_l.copy()
        df_l["Year"] = pd.to_numeric(df_l["Year"], errors="coerce")
        df_l = df_l.dropna(subset=["Year"]).copy()
        df_l["Year"] = df_l["Year"].astype(int)
        df_l["BenefitFlow"] = pd.to_numeric(df_l["BenefitFlow"], errors="coerce").fillna(0.0)
        return (
            df_l.pivot_table(index="Year", columns="Dimension", values="BenefitFlow", aggfunc="sum")
            .reset_index()
        )

    alt_full = res.get("benefits_by_year_full")
    if isinstance(alt_full, pd.DataFrame) and "Year" in alt_full.columns:
        df_full = alt_full.copy()
        df_full["Year"] = pd.to_numeric(df_full["Year"], errors="coerce")
        df_full = df_full.dropna(subset=["Year"]).copy()
        df_full["Year"] = df_full["Year"].astype(int)
        dim_cols_full = [c for c in df_full.columns if c != "Year"]
        if dim_cols_full:
            for c in dim_cols_full:
                df_full[c] = pd.to_numeric(df_full[c], errors="coerce").fillna(0.0)
            return df_full[["Year"] + dim_cols_full]

    alt = res.get("benefits_by_year")
    if isinstance(alt, pd.DataFrame) and "Year" in alt.columns:
        df_alt = alt.copy()
        df_alt["Year"] = pd.to_numeric(df_alt["Year"], errors="coerce")
        df_alt = df_alt.dropna(subset=["Year"]).copy()
        df_alt["Year"] = df_alt["Year"].astype(int)
        dim_cols = [c for c in df_alt.columns if c != "Year"]
        if dim_cols:
            for c in dim_cols:
                df_alt[c] = pd.to_numeric(df_alt[c], errors="coerce").fillna(0.0)
            return df_alt[["Year"] + dim_cols]

    alt_proj = res.get("benefits_by_project_dimension_by_year")
    if isinstance(alt_proj, pd.DataFrame):
        return _benefit_dim_from_project_table(alt_proj)

    return pd.DataFrame(columns=["Year"])


def prepare_dashboard_data(results: Dict[str, Dict[str, Any]]) -> DashboardData:
    stems = sorted(results.keys())
    if not stems:
        raise RuntimeError("No scenario pickles supplied")

    detection = _detect_comparison_prefixes(stems)
    auto_prefixes: List[str] = detection["prefixes"]
    auto_labels: Dict[str, str] = detection["label_by_prefix"]

    first = results[stems[0]]
    start_fy = int(first.get("calendar", {}).get("start_fy", 2025))
    model_years = int(first.get("calendar", {}).get("years", 60))
    benefit_rate = float(first.get("benefit_rate", 0.02))

    scenario_meta: Dict[str, Dict[str, Any]] = {}
    used_codes: set[str] = set()

    for stem in stems:
        res = results[stem]
        is_comp, pref = _is_comparison_stem_auto(stem, auto_prefixes)
        pref_label = auto_labels.get(pref, pref.strip("_-")) if is_comp else ""
        base_stem = _strip_detected_prefix(stem, pref) if is_comp else stem

        conf = _parse_confidence(stem)
        mode = _parse_mode(stem)
        env_val = _parse_surplus_from_stem(stem) if mode in {"fixed", "buffered", "cash"} else None
        yoy_buf = _parse_buffer_from_stem(stem) if mode == "buffered" else None
        cash_buf = _parse_cash_from_stem(stem) if mode == "cash" else None
        steep, horizon = _parse_benefit_scenario(stem, res)
        obj_dim_raw = (res.get("objective", {}) or {}).get("primary_dim")
        inferred_dim = _infer_dimension_from_stem(base_stem)
        obj_dim = str(obj_dim_raw).strip() if obj_dim_raw else ""
        if inferred_dim and inferred_dim.strip():
            if not obj_dim or obj_dim.lower() != inferred_dim.strip().lower():
                obj_dim = inferred_dim.strip()
        if not obj_dim:
            obj_dim = "Total"

        cache_file = res.get("_cache_file", f"{stem}.pkl")
        profile_label = _derive_profile_label(cache_file)

        pv_by_dim_raw = res.get("benefit_pv_by_dim") or res.get("pv_by_dimension") or {}
        pv_by_dim = {str(k): float(v) for k, v in (pv_by_dim_raw or {}).items()}
        pv_total_raw = res.get("benefit_pv_total")
        if pv_total_raw is None:
            pv_total_raw = res.get("pv_total")
            if pv_total_raw is None and pv_by_dim:
                pv_total_raw = sum(pv_by_dim.values())
        pv_total = float(pv_total_raw or 0.0)
        pv_primary_raw = res.get("benefit_pv_primary")
        if pv_primary_raw is None:
            if obj_dim and obj_dim in pv_by_dim:
                pv_primary_raw = pv_by_dim[obj_dim]
            elif "Total" in pv_by_dim:
                pv_primary_raw = pv_by_dim["Total"]
            else:
                pv_primary_raw = pv_total
        pv_primary = float(pv_primary_raw or 0.0)

        code = _build_scenario_code(conf, steep, horizon, mode, env_val, yoy_buf, cash_buf, obj_dim, pref if is_comp else "", pref_label)
        title = _build_scenario_title(conf, steep, horizon, mode, env_val, yoy_buf, cash_buf, obj_dim)

        base_code = code
        k = 2
        while code in used_codes:
            code = f"{base_code}-{k}"
            k += 1
        used_codes.add(code)

        scenario_meta[stem] = {
            "Conf": conf,
            "Mode": mode,
            "Envelope": env_val if env_val is not None else "",
            "Buffer": yoy_buf if yoy_buf is not None else "",
            "CashPlus": cash_buf if cash_buf is not None else "",
            "BenSteep": steep,
            "BenHorizon": horizon,
            "ObjectiveDim": obj_dim,
            "ObjectiveDimShort": dim_short(obj_dim),
            "Code": code,
            "Title": title,
            "IsComp": 1 if is_comp else 0,
            "Prefix": pref if is_comp else "",
            "CompLabel": pref_label,
            "BaseStem": base_stem if is_comp else stem,
            "OrigStem": stem,
            "CacheStem": res.get("_cache_stem", stem),
            "CacheFile": res.get("_cache_file", f"{stem}.pkl"),
            "StartFY": int(res.get("calendar", {}).get("start_fy", start_fy)),
            "HorizonYears": int(res.get("calendar", {}).get("years", model_years)),
            "BenRate": float(res.get("benefit_rate", benefit_rate)),
            "BenefitPVByDim": pv_by_dim,
            "BenefitPVTotal": pv_total,
            "BenefitPVPrimary": pv_primary,
            "Profile": profile_label,
        }

    scenario_meta_by_code = {
        meta["Code"]: {**meta, "_stem": stem}
        for stem, meta in scenario_meta.items()
    }

    comparison_pairs: List[Dict[str, Any]] = []
    for stem in stems:
        meta = scenario_meta[stem]
        if meta["IsComp"] == 1:
            base_stem = meta["BaseStem"]
            base_meta = scenario_meta.get(base_stem)
            if base_meta:
                comparison_pairs.append(
                    {
                        "BaseStem": base_stem,
                        "BaseCode": base_meta["Code"],
                        "BaseTitle": base_meta["Title"],
                        "CompStem": stem,
                        "CompCode": meta["Code"],
                        "CompTitle": meta["Title"],
                        "Prefix": meta["Prefix"],
                        "CompLabel": meta["CompLabel"],
                        "PairKey": f"{base_meta['Code']}|{meta['Code']}",
                    }
                )

    cf_rows, ben_rows, bendim_rows, sched_rows, spmat_rows = [], [], [], [], []
    years_union: set[int] = set()
    dims_set: set[str] = set()

    for stem in stems:
        res = results[stem]
        meta = scenario_meta[stem]
        code = meta["Code"]
        mode = meta["Mode"]
        objective_dim = meta.get("ObjectiveDim")
        if objective_dim:
            dims_set.add(str(objective_dim))
        env_val = meta["Envelope"] if meta["Envelope"] != "" else None
        yoy_buf = meta["Buffer"] if meta["Buffer"] != "" else None
        cash_buf = meta["CashPlus"] if meta["CashPlus"] != "" else None

        calendar_info = res.get("calendar") or {}
        cal_start = calendar_info.get("start_fy")
        cal_years = calendar_info.get("years")
        try:
            cal_start_int = int(cal_start)
            cal_years_int = int(cal_years)
        except (TypeError, ValueError):
            cal_start_int = None
            cal_years_int = None
        if cal_start_int is not None and cal_years_int is not None and cal_years_int > 0:
            years_union.update(cal_start_int + i for i in range(cal_years_int))

        meta_window = (res.get("meta") or {}).get("pv_window") or {}
        base_year = meta_window.get("base_year")
        last_year = meta_window.get("last_pv_year")
        try:
            base_year_int = int(base_year)
            last_year_int = int(last_year)
        except (TypeError, ValueError):
            base_year_int = None
            last_year_int = None
        if (
            base_year_int is not None
            and last_year_int is not None
            and last_year_int >= base_year_int
        ):
            years_union.update(range(base_year_int, last_year_int + 1))

        cf = res.get("cash_flow", pd.DataFrame(columns=["Year", "Spend", "ClosingNet", "Envelope"]))
        if not cf.empty:
            cfd = cf[["Year", "Spend", "ClosingNet", "Envelope"]].copy()
            cfd["Year"] = pd.to_numeric(cfd["Year"], errors="coerce").astype(int)
            if mode == "fixed" and env_val is not None:
                cfd["Envelope"] = float(env_val)
        else:
            cfd = pd.DataFrame(columns=["Year", "Spend", "ClosingNet", "Envelope"])
        cfd.insert(0, "Code", code)
        cfd.insert(0, "Key", cfd["Code"] + "|" + cfd["Year"].astype(str))
        cf_rows.append(cfd)

        ben_total = _normalise_total_benefit(res)
        ben_total.insert(0, "Code", code)
        ben_total.insert(0, "Key", ben_total["Code"] + "|" + ben_total["Year"].astype(str))
        ben_rows.append(ben_total)

        df_w = _normalise_benefit_dim_wide(res)
        if not df_w.empty:
            df_w["Year"] = pd.to_numeric(df_w["Year"], errors="coerce").astype(int)
            dim_cols = [c for c in df_w.columns if c != "Year"]
            for c in dim_cols:
                df_w[c] = pd.to_numeric(df_w[c], errors="coerce").fillna(0.0)
                if str(c).strip():
                    dims_set.add(str(c))
            long = df_w.melt(id_vars="Year", value_vars=dim_cols, var_name="Dimension", value_name="BenefitFlow")
            long["BenefitFlow"] = pd.to_numeric(long["BenefitFlow"], errors="coerce").fillna(0.0)
            long.insert(0, "Code", code)
            long.insert(0, "Key", long["Code"] + "|" + long["Dimension"].astype(str) + "|" + long["Year"].astype(str))
            bendim_rows.append(long[["Key", "Code", "Dimension", "Year", "BenefitFlow"]])

        sched = res.get("schedule", pd.DataFrame())
        if not sched.empty:
            sched_rows.append(sched[["Project", "StartFY", "EndFY", "Dur"]].assign(Code=code))

        spend = res.get("spend", pd.DataFrame())
        if not spend.empty:
            sp0 = spend.copy()
            if "Total Spend" in sp0.index:
                sp0 = sp0.drop(index="Total Spend")
            year_cols = [c for c in sp0.columns if str(c).isdigit()]
            years_union.update(int(c) for c in year_cols)
            sp0 = sp0[year_cols]
            sp0.columns = [int(c) for c in sp0.columns]
            for proj, row in sp0.iterrows():
                rec = {"Key": f"{code}|{proj}", "Code": code, "Project": str(proj)}
                rec.update({int(year): float(row[year]) for year in sp0.columns})
                spmat_rows.append(rec)

    df_scen = pd.DataFrame(
        [
            {
                "Key": build_scenario_key(
                    meta["Conf"],
                    meta["BenSteep"],
                    int(meta["BenHorizon"]),
                    meta["Mode"],
                    float(meta["Envelope"]) if meta["Envelope"] != "" else None,
                    float(meta["Buffer"])
                    if (meta["Mode"] == "buffered" and meta["Buffer"] != "")
                    else (
                        float(meta["CashPlus"])
                        if (meta["Mode"] == "cash" and meta["CashPlus"] != "")
                        else None
                    ),
                    objective_dim=meta.get("ObjectiveDim"),
                ),
                "Conf": meta["Conf"],
                "BenSteep": meta["BenSteep"],
                "BenHorizon": meta["BenHorizon"],
                "Mode": meta["Mode"],
                "EnvStr": str(int(round(meta["Envelope"]))) if meta["Envelope"] != "" else "",
                "BuffStr": (
                    "+/-" + str(int(round(meta["Buffer"])))
                    if (meta["Mode"] == "buffered" and meta["Buffer"] != "")
                    else ("cash+" + str(int(round(meta["CashPlus"]))) if (meta["Mode"] == "cash" and meta["CashPlus"] != "") else "")
                ),
                "Code": meta["Code"],
                "Envelope": meta["Envelope"],
                "Buffer": meta["Buffer"],
                "CashPlus": meta["CashPlus"],
                "ObjectiveDim": meta["ObjectiveDim"],
                "ObjectiveDimShort": meta.get("ObjectiveDimShort", ""),
                "ScenarioTitle": meta["Title"],
                "OrigStem": meta["OrigStem"],
                "CacheStem": meta["CacheStem"],
                "CacheFile": meta["CacheFile"],
                "Profile": meta["Profile"],
                "StartFY": meta["StartFY"],
                "HorizonYears": meta["HorizonYears"],
                "BenRate": meta["BenRate"],
                "IsComp": meta["IsComp"],
            }
            for meta in scenario_meta.values()
        ]
    ).drop_duplicates(subset=["Key", "Code"]).reset_index(drop=True)

    if "OrigStem" in df_scen.columns:
        df_scen["Conf"] = df_scen["OrigStem"].astype(str).apply(_parse_confidence)

    df_cf = pd.concat(cf_rows, ignore_index=True) if cf_rows else pd.DataFrame(columns=["Key", "Code", "Year", "Spend", "ClosingNet", "Envelope"])
    df_cf = df_cf[["Key", "Code", "Year", "Spend", "ClosingNet", "Envelope"]]

    df_ben = pd.concat(ben_rows, ignore_index=True) if ben_rows else pd.DataFrame(columns=["Key", "Code", "Year", "BenefitFlow"])
    df_ben = df_ben[["Key", "Code", "Year", "BenefitFlow"]]

    df_bendim = pd.concat(bendim_rows, ignore_index=True) if bendim_rows else pd.DataFrame(columns=["Key", "Code", "Dimension", "Year", "BenefitFlow"])
    df_bendim = df_bendim[["Key", "Code", "Dimension", "Year", "BenefitFlow"]]

    df_sched = pd.concat(sched_rows, ignore_index=True) if sched_rows else pd.DataFrame(columns=["Code", "Project", "StartFY", "EndFY", "Dur"])
    if not df_sched.empty and "Code" not in df_sched.columns:
        df_sched.insert(0, "Code", df_sched.pop("Code"))
    df_sched = df_sched[["Code", "Project", "StartFY", "EndFY", "Dur"]]

    years_union = years_union or {start_fy + i for i in range(model_years)}
    years = sorted(years_union)
    spmat_cols = ["Key", "Code", "Project"] + years
    df_spmat = pd.DataFrame(spmat_rows, columns=spmat_cols).fillna(0.0) if spmat_rows else pd.DataFrame(columns=spmat_cols)

    dims = sorted({str(d) for d in dims_set if str(d).strip()}) or ["Total"]
    projects = sorted(df_sched["Project"].dropna().unique().tolist()) if not df_sched.empty else []

    return DashboardData(
        raw_results=results,
        scenario_meta_by_stem=scenario_meta,
        scenario_meta_by_code=scenario_meta_by_code,
        scenarios=df_scen,
        comparison_pairs=pd.DataFrame(comparison_pairs),
        cf=df_cf,
        benefit=df_ben,
        benefit_dim=df_bendim,
        schedule=df_sched,
        spend_matrix=df_spmat,
        years=years,
        dims=dims,
        projects=projects,
        start_fy=start_fy,
        model_years=model_years,
        benefit_rate=benefit_rate,
        auto_prefixes=auto_prefixes,
        auto_labels=auto_labels,
    )


def build_scenario_key(
    conf: str,
    benefit_steep: str,
    benefit_horizon: int,
    mode: str,
    envelope: Optional[float],
    buffer_value: Optional[float],
    *,
    objective_dim: Optional[str] = None,
) -> str:
    env_str = str(int(round(envelope))) if envelope is not None else ""
    if mode == "buffered":
        buf_str = f"+/-{int(round(buffer_value))}" if buffer_value is not None else ""
    elif mode == "cash":
        buf_str = f"cash+{int(round(buffer_value))}" if buffer_value is not None else ""
    else:
        buf_str = ""
    dim_str = dim_short(objective_dim) if objective_dim else ""
    return f"{conf}|{benefit_steep}|{benefit_horizon}|{mode}|{env_str}|{buf_str}|{dim_str}"


def find_scenario_code(
    data: DashboardData,
    *,
    conf: str,
    benefit_steep: str,
    benefit_horizon: int,
    mode: str,
    envelope: Optional[float],
    buffer_value: Optional[float],
    objective_dim: Optional[str] = None,
    prefer_comparison: bool = False,
    profile: Optional[str] = None,
    scenarios_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    df_source = scenarios_df if scenarios_df is not None else data.scenarios
    if profile and "Profile" in df_source.columns:
        profile_mask = df_source["Profile"] == profile
        if profile_mask.any():
            df_source = df_source[profile_mask]
    key = build_scenario_key(
        conf,
        benefit_steep,
        benefit_horizon,
        mode,
        envelope,
        buffer_value,
        objective_dim=objective_dim,
    )
    df = df_source[df_source["Key"] == key]
    if df.empty and objective_dim and "ObjectiveDim" in df_source.columns:
        dim_series = df_source["ObjectiveDim"].astype(str).str.strip()
        target = str(objective_dim).strip().lower()
        short_target = dim_short(objective_dim).lower()
        dim_mask = (
            (dim_series.str.lower() == target)
            | (dim_series.apply(dim_short).str.lower() == short_target)
        )
        df = df_source[dim_mask & (df_source["Key"].str.startswith(f"{conf}|{benefit_steep}|{benefit_horizon}|{mode}|"))]
    if df.empty:
        return None
    flag = 1 if prefer_comparison else 0
    subset = df[df["IsComp"] == flag]
    if not subset.empty:
        return str(subset.iloc[0]["Code"])
    return str(df.iloc[0]["Code"])


def extract_project_runs(data: DashboardData, code: str, min_value: float = 1e-6) -> List[ProjectRun]:
    df = data.spend_matrix[data.spend_matrix["Code"] == code]
    runs: List[ProjectRun] = []
    years = data.years
    for _, row in df.iterrows():
        project = str(row["Project"])
        values = [float(row.get(year, 0.0)) for year in years]
        nz = [(year, value) for year, value in zip(years, values) if abs(value) > min_value]
        if not nz:
            continue
        start = nz[0][0]
        end = nz[-1][0]
        total = sum(value for _, value in nz)
        runs.append(ProjectRun(project=project, start_year=start, end_year=end, total_spend=total, values=values))
    runs.sort(key=lambda r: (r.start_year, r.project))
    return runs


def scenario_metadata(data: DashboardData, code: str) -> Optional[Dict[str, Any]]:
    return data.scenario_meta_by_code.get(code)





