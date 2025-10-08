
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
    name = stem.split(".", 1)[0]
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

    ordered = sorted(counts.keys(), key=lambda p: (-counts[p], p))

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

    m = re.search(r"yoy(?:\+/-|\u00B1)(\d+)", stem, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"yoy\+(\d+)", stem, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


def _parse_cash_from_stem(stem: str) -> Optional[float]:
    import re

    m = re.search(r"cash\+(\d+)", stem, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


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
    if "bencb" in s:
        return "cash"
    if "benbuf" in s:
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
    for f in sorted(cache_dir.glob("*.pkl")):
        if f.stem.endswith("_noSol"):
            continue
        import pickle

        with f.open("rb") as fh:
            res = pickle.load(fh)
        res["_cache_file"] = f.name
        res["_cache_stem"] = f.stem
        res["_cache_path"] = str(f)
        results[f.stem] = res
    if not results:
        raise RuntimeError("No scenarios found (cache empty).")
    return results


def prepare_dashboard_data(results: Dict[str, Dict[str, Any]]) -> DashboardData:
    stems = sorted(results.keys())
    if not stems:
        raise RuntimeError("No scenario pickles supplied")

    detection = _detect_comparison_prefixes(stems)
    auto_prefixes: List[str] = detection["prefixes"]
    auto_labels: Dict[str, str] = detection["label_by_prefix"]

    first = results[stems[0]]
    start_fy = int(first.get("calendar", {}).get("start_fy", 2025))
    model_years = int(first.get("calendar", {}).get("years", 50))
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
        obj_dim = (res.get("objective", {}) or {}).get("primary_dim", "Total")

        cache_file = res.get("_cache_file", f"{stem}.pkl")
        profile_label = _derive_profile_label(cache_file)

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
            "BenefitPVByDim": {k: float(v) for k, v in (res.get("benefit_pv_by_dim") or {}).items()},
            "BenefitPVTotal": float(res.get("benefit_pv_total", 0.0)),
            "BenefitPVPrimary": float(res.get("benefit_pv_primary", 0.0)),
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
        env_val = meta["Envelope"] if meta["Envelope"] != "" else None
        yoy_buf = meta["Buffer"] if meta["Buffer"] != "" else None
        cash_buf = meta["CashPlus"] if meta["CashPlus"] != "" else None

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

        ben = res.get("benefit_flow", pd.DataFrame(columns=["Year", "BenefitFlow"]))
        if not ben.empty:
            ben_total = ben[["Year", "BenefitFlow"]].copy()
            ben_total["Year"] = pd.to_numeric(ben_total["Year"], errors="coerce").astype(int)
            ben_total["BenefitFlow"] = pd.to_numeric(ben_total["BenefitFlow"], errors="coerce").fillna(0.0)
        else:
            ben_total = pd.DataFrame(columns=["Year", "BenefitFlow"])
        ben_total.insert(0, "Code", code)
        ben_total.insert(0, "Key", ben_total["Code"] + "|" + ben_total["Year"].astype(str))
        ben_rows.append(ben_total)

        bendim_w = res.get("benefit_flow_by_dim_wide")
        if isinstance(bendim_w, pd.DataFrame) and "Year" in bendim_w.columns:
            df_w = bendim_w.copy()
        else:
            bendim_l = res.get("benefit_flow_by_dim_long")
            if isinstance(bendim_l, pd.DataFrame) and {"Year", "Dimension", "BenefitFlow"}.issubset(bendim_l.columns):
                df_w = bendim_l.pivot_table(index="Year", columns="Dimension", values="BenefitFlow", aggfunc="sum").reset_index()
            else:
                df_w = pd.DataFrame(columns=["Year"])
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
                "Key": f"{meta['Conf']}|{meta['BenSteep']}|{meta['BenHorizon']}|{meta['Mode']}|"
                f"{str(int(round(meta['Envelope']))) if meta['Envelope'] != '' else ''}|"
                f"{('+/-' + str(int(round(meta['Buffer'])))) if (meta['Mode'] == 'buffered' and meta['Buffer'] != '') else (('cash+' + str(int(round(meta['CashPlus'])))) if (meta['Mode'] == 'cash' and meta['CashPlus'] != '') else '')}",
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
) -> str:
    env_str = str(int(round(envelope))) if envelope is not None else ""
    if mode == "buffered":
        buf_str = f"+/-{int(round(buffer_value))}" if buffer_value is not None else ""
    elif mode == "cash":
        buf_str = f"cash+{int(round(buffer_value))}" if buffer_value is not None else ""
    else:
        buf_str = ""
    return f"{conf}|{benefit_steep}|{benefit_horizon}|{mode}|{env_str}|{buf_str}"


def find_scenario_code(
    data: DashboardData,
    *,
    conf: str,
    benefit_steep: str,
    benefit_horizon: int,
    mode: str,
    envelope: Optional[float],
    buffer_value: Optional[float],
    prefer_comparison: bool = False,
    profile: Optional[str] = None,
    scenarios_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    df_source = scenarios_df if scenarios_df is not None else data.scenarios
    if profile and "Profile" in df_source.columns:
        profile_mask = df_source["Profile"] == profile
        if profile_mask.any():
            df_source = df_source[profile_mask]
    key = build_scenario_key(conf, benefit_steep, benefit_horizon, mode, envelope, buffer_value)
    df = df_source[df_source["Key"] == key]
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





