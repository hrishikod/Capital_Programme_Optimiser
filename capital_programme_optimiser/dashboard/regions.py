from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
import re
import unicodedata
import numpy as np
import requests
import pandas as pd

from .data import DashboardData
from ..config import load_project_region_mapping, load_settings


# -----------------------------
# Region geometry source (ArcGIS REST)
# -----------------------------

@dataclass(frozen=True)
class RegionGeometrySource:
    # Stats NZ/ArcGIS services occasionally move; keep fields minimal and modern.
    url: str = (
        "https://services.arcgis.com/XTtANUDT8Va4DLwI/arcgis/rest/services/"
        "nz_regional_councils/FeatureServer/0/query"
    )
    where: str = "1=1"
    # Ask for all attributes so legacy/regional name fields (REGC_NAME, etc.) are available.
    out_fields: Tuple[str, ...] = ("*",)
    spatial_ref: int = 4326


GEOMETRY_LOCAL_PATH = Path(__file__).with_name("nz_regional_councils_2025.geojson")

SETTINGS = load_settings()
BENEFIT_SCENARIOS = dict(SETTINGS.data.benefit_scenarios)
SCORING_WORKBOOK = SETTINGS.scoring_workbook()

DEFAULT_ANNUAL_POP_GROWTH = 0.01

# Prefer 2025 fields first; keep legacy fallbacks as distant backups
NAME_FIELD_PRIORITY: Tuple[str, ...] = (
    "REGC2025_V1_00_NAME",
    "REGC_NAME",      # legacy fallback if seen on some services
    "REGC_name",      # another legacy form
)

ASCII_FIELD_PRIORITY: Tuple[str, ...] = (
    "REGC2025_V1_00_NAME_ASCII",
    "REGC_NAME_ASCII",  # legacy
    "REGC_name_ascii",  # legacy
)

NAME_FIELD_CANDIDATES: Tuple[str, ...] = tuple(
    dict.fromkeys((*NAME_FIELD_PRIORITY, *ASCII_FIELD_PRIORITY))
)

# -----------------------------
# Normalisation helpers
# -----------------------------

def _normalise_region_label(value: Any) -> str:
    """Lowercase, strip accents, normalise punctuation, drop the word 'region'."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\bregion\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _ascii_region_name(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


OFFICIAL_REGION_TITLES: Tuple[str, ...] = (
    "Northland Region",
    "Auckland Region",
    "Waikato Region",
    "Bay of Plenty Region",
    "Gisborne Region",
    "Hawke's Bay Region",
    "Taranaki Region",
    "Manawatū-Whanganui Region",
    "Wellington Region",
    "Tasman Region",
    "Nelson Region",
    "Marlborough Region",
    "West Coast Region",
    "Canterbury Region",
    "Otago Region",
    "Southland Region",
    "Area Outside Region",
)

OFFICIAL_REGION_NAMES: Dict[str, str] = {
    _normalise_region_label(name): name for name in OFFICIAL_REGION_TITLES
}

OFFICIAL_REGION_ASCII: Dict[str, str] = {
    key: _ascii_region_name(value) for key, value in OFFICIAL_REGION_NAMES.items()
}

# --- NEW: detect English tails in bilingual/council labels ---
_ENGLISH_REGION_RE = re.compile(
    r"([A-Za-z\u00C0-\u017F][A-Za-z\u00C0-\u017F\s'\-]*\bRegion)\b"
)
_COUNCIL_RE = re.compile(
    r"([A-Za-z\u00C0-\u017F][A-Za-z\u00C0-\u017F\s'\-]*)\bRegional Council\b",
    flags=re.IGNORECASE,
)


def _canonical_region_name(value: Any) -> Optional[str]:
    """
    Map arbitrary input to official 'X Region' label if possible.

    Handles:
      - plain English ('Northland Region')
      - bilingual ('Te Tai Tokerau / Northland Region')
      - council form ('Northland Regional Council')
      - minor punctuation/diacritics differences
    """
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    # Direct try with our normaliser
    norm = _normalise_region_label(raw)
    if norm:
        hit = OFFICIAL_REGION_NAMES.get(norm)
        if hit:
            return hit

    # Common oversight: caller forgot the 'Region' suffix
    fallback_norm = _normalise_region_label(f"{raw} Region")
    if fallback_norm and fallback_norm != norm:
        hit = OFFICIAL_REGION_NAMES.get(fallback_norm)
        if hit:
            return hit

    # NEW: bilingual names that contain an English tail ending with 'Region'
    # e.g. 'Te Tai Tokerau / Northland Region' -> 'Northland Region'
    m = _ENGLISH_REGION_RE.search(raw)
    if m:
        english_tail = m.group(1).strip()
        hit = OFFICIAL_REGION_NAMES.get(_normalise_region_label(english_tail))
        if hit:
            return hit

    # NEW: council form e.g. 'Northland Regional Council' -> 'Northland Region'
    m2 = _COUNCIL_RE.search(raw)
    if m2:
        guess = f"{m2.group(1).strip()} Region"
        hit = OFFICIAL_REGION_NAMES.get(_normalise_region_label(guess))
        if hit:
            return hit

    return None





def _canonical_join_key(value: Any) -> str:
    canonical = _canonical_region_name(value)
    if canonical:
        return canonical
    text = str(value).strip() if value is not None else ""
    return text


# -----------------------------
# GeoJSON helpers
# -----------------------------

# --- Lightweight coordinate rounding to shrink GeoJSON payload size ---

def _round_coords_inplace(coords: Any, decimals: int) -> Any:
    if isinstance(coords, (list, tuple)):
        out = []
        for item in coords:
            if isinstance(item, (list, tuple)):
                out.append(_round_coords_inplace(item, decimals))
            elif isinstance(item, (int, float)):
                out.append(round(float(item), decimals))
            else:
                out.append(item)
        return out
    return coords

def simplify_geojson_precision_inplace(geojson: Dict[str, Any], *, decimals: int = 5) -> None:
    """
    Round coordinate precision in-place to reduce payload size for client render.
    Keeps topology identical for visualization purposes.
    """
    if not isinstance(geojson, dict):
        return
    for feature in geojson.get("features", []):
        geom = feature.get("geometry")
        if isinstance(geom, dict) and "coordinates" in geom:
            geom["coordinates"] = _round_coords_inplace(geom["coordinates"], decimals)


def _resolve_geojson_name_fields(geojson: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Pick the best available name fields present in the payload."""
    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        if not props:
            continue
        name_field = next(
            (field for field in NAME_FIELD_PRIORITY if field in props and props[field]),
            None,
        )
        ascii_field = next(
            (field for field in ASCII_FIELD_PRIORITY if field in props and props[field]),
            None,
        )
        if name_field:
            return name_field, ascii_field
    # Fallbacks
    fallback_name = "REGC2025_V1_00_NAME"
    fallback_ascii = "REGC2025_V1_00_NAME_ASCII"
    return fallback_name, fallback_ascii


def _ensure_official_geojson_fields(geojson: Dict[str, Any]) -> bool:
    """Inject/standardise official 2025 name fields in-place for robust joins."""
    changed = False
    if not isinstance(geojson, dict):
        return changed

    for feature in geojson.get("features", []):
        props = feature.get("properties")
        if not isinstance(props, dict):
            continue
        candidates: List[str] = []
        for field in NAME_FIELD_CANDIDATES:
            if not field:
                continue
            value = props.get(field)
            if value is None:
                continue
            text_value = str(value).strip()
            if not text_value:
                continue
            candidates.append(text_value)

        canonical = None
        for label in candidates:
            canonical = _canonical_region_name(label)
            if canonical:
                break
        if canonical is None and candidates:
            canonical = candidates[0].strip()
        if not canonical:
            continue

        ascii_value = _ascii_region_name(canonical)
        if props.get("REGC2025_V1_00_NAME") != canonical:
            props["REGC2025_V1_00_NAME"] = canonical
            changed = True
        if props.get("REGC2025_V1_00_NAME_ASCII") != ascii_value:
            props["REGC2025_V1_00_NAME_ASCII"] = ascii_value
            changed = True
    return changed

def _geojson_has_official_field(geojson: Dict[str, Any]) -> bool:
    if not isinstance(geojson, dict):
        return False
    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        if not props.get("REGC2025_V1_00_NAME"):
            return False
    return True


def _iter_lonlat_pairs(coords: Any):
    """Yield (lon, lat) pairs from a GeoJSON coordinate structure."""
    if isinstance(coords, (list, tuple)):
        if coords and isinstance(coords[0], (int, float)) and len(coords) >= 2:
            yield float(coords[0]), float(coords[1])
        else:
            for item in coords:
                yield from _iter_lonlat_pairs(item)


def _geojson_is_lonlat(geojson: Dict[str, Any], *, sample_limit: int = 50) -> bool:
    """Return True when coordinates appear to be lon/lat (EPSG:4326)."""
    if not isinstance(geojson, dict):
        return False
    sampled = 0
    for feature in geojson.get("features", []):
        geom = feature.get("geometry")
        if not isinstance(geom, dict):
            continue
        coords = geom.get("coordinates")
        if coords is None:
            continue
        for lon, lat in _iter_lonlat_pairs(coords):
            if not (np.isfinite(lon) and np.isfinite(lat)):
                continue
            if abs(lon) > 180.0 or abs(lat) > 90.0:
                return False
            sampled += 1
            if sampled >= sample_limit:
                return True
    return True

def audit_region_coverage(mapping_df: pd.DataFrame) -> None:
    """Print any regional gaps between the GeoJSON and the project mapping."""
    if mapping_df is None or mapping_df.empty:
        print(">> Mapping is empty; cannot audit coverage")
        return
    geojson = fetch_region_geojson()
    geo_regions = {
        _canonical_region_name(
            (feature.get("properties") or {}).get("REGC2025_V1_00_NAME")
            or (feature.get("properties") or {}).get("REGC_name")
        )
        for feature in geojson.get("features", [])
    } if geojson else set()
    geo_regions = {name for name in geo_regions if name}
    mapping_regions = {
        _canonical_region_name(value)
        for value in mapping_df.get("region", pd.Series(dtype=object)).dropna().tolist()
    }
    mapping_regions = {name for name in mapping_regions if name}
    missing_in_mapping = sorted(geo_regions - mapping_regions)
    missing_in_geojson = sorted(mapping_regions - geo_regions)
    print(">> Missing in mapping (present in GeoJSON):", missing_in_mapping)
    print(">> Missing in GeoJSON (present in mapping):", missing_in_geojson)



def preview_join_status(region_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame showing which GeoJSON regions are missing from region_df."""
    geojson = fetch_region_geojson()
    feat_names = []
    if geojson:
        for feature in geojson.get("features", []):
            props = feature.get("properties") or {}
            name = _canonical_region_name(
                props.get("REGC2025_V1_00_NAME") or props.get("REGC_name")
            )
            if name:
                feat_names.append(name)
    want = pd.DataFrame({"region": sorted(set(feat_names))})
    got = region_df[["region"]].drop_duplicates() if not region_df.empty else pd.DataFrame({"region": []})
    merged = want.merge(got, on="region", how="left", indicator=True)
    return merged.assign(missing=lambda d: d["_merge"].eq("left_only")).drop(columns=["_merge"])




def _geojson_name_lookup(geojson: Dict[str, Any]) -> Dict[str, str]:
    """Build a normalised name → canonical name lookup from the GeoJSON."""
    _ensure_official_geojson_fields(geojson)
    name_field, ascii_field = _resolve_geojson_name_fields(geojson)
    candidate_fields = [field for field in NAME_FIELD_CANDIDATES if field]
    lookup: Dict[str, str] = {}

    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        if not props:
            continue

        canonical_raw = props.get(name_field)
        canonical = _canonical_region_name(canonical_raw) if canonical_raw is not None else None
        if not canonical and ascii_field:
            ascii_raw = props.get(ascii_field)
            canonical = _canonical_region_name(ascii_raw) if ascii_raw is not None else None

        candidates: List[str] = []
        for field in candidate_fields:
            value = props.get(field)
            if value is None:
                continue
            text_value = str(value).strip()
            if not text_value:
                continue
            candidates.append(text_value)
            if not canonical:
                candidate_canonical = _canonical_region_name(text_value)
                if candidate_canonical:
                    canonical = candidate_canonical

        if not canonical and candidates:
            canonical = candidates[0].strip()
        if not canonical:
            continue

        canonical_text = str(canonical).strip()
        if not canonical_text:
            continue

        # Always index the canonical and its ASCII variant
        ascii_alias = _ascii_region_name(canonical_text)
        lookup.setdefault(_normalise_region_label(canonical_text), canonical_text)
        if ascii_alias:
            lookup.setdefault(_normalise_region_label(ascii_alias), canonical_text)

        # Index all candidate labels (bilingual etc) to the canonical
        for label in candidates:
            norm = _normalise_region_label(label)
            if norm:
                lookup.setdefault(norm, canonical_text)

        # NEW: also index the plain-English tail '... Region' if present
        m_tail = _ENGLISH_REGION_RE.search(canonical_text)
        if m_tail:
            tail = m_tail.group(1).strip()
            lookup.setdefault(_normalise_region_label(tail), canonical_text)

        # NEW: and the 'Regional Council' variant as 'X Region'
        m_council = _COUNCIL_RE.search(canonical_text)
        if m_council:
            guess = f"{m_council.group(1).strip()} Region"
            lookup.setdefault(_normalise_region_label(guess), canonical_text)

    return lookup



def get_geojson_name_field(geojson: Dict[str, Any]) -> str:
    name_field, _ = _resolve_geojson_name_fields(geojson)
    return name_field


@lru_cache(maxsize=1)
def fetch_region_geojson(
    source: RegionGeometrySource = RegionGeometrySource(),
    local_path: Optional[Path] = GEOMETRY_LOCAL_PATH,
) -> Dict[str, Any]:
    """Fetch regional polygons as GeoJSON, caching to disk and memory."""
    lp: Optional[Path] = Path(local_path) if local_path is not None else None

    def _download(spatial_ref: Optional[int] = None) -> Dict[str, Any]:
        target_sr = spatial_ref if spatial_ref is not None else source.spatial_ref or 4326
        params = {
            "where": source.where,
            "outFields": ",".join(source.out_fields),
            "returnGeometry": "true",
            "f": "geojson",
            "outSR": str(target_sr),
            "resultRecordCount": "2000",
        }
        r = requests.get(source.url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        _ensure_official_geojson_fields(data)
        # ↓ New: shrink payload written to disk + sent to browser
        simplify_geojson_precision_inplace(data, decimals=5)
        return data

    geojson: Optional[Dict[str, Any]] = None

    # Try local cache first
    if lp is not None and lp.exists():
        with lp.open("r", encoding="utf-8") as fh:
            geojson = json.load(fh)
        if geojson:
            changed = _ensure_official_geojson_fields(geojson)
            # ↓ New: ensure the cached file also uses simplified precision
            simplify_geojson_precision_inplace(geojson, decimals=5)
            if not _geojson_has_official_field(geojson):
                geojson = _download(4326)
                if lp is not None and geojson:
                    try:
                        lp.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
                    except Exception:
                        pass
                return geojson
            if not _geojson_is_lonlat(geojson):
                geojson = _download(4326)
                if lp is not None and geojson:
                    try:
                        lp.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
                    except Exception:
                        pass
                return geojson
            if changed:
                try:
                    lp.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
            return geojson

    # Otherwise download
    geojson = _download(4326)
    if geojson and not _geojson_is_lonlat(geojson):
        return geojson
    if lp is not None and geojson:
        try:
            lp.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return geojson


# -----------------------------
# Mapping normalisation
# -----------------------------

def _standardise_mapping_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename common variants from the user PKL/CSV to canonical snake_case:
    Project, Region, Join key, GDP per capita, Population
      -> project, region, join_key, gdp_per_capita, population
    """
    if df is None or df.empty:
        return df

    def norm_col(c: str) -> str:
        return re.sub(r"\s+", " ", c.strip().lower())

    rename_map: Dict[str, str] = {}
    for c in df.columns:
        lc = norm_col(str(c))
        if lc in {"project", "project id", "code"}:
            rename_map[c] = "project"
        elif lc in {"region", "region name"}:
            rename_map[c] = "region"
        elif lc in {"join key", "join_key", "joinkey", "join region", "join"}:
            rename_map[c] = "join_key"
        elif "gdp" in lc and "capita" in lc:
            rename_map[c] = "gdp_per_capita"
        elif lc in {"population", "pop"}:
            rename_map[c] = "population"

    out = df.rename(columns=rename_map).copy()

    # Ensure required columns exist
    if "region" not in out.columns:
        raise KeyError("Mapping must contain a 'Region' column (or equivalent).")
    if "project" not in out.columns:
        raise KeyError("Mapping must contain a 'Project' column (or equivalent).")
    if "join_key" not in out.columns:
        # If no explicit join key, default to region label
        out["join_key"] = out["region"]

    # Coerce numerics where present
    if "gdp_per_capita" in out.columns:
        out["gdp_per_capita"] = pd.to_numeric(out["gdp_per_capita"], errors="coerce")
    else:
        out["gdp_per_capita"] = np.nan

    if "population" in out.columns:
        out["population"] = pd.to_numeric(out["population"], errors="coerce")
    else:
        out["population"] = np.nan

    # Normalised forms for joins
    out["project_norm"] = _normalise_project(out["project"])
    out["join_key"] = out["join_key"].astype(str).str.strip()
    out["region"] = out["region"].astype(str).str.strip()
    out["join_key_norm"] = out["join_key"].map(_normalise_region_label)

    return out


@lru_cache(maxsize=1)
def load_region_mapping(path: Optional[Path] = None) -> pd.DataFrame:
    raw = load_project_region_mapping(path)
    return _harmonise_join_keys(raw)


def _harmonise_join_keys(mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise mapping columns; align join_key/region to official names using:
      1) GeoJSON lookup (if available)
      2) OFFICIAL_REGION_NAMES (normalized)
      3) Canonicalization fallback by appending 'Region'
    Also guarantees presence of: project_norm, join_key_norm
    """
    if mapping is None or mapping.empty:
        return mapping

    aligned = _standardise_mapping_columns(mapping)
    aligned["join_key"] = aligned["join_key"].map(_canonical_join_key)
    aligned["region"] = aligned["region"].map(_canonical_join_key)
    geojson = fetch_region_geojson()
    lookup = _geojson_name_lookup(geojson) if geojson else {}

    norm_join = aligned["join_key"].map(_normalise_region_label)
    norm_region = aligned["region"].map(_normalise_region_label)

    resolved = norm_join.map(lookup).fillna(norm_region.map(lookup))
    resolved = resolved.fillna(norm_join.map(OFFICIAL_REGION_NAMES))
    resolved = resolved.fillna(norm_region.map(OFFICIAL_REGION_NAMES))

    fallback = aligned["join_key"].map(_canonical_region_name).fillna(
        aligned["region"].map(_canonical_region_name)
    )
    resolved = resolved.fillna(fallback)

    # Apply canonical names where found
    mask = resolved.notna()
    aligned.loc[mask, "join_key"] = resolved[mask]
    aligned.loc[mask, "region"] = resolved[mask]

    aligned["join_key_norm"] = aligned["join_key"].map(_normalise_region_label)

    # Keep a single row per project if duplicates appear; last-one-wins is fine
    aligned = aligned.copy()

    return aligned


# -----------------------------
# Metrics
# -----------------------------

def region_baselines(mapping: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    catalog = mapping[["region", "join_key", "population", "gdp_per_capita"]].drop_duplicates(subset=["join_key"]).copy()
    catalog["population"] = pd.to_numeric(catalog["population"], errors="coerce").fillna(0.0)
    catalog["gdp_per_capita"] = pd.to_numeric(catalog["gdp_per_capita"], errors="coerce").fillna(0.0)

    pop_total = catalog["population"].sum()
    pop_share = {row.region: (row.population / pop_total) if pop_total > 0 else np.nan
                 for row in catalog.itertuples()}

    gdp_mass = catalog["population"] * catalog["gdp_per_capita"]
    gdp_total = gdp_mass.sum()
    gdp_share = {row.region: (row.population * row.gdp_per_capita / gdp_total) if gdp_total > 0 else np.nan
                 for row in catalog.itertuples()}

    return catalog, pop_share, gdp_share


def _safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace({0: np.nan})
    return num.div(denom)


def _prepare_region_info(mapping: pd.DataFrame) -> pd.DataFrame:
    cols = ["region", "join_key", "join_key_norm", "gdp_per_capita", "population"]
    missing = [c for c in cols if c not in mapping.columns]
    if missing:
        raise KeyError(f"Mapping missing required columns after standardisation: {missing}")
    info = mapping[cols].drop_duplicates(subset=["join_key"]).copy()
    info["population"] = pd.to_numeric(info["population"], errors="coerce")
    info["gdp_per_capita"] = pd.to_numeric(info["gdp_per_capita"], errors="coerce")
    return info


def _project_population_years(
    region_info: pd.DataFrame,
    years: Sequence[int],
    growth_rate: float = DEFAULT_ANNUAL_POP_GROWTH,
) -> pd.DataFrame:
    """Return projected population per region/year using a constant growth rate."""
    if region_info is None or region_info.empty:
        return pd.DataFrame(columns=["Year", "region", "population"])
    if not years:
        return pd.DataFrame(columns=["Year", "region", "population"])
    years_sorted = sorted(int(y) for y in years)
    if not years_sorted:
        return pd.DataFrame(columns=["Year", "region", "population"])
    base_year = years_sorted[0]
    base_df = (
        region_info[["region", "population"]]
        .drop_duplicates(subset=["region"])
        .copy()
    )
    base_series = pd.to_numeric(base_df["population"], errors="coerce")
    base_series.index = base_df["region"].tolist()
    records = []
    for region, base_value in base_series.items():
        try:
            base_float = float(base_value)
        except (TypeError, ValueError):
            base_float = np.nan
        for year in years_sorted:
            if not np.isfinite(base_float):
                projected_value = np.nan
            else:
                projected_value = base_float * pow(1.0 + float(growth_rate), year - base_year)
            records.append((year, region, projected_value))
    if not records:
        return pd.DataFrame(columns=["Year", "region", "population"])
    projected = pd.DataFrame(records, columns=["Year", "region", "population"])
    projected["Year"] = projected["Year"].astype(int)
    return projected


def _normalise_project(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )



def _benefit_scenario_sheet(meta: Optional[Dict[str, Any]]) -> Optional[str]:
    if not meta:
        return None
    steep = str(meta.get("BenSteep", "" )).strip()
    horizon = meta.get("BenHorizon")
    if not steep or horizon in (None, ""):
        return None
    try:
        horizon_int = int(horizon)
    except (TypeError, ValueError):
        return None
    key = f"{steep.upper()}{horizon_int}"
    return BENEFIT_SCENARIOS.get(key)


@lru_cache(maxsize=16)
def _load_benefit_table(sheet_name: str) -> pd.DataFrame:
    if not sheet_name:
        return pd.DataFrame()
    try:
        return pd.read_excel(SCORING_WORKBOOK, sheet_name=sheet_name, engine="openpyxl")
    except Exception:
        return pd.DataFrame()


def _total_benefit_matrix_from_dim(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Construct a project x year table using total (or summed) benefit flows."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    working = df.copy()
    if isinstance(working.index, pd.MultiIndex):
        working = working.reset_index()
    if "Project" not in working.columns:
        return None
    year_candidates: List[Any] = []
    for col in working.columns:
        if col in {"Project", "project", "Dimension"}:
            continue
        if isinstance(col, (int, np.integer)):
            year_candidates.append(col)
            continue
        try:
            if str(col).strip().isdigit():
                year_candidates.append(col)
        except Exception:
            continue
    if not year_candidates:
        return None
    if "Dimension" in working.columns:
        dim_series = working["Dimension"].astype(str).str.strip()
        total_mask = dim_series.str.lower() == "total"
        if total_mask.any():
            working = working[total_mask].copy()
    rename_map: Dict[Any, int] = {}
    for col in year_candidates:
        try:
            rename_map[col] = int(str(col))
        except (TypeError, ValueError):
            continue
    if not rename_map:
        return None
    working = working.rename(columns=rename_map)
    value_cols = list(rename_map.values())
    working[value_cols] = working[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    grouped = working.groupby("Project", as_index=True)[value_cols].sum()
    grouped.index = grouped.index.map(lambda x: str(x).strip())
    grouped = grouped.loc[:, sorted(grouped.columns)]
    return grouped


def _benefit_region_from_raw_result(
    raw_result: Dict[str, Any],
    mapping_df: pd.DataFrame,
    years: List[int],
    regions: List[str],
) -> Optional[pd.DataFrame]:
    benefit_matrix = raw_result.get("benefit_by_project_total")
    if not isinstance(benefit_matrix, pd.DataFrame) or benefit_matrix.empty:
        benefit_matrix = _total_benefit_matrix_from_dim(
            raw_result.get("benefits_by_project_dimension_by_year")
        )
    if benefit_matrix is None or benefit_matrix.empty:
        return None

    df = benefit_matrix.copy()
    df.index = df.index.map(lambda x: str(x).strip())
    drop_mask = df.index.str.lower().isin({"total", "total benefit"})
    df = df[~drop_mask]
    if df.empty:
        return None

    column_years: Dict[Any, int] = {}
    for col in df.columns:
        try:
            column_years[col] = int(str(col))
        except (TypeError, ValueError):
            continue
    if not column_years:
        return None

    df = df[list(column_years.keys())]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df.rename(columns=column_years, inplace=True)

    df = df.reset_index().rename(columns={"index": "Project"})
    df["project_norm"] = _normalise_project(df["Project"])
    year_cols = sorted(column_years.values())
    long = df.melt(id_vars=["project_norm"], value_vars=year_cols, var_name="Year", value_name="Benefit_Year")
    long["Year"] = pd.to_numeric(long["Year"], errors="coerce").astype("Int64")
    long = long[long["Year"].notna()].copy()
    if long.empty:
        return None
    long["Year"] = long["Year"].astype(int)

    valid_years = {int(y) for y in years}
    long = long[long["Year"].isin(valid_years)]
    if long.empty:
        return None

    mapping_norm = mapping_df[["project_norm", "region"]].drop_duplicates()
    benefit_proj = long.merge(mapping_norm, on="project_norm", how="left")
    benefit_proj["region"] = benefit_proj["region"].fillna("Unmapped")

    region_list = list(regions)
    if "Unmapped" in benefit_proj["region"].values and "Unmapped" not in region_list:
        region_list.append("Unmapped")

    benefit_region = benefit_proj.groupby(["Year", "region"], as_index=False)["Benefit_Year"].sum()
    full_index = pd.MultiIndex.from_product([sorted(valid_years), region_list], names=["Year", "region"])
    benefit_region = (
        benefit_region
        .set_index(["Year", "region"])
        .reindex(full_index, fill_value=0.0)
        .reset_index()
        .sort_values(["region", "Year"])
    )

    total_by_year = benefit_region.groupby("Year")["Benefit_Year"].sum().sort_index().rename("Benefit_National")
    benefit_region = benefit_region.merge(total_by_year, on="Year", how="left")
    benefit_region["BenefitShare_Year"] = _safe_divide(
        benefit_region["Benefit_Year"], benefit_region["Benefit_National"]
    ).fillna(0.0)
    benefit_region["Benefit_Cum_Region"] = benefit_region.groupby("region")["Benefit_Year"].cumsum()
    benefit_region["Benefit_Cum_National"] = benefit_region["Year"].map(total_by_year.cumsum())
    benefit_region["BenefitShare_Cum"] = _safe_divide(
        benefit_region["Benefit_Cum_Region"], benefit_region["Benefit_Cum_National"]
    ).fillna(0.0)
    return benefit_region

def _extract_total_benefit_map(benefit_df: pd.DataFrame) -> Dict[str, List[float]]:
    if benefit_df is None or benefit_df.empty or "Project" not in benefit_df.columns:
        return {}
    df = benefit_df.copy()
    df["project_norm"] = _normalise_project(df["Project"])
    if "Dimension" in df.columns:
        df["_dim_norm"] = df["Dimension"].astype(str).str.strip().str.lower()
    else:
        df["_dim_norm"] = "total"
    tcols = [
        c
        for c in df.columns
        if re.fullmatch(r"[tT]\s*\+\s*(\d+)", str(c).strip())
    ]
    if not tcols:
        return {}
    for c in tcols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    total_mask = df["_dim_norm"] == "total"
    if total_mask.any():
        total_df = df[total_mask].groupby("project_norm", as_index=False)[tcols].sum()
    else:
        total_df = df.groupby("project_norm", as_index=False)[tcols].sum()
    benefit_map: Dict[str, List[float]] = {}
    for row in total_df.itertuples(index=False):
        proj = getattr(row, "project_norm", "")
        if not proj:
            continue
        benefit_map[proj] = [float(getattr(row, c, 0.0)) for c in tcols]
    return benefit_map


def _compute_region_benefit_metrics(
    data: DashboardData,
    scenario_code: str,
    mapping_df: pd.DataFrame,
    years: List[int],
    regions: List[str],
) -> Optional[pd.DataFrame]:
    meta = getattr(data, "scenario_meta_by_code", {}).get(scenario_code)
    sheet_name = _benefit_scenario_sheet(meta)
    if not sheet_name:
        return None
    benefit_source = _load_benefit_table(sheet_name)
    benefit_map = _extract_total_benefit_map(benefit_source)
    if not benefit_map:
        return None
    schedule_df = data.schedule[data.schedule["Code"] == scenario_code]
    if schedule_df.empty or "StartFY" not in schedule_df.columns:
        return None
    schedule_df = schedule_df.copy()
    schedule_df["project_norm"] = _normalise_project(schedule_df["Project"])
    schedule_df["StartFY"] = pd.to_numeric(schedule_df["StartFY"], errors="coerce").astype("Int64")
    schedule_df = schedule_df[schedule_df["StartFY"].notna()].copy()
    if schedule_df.empty:
        return None
    schedule_df["StartFY"] = schedule_df["StartFY"].astype(int)
    year_set = {int(y) for y in years}
    records: List[Tuple[int, str, float]] = []
    for row in schedule_df.itertuples(index=False):
        proj = getattr(row, "project_norm", "")
        flows = benefit_map.get(proj)
        if not flows:
            continue
        start_fy = int(getattr(row, "StartFY"))
        for offset, value in enumerate(flows):
            if not value:
                continue
            year = start_fy + offset
            if year not in year_set:
                continue
            records.append((year, proj, float(value)))
    if not records:
        return None
    benefit_proj = pd.DataFrame(records, columns=["Year", "project_norm", "Benefit_Year"])
    if "project_norm" not in mapping_df.columns:
        mapping_df = mapping_df.copy()
        mapping_df["project_norm"] = _normalise_project(mapping_df["project"])
    region_lookup = mapping_df[["project_norm", "region"]].drop_duplicates()
    benefit_proj = benefit_proj.merge(region_lookup, on="project_norm", how="left")
    benefit_proj["region"] = benefit_proj["region"].fillna("Unmapped")
    region_list = list(regions)
    if "Unmapped" in benefit_proj["region"].values and "Unmapped" not in region_list:
        region_list.append("Unmapped")
    benefit_region = (
        benefit_proj.groupby(["Year", "region"], as_index=False)["Benefit_Year"].sum()
    )
    full_index = pd.MultiIndex.from_product([years, region_list], names=["Year", "region"])
    benefit_region = (
        benefit_region
        .set_index(["Year", "region"])
        .reindex(full_index, fill_value=0.0)
        .reset_index()
        .sort_values(["region", "Year"])
    )
    total_by_year = (
        benefit_region.groupby("Year")["Benefit_Year"].sum().sort_index().rename("Benefit_National")
    )
    benefit_region = benefit_region.merge(total_by_year, on="Year", how="left")
    benefit_region["BenefitShare_Year"] = _safe_divide(
        benefit_region["Benefit_Year"], benefit_region["Benefit_National"]
    ).fillna(0.0)
    benefit_region["Benefit_Cum_Region"] = benefit_region.groupby("region")["Benefit_Year"].cumsum()
    benefit_region["Benefit_Cum_National"] = benefit_region["Year"].map(total_by_year.cumsum())
    benefit_region["BenefitShare_Cum"] = _safe_divide(
        benefit_region["Benefit_Cum_Region"], benefit_region["Benefit_Cum_National"]
    ).fillna(0.0)
    return benefit_region
def compute_region_metrics(
    data: DashboardData,
    scenario_code: str,
    *,
    mapping: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute annual spend/benefit metrics per region for a scenario."""
    if scenario_code is None:
        raise ValueError("scenario_code must be provided")

    mapping_df = mapping if mapping is not None else load_region_mapping()
    if mapping_df is None or mapping_df.empty:
        raise ValueError("Project-region mapping is empty")

    mapping_df = mapping_df.copy()
    if "project_norm" not in mapping_df.columns:
        mapping_df["project_norm"] = _normalise_project(mapping_df["project"])

    region_info = _prepare_region_info(mapping_df)
    
    geojson = fetch_region_geojson()
    geo_regions = sorted({
        _canonical_region_name(
            (feature.get("properties") or {}).get("REGC2025_V1_00_NAME")
            or (feature.get("properties") or {}).get("REGC_name")
        )
        for feature in (geojson.get("features", []) if geojson else [])
    })
    geo_regions = [name for name in geo_regions if name]
    existing_regions = set(region_info["region"].tolist())
    all_region_names = sorted(existing_regions.union(geo_regions))
    missing_regions = [name for name in all_region_names if name not in existing_regions]
    if missing_regions:
        filler = pd.DataFrame(
            {
                "region": missing_regions,
                "join_key": missing_regions,
                "join_key_norm": [_normalise_region_label(name) for name in missing_regions],
                "gdp_per_capita": np.nan,
                "population": np.nan,
            }
        )
        region_info = pd.concat([region_info, filler], ignore_index=True)
    
    pop_total = region_info["population"].sum(skipna=True)
    gdp_total = (region_info["gdp_per_capita"] * region_info["population"]).sum(skipna=True)
    
    spend_df = data.spend_matrix[data.spend_matrix["Code"] == scenario_code]
    if spend_df.empty:
        raise ValueError(f"No spend matrix data for scenario code {scenario_code}")

    candidate_years = list(data.years)
    value_vars = [y for y in candidate_years if y in spend_df.columns]

    long_df = spend_df.melt(
        id_vars=["Code", "Project"],
        value_vars=value_vars,
        var_name="Year",
        value_name="Spend_M",
    )
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce").astype("Int64")
    long_df = long_df[long_df["Year"].notna()].copy()
    long_df["Year"] = long_df["Year"].astype(int)

    long_df["Spend_M"] = pd.to_numeric(long_df["Spend_M"], errors="coerce").fillna(0.0)

    long_df["project_norm"] = _normalise_project(long_df["Project"])
    if "project_norm" not in mapping_df.columns:
        mapping_df = mapping_df.copy()
        mapping_df["project_norm"] = _normalise_project(mapping_df["project"])

    merged = long_df.merge(
        mapping_df,
        how="left",
        on="project_norm",
        suffixes=("", "_map"),
    )

    merged["region"] = merged["region"].fillna("Unmapped")
    merged.loc[merged["region"] == "Unmapped", ["join_key", "join_key_norm"]] = ""
    merged.loc[merged["region"] == "Unmapped", ["population", "gdp_per_capita"]] = np.nan

    region_spend = merged.groupby(["Year", "region"], as_index=False)["Spend_M"].sum()

    if (merged["region"] == "Unmapped").any() and "Unmapped" not in region_info["region"].values:
        extra = pd.DataFrame(
            {
                "region": ["Unmapped"],
                "join_key": [""],
                "join_key_norm": [""],
                "gdp_per_capita": [0.0],
                "population": [0.0],
            }
        )
        region_info = pd.concat([region_info, extra], ignore_index=True)

    years_present = sorted(region_spend["Year"].unique().tolist())
    all_regions = sorted(region_info["region"].unique().tolist())
    full_index = pd.MultiIndex.from_product([years_present, all_regions], names=["Year", "region"])
    region_spend = (
        region_spend
        .set_index(["Year", "region"])
        .reindex(full_index, fill_value=0.0)
        .reset_index()
    )

    region_spend = region_spend.merge(
        region_info,
        how="left",
        on="region",
        suffixes=("", "_info"),
    )

    projected_pop = _project_population_years(region_info, years_present)
    if not projected_pop.empty:
        region_spend = region_spend.merge(
            projected_pop,
            on=["Year", "region"],
            how="left",
            suffixes=("", "_proj"),
        )
        if "population_proj" in region_spend.columns:
            region_spend["population_proj"] = pd.to_numeric(
                region_spend["population_proj"],
                errors="coerce",
            )
            region_spend["population"] = region_spend["population_proj"].combine_first(
                region_spend["population"]
            )
            region_spend.drop(columns=["population_proj"], inplace=True)

    region_spend["population"] = pd.to_numeric(region_spend["population"], errors="coerce")

    total_by_year = region_spend.groupby("Year")["Spend_M"].sum().rename("Spend_National")
    total_cum = total_by_year.sort_index().cumsum()

    region_spend = region_spend.merge(total_by_year, on="Year", how="left")
    region_spend["Share_Year"] = _safe_divide(
        region_spend["Spend_M"], region_spend["Spend_National"]
    ).fillna(0.0)

    region_spend = region_spend.sort_values(["region", "Year"])
    region_spend["Spend_Cum_Region"] = region_spend.groupby("region")["Spend_M"].cumsum()
    region_spend["Spend_Cum_National"] = region_spend["Year"].map(total_cum)
    region_spend["Share_Cum"] = _safe_divide(
        region_spend["Spend_Cum_Region"], region_spend["Spend_Cum_National"]
    ).fillna(0.0)

    region_spend["PerCap_Year"] = _safe_divide(
        region_spend["Spend_M"], region_spend["population"]
    )
    region_spend["PerCap_Cum"] = _safe_divide(
        region_spend["Spend_Cum_Region"], region_spend["population"]
    )

    pop_share_series = region_info.set_index("region")["population"]
    pop_share_series = pop_share_series / pop_total if pop_total > 0 else pd.Series(dtype=float)

    gdp_mass = region_info.set_index("region")["gdp_per_capita"] * region_info.set_index("region")["population"]
    gdp_share_series = gdp_mass / gdp_total if gdp_total > 0 else pd.Series(dtype=float)

    pop_share_map = pop_share_series.to_dict() if not pop_share_series.empty else {}
    gdp_share_map = gdp_share_series.to_dict() if not gdp_share_series.empty else {}

    region_spend["Pop_Share_Benchmark"] = region_spend["region"].map(pop_share_map)
    region_spend["GDP_Share_Benchmark"] = region_spend["region"].map(gdp_share_map)

    region_spend["OU_vs_Pop"] = region_spend["Share_Cum"] - region_spend["Pop_Share_Benchmark"]
    region_spend["OU_vs_GDP"] = region_spend["Share_Cum"] - region_spend["GDP_Share_Benchmark"]

    region_spend["Ramp_Rate"] = region_spend.groupby("region")["Share_Cum"].diff()
    region_spend["Ramp_Rate"] = region_spend["Ramp_Rate"].fillna(region_spend["Share_Cum"])

    benefit_cols = [
        "Benefit_Year",
        "Benefit_National",
        "Benefit_Cum_Region",
        "Benefit_Cum_National",
        "BenefitShare_Year",
        "BenefitShare_Cum",
    ]
    benefit_frame = None
    scenario_meta_lookup = getattr(data, "scenario_meta_by_code", {})
    meta = scenario_meta_lookup.get(scenario_code)
    raw_results = getattr(data, "raw_results", {})
    if meta:
        stem = meta.get("_stem")
        if stem:
            raw_result = raw_results.get(stem)
            if raw_result is not None:
                benefit_frame = _benefit_region_from_raw_result(
                    raw_result,
                    mapping_df,
                    years_present,
                    all_regions,
                )
    if benefit_frame is None:
        benefit_frame = _compute_region_benefit_metrics(
            data,
            scenario_code,
            mapping_df,
            years_present,
            all_regions,
        )
    if benefit_frame is not None:
        region_spend = region_spend.merge(
            benefit_frame[["Year", "region"] + benefit_cols],
            on=["Year", "region"],
            how="left",
        )
    else:
        for col in benefit_cols:
            region_spend[col] = 0.0
    for col in benefit_cols:
        region_spend[col] = pd.to_numeric(region_spend[col], errors="coerce").fillna(0.0)

    region_spend.rename(
        columns={
            "Spend_M": "Spend_Year",
        },
        inplace=True,
    )

    return region_spend[
        [
            "Year",
            "region",
            "join_key",
            "Spend_Year",
            "Spend_National",
            "Spend_Cum_Region",
            "Spend_Cum_National",
            "Share_Year",
            "Share_Cum",
            "PerCap_Year",
            "PerCap_Cum",
            "Pop_Share_Benchmark",
            "GDP_Share_Benchmark",
            "OU_vs_Pop",
            "OU_vs_GDP",
            "Ramp_Rate",
            "Benefit_Year",
            "Benefit_National",
            "Benefit_Cum_Region",
            "Benefit_Cum_National",
            "BenefitShare_Year",
            "BenefitShare_Cum",
            "population",
            "gdp_per_capita",
        ]
    ]
