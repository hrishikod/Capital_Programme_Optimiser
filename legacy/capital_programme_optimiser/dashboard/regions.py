from __future__ import annotations

import json
import re
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
    from shapely.geometry.polygon import orient
    from shapely.ops import unary_union
    try:
        from shapely.validation import make_valid as _shapely_make_valid
    except ImportError:
        _shapely_make_valid = None
    _HAS_SHAPELY = True
except Exception:  # pragma: no cover - optional dependency
    GeometryCollection = MultiPolygon = Polygon = None  # type: ignore
    mapping = shape = orient = unary_union = None  # type: ignore
    _shapely_make_valid = None
    _HAS_SHAPELY = False

from ..config import load_project_region_mapping, load_settings
from .data import DashboardData

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
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ATTRIBUTE_WORKBOOK = REPO_ROOT / "Cost_benefit_streams.xlsx"

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

_REGION_ALIAS_MAP: Dict[str, str] = {
    _normalise_region_label("Northland"): "Northland Region",
    _normalise_region_label("Auckland"): "Auckland Region",
    _normalise_region_label("Waikato"): "Waikato Region",
    _normalise_region_label("Bay of Plenty"): "Bay of Plenty Region",
    _normalise_region_label("Gisborne"): "Gisborne Region",
    _normalise_region_label("Tairawhiti"): "Gisborne Region",
    _normalise_region_label("Tairāwhiti"): "Gisborne Region",
    _normalise_region_label("Tairwhiti"): "Gisborne Region",
    _normalise_region_label("Tairāwhiti (Gisborne)"): "Gisborne Region",
    _normalise_region_label("Tairwhiti (Gisborne)"): "Gisborne Region",
    _normalise_region_label("Hawkes Bay"): "Hawke's Bay Region",
    _normalise_region_label("Hawke's Bay"): "Hawke's Bay Region",
    _normalise_region_label("Taranaki"): "Taranaki Region",
    _normalise_region_label("Manawatu Whanganui"): "Manawatū-Whanganui Region",
    _normalise_region_label("Manawatū Whanganui"): "Manawatū-Whanganui Region",
    _normalise_region_label("Manawatū/Whanganui"): "Manawatū-Whanganui Region",
    _normalise_region_label("Manawatu/Whanganui"): "Manawatū-Whanganui Region",
    _normalise_region_label("ManawatWhanganui"): "Manawatū-Whanganui Region",
    _normalise_region_label("Wellington"): "Wellington Region",
    _normalise_region_label("Tasman"): "Tasman Region",
    _normalise_region_label("Nelson"): "Nelson Region",
    _normalise_region_label("Marlborough"): "Marlborough Region",
    _normalise_region_label("West Coast"): "West Coast Region",
    _normalise_region_label("Canterbury"): "Canterbury Region",
    _normalise_region_label("Otago"): "Otago Region",
    _normalise_region_label("Southland"): "Southland Region",
    _normalise_region_label("Area Outside"): "Area Outside Region",
}

AREA_OUTSIDE_REGION = "Area Outside Region"
UNMAPPED_REGION_LABEL = "Unmapped"
DISPLAY_REGION_TITLES: Tuple[str, ...] = tuple(
    name for name in OFFICIAL_REGION_TITLES if name != AREA_OUTSIDE_REGION
)
DISPLAY_REGION_SET = set(DISPLAY_REGION_TITLES)

# Region stats (year ended Mar 2024).
REGION_BASELINES_2024: Dict[str, Dict[str, float]] = {
    "Marlborough Region": {"gdp_per_capita": 84296.0, "population": 52300.0},
    "Southland Region": {"gdp_per_capita": 83620.0, "population": 106100.0},
    "Taranaki Region": {"gdp_per_capita": 85362.0, "population": 130800.0},
    "West Coast Region": {"gdp_per_capita": 75057.0, "population": 34800.0},
}

# Allocation weights to distribute National projects across regions.
NATIONAL_PROJECT_REGION_WEIGHTS: Dict[str, float] = {
    "Auckland Region": 0.336854922,
    "Bay of Plenty Region": 0.066535544,
    "Canterbury Region": 0.130073991,
    "Gisborne Region": 0.009984078,
    "Hawke's Bay Region": 0.034728856,
    "Manawatū-Whanganui Region": 0.04932097,
    "Nelson Region": 0.010339983,
    "Northland Region": 0.038362836,
    "Otago Region": 0.048178327,
    "Tasman Region": 0.011239112,
    "Waikato Region": 0.100440199,
    "Wellington Region": 0.103137585,
    "Marlborough Region": 0.009796759,
    "Southland Region": 0.019874497,
    "Taranaki Region": 0.024501264,
    "West Coast Region": 0.006518685,
}

_NATIONAL_WEIGHT_TOTAL = float(sum(NATIONAL_PROJECT_REGION_WEIGHTS.values()))
if _NATIONAL_WEIGHT_TOTAL > 0:
    NATIONAL_PROJECT_REGION_WEIGHTS = {
        region: float(weight) / _NATIONAL_WEIGHT_TOTAL
        for region, weight in NATIONAL_PROJECT_REGION_WEIGHTS.items()
    }


def _is_national_region_label(value: Any) -> bool:
    text = str(value).strip().lower() if value is not None else ""
    return text == "national"


def _is_unmapped_region_label(value: Any) -> bool:
    text = str(value).strip().lower() if value is not None else ""
    return text == UNMAPPED_REGION_LABEL.lower()

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
        alias_hit = _REGION_ALIAS_MAP.get(norm)
        if alias_hit:
            return alias_hit
        hit = OFFICIAL_REGION_NAMES.get(norm)
        if hit:
            return hit

    # Common oversight: caller forgot the 'Region' suffix
    fallback_norm = _normalise_region_label(f"{raw} Region")
    if fallback_norm and fallback_norm != norm:
        alias_hit = _REGION_ALIAS_MAP.get(fallback_norm)
        if alias_hit:
            return alias_hit
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


def _collect_polygon_parts(geom) -> List[Polygon]:
    """Return oriented polygon parts from any Shapely geometry."""
    parts: List[Polygon] = []
    if not _HAS_SHAPELY:
        return parts
    if geom is None or getattr(geom, "is_empty", True):
        return parts
    if isinstance(geom, Polygon):
        oriented = orient(geom, sign=1.0)
        if not oriented.is_empty:
            parts.append(oriented)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            parts.extend(_collect_polygon_parts(poly))
    elif isinstance(geom, GeometryCollection):
        for sub in geom.geoms:
            parts.extend(_collect_polygon_parts(sub))
    return parts


def _make_valid_oriented(geom):
    """Return a valid polygon/multipolygon with consistent ring orientation."""
    if not _HAS_SHAPELY:
        return None
    if geom is None or getattr(geom, "is_empty", True):
        return None
    candidate = geom
    if not candidate.is_valid:
        if _shapely_make_valid is not None:
            candidate = _shapely_make_valid(candidate)
        else:
            candidate = candidate.buffer(0)
        if candidate is None or getattr(candidate, "is_empty", True):
            return None
    parts = _collect_polygon_parts(candidate)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    merged = unary_union(parts)
    if isinstance(merged, Polygon):
        return orient(merged, sign=1.0)
    if isinstance(merged, MultiPolygon):
        return MultiPolygon([orient(poly, sign=1.0) for poly in merged.geoms if not poly.is_empty])
    # Fallback: stitch parts into a MultiPolygon
    return MultiPolygon(parts)


def _repair_geojson_geometry(geojson: Dict[str, Any]) -> bool:
    """Ensure GeoJSON polygons are valid and consistently oriented."""
    if not _HAS_SHAPELY or not isinstance(geojson, dict):
        return False
    changed = False
    for feature in geojson.get("features", []):
        geom = feature.get("geometry")
        if not isinstance(geom, dict):
            continue
        try:
            shaped = shape(geom)
        except Exception:
            continue
        repaired = _make_valid_oriented(shaped)
        if repaired is None or getattr(repaired, "is_empty", True):
            continue
        original_type = geom.get("type")
        if original_type == "MultiPolygon" and isinstance(repaired, Polygon):
            repaired = MultiPolygon([repaired])
        if not repaired.is_valid and _shapely_make_valid is not None:
            repaired_valid = _shapely_make_valid(repaired)
            if repaired_valid and not repaired_valid.is_empty:
                repaired = _make_valid_oriented(repaired_valid) or repaired
        try:
            repaired_mapping = mapping(repaired)
        except Exception:
            continue
        if repaired_mapping != geom:
            feature["geometry"] = repaired_mapping
            changed = True
    return changed


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
        _repair_geojson_geometry(data)
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
            geometry_fixed = _repair_geojson_geometry(geojson)
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
            if changed or geometry_fixed:
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


def _clean_mapping_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


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


def _read_workbook_region_sheet(workbook_path: Path, *, sheet_name: str) -> pd.DataFrame:
    df = _read_excel_safe(workbook_path, sheet_name=sheet_name)
    if df.empty:
        return pd.DataFrame(columns=["project", "region"])
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    lookup = {str(col).strip().lower(): col for col in df.columns}
    project_col = lookup.get("project")
    region_col = lookup.get("region")
    if project_col is None or region_col is None:
        return pd.DataFrame(columns=["project", "region"])

    out = pd.DataFrame()
    out["project"] = df[project_col].map(_clean_mapping_value)
    out["region"] = df[region_col].map(_clean_mapping_value)
    out = out.dropna(subset=["project"]).copy()
    out["project"] = out["project"].astype(str).str.strip()
    out["region"] = out["region"].fillna(UNMAPPED_REGION_LABEL).astype(str).str.strip()
    out = out[out["project"].ne("")]
    out["project_norm"] = _normalise_project(out["project"])
    out["region"] = out["region"].apply(
        lambda value: _canonical_region_name(value) or _canonical_join_key(value) or UNMAPPED_REGION_LABEL
    )
    out["join_key"] = out["region"].apply(_canonical_join_key)
    out["join_key_norm"] = out["join_key"].map(_normalise_region_label)
    return out[["project", "project_norm", "region", "join_key", "join_key_norm"]]


def _load_workbook_project_region_mapping(
    workbook_path: Optional[Path] = None,
) -> pd.DataFrame:
    workbook = Path(workbook_path) if workbook_path is not None else DEFAULT_ATTRIBUTE_WORKBOOK
    if not workbook.exists():
        return pd.DataFrame(
            columns=["project", "project_norm", "region", "join_key", "join_key_norm"]
        )

    costs = _read_workbook_region_sheet(workbook, sheet_name="Costs")
    benefits = _read_workbook_region_sheet(workbook, sheet_name="Benefits Linear 40yrs")
    if costs.empty and benefits.empty:
        return pd.DataFrame(
            columns=["project", "project_norm", "region", "join_key", "join_key_norm"]
        )

    if costs.empty:
        combined = benefits.copy()
    elif benefits.empty:
        combined = costs.copy()
    else:
        combined = (
            costs.set_index("project_norm")
            .combine_first(benefits.set_index("project_norm"))
            .reset_index()
        )
    if "project_norm" not in combined.columns:
        combined["project_norm"] = _normalise_project(combined.get("project"))
    combined = combined.drop_duplicates(subset=["project_norm"], keep="first")
    return combined


def _merge_project_mapping_sources(
    base_mapping: pd.DataFrame,
    workbook_mapping: pd.DataFrame,
) -> pd.DataFrame:
    if base_mapping is None or base_mapping.empty:
        return workbook_mapping.copy()
    if workbook_mapping is None or workbook_mapping.empty:
        return base_mapping.copy()

    baseline_regions = (
        base_mapping[["region", "join_key", "join_key_norm", "population", "gdp_per_capita"]]
        .dropna(subset=["region"])
        .drop_duplicates(subset=["region"], keep="first")
    )
    baseline_regions = baseline_regions.copy()
    baseline_regions["join_key_norm"] = baseline_regions["join_key_norm"].fillna(
        baseline_regions["join_key"].map(_normalise_region_label)
    )

    workbook = workbook_mapping.copy()
    workbook["region"] = workbook["region"].fillna(UNMAPPED_REGION_LABEL).astype(str).str.strip()
    workbook["join_key"] = workbook["join_key"].fillna(workbook["region"]).astype(str).str.strip()
    workbook["join_key_norm"] = workbook["join_key_norm"].fillna(
        workbook["join_key"].map(_normalise_region_label)
    )

    workbook = workbook.merge(
        baseline_regions[
            ["region", "join_key_norm", "population", "gdp_per_capita"]
        ].rename(
            columns={
                "region": "_baseline_region",
                "population": "_baseline_population",
                "gdp_per_capita": "_baseline_gdp_per_capita",
            }
        ),
        on="join_key_norm",
        how="left",
    )
    population_series = (
        workbook["population"]
        if "population" in workbook.columns
        else pd.Series(np.nan, index=workbook.index)
    )
    gdp_series = (
        workbook["gdp_per_capita"]
        if "gdp_per_capita" in workbook.columns
        else pd.Series(np.nan, index=workbook.index)
    )
    workbook["population"] = pd.to_numeric(population_series, errors="coerce").combine_first(
        pd.to_numeric(workbook.get("_baseline_population"), errors="coerce")
    )
    workbook["gdp_per_capita"] = pd.to_numeric(gdp_series, errors="coerce").combine_first(
        pd.to_numeric(workbook.get("_baseline_gdp_per_capita"), errors="coerce")
    )
    workbook["region"] = workbook["_baseline_region"].combine_first(workbook["region"])
    workbook.drop(
        columns=["_baseline_region", "_baseline_population", "_baseline_gdp_per_capita"],
        inplace=True,
        errors="ignore",
    )

    workbook = workbook.reindex(columns=base_mapping.columns, fill_value=np.nan)

    combined = pd.concat([base_mapping, workbook], ignore_index=True, sort=False)
    combined = _harmonise_join_keys(combined)
    return combined


@lru_cache(maxsize=8)
def load_region_mapping(
    path: Optional[Path] = None,
    *,
    workbook_path: Optional[Path] = None,
    include_workbook: bool = True,
) -> pd.DataFrame:
    raw = load_project_region_mapping(path)
    base_mapping = _harmonise_join_keys(raw)
    if not include_workbook:
        return base_mapping
    workbook_mapping = _load_workbook_project_region_mapping(workbook_path)
    return _merge_project_mapping_sources(base_mapping, workbook_mapping)


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

    aligned["population"] = pd.to_numeric(aligned["population"], errors="coerce")
    aligned["gdp_per_capita"] = pd.to_numeric(aligned["gdp_per_capita"], errors="coerce")

    # Fill missing population/GDP values from any other row mapped to the same canonical region.
    region_baseline = (
        aligned[["region", "population", "gdp_per_capita"]]
        .dropna(subset=["region"])
        .drop_duplicates(subset=["region"], keep="first")
        .set_index("region")
    )
    aligned["population"] = aligned["population"].combine_first(aligned["region"].map(region_baseline["population"]))
    aligned["gdp_per_capita"] = aligned["gdp_per_capita"].combine_first(
        aligned["region"].map(region_baseline["gdp_per_capita"])
    )

    # Keep a single row per project if duplicates appear; last-one-wins ensures workbook overrides baseline mapping.
    if "project_norm" in aligned.columns:
        aligned = aligned.drop_duplicates(subset=["project_norm"], keep="last").copy()
    else:
        aligned = aligned.copy()

    return aligned


# -----------------------------
# Metrics
# -----------------------------

def _baseline_2024(region_name: str) -> Dict[str, float]:
    return dict(REGION_BASELINES_2024.get(str(region_name).strip(), {}))


def _ensure_display_region_info(region_info: pd.DataFrame) -> pd.DataFrame:
    if region_info is None:
        return pd.DataFrame(columns=["region", "join_key", "join_key_norm", "gdp_per_capita", "population"])
    region_info = region_info.copy()
    region_info["region"] = region_info["region"].map(_canonical_join_key)
    region_info["join_key"] = region_info["join_key"].map(_canonical_join_key)

    if "join_key_norm" not in region_info.columns:
        region_info["join_key_norm"] = region_info["join_key"].map(_normalise_region_label)

    region_info = region_info[~region_info["region"].map(_is_national_region_label)].copy()
    region_info = region_info[region_info["region"].isin(DISPLAY_REGION_SET)].copy()

    region_info["population"] = pd.to_numeric(region_info["population"], errors="coerce")
    region_info["gdp_per_capita"] = pd.to_numeric(region_info["gdp_per_capita"], errors="coerce")

    for region_name, baseline in REGION_BASELINES_2024.items():
        mask = region_info["region"] == region_name
        if not mask.any():
            continue
        baseline_pop = float(baseline.get("population", np.nan))
        baseline_gdp = float(baseline.get("gdp_per_capita", np.nan))
        if np.isfinite(baseline_pop):
            region_info.loc[mask, "population"] = region_info.loc[mask, "population"].where(
                region_info.loc[mask, "population"].notna() & (region_info.loc[mask, "population"] > 0),
                baseline_pop,
            )
        if np.isfinite(baseline_gdp):
            region_info.loc[mask, "gdp_per_capita"] = region_info.loc[mask, "gdp_per_capita"].where(
                region_info.loc[mask, "gdp_per_capita"].notna() & (region_info.loc[mask, "gdp_per_capita"] > 0),
                baseline_gdp,
            )

    existing = set(region_info["region"].dropna().astype(str).tolist())
    missing = [name for name in DISPLAY_REGION_TITLES if name not in existing]
    if missing:
        filler = []
        for name in missing:
            baseline = _baseline_2024(name)
            filler.append(
                {
                    "region": name,
                    "join_key": name,
                    "join_key_norm": _normalise_region_label(name),
                    "gdp_per_capita": baseline.get("gdp_per_capita", np.nan),
                    "population": baseline.get("population", np.nan),
                }
            )
        region_info = pd.concat([region_info, pd.DataFrame(filler)], ignore_index=True)

    region_info["join_key_norm"] = region_info["join_key"].map(_normalise_region_label)
    return region_info


def _distribute_national_rows(
    df: pd.DataFrame,
    *,
    region_col: str,
    value_col: str,
    weights: Optional[Dict[str, float]] = None,
    spread_national: bool = True,
    spread_unmapped: bool = True,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    is_national = df[region_col].map(_is_national_region_label)
    is_unmapped = df[region_col].map(_is_unmapped_region_label)

    spread_mask = (is_national & bool(spread_national)) | (is_unmapped & bool(spread_unmapped))
    exclude_mask = (is_national & (not bool(spread_national))) | (is_unmapped & (not bool(spread_unmapped)))
    if not spread_mask.any() and not exclude_mask.any():
        return df

    base = df.loc[~(spread_mask | exclude_mask)].copy()
    if not spread_mask.any():
        return base

    weights_map = dict(weights or NATIONAL_PROJECT_REGION_WEIGHTS)
    weights_map = {k: float(v) for k, v in weights_map.items() if k in DISPLAY_REGION_SET and float(v) > 0}
    weight_total = float(sum(weights_map.values()))
    if weight_total <= 0:
        return pd.concat([base, df.loc[spread_mask].copy()], ignore_index=True)
    if abs(weight_total - 1.0) > 1e-8:
        weights_map = {k: v / weight_total for k, v in weights_map.items()}

    nat = df.loc[spread_mask].copy()

    nat[value_col] = pd.to_numeric(nat[value_col], errors="coerce").fillna(0.0)
    nat = nat[nat[value_col] != 0.0].copy()
    if nat.empty:
        return base

    nat["_weight_key"] = 1
    weight_df = pd.DataFrame(
        {
            "_target_region": list(weights_map.keys()),
            "_weight": list(weights_map.values()),
            "_weight_key": 1,
        }
    )
    expanded = nat.merge(weight_df, on="_weight_key", how="inner").drop(columns=["_weight_key"])
    expanded[region_col] = expanded["_target_region"]
    expanded[value_col] = expanded[value_col] * expanded["_weight"]
    expanded.drop(columns=["_weight", "_target_region"], inplace=True)

    return pd.concat([base, expanded], ignore_index=True)


def _apply_region_spread_policy(
    df: pd.DataFrame,
    *,
    year_col: str,
    region_col: str,
    value_col: str,
    spread_national: bool,
    spread_unmapped: bool,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        empty_diag = pd.DataFrame(
            columns=[
                year_col,
                "full_total",
                "source_national",
                "source_unmapped",
                "included_total",
                "excluded_national",
                "excluded_unmapped",
                "excluded_total",
            ]
        )
        return df, empty_diag

    working = df.copy()
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce").fillna(0.0)
    is_national = working[region_col].map(_is_national_region_label)
    is_unmapped = working[region_col].map(_is_unmapped_region_label)

    full_total = working.groupby(year_col)[value_col].sum()
    source_national = working.loc[is_national].groupby(year_col)[value_col].sum()
    source_unmapped = working.loc[is_unmapped].groupby(year_col)[value_col].sum()

    distributed = _distribute_national_rows(
        working,
        region_col=region_col,
        value_col=value_col,
        weights=weights,
        spread_national=spread_national,
        spread_unmapped=spread_unmapped,
    )
    included_total = distributed.groupby(year_col)[value_col].sum()

    diag = pd.DataFrame({"full_total": full_total})
    diag["source_national"] = source_national
    diag["source_unmapped"] = source_unmapped
    diag["included_total"] = included_total
    diag = diag.fillna(0.0)
    diag["excluded_national"] = diag["source_national"] if not spread_national else 0.0
    diag["excluded_unmapped"] = diag["source_unmapped"] if not spread_unmapped else 0.0
    diag["excluded_total"] = diag["excluded_national"] + diag["excluded_unmapped"]
    diag = diag.reset_index()
    return distributed, diag


def region_baselines(mapping: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    catalog = mapping[["region", "join_key", "population", "gdp_per_capita"]].drop_duplicates(subset=["join_key"]).copy()
    catalog["join_key_norm"] = catalog["join_key"].map(_normalise_region_label)
    catalog = _ensure_display_region_info(catalog)
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
    *,
    spread_national: bool,
    spread_unmapped: bool,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
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
    benefit_proj["region"] = benefit_proj["region"].fillna(UNMAPPED_REGION_LABEL).replace({"": UNMAPPED_REGION_LABEL})
    benefit_proj, benefit_diag = _apply_region_spread_policy(
        benefit_proj,
        year_col="Year",
        region_col="region",
        value_col="Benefit_Year",
        spread_national=spread_national,
        spread_unmapped=spread_unmapped,
    )

    region_list = list(regions)
    if (
        UNMAPPED_REGION_LABEL in benefit_proj["region"].values
        and UNMAPPED_REGION_LABEL not in region_list
        and spread_unmapped
    ):
        region_list.append(UNMAPPED_REGION_LABEL)

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
    return benefit_region, benefit_diag

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
    *,
    spread_national: bool,
    spread_unmapped: bool,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
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
    benefit_proj["region"] = benefit_proj["region"].fillna(UNMAPPED_REGION_LABEL).replace({"": UNMAPPED_REGION_LABEL})
    benefit_proj, benefit_diag = _apply_region_spread_policy(
        benefit_proj,
        year_col="Year",
        region_col="region",
        value_col="Benefit_Year",
        spread_national=spread_national,
        spread_unmapped=spread_unmapped,
    )
    region_list = list(regions)
    if (
        UNMAPPED_REGION_LABEL in benefit_proj["region"].values
        and UNMAPPED_REGION_LABEL not in region_list
        and spread_unmapped
    ):
        region_list.append(UNMAPPED_REGION_LABEL)
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
    return benefit_region, benefit_diag
def compute_region_metrics(
    data: DashboardData,
    scenario_code: str,
    *,
    mapping: Optional[pd.DataFrame] = None,
    spread_national: bool = True,
    spread_unmapped: bool = True,
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

    region_info = _ensure_display_region_info(region_info)

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

    merged["region"] = merged["region"].fillna(UNMAPPED_REGION_LABEL)
    merged["region"] = merged["region"].replace({"": UNMAPPED_REGION_LABEL})
    merged.loc[merged["region"] == UNMAPPED_REGION_LABEL, ["join_key", "join_key_norm"]] = ""
    merged.loc[merged["region"] == UNMAPPED_REGION_LABEL, ["population", "gdp_per_capita"]] = np.nan

    merged, spend_diag = _apply_region_spread_policy(
        merged,
        year_col="Year",
        region_col="region",
        value_col="Spend_M",
        spread_national=spread_national,
        spread_unmapped=spread_unmapped,
    )

    region_spend = merged.groupby(["Year", "region"], as_index=False)["Spend_M"].sum()

    if (
        spread_unmapped
        and (merged["region"] == UNMAPPED_REGION_LABEL).any()
        and UNMAPPED_REGION_LABEL not in region_info["region"].values
    ):
        extra = pd.DataFrame(
            {
                "region": [UNMAPPED_REGION_LABEL],
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
    benefit_diag = pd.DataFrame()
    scenario_meta_lookup = getattr(data, "scenario_meta_by_code", {})
    meta = scenario_meta_lookup.get(scenario_code)
    raw_results = getattr(data, "raw_results", {})
    if meta:
        stem = meta.get("_stem")
        if stem:
            raw_result = raw_results.get(stem)
            if raw_result is not None:
                benefit_bundle = _benefit_region_from_raw_result(
                    raw_result,
                    mapping_df,
                    years_present,
                    all_regions,
                    spread_national=spread_national,
                    spread_unmapped=spread_unmapped,
                )
                if benefit_bundle is not None:
                    benefit_frame, benefit_diag = benefit_bundle
    if benefit_frame is None:
        benefit_bundle = _compute_region_benefit_metrics(
            data,
            scenario_code,
            mapping_df,
            years_present,
            all_regions,
            spread_national=spread_national,
            spread_unmapped=spread_unmapped,
        )
        if benefit_bundle is not None:
            benefit_frame, benefit_diag = benefit_bundle
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

    if spend_diag is None or spend_diag.empty:
        spend_diag = pd.DataFrame({"Year": years_present})
        spend_diag["full_total"] = 0.0
        spend_diag["source_national"] = 0.0
        spend_diag["source_unmapped"] = 0.0
        spend_diag["included_total"] = 0.0
        spend_diag["excluded_national"] = 0.0
        spend_diag["excluded_unmapped"] = 0.0
        spend_diag["excluded_total"] = 0.0
    else:
        spend_diag = spend_diag.rename(columns={"Year": "Year"})
    spend_diag = spend_diag.sort_values("Year").copy()
    spend_diag["full_total_cum"] = spend_diag["full_total"].cumsum()
    spend_diag["included_total_cum"] = spend_diag["included_total"].cumsum()
    spend_diag["excluded_total_cum"] = spend_diag["excluded_total"].cumsum()
    spend_diag_idx = spend_diag.set_index("Year")

    if benefit_diag is None or benefit_diag.empty:
        benefit_diag = pd.DataFrame({"Year": years_present})
        benefit_diag["full_total"] = 0.0
        benefit_diag["source_national"] = 0.0
        benefit_diag["source_unmapped"] = 0.0
        benefit_diag["included_total"] = 0.0
        benefit_diag["excluded_national"] = 0.0
        benefit_diag["excluded_unmapped"] = 0.0
        benefit_diag["excluded_total"] = 0.0
    else:
        benefit_diag = benefit_diag.rename(columns={"Year": "Year"})
    benefit_diag = benefit_diag.sort_values("Year").copy()
    benefit_diag["full_total_cum"] = benefit_diag["full_total"].cumsum()
    benefit_diag["included_total_cum"] = benefit_diag["included_total"].cumsum()
    benefit_diag["excluded_total_cum"] = benefit_diag["excluded_total"].cumsum()
    benefit_diag_idx = benefit_diag.set_index("Year")

    spend_diag_columns = {
        "Spend_Full_Year": "full_total",
        "Spend_Source_National_Year": "source_national",
        "Spend_Source_Unmapped_Year": "source_unmapped",
        "Spend_Included_Year": "included_total",
        "Spend_Excluded_National_Year": "excluded_national",
        "Spend_Excluded_Unmapped_Year": "excluded_unmapped",
        "Spend_Excluded_Year": "excluded_total",
        "Spend_Full_Cum": "full_total_cum",
        "Spend_Included_Cum": "included_total_cum",
        "Spend_Excluded_Cum": "excluded_total_cum",
    }
    for output_col, source_col in spend_diag_columns.items():
        region_spend[output_col] = region_spend["Year"].map(spend_diag_idx[source_col]).fillna(0.0)

    benefit_diag_columns = {
        "Benefit_Full_Year": "full_total",
        "Benefit_Source_National_Year": "source_national",
        "Benefit_Source_Unmapped_Year": "source_unmapped",
        "Benefit_Included_Year": "included_total",
        "Benefit_Excluded_National_Year": "excluded_national",
        "Benefit_Excluded_Unmapped_Year": "excluded_unmapped",
        "Benefit_Excluded_Year": "excluded_total",
        "Benefit_Full_Cum": "full_total_cum",
        "Benefit_Included_Cum": "included_total_cum",
        "Benefit_Excluded_Cum": "excluded_total_cum",
    }
    for output_col, source_col in benefit_diag_columns.items():
        region_spend[output_col] = region_spend["Year"].map(benefit_diag_idx[source_col]).fillna(0.0)

    region_spend["Spread_National_Enabled"] = bool(spread_national)
    region_spend["Spread_Unmapped_Enabled"] = bool(spread_unmapped)

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
            "Spend_Full_Year",
            "Spend_Source_National_Year",
            "Spend_Source_Unmapped_Year",
            "Spend_Included_Year",
            "Spend_Excluded_National_Year",
            "Spend_Excluded_Unmapped_Year",
            "Spend_Excluded_Year",
            "Spend_Full_Cum",
            "Spend_Included_Cum",
            "Spend_Excluded_Cum",
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
            "Benefit_Full_Year",
            "Benefit_Source_National_Year",
            "Benefit_Source_Unmapped_Year",
            "Benefit_Included_Year",
            "Benefit_Excluded_National_Year",
            "Benefit_Excluded_Unmapped_Year",
            "Benefit_Excluded_Year",
            "Benefit_Full_Cum",
            "Benefit_Included_Cum",
            "Benefit_Excluded_Cum",
            "BenefitShare_Year",
            "BenefitShare_Cum",
            "Spread_National_Enabled",
            "Spread_Unmapped_Enabled",
            "population",
            "gdp_per_capita",
        ]
    ]
