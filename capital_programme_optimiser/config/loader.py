from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
import pandas as pd
import re


@dataclass
class ForcedStartRule:
    start: Optional[int] = None
    include: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForcedStartRule":
        if data is None:
            return cls()
        return cls(start=data.get("start"), include=data.get("include"))


@dataclass
class DataConfig:
    scoring_workbook: str
    costs_sheet: str
    benefit_scenarios: Dict[str, str]


@dataclass
class OptimisationConfig:
    cost_types: List[str]
    start_fy: int
    years: int
    max_starts: int
    benefit_discount_rate: float
    run_benefit_plusminus: bool
    plusminus_levels_m: List[float]
    surplus_options_m: Dict[str, float]
    early_stop_pct: float
    early_stop: bool
    greedy_warm_start: bool
    solve_seconds: int
    verbose: int
    cash_pv_rate: float


@dataclass
class PathsConfig:
    cache_dir: Path
    dashboard_output_dir: Path
    preset_dir: Path
    saved_dir: Path
    default_preset: str = ""


@dataclass
class UIConfig:
    default_run_mode: str = "standard"
    baseline_options_m: List[int] = field(default_factory=list)
    buffer_levels_m: List[int] = field(default_factory=list)


@dataclass
class Settings:
    root: Path
    data: DataConfig
    optimisation: OptimisationConfig
    paths: PathsConfig
    ui: UIConfig = field(default_factory=UIConfig)
    forced_start: Dict[str, ForcedStartRule] = field(default_factory=dict)

    def cache_dir(self) -> Path:
        return (self.root / self.paths.cache_dir).resolve()

    def dashboard_output_dir(self) -> Path:
        return (self.root / self.paths.dashboard_output_dir).resolve()

    def preset_dir(self) -> Path:
        return (self.root / self.paths.preset_dir).resolve()

    def saved_dir(self) -> Path:
        return (self.root / self.paths.saved_dir).resolve()

    def scoring_workbook(self) -> Path:
        return (self.root / self.data.scoring_workbook).resolve()


DEFAULT_SETTINGS_PATH = Path(__file__).with_name("settings.yaml")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_settings(path: Optional[Path] = None) -> Settings:
    cfg_path = path or DEFAULT_SETTINGS_PATH
    raw = _load_yaml(cfg_path)

    data_cfg = DataConfig(**raw["data"])
    opt_raw = raw["optimisation"]
    opt_cfg = OptimisationConfig(
        cost_types=list(opt_raw.get("cost_types", [])),
        start_fy=int(opt_raw["start_fy"]),
        years=int(opt_raw["years"]),
        max_starts=int(opt_raw["max_starts"]),
        benefit_discount_rate=float(opt_raw["benefit_discount_rate"]),
        run_benefit_plusminus=bool(opt_raw.get("run_benefit_plusminus", True)),
        plusminus_levels_m=[float(v) for v in opt_raw.get("plusminus_levels_m", [])],
        surplus_options_m={k: float(v) for k, v in opt_raw.get("surplus_options_m", {}).items()},
        early_stop_pct=float(opt_raw.get("early_stop_pct", 0.5)),
        early_stop=bool(opt_raw.get("early_stop", True)),
        greedy_warm_start=bool(opt_raw.get("greedy_warm_start", True)),
        solve_seconds=int(opt_raw.get("solve_seconds", 1200)),
        verbose=int(opt_raw.get("verbose", 1)),
        cash_pv_rate=float(opt_raw.get("cash_pv_rate", 0.015)),
    )

    paths_raw = raw.get("paths", {}) or {}
    cache_dir = Path(paths_raw.get("cache_dir", "scenario_cache"))
    dashboard_dir = Path(paths_raw.get("dashboard_output_dir", "."))
    preset_dir = Path(paths_raw.get("preset_dir", "scenario_sets/presets"))
    saved_dir = Path(paths_raw.get("saved_dir", "scenario_sets/saved_runs"))
    default_preset = str(paths_raw.get("default_preset", ""))

    paths_cfg = PathsConfig(
        cache_dir=cache_dir,
        dashboard_output_dir=dashboard_dir,
        preset_dir=preset_dir,
        saved_dir=saved_dir,
        default_preset=default_preset,
    )

    ui_raw = raw.get("ui", {}) or {}
    baseline_opts = [int(v) for v in ui_raw.get("baseline_options_m", [])]
    buffer_levels = [int(v) for v in ui_raw.get("buffer_levels_m", [])]
    default_mode = str(ui_raw.get("default_run_mode", "standard"))
    try:
        ui_cfg = UIConfig(
            default_run_mode=default_mode,
            baseline_options_m=baseline_opts,
            buffer_levels_m=buffer_levels,
        )
    except TypeError:
        ui_cfg = UIConfig(default_run_mode=default_mode)
        setattr(ui_cfg, "baseline_options_m", baseline_opts)
        setattr(ui_cfg, "buffer_levels_m", buffer_levels)

    forced = {
        name: ForcedStartRule.from_dict(rule if isinstance(rule, dict) else {})
        for name, rule in raw.get("forced_start", {}).items()
    }

    return Settings(
        root=Path(raw["root"]).expanduser(),
        data=data_cfg,
        optimisation=opt_cfg,
        paths=paths_cfg,
        ui=ui_cfg,
        forced_start=forced,
    )




def _normalise_text(value: Any) -> str:
    if value is None:
        return ''
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value.lower()


def _find_column(columns: Iterable[Any], *candidates: str) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def load_project_region_mapping(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the project-to-region mapping table from the packaged pickle file.

    The return frame has normalised column names and helper columns for joins.
    """
    file_path = path if path is not None else Path(__file__).with_name('project_region_mapping.pkl')
    if not file_path.exists():
        raise FileNotFoundError(f"Project-region mapping not found: {file_path}")

    df = pd.read_pickle(file_path)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    col_project = _find_column(df.columns, 'project')
    col_region = _find_column(df.columns, 'region')
    col_join = _find_column(df.columns, 'join key', 'join_key', 'joinkey')
    col_gdp = _find_column(df.columns, 'gdp per capita', 'gdp_per_capita', 'gdp_pc')
    col_pop = _find_column(df.columns, 'population', 'pop')

    required = {
        'project': col_project,
        'region': col_region,
        'join_key': col_join,
        'gdp_per_capita': col_gdp,
        'population': col_pop,
    }
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(f"Missing expected columns in project-region mapping: {missing}")

    df_proc = df[[required['project'], required['region'], required['join_key'], required['gdp_per_capita'], required['population']]].copy()
    df_proc.columns = ['project', 'region', 'join_key', 'gdp_per_capita', 'population']

    df_proc['project'] = df_proc['project'].astype(str).str.strip()
    df_proc['region'] = df_proc['region'].astype(str).str.strip()
    df_proc['join_key'] = df_proc['join_key'].astype(str).str.strip()

    df_proc['project_norm'] = df_proc['project'].map(_normalise_text)
    df_proc['join_key_norm'] = df_proc['join_key'].map(_normalise_text)

    df_proc['gdp_per_capita'] = pd.to_numeric(df_proc['gdp_per_capita'], errors='coerce').fillna(0.0)
    df_proc['population'] = pd.to_numeric(df_proc['population'], errors='coerce').fillna(0.0)

    return df_proc
